"""``ResidualNormOperator`` (Gemma's and LFM3's fp32 residual stream): the residual update and norm of a chunk (module
docstring of ``prefill.norm``)."""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ...operators import templating
from ...operators.base import ArgSpec, ArtcOperator
from ...operators.design import COLS, stream_walk
from ...operators.kernel_source import kernel_prefix
from ...operators.util import Attr, bf16_value, regions

MODES = ("in", "post", "plain")
RCH = 16  # chunks per row (ResidualNormOperator: E / 16 columns each, 160 for E = 2560, 128 for 2048)

_RSRC = templating.load("artlite/prefill/norm/residual.cc.jinja")


def residual_params(w_post: np.ndarray | None, w1: np.ndarray | None, scale: float, E: int) -> np.ndarray:
    """A norm's parameter object ([8 rows][E] bf16 as uint16: row 0 w_post, row 1 w1, row 2 [s hi, s lo])."""
    out = np.zeros((8, E), np.uint16)

    def b16(v: np.ndarray) -> np.ndarray:
        return np.asarray(v, np.float32).astype(bfloat16).view(np.uint16)

    if w_post is not None:
        out[0] = b16(w_post)
    if w1 is not None:
        out[1] = b16(w1)
    hi = bf16_value(scale)
    out[2, :2] = b16(np.array([hi, scale - hi]))
    return out



class ResidualNormOperator(ArtcOperator):
    """The residual update and norm of T tokens (module docstring). Runtime sequence: ``in``: (P, X, XN); ``post`` /
    ``plain``: (P, X, D, XO, XN), XO = X in place. P: the parameters (``params``), X / XO: (T, E) fp32 (as bf16
    pairs), D: the deltas (2 T 8 d_pitch per delta, ``d_stride`` apart), XN: panels."""

    STACK = 1024
    p = kernel_prefix("gn")

    def __init__(self, *, T: int, E: int, mode: str, eps: float, d_pitch: int = 0, deltas: int = 1, d_stride: int = 0,
                 scale: bool = False, cols: int = COLS):
        """``scale``: x' = (...) s with s from the parameters (the layer's output scale), else s = 1 (cheaper)."""
        assert mode in MODES and E % 128 == 0, (mode, E)
        assert T % 8 == 0 and T >= 8 * cols and 1 <= cols <= COLS and deltas in (1, 2, 4)
        assert mode == "post" or not scale
        self.T, self.E, self.mode, self.eps, self.cols, self.scale = T, E, mode, eps, cols, scale
        self.ch = E // RCH  # (columns per chunk)
        self.d_pitch = d_pitch or E // COLS
        assert self.d_pitch >= E // COLS
        self.deltas = deltas if mode != "in" else 0
        self.d_stride = d_stride or T * COLS * self.d_pitch
        super().__init__()

    def workspace(self) -> dict[str, int]:
        E = self.E
        sizes = [("wp", 2 * E), ("w1", 2 * E), ("ss", 256), ("ah", 128), ("al", 128), ("ih", 128), ("il", 128),
                 ("sh", 128), ("sl", 128)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        return _RSRC.render(p=self.p, E=self.E, CH=self.ch, eps=repr(self.eps), mode=self.mode, deltas=self.deltas,
                           scale=self.scale, w=Attr(self.workspace()))

    @property
    def n_delta(self) -> int:
        return (self.deltas - 1) * self.d_stride + self.T * COLS * self.d_pitch

    def get_arg_spec(self) -> list[ArgSpec]:
        T, E = self.T, self.E
        specs = [ArgSpec("in", (8 * E,), bfloat16), ArgSpec("in", (2 * T * E,), bfloat16)]
        if self.mode != "in":
            specs += [ArgSpec("in", (self.n_delta,), bfloat16), ArgSpec("out", (2 * T * E,), bfloat16)]
        specs.append(ArgSpec("out", (T * E * 9 // 16,), bfloat16))
        return specs

    def get_mlir_artifact(self) -> Module:
        T, E, p, mode, nd = self.T, self.E, self.p, self.mode, max(self.deltas, 1)
        ch, n_kc = self.ch, RCH
        n_gs = [len(range(c, T // 8, self.cols)) for c in range(self.cols)]  # (core c: token groups c, c + cols, ..)
        bf = np.dtype[bfloat16]
        xc_ty = np.ndarray[(2 * 8 * ch,), bf]  # a chunk of x (8 tokens x ch fp32)
        dc_ty = np.ndarray[(nd * 8 * ch,), bf]  # a chunk of d ([delta][8 tokens][ch] bf16) or of the parameters
        pan_ty = np.ndarray[(ch // 8 * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 2,), bf]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [dc_ty, ws_ty, np.int32])
        k_ss = Kernel(f"{p}_ss", "kernel.o", [xc_ty if mode == "in" else dc_ty, ws_ty, np.int32])
        k_upd = Kernel(f"{p}_upd", "kernel.o", [xc_ty, dc_ty, ws_ty, xc_ty] + ([pan_ty] if mode == "plain" else [])
                       + [np.int32]) if mode != "in" else None
        k_pan = Kernel(f"{p}_pan", "kernel.o", [xc_ty] + ([dc_ty] if mode == "post" else []) + [ws_ty, pan_ty, np.int32]) \
            if mode != "plain" else None
        # replays per group: (x, d)
        x_rep, d_rep = {"in": (2, 1), "post": (2, 3), "plain": (1, 2)}[mode]

        def replayed(name: str, c: int, n: int, k: int, rep: int) -> tuple[ObjectFifo, ObjectFifo]:
            """A group object of k blocks of [8 tokens][n] through the memtile, sent ``rep`` times as chunks
            [kc][block k][8 tokens][n / n_kc]."""
            ch = n // n_kc
            l2 = ObjectFifo(np.ndarray[(k * 8 * n,), bf], name=f"{name}{c}", depth=2)
            walk = stream_walk(k * 8 * n, [(n_kc, ch), (k, 8 * n), (8, n), (ch, 1)])
            return l2, l2.cons().forward(obj_type=np.ndarray[(k * 8 * ch,), bf], name=f"{name}1{c}", depth=2,
                                          tile=Tile(c, 1), to_stream=walk, repeat_count=rep if rep > 1 else None)

        xf = [replayed("gnx", c, 2 * E, 1, x_rep) for c in range(self.cols)]
        df = [replayed("gnd", c, E, nd, d_rep) for c in range(self.cols)]
        of = [ObjectFifo(xc_ty, name=f"gno{c}", depth=2) for c in range(self.cols)] if mode != "in" else []
        pf = [ObjectFifo(pan_ty, name=f"gnp{c}", depth=2) for c in range(self.cols)]

        def core(x_f, d_f, p_f, prm, ss, ws, *rest):
            rest = list(rest)
            n_g = rest.pop()
            upd = rest.pop(0) if mode != "in" else None
            pan = rest.pop(0) if mode != "plain" else None
            o_f = rest.pop(0) if mode != "in" else None
            for _ in forever_():
                for rep in range(d_rep):  # (the parameters: d's first object, as often as d's groups are replayed)
                    for kc in range_(n_kc):
                        d = d_f.acquire(1)
                        if rep == 0:
                            prm(d, ws, kc)
                        d_f.release(1)
                for _ in range_(n_g):
                    a_f = x_f if mode == "in" else d_f
                    for kc in range_(n_kc):  # pass A
                        v = a_f.acquire(1)
                        ss(v, ws, kc)
                        a_f.release(1)
                    if mode != "in":
                        for kc in range_(n_kc):  # pass B
                            x = x_f.acquire(1)
                            d = d_f.acquire(1)
                            xo = o_f.acquire(1)
                            if mode == "plain":
                                pn = p_f.acquire(1)
                                upd(x, d, ws, xo, pn, kc)
                                p_f.release(1)
                            else:
                                upd(x, d, ws, xo, kc)
                            o_f.release(1)
                            d_f.release(1)
                            x_f.release(1)
                    if mode != "plain":
                        for kc in range_(n_kc):  # pass C
                            x = x_f.acquire(1)
                            pn = p_f.acquire(1)
                            if mode == "post":
                                d = d_f.acquire(1)
                                pan(x, d, ws, pn, kc)
                                d_f.release(1)
                            else:
                                pan(x, ws, pn, kc)
                            p_f.release(1)
                            x_f.release(1)

        workers = []
        for c in range(self.cols):
            args = [xf[c][1].cons(), df[c][1].cons(), pf[c].prod(), k_prm, k_ss, Buffer(ws_ty, name=f"gn_ws{c}")]
            args += [k for k in (k_upd, k_pan) if k is not None]
            if mode != "in":
                args.append(of[c].prod())
            workers.append(Worker(core, [*args, n_gs[c]], tile=Tile(c, 2), stack_size=self.STACK,
                                  dynamic_objfifo_lowering=True, while_true=False))

        n = 2 * T * E  # (x in bf16 units)
        dp, ed = self.d_pitch, COLS * self.d_pitch

        x_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
        d_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
        o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
        p_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]
        cols = self.cols

        def seq(*a):
            if mode == "in":
                P, X, XN, xh, dh, ph = a
            else:
                P, X, D, XO, XN, xh, dh, ph, oh = a
            tg = TaskGroup()
            for c in range(cols):
                n_g = n_gs[c]
                # the parameters as d's first object (read nd times: an object of nd blocks); rows of <= 512 words
                dh[c].fill(P, TensorAccessPattern((8 * E,), 0, [1, nd, 8 * E // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                # x: the core's groups, 8 rows of E fp32 each
                xh[c].fill(X, TensorAccessPattern((n,), 16 * E * c, [1, n_g, 16 * E // 1024, 1024],
                                                  [0, 16 * cols * E, 1024, 1]), group=tg)
                if mode != "in":
                    # d: [group][delta][8 tokens x 8 column blocks][E / 8] (row stride 8 d_pitch = 8 blocks)
                    dh[c].fill(D, TensorAccessPattern((self.n_delta,), 8 * ed * c, [n_g, nd, 8 * COLS, E // COLS],
                                                      [8 * cols * ed, self.d_stride, dp, 1]), group=tg)
                    oh[c].drain(XO, TensorAccessPattern((n,), 16 * E * c, [n_g, n_kc, 8, 2 * ch],
                                                        [16 * cols * E, 2 * ch, 2 * E, 1]), group=tg, wait=True)
                ph[c].drain(XN, TensorAccessPattern((T * E * 9 // 16,), c * (E // 8) * 36,
                                                    [1, n_g, 8, E // 64 * 36], [0, cols * (E // 8) * 36, E // 64 * 36, 1]),
                            group=tg, wait=True)
            tg.finish()

        types = [np.ndarray[(8 * E,), bf], np.ndarray[(n,), bf]]
        if mode != "in":
            types += [np.ndarray[(self.n_delta,), bf], np.ndarray[(n,), bf]]
        types.append(np.ndarray[(T * E * 9 // 16,), bf])
        rt = Runtime(seq, [*types, x_h, d_h, p_h, *([o_h] if mode != "in" else [])])
        return Program(NPU2(), rt, workers=workers).resolve_program()
