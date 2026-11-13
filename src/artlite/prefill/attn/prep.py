"""``PrepOperator``: the attention's prep (Q, K, V of a chunk into the attention's operands and the cache's records;
module docstring of ``prefill.attn``)."""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ...operators import templating
from ...operators.base import ArgSpec, ArtcOperator
from ...operators.chain import ChainPart, chained_fill
from ...operators.design import COLS, MAX_BD_ITER
from ...operators.kernel_source import kernel_prefix
from ...operators.util import Attr, bf16_hilo, regions
from ..gemm import Layout
from .shape import P_CS, P_KV, Shape

_PREP = templating.load("artlite/prefill/attn/prep.cc.jinja")


def _cores(n_tg: int, n_grp: int, rg: int) -> tuple:
    """The prm / K / V core bodies both designs share: (the norm's weights from the B stream's first object, K's group
    objects, V's) over ``n_grp`` cache groups of ``rg`` records."""

    def prm(b_h, fn, ws):  # (norm: the layer's weights, the B stream's first object)
        if fn is not None:
            w = b_h.acquire(1)
            fn(w, ws)
            b_h.release(1)

    def core_k(a_h, b_h, o_h, fn, ws, k_prm):
        for _ in forever_():
            prm(b_h, k_prm, ws)
            for _ in range_(n_grp):
                out = o_h.acquire(1)
                for r in range_(rg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    fn(x, cs, ws, out, r)
                    b_h.release(1)
                    a_h.release(1)
                o_h.release(1)

    def core_v(a_h, o_h, fn):
        for _ in forever_():
            for _ in range_(n_grp):
                out = o_h.acquire(1)
                for r in range_(rg):
                    x = a_h.acquire(1)
                    fn(x, out, r)
                    a_h.release(1)
                o_h.release(1)

    return prm, core_k, core_v


class PrepOperator(ArtcOperator):
    """The attention prep of T tokens (module docstring). Runtime sequence: G (the qkv GEMM's output, ``layout``), CS
    (``cs_table`` of every position, as bf16 pairs), QS (per token group and column the heads' Q units; a sub-chunk's
    part of a chunk's), KV (the layer's cache: COLS KV heads of ``kv_bytes`` each, ``kv_pad`` bytes past them); with
    ``norm`` (Qwen3: q and k RMS-normed per head with weights per dim before RoPE, ``eps``) NW, the layer's [q_norm |
    k_norm] fp32 (``nw_bytes``: the B stream's first object, before the cos / sin rows). Parameters ``params``."""

    STACK = 1024
    pfx = kernel_prefix("pap")

    def __init__(self, *, T: int, shape: Shape, layout: Layout, kv_bytes: int, cs_positions: int, kv_pad: int = 0,
                 norm: bool = False, eps: float = 1e-6, kv0: int = 0):
        """``kv0``: the bytes of a KV head's cache before its group 0 (a sliding-window cache's guard)."""
        s = shape
        assert T % s.pg == 0 and T // s.pg <= 64 and {"q", "k", "v"} <= set(layout.off)
        assert s.qh in (2, 3, 4) and s.used % 1152 == 0 and s.per in (1, 2)
        assert s.per == 1 or (s.qh == 2 and s.hd % (2 * 64) == 0), "two columns per KV head: a column's two heads"
        assert not s.gate or (s.per == 2 and "g" in layout.off), "the gates: the cores two columns of a KV head free"
        self.T, self.shape, self.layout, self.kv_bytes, self.cs_positions = T, s, layout, kv_bytes, cs_positions
        self.kv_pad, self.norm, self.eps, self.kv0 = kv_pad, norm, eps, kv0
        self.nq = [-(-s.qh // 2), s.qh // 2]  # (the Q cores' heads)
        super().__init__()

    @staticmethod
    def nw_bytes(hd: int) -> int:
        """Bytes of a layer's NW object (the B stream's objects: 8 positions' cos / sin; its weights 8 hd)."""
        return 8 * 2 * hd * 2

    def workspace(self) -> dict[str, int]:
        hd = self.shape.hd
        return regions([("row", 4 * hd), ("stage", 32 * hd)] + ([("nrm", 8 * hd)] if self.norm else []))

    def get_kernel_src(self) -> str:
        p = self.pfx
        s = self.shape
        return _PREP.render(p=p, HD=s.hd, ND=s.nd, KP=s.kp, QP=s.qp, KREC=s.rec.k_bytes, QUNIT=s.qunit,
                            qc=bf16_hilo(s.scale), w=Attr(self.workspace()), NORM=self.norm, eps=repr(self.eps),
                            GATE=s.gate, QH=s.qh)

    @property
    def n_cs(self) -> int:
        return self.cs_positions * 2 * self.shape.hd

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument: [column][token group][head][Q unit] (with gates: each token group's then)."""
        return self.T // 8 * COLS * self.shape.tgu // 2

    @property
    def n_kv(self) -> int:
        return (self.shape.kvh * self.kv_bytes + self.kv_pad) // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        nw = [ArgSpec("in", (self.nw_bytes(self.shape.hd) // 2,), bfloat16)] if self.norm else []
        return [ArgSpec("in", (self.T * self.layout.N,), bfloat16), ArgSpec("in", (self.n_cs,), bfloat16),
                ArgSpec("out", (self.n_qs,), bfloat16), ArgSpec("out", (self.n_kv,), bfloat16), *nw]

    def params(self, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 (a group start)."""
        s = self.shape
        assert p0 % s.pg == 0 and p0 + self.T <= self.cs_positions
        return {P_CS: 2 * s.hd * p0, P_KV: s.group // 2 * (p0 // s.pg)}

    def get_mlir_artifact(self) -> Module:
        return self._pair_design() if self.shape.per == 2 else self._column_design()

    def _column_design(self) -> Module:
        """One column per KV head (``Shape.per`` 1): Q cores (c, 2), (c, 3), K (c, 4), V (c, 5)."""
        p, T, lay, s = self.pfx, self.T, self.layout, self.shape
        hd, N, pitch, nq = s.hd, lay.N, lay.pitch, self.nq
        bf = np.dtype[bfloat16]
        q_ty = [np.ndarray[(8 * n * hd,), bf] for n in nq]  # 8 rows of a Q core's heads
        row_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of K or V
        cs_ty = np.ndarray[(8 * 2 * hd,), bf]  # 8 positions x [cos | sin] fp32
        qo_ty = [np.ndarray[(n * s.qunit // 2,), bf] for n in nq]
        kg_ty = np.ndarray[(s.kp * s.used // 2,), bf]  # a group's K objects (their records)
        vg_ty = np.ndarray[(s.used // 2,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_q = Kernel(f"{p}_q0", "kernel.o", [q_ty[0], cs_ty, ws_ty, qo_ty[0], np.int32, np.int32])
        k_q1 = Kernel(f"{p}_q1", "kernel.o", [q_ty[1], cs_ty, ws_ty, qo_ty[1], np.int32, np.int32])
        k_k = Kernel(f"{p}_k", "kernel.o", [row_ty, cs_ty, ws_ty, kg_ty, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [row_ty, vg_ty, np.int32])
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty]) if self.norm else None
        p_cs, p_kv = ScratchpadParameter(P_CS, np.int32), ScratchpadParameter(P_KV, np.int32)
        n_tg, rg, n_grp = T // 8, s.rg, T // s.pg

        # per column and token group [Q core 0's rows | Q core 1's | K rows | V rows], split in the memtile
        sizes = [8 * nq[0] * hd, 8 * nq[1] * hd, 8 * hd, 8 * hd]
        offs = [sum(sizes[:i]) for i in range(4)]
        a3 = [ObjectFifo(np.ndarray[(sum(sizes),), bf], name=f"lp_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split(offs, tile=Tile(c, 1), depths=[2] * 4, obj_types=[*q_ty, row_ty, row_ty],
                                 names=[f"lp_a{c}_{i}" for i in range(4)]) for c in range(COLS)]
        b = [ObjectFifo(cs_ty, name=f"lp_b{c}", depth=2) for c in range(COLS)]
        # the Q cores' units joined per token group; the K and V cores' group objects joined per group
        o3 = [ObjectFifo(np.ndarray[(s.qh * s.qunit // 2,), bf], name=f"lp_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([0, nq[0] * s.qunit // 2], tile=Tile(c, 1), depths=[2, 2], obj_types=qo_ty,
                                names=[f"lp_o{c}_{i}" for i in range(2)]) for c in range(COLS)]
        g3 = [ObjectFifo(np.ndarray[(s.objects * s.used // 2,), bf], name=f"lp_g{c}", depth=2) for c in range(COLS)]
        g1 = [g3[c].prod().join([0, s.kp * s.used // 2], tile=Tile(c, 1), depths=[2, 2], obj_types=[kg_ty, vg_ty],
                                names=[f"lp_k{c}", f"lp_v{c}"]) for c in range(COLS)]

        def core_q(a_h, b_h, o_h, fn, ws, n, k_prm):
            for _ in forever_():
                prm(b_h, k_prm, ws)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    for h in range(n):
                        fn(x, cs, ws, out, h, n)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        prm, core_k, core_v = _cores(n_tg, n_grp, rg)
        workers = []
        for c in range(COLS):
            for i, fn in enumerate((k_q, k_q1)):
                workers.append(Worker(core_q, [a1[c][i].cons(), b[c].cons(), o1[c][i].prod(), fn,
                                               Buffer(ws_ty, name=f"lp_ws{c}_{i}"), nq[i], k_prm],
                                      tile=Tile(c, 2 + i), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            workers.append(Worker(core_k, [a1[c][2].cons(), b[c].cons(), g1[c][0].prod(), k_k,
                                           Buffer(ws_ty, name=f"lp_ws{c}_k"), k_prm],
                                  tile=Tile(c, 4), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                  while_true=False))
            workers.append(Worker(core_v, [a1[c][3].cons(), g1[c][1].prod(), k_v], tile=Tile(c, 5),
                                  stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv, n_qs = self.n_cs, self.n_kv, self.n_qs
        uw, qw = s.qunit // 2, s.qh * s.qunit // 2  # (bf16: a Q unit, a column's units of a token group)
        hb, g2, o2, ow = self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2

        n_nw = self.nw_bytes(hd) // 2

        def seq(G, CS, QS, KV, *rest):
            *nw, ah, bh, oh, gh = rest
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                for t0 in range(0, n_tg, MAX_BD_ITER):  # (a chained fill repeats at most MAX_BD_ITER times: that many token groups)
                    q0, k0 = c * pitch + lay.off["q"] + 8 * t0 * N, c * pitch + lay.off["k"] + 8 * t0 * N
                    nt = min(MAX_BD_ITER, n_tg - t0)
                    chained_fill(ah[c], G, [
                        TensorAccessPattern((T * N,), q0, [nt, 1, 8, nq[0] * hd], [8 * N, 0, N, 1]),
                        TensorAccessPattern((T * N,), q0 + nq[0] * hd, [nt, 1, 8, nq[1] * hd], [8 * N, 0, N, 1]),
                        TensorAccessPattern((T * N,), k0, [nt, 2, 8, hd], [8 * N, lay.off["v"] - lay.off["k"], N, 1])],
                        tg)
                cs = TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 2 * hd], [0, 0, 0, 1])
                if nw:
                    chained_fill(bh[c], CS, [ChainPart(nw[0], TensorAccessPattern((n_nw,), 0, [1, 1, 1, n_nw], [0, 0, 0, 1])),
                                             ChainPart(CS, cs, offset_parameter=p_cs)], tg)
                else:
                    bh[c].fill(CS, cs, offset_parameter=p_cs, group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * n_tg * qw, [1, n_tg, s.qh, uw], [0, qw, uw, 1]),
                            group=tg, wait=True)
                # per group its objects' records (rows of 576 elements: a shim BD's inner dimension <= 1023 words)
                gh[c].drain(KV, TensorAccessPattern((n_kv,), c * hb, [n_grp, s.objects, ow // 576, 576],
                                                    [g2, o2, 576, 1]), offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(n_qs,), bf],
                           np.ndarray[(n_kv,), bf], *([np.ndarray[(n_nw,), bf]] if self.norm else []),
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(g3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


    def _pair_design(self) -> Module:
        """Two columns per KV head (``Shape.per``): column c = 2 h + j takes its two heads' Q (cores (c, 2), (c, 3)); KV head
        h's K records from both columns' K halves on core (2 h, 4), its V records on (2 h + 1, 5); each column's gates
        (``Shape.gate``: sigmoid of its heads' logits, after its Q units in QS) on its free core ((2 h, 5), (2 h + 1, 4))."""
        p, T, lay, s = self.pfx, self.T, self.layout, self.shape
        hd, N, pitch, qh = s.hd, lay.N, lay.pitch, s.qh
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of a Q core's head
        row_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of K or V (both halves)
        gi_ty = np.ndarray[(8 * qh,), bf]  # 8 rows of the column's gate logits
        cs_ty = np.ndarray[(8 * 2 * hd,), bf]
        qo_ty = np.ndarray[(s.qunit // 2,), bf]
        go_ty = np.ndarray[(s.gw // 2,), bf]
        kg_ty = np.ndarray[(s.kp * s.used // 2,), bf]
        vg_ty = np.ndarray[(s.used // 2,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_q = [Kernel(f"{p}_q{i}", "kernel.o", [q_ty, cs_ty, ws_ty, qo_ty, np.int32, np.int32]) for i in range(2)]
        k_k = Kernel(f"{p}_k", "kernel.o", [row_ty, cs_ty, ws_ty, kg_ty, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [row_ty, vg_ty, np.int32])
        k_g = Kernel(f"{p}_g", "kernel.o", [gi_ty, go_ty]) if s.gate else None
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty]) if self.norm else None
        p_cs, p_kv = ScratchpadParameter(P_CS, np.int32), ScratchpadParameter(P_KV, np.int32)
        n_tg, rg, n_grp = T // 8, s.rg, T // s.pg
        gate = s.gate

        def parts(j: int) -> list[tuple[object, int]]:  # (per column the memtile object's pieces: rows 2 .. 5)
            kv = [(row_ty, 8 * hd)]
            g = [(gi_ty, 8 * qh)] if gate else []
            return [(q_ty, 8 * hd), (q_ty, 8 * hd), *(kv + g if j == 0 else g + kv)]

        a1, o1, kvo = {}, {}, {}
        for c in range(COLS):
            j = c % 2
            pc = parts(j)
            offs = [sum(n for _, n in pc[:i]) for i in range(len(pc))]
            a3 = ObjectFifo(np.ndarray[(sum(n for _, n in pc),), bf], name=f"lp_a{c}", depth=2)
            a1[c] = (a3, a3.cons().split(offs, tile=Tile(c, 1), depths=[2] * len(pc), obj_types=[t for t, _ in pc],
                                         names=[f"lp_a{c}_{i}" for i in range(len(pc))]))
            o3 = ObjectFifo(np.ndarray[(s.tgu // 2,), bf], name=f"lp_o{c}", depth=2)
            jo = [0, s.qunit // 2] + ([qh * s.qunit // 2] if gate else [])
            o1[c] = (o3, o3.prod().join(jo, tile=Tile(c, 1), depths=[2] * len(jo),
                                        obj_types=[qo_ty, qo_ty] + ([go_ty] if gate else []),
                                        names=[f"lp_o{c}_{i}" for i in range(len(jo))]))
            kvo[c] = ObjectFifo(kg_ty if j == 0 else vg_ty, name=f"lp_g{c}", depth=2)
        b = [ObjectFifo(cs_ty, name=f"lp_b{c}", depth=2) for c in range(COLS)]

        def core_q(a_h, b_h, o_h, fn, ws, k_prm):
            for _ in forever_():
                prm(b_h, k_prm, ws)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, cs, ws, out, 0, 1)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        prm, core_k, core_v = _cores(n_tg, n_grp, rg)

        def core_g(a_h, o_h, fn):
            for _ in forever_():
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, out)
                    o_h.release(1)
                    a_h.release(1)

        workers = []
        for c in range(COLS):
            j, sp, jn = c % 2, a1[c][1], o1[c][1]
            for i in range(2):
                workers.append(Worker(core_q, [sp[i].cons(), b[c].cons(), jn[i].prod(), k_q[i],
                                               Buffer(ws_ty, name=f"lp_ws{c}_{i}"), k_prm],
                                      tile=Tile(c, 2 + i), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            if j == 0:
                workers.append(Worker(core_k, [sp[2].cons(), b[c].cons(), kvo[c].prod(), k_k,
                                               Buffer(ws_ty, name=f"lp_ws{c}_k"), k_prm],
                                      tile=Tile(c, 4), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            else:
                workers.append(Worker(core_v, [sp[3 if gate else 2].cons(), kvo[c].prod(), k_v], tile=Tile(c, 5),
                                      stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
            if gate:
                workers.append(Worker(core_g, [sp[3 if j == 0 else 2].cons(), jn[2].prod(), k_g],
                                      tile=Tile(c, 5 if j == 0 else 4), stack_size=2 * self.STACK,  # (kmath's sigmoid)
                                      dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv, n_qs, n_nw = self.n_cs, self.n_kv, self.n_qs, self.nw_bytes(hd) // 2
        tw = s.tgu // 2  # (bf16 of a column's token group in QS)
        hb, g2, o2, ow, k0b = self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2, self.kv0 // 2

        def seq(G, CS, QS, KV, *rest):
            *nw, ah, bh, oh, gh = rest
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                h, j = divmod(c, 2)
                for t0 in range(0, n_tg, MAX_BD_ITER):  # (a chained fill repeats at most MAX_BD_ITER times: that many token groups)
                    nt = min(MAX_BD_ITER, n_tg - t0)
                    q0 = c * pitch + lay.off["q"] + 8 * t0 * N
                    kv = TensorAccessPattern((T * N,), 2 * h * pitch + lay.off["k" if j == 0 else "v"] + 8 * t0 * N,
                                             [nt, 8, 2, hd // 2], [8 * N, N, pitch, 1])  # (both columns' halves)
                    pieces = [TensorAccessPattern((T * N,), q0, [nt, 1, 8, hd], [8 * N, 0, N, 1]),
                              TensorAccessPattern((T * N,), q0 + hd, [nt, 1, 8, hd], [8 * N, 0, N, 1])]
                    g = [TensorAccessPattern((T * N,), c * pitch + lay.off["g"] + 8 * t0 * N, [nt, 1, 8, qh],
                                             [8 * N, 0, N, 1])] if gate else []
                    chained_fill(ah[c], G, pieces + ([kv] + g if j == 0 else g + [kv]), tg)
                cs = TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 2 * hd], [0, 0, 0, 1])
                if nw:
                    chained_fill(bh[c], CS, [ChainPart(nw[0], TensorAccessPattern((n_nw,), 0, [1, 1, 1, n_nw], [0, 0, 0, 1])),
                                             ChainPart(CS, cs, offset_parameter=p_cs)], tg)
                else:
                    bh[c].fill(CS, cs, offset_parameter=p_cs, group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * n_tg * tw, [1, 1, n_tg, tw], [0, 0, tw, 1]),
                            group=tg, wait=True)
                if j == 0:  # (the KV head's K objects of each group; its V object: column 2 h + 1's)
                    pat = TensorAccessPattern((n_kv,), h * hb + k0b, [n_grp, s.kp, ow // 576, 576], [g2, o2, 576, 1])
                else:
                    pat = TensorAccessPattern((n_kv,), h * hb + k0b + s.kp * o2, [1, n_grp, ow // 576, 576],
                                              [0, g2, 576, 1])
                gh[c].drain(KV, pat, offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(n_qs,), bf],
                           np.ndarray[(n_kv,), bf], *([np.ndarray[(n_nw,), bf]] if self.norm else []),
                           [a1[c][0].prod(tile=Tile(c, 0)) for c in range(COLS)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [o1[c][0].cons(tile=Tile(c, 0)) for c in range(COLS)],
                           [kvo[c].cons(tile=Tile(c, 0)) for c in range(COLS)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()

