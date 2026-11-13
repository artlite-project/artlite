"""``PrefillNormOperator``'s replay design (E > 2048, or two deltas): the memtile holds a group and sends it twice."""

from typing import TYPE_CHECKING

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ...operators.chain import chained_fill
from ...operators.design import COLS, stream_walk
from .plain import CH

if TYPE_CHECKING:
    from .prefill_norm import PrefillNormOperator


def replay_design(op: "PrefillNormOperator") -> Module:
    """E > 2048: x (and delta) groups through the memtile, sent twice (pass A, pass B); ``pre``: the weight objects
    w1, then w2."""
    T, add, E, p, nd, pre = op.T, op.add, op.E, op.p, op.deltas, op.pre
    P, col = op.out_pitch, op.out_col
    n_g = T // (8 * op.cols)
    n_kc = E // CH
    bf = np.dtype[bfloat16]
    chunk_ty = np.ndarray[(8 * CH,), bf]
    pan_ty = np.ndarray[(CH // 8 * 36,), bf]
    w_ty = np.ndarray[(E,), bf]
    st_ty = np.ndarray[(256 + 64,), np.dtype[np.float32]]
    k_w = Kernel(f"{p}_wc", "kernel.o", [chunk_ty, w_ty, np.int32])
    sfx = {0: "0p" if pre else "0", 1: "", 2: "2"}[nd]
    dch = [np.ndarray[(8 * CH * nd,), bf]] if nd else []  # (two deltas: one object of both chunks)
    w1 = [w_ty] if pre else []  # (pre: w1 before the deltas' place)
    k_a = Kernel(f"{p}_ra{sfx}", "kernel.o", [chunk_ty, *dch, *w1, st_ty, np.int32])
    k_b = Kernel(f"{p}_rb{sfx}", "kernel.o",
                 [chunk_ty, *dch, *w1, w_ty, st_ty, pan_ty, *([chunk_ty] if add else []), np.int32])
    # memtile: a group [8 tokens][E] -> the core as chunks [kc][8 tokens][CH], twice; two deltas: their groups
    # side by side, as chunk pairs [kc][delta][8 tokens][CH] (a core has two input DMA channels, a shim two
    # output channels)
    chunks = stream_walk(8 * E, [(n_kc, CH), (8, E), (CH, 1)])
    pairs = stream_walk(16 * E, [(n_kc, CH), (2, 8 * E), (8, E), (CH, 1)])

    def replayed(name: str, c: int, k: int = 1) -> tuple[ObjectFifo, ObjectFifo]:
        l2 = ObjectFifo(np.ndarray[(8 * E * k,), bf], name=f"{name}{c}", depth=2)
        return l2, l2.cons().forward(obj_type=np.ndarray[(8 * CH * k,), bf], name=f"{name}1{c}", depth=2,
                                      tile=Tile(c, 1), to_stream=chunks if k == 1 else pairs, repeat_count=2)

    xf = [replayed("pnx", c) for c in range(op.cols)]
    df = [replayed("pnd", c, nd) for c in range(op.cols)] if nd else []
    of = [ObjectFifo(chunk_ty, name=f"pn_o{c}", depth=2) for c in range(op.cols)] if add else []
    pf = [ObjectFifo(pan_ty, name=f"pn_p{c}", depth=2) for c in range(op.cols)]

    def core(x_f, p_f, kw, ka, kb, wb, st, *rest):
        d_f, o_f = rest if add else (None, None)
        wp = list(rest) if pre else []  # (pre: w1's buffer)
        for _ in forever_():
            for wt in [*wp, wb]:  # the weight objects (w in every row), each sent twice
                for rep in range(2):
                    for kc in range_(n_kc):
                        w = x_f.acquire(1)
                        if rep == 0:
                            kw(w, wt, kc)
                        x_f.release(1)
            for _ in range_(n_g):
                for kc in range_(n_kc):
                    x = x_f.acquire(1)
                    if add:
                        d = d_f.acquire(1)
                        ka(x, d, st, kc)
                        d_f.release(1)
                    else:
                        ka(x, *wp, st, kc)
                    x_f.release(1)
                for kc in range_(n_kc):
                    x = x_f.acquire(1)
                    pn = p_f.acquire(1)
                    if add:
                        d = d_f.acquire(1)
                        xo = o_f.acquire(1)
                        kb(x, d, wb, st, pn, xo, kc)
                        o_f.release(1)
                        d_f.release(1)
                    else:
                        kb(x, *wp, wb, st, pn, kc)
                    p_f.release(1)
                    x_f.release(1)

    workers = []
    for c in range(op.cols):
        args = [xf[c][1].cons(), pf[c].prod(), k_w, k_a, k_b, Buffer(w_ty, name=f"pn_w{c}"),
                Buffer(st_ty, name=f"pn_st{c}")]
        if add:
            args += [df[c][1].cons(), of[c].prod()]
        elif pre:
            args.append(Buffer(w_ty, name=f"pn_w1{c}"))
        workers.append(Worker(core, args, tile=Tile(c, 2), stack_size=op.STACK, while_true=False))

    n = T * E

    def groups(c: int) -> TensorAccessPattern:  # the core's groups (8 consecutive rows each)
        return TensorAccessPattern((n,), 8 * E * c, [1, n_g, 8 * E // 1024, 1024], [0, 8 * op.cols * E, 1024, 1])

    dp, n_d = op.d_pitch, T * COLS * op.d_pitch
    ed = COLS * dp

    def dgroups(c: int) -> TensorAccessPattern | list[TensorAccessPattern]:  # [group][8 tokens][column block][E / 8]
        if nd == 2:  # (both deltas of a group, [group][delta][8 tokens x column block][E / 8]: a chain of a tap per
            # delta, their stride may exceed a BD's)
            return [TensorAccessPattern((op.d_stride + n_d,), 8 * ed * c + k * op.d_stride,
                                        [n_g, 1, 8 * COLS, E // COLS], [8 * op.cols * ed, 0, dp, 1]) for k in range(2)]
        if dp == E // COLS:
            return groups(c)
        return TensorAccessPattern((n_d,), 8 * ed * c, [n_g, 8, COLS, E // COLS], [8 * op.cols * ed, ed, dp, 1])

    x_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
    d_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
    o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
    p_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]

    def seq(*a):
        if add:
            x, d, w, xo, xn, xh, ph, dh, oh = a
        else:
            x, w, xn, xh, ph = a
        tg = TaskGroup()
        for c in range(op.cols):
            # w as a group-shaped object (8 copies; pre: w1's, then w2's)
            nw = 2 if pre else 1
            wr = 1024 if E % 1024 == 0 else 512  # (a shim BD's rows of at most 1023 words)
            xh[c].fill(w, TensorAccessPattern((nw * E,), 0, [nw, 8, E // wr, wr], [E if pre else 0, 0, wr, 1]),
                       group=tg)
            xh[c].fill(x, groups(c), group=tg)
            if add:
                dg = dgroups(c)
                if isinstance(dg, list):
                    chained_fill(dh[c], d, dg, tg)
                else:
                    dh[c].fill(d, dg, group=tg)
                # x_out chunks [group][kc][8 tokens][CH]
                oh[c].drain(xo, TensorAccessPattern((n,), 8 * E * c, [n_g, n_kc, 8, CH], [8 * op.cols * E, CH, E, 1]),
                            group=tg, wait=True)
            ph[c].drain(xn, TensorAccessPattern((T * P * 9 // 16,), (c * (P // 8) + col // 8) * 36,
                                                [1, n_g, 8, E // 64 * 36], [0, op.cols * (P // 8) * 36, E // 64 * 36, 1]),
                        group=tg, wait=True)
        tg.finish()

    vt = np.ndarray[(n,), bf]
    n_ds = (nd - 1) * op.d_stride + n_d
    types = [vt] + ([np.ndarray[(n_ds,), bf]] if add else []) + [np.ndarray[((2 if pre else 1) * E,), bf]]
    types += [vt] if add else []
    types.append(np.ndarray[(T * P * 9 // 16,), bf])
    rt = Runtime(seq, [*types, x_h, p_h, *([d_h, o_h] if add else [])])
    return Program(NPU2(), rt, workers=workers).resolve_program()
