"""``PrefillNormOperator``'s plain design: a group of 8 tokens in L1 next to the streams, two passes over x."""

from typing import TYPE_CHECKING

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ...operators.design import COLS

if TYPE_CHECKING:
    from .prefill_norm import PrefillNormOperator

CH = 256  # columns per chunk (E / 8 for E < 2048)


def plain_design(op: "PrefillNormOperator") -> Module:
    T, add = op.T, op.add
    EMB, p = op.E, op.p
    n_g = T // (8 * op.cols)  # token groups per core
    bf = np.dtype[bfloat16]
    CH = op.ch
    chunk_ty = np.ndarray[(8 * CH,), bf]
    grp_ty = np.ndarray[(8 * EMB,), bf]
    nl = 2 if op.lo else 1  # (lo: the chunk's hi panels, then its lo panels)
    pan_ty = np.ndarray[(nl * CH // 8 * 36,), bf]  # CH / 8 panels of 72 B
    w_ty = np.ndarray[(EMB,), bf]
    st_ty = np.ndarray[(256 + 64,), np.dtype[np.float32]]
    E = EMB
    k_w = Kernel(f"{p}_w", "kernel.o", [chunk_ty, w_ty])
    k_a = (Kernel(f"{p}_a", "kernel.o", [chunk_ty, chunk_ty, grp_ty, st_ty, np.int32]) if add else
           Kernel(f"{p}_a0p", "kernel.o", [chunk_ty, grp_ty, w_ty, st_ty, np.int32]) if op.pre else
           Kernel(f"{p}_a0", "kernel.o", [chunk_ty, grp_ty, st_ty, np.int32]))
    k_b = Kernel(f"{p}_bl" if op.lo else f"{p}_b", "kernel.o", [grp_ty, w_ty, st_ty, pan_ty, np.int32])
    pre, P, col = op.pre, op.out_pitch, op.out_col

    xf = [ObjectFifo(chunk_ty, name=f"pn_x{c}", depth=2) for c in range(op.cols)]
    df = [ObjectFifo(chunk_ty, name=f"pn_d{c}", depth=2) for c in range(op.cols)] if add else []
    of = [ObjectFifo(grp_ty, name=f"pn_o{c}", depth=1) for c in range(op.cols)] if add else []
    pf = [ObjectFifo(pan_ty, name=f"pn_p{c}", depth=2) for c in range(op.cols)]

    def core(x_f, p_f, kw, ka, kb, wb, st, *rest):
        if add:
            d_f, o_f = rest
        elif pre:
            gb, w1 = rest
        else:
            (gb,) = rest
        for _ in forever_():
            if pre:  # (pre: w1, then w2)
                w = x_f.acquire(1)
                kw(w, w1)
                x_f.release(1)
            w = x_f.acquire(1)
            kw(w, wb)
            x_f.release(1)
            for _ in range_(n_g):
                g = o_f.acquire(1) if add else gb
                for kc in range_(E // CH):
                    x = x_f.acquire(1)
                    if add:
                        d = d_f.acquire(1)
                        ka(x, d, g, st, kc)
                        d_f.release(1)
                    elif pre:
                        ka(x, g, w1, st, kc)
                    else:
                        ka(x, g, st, kc)
                    x_f.release(1)
                for kc in range_(E // CH):
                    p = p_f.acquire(1)
                    kb(g, wb, st, p, kc)
                    p_f.release(1)
                if add:
                    o_f.release(1)

    workers = []
    for c in range(op.cols):
        args = [xf[c].cons(), pf[c].prod(), k_w, k_a, k_b, Buffer(w_ty, name=f"pn_w{c}"),
                Buffer(st_ty, name=f"pn_st{c}")]
        args += [df[c].cons(), of[c].prod()] if add else [Buffer(grp_ty, name=f"pn_g{c}")]
        args += [Buffer(w_ty, name=f"pn_w1{c}")] if pre else []
        workers.append(Worker(core, args, tile=Tile(c, 2), stack_size=op.STACK, while_true=False))

    n = T * E

    def grp(c: int) -> TensorAccessPattern:  # the core's groups as [group][chunk][8 tokens][256]
        return TensorAccessPattern((n,), 8 * E * c, [n_g, E // CH, 8, CH], [8 * op.cols * E, CH, E, 1])

    dp, nd = op.d_pitch, T * COLS * op.d_pitch
    assert dp == E // COLS or E // COLS == CH, "padded delta: one chunk per column block"

    def dgrp(c: int) -> TensorAccessPattern:  # delta's: chunk k = column block k
        ed = COLS * dp
        return TensorAccessPattern((nd,), 8 * ed * c, [n_g, E // CH, 8, CH], [8 * op.cols * ed, dp, ed, 1])

    x_h = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
    d_h = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
    o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
    p_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]

    def seq(*a):
        if add:
            x, d, w, xo, xn, xh, ph, dh, oh = a
        else:
            x, w, xn, xh, ph = a
        tg = TaskGroup()
        for c in range(op.cols):
            # (a shim BD's inner dimension is <= 1023 words: rows of <= 512 words below)
            nw = (2 if pre else 1) * E
            xh[c].fill(w, TensorAccessPattern((nw,), 0, [1, 1, nw // 1024, 1024], [0, 0, 1024, 1]), group=tg)
            xh[c].fill(x, grp(c), group=tg)
            if add:
                dh[c].fill(d, dgrp(c), group=tg)
                oh[c].drain(xo, TensorAccessPattern((n,), 8 * E * c, [1, n_g, 8 * E // 1024, 1024],
                                                    [0, 8 * op.cols * E, 1024, 1]), group=tg, wait=True)
            if op.lo:  # (a group's chunks [hi | lo] -> [hi panels | lo panels]: rows of 2 E)
                ph[c].drain(xn, TensorAccessPattern((2 * n * 9 // 16,), c * (E // 4) * 36,
                                                    [n_g, 8, 2, E // 64 * 36],
                                                    [op.cols * (E // 4) * 36, E // 64 * 36, E // 8 * 36, 1]),
                            group=tg, wait=True)
                continue
            ph[c].drain(xn, TensorAccessPattern((T * P * 9 // 16,), (c * (P // 8) + col // 8) * 36,
                                                [1, n_g, 8, E // 64 * 36], [0, op.cols * (P // 8) * 36, E // 64 * 36, 1]),
                        group=tg, wait=True)
        tg.finish()

    vt = np.ndarray[(n,), bf]
    types = [vt] + ([np.ndarray[(nd,), bf]] if add else []) + [np.ndarray[((2 if pre else 1) * E,), bf]]
    types += [vt] if add else []
    types.append(np.ndarray[((2 if op.lo else 1) * T * P * 9 // 16,), bf])
    rt = Runtime(seq, [*types, x_h, p_h, *([d_h, o_h] if add else [])])
    return Program(NPU2(), rt, workers=workers).resolve_program()
