"""``PrefillNormOperator``'s hold design (more than two deltas): a group's sum held in core L1 between the passes."""

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

if TYPE_CHECKING:
    from .prefill_norm import PrefillNormOperator

HG, HCH = 4, 64  # tokens per group, columns per chunk


def hold_design(op: "PrefillNormOperator") -> Module:
    """deltas > 2: core c takes tokens [c T / cols, (c + 1) T / cols) in groups of HG, a group's x + d1 + .. + dn held
    in L1 between pass A (x chunks straight from the shim, the deltas' chunks re-laid by the memtile:
    [kc][delta][HG][HCH]) and pass B (half panels: the group's HG tokens of each panel's 8); the memtile joins two
    groups' half panels into whole panel rows (the shim writes long runs: scattered 36-byte writes ran at a fraction
    of the DDR rate); each byte read once."""
    T, E, p, nd, cols = op.T, op.E, op.p, op.deltas, op.cols
    tc = T // cols  # (a core's tokens)
    n_g, n_kc, n_w, n_r = tc // HG, E // HCH, E // (HG * HCH), tc // 8
    bf = np.dtype[bfloat16]
    chunk_ty = np.ndarray[(HG * HCH,), bf]
    dch_ty = np.ndarray[(nd * HG * HCH,), bf]
    pan_ty = np.ndarray[(HCH // 8 * 18,), bf]  # (HCH / 8 half panels of 36 B)
    row_ty = np.ndarray[(E // 8 * 36,), bf]  # (a panel row: 8 tokens)
    v_ty, w_ty = np.ndarray[(HG * E,), bf], np.ndarray[(E,), bf]
    st_ty = np.ndarray[(4 * 32 + 64,), np.dtype[np.float32]]
    k_w = Kernel(f"{p}_w", "kernel.o", [chunk_ty, w_ty, np.int32])
    k_a = Kernel(f"{p}_a", "kernel.o", [chunk_ty, dch_ty, v_ty, chunk_ty, st_ty, np.int32])
    k_b = Kernel(f"{p}_b", "kernel.o", [v_ty, w_ty, st_ty, pan_ty, np.int32])
    relay = stream_walk(nd * HG * E, [(n_kc, HCH), (nd, HG * E), (HG, E), (HCH, 1)])
    # (the memtile: a group's deltas double-buffered where they fit next to two panel rows)
    d_depth = 2 if 2 * nd * HG * E * 2 + 2 * E // 8 * 72 <= 500 * 1024 else 1

    def deltas(c: int) -> tuple[ObjectFifo, ObjectFifo]:  # (a group's deltas [delta][HG][E] -> chunks)
        l2 = ObjectFifo(np.ndarray[(nd * HG * E,), bf], name=f"pnhd{c}", depth=d_depth)
        return l2, l2.cons().forward(obj_type=dch_ty, name=f"pnhd1{c}", depth=2, tile=Tile(c, 1), to_stream=relay)

    def panels(c: int) -> tuple[ObjectFifo, ObjectFifo]:  # (two groups' half panels [half][panel][36 B] -> a row)
        rows = ObjectFifo(row_ty, name=f"pnhr{c}", depth=2)
        (halves,) = rows.prod().join([0], tile=Tile(c, 1), obj_types=[pan_ty], depths=[2], names=[f"pnhp{c}"],
                                     from_stream=[stream_walk(E // 8 * 36, [(2, 18), (E // 8, 36), (18, 1)])])
        return halves, rows

    xf = [ObjectFifo(chunk_ty, name=f"pnh_x{c}", depth=2) for c in range(cols)]
    df = [deltas(c) for c in range(cols)]
    of = [ObjectFifo(chunk_ty, name=f"pnh_o{c}", depth=2) for c in range(cols)]
    pf = [panels(c) for c in range(cols)]

    def core(x_f, d_f, o_f, p_f, kw, ka, kb, vb, wb, st):
        for _ in forever_():
            for j in range_(n_w):  # (w: the first objects of the x stream)
                w = x_f.acquire(1)
                kw(w, wb, j)
                x_f.release(1)
            for _ in range_(n_g):
                for kc in range_(n_kc):
                    x, d, xo = x_f.acquire(1), d_f.acquire(1), o_f.acquire(1)
                    ka(x, d, vb, xo, st, kc)
                    o_f.release(1)
                    d_f.release(1)
                    x_f.release(1)
                for kc in range_(n_kc):
                    pn = p_f.acquire(1)
                    kb(vb, wb, st, pn, kc)
                    p_f.release(1)

    workers = [Worker(core, [xf[c].cons(), df[c][1].cons(), of[c].prod(), pf[c][0].prod(), k_w, k_a, k_b,
                             Buffer(v_ty, name=f"pnh_v{c}"), Buffer(w_ty, name=f"pnh_w{c}"),
                             Buffer(st_ty, name=f"pnh_st{c}")],
                      tile=Tile(c, 2), stack_size=op.STACK, while_true=False) for c in range(cols)]
    n, dp, n_d = T * E, op.d_pitch, T * COLS * op.d_pitch
    ed = COLS * dp

    def chunks(c: int) -> TensorAccessPattern:  # (core c's groups as [group][kc][HG tokens][HCH])
        return TensorAccessPattern((n,), tc * E * c, [n_g, n_kc, HG, HCH], [HG * E, HCH, E, 1])

    x_h = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
    d_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
    o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
    p_h = [f[1].cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]
    n_ds = (nd - 1) * op.d_stride + n_d
    rb = E // 8 * 36  # (a panel row's bf16; in runs of 1152)
    assert rb % 1152 == 0

    def seq(x, d, w, xo, xn, xh, dh, oh, ph):
        tg = TaskGroup()
        for c in range(cols):
            xh[c].fill(w, TensorAccessPattern((E,), 0, [1, 1, E // 1024, 1024], [0, 0, 1024, 1]), group=tg)
            xh[c].fill(x, chunks(c), group=tg)
            # (per group a tap per delta: its HG rows of 8 column blocks, d_pitch apart)
            chained_fill(dh[c], d, [TensorAccessPattern((n_ds,), tc * ed * c + k * op.d_stride,
                                                        [n_g, 1, HG * COLS, E // COLS], [HG * ed, 0, dp, 1])
                                    for k in range(nd)], tg)
            oh[c].drain(xo, chunks(c), group=tg, wait=True)
            ph[c].drain(xn, TensorAccessPattern((n * 9 // 16,), n_r * rb * c, [1, n_r, rb // 1152, 1152],
                                                [0, rb, 1152, 1]), group=tg, wait=True)
        tg.finish()

    vt = np.ndarray[(n,), bf]
    rt = Runtime(seq, [vt, np.ndarray[(n_ds,), bf], w_ty, vt, np.ndarray[(n * 9 // 16,), bf], x_h, d_h, o_h, p_h])
    return Program(NPU2(), rt, workers=workers).resolve_program()
