"""The streams and the closing steps GemmQ and GemmQd designs share: B's chain (shim -> memtile -> the column's L1), A's
strips (one per MAC row, broadcast to the columns), C's join in each column's memtile, the runtime sequence over the
shim's producers and consumers, and the pinning of the layout's L1 regions. The operators keep what differs: their
extra FIFOs (X or D), their cores and their buffers."""

from collections.abc import Callable
from typing import Any

import numpy as np
from aie.ir import Module
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ..operators.design import COLS
from ..operators.l1_layout import pin_l1_buffers
from .dataflow import Dataflow


def nbytes(ty: Any) -> int:
    """The bytes of an ``np.ndarray[shape, np.dtype[T]]`` type."""
    shape, dt = ty.__args__
    return int(np.prod(shape)) * np.dtype(dt.__args__[0]).itemsize


def b_chain(df: Dataflow, l2_ty: Any, l1_ty: Any, depth: int) -> tuple[list[ObjectFifo], list[Any]]:
    """B per column: the shim's fill into the memtile (``gqB_<c>``), forwarded to the column's core (``gqB1_<c>``) in
    the k-tile objects' order."""
    l3 = [ObjectFifo(l2_ty, name=f"gqB_{c}", depth=2, from_stream_per_cons=df.b_wdims) for c in range(COLS)]
    l1 = [l3[c].cons().forward(obj_type=l1_ty, name=f"gqB1_{c}", depth=depth, tile=Tile(c, 1), to_stream=df.b_rdims)
          for c in range(COLS)]
    return l3, l1


def a_chain(df: Dataflow, l2_ty: Any, l1_ty: Any, rows: int) -> tuple[list[ObjectFifo], list[Any]]:
    """A per core row: one strip in the memtile (2 row, 1), broadcast to the 8 columns."""
    l3 = [ObjectFifo(l2_ty, name=f"gqA_{row}", depth=1 if df.a_res else 2, from_stream_per_cons=df.a_wdims)
          for row in range(rows)]
    l1 = [l3[row].cons().forward(obj_type=l1_ty, name=f"gqA1_{row}", depth=2, tile=Tile(2 * row, 1),
                                 repeat_count=df.a_repeat if df.a_repeat > 1 else None)
          for row in range(rows)]
    return l3, l1


def c_join(df: Dataflow, l2_ty: Any, l1_ty: Any, rows: int, depth: int) -> tuple[list[ObjectFifo], list[list[Any]]]:
    """C per column: the column's cores' objects (``gqC1_<c>_<row>``, ``l1_ty``) join in its memtile (``gqC_<c>``);
    the second list is indexed by row, then column."""
    l3: list[ObjectFifo] = []
    l1: list[list[Any]] = [[] for _ in range(rows)]
    for c in range(COLS):
        f = ObjectFifo(l2_ty, name=f"gqC_{c}", depth=2, to_stream=df.c_dims)
        l3.append(f)
        parts = f.prod().join([df.c_l1 * i for i in range(rows)], obj_types=[l1_ty] * rows,
                              names=[f"gqC1_{c}_{row}" for row in range(rows)], depths=[depth] * rows, tile=Tile(c, 1))
        for row in range(rows):
            l1[row].append(parts[row])
    return l3, l1


def resolve(op: Any, workers: list[Any], streams: tuple[list, list, list], rows: int, *, c_bfp16: bool) -> Module:
    """The design's module: the runtime sequence (``op._sequence``) over the shim's side of the A, B and C streams
    (A_l3, B_l3, C_l3) with the operator's A panels, weights and C as arguments."""
    a_l3, b_l3, c_l3 = streams
    A_ty = np.ndarray[(op.M * op.a_K * 9 // 16,), np.dtype[bfloat16]]
    W_ty = np.ndarray[(op.w_bytes // 2,), np.dtype[bfloat16]]  # bf16: fused ELFs pass bf16 buffers
    C_ty = np.ndarray[(op.M * op.N * 9 // 16 if c_bfp16 else op.M * op.N,), np.dtype[bfloat16]]
    a_prods = [a_l3[row].prod(tile=Tile(2 * row, 0)) for row in range(rows)]
    b_prods = [b_l3[c].prod(tile=Tile(c, 0)) for c in range(COLS)]
    c_conses = [c_l3[c].cons(tile=Tile(c, 0)) for c in range(COLS)]
    sequence = op._sequence(a_prods, b_prods, c_conses, rows)
    rt = Runtime(sequence, [A_ty, W_ty, C_ty, [a_prods, b_prods, c_conses]])
    return Program(NPU2(), rt, workers=workers).resolve_program()


def pin(module: Module, layout: dict[str, Any], roles: dict[str, str],
        fifo_filter: Callable[[str, list[int] | None], list[int] | None] | None = None) -> None:
    """Pin the layout's regions: a FIFO ``<prefix>_...`` to the region of ``roles[prefix]``, a worker buffer
    ``gq_<role>_...`` to its role's. ``fifo_filter(name, regions)``: the FIFO's regions as the layout gives them,
    changed or dropped (None) by the operator."""

    def worker_buffer(name: str) -> int | None:
        role = name.split("_")[1] if name.startswith("gq_") else None
        return layout[role][0] if role in layout else None

    def fifo(name: str, depth: int) -> list[int] | None:
        regions = layout.get(roles.get(name.split("_")[0], ""))
        return regions if fifo_filter is None else fifo_filter(name, regions)

    pin_l1_buffers(module, fifo=fifo, buffer=worker_buffer)
