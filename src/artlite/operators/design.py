"""Building blocks of the operators' array designs (aie.iron): NPU2 geometry, i32 loops and access patterns."""

import math
from collections.abc import Iterator, Sequence

from aie.dialects import arith, index, scf
from aie.dialects.aie import T
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Attribute, IndexType, InsertionPoint
from aie.iron.dataflow import objectfifo as _objectfifo

COLS = 8  # array columns
ROWS = 4  # compute rows per column (rows 2-5; memtile row 1, shim row 0)


def range32(stop: object, *, unroll: bool = True) -> Iterator[object]:
    """``scf.for`` from 0 to stop (an int or an i32 value) with i32 bounds; yields the i32 induction variable, for a
    body that computes with it. Otherwise use IRON's ``range_``: aiecc narrows its loop to i32 when the bound is a
    constant or read from a buffer (``range_(ws[i])``; aie-core-int-range-narrowing; the same code). ``unroll=False``:
    LLVM keeps the loop rolled (a dynamic core rotates its FIFO objects' pointers on every release, and opt fully
    unrolls a constant-trip loop to drop that rotation: program memory)."""
    c0, c1 = arith.constant(T.i32(), 0), arith.constant(T.i32(), 1)
    stop = arith.constant(T.i32(), stop) if isinstance(stop, int) else stop
    if isinstance(stop.type, IndexType):
        stop = index.casts(T.i32(), stop)
    for_op = scf.ForOp(c0, stop, c1)
    if not unroll:
        for_op.attributes["llvm.loop_annotation"] = Attribute.parse("#llvm.loop_annotation<unroll = <disable = true>>")
    with InsertionPoint(for_op.body):
        yield for_op.induction_variable
        scf.yield_([])


def times_i32(value: object, k: int) -> object:
    """An i32 value (e.g. a scratchpad parameter's) times the constant k, in a core's program."""
    return arith.muli(value, arith.constant(T.i32(), k))


def linear_tap(size: int, offset: int, length: int) -> TensorAccessPattern:
    """``length`` consecutive elements from ``offset`` of a 1-D tensor of ``size`` elements."""
    return TensorAccessPattern((size,), offset, [1, 1, 1, length], [0, 0, 0, 1])


def stream_walk(n: int, dims: Sequence[tuple[int, int]],
                pad: Sequence[tuple[int, int]] | None = None) -> TensorAccessPattern:
    """An ObjectFifo walk (``to_stream`` / ``from_stream``) of each transfer of ``n`` elements: ``dims``
    are (size, stride) pairs, outermost first; ``pad`` (a memtile's MM2S only) gives (before, after)
    elements per dimension. A padded MM2S moves exactly its padded walk (mlir-aie sets the BD length to
    it), an unpadded one the whole object: all-zero ``pad`` reads only the walk's part of the object."""
    tap = TensorAccessPattern((n,), 0, [s for s, _ in dims], [st for _, st in dims])
    if pad and not any(b or a for b, a in pad):  # (TensorAccessPattern.pad drops an all-zero padding)
        return TensorAccessPattern._raw(tap.tensor_dims, tap.offset, tap.sizes, tap.strides, tuple(map(tuple, pad)))
    return tap.pad(pad) if pad else tap


_check_walk_size = _objectfifo._check_walk_size


def _check_padded_walk_size(tap: TensorAccessPattern | None, sizes: list[int], emits: int, what: str) -> None:
    """IRON's check that a padded ``to_stream`` emits exactly one consumer object, relaxed to whole objects:
    a memtile object may pad into several (GemmQd's planar weights: 4 k-tiles -> 4 L1 objects); a walk with all-zero
    padding (``stream_walk``) may emit less than an object (a partial read) or more (rows read past their end, into
    the next): only towards a shim, whose drains set the transfer lengths (GemmQd's packed C rows, whole-line
    windows)."""
    if tap is not None and tap.padding is not None:
        n = math.prod(tap.padded_sizes)
        partial = not any(b or a for b, a in tap.padding)
        if n % emits == 0 or partial:
            emits = n
    _check_walk_size(tap, sizes, emits, what)


_objectfifo._check_walk_size = _check_padded_walk_size
