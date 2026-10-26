"""Host model of the AIE DMA address generators: check data layouts without an NPU.

An operator's data movement is a chain of hops: a shim BD reads DDR through a
``TensorAccessPattern``; a memtile ObjectFifo writes each incoming object with its
S2MM walk (``from_stream_per_cons``) and sends it with its MM2S walk (``to_stream``),
possibly ``repeat_count`` times; the receiving FIFO cuts the stream into its own
objects. A walk is a ``TensorAccessPattern`` at offset 0, in elements of the object
type: stream element k goes to / comes from offset sum(index_d * stride_d) of the
k-th iteration. When an object is longer than the walk covers, the outermost
dimension keeps stepping (the DMA's unbounded outer loop), e.g. one core's
un-panelizing pattern applied to a joined object of 4 cores.

Test a design by pushing uniquely numbered data through its hops and comparing
each consumer object with what the kernel expects (``tests/gemm/test_dma.py``). ``check_*`` encode the BD limits met so far.
"""

from collections.abc import Sequence

import numpy as np
from aie.helpers.taplib import TensorAccessPattern

SHIM_INNER_WORDS_MAX = 1023  # innermost wrap of a multi-dimensional BD, in words (10-bit wrap)
MEMTILE_BD_BYTES_MAX = 512 * 1024 - 4  # one memtile BD: a 17-bit length in words (442 KB A strips run; a BD chain
# cannot carry a repeat)
MEMTILE_DIMS_MAX = 4
SHIM_DIMS_MAX = 4  # 3 address dimensions + the repeat


def walk_offsets(walk: TensorAccessPattern | None, length: int) -> np.ndarray:
    """Element offsets, in stream order, of a walk over an object of ``length`` elements (None:
    linear); -1 where the walk's padding inserts an element. A padded walk moves exactly its padded
    sizes (mlir-aie sets such a BD's length to them, also below the object's: a partial read); an
    unpadded one the object (the outermost dimension keeps stepping)."""
    if walk is None:
        return np.arange(length)
    sizes = list(walk.sizes)
    if not walk.padding:
        inner = int(np.prod(sizes[1:]))
        assert length % inner == 0, f"object of {length} elements is not a whole number of {inner}-element rows"
        sizes[0] = length // inner
    idx = (np.array(walk.strides).reshape(-1, *[1] * len(sizes)) * np.indices(sizes)).sum(0)
    return (np.pad(idx, walk.padding, constant_values=-1) if walk.padding else idx).reshape(-1)


def tap_offsets(tap: TensorAccessPattern) -> np.ndarray:
    """Element offsets of a shim tensor access pattern, in stream order."""
    sizes, strides = list(tap.sizes), np.array(tap.strides)
    idx = np.indices(sizes).reshape(len(sizes), -1)
    return tap.offset + (strides[:, None] * idx).sum(0)


def read(tensor: np.ndarray, tap: TensorAccessPattern) -> np.ndarray:
    """Shim MM2S: the stream a fill with ``tap`` sends from a flat tensor."""
    return tensor.reshape(-1)[tap_offsets(tap)]


def write(tensor: np.ndarray, tap: TensorAccessPattern, stream: np.ndarray) -> None:
    """Shim S2MM: a drain with ``tap`` writes ``stream`` into a flat tensor (in place)."""
    tensor.reshape(-1)[tap_offsets(tap)] = stream


def receive(stream: np.ndarray, obj_len: int, walk: TensorAccessPattern | None = None) -> list[np.ndarray]:
    """A FIFO endpoint cuts ``stream`` into objects of ``obj_len`` elements, each written with
    ``walk`` (S2MM, ``from_stream_per_cons``)."""
    assert stream.size % obj_len == 0, f"stream of {stream.size} is not a whole number of {obj_len}-element objects"
    assert walk is None or walk.padding is None, "only an MM2S walk pads"
    off = walk_offsets(walk, obj_len)
    out = []
    for i in range(stream.size // obj_len):
        obj = np.zeros(obj_len, stream.dtype)
        obj[off] = stream[i * obj_len:(i + 1) * obj_len]
        out.append(obj)
    return out


def send(objects: list[np.ndarray], walk: TensorAccessPattern | None = None, repeat: int = 1) -> np.ndarray:
    """A FIFO endpoint sends each object with ``walk`` (MM2S, ``to_stream``; padding sends zeros),
    ``repeat`` times in a row."""
    parts = []
    for obj in objects:
        off = walk_offsets(walk, obj.size)
        s = np.where(off >= 0, obj[np.maximum(off, 0)], 0)
        parts += [s] * repeat
    return np.concatenate(parts) if parts else np.zeros(0)


def join(parts: list[list[np.ndarray]]) -> list[np.ndarray]:
    """A memtile join: object i is the concatenation of object i of every producer."""
    return [np.concatenate(objs) for objs in zip(*parts, strict=True)]


def canonical(sizes: Sequence[int], strides: Sequence[int]) -> tuple[list[int], list[int]]:
    """Drop size-1 dimensions and merge contiguous ones (stride of a dim = size x stride of the next):
    the address generator a BD actually needs."""
    dims = [(sz, st) for sz, st in zip(sizes, strides) if sz > 1]
    out: list[list[int]] = []
    for sz, st in reversed(dims):
        if out and st == out[-1][0] * out[-1][1]:
            out[-1][0] *= sz
        else:
            out.append([sz, st])
    out.reverse()
    return [d[0] for d in out], [d[1] for d in out]


def check_shim_tap(tap: TensorAccessPattern, elem_bytes: int) -> None:
    """Raise if a shim BD cannot express ``tap``: at most 3 address dimensions plus the repeat, and
    in a multi-dimensional BD an innermost wrap of at most 1023 words (a linear transfer has no
    such limit)."""
    sizes, strides = canonical(tap.sizes, tap.strides)
    if len(sizes) > SHIM_DIMS_MAX:
        raise ValueError(f"shim tap has {len(sizes)} dimensions (max {SHIM_DIMS_MAX})")
    if len(sizes) > 1:
        inner_words = sizes[-1] * elem_bytes // 4 if strides[-1] == 1 else sizes[-1]
        if inner_words > SHIM_INNER_WORDS_MAX:
            raise ValueError(f"shim tap innermost dimension of {inner_words} words (max {SHIM_INNER_WORDS_MAX}); "
                             f"split it, e.g. [n][{SHIM_INNER_WORDS_MAX // 2 + 1}]")


def check_memtile_object(obj_bytes: int, walk: TensorAccessPattern | None = None, *, repeated: bool = False) -> None:
    """Raise if a memtile BD cannot move one object of ``obj_bytes`` with ``walk``."""
    if walk is not None and walk.rank > MEMTILE_DIMS_MAX:
        raise ValueError(f"memtile walk has {walk.rank} dimensions (max {MEMTILE_DIMS_MAX})")
    if obj_bytes > MEMTILE_BD_BYTES_MAX and repeated:
        raise ValueError(f"a repeated memtile object of {obj_bytes} B exceeds one BD ({MEMTILE_BD_BYTES_MAX} B)")
