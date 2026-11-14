"""What the ops share: memtile bursts, the MAC's walks, the command's builder and the memtile bounds.

Sizes of the overlay's layouts (``design``, ``layout``): a pair of 16 token panels is ``PAIR`` bytes; a B slot ``D.Q`` holds
five of them.
"""

import functools
import struct

import numpy as np

from ...txn import HardwareLimitError
from .. import core as K
from .. import design as D
from .. import layout as L

PAIR = 16 * L.PANEL  # (a pair of token panels' bytes of one k-tile column: the MAC's A step)
assert D.Q == 5 * PAIR
CBAND = 16 * 160  # (a bf16 C tile's band of 16 tokens: 16 x 80 channels)
CCOL = 128  # (a C tile's next 64 channels, bytes)
MEMTILE = 512 * 1024  # (a memtile's bytes: a walk past them hangs the DMA)
ZERO_BYTES = 4096  # (a conv's zeros in the memtile: the edge columns' missing halo rows)
ROPE2_TAB = 5120  # (a rope2 tile's tables [A | B], m 16: 2 x 16 tokens x 80 bf16)
ROPE40_TAB = 2560  # (a rope40 tile's table, m 16: 16 tokens x 80 bf16 [rb][cb][64])
assert 2 * ROPE40_TAB == ROPE2_TAB == 2 * CBAND

# the MAC's walks of B and C (three dims_3d_t with A's, ``core.C_PATS``: size 1, step, count, wrap, wrap again, 2 pad words):
# the 4-band forms (m 64, the 4-band convs: the band pairs (0, 2), (1, 3) alternate per block, C's halves in two L1
# regions) and the 1-band forms (m 16, the other convs)
B_WALK_4 = (1, -PAIR, 4, 0, -D.Q)
C_WALK_4 = (1, CBAND, 4, CCOL - CBAND, 6 * CCOL - CBAND)
B_WALK_1 = (4, 0, 0, -D.Q, -D.Q)
C_WALK_1 = (4, CCOL, 0, 6 * CCOL, 6 * CCOL)
PATS_4 = (*B_WALK_4, 0, 0, *C_WALK_4, 0, 0)


def limit(what: str, nbytes: int, bound: int, why: str) -> None:
    """Raise ``HardwareLimitError`` unless nbytes <= bound (what: the thing measured; why: what past it does)."""
    if nbytes > bound:
        raise HardwareLimitError(f"{what}: {nbytes} B past {bound} B ({why})")


def check_memtile(addr: int, nbytes: int, what: str) -> None:
    """The memtile bytes [addr, addr + nbytes) of ``what`` lie in the memtile (an over-read hangs the DMA)."""
    limit(what, addr + nbytes, MEMTILE, "the memtile's end; an over-read hangs")


def check_heap(addr: int, nbytes: int, what: str) -> None:
    """The memtile bytes [addr, addr + nbytes) of ``what`` lie in the tensors' part (``design.HEAP``)."""
    limit(what, addr + nbytes, D.HEAP, "the memtile's tensors")


@functools.cache
def contig(words: int) -> list[tuple[int, int]]:
    """Dims of ``words`` contiguous words (d0 <= 1023)."""
    d0 = next(d for d in range(min(words, 1023), 0, -1) if words % d == 0)
    return [(d0, 1), (words // d0, d0)] if words > d0 else [(d0, 1)]


def dims(*ds: tuple[int, int]) -> list[tuple[int, int]]:
    """A memtile walk (size, stride) innermost first without its 1-dims; a fourth dim's size is implied."""
    out = [d for d in ds[:-1] if d[0] != 1] + [ds[-1]]
    assert len(out) <= 4, ds
    if len(out) == 4:
        out[3] = (0, out[3][1])
    return out


def f32(v: float) -> int:
    """The bits of the float32 v as a command word."""
    return struct.unpack("<i", struct.pack("<f", v))[0]


def i32(words: object) -> np.ndarray:
    """Words (non-negative or negative ints) wrapped to int32."""
    return (np.asarray(words, np.int64) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)


class Command:
    """An op's command: one B slot (``D.Q`` bytes) of header words ``core.C_*``, the kind's regions and the biases."""

    def __init__(self, kind: int):
        self.w = np.zeros(D.Q // 4, np.int32)
        self.w[K.C_KIND] = kind

    def set(self, words: list[int], values: tuple | list) -> None:
        """Header words at the indices ``words`` := values."""
        self.w[words] = values

    def at(self, word: int, values: object) -> None:
        """The consecutive words from ``word`` := values."""
        values = np.asarray(values)
        self.w[word:word + values.size] = values.reshape(-1)

    def bytes(self) -> np.ndarray:
        """The slot as bytes (a view)."""
        return self.w.view(np.uint8)


def panel_recv(addr: int, np_: int, nrb: int, ncb: int, nblk: int, junk_rb: int = 0) -> dict:
    """A receive BD of one tile's output panels [block][rb][cb] -> panel tensor kb = 10 block + cb (ncb per block) from
    ``addr``'s kb 0, token panel rb (junk_rb more padding rows per block: panels beyond the tokens)."""
    rbs = nrb + junk_rb
    return {"addr": addr, "nwords": nblk * rbs * ncb * 18,
            "dims": [(18, 1), (ncb, np_ * 18), (rbs, 18), (0, ncb * np_ * 18)]}


def mac_walks(m: int) -> np.ndarray:
    """The MAC's walks of A, B and C (``core.C_WALKS``, ``C_PATS``: three dims_3d_t) of a GEMM's m-token C tile (m 64:
    the 4-band forms; m 16: one band)."""
    if m == 64:
        w = [(1, PAIR, 4, -3 * PAIR, -2 * PAIR), B_WALK_4, C_WALK_4]
    else:
        w = [(4, -PAIR, m // 16 - 1, 0, 0), (4, 0, m // 16 - 1, -D.Q, -D.Q), (4, CCOL, m // 16 - 1, 6 * CCOL, 6 * CCOL)]
    return np.array([x for d in w for x in (*d, 0, 0)], np.int64).astype(np.int32)


def band_walk(a: list[int]) -> tuple[int, tuple]:
    """The MAC's A walk of a conv call over the bands at object byte offsets a: 4 bands (the pairs (0, 2), (1, 3) equally
    far apart) or one pair; returns (bands, the walk)."""
    if len(a) == 2:
        return 2, (1, a[1] - a[0] - PAIR, 4, a[0] - a[1] - PAIR, 0)
    assert a[2] - a[0] == a[3] - a[1]
    return 4, (1, a[2] - a[0] - PAIR, 4, a[0] - a[2] - PAIR, a[1] - a[2] - PAIR)


def gn_window(ch0: int, n: int, gsize: int) -> int:
    """The first group of the 16-group window of a GroupNorm record (16-aligned) holding channels [ch0, ch0 + n): a
    producer's record and its consumer's must agree."""
    g0 = ch0 // gsize // 16 * 16
    assert (ch0 + n - 1) // gsize < g0 + 16, (ch0, n, gsize)
    return g0
