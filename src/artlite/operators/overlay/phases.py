"""Host side of the decode overlay's GEMV phases (``overlay.decode``): the K groups of a phase, their
metadata and weight reads.

A phase is one region of the image (``image.Region``) against S <= 4 slots' inputs. Its K chunks run in groups of at
most ``decode.MAX_K`` elements (the x panels of a group live in the helper core's memory): each group is an input of its own
(a metadata object and the slots' x interleaved: ``decode.input_fill``) followed by its tiles, a row tile's chunks of
the group in a row.
"""

from dataclasses import dataclass

import numpy as np

from ..formats import FORMATS
from . import gemv
from .decode import (
    B_F32,
    B_G2,
    B_G4,
    B_PAIR,
    F_EMITR,
    F_LOADR,
    F_MAIN,
    F_MSG,
    F_NORM,
    F_POST,
    F_PZERO,
    M_B,
    M_FLAGS,
    M_G,
    M_K,
    M_NB,
    M_NRT,
    M_NS,
    M_NT,
    M_NX,
    M_SLOT,
    MAX_K,
    META,
    NBLK,
    POST_K,
    n_inputs,
)
from .image import Region


@dataclass
class Group:
    """A K group of a phase: chunks k0 .. k0 + g, the metadata of its input."""

    k0: int  # first K chunk
    g: int  # chunks
    meta: np.ndarray  # (META,) int32
    kc: int  # columns of a chunk

    @property
    def K(self) -> int:
        return self.g * self.kc


def valid_rows(region: Region) -> list[int]:
    """Valid rows per column of each block (the same in every column)."""
    return [int((b.rows[0] >= 0).sum()) for b in region.blocks]


def group_chunks(region: Region) -> list[tuple[int, int]]:
    """(first chunk, chunks) of the K groups: the region's own groups of chunks (``kc_group``) when they fit, a row
    tile's chunks split evenly into groups of at most MAX_G = MAX_K / kc (27B o: 6 -> 3 + 3), or groups of MAX_G of
    a chunk-major region (27B down: 17 -> 5, 5, 5, 2)."""
    n, g = region.n_kc, region.kc_group
    max_g = MAX_K // region.kc
    if g <= max_g and n % g == 0 and (g == n or g > 1):
        return [(k, g) for k in range(0, n, g)]
    if g == n:
        parts = -(-n // max_g)
        sizes = [n // parts + (i < n % parts) for i in range(parts)]
        out, k = [], 0
        for s in sizes:
            out.append((k, s))
            k += s
        return out
    assert g == 1, (n, g)
    return [(k, min(max_g, n - k)) for k in range(0, n, max_g)]


def groups(region: Region, flags: int, S: int, *, f32_ab: bool = False,
           block_flags: dict[str, int] | None = None) -> list[Group]:
    """The K groups of a phase on ``region``: F_NORM, F_POST, F_PZERO on the first, F_EMITR on the last (an F_POST
    header: input objects of POST_K). The "ab" block (alpha, beta; or "alpha" of BFP16 planes, B_PAIR, and "beta")
    goes out as fp32 with F_MSG (the GDN core's) or ``f32_ab``; ``block_flags``: a block's B_* bits by its name; a block
    of row groups (``Block.G``) gets B_G2 / B_G4."""
    assert len(region.blocks) <= NBLK and 1 <= S <= gemv.SLOTS
    blk, rt = [], 0
    for b, nv in zip(region.blocks, valid_rows(region)):
        rt += b.tiles
        f32 = B_F32 if (flags & F_MSG or f32_ab) and b.name in ("ab", "beta") else 0
        f32 |= B_PAIR if b.name == "alpha" else 0
        assert b.G in (1, 2, 4), b.G
        g = {1: 0, 2: B_G2, 4: B_G4}[b.G]
        blk += [rt, FORMATS[b.fmt].fid | f32 | g | (block_flags or {}).get(b.name, 0), nv]
    parts = group_chunks(region)
    out = []
    for i, (k0, g) in enumerate(parts):
        m = np.zeros(META, np.int32)
        last = i + 1 == len(parts)
        first = F_NORM | F_POST | F_PZERO
        f = F_MAIN | flags & ~(first | F_EMITR) | (flags & first if i == 0 else 0) | (flags & F_EMITR if last else 0)
        nx = g * region.kc // POST_K if flags & F_POST else n_inputs(g * region.kc)
        m[M_FLAGS], m[M_NS], m[M_NX], m[M_K] = f, S, nx, g * region.kc
        m[M_NT], m[M_G], m[M_NRT], m[M_NB] = region.n_rt * g, g, region.n_rt, len(region.blocks)
        m[M_B:M_B + len(blk)] = blk
        out.append(Group(k0, g, m, region.kc))
    return out


def a_reads(region: Region, c: int, group: Group) -> list[tuple[int, int, tuple | None, int, tuple | None]]:
    """The A fill of column c's tiles of a group: (byte offset in the image, bytes, BD dims, repeat, iteration), the
    tiles in the order the cores run them (row tile major, the group's chunks in a row)."""
    base = region.offset + c * region.col_stride
    S, n, g_img, n_rt = region.tile_bytes, region.n_kc, region.kc_group, region.n_rt
    tw = S // 4  # words of a tile, read in n0-word pieces (a BD's inner dimension is < 1024 words)
    n0 = next(k for k in range(1023, 0, -1) if tw % k == 0)
    if group.g == g_img:  # a block of whole row tiles' chunk groups: contiguous
        return [(base + group.k0 * n_rt * S, n_rt * group.g * S, None, 1, None)]
    if g_img == n:  # part of each row tile's chunks: per row tile g tiles, n tiles apart
        return [(base + group.k0 * S, n_rt * group.g * S, (n0, group.g * tw // n0, n0, n * tw), 1, None)]
    # chunk-major (g_img = 1): per row tile (repeat, iteration) its g chunks, n_rt tiles apart
    return [(base + group.k0 * n_rt * S, group.g * S, (n0, tw // n0, n0, n_rt * tw), n_rt, (tw, n_rt))]


def loadr_meta(slot: int) -> np.ndarray:
    """Metadata of an F_LOADR object (the payload: slot ``slot``'s residual slice, fp32)."""
    m = np.zeros(META, np.int32)
    m[M_FLAGS], m[M_SLOT] = F_LOADR | F_MAIN, slot
    return m
