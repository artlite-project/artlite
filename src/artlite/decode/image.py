"""The weight image of a decode overlay: GGUF matrices re-laid-out into the tile formats of ``artlite.formats``,
packed in memory at every load, straight from the mapped GGUF file (``TileImage``; the models' row assignments:
``qwen35.image``, ``gemma4.image``, ``llama.image``).

The decode overlay streams a column's tiles of each phase in order; the prefill GEMMs (GemmQ / GemmQd) read the same
tiles in place. The image is a list of regions (``Region``: one per layer and phase, and the LM head), each holding
8 column blocks of ``col_bytes``, ``col_stride`` apart: region after region (region-major), or, ``column_major``, each
column's blocks of every region in a row (a column's stream for a whole step is one run of the image: a decode step
streams it with one task per column; a normed phase's norm-weight objects sit in it before its tiles, as a data region
"<phase>_norm": ``norm_objects``), in layout blocks of whole layers of at most ``BLOCK_BYTES`` (a prefill GEMM's weight
argument spans its region's 8 columns: a runtime-sequence argument stays below 4 GB; a step's run then takes a task
per block, ``runs``); column c's block holds the column's rows of one or more matrices (``Block``) as
row tiles of R rows (the format's rows per tile object), every row tile over the K chunks of ``kc`` columns of the
input, in ``gemm_q``'s convention: K chunks in groups of ``kc_group``, group G of row tile j at tile
(G n_rt + j) kc_group (n_rt: the region's row tiles; kc_group = all chunks: a row tile's chunks consecutive).

A region whose input is longer than an activation object's payload (``payload``: the model's) takes its K chunks in
groups (``kc_group``: the largest divisor of the chunk count whose group fits the payload); every other region keeps a
row tile's chunks consecutive. An input of no whole number of chunks (MiniCPM-V's ffn 3584) is padded to the next one:
its tiles' columns past the matrix are zero super-blocks; an input of a fraction 1 / G of a chunk (an MoE's down, K 512
or 256) takes row groups instead (``Block.G``, ``group_rows``: G rows of K columns per tile row, no padding bytes). An
input half a chunk short of whole chunks in one K group (the 4B Spark-X's hidden 2560) may be a ``Region.half`` region
(``tails``): its phases' inputs of that K, the blocks the model pairs (``pairs``: those the prefill reads in pairs)
without zero super-blocks (``Block.pt``).
The image starts and ends with ``PAD`` bytes: a GEMM reads a Q4_K header unit from 1 KB before a tile, and past a
region's last tile.

An MoE layer's experts are slab regions (``slabs``, after the column-major or region-major regions): one block per
expert half instead of per column (``Region.cols``), the layer's slab regions back to back in a half (``_add_slab``),
so that expert e's two halves (blocks 2 e, 2 e + 1) are one contiguous run of the image that a decode column streams
whole.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from .. import _native
from ..formats import FORMATS, choose_tile_bytes, gu_tiles, has_tile_format, iq1, packs_from, slot, tile_format
from ..gguf_model import GGUFFile, QTensor
from ..operators.design import COLS
from ..paro import ParoTensor
from .protocol import NX_PER_NORM, OBJ_K, POST_K
from .screen import HALF, derived, plane_rows

PAD = 64 * 1024
BLOCK_BYTES = 3 << 30  # a column-major layout block at most (8 columns of its layers' regions)
COL_SLOT = 256 * 1024  # a column-major stride is an odd multiple of it (_col_stride)


def _col_stride(nbytes: int) -> int:
    """The stride of a column-major layout block's column streams of ``nbytes``: the next odd multiple of COL_SLOT,
    so the 8 columns' streams start in the 8 different 256 KB slots of a 2 MB span (a decode step of 1B Q4_K_M: 15.0 ms,
    15.4 with the streams back to back; streams that start near the same offset modulo 2 MB, or a third or a half of
    it apart, slow the step's DMA by up to 3%)."""
    k = -(-nbytes // COL_SLOT)
    return (k | 1) * COL_SLOT


def kc_group(n_kc: int, xp: int, kc: int) -> int:
    """K chunks (of kc columns) per group of a region of n_kc chunks: the largest divisor whose group fits ``xp``
    elements."""
    return max(g for g in range(1, n_kc + 1) if n_kc % g == 0 and g * kc <= xp)



def _pack_tiles(out: np.ndarray, blocks: list[tuple], **kw: Any) -> None:
    """``_native.pack_tiles`` with the IQ1_S grid's masks where a block is of IQ1_S tiles (``formats.iq1``'s MK)."""
    masks = iq1.grid_masks() if any(b[0] == "IQ1_S" for b in blocks) else None
    _native.pack_tiles(out, blocks, iq1_masks=masks, **kw)

@dataclass(frozen=True)
class Block:
    """A run of row tiles in every column's block of a region: ``rows`` (cols, n) indices into the stack of the
    ``tensors`` (their rows one after the other; -1: a zero padding row), n a multiple of R G: tile row v holds rows
    v G .. v G + G - 1 of them one after the other along K (G > 1: row groups, ``group_rows``). F32 tensors (an MoE
    router's) are BFP16 tiles of two planes per row: index 2 i + p is plane p of row i (0: hi = bfp16(w), 1: lo =
    bfp16(w - hi); ``formats.bf16``): ``pairs``, the sum of a pair's outputs is the row's.

    A split gate / up block (``fmt2``: gate and up in two tile formats, ``split_rows``): per SiLU group of 64 outputs its
    gate rows in ``na`` tiles of ``fmt`` (R rows), then its up rows in ``nb`` tiles of ``fmt2`` (R2 rows), each part's
    last tile ending in padding rows."""

    name: str
    tensors: tuple[str, ...]
    rows: np.ndarray
    fmt: str
    R: int
    G: int = 1
    pairs: bool = False
    fmt2: str = ""
    R2: int = 0
    pt: bool = False  # (a ``Region.half`` region's) its row tiles in pairs that share their last chunk's tile (``Region``)

    @property
    def formats(self) -> tuple[str, ...]:
        """Its tile formats (a split block's two)."""
        return (self.fmt, self.fmt2) if self.fmt2 else (self.fmt,)

    @property
    def na(self) -> int:
        """(split) Tiles of a group's gate rows."""
        return -(-SPLIT_ROWS // self.R)

    @property
    def nb(self) -> int:
        """(split) Tiles of a group's up rows."""
        return -(-SPLIT_ROWS // self.R2)

    @property
    def tiles(self) -> int:
        if self.fmt2:
            return self.rows.shape[1] // (self.na * self.R + self.nb * self.R2) * (self.na + self.nb)
        return self.rows.shape[1] // (self.R * self.G)

    def parts(self, ffn: int = 0) -> list[tuple[str, int, int, tuple[str, ...], np.ndarray]]:
        """The block's runs of row tiles of one format: (format, R, first row tile, tensors, rows (cols, n)); a split
        block's per group and part, each of its own tensor (ffn: the gate's rows, the up rows' index past them)."""
        if not self.fmt2:
            return [(self.fmt, self.R, 0, self.tensors, self.rows)]
        ga, gb = self.na * self.R, self.nb * self.R2
        out = []
        for j in range(self.rows.shape[1] // (ga + gb)):
            r = self.rows[:, j * (ga + gb):(j + 1) * (ga + gb)]
            up = r[:, ga:]
            out += [(self.fmt, self.R, j * (self.na + self.nb), self.tensors[:1], r[:, :ga]),
                    (self.fmt2, self.R2, j * (self.na + self.nb) + self.na, self.tensors[1:], np.where(up >= 0, up - ffn, -1))]
        return out


@dataclass
class Region:
    """The tiles of one phase: ``cols`` column blocks (the 8 columns'; a slab region's: one per expert half) of
    ``col_bytes`` from ``offset`` on.

    A ``half`` region's input is half a chunk short of its K chunks (``K_in``: the 4B Spark-X's hidden 2560 in 3 chunks;
    a row tile's chunks consecutive, one K group): the last chunk's columns are a tail of nsb / 2 super-blocks. A ``pt``
    block of it holds its row tiles in pairs (2 p, 2 p + 1) of 2 n_kc - 1 tiles each, ``PAIR``: the full chunks of 2 p,
    then the pair's tail tile (super-blocks 0 .. nsb / 2 - 1: row tile 2 p's tail, the rest: 2 p + 1's, as a tile of
    row groups of 2, ``group_rows``), then the full chunks of 2 p + 1; an odd block's last row tile is a single, its
    tiles as a plain block's (its tail tile's second half zero super-blocks). The decode streams a pair's tiles in that
    order (the main core runs a tile's first super-blocks, the helper the others: 2 p's tail on the main core, 2 p + 1's
    on the helper, whose partial rows start 2 p + 1's sums, ``decode.design``); the prefill GEMMs read pass pairs of
    their even and odd row tiles (``gemm.geometry.GemmSpec.ptail``), the other blocks' tail tiles for their first half
    (``Geometry.tail``). The 4B Q4_K_M's gate / up and LM head in pairs: a decode step streams 3.04 GB instead of 3.33."""

    layer: int  # -1: the LM head
    name: str
    blocks: list[Block]
    n_kc: int
    kc_group: int
    tile_bytes: int
    kc: int
    data: int = 0  # (a data region: bytes per column, no tiles: ``TileImage.norm_objects``)
    offset: int = 0  # (column 0's block)
    col_stride: int = 0  # (from a column's block to the next)
    rt0: dict[str, int] = field(default_factory=dict)  # first row tile of each block
    cols: int = COLS
    half: bool = False  # (the input: ``K_in`` = K - kc / 2; module docstring)
    t0: dict[str, int] = field(default_factory=dict)  # (``half``) first tile of each block in a column's block

    def __post_init__(self) -> None:
        assert self.n_kc % self.kc_group == 0
        assert not self.half or (self.kc_group == self.n_kc and self.n_kc >= 2), (self.name, self.n_kc, self.kc_group)
        assert self.half or not any(b.pt for b in self.blocks), self.name
        rt = t = 0
        for b in self.blocks:
            self.rt0[b.name], self.t0[b.name] = rt, t
            rt += b.tiles
            t += self.block_tiles(b)
        self.n_rt, self.n_t = rt, t

    @property
    def K(self) -> int:
        return self.n_kc * self.kc

    @property
    def K_in(self) -> int:
        """The input's elements (a ``half`` region's: half a chunk short of K)."""
        return self.K - self.kc // 2 if self.half else self.K

    @property
    def pair(self) -> int:
        """Tiles of a pair of a ``pt`` block's row tiles."""
        return 2 * self.n_kc - 1

    def block_tiles(self, b: Block) -> int:
        """Tiles of block b in a column's block."""
        n = b.tiles
        return n // 2 * self.pair + n % 2 * self.n_kc if b.pt else n * self.n_kc

    @property
    def col_bytes(self) -> int:
        return self.data or self.n_t * self.tile_bytes

    @property
    def nbytes(self) -> int:
        return self.cols * self.col_bytes

    @property
    def read_bytes(self) -> int:
        """Bytes of tiles a prefill GEMM over the input reads (a ``half`` region's: half a tail tile per row tile)."""
        return self.cols * self.n_rt * (2 * self.n_kc - 1) * self.tile_bytes // 2 if self.half else self.nbytes

    def block_of(self, rt: int) -> Block:
        """The block of row tile rt."""
        return next(b for b in reversed(self.blocks) if self.rt0[b.name] <= rt)

    def tile(self, rt: int, kc: int) -> int:
        """Index in a column block of row tile rt's tile of K chunk kc (a ``pt`` block's last chunk: the pair's tail tile,
        row tile rt's super-blocks in its half ``tail_half``)."""
        if not self.half:
            g = self.kc_group
            return (kc // g * self.n_rt + rt) * g + kc % g
        b = self.block_of(rt)
        j = rt - self.rt0[b.name]
        if not b.pt:
            return self.t0[b.name] + j * self.n_kc + kc
        base = self.t0[b.name] + j // 2 * self.pair
        return base + (self.n_kc - 1 if kc == self.n_kc - 1 else kc + (self.n_kc if j % 2 else 0))

    def tail_half(self, rt: int) -> int:
        """Which half of its last chunk's tile holds row tile rt's tail: 1 for the second of a ``pt`` block's pair."""
        if not self.half:
            return 0
        b = self.block_of(rt)
        return int(b.pt and (rt - self.rt0[b.name]) % 2 == 1)

    def block(self, name: str) -> Block:
        return next(b for b in self.blocks if b.name == name)


def _pad_rows(rows: np.ndarray, R: int) -> np.ndarray:
    """(cols, n) rows -> (cols, ceil(n / R) R), padded with -1."""
    n = -(-rows.shape[1] // R) * R
    out = np.full((rows.shape[0], n), -1, np.int64)
    out[:, :rows.shape[1]] = rows
    return out


class TileImage:
    """A weight image of a GGUF file: ``regions`` and ``slabs`` (``region(layer, name)``), ``nbytes``; ``pack(out)``
    fills a buffer of nbytes. A model's subclass adds its regions (``_add``, ``_add_slab``) and calls ``_place``."""

    def __init__(self, gguf: GGUFFile, *, kc: int, payload: int, tile_bytes: int, vocab: int,
                 max_rows: dict[str, int] | None = None, column_major: bool = False, tails: frozenset[str] = frozenset()):
        """kc: the K chunk of a tile (columns); payload: the elements of an activation object's payload (the K groups);
        tile_bytes: the object size (0: chosen from the file's formats, ``choose_tile_bytes``); max_rows: the rows of
        a format's tile at most (default: the format's); column_major: the layout (module docstring); tails: the regions
        (names) whose input of half a chunk short of whole chunks is a ``Region.half`` region, its blocks of one tile
        format read in the super-block split ``pt`` blocks."""
        self.gguf, self.kc, self.nsb, self.payload, self.vocab = gguf, kc, kc // 256, payload, vocab
        self.tails = tails
        self.tile_bytes = tile_bytes or self._choose_tile_bytes()
        mr = max_rows or {}
        self.R = {f: FORMATS[f].rows(self.tile_bytes, self.nsb, mr.get(f, 0)) for f in FORMATS
                  if FORMATS[f].rows(self.tile_bytes, self.nsb, mr.get(f, 0))}
        self.column_major = column_major
        self.regions: list[Region] = []
        self.slabs: list[Region] = []
        self._own: dict[str, QTensor] = {}  # (tensors a model substitutes: re-laid-out copies in memory)
        self._views: dict[str, QTensor] = {}  # (the file's tensors as rows of another width: ``view``)
        self._chans: dict[str, np.ndarray] = {}  # (PARO linears: their K order, ``tensor``)
        self._paro_views: dict[str, int] = {}  # (PARO linears read as rows of a view: rows of the view per row)

    def tensor(self, name: str) -> QTensor:
        """A tensor the blocks name (the file's, its ``view``, or the model's substitute: ``_own``; a PARO linear with
        its K in the image's order: ``paro_rows``)."""
        t = self._own.get(name) or self._views.get(name) or self.gguf.tensor(name)
        if isinstance(t, ParoTensor) and t.chan is None:
            if name not in self._chans:
                self._chans[name] = t.rot().tan_form().chan
            t.chan = self._chans[name]
        return t

    def view(self, name: str, row_sb: int) -> None:
        """The blocks read the file's tensor ``name`` as rows of row_sb super-blocks (a row of the file: several rows of
        the view, one after the other; an MoE down's halves)."""
        t = self.gguf.tensor(name)
        n = t.cols // (256 * row_sb)
        assert t.cols == n * 256 * row_sb, (name, t.cols, row_sb)
        if isinstance(t, ParoTensor):  # (its rows of blocks as rows of the view: ``paro_rows``)
            self._paro_views[name] = n
            return
        self._views[name] = QTensor(name=name, qtype=t.qtype, rows=t.rows * n, cols=t.cols // n,
                                    raw=t.raw.reshape(t.rows * n, -1))

    def qtype(self, name: str) -> str:
        """A tensor's GGUF type (a substitute's: ``_own``)."""
        return self._own[name].qtype if name in self._own else self.gguf.tensor_info[name][1]

    def fmt(self, name: str) -> str:
        """A tensor's tile format (F32: BFP16 planes, ``Block``)."""
        qtype = self.qtype(name)
        return "BFP16" if qtype == "F32" else tile_format(qtype)

    def _choose_tile_bytes(self) -> int:
        """The object size of the formats in the file (``formats.choose_tile_bytes``: rows streamed per format and K
        chunk, ffn_gate / ffn_up as the gate/up phase's SiLU groups, ``gate_up_rows``; token_embd only as the tied LM
        head)."""
        rows: dict[str, int] = {}
        gu: dict[str, int] = {}
        tied = "output.weight" not in self.gguf
        for name, (dims, qtype, _, _) in self.gguf.tensor_info.items():
            if len(dims) == 2 and dims[0] % 256 == 0 and (tied or name != "token_embd.weight") and has_tile_format(qtype):
                f, n_kc = tile_format(qtype), -(-dims[0] // self.kc)  # (a K of part of a chunk: padded, ``_region``)
                if name.endswith("ffn_gate.weight"):
                    gu[f] = gu.get(f, 0) + dims[1] // 64 * n_kc
                elif not name.endswith("ffn_up.weight"):
                    rows[f] = rows.get(f, 0) + dims[1] * n_kc
        return choose_tile_bytes(rows, self.nsb, gu)

    def _add(self, layer: int, name: str, parts: list[tuple], K: int, *, G: int = 1) -> None:
        """A region of the blocks ``parts`` = (name, tensors, rows (COLS, n)[, tile format]) over an input of K elements
        (the format: a plane of the tensors' own, ``formats.packs_from``; default: theirs); G > 1: K = kc / G, the blocks'
        rows given as row groups (``group_rows``)."""
        self.regions.append(self._region(layer, name, parts, K, G))

    def _add_slab(self, layer: int, name: str, parts: list[tuple[str, tuple[str, ...], np.ndarray]], K: int, *,
                  G: int = 1) -> None:
        """A slab region (as ``_add``, rows (2 E, n): a block per expert half), after the layer's previous ones in a
        half."""
        self.slabs.append(self._region(layer, name, parts, K, G))

    def _region(self, layer: int, name: str, parts: list[tuple], K: int, G: int) -> Region:
        assert G == 1 or G * K == self.kc, (name, K, G)
        n_kc = -(-K // self.kc)
        g = kc_group(n_kc, self.payload, self.kc)
        half = name in self.tails and K % self.kc == self.kc // 2 and n_kc >= 2 and g == n_kc and G == 1
        blocks = []
        for bname, tensors, rows, *own in parts:  # (own: the block's tile format, a plane of its tensors': ``packs_from``)
            fmts = [self.fmt(t) for t in tensors]
            if own:
                assert all(packs_from(own[0], self.qtype(t)) or derived(own[0], self.qtype(t)) for t in tensors) \
                    and G == 1, (bname, own, fmts)
                fmts = list(own)
            if len(set(fmts)) > 1:
                if name not in GU_REGIONS or len(tensors) != 2 or G > 1:
                    raise NotImplementedError(f"{', '.join(tensors)}: one tile format per block, not {sorted(set(fmts))} "
                                              f"(gate and up of two formats: a dense layer's gate / up only)")
                ffn, (fa, fb) = self.tensor(tensors[0]).rows, fmts
                blocks.append(Block(bname, tensors, split_rows(ffn, self.R[fa], self.R[fb], rows.shape[0]), fa,
                                    self.R[fa], fmt2=fb, R2=self.R[fb]))
                continue
            fmt = fmts[0]
            R = self.R[fmt]
            assert G == 1 or rows.shape[1] % (R * G) == 0, (name, bname)
            pairs = all(self.qtype(t) == "F32" for t in tensors)
            # (a tail tile's halves: the super-block split's)
            pt = half and not pairs and not slot(fmt).mmv_split(fmt, self.nsb) and self.pairs(name, bname, fmt, R)
            blocks.append(Block(bname, tensors, rows if G > 1 else _pad_rows(rows, R), fmt, R, G, pairs, pt=pt))
        return Region(layer, name, blocks, n_kc, g, self.tile_bytes, self.kc, cols=parts[0][2].shape[0], half=half)

    def pairs(self, region: str, block: str, fmt: str, R: int) -> bool:
        """Whether a block of a ``Region.half`` region holds its row tiles in pairs (``Block.pt``): the model's (the
        prefill must read them: ``prefill.gemm.pair_tails``)."""
        return False

    def _add_data(self, layer: int, name: str, nbytes: int) -> None:
        """A data region of nbytes per column ("<phase>_norm": ``norm_objects``, written at pack time)."""
        self.regions.append(Region(layer, name, [], 1, 1, self.tile_bytes, self.kc, data=nbytes))

    def _place(self) -> None:
        """The regions' offsets (after the leading PAD) and column strides, then the slab regions', ``nbytes``."""
        if self.column_major:
            end = PAD
            for blk in self._blocks():
                stream = _col_stride(sum(r.col_bytes for r in blk))
                off = end
                for r in blk:
                    r.offset, r.col_stride = off, stream
                    off += r.col_bytes
                end += COLS * stream
        else:
            end = PAD
            for r in self.regions:
                r.offset, r.col_stride = end, r.col_bytes
                end += r.nbytes
        for L in dict.fromkeys(r.layer for r in self.slabs):  # (a layer's halves: its slab regions back to back)
            rs = [r for r in self.slabs if r.layer == L]
            half = sum(r.col_bytes for r in rs)
            off = end
            for r in rs:
                assert r.cols == rs[0].cols
                r.offset, r.col_stride = off, half
                off += r.col_bytes
            end += rs[0].cols * half
        self.nbytes = -(-(end + PAD) // 4096) * 4096
        self._index = {(r.layer, r.name): r for r in self.regions + self.slabs}

    def _blocks(self) -> list[list[Region]]:
        """The regions of a column-major image in layout blocks of whole layers, each of at most BLOCK_BYTES (one
        block when the image is smaller)."""
        layers: list[list[Region]] = []
        for r in self.regions:
            if layers and layers[-1][0].layer == r.layer:
                layers[-1].append(r)
            else:
                layers.append([r])
        blocks: list[list[Region]] = [[]]
        for rs in layers:
            if blocks[-1] and COLS * sum(r.col_bytes for r in blocks[-1] + rs) > BLOCK_BYTES:
                blocks.append([])
            blocks[-1] += rs
        return blocks

    def runs(self, first: Region, last: Region) -> list[tuple[Region, int]]:
        """The regions first .. last (in the image's order) as runs within one layout block each: (the run's first
        region, a column's bytes of it)."""
        i = next(k for k, r in enumerate(self.regions) if r is first)
        j = next(k for k, r in enumerate(self.regions) if r is last)
        out: list[tuple[Region, int]] = []
        for r in self.regions[i:j + 1]:
            if out and out[-1][0].col_stride == r.col_stride and out[-1][0].offset + out[-1][1] == r.offset:
                out[-1] = (out[-1][0], out[-1][1] + r.col_bytes)
            else:
                out.append((r, r.col_bytes))
        return out

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        """The norm-weight objects of phase ``name`` (uint8, the bytes of its data region "<name>_norm", or (COLS, ...):
        each column's own; the model's)."""
        raise NotImplementedError

    def data_objects(self, layer: int, name: str) -> np.ndarray:
        """The objects of data region ``name`` (``_fill_data``): a "<phase>_norm" region's ``norm_objects``."""
        return self.norm_objects(layer, name.removesuffix("_norm"))

    def _fill_data(self, out: np.ndarray) -> None:
        """Write the data regions and slabs: every column's copy of a region's ``data_objects`` (or its own)."""
        for r in self.regions + self.slabs:
            if r.data:
                objs = self.data_objects(r.layer, r.name)
                per = objs.reshape(objs.shape[0], -1) if objs.ndim == 3 else objs.reshape(1, -1)
                for c in range(r.cols):  # (a slab's: objects of fewer columns than its halves repeat)
                    out[r.offset + c * r.col_stride:][:per.shape[1]] = per[c % per.shape[0]]

    def region(self, layer: int, name: str) -> Region:
        return self._index[(layer, name)]

    def formats(self) -> set[str]:
        """The tile formats of the image's blocks (a split block's both)."""
        return {f for r in self.regions + self.slabs for b in r.blocks for f in b.formats}

    @property
    def split(self) -> bool:
        """Whether a gate / up block is split (``Block.fmt2``: the decode design's B_SPLIT tiles)."""
        return any(b.fmt2 for r in self.regions + self.slabs for b in r.blocks)

    def streamed_bytes(self) -> int:
        """Bytes of tiles a token streams (every region once)."""
        return sum(r.nbytes for r in self.regions)

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    def _native_blocks(self, r: Region, b: Block) -> list[tuple]:
        """A block for ``_native.pack_tiles``: its runs of one format (``Block.parts``); a ``half`` region's by their
        first tile in a column's block, a ``pt`` block's in pairs (``Region``)."""
        out = []
        for fmt, R, rt, names, rows in b.parts(self.tensor(b.tensors[0]).rows):
            ts = [self.tensor(t) for t in names]
            blk = (_native_format(fmt, ts), R, self.tile_bytes, self.nsb, r.n_kc, r.kc_group, r.n_rt,
                   r.rt0[b.name] + rt, r.offset, r.col_stride, [_raw_bytes(t) for t in ts], np.ascontiguousarray(rows),
                   ts[0].cols // 256, b.G)
            if r.half:
                blk += (r.t0[b.name] + rt * r.n_kc, int(b.pt), r.n_t)
            out.append(blk)
        return out

    def pack(self, out: np.ndarray, *, threads: int = 0) -> bool:
        """Pack every region into ``out`` (uint8, nbytes; the PAD bytes are left as they are): the C++ packer
        (``_native.pack_tiles``: threads over items of 2048 rows, non-temporal stores; the file's tensors through views
        of its own, read ahead and unmapped tensor by tensor: the process never holds more of the file than the packer's
        window), then the data regions. Returns whether bytes went through the CPU caches (the data regions: flush a
        device buffer)."""
        assert out.dtype == np.uint8 and out.size >= self.nbytes
        blocks = [(r, b) for r in self.regions + self.slabs for b in r.blocks]
        paro = [x for x in blocks if any(isinstance(self.tensor(t), ParoTensor) for t in x[1].tensors)]
        # (a half-precision head's Q4_0 plane: rows quantized here, ``decode.screen``)
        paro += [x for x in blocks if any(derived(x[1].fmt, self.qtype(t)) for t in x[1].tensors)]
        own = [x for x in blocks if set(x[1].tensors) & set(self._own) and not _member(x, paro)]  # (tensors in memory)
        mapped = [x for x in blocks if not _member(x, own) and not _member(x, paro)]
        _pack_tiles(out, [x for r, b in mapped for x in self._native_blocks(r, b)], threads=threads,
                    files=self.gguf.files)
        if own:
            _pack_tiles(out, [x for r, b in own for x in self._native_blocks(r, b)], threads=threads)
        for r, b in paro:  # (a block of PARO linears: transcoded into rows of blocks, packed, freed, block by block; the
            # file's pages they were read from given back)
            names = [t for t in b.tensors if t not in self._own]
            self._own.update({t: self.paro_rows(t) for t in names})
            _pack_tiles(out, self._native_blocks(r, b), threads=threads)
            for t in names:
                del self._own[t]
                self.gguf.release(t)
        self._fill_data(out)
        return any(r.data for r in self.regions)

    def paro_rows(self, name: str) -> QTensor:
        """A PARO linear's rows of blocks for the packer (``paro.ParoTensor.blocks``), its K in the storage order of its
        rotation's tan form within each group (``paro.TanForm.chan``: the decode and the prefill hand the GEMMs x' in
        that order); a half-precision head's Q4_0 plane (``decode.screen.plane_rows``)."""
        t = self.tensor(name)
        if not isinstance(t, ParoTensor):
            assert t.qtype in HALF, (name, t.qtype)
            return QTensor(name=name, qtype="Q4_0", rows=t.rows, cols=t.cols, raw=plane_rows(t))
        b = t.blocks()
        if n := self._paro_views.get(name):
            return QTensor(name=name, qtype=b.qtype, rows=b.rows * n, cols=b.cols // n, raw=b.raw.reshape(b.rows * n, -1))
        return b

    def repack(self, out: np.ndarray, r: Region, b: Block, col: int, rt: int) -> list[tuple[int, int]]:
        """Pack row tile rt of block b in column ``col`` again (its ``rows`` changed) into out (``pack``'s buffer), with
        the other row tile of its pair in a ``pt`` block (their shared tail tile): the (offset, bytes) of the tiles (a
        half-precision head's Q4_0 plane: those rows quantized again)."""
        assert not b.fmt2, "a split block's row tiles are of two formats"
        n = b.R * b.G
        lo = rt - rt % 2 if b.pt else rt
        hi = min(lo + 2, b.tiles) if b.pt else rt + 1
        rows = np.ascontiguousarray(b.rows[col, lo * n:hi * n])
        name = b.tensors[0]
        plane = derived(b.fmt, self.qtype(name))
        if plane:  # (the rows' plane alone, the rows renumbered into it)
            t = self.tensor(name)
            ids, rows = np.unique(rows[rows >= 0]), rows.copy()
            rows[rows >= 0] = np.searchsorted(ids, rows[rows >= 0])
            self._own[name] = QTensor(name=name, qtype="Q4_0", rows=len(ids), cols=t.cols,
                                      raw=plane_rows(replace(t, rows=len(ids), raw=np.asarray(t.raw)[ids])))
        try:
            blk = list(self._native_blocks(r, b)[0])
        finally:
            if plane:
                del self._own[name]
        blk[7] += lo
        blk[8] += col * r.col_stride
        blk[11] = rows.reshape(1, -1)
        if r.half:
            blk[14] = r.tile(r.rt0[b.name] + lo, 0)
        _pack_tiles(out, [tuple(blk)], threads=1)
        tiles = sorted({r.tile(r.rt0[b.name] + t, k) for t in range(lo, hi) for k in range(r.n_kc)})
        return [(r.offset + col * r.col_stride + i * r.tile_bytes, r.tile_bytes) for i in tiles]


# ----------------------------------------------------------------------
# Tiles of row groups
# ----------------------------------------------------------------------


class NormedImage(TileImage):
    """A column-major image of an attention model (``llama.image``, ``lfm.image``, ``gemma4.image``): the model's config
    (``cfg``), the layers it lays out (``layers``: default all) and each normed phase's norm-weight objects (a data region
    "<phase>_norm" before its tiles: ``norm_objects``). A model supplies ``norms(L, name)``: the tensor names (w1, w_post,
    ...) of a normed phase; w_post None: a plain normed input (objects of [w1 4096]), else a header of the post-norm
    residual update (``protocol.F_POST``: [w1 2048 | w_post 2048] per 8 input objects); w1 None: ones."""

    def __init__(self, gguf: GGUFFile, cfg: Any, *, layers: list[int] | None, **image: Any):
        self.cfg = cfg
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        super().__init__(gguf, column_major=True, **image)

    def norms(self, L: int, name: str) -> tuple[str | None, ...]:
        """The tensor names (w1, w_post, ...) of the normed phase ``name`` of layer L (-1: the LM head)."""
        raise NotImplementedError

    def n_norm(self, L: int, name: str) -> int:
        """Norm-weight objects of a phase: [w1 4096] (a plain input) or [w1 2048 | w_post 2048] (a header)."""
        per = NX_PER_NORM * (OBJ_K if self.norms(L, name)[1] is None else POST_K)
        return -(-self.cfg.hidden_size // per)

    def _norm(self, L: int, name: str) -> None:
        self._add_data(L, name + "_norm", self.n_norm(L, name) * self.tile_bytes)

    def _vec(self, name: str | None) -> np.ndarray:
        """A norm weight (hidden,) fp32 (None: ones)."""
        if name is None:
            return np.ones(self.cfg.hidden_size, np.float32)
        return np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's norm-weight objects (bf16): [w1] or per object [w1 | w_post]."""
        w1, wp = self.norms(layer, name)[:2]
        H, n, T = self.cfg.hidden_size, self.n_norm(layer, name), self.tile_bytes
        vecs = [self._vec(w1)] + ([] if wp is None else [self._vec(wp)])
        parts = np.zeros((len(vecs), n * NX_PER_NORM * (OBJ_K if wp is None else POST_K)), np.float32)
        for i, v in enumerate(vecs):
            parts[i, :H] = v
        objs = parts.reshape(len(vecs), n, -1).transpose(1, 0, 2).reshape(n, -1)  # [object][w1 | w_post]
        b = np.ascontiguousarray(objs).astype(bfloat16).view(np.uint8)
        out = np.zeros((n, T), np.uint8)
        out[:, :b.shape[1]] = b
        return out


def _member(x: tuple, xs: list[tuple]) -> bool:
    """Whether the (region, block) x is one of xs (by identity)."""
    return any(x[0] is r and x[1] is b for r, b in xs)


def _native_format(fmt: str, tensors: list[QTensor]) -> str:
    """A tile format's name for ``_native.pack_tiles`` (BFP16: by its tensors' type, "BFP16/BF16", "BFP16/F16" or
    "BFP16/F32")."""
    return f"{fmt}/{tensors[0].qtype}" if fmt == "BFP16" else fmt


def _raw_bytes(t: QTensor) -> np.ndarray:
    """A tensor's rows as bytes (BF16 / F16: (rows, 2 cols), F32: (rows, 4 cols))."""
    return t.raw if t.raw.dtype == np.uint8 else t.raw.view(np.uint8)


def pack_rows(tensors: list[QTensor], rows: np.ndarray, fmt: str, R: int, S: int, nch: int, *, kc: int = 1024,
              G: int = 1) -> np.ndarray:
    """Tiles (n / (R G), nch, S) of ``rows`` (-1: zero rows; n a multiple of R G; F32 tensors: BFP16 planes,
    ``Block``) of the stacked tensors over nch K chunks of kc (``_native.pack_tiles``; columns past the tensors' as
    zero super-blocks): tile (t, k) = ``pack_tile(fmt, fields of rows t R .. of chunk k, S)``; G > 1 (one chunk):
    tile row v = rows v G .. v G + G - 1 along K."""
    t = tensors[0]
    assert packs_from(fmt, t.qtype) and t.cols * G <= nch * kc
    assert rows.size % (R * G) == 0
    n = rows.size // (R * G)
    out = np.zeros(n * nch * S, np.uint8)
    _pack_tiles(out, [(_native_format(fmt, tensors), R, S, kc // 256, nch, nch, n, 0, 0, out.size,
                              [_raw_bytes(x) for x in tensors], np.ascontiguousarray(rows, np.int64).reshape(1, -1),
                              t.cols // 256, G)],
                       threads=1)
    return out.reshape(n, nch, S)


def column_rows(n: int) -> np.ndarray:
    """(COLS, n / COLS) rows: column c's are n / COLS c .. (an output projection's, the LM head's)."""
    per = n // COLS
    return per * np.arange(COLS)[:, None] + np.arange(per)[None]


def gate_up_rows(ffn: int, R: int, cols: int = COLS) -> np.ndarray:
    """(cols, n) rows of the stack [gate; up] (ffn rows each) for the gate/up phase: column c's outputs ffn / cols c ..
    in groups of 64, each as 8 x [8 gate | 8 up] rows (128) in ``gu_tiles(R)`` tiles (-1: padding rows). (An MoE's
    experts: cols = the expert halves, ``Region.cols``.)"""
    fr = ffn // cols  # gate (and up) rows per column
    gu = np.full((cols, fr // 64, gu_tiles(R) * R), -1, np.int64)
    g0 = fr * np.arange(cols)[:, None] + 64 * np.arange(fr // 64)  # (cols, groups): a group's first gate row
    gu[:, :, :128] = (g0[..., None, None, None] + 8 * np.arange(8)[:, None, None] + ffn * np.arange(2)[:, None]
                      + np.arange(8)).reshape(cols, fr // 64, 128)  # 8 x [8 gate | 8 up]
    return gu.reshape(cols, -1)


SPLIT_ROWS = 64  # (a split gate / up block) rows of each part of a SiLU group: its 64 outputs' gate rows, then their up rows
GU_REGIONS = ("gate_up",)  # the regions whose gate / up block may be split (``Block.fmt2``: dense layers')


def split_rows(ffn: int, ra: int, rb: int, cols: int = COLS) -> np.ndarray:
    """(cols, n) rows of the stack [gate; up] (ffn rows each) of a split gate / up block (``Block.fmt2``): column c's
    outputs ffn / cols c .. in groups of 64, each as the 64 gate rows in ceil(64 / ra) tiles of ra rows, then the 64 up
    rows in ceil(64 / rb) tiles of rb rows (-1: padding rows)."""
    fr = ffn // cols
    na, nb = -(-SPLIT_ROWS // ra) * ra, -(-SPLIT_ROWS // rb) * rb
    out = np.full((cols, fr // 64, na + nb), -1, np.int64)
    g0 = fr * np.arange(cols)[:, None] + 64 * np.arange(fr // 64)  # (cols, groups): a group's first gate row
    out[:, :, :64] = g0[..., None] + np.arange(64)
    out[:, :, na:na + 64] = g0[..., None] + ffn + np.arange(64)
    return out.reshape(cols, -1)


def group_rows(rows: np.ndarray, R: int, G: int) -> np.ndarray:
    """(cols, n) rows of a matrix of K = kc / G as row-group tiles (``Block.G``): (cols, t R G) - tile i's row r holds
    rows[:, G R i + R j + r] at slot j < G (its K columns j K ..), -1 past n: a tile's G R outputs in their order,
    slot-major."""
    cols, n = rows.shape
    t = -(-n // (R * G))
    i, r, j = np.meshgrid(np.arange(t), np.arange(R), np.arange(G), indexing="ij")
    src = (G * R * i + R * j + r).reshape(-1)  # (the entries in tile, row, slot order)
    return np.ascontiguousarray(np.concatenate([rows, np.full((cols, 1), -1, np.int64)], 1)[:, np.minimum(src, n)])
