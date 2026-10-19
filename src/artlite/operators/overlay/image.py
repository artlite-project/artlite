"""The weight image of a decode overlay: GGUF matrices re-laid-out into the tile formats of ``operators.formats``,
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
its tiles' columns past the matrix are zero super-blocks. The image starts and ends with ``PAD`` bytes: a GEMM reads a Q4_K header unit from 1 KB
before a tile, and past a region's last tile.
"""

from dataclasses import dataclass, field

import numpy as np

from ... import _native
from ...gguf_model import GGUFFile, QTensor
from ..design import COLS
from ..formats import FORMATS, choose_tile_bytes, gu_tiles, has_tile_format, tile_format

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


@dataclass(frozen=True)
class Block:
    """A run of row tiles in every column's block of a region: ``rows`` (COLS, n) indices into the stack of the
    ``tensors`` (their rows one after the other; -1: a zero padding row), n a multiple of R."""

    name: str
    tensors: tuple[str, ...]
    rows: np.ndarray
    fmt: str
    R: int

    @property
    def tiles(self) -> int:
        return self.rows.shape[1] // self.R


@dataclass
class Region:
    """The tiles of one phase: 8 column blocks of ``col_bytes`` from ``offset`` on."""

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

    def __post_init__(self) -> None:
        assert self.n_kc % self.kc_group == 0
        rt = 0
        for b in self.blocks:
            self.rt0[b.name] = rt
            rt += b.tiles
        self.n_rt = rt

    @property
    def K(self) -> int:
        return self.n_kc * self.kc

    @property
    def col_bytes(self) -> int:
        return self.data or self.n_rt * self.n_kc * self.tile_bytes

    @property
    def nbytes(self) -> int:
        return COLS * self.col_bytes

    def tile(self, rt: int, kc: int) -> int:
        """Index in a column block of row tile rt's tile of K chunk kc."""
        g = self.kc_group
        return (kc // g * self.n_rt + rt) * g + kc % g

    def block(self, name: str) -> Block:
        return next(b for b in self.blocks if b.name == name)


def _pad_rows(rows: np.ndarray, R: int) -> np.ndarray:
    """(COLS, n) rows -> (COLS, ceil(n / R) R), padded with -1."""
    n = -(-rows.shape[1] // R) * R
    out = np.full((rows.shape[0], n), -1, np.int64)
    out[:, :rows.shape[1]] = rows
    return out


class TileImage:
    """A weight image of a GGUF file: ``regions`` (``region(layer, name)``), ``nbytes``; ``pack(out)`` fills a buffer of
    nbytes. A model's subclass adds its regions (``_add``) and calls ``_place``."""

    def __init__(self, gguf: GGUFFile, *, kc: int, payload: int, tile_bytes: int, vocab: int,
                 max_rows: dict[str, int] | None = None, column_major: bool = False):
        """kc: the K chunk of a tile (columns); payload: the elements of an activation object's payload (the K groups);
        tile_bytes: the object size (0: chosen from the file's formats, ``choose_tile_bytes``); max_rows: the rows of
        a format's tile at most (default: the format's); column_major: the layout (module docstring)."""
        self.gguf, self.kc, self.nsb, self.payload, self.vocab = gguf, kc, kc // 256, payload, vocab
        self.tile_bytes = tile_bytes or self._choose_tile_bytes()
        mr = max_rows or {}
        self.R = {f: FORMATS[f].rows(self.tile_bytes, self.nsb, mr.get(f, 0)) for f in FORMATS
                  if FORMATS[f].rows(self.tile_bytes, self.nsb, mr.get(f, 0))}
        self.column_major = column_major
        self.regions: list[Region] = []
        self._own: dict[str, QTensor] = {}  # (tensors a model substitutes: re-laid-out copies in memory)

    def tensor(self, name: str) -> QTensor:
        """A tensor the blocks name (the file's, or the model's substitute: ``_own``)."""
        return self._own[name] if name in self._own else self.gguf.tensor(name)

    def fmt(self, name: str) -> str:
        return tile_format(self.gguf.tensor_info[name][1])

    def _choose_tile_bytes(self) -> int:
        """The object size of the formats in the file (``formats.choose_tile_bytes``: rows streamed per format and K
        chunk, ffn_gate / ffn_up as the gate/up phase's SiLU groups, ``gate_up_rows``; token_embd only as the tied LM
        head)."""
        rows: dict[str, int] = {}
        gu: dict[str, int] = {}
        tied = "output.weight" not in self.gguf
        for name, (dims, qtype, _, _) in self.gguf.tensor_info.items():
            if len(dims) == 2 and dims[0] % self.kc == 0 and (tied or name != "token_embd.weight") and has_tile_format(qtype):
                f, n_kc = tile_format(qtype), dims[0] // self.kc
                if name.endswith("ffn_gate.weight"):
                    gu[f] = gu.get(f, 0) + dims[1] // 64 * n_kc
                elif not name.endswith("ffn_up.weight"):
                    rows[f] = rows.get(f, 0) + dims[1] * n_kc
        return choose_tile_bytes(rows, self.nsb, gu)

    def _add(self, layer: int, name: str, parts: list[tuple[str, tuple[str, ...], np.ndarray]], K: int) -> None:
        """A region of the blocks ``parts`` = (name, tensors, rows (COLS, n)) over an input of K elements."""
        blocks = []
        for bname, tensors, rows in parts:
            fmts = {self.fmt(t) for t in tensors}
            assert len(fmts) == 1, f"{tensors}: one tile format per block ({fmts})"
            fmt = fmts.pop()
            R = self.R[fmt]
            blocks.append(Block(bname, tensors, _pad_rows(rows, R), fmt, R))
        n_kc = -(-K // self.kc)
        self.regions.append(Region(layer, name, blocks, n_kc, kc_group(n_kc, self.payload, self.kc), self.tile_bytes,
                                   self.kc))

    def _add_data(self, layer: int, name: str, nbytes: int) -> None:
        """A data region of nbytes per column ("<phase>_norm": ``norm_objects``, written at pack time)."""
        self.regions.append(Region(layer, name, [], 1, 1, self.tile_bytes, self.kc, data=nbytes))

    def _place(self) -> None:
        """The regions' offsets (after the leading PAD) and column strides, ``nbytes``."""
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
        self.nbytes = -(-(end + PAD) // 4096) * 4096
        self._index = {(r.layer, r.name): r for r in self.regions}

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
        """The norm-weight objects of phase ``name`` (uint8, the bytes of its data region "<name>_norm"; the model's)."""
        raise NotImplementedError

    def _fill_data(self, out: np.ndarray) -> None:
        """Write the data regions: every column's copy of a phase's ``norm_objects``."""
        for r in self.regions:
            if r.data:
                objs = self.norm_objects(r.layer, r.name.removesuffix("_norm")).reshape(-1)
                for c in range(COLS):
                    out[r.offset + c * r.col_stride:][:objs.size] = objs

    def region(self, layer: int, name: str) -> Region:
        return self._index[(layer, name)]

    def streamed_bytes(self) -> int:
        """Bytes of tiles a token streams (every region once)."""
        return sum(r.nbytes for r in self.regions)

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    def _native_block(self, r: Region, b: Block) -> tuple:
        """A block for ``_native.pack_tiles``."""
        ts = [self.tensor(t) for t in b.tensors]
        return (_native_format(b.fmt, ts), b.R, self.tile_bytes, self.nsb, r.n_kc, r.kc_group, r.n_rt, r.rt0[b.name],
                r.offset, r.col_stride, [_raw_bytes(t) for t in ts], b.rows, ts[0].cols // 256)

    def pack(self, out: np.ndarray, *, threads: int = 0, release: bool = True) -> bool:
        """Pack every region into ``out`` (uint8, nbytes; the PAD bytes are left as they are): the C++ packer
        (``_native.pack_tiles``: threads over items of 2048 rows, non-temporal stores, the file read ahead), then the
        data regions. ``release``: give back the GGUF file's pages of each packed tensor (they stay in the page
        cache). Returns whether bytes went through the CPU caches (the data regions: flush a device buffer)."""
        assert out.dtype == np.uint8 and out.size >= self.nbytes
        blocks = [(r, b) for r in self.regions for b in r.blocks]
        own = [x for x in blocks if set(x[1].tensors) & set(self._own)]  # (memory of their own: never released)
        mapped = [x for x in blocks if x not in own]
        _native.pack_tiles(out, [self._native_block(r, b) for r, b in mapped], threads=threads, release=release,
                           file=(self.gguf.path, self.gguf.data))
        if own:
            _native.pack_tiles(out, [self._native_block(r, b) for r, b in own], threads=threads)
        self._fill_data(out)
        return any(r.data for r in self.regions)


# ----------------------------------------------------------------------
# Tiles of row groups
# ----------------------------------------------------------------------


def _native_format(fmt: str, tensors: list[QTensor]) -> str:
    """A tile format's name for ``_native.pack_tiles`` (BFP16: by its tensors' type, "BFP16/BF16" or "BFP16/F16")."""
    return f"{fmt}/{tensors[0].qtype}" if fmt == "BFP16" else fmt


def _raw_bytes(t: QTensor) -> np.ndarray:
    """A tensor's rows as bytes (BF16 / F16: (rows, 2 cols))."""
    return t.raw if t.raw.dtype == np.uint8 else t.raw.view(np.uint8)


def pack_rows(tensors: list[QTensor], rows: np.ndarray, fmt: str, R: int, S: int, nch: int, *, kc: int = 1024
              ) -> np.ndarray:
    """Tiles (n / R, nch, S) of ``rows`` (-1: zero rows; n a multiple of R) of the stacked tensors over nch K
    chunks of kc (``_native.pack_tiles``; columns past the tensors' as zero super-blocks): tile (g, k) =
    ``pack_tile(fmt, fields of rows g R .. of chunk k, S)``."""
    t = tensors[0]
    assert tile_format(t.qtype) == fmt and t.cols <= nch * kc and rows.size % R == 0
    G = rows.size // R
    out = np.zeros(G * nch * S, np.uint8)
    _native.pack_tiles(out, [(_native_format(fmt, tensors), R, S, kc // 256, nch, nch, G, 0, 0, out.size,
                              [_raw_bytes(x) for x in tensors], np.ascontiguousarray(rows, np.int64).reshape(1, -1),
                              t.cols // 256)],
                       threads=1)
    return out.reshape(G, nch, S)


def column_rows(n: int) -> np.ndarray:
    """(COLS, n / COLS) rows: column c's are n / COLS c .. (an output projection's, the LM head's)."""
    per = n // COLS
    return per * np.arange(COLS)[:, None] + np.arange(per)[None]


def gate_up_rows(ffn: int, R: int) -> np.ndarray:
    """(COLS, n) rows of the stack [gate; up] (ffn rows each) for the gate/up phase: column c's outputs ffn / 8 c ..
    in groups of 64, each as 8 x [8 gate | 8 up] rows (128) in ``gu_tiles(R)`` tiles (-1: padding rows)."""
    fr = ffn // COLS  # gate (and up) rows per column
    gu = np.full((COLS, fr // 64, gu_tiles(R) * R), -1, np.int64)
    g0 = fr * np.arange(COLS)[:, None] + 64 * np.arange(fr // 64)  # (COLS, groups): a group's first gate row
    gu[:, :, :128] = (g0[..., None, None, None] + 8 * np.arange(8)[:, None, None] + ffn * np.arange(2)[:, None]
                      + np.arange(8)).reshape(COLS, fr // 64, 128)  # 8 x [8 gate | 8 up]
    return gu.reshape(COLS, -1)
