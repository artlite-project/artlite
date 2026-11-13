"""Shared pieces of the tile-format slots: the ``TileFormat`` record, the ``Slot`` interface (the hooks the
operators call), host packing helpers and the kernel-source fragments several slots include."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from ..operators import templating

KC = 2048  # a K chunk of 8 super-blocks (the GEMM operators' default; the images' tiles: 1024 or 512)
SB = 256  # super-block (8 groups of 32)
KT = 64  # the prefill GEMMs' k-tile (a super-block: 4 k-tiles of 2 groups)


@dataclass(frozen=True)
class TileFormat:
    """name: GGUF-style name; fid: id in the overlay's config words; sb_bytes: bytes per row
    and super-block; max_rows / row_step: row counts a tile may have; slice_row: code bytes per
    row of one super-block slice; header_sb: per-row header bytes per super-block (Q6_K: plus a d
    plane of 2 B per row and super-block, padded to 32 B); planes: the codes of a super-block slice as 32-byte planes per row (a row's 8 words of one step:
    Q4 nibbles 4, Q5_K 5, Q6_K 6; word-major Q2_K 2, Q3_K 3: 2 planes words per k-tile; 0: no codes); plane_major:
    the slice is [plane][row][32 B] (else [row][plane][32 B]): the prefill GEMM's memtile gathers either into
    [plane][row] (``gemm_q``);
    fixed_rows: rows of a fixed-size tile (BFP16: 8-row MMUL panels; the object size follows); pad8_row: code bytes per row
    and super-block stored for the rows rounded up to a multiple of 8 (IQ3_XXS: its indices in 8-row panels); unit_align:
    each super-block's slice padded to a multiple of these bytes (``unit_bytes``; IQ3_XXS: a super-block unit holds the
    slice's indices, sign / scale words and d); unit_fixed: the bytes of every super-block's slice whatever the rows
    (IQ3_S: 17 .. 20 rows in 3 row groups, the last group's spare bytes holding other fields); ws_planes: GemmQ's expanded
    scale planes (tile-major formats); dq_row_cycles: GemmQd's planar dequant, cycles per row (MMUL
    column) and k-tile on the dequantizing core (measured; ``gemm_qd.k_tile_cycles``). Methods take
    ``nsb``, the super-blocks of a tile (8: 2048 columns); word_major: the tile is stored [word][row][4 B] (header words, then each
    super-block's code words: a row group's word is one contiguous load), else [row][...] parts; the prefill GEMM's
    memtile gathers a k-tile's words into [word][row]; mmv: the name of the decode GEMV's kernel (``decode.gemv``:
    ``{p}_<mmv><H>`` from the slot's ``mmv/<fmt>`` source; "": the decode does not read these tiles)."""

    name: str
    fid: int
    sb_bytes: int
    max_rows: int
    row_step: int
    slice_row: int
    header_sb: int
    planes: int = 0
    plane_major: bool = False
    fixed_rows: int | None = None
    pad8_row: int = 0
    unit_align: int = 0
    unit_fixed: int = 0
    header_pad32: bool = False
    ws_planes: int = 0
    dq_row_cycles: float | None = None
    word_major: bool = False
    mmv: str = ""

    @property
    def dequant(self) -> bool:
        """Integer codes the GEMMs dequantize (BFP16 tiles already hold MMUL operands)."""
        return self.planes > 0

    def row_bytes(self, nsb: int = 8) -> int:
        return self.sb_bytes * nsb

    def tile_bytes(self, nsb: int = 8) -> int | None:
        """Fixed overlay object size (None: chosen from the model's formats)."""
        return self.fixed_rows * self.row_bytes(nsb) if self.fixed_rows else None

    def header_bytes(self, rows: int, nsb: int = 8) -> int:
        """Bytes of the per-row scale planes at the start of a tile."""
        pad = -(-2 * nsb * rows // 32) * 32 if self.header_pad32 else 0
        return pad + self.header_sb * nsb * rows

    def slice_bytes(self, rows: int) -> int:
        """Bytes of the codes of one super-block (all rows) after the header."""
        return self.slice_row * rows

    def unit_bytes(self, rows: int) -> int:
        """Bytes of one super-block's slice as stored (its padding rows and ``unit_align`` included)."""
        if self.unit_fixed:
            return self.unit_fixed
        n = self.slice_bytes(rows) + self.pad8_row * (-rows % 8)
        return -(-n // self.unit_align) * self.unit_align if self.unit_align else n

    def used_bytes(self, rows: int, nsb: int = 8) -> int:
        return self.header_bytes(rows, nsb) + nsb * self.unit_bytes(rows)

    def rows(self, tile_bytes: int, nsb: int = 8, max_rows: int = 0) -> int:
        """Rows per tile of ``tile_bytes`` (0: does not fit), at most ``max_rows`` (default: the format's)."""
        r = min(max_rows or self.max_rows, tile_bytes // self.row_bytes(nsb))
        r -= r % self.row_step
        while r > 0 and self.used_bytes(r, nsb) > tile_bytes:
            r -= self.row_step
        return r


# kernel declaration of an operator: (name suffix, argument types) -> its ``aie.iron.Kernel`` "{p}_<suffix>"
KernelFn = Callable[[str, list], object]
i32 = np.int32


@dataclass(frozen=True)
class DqKernels:
    """The dequantizing core's kernels of a GemmQd (``Slot.qd_kernels``). Per K chunk its loop calls
    ``hdr(b, hb if collect else ws, [unit,] kt)`` per header unit and k-tile slot, ``chunk(hb, ws)`` once,
    then per super-block ``sb(hb, ws, sb)`` and per k-tile ``dq(b, ws, d, sb, kt, [half,] [ex])`` (GemmQ ``hg`` > 1:
    the header unit once per group of K chunks, ``sb(hb, ws, sb, q)`` with the chunk q in the group); ``ex_ty``:
    the type of a buffer of the core that ``dq`` takes last. The optional kernels and the ``ex`` buffer are
    the Worker's arguments after ``hb``, in this order. ``units`` (``Slot.unit_bytes``): no header kernel (``hdr``
    None); per unit ``sb(b, ws, hb, sb)`` on the weight object, then ``dq`` per k-tile on the same object. ``tables``:
    (type, words) of a lookup table the core holds twice (buffers "tab", "tabb" of ``Slot.qd_scratch``, in two 16 KB
    regions: IQ3_XXS's grid for the gathers), passed to ``dq`` after ``ex``."""

    hdr: object
    dq: object
    collect: bool = False
    unit: bool = True
    sb: object = None
    chunk: object = None
    ex_ty: object = None
    units: bool = False
    tables: tuple[object, np.ndarray] | None = None


class Slot:
    """One family of tile formats: its ``TileFormat`` records, host packing, kernel-source fragments (``sources``:
    Jinja sources by include name; ``"gemm/<fmt>"``: the prefill GEMM's dequant kernels; ``"mmv/<fmt>"``: the decode
    GEMV's kernels) and the hooks the GEMM operators and the decode GEMV call. The defaults are those of a format without the hook's feature. A slot must not import the
    operators (they import the slots)."""

    formats: tuple[TileFormat, ...] = ()
    field = "q"  # the GGUF field (``QTensor.fields``) whose shape gives the rows and columns
    sources: ClassVar[dict[str, str]] = {}

    def format(self, fmt: str) -> TileFormat:
        return next(f for f in self.formats if f.name == fmt)

    # host side

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        """The parts of a tile (``pack_tile``) from the fields of R rows x nsb super-blocks."""
        raise NotImplementedError(fmt)

    # prefill GEMM paths (``gemm_q``)

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        """GemmQd reads these tiles through the planar gather."""
        f = self.format(fmt)
        return f.plane_major or f.word_major

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        """GemmQ and GemmQd read these tiles through the tile-major permutation."""
        return False

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        """GemmQd's planar passes may hold ``tiles`` tiles of ``rows`` rows (the planner's other limits aside)."""
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        """Planar tiles of super-block units (a unit holds one super-block's codes and scales of every row, nothing
        else is in the tile): the unit's bytes, else 0. GemmQd's weight object is then one unit of the pass's tiles,
        each tile's padded to whole 32-byte lines (``unit_pitch``); the dequantizing core calls ``sb(b, ws, hb, sb)``
        once per unit and ``dq`` per k-tile on it (``DqKernels.units``), and reads no header units."""
        return 0

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        """Planar tiles whose gathered rows hold tile row r's header at row r too."""
        return False

    def tile_major_dims(self, op: object) -> tuple[list, list] | None:
        """(memtile write, read dims) of the tile-major permutation; None: gemm_q's 16-row nibble tiles."""
        return None

    def gemm_q_header(self, op: object, kernel: KernelFn, B_l1_ty: object, ws_ty: object,
                      st_ty: object) -> tuple[object, object]:
        """GemmQ's header kernels (hdr, expand): ``hdr(b, ws, plane, kt)`` per header plane and k-tile slot,
        or with an expand kernel ``hdr(b, st, kt)`` collecting the slots in the staging buffer and
        ``expand(st, ws)`` once per K chunk (``gemm_q_collects``)."""
        return kernel("hdr", [B_l1_ty, ws_ty, i32, i32]), None

    gemm_q_collects = False  # GemmQ's header kernels collect the header's k-tile slots in the staging buffer

    # GemmQd's dequantizing core (``gemm_qd``)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        """(count, bytes) of the dequantizing core's buffers besides B and D: the scale workspace ``ws``, the
        header slots ``hb`` (and Q5_K's ``ex``)."""
        return {"ws": (1, max(1024 * op.nsb * op.ws_planes * op.tq // 4, 64)), "hb": (1, KT * op.n * 2)}

    def qd_kernels(self, op: object, kernel: KernelFn, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        """The dequantizing core's kernels (declared in this order)."""
        return DqKernels(hdr=kernel("hdr", [B_l1_ty, ws_ty, i32, i32]), dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty))

    def qd_template_vars(self, op: object) -> dict[str, int]:
        """Extra variables of GemmQd's kernel template."""
        return {}

    # the decode GEMV (``decode.gemv``: two cores per tile)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        """The decode GEMV's kernels take tiles of nsb super-blocks."""
        return False

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        """The designs' token slots the decode GEMV's kernels take (``decode.gemv``: 8, x as single bfp16 rows)."""
        return (4,)

    def mmv_split(self, fmt: str, nsb: int) -> int:
        """Rows of a row-split tile (core H takes rows R / 2 H .. of every super-block), or 0: the super-block split
        (core H takes super-blocks nsb / 2 H .. of every row)."""
        return 0

    def mmv_helper_y(self, fmt: str, nsb: int) -> int:
        """Where the second core's rows go in a row-split tile's output (floats; 0: added to the first core's)."""
        return 0

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        """The sources the decode GEMV includes for a format over nsb super-blocks, in order (shared parts first:
        once per kernel)."""
        return (f"mmv/{fmt}",)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        """Variables of the ``mmv/<fmt>`` source for tiles of tile_bytes over nsb super-blocks."""
        return {}

    def mmv_table(self, fmt: str) -> np.ndarray | None:
        """The decode GEMV's gather table of a format (uint32 words, at most ``TAB_BYTES``: each core half holds two
        copies at ``TAB``), or None."""
        return None


def dq_kernel(op: object, kernel: KernelFn, B_l1_ty: object, ws_ty: object, D_ty: object,
              extra: list | None = None) -> object:
    """GemmQd's ``dq(b, ws, d, sb, kt, [half,] *extra)``."""
    return kernel("dq", [B_l1_ty, ws_ty, D_ty, i32, i32] + [i32] * (op.halves - 1) + (extra or []))


def collected_header(kernel: KernelFn, B_l1_ty: object, ws_ty: object, hb_ty: object,
                     unit: bool = False, chunk: bool = False) -> tuple[object, object]:
    """GemmQd's header kernels of formats whose header slots are collected (``_hcp``, with the unit index if
    ``unit``) and expanded per super-block (``_hdr_sb``, with the K chunk in the unit if ``chunk``: GemmQ ``hg``)."""
    return (kernel("hcp", [B_l1_ty, hb_ty, i32] + ([i32] if unit else [])),
            kernel("hdr_sb", [hb_ty, ws_ty, i32] + ([i32] if chunk else [])))


MMV_SLOTS = 4  # token slots of the decode GEMV's x panels (``decode.gemv``)
# the decode GEMV's gather table: two copies per core half (main core, helper: addresses in their own data memory, in
# two 16 KB regions each, in gaps the decode overlay's buffers leave), each at most TAB_BYTES (``Slot.mmv_table``)
TAB: tuple[tuple[int, int], tuple[int, int]] = ((0x7B400, 0x7F780), (0x70800, 0x7F780))
TAB_BYTES = 2048


# ----------------------------------------------------------------------
# Host-side packing (lossless re-layout of GGUF fields)
# ----------------------------------------------------------------------


def _word_order(nsb: int) -> np.ndarray:
    """k of each position of one packed row of nsb super-blocks (nibble granular)."""
    sb, s, i, j = np.meshgrid(np.arange(nsb), np.arange(4), np.arange(8), np.arange(8), indexing="ij")
    return (256 * sb + 32 * i + 8 * s + j).reshape(-1)


def _nibbles(u: np.ndarray) -> np.ndarray:
    """(R, n) codes in [0, 15] -> (R, n / 2) bytes, low nibble first."""
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def _bf16_bytes(x: np.ndarray) -> np.ndarray:
    """float array (..., n) -> (..., 2n) uint8 of the bf16 values (round to nearest even)."""
    return np.ascontiguousarray(x, dtype=np.float32).astype(bfloat16).view(np.uint8).reshape(*x.shape[:-1], -1)


def sb_major(b: np.ndarray, R: int, n: int) -> np.ndarray:
    """(R, n * m) row bytes -> n slices of (R, m)."""
    return b.reshape(R, n, -1).transpose(1, 0, 2)


# ----------------------------------------------------------------------
# Kernel-source fragments shared by several slots
# ----------------------------------------------------------------------

SOURCES = {
# 16 table entries by 4x16 gathers (IQ4, IQ3_S)
"mmv/g16": templating.source("artlite/formats/common.mmv_g16.cc.jinja"),
# the decode GEMV's scale tables and per-super-block scaling of the formats with a bf16 scale per 32-k group (Q4_0, Q4_1,
# Q8_0)
"mmv/trq": templating.source("artlite/formats/common.mmv_trq.cc.jinja"),
"mmv/dtab": templating.source("artlite/formats/common.mmv_dtab.cc.jinja"),
# the 6-bit scale expansion of Q4_K and Q5_K (``outs``: Q4_K's scattered group stores)
"gemm/plane8": templating.source("artlite/formats/common.gemm_plane8.cc.jinja"),
# GemmQ / GemmQd's dequant of the tile-major Q4 tiles (Q4_0 / Q4_1 of 16 rows x 2048, Q4_K); BFP16 kernels render it
# too, unused
"gemm/dq_tile_major": templating.source("artlite/formats/common.gemm_dq_tile_major.cc.jinja"),
}
