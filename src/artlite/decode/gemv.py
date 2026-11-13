"""Decode GEMV on the MMUL for several token slots (the decode overlays' main and helper cores, ``decode.design``):
the tiles of the weight image (``formats``) against bfp16 panels of up to 4 slots' x, with exact integer weight panels
(BFP16 tiles: their own panels). Two cores share a tile: the super-block split (every format's tiles of 1024
columns: one core takes super-blocks 0, 1 (``{p}_mmv<0>``), the other 2, 3 (``{p}_mmv<1>``)) or the row split
(Gemma's 512-column Q4_0 / Q8_0 tiles: each core its half of the rows of every super-block, ``row_split``). A paged
format's kernels (``render(paged=)``) are code pages (``paging``, the memtile transport): a core loads the format's
page for its half before a tile of it unless the page is in, and calls the page's entry (``{p}_pgf<H>_<format>``).

This module is the frame: the shared helpers, the x and sum panels and the dispatcher ``{p}_mmv<H>``. The formats'
kernels and template variables are their slots' (``Slot.mmv_*``, ``sources["mmv/<fmt>"]``: ``formats.q4k`` etc.). The
host model of the arithmetic and a two-core harness (``MmvOperator``) are the tests' (``tests/decode/mmv.py``).

A tile covers a K chunk of ``nsb`` super-blocks (``Chunk``: 256 nsb k).

x panels: per k-block (8 consecutive k) one 8x8 bfp16 panel, rows a = [hi of slot 0 .. 3 | lo of slot
0 .. 3] (hi = bfp16(x), lo = bfp16(x - hi): ~14 bits below each block's maximum), 72 B, in k order (a super-block's
32 panels in a row); a chunk has 32 nsb. Sum panels: the slots' sums of x over 32-k sub-blocks
(per super-block one panel [a][sub-block 8]) and over 16-k ones (two per super-block), as hi + lo rows, in slots of
two panels (``XS_SLOT``): slot j < nsb / 2: the 32-k panels of super-blocks 2 j, 2 j + 1; slot nsb / 2 + sb: the two
16-k panels of super-block sb.

Designs of 8 slots (``render(slots=8)``: every format of the super-block split) take x as single bfp16, rows a = slot
0 .. 7 (the prefill GEMMs' activations; [hi | lo] of 8 slots would be two panels per k block: twice the MMULs and 92 KB
of x panels for a K group of 5120): the same panels and MMULs. Their sums stay exact as two panels each, [hi rows of
the slots] then [lo rows] (the codes' centering subtracts z XS from the codes' sums: XS must be the sum of the
quantized x), in slots of four panels (``xs_slot(8)``) in the same order; c is bf16(C) (64 lanes [slot][r]), y
[g][slot 8][r 8]. A pass that holds 64-lane values across a loop's iterations spills them (Q4_K's offsets: a 1.6 KB
frame, which every core's stack must hold): pop the sums per super-block instead.

A weight panel is 8 rows x 8 k of the tile's integer codes as bfp16 mantissas at exponent 133 (value = code), so the
MMUL sums exact products: C[a][r] = sum_k X[a][k] W[r][k] in fp32. The codes of a row are words of 8 consecutive k
(``formats``' layouts), so 8 rows' words are transposed into panels first (``_tr8``: two shuffle stages per 8x8
words). Per sub-block (32 k, Q6_K, Q2_K, Q3_K: 16 k) the partial sums of the hi and lo rows are added and rounded
once to bf16 (c, 32 lanes [slot 4][r 8]), with the codes centered (q - 8, - 16, - 32: the sum panels against -z e_i
start each chain; Q2_K, Q3_K: x against a constant -z panel), so that c stays small; then y += c (d sc) with d sc
(exact in fp32) as bf16 hi + lo (``_sc``; Q2_K, Q3_K: the integer scales, ``formats.k23``). Each core
keeps its transposed codes, its tables and partial sums in its scratch (``SCR_BYTES``; Q4_K: the codes in place).

Output: y [g 2][slot][r 8] fp32 (rows 8 g ..), accumulated.
"""

from ..formats import FORMATS, kernel_template, slot
from ..formats import SLOTS as FORMAT_SLOTS
from ..formats.common import MMV_SLOTS
from ..operators import paging, templating
from .protocol import B_G4

SLOTS = MMV_SLOTS  # token slots of an x panel (designs of 8: ``render(slots=8)``)
KB = 8  # k per block
PANEL = 72  # bytes of a bfp16 8x8 panel
XS_SLOT = 192  # bytes of a slot of sum panels (two panels, 64-byte aligned: a panel stream must start aligned)
Y_FLOATS = 64  # a tile's output: [g 2][slot 4][r 8] fp32 (tiles of up to 32 rows in the super-block split: [g 4], 128)


def xs_slot(slots: int = SLOTS) -> int:
    """Bytes of a slot of sum panels: two panels of 4 slots' [hi | lo] rows, or four of 8 slots' (hi, lo each)."""
    return XS_SLOT if slots == 4 else 320


def y_floats(slots: int = SLOTS) -> int:
    """A 16-row tile's output: [g 2][slot][r 8] fp32."""
    return 16 * slots


def scr_bytes(slots: int = SLOTS) -> int:
    """A core's scratch (Q6_K's of 8 slots: c [i][h][g][64] bf16 after its 5 KB of codes and tables)."""
    return SCR_BYTES if slots == 4 else 9216
SCR_BYTES = 8192  # a core's scratch: transposed codes, tables, partial sums
RG_FORMATS = ("Q4_0", "Q4_K", "Q6_K", "Q3_K", "IQ3_XXS", "Q8_0")  # (``render(rg=)``) the formats of row-group tiles
NZ_FORMATS = ("Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K")  # the formats whose kernels read the -z e_i panels (``_nz``)


def supported(fmt: str, nsb: int, slots: int = SLOTS) -> bool:
    """The decode GEMV reads tiles of this format over nsb super-blocks for designs of ``slots`` slots (its slot's
    kernels)."""
    return bool(FORMATS[fmt].mmv) and slot(fmt).mmv_supports(fmt, nsb) and slots in slot(fmt).mmv_slots(fmt)


def row_split(fmt: str, nsb: int) -> int:
    """Rows of a row-split tile (both cores take all super-blocks, the second core the second half of the rows), or 0:
    the super-block split (``Slot.mmv_split``)."""
    return slot(fmt).mmv_split(fmt, nsb)


def row_splits(nsb: int) -> dict[str, int]:
    """{format: rows} of the formats read row-split over nsb super-blocks (an image's ``max_rows``)."""
    return {f: row_split(f, nsb) for f in FORMATS if supported(f, nsb) and row_split(f, nsb)}


def helper_y(fmt: str, nsb: int) -> int:
    """Where the second core's rows start in a row-split tile's output (floats: [g][slot][r]; 0: added to the first
    core's, or Q8_0's: its helper writes its row group 1 itself)."""
    return slot(fmt).mmv_helper_y(fmt, nsb)


def tile_rows_of(fmt: str, tile_bytes: int, nsb: int) -> int:
    """Rows of a format's tile in the decode GEMV (a row-split format's are fixed: ``row_split``)."""
    rs = row_split(fmt, nsb)
    r = FORMATS[fmt].rows(tile_bytes, nsb, rs)
    assert not rs or r == rs, (fmt, tile_bytes, nsb, r)
    return r


class Chunk:
    """The K chunk of a tile: ``nsb`` super-blocks, ``k`` columns, the bytes of its x panels (``xp``) and of its sum
    panels (``xs``: nsb / 2 slots of 32-k panels, nsb of 16-k ones) for designs of ``slots`` slots."""

    def __init__(self, nsb: int = 4, slots: int = SLOTS):
        assert nsb % 2 == 0 and slots in (4, 8), (nsb, slots)
        self.nsb, self.k, self.slots = nsb, 256 * nsb, slots
        self.xp = self.k // KB * PANEL
        self.xs = (nsb // 2 + nsb) * xs_slot(slots)


_SRC = templating.source("artlite/decode/gemv.cc.jinja")


_TEMPLATE = kernel_template(_SRC)


def page_entries(p: str, paged: tuple[str, ...], h: int) -> dict[str, tuple[str, ...]]:
    """The pages of core half h (``paging.link``'s ``pages``) of ``render(p, paged=)``: per paged format its entry, in
    page-id order."""
    return {f: (f"{p}_pgf{h}_{f.lower()}",) for f in paged}


def render(p: str, fmts: list[str], *, tile_bytes: int = 9280, nsb: int = 4, paged: tuple[str, ...] = (),
           pg: paging.PageSpec | None = None, rg: bool = False, yb: int = 64, slots: int = SLOTS,
           tab: tuple[tuple[int, int], tuple[int, int]] | None = None) -> str:
    """The kernel source of ``{p}_mmv<H>`` for the tile formats ``fmts`` (tiles of ``tile_bytes`` over nsb
    super-blocks): the frame and the slots' ``mmv/<fmt>`` sources (``Slot.mmv_parts``). ``paged``: formats whose kernels
    are code pages (``pg``: their pager), in page-id order (from 1). ``rg``: the format word's B_G4 tiles of the
    ``RG_FORMATS`` (row groups of 4, an MoE's expert downs: ``decode.image.Block.G``) put super-block 2 H + 1's rows
    ``yb`` floats after 2 H's instead of adding them. ``slots``: the design's token slots (module docstring). ``tab``: the
    gather tables' two copies per core half (default ``formats.TAB``)."""
    assert all(supported(f, nsb, slots) for f in fmts), (fmts, nsb, slots)
    assert all(f in fmts and not row_split(f, nsb) for f in paged) and (pg is not None or not paged), (paged, fmts)
    assert not rg or (nsb == 4 and slots == 4), (rg, nsb, slots)
    assert slots == 4 or not any(row_split(f, nsb) for f in fmts), (fmts, nsb, slots)  # (row-split tiles: 4 slots)
    parts: list[str] = []
    var: dict[str, int] = {}
    for s in FORMAT_SLOTS:
        for f in s.formats:
            if f.name in fmts:
                parts += [x for x in s.mmv_parts(f.name, nsb) if x not in parts]
                var.update(s.mmv_vars(f.name, tile_bytes, nsb))
    if tab is not None:
        var["TAB"] = tab
    pager = paging.source(p, wake=pg.wake, request=True) if paged else ""
    # (keep: every format paged, the inline tables of kernel_lib its pages read kept by resident code; keep_nz: a paged
    # format reads the -z e_i panels and no resident one does: a page writes no data of its own)
    nz = [f for f in fmts if f in NZ_FORMATS]
    return _TEMPLATE.render(p=p, fmts=sorted(fmts), paged=list(paged), pg=pg, pager=pager, parts=parts,
                            keep=bool(paged) and set(paged) == set(fmts),
                            keep_nz=bool(nz) and set(nz) <= set(paged),
                            rg=rg,
                            rg_fmts=RG_FORMATS if rg else (), B_G4=B_G4, yb=yb,
                            rowsplit=any(row_split(f, nsb) for f in fmts),
                            fid={f: FORMATS[f].fid for f in FORMATS}, fn={f: FORMATS[f].mmv for f in FORMATS}, NSB=nsb,
                            PANEL=PANEL, XS_SLOT=xs_slot(slots), slots=slots, L=8 * slots, **var)
