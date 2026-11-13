"""GemmQ's / GemmQd's geometry: what a GEMM's shape, tile format and options make of its passes, tiles and C objects.

``GemmSpec`` is the request (the operators' keyword arguments, documented there); ``Geometry.variants`` derives from it,
as pure host arithmetic (no aie import), every size the data movement (``dataflow``), the L1 buffers and the kernels
are written from: the weight fills' units, the MMUL columns, the C objects' rows, the K chunk groups. A geometry is
frozen: the operator picks the tokens per core row ``m`` (the largest that fits L1, ``with_m``) and, where whole-line
drains are possible, one of the two drain variants, and keeps the chosen geometry as ``op.geo`` (whose fields the
operator reads as its own attributes).
"""

import dataclasses
import operator
from dataclasses import dataclass, replace

from ..formats import FORMATS, KC, slot
from ..operators.design import COLS, MAX_INNER_WORDS
from .pacing import PACE_CLASSES

TILES = 4  # tiles per pass
M_MAX = 64  # GemmQ's tokens per core row and chunk (L1: A and C scale with it)
L1_BUDGET = 62 * 1024  # bytes of core data memory the buffers may use (incl. the 2 KB stack)
B_DEPTH = 2  # L1 weight objects (one k-tile each)
LINE = 32  # C columns of a 64-byte memory line


class L1Error(ValueError):
    """The design's buffers do not fit a core's data memory (e.g. a GemmQd ``m`` too large for the pass)."""


@dataclass(frozen=True)
class PackPlan:
    """A C row's 8-value items (``pack2``) from two slots' MAC rows: per item its segments (slot, window, lanes: a bit
    mask): lane l of the segment's lanes is the slot row's value window + l; ``halves``: some item needs the rows as
    32-lane halves (a window not a whole sub-block row)."""

    items: tuple[tuple[tuple[int, int, int], ...], ...]
    halves: bool


def pack_plan(n: int, n_out: int, n_c: int, tiles: int, R: int) -> PackPlan:
    """The items of a C row of ``n_c`` columns: per tile t [slot 0's R rows | slot 1's], value t R + r of a slot's
    MAC row (n columns); the columns past n_out repeat the last item's segment."""
    assert n_out == 2 * tiles * R and n_c % 8 == 0 and n <= 64
    items = []
    for k in range(n_c // 8):
        segs: list[list[int]] = []
        for lane in range(8):
            o = min(8 * k + lane, n_out - 1)
            t, q = divmod(o, 2 * R)
            s_, r = divmod(q, R)
            w = t * R + r - lane
            if segs and segs[-1][0] == s_ and segs[-1][1] == w:
                segs[-1][3] = lane + 1
            else:
                segs.append([s_, w, lane, lane + 1])
        items.append(tuple((s_, w, (1 << hi) - (1 << lo)) for s_, w, lo, hi in segs))
    halves = any(len(it) > 1 or it[0][1] % 8 or not 0 <= it[0][1] <= n - 8 for it in items)
    return PackPlan(tuple(items), halves)


def line_carry(out_cols: list[list[int]], n_out: int) -> tuple[tuple[int, int], ...] | None:
    """Whole-line C drains of packed passes (``lines``): per pass (cr, sh), its first column's offset in its line (the
    previous pass's last cr columns, which its drain repeats) and where they start in the previous pass's row (0: a
    pass that starts a run of contiguous passes, at a line boundary). None when a run starts inside a line or the
    columns' passes differ (their windows would not be one design's)."""
    out = None
    for cols in out_cols:
        carry = []
        for p, o in enumerate(cols):
            cr = o % LINE if p and o == cols[p - 1] + n_out else 0
            if (o - cr) % LINE:
                return None
            carry.append((cr, (o - cr) - (cols[p - 1] - carry[-1][0]) if cr else 0))
        if out is not None and tuple(carry) != out:
            return None
        out = tuple(carry)
    return out


def two_parts(out_cols: list[list[int]], halves: int) -> bool:
    """Whether every C object of 32-output bfp16 parts can hold two from a 64-output boundary (576-byte panel rows,
    whole lines: ``lines``): a pass's halves, or the passes in pairs (an even count, each pair's outputs contiguous)."""
    for cols in out_cols:
        if halves == 2:
            if any(o % 64 for o in cols):
                return False
        elif len(cols) % 2 or any(cols[p] % 64 or cols[p + 1] != cols[p] + 32 for p in range(0, len(cols), 2)):
            return False
    return True


def line_window(carry: tuple[tuple[int, int], ...], n_out: int) -> int:
    """Columns of every pass's whole-line drain: the most a pass carries and its own columns, in whole lines."""
    return -(-(max(cr for cr, _ in carry) + n_out) // LINE) * LINE


def planar(fmt: str, rows: int, kc: int) -> bool:
    """GemmQdOperator reads these tiles through the planar gather (any row count: the memtile collects a
    k-tile's 32-byte planes, or words of word-major tiles, of the pass's rows, ``TileFormat.planes``): Q5_K, Q6_K,
    Q2_K, Q3_K, and Q4_0 / Q4_1 tiles of other than 16 rows or of 512 columns."""
    return slot(fmt).planar(fmt, rows, kc)


def tile_major(fmt: str, rows: int, kc: int) -> bool:
    """GemmQ and GemmQd read these tiles through the tile-major permutation (16-row blocks of words, MMUL
    panels): 16-row Q4_0 / Q4_1 tiles of 2048 or 1024 columns, 16-row Q4_K, BFP16."""
    return slot(fmt).tile_major(fmt, rows, kc)


@dataclass(frozen=True, kw_only=True)
class GemmSpec:
    """A GEMM's request: the keyword arguments of ``GemmQOperator`` (documented there), hashable (tuples), the key of a
    prefill's one-design-per-distinct-GEMM sharing (``operator_kwargs``: the operator's keywords)."""

    fmt: str = "Q4_0"
    rows: int = 16
    M: int
    K: int
    N: int
    tile_bytes: int
    col_bytes: int
    passes: int
    out_cols: tuple[tuple[int, ...], ...]
    silu: bool = False
    act: str = "silu"
    nrows: int = 4
    c_bfp16: bool = False
    ablate: tuple[str, ...] = ()
    kc: int = KC
    kc_group: int = 0
    rt_stride: int = 0
    tiles: int = 0
    c_pack: bool = False
    region_rt: int = 0
    m: int = 0
    blocks: tuple[tuple[int, int], ...] = ()
    a_cols: tuple[int, int] = (0, 0)
    k_rep: int = 1
    c_pairs: bool = False
    groups: int = 1
    slots: int = 0
    mc_param: str = ""
    mc_unit: int = 0
    lines: bool = False
    pace: int | None = None

    def __post_init__(self) -> None:
        for name in ("out_cols", "blocks", "a_cols", "ablate"):
            value = getattr(self, name)
            object.__setattr__(self, name, tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in value))

    def operator_kwargs(self) -> dict[str, object]:
        """The operator's keyword arguments: the fields that differ from their defaults (the required ones always)."""
        out: dict[str, object] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.default is dataclasses.MISSING or v != f.default:
                out[f.name] = [list(c) for c in v] if f.name == "out_cols" else v
        return out


@dataclass(frozen=True)
class Geometry:
    """What ``GemmSpec`` makes of the passes, tiles and C objects (``variants``); ``m``: the tokens per core row (0:
    not chosen yet, ``with_m``). Sizes in the units of the dataflow's FIFO objects (``dataflow.Dataflow``)."""

    # the request, as the design reads it
    fmt: str
    R: int  # (rows of a tile)
    M: int
    K: int
    N: int
    kc: int
    nrows: int  # core rows used (fewer rows: a smaller program image, cheaper reconfiguration)
    tile_bytes: int
    col_bytes: int
    passes: int
    out_cols: tuple[tuple[int, ...], ...]
    silu: bool  # tiles of 8 gate + 8 up rows -> C = act(gate) * up (N counts outputs)
    act: str  # the gate/up epilogue's activation: silu, or gelu (tanh form, kmath.GELU: Gemma)
    c_bfp16: bool  # bfp16ebs8 C panels: [M / 8][N / 8][72 B] (the gate/up epilogue's: the down GEMM's A)
    ablate: tuple[str, ...]
    k_rep: int
    c_pairs: bool
    G: int  # row groups (``groups``)
    pace: int | None
    mc_param: str
    mc_unit: int
    dq_core: bool  # GemmQd: one dequantizing core per column
    # weights
    planar: bool
    nsb: int  # super-block slices per K chunk
    units: int  # planar tiles of super-block units (``Slot.unit_bytes``): the weight objects are units of the pass's tiles
    slice: int
    hdr_units: int
    hdr_skip: int
    hg: int
    us: int
    dequant: bool  # (BFP16 tiles already hold MMUL operands)
    ws_planes: int
    pair: bool  # GemmQ's dequant: each pair of core rows splits every k-tile (GemmQd: one dequantizing core)
    tq: int  # tiles of a core's k-tile objects (its dequant; GemmQd: the pass's)
    # passes and K chunks
    tiles: int  # decode tiles per pass
    TR: int  # weight rows per pass
    blocks: tuple[tuple[int, int], ...]
    pass_rt: tuple[int, ...]  # first row tile of each pass
    region_rt: int
    n_kc: int
    kc_group: int
    rt_stride: int
    KA: int  # A's columns
    a_k0: int
    a_K: int
    overrun: int
    w_bytes: int
    # MMUL and C objects
    halves: int
    n_read: int  # rows the memtile reads per pass
    n: int  # MMUL columns (planar: the rows padded to a multiple of 16)
    nm: int  # MMUL columns of a D object and a MAC core's C tile
    c_pack: bool
    slots: int
    vp: int  # C objects per pass
    n_out: int
    n_c: int
    carry: tuple[tuple[int, int], ...] | None
    win: int
    c_parts: int
    m: int = 0

    @property
    def mc(self) -> int:
        """Token chunks: M / (nrows m)."""
        return self.M // (self.nrows * self.m)

    def with_m(self, m: int) -> "Geometry":
        return replace(self, m=m)

    @property
    def c_tail(self) -> int:
        """Elements after a packed C object's rows (whole-line windows: the memtile reads the last row's window into
        them, ``lines``)."""
        return self.win - self.n_c if self.carry is not None else 0

    @property
    def hdr_direct(self) -> bool:
        """Planar tiles whose gathered rows hold tile row r's header at row r too (Q5_K's scale planes,
        Q4_0 / Q4_1 headers of 2048 columns; not Q6_K's d chunks or a half-slice Q4 header row pair); tiles of units
        (the dequant takes the rows it converts from the unit)."""
        return bool(self.units) or slot(self.fmt).hdr_direct(self.fmt, self.nsb)

    @property
    def unit_pitch(self) -> int:
        """Bytes of a tile's unit in the dequantizing core's weight object (whole 32-byte lines: aligned loads)."""
        return -(-self.units // 32) * 32

    @property
    def b_l1_words(self) -> int:
        """Words of an L1 weight object (one k-tile; planar: [plane][word 2][rows padded to n]; units: a unit of the
        pass's tiles, [tile][``unit_pitch``])."""
        if self.units:
            return self.tiles * self.unit_pitch // 4
        return 2 * FORMATS[self.fmt].planes * self.n if self.planar else \
            (self.tq if self.pair else self.tiles) * self.slice // 16

    @property
    def fill_units(self) -> tuple[int, int]:
        """(units per tile and K chunk, bf16 per unit) of the weight fills: header planes + super-block
        slices (each as ``us`` sub-units)."""
        hu = 0 if self.hg > 1 else self.hdr_units  # (hg > 1: the header units apart, ``Dataflow.hg_bds``)
        return (hu + self.nsb) * self.us, self.slice // 2 // self.us

    @property
    def hdr_groups(self) -> list[tuple[int, int]]:
        """(first K chunk, chunks) of each header unit: ``hg`` consecutive chunks of one chunk group (``kc_group``) at
        most; with hg = 1 one unit per chunk."""
        g = self.kc_group
        return [(k0, min(self.hg, G + g - k0)) for G in range(0, self.n_kc, g) for k0 in range(G, G + g, self.hg)]

    @property
    def pass_fill_words(self) -> int:
        """bf16 units of a column's weight fills per pass (every K chunk)."""
        nsl, sw = self.fill_units
        hdr = len(self.hdr_groups) * self.slice // 2 if self.hg > 1 else 0
        return self.k_rep * self.tiles * (self.n_kc * nsl * sw + hdr)

    @property
    def hdr_plane0(self) -> int:
        """The first plane of a plane-major header unit with header data (the earlier ones hold the previous
        tile's codes: ``hdr_skip``)."""
        return self.hdr_skip // (32 * self.R) if FORMATS[self.fmt].plane_major else 0

    @property
    def hdr_slot_words(self) -> int:
        """Words of a collected header slot (planes ``hdr_plane0`` .. of a k-tile object)."""
        return 2 * (FORMATS[self.fmt].planes - self.hdr_plane0) * self.n

    @staticmethod
    def variants(spec: GemmSpec, *, dq_core: bool) -> tuple["Geometry", "Geometry | None"]:
        """The geometry of ``spec`` (m not chosen) with plain C drains, and where whole-line drains are possible
        (``lines``: packed passes starting at line boundaries, or two gate/up parts in one C object) the geometry
        with them (None: there is no such variant)."""
        fmt, rows, kc, M, K, nrows, groups = spec.fmt, spec.rows, spec.kc, spec.M, spec.K, spec.nrows, spec.groups
        silu, k_rep, c_pairs = spec.silu, spec.k_rep, spec.c_pairs
        is_planar = planar(fmt, rows, kc)
        assert is_planar or tile_major(fmt, rows, kc), f"no GEMM path reads {fmt} tiles of {rows} rows x {kc}"
        assert not is_planar or dq_core, f"{fmt} tiles of {rows} rows x {kc}: GemmQdOperator"
        assert spec.pace is None or 0 <= spec.pace < len(PACE_CLASSES), spec.pace
        assert M % (nrows * 16) == 0 and K % kc == 0 and kc % 256 == 0 and spec.tile_bytes % 4 == 0
        assert kc in (KC, 1024, 512)
        nsb = kc // 256
        # planar tiles of super-block units (``Slot.unit_bytes``): the weight objects are units of the pass's tiles
        units = slot(fmt).unit_bytes(fmt, rows, nsb) if is_planar else 0
        assert groups == 1 or (dq_core and k_rep == 1 and not silu and kc // 256 % groups == 0), groups
        KA = k_rep * K // groups
        a_k0, a_K = spec.a_cols[0], spec.a_cols[1] or KA
        assert a_k0 % kc == 0 and a_k0 + KA <= a_K
        tiles = spec.tiles or 64 // rows
        TR = tiles * rows
        act = spec.act
        assert act in ("silu", "gelu") and (silu or act == "silu"), act
        assert act == "silu" or dq_core, "the GELU epilogue: GemmQd"
        # rows the memtile reads per pass: a planar SiLU pass of a whole group (128 rows and the padding rows of the
        # group's last tile) drops the padding rows when every part of the tile, header included, maps tile row r to
        # gathered row r (``hdr_direct``); else the dequant skips them
        # whole groups run in two halves of 64 rows (4 [8 gate | 8 up] blocks each): the dequant writes a
        # k-tile's rows as two D objects, a MAC core pops one A for both into two C tiles of 64 columns
        # (L1 for 96 tokens per MAC row; a whole group's C tile does not fit) and drains them as two passes
        halves = 1
        if silu and is_planar:
            gt = -(-128 // rows)  # tiles of a SiLU group (formats.gu_tiles)
            if tiles == gt:
                halves = 2
            else:  # (half groups of whole tiles)
                assert 64 % rows == 0 and TR % 64 == 0, f"planar SiLU passes: a group of {gt} tiles"
        hdr_direct = bool(units) or slot(fmt).hdr_direct(fmt, nsb)
        n_read = 128 if halves == 2 and hdr_direct else TR
        n = -(-n_read // 16) * 16
        assert n == TR or is_planar, (fmt, tiles)
        nm = 64 if halves == 2 else n
        passes, out_cols = spec.passes, spec.out_cols
        blocks = spec.blocks or ((0, passes),)
        assert sum(cnt for _, cnt in blocks) == passes
        pass_rt = tuple(rt0 + tiles * i for rt0, cnt in blocks for i in range(cnt))
        rt_end = max(rt0 + tiles * cnt for rt0, cnt in blocks)  # row tiles read
        region_rt = spec.region_rt or rt_end  # row tiles of a column's region (the chunk group stride)
        n_kc = K // kc
        assert region_rt <= rt_end or (spec.kc_group or n_kc) == 1  # (one chunk per group: any chunk stride)
        # a row tile's K chunks come in groups of kc_group consecutive tiles (row tiles of a group
        # consecutive): the image's groups of a long-K projection (``decode.image.kc_group``: the decode overlay's
        # activation objects)
        kc_group = spec.kc_group or n_kc
        assert n_kc % kc_group == 0
        # tiles from a row tile's chunk group to the next row tile's (default kc_group; more: the GEMM reads the first
        # kc_group chunks of the image's groups of rt_stride, a K part)
        rt_stride = spec.rt_stride or kc_group
        assert rt_stride >= kc_group
        assert len(out_cols) == COLS and all(len(o) == passes for o in out_cols)
        c_bfp16 = spec.c_bfp16
        assert c_bfp16 == silu, "the gate/up epilogue: bfp16 C panels"
        assert not (c_pairs or k_rep > 1) or (fmt == "BFP16" and not silu and not spec.c_pack), \
            "router pairs: BFP16 tiles"
        # (whole 16-row blocks: the memtile un-panelizes the MAC tile; two slots per C object: packed)
        c_pack = spec.c_pack and (TR != n or spec.slots == 2)
        assert not (spec.c_pack and silu), "packed C: not with the gate/up epilogue"
        # C object columns (whole 32-bit words: an odd row count packs one zero column more)
        # row groups: a C object holds ``slots`` slots of the pass's rows (an even count of outputs per tile: Q6_K's 11
        # rows in pairs), as [tile][slot][row] (c_pack) or the MMUL tile (one slot of 16-row tiles)
        slots = spec.slots or (1 if groups == 1 or rows % 2 == 0 else 2)
        vp = groups // slots  # (C objects per pass)
        assert groups % slots == 0 and (groups == 1 or slots == 1 or c_pack), (groups, rows)
        n_out = slots * TR if c_pack and groups > 1 else TR + TR % 2 if c_pack else nm // 2 if silu else n
        assert n_out % 2 == 0, "C rows of whole 32-bit words"
        # packed C object rows: whole 16-byte items, the memtile drops the columns past n_out
        n_c = -(-n_out // 8) * 8 if c_pack else n_out
        # whole-line drains (where they fit): C object rows of a pass's carry and its own columns, drained in windows
        # of ``win`` columns
        N = spec.N
        carry = line_carry(out_cols, n_out) \
            if spec.lines and c_pack and groups == 1 and slots == 1 and N % LINE == 0 else None
        n_c_lines, win = n_c, n_out
        if carry is not None:
            n_c_lines = -(-(max(cr for cr, _ in carry) + n_out) // 8) * 8
            win = line_window(carry, n_out)
        # gate/up (bfp16 C, 32 outputs a pass or half: 288-byte panel rows, half of them ending inside a line): C objects
        # of two such parts, [part][group][4 panels], which the memtile drains as 576-byte rows (``lines``)
        c_parts = 2 if spec.lines and c_bfp16 and n_out == 32 and N % 64 == 0 and two_parts(out_cols, halves) else 1
        slice_bytes = units or FORMATS[fmt].slice_bytes(rows)
        # header units of a slice each (d; d, m for Q4_1; Q4_K records). A header shorter than whole
        # slices (Q4_K of 1024 columns: 1 KB) is read as a slice-sized unit that starts hdr_skip bytes
        # before the tile (its first part belongs to the previous tile): every unit the memtile permutes
        # is one slice of the pass's tiles, at the cost of hdr_skip extra bytes per tile
        hb = 0 if units else FORMATS[fmt].header_bytes(rows, nsb)
        hdr_units = -(-hb // slice_bytes)
        hdr_skip = hdr_units * slice_bytes - hb
        assert hdr_skip == 0 or (fmt in ("Q4_K", "Q4_0") and nsb == 4) or is_planar, (fmt, nsb)
        # Gemma's 512-column Q4_0 tiles of 32 rows (a 1 KB header, a quarter of a slice): a header unit holds the
        # headers of up to ``hg`` = 4 consecutive K chunks of a row tile ([chunk][1 KB], ``hdr_groups``), so the fills
        # read the tiles' bytes once (a slice-sized unit per chunk from hdr_skip before the tile: 1.33x; gate / up at
        # T = 288 has a 0.81 ms DMA floor)
        hg = 4 if is_planar and fmt == "Q4_0" and nsb == 2 and rows == 32 else 1
        if hg > 1:
            assert hb * hg == slice_bytes and hdr_units == 1
            hdr_skip = 0
        # a unit longer than a shim BD's inner dimension (1023 words: Gemma's 32-row Q4_0 slices, 1024) is read as
        # ``us`` sub-units of 16 rows (their rows contiguous): the memtile gathers a pass's unit as [sub-unit][tile]
        # [row 16], the dequant reads its rows in that order and writes the MMUL columns in the tiles' order. A
        # super-block unit (Gemma's 16-row Q8_0, 1088 words) is read in equal parts that the memtile writes back in
        # the unit's order.
        us = 1
        if is_planar and slice_bytes // 4 > MAX_INNER_WORDS:
            f = FORMATS[fmt]
            if units:
                us = next(u for u in range(2, 9)
                          if slice_bytes % (4 * u) == 0 and slice_bytes // u // 4 <= MAX_INNER_WORDS)
            else:
                assert not (f.plane_major or f.word_major) and rows % 16 == 0, (fmt, rows)
                us = rows // 16
        dequant = FORMATS[fmt].dequant
        pair = dequant and not dq_core
        assert not pair or (tiles == TILES and nrows % 2 == 0), "GemmQ dequant: 4 tiles per pass, row pairs"
        tq = TILES // 2 if pair else tiles if dq_core and not is_planar else TILES
        # the W argument: hdr_skip bytes before the matrix, then its columns' regions (and the tiles the
        # last column's extra row tiles read after its region: those of the next region in the image), through the
        # last column's: a column stride longer than the region (a column-major image's) is not read past it there,
        # where the image may end
        nk, g, rt, rs = n_kc, kc_group, region_rt, rt_stride
        last = ((nk - 1) // g) * rt * rs + (rt_end - 1) * rs + (nk - 1) % g + 1 if nk > 1 or rs > 1 else rt_end
        region = (rt * rs * (nk // g) if nk > 1 or rs > 1 else rt) * spec.tile_bytes
        overrun = max(0, last * spec.tile_bytes - region)
        w_bytes = COLS * spec.col_bytes + hdr_skip + overrun - max(0, spec.col_bytes - region)
        assert slice_bytes in (2048, 2304) or is_planar, "super-block slices: 16 x 128 B (Q4), 8 x 288 B (BFP16)"
        plain = Geometry(
            fmt=fmt, R=rows, M=M, K=K, N=N, kc=kc, nrows=nrows, tile_bytes=spec.tile_bytes, col_bytes=spec.col_bytes,
            passes=passes, out_cols=out_cols, silu=silu, act=act, c_bfp16=c_bfp16, ablate=spec.ablate, k_rep=k_rep,
            c_pairs=c_pairs, G=groups, pace=spec.pace, mc_param=spec.mc_param, mc_unit=spec.mc_unit, dq_core=dq_core,
            planar=is_planar, nsb=nsb, units=units, slice=slice_bytes, hdr_units=hdr_units, hdr_skip=hdr_skip, hg=hg,
            us=us, dequant=dequant, ws_planes=FORMATS[fmt].ws_planes, pair=pair, tq=tq, tiles=tiles, TR=TR,
            blocks=blocks, pass_rt=pass_rt, region_rt=region_rt, n_kc=n_kc, kc_group=kc_group, rt_stride=rt_stride,
            KA=KA, a_k0=a_k0, a_K=a_K, overrun=overrun, w_bytes=w_bytes, halves=halves, n_read=n_read, n=n, nm=nm,
            c_pack=c_pack, slots=slots, vp=vp, n_out=n_out, n_c=n_c, carry=None, win=n_out, c_parts=1)
        if carry is None and c_parts == 1:
            return plain, None
        return plain, replace(plain, carry=carry, n_c=n_c_lines, win=win, c_parts=c_parts)


@dataclass(frozen=True)
class L1Plan:
    """The L1 choices of a GemmQd's cores (GemmQ has none: the defaults): the C objects (1: the next pass waits for the
    drain), the SiLU scratch in ``sxp`` parts, the dequantizing core's weight objects, the gate/up epilogue's two C tiles
    in different L1 regions (``pingpong``) and in halves 16 KB apart (``csplit``)."""

    c_depth: int = 2
    sxp: int = 1
    b_depth: int = B_DEPTH
    pingpong: bool = False
    csplit: int = 0


def expose(cls: type, source: type, via: str) -> None:
    """Make the fields and properties of the dataclass ``source`` readable as attributes of ``cls`` (its instances' own
    ``via`` attribute holds the source's instance): an operator reads its geometry as its own attributes."""
    names = [f.name for f in dataclasses.fields(source)] + [n for n, v in vars(source).items() if isinstance(v, property)]
    for name in names:
        assert name not in vars(cls), f"{cls.__name__}.{name}"
        setattr(cls, name, property(operator.attrgetter(f"{via}.{name}")))
