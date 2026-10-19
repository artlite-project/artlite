"""The prefill GEMMs of a model on its decode overlay's weight image (``overlay.image.TileImage``: qwen35's
``QwenImage``, gemma4's ``GemmaImage``): every projection of a T-token chunk, read in place by GemmQ / GemmQd
(``gemm_q``, ``gemm_qd``; the image's tile convention is theirs: K chunks in groups of ``kc_group``, group G of row
tile j at (G n_rt + j) kc_group).

``plan_phase`` turns an image region into its GEMMs (``Gemm``: the operator and the byte offset of its W argument in
the image) and the layout of their output (``Layout``):

- input projections (qwen35 gdn_in: qkv | z | alpha, beta; attn_in: q + gate | k half | v half; gemma4 qkv: q | the
  column's quarter of [K; V], gate): C rows of ``COLS`` column slices of ``pitch`` columns, block b of column c from
  c pitch + off[b] (its valid rows contiguous, the tile and pass padding after them): the attention (and GDN) cores
  of column c read their column's slice;
- output projections (``OUT``: into the residual stream): column c's rows (hidden / 8) at c d_pitch (``d_pitch``: one
  pitch for every layer, the residual add reads it);
- gate_up: act(gate) up as bfp16 panels (the down projection's A; ``act``: silu, or gelu for Gemma), column c's
  outputs (ffn / 8) at their place, in rows of the down projection's K (the ffn padded to whole K chunks: MiniCPM-V's
  3584 in rows of 4096, the padding columns never written: zero).

A run of consecutive blocks of one tile format is one GEMM: Q4_K and BFP16 tile-major (64-row passes, GemmQ or GemmQd),
planar tiles (Q5_K / Q6_K, Gemma's Q4_0 of 32 rows x 512 columns, ...: ``gemm_qd.choose_planar_plan``). Limits of a
GEMM's weight fill: a chained fill holds one shim BD per K chunk and block (16 per shim tile, with the A and C
transfers': ``MAX_CHAIN_BDS``), so runs take at most MAX_CHAIN_BDS / n_kc blocks and a projection of more K chunks (the
27B's down: 17; Gemma's down: 20 in two groups of 10) runs as GEMMs of K parts (whole chunk groups, or ``k_part``
chunks of a group; parts of one width one design, their A arguments shifted: ``Gemm.a_off``), each writing its own C
(``Gemm.part``: the residual add sums them). Parts also keep a long-K projection's A strip resident in the memtile
(``resident_k_part``). A GEMM runs at most 64 passes (the memtile resends a resident A strip once per pass, BDs iterate
at most 64 times: the 27B's gate_up in 68 passes hung the array): the 27B's gate_up runs as two GEMMs.

Every path dequantizes the weights into bfp16 panels (GemmQ: pairs of cores split each k-tile's dequant): a projection
is within ~1.35 2^-9 sqrt(K / 64) of the exact product (relative rms; ``tests/rtc/test_gemm_q.py``).
"""

from dataclasses import dataclass

from ..design import COLS
from ..formats import gu_tiles
from ..gemm.gemm_q import GemmQOperator
from ..gemm.gemm_q import planar as is_planar
from ..gemm.gemm_qd import GemmQdOperator, choose_planar_plan, gemm_mac_rows, planar_gemm
from ..overlay.image import Block, Region, TileImage
from ..util import arg_bytes

MAX_CHAIN_BDS = 14  # shim BDs of a GEMM's chained weight fill (Llama-3.1-8B's down, 14 chunks, the largest built)
ROWS = 64  # rows of a tile-major pass (gemm_q: 4 tiles of 16 rows, 8 BFP16 tiles of 8)
OUT = {"gdn_out", "attn_out", "down", "o", "proj"}  # projections into the residual stream
WHICH = {"gdn_in": "qkv", "attn_in": "qkv", "gdn_out": "o", "attn_out": "o", "gate_up": "gu", "down": "down",
         "qkv": "qkv", "o": "o", "gate": "qkv", "proj": "o"}


@dataclass(frozen=True)
class Layout:
    """C rows of an input projection: column c's slice at c pitch, block b's rows from off[b] on."""

    pitch: int
    off: dict[str, int]

    @property
    def N(self) -> int:
        return COLS * self.pitch


@dataclass
class Gemm:
    op: GemmQOperator
    w_off: int  # the W argument: image bytes w_off .. w_off + op.w_bytes
    part: int = 0  # K part (output projections split over K: each into its own C)
    a_off: int = 0  # the A argument from this bf16 element of the panels on (K parts of one design: the part's columns)


@dataclass
class PhasePlan:
    gemms: list[Gemm]
    layout: Layout | None  # (input projections)
    parts: int = 1


def _gemm(kw: dict[str, object], T: int, which: str, ops: dict[str, GemmQOperator] | None = None) -> GemmQOperator:
    """GemmQ or GemmQd by the tokens and format (``gemm_qd.gemm_mac_rows``); with ``ops``, one object per
    distinct GEMM (the layers of a fused prefill share them: one design each in the ELF)."""
    key = repr((sorted(kw.items()), T, which))
    if ops is not None and key in ops:
        return ops[key]
    planar = is_planar(kw["fmt"], kw["rows"], kw["kc"])
    assert kw["passes"] <= 64, kw["passes"]
    mac_rows = gemm_mac_rows(T, which, kw["fmt"], planar)
    if planar and mac_rows == 3 and T >= 256:
        op = planar_gemm(kw, T)
    else:
        op = GemmQdOperator(mac_rows=mac_rows, **kw) if mac_rows else \
            GemmQOperator(nrows=2 if T <= 64 else 4, **kw)
    if ops is not None:
        ops[key] = op
    return op


def _tp(b: Block) -> int:
    """Tiles of a tile-major pass (``ROWS`` rows) of a block's tiles."""
    assert ROWS % b.R == 0, f"tile-major {b.fmt} tiles of {b.R} rows"
    return ROWS // b.R


def _runs(region: Region) -> list[list[int]]:
    """Runs of consecutive blocks of one tile format (indices), at most MAX_CHAIN_BDS / n_kc blocks each."""
    runs: list[list[int]] = []
    per = max(1, MAX_CHAIN_BDS // region.n_kc)
    for i, b in enumerate(region.blocks):
        if runs and region.blocks[runs[-1][0]].fmt == b.fmt and len(runs[-1]) < per:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _valid(region: Region, i: int) -> int:
    return int((region.blocks[i].rows[0] >= 0).sum())


def _plan(region: Region, run: list[int], T: int, K: int, few: bool) -> object:
    b = region.blocks[run[0]]
    return choose_planar_plan([_valid(region, i) for i in run], b.R, few_passes=few, T=T, fmt=b.fmt, K=K,
                              kc_group=region.kc_group, tile_bytes=region.tile_bytes, kc=region.kc)


def _extents(region: Region, run: list[int], T: int, K: int, few: bool) -> list[int]:
    """Output columns per column of each block of a run (whole passes)."""
    b = region.blocks[run[0]]
    if is_planar(b.fmt, b.R, region.kc):
        return list(_plan(region, run, T, K, few).pitches)
    return [ROWS * -(-region.blocks[i].tiles // _tp(region.blocks[i])) for i in run]


def d_pitch(image: TileImage, T: int) -> int:
    """The output projections' column pitch: the widest column block of any layer."""
    p = 0
    for r in image.regions:
        if r.name in OUT:
            p = max(p, *_extents(r, [0], T, r.K, r.name == "down"))
    return p


def _in_layout(region: Region, T: int, pitch: int = 0) -> Layout:
    """An input projection's output layout: per column [block 0 | block 1 | ...], each from a 32-column boundary,
    at a pitch of at least ``pitch``."""
    off: dict[str, int] = {}
    col = 0
    for run in _runs(region):
        for i, e in zip(run, _extents(region, run, T, region.K, False)):
            off[region.blocks[i].name] = col
            col = -(-(col + e) // 32) * 32
    return Layout(max(-(-col // 64) * 64, pitch), off)


def phase_pitch(image: TileImage, name: str, T: int) -> int:
    """The column pitch of a phase's output in every layer (``plan_phase``'s ``pitch``): an input projection's
    widest layout (its consumers read every layer's rows at one pitch), the output projections' ``d_pitch``."""
    if name in OUT:
        return d_pitch(image, T)
    return max(_in_layout(r, T).pitch for r in image.regions if r.name == name)


def _kw(region: Region, run: list[int], T: int, K: int, N: int, out_cols: list[list[int]], *, few: bool,
        k_part: tuple[int, int] = (0, 0)) -> dict[str, object]:
    """GemmQ arguments of a run of blocks (its passes, the row tiles they read) over K chunks k_part = (first,
    count) (default: all)."""
    b0 = region.blocks[run[0]]
    rt0 = region.rt0[b0.name]
    kw: dict[str, object] = {"fmt": b0.fmt, "rows": b0.R, "M": T, "K": K, "N": N, "tile_bytes": region.tile_bytes,
                             "col_bytes": region.col_stride, "passes": len(out_cols[0]), "out_cols": out_cols,
                             "kc": region.kc, "kc_group": region.kc_group}
    if is_planar(b0.fmt, b0.R, region.kc):
        plan = _plan(region, run, T, region.K, few)
        blocks = tuple((rt0 + r, n) for r, n in plan.blocks)
        kw.update(tiles=plan.tiles, c_pack=True, blocks=blocks)
        rt_end = max(r + plan.tiles * n for r, n in blocks)
    else:
        blocks = tuple((region.rt0[region.blocks[i].name], -(-region.blocks[i].tiles // _tp(region.blocks[i])))
                       for i in run)
        kw.update(blocks=blocks)
        rt_end = max(r + _tp(b0) * n for r, n in blocks)
    # (the chunk group stride; a run that reads past the region: its last column's extra tiles counted)
    kw["region_rt"] = min(region.n_rt, rt_end)
    if k_part[1]:
        kw["a_cols"] = (k_part[0] * region.kc, region.K)
    return kw


def resident_k_part(image: TileImage, region: Region, T: int, *, pitch: int, k_part: int = 0) -> int:
    """K chunks per part (``plan_phase``'s ``k_part``) for an output projection whose A strips do not stay in the
    memtile as planned with ``k_part``: the fewest parts (of whole chunk groups, or within a group) whose strips do, if
    they cost less by ``_part_seconds`` (else ``k_part``)."""
    if region.name not in OUT:
        return k_part
    base = plan_phase(image, region, T, pitch=pitch, k_part=k_part)
    if all(g.op.dataflow().a_res for g in base.gemms):
        return k_part
    g, nk = region.kc_group, region.n_kc
    sizes = [n for n in range(min(nk - 1, MAX_CHAIN_BDS), 0, -1) if n % g == 0] + \
        [d for d in range(g - 1, 0, -1) if g % d == 0]
    best, cost = k_part, _part_seconds(base)
    for n in sorted(sizes, key=lambda n: (-(-nk // n), -n)):  # (the fewest parts first)
        plan = plan_phase(image, region, T, pitch=pitch, k_part=n)
        if all(p.op.dataflow().a_res for p in plan.gemms):
            if (c := _part_seconds(plan)) < cost:
                best, cost = n, c
            break
    return best


def _part_seconds(plan: PhasePlan) -> float:
    """What an output projection's K split costs per call (the rest is the same): its A reads (a strip refetched per
    pass when not resident), each part's C written and read by the residual add, at the DDR read rate (~56 GB/s), and
    ~50 us per part (fitted: the 27B's down in 2 -> 6 Q6_K parts at T = 256)."""
    a = sum(g.op.M * g.op.K * 9 / 8 * (1 if g.op.dataflow().a_res else g.op.passes) for g in plan.gemms)
    c = 2 * plan.parts * max(arg_bytes(g.op, 2) for g in plan.gemms)
    return (a + c) / 56e9 + 50e-6 * plan.parts


def plan_phase(image: TileImage, region: Region, T: int, *, pitch: int = 0, ops: dict[str, GemmQOperator] | None = None,
               act: str = "silu", k_part: int = 0) -> PhasePlan:
    """The GEMMs of a region for T tokens (``pitch``: the output column pitch, ``phase_pitch``; the output
    projections' is required, an input projection's at least its own). ``ops``: the GEMM objects of earlier plans,
    reused for identical GEMMs (and extended). ``act``: gate_up's activation (silu, gelu). ``k_part``: an output
    projection in K parts of that many chunks (dividing a chunk group: each part's A strip may stay in the memtile, and
    each accumulates fewer k-tiles in its bf16 C tile; the parts' sum in the residual add)."""
    name, K, S, KC = region.name, region.K, region.tile_bytes, region.kc
    which = WHICH[name]
    if name == "gate_up":
        (b,) = region.blocks
        if is_planar(b.fmt, b.R, KC):  # (a pass: one group of 8 x [8 gate | 8 up] rows, 64 outputs, in two halves)
            tp, width = gu_tiles(b.R), 64
        else:  # (a pass: 64 rows, 4 x [8 gate | 8 up]: 4 Q4_K tiles, 8 BFP16 tiles)
            tp, width = _tp(b), 32
        passes = b.tiles // tp
        ffn = image.cfg.intermediate_size // COLS
        assert passes * width == ffn
        ng = -(-passes // 64)
        gemms = []
        for i in range(ng):  # (GEMMs of consecutive passes)
            p0, p1 = passes * i // ng, passes * (i + 1) // ng
            oc = [[ffn * c + width * p for p in range(p0, p1)] for c in range(COLS)]
            kw = {"fmt": b.fmt, "rows": b.R, "M": T, "K": K, "N": -(-ffn * COLS // KC) * KC, "tile_bytes": S,
                  "col_bytes": region.col_stride, "passes": p1 - p0, "out_cols": oc, "kc": KC,
                  "kc_group": region.kc_group, "silu": True, "act": act, "c_bfp16": True,
                  "blocks": ((tp * p0, p1 - p0),)}
            if is_planar(b.fmt, b.R, KC):
                kw["tiles"] = tp
            op = _gemm(kw, T, which, ops)
            gemms.append(Gemm(op, region.offset - op.hdr_skip))
        return PhasePlan(gemms, None)
    if name in OUT:
        assert len(region.blocks) == 1 and pitch
        ext = _extents(region, [0], T, K, name == "down")[0]
        assert ext <= pitch
        b = region.blocks[0]
        if is_planar(b.fmt, b.R, KC):
            oc = _plan(region, [0], T, K, name == "down").out_cols([0], [pitch])
        else:
            oc = [[pitch * c + 64 * p for p in range(ext // 64)] for c in range(COLS)]
        nk, g = region.n_kc, region.kc_group
        if k_part and k_part < nk and g % k_part == 0:  # (parts of k_part chunks of the chunk groups: one design)
            assert k_part <= MAX_CHAIN_BDS, k_part
            parts = nk // k_part
            gemms = []
            for i in range(parts):
                k0 = i * k_part
                kw = _kw(region, [0], T, k_part * KC, COLS * pitch, oc, few=name == "down", k_part=(0, k_part))
                kw.update(kc_group=k_part, rt_stride=g)
                op = _gemm(kw, T, which, ops)
                gemms.append(Gemm(op, region.offset + ((k0 // g) * region.n_rt * g + k0 % g) * S - op.hdr_skip,
                                  part=i, a_off=k0 * KC // 8 * 36))
            return PhasePlan(gemms, None, parts)
        if nk <= (k_part or MAX_CHAIN_BDS):
            op = _gemm(_kw(region, [0], T, K, COLS * pitch, oc, few=name == "down"), T, which, ops)
            return PhasePlan([Gemm(op, region.offset - op.hdr_skip)], None)
        # K parts of whole chunk groups (chunks outer: any split), at most k_part chunks each (else what a chained fill
        # holds); parts of one width are one design, their A arguments shifted by the part's columns (one ConfigureOp
        # for them in a fused program)
        assert g <= MAX_CHAIN_BDS, f"a K chunk group of {g} chunks: more than a chained fill holds"
        assert not k_part or k_part % g == 0, (k_part, g)
        groups = nk // g
        parts = -(-groups // (min(k_part or MAX_CHAIN_BDS, MAX_CHAIN_BDS) // g))
        bounds = [g * (groups * i // parts) for i in range(parts + 1)]
        gemms = []
        for i in range(parts):
            k0, k1 = bounds[i], bounds[i + 1]
            kw = _kw(region, [0], T, (k1 - k0) * KC, COLS * pitch, oc, few=name == "down", k_part=(0, k1 - k0))
            op = _gemm(kw, T, which, ops)
            gemms.append(Gemm(op, region.offset + k0 * region.n_rt * S - op.hdr_skip, part=i, a_off=k0 * KC // 8 * 36))
        return PhasePlan(gemms, None, parts)
    lay = _in_layout(region, T, pitch)
    off = lay.off
    gemms = []
    for run in _runs(region):
        b0 = region.blocks[run[0]]
        col0s = [off[region.blocks[i].name] for i in run]
        if is_planar(b0.fmt, b0.R, KC):
            oc = _plan(region, run, T, K, False).out_cols(col0s, [lay.pitch] * len(run))
        else:
            oc = [[lay.pitch * c + o + 64 * p for i, o in zip(run, col0s)
                   for p in range(-(-region.blocks[i].tiles // _tp(region.blocks[i])))] for c in range(COLS)]
        op = _gemm(_kw(region, run, T, K, lay.N, oc, few=False), T, which, ops)
        gemms.append(Gemm(op, region.offset - op.hdr_skip))
    return PhasePlan(gemms, lay)
