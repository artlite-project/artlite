"""An MoE layer's prefill GEMMs on its decode image (``qwen35.image``'s "moe_in", "sdown" and the expert slabs; GemmQ /
GemmQd, ``prefill.gemm``), model-independent.

- ``router``: the router's rows (BFP16 hi / lo planes per expert, the shared gate's planes after them) against x as
  [hi | lo] panels (``PrefillNormOperator(lo=True)``: the weights' K chunks stream twice, ``k_rep``), fp32 C, each
  expert's [hi, lo] row pair summed (``c_pairs``): per token and column c, row width ``router_pitch`` fp32, the column's
  E / 8 logits from c ``router_pitch`` on, then the shared gate's logit (the second pass's first pair; the rest of
  that pass: the zero rows of the gate's tile and the next region's tiles, never read).
- ``shared_gate_up``: the shared expert's gate / up rows ("sgu", ``overlay.image.gate_up_rows``) on the hi panels, its
  silu(gate) up as bfp16 panels of the shared width.
- ``shared_down``: the shared expert's down ("sdown": row groups of G = kc / width rows per tile row,
  ``overlay.image.group_rows``) on those panels: GemmQd with ``groups`` (per pass G slots over the whole A strip),
  ``ROW_GROUP_TILES`` tiles per pass, column c's outputs from c ``pitch`` on (Q6_K's 11-row tiles: G R = 22 rows per
  tile, the column's last tile padded: a pitch of whole tiles).
"""

from ..design import COLS
from ..gemm.gemm_q import planar
from ..gemm.gemm_qd import GemmQdOperator
from ..prefill import gemm as PG

ROW_GROUP_TILES = 4  # tiles per pass of a row-group GEMM (Q6_K: 44 rows, a 48-column MMUL)

PASS = 64  # rows of a tile-major pass


def router_passes(region: object) -> int:
    """Passes over a column's router and shared-gate tiles."""
    b, s = region.block("router"), region.block("sgate")
    return -(-(b.tiles + s.tiles) * b.R // PASS)


def router_pitch(region: object) -> int:
    """fp32 values per token and column of the router's output (a pass: PASS / 2 pairs)."""
    return router_passes(region) * PASS // 2


CHUNK_PARAM, CHUNK_UNIT = "moe_t", 256  # a whole-chunk GEMM's token chunks (``GemmQOperator.mc_param``): a chunk's
# tokens / CHUNK_UNIT, the cores' own chunks of fewer tokens scaled from it


def router(image: object, region: object, T: int, *, ops: dict | None = None, whole: bool = False) -> PG.Gemm:
    """The router GEMM of T tokens (module docstring); C in bf16 units: per token COLS router_pitch fp32 values.
    ``whole``: the token chunks' count a scratchpad parameter (``chunk_param``: one design for any T)."""
    b = region.block("router")
    assert b.fmt == "BFP16" and region.block("sgate").fmt == "BFP16"
    passes, pitch = router_passes(region), 2 * router_pitch(region)  # (pitch: bf16 units)
    kw = {"fmt": "BFP16", "rows": b.R, "M": T, "K": region.K, "N": COLS * pitch, "tile_bytes": region.tile_bytes,
          "col_bytes": region.col_stride, "passes": passes,
          "out_cols": [[pitch * c + PASS * p for p in range(passes)] for c in range(COLS)], "kc": region.kc,
          "kc_group": region.kc_group, "blocks": ((region.rt0["router"], passes),),
          "region_rt": min(region.n_rt, region.rt0["router"] + passes * PASS // b.R), "k_rep": 2, "c_pairs": True}
    if whole:
        kw.update(mc_param=CHUNK_PARAM, mc_unit=CHUNK_UNIT)
    op = PG._gemm(kw, T, "qkv", ops)
    return PG.Gemm(op, region.offset - op.hdr_skip)


def shared_gate_up(image: object, region: object, T: int, *, width: int, ops: dict | None = None,
                   whole: bool = False) -> PG.Gemm:
    """The shared expert's gate / up GEMM of T tokens on x's hi panels (A of 2 hidden columns: [hi | lo]):
    silu(gate) up as bfp16 panels of ``width`` (the shared expert's) columns, column c's from width / COLS c on.
    ``whole``: as ``router``'s."""
    b = region.block("sgu")
    per = width // COLS  # (outputs per column)
    tp = PASS // b.R  # (tiles per pass: 4 x [8 gate | 8 up] rows, 32 outputs)
    passes = b.tiles // tp
    assert passes * PASS // 2 == per, (passes, per)
    kw = {"fmt": b.fmt, "rows": b.R, "M": T, "K": region.K, "N": width, "tile_bytes": region.tile_bytes,
          "col_bytes": region.col_stride, "passes": passes,
          "out_cols": [[per * c + PASS // 2 * p for p in range(passes)] for c in range(COLS)], "kc": region.kc,
          "kc_group": region.kc_group, "silu": True, "act": "silu", "c_bfp16": True,
          "blocks": ((region.rt0["sgu"], passes),), "region_rt": min(region.n_rt, region.rt0["sgu"] + b.tiles),
          "a_cols": (0, 2 * region.K)}
    if whole:
        kw.update(mc_param=CHUNK_PARAM, mc_unit=CHUNK_UNIT)
    op = PG._gemm(kw, T, "gu", ops)
    return PG.Gemm(op, region.offset - op.hdr_skip)


def shared_pitch(image: object, layers: list[int]) -> int:
    """The shared down's output column pitch over the layers: the widest column of row-group tiles, in whole 32 values
    (the combine's 64-byte loads of a column's block)."""
    out = 0
    for L in layers:
        b = image.region(L, "sdown").blocks[0]
        out = max(out, b.tiles * b.G * b.R)
    return -(-out // 32) * 32


def row_group_gemm(region: object, T: int, *, pitch: int, row: int = 0, ops: dict | None = None, mac_rows: int = 0,
                   m: int = 0, whole: bool = False) -> PG.Gemm:
    """A dense GEMM of T tokens on a region of row-group tiles (one block: the shared down): GemmQd, ``groups`` G,
    passes of ROW_GROUP_TILES tiles, column c's outputs from c pitch on, in rows of ``row`` bf16 (default COLS pitch:
    the columns past them left alone); mac_rows / m: the MAC rows and tokens per MAC row (default: 2 x 64 from T 128 on,
    else 1 x T); ``whole``: as ``router``'s."""
    (b,) = region.blocks
    G, R, tiles = b.G, b.R, ROW_GROUP_TILES
    passes = -(-b.tiles // tiles)
    assert tiles * passes * G * R <= pitch, (pitch, b.tiles)
    if not mac_rows:
        mac_rows, m = (2, 64) if T >= 128 else (1, max(16, T))
    assert not row or row >= COLS * pitch
    kw = {"fmt": b.fmt, "rows": R, "M": T, "K": region.n_kc * region.kc, "N": row or COLS * pitch,
          "tile_bytes": region.tile_bytes, "col_bytes": region.col_stride, "passes": passes,
          "out_cols": [[pitch * c + tiles * G * R * p for p in range(passes)] for c in range(COLS)], "kc": region.kc,
          "kc_group": region.kc_group, "blocks": ((0, passes),), "region_rt": min(region.n_rt, tiles * passes),
          "groups": G}
    if whole:
        kw.update(mc_param=CHUNK_PARAM, mc_unit=CHUNK_UNIT)
    if planar(b.fmt, R, region.kc):
        kw.update(tiles=tiles, c_pack=True)
    key = repr((sorted(kw.items()), mac_rows, m))
    if ops is not None and key in ops:
        op = ops[key]
    else:
        op = GemmQdOperator(mac_rows=mac_rows, m=m, **kw)
        if ops is not None:
            ops[key] = op
    return PG.Gemm(op, region.offset - op.hdr_skip)
