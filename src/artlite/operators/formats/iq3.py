"""IQ3_XXS tiles (98 B per row and super-block, the GGUF block's size; R <= 24 rows in 3 row groups of 8): per super-block a
unit of ``unit_bytes`` (``Iq3Slot.unit``: R 23 2320 B, 4 per 1024-column tile, nothing else in the tile) holding

- the grid indices [32-k group i 8][panel pair h 2][8-k panel l' 2][row group g][row 8][2] (the GGUF block's qs: each
  row's two indices of panel l = 2 h + l'; a panel's rows one list of 8 G rows; rows past R zero);
- the sign / scale words [i][row]: byte l = the prefix XOR c of group i's 7-bit sign field l (k 32 i + 8 l ..; c_j =
  f_0 ^ .. ^ f_j: the 8 signs, the 8th the parity of the field's, are c ^ (c << 1)) | bit l of its scale ls at bit 7;
- d [row] bf16;

in the order [indices | sign / scale words | d] (even super-blocks), [d | sign / scale words | indices] (odd ones: the
unit is 16 B off a 32-byte line, so the indices of every super-block start on one).

w = d (2 ls + 1) / 4 q, q = the index's grid value (``gguf_model.iq3xxs_grid``: 4 values per index, in {4 .. 62})
negated where its sign is set.

Decode (``mmv/IQ3_XXS``): a weight panel (8 rows x 8 k) is 16 grid entries read by the gather unit (``load_4x32``: four
lanes per instruction, entry i at byte 8 i of a table whose 32-byte groups of 8 entries are stored twice; lanes 0, 1
of an instruction from one copy, 2, 3 from the other, the copies in two 16 KB regions: in one region a gather waits a
cycle), its signs applied by a masked subtract (``vaddsub8``: the 8 signs of a row one byte of the 64-bit mask): ~6.5
cycles per panel, the move slot's 5 operations and the gathers' memory ports; then the MMUL against the x panels.
A core's half tile (two super-blocks) takes ~2300 cycles. The table (``grid_words``) is a resident buffer of each core
at ``TAB`` (a page brings no data); the panel loop takes its two copies as pointer parameters (llvm-aie gives the
gathers a memory operand only then)."""

from typing import ClassVar

import numpy as np

from . import k23
from .common import DqKernels, Slot, TileFormat, dq_kernel, i32

# the grid table's two copies per core half (main core, helper: addresses in their own data memory, two 16 KB regions
# each, in gaps the decode overlay's buffers leave)
TAB: tuple[tuple[int, int], tuple[int, int]] = ((0x7B800, 0x7F780), (0x77400, 0x7F780))
TAB_BYTES = 2048


def grid_words() -> np.ndarray:
    """A copy of the grid table for the gathers (TAB_BYTES / 4 uint32): entry i (its 4 values, int8) at byte 8 i, i.e.
    per group of 8 entries their 32 bytes twice."""
    from ...gguf_model import iq3xxs_grid

    g = np.ascontiguousarray(iq3xxs_grid()).view(np.uint32).reshape(32, 8)
    return np.concatenate([g, g], 1).reshape(-1)


def ss_words(signs: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """(R, n groups x 4) 7-bit sign fields, (R, n) scales -> (R, n) uint32 words: byte l = the prefix XOR of field l |
    bit l of ls << 7."""
    s = signs.reshape(*ls.shape, 4).astype(np.uint32)
    for k in (1, 2, 4):
        s ^= s << k
    s &= 0x7F
    bits = (ls[..., None].astype(np.uint32) >> np.arange(4, dtype=np.uint32)) & 1
    return ((s | bits << 7) << (8 * np.arange(4, dtype=np.uint32))).sum(-1, dtype=np.uint32)


# the signs and scales of 16 rows of a group (the decode GEMV and the prefill dequant)
_SIGNS = r"""// The signs and scales of 16 rows' sign / scale words v of a group ([r 16][l 4]): m = the signs [l][r 16]
// (bit j of byte r: value 8 l + j of row r negative; m = c ^ (c << 1) of the field c, XOR as (c | t) - (c & t): the
// subtraction a vaddsub (LLVM folds a plain one into an XOR, 5 operations));
// s = (2 ls + 1) / 4 [r 16] (an MMUL: the bytes' bit 7 against a panel of -2^l at 2^-8, from 0.25)
[[clang::always_inline]] inline void {{ p }}_iq3_signs(v64int8 v, v64bfp16ebs8 LX, v64bfp16ebs8 LW, v2int32 all,
                                                      v64int8 &m, v8int32 &s)
{
    const aie::vector<int8, 64> V(v);
    LX.mantissa = (v64int8)aie::bit_and(V, aie::broadcast<int8, 64>(-128));
    const aie::vector<bfloat16, 64> sv = aie::accum<accfloat, 64>(
        mac_8x8_8x8T(LX, 1, LW, 1, (v64accfloat)aie::accum<accfloat, 64>(aie::broadcast<float, 64>(0.25f))))
        .template to_vector<bfloat16>();  // [a 8][n 8]: rows 2 a + n for n < 2
    s = extract_v8int32(::shuffle((v16int32)(v32bfloat16)sv.template extract<32>(0),
                                  (v16int32)(v32bfloat16)sv.template extract<32>(1), T32_8x4_lo), 0);
    const aie::vector<int8, 64> c = aie::bit_and(V, aie::broadcast<int8, 64>(0x7F)), t = aie::add(V, V);
    const v64int8 m8 = __builtin_aie2p_vaddsub8(aie::bit_or(c, t), aie::bit_and(c, t), all);
    m = ::shuffle(m8, m8, T8_16x4);
}

"""

# the decode GEMV (``overlay.gemv``; the header's realignment and the scaling by d: ``mmv/k23``)
_MMV = r"""// ---- IQ3_XXS ({{ RI3 }} rows, 3 row groups; rows past R: junk): per super-block a unit of {{ U3 }} B, [indices [i][h][l'
// 2][g][r 8][2] | sign / scale words [i][row] (byte l: the sign field c l | bit l of ls) | d [row] bf16] (odd super-blocks
// in the reverse order of the three). Core H: super-blocks 2 H, 2 H + 1; per super-block u, in 4 loops (a weight panel
// built and multiplied in one loop leaves too few registers to pipeline the gathers):
// - _iq3_ss: per panel (group i, panel l, row group g) its signs, per group and row the scale (2 ls + 1) / 4;
// - _iq3_w: per panel the gather offsets 8 idx, 4 gathers of the grid, the signs: int8 weight panels;
// - _iq3_x: each x panel against the 3 row groups' weight panels: the groups' sums c, in place of their panels;
// - _iq3_sc: y[g] += hilo(sum_i c_i S_i) d.
// Scratch: w / c [i 8][l 4][g 3][64 B] (0 .. 6144) | mk [i][l 4][g 4][r 8] (6144) | sct [i][g 4][r 8] bf16 (7168) |
// dg [g 4][r 8] bf16 (7936) | the tables' addresses (8000 .. 8008)

""" + _SIGNS + r"""// A super-block's signs and scales (w: its sign / scale words, [i][row] 4 R B per group): per group i rows 0 .. 15 (A)
// and 16 .. 31 (B; 16 .. R - 1: row group 2, the rest junk) of its words (_k23w2); mk [i][l 4][g 4][r 8] (the 16-byte
// rows of A's and B's signs interleaved), sct [i][g 4][r 8]
__attribute__((noinline)) void {{ p }}_iq3_ss(const uint8_t *__restrict w, v16int32 *__restrict mk, bfloat16 *__restrict sct)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    v16int32 lw = broadcast_zero_to_v16int32();  // (panel rows 0, 1: -1, -2, -4, -8 in k 0 .. 3 / 4 .. 7)
    lw = upd_elem(lw, 0, (int)0xF8FCFEFF);
    lw = upd_elem(lw, 3, (int)0xF8FCFEFF);
    v64bfp16ebs8 LW, LX;
    LW.mantissa = (v64int8)lw;
    LW.exponent = al::e8(125);
    LX.exponent = al::e8(0x85);
    const v2int32 all = (v2int32)~0ULL;  // (vaddsub: subtract in every lane)
    uintptr_t o = (uintptr_t)w;
    bfloat16 *__restrict ps = sct;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 8; i++) {
        v16int32 A, B;
        {{ p }}_k23w2(o, {{ 4 * RI3 }}, A, B);
        v64int8 ma, mb;
        v8int32 sa, sb;
        {{ p }}_iq3_signs((v64int8)A, LX, LW, all, ma, sa);
        {{ p }}_iq3_signs((v64int8)B, LX, LW, all, mb, sb);
        mk[0] = ::shuffle((v16int32)ma, (v16int32)mb, T128_2x4_lo);
        mk[1] = ::shuffle((v16int32)ma, (v16int32)mb, T128_2x4_hi);
        *(v8int32 *)ps = sa;
        *(v8int32 *)(ps + 16) = sb;
        o += {{ 4 * RI3 }};
        mk += 2;
        ps += 32;
    }
}

// The weight panels of a super-block (96: wp [i][l][g 3]; ix: its indices [i][h][l' 2][g][r][2]): per (group i, panel
// pair h) and (panel l', row group g) the gather offsets 8 ix (one load per two: upshifted into an accumulator) + the
// copies' bases (lanes 0, 1: tab_ab, 2, 3: tab_cd), 4 gathers per panel, the signs (mk [i][l][g 4][r 8]: the pair's 64
// bytes, one load)
__attribute__((noinline)) void {{ p }}_iq3_w(const uint8_t *__restrict ix, const uint8_t *__restrict mk,
                                             const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd,
                                             uint8_t *__restrict wp)
{
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v32uint8 __aie_dm_resource_a *__restrict pi = (const v32uint8 __aie_dm_resource_a *)ix;
    const v16int32 __aie_dm_resource_b *__restrict pm = (const v16int32 __aie_dm_resource_b *)mk;
    v64int8 __aie_dm_resource_c *__restrict pw = (v64int8 __aie_dm_resource_c *)wp;
    int cw = 0;
    for (int t = 0; t < 16; t++) {  // (pre-RA: II 39; post-RA 44)
        const v16int32 m = *pm++;  // ([l' 2][g 4][r 8])
        const v32acc32 q0 = ups_to_v32acc32(pi[0], 3, 0), q1 = ups_to_v32acc32(pi[1], 3, 0), q2 = ups_to_v32acc32(pi[2], 3, 0);
        pi += 3;
        pw[0] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q0, 0), b16), ext_v2int32(m, 0));  // (l' 0: g 0, 1, 2)
        pw[1] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q0, 1), b16), ext_v2int32(m, 1));
        pw[2] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q1, 0), b16), ext_v2int32(m, 2));
        pw[3] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q1, 1), b16), ext_v2int32(m, 4));  // (l' 1)
        pw[4] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q2, 0), b16), ext_v2int32(m, 5));
        pw[5] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q2, 1), b16), ext_v2int32(m, 6));
        pw = add_2d_ptr(pw, 6, 1, cw, 6);  // (panels l = 2 h, 2 h + 1 of the 3 row groups)
    }
}

// The 3 row groups' 32-k groups of a super-block against xp (its 32 x panels, each popped once): c (i, g) = bf16(C[hi
// rows] + C[lo rows]) of the group's 4 panels, in place of its panel (l 0, g) (w: [i][l][g][64 B], c: 64 B per (i, g)
// at 64 (12 i + g))
__attribute__((noinline)) void {{ p }}_iq3_x(const uint8_t *__restrict xp, const uint8_t *__restrict w,
                                             bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    const int8_t __aie_dm_resource_b *__restrict pw = (const int8_t __aie_dm_resource_b *)w;
    bfloat16 *__restrict pc = c;
    v64bfp16ebs8 W;
    W.exponent = al::e8(0x85);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 8; i++) {
        v64accfloat C0, C1, C2;
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
            const v64bfp16ebs8 X = xs.pop();
            W.mantissa = *(const v64int8 __aie_dm_resource_b *)(pw + 192 * l);
            C0 = l ? mac_8x8_8x8T(X, 1, W, 1, C0) : mul_8x8_8x8T(X, 1, W, 1);
            W.mantissa = *(const v64int8 __aie_dm_resource_b *)(pw + 192 * l + 64);
            C1 = l ? mac_8x8_8x8T(X, 1, W, 1, C1) : mul_8x8_8x8T(X, 1, W, 1);
            W.mantissa = *(const v64int8 __aie_dm_resource_b *)(pw + 192 * l + 128);
            C2 = l ? mac_8x8_8x8T(X, 1, W, 1, C2) : mul_8x8_8x8T(X, 1, W, 1);
        }
        aie::store_v(pc, {{ p }}_sum2(C0));
        aie::store_v(pc + 32, {{ p }}_sum2(C1));
        aie::store_v(pc + 64, {{ p }}_sum2(C2));
        pw += 768;
        pc += 384;
    }
}

// y[g] += hilo(sum over the 8 groups i of c (i, g) sct[i][g][r]) dg[g] for the 3 row groups (c (i, g): 32 bf16 at
// c + 384 i + 32 g)
__attribute__((noinline)) void {{ p }}_iq3_sc(const bfloat16 *__restrict c, const bfloat16 *__restrict sct,
                                              const bfloat16 *__restrict dg, float *__restrict y)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 32> Y0 = aie::zeros<accfloat, 32>(), Y1 = aie::zeros<accfloat, 32>(), Y2 = aie::zeros<accfloat, 32>();
    const bfloat16 __aie_dm_resource_a *__restrict pc = (const bfloat16 __aie_dm_resource_a *)c;
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)sct;
    AIE_LOOP_UNROLL_FULL  // (pipelined: a ramp longer than the 8 iterations)
    for (int i = 0; i < 8; i++) {
        Y0 = aie::mac(Y0, aie::load_v<32>(pc + 384 * i), aie::load_v<8>(ps + 32 * i).template grow_replicate<32>());
        Y1 = aie::mac(Y1, aie::load_v<32>(pc + 384 * i + 32), aie::load_v<8>(ps + 32 * i + 8).template grow_replicate<32>());
        Y2 = aie::mac(Y2, aie::load_v<32>(pc + 384 * i + 64), aie::load_v<8>(ps + 32 * i + 16).template grow_replicate<32>());
    }
    {{ p }}_k23d(y, Y0, dg);
    {{ p }}_k23d(y + 32, Y1, dg + 8);
    {{ p }}_k23d(y + 64, Y2, dg + 16);
}

// super-blocks 2 H, 2 H + 1 of an IQ3_XXS tile: y[g][32] (rows 8 g ..) += their rows against xp (their 64 panels){{ "; ys:
// the second super-block's rows at y + ys (row groups)" if rg else "" }}
template <int H>
__attribute__((noinline)) void {{ p }}_iq3(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr{{ ", int ys" if rg else "" }})
{
    uint8_t *mk = scr + 6144;
    bfloat16 *sct = (bfloat16 *)(scr + 7168), *dg = (bfloat16 *)(scr + 7936);
    // (the table copies' addresses through a volatile word: constants at the call would specialize _iq3_w, and the
    // gathers' memory operands need pointers, AIEPostSelectOptimize asserts on immediates)
    volatile int32_t *ta = (volatile int32_t *)(scr + 8000);
    ta[0] = H ? {{ TAB[1][0] }} : {{ TAB[0][0] }};
    ta[1] = H ? {{ TAB[1][1] }} : {{ TAB[0][1] }};
    const uint32_t *tab_ab = (const uint32_t *)(intptr_t)ta[0], *tab_cd = (const uint32_t *)(intptr_t)ta[1];
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        uint8_t *un = a + {{ U3 }} * (2 * H + u);  // (the super-block's unit: [ix | ss | d], odd ones [d | ss | ix])
        {{ p }}_iq3_ss(un + (u ? {{ U3 - I3 - S3 }} : {{ I3 }}), (v16int32 *)mk, sct);
        const aie::vector<bfloat16, 32> dv = aie::vector<int32, 16>({{ p }}_k23w1((uintptr_t)(un + (u ? 0 : {{ I3 + S3 }})), {{ 2 * RI3 }})).template cast_to<bfloat16>();
        AIE_LOOP_UNROLL_FULL
        for (int g = 0; g < 3; g++)
            aie::store_v(dg + 8 * g, dv.template extract<8>(g));
        {{ p }}_iq3_w(un + (u ? {{ U3 - I3 }} : 0), mk, tab_ab, tab_cd, scr);
        {{ p }}_iq3_x(xp + {{ 32 * PANEL }} * u, scr, (bfloat16 *)scr);
        {{ p }}_iq3_sc((const bfloat16 *)scr, sct, dg, y{{ " + u * ys" if rg else "" }});
    }
}
"""

# one weight panel from its 16 grid entries' addresses and the signs (before ``mmv/IQ3_XXS``, which calls it)
_PANEL = r"""// ---- IQ3_XXS: a weight panel (8 rows x 8 k) from its 16 entries' addresses a (lanes 2 r, 2 r + 1: row r's two) and its
// signs m (bit 8 r + j: value j of row r negative): 4 gathers, the entries compacted, zero - w where a sign is set
[[clang::always_inline]] inline v64int8 {{ p }}_iq3_panel(v16int32 a, v2int32 m)
{
    const v8int32 a0 = extract_v8int32(a, 0), a1 = extract_v8int32(a, 1);
    const v16int32 c01 = concat(load_4x32_lo(a0), load_4x32_hi(a0));  // (entries 0 .. 7 in the even 32-bit lanes)
    const v16int32 c23 = concat(load_4x32_lo(a1), load_4x32_hi(a1));
    return __builtin_aie2p_vaddsub8(broadcast_zero_to_v64int8(), (v64int8)::shuffle(c01, c23, T32_16x2_lo), m);
}
"""


# GemmQd (prefill) on super-block units: the pass's rows compacted (``_compact``: tile t's rows from R t on)
_GEMM = r"""// IQ3_XXS (GemmQd, super-block units: formats/iq3.py): the weight object is a unit of the pass's {{ tiles }} tiles
// ({{ U3 }} B each, at a pitch of {{ UP }} B: [indices [i][h][l' 2][g][r 8][2] | sign / scale words [i][row] | d [row]],
// odd super-blocks [d | sign / scale words | indices]). Its rows compacted: MMUL row q is tile q / {{ R }}'s row q %
// {{ R }} ({{ n }} rows; past the tiles d = 0, so w = 0). Per unit _sb: d, the signs and the scales S = bf16(d (2 ls + 1)
// / 4) of the compacted rows ([i][16-row block][l][16] signs, [i][q] scales); per k-tile _dq (its first call of a
// k-tile: _prep, the k-tile's 8 index lists compacted) the panels' codes (_iq3g: per panel the 4 gathers of the grid,
// the signs as 64 +- g by vaddsub), then the panels (_iq3s: w = S (192 +- g) - 192 S exact in fp32, rounded once
// into bfp16).
// Scratch (ws, bytes): S [i 8][{{ n }}] bf16 | signs [i 8][{{ n // 16 }}][64] | at {{ WU }} codes [{{ n_mac // 16 }} blocks][ks
// 8][rh 2][64] (_sb: the sign / scale words) | at {{ WL }} lists [ks 8][{{ LS }} B: {{ n }} x 2, whole 64-byte lines (aligned
// stores)]; hb: d [{{ n }}] bf16
""" + _SIGNS + r"""
{# Compaction (MMUL row q: tile q / R's row q % R): tile t's rows of e bytes each from src + t UP; the window of rows
   from q0 = 64 c / e on is the window of rows from 0 on moved by 64 c bytes: per tile one aligned pointer a and one
   shift s (``windows``), per 64 output bytes the tiles' windows selected (``rows``: the first tile's whole; zero: rows
   past the tiles 0, else they hold junk) #}
{% macro windows(src, e) -%}
{% for t in range(tiles) %}
        const uintptr_t o{{ t }} = (uintptr_t)({{ src }} + {{ t * UP - e * R * t }});
        const v16int32 *a{{ t }} = (const v16int32 *)(o{{ t }} & ~(uintptr_t)63);
        const unsigned s{{ t }} = (unsigned)(o{{ t }} & 63);
{% endfor %}
{%- endmacro %}

{% macro rows(dst, e, q0, nq, zero=False) -%}
        {   // (rows {{ q0 }} .. {{ q0 + nq - 1 }})
{% set c = e * q0 // 64 %}
{% if zero %}
            aie::vector<int8, 64> v = aie::zeros<int8, 64>();
{% endif %}
{% for t, lo, hi in seg(q0, q0 + nq) %}
{% set w = "(v64int8)shift_bytes(a%d[%d], a%d[%d], s%d)" % (t, c, t, c + 1, t) %}
{% if loop.first and not zero %}
            aie::vector<int8, 64> v({{ w }});
{% else %}
            v = aie::select(v, aie::vector<int8, 64>({{ w }}),
                            aie::mask<64>::from_uint64({{ "0x%016Xull" % mask(e * (lo - q0), e * (hi - q0) if zero else 64) }}));
{% endif %}
{% endfor %}
            aie::store_v((int8_t *)({{ dst }}), v{{ ".template extract<32>(0)" if e * nq == 32 else "" }});
        }
{%- endmacro %}

// A unit's sign / scale words of the compacted rows (from bs: [i 8][row {{ R }}] per tile) -> pw [i 8][{{ n // 16 }}][16 rows]
__attribute__((noinline)) void {{ p }}_sbw(const uint8_t *__restrict bs, uint8_t *__restrict pw)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 8; i++) {
{{ windows("bs", 4) }}
{% for B in range(n // 16) %}
{{ rows("pw + %d" % (64 * B), 4, 16 * B, 16) }}
{% endfor %}
        bs += {{ 4 * R }};
        pw += {{ 4 * n }};
    }
}

// The signs and scales of the compacted rows from their words (pw [i 8][{{ n // 16 }}][16 rows]): the signs [l][r 16] of
// 16 rows -> pm [i 8][{{ n // 16 }}][64], S = bf16(s d) -> ps [i 8][{{ n }}] (dc: d of the compacted rows)
__attribute__((noinline)) void {{ p }}_sbs(const uint8_t *__restrict pw, uint8_t *__restrict pm, bfloat16 *__restrict ps,
                                           const bfloat16 *__restrict dc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    v16int32 lw = broadcast_zero_to_v16int32();  // (the signs' scale panel: _iq3_signs)
    lw = upd_elem(lw, 0, (int)0xF8FCFEFF);
    lw = upd_elem(lw, 3, (int)0xF8FCFEFF);
    v64bfp16ebs8 LW, LX;
    LW.mantissa = (v64int8)lw;
    LW.exponent = al::e8(125);
    LX.exponent = al::e8(0x85);
    const v2int32 all = (v2int32)~0ULL;
    const v64int8 __aie_dm_resource_a *__restrict pv = (const v64int8 __aie_dm_resource_a *)pw;
    v64int8 __aie_dm_resource_b *__restrict po = (v64int8 __aie_dm_resource_b *)pm;
    const bfloat16 __aie_dm_resource_c *__restrict pd = (const bfloat16 __aie_dm_resource_c *)dc;
    dims_2d_t wd = dims_2d_from_steps({{ n // 16 }}, 16, 0);  // (the blocks' d, again for each group)
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int it = 0; it < {{ n // 2 }}; it++) {  // (group i, 16-row block) = (it / {{ n // 16 }}, it % {{ n // 16 }})
        v64int8 m;
        v8int32 sv;
        {{ p }}_iq3_signs(*pv++, LX, LW, all, m, sv);
        *po++ = m;
        aie::store_v(ps, aie::mul(aie::vector<bfloat16, 16>((v16bfloat16)sv), aie::load_v<16>(pd)).template to_vector<bfloat16>());
        ps += 16;
        pd = add_2d_ptr(pd, wd);
    }
}

// One unit's d, signs and scales of the compacted rows (par: the unit's parity: its parts' offsets)
__attribute__((noinline)) void {{ p }}_sbp(const uint8_t *__restrict b, uint8_t *__restrict ws, bfloat16 *__restrict dc,
                                           int par)
{
    {
        const uint8_t *bd = b + (par ? 0 : {{ I3 + S3 }});
{{ windows("bd", 2) }}
{% for c in range(0, n, 32) %}
{{ rows("dc + %d" % c, 2, c, [32, n - c] | min, zero=True) }}
{% endfor %}
    }
    {{ p }}_sbw(b + (par ? {{ U3 - I3 - S3 }} : {{ I3 }}), ws + {{ WU }});  // (the words in the codes' place)
    {{ p }}_sbs(ws + {{ WU }}, ws + {{ 16 * n }}, (bfloat16 *)ws, dc);
}

void {{ p }}_sb(const uint8_t *__restrict b, bfloat16 *__restrict ws, bfloat16 *__restrict hb, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    {{ p }}_sbp(b, (uint8_t *)ws, hb, sb & 1);
}

// k-tile kt's 8 index lists (ks = 4 i' + 2 h + l': tile t's 48 bytes [r 24][2] at 384 kt + 48 ks of its indices) of
// the compacted rows
__attribute__((noinline)) void {{ p }}_prep(const uint8_t *__restrict b, uint8_t *__restrict ws, int kt, int par)
{
    const uint8_t *bi = b + (par ? {{ U3 - I3 }} : 0) + 384 * kt;
    uint8_t *__restrict pl = ws + {{ WL }};
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int ks = 0; ks < 8; ks++) {
{{ windows("bi", 2) }}
{% for c in range(0, n, 32) %}
{{ rows("pl + %d" % (2 * c), 2, c, [32, n - c] | min) }}
{% endfor %}
        bi += 48;
        pl += {{ LS }};
    }
}

// a panel's codes 64 +- g (8 rows x 8 k; a: its 16 grid entries' addresses, m: its signs, bit 8 r + j: value j of row
// r negative): 4 gathers, the entries compacted, 64 - g where a sign is set
[[clang::always_inline]] inline v64int8 {{ p }}_iq3u(v16int32 a, v2int32 m)
{
    const v8int32 a0 = extract_v8int32(a, 0), a1 = extract_v8int32(a, 1);
    const v16int32 c01 = concat(load_4x32_lo(a0), load_4x32_hi(a0));  // (entries 0 .. 7 in the even 32-bit lanes)
    const v16int32 c23 = concat(load_4x32_lo(a1), load_4x32_hi(a1));
    return __builtin_aie2p_vaddsub8(broadcast_to_v64int8(64), (v64int8)::shuffle(c01, c23, T32_16x2_lo), m);
}

// The codes of a dq call's panels (u [16-row block][ks 8][rh 2][64]): per (block, group i') the 4 lists' 16 rows (one
// load each: upshifted into an accumulator) + the copies' bases (lanes 0, 1: tab_ab, 2, 3: tab_cd), the signs ([l 4][rh
// 2][8]: two 32-byte loads; one 64-byte load left the loop unpipelined at 48 rows), 8 panels
__attribute__((noinline)) void {{ p }}_iq3g(const uint8_t *__restrict li, const uint8_t *__restrict mk,
                                            const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd,
                                            uint8_t *__restrict u)
{
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v32uint8 __aie_dm_resource_a *__restrict pi = (const v32uint8 __aie_dm_resource_a *)li;
    const v8int32 __aie_dm_resource_b *__restrict pm = (const v8int32 __aie_dm_resource_b *)mk;
    v64int8 __aie_dm_resource_c *__restrict pu = (v64int8 __aie_dm_resource_c *)u;
    // (lists: i' by 4, then the next block's 16 rows; signs: i' by a group's, the next block 64 B)
    dims_2d_t wi = dims_2d_from_steps(2, {{ 4 * LS // 32 }}, 1);
    dims_2d_t wm = dims_2d_from_steps(2, {{ 4 * n // 32 }}, 2);
    for (int it = 0; it < {{ n_mac // 8 }}; it++) {  // (16-row block, i') = (it / 2, it % 2)
        const v16int32 m0 = set_v16int32(0, pm[0]), m1 = set_v16int32(0, pm[1]);
{% for j in range(4) %}
        const v32acc32 q{{ j }} = ups_to_v32acc32(pi[{{ j * LS // 32 }}], 3, 0);  // (list 4 i' + {{ j }})
{% endfor %}
{% for j in range(8) %}
        pu[{{ j }}] = {{ p }}_iq3u(add((v16int32)extract_v16acc32(q{{ j // 2 }}, {{ j % 2 }}), b16), ext_v2int32(m{{ j // 4 }}, {{ j % 4 }}));
{% endfor %}
        pi = add_2d_ptr(pi, wi);
        pm = add_2d_ptr(pm, wm);
        pu += 8;
    }
}

// The panels from their codes u and the scales (S: the call's rows of group 2 kt, the next group's {{ n }} further): per
// (16-row block, group i') the two row groups' scales over their rows' lanes; per panel the codes as bf16 128 + u (zero-
// extended on the load, + 0x4300), w = S (128 + u) - 192 S = S (+- g), exact in fp32, rounded once
__attribute__((noinline)) void {{ p }}_iq3s(const uint8_t *__restrict u, const bfloat16 *__restrict S,
                                            uint8_t *__restrict stage)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const uint8_t __aie_dm_resource_a *__restrict pu = (const uint8_t __aie_dm_resource_a *)u;
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)S;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> out((bfp16ebs8 *)stage);
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4300);
    const aie::vector<bfloat16, 64> k192 = aie::broadcast<bfloat16, 64>(-192.0f);
    dims_2d_t wsc = dims_2d_from_steps(2, {{ n }}, 16);  // (i' by a group's rows, then the next block)
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int it = 0; it < {{ n_mac // 8 }}; it++) {  // (16-row block, i') = (it / 2, it % 2)
        aie::vector<bfloat16, 64> S0, S1;
        al::expand16_bc(aie::load_v<16>(ps), S0, S1);
        const aie::accum<accfloat, 64> C0 = aie::mul(S0, k192), C1 = aie::mul(S1, k192);
{% for j in range(8) %}
        out.push(aie::mac(C{{ j % 2 }}, aie::concat(aie::add(aie::unpack(aie::load_v<32>(pu + {{ 64 * j }})), k43).template cast_to<bfloat16>(),
                                         aie::add(aie::unpack(aie::load_v<32>(pu + {{ 64 * j + 32 }})), k43).template cast_to<bfloat16>()),
                          S{{ j % 2 }}).template to_vector<bfp16ebs8>());
{% endfor %}
        pu += 512;
        ps = add_2d_ptr(ps, wsc);
    }
}

// k-tile kt -> {{ n_mac }} x 64 bfp16 panels (rows {{ n_mac }} half ..) in the MAC pop order [16-row block][k-step ks][8-row
// half]: on its first call the k-tile's index lists (_prep), the codes (_iq3g), then the panels (_iq3s)
void {{ p }}_dq(const uint8_t *__restrict b, bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt{{ half_arg }}, const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd)
{
{% if "ev_dq" in ablate or "ev_kt" in ablate %}
    event0();
{% endif %}
{% if "dq" in ablate %}
    return;
{% endif %}
{% if halves > 1 %}
    if (half == 0)
        {{ p }}_prep(b, (uint8_t *)ws, kt, sb & 1);
    const int r0 = {{ n_mac }} * half;  // (the half's first row)
{% else %}
    {{ p }}_prep(b, (uint8_t *)ws, kt, sb & 1);
    const int r0 = 0;
{% endif %}
    uint8_t *wsb = (uint8_t *)ws;
    {{ p }}_iq3g(wsb + {{ WL }} + 2 * r0, wsb + {{ 16 * n }} + {{ 8 * n }} * kt + 4 * r0, tab_ab, tab_cd, wsb + {{ WU }});
    {{ p }}_iq3s(wsb + {{ WU }}, ws + {{ 2 * n }} * kt + r0, stage);
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


class Iq3Slot(Slot):
    formats = (TileFormat("IQ3_XXS", 10, 98, 24, 1, 98, 0, planes=3, pad8_row=64, unit_align=16, dq_row_cycles=11.0,
                          mmv="iq3"),)
    sources: ClassVar[dict[str, str]] = {"mmv/IQ3_XXS": _MMV, "mmv/iq3panel": _PANEL, "gemm/IQ3_XXS": _GEMM}

    @staticmethod
    def unit(R: int) -> tuple[int, int, int, int]:
        """(unit, indices, sign / scale words, d) bytes of a super-block's unit of R rows (module docstring)."""
        I, S, D = 512 * -(-R // 8), 32 * R, 2 * R
        return -(-(I + S + D) // 16) * 16, I, S, D

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert nsb % 2 == 0, nsb
        U, I, S, D = self.unit(R)
        d = k23.bf16_u16(f["d"]).astype(np.uint16)  # (R, nsb)
        G = -(-R // 8)
        idx = np.zeros((8 * G, nsb, 8, 4, 2), np.uint8)  # [row][sb][i][l][2]
        idx[:R] = f["idx"].reshape(R, nsb, 8, 4, 2)
        ix = idx.reshape(G, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6).reshape(nsb, I)  # [sb][i][h][l'][g][r][2]
        ss = ss_words(f["signs"], f["ls"]).reshape(R, nsb, 8).transpose(1, 2, 0)  # [sb][i][row]
        units = np.zeros((nsb, U), np.uint8)
        for sb in range(nsb):
            parts = [ix[sb], np.ascontiguousarray(ss[sb]).view(np.uint8).reshape(-1),
                     np.ascontiguousarray(d[:, sb]).view(np.uint8)]
            if sb % 2:  # (odd: [d | ss | ix], ix at the unit's end)
                units[sb, :D], units[sb, U - I - S:U - I], units[sb, U - I:] = parts[2], parts[1], parts[0]
            else:
                units[sb, :I], units[sb, I:I + S], units[sb, I + S:I + S + D] = parts
        return [units.reshape(-1)]

    # GemmQd (prefill): super-block units

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return self.unit(rows)[0]

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 5 units in the dequantizing core's L1: 23 KB; MMUL columns <= 128)
        return tiles <= 5

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: S, the signs, a call's codes, the index lists (module docstring of gemm/IQ3_XXS); hb: d; the grid
        # table's two copies (in two 16 KB regions: ``gemm_qd``)
        return {"ws": (1, self._ws(op)["size"]), "hb": (1, max(2 * op.n, 64)), "tab": (1, TAB_BYTES),
                "tabb": (1, TAB_BYTES)}

    def _ws(self, op: object) -> dict[str, int]:
        """Byte offsets in the prefill dequant's scratch: the codes (WU: a dq call's, or a unit's sign / scale words),
        the index lists (WL); its size."""
        WU = 48 * op.n
        WL = WU + max(64 * op.nm, 32 * op.n)
        return {"WU": WU, "WL": WL, "size": WL + 8 * self.list_bytes(op.n)}

    @staticmethod
    def list_bytes(n: int) -> int:
        """Bytes of a compacted index list of n rows in the prefill dequant's scratch (whole 64-byte lines)."""
        return -(-2 * n // 64) * 64

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == 23 and op.n % 16 == 0 and op.nm % 16 == 0, (op.R, op.n)
        tab_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, grid_words()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        U, I, S, _ = self.unit(op.R)
        R, T = op.R, op.tiles

        def seg(q0: int, q1: int) -> list[tuple[int, int, int]]:  # (tile, first row, end) of compacted rows q0 .. q1
            return [(t, max(q0, R * t), min(q1, R * (t + 1))) for t in range(q0 // R, min(T, -(-q1 // R)))]

        def mask(lo: int, hi: int) -> int:  # (bytes lo .. hi - 1 of 64)
            return ((1 << hi) - 1) & ~((1 << lo) - 1)

        ws = self._ws(op)
        return {"UP": op.unit_pitch, "U3": U, "I3": I, "S3": S, "LS": self.list_bytes(op.n), "WU": ws["WU"],
                "WL": ws["WL"], "seg": seg, "mask": mask}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/iq3panel", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert 16 < R <= 24, R  # (3 row groups: the kernels' layout)
        U, I, S, _ = self.unit(R)
        return {"RI3": R, "TAB": TAB, "U3": U, "I3": I, "S3": S}


SLOT = Iq3Slot()
