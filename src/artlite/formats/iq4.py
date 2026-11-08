"""IQ4_NL and IQ4_XS tiles (R = 8 or 16 rows in row groups of 8; IQ4_NL 144, IQ4_XS 136 B per row and super-block,
the GGUF blocks' sizes): per super-block a unit of ``unit_bytes`` (R 16: IQ4_NL 2304 B, IQ4_XS 2176 B; 4 per
1024-column tile, nothing else in the tile) holding

- the codes [32-k group i 8][8-k step s 4][row group g][row 8][4 B]: a word = the 8 codes of k 32 i + 8 s .. of a row,
  byte b = codes 2 b (low nibble) and 2 b + 1 (high): an 8 x 8 weight panel's 32 bytes in a row;
- IQ4_NL: d [i 8][row] bf16;
- IQ4_XS: the 6-bit scales ls as a low plane (nibbles [i 8][row]: ls & 15) and a high plane (nibbles [i' 4][row]:
  (ls_i' >> 4) | (ls_i'+4 >> 4) << 2), then d [row] bf16;

nibbles in pairs per byte (the first one low), in the order of the list.

w = d v[q] (IQ4_NL: d per 32-k group), w = d (ls - 32) v[q] (IQ4_XS), v = ``gguf_model.IQ4_VALUES`` (16 int8 values).

Decode (``mmv/iq4``): a weight panel (8 rows x 8 k) is its 32 code bytes looked up by the gather unit in a table of the
256 byte values' value pairs [v[lo], v[hi]] (``pair_words``: 16-bit entries, 4x16 gathers, ``common.TAB``): 8 gathers
per panel, two panels per iteration (II 20, bound by the gathers' DM_ADA port); then the MMUL against the x panels, per
32-k group the rows' sums c, and y += hilo(sum_i c_i (ls_i - 32)) d (IQ4_XS: the integer scales exact in bf16) or
y += sum_i c_i d_i (IQ4_NL: bf16 d).

Prefill (``gemm/IQ4_NL``, ``gemm/IQ4_XS``: GemmQd on super-block units, a pass of whole 16-row tiles, MMUL row block t =
tile t): per k-tile and tile the 8 k-steps' 64 code bytes each by 16 gathers of a table of the byte values' bf16 pairs
(``bf16_pair_words``: 32-bit entries, 4x32), w = S v rounded once into bfp16, S = d (IQ4_NL) or bf16(d (ls - 32))
(IQ4_XS: per unit, ``_sb``)."""

from typing import ClassVar

import numpy as np

from . import k23
from .common import TAB, DqKernels, Slot, TileFormat, dq_kernel, i32


def _pairs(nib: np.ndarray) -> np.ndarray:
    """(..., 2 n) nibbles -> (..., n) bytes, the first of a pair low."""
    return (nib[..., 0::2] | nib[..., 1::2] << 4).astype(np.uint8)


def codes(q: np.ndarray, R: int, nsb: int) -> np.ndarray:
    """(R, 256 nsb) codes -> (nsb, 128 R) bytes [sb][i][s][g][r][4 B] (module docstring)."""
    G = R // 8
    b = _pairs(q.reshape(R, nsb, 8, 4, 8))  # [row][sb][i][s][4 B]
    return np.ascontiguousarray(b.reshape(G, 8, nsb, 8, 4, 4).transpose(2, 3, 4, 0, 1, 5)).reshape(nsb, -1)


def pair_words() -> np.ndarray:
    """The decode's table (256 uint32 = 1 KB): entry b = [v[b & 15], v[b >> 4]] (int8 pair) at byte 4 b, i.e. per group of
    16 entries their 32 bytes twice."""
    from ..gguf_model import IQ4_VALUES

    v = IQ4_VALUES.view(np.uint8).astype(np.uint16)
    e = v[np.arange(256) & 15] | v[np.arange(256) >> 4] << 8
    return np.concatenate([e.reshape(16, 16)] * 2, 1).reshape(-1).view(np.uint32)


TAB_PREFILL = 2048  # (the prefill's table copies)


def bf16_pair_words() -> np.ndarray:
    """The prefill's table (512 uint32 = 2 KB): entry b = [bf16(v[b & 15]), bf16(v[b >> 4])] at byte 8 b, i.e. per group
    of 8 entries their 32 bytes twice (4x32 gathers)."""
    from ..gguf_model import IQ4_VALUES

    v = (IQ4_VALUES.astype(np.float32).view(np.uint32) >> 16).astype(np.uint32)
    e = v[np.arange(256) & 15] | v[np.arange(256) >> 4] << 16
    return np.concatenate([e.reshape(32, 8)] * 2, 1).reshape(-1)


# GemmQd (prefill) on super-block units of whole 16-row tiles
_GEMM = r"""// {{ fmt }} (GemmQd, super-block units: formats/iq4.py): the weight object is a unit of the pass's {{ tiles }} tiles
// ({{ UP }} B each: [codes [i][s][g 2][r 8][4 B] | {{ "d [i][row 16] bf16" if fmt == "IQ4_NL" else "ls planes, d [row 16]" }}]); MMUL
// row block t is tile t's 16 rows. {{ "Per unit _sb: S = bf16(d (ls - 32)) [i][16 t + r] in ws; " if fmt == "IQ4_XS" else "" }}per k-tile and 16-row
// block the k-steps' panels: 64 code bytes, 16 gathers of the bf16 pairs [v[lo], v[hi]] (entry at 8 b), w = S v
// rounded once (bf16 times bf16: exact in fp32)

// 16 entries of the 4x32 table (a: their addresses) -> 32 bf16 (entries in order)
[[clang::always_inline]] inline v16int32 {{ p }}_g32(v16int32 a)
{
    const v8int32 a0 = extract_v8int32(a, 0), a1 = extract_v8int32(a, 1);
    const v16int32 c01 = concat(load_4x32_lo(a0), load_4x32_hi(a0));  // (entries 0 .. 7 in the even 32-bit lanes)
    const v16int32 c23 = concat(load_4x32_lo(a1), load_4x32_hi(a1));
    return ::shuffle(c01, c23, T32_16x2_lo);
}

{% if fmt == "IQ4_XS" %}
// A unit's scales of the pass's rows: per tile S [i][16 t + r] = bf16(d (ls - 32)) (ls from the low plane, nibbles [i][row
// 16], and the high one, nibbles [i' 4][row 16] = crumb i' | crumb i' + 4 << 2; bf16(128 + ls) by its bits, minus 160)
void {{ p }}_sb(const uint8_t *__restrict b, bfloat16 *__restrict ws, bfloat16 *__restrict hb, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<uint8, 64> m3 = aie::broadcast<uint8, 64>(3);
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4300);
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k160(aie::broadcast<float, 32>(-160.0f));
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tiles }}; t++) {
        const uint8_t *u = b + {{ UP }} * t + 2048;
        const aie::vector<uint8, 64> h = aie::unpack(aie::load_v<64>((const uint4 *)(u + 64)));  // ([i' 4][row 16])
        const aie::vector<bfloat16, 32> d = aie::concat(aie::load_v<16>((const bfloat16 *)(u + 96)), aie::load_v<16>((const bfloat16 *)(u + 96)));
        AIE_LOOP_UNROLL_FULL
        for (int g = 0; g < 2; g++) {  // (groups 4 g ..)
            const aie::vector<uint8, 64> l = aie::unpack(aie::load_v<64>((const uint4 *)(u + 32 * g)));
            const aie::vector<uint8, 64> c = g ? aie::downshift(h, 2) : aie::bit_and(h, m3);
            const aie::vector<uint8, 64> ls = aie::bit_or(l, aie::upshift(aie::bit_and(c, m3), 4));
            AIE_LOOP_UNROLL_FULL
            for (int j = 0; j < 2; j++) {  // (groups 4 g + 2 j, + 1)
                const aie::vector<bfloat16, 32> v = aie::add(aie::unpack(ls.template extract<32>(j)), k43).template cast_to<bfloat16>();
                const aie::vector<bfloat16, 32> sv = aie::mac(k160, v, one).template to_vector<bfloat16>();
                const aie::vector<bfloat16, 32> S = aie::mul(sv, d).template to_vector<bfloat16>();
                aie::store_v(ws + {{ n }} * (4 * g + 2 * j) + 16 * t, S.template extract<16>(0));
                aie::store_v(ws + {{ n }} * (4 * g + 2 * j + 1) + 16 * t, S.template extract<16>(1));
            }
        }
    }
}
{% else %}
void {{ p }}_sb(const uint8_t *__restrict b, bfloat16 *__restrict ws, bfloat16 *__restrict hb, int32_t sb)
{
}
{% endif %}

// k-tile kt -> {{ n_mac }} x 64 bfp16 panels (rows {{ n_mac }} half ..) in the MAC pop order [16-row block][k-step ks][8-row
// half]
void {{ p }}_dq(const uint8_t *__restrict b, bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt{{ half_arg }}, const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd)
{
{% if "ev_dq" in ablate or "ev_kt" in ablate %}
    event0();
{% endif %}
{% if "dq" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int t0 = {{ "half * %d" % (n_mac // 16) if halves > 1 else "0" }};
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> out((bfp16ebs8 *)stage);
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v64int8 eight = broadcast_to_v64int8(8);
    // one loop over (tile t, k-step ks): the codes by 64 B, the next tile a unit on; the scales of group 2 kt + ks / 4
    const v64uint8 __aie_dm_resource_a *__restrict pc = (const v64uint8 __aie_dm_resource_a *)(b + {{ UP }} * t0 + 512 * kt);
    dims_2d_t wc = dims_2d_from_steps(8, 1, {{ UP // 64 }});
{% if fmt == "IQ4_NL" %}
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)(b + {{ UP }} * t0 + 2048 + 64 * kt);
    dims_3d_t wsc = dims_3d_from_steps(4, 0, 2, 16, {{ UP // 2 }});
{% else %}
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)ws + {{ 2 * n }} * kt + 16 * t0;
    dims_3d_t wsc = dims_3d_from_steps(4, 0, 2, {{ n }}, 16);
{% endif %}
    AIE_TRY_INITIATION_INTERVAL(18)  // (post-RA II 21, bound 18: the gathers' DM_ADA; nested per tile: II 28 + a ramp per tile)
    for (int it = 0; it < {{ n_mac // 2 }}; it++) {
        aie::vector<bfloat16, 64> S0, S1;
        al::expand16_bc(aie::load_v<16>(ps), S0, S1);
        const v64acc32 q = mul_elem_64_2(*pc, eight);  // (8 b: [r 16][4 B])
        const v16int32 e0 = {{ p }}_g32(add((v16int32)extract_v16acc32(q, 0), b16));
        const v16int32 e1 = {{ p }}_g32(add((v16int32)extract_v16acc32(q, 1), b16));
        const v16int32 e2 = {{ p }}_g32(add((v16int32)extract_v16acc32(q, 2), b16));
        const v16int32 e3 = {{ p }}_g32(add((v16int32)extract_v16acc32(q, 3), b16));
        out.push(aie::mul(aie::concat(aie::vector<bfloat16, 32>((v32bfloat16)e0), aie::vector<bfloat16, 32>((v32bfloat16)e1)), S0)
                     .template to_vector<bfp16ebs8>());
        out.push(aie::mul(aie::concat(aie::vector<bfloat16, 32>((v32bfloat16)e2), aie::vector<bfloat16, 32>((v32bfloat16)e3)), S1)
                     .template to_vector<bfp16ebs8>());
        pc = add_2d_ptr(pc, wc);
        ps = add_3d_ptr(ps, wsc);
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""

# the decode GEMV (``decode.gemv``; the 4x16 gathers: ``mmv/g16``, the scaling by d: ``mmv/k23``)
_MMV = r"""// ---- IQ4_NL / IQ4_XS (formats/iq4.py: {{ R4 }} rows, 2 row groups): per super-block a unit, [codes [i][s][g][r][4 B] |
// IQ4_NL: d [i][row] bf16 | IQ4_XS: ls low nibbles [i][row], high crumbs [i' 4][row], d [row] bf16]. Core H: super-blocks
// 2 H, 2 H + 1; per super-block u: the weight panels (_iq4_w), each x panel against them (_iq4_x: the groups' sums c in
// place of panels (i, 0, g)), y += sum_i c_i S_i (IQ4_XS: S = ls - 32 then hilo times d, _iq4xs_ss / _iq4_sc).
// Scratch: w / c [i 8][s 4][g 2][64 B] (0 .. 4096) | sct [i][row 16] bf16 (4096) | the table's addresses (8000 .. 8008)

// The 64 weight panels of a super-block (codes [i][s][g][32 B], two panels per 64-byte load) -> wp [i][s][g][64 B]: per
// panel its 32 bytes' entries (4 b + the copies' bases), 8 gathers
__attribute__((noinline)) void {{ p }}_iq4_w(const uint8_t *__restrict codes, const uint32_t *__restrict tab_ab,
                                             const uint32_t *__restrict tab_cd, uint8_t *__restrict wp)
{
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v64int8 four = broadcast_to_v64int8(4);
    const v64uint8 __aie_dm_resource_a *__restrict pi = (const v64uint8 __aie_dm_resource_a *)codes;
    v64int8 __aie_dm_resource_c *__restrict pw = (v64int8 __aie_dm_resource_c *)wp;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 32; t++) {  // (II 20: the gathers' DM_ADA 17)
        const v64acc32 q = mul_elem_64_2(*pi++, four);
        pw[0] = (v64int8)concat({{ p }}_g16(add((v16int32)extract_v16acc32(q, 0), b16)), {{ p }}_g16(add((v16int32)extract_v16acc32(q, 1), b16)));
        pw[1] = (v64int8)concat({{ p }}_g16(add((v16int32)extract_v16acc32(q, 2), b16)), {{ p }}_g16(add((v16int32)extract_v16acc32(q, 3), b16)));
        pw += 2;
    }
}

// The 2 row groups' 32-k groups of a super-block against xp (its 32 x panels, each popped once): c (i, g) = bf16(C[hi
// rows] + C[lo rows]) of the group's 4 panels, in place of its panel (i, 0, g) (w: [i][s][g][64 B], c: 64 B per (i, g)
// at 512 i + 64 g)
__attribute__((noinline)) void {{ p }}_iq4_x(const uint8_t *__restrict xp, const uint8_t *__restrict w,
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
        v64accfloat C0, C1;
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < 4; s++) {
            const v64bfp16ebs8 X = xs.pop();
            W.mantissa = *(const v64int8 __aie_dm_resource_b *)(pw + 128 * s);
            C0 = s ? mac_8x8_8x8T(X, 1, W, 1, C0) : mul_8x8_8x8T(X, 1, W, 1);
            W.mantissa = *(const v64int8 __aie_dm_resource_b *)(pw + 128 * s + 64);
            C1 = s ? mac_8x8_8x8T(X, 1, W, 1, C1) : mul_8x8_8x8T(X, 1, W, 1);
        }
        aie::store_v(pc, {{ p }}_sum2(C0));
        aie::store_v(pc + 32, {{ p }}_sum2(C1));
        pw += 512;
        pc += 256;
    }
}

// y[g] += sum over the 8 groups i of c (i, g) sct[i][g][r] for the 2 row groups (c (i, g): 32 bf16 at c + 256 i + 32 g),
// then (dg: IQ4_XS) y[g] += hilo(that sum) dg[g] (fp32 sums)
template <bool D>
__attribute__((noinline)) void {{ p }}_iq4_sc(const bfloat16 *__restrict c, const bfloat16 *__restrict sct,
                                              const bfloat16 *__restrict dg, float *__restrict y)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 32> Y0, Y1;
    if constexpr (D) {
        Y0 = aie::zeros<accfloat, 32>();
        Y1 = aie::zeros<accfloat, 32>();
    } else {
        Y0 = aie::accum<accfloat, 32>(aie::load_v<32>(y));
        Y1 = aie::accum<accfloat, 32>(aie::load_v<32>(y + 32));
    }
    const bfloat16 __aie_dm_resource_a *__restrict pc = (const bfloat16 __aie_dm_resource_a *)c;
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)sct;
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < 8; i++) {
        Y0 = aie::mac(Y0, aie::load_v<32>(pc + 256 * i), aie::load_v<8>(ps + 16 * i).template grow_replicate<32>());
        Y1 = aie::mac(Y1, aie::load_v<32>(pc + 256 * i + 32), aie::load_v<8>(ps + 16 * i + 8).template grow_replicate<32>());
    }
    if constexpr (D) {
        {{ p }}_k23d(y, Y0, dg);
        {{ p }}_k23d(y + 32, Y1, dg + 8);
    } else {
        aie::store_v(y, Y0.template to_vector<float>());
        aie::store_v(y + 32, Y1.template to_vector<float>());
    }
}

// IQ4_XS: sct [i][row 16] = ls - 32 (bf16, exact) from the low plane (lo: nibbles [i 8][row 16]) and the high one (hi:
// nibbles [i' 4][row 16] = crumb i' | crumb i' + 4 << 2): bf16(128 + ls) by its bits 0x4300 | ls, minus 160
__attribute__((noinline)) void {{ p }}_iq4xs_ss(const uint8_t *__restrict lo, const uint8_t *__restrict hi,
                                                bfloat16 *__restrict sct)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<uint8, 64> h = aie::unpack(aie::load_v<64>((const uint4 *)hi));  // ([i' 4][row 16])
    const aie::vector<uint8, 64> m3 = aie::broadcast<uint8, 64>(3);
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4300);
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k160(aie::broadcast<float, 32>(-160.0f));
    AIE_LOOP_UNROLL_FULL
    for (int u = 0; u < 2; u++) {  // (groups 4 u ..)
        const aie::vector<uint8, 64> l = aie::unpack(aie::load_v<64>((const uint4 *)(lo + 32 * u)));
        const aie::vector<uint8, 64> c = u ? aie::downshift(h, 2) : aie::bit_and(h, m3);
        const aie::vector<uint8, 64> ls = aie::bit_or(l, aie::upshift(aie::bit_and(c, m3), 4));
        AIE_LOOP_UNROLL_FULL
        for (int j = 0; j < 2; j++) {
            const aie::vector<bfloat16, 32> v = aie::add(aie::unpack(ls.template extract<32>(j)), k43).template cast_to<bfloat16>();
            aie::store_v(sct + 64 * u + 32 * j, aie::mac(k160, v, one).template to_vector<bfloat16>());
        }
    }
}

// super-blocks 2 H, 2 H + 1 of an IQ4 tile (X: IQ4_XS): y[g][32] (rows 8 g ..) += their rows against xp (their 64 panels)
template <int H, bool X>
[[clang::always_inline]] inline void {{ p }}_iq4(uint8_t *__restrict a, const uint8_t *__restrict xp, float *__restrict y,
                                                 uint8_t *__restrict scr)
{
    constexpr int U = X ? {{ 136 * R4 }} : {{ 144 * R4 }};
    bfloat16 *sct = (bfloat16 *)(scr + 4096);
    // (the table copies' addresses through a volatile word: constants at the call would specialize _iq4_w, and the
    // gathers' memory operands need pointers)
    volatile int32_t *ta = (volatile int32_t *)(scr + 8000);
    ta[0] = H ? {{ TAB[1][0] }} : {{ TAB[0][0] }};
    ta[1] = H ? {{ TAB[1][1] }} : {{ TAB[0][1] }};
    const uint32_t *tab_ab = (const uint32_t *)(intptr_t)ta[0], *tab_cd = (const uint32_t *)(intptr_t)ta[1];
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        uint8_t *un = a + U * (2 * H + u), *hd = un + {{ 128 * R4 }};
        if constexpr (X)
            {{ p }}_iq4xs_ss(hd, hd + {{ 4 * R4 }}, sct);
        {{ p }}_iq4_w(un, tab_ab, tab_cd, scr);
        {{ p }}_iq4_x(xp + {{ 32 * PANEL }} * u, scr, (bfloat16 *)scr);
        {{ p }}_iq4_sc<X>((const bfloat16 *)scr, X ? sct : (const bfloat16 *)hd, (const bfloat16 *)(hd + {{ 6 * R4 }}), y);
    }
}
"""

_NL = r"""template <int H>
__attribute__((noinline)) void {{ p }}_iq4nl(uint8_t *__restrict a, const uint8_t *__restrict xp, const uint8_t *__restrict xs,
                                             float *__restrict y, uint8_t *__restrict scr)
{
    {{ p }}_iq4<H, false>(a, xp, y, scr);
}
"""

_XS = r"""template <int H>
__attribute__((noinline)) void {{ p }}_iq4xs(uint8_t *__restrict a, const uint8_t *__restrict xp, const uint8_t *__restrict xs,
                                             float *__restrict y, uint8_t *__restrict scr)
{
    {{ p }}_iq4<H, true>(a, xp, y, scr);
}
"""


class Iq4Slot(Slot):
    formats = (TileFormat("IQ4_NL", 11, 144, 16, 8, 144, 0, planes=4, unit_align=64, dq_row_cycles=9.0, mmv="iq4nl"),
               TileFormat("IQ4_XS", 12, 136, 16, 8, 136, 0, planes=4, unit_align=64, dq_row_cycles=9.0,
                          mmv="iq4xs"))

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R % 8 == 0, R
        parts = [codes(f["q"], R, nsb)]
        if fmt == "IQ4_NL":
            d = k23.bf16_u16(f["d"]).astype(np.uint16).reshape(R, nsb, 8).transpose(1, 2, 0)  # [sb][i][row]
            parts.append(np.ascontiguousarray(d).view(np.uint8).reshape(nsb, -1))
        else:
            ls = f["ls"].reshape(R, nsb, 8).transpose(1, 2, 0)  # [sb][i][row]
            hi = ls >> 4
            parts += [_pairs((ls & 15).reshape(nsb, -1)), _pairs((hi[:, :4] | hi[:, 4:] << 2).reshape(nsb, -1)),
                      np.ascontiguousarray(k23.bf16_u16(f["d"]).astype(np.uint16).T).view(np.uint8)]
        units = np.concatenate(parts, axis=1)
        assert units.shape[1] == self.format(fmt).unit_bytes(R), (fmt, R, units.shape)
        return [units.reshape(-1)]

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return self.format(fmt).unit_bytes(rows)

    sources: ClassVar[dict[str, str]] = {"mmv/iq4": _MMV, "mmv/IQ4_NL": _NL, "mmv/IQ4_XS": _XS, "gemm/IQ4_NL": _GEMM,
                                         "gemm/IQ4_XS": _GEMM}

    # GemmQd (prefill): super-block units of whole 16-row tiles

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 8 units in the dequantizing core's L1: 36 KB; MMUL columns <= 128)
        return tiles <= 8

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: IQ4_XS's scales S [i][n] bf16; the bf16 pair table's two copies (``gemm_qd``: two regions)
        return {"ws": (1, 16 * op.n if op.fmt == "IQ4_XS" else 64), "hb": (1, 64), "tab": (1, TAB_PREFILL),
                "tabb": (1, TAB_PREFILL)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == 16 and op.unit_pitch % 64 == 0, (op.R, op.unit_pitch)
        tab_ty = np.ndarray[(TAB_PREFILL // 4,), np.dtype[np.uint32]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, bf16_pair_words()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        return {"UP": op.unit_pitch}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/g16", "mmv/iq4", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == 16, R  # (2 row groups: the kernels' layout)
        return {"R4": R, "TAB": TAB}

    def mmv_table(self, fmt: str) -> np.ndarray:
        return pair_words()


SLOT = Iq4Slot()
