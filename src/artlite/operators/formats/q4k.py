"""Q4_K tiles (144 B per row and super-block): per-row scale records (``pack``) | nibbles.

Prefill: 16-row tiles take the tile-major permutation (GemmQ: ``_hcp`` collects the header slots in the staging
buffer, ``_hdr_q4k`` expands them per K chunk; GemmQd: ``_hdr_sb`` per super-block; 1024-column tiles read their
1 KB header as a slice-sized unit from 1 KB before the tile, ``hdr_skip``)."""

from typing import ClassVar

import numpy as np

from .common import (
    DqKernels,
    Slot,
    TileFormat,
    _bf16_bytes,
    _nibbles,
    _word_order,
    collected_header,
    dq_kernel,
    sb_major,
)

# GemmQ / GemmQd: the records' expansion (with the shared 6-bit plane8), then the tile-major dequant
_GEMM = r"""// Q4_K: the header unit holds the per-row scale records (formats.pack_tile: per row sc lo
// nibbles, mn lo nibbles, sc hi crumbs, mn hi crumbs, d, -dmin), permuted like the codes. ws plane
// 0 <- A = bf16(d sc), plane 1 <- B = bf16(-dmin mn) (the decode's values), in the layout dq reads.
{% if nsb == 8 %}
// 2048-column tiles: record words w 0-7 sc lo, 8-15 mn lo, 16-19 sc hi, 20-23 mn hi, 24-27 d, 28-31
// -dmin; word w of tile t is at [w/2 % 4][t {{ tq }}][w % 2][w / 8][row 16].
#define {{ p }}_WOFF(t, w) (4 * (((((((w) & 7) >> 1) * {{ tq }} + (t)) * 2 + ((w) & 1)) * 4 + ((w) >> 3)) * 16))
{% set outs = [0, 16, 128 * tq, 128 * tq + 16, 256 * tq, 256 * tq + 16, 384 * tq, 384 * tq + 16] %}
{% else %}
// 1024-column tiles: record words w 0-3 sc lo, 4-7 mn lo, 8-9 sc hi, 10-11 mn hi, 12-13 d, 14-15 -dmin
// (16 per row). The header (16 rows x 16 words) is the upper half of a slice-sized unit read from 1 KB
// before the tile, so row r of word w lies at memtile-permutation row 8 + r / 2, step 2 (r % 2) + w / 8:
// the 8 even (e = 0) or odd (e = 1) rows of word w of tile t are 8 consecutive words at {{ p }}_HW(t, w, e)
// bytes (hb = the unit's 4 k-tile slots of {{ 128 * tq }} words).
#define {{ p }}_HW(t, w, e) (4 * ((((w) & 7) >> 1) * {{ 128 * tq }} + (t) * 128 + ((w) & 1) * 64 + (2 * (e) + ((w) >> 3)) * 16 + 8))
{% set outs = [0, 16, 32 * tq, 32 * tq + 16, 64 * tq, 64 * tq + 16, 96 * tq, 96 * tq + 16] %}
// 64 values of 8 rows (even, odd) -> [row 16][8] in row order (the rows' 8-element chunks interleaved)
[[clang::always_inline]] inline aie::vector<uint8, 128> {{ p }}_rows16(const uint8_t *__restrict pe,
                                                                      const uint8_t *__restrict po)
{
    auto [z0, z1] = aie::interleave_zip(aie::load_v<64>((const uint4 *)pe).unpack(),
                                        aie::load_v<64>((const uint4 *)po).unpack(), 8);
    return aie::concat(z0, z1);
}
{% if not dqb %}
// _hdr_q4k's parts. The bytes are transposed before the arithmetic (one 8x8 shuffle per 64 values; the bf16
// results took 12 per 128) and each crumb word serves its two super-blocks.
// 8 even / 8 odd rows' words at pe / po (8 nibbles each) -> bytes [nibble n 8][row 16]
[[clang::always_inline]] inline aie::vector<uint8, 128> {{ p }}_t16(const uint8_t *__restrict pe, const uint8_t *__restrict po)
{
    const v64uint8 e = (v64uint8)aie::load_v<64>((const uint4 *)pe).unpack();  // [row' 8][n 8]
    const v64uint8 o = (v64uint8)aie::load_v<64>((const uint4 *)po).unpack();
    auto [z0, z1] = aie::interleave_zip(aie::vector<uint8, 64>(::shuffle(e, T8_8x8)),
                                        aie::vector<uint8, 64>(::shuffle(o, T8_8x8)), 1);  // [n][row' 8][e 2]
    return aie::concat(z0, z1);
}
// half h (super-block parity) of the 16 rows' bf16 pairs of a scale word, tiled 4 times: lanes [4][row 16]
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_dh(const uint8_t *__restrict pe, const uint8_t *__restrict po,
                                                                    const int h)
{
    auto [d0, d1] = aie::interleave_zip(aie::load_v<8>((const uint32 *)pe), aie::load_v<8>((const uint32 *)po), 1);
    const aie::vector<bfloat16, 32> dd = aie::concat(d0, d1).template cast_to<bfloat16>();  // [row 16][half 2]
    const aie::vector<bfloat16, 16> l = h ? aie::filter_odd(dd, 1) : aie::filter_even(dd, 1);
    return aie::concat(l, l, l, l);
}
// 64 values [group 4][row 16] (lo: low nibbles, cr: crumb nibbles of the super-block pair, masked by cm: 3 (crumb
// << 4 = 16 (cr & 3)) or 12 (4 (cr & 12))) -> o0, o1 (groups 0 1, 2 3) = bf16(d v): v = lo + 16 c on the integer
// MAC, OR 0x4300 = the bf16 bits of 128 + v, then (128 + v) d - 128 d exact in fp32, one rounding
[[clang::always_inline]] inline void {{ p }}_g4(const int cm, aie::vector<uint8, 64> lo, aie::vector<uint8, 64> cr,
                                                aie::vector<bfloat16, 64> d, bfloat16 *__restrict o0,
                                                bfloat16 *__restrict o1)
{
    aie::accum<acc32, 64> v = aie::mul(lo, aie::broadcast<uint8, 64>(1));
    v = aie::mac(v, aie::bit_and(cr, aie::broadcast<uint8, 64>(cm)), aie::broadcast<uint8, 64>(cm == 3 ? 16 : 4));
    const aie::vector<bfloat16, 64> w =
        aie::bit_or(v.template to_vector<uint16>(), aie::broadcast<uint16, 64>(0x4300)).template cast_to<bfloat16>();
    const aie::vector<bfloat16, 64> r =
        aie::mac(aie::mul(d, aie::broadcast<bfloat16, 64>(-128.0f)), w, d).template to_vector<bfloat16>();
    aie::store_v(o0, r.template extract<32>(0));
    aie::store_v(o1, r.template extract<32>(1));
}
{% endif %}
{% endif %}
{% include "gemm/plane8" %}
// k-tile slot kt of the Q4_K header (one {{ tq // 2 }} KB L1 object) -> hb + {{ tq // 2 }} KB kt: the records of a
// super-block span all four slots, so _hdr_q4k expands them once the header is complete
void {{ p }}_hcp(const bfloat16 *__restrict b, bfloat16 *__restrict hb, int32_t kt)
{
    // (walked pointers: II 1; indexed b + 32 i, hb + 32 i shared one index register, II 6-9)
    const bfloat16 *__restrict s = b;
    bfloat16 *__restrict d = hb + {{ 256 * tq }} * kt;
    for (int i = 0; i < {{ 8 * tq }}; i++) {
        aie::store_v(d, aie::load_v<32>(s));
        s += 32;
        d += 32;
    }
}

{% macro hdr_body() %}
{% if nsb == 8 %}
            unsigned hsh = (sb >> 2) << 1, ssh = (sb & 1) << 4;
            bfloat16 *__restrict o = ws + (((2 * sb) & 3) * 8 + (sb >> 1)) * {{ 32 * tq }} + t * 32;
            {{ p }}_plane8(aie::load_v<128>((const uint4 *)(b + {{ p }}_WOFF(t, sb))).unpack(),
                           aie::load_v<128>((const uint4 *)(b + {{ p }}_WOFF(t, 16 + (sb & 3)))).unpack(),
                           aie::load_v<16>((const uint32 *)(b + {{ p }}_WOFF(t, 24 + (sb >> 1)))), hsh, ssh, o);
            {{ p }}_plane8(aie::load_v<128>((const uint4 *)(b + {{ p }}_WOFF(t, 8 + sb))).unpack(),
                           aie::load_v<128>((const uint4 *)(b + {{ p }}_WOFF(t, 20 + (sb & 3)))).unpack(),
                           aie::load_v<16>((const uint32 *)(b + {{ p }}_WOFF(t, 28 + (sb >> 1)))), hsh, ssh, o + {{ 1024 * tq }});
{% else %}
            // groups 8 sb .. 8 sb + 7 -> ws [group / 2][tile][group % 2][row 16]
            unsigned hsh = (sb >> 1) << 1, ssh = (sb & 1) << 4;
            bfloat16 *__restrict o = ws + sb * {{ 128 * tq }} + t * 32;
            const int ws_ = 12 + (sb >> 1), wm = 14 + (sb >> 1);
            auto [d0, d1] = aie::interleave_zip(aie::load_v<8>((const uint32 *)(b + {{ p }}_HW(t, ws_, 0))),
                                                aie::load_v<8>((const uint32 *)(b + {{ p }}_HW(t, ws_, 1))), 1);
            {{ p }}_plane8({{ p }}_rows16(b + {{ p }}_HW(t, sb, 0), b + {{ p }}_HW(t, sb, 1)),
                           {{ p }}_rows16(b + {{ p }}_HW(t, 8 + (sb & 1), 0), b + {{ p }}_HW(t, 8 + (sb & 1), 1)),
                           aie::concat(d0, d1), hsh, ssh, o);
            auto [m0, m1] = aie::interleave_zip(aie::load_v<8>((const uint32 *)(b + {{ p }}_HW(t, wm, 0))),
                                                aie::load_v<8>((const uint32 *)(b + {{ p }}_HW(t, wm, 1))), 1);
            {{ p }}_plane8({{ p }}_rows16(b + {{ p }}_HW(t, 4 + sb, 0), b + {{ p }}_HW(t, 4 + sb, 1)),
                           {{ p }}_rows16(b + {{ p }}_HW(t, 10 + (sb & 1), 0), b + {{ p }}_HW(t, 10 + (sb & 1), 1)),
                           aie::concat(m0, m1), hsh, ssh, o + {{ 128 * tq * nsb }});
{% endif %}
{% endmacro %}
{% if dqb %}
// super-block sb's records of the 4 tiles (GemmQd expands them before the super-block's k-tiles: no
// long pause of the dequantizing core at a K chunk's start)
void {{ p }}_hdr_sb(const uint8_t *__restrict b, bfloat16 *__restrict ws, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tq }}; t++) {
{{ hdr_body() }}
    }
}
{% else %}
void {{ p }}_hdr_q4k(const uint8_t *__restrict b, bfloat16 *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if nsb == 8 %}
    for (int t = 0; t < {{ tq }}; t++)
        AIE_LOOP_NO_UNROLL
        for (int sb = 0; sb < {{ nsb }}; sb++) {
{{ hdr_body() }}
        }
{% else %}
    // per plane (A: sc, d; B: mn, -dmin) and tile: super-blocks (0, 2) on crumb word 8 (& 3, & 12), (1, 3) on word 9;
    // d of sb in half sb % 2 of word 12 + sb / 2 (plane B: words + 4 / + 2, 4 KB / 2 KB further). 920 cycles (the
    // per-super-block form with its transposes of the results: 2246)
    AIE_LOOP_NO_UNROLL
    for (int pl = 0; pl < 2; pl++) {
        const uint8_t *__restrict bl = b + {{ 1024 * tq }} * pl;  // lo words 4 pl ..
        const uint8_t *__restrict bc = b + {{ 512 * tq }} * pl;  // crumb / scale words 8 + 2 pl ..
        bfloat16 *__restrict o = ws + {{ 128 * tq * nsb }} * pl;
        AIE_LOOP_NO_UNROLL
        for (int t = 0; t < {{ tq }}; t++) {
{% for par in range(2) %}
            {
                const aie::vector<uint8, 128> c = {{ p }}_t16(bc + {{ p }}_HW(0, {{ 8 + par }}, 0), bc + {{ p }}_HW(0, {{ 8 + par }}, 1));
{% for sb in (par, par + 2) %}
                {
                    const aie::vector<bfloat16, 64> d =
                        {{ p }}_dh(bc + {{ p }}_HW(0, {{ 12 + sb // 2 }}, 0), bc + {{ p }}_HW(0, {{ 12 + sb // 2 }}, 1), {{ par }});
                    const aie::vector<uint8, 128> l = {{ p }}_t16(bl + {{ p }}_HW(0, {{ sb }}, 0), bl + {{ p }}_HW(0, {{ sb }}, 1));
                    {{ p }}_g4({{ 3 if sb < 2 else 12 }}, l.template extract<64>(0), c.template extract<64>(0), d, o + {{ 128 * tq * sb }},
                               o + {{ 128 * tq * sb + 32 * tq }});
                    {{ p }}_g4({{ 3 if sb < 2 else 12 }}, l.template extract<64>(1), c.template extract<64>(1), d, o + {{ 128 * tq * sb + 64 * tq }},
                               o + {{ 128 * tq * sb + 96 * tq }});
                }
{% endfor %}
            }
{% endfor %}
            bl += 512;
            bc += 512;
            o += 32;
        }
    }
{% endif %}
}
{% endif %}

{% include "gemm/dq_tile_major" %}
"""


# the decode GEMV (``overlay.gemv``)
_MMV = r"""// ---- Q4_K (16 rows): header [row 16][64 B] = sc lo nibbles 16 | mn lo 16 | sc crumbs 8 | mn crumbs 8 | d 4 x bf16 |
// -dmin 4 x bf16 (groups 8 sb + i; crumb nibble n = c(group n) + 4 c(group n + 16)), codes [sb][row 16][128 B] =
// words (s 4, i 8) of 8 consecutive k (k = 256 sb + 32 i + 8 s + j). Scratch: hw [g][w 16][8 rows x 4 B] (1 KB) |
// tab [sb'][g][hl][i][r] (1 KB) | c [sb'][i][g][32] (2 KB)

// Pass 1: the header -> word panels hw
__attribute__((noinline)) void {{ p }}_q4k_hw(const uint8_t *__restrict a, uint8_t *__restrict hw)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 4; t++) {  // (g, half): rows 8 g .. at 512 g, words 8 half .. at 32 half
        v16int32 o0, o1, o2, o3;
        {{ p }}_tr8<64>(a + 512 * (t >> 1) + 32 * (t & 1), o0, o1, o2, o3);
        v16int32 *o = (v16int32 *)(hw + 256 * t);
        o[0] = o0;
        o[1] = o1;
        o[2] = o2;
        o[3] = o3;
    }
}

// Pass 1: the codes of two super-blocks in place, per (sb, g, s) -> panel (s, i) at row 8 g + i's chunk s
// ([sb][g][i][s][32 B])
__attribute__((noinline)) void {{ p }}_q4k_tr(const uint8_t *__restrict src, uint8_t *__restrict dst)
{
    const uint8_t *s = src;
    uint8_t *d = dst;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 16; t++) {  // (sb, g, s): 1 KB blocks [row 8][128 B], chunk s at 32 s
        v16int32 o0, o1, o2, o3;
        {{ p }}_tr8<128>(s, o0, o1, o2, o3);
        {{ p }}_put8<128>(d, o0, o1, o2, o3);
        s += (t & 3) == 3 ? 1024 - 96 : 32;
        d += (t & 3) == 3 ? 1024 - 96 : 32;
    }
}

// Pass 2: per (g, sb) the table TD = d sc ([i][r], sc = lo + 16 crumb through I) -> tab; y[g] += the offsets
// hilo(XS sc) 8 d + hilo(XS mn) (-dmin) (w = d sc (q - 8) + 8 d sc - dmin mn). HI: the second core's super-blocks 2, 3
// (crumbs in bits 2-3){{ "; ys: the second super-block's rows at y + ys (row groups)" if rg else "" }}
template <int HI>
__attribute__((noinline)) void {{ p }}_q4k_tab(const uint8_t *__restrict hw0, const uint8_t *__restrict xs,
                                               bfloat16 *__restrict tab, float *__restrict y{{ ", int ys" if rg else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const v64bfp16ebs8 I = al::eye();
    const v64uint8 msk = broadcast_to_v64uint8((unsigned char)(HI ? 12 : 3));
    al::BIn<> xsp((const bfp16ebs8 *)(xs + {{ XS_SLOT }} * HI));
    const v64bfp16ebs8 X0 = xsp.pop(), X1 = xsp.pop();
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < 2; g++) {
        const uint8_t *hw = hw0 + 512 * g;
        aie::accum<accfloat, 64> Y = aie::zeros<accfloat, 64>();
        // d, -dmin of the pair: word panels 12 + HI, 14 + HI ([row][sb % 2] bf16)
        const aie::vector<bfloat16, 16> dp = aie::load_v<16>((const bfloat16 *)(hw + 384 + 32 * HI));
        const aie::vector<bfloat16, 16> mp = aie::load_v<16>((const bfloat16 *)(hw + 448 + 32 * HI));
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 2; u++) {  // sb = 2 HI + u
            const v64bfp16ebs8 X = u ? X1 : X0;
            v64bfp16ebs8 Sl, Sc, Ml, Mc;
            Sl.exponent = Ml.exponent = {{ p }}_e8(0x85);
            Sc.exponent = Mc.exponent = {{ p }}_e8(HI ? 0x87 : 0x89);
            Sl.mantissa = (v64int8)unpack(*(const v64uint4 *)(hw + 32 * (2 * HI + u)));
            Ml.mantissa = (v64int8)unpack(*(const v64uint4 *)(hw + 128 + 32 * (2 * HI + u)));
            Sc.mantissa = (v64int8)band(unpack(*(const v64uint4 *)(hw + 256 + 32 * u)), msk);
            Mc.mantissa = (v64int8)band(unpack(*(const v64uint4 *)(hw + 320 + 32 * u)), msk);
            const aie::vector<bfloat16, 8> d8 = u ? aie::filter_odd(dp, 1) : aie::filter_even(dp, 1);
            const aie::vector<bfloat16, 8> m8 = u ? aie::filter_odd(mp, 1) : aie::filter_even(mp, 1);
            const aie::vector<bfloat16, 64> D = {{ p }}_rep8v(d8);
            {{ p }}_td(mac_8x8_8x8T(I, 1, Sc, 0, mul_8x8_8x8T(I, 1, Sl, 0)), D, tab + 256 * u + 128 * g);
            Y = {{ p }}_mo(mac_8x8_8x8T(X, 1, Sc, 0, mul_8x8_8x8T(X, 1, Sl, 0)),
                           aie::mul(D, aie::broadcast<bfloat16, 64>(8.0f)).template to_vector<bfloat16>(), Y);
            Y = {{ p }}_mo(mac_8x8_8x8T(X, 1, Mc, 0, mul_8x8_8x8T(X, 1, Ml, 0)), {{ p }}_rep8v(m8), Y);
{% if rg %}
            {{ p }}_yadd(y + 32 * g + u * ys, Y);  // (per super-block: its rows of their own with ys)
            Y = aie::zeros<accfloat, 64>();
{% endif %}
        }
{% if not rg %}
        {{ p }}_yadd(y + 32 * g, Y);
{% endif %}
    }
}

// Pass 3: c_g,sb,i = bf16(C[hi rows] + C[lo rows]) -> c [sb][i][g][32]; C = -8 XS_i + sum_s X(sb, i, s) W_g(sb, s, i)
// (the unsigned codes, centered)
__attribute__((noinline)) void {{ p }}_q4k_mm(const uint8_t *__restrict xp, const uint8_t *__restrict xsum,
                                              const uint8_t *__restrict w, bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    al::BIn<aie_dm_resource::d> xz((const bfp16ebs8 *)xsum);
    const uint8_t __aie_dm_resource_b *__restrict pw = (const uint8_t __aie_dm_resource_b *)w;
    bfloat16 *__restrict pc = c;
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < 2; sb++) {
        const v64bfp16ebs8 Z = xz.pop();
        const int8_t __aie_dm_resource_d *__restrict pn = (const int8_t __aie_dm_resource_d *){{ p }}_nz[0][0];
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int i = 0; i < 8; i++) {
            const v64bfp16ebs8 N = {{ p }}_np(pn);
            pn += 64;
            const v64bfp16ebs8 x0 = xs.pop(), x1 = xs.pop(), x2 = xs.pop(), x3 = xs.pop();
            v64accfloat C0 = mul_8x8_8x8T(Z, 1, N, 1);
            v64accfloat C1 = mul_8x8_8x8T(Z, 1, N, 1);
            C0 = mac_8x8_8x8T(x0, 1, {{ p }}_wu(pw), 0, C0);
            C1 = mac_8x8_8x8T(x0, 1, {{ p }}_wu(pw + 1024), 0, C1);
            C0 = mac_8x8_8x8T(x1, 1, {{ p }}_wu(pw + 32), 0, C0);
            C1 = mac_8x8_8x8T(x1, 1, {{ p }}_wu(pw + 1056), 0, C1);
            C0 = mac_8x8_8x8T(x2, 1, {{ p }}_wu(pw + 64), 0, C0);
            C1 = mac_8x8_8x8T(x2, 1, {{ p }}_wu(pw + 1088), 0, C1);
            C0 = mac_8x8_8x8T(x3, 1, {{ p }}_wu(pw + 96), 0, C0);
            C1 = mac_8x8_8x8T(x3, 1, {{ p }}_wu(pw + 1120), 0, C1);
            aie::store_v(pc, {{ p }}_sum2(C0));
            aie::store_v(pc + 32, {{ p }}_sum2(C1));
            pw += 128;
            pc += 64;
        }
        pw += 1024;
    }
}

// super-blocks 2 HI, 2 HI + 1 of a Q4_K tile
template <int HI>
__attribute__((noinline)) void {{ p }}_q4k(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr{{ ", int ys" if rg else "" }})
{
    uint8_t *codes = a + 1024 + 4096 * HI;
    bfloat16 *tab = (bfloat16 *)(scr + 1024);
    bfloat16 *c = (bfloat16 *)(scr + 2048);
    {{ p }}_q4k_hw(a, scr);
    {{ p }}_q4k_tr(codes, codes);
    {{ p }}_q4k_tab<HI>(scr, xs, tab, y{{ ", ys" if rg else "" }});
    {{ p }}_q4k_mm(xp, xs + {{ XS_SLOT }} * HI, codes, c);
    {{ p }}_sc(c, tab, y, 128, 8, 64);
    {{ p }}_sc(c + 512, tab + 256, y{{ " + ys" if rg else "" }}, 128, 8, 64);
}
"""


class Q4KSlot(Slot):
    formats = (TileFormat("Q4_K", 3, 144, 16, 8, 128, 16, planes=4, ws_planes=2, mmv="q4k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q4_K": _GEMM, "mmv/Q4_K": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        g = 8 * nsb  # groups of the chunk (g = 8 sb + i)

        def six(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            v = v.reshape(R, g)
            c = v >> 4
            return _nibbles(v & 15), _nibbles(c[:, :g // 2] + 4 * c[:, g // 2:])

        (sl, sh), (ml, mh) = six(f["sc"]), six(f["mn"])
        # every vector load naturally aligned: the nibbles first, then the smaller parts
        per_row = np.concatenate([sl, ml, sh, mh, _bf16_bytes(f["d"]), _bf16_bytes(-f["dmin"])], axis=1)
        return [per_row, sb_major(_nibbles(f["q"][:, _word_order(nsb)]), R, nsb)]

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        return rows == 16

    def gemm_q_header(self, op: object, kernel: object, B_l1_ty: object, ws_ty: object,
                      st_ty: object) -> tuple[object, object]:
        # header slots collected in the staging buffer, then expanded
        return kernel("hcp", [B_l1_ty, st_ty, np.int32]), kernel("hdr_q4k", [st_ty, ws_ty])

    gemm_q_collects = True

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        # header units collected, then expanded per super-block
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty)
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True, unit=False)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)


SLOT = Q4KSlot()
