"""Q2_K tiles (shared layout and prefill kernels: ``k23``).

Code words per super-block (16 words, 2 planes): word j = sub-block j, nibble b = c(16 j + b) + 4 c(16 j + 8 + b).
Header words per row (20 B per super-block): [sb][sc h 0, sc h 1, mn h 0, mn h 1]
(nibble s: sub-block 8 h + s) | [sb]: bf16 d, bf16 dmin."""

from typing import ClassVar

import numpy as np

from . import k23
from .common import Slot, TileFormat


def header_words(nsb: int) -> int:
    """Header words per row."""
    return 5 * nsb


# GemmQd: the planar dequant (the header kernels: gemm/k23)
_GEMM = r"""{% include "gemm/k23" %}
// super-block sb's scales of the pass's rows: ws [16-row block][h 2][A rows 0 .. 7, A 8 .. 15, B 0 .. 7, B 8 .. 15][s 8]
// [r 8] = bf16(d sc), bf16(-dmin mn) of sub-block 8 h + s (the decode's roundings; the MMUL against I turns a word's
// nibbles [r][s] into lanes [s][r]; zero for the padding rows)
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const {{ p }}_BV I = al::eye();
    {{ p }}_BV H;
    H.exponent = {{ p }}_e8(0x85);  // (value = mantissa)
    const uint32_t *__restrict hw = hb + ({{ hpad }} + 4 * sb) * {{ n }};  // sc h 0, sc h 1, mn h 0, mn h 1
    const uint32_t *__restrict dw = hb + ({{ hpad + 4 * nsb }} + sb) * {{ n }};  // d | dmin
    bfloat16 *__restrict o = ws;
    int cq = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < {{ n // 8 }}; q++) {  // (16-row block, its rows 0 .. 7 or 8 .. 15) = (q / 2, q % 2)
        aie::vector<bfloat16, 64> d64, m64;
        {{ p }}_d64(dw + 8 * q, d64, m64);
        m64 = aie::neg(m64);
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++) {  // sc h 0, sc h 1, mn h 0, mn h 1
            H.mantissa = {{ p }}_nib(hw + u * {{ n }} + 8 * q);
            {{ p }}_scale64(mul_8x8_8x8T(I, 0, H, 0), u < 2 ? d64 : m64, o + 256 * (u & 1) + 128 * (u >> 1));
        }
        o = add_2d_ptr(o, 448, 1, cq, 64);
    }
}

// k-tile kt of super-block sb -> {{ n_mac }} x 64 bfp16 panels in the MAC pop order [16-row block][k-step 2 jj + a][8-row
// half]: word jj = sub-block j = 4 kt + jj, nibble b = c(16 j + b) + 4 c(16 j + 8 + b) (a = 0: n & 3, a = 1: n & 12).
// w = A c + B = A (128 + c) + (B - 128 A) (a = 1: the byte 4 c zipped with 0x42 is 32 + c, + (B - 32 A)), exact in fp32,
// rounded once; A, B per lane (_x8)
void {{ p }}_dq(const uint8_t *__restrict b, const bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt{{ half_arg }})
{
{% if "ev_dq" in ablate or "ev_kt" in ablate %}
    event0();
{% endif %}
{% if "dq" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if halves > 1 %}
    b += half * {{ 4 * n_mac }};  // (the half's 16-row blocks)
    ws += half * {{ 32 * n_mac }};
{% endif %}
    const uint8_t *__restrict pw = b;
    const bfloat16 __aie_dm_resource_c *__restrict ps =  // (sub-block j = 4 kt + jj: h = kt / 2, s = 4 (kt % 2) + jj)
        (const bfloat16 __aie_dm_resource_c *)(ws + 256 * (kt >> 1) + 32 * (kt & 1));
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)stage);
    const aie::vector<uint8, 128> m3 = aie::broadcast<uint8, 128>(3), m12 = aie::broadcast<uint8, 128>(12);
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43), c42 = aie::broadcast<uint8, 64>(0x42);
    const aie::vector<bfloat16, 64> k128 = aie::broadcast<bfloat16, 64>(-128.0f);
    const aie::vector<bfloat16, 64> k96 = aie::broadcast<bfloat16, 64>(96.0f);
    const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
    int cw = 0, cs = 0;
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_REG_REWRITE_SWPAWARE  // (II 29 per 4 panels)
    for (int it = 0; it < {{ n_mac // 4 }}; it++) {  // (16-row block, sub-block jj) = (it / 4, it % 4)
        const aie::vector<uint8, 128> v = aie::unpack(aie::load_v<128>((const uint4 __aie_dm_resource_a *)pw));
        const aie::vector<uint8, 128> q0 = aie::bit_and(v, m3), q1 = aie::bit_and(v, m12);
        // per 8-row half: A, C = B - 128 A per lane; the panels in pop order (a = 1 of the first half waits in w10)
        {{ p }}_BV w10;
        {
            const aie::vector<bfloat16, 64> A = {{ p }}_x8(aie::load_v<8>(ps));
            const aie::accum<accfloat, 64> C = aie::mac(aie::mul(A, k128), {{ p }}_x8(aie::load_v<8>(ps + 128)), one);
            out.push(aie::mac(C, al::zipw(q0.template extract<64>(0), c43), A).template to_vector<bfp16ebs8>());
            w10 = aie::mac(aie::mac(C, A, k96), al::zipw(q1.template extract<64>(0), c42), A).template to_vector<bfp16ebs8>();
        }
        {
            const aie::vector<bfloat16, 64> A = {{ p }}_x8(aie::load_v<8>(ps + 64));
            const aie::accum<accfloat, 64> C = aie::mac(aie::mul(A, k128), {{ p }}_x8(aie::load_v<8>(ps + 192)), one);
            out.push(aie::mac(C, al::zipw(q0.template extract<64>(1), c43), A).template to_vector<bfp16ebs8>());
            out.push(w10);
            out.push(aie::mac(aie::mac(C, A, k96), al::zipw(q1.template extract<64>(1), c42), A).template to_vector<bfp16ebs8>());
        }
        // next sub-block's word, or the next 16-row block's first
        pw = add_2d_byte(pw, {{ 64 - 12 * n }}, 3, cw, {{ 4 * n }});
        ps = add_2d_ptr(ps, 488, 3, cs, 8);
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


# the decode GEMV (``overlay.gemv``; the realignment and the scaling: ``mmv/k23``)
_MMV = r"""// ---- Q2_K ({{ R2 }} rows, {{ G2 }} row groups; row groups past R: junk): header words [sb][sc h 0, sc h 1, mn h 0, mn h 1]
// (nibble s: sub-block 8 h + s) | [sb] d | dmin | codes [sb][word j 16] (word j = sub-block j: nibble b = c(16 j + b) +
// 4 c(16 j + 8 + b)), word-major. Core H: super-blocks 2 H, 2 H + 1 (the K-quants' split); per super-block u the integer
// scales and the offsets (_tab: y = d (sum c sc + 2 XS16 sc) - dmin XS16 mn, the offsets' sums on the MMUL), the
// sub-block sums c (_mm: the codes centered, q - 2; every row group per sub-block: independent chains), then d (sum c sc
// + 2 XS16 sc) (_k23sc). Scratch: hw [sc / mn 8 | d 2][128 B] (1280 B) | sct [g 4][j 16][r] | yu [g 4][32] fp32 | dg
// [g 4][r] | c [j 16][g 4][32]

// Super-block u: its integer scales sc (I against the rows' nibbles: [j][r]) -> sct [g][j][r], d -> dg, yu[g] = 2 XS16 sc
// (fp32, the hi and lo rows summed), y[g] += hilo(XS16 mn) (-dmin) (xs16: the super-block's 16-k sum panels)
__attribute__((noinline)) void {{ p }}_q2k_tab(const uint8_t *__restrict hw, const uint8_t *__restrict xs16,
                                               bfloat16 *__restrict sct, float *__restrict yu, bfloat16 *__restrict dg,
                                               float *__restrict y, int u)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const v64bfp16ebs8 I = al::eye();
    al::BIn<> xsp((const bfp16ebs8 *)xs16);
    const v64bfp16ebs8 X0 = xsp.pop(), X1 = xsp.pop();
    const uint8_t __aie_dm_resource_a *ps = (const uint8_t __aie_dm_resource_a *)(hw + 512 * u);
    const uint8_t __aie_dm_resource_b *pd = (const uint8_t __aie_dm_resource_b *)(hw + 1024 + 128 * u);
    bfloat16 *__restrict pt = sct, *__restrict pg = dg;
    float *__restrict pu = yu, *__restrict py = y;
    v64bfp16ebs8 S0, S1, M0, M1;
    S0.exponent = S1.exponent = M0.exponent = M1.exponent = {{ p }}_e8(0x85);
    AIE_LOOP_NO_UNROLL  // (pointers walked: recomputed ones chained the y store to the next group's loads, RecMII 39)
    for (int g = 0; g < 4; g++) {
        S0.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_a *)ps);
        S1.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_a *)(ps + 128));
        M0.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_a *)(ps + 256));
        M1.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_a *)(ps + 384));
        ps += 32;
        const aie::vector<bfloat16, 16> dm = aie::load_v<16>((const bfloat16 __aie_dm_resource_b *)pd);  // [row 8][d, dmin]
        pd += 32;
        aie::store_v(pg, aie::filter_even(dm, 1));
        pg += 8;
        aie::store_v(pt, aie::accum<accfloat, 64>(mul_8x8_8x8T(I, 1, S0, 0)).template to_vector<bfloat16>());
        aie::store_v(pt + 64, aie::accum<accfloat, 64>(mul_8x8_8x8T(I, 1, S1, 0)).template to_vector<bfloat16>());
        pt += 128;
        const aie::vector<float, 64> ms = aie::accum<accfloat, 64>(mac_8x8_8x8T(X1, 1, S1, 0, mul_8x8_8x8T(X0, 1, S0, 0))).template to_vector<float>();
        const aie::vector<float, 32> m2 = aie::add(ms.template extract<32>(0), ms.template extract<32>(1));
        aie::store_v(pu, aie::add(m2, m2));
        pu += 32;
        const aie::vector<bfloat16, 64> Dm = aie::mul({{ p }}_rep8v(aie::filter_odd(dm, 1)), aie::broadcast<bfloat16, 64>(-1.0f)).template to_vector<bfloat16>();
        {{ p }}_yadd(py, {{ p }}_mo(mac_8x8_8x8T(X1, 1, M1, 0, mul_8x8_8x8T(X0, 1, M0, 0)), Dm, aie::zeros<accfloat, 64>()));
        py += 32;
    }
}

// c_j,g = bf16(C[hi rows] + C[lo rows]) -> c [j][g 4][32] for the 16 sub-blocks j of a super-block, every row group:
// C = x(2 j) (w & 3 - 2) + x(2 j + 1) ((w & 12) / 4 - 2) (w: word j's row group, realigned from the tile: o the super-block's
// first word; xp: its 32 panels; the -2: x against a constant panel, once per sub-block for the four groups); the panels'
// exponents set outside the loop (a code panel: one vband)
__attribute__((noinline)) void {{ p }}_q2k_mm(const uint8_t *__restrict xp, const uint8_t *w, bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    const v64uint8 m3 = broadcast_to_v64uint8((unsigned char)3), m12 = broadcast_to_v64uint8((unsigned char)12);
    v64bfp16ebs8 N2, A0, A1, A2, A3, B0, B1, B2, B3;  // (the zero point, the codes & 3, the codes & 12 per row group)
    N2.mantissa = broadcast_to_v64int8(-2);
    N2.exponent = {{ p }}_e8(0x85);
    A0.exponent = A1.exponent = A2.exponent = A3.exponent = {{ p }}_e8(0x85);
    B0.exponent = B1.exponent = B2.exponent = B3.exponent = {{ p }}_e8(0x83);
    uintptr_t o = (uintptr_t)w;
    bfloat16 *__restrict pc = c;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int s = 0; s < 16; s++) {
        const v64bfp16ebs8 x0 = xs.pop(), x1 = xs.pop();
        v16int32 lo, hi;
        {{ p }}_k23w2(o, {{ 4 * R2 }}, lo, hi);
        const v64accfloat Z = mac_8x8_8x8T(x1, 1, N2, 1, mul_8x8_8x8T(x0, 1, N2, 1));
{% for g in range(4) %}
        {
            const v64uint8 q = unpack(extract_v64uint4((v128uint4){{ "lo" if g < 2 else "hi" }}, {{ g % 2 }}));
            A{{ g }}.mantissa = (v64int8)band(q, m3);
            B{{ g }}.mantissa = (v64int8)band(q, m12);
            aie::store_v(pc + {{ 32 * g }}, {{ p }}_sum2(mac_8x8_8x8T(x1, 1, B{{ g }}, 0, mac_8x8_8x8T(x0, 1, A{{ g }}, 0, Z))));
        }
{% endfor %}
        o += {{ 4 * R2 }};
        pc += 128;
    }
}

// super-blocks 2 H, 2 H + 1 of a Q2_K tile: y[g][32] (rows 8 g ..) += their rows against xp (their 64 panels)
template <int H>
__attribute__((noinline)) void {{ p }}_q2k(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    uint8_t *hw = scr;
    bfloat16 *sct = (bfloat16 *)(scr + 1280), *dg = (bfloat16 *)(scr + 2816), *c = (bfloat16 *)(scr + 2880);
    float *yu = (float *)(scr + 2304);
    {{ p }}_k23al(a + {{ 4 * R2 * 8 }} * H, hw, 8, {{ 4 * R2 }});  // (sc / mn of the two)
    {{ p }}_k23al(a + {{ 4 * R2 * 16 }} + {{ 4 * R2 * 2 }} * H, hw + 1024, 2, {{ 4 * R2 }});  // (d | dmin)
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        const uint8_t *xs16 = xs + {{ XS_SLOT }} * (2 + 2 * H + u);
        const uint8_t *w = a + {{ 4 * R2 * NW2 }} + {{ 4 * R2 * 16 }} * (2 * H + u);
        {{ p }}_q2k_tab(hw, xs16, sct, yu, dg, y, u);
        {{ p }}_q2k_mm(xp + {{ 32 * PANEL }} * u, w, c);
        {{ p }}_k23sc(c, sct, yu, dg, y);
    }
}
"""

class Q2KSlot(k23.PlanarK23, Slot):
    formats = (TileFormat("Q2_K", 7, 84, 32, 1, 64, 20, planes=2,
                          word_major=True, dq_row_cycles=9.0, mmv="q2k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q2_K": _GEMM, "mmv/Q2_K": _MMV}
    ws_bytes = 512  # (per 8-row block: A, B of 16 sub-blocks)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        hw = np.zeros((R, header_words(nsb)), np.uint32)
        h = np.stack([f["sc"], f["mn"]], axis=2).reshape(R, nsb, 2, 2, 8)  # [sb][sc | mn][h][s]
        hw[:, :4 * nsb] = k23.words(h).reshape(R, 4 * nsb)
        hw[:, 4 * nsb:5 * nsb] = k23.bf16_u16(f["d"]) | (k23.bf16_u16(f["dmin"]) << 16)
        c = f["q"].reshape(R, nsb, 16, 2, 8)  # [j][a][b]
        return k23.tile_parts(hw, k23.words(c[..., 0, :] + 4 * c[..., 1, :]), R, nsb)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        v = super().mmv_vars(fmt, tile_bytes, nsb)
        assert 24 < v["R2"] <= 32, v  # (4 row groups: the kernels' layout)
        return {**v, "NW2": header_words(nsb)}


SLOT = Q2KSlot()
