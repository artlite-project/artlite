"""Q3_K tiles (shared layout and prefill kernels: ``k23``).

Code words per super-block (24 words, 3 planes): words 3 p + t of quad p (k 32 p .. 32 p + 31: blocks t = 0 .. 3,
sub-blocks 2 p, 2 p + 1): nibbles b of A = q(t 0) + 8 q3_0, B = q(t 1) + 8 q3_1, C = q(t 2) + 8 q3_2, with
q(t 3) = q3_0 + 2 q3_1 + 4 q3_2 (signed unpack: B & 16 = 16 q3_1, C & 32 = 32 q3_2).
Header words per row (14 B per super-block): [sb][h 2]: the low nibbles of the 6-bit
scales u = sc + 32 of sub-blocks 8 h .. 8 h + 7 | [sb]: the high crumbs, nibble s = c(s) + 4 c(s + 8) | [sb pair]:
bf16 d of both."""

from typing import ClassVar

import numpy as np

from . import k23
from .common import DqKernels, Slot, TileFormat, collected_header, dq_kernel


def header_words(nsb: int) -> int:
    """Header words per row."""
    return (7 * nsb + 1) // 2


# GemmQd: the planar dequant (the header kernels: gemm/k23)
_GEMM = r"""{% include "gemm/k23" %}
// super-block sb's scales of the pass's rows: ws [16-row block][h 2][rows 0 .. 7, 8 .. 15][s 8][r 8] = bf16(d (sc - 32))
// of sub-block 8 h + s (the decode's rounding; as its _q3k_hdr: the MMUL against I sums the low nibbles, the crumbs
// (& 3 at exponent 137: 16 c; & 12 at 135) and a constant -32 panel in lanes [s][r]; zero for the padding rows)
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const {{ p }}_BV I = al::eye();
    {{ p }}_BV L, C0, C1, N;
    L.exponent = al::e8(0x85);
    C0.exponent = al::e8(0x89);  // c of sub-blocks 0 .. 7: x 16
    C1.exponent = al::e8(0x87);  // 4 c of 8 .. 15: x 4
    N.mantissa = broadcast_to_v64int8(-1);  // -32
    N.exponent = al::e8(0x8A);
    const v64int8 m3 = broadcast_to_v64int8(3), m12 = broadcast_to_v64int8(12);
    const uint32_t *__restrict lw = hb + ({{ hpad }} + 2 * sb) * {{ n }};  // low nibbles h 0, h 1
    const uint32_t *__restrict cw = hb + ({{ hpad + 2 * nsb }} + sb) * {{ n }};  // crumbs
    const uint32_t *__restrict dw = hb + ({{ hpad + 3 * nsb }} + (sb >> 1)) * {{ n }};  // d of the super-block pair
    const int dhi = sb & 1;
    bfloat16 *__restrict o = ws;
    int cq = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < {{ n // 8 }}; q++) {  // (16-row block, its rows 0 .. 7 or 8 .. 15) = (q / 2, q % 2)
        aie::vector<bfloat16, 64> d0, d1;
        {{ p }}_d64(dw + 8 * q, d0, d1);
        const aie::vector<bfloat16, 64> d64 = dhi ? d1 : d0;
        const v64int8 c = {{ p }}_nib(cw + 8 * q);
        L.mantissa = {{ p }}_nib(lw + 8 * q);
        C0.mantissa = band(c, m3);
        {{ p }}_scale64(mac_8x8_8x8T(I, 0, N, 1, mac_8x8_8x8T(I, 0, C0, 0, mul_8x8_8x8T(I, 0, L, 0))), d64, o);
        L.mantissa = {{ p }}_nib(lw + {{ n }} + 8 * q);
        C1.mantissa = band(c, m12);
        {{ p }}_scale64(mac_8x8_8x8T(I, 0, N, 1, mac_8x8_8x8T(I, 0, C1, 0, mul_8x8_8x8T(I, 0, L, 0))), d64, o + 128);
        o = add_2d_ptr(o, 192, 1, cq, 64);
    }
}

// k-tile kt of super-block sb -> {{ n_mac }} x 64 bfp16 panels in the MAC pop order [16-row block][k-step 4 qq + t][8-row
// half]: block quad p = 2 kt + qq, its words 3 qq + t = the signed nibbles A, B, C = q(t) + 8 bit_t(q(3)) of blocks t =
// 0 .. 2 (sub-block 2 p: t = 0, 1; 2 p + 1: t = 2, 3). Two loops through cb: the code bytes q(t) = A & 7, B & 7, C & 7,
// 8 q(3) = (A & 8) | (B & 16) | (C & 32) and the scales S per lane; then w = S (q - 4) = S (128 + q) - 132 S for t < 3,
// (S / 8) (128 + 8 q(3)) - 20 S for t = 3, exact in fp32, rounded once (one loop does not pipeline, its live registers
// exceed the file: 590 instead of ~750 cycles per 64 rows)
void {{ p }}_dq(const uint8_t *__restrict b, const bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt{{ half_arg }}, uint8_t *__restrict cb)
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
    ws += half * {{ 16 * n_mac }};  // (its 16-row blocks' scales)
{% endif %}
    {  // cb [16-row block, quad qq]: the code bytes [t 4][8-row half 2][64 B], the scales per lane [A0, A1, E0, E1][64 bf16]
        const uint8_t *__restrict pw = b;
        const bfloat16 __aie_dm_resource_c *__restrict ps =  // (sub-blocks j = 4 kt + 2 qq, + 1: h = kt / 2, s = 4 (kt % 2) + 2 qq)
            (const bfloat16 __aie_dm_resource_c *)(ws + 128 * (kt >> 1) + 32 * (kt & 1));
        uint8_t *__restrict pc = cb;
        const aie::vector<int8, 64> m7 = aie::broadcast<int8, 64>(7), m8 = aie::broadcast<int8, 64>(8);
        const aie::vector<int8, 64> m16 = aie::broadcast<int8, 64>(16), m32 = aie::broadcast<int8, 64>(32);
        int cw = 0, cs = 0;
        // (II 28, the mv slot's bound; the default SWP-aware register rewrite: 34)
        AIE_PREPARE_FOR_POSTPIPELINING AIE_LOOP_REG_REWRITE_BASIC
        for (int it = 0; it < {{ n_mac // 8 }}; it++) {  // (16-row block, quad qq) = (it / 2, it % 2)
{% for h in range(2) %}
            {
                const aie::vector<int8, 64> wa = aie::unpack(aie::load_v<64>((const int4 __aie_dm_resource_a *)(pw + {{ 32 * h }})));
                const aie::vector<int8, 64> wb = aie::unpack(aie::load_v<64>((const int4 __aie_dm_resource_a *)(pw + {{ 4 * n + 32 * h }})));
                const aie::vector<int8, 64> wc = aie::unpack(aie::load_v<64>((const int4 __aie_dm_resource_a *)(pw + {{ 8 * n + 32 * h }})));
                aie::store_v(pc + {{ 64 * h }}, aie::bit_and(wa, m7).template cast_to<uint8>());
                aie::store_v(pc + {{ 128 + 64 * h }}, aie::bit_and(wb, m7).template cast_to<uint8>());
                aie::store_v(pc + {{ 256 + 64 * h }}, aie::bit_and(wc, m7).template cast_to<uint8>());
                aie::store_v(pc + {{ 384 + 64 * h }},
                             aie::bit_or(aie::bit_or(aie::bit_and(wa, m8), aie::bit_and(wb, m16)), aie::bit_and(wc, m32)).template cast_to<uint8>());
            }
{% endfor %}
{% for u, o in ((0, 0), (1, 64), (2, 8), (3, 72)) %}
            aie::store_v((bfloat16 *)(pc + {{ 512 + 128 * u }}), al::expand8(aie::load_v<8>(ps + {{ o }})));
{% endfor %}
            pw = add_2d_byte(pw, {{ 64 - 12 * n }}, 1, cw, {{ 12 * n }});  // (the next quad's words, or the next block's)
            ps = add_2d_ptr(ps, 240, 1, cs, 16);
            pc += 1024;
        }
    }
    const uint8_t __aie_dm_resource_a *__restrict pc = (const uint8_t __aie_dm_resource_a *)cb;
    const bfloat16 __aie_dm_resource_b *__restrict pv = (const bfloat16 __aie_dm_resource_b *)(cb + 512);  // (the scales)
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> out((bfp16ebs8 *)stage);
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
    const aie::vector<bfloat16, 64> k132 = aie::broadcast<bfloat16, 64>(-132.0f);
    const aie::vector<bfloat16, 64> k20 = aie::broadcast<bfloat16, 64>(-20.0f);
    const aie::vector<bfloat16, 64> k8 = aie::broadcast<bfloat16, 64>(0.125f);
    // (no pragma: II 34 per 8 panels, with AIE_PREPARE_FOR_POSTPIPELINING 35; the 8 pushes' chain: RecMII 21-24)
    for (int it = 0; it < {{ n_mac // 8 }}; it++) {
{% for t in range(4) %}
{% for h in range(2) %}
        {
            const aie::vector<bfloat16, 64> S = aie::load_v<64>(pv + {{ 64 * (2 * (t // 2) + h) }});
{% if t < 3 %}
            out.push(aie::mac(aie::mul(S, k132), al::zipw(aie::load_v<64>(pc + {{ 128 * t + 64 * h }}), c43), S).template to_vector<bfp16ebs8>());
{% else %}
            out.push(aie::mac(aie::mul(S, k20), al::zipw(aie::load_v<64>(pc + {{ 384 + 64 * h }}), c43),
                              aie::mul(S, k8).template to_vector<bfloat16>()).template to_vector<bfp16ebs8>());
{% endif %}
        }
{% endfor %}
{% endfor %}
        pc += 1024;
        pv += 512;
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


# the decode GEMV (``overlay.gemv``; the realignment and the scaling: ``mmv/k23``)
_MMV = r"""// ---- Q3_K ({{ R3 }} rows, 3 row groups; rows past R: junk): header words [sb][h 2] the low nibbles of u = sc + 32 | [sb] the
// high crumbs (nibble s = c(s) + 4 c(s + 8)) | [sb pair] d | codes [sb][word 3 p + t 24] (quad p: the signed nibbles A, B,
// C = q(t) + 8 bit_t(q(3)) of blocks t = 0 .. 2), word-major. Core H: super-blocks 2 H, 2 H + 1; per super-block u its
// signed integer scales u - 32 (_tab), the sub-block sums c (_mm: w = d sc (q - 4); a row group per pass), then d sum c
// sc (_k23sc). Scratch: hw [lo 4 | crumbs, d: 5 words][128 B] (1152 B) | sct [g 4][j 16][r]
// | yu [g 4][32] (zeros) | dg [g 4][r] | c [j 16][g 4][32] | (6912) the masks [2][4][64]

// Super-block u's scales u - 32 of the 16 sub-blocks: I against the low nibbles, the crumbs (j < 8: & 3 at 16, j >= 8:
// & 12 at 4) and a -32 panel ([s][r]) -> sct [g][j][r]; d -> dg; yu = 0 (both halves per call: their exponents and masks
// constants, a runtime choice around the loop crashes Peano's PreLegalizerCombiner; the groups unrolled, the d pick a
// shuffle by u: no branch)
__attribute__((noinline)) void {{ p }}_q3k_tab(const uint8_t *__restrict hw, bfloat16 *__restrict sct, float *__restrict yu,
                                               bfloat16 *__restrict dg, int H, int u)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const v64bfp16ebs8 I = al::eye();
    v64bfp16ebs8 L0, L1, C0, C1, N;
    L0.exponent = L1.exponent = al::e8(0x85);
    C0.exponent = al::e8(0x89);
    C1.exponent = al::e8(0x87);
    N.mantissa = broadcast_to_v64int8(-1);  // (-32)
    N.exponent = al::e8(0x8A);
    const v64int8 m3 = broadcast_to_v64int8(3), m12 = broadcast_to_v64int8(12);
    const uint8_t __aie_dm_resource_a *pl = (const uint8_t __aie_dm_resource_a *)(hw + 256 * u);
    const uint8_t __aie_dm_resource_b *pc = (const uint8_t __aie_dm_resource_b *)(hw + 512 + 128 * u);
    const uint8_t __aie_dm_resource_b *pd = (const uint8_t __aie_dm_resource_b *)(hw + 512 + 128 * (4 - H));
    bfloat16 *__restrict pt = sct, *__restrict pg = dg;
    float *__restrict pu = yu;
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < 3; g++) {
        L0.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_a *)pl);
        L1.mantissa = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_a *)(pl + 128));
        pl += 32;
        const v64int8 cr = (v64int8)unpack(*(const v64uint4 __aie_dm_resource_b *)pc);
        pc += 32;
        C0.mantissa = band(cr, m3);
        C1.mantissa = band(cr, m12);
        const aie::vector<bfloat16, 16> dd = aie::load_v<16>((const bfloat16 __aie_dm_resource_b *)pd);  // [row 8][d 2 H, 2 H + 1]
        pd += 32;
        aie::store_v(pg, aie::filter_even(aie::shuffle_down(dd, u), 1));
        pg += 8;
        aie::store_v(pt, aie::accum<accfloat, 64>(mac_8x8_8x8T(I, 1, N, 1, mac_8x8_8x8T(I, 1, C0, 0, mul_8x8_8x8T(I, 1, L0, 0)))).template to_vector<bfloat16>());
        aie::store_v(pt + 64, aie::accum<accfloat, 64>(mac_8x8_8x8T(I, 1, N, 1, mac_8x8_8x8T(I, 1, C1, 0, mul_8x8_8x8T(I, 1, L1, 0)))).template to_vector<bfloat16>());
        pt += 128;
        aie::store_v(pu, aie::zeros<float, 32>());
        pu += 32;
    }
}

// c_j,g = bf16(C[hi rows] + C[lo rows]) -> c [j][g 4][32] for the 16 sub-blocks j of a super-block, row group by row
// group: per quad q (sub-blocks 2 q, 2 q + 1) C(2 q) = x(4 q) (A & 7 - 4) + x(4 q + 1) (B & 7 - 4), C(2 q + 1) = x(4 q + 2)
// (C & 7 - 4) + x(4 q + 3) (((A & 8) | (B & 16) | (C & 32)) / 8 - 4) (the -4: x against a constant panel; A, B, C: the
// quad's words' row group: the low 32 B of 64-byte pops from w + 32 g, each after a fill and followed by a skip of 4 R - 64
// bytes: the load FIFO realigns the words; w the super-block's first word, xp its 32 panels); the panels' exponents set
// outside the loops; the masks loaded from mt (two copies of 7, 8, 16, 32, walked: in registers they left the loop 3
// spills and II 20, loaded II 13). (Row group 2's pops read up to 104 B past the tile: the A objects lie between other
// buffers)
__attribute__((noinline)) void {{ p }}_q3k_mm(const uint8_t *__restrict xp, const uint8_t *w, bfloat16 *__restrict c,
                                              int8_t *__restrict mt)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
{% for cp in range(2) %}
{% for v in (7, 8, 16, 32) %}
    aie::store_v(mt + {{ 256 * cp + 64 * loop.index0 }}, aie::broadcast<int8, 64>({{ v }}));
{% endfor %}
{% endfor %}
    v64bfp16ebs8 N4, W0, W1, W2, W3;  // (the zero point; blocks 0 .. 3)
    N4.mantissa = broadcast_to_v64int8(-4);
    N4.exponent = al::e8(0x85);
    W0.exponent = W1.exponent = W2.exponent = al::e8(0x85);
    W3.exponent = al::e8(0x82);
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < 3; g++) {
        al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
        v128int4 __aie_dm_resource_b *pw = (v128int4 __aie_dm_resource_b *)(w + 32 * g);
        fifo_state_t fs;
        fifo_ld_reset(pw, fs);
        bfloat16 *__restrict pc = c + 32 * g;
        const int8_t __aie_dm_resource_c *pm = (const int8_t __aie_dm_resource_c *)mt;
        int cm = 0;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int q = 0; q < 8; q++) {
            const v64bfp16ebs8 x0 = xs.pop(), x1 = xs.pop(), x2 = xs.pop(), x3 = xs.pop();
            fifo_ld_fill(pw, fs);
            const v64int8 a = unpack(extract_v64int4(fifo_ld_pop_1d_byte(pw, fs, {{ 4 * R3 - 64 }}), 0));
            fifo_ld_fill(pw, fs);
            const v64int8 b = unpack(extract_v64int4(fifo_ld_pop_1d_byte(pw, fs, {{ 4 * R3 - 64 }}), 0));
            fifo_ld_fill(pw, fs);
            const v64int8 d = unpack(extract_v64int4(fifo_ld_pop_1d_byte(pw, fs, {{ 4 * R3 - 64 }}), 0));
            const v64int8 m7 = *(const v64int8 __aie_dm_resource_c *)pm;
            W0.mantissa = band(a, m7);
            W1.mantissa = band(b, m7);
            W2.mantissa = band(d, m7);
            W3.mantissa = bor(bor(band(a, *(const v64int8 __aie_dm_resource_c *)(pm + 64)),
                                  band(b, *(const v64int8 __aie_dm_resource_c *)(pm + 128))),
                              band(d, *(const v64int8 __aie_dm_resource_c *)(pm + 192)));
            pm = add_2d_ptr(pm, -256, 1, cm, 256);
            v64accfloat C0 = mac_8x8_8x8T(x1, 1, N4, 1, mul_8x8_8x8T(x0, 1, N4, 1));
            v64accfloat C1 = mac_8x8_8x8T(x3, 1, N4, 1, mul_8x8_8x8T(x2, 1, N4, 1));
            C0 = mac_8x8_8x8T(x0, 1, W0, 0, C0);
            C1 = mac_8x8_8x8T(x2, 1, W2, 0, C1);
            C0 = mac_8x8_8x8T(x1, 1, W1, 0, C0);
            C1 = mac_8x8_8x8T(x3, 1, W3, 0, C1);
            aie::store_v(pc, {{ p }}_sum2(C0));
            aie::store_v(pc + 128, {{ p }}_sum2(C1));
            pc += 256;
        }
    }
}

// super-blocks 2 H, 2 H + 1 of a Q3_K tile: y[g][32] (rows 8 g ..) += their rows against xp (their 64 panels){{ "; ys:
// the second super-block's rows at y + ys (row groups)" if rg else "" }}
template <int H>
__attribute__((noinline)) void {{ p }}_q3k(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr{{ ", int ys" if rg else "" }})
{
    uint8_t *hw = scr;
    bfloat16 *sct = (bfloat16 *)(scr + 1152), *dg = (bfloat16 *)(scr + 2688), *c = (bfloat16 *)(scr + 2752);
    float *yu = (float *)(scr + 2176);
    int8_t *mt = (int8_t *)(scr + 6912);  // (the masks)
    {{ p }}_k23al(a + {{ 4 * R3 * 4 }} * H, hw, 4, {{ 4 * R3 }});  // (the low nibbles of the two)
    {{ p }}_k23al(a + {{ 4 * R3 * 8 }} + {{ 4 * R3 * 2 }} * H, hw + 512, 5, {{ 4 * R3 }});  // (their crumbs, .., their d)
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        {{ p }}_q3k_tab(hw, sct, yu, dg, H, u);
        {{ p }}_q3k_mm(xp + {{ 32 * PANEL }} * u, a + {{ 4 * R3 * NW3 }} + {{ 4 * R3 * 24 }} * (2 * H + u), c, mt);
        {{ p }}_k23sc(c, sct, yu, dg, y{{ " + u * ys" if rg else "" }});
    }
}
"""

class Q3KSlot(k23.PlanarK23, Slot):
    formats = (TileFormat("Q3_K", 8, 110, 32, 1, 96, 14, planes=3,
                          word_major=True, dq_row_cycles=10.0, mmv="q3k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q3_K": _GEMM, "mmv/Q3_K": _MMV}
    ws_bytes = 256  # (per 8-row block: A of 16 sub-blocks)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # (cb: a dq call's code bytes and scales per lane, 1 KB per 16-row block and quad)
        return {**super().qd_scratch(op), "ex": (1, 128 * op.nm)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty, unit=True)
        cb_ty = np.ndarray[(self.qd_scratch(op)["ex"][1],), np.dtype[np.uint8]]
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [cb_ty]), collect=True, unit=True,
                         ex_ty=cb_ty)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        hw = np.zeros((R, header_words(nsb)), np.uint32)
        u = (f["sc"].astype(np.int16) + 32).astype(np.uint8).reshape(R, nsb, 16)
        hw[:, :2 * nsb] = k23.words((u & 15).reshape(R, nsb, 2, 8)).reshape(R, 2 * nsb)
        cr = u >> 4
        hw[:, 2 * nsb:3 * nsb] = k23.words(cr[..., :8] + 4 * cr[..., 8:])
        d = k23.bf16_u16(f["d"])  # (R, nsb)
        hw[:, 3 * nsb:3 * nsb + nsb // 2] = d[:, 0::2] | (d[:, 1::2] << 16)
        c = f["q"].reshape(R, nsb, 8, 4, 8)  # [p][t][b]
        q3 = c[..., 3, :]
        nib = np.stack([c[..., t, :] + 8 * ((q3 >> t) & 1) for t in range(3)], axis=3)  # (R, nsb, p, t, b)
        return k23.tile_parts(hw, k23.words(nib), R, nsb)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        v = super().mmv_vars(fmt, tile_bytes, nsb)
        assert 16 < v["R3"] <= 24, v  # (3 row groups: the kernels' layout)
        return {**v, "NW3": header_words(nsb)}


SLOT = Q3KSlot()
