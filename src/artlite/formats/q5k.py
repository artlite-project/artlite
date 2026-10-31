"""Q5_K tiles (176 B per row and super-block): scale planes [sb pair p][row][8 words]: sc lo nibbles of
super-blocks 2p, 2p + 1 | mn lo nibbles | high crumbs (nibble i = c_sc + 4 c_mn of group i) | d bf16 x 2 | -dmin
bf16 x 2 | codes [sb][plane 5][row][32 B]: the nibbles of step s (plane s), then qh (nibble of lane l: bit s = the
high bit of step s). Planes of 32 B per row, as Q6_K's: the prefill's dequantizing core gathers both with one
memtile pattern.

Prefill: GemmQd's planar gather (``_hcp``, ``_hdr_sb``, ``_expand`` per k-tile into the ``ex`` buffer, ``_dq``)."""

from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from .common import (
    DqKernels,
    Slot,
    TileFormat,
    _bf16_bytes,
    _nibbles,
    _word_order,
    collected_header,
    dq_kernel,
)

# GemmQd: the 6-bit scale expansion (the shared plane8) and the planar kernels
_GEMM = r"""// Q5_K: _hdr_sb expands a super-block's 6-bit scales of 16 rows at a time into ws [16-row block][A | B][group 8][row 16]
{% include "gemm/plane8" %}

// Q5_K (GemmQd): weight objects [plane 5][word ii 2][g {{ n }}] words (planes: the nibbles of steps 0 .. 3, the
// qh nibbles; word ii: group 2 kt + ii), rows g of the pass's tiles zero-padded to {{ n }}. The header unit (the
// tiles' scale planes [pair][row][8 words], permuted like the codes: its planes {{ 5 - nsb // 2 }} .. 4) spans
// the four k-tile slots: _hcp collects their planes {{ hp0 }} .. 4 in hb ([kt][plane][ii][g] words), _hdr_sb expands a
// super-block's scales before its k-tiles. Scale word w of a plane is in slot w / 2, word w % 2: sc lo (sb % 2), mn lo
// (2 + sb % 2), crumbs (4 + sb % 2), d pair (6), -dmin pair (7).
void {{ p }}_hcp(const bfloat16 *__restrict b, bfloat16 *__restrict hb, int32_t kt)
{
    const bfloat16 *s = b + {{ 4 * hp0 * n }};  // (before the call: al::copy)
    al::copy<{{ 2 * hs }}, 32>(hb + {{ 2 * hs }} * kt, s);
}

// super-block sb's scales of the pass's rows: ws [16-row block][A | B][group i 8][row 16] = bf16(d sc), bf16(-dmin mn)
// (the decode's roundings; zero for the padding rows, whose words are zero)
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const uint32_t *__restrict w = hb + 2 * ({{ 5 - nsb // 2 - hp0 }} + (sb >> 1)) * {{ n }} + (sb & 1) * {{ n }};
    const unsigned ssh = (sb & 1) << 4;
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < {{ n }}; g += 16) {
        const aie::vector<uint8, 128> cr = aie::load_v<128>((const uint4 *)(w + {{ 2 * hs }} + g)).unpack();
        {{ p }}_plane8(aie::load_v<128>((const uint4 *)(w + g)).unpack(), cr,
                       aie::load_v<16>(w + {{ 3 * hs }} - (sb & 1) * {{ n }} + g), 0, ssh, ws + 16 * g);
        {{ p }}_plane8(aie::load_v<128>((const uint4 *)(w + {{ hs }} + g)).unpack(), cr,
                       aie::load_v<16>(w + {{ 3 * hs + n }} - (sb & 1) * {{ n }} + g), 2, ssh, ws + 16 * g + 128);
    }
}

// the k-tile's scales per MAC lane (groups 2 kt + ii of 16 rows, each row's scale repeated over its 8 k):
// ex [16-row block][word ii][A0 | A1 bf16 x 64 | C0 | C1 fp32 x 64] with C = B - 128 A, exact in fp32
// (ws: the super-block's scales from group 2 kt on)
__attribute__((noinline)) void {{ p }}_expand(const bfloat16 *__restrict ws, bfloat16 *__restrict ex)
{
    const bfloat16 __aie_dm_resource_c *__restrict pa = (const bfloat16 __aie_dm_resource_c *)ws;
    const bfloat16 __aie_dm_resource_d *__restrict pb = (const bfloat16 __aie_dm_resource_d *)(ws + 128);
    bfloat16 *__restrict pe = ex;
    const aie::vector<bfloat16, 64> k128 = aie::broadcast<bfloat16, 64>(-128.0f);
    const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
    int ca = 0, cb = 0;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int it = 0; it < {{ n // 8 }}; it++) {  // (16-row block, word ii) = (it / 2, it % 2)
        aie::vector<bfloat16, 64> A0, A1, B0, B1;
        al::expand16_bc(aie::load_v<16>(pa), A0, A1);
        al::expand16_bc(aie::load_v<16>(pb), B0, B1);
        aie::store_v(pe, A0);
        aie::store_v(pe + 64, A1);
        aie::store_v((float *)(pe + 128), aie::mac(aie::mul(A0, k128), B0, one).template to_vector<float>());
        aie::store_v((float *)(pe + 256), aie::mac(aie::mul(A1, k128), B1, one).template to_vector<float>());
        pe += 384;
        // next word ii (group), or the next 16-row block's
        pa = add_2d_ptr(pa, 240, 1, ca, 16);
        pb = add_2d_ptr(pb, 240, 1, cb, 16);
    }
}

// k-tile kt of super-block sb -> {{ n }} x 64 bfp16 panels in the MAC pop order [16-row block][k-step 4 ii + s]
// [8-row half]: w = A q + B = (128 + q) A + C with q = lo + 16 hb (hb: bit s of the qh nibble), exact in fp32,
// rounded once. The scales come expanded (_expand: the loop has room for the 8 panels of a 16-row block, the
// steps' shifts compile-time constants) and the pointers walk 2-d patterns: II 50 per 8 panels.
void {{ p }}_dq(const uint8_t *__restrict b, const bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt{{ half_arg }}, bfloat16 *__restrict ex)
{
{% if "ev_dq" in ablate or "ev_kt" in ablate %}
    event0();
{% endif %}
{% if "dq" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if halves > 1 %}
    if (half == 0)  // (the k-tile's scales of both halves)
        {{ p }}_expand(ws + 32 * kt, ex);
    b += half * {{ 4 * n_mac }};  // (the half's 16-row blocks)
    const bfloat16 __aie_dm_resource_c *__restrict pe = (const bfloat16 __aie_dm_resource_c *)(ex + half * {{ n_mac // 8 * 384 }});
{% else %}
    {{ p }}_expand(ws + 32 * kt, ex);
    const bfloat16 __aie_dm_resource_c *__restrict pe = (const bfloat16 __aie_dm_resource_c *)ex;
{% endif %}
    const uint8_t *__restrict ph = b + {{ 4 * 2 * 4 * n }};  // the qh plane
    const uint8_t *__restrict pl = b;  // the nibbles of step s at + {{ 4 * 2 * n }} s
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)stage);
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
    const aie::vector<uint8, 64> c16 = aie::broadcast<uint8, 64>(16);
    int ch = 0, cl = 0;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int it = 0; it < {{ n_mac // 8 }}; it++) {  // (16-row block, word ii) = (it / 2, it % 2)
        const aie::vector<bfloat16, 64> A0 = aie::load_v<64>(pe), A1 = aie::load_v<64>(pe + 64);
        const aie::accum<accfloat, 64> C0(aie::load_v<64>((const float __aie_dm_resource_d *)(pe + 128)));
        const aie::accum<accfloat, 64> C1(aie::load_v<64>((const float __aie_dm_resource_d *)(pe + 256)));
        // 16 qh: read back shifted down by the step (a multiply, not an upshift: the shifts 0 .. 3 then fit the 4
        // shift registers, II 52 -> 50)
        const aie::accum<acc32, 64> u0 = aie::mul(aie::unpack(aie::load_v<64>((const uint4 __aie_dm_resource_a *)ph)), c16);
        const aie::accum<acc32, 64> u1 =
            aie::mul(aie::unpack(aie::load_v<64>((const uint4 __aie_dm_resource_a *)(ph + 32))), c16);
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < 4; s++) {
            const aie::vector<uint8, 64> q0 = aie::bit_or(
                aie::unpack(aie::load_v<64>((const uint4 __aie_dm_resource_a *)(pl + {{ 4 * 2 * n }} * s))),
                aie::bit_and(u0.template to_vector<uint8>(s), c16));
            const aie::vector<uint8, 64> q1 = aie::bit_or(
                aie::unpack(aie::load_v<64>((const uint4 __aie_dm_resource_a *)(pl + {{ 4 * 2 * n }} * s + 32))),
                aie::bit_and(u1.template to_vector<uint8>(s), c16));
            out.push(aie::mac(C0, al::zipw(q0, c43), A0).template to_vector<bfp16ebs8>());
            out.push(aie::mac(C1, al::zipw(q1, c43), A1).template to_vector<bfp16ebs8>());
        }
        pe += 384;
        // next word ii, or the next 16-row block's word 0
        ph = add_2d_byte(ph, {{ 64 - 4 * n }}, 1, ch, {{ 4 * n }});
        pl = add_2d_byte(pl, {{ 64 - 4 * n }}, 1, cl, {{ 4 * n }});
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


# the decode GEMV (``decode.gemv``)
_MMV = r"""// ---- Q5_K (R = {{ R5 }} rows, row group 1 = rows 8 .. and junk): scale planes [pair 2][row R][8 words: sc lo nibbles
// (sb 2 p), (2 p + 1) | mn lo | mn lo | crumbs (2 p) | (2 p + 1) (nibble i = c_sc + 4 c_mn) | d pair | -dmin pair] | codes
// [sb][plane 5][row R][32 B]: the low nibbles of step s (plane s), then qh (nibble of lane l: bit s = the high bit h
// of step s). A code panel is two MMULs: the low nibbles and the masked qh at x 16 / 2^s (16 h). Scratch per
// super-block: [i][step pair][g][ql s | ql s + 1 | h s | h s + 1: 32 B each] (4 KB) | tab [g][hl][i][r] | c [i][g][32]

// Pass 1: the codes of super-block sb -> scr (h 0 .. 3: the qh nibbles masked by 1 .. 8)
__attribute__((noinline)) void {{ p }}_q5k_tr(const uint8_t *__restrict src, uint8_t *__restrict scr)
{
    {{ p }}_trq(src, src + {{ 4 * 32 * R5 }}, scr, {{ 32 * R5 }}, 0, 0, 0x11111111, 0x22222222, 2);
}

// Pass 2: the table TD = d sc of super-block sb -> tab [g][hl][i][r]; y[g] += hilo(XS sc) 16 d + hilo(XS mn) (-dmin)
__attribute__((noinline)) void {{ p }}_q5k_tab(const uint8_t *__restrict a, const uint8_t *__restrict xs,
                                               bfloat16 *__restrict tab, float *__restrict y, int sb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int sbl = sb & 1;
    const v64bfp16ebs8 I = al::eye();
    const v64uint8 m3 = broadcast_to_v64uint8((unsigned char)3), m12 = broadcast_to_v64uint8((unsigned char)12);
    al::BIn<> xsp((const bfp16ebs8 *)xs);  // (the pair's slot: the panel of sb is the (sbl + 1)-th)
    v64bfp16ebs8 X = xsp.pop();
    if (sbl)
        X = xsp.pop();
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < 2; g++) {
        v16int32 o0, o1, o2, o3;  // word panels of the pair's scale planes, rows 8 g ..: [w0 | w1], [w2 | w3], ..
        {{ p }}_tr8<32>(a + {{ 32 * R5 }} * (sb >> 1) + 256 * g, o0, o1, o2, o3);
        o0 = shift_bytes(o0, o0, 32 * sbl);  // (the words of sb: of each pair the first or the second)
        o1 = shift_bytes(o1, o1, 32 * sbl);
        o2 = shift_bytes(o2, o2, 32 * sbl);
        const v64uint8 cr = unpack(extract_v64uint4((v128uint4)o2, 0));  // crumbs of sb
        v64bfp16ebs8 Sl, Sc, Ml, Mc;
        Sl.exponent = Ml.exponent = al::e8(0x85);
        Sc.exponent = al::e8(0x89);
        Mc.exponent = al::e8(0x87);
        Sl.mantissa = (v64int8)unpack(extract_v64uint4((v128uint4)o0, 0));
        Ml.mantissa = (v64int8)unpack(extract_v64uint4((v128uint4)o1, 0));
        Sc.mantissa = (v64int8)band(cr, m3);
        Mc.mantissa = (v64int8)band(cr, m12);
        const aie::vector<bfloat16, 32> dp = aie::vector<bfloat16, 32>((v32bfloat16)o3);  // [d pairs | -dmin pairs]
        const aie::vector<bfloat16, 16> dq = dp.template extract<16>(0), mq = dp.template extract<16>(1);
        const aie::vector<bfloat16, 64> D = {{ p }}_pick(dq, sbl).template grow_replicate<64>();
        {{ p }}_td(mac_8x8_8x8T(I, 1, Sc, 0, mul_8x8_8x8T(I, 1, Sl, 0)), D, tab + 128 * g);
        aie::accum<accfloat, 64> Y = {{ p }}_mo(mac_8x8_8x8T(X, 1, Sc, 0, mul_8x8_8x8T(X, 1, Sl, 0)),
                                                aie::mul(D, aie::broadcast<bfloat16, 64>(16.0f)).template to_vector<bfloat16>(),
                                                aie::zeros<accfloat, 64>());
        Y = {{ p }}_mo(mac_8x8_8x8T(X, 1, Mc, 0, mul_8x8_8x8T(X, 1, Ml, 0)), {{ p }}_pick(mq, sbl).template grow_replicate<64>(), Y);
        {{ p }}_yadd(y + 32 * g, Y);
    }
}

// Pass 3: c_g,i = bf16(C[hi rows] + C[lo rows]) -> c [i][g][32]; C = -16 XS_i + sum_s X_s (ql_s + 16 h_s) (xsum:
// the pair's slot, the panel of the super-block the (sbl + 1)-th)
__attribute__((noinline)) void {{ p }}_q5k_mm(const uint8_t *__restrict xp, const uint8_t *__restrict xsum,
                                              const uint8_t *__restrict w, bfloat16 *__restrict c, int sbl)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    al::BIn<aie_dm_resource::d> xz((const bfp16ebs8 *)xsum);
    v64bfp16ebs8 Z = xz.pop();
    if (sbl)
        Z = xz.pop();
    const uint8_t __aie_dm_resource_b *__restrict pw = (const uint8_t __aie_dm_resource_b *)w;
    const int8_t __aie_dm_resource_d *__restrict pn = (const int8_t __aie_dm_resource_d *){{ p }}_nz[1][0];
    bfloat16 *__restrict pc = c;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 8; i++) {
        const v64bfp16ebs8 N = {{ p }}_np(pn);
        pn += 64;
        v64accfloat C0 = mul_8x8_8x8T(Z, 1, N, 1);
        v64accfloat C1 = mul_8x8_8x8T(Z, 1, N, 1);
{% for st in range(4) %}
        {
            const v64bfp16ebs8 x = xs.pop();
{% set o = 256 * (st // 2) + 32 * (st % 2) %}
            C0 = mac_8x8_8x8T(x, 1, {{ p }}_wu(pw + {{ o }}), 0, C0);
            C1 = mac_8x8_8x8T(x, 1, {{ p }}_wu(pw + {{ o + 128 }}), 0, C1);
            C0 = mac_8x8_8x8T(x, 1, {{ p }}_wu(pw + {{ o + 64 }}, {{ 0x89 - st }}), 0, C0);
            C1 = mac_8x8_8x8T(x, 1, {{ p }}_wu(pw + {{ o + 192 }}, {{ 0x89 - st }}), 0, C1);
        }
{% endfor %}
        aie::store_v(pc, {{ p }}_sum2(C0));
        aie::store_v(pc + 32, {{ p }}_sum2(C1));
        pw += 512;
        pc += 64;
    }
}

template <int HI>
__attribute__((noinline)) void {{ p }}_q5k(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    bfloat16 *__restrict tab = (bfloat16 *)(scr + 4096);  // [g][hl][i][r]
    bfloat16 *__restrict c = (bfloat16 *)(scr + 4608);  // [i][g][32]
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        const int sb = 2 * HI + u;
        {{ p }}_q5k_tr(a + {{ q5_codes }} + {{ 5 * 32 * R5 }} * sb, scr);
        {{ p }}_q5k_tab(a, xs + {{ XS_SLOT }} * HI, tab, y, sb);
        {{ p }}_q5k_mm(xp + {{ 32 * PANEL }} * u, xs + {{ XS_SLOT }} * HI, scr, c, u);
        {{ p }}_sc(c, tab, y, 128, 8, 64);
    }
}
"""


class Q5KSlot(Slot):
    formats = (TileFormat("Q5_K", 6, 176, 16, 1, 160, 16, planes=5, plane_major=True, ws_planes=2,
                          dq_row_cycles=10.5, mmv="q5k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q5_K": _GEMM, "mmv/Q5_K": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        p = f["q"][:, _word_order(nsb)].reshape(R, nsb, 4, 64)  # [row][sb][s][lane]
        ql = _nibbles((p & 15).reshape(R, -1)).reshape(R, nsb, 4, 32)
        hb = (p >> 4).astype(np.uint8)
        qh = _nibbles((hb[:, :, 0] | hb[:, :, 1] << 1 | hb[:, :, 2] << 2 | hb[:, :, 3] << 3).reshape(R, -1))
        codes = np.concatenate([ql, qh.reshape(R, nsb, 1, 32)], axis=2).transpose(1, 2, 0, 3)  # [sb][plane][row][32]
        sc, mn = f["sc"].reshape(R, nsb, 8), f["mn"].reshape(R, nsb, 8)

        def lo(v: np.ndarray) -> np.ndarray:  # the low nibbles: [row][pair][sb][4 B]
            return _nibbles((v & 15).reshape(R, -1)).reshape(R, nsb // 2, 2, 4)

        cr = _nibbles(((sc >> 4) | (mn >> 4) << 2).reshape(R, -1)).reshape(R, nsb // 2, 2, 4)
        dd = _bf16_bytes(f["d"]).reshape(R, nsb // 2, 2, 2).reshape(R, nsb // 2, 1, 4)
        dm = _bf16_bytes(-f["dmin"]).reshape(R, nsb // 2, 1, 4)
        rec = np.concatenate([lo(sc), lo(mn), cr, dd, dm], axis=2).reshape(R, nsb // 2, 32)  # words 0..7
        return [rec.transpose(1, 0, 2), codes]

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        return True  # (the scale planes)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the super-block's scales [group 8][A | B][g] bf16; hb: the 4 header slots; ex: the k-tile's scales
        # expanded per MAC lane [16-row block][word 2][A 2 x 64 bf16 | C 2 x 64 fp32]
        return {"ws": (1, 32 * op.n), "hb": (1, 4 * 4 * op._hdr_slot_words), "ex": (1, 96 * op.n)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty)
        # (the k-tile's expanded scales: a buffer of the dequantizing core)
        ex_ty = np.ndarray[(self.qd_scratch(op)["ex"][1] // 2,), np.dtype[bfloat16]]
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [ex_ty]), collect=True,
                         unit=False, ex_ty=ex_ty)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/trq", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R5 = self.format(fmt).rows(tile_bytes, nsb)
        assert R5 <= 16, R5
        return {"R5": R5, "q5_codes": nsb // 2 * 32 * R5}


SLOT = Q5KSlot()
