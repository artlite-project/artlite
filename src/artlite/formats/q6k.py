"""Q6_K tiles (210 B per row and super-block): d [row][sb] bf16 (padded to 32 B) | sc planes [sb pair p][row]
[32 B]: int8 [sb 2][h][i] (sub-block 2 i + h) | 2 half-super-block slices [ql step 2h: R x 32 B | ql step 2h + 1:
R x 32 B | qh: R x 32 B]: ql nibbles of a step, qh nibble l = c(step 2h) + 4 c(step 2h + 1) of lane l (c = high 2
bits). Every part is planes of 32 B per row: the prefill GEMM's memtile gathers a k-tile's words of all rows with
one 4-dimensional pattern.

Prefill: GemmQd's planar gather (``_hcp``, ``_hdr_d`` once per K chunk, ``_hdr_sb``, ``_dq``)."""

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
)

# GemmQd: the planar kernels
_GEMM = r"""// Q6_K (GemmQd): weight objects [hj 6][word ii 2][g {{ n }}] words, hj = 3 h + j (j: ql of step 2 h,
// ql of step 2 h + 1, qh of the half; word ii: group 2 kt + ii), rows g of the pass's tiles zero-padded
// to {{ n }}. The header unit (the tiles' d chunks and sc planes, permuted like the codes) spans the
// four k-tile slots: _hcp collects their planes {{ hp0 }} .. 5 in hb ([kt][hj][ii][g] words), _hdr_d extracts d per
// row and super-block once per K chunk, _hdr_sb the super-block's scales bf16(d sc) before its k-tiles.
void {{ p }}_hcp(const bfloat16 *__restrict b, bfloat16 *__restrict hb, int32_t kt)
{
    const bfloat16 *s = b + {{ 4 * hp0 * n }};  // (before the call: al::copy)
    al::copy<{{ 2 * hs }}, 32>(hb + {{ 2 * hs }} * kt, s);
}

// d[g][sb] of the pass's rows -> dv = ws + {{ 16 * n }}: [sb][g {{ n }}] fp32 (0 for the padding rows). The d
// plane of a tile (bf16 e = {{ nsb }} r + sb, 16 per 32-byte chunk) is its rows {{ R - n_d }} .. {{ R - 1 }} of hj {{ hj_d }}.
void {{ p }}_hdr_d(const uint32_t *__restrict hb, bfloat16 *__restrict ws)
{
    uint32_t *__restrict dv = (uint32_t *)(ws + {{ 16 * n }});
{% if TR < n %}
    // (the padding rows: the last 16-word line of each super-block, zeroed before the values: n - TR < 16)
{% for sb in range(nsb) %}
    aie::store_v(dv + {{ sb * n + n - 16 }}, aie::zeros<uint32, 16>());
{% endfor %}
{% endif %}
    // (e = {{ nsb }} r + sb unrolled: constant offsets, one load per word (sb even: its low half, odd: its high half);
    // the rolled loop over e with its index math: II 17 per value, 4660 cycles instead of 281)
    const uint32_t *__restrict src = hb + {{ 2 * (hj_d - hp0) * n + R - n_d }};
    uint32_t *__restrict dst = dv;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tiles }}; t++) {
{% for e in range(0, nsb * R, 2) %}
{% set w = (e % 16) // 2 %}
        {
            const uint32_t v = src[{{ (w // 2) * hs + (w % 2) * n + e // 16 }}];
            dst[{{ (e % nsb) * n + e // nsb }}] = v << 16;
            dst[{{ ((e + 1) % nsb) * n + (e + 1) // nsb }}] = v & 0xFFFF0000u;
        }
{% endfor %}
        src += {{ R }};
        dst += {{ R }};
    }
}

// super-block sb's scales of the pass's rows: ws [kt 4][h 2][ii 2][g {{ n }}] = bf16(d sc[h][2 kt + ii])
// (the decode's rounding). sc[g][sb][h][i] is byte i % 4 of word (2 (sb % 2) + h, i / 4) of plane
// {{ hj_sc0 }} + sb / 2.
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict dv = (const float *)(ws + {{ 16 * n }}) + sb * {{ n }};
    const uint32_t *__restrict sc = hb + ({{ 2 * (hj_sc0 - hp0) }} + 2 * (sb >> 1)) * {{ n }} + 2 * (sb & 1) * {{ hs }};
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < {{ n }}; g += 16) {
        const aie::vector<bfloat16, 16> d16 = aie::accum<accfloat, 16>(aie::load_v<16>(dv + g)).template to_vector<bfloat16>();
        AIE_LOOP_NO_UNROLL
        for (int hw = 0; hw < 4; hw++) {  // (h, wi) = (hw / 2, hw % 2)
            const int h = hw >> 1, wi = hw & 1;
            // the 4 sc bytes (i = 4 wi ..) of word (2 (sb % 2) + h, wi) of 16 rows
            const aie::vector<int8, 64> by = aie::load_v<16>(sc + h * {{ hs }} + wi * {{ n }} + g).template cast_to<int8>();
            const aie::vector<int8, 32> lo = aie::filter_even(by, 2), hi = aie::filter_odd(by, 2);  // bytes 0 1, 2 3
            // i = 4 wi + k: k-tile kt = i / 2 = 2 wi + k / 2, word ii = k % 2 -> ws + (kt 4 + h 2 + ii) n
            bfloat16 *__restrict o = ws + (8 * wi + 2 * h) * {{ n }} + g;
{% for k in range(4) %}
            {
                const aie::vector<int8, 16> v = {{ "aie::filter_even" if k % 2 == 0 else "aie::filter_odd" }}({{ "lo" if k < 2 else "hi" }}, 1);
                const aie::vector<bfloat16, 16> f = aie::to_float<bfloat16>(v.template grow<32>()).template extract<16>(0);
                aie::store_v(o + {{ ((k // 2) * 4 + k % 2) * n }}, aie::mul(f, d16).template to_vector<bfloat16>());
            }
{% endfor %}
        }
    }
}

// k-tile kt of super-block sb -> {{ n }} x 64 bfp16 panels in the MAC pop order [16-row block][k-step
// 4 ii + 2 h + s'][8-row half]: w = d sc (q - 32) = (128 + q) a - 160 a with q = ql + 16 c (c of step
// 2 h + s' in qh bits 2 s', 2 s' + 1), exact in fp32, rounded once.
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
    ws += half * {{ n_mac }};
{% endif %}
    const bfloat16 __aie_dm_resource_c *__restrict ps = (const bfloat16 __aie_dm_resource_c *)(ws + kt * {{ 4 * n }});
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)stage);
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
    const aie::vector<bfloat16, 64> k160 = aie::broadcast<bfloat16, 64>(-160.0f);
    const aie::vector<uint8, 128> m3 = aie::broadcast<uint8, 128>(3);
    const aie::vector<uint8, 128> m12 = aie::broadcast<uint8, 128>(12);
    // (h in the loop index: half the body per iteration pipelines post-RA, 1109 cycles per k-tile at n = 80 with
    // both halves unrolled. The scale and code pointers walk the (h, ii, block) order in 3-D steps: II 30, mv-bound
    // (expand16_bc: 4 shuffles instead of 7); index math from it: 37 alu ops, II 46)
    dims_3d_t ds = dims_3d_from_steps(2, {{ 2 * n }}, 2, {{ n }}, 16);
    dims_3d_t dc = dims_3d_from_steps(2, {{ 4 * 3 * 2 * n }}, 2, {{ 4 * n }}, 64);
    const uint8_t *__restrict row = b;
    for (int it = 0; it < {{ n_mac // 4 }}; it++) {  // (16-row block, word ii, half h) = (it / 4, it / 2 % 2, it % 2)
        aie::vector<bfloat16, 64> S0, S1;
        al::expand16_bc(aie::load_v<16>(ps), S0, S1);
        const aie::vector<uint8, 128> qh = aie::unpack(aie::load_v<128>((const uint4 __aie_dm_resource_a *)(row + {{ 4 * 2 * 2 * n }})));
        const aie::accum<accfloat, 64> C0 = aie::mul(S0, k160), C1 = aie::mul(S1, k160);
{% for sp in range(2) %}
        {
            const aie::vector<uint8, 128> ql = aie::unpack(aie::load_v<128>((const uint4 __aie_dm_resource_a *)(row + {{ 4 * sp * 2 * n }})));
            const aie::vector<uint8, 128> q = aie::bit_or(ql, {{ "aie::upshift(aie::bit_and(qh, m3), 4)" if sp == 0 else "aie::upshift(aie::bit_and(qh, m12), 2)" }});
            out.push(aie::mac(C0, al::zipw(q.template extract<64>(0), c43), S0).template to_vector<bfp16ebs8>());
            out.push(aie::mac(C1, al::zipw(q.template extract<64>(1), c43), S1).template to_vector<bfp16ebs8>());
        }
{% endfor %}
        ps = add_3d_ptr(ps, ds);
        row = add_3d_byte(row, dc);
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


# the decode GEMV (``decode.gemv``)
_MMV = r"""// ---- Q6_K (R = {{ R6 }} rows, row group 1 = rows 8 .. and junk): d [row][sb 4] bf16 (padded to {{ q6_dpad }} B) | sc
// planes [pair 2][row R][32 B] = int8 [sb 2][h 2][i 8] (sub-block 2 i + h) | codes [hs = 2 sb + h][plane 3][row R][32 B]:
// the low nibbles of steps 2 h, 2 h + 1, then qh (nibble = c(2 h) + 4 c(2 h + 1), c the codes' high bits); lane l = 8 i +
// j of a plane is k = 256 sb + 32 i + 8 s + j. A code panel is two MMULs: the low nibbles and the masked qh at x 16 (n &
// 3) or x 4 (n & 12): 16 c. Scratch per super-block: [i][h][g][ql0 | ql1 | c0 | c1] (4 KB) | tab [g][h][hl][i][r] (1 KB)
// | c [i][h][g][32] (2 KB)

// Pass 1: the codes of super-block sb -> scr (c0, c1: the qh nibbles masked, still packed)
__attribute__((noinline)) void {{ p }}_q6k_tr(const uint8_t *__restrict src, uint8_t *__restrict scr)
{
    const v16int32 m3 = broadcast_to_v16int32((int)0x33333333), m12 = broadcast_to_v16int32((int)0xCCCCCCCC);
    const uint8_t *s = src;
    uint8_t *d = scr;
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < 2; h++) {
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int q = 0; q < 2; q++) {  // the low-nibble planes of steps 2 h + q, both row groups (256 B apart)
            v16int32 o0, o1, o2, o3, u0, u1, u2, u3;
            {{ p }}_tr8<32>(s, o0, o1, o2, o3);
            {{ p }}_tr8<32>(s + 256, u0, u1, u2, u3);
            {{ p }}_put8<512>(d, o0, o1, o2, o3);
            {{ p }}_put8<512>(d + 128, u0, u1, u2, u3);
            s += {{ 32 * R6 }};
            d += 32;
        }
        v16int32 o0, o1, o2, o3, u0, u1, u2, u3;  // qh
        {{ p }}_tr8<32>(s, o0, o1, o2, o3);
        {{ p }}_tr8<32>(s + 256, u0, u1, u2, u3);
        {{ p }}_put8<512>(d, band(o0, m3), band(o1, m3), band(o2, m3), band(o3, m3));
        {{ p }}_put8<512>(d + 32, band(o0, m12), band(o1, m12), band(o2, m12), band(o3, m12));
        {{ p }}_put8<512>(d + 128, band(u0, m3), band(u1, m3), band(u2, m3), band(u3, m3));
        {{ p }}_put8<512>(d + 160, band(u0, m12), band(u1, m12), band(u2, m12), band(u3, m12));
        s += {{ 32 * R6 }};
        d += 256 - 64;
    }
}

// Pass 2: the tables of super-block sb: TD = sc d (I against the rows' sc: [i][r]) -> tab [g][h][hl][i][r]
__attribute__((noinline)) void {{ p }}_q6k_tab(const uint8_t *__restrict a, bfloat16 *__restrict tab, int sb)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const v64bfp16ebs8 I = al::eye();
    const int sbl = sb & 1;
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < 2; g++) {
        // sc of the pair's rows 8 g .. -> word panels (word w = 4 sbl + 2 h + ih: i = 4 ih ..): [row][ih] per h; the
        // rows' words 4 sbl .. at 16 sbl
        v16int32 o0, o1, o2, o3;
        {{ p }}_tr8<32>(a + {{ q6_dpad }} + {{ 32 * R6 }} * (sb >> 1) + 256 * g, o0, o1, o2, o3);
        if (sbl) {
            o0 = o2;
            o1 = o3;
        }
        // d of the rows: [row 8][sb 4] bf16
        const aie::vector<bfloat16, 32> dv = aie::load_v<32>((const bfloat16 *)(a + 64 * g));
        const aie::vector<bfloat16, 16> d2 = (sb >> 1) ? aie::filter_odd(dv, 2) : aie::filter_even(dv, 2);
        const aie::vector<bfloat16, 64> D = {{ p }}_pick(d2, sbl).template grow_replicate<64>();
        v64bfp16ebs8 H;
        H.exponent = al::e8(0x85);
        H.mantissa = (v64int8)shuffle(o0, o0, T32_2x8);  // [row 8][words 2] = int8 [row][i 8]
        {{ p }}_td(mul_8x8_8x8T(I, 1, H, 1), D, tab + 256 * g);
        H.mantissa = (v64int8)shuffle(o1, o1, T32_2x8);
        {{ p }}_td(mul_8x8_8x8T(I, 1, H, 1), D, tab + 256 * g + 128);
    }
}

// Pass 3: c_g,i,h = bf16(C[hi rows] + C[lo rows]) -> c [i][h][g][32]; C = -32 XS16_2i+h + sum over the two steps of
// X (ql + 16 c)
__attribute__((noinline)) void {{ p }}_q6k_mm(const uint8_t *__restrict xp, const uint8_t *__restrict xsum,
                                              const uint8_t *__restrict w, bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    al::BIn<aie_dm_resource::d> xz((const bfp16ebs8 *)xsum);
    const uint8_t __aie_dm_resource_b *__restrict pw = (const uint8_t __aie_dm_resource_b *)w;
    bfloat16 *__restrict pc = c;
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 2; k++) {  // the sum panels of sub-blocks 0 .. 7, 8 .. 15
        const v64bfp16ebs8 Z = xz.pop();
        const int8_t __aie_dm_resource_d *__restrict pn = (const int8_t __aie_dm_resource_d *){{ p }}_nz[2][0];
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int t = 0; t < 8; t++) {  // (i, h): sub-block 2 i + h = 8 k + t
            const v64bfp16ebs8 N = {{ p }}_np(pn);
            pn += 64;
            const v64bfp16ebs8 x0 = xs.pop(), x1 = xs.pop();
            v64accfloat C0 = mul_8x8_8x8T(Z, 1, N, 1);
            v64accfloat C1 = mul_8x8_8x8T(Z, 1, N, 1);
            C0 = mac_8x8_8x8T(x0, 1, {{ p }}_wu(pw), 0, C0);
            C1 = mac_8x8_8x8T(x0, 1, {{ p }}_wu(pw + 128), 0, C1);
            C0 = mac_8x8_8x8T(x0, 1, {{ p }}_wu(pw + 64, 0x89), 0, C0);
            C1 = mac_8x8_8x8T(x0, 1, {{ p }}_wu(pw + 192, 0x89), 0, C1);
            C0 = mac_8x8_8x8T(x1, 1, {{ p }}_wu(pw + 32), 0, C0);
            C1 = mac_8x8_8x8T(x1, 1, {{ p }}_wu(pw + 160), 0, C1);
            C0 = mac_8x8_8x8T(x1, 1, {{ p }}_wu(pw + 96, 0x87), 0, C0);
            C1 = mac_8x8_8x8T(x1, 1, {{ p }}_wu(pw + 224, 0x87), 0, C1);
            aie::store_v(pc, {{ p }}_sum2(C0));
            aie::store_v(pc + 32, {{ p }}_sum2(C1));
            pw += 256;
            pc += 64;
        }
    }
}

template <int HI>
__attribute__((noinline)) void {{ p }}_q6k(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr{{ ", int ys" if rg else "" }})
{
    bfloat16 *__restrict tab = (bfloat16 *)(scr + 4096);  // [g][h][hl][i][r]
    bfloat16 *__restrict c = (bfloat16 *)(scr + 5120);  // [i][h][g][32]
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        const int sb = 2 * HI + u;
{% if rg %}
        float *__restrict yu = y + u * ys;  // (row groups: each super-block's rows of their own)
{% endif %}
        {{ p }}_q6k_tr(a + {{ q6_codes }} + {{ 2 * 3 * 32 * R6 }} * sb, scr);
        {{ p }}_q6k_tab(a, tab, sb);
        {{ p }}_q6k_mm(xp + {{ 32 * PANEL }} * u, xs + {{ XS_SLOT }} * (2 + sb), scr, c);
        // (sub-blocks n = 2 i + h in c's order; their tables at 128 h + 8 i: the h = 0 ones, then the h = 1 ones)
        {{ p }}_sc(c, tab, {{ "yu" if rg else "y" }}, 256, 8, 128);
        {{ p }}_sc(c + 64, tab + 128, {{ "yu" if rg else "y" }}, 256, 8, 128);
    }
}
"""


class Q6KSlot(Slot):
    formats = (TileFormat("Q6_K", 2, 210, 16, 1, 192, 16, planes=6, plane_major=True, header_pad32=True,
                          ws_planes=2, dq_row_cycles=12.5, mmv="q6k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q6_K": _GEMM, "mmv/Q6_K": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        hs = 2 * nsb  # half super-blocks
        p = f["q"][:, _word_order(nsb)].reshape(R, nsb, 4, 64)  # [row][sb][s][lane]
        ql = _nibbles((p & 15).reshape(R, -1)).reshape(R, hs, 2, 32)  # [row][hs][step 2h, 2h + 1][lanes]
        c = p >> 4
        qh = _nibbles((c[:, :, 0::2] + 4 * c[:, :, 1::2]).reshape(R, -1)).reshape(R, hs, 1, 32)  # [row][hs][lanes]
        sc = f["sc"].reshape(R, nsb, 8, 2).transpose(0, 1, 3, 2)  # [row][sb][h][i] = sc[2 i + h]
        d = np.zeros(-(-2 * nsb * R // 32) * 32, np.uint8)
        d[:2 * nsb * R] = _bf16_bytes(f["d"]).reshape(-1)
        planes = np.ascontiguousarray(sc).view(np.uint8).reshape(R, nsb // 2, 32).transpose(1, 0, 2)  # [pair][row][32]
        halves = np.concatenate([ql, qh], axis=2).transpose(1, 2, 0, 3)  # [hs][ql 2h, ql 2h + 1, qh][row][32 B]
        return [d, planes, halves]

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the super-block's scales [kt][h][ii][g] bf16, then d [sb][g] fp32
        return {"ws": (1, 32 * op.n + 4 * op.nsb * op.n), "hb": (1, 4 * 4 * op._hdr_slot_words)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty)
        chunk = kernel("hdr_d", [hb_ty, ws_ty])  # d of the rows, once per K chunk
        return DqKernels(hdr=hdr, sb=sb, chunk=chunk, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True,
                         unit=False)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        """Where the header unit's parts land after the permutation ([h][j][row] chunks of 32 B = hj planes):
        d chunks n_d at rows R - n_d .. of hj_d, sc planes from hj_sc0 on."""
        R, nsb = op.R, op.nsb
        n_d = -(-2 * nsb * R // 32)
        assert n_d <= R and (6 * R - n_d - nsb // 2 * R) * 32 == op.hdr_skip, (n_d, op.hdr_skip)
        return {"n_d": n_d, "hj_d": 5 - nsb // 2, "hj_sc0": 6 - nsb // 2}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R6 = self.format(fmt).rows(tile_bytes, nsb)
        assert R6 <= 16, R6
        dpad = -(-2 * nsb * R6 // 32) * 32
        return {"R6": R6, "q6_dpad": dpad, "q6_codes": dpad + nsb // 2 * 32 * R6}


SLOT = Q6KSlot()
