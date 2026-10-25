"""Q8_0 tiles (272 bytes per row and super-block): the codes as the decode GEMV's MMUL panels [row group R / 8][group
i][step s][row 8][8 B] per super-block (codes 8 s .. 8 s + 7 of group i of the 8 rows: one 64-byte B panel each, no
transposition in the kernel) and d [row][i] bf16:

- 1024-column tiles (``units``: Qwen's GDN z, 8 rows): per super-block a unit [codes 256 R | d [row][i] 16 R] (2176 B,
  32-byte aligned), what a prefill GEMM's weight object holds whole;
- 512-column tiles (Gemma's LM head, 16 rows; decode only): d [row][sb][i] first, then the super-blocks' codes.

Decode (``overlay.gemv``): Gemma's LM head by the row split (a core per row group), Qwen's z by the super-block split
(as the K-quants': pageable).

Prefill (``gemm/Q8_0``, GemmQd on units: ``Slot.unit_bytes``): per unit d expanded per lane ([row][k] = d[row]), per
panel the codes to fp32 (exact) times d (exact), rounded once into bfp16 (bfp16(bf16(d) q))."""

from typing import ClassVar

import numpy as np

from .common import DqKernels, Slot, TileFormat, _bf16_bytes, dq_kernel, i32

# the decode GEMV (``overlay.gemv``; the scale tables: ``mmv/dtab``)
_MMV = r"""// a weight panel of 64 int8 codes (8 rows x 8 k), value = code
template <typename T>
[[clang::always_inline]] inline v64bfp16ebs8 {{ p }}_w8(const T *p)
{
    v64bfp16ebs8 n;
    n.mantissa = *(const v64int8 __aie_dm_resource_b *)p;
    n.exponent = al::e8(0x85);
    return n;
}

// ---- Q8_0 ({{ R80 }} rows): {{ "per super-block a unit [codes [i][s][row 8][8 B] (MMUL panels) | d [row][i] bf16] (2176 B)" if R80 == 8 else "d [row][sb][i] bf16 | codes [sb][g][i][s][row 8][8 B] (MMUL panels)" }}. {{ "Core H: super-blocks 2 H, 2 H + 1 of the row group (y[0 ..])" if R80 == 8 else "Core H: row group H (y[32 H ..])" }}.
// Scratch: tab [sb][i][r] (128 B per sb) | c [sb][i][32] (512 B per sb)

// c_sb,i = bf16(C[hi rows] + C[lo rows]) -> c [sb][i][32] for NS super-blocks (their slices of the row group SK bytes
// apart); C = sum_s X(sb, i, s) W(sb, i, s); groups i, i + 1 in two chains
template <int NS, int SK>
__attribute__((noinline)) void {{ p }}_q80_mm(const uint8_t *__restrict xp, const uint8_t *__restrict w, bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    const int8_t __aie_dm_resource_b *__restrict pw = (const int8_t __aie_dm_resource_b *)w;
    bfloat16 *__restrict pc = c;
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < NS; sb++) {
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int i = 0; i < 8; i += 2) {
            const v64bfp16ebs8 x0 = xs.pop(), x1 = xs.pop(), x2 = xs.pop(), x3 = xs.pop();
            const v64bfp16ebs8 x4 = xs.pop(), x5 = xs.pop(), x6 = xs.pop(), x7 = xs.pop();
            v64accfloat C0 = mul_8x8_8x8T(x0, 1, {{ p }}_w8(pw), 1);
            v64accfloat C1 = mul_8x8_8x8T(x4, 1, {{ p }}_w8(pw + 256), 1);
            C0 = mac_8x8_8x8T(x1, 1, {{ p }}_w8(pw + 64), 1, C0);
            C1 = mac_8x8_8x8T(x5, 1, {{ p }}_w8(pw + 320), 1, C1);
            C0 = mac_8x8_8x8T(x2, 1, {{ p }}_w8(pw + 128), 1, C0);
            C1 = mac_8x8_8x8T(x6, 1, {{ p }}_w8(pw + 384), 1, C1);
            C0 = mac_8x8_8x8T(x3, 1, {{ p }}_w8(pw + 192), 1, C0);
            C1 = mac_8x8_8x8T(x7, 1, {{ p }}_w8(pw + 448), 1, C1);
            aie::store_v(pc, {{ p }}_sum2(C0));
            aie::store_v(pc + 32, {{ p }}_sum2(C1));
            pw += 512;
            pc += 64;
        }
        pw += SK;  // (the next super-block's slice, the same row group)
    }
}

{% if R80 == 8 %}
// the d [row 8][i 8] of two consecutive units (from u: codes, then d) -> tab [sb 2][i][r]
__attribute__((noinline)) void {{ p }}_q80_d(const uint8_t *__restrict u, bfloat16 *__restrict tab)
{
    aie::store_v(tab, aie::transpose(aie::load_v<64>((const bfloat16 *)(u + 2048)), 8, 8));
    aie::store_v(tab + 64, aie::transpose(aie::load_v<64>((const bfloat16 *)(u + 2176 + 2048)), 8, 8));
}

{% endif %}
template <int H>
__attribute__((noinline)) void {{ p }}_q80(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    bfloat16 *tab = (bfloat16 *)scr;
    bfloat16 *c = (bfloat16 *)(scr + 1024);
{% if R80 == 8 %}
    {{ p }}_q80_d(a + 4352 * H, tab);  // (the rows' d of super-blocks 2 H, 2 H + 1)
    {{ p }}_q80_mm<2, 128>(xp, a + 4352 * H, c);
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < 2; sb++)
        {{ p }}_sc1<1>(c + 256 * sb, tab + 64 * sb, y);
{% else %}
    {{ p }}_dtab<1>((const bfloat16 *)a + {{ 64 * NSB }} * H, tab);
    {{ p }}_q80_mm<{{ NSB }}, {{ 256 * R80 - 2048 }}>(xp, a + {{ 16 * NSB * R80 }} + 2048 * H, c);
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < {{ NSB }}; sb++)
        {{ p }}_sc1<1>(c + 256 * sb, tab + 64 * sb, y + 32 * H);
{% endif %}
}
"""


# GemmQd (prefill) on units of 8-row tiles
_GEMM = r"""// Q8_0 (GemmQd, super-block units: formats/q8.py): the weight object is a unit of the pass's {{ tiles }} tiles of 8
// rows, [tile][codes [i 8][s 4][r 8][8 B] (2048 B) | d [r 8][i 8] bf16 (128 B)] at a pitch of {{ UP }} B. _sb: per tile
// and group i the 64-lane scale vector [r][k] = d[r][i] -> ws [tile][i][64]; _dq: per panel the codes to fp32 (exact)
// times it (exact), rounded once.
void {{ p }}_sb(const uint8_t *__restrict b, bfloat16 *__restrict ws, bfloat16 *__restrict hb, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    const uint8_t *__restrict pb = b + 2048;
    bfloat16 *__restrict pw = ws;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tiles }}; t++) {
        const aie::vector<bfloat16, 64> dd = aie::transpose(aie::load_v<64>((const bfloat16 *)pb), 8, 8);  // [i][r]
{% for i in range(8) %}
        aie::store_v(pw + {{ 64 * i }}, al::expand8_of(dd.template extract<32>({{ i // 4 }}), {{ i % 4 }}));
{% endfor %}
        pb += {{ UP }};
        pw += 512;
    }
}

// k-tile kt of the unit -> {{ n }} x 64 bfp16 panels in the MAC pop order [16-row block][k-step ks][8-row half]: tile
// 2 block + half, group i = 2 kt + ks / 4, step ks % 4
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
    const int8_t __aie_dm_resource_a *__restrict pq = (const int8_t __aie_dm_resource_a *)(b + 512 * kt);
    const bfloat16 __aie_dm_resource_c *__restrict pd = (const bfloat16 __aie_dm_resource_c *)(ws + 128 * kt);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)stage);
    // (codes: k-steps 64 B apart, then the next block 2 tiles on; scales: 4 k-steps per group, the second group, then
    // the next block)
    dims_2d_t wq = dims_2d_from_steps(8, 64, {{ 2 * UP }});
    dims_3d_t wd = dims_3d_from_steps(4, 0, 2, 64, 1024);
    for (int it = 0; it < {{ n // 2 }}; it++) {  // (16-row block, k-step) = (it / 8, it % 8)
        const aie::vector<bfloat16, 64> D0 = aie::load_v<64>(pd), D1 = aie::load_v<64>(pd + 512);
        out.push(aie::mul(aie::to_float<bfloat16>(aie::load_v<64>(pq)), D0).template to_vector<bfp16ebs8>());
        out.push(aie::mul(aie::to_float<bfloat16>(aie::load_v<64>(pq + {{ UP }})), D1).template to_vector<bfp16ebs8>());
        pq = add_2d_ptr(pq, wq);
        pd = add_3d_ptr(pd, wd);
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


class Q8Slot(Slot):
    formats = (TileFormat("Q8_0", 9, 272, 16, 8, 256, 16, planes=8, dq_row_cycles=4.0, mmv="q80"),)
    sources: ClassVar[dict[str, str]] = {"mmv/Q8_0": _MMV, "gemm/Q8_0": _GEMM}

    @staticmethod
    def units(nsb: int) -> bool:
        """Tiles of nsb super-blocks stored as super-block units (module docstring)."""
        return nsb == 4

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R % 8 == 0, R
        q = np.asarray(f["q"], np.int8).reshape(R // 8, 8, nsb, 8, 4, 8)  # [g][r][sb][i][s][8]
        codes = np.ascontiguousarray(q.transpose(2, 0, 3, 4, 1, 5)).view(np.uint8).reshape(nsb, -1)  # [sb][g][i][s][r][8]
        d = _bf16_bytes(f["d"])  # [row][sb][i]
        if self.units(nsb):
            return [np.concatenate([codes, d.reshape(R, nsb, 16).transpose(1, 0, 2).reshape(nsb, -1)], 1)]
        return [d, codes]

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return self.units(kc // 256)  # (512-column tiles: decode only)

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return 272 * rows if self.units(nsb) else 0

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (whole 16-row blocks; two weight objects of up to 8 units in the dequantizing core's L1: 34 KB)
        return tiles * rows % 16 == 0 and tiles <= 8

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the scale vectors of the unit ([tile][i][64] bf16); hb unused
        return {"ws": (1, 1024 * op.tiles), "hb": (1, 64)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == 8 and op.halves == 1 and op.n == op.TR, "Q8_0: passes of whole 16-row blocks"
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), units=True)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        return {"UP": op.unit_pitch}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return True

    def mmv_split(self, fmt: str, nsb: int) -> int:
        # (512-column tiles: each core one row group of every super-block; 1024: 8 rows, the super-block split)
        return 16 if nsb == 2 else 0

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/dtab", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb, self.mmv_split(fmt, nsb))
        assert R == (16 if nsb == 2 else 8), (R, nsb)
        return {"R80": R}


SLOT = Q8Slot()
