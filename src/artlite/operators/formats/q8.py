"""Q8_0 tiles (272 bytes per row and super-block): d [row][sb][i] bf16 | codes as the decode GEMV's MMUL panels
[sb][row group R / 8][group i][step s][row 8][8 B] (codes 8 s .. 8 s + 7 of group i of the 8 rows: one 64-byte B
panel each, no transposition in the kernel). Decode only (``overlay.gemv``, the Gemma LM head): no prefill GEMM
reads them."""

from typing import ClassVar

import numpy as np

from .common import Slot, TileFormat, _bf16_bytes

# the decode GEMV (``overlay.gemv``; the scale tables: ``mmv/dtab``)
_MMV = r"""// a weight panel of 64 int8 codes (8 rows x 8 k), value = code
template <typename T>
[[clang::always_inline]] inline v64bfp16ebs8 {{ p }}_w8(const T *p)
{
    v64bfp16ebs8 n;
    n.mantissa = *(const v64int8 __aie_dm_resource_b *)p;
    n.exponent = {{ p }}_e8(0x85);
    return n;
}

// ---- Q8_0 ({{ R80 }} rows): d [row][sb][i] bf16 | codes [sb][g][i][s][row 8][8 B] (MMUL panels). Core H: row group H
// (y[32 H ..]). Scratch: tab [sb][i][r] (128 B per sb) | c [sb][i][32] (512 B per sb)

// c_sb,i = bf16(C[hi rows] + C[lo rows]) -> c [sb][i][32]; C = sum_s X(sb, i, s) W(sb, i, s); groups i, i + 1 in two
// chains
__attribute__((noinline)) void {{ p }}_q80_mm(const uint8_t *__restrict xp, const uint8_t *__restrict w, bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    const int8_t __aie_dm_resource_b *__restrict pw = (const int8_t __aie_dm_resource_b *)w;
    bfloat16 *__restrict pc = c;
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < {{ NSB }}; sb++) {
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
        pw += {{ 256 * R80 - 2048 }};  // (the next super-block's slice, the same row group)
    }
}

template <int H>
__attribute__((noinline)) void {{ p }}_q80(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    bfloat16 *tab = (bfloat16 *)scr;
    bfloat16 *c = (bfloat16 *)(scr + 1024);
    {{ p }}_dtab<1>((const bfloat16 *)a + {{ 64 * NSB }} * H, tab);
    {{ p }}_q80_mm(xp, a + {{ 16 * NSB * R80 }} + 2048 * H, c);
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < {{ NSB }}; sb++)
        {{ p }}_sc1<1>(c + 256 * sb, tab + 64 * sb, y + 32 * H);
}
"""


class Q8Slot(Slot):
    formats = (TileFormat("Q8_0", 9, 272, 16, 8, 256, 16, mmv="q80"),)
    sources: ClassVar[dict[str, str]] = {"mmv/Q8_0": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R % 8 == 0, R
        q = np.asarray(f["q"], np.int8).reshape(R // 8, 8, nsb, 8, 4, 8)  # [g][r][sb][i][s][8]
        codes = q.transpose(2, 0, 3, 4, 1, 5)  # [sb][g][i][s][r][8]
        return [_bf16_bytes(f["d"]), np.ascontiguousarray(codes).view(np.uint8).reshape(nsb, -1)]

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        raise NotImplementedError("Q8_0 tiles: decode only (no prefill GEMM)")

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return True

    def mmv_split(self, fmt: str, nsb: int) -> int:
        return 16  # (each core one row group of every super-block)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/dtab", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb, self.mmv_split(fmt, nsb))
        assert R == 16, R
        return {"R80": R}


SLOT = Q8Slot()
