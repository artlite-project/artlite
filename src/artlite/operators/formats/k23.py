"""Shared parts of the Q2_K and Q3_K tile formats (``q2k``, ``q3k``; this slot has no formats of its own).

Tiles of up to 32 rows, stored word-major ([word][row][4 B]: a row group's word is one 32-byte load), the header words
first, then each super-block's code words. Sub-block j = 8 h + s of a super-block (half h, slot s) is k = 16 j + 8 a +
b (a: its two blocks of 8, b = 0 .. 7).

Prefill (GemmQd's planar gather, ``gemm/k23``): k-tile objects [word][row] (a k-tile is 4 Q2_K / 6 Q3_K words), the
header word-major too (row-direct, ``hdr_direct``) in slice-sized units from before the tile: ``_hcp`` collects them,
``_hdr_sb`` turns a super-block's scales into bf16 per row and sub-block (an MMUL against I), ``_dq`` expands them per
lane (``_x8``: a broadcast and two 8x8 transposes, 3 shuffles per 64 lanes, expand8's zips 5) and computes
w = S (128 + q) + C, rounded once to bfp16 (Q2_K 9, Q3_K 10 cycles per row and k-tile)."""

from typing import ClassVar

import numpy as np

from .common import DqKernels, Slot, _bf16_bytes, collected_header, dq_kernel


def words(nib: np.ndarray) -> np.ndarray:
    """(..., 8) nibbles -> (...) uint32 words, nibble j at bits 4 j."""
    return (nib.astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32))).sum(-1, dtype=np.uint32)


def bf16_u16(x: np.ndarray) -> np.ndarray:
    """fp32 -> the bits of its bf16 (as uint32)."""
    return _bf16_bytes(x[..., None]).view(np.uint16)[..., 0].astype(np.uint32)


def tile_parts(hw: np.ndarray, w: np.ndarray, R: int, nsb: int) -> list[np.ndarray]:
    """The parts of a tile: the header words (R, nw) and the code words (R, nsb, words), word-major."""
    codes = np.ascontiguousarray(w.reshape(R, nsb, -1).transpose(1, 2, 0))  # [sb][word][row]
    return [np.ascontiguousarray(hw.T).view(np.uint8), codes.view(np.uint8)]


# GemmQd (prefill): the planar kernels both formats share (``gemm/Q2_K``, ``gemm/Q3_K`` include them)
_GEMM = r"""// {{ fmt }} (GemmQd, planar: formats/k23.py): weight objects [word {{ W }}][g {{ n }}] (the k-tile's
// code words of rows g of the pass's tiles, zero-padded to {{ n }}). The header ({{ nw }} words per row, word-major like
// the codes: gathered row g holds the header of the same tile row) comes as {{ hu }} unit(s) of {{ cw }} words from
// {{ skip }} words before the tile (the previous tile's codes): _hcp collects the units' k-tile slots in hb ([word][g]:
// header word w at {{ hpad }} + w), _hdr_sb turns a super-block's scales into ws before its k-tiles.
void {{ p }}_hcp(const bfloat16 *__restrict b, bfloat16 *__restrict hb, int32_t unit, int32_t kt)
{
    const int w0 = {{ cw }} * unit + {{ W }} * kt - {{ skip - hpad }};  // (the hb word of the slot's first)
    if (w0 < 0)
        return;  // (codes of the previous tile)
    const bfloat16 *__restrict s = b;  // (walked pointers)
    bfloat16 *__restrict d = hb + {{ 2 * n }} * w0;
    for (int i = 0; i < {{ W * n // 16 }}; i++) {
        aie::store_v(d, aie::load_v<32>(s));
        s += 32;
        d += 32;
    }
}

using {{ p }}_BV = v64bfp16ebs8;
// [s]: the one-hot panel of column s (W [k][m] = 1 for m = s): X W^T repeats column s of X over each row's 8 lanes
alignas(64) static const int8_t {{ p }}_onehot[8][64] = {
{% for s in range(8) %}
    { {% for i in range(64) %}{{ 1 if i % 8 == s else 0 }}, {% endfor %}},
{% endfor %}
};

[[clang::always_inline]] inline v8int8 {{ p }}_e8(int e)
{
    v2int32 v;
    v[0] = e * 0x01010101;
    v[1] = e * 0x01010101;
    return (v8int8)v;
}

// the nibbles of 8 rows' words (32 B at w) as bytes [r][nibble]
[[clang::always_inline]] inline v64int8 {{ p }}_nib(const uint32_t *w)
{
    return (v64int8)unpack(*(const v64uint4 *)w);
}

// lanes [s][r]: the bf16 halves (lo, hi) of 8 rows' 32-bit words (32 B at w), the 8 values repeated 8 times
[[clang::always_inline]] inline void {{ p }}_d64(const uint32_t *w, aie::vector<bfloat16, 64> &lo,
                                                 aie::vector<bfloat16, 64> &hi)
{
    const aie::vector<bfloat16, 16> v = aie::load_v<16>((const bfloat16 *)w);
    const aie::vector<bfloat16, 16> l = aie::concat(aie::filter_even(v, 1), aie::filter_even(v, 1));
    const aie::vector<bfloat16, 16> h = aie::concat(aie::filter_odd(v, 1), aie::filter_odd(v, 1));
    const aie::vector<bfloat16, 32> l2 = aie::concat(l, l), h2 = aie::concat(h, h);
    lo = aie::concat(l2, l2);
    hi = aie::concat(h2, h2);
}

// lanes [r][8]: 8 values (r), each repeated over its row's 8 lanes (a broadcast and two 8x8 transposes)
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_x8(aie::vector<bfloat16, 8> v)
{
    const v32bfloat16 b = (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)v.template grow<32>(), 0);  // v 4 times
    return aie::concat(aie::vector<bfloat16, 32>(shuffle(b, b, T16_8x8_lo)), aie::vector<bfloat16, 32>(shuffle(b, b, T16_8x8_hi)));
}

// S = bf16(d T) (T: integer scales of 8 rows, lanes [s][r])
[[clang::always_inline]] inline void {{ p }}_scale64(v64accfloat T, aie::vector<bfloat16, 64> d64,
                                                     bfloat16 *__restrict S)
{
    aie::store_v(S, aie::mul(aie::accum<accfloat, 64>(T).template to_vector<bfloat16>(), d64).template to_vector<bfloat16>());
}

"""


# the decode GEMV (``overlay.gemv``): a row group's word (8 rows, 32 B) is 4-byte aligned only (R rows of 4 B per word):
# the header words a core reads come into its scratch (``_k23al``), the code words are realigned in the MMUL loops
# (``_k23w``)
_MMV = r"""// ---- Q2_K / Q3_K (formats/k23.py): n >= 2 words of R rows (wb = 4 R B each, from src) -> dst [word][32 rows] (128 B per
// word, rows R .. junk): per word three aligned 64-byte loads (the third only where the word reaches it: within the
// tile, whose end is 64-byte aligned) and two byte shifts
__attribute__((noinline)) void {{ p }}_k23al(const uint8_t *src, uint8_t *__restrict dst, int n, int wb)
{
    uintptr_t o = (uintptr_t)src;
    v16int32 *__restrict pd = (v16int32 *)dst;
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (int w = 0; w < n; w++) {
        const uintptr_t b = o & ~(uintptr_t)63;
        const unsigned sh = o & 63;
        const uintptr_t b2 = sh + wb > 128 ? b + 128 : b + 64;
        const v16int32 v0 = *(const v16int32 __aie_dm_resource_a *)b, v1 = *(const v16int32 __aie_dm_resource_a *)(b + 64);
        const v16int32 v2 = *(const v16int32 __aie_dm_resource_a *)b2;
        pd[0] = shift_bytes(v0, v1, sh);
        pd[1] = shift_bytes(v1, v2, sh);
        pd += 2;
        o += wb;
    }
}

// rows 0 .. 15 (lo) and 16 .. 31 (hi) of a word at o (4-byte aligned, within the tile, R > 16): the aligned 64-byte
// blocks around it shifted (the third only where the word reaches it)
[[clang::always_inline]] inline void {{ p }}_k23w2(uintptr_t o, int wb, v16int32 &lo, v16int32 &hi)
{
    const uintptr_t b = o & ~(uintptr_t)63;
    const unsigned sh = o & 63;
    const v16int32 v1 = *(const v16int32 __aie_dm_resource_b *)(b + 64);
    lo = shift_bytes(*(const v16int32 __aie_dm_resource_b *)b, v1, sh);
    hi = shift_bytes(v1, *(const v16int32 __aie_dm_resource_b *)(sh + wb > 128 ? b + 128 : b + 64), sh);
}

// 64 bytes at o (4-byte aligned) of which the first n are the tile's: the aligned 64-byte block, and the next one where
// those n reach it, shifted
[[clang::always_inline]] inline v16int32 {{ p }}_k23w1(uintptr_t o, int n)
{
    const uintptr_t b = o & ~(uintptr_t)63;
    const unsigned sh = o & 63;
    return shift_bytes(*(const v16int32 __aie_dm_resource_b *)b, *(const v16int32 __aie_dm_resource_b *)(sh + n > 64 ? b + 64 : b), sh);
}

// y += hilo(Y) d (32 lanes [slot][r]; d: the 8 rows' bf16)
[[clang::always_inline]] inline void {{ p }}_k23d(float *__restrict y, aie::accum<accfloat, 32> Y, const bfloat16 *d)
{
    const aie::vector<bfloat16, 32> h = Y.template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> l = aie::msc(Y, h, aie::broadcast<bfloat16, 32>(1.0f)).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> D = {{ p }}_rep4(d);
    aie::store_v(y, aie::mac(aie::mac(aie::accum<accfloat, 32>(aie::load_v<32>(y)), h, D), l, D).template to_vector<float>());
}

// y[g] += hilo(yu[g] + sum over the 16 sub-blocks j of c[j][g] sct[g][j][r]) dg[g] for 4 row groups g (Q3_K's three:
// the fourth junk): c [j][g 4][32] the sub-block sums (bf16), sct [g][j][r] the integer scales (bf16), yu [g][32] fp32 (the
// offsets), dg [g][r] the rows' d; the products exact, their sums in fp32, d as hi + lo once
__attribute__((noinline)) void {{ p }}_k23sc(const bfloat16 *__restrict c, const bfloat16 *__restrict sct,
                                             const float *__restrict yu, const bfloat16 *__restrict dg, float *__restrict y)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 32> Y0(aie::load_v<32>(yu)), Y1(aie::load_v<32>(yu + 32)), Y2(aie::load_v<32>(yu + 64)),
        Y3(aie::load_v<32>(yu + 96));
    const bfloat16 __aie_dm_resource_a *__restrict pc = (const bfloat16 __aie_dm_resource_a *)c;
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)sct;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < 16; j++) {
        Y0 = aie::mac(Y0, aie::load_v<32>(pc), {{ p }}_rep4(ps));
        Y1 = aie::mac(Y1, aie::load_v<32>(pc + 32), {{ p }}_rep4(ps + 128));
        Y2 = aie::mac(Y2, aie::load_v<32>(pc + 64), {{ p }}_rep4(ps + 256));
        Y3 = aie::mac(Y3, aie::load_v<32>(pc + 96), {{ p }}_rep4(ps + 384));
        pc += 128;
        ps += 8;
    }
    {{ p }}_k23d(y, Y0, dg);
    {{ p }}_k23d(y + 32, Y1, dg + 8);
    {{ p }}_k23d(y + 64, Y2, dg + 16);
    {{ p }}_k23d(y + 96, Y3, dg + 24);
}

"""

class K23Slot(Slot):
    sources: ClassVar[dict[str, str]] = {"gemm/k23": _GEMM, "mmv/k23": _MMV}


SLOT = K23Slot()


class PlanarK23:
    """GemmQd hooks of the Q2_K / Q3_K slots (their ``gemm/<fmt>`` kernels include ``gemm/k23``): header units read
    through the planar gather (row-direct: word-major), ``_hcp`` collects them, ``_hdr_sb`` computes a super-block's
    scales (``ws_bytes`` per 8-row block); and their decode GEMV's hooks (``mmv/k23``: the realignment)."""

    ws_bytes = 0

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R <= 32, (fmt, R)
        return {f"R{fmt[1]}": R, f"G{fmt[1]}": -(-R // 8)}

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        return True

    def _words(self, op: object) -> dict[str, int]:
        """Words per k-tile W and per slice cw, header words nw, skipped words before the header (whole k-tile slots
        are not collected) and hpad, the junk words hb starts with."""
        f = self.format(op.fmt)
        W, nw = 2 * f.planes, f.header_bytes(1, op.nsb) // 4
        skip = op.hdr_skip // (4 * op.R)
        return {"W": W, "cw": 4 * W, "nw": nw, "skip": skip, "hpad": skip % W}

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        w = self._words(op)
        # ws: the super-block's scales; hb: the header words [word][g]
        return {"ws": (1, self.ws_bytes * op.n // 8), "hb": (1, 4 * (w["nw"] + w["hpad"]) * op.n)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty, unit=True)
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True, unit=True)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        return self._words(op)
