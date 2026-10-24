"""IQ3_XXS tiles (98 B per row and super-block, the GGUF block's size; R <= 24 rows in 3 row groups of 8): the grid
indices [sb][32-k group i][panel pair h][row group g][8-k panel l' 2][row 8][2] (the GGUF block's qs: each row's two
indices of panel l = 2 h + l'; rows past R zero: 64 B per super-block and missing row, of the tile's padding), header
words [sb pair][row] (bf16 d of both super-blocks), the sign / scale words [sb][i][row]: byte l = the prefix XOR c of
group i's 7-bit sign field l (k 32 i + 8 l ..; c_j = f_0 ^ .. ^ f_j: the 8 signs, the 8th the parity of the field's,
are c ^ (c << 1)) | bit l of its scale ls at bit 7.

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
from .common import Slot, TileFormat

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


# the decode GEMV (``overlay.gemv``; the header's realignment and the scaling by d: ``mmv/k23``)
_MMV = r"""// ---- IQ3_XXS ({{ RI3 }} rows, 3 row groups; rows past R: junk): indices [sb][i][h][g][l' 2][r 8][2] | header
// words [sb pair][row] bf16 d (at {{ IX }}) | sign / scale words [sb][i][row] (byte l: the sign field c l | bit l of ls).
// Core H: super-blocks 2 H, 2 H + 1; per super-block u, in 4 loops (a weight panel built and multiplied in one loop
// leaves too few registers to pipeline the gathers):
// - _iq3_ss: per panel (group i, panel l, row group g) its signs, per group and row the scale (2 ls + 1) / 4;
// - _iq3_w: per panel the gather offsets 8 idx, 4 gathers of the grid, the signs: int8 weight panels;
// - _iq3_x: each x panel against the 3 row groups' weight panels: the groups' sums c, in place of their panels;
// - _iq3_sc: y[g] += hilo(sum_i c_i S_i) d.
// Scratch: w / c [i 8][l 4][g 3][64 B] (0 .. 6144) | mk [i][l 4][g 4][r 8] (6144) | sct [i][g 4][r 8] bf16 (7168) |
// hw [2][128 B] (7680) | dg [g 4][r 8] bf16 (7936) | the tables' addresses (8000 .. 8008)

// The signs and scales of 16 rows' sign / scale words v of a group ([r 16][l 4]): m = the signs [l][r 16]
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

// A super-block's signs and scales (w: its sign / scale words, [i][row] 4 R B per group): per group i rows 0 .. 15 (A)
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

// The weight panels of a super-block (96: wp [i][l][g 3]; ix: its indices [i][h][g][l' 2][r][2]): per (group i, panel
// pair h) and row group g the gather offsets 8 ix (one load: upshifted into an accumulator) + the copies' bases (lanes
// 0, 1: tab_ab, 2, 3: tab_cd), 4 gathers per panel, the signs (mk [i][l][g 4][r 8]: the pair's 64 bytes, one load)
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
        pw[0] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q0, 0), b16), ext_v2int32(m, 0));
        pw[3] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q0, 1), b16), ext_v2int32(m, 4));
        pw[1] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q1, 0), b16), ext_v2int32(m, 1));
        pw[4] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q1, 1), b16), ext_v2int32(m, 5));
        pw[2] = {{ p }}_iq3_panel(add((v16int32)extract_v16acc32(q2, 0), b16), ext_v2int32(m, 2));
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
    uint8_t *mk = scr + 6144, *hw = scr + 7680;
    bfloat16 *sct = (bfloat16 *)(scr + 7168), *dg = (bfloat16 *)(scr + 7936);
    // (the table copies' addresses through a volatile word: constants at the call would specialize _iq3_w, and the
    // gathers' memory operands need pointers, AIEPostSelectOptimize asserts on immediates)
    volatile int32_t *ta = (volatile int32_t *)(scr + 8000);
    ta[0] = H ? {{ TAB[1][0] }} : {{ TAB[0][0] }};
    ta[1] = H ? {{ TAB[1][1] }} : {{ TAB[0][1] }};
    const uint32_t *tab_ab = (const uint32_t *)(intptr_t)ta[0], *tab_cd = (const uint32_t *)(intptr_t)ta[1];
    {{ p }}_k23al(a + {{ IX }}, hw, 2, {{ 4 * RI3 }});  // (the header words: d of every row)
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        {{ p }}_iq3_ss(a + {{ IX + 8 * RI3 }} + {{ 32 * RI3 }} * (2 * H + u), (v16int32 *)mk, sct);
        AIE_LOOP_UNROLL_FULL
        for (int g = 0; g < 3; g++) {
            const aie::vector<bfloat16, 16> dd = aie::load_v<16>((const bfloat16 *)(hw + 128 * H + 32 * g));  // [r 8][2]
            aie::store_v(dg + 8 * g, aie::filter_even(aie::shuffle_down(dd, u), 1));
        }
        {{ p }}_iq3_w(a + 1536 * (2 * H + u), mk, tab_ab, tab_cd, scr);
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


class Iq3Slot(Slot):
    formats = (TileFormat("IQ3_XXS", 10, 98, 24, 1, 96, 2, planes=3, pad8_row=64, mmv="iq3"),)
    sources: ClassVar[dict[str, str]] = {"mmv/IQ3_XXS": _MMV, "mmv/iq3panel": _PANEL}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert nsb % 2 == 0, nsb
        d = k23.bf16_u16(f["d"])  # (R, nsb)
        hw = (d[:, 0::2] | (d[:, 1::2] << 16)).astype(np.uint32)
        G = -(-R // 8)
        idx = np.zeros((8 * G, nsb, 8, 4, 2), np.uint8)  # [row][sb][i][l][2]
        idx[:R] = f["idx"].reshape(R, nsb, 8, 4, 2)
        ix = idx.reshape(G, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 0, 5, 1, 6)  # [sb][i][h][g][l'][r][2]
        ss = ss_words(f["signs"], f["ls"]).reshape(R, nsb, 8)  # [row][sb][i]
        return [np.ascontiguousarray(ix).reshape(-1), np.ascontiguousarray(hw.T).view(np.uint8).reshape(-1),
                np.ascontiguousarray(ss.transpose(1, 2, 0)).view(np.uint8).reshape(-1)]

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/iq3panel", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert 16 < R <= 24, R  # (3 row groups: the kernels' layout)
        return {"RI3": R, "TAB": TAB, "IX": 512 * 3 * nsb}


SLOT = Iq3Slot()
