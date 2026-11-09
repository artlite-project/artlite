"""IQ3_S tiles (R = 17 .. 20 rows in 3 row groups of 8, the third of r2 = R - 16 rows; 110 B per row and super-block,
the GGUF block's size): per super-block a unit of ``UNIT`` B (4 per 1024-column tile, nothing else in the tile) holding

- A (1536 B): the grid indices' low bytes as records [32-k group i 8][panel pair h 2][8-k panel l' 2] of 48 B: rows 0
  .. 7 (row r's two indices of panel l = 2 h + l' at 2 r, 2 r + 1), rows 8 .. 15, rows 16 .. R - 1 (2 r2 B), zeros, then
  at byte 40 the panels' 9th index bits, a 16-bit mask per row group g (bit 2 r + j: row 8 g + r's index j), zeros;
- B (768 B): the signs as records [i 8][panel l 4] of 24 B: rows 0 .. 7's sign bytes (bit j: value 8 l + j of group i
  negative), rows 8 .. 15, rows 16 .. R - 1, zeros, then at byte 20 a 4-byte slot of C;
- C (the 32 slots of B, slot n in record n): slot r < R row r's scale bytes (the GGUF block's: groups 2 j, 2 j + 1 in
  byte j's low and high nibble), slot R + s rows 2 s and 2 s + 1's d (bf16), zeros past them.

So each weight panel (8 rows x 8 k: 16 indices) has its index bytes in 16 contiguous bytes and its signs in 8 (rows past
R: other bytes, whose panel rows the kernels drop). w = d (1 + 2 sc) q, q = the 9-bit index's grid value
(``gguf_model.iq3s_grid``: 4 values per index, in {1, 3, .., 15}) negated where its sign is set.

Decode (``mmv/IQ3_S``): per super-block the panels' 9th bits as bytes 0 / 128 (``_iq3s_hb``: per 4 records their 12
masks as 3 x 64 bits), then per 4 panels (64 index bytes) the gather offsets 4 (qs + 256 qh) on the MMUL (``mul_elem`` +
``mac_elem``; a select from a mask in the gather loop leaves it unpipelined), 4 gathers of 16-bit entries per panel
(``grid_halves``: the grid's 4 values as nibbles, 2 KB per copy at ``common.TAB``), the nibbles unpacked, the signs by
``vaddsub8`` from the tile's masks; the MMUL and the scaling as IQ3_XXS's (``mmv/iq3c``), the scales 1 + 2 sc and d
from C's slots (``_iq3s_ss``).

Prefill (``gemm/IQ3_S``: GemmQd on super-block units, IQ3_XXS's structure): the pass's rows compacted (MMUL row q = tile
q / R's row q % R); per unit the scales S = bf16((1 + 2 sc) d) and the signs of the compacted rows; per k-tile the index
lists compacted (the low bytes and the 9th bits as bytes, 4 lists per iteration so that every window's shift is a
constant: the weight object, its records' k-tiles and the units are 64-byte aligned), per 4 panels the gather offsets 4 lo
+ 8 hb on the MMUL (the decode's form), the codes 64 +- g by 4x16 gathers of the nibble grid (unpacked), the panels w = S
(128 + u) - 192 S rounded once (as IQ3_XXS's ``_iq3s``)."""

from typing import ClassVar

import numpy as np

from . import k23
from .common import TAB, TAB_BYTES, DqKernels, Slot, TileFormat, dq_kernel, i32
from .iq3 import Iq3Slot

UNIT = 2304  # bytes of a super-block's unit (A + B)


def grid_halves() -> np.ndarray:
    """The grid as 16-bit entries for the decode's 4x16 gathers (``TAB_BYTES`` / 4 uint32): entry i (its 4 values,
    one nibble each, value j in nibble j) at byte 4 i of a table whose 32-byte groups of 16 entries are stored twice."""
    from ..gguf_model import iq3s_grid

    g = iq3s_grid().astype(np.uint16)
    e = g[:, 0] | g[:, 1] << 4 | g[:, 2] << 8 | g[:, 3] << 12
    return np.concatenate([e.reshape(32, 16)] * 2, 1).reshape(-1).view(np.uint32)


_MMV = r"""// ---- IQ3_S (formats/iq3s.py: {{ R3S }} rows, 3 row groups; rows past R: junk): per super-block a unit of 2304 B,
// [A: index records [i][h][l' 2] of 48 B (rows' low index bytes; at 40 the 9th bits, a 16-bit mask per row group) | B:
// sign records [i][l 4] of 24 B (8 B per row group's panel; at 20 a slot of C: the rows' scale bytes, then d)]. Core H:
// super-blocks 2 H, 2 H + 1; per super-block u: the 9th bits as bytes (_iq3s_hb), the weight panels (_iq3s_w), the
// scales (_iq3s_ss), the MMUL and y += hilo(sum_i c_i (1 + 2 sc_i)) d (mmv/iq3c).
// Scratch: w / c [i 8][l 4][g 3][64 B] (0 .. 6144) | hb [panel 96][16] (6144; then C's slots st (6144 .. 6272) and sct
// [i][g 4][r 8] bf16 (6272)) | the table's addresses (8000 .. 8008)

// The panels' 9th index bits (A's masks at 48 n + 40: [g0 | g1 | g2 | 0] 16 bits each) -> hb [panel][16]: 0 or 128 per
// index, panels in A's order (per 4 records 12 panels, their masks as 3 x 64 bits)
__attribute__((noinline)) void {{ p }}_iq3s_hb(const uint8_t *__restrict ix, uint8_t *__restrict hb)
{
    const v64uint8 k128 = broadcast_to_v64uint8((unsigned char)128), z = broadcast_zero_to_v64uint8();
    const uint64_t *__restrict pm = (const uint64_t *)(ix + 40);
    v64uint8 *__restrict po = (v64uint8 *)hb;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 8; t++) {
        const uint64_t L0 = pm[0], L1 = pm[6], L2 = pm[12], L3 = pm[18];  // (records 6 words apart)
        po[0] = ::sel(z, k128, L0 | L1 << 48);
        po[1] = ::sel(z, k128, L1 >> 16 | L2 << 32);
        po[2] = ::sel(z, k128, L2 >> 32 | L3 << 16);
        pm += 24;
        po += 3;
    }
}

// a panel from its 16 entries' addresses (lanes 2 r, 2 r + 1: row r's two) and its signs m (bit 8 r + j: value j of row
// r negative): 4 gathers, the entries' nibbles unpacked, zero - w where a sign is set
[[clang::always_inline]] inline v64int8 {{ p }}_iq3s_panel(v16int32 a, v2int32 m)
{
    const v64uint8 u = unpack((v64uint4){{ p }}_g16(a));
    return __builtin_aie2p_vaddsub8(broadcast_zero_to_v64int8(), (v64int8)u, m);
}

// The 96 weight panels of a super-block (ix: A, 4 panels per 64 bytes; hb: their 9th bits as bytes; sg: B, a panel's
// signs at 8 p) -> wp [i][l][g 3][64 B]
__attribute__((noinline)) void {{ p }}_iq3s_w(const uint8_t *__restrict ix, const uint8_t *__restrict hb,
                                              const uint8_t *__restrict sg, const uint32_t *__restrict tab_ab,
                                              const uint32_t *__restrict tab_cd, uint8_t *__restrict wp)
{
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v64int8 four = broadcast_to_v64int8(4), eight = broadcast_to_v64int8(8);
    const v64uint8 __aie_dm_resource_a *__restrict pi = (const v64uint8 __aie_dm_resource_a *)ix;
    const v64uint8 __aie_dm_resource_d *__restrict ph = (const v64uint8 __aie_dm_resource_d *)hb;
    const v8int32 __aie_dm_resource_b *__restrict ps = (const v8int32 __aie_dm_resource_b *)sg;
    v64int8 __aie_dm_resource_c *__restrict pw = (v64int8 __aie_dm_resource_c *)wp;
    AIE_TRY_INITIATION_INTERVAL(24)  // (post-RA II 29, the heuristic search's 40: qrs 24 = the gathers, unpacks, stores)
    for (int t = 0; t < 24; t++) {
        const v64acc32 q = mac_elem_64_2(*ph++, eight, mul_elem_64_2(*pi++, four));  // (4 (qs + 256 qh) of 4 panels)
        const v16int32 sv = set_v16int32(0, *ps++);
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++)
            pw[u] = {{ p }}_iq3s_panel(add((v16int32)extract_v16acc32(q, u), b16), ext_v2int32(sv, u));
        pw += 4;
    }
}

// C's slots (B + 24 n + 20) -> st [32] words; sct [i][g 4][r 8] = 1 + 2 sc (bf16, exact: bf16(128 + 1 + 2 sc) by its bits
// 0x4300 | (1 + 2 sc), minus 128) from st's [row 24][4 B] (rows past R: other slots), per row group a transpose
__attribute__((noinline)) void {{ p }}_iq3s_ss(const uint8_t *__restrict b, int32_t *__restrict st, bfloat16 *__restrict sct)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict pc = (const int32_t *)(b + 20);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int n = 0; n < 32; n++)
        st[n] = pc[6 * n];
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4301);
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k128(aie::broadcast<float, 32>(-128.0f));
    AIE_LOOP_UNROLL_FULL
    for (int g = 0; g < 3; g++) {
        const aie::vector<uint8, 64> v = aie::unpack(aie::load_v<64>((const uint4 *)(st + 8 * g)));  // ([r 8][i 8])
        const aie::vector<uint8, 64> v2 = aie::add(v, v);
        const aie::vector<bfloat16, 32> s0 = aie::mac(k128, aie::add(aie::unpack(v2.template extract<32>(0)), k43).template cast_to<bfloat16>(), one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> s1 = aie::mac(k128, aie::add(aie::unpack(v2.template extract<32>(1)), k43).template cast_to<bfloat16>(), one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> t = aie::transpose(aie::concat(s0, s1), 8, 8);  // ([i][r])
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < 8; i++)
            aie::store_v(sct + 32 * i + 8 * g, t.template extract<8>(i));
    }
}

// super-blocks 2 H, 2 H + 1 of an IQ3_S tile: y[g][32] (rows 8 g ..) += their rows against xp (their 64 panels)
template <int H>
__attribute__((noinline)) void {{ p }}_iq3s(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                            const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    uint8_t *hb = scr + 6144;
    int32_t *st = (int32_t *)(scr + 6144);
    bfloat16 *sct = (bfloat16 *)(scr + 6272);
    // (the table copies' addresses through a volatile word: constants at the call would specialize _iq3s_w, and the
    // gathers' memory operands need pointers)
    volatile int32_t *ta = (volatile int32_t *)(scr + 8000);
    ta[0] = H ? {{ TAB[1][0] }} : {{ TAB[0][0] }};
    ta[1] = H ? {{ TAB[1][1] }} : {{ TAB[0][1] }};
    const uint32_t *tab_ab = (const uint32_t *)(intptr_t)ta[0], *tab_cd = (const uint32_t *)(intptr_t)ta[1];
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++) {
        uint8_t *un = a + {{ UNIT }} * (2 * H + u);
        {{ p }}_iq3s_hb(un, hb);
        {{ p }}_iq3s_w(un, hb, un + 1536, tab_ab, tab_cd, scr);
        {{ p }}_iq3s_ss(un + 1536, st, sct);
        {{ p }}_iq3_x(xp + {{ 32 * PANEL }} * u, scr, (bfloat16 *)scr);
        {{ p }}_iq3_sc((const bfloat16 *)scr, sct, (const bfloat16 *)(st + {{ R3S }}), y);
    }
}
"""


# GemmQd (prefill) on super-block units: the pass's rows compacted (as IQ3_XXS's)
_GEMM = r"""// IQ3_S (GemmQd, super-block units: formats/iq3s.py): the weight object is a unit of the pass's {{ tiles }} tiles
// ({{ UP }} B each: [A: index records [i][h][l' 2] of 48 B (rows' low bytes [r][2], at 40 the 9th bits' masks) | B: sign
// records [i][l 4] of 24 B (rows' bytes, at 20 a slot of C: the rows' scale bytes, then d)]). Its rows compacted: MMUL row
// q is tile q / {{ R }}'s row q % {{ R }} ({{ n }} rows; past the tiles d = 0, so w = 0). Per unit _sb: C's slots straight to the
// compacted rows' scale words and d, the scales S = bf16((1 + 2 sc) d) [i][q]; per k-tile _dq (its first call of a k-tile:
// _prep, the k-tile's 8 index lists compacted, low bytes and 9th bits as bytes; _sbg, its records' signs compacted) the
// panels' codes (_iq3sg: per 4 panels the gather offsets on the MMUL, 4 gathers per panel of the nibble grid, unpacked, the
// signs as 64 +- g by vaddsub), then the panels (_iq3s). The compactions' windows have constant blocks and shifts: the
// weight object and the scratch are 64-byte aligned (pinned).
// Scratch (ws, bytes): S [i 8][{{ n }}] bf16 | the k-tile's signs [ks 8][{{ n }}] | at {{ WL }} the index lists' low bytes [ks
// 8][{{ LS }} B], then their 9th bits [ks 8][{{ LS }} B] (_sb's, before a unit's k-tiles: the scale words [{{ n }}]); hb: d [{{ n }}]
// bf16; wc: a dq call's codes [{{ n_mac // 16 }} blocks][ks 8][rh 2][64]

{% include "mmv/g16" %}
// S [i][q] = bf16((1 + 2 sc) d) of the compacted rows from their scale words (scw [q]: nibble i = sc of group i) and d (dc
// [q]): per 16 rows the nibbles [r][i] as bf16(128 + 1 + 2 sc) by their bits, minus 128, times d, per 8 rows a transpose
__attribute__((noinline)) void {{ p }}_sbs(const uint8_t *__restrict scw, const bfloat16 *__restrict dc, bfloat16 *__restrict S)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4301);
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k128(aie::broadcast<float, 32>(-128.0f));
    AIE_LOOP_NO_UNROLL
    for (int B = 0; B < {{ n // 16 }}; B++) {
        aie::vector<bfloat16, 64> D0, D1;
        al::expand16_bc(aie::load_v<16>(dc + 16 * B), D0, D1);
        AIE_LOOP_UNROLL_FULL
        for (int h = 0; h < 2; h++) {  // (rows 16 B + 8 h ..: nibbles [r 8][i 8])
            const aie::vector<uint8, 64> v = aie::unpack(aie::load_v<64>((const uint4 *)(scw + 64 * B + 32 * h)));
            const aie::vector<uint8, 64> v2 = aie::add(v, v);
            const aie::vector<bfloat16, 32> s0 = aie::mac(k128, aie::add(aie::unpack(v2.template extract<32>(0)), k43).template cast_to<bfloat16>(), one).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 32> s1 = aie::mac(k128, aie::add(aie::unpack(v2.template extract<32>(1)), k43).template cast_to<bfloat16>(), one).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 64> sd = aie::mul(aie::concat(s0, s1), h ? D1 : D0).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 64> tr = aie::transpose(sd, 8, 8);  // ([i][r])
            AIE_LOOP_UNROLL_FULL
            for (int i = 0; i < 8; i++)
                aie::store_v(S + {{ n }} * i + 16 * B + 8 * h, tr.template extract<8>(i));
        }
    }
}

// The signs of k-tile kt's compacted rows: per record (i, l) of B (8 kt ..) its rows' bytes -> pm [ks 8][{{ n }}] (ks = 4 i'
// + l); every window (tile t's rows of record 8 kt + r at 24 r + UP t - R t from the k-tile's 64-byte aligned 192 B of
// records) has a constant block and shift
__attribute__((noinline)) void {{ p }}_sbg(const uint8_t *__restrict b, uint8_t *__restrict pm, int kt)
{
    const v16int32 *__restrict pb = (const v16int32 *)(b + 1536 + 192 * kt);
    uint8_t *__restrict po = pm;
    {
{% for r8 in range(8) %}
{% for q0 in range(0, n, 64) %}
{% set nq = [64, n - q0] | min %}
        {   // (record 8 j + {{ r8 }}, rows {{ q0 }} .. {{ q0 + nq - 1 }})
{% for t, lo, hi in seg(q0, q0 + nq) %}
{% set K = 24 * r8 + (UP - R) * t + q0 %}
{% set w = "(v64int8)shift_bytes(pb[%d], pb[%d], %d)" % (K // 64, K // 64 + 1, K % 64) if K % 64 else "(v64int8)pb[%d]" % (K // 64) %}
{% if loop.first %}
            aie::vector<int8, 64> v({{ w }});
{% else %}
            v = aie::select(v, aie::vector<int8, 64>({{ w }}), aie::mask<64>::from_uint64({{ "0x%016Xull" % mask(lo - q0, 64) }}));
{% endif %}
{% endfor %}
            aie::store_v((int8_t *)(po + {{ n * r8 + q0 }}), v{{ ".template extract<32>(0)" if nq == 32 else "" }});
        }
{% endfor %}
{% endfor %}
    }
}

// One unit's C slots (B + 24 k + 20 of tile t: slot k < R row k's scale bytes, slot R + s rows 2 s, 2 s + 1's d) straight
// to the compacted rows' scale words (scw [q], q = R t + k) and d (dc [q] bf16), rows past the tiles d = 0; then S
__attribute__((noinline)) void {{ p }}_sbp(const uint8_t *__restrict b, uint8_t *__restrict ws, bfloat16 *__restrict dc)
{
    int32_t *__restrict scw = (int32_t *)(ws + {{ WU }});
    int32_t *__restrict dw = (int32_t *)dc;
{% if R * tiles < n %}
    for (int c = {{ (R * tiles) // 32 * 32 }}; c < {{ n }}; c += 32)
        aie::store_v(dc + c, aie::zeros<bfloat16, 32>());
{% endif %}
{% for t in range(tiles) %}
{% for k in range(R + R // 2) %}
{% if R * t + k < n or k >= R %}
{% if k < R %}
    scw[{{ R * t + k }}] = *(const int32_t *)(b + {{ UP * t + 1556 + 24 * k }});
{% elif R * t + 2 * (k - R) < n %}
    dw[{{ (R * t) // 2 + k - R }}] = *(const int32_t *)(b + {{ UP * t + 1556 + 24 * k }});
{% endif %}
{% endif %}
{% endfor %}
{% endfor %}
    {{ p }}_sbs((const uint8_t *)scw, dc, (bfloat16 *)ws);
}

void {{ p }}_sb(const uint8_t *__restrict b, bfloat16 *__restrict ws, bfloat16 *__restrict hb, int32_t sb)
{
{% if "hdr" in ablate %}
    return;
{% endif %}
    {{ p }}_sbp(b, (uint8_t *)ws, hb);
}

// k-tile kt's 8 index lists (ks = 4 i' + 2 h + l': record 8 kt + ks, tile t's 40 bytes [r][2] at its start) of the compacted
// rows: their low bytes (lo, [ks][LS]; _prepl) and their 9th bits as bytes 0 / 128 (hb, [ks][LS]; _preph: per tile its
// record's masks as bytes, compacted in registers); 4 lists per iteration, so that every window (tile t's rows of list ks
// at 48 ks + UP t - 40 t from the k-tile's 64-byte aligned records) has a constant block and shift
__attribute__((noinline)) void {{ p }}_prepl(const uint8_t *__restrict b, uint8_t *__restrict pl)
{
    const v16int32 *__restrict pb = (const v16int32 *)b;
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < 2; j++) {
{% for k4 in range(4) %}
{% for c in range(0, n, 32) %}
{% set nq = [32, n - c] | min %}
        {   // (list 4 j + {{ k4 }}, rows {{ c }} .. {{ c + nq - 1 }})
{% for t, lo, hi in seg(c, c + nq) %}
{% set K = 48 * k4 + UP * t - 40 * t + 2 * c %}
{% set lw = "(v64int8)shift_bytes(pb[%d], pb[%d], %d)" % (K // 64, K // 64 + 1, K % 64) if K % 64 else "(v64int8)pb[%d]" % (K // 64) %}
{% if loop.first %}
            aie::vector<int8, 64> v({{ lw }});
{% else %}
            v = aie::select(v, aie::vector<int8, 64>({{ lw }}), aie::mask<64>::from_uint64({{ "0x%016Xull" % mask(2 * (lo - c), 64) }}));
{% endif %}
{% endfor %}
            aie::store_v((int8_t *)(pl + {{ k4 * LS + 2 * c }}), v);
        }
{% endfor %}
{% endfor %}
        pb += 3;
        pl += {{ 4 * LS }};
    }
}

__attribute__((noinline)) void {{ p }}_preph(const uint8_t *__restrict b, uint8_t *__restrict ph)
{
    const v64uint8 k128 = broadcast_to_v64uint8((unsigned char)128), z = broadcast_zero_to_v64uint8();
    const uint8_t *__restrict pb = b;
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < 2; j++) {
{% for k4 in range(4) %}
{% for c in range(0, n, 32) %}
{% set nq = [32, n - c] | min %}
        {   // (list 4 j + {{ k4 }}, rows {{ c }} .. {{ c + nq - 1 }})
{% for t, lo, hi in seg(c, c + nq) %}
{% set d = 2 * c - 40 * t %}
            const v64uint8 h{{ t }} = ::sel(z, k128, *(const uint64_t *)(pb + {{ 48 * k4 + UP * t + 40 }}));
{% set hw = ("(v64int8)shift_bytes(h%d, z, %d)" % (t, d) if d > 0 else "(v64int8)h%d" % t) if d >= 0 else "(v64int8)shift_bytes(z, h%d, %d)" % (t, 64 + d) %}
{% if loop.first %}
            aie::vector<int8, 64> w({{ hw }});
{% else %}
            w = aie::select(w, aie::vector<int8, 64>({{ hw }}), aie::mask<64>::from_uint64({{ "0x%016Xull" % mask(2 * (lo - c), 64) }}));
{% endif %}
{% endfor %}
            aie::store_v((int8_t *)(ph + {{ k4 * LS + 2 * c }}), w);
        }
{% endfor %}
{% endfor %}
        pb += 192;
        ph += {{ 4 * LS }};
    }
}

__attribute__((noinline)) void {{ p }}_prep(const uint8_t *__restrict b, uint8_t *__restrict ws, int kt)
{
    {{ p }}_prepl(b + 384 * kt, ws + {{ WL }});
    {{ p }}_preph(b + 384 * kt, ws + {{ WL + 8 * LS }});
}

// a panel's codes 64 +- g (a: its 16 entries' addresses, m: its signs): 4 gathers of the nibble grid, unpacked, 64 - g
// where a sign is set
[[clang::always_inline]] inline v64int8 {{ p }}_iq3su(v16int32 a, v2int32 m)
{
    const v64uint8 u = unpack((v64uint4){{ p }}_g16(a));
    return __builtin_aie2p_vaddsub8(broadcast_to_v64int8(64), (v64int8)u, m);
}

// The codes of a dq call's panels (u [16-row block][ks 8][rh 2][64]): per (32 rows, list ks) the list's rows (lo: low bytes,
// hb: 9th bits as 0 / 128) as the gather offsets 4 lo + 8 hb on the MMUL + the copies' bases, their signs (mk [ks][q]: one
// 32-byte load), 4 panels (blocks 2 bp, 2 bp + 1, both halves)
__attribute__((noinline)) void {{ p }}_iq3sg(const uint8_t *__restrict lo, const uint8_t *__restrict hb,
                                             const uint8_t *__restrict mk, const uint32_t *__restrict tab_ab,
                                             const uint32_t *__restrict tab_cd, uint8_t *__restrict u)
{
    const v16int32 b16 = ::sel(broadcast_s32((long)tab_ab), broadcast_s32((long)tab_cd), 0xCCCC);
    const v64int8 four = broadcast_to_v64int8(4), eight = broadcast_to_v64int8(8);
    const v64uint8 __aie_dm_resource_a *__restrict pl = (const v64uint8 __aie_dm_resource_a *)lo;
    const v64uint8 __aie_dm_resource_d *__restrict ph = (const v64uint8 __aie_dm_resource_d *)hb;
    const v8int32 __aie_dm_resource_b *__restrict pm = (const v8int32 __aie_dm_resource_b *)mk;
    v64int8 __aie_dm_resource_c *__restrict pu = (v64int8 __aie_dm_resource_c *)u;
    // (lists: by a list, then the next 32 rows; signs: by a record, then the next 32 rows; panels: by a list's 2, then the
    // next two blocks' 32)
    dims_2d_t wl = dims_2d_from_steps(8, {{ LS // 64 }}, 1);
    dims_2d_t wh = dims_2d_from_steps(8, {{ LS // 64 }}, 1);
    dims_2d_t wm = dims_2d_from_steps(8, {{ n // 32 }}, 1);
    dims_2d_t wu = dims_2d_from_steps(8, 2, 32);
    for (int it = 0; it < {{ n_mac // 4 }}; it++) {  // (pre-RA II 31: qrs 24 = the gathers, unpacks and stores)  // (32 rows bp, list ks) = (it / 8, it % 8)
        const v64acc32 q = mac_elem_64_2(*ph, eight, mul_elem_64_2(*pl, four));
        const v16int32 sv = set_v16int32(0, *pm);
{% for j in range(4) %}
        pu[{{ (j // 2) * 16 + j % 2 }}] = {{ p }}_iq3su(add((v16int32)extract_v16acc32(q, {{ j }}), b16), ext_v2int32(sv, {{ j }}));
{% endfor %}
        pl = add_2d_ptr(pl, wl);
        ph = add_2d_ptr(ph, wh);
        pm = add_2d_ptr(pm, wm);
        pu = add_2d_ptr(pu, wu);
    }
}

// The panels from their codes u and the scales (S: the call's rows of group 2 kt, the next group's {{ n }} further): per
// (16-row block, group i') the two row groups' scales over their rows' lanes; per panel the codes as bf16 128 + u (zero-
// extended on the load, + 0x4300), w = S (128 + u) - 192 S = S (+- g), exact in fp32, rounded once
__attribute__((noinline)) void {{ p }}_iq3s(const uint8_t *__restrict u, const bfloat16 *__restrict S,
                                            uint8_t *__restrict stage)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const uint8_t __aie_dm_resource_a *__restrict pu = (const uint8_t __aie_dm_resource_a *)u;
    const bfloat16 __aie_dm_resource_b *__restrict ps = (const bfloat16 __aie_dm_resource_b *)S;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> out((bfp16ebs8 *)stage);
    const aie::vector<uint16, 32> k43 = aie::broadcast<uint16, 32>(0x4300);
    const aie::vector<bfloat16, 64> k192 = aie::broadcast<bfloat16, 64>(-192.0f);
    dims_2d_t wsc = dims_2d_from_steps(2, {{ n }}, 16);  // (i' by a group's rows, then the next block)
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int it = 0; it < {{ n_mac // 8 }}; it++) {  // (16-row block, i') = (it / 2, it % 2)
        aie::vector<bfloat16, 64> S0, S1;
        al::expand16_bc(aie::load_v<16>(ps), S0, S1);
        const aie::accum<accfloat, 64> C0 = aie::mul(S0, k192), C1 = aie::mul(S1, k192);
{% for j in range(8) %}
        out.push(aie::mac(C{{ j % 2 }}, aie::concat(aie::add(aie::unpack(aie::load_v<32>(pu + {{ 64 * j }})), k43).template cast_to<bfloat16>(),
                                         aie::add(aie::unpack(aie::load_v<32>(pu + {{ 64 * j + 32 }})), k43).template cast_to<bfloat16>()),
                          S{{ j % 2 }}).template to_vector<bfp16ebs8>());
{% endfor %}
        pu += 512;
        ps = add_2d_ptr(ps, wsc);
    }
}

// k-tile kt -> {{ n_mac }} x 64 bfp16 panels (rows {{ n_mac }} half ..) in the MAC pop order [16-row block][k-step ks][8-row
// half]: on its first call the k-tile's lists (_prep), the codes (_iq3sg), then the panels (_iq3s)
void {{ p }}_dq(const uint8_t *__restrict b, bfloat16 *__restrict ws, uint8_t *__restrict stage,
                int32_t sb, int32_t kt{{ half_arg }}, uint8_t *__restrict wc, const uint32_t *__restrict tab_ab,
                const uint32_t *__restrict tab_cd)
{
{% if "ev_dq" in ablate or "ev_kt" in ablate %}
    event0();
{% endif %}
{% if "dq" in ablate %}
    return;
{% endif %}
{% if halves > 1 %}
    if (half == 0) {
        {{ p }}_prep(b, (uint8_t *)ws, kt);
        {{ p }}_sbg(b, (uint8_t *)ws + {{ 16 * n }}, kt);
    }
    const int r0 = {{ n_mac }} * half;  // (the half's first row)
{% else %}
    {{ p }}_prep(b, (uint8_t *)ws, kt);
    {{ p }}_sbg(b, (uint8_t *)ws + {{ 16 * n }}, kt);
    const int r0 = 0;
{% endif %}
    uint8_t *wsb = (uint8_t *)ws;
    {{ p }}_iq3sg(wsb + {{ WL }} + 2 * r0, wsb + {{ WL + 8 * LS }} + 2 * r0, wsb + {{ 16 * n }} + r0, tab_ab, tab_cd, wc);
    {{ p }}_iq3s(wc, ws + {{ 2 * n }} * kt + r0, stage);
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

"""


class Iq3sSlot(Slot):
    formats = (TileFormat("IQ3_S", 13, 110, 20, 1, 110, 0, planes=3, unit_fixed=UNIT, dq_row_cycles=12.0, mmv="iq3s"),)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert 16 < R <= 20, R
        r2 = R - 16
        idx = np.zeros((24, nsb, 8, 2, 2, 2), np.uint16)  # [row][sb][i][h][l'][j]
        idx[:R] = f["idx"].reshape(R, nsb, 8, 2, 2, 2)
        a = np.zeros((nsb, 8, 2, 2, 48), np.uint8)
        low = (idx & 0xFF).astype(np.uint8).reshape(3, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6)  # [..][g][r][j]
        a[..., :32] = low[:, :, :, :, :2].reshape(nsb, 8, 2, 2, 32)
        a[..., 32:32 + 2 * r2] = low[:, :, :, :, 2, :r2].reshape(nsb, 8, 2, 2, -1)
        hi = (idx >> 8).astype(np.uint16).reshape(3, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6)
        mask = (hi.reshape(nsb, 8, 2, 2, 3, 16) << np.arange(16, dtype=np.uint16)).sum(-1, dtype=np.uint16)
        a[..., 40:46] = mask.view(np.uint8).reshape(nsb, 8, 2, 2, 6)
        sg = np.zeros((24, nsb, 8, 4), np.uint8)  # [row][sb][i][l]
        sg[:R] = f["signs"].reshape(R, nsb, 8, 4)
        b = np.zeros((nsb, 32, 24), np.uint8)
        s = sg.transpose(1, 2, 3, 0).reshape(nsb, 32, 24)  # [sb][i l][row]
        b[..., :16 + r2] = s[..., :16 + r2]
        c = np.zeros((nsb, 32, 4), np.uint8)
        sc = f["sc"].reshape(R, nsb, 4, 2)
        c[:, :R] = (sc[..., 0] | sc[..., 1] << 4).transpose(1, 0, 2)
        d = np.zeros((2 * -(-R // 2), nsb), np.uint16)
        d[:R] = k23.bf16_u16(f["d"]).astype(np.uint16)
        c[:, R:R + d.shape[0] // 2] = np.ascontiguousarray(d.T).view(np.uint8).reshape(nsb, -1, 4)
        b[..., 20:] = c
        return [np.concatenate([a.reshape(nsb, -1), b.reshape(nsb, -1)], 1).reshape(-1)]

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return UNIT

    sources: ClassVar[dict[str, str]] = {"mmv/IQ3_S": _MMV, "gemm/IQ3_S": _GEMM}

    # GemmQd (prefill): super-block units, rows compacted

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 7 units in the dequantizing core's L1: 32 KB, a gate / up pass of a whole SiLU
        # group of 128 rows; MMUL columns <= 128, in whole 32-row blocks: the dequant's 4-panel iterations)
        return tiles == 7 or tiles < 7 and -(-rows * tiles // 16) * 16 % 32 == 0

    def _ws(self, op: object) -> dict[str, int]:
        """Byte offsets in the prefill dequant's scratch (``gemm/IQ3_S``'s): WL the index lists' low bytes, then their 9th
        bits (LS bytes each: whole 64-byte blocks of 32 rows), at WU = WL between k-tiles the scale words (``_sb``); its
        size."""
        LS = Iq3Slot.list_bytes(-(-op.n // 32) * 32)
        WU = WL = 24 * op.n  # (_sb's scratch while no k-tile's lists are live)
        return {"WU": WU, "WL": WL, "LS": LS, "size": WL + max(16 * LS, 4 * op.n)}

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: S, a k-tile's signs, the lists (``_ws``); hb: d; ex: a dq call's codes; the grid's two copies (in two 16 KB
        # regions: ``gemm_qd``)
        return {"ws": (1, self._ws(op)["size"]), "hb": (1, max(2 * op.n, 64)), "ex": (1, 64 * op.nm),
                "tab": (1, TAB_BYTES), "tabb": (1, TAB_BYTES)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        # (_sbp: d pairs; _iq3sg: 32 rows per iteration)
        assert 16 < op.R <= 20 and op.R % 2 == 0 and op.n % 32 == 0 and op.nm % 32 == 0, (op.R, op.n, op.nm)
        # (the constant windows of _prep and _sbg: the weight objects and the scratch 64-byte aligned, as the dequantizing
        # core's layouts place every buffer)
        assert op.unit_pitch % 64 == 0, op.unit_pitch
        tab_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
        ex_ty = np.ndarray[(64 * op.nm,), np.dtype[np.uint8]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [ex_ty, tab_ty, tab_ty]), units=True,
                         ex_ty=ex_ty, tables=(tab_ty, grid_halves()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        R, T = op.R, op.tiles

        def seg(q0: int, q1: int) -> list[tuple[int, int, int]]:  # (tile, first row, end) of compacted rows q0 .. q1
            return [(t, max(q0, R * t), min(q1, R * (t + 1))) for t in range(q0 // R, min(T, -(-q1 // R)))]

        def mask(lo: int, hi: int) -> int:  # (bytes lo .. hi - 1 of 64)
            return ((1 << hi) - 1) & ~((1 << lo) - 1)

        return {"UP": op.unit_pitch, "seg": seg, "mask": mask, **self._ws(op)}

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/g16", "mmv/iq3c", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == 20, R  # (d of rows 2 s, 2 s + 1 at slot R + s: 16-byte aligned for the kernels' loads)
        return {"R3S": R, "TAB": TAB, "UNIT": UNIT}

    def mmv_table(self, fmt: str) -> np.ndarray:
        return grid_halves()


SLOT = Iq3sSlot()
