"""Q4_0 and Q4_1 tiles (bytes per row and super-block):
  Q4_0 (144): d [row][sb][i] bf16 | nibbles [row][s][i] words
  Q4_1 (160): d | m [row][sb][i] bf16 | nibbles

Prefill: tiles of 16 rows x 2048 or 1024 columns take the tile-major permutation (GemmQ, GemmQd; ``_hdr`` expands the
scale header: 1024 columns read it as a slice-sized unit, Q4_0's from 1 KB before the tile), any other (14 or 32
rows, or 512 columns) GemmQd's planar gather (the nibbles as 4 planes
[row][step][32 B]; Gemma's 512-column Q4_0 tiles of 32 rows: one header unit per 4 K chunks, ``GemmQOperator.hg``)."""

from typing import ClassVar

import numpy as np

from .common import (
    KC,
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

# GemmQ / GemmQd: the scale header's de-interleave and the tile-major dequant, or the planar kernels
_GEMM = r"""{% if not planar and nsb == 8 %}
// k-tile slot kt (0..3) of a scale-header plane (one {{ tq // 2 }} KB L1 object):
// [tile {{ tq }}][il 2][s 4][row 16] words of two bf16 (group pair h = 0, 1) ->
// ws [kt][il][s][tile][h][row] (de-interleaved; groups 16 s + 4 kt + 2 il + h
// of the tile row's 64 scales).
void {{ p }}_hdr(const bfloat16 *__restrict b, bfloat16 *__restrict ws, int32_t plane, int32_t kt)
{
    ws += {{ 1024 * tq }} * plane + {{ 256 * tq }} * kt;
    for (int tl = 0; tl < {{ tq }}; tl++) {
        bfloat16 *__restrict w = ws + tl * 32;
        const bfloat16 *__restrict src = b + tl * 8 * 32;
        AIE_LOOP_NO_UNROLL
        for (int is = 0; is < 8; is++) {  // (walked pointers: indexed src + 32 is, w + 128 is ran at II 7)
            aie::vector<bfloat16, 32> v = aie::load_v<32>(src);
            auto [e, o] = aie::interleave_unzip(v, v, 1);
            aie::store_v(w, aie::concat(e.template extract<16>(0), o.template extract<16>(0)));
            src += 32;
            w += {{ 32 * tq }};
        }
    }
}

{% include "gemm/dq_tile_major" %}
{% elif not planar %}
{% set parts = 2 if fmt == "Q4_1" else 1 %}
// k-tile slot kt of the header unit of 1024-column tiles (a {{ tq // 2 }} KB L1 object; {{ "d | m: the whole slice-sized unit" if parts == 2 else "d: the upper half of a slice-sized unit read from 1 KB before the tile" }}). The memtile's permutation
// puts row r of header word w (the bf16 pair of groups 2 w, 2 w + 1 of the row's 32) at permutation row {{ "8 P + r / 2 (part P)" if parts == 2 else "8 + r / 2" }},
// step 2 (r % 2) + w / 8 of k-tile slot (w % 8) / 2: slot kt holds the words 8 hw + 2 kt + j (hw, j = 0, 1), tile t's
// 8 even / odd rows of a word as 8 consecutive words -> ws [part][w][tile][h][row 16] (the dequant's layout: group 2 w + h
// = 8 sb + 2 kt' + h of super-block sb and k-tile kt')
void {{ p }}_hdr(const bfloat16 *__restrict b, bfloat16 *__restrict ws, int32_t plane, int32_t kt)
{
    const uint32_t *__restrict src = (const uint32_t *)b;
    bfloat16 *__restrict dst = ws + {{ 64 * tq }} * kt;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tq }}; t++) {
{% for part in range(parts) %}
{% for hw in range(2) %}
{% for j in range(2) %}
        {
            const uint32_t *__restrict w = src + {{ 64 * j + 16 * hw + (8 * part if parts == 2 else 8) }};
            auto [r0, r1] = aie::interleave_zip(aie::load_v<8>(w), aie::load_v<8>(w + 32), 1);
            const aie::vector<bfloat16, 32> d = aie::concat(r0, r1).template cast_to<bfloat16>();  // [row 16][h 2]
            aie::store_v(dst + {{ 512 * tq * part + (8 * hw + j) * 32 * tq }}, aie::filter_even(d, 1));
            aie::store_v(dst + {{ 512 * tq * part + (8 * hw + j) * 32 * tq + 16 }}, aie::filter_odd(d, 1));
        }
{% endfor %}
{% endfor %}
{% endfor %}
        src += 128;
        dst += 32;
    }
}

{% include "gemm/dq_tile_major" %}
{% else %}
{% set parts = 2 if fmt == "Q4_1" else 1 %}
// {{ fmt }} (GemmQd, planar: {{ R }}-row tiles of {{ 256 * nsb }} columns): weight objects [step s 4][word ii 2][g {{ n }}]
// words (the nibbles of step s of groups 2 kt + ii), rows g of the pass's tiles zero-padded to {{ n }}. The header
// ({{ hu }} unit{{ "s" if hu > 1 else "" }}: d{{ " | m" if parts == 2 else "" }} [row][sb][group] bf16, gathered like the codes, so a word holds the
// bf16 pair of groups 2 j, 2 j + 1 of one row) spans the four k-tile slots of each unit: _hcp collects them in hb
// ([unit][kt][s][ii][g] words), _hdr_sb expands a super-block's scales before its k-tiles into ws [group 8][d | m][g].
void {{ p }}_hcp(const bfloat16 *__restrict b, bfloat16 *__restrict hb, int32_t unit, int32_t kt)
{
    const bfloat16 *__restrict s = b;  // (walked pointers)
    bfloat16 *__restrict d = hb + {{ 16 * n }} * (4 * unit + kt);
    for (int i = 0; i < {{ 16 * n // 32 }}; i++) {
        aie::store_v(d, aie::load_v<32>(s));
        s += 32;
        d += 32;
    }
}

{% if nsb == 8 %}
// the header of 2048 columns: tile row r's 128 B per part are gathered row r: group pair j of super-block sb is
// word (sb % 2) 4 + j of step sb / 2, i.e. slot 4 part + 2 (sb % 2) + j / 2, word ii j % 2
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
    const uint32_t *__restrict src = hb + (sb & 1) * {{ 16 * n }} + (sb >> 1) * {{ 2 * n }};
    bfloat16 *__restrict dst = ws;
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < {{ n }}; g += 16) {
{% for part in range(parts) %}
{% for j in range(4) %}
        {
            const aie::vector<bfloat16, 32> v = aie::load_v<16>(src + {{ (4 * part + j // 2) * 8 * n + (j % 2) * n }}).template cast_to<bfloat16>();
            aie::store_v(dst + {{ 2 * j * 2 * n + part * n }}, aie::filter_even(v, 1));
            aie::store_v(dst + {{ (2 * j + 1) * 2 * n + part * n }}, aie::filter_odd(v, 1));
        }
{% endfor %}
{% endfor %}
        src += 16;
        dst += 16;
    }
}
{% elif nsb == 2 %}
{% set Rs = R // us %}
{% if hg > 1 %}
// the header of 512 columns: the unit holds the headers of up to {{ hg }} K chunks ([chunk q][tile row r][32 B]): tile row
// r's 32 B of chunk q are step s = r % 4 of unit row u = 8 q + r / 4, gathered at row (u / {{ Rs }}) {{ tiles * Rs }} + {{ Rs }} t + u % {{ Rs }}
// (sub-units); group pair j of super-block sb is word 4 sb + j (slot 2 sb + j / 2, word ii j % 2): per 8 unit rows the
// 4 steps' words, transposed [s][u] -> [u][s] = rows 4 u + s
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb, int32_t q)
{
    const uint32_t *__restrict src = hb + sb * {{ 16 * n }} + (q >> 1) * {{ tiles * Rs }} + (q & 1) * 8;
{% else %}
// the header of 512 columns: part P's 32 B of tile row r are step s = r % 4 of unit row u = {{ hrow0 }} + P R / 4 + r / 4
// (the unit starts {{ 128 * hrow0 }} B before the tile){{ ", gathered at row (u / %d) %d + %d t + u %% %d (sub-units)" % (Rs, tiles * Rs, Rs, Rs) if us > 1 else "" }}; group pair j of
// super-block sb is word 4 sb + j (slot 2 sb + j / 2, word ii j % 2): per 8 unit rows the 4 steps' words, transposed
// [s][u] -> [u][s] = rows 4 u + s
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
    const uint32_t *__restrict src = hb + sb * {{ 16 * n }};
{% endif %}
    bfloat16 *__restrict dst = ws;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tiles }}; t++) {
{% for uu in range(R // 32) %}
{% for part in range(parts) %}
{% set u0 = (0 if hg > 1 else hrow0) + part * R // 4 + 8 * uu %}
{% set g0 = (u0 // Rs) * tiles * Rs + u0 % Rs if us > 1 else u0 %}
{% for j in range(4) %}
        {
            const uint32_t *__restrict w = src + {{ (j // 2) * 8 * n + (j % 2) * n + g0 }};
            const aie::vector<uint32, 32> v = aie::concat(aie::load_v<8>(w), aie::load_v<8>(w + {{ 2 * n }}),
                                                          aie::load_v<8>(w + {{ 4 * n }}), aie::load_v<8>(w + {{ 6 * n }}));
            const aie::vector<bfloat16, 64> r = aie::transpose(v, 4, 8).template cast_to<bfloat16>();  // rows 32 uu ..
            aie::store_v(dst + {{ 2 * j * 2 * n + part * n + 32 * uu }}, aie::filter_even(r, 1));
            aie::store_v(dst + {{ (2 * j + 1) * 2 * n + part * n + 32 * uu }}, aie::filter_odd(r, 1));
        }
{% endfor %}
{% endfor %}
{% endfor %}
        src += {{ Rs if us > 1 else R }};
        dst += {{ R }};
    }
}
{% else %}
// the header of 1024 columns: part P's 64 B of tile row r start at byte {{ 64 * R }} P' + 64 r of the unit
// ({{ "the unit starts " ~ (64 * R) ~ " B before the tile, P' = 1" if fmt == "Q4_0" else "P' = P" }}): gathered row u = {{ R // 2 }} P' + r / 2, rows r = 2 v, 2 v + 1 in steps
// sb / 2 and 2 + sb / 2; group pair j is word (sb % 2) 4 + j (slot 2 (sb % 2) + j / 2, word ii j % 2)
void {{ p }}_hdr_sb(const uint32_t *__restrict hb, bfloat16 *__restrict ws, int32_t sb)
{
    const uint32_t *__restrict src = hb + (sb & 1) * {{ 16 * n }} + (sb >> 1) * {{ 2 * n }};
{% if R == 16 %}
    bfloat16 *__restrict dst = ws;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tiles }}; t++) {
{% for part in range(parts) %}
{% set u0 = 8 * (1 if fmt == "Q4_0" else part) %}
{% for j in range(4) %}
        {
            const uint32_t *__restrict w = src + {{ (j // 2) * 8 * n + (j % 2) * n + u0 }};
            auto [z0, z1] = aie::interleave_zip(aie::load_v<8>(w), aie::load_v<8>(w + {{ 4 * n }}), 1);
            const aie::vector<bfloat16, 32> v = aie::concat(z0, z1).template cast_to<bfloat16>();  // rows 0 .. 15
            aie::store_v(dst + {{ 2 * j * 2 * n + part * n }}, aie::filter_even(v, 1));
            aie::store_v(dst + {{ (2 * j + 1) * 2 * n + part * n }}, aie::filter_odd(v, 1));
        }
{% endfor %}
{% endfor %}
        src += 16;
        dst += 16;
    }
{% else %}
    // (rows of a 16-row block span tiles: 32-bit stores of row pairs)
    uint32_t *__restrict dst = (uint32_t *)ws;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ tiles }}; t++) {
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ R // 2 }}; v++) {
{% for part in range(parts) %}
{% set u0 = (R // 2) * (1 if fmt == "Q4_0" else part) %}
{% for j in range(4) %}
            {
                const uint32_t we = src[{{ (j // 2) * 8 * n + (j % 2) * n + u0 }} + v], wo = src[{{ (j // 2) * 8 * n + (j % 2) * n + u0 + 4 * n }} + v];
                dst[{{ (2 * j * 2 * n + part * n) // 2 }} + v] = (we & 0xFFFFu) | (wo << 16);
                dst[{{ ((2 * j + 1) * 2 * n + part * n) // 2 }} + v] = (we >> 16) | (wo & 0xFFFF0000u);
            }
{% endfor %}
{% endfor %}
        }
        src += {{ R }};
        dst += {{ R // 2 }};
    }
{% if n > TR %}
    // (the padding rows' scales: zero, so their MMUL columns are 0)
    uint32_t *__restrict wz = (uint32_t *)ws;
    AIE_LOOP_NO_UNROLL
    for (int g = {{ TR // 2 }}; g < {{ n // 2 }}; g++) {
{% for part in range(parts) %}
{% for i in range(8) %}
        wz[{{ (i * 2 * n + part * n) // 2 }} + g] = 0;
{% endfor %}
{% endfor %}
    }
{% endif %}
{% endif %}
}
{% endif %}

// k-tile kt of super-block sb -> {{ n }} x 64 bfp16 panels in the MAC pop order [16-row block][k-step 4 ii + s]
// [8-row half]: {{ "w = d q + m = (128 + q) d + (m - 128 d)" if parts == 2 else "w = d (q - 8) = (128 + q) d - 136 d" }}, exact in fp32, rounded once
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
{% if us == 1 %}
    b += half * {{ 4 * n_mac }};  // (the half's 16-row blocks)
{% endif %}
    ws += half * {{ n_mac }};
{% endif %}
    const bfloat16 __aie_dm_resource_c *__restrict pa = (const bfloat16 __aie_dm_resource_c *)(ws + 4 * kt * {{ n }});
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)stage);
    const aie::vector<uint8, 64> c43 = aie::broadcast<uint8, 64>(0x43);
{% if parts == 2 %}
    const aie::vector<bfloat16, 64> k128 = aie::broadcast<bfloat16, 64>(-128.0f);
{% else %}
    const aie::vector<bfloat16, 64> k136 = aie::broadcast<bfloat16, 64>(-136.0f);
{% endif %}
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int it = 0; it < {{ n_mac // 8 }}; it++) {  // (16-row block, word ii) = (it / 2, it % 2)
        const int nb = it >> 1, ii = it & 1;
        aie::vector<bfloat16, 64> D0, D1;
        al::expand16(aie::load_v<16>(pa + 2 * ii * {{ n }} + 16 * nb), D0, D1);
{% if parts == 2 %}
        aie::vector<bfloat16, 64> M0, M1;
        al::expand16(aie::load_v<16>(pa + (2 * ii + 1) * {{ n }} + 16 * nb), M0, M1);
        const aie::accum<accfloat, 64> C0 = aie::mac(aie::accum<accfloat, 64>(M0), D0, k128);  // (no ones vector: one spill less)
        const aie::accum<accfloat, 64> C1 = aie::mac(aie::accum<accfloat, 64>(M1), D1, k128);
{% else %}
        const aie::accum<accfloat, 64> C0 = aie::mul(D0, k136);
        const aie::accum<accfloat, 64> C1 = aie::mul(D1, k136);
{% endif %}
{% if us > 1 %}
        // (the gathered rows [sub-unit {{ us }}][tile {{ tiles }}][16 rows]: 16-row block nbn of the tiles' order is gb)
        const int nbn = {{ "half * %d + " % (n_mac // 16) if halves > 1 else "" }}nb;
        const int gb = (nbn & {{ us - 1 }}) * {{ tiles }} + (nbn >> {{ (us - 1).bit_length() }});
        const uint8_t *__restrict row = b + 4 * (ii * {{ n }} + 16 * gb);
{% else %}
        const uint8_t *__restrict row = b + 4 * (ii * {{ n }} + 16 * nb);
{% endif %}
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < 4; s++) {
            const aie::vector<uint8, 128> q = aie::unpack(aie::load_v<128>((const uint4 __aie_dm_resource_a *)(row + {{ 4 * 2 * n }} * s)));
            out.push(aie::mac(C0, al::zipw(q.template extract<64>(0), c43), D0).template to_vector<bfp16ebs8>());
            out.push(aie::mac(C1, al::zipw(q.template extract<64>(1), c43), D1).template to_vector<bfp16ebs8>());
        }
    }
{% if "ev_dq" in ablate %}
    event1();
{% endif %}
}

{% endif %}
"""


# the decode GEMV (``overlay.gemv``; the scale tables: ``mmv/dtab``): the codes' MMUL (``mmv/q4mm``), Gemma's row-split
# Q4_0 tiles of 512 columns, the super-block split of 1024 (its transposition: ``mmv/q4s``)
_MM = r"""// ---- Q4_0 / Q4_1: c_g,sb,i = bf16(C[hi rows] + C[lo rows]) -> c [sb][i][g][32]; C = -8 XS_i + sum_s X(sb, i, s)
// W_g(sb, s, i) (the codes centered) of NS super-blocks, w [sb][g][i][s][32 B] (each super-block ADV + 1024 B after the
// last)
template <int NS, int ADV>
__attribute__((noinline)) void {{ p }}_q40_mm(const uint8_t *__restrict xp, const uint8_t *__restrict xsum,
                                              const uint8_t *__restrict w, bfloat16 *__restrict c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BInR<aie_dm_resource::a> xs((const bfp16ebs8 *)xp);
    const uint8_t __aie_dm_resource_b *__restrict pw = (const uint8_t __aie_dm_resource_b *)w;
    bfloat16 *__restrict pc = c;
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < NS; sb++) {
        al::BIn<aie_dm_resource::d> xz((const bfp16ebs8 *)(xsum + {{ XS_SLOT }} * (sb >> 1)));  // (the pair's slot)
        v64bfp16ebs8 Z = xz.pop();
        if (sb & 1)
            Z = xz.pop();
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
        pw += ADV;  // (the next super-block's codes)
    }
}
"""

_Q4S = r"""// ---- Q4_0 / Q4_1 tiles of 1024 columns (R rows, row group 1 = rows 8 .. and junk): d [row][sb 4][i 8] bf16 (Q4_1: then
// m [row][sb][i]) | codes [sb][row R][128 B]. Core H: super-blocks 2 H, 2 H + 1 of every row (the K-quants' split).
// Scratch: the codes transposed [u 2][g 2][i 8][s 4][32 B] (4 KB) | tab [u][g][i][r] (512 B) | c [u][i][g][32] (2 KB)

// The codes of super-blocks 2 H, 2 H + 1 (src: the first's [row][128 B], the second rb B after it) -> dst
// [u][g][i][s][32 B]: panel (s, i) of row group g = word i of chunk s of its rows
__attribute__((noinline)) void {{ p }}_q4s_tr(const uint8_t *__restrict src, uint8_t *__restrict dst, int rb)
{
    const uint8_t *s = src;
    uint8_t *d = dst;
    const int sj = rb - 1024 - 96;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 16; t++) {  // (u, g, s): [row 8][128 B] blocks, chunk s at 32 s
        v16int32 o0, o1, o2, o3;
        {{ p }}_tr8<128>(s, o0, o1, o2, o3);
        {{ p }}_put8<128>(d, o0, o1, o2, o3);
        s += (t & 3) != 3 ? 32 : ((t & 7) != 7 ? 1024 - 96 : sj);
        d += (t & 3) != 3 ? 32 : 1024 - 96;
    }
}
"""

_MMV = r"""{% if NSB == 2 %}
// ---- Q4_0 ({{ R40 }} rows, 512 columns): d [row][sb][i] bf16 | codes [sb][row][128 B] = words (s 4, i 8) of 8 consecutive k
// (as Q4_K's). Core H: rows 16 H .. 16 H + 15 of every super-block (the row split). Scratch: tab [sb][g][i][r] (256 B
// per sb) | c [sb][i][g][32] (1 KB per sb)

// Pass 1: the core's codes in place, per (sb, g, s) -> panel (s, i) at row 8 g + i's chunk s ([sb][g][i][s][32 B])
__attribute__((noinline)) void {{ p }}_q40_tr(const uint8_t *__restrict src, uint8_t *__restrict dst)
{
    const uint8_t *s = src;
    uint8_t *d = dst;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < {{ 8 * NSB }}; t++) {  // (sb, g, s): 1 KB blocks [row 8][128 B], chunk s at 32 s
        v16int32 o0, o1, o2, o3;
        {{ p }}_tr8<128>(s, o0, o1, o2, o3);
        {{ p }}_put8<128>(d, o0, o1, o2, o3);
        const int step = (t & 3) != 3 ? 32 : ((t & 7) != 7 ? 1024 - 96 : {{ 128 * R40 - 1024 - 96 }});
        s += step;
        d += step;
    }
}

// rows 16 H .. 16 H + 15 of a Q4_0 tile: y[g][32] (rows 8 g ..) += their sums against xp (the chunk's panels)
template <int H>
__attribute__((noinline)) void {{ p }}_q40(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    uint8_t *codes = a + {{ 16 * NSB * R40 }} + 2048 * H;
    bfloat16 *tab = (bfloat16 *)scr;
    bfloat16 *c = (bfloat16 *)(scr + 1024);
    {{ p }}_q40_tr(codes, codes);
    {{ p }}_dtab<2>((const bfloat16 *)a + {{ 128 * NSB }} * H, tab);
    {{ p }}_q40_mm<{{ NSB }}, {{ 128 * R40 - 1024 }}>(xp, xs, codes, c);
    AIE_LOOP_NO_UNROLL
    for (int sb = 0; sb < {{ NSB }}; sb++)
        {{ p }}_sc1<2>(c + 512 * sb, tab + 128 * sb, y);
}
{% else %}
// ---- Q4_0 ({{ R40 }} rows): d [row][sb][i] bf16 | codes (mmv/q4s). y[g][32] (rows 8 g ..) += the rows of super-blocks 2 H,
// 2 H + 1 against xp (their 64 panels)
template <int H>
__attribute__((noinline)) void {{ p }}_q40(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    bfloat16 *tab = (bfloat16 *)(scr + 4096);
    bfloat16 *c = (bfloat16 *)(scr + 4608);
    {{ p }}_q4s_tr(a + {{ 64 * R40 }} + {{ 256 * R40 }} * H, scr, {{ 128 * R40 }});
    {{ p }}_dtab<2, 2>((const bfloat16 *)a + 16 * H, tab);
    {{ p }}_q40_mm<2, 1024>(xp, xs + {{ XS_SLOT }} * H, scr, c);
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++)
        {{ p }}_sc1<2>(c + 512 * u, tab + 128 * u, y);
}
{% endif %}
"""

_MMV41 = r"""// ---- Q4_1 ({{ R41 }} rows): d | m [row][sb][i] bf16 | codes (mmv/q4s); w = d (q - 8) + m + 8 d

// The offsets: y[g] += sum_i XS_i (m_i + 8 d_i) of super-blocks 2 H, 2 H + 1: the core's 32-k sum panels against M =
// m + 8 d [r][i] (fp32, exact) as bfp16 hi + lo B panels (~14 bits); y += their hi and lo rows' sums
__attribute__((noinline)) void {{ p }}_q41_off(const uint8_t *__restrict a, const uint8_t *__restrict xsum,
                                               float *__restrict y, int H)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::BIn<> xz((const bfp16ebs8 *)xsum);
    const v64bfp16ebs8 X0 = xz.pop(), X1 = xz.pop();
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < 2; g++) {
        aie::accum<accfloat, 64> Y = aie::zeros<accfloat, 64>();
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 2; u++) {  // (super-block 2 H + u: [row 8 g ..][i 8])
            const bfloat16 *d = (const bfloat16 *)a + 256 * g + 16 * H + 8 * u, *m = d + {{ 32 * R41 }};
            const aie::vector<bfloat16, 64> D = aie::concat(
                aie::load_v<8>(d), aie::load_v<8>(d + 32), aie::load_v<8>(d + 64), aie::load_v<8>(d + 96),
                aie::load_v<8>(d + 128), aie::load_v<8>(d + 160), aie::load_v<8>(d + 192), aie::load_v<8>(d + 224));
            const aie::vector<bfloat16, 64> M = aie::concat(
                aie::load_v<8>(m), aie::load_v<8>(m + 32), aie::load_v<8>(m + 64), aie::load_v<8>(m + 96),
                aie::load_v<8>(m + 128), aie::load_v<8>(m + 160), aie::load_v<8>(m + 192), aie::load_v<8>(m + 224));
            al::BV hi, lo;
            al::split_hilo(aie::mac(aie::accum<accfloat, 64>(M), D, aie::broadcast<bfloat16, 64>(8.0f)), nI, hi, lo);
            const v64bfp16ebs8 X = u ? X1 : X0;
            Y = aie::accum<accfloat, 64>(mac_8x8_8x8T(X, 1, lo, 1, mac_8x8_8x8T(X, 1, hi, 1, Y)));
        }
        {{ p }}_yadd(y + 32 * g, Y);
    }
}

// super-blocks 2 H, 2 H + 1 of a Q4_1 tile (d (q - 8) as Q4_0's, then the offsets)
template <int H>
__attribute__((noinline)) void {{ p }}_q41(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    bfloat16 *tab = (bfloat16 *)(scr + 4096);
    bfloat16 *c = (bfloat16 *)(scr + 4608);
    {{ p }}_q4s_tr(a + {{ 128 * R41 }} + {{ 256 * R41 }} * H, scr, {{ 128 * R41 }});
    {{ p }}_dtab<2, 2>((const bfloat16 *)a + 16 * H, tab);
    {{ p }}_q40_mm<2, 1024>(xp, xs + {{ XS_SLOT }} * H, scr, c);
    AIE_LOOP_NO_UNROLL
    for (int u = 0; u < 2; u++)
        {{ p }}_sc1<2>(c + 512 * u, tab + 128 * u, y);
    {{ p }}_q41_off(a, xs + {{ XS_SLOT }} * H, y, H);
}
"""


class Q4Slot(Slot):
    formats = (
        TileFormat("Q4_0", 0, 144, 16, 2, 128, 16, planes=4, ws_planes=1, dq_row_cycles=6.0, mmv="q40"),
        TileFormat("Q4_1", 1, 160, 16, 2, 128, 32, planes=4, ws_planes=2, dq_row_cycles=6.0, mmv="q41"),
    )

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        parts = [_bf16_bytes(f["d"])]
        if fmt == "Q4_1":
            parts.append(_bf16_bytes(f["m"]))
        parts.append(sb_major(_nibbles(f["q"][:, _word_order(nsb)]), R, nsb))
        return parts

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return not self.tile_major(fmt, rows, kc)

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        return rows == 16 and kc in (KC, KC // 2)

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        return nsb == 8  # (headers of 2048 columns; a half-slice header's row pairs are not)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        if not op.planar:
            return super().qd_scratch(op)
        # ws: the super-block's scales [group 8][d | m][g] bf16; hb: the header units' slots
        return {"ws": (1, 32 * op.n), "hb": (1, op.hdr_units * 4 * 4 * op.b_l1_words)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        if not op.planar:
            return super().qd_kernels(op, kernel, B_l1_ty=B_l1_ty, ws_ty=ws_ty, hb_ty=hb_ty, D_ty=D_ty)
        # the slots of every header unit collected, then expanded per super-block
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty, unit=True, chunk=op.hg > 1)
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True, unit=True)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        if op.planar and op.nsb == 2:  # (the header rows of a 512-column tile: whole unit rows of 128 B, 32-row blocks)
            assert op.hdr_skip % 128 == 0 and op.R % 32 == 0, (op.hdr_skip, op.R)
            assert op.hg == 1 or (op.R == 32 and op.us == 2 and op.fmt == "Q4_0"), (op.hg, op.R, op.us)
        return {"hrow0": op.hdr_skip // 128, "hg": op.hg}

    sources: ClassVar[dict[str, str]] = {"gemm/Q4_0": _GEMM, "gemm/Q4_1": _GEMM, "mmv/q4mm": _MM, "mmv/q4s": _Q4S,
                                         "mmv/Q4_0": _MMV, "mmv/Q4_1": _MMV41}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4 or (fmt == "Q4_0" and nsb == 2)

    def mmv_split(self, fmt: str, nsb: int) -> int:
        return 32 if nsb == 2 else 0  # (512 columns: each core 16 rows of both super-blocks)

    def mmv_helper_y(self, fmt: str, nsb: int) -> int:
        return 64 if nsb == 2 else 0

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/dtab", "mmv/q4mm", *(("mmv/q4s",) if nsb == 4 else ()), f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb, self.mmv_split(fmt, nsb))
        assert R == 32 if nsb == 2 else R <= 16, (fmt, nsb, R)
        return {"R40" if fmt == "Q4_0" else "R41": R}


SLOT = Q4Slot()
