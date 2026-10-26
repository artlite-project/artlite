"""Prefill GEMM on the decode overlay's GGUF tiles: C = A @ W^T.

A: (M, K) bf16 activations (M = prompt tokens). W: tiles of the decode
overlay (``formats.pack_tile``: per-row scale planes, then the codes
of the super-blocks), read in place, so prefill and decode share one weight
file. C: (M, N) bf16.

Work split: core row r takes tokens [m r, m r + m), column c
the tiles of its row range, in passes of 4 tiles (64 rows for R = 16). Per
pass the tiles are streamed per K chunk of 2048 as 9 memtile objects of 4 x
2 KB (the scale header, then super-blocks 0..7); each is cut into four 2 KB
L1 objects (k-tiles of 64 k).

The memtile does the transpose the MMUL needs: tile rows store 32-bit words
of 8 consecutive k (word (s, i) = k in [32 i + 8 s, +8) of a super-block), a
bfp16 B panel holds 8 rows x 8 k. The memtile S2MM writes each super-block as
[tile][i][s][row] and the MM2S reads it as [kt][tile][i % 2][s][row] (i = 2 kt
+ i % 2), i.e. every L1 k-tile arrives as 32-byte packed panels in the
MAC loop's B pop order (``mac.py``) [16-row block][k-step][8-row half][row][word].

The core rows of a column dequantize in pairs (rows 2 / 3, 4 / 5): the lower core of a pair converts the pass's
tiles 0, 1 of every k-tile, the upper one tiles 2, 3 (the memtile ``split``s each k-tile object between them), each
into bfp16 panels in its own memory (zip each code byte with 0x43 -> bf16 128 + q, times the expanded group scales
plus -136 d: exactly d (q - 8) in fp32, one rounding) that it hands to its partner through a FIFO (adjacent cores:
shared memory, no DMA). Both run the MAC loop (``mac.py`` split; A as bfp16 panels from DDR) over their own half,
then the partner's, so the pair never streams B from one memory at once, into the L1 C tile (bf16 between
k-tiles). The k-tiles alternate between two C tiles in different 16 KB regions (in one region the MAC loop's C loads
and stores conflict, ~90 per k-tile); the second one (``st``) also collects the Q4_K header at a K chunk's start and
is the SiLU epilogue's scratch (the sum is in the first after every even number of k-tiles). The scale header passes
through the same permutation; ``_hdr`` (Q4_K: ``_hcp`` + ``_hdr_q4k``) expands the core's two tiles' scales into a
workspace. (Every core converting whole k-tiles cost ~980 instead of ~800 cycles per k-tile: 9B qwen35 gate/up 2.19 ->
1.71 ms.) BFP16 tiles hold MMUL operands: the MAC reads the k-tile objects.

Tokens come in chunks of 4 m (m <= 64 per core row, the largest that fits L1
(``_l1_bytes``); the weights stream once per chunk; the weights reach the cores
in 2 KB k-tile objects so that m = 64 fits); ``nrows = 2`` halves the program
images for short prompts (each op of a fused ELF is a reconfiguration). The core
buffers are pinned (``_l1_layout``): each stream of the MAC loop (A, the
dequantized half, the two C tiles) has its own 16 KB bank region (memory stalls per k-tile at m = 64 with
one C tile: 82 cycles instead of 222 unpinned). At m = 64 the A stream (4.6 KB per k-tile, ~580 cycles of a
64-bit DMA channel) is not the limit: the dequant + MAC take ~850 cycles per k-tile. A comes as bfp16ebs8
panels [M / 8][K / 8][72 B] (half the bytes of bf16, which would bound 32 tokens per core row: 128 KB
per pass; resident in the memtile up to ``A_RESIDENT_MAX``, else refetched per pass), ``c_bfp16``
writes the gate/up SiLU output in the same layout for the down GEMM.

Tiles of 1024 columns (``kc=1024``: the Llama and Qwen images): a row tile's K chunks are consecutive tiles
(``kc_group``: in groups of the decode's activation objects for down). A Q4_K tile's header (1 KB)
is half a slice; it is read as a slice-sized unit starting 1 KB before the tile (``hdr_skip``: every
unit the memtile permutes is then one slice of the pass's tiles; ~10% more weight bytes). The
per-pass pattern [chunk][unit][tile][words] has more dimensions than a shim BD (3 + repeat), so the
weights of all passes come as one chained task per column and token chunk (``b_chain``: one BD per
K chunk, each iterating over the passes): per-pass fills would need a task each (per-pass groups
blew the prefill's control code past a 64 MB heap BO), and more than 4 queued fills on a
channel stall the controller before the A fills are issued (a hang).
``gemm_qd.GemmQdOperator`` runs the dequant on one core per column instead of on every core; it
also reads every other dequantized tile through the planar gather (``planar``: Q5_K, Q6_K, Q2_K, Q3_K, and
Q4_0 / Q4_1 tiles of other than 16 rows or of 512 columns; ``tiles`` per pass, ``blocks``, ``c_pack``,
``region_rt``: see GemmQd).
"""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, ScratchpadParameter, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ..formats import FORMATS, KC, KT, kernel_template, slot
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS, ROWS, range32, stream_walk, times_i32
from ..operators.kernel_source import kernel_prefix
from ..operators.l1_layout import pin_l1_buffers
from .mac import MAC, render_mac

MACF = kernel_prefix("macf")  # (the fp32-C MAC helper: ``c_pairs``)

TILES = 4  # tiles per pass
A_RESIDENT_MAX = 448 * 1024  # bytes of a core row's A strip kept in its memtile (one buffer: the
# memtile resends it per pass with a BD repeat; depth x repeat BDs must stay within 24 per channel; one
# memtile BD moves < 512 KB (17-bit length; 442 KB strips run) and a split chain cannot carry the repeat)
MEMTILE_BYTES = 512 * 1024
M_MAX = 64  # GemmQ's tokens per core row and chunk (L1: A and C scale with it; see _l1_bytes)
L1_BUDGET = 62 * 1024  # bytes of core data memory the buffers may use (incl. the 2 KB stack)
B_DEPTH = 2  # L1 weight objects (one k-tile each)


class L1Error(ValueError):
    """The design's buffers do not fit a core's data memory (e.g. a GemmQd ``m`` too large for the pass)."""


@dataclass(frozen=True)
class PackPlan:
    """A C row's 8-value items (``pack2``) from two slots' MAC rows: per item its segments (slot, window, lanes: a bit
    mask): lane l of the segment's lanes is the slot row's value window + l; ``halves``: some item needs the rows as
    32-lane halves (a window not a whole sub-block row)."""

    items: tuple[tuple[tuple[int, int, int], ...], ...]
    halves: bool


def pack_plan(n: int, n_out: int, n_c: int, tiles: int, R: int) -> PackPlan:
    """The items of a C row of ``n_c`` columns: per tile t [slot 0's R rows | slot 1's], value t R + r of a slot's
    MAC row (n columns); the columns past n_out repeat the last item's segment."""
    assert n_out == 2 * tiles * R and n_c % 8 == 0 and n <= 64
    items = []
    for k in range(n_c // 8):
        segs: list[list[int]] = []
        for lane in range(8):
            o = min(8 * k + lane, n_out - 1)
            t, q = divmod(o, 2 * R)
            s_, r = divmod(q, R)
            w = t * R + r - lane
            if segs and segs[-1][0] == s_ and segs[-1][1] == w:
                segs[-1][3] = lane + 1
            else:
                segs.append([s_, w, lane, lane + 1])
        items.append(tuple((s_, w, (1 << hi) - (1 << lo)) for s_, w, lo, hi in segs))
    halves = any(len(it) > 1 or it[0][1] % 8 or not 0 <= it[0][1] <= n - 8 for it in items)
    return PackPlan(tuple(items), halves)


LINE = 32  # C columns of a 64-byte memory line


def line_carry(out_cols: list[list[int]], n_out: int) -> tuple[tuple[int, int], ...] | None:
    """Whole-line C drains of packed passes (``lines``): per pass (cr, sh), its first column's offset in its line (the
    previous pass's last cr columns, which its drain repeats) and where they start in the previous pass's row (0: a
    pass that starts a run of contiguous passes, at a line boundary). None when a run starts inside a line or the
    columns' passes differ (their windows would not be one design's)."""
    out = None
    for cols in out_cols:
        carry = []
        for p, o in enumerate(cols):
            cr = o % LINE if p and o == cols[p - 1] + n_out else 0
            if (o - cr) % LINE:
                return None
            carry.append((cr, (o - cr) - (cols[p - 1] - carry[-1][0]) if cr else 0))
        if out is not None and tuple(carry) != out:
            return None
        out = tuple(carry)
    return out


def two_parts(out_cols: list[list[int]], halves: int) -> bool:
    """Whether every C object of 32-output bfp16 parts can hold two from a 64-output boundary (576-byte panel rows,
    whole lines: ``lines``): a pass's halves, or the passes in pairs (an even count, each pair's outputs contiguous)."""
    for cols in out_cols:
        if halves == 2:
            if any(o % 64 for o in cols):
                return False
        elif len(cols) % 2 or any(cols[p] % 64 or cols[p + 1] != cols[p] + 32 for p in range(0, len(cols), 2)):
            return False
    return True


def line_window(carry: tuple[tuple[int, int], ...], n_out: int) -> int:
    """Columns of every pass's whole-line drain: the most a pass carries and its own columns, in whole lines."""
    return -(-(max(cr for cr, _ in carry) + n_out) // LINE) * LINE


def planar(fmt: str, rows: int, kc: int) -> bool:
    """GemmQdOperator reads these tiles through the planar gather (any row count: the memtile collects a
    k-tile's 32-byte planes, or words of word-major tiles, of the pass's rows, ``TileFormat.planes``): Q5_K, Q6_K,
    Q2_K, Q3_K, and Q4_0 / Q4_1 tiles of other than 16 rows or of 512 columns."""
    return slot(fmt).planar(fmt, rows, kc)


def tile_major(fmt: str, rows: int, kc: int) -> bool:
    """GemmQ and GemmQd read these tiles through the tile-major permutation (16-row blocks of words, MMUL
    panels): 16-row Q4_0 / Q4_1 tiles of 2048 or 1024 columns, 16-row Q4_K, BFP16."""
    return slot(fmt).tile_major(fmt, rows, kc)

# the dequant kernels of the format (``gemm/<fmt>``) are its slot's fragment (``formats``)
_KERNEL_TPL = kernel_template(r"""

namespace {

{{ mac_src }}
{% if silu and act == "gelu" %}
{% set gact = km.act(km.GELU, table_arg=True, walk=(2, 128)) %}
{{ gact.src }}
{% endif %}
{# the gate/up epilogues' last loop: h = y up (y in the scratch, up the C tile's up halves) -> hout as bfp16 panels #}
{% macro times_up(mp_) %}
    {
        const bfloat16 __aie_dm_resource_a *__restrict pu = (const bfloat16 __aie_dm_resource_a *)cacc + 64;
        const bfloat16 __aie_dm_resource_d *__restrict py = (const bfloat16 __aie_dm_resource_d *)scratch;
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::b> out((bfp16ebs8 *)hout);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int i = 0; i < {{ mp_ * n // 128 }}; i++) {
            aie::vector<bfloat16, 64> hv = aie::mul(aie::load_v<64>(py), aie::load_v<64>(pu)).template to_vector<bfloat16>();
            out.push(aie::accum<accfloat, 64>(hv).template to_vector<bfp16ebs8>());
            pu += 128;
            py += 64;
        }
    }
{%- endmacro %}

}  // namespace

extern "C" {

{% if csplit %}
// (the tile's halves {{ csplit }} bytes apart: rows {{ m // 2 }}.. after the first half's 16 KB region)
void {{ p }}_zero(bfloat16 *__restrict out)
{
    al::zero<{{ m * n // 2 }}, 32>(out);
    al::zero<{{ m * n // 2 }}, 32>(out + {{ csplit // 2 }});
}
{% else %}
void {{ p }}_zero(bfloat16 *__restrict out)
{
    al::zero<{{ m * n }}, 32>(out);
}
{% endif %}

{# the dequantizing kernels: n = the pass's MMUL columns, a dq call converts n_mac of them (halves: one half) #}
{% set n_mac = n %}
{% set n = nd %}
{% set half_arg = ", int32_t half" if halves > 1 else "" %}

{% include "gemm/" ~ fmt %}

{% set n = n_mac %}
{% if c_pack and carry %}
{% set nv = (n_c + 31) // 32 %}
// whole-line drains: C (m x {{ n_c }} row-major; the memtile drains windows of {{ win }} columns from each row) row t =
// [the previous pass's last cr columns, from column sh of the row it left in this C object | the pass's {{ n_out }}
// columns of the MAC tile (m x {{ n }}, 8x8 row-major sub-blocks)]; (cr, sh) of each pass from the tables
static const int8_t {{ p }}_cr[{{ carry | length }}] = { {% for cr, sh in carry %}{{ cr }}{{ ", " if not loop.last }}{% endfor %} };
static const int16_t {{ p }}_sh[{{ carry | length }}] = { {% for cr, sh in carry %}{{ sh }}{{ ", " if not loop.last }}{% endfor %} };

void {{ p }}_pack(const bfloat16 *__restrict acc, bfloat16 *c, int32_t pass)
{
    const int cr = {{ p }}_cr[pass];
    const int sh = {{ p }}_sh[pass];
    if (cr == 0) {  // (the pass's columns from column 0: 16-byte items, the source walking [band][row][block])
        const bfloat16 __aie_dm_resource_a *__restrict src = (const bfloat16 __aie_dm_resource_a *)acc;
        bfloat16 *__restrict dst = c;
        dims_3d_t d = dims_3d_from_steps({{ n_c // 8 }}, 64, 8, 8, {{ 8 * n }});
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int i = 0; i < {{ m * n_c // 8 }}; i++) {
            aie::store_v(dst, aie::load_v<8>(src));
            dst += 8;
            src = add_3d_ptr(src, d);
        }
        return;
    }
    const aie::vector<bfloat16, 8> z8 = aie::zeros<bfloat16, 8>();
    const aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
    const aie::mask<32> head = aie::mask<32>::from_uint32((1u << cr) - 1u);
    const unsigned up = 2 * (32 - cr);  // (bytes: the own columns move cr lanes up)
    const int ks = std::min(sh, {{ n_c - 32 }});  // (the carry's 4 items within the row: its columns are)
    const unsigned kb = 2 * (sh - ks);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < {{ m }}; t++) {
        const bfloat16 __aie_dm_resource_a *s = (const bfloat16 __aie_dm_resource_a *)acc + (t / 8) * {{ 8 * n }} + (t % 8) * 8;
        bfloat16 *row = c + t * {{ n_c }};
        // (own columns 32 j.. : 8-column items of the token's sub-block row; zeros past the MAC tile)
{% for j in range(nv) %}
        const aie::vector<bfloat16, 32> o{{ j }} = aie::concat({% for i in range(4) %}{{ "aie::load_v<8>(s + %d)" % (64 * (4 * j + i)) if 4 * j + i < n // 8 else "z8" }}{{ ", " if not loop.last }}{% endfor %});
{% endfor %}
        // (the carry, loaded before the stores overwrite where it was)
        const aie::vector<bfloat16, 32> k = aie::vector<bfloat16, 32>(::shift_bytes(aie::concat({% for i in range(4) %}aie::load_v<8>(row + ks + {{ 8 * i }}){{ ", " if not loop.last }}{% endfor %}), z, kb));
{% for j in range(nv) %}
{% if j == 0 %}
        const aie::vector<bfloat16, 32> w0 = aie::select(aie::vector<bfloat16, 32>(::shift_bytes(z, o0, up)), k, head);
{% else %}
        const aie::vector<bfloat16, 32> w{{ j }} = aie::vector<bfloat16, 32>(::shift_bytes(o{{ j - 1 }}, o{{ j }}, up));
{% endif %}
{% for i in range(4) if 32 * j + 8 * i < n_c %}
        aie::store_v(row + {{ 32 * j + 8 * i }}, w{{ j }}.extract<8>({{ i }}));
{% endfor %}
{% endfor %}
    }
}

{% elif c_pack %}
// C (m x {{ n_c }} row-major; the memtile drains its first {{ n_out }} columns) = the first {{ n_c }} columns of the
// MAC tile (m x {{ n }}, 8x8 row-major sub-blocks): 16-byte items (8 columns of a sub-block row), the source
// walking [band][row][block]
void {{ p }}_pack(const bfloat16 *__restrict acc, bfloat16 *__restrict c)
{
    const bfloat16 __aie_dm_resource_a *__restrict src = (const bfloat16 __aie_dm_resource_a *)acc;
    bfloat16 *__restrict dst = c;
    dims_3d_t d = dims_3d_from_steps({{ n_c // 8 }}, 64, 8, 8, {{ 8 * n }});
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < {{ m * n_c // 8 }}; i++) {
        aie::store_v(dst, aie::load_v<8>(src));
        dst += 8;
        src = add_3d_ptr(src, d);
    }
}

{% if slots == 2 %}
{% macro window(s, w) -%}
{%- if w == 0 %}h{{ s }}0{% elif w == 32 %}h{{ s }}1{% elif w < 0 %}aie::shuffle_up_fill(h{{ s }}0, z, {{ -w }})
{%- elif w < 32 %}aie::shuffle_down_fill(h{{ s }}0, h{{ s }}1, {{ w }}){% else %}aie::shuffle_down_fill(h{{ s }}1, z, {{ w - 32 }}){% endif %}
{%- endmacro %}
// row groups, two slots per C object: C (m x {{ n_c }} row-major) = per tile t of the pass [slot 0's {{ R }} rows |
// slot 1's], from the slots' MAC tiles a0, a1 (m x {{ n }}, 8x8 row-major sub-blocks): per token row its 8-value
// items (``pack_plan``), each a sub-block row of one slot or lanes of two slots' rows shifted into place (a row's
// values as two 32-lane halves h<slot>0, h<slot>1)
void {{ p }}_pack2(const bfloat16 *__restrict a0, const bfloat16 *__restrict a1, bfloat16 *__restrict c)
{
{% if pack_plan.halves %}
    const aie::vector<bfloat16, 8> z8 = aie::zeros<bfloat16, 8>();
    const aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
{% endif %}
    bfloat16 *__restrict dst = c;
    for (int rb = 0; rb < {{ m // 8 }}; rb++)
        for (int rr = 0; rr < 8; rr++) {
            const bfloat16 *__restrict s0 = a0 + rb * {{ 8 * n }} + rr * 8;
            const bfloat16 *__restrict s1 = a1 + rb * {{ 8 * n }} + rr * 8;
{% if pack_plan.halves %}
{% for sl in (0, 1) %}
            const aie::vector<bfloat16, 32> h{{ sl }}0 = aie::concat({% for j in range(4) %}{{ "aie::load_v<8>(s%d + %d)" % (sl, 64 * j) if j < n // 8 else "z8" }}{{ ", " if not loop.last }}{% endfor %});
            const aie::vector<bfloat16, 32> h{{ sl }}1 = aie::concat({% for j in range(4, 8) %}{{ "aie::load_v<8>(s%d + %d)" % (sl, 64 * j) if j < n // 8 else "z8" }}{{ ", " if not loop.last }}{% endfor %});
{% endfor %}
{% endif %}
{% for item in pack_plan.items %}
{% if item | length == 1 and item[0][1] % 8 == 0 and 0 <= item[0][1] <= n - 8 %}
            aie::store_v(dst + {{ 8 * loop.index0 }}, aie::load_v<8>(s{{ item[0][0] }} + {{ item[0][1] // 8 * 64 }}));
{% else %}
            {
                aie::vector<bfloat16, 32> v = {{ window(item[0][0], item[0][1]) }};
{% for s_, w, lanes in item[1:] %}
                v = aie::select(v, {{ window(s_, w) }}, aie::mask<32>::from_uint32({{ "0x%Xu" % lanes }}));
{% endfor %}
                aie::store_v(dst + {{ 8 * loop.index0 }}, v.extract<8>(0));
            }
{% endif %}
{% endfor %}
            dst += {{ n_c }};
        }
}

{% endif %}
{% endif %}
{% if silu and act == "silu" %}
// gate/up epilogue: C 8x8 sub-block pairs (2 t, 2 t + 1) of a tile are 8 gate and 8 up rows (acc: [gate
// 64 | up 64] per 128 values); h = silu(gate) * up -> hout (m x {{ n // 2 }}, 8x8 sub-blocks; as bfp16ebs8 panels (8 tokens x 8 outputs)).
// Three pipelined loops through the scratch (GemmQ: the second C tile, idle here), 1627 instead of 3008
// cycles per pass at m = n = 64, no spills: kmath.silu_tanh's two (the gate halves into the scratch), then times up
// (II 4 per 64 values: ``times_up``).
{% set parts = 2 if csplit else sxp %}
{% set mp = m // parts %}
{% set silu_t = km.silu_tanh(walk=(2, 128)) %}
{{ silu_t.src }}

void {{ p }}_silu(const bfloat16 *__restrict cacc_, bfloat16 *__restrict hout_, bfloat16 *__restrict scratch,
                  int32_t zero{{ ", int32_t c_part" if c_parts == 2 else "" }})
{
{% if "act" in ablate %}
    return;
{% endif %}
{% if "ev_act" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if c_parts == 2 %}
    hout_ += c_part * {{ m * n // 2 * 9 // 16 }};  // (the C object's part: a pass of two, or a pass's half)
{% endif %}
{% if parts > 1 %}
    // ({{ parts }} parts of {{ mp }} rows: the scratch holds one part's{{ "; the C tile's halves" if csplit else "" }})
    AIE_LOOP_NO_UNROLL
    for (int part = 0; part < {{ parts }}; part++) {
    const bfloat16 *__restrict cacc = cacc_ + part * {{ csplit // 2 if csplit else mp * n }};
    bfloat16 *__restrict hout = hout_ + part * {{ mp * n // 2 * 9 // 16 }};
{% else %}
    const bfloat16 *__restrict cacc = cacc_;
    bfloat16 *__restrict hout = hout_;
{% endif %}
    {{ silu_t }}<{{ mp * n // 2 }}>(cacc, scratch, zero);
{{ times_up(mp) }}
{% if parts > 1 %}
    }
{% endif %}
{% if "ev_act" in ablate %}
    event1();
{% endif %}
}

{% endif %}
{% if silu and act == "gelu" %}
// gate/up epilogue (GELU, Gemma): the C sub-block pairs as for SiLU; h = gelu(bf16(gate)) * up -> hout (bfp16ebs8
// panels). Three pipelined loops through the scratch: kmath.act's two (``tab_ab``, ``tab_cd``: the core's copies of the
// table, in two 16 KB regions), then times up.
{% set parts = 2 if csplit else sxp %}
{% set mp = m // parts %}
void {{ p }}_gelu(const bfloat16 *__restrict cacc_, bfloat16 *__restrict hout_, bfloat16 *__restrict scratch,
                  const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd{{ ", int32_t c_part" if c_parts == 2 else "" }})
{
{% if "act" in ablate %}
    return;
{% endif %}
{% if "ev_act" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if c_parts == 2 %}
    hout_ += c_part * {{ m * n // 2 * 9 // 16 }};  // (the C object's part: a pass of two, or a pass's half)
{% endif %}
{% if parts > 1 %}
    AIE_LOOP_NO_UNROLL
    for (int part = 0; part < {{ parts }}; part++) {
    const bfloat16 *__restrict cacc = cacc_ + part * {{ csplit // 2 if csplit else mp * n }};
    bfloat16 *__restrict hout = hout_ + part * {{ mp * n // 2 * 9 // 16 }};
{% else %}
    const bfloat16 *__restrict cacc = cacc_;
    bfloat16 *__restrict hout = hout_;
{% endif %}
    {{ gact }}<{{ mp * n // 2 }}>(cacc, scratch, tab_ab, tab_cd);
{{ times_up(mp) }}
{% if parts > 1 %}
    }
{% endif %}
{% if "ev_act" in ablate %}
    event1();
{% endif %}
}

{% endif %}
{% if pair %}
// C (m x {{ n }}, 8x8 row-major sub-blocks) += A (m x 64 bfp16ebs8 panels in MAC pop order) @ the pass's B in two
// halves of {{ n // 2 }} columns (GemmQ pair): the core's own (columns {{ n // 2 }} h .., in its memory), then its partner's (in the
// partner's memory): the pair never streams B from one memory at the same time (one loop, gemm/mac.py split)
// (k-tile kt: C in pC, out pCb when kt is even, else from pCb into pC: two C tiles in different 16 KB regions)
void {{ p }}_mm(const bfloat16 *__restrict pA_in, const void *__restrict own, const void *__restrict other,
                bfloat16 *__restrict pC, bfloat16 *__restrict pCb, int32_t h, int32_t kt)
{
{% if "ev_mm" in ablate %}
    event0();
{% endif %}
{% if "mm" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const bool odd = kt & 1;
    {{ mac }}((const bfp16ebs8 *)pA_in, own, other, (const bfloat16 __aie_dm_resource_c *)(odd ? pCb : pC),
                (bfloat16 __aie_dm_resource_c *)(odd ? pC : pCb), h);
{% if "ev_mm" in ablate or "ev_kt" in ablate %}
    event1();
{% endif %}
}
{% else %}
// C (m x n, 8x8 row-major sub-blocks) += A (m x 64 bfp16ebs8 panels in MAC pop order) @ stage
void {{ p }}_mm(const bfloat16 *__restrict pA_in, const void *__restrict stage, bfloat16 *__restrict pC)
{
{% if "ev_mm" in ablate %}
    event0();
{% endif %}
{% if "mm" in ablate %}
    return;
{% endif %}
    // (every entry sets the rounding: BFP16 tiles have no dq kernel that would, and
    // the default truncates the C tiles and B conversions: 5.7e-2 instead of 1.2e-2 rel-rms)
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if csplit %}
{% set nb, half, band = n // 16, m // 32, 16 * 72 %}
    // C's halves {{ csplit }} bytes apart (two 16 KB regions): the blocks of bands b and b + {{ half }} alternate, so a block's
    // C stores (the loop's second stage) meet the next block's C loads in the other region
    dims_3d_t patA(1, {{ (half - 1) * band }}, {{ nb - 1 }}, {{ -(half + 1) * band }}, {{ -half * band }});
    dims_3d_t patB(1, {{ -band }}, {{ nb - 1 }}, 0, {{ -nb * band }});
    dims_3d_t patL(1, {{ csplit // 2 }}, {{ nb - 1 }}, {{ 128 - csplit // 2 }}, {{ 16 * n - (nb - 1) * 128 - csplit // 2 }});
    dims_3d_t patS = patL;
    {{ macs }}((const bfp16ebs8 *)pA_in, stage, (const bfloat16 __aie_dm_resource_c *)pC,
                (bfloat16 __aie_dm_resource_c *)pC, patA, patB, patL, patS);
{% else %}
    {{ mac }}((const bfp16ebs8 *)pA_in, stage, (const bfloat16 __aie_dm_resource_c *)pC,
                (bfloat16 __aie_dm_resource_c *)pC);
{% endif %}
{% if "ev_mm" in ablate or "ev_kt" in ablate %}
    event1();
{% endif %}
}
{% endif %}

{% if c_pairs %}
void {{ p }}_zerof(float *__restrict acc)
{
    al::zero<{{ m * n }}, 16>(acc);
}

// acc (m x n fp32, 8x8 row-major sub-blocks) += A @ stage
void {{ p }}_mmf(const bfloat16 *__restrict pA_in, const void *__restrict stage, float *__restrict acc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ macf }}((const bfp16ebs8 *)pA_in, stage, (const float __aie_dm_resource_c *)acc,
                 (float __aie_dm_resource_c *)acc);
}

// c (m x {{ n // 2 }} fp32, row-major) = acc's columns 2 i + 2 i + 1 (an MoE router's [hi, lo] rows of an expert)
void {{ p }}_pairs(const float *__restrict acc, float *__restrict c)
{
    for (int rb = 0; rb < {{ m // 8 }}; rb++)
        for (int cb = 0; cb < {{ n // 8 }}; cb++) {
            const float *__restrict s = acc + rb * {{ 8 * n }} + cb * 64;
            float *__restrict d = c + rb * {{ 4 * n }} + cb * 4;
            for (int r = 0; r < 8; r++)
                for (int j = 0; j < 4; j++)
                    d[{{ n // 2 }} * r + j] = s[8 * r + 2 * j] + s[8 * r + 2 * j + 1];
        }
}

{% endif %}
{% if pingpong %}
// C out (m x n) = C in + A @ stage: the k-tiles alternate between two C tiles in different 16 KB regions,
// so the MAC loop's C loads and stores never meet in one region's banks
void {{ p }}_mm2(const bfloat16 *__restrict pA_in, const void *__restrict stage, const bfloat16 *__restrict pCin,
                 bfloat16 *__restrict pCout)
{
{% if "ev_mm" in ablate %}
    event0();
{% endif %}
{% if "mm" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ mac }}((const bfp16ebs8 *)pA_in, stage, (const bfloat16 __aie_dm_resource_c *)pCin,
                (bfloat16 __aie_dm_resource_c *)pCout);
{% if "ev_mm" in ablate or "ev_kt" in ablate %}
    event1();
{% endif %}
}

{% endif %}
}  // extern "C"
""")


@dataclass(frozen=True)
class GemmQDataflow:
    """GemmQ's data movement, shared by the design and ``tests/gemm/test_dma.py``
    (sizes in elements of the FIFO object types: weights i32 words, A and C bf16 units).

    B (weights): shim -> memtile objects of ``b_obj`` words (one super-block slice of the
    pass's tiles, written with ``b_wdims``) -> L1 objects of ``b_l1`` words (one 64-k tile,
    sent with ``b_rdims``, which may zero-pad it) to the column's cores (``b_parts`` = 2: the memtile object in two
    segments, each written with ``b_wdims``, segment h read with ``b_rdims`` to the cores of half h). A: shim ->
    memtile objects of ``a_obj``
    (bfp16 panels: the row's strip if ``a_res``, sent ``a_repeat`` times, else K chunks; written with
    ``a_wdims`` in MAC pop order) -> L1 objects of ``a_l1`` (one k-tile). C: each core's objects of ``c_l1`` joined in the
    memtile, sent to the shim with ``c_dims``."""

    b_obj: int
    b_l1: int
    b_wdims: TensorAccessPattern | None
    b_rdims: TensorAccessPattern | None
    b_parts: int  # (pair: the memtile object in 2 segments, each written with b_wdims and read with b_rdims)
    a_res: bool
    a_obj: int
    a_l1: int
    a_wdims: TensorAccessPattern | None
    a_repeat: int
    c_l1: int
    c_dims: TensorAccessPattern | None


class GemmQOperator(ArtcOperator):
    """C (M, N) = A (M, K) @ W^T for W in decode-overlay GGUF tiles: GemmQ reads the tiles of the tile-major
    permutation (``tile_major``: 16-row Q4_0 / Q4_1 tiles of 2048 or 1024 columns, Q4_K, BFP16); the planar
    path (``planar``: every other dequantized tile) is GemmQdOperator's.

    W is passed as the byte region of the decode weight buffer that holds the
    matrix (preceded by ``hdr_skip`` bytes: 1 KB for Q4_K tiles of 1024 columns): column c's
    tiles start at ``c * col_bytes``. ``out_cols[c][p]`` is the first output
    column of pass p of column c (4 row tiles = 4 R output columns, or 2 R
    with ``silu``: tiles of 8 gate + 8 up rows, output silu(gate) * up).
    ``kc``: the K columns of a tile (default 2048; the images' 1024, Gemma's 512).
    A row tile's K chunks are consecutive tiles in groups of ``kc_group`` (default: all of them), group G of row
    tile j at ``(G n_rt + j) kc_group`` (n_rt: row tiles of the column, ``region_rt``; ``rt_stride`` instead of
    kc_group: the first kc_group chunks of groups of rt_stride, a K part of the image's groups; the passes
    may read past them: a block padded to whole passes reads the next row tiles, of the next block or
    region, whose outputs land in the column's padding). ``blocks``: (first row tile, passes) of
    runs of passes (default one run from row tile 0): e.g. a QKV projection's Q, K and V row tiles,
    each padded to whole passes.
    """

    DQ_CORE = False  # (GemmQdOperator: one dequantizing core per column)
    p = kernel_prefix("gq")  # (the kernels' names: ``kernel_source``)

    def __init__(
        self,
        *,
        fmt: str = "Q4_0",
        rows: int = 16,
        M: int,
        K: int,
        N: int,
        tile_bytes: int,
        col_bytes: int,
        passes: int,
        out_cols: list[list[int]],
        silu: bool = False,
        act: str = "silu",
        nrows: int = ROWS,
        c_bfp16: bool = False,
        ablate: tuple[str, ...] = (),
        kc: int = KC,
        kc_group: int = 0,
        rt_stride: int = 0,
        tiles: int = 0,
        c_pack: bool = False,
        region_rt: int = 0,
        m: int = 0,
        blocks: tuple[tuple[int, int], ...] = (),
        a_cols: tuple[int, int] = (0, 0),
        k_rep: int = 1,
        c_pairs: bool = False,
        groups: int = 1,
        slots: int = 0,
        mc_param: str = "",
        mc_unit: int = 0,
        lines: bool = False,
    ):
        """``tiles``: decode tiles per pass (default 64 rows' worth; planar tiles: any count, MMUL-padded
        to a multiple of 16 rows; with ``silu`` whole SiLU groups, ``formats.gu_tiles``, or half groups of
        tiles whose rows divide 64);
        ``c_pack``: C objects of the pass's rows only (row-major, packed from the MAC tile by an epilogue:
        the MMUL padding rows never leave the core; an odd row count gets one zero column more, whole
        32-bit words);
        ``lines``: C rows of whole 64-byte lines where they fit (a row that starts or ends inside a line makes the
        memory read and merge that line: on a machine without a cache in front of DRAM, Krackan Point, 15.6 instead of
        73.2 GB/s, ``artlite-probe``). Packed passes (``line_carry``): each pass writes ``win`` columns from the line
        boundary at or before its first column, the previous pass's last columns (kept in the C object, whose rows the
        memtile reads ``win`` columns of, into the next row) first, stale columns after its own (the next pass's, or
        padding: the caller leaves room for the last window, ``line_window``). Gate/up (``c_parts``): C objects of two
        32-output parts, drained as 576-byte rows;
        ``a_cols`` = (k0, width): A is the columns k0 .. k0 + K of panels ``width`` columns wide (a projection split
        over K: each part a GEMM of its chunks; default (0, K)). ``nrows``: even for dequantized tiles (pairs).
        ``k_rep``: A has k_rep K columns, the weights' K chunks stream k_rep times (BFP16 tiles: an MoE router's rows
        against x as [hi | lo] panels: K' = 2 K). ``c_pairs``: C in fp32, the sums of each pair of adjacent output
        columns (an MoE router's [hi, lo] rows of an expert: its logit), as row-major fp32 objects of n / 2 values (the
        C argument in bf16 units: two per value; ``out_cols`` count them). ``groups``: row-group tiles
        (``decode.image.Block.G``: a tile row holds G rows of kc / G columns, tile i's row r slot j = output row
        G R i + R j + r; A has K / G columns, the K chunks' parts of a slot one after the other): per pass G slots, each a
        GEMM of the A strip with its super-blocks (the strip resent per slot), C objects of ``slots`` slots
        (default: one, two for tiles of an odd row count; GemmQd only). ``mc_param``: the cores run as many token
        chunks as this scratchpad parameter says, in units of ``mc_unit`` tokens (a multiple of nrows m; default
        nrows m): the designs of any M are one (the runtime sequence: M / (nrows m) chunks).

        Experiment switches (timing only, wrong results): ``ablate`` "dq" / "mm" compile the
        kernel bodies out ("hdr": GemmQd's Q4_K header expansion; "act": the gate/up epilogue), "ev_dq" / "ev_mm" /
        "ev_kt" / "ev_act" put event0/1 markers around dq, mm, dq..mm of a k-tile or the gate/up epilogue, "noperm"
        drops the memtile transposes, "nosplit" keeps GemmQd's C tiles whole (``csplit``)."""
        self.planar = planar(fmt, rows, kc)
        assert self.planar or tile_major(fmt, rows, kc), f"no GEMM path reads {fmt} tiles of {rows} rows x {kc}"
        assert not self.planar or self.DQ_CORE, f"{fmt} tiles of {rows} rows x {kc}: GemmQdOperator"
        self.ablate = tuple(ablate)
        self.mc_param = mc_param
        self.mc_unit = mc_unit
        self.nrows = nrows  # core rows used (fewer rows: a smaller program image, cheaper reconfiguration)
        assert M % (nrows * 16) == 0 and K % kc == 0 and kc % 256 == 0 and tile_bytes % 4 == 0
        assert kc in (KC, 1024, 512)
        self.kc = kc
        self.nsb = kc // 256  # super-block slices per K chunk
        self.fmt, self.R = fmt, rows
        # planar tiles of super-block units (``Slot.unit_bytes``): the weight objects are units of the pass's tiles
        self.units = slot(fmt).unit_bytes(fmt, rows, self.nsb) if self.planar else 0
        self.M, self.K, self.N = M, K, N
        self.k_rep, self.c_pairs, self.G = k_rep, c_pairs, groups
        assert groups == 1 or (self.DQ_CORE and k_rep == 1 and not silu and kc // 256 % groups == 0), groups
        self.KA = k_rep * K // groups  # (A's columns)
        self.a_k0, self.a_K = a_cols[0], a_cols[1] or self.KA
        assert self.a_k0 % kc == 0 and self.a_k0 + self.KA <= self.a_K
        self.tiles = tiles or 64 // rows  # decode tiles per pass (64 rows)
        self.TR = self.tiles * rows  # weight rows per pass
        self.silu = silu  # tiles of 8 gate + 8 up rows -> C = act(gate) * up (N counts outputs)
        assert act in ("silu", "gelu") and (silu or act == "silu"), act
        self.act = act  # the gate/up epilogue's activation: silu, or gelu (tanh form, kmath.GELU: Gemma)
        assert act == "silu" or self.DQ_CORE, "the GELU epilogue: GemmQd"
        # rows the memtile reads per pass: a planar SiLU pass of a whole group (128 rows and the padding rows
        # of the group's last tile) drops the padding rows when every part of the tile, header included, maps
        # tile row r to gathered row r (``hdr_direct``); else the dequant skips them
        # whole groups run in two halves of 64 rows (4 [8 gate | 8 up] blocks each): the dequant writes a
        # k-tile's rows as two D objects, a MAC core pops one A for both into two C tiles of 64 columns
        # (L1 for 96 tokens per MAC row; a whole group's C tile does not fit) and drains them as two passes
        self.halves = 1
        if silu and self.planar:
            gt = -(-128 // rows)  # tiles of a SiLU group (formats.gu_tiles)
            if self.tiles == gt:
                self.halves = 2
            else:  # (half groups of whole tiles)
                assert 64 % rows == 0 and self.TR % 64 == 0, f"planar SiLU passes: a group of {gt} tiles"
        self.n_read = 128 if self.halves == 2 and self.hdr_direct else self.TR
        self.n = -(-self.n_read // 16) * 16  # MMUL columns (planar: the rows padded to a multiple of 16)
        assert self.n == self.TR or self.planar, (fmt, self.tiles)
        self.nm = 64 if self.halves == 2 else self.n  # MMUL columns of a D object and a MAC core's C tile
        self.tile_bytes, self.col_bytes = tile_bytes, col_bytes
        self.passes, self.out_cols = passes, out_cols
        self.blocks = tuple(blocks) or ((0, passes),)
        assert sum(n for _, n in self.blocks) == passes
        self.pass_rt = [rt0 + self.tiles * i for rt0, n in self.blocks for i in range(n)]  # first row tile of each pass
        rt_end = max(rt0 + self.tiles * n for rt0, n in self.blocks)  # row tiles read
        self.region_rt = region_rt or rt_end  # row tiles of a column's region (the chunk group stride)
        assert self.region_rt <= rt_end or (kc_group or K // kc) == 1  # (one chunk per group: any chunk stride)
        self.n_kc = K // kc
        # a row tile's K chunks come in groups of kc_group consecutive tiles (row tiles of a group
        # consecutive): the image's groups of a long-K projection (``decode.image.kc_group``: the decode overlay's
        # activation objects)
        self.kc_group = kc_group or self.n_kc
        assert self.n_kc % self.kc_group == 0
        # tiles from a row tile's chunk group to the next row tile's (default kc_group; more: the GEMM reads the first
        # kc_group chunks of the image's groups of rt_stride, a K part)
        self.rt_stride = rt_stride or self.kc_group
        assert self.rt_stride >= self.kc_group
        assert len(out_cols) == COLS and all(len(o) == passes for o in out_cols)

        # bfp16ebs8 activations: A / C as panels of 8 rows (tokens) x 8 k, [M / 8][K / 8][72 B]
        # (half the bytes of bf16 through the core's A input port, the limit at m = 32)
        self.c_bfp16 = c_bfp16
        assert c_bfp16 == silu, "the gate/up epilogue: bfp16 C panels"
        assert not (c_pairs or k_rep > 1) or (fmt == "BFP16" and not silu and not c_pack), "router pairs: BFP16 tiles"
        # (whole 16-row blocks: the memtile un-panelizes the MAC tile; two slots per C object: packed)
        self.c_pack = c_pack and (self.TR != self.n or slots == 2)
        assert not (c_pack and silu), "packed C: not with the gate/up epilogue"
        # C object columns (whole 32-bit words: an odd row count packs one zero column more)
        # row groups: a C object holds ``slots`` slots of the pass's rows (an even count of outputs per tile: Q6_K's 11
        # rows in pairs), as [tile][slot][row] (c_pack) or the MMUL tile (one slot of 16-row tiles)
        self.slots = slots or (1 if groups == 1 or rows % 2 == 0 else 2)
        self.vp = groups // self.slots  # (C objects per pass)
        assert groups % self.slots == 0 and (groups == 1 or self.slots == 1 or self.c_pack), (groups, rows)
        self.n_out = self.slots * self.TR if self.c_pack and groups > 1 else \
            self.TR + self.TR % 2 if self.c_pack else self.nm // 2 if silu else self.n
        assert self.n_out % 2 == 0, "C rows of whole 32-bit words"
        # packed C object rows: whole 16-byte items, the memtile drops the columns past n_out
        self.n_c = -(-self.n_out // 8) * 8 if self.c_pack else self.n_out
        # whole-line drains (where they fit: below): C object rows of a pass's carry and its own columns, drained in
        # windows of ``win`` columns
        self.carry = line_carry(out_cols, self.n_out) \
            if lines and self.c_pack and groups == 1 and self.slots == 1 and N % LINE == 0 else None
        n_c, self.win = self.n_c, self.n_out
        if self.carry is not None:
            self.n_c = -(-(max(cr for cr, _ in self.carry) + self.n_out) // 8) * 8
            self.win = line_window(self.carry, self.n_out)
        # gate/up (bfp16 C, 32 outputs a pass or half: 288-byte panel rows, half of them ending inside a line): C objects
        # of two such parts, [part][group][4 panels], which the memtile drains as 576-byte rows (``lines``)
        self.c_parts = 2 if lines and self.c_bfp16 and self.n_out == 32 and N % 64 == 0 and two_parts(out_cols,
                                                                                                    self.halves) else 1
        self.slice = self.units or FORMATS[fmt].slice_bytes(rows)
        # header units of a slice each (d; d, m for Q4_1; Q4_K records). A header shorter than whole
        # slices (Q4_K of 1024 columns: 1 KB) is read as a slice-sized unit that starts hdr_skip bytes
        # before the tile (its first part belongs to the previous tile): every unit the memtile permutes
        # is one slice of the pass's tiles, at the cost of hdr_skip extra bytes per tile
        hb = 0 if self.units else FORMATS[fmt].header_bytes(rows, self.nsb)
        self.hdr_units = -(-hb // self.slice)
        self.hdr_skip = self.hdr_units * self.slice - hb
        assert self.hdr_skip == 0 or (fmt in ("Q4_K", "Q4_0") and self.nsb == 4) or self.planar, (fmt, self.nsb)
        # Gemma's 512-column Q4_0 tiles of 32 rows (a 1 KB header, a quarter of a slice): a header unit holds the
        # headers of up to ``hg`` = 4 consecutive K chunks of a row tile ([chunk][1 KB], ``hdr_groups``), so the fills
        # read the tiles' bytes once (a slice-sized unit per chunk from hdr_skip before the tile: 1.33x; gate / up at
        # T = 288 has a 0.81 ms DMA floor)
        self.hg = 4 if self.planar and fmt == "Q4_0" and self.nsb == 2 and rows == 32 else 1
        if self.hg > 1:
            assert hb * self.hg == self.slice and self.hdr_units == 1
            self.hdr_skip = 0
        # a unit longer than a shim BD's inner dimension (1023 words: Gemma's 32-row Q4_0 slices, 1024) is read as
        # ``us`` sub-units of 16 rows (their rows contiguous): the memtile gathers a pass's unit as [sub-unit][tile]
        # [row 16], the dequant reads its rows in that order and writes the MMUL columns in the tiles' order
        self.us = 1
        if self.planar and self.slice // 4 > 1023:
            f = FORMATS[fmt]
            assert not (f.plane_major or f.word_major) and rows % 16 == 0, (fmt, rows)
            self.us = rows // 16
        self.dequant = FORMATS[fmt].dequant  # (BFP16 tiles already hold MMUL operands)
        # GemmQ's dequant: each pair of core rows splits every k-tile (GemmQd: one dequantizing core)
        self.pair = self.dequant and not self.DQ_CORE
        assert not self.pair or (self.tiles == TILES and nrows % 2 == 0), "GemmQ dequant: 4 tiles per pass, row pairs"
        # tiles of a core's k-tile objects (its dequant; GemmQd: the pass's)
        self.tq = TILES // 2 if self.pair else self.tiles if self.DQ_CORE and not self.planar else TILES
        # the W argument: hdr_skip bytes before the matrix, then its columns' regions (and the tiles the
        # last column's extra row tiles read after its region: those of the next region in the image)
        nk, g, rt, rs = self.n_kc, self.kc_group, self.region_rt, self.rt_stride
        last = ((nk - 1) // g) * rt * rs + (rt_end - 1) * rs + (nk - 1) % g + 1 if nk > 1 or rs > 1 else rt_end
        self.overrun = max(0, last - (rt * rs * (nk // g) if nk > 1 or rs > 1 else rt)) * tile_bytes
        self.w_bytes = COLS * col_bytes + self.hdr_skip + self.overrun
        self.ws_planes = FORMATS[fmt].ws_planes  # expanded scale planes
        assert self.slice in (2048, 2304) or self.planar, "super-block slices: 16 x 128 B (Q4), 8 x 288 B (BFP16)"
        # tokens per core row and chunk (the weights stream once per chunk): the largest that
        # divides the prompt and fits L1
        # (m: GemmQd's tokens per MAC row, chosen by the caller)
        def fit() -> list[int]:
            return [m_ for m_ in ((m,) if m else (64, 48, 32, 16))
                    if (m or m_ <= M_MAX) and M % (nrows * m_) == 0 and self._l1_bytes(m_) <= L1_BUDGET]

        ms = fit()
        if self.carry is not None or self.c_parts == 2:  # (whole-line drains where they keep the tokens per core row,
            # the pinned L1 layout, the A strip resident in the memtile and the drains' queue)
            lines_on = self.carry, self.n_c, self.win, self.c_parts
            lines_off = None, n_c, self.n_out, 1
            self.carry, self.n_c, self.win, self.c_parts = lines_off
            plain = fit()
            if ms and ms[:1] == plain[:1] and (lines_on[0] is None or self._drains_fit(lines_on[0])):
                self.m = plain[0]
                pinned, a_res = self._l1_layout() is not None, self.dataflow().a_res
                self.carry, self.n_c, self.win, self.c_parts = lines_on
                self._l1_bytes(self.m)
                if (pinned and self._l1_layout() is None) or (a_res and not self.dataflow().a_res):
                    self.carry, self.n_c, self.win, self.c_parts = lines_off
                    ms = plain
            else:
                ms = plain
        if not ms:
            raise L1Error(f"no token count per core row fits L1 ({m or 'any'})")
        self.m = ms[0]
        self._l1_bytes(self.m)  # (the chosen variant's buffers)
        self.mc = M // (nrows * self.m)
        super().__init__()

    @property
    def c_tail(self) -> int:
        """Elements after a packed C object's rows (whole-line windows: the memtile reads the last row's window into
        them, ``lines``)."""
        return self.win - self.n_c if self.carry is not None else 0

    def _drains_fit(self, carry: tuple[tuple[int, int], ...]) -> bool:
        """Whether whole-line windows (``carry``) keep a column's C drains within its shim channel's task queue (4: two
        token chunks in flight, up to 2 drains each, or no more drains than without windows) and its tile's 16 BDs (the
        weight fills', an A strip's, the drains' per chunk in flight; ``_sequence``)."""
        cols = self.c_out_cols(0)
        runs = len(_runs([o - cr for o, (cr, _) in zip(cols, carry, strict=True)]))
        if runs <= len(_runs(cols)):
            return True
        if runs > 2:
            return False
        chains = [self.b_chain(c) for c in range(COLS)]
        depth = 1 if chains[0] is not None and sum(map(len, chains[0])) > 4 else 2
        fills = max(sum(map(len, ch)) if ch is not None else len(self.b_taps(c)) for c, ch in enumerate(chains))
        return depth * (fills + 1 + runs) <= 16

    def _chunk_passes(self, prm: object, index: bool = False, step: int = 1) -> Iterator[object]:
        """The cores' loop over token chunks and passes: mc chunks, or ``mc_param``'s (prm: its handle); ``index``:
        yields the pass's index (else None); ``step``: one iteration per that many passes."""
        passes = self.passes // step
        if prm is None:
            if index:
                for _ in range_(self.mc):
                    for p in range_(passes):
                        yield p
                return
            for _ in range_(self.mc * passes):
                yield None
            return
        n, scale = prm.read(), (self.mc_unit or self.nrows * self.m) // (self.nrows * self.m)
        assert scale * self.nrows * self.m == (self.mc_unit or self.nrows * self.m), (self.mc_unit, self.nrows, self.m)
        for _ in range32(times_i32(n, scale) if scale > 1 else n):
            for p in range_(passes):
                yield p if index else None

    def _mc_prm(self) -> object:
        """``mc_param``'s handle for the workers (None without it)."""
        return ScratchpadParameter(self.mc_param, np.int32) if self.mc_param else None

    @property
    def hdr_direct(self) -> bool:
        """Planar tiles whose gathered rows hold tile row r's header at row r too (Q5_K's scale planes,
        Q4_0 / Q4_1 headers of 2048 columns; not Q6_K's d chunks or a half-slice Q4 header row pair); tiles of units
        (the dequant takes the rows it converts from the unit)."""
        return bool(self.units) or slot(self.fmt).hdr_direct(self.fmt, self.nsb)

    @property
    def unit_pitch(self) -> int:
        """Bytes of a tile's unit in the dequantizing core's weight object (whole 32-byte lines: aligned loads)."""
        return -(-self.units // 32) * 32

    @property
    def b_l1_words(self) -> int:
        """Words of an L1 weight object (one k-tile; planar: [plane][word 2][rows padded to n]; units: a unit of the
        pass's tiles, [tile][``unit_pitch``])."""
        if self.units:
            return self.tiles * self.unit_pitch // 4
        return 2 * FORMATS[self.fmt].planes * self.n if self.planar else \
            (self.tq if self.pair else self.tiles) * self.slice // 16

    def _buffers(self, m: int) -> dict[str, tuple[int, int]]:
        """(count, bytes) of each core buffer with m tokens per core row: the FIFO objects A
        (one k-tile), B (one k-tile of the pass's tiles) and C, the MAC target ``acc`` (silu
        only), the scale workspace ``ws``, the core's half of a dequantized k-tile ``X`` (its FIFO to the partner) and
        ``st``, the second C tile (also the collected header, ``Slot.gemm_q_collects``, and the SiLU scratch); BFP16:
        A, B, C, ``acc`` and ``st``, the SiLU scratch. get_mlir_artifact asserts its buffer types against
        this."""
        if self.pair:
            return {"A": (2, m * 72), "B": (B_DEPTH, 4 * self.b_l1_words),
                    "C": (2, m * self.n_out * 9 // 8 * self.c_parts if self.c_bfp16 else m * self.n_c * 2),
                    "acc": (1 if self.silu else 0, m * self.n * 2),
                    "ws": (1, 1024 * self.nsb * self.ws_planes * self.tq // TILES),
                    "st": (1, max(m * self.n * 2, 16 * self.b_l1_words if slot(self.fmt).gemm_q_collects else 0)),
                    "X": (2, self.n // 2 * KT * 9 // 8)}
        return {"A": (2, m * 72), "B": (B_DEPTH, 4 * self.b_l1_words),
                "C": (2, m * self.n_out * 9 // 8 * self.c_parts if self.c_bfp16 else m * self.n_c * 2),
                "acc": (1 if self.silu or self.c_pairs else 0, m * self.n * (4 if self.c_pairs else 2)),
                "st": (1, max(64, m * self.n if self.silu else 0))}

    def _l1_bytes(self, m: int) -> int:
        """Core data memory of the design with m tokens per core row (buffers and the 2 KB stack)."""
        return sum(n * b for n, b in self._buffers(m).values()) + 2048

    def _l1_layout(self) -> dict[str, list[int]] | None:
        """Byte addresses of a core's buffers, or None (automatic placement) if they do not fit.

        Core data memory behaves like four 16 KB regions of two interleaved banks. The MAC
        loop streams A (the A objects), its B operands (the dequantized halves ``X``, or the B objects of BFP16
        tiles) and the C tile(s) every cycle: each gets its own region. The stack is at 0.
        """
        sizes = self._buffers(self.m)
        target = "acc" if self.silu or self.c_pairs else "C"  # the MAC's C tile
        # (buffer, allowed regions in order of preference)
        if self.pair:  # the MAC streams X (or the partner's), A and the two C tiles: one region each; the dequant
            # reads B and ws and writes X: three regions; the C objects of the epilogue (cold) next to X
            order = [("X", [0]), ("A", [1]), (target, [2]), ("st", [3]), ("ws", [2, 3, 0])]
            order += [("C", [0, 3, 1])] if self.silu else []
            order += [("B", [3, 1, 0, 2])]
        else:
            order = [("st", [0]), ("A", [1]), (target, [2])] + ([("C", [0, 3, 1])] if self.silu or self.c_pairs else [])
            order += [("B", [3])]
        free = [16384 - 2048] + [16384] * 3
        nxt = [2048, 16384, 32768, 49152]
        layout: dict[str, list[int]] = {}
        for name, regions in order:
            n, sz = sizes[name]
            need = n * ((sz + 63) // 64 * 64)
            g = next((g for g in regions if free[g] >= need), None)
            if g is None:
                return None
            layout[name] = [nxt[g] + i * ((sz + 63) // 64 * 64) for i in range(n)]
            free[g] -= need
            nxt[g] += need
        return layout

    def get_arg_spec(self) -> list[ArgSpec]:
        return [
            ArgSpec("in", (self.M * self.a_K * 9 // 16,), torch.bfloat16),
            ArgSpec("in", (self.w_bytes // 2,), torch.bfloat16),
            ArgSpec("out", (self.M * self.N * 9 // 16,) if self.c_bfp16 else (self.M, self.N), torch.bfloat16),
        ]

    def get_kernel_src(self) -> str:
        # the MAC: B as the dequantized bfp16 panels, or BFP16 tiles
        mac = render_mac(k=KT, split=self.pair)
        if self.c_pairs:
            mac += render_mac(name=MACF, k=KT, f32=True)
        return _KERNEL_TPL.render(p=self.p, mac=f"{MAC}<{self.m}, {self.n}>", m=self.m, n=self.n, nd=self.n, halves=1, k=KT, silu=self.silu, act=self.act,
                                  fmt=self.fmt,
                                  nsb=self.nsb, dqb=False,
                                  c_bfp16=self.c_bfp16, ablate=self.ablate, mac_src=mac,
                                  c_pack=False, carry=None, win=0, c_parts=self.c_parts, pingpong=False, planar=False,
                                  sxp=1, csplit=0, tq=self.tq, pair=self.pair,
                                  c_pairs=self.c_pairs, macf=f"{MACF}<{self.m}, {self.n}>", slots=1)

    def dataflow(self) -> GemmQDataflow:
        m, K, no = self.m, self.KA, self.n_out
        r = t = 8
        cb = self.c_bfp16
        noperm = "noperm" in self.ablate
        obj_w = self.tiles * self.slice // 4  # words per memtile object (the pass's tiles x one slice)
        rpad = None
        if self.planar:
            # tile slices of 32-byte planes per row (plane major: [plane][row R][8 words], Q5_K / Q6_K; row major:
            # the Q4 nibbles [row R][step s][8 words]): S2MM writes [plane][g = t R + r][8 words]; MM2S reads
            # k-tile objects [plane][word ii 2][g] (words 2 kt + ii: groups 2 kt + ii) of the first n_read rows,
            # zero-padded to n (64-byte aligned runs of 16 rows for the dequant's loads)
            R, T, TR, P = self.R, self.tiles, self.TR, FORMATS[self.fmt].planes
            if self.units:  # (the unit of the pass's tiles as it arrives, each tile's padded to whole lines)
                uw, pw = self.units // 4, self.unit_pitch // 4
                wdims, rdims = None, [(T, uw), (uw, 1)]
                rpad = [(0, 0), (0, pw - uw)] if pw > uw else None
            elif FORMATS[self.fmt].word_major:
                # word-major slices ([word][row R], Q2_K / Q3_K: 2 P words per k-tile): S2MM writes [word][g];
                # MM2S reads k-tile objects [word 2 P][g] (words 2 P kt ..)
                wdims = [(T, R), (8 * P, TR), (R, 1)]
                rdims = [(4, 2 * P * TR), (2 * P, TR), (self.n_read, 1)]
            else:
                if FORMATS[self.fmt].plane_major:
                    wdims = [(T, 8 * R), (P, 8 * TR), (8 * R, 1)]
                elif self.us > 1:  # (sub-units: the stream [sub-unit][tile][row 16][step][8 words], g = its row)
                    wdims = [(TR, 8), (P, 8 * TR), (8, 1)]
                else:
                    wdims = [(T, 8 * R), (R, 8), (P, 8 * TR), (8, 1)]
                rdims = [(4, 2), (P, 8 * TR), (2, 1), (self.n_read, 8)]
            # (a padded MM2S BD moves its padded walk, not the object: reading fewer rows than the memtile
            # holds needs a padding, if only of zeros)
            if not self.units and (self.n > self.n_read or self.n_read < TR):
                rpad = [(0, 0)] * (len(rdims) - 1) + [(0, self.n - self.n_read)]
        elif (dims := slot(self.fmt).tile_major_dims(self)) is not None:  # (BFP16)
            wdims, rdims = dims
        else:
            # Q4: tile rows store words of 8 consecutive k ((s, i) = k in [32 i + 8 s, +8)), a MMUL panel
            # is 8 rows x 8 k: S2MM writes each slice [tile][i][s][row], MM2S reads k-tile objects
            # [kt][tile][i % 2][s][row] (i = 2 kt + i % 2) = packed panels in the dequant's order
            # (pair: the memtile object in two segments of tq tiles, each written and read the same way: a
            # distribute link writes its input per segment; segment h to the cores of half h)
            wdims = [(self.tq, 512), (16, 1), (4, 16), (8, 64)]
            rdims = [(4, 128), (self.tq, 512), (2, 64), (64, 1)]
        # A strip of a core row: resident in the memtile and resent per pass when it fits next to the
        # column's B and C FIFOs, else refetched per pass in K chunks
        c_obj = m * no * 9 // 8 * self.c_parts if cb else (m * self.n_c + self.c_tail) * 2
        others = 2 * obj_w * 4 + 2 * self.nrows * c_obj  # (B, C: depth 2)
        a_res = m * K * 9 // 8 <= min(A_RESIDENT_MAX, MEMTILE_BYTES - others - 4096)
        ak = K if a_res else self.kc
        # bfp16 panels (36 bf16 units) arrive [kt][token group g = 2 band + half][k-step][panel] (shim
        # pattern); the memtile stores them [kt][band][k-step][half][panel] = the MAC pop order
        a_wdims = [(ak // KT * (m // 16), 8 * 72), (2, 36), (8, 72), (36, 1)]
        a_obj = m * ak * 9 // 16
        c_l1 = m * no * 9 // 16 * self.c_parts if cb else m * self.n_c + self.c_tail
        parts = 2 if self.pair else 1
        return GemmQDataflow(
            b_obj=obj_w, b_l1=self.b_l1_words,
            b_wdims=None if noperm or wdims is None else stream_walk(obj_w // parts, wdims),
            b_rdims=None if noperm else stream_walk(obj_w // parts, rdims, rpad), b_parts=parts,
            a_res=a_res, a_obj=a_obj, a_l1=m * 36, a_wdims=stream_walk(a_obj, a_wdims),
            a_repeat=self.passes * self.G if a_res and self.passes * self.G > 1 else 1,
            c_l1=c_l1,
            # the join of the 4 cores' 8x8 sub-blocks is un-panelized to row-major on the way to the shim
            # (packed C objects: row-major already, the columns past n_out dropped: a partial read of each
            # core's segment of the join)
            # (whole-line windows: win columns of each row, the rest from the next row's or the object's tail, ``lines``)
            # (two bfp16 parts: [part][group][panels] to the groups' rows of both parts' panels)
            c_dims=(stream_walk(c_l1, [(m // 8, no // 8 * 36), (2, m // 8 * no // 8 * 36), (no // 8 * 36, 1)])
                    if self.c_parts == 2 else
                    stream_walk(c_l1, [(m, self.n_c), (self.win, 1)], [(0, 0), (0, 0)])
                    if self.n_c != self.win else
                    None if cb or self.c_pack or self.c_pairs else
                    stream_walk(c_l1, [(m // r, r * no), (r, t), (no // t, r * t), (t, 1)])))

    @property
    def hdr_groups(self) -> list[tuple[int, int]]:
        """(first K chunk, chunks) of each header unit: ``hg`` consecutive chunks of one chunk group (``kc_group``) at
        most; with hg = 1 one unit per chunk."""
        g = self.kc_group
        return [(k0, min(self.hg, G + g - k0)) for G in range(0, self.n_kc, g) for k0 in range(G, G + g, self.hg)]

    @property
    def pass_fill_words(self) -> int:
        """bf16 units of a column's weight fills per pass (every K chunk)."""
        nsl, sw = self._units
        hdr = len(self.hdr_groups) * self.slice // 2 if self.hg > 1 else 0
        return self.k_rep * self.tiles * (self.n_kc * nsl * sw + hdr)

    def _hg_bds(self, c: int, j0: int, np_: int) -> list[TensorAccessPattern]:
        """``hg`` > 1: the weight fill of passes j0 .. (row tiles j0 + tiles p) as BDs [np_ (passes), 3 dims]: per
        header group its unit's two 16-row sub-units ([tile][chunks 2 su, 2 su + 1][1 KB]; a group of fewer than 4
        chunks reads its last tile's header and first codes into the unused slots), then per chunk its codes
        ([sub-unit 4][tile][2 KB] after the header)."""
        tw, (nsl, sw) = self.tile_bytes // 2, self._units
        nt, g, rs, n_rt = self.tiles, self.kc_group, self.rt_stride, self.region_rt
        base = c * self.col_bytes // 2
        w_dims = (1, self.w_bytes // 2)
        hw = self.slice // 2 // self.hg  # (a tile's header, bf16 units)

        def at(k: int) -> int:  # tile (j0, k)
            return base + ((k // g) * n_rt * rs + j0 * rs + k % g) * tw

        out = []
        for k0, r in self.hdr_groups:
            for su in range(2):
                qa, qb = min(2 * su, r - 1), min(2 * su + 1, r - 1)
                step = at(k0 + qb) - at(k0 + qa) if qb > qa else hw  # (no zero strides in a shim BD)
                out.append(TensorAccessPattern(w_dims, at(k0 + qa), [np_, nt, 2, hw], [nt * rs * tw, rs * tw, step, 1]))
            out += [TensorAccessPattern(w_dims, at(k) + hw, [np_, nsl, nt, sw], [nt * rs * tw, sw, rs * tw, 1])
                    for k in range(k0, k0 + r)]
        return out

    @property
    def _units(self) -> tuple[int, int]:
        """(units per tile and K chunk, bf16 per unit) of the weight fills: header planes + super-block
        slices (each as ``us`` sub-units)."""
        hu = 0 if self.hg > 1 else self.hdr_units  # (hg > 1: the header units apart, ``_hg_bds``)
        return (hu + self.nsb) * self.us, self.slice // 2 // self.us

    def b_taps(self, c: int) -> list[tuple[int, TensorAccessPattern]]:
        """(first pass, tap) of the weight fills of column c in bf16 units (one fill for all passes of a
        block where one pattern covers them): per pass and K chunk the header units and the super-block
        slices of the pass's tiles."""
        tw, (nsl, sw) = self.tile_bytes // 2, self._units
        nt, nk, g, rs = self.tiles, self.n_kc, self.kc_group, self.rt_stride
        # units start hdr_skip bytes before each tile (a header shorter than a slice: see __init__); the
        # W argument starts hdr_skip bytes before the matrix
        base = c * self.col_bytes // 2
        w_dims = (1, self.w_bytes // 2)
        n_rt = self.region_rt  # row tiles of the column
        out: list[tuple[int, TensorAccessPattern]] = []
        p0 = 0
        for rt0, np_ in self.blocks:
            if self.hg > 1:  # (per pass: the header groups' units and the chunks' codes, ``_hg_bds``)
                for p_ in range(p0, p0 + np_):
                    out += [(p_, t) for t in self._hg_bds(c, self.pass_rt[p_], 1)]
            elif nk == 1:
                out.append((p0, TensorAccessPattern(w_dims, base + rt0 * rs * tw, [np_, nsl, nt, sw],
                                                    [nt * rs * tw, sw, rs * tw, 1])))
            elif g == nk and tw == nsl * sw:  # (chunk, unit) as one dimension: tiles of whole slices, no padding
                out.append((p0, TensorAccessPattern(w_dims, base + rt0 * rs * tw, [np_, nk * nsl, nt, sw],
                                                    [nt * rs * tw, sw, rs * tw, 1])))
            else:  # per pass (more than the 3 dimensions + repeat of a shim BD): tile (j, k) of row tile j and
                # chunk k is at (k // g) n_rt rs + j rs + k % g
                for p_ in range(p0, p0 + np_):
                    j0 = self.pass_rt[p_]
                    if g in (1, nk):  # the chunks at one stride: one fill per pass
                        sk = tw if g == nk else n_rt * rs * tw
                        out.append((p_, TensorAccessPattern(w_dims, base + j0 * rs * tw, [nk, nsl, nt, sw],
                                                            [sk, sw, rs * tw, 1])))
                    else:
                        out += [(p_, TensorAccessPattern(w_dims, base + (G * n_rt + j0) * rs * tw, [g, nsl, nt, sw],
                                                         [tw, sw, rs * tw, 1])) for G in range(nk // g)]
            p0 += np_
        return out

    def b_chain(self, c: int) -> list[list[TensorAccessPattern]] | None:
        """The weight fill of column c as one chained task per block, when b_taps needs fills per pass:
        one BD per K chunk, each 4-dimensional [pass (BD iteration), unit, tile, words] (the task repeats
        once per pass of the block and every BD of the chain advances its iteration then), or None."""
        if self.hg > 1:
            assert max(n for _, n in self.blocks) <= 64, "a BD iterates at most 64 times"
            chains = [self._hg_bds(c, rt0, np_) for rt0, np_ in self.blocks]
            assert sum(map(len, chains)) <= 14, "chained weight fills of more than 14 shim BDs"
            return chains
        taps = self.b_taps(c)
        if (len(taps) == len(self.blocks) and self.k_rep == 1) or max(n for _, n in self.blocks) > 64:
            return None  # (one fill per block already; a BD iterates at most 64 times)
        tw, (nsl, sw) = self.tile_bytes // 2, self._units
        nt, nk, g, rs = self.tiles, self.n_kc, self.kc_group, self.rt_stride
        base = c * self.col_bytes // 2
        n_rt = self.region_rt
        w_dims = (1, self.w_bytes // 2)
        # chunk k: tile (j, k) at (k // g) n_rt rs + j rs + k % g; pass p's tiles j = rt0 + nt p ..
        return [[TensorAccessPattern(w_dims, base + ((k // g) * n_rt * rs + rt0 * rs + k % g) * tw, [np_, nsl, nt, sw],
                                     [nt * rs * tw, sw, rs * tw, 1]) for k in list(range(nk)) * self.k_rep]
                for rt0, np_ in self.blocks]

    def a_tap(self, ch: int, row: int) -> TensorAccessPattern:
        """Core row ``row``'s A strip of token chunk ``ch`` (bf16 units): once (resident, resent by
        the memtile) or once per pass."""
        m, K, aK, df = self.m, self.KA, self.a_K, self.dataflow()
        n = 1 if df.a_res else self.passes
        # panels [group][k8][36]: read [kt][group][8 k-steps x panel]
        return TensorAccessPattern((self.M * aK * 9 // 16,),
                                   (ch * self.nrows + row) * (m // 8) * (aK // 8) * 36 + self.a_k0 // 8 * 36,
                                   [n, K // KT, m // 8, 8 * 36], [0, 8 * 36, aK // 8 * 36, 1])

    def c_chain(self, ch: int, c: int) -> list[TensorAccessPattern]:
        """Row groups: the C drain of column c for token chunk ``ch`` as one chained task, a BD per C object of a pass
        (its slots' rows of the pass's tiles: [token][tile G R apart][slots R]), each iterating over the passes (bf16
        units; the passes' outputs evenly spaced)."""
        m, rows, G, R = self.m, self.nrows, self.G, self.R
        cols = self.out_cols[c]
        step = cols[1] - cols[0] if len(cols) > 1 else G * self.TR
        assert all(b - a == step for a, b in itertools.pairwise(cols)) and self.passes <= 64, cols
        assert (self.slots * R) % 2 == 0 and (G * R) % 2 == 0, "C runs of whole 32-bit words"
        return [TensorAccessPattern((self.M, self.N), ch * rows * m * self.N + cols[0] + self.slots * R * v,
                                    [self.passes, rows * m, self.tiles, self.slots * R], [step, self.N, G * R, 1])
                for v in range(self.vp)]

    def c_out_cols(self, c: int) -> list[int]:
        """The first output column of each C object of column c (``out_cols``; two per pass in halves; of two parts:
        one per pass, or per pair of passes)."""
        if self.c_parts == 2:
            return list(self.out_cols[c]) if self.halves == 2 else list(self.out_cols[c][0::2])
        return [o + self.n_out * h for o in self.out_cols[c] for h in range(self.halves)]

    def c_taps(self, ch: int, c: int) -> list[TensorAccessPattern]:
        """The C drains of column c for token chunk ``ch`` (bf16 units): one per run of evenly
        spaced C objects' outputs."""
        m, no, rows = self.m, self.n_out * self.c_parts, self.nrows
        cols = self.c_out_cols(c)
        if self.carry is not None:  # (whole-line windows from each pass's line boundary)
            cols, no = [o - cr for o, (cr, _) in zip(cols, self.carry, strict=True)], self.win
        taps = []
        for p0, cnt, step in _runs(cols):
            if self.c_bfp16:  # panels [group][N / 8][36]: [pass][the chunk's groups][no / 8 blocks][panel]
                taps.append(TensorAccessPattern((self.M * self.N * 9 // 16,),
                                                (ch * rows * m // 8 * self.N + cols[p0]) // 8 * 36,
                                                [cnt, rows * m // 8, no // 8, 36],
                                                [step // 8 * 36, self.N // 8 * 36, 36, 1]))
            else:
                taps.append(TensorAccessPattern((self.M, self.N), ch * rows * m * self.N + cols[p0],
                                                [1, cnt, rows * m, no], [0, step, self.N, 1]))
        return taps

    def _sequence(self, a_prods: list, b_prods: list, c_conses: list, ROWS: int) -> object:
        """The runtime sequence (A, W, C): weight fills (``b_taps``), A strips (``a_tap``) of the
        ``ROWS`` token rows and C drains (``c_taps``)."""

        def sequence(A, W, C, _):
            # per token chunk one task group: its B fills (one for all passes, a chained one, or <= 4
            # per-pass ones: 1B down), A strips and C drains (one drain BD per run of evenly spaced
            # pass outputs covers many C objects); at most 2 chunks in flight. A chunk's last drain
            # completing proves its fills completed, so its BDs are recycled then.
            taps = [self.b_taps(c) for c in range(COLS)]
            chains = [self.b_chain(c) for c in range(COLS)]
            assert all(ch_ is not None for ch_ in chains) or max(len(t) for t in taps) <= 4, \
                "per-pass weight fills beyond a shim queue (4 tasks) and no chained fill (passes > 64)"
            assert chains[0] is None or len(chains[0]) <= 4, "chained fills beyond a shim queue (4 tasks)"
            # a chained weight fill holds one BD per K chunk until its chunk's drains complete
            depth = 1 if chains[0] is not None and sum(map(len, chains[0])) > 4 else 2
            inflight: list[TaskGroup] = []
            for ch in range(self.mc):
                tg = TaskGroup()
                for c in range(COLS):
                    if chains[c] is not None:
                        for chain in chains[c]:
                            chained_fill(b_prods[c], W, chain, tg)
                        continue
                    for _, tap in taps[c]:
                        b_prods[c].fill(W, tap, group=tg, wait=False)
                for row in range(ROWS):
                    a_prods[row].fill(A, self.a_tap(ch, row), group=tg, wait=False)
                for c in range(COLS):
                    if self.G > 1:
                        chained_fill(c_conses[c], C, self.c_chain(ch, c), tg, wait=True)
                        continue
                    for tap in self.c_taps(ch, c):
                        c_conses[c].drain(C, tap, group=tg, wait=True)
                inflight.append(tg)
                if len(inflight) >= depth:
                    inflight.pop(0).finish()
            for tg in inflight:
                tg.finish()

        return sequence

    def get_mlir_artifact(self) -> Module:
        ROWS = self.nrows  # shadows the module default
        m, n = self.m, self.n
        i32 = np.int32
        df = self.dataflow()
        B_l2_ty = np.ndarray[(df.b_obj,), np.dtype[i32]]
        B_l1_ty = np.ndarray[(df.b_l1,), np.dtype[i32]]
        cb = self.c_bfp16
        A_l2_ty = np.ndarray[(df.a_obj,), np.dtype[bfloat16]]
        A_l1_ty = np.ndarray[(df.a_l1,), np.dtype[bfloat16]]
        no = self.n_out
        c_elems = df.c_l1
        C_l1_ty = np.ndarray[(c_elems,) if cb else (m, no), np.dtype[bfloat16]]
        C_l2_ty = np.ndarray[(ROWS * c_elems,) if cb else (ROWS * m, no), np.dtype[bfloat16]]
        acc_ty = np.ndarray[(m, n), np.dtype[np.float32 if self.c_pairs else bfloat16]]  # MAC target (silu, c_pairs)
        bufs = self._buffers(m)
        # the second C tile (dequant: also the collected header) and the SiLU epilogue's scratch (m x n / 2 bf16)
        st_ty = np.ndarray[(bufs["st"][1] // 2,), np.dtype[bfloat16]]

        p = self.p
        k_zero = Kernel(f"{p}_zerof" if self.c_pairs else f"{p}_zero", "kernel.o", [acc_ty])
        k_silu = Kernel(f"{p}_{self.act}", "kernel.o", [acc_ty, C_l1_ty, st_ty, np.int32,
                                                         *[np.int32] * (self.c_parts - 1)]) if self.silu else None
        pair = self.pair
        if pair:
            ws_ty = np.ndarray[(bufs["ws"][1] // 2,), np.dtype[bfloat16]]  # planes x nsb x 128 tq
            X_ty = np.ndarray[(bufs["X"][1],), np.dtype[np.uint8]]  # a core's dequantized half of a k-tile
            k_hdr, k_hdr2 = slot(self.fmt).gemm_q_header(
                self, lambda name, args: Kernel(f"{p}_{name}", "kernel.o", args), B_l1_ty, ws_ty, st_ty)
            k_dq = Kernel(f"{p}_dq", "kernel.o", [B_l1_ty, ws_ty, X_ty, np.int32, np.int32])
            k_mm = Kernel(f"{p}_mm", "kernel.o", [A_l1_ty, X_ty, X_ty, acc_ty, st_ty, np.int32, np.int32])
        else:
            k_hdr2 = None
            k_mm = Kernel(f"{p}_mmf" if self.c_pairs else f"{p}_mm", "kernel.o", [A_l1_ty, B_l1_ty, acc_ty])
        k_pairs = Kernel(f"{p}_pairs", "kernel.o", [acc_ty, C_l1_ty]) if self.c_pairs else None

        # B: shim -> memtile -> the column's cores (broadcast), k-tile objects in dequant / MAC order (pair: segment
        # h of each memtile object to the cores of half h, rows h, h + 2)
        b_l3 = [ObjectFifo(B_l2_ty, name=f"gqB_{c}", depth=2, from_stream_per_cons=df.b_wdims)
                for c in range(COLS)]
        if pair:
            b_l1 = [b_l3[c].cons().split([df.b_obj // 2 * h for h in range(2)], tile=Tile(c, 1), depths=[B_DEPTH] * 2,
                                          obj_types=[B_l1_ty] * 2, names=[f"gqB1_{c}_{h}" for h in range(2)],
                                          to_stream=[df.b_rdims] * 2)
                    for c in range(COLS)]
        else:
            b_l1 = [b_l3[c].cons().forward(obj_type=B_l1_ty, name=f"gqB1_{c}", depth=B_DEPTH, tile=Tile(c, 1),
                                            to_stream=df.b_rdims)
                    for c in range(COLS)]
        # each core's dequantized half, consumed by its partner (adjacent cores: shared memory, no DMA)
        x_f = {(c, row): ObjectFifo(X_ty, name=f"gqX_{c}_{row}", depth=2) for c in range(COLS) for row in range(ROWS)
               } if pair else {}
        # A: one strip per core row in memtile (2 row, 1), broadcast to the 8 columns
        a_l3 = [ObjectFifo(A_l2_ty, name=f"gqA_{row}", depth=1 if df.a_res else 2,
                           from_stream_per_cons=df.a_wdims)
                for row in range(ROWS)]
        a_l1 = [a_l3[row].cons().forward(obj_type=A_l1_ty, name=f"gqA1_{row}", depth=2, tile=Tile(2 * row, 1),
                                          repeat_count=df.a_repeat if df.a_repeat > 1 else None)
                for row in range(ROWS)]
        # C: the column's cores join in its memtile
        c_l3 = []
        c_l1: list[list[ObjectFifo]] = [[] for _ in range(ROWS)]
        for c in range(COLS):
            f = ObjectFifo(C_l2_ty, name=f"gqC_{c}", depth=2, to_stream=df.c_dims)
            c_l3.append(f)
            parts = f.prod().join([c_elems * i for i in range(ROWS)], obj_types=[C_l1_ty] * ROWS,
                                  names=[f"gqC1_{c}_{row}" for row in range(ROWS)], depths=[2] * ROWS,
                                  tile=Tile(c, 1))
            for row in range(ROWS):
                c_l1[row].append(parts[row])

        n_kc, planes = self.n_kc, self.hdr_units
        q4k = k_hdr2 is not None  # (Q4_K: the header slots collected in st, expanded per K chunk)

        silu = self.silu

        c_pairs = self.c_pairs

        parts = self.c_parts  # (C objects of two passes: their outputs in turn)

        def core_fn(prm, bf, af, cf, zero, mm, st, *extra):  # BFP16: the k-tile objects are MMUL panels already
            for _ in self._chunk_passes(prm, step=parts):  # token chunks x passes
                c = cf.acquire(1)
                for h in range(parts):
                    acc = extra[1] if silu or c_pairs else c
                    zero(acc)
                    for _ in range_(self.k_rep * n_kc * (self.kc // KT)):
                        b = bf.acquire(1)
                        a = af.acquire(1)
                        mm(a, b, acc)
                        af.release(1)
                        bf.release(1)
                    if silu:
                        extra[0](acc, c, st, 0, *([h] if parts == 2 else []))  # (0: the kernel's opaque runtime zero)
                    elif c_pairs:
                        extra[0](acc, c)
                cf.release(1)

        def pair_fn(prm, bf, af, cf, xo, xi, zero, hdr, dq, mm, ws, st, h, *extra):
            for _ in self._chunk_passes(prm, step=parts):  # token chunks x passes
                c = cf.acquire(1)
                for part in range(parts):
                    pass_fn(bf, af, xo, xi, zero, hdr, dq, mm, ws, st, h, c, part, *extra)
                cf.release(1)

        def pass_fn(bf, af, xo, xi, zero, hdr, dq, mm, ws, st, h, c, part, *extra):
            acc = extra[1] if silu else c
            zero(acc)
            for _ in range_(n_kc):
                for pl in range(planes):  # header planes, one k-tile slot per L1 object (this half's tiles)
                    for kt in range_(4):
                        b = bf.acquire(1)
                        hdr(b, st if q4k else ws, *([] if q4k else [pl]), kt)
                        bf.release(1)
                if q4k:
                    extra[-1](st, ws)
                for sb in range_(self.nsb):
                    for kt in range_(4):
                        b = bf.acquire(1)
                        x = xo.acquire(1)
                        dq(b, ws, x, sb, kt)
                        # (the partner may read x now; this core keeps reading it: the FIFO hands the object back
                        # for the next k-tile but one, after the partner's release)
                        xo.release(1)
                        bf.release(1)
                        a = af.acquire(1)
                        y = xi.acquire(1)
                        mm(a, x, y, acc, st, h, kt)  # (the C tiles acc and st in turn: an even count per chunk)
                        xi.release(1)
                        af.release(1)
            if silu:
                extra[0](acc, c, st, 0, *([part] if parts == 2 else []))  # (0: the kernel's opaque runtime zero)

        workers = []
        prm = self._mc_prm()
        for row in range(ROWS):
            for c in range(COLS):
                st = Buffer(st_ty, name=f"gq_st_{c}_{row}")
                extra = ([k_silu, Buffer(acc_ty, name=f"gq_acc_{c}_{row}")] if self.silu else
                         [k_pairs, Buffer(acc_ty, name=f"gq_acc_{c}_{row}")] if self.c_pairs else [])
                if k_hdr2 is not None:
                    extra.append(k_hdr2)
                if pair:
                    h = row % 2
                    ws = Buffer(ws_ty, name=f"gq_ws_{c}_{row}")
                    args = [prm, b_l1[c][h].cons(), a_l1[row].cons(), c_l1[row][c].prod(), x_f[c, row].prod(),
                            x_f[c, row ^ 1].cons(), k_zero, k_hdr, k_dq, k_mm, ws, st, h, *extra]
                    workers.append(Worker(pair_fn, args, tile=Tile(c, row + 2), stack_size=2048,
                                          dynamic_objfifo_lowering=True))
                    continue
                workers.append(Worker(core_fn, [prm, b_l1[c].cons(), a_l1[row].cons(), c_l1[row][c].prod(), k_zero, k_mm,
                                                 st, *extra],
                                      tile=Tile(c, row + 2), stack_size=2048, dynamic_objfifo_lowering=True))

        A_ty = np.ndarray[(self.M * self.a_K * 9 // 16,), np.dtype[bfloat16]]
        W_ty = np.ndarray[(self.w_bytes // 2,), np.dtype[bfloat16]]  # bf16: fused ELFs pass bf16 buffers
        C_ty = np.ndarray[(self.M * self.N * 9 // 16 if cb else self.M * self.N,), np.dtype[bfloat16]]
        a_prods = [a_l3[row].prod(tile=Tile(2 * row, 0)) for row in range(ROWS)]
        b_prods = [b_l3[c].prod(tile=Tile(c, 0)) for c in range(COLS)]
        c_conses = [c_l3[c].cons(tile=Tile(c, 0)) for c in range(COLS)]

        sequence = self._sequence(a_prods, b_prods, c_conses, ROWS)
        rt = Runtime(sequence, [A_ty, W_ty, C_ty, [a_prods, b_prods, c_conses]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        # the L1 budget / pinned layout describe these buffers
        def nbytes(ty: object) -> int:
            shape, dt = ty.__args__
            return int(np.prod(shape)) * np.dtype(dt.__args__[0]).itemsize

        actual = {"A": nbytes(A_l1_ty), "B": nbytes(B_l1_ty), "C": nbytes(C_l1_ty), "acc": nbytes(acc_ty),
                  "st": nbytes(st_ty), **({"ws": nbytes(ws_ty), "X": nbytes(X_ty)} if pair else {})}
        assert actual == {k: b for k, (_, b) in self._buffers(m).items()}, (actual, self._buffers(m))
        layout = self._l1_layout()
        if layout is not None:
            roles = {"gqA1": "A", "gqB1": "B", "gqC1": "C", "gqX": "X"}

            def worker_buffer(name: str) -> int | None:  # gq_<role>_<col>_<row>
                role = name.split("_")[1] if name.startswith("gq_") else None
                return layout[role][0] if role in layout else None

            pin_l1_buffers(module, fifo=lambda name, depth: layout.get(roles.get(name.split("_")[0], "")),
                           buffer=worker_buffer)
        return module


@dataclass(frozen=True)
class ChainPart:
    """A BD of ``chained_fill`` reading another runtime buffer than the fill's, or with a scratchpad parameter: the
    element offset added to the tap's, or the transfer's length in passes of the tap (as ``ObjectFifoHandle.fill``),
    or in ``length_unit`` elements (a tap of one pass per unit: [1, 2, d1, d0] with length_unit d1 d0 keeps d2's stride,
    which a d2 of size 1 loses)."""

    src: object
    tap: TensorAccessPattern
    offset_parameter: object = None
    length_parameter: object = None
    length_unit: int | None = None


def chained_fill(prod: object, src: object, taps: list[TensorAccessPattern | ChainPart], group: TaskGroup, *,
                 wait: bool = False, repeat_parameter: object = None) -> None:
    """A runtime-sequence fill (or drain: a consumer handle) of ``prod`` (a shim ObjectFifoHandle) from ``src`` as ONE
    task whose BD chain reads ``taps`` in order, the task repeated taps[0].sizes[0] times: every tap is
    [repeat, d2, d1, d0] with the repeat as its BD's iteration (all taps the same repeat count), so
    the stream is [repeat][tap][d2][d1][d0]. A ``ChainPart`` reads its own buffer and parameters. Enrolled in
    ``group`` (freed when it finishes; ``wait``: awaited). ``repeat_parameter``: a ScratchpadParameter n, the task
    runs n + 1 times (at most taps[0].sizes[0])."""
    from aie.dialects.aie import EndOp, bds, next_bd
    from aie.dialects.aiex import dma_configure_task_for, dma_start_task, shim_dma_bd
    from aie.iron.runtime._context import active_sequence
    from aie.iron.runtime.dmatask import DMATask
    from aie.iron.runtime.endpoint import RuntimeEndpoint

    rt = active_sequence()._runtime
    rp = rt.register_parameter(repeat_parameter)
    parts = [t if isinstance(t, ChainPart) else ChainPart(src, t) for t in taps]
    parts = [ChainPart(p.src, p.tap, rt.register_parameter(p.offset_parameter),
                       rt.register_parameter(p.length_parameter), p.length_unit) for p in parts]
    reps = parts[0].tap.sizes[0]
    assert all(len(p.tap.sizes) == 4 and p.tap.sizes[0] == reps for p in parts) and reps <= 64

    class _Chain(DMATask):
        def resolve(self, loc: object = None, ip: object = None) -> None:
            task = dma_configure_task_for(self._alloc, repeat_count=reps - 1, issue_token=wait)
            with bds(task) as bd:
                for i, p in enumerate(parts):
                    with bd[i]:
                        shim_dma_bd(p.src.op, tap=p.tap, offset_parameter=p.offset_parameter,
                                    length_parameter=p.length_parameter, length_unit=p.length_unit,
                                    transfer_len=0 if p.length_parameter is not None else None)
                        if i + 1 < len(parts):
                            next_bd(bd[i + 1])
                        else:
                            EndOp()
            self._task = task
            dma_start_task(task, repeat_parameter=rp)

    if prod._endpoint is None:
        prod.endpoint = RuntimeEndpoint(prod._shim_tile)
    active_sequence().note_fifo(prod)
    active_sequence().emit_transfer(_Chain(prod.name, src, tap=parts[0].tap, task_group=group, wait=wait), group)


def _runs(cols: list[int]) -> list[tuple[int, int, int]]:
    """Split pass output columns into runs (first pass, count, column step) of at most 64 (a drain BD's iteration
    count)."""
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < len(cols):
        step = cols[i + 1] - cols[i] if i + 1 < len(cols) else 1
        j = i + 1
        while j < len(cols) and cols[j] - cols[j - 1] == step and step > 0 and j - i < 64:
            j += 1
        out.append((i, j - i, max(step, 1)))
        i = j
    return out
