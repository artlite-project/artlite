"""The prefills' residual stream: RMSNorms (with the residual update) whose normalized rows go out as bfp16ebs8
panels of 8 tokens x 8 columns, [T / 8][E / 8][72 B] (the layout GemmQOperator reads: half the bytes of bf16 through the
GEMM's A port).

``ResidualNormOperator`` (Gemma): the residual x (T, E) fp32, modes

- ``in`` (the first layer's input): xn = x / rms(x) w1;
- ``post`` (Gemma after the O projection and after a layer's per-layer projection): x' = (x + d / rms(d) w_post) s,
  then xn = x' / rms(x') w1 (the next GEMM's input);
- ``plain`` (Gemma after the down projection): x' = x + d / rms(d) w_post, xn = x' (the per-layer input gate reads the
  residual itself);

with d the projection's output (bf16, column c's E / 8 values at c d_pitch of a row of 8 d_pitch; ``deltas`` = 2, 4:
the sum of K parts, d_stride elements apart), s the layer's output scale (``post`` after the per-layer projection; 1
elsewhere). x' overwrites x.

The arithmetic: d w_post exact in fp32 (d1 w + d2 w for two parts), times a = s / rms(d) (fp32, as bf16 hi / lo; the
product's hi / lo parts: three products), plus x s (x hi / lo times s hi / lo, or x itself when s = 1): within ~2^-16
of the exact update. The norm's input xn = bf16(x') w1 / rms(x') (rms of the bf16
values; 1 / rms as bf16 hi / lo) as the GEMM's single bfp16 panels.

``cols`` cores (one per column) take the token groups of 8 in turn (core c: groups c, c + cols, ..). Per group the
memtile holds x (8 x E fp32) and d and sends them to the core once per pass (``repeat_count``), in chunks of 8 tokens x
``ch`` columns; each pass works in panels (8 tokens x 8 columns: the lanes 8 t + e): pass A sums d's squares (``in``:
x's), pass B computes x' (and its squares; ``plain``: also its panels), pass C x' again (the same
arithmetic) and its panels. The parameters (w_post, w1, s: ``residual_params``, [8 rows][E] bf16: row 0 w_post, row 1
w1, row 2 the hi / lo bits of s) are d's first object.

``PrefillNormOperator`` (Qwen, Llama): for a (T, E) bf16 prompt activation x (T a multiple of 64, E the hidden size)
  add:    x += delta (bf16, written back), xn = rmsnorm(x) * w
  no add: xn = rmsnorm(x) * w
with xn as panels.

``cols`` cores (one per column), 8 tokens per step: core c takes token groups c, c + cols, ...
Per group, pass A streams x (and delta) in chunks of 8 tokens x ``ch`` columns (256; E / 8 below 2048: the norm weight
is one chunk-sized object of the x stream), keeps x (+ delta)
in the group's x_out object (16 E bytes, the buffer of pass B) and sums the squares; pass B
gathers 8 x 8 panels, multiplies by w and 1 / rms (bf16 hi + lo) and converts them to
bfp16. The norm weight is the first object of the x stream.

E > 2048 ("replay": a group does not fit L1 next to the streams): x (and delta) reach the core
through its memtile, which holds the group and sends it twice; pass A only sums the squares,
pass B adds x + delta again, writes x_out chunk by chunk and emits the panels. The norm weight
is then the first group object of the x stream (w read 8 times: a group-shaped object).

More than two deltas ("hold": the replay's memtile would hold a group's deltas twice over): each core a contiguous
token range in groups of 4, a group's x + d1 + .. + dn (4 x E bf16, at most 40 KB) held in core L1 between pass A (x
chunks straight from the shim, the deltas' chunks re-laid by the memtile) and pass B, which emits half panels (the
group's 4 tokens of each panel's 8, two per 64-lane vector); the memtile joins two groups' halves into whole panel
rows for the shim; every byte read once.

Runtime sequence: (x, [delta,] w, [x_out,] xn) with x_out = x in place (add mode; ``deltas`` = n > 1:
delta = [d1 | d2 | ..] (``d_stride`` apart), x + d1 + .. + dn summed in fp32 and rounded once, the outputs of a
projection in K parts; two: replay mode whatever E, more: hold mode).
``d_pitch``: delta's column c (E / 8 values) at c d_pitch of a row of 8 d_pitch (a Q6_K down
projection's padding rows after each column's values).
"""

import numpy as np
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ..gemm.gemm_q import chained_fill
from ..operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from ..operators.design import COLS, stream_walk
from ..operators.kernel_source import kernel_prefix
from ..operators.util import Attr, bf16_value, regions

CH = 256  # columns per chunk (PrefillNormOperator; E / 8 for E < 2048)

_SRC = JINJA_ENV.from_string(r"""
namespace {
// st: ss[8][32] -> inv_hi[64] | inv_lo[64] at st + 256 (bf16, lane 8 t + e = token t's 1 / rms)
__attribute__((noinline)) void {{ p }}_inv(float *__restrict st)
{
    float *__restrict q = st + 256;  // (fp32 scratch under the output: the tokens' sums folded to 8 lanes, 8 t + e)
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 4; t++) {  // (two tokens: their chunks of 8 summed by two unzips; a 64-lane fp32 vector built
                                   // in registers spills whole accumulators)
        const aie::vector<float, 64> v = aie::load_v<64>(st + 64 * t);
        const auto [a, b] = aie::interleave_unzip(v.template extract<32>(0), v.template extract<32>(1), 8);
        const aie::vector<float, 32> h = aie::add(a, b);
        const auto [c, d] = aie::interleave_unzip(h.template extract<16>(0), h.template extract<16>(1), 8);
        aie::store_v(q + 16 * t, aie::add(c, d));
    }
    aie::vector<bfloat16, 16> hi, lo;
    al::split2<16>(al::inv_rms<{{ E }}, 8>(al::group8<false>(aie::load_v<64>(q)).template grow<16>(), {{ eps }}f), hi, lo);
    aie::store_v((bfloat16 *)(st + 256), al::expand8(hi.template extract<8>(0)));
    aie::store_v((bfloat16 *)(st + 256) + 64, al::expand8(lo.template extract<8>(0)));
}
}  // namespace

extern "C" {

void {{ p }}_w(const bfloat16 *__restrict in, bfloat16 *__restrict w)
{
    al::copy<{{ E }}, 32>(w, in);
}

// pass A, chunk kc of a group: g[t][256 kc + j] = x (+ d), ss[t][32 lanes] += squares
void {{ p }}_a(const bfloat16 *__restrict x, const bfloat16 *__restrict d, bfloat16 *__restrict g,
                 float *__restrict ss, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int t = 0; t < 8; t++) {
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)(x + {{ C }} * t);
        const bfloat16 __aie_dm_resource_b *__restrict pd = (const bfloat16 __aie_dm_resource_b *)(d + {{ C }} * t);
        bfloat16 *__restrict pg = g + {{ E }} * t + {{ C }} * kc;
        aie::accum<accfloat, 32> acc(aie::load_v<32>(ss + 32 * t));
        if (kc == 0)
            acc = aie::zeros<accfloat, 32>();
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ C // 32 }}; j++) {
            aie::vector<bfloat16, 32> v = aie::add(aie::load_v<32>(px), aie::load_v<32>(pd));
            px += 32;
            pd += 32;
            aie::store_v(pg, v);
            pg += 32;
            acc = aie::mac_square(acc, v);
        }
        aie::store_v(ss + 32 * t, acc.template to_vector<float>());
    }
}

// the same without the residual add
{% if pre %}
// pass A without a delta, its rms over x w1 (``pre``): g[t][256 kc + j] = x w1 (bf16), ss[t][32 lanes] += its squares
void {{ p }}_a0p(const bfloat16 *__restrict x, bfloat16 *__restrict g, const bfloat16 *__restrict w1, float *__restrict ss,
                 int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int t = 0; t < 8; t++) {
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)(x + {{ C }} * t);
        const bfloat16 __aie_dm_resource_b *__restrict pw = (const bfloat16 __aie_dm_resource_b *)(w1 + {{ C }} * kc);
        bfloat16 *__restrict pg = g + {{ E }} * t + {{ C }} * kc;
        aie::accum<accfloat, 32> acc(aie::load_v<32>(ss + 32 * t));
        if (kc == 0)
            acc = aie::zeros<accfloat, 32>();
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ C // 32 }}; j++) {
            const aie::vector<bfloat16, 32> v = aie::mul(aie::load_v<32>(px), aie::load_v<32>(pw)).template to_vector<bfloat16>();
            px += 32;
            pw += 32;
            aie::store_v(pg, v);
            pg += 32;
            acc = aie::mac_square(acc, v);
        }
        aie::store_v(ss + 32 * t, acc.template to_vector<float>());
    }
}

{% endif %}
void {{ p }}_a0(const bfloat16 *__restrict x, bfloat16 *__restrict g, float *__restrict ss, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int t = 0; t < 8; t++) {
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)(x + {{ C }} * t);
        bfloat16 *__restrict pg = g + {{ E }} * t + {{ C }} * kc;
        aie::accum<accfloat, 32> acc(aie::load_v<32>(ss + 32 * t));
        if (kc == 0)
            acc = aie::zeros<accfloat, 32>();
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ C // 32 }}; j++) {
            aie::vector<bfloat16, 32> v = aie::load_v<32>(px);
            px += 32;
            aie::store_v(pg, v);
            pg += 32;
            acc = aie::mac_square(acc, v);
        }
        aie::store_v(ss + 32 * t, acc.template to_vector<float>());
    }
}

{% if replay %}
}  // extern "C"

// x + d + d2 in fp32, rounded once to bf16 (a residual add of a projection's two K parts)
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_add2(aie::vector<bfloat16, 32> x, aie::vector<bfloat16, 32> d,
                                                                    aie::vector<bfloat16, 32> d2)
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    return aie::mac(aie::mac(aie::accum<accfloat, 32>(x), d, one), d2, one).template to_vector<bfloat16>();
}

extern "C" {

// replay mode, the weight object's chunk kc (w repeated in its 8 rows) -> w[256 kc ..]
void {{ p }}_wc(const bfloat16 *__restrict in, bfloat16 *__restrict w, int32_t kc)
{
    al::copy<{{ C }}, 32>(w + {{ C }} * kc, in);
}

{% for nd, sfx in ((1, ""), (0, "0"), (2, "2")) %}
// replay mode, pass A, chunk kc of a group: ss[t][32 lanes] += squares of x{{ " + d" if nd == 1 }}{{ " + d + d2 (summed in fp32, rounded once)" if nd == 2 }}
void {{ p }}_ra{{ sfx }}(const bfloat16 *__restrict x, {{ "const bfloat16 *__restrict d, " if nd }}float *__restrict ss, int32_t kc)
{
{% if nd == 2 %}
    const bfloat16 *__restrict d2 = d + {{ 8 * C }};  // (the chunk object holds both deltas' chunks)
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int t = 0; t < 8; t++) {
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)(x + {{ C }} * t);
{% if nd %}
        const bfloat16 __aie_dm_resource_b *__restrict pd = (const bfloat16 __aie_dm_resource_b *)(d + {{ C }} * t);
{% endif %}
{% if nd == 2 %}
        const bfloat16 *__restrict pd2 = d2 + {{ C }} * t;
{% endif %}
        aie::accum<accfloat, 32> acc(aie::load_v<32>(ss + 32 * t));
        if (kc == 0)
            acc = aie::zeros<accfloat, 32>();
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ C // 32 }}; j++) {
{% if nd == 2 %}
            const aie::vector<bfloat16, 32> v = {{ p }}_add2(aie::load_v<32>(px), aie::load_v<32>(pd), aie::load_v<32>(pd2));
            pd += 32;
            pd2 += 32;
{% elif nd %}
            aie::vector<bfloat16, 32> v = aie::add(aie::load_v<32>(px), aie::load_v<32>(pd));
            pd += 32;
{% else %}
            aie::vector<bfloat16, 32> v = aie::load_v<32>(px);
{% endif %}
            px += 32;
            acc = aie::mac_square(acc, v);
        }
        aie::store_v(ss + 32 * t, acc.template to_vector<float>());
    }
}

// replay mode, pass B, chunk kc: v = x{{ " + d" if nd == 1 }}{{ " + d + d2" if nd == 2 }}{{ " -> xo (x_out)" if nd }}, 32 panels of v * w * (1 / rms) -> out
void {{ p }}_rb{{ sfx }}(const bfloat16 *__restrict x, {{ "const bfloat16 *__restrict d, " if nd }}const bfloat16 *__restrict w,
                float *__restrict st, bfloat16 *__restrict out, {{ "bfloat16 *__restrict xo, " if nd }}int32_t kc)
{
{% if nd == 2 %}
    const bfloat16 *__restrict d2 = d + {{ 8 * C }};
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict ih = (bfloat16 *)(st + 256);
    bfloat16 *__restrict il = ih + 64;
    if (kc == 0)
        {{ p }}_inv(st);
    const aie::vector<bfloat16, 64> vh = aie::load_v<64>(ih);
    const aie::vector<bfloat16, 64> vl = aie::load_v<64>(il);
{% if nd == 2 %}
    {  // (x + d + d2 first, into xo: the panels read it)
        const bfloat16 *__restrict qx = x, *__restrict qd = d, *__restrict qd2 = d2;
        bfloat16 *__restrict qo = xo;
        for (int i = 0; i < {{ 8 * C // 32 }}; i++) {
            aie::store_v(qo, {{ p }}_add2(aie::load_v<32>(qx), aie::load_v<32>(qd), aie::load_v<32>(qd2)));
            qx += 32;
            qd += 32;
            qd2 += 32;
            qo += 32;
        }
    }
{% endif %}
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> o((bfp16ebs8 *)out);
    const bfloat16 *__restrict px = {{ "xo" if nd == 2 else "x" }};
{% if nd == 1 %}
    const bfloat16 *__restrict pd = d;
{% endif %}
    const bfloat16 *__restrict pw = w + {{ C }} * kc;
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ C // 8 }}; kb++) {
        aie::vector<bfloat16, 64> xp = aie::concat(
            aie::concat(aie::load_v<8>(px), aie::load_v<8>(px + {{ C }})),
            aie::concat(aie::load_v<8>(px + {{ 2 * C }}), aie::load_v<8>(px + {{ 3 * C }})),
            aie::concat(aie::load_v<8>(px + {{ 4 * C }}), aie::load_v<8>(px + {{ 5 * C }})),
            aie::concat(aie::load_v<8>(px + {{ 6 * C }}), aie::load_v<8>(px + {{ 7 * C }})));
{% if nd == 1 %}
        xp = aie::add(xp, aie::concat(
            aie::concat(aie::load_v<8>(pd), aie::load_v<8>(pd + {{ C }})),
            aie::concat(aie::load_v<8>(pd + {{ 2 * C }}), aie::load_v<8>(pd + {{ 3 * C }})),
            aie::concat(aie::load_v<8>(pd + {{ 4 * C }}), aie::load_v<8>(pd + {{ 5 * C }})),
            aie::concat(aie::load_v<8>(pd + {{ 6 * C }}), aie::load_v<8>(pd + {{ 7 * C }}))));
        pd += 8;
{% endif %}
        aie::vector<bfloat16, 64> xw = aie::mul(xp, aie::load_v<8>(pw).template grow_replicate<64>())
                                           .template to_vector<bfloat16>();
        o.push(aie::mac(aie::mul(xw, vh), xw, vl).template to_vector<bfp16ebs8>());
        px += 8;
        pw += 8;
    }
{% if nd == 1 %}
    {  // (walked pointers: II 1 instead of an index shared by three streams)
        const bfloat16 *__restrict qx = x, *__restrict qd = d;
        bfloat16 *__restrict qo = xo;
        for (int i = 0; i < {{ 8 * C // 32 }}; i++) {
            aie::store_v(qo, aie::add(aie::load_v<32>(qx), aie::load_v<32>(qd)));
            qx += 32;
            qd += 32;
            qo += 32;
        }
    }
{% endif %}
}
{% endfor %}
{% else %}
// pass B, chunk kc: 32 panels (8 tokens x 8 columns) of g * w * (1 / rms) as bfp16ebs8 -> out.
// st: ss[8][32] | inv_hi[64] | inv_lo[64] (bf16, lane 8 t + e = the token's 1 / rms)
void {{ p }}_b(const bfloat16 *__restrict g, const bfloat16 *__restrict w, float *__restrict st,
                 bfloat16 *__restrict out, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict ih = (bfloat16 *)(st + 256);
    bfloat16 *__restrict il = ih + 64;
    if (kc == 0)
        {{ p }}_inv(st);
    const aie::vector<bfloat16, 64> vh = aie::load_v<64>(ih);
    const aie::vector<bfloat16, 64> vl = aie::load_v<64>(il);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> o((bfp16ebs8 *)out);
    const bfloat16 *__restrict pg = g + {{ C }} * kc;
    const bfloat16 *__restrict pw = w + {{ C }} * kc;
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ C // 8 }}; kb++) {
        aie::vector<bfloat16, 64> xp = aie::concat(
            aie::concat(aie::load_v<8>(pg), aie::load_v<8>(pg + {{ E }})),
            aie::concat(aie::load_v<8>(pg + {{ 2 * E }}), aie::load_v<8>(pg + {{ 3 * E }})),
            aie::concat(aie::load_v<8>(pg + {{ 4 * E }}), aie::load_v<8>(pg + {{ 5 * E }})),
            aie::concat(aie::load_v<8>(pg + {{ 6 * E }}), aie::load_v<8>(pg + {{ 7 * E }})));
        aie::vector<bfloat16, 64> xw = aie::mul(xp, aie::load_v<8>(pw).template grow_replicate<64>())
                                           .template to_vector<bfloat16>();
        o.push(aie::mac(aie::mul(xw, vh), xw, vl).template to_vector<bfp16ebs8>());
        pg += 8;
        pw += 8;
    }
}

{% if lo %}
// pass B with the lo panels: the chunk's 32 panels of v = g * w * (1 / rms) as bfp16(v) -> out, then 32 of
// bfp16(v - bfp16(v)) -> out + 32 panels (x hi + lo: an MoE router's input). g w is exact in fp32 (bf16 x bf16) and
// split into bf16 hi + lo; v = (hi + lo)(inv hi + inv lo) in fp32
void {{ p }}_bl(const bfloat16 *__restrict g, const bfloat16 *__restrict w, float *__restrict st,
                bfloat16 *__restrict out, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict ih = (bfloat16 *)(st + 256);
    bfloat16 *__restrict il = ih + 64;
    if (kc == 0)
        {{ p }}_inv(st);
    const aie::vector<bfloat16, 64> vh = aie::load_v<64>(ih);
    const aie::vector<bfloat16, 64> vl = aie::load_v<64>(il);
    const al::BV nI = al::neg_eye();
    const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> o((bfp16ebs8 *)out);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> ol((bfp16ebs8 *)out + {{ C }});  // (C / 8 panels of 8 blocks)
    const bfloat16 *__restrict pg = g + {{ C }} * kc;
    const bfloat16 *__restrict pw = w + {{ C }} * kc;
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ C // 8 }}; kb++) {
        aie::vector<bfloat16, 64> xp = aie::concat(
            aie::concat(aie::load_v<8>(pg), aie::load_v<8>(pg + {{ E }})),
            aie::concat(aie::load_v<8>(pg + {{ 2 * E }}), aie::load_v<8>(pg + {{ 3 * E }})),
            aie::concat(aie::load_v<8>(pg + {{ 4 * E }}), aie::load_v<8>(pg + {{ 5 * E }})),
            aie::concat(aie::load_v<8>(pg + {{ 6 * E }}), aie::load_v<8>(pg + {{ 7 * E }})));
        const aie::accum<accfloat, 64> xw = aie::mul(xp, aie::load_v<8>(pw).template grow_replicate<64>());
        const aie::vector<bfloat16, 64> wh = xw.template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> wl = aie::msc(xw, wh, one).template to_vector<bfloat16>();
        al::BV h, l;
        al::split_hilo(aie::mac(aie::mac(aie::mac(aie::mul(wh, vh), wh, vl), wl, vh), wl, vl), nI, h, l);
        o.push(h);
        ol.push(l);
        pg += 8;
        pw += 8;
    }
}
{% endif %}
{% endif %}

}  // extern "C"
""")


HG, HCH = 4, 64  # the hold design's tokens per group, columns per chunk (PrefillNormOperator, deltas > 2)

_HOLD_SRC = JINJA_ENV.from_string(r"""
namespace {
// st: ss[4][32] -> inv_hi[64] | inv_lo[64] at st + 128 (bf16, lane 32 h + 8 t + e = token t's 1 / rms)
__attribute__((noinline)) void {{ p }}_inv(float *__restrict st)
{
    aie::vector<float, 64> p = aie::zeros<float, 64>();  // lane 8 t + e: token t's 32 sums folded to 8
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < {{ G }}; t++) {
        const aie::vector<float, 32> v = aie::load_v<32>(st + 32 * t);
        const aie::vector<float, 16> h = aie::add(v.template extract<16>(0), v.template extract<16>(1));
        p.insert(t, aie::add(h.template extract<8>(0), h.template extract<8>(1)));
    }
    const aie::vector<float, 16> s = al::group8<false>(p).template grow<16>();
    const aie::vector<float, 32> v = al::rows4(al::inv_rms<{{ E }}, {{ G }}>(s, {{ eps }}f));
    aie::vector<bfloat16, 64> hi, lo;
    al::split2<64>(aie::concat(v, v), hi, lo);
    aie::store_v((bfloat16 *)(st + 128), hi);
    aie::store_v((bfloat16 *)(st + 128) + 64, lo);
}
}  // namespace

extern "C" {

// the weight object j (w[{{ G * C }} j ..]) -> w
void {{ p }}_w(const bfloat16 *__restrict in, bfloat16 *__restrict w, int32_t j)
{
    al::copy<{{ G * C }}, 32>(w + {{ G * C }} * j, in);
}

// pass A, chunk kc of a group: v = x + d1 + .. + d{{ nd }} (summed in fp32, rounded once) -> the group's v[t][C kc ..]
// and xo (x_out); ss[t][32 lanes] += squares
void {{ p }}_a(const bfloat16 *__restrict x, const bfloat16 *__restrict d, bfloat16 *__restrict v,
               bfloat16 *__restrict xo, float *__restrict ss, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    for (int t = 0; t < {{ G }}; t++) {
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)(x + {{ C }} * t);
        const bfloat16 __aie_dm_resource_b *__restrict pd = (const bfloat16 __aie_dm_resource_b *)(d + {{ C }} * t);
        bfloat16 *__restrict pv = v + {{ E }} * t + {{ C }} * kc;
        bfloat16 *__restrict po = xo + {{ C }} * t;
        aie::accum<accfloat, 32> acc(aie::load_v<32>(ss + 32 * t));
        if (kc == 0)
            acc = aie::zeros<accfloat, 32>();
        for (int j = 0; j < {{ C // 32 }}; j++) {
            aie::accum<accfloat, 32> s(aie::load_v<32>(px));
{% for i in range(nd) %}
            s = aie::mac(s, aie::load_v<32>(pd + {{ G * C * i }}), one);
{% endfor %}
            const aie::vector<bfloat16, 32> u = s.template to_vector<bfloat16>();
            aie::store_v(pv, u);
            aie::store_v(po, u);
            acc = aie::mac_square(acc, u);
            px += 32;
            pd += 32;
            pv += 32;
            po += 32;
        }
        aie::store_v(ss + 32 * t, acc.template to_vector<float>());
    }
}

// pass B, chunk kc: the group's half panels (4 tokens x 8 columns, 36 B) of v w (1 / rms) as bfp16ebs8, two per
// 64-lane vector (lanes 32 h + 8 t + e: column block 2 kb + h)
void {{ p }}_b(const bfloat16 *__restrict v, const bfloat16 *__restrict w, float *__restrict st,
               bfloat16 *__restrict out, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict ih = (bfloat16 *)(st + 128);
    bfloat16 *__restrict il = ih + 64;
    if (kc == 0)
        {{ p }}_inv(st);
    const aie::vector<bfloat16, 64> vh = aie::load_v<64>(ih);
    const aie::vector<bfloat16, 64> vl = aie::load_v<64>(il);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c> o((bfp16ebs8 *)out);
    const bfloat16 *__restrict pv = v + {{ C }} * kc;
    const bfloat16 *__restrict pw = w + {{ C }} * kc;
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ C // 16 }}; kb++) {
        aie::vector<bfloat16, 64> xp = aie::concat(
            aie::concat(aie::load_v<8>(pv), aie::load_v<8>(pv + {{ E }})),
            aie::concat(aie::load_v<8>(pv + {{ 2 * E }}), aie::load_v<8>(pv + {{ 3 * E }})),
            aie::concat(aie::load_v<8>(pv + 8), aie::load_v<8>(pv + {{ E + 8 }})),
            aie::concat(aie::load_v<8>(pv + {{ 2 * E + 8 }}), aie::load_v<8>(pv + {{ 3 * E + 8 }})));
        const aie::vector<bfloat16, 64> wr = aie::concat(aie::load_v<8>(pw).template grow_replicate<32>(),
                                                         aie::load_v<8>(pw + 8).template grow_replicate<32>());
        aie::vector<bfloat16, 64> xw = aie::mul(xp, wr).template to_vector<bfloat16>();
        o.push(aie::mac(aie::mul(xw, vh), xw, vl).template to_vector<bfp16ebs8>());
        pv += 16;
        pw += 16;
    }
}

}  // extern "C"
""")


class PrefillNormOperator(ArtcOperator):
    """(Add +) RMSNorm of a prompt with bfp16ebs8 panel output; see the module docstring."""

    STACK = 1024
    p = kernel_prefix("pn")

    def __init__(self, *, T: int, add: bool = True, E: int = 2048, eps: float = 1e-5, d_pitch: int = 0,
                 cols: int = COLS, deltas: int = 1, d_stride: int = 0, lo: bool = False, pre: bool = False,
                 out_pitch: int = 0, out_col: int = 0):
        """cols: cores (one per column) taking the token groups in turn: few for short prompts, where the
        op's reconfiguration costs more than its work. deltas = 2 (replay mode): x += d + d2 (a projection in two
        K parts: summed in fp32, rounded once), both at d_pitch, d2 ``d_stride`` elements after d
        in the delta argument (default T 8 d_pitch; e.g. the C size of a GEMM that pads the tokens). lo: xn as
        hi + lo panels (an MoE router's x), panels of 2 E columns: per token group [bfp16(v) | bfp16(v - bfp16(v))]
        (one delta, E <= 2048). pre (no add): the weight argument [w1 | w2], xn = (x w1) w2 / rms(x w1) (a norm of a
        norm's output: an MTP drafter's hnorm of the model's output norm). out_pitch, out_col (the plain design): xn's
        panel rows ``out_pitch`` columns wide (default E), these at columns out_col .. (another norm's beside them)."""
        assert T % (8 * cols) == 0 and T > 0 and E % 1024 == 0 and 1 <= cols <= COLS
        self.T, self.add, self.E, self.eps, self.cols, self.lo = T, add, E, eps, cols, lo
        self.d_pitch = d_pitch or E // COLS  # (pitch of delta's column blocks)
        assert self.d_pitch >= E // COLS and (add or not d_pitch)
        self.deltas = deltas if add else 0
        # more deltas: the hold design (the replay design's memtile holds a group's deltas twice over)
        self.hold = self.deltas > 2
        # the group (8 tokens x E bf16) does not fit L1 next to the streams; two deltas: the replay design's
        self.replay = not self.hold and (16 * E > 32768 or (add and deltas == 2))
        assert self.deltas in (0, 1) or (self.deltas == 2 and self.replay) or self.hold
        assert not self.hold or (T % (8 * cols) == 0 and HG * E * 2 <= 40960
                                 and self.deltas * HG * E * 2 + 2 * E // 8 * 72 <= 500 * 1024), (T, E, deltas, cols)
        self.d_stride = d_stride or T * COLS * self.d_pitch
        assert self.d_stride >= T * COLS * self.d_pitch and (self.deltas >= 2 or not d_stride)
        self.ch = min(CH, E // 8)  # (columns per chunk)
        assert self.replay or self.hold or 8 * self.ch == E, "the norm weight: one chunk object"
        assert not lo or not (self.replay or self.hold), "lo panels: the plain design"
        self.pre = pre
        self.out_pitch, self.out_col = out_pitch or E, out_col
        assert not pre or not add, "pre: a norm without a delta"
        assert (self.out_pitch, out_col) == (E, 0) or not (lo or self.replay or self.hold) and (
            out_col % 8 == 0 and out_col + E <= self.out_pitch), (out_pitch, out_col)
        super().__init__()

    def get_arg_spec(self) -> list[ArgSpec]:
        EMB = self.E
        n = self.T * EMB
        specs = [ArgSpec("in", (n,), torch.bfloat16)]
        if self.add:  # (two deltas: one argument, [delta | delta2])
            specs.append(ArgSpec("in", ((self.deltas - 1) * self.d_stride + self.T * COLS * self.d_pitch,),
                                 torch.bfloat16))
        specs.append(ArgSpec("in", ((2 if self.pre else 1) * EMB,), torch.bfloat16))
        if self.add:
            specs.append(ArgSpec("out", (n,), torch.bfloat16))
        specs.append(ArgSpec("out", ((2 if self.lo else 1) * self.T * self.out_pitch * 9 // 16,), torch.bfloat16))
        return specs

    def get_kernel_src(self) -> str:
        if self.hold:
            return _HOLD_SRC.render(p=self.p, E=self.E, C=HCH, G=HG, eps=repr(self.eps), nd=self.deltas)
        # (the same source for add and no add: one copy in a fused ELF)
        return _SRC.render(p=self.p, E=self.E, C=self.ch, eps=repr(self.eps), replay=self.replay, lo=self.lo, pre=self.pre)

    def get_mlir_artifact(self) -> Module:
        if self.hold:
            return self._hold_design()
        if self.replay:
            return self._replay_design()
        T, add = self.T, self.add
        EMB, p = self.E, self.p
        n_g = T // (8 * self.cols)  # token groups per core
        bf = np.dtype[bfloat16]
        CH = self.ch
        chunk_ty = np.ndarray[(8 * CH,), bf]
        grp_ty = np.ndarray[(8 * EMB,), bf]
        nl = 2 if self.lo else 1  # (lo: the chunk's hi panels, then its lo panels)
        pan_ty = np.ndarray[(nl * CH // 8 * 36,), bf]  # CH / 8 panels of 72 B
        w_ty = np.ndarray[(EMB,), bf]
        st_ty = np.ndarray[(256 + 64,), np.dtype[np.float32]]
        E = EMB
        k_w = Kernel(f"{p}_w", "kernel.o", [chunk_ty, w_ty])
        k_a = (Kernel(f"{p}_a", "kernel.o", [chunk_ty, chunk_ty, grp_ty, st_ty, np.int32]) if add else
               Kernel(f"{p}_a0p", "kernel.o", [chunk_ty, grp_ty, w_ty, st_ty, np.int32]) if self.pre else
               Kernel(f"{p}_a0", "kernel.o", [chunk_ty, grp_ty, st_ty, np.int32]))
        k_b = Kernel(f"{p}_bl" if self.lo else f"{p}_b", "kernel.o", [grp_ty, w_ty, st_ty, pan_ty, np.int32])
        pre, P, col = self.pre, self.out_pitch, self.out_col

        xf = [ObjectFifo(chunk_ty, name=f"pn_x{c}", depth=2) for c in range(self.cols)]
        df = [ObjectFifo(chunk_ty, name=f"pn_d{c}", depth=2) for c in range(self.cols)] if add else []
        of = [ObjectFifo(grp_ty, name=f"pn_o{c}", depth=1) for c in range(self.cols)] if add else []
        pf = [ObjectFifo(pan_ty, name=f"pn_p{c}", depth=2) for c in range(self.cols)]

        def core(x_f, p_f, kw, ka, kb, wb, st, *rest):
            if add:
                d_f, o_f = rest
            elif pre:
                gb, w1 = rest
            else:
                (gb,) = rest
            for _ in forever_():
                if pre:  # (pre: w1, then w2)
                    w = x_f.acquire(1)
                    kw(w, w1)
                    x_f.release(1)
                w = x_f.acquire(1)
                kw(w, wb)
                x_f.release(1)
                for _ in range_(n_g):
                    g = o_f.acquire(1) if add else gb
                    for kc in range_(E // CH):
                        x = x_f.acquire(1)
                        if add:
                            d = d_f.acquire(1)
                            ka(x, d, g, st, kc)
                            d_f.release(1)
                        elif pre:
                            ka(x, g, w1, st, kc)
                        else:
                            ka(x, g, st, kc)
                        x_f.release(1)
                    for kc in range_(E // CH):
                        p = p_f.acquire(1)
                        kb(g, wb, st, p, kc)
                        p_f.release(1)
                    if add:
                        o_f.release(1)

        workers = []
        for c in range(self.cols):
            args = [xf[c].cons(), pf[c].prod(), k_w, k_a, k_b, Buffer(w_ty, name=f"pn_w{c}"),
                    Buffer(st_ty, name=f"pn_st{c}")]
            args += [df[c].cons(), of[c].prod()] if add else [Buffer(grp_ty, name=f"pn_g{c}")]
            args += [Buffer(w_ty, name=f"pn_w1{c}")] if pre else []
            workers.append(Worker(core, args, tile=Tile(c, 2), stack_size=self.STACK, while_true=False))

        n = T * E

        def grp(c: int) -> TensorAccessPattern:  # the core's groups as [group][chunk][8 tokens][256]
            return TensorAccessPattern((n,), 8 * E * c, [n_g, E // CH, 8, CH], [8 * self.cols * E, CH, E, 1])

        dp, nd = self.d_pitch, T * COLS * self.d_pitch
        assert dp == E // COLS or E // COLS == CH, "padded delta: one chunk per column block"

        def dgrp(c: int) -> TensorAccessPattern:  # delta's: chunk k = column block k
            ed = COLS * dp
            return TensorAccessPattern((nd,), 8 * ed * c, [n_g, E // CH, 8, CH], [8 * self.cols * ed, dp, ed, 1])

        x_h = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
        d_h = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
        o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
        p_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]

        def seq(*a):
            if add:
                x, d, w, xo, xn, xh, ph, dh, oh = a
            else:
                x, w, xn, xh, ph = a
            tg = TaskGroup()
            for c in range(self.cols):
                # (a shim BD's inner dimension is <= 1023 words: rows of <= 512 words below)
                nw = (2 if pre else 1) * E
                xh[c].fill(w, TensorAccessPattern((nw,), 0, [1, 1, nw // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                xh[c].fill(x, grp(c), group=tg)
                if add:
                    dh[c].fill(d, dgrp(c), group=tg)
                    oh[c].drain(xo, TensorAccessPattern((n,), 8 * E * c, [1, n_g, 8 * E // 1024, 1024],
                                                        [0, 8 * self.cols * E, 1024, 1]), group=tg, wait=True)
                if self.lo:  # (a group's chunks [hi | lo] -> [hi panels | lo panels]: rows of 2 E)
                    ph[c].drain(xn, TensorAccessPattern((2 * n * 9 // 16,), c * (E // 4) * 36,
                                                        [n_g, 8, 2, E // 64 * 36],
                                                        [self.cols * (E // 4) * 36, E // 64 * 36, E // 8 * 36, 1]),
                                group=tg, wait=True)
                    continue
                ph[c].drain(xn, TensorAccessPattern((T * P * 9 // 16,), (c * (P // 8) + col // 8) * 36,
                                                    [1, n_g, 8, E // 64 * 36], [0, self.cols * (P // 8) * 36, E // 64 * 36, 1]),
                            group=tg, wait=True)
            tg.finish()

        vt = np.ndarray[(n,), bf]
        types = [vt] + ([np.ndarray[(nd,), bf]] if add else []) + [np.ndarray[((2 if pre else 1) * E,), bf]]
        types += [vt] if add else []
        types.append(np.ndarray[((2 if self.lo else 1) * T * P * 9 // 16,), bf])
        rt = Runtime(seq, [*types, x_h, p_h, *([d_h, o_h] if add else [])])
        return Program(NPU2(), rt, workers=workers).resolve_program()

    def _replay_design(self) -> Module:
        """E > 2048: x (and delta) groups through the memtile, sent twice (pass A, pass B)."""
        T, add, E, p, nd = self.T, self.add, self.E, self.p, self.deltas
        n_g = T // (8 * self.cols)
        n_kc = E // CH
        bf = np.dtype[bfloat16]
        chunk_ty = np.ndarray[(8 * CH,), bf]
        pan_ty = np.ndarray[(CH // 8 * 36,), bf]
        w_ty = np.ndarray[(E,), bf]
        st_ty = np.ndarray[(256 + 64,), np.dtype[np.float32]]
        k_w = Kernel(f"{p}_wc", "kernel.o", [chunk_ty, w_ty, np.int32])
        sfx = {0: "0", 1: "", 2: "2"}[nd]
        dch = [np.ndarray[(8 * CH * nd,), bf]] if nd else []  # (two deltas: one object of both chunks)
        k_a = Kernel(f"{p}_ra{sfx}", "kernel.o", [chunk_ty, *dch, st_ty, np.int32])
        k_b = Kernel(f"{p}_rb{sfx}", "kernel.o",
                     [chunk_ty, *dch, w_ty, st_ty, pan_ty, *([chunk_ty] if add else []), np.int32])
        # memtile: a group [8 tokens][E] -> the core as chunks [kc][8 tokens][CH], twice; two deltas: their groups
        # side by side, as chunk pairs [kc][delta][8 tokens][CH] (a core has two input DMA channels, a shim two
        # output channels)
        chunks = stream_walk(8 * E, [(n_kc, CH), (8, E), (CH, 1)])
        pairs = stream_walk(16 * E, [(n_kc, CH), (2, 8 * E), (8, E), (CH, 1)])

        def replayed(name: str, c: int, k: int = 1) -> tuple[ObjectFifo, ObjectFifo]:
            l2 = ObjectFifo(np.ndarray[(8 * E * k,), bf], name=f"{name}{c}", depth=2)
            return l2, l2.cons().forward(obj_type=np.ndarray[(8 * CH * k,), bf], name=f"{name}1{c}", depth=2,
                                          tile=Tile(c, 1), to_stream=chunks if k == 1 else pairs, repeat_count=2)

        xf = [replayed("pnx", c) for c in range(self.cols)]
        df = [replayed("pnd", c, nd) for c in range(self.cols)] if nd else []
        of = [ObjectFifo(chunk_ty, name=f"pn_o{c}", depth=2) for c in range(self.cols)] if add else []
        pf = [ObjectFifo(pan_ty, name=f"pn_p{c}", depth=2) for c in range(self.cols)]

        def core(x_f, p_f, kw, ka, kb, wb, st, *rest):
            d_f, o_f = rest if add else (None, None)
            for _ in forever_():
                for rep in range(2):  # the weight object (w in every row), sent twice
                    for kc in range_(n_kc):
                        w = x_f.acquire(1)
                        if rep == 0:
                            kw(w, wb, kc)
                        x_f.release(1)
                for _ in range_(n_g):
                    for kc in range_(n_kc):
                        x = x_f.acquire(1)
                        if add:
                            d = d_f.acquire(1)
                            ka(x, d, st, kc)
                            d_f.release(1)
                        else:
                            ka(x, st, kc)
                        x_f.release(1)
                    for kc in range_(n_kc):
                        x = x_f.acquire(1)
                        pn = p_f.acquire(1)
                        if add:
                            d = d_f.acquire(1)
                            xo = o_f.acquire(1)
                            kb(x, d, wb, st, pn, xo, kc)
                            o_f.release(1)
                            d_f.release(1)
                        else:
                            kb(x, wb, st, pn, kc)
                        p_f.release(1)
                        x_f.release(1)

        workers = []
        for c in range(self.cols):
            args = [xf[c][1].cons(), pf[c].prod(), k_w, k_a, k_b, Buffer(w_ty, name=f"pn_w{c}"),
                    Buffer(st_ty, name=f"pn_st{c}")]
            if add:
                args += [df[c][1].cons(), of[c].prod()]
            workers.append(Worker(core, args, tile=Tile(c, 2), stack_size=self.STACK, while_true=False))

        n = T * E

        def groups(c: int) -> TensorAccessPattern:  # the core's groups (8 consecutive rows each)
            return TensorAccessPattern((n,), 8 * E * c, [1, n_g, 8 * E // 1024, 1024], [0, 8 * self.cols * E, 1024, 1])

        dp, n_d = self.d_pitch, T * COLS * self.d_pitch
        ed = COLS * dp

        def dgroups(c: int) -> TensorAccessPattern | list[TensorAccessPattern]:  # [group][8 tokens][column block][E / 8]
            if nd == 2:  # (both deltas of a group, [group][delta][8 tokens x column block][E / 8]: a chain of a tap per
                # delta, their stride may exceed a BD's)
                return [TensorAccessPattern((self.d_stride + n_d,), 8 * ed * c + k * self.d_stride,
                                            [n_g, 1, 8 * COLS, E // COLS], [8 * self.cols * ed, 0, dp, 1]) for k in range(2)]
            if dp == E // COLS:
                return groups(c)
            return TensorAccessPattern((n_d,), 8 * ed * c, [n_g, 8, COLS, E // COLS], [8 * self.cols * ed, ed, dp, 1])

        x_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
        d_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
        o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
        p_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]

        def seq(*a):
            if add:
                x, d, w, xo, xn, xh, ph, dh, oh = a
            else:
                x, w, xn, xh, ph = a
            tg = TaskGroup()
            for c in range(self.cols):
                # w as a group-shaped object (8 copies)
                xh[c].fill(w, TensorAccessPattern((E,), 0, [1, 8, E // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                xh[c].fill(x, groups(c), group=tg)
                if add:
                    dg = dgroups(c)
                    if isinstance(dg, list):
                        chained_fill(dh[c], d, dg, tg)
                    else:
                        dh[c].fill(d, dg, group=tg)
                    # x_out chunks [group][kc][8 tokens][CH]
                    oh[c].drain(xo, TensorAccessPattern((n,), 8 * E * c, [n_g, n_kc, 8, CH], [8 * self.cols * E, CH, E, 1]),
                                group=tg, wait=True)
                ph[c].drain(xn, TensorAccessPattern((n * 9 // 16,), c * (E // 8) * 36,
                                                    [1, n_g, 8, E // 64 * 36], [0, self.cols * (E // 8) * 36, E // 64 * 36, 1]),
                            group=tg, wait=True)
            tg.finish()

        vt = np.ndarray[(n,), bf]
        n_ds = (nd - 1) * self.d_stride + n_d
        types = [vt] + ([np.ndarray[(n_ds,), bf]] if add else []) + [np.ndarray[(E,), bf]] + ([vt] if add else [])
        types.append(np.ndarray[(n * 9 // 16,), bf])
        rt = Runtime(seq, [*types, x_h, p_h, *([d_h, o_h] if add else [])])
        return Program(NPU2(), rt, workers=workers).resolve_program()


    def _hold_design(self) -> Module:
        """deltas > 2: core c takes tokens [c T / cols, (c + 1) T / cols) in groups of HG, a group's x + d1 + .. + dn held
        in L1 between pass A (x chunks straight from the shim, the deltas' chunks re-laid by the memtile:
        [kc][delta][HG][HCH]) and pass B (half panels: the group's HG tokens of each panel's 8); the memtile joins two
        groups' half panels into whole panel rows (the shim writes long runs: scattered 36-byte writes ran at a fraction
        of the DDR rate); each byte read once."""
        T, E, p, nd, cols = self.T, self.E, self.p, self.deltas, self.cols
        tc = T // cols  # (a core's tokens)
        n_g, n_kc, n_w, n_r = tc // HG, E // HCH, E // (HG * HCH), tc // 8
        bf = np.dtype[bfloat16]
        chunk_ty = np.ndarray[(HG * HCH,), bf]
        dch_ty = np.ndarray[(nd * HG * HCH,), bf]
        pan_ty = np.ndarray[(HCH // 8 * 18,), bf]  # (HCH / 8 half panels of 36 B)
        row_ty = np.ndarray[(E // 8 * 36,), bf]  # (a panel row: 8 tokens)
        v_ty, w_ty = np.ndarray[(HG * E,), bf], np.ndarray[(E,), bf]
        st_ty = np.ndarray[(4 * 32 + 64,), np.dtype[np.float32]]
        k_w = Kernel(f"{p}_w", "kernel.o", [chunk_ty, w_ty, np.int32])
        k_a = Kernel(f"{p}_a", "kernel.o", [chunk_ty, dch_ty, v_ty, chunk_ty, st_ty, np.int32])
        k_b = Kernel(f"{p}_b", "kernel.o", [v_ty, w_ty, st_ty, pan_ty, np.int32])
        relay = stream_walk(nd * HG * E, [(n_kc, HCH), (nd, HG * E), (HG, E), (HCH, 1)])
        # (the memtile: a group's deltas double-buffered where they fit next to two panel rows)
        d_depth = 2 if 2 * nd * HG * E * 2 + 2 * E // 8 * 72 <= 500 * 1024 else 1

        def deltas(c: int) -> tuple[ObjectFifo, ObjectFifo]:  # (a group's deltas [delta][HG][E] -> chunks)
            l2 = ObjectFifo(np.ndarray[(nd * HG * E,), bf], name=f"pnhd{c}", depth=d_depth)
            return l2, l2.cons().forward(obj_type=dch_ty, name=f"pnhd1{c}", depth=2, tile=Tile(c, 1), to_stream=relay)

        def panels(c: int) -> tuple[ObjectFifo, ObjectFifo]:  # (two groups' half panels [half][panel][36 B] -> a row)
            rows = ObjectFifo(row_ty, name=f"pnhr{c}", depth=2)
            (halves,) = rows.prod().join([0], tile=Tile(c, 1), obj_types=[pan_ty], depths=[2], names=[f"pnhp{c}"],
                                         from_stream=[stream_walk(E // 8 * 36, [(2, 18), (E // 8, 36), (18, 1)])])
            return halves, rows

        xf = [ObjectFifo(chunk_ty, name=f"pnh_x{c}", depth=2) for c in range(cols)]
        df = [deltas(c) for c in range(cols)]
        of = [ObjectFifo(chunk_ty, name=f"pnh_o{c}", depth=2) for c in range(cols)]
        pf = [panels(c) for c in range(cols)]

        def core(x_f, d_f, o_f, p_f, kw, ka, kb, vb, wb, st):
            for _ in forever_():
                for j in range_(n_w):  # (w: the first objects of the x stream)
                    w = x_f.acquire(1)
                    kw(w, wb, j)
                    x_f.release(1)
                for _ in range_(n_g):
                    for kc in range_(n_kc):
                        x, d, xo = x_f.acquire(1), d_f.acquire(1), o_f.acquire(1)
                        ka(x, d, vb, xo, st, kc)
                        o_f.release(1)
                        d_f.release(1)
                        x_f.release(1)
                    for kc in range_(n_kc):
                        pn = p_f.acquire(1)
                        kb(vb, wb, st, pn, kc)
                        p_f.release(1)

        workers = [Worker(core, [xf[c].cons(), df[c][1].cons(), of[c].prod(), pf[c][0].prod(), k_w, k_a, k_b,
                                 Buffer(v_ty, name=f"pnh_v{c}"), Buffer(w_ty, name=f"pnh_w{c}"),
                                 Buffer(st_ty, name=f"pnh_st{c}")],
                          tile=Tile(c, 2), stack_size=self.STACK, while_true=False) for c in range(cols)]
        n, dp, n_d = T * E, self.d_pitch, T * COLS * self.d_pitch
        ed = COLS * dp

        def chunks(c: int) -> TensorAccessPattern:  # (core c's groups as [group][kc][HG tokens][HCH])
            return TensorAccessPattern((n,), tc * E * c, [n_g, n_kc, HG, HCH], [HG * E, HCH, E, 1])

        x_h = [f.prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
        d_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
        o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
        p_h = [f[1].cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]
        n_ds = (nd - 1) * self.d_stride + n_d
        rb = E // 8 * 36  # (a panel row's bf16; in runs of 1152)
        assert rb % 1152 == 0

        def seq(x, d, w, xo, xn, xh, dh, oh, ph):
            tg = TaskGroup()
            for c in range(cols):
                xh[c].fill(w, TensorAccessPattern((E,), 0, [1, 1, E // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                xh[c].fill(x, chunks(c), group=tg)
                # (per group a tap per delta: its HG rows of 8 column blocks, d_pitch apart)
                chained_fill(dh[c], d, [TensorAccessPattern((n_ds,), tc * ed * c + k * self.d_stride,
                                                            [n_g, 1, HG * COLS, E // COLS], [HG * ed, 0, dp, 1])
                                        for k in range(nd)], tg)
                oh[c].drain(xo, chunks(c), group=tg, wait=True)
                ph[c].drain(xn, TensorAccessPattern((n * 9 // 16,), n_r * rb * c, [1, n_r, rb // 1152, 1152],
                                                    [0, rb, 1152, 1]), group=tg, wait=True)
            tg.finish()

        vt = np.ndarray[(n,), bf]
        rt = Runtime(seq, [vt, np.ndarray[(n_ds,), bf], w_ty, vt, np.ndarray[(n * 9 // 16,), bf], x_h, d_h, o_h, p_h])
        return Program(NPU2(), rt, workers=workers).resolve_program()


MODES = ("in", "post", "plain")
RCH = 160  # columns per chunk (ResidualNormOperator: E / 16 for E = 2560)

_RSRC = JINJA_ENV.from_string(r"""
// workspace (bytes): wp [E] bf16 | w1 [E] bf16 | ss [64] fp32 | a hi, lo [64] bf16 | iv hi, lo [64] bf16 | s hi, lo [64] bf16
#define NW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// 8 rows (pitch P elements) x 8 consecutive elements -> 64 lanes (lane 8 t + e)
template <typename T, int P>
[[clang::always_inline]] inline aie::vector<T, 64> {{ p }}_gather8(const T *__restrict p)
{
    return aie::concat(aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + P)),
                       aie::concat(aie::load_v<8>(p + 2 * P), aie::load_v<8>(p + 3 * P)),
                       aie::concat(aie::load_v<8>(p + 4 * P), aie::load_v<8>(p + 5 * P)),
                       aie::concat(aie::load_v<8>(p + 6 * P), aie::load_v<8>(p + 7 * P)));
}

// 64 lanes -> 8 rows (pitch P) x 8 elements
template <int P>
[[clang::always_inline]] inline void {{ p }}_scatter8(float *__restrict p, aie::vector<float, 64> v)
{
{% for t in range(8) %}
    aie::store_v(p + {{ t }} * P, v.template extract<8>({{ t }}));
{% endfor %}
}

// x' of a panel: (x + d w a) s (module docstring), lanes 8 t + e; d2 ..: the other K parts (deltas > 1)
[[clang::always_inline]] inline aie::vector<float, 64> {{ p }}_xp(aie::vector<float, 64> x, aie::vector<bfloat16, 64> d,
{% for i in range(2, deltas + 1) %}
                                                               aie::vector<bfloat16, 64> d{{ i }},
{% endfor %}
                                                               aie::vector<bfloat16, 64> w,
                                                               aie::vector<bfloat16, 64> ah, aie::vector<bfloat16, 64> al,
                                                               aie::vector<bfloat16, 64> sh, aie::vector<bfloat16, 64> sl)
{
    aie::accum<accfloat, 64> u = aie::mul(d, w);
{% for i in range(2, deltas + 1) %}
    u = aie::mac(u, d{{ i }}, w);
{% endfor %}
    const aie::vector<float, 64> uf = u.template to_vector<float>();
    const aie::vector<bfloat16, 64> uh = aie::accum<accfloat, 64>(uf).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> ul = aie::accum<accfloat, 64>(
        aie::sub(uf, aie::accum<accfloat, 64>(uh).template to_vector<float>())).template to_vector<bfloat16>();
{% if scale %}
    const aie::vector<bfloat16, 64> xh = aie::accum<accfloat, 64>(x).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 64> xl = aie::accum<accfloat, 64>(
        aie::sub(x, aie::accum<accfloat, 64>(xh).template to_vector<float>())).template to_vector<bfloat16>();
    aie::accum<accfloat, 64> acc = aie::mac(aie::mac(aie::mul(xh, sh), xh, sl), xl, sh);
    acc = aie::mac(aie::mac(aie::mac(acc, uh, ah), uh, al), ul, ah);
{% else %}
    aie::accum<accfloat, 64> acc = aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 64>(x), uh, ah), uh, al), ul, ah);
{% endif %}
    return acc.template to_vector<float>();
}

// the 8 tokens' 1 / rms from the lanes' sums of squares (lane 8 t + e) -> hi, lo (lane 8 t + e) of 1 / rms times
// the scale (hi + lo; 1 / rms alone when sh = 1, sl = 0)
[[clang::always_inline]] inline void {{ p }}_inv(const float *__restrict ss, aie::vector<bfloat16, 64> sh,
                                                 aie::vector<bfloat16, 64> sl, bfloat16 *__restrict hi,
                                                 bfloat16 *__restrict lo)
{
    const aie::vector<float, 8> s8 = al::group8<false>(aie::load_v<64>(ss));
    aie::vector<bfloat16, 16> ih, il;
    al::split2<16>(al::inv_rms<{{ E }}, 8>(s8.template grow<16>(), {{ eps }}f), ih, il);
    // (v sh + v sl, v = vh + vl: the four products in one accumulator)
    const aie::vector<bfloat16, 64> vh = al::expand8(ih.template extract<8>(0)), vl = al::expand8(il.template extract<8>(0));
    const aie::vector<float, 64> b = aie::mac(aie::mac(aie::mac(aie::mul(vh, sh), vl, sh), vh, sl), vl, sl).template to_vector<float>();
    const aie::vector<bfloat16, 64> h = aie::accum<accfloat, 64>(b).template to_vector<bfloat16>();
    aie::store_v(hi, h);
    aie::store_v(lo, aie::accum<accfloat, 64>(aie::sub(b, aie::accum<accfloat, 64>(h).template to_vector<float>()))
                         .template to_vector<bfloat16>());
}

}  // namespace

extern "C" {

// the parameters' chunk kc ([8 rows][CH] bf16 of [w_post | w1 | s hi, lo])
void {{ p }}_prm(const bfloat16 *__restrict pr, bfloat16 *__restrict ws, int32_t kc)
{
    al::copy<{{ CH }}, 32>(NW(bfloat16, {{ w.wp }}) + {{ CH }} * kc, pr);
    al::copy<{{ CH }}, 32>(NW(bfloat16, {{ w.w1 }}) + {{ CH }} * kc, pr + {{ CH }});
    if (kc == 0) {
        aie::store_v(NW(bfloat16, {{ w.sh }}), aie::broadcast<bfloat16, 64>(pr[2 * {{ CH }}]));
        aie::store_v(NW(bfloat16, {{ w.sl }}), aie::broadcast<bfloat16, 64>(pr[2 * {{ CH }} + 1]));
    }
}

// pass A, chunk kc: the lanes' sums of squares of {{ "x" if mode == "in" else "d" }} (8 tokens x CH)
void {{ p }}_ss(const bfloat16 *__restrict v, bfloat16 *__restrict ws, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict ss = NW(float, {{ w.ss }});
    aie::accum<accfloat, 64> acc(aie::load_v<64>(ss));
    if (kc == 0)
        acc = aie::zeros<accfloat, 64>();
{% if mode == "in" %}
    const float *__restrict px = (const float *)v;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int kb = 0; kb < {{ CH // 8 }}; kb++) {
        acc = aie::mac_square(acc, aie::accum<accfloat, 64>({{ p }}_gather8<float, {{ CH }}>(px)).template to_vector<bfloat16>());
        px += 8;
    }
{% else %}
    const bfloat16 *__restrict pd = v;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int kb = 0; kb < {{ CH // 8 }}; kb++) {
{% if deltas > 1 %}
        aie::accum<accfloat, 64> s({{ p }}_gather8<bfloat16, {{ CH }}>(pd));
{% for i in range(1, deltas) %}
        s = aie::add(s, {{ p }}_gather8<bfloat16, {{ CH }}>(pd + {{ 8 * CH * i }}));
{% endfor %}
        acc = aie::mac_square(acc, s.template to_vector<bfloat16>());
{% else %}
        acc = aie::mac_square(acc, {{ p }}_gather8<bfloat16, {{ CH }}>(pd));
{% endif %}
        pd += 8;
    }
{% endif %}
    aie::store_v(ss, acc.template to_vector<float>());
}

{% if mode != "in" %}
// pass B, chunk kc: x' -> xo (8 tokens x CH fp32){{ ", its panels -> pan" if mode == "plain" else ", the lanes' sums of squares of bf16(x')" }}
void {{ p }}_upd(const bfloat16 *__restrict x_, const bfloat16 *__restrict d, bfloat16 *__restrict ws,
                 bfloat16 *__restrict xo_{{ ", bfloat16 *__restrict pan" if mode == "plain" }}, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict ss = NW(float, {{ w.ss }});
    const aie::vector<bfloat16, 64> sh = aie::load_v<64>(NW(bfloat16, {{ w.sh }})), sl = aie::load_v<64>(NW(bfloat16, {{ w.sl }}));
    if (kc == 0)  // (a = s / rms(d) from pass A's sums)
        {{ p }}_inv(ss, sh, sl, NW(bfloat16, {{ w.ah }}), NW(bfloat16, {{ w.al }}));
    const aie::vector<bfloat16, 64> ah = aie::load_v<64>(NW(bfloat16, {{ w.ah }})), al_ = aie::load_v<64>(NW(bfloat16, {{ w.al }}));
{% if mode == "post" %}
    aie::accum<accfloat, 64> acc(aie::load_v<64>(ss));
    if (kc == 0)
        acc = aie::zeros<accfloat, 64>();
{% else %}
    al::BOut<> o((bfp16ebs8 *)pan);
{% endif %}
    const float *__restrict px = (const float *)x_;
    float *__restrict xo = (float *)xo_;
    const bfloat16 *__restrict pw = NW(bfloat16, {{ w.wp }}) + {{ CH }} * kc;
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ CH // 8 }}; kb++) {
        const aie::vector<float, 64> xp = {{ p }}_xp({{ p }}_gather8<float, {{ CH }}>(px), {{ p }}_gather8<bfloat16, {{ CH }}>(d + 8 * kb),
{% for i in range(1, deltas) %}
                                                    {{ p }}_gather8<bfloat16, {{ CH }}>(d + {{ 8 * CH * i }} + 8 * kb),
{% endfor %}
                                                    aie::load_v<8>(pw).template grow_replicate<64>(), ah, al_, sh, sl);
        {{ p }}_scatter8<{{ CH }}>(xo, xp);
{% if mode == "post" %}
        acc = aie::mac_square(acc, aie::accum<accfloat, 64>(xp).template to_vector<bfloat16>());
{% else %}
        o.push(aie::accum<accfloat, 64>(xp).template to_vector<bfp16ebs8>());
{% endif %}
        px += 8;
        xo += 8;
        pw += 8;
    }
{% if mode == "post" %}
    aie::store_v(ss, acc.template to_vector<float>());
{% endif %}
}
{% endif %}

{% if mode != "plain" %}
// pass C, chunk kc: the panels of bf16(x') w1 / rms(x'){{ " (x' recomputed: pass B's arithmetic)" if mode == "post" }}
void {{ p }}_pan(const bfloat16 *__restrict x_, {{ "const bfloat16 *__restrict d, " if mode == "post" }}bfloat16 *__restrict ws,
                 bfloat16 *__restrict pan, int32_t kc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    if (kc == 0) {  // (1 / rms(x') from the sums of squares)
        const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>(1.0f);
        {{ p }}_inv(NW(float, {{ w.ss }}), one, aie::zeros<bfloat16, 64>(), NW(bfloat16, {{ w.ih }}), NW(bfloat16, {{ w.il }}));
    }
    const aie::vector<bfloat16, 64> ih = aie::load_v<64>(NW(bfloat16, {{ w.ih }})), il = aie::load_v<64>(NW(bfloat16, {{ w.il }}));
{% if mode == "post" %}
    const aie::vector<bfloat16, 64> sh = aie::load_v<64>(NW(bfloat16, {{ w.sh }})), sl = aie::load_v<64>(NW(bfloat16, {{ w.sl }}));
    const aie::vector<bfloat16, 64> ah = aie::load_v<64>(NW(bfloat16, {{ w.ah }})), al_ = aie::load_v<64>(NW(bfloat16, {{ w.al }}));
    const bfloat16 *__restrict pw = NW(bfloat16, {{ w.wp }}) + {{ CH }} * kc;
{% endif %}
    const bfloat16 *__restrict p1 = NW(bfloat16, {{ w.w1 }}) + {{ CH }} * kc;
    const float *__restrict px = (const float *)x_;
    al::BOut<> o((bfp16ebs8 *)pan);
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < {{ CH // 8 }}; kb++) {
{% if mode == "post" %}
        const aie::vector<float, 64> xp = {{ p }}_xp({{ p }}_gather8<float, {{ CH }}>(px), {{ p }}_gather8<bfloat16, {{ CH }}>(d + 8 * kb),
{% for i in range(1, deltas) %}
                                                    {{ p }}_gather8<bfloat16, {{ CH }}>(d + {{ 8 * CH * i }} + 8 * kb),
{% endfor %}
                                                    aie::load_v<8>(pw).template grow_replicate<64>(), ah, al_, sh, sl);
        pw += 8;
{% else %}
        const aie::vector<float, 64> xp = {{ p }}_gather8<float, {{ CH }}>(px);
{% endif %}
        const aie::vector<bfloat16, 64> xw = aie::mul(aie::accum<accfloat, 64>(xp).template to_vector<bfloat16>(),
                                                      aie::load_v<8>(p1).template grow_replicate<64>()).template to_vector<bfloat16>();
        o.push(aie::mac(aie::mul(xw, ih), xw, il).template to_vector<bfp16ebs8>());
        px += 8;
        p1 += 8;
    }
}
{% endif %}

}  // extern "C"
""")


def residual_params(w_post: np.ndarray | None, w1: np.ndarray | None, scale: float, E: int) -> np.ndarray:
    """A norm's parameter object ([8 rows][E] bf16 as uint16: row 0 w_post, row 1 w1, row 2 [s hi, s lo])."""
    out = np.zeros((8, E), np.uint16)

    def b16(v: np.ndarray) -> np.ndarray:
        return torch.from_numpy(np.asarray(v, np.float32)).bfloat16().view(torch.int16).numpy().view(np.uint16)

    if w_post is not None:
        out[0] = b16(w_post)
    if w1 is not None:
        out[1] = b16(w1)
    hi = bf16_value(scale)
    out[2, :2] = b16(np.array([hi, scale - hi]))
    return out


class ResidualNormOperator(ArtcOperator):
    """The residual update and norm of T tokens (module docstring). Runtime sequence: ``in``: (P, X, XN); ``post`` /
    ``plain``: (P, X, D, XO, XN), XO = X in place. P: the parameters (``params``), X / XO: (T, E) fp32 (as bf16
    pairs), D: the deltas (2 T 8 d_pitch per delta, ``d_stride`` apart), XN: panels."""

    STACK = 1024
    p = kernel_prefix("gn")

    def __init__(self, *, T: int, E: int, mode: str, eps: float, d_pitch: int = 0, deltas: int = 1, d_stride: int = 0,
                 scale: bool = False, cols: int = COLS):
        """``scale``: x' = (...) s with s from the parameters (the layer's output scale), else s = 1 (cheaper)."""
        assert mode in MODES and E % RCH == 0 and E % 128 == 0, (mode, E)
        assert T % 8 == 0 and T >= 8 * cols and 1 <= cols <= COLS and deltas in (1, 2, 4)
        assert mode == "post" or not scale
        self.T, self.E, self.mode, self.eps, self.cols, self.scale = T, E, mode, eps, cols, scale
        self.d_pitch = d_pitch or E // COLS
        assert self.d_pitch >= E // COLS
        self.deltas = deltas if mode != "in" else 0
        self.d_stride = d_stride or T * COLS * self.d_pitch
        super().__init__()

    def workspace(self) -> dict[str, int]:
        E = self.E
        sizes = [("wp", 2 * E), ("w1", 2 * E), ("ss", 256), ("ah", 128), ("al", 128), ("ih", 128), ("il", 128),
                 ("sh", 128), ("sl", 128)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        return _RSRC.render(p=self.p, E=self.E, CH=RCH, eps=repr(self.eps), mode=self.mode, deltas=self.deltas,
                           scale=self.scale, w=Attr(self.workspace()))

    @property
    def n_delta(self) -> int:
        return (self.deltas - 1) * self.d_stride + self.T * COLS * self.d_pitch

    def get_arg_spec(self) -> list[ArgSpec]:
        T, E = self.T, self.E
        specs = [ArgSpec("in", (8 * E,), torch.bfloat16), ArgSpec("in", (2 * T * E,), torch.bfloat16)]
        if self.mode != "in":
            specs += [ArgSpec("in", (self.n_delta,), torch.bfloat16), ArgSpec("out", (2 * T * E,), torch.bfloat16)]
        specs.append(ArgSpec("out", (T * E * 9 // 16,), torch.bfloat16))
        return specs

    def get_mlir_artifact(self) -> Module:
        T, E, p, mode, nd = self.T, self.E, self.p, self.mode, max(self.deltas, 1)
        n_kc = E // RCH
        n_gs = [len(range(c, T // 8, self.cols)) for c in range(self.cols)]  # (core c: token groups c, c + cols, ..)
        bf = np.dtype[bfloat16]
        xc_ty = np.ndarray[(2 * 8 * RCH,), bf]  # a chunk of x (8 tokens x RCH fp32)
        dc_ty = np.ndarray[(nd * 8 * RCH,), bf]  # a chunk of d ([delta][8 tokens][RCH] bf16) or of the parameters
        pan_ty = np.ndarray[(RCH // 8 * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 2,), bf]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [dc_ty, ws_ty, np.int32])
        k_ss = Kernel(f"{p}_ss", "kernel.o", [xc_ty if mode == "in" else dc_ty, ws_ty, np.int32])
        k_upd = Kernel(f"{p}_upd", "kernel.o", [xc_ty, dc_ty, ws_ty, xc_ty] + ([pan_ty] if mode == "plain" else [])
                       + [np.int32]) if mode != "in" else None
        k_pan = Kernel(f"{p}_pan", "kernel.o", [xc_ty] + ([dc_ty] if mode == "post" else []) + [ws_ty, pan_ty, np.int32]) \
            if mode != "plain" else None
        # replays per group: (x, d)
        x_rep, d_rep = {"in": (2, 1), "post": (2, 3), "plain": (1, 2)}[mode]

        def replayed(name: str, c: int, n: int, k: int, rep: int) -> tuple[ObjectFifo, ObjectFifo]:
            """A group object of k blocks of [8 tokens][n] through the memtile, sent ``rep`` times as chunks
            [kc][block k][8 tokens][n / n_kc]."""
            ch = n // n_kc
            l2 = ObjectFifo(np.ndarray[(k * 8 * n,), bf], name=f"{name}{c}", depth=2)
            walk = stream_walk(k * 8 * n, [(n_kc, ch), (k, 8 * n), (8, n), (ch, 1)])
            return l2, l2.cons().forward(obj_type=np.ndarray[(k * 8 * ch,), bf], name=f"{name}1{c}", depth=2,
                                          tile=Tile(c, 1), to_stream=walk, repeat_count=rep if rep > 1 else None)

        xf = [replayed("gnx", c, 2 * E, 1, x_rep) for c in range(self.cols)]
        df = [replayed("gnd", c, E, nd, d_rep) for c in range(self.cols)]
        of = [ObjectFifo(xc_ty, name=f"gno{c}", depth=2) for c in range(self.cols)] if mode != "in" else []
        pf = [ObjectFifo(pan_ty, name=f"gnp{c}", depth=2) for c in range(self.cols)]

        def core(x_f, d_f, p_f, prm, ss, ws, *rest):
            rest = list(rest)
            n_g = rest.pop()
            upd = rest.pop(0) if mode != "in" else None
            pan = rest.pop(0) if mode != "plain" else None
            o_f = rest.pop(0) if mode != "in" else None
            for _ in forever_():
                for rep in range(d_rep):  # (the parameters: d's first object, as often as d's groups are replayed)
                    for kc in range_(n_kc):
                        d = d_f.acquire(1)
                        if rep == 0:
                            prm(d, ws, kc)
                        d_f.release(1)
                for _ in range_(n_g):
                    a_f = x_f if mode == "in" else d_f
                    for kc in range_(n_kc):  # pass A
                        v = a_f.acquire(1)
                        ss(v, ws, kc)
                        a_f.release(1)
                    if mode != "in":
                        for kc in range_(n_kc):  # pass B
                            x = x_f.acquire(1)
                            d = d_f.acquire(1)
                            xo = o_f.acquire(1)
                            if mode == "plain":
                                pn = p_f.acquire(1)
                                upd(x, d, ws, xo, pn, kc)
                                p_f.release(1)
                            else:
                                upd(x, d, ws, xo, kc)
                            o_f.release(1)
                            d_f.release(1)
                            x_f.release(1)
                    if mode != "plain":
                        for kc in range_(n_kc):  # pass C
                            x = x_f.acquire(1)
                            pn = p_f.acquire(1)
                            if mode == "post":
                                d = d_f.acquire(1)
                                pan(x, d, ws, pn, kc)
                                d_f.release(1)
                            else:
                                pan(x, ws, pn, kc)
                            p_f.release(1)
                            x_f.release(1)

        workers = []
        for c in range(self.cols):
            args = [xf[c][1].cons(), df[c][1].cons(), pf[c].prod(), k_prm, k_ss, Buffer(ws_ty, name=f"gn_ws{c}")]
            args += [k for k in (k_upd, k_pan) if k is not None]
            if mode != "in":
                args.append(of[c].prod())
            workers.append(Worker(core, [*args, n_gs[c]], tile=Tile(c, 2), stack_size=self.STACK,
                                  dynamic_objfifo_lowering=True, while_true=False))

        n = 2 * T * E  # (x in bf16 units)
        dp, ed = self.d_pitch, COLS * self.d_pitch

        x_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(xf)]
        d_h = [f[0].prod(tile=Tile(c, 0)) for c, f in enumerate(df)]
        o_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]
        p_h = [f.cons(tile=Tile(c, 0)) for c, f in enumerate(pf)]
        cols = self.cols

        def seq(*a):
            if mode == "in":
                P, X, XN, xh, dh, ph = a
            else:
                P, X, D, XO, XN, xh, dh, ph, oh = a
            tg = TaskGroup()
            for c in range(cols):
                n_g = n_gs[c]
                # the parameters as d's first object (read nd times: an object of nd blocks); rows of <= 512 words
                dh[c].fill(P, TensorAccessPattern((8 * E,), 0, [1, nd, 8 * E // 1024, 1024], [0, 0, 1024, 1]), group=tg)
                # x: the core's groups, 8 rows of E fp32 each
                xh[c].fill(X, TensorAccessPattern((n,), 16 * E * c, [1, n_g, 16 * E // 1024, 1024],
                                                  [0, 16 * cols * E, 1024, 1]), group=tg)
                if mode != "in":
                    # d: [group][delta][8 tokens x 8 column blocks][E / 8] (row stride 8 d_pitch = 8 blocks)
                    dh[c].fill(D, TensorAccessPattern((self.n_delta,), 8 * ed * c, [n_g, nd, 8 * COLS, E // COLS],
                                                      [8 * cols * ed, self.d_stride, dp, 1]), group=tg)
                    oh[c].drain(XO, TensorAccessPattern((n,), 16 * E * c, [n_g, n_kc, 8, 2 * RCH],
                                                        [16 * cols * E, 2 * RCH, 2 * E, 1]), group=tg, wait=True)
                ph[c].drain(XN, TensorAccessPattern((T * E * 9 // 16,), c * (E // 8) * 36,
                                                    [1, n_g, 8, E // 64 * 36], [0, cols * (E // 8) * 36, E // 64 * 36, 1]),
                            group=tg, wait=True)
            tg.finish()

        types = [np.ndarray[(8 * E,), bf], np.ndarray[(n,), bf]]
        if mode != "in":
            types += [np.ndarray[(self.n_delta,), bf], np.ndarray[(n,), bf]]
        types.append(np.ndarray[(T * E * 9 // 16,), bf])
        rt = Runtime(seq, [*types, x_h, d_h, p_h, *([o_h] if mode != "in" else [])])
        return Program(NPU2(), rt, workers=workers).resolve_program()
