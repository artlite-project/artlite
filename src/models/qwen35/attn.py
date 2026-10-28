"""The attention core of the multi-slot decode overlay (``qwen35/decode.py``): the full-attention layers (head_dim 256,
NeoX RoPE on the first 64 dims, q / k RMS-normed per head, a sigmoid output gate) for S <= 4 token slots at positions
p .. p + S - 1.

The KV heads' columns form chains of NPH = 8 / KV heads (9B / 27B NPH 2, MiniCPM-V 4): column c attends with its QC
query heads (27B 3, 9B 2, MiniCPM-V 1) to KV head c // NPH over the whole cache (``decode.kv``'s layout: groups of 32
positions, two K objects and a V object). A step's attention runs in sub-steps (``SUB`` slots of
one group at most, ``decode.vring.substeps``) whose S x QC query rows (row s QC + h: slot s, head h) go 4 to a set of
MMUL panels ([rows hi | rows lo] per 8 dims). Per sub-step the core gets:

- from the X stream (the others skip them: ``M_SKIP``): the ``F_ATT`` object [q_norm, k_norm (256 fp32 each) | cos, sin
  per slot of the step (4 x 64 fp32)], the slots' q objects ([q | gate] x QC heads, bf16, as attn_in drained them),
  then the slots' raw K, V objects ([K 256 | V 256] per KV head, bf16);
- from the A stream (broadcast from the main core's, whose objects it skips): the cache of its head up to the
  sub-step's group, then (a core that attends to that group) the tail objects (``M_NASKIP`` objects in all).

It inserts the slots' K (normed, rotated, bf16 hi + lo blocks) into their records (in its copy), scores every record
against all sets at once (log2 units: Q x log2(e) / 16; each K panel popped once for every set), masks each row's
positions past its own, softmax with a running maximum per row (masked positions add exactly nothing; rescaling y and l
once per group, a row whose maximum stayed keeps them exactly: ``al::merge``), y += P V over octets (two sets' 8 rows,
P hi and P lo as two MMULs into one accumulator: a record's P V the same roundings alone or with its pair, each V panel
popped once for every octet). A group's softmax (maxima, rescale, P panels) runs after its second K object is released,
while the V object streams into the core's one A buffer (``asoft``): the scores need no cache bytes. What a position
sees of a V record is its panels
encoded from the exact values of its positions up to it (``kernel_lib.h``'s V records): the open records' values come
from the ring of the layer's KV head (``decode.vring``; or after a prefill the record's V panels: ``M_ATREC``) as the
tail objects, after the cache; the core puts the slots' V in and per slot encodes the record up to the slot and adds
its P V. Its outputs go to the main core (shared-memory FIFO R, ``C_LEN`` values per object), which relays them to C:
o = y / l * sigmoid(gate) ([head][k block: 8 dims][slot S][8] bf16: into the sub-step's slots of the attn_out phase's
input), then the records of the sub-step's first and last positions (the same twice when one record holds both): a
chain's first column their K records, its last one their V panels up to the last position and their rows (the ring),
the columns between none (``A_ROLE``).

A sub-step from 4 groups on (``AttnShape.splits``: the same for every position of its group) reads each head's cache
once per chain instead of once per column: every core of a chain takes the chain's NPH QC heads (the chain's q objects
in column order on X) and a run of the groups (``AttnShape.bounds``): the first core the groups 0 .. ``M_GATT`` - 1
(then it reads the sub-step's group from ``M_GJ`` on only to insert its K records), the last one the groups up to the
sub-step's. The partials (m, l, y) merge west to east: a core's partial stays in its workspace until its east
neighbour (shared memory) has merged it into its own, and the last core emits o for the chain's heads (a token
ObjectFifo orders each neighbour pair). So a slot's attention is the same in every step it can be part of: a step of
S slots gives each the numbers of a step of one.

Time (the 27B, one layer, 32K positions, a split sub-step of the chain's 2 cores): 2.12 ms for one slot and 2.14 for
two, the DMA floor of the cache's 9280-byte objects (2.11 with the cores' work compiled out: every column's share of
the DRAM port); a step of 3-4 slots runs two sub-steps. More slots per sub-step need L1 for their rows: per 4 rows ~7
KB (y, l fp32, the Q panels, scores, maxima, gates), the 27B's core holds 12.
"""

import math

import numpy as np

from artlite.decode.kv import Records
from artlite.model_config import QwenConfig
from artlite.operators.base import JINJA_ENV
from artlite.operators.design import COLS
from artlite.operators.util import Attr, bf16_hilo, regions

HD = 256  # head_dim
ND = HD // 8  # d-blocks (panels per record part)
REC = Records(HD)
REC_K, REC_V, ROWS_B = REC.k_bytes, REC.v_bytes, REC.rows_bytes  # bytes of a K record, a V record, its rows (a ring slot)
POS_PER_GROUP = 32  # 4 records: two K objects and a V object
OBJ_BYTES = 9280  # the overlay's objects: a weight tile (1024 columns: 16 rows of Q4_0, Q4_K) and a cache object
ROT = 32  # rotated pairs (dims i, i + 32 for i < 32)
F_ATT = 1 << 10  # M_FLAGS of an attention core's object
SMAX = 4
SUB = 2  # slots of a sub-step at most (``substeps``; a split one: ``AttnShape.split_sub``)
U = 3  # sub-steps of a step at most (``AttnShape.u``)
MAX_SPLIT_ROWS = 12  # a split sub-step's query rows at most (the workspace's sets)


# metadata words of an F_ATT object: decode.py's M_NS (slots), M_SKIP (X objects after it), M_NASKIP (the cache's A
# objects and the tail objects), M_NREL (R objects to relay), M_POS (the first slot's position), M_SA (its slot in the
# step), M_ATREC (record mode: the first tail object is the open record's V panels), M_NTAIL (tail objects)

# state words of the attention core (the first ones are trip counts of its program)
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NQ, A_NQP, A_NKV, A_NOBJ, A_NE, A_TP, A_TC, A_S, A_P, A_I, A_EIDX, A_ROLE, A_QS,
 A_KS, A_XL, A_QR, A_G0, A_GATT, A_GJ, A_NO, A_SA, A_TREC, A_NTAIL, A_SOFT) = range(28)  # (A_SOFT: ``asoft``'s group + 1)
A_WORDS = 32
ROLE_K, ROLE_V, ROLE_O = 0, 1, 2  # A_ROLE: a chain's first core (its K records), its last (V panels, rows), between

_SRC = JINJA_ENV.from_string(r"""
// The attention core's workspace (offsets in bytes: AttnShape.workspace())
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// 256 bf16 x -> fp32 out = x / rms(x) * w, then the NeoX rotation of dims 0 .. 63 (cs, sn: 32 fp32 each)
__attribute__((noinline)) void {{ p }}_norm_rope(const bfloat16 *__restrict x, const float *__restrict w,
                                                 const float *__restrict cs, const float *__restrict sn,
                                                 float *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    for (int i = 0; i < {{ HD }}; i += 64)
        ss = aie::mac_square(ss, aie::load_v<64>(x + i));
    const float inv = al::inv_rms<{{ HD }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    const aie::vector<float, 32> iv = aie::broadcast<float, 32>(inv);
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ HD }}; i += 32) {
        const aie::vector<float, 32> xf = aie::accum<accfloat, 32>(aie::load_v<32>(x + i)).template to_vector<float>();
        aie::store_v(out + i, al::mulff<32>(al::mulff<32>(xf, iv), aie::load_v<32>(w + i)));
    }
    const aie::vector<float, 32> a = aie::load_v<32>(out), b = aie::load_v<32>(out + 32);
    const aie::vector<float, 32> c = aie::load_v<32>(cs), s = aie::load_v<32>(sn);
    aie::store_v(out, aie::sub(al::mulff<32>(a, c), al::mulff<32>(b, s)));
    aie::store_v(out + 32, aie::add(al::mulff<32>(a, s), al::mulff<32>(b, c)));
}

// Scores of one record (8 positions) against NS sets: so[128 s + 8 r + j] (row r < 4 of set s, position j), log2
// units. Each K panel is popped once for all sets (Q: [d-block][set of NS]); K hi, then K lo (a loop of three panel
// streams spills their state), one MMUL chain per set: a row's scores the same sums whatever NS
template <int NS>
__attribute__((noinline)) void {{ p }}_qk(const uint8_t *__restrict k, const uint8_t *__restrict qa, float *__restrict so)
{
    al::MM8 a[NS];
    AIE_LOOP_UNROLL_FULL
    for (int s = 0; s < NS; s++)
        a[s] = al::MM8(aie::zeros<accfloat, 64>());
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
    AIE_LOOP_UNROLL_FULL
    for (int half = 0; half < 2; half++) {
        al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int d = 0; d < {{ ND }}; d++) {
            const al::BV K = ks.pop();
            AIE_LOOP_UNROLL_FULL
            for (int s = 0; s < NS; s++)
                a[s].mac(qs.pop(), aie::op_transpose(K));
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int s = 0; s < NS; s++) {  // (q rows hi + lo)
        const aie::vector<float, 64> c = a[s].to_accum().template to_vector<float>();
        aie::store_v(so + 128 * s, aie::add(c.template extract<32>(0), c.template extract<32>(1)));
    }
}

{# 2^x of 32 lanes: the rescale and merge factors, the gate's sigmoid (exactly 0 from -127 on); P's (bias 192:
   pipelines, masked scores -16384) #}
{% set exp2, e192 = km.exp2(32, below="zero"), km.exp2(64, bias=192, below="bounded", inline=True) %}
{{ exp2.src }}
{{ e192.src }}

// y += P V of records v0, v1 (their first V panels; a lone record: v1 any, its P 0) for each octet's 8 rows (sets 2 o,
// 2 o + 1; HALF: the last octet set 2 o alone): per record two MMULs into one accumulator, [y_2o | y_2o+1] (each set's
// [db][32] of y), P hi then P lo: per record the same roundings whichever records go together (y + 0 is y). pp: per
// octet [P0 hi | P0 lo | P1 hi | P1 lo]; y in place (yl, ys the same buffer as distinct restrict parameters, each block
// read before it is written)
__attribute__((noinline)) void {{ p }}_pv(const uint8_t *__restrict v0, const uint8_t *__restrict v1,
                                          const uint8_t *__restrict pp, const float __aie_dm_resource_c *__restrict yl,
                                          float __aie_dm_resource_d *__restrict ys)
{
    al::BIn<> ps((const bfp16ebs8 *)pp);
    al::BV Ah[{{ NO }}], Al[{{ NO }}], Bh[{{ NO }}], Bl[{{ NO }}];
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < {{ NO }}; o++) {
        Ah[o] = ps.pop();
        Al[o] = ps.pop();
        Bh[o] = ps.pop();
        Bl[o] = ps.pop();
    }
    al::BInR<aie_dm_resource::a> va((const bfp16ebs8 *)v0);  // (after the P pops: their streams' state not live meanwhile)
    al::BInR<aie_dm_resource::b> vb((const bfp16ebs8 *)v1);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV A = va.pop(), B = vb.pop();
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < {{ NO }}; o++) {
            const bool half = {{ "true" if HALF else "false" }} && o == {{ NO - 1 }};
            al::MM8 c(aie::accum<accfloat, 64>(aie::concat(aie::load_v<32>(yl + {{ 64 * (ND + 1) }} * o),
                                                           half ? aie::zeros<float, 32>() : aie::load_v<32>(yl + {{ 64 * (ND + 1) }} * o + {{ 32 * (ND + 1) }}))));
            c.mac(Ah[o], aie::op_transpose(A));
            c.mac(Al[o], aie::op_transpose(A));
            c.mac(Bh[o], aie::op_transpose(B));
            c.mac(Bl[o], aie::op_transpose(B));
            const aie::vector<float, 64> r = c.to_accum().template to_vector<float>();
            aie::store_v(ys + {{ 64 * (ND + 1) }} * o, r.template extract<32>(0));
            if (!half)
                aie::store_v(ys + {{ 64 * (ND + 1) }} * o + {{ 32 * (ND + 1) }}, r.template extract<32>(1));
        }
        yl += 32;
        ys += 32;
    }
}

// The P panels of a record (k: its index in the group's 4) for octet o's rows (sets 2 o, 2 o + 1; the last octet of
// an odd count set 2 o alone; r0, r1: the two sets' lanes to keep): P = 2^(s - m) (s the record's scores, m the
// running maxima), 0 below m - 63.5 (the masked scores, -16384, included), l += P, pp = [P hi | P lo] (bfp16 of the 8
// rows' P, of its residual)
__attribute__((noinline)) void {{ p }}_psoft(int32_t *__restrict ws, int k, int o, uint32_t r0, uint32_t r1,
                                             uint8_t *__restrict pp)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict s = AW(float, {{ a.sc }}) + 256 * o + 32 * k;
    const float *__restrict m = AW(float, {{ a.ml }}) + 128 * o;
    float *__restrict l = AW(float, {{ a.y }}) + {{ 64 * (ND + 1) }} * o + {{ 32 * ND }};
    const aie::vector<float, 32> c = aie::broadcast<float, 32>(192.0f);
    const aie::accum<accfloat, 64> A = aie::concat(aie::add(aie::accum<accfloat, 32>(aie::load_v<32>(s)), aie::sub(c, aie::load_v<32>(m))),
                                                   aie::add(aie::accum<accfloat, 32>(aie::load_v<32>(s + 128)), aie::sub(c, aie::load_v<32>(m + 64))));
    const aie::mask<64> keep = aie::mask<64>::from_uint64((uint64_t)r1 << 32 | r0) &
                               aie::ge(A.template to_vector<bfloat16>(), aie::broadcast<bfloat16, 64>(129.0f));
    const aie::vector<float, 64> P = aie::select(aie::zeros<float, 64>(), {{ e192 }}(A).template to_vector<float>(), keep);
    aie::store_v(l, aie::add(aie::load_v<32>(l), P.template extract<32>(0)));
    if ({{ "o < " ~ (NO - 1) if HALF else "true" }})
        aie::store_v(l + {{ 32 * (ND + 1) }}, aie::add(aie::load_v<32>(l + {{ 32 * (ND + 1) }}), P.template extract<32>(1)));
    al::BV hi, lo;
    al::split_hilo(aie::accum<accfloat, 64>(P), al::neg_eye(), hi, lo);
    al::BOut<> po((bfp16ebs8 *)pp);
    po.push(hi);
    po.push(lo);
}

// The lanes of query rows lo .. hi - 1 in set q (8 per row)
[[clang::always_inline]] inline uint32_t {{ p }}_rowmask(int q, int lo, int hi)
{
    const int a = lo - 4 * q < 0 ? 0 : lo - 4 * q, b = hi - 4 * q > 4 ? 4 : hi - 4 * q;
    return a >= b ? 0u : (b == 4 ? ~0u : (1u << (8 * b)) - 1u) & ~((1u << (8 * a)) - 1u);
}

// y, l of a set (ND + 1 blocks) = yi + yi (a0 - 1) + yw a1 (al::merge: a row whose factor is 1 keeps its values)
__attribute__((noinline)) void {{ p }}_mrg(const float __aie_dm_resource_a *__restrict yi, float *__restrict y,
                                           const float __aie_dm_resource_b *__restrict yw, aie::vector<float, 32> a0,
                                           aie::vector<float, 32> a1)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::merge(yi, y, yw, a0, a1, {{ ND + 1 }});
}

// The position's K blocks into slot j0 of its record's panels (hi panels, then lo), in place
__attribute__((noinline, minsize)) void {{ p }}_put_k(uint8_t *__restrict rec, const uint8_t *__restrict kb, int j0)
{
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < {{ 2 * ND }}; db++) {
        uint8_t *__restrict d = rec + 72 * db + 9 * j0;
        const uint8_t *__restrict s = kb + 9 * db;
        AIE_LOOP_NO_UNROLL
        for (int b = 0; b < 9; b++)
            d[b] = s[b];
    }
}

// A record's V panels -> its rows [d-block 32][position 8][8] bf16 (al::v_rows)
__attribute__((noinline, minsize)) void {{ p }}_vrows(const uint8_t *__restrict rec, bfloat16 *__restrict rows)
{
    al::v_rows(rec, rows, {{ ND }});
}

// A position's V (src: 256 bf16) into its row of a record's rows (dst: the row's first 8 values)
__attribute__((noinline)) void {{ p }}_rowput(bfloat16 *__restrict dst, const bfloat16 *__restrict src)
{
    al::v_put(dst, src, {{ ND }});
}

// rows, those from position n on as zeros -> the record's V panels (al::v_enc)
__attribute__((noinline)) void {{ p }}_venc(const bfloat16 *__restrict rows, uint8_t *__restrict rec, int n)
{
    al::v_enc(rows, rec, {{ ND }}, n);
}

// The rows of a set whose positions past their own are in record r (scores masked): lanes of row q (q < 4: query row 4
// set + q, slot (4 set + q) / QR at position p + slot) at positions 8 r + j > p + slot
__attribute__((noinline, minsize)) uint32_t {{ p }}_mask(int set, int r, int p, int S, int QR)
{
    uint32_t m = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < 4; q++) {
        const int row = 4 * set + q, s = QR == {{ QC }} ? al::div_small<{{ QC }}>(row) : al::div_small<{{ NPH * QC }}>(row);
        int n = p + s - 8 * r + 1;  // valid positions of the record for the row
        n = n < 0 ? 0 : (n > 8 ? 8 : n);
        if (s < S)
            m |= ((0xFFu << n) & 0xFFu) << (8 * q);
    }
    return m;
}

// Tail object t of a column that attends to the newest group, after its range: the open record r = (t ? p + S - 1 :
// p) >> 3 as rows ([d-block 32][position 8][8] bf16): its ring slot (in place), or (A_TREC, t = 0) its V panels as rows
// in qa; t = 1 of a step in one record: nothing more (the emission repeats the first's). The step's V rows in (the
// rows past its last position: as they were, read by no one). Per slot that reaches the record: the record's V panels
// from its rows up to the slot's position (into kb: what that position sees of the record, whatever the step), y += P
// V of the slot's rows. A chain's last column then keeps the last panels and the rows for its emission ([V | rows] per
// record from rec on).
__attribute__((noinline, minsize)) void {{ p }}_tobj(uint8_t *__restrict a, int32_t *__restrict ws, int t)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}], r0 = p >> 3, r1 = (p + S - 1) >> 3;
    if (t && r1 == r0)
        return;
    const int P0 = 8 * (t ? r1 : r0), last = p + S - 1 - P0 < 7 ? p + S - 1 - P0 : 7;  // (the step's last row of the record)
    bfloat16 *__restrict rows = st[{{ A_TREC }}] && !t ? AW(bfloat16, {{ a.qa }}) : (bfloat16 *)a;
    if (st[{{ A_TREC }}] && !t)
        {{ p }}_vrows(a, rows);
    AIE_LOOP_NO_UNROLL
    for (int j = P0 > p ? 0 : p - P0; j <= last; j++)  // (row j: slot P0 + j - p)
        {{ p }}_rowput(rows + 8 * j, AW(bfloat16, {{ a.vn }}) + {{ HD }} * (P0 + j - p));
    uint8_t *__restrict v = AW(uint8_t, {{ a.kb }});  // (the step's K blocks are in)
    float *__restrict y = AW(float, {{ a.y }});
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    AIE_LOOP_NO_UNROLL
    for (int s = -1; s < S; s++) {  // (s = -1: the second record's P panels 0, once)
        const int j = p + s - P0;
        if (s >= 0 && j < 0)  // (a slot before the record)
            continue;
        if (s >= 0)
            {{ p }}_venc(rows, v, j < 7 ? j + 1 : 8);
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < {{ NO }}; o++)  // (the slot's rows: P 0 elsewhere, their y and l as they are)
            {{ p }}_psoft(ws, (P0 >> 3) & 3, o, s < 0 ? 0u : {{ p }}_rowmask(2 * o, QR * s, QR * s + QR),
                          s < 0 ? 0u : {{ p }}_rowmask(2 * o + 1, QR * s, QR * s + QR), pp + 288 * o + (s < 0 ? 144 : 0));
        if (s >= 0)
            {{ p }}_pv(v, v, pp, (const float __aie_dm_resource_c *)y, (float __aie_dm_resource_d *)y);
    }
    if (st[{{ A_ROLE }}] == {{ ROLE_V }}) {  // (the emission: the panels of the record's rows up to the step's last position, its rows)
        uint8_t *__restrict e = AW(uint8_t, {{ a.rec }}) + {{ REC_V + ROWS_B }} * t;
        al::copy<{{ REC_V // 4 }}, 16>((uint32_t *)e, (const uint32_t *)v);
        al::copy<{{ ROWS_B // 4 }}, 16>((uint32_t *)(e + {{ REC_V }}), (const uint32_t *)rows);
    }
}

}  // namespace

extern "C" {

// A new X object: F_ATT (the (sub-)step: norms, rotations; q, raw K, V objects and the cache follow), a main core's
// phase (its input objects and A objects: skipped in the main core's order, the norm-weight objects around their
// inputs: X and A are broadcast to both cores), or another core's first object (M_SKIP more follow)
__attribute__((minsize)) void {{ p }}_abegin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int flags = m[{{ M_FLAGS }}];
    for (int i = 0; i < {{ A_S }}; i++)
        st[i] = 0;
    if (flags & {{ F_MAIN }}) {
        if (!(flags & {{ F_LOADR }})) {  // its input objects (under its norm-weight objects), its tiles
            const int nx = m[{{ M_NX }}];
            if (flags & {{ F_NORM }}) {
                st[{{ A_NNO }}] = (nx + {{ NX_PER_NORM - 1 }}) / {{ NX_PER_NORM }};
                st[{{ A_XL }}] = nx;
            } else {
                st[{{ A_NXSKIP }}] = nx;
            }
            st[{{ A_NASKIP }}] = m[{{ M_NT }}];
        }
        return;
    }
    if (!(flags & {{ F_ATT }})) {  // (a GDN step's objects)
        st[{{ A_NXSKIP }}] = m[{{ M_SKIP }}];
        return;
    }
    // (the column's place in its KV head's chain: its first core emits the K records, the last the V panels and rows)
    const int S = m[{{ M_NS }}], split = m[{{ M_SPLIT }}], k = col - {{ NPH }} * al::div_small<{{ NPH }}>(col);
    const int last = k == {{ NPH - 1 }};
    st[{{ A_S }}] = S;
    st[{{ A_QR }}] = split ? {{ NPH * QC }} : {{ QC }};  // (query rows per slot: the column's heads, or the chain's)
    st[{{ A_NQ }}] = split ? {{ NPH }} * S : S;
    st[{{ A_NQP }}] = 1;
    st[{{ A_NKV }}] = S;
    st[{{ A_NOBJ }}] = m[{{ M_NASKIP }}];
    st[{{ A_P }}] = m[{{ M_POS }}];
    st[{{ A_SA }}] = m[{{ M_SA }}];
    st[{{ A_TREC }}] = m[{{ M_ATREC }}];
    st[{{ A_NTAIL }}] = m[{{ M_NTAIL }}];
    st[{{ A_I }}] = 0;
    st[{{ A_ROLE }}] = k == 0 ? {{ ROLE_K }} : last ? {{ ROLE_V }} : {{ ROLE_O }};
    st[{{ A_QS }}] = 0;
    st[{{ A_KS }}] = 0;
    st[{{ A_G0 }}] = m[{{ M_G0 }}];
    st[{{ A_GATT }}] = m[{{ M_GATT }}];
    st[{{ A_GJ }}] = m[{{ M_GJ }}];
    st[{{ A_TP }}] = split && !last;  // (a split step: its partial to the east neighbour, which merges it; the last emits o)
    st[{{ A_TC }}] = split && k;
    st[{{ A_NO }}] = 4 * S * (split ? (last ? {{ NPH * QC }} : 0) : {{ QC }});
    al::copy<{{ 2 * HD + 64 * SMAX }}, 16>(AW(float, {{ a.nrm }}), (const float *)(x + {{ 2 * META }}));
}

// A main core's norm-weight object: the input objects under it
__attribute__((minsize)) void {{ p }}_anb(int32_t *__restrict ws)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int left = st[{{ A_XL }}], n = left > {{ NX_PER_NORM }} ? {{ NX_PER_NORM }} : left;
    st[{{ A_NXN }}] = n;
    st[{{ A_XL }}] = left - n;
}

// The next q object (a slot's heads, the step's slot A_SA + s; split: the slots of the chain's columns in order): its
// heads' q (normed, rotated, x log2(e) / 16) into the staging rows (row s QR + its heads' index in the chain),
// sigmoid(gate)
__attribute__((minsize)) void {{ p }}_aq(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_QS }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}];
    st[{{ A_QS }}] = i + 1;
    const int kc = QR == {{ QC }} ? 0 : (S == 1 ? i : i >> 1), s = i - S * kc, h0 = {{ QC }} * kc;  // (S <= 2: SUB)
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    const float *__restrict nrm = AW(float, {{ a.nrm }});
    const float *__restrict cs = nrm + {{ 2 * HD }} + 64 * (st[{{ A_SA }}] + s);
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < {{ QC }}; h++) {
        const int r = QR * s + h0 + h;
        float *__restrict q = AW(float, {{ a.y }}) + {{ HD }} * r;
        {{ p }}_norm_rope(pay + {{ 2 * HD }} * h, nrm, cs, cs + 32, q);
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < {{ HD }}; i += 32) {
            aie::store_v(q + i, al::mul_hilo<32, {{ qc[0] }}f, {{ qc[1] }}f>(aie::load_v<32>(q + i)));
            // sigmoid(gate) = 1 / (1 + 2^(-gate log2 e)) (bf16-level: the output is rounded to bf16)
            const aie::vector<float, 32> gv = aie::accum<accfloat, 32>(aie::load_v<32>(pay + {{ 2 * HD }} * h + {{ HD }} + i)).template to_vector<float>();
            const aie::vector<float, 32> e = {{ exp2 }}(al::log2e<32>(aie::sub(aie::zeros<float, 32>(), gv)));
            aie::store_v(AW(bfloat16, {{ a.gs }}) + {{ HD }} * r + i,
                         aie::accum<accfloat, 32>(aie::inv(aie::add(e, aie::broadcast<float, 32>(1.0f)))).template to_vector<bfloat16>());
        }
    }
}

// The q rows are in: the Q panels of every set (rows past S QR zero), y and l = 0, the running maxima (a floor below
// any score)
__attribute__((minsize)) void {{ p }}_aqpan(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int nr = AW(int32_t, {{ a.st }})[{{ A_S }}] * AW(int32_t, {{ a.st }})[{{ A_QR }}];  // (query rows)
    const al::BV nI = al::neg_eye();
    float *__restrict q = AW(float, {{ a.y }});
    al::BOut<> o((bfp16ebs8 *)AW(uint8_t, {{ a.qa }}));
    const int ns = {{ "nr <= 4 ? 1 : " if SETS == 2 }}{{ SETS }};  // (the sets ``qk`` scores)
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < {{ ND }}; db++) {  // ([d-block][set]: the scores' one stream)
        AIE_LOOP_NO_UNROLL
        for (int set = 0; set < ns; set++) {
            aie::vector<float, 32> v = aie::zeros<float, 32>();
{% for qq in range(4) %}
            if (4 * set + {{ qq }} < nr)
                v.insert({{ qq }}, aie::load_v<8>(q + {{ HD }} * (4 * set + {{ qq }}) + 8 * db));
{% endfor %}
            o.push(al::hilo4(aie::accum<accfloat, 32>(v), nI));
        }
    }
    for (int set = 0; set < {{ SETS }}; set++)
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set, aie::broadcast<float, 32>(-8192.0f));  // (bf16-exact)
    al::zero<{{ 32 * (ND + 1) * SETS }}, 32>(q);
}

// The raw K, V object of the next slot (the step's A_SA + s): its K (normed, rotated) as hi + lo blocks, its V (bf16)
__attribute__((minsize)) void {{ p }}_akv(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int s = st[{{ A_KS }}];
    st[{{ A_KS }}] = s + 1;
    const bfloat16 *__restrict kv = x + {{ 2 * META }} + {{ 2 * HD }} * al::div_small<{{ NPH }}>(col);
    const float *__restrict nrm = AW(float, {{ a.nrm }}) + 64 * st[{{ A_SA }}];
    float *__restrict kf = AW(float, {{ a.kf }});
    {{ p }}_norm_rope(kv, AW(float, {{ a.nrm }}) + {{ HD }}, nrm + {{ 2 * HD }} + 64 * s, nrm + {{ 2 * HD + 32 }} + 64 * s, kf);
    uint8_t *__restrict kb = AW(uint8_t, {{ a.kb }}) + {{ 18 * ND }} * s;
    for (int i = 0; i < {{ ND // 8 }}; i++) {
        al::BV hi, lo;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(kf + 64 * i)), al::neg_eye(), hi, lo);
        al::BOut<> o((bfp16ebs8 *)(kb + 72 * i));
        o.push(hi);
        al::BOut<> ol((bfp16ebs8 *)(kb + {{ 9 * ND }} + 72 * i));
        ol.push(lo);
    }
    al::copy<{{ HD }}, 64>(AW(bfloat16, {{ a.vn }}) + {{ HD }} * s, kv + {{ HD }});
}

// A cache object (i = 3 g + ph of group g): K objects (ph 0, 1: records 4 g + 2 ph, + 1): the slots' K inserted into
// their records, the records' scores of every set (then ``asoft``: the group's softmax, while the V object streams in);
// the V object (ph 2): y += P V of its records before the open ones (the open ones: the tail objects, after the column's
// range, from their scores, still in sc); or a tail object (``tobj``)
__attribute__((minsize)) void {{ p }}_aobj(uint8_t *__restrict a, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_I }}], p = st[{{ A_P }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}];
    st[{{ A_I }}] = i + 1;
    const int t = i - (st[{{ A_NOBJ }}] - st[{{ A_NTAIL }}]);
    if (t >= 0) {
        {{ p }}_tobj(a, ws, t);
        return;
    }
    const int k3 = al::div_small<3>(i), ph = i - 3 * k3;
    // its group: the core's attention groups G0 .. GATT - 1, then (split, even core) the new positions' groups from GJ on
    const int na = st[{{ A_GATT }}] - st[{{ A_G0 }}];
    const int att = k3 < na, g = att ? st[{{ A_G0 }}] + k3 : st[{{ A_GJ }}] + k3 - na;
    const int rp = p >> 3, rl = (p + S - 1) >> 3;  // the records of the first and last slot
    float *__restrict sc = AW(float, {{ a.sc }});  // [set][record 4][8 q + j]
    if (ph < 2) {
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 2; k++) {
            const int r = 4 * g + 2 * ph + k;
            uint8_t *__restrict rec = a + {{ REC_K }} * k;
            if (r >= rp && r <= rl) {
                AIE_LOOP_NO_UNROLL
                for (int s = 0; s < S; s++)
                    if (((p + s) >> 3) == r)
                        {{ p }}_put_k(rec, AW(uint8_t, {{ a.kb }}) + {{ 18 * ND }} * s, (p + s) & 7);
                if (st[{{ A_ROLE }}] == {{ ROLE_K }})
                    al::copy<{{ REC_K // 4 }}, 16>(AW(uint32_t, {{ a.rec }}) + {{ REC_K // 4 }} * (r - rp), (const uint32_t *)rec);
            }
            if (!att)
                continue;
            float *__restrict so = sc + 32 * (2 * ph + k);  // (every set's, the unused ones' too: their P stays finite)
{% if SETS == 2 %}
            if (r <= rl && QR * S <= 4)  // (a sub-step of one set's rows: the other set's scores stay as they were, its
                {{ p }}_qk<1>(rec, AW(uint8_t, {{ a.qa }}), so);  // rows unused)
            else if (r <= rl)
{% else %}
            if (r <= rl)
{% endif %}
                {{ p }}_qk<{{ SETS }}>(rec, AW(uint8_t, {{ a.qa }}), so);
            AIE_LOOP_NO_UNROLL
            for (int set = 0; set < (r >= rp ? {{ SETS }} : 0); set++)  // (masked: -16384, P 0)
                aie::store_v(so + 128 * set, r > rl ? aie::broadcast<float, 32>(-16384.0f) :
                             aie::select(aie::load_v<32>(so + 128 * set), aie::broadcast<float, 32>(-16384.0f),
                                         aie::mask<32>::from_uint32({{ p }}_mask(set, r, p, S, QR))));
        }
        st[{{ A_SOFT }}] = att && ph ? g + 1 : 0;
        return;
    }
    if (!att)
        return;
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < rp - 4 * g && k < 4; k += 2)  // (the records before the open ones: their P in pp, ``asoft``)
        {{ p }}_pv(a + {{ REC_V }} * k, a + {{ REC_V }} * (k + 1), AW(uint8_t, {{ a.pp }}) + {{ 144 * NO }} * k,
                   (const float __aie_dm_resource_c *)AW(float, {{ a.y }}), (float __aie_dm_resource_d *)AW(float, {{ a.y }}));
}

// After a cache object's release (the core's next object streams in meanwhile): when it was the second K object of an
// attended group, the group's softmax: per set the group's maximum per row, y and l rescaled by 2^(m - m_new) when one
// rises (a row whose maximum stayed keeps its y and l exactly), the P panels of the records before the open ones (pp:
// per pair of records [octet][P0 hi | P0 lo | P1 hi | P1 lo]; a lone record's second P 0)
__attribute__((minsize)) void {{ p }}_asoft(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int g = st[{{ A_SOFT }}] - 1, rp = st[{{ A_P }}] >> 3;
    if (g < 0)
        return;
    st[{{ A_SOFT }}] = 0;
    float *__restrict sc = AW(float, {{ a.sc }});
    float *__restrict y = AW(float, {{ a.y }});
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < {{ SETS }}; set++) {
        // the group's maximum per row; y, l rescaled by 2^(m - m_new) when one rises (a row whose maximum stayed keeps
        // its y and l exactly)
        const float *__restrict ss = sc + 128 * set;
        float *__restrict ml = AW(float, {{ a.ml }}) + 64 * set;
        const aie::vector<float, 32> m0 = aie::load_v<32>(ml);
        const aie::vector<float, 32> mg = aie::max(aie::max(aie::load_v<32>(ss), aie::load_v<32>(ss + 32)),
                                                   aie::max(aie::load_v<32>(ss + 64), aie::load_v<32>(ss + 96)));
        const aie::vector<float, 32> mn = aie::max(m0, al::max8(mg));
        float *__restrict ys = y + {{ 32 * (ND + 1) }} * set;
        // (else alpha = 1. mn >= m0: the bits of the non-negative difference compare as an integer; a float compare
        // would be a soft-float call, __gtsf2)
        if (__builtin_bit_cast(int32_t, (float)aie::reduce_max(aie::sub(mn, m0))) > 0)
            {{ p }}_mrg((const float __aie_dm_resource_a *)ys, ys, (const float __aie_dm_resource_b *)ys,
                        {{ exp2 }}(aie::sub(m0, mn)), aie::zeros<float, 32>());
        aie::store_v(ml, mn);
    }
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 4; k++) {  // (P V two records at a time: the same roundings as one at a time; P 0 from rp on)
        const uint32_t keep = 4 * g + k < rp ? ~0u : 0u;
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < {{ NO }}; o++)
            {{ p }}_psoft(ws, k, o, keep, keep, AW(uint8_t, {{ a.pp }}) + {{ 144 * NO }} * (k & 2) + 288 * o + 144 * (k & 1));
    }
}

// The attention is through (split: first the west neighbour's partial merged in, pw its workspace through shared
// memory; the partial of a core with an east neighbour then stays for it): 1 / l per row (8 lanes each); st[A_NE] R
// objects (o, then the records)
__attribute__((minsize)) void {{ p }}_aend(int32_t *__restrict ws, const int32_t *__restrict pw)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    st[{{ A_EIDX }}] = 0;
    if (st[{{ A_NOBJ }}] == 0) {
        st[{{ A_NE }}] = 0;
        return;
    }
    const int S = st[{{ A_S }}], nsets = (st[{{ A_QR }}] * S + 3) >> 2;
    const int role = st[{{ A_ROLE }}];
    st[{{ A_NE }}] = st[{{ A_NO }}] + (role == {{ ROLE_V }} ? {{ 2 * (REC_V + ROWS_B) // (2 * C_LEN) }} :
                                       role == {{ ROLE_K }} ? {{ 2 * REC_K // (2 * C_LEN) }} : 0);
    for (int set = 0; set < (st[{{ A_TC }}] ? nsets : 0); set++) {  // m = max(m_e, m_o): y, l = both rescaled, summed
        float *__restrict ml = AW(float, {{ a.ml }}) + 64 * set;
        const float *__restrict me = (const float *)pw + {{ a.ml // 4 }} + 64 * set;
        const aie::vector<float, 32> m0 = aie::load_v<32>(ml), m1 = aie::load_v<32>(me);
        const aie::vector<float, 32> mx = aie::max(m0, m1);
        aie::store_v(ml, mx);
        float *__restrict y = AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set;
        const float *__restrict ye = (const float *)pw + {{ a.y // 4 }} + {{ 32 * (ND + 1) }} * set;
        {{ p }}_mrg((const float __aie_dm_resource_a *)y, y, (const float __aie_dm_resource_b *)ye,
                    {{ exp2 }}(aie::sub(m0, mx)), {{ exp2 }}(aie::sub(m1, mx)));
    }
    if (st[{{ A_TP }}])  // (a split step's core with an east neighbour: its partial stays for it)
        return;
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {
        const float *__restrict l = AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set + {{ 32 * ND }};
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set + 32, al::rows4(aie::inv(al::sum8(aie::load_v<32>(l)))));
    }
}

// The next R object: of o ([head QR][k block 32][slot S][8] bf16: the sub-step's slots, o = y / l * sigmoid(gate)),
// then of the records (a chain's first column the K records of the step's first and last positions, its last one
// their V panels and rows: the same record twice when one holds both)
__attribute__((minsize)) void {{ p }}_aemit(int32_t *__restrict ws, bfloat16 *__restrict r)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}], S = st[{{ A_S }}], no = st[{{ A_NO }}], QR = st[{{ A_QR }}];
    st[{{ A_EIDX }}] = e + 1;
    if (e >= no) {
        const int k = e - no, p = st[{{ A_P }}], one = (p >> 3) == ((p + S - 1) >> 3);
        const int nr = st[{{ A_ROLE }}] == {{ ROLE_V }} ? {{ (REC_V + ROWS_B) // (2 * C_LEN) }} : {{ REC_K // (2 * C_LEN) }};
        const int second = k >= nr;
        aie::store_v(r, aie::load_v<{{ C_LEN }}>(AW(bfloat16, {{ a.rec }}) + {{ C_LEN }} * (k - (second ? nr : 0) + (second && !one ? nr : 0))));
        return;
    }
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 8; i += 2) {  // (two chunks of 8 values: chunk c = (head, k block) S + slot)
        aie::vector<float, 16> yv, il, g;
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < 2; k++) {
            const int c = 8 * e + i + k, hk = S == 1 ? c : c >> 1, s = c - S * hk, row = QR * s + (hk >> 5), db = hk & 31;
            const int set = row >> 2, q = row & 3;
            yv.insert(k, aie::load_v<8>(AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set + 32 * db + 8 * q));
            il.insert(k, aie::broadcast<float, 8>(AW(float, {{ a.ml }})[64 * set + 32 + 8 * q]));
            g.insert(k, aie::accum<accfloat, 8>(aie::load_v<8>(AW(bfloat16, {{ a.gs }}) + {{ HD }} * row + 8 * db)).template to_vector<float>());
        }
        aie::store_v(r + 8 * i, aie::accum<accfloat, 16>(al::mulff<16>(al::mulff<16>(yv, il), g)).template to_vector<bfloat16>());
    }
}

}  // extern "C"
""")


class AttnShape:
    """The attention core's sizes for a model: QC query heads per column, NPH columns per KV head (a chain), the
    query-row sets, the cache's capacity."""

    def __init__(self, cfg: QwenConfig, max_positions: int, *, c_len: int = 64, a_bytes: int = OBJ_BYTES, xpay: int = 4096):
        assert cfg.head_dim == HD and cfg.rope_dims == 2 * ROT and cfg.num_attention_heads % COLS == 0
        self.kvh = cfg.num_key_value_heads
        assert COLS % self.kvh == 0 and COLS // self.kvh >= 2 and 2 * HD * self.kvh <= xpay
        self.nph = COLS // self.kvh
        self.qc = cfg.num_attention_heads // COLS
        assert 1 <= self.qc <= 4 and 2 * 2 * self.qc * HD <= xpay and 4 * (2 * HD + 64 * SMAX) <= xpay
        assert REC.pg(a_bytes) == POS_PER_GROUP
        # (a split sub-step's slots: SUB, or one where a chain's heads would need more query rows than the workspace
        # holds: Qwen3.6-35B-A3B's 8 per KV head; then a step takes up to SMAX sub-steps)
        self.split_sub = SUB if SUB * self.nph * self.qc <= MAX_SPLIT_ROWS else 1
        self.u = U if self.split_sub == SUB else SMAX
        self.rows = max(self.split_sub * self.nph * self.qc, SUB * self.qc)  # (a sub-step's query rows at most)
        self.sets = -(-self.rows // 4)
        self.octets = -(-self.sets // 2)  # (P V's: two sets each, the last one alone for an odd count)
        self.c_len, self.a_bytes = c_len, a_bytes
        self.eps = cfg.rms_norm_eps
        self.groups = -(-max_positions // POS_PER_GROUP)
        self.max_positions = self.groups * POS_PER_GROUP
        self.head_bytes = 3 * a_bytes * self.groups  # a KV head's cache

    def role(self, c: int) -> int:
        """Column c's role in its KV head's chain (``ROLE_K``: its first, ``ROLE_V``: its last, ``ROLE_O``)."""
        k = c % self.nph
        return ROLE_K if k == 0 else ROLE_V if k == self.nph - 1 else ROLE_O

    def n_rel(self, c: int, split: bool, n: int) -> int:
        """R objects of a sub-step of n slots of column c: o (split: the chain's last column's of the chain's heads),
        then two records (the chain's first column K records, its last one V panels and rows)."""
        role = self.role(c)
        no = 4 * n * ((self.nph * self.qc if role == ROLE_V else 0) if split else self.qc)
        return no + {ROLE_K: 2 * REC_K, ROLE_V: 2 * (REC_V + ROWS_B), ROLE_O: 0}[role] // (2 * self.c_len)

    @staticmethod
    def splits(g: int) -> bool:
        """Whether a sub-step whose slots are in group g splits the cache along the chains (from 4 groups on: the
        same for each of its positions in every step)."""
        return g >= 3

    def bounds(self, g: int) -> list[int]:
        """A split sub-step in group g: the groups of the chain's cores, core k's from bounds[k] to bounds[k + 1]."""
        G = g + 1
        assert G >= self.nph
        return [G * k // self.nph for k in range(self.nph + 1)]

    def workspace(self) -> dict[str, int]:
        """Byte offsets of the core's workspace regions (and "size"). y: per set its blocks [db][row 4 x 8] fp32, then
        l (before the panels: the q rows [row][256] fp32); sc: per set the group's scores [record 4][row 4 x 8]; pp: per
        pair of records [octet][P0 hi | P0 lo | P1 hi | P1 lo]; rec: the even column's K records to emit, or the odd
        column's [V panels | rows] per record, on into qa ([d-block][set] Q panels; done with by the tail objects,
        which decode a record's V panels in record mode into qa)."""
        sizes = [("st", 4 * A_WORDS), ("nrm", 4 * (2 * HD + 64 * SMAX)), ("y", 4 * 32 * (ND + 1) * self.sets),
                 ("gs", 2 * self.rows * HD), ("kf", 4 * HD), ("kb", 18 * ND * SMAX), ("vn", 2 * HD * SMAX),
                 ("sc", 4 * 128 * self.sets), ("pp", 576 * self.octets), ("ml", 4 * 64 * self.sets), ("rec", 2 * REC_K),
                 ("qa", 72 * ND * self.sets)]
        out = regions(sizes)
        assert 32 * (ND + 1) * self.sets >= self.rows * HD and 18 * ND * SMAX >= REC_V
        assert out["qa"] - out["rec"] == 2 * REC_K and 2 * (REC_V + ROWS_B) <= out["size"] - out["rec"]
        assert ROWS_B <= 72 * ND * self.sets
        return out

    def render(self, p: str, consts: dict[str, int]) -> str:
        return _SRC.render(p=p, a=Attr(self.workspace()), QC=self.qc, NPH=self.nph, SETS=self.sets, NO=self.octets,
                           HALF=self.sets % 2, SMAX=SMAX, HD=HD, ND=ND,
                           REC_K=REC_K, REC_V=REC_V, ROWS_B=ROWS_B, eps=repr(self.eps), C_LEN=self.c_len,
                           qc=bf16_hilo(math.log2(math.e) / 16),
                           **(consts | {"F_ATT": F_ATT} | {k: v for k, v in globals().items()
                                                            if k.startswith(("A_", "ROLE_"))}))


# ----------------------------------------------------------------------
# Host side: the RoPE angles
# ----------------------------------------------------------------------


def rope_cs(cfg: QwenConfig, position: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin (32 fp32) of a position's angles (``QwenConfig.rope_theta``: a text position or (T, H, W))."""
    th = cfg.rope_theta(np.reshape(position, (1, -1)) if np.ndim(position) else [position])[0].astype(np.float64)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
