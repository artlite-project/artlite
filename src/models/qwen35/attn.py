"""The attention core of the multi-slot decode overlay (``qwen35/decode.py``): the full-attention layers (head_dim 256,
NeoX RoPE on the first 64 dims, q / k RMS-normed per head, a sigmoid output gate) for S <= 4 token slots at positions
p .. p + S - 1 (a sub-step; a design of 8 slots: 8, "One pass").

The KV heads' columns form chains of NPH = 8 / KV heads (9B / 27B NPH 2, MiniCPM-V 4): column c attends with its QC
query heads (27B 3, 9B 2, MiniCPM-V 1) to KV head c // NPH over the whole cache (``decode.kv``'s d-block-major layout:
groups of 32 positions, two K objects of records [d-block][hi | lo] and a V object of pairs of records [d-block][record
2]). A step's attention runs in sub-steps (``AttnShape.sub`` slots of one group at most, ``decode.vring.substeps``)
whose S x QC query rows (row s QC + h: slot s, head h) go 4 to a set of MMUL panels ([rows hi | rows lo] per 8 dims),
two sets to an octet. Per sub-step the core gets:

- from the X stream (the others skip them: ``M_SKIP``): the ``F_ATT`` object [q_norm, k_norm (256 fp32 each) | cos, sin
  per slot of the sub-step (up to 4 x 64 fp32)], the slots' q objects ([q | gate] x QC heads, bf16, as attn_in drained them),
  then the slots' raw K, V objects ([K 256 | V 256] per KV head, bf16);
- from the A stream (broadcast from the main core's, whose objects it skips; two buffers: an object streams in while
  the core works on the last one): the cache of its head up to the sub-step's group, then (a core that attends to that
  group) the two tail objects (``M_NASKIP`` objects in all).

It inserts the slots' K (normed, rotated, bf16 hi + lo blocks) into their records (in its copy), scores every record
against each octet's two sets at once (log2 units: Q x log2(e) / 16; per d-block the record's K panels and the octet's Q
panels each popped once), masks each row's positions past its own, softmax with a running maximum per row (masked
positions add exactly nothing; rescaling y and l once per group, a row whose maximum stayed keeps them exactly:
``al::merge``), y += P V over octets (two sets' 8 rows, P hi and P lo as two MMULs into one accumulator: a record's P V
the same roundings alone or with its pair; a pair's V panels on one stream, each popped once for every octet). A group's
softmax (maxima, rescale, P panels) runs after its second K object (``asoft``). What a position sees of a V record is
its panels encoded from the exact values of its positions up to it (``kernel_lib.h``'s V records): the open records'
values come from the ring of the layer's KV head (``decode.vring``; or after a prefill the record's V panels:
``M_ATREC``) as the tail objects, after the cache; the core puts the slots' V in and per slot encodes the record up to
the slot and adds its P V. Its outputs go to the main core (shared-memory FIFO R, ``C_LEN`` values per object), which
relays them to C: o = y / l * sigmoid(gate) ([head][k block: 8 dims][slot S][8] bf16: into the sub-step's slots of the
attn_out phase's input), then the records of the sub-step's first and last positions (the same twice when one record
holds both): a chain's first column their K records, its last one their V panels up to the last position and their rows
(the ring), the columns between none (``A_ROLE``). The core holds its last two A objects until then and reads the
records from them (``HOLD``): the tail objects, or the K objects of the sub-step's group (a chain's first column gets
them last: split, only them of that group; else again after its tails).

A sub-step from 4 groups on (``AttnShape.splits``: the same for every position of its group) reads each head's cache
once per chain instead of once per column: every core of a chain takes the chain's NPH QC heads (the chain's q objects
in column order on X) and a run of the groups (``AttnShape.bounds``): the first core the groups 0 .. ``M_GATT`` - 1
(then the sub-step group's K objects from ``M_GJ`` on only for its K records), the last one the groups up to the
sub-step's. The partials (m, l, y) merge west to east: a core's partial stays in its workspace until its east
neighbour (shared memory) has merged it into its own, and the last core emits o for the chain's heads (a token
ObjectFifo orders each neighbour pair). So a slot's attention is the same in every step it can be part of: a step of
S slots gives each the numbers of a step of one.

One pass (a design of 8 slots, ``AttnShape.one``): a sub-step holds all of a step's slots in its group, and a core attends
with its own QC query heads only (up to 24 rows) over the whole cache, which the chain reads once: the first column's A
stream reaches both cores of a KV head's chain (the last core's memtile link: the two columns' metadata count the same
objects, and a broadcast's objects wait for every consumer), and no partials move. The rows are single bfp16 (an octet's
Q and P as one panel of 8 rows, as an 8-slot design's x, at the model's KL: half the MMULs and of the panels' L1), a
group's P V one pass over its V object per octet (``pv4``: four MMULs per read of y), and a group that raises no row's
maximum costs a compare. The first core's K objects come after the tails: the last core, holding its tails until its
emission, then skips them (``A_NAF``). A step's slots at 32K positions: 2.13 ms a layer at 2 octets (S 3 .. 5), 3.2 at 3.

L1: a sub-step's rows need per set ~5.3 KB (y and l fp32, scores, maxima, P panels) plus its Q panels (2.3 KB) and the
rows' gates (0.5 KB a row). Regions the core reads rarely (the gates, ...: ``AttnShape.ext``) lie in the main core's
memory, which the core reads through shared memory: the main core's workspace past its residual (``EXT_W`` bytes: GEMV
scratch, idle while an attention sub-step runs; its next phase starts after a sync on the attention's outputs).

Time (the 27B, one layer, 32K positions, a split sub-step of the chain's 2 cores): 2.04 / 2.05 ms for one / two slots,
the DMA floor of the cache's 9280-byte objects (every column's share of the DRAM port); 2.12 / 2.13 for three / four
(24 rows).
"""

import math

import numpy as np

from artlite.decode.design import XB
from artlite.decode.kv import Records
from artlite.model_config import QwenConfig
from artlite.operators.base import JINJA_ENV
from artlite.operators.design import COLS
from artlite.operators.util import Attr, bf16_hilo, regions, rup

HD = 256  # head_dim
ND = HD // 8  # d-blocks (panels per record part)
REC = Records(HD)
REC_K, REC_V, ROWS_B = REC.k_bytes, REC.v_bytes, REC.rows_bytes  # bytes of a K record, a V record, its rows (a ring slot)
POS_PER_GROUP = 32  # 4 records: two K objects and a V object
OBJ_BYTES = 9280  # the overlay's objects: a weight tile (1024 columns: 16 rows of Q4_0, Q4_K) and a cache object
ROT = 32  # rotated pairs (dims i, i + 32 for i < 32)
F_ATT = 1 << 10  # M_FLAGS of an attention core's object
SMAX = 4
MAX_ROWS = 16  # query rows of a sub-step at most (``AttnShape.sub``), the gates in the core's sight
MAX_ROWS_G = 24  # with the main core applying the gates (``AttnShape.main_gates``: dense models' designs)
V_SCATTER = (18, ND, 36, 36 * ND)  # a V record's drain into its pair (txn.shim_bd dims, words): ND panels 144 B apart
TAILS = 2  # tail objects of a sub-step that attends to its group: the ring slots of its first and last records
HOLD = 2  # the last A objects of an attention sub-step, held until its emission
STACK = 1536  # the core's stack (its deepest kernel call: ~1.2 KB)
L1 = 65536
EXT_W = 9728  # bytes of the main core's workspace past its residual (``decode.design``: inv .. scr), idle in a sub-step
EXT_X = 5760  # bytes of a dense design's main core's sum panels (``decode.design.xs_bytes``), idle in a sub-step
EXT_W8, EXT_X8 = 12288, 9600  # (the same of a design of 8 slots)
# state words of the main core for the gates (main_gates: its workspace's words past decode.design's of a dense design):
# the q objects to gate, those seen, the slots, the query rows per slot, the R objects of o, those relayed
MG_Q, MG_I, MG_S, MG_QR, MG_NO, MG_E, MG_SI = range(56, 63)


# metadata words of an F_ATT object: decode.py's M_NS (slots), M_SKIP (X objects after it), M_NASKIP (the cache's A
# objects and the tail objects), M_NREL (R objects to relay), M_POS (the first slot's position), M_SA (the first slot's
# row of angles in the object: 0, each sub-step's object holds its own slots'), M_ATREC (record mode: the first tail object is the open record's V panels), M_NTAIL (tail objects)

# state words of the attention core (the first ones are trip counts of its program). A_NOBJ: the A objects before the
# held ones; A_NKH: the sub-step group's K objects after the tails (a chain's first column, not split); A_NRM, A_VN, A_GS,
# A_QA (one per octet): the addresses of the norms, the step's V, the gates, the Q panels; A_HP: the held objects (in
# order), A_RW: the open records' rows (per tail object)
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NQ, A_NQP, A_NKV, A_NOBJ, A_NE, A_TP, A_TC, A_S, A_P, A_I, A_EIDX, A_ROLE, A_QS,
 A_KS, A_XL, A_QR, A_G0, A_GATT, A_GJ, A_NO, A_SA, A_TREC, A_NTAIL, A_SOFT, A_NKH, A_NRM, A_VN, A_GS) = range(32)
A_QA, A_HP, A_RW = 32, 36, 38  # (A_QA: up to 4 octets)
A_LPER = A_TP  # (a main core's phase of a packed design of 8 slots: log2 of its input objects' strips)
A_WORDS = 40
A_NAF, A_SI, A_RWS = 40, 41, 42  # (one pass) the A objects after the held ones, 65536 / S, the record rows' scratch
ROLE_K, ROLE_V, ROLE_O = 0, 1, 2  # A_ROLE: a chain's first core (its K records), its last (V panels, rows), between

_SRC = JINJA_ENV.from_string(r"""
// The attention core's workspace (offsets in bytes: AttnShape.workspace(), the state words first); AP: a region whose
// address is in state word w (AttnShape.ext: in the workspace or in the main core's memory)
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))
#define AP(T, w) ((T *)(uintptr_t)((const int32_t *)ws)[w])

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
    aie::vector<bfloat16, 32> ih, il;  // (x is bf16: x / rms = x ih + x il, mulff's products less x's lo part, 0)
    al::split2<32>(aie::broadcast<float, 32>(inv), ih, il);
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ HD }}; i += 32) {
        const aie::vector<bfloat16, 32> xb = aie::load_v<32>(x + i);
        aie::store_v(out + i, al::mulff<32>(aie::mac(aie::mul(xb, ih), xb, il).template to_vector<float>(), aie::load_v<32>(w + i)));
    }
    const aie::vector<float, 32> a = aie::load_v<32>(out), b = aie::load_v<32>(out + 32);
    const aie::vector<float, 32> c = aie::load_v<32>(cs), s = aie::load_v<32>(sn);
    aie::store_v(out, aie::sub(al::mulff<32>(a, c), al::mulff<32>(b, s)));
    aie::store_v(out + 32, aie::add(al::mulff<32>(a, s), al::mulff<32>(b, c)));
}

// Scores of one record (8 positions) against NS sets: so[128 s + 8 r + j] (row r < 4 of set s, position j), log2
// units. Per d-block the record's K hi and lo panels ([d-block][hi | lo]) and the sets' Q panels ([d-block][set of
// NS]), each popped once; per set an MMUL chain over the K hi panels and one over the K lo panels (a chain's MMULs
// issue 3 cycles apart: 2 NS chains for II 4), summed at the end: a row's scores the same sums whatever NS
{% if ONE %}
// (one pass: an octet's single panels of 8 rows: so[64 r + j])
[[clang::always_inline]] inline void {{ p }}_qk1(const uint8_t *__restrict k, const uint8_t *__restrict qa, float *__restrict so)
{
    al::MM8 h(aie::zeros<accfloat, 64>()), l(aie::zeros<accfloat, 64>());
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV Kh = ks.pop(), Kl = ks.pop(), Q = qs.pop();
        h.mac(Q, aie::op_transpose(Kh));
        l.mac(Q, aie::op_transpose(Kl));
    }
    const aie::vector<float, 64> c = aie::add(h.to_accum(), l.to_accum()).template to_vector<float>();
    aie::store_v(so, c.template extract<32>(0));
    aie::store_v(so + 128, c.template extract<32>(1));
}
{% else %}
template <int NS>
__attribute__((noinline)) void {{ p }}_qk(const uint8_t *__restrict k, const uint8_t *__restrict qa, float *__restrict so)
{
    al::MM8 h[NS], l[NS];
    AIE_LOOP_UNROLL_FULL
    for (int s = 0; s < NS; s++) {
        h[s] = al::MM8(aie::zeros<accfloat, 64>());
        l[s] = al::MM8(aie::zeros<accfloat, 64>());
    }
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV Kh = ks.pop(), Kl = ks.pop();
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < NS; s++) {
            const al::BV Q = qs.pop();
            h[s].mac(Q, aie::op_transpose(Kh));
            l[s].mac(Q, aie::op_transpose(Kl));
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int s = 0; s < NS; s++) {  // (q rows hi + lo)
        const aie::vector<float, 64> c = aie::add(h[s].to_accum(), l[s].to_accum()).template to_vector<float>();
        aie::store_v(so + 128 * s, aie::add(c.template extract<32>(0), c.template extract<32>(1)));
    }
}
{% endif %}

{# 2^x of 32 lanes: the rescale and merge factors, the gate's sigmoid (exactly 0 from -127 on); P's (bias 192:
   pipelines, masked scores -16384) #}
{% set exp2, e192 = km.exp2(32, below="zero"), km.exp2(64, bias=192, below="bounded", inline=True) %}
{{ exp2.src }}
{{ e192.src }}

// 32 gates: sigmoid(g) = 1 / (1 + 2^(-g log2 e)) (bf16-level: the output is rounded to bf16; -g log2 e: al::log2e's
// products of the bf16 g, negated)
[[clang::always_inline]] inline void {{ p }}_gate(const bfloat16 *__restrict g, bfloat16 *__restrict out)
{
    const aie::vector<bfloat16, 32> gb = aie::load_v<32>(g);
    const aie::vector<float, 32> e = {{ exp2 }}(aie::mac(aie::mul(gb, aie::broadcast<bfloat16, 32>(-1.4453125f)), gb,
                                                 aie::broadcast<bfloat16, 32>(0.00262451171875f)).template to_vector<float>());
    aie::store_v(out, aie::accum<accfloat, 32>(aie::inv(aie::add(e, aie::broadcast<float, 32>(1.0f)))).template to_vector<bfloat16>());
}

// y += P V of a pair of records (v: their V panels [d-block][record 2]; a lone record: the second any finite panels,
// its P 0) for NO octets' 8 rows each (sets 2 o, 2 o + 1): per record two MMULs into one accumulator, [y_2o | y_2o+1]
// (each set's [db][32] of y), P hi then P lo: per record the same roundings whichever records go together (y + 0 is
// y). pp: per octet [P0 hi | P0 lo | P1 hi | P1 lo]; y in place (yl, ys the same buffer as distinct restrict
// parameters, each block read before it is written). (At most two octets: three octets' 12 P panels spill, and the
// spilled panels come back wrong.)
{% if not ONE %}
template <int NO>
__attribute__((noinline)) void {{ p }}_pv(const uint8_t *__restrict v, const uint8_t *__restrict pp,
                                          const float __aie_dm_resource_c *__restrict yl,
                                          float __aie_dm_resource_d *__restrict ys)
{
    al::BIn<> ps((const bfp16ebs8 *)pp);
    al::BV Ah[NO], Al[NO], Bh[NO], Bl[NO];
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < NO; o++) {
        Ah[o] = ps.pop();
        Al[o] = ps.pop();
        Bh[o] = ps.pop();
        Bl[o] = ps.pop();
    }
    al::BInR<aie_dm_resource::b> vs((const bfp16ebs8 *)v);  // (after the P pops: their stream's state not live then)
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV A = vs.pop(), B = vs.pop();
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < NO; o++) {
            al::MM8 c(aie::accum<accfloat, 64>(aie::concat(aie::load_v<32>(yl + {{ 64 * (ND + 1) }} * o),
                                                           aie::load_v<32>(yl + {{ 64 * (ND + 1) }} * o + {{ 32 * (ND + 1) }}))));
            c.mac(Ah[o], aie::op_transpose(A));
            c.mac(Al[o], aie::op_transpose(A));
            c.mac(Bh[o], aie::op_transpose(B));
            c.mac(Bl[o], aie::op_transpose(B));
            const aie::vector<float, 64> r = c.to_accum().template to_vector<float>();
            aie::store_v(ys + {{ 64 * (ND + 1) }} * o, r.template extract<32>(0));
            aie::store_v(ys + {{ 64 * (ND + 1) }} * o + {{ 32 * (ND + 1) }}, r.template extract<32>(1));
        }
        yl += 32;
        ys += 32;
    }
}
{% endif %}

{% if ONE %}
// y += P V of two pairs of records v0, v1 (a V object's, or one twice) for NO octets, one read and one write of y per
// d-block for the four records' MMULs: pp [pair 2][octet (a pair's PS bytes)][P0 | P1]; y in place (yl, ys the same buffer
// as distinct restrict parameters, each block read before it is written)
template <int NO>
__attribute__((noinline)) void {{ p }}_pv4(const uint8_t *v, const uint8_t *vb, const uint8_t *__restrict pp,
                                           const float __aie_dm_resource_c *__restrict yl,
                                           float __aie_dm_resource_d *__restrict ys)
{
    al::BV P0[NO], P1[NO], P2[NO], P3[NO];
    {
        al::BIn<> pa((const bfp16ebs8 *)pp);
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < NO; o++) {
            P0[o] = pa.pop();
            P1[o] = pa.pop();
        }
    }
    {
        al::BIn<> pb((const bfp16ebs8 *)(pp + {{ PS }}));
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < NO; o++) {
            P2[o] = pb.pop();
            P3[o] = pb.pop();
        }
    }
    al::BInR<aie_dm_resource::b> v0((const bfp16ebs8 *)v);
    al::BIn<> v1((const bfp16ebs8 *)vb);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV A = v0.pop(), B = v0.pop(), C = v1.pop(), D = v1.pop();
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < NO; o++) {
            al::MM8 c(aie::accum<accfloat, 64>(aie::concat(aie::load_v<32>(yl + {{ 64 * (ND + 1) }} * o),
                                                           aie::load_v<32>(yl + {{ 64 * (ND + 1) }} * o + {{ 32 * (ND + 1) }}))));
            c.mac(P0[o], aie::op_transpose(A));
            c.mac(P1[o], aie::op_transpose(B));
            c.mac(P2[o], aie::op_transpose(C));
            c.mac(P3[o], aie::op_transpose(D));
            const aie::vector<float, 64> r = c.to_accum().template to_vector<float>();
            aie::store_v(ys + {{ 64 * (ND + 1) }} * o, r.template extract<32>(0));
            aie::store_v(ys + {{ 64 * (ND + 1) }} * o + {{ 32 * (ND + 1) }}, r.template extract<32>(1));
        }
        yl += 32;
        ys += 32;
    }
}

{% endif %}
// P V (``pv``, ``pv4``) of a pair of records v (of two: vb too) for the sub-step's no octets: the first two (the second
// past the rows of a sub-step of one: stale products into rows no one reads), then a third alone
[[clang::always_inline]] inline void {{ p }}_pvs(const uint8_t *v, const uint8_t *vb, const uint8_t *__restrict pp,
                                                 float *__restrict y, int no)
{
    {{ p }}_pv{{ "4" if ONE else "" }}<2>({{ "v, vb" if ONE else "v" }}, pp, (const float __aie_dm_resource_c *)y, (float __aie_dm_resource_d *)y);
{% if NO > 2 %}
    if (no > 2)
        {{ p }}_pv{{ "4" if ONE else "" }}<1>({{ "v, vb" if ONE else "v" }}, pp + {{ 2 * PO }}, (const float __aie_dm_resource_c *)(y + {{ 128 * (ND + 1) }}),
                      (float __aie_dm_resource_d *)(y + {{ 128 * (ND + 1) }}));
{% endif %}
}

// The P panels of a record (k: its index in the group's 4) for octet o's rows (sets 2 o, 2 o + 1; r0, r1: the two
// sets' lanes to keep): P = 2^(s - m) (s the record's scores, m the running maxima), 0 below m - 63.5 (the masked
// scores, -16384, included), l += P, pp = [P hi | P lo] (bfp16 of the 8 rows' P, of its residual)
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
    aie::store_v(l + {{ 32 * (ND + 1) }}, aie::add(aie::load_v<32>(l + {{ 32 * (ND + 1) }}), P.template extract<32>(1)));
    al::BOut<> po((bfp16ebs8 *)pp);
{% if ONE %}
    po.push(aie::accum<accfloat, 64>(P).template to_vector<bfp16ebs8>());
{% else %}
    al::BV hi, lo;
    al::split_hilo(aie::accum<accfloat, 64>(P), al::neg_eye(), hi, lo);
    po.push(hi);
    po.push(lo);
{% endif %}
}

// The lanes of query rows lo .. hi - 1 in set q (8 per row)
[[clang::always_inline]] inline uint32_t {{ p }}_rowmask(int q, int lo, int hi)
{
    const int a = lo - 4 * q < 0 ? 0 : lo - 4 * q, b = hi - 4 * q > 4 ? 4 : hi - 4 * q;
    return a >= b ? 0u : (b == 4 ? ~0u : (1u << (8 * b)) - 1u) & ~((1u << (8 * a)) - 1u);
}

{% if not ONE %}
// c / S of a slot count S <= 4
[[clang::always_inline]] inline int {{ p }}_divs(int c, int S)
{
    return S == 3 ? al::div_small<3>(c) : c >> (S >> 1);
}
{% endif %}

// The octets of a sub-step's query rows
[[clang::always_inline]] inline int {{ p }}_octets(const int32_t *__restrict st)
{
    return (st[{{ A_S }}] * st[{{ A_QR }}] + 7) >> 3;
}

// y, l of a set (ND + 1 blocks) = yi + yi (a0 - 1) + yw a1 (al::merge: a row whose factor is 1 keeps its values)
__attribute__((noinline)) void {{ p }}_mrg(const float __aie_dm_resource_a *__restrict yi, float *__restrict y,
                                           const float __aie_dm_resource_b *__restrict yw, aie::vector<float, 32> a0,
                                           aie::vector<float, 32> a1)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::merge(yi, y, yw, a0, a1, {{ ND + 1 }});
}

// The slots' K blocks into the records r0, r0 + 1 of a K object (those of the sub-step's; per slot its record's slot of
// every panel, [d-block][hi | lo]: kb's blocks [hi | lo][d-block]), in place
__attribute__((noinline, minsize)) void {{ p }}_put_ks(uint8_t *__restrict a, int32_t *__restrict ws, int r0)
{
    const int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const int r = (p + s) >> 3;
        if (r != r0 && r != r0 + 1)
            continue;
        uint8_t *__restrict d = a + {{ REC_K }} * (r - r0) + 9 * ((p + s) & 7);
        const uint8_t *__restrict kb = AW(uint8_t, {{ a.kb }}) + {{ 18 * ND }} * s;
        AIE_LOOP_NO_UNROLL
        for (int db = 0; db < {{ 2 * ND }}; db++) {
            const uint8_t *__restrict src = kb + 9 * (db >> 1) + {{ 9 * ND }} * (db & 1);
            AIE_LOOP_NO_UNROLL
            for (int b = 0; b < 9; b++)
                d[72 * db + b] = src[b];
        }
    }
}

// A record's V panels (of the pair [d-block][record 2] at pr: odd, the second) -> its rows [d-block 32][position 8][8]
// bf16 (al::v_rows)
__attribute__((noinline, minsize)) void {{ p }}_vrows(const uint8_t *__restrict pr, bfloat16 *__restrict rows, int odd)
{
    al::BIn<> vi((const bfp16ebs8 *)pr);
    if (odd)
        vi.pop();
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < {{ ND }}; db++) {
        al::MM8 t(aie::zeros<accfloat, 64>());
        t.mac(al::eye(), aie::op_transpose(vi.pop()));  // [position][dim]
        vi.pop();
        aie::store_v(rows + 64 * db, t.to_accum().template to_vector<bfloat16>());
    }
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

// A record's V panels -> a pair of records [d-block][record 2] of it twice (``pv`` of a lone record)
__attribute__((noinline, minsize)) void {{ p }}_vdup(const uint8_t *__restrict rec, uint8_t *__restrict pr)
{
    al::BIn<> vi((const bfp16ebs8 *)rec);
    al::BOut<> po((bfp16ebs8 *)pr);
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < {{ ND }}; db++) {
        const al::BV x = vi.pop();
        po.push(x);
        po.push(x);
    }
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
// p) >> 3 as rows ([d-block 32][position 8][8] bf16): its ring slot (in place), or (A_TREC, t = 0) its V panels (the
// cache's pair of records from r & ~1 on) as rows in the first octet's Q panels (done with); t = 1 of a step in one
// record: nothing more (the emission repeats the first's). The step's V rows in (the rows past its last position: as
// they were, read by no one). Per slot that reaches the record: the record's V panels from its rows up to the slot's
// position (what that position sees of the record, whatever the step; in the object past the pair of records, then
// twice as a pair for ``pv`` whose second P is 0 in the second octet's Q panels), y += P V of the slot's rows. The rows
// stay for the emission (A_RW: the object is held).
__attribute__((noinline, minsize)) void {{ p }}_tobj(uint8_t *__restrict a, int32_t *__restrict ws, int t)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}], r0 = p >> 3, r1 = (p + S - 1) >> 3;
    if (t && r1 == r0)
        return;
    const int P0 = 8 * (t ? r1 : r0), last = p + S - 1 - P0 < 7 ? p + S - 1 - P0 : 7;  // (the step's last row of the record)
{% if ONE %}
    bfloat16 *__restrict rows = st[{{ A_TREC }}] && !t ? AP(bfloat16, {{ A_RWS }}) : (bfloat16 *)a;
{% else %}
    bfloat16 *__restrict rows = st[{{ A_TREC }}] && !t ? AP(bfloat16, {{ A_QA }}) : (bfloat16 *)a;
{% endif %}
    st[{{ A_RW }} + t] = (int32_t)(uintptr_t)rows;
    if (st[{{ A_TREC }}] && !t)
        {{ p }}_vrows(a, rows, r0 & 1);
    AIE_LOOP_NO_UNROLL
    for (int j = P0 > p ? 0 : p - P0; j <= last; j++)  // (row j: slot P0 + j - p)
        {{ p }}_rowput(rows + 8 * j, AP(bfloat16, {{ A_VN }}) + {{ HD }} * (P0 + j - p));
{% if ONE %}
    uint8_t *v = a + {{ ROWS_B }}, *e = a + {{ OBJ - REC_V }};  // (the object past its rows: read by no one; the pair is
    // made from its end up, each record's panels read before the pair reaches them)
{% else %}
    uint8_t *__restrict v = AP(uint8_t, {{ A_QA }} + 1);  // (the second octet's Q panels, done with)
    uint8_t *__restrict e = a + {{ 2 * REC_V }};  // (the object past its rows or pair of records: read by no one)
{% endif %}
    float *__restrict y = AW(float, {{ a.y }});
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    const int no = {{ p }}_octets(st);
{% if ONE %}
    al::zero<{{ 2 * PS }}, 64>(pp);  // (the other records' P panels 0: their bytes)
{% endif %}
    AIE_LOOP_NO_UNROLL
    for (int s = {{ 0 if ONE else -1 }}; s < S; s++) {  // (s = -1: the second record's P panels 0, once)
        const int j = p + s - P0;
        if (s >= 0 && j < 0)  // (a slot before the record)
            continue;
        if (s >= 0) {
            {{ p }}_venc(rows, e, j < 7 ? j + 1 : 8);
            {{ p }}_vdup(e, v);
        }
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < no; o++)  // (the slot's rows: P 0 elsewhere, their y and l as they are)
            {{ p }}_psoft(ws, (P0 >> 3) & 3, o, s < 0 ? 0u : {{ p }}_rowmask(2 * o, QR * s, QR * s + QR),
                          s < 0 ? 0u : {{ p }}_rowmask(2 * o + 1, QR * s, QR * s + QR), pp + {{ PO }} * o + (s < 0 ? {{ PO // 2 }} : 0));
        if (s >= 0)
            {{ p }}_pvs(v, v, pp, y, no);
    }
}

}  // namespace

extern "C" {

// A new X object: F_ATT (the (sub-)step: norms, rotations; q, raw K, V objects and the cache follow), a main core's
// phase (its input objects and A objects: skipped in the main core's order, the norm-weight objects around their
// inputs: X and A are broadcast to both cores), or another core's first object (M_SKIP more follow). mw, mx: the main
// core's workspace and sum panels (its idle regions: AttnShape.ext)
__attribute__((minsize)) void {{ p }}_abegin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col,
                                            int32_t *__restrict mw, uint8_t *__restrict mx)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int flags = m[{{ M_FLAGS }}];
    for (int i = 0; i < {{ A_S }}; i++)
        st[i] = 0;
    if (flags & {{ F_MAIN }}) {
        if (!(flags & {{ F_LOADR }})) {  // its input objects (under its norm-weight objects), its tiles
            const int nx = m[{{ M_NX }}];
{% if PACKED8 %}
            // (a packed design of 8 slots: input objects of 8 / M_SP strips each)
            const int sp = m[{{ M_SP }}], lper = sp == 1 ? 3 : (sp == 2 ? 2 : (sp == 4 ? 1 : 0));
            st[{{ A_LPER }}] = lper;
{% endif %}
            if (flags & {{ F_NORM }}) {
                st[{{ A_NNO }}] = (nx + {{ NX_PER_NORM - 1 }}) / {{ NX_PER_NORM }};
                st[{{ A_XL }}] = nx;
            } else {
{% if PACKED8 %}
                st[{{ A_NXSKIP }}] = (nx + (1 << lper) - 1) >> lper;
{% else %}
                st[{{ A_NXSKIP }}] = nx;
{% endif %}
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
{% if ONE %}
    st[{{ A_NOBJ }}] = m[{{ M_ANA }}] - {{ HOLD }};
    st[{{ A_NAF }}] = last ? {{ HOLD }} : 0;  // (a chain's last core: the first's K objects after its held ones)
    st[{{ A_SI }}] = m[{{ M_SINV }}];
{% else %}
    st[{{ A_NOBJ }}] = m[{{ M_NASKIP }}] - {{ HOLD }};
{% endif %}
    st[{{ A_P }}] = m[{{ M_POS }}];
    st[{{ A_SA }}] = m[{{ M_SA }}];
    st[{{ A_TREC }}] = m[{{ M_ATREC }}];
    st[{{ A_NTAIL }}] = m[{{ M_NTAIL }}];
    st[{{ A_I }}] = 0;
    st[{{ A_ROLE }}] = k == 0 ? {{ ROLE_K }} : last ? {{ ROLE_V }} : {{ ROLE_O }};
    st[{{ A_NKH }}] = !split && k == 0 ? {{ HOLD }} : 0;
    st[{{ A_QS }}] = 0;
    st[{{ A_KS }}] = 0;
    st[{{ A_G0 }}] = m[{{ M_G0 }}];
    st[{{ A_GATT }}] = m[{{ M_GATT }}];
    st[{{ A_GJ }}] = m[{{ M_GJ }}];
    st[{{ A_TP }}] = split && !last;  // (a split step: its partial to the east neighbour, which merges it; the last emits o)
    st[{{ A_TC }}] = split && k;
    st[{{ A_NO }}] = 4 * S * (split ? (last ? {{ NPH * QC }} : 0) : {{ QC }});
{% for w, (r, off) in at %}
    st[{{ w }}] = (int32_t)((uintptr_t){{ r }} + {{ off }});
{% endfor %}
    al::copy<{{ 2 * HD + 64 * SMAX }}, 16>(AP(float, {{ A_NRM }}), (const float *)(x + {{ 2 * META }}));
    al::zero<{{ 32 * (ND + 1) * SETS }}, 32>(AW(float, {{ a.y }}));  // (the q rows' staging: rows past the sub-step's 0)
}

// A main core's norm-weight object: the input objects under it
__attribute__((minsize)) void {{ p }}_anb(int32_t *__restrict ws)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int left = st[{{ A_XL }}], n = left > {{ NX_PER_NORM }} ? {{ NX_PER_NORM }} : left;
{% if PACKED8 %}
    st[{{ A_NXN }}] = (n + (1 << st[{{ A_LPER }}]) - 1) >> st[{{ A_LPER }}];
{% else %}
    st[{{ A_NXN }}] = n;
{% endif %}
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
{% if ONE %}
    const int s = i, h0 = 0;
{% else %}
    const int kc = QR == {{ QC }} ? 0 : {{ p }}_divs(i, S), s = i - S * kc, h0 = {{ QC }} * kc;
{% endif %}
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    const float *__restrict nrm = AP(float, {{ A_NRM }});
    const float *__restrict cs = nrm + {{ 2 * HD }} + 64 * (st[{{ A_SA }}] + s);
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < {{ QC }}; h++) {
        const int r = QR * s + h0 + h;
        float *__restrict q = AW(float, {{ a.y }}) + {{ HD }} * r;
        {{ p }}_norm_rope(pay + {{ 2 * HD }} * h, nrm, cs, cs + 32, q);
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < {{ HD }}; i += 32) {
            aie::store_v(q + i, al::mul_hilo<32, {{ qc[0] }}f, {{ qc[1] }}f>(aie::load_v<32>(q + i)));
{% if not GATE_MAIN %}
            {{ p }}_gate(pay + {{ 2 * HD }} * h + {{ HD }} + i, AP(bfloat16, {{ A_GS }}) + {{ HD }} * r + i);
{% endif %}
        }
    }
}

// The q rows are in: the Q panels of the sub-step's octets ([d-block][set 2] each, rows past S QR zero), y and l = 0,
// the running maxima (a floor below any score)
__attribute__((minsize)) void {{ p }}_aqpan(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int no = {{ p }}_octets(st);
{% if not ONE %}
    const al::BV nI = al::neg_eye();
{% endif %}
    float *__restrict q = AW(float, {{ a.y }});
    AIE_LOOP_NO_UNROLL
    for (int o = 0; o < no; o++) {
        al::BOut<> po((bfp16ebs8 *)AP(uint8_t, {{ A_QA }} + o));
        AIE_LOOP_NO_UNROLL
        for (int db = 0; db < {{ ND }}; db++) {
{% if ONE %}
            aie::vector<float, 64> v;  // (a panel of the octet's 8 rows; the rows past the sub-step's zero: abegin)
{% for qq in range(8) %}
            v.insert({{ qq }}, aie::load_v<8>(q + {{ HD }} * (8 * o + {{ qq }}) + 8 * db));
{% endfor %}
            po.push(aie::accum<accfloat, 64>(v).template to_vector<bfp16ebs8>());
{% else %}
            AIE_LOOP_NO_UNROLL
            for (int set = 2 * o; set < 2 * o + 2; set++) {  // (the rows past the sub-step's zero: abegin)
                aie::vector<float, 32> v;
{% for qq in range(4) %}
                v.insert({{ qq }}, aie::load_v<8>(q + {{ HD }} * (4 * set + {{ qq }}) + 8 * db));
{% endfor %}
                po.push(al::hilo4(aie::accum<accfloat, 32>(v), nI));
            }
{% endif %}
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
    const float *__restrict nrm = AP(float, {{ A_NRM }});
    const float *__restrict cs = nrm + {{ 2 * HD }} + 64 * (st[{{ A_SA }}] + s);
    float *__restrict kf = AW(float, {{ a.sc }});  // (the scores' region, before the cache)
    {{ p }}_norm_rope(kv, nrm + {{ HD }}, cs, cs + 32, kf);
    uint8_t *__restrict kb = AW(uint8_t, {{ a.kb }}) + {{ 18 * ND }} * s;
    for (int i = 0; i < {{ ND // 8 }}; i++) {
        al::BV hi, lo;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(kf + 64 * i)), al::neg_eye(), hi, lo);
        al::BOut<> o((bfp16ebs8 *)(kb + 72 * i));
        o.push(hi);
        al::BOut<> ol((bfp16ebs8 *)(kb + {{ 9 * ND }} + 72 * i));
        ol.push(lo);
    }
    al::copy<{{ HD }}, 64>(AP(bfloat16, {{ A_VN }}) + {{ HD }} * s, kv + {{ HD }});
}

// A sub-step's A object i: a cache object (i = 3 k + ph: group k of its run, K objects ph 0, 1: records 4 g + 2 ph,
// + 1: the slots' K inserted into their records, the records' scores of the sub-step's octets (then ``asoft``: the
// group's softmax); the V object ph 2: y += P V of its records before the open ones (the open ones: the tail objects,
// after the column's range, from their scores, still in sc)), a tail object (``tobj``), or the sub-step group's K object
// again (A_NKH: the slots' K in, for the emission). The K objects of the sub-step's group that a column reads only for
// its records are its last two: held (A_HP)
{{ "" if ONE else "__attribute__((minsize)) " }}void {{ p }}_aobj(uint8_t *__restrict a, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_I }}], p = st[{{ A_P }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}];
    st[{{ A_I }}] = i + 1;
    const int rp = p >> 3, rl = (p + S - 1) >> 3;  // the records of the first and last slot
    const int n = st[{{ A_NOBJ }}] + {{ HOLD }}, nk = st[{{ A_NKH }}];
    if (i >= n - nk) {
        const int kh = i - (n - nk);
        {{ p }}_put_ks(a, ws, (rp & ~3) + 2 * kh);
        st[{{ A_HP }} + kh] = (int32_t)(uintptr_t)a;
        return;
    }
    const int t = i - (n - nk - st[{{ A_NTAIL }}]);
    if (t >= 0) {
        st[{{ A_HP }} + t] = (int32_t)(uintptr_t)a;
        {{ p }}_tobj(a, ws, t);
        return;
    }
    const int k3 = al::div_small<3>(i), ph = i - 3 * k3;
    // its group: the core's attention groups G0 .. GATT - 1, then (split, even core) the new positions' groups from GJ on
    const int na = st[{{ A_GATT }}] - st[{{ A_G0 }}];
    const int att = k3 < na, g = att ? st[{{ A_G0 }}] + k3 : st[{{ A_GJ }}] + k3 - na;
    float *__restrict sc = AW(float, {{ a.sc }});  // [set][record 4][8 q + j]
    if (ph < 2) {
        if (4 * g + 2 * ph + 1 >= rp && 4 * g + 2 * ph <= rl)
            {{ p }}_put_ks(a, ws, 4 * g + 2 * ph);
        if (!att) {  // (the sub-step's group, for the K records: the last two objects, held)
            st[{{ A_HP }} + ph] = (int32_t)(uintptr_t)a;
            return;
        }
        const int no = {{ p }}_octets(st);
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 2; k++) {
            const int r = 4 * g + 2 * ph + k;
            const uint8_t *__restrict rec = a + {{ REC_K }} * k;
            float *__restrict so = sc + 32 * (2 * ph + k);
            AIE_LOOP_NO_UNROLL
            for (int o = 0; o < (r <= rl ? no : 0); o++)
                {{ p }}_qk{{ "1" if ONE else "<2>" }}(rec, AP(const uint8_t, {{ A_QA }} + o), so + 256 * o);
            AIE_LOOP_NO_UNROLL
            for (int set = 0; set < (r >= rp ? 2 * no : 0); set++)  // (masked: -16384, P 0)
                aie::store_v(so + 128 * set, r > rl ? aie::broadcast<float, 32>(-16384.0f) :
                             aie::select(aie::load_v<32>(so + 128 * set), aie::broadcast<float, 32>(-16384.0f),
                                         aie::mask<32>::from_uint32({{ p }}_mask(set, r, p, S, QR))));
        }
        st[{{ A_SOFT }}] = ph ? g + 1 : 0;
        return;
    }
    if (!att)
        return;
    const int no = {{ p }}_octets(st);
{% if ONE %}
    {{ p }}_pvs(a, a + {{ 2 * REC_V }}, AW(uint8_t, {{ a.pp }}), AW(float, {{ a.y }}), no);  // (P 0 from the open records on)
{% else %}
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < rp - 4 * g && k < 4; k += 2)  // (the records before the open ones: their P in pp, ``asoft``)
        {{ p }}_pvs(a + {{ REC_V }} * k, nullptr, AW(uint8_t, {{ a.pp }}) + {{ PS // 2 }} * k, AW(float, {{ a.y }}), no);
{% endif %}
}

// After a cache object: when it was the second K object of an attended group, the group's softmax: per set the group's
// maximum per row, y and l rescaled by 2^(m - m_new) when one rises (a row whose maximum stayed keeps its y and l
// exactly), the P panels of the records before the open ones (pp: per pair of records [octet][P0 hi | P0 lo | P1 hi |
// P1 lo]; a lone record's second P 0)
{{ "" if ONE else "__attribute__((minsize)) " }}void {{ p }}_asoft(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int g = st[{{ A_SOFT }}] - 1, rp = st[{{ A_P }}] >> 3, no = {{ p }}_octets(st);
    if (g < 0)
        return;
    st[{{ A_SOFT }}] = 0;
    float *__restrict sc = AW(float, {{ a.sc }});
    float *__restrict y = AW(float, {{ a.y }});
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < 2 * no; set++) {
        // the group's maximum per row; y, l rescaled by 2^(m - m_new) when one rises (a row whose maximum stayed keeps
        // its y and l exactly)
        const float *__restrict ss = sc + 128 * set;
        float *__restrict ml = AW(float, {{ a.ml }}) + 64 * set;
        const aie::vector<float, 32> m0 = aie::load_v<32>(ml);
        const aie::vector<float, 32> mg = aie::max(aie::max(aie::load_v<32>(ss), aie::load_v<32>(ss + 32)),
                                                   aie::max(aie::load_v<32>(ss + 64), aie::load_v<32>(ss + 96)));
{% if ONE %}
        if (aie::lt(m0, mg).empty())  // (no row's maximum rises: most groups, at depth)
            continue;
{% endif %}
        const aie::vector<float, 32> mn = aie::max(m0, al::max8(mg));
        float *__restrict ys = y + {{ 32 * (ND + 1) }} * set;
        // (else alpha = 1. mn >= m0: one rose where their bits differ, an integer compare; a float compare would be
        // a soft-float call, __gtsf2)
        if (!aie::eq(mn.template cast_to<int32>(), m0.template cast_to<int32>()).full())
            {{ p }}_mrg((const float __aie_dm_resource_a *)ys, ys, (const float __aie_dm_resource_b *)ys,
                        {{ exp2 }}(aie::sub(m0, mn)), aie::zeros<float, 32>());
        aie::store_v(ml, mn);
    }
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 4; k++) {  // (P V two records at a time: the same roundings as one at a time; P 0 from rp on)
        const uint32_t keep = 4 * g + k < rp ? ~0u : 0u;
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < no; o++)
            {{ p }}_psoft(ws, k, o, keep, keep, AW(uint8_t, {{ a.pp }}) + {{ PS // 2 }} * (k & 2) + {{ PO }} * o + {{ PO // 2 }} * (k & 1));
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
    const int S = st[{{ A_S }}], nsets = (st[{{ A_QR }}] * S + 3) >> 2;
    const int role = st[{{ A_ROLE }}];
    st[{{ A_NE }}] = st[{{ A_NO }}] + (role == {{ ROLE_V }} ? {{ 2 * (REC_V + ROWS_B) // (2 * C_LEN) }} :
                                       role == {{ ROLE_K }} ? {{ 2 * REC_K // (2 * C_LEN) }} : 0);
{% if not ONE %}
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
{% endif %}
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {
        const float *__restrict l = AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set + {{ 32 * ND }};
        const aie::vector<float, 16> sl = al::sum8(aie::load_v<32>(l));
        aie::vector<float, 16> il = aie::broadcast<float, 16>(1.0f);  // (one scalar inv per row)
        AIE_LOOP_UNROLL_FULL
        for (int q = 0; q < 4; q++)
            il[q] = aie::inv(sl[q]);
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set + 32, al::rows4(il));
    }
}

// The next R object: of o ([head QR][k block 32][slot S][8] bf16: the sub-step's slots, o = y / l * sigmoid(gate)),
// then of the records (a chain's first column the K records of the step's first and last positions, from the held K
// objects; its last one their V panels up to the last position, encoded again, and rows, from the held tail objects:
// the same record twice when one holds both)
__attribute__((minsize)) void {{ p }}_aemit(int32_t *__restrict ws, bfloat16 *__restrict r)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}], S = st[{{ A_S }}], no = st[{{ A_NO }}], QR = st[{{ A_QR }}];
    st[{{ A_EIDX }}] = e + 1;
    if (e >= no) {
        const int k = e - no, p = st[{{ A_P }}], one = (p >> 3) == ((p + S - 1) >> 3);
        const uint8_t *__restrict src;
        if (st[{{ A_ROLE }}] == {{ ROLE_V }}) {
            const int second = k >= {{ (REC_V + ROWS_B) // (2 * C_LEN) }}, t = second && !one;
            const int kk = k - (second ? {{ (REC_V + ROWS_B) // (2 * C_LEN) }} : 0);
            const uint8_t *__restrict rows = AP(const uint8_t, {{ A_RW }} + t);
            if (kk == 0) {  // (its V panels up to the step's last position)
                const int nv = p + S - 8 * ((t ? p + S - 1 : p) >> 3);
                {{ p }}_venc((const bfloat16 *)rows, AW(uint8_t, {{ a.kb }}), nv < 8 ? nv : 8);
            }
            src = kk < {{ REC_V // (2 * C_LEN) }} ? AW(const uint8_t, {{ a.kb }}) + {{ 2 * C_LEN }} * kk
                                                  : rows + {{ 2 * C_LEN }} * (kk - {{ REC_V // (2 * C_LEN) }});
        } else {
            const int second = k >= {{ REC_K // (2 * C_LEN) }}, rr = second && !one ? (p + S - 1) >> 3 : p >> 3;
            src = AP(const uint8_t, {{ A_HP }} + ((rr >> 1) & 1)) + {{ REC_K }} * (rr & 1) +
                  {{ 2 * C_LEN }} * (k - (second ? {{ REC_K // (2 * C_LEN) }} : 0));
        }
        aie::store_v(r, aie::load_v<{{ C_LEN }}>((const bfloat16 *)src));
        return;
    }
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 8; i += 2) {  // (two chunks of 8 values: chunk c = (head, k block) S + slot)
        aie::vector<float, 16> yv, il;
{% if not GATE_MAIN %}
        aie::vector<bfloat16, 16> g;
{% endif %}
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < 2; k++) {
{% if ONE %}
            const int c = 8 * e + i + k, hk = (c * st[{{ A_SI }}]) >> 16, s = c - S * hk, row = QR * s + (hk >> 5), db = hk & 31;
{% else %}
            const int c = 8 * e + i + k, hk = {{ p }}_divs(c, S), s = c - S * hk, row = QR * s + (hk >> 5), db = hk & 31;
{% endif %}
            const int set = row >> 2, q = row & 3;
            yv.insert(k, aie::load_v<8>(AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set + 32 * db + 8 * q));
            il.insert(k, aie::broadcast<float, 8>(AW(float, {{ a.ml }})[64 * set + 32 + 8 * q]));
{% if not GATE_MAIN %}
            g.insert(k, aie::load_v<8>(AP(bfloat16, {{ A_GS }}) + {{ HD }} * row + 8 * db));
{% endif %}
        }
{% if GATE_MAIN %}
        aie::store_v((float *)r + 8 * i, al::mulff<16>(yv, il));  // (fp32: the main core multiplies by the gates)
{% else %}
        // (g bf16: mulff's products less g's lo part, 0)
        aie::store_v(r + 8 * i, al::mulc<16>(al::mulff<16>(yv, il), g).template to_vector<bfloat16>());
{% endif %}
    }
}

}  // extern "C"
""")

# The main core's part of the gates (``AttnShape.main_gates``; decode.design's ``gate`` designs): the attention core's L1
# holds none. Rendered after the attention core's source (its helpers).
_MAIN_SRC = JINJA_ENV.from_string(r"""
namespace {

// An object of another core (the main core's begin): an F_ATT object of a column that emits o: its q objects' gates
// and its o's R objects (main_gates' state words MG_*)
[[clang::always_inline]] inline void {{ p }}_gbegin(const int32_t *__restrict m, int32_t *__restrict st, int32_t col)
{
    const int S = m[{{ M_NS }}], split = m[{{ M_SPLIT }}], k = col - {{ NPH }} * al::div_small<{{ NPH }}>(col);
    const int emit = (m[{{ M_FLAGS }}] & {{ F_ATT }}) && (!split || k == {{ NPH - 1 }}), qr = split ? {{ NPH * QC }} : {{ QC }};
    st[{{ MG_Q }}] = emit ? (split ? {{ NPH }} * S : S) : 0;
    st[{{ MG_I }}] = 0;
    st[{{ MG_S }}] = S;
    st[{{ MG_QR }}] = qr;
    st[{{ MG_NO }}] = emit ? 4 * S * qr : 0;
    st[{{ MG_E }}] = 0;
{% if ONE %}
    st[{{ MG_SI }}] = m[{{ M_SINV }}];
{% endif %}
}

}  // namespace

extern "C" {

// A skipped X object: a q object of the column's o (aq's order): its heads' sigmoid(gate) into the gate table gt
// ([row][256] bf16 in the helper's x panels, idle in the attention)
__attribute__((minsize)) void {{ p }}_mgate(const bfloat16 *__restrict x, int32_t *__restrict ws, bfloat16 *__restrict gt)
{
    int32_t *__restrict st = ws;
    const int i = st[{{ MG_I }}];
    if (i >= st[{{ MG_Q }}])
        return;
    aie::set_rounding(aie::rounding_mode::conv_even);
    st[{{ MG_I }}] = i + 1;
    const int S = st[{{ MG_S }}], QR = st[{{ MG_QR }}];
{% if ONE %}
    const int s = i, h0 = 0;
{% else %}
    const int kc = QR == {{ QC }} ? 0 : {{ p }}_divs(i, S), s = i - S * kc, h0 = {{ QC }} * kc;
{% endif %}
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < {{ QC }}; h++) {
        bfloat16 *__restrict g = gt + {{ HD }} * (QR * s + h0 + h);
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < {{ HD }}; i += 32)
            {{ p }}_gate(pay + {{ 2 * HD }} * h + {{ HD }} + i, g + i);
    }
}

// An R object -> a C object: of o (fp32 y / l: the gates' products, bf16; aemit's chunks), or a record's (bf16)
__attribute__((minsize)) void {{ p }}_mrelay(const bfloat16 *__restrict r, bfloat16 *__restrict o, int32_t *__restrict ws,
                                            const bfloat16 *__restrict gt)
{
    int32_t *__restrict st = ws;
    const int e = st[{{ MG_E }}];
    st[{{ MG_E }}] = e + 1;
    if (e >= st[{{ MG_NO }}]) {
        aie::store_v(o, aie::load_v<{{ C_LEN }}>(r));
        return;
    }
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int S = st[{{ MG_S }}], QR = st[{{ MG_QR }}];
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 8; i += 2) {
        aie::vector<bfloat16, 16> g;
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < 2; k++) {
{% if ONE %}
            const int c = 8 * e + i + k, hk = (c * st[{{ MG_SI }}]) >> 16, s = c - S * hk;
{% else %}
            const int c = 8 * e + i + k, hk = {{ p }}_divs(c, S), s = c - S * hk;
{% endif %}
            g.insert(k, aie::load_v<8>(gt + {{ HD }} * (QR * s + (hk >> 5)) + 8 * (hk & 31)));
        }
        // (g bf16: mulff's products less g's lo part, 0)
        aie::store_v(o + 8 * i, al::mulc<16>(aie::load_v<16>((const float *)r + 8 * i), g).template to_vector<bfloat16>());
    }
}

}  // extern "C"
""")


class AttnShape:
    """The attention core's sizes for a model: QC query heads per column, NPH columns per KV head (a chain), the
    slots of a sub-step and their query-row sets, the cache's capacity, the workspace and its regions in the main core's
    memory."""

    def __init__(self, cfg: QwenConfig, max_positions: int, *, c_len: int = 64, a_bytes: int = OBJ_BYTES, xpay: int = 4096,
                 slots: int = SMAX):
        """slots: a step's slots at most (a design of 8: ``one``)."""
        assert cfg.head_dim == HD and cfg.rope_dims == 2 * ROT and cfg.num_attention_heads % COLS == 0
        self.one = slots == 8  # (module docstring "One pass")
        self.smax = slots if self.one else SMAX
        self.kvh = cfg.num_key_value_heads
        assert COLS % self.kvh == 0 and COLS // self.kvh >= 2 and 2 * HD * self.kvh <= xpay
        self.nph = COLS // self.kvh
        self.qc = cfg.num_attention_heads // COLS
        assert 1 <= self.qc <= 4 and 2 * 2 * self.qc * HD <= xpay and 4 * (2 * HD + 64 * self.smax) <= xpay
        assert REC.pg(a_bytes) == POS_PER_GROUP
        # (a sub-step's slots: as many as the core holds the chain's query rows for; then a step takes up to u
        # sub-steps, one more where its slots cross a group. A dense model's main core has the program memory to apply
        # the gates: then the core takes more rows)
        self.sub = slots if self.one else min(SMAX, (MAX_ROWS if cfg.moe else MAX_ROWS_G) // (self.nph * self.qc))
        assert self.sub >= 1
        self.u = -(-(slots - 1) // self.sub) + 1  # (a step's sub-steps at most)
        self.rows = self.sub * self.qc if self.one else self.sub * self.nph * self.qc  # (a sub-step's query rows at most)
        self.main_gates = self.rows > MAX_ROWS
        self.octets = -(-self.rows // 8)
        self.sets = 2 * self.octets
        self.po = 144 if self.one else 288  # (an octet's P panels of a pair of records)
        self.ps = rup(self.po * self.octets, 64)  # (a pair of records' P panels: a panel stream starts 64-byte aligned)
        assert A_QA + self.octets <= A_HP and self.octets <= 3  # (pvs)
        self.c_len, self.a_bytes = c_len, a_bytes
        self.r_bytes = 2 * c_len * (2 if self.main_gates else 1)  # (an R object: o fp32 when the main core gates it)
        self.eps = cfg.rms_norm_eps
        self.groups = -(-max_positions // POS_PER_GROUP)
        self.max_positions = self.groups * POS_PER_GROUP
        self.head_bytes = 3 * a_bytes * self.groups  # a KV head's cache
        self._plan()

    def _plan(self) -> None:
        """The workspace (the core's L1 next to its two A buffers, X buffer, R buffers, token and stack) and the regions
        in the main core's memory (``ext``: region, offset; "mw" its workspace past its residual, "mx" its sum panels):
        the gates there (unless the main core applies them), the norms and the step's V too where the octets' Q panels
        then fit the core's L1, else the Q panels that do not."""
        y = ("y", 4 * 32 * (ND + 1) * self.sets)
        own = [("st", 4 * (A_RWS + 1 if self.one else A_WORDS)), y, ("ml", 4 * 64 * self.sets), ("sc", 4 * 128 * self.sets),
               ("pp", 2 * self.ps)]
        fixed = 2 * self.a_bytes + XB + 2 * self.r_bytes + 64 + STACK
        self.ext_w, self.ext_x = (EXT_W8, EXT_X8) if self.one else (EXT_W, EXT_X)
        free = {"mw": self.ext_w, "mx": self.ext_x}
        self.ext: dict[str, tuple[str, int]] = {}

        def put(name: str, n: int, order: tuple[str, ...] = ("mw", "mx")) -> None:
            r = next(r for r in order if free[r] >= n)
            self.ext[name] = (r, (self.ext_w if r == "mw" else self.ext_x) - free[r])
            free[r] -= n

        own.append(("kb", 18 * ND * self.smax))
        cold = [("nrm", 4 * (2 * HD + 64 * self.smax)), ("vn", 2 * HD * self.smax)]
        if self.one:
            for name, n in cold:
                put(name, n)
            put("rws", ROWS_B, ("mx", "mw"))
            pan = 72 * ND
            n_own = (L1 - fixed - regions(own)["size"]) // pan
            own += [(f"qa{o}", pan) for o in range(min(n_own, self.octets))]
            for o in range(n_own, self.octets):
                put(f"qa{o}", pan, ("mx", "mw"))
        else:
            qa = [(f"qa{o}", 2 * 72 * ND) for o in range(self.octets)]
            if not self.main_gates:
                put("gs", 2 * self.rows * HD)
                own += cold + qa
            else:
                for name, n in cold:
                    put(name, n)
                n_own = (L1 - fixed - regions(own)["size"]) // qa[0][1]
                own += qa[:n_own]
                for name, n in qa[n_own:]:
                    put(name, n, ("mx", "mw"))
            assert "qa0" in dict(own) and self.octets >= 2  # (tobj: in record mode the open record's rows in the first
            assert ROWS_B <= 2 * 72 * ND and 2 * REC_V <= 2 * 72 * ND  # octet's Q panels, a slot's V pair in the second's,
            assert max(ROWS_B, 2 * REC_V) + REC_V <= self.a_bytes  # its V panels in the tail object past its rows / pair)
        self._ws = regions(own)
        assert self._ws["size"] + fixed <= L1, (self._ws["size"], fixed)
        assert 4 * 128 * self.sets >= 4 * HD  # (kf: the scores' region)

    def at(self, tail: int) -> list[tuple[int, tuple[str, int]]]:
        """The address state words abegin sets: (word, (base, offset)); tail: the main core's workspace offset past its
        residual."""
        bases = {"mw": ("mw", tail), "mx": ("mx", 0)}
        out = []
        names = [(A_NRM, "nrm"), (A_VN, "vn")] + [(A_GS, "gs")] * (not self.main_gates) + [(A_RWS, "rws")] * self.one
        for w, name in names + [(A_QA + o, f"qa{o}") for o in range(self.octets)]:
            if name in self.ext:
                r, off = self.ext[name]
                out.append((w, (bases[r][0], bases[r][1] + off)))
            else:
                out.append((w, ("ws", self._ws[name])))
        return out

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
        l (before the panels: the q rows [row][256] fp32); sc: per set the group's scores [record 4][row 4 x 8] (before
        the cache: a slot's K, fp32); pp: per pair of records (``ps`` bytes) [octet][P0 hi | P0 lo | P1 hi | P1 lo]; qa<o>: octet o's Q
        panels [d-block][set 2] (the first one's, done with, in record mode the open record's rows)."""
        return dict(self._ws)

    def render(self, p: str, consts: dict[str, int], *, tail: int, packed8: bool = False) -> str:
        """packed8: the main core's inputs packed in a design of 8 slots (``decode.design.M_SP``)."""
        names = {"ws": "ws", "mw": "mw", "mx": "mx"}
        at = [(w, (names[r], off)) for w, (r, off) in self.at(tail)]
        v = dict(p=p, a=Attr(self.workspace()), at=at, QC=self.qc, NPH=self.nph, SETS=self.sets, NO=self.octets, SMAX=self.smax,
                 HD=HD, ND=ND, HOLD=HOLD, PS=self.ps, PO=self.po, REC_K=REC_K, REC_V=REC_V, ROWS_B=ROWS_B, eps=repr(self.eps),
                 C_LEN=self.c_len, ONE=self.one, OBJ=self.a_bytes,
                 GATE_MAIN=self.main_gates, qc=bf16_hilo(math.log2(math.e) / 16), PACKED8=packed8,
                 **(consts | {"F_ATT": F_ATT} | {k: v for k, v in globals().items() if k.startswith(("A_", "ROLE_", "MG_"))}))
        return _SRC.render(**v) + (_MAIN_SRC.render(**v) if self.main_gates else "")


# ----------------------------------------------------------------------
# Host side: the RoPE angles
# ----------------------------------------------------------------------


def rope_cs(cfg: QwenConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin (n, 32) fp32 of positions' angles (``QwenConfig.rope_theta``: (n,) text positions or (n, 3) (T, H, W))."""
    th = cfg.rope_theta(positions).astype(np.float64)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
