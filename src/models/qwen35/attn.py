"""The attention core of the multi-slot decode overlay (``qwen35/decode.py``): the full-attention layers (head_dim 256,
NeoX RoPE on the first 64 dims, q / k RMS-normed per head, a sigmoid output gate) for the token slots of a step at
positions p .. p + S - 1.

The KV heads' columns form chains of NPH = 8 / KV heads (9B / 27B NPH 2, MiniCPM-V 4): column c attends with its QC
query heads (27B 3, 9B 2, MiniCPM-V 1) to KV head c // NPH over the whole cache (``decode.kv``'s d-block-major layout with
one K panel per 8 dims: groups of 32 positions, a K object of pairs of records [d-block][record 2] and a V object of pairs
of records [d-block][record 2]). Two cores, by design:

One pass (a design of 8 slots, ``AttnShape.one``; source ``_ONE``): the rows are a head's slots: octet h holds query head h's
8 slots as rows of one MMUL panel, single bfp16 Q and P, whatever the step's S (rows past S zero): a step's compute does
not follow S, it only has to stay below the stream's time. The chain reads the cache once: the first column's A stream
reaches every core of a KV head's chain. Per step a core gets

- from the X stream (the others skip them: ``M_SKIP``): the ``F_ATT`` object [q_norm x log2(e) / 16, k_norm (256 fp32 each)
  | cos, sin per slot (8 x 64 fp32)], the slots' q objects ([q | gate] x QC heads, bf16, as attn_in drained them), then the
  slots' raw K, V objects ([K 256 | V 256] per KV head, bf16);
- from the A stream (broadcast from the main core's, whose objects it skips; two buffers: an object streams in while
  the core works on the last one): the cache up to the group of the step's last position, then the two tail objects.

The slots' K rows (normed, rotated) wait as one panel per d-block by position & 7, a step's at most 8 positions apart there
(``kr``, ``kn``); each open record takes its rows as its K object passes (``kmerge``: lanes of the popped panels, in the core's
copy, the last core's and the first's own), and a K object scores both pairs of records against the octets, each K panel
popped once (``qk1``). Each row's positions past its own are masked; P = 2^(s - m), m the row's maximum in the first group,
held for the whole pass (scores span less than P's 2^63.5: no rescale; QK starts from the baseline 192 - m), y += P V over the V object's records. A step may cross a group: the first group's open record keeps its scores in ``sc`` past the
second group's (its slot is the group's last, the second group's open record its first). The tail objects are the open
records' rows (the ring: ``decode.vring``, or after a prefill the record's V panels, ``M_ATREC``): the core puts the slots' V in,
encodes the record's V panels up to the step's last position and adds their P V (a position's rows then see the later
positions' block exponents, at the rounding level, as a step's GDN does).

The outputs go to the main core (shared-memory FIFO R, ``R_LEN`` values per object), which applies the gates (``_MAIN_ONE``: the
core's L1 holds none) and relays them to C: o ([head][k block][slot 8][8]: an octet's rows as they lie), then a chain's first
core its merged K records of the step's first and last positions, its last core their V panels and rows (the same record
twice when one holds both). The two emit at once: neither waits for an object of the other.

L1: 3 octets take 25 KB of y and l, 3 KB of scores, 7 KB of Q panels; what the core reads rarely lies in the main core's
memory (``AttnShape.ext``), which the core reads through shared memory: the main core's workspace past its residual (``EXT_W8`` bytes:
GEMV scratch, idle while the attention runs; its next phase starts after a sync on the attention's outputs) and its sum
panels (``EXT_X8``).

Sub-steps (designs of 4 slots; source ``_SRC``): a step's attention runs in sub-steps (``AttnShape.sub`` slots of one group
at most, ``decode.vring.substeps``) whose S x QC query rows (row s QC + h: slot s, head h) go 8 to an octet, one MMUL panel of
single bfp16 per 8 dims, 4 to a set in y and the scores. Per sub-step the core gets from X the ``F_ATT`` object [q_norm x log2(e) / 16, k_norm | the
sub-step's cos, sin], q and raw K, V objects as above, and from A the cache of its head up to the sub-step's group, then (a core that
attends to that group) the tail objects of its records.

It inserts the slots' K into their records (in its copy), scores every record against each octet's two sets at once (log2 units:
Q x log2(e) / 16; per d-block the pair's K panels and the octet's Q panels each popped once), masks each row's positions past
its own, softmax with P = 2^(s - m), m the row's maximum in the core's first group, held for the pass (masked positions add
exactly nothing; scores span less than P's 2^63.5: no rescale), y += P V over octets (single bfp16 P, a V object's four records
in one read and write of y: ``pv4``). A group's softmax runs after its K object (``asoft``). The open records' values come from the ring of
the layer's KV head (``decode.vring``; or after a prefill the record's V panels: ``M_ATREC``) as the tail objects, after the
cache; the core puts the slots' V in, encodes the record up to the sub-step's last position and adds its P V (a position's
rows then see the later positions' block exponents, at the rounding level, as a step's GDN does).
Its outputs go to the main core (shared-memory FIFO R, ``C_LEN`` values per object), which relays them
to C: o = y / l * sigmoid(gate) ([head][k block: 8 dims][slot S][8] bf16: into the sub-step's slots of the attn_out phase's input),
then the records of the sub-step's first and last positions (of one slot: its record): a chain's first column the K object
of their group's pairs
that hold them, its last one their V panels up to the last position and their rows (the ring), the columns between none
(``A_ROLE``). The core holds its last two A objects until then and reads the records from them (``HOLD``): the tail objects,
or the sub-step group's K and V objects (a chain's first column gets them last: split, only those of that group; else again
after its tails).

A sub-step from 4 groups on (``AttnShape.splits``: the same for every position of its group) reads each head's cache
once per chain instead of once per column: every core of a chain takes the chain's NPH QC heads (the chain's q objects
in column order on X) and a run of the groups (``AttnShape.bounds``): the first core the groups 0 .. ``M_GATT`` - 1
(then the sub-step group's K and V objects from ``M_GJ`` on only for its K records), the last one the groups up to the
sub-step's. The partials (m, l, y) merge west to east: a core's partial stays in its workspace until its east
neighbour (shared memory) has merged it into its own, and the last core emits o for the chain's heads (a token
ObjectFifo orders each neighbour pair).

A sub-step of more slots than a chain of 4 holds (``AttnShape.halves``: the 35B-A3B's 3 and 4 slots, 32 rows at most against
a core's 16) runs on the chain's halves (``SPLIT_HALF``): both cores of a half take its two columns' heads, the first
the first half of the groups, the second the rest (and the tails), and merge as a chain of 2. Both halves read the same
objects, once: a chain's first two columns read the cache on fills whose packet id also reaches the memtiles of the
two columns east of them (``decode.design``'s "Shared A objects"), which take them as if their own shims had read them. Two
streams per KV head read ~46 GB/s, one for a whole chain ~28 (a chain of 4 in the one-pass form: S 1 +30% at 32K on the 1B):
sub-steps of fewer slots stay on the whole chain, a stream per column, ~56.

L1: a sub-step's rows need per set ~5.3 KB (y and l fp32, scores, references, P panels) plus its Q panels (2.3 KB) and the
rows' gates (0.5 KB a row); the sigmoid's table takes 2.1 KB (``gate_tab``). Regions the core reads rarely (the gates, the
step's V, ...: ``AttnShape.ext``) lie in the main core's memory: its workspace past its residual (``EXT_W`` bytes), its sum
panels (``mx_bytes``), idle in a sub-step.
"""

import numpy as np

from artlite.decode.design import XB
from artlite.decode.kv import Records
from artlite.model_config import QwenConfig
from artlite.operators.base import JINJA_ENV
from artlite.operators.design import COLS
from artlite.operators.kmath import SIGMOID
from artlite.operators.util import Attr, regions, rup

HD = 256  # head_dim
ND = HD // 8  # d-blocks (panels per record part)
REC = Records(HD, 1)
REC_K, REC_V, ROWS_B = REC.k_bytes, REC.v_bytes, REC.rows_bytes  # bytes of a K record, a V record, its rows (a ring slot)
POS_PER_GROUP = 32  # 4 records: a K object and a V object
OBJS = REC.objects  # cache objects of a group
OBJ_BYTES = 9280  # the overlay's objects: a weight tile (1024 columns: 16 rows of Q4_0, Q4_K) and a cache object
ROT = 32  # rotated pairs (dims i, i + 32 for i < 32)
F_ATT = 1 << 10  # M_FLAGS of an attention core's object
SMAX = 4
MAX_ROWS = 16  # query rows of a sub-step at most (``AttnShape.sub``), the gates in the core's sight
MAX_ROWS_G = 24  # with the main core applying the gates (``AttnShape.main_gates``: dense models' designs)
V_SCATTER = (18, ND, 36, 36 * ND)  # a V record's drain into its pair (txn.shim_bd dims, words): ND panels 144 B apart
K_SCATTER = V_SCATTER  # (one pass) a K record's drain into its pair
TAILS = 2  # tail objects of a sub-step that attends to its group: the ring slots of its first and last records
HOLD = 2  # the last A objects of an attention sub-step, held until its emission
STACK = 1536  # the core's stack (its deepest kernel call: ~1.2 KB)
L1 = 65536
EXT_W = 9728  # bytes of the main core's workspace past its residual (``decode.design``: inv .. scr), idle in a sub-step
EXT_X = 5760  # bytes of a dense design's main core's sum panels (``decode.design.xs_bytes``), idle in a sub-step
EXT_W8, EXT_X8 = 12288, 9600  # (the same of a design of 8 slots)
PO = 288  # bytes of an octet's P panels of a group (4 panels)
# state words of the main core for the gates (main_gates: its workspace's words past decode.design's of a dense design):
# the q objects to gate, those seen, the slots, the query rows per slot, the R objects of o, those relayed
MG_Q, MG_I, MG_S, MG_QR, MG_NO, MG_E, MG_SI = range(56, 63)
SPLIT_CHAIN, SPLIT_HALF = 1, 2  # M_SPLIT: the chain divides the cache (0: each column over all of it); its halves do


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
A_RWS, A_KR, A_KN, A_KM = 40, 41, 42, 43  # (one pass) the record rows' scratch; the new K's rows, panels; the merged records
ROLE_K, ROLE_V, ROLE_O = 0, 1, 2  # A_ROLE: a chain's first core (its K records), its last (V panels, rows), between

# (kernels both cores use)
_PV4 = r"""// y += P V of two pairs of records v, vb (a V object's, or one twice) for NO octets, one read and one write of y per
// d-block for the four records' MMULs: pp [octet][P0 | P1 | P2 | P3]; y in place (yl, ys the same buffer as distinct
// restrict parameters, each block read before it is written)
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
            P2[o] = pa.pop();
            P3[o] = pa.pop();
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

"""

_QPAN = r"""// The rows q (fp32, [row][256]) of NO octets -> their Q panels at out ([d-block][octet NO])
template <int NO>
__attribute__((noinline)) void {{ p }}_qpan(const float *__restrict q, uint8_t *__restrict out)
{
    al::BOut<> po((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < NO; o++) {
            aie::vector<float, 64> v;
{% for r in range(8) %}
            v.insert({{ r }}, aie::load_v<8>(q + {{ HD }} * (8 * o + {{ r }}) + 8 * db));
{% endfor %}
            po.push(aie::accum<accfloat, 64>(v).template to_vector<bfp16ebs8>());
        }
    }
}

"""

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

// Scores of a pair of K records ([d-block][record 2]) against NO octets (their rows' Q panels, [d-block][octet NO]):
// so[256 o + 64 r + j] (log2 units: record r of the pair, j its rows' lanes), every K panel popped once for the octets (II 4
// for 4 MMULs). NQ: the panels per d-block, the octets the last NO of them
template <int NO, int NQ>
__attribute__((noinline)) void {{ p }}_qk(const uint8_t *__restrict k, const uint8_t *__restrict qa, float *__restrict so)
{
    al::MM8 a[NO], b[NO];
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < NO; o++) {
        a[o] = al::MM8(aie::zeros<accfloat, 64>());
        b[o] = al::MM8(aie::zeros<accfloat, 64>());
    }
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV K0 = ks.pop(), K1 = ks.pop();
        AIE_LOOP_UNROLL_FULL
        for (int o = NO - NQ; o < NO; o++) {
            const al::BV Q = qs.pop();
            if (o >= 0) {
                a[o].mac(Q, aie::op_transpose(K0));
                b[o].mac(Q, aie::op_transpose(K1));
            }
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < NO; o++) {
        const aie::vector<float, 64> x = a[o].to_accum().template to_vector<float>(), z = b[o].to_accum().template to_vector<float>();
        aie::store_v(so + 256 * o, x.template extract<32>(0));
        aie::store_v(so + 256 * o + 128, x.template extract<32>(1));
        aie::store_v(so + 256 * o + 32, z.template extract<32>(0));
        aie::store_v(so + 256 * o + 160, z.template extract<32>(1));
    }
}

{# 2^x of 32 lanes: the merge factors (exactly 0 from -127 on); P's (bias 192: pipelines, masked scores -16384) #}
{% set exp2, e192 = km.exp2(32, below="zero"), km.exp2(64, bias=192, below="bounded", above="clamp", inline=True) %}
{{ exp2.src }}
{{ e192.src }}
{% if not GATE_MAIN %}
{% set sig = km.act(km.SIGMOID, table_arg=True) %}
{{ sig.src }}
{% else %}

// 32 gates: sigmoid(g) = 1 / (1 + 2^(-g log2 e)) (bf16-level: the output is rounded to bf16; -g log2 e: al::log2e's
// products of the bf16 g, negated)
[[clang::always_inline]] inline void {{ p }}_gate(const bfloat16 *__restrict g, bfloat16 *__restrict out)
{
    const aie::vector<bfloat16, 32> gb = aie::load_v<32>(g);
    const aie::vector<float, 32> e = {{ exp2 }}(aie::mac(aie::mul(gb, aie::broadcast<bfloat16, 32>(-1.4453125f)), gb,
                                                 aie::broadcast<bfloat16, 32>(0.00262451171875f)).template to_vector<float>());
    aie::store_v(out, aie::accum<accfloat, 32>(aie::inv(aie::add(e, aie::broadcast<float, 32>(1.0f)))).template to_vector<bfloat16>());
}
{% endif %}

""" + _PV4 + r"""// P V (``pv4``) of the pairs of records v, vb for the sub-step's no octets: two at once (the second past the rows of a short
// sub-step: stale products into rows no one reads), then a third alone
[[clang::always_inline]] inline void {{ p }}_pvs(const uint8_t *v, const uint8_t *vb, const uint8_t *__restrict pp,
                                                 float *__restrict y, int no)
{
    {{ p }}_pv4<2>(v, vb, pp, (const float __aie_dm_resource_c *)y, (float __aie_dm_resource_d *)y);
{% if NO > 2 %}
    if (no > 2)
        {{ p }}_pv4<1>(v, vb, pp + {{ 2 * PO }}, (const float __aie_dm_resource_c *)(y + {{ 128 * (ND + 1) }}),
                       (float __aie_dm_resource_d *)(y + {{ 128 * (ND + 1) }}));
{% endif %}
}

// The P panel of a record (k: its index in the group's 4) for octet o's rows (sets 2 o, 2 o + 1): P = 2^(s - m) (s the record's
// scores, m the rows' references), 0 below m - 63.5 (the masked scores, -16384, included), l += P, pp = P (bfp16)
__attribute__((noinline)) void {{ p }}_psoft(int32_t *__restrict ws, int k, int o, uint8_t *__restrict pp)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict s = AW(float, {{ a.sc }}) + 256 * o + 32 * k;
    const float *__restrict m = AW(float, {{ a.ml }}) + 128 * o;
    float *__restrict l = AW(float, {{ a.y }}) + {{ 64 * (ND + 1) }} * o + {{ 32 * ND }};
    const aie::vector<float, 32> c = aie::broadcast<float, 32>(192.0f);
    const aie::accum<accfloat, 64> A = aie::concat(aie::add(aie::accum<accfloat, 32>(aie::load_v<32>(s)), aie::sub(c, aie::load_v<32>(m))),
                                                   aie::add(aie::accum<accfloat, 32>(aie::load_v<32>(s + 128)), aie::sub(c, aie::load_v<32>(m + 64))));
    const aie::mask<64> keep = aie::ge(A.template to_vector<bfloat16>(), aie::broadcast<bfloat16, 64>(129.0f));
    const aie::vector<float, 64> P = aie::select(aie::zeros<float, 64>(), {{ e192 }}(A).template to_vector<float>(), keep);
    aie::store_v(l, aie::add(aie::load_v<32>(l), P.template extract<32>(0)));
    aie::store_v(l + {{ 32 * (ND + 1) }}, aie::add(aie::load_v<32>(l + {{ 32 * (ND + 1) }}), P.template extract<32>(1)));
    al::BOut<> po((bfp16ebs8 *)pp);
    po.push(aie::accum<accfloat, 64>(P).template to_vector<bfp16ebs8>());
}

""" + _QPAN + r"""// c / S of a slot count S <= 4
[[clang::always_inline]] inline int {{ p }}_divs(int c, int S)
{
    return S == 3 ? al::div_small<3>(c) : c >> (S >> 1);
}

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

// A slot's K blocks (src: [d-block]) into their panels (d: a block, then its place in the next d-block's)
__attribute__((noinline)) void {{ p }}_kcopy(uint8_t *__restrict d, const uint8_t *__restrict src)
{
    dims_2d_t w = dims_2d_from_steps(9, 1, 144);  // (block starts 144 bytes apart)
    AIE_LOOP_NO_UNROLL
    for (int b = 0; b < {{ 9 * ND }}; b++) {
        *d = *src++;
        d = add_2d_byte(d, w);
    }
}

// The slots' K blocks into the records of a K object from record r0 on (those of the sub-step's; per slot its record's
// block of every panel, [pair][d-block][record 2]: kb's blocks [slot][d-block]), in place
__attribute__((noinline, minsize)) void {{ p }}_put_ks(uint8_t *__restrict a, int32_t *__restrict ws, int r0)
{
    const int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const int i = ((p + s) >> 3) - r0;
        if (i >= 0 && i <= 3)
            {{ p }}_kcopy(a + {{ 2 * REC_K }} * (i >> 1) + 72 * (i & 1) + 9 * ((p + s) & 7), AW(uint8_t, {{ a.kb }}) + {{ 9 * ND }} * s);
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
{% if HALF %}
        const int row = 4 * set + q, s = QR == {{ QC }} ? al::div_small<{{ QC }}>(row) : QR == {{ 2 * QC }} ? al::div_small<{{ 2 * QC }}>(row) : al::div_small<{{ NPH * QC }}>(row);
{% else %}
        const int row = 4 * set + q, s = QR == {{ QC }} ? al::div_small<{{ QC }}>(row) : al::div_small<{{ NPH * QC }}>(row);
{% endif %}
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
// they were, read by no one). The record's V panels from its rows up to the sub-step's last position (in the object past
// the pair of records, then twice as a pair for ``pv`` whose second P is 0 in the second octet's Q panels), y += P V (rows
// past their positions: the masked scores of ``aobj``). The rows stay for the emission (A_RW: the object is held).
__attribute__((noinline, minsize)) void {{ p }}_tobj(uint8_t *__restrict a, int32_t *__restrict ws, int t)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], r0 = p >> 3, r1 = (p + S - 1) >> 3;
    if (t && r1 == r0)
        return;
    const int r = t ? r1 : r0, P0 = 8 * r, last = p + S - 1 - P0 < 7 ? p + S - 1 - P0 : 7;  // (the step's last row of the record)
    bfloat16 *__restrict rows = st[{{ A_TREC }}] && !t ? AP(bfloat16, {{ A_QA }}) : (bfloat16 *)a;
    st[{{ A_RW }} + t] = (int32_t)(uintptr_t)rows;
    if (st[{{ A_TREC }}] && !t)
        {{ p }}_vrows(a, rows, r0 & 1);
    AIE_LOOP_NO_UNROLL
    for (int j = P0 > p ? 0 : p - P0; j <= last; j++)  // (row j: slot P0 + j - p)
        {{ p }}_rowput(rows + 8 * j, AP(bfloat16, {{ A_VN }}) + {{ HD }} * (P0 + j - p));
    uint8_t *__restrict v = AP(uint8_t, {{ A_QA }} + 1);  // (the second octet's Q panels, done with)
    uint8_t *__restrict e = a + {{ 2 * REC_V }};  // (the object past its rows or pair of records: read by no one)
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    const int no = {{ p }}_octets(st);
    {{ p }}_venc(rows, e, last + 1);
    {{ p }}_vdup(e, v);
    al::zero<{{ PS }}, 64>(pp);  // (the pair's second record: P 0)
    AIE_LOOP_NO_UNROLL
    for (int o = 0; o < no; o++)  // (rows past their positions: the masked scores of ``aobj``)
        {{ p }}_psoft(ws, r & 3, o, pp + {{ PO }} * o);
    {{ p }}_pvs(v, v, pp, AW(float, {{ a.y }}), no);
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
    // (the column's place in its KV head's chain: its first core emits the K records, the last the V panels and rows;
    // nc: the cores that divide the cache for the same rows, kc the core's place among them, lc: it is their last)
    const int S = m[{{ M_NS }}], split = m[{{ M_SPLIT }}], k = col - {{ NPH }} * al::div_small<{{ NPH }}>(col);
    const int last = k == {{ NPH - 1 }};
{% if HALF %}
    const int nc = split == {{ SPLIT_HALF }} ? 2 : {{ NPH }}, kc = k & (nc - 1), lc = kc == nc - 1;
{% else %}
    const int nc = {{ NPH }}, kc = k, lc = last;
{% endif %}
    st[{{ A_S }}] = S;
    st[{{ A_QR }}] = split ? nc * {{ QC }} : {{ QC }};  // (query rows per slot: the column's heads, or the cores')
    st[{{ A_NQ }}] = split ? nc * S : S;
    st[{{ A_NQP }}] = 1;
    st[{{ A_NKV }}] = S;
    st[{{ A_NOBJ }}] = m[{{ M_NASKIP }}] - {{ HOLD }};
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
    st[{{ A_TP }}] = split && !lc;  // (a split step: its partial to the east neighbour, which merges it; the last emits o)
    st[{{ A_TC }}] = split && kc;
    st[{{ A_NO }}] = 4 * S * (split ? (lc ? nc * {{ QC }} : 0) : {{ QC }});
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
// heads' q (normed, rotated, x log2(e) / 16: the scale in q_norm) into the staging rows (row s QR + its heads' index in the chain),
// sigmoid(gate) (ta, tb: its table)
__attribute__((minsize)) void {{ p }}_aq(const bfloat16 *__restrict x, int32_t *__restrict ws{% if not GATE_MAIN %},
                                         const uint32_t *__restrict ta, const uint32_t *__restrict tb{% endif %})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_QS }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}];
    st[{{ A_QS }}] = i + 1;
    const int kc = QR == {{ QC }} ? 0 : {{ p }}_divs(i, S), s = i - S * kc, h0 = {{ QC }} * kc;
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    const float *__restrict nrm = AP(float, {{ A_NRM }});
    const float *__restrict cs = nrm + {{ 2 * HD }} + 64 * (st[{{ A_SA }}] + s);
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < {{ QC }}; h++) {
        const int r = QR * s + h0 + h;
        float *__restrict q = AW(float, {{ a.y }}) + {{ HD }} * r;
        {{ p }}_norm_rope(pay + {{ 2 * HD }} * h, nrm, cs, cs + 32, q);
{% if not GATE_MAIN %}
        {{ sig }}<{{ HD }}>(pay + {{ 2 * HD }} * h + {{ HD }}, AP(bfloat16, {{ A_GS }}) + {{ HD }} * r, ta, tb);
{% endif %}
    }
}

// The q rows are in: the octets' Q panels (rows past S QR zero: abegin), y and l = 0, the rows' references
__attribute__((minsize)) void {{ p }}_aqpan(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict q = AW(float, {{ a.y }});
    {{ p }}_qpan<2>(q, AP(uint8_t, {{ A_QA }}));
{% if NO > 2 %}
    {{ p }}_qpan<2>(q + {{ 8 * HD }}, AP(uint8_t, {{ A_QA }} + 1));  // (octets 1, 2: ``qk`` takes the second)
{% endif %}
    for (int set = 0; set < {{ SETS }}; set++)
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set, aie::broadcast<float, 32>(-8192.0f));  // (bf16-exact)
    al::zero<{{ 32 * (ND + 1) * SETS }}, 32>(q);
}

// The raw K, V object of the next slot (the step's A_SA + s): its K (normed, rotated) as blocks of 8 dims, its V (bf16)
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
    al::BOut<> ko((bfp16ebs8 *)(AW(uint8_t, {{ a.kb }}) + {{ 9 * ND }} * s));
    for (int i = 0; i < {{ ND // 8 }}; i++)
        ko.push(aie::accum<accfloat, 64>(aie::load_v<64>(kf + 64 * i)).template to_vector<bfp16ebs8>());
    al::copy<{{ HD }}, 64>(AP(bfloat16, {{ A_VN }}) + {{ HD }} * s, kv + {{ HD }});
}

// A sub-step's A object i: a cache object (i = 2 k + ph: group k of its run, K object ph 0: the slots' K inserted into
// its records, the records' scores of the sub-step's octets (then ``asoft``: the group's softmax); the V object ph 1: y +=
// P V of its records before the open ones (the open ones: the tail objects, after the column's range, from their
// scores, still in sc)), a tail object (``tobj``), or the sub-step group's K object again and its V object (A_NKH: the
// slots' K in, for the emission). The objects of the sub-step's group that a column reads only for its records are its
// last two: held (A_HP)
__attribute__((minsize)) void {{ p }}_aobj(uint8_t *__restrict a, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_I }}], p = st[{{ A_P }}], S = st[{{ A_S }}], QR = st[{{ A_QR }}];
    st[{{ A_I }}] = i + 1;
    const int rp = p >> 3, rl = (p + S - 1) >> 3;  // the records of the first and last slot
    const int n = st[{{ A_NOBJ }}] + {{ HOLD }}, nk = st[{{ A_NKH }}];
    if (i >= n - nk) {
        const int kh = i - (n - nk);
        if (!kh)
            {{ p }}_put_ks(a, ws, rp & ~3);
        st[{{ A_HP }} + kh] = (int32_t)(uintptr_t)a;
        return;
    }
    const int t = i - (n - nk - st[{{ A_NTAIL }}]);
    if (t >= 0) {
        st[{{ A_HP }} + t] = (int32_t)(uintptr_t)a;
        {{ p }}_tobj(a, ws, t);
        return;
    }
    const int k2 = i >> 1, ph = i & 1;
    // its group: the core's attention groups G0 .. GATT - 1, then (split, even core) the new positions' group from GJ on
    const int na = st[{{ A_GATT }}] - st[{{ A_G0 }}];
    const int att = k2 < na, g = att ? st[{{ A_G0 }}] + k2 : st[{{ A_GJ }}] + k2 - na;
    float *__restrict sc = AW(float, {{ a.sc }});  // [set][record 4][8 q + j]
    if (!ph) {
        if (4 * g + 3 >= rp && 4 * g <= rl && (att || st[{{ A_ROLE }}] == {{ ROLE_K }}))
            {{ p }}_put_ks(a, ws, 4 * g);
        if (!att) {  // (the sub-step's group, for the K records: the last two objects, held)
            st[{{ A_HP }}] = (int32_t)(uintptr_t)a;
            return;
        }
        const int no = {{ p }}_octets(st);
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 2; k++) {
            const int r = 4 * g + 2 * k;
            if (r <= rl) {
                {{ p }}_qk<2, 2>(a + {{ 2 * REC_K }} * k, AP(const uint8_t, {{ A_QA }}), sc + 64 * k);
{% if NO > 2 %}
                if (no > 2)
                    {{ p }}_qk<1, 2>(a + {{ 2 * REC_K }} * k, AP(const uint8_t, {{ A_QA }} + 1), sc + 512 + 64 * k);
{% endif %}
            }
            AIE_LOOP_NO_UNROLL
            for (int j = 0; j < 2; j++) {
                float *__restrict so = sc + 32 * (2 * k + j);
                AIE_LOOP_NO_UNROLL
                for (int set = 0; set < (r + j >= rp ? 2 * no : 0); set++)  // (masked: -16384, P 0)
                    aie::store_v(so + 128 * set, r + j > rl ? aie::broadcast<float, 32>(-16384.0f) :
                                 aie::select(aie::load_v<32>(so + 128 * set), aie::broadcast<float, 32>(-16384.0f),
                                             aie::mask<32>::from_uint32({{ p }}_mask(set, r + j, p, S, QR))));
            }
        }
        st[{{ A_SOFT }}] = g + 1;
        return;
    }
    if (!att) {
        st[{{ A_HP }} + 1] = (int32_t)(uintptr_t)a;
        return;
    }
    if (rp > 4 * g)  // (the records before the open ones: their P in pp, ``asoft``; P 0 from there on)
        {{ p }}_pvs(a, a + {{ 2 * REC_V }}, AW(uint8_t, {{ a.pp }}), AW(float, {{ a.y }}), {{ p }}_octets(st));
}

// After a cache object: when it was the K object of an attended group, the group's softmax (the core's first group:
// first each row's reference m), the P panels of the records before the open ones (pp: [octet][P0 | P1 | P2 | P3], 0 from the
// open ones on)
__attribute__((minsize)) void {{ p }}_asoft(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int g = st[{{ A_SOFT }}] - 1, rp = st[{{ A_P }}] >> 3, no = {{ p }}_octets(st);
    if (g < 0)
        return;
    st[{{ A_SOFT }}] = 0;
    float *__restrict sc = AW(float, {{ a.sc }});
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < (st[{{ A_I }}] == 1 ? 2 * no : 0); set++) {  // (the first group: each row's m, for the pass)
        const float *__restrict ss = sc + 128 * set;
        const aie::vector<float, 32> mg = aie::max(aie::max(aie::load_v<32>(ss), aie::load_v<32>(ss + 32)),
                                                   aie::max(aie::load_v<32>(ss + 64), aie::load_v<32>(ss + 96)));
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set, al::max8(mg));
    }
    const int n = rp - 4 * g;
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    if (n > 0 && n < 4)
        al::zero<{{ PS }}, 64>(pp);
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < n && k < 4; k++) {
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < no; o++)
            {{ p }}_psoft(ws, k, o, pp + {{ PO }} * o + 72 * k);
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
    st[{{ A_NE }}] = st[{{ A_NO }}] + (S == 1 ? 1 : 2) * (role == {{ ROLE_V }} ? {{ (REC_V + ROWS_B) // (2 * C_LEN) }} :
                                                       role == {{ ROLE_K }} ? {{ PAIR_K // (2 * C_LEN) }} : 0);
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
        const aie::vector<float, 16> sl = al::sum8(aie::load_v<32>(l));
        aie::vector<float, 16> il = aie::broadcast<float, 16>(1.0f);  // (one scalar inv per row)
        AIE_LOOP_UNROLL_FULL
        for (int q = 0; q < 4; q++)
            il[q] = aie::inv(sl[q]);
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set + 32, al::rows4(il));
    }
}

// The next R object: of o ([head QR][k block 32][slot S][8] bf16: the sub-step's slots, o = y / l * sigmoid(gate)),
// then of the records (a chain's first column the pairs of K records of the step's first and last positions, from the held
// K object; its last one their V panels up to the last position, encoded again, and rows, from the held tail objects:
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
            const int second = k >= {{ PAIR_K // (2 * C_LEN) }}, rr = second && !one ? (p + S - 1) >> 3 : p >> 3;
            src = AP(const uint8_t, {{ A_HP }}) + {{ PAIR_K }} * ((rr >> 1) & 1) + {{ 2 * C_LEN }} * (k - (second ? {{ PAIR_K // (2 * C_LEN) }} : 0));
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
            const int c = 8 * e + i + k, hk = {{ p }}_divs(c, S), s = c - S * hk, row = QR * s + (hk >> 5), db = hk & 31;
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

_ONE = JINJA_ENV.from_string(r"""
// The attention core of one pass (``AttnShape.one``). Offsets in bytes: AttnShape.workspace(), the state words first; AP: a
// region whose address is in state word w (AttnShape.ext: in the workspace or in the main core's memory)
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

// Scores of a pair of K records ([d-block][record 2]) against NO octets (their rows' Q panels, [d-block][octet NO]), from
// the rows' baselines mb ([octet][set 2][b 32 | 1 / l 32]): so[256 o + 64 r + j] = A = s + b (log2 units, row j of record r),
// every K panel popped once for the octets (II 4 for 4 MMULs)
template <int NO>
[[clang::always_inline]] inline void {{ p }}_qk1(const uint8_t *__restrict k, const uint8_t *__restrict qa, const float *__restrict mb, float *__restrict so)
{
    al::MM8 a[NO], b[NO];
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < NO; o++) {
        const aie::accum<accfloat, 64> c(aie::concat(aie::load_v<32>(mb + 128 * o), aie::load_v<32>(mb + 128 * o + 64)));
        a[o] = al::MM8(c);
        b[o] = al::MM8(c);
    }
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ ND }}; d++) {
        const al::BV K0 = ks.pop(), K1 = ks.pop();
        AIE_LOOP_UNROLL_FULL
        for (int o = 0; o < NO; o++) {
            const al::BV Q = qs.pop();
            a[o].mac(Q, aie::op_transpose(K0));
            b[o].mac(Q, aie::op_transpose(K1));
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < NO; o++) {
        const aie::vector<float, 64> x = a[o].to_accum().template to_vector<float>(), z = b[o].to_accum().template to_vector<float>();
        aie::store_v(so + 256 * o, x.template extract<32>(0));
        aie::store_v(so + 256 * o + 128, x.template extract<32>(1));
        aie::store_v(so + 256 * o + 32, z.template extract<32>(0));
        aie::store_v(so + 256 * o + 160, z.template extract<32>(1));
    }
}

{# P's 2^x in two steps (bias 192; masked scores -16384) #}
{% set e192 = km.exp2(32, bias=192, below="bounded", above="clamp") %}
{{ e192.parts.reduce.src }}
{{ e192.parts.poly.src }}

""" + _PV4 + r"""// P V (``pv4``) of the pairs of records v, vb for the octets: two at once, then a third alone
[[clang::always_inline]] inline void {{ p }}_pvs(const uint8_t *v, const uint8_t *vb, const uint8_t *__restrict pp, float *__restrict y)
{
    {{ p }}_pv4<{{ QC if QC < 2 else 2 }}>(v, vb, pp, (const float __aie_dm_resource_c *)y, (float __aie_dm_resource_d *)y);
{% if QC > 2 %}
    {{ p }}_pv4<1>(v, vb, pp + {{ 2 * PO }}, (const float __aie_dm_resource_c *)(y + {{ 128 * (ND + 1) }}),
                   (float __aie_dm_resource_d *)(y + {{ 128 * (ND + 1) }}));
{% endif %}
}

// P = 2^(s - m) of a group's scores, two passes (each pipelined; as one loop the exp's constants spill). The first: n vectors of
// A at s -> P, f of kmath.exp2's first step in their place
template <int N>
__attribute__((noinline)) void {{ p }}_pexp1(float *__restrict s)
{
    AIE_LOOP_UNROLL(2)
    for (int t = 0; t < N; t++) {
        aie::accum<accfloat, 32> A(aie::load_v<32>(s));
        aie::vector<bfloat16, 32> P, f;
        {{ e192.parts.reduce }}(A, P, f);
        aie::store_v((bfloat16 *)s, P);
        aie::store_v((bfloat16 *)(s + 16), f);
        s += 32;
    }
}

// The second, N records of an octet (p0, p1: its two sets' P, f): their P as panels at pp, l += P
template <int N>
__attribute__((noinline)) void {{ p }}_pexp2(const bfloat16 *__restrict p0, const bfloat16 *__restrict p1, float *__restrict l, uint8_t *__restrict pp)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BOut<> po((bfp16ebs8 *)pp);
    aie::accum<accfloat, 32> l0(aie::load_v<32>(l)), l1(aie::load_v<32>(l + {{ 32 * (ND + 1) }}));
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < N; t++) {
        const aie::accum<accfloat, 32> r0 = {{ e192.parts.poly }}(aie::load_v<32>(p0), aie::load_v<32>(p0 + 32));
        const aie::accum<accfloat, 32> r1 = {{ e192.parts.poly }}(aie::load_v<32>(p1), aie::load_v<32>(p1 + 32));
        l0 = aie::add(l0, r0);
        l1 = aie::add(l1, r1);
        po.push(aie::concat(r0, r1).template to_vector<bfp16ebs8>());
        p0 += 64;
        p1 += 64;
    }
    aie::store_v(l, l0.template to_vector<float>());
    aie::store_v(l + {{ 32 * (ND + 1) }}, l1.template to_vector<float>());
}

// The P panel of a record (k: its index in the group's 4) for octet o's rows
__attribute__((noinline, minsize)) void {{ p }}_psoft(int32_t *__restrict ws, int k, int o, uint8_t *__restrict pp)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict s = AW(float, {{ a.sc }}) + 256 * o + 32 * k;
    {{ p }}_pexp1<1>(s);
    {{ p }}_pexp1<1>(s + 128);
    {{ p }}_pexp2<1>((const bfloat16 *)s, (const bfloat16 *)(s + 128), AW(float, {{ a.y }}) + {{ 64 * (ND + 1) }} * o + {{ 32 * ND }}, pp);
}

// The lanes of rows lo .. hi - 1 of the rows s0 .. s0 + 3 of a panel (8 per row)
[[clang::always_inline]] inline uint32_t {{ p }}_lanes(int s0, int lo, int hi)
{
    const int a = lo - s0 < 0 ? 0 : lo - s0, b = hi - s0 > 4 ? 4 : hi - s0;
    return a >= b ? 0u : (b == 4 ? ~0u : (1u << (8 * b)) - 1u) & ~((1u << (8 * a)) - 1u);
}

// The new K rows j0 .. j1 - 1 of kn into record ``odd`` of the pair at a ([d-block][record 2]), in place, and that
// record to ks
__attribute__((noinline, minsize)) void {{ p }}_kmerge(uint8_t *a, const uint8_t *__restrict kn, uint8_t *__restrict ks, int odd, int j0, int j1)
{
    const uint64_t mask = (uint64_t){{ p }}_lanes(4, j0, j1) << 32 | {{ p }}_lanes(0, j0, j1), em = (1u << j1) - (1u << j0);
    const aie::mask<64> m0 = aie::mask<64>::from_uint64(odd ? 0 : mask), m1 = aie::mask<64>::from_uint64(odd ? mask : 0);
    const aie::mask<64> e0 = aie::mask<64>::from_uint64(odd ? 0 : em), e1 = aie::mask<64>::from_uint64(odd ? em : 0);
    {
    al::BIn<> in((const bfp16ebs8 *)a), nw((const bfp16ebs8 *)kn);
    al::BOut<> ot((bfp16ebs8 *)a);
    AIE_LOOP_NO_UNROLL
    for (int d = 0; d < {{ ND }}; d++) {
        const v64bfp16ebs8 r0 = in.pop().to_native(), r1 = in.pop().to_native(), n = nw.pop().to_native();
        const aie::vector<int8, 64> nm(n.mantissa), ne(::insert(aie::zeros<int8, 64>(), 0, n.exponent));
        v64bfp16ebs8 x = r0;
        x.mantissa = aie::select(aie::vector<int8, 64>(r0.mantissa), nm, m0);
        x.exponent = extract_v8int8(aie::select(aie::vector<int8, 64>(::insert(aie::zeros<int8, 64>(), 0, r0.exponent)), ne, e0), 0);
        ot.push(al::BV(x));
        x.mantissa = aie::select(aie::vector<int8, 64>(r1.mantissa), nm, m1);
        x.exponent = extract_v8int8(aie::select(aie::vector<int8, 64>(::insert(aie::zeros<int8, 64>(), 0, r1.exponent)), ne, e1), 0);
        ot.push(al::BV(x));
    }
    }
    al::BIn<> rd((const bfp16ebs8 *)(a + 72 * odd));
    al::BOut<> os((bfp16ebs8 *)ks);
    AIE_LOOP_NO_UNROLL
    for (int d = 0; d < {{ ND }}; d++) {
        os.push(rd.pop());
        rd.pop();
    }
}

""" + _QPAN + r"""// A slot's K (256 fp32) into its row of the new rows ([d-block][position 8][8] bf16: dst its first values)
__attribute__((noinline)) void {{ p }}_krow(const float *__restrict k, bfloat16 *__restrict dst)
{
    for (int db = 0; db < {{ ND }}; db += 4) {
        const aie::vector<bfloat16, 32> b = aie::accum<accfloat, 32>(aie::load_v<32>(k + 8 * db)).template to_vector<bfloat16>();
        aie::store_v(dst + 64 * db, b.template extract<8>(0));
        aie::store_v(dst + 64 * db + 64, b.template extract<8>(1));
        aie::store_v(dst + 64 * db + 128, b.template extract<8>(2));
        aie::store_v(dst + 64 * db + 192, b.template extract<8>(3));
    }
}

// The new rows -> their panels
__attribute__((noinline)) void {{ p }}_kpan(const bfloat16 *__restrict kr, uint8_t *__restrict kn)
{
    al::BOut<> po((bfp16ebs8 *)kn);
    for (int db = 0; db < {{ ND }}; db++)
        po.push(aie::accum<accfloat, 64>(aie::load_v<64>(kr + 64 * db)).template to_vector<bfp16ebs8>());
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

// A record's V panels -> a pair of records [d-block][record 2] of it twice
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

// The lanes of rows s0 .. s0 + 3 of a record of positions 8 r .. past their own: row s (slot s, lanes 8 s ..) at position p + s
__attribute__((noinline, minsize)) uint32_t {{ p }}_past(int s0, int r, int p)
{
    uint32_t m = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < 4; q++) {
        int n = p + s0 + q - 8 * r + 1;  // (valid positions of the record for the row)
        n = n < 0 ? 0 : (n > 8 ? 8 : n);
        m |= ((0xFFu << n) & 0xFFu) << (8 * q);
    }
    return m;
}

// Tail object t of the step (t = 0: the record of its first position; 1: of its last, nothing more when one holds both):
// the record's rows ([d-block 32][position 8][8] bf16): its ring slot, in place, or (A_TREC, t = 0) from its V panels (the
// cache's pair of records from r & ~1 on) in the rows' scratch. The step's V rows in, the record's V panels up to the
// step's last position (the last core keeps them for its emission), y += P V (rows past their positions: the masked
// scores of ``aobj``). The rows stay for the emission (A_RW: the object is held).
__attribute__((noinline, minsize)) void {{ p }}_tobj(uint8_t *__restrict a, int32_t *__restrict ws, int t)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], r0 = p >> 3, r1 = (p + S - 1) >> 3;
    if (t && r1 == r0)
        return;
    const int r = t ? r1 : r0, P0 = 8 * r, last = p + S - 1 - P0 < 7 ? p + S - 1 - P0 : 7;
    bfloat16 *__restrict rows = st[{{ A_TREC }}] && !t ? AP(bfloat16, {{ A_RWS }}) : (bfloat16 *)a;
    st[{{ A_RW }} + t] = (int32_t)(uintptr_t)rows;
    if (st[{{ A_TREC }}] && !t)
        {{ p }}_vrows(a, rows, r0 & 1);
    AIE_LOOP_NO_UNROLL
    for (int j = P0 > p ? 0 : p - P0; j <= last; j++)
        {{ p }}_rowput(rows + 8 * j, AP(bfloat16, {{ A_VN }}) + {{ HD }} * (P0 + j - p));
    uint8_t *__restrict e = st[{{ A_ROLE }}] == {{ ROLE_V }} ? AP(uint8_t, {{ A_KM }}) + {{ REC_V }} * t : AP(uint8_t, {{ A_KN }});
    uint8_t *v = a + {{ ROWS_B }};  // (the object past its rows: read by no one)
    {{ p }}_venc(rows, e, last + 1);
    {{ p }}_vdup(e, v);
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    al::zero<{{ PS }}, 64>(pp);
    AIE_LOOP_NO_UNROLL
    for (int o = 0; o < {{ QC }}; o++)
        {{ p }}_psoft(ws, r & 3, o, pp + {{ PO }} * o);
    {{ p }}_pvs(v, v, pp, AW(float, {{ a.y }}));
}

// The new K into the open ones of a pair of records (r, r + 1) of a K object rec, the records to the merged ones' region
__attribute__((noinline, minsize)) void {{ p }}_kopen(uint8_t *rec, int32_t *__restrict ws, int r)
{
    const int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], rp = p >> 3, rl = (p + S - 1) >> 3;
    AIE_LOOP_NO_UNROLL
    for (int q = r; q < r + 2; q++) {
        if (q < rp || q > rl)
            continue;
        const int a = p - 8 * q, b = p + S - 8 * q;
        {{ p }}_kmerge(rec, AP(const uint8_t, {{ A_KN }}), AP(uint8_t, {{ A_KM }}) + {{ REC_K }} * (q != rp), q - r, a < 0 ? 0 : a, b > 8 ? 8 : b);
    }
}

// Records r, r + 1 from the first open one on: the scores of each row's positions past its own -16384 (P 0), of a record
// past the step's all of them, but for a record that keeps the scores of the first group's open one (``aobj``)
__attribute__((noinline, minsize)) void {{ p }}_kmask(int32_t *__restrict ws, int r)
{
    const int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], rp = p >> 3, rl = (p + st[{{ A_S }}] - 1) >> 3;
    const aie::vector<float, 32> lo = aie::broadcast<float, 32>(-16384.0f);
    AIE_LOOP_NO_UNROLL
    for (int q = r; q < r + 2; q++) {
        if (q < rp || (((rp ^ rl) >> 2) && q > rl && (q & 3) == 3))
            continue;
        const aie::mask<32> m0 = aie::mask<32>::from_uint32(q > rl ? ~0u : {{ p }}_past(0, q, p));
        const aie::mask<32> m1 = aie::mask<32>::from_uint32(q > rl ? ~0u : {{ p }}_past(4, q, p));
        float *__restrict so = AW(float, {{ a.sc }}) + 32 * (q & 3);
        AIE_LOOP_NO_UNROLL
        for (int set = 0; set < {{ SETS }}; set += 2) {
            aie::store_v(so + 128 * set, aie::select(aie::load_v<32>(so + 128 * set), lo, m0));
            aie::store_v(so + 128 * set + 128, aie::select(aie::load_v<32>(so + 128 * set + 128), lo, m1));
        }
    }
}

// The first group's scores in (from baselines 0): each row's reference m its maximum in them (a later score above it by up to
// 2^63.5: scores span less than that, P clamps there): the softmax needs no rescale. Baselines b = 192 - m: scores
// as A = s + b from QK on, this group's too
__attribute__((noinline, minsize)) void {{ p }}_m0(int32_t *__restrict ws)
{
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < {{ SETS }}; set++) {
        float *__restrict ss = AW(float, {{ a.sc }}) + 128 * set;
        const aie::vector<float, 32> m = al::max8(aie::max(aie::max(aie::load_v<32>(ss), aie::load_v<32>(ss + 32)),
                                                           aie::max(aie::load_v<32>(ss + 64), aie::load_v<32>(ss + 96))));
        const aie::vector<float, 32> b = aie::sub(aie::broadcast<float, 32>(192.0f), m);
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set, b);
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 128; k += 32)
            aie::store_v(ss + k, aie::add(aie::load_v<32>(ss + k), b));
    }
}

}  // namespace

extern "C" {

// A new X object: F_ATT (the step: norms, rotations; q, raw K, V objects and the cache follow), a main core's phase (its
// input objects and A objects: skipped in the main core's order, the norm-weight objects around their inputs: X and A
// are broadcast to both cores), or another core's first object (M_SKIP more follow). mw, mx: the main core's workspace
// and sum panels (its idle regions: AttnShape.ext)
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
    const int S = m[{{ M_NS }}], k = col - {{ NPH }} * al::div_small<{{ NPH }}>(col);
    st[{{ A_S }}] = S;
    st[{{ A_NQ }}] = S;
    st[{{ A_NQP }}] = 1;
    st[{{ A_NKV }}] = S;
    st[{{ A_NOBJ }}] = m[{{ M_ANA }}] - {{ HOLD }};
    st[{{ A_P }}] = m[{{ M_POS }}];
    st[{{ A_TREC }}] = m[{{ M_ATREC }}];
    st[{{ A_I }}] = 0;
    st[{{ A_ROLE }}] = k == 0 ? {{ ROLE_K }} : k == {{ NPH - 1 }} ? {{ ROLE_V }} : {{ ROLE_O }};
    st[{{ A_QS }}] = 0;
    st[{{ A_KS }}] = 0;
    st[{{ A_NO }}] = {{ NO_OBJ }};
{% for w, (r, off) in at %}
    st[{{ w }}] = (int32_t)((uintptr_t){{ r }} + {{ off }});
{% endfor %}
    al::copy<{{ 2 * HD + 64 * SMAX }}, 16>(AP(float, {{ A_NRM }}), (const float *)(x + {{ 2 * META }}));
    al::zero<{{ 32 * (ND + 1) * SETS }}, 32>(AW(float, {{ a.y }}));  // (the q rows' staging: rows past the step's 0)
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

// The next q object (slot s): its heads' q (normed, rotated, x log2(e) / 16: the scale in q_norm) into the staging rows
// (head h's octet h, row s)
__attribute__((minsize)) void {{ p }}_aq(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int s = st[{{ A_QS }}];
    st[{{ A_QS }}] = s + 1;
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    const float *__restrict nrm = AP(float, {{ A_NRM }});
    const float *__restrict cs = nrm + {{ 2 * HD }} + 64 * s;
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < {{ QC }}; h++)
        {{ p }}_norm_rope(pay + {{ 2 * HD }} * h, nrm, cs, cs + 32, AW(float, {{ a.y }}) + {{ HD }} * (8 * h + s));
}

// The q rows are in: the octets' Q panels (rows past the step's zero: abegin), y and l = 0, the rows' baselines 0
__attribute__((minsize)) void {{ p }}_aqpan(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict q = AW(float, {{ a.y }});
    {{ p }}_qpan<{{ QC if QC < 2 else 2 }}>(q, AP(uint8_t, {{ A_QA }}));
{% if QC > 2 %}
    {{ p }}_qpan<1>(q + {{ 16 * HD }}, AP(uint8_t, {{ A_QA }} + 1));
{% endif %}
    al::zero<{{ 32 * (ND + 1) * SETS }}, 32>(q);
    al::zero<{{ 64 * SETS }}, 32>(AW(float, {{ a.ml }}));
}

// The raw K, V object of the next slot: its K (normed, rotated) into the new rows, its V (bf16); after the last slot the
// new rows' panels
__attribute__((minsize)) void {{ p }}_akv(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int s = st[{{ A_KS }}], S = st[{{ A_S }}];
    st[{{ A_KS }}] = s + 1;
    const bfloat16 *__restrict kv = x + {{ 2 * META }} + {{ 2 * HD }} * al::div_small<{{ NPH }}>(col);
    const float *__restrict nrm = AP(float, {{ A_NRM }});
    const float *__restrict cs = nrm + {{ 2 * HD }} + 64 * s;
    float *__restrict kf = AW(float, {{ a.sc }});  // (the scores' region, before the cache)
    {{ p }}_norm_rope(kv, nrm + {{ HD }}, cs, cs + 32, kf);
    {{ p }}_krow(kf, AP(bfloat16, {{ A_KR }}) + 8 * ((st[{{ A_P }}] + s) & 7));
    al::copy<{{ HD }}, 64>(AP(bfloat16, {{ A_VN }}) + {{ HD }} * s, kv + {{ HD }});
    if (s + 1 == S)
        {{ p }}_kpan(AP(const bfloat16, {{ A_KR }}), AP(uint8_t, {{ A_KN }}));
}

// A step's A object: a cache object i (group i / 2's K object: the new K into its open records, the scores of its pairs of
// records, then ``asoft``: the group's softmax; its V object: y += P V of the records before the open ones), or a tail
// object (``tobj``: held)
void {{ p }}_aobj(uint8_t *__restrict a, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_I }}], p = st[{{ A_P }}], S = st[{{ A_S }}];
    st[{{ A_I }}] = i + 1;
    const int rp = p >> 3, rl = (p + S - 1) >> 3, t = i - st[{{ A_NOBJ }}];
    if (t >= 0) {
        st[{{ A_HP }} + t] = (int32_t)(uintptr_t)a;
        {{ p }}_tobj(a, ws, t);
        return;
    }
    const int g = i >> 1;
    if (i & 1) {
        {{ p }}_pvs(a, a + {{ 2 * REC_V }}, AW(uint8_t, {{ a.pp }}), AW(float, {{ a.y }}));  // (P 0 from the open records on)
        return;
    }
    float *__restrict sc = AW(float, {{ a.sc }});  // [set][record 4][8 q + j]
    const int open = 4 * g + 3 >= rp;
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 2; k++) {
        const int r = 4 * g + 2 * k;
        uint8_t *rec = a + {{ 2 * REC_K }} * k;
        if (open)
            {{ p }}_kopen(rec, ws, r);
        if (r <= rl) {
            {{ p }}_qk1<{{ QC if QC < 2 else 2 }}>(rec, AP(const uint8_t, {{ A_QA }}), AW(const float, {{ a.ml }}), sc + 64 * k);
{% if QC > 2 %}
            {{ p }}_qk1<1>(rec, AP(const uint8_t, {{ A_QA }} + 1), AW(const float, {{ a.ml }}) + 256, sc + 512 + 64 * k);
{% endif %}
        }
        if (open)
            {{ p }}_kmask(ws, r);
    }
    st[{{ A_SOFT }}] = g + 1;
}

// After a cache object: when it was a group's K object, the P panels of its records before the open ones (the first group:
// first the rows' baselines, ``m0``)
void {{ p }}_asoft(int32_t *__restrict ws)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int g = st[{{ A_SOFT }}] - 1, rp = st[{{ A_P }}] >> 3;
    if (g < 0)
        return;
    st[{{ A_SOFT }}] = 0;
    aie::set_rounding(aie::rounding_mode::conv_even);
    if (!g)
        {{ p }}_m0(ws);
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    const int n = rp - 4 * g;  // (the records before the open ones: P of those, 0 from there on)
    if (n >= 4) {
        float *__restrict sc = AW(float, {{ a.sc }});
        float *__restrict y = AW(float, {{ a.y }}) + {{ 32 * ND }};
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < {{ QC }}; o++) {
            {{ p }}_pexp1<8>(sc);
            {{ p }}_pexp2<4>((const bfloat16 *)sc, (const bfloat16 *)(sc + 128), y, pp);
            sc += 256;
            y += {{ 64 * (ND + 1) }};
            pp += {{ PO }};
        }
        return;
    }
    al::zero<{{ PS }}, 64>(pp);
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < n; k++) {
        AIE_LOOP_NO_UNROLL
        for (int o = 0; o < {{ QC }}; o++)
            {{ p }}_psoft(ws, k, o, pp + {{ PO }} * o + 72 * k);
    }
}

// The attention is through: 1 / l per row; st[A_NE] R objects (o, then the records)
__attribute__((minsize)) void {{ p }}_aend(int32_t *__restrict ws, const int32_t *__restrict pw)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    st[{{ A_EIDX }}] = 0;
    const int role = st[{{ A_ROLE }}];
    st[{{ A_NE }}] = st[{{ A_NO }}] + (role == {{ ROLE_V }} ? {{ 2 * (REC_V + ROWS_B) // (2 * C_LEN) }} :
                                       role == {{ ROLE_K }} ? {{ 2 * REC_K // (2 * C_LEN) }} : 0);
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < {{ SETS }}; set++) {
        const float *__restrict l = AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set + {{ 32 * ND }};
        const aie::vector<float, 16> sl = al::sum8(aie::load_v<32>(l));
        aie::vector<float, 16> il = aie::broadcast<float, 16>(1.0f);  // (one scalar inv per row)
        AIE_LOOP_UNROLL_FULL
        for (int q = 0; q < 4; q++)
            il[q] = aie::inv(sl[q]);
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set + 32, al::rows4(il));
    }
}

// The next R object: of o ([head QC][k block 32][slot][8] fp32 y / l: an octet's rows as they lie), then of the records (a
// chain's first column the K records of the step's first and last positions, merged; its last one their V panels up
// to the last position and rows, from the held tail objects: the same record twice when one holds both)
void {{ p }}_aemit(int32_t *__restrict ws, bfloat16 *__restrict r)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}], no = st[{{ A_NO }}];
    st[{{ A_EIDX }}] = e + 1;
    if (e >= no) {
        const int k = e - no, p = st[{{ A_P }}], S = st[{{ A_S }}], one = (p >> 3) == ((p + S - 1) >> 3);
        const uint8_t *__restrict src;
        if (st[{{ A_ROLE }}] == {{ ROLE_V }}) {
            const int second = k >= {{ (REC_V + ROWS_B) // (2 * C_LEN) }}, t = second && !one;
            const int kk = k - (second ? {{ (REC_V + ROWS_B) // (2 * C_LEN) }} : 0);
            src = kk < {{ REC_V // (2 * C_LEN) }} ? AP(const uint8_t, {{ A_KM }}) + {{ REC_V }} * t + {{ 2 * C_LEN }} * kk
                                                  : AP(const uint8_t, {{ A_RW }} + t) + {{ 2 * C_LEN }} * (kk - {{ REC_V // (2 * C_LEN) }});
        } else {
            const int second = k >= {{ REC_K // (2 * C_LEN) }};
            src = AP(const uint8_t, {{ A_KM }}) + {{ REC_K }} * (second && !one) + {{ 2 * C_LEN }} * (k - (second ? {{ REC_K // (2 * C_LEN) }} : 0));
        }
        aie::store_v(r, aie::load_v<{{ C_LEN }}>((const bfloat16 *)src));
        return;
    }
    const int o = e >> 5, db = e & 31;
    const float *__restrict y = AW(float, {{ a.y }}) + {{ 64 * (ND + 1) }} * o + 32 * db;
    const float *__restrict il = AW(float, {{ a.ml }}) + 128 * o + 32;
    aie::store_v((float *)r, al::mulff<32>(aie::load_v<32>(y), aie::load_v<32>(il)));
    aie::store_v((float *)r + 32, al::mulff<32>(aie::load_v<32>(y + {{ 32 * (ND + 1) }}), aie::load_v<32>(il + 64)));
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
    const int kc = QR == {{ QC }} ? 0 : {{ p }}_divs(i, S), s = i - S * kc, h0 = {{ QC }} * kc;
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
            const int c = 8 * e + i + k, hk = {{ p }}_divs(c, S), s = c - S * hk;
            g.insert(k, aie::load_v<8>(gt + {{ HD }} * (QR * s + (hk >> 5)) + 8 * (hk & 31)));
        }
        // (g bf16: mulff's products less g's lo part, 0)
        aie::store_v(o + 8 * i, al::mulc<16>(aie::load_v<16>((const float *)r + 8 * i), g).template to_vector<bfloat16>());
    }
}

}  // extern "C"
""")


# The main core's part of one pass: the gates of the query rows, per step (the attention core's L1 holds none)
_MAIN_ONE = JINJA_ENV.from_string(r"""
{% set exp2 = km.exp2(32, below="zero", inline=True) %}
{{ exp2.src }}

// A head's gates: sigmoid(g) = 1 / (1 + 2^(-g log2 e)) (bf16-level: the output is rounded to bf16; -g log2 e: al::log2e's
// products of the bf16 g, negated)
__attribute__((noinline)) void {{ p }}_gates(const bfloat16 *__restrict g, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int j = 0; j < {{ HD // 32 }}; j++) {
        const aie::vector<bfloat16, 32> gb = aie::load_v<32>(g);
        const aie::vector<float, 32> e = {{ exp2 }}(aie::mac(aie::mul(gb, aie::broadcast<bfloat16, 32>(-1.4453125f)), gb,
                                                     aie::broadcast<bfloat16, 32>(0.00262451171875f)).template to_vector<float>());
        aie::store_v(out, aie::accum<accfloat, 32>(aie::inv(aie::add(e, aie::broadcast<float, 32>(1.0f)))).template to_vector<bfloat16>());
        g += 32;
        out += 32;
    }
}

namespace {

// An object of another core (the main core's begin): an F_ATT object: its q objects' gates and its o's R objects
// (the state words MG_*)
[[clang::always_inline]] inline void {{ p }}_gbegin(const int32_t *__restrict m, int32_t *__restrict st, int32_t col)
{
    const int att = m[{{ M_FLAGS }}] & {{ F_ATT }};
    st[{{ MG_Q }}] = att ? m[{{ M_NS }}] : 0;
    st[{{ MG_I }}] = 0;
    st[{{ MG_NO }}] = att ? {{ NO_OBJ }} : 0;
    st[{{ MG_E }}] = 0;
}

}  // namespace

extern "C" {

// A skipped X object: a q object: its heads' sigmoid(gate) into the gate table gt ([row 8 h + s][256] bf16 in the helper's x
// panels, idle in the attention)
__attribute__((minsize)) void {{ p }}_mgate(const bfloat16 *__restrict x, int32_t *__restrict ws, bfloat16 *__restrict gt)
{
    int32_t *__restrict st = ws;
    const int i = st[{{ MG_I }}];
    if (i >= st[{{ MG_Q }}])
        return;
    aie::set_rounding(aie::rounding_mode::conv_even);
    st[{{ MG_I }}] = i + 1;
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < {{ QC }}; h++)
        {{ p }}_gates(pay + {{ 2 * HD }} * h + {{ HD }}, gt + {{ HD }} * (8 * h + i));
}

// An R object -> a C object: of o (fp32 y / l [slot 8][8] of a head's k block: the gates' products, bf16), or a record's
// (bf16)
void {{ p }}_mrelay(const bfloat16 *__restrict r, bfloat16 *__restrict o, int32_t *__restrict ws, const bfloat16 *__restrict gt)
{
    int32_t *__restrict st = ws;
    const int e = st[{{ MG_E }}];
    st[{{ MG_E }}] = e + 1;
    if (e >= st[{{ MG_NO }}]) {
        aie::store_v(o, aie::load_v<{{ C_LEN }}>(r));
        return;
    }
    aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *__restrict g = gt + {{ 8 * HD }} * (e >> 5) + 8 * (e & 31);
    aie::vector<bfloat16, 64> gv;
{% for s in range(8) %}
    gv.insert({{ s }}, aie::load_v<8>(g + {{ HD * s }}));
{% endfor %}
    // (g bf16: mulff's products less g's lo part, 0)
    aie::store_v(o, al::mulc<64>(aie::load_v<64>((const float *)r), gv).template to_vector<bfloat16>());
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
        self.sub1 = slots if self.one else min(SMAX, (MAX_ROWS if cfg.moe else MAX_ROWS_G) // (self.nph * self.qc))
        assert self.sub1 >= 1
        # (more slots than a chain of 4 holds: its halves, each with half the rows, on two streams)
        self.halves = not self.one and self.nph == 4 and self.sub1 < SMAX
        self.a_bcast = tuple((c, c + 2) for c in range(COLS) if c % 4 < 2) if self.halves else ()
        self.sub = SMAX if self.halves else self.sub1
        self.u = 1 if self.one else -(-(slots - 1) // self.sub) + 1  # (a step's sub-steps at most)
        self.rows = 8 * self.qc if self.one else self.sub1 * self.nph * self.qc  # (a sub-step's query rows at most)
        self.main_gates = self.one or self.rows > MAX_ROWS
        assert not (self.halves and self.main_gates)  # (``_MAIN_SRC``: a chain's last column gates o)
        self.octets = -(-self.rows // 8)
        self.sets = 2 * self.octets
        self.po = PO
        self.ps = rup(self.po * self.octets, 64)  # (a group's P panels: a panel stream starts 64-byte aligned)
        assert A_QA + self.octets <= A_HP and self.octets <= 3  # (pvs)
        self.c_len, self.a_bytes = c_len, a_bytes
        self.r_bytes = 2 * c_len * (2 if self.main_gates else 1)  # (an R object: o fp32 when the main core gates it)
        self.eps = cfg.rms_norm_eps
        self.groups = -(-max_positions // POS_PER_GROUP)
        self.max_positions = self.groups * POS_PER_GROUP
        self.head_bytes = OBJS * a_bytes * self.groups  # a KV head's cache
        self._plan()

    def _plan(self) -> None:
        """The workspace (the core's L1 next to its two A buffers, X buffer, R buffers, token and stack) and the regions
        in the main core's memory (``ext``: region, offset; "mw" its workspace past its residual, "mx" its sum panels):
        the gates there (unless the main core applies them), the norms and the step's V too where the octets' Q panels
        then fit the core's L1, else the Q panels that do not."""
        y = ("y", 4 * 32 * (ND + 1) * self.sets)
        own = [("st", 4 * (A_KM + 1 if self.one else A_WORDS)), y, ("ml", 4 * 64 * self.sets), ("sc", 4 * 128 * self.sets),
               ("pp", self.ps)]
        self.gate_tab = not self.one and not self.main_gates  # (the sigmoid's table in the core's L1)
        fixed = 2 * self.a_bytes + XB + 2 * self.r_bytes + 64 + STACK + (SIGMOID.table_words().nbytes if self.gate_tab else 0)
        self.ext_w, self.ext_x = (EXT_W8, EXT_X8) if self.one else (EXT_W, EXT_X)
        free = {"mw": self.ext_w, "mx": self.ext_x}
        self.ext: dict[str, tuple[str, int]] = {}

        def put(name: str, n: int, order: tuple[str, ...] = ("mw", "mx")) -> None:
            r = next(r for r in order if free[r] >= n)
            self.ext[name] = (r, (self.ext_w if r == "mw" else self.ext_x) - free[r])
            free[r] -= n

        cold = [("nrm", 4 * (2 * HD + 64 * self.smax)), ("vn", 2 * HD * self.smax)]
        if self.one:
            for name, n in cold:
                put(name, n)
            put("rws", ROWS_B)
            put("kr", ROWS_B, ("mx", "mw"))
            put("km", 2 * REC_K, ("mx", "mw"))
            pan = 72 * ND
            own += [("qa0", min(self.qc, 2) * pan), ("qa1", pan)] if self.qc > 2 else [("qa0", self.qc * pan)]
            own.append(("kn", REC_K))
        else:
            own.append(("kb", max(9 * ND * self.smax, REC_V)))  # (the slots' K blocks; aemit: a record's V panels)
            qa = [(f"qa{o}", 2 * 72 * ND) for o in range(self.octets)]
            if not self.main_gates:
                put("gs", 2 * self.rows * HD)
                put(*cold[1], ("mx",))
                own += cold[:1] + qa
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
        self.mx_bytes = self.ext_x - free["mx"]  # (of the main core's sum panels)
        self._ws = regions(own)
        assert self._ws["size"] + fixed <= L1, (self._ws["size"], fixed)
        assert 4 * 128 * self.sets >= 4 * HD  # (kf: the scores' region)

    def at(self, tail: int) -> list[tuple[int, tuple[str, int]]]:
        """The address state words abegin sets: (word, (base, offset)); tail: the main core's workspace offset past its
        residual."""
        bases = {"mw": ("mw", tail), "mx": ("mx", 0)}
        out = []
        names = [(A_NRM, "nrm"), (A_VN, "vn")] + [(A_GS, "gs")] * (not self.main_gates)
        names += [(A_RWS, "rws"), (A_KR, "kr"), (A_KN, "kn"), (A_KM, "km")] if self.one else []
        for w, name in names + [(A_QA + o, f"qa{o}") for o in range((1 + (self.qc > 2)) if self.one else self.octets)]:
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
        if split == SPLIT_HALF:
            no = 4 * n * 2 * self.qc * (c & 1)
        else:
            no = 32 * self.qc if self.one else 4 * n * ((self.nph * self.qc if role == ROLE_V else 0) if split else self.qc)
        nr = self.records(n)
        k = REC_K if self.one else 2 * REC_K  # (one pass: a K record; sub-steps: its pair)
        return no + {ROLE_K: nr * k, ROLE_V: nr * (REC_V + ROWS_B), ROLE_O: 0}[role] // (2 * self.c_len)

    @staticmethod
    def splits(g: int) -> bool:
        """Whether a sub-step whose slots are in group g splits the cache along the chains (from 4 groups on: the
        same for each of its positions in every step)."""
        return g >= 3

    def records(self, n: int) -> int:
        """A sub-step of n slots' records out and tail objects: of its first and last positions (one pass: the same
        record twice when one holds both; sub-steps of one slot: one)."""
        return 2 if self.one or n > 1 else 1

    def split_of(self, g: int, n: int) -> int:
        """A sub-step's M_SPLIT: n slots in group g."""
        return 0 if not self.splits(g) else SPLIT_HALF if n > self.sub1 else SPLIT_CHAIN

    def bounds(self, g: int, split: int = SPLIT_CHAIN) -> list[int]:
        """A split sub-step in group g: the groups of the cores, core k's from bounds[k] to bounds[k + 1] (SPLIT_HALF: of
        each half of the chain)."""
        G, n = g + 1, 2 if split == SPLIT_HALF else self.nph
        assert G >= n
        return [G * k // n for k in range(n + 1)]

    def workspace(self) -> dict[str, int]:
        """Byte offsets of the core's workspace regions (and "size"). y: per set its blocks [db][row 4 x 8] fp32, then
        l (before the panels: the q rows [row][256] fp32); sc: per set the group's scores [record 4][row 4 x 8] (before
        the cache: a slot's K, fp32); pp: a group's P panels (``ps`` bytes) [octet][P0 | P1 | P2 | P3]; qa<o>: the Q panels, [d-block][octet 2]
        (qa1: octets 1, 2 of three; qa0, done with, in record mode the open record's rows; qa1 a tail's V pair)."""
        return dict(self._ws)

    def render(self, p: str, consts: dict[str, int], *, tail: int, packed8: bool = False) -> str:
        """packed8: the main core's inputs packed in a design of 8 slots (``decode.design.M_SP``)."""
        names = {"ws": "ws", "mw": "mw", "mx": "mx"}
        at = [(w, (names[r], off)) for w, (r, off) in self.at(tail)]
        v = dict(p=p, a=Attr(self.workspace()), at=at, QC=self.qc, NPH=self.nph, SETS=self.sets, NO=self.octets, SMAX=self.smax,
                 HD=HD, ND=ND, HOLD=HOLD, PS=self.ps, PO=self.po, REC_K=REC_K, REC_V=REC_V, ROWS_B=ROWS_B, eps=repr(self.eps),
                 C_LEN=self.c_len, OBJ=self.a_bytes, PAIR_K=2 * REC_K, NO_OBJ=32 * self.qc,
                 GATE_MAIN=self.main_gates, PACKED8=packed8, HALF=self.halves,
                 SPLIT_HALF=SPLIT_HALF,
                 **(consts | {"F_ATT": F_ATT} | {k: v for k, v in globals().items() if k.startswith(("A_", "ROLE_", "MG_"))}))
        if self.one:
            return _ONE.render(**v) + _MAIN_ONE.render(**v)
        return _SRC.render(**v) + (_MAIN_SRC.render(**v) if self.main_gates else "")


# ----------------------------------------------------------------------
# Host side: the RoPE angles
# ----------------------------------------------------------------------


def rope_cs(cfg: QwenConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin (n, 32) fp32 of positions' angles (``QwenConfig.rope_theta``: (n,) text positions or (n, 3) (T, H, W))."""
    th = cfg.rope_theta(positions).astype(np.float64)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
