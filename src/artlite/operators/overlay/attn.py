"""The attention cores of the decode overlays on host-built transactions (Gemma 4: ``gemma4/decode.py``, Llama:
``llama/decode.py``): per step S <= 4 token slots at positions p .. p + S - 1, a model's head dims (one or two kinds of
layer: Gemma's sliding-window 256 / full 512, Llama's 64 or 128), RoPE on the dim pairs (i, i + hd / 2) of i < rot
(per kind), scores in log2 units: q k times a constant per layer (the F_ATT object's: Gemma q_norm k_norm log2(e) with q
and k RMS-normed per head, ``Attn(qk_norm=True)``; Llama log2(e) / sqrt(hd)), v as it is or RMS-normed without a weight
(``v_norm``).

The KV cache of a (layer, KV head) is groups of objects of ``a_bytes`` (``overlay.kv``): ``kp`` K objects (records of 8
positions: [K panels ND] per record, kp = 2: [hi panels ND | lo panels ND], kp = 1: one bfp16 block per 8 dims) and the
V object of their records ([record][ND panels]), linear in the positions (Gemma's sliding-window caches after a guard of
the window's groups: the prefill's attention reads a window from 16 groups before a position, ``gemma4.overlay.GUARD``).

A V block spans 8 positions: what position q sees of a record is its V panels encoded from the exact values (bf16) of
its positions up to q (all 8 when q is past it), so no position's V depends on later positions, on the step's other
slots or on an earlier, rolled-back step (rounding a record again whenever a position joins does all three). The open
records - those of the step's positions p .. p + S - 1 - need their exact values: each (layer, KV head) keeps a ring of
two slots next to the cache (record r in slot r & 1: its 8 positions' values [d-block][position 8][8] bf16,
``overlay.vring``), and the chain's end gets the slots of records p >> 3 and (p + S - 1) >> 3 after its range (the tail
objects, ``tails``; after a prefill, which writes no values, the first one is the record's V panels instead, whose
values are exact: ``M_ATREC``); per slot it encodes the open record up to the slot's position and adds its P V there,
with the final running maxima.

The ``chain`` columns of a KV head (Gemma 4: columns 4 h .. 4 h + 3; Llama 1: column h) attend together, with all
query heads of the KV head (``M_ANH``: Gemma 4, its drafter 2, Llama 3 or 4; a set of 4 query rows holds row nh s + h
of slot s, head h). A layer's attention runs in sub-steps (``overlay.vring.substeps``): the slots of one pair of
records (16 positions: P V adds records 2 k, 2 k + 1 of a group together), at most as many as L1 holds query rows for
(``Attn.rows``: 16 up to head_dim 256, 8 at 512); the columns' ranges of the cache's groups follow from the slots'
group alone (``overlay.kv.ranges``: even shares, a sliding-window cache from the window's guard on). So a slot's
attention - which column reads which groups, which partials merge, which records add together in which order - is
the same in every step it can be part of, and its numbers too: a step of S slots gives each the logits of a step of
one. Softmax with a running maximum per row (masked positions add exactly nothing; a row's numbers depend on its own
scores only); at the end each column merges the partial (m, l, y) of its west neighbour in the chain (shared memory,
ordered by token ObjectFifos) into its own, so the chain's last column holds the whole and emits o = y / l on D into
the sub-step's slots of the O projection's packed input ([k block][slot SP][8] bf16, heads in a row:
``overlay.decode.packed_slots``). The last column holds the newest group: of a layer with its own cache it inserts the
sub-step's K (rotated, normed if ``qk_norm``; bfp16 blocks, kp per 8 dims) into their records as they pass and its V
rows into the open records' rows at the tail objects, and emits the K records of the sub-step's first and last
positions as the K objects come, then per open record its V panels and its ring slot (``overlay.kv.emit_order``) on D
before o; the next sub-step reads them from DDR. D objects: ``d_len`` bf16 (min(256, head_dim) bytes: every record,
ring slot and o is a whole number of them).

Per step the core gets on X (broadcast with the main core's; it skips the main core's objects and the main core its):
the token objects (``F_TOK``, t = SMAX kind + a per layer kind and slot a: of the sub-step from a on its first
position and slot, its slots, the columns' ranges and A objects, and the step's cos / sin of the rotated pairs per
slot, fp32; for the main cores the A objects of the sub-step that ends at a, ``M_TOK_NG``), then per attention layer
and sub-step the KV head's q objects of its slots ([head nh][hd] bf16; the first one's metadata words: the ``F_ATT``
object - head_dim, kind, own cache, the score constant, its token) and, for a layer with its own cache, their raw K / V
objects ([K hd | V hd] bf16): the main cores skip them one at a time and after the last the cache's objects (its
M_NASKIP: the token of its slot). On A (broadcast with the main core's) its range of the cache (and the chain's end
its tail objects).
"""

import math

import numpy as np
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

from ..base import JINJA_ENV
from ..design import COLS
from ..util import Attr, bf16_bits, bf16_hilo, regions, rup
from .kv import Records
from .vring import ring_rows, substeps, tails

__all__ = ["ring_rows", "substeps", "tails"]  # (the shared helpers, re-exported for the overlays and tests)

SMAX = 4  # token slots
F_ATT = 1 << 17  # M_FLAGS of a sub-step's first q object (the main cores skip it, the cache by its token)
U = 3  # sub-steps of a layer kind per step at most (``substeps``: slots of 2 at head_dim 512)

# metadata words (the shared overlay's M_MODEL = 24 on): an F_TOK object's last cached position before the step (the
# masks: a slot s attends to positions <= M_POS + s); an F_ATT object's head_dim, kind (its index in the head dims),
# own cache (the layer computes K / V), the score constant (bf16 hi, lo bits), the window (0: none), the query heads
# per KV head; a token object's per-column first positions (M_TOK_G0 + c) next to the shared overlay's M_TOK_NG + c
# (the columns' cache A objects)
M_POS, M_AHD, M_AKIND, M_AOWN, M_QC_HI, M_QC_LO, M_AWIN, M_ANH = range(24, 32)
M_TOK_NA, M_TOK_G0 = 32, 56  # (per column: its attended cache objects | its A objects << 16, its range's first position)
M_TOK_A = 44  # a token object's first slot of the step (its slots: a .. a + M_NS - 1)
M_ATREC = 32  # an F_ATT object's: the first tail object is the open record's V panels (the cache's last writer was the
# prefill), not its ring slot
M_ATOK = 33  # an F_ATT object's token (its sub-step's: the shared overlay's M_NASKIP = -1 - M_ATOK)

# state words of the attention core (the first ones are trip counts of its program)
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NQ, A_NQP, A_NKV, A_NOBJ, A_NE, A_NO, A_NPS, A_XL, A_P, A_S, A_I, A_EIDX,
 A_HD, A_KIND, A_OWN, A_WIN, A_QC, A_NH, A_SA, A_POS, A_MERGE, A_CEND, A_INS, A_QS, A_KS, A_EOFF0, A_EN0, A_EOFF1,
 A_EN1, A_EREC) = range(34)
A_NHM = 34  # (query heads per KV head not a power of 2) 64 / nh rounded up: a row's slot (r nhm) >> 6 for r < 16
A_LPER = 35  # log2 of the strips of a main core phase's input object (``overlay.decode.packed_slots``)
A_NATT = 36  # the attended cache objects (the others after them are skipped)
A_YNZ = 37  # y holds P V terms (else zero: a rise of the running maxima needs no rescale)
A_TREC = 38  # (M_ATREC)
A_TOK = 40  # per token t: the column's range's first position, its cache objects, its attended ones, p << 8 | a << 4 | S
# (a token of no sub-step: S = 0, no cache objects)
A_WORDS = A_TOK + 4 * 2 * SMAX  # (two kinds of layer at most)
assert A_EREC < A_NHM < A_LPER < A_NATT < A_YNZ < A_TREC < A_TOK
# the event windows of a stats build (the decode operator's win=): the attention core's kernels
WINS = {"abegin": 11, "aq": 12, "aqpan": 13, "akv": 14, "kobj": 15, "vobj": 16, "arec": 17, "aend": 18, "aemit": 19,
        "put_k": 20, "tobj": 21, "scores": 22, "pv": 23,
        # (spans: an attention step from its region's first object to its end; the first cache object's wait from
        # there; the merge's wait for the west neighbour from the last cache object)
        "aspan": 40, "afirst": 41, "achain": 42}


_SRC = JINJA_ENV.from_string(r"""
// The attention core's workspace (offsets in bytes: overlay.attn.Attn.workspace())
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// The event window of a stats build (the decode operator's win=): event0 / event1 around region K
template <int K>
struct {{ p }}_AWin {
    [[clang::always_inline]] {{ p }}_AWin() { if constexpr (K == {{ WIN }}) event0(); }
    [[clang::always_inline]] ~{{ p }}_AWin() { if constexpr (K == {{ WIN }}) event1(); }
};

{% if QK_NORM %}
// x (hd bf16) -> out (hd fp32) = x / rms(x), then the pairs (i, i + hd / 2) of i < rot rotated by cs (cos rot | sin rot,
// fp32). (One copy for every head dim: runtime trip counts, not a hot loop)
__attribute__((noinline, minsize)) void {{ p }}_nrope(const bfloat16 *__restrict x, const float *__restrict cs, float *__restrict out,
                                             int hd, int rot)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < hd; i += 64)
        ss = aie::mac_square(ss, aie::load_v<64>(x + i));
    const float sum = aie::reduce_add(ss.template to_vector<float>());
    const float inv = al::inv_rms(sum, {{ by_hd(INV_HD) }}, {{ eps }}f);
    aie::vector<bfloat16, 64> ih, il;  // (x bf16: x 1 / rms as its two products with 1 / rms's parts)
    al::bf16_hilo<64>(inv, ih, il);
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < hd; i += 64) {
        const aie::vector<bfloat16, 64> xv = aie::load_v<64>(x + i);
        aie::store_v(out + i, aie::mac(aie::mul(xv, ih), xv, il).template to_vector<float>());
    }
{% else %}
// x (hd bf16) -> out (hd fp32), the pairs (i, i + hd / 2) of i < rot rotated by cs (cos rot | sin rot, fp32)
__attribute__((noinline, minsize)) void {{ p }}_nrope(const bfloat16 *__restrict x, const float *__restrict cs, float *__restrict out,
                                             int hd, int rot)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < hd; i += 32)
        aie::store_v(out + i, aie::accum<accfloat, 32>(aie::load_v<32>(x + i)).template to_vector<float>());
{% endif %}
    const int half = hd >> 1;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < rot; i += 32) {
        const aie::vector<float, 32> a = aie::load_v<32>(out + i), b = aie::load_v<32>(out + half + i);
        const aie::vector<float, 32> c = aie::load_v<32>(cs + i), s = aie::load_v<32>(cs + rot + i);
        aie::accum<accfloat, 32> o0, o1;
        al::rot2(a, b, c, s, o0, o1);
        aie::store_v(out + i, o0.template to_vector<float>());
        aie::store_v(out + half + i, o1.template to_vector<float>());
    }
}

// 2^x of 64 lanes (al::exp2f: one copy for the softmax and the merge; two records or two partials at a time, two
// independent chains in one body: 88 cycles, 70 for 32 lanes): exactly 0 for the masked scores (-1e30)
__attribute__((noinline)) aie::vector<float, 64> {{ p }}_exp2(aie::vector<float, 64> x)
{
    return al::exp2f<64, true>(x);
}

// Q (the set's panels) against {{ DB }} d-blocks of K panels: two MMUL chains (even, odd d-blocks), two panel streams
[[clang::always_inline]] inline aie::accum<accfloat, 64> {{ p }}_qk(const uint8_t *__restrict k, const uint8_t *__restrict qa)
{
    al::MM8 a0(aie::zeros<accfloat, 64>()), a1(aie::zeros<accfloat, 64>());
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int d = 0; d < {{ DB // 2 }}; d++) {
        const al::BV q0 = qs.pop(), k0 = ks.pop(), q1 = qs.pop(), k1 = ks.pop();
        a0.mac(q0, aie::op_transpose(k0));
        a1.mac(q1, aie::op_transpose(k1));
    }
    return aie::add(a0.to_accum(), a1.to_accum());
}

// y = yi + yi (a0 - 1) + yw a1 over n blocks of a set's rows (its y and l: nd + 1; al::merge): the chain's merge, a
// rescale (a1 = 0). A row whose factor is 1 keeps its y and l exactly.
__attribute__((noinline)) void {{ p }}_mrg(const float __aie_dm_resource_a *__restrict yi, float *__restrict y,
                                           const float __aie_dm_resource_b *__restrict yw, aie::vector<float, 32> a0,
                                           aie::vector<float, 32> a1, int n)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::merge{{ MINT }}(yi, y, yw, a0, a1, n);
}

{% if ONE_HD %}
// Scores of n records (8 positions each, their K panels in order: {{ KP }} x {{ ND }} per record{{ ", hi then lo" if KP == 2 else "" }}) against a set's
// Q panels: so[32 r + 8 q + j] (row q < 4, position j), log2 units. Per record (and K part) the Q panels again (a stream
// opened per pass), the records' K panels one stream; two MMUL chains (even, odd d-blocks), unrolled up to 16 MMULs
// per record (program memory)
__attribute__((noinline)) void {{ p }}_scores(const uint8_t *__restrict k, const uint8_t *__restrict qa, float *__restrict so,
                                              int n)
{
    {{ p }}_AWin<{{ WINS['scores'] }}> win_;
    al::BInR<aie_dm_resource::a> ks((const bfp16ebs8 *)k);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int r = 0; r < n; r++) {
        al::MM8 a0(aie::zeros<accfloat, 64>()), a1(aie::zeros<accfloat, 64>());
{% for h in range(KP) %}
        {
            al::BInR<aie_dm_resource::b> qs((const bfp16ebs8 *)qa);
{% if KP * ND <= 16 %}
            AIE_LOOP_UNROLL_FULL
{% else %}
            AIE_PREPARE_FOR_POSTPIPELINING
{% endif %}
            for (int d = 0; d < {{ ND // 2 }}; d++) {
                const al::BV q0 = qs.pop(), k0 = ks.pop(), q1 = qs.pop(), k1 = ks.pop();
                a0.mac(q0, aie::op_transpose(k0));
                a1.mac(q1, aie::op_transpose(k1));
            }
        }
{% endfor %}
        const aie::vector<float, 64> c = aie::add(a0.to_accum(), a1.to_accum()).template to_vector<float>();
        aie::store_v(so, aie::add(c.template extract<32>(0), c.template extract<32>(1)));  // q rows hi + lo
        so += 32;
    }
}
{% else %}
// Scores of one record (8 positions, nd d-blocks{{ ": K hi, then K lo" if KP == 2 else "" }}) against a set's Q panels: so[8 r + j] (row r < 4,
// position j), log2 units ({{ DB }} d-blocks per pass: head_dim 512 in two)
__attribute__((noinline)) void {{ p }}_scores(const uint8_t *__restrict k, const uint8_t *__restrict qa, float *__restrict so,
                                              int nd)
{
    {{ p }}_AWin<{{ WINS['scores'] }}> win_;
    aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < 72 * nd; h += {{ 72 * DB }})  // (one copy of the products, program memory)
{% if KP == 2 %}
        acc = aie::add(acc, aie::add({{ p }}_qk(k + h, qa + h), {{ p }}_qk(k + 72 * nd + h, qa + h)));
{% else %}
        acc = aie::add(acc, {{ p }}_qk(k + h, qa + h));
{% endif %}
    const aie::vector<float, 64> c = acc.template to_vector<float>();
    aie::store_v(so, aie::add(c.template extract<32>(0), c.template extract<32>(1)));  // q rows hi + lo
}
{% endif %}

// y[db][8 r + i] += P V of two records' nd d-blocks (records 2 k, 2 k + 1 of a group, or one with a zero P panel:
// ``substeps``; v0, v1: their first V panels, V panels also where a P panel is zero: other bytes read as panels made
// 0 x V NaN; pp: their two P panels; yl, ys: y)
__attribute__((noinline)) void {{ p }}_pv(const uint8_t *__restrict v0, const uint8_t *__restrict v1, const uint8_t *__restrict pp,
                                          const float *__restrict yl, float *__restrict ys, int nd)
{
    {{ p }}_AWin<{{ WINS['pv'] }}> win_;
    al::BIn<> ps((const bfp16ebs8 *)pp);
    const al::BV P0 = ps.pop(), P1 = ps.pop();
    al::BInR<aie_dm_resource::a> vs0((const bfp16ebs8 *)v0);
    al::BInR<aie_dm_resource::b> vs1((const bfp16ebs8 *)v1);
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT({{ ND_MIN }})
    for (int d = 0; d < nd; d++) {
        al::MM8 c(aie::accum<accfloat, 64>(aie::concat(aie::load_v<32>(yl), aie::zeros<float, 32>())));
        c.mac(P0, aie::op_transpose(vs0.pop()));
        c.mac(P1, aie::op_transpose(vs1.pop()));
        const aie::vector<float, 64> o = c.to_accum().template to_vector<float>();
        aie::store_v(ys, aie::add(o.template extract<32>(0), o.template extract<32>(1)));
        yl += 32;
        ys += 32;
    }
}

// The exponent bytes of a K insert: n bytes of ke, 72 B apart from d
__attribute__((noinline, minsize)) void {{ p }}_put_ke(uint8_t *__restrict d, const uint8_t __aie_dm_resource_b *__restrict ke, int n)
{
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < n; db++) {
        *d = *ke;
        d += 72;
        ke += 1;
    }
}

// The position's K blocks ({{ KP }} nd{{ ": hi nd, then lo nd" if KP == 2 else "" }}; km their mantissas, 64 B per 8 blocks, ke their exponents) into
// slot j0 of its record's panels, in place: per panel the mantissas of row j0 by a select of the new block broadcast
// to the 8 rows, then the exponent bytes
__attribute__((noinline)) void {{ p }}_put_k(uint8_t *__restrict rec, const int8_t *__restrict km, const uint8_t *__restrict ke,
                                             int j0, int nd)
{
    {{ p }}_AWin<{{ WINS['put_k'] }}> win_;
    {
        al::BIn<aie_dm_resource::a> pi((const bfp16ebs8 *)rec);
        al::BOut<> po((bfp16ebs8 *)rec);
        const aie::mask<64> row = aie::mask<64>::from_uint64(0xFFull << (8 * j0));
        const int8_t __aie_dm_resource_b *__restrict pm = (const int8_t __aie_dm_resource_b *)km;
        AIE_LOOP_NO_UNROLL
        for (int g = 0; g < (nd >> {{ 4 - KP }}); g++) {
            const v16int32 nm = (v16int32)aie::load_v<64>(pm).to_native();
            pm += 64;
            AIE_LOOP_UNROLL_FULL
            for (int d = 0; d < 8; d++) {
                v64bfp16ebs8 b = (v64bfp16ebs8)pi.pop();
                b.mantissa = (v64int8)aie::select(aie::vector<int8_t, 64>(b.mantissa),
                                                  aie::vector<int8_t, 64>((v64int8)broadcast_elem_s64(nm, d)), row);
                po.push(al::BV(b));
            }
        }
    }
    {{ p }}_put_ke(rec + 9 * j0, (const uint8_t __aie_dm_resource_b *)ke, {{ KP }} * nd);
}

// A record's V panels -> its rows [d-block nd][position 8][8] bf16 (al::v_rows)
__attribute__((noinline, minsize)) void {{ p }}_vrows(const uint8_t *__restrict rec, bfloat16 *__restrict rows, int nd)
{
    al::v_rows(rec, rows, nd);
}

// A position's V (src: 8 nd bf16) into its row of a record's rows (dst: the row's first 8 values)
__attribute__((noinline)) void {{ p }}_rowput(bfloat16 *__restrict dst, const bfloat16 *__restrict src, int nd)
{
    al::v_put{{ VPUT }}(dst, src, nd);
}

// rows [d-block nd][position 8][8] bf16, those from n on as zeros -> the record's V panels (al::v_enc)
__attribute__((noinline)) void {{ p }}_venc(const bfloat16 *__restrict rows, uint8_t *__restrict rec, int nd, int n)
{
    al::v_enc{{ MINT }}(rows, rec, nd, n);
}

{% if NH_POW2 %}
// The lanes of a set's rows (row 4 set + q: slot (4 set + q) >> lnh) to mask at the record of positions P0 .. P0 + 7:
// past the slot's position, or (a window win) before its window, or before position 0
__attribute__((noinline, minsize)) uint32_t {{ p }}_mask(int set, int P0, int p, int S, int lnh, int win)
{% else %}
// The lanes of a set's rows (row 4 set + q: slot ((4 set + q) nhm) >> 6) to mask at the record of positions P0 .. P0 +
// 7: past the slot's position, or (a window win) before its window, or before position 0
__attribute__((noinline, minsize)) uint32_t {{ p }}_mask(int set, int P0, int p, int S, int nhm, int win)
{% endif %}
{
    uint32_t m = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < 4; q++) {
{% if NH_POW2 %}
        const int s = (4 * set + q) >> lnh;
{% else %}
        const int s = ((4 * set + q) * nhm) >> 6;
{% endif %}
        if (s >= S)
            continue;
        int hi = p + s - P0 + 1;  // (positions P0 + j, j < hi: not past the slot's)
        hi = hi < 0 ? 0 : (hi > 8 ? 8 : hi);
        int lo = (win ? p + s - win + 1 : 0) - P0;
        lo = lo < -P0 ? -P0 : lo;
        lo = lo < 0 ? 0 : (lo > 8 ? 8 : lo);
        const uint32_t valid = ((1u << hi) - 1u) & ~((1u << lo) - 1u);
        m |= (~valid & 0xFFu) << (8 * q);
    }
    return m;
}

// The records of positions p and p + S - 1 to emit: the record of positions Pr .. at L1 address off (rb bytes)
__attribute__((noinline, minsize)) void {{ p }}_mark(int32_t *__restrict st, int Pr, int off, int rb)
{
    const int p = st[{{ A_P }}], S = st[{{ A_S }}];
    if (!st[{{ A_INS }}] || ((Pr >> 3) != (p >> 3) && (Pr >> 3) != ((p + S - 1) >> 3)))
        return;
    const int n = st[{{ A_EN0 }}] ? 1 : 0;  // (a second record: range 1)
    st[{{ A_EOFF0 }} + 2 * n] = off;
    st[{{ A_EN0 }} + 2 * n] = rb >> {{ D_SHIFT }};
    if (S > 1 && (p >> 3) == ((p + S - 1) >> 3)) {  // (one record of several slots: emitted twice)
        st[{{ A_EOFF1 }}] = off;
        st[{{ A_EN1 }}] = rb >> {{ D_SHIFT }};
    }
}

{% if ONE_HD %}
// The masks of a record (positions Pr ..) of each set: its lanes past a slot's position, before a window or position 0
__attribute__((noinline, minsize)) void {{ p }}_mrec(const int32_t *__restrict st, float *__restrict so, int Pr, int nsets)
{
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], {{ "lnh = st[%d] == 4 ? 2 : 1" % A_NH if NH_POW2 else "nhm = st[%d]" % A_NHM }};
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {
        aie::store_v(so, aie::select(aie::load_v<32>(so), aie::broadcast<float, 32>(-1e30f),
                                     aie::mask<32>::from_uint32({{ p }}_mask(set, Pr, p, S, {{ "lnh" if NH_POW2 else "nhm" }}, st[{{ A_WIN }}]))));
        so += {{ 32 * RG_MAX }};
    }
}

// A K object ({{ KPER[HDS[0]] }} records, record index ri0 .. of its group, positions P0 + 8 r ..): the step's K inserted into
// its records and they marked for emission (the chain's last column), the scores of the records up to the step's last
// position per set (those after it: none, vobj reads none and an open record's scores stay for the tail objects), the
// masks of those with lanes past a slot's position or before a window or position 0
__attribute__((noinline, minsize)) void {{ p }}_kobj(uint8_t *__restrict a, int32_t *__restrict ws, int ri0, int P0)
{
    {{ p }}_AWin<{{ WINS['kobj'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], last = p + S - 1, nsets = (st[{{ A_NH }}] * S + 3) >> 2;
    if (st[{{ A_INS }}]) {  // (the records of positions p .. p + S - 1 in this object)
        const int r0 = (p - P0) >> 3, r1 = (last - P0) >> 3;
        AIE_LOOP_NO_UNROLL
        for (int r = r0 < 0 ? 0 : r0; r <= r1 && r < {{ KPER[HDS[0]] }}; r++) {
            uint8_t *__restrict rec = a + {{ 72 * KP * ND }} * r;
            const int Pr = P0 + 8 * r;
            AIE_LOOP_NO_UNROLL
            for (int s = 0; s < S; s++)
                if (((p + s) >> 3) == (Pr >> 3))
                    {{ p }}_put_k(rec, AW(int8_t, {{ a.kb }}) + {{ KB_SLOT }} * s, AW(uint8_t, {{ a.kb }}) + {{ KB_SLOT }} * s + {{ 8 * KP * ND }},
                                 (p + s) & 7, {{ ND }});
            {{ p }}_mark(st, Pr, (int)(intptr_t)rec, {{ 72 * KP * ND }});
        }
    }
    const int n0 = last >= P0 ? ((last - P0) >> 3) + 1 : 0, nr = n0 < {{ KPER[HDS[0]] }} ? n0 : {{ KPER[HDS[0]] }};
    float *__restrict sc = AW(float, {{ a.sc }}) + 32 * ri0;  // [set][record][8 q + j]
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++)
        {{ p }}_scores(a, AW(uint8_t, {{ a.qa }}) + {{ 72 * ND }} * set, sc + {{ 32 * RG_MAX }} * set, nr);
    // masks: the records before max(window, 0) (rw of them), those from the first with a position past p (rm)
    const int lo = st[{{ A_WIN }}] && p + S - st[{{ A_WIN }}] > 0 ? p + S - st[{{ A_WIN }}] : 0;
    const int w0 = (lo - P0 + 7) >> 3, rw = w0 < 0 ? 0 : (w0 < nr ? w0 : nr), m0 = (p - P0 + 1) >> 3, rm = m0 < rw ? rw : m0;
    AIE_LOOP_NO_UNROLL
    for (int r = 0; r < rw; r++)
        {{ p }}_mrec(st, sc + 32 * r, P0 + 8 * r, nsets);
    AIE_LOOP_NO_UNROLL
    for (int r = rm; r < nr; r++)
        {{ p }}_mrec(st, sc + 32 * r, P0 + 8 * r, nsets);
}
{% else %}
// A K object (kper records, record index ri0 .. of its group, positions P0 + 8 r ..): the step's K inserted (the chain's
// last column), the records to emit (the chain's last column), the scores per set
__attribute__((noinline, minsize)) void {{ p }}_kobj(uint8_t *__restrict a, int32_t *__restrict ws, int ri0, int P0)
{
    {{ p }}_AWin<{{ WINS['kobj'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], nd = st[{{ A_HD }}] >> 3, kper = {{ by_nd(KPER) }};
    const int nsets = (st[{{ A_NH }}] * S + 3) >> 2, {{ "lnh = st[%d] == 4 ? 2 : 1" % A_NH if NH_POW2 else "nhm = st[%d]" % A_NHM }};
    const int lo_all = st[{{ A_WIN }}] ? p + S - st[{{ A_WIN }}] : 0;  // (every row's window starts at or before it)
    float *__restrict sc = AW(float, {{ a.sc }});  // [set][record][8 q + j]
    AIE_LOOP_NO_UNROLL
    for (int r = 0; r < kper; r++) {
        uint8_t *__restrict rec = a + {{ 72 * KP }} * nd * r;
        const int Pr = P0 + 8 * r, ri = ri0 + r;
        if (st[{{ A_INS }}] && Pr + 8 > p && Pr < p + S) {
            AIE_LOOP_NO_UNROLL
            for (int s = 0; s < S; s++)
                if (((p + s) >> 3) == (Pr >> 3))
                    {{ p }}_put_k(rec, AW(int8_t, {{ a.kb }}) + {{ KB_SLOT }} * s, AW(uint8_t, {{ a.kb }}) + {{ KB_SLOT }} * s + {{ 8 * KP }} * nd,
                                 (p + s) & 7, nd);
        }
        {{ p }}_mark(st, Pr, (int)(intptr_t)rec, {{ 72 * KP }} * nd);
        AIE_LOOP_NO_UNROLL
        for (int set = 0; set < nsets; set++) {
            float *__restrict so = sc + 32 * ({{ RG_MAX }} * set + ri);
            if (Pr > p + S - 1)  // (no scores: vobj reads none past the step's last position, and an open record's
                continue;        // scores stay for the tail objects though the step's next group comes)
            if (Pr + 8 <= 0) {
                aie::store_v(so, aie::broadcast<float, 32>(-1e30f));
                continue;
            }
            {{ p }}_scores(rec, AW(uint8_t, {{ a.qa }}) + 72 * nd * set, so, nd);
            if (Pr + 7 > p || Pr < lo_all)
                aie::store_v(so, aie::select(aie::load_v<32>(so), aie::broadcast<float, 32>(-1e30f),
                                             aie::mask<32>::from_uint32({{ p }}_mask(set, Pr, p, S, {{ "lnh" if NH_POW2 else "nhm" }}, st[{{ A_WIN }}]))));
        }
    }
}
{% endif %}

// per row r < 4 (lanes 8 r ..): the maximum of its 8 lanes, in all 8
[[clang::always_inline]] inline aie::vector<bfloat16, 32> {{ p }}_max8(aie::vector<bfloat16, 32> v)
{
    v = aie::max(v, aie::shuffle_down_rotate(v, 1));
    v = aie::max(v, aie::shuffle_down_rotate(v, 2));
    v = aie::max(v, aie::shuffle_down_rotate(v, 4));  // lane 8 r: the max of lanes 8 r .. 8 r + 7
    aie::vector<bfloat16, 32> r = aie::broadcast<bfloat16, 32>(v[0]);
    r = aie::select(r, aie::broadcast<bfloat16, 32>(v[8]), aie::mask<32>::from_uint32(0x0000FF00u));
    r = aie::select(r, aie::broadcast<bfloat16, 32>(v[16]), aie::mask<32>::from_uint32(0x00FF0000u));
    return aie::select(r, aie::broadcast<bfloat16, 32>(v[24]), aie::mask<32>::from_uint32(0xFF000000u));
}

// Two records' P panels (e: their P, 32 lanes each) -> pp (2 x 72 B: per record [P hi rows | P lo rows])
__attribute__((noinline)) void {{ p }}_ppan(aie::vector<float, 64> e, uint8_t *__restrict pp)
{
    const al::BV nI = al::neg_eye();
    al::BOut<> o((bfp16ebs8 *)pp);
    o.push(al::hilo4(aie::accum<accfloat, 32>(e.template extract<32>(0)), nI));
    o.push(al::hilo4(aie::accum<accfloat, 32>(e.template extract<32>(1)), nI));
}

{% if ONE_HD %}
// The maximum per lane of a group's scores (n records from ss) and mg, in bf16
__attribute__((noinline)) aie::vector<bfloat16, 32> {{ p }}_gmax(const float __aie_dm_resource_a *__restrict ss,
                                                                 aie::vector<bfloat16, 32> mg, int n)
{
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int r = 0; r < n; r++) {
        mg = aie::max(mg, aie::accum<accfloat, 32>(aie::load_v<32>(ss)).template to_vector<bfloat16>());
        ss += 32;
    }
    return mg;
}

// The P panels of a group's records 0 .. nc - 1 (scores from ss against the maxima mn; an odd count's partner zero) ->
// pp (per record [P hi rows | P lo rows]), l += their sums. Two loops: the exponentials in place of the scores (the
// records before the open ones: no one reads their scores again), then the panels
__attribute__((noinline)) void {{ p }}_pexp(float *__restrict ss, aie::vector<float, 32> mn, uint8_t *__restrict pp,
                                            float *__restrict l, int nc)
{
    aie::vector<float, 32> lv = aie::load_v<32>(l);
    float *__restrict se = ss;
    AIE_LOOP_NO_UNROLL
    for (int r = 0; r < nc; r += 2) {
        const aie::vector<float, 32> s1 = aie::select(aie::load_v<32>(se + 32), aie::broadcast<float, 32>(-1e30f),
                                                      aie::mask<32>::from_uint32(r + 1 < nc ? 0u : ~0u));
        const aie::vector<float, 64> e = {{ p }}_exp2(aie::concat(aie::sub(aie::load_v<32>(se), mn), aie::sub(s1, mn)));
        lv = aie::add(lv, aie::add(e.template extract<32>(0), e.template extract<32>(1)));
        aie::store_v(se, e.template extract<32>(0));
        if (r + 1 < nc)  // (an odd count: the next record is open, its scores stay)
            aie::store_v(se + 32, e.template extract<32>(1));
        se += 64;
    }
    aie::store_v(l, lv);
    AIE_LOOP_NO_UNROLL
    for (int r = 0; r < nc; r += 2) {
        const aie::vector<float, 32> e1 = r + 1 < nc ? aie::load_v<32>(ss + 32) : aie::zeros<float, 32>();
        {{ p }}_ppan(aie::concat(aie::load_v<32>(ss), e1), pp + 72 * r);
        ss += 64;
    }
}

// The V object of a group ({{ RG[HDS[0]] }} records, positions P0 + 8 r ..): per set the group's softmax terms with the running
// maxima and y += P V of its records before the open ones (at the chain's end: from the step's first record on; the
// tail objects add theirs from their scores, still in sc: kobj writes none past the step's last position)
__attribute__((noinline, minsize)) void {{ p }}_vobj(uint8_t *__restrict a, int32_t *__restrict ws, int P0)
{
    {{ p }}_AWin<{{ WINS['vobj'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], nsets = (st[{{ A_NH }}] * S + 3) >> 2;
    float *__restrict sc = AW(float, {{ a.sc }});
    const int last = p + S - 1, nr = last >= P0 ? ((last - P0) >> 3) + 1 : 0, nv = nr < {{ RG[HDS[0]] }} ? nr : {{ RG[HDS[0]] }};
    const int no = (8 * (p >> 3) - P0) >> 3, nc = no < 0 ? 0 : (no < nv ? no : nv);  // (records before the open ones)
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {
        // the group's maximum per row; y, l rescaled by 2^(m - m_new) when one rises
        float *__restrict ml = AW(float, {{ a.m }}) + 32 * set;
        // (in bf16, native: the maxima only bound the exponents, y / l is the same with any; m0 is bf16-exact)
        const aie::vector<float, 32> m0 = aie::load_v<32>(ml);
        const aie::vector<bfloat16, 32> m0b = aie::accum<accfloat, 32>(m0).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> mnb = {{ p }}_max8({{ p }}_gmax((const float __aie_dm_resource_a *)(sc + {{ 32 * RG_MAX }} * set), m0b, nv));
        const aie::vector<float, 32> mn = aie::accum<accfloat, 32>(mnb).template to_vector<float>();
        aie::store_v(ml, mn);
        // (else alpha = 1; y and l zero before the first P V terms. A set's rows rescaled together: the rows whose
        // maximum stayed, alpha = 1, keep their y and l exactly: + y (alpha - 1))
        if (st[{{ A_YNZ }}] && aie::lt(m0b, mnb).to_uint32()) {
            const aie::vector<float, 32> alpha = {{ p }}_exp2(aie::concat(aie::sub(m0, mn), aie::zeros<float, 32>())).template extract<32>(0);
            float *__restrict y = AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set;
            {{ p }}_mrg((const float __aie_dm_resource_a *)y, y, (const float __aie_dm_resource_b *)y, alpha,
                        aie::zeros<float, 32>(), {{ ND + 1 }});
        }
    }
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {  // the records' P panels, y += P V
        const aie::vector<float, 32> mn = aie::load_v<32>(AW(float, {{ a.m }}) + 32 * set);
        float *__restrict y = AW(float, {{ a.y }}) + {{ 32 * (ND + 1) }} * set;
        {{ p }}_pexp(sc + {{ 32 * RG_MAX }} * set, mn, pp, y + {{ 32 * ND }}, nc);
        AIE_LOOP_NO_UNROLL
        for (int r = 0; r < nc; r += 2)  // (an odd count's partner: the next record, inside the object: rg is even)
            {{ p }}_pv(a + {{ 72 * ND }} * r, a + {{ 72 * ND }} * (r + 1), pp + 72 * r, y, y, {{ ND }});
    }
    st[{{ A_YNZ }}] = 1;
}
{% else %}
// The V object of a group (rg records, positions P0 + 8 r ..): per set the group's softmax terms with the running
// maxima and y += P V of its records before the open ones (at the chain's end: from the step's first record on; the
// tail objects add theirs from their scores, still in sc: kobj writes none past the step's last position)
__attribute__((noinline, minsize)) void {{ p }}_vobj(uint8_t *__restrict a, int32_t *__restrict ws, int P0)
{
    {{ p }}_AWin<{{ WINS['vobj'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], nd = st[{{ A_HD }}] >> 3, rg = {{ by_nd(RG) }};
    const int nsets = (st[{{ A_NH }}] * S + 3) >> 2;
    float *__restrict sc = AW(float, {{ a.sc }});
    const int last = p + S - 1, nr = last >= P0 ? ((last - P0) >> 3) + 1 : 0, nv = nr < rg ? nr : rg;  // (records up to
    // the step's last position: those after it add nothing)
    const int no = (8 * (p >> 3) - P0) >> 3, nc = no < 0 ? 0 : (no < nv ? no : nv);  // (records before the open ones)
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {
        // the group's maximum per row; y, l rescaled by 2^(m - m_new) when one rises
        const float *__restrict ss = sc + 32 * {{ RG_MAX }} * set;
        float *__restrict ml = AW(float, {{ a.m }}) + 32 * set;
        // (in bf16, native: the maxima only bound the exponents, y / l is the same with any; m0 is bf16-exact)
        const aie::vector<float, 32> m0 = aie::load_v<32>(ml);
        aie::vector<bfloat16, 32> mg = aie::accum<accfloat, 32>(m0).template to_vector<bfloat16>();
        AIE_LOOP_NO_UNROLL
        for (int r = 0; r < nv; r++)
            mg = aie::max(mg, aie::accum<accfloat, 32>(aie::load_v<32>(ss + 32 * r)).template to_vector<bfloat16>());
        const aie::vector<bfloat16, 32> m0b = aie::accum<accfloat, 32>(m0).template to_vector<bfloat16>(), mnb = {{ p }}_max8(mg);
        const aie::vector<float, 32> mn = aie::accum<accfloat, 32>(mnb).template to_vector<float>();
        aie::store_v(ml, mn);
        // (else alpha = 1; y and l zero before the first P V terms. A set's rows rescaled together: the rows whose
        // maximum stayed, alpha = 1, keep their y and l exactly: + y (alpha - 1))
        if (st[{{ A_YNZ }}] && aie::lt(m0b, mnb).to_uint32()) {
            const aie::vector<float, 32> alpha = {{ p }}_exp2(aie::concat(aie::sub(m0, mn), aie::zeros<float, 32>())).template extract<32>(0);
            float *__restrict y = AW(float, {{ a.y }}) + 32 * (nd + 1) * set;
            {{ p }}_mrg((const float __aie_dm_resource_a *)y, y, (const float __aie_dm_resource_b *)y, alpha,
                        aie::zeros<float, 32>(), nd + 1);
        }
    }
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {  // the records' P panels, y += P V
        const float *__restrict ss = sc + 32 * {{ RG_MAX }} * set;
        const aie::vector<float, 32> mn = aie::load_v<32>(AW(float, {{ a.m }}) + 32 * set);
        float *__restrict y = AW(float, {{ a.y }}) + 32 * (nd + 1) * set;
        aie::vector<float, 32> l = aie::load_v<32>(y + 32 * nd);
        AIE_LOOP_NO_UNROLL
        for (int r = 0; r < nc; r += 2) {  // (two records: an odd count's partner zero)
            const aie::vector<float, 32> s1 = r + 1 < nc ? aie::load_v<32>(ss + 32 * r + 32) : aie::broadcast<float, 32>(-1e30f);
            const aie::vector<float, 64> e = {{ p }}_exp2(aie::concat(aie::sub(aie::load_v<32>(ss + 32 * r), mn), aie::sub(s1, mn)));
            l = aie::add(l, aie::add(e.template extract<32>(0), e.template extract<32>(1)));
            {{ p }}_ppan(e, pp + 72 * r);
        }
        aie::store_v(y + 32 * nd, l);
        AIE_LOOP_NO_UNROLL
        for (int r = 0; r < nc; r += 2)  // (an odd count's partner: the next record, inside the object: rg is even)
            {{ p }}_pv(a + 72 * nd * r, a + 72 * nd * (r + 1), pp + 72 * r, y, y, nd);
    }
    st[{{ A_YNZ }}] = 1;
}
{% endif %}

// The P panels of an open record for a slot's rows of a set (sc: the record's scores; m, l: the set's maxima, sums;
// rows: the slot's lanes): l += P, pp = [P hi + lo | zeros (no second record)]
__attribute__((noinline)) void {{ p }}_pslot(const float *__restrict sc, const float *__restrict m, float *__restrict l,
                                             uint8_t *__restrict pp, uint32_t rows)
{
    const aie::vector<float, 32> e = {{ p }}_exp2(aie::concat(aie::sub(aie::load_v<32>(sc), aie::load_v<32>(m)),
                                                              aie::zeros<float, 32>())).template extract<32>(0);
    const aie::vector<float, 32> P = aie::select(aie::zeros<float, 32>(), e, aie::mask<32>::from_uint32(rows));
    aie::store_v(l, aie::add(aie::load_v<32>(l), P));
    {{ p }}_ppan(aie::concat(P, aie::zeros<float, 32>()), pp);
}

// Tail object t of the chain's end, after its range: the open record r = (t ? p + S - 1 : p) >> 3 as rows ([d-block
// nd][position 8][8] bf16): t = 0 its ring slot (in place), or (A_TREC) its V panels as rows in qa; t = 1 (a second open
// record) its ring slot, or none before the step's at a layer with its own cache (in place: the object's bytes as the
// rows). Such a layer's step V rows in (the rows past the step's last position: as they were, read by no one). Per slot
// that reaches the record: the record's V panels from its rows up to the slot's position (into kb: what that position
// sees of the record, whatever the step), y += P V of the slot's rows. The chain's end: the last panels (the rows up to
// the step's last position) emitted with the rows (the ring slot); t = 1 of a step of several slots in one record: both
// again (the object: the slot as before the step, its rows put in again).
__attribute__((noinline, minsize)) void {{ p }}_tobj(uint8_t *__restrict a, int32_t *__restrict ws, int t)
{
    {{ p }}_AWin<{{ WINS['tobj'] }}> win_;
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int p = st[{{ A_P }}], S = st[{{ A_S }}], hd = st[{{ A_HD }}], nd = hd >> 3, r0 = p >> 3, r1 = (p + S - 1) >> 3;
{% if NH_POW2 %}
    const int lnh = st[{{ A_NH }}] == 4 ? 2 : 1, rec = st[{{ A_TREC }}] && (t == 0 || r1 == r0);
{% else %}
    const int nh = st[{{ A_NH }}], rec = st[{{ A_TREC }}] && (t == 0 || r1 == r0);
{% endif %}
    bfloat16 *__restrict rows = rec ? AW(bfloat16, {{ a.qa }}) : (bfloat16 *)a;
    const int P0 = 8 * (t ? r1 : r0), last = p + S - 1 - P0 < 7 ? p + S - 1 - P0 : 7;  // (the step's last row of the record)
    const int ri = (P0 >> 3) & (({{ by_nd(RG) }}) - 1);  // (its scores, in sc: ``vobj``)
    uint8_t *__restrict v = AW(uint8_t, {{ a.kb }});  // (the step's K blocks are in)
    if (t == 0 && rec)
        {{ p }}_vrows(a, rows, nd);
    AIE_LOOP_NO_UNROLL
    for (int j = P0 > p ? 0 : p - P0; st[{{ A_INS }}] && !(t && rec) && j <= last; j++)  // (row j: slot P0 + j - p)
        {{ p }}_rowput(rows + 8 * j, AW(bfloat16, {{ a.vn }}) + hd * (P0 + j - p), nd);
    if (t == 0 || r1 != r0) {
        uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
        AIE_LOOP_NO_UNROLL
        for (int s = P0 > p ? P0 - p : 0; s < S; s++) {
{% if NH_POW2 %}
            const int j = p + s - P0, set = (s << lnh) >> 2;  // (the slot's row of the record, its set)
            {{ p }}_venc(rows, v, nd, j < 7 ? j + 1 : 8);
            float *__restrict y = AW(float, {{ a.y }}) + 32 * (nd + 1) * set;
            {{ p }}_pslot(AW(float, {{ a.sc }}) + 32 * ({{ RG_MAX }} * set + ri), AW(float, {{ a.m }}) + 32 * set, y + 32 * nd,
                          pp, lnh == 2 ? ~0u : 0xFFFFu << (16 * (s & 1)));  // (the slot's rows of the set)
            {{ p }}_pv(v, v, pp, y, y, nd);
{% else %}
            const int j = p + s - P0, q0 = nh * s, q1 = q0 + nh;  // (the slot's row of the record, its query rows)
            {{ p }}_venc(rows, v, nd, j < 7 ? j + 1 : 8);
            AIE_LOOP_NO_UNROLL
            for (int set = q0 >> 2; set <= (q1 - 1) >> 2; set++) {  // (the sets of its rows: their lanes there)
                const int lo = q0 - 4 * set, hi = q1 - 4 * set;
                const uint32_t lanes = (hi >= 4 ? ~0u : (1u << (8 * hi)) - 1u) & (lo <= 0 ? ~0u : ~((1u << (8 * lo)) - 1u));
                float *__restrict y = AW(float, {{ a.y }}) + 32 * (nd + 1) * set;
                {{ p }}_pslot(AW(float, {{ a.sc }}) + 32 * ({{ RG_MAX }} * set + ri), AW(float, {{ a.m }}) + 32 * set,
                              y + 32 * nd, pp, lanes);
                {{ p }}_pv(v, v, pp, y, y, nd);
            }
{% endif %}
        }
        st[{{ A_YNZ }}] = 1;
    }
    if (st[{{ A_INS }}]) {
        st[{{ A_EOFF0 }}] = (int)(intptr_t)v;
        st[{{ A_EN0 }}] = (72 * nd) >> {{ D_SHIFT }};
        st[{{ A_EOFF1 }}] = (int)(intptr_t)rows;
        st[{{ A_EN1 }}] = (16 * hd) >> {{ D_SHIFT }};
    }
}

// o = y il of a set's rows (y: [db nd][row 4][8] fp32, il: their 1 / l, 8 lanes per row) -> bf16 at oq + 8 sp db of
// row q (the O projection's packed input, ``aend``)
__attribute__((noinline, minsize)) void {{ p }}_pack(const float *__restrict y, aie::vector<float, 32> il, bfloat16 *__restrict o0,
                                            bfloat16 *__restrict o1, bfloat16 *__restrict o2, bfloat16 *__restrict o3,
                                            int nd, int step)
{
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < nd; db++) {
        const aie::vector<bfloat16, 32> o = aie::accum<accfloat, 32>(al::mulff<32>(aie::load_v<32>(y), il)).template to_vector<bfloat16>();
        y += 32;
        aie::store_v(o0, o.template extract<8>(0));
        aie::store_v(o1, o.template extract<8>(1));
        aie::store_v(o2, o.template extract<8>(2));
        aie::store_v(o3, o.template extract<8>(3));
        o0 += step;
        o1 += step;
        o2 += step;
        o3 += step;
    }
}

}  // namespace

extern "C" {

// A new X object: a token object (a sub-step's positions, the column's ranges, cos / sin), a sub-step's first q object
// (F_ATT: the layer's attention (sub-)step), a main core's phase (its input and A objects, skipped in the main core's order:
// X and A are broadcast to both cores), or another core's object
__attribute__((minsize)) void {{ p }}_abegin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t col)
{
    {{ p }}_AWin<{{ WINS['abegin'] }}> win_;
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int flags = m[{{ M_FLAGS }}];
    for (int i = 0; i < {{ A_P }}; i++)
        st[i] = 0;
    if (flags & {{ F_MAIN }}) {
        if (!(flags & {{ F_LOADR }})) {  // its input objects (under its norm-weight objects), its tiles
            const int nx = m[{{ M_NX }}], S = m[{{ M_NS }}], lper = S == 1 ? 2 : (S == 2 ? 1 : 0);
            st[{{ A_NPS }}] = (flags & {{ F_POST }}) ? 1 : 0;
            st[{{ A_LPER }}] = lper;
            if (flags & {{ F_NORM + F_POST }}) {
                st[{{ A_NNO }}] = (nx + {{ NX_PER_NORM - 1 }}) / {{ NX_PER_NORM }};
                st[{{ A_XL }}] = nx;
            } else {
                st[{{ A_NXSKIP }}] = (nx + (1 << lper) - 1) >> lper;
            }
            st[{{ A_NASKIP }}] = m[{{ M_NT }}];
        }
        return;
    }
    if (flags & {{ F_TOK }}) {  // a sub-step's token object (t = SMAX kind + its first slot)
{% if ROT|length == 2 %}
        const int t = m[{{ M_TOK_KIND }}], kind = t >= {{ SMAX }};
{% else %}
        const int t = m[{{ M_TOK_KIND }}];
{% endif %}
        int32_t *__restrict tk = st + {{ A_TOK }} + 4 * t;
        tk[0] = m[{{ M_TOK_G0 }} + col];
        tk[1] = m[{{ M_TOK_NA }} + col] >> 16;
        tk[2] = m[{{ M_TOK_NA }} + col] & 0xFFFF;
        tk[3] = (m[{{ M_POS }}] << 8) | (m[{{ M_TOK_A }}] << 4) | m[{{ M_NS }}];
        const float *__restrict src = (const float *)(x + {{ 2 * META }});
{% if ROT|length == 2 %}
        float *__restrict cs = AW(float, {{ a.cs }}) + (kind ? {{ 2 * SMAX * ROT[0] }} : 0);
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < (kind ? {{ 2 * SMAX * ROT[1] }} : {{ 2 * SMAX * ROT[0] }}); i += 16)  // (a loop: program memory)
{% else %}
        float *__restrict cs = AW(float, {{ a.cs }});
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < {{ 2 * SMAX * ROT[0] }}; i += 16)  // (a loop: program memory)
{% endif %}
            aie::store_v(cs + i, aie::load_v<16>(src + i));
        return;
    }
    if (!(flags & {{ F_ATT }})) {
        st[{{ A_NXSKIP }}] = m[{{ M_SKIP }}];
        return;
    }
    // the layer's attention (sub-)step: per column of the KV head its place in the chain
    const int32_t *__restrict tk = st + {{ A_TOK }} + 4 * m[{{ M_ATOK }}];
    const int S = tk[3] & 15, hd = m[{{ M_AHD }}], j = col & {{ CHAIN - 1 }}, nq = m[{{ M_ANH }}];
    st[{{ A_P }}] = tk[3] >> 8;
    st[{{ A_S }}] = S;
    st[{{ A_SA }}] = (tk[3] >> 4) & 15;
    st[{{ A_HD }}] = hd;
    st[{{ A_OWN }}] = m[{{ M_AOWN }}];
    st[{{ A_WIN }}] = m[{{ M_AWIN }}];
    st[{{ A_TREC }}] = m[{{ M_ATREC }}];
    st[{{ A_QC }}] = m[{{ M_QC_HI }}] | (m[{{ M_QC_LO }}] << 16);
    st[{{ A_NH }}] = nq;
{% if not NH_POW2 %}
    st[{{ A_NHM }}] = nq == 1 ? 64 : (nq == 2 ? 32 : (nq == 3 ? 22 : 16));
{% endif %}
    st[{{ A_POS }}] = tk[0];
    st[{{ A_MERGE }}] = j > 0;
    st[{{ A_CEND }}] = j == {{ CHAIN - 1 }};
    st[{{ A_INS }}] = st[{{ A_OWN }}] && st[{{ A_CEND }}];
    st[{{ A_NQP }}] = 1;
    st[{{ A_NQ }}] = S - 1;
    st[{{ A_NKV }}] = st[{{ A_OWN }}] ? S : 0;
    st[{{ A_NOBJ }}] = tk[1];
    st[{{ A_NATT }}] = tk[2];
    st[{{ A_NO }}] = st[{{ A_CEND }}] ? (nq * hd * S) >> {{ D_SHIFT - 1 }} : 0;  // (o objects)
    st[{{ A_QS }}] = 0;
    // the F_ATT object is the sub-step's first q object: its heads into qa (free by now), ``aq0`` once the chain lets
    // the column's q rows overwrite its last partial (the east neighbour merged it)
    al::copy<{{ QCOPY }}, 64>(AW(bfloat16, {{ a.qa }}), x + {{ 2 * META }});  // ([head nh][hd])
    st[{{ A_I }}] = 0;
    st[{{ A_KS }}] = 0;
{% if WIN in (40, 41) %}
    event0();
{% endif %}
}

// A main core's norm-weight object: the input objects under it
__attribute__((minsize)) void {{ p }}_anb(int32_t *__restrict ws)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int left = st[{{ A_XL }}], n = left > {{ NX_PER_NORM }} ? {{ NX_PER_NORM }} : left, lper = st[{{ A_LPER }}];
    st[{{ A_NXN }}] = (n + (1 << lper) - 1) >> lper;
    st[{{ A_XL }}] = left - n;
}

// The next q object (slot s of the sub-step, the step's a + s: [head nh][hd]): its heads {{ "normed and " if QK_NORM else "" }}rotated into the
// query rows (row s nh + h)
__attribute__((minsize)) void {{ p }}_aq(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_AWin<{{ WINS['aq'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int s = st[{{ A_QS }}], nh = st[{{ A_NH }}], hd = st[{{ A_HD }}];
    st[{{ A_QS }}] = s + 1;
{% if ROT|length == 2 %}
    const int rot = hd == {{ HDS[0] }} ? {{ ROT[0] }} : {{ ROT[1] }};
    const float *__restrict cs = AW(float, {{ a.cs }}) + (hd == {{ HDS[0] }} ? 0 : {{ 2 * SMAX * ROT[0] }}) + 2 * rot * (st[{{ A_SA }}] + s);
{% else %}
    const int rot = {{ ROT[0] }};
    const float *__restrict cs = AW(float, {{ a.cs }}) + 2 * rot * (st[{{ A_SA }}] + s);
{% endif %}
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < nh; h++)
        {{ p }}_nrope(x + {{ 2 * META }} + hd * h, cs, AW(float, {{ a.y }}) + hd * (nh * s + h), hd, rot);
}

// The sub-step's first q object, as abegin left it in qa
__attribute__((minsize)) void {{ p }}_aq0(int32_t *__restrict ws)
{
    {{ p }}_aq(AW(bfloat16, {{ a.qa }}) - {{ 2 * META }}, ws);
}

// {{ DB }} d-blocks of a set's 4 q rows (fp32, rows rs apart from q; rows past nv zero) times the score constant qc (bf16 hi |
// lo << 16: three products in one accumulator) as hi + lo panels -> qa ({{ 72 * DB }} B)
__attribute__((noinline)) void {{ p }}_qpan(const float *__restrict q, uint8_t *__restrict qa, int rs, int nv, int qc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::mask<32> valid = aie::mask<32>::from_uint32(nv >= 4 ? ~0u : (1u << (8 * nv)) - 1u);
    const aie::vector<bfloat16, 32> c_hi = aie::broadcast<bfloat16, 32>(__builtin_bit_cast(bfloat16, (uint16_t)(qc & 0xFFFF)));
    const aie::vector<bfloat16, 32> c_lo = aie::broadcast<bfloat16, 32>(__builtin_bit_cast(bfloat16, (uint16_t)((uint32_t)qc >> 16)));
    const al::BV nI = al::neg_eye();
    al::BOut<> o((bfp16ebs8 *)qa);
    const float __aie_dm_resource_a *__restrict p0 = (const float __aie_dm_resource_a *)q;
    const float __aie_dm_resource_a *__restrict p1 = p0 + rs;
    const float __aie_dm_resource_b *__restrict p2 = (const float __aie_dm_resource_b *)(q + 2 * rs);
    const float __aie_dm_resource_b *__restrict p3 = p2 + rs;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ DB }}; db++) {
        aie::vector<float, 32> v = aie::concat(aie::load_v<8>(p0), aie::load_v<8>(p1), aie::load_v<8>(p2), aie::load_v<8>(p3));
        p0 += 8;
        p1 += 8;
        p2 += 8;
        p3 += 8;
        v = aie::select(aie::zeros<float, 32>(), v, valid);
        const aie::vector<bfloat16, 32> h = aie::accum<accfloat, 32>(v).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> l = aie::accum<accfloat, 32>(
            aie::sub(v, aie::accum<accfloat, 32>(h).template to_vector<float>())).template to_vector<bfloat16>();
        o.push(al::hilo4(aie::mac(aie::mac(aie::mul(h, c_hi), l, c_hi), h, c_lo), nI));
    }
}

// The q rows are in: the Q panels of every set, scaled by the score constant (rows past S nh zero), y = 0, the running
// maxima (a floor below any score: a row without positions here keeps l = 0) and sums
__attribute__((minsize)) void {{ p }}_aqpan(int32_t *__restrict ws)
{
    {{ p }}_AWin<{{ WINS['aqpan'] }}> win_;
    const int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int hd = st[{{ A_HD }}], nd = hd >> 3, nr = st[{{ A_S }}] * st[{{ A_NH }}];
    float *__restrict q = AW(float, {{ a.y }});
    const int nsets = (nr + 3) >> 2;
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < nsets; set++) {
        AIE_LOOP_NO_UNROLL
        for (int h = 0; h < nd; h += {{ DB }})  // (rows of the set with a slot: nr - 4 set)
            {{ p }}_qpan(q + hd * 4 * set + 8 * h, AW(uint8_t, {{ a.qa }}) + 72 * (nd * set + h), hd, nr - 4 * set, st[{{ A_QC }}]);
        aie::store_v(AW(float, {{ a.m }}) + 32 * set, aie::broadcast<float, 32>(-8192.0f));  // (bf16-exact)
    }
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ Y_FLOATS }}; i += 64)  // (y, l = 0; a loop: program memory)
        aie::store_v(q + i, aie::zeros<float, 64>());
    AW(int32_t, {{ a.st }})[{{ A_YNZ }}] = 0;
}

// The raw K, V object of the next slot ([K hd | V hd]): its K ({{ "normed, " if QK_NORM else "" }}rotated) as {{ "hi + lo blocks" if KP == 2 else "bfp16 blocks" }} (mantissas: 64 B per 8
// blocks, then the exponents), its V ({{ "normed" if V_NORM else "as it is" }}) bf16
__attribute__((minsize)) void {{ p }}_akv(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t kvh)
{
    {{ p }}_AWin<{{ WINS['akv'] }}> win_;
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int s = st[{{ A_KS }}], hd = st[{{ A_HD }}], nd = hd >> 3;
    st[{{ A_KS }}] = s + 1;
{% if ROT|length == 2 %}
    const int rot = hd == {{ HDS[0] }} ? {{ ROT[0] }} : {{ ROT[1] }};
    const float *__restrict cs = AW(float, {{ a.cs }}) + (hd == {{ HDS[0] }} ? 0 : {{ 2 * SMAX * ROT[0] }}) + 2 * rot * (st[{{ A_SA }}] + s);
{% else %}
    const int rot = {{ ROT[0] }};
    const float *__restrict cs = AW(float, {{ a.cs }}) + 2 * rot * (st[{{ A_SA }}] + s);
{% endif %}
    float *__restrict kf = AW(float, {{ a.kf }});
    const bfloat16 *__restrict kv = x + {{ 2 * META }};
    {{ p }}_nrope(kv, cs, kf, hd, rot);
    int8_t *__restrict km = AW(int8_t, {{ a.kb }}) + {{ KB_SLOT }} * s;
    int8_t *__restrict ke = km + {{ 8 * KP }} * nd;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < nd / 8; i++) {
{% if KP == 2 %}
        al::BV hi, lo;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(kf + 64 * i)), al::neg_eye(), hi, lo);
        const v64bfp16ebs8 h = (v64bfp16ebs8)hi, l = (v64bfp16ebs8)lo;
        *(v64int8 *)(km + 64 * i) = h.mantissa;
        *(v64int8 *)(km + 8 * nd + 64 * i) = l.mantissa;
        *(v8int8 *)(ke + 8 * i) = h.exponent;
        *(v8int8 *)(ke + nd + 8 * i) = l.exponent;
{% else %}
        const v64bfp16ebs8 h = (v64bfp16ebs8)aie::accum<accfloat, 64>(aie::load_v<64>(kf + 64 * i)).template to_vector<bfp16ebs8>();
        *(v64int8 *)(km + 64 * i) = h.mantissa;
        *(v8int8 *)(ke + 8 * i) = h.exponent;
{% endif %}
    }
{% if V_NORM %}
    {{ p }}_nrope(kv + hd, cs, kf, hd, 0);  // V / rms(V)
    bfloat16 *__restrict vn = AW(bfloat16, {{ a.vn }}) + hd * s;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < hd; i += 32)
        aie::store_v(vn + i, aie::accum<accfloat, 32>(aie::load_v<32>(kf + i)).template to_vector<bfloat16>());
{% else %}
    bfloat16 *__restrict vn = AW(bfloat16, {{ a.vn }}) + hd * s;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < hd; i += 32)
        aie::store_v(vn + i, aie::load_v<32>(kv + hd + i));
{% endif %}
}

// A cache object (i = {{ KP + 1 }} k + ph of the column's group k: K objects ph < {{ KP }}, the V object ph {{ KP }}), or at the chain's end
// one of its last tails (``tobj``); st[A_NE]: the D objects of the records to emit from it
__attribute__((minsize)) void {{ p }}_aobj(uint8_t *__restrict a, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_I }}];
    st[{{ A_I }}] = i + 1;
{% if WIN == 41 %}
    if (i == 0)
        event1();
{% endif %}
    const int k3 = al::div_small<{{ KP + 1 }}>(i), ph = i - {{ KP + 1 }} * k3, hd = st[{{ A_HD }}];
    st[{{ A_EN0 }}] = 0;
    st[{{ A_EN1 }}] = 0;
    st[{{ A_EIDX }}] = 0;
    st[{{ A_NE }}] = 0;
    const int t = i - (st[{{ A_NOBJ }}] - (st[{{ A_S }}] > 1 ? 2 : 1));  // (``tails``)
    if (st[{{ A_CEND }}] && t >= 0) {
        {{ p }}_tobj(a, ws, t);
    } else {
        if (i >= st[{{ A_NATT }}])  // (an object past the range: skipped)
            return;
        const int kper = {{ by_hd(KPER) }};
        const int P0 = st[{{ A_POS }}] + {{ 8 * KP }} * kper * k3;  // (the group's first position)
        if (ph < {{ KP }})
            {{ p }}_kobj(a, ws, kper * ph, P0 + 8 * kper * ph);
        else
            {{ p }}_vobj(a, ws, P0);
    }
    st[{{ A_NE }}] = st[{{ A_EN0 }}] + st[{{ A_EN1 }}];
{% if WIN == 42 %}
    if (i + 1 == st[{{ A_NOBJ }}])
        event0();
{% endif %}
}

// The next D object of a record to emit (A_EOFF0, A_EOFF1: the records' L1 addresses, in the cache object a or the
// workspace)
__attribute__((minsize)) void {{ p }}_arec(const uint8_t *__restrict a, int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_AWin<{{ WINS['arec'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}];
    st[{{ A_EIDX }}] = e + 1;
    const int n0 = st[{{ A_EN0 }}], k = e < n0 ? 0 : 1;
    const uint8_t *__restrict src = (const uint8_t *)(intptr_t)(k ? st[{{ A_EOFF1 }}] + {{ 2 * D_LEN }} * (e - n0) : st[{{ A_EOFF0 }}] + {{ 2 * D_LEN }} * e);
    aie::store_v(d, aie::load_v<{{ D_LEN }}>((const bfloat16 *)src));
}

// The attention is through: the west neighbour's partial (pw: its workspace, through shared memory) merged in when the
// chain continues from it; at the chain's end o = y / l into the Q panels' region (free by now): [k block (head, 8
// dims)][slot S][8] bf16, the sub-step's slots of the O projection's packed input (the drain puts them at slot a);
// st[A_NE]: its D objects
__attribute__((minsize)) void {{ p }}_aend(int32_t *__restrict ws, const int32_t *__restrict pw)
{
    {{ p }}_AWin<{{ WINS['aend'] }}> win_;
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
{% if WIN == 42 %}
    event1();
{% endif %}
    st[{{ A_EIDX }}] = 0;
    st[{{ A_NE }}] = st[{{ A_NO }}];
    const int nd = st[{{ A_HD }}] >> 3, nsets = (st[{{ A_NH }}] * st[{{ A_S }}] + 3) >> 2;
{% if CHAIN > 1 %}
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < (st[{{ A_MERGE }}] ? nsets : 0); set++) {  // m = max(m_w, m): y, l = both rescaled, summed
        float *__restrict ml = AW(float, {{ a.m }}) + 32 * set;
        const float *__restrict mw = (const float *)pw + {{ a.m // 4 }} + 32 * set;
        const aie::vector<float, 32> m0 = aie::load_v<32>(ml), m1 = aie::load_v<32>(mw);
        const aie::vector<float, 32> mx = aie::max(m0, m1);
        const aie::vector<float, 64> a01 = {{ p }}_exp2(aie::concat(aie::sub(m0, mx), aie::sub(m1, mx)));
        const aie::vector<float, 32> a0 = a01.template extract<32>(0), a1 = a01.template extract<32>(1);
        aie::store_v(ml, mx);
        float *__restrict y = AW(float, {{ a.y }}) + 32 * (nd + 1) * set;
        const float *__restrict yw = (const float *)pw + {{ a.y // 4 }} + 32 * (nd + 1) * set;
        {{ p }}_mrg((const float __aie_dm_resource_a *)y, y, (const float __aie_dm_resource_b *)yw, a0, a1, nd + 1);
    }
{% endif %}
{% if NH_POW2 %}
    const int sp = st[{{ A_S }}], nh = st[{{ A_NH }}], lnh = nh == 4 ? 2 : 1;
{% else %}
    const int sp = st[{{ A_S }}], nh = st[{{ A_NH }}], nhm = st[{{ A_NHM }}];
{% endif %}
    bfloat16 *__restrict o = AW(bfloat16, {{ a.qa }});
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < (st[{{ A_CEND }}] ? nsets : 0); set++) {
        const float *__restrict l = AW(float, {{ a.y }}) + 32 * (nd + 1) * set + 32 * nd;
        const aie::vector<float, 32> il = al::rows4(aie::inv(al::sum8(aie::load_v<32>(l))));
        bfloat16 *po[4];
        for (int q = 0; q < 4; q++) {  // row 4 set + q: slot r / nh, head r % nh (a row past the slots: kf, scratch)
            const int r = 4 * set + q;
{% if NH_POW2 %}
            po[q] = (r >> lnh) < sp ? o + 8 * (sp * nd * (r & (nh - 1)) + (r >> lnh)) : AW(bfloat16, {{ a.kf }});
{% else %}
            const int s = (r * nhm) >> 6;
            po[q] = s < sp ? o + 8 * (sp * nd * (r - nh * s) + s) : AW(bfloat16, {{ a.kf }});
{% endif %}
        }
        {{ p }}_pack(AW(float, {{ a.y }}) + 32 * (nd + 1) * set, il, po[0], po[1], po[2], po[3], nd, 8 * sp);
    }
{% if WIN == 40 %}
    if (!st[{{ A_NE }}])
        event1();
{% endif %}
}

// The next D object of o (``aend``)
__attribute__((minsize)) void {{ p }}_aemit(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_AWin<{{ WINS['aemit'] }}> win_;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}];
    st[{{ A_EIDX }}] = e + 1;
    const bfloat16 *__restrict o = AW(bfloat16, {{ a.qa }}) + {{ D_LEN }} * e;
{% if D_LEN >= 64 %}
{% for i in range(0, D_LEN, 64) %}
    aie::store_v(d{{ " + %d" % i if i else "" }}, aie::load_v<64>(o{{ " + %d" % i if i else "" }}));
{% endfor %}
{% else %}
    aie::store_v(d, aie::load_v<{{ D_LEN }}>(o));
{% endif %}
{% if WIN == 40 %}
    if (e + 1 == st[{{ A_NE }}])
        event1();
{% endif %}
}

}  // extern "C"
""")


class Attn:
    """The attention cores' shapes: the head dims ``hds`` (one per kind of layer, at most two) and their rotated pairs
    ``rot``, ``chain`` columns per KV head (COLS / chain KV heads), at most ``heads`` query heads per KV head (a layer's:
    the F_ATT object's M_ANH), ``kp`` K panels per record (2: hi + lo, 1: single), RMS norms of q / k (``qk_norm``: the
    weight folded into the score constant) and of v (``v_norm``), the cache objects of ``a_bytes`` (the A stream's)."""

    def __init__(self, *, hds: tuple[int, ...], rot: tuple[int, ...], chain: int, heads: int, a_bytes: int, kp: int = 2,
                 qk_norm: bool = False, v_norm: bool = False, eps: float = 1e-6, c_len: int = 64):
        assert 1 <= len(hds) <= 2 and len(rot) == len(hds) and all(hd % 64 == 0 and hd <= 512 for hd in hds), hds
        assert all(r % 32 == 0 and r <= hd // 2 for r, hd in zip(rot, hds)) and kp in (1, 2)
        assert chain in (1, 2, 4) and 1 <= heads <= 4, (chain, heads)
        self.hds, self.rot, self.chain, self.heads, self.kp = tuple(hds), tuple(rot), chain, heads, kp
        self.qk_norm, self.v_norm, self.eps, self.a_bytes, self.c_len = qk_norm, v_norm, eps, a_bytes, c_len
        self.kv_heads = COLS // chain
        self.d_len = min(256, min(hds)) // 2  # (bf16 of a D object: every record, ring slot and o in whole objects)
        self.kper = {hd: Records(hd, kp).kper(a_bytes) for hd in hds}  # K records per K object
        self.rg = {hd: Records(hd, kp).rg(a_bytes) for hd in hds}  # records per group
        self.pg = {hd: Records(hd, kp).pg(a_bytes) for hd in hds}  # positions per group
        assert all(k > 0 and self.rg[hd] % 2 == 0 for hd, k in self.kper.items()), self.kper

    @staticmethod
    def rows(hd: int) -> int:
        """Query rows a column holds per sub-step (L1)."""
        return 16 if hd <= 256 else 8

    def slots(self, hd: int) -> int:
        """Slots of a sub-step at most (its query rows: ``rows``)."""
        return min(SMAX, self.rows(hd) // self.heads)

    def y_floats(self) -> int:
        """fp32 of an attention core's y: per set of 4 query rows its blocks [d-block][32], then l [32]; at least the
        query rows a column holds (``rows``: before the panels)."""
        y = max(self.rows(hd) // 4 * (hd // 8 + 1) * 32 for hd in self.hds)
        assert y >= max(self.rows(hd) * hd for hd in self.hds)
        return y

    def workspace(self) -> dict[str, int]:
        """Byte offsets of an attention core's workspace regions (and "size"). y: per set of 4 rows its blocks [d-block]
        [32] fp32, then l [32] (merged and rescaled together; before the panels the q rows); qa: the Q panels; m: per set
        the rows' running maxima. At the tail objects (``tobj``): an open record's rows decoded from its V panels in qa
        (the Q panels are done with, o not there yet), its panels in kb (the step's K blocks are in)."""
        hm = max(self.hds)
        qa = max(self.rows(hd) // 4 * 72 * hd // 8 for hd in self.hds)
        rg = max(self.rg.values())
        sets = max(self.rows(hd) // 4 for hd in self.hds)
        kb = max(self.kb_slot * SMAX, 72 * hm // 8)  # (the slots' new K blocks; a record's V panels)
        sizes = [("st", 4 * A_WORDS), ("cs", 4 * 2 * SMAX * sum(self.rot)), ("y", 4 * self.y_floats()), ("qa", qa),
                 ("kf", 4 * hm), ("kb", kb), ("vn", 2 * hm * SMAX), ("sc", 4 * 32 * rg * sets), ("pp", 72 * rg),
                 ("m", 4 * 32 * sets)]
        out = regions(sizes)
        assert 16 * hm <= qa and 2 * self.heads * hm <= qa  # (a record's rows; the first q object's heads)
        return out

    @property
    def kb_slot(self) -> int:
        """Bytes of a slot's new K blocks (kp nd blocks of 9 B, mantissas and exponents apart; from a multiple of 64 B
        on: the mantissas' vector stores)."""
        return rup(9 * self.kp * max(self.hds) // 8)

    def render(self, p: str, consts: dict[str, int], win: str = "") -> str:
        nd_min = min(32, min(self.hds) // 8)
        hds = self.hds

        def by(var: str, vals: dict[int, object], key: object = lambda hd: hd) -> str:  # (a value per head dim)
            if len(hds) == 1:
                return str(vals[hds[0]])
            return f"{var} == {key(hds[0])} ? {vals[hds[0]]} : {vals[hds[1]]}"

        return _SRC.render(p=p, a=Attr(self.workspace()), eps=repr(self.eps), SMAX=SMAX, ROT=self.rot, HDS=hds,
                           WINS=WINS, KPER=self.kper, RG=self.rg, RG_MAX=max(self.rg.values()), D_LEN=self.d_len,
                           D_SHIFT=int(math.log2(2 * self.d_len)), WIN=WINS.get(win, 0), Y_FLOATS=self.y_floats(),
                           KB_SLOT=self.kb_slot, KP=self.kp, DB=nd_min, ND_MIN=nd_min, ONE_HD=len(hds) == 1,
                           ND=hds[0] // 8,
                           MINT="" if nd_min >= 32 else f"<{nd_min}>", VPUT="" if nd_min >= 32 else f"<0, {nd_min}>",
                           QK_NORM=self.qk_norm, V_NORM=self.v_norm, CHAIN=self.chain, QCOPY=self.heads * max(hds),
                           NH_POW2=self.heads in (1, 2, 4), INV_HD={hd: f"1.0f / {hd}" for hd in hds},
                           by_hd=lambda vals: by("hd", vals), by_nd=lambda vals: by("nd", vals, lambda hd: hd // 8),
                           **(consts | {"F_ATT": F_ATT} | {k: v for k, v in globals().items() if k[:2] in ("A_", "M_")}))

    def workers(self, s: object, *, prefix: str, row: int, stack: int, a_depth: int = 1
                ) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        """The attention cores' workers on row ``row`` and their D streams (column, ObjectFifo) on the decode design's
        shared streams ``s`` (``overlay.decode._Streams``; kernel prefix s.p, object names from ``prefix``); ``a_depth``:
        their A objects in L1 (2: the next cache object arrives while the core works on one)."""
        p, P, chain = s.p, prefix, self.chain
        bf, i32 = np.dtype[bfloat16], np.dtype[np.int32]
        x_ty, a_ty = s.x_ty, s.a_ty
        AW = self.workspace()
        aws_ty = np.ndarray[(AW["size"] // 4,), i32]
        d_ty = np.ndarray[(self.d_len,), bf]
        tok_ty = np.ndarray[(16,), i32]
        i_a = AW["st"] // 4
        k_abegin = Kernel(f"{p}_abegin", "kernel.o", [x_ty, aws_ty, np.int32])
        k_anb = Kernel(f"{p}_anb", "kernel.o", [aws_ty])
        k_aq = Kernel(f"{p}_aq", "kernel.o", [x_ty, aws_ty])
        k_aq0 = Kernel(f"{p}_aq0", "kernel.o", [aws_ty])
        k_aqpan = Kernel(f"{p}_aqpan", "kernel.o", [aws_ty])
        k_akv = Kernel(f"{p}_akv", "kernel.o", [x_ty, aws_ty, np.int32])
        k_aobj = Kernel(f"{p}_aobj", "kernel.o", [a_ty, aws_ty])
        k_arec = Kernel(f"{p}_arec", "kernel.o", [a_ty, aws_ty, d_ty])
        k_aend = Kernel(f"{p}_aend", "kernel.o", [aws_ty, aws_ty])
        k_aemit = Kernel(f"{p}_aemit", "kernel.o", [aws_ty, d_ty])

        def attn_fn(a_f, x_f, d_f, *rest):
            # tl / tr: the chain's tokens from the west neighbour / to the east one (columns chain h .. chain h + chain
            # - 1): a column produces tr around its attention step (its partial stays in its workspace until the east
            # one merged it), consumes tl around its end (the merge of the west partial, pw)
            rest = list(rest)
            col = rest.pop()
            j = col % chain
            tl = rest.pop(0) if j > 0 else None
            tr = rest.pop(0) if j < chain - 1 else None
            abegin, anb, aq, aq0, aqpan, akv, aobj, arec, aend, aemit, ws, pw = rest
            for _ in forever_():
                x = x_f.acquire(1)
                abegin(x, ws, col)
                x_f.release(1)
                for _ in range_(ws[i_a + A_NPS]):  # (a header's partial sums object)
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_a + A_NNO]):  # (a main core's normed phase: as it consumes them)
                    a_f.acquire(1)
                    anb(ws)
                    for _ in range_(ws[i_a + A_NXN]):
                        x_f.acquire(1)
                        x_f.release(1)
                    a_f.release(1)
                for _ in range_(ws[i_a + A_NXSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_a + A_NASKIP]):
                    a_f.acquire(1)
                    a_f.release(1)
                for _ in range_(ws[i_a + A_NQP]):  # (an attention step)
                    if tr is not None:
                        tr.acquire(1)
                    aq0(ws)  # (its first q object, from abegin)
                    for _ in range_(ws[i_a + A_NQ]):
                        x = x_f.acquire(1)
                        aq(x, ws)
                        x_f.release(1)
                    aqpan(ws)
                    for _ in range_(ws[i_a + A_NKV]):
                        x = x_f.acquire(1)
                        akv(x, ws, col // chain)
                        x_f.release(1)
                    for _ in range_(ws[i_a + A_NOBJ]):
                        a = a_f.acquire(1)
                        aobj(a, ws)
                        for _ in range_(ws[i_a + A_NE]):  # (records to emit from it)
                            d = d_f.acquire(1)
                            arec(a, ws, d)
                            d_f.release(1)
                        a_f.release(1)
                    if tl is not None:
                        tl.acquire(1)
                    aend(ws, pw)
                    if tl is not None:
                        tl.release(1)
                    if tr is not None:
                        tr.release(1)
                    for _ in range_(ws[i_a + A_NE]):  # (o)
                        d = d_f.acquire(1)
                        aemit(ws, d)
                        d_f.release(1)

        aws = [Buffer(aws_ty, name=f"{P}_aws{c}", initial_value=np.zeros(AW["size"] // 4, np.int32),
                      tile=Tile(c, row)) for c in range(COLS)]
        toks = {c: ObjectFifo(tok_ty, name=f"{P}_T{c}", depth=1) for c in range(COLS) if c % chain < chain - 1}
        workers, d_fifos = [], []
        for c in range(COLS):
            d = ObjectFifo(d_ty, name=f"{P}_D{c}", depth=2)
            d_fifos.append((c, d))
            j = c % chain
            links = ([toks[c - 1].cons()] if j > 0 else []) + ([toks[c].prod()] if j < chain - 1 else [])
            workers.append(Worker(attn_fn, [s.a_l1[c].cons(depth=a_depth), s.x_l1[c].cons(depth=1), d.prod(), *links,
                                            k_abegin, k_anb, k_aq, k_aq0, k_aqpan, k_akv, k_aobj, k_arec, k_aend, k_aemit,
                                            aws[c], aws[c - 1] if j > 0 else aws[c], c],
                                  tile=Tile(c, row), stack_size=stack, dynamic_objfifo_lowering=True, while_true=False))
        return workers, d_fifos


# ----------------------------------------------------------------------
# Host side: the RoPE tables, the score constant
# ----------------------------------------------------------------------


def rope_tables(inv_freq: np.ndarray, positions: list[int] | range, rot: int) -> np.ndarray:
    """(SMAX, 2 rot) fp32: per slot [cos | sin] of the rotated pairs' angles position inv_freq[i], i < rot (unused slots
    zero)."""
    out = np.zeros((SMAX, 2 * rot), np.float32)
    for s, pos in enumerate(positions):
        th = pos * np.asarray(inv_freq)[:rot]
        out[s, :rot], out[s, rot:] = np.cos(th), np.sin(th)
    return out


def score_bits(c: float) -> tuple[int, int]:
    """The bf16 hi, lo bits of the score constant c log2(e) (scores q k c in log2 units)."""
    hi, lo = bf16_hilo(c / math.log(2))
    return bf16_bits(hi), bf16_bits(lo)
