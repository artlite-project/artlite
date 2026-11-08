"""The DFlash2 block drafter's cores (``dflash``): a design of its own on the shared decode overlay (``decode.design``, 8 slots:
the main and helper cores as every model's), in a hardware context of its own, with

- the conv core (c, 2), which holds the block's residual: res [k block 80][slot 8][8] fp32 of the column's 640 channels. It
  emits x for the next normed phase, computes conv-in from res, the phase's rows (dyn) and the norm's sums of squares (``sq``,
  the helper's, in this core's memory), conv-out from the K-group sums of an output phase, into res. A group of 16 channels
  and its 2 + 2 taps' coefficients stay in one column: dyn's rows are the column's own (``dflash.conv_rows``).
- the attention core (c, 5): a KV head per column, its 4 query heads as octets of the 8 slots' rows (``attn``'s one-pass form: an
  MMUL panel of 8 rows, single bfp16 Q and P), non-causal over the block.

Per block position p + s a row attends to the ``window`` positions up to its own and to the whole block. A KV head's cache (the
buffer DKV) keeps closed records only, immutable, in a ring (``DFlashShape.ring``) of A objects, one per 32 positions (the prefill's
chunks start at multiples of it): [K | V] of 4 records, both in pairs [d-block][record 2] (``decode.kv``, kp = 1). The positions past the last closed record (at most 7 of the sequence, then the block's 8: always
inside two records) are one A object, the open object: exact rows [K | V][record & 1][d-block 16][position 8][8] bf16. The block's
K and V go into the core's copy of it and nowhere else; an injection (rows of the verified tokens) writes the object back, and a
record it closes into the ring, as K and V panels.

A step of the core (the F_ATT object, its rope object, the layer's norms, a q object per slot): the open object first, its pair of
records scored and added (the rows' references m are their maxima there: each row's own position is in it), then the ring's objects in
position order: scores, P, y += P V. The window's edge and the ring's slots past the closed records are
masked per lane (P exactly 0). At the end o = y / l goes to the main core (R), which relays it to C.

Conv core objects (``F_CV``, for the others an object of another core; ``M_CVOP``):

- ``CV_LOAD``: payload objects of res (fp32 [k block][slot 8][8]), the block's input rows; then x goes out (``M_CVNE`` D objects);
- ``CV_IN``: the object's payload [w 640 | b0 640 | b1 640] bf16 (the norm's weights, the conv's base of side 0, tap 0 and 1), then one
  object of dyn [slot 8][192] bf16 (rows [side 2][tap 2][40 groups]: side 1's stay for ``CV_OUT``): the phase's input x from res, x =
  res w / rms, out_s = (b0 + d0_s) x_s + (b1 + d1_s) x_(s - 1) as bf16 x on D;
- ``CV_OUT``: the payload [b0 640 | b1 640], then objects of y ([k block][slot 8][8] bf16, 32 k blocks each): res += (b0 + d0_s) y_s +
  (b1 + d1_s) y_(s - 1), then x from res on D.
"""

import numpy as np
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

from artlite.decode import design as OD
from artlite.decode.design import C_LEN, META, XPAY, DecodeOverlayOperator, _Streams
from artlite.decode.kv import Records
from artlite.model_config import DFlashConfig
from artlite.operators.base import JINJA_ENV
from artlite.operators.design import COLS
from artlite.operators.util import Attr, regions

from .attn import _PV4, _QPAN, OBJ_BYTES

HD, QC, QH = 128, 4, 8 * 128  # (a head's dims, the query heads per column, the rows' stride of an octet)
ND = HD // 8
REC = Records(HD, 1)
REC_K, REC_V, ROWS_B = REC.k_bytes, REC.v_bytes, REC.rows_bytes
POS_PER_GROUP = 32  # an A object of the ring: 4 records [K pairs 2 | V pairs 2]
MAX_CHUNK = 2048  # a prefill chunk's positions at most
MAX_PADDING = 256  # a prefill chunk's padding rows at most (they write records too: over the ring's oldest slots, not the window's)
OPEN_BYTES = 4 * ROWS_B  # the open object's rows
PAIR = 2 * REC_K  # a pair of records
PO = 288  # an octet's P panels of 4 records
CHUNK = 4 * PO  # the 4 octets' P panels of 4 records
F_ATT = 1 << 10  # M_FLAGS of an attention step's first object
F_CV = 1 << 11  # M_FLAGS of a conv core's object
CV_LOAD, CV_IN, CV_OUT = range(3)
D_BYTES = 2048  # a conv core's D object: 16 k blocks of x
XK = 80  # k blocks of a column's x
H8 = 8 * XK  # a column's channels
GROUPS = H8 // 16  # conv groups of a column
DYN_ROWS = 192  # dyn's rows per slot in its object: whole C objects of its 160
M_CVOP, M_CVNE, M_POS, M_INJ, M_KEEP, M_ANOBJ, M_G0 = range(OD.M_MODEL, OD.M_MODEL + 7)
# conv core state words (the first ones are trip counts of its program)
V_NSKIP, V_NOBJ, V_NE, V_OP, V_I, V_IE = range(6)
# attention core state words (the first ones are trip counts of its program)
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NSTEP, A_NQ, A_NE1, A_NOBJ, A_NE, A_XL, A_LPER, A_P, A_S, A_INJ, A_KEEP, A_I, A_G0, A_QS,
 A_EIDX, A_SOFT, A_NRM, A_ROPE, A_KO, A_VO) = range(24)
STATE_WORDS = 32
assert A_VO < STATE_WORDS
STACK = 1600
V_STACK = 1024
NE_BLOCK = QC * ND  # R objects of a block's o
NE_INJ = (OPEN_BYTES + 2 * REC_K) // (2 * C_LEN)  # R objects of an injection: the open object, a record's K and V panels

_COMMON = r"""
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))
#define AP(T, w) ((T *)(uintptr_t)((const int32_t *)ws)[w])
"""

_CONV = JINJA_ENV.from_string(r"""
// The conv core (``dflash_core``'s docstring). Offsets in bytes: DFlashShape.conv_workspace()
#define VW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// The 8 slots' coefficients of group g of a tap (dyn [slot 8][192]: row r) -> 64 lanes, [slot][8]
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_slots(const bfloat16 *__restrict dyn, int r)
{
    return al::concat({% for s in range(8) %}aie::broadcast<bfloat16, 8>(dyn[{{ DYN_ROWS * s }} + r]){{ ", " if not loop.last }}{% endfor %});
}

// 8 values -> 64 lanes, [slot][8]
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_rep(aie::vector<bfloat16, 8> b)
{
    return b.grow_replicate<64>();
}

// [slot][8] -> each slot's row its predecessor's, slot 0 zeros
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_prev(aie::vector<bfloat16, 64> v)
{
    return aie::shuffle_up_fill(v, aie::zeros<bfloat16, 64>(), 8);
}

}  // namespace

extern "C" {

// A new X object: the core's (``CV_*``: its payload's constants, ``M_SKIP`` objects after it), or another's to skip
__attribute__((minsize)) void {{ p }}_vbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = VW(int32_t, {{ v.st }});
    const int flags = m[{{ M_FLAGS }}];
    st[{{ V_NSKIP }}] = 0;
    st[{{ V_NOBJ }}] = 0;
    st[{{ V_NE }}] = 0;
    if (flags & {{ F_CV }}) {
        st[{{ V_NOBJ }}] = m[{{ M_SKIP }}];
        st[{{ V_NE }}] = m[{{ M_CVNE }}];
        st[{{ V_OP }}] = m[{{ M_CVOP }}];
        st[{{ V_I }}] = 0;
        st[{{ V_IE }}] = 0;
        al::copy<{{ 3 * H8 }}, 32>(VW(bfloat16, {{ v.cs }}), x + {{ 2 * META }});
    } else if (!(flags & {{ F_MAIN }})) {
        st[{{ V_NSKIP }}] = m[{{ M_SKIP }}];
    } else if (!(flags & {{ F_LOADR }})) {  // (its input objects: of 8 / M_SP strips)
        const int sp = m[{{ M_SP }}], lper = sp == 1 ? 3 : (sp == 2 ? 2 : (sp == 4 ? 1 : 0));
        st[{{ V_NSKIP }}] = (m[{{ M_NX }}] + (1 << lper) - 1) >> lper;
    }
}

// The next object after a ``CV_*`` object
__attribute__((minsize)) void {{ p }}_vobj(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = VW(int32_t, {{ v.st }});
    const int op = st[{{ V_OP }}], i = st[{{ V_I }}];
    st[{{ V_I }}] = i + 1;
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    if (op == {{ CV_IN }}) {
        al::copy<{{ 8 * DYN_ROWS }}, 32>(VW(bfloat16, {{ v.dyn }}), pay);
        return;
    }
    float *__restrict r = VW(float, {{ v.res }});
    if (op == {{ CV_LOAD }}) {
        al::copy<{{ XPAY // 4 }}, 16>(r + 1024 * i, (const float *)pay);
        return;
    }
    r += 2048 * i;
    // (y's next 32 k blocks, 16 in the last object: conv-out into res)
    const bfloat16 *__restrict b = VW(bfloat16, {{ v.cs }}) + 32 * 8 * i;
    const bfloat16 *__restrict dyn = VW(bfloat16, {{ v.dyn }});
    const int k0 = 32 * i, n = i == {{ XK // 32 }} ? {{ XK % 32 }} : 32;
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < n; k++) {
        const int g = (k0 + k) >> 1;
        const aie::vector<bfloat16, 64> d0 = {{ p }}_slots(dyn, {{ 2 * GROUPS }} + g), d1 = {{ p }}_slots(dyn, {{ 3 * GROUPS }} + g);
        const aie::vector<bfloat16, 64> b0 = {{ p }}_rep(aie::load_v<8>(b + 8 * k)), b1 = {{ p }}_rep(aie::load_v<8>(b + {{ H8 }} + 8 * k));
        const aie::vector<bfloat16, 64> y = aie::load_v<64>(pay + 64 * k), yp = {{ p }}_prev(y);
        aie::accum<accfloat, 64> a(aie::load_v<64>(r + 64 * k));
        a = aie::mac(aie::mac(aie::mac(aie::mac(a, b0, y), d0, y), b1, yp), d1, yp);
        aie::store_v(r + 64 * k, a.template to_vector<float>());
    }
}

// The next D object (16 k blocks of x): conv-in of the phase, or res
void {{ p }}_vout(int32_t *__restrict ws, bfloat16 *__restrict d, const float *__restrict sq)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = VW(int32_t, {{ v.st }});
    const int e = st[{{ V_IE }}];
    st[{{ V_IE }}] = e + 1;
    const float *__restrict r = VW(float, {{ v.res }}) + 64 * 16 * e;
    if (st[{{ V_OP }}] != {{ CV_IN }}) {
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 16; k++)
            aie::store_v(d + 64 * k, aie::accum<accfloat, 64>(aie::load_v<64>(r + 64 * k)).template to_vector<bfloat16>());
        return;
    }
    const bfloat16 *__restrict cs = VW(bfloat16, {{ v.cs }}) + 8 * 16 * e;
    const bfloat16 *__restrict dyn = VW(bfloat16, {{ v.dyn }});
    float *__restrict ir = VW(float, {{ v.ir }});
    if (!e) {  // (the slots' 1 / rms, from the helper's sums of squares, as bf16 hi | lo per lane)
        const aie::vector<float, 8> ss = al::group8<false>(aie::load_v<64>(sq));
        aie::store_v(ir, al::bcast8(al::inv_rms<{{ H }}, 8>(ss.template grow<16>(), {{ eps }}f).template extract<8>(0)));
    }
    aie::vector<bfloat16, 64> ih, il;
    al::split2<64>(aie::load_v<64>(ir), ih, il);
    const int k0 = 16 * e;
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 16; k++) {
        const int g = (k0 + k) >> 1;
        const aie::vector<bfloat16, 64> xb = aie::accum<accfloat, 64>(aie::load_v<64>(r + 64 * k)).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> w = {{ p }}_rep(aie::load_v<8>(cs + 8 * k));
        const aie::vector<float, 64> xr = aie::mac(aie::mul(xb, ih), xb, il).template to_vector<float>();
        const aie::vector<bfloat16, 64> xn = al::mulc<64>(xr, w).template to_vector<bfloat16>(), xp = {{ p }}_prev(xn);
        const aie::vector<bfloat16, 64> b0 = {{ p }}_rep(aie::load_v<8>(cs + {{ H8 }} + 8 * k)), b1 = {{ p }}_rep(aie::load_v<8>(cs + {{ 2 * H8 }} + 8 * k));
        const aie::vector<bfloat16, 64> d0 = {{ p }}_slots(dyn, g), d1 = {{ p }}_slots(dyn, {{ GROUPS }} + g);
        const aie::accum<accfloat, 64> a = aie::mac(aie::mac(aie::mac(aie::mul(b0, xn), d0, xn), b1, xp), d1, xp);
        aie::store_v(d + 64 * k, a.template to_vector<bfloat16>());
    }
}

}  // extern "C"
""")

_ATT = JINJA_ENV.from_string(r"""
// The attention core (``dflash_core``'s docstring). Offsets in bytes: DFlashShape.workspace(); AP: a region whose address is in
// state word w (DFlashShape.ext: in the main core's memory)
""" + _COMMON + r"""
namespace {

// x (256 bf16 stride 128) -> fp32 out = x / rms(x) w, then the NeoX rotation of all HD dims (cs, sn: 64 fp32 each)
__attribute__((noinline)) void {{ p }}_norm_rope(const bfloat16 *__restrict x, const float *__restrict w,
                                                 const float *__restrict cs, const float *__restrict sn,
                                                 float *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::mac_square(aie::zeros<accfloat, 64>(), aie::load_v<64>(x));
    ss = aie::mac_square(ss, aie::load_v<64>(x + 64));
    const float inv = al::inv_rms<{{ HD }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    aie::vector<bfloat16, 32> ih, il;  // (x is bf16: x / rms = x ih + x il, mulff's products less x's lo part, 0)
    al::split2<32>(aie::broadcast<float, 32>(inv), ih, il);
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ HD }}; i += 32) {
        const aie::vector<bfloat16, 32> xb = aie::load_v<32>(x + i);
        aie::store_v(out + i, al::mulff<32>(aie::mac(aie::mul(xb, ih), xb, il).template to_vector<float>(), aie::load_v<32>(w + i)));
    }
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ HD // 2 }}; i += 32) {
        const aie::vector<float, 32> a = aie::load_v<32>(out + i), b = aie::load_v<32>(out + {{ HD // 2 }} + i);
        const aie::vector<float, 32> c = aie::load_v<32>(cs + i), s = aie::load_v<32>(sn + i);
        aie::store_v(out + i, aie::sub(al::mulff<32>(a, c), al::mulff<32>(b, s)));
        aie::store_v(out + {{ HD // 2 }} + i, aie::add(al::mulff<32>(a, s), al::mulff<32>(b, c)));
    }
}

// Scores of a pair of K records ([d-block][record 2]) against two octets (their rows' Q panels, [d-block][octet 2]), from the rows'
// baselines mb ([octet][set 2][b 32 | 1 / l 32]): so[256 o + 128 h + 32 r + j] = s + b (log2 units; set h, record r of the pair
// at ``at``), every K panel popped once for the octets
__attribute__((noinline)) void {{ p }}_qk1(const uint8_t *__restrict k, const uint8_t *__restrict qa, const float *__restrict mb,
                                           float *__restrict so, int at)
{
    al::MM8 a[2], b[2];
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < 2; o++) {
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
        for (int o = 0; o < 2; o++) {
            const al::BV Q = qs.pop();
            a[o].mac(Q, aie::op_transpose(K0));
            b[o].mac(Q, aie::op_transpose(K1));
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < 2; o++) {
        const aie::vector<float, 64> x = a[o].to_accum().template to_vector<float>(), z = b[o].to_accum().template to_vector<float>();
        aie::store_v(so + 256 * o + at, x.template extract<32>(0));
        aie::store_v(so + 256 * o + 128 + at, x.template extract<32>(1));
        aie::store_v(so + 256 * o + at + 32, z.template extract<32>(0));
        aie::store_v(so + 256 * o + 160 + at, z.template extract<32>(1));
    }
}

{# P's 2^x in two steps (bias 192; masked scores -16384) #}
{% set e192 = km.exp2(32, bias=192, below="bounded", above="clamp") %}
{{ e192.parts.reduce.src }}
{{ e192.parts.poly.src }}

""" + _PV4 + r"""// y += P V of the 4 octets (two calls) of pairs v, vb and their P panels (pp: [octet 4][P 4])
__attribute__((noinline)) void {{ p }}_pvs(const uint8_t *v, const uint8_t *vb, const uint8_t *__restrict pp, float *__restrict y)
{
    {{ p }}_pv4<2>(v, vb, pp, (const float __aie_dm_resource_c *)y, (float __aie_dm_resource_d *)y);
    {{ p }}_pv4<2>(v, vb, pp + {{ 2 * PO }}, (const float __aie_dm_resource_c *)(y + {{ 128 * (ND + 1) }}),
                   (float __aie_dm_resource_d *)(y + {{ 128 * (ND + 1) }}));
}

// P = 2^(s - m) of 4 records' scores of the 4 octets: P panels at pp, l += P (two passes: each pipelined)
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

__attribute__((noinline, minsize)) void {{ p }}_soft4(int32_t *__restrict ws, uint8_t *__restrict pp)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
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
}

""" + _QPAN + r"""// A row's K (HD fp32) into its row of the open rows ([d-block][position 8][8] bf16: dst its first values)
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

// A record's rows -> its K, V panels ([d-block], as ``kpair`` and ``vpair`` hold them)
__attribute__((noinline, minsize)) void {{ p }}_krec(const bfloat16 *__restrict r, uint8_t *__restrict out)
{
    al::BOut<> po((bfp16ebs8 *)out);
    AIE_LOOP_NO_UNROLL
    for (int db = 0; db < {{ ND }}; db++)
        po.push(aie::accum<accfloat, 64>(aie::load_v<64>(r + 64 * db)).template to_vector<bfp16ebs8>());
}

// Records' rows ([d-block][position 8][8] bf16) -> K panels [d-block][record 2] (a block per position's 8 dims)
__attribute__((noinline)) void {{ p }}_kpair(const bfloat16 *__restrict r0, const bfloat16 *__restrict r1, uint8_t *__restrict out)
{
    al::BOut<> po((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        po.push(aie::accum<accfloat, 64>(aie::load_v<64>(r0 + 64 * db)).template to_vector<bfp16ebs8>());
        po.push(aie::accum<accfloat, 64>(aie::load_v<64>(r1 + 64 * db)).template to_vector<bfp16ebs8>());
    }
}

// Records' rows, those from position n0 / n1 on as zeros -> V panels [d-block][record 2] (a block per dim over the positions)
__attribute__((noinline)) void {{ p }}_vpair(const bfloat16 *__restrict r0, const bfloat16 *__restrict r1, uint8_t *__restrict out, int n0, int n1)
{
    al::BOut<> po((bfp16ebs8 *)out);
    const aie::mask<64> k0 = aie::mask<64>::from_uint64(n0 < 8 ? (1ull << (8 * n0)) - 1 : ~0ull);
    const aie::mask<64> k1 = aie::mask<64>::from_uint64(n1 < 8 ? (1ull << (8 * n1)) - 1 : ~0ull);
    for (int db = 0; db < {{ ND }}; db++) {
        const aie::vector<bfloat16, 64> a = aie::select(aie::zeros<bfloat16, 64>(), aie::load_v<64>(r0 + 64 * db), k0);
        const aie::vector<bfloat16, 64> b = aie::select(aie::zeros<bfloat16, 64>(), aie::load_v<64>(r1 + 64 * db), k1);
        po.push(aie::accum<accfloat, 64>(aie::transpose(a, 8, 8)).template to_vector<bfp16ebs8>());
        po.push(aie::accum<accfloat, 64>(aie::transpose(b, 8, 8)).template to_vector<bfp16ebs8>());
    }
}

// Of rows s0 .. s0 + 3 of an octet the lanes of record r whose positions are not attended: past ``end`` (exclusive; every row) or
// before the row's window (row s: from lo + s on)
__attribute__((noinline, minsize)) uint32_t {{ p }}_masked(int s0, int r, int lo, int end)
{
    const int hi = end - 8 * r;  // (valid below it)
    const uint32_t past = hi <= 0 ? 0xFFu : hi >= 8 ? 0u : (0xFFu << hi) & 0xFFu;
    uint32_t m = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < 4; q++) {
        int n = lo + s0 + q - 8 * r;
        n = n < 0 ? 0 : (n > 8 ? 8 : n);
        m |= (past | ((1u << n) - 1)) << (8 * q);
    }
    return m;
}

// Scores of 4 records from r on that the rows do not attend: -16384 (P exactly 0)
__attribute__((noinline, minsize)) void {{ p }}_kmask(int32_t *__restrict ws, int r, int lo, int end)
{
    const aie::vector<float, 32> neg = aie::broadcast<float, 32>(-16384.0f);
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < 4; q++) {
        const int a = r + q;
        if (8 * a >= lo + 7 && 8 * (a + 1) <= end)
            continue;
        const aie::mask<32> m0 = aie::mask<32>::from_uint32({{ p }}_masked(0, a, lo, end));
        const aie::mask<32> m1 = aie::mask<32>::from_uint32({{ p }}_masked(4, a, lo, end));
        float *__restrict so = AW(float, {{ a.sc }}) + 32 * q;
        AIE_LOOP_NO_UNROLL
        for (int set = 0; set < {{ SETS }}; set += 2) {
            aie::store_v(so + 128 * set, aie::select(aie::load_v<32>(so + 128 * set), neg, m0));
            aie::store_v(so + 128 * set + 128, aie::select(aie::load_v<32>(so + 128 * set + 128), neg, m1));
        }
    }
}

// The open pair's scores in (from baselines 0, the 4 records' last two all masked): each row's reference m its maximum there, baselines
// b = 192 - m, scores as A = s + b from here on
__attribute__((noinline, minsize)) void {{ p }}_m0(int32_t *__restrict ws)
{
    AIE_LOOP_NO_UNROLL
    for (int set = 0; set < {{ SETS }}; set++) {
        float *__restrict ss = AW(float, {{ a.sc }}) + 128 * set;
        const aie::vector<float, 32> m = al::max8(aie::max(aie::load_v<32>(ss), aie::load_v<32>(ss + 32)));
        const aie::vector<float, 32> b = aie::sub(aie::broadcast<float, 32>(192.0f), m);
        aie::store_v(AW(float, {{ a.ml }}) + 64 * set, b);
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 128; k += 32)
            aie::store_v(ss + k, aie::add(aie::load_v<32>(ss + k), b));
    }
}

}  // namespace

extern "C" {

// A new X object: F_ATT (a step: its rope and norm objects, q objects and the open object follow), a main core's phase (its input
// objects and A objects skipped in the main core's order, the norm-weight objects around their inputs), another core's. mw: the
// main core's workspace (its idle regions: DFlashShape.ext)
__attribute__((minsize)) void {{ p }}_abegin(const bfloat16 *__restrict x, int32_t *__restrict ws, int32_t *__restrict mw)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int flags = m[{{ M_FLAGS }}];
    for (int i = 0; i < {{ A_LPER }}; i++)
        st[i] = 0;
    if (flags & {{ F_MAIN }}) {
        if (!(flags & {{ F_LOADR }})) {
            const int nx = m[{{ M_NX }}], sp = m[{{ M_SP }}], lper = sp == 1 ? 3 : (sp == 2 ? 2 : (sp == 4 ? 1 : 0));
            st[{{ A_LPER }}] = lper;
            if (flags & {{ F_NORM }}) {
                st[{{ A_NNO }}] = (nx + {{ NX_PER_NORM - 1 }}) / {{ NX_PER_NORM }};
                st[{{ A_XL }}] = nx;
            } else {
                st[{{ A_NXSKIP }}] = (nx + (1 << lper) - 1) >> lper;
            }
            st[{{ A_NASKIP }}] = m[{{ M_NT }}];
        }
        return;
    }
    if (!(flags & {{ F_ATT }})) {
        st[{{ A_NXSKIP }}] = m[{{ M_SKIP }}];
        return;
    }
    const int S = m[{{ M_NS }}], inj = m[{{ M_INJ }}];
    st[{{ A_NSTEP }}] = 1;
    st[{{ A_NQ }}] = S;
    st[{{ A_S }}] = S;
    st[{{ A_P }}] = m[{{ M_POS }}];
    st[{{ A_INJ }}] = inj;
    st[{{ A_KEEP }}] = m[{{ M_KEEP }}];
    st[{{ A_NE1 }}] = inj ? {{ NE_INJ }} : 0;
    st[{{ A_NOBJ }}] = m[{{ M_ANOBJ }}];
    st[{{ A_NE }}] = inj ? 0 : {{ NO_OBJ }};
    st[{{ A_G0 }}] = m[{{ M_G0 }}];
    st[{{ A_I }}] = 0;
    st[{{ A_QS }}] = 0;
    st[{{ A_EIDX }}] = 0;
{% for w, off in at %}
    st[{{ w }}] = (int32_t)((uintptr_t)mw + {{ off }});
{% endfor %}
}

// A main core's norm-weight object: the input objects under it
__attribute__((minsize)) void {{ p }}_anb(int32_t *__restrict ws)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int left = st[{{ A_XL }}], n = left > {{ NX_PER_NORM }} ? {{ NX_PER_NORM }} : left;
    st[{{ A_NXN }}] = (n + (1 << st[{{ A_LPER }}]) - 1) >> st[{{ A_LPER }}];
    st[{{ A_XL }}] = left - n;
}

// The rope object: the rows' cos, sin ([slot 8][cos 64 | sin 64] fp32); the norms: q_norm x log2(e) / sqrt(HD), k_norm
__attribute__((minsize)) void {{ p }}_arope(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    al::copy<{{ 128 * 8 }}, 16>(AP(float, {{ A_ROPE }}), (const float *)(x + {{ 2 * META }}));
}

__attribute__((minsize)) void {{ p }}_anorm(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    al::copy<{{ 2 * HD }}, 16>(AP(float, {{ A_NRM }}), (const float *)(x + {{ 2 * META }}));
}

// The next row's object ([q 4 heads | K | V], an injection's [K | V]): q (head h's octet h, row s) normed, rotated; K (normed, rotated)
// and V into the open rows a (rows an injection does not keep: none)
__attribute__((minsize)) void {{ p }}_aq(const bfloat16 *__restrict x, int32_t *__restrict ws, uint8_t *__restrict a)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int s = st[{{ A_QS }}], inj = st[{{ A_INJ }}];
    st[{{ A_QS }}] = s + 1;
    const bfloat16 *__restrict pay = x + {{ 2 * META }};
    const float *__restrict nrm = AP(float, {{ A_NRM }});
    const float *__restrict cs = AP(float, {{ A_ROPE }}) + {{ HD }} * s;
    if (!inj) {
        AIE_LOOP_NO_UNROLL
        for (int h = 0; h < {{ QC }}; h++)
            {{ p }}_norm_rope(pay + {{ HD }} * h, nrm, cs, cs + {{ HD // 2 }}, AW(float, {{ a.y }}) + {{ HD }} * (8 * h + s));
    } else if (s >= st[{{ A_KEEP }}]) {
        return;
    }
    const bfloat16 *__restrict kv = pay + (inj ? 0 : {{ QC * HD }});
    const int q = st[{{ A_P }}] + s, slot = (q >> 3) & 1, j = q & 7;
    float *__restrict kf = AW(float, {{ a.sc }});  // (the scores' region, before the cache)
    {{ p }}_norm_rope(kv, nrm + {{ HD }}, cs, cs + {{ HD // 2 }}, kf);
    bfloat16 *__restrict rows = (bfloat16 *)a;
    {{ p }}_krow(kf, rows + {{ ROWS_B // 2 }} * slot + 8 * j);
    al::v_put<{{ ND }}>(rows + {{ ROWS_B // 2 }} * (2 + slot) + 8 * j, kv + {{ HD }}, {{ ND }});
}

// The rows are in. A block: the Q panels, y, l and the baselines zero, the open records' K and V panels, their scores (the references
// m), P, y += P V. An injection: the record it closes as K and V panels.
__attribute__((minsize)) void {{ p }}_aopen(int32_t *__restrict ws, uint8_t *__restrict a)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const bfloat16 *__restrict rows = (const bfloat16 *)a;
    const int p = st[{{ A_P }}], r = p >> 3, odd = r & 1;
    uint8_t *__restrict ko = AP(uint8_t, {{ A_KO }}), *__restrict vo = AP(uint8_t, {{ A_VO }});
    if (st[{{ A_INJ }}]) {
        const int end = p + st[{{ A_KEEP }}], rc = (end >> 3) - 1;  // (the record it closes, when it reaches its end)
        if (rc >= r) {
            const int sl = rc & 1;
            {{ p }}_krec(rows + {{ ROWS_B // 2 }} * sl, ko);
            al::v_enc<{{ ND }}>(rows + {{ ROWS_B // 2 }} * (2 + sl), vo, {{ ND }}, 8);
        }
        return;
    }
    const int S = st[{{ A_S }}], last = p + S;  // (the rows see the whole block)
    float *__restrict q = AW(float, {{ a.y }});
    {{ p }}_qpan<2>(q, AW(uint8_t, {{ a.qa0 }}));
    {{ p }}_qpan<2>(q + {{ 16 * HD }}, AW(uint8_t, {{ a.qa1 }}));
    al::zero<{{ 32 * (ND + 1) * SETS }}, 32>(q);
    al::zero<{{ 64 * SETS }}, 32>(AW(float, {{ a.ml }}));
    const int n0 = last - 8 * r, n1 = last - 8 * (r + 1);
    {{ p }}_kpair(rows + {{ ROWS_B // 2 }} * odd, rows + {{ ROWS_B // 2 }} * (1 - odd), ko);
    {{ p }}_vpair(rows + {{ ROWS_B // 2 }} * (2 + odd), rows + {{ ROWS_B // 2 }} * (3 - odd), vo, n0 > 8 ? 8 : n0, n1 < 0 ? 0 : n1);
    float *__restrict sc = AW(float, {{ a.sc }});
    {{ p }}_qk1(ko, AW(const uint8_t, {{ a.qa0 }}), AW(const float, {{ a.ml }}), sc, 0);
    {{ p }}_qk1(ko, AW(const uint8_t, {{ a.qa1 }}), AW(const float, {{ a.ml }}) + 256, sc + 512, 0);
    {{ p }}_kmask(ws, r, -(1 << 30), last);
    {{ p }}_m0(ws);
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    {{ p }}_soft4(ws, pp);
    {{ p }}_pvs(vo, vo, pp, AW(float, {{ a.y }}));
}

// The ring's next object: scores of its 4 records, P, y += P V (none of them attended: nothing)
void {{ p }}_aobj(uint8_t *__restrict a, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int i = st[{{ A_I }}];
    st[{{ A_I }}] = i + 1;
    const int p = st[{{ A_P }}], lo = p - {{ WINDOW }} + 1, end = 8 * (p >> 3), r = 4 * (st[{{ A_G0 }}] + i);
    if (8 * (r + 4) <= lo || 8 * r >= end)
        return;
    float *__restrict sc = AW(float, {{ a.sc }});
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 2; k++) {
        {{ p }}_qk1(a + {{ PAIR }} * k, AW(const uint8_t, {{ a.qa0 }}), AW(const float, {{ a.ml }}), sc, 64 * k);
        {{ p }}_qk1(a + {{ PAIR }} * k, AW(const uint8_t, {{ a.qa1 }}), AW(const float, {{ a.ml }}) + 256, sc + 512, 64 * k);
    }
    {{ p }}_kmask(ws, r, lo, end);
    uint8_t *__restrict pp = AW(uint8_t, {{ a.pp }});
    {{ p }}_soft4(ws, pp);
    {{ p }}_pvs(a + {{ 2 * PAIR }}, a + {{ 3 * PAIR }}, pp, AW(float, {{ a.y }}));
}

// The attention is through: 1 / l per row
__attribute__((minsize)) void {{ p }}_aend(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
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

// The next R object of o ([head 4][k block 16][slot 8][8] bf16 y / l)
void {{ p }}_aemit(int32_t *__restrict ws, bfloat16 *__restrict r)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}];
    st[{{ A_EIDX }}] = e + 1;
    const int o = e >> 4, db = e & 15;
    const float *__restrict y = AW(float, {{ a.y }}) + {{ 64 * (ND + 1) }} * o + 32 * db;
    const float *__restrict il = AW(float, {{ a.ml }}) + 128 * o + 32;
    const aie::vector<float, 32> a = al::mulff<32>(aie::load_v<32>(y), aie::load_v<32>(il));
    const aie::vector<float, 32> b = al::mulff<32>(aie::load_v<32>(y + {{ 32 * (ND + 1) }}), aie::load_v<32>(il + 64));
    aie::store_v(r, aie::accum<accfloat, 64>(aie::concat(a, b)).template to_vector<bfloat16>());
}

// An injection's next R object: the open object, then the record it closes (K, V panels: the host drains them into the ring, or past
// it when none closes)
__attribute__((minsize)) void {{ p }}_aemit1(int32_t *__restrict ws, bfloat16 *__restrict r, const uint8_t *__restrict a)
{
    int32_t *__restrict st = AW(int32_t, {{ a.st }});
    const int e = st[{{ A_EIDX }}];
    st[{{ A_EIDX }}] = e + 1;
    const int n = {{ OPEN_BYTES // (2 * C_LEN) }}, nr = {{ REC_K // (2 * C_LEN) }};
    const uint8_t *__restrict src = e < n ? a + {{ 2 * C_LEN }} * e : e < n + nr ? AP(const uint8_t, {{ A_KO }}) + {{ 2 * C_LEN }} * (e - n)
                                                                                  : AP(const uint8_t, {{ A_VO }}) + {{ 2 * C_LEN }} * (e - n - nr);
    aie::store_v(r, aie::load_v<{{ C_LEN }}>((const bfloat16 *)src));
}

}  // extern "C"
""")


class DFlashShape:
    """The cores' sizes for a drafter ``cfg`` (hidden 5120, 8 KV heads of 128, 4 query heads per KV head), the conv and attention
    workspaces, the ring of groups a window spans."""

    def __init__(self, cfg: DFlashConfig, *, a_bytes: int = OBJ_BYTES):
        assert cfg.head_dim == HD and cfg.kv_heads == COLS and cfg.heads == QC * COLS and cfg.hidden == COLS * H8
        assert cfg.block == 8 and cfg.conv_kernel == 2 and cfg.conv_group == 16, cfg
        assert 4 * PAIR <= a_bytes and OPEN_BYTES <= a_bytes
        assert 3 * H8 * 2 <= XPAY and 8 * DYN_ROWS * 2 <= XPAY and 8 * 128 * 4 <= XPAY
        self.cfg, self.a_bytes = cfg, a_bytes
        self.window = cfg.window
        # (objects: a window's closed records in distinct ones, what a chunk's padding rows overwrite past them)
        self.ring = -(-(cfg.window + 8 + MAX_PADDING) // POS_PER_GROUP) + 1
        self.over = MAX_CHUNK // POS_PER_GROUP  # (objects past the ring that a prefill chunk's writes reach)
        self.sets = 2 * QC
        self.eps = cfg.rms_norm_eps

    def conv_workspace(self) -> dict[str, int]:
        """The conv core's workspace: state words, res, the constants of an object, dyn, the slots' 1 / rms."""
        return regions([("st", 4 * 16), ("res", 4 * 8 * 8 * XK), ("cs", 2 * 3 * H8), ("dyn", 2 * 8 * DYN_ROWS), ("ir", 4 * 64)])

    def workspace(self) -> dict[str, int]:
        """The attention core's workspace. y: per set its blocks [db][row 4 x 8] fp32, then l (before the panels: the q rows
        [row][HD] fp32); sc: per octet the scores of 4 records [set 2][record 4][32] (before the cache: a row's K); pp: their P
        panels, [octet][P 4]; qa0, qa1: the Q panels [d-block][octet 2]."""
        return regions([("st", 4 * STATE_WORDS), ("y", 4 * 32 * (ND + 1) * self.sets), ("ml", 4 * 64 * self.sets),
                        ("sc", 4 * 128 * self.sets), ("pp", CHUNK), ("qa0", 2 * 72 * ND), ("qa1", 2 * 72 * ND)])

    def ext(self) -> dict[str, int]:
        """Regions in the main core's idle memory, past its residual: offsets from there."""
        return regions([("nrm", 4 * 2 * HD), ("rope", 4 * 128 * 8), ("ko", PAIR), ("vo", PAIR)])

    def render(self, p: str, consts: dict[str, int], *, tail: int) -> str:
        e = self.ext()
        at = [(A_NRM, tail + e["nrm"]), (A_ROPE, tail + e["rope"]), (A_KO, tail + e["ko"]), (A_VO, tail + e["vo"])]
        v = dict(p=p, a=Attr(self.workspace()), v=Attr(self.conv_workspace()), at=at, QC=QC, SETS=self.sets, HD=HD, ND=ND, PO=PO, CHUNK=CHUNK,
                 PAIR=PAIR, REC_K=REC_K, REC_V=REC_V, ROWS_B=ROWS_B, OPEN_BYTES=OPEN_BYTES, C_LEN=C_LEN, eps=repr(self.eps),
                 WINDOW=self.window, META=META, XPAY=XPAY, H8=H8, XK=XK, GROUPS=GROUPS, DYN_ROWS=DYN_ROWS, H=self.cfg.hidden,
                 NO_OBJ=NE_BLOCK, NE_INJ=NE_INJ, CV_LOAD=CV_LOAD, CV_IN=CV_IN, CV_OUT=CV_OUT,
                 **(consts | {"F_ATT": F_ATT, "F_CV": F_CV} | {k: w for k, w in globals().items() if k.startswith(("A_", "V_", "M_"))}))
        return _CONV.render(**v) + _ATT.render(**v)


class DFlashOperator(DecodeOverlayOperator):
    """The drafter's design (module docstring): ``rows``: {tile format: rows per tile} of tiles of ``tile_bytes``; ``shape``: its
    cores' sizes."""

    PREFIX = "dfl"
    A_FWD = 2
    LENDS_WS = True
    CONV_ROW, ATTN_ROW = 2, 5
    CONV_AT = 0x4000  # (the conv workspace: regions 1 .. 3, below the helper's scratch)

    def __init__(self, *, hidden: int, rows: dict[str, int], shape: DFlashShape, tile_bytes: int = OBJ_BYTES, eps: float = 1e-6):
        self.shape = shape
        self.relay = True
        super().__init__(hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, slots=8, packed=True, bcast=True)

    def _consts(self) -> dict[str, int]:
        return super()._consts() | {k: v for k, v in globals().items() if k[:2] == "M_"}

    def _extra_src(self, p: str, consts: dict[str, int]) -> str:
        W = self.workspace()
        assert W["size"] - W["inv"] >= self.shape.ext()["size"], W
        return self.shape.render(p, {k: v for k, v in consts.items() if k[:2] in ("M_", "F_")}
                                 | {"NX_PER_NORM": self.nx_per_norm}, tail=W["inv"])

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS))

    def _pins(self) -> dict[str, int]:
        return {f"{self.PREFIX}_vws{c}": self.CONV_AT for c in range(COLS)}

    def _x_rows(self) -> tuple[int, ...]:
        return (self.CONV_ROW, self.ATTN_ROW)

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        p, P, sh = s.p, self.PREFIX, self.shape
        bf, i32 = np.dtype[bfloat16], np.dtype[np.int32]
        x_ty, a_ty = s.x_ty, s.a_ty
        VW, AW = sh.conv_workspace(), sh.workspace()
        vws_ty = np.ndarray[(VW["size"] // 4,), i32]
        aws_ty = np.ndarray[(AW["size"] // 4,), i32]
        mws_ty = np.ndarray[(self.workspace()["size"] // 4,), i32]
        d_ty = np.ndarray[(D_BYTES // 2,), bf]
        sq_ty = np.ndarray[(8 * self.slots,), np.dtype[np.float32]]
        i_v, i_a = VW["st"] // 4, AW["st"] // 4
        k_vbegin = Kernel(f"{p}_vbegin", "kernel.o", [x_ty, vws_ty])
        k_vobj = Kernel(f"{p}_vobj", "kernel.o", [x_ty, vws_ty])
        k_vout = Kernel(f"{p}_vout", "kernel.o", [vws_ty, d_ty, sq_ty])
        k_abegin = Kernel(f"{p}_abegin", "kernel.o", [x_ty, aws_ty, mws_ty])
        k_anb = Kernel(f"{p}_anb", "kernel.o", [aws_ty])
        k_arope = Kernel(f"{p}_arope", "kernel.o", [x_ty, aws_ty])
        k_anorm = Kernel(f"{p}_anorm", "kernel.o", [x_ty, aws_ty])
        k_aq = Kernel(f"{p}_aq", "kernel.o", [x_ty, aws_ty, a_ty])
        k_aopen = Kernel(f"{p}_aopen", "kernel.o", [aws_ty, a_ty])
        k_aobj = Kernel(f"{p}_aobj", "kernel.o", [a_ty, aws_ty])
        k_aend = Kernel(f"{p}_aend", "kernel.o", [aws_ty])
        k_aemit = Kernel(f"{p}_aemit", "kernel.o", [aws_ty, s.r_ty])
        k_aemit1 = Kernel(f"{p}_aemit1", "kernel.o", [aws_ty, s.r_ty, a_ty])

        def conv_fn(x_f, d_f, vbegin, vobj, vout, ws, sq):
            for _ in forever_():
                x = x_f.acquire(1)
                vbegin(x, ws)
                x_f.release(1)
                for _ in range_(ws[i_v + V_NSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_v + V_NOBJ]):
                    x = x_f.acquire(1)
                    vobj(x, ws)
                    x_f.release(1)
                for _ in range_(ws[i_v + V_NE]):
                    d = d_f.acquire(1)
                    vout(ws, d, sq)
                    d_f.release(1)

        def attn_fn(a_f, x_f, r_f, abegin, anb, arope, anorm, aq, aopen, aobj, aend, aemit, aemit1, ws, mws):
            for _ in forever_():
                x = x_f.acquire(1)
                abegin(x, ws, mws)
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
                for _ in range_(ws[i_a + A_NSTEP]):
                    x = x_f.acquire(1)
                    arope(x, ws)
                    x_f.release(1)
                    x = x_f.acquire(1)
                    anorm(x, ws)
                    x_f.release(1)
                    a = a_f.acquire(1)  # the open object
                    for _ in range_(ws[i_a + A_NQ]):
                        x = x_f.acquire(1)
                        aq(x, ws, a)
                        x_f.release(1)
                    aopen(ws, a)
                    for _ in range_(ws[i_a + A_NE1]):
                        r = r_f.acquire(1)
                        aemit1(ws, r, a)
                        r_f.release(1)
                    a_f.release(1)
                    for _ in range_(ws[i_a + A_NOBJ]):
                        a = a_f.acquire(1)
                        aobj(a, ws)
                        a_f.release(1)
                    aend(ws)
                    for _ in range_(ws[i_a + A_NE]):
                        r = r_f.acquire(1)
                        aemit(ws, r)
                        r_f.release(1)

        workers, d_fifos = [], []
        for c in range(COLS):
            d_fifos.append((c, ObjectFifo(d_ty, name=f"{P}_D{c}", depth=1)))
            vws = Buffer(vws_ty, name=f"{P}_vws{c}", initial_value=np.zeros(VW["size"] // 4, np.int32), tile=Tile(c, self.CONV_ROW))
            workers.append(Worker(conv_fn, [s.x_l1[c].cons(depth=1, **s.x_kw), d_fifos[c][1].prod(), k_vbegin, k_vobj, k_vout, vws, s.sq[c]],
                                  tile=Tile(c, self.CONV_ROW), stack_size=V_STACK, dynamic_objfifo_lowering=True, while_true=False))
            aws = Buffer(aws_ty, name=f"{P}_aws{c}", initial_value=np.zeros(AW["size"] // 4, np.int32), tile=Tile(c, self.ATTN_ROW))
            workers.append(Worker(attn_fn, [s.a_l1[c].cons(depth=2), s.x_l1[c].cons(depth=1, **s.x_kw), s.r_fifos[c].prod(), k_abegin, k_anb,
                                            k_arope, k_anorm, k_aq, k_aopen, k_aobj, k_aend, k_aemit, k_aemit1, aws, s.ws[c]],
                                  tile=Tile(c, self.ATTN_ROW), stack_size=STACK, dynamic_objfifo_lowering=True, while_true=False))
        return workers, d_fifos
