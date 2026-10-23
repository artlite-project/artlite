"""The GDN core of the multi-slot decode overlay (``qwen35/decode.py``): one Gated DeltaNet step of S <= 4 token
slots per GDN layer and column, with a lazy commit of the recurrent state.

Column c holds q / k heads {2c, 2c + 1} and its HC v heads (``image.gdn_v_heads``; v head h of the column reads local
k head h & 1). The heads come in pairs (2 c + 16 p, + 1). Per step the core gets:

- the main core's message (the gdn_in phase's rows, stream M, ``C_LEN`` values per object; object j of slots 0 .. S - 1,
  then j + 1): per slot [raw qkv (CH = (4 + HC) 128 channels: q 2 heads, k 2 heads, v HC heads) | z (HC x 128) | alpha,
  beta (HC fp32 each)], all of them before any X object of the step (the main core skips those only after its phase);
- X objects (the others skip them: the first one's metadata, ``F_GDN`` with ``M_SKIP``), each ``XPAY`` bytes of
  payload: the conv objects of the q / k blocks (block b: 128 channels, [taps 4 x (w hi | w lo) bf16 | history 7 x 128
  bf16 | pad 256 B]), the K object of the previous step's tokens ([token 4][k head 2][k hi | k lo]), then per pair of v
  heads: its two conv objects, the delta object of the previous step ([head 2][token 4][128] fp32) and its 32 state
  panels (fp32, ``gdn.state_panels``' order);
- from the gdn_in metadata object: S, the tokens to replay (``M_NREP``), the conv shift (``M_CSHIFT``: the accepted tokens
  of the previous step) and the commit flag (``M_COMMIT``).

A conv object's history holds the 3 rows before the previous step's tokens and those tokens' raw rows: the step's base
rows are rows cshift .. cshift + 2. Its new history ([base 3 | the step's S rows]) and pad go out on D in the object's
place. The pads: block 0 [d of the step's tokens (the next step's replay) [token 4][8] fp32 | dt 8 | A 8 fp32], blocks
1, 2 ssm_norm (64 fp32 each), passed through.

Lazy commit: per panel the core replays the previous step's ``M_NREP`` accepted tokens (S <- S + d S + k delta^T from
the kept k, delta, d: the same arithmetic as their first pass), writes that base state out, then runs the step's S
tokens on it in place (their updates are not written: the next step replays the accepted ones). With ``M_COMMIT`` it
writes the state after the step's tokens instead (S = 1 decoding, before a prefill). Outputs on D (objects of
``D_BYTES``): the q / k blocks' histories (2 blocks per object), the step's K object, and per pair: its v blocks'
histories, the 32 state panels, the delta object of the step's tokens and the gated outputs rms(o) ssm_norm silu(z)
of the pair's heads, slot-interleaved as the gdn_out phase reads them ([head 2][k block 16][slot 4][8] bf16, then
2 KB of padding).
"""

import math

import numpy as np

from ...model_config import QwenConfig
from .. import kmath
from ..base import JINJA_ENV
from ..design import COLS
from ..kernel_source import kernel_prefix
from ..util import Attr, bf16_hilo, regions

D = 128  # ssm_state: k and v dims of a head
PANEL = 4096  # bytes of a state panel (128 i x 8 j fp32)
D_BYTES = 4096  # bytes of a D object
SMAX = 4  # token slots
HIST = 3 + SMAX  # history rows of a conv object
F_GDN = 1 << 8  # M_FLAGS of the first GDN object of a step (the other cores skip it and M_SKIP objects after it)

# state words of the GDN core's workspace (the first ones are trip counts of its program)
(G_NSKIP, G_NMSG, G_ON, G_NCP, G_KO, G_PO, G_NS, G_NREP, G_CSHIFT, G_COMMIT, G_MJ, G_MS, G_CB, G_PAIR,
 G_PANEL) = range(15)
G_WORDS = 16


# The GDN math in fp32 (kmath, 16 lanes: one out-of-line exp2 for all of it; ``gm`` in the templates) and the helpers
# rendered with p, eps, gm, sigmoid (whether to define it) in an anonymous namespace: the decode's GDN core and the
# prefill scan (gdn_prefill.py: under GDN, one copy for its operators)
GDN = kernel_prefix("gdn")
EXP2 = kmath.exp2(16, prec="fp32")
MATH = {"exp2": EXP2, "expm1": kmath.expm1(16, exp_fn=EXP2), "softplus": kmath.softplus(16, exp_fn=EXP2),
        "sigmoid": kmath.sigmoid(16, exp_fn=EXP2), "silu": kmath.silu(32)}
HELPERS = r"""{{ gm.expm1.src }}
{{ gm.softplus.src }}
{% if sigmoid %}
{{ gm.sigmoid.src }}
{% endif %}
{# silu(x) of 32 bf16 lanes: one copy (inlined it needs a large frame) #}
{{ gm.silu.src }}

// x (64 fp32) -> its bf16 hi and lo parts
[[clang::always_inline]] inline void {{ p }}_split(aie::vector<float, 64> x, aie::vector<bfloat16, 64> &hi,
                                                   aie::vector<bfloat16, 64> &lo)
{
    hi = aie::accum<accfloat, 64>(x).template to_vector<bfloat16>();
    lo = aie::accum<accfloat, 64>(aie::sub(x, aie::accum<accfloat, 64>(hi).template to_vector<float>())).template to_vector<bfloat16>();
}

// 8 bf16 at p -> 64 lanes, lane 8 j + i = p[i]
[[clang::always_inline]] inline aie::vector<bfloat16, 64> {{ p }}_tile8(const bfloat16 *p)
{
    const v32bfloat16 b = (v32bfloat16)broadcast_elem_128((v16int32)(v32bfloat16)aie::load_v<8>(p).template grow<32>(), 0);
    return aie::concat(aie::vector<bfloat16, 32>(b), aie::vector<bfloat16, 32>(b));
}

// 128 fp32 x -> x / sqrt(sum x^2 + eps) as bf16 hi, lo at h, l (64 B aligned)
__attribute__((noinline)) void {{ p }}_l2(const float *__restrict x, bfloat16 *__restrict h, bfloat16 *__restrict l)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    aie::vector<bfloat16, 64> xh[2], xl[2];
    for (int k = 0; k < 2; k++) {
        {{ p }}_split(aie::load_v<64>(x + 64 * k), xh[k], xl[k]);
        ss = aie::mac(aie::mac(aie::mac(ss, xh[k], xh[k]), xh[k], xl[k]), xh[k], xl[k]);
    }
    const float inv = al::inv_rms<1>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    const aie::vector<float, 64> iv = aie::broadcast<float, 64>(inv);
    for (int k = 0; k < 2; k++) {
        aie::vector<bfloat16, 64> a, b;
        {{ p }}_split(al::mulff<64>(aie::load_v<64>(x + 64 * k), iv), a, b);
        aie::store_v(h + 64 * k, a);
        aie::store_v(l + 64 * k, b);
    }
}
"""

_SRC = JINJA_ENV.from_string(r"""
// The GDN core's workspace (offsets in bytes: GdnShape.workspace())
#define GW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

""" + HELPERS + r"""

// pass 1 of a panel sp (16 i-blocks x [j 8][i 8] fp32): uw[0 .. 8) = S^T k, uw[8 .. 16) = S^T q (k, q: [hi | lo] x 128
// of the token's k head)
__attribute__((noinline)) void {{ p }}_p1(const float *__restrict sp, const bfloat16 *__restrict k,
                                          const bfloat16 *__restrict q, float *__restrict uw)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> au = aie::zeros<accfloat, 64>(), aw = aie::zeros<accfloat, 64>();
    const float *__restrict ps = sp;
    const bfloat16 *__restrict pk = k;
    const bfloat16 *__restrict pq = q;
    for (int b = 0; b < 16; b++) {
        aie::vector<bfloat16, 64> sh, sl;
        {{ p }}_split(aie::load_v<64>(ps), sh, sl);
        ps += 64;
        const aie::vector<bfloat16, 64> kh_ = {{ p }}_tile8(pk), kl = {{ p }}_tile8(pk + {{ D }});
        const aie::vector<bfloat16, 64> qh = {{ p }}_tile8(pq), ql = {{ p }}_tile8(pq + {{ D }});
        pk += 8;
        pq += 8;
        au = aie::mac(aie::mac(aie::mac(au, sh, kh_), sh, kl), sl, kh_);
        aw = aie::mac(aie::mac(aie::mac(aw, sh, qh), sh, ql), sl, qh);
    }
    aie::store_v(uw, aie::concat(al::group8<false>(au.template to_vector<float>()),
                                 al::group8<false>(aw.template to_vector<float>())));
}

// pass 2 of a panel: out = S + d S + k delta^T (k: [k hi | k lo] x 128; dl: the 8 deltas of the panel's j); out may be sp
__attribute__((noinline)) void {{ p }}_p2(const float *sp, float *out, const bfloat16 *__restrict k,
                                          const float *__restrict dl, float d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::vector<bfloat16, 64> eh, el;
    {
        const aie::vector<float, 16> dv = aie::concat(aie::load_v<8>(dl), aie::zeros<float, 8>());
        const aie::vector<bfloat16, 16> dh = aie::accum<accfloat, 16>(dv).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 16> dlo = aie::accum<accfloat, 16>(aie::sub(dv, aie::accum<accfloat, 16>(dh).template to_vector<float>())).template to_vector<bfloat16>();
        eh = al::expand8(dh.template extract<8>(0));
        el = al::expand8(dlo.template extract<8>(0));
    }
    aie::vector<bfloat16, 64> gh, gl;  // d as bf16 hi + lo
    {{ p }}_split(aie::broadcast<float, 64>(d), gh, gl);
    const float *ps = sp;
    const bfloat16 *__restrict pk = k;
    float *po = out;
    for (int b = 0; b < 16; b++) {
        const aie::vector<float, 64> s = aie::load_v<64>(ps);
        ps += 64;
        aie::vector<bfloat16, 64> sh, sl;
        {{ p }}_split(s, sh, sl);
        const aie::vector<bfloat16, 64> kh_ = {{ p }}_tile8(pk), kl = {{ p }}_tile8(pk + {{ D }});
        pk += 8;
        aie::accum<accfloat, 64> acc(s);
        acc = aie::mac(aie::mac(aie::mac(acc, sh, gh), sl, gh), sh, gl);
        acc = aie::mac(aie::mac(aie::mac(acc, kh_, eh), kh_, el), kl, eh);
        aie::store_v(po, acc.template to_vector<float>());
        po += 64;
    }
}

// 4 KB
__attribute__((noinline)) void {{ p }}_cp4k(int32_t *__restrict dst, const int32_t *__restrict src)
{
    al::copy<1024, 16>(dst, src);
}

// k . q of a token's k head (k, q: [hi | lo] x 128)
__attribute__((noinline)) float {{ p }}_kdotq(const bfloat16 *__restrict k, const bfloat16 *__restrict q)
{
    aie::accum<accfloat, 64> a = aie::zeros<accfloat, 64>();
    for (int i = 0; i < {{ D }}; i += 64) {
        const aie::vector<bfloat16, 64> kh_ = aie::load_v<64>(k + i), kl = aie::load_v<64>(k + {{ D }} + i);
        const aie::vector<bfloat16, 64> qh = aie::load_v<64>(q + i), ql = aie::load_v<64>(q + {{ D }} + i);
        a = aie::mac(aie::mac(aie::mac(a, kh_, qh), kh_, ql), kl, qh);
    }
    return aie::reduce_add(a.template to_vector<float>());
}

// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) ([slot][d 16 | beta 16 | k . q 16] fp32; prm:
// dt 16 | A 16)
__attribute__((noinline)) void {{ p }}_ggates(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int S = GW(int32_t, {{ g.st }})[{{ G_NS }}];
    const float *__restrict prm = GW(float, {{ g.prm }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const float *__restrict ab = (const float *)(GW(bfloat16, {{ g.msg }}) + {{ MSGP }} * s + {{ CH + HC * D }});
        aie::vector<float, 16> al_ = aie::zeros<float, 16>(), be = aie::zeros<float, 16>();
{% for h in range(HC) %}
        al_[{{ h }}] = ab[{{ h }}];
        be[{{ h }}] = ab[{{ HC + h }}];
{% endfor %}
        float *__restrict gt = GW(float, {{ g.gt }}) + 48 * s;
        const aie::vector<float, 16> sp = {{ gm.softplus }}(aie::add(al_, aie::load_v<16>(prm)));
        aie::store_v(gt, {{ gm.expm1 }}(al::mulff<16>(sp, aie::load_v<16>(prm + 16))));
        // sigmoid(beta) = e^(-softplus(-beta)) (one copy of the math less)
        const aie::vector<float, 16> sn = {{ gm.softplus }}(aie::sub(aie::zeros<float, 16>(), be));
        aie::store_v(gt + 16, {{ gm.exp2 }}(al::log2e<16>(aie::sub(aie::zeros<float, 16>(), sn))));
    }
}

}  // namespace

extern "C" {

// A new X object: a main core's phase (its input objects follow: skipped), the gdn_in phase (F_MSG: the step), or
// another core's first object (M_SKIP more follow)
void {{ p }}_gbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int flags = m[{{ M_FLAGS }}];
    st[{{ G_NSKIP }}] = 0;
    st[{{ G_NMSG }}] = 0;
    st[{{ G_ON }}] = 0;
    if (!(flags & {{ F_MAIN }}))
        st[{{ G_NSKIP }}] = m[{{ M_SKIP }}];
    else if (!(flags & {{ F_LOADR }}))
        st[{{ G_NSKIP }}] = m[{{ M_NX }}];
    if (!(flags & {{ F_MSG }}))
        return;
    const int S = m[{{ M_NS }}];
    st[{{ G_NS }}] = S;
    st[{{ G_NMSG }}] = S * {{ n_msg }};
    st[{{ G_ON }}] = 1;
    st[{{ G_NCP }}] = 2 + {{ HC // 2 }};
    st[{{ G_NREP }}] = m[{{ M_NREP }}];
    st[{{ G_CSHIFT }}] = m[{{ M_CSHIFT }}];
    st[{{ G_COMMIT }}] = m[{{ M_COMMIT }}];
    st[{{ G_MJ }}] = 0;
    st[{{ G_MS }}] = 0;
    st[{{ G_CB }}] = 0;
    st[{{ G_PAIR }}] = 0;
    st[{{ G_PANEL }}] = 0;
}

// The next message object: object j of slot s
void {{ p }}_gmsg(const bfloat16 *__restrict m, int32_t *__restrict ws)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int j = st[{{ G_MJ }}], s = st[{{ G_MS }}];
    const int last = s + 1 == st[{{ G_NS }}];
    st[{{ G_MS }}] = last ? 0 : s + 1;
    st[{{ G_MJ }}] = last ? j + 1 : j;
    aie::store_v(GW(bfloat16, {{ g.msg }}) + {{ MSGP }} * s + {{ C_LEN }} * j, aie::load_v<{{ C_LEN }}>(m));
}

// Conv object x of the next block: the conv of the S slots (silu; q, k l2-normed as bf16 hi + lo, v fp32), the block's
// new history and pad into its half of the D object d (block 0: the previous step's d, dt, A from its pad -> the gates;
// the step's d into it; blocks 1, 2: ssm_norm)
void {{ p }}_gconv(const bfloat16 *__restrict x, int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int b = st[{{ G_CB }}];
    st[{{ G_CB }}] = b + 1;
    st[{{ G_KO }}] = b == 3;  // (after the k blocks: the K objects; after a pair's v blocks: its panels)
    st[{{ G_PO }}] = b >= 4 && (b & 1);
    const int S = st[{{ G_NS }}];
    const bfloat16 *__restrict w = x + {{ 2 * META }};  // [tap 4][hi 128 | lo 128]
    const bfloat16 *__restrict base = w + {{ 8 * D }} + {{ D }} * st[{{ G_CSHIFT }}];  // the step's 3 base rows
    const bfloat16 *__restrict msg = GW(bfloat16, {{ g.msg }}) + {{ D }} * b;
    bfloat16 *__restrict h = d + {{ D_BYTES // 4 }} * (b & 1);  // [7][128] (+ the pad: 128 bf16)
    const float *__restrict pad = (const float *)(w + {{ 8 * D + HIST * D }});  // (64 fp32)
    float *__restrict pad_out = (float *)(h + {{ HIST * D }});
    al::copy<64, 16>(pad_out, pad);
    if (b == 0) {  // d of the previous step's tokens (the replay); dt, A -> the gates; the step's d
        float *__restrict prm = GW(float, {{ g.prm }});
        for (int r = 0; r < {{ SMAX }}; r++)
            aie::store_v(GW(float, {{ g.dold }}) + 16 * r, aie::concat(aie::load_v<8>(pad + 8 * r), aie::zeros<float, 8>()));
        aie::store_v(prm, aie::concat(aie::load_v<8>(pad + {{ 8 * SMAX }}), aie::zeros<float, 8>()));
        aie::store_v(prm + 16, aie::concat(aie::load_v<8>(pad + {{ 8 * SMAX + 8 }}), aie::zeros<float, 8>()));
        {{ p }}_ggates(ws);
        for (int r = 0; r < {{ SMAX }}; r++)
            aie::store_v(pad_out + 8 * r, aie::load_v<8>(GW(float, {{ g.gt }}) + 48 * r));
    } else if (b < 3) {
        al::copy<64, 16>(GW(float, {{ g.prm }}) + 32 + 64 * (b - 1), pad);
    }
    al::copy<{{ 3 * D }}, 64>(h, base);
    float *__restrict y = GW(float, {{ g.y }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        al::copy<{{ D }}, 64>(h + {{ D }} * (3 + s), msg + {{ MSGP }} * s);
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ D }}; v += 64) {
            aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
            for (int t = 0; t < 4; t++) {
                const int r = s + t;  // (rows 0 .. 2: the base, then the slots' raw rows)
                const bfloat16 *row = r < 3 ? base + {{ D }} * r : msg + {{ MSGP }} * (r - 3);
                const aie::vector<bfloat16, 64> xt = aie::load_v<64>(row + v);
                acc = aie::mac(aie::mac(acc, xt, aie::load_v<64>(w + {{ 2 * D }} * t + v)),
                               xt, aie::load_v<64>(w + {{ 2 * D }} * t + {{ D }} + v));
            }
            aie::store_v(y + v, acc.template to_vector<float>());
        }
        AIE_LOOP_NO_UNROLL
        for (int c = 0; c < {{ D }}; c += 32)  // silu
            aie::store_v(y + c, {{ gm.silu }}(aie::accum<accfloat, 32>(aie::load_v<32>(y + c)).template to_vector<bfloat16>()));
        // kq: [k, q][slot 4][head 2][hi | lo][128]
        bfloat16 *__restrict kq = GW(bfloat16, {{ g.kq }}) + {{ 4 * D }} * s;
        if (b < 4)
            {{ p }}_l2(y, kq + (b < 2 ? {{ 4 * D * SMAX }} + {{ 2 * D }} * b : {{ 2 * D }} * (b - 2)),
                       kq + (b < 2 ? {{ 4 * D * SMAX }} + {{ 2 * D }} * b : {{ 2 * D }} * (b - 2)) + {{ D }});
        else
            al::copy<{{ D }}, 64>(GW(float, {{ g.v }}) + {{ SMAX * D }} * ((b - 4) & 1) + {{ D }} * s, y);
        if (b == 3) {  // k . q of the slot's heads
            float *__restrict gt = GW(float, {{ g.gt }}) + 48 * s;
            const float k0 = {{ p }}_kdotq(kq, kq + {{ 4 * D * SMAX }});
            const float k1 = {{ p }}_kdotq(kq + {{ 2 * D }}, kq + {{ 4 * D * SMAX + 2 * D }});
{% for hh in range(HC) %}
            gt[{{ 32 + hh }}] = {{ 'k1' if hh % 2 else 'k0' }};
{% endfor %}
        }
    }
}

// The K object of the previous step's tokens -> the workspace
void {{ p }}_gkold(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_cp4k(GW(int32_t, {{ g.kold }}), (const int32_t *)(x + {{ 2 * META }}));
}

// The step's K object ([slot 4][k head 2][k hi | k lo])
void {{ p }}_gknew(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k((int32_t *)d, GW(int32_t, {{ g.kq }}));
}

// The delta object of the previous step (the pair's heads) -> the workspace
void {{ p }}_gdl(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_cp4k(GW(int32_t, {{ g.dl }}), (const int32_t *)(x + {{ 2 * META }}));
}

// State panel x of the pair (its next one): the replay, the base out (or the state after the step's tokens: commit),
// the step's tokens (their deltas, o)
void {{ p }}_gpanel(bfloat16 *__restrict x, int32_t *__restrict ws, float *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int pi = st[{{ G_PANEL }}];
    st[{{ G_PANEL }}] = pi + 1;
    const int hl = pi >> 4, pj = pi & 15, h = 2 * st[{{ G_PAIR }}] + hl;  // (v head h reads k head h & 1 = hl)
    const int S = st[{{ G_NS }}], nrep = st[{{ G_NREP }}], commit = st[{{ G_COMMIT }}];
    float *sp = (float *)(x + {{ 2 * META }});
    float *__restrict dl = GW(float, {{ g.dl }}) + {{ SMAX * D }} * hl + 8 * pj;  // [hl][token][128]
    for (int r = 0; r < nrep; r++)
        {{ p }}_p2(sp, sp, GW(bfloat16, {{ g.kold }}) + {{ 4 * D }} * r + {{ 2 * D }} * hl, dl + {{ D }} * r,
                   GW(float, {{ g.dold }})[16 * r + h]);
    if (!commit)
        {{ p }}_cp4k((int32_t *)out, (const int32_t *)sp);
    float *__restrict uw = GW(float, {{ g.uw }});
    for (int s = 0; s < S; s++) {
        const bfloat16 *__restrict kq = GW(bfloat16, {{ g.kq }}) + {{ 4 * D }} * s + {{ 2 * D }} * hl;  // (its k)
        {{ p }}_p1(sp, kq, kq + {{ 4 * D * SMAX }}, uw);
        const float *__restrict gt = GW(float, {{ g.gt }}) + 48 * s;
        const aie::vector<float, 16> dv = aie::broadcast<float, 16>(gt[h]);
        const aie::vector<float, 16> ut = aie::concat(aie::load_v<8>(uw), aie::zeros<float, 8>());
        const aie::vector<float, 16> wt = aie::concat(aie::load_v<8>(uw + 8), aie::zeros<float, 8>());
        const float *__restrict v = GW(float, {{ g.v }}) + {{ SMAX * D }} * hl + {{ D }} * s + 8 * pj;
        const aie::vector<float, 16> u = aie::add(ut, al::mulff<16>(dv, ut));
        const aie::vector<float, 16> de = al::mulff<16>(aie::broadcast<float, 16>(gt[16 + h]),
                                                        aie::sub(aie::concat(aie::load_v<8>(v), aie::zeros<float, 8>()), u));
        const aie::vector<float, 16> oo = aie::add(aie::add(wt, al::mulff<16>(dv, wt)), al::mulff<16>(aie::broadcast<float, 16>(gt[32 + h]), de));
        aie::store_v(GW(float, {{ g.o }}) + {{ SMAX * D }} * hl + {{ D }} * s + 8 * pj,
                     al::mul_hilo<16, {{ rs[0] }}f, {{ rs[1] }}f>(oo).template extract<8>(0));
        aie::store_v(dl + {{ D }} * s, de.template extract<8>(0));
        if (s + 1 < S)
            {{ p }}_p2(sp, sp, kq, dl + {{ D }} * s, gt[h]);
        else if (commit)
            {{ p }}_p2(sp, out, kq, dl + {{ D }} * s, gt[h]);
    }
}

// The pair's deltas of the step's tokens (the next step's replay)
void {{ p }}_gdlout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k((int32_t *)d, GW(int32_t, {{ g.dl }}));
}

// The pair's gated outputs rms(o) ssm_norm silu(z) -> d ([head 2][k block 16][slot 4][8] bf16); the next pair
void {{ p }}_gout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int pair = st[{{ G_PAIR }}], S = st[{{ G_NS }}];
    st[{{ G_PAIR }}] = pair + 1;
    st[{{ G_PANEL }}] = 0;
    const float *__restrict nw = GW(float, {{ g.prm }}) + 32;
    bfloat16 *__restrict g = GW(bfloat16, {{ g.y }});  // [slot 4][128]
    AIE_LOOP_NO_UNROLL
    for (int hl = 0; hl < 2; hl++) {
        al::zero<{{ SMAX * D }}, 64>(g);
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < S; s++) {
            const float *__restrict o = GW(float, {{ g.o }}) + {{ SMAX * D }} * hl + {{ D }} * s;
            const bfloat16 *__restrict z = GW(bfloat16, {{ g.msg }}) + {{ MSGP }} * s + {{ CH }} + {{ D }} * (2 * pair + hl);
            aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
            for (int i = 0; i < {{ D }}; i += 64) {
                aie::vector<bfloat16, 64> a, b;
                {{ p }}_split(aie::load_v<64>(o + i), a, b);
                ss = aie::mac(aie::mac(aie::mac(ss, a, a), a, b), a, b);
            }
            const aie::vector<float, 32> iv = aie::broadcast<float, 32>(al::inv_rms<{{ D }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f));
            AIE_LOOP_NO_UNROLL
            for (int i = 0; i < {{ D }}; i += 32) {
                aie::vector<float, 32> t = al::mulff<32>(al::mulff<32>(aie::load_v<32>(o + i), iv), aie::load_v<32>(nw + i));
                t = al::mulff<32>(t, {{ gm.silu }}(aie::load_v<32>(z + i)));
                aie::store_v(g + {{ D }} * s + i, aie::accum<accfloat, 32>(t).template to_vector<bfloat16>());
            }
        }
        bfloat16 *__restrict dh = d + {{ 16 * 32 }} * hl;
        AIE_LOOP_NO_UNROLL
        for (int kb = 0; kb < 16; kb++)
            aie::store_v(dh + 32 * kb, aie::concat(aie::load_v<8>(g + 8 * kb), aie::load_v<8>(g + {{ D }} + 8 * kb),
                                                   aie::load_v<8>(g + {{ 2 * D }} + 8 * kb), aie::load_v<8>(g + {{ 3 * D }} + 8 * kb)));
    }
}

}  // extern "C"
""")


class GdnShape:
    """The GDN core's sizes for a model: HC v heads and CH conv channels per column, the message, the workspace."""

    def __init__(self, cfg: QwenConfig, *, c_len: int = 64, xpay: int = 4096, meta: int = 64):
        assert cfg.ssm_state == D and cfg.ssm_k_heads == 2 * COLS and cfg.ssm_v_heads % (2 * COLS) == 0
        self.hc = cfg.ssm_v_heads // COLS
        self.ch = (4 + self.hc) * D
        self.nb = self.ch // D  # conv blocks
        self.c_len, self.xpay, self.meta = c_len, xpay, meta
        self.msg_values = self.ch + self.hc * D + 4 * self.hc  # (alpha, beta as fp32: 2 bf16 each)
        self.n_msg = -(-self.msg_values // c_len)
        self.msgp = self.n_msg * c_len  # bf16 elements of a slot's message
        self.eps = cfg.rms_norm_eps
        assert 8 * D * 2 + HIST * D * 2 + 4 * SMAX * 16 <= xpay  # a conv object
        assert SMAX * 2 * 2 * D * 2 <= xpay and 2 * SMAX * D * 4 <= xpay  # the K and delta objects
        assert self.hc <= 8

    @property
    def n_panels(self) -> int:
        """State panels of a column."""
        return 16 * self.hc

    @property
    def n_objects(self) -> int:
        """X objects of a step (after the gdn_in phase's): conv q / k 4, K, per pair 2 conv + delta + 32 panels."""
        return 5 + self.hc // 2 * 35

    def workspace(self) -> dict[str, int]:
        sizes = [("st", 4 * G_WORDS), ("prm", 4 * 160), ("msg", 2 * SMAX * self.msgp), ("kq", 2 * SMAX * 8 * D),
                 ("gt", 4 * SMAX * 48), ("v", 4 * 2 * SMAX * D), ("o", 4 * 2 * SMAX * D), ("dl", 4 * 2 * SMAX * D),
                 ("kold", 2 * SMAX * 4 * D), ("dold", 4 * SMAX * 16), ("y", max(4 * D, 2 * SMAX * D)), ("uw", 4 * 16)]
        return regions(sizes)

    def render(self, p: str, consts: dict[str, int]) -> str:
        return _SRC.render(p=p, g=Attr(self.workspace()), gm=MATH, sigmoid=False, CH=self.ch, HC=self.hc, D=D, n_msg=self.n_msg,
                           MSGP=self.msgp, C_LEN=self.c_len, META=self.meta, XPAY=self.xpay, D_BYTES=D_BYTES,
                           PANEL=PANEL, SMAX=SMAX, HIST=HIST, eps=repr(self.eps),
                           rs=bf16_hilo(1 / math.sqrt(D)), **consts,
                           **{k: v for k, v in globals().items() if k.startswith("G_")})


# ----------------------------------------------------------------------
# Host side: object layouts
# ----------------------------------------------------------------------


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def hilo_bits(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fp32 -> bf16 bit patterns of hi and lo (x - hi)."""
    x = np.asarray(x, np.float32)
    hi = bf16_bits(x)
    hv = (hi.astype(np.uint32) << 16).view(np.float32)
    return hi, bf16_bits(x - hv)


def state_panels(m: np.ndarray) -> np.ndarray:
    """llama.cpp's state of the column's v heads (HC, j, i) fp32 ([h][j][i] = S[i][j]) -> the cores' panel order
    (HC, 16 j-panels, 16 i-blocks, 8 j, 8 i) as (HC * 16, 1024) fp32 (one row per panel)."""
    hc = m.shape[0]
    s = np.asarray(m, np.float32).reshape(hc, 16, 8, 16, 8)  # [h][jp][jj][ib][ii]
    return np.ascontiguousarray(s.transpose(0, 1, 3, 2, 4)).reshape(hc * 16, 1024)


def state_from_panels(p: np.ndarray, hc: int) -> np.ndarray:
    """Inverse of ``state_panels``."""
    s = np.asarray(p, np.float32).reshape(hc, 16, 16, 8, 8).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(s).reshape(hc, D, D)


def conv_payloads(sh: GdnShape, w: np.ndarray, dt: np.ndarray, ssm_a: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """The column's conv objects' payloads (NB, XPAY) uint8 at load: the weights (CH, 4) fp32 as [tap 4][hi 128 | lo 128]
    bf16, a zero history, the pads (block 0: d zero, dt, A of the column's heads; blocks 1, 2: ssm_norm)."""
    hi, lo = hilo_bits(np.asarray(w, np.float32).T)  # (4, CH)
    out = np.zeros((sh.nb, sh.xpay), np.uint8)
    pad0 = 2 * (8 * D + HIST * D)
    for b in range(sh.nb):
        wb = np.stack([hi[:, b * D:(b + 1) * D], lo[:, b * D:(b + 1) * D]], 1)  # (4, 2, 128)
        out[b, :wb.nbytes] = np.ascontiguousarray(wb).view(np.uint8).reshape(-1)
    pad = np.zeros(64, np.float32)
    pad[8 * SMAX:8 * SMAX + sh.hc], pad[8 * SMAX + 8:8 * SMAX + 8 + sh.hc] = dt, ssm_a
    out[0, pad0:pad0 + 256] = pad.view(np.uint8)
    nb = np.asarray(norm, np.float32)
    out[1, pad0:pad0 + 256], out[2, pad0:pad0 + 256] = nb[:64].view(np.uint8), nb[64:].view(np.uint8)
    return out
