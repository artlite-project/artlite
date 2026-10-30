"""The GDN core of the multi-slot decode overlay (``qwen35/decode.py``): one Gated DeltaNet step of S <= 4 token
slots (8 slots: below) per GDN layer and column, with a lazy commit of the recurrent state.

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

8 slots (``GdnShape(slots=8)``, designs of steps of up to 8 tokens): the message of 8 slots (34 KB) and the per-pair
arrays twice the size do not fit the core's L1 (which also holds the helper's scratch: ``decode.design``'s 8-slot
partition), so the core runs per v head, its v heads grouped by k head (k head kh's v heads h = kh, kh + 2, ..: one k
head's q, k and K objects in the workspace at a time), and gdn_in's rows reach it through DDR: gdn_in drains them (C)
into the step's objects (3 drains per column, its objects (half, slot) as [half][slot 8][64]: the q / k blocks', the v
blocks' then z, ab), then a sync. A column's objects of a layer are payload slots of XPAY bytes (each object's metadata
the 256 bytes before it): the header H (F_GDN, M_SKIP; the slots' ab, dt, A, ssm_norm, the previous step's d per token
from the last gated outputs' D object), per k head [q A, q B, K, k A, k B], per v head [A, B, DL, Z]. A conv block's
two objects: A = [weights (2 KB) | its history's rows 0 .. 7], B = [the history's rows 8 .. 10 | the step's S rows]:
the history (3 + 8 rows, 2816 bytes, after the weights) holds the previous step's [base 3 | S rows], from which the core
takes the step's base rows at cshift, so the host patches nothing per step; the new history [base 3 | the step's S rows]
goes out over it (D: from A's second half on). The step's words (S, M_NREP, M_CSHIFT, M_COMMIT) come from the gdn_in
phase's metadata object (``F_GIN``). Per step: the header (``gbegin``: the gates), per k head its q block, the previous
step's K object, its k block (then the q history, the step's K object and the k history out: 6 KB apart, one drain),
then per v head its block, deltas and z, its 16 state panels, and its history, deltas and gated outputs out ([k block 16][slot 8][8] bf16: gdn_out's x of the head;
the 2 KB after them the step's d per token, into the header from the last head, else the sink). A column's DMA tasks
per layer: 14 fills (its X objects in runs: header + k head 0's group, k head 1's, per v head its group and its
panels) and 21 drains.

Split (``GdnShape(split=True)``, 8 slots): the helper core (idle in a GDN step) runs half of a v head's state panels.
The X objects bring them in pairs (i, 8 + i); the core copies panel i into the helper's x panels (its north
neighbour's memory) with the D object it holds for the helper's out, and releases its lock ``L_HRDY``; the helper runs
the panel on the core's workspace while the core runs panel 8 + i, its out in the helper's memory (``H_OUT``); after the
helper's ``L_HDONE`` the core's out follows into the next D object (the core holds one D object at a time: an acquire
of two gives the fifo a third buffer). The header's ``M_HPAN`` (the helper's panels per step) reaches the helper as a
K_HELP command from the main core's begin (``decode.design``'s ``help_flag``). The panels are the same arithmetic on
either core: the outputs are the unsplit core's bit for bit.
"""

import math

import numpy as np

from artlite.model_config import QwenConfig
from artlite.operators import kmath
from artlite.operators.base import JINJA_ENV
from artlite.operators.design import COLS
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.util import Attr, bf16_hilo, regions

D = 128  # ssm_state: k and v dims of a head
PANEL = 4096  # bytes of a state panel (128 i x 8 j fp32)
D_BYTES = 4096  # bytes of a D object
SMAX = 4  # token slots (of a 4-slot design's core)
HIST = 3 + SMAX  # history rows of a conv object
F_GDN = 1 << 8  # M_FLAGS of the first GDN object of a step (the other cores skip it and M_SKIP objects after it)
F_GIN = 1 << 9  # (8 slots) M_FLAGS of the gdn_in phase's metadata object: the GDN core takes the step's words from it

# state words of the GDN core's workspace (the first ones are trip counts of its program)
(G_NSKIP, G_NMSG, G_ON, G_NCP, G_KO, G_PO, G_NS, G_NREP, G_CSHIFT, G_COMMIT, G_MJ, G_MS, G_CB, G_PAIR,
 G_PANEL) = range(15)
G_KH, G_VI, G_VH = G_PAIR, G_MJ, G_MS  # (8 slots) the k head, its v heads done, the v head in the column
G_HPJ = G_KO  # (8 slots, split) the helper's next panel of the head
# (8 slots, split) the GDN tile's locks of the panels the helper runs (``GdnShape(split=)``): handed over, done; the
# header's metadata word of the helper's panels per step
L_HRDY, L_HDONE = 12, 13
M_HPAN = 35
H_OUT = 1088  # (split) words into the helper's slot of the GDN core's panel out (4 KB after it)
G_WORDS = 16
# (8 slots) a column's objects of a GDN layer: payload slots of XPAY bytes (each object's metadata the 256 bytes before
# its payload): the header (H_SLOT), per k head kh its group [q A, q B, K, k A, k B] (from KH_SLOT + KH_SPAN kh), per v
# head h its group [A, B, DL, Z] (from V_SLOT + V_SPAN h); the header's payload: the slots' ab at H_AB (a gdn_in object
# each: 128 bytes), dt 16 | A 16 | ssm_norm 128 fp32 at H_PRM, the step's d per token ([token 8][8] fp32: the last
# gated outputs' D object's second half) at H_DOLD
H_SLOT, KH_SLOT, KH_SPAN, V_SLOT, V_SPAN = 0, 1, 5, 11, 4
H_AB, H_PRM, H_DOLD = 0, 1024, 2048


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

// 128 fp32 x -> x / sqrt(sum x^2 + eps) as bf16 hi, lo at h, l (64 B aligned)
__attribute__((noinline)) void {{ p }}_l2(const float *__restrict x, bfloat16 *__restrict h, bfloat16 *__restrict l)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    aie::vector<bfloat16, 64> xh[2], xl[2];
    for (int k = 0; k < 2; k++) {
        al::split2<64>(aie::load_v<64>(x + 64 * k), xh[k], xl[k]);
        ss = aie::mac(aie::mac(aie::mac(ss, xh[k], xh[k]), xh[k], xl[k]), xh[k], xl[k]);
    }
    const float inv = al::inv_rms<1>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    const aie::vector<float, 64> iv = aie::broadcast<float, 64>(inv);
    for (int k = 0; k < 2; k++) {
        aie::vector<bfloat16, 64> a, b;
        al::split2<64>(al::mulff<64>(aie::load_v<64>(x + 64 * k), iv), a, b);
        aie::store_v(h + 64 * k, a);
        aie::store_v(l + 64 * k, b);
    }
}
"""

# The state panels' passes and the small copies, shared by the 4- and 8-slot cores
PANELS = r"""// pass 1 of a panel sp (16 i-blocks x [j 8][i 8] fp32): uw[0 .. 8) = S^T k, uw[8 .. 16) = S^T q (k, q: [hi | lo] x 128
// of the token's k head)
__attribute__((noinline)) void {{ p }}_p1(const float *__restrict sp, const bfloat16 *__restrict k,
                                          const bfloat16 *__restrict q, float *__restrict uw)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> au = aie::zeros<accfloat, 64>(), aw = aie::zeros<accfloat, 64>();
    const float __aie_dm_resource_a *__restrict ps = (const float __aie_dm_resource_a *)sp;
    const bfloat16 __aie_dm_resource_b *__restrict pk = (const bfloat16 __aie_dm_resource_b *)k;
    const bfloat16 __aie_dm_resource_b *__restrict pq = (const bfloat16 __aie_dm_resource_b *)q;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int b = 0; b < 16; b++) {
        aie::vector<bfloat16, 64> sh, sl;
        al::split2<64>(aie::load_v<64>(ps), sh, sl);
        ps += 64;
        const aie::vector<bfloat16, 64> kh_ = aie::load_v<8>(pk).template grow_replicate<64>(), kl = aie::load_v<8>(pk + {{ D }}).template grow_replicate<64>();
        const aie::vector<bfloat16, 64> qh = aie::load_v<8>(pq).template grow_replicate<64>(), ql = aie::load_v<8>(pq + {{ D }}).template grow_replicate<64>();
        pk += 8;
        pq += 8;
        au = aie::mac(aie::mac(aie::mac(au, sh, kh_), sh, kl), sl, kh_);
        aw = aie::mac(aie::mac(aie::mac(aw, sh, qh), sh, ql), sl, qh);
    }
    aie::store_v(uw, aie::concat(al::group8<false>(au.template to_vector<float>()),
                                 al::group8<false>(aw.template to_vector<float>())));
}

// pass 2 of a panel: out = (S + d S) + k delta^T (k: [k hi | k lo] x 128; dl: the 8 deltas of the panel's j; the two
// terms in accumulators of their own: chains of three MACs); out may be sp (each block is read before it is written)
__attribute__((noinline)) void {{ p }}_p2(const float *__restrict sp, float *__restrict out, const bfloat16 *__restrict k,
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
    al::split2<64>(aie::broadcast<float, 64>(d), gh, gl);
    const float __aie_dm_resource_a *__restrict ps = (const float __aie_dm_resource_a *)sp;
    const bfloat16 __aie_dm_resource_b *__restrict pk = (const bfloat16 __aie_dm_resource_b *)k;
    float *__restrict po = out;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int b = 0; b < 16; b++) {
        const aie::vector<float, 64> s = aie::load_v<64>(ps);
        ps += 64;
        aie::vector<bfloat16, 64> sh, sl;
        al::split2<64>(s, sh, sl);
        const aie::vector<bfloat16, 64> kh_ = aie::load_v<8>(pk).template grow_replicate<64>(), kl = aie::load_v<8>(pk + {{ D }}).template grow_replicate<64>();
        pk += 8;
        const aie::accum<accfloat, 64> a1 = aie::mac(aie::mac(aie::mul(kh_, eh), kh_, el), kl, eh);
        const aie::accum<accfloat, 64> a0 = aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 64>(s), sh, gh), sl, gh), sh, gl);
        aie::store_v(po, aie::add(a0, a1).template to_vector<float>());
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

"""

_SRC = JINJA_ENV.from_string(r"""
// The GDN core's workspace (offsets in bytes: GdnShape.workspace())
#define GW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

""" + HELPERS + r"""

""" + PANELS + r"""// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) ([slot][d 16 | beta 16 | k . q 16] fp32; prm:
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
        const aie::vector<float, 16> sw = aie::load_v<16>(uw);  // [S^T k | S^T q] -> (1 + d) of both
        const aie::vector<float, 16> uw2 = aie::add(sw, al::mulff<16>(aie::broadcast<float, 16>(gt[h]), sw));
        const float *__restrict v = GW(float, {{ g.v }}) + {{ SMAX * D }} * hl + {{ D }} * s + 8 * pj;
        const aie::vector<float, 16> de = al::mulff<16>(aie::broadcast<float, 16>(gt[16 + h]),
                                                        aie::concat(aie::sub(aie::load_v<8>(v), uw2.template extract<8>(0)), aie::zeros<float, 8>()));
        const aie::vector<float, 16> oo = aie::add(aie::concat(uw2.template extract<8>(1), aie::zeros<float, 8>()),
                                                   al::mulff<16>(aie::broadcast<float, 16>(gt[32 + h]), de));
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
                al::split2<64>(aie::load_v<64>(o + i), a, b);
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


# The core of steps of up to 8 slots (``GdnShape(slots=8)``, module docstring "8 slots")
_SRC8 = JINJA_ENV.from_string(r"""
// The GDN core's workspace (offsets in bytes: GdnShape.workspace())
#define GW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

""" + HELPERS + r"""

""" + PANELS + r"""// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) ([slot][d 16 | beta 16 | k . q 16] fp32; prm:
// dt 16 | A 16) of the header's ab (per slot its gdn_in object: alpha HC | beta HC fp32)
__attribute__((noinline)) void {{ p }}_ggates(const float *__restrict hab, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int S = GW(int32_t, {{ g.st }})[{{ G_NS }}];
    const float *__restrict prm = GW(float, {{ g.prm }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const float *__restrict ab = hab + {{ C_LEN // 2 }} * s;
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

// The header's payload: dt, A, ssm_norm, the previous step's d per token (the replay's), the slots' ab -> the gates
__attribute__((noinline)) void {{ p }}_ghead(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const float *__restrict pay = (const float *)(x + {{ 2 * META }});
    al::copy<160, 16>(GW(float, {{ g.prm }}), pay + {{ H_PRM // 4 }});
    for (int r = 0; r < {{ SMAX }}; r++)
        aie::store_v(GW(float, {{ g.dold }}) + 16 * r, aie::concat(aie::load_v<8>(pay + {{ H_DOLD // 4 }} + 8 * r), aie::zeros<float, 8>()));
    {{ p }}_ggates(pay + {{ H_AB // 4 }}, ws);
    st[{{ G_CB }}] = 0;
    st[{{ G_PANEL }}] = {{ 8 if split else 0 }};
    st[{{ G_HPJ }}] = 0;
    st[{{ G_KH }}] = 0;
    st[{{ G_VI }}] = 0;
    st[{{ G_VH }}] = 0;
}

}  // namespace

extern "C" {

// A new X object: the gdn_in phase's metadata object (F_GIN: the step's S, replay, shift and commit words; its input
// objects follow: skipped), another main core phase's (skipped too), the header (F_GDN: ``ghead``, the step's objects
// follow), or another core's first object (M_SKIP more follow)
void {{ p }}_gbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int flags = m[{{ M_FLAGS }}];
    st[{{ G_NSKIP }}] = 0;
    st[{{ G_ON }}] = 0;
    if (flags & {{ F_GIN }}) {
        st[{{ G_NS }}] = m[{{ M_NS }}];
        st[{{ G_NREP }}] = m[{{ M_NREP }}];
        st[{{ G_CSHIFT }}] = m[{{ M_CSHIFT }}];
        st[{{ G_COMMIT }}] = m[{{ M_COMMIT }}];
    }
    if (flags & {{ F_GDN }}) {
        {{ p }}_ghead(x, ws);
        st[{{ G_ON }}] = 1;
    } else if (!(flags & {{ F_MAIN }})) {
        st[{{ G_NSKIP }}] = m[{{ M_SKIP }}];
    } else if (!(flags & {{ F_LOADR }})) {  // (its input objects: of 8 / M_SP strips)
        const int sp = m[{{ M_SP }}], lper = sp == 1 ? 3 : (sp == 2 ? 2 : (sp == 4 ? 1 : 0));
        st[{{ G_NSKIP }}] = (m[{{ M_NX }}] + (1 << lper) - 1) >> lper;
    }
}

// A conv block's first object ([weights: tap 4 x (hi 128 | lo 128) bf16 | its history's rows 0 .. 7]): the weights ->
// cw, the step's base rows (the history's rows cshift .. cshift + 2) among them -> rows 0 .. 2 of the block's new
// history hb
void {{ p }}_gw(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    bfloat16 *__restrict h = GW(bfloat16, {{ g.hb }});
    const bfloat16 *__restrict w = x + {{ 2 * META }};
    al::copy<{{ 8 * D }}, 64>(GW(bfloat16, {{ g.cw }}), w);
    const int c0 = st[{{ G_CSHIFT }}];
    for (int i = 0; i < 3; i++)
        if (c0 + i < 8)
            al::copy<{{ D }}, 64>(h + {{ D }} * i, w + {{ 8 * D }} + {{ D }} * (c0 + i));
}

// A conv block's second object ([its history's rows 8 .. 10 | the step's S rows as [half 2][slot 8][64]]): the rest of
// the base rows and the step's rows into hb ([base 3 | S rows]: the next step's history; its rows past 3 + S stale),
// the conv of the S slots on hb's rows (silu; q (block 0 of a k head), k (1) l2-normed as bf16 hi + lo into kq, then
// k . q per slot; a v head's fp32 into v)
void {{ p }}_gconv(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    bfloat16 *__restrict h = GW(bfloat16, {{ g.hb }});
    const int b = st[{{ G_CB }}];
    st[{{ G_CB }}] = b + 1;
    const int S = st[{{ G_NS }}], c0 = st[{{ G_CSHIFT }}];
    const bfloat16 *__restrict r8 = x + {{ 2 * META }};
    for (int i = 0; i < 3; i++)
        if (c0 + i >= 8)
            al::copy<{{ D }}, 64>(h + {{ D }} * i, r8 + {{ D }} * (c0 + i - 8));
    const bfloat16 *__restrict w = GW(bfloat16, {{ g.cw }});
    float *__restrict y = GW(float, {{ g.y }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        aie::store_v(h + {{ D }} * (3 + s), aie::load_v<64>(r8 + {{ 3 * D }} + 64 * s));
        aie::store_v(h + {{ D }} * (3 + s) + 64, aie::load_v<64>(r8 + {{ 3 * D + 64 * SMAX }} + 64 * s));
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ D }}; v += 64) {
            aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
            for (int t = 0; t < 4; t++) {
                const aie::vector<bfloat16, 64> xt = aie::load_v<64>(h + {{ D }} * (s + t) + v);
                acc = aie::mac(aie::mac(acc, xt, aie::load_v<64>(w + {{ 2 * D }} * t + v)),
                               xt, aie::load_v<64>(w + {{ 2 * D }} * t + {{ D }} + v));
            }
            aie::store_v(y + v, acc.template to_vector<float>());
        }
        AIE_LOOP_NO_UNROLL
        for (int c = 0; c < {{ D }}; c += 32)  // silu
            aie::store_v(y + c, {{ gm.silu }}(aie::accum<accfloat, 32>(aie::load_v<32>(y + c)).template to_vector<bfloat16>()));
        // kq: [k, q][slot 8][hi | lo][128]
        bfloat16 *__restrict kq = GW(bfloat16, {{ g.kq }}) + {{ 2 * D }} * s;
        if (b < 2)
            {{ p }}_l2(y, kq + (b ? 0 : {{ 2 * D * SMAX }}), kq + (b ? 0 : {{ 2 * D * SMAX }}) + {{ D }});
        else
            al::copy<{{ D }}, 64>(GW(float, {{ g.v }}) + {{ D }} * s, y);
        if (b == 1)  // k . q of the slot's heads
            GW(float, {{ g.gt }})[48 * s + 32] = {{ p }}_kdotq(kq, kq + {{ 2 * D * SMAX }});
    }
}

// A conv block's new history (hb: [base 3 | S rows], 11 rows) -> d
void {{ p }}_ghist(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    al::copy<{{ 11 * D }}, 64>(d, GW(bfloat16, {{ g.hb }}));
}

// The K object of the previous step's tokens of the k head ([token 8][k hi | k lo]) -> the workspace
void {{ p }}_gkold(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_cp4k(GW(int32_t, {{ g.kold }}), (const int32_t *)(x + {{ 2 * META }}));
}

// The step's K object of the k head ([slot 8][k hi | k lo])
void {{ p }}_gknew(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k((int32_t *)d, GW(int32_t, {{ g.kq }}));
}

// A v head's delta object of the previous step ([token 8][128] fp32) -> the workspace
void {{ p }}_gdl(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_cp4k(GW(int32_t, {{ g.dl }}), (const int32_t *)(x + {{ 2 * META }}));
}

// A v head's z object ([half 2][slot 8][64] bf16) -> the workspace ([slot 8][128])
void {{ p }}_gz(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const bfloat16 *__restrict z = x + {{ 2 * META }};
    bfloat16 *__restrict zb = GW(bfloat16, {{ g.zb }});
    for (int s = 0; s < {{ SMAX }}; s++) {
        aie::store_v(zb + {{ D }} * s, aie::load_v<64>(z + 64 * s));
        aie::store_v(zb + {{ D }} * s + 64, aie::load_v<64>(z + {{ 64 * SMAX }} + 64 * s));
    }
}

}  // extern "C"

namespace {

// State panel pj of the v head at sp (in place): the replay, the base out (or the state after the step's tokens:
// commit), the step's tokens (their deltas, o); ws: the GDN core's workspace, uw: the running core's scratch
__attribute__((noinline)) void {{ p }}_gpan(float *sp, float *__restrict out, int32_t *__restrict ws, int pj,
                                            float *__restrict uw)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int h = st[{{ G_VH }}];  // (the v head in the column: its gates)
    const int S = st[{{ G_NS }}], nrep = st[{{ G_NREP }}], commit = st[{{ G_COMMIT }}];
    float *__restrict dl = GW(float, {{ g.dl }}) + 8 * pj;  // [token][128]
    for (int r = 0; r < nrep; r++)
        {{ p }}_p2(sp, sp, GW(bfloat16, {{ g.kold }}) + {{ 2 * D }} * r, dl + {{ D }} * r, GW(float, {{ g.dold }})[16 * r + h]);
    if (!commit)
        {{ p }}_cp4k((int32_t *)out, (const int32_t *)sp);
    for (int s = 0; s < S; s++) {
        const bfloat16 *__restrict kq = GW(bfloat16, {{ g.kq }}) + {{ 2 * D }} * s;  // (its k)
        {{ p }}_p1(sp, kq, kq + {{ 2 * D * SMAX }}, uw);
        const float *__restrict gt = GW(float, {{ g.gt }}) + 48 * s;
        const aie::vector<float, 16> sw = aie::load_v<16>(uw);  // [S^T k | S^T q] -> (1 + d) of both
        const aie::vector<float, 16> uw2 = aie::add(sw, al::mulff<16>(aie::broadcast<float, 16>(gt[h]), sw));
        const float *__restrict v = GW(float, {{ g.v }}) + {{ D }} * s + 8 * pj;
        const aie::vector<float, 16> de = al::mulff<16>(aie::broadcast<float, 16>(gt[16 + h]),
                                                        aie::concat(aie::sub(aie::load_v<8>(v), uw2.template extract<8>(0)), aie::zeros<float, 8>()));
        const aie::vector<float, 16> oo = aie::add(aie::concat(uw2.template extract<8>(1), aie::zeros<float, 8>()),
                                                   al::mulff<16>(aie::broadcast<float, 16>(gt[32]), de));
        aie::store_v(GW(float, {{ g.o }}) + {{ D }} * s + 8 * pj,
                     al::mul_hilo<16, {{ rs[0] }}f, {{ rs[1] }}f>(oo).template extract<8>(0));
        aie::store_v(dl + {{ D }} * s, de.template extract<8>(0));
        if (s + 1 < S)
            {{ p }}_p2(sp, sp, kq, dl + {{ D }} * s, gt[h]);
        else if (commit)
            {{ p }}_p2(sp, out, kq, dl + {{ D }} * s, gt[h]);
    }
}

}  // namespace

extern "C" {

// State panel x of the v head (its next one; split: of its second half): ``gpan``
void {{ p }}_gpanel(bfloat16 *__restrict x, int32_t *__restrict ws, float *__restrict out)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int pj = st[{{ G_PANEL }}];
    st[{{ G_PANEL }}] = pj + 1;
    {{ p }}_gpan((float *)(x + {{ 2 * META }}), out, ws, pj, GW(float, {{ g.uw }}));
}
{% if split %}

// (split) Panel x of the v head's first half, for the helper: copied into the helper's slot hs ([panel 1024 words |
// its D object d, its panel | the helper's scratch | the GDN core's panel out: H_OUT]: hs the helper's x panels, idle
// in a GDN step), then the helper is told
void {{ p }}_ghand(const bfloat16 *__restrict x, int32_t *__restrict ws, bfloat16 *__restrict d, uint8_t *__restrict hs8)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    int32_t *__restrict hs = (int32_t *)hs8;
    const int pj = st[{{ G_HPJ }}];
    st[{{ G_HPJ }}] = pj + 1;
    {{ p }}_cp4k(hs, (const int32_t *)(x + {{ 2 * META }}));
    hs[1024] = (int32_t)(uintptr_t)d;
    hs[1025] = pj;
    release(hs, {{ 48 + L_HRDY }}, 1);
}

// (split) Panel x of the v head's second half, its out into hs (one D object at a time: the helper's is held)
void {{ p }}_gown(bfloat16 *__restrict x, int32_t *__restrict ws, uint8_t *__restrict hs8)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int pj = st[{{ G_PANEL }}];
    st[{{ G_PANEL }}] = pj + 1;
    {{ p }}_gpan((float *)(x + {{ 2 * META }}), (float *)hs8 + {{ H_OUT }}, ws, pj, GW(float, {{ g.uw }}));
}

// (split) The helper's panel is out
void {{ p }}_gwait(int32_t *__restrict ws)
{
    acquire_greater_equal(ws, {{ 48 + L_HDONE }}, 1);
}

// (split) The GDN core's panel out of hs -> d
void {{ p }}_gput(uint8_t *__restrict hs8, bfloat16 *__restrict d)
{
    {{ p }}_cp4k((int32_t *)d, (const int32_t *)hs8 + {{ H_OUT }});
}

// (split, the helper) The panel the GDN core handed over in hs (the D object's address as the GDN core sees it: its
// memory is the helper's south neighbour's), on the GDN core's workspace gws
void {{ p }}_hgdn(int32_t *__restrict gws, uint8_t *__restrict hs8)
{
    int32_t *__restrict hs = (int32_t *)hs8;
    acquire_greater_equal(hs, {{ L_HRDY }}, 1);  // (the GDN tile's locks: the helper's 0 .. 15)
    {{ p }}_gpan((float *)hs, (float *)(uintptr_t)(hs[1024] - 0x30000), gws, hs[1025], (float *)(hs + 1040));
    release(hs, {{ L_HDONE }}, 1);
}
{% endif %}

// The v head's deltas of the step's tokens (the next step's replay)
void {{ p }}_gdlout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k((int32_t *)d, GW(int32_t, {{ g.dl }}));
}

// The v head's gated outputs rms(o) ssm_norm silu(z) -> d ([k block 16][slot 8][8] bf16), then the step's d per token
// ([token 8][8] fp32: the next step's replay, from the last head's into the header); the next v head (of the next k
// head after the last of this one's)
void {{ p }}_gout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int S = st[{{ G_NS }}];
    const float *__restrict nw = GW(float, {{ g.prm }}) + 32;
    bfloat16 *__restrict g = GW(bfloat16, {{ g.y }});  // [slot 8][128]
    al::zero<{{ SMAX * D }}, 64>(g);
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const float *__restrict o = GW(float, {{ g.o }}) + {{ D }} * s;
        const bfloat16 *__restrict z = GW(bfloat16, {{ g.zb }}) + {{ D }} * s;
        aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
        for (int i = 0; i < {{ D }}; i += 64) {
            aie::vector<bfloat16, 64> a, b;
            al::split2<64>(aie::load_v<64>(o + i), a, b);
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
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < 16; kb++)
        aie::store_v(d + 64 * kb, aie::concat(aie::concat(aie::load_v<8>(g + 8 * kb), aie::load_v<8>(g + {{ D }} + 8 * kb),
                                                          aie::load_v<8>(g + {{ 2 * D }} + 8 * kb), aie::load_v<8>(g + {{ 3 * D }} + 8 * kb)),
                                              aie::concat(aie::load_v<8>(g + {{ 4 * D }} + 8 * kb), aie::load_v<8>(g + {{ 5 * D }} + 8 * kb),
                                                          aie::load_v<8>(g + {{ 6 * D }} + 8 * kb), aie::load_v<8>(g + {{ 7 * D }} + 8 * kb))));
    float *__restrict dd = (float *)(d + {{ 16 * 64 }});
    for (int r = 0; r < {{ SMAX }}; r++)
        aie::store_v(dd + 8 * r, aie::load_v<8>(GW(float, {{ g.gt }}) + 48 * r));
    st[{{ G_PANEL }}] = {{ 8 if split else 0 }};
    st[{{ G_HPJ }}] = 0;
    const int vi = st[{{ G_VI }}] + 1, last = vi == {{ HC // 2 }};
    const int kh = st[{{ G_KH }}] + last;
    st[{{ G_VI }}] = last ? 0 : vi;
    st[{{ G_KH }}] = kh;
    st[{{ G_VH }}] = 2 * (last ? 0 : vi) + kh;
    if (last)
        st[{{ G_CB }}] = 0;
}

}  // extern "C"
""")


class GdnShape:
    """The GDN core's sizes for a model: HC v heads and CH conv channels per column, the message, the workspace; the
    core of steps of up to ``slots`` (4: the message on M; 8: module docstring "8 slots")."""

    def __init__(self, cfg: QwenConfig, *, c_len: int = 64, xpay: int = 4096, meta: int = 64, slots: int = SMAX,
                 split: bool = False):
        assert cfg.ssm_state == D and cfg.ssm_k_heads == 2 * COLS and cfg.ssm_v_heads % (2 * COLS) == 0
        assert slots in (SMAX, 8) and (not split or slots == 8)
        self.slots, self.split = slots, split
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
        assert self.hc <= 8 and self.hc % 2 == 0
        # (8 slots: the gdn_in objects of a slot: qkv rows, z, ab; the ab object at a C object's start)
        assert slots == SMAX or (self.ch + D * self.hc) % c_len == 0 and 4 * self.hc <= c_len

    @property
    def n_panels(self) -> int:
        """State panels of a column."""
        return 16 * self.hc

    @property
    def n_objects(self) -> int:
        """X objects of a step (after the gdn_in phase's): conv q / k 4, K, per pair 2 conv + delta + 32 panels; 8 slots:
        the header, per k head 5, per v head [A, B, DL, Z] + 16 panels."""
        return 5 + self.hc // 2 * 35 if self.slots == SMAX else 1 + 2 * KH_SPAN + self.hc * (V_SPAN + 16)

    @property
    def n_slots(self) -> int:
        """(8 slots) Payload slots of a column's objects of a layer (module docstring "8 slots")."""
        return V_SLOT + V_SPAN * self.hc

    def kh_slot(self, kh: int) -> int:
        """(8 slots) The first payload slot of k head kh's group [q A, q B, K, k A, k B]."""
        return KH_SLOT + KH_SPAN * kh

    def v_slot(self, h: int) -> int:
        """(8 slots) The first payload slot of v head h's group [A, B, DL, Z]."""
        return V_SLOT + V_SPAN * h

    def block_slot(self, b: int) -> int:
        """(8 slots) The A object's payload slot of conv block b (q heads 0, 1, k heads 0, 1, then the v heads)."""
        return self.kh_slot(b % 2) + 3 * (b // 2) if b < 4 else self.v_slot(b - 4)

    def workspace(self) -> dict[str, int]:
        if self.slots == 8:
            S = self.slots
            sizes = [("st", 4 * G_WORDS), ("prm", 4 * 160), ("gt", 4 * S * 48), ("kq", 2 * 2 * S * 2 * D),
                     ("kold", 2 * S * 2 * D), ("v", 4 * S * D), ("o", 4 * S * D), ("dl", 4 * S * D), ("dold", 4 * S * 16),
                     ("cw", 2 * 8 * D), ("zb", 2 * S * D), ("y", 2 * S * D), ("uw", 4 * 16), ("hb", 2 * 11 * D)]
            return regions(sizes)
        sizes = [("st", 4 * G_WORDS), ("prm", 4 * 160), ("msg", 2 * SMAX * self.msgp), ("kq", 2 * SMAX * 8 * D),
                 ("gt", 4 * SMAX * 48), ("v", 4 * 2 * SMAX * D), ("o", 4 * 2 * SMAX * D), ("dl", 4 * 2 * SMAX * D),
                 ("kold", 2 * SMAX * 4 * D), ("dold", 4 * SMAX * 16), ("y", max(4 * D, 2 * SMAX * D)), ("uw", 4 * 16)]
        return regions(sizes)

    def render(self, p: str, consts: dict[str, int]) -> str:
        if self.slots == 8:
            return _SRC8.render(p=p, g=Attr(self.workspace()), gm=MATH, sigmoid=False, HC=self.hc, D=D, C_LEN=self.c_len,
                                META=self.meta, SMAX=self.slots, eps=repr(self.eps), rs=bf16_hilo(1 / math.sqrt(D)),
                                F_GIN=F_GIN, F_GDN=F_GDN, H_AB=H_AB, H_PRM=H_PRM, H_DOLD=H_DOLD, split=self.split,
                                L_HRDY=L_HRDY, L_HDONE=L_HDONE, H_OUT=H_OUT, **consts,
                                **{k: v for k, v in globals().items() if k.startswith("G_")})
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


def conv_weights(w: np.ndarray) -> np.ndarray:
    """(8 slots) The conv blocks' weights (CH, 4) fp32 -> per block its A object's first 2 KB: [tap 4][hi 128 | lo 128]
    bf16 (NB, 2048) uint8."""
    hi, lo = hilo_bits(np.asarray(w, np.float32).T)  # (4, CH)
    nb = hi.shape[1] // D
    wb = np.stack([hi.reshape(4, nb, D), lo.reshape(4, nb, D)], 2).transpose(1, 0, 2, 3)  # (NB, 4, 2, 128)
    return np.ascontiguousarray(wb).view(np.uint8).reshape(nb, -1)


def head_payload(dt: np.ndarray, ssm_a: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """(8 slots) The header's static part (at H_PRM): dt, A of the column's heads (16 fp32 each), ssm_norm (128 fp32)."""
    out = np.zeros(160, np.float32)
    out[:len(dt)], out[16:16 + len(ssm_a)], out[32:] = dt, ssm_a, norm
    return out.view(np.uint8)


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
