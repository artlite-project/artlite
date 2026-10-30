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
from the last gated outputs' D object), per k head [q A, q B, K (K^T panels), k A, k B], per v head [A, B, DL (U), Z]. A conv block's
two objects: A = [weights (2 KB) | its history's rows 0 .. 7], B = [the history's rows 8 .. 10 | the step's S rows]:
the history (3 + 8 rows, 2816 bytes, after the weights) holds the previous step's [base 3 | S rows], from which the core
takes the step's base rows at cshift, so the host patches nothing per step; the new history [base 3 | the step's S rows]
goes out over it (D: from A's second half on). The step's words (S, M_NREP, M_CSHIFT, M_COMMIT) come from the gdn_in
phase's metadata object (``F_GIN``). Per step: the header (``gbegin``: the gates), per k head its q block, the previous
step's K object, its k block (then the q history, the step's K object and the k history out: 6 KB apart, one drain),
then per v head its block, U and z, its 16 state panels, and its history, U and gated outputs out ([k block 16][slot 8][8]
bf16: gdn_out's x of the head; the 2 KB after them the step's d per token, into the header from the last head, else the
sink). A column's DMA tasks
per layer: 14 fills (its X objects in runs: header + k head 0's group, k head 1's, per v head its group and its
panels) and 21 drains.

The 8-slot core's delta rule is chunkwise (the WY / UT form, the prefill's ``gdn_prefill``, on one block of 8 tokens: its
docstring has the equations): a step costs about one pass per panel, whatever its S (a token-by-token pass per panel
took ~6000 cycles at S 8 against the panel's ~2170 of DMA, and held the weights' stream for it). Per k head the Gram
blocks of its K, Q panels and per v head T = (I + L)^-1 and M (on the helper core, ``hgdn``, which reaches the core's
memory and is idle in a GDN step: asked by the core's lock ``L_HRDY`` after the head's objects, ``L_HDONE`` before they are
used; the header's ``M_HPAN`` tells it the heads of a step through a K_HELP command of the main core), per panel: the
previous step's accepted tokens' update in one pass (the replay, from the kept K^T panels and U), the base out, X^T, Y^T
of the step's tokens on the MMUL, R, U, O from T, M: ~690 cycles at every S. Token rows past S are zero (K, Q, beta, d). A token's numbers then depend on how the tokens were grouped
into steps at the rounding level (hi + lo panels: ~3e-5 of o, ~70x below its bf16 rounding).

4 slots, chunkwise (``GdnShape(chunk=True)``, an MoE model with its drafter: ``overlay.chunk_of``): the
same delta rule on the block's 8 lanes, tokens past S zero, so a step costs one pass per panel at every S up to 4. A step's time
at S 1 is its objects through DDR, so this core keeps the message on M in its workspace (no drains, no sync after gdn_in) and
packs what the 8-slot core spreads: it runs per v head, its v heads
grouped by k head, on the payload slots of its groups (each object's metadata the 256 bytes before it; the first object's:
F_GDN, M_SKIP): per k head [P, W, K], per v head [A, U]. P = [the q block's history | the previous step's d per token at P_D |
dt, A, ssm_norm at P_PRM (the first k head's)], W = [q weights | k weights], K = [the previous step's K^T panels | at K_H the k
block's history], A = [the v block's weights | its history], U = the v head's U of the previous step ([panel 16][j 8][t 8]
fp32); histories of 7 rows ([base 3 | the step's S rows], taken at cshift as in 8 slots). Per step, after the message: per k
head P, W (its q block, Q's panels), the P object out (its new history, the step's d, the parameters), K (its k block, K's panels,
the Gram blocks), the K object out, then per v head A (its block), U (the head's constants: on this core, the helper of an MoE
design has no room for them), 16 state panels and out: 16 state panels, [its gated outputs (1 KB) | 1 KB |
its history], U. Rows, z and ab are read from the message; the D objects' parts lie in a row in the workspace (kt | hb | dn |
prm). Per panel the update and the decay are one pass (``dupdate``: 180 cycles, 345 apart; a panel ~530). Program memory is what bounds this core: cold functions
are ``minsize`` where that measured smaller, vectors go by value through no call (a 64-lane argument costs more at its
call sites than an inlined product).
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
G_HK = G_KO  # (8 slots) the helper's next work: a k head's (1), a v head's (0)
G_WORDS = 16
G_TA = G_WORDS  # (8 slots) the SiLU table's addresses
PAN = 72  # bytes of a bfp16 panel (8 x 8)
# (8 slots) the GDN tile's locks of the k and v heads' constants, which the helper core computes (``hgdn``): asked, done;
# the header's metadata word of the helper's runs per step; the SiLU table's two copies in the core's regions 0, 1 (bytes)
L_HRDY, L_HDONE = 12, 13
M_HPAN = 35
SILU_TAB = (0x739C0, 0x7F9C0)
# (8 slots) a column's objects of a GDN layer: payload slots of XPAY bytes (each object's metadata the 256 bytes before
# its payload): the header (H_SLOT), per k head kh its group [q A, q B, K, k A, k B] (from KH_SLOT + KH_SPAN kh), per v
# head h its group [A, B, DL, Z] (from V_SLOT + V_SPAN h); the header's payload: the slots' ab at H_AB (a gdn_in object
# each: 128 bytes), dt 16 | A 16 | ssm_norm 128 fp32 at H_PRM, the step's d per token ([token 8][8] fp32: the last
# gated outputs' D object's second half) at H_DOLD
H_SLOT, KH_SLOT, KH_SPAN, V_SLOT, V_SPAN = 0, 1, 5, 11, 4
H_AB, H_PRM, H_DOLD = 0, 1024, 2048
# (4 slots, chunkwise) a column's objects of a GDN layer: per k head its group [P, W, K], per v head its [A, U]; a P
# object's bytes: its q block's history, the step's d per token at P_D, the ssm parameters at P_PRM; a K object's K^T panels,
# then at K_H its k block's history
KH_SPAN4, V_SLOT4, V_SPAN4, P_D, P_PRM, K_H = 3, 6, 2, 1792, 2048, 2304


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
# The chunkwise core's pieces (module docstring "8 slots", "4 slots, chunkwise"): rows and sizes of a step's tokens by SMAX
# against the block's 8 lanes
_CA = r"""{% set cold = "noinline" if SMAX == 8 else "noinline, minsize" %}

// The GDN core's workspace (offsets in bytes: GdnShape.workspace())
#define GW(T, off) ((T *)((uint8_t *)ws + (off)))
#define TR(x) aie::op_transpose(x)

namespace {

{{ gm.expm1.src }}
{{ gm.softplus.src }}
{% set silu = km.act(km.SILU, f32=True, table_arg=True) %}
{{ silu.src }}

// 1024 SiLUs: t scratch, y fp32 (the table's addresses run-time values: llvm-aie's gathers want pointers)
[[clang::always_inline]] inline void {{ p }}_silu(const bfloat16 *__restrict x, bfloat16 *__restrict t, float *__restrict y,
                                                 const int32_t *__restrict st)
{
    {{ silu }}<{{ SMAX * D }}>(x, t, y, (const uint32_t *)(uintptr_t)st[{{ G_TA }}], (const uint32_t *)(uintptr_t)st[{{ G_TA + 1 }}]);
}

// 1 / rms of 8 rows' squares summed (16 lanes: the rows', 0)
__attribute__((noinline)) aie::vector<float, 16> {{ p }}_irms(aie::vector<float, 16> ss, int n)
{
    return n == 1 ? al::inv_rms<1, 8>(ss, {{ eps }}f) : al::inv_rms<{{ D }}, 8>(ss, {{ eps }}f);
}

{% if SMAX == 8 %}
// 4 KB
__attribute__((noinline)) void {{ p }}_cp4k(int32_t *__restrict dst, const int32_t *__restrict src)
{
    al::copy<1024, 16>(dst, src);
}
{% else %}
// n lines of 64 B
__attribute__((noinline)) void {{ p }}_cp4k(void *__restrict dst, const void *__restrict src, int n = 64)
{
    int32_t *__restrict d = (int32_t *)dst;
    const int32_t *__restrict s = (const int32_t *)src;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i++) {
        aie::store_v(d, aie::load_v<16>(s));
        d += 16;
        s += 16;
    }
}
{% endif %}

// C += A (hi + lo) B (hi + lo)^T: three products
[[clang::always_inline]] inline void {{ p }}_mm3(al::MM8 &c, const al::BV &ah, const al::BV &al_, const al::BV &bh,
                                               const al::BV &bl)
{
    c.mac(ah, TR(bh));
    c.mac(ah, TR(bl));
    c.mac(al_, TR(bh));
}

// x (64 fp32) -> panels hi, lo at out
[[clang::always_inline]] inline void {{ p }}_push2(aie::accum<accfloat, 64> x, const al::BV &nI, uint8_t *out)
{
    al::BV h, l;
    al::split_hilo(x, nI, h, l);
    al::BOut<> po((bfp16ebs8 *)out);
    po.push(h);
    po.push(l);
}

// an 8 x 8 block of rows at stride N -> 64 lanes [r][c]
template <int N, typename T>
[[clang::always_inline]] inline aie::vector<T, 64> {{ p }}_blk(const T *__restrict p)
{
    return aie::concat(aie::concat(aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + N)),
                                   aie::concat(aie::load_v<8>(p + 2 * N), aie::load_v<8>(p + 3 * N))),
                       aie::concat(aie::concat(aie::load_v<8>(p + 4 * N), aie::load_v<8>(p + 5 * N)),
                                   aie::concat(aie::load_v<8>(p + 6 * N), aie::load_v<8>(p + 7 * N))));
}

{% if SMAX == 8 %}
// x [t 8][8 n values] at row stride N bf16 -> y [panel n][j 8][t 8]
__attribute__(({{ cold }})) void {{ p }}_tr8(const bfloat16 *__restrict x, bfloat16 *__restrict y, int n, int N)
{
    const bfloat16 *__restrict px = x;
    bfloat16 *__restrict py = y;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i++) {
        aie::vector<bfloat16, 64> v;
        const bfloat16 *pr = px;
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 8; t++) {
            v.insert(t, aie::load_v<8>(pr));
            pr += N;
        }
        aie::store_v(py, aie::transpose(v, 8, 8));
        px += 8;
        py += 64;
    }
}

{% endif %}
// y[t][0 .. D) = bf16(sum over the taps of w (hi + lo) x) of the 8 rows t: x rows t .. t + 3 of r (pitch D: the 3 base
// rows, then the slots'), w [tap 4][hi, lo][D]
__attribute__((noinline)) void {{ p }}_conv8(const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                           bfloat16 *__restrict y)
{
    for (int c = 0; c < {{ D }}; c += 32) {
        aie::vector<bfloat16, 32> wh[4], wl[4];
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 4; t++) {
            wh[t] = aie::load_v<32>(w + {{ 2 * D }} * t + c);
            wl[t] = aie::load_v<32>(w + {{ 2 * D }} * t + {{ D }} + c);
        }
        const bfloat16 *__restrict pr = r + c;
        bfloat16 *__restrict py = y + c;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int t = 0; t < {{ SMAX }}; t++) {
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < 4; k++) {
                const aie::vector<bfloat16, 32> xk = aie::load_v<32>(pr + {{ D }} * k);
                acc = aie::mac(aie::mac(acc, xk, wh[k]), xk, wl[k]);
            }
            aie::store_v(py, acc.template to_vector<bfloat16>());
            pr += {{ D }};
            py += {{ D }};
        }
    }
}

"""

_L2ROWS = r"""// The rows y [8][D] (fp32) l2-normalized in place (q: times 1 / sqrt(D), the scores' scale), rows S .. 7 zero; sq: 64 + 8
// floats of scratch
__attribute__((noinline)) void {{ p }}_l2rows(float *y, float *__restrict sq, int S, int q)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *py = y;
    float *__restrict ps = sq;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 8; t++) {
        aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
        AIE_LOOP_NO_UNROLL
        for (int k = 0; k < 2; k++) {
            aie::vector<bfloat16, 64> a, b;
            al::split2<64>(aie::load_v<64>(py), a, b);
            py += 64;
            ss = aie::mac(aie::mac(aie::mac(ss, a, a), a, b), a, b);
        }
        const aie::vector<float, 64> v = ss.template to_vector<float>();
        const aie::vector<float, 32> h = aie::add(v.template extract<32>(0), v.template extract<32>(1));
        const aie::vector<float, 16> g = aie::add(h.template extract<16>(0), h.template extract<16>(1));
        aie::store_v(ps, aie::add(g.template extract<8>(0), g.template extract<8>(1)));
        ps += 8;
    }
    aie::vector<float, 16> iv = {{ p }}_irms(al::group8<false>(aie::load_v<64>(sq)).template grow<16>(), 1);
    if (q)
        iv = al::mul_hilo<16, {{ rs[0] }}f, {{ rs[1] }}f>(iv);
    aie::store_v(sq + 64, aie::select(aie::zeros<float, 16>(), iv, aie::mask<16>::from_uint32((1u << S) - 1u)).template extract<8>(0));
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 16; t++)
        aie::store_v(y + 64 * t, al::mulff<64>(aie::load_v<64>(y + 64 * t), aie::broadcast<float, 64>(sq[64 + (t >> 1)])));
}

"""

_CN = r"""// The 8 tokens' 1 / rms of o ([panel 16][j 8][t 8] fp32) at sq + 64; sq: 64 + 8 floats of scratch
__attribute__(({{ cold }})) void {{ p }}_gnorm(const float *__restrict o, float *__restrict sq{{ ", int n" if SMAX < 8 else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    const float *__restrict po = o;
    for (int kb = 0; kb < 16; kb++) {
        aie::vector<bfloat16, 64> a, b;
        al::split2<64>(aie::load_v<64>(po), a, b);
        po += 64;
        ss = aie::mac(aie::mac(aie::mac(ss, a, a), a, b), a, b);
    }
    const aie::vector<float, 64> q = ss.template to_vector<float>();  // ([j][t]: the sums over the j rows per lane t)
    aie::vector<float, 8> g = aie::zeros<float, 8>();
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < 8; j++)
        g = aie::add(g, q.template extract<8>(j));
    aie::store_v(sq + 64, {{ p }}_irms(g.template grow<16>(), {{ "n" if SMAX < 8 else D }}).template extract<8>(0));
}

"""

_PANELS8 = r"""// The panels of the rows y ([t 8][D] fp32): per i-block hi, lo at out + 288 ib
__attribute__((noinline)) void {{ p }}_panels(const float *__restrict y, uint8_t *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    const float *__restrict py = y;
    uint8_t *__restrict po = out;
    for (int ib = 0; ib < 16; ib++) {
        {{ p }}_push2(aie::accum<accfloat, 64>({{ p }}_blk<{{ D }}>(py)), nI, po);
        py += 8;
        po += {{ 4 * PAN }};
    }
}

"""

_PANELS4 = r"""// The block's rows v ([ib 16][i 8][t 8] fp32) l2-normalized (q: times 1 / sqrt(D), the scores' scale; tokens from S on zero):
// per i-block the panels hi, lo of [i 8][t 8] at kt, of its transpose at kq (4 panels apart: I (hi + lo)^T exactly, then
// split); sq: 64 + 8 floats of scratch
__attribute__((noinline)) void {{ p }}_panels(const float *__restrict v, float *__restrict sq, uint8_t *__restrict kt,
                                            uint8_t *__restrict kq, int S, int q)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ p }}_gnorm(v, sq, 1);
    aie::vector<float, 16> iv = aie::load_v<8>(sq + 64).template grow<16>();
    if (q)
        iv = al::mul_hilo<16, {{ rs[0] }}f, {{ rs[1] }}f>(iv);
    const aie::vector<float, 64> m = aie::select(aie::zeros<float, 16>(), iv, aie::mask<16>::from_uint32((1u << S) - 1u)).template extract<8>(0).template grow_replicate<64>();
    const al::BV I = al::eye(), nI = al::neg_eye();
    const float *__restrict pv = v;
    uint8_t *__restrict pt = kt, *__restrict pq = kq;
    for (int ib = 0; ib < 16; ib++) {
        al::BV h, l;
        al::split_hilo(aie::accum<accfloat, 64>(al::mulff<64>(aie::load_v<64>(pv), m)), nI, h, l);
        pv += 64;
        al::BOut<> po((bfp16ebs8 *)pt);
        po.push(h);
        po.push(l);
        pt += {{ 2 * PAN }};
        al::MM8 t(aie::zeros<accfloat, 64>());
        t.mac(I, TR(h));
        t.mac(I, TR(l));
        {{ p }}_push2(t.to_accum(), nI, pq);
        pq += {{ 4 * PAN }};
    }
}

"""

_GRAM = r"""// The k head's Gram blocks of its K, Q panels (kq: [ib 16][k hi, k lo, q hi, q lo]) on the MMUL: g = K K^T, Q K^T ([t][s]
// fp32), and the K^T panels kt ([ib][hi, lo] of the [i 8][t 8] blocks: I (K hi + K lo)^T exactly, then split)
__attribute__(({{ cold }})) void {{ p }}_gram(const uint8_t *__restrict kq, float *__restrict g{{ ", uint8_t *__restrict kt" if SMAX == 8 else "" }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {
        al::MM8 a(aie::zeros<accfloat, 64>()), b(aie::zeros<accfloat, 64>());
        al::BIn<> ks((const bfp16ebs8 *)kq);
        for (int ib = 0; ib < 16; ib++) {
            const al::BV kh = ks.pop(), kl = ks.pop(), qh = ks.pop(), ql = ks.pop();
            a.mac(kh, TR(kh));
            b.mac(qh, TR(kh));
            a.mac(kh, TR(kl));
            b.mac(qh, TR(kl));
            a.mac(kl, TR(kh));
            b.mac(ql, TR(kh));
        }
        aie::store_v(g, a.to_accum().template to_vector<float>());
        aie::store_v(g + 64, b.to_accum().template to_vector<float>());
    }
{% if SMAX == 8 %}
    const al::BV I = al::eye(), nI = al::neg_eye();
    al::BIn<> ks((const bfp16ebs8 *)kq);
    uint8_t *__restrict po = kt;
    for (int ib = 0; ib < 16; ib++) {
        const al::BV h = ks.pop(), l = ks.pop();
        ks.pop();
        ks.pop();
        al::MM8 t(aie::zeros<accfloat, 64>());
        t.mac(I, TR(h));
        t.mac(I, TR(l));
        {{ p }}_push2(t.to_accum(), nI, po);
        po += {{ 2 * PAN }};
    }
{% endif %}
}

"""

_CB = r"""// The decays of 8 tokens (d fp32, 0 past the step's): er [row t][16]: lanes s < t E[t][s] = prod_{u = s + 1 .. t} (1 + d_u)
// - 1, lane 8 prod_{u = 0 .. t} (1 + d_u) - 1 (no cancellation for weak decays), three bf16 products per factor
__attribute__(({{ cold }})) void {{ p }}_decays(const float *__restrict d, float *__restrict er)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *__restrict dh = (bfloat16 *)(er + 128), *__restrict dl = dh + 16;
    {
        aie::vector<bfloat16, 16> h, l;
        al::split2<16>(aie::load_v<16>(d), h, l);
        aie::store_v(dh, h);
        aie::store_v(dl, l);
    }
    aie::vector<float, 16> e = aie::zeros<float, 16>();
    unsigned m = 0x100u;  // (lanes s < t, the start)
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < {{ SMAX }}; k++) {
        const aie::vector<bfloat16, 16> bh = aie::broadcast<bfloat16, 16>(dh[k]), bl = aie::broadcast<bfloat16, 16>(dl[k]);
        aie::vector<bfloat16, 16> eh, el;
        al::split2<16>(e, eh, el);
        aie::accum<accfloat, 16> acc(aie::add(e, aie::broadcast<float, 16>(d[k])));
        acc = aie::mac(aie::mac(aie::mac(acc, bh, eh), bh, el), bl, eh);
        e = aie::select(aie::zeros<float, 16>(), acc.template to_vector<float>(), aie::mask<16>::from_uint32(m));
        aie::store_v(er + 16 * k, e);
        m = ((m & 0xFFu) << 1) | 1u | 0x100u;
    }
}

// L = beta (1 + E) K K^T (s < t), M = (1 + E) Q K^T (s <= t) from the Gram blocks g and er: L as bf16 hi, lo (lb: [hi |
// lo][t][s]), M's panels at pn + 2 panels
__attribute__((noinline)) void {{ p }}_lm(const float *__restrict be, const float *__restrict g, const float *__restrict er,
                                        bfloat16 *__restrict lb, uint8_t *__restrict pn)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    const aie::vector<float, 64> e = {{ p }}_blk<16>(er);
    const aie::vector<float, 64> kk = aie::load_v<64>(g), q = aie::load_v<64>(g + 64);
    const aie::vector<float, 64> l = al::mulff<64>(al::bcast8(aie::load_v<8>(be)), aie::add(kk, al::mulff<64>(e, kk)));
    aie::vector<bfloat16, 64> h, lo;
    al::split2<64>(aie::select(aie::zeros<float, 64>(), l, aie::mask<64>::from_uint64(0x{{ strict }}ull)), h, lo);
    aie::store_v(lb, h);
    aie::store_v(lb + 64, lo);
    {{ p }}_push2(aie::accum<accfloat, 64>(aie::select(aie::zeros<float, 64>(), aie::add(q, al::mulff<64>(e, q)),
                                                      aie::mask<64>::from_uint64(0x{{ lower }}ull))), nI, pn + {{ 2 * PAN }});
}

// T = (I + L)^-1 by row substitution, r[t] = e_t - sum_{s < t} L[t][s] r[s]: per s three bf16 products into the row's
// accumulator (L from lb, the finished rows as bf16 hi, lo in rb: [hi | lo][t][16]); tr: the rows (fp32); its panels at pn
__attribute__((noinline)) void {{ p }}_tinv(const bfloat16 *__restrict lb, float *__restrict tr, bfloat16 *__restrict rb,
                                          uint8_t *__restrict pn)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
{% if SMAX < 8 %}
    al::zero<128, 16>(tr);  // (the rows from SMAX on: U's lanes stay zero)
{% endif %}
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < {{ SMAX }}; t++) {
        aie::vector<float, 16> x = aie::zeros<float, 16>();
        x[t] = 1.0f;
        aie::store_v(tr + 16 * t, x);
        aie::store_v(rb + 16 * t, aie::accum<accfloat, 16>(x).template to_vector<bfloat16>());
        aie::store_v(rb + 128 + 16 * t, aie::zeros<bfloat16, 16>());
    }
    AIE_LOOP_NO_UNROLL
    for (int t = 1; t < {{ SMAX }}; t++) {
        // (three chains, one per product: a chain of 24 dependent MACs would bind)
        aie::accum<accfloat, 16> a0(aie::load_v<16>(tr + 16 * t)), a1(aie::zeros<accfloat, 16>()), a2(aie::zeros<accfloat, 16>());
        const bfloat16 *__restrict ph = lb + 8 * t;
        const bfloat16 *__restrict qh = rb;
        AIE_LOOP_UNROLL_FULL
        for (int s = 0; s < {{ SMAX }}; s++) {
            const aie::vector<bfloat16, 16> ch = aie::broadcast<bfloat16, 16>(ph[0]), cl = aie::broadcast<bfloat16, 16>(ph[64]);
            const aie::vector<bfloat16, 16> xh = aie::load_v<16>(qh), xl = aie::load_v<16>(qh + 128);
            a0 = aie::msc(a0, ch, xh);
            a1 = aie::msc(a1, ch, xl);
            a2 = aie::msc(a2, cl, xh);
            ph++;
            qh += 16;
        }
        const aie::accum<accfloat, 16> acc = aie::add(aie::add(a0, a1), a2);
        const aie::vector<float, 16> x = acc.template to_vector<float>();
        const aie::vector<bfloat16, 16> h = acc.template to_vector<bfloat16>();
        aie::store_v(tr + 16 * t, x);
        aie::store_v(rb + 16 * t, h);
        aie::store_v(rb + 128 + 16 * t, aie::accum<accfloat, 16>(aie::sub(x, aie::accum<accfloat, 16>(h).template to_vector<float>())).template to_vector<bfloat16>());
    }
    {{ p }}_push2(aie::accum<accfloat, 64>({{ p }}_blk<16>(tr)), al::neg_eye(), pn);
}

// The state's factor over the first R of the tokens with the decays er: (uf: their U factors 1 + E[R - 1][s], 0 from R
// on; dc: E of the R tokens, bf16 hi [64], lo [64])
__attribute__((noinline)) void {{ p }}_fact(const float *__restrict er, int R, float *__restrict uf, bfloat16 *__restrict dc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict row = er + 16 * (R - 1);
    aie::store_v(uf, aie::select(aie::zeros<float, 16>(), aie::add(aie::load_v<16>(row), aie::broadcast<float, 16>(1.0f)),
                                 aie::mask<16>::from_uint32((1u << R) - 1u)).template extract<8>(0));
{% if SMAX == 8 %}
    aie::vector<bfloat16, 64> ch, cl;
    al::bf16_hilo<64>(row[8], ch, cl);
    aie::store_v(dc, ch);
    aie::store_v(dc + 64, cl);
{% else %}
    aie::vector<bfloat16, 16> ch, cl;
    al::bf16_hilo<16>(row[8], ch, cl);
    aie::store_v(dc, ch);
    aie::store_v(dc + 16, cl);
{% endif %}
}

// A v head's constants from its gates: T, M (pn), [beta | Gamma - 1] per token (fv: 64 lanes each), the factors of the
// replay's tokens from their d and of a commit
__attribute__(({{ cold }})) void {{ p }}_vhead(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int h = st[{{ G_VH }}], R = st[{{ G_NREP }}];
    float *__restrict er = GW(float, {{ g.er }});
    float *__restrict tv = GW(float, {{ g.tv }});  // (d [16] | beta [16])
    const float *__restrict gt = GW(float, {{ g.gt }});
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ SMAX }}; t++) {
        tv[t] = gt[32 * t + h];
        tv[16 + t] = gt[32 * t + 16 + h];
    }
    {{ p }}_decays(tv, er);
    uint8_t *__restrict pn = GW(uint8_t, {{ g.pn }});
    {{ p }}_lm(tv + 16, GW(float, {{ g.gg }}), er, GW(bfloat16, {{ g.lb }}), pn);
    {{ p }}_tinv(GW(bfloat16, {{ g.lb }}), GW(float, {{ g.xy }}), GW(bfloat16, {{ g.rb }}), pn);  // (xy: free between panels)
    float *__restrict fv = GW(float, {{ g.fv }});
{% if SMAX == 8 %}
    aie::store_v(fv, aie::load_v<8>(tv + 16).template grow_replicate<64>());
{% else %}
    aie::store_v(fv, aie::load_v<8>(tv + 16));
{% endif %}
    aie::vector<float, 8> es{{ " = aie::zeros<float, 8>()" if SMAX < 8 else "" }};
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < {{ SMAX }}; t++)
        es[t] = er[16 * t + 8];
{% if SMAX == 8 %}
    aie::store_v(fv + 64, es.template grow_replicate<64>());
{% else %}
    aie::store_v(fv + 8, es);
{% endif %}
    if (st[{{ G_COMMIT }}])
        {{ p }}_fact(er, st[{{ G_NS }}], GW(float, {{ g.uf }}) + 8, GW(bfloat16, {{ g.dc }}) + {{ 128 if SMAX == 8 else 32 }});
    if (R) {
        const float *__restrict dold = GW(float, {{ g.dold }});
        AIE_LOOP_NO_UNROLL
        for (int t = 0; t < {{ SMAX }}; t++)
            tv[t] = dold[8 * t + h];
        {{ p }}_decays(tv, er);
        {{ p }}_fact(er, R, GW(float, {{ g.uf }}), GW(bfloat16, {{ g.dc }}));
    }
}

// X^T, Y^T of the step's tokens on a panel: its 16 [j 8][i 8] blocks (fp32, split hi + lo on the MMUL) against the K, Q
// panels: xy = [X^T | Y^T] ([j][t] fp32)
__attribute__((noinline)) void {{ p }}_xy(const float *__restrict s, const uint8_t *__restrict kq, float *__restrict xy)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::MM8 x(aie::zeros<accfloat, 64>()), y(aie::zeros<accfloat, 64>());
    al::BIn<> ks((const bfp16ebs8 *)kq);
    const float *__restrict ps = s;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int ib = 0; ib < 16; ib++) {
        al::BV sh, sl;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(ps)), nI, sh, sl);
        ps += 64;
        const al::BV kh = ks.pop(), kl = ks.pop(), qh = ks.pop(), ql = ks.pop();
        x.mac(sh, TR(kh));
        y.mac(sh, TR(qh));
        x.mac(sh, TR(kl));
        y.mac(sh, TR(ql));
        x.mac(sl, TR(kh));
        y.mac(sl, TR(qh));
    }
    aie::store_v(xy, x.to_accum().template to_vector<float>());
    aie::store_v(xy + 64, y.to_accum().template to_vector<float>());
}

// R^T = beta (V^T - Gamma X^T); U^T = R^T T^T (u: [j][t] fp32); O^T = Gamma Y^T + U^T M^T (o: [j][t] fp32). xy, vt (64
// fp32): the panel's; fv: [beta | Gamma - 1]; pn: [T | M] (hi, lo each)
__attribute__((noinline)) void {{ p }}_solve(const float *__restrict xy, const float *__restrict vt, const float *__restrict fv,
                                           const uint8_t *__restrict pn, float *__restrict u, float *__restrict o)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
{% if SMAX == 8 %}
    const aie::vector<float, 64> g = aie::load_v<64>(fv + 64);
{% else %}
    const aie::vector<float, 64> g = aie::load_v<8>(fv + 8).template grow_replicate<64>();
{% endif %}
    const aie::vector<float, 64> x = aie::load_v<64>(xy), y = aie::load_v<64>(xy + 64);
    al::BV rh, rl;
    al::split_hilo(aie::accum<accfloat, 64>(al::mulff<64>({{ "aie::load_v<64>(fv)" if SMAX == 8 else "aie::load_v<8>(fv).template grow_replicate<64>()" }}, aie::sub(aie::sub(aie::load_v<64>(vt), x), al::mulff<64>(g, x)))), nI, rh, rl);
    al::BIn<> ps((const bfp16ebs8 *)pn);
    const al::BV th = ps.pop(), tl = ps.pop(), mh = ps.pop(), ml = ps.pop();
    al::MM8 uu(aie::zeros<accfloat, 64>());
    {{ p }}_mm3(uu, rh, rl, th, tl);
    const aie::accum<accfloat, 64> ua = uu.to_accum();
    aie::store_v(u, ua.template to_vector<float>());
    al::BV uh, ul;
    al::split_hilo(ua, nI, uh, ul);
    al::MM8 c(aie::accum<accfloat, 64>(aie::add(y, al::mulff<64>(g, y))));
    {{ p }}_mm3(c, uh, ul, mh, ml);
    aie::store_v(o, c.to_accum().template to_vector<float>());
}

{% if SMAX == 8 %}
// so = si + dc si (bf16 hi / lo products: dc bf16 hi [64], lo [64]): a panel's 64 lanes x 16 blocks; si, so distinct, or
// the same buffer as distinct parameters (each block is read before it is written)
__attribute__((noinline)) void {{ p }}_decay(const float *__restrict si, float *__restrict so, const bfloat16 *__restrict dc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 64> ch = aie::load_v<64>(dc), cl = aie::load_v<64>(dc + 64);
    const float __aie_dm_resource_c *__restrict pi = (const float __aie_dm_resource_c *)si;
    float __aie_dm_resource_d *__restrict po = (float __aie_dm_resource_d *)so;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 16; i++) {
        const aie::vector<float, 64> v = aie::load_v<64>(pi);
        pi += 64;
        aie::vector<bfloat16, 64> vh, vl;
        al::split2<64>(v, vh, vl);
        aie::store_v(po, aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 64>(v), vh, ch), vh, cl), vl, ch).template to_vector<float>());
        po += 64;
    }
}

{% else %}
// so = si + dc si + U'^T K^T in one pass (si, so the same panel as distinct parameters, or two; kt: the K^T panels, up: U'; n:
// as ``update``'s in the 8-slot core: 1, unknown at compile time, block vectors popped ahead of the inner loop)
__attribute__((noinline)) void {{ p }}_dupdate(const float *__restrict si, float *__restrict so, const bfloat16 *__restrict dc,
                                             const uint8_t *__restrict kt, const uint8_t *__restrict up, int n)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 64> ch = aie::load_v<16>(dc).template grow_replicate<64>(), cl = aie::load_v<16>(dc + 16).template grow_replicate<64>();
    const float __aie_dm_resource_c *__restrict pi = (const float __aie_dm_resource_c *)si;
    float __aie_dm_resource_d *__restrict po = (float __aie_dm_resource_d *)so;
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < n; j++) {
        al::BIn<> us((const bfp16ebs8 *)up);
        const al::BV uh = us.pop(), ul = us.pop();
        al::BIn<> ks((const bfp16ebs8 *)kt);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int ib = 0; ib < 16; ib++) {
            const aie::vector<float, 64> v = aie::load_v<64>(pi);
            pi += 64;
            aie::vector<bfloat16, 64> vh, vl;
            al::split2<64>(v, vh, vl);
            al::MM8 c(aie::mac(aie::mac(aie::mac(aie::accum<accfloat, 64>(v), vh, ch), vh, cl), vl, ch));
            const al::BV kh = ks.pop(), kl = ks.pop();
            {{ p }}_mm3(c, uh, ul, kh, kl);
            aie::store_v(po, c.to_accum().template to_vector<float>());
            po += 64;
        }
    }
}

{% endif %}
// The panels (hi, lo) of U^T times uf per token -> up
__attribute__((noinline)) void {{ p }}_uprep(const float *__restrict u, const float *__restrict uf, uint8_t *__restrict up)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {{ p }}_push2(aie::accum<accfloat, 64>(al::mulff<64>(aie::load_v<8>(uf).template grow_replicate<64>(), aie::load_v<64>(u))),
                  al::neg_eye(), up);
}

{% if SMAX == 8 %}
// S += U'^T K^T in place (si, so the same panel as distinct parameters): its 16 blocks against the K^T panels kt ([ib][hi,
// lo]), up: U' (popped inside a loop of one iteration, n = 1 unknown at compile time: popped in the function's first block
// and held across the inner loop, block vectors exhaust llvm-aie's register allocator)
__attribute__((noinline)) void {{ p }}_update(const float *__restrict si, float *__restrict so, const uint8_t *__restrict kt,
                                            const uint8_t *__restrict up, int n)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float __aie_dm_resource_c *__restrict pi = (const float __aie_dm_resource_c *)si;
    float __aie_dm_resource_d *__restrict po = (float __aie_dm_resource_d *)so;
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < n; j++) {
        al::BIn<> us((const bfp16ebs8 *)up);
        const al::BV uh = us.pop(), ul = us.pop();
        al::BIn<> ks((const bfp16ebs8 *)kt);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int ib = 0; ib < 16; ib++) {
            al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(pi)));
            pi += 64;
            const al::BV kh = ks.pop(), kl = ks.pop();
            {{ p }}_mm3(c, uh, ul, kh, kl);
            aie::store_v(po, c.to_accum().template to_vector<float>());
            po += 64;
        }
    }
}

{% endif %}
"""

_ENTRY8 = r"""// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) ([slot][d 16 | beta 16] fp32, 0 past the step's;
// prm: dt 16 | A 16) of the header's ab (per slot its gdn_in object: alpha HC | beta HC fp32)
__attribute__((noinline)) void {{ p }}_ggates(const float *__restrict hab, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int S = GW(int32_t, {{ g.st }})[{{ G_NS }}];
    const float *__restrict prm = GW(float, {{ g.prm }});
    float *__restrict gt = GW(float, {{ g.gt }});
    al::zero<{{ 32 * SMAX }}, 16>(gt);
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const float *__restrict ab = hab + {{ C_LEN // 2 }} * s;
        aie::vector<float, 16> al_ = aie::zeros<float, 16>(), be = aie::zeros<float, 16>();
{% for h in range(HC) %}
        al_[{{ h }}] = ab[{{ h }}];
        be[{{ h }}] = ab[{{ HC + h }}];
{% endfor %}
        const aie::vector<float, 16> sp = {{ gm.softplus }}(aie::add(al_, aie::load_v<16>(prm)));
        aie::store_v(gt, {{ gm.expm1 }}(al::mulff<16>(sp, aie::load_v<16>(prm + 16))));
        // sigmoid(beta) = e^(-softplus(-beta)) (one copy of the math less)
        const aie::vector<float, 16> sn = {{ gm.softplus }}(aie::sub(aie::zeros<float, 16>(), be));
        aie::store_v(gt + 16, {{ gm.exp2 }}(al::log2e<16>(aie::sub(aie::zeros<float, 16>(), sn))));
        gt += 32;
    }
}

// The header's payload: dt, A, ssm_norm, the previous step's d per token (the replay's), the slots' ab -> the gates
__attribute__((noinline)) void {{ p }}_ghead(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const float *__restrict pay = (const float *)(x + {{ 2 * META }});
    al::copy<160, 16>(GW(float, {{ g.prm }}), pay + {{ H_PRM // 4 }});
    al::copy<{{ 8 * SMAX }}, 16>(GW(float, {{ g.dold }}), pay + {{ H_DOLD // 4 }});
    {{ p }}_ggates(pay + {{ H_AB // 4 }}, ws);
    st[{{ G_TA }}] = 0x{{ "%x" % TAB[0] }};
    st[{{ G_TA + 1 }}] = 0x{{ "%x" % TAB[1] }};
    st[{{ G_CB }}] = 0;
    st[{{ G_PANEL }}] = 0;
    st[{{ G_KH }}] = 0;
    st[{{ G_VI }}] = 0;
    st[{{ G_VH }}] = 0;
}

}  // namespace

extern "C" {

// A new X object: the gdn_in phase's metadata object (F_GIN: the step's S, replay, shift and commit words; its input
// objects follow: skipped), another main core phase's (skipped too), the header (F_GDN: ``ghead``, the step's objects
// follow), or another core's first object (M_SKIP more follow)
__attribute__((minsize)) void {{ p }}_gbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
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
__attribute__((minsize)) void {{ p }}_gw(const bfloat16 *__restrict x, int32_t *__restrict ws)
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
// the base rows and the step's rows into hb ([base 3 | S rows]: the next step's history; its rows past 3 + S stale), the
// conv of hb's rows (their first S rows the step's), SiLU: q (block 0 of a k head), k (1) l2-normed as panels (kq), then
// the Gram blocks; a v head's as V^T ([panel 16][j 8][t 8] fp32)
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
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        aie::store_v(h + {{ D }} * (3 + s), aie::load_v<64>(r8 + {{ 3 * D }} + 64 * s));
        aie::store_v(h + {{ D }} * (3 + s) + 64, aie::load_v<64>(r8 + {{ 3 * D + 64 * SMAX }} + 64 * s));
    }
    // (scratch until the head's z object and panels: zb, y, o, vt)
    bfloat16 *__restrict yc = GW(bfloat16, {{ g.zb }}), *__restrict t = GW(bfloat16, {{ g.y }});
    float *__restrict ys = GW(float, {{ g.o }}), *__restrict vt = GW(float, {{ g.vt }});
    {{ p }}_conv8(h, GW(bfloat16, {{ g.cw }}), yc);
    if (b >= 2) {
        {{ p }}_tr8(yc, (bfloat16 *)ys, 8, {{ D }});
        {{ p }}_tr8(yc + 64, (bfloat16 *)ys + 512, 8, {{ D }});
        {{ p }}_silu((bfloat16 *)ys, t, vt, st);
        return;
    }
    {{ p }}_silu(yc, t, ys, st);
    uint8_t *__restrict kq = GW(uint8_t, {{ g.kq }});
    {{ p }}_l2rows(ys, GW(float, {{ g.sq }}), S, !b);
    {{ p }}_panels(ys, kq + (b ? 0 : {{ 2 * PAN }}));
    if (b) {  // (the helper: the Gram blocks)
        st[{{ G_HK }}] = 1;
        release(ws, {{ 48 + L_HRDY }}, 1);
    }
}

// A conv block's new history (hb: [base 3 | S rows], 11 rows) -> d
void {{ p }}_ghist(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    al::copy<{{ 11 * D }}, 64>(d, GW(bfloat16, {{ g.hb }}));
}

// The K^T panels of the previous step's tokens of the k head -> the workspace
void {{ p }}_gkold(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    al::copy<{{ 16 * 2 * PAN // 4 }}, 16>(GW(int32_t, {{ g.kold }}), (const int32_t *)(x + {{ 2 * META }}));
}

// The step's K^T panels of the k head
void {{ p }}_gknew(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    al::copy<{{ 16 * 2 * PAN // 4 }}, 16>((int32_t *)d, GW(int32_t, {{ g.kt }}));
}

// A v head's U of the previous step ([panel 16][j 8][t 8] fp32) -> the workspace
void {{ p }}_gdl(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_cp4k(GW(int32_t, {{ g.dl }}), (const int32_t *)(x + {{ 2 * META }}));
}

// A v head's z object ([half 2][slot 8][64] bf16) -> the workspace ([panel 16][j 8][t 8]); the head's constants
void {{ p }}_gz(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const bfloat16 *__restrict z = x + {{ 2 * META }};
    bfloat16 *__restrict zb = GW(bfloat16, {{ g.zb }});
    {{ p }}_tr8(z, zb, 8, 64);
    {{ p }}_tr8(z + {{ 64 * SMAX }}, zb + {{ 64 * 8 }}, 8, 64);
    GW(int32_t, {{ g.st }})[{{ G_HK }}] = 0;
    release(ws, {{ 48 + L_HRDY }}, 1);  // (the helper: the head's constants)
}

// The helper's work is through
void {{ p }}_gwait(int32_t *__restrict ws)
{
    acquire_greater_equal(ws, {{ 48 + L_HDONE }}, 1);
}

// (the helper) On the GDN core's workspace gws, its turn taken from the GDN tile's locks (the helper's 0 .. 15): a k
// head's Gram blocks or a v head's constants; hs: unused
void {{ p }}_hgdn(int32_t *__restrict gws, uint8_t *__restrict hs)
{
    acquire_greater_equal(gws, {{ L_HRDY }}, 1);
    int32_t *__restrict ws = gws;
    if (GW(int32_t, {{ g.st }})[{{ G_HK }}])
        {{ p }}_gram(GW(uint8_t, {{ g.kq }}), GW(float, {{ g.gg }}), GW(uint8_t, {{ g.kt }}));
    else
        {{ p }}_vhead(gws);
    release(gws, {{ L_HDONE }}, 1);
}

"""

_ENTRY4 = r"""// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) ([slot][d 16 | beta 16] fp32, 0 past the step's;
// prm: dt 16 | A 16) of their messages' ab (alpha HC | beta HC fp32)
__attribute__((noinline)) void {{ p }}_ggates(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int S = GW(int32_t, {{ g.st }})[{{ G_NS }}];
    const float *__restrict prm = GW(float, {{ g.prm }});
    float *__restrict gt = GW(float, {{ g.gt }}), *__restrict dn = GW(float, {{ g.dn }});
    al::zero<{{ 32 * SMAX }}, 16>(gt);
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const aie::vector<float, 16> ab = aie::load_v<16>((const float *)(GW(bfloat16, {{ g.msg }}) + {{ MSGP }} * s + {{ CH + HC * D }}));
        // softplus(alpha + dt), then softplus(-beta) in the lanes after: sigmoid(beta) = e^(-softplus(-beta)) (one call)
        const aie::vector<float, 16> sp = {{ gm.softplus }}(aie::select(aie::sub(aie::zeros<float, 16>(), ab), aie::add(ab, aie::load_v<16>(prm)),
                                                                     aie::mask<16>::from_uint32({{ 2 ** HC - 1 }}u)));
        const aie::vector<float, 16> dv = {{ gm.expm1 }}(al::mulff<16>(sp, aie::load_v<16>(prm + 16)));
        aie::store_v(gt, dv);
        aie::store_v(dn, dv.template extract<8>(0));
        aie::store_v(gt + 16, {{ gm.exp2 }}(al::log2e<16>(aie::sub(aie::zeros<float, 16>(), aie::shuffle_down(sp, {{ HC }})))));
        gt += 32;
        dn += 8;
    }
}

// The rows y ([t 4][D] fp32) as a v head's lanes: v [panel 16][j 8][t 8] fp32, lanes t from 4 on zero
__attribute__(({{ cold }})) void {{ p }}_r2p(const float *__restrict y, float *__restrict v)
{
    const float *__restrict py = y;
    float *__restrict pv = v;
    AIE_LOOP_NO_UNROLL
    for (int ib = 0; ib < 16; ib++) {
        const auto [a0, a1] = aie::interleave_zip(aie::load_v<8>(py), aie::load_v<8>(py + {{ 2 * D }}), 1);
        const auto [b0, b1] = aie::interleave_zip(aie::load_v<8>(py + {{ D }}), aie::load_v<8>(py + {{ 3 * D }}), 1);
        const auto [c0, c1] = aie::interleave_zip(aie::concat(a0, a1), aie::concat(b0, b1), 1);
        const auto [d0, d1] = aie::interleave_zip(aie::concat(c0, c1), aie::zeros<float, 32>(), 4);
        aie::store_v(pv, aie::concat(d0, d1));
        py += 8;
        pv += 64;
    }
}

// Conv block b on weights w ([tap 4][hi 128 | lo 128]): the slots' rows after the base in hb ([base 3 | S rows]: the next step's
// history), the conv of its first 4 rows, SiLU: vt ([panel 16][j 8][t 8] fp32)
__attribute__((noinline)) void {{ p }}_cblock(int32_t *__restrict ws, const bfloat16 *__restrict w, int b)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    bfloat16 *__restrict h = GW(bfloat16, {{ g.hb }});
    const bfloat16 *__restrict r = GW(bfloat16, {{ g.msg }}) + {{ D }} * b;
    const int S = st[{{ G_NS }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        {{ p }}_cp4k(h + {{ D }} * (3 + s), r, {{ 2 * D // 64 }});
        r += {{ MSGP }};
    }
    // (scratch: zb, o)
    bfloat16 *__restrict yc = GW(bfloat16, {{ g.zb }});
    {{ p }}_conv8(h, w, yc);
    {{ p }}_silu(yc, yc + {{ SMAX * D }}, GW(float, {{ g.o }}), st);
    {{ p }}_r2p(GW(float, {{ g.o }}), GW(float, {{ g.vt }}));
}

}  // namespace

extern "C" {

// A new X object: a main core's phase (its input objects follow: skipped), the gdn_in phase (F_MSG: the step), or
// another core's first object (M_SKIP more follow)
__attribute__((minsize)) void {{ p }}_gbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
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
    st[{{ G_NREP }}] = m[{{ M_NREP }}];
    st[{{ G_CSHIFT }}] = m[{{ M_CSHIFT }}];
    st[{{ G_COMMIT }}] = m[{{ M_COMMIT }}];
    st[{{ G_MJ }}] = 0;
    st[{{ G_MS }}] = 0;
    st[{{ G_KH }}] = 0;
    st[{{ G_PANEL }}] = 0;
    st[{{ G_TA }}] = 0x{{ "%x" % TAB[0] }};
    st[{{ G_TA + 1 }}] = 0x{{ "%x" % TAB[1] }};
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

// A k head's P object: its q block's history -> the step's base rows; the first k head's: ssm parameters, the previous
// step's d, the gates
void {{ p }}_gq(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict p = x + {{ 2 * META }};
    if (!st[{{ G_KH }}]) {
        st[{{ G_VI }}] = 0;
        {{ p }}_cp4k(GW(float, {{ g.prm }}), p + {{ P_PRM // 2 }}, 10);
        {{ p }}_cp4k(GW(float, {{ g.dold }}), p + {{ P_D // 2 }}, 2);
        {{ p }}_ggates(ws);
    }
    {{ p }}_cp4k(GW(bfloat16, {{ g.hb }}), p + {{ D }} * st[{{ G_CSHIFT }}], {{ 3 * 2 * D // 64 }});
}

// A k head's W object ([q weights | k weights]): its q block, Q's panels; k's weights kept
void {{ p }}_gqw(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict w = x + {{ 2 * META }};
    {{ p }}_cblock(ws, w, st[{{ G_KH }}]);
    {{ p }}_panels(GW(float, {{ g.vt }}), GW(float, {{ g.sq }}), GW(uint8_t, {{ g.kt }}), GW(uint8_t, {{ g.kq }}) + {{ 2 * PAN }}, st[{{ G_NS }}], 1);
    {{ p }}_cp4k(GW(bfloat16, {{ g.cw }}), w + {{ 8 * D }}, {{ 16 * D // 64 }});
}

// A k head's K object ([K^T panels of the previous step | its k block's history]): its k block, K's panels, the Gram blocks
void {{ p }}_gk(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict p = x + {{ 2 * META }};
    uint8_t *__restrict kq = GW(uint8_t, {{ g.kq }});
    {{ p }}_cp4k(GW(uint8_t, {{ g.kold }}), p, {{ 16 * 2 * PAN // 64 }});
    {{ p }}_cp4k(GW(bfloat16, {{ g.hb }}), p + {{ K_H // 2 }} + {{ D }} * st[{{ G_CSHIFT }}], {{ 3 * 2 * D // 64 }});
    {{ p }}_cblock(ws, GW(bfloat16, {{ g.cw }}), 2 + st[{{ G_KH }}]);
    {{ p }}_panels(GW(float, {{ g.vt }}), GW(float, {{ g.sq }}), GW(uint8_t, {{ g.kt }}), kq, st[{{ G_NS }}], 0);
    {{ p }}_gram(kq, GW(float, {{ g.gg }}));
}

// A v head's A object ([weights | history]): its block
void {{ p }}_gv(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict w = x + {{ 2 * META }};
    {{ p }}_cp4k(GW(bfloat16, {{ g.hb }}), w + {{ 8 * D }} + {{ D }} * st[{{ G_CSHIFT }}], {{ 3 * 2 * D // 64 }});
    {{ p }}_cblock(ws, w, 4 + st[{{ G_VH }}]);
}

// A v head's U of the previous step ([panel 16][j 8][t 8] fp32); the head's constants
void {{ p }}_gdl(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_cp4k(GW(int32_t, {{ g.dl }}), x + {{ 2 * META }});
    {{ p }}_vhead(ws);
}

"""

_GPANEL = r"""// State panel x of the v head (its next one, in place): the replay, the base out (or the state after the step's tokens:
// commit), the step's tokens
__attribute__((minsize)) void {{ p }}_gpanel(bfloat16 *__restrict x, int32_t *__restrict ws, float *__restrict out)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int pj = st[{{ G_PANEL }}];
    st[{{ G_PANEL }}] = pj + 1;
    float *sp = (float *)(x + {{ 2 * META }});
    const int R = st[{{ G_NREP }}], commit = st[{{ G_COMMIT }}];
    float *__restrict dl = GW(float, {{ g.dl }}) + 64 * pj;
    float *__restrict uf = GW(float, {{ g.uf }});
    float *s0 = commit ? sp : out;  // (the state before the step's tokens)
    uint8_t *__restrict up = GW(uint8_t, {{ g.up }});
    if (R) {
{% if SMAX == 8 %}
        {{ p }}_decay(sp, s0, GW(bfloat16, {{ g.dc }}));
        {{ p }}_uprep(dl, uf, up);
        {{ p }}_update(s0, s0, GW(uint8_t, {{ g.kold }}), up, R > 0);
{% else %}
        {{ p }}_uprep(dl, uf, up);
        {{ p }}_dupdate(sp, s0, GW(bfloat16, {{ g.dc }}), GW(uint8_t, {{ g.kold }}), up, R > 0);
{% endif %}
    } else if (!commit) {
        {{ p }}_cp4k((int32_t *)out, (const int32_t *)sp);
    }
    float *__restrict xy = GW(float, {{ g.xy }});
    {{ p }}_xy(s0, GW(uint8_t, {{ g.kq }}), xy);
    {{ p }}_solve(xy, GW(float, {{ g.vt }}) + 64 * pj, GW(float, {{ g.fv }}), GW(uint8_t, {{ g.pn }}), dl, GW(float, {{ g.o }}) + 64 * pj);
    if (commit) {
{% if SMAX == 8 %}
        {{ p }}_decay(sp, out, GW(bfloat16, {{ g.dc }}) + 128);
        {{ p }}_uprep(dl, uf + 8, up);
        {{ p }}_update(out, out, GW(uint8_t, {{ g.kt }}), up, commit > 0);
{% else %}
        {{ p }}_uprep(dl, uf + 8, up);
        {{ p }}_dupdate(sp, out, GW(bfloat16, {{ g.dc }}) + 32, GW(uint8_t, {{ g.kt }}), up, commit > 0);
{% endif %}
    }
}

"""

_OUT8 = r"""// The v head's U of the step's tokens (the next step's replay)
void {{ p }}_gdlout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k((int32_t *)d, GW(int32_t, {{ g.dl }}));
}

// The v head's gated outputs rms(o) ssm_norm silu(z) -> d ([k block 16][slot 8][8] bf16), then the step's d per token
// ([token 8][8] fp32: the next step's replay, from the last head's into the header); the next v head (of the next k
// head after the last of this one's). (o, z: [panel 16][j 8][t 8]; the rows past S go nowhere)
void {{ p }}_gout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const float *__restrict nw = GW(float, {{ g.prm }}) + 32;
    float *__restrict sz = GW(float, {{ g.vt }});  // (silu(z): the panels are through)
    {{ p }}_silu(GW(bfloat16, {{ g.zb }}), GW(bfloat16, {{ g.y }}), sz, st);
    const float *__restrict o = GW(float, {{ g.o }});
    {{ p }}_gnorm(o, GW(float, {{ g.sq }}));
    const aie::vector<float, 64> iv = aie::load_v<8>(GW(float, {{ g.sq }}) + 64).template grow_replicate<64>();
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < 16; kb++) {
        const aie::vector<float, 64> w = al::bcast8(aie::load_v<8>(nw + 8 * kb));
        const aie::vector<float, 64> t = al::mulff<64>(al::mulff<64>(al::mulff<64>(aie::load_v<64>(o + 64 * kb), iv), w), aie::load_v<64>(sz + 64 * kb));
        aie::store_v(d + 64 * kb, aie::transpose(aie::accum<accfloat, 64>(t).template to_vector<bfloat16>(), 8, 8));
    }
    float *__restrict dd = (float *)(d + {{ 16 * 64 }});
    for (int r = 0; r < {{ SMAX }}; r++)
        aie::store_v(dd + 8 * r, aie::load_v<8>(GW(float, {{ g.gt }}) + 32 * r));
    st[{{ G_PANEL }}] = 0;
    const int vi = st[{{ G_VI }}] + 1, last = vi == {{ HC // 2 }};
    const int kh = st[{{ G_KH }}] + last;
    st[{{ G_VI }}] = last ? 0 : vi;
    st[{{ G_KH }}] = kh;
    st[{{ G_VH }}] = 2 * (last ? 0 : vi) + kh;
    if (last)
        st[{{ G_CB }}] = 0;
}

}  // extern "C"
"""

_OUT4 = r"""// The k head's P object: its q block's new history, the step's d per token (the next step's replay), the ssm parameters
void {{ p }}_ghq(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(bfloat16, {{ g.hb }}), {{ (P_PRM + 640) // 64 }});
}

// The k head's K object: the step's K^T panels, its k block's new history
void {{ p }}_ghk(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(uint8_t, {{ g.kt }}));
}

// The v head's U of the step's tokens (the next step's replay)
void {{ p }}_gdlout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(int32_t, {{ g.dl }}));
}

// The v head's gated outputs rms(o) ssm_norm silu(z) -> d ([k block 16][slot 4][8] bf16; 1 KB of padding), its new history 2 KB
// on; the next v head (of the next k head after the last of this one's). (o: [panel 16][j 8][t 8]; z from the message, silu(z)
// in vt, the panels being through; the rows past S go nowhere)
void {{ p }}_gout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const float *__restrict nw = GW(float, {{ g.prm }}) + 32;
    float *__restrict sz = GW(float, {{ g.vt }});
    bfloat16 *__restrict zb = GW(bfloat16, {{ g.zb }});
    const bfloat16 *__restrict z = GW(bfloat16, {{ g.msg }}) + {{ CH }} + {{ D }} * st[{{ G_VH }}];
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ SMAX }}; t++)
        {{ p }}_cp4k(zb + {{ D }} * t, z + {{ MSGP }} * t, {{ 2 * D // 64 }});
    {{ p }}_silu(zb, zb + {{ SMAX * D }}, GW(float, {{ g.cw }}), st);
    {{ p }}_r2p(GW(float, {{ g.cw }}), sz);
    const float *__restrict o = GW(float, {{ g.o }});
    {{ p }}_gnorm(o, GW(float, {{ g.sq }}), {{ D }});
    const aie::vector<float, 64> iv = aie::load_v<8>(GW(float, {{ g.sq }}) + 64).template grow_replicate<64>();
    AIE_LOOP_NO_UNROLL
    for (int kb = 0; kb < 16; kb++) {
        const aie::vector<float, 64> w = al::bcast8(aie::load_v<8>(nw + 8 * kb));
        const aie::vector<float, 64> t = al::mulff<64>(al::mulff<64>(al::mulff<64>(aie::load_v<64>(o + 64 * kb), iv), w), aie::load_v<64>(sz + 64 * kb));
        aie::store_v(d + 32 * kb, aie::transpose(aie::accum<accfloat, 64>(t).template to_vector<bfloat16>(), 8, 8).template extract<32>(0));
    }
    {{ p }}_cp4k(d + {{ 16 * 64 }}, GW(bfloat16, {{ g.hb }}), {{ 7 * 2 * D // 64 }});
    st[{{ G_PANEL }}] = 0;
    const int vi = st[{{ G_VI }}] + 1, last = vi == {{ HC // 2 }};
    const int kh = st[{{ G_KH }}] + last;
    st[{{ G_VI }}] = last ? 0 : vi;
    st[{{ G_KH }}] = kh;
    st[{{ G_VH }}] = 2 * (last ? 0 : vi) + kh;
}

}  // extern "C"
"""

_SRC8 = JINJA_ENV.from_string(_CA + _L2ROWS + _CN + _PANELS8 + _GRAM + _CB + _ENTRY8 + _GPANEL + _OUT8)
_SRC4 = JINJA_ENV.from_string(_CA + _CN + _PANELS4 + _GRAM + _CB + _ENTRY4 + _GPANEL + _OUT4)


class GdnShape:
    """The GDN core's sizes for a model: HC v heads and CH conv channels per column, the message, the workspace; the
    core of steps of up to ``slots`` (4: the message on M; 8: module docstring "8 slots"), of 4 slots ``chunk``wise (module
    docstring "4 slots, chunkwise")."""

    def __init__(self, cfg: QwenConfig, *, c_len: int = 64, xpay: int = 4096, meta: int = 64, slots: int = SMAX,
                 chunk: bool = False):
        assert cfg.ssm_state == D and cfg.ssm_k_heads == 2 * COLS and cfg.ssm_v_heads % (2 * COLS) == 0
        assert slots in (SMAX, 8)
        self.slots = slots
        self.chunk = chunk or slots == 8
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
        if self.slots == 8:
            return 1 + 2 * KH_SPAN + self.hc * (V_SPAN + 16)
        return V_SLOT4 + self.hc * (V_SPAN4 + 16) if self.chunk else 5 + self.hc // 2 * 35

    @property
    def n_slots(self) -> int:
        """(8 slots) Payload slots of a column's objects of a layer (module docstring "8 slots")."""
        return self.v_slot(self.hc)

    def kh_slot(self, kh: int) -> int:
        """The first payload slot of k head kh's group (8 slots: [q A, q B, K, k A, k B]; 4: [P, W, K])."""
        return KH_SLOT + KH_SPAN * kh if self.slots == 8 else KH_SPAN4 * kh

    def v_slot(self, h: int) -> int:
        """The first payload slot of v head h's group (8 slots: [A, B, DL, Z]; 4: [A, U])."""
        return V_SLOT + V_SPAN * h if self.slots == 8 else V_SLOT4 + V_SPAN4 * h

    def block_slot(self, b: int) -> int:
        """(8 slots) The A object's payload slot of conv block b (q heads 0, 1, k heads 0, 1, then the v heads)."""
        return self.kh_slot(b % 2) + 3 * (b // 2) if b < 4 else self.v_slot(b - 4)

    def weights_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's weights."""
        if self.slots == 8 or b >= 4:
            return self.xpay * self.block_slot(b)
        return self.xpay * (self.kh_slot(b % 2) + 1) + (2048 if b >= 2 else 0)

    def hist_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's history."""
        if self.slots == 8 or b >= 4:
            return self.xpay * self.block_slot(b) + 2048
        return self.xpay * self.kh_slot(b % 2) + (self.xpay * 2 + K_H if b >= 2 else 0)

    def workspace(self) -> dict[str, int]:
        if self.chunk and self.slots == 4:
            S = self.slots
            # (kt | hb | dn | prm: a K object's and a P object's bytes in a row, the D objects' copies one call each)
            assert K_H + 2 * (3 + S) * D == 4096 and P_D == 2 * (3 + S) * D
            sizes = [("st", 4 * (G_WORDS + 2)), ("msg", 2 * S * self.msgp), ("gt", 4 * S * 32), ("kq", 16 * 4 * PAN), ("kold", 16 * 2 * PAN),
                     ("gg", 4 * 128), ("vt", 4 * 8 * D), ("o", 4 * 8 * D), ("dl", 4 * 8 * D), ("dold", 4 * S * 8), ("er", 4 * 128 + 64),
                     ("tv", 4 * 32), ("lb", 2 * 128), ("rb", 2 * 256), ("pn", 4 * PAN), ("fv", 4 * 16), ("uf", 4 * 16), ("dc", 2 * 64),
                     ("xy", 4 * 128), ("up", 2 * PAN), ("sq", 4 * 80), ("cw", 2 * 8 * D), ("zb", 2 * 8 * D), ("kt", K_H), ("hb", P_D),
                     ("dn", P_PRM - P_D), ("prm", 4 * 160)]
            out = regions(sizes)
            assert out["hb"] == out["kt"] + K_H and out["prm"] == out["hb"] + P_PRM
            return out
        if self.slots == 8:
            S = self.slots
            sizes = [("st", 4 * (G_WORDS + 2)), ("prm", 4 * 160), ("gt", 4 * S * 32), ("kq", 16 * 4 * PAN), ("kt", 16 * 2 * PAN),
                     ("kold", 16 * 2 * PAN), ("gg", 4 * 128), ("vt", 4 * S * D), ("o", 4 * S * D), ("dl", 4 * S * D),
                     ("dold", 4 * S * 8), ("er", 4 * 128 + 64), ("tv", 4 * 32), ("lb", 2 * 128), ("rb", 2 * 256),
                     ("pn", 4 * PAN), ("fv", 4 * 128), ("uf", 4 * 16), ("dc", 2 * 256), ("xy", 4 * 128), ("up", 2 * PAN),
                     ("sq", 4 * 80), ("cw", 2 * 8 * D), ("zb", 2 * S * D), ("y", 2 * S * D), ("hb", 2 * 11 * D)]
            return regions(sizes)
        sizes = [("st", 4 * G_WORDS), ("prm", 4 * 160), ("msg", 2 * SMAX * self.msgp), ("kq", 2 * SMAX * 8 * D),
                 ("gt", 4 * SMAX * 48), ("v", 4 * 2 * SMAX * D), ("o", 4 * 2 * SMAX * D), ("dl", 4 * 2 * SMAX * D),
                 ("kold", 2 * SMAX * 4 * D), ("dold", 4 * SMAX * 16), ("y", max(4 * D, 2 * SMAX * D)), ("uw", 4 * 16)]
        return regions(sizes)

    def render(self, p: str, consts: dict[str, int]) -> str:
        strict = sum(1 << (8 * t + j) for t in range(8) for j in range(t))
        lower = sum(1 << (8 * t + j) for t in range(8) for j in range(t + 1))
        if self.chunk and self.slots == 4:
            return _SRC4.render(p=p, g=Attr(self.workspace()), gm=MATH, HC=self.hc, CH=self.ch, D=D, C_LEN=self.c_len, META=self.meta,
                                MSGP=self.msgp, n_msg=self.n_msg, SMAX=self.slots, eps=repr(self.eps), rs=bf16_hilo(1 / math.sqrt(D)),
                                PAN=PAN, P_D=P_D, P_PRM=P_PRM, K_H=K_H, strict=f"{strict:016X}", lower=f"{lower:016X}", TAB=SILU_TAB, **consts,
                                **{k: v for k, v in globals().items() if k.startswith("G_")})
        if self.slots == 8:
            return _SRC8.render(p=p, g=Attr(self.workspace()), gm=MATH, HC=self.hc, D=D, C_LEN=self.c_len, META=self.meta,
                                SMAX=self.slots, eps=repr(self.eps), rs=bf16_hilo(1 / math.sqrt(D)), F_GIN=F_GIN, F_GDN=F_GDN,
                                H_AB=H_AB, H_PRM=H_PRM, H_DOLD=H_DOLD, PAN=PAN, strict=f"{strict:016X}", lower=f"{lower:016X}",
                                L_HRDY=L_HRDY, L_HDONE=L_HDONE, TAB=SILU_TAB, **consts, **{k: v for k, v in globals().items() if k.startswith("G_")})
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
