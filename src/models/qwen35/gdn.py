"""The GDN core of the multi-slot decode overlay (``qwen35/decode.py``): one Gated DeltaNet step of S <= 4 token
slots (chunkwise and 8 slots: below) per GDN layer and column, with a lazy commit of the recurrent state.

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

Chunkwise (``GdnShape(chunk=True)``: an MoE model with its drafter, ``overlay.chunk_of``, 4 slots; every design of 8 slots):
the delta rule in its WY / UT form (the prefill's ``gdn_prefill``: its docstring has the equations) in blocks of 8 positions
(position mod 8: the block's lanes), a step about one pass per panel whatever its S, each lane's numbers the same whatever the
steps that run it (a draft rejected or accepted, a step of one or of eight: the outputs of steps of S slots are those of steps
of one, bit for bit). The state is materialized only at a block's end: a step reads its block's base state and recomputes every lane of the
block up to its own (the lanes before it from the k / v histories and the gates of the block, [the 3 rows before the block
| its lanes' rows], which run on in the objects, read from row M_KSHIFT: 8 after a block's end); a lane's U by forward
substitution in lane order (``solve``), O from it and Y; the products along the lanes are per lane in fp32 (bfp16 blocks run
along i, a lane's own), so no lane's rounding depends on another's. The states are double-buffered (``QwenOverlay._half``:
the half a position's block starts from): a step reads its block's half and writes the other, the block's update
(S + dc S + U'^T K^T over its 8 lanes) when it reaches the block's end, else nothing that is read (the other half is
read only after a block's end wrote it); a step past the block's end then runs
the next block's lanes from that update in L1 (their V in o's rows the outputs do not take, their K panels in k2). A
prefill ends anywhere: the decode's first block after it has no lanes before the prefill's end, whose zero gates leave the
state as it is, so that blocks start at multiples of 8 and a prefill (at a multiple of 32) at a block's start, the state
whole. The core runs per v head, its v heads grouped by k head (k head kh's v heads h = kh, kh + 2, ..), on the payload slots
of its groups (each object's metadata the 256 bytes before it, the previous payload's last; the first object's: F_GDN,
M_SKIP).

4 slots: a step's time at S 1 is its objects through DDR, so the core keeps the message on M in its workspace (no drains, no
sync after gdn_in): per k head [P, W, K], per v head [A, H]. P = [the q block's history (7 rows: [base 3 | the step's S rows], taken at cshift) | the block's lanes' gates
at P_D ([lane 12][d 8 | beta 8] fp32) | dt, A, ssm_norm at P_PRM (the first k head's)], W = [q weights | k weights], K = the
k block's history (16 rows), A = [the v block's weights | its history's rows 0 .. 7], H = [rows 8 .. 15 | 2 KB]. Per step,
after the message: per k head P (the gates), W (its q block at the step's lanes, Q's panels), the P object out, K (its k
block over the block's lanes: K's panels, the Gram blocks, K^T's panels), the K object out, then per v head A, H (its V, the
lanes' constants: decays, L and M's columns, ``vhead``), 16 state panels and out: 16 state panels, [its gated outputs (1 KB)
| 1 KB | its history's rows 0 .. 7], [rows 8 .. 15 | 2 KB]. Conv outputs come out in panel layout ([panel 16][lane 8][8]:
``pconv``). Program memory is what bounds this core: cold functions are ``minsize`` where that measured smaller, vectors go by
value through no call (a 64-lane argument costs more at its call sites than an inlined product).

8 slots (``GdnShape(slots=8)``): the message of 8 slots (34 KB) does not fit the core's L1 (which also holds the helper's
scratch: ``decode.design``'s 8-slot partition), so gdn_in's rows reach it through DDR: gdn_in drains them (C) into the step's
objects (3 drains per column, a block's rows (half, slot) as [half][slot 8][64]: the q blocks' and the K objects, the R
objects' v rows then z, ab), then a sync. The header [the slots' ab | dt, A, ssm_norm | the block's lanes' gates (16 lanes,
from the last gated outputs' D object)], per k head [q A, q B, k A, k B, K], per v head [A, B, R]: a conv block's A = [weights
(2 KB) | its history's rows 0 .. 7], B = [rows 8 .. | the step's q rows] (q: [base 3 | the step's S rows], taken at cshift) or
[rows 8 .. 23] (a k or v block's history, block-linear as the 4-slot core's, 24 rows: 3 + 7 lanes before a step + 8 of it +
6 past them), K = the step's k rows, R = [the step's v rows | z]. The step's words come from the gdn_in phase's metadata object
(``F_GIN``). Per step: the header (``gbegin``: the gates), per k head its q block (its history out), its k block and K object
(K's panels; its history out), then per v head A, B, R (its V), 16 state panels, its history and gated outputs out ([k block
16][slot 8][8] bf16: gdn_out's x of the head; the 2 KB after them the block's gates, into the header from the last head, else
the sink). The helper core (``hgdn``: it reaches the core's memory and is idle in a GDN step; asked by the core's lock
``L_HRDY``, done on ``L_HDONE``; the header's ``M_HPAN`` tells it the requests of a step through a K_HELP command of the main
core) computes per k head the Gram blocks and K^T's panels (after its K object; the first v head's R object waits for them),
per v head its chunks' constants (``vhead``, after its R object; the panels wait for them).

Branch (8 slots, ``QwenOverlay.step``'s branch; ``M_BRANCH`` in gdn_in's metadata): the step's last L slots follow slot c
instead of the chain's last, at positions inside slot c's block. The chain is the step's other slots (S counts them), the
branch a third chunk from the block's base over the lanes through its last: the chain's to slot c and its own (its conv
windows: slot c's rows, then its own; its K and Q panels: the chain's with its lanes', its gates after the chain's in dn's
lanes 16 ..), each lane as in a step of one at its position. Its regions lie in the helper's scratch (``HS8_AT``, pinned:
idle while the helper waits in ``hgdn``, which the step's last v head releases only after its panels: the helper's next phase
starts as the first heads' outputs come in), its outputs' lanes go into o's lanes past the chain's, where its slots' outputs
read them.
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
G_KH, G_VI, G_VH = G_PAIR, G_MJ, G_MS  # (chunkwise) the k head, its v heads done, the v head in the column
G_HK = G_KO  # (8 slots) the helper's next work: a k head's Gram blocks (1), a v head's constants (0)
G_KSHIFT = G_PO  # (chunkwise) the k / v histories' base row (M_KSHIFT)
G_NPD, G_NPX = G_NCP, G_COMMIT  # (8 slots) a v head's state panels with a D object (a step that ends its block: its update
# out), without one (else: nothing of them is read)
G_BR = 15  # (8 slots) a branching step's word (``M_BRANCH``: its parent slot + 1 | its slots << 8, or 0)
G_WORDS = 16
G_TA = G_WORDS  # (chunkwise) the SiLU table's addresses
G_BN = G_WORDS + 2  # (8 slots) a branching step's branch chunk: the block's lanes through its last (0: none)
PAN = 72  # bytes of a bfp16 panel (8 x 8)
# (8 slots) the GDN tile's locks of the k and v heads' constants, which the helper core computes (``hgdn``): asked, done;
# the header's metadata word of the helper's runs per step (its heads' and the step's end); the SiLU table's two copies in the core's regions 0, 1 (bytes)
L_HRDY, L_HDONE = 12, 13
M_HPAN = 35
SILU_TAB = (0x739C0, 0x7F9C0)
# (8 slots) the helper's scratch on the GDN tile (local address, pinned: the helper waits in ``hgdn`` until the step's end; the
# cores address it from the workspace, the helper's view of the tile differs from the GDN core's):
# a branching step's branch there, V and O ([panel 16][lane 8][8] fp32) at HB_V, K and Q panels at HB_K, HB_Q, its rows'
# scratch (the 3 rows before it, its own) at HB_R
GWS_AT = 0x4000  # (chunkwise) the GDN workspace's local address (pinned: regions 1 .. 3, below the helper's scratch (8 slots)
# and a SiLU table copy)
HS8_AT = 0xD1C0
HB_V, HB_K, HB_Q, HB_R = 0, 4096, 4096 + 16 * 2 * PAN, 4096 + 32 * 2 * PAN
# (8 slots) a column's objects of a GDN layer: payload slots of XPAY bytes (each object's metadata the 256 bytes before
# its payload): the header (H_SLOT), per k head kh its group [q A, q B, k A, k B, K] (from KH_SLOT + KH_SPAN kh), per v
# head h its group [A, B, R] (from V_SLOT + V_SPAN h); the header's payload: the slots' ab at H_AB (a gdn_in object
# each: 128 bytes), dt 16 | A 16 | ssm_norm 128 fp32 at H_PRM, the block's lanes' gates at H_DN ([lane 16][d 8 | beta 8]
# fp32: the last gated outputs' D object's second half)
H_SLOT, KH_SLOT, KH_SPAN, V_SLOT, V_SPAN = 0, 1, 5, 11, 3
H_AB, H_PRM, H_DN = 0, 1024, 2048
# (chunkwise, 4 slots) a column's objects of a GDN layer: per k head its group [P, W, K], per v head its [A, H]; a P
# object's bytes: its q block's history, the block's lanes' gates at P_D ([lane 12][d 8 | beta 8] fp32), the ssm parameters
# at P_PRM; a K object: its k block's history (16 rows); A: the v block's weights, its history's rows 0 .. 7 at 2048; H: the
# history's rows 8 .. 15
KH_SPAN4, V_SLOT4, V_SPAN4, P_D, P_PRM = 3, 6, 2, 1792, 2560


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
            aie::store_v(dh + 32 * kb, al::gather<4, 8>(g + 8 * kb, {{ D }}));
    }
}

}  // extern "C"
""")


# The chunkwise cores' pieces (module docstring "Chunkwise"): a block's 8 lanes whatever the step's slots
_CA = r"""{% set cold = "noinline, minsize" %}

// The GDN core's workspace (offsets in bytes: GdnShape.workspace())
#define GW(T, off) ((T *)((uint8_t *)ws + (off)))
#define TR(x) aie::op_transpose(x)

namespace {

{{ gm.expm1.src }}
{{ gm.softplus.src }}
{% set silu = km.act(km.SILU, f32=True, table_arg=True, small=True) %}
{{ silu.src }}

// 512 SiLUs: t scratch, y fp32 (the table's addresses run-time values: llvm-aie's gathers want pointers)
[[clang::always_inline]] inline void {{ p }}_silu(const bfloat16 *__restrict x, bfloat16 *__restrict t, float *__restrict y,
                                                 const int32_t *__restrict st)
{
    {{ silu }}<{{ 4 * D }}>(x, t, y, (const uint32_t *)(uintptr_t)st[{{ G_TA }}], (const uint32_t *)(uintptr_t)st[{{ G_TA + 1 }}]);
}

// 1 / rms of 8 rows' squares summed (16 lanes: the rows', 0) over n = 1 or D values (al::inv_rms<n, 8>: powers of two)
__attribute__((noinline)) aie::vector<float, 16> {{ p }}_irms(aie::vector<float, 16> ss, int n)
{
    return al::inv_sqrt_eps<8>(al::mulc<16>(ss, aie::broadcast<bfloat16, 16>(n == 1 ? 1.0f : {{ 1.0 / D }}f)), {{ eps }}f);
}

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
    return al::gather<8, 8>(p, N);
}

// The outputs t < n of the conv, bf16(sum over the taps of w (hi + lo) x) of x rows t .. t + 3 of r (pitch D: the 3 base
// rows, then the lanes'), at lanes (l0 + t) & 7 of y ([panel 16][lane 8][8] bf16; its other lanes untouched); w [tap 4][hi,
// lo][D]
__attribute__((noinline)) void {{ p }}_pconv(const bfloat16 *__restrict r, const bfloat16 *__restrict w, bfloat16 *__restrict y,
                                           int n, int l0)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int c = 0; c < {{ D }}; c += 32) {
        aie::vector<bfloat16, 32> wh[4], wl[4];
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 4; t++) {
            wh[t] = aie::load_v<32>(w + {{ 2 * D }} * t + c);
            wl[t] = aie::load_v<32>(w + {{ 2 * D }} * t + {{ D }} + c);
        }
        const bfloat16 *__restrict pr = r + c;
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int t = 0; t < n; t++) {
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < 4; k++) {
                const aie::vector<bfloat16, 32> xk = aie::load_v<32>(pr + {{ D }} * k);
                acc = aie::mac(aie::mac(acc, xk, wh[k]), xk, wl[k]);
            }
            const aie::vector<bfloat16, 32> v = acc.template to_vector<bfloat16>();
            bfloat16 *__restrict py = y + 8 * c + 8 * ((l0 + t) & 7);
            AIE_LOOP_UNROLL_FULL
            for (int q = 0; q < 4; q++)
                aie::store_v(py + 64 * q, v.template extract<8>(q));
            pr += {{ D }};
        }
    }
}

"""

_CN4 = r"""// The lanes' 1 / rms of o ([panel 16][t 8][j 8] fp32: a lane's values in a row) at sq + 64; sq: 64 + 8 floats of scratch
__attribute__(({{ cold }})) void {{ p }}_gnorm(const float *__restrict o, float *__restrict sq)
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
    aie::store_v(sq + 64, {{ p }}_irms(al::group8<false>(ss.template to_vector<float>()).template grow<16>(), {{ D }}).template extract<8>(0));
}

"""

_L2P4 = r"""// The panels of y ([panel 16][lane 8][8] fp32: a lane's row of D values in its panels) with each lane's row
// l2-normalized (q: times 1 / sqrt(D), the scores' scale; the lanes outside the mask rows zero): per i-block hi, lo at out +
// 2 PAN ib (y unchanged; sq: 64 + 8 floats of scratch)
__attribute__(({{ cold }})) void {{ p }}_l2panels(const float *__restrict y, float *__restrict sq, uint32_t rows, int q,
                                                uint8_t *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    const float *__restrict py = y;
    for (int ib = 0; ib < 16; ib++) {
        aie::vector<bfloat16, 64> a, b;
        al::split2<64>(aie::load_v<64>(py), a, b);
        py += 64;
        ss = aie::mac(aie::mac(aie::mac(ss, a, a), a, b), a, b);
    }
    aie::vector<float, 16> iv = {{ p }}_irms(al::group8<false>(ss.template to_vector<float>()).template grow<16>(), 1);
    if (q)
        iv = al::mul_hilo<16, {{ rs[0] }}f, {{ rs[1] }}f>(iv);
    const aie::vector<float, 64> ib8 = al::bcast8(aie::select(aie::zeros<float, 16>(), iv, aie::mask<16>::from_uint32(rows)).template extract<8>(0));
    const al::BV nI = al::neg_eye();
    py = y;
    uint8_t *__restrict po = out;
    for (int ib = 0; ib < 16; ib++) {
        {{ p }}_push2(aie::accum<accfloat, 64>(al::mulff<64>(aie::load_v<64>(py), ib8)), nI, po);
        py += 64;
        po += {{ 2 * PAN }};
    }
}

"""

_GRAM4 = r"""// The Gram blocks of K panels k and Q panels q ([ib 16][hi, lo] each) on the MMUL: g = K K^T, Q K^T ([t][s] fp32); with
// kt, the K^T panels ([ib][hi, lo] of the [i 8][t 8] blocks: I (K hi + K lo)^T exactly, then split)
__attribute__(({{ cold }})) void {{ p }}_gram(const uint8_t *__restrict k, const uint8_t *__restrict q, float *__restrict g,
                                            uint8_t *__restrict kt)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {
        al::MM8 a(aie::zeros<accfloat, 64>()), b(aie::zeros<accfloat, 64>());
        al::BIn<> ks((const bfp16ebs8 *)k), qs((const bfp16ebs8 *)q);
        for (int ib = 0; ib < 16; ib++) {
            const al::BV kh = ks.pop(), kl = ks.pop(), qh = qs.pop(), ql = qs.pop();
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
    if (!kt)
        return;
    const al::BV I = al::eye(), nI = al::neg_eye();
    al::BIn<> ks((const bfp16ebs8 *)k);
    uint8_t *__restrict po = kt;
    for (int ib = 0; ib < 16; ib++) {
        const al::BV h = ks.pop(), l = ks.pop();
        al::MM8 t(aie::zeros<accfloat, 64>());
        t.mac(I, TR(h));
        t.mac(I, TR(l));
        {{ p }}_push2(t.to_accum(), nI, po);
        po += {{ 2 * PAN }};
    }
}

"""

_CB4 = r"""// The decays of a block's 8 lanes (d fp32, 0 past its lanes): er [row t][16]: lanes s < t E[t][s] = prod_{u = s + 1 .. t}
// (1 + d_u) - 1, lane 8 prod_{u = 0 .. t} (1 + d_u) - 1 (no cancellation for weak decays), three bf16 products per factor;
// gm: lane 8 of each row (Gamma - 1 per lane)
__attribute__(({{ cold }})) void {{ p }}_decays(const float *__restrict d, float *__restrict er, float *__restrict gm)
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
    for (int k = 0; k < 8; k++) {
        const aie::vector<bfloat16, 16> bh = aie::broadcast<bfloat16, 16>(dh[k]), bl = aie::broadcast<bfloat16, 16>(dl[k]);
        aie::vector<bfloat16, 16> eh, el;
        al::split2<16>(e, eh, el);
        aie::accum<accfloat, 16> acc(aie::add(e, aie::broadcast<float, 16>(d[k])));
        acc = aie::mac(aie::mac(aie::mac(acc, bh, eh), bh, el), bl, eh);
        e = aie::select(aie::zeros<float, 16>(), acc.template to_vector<float>(), aie::mask<16>::from_uint32(m));
        aie::store_v(er + 16 * k, e);
        gm[k] = e[8];
        m = ((m & 0xFFu) << 1) | 1u | 0x100u;
    }
}

// Column s of an 8 x 8 block x ([t][s]: lane 8 t + s) as 8 lanes over t
template <int S>
[[clang::always_inline]] inline aie::vector<float, 8> {{ p }}_col(aie::vector<float, 64> x)
{
    const aie::vector<float, 32> a = (S & 4) ? aie::filter_odd(x, 4) : aie::filter_even(x, 4);
    const aie::vector<float, 16> b = (S & 2) ? aie::filter_odd(a, 2) : aie::filter_even(a, 2);
    return (S & 1) ? aie::filter_odd(b, 1) : aie::filter_even(b, 1);
}

// L = beta (1 + E) K K^T (s < t), M = (1 + E) Q K^T (s <= t) from the Gram blocks g and er, as their columns ([s][t] fp32: lc,
// mc; column s: lane s's U's factor in each lane t)
__attribute__((noinline)) void {{ p }}_lm(const float *__restrict be, const float *__restrict g, const float *__restrict er,
                                        float *__restrict lc, float *__restrict mc)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<float, 64> e = {{ p }}_blk<16>(er);
    const aie::vector<float, 64> kk = aie::load_v<64>(g), q = aie::load_v<64>(g + 64);
    const aie::vector<float, 64> l = aie::select(aie::zeros<float, 64>(), al::mulff<64>(al::bcast8(aie::load_v<8>(be)), aie::add(kk, al::mulff<64>(e, kk))),
                                                 aie::mask<64>::from_uint64(0x{{ strict }}ull));
    const aie::vector<float, 64> m = aie::select(aie::zeros<float, 64>(), aie::add(q, al::mulff<64>(e, q)), aie::mask<64>::from_uint64(0x{{ lower }}ull));
{% for c in range(8) %}
    aie::store_v(lc + {{ 8 * c }}, {{ p }}_col<{{ c }}>(l));
    aie::store_v(mc + {{ 8 * c }}, {{ p }}_col<{{ c }}>(m));
{% endfor %}
}

// A chunk's constants for a v head from its lanes' gates (dn: [lane][d 8 | beta 8] fp32): the decays, L and M's columns
// and [beta | Gamma - 1] per lane (cc: [lc 64 | mc 64 | beta 8 | Gamma - 1 8]); chunk 0: the block's lanes 0 .. n1 - 1
// (and the block's update's factors when it ends in the step), chunk 1: the next block's lanes of a step past the block's end
__attribute__(({{ cold }})) void {{ p }}_vhead(int32_t *__restrict ws, int c)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int h = st[{{ G_VH }}], e = st[{{ G_NREP }}] + st[{{ G_NS }}];
    const int n = {% if SMAX == 8 %}c > 1 ? st[{{ G_BN }}] : {% endif %}c ? e - 8 : (e < 8 ? e : 8);
    float *__restrict er = GW(float, {{ g.er }});
    float *__restrict tv = GW(float, {{ g.xy }});  // (d [16] | beta [16])
    const float *__restrict dn = GW(float, {{ g.dn }}) + 128 * c;
    float *__restrict cc = GW(float, {{ g.cc }}) + 144 * c;
    al::zero<32, 16>(tv);
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < n; t++) {
        tv[t] = dn[16 * t + h];
        tv[16 + t] = dn[16 * t + 8 + h];
    }
    {{ p }}_decays(tv, er, cc + 136);
    {{ p }}_lm(tv + 16, GW(float, {{ g.gg }}) + 128 * c, er, cc, cc + 64);
    aie::store_v(cc + 128, aie::load_v<8>(tv + 16));
    if (!c && e >= 8) {  // the block's update's factors: U' = U (1 + E[7][s]) per lane, the state's E of the 8 lanes
        const float *__restrict row = er + 112;
        aie::store_v(GW(float, {{ g.uf }}), aie::add(aie::load_v<8>(row), aie::broadcast<float, 8>(1.0f)));
        aie::vector<bfloat16, 16> ch, cl;
        al::bf16_hilo<16>(row[8], ch, cl);
        aie::store_v(GW(bfloat16, {{ g.dc }}), ch);
        aie::store_v(GW(bfloat16, {{ g.dc }}) + 16, cl);
    }
}

// X, Y of a chunk's lanes on a panel: its 16 [j 8][i 8] blocks (fp32, split hi + lo on the MMUL) against the lanes' K, Q
// panels k, q ([ib][hi, lo] of [t 8][i 8] blocks: a lane's own blocks): xy = [X | Y] ([t][j] fp32)
__attribute__((noinline)) void {{ p }}_xy(const float *__restrict s, const uint8_t *__restrict k, const uint8_t *__restrict q,
                                        float *__restrict xy)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV nI = al::neg_eye();
    al::MM8 x(aie::zeros<accfloat, 64>()), y(aie::zeros<accfloat, 64>());
    al::BIn<> ks((const bfp16ebs8 *)k), qs((const bfp16ebs8 *)q);
    const float *__restrict ps = s;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int ib = 0; ib < 16; ib++) {
        al::BV sh, sl;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(ps)), nI, sh, sl);
        ps += 64;
        const al::BV kh = ks.pop(), kl = ks.pop(), qh = qs.pop(), ql = qs.pop();
        x.mac(kh, TR(sh));
        y.mac(qh, TR(sh));
        x.mac(kh, TR(sl));
        y.mac(qh, TR(sl));
        x.mac(kl, TR(sh));
        y.mac(ql, TR(sh));
    }
    aie::store_v(xy, x.to_accum().template to_vector<float>());
    aie::store_v(xy + 64, y.to_accum().template to_vector<float>());
}

// A chunk's lanes on a panel ([t][j] fp32: X, Y in xy, V in v): R = beta (V - Gamma X); U by forward substitution in lane
// order (lane t: R_t - sum_{s < t} L[t][s] U_s, s increasing: a lane's U is the same in a step of any lanes), O = Gamma Y +
// sum_{s <= t} M[t][s] U_s, over the chunk's n lanes (s < n: the lanes past them take no part in theirs). cc: the chunk's
// constants (``vhead``); u: the lanes' U; o: the lanes of omask (v may be o: read first)
__attribute__((noinline)) void {{ p }}_solve(const float *__restrict xy, const float *v, const float *__restrict cc,
                                           float *__restrict u, float *o, uint64_t omask, int n)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<float, 64> g = al::bcast8(aie::load_v<8>(cc + 136)), be = al::bcast8(aie::load_v<8>(cc + 128));
    const aie::vector<float, 64> x = aie::load_v<64>(xy), y = aie::load_v<64>(xy + 64);
    aie::vector<float, 64> a = al::mulff<64>(be, aie::sub(aie::sub(aie::load_v<64>(v), x), al::mulff<64>(g, x)));
    aie::accum<accfloat, 64> c(aie::add(y, al::mulff<64>(g, y)));
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < n; s++) {
        aie::vector<bfloat16, 64> uh, ul, lh, ll, mh, ml;
        al::split2<64>(a.template extract<8>(s).template grow_replicate<64>(), uh, ul);  // (U_s over j, every row)
        al::split2<64>(al::bcast8(aie::load_v<8>(cc + 8 * s)), lh, ll);
        al::split2<64>(al::bcast8(aie::load_v<8>(cc + 64 + 8 * s)), mh, ml);
        a = aie::msc(aie::msc(aie::msc(aie::accum<accfloat, 64>(a), lh, uh), lh, ul), ll, uh).template to_vector<float>();
        c = aie::mac(aie::mac(aie::mac(c, mh, uh), mh, ul), ml, uh);
    }
    aie::store_v(u, a);
    aie::store_v(o, aie::select(aie::load_v<64>(o), c.template to_vector<float>(), aie::mask<64>::from_uint64(omask)));
}

// so = si + dc si + U'^T K^T in one pass (kt: the K^T panels, up: U'; n: 1, unknown at compile time: the block vectors
// popped ahead of the inner loop, which the register allocation needs)
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

// The panels (hi, lo) of U'^T, U' = U ([t][j]) times uf per lane: [j 8][t 8], blocks along the lanes (the update's sums over
// them): a lane's hi, lo, transposed on the MMUL exactly, then split again
__attribute__((noinline)) void {{ p }}_uprep(const float *__restrict u, const float *__restrict uf, uint8_t *__restrict up)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const al::BV I = al::eye(), nI = al::neg_eye();
    al::BV h, l;
    al::split_hilo(aie::accum<accfloat, 64>(al::mulff<64>(al::bcast8(aie::load_v<8>(uf)), aie::load_v<64>(u))), nI, h, l);
    al::MM8 t(aie::zeros<accfloat, 64>());
    t.mac(I, TR(h));
    t.mac(I, TR(l));
    {{ p }}_push2(t.to_accum(), nI, up);
}

"""
_ENTRY8 = r"""// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) from the header's ab (per slot its gdn_in object:
// alpha HC | beta HC fp32; prm: dt 16 | A 16) into dn at their lanes t0 .. ([lane][d 8 | beta 8] fp32; a branch's slots after
// the chain's)
__attribute__((noinline)) void {{ p }}_ggates(const float *__restrict hab, int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int S = st[{{ G_NS }}] + (st[{{ G_BR }}] >> 8);
    const float *__restrict prm = GW(float, {{ g.prm }});
    float *__restrict dn = GW(float, {{ g.dn }}) + 16 * st[{{ G_NREP }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const aie::vector<float, 16> ab = aie::load_v<16>(hab + {{ C_LEN // 2 }} * s);  // (alpha HC | beta HC | ..)
        // softplus(alpha + dt), then softplus(-beta) in the lanes after: sigmoid(beta) = e^(-softplus(-beta)) (one call)
        const aie::vector<float, 16> sp = {{ gm.softplus }}(aie::select(aie::sub(aie::zeros<float, 16>(), ab), aie::add(ab, aie::load_v<16>(prm)),
                                                                     aie::mask<16>::from_uint32({{ 2 ** HC - 1 }}u)));
        aie::store_v(dn, {{ gm.expm1 }}(al::mulff<16>(sp, aie::load_v<16>(prm + 16))).template extract<8>(0));
        aie::store_v(dn + 8, {{ gm.exp2 }}(al::log2e<16>(aie::sub(aie::zeros<float, 16>(), aie::shuffle_down(sp, {{ HC }})))).template extract<8>(0));
        dn += 16;
    }
}

// The step's rows of a block (a branch's after the chain's), [half 2][slot 8][64] at r -> h ([row][D] bf16)
__attribute__((noinline)) void {{ p }}_rows(int32_t *__restrict ws, bfloat16 *__restrict h, const bfloat16 *__restrict r)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < st[{{ G_NS }}] + (st[{{ G_BR }}] >> 8); s++) {
        aie::store_v(h + {{ D }} * s, aie::load_v<64>(r + 64 * s));
        aie::store_v(h + {{ D }} * s + 64, aie::load_v<64>(r + {{ 64 * SMAX }} + 64 * s));
    }
}

// The conv of rows r on weights w: outputs t < n at lanes (l0 + t) & 7, SiLU: y ([panel 16][lane 8][8] fp32; the other lanes:
// the values of zb's, the conv outputs before; scratch: zb, cc)
__attribute__((noinline)) void {{ p }}_cblock(int32_t *__restrict ws, const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                            int n, int l0, float *__restrict y)
{
    bfloat16 *__restrict yc = GW(bfloat16, {{ g.zb }}), *__restrict tt = GW(bfloat16, {{ g.cc }});
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    {{ p }}_pconv(r, w, yc, n, l0);
    {{ p }}_silu(yc, tt, y, st);
    {{ p }}_silu(yc + {{ 4 * D }}, tt, y + {{ 4 * D }}, st);
}

// A branching step's branch at its lanes (its conv windows: the 3 rows before it, then its own): the rows r3 (its first one's
// 3 rows before) and rb (its own) into the scratch, ``cblock`` -> y (zb's other lanes as they are)
__attribute__((noinline, minsize)) void {{ p }}_bconv(int32_t *__restrict ws, const bfloat16 *__restrict r3, const bfloat16 *__restrict rb,
                                                    const bfloat16 *__restrict w, float *__restrict y)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int L = st[{{ G_BR }}] >> 8;
    bfloat16 *__restrict r = (bfloat16 *)GW(uint8_t, {{ HS_OFF + HB_R }});
    {{ p }}_cp4k(r, r3, {{ 3 * 2 * D // 64 }});
    {{ p }}_cp4k(r + {{ 3 * D }}, rb, L * {{ 2 * D // 64 }});
    {{ p }}_cblock(ws, r, w, L, st[{{ G_BN }}] - L, y);
}

}  // namespace

extern "C" {

// A new X object: the gdn_in phase's metadata object (F_GIN: the step's words; its input objects follow: skipped), another
// main core phase's (skipped too), the header (F_GDN: the ssm parameters, the block's lanes' gates before the step (from row
// M_KSHIFT), the step's; the step's objects follow), or another core's first object (M_SKIP more follow)
__attribute__((minsize)) void {{ p }}_gbegin(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict m = (const int32_t *)x;
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int flags = m[{{ M_FLAGS }}];
    st[{{ G_NSKIP }}] = 0;
    st[{{ G_ON }}] = 0;
    if (flags & {{ F_GIN }}) {  // (a branching step: S counts the chain's slots; the branch chunk the lanes 0 .. t0 + c + L)
        const int br = m[{{ M_BRANCH }}];
        st[{{ G_NS }}] = m[{{ M_NS }}] - (br >> 8);
        st[{{ G_NREP }}] = m[{{ M_NREP }}];
        st[{{ G_CSHIFT }}] = m[{{ M_CSHIFT }}];
        st[{{ G_KSHIFT }}] = m[{{ M_KSHIFT }}];
        st[{{ G_BR }}] = br;
        st[{{ G_BN }}] = br ? m[{{ M_NREP }}] + (br & 255) + (br >> 8) : 0;
        const int ends = m[{{ M_NREP }}] + st[{{ G_NS }}] >= 8;
        st[{{ G_NPD }}] = ends ? 16 : 0;
        st[{{ G_NPX }}] = ends ? 0 : 16;
    }
    if (flags & {{ F_GDN }}) {
        const float *__restrict pay = (const float *)(x + {{ 2 * META }});
        {{ p }}_cp4k(GW(float, {{ g.prm }}), pay + {{ H_PRM // 4 }}, 10);
        {{ p }}_cp4k(GW(float, {{ g.dn }}), pay + {{ H_DN // 4 }} + 16 * st[{{ G_KSHIFT }}], st[{{ G_NREP }}]);
        st[{{ G_TA }}] = 0x{{ "%x" % TAB[0] }};
        st[{{ G_TA + 1 }}] = 0x{{ "%x" % TAB[1] }};
        {{ p }}_ggates(pay + {{ H_AB // 4 }}, ws);
        if (const int nb = st[{{ G_BN }}]) {  // (the branch chunk's gates at dn's lanes 16 ..: the chain's to its parent, its own)
            const int L = st[{{ G_BR }}] >> 8;
            float *__restrict dn = GW(float, {{ g.dn }});
            {{ p }}_cp4k(dn + 256, dn, nb - L);
            {{ p }}_cp4k(dn + 256 + 16 * (nb - L), dn + 16 * (st[{{ G_NREP }}] + st[{{ G_NS }}]), L);
        }
        st[{{ G_CB }}] = 0;
        st[{{ G_PANEL }}] = 0;
        st[{{ G_KH }}] = 0;
        st[{{ G_VI }}] = 0;
        st[{{ G_VH }}] = 0;
        st[{{ G_ON }}] = 1;
    } else if (!(flags & {{ F_MAIN }})) {
        st[{{ G_NSKIP }}] = m[{{ M_SKIP }}];
    } else if (!(flags & {{ F_LOADR }})) {  // (its input objects: of 8 / M_SP strips)
        const int sp = m[{{ M_SP }}], lper = sp == 1 ? 3 : (sp == 2 ? 2 : (sp == 4 ? 1 : 0));
        st[{{ G_NSKIP }}] = (m[{{ M_NX }}] + (1 << lper) - 1) >> lper;
    }
}

// A conv block's first object ([weights: tap 4 x (hi 128 | lo 128) bf16 | its history's rows 0 .. 7]): the weights -> cw;
// a q block's base rows among the rows (its history's rows cshift .. cshift + 2) -> hb's rows 0 .. 2; a k or v block's rows
// -> vh (from M_KSHIFT 8 on: none of them)
__attribute__((minsize)) void {{ p }}_gw(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict w = x + {{ 2 * META }};
    {{ p }}_cp4k(GW(bfloat16, {{ g.cw }}), w, {{ 16 * D // 64 }});
    if (st[{{ G_CB }}] == 0) {
        const int c0 = st[{{ G_CSHIFT }}];
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < 3; i++)
            if (c0 + i < 8)
                {{ p }}_cp4k(GW(bfloat16, {{ g.hb }}) + {{ D }} * i, w + {{ 8 * D }} + {{ D }} * (c0 + i), {{ 2 * D // 64 }});
    } else if (!st[{{ G_KSHIFT }}]) {
        {{ p }}_cp4k(GW(bfloat16, {{ g.vh }}), w + {{ 8 * D }}, {{ 16 * D // 64 }});
    }
}

// A conv block's second object: a q block's ([its history's rows 8 .. 10 | the step's rows as [half 2][slot 8][64]]): the
// rest of the base rows and the step's rows into hb ([base 3 | S rows]: the next step's history), its conv at the step's
// lanes, Q's panels; a k or v block's ([its history's rows 8 .. 23]): the rows into vh (from row M_KSHIFT on)
void {{ p }}_gconv(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict r8 = x + {{ 2 * META }};
    const int b = st[{{ G_CB }}];
    st[{{ G_CB }}] = b + 1;
    if (b) {
        {{ p }}_cp4k(GW(bfloat16, {{ g.vh }}) + {{ D }} * (8 - st[{{ G_KSHIFT }}]), r8, {{ 32 * D // 64 }});
        return;
    }
    bfloat16 *__restrict h = GW(bfloat16, {{ g.hb }});
    const unsigned S = st[{{ G_NS }}], t0 = st[{{ G_NREP }}], c0 = st[{{ G_CSHIFT }}], m = ((1u << S) - 1u) << t0;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 3; i++)
        if (c0 + i >= 8)
            {{ p }}_cp4k(h + {{ D }} * i, r8 + {{ D }} * (c0 + i - 8), {{ 2 * D // 64 }});
    {{ p }}_rows(ws, h + {{ 3 * D }}, r8 + {{ 3 * D }});
    float *__restrict y = GW(float, {{ g.vt }});
    {{ p }}_cblock(ws, h, GW(bfloat16, {{ g.cw }}), S, t0, y);
    {{ p }}_l2panels(y, GW(float, {{ g.xy }}), (m | m >> 8) & 0xFFu, 1, GW(uint8_t, {{ g.qp }}));
    if (const int nb = st[{{ G_BN }}]) {  // (the branch's Q at its lanes: its parent's rows before it, slots c - 2 .. c)
        const int br = st[{{ G_BR }}], L = br >> 8;
        {{ p }}_bconv(ws, h + {{ D }} * (br & 255), h + {{ D }} * (3 + S), GW(bfloat16, {{ g.cw }}), y);
        {{ p }}_l2panels(y, GW(float, {{ g.xy }}), (1u << nb) - (1u << (nb - L)), 1, (uint8_t *)GW(uint8_t, {{ HS_OFF + HB_Q }}));
    }
}

// A k head's K object (the step's k rows as [half 2][slot 8][64]): its block over the block's lanes from vh ([the 3 rows
// before the block | its lanes' rows], the step's at lane t0), K's panels; a branching step's branch K's (the block's lanes to
// its parent, the conv outputs in zb, then its own: the same panels of those lanes); past the block's end in the step, the
// next block's lanes' K panels; then the helper's Gram blocks (``hgdn``)
void {{ p }}_gk(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int t0 = st[{{ G_NREP }}], e = t0 + st[{{ G_NS }}], n = e < 8 ? e : 8, nb = st[{{ G_BN }}];
    bfloat16 *__restrict h = GW(bfloat16, {{ g.vh }});
    float *__restrict y = GW(float, {{ g.vt }}), *__restrict sq = GW(float, {{ g.xy }});
    {{ p }}_rows(ws, h + {{ D }} * (3 + t0), x + {{ 2 * META }});
    {{ p }}_cblock(ws, h, GW(bfloat16, {{ g.cw }}), n, 0, y);
    {{ p }}_l2panels(y, sq, (1u << n) - 1u, 0, GW(uint8_t, {{ g.kq }}));
    if (nb) {
        {{ p }}_bconv(ws, h + {{ D }} * (nb - (st[{{ G_BR }}] >> 8)), h + {{ D }} * (3 + e), GW(bfloat16, {{ g.cw }}), y);
        {{ p }}_l2panels(y, sq, (1u << nb) - 1u, 0, (uint8_t *)GW(uint8_t, {{ HS_OFF + HB_K }}));
    }
    if (e > 8) {  // (the next block's lanes)
        {{ p }}_cblock(ws, h + {{ 8 * D }}, GW(bfloat16, {{ g.cw }}), e - 8, 0, y);
        {{ p }}_l2panels(y, sq, (1u << (e - 8)) - 1u, 0, GW(uint8_t, {{ g.k2 }}));
    }
    st[{{ G_HK }}] = 1;
    release(ws, {{ 48 + L_HRDY }}, 1);
}

// A v head's R object ([the step's v rows | its z rows], each [half 2][slot 8][64]): its block over the block's lanes as V
// ([panel 16][lane 8][8] in vt; past the block's end in the step, the next block's lanes' V in o: lanes the block's outputs
// do not take), z kept; then (the k head's Gram blocks through) the helper's chunk constants
void {{ p }}_gvr(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int t0 = st[{{ G_NREP }}], e = t0 + st[{{ G_NS }}];
    const bfloat16 *__restrict r = x + {{ 2 * META }};
    bfloat16 *__restrict h = GW(bfloat16, {{ g.vh }});
    {{ p }}_rows(ws, h + {{ D }} * (3 + t0), r);
    {{ p }}_cp4k(GW(bfloat16, {{ g.zr }}), r + {{ 8 * D }}, {{ 16 * D // 64 }});
    {{ p }}_cblock(ws, h, GW(bfloat16, {{ g.cw }}), e < 8 ? e : 8, 0, GW(float, {{ g.vt }}));
    if (const int nb = st[{{ G_BN }}])  // (the branch's V: zb's chain lanes, its own at its lanes: SiLU into vb)
        {{ p }}_bconv(ws, h + {{ D }} * (nb - (st[{{ G_BR }}] >> 8)), h + {{ D }} * (3 + e), GW(bfloat16, {{ g.cw }}),
                      (float *)GW(uint8_t, {{ HS_OFF + HB_V }}));
    if (e > 8)  // (the next block's lanes)
        {{ p }}_cblock(ws, h + {{ 8 * D }}, GW(bfloat16, {{ g.cw }}), e - 8, 0, GW(float, {{ g.o }}));
    if (st[{{ G_HK }}]) {  // (the k head's first v head: its Gram blocks first)
        acquire_greater_equal(ws, {{ 48 + L_HDONE }}, 1);
        st[{{ G_HK }}] = 0;
    }
    release(ws, {{ 48 + L_HRDY }}, 1);
}

// The helper's work is through
void {{ p }}_gwait(int32_t *__restrict ws)
{
    acquire_greater_equal(ws, {{ 48 + L_HDONE }}, 1);
}

// (the helper) On the GDN core's workspace gws, its turn taken from the GDN tile's locks (the helper's 0 .. 15): a k head's
// Gram blocks and K^T's panels (past the block's end in the step, the next block's Gram blocks too), a v head's chunk constants
// (``vhead``), or the step's end (its last request: until then the helper's scratch holds a branch's regions); hs: unused
void {{ p }}_hgdn(int32_t *__restrict gws, uint8_t *__restrict hs)
{
    acquire_greater_equal(gws, {{ L_HRDY }}, 1);
    int32_t *__restrict ws = gws;
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int nc = 1 + (st[{{ G_NREP }}] + st[{{ G_NS }}] > 8);
    if (st[{{ G_HK }}] == 2)  // (the step's end: its scratch, which held a branch, is the helper's again)
        return;
    if (st[{{ G_HK }}]) {
        {{ p }}_gram(GW(uint8_t, {{ g.kq }}), GW(uint8_t, {{ g.qp }}), GW(float, {{ g.gg }}), GW(uint8_t, {{ g.kt }}));
        if (nc > 1)
            {{ p }}_gram(GW(uint8_t, {{ g.k2 }}), GW(uint8_t, {{ g.qp }}), GW(float, {{ g.gg }}) + 128, nullptr);
        if (st[{{ G_BN }}])  // (the branch chunk's)
            {{ p }}_gram((const uint8_t *)GW(uint8_t, {{ HS_OFF + HB_K }}), (const uint8_t *)GW(uint8_t, {{ HS_OFF + HB_Q }}),
                         GW(float, {{ g.gg }}) + 256, nullptr);
    } else {
        AIE_LOOP_NO_UNROLL
        for (int c = 0; c < nc; c++)
            {{ p }}_vhead(gws, c);
        if (st[{{ G_BN }}])
            {{ p }}_vhead(gws, 2);
    }
    release(gws, {{ L_HDONE }}, 1);
}

"""

_ENTRY4 = r"""// The slots' gates d = expm1(A softplus(alpha + dt)), beta = sigmoid(beta) from their messages' ab (alpha HC | beta HC fp32;
// prm: dt 16 | A 16) into dn at their lanes t0 .. ([lane][d 8 | beta 8] fp32)
__attribute__((noinline)) void {{ p }}_ggates(int32_t *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int S = st[{{ G_NS }}];
    const float *__restrict prm = GW(float, {{ g.prm }});
    float *__restrict dn = GW(float, {{ g.dn }}) + 16 * st[{{ G_NREP }}];
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {
        const aie::vector<float, 16> ab = aie::load_v<16>((const float *)(GW(bfloat16, {{ g.msg }}) + {{ MSGP }} * s + {{ CH + HC * D }}));
        // softplus(alpha + dt), then softplus(-beta) in the lanes after: sigmoid(beta) = e^(-softplus(-beta)) (one call)
        const aie::vector<float, 16> sp = {{ gm.softplus }}(aie::select(aie::sub(aie::zeros<float, 16>(), ab), aie::add(ab, aie::load_v<16>(prm)),
                                                                     aie::mask<16>::from_uint32({{ 2 ** HC - 1 }}u)));
        aie::store_v(dn, {{ gm.expm1 }}(al::mulff<16>(sp, aie::load_v<16>(prm + 16))).template extract<8>(0));
        aie::store_v(dn + 8, {{ gm.exp2 }}(al::log2e<16>(aie::sub(aie::zeros<float, 16>(), aie::shuffle_down(sp, {{ HC }})))).template extract<8>(0));
        dn += 16;
    }
}

// The step's S rows of conv block b (the message's) -> h ([row][D] bf16)
__attribute__((noinline)) void {{ p }}_rows(int32_t *__restrict ws, bfloat16 *__restrict h, int b)
{
    const bfloat16 *__restrict r = GW(bfloat16, {{ g.msg }}) + {{ D }} * b;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < GW(int32_t, {{ g.st }})[{{ G_NS }}]; s++) {
        {{ p }}_cp4k(h + {{ D }} * s, r, {{ 2 * D // 64 }});
        r += {{ MSGP }};
    }
}

// The conv of rows r on weights w: outputs t < n at lanes (l0 + t) & 7, SiLU: y ([panel 16][lane 8][8] fp32; the other lanes:
// the values of zb's, the conv outputs before; scratch: zb, cc)
__attribute__((noinline)) void {{ p }}_cblock(int32_t *__restrict ws, const bfloat16 *__restrict r, const bfloat16 *__restrict w,
                                            int n, int l0, float *__restrict y)
{
    bfloat16 *__restrict yc = GW(bfloat16, {{ g.zb }}), *__restrict tt = GW(bfloat16, {{ g.cc }});
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    {{ p }}_pconv(r, w, yc, n, l0);
    {{ p }}_silu(yc, tt, y, st);
    {{ p }}_silu(yc + {{ 4 * D }}, tt, y + {{ 4 * D }}, st);
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
    st[{{ G_KSHIFT }}] = m[{{ M_KSHIFT }}];
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

// A k head's P object: its q block's history -> the step's base rows; the first k head's: the ssm parameters, the block's
// lanes' gates before the step (from row M_KSHIFT), the step's
void {{ p }}_gq(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict p = x + {{ 2 * META }};
    if (!st[{{ G_KH }}]) {
        st[{{ G_VI }}] = 0;
        {{ p }}_cp4k(GW(float, {{ g.prm }}), p + {{ P_PRM // 2 }}, 10);
        {{ p }}_cp4k(GW(float, {{ g.dn }}), p + {{ P_D // 2 }} + 32 * st[{{ G_KSHIFT }}], st[{{ G_NREP }}]);
        {{ p }}_ggates(ws);
    }
    {{ p }}_cp4k(GW(bfloat16, {{ g.hb }}), p + {{ D }} * st[{{ G_CSHIFT }}], {{ 3 * 2 * D // 64 }});
}

// A k head's W object ([q weights | k weights]): its q block at the step's lanes, Q's panels; k's weights kept
void {{ p }}_gqw(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict w = x + {{ 2 * META }};
    const unsigned S = st[{{ G_NS }}], t0 = st[{{ G_NREP }}], m = ((1u << S) - 1u) << t0;
    bfloat16 *__restrict h = GW(bfloat16, {{ g.hb }});
    float *__restrict y = GW(float, {{ g.vt }});
    {{ p }}_rows(ws, h + {{ 3 * D }}, st[{{ G_KH }}]);
    {{ p }}_cblock(ws, h, w, S, t0, y);
    {{ p }}_l2panels(y, GW(float, {{ g.xy }}), (m | m >> 8) & 0xFFu, 1, GW(uint8_t, {{ g.qp }}));
    {{ p }}_cp4k(GW(bfloat16, {{ g.cw }}), w + {{ 8 * D }}, {{ 16 * D // 64 }});
}

// A k head's K object (its k block's history: [the 3 rows before the block | its lanes' rows], from row M_KSHIFT on): its k
// block over the block's lanes, K's panels (a lane's own blocks), the Gram blocks and K^T's panels; past the block's end in
// the step, the next block's lanes' K panels and Gram blocks
void {{ p }}_gk(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const bfloat16 *__restrict p = x + {{ 2 * META }};
    const int t0 = st[{{ G_NREP }}], e = t0 + st[{{ G_NS }}], n = e < 8 ? e : 8;
    bfloat16 *__restrict h = GW(bfloat16, {{ g.kh }});
    float *__restrict y = GW(float, {{ g.vt }}), *__restrict sq = GW(float, {{ g.xy }});
    const uint8_t *__restrict qp = GW(uint8_t, {{ g.qp }});
    {{ p }}_cp4k(h, p + {{ D }} * st[{{ G_KSHIFT }}], (3 + t0) * {{ 2 * D // 64 }});
    {{ p }}_rows(ws, h + {{ D }} * (3 + t0), 2 + st[{{ G_KH }}]);
    AIE_LOOP_NO_UNROLL
    for (int c = 0; c < 1 + (e > 8); c++) {  // (the block's lanes, then the next block's)
        const int nc = c ? e - 8 : n;
        uint8_t *__restrict k = c ? GW(uint8_t, {{ g.k2 }}) : GW(uint8_t, {{ g.kq }});
        {{ p }}_cblock(ws, h + {{ 8 * D }} * c, GW(bfloat16, {{ g.cw }}), nc, 0, y);
        {{ p }}_l2panels(y, sq, (1u << nc) - 1u, 0, k);
        {{ p }}_gram(k, qp, GW(float, {{ g.gg }}) + 128 * c, c ? nullptr : GW(uint8_t, {{ g.kt }}));
    }
}

// A v head's A object ([weights | its history's rows 0 .. 7]): the weights, the rows (from M_KSHIFT 8 on: none)
void {{ p }}_gv(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const bfloat16 *__restrict w = x + {{ 2 * META }};
    {{ p }}_cp4k(GW(bfloat16, {{ g.cw }}), w, {{ 16 * D // 64 }});
    if (!GW(int32_t, {{ g.st }})[{{ G_KSHIFT }}])
        {{ p }}_cp4k(GW(bfloat16, {{ g.kh }}), w + {{ 8 * D }}, {{ 16 * D // 64 }});
}

// A v head's H object ([its history's rows 8 .. 15 | 2 KB]): its block over the block's lanes as V ([panel 16][lane 8][8] in
// vt); past the block's end in the step, the next block's lanes' V (in o: lanes the block's outputs do not take); the
// chunks' constants
void {{ p }}_gvh(const bfloat16 *__restrict x, int32_t *__restrict ws)
{
    const int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int t0 = st[{{ G_NREP }}], e = t0 + st[{{ G_NS }}];
    bfloat16 *__restrict h = GW(bfloat16, {{ g.kh }});
    {{ p }}_cp4k(h + {{ D }} * (8 - st[{{ G_KSHIFT }}]), x + {{ 2 * META }}, {{ 16 * D // 64 }});
    {{ p }}_rows(ws, h + {{ D }} * (3 + t0), 4 + st[{{ G_VH }}]);
    {{ p }}_cblock(ws, h, GW(bfloat16, {{ g.cw }}), e < 8 ? e : 8, 0, GW(float, {{ g.vt }}));
    if (e > 8)  // (the next block's lanes)
        {{ p }}_cblock(ws, h + {{ 8 * D }}, GW(bfloat16, {{ g.cw }}), e - 8, 0, GW(float, {{ g.o }}));
    {{ p }}_vhead(ws, 0);  // (after the convs: their scratch is the constants')
    if (e > 8)
        {{ p }}_vhead(ws, 1);
}

"""
_GPANEL4 = r"""// State panel x of the v head (its next one, in place): the block's lanes' X, Y (``xy``), U, O (``solve``; the step's lanes'
// O kept); out (the other half's panel): when the block ends in the step its update over the 8 lanes (S + dc S + U'^T K^T),
// then past the block's end the next block's lanes from it (8 slots: else no D object, ``gpanel0``; 4: nothing)
__attribute__((minsize)) void {{ p }}_gpanel(bfloat16 *__restrict x, int32_t *__restrict ws, float *__restrict out)
{
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int pj = st[{{ G_PANEL }}];
    st[{{ G_PANEL }}] = pj + 1;
    const int t0 = st[{{ G_NREP }}], e = t0 + st[{{ G_NS }}];
    float *sp = (float *)(x + {{ 2 * META }});
    float *__restrict u = GW(float, {{ g.u }}), *__restrict xy = GW(float, {{ g.xy }});
    float *o = GW(float, {{ g.o }}) + 64 * pj;
    const uint8_t *__restrict qp = GW(uint8_t, {{ g.qp }});
    {{ p }}_xy(sp, GW(uint8_t, {{ g.kq }}), qp, xy);
    {{ p }}_solve(xy, GW(float, {{ g.vt }}) + 64 * pj, GW(float, {{ g.cc }}), u, o, ~0ull << (8 * t0), e < 8 ? e : 8);  // (lanes past the step's go nowhere)
{% if SMAX == 8 %}
    if (e >= 8) {  // (out: the other half's, which only a block's end writes before it is read)
{% else %}
    if (e < 8)  // (out: the other half's, which only a block's end writes before it is read)
        return;
{% endif %}
    uint8_t *__restrict up = GW(uint8_t, {{ g.xy }});
    {{ p }}_uprep(u, GW(float, {{ g.uf }}), up);
    {{ p }}_dupdate(sp, out, GW(bfloat16, {{ g.dc }}), GW(uint8_t, {{ g.kt }}), up, st[{{ G_ON }}]);
    if (e > 8) {
        {{ p }}_xy(out, GW(uint8_t, {{ g.k2 }}), qp, xy);
        {{ p }}_solve(xy, o, GW(float, {{ g.cc }}) + 144, u, o, (1ull << (8 * (e - 8))) - 1ull, e - 8);
    }
{% if SMAX == 8 %}
    }
    if (const int nb = st[{{ G_BN }}]) {  // (a branching step's branch: its chunk of lanes 0 .. nb - 1 from the block's base, V and O
        // in vb; its lanes' O into o's lanes past the chain's, where its slots' outputs read them)
        float *vb = (float *)GW(uint8_t, {{ HS_OFF + HB_V }}) + 64 * pj;
        {{ p }}_xy(sp, (const uint8_t *)GW(uint8_t, {{ HS_OFF + HB_K }}), (const uint8_t *)GW(uint8_t, {{ HS_OFF + HB_Q }}), xy);
        {{ p }}_solve(xy, vb, GW(float, {{ g.cc }}) + 288, u, vb, ~0ull, nb);
        const int L = st[{{ G_BR }}] >> 8;
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < L; i++)
            aie::store_v(o + 8 * ((e + i) & 7), aie::load_v<8>(vb + 8 * (nb - L + i)));
    }
{% endif %}
}
{% if SMAX == 8 %}

// A state panel of a step that does not end its block: no D object (nothing of the other half is read before a block's end)
void {{ p }}_gpanel0(bfloat16 *__restrict x, int32_t *__restrict ws)
{
    {{ p }}_gpanel(x, ws, nullptr);
}
{% endif %}

"""
_OUT8 = r"""// The q block's new history ([base 3 | the step's S rows]; a D object's 4 KB from hb on)
void {{ p }}_ghq(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(bfloat16, {{ g.hb }}));
}

// A k or v block's history in the block ([the 3 rows before it | its lanes' rows]), rows 0 .. 15
void {{ p }}_ghv(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(bfloat16, {{ g.vh }}));
}

// ... its rows 16 .. 23 (then 2 KB into the next object's first half: the step's rows, taken)
void {{ p }}_ghv2(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(bfloat16, {{ g.vh }}) + {{ 16 * D }});
}

// The v head's gated outputs rms(o) ssm_norm silu(z) of the step's lanes (t0 + r) & 7 (a branching step's branch slots: o's
// lanes past the chain's, ``gpanel``) -> d ([k block 16][slot 8][8] bf16), then the block's lanes' gates ([lane 16][d 8 | beta 8]
// fp32: the next step's, from the last head's into the header); the next v head (of the next k head after the last of this
// one's). (o: [panel 16][lane 8][8]; z: [half 2][slot 8][64], its SiLU in vt, the panels being through; the slots past the
// step's go nowhere; its kb loop no hardware loop)
void {{ p }}_gout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int t0 = st[{{ G_NREP }}];
    const float *__restrict nw = GW(float, {{ g.prm }}) + 32;
    float *__restrict sz = GW(float, {{ g.vt }});
    {{ p }}_silu(GW(bfloat16, {{ g.zr }}), GW(bfloat16, {{ g.cc }}), sz, st);
    {{ p }}_silu(GW(bfloat16, {{ g.zr }}) + {{ 4 * D }}, GW(bfloat16, {{ g.cc }}), sz + {{ 4 * D }}, st);
    const float *__restrict o = GW(float, {{ g.o }});
    {{ p }}_gnorm(o, GW(float, {{ g.xy }}));
    const aie::vector<float, 64> iv = al::bcast8(aie::shuffle_down_rotate(aie::load_v<8>(GW(float, {{ g.xy }}) + 64), t0));
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int kb = 0; kb < 16; kb++) {
        const float *__restrict ok = o + 64 * kb;
        aie::vector<float, 64> ov;
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++)
            ov.insert(r, aie::load_v<8>(ok + 8 * ((t0 + r) & 7)));
        const aie::vector<float, 64> w = aie::load_v<8>(nw + 8 * kb).template grow_replicate<64>();
        const aie::vector<float, 64> zk = al::gather<8, 8>(sz + {{ 4 * D }} * (kb >> 3) + 8 * (kb & 7), 64);
        const aie::vector<float, 64> t = al::mulff<64>(al::mulff<64>(al::mulff<64>(ov, iv), w), zk);
        aie::store_v(d + 64 * kb, aie::accum<accfloat, 64>(t).template to_vector<bfloat16>());
    }
    {{ p }}_cp4k(d + {{ 8 * D }}, GW(float, {{ g.dn }}), 16);
    st[{{ G_PANEL }}] = 0;
    const int vi = st[{{ G_VI }}] + 1, last = vi == {{ HC // 2 }};
    const int kh = st[{{ G_KH }}] + last;
    st[{{ G_VI }}] = last ? 0 : vi;
    st[{{ G_KH }}] = kh;
    st[{{ G_VH }}] = 2 * (last ? 0 : vi) + kh;
    if (last)
        st[{{ G_CB }}] = 0;
    if (kh == 2) {  // (the step's last v head: the helper's last request, ``hgdn``)
        st[{{ G_HK }}] = 2;
        release(ws, {{ 48 + L_HRDY }}, 1);
    }
}

}  // extern "C"
"""

_OUT4 = r"""// The k head's P object: its q block's new history, the block's lanes' gates, the ssm parameters
void {{ p }}_ghq(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(bfloat16, {{ g.hb }}), {{ (P_PRM + 640) // 64 }});
}

// The k head's K object: its k block's history (the base rows, then the block's lanes')
void {{ p }}_ghk(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(uint8_t, {{ g.kh }}));
}

// The v head's gated outputs rms(o) ssm_norm silu(z) of the step's lanes (t0 + r) & 7, r < 4 -> d ([k block 16][slot 4][8]
// bf16; 1 KB of padding), its history's rows 0 .. 7 2 KB on; the next v head (of the next k head after the last of this one's).
// (o: [panel 16][lane 8][8]; z from the message, its SiLU's rows in vt, the panels being through; the slots past the step's go
// nowhere; its kb loop no hardware loop: a minimum of 1 iteration)
void {{ p }}_gout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    int32_t *__restrict st = GW(int32_t, {{ g.st }});
    const int t0 = st[{{ G_NREP }}];
    const float *__restrict nw = GW(float, {{ g.prm }}) + 32;
    float *__restrict sz = GW(float, {{ g.vt }});
    bfloat16 *__restrict zb = GW(bfloat16, {{ g.zb }});
    const bfloat16 *__restrict z = GW(bfloat16, {{ g.msg }}) + {{ CH }} + {{ D }} * st[{{ G_VH }}];
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < {{ SMAX }}; t++)
        {{ p }}_cp4k(zb + {{ D }} * t, z + {{ MSGP }} * t, {{ 2 * D // 64 }});
    {{ p }}_silu(zb, GW(bfloat16, {{ g.cc }}), sz, st);
    const float *__restrict o = GW(float, {{ g.o }});
    {{ p }}_gnorm(o, GW(float, {{ g.xy }}));
    const aie::vector<float, 32> iv = al::bcast8(aie::shuffle_down_rotate(aie::load_v<8>(GW(float, {{ g.xy }}) + 64), t0)).template extract<32>(0);
    const int l0 = 8 * t0, l1 = 8 * ((t0 + 1) & 7), l2 = 8 * ((t0 + 2) & 7), l3 = 8 * ((t0 + 3) & 7);
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int kb = 0; kb < 16; kb++) {
        const float *__restrict ok = o + 64 * kb;
        const aie::vector<float, 32> ov = al::concat(aie::load_v<8>(ok + l0), aie::load_v<8>(ok + l1), aie::load_v<8>(ok + l2), aie::load_v<8>(ok + l3));
        const aie::vector<float, 32> w = aie::load_v<8>(nw + 8 * kb).template grow_replicate<32>();
        const aie::vector<float, 32> zk = al::gather<4, 8>(sz + 8 * kb, {{ D }});
        const aie::vector<float, 32> t = al::mulff<32>(al::mulff<32>(al::mulff<32>(ov, iv), w), zk);
        aie::store_v(d + 32 * kb, aie::accum<accfloat, 32>(t).template to_vector<bfloat16>());
    }
    {{ p }}_cp4k(d + {{ 8 * D }}, GW(bfloat16, {{ g.kh }}), {{ 16 * D // 64 }});
    st[{{ G_PANEL }}] = 0;
    const int vi = st[{{ G_VI }}] + 1, last = vi == {{ HC // 2 }};
    const int kh = st[{{ G_KH }}] + last;
    st[{{ G_VI }}] = last ? 0 : vi;
    st[{{ G_KH }}] = kh;
    st[{{ G_VH }}] = 2 * (last ? 0 : vi) + kh;
}

// The v head's history's rows 8 .. 15 (then 2 KB that go nowhere)
void {{ p }}_gvhout(int32_t *__restrict ws, bfloat16 *__restrict d)
{
    {{ p }}_cp4k(d, GW(bfloat16, {{ g.kh }}) + {{ 8 * D }}, {{ 16 * D // 64 }});
}

}  // extern "C"
"""

_SRC8 = JINJA_ENV.from_string(_CA + _L2P4 + _CN4 + _GRAM4 + _CB4 + _ENTRY8 + _GPANEL4 + _OUT8)
_SRC4 = JINJA_ENV.from_string(_CA + _L2P4 + _CN4 + _GRAM4 + _CB4 + _ENTRY4 + _GPANEL4 + _OUT4)


class GdnShape:
    """The GDN core's sizes for a model: HC v heads and CH conv channels per column, the message, the workspace; the
    core of steps of up to ``slots`` (4: the message on M; 8: chunkwise, module docstring "8 slots"), of 4 slots ``chunk``wise
    (module docstring "Chunkwise")."""

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
        the header, per k head 5, per v head [A, B, R] + 16 panels."""
        if self.slots == 8:
            return 1 + 2 * KH_SPAN + self.hc * (V_SPAN + 16)
        return V_SLOT4 + self.hc * (V_SPAN4 + 16) if self.chunk else 5 + self.hc // 2 * 35

    @property
    def n_slots(self) -> int:
        """(8 slots) Payload slots of a column's objects of a layer (module docstring "8 slots")."""
        return self.v_slot(self.hc)

    def kh_slot(self, kh: int) -> int:
        """The first payload slot of k head kh's group (8 slots: [q A, q B, k A, k B, K]; 4: [P, W, K])."""
        return KH_SLOT + KH_SPAN * kh if self.slots == 8 else KH_SPAN4 * kh

    def v_slot(self, h: int) -> int:
        """The first payload slot of v head h's group (8 slots: [A, B, R]; 4: [A, H])."""
        return V_SLOT + V_SPAN * h if self.slots == 8 else V_SLOT4 + V_SPAN4 * h

    def block_slot(self, b: int) -> int:
        """(8 slots) The A object's payload slot of conv block b (q heads 0, 1, k heads 0, 1, then the v heads)."""
        return self.kh_slot(b % 2) + 2 * (b // 2) if b < 4 else self.v_slot(b - 4)

    def weights_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's weights."""
        if self.slots == 8 or b >= 4:
            return self.xpay * self.block_slot(b)
        return self.xpay * (self.kh_slot(b % 2) + 1) + (2048 if b >= 2 else 0)

    def hist_at(self, b: int) -> int:
        """Bytes from a column's first payload to conv block b's history."""
        if self.slots == 8 or b >= 4:
            return self.xpay * self.block_slot(b) + 2048
        return self.xpay * (self.kh_slot(b % 2) + (2 if b >= 2 else 0))

    def workspace(self) -> dict[str, int]:
        if self.chunk and self.slots == 4:
            S = self.slots
            # (kq, qp, k2: the block's K, the step's Q (at their lanes), the next block's K panels ([ib 16][hi, lo]); gg: the
            # Gram blocks of the two blocks; cc: per chunk its constants [L's, M's columns | beta | Gamma - 1]; xy: also the
            # scratch of the norms, the gates' rows (vhead) and U' (uprep); kh: a k head's k history, then a v head's;
            # hb | dn | prm: a P object's in a row, its D object one call)
            assert P_D == 2 * (3 + S) * D and P_PRM - P_D == 64 * 12
            sizes = [("st", 4 * (G_WORDS + 2)), ("msg", 2 * S * self.msgp), ("kq", 16 * 2 * PAN), ("qp", 16 * 2 * PAN),
                     ("k2", 16 * 2 * PAN), ("kt", 16 * 2 * PAN), ("gg", 4 * 256), ("vt", 4 * 8 * D), ("o", 4 * 8 * D),
                     ("u", 4 * 64), ("er", 4 * 128 + 64), ("cc", 2 * 4 * 144), ("uf", 4 * 16), ("dc", 2 * 64),
                     ("xy", 4 * 128), ("cw", 2 * 8 * D), ("zb", 2 * 8 * D), ("kh", 4096), ("hb", P_D),
                     ("dn", P_PRM - P_D), ("prm", 4 * 160)]
            out = regions(sizes)
            assert out["prm"] == out["hb"] + P_PRM
            return out
        if self.slots == 8:
            # (the 4-slot chunkwise core's regions but the message; dn: the block's gates of 16 lanes, then a branch chunk's 8;
            # gg, cc: a branch chunk's third; a branch's own regions in the helper's scratch, HS8_AT; vh: a k or v block's
            # history [the 3 rows before the block | its lanes' rows] (24 rows), whose last D object runs 2 KB past it (zr: the v
            # head's z rows, [half 2][slot 8][64] bf16); hb: the q history [base 3 | the step's 8 rows] in vh's first rows (out
            # before the k block's history comes in), a D object's 4 KB from it on)
            sizes = [("st", 4 * (G_BN + 1)), ("prm", 4 * 160), ("dn", 4 * 24 * 16), ("kq", 16 * 2 * PAN), ("qp", 16 * 2 * PAN),
                     ("k2", 16 * 2 * PAN), ("kt", 16 * 2 * PAN), ("gg", 4 * 384), ("vt", 4 * 8 * D), ("o", 4 * 8 * D),
                     ("u", 4 * 64), ("er", 4 * 128 + 64), ("cc", 3 * 4 * 144), ("uf", 4 * 16), ("dc", 2 * 64),
                     ("xy", 4 * 128), ("cw", 2 * 8 * D), ("zb", 2 * 8 * D), ("vh", 2 * 24 * D), ("zr", 2 * 8 * D)]
            out = regions(sizes)
            out["hb"] = out["vh"]
            assert out["vh"] + 2 * 24 * D + D_BYTES // 2 <= out["size"] and HB_R + 5 * 2 * D <= 10240, out
            return out
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
                                PAN=PAN, P_D=P_D, P_PRM=P_PRM, strict=f"{strict:016X}", lower=f"{lower:016X}", TAB=SILU_TAB, **consts,
                                **{k: v for k, v in globals().items() if k.startswith("G_")})
        if self.slots == 8:
            return _SRC8.render(p=p, g=Attr(self.workspace()), gm=MATH, HC=self.hc, D=D, C_LEN=self.c_len, META=self.meta,
                                SMAX=self.slots, eps=repr(self.eps), rs=bf16_hilo(1 / math.sqrt(D)), F_GIN=F_GIN, F_GDN=F_GDN,
                                H_AB=H_AB, H_PRM=H_PRM, H_DN=H_DN, HS_OFF=HS8_AT - GWS_AT, HB_V=HB_V, HB_K=HB_K, HB_Q=HB_Q, HB_R=HB_R,
                                PAN=PAN, strict=f"{strict:016X}", lower=f"{lower:016X}", L_HRDY=L_HRDY, L_HDONE=L_HDONE,
                                TAB=SILU_TAB, **consts, **{k: v for k, v in globals().items() if k.startswith("G_")})
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
