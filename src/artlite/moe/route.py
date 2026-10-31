"""An MoE layer's routing on the decode overlay's cores (``decode.design``'s "MoE": the helper core of every column runs
it on the route object), model-independent: E experts, the top K of each token slot, softmax weights over the K
(llama.cpp's ``build_moe_ffn`` with softmax gating, the weights renormalized: the tests' ``qwen35/ref.py``).

Per slot the core takes the K largest logits (fp32, compared as ordered integer keys of their bits: ties go to the
lower id) and their weights w = e^(l - l_max) / sum over the K (fp32 to ~2^-16: ``kmath.exp2``,
``kmath.recip``). The union U of the slots' experts (by id) is dealt over the COLS columns (``deal``): U[i] whole to
column i % COLS for i < COLS nw (nw = |U| // COLS), the remaining r experts' 2 r halves j = 2 t + h to column j % COLS;
so every column gets ceil(2 |U| / COLS) halves at most, in at most nw + 2 <= MAX_ITEMS runs (items), the same deal on
every core. An item's weight in a slot: its expert's weight in the slot's top K, else 0.

Column c's items become control packets (``packet_words``): each writes a whole shim BD (``EXPERT_BDS`` in turn: the
item's halves as one run of the image, half k at M_RBASE + k M_RSTR, chained to the next item's BD), then one push on
MM2S0 (the A stream) and a last write of shim lock ``LOCK_DONE`` (the transaction waits for it before its own later A
pushes, which would otherwise overtake the array's). Only the length and address words are computed; the other words
are constants of this source. The core emits the packets, then the expert object's metadata (``decode.design``
M_NITEM, M_ITEM: the items and their weights; M_NT their A objects), as ``RBUF_OBJECTS`` C objects: the transaction
drains them (the drain releases ``LOCK_CP``), streams the packets back into the shim's own TileControl (the control
task acquires ``LOCK_CP``) and later the expert object from DDR into the X stream.
"""

import numpy as np

from .. import txn
from ..operators.base import JINJA_ENV
from ..operators.fused.ctrlpkt import _parity, _slot

EXPERT_BDS = (3, 12, 13, 14, 15)  # the shim BDs of a column's items (A: 0 .. 2, the GDN cores' D stream: 12 .. 15, idle
# during the FFN)
MAX_ITEMS = len(EXPERT_BDS)
LOCK_CP, LOCK_DONE = 0, 1  # shim locks: the packets are in DDR; the array's last packet is through
CID = 9  # the control packets' stream id (on the X stream's packet-switched channel, routed to the shim's TileControl)
PACKET_WORDS = 12 * MAX_ITEMS + 3 + 3  # (per BD two 4-word writes, the push, the lock write; each with 2 header words)
RBUF_OBJECTS = 5  # C objects of the routing: the packets (3 of 32 words), the expert object's metadata (2)
RBUF_META = 96  # the metadata's first word in them
# the expert object's plan (the main core's workspace): its metadata, the gate / up and down templates, the
# sub-phases, a sub-phase's metadata
PLAN_SUBS, PLAN_META, PLAN_WORDS = 192, 208, 272
# shim BD words 3 .. 7 of an item's task: no dimensions, 512-byte bursts, AxCACHE 2, AxQoS 0 (never nonzero), valid
BD_W4, BD_W5, BD_VALID, BD_USE_NEXT = 3 << 30, 2 << 24, 1 << 25, 1 << 26


# the control packets' route on a shim's MM2S1 (stream-switch slave South 7; aiecc: slot 0, the X objects' id -> a low
# arbiter -> North, each packet flow of the shim its own from 0 up; its control-packet overlay: arbiter 5): slot 1 (CID) ->
# arbiter 4 -> master Tile_Ctrl, packet mode, the header kept (AM029 PL_MODULE Stream_Switch_*)
SW_SOUTH7_SLOT1, SW_MASTER_TILE_CTRL, CTL_ARB = 0x3F294, 0x3F000, 4
SW_SLOTS = range(0x3F200, 0x3F400, 4)  # (the slave ports' slots)


def control_route(t: txn.Txn, col: int) -> None:
    """Route column col's control packets on its shim's MM2S1 to its TileControl (two writes; ``check_route``)."""
    t.write32(txn.tile_address(col, 0, SW_SOUTH7_SLOT1), _slot(CID, CTL_ARB))
    t.write32(txn.tile_address(col, 0, SW_MASTER_TILE_CTRL), 1 << 31 | 1 << 30 | 1 << 3 | CTL_ARB)


def check_route(values: dict[int, int], cols: range) -> None:
    """The configuration (``paging.cdo_values``) leaves the control route's registers to ``control_route`` and no stream
    on its arbiter: the words of such a stream would reach TileControl."""
    for c in cols:
        for reg in (SW_SOUTH7_SLOT1, SW_MASTER_TILE_CTRL):
            assert txn.tile_address(c, 0, reg) not in values, f"column {c}: the configuration writes {reg:#x}"
        for reg in SW_SLOTS:
            v = values.get(txn.tile_address(c, 0, reg), 0)
            assert not (v >> 8 & 1 and v & 7 == CTL_ARB and not v >> 4 & 3), f"column {c}: {reg:#x} = {v:#x} on the control route's arbiter"


def top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The cores' top k of each slot's fp32 logits (S, E): (ids (S, k) in decreasing logit order, ties to the lower id;
    weights (S, k): the softmax over the k, float64)."""
    lg = np.asarray(logits, np.float32)
    order = np.argsort(-lg.astype(np.float64), axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(lg, order, 1).astype(np.float64)
    e = np.exp(top - top[:, :1])
    return order, e / e.sum(1, keepdims=True)


def deal(union: list[int], cols: int) -> list[list[tuple[int, int, int]]]:
    """The columns' items (expert, first half, last half) of the union's experts (increasing ids): module docstring."""
    nw, r = divmod(len(union), cols)
    out: list[list[tuple[int, int, int]]] = [[] for _ in range(cols)]
    for i in range(cols * nw):
        out[i % cols].append((union[i], 0, 1))
    for j in range(2 * r):
        out[j % cols].append((union[cols * nw + j // 2], j % 2, j % 2))
    assert all(len(items) <= MAX_ITEMS for items in out)
    return out


def route(logits: np.ndarray, k: int, cols: int) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int, int]]]]:
    """The routing of S slots' logits (S, E): (ids, weights: ``top_k``; the columns' items: ``deal`` of the union)."""
    ids, w = top_k(logits, k)
    return ids, w, deal(sorted({int(e) for e in ids.reshape(-1)}), cols)


def item_weights(ids: np.ndarray, w: np.ndarray, e: int) -> np.ndarray:
    """An item's weights per slot (S,): expert e's weight in each slot's top k, else 0."""
    hit = ids == e
    return np.where(hit.any(1), (w * hit).sum(1), 0.0)


def _ctl(addr: int, n: int) -> int:
    """The control header of a write of n words at a shim's register addr (odd parity)."""
    return _parity(((n - 1) << 20) | (addr & 0xFFFFF))


def packet_words(items: list[tuple[int, int, int]], *, base: int, stride: int, a_packet: int | None = None) -> np.ndarray:
    """The control packets of a column's items (``PACKET_WORDS`` uint32, as the core writes them): the halves of item i
    (k = 2 e + h0 .. 2 e + h1) at base + k stride, in BD EXPERT_BDS[i] chained to the next item's; the unused BDs'
    words zero (never pushed). a_packet: the A stream's packet id, where it is packet-switched."""
    sh = _parity(CID)
    out: list[int] = []
    n = len(items)
    assert 1 <= n <= MAX_ITEMS
    for i, bd in enumerate(EXPERT_BDS):
        w = [0] * 8
        if i < n:
            e, h0, h1 = items[i]
            a = base + (2 * e + h0) * stride
            w2 = 0 if a_packet is None else 1 << 30 | a_packet << 19
            w = [(h1 - h0 + 1) * stride // 4, a & 0xFFFFFFFF, (a >> 32) & 0xFFFF | w2, 0, BD_W4, BD_W5, 0,
                 BD_VALID | (BD_USE_NEXT | EXPERT_BDS[i + 1] << 27 if i + 1 < n else 0)]
        reg = txn.SHIM_BD0 + txn.BD_BYTES * bd
        out += [sh, _ctl(reg, 4), *w[:4], sh, _ctl(reg + 16, 4), *w[4:]]
    out += [sh, _ctl(txn.QUEUE[txn.MM2S][0], 1), EXPERT_BDS[0]]
    out += [sh, _ctl(txn.SHIM_LOCK0 + 16 * LOCK_DONE, 1), 1]
    assert len(out) == PACKET_WORDS
    return np.array(out, np.uint32)


_SRC = JINJA_ENV.from_string(r"""
// ---- MoE routing (moe/route.py): the helper, on the route object m (its META words, then the slots'
// logits [slot][E] fp32) for column col -> the control packets and the expert object's metadata in rbuf (the main core
// emits them next), the shared expert's gate per slot (sigmoid of S_SG) in wsc
alignas(64) static const uint32_t {{ p }}_ech[{{ 2 * MAX_ITEMS }}] = { {% for h in ch %}{{ h }}u, {% endfor %}};  // (the BDs' control headers)
alignas(64) static const int32_t {{ p }}_ebd[{{ MAX_ITEMS + 1 }}] = { {% for b in bds %}{{ b }}, {% endfor %}0 };

{% set exp2, recip = km.exp2(16, prec="fp32", inline=True), km.recip(16, domain="normal", steps=3) %}
{{ exp2.src }}
{{ recip.src }}

// v (16 fp32, in place) -> e^v, 1 / v (fp32 to ~2^-16; one copy each)
__attribute__((noinline)) void {{ p }}_rexp(float *__restrict v)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::store_v(v, {{ exp2 }}(al::log2e<16>(aie::load_v<16>(v))));
}

__attribute__((noinline)) void {{ p }}_rrcp(float *__restrict v)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    aie::store_v(v, {{ recip }}(aie::load_v<16>(v)));
}

// v (16 fp32, in place) -> the softmax of its lanes (e^(v - v[0]) / sum; v[0] the largest; t: 16 fp32 of scratch)
__attribute__((noinline, minsize)) void {{ p }}_rsoft(float *__restrict v, float *__restrict t)
{
    aie::store_v(v, aie::sub(aie::load_v<16>(v), aie::broadcast<float, 16>(v[0])));
    {{ p }}_rexp(v);
    aie::store_v(t, aie::broadcast<float, 16>(aie::reduce_add(aie::load_v<16>(v))));
    {{ p }}_rrcp(t);
    aie::store_v(v, al::mulff<16>(aie::load_v<16>(v), aie::load_v<16>(t)));
}

// v (16 fp32, in place) -> sigmoid(v) = r or t r (v < 0), t = e^(-|v|), r = 1 / (1 + t) (t, r: 16 fp32 of scratch each)
__attribute__((noinline, minsize)) void {{ p }}_rsig(float *__restrict v, float *__restrict t, float *__restrict r)
{
    const aie::vector<float, 16> x = aie::load_v<16>(v);
    aie::store_v(t, aie::min(x, aie::sub(aie::zeros<float, 16>(), x)));
    {{ p }}_rexp(t);
    aie::store_v(r, aie::add(aie::broadcast<float, 16>(1.0f), aie::load_v<16>(t)));
    {{ p }}_rrcp(r);
    const aie::vector<float, 16> rv = aie::load_v<16>(r);
    aie::store_v(v, aie::select(rv, al::mulff<16>(aie::load_v<16>(t), rv), aie::lt(aie::load_v<16>(v), aie::zeros<float, 16>())));
}

__attribute__((noinline, minsize)) void {{ p }}_route(const int32_t *__restrict m, int32_t *__restrict ws, int col)
{
    int32_t *__restrict st = WS(int32_t, {{ w_.st }});
    // (scratch: the hold buffer, idle until the expert's first gate / up: the slots' top K ids and weights [slot][16],
    // the union, a vector's lanes, the items)
    int32_t *__restrict ids = WS(int32_t, {{ w_.hold }});
    float *__restrict wt = WS(float, {{ w_.hold }}) + 64;
    int32_t *__restrict U = WS(int32_t, {{ w_.hold }}) + 128;
    int32_t *__restrict lv = WS(int32_t, {{ w_.hold }}) + 192;  // (and 32 words of scratch)
    int32_t *__restrict it = WS(int32_t, {{ w_.hold }}) + 240;
    uint32_t used[{{ E // 32 }}];
    const int S = m[{{ M_NS }}];
    const int32_t *__restrict lg = m + {{ META }};
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ E // 32 }}; i++)
        used[i] = 0;
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < S; s++) {  // the slot's top K: insertion into a sorted list
        int32_t tk[{{ K }}];
        int32_t *__restrict ti = ids + {{ K }} * s;
        AIE_LOOP_NO_UNROLL
        for (int j = 0; j < {{ K }}; j++) {
            tk[j] = (int32_t)0x80000000;
            ti[j] = 0;
        }
        const int32_t *__restrict kl = lg + {{ E }} * s;
        AIE_LOOP_NO_UNROLL
        for (int e = 0; e < {{ E }}; e++) {
            const int32_t b = kl[e], k = b ^ ((b >> 31) & 0x7FFFFFFF);  // (fp32 order as int32 order)
            if (k > tk[{{ K - 1 }}]) {
                int j = {{ K - 1 }};
                AIE_LOOP_NO_UNROLL
                for (; j > 0 && k > tk[j - 1]; j--) {
                    tk[j] = tk[j - 1];
                    ti[j] = ti[j - 1];
                }
                tk[j] = k;
                ti[j] = e;
            }
        }
        int32_t *__restrict w = (int32_t *)(wt + 16 * s);  // (the K logits, then -1e30: weights below 2^-126 of the sum)
        AIE_LOOP_NO_UNROLL
        for (int j = 0; j < 16; j++)
            w[j] = j < {{ K }} ? kl[ti[j]] : (int32_t)0xF149F2CA;
        {{ p }}_rsoft(wt + 16 * s, (float *)lv);
        AIE_LOOP_NO_UNROLL
        for (int j = 0; j < {{ K }}; j++)
            used[ti[j] >> 5] |= 1u << (ti[j] & 31);
    }
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < 16; s++)  // the shared expert's gate per slot -> wsc [slot][8]
        lv[s] = s < {{ SLOTS }} ? st[{{ S_SG }} + s] : 0;
    {{ p }}_rsig((float *)lv, (float *)lv + 16, (float *)lv + 32);
    AIE_LOOP_NO_UNROLL
    for (int s = 0; s < {{ SLOTS }}; s++)
        aie::store_v(WS(float, {{ w_.wsc }}) + 8 * s, aie::broadcast<float, 8>(((const float *)lv)[s]));
    int u = 0;  // the union, by id
    AIE_LOOP_NO_UNROLL
    for (int e = 0; e < {{ E }}; e++)
        if ((used[e >> 5] >> (e & 31)) & 1)
            U[u++] = e;
    // column col's items (e | h0 << 8 | h1 << 9): whole experts U[col + COLS q], then the halves j = col, col + COLS
    // of the rest
    const int nw = u >> {{ (COLS - 1).bit_length() }}, rest = u - {{ COLS }} * nw;
    int n = 0;
    AIE_LOOP_NO_UNROLL
    for (int q = 0; q < nw; q++)
        it[n++] = U[col + {{ COLS }} * q] | 1 << 9;
    AIE_LOOP_NO_UNROLL
    for (int j = col; j < 2 * rest; j += {{ COLS }})
        it[n++] = U[{{ COLS }} * nw + (j >> 1)] | (j & 1) << 8 | (j & 1) << 9;
    // the expert object's metadata: its items with their weights per slot, its A objects
    int32_t *__restrict em = WS(int32_t, {{ w_.rbuf }}) + {{ RBUF_META }};
    {{ p }}_zv(em, {{ META // 16 }});
    em[{{ M_FLAGS }}] = {{ F_MAIN + F_EXPERT }};
    em[{{ M_NS }}] = S;
    em[{{ M_NITEM }}] = n;
    const int per = m[{{ M_RTGU }}] + m[{{ M_RTDN }}];
    int nt = 0;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i++) {
        const int e = it[i] & 255;
        nt += (((it[i] >> 9) & 1) - ((it[i] >> 8) & 1) + 1) * per;
        int32_t *__restrict ei = em + {{ M_ITEM }} + {{ ITEM_WORDS }} * i;
        ei[0] = it[i];
        AIE_LOOP_NO_UNROLL
        for (int s = 0; s < S; s++)
            AIE_LOOP_NO_UNROLL
            for (int j = 0; j < {{ K }}; j++)
                if (ids[{{ K }} * s + j] == e)
                    ei[1 + s] = ((const int32_t *)wt)[16 * s + j];
    }
    em[{{ M_NT }}] = nt;
    // the control packets: per item a whole BD (its halves' run, chained to the next item's), the push, the lock
    uint32_t *__restrict pk = (uint32_t *)WS(int32_t, {{ w_.rbuf }});
    const uint32_t blo = (uint32_t)m[{{ M_RBASE }}], bhi = (uint32_t)m[{{ M_RBASE + 1 }}], str = (uint32_t)m[{{ M_RSTR }}];
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ MAX_ITEMS }}; i++) {
        uint32_t w0 = 0, w1 = 0, w2 = 0, w4 = 0, w5 = 0, w7 = 0;
        if (i < n) {
            const int h0 = (it[i] >> 8) & 1, h1 = (it[i] >> 9) & 1;
            const uint32_t off = (uint32_t)(2 * (it[i] & 255) + h0) * str;
            w1 = blo + off;
            w2 = ((bhi + (w1 < off ? 1u : 0u)) & 0xFFFFu) | {{ bd_w2 }}u;
            w0 = (uint32_t)(h1 - h0 + 1) * (str >> 2);
            w4 = {{ BD_W4 }}u;
            w5 = {{ BD_W5 }}u;
            w7 = {{ BD_VALID }}u | (i + 1 < n ? {{ BD_USE_NEXT }}u | (uint32_t){{ p }}_ebd[i + 1] << 27 : 0u);
        }
        uint32_t *__restrict q = pk + 12 * i;
        q[0] = {{ sh }}u;
        q[1] = {{ p }}_ech[2 * i];
        q[2] = w0;
        q[3] = w1;
        q[4] = w2;
        q[5] = 0;
        q[6] = {{ sh }}u;
        q[7] = {{ p }}_ech[2 * i + 1];
        q[8] = w4;
        q[9] = w5;
        q[10] = 0;
        q[11] = w7;
    }
    pk[{{ 12 * MAX_ITEMS }}] = {{ sh }}u;
    pk[{{ 12 * MAX_ITEMS + 1 }}] = {{ ch_push }}u;
    pk[{{ 12 * MAX_ITEMS + 2 }}] = {{ bds[0] }}u;
    pk[{{ 12 * MAX_ITEMS + 3 }}] = {{ sh }}u;
    pk[{{ 12 * MAX_ITEMS + 4 }}] = {{ ch_lock }}u;
    pk[{{ 12 * MAX_ITEMS + 5 }}] = 1u;
    st[{{ S_NS }}] = 1;  // (rbuf's objects in order)
    st[{{ S_NE }}] = {{ RBUF_OBJECTS }};
    st[{{ S_ESRC }}] = 5;
    st[{{ S_EJ }}] = 0;
    st[{{ S_ESLOT }}] = 0;
}
""")


def render(p: str, consts: dict[str, object], *, experts: int, top: int, cols: int, a_packet: int | None = None) -> str:
    """The routing kernel ``{p}_route`` (in the helper's anonymous namespace; ``WS``, ``{p}_zv`` and the workspace
    regions ``w_`` from the overlay's kernel source; ``consts``: its M_ / F_ / S_ words, META, SLOTS; a_packet: the A
    stream's packet id where it is packet-switched: the items' BDs carry its header)."""
    assert experts % 32 == 0 and experts <= 256 and top <= 8 and cols == 8
    assert RBUF_META + consts["META"] <= 32 * RBUF_OBJECTS and PACKET_WORDS <= RBUF_META
    regs = [txn.SHIM_BD0 + txn.BD_BYTES * bd for bd in EXPERT_BDS]
    ch = [h for r in regs for h in (_ctl(r, 4), _ctl(r + 16, 4))]
    return _SRC.render(p=p, E=experts, K=top, COLS=cols, MAX_ITEMS=MAX_ITEMS, bds=EXPERT_BDS, ch=ch,
                       sh=_parity(CID), ch_push=_ctl(txn.QUEUE[txn.MM2S][0], 1),
                       ch_lock=_ctl(txn.SHIM_LOCK0 + 16 * LOCK_DONE, 1), RBUF_OBJECTS=RBUF_OBJECTS,
                       RBUF_META=RBUF_META, BD_W4=BD_W4, BD_W5=BD_W5, BD_VALID=BD_VALID, BD_USE_NEXT=BD_USE_NEXT,
                       bd_w2=0 if a_packet is None else 1 << 30 | a_packet << 19,
                       **consts)
