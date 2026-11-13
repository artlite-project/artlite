"""An MoE layer's routing on the decode overlay's cores (``decode.design``'s "MoE": the helper core of every column runs
it on the route object), model-independent: E experts, the top K of each token slot, softmax weights over the K
(llama.cpp's ``build_moe_ffn`` with softmax gating, the weights renormalized: the tests' ``moe/ref.py``).

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
from ..operators import templating
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
# arbiter -> North, the packet flows through the shim's switch on arbiters from 0 up, msels from 0 up; its control-packet
# overlay: arbiter 5): slot 1 (CID) -> arbiter 4, msel 3 -> master Tile_Ctrl, packet mode, the header kept (AM029 PL_MODULE
# Stream_Switch_*). A flow may share the arbiter (the shared A streams of a decode design's rounds take up to 5 arbiters in a
# switch they pass), not its msel: the arbiter sends a packet of msel m to every master that enables m.
SW_SOUTH7_SLOT1, SW_MASTER_TILE_CTRL, CTL_ARB, CTL_MSEL = 0x3F294, 0x3F000, 4, 3
SW_SLOTS = range(0x3F200, 0x3F400, 4)  # (the slave ports' slots)


def control_route(t: txn.Txn, col: int) -> None:
    """Route column col's control packets on its shim's MM2S1 to its TileControl (two writes; ``check_route``)."""
    t.write32(txn.tile_address(col, 0, SW_SOUTH7_SLOT1), _slot(CID, CTL_ARB) | CTL_MSEL << 4)
    t.write32(txn.tile_address(col, 0, SW_MASTER_TILE_CTRL), 1 << 31 | 1 << 30 | 1 << (3 + CTL_MSEL) | CTL_ARB)


def check_route(values: dict[int, int], cols: range) -> None:
    """The configuration (``paging.cdo_values``) leaves the control route's registers to ``control_route`` and no stream
    on its arbiter and msel: the words of such a stream would reach TileControl."""
    for c in cols:
        for reg in (SW_SOUTH7_SLOT1, SW_MASTER_TILE_CTRL):
            assert txn.tile_address(c, 0, reg) not in values, f"column {c}: the configuration writes {reg:#x}"
        for reg in SW_SLOTS:
            v = values.get(txn.tile_address(c, 0, reg), 0)
            assert not (v >> 8 & 1 and v & 7 == CTL_ARB and v >> 4 & 3 == CTL_MSEL), \
                f"column {c}: {reg:#x} = {v:#x} on the control route's arbiter and msel"


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


_SRC = templating.load("artlite/moe/route.cc.jinja")


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
