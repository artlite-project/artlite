"""The attention cores of the decode overlays on host-built transactions (Gemma 4: ``gemma4/decode.py``, Llama and
Qwen3: ``llama/decode.py``, LFM3: ``lfm/decode.py``): per step S <= 4 token slots at positions p .. p + S - 1, one or two
kinds of layer (Gemma's sliding-window head_dim 256 / full 512, Llama's one of 64 or 128, LFM3's sliding-window and full
of one head_dim 128, told apart by their rotated pairs: the F_ATT object's kind), RoPE on the dim pairs (i, i + hd / 2)
of i < rot (per kind), scores in log2 units: q k times a constant per layer (the F_ATT object's: Gemma q_norm k_norm
log2(e) with q and k RMS-normed per head, ``Attn(qk_norm=True)``; Llama log2(e) / sqrt(hd), Qwen3 too, with q and k
RMS-normed per head and times their weights per dim before RoPE, ``qk_weights``), v as it is or RMS-normed without a
weight (``v_norm``).

The KV cache of a (layer, KV head) is groups of objects of ``a_bytes`` (``decode.kv``): ``kp`` K objects (records of 8
positions: [K panels ND] per record, kp = 2: [hi panels ND | lo panels ND], kp = 1: one bfp16 block per 8 dims) and the
V object of their records ([record][ND panels]), linear in the positions (Gemma's sliding-window caches after a guard of
the window's groups: the prefill's attention reads a window from 16 groups before a position, ``gemma4.overlay.GUARD``).

A V block spans 8 positions: what position q sees of a record is its V panels encoded from the exact values (bf16) of
its positions up to q (all 8 when q is past it), so no position's V depends on later positions, on the step's other
slots or on an earlier, rolled-back step (rounding a record again whenever a position joins does all three). The open
records - those of the step's positions p .. p + S - 1 - need their exact values: each (layer, KV head) keeps a ring of
two slots next to the cache (record r in slot r & 1: its 8 positions' values [d-block][position 8][8] bf16,
``decode.vring``), and the chain's end gets the slots of records p >> 3 and (p + S - 1) >> 3 after its range (the tail
objects, ``tails``; after a prefill, which writes no values, the first one is the record's V panels instead, whose
values are exact: ``M_ATREC``); per slot it encodes the open record up to the slot's position and adds its P V there,
with the final running maxima.

The ``chain`` columns of a KV head (Gemma 4: columns 4 h .. 4 h + 3; Llama 1: column h) attend together, with all
query heads of the KV head (``M_ANH``: Gemma 4, its drafter 2, Llama 3 or 4; a set of 4 query rows holds row nh s + h
of slot s, head h). A layer's attention runs in sub-steps (``decode.vring.substeps``): the slots of one pair of
records (16 positions: P V adds records 2 k, 2 k + 1 of a group together), at most as many as L1 holds query rows for
(``Attn.rows``: 16 up to head_dim 256, 8 at 512); the columns' ranges of the cache's groups follow from the slots'
group alone (``decode.kv.ranges``: even shares, a sliding-window cache from the window's guard on). So a slot's
attention - which column reads which groups, which partials merge, which records add together in which order - is
the same in every step it can be part of, and its numbers too: a step of S slots gives each the logits of a step of
one. Softmax with a running maximum per row (masked positions add exactly nothing; a row's numbers depend on its own
scores only); at the end each column merges the partial (m, l, y) of its west neighbour in the chain (shared memory,
ordered by token ObjectFifos) into its own, so the chain's last column holds the whole and emits o = y / l on D into
the sub-step's slots of the O projection's packed input ([k block][slot SP][8] bf16, heads in a row:
``decode.protocol.packed_slots``). The last column holds the newest group: of a layer with its own cache it inserts the
sub-step's K (rotated, normed if ``qk_norm``; bfp16 blocks, kp per 8 dims) into their records as they pass and its V
rows into the open records' rows at the tail objects, and emits the K records of the sub-step's first and last
positions as the K objects come, then per open record its V panels and its ring slot (``decode.kv.emit_order``) on D
before o; the next sub-step reads them from DDR. D objects: ``d_len`` bf16 (min(256, head_dim) bytes: every record,
ring slot and o is a whole number of them).

LFM3's gate (``Attn(gate=)``): per head a sigmoid of the q object's gate logit times o, folded into l (l (1 + 2^(-g
log2 e)), the softmax's exp2). Its operator cores also run the conv layers' step between attention layers (``Attn(conv=)``,
``models.lfm.conv``), and their program is code pages (``Attn(pages=)``, ``page_entries``: the worker's kernels resident
stubs of three pages, the memtile transport of ``operators.paging``).

With one head dim (Llama) the softmax and P V run per octet (two sets' 8 rows): P of a group's records by kmath's
bias-192 exp2 (masked scores -16384; P 0 below m - 63.5 and past the sub-step's rows) as [P hi | P lo] panels of the
8 rows, and P V of a record pair as P hi and P lo two MMULs per record into one 8-row accumulator, each V panel popped
once for both octets (a tail object's record with its partner's P 0: the same roundings). Llama 3.2 1B at 16K
positions: steps of 1 / 2 / 3 / 4 tokens 20.2 / 21.6 / 25.2 / 26.4 ms; 3.1 8B 104.2 / 106.4 / 111.7 / 115.4.

Per step the core gets on X (broadcast with the main core's; it skips the main core's objects and the main core its):
the token objects (``F_TOK``, t = SLOTS kind + a per layer kind and slot a: of the sub-step from a on its first
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

from ..operators import paging, templating
from ..operators.design import COLS
from ..operators.util import Attr, bf16_bits, bf16_hilo, regions, rup
from ..operators.words import WordMap
from .kv import Records
from .protocol import M_MODEL, META, NX_PER_NORM, PHASE, SLOTS, TOKEN, XMETA
from .protocol import WINS as DESIGN_WINS

U = 3  # sub-steps of a layer kind per step at most (``substeps``: slots of 2 at head_dim 512)

# metadata words (the protocol's ``M_MODEL`` = 24 on): an F_TOK object's last cached position before the step (the
# masks: a slot s attends to positions <= M_POS + s); an F_ATT object's head_dim, kind (its index in the head dims), own
# cache (the layer computes K / V), the score constant (bf16 hi, lo bits), the window (0: none), the query heads per KV
# head
ATT = XMETA.kind("attention")
M_POS, M_AHD, M_AKIND, M_AOWN, M_QC_HI, M_QC_LO, M_AWIN, M_ANH = ATT.words(
    "M_POS M_AHD M_AKIND M_AOWN M_QC_HI M_QC_LO M_AWIN M_ANH", at=M_MODEL)
# the first tail object is the open record's V panels (the cache's last writer was the prefill), not its ring slot
M_ATREC = ATT.word("M_ATREC")
M_ATOK = ATT.word("M_ATOK")  # the token (its sub-step's: the protocol's M_NASKIP = -1 - M_ATOK)
F_ATT = ATT.flag("F_ATT", 17)  # M_FLAGS of a sub-step's first q object (the main cores skip it, the cache by its token)
# a token object's per-column first positions (M_TOK_G0 + c) and attended cache objects | its A objects << 16 (M_TOK_NA + c),
# next to the protocol's M_TOK_NG + c (the columns' cache A objects); its first slot of the step (its slots: a .. a + M_NS - 1)
TOKEN_ATT = XMETA.kind("attention token", extends=(TOKEN,))
M_TOK_NA = TOKEN_ATT.block("M_TOK_NA", COLS, at=32)
M_TOK_A = TOKEN_ATT.word("M_TOK_A", at=44)
M_TOK_G0 = TOKEN_ATT.block("M_TOK_G0", COLS, at=56)

# state words of the attention core (the first ones are trip counts of its program); A_TOK: per token t the column's range's
# first position, its cache objects, its attended ones, p << 8 | a << 4 | S (a token of no sub-step: S = 0, no cache objects;
# two kinds of layer at most)
A_TOK_AT = 40
A_WORDS = A_TOK_AT + 4 * 2 * SLOTS
STATE = WordMap("attention core state", A_WORDS)
(A_NNO, A_NXN, A_NXSKIP, A_NASKIP, A_NQ, A_NQP, A_NKV, A_NOBJ, A_NE, A_NO, A_NPS, A_XL, A_P, A_S, A_I, A_EIDX,
 A_HD, A_KIND, A_OWN, A_WIN, A_QC, A_NH, A_SA, A_POS, A_MERGE, A_CEND, A_INS, A_QS, A_KS, A_EOFF0, A_EN0, A_EOFF1,
 A_EN1, A_EREC) = STATE.common.words(
    "A_NNO A_NXN A_NXSKIP A_NASKIP A_NQ A_NQP A_NKV A_NOBJ A_NE A_NO A_NPS A_XL A_P A_S A_I A_EIDX A_HD A_KIND A_OWN A_WIN "
    "A_QC A_NH A_SA A_POS A_MERGE A_CEND A_INS A_QS A_KS A_EOFF0 A_EN0 A_EOFF1 A_EN1 A_EREC", at=0)
A_NHM = STATE.common.word("A_NHM")  # (query heads per KV head not a power of 2) 64 / nh rounded up: a row's slot (r nhm) >> 6 for r < 16
A_LPER = STATE.common.word("A_LPER")  # log2 of the strips of a main core phase's input object (``decode.protocol.packed_slots``)
A_NATT = STATE.common.word("A_NATT")  # the attended cache objects (the others after them are skipped)
A_YNZ = STATE.common.word("A_YNZ")  # y holds P V terms (else zero: a rise of the running maxima needs no rescale)
A_TREC = STATE.common.word("A_TREC")  # (M_ATREC)
A_TOK = STATE.common.block("A_TOK", 4 * 2 * SLOTS, at=A_TOK_AT)
# the event windows of a stats build (the decode operator's win=): the attention core's kernels
WINS = {"abegin": 11, "aq": 12, "aqpan": 13, "akv": 14, "kobj": 15, "vobj": 16, "arec": 17, "aend": 18, "aemit": 19,
        "put_k": 20, "tobj": 21, "scores": 22, "pv": 23,
        # (spans: an attention step from its region's first object to its end; the first cache object's wait from
        # there; the merge's wait for the west neighbour from the last cache object)
        "aspan": 40, "afirst": 41, "achain": 42}
assert not set(WINS.values()) & set(DESIGN_WINS.values()), "event windows of the design and of the attention core collide"


def consts() -> dict[str, int]:
    """The words, flags and ids the attention cores' kernels take: their own, and of the protocol the phase objects they
    skip (``PHASE``) and the token objects (``TOKEN``)."""
    return XMETA.consts(PHASE, ATT, TOKEN_ATT) | STATE.consts() | {"A_WORDS": A_WORDS}


_SRC = templating.load("artlite/decode/attn.cc.jinja")


class Attn:
    """The attention cores' shapes: the kinds of layer (at most two) by their rotated pairs ``rot`` and head dims ``hds``
    (one per kind, or one for both: the F_ATT object's M_AKIND tells a layer's kind), ``chain`` columns per KV head
    (COLS / chain KV heads), at most ``heads`` query heads per KV head (a layer's: the F_ATT object's M_ANH), ``kp`` K
    panels per record (2: hi + lo, 1: single), RMS norms of q / k (``qk_norm``: the weight folded into the score
    constant; ``qk_weights``: a weight per dim, Qwen3's, hd fp32 after the q heads of every q object and after [K | V] of
    every raw K / V object, one head dim) and of v (``v_norm``), a sigmoid output gate per head (``gate``: its logits in
    the q objects, ``goff``; the chain's end emits o = y sigmoid(g) / l), the cache objects of ``a_bytes`` (the A
    stream's). ``conv``: a second job of the cores between attention layers (LFM3's short convolution,
    ``models.lfm.conv.ConvJob``: its kernels, its state words after A_WORDS, its workspace over the attention's scratch,
    its X objects from one with its ``flag``, its outputs on D). ``pages``: the cores' code as pages (a program beyond 16
    KB: ``PAGES``), the memtile transport's words (``paging.PageSpec``: one core half)."""

    def __init__(self, *, hds: tuple[int, ...], rot: tuple[int, ...], chain: int, heads: int, a_bytes: int, kp: int = 2,
                 qk_norm: bool = False, qk_weights: bool = False, v_norm: bool = False, eps: float = 1e-6,
                 c_len: int = 64, gate: bool = False, conv: object = None, pages: paging.PageSpec | None = None):
        assert 1 <= len(hds) <= len(rot) <= 2 and all(hd % 64 == 0 and hd <= 512 for hd in hds), (hds, rot)
        assert all(r % 32 == 0 and r <= hds[min(i, len(hds) - 1)] // 2 for i, r in enumerate(rot)) and kp in (1, 2)
        assert chain in (1, 2, 4) and 1 <= heads <= 4, (chain, heads)
        assert not qk_weights or (qk_norm and not v_norm and len(hds) == 1), "per-dim weights: q / k norms, one head dim"
        assert not gate or (len(hds) == 1 and heads % chain == 0), "gates: one head dim, a chain column's heads alike"
        self.hds, self.rot, self.chain, self.heads, self.kp = tuple(hds), tuple(rot), chain, heads, kp
        self.qk_norm, self.v_norm, self.eps, self.a_bytes, self.c_len = qk_norm, v_norm, eps, a_bytes, c_len
        self.qk_weights, self.gate, self.conv, self.pages = qk_weights, gate, conv, pages
        self.kv_heads = COLS // chain
        self.d_len = min(256, min(hds)) // 2  # (bf16 of a D object: every record, ring slot and o in whole objects)
        self.kper = {hd: Records(hd, kp).kper(a_bytes) for hd in hds}  # K records per K object
        self.rg = {hd: Records(hd, kp).rg(a_bytes) for hd in hds}  # records per group
        self.pg = {hd: Records(hd, kp).pg(a_bytes) for hd in hds}  # positions per group
        assert all(k > 0 and self.rg[hd] % 2 == 0 for hd, k in self.kper.items()), self.kper

    @property
    def goff(self) -> int:
        """(``gate``) Where a q object's gate logits start (bf16 elements of its payload): after its heads (and the q
        norm's weights); chain column c's heads' at goff + 64 c (a C object each: the qkv phase's last of the column)."""
        return self.heads * self.hds[0] + (2 * self.hds[0] if self.qk_weights else 0)

    @staticmethod
    def rows(hd: int) -> int:
        """Query rows a column holds per sub-step (L1)."""
        return 16 if hd <= 256 else 8

    def slots(self, hd: int) -> int:
        """Slots of a sub-step at most (its query rows: ``rows``)."""
        return min(SLOTS, self.rows(hd) // self.heads)

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
        kb = max(self.kb_slot * SLOTS, 72 * hm // 8)  # (the slots' new K blocks; a record's V panels)
        words = A_WORDS + (self.conv.words if self.conv is not None else 0)
        sizes = [("st", 4 * words), ("cs", 4 * 2 * SLOTS * sum(self.rot)), ("y", 4 * self.y_floats()), ("qa", qa),
                 ("kf", 4 * hm), ("kb", kb), ("vn", 2 * hm * SLOTS), ("sc", 4 * 32 * rg * sets),
                 ("pp", self.ppo * sets // 2 if len(self.hds) == 1 else 72 * rg), ("m", 4 * 32 * sets)]
        if self.gate:
            sizes.append(("gt", 4 * 4 * sets))  # (1 + e^-g per query row)
        if self.pages is not None:
            sizes.append(("keep", 192))  # (the pager's slow path: the pages' constant data read into it)
        out = regions(sizes)
        assert 16 * hm <= qa and 2 * self.heads * hm <= qa  # (a record's rows; the first q object's heads)
        assert not self.gate or 2 * (self.goff + 64 * self.chain) <= qa  # (and its gates)
        assert self.conv is None or out["size"] - out["y"] >= self.conv.need, (out, self.conv.need)  # (the conv's)
        return out

    @property
    def ppo(self) -> int:
        """(one head dim) Bytes of an octet's P panels in pp: per record of a group [P hi | P lo] of its 8 rows."""
        return 144 * max(self.rg.values())

    @property
    def kb_slot(self) -> int:
        """Bytes of a slot's new K blocks (kp nd blocks of 9 B, mantissas and exponents apart; from a multiple of 64 B
        on: the mantissas' vector stores)."""
        return rup(9 * self.kp * max(self.hds) // 8)

    def render(self, p: str, win: str = "") -> str:
        nd_min = min(32, min(self.hds) // 8)
        hds = self.hds
        ws = self.workspace()
        conv = self.conv
        pg = self.pages
        X = "_x" if pg is not None else ""
        src = "" if conv is None else conv.render(p, base=ws["y"], st=ws["st"], X=X, consts=STATE.consts() | {"A_WORDS": A_WORDS})

        def by(var: str, vals: dict[int, object], key: object = lambda hd: hd) -> str:  # (a value per head dim)
            if len(hds) == 1:
                return str(vals[hds[0]])
            return f"{var} == {key(hds[0])} ? {vals[hds[0]]} : {vals[hds[1]]}"

        return src + _SRC.render(p=p, a=Attr(ws), eps=repr(self.eps), SMAX=SLOTS, ROT=self.rot, HDS=hds,
                           WINS=WINS, KPER=self.kper, RG=self.rg, RG_MAX=max(self.rg.values()), D_LEN=self.d_len,
                           D_SHIFT=int(math.log2(2 * self.d_len)), WIN=WINS.get(win, 0), Y_FLOATS=self.y_floats(),
                           KB_SLOT=self.kb_slot, KP=self.kp, DB=nd_min, ND_MIN=nd_min, ONE_HD=len(hds) == 1,
                           NO=max(self.rows(hd) for hd in hds) // 8, PPO=self.ppo,
                           ND=hds[0] // 8,
                           MINT="" if nd_min >= 32 else f"<{nd_min}>", VPUT="" if nd_min >= 32 else f"<0, {nd_min}>",
                           QK_NORM=self.qk_norm, QK_W=self.qk_weights, V_NORM=self.v_norm, CHAIN=self.chain,
                           QCOPY=self.heads * max(hds), GATE=self.gate, GOFF=self.goff if self.gate else 0,
                           GPER=self.heads // self.chain,
                           K0=f"hd == {hds[0]}" if len(hds) == 2 else f"st[{A_KIND}] == 0",
                           CONV=conv is not None, CONV_FLAG=conv.flag if conv else 0, X=X, PAGED=pg is not None, PG=pg,
                           PAGER=paging.source(p, wake=pg.wake, request=True) if pg is not None else "",
                           CONV_BEGIN=conv.begin_src(p) if conv else "", CONV_TOK=conv.token_src(A_WORDS) if conv else "",
                           NH_POW2=self.heads in (1, 2, 4), INV_HD={hd: f"1.0f / {hd}" for hd in hds},
                           by_hd=lambda vals: by("hd", vals), by_nd=lambda vals: by("nd", vals, lambda hd: hd // 8),
                           META=META, NX_PER_NORM=NX_PER_NORM, COLS=COLS, **consts())

    def workers(self, s: object, *, prefix: str, row: int, stack: int, a_depth: int = 1,
                extra: dict[int, list] | None = None) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        """The attention cores' workers on row ``row`` and their D streams (column, ObjectFifo) on the decode design's
        shared streams ``s`` (``decode.design._Streams``; kernel prefix s.p, object names from ``prefix``); ``a_depth``:
        their A objects in L1 (2: the next cache object arrives while the core works on one); ``extra``: per column
        buffers of the core that its kernels reach through their addresses (the pager's state words)."""
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
        conv = self.conv
        k_conv = [Kernel(f"{p}_cobj", "kernel.o", [x_ty, aws_ty]), Kernel(f"{p}_cemit", "kernel.o", [aws_ty, d_ty])] \
            if conv is not None else []
        if conv is not None:
            assert conv.d_bytes == 2 * self.d_len, (conv.d_bytes, self.d_len)

        def attn_fn(a_f, x_f, d_f, *rest):
            # tl / tr: the chain's tokens from the west neighbour / to the east one (columns chain h .. chain h + chain
            # - 1): a column produces tr around its attention step (its partial stays in its workspace until the east
            # one merged it), consumes tl around its end (the merge of the west partial, pw)
            rest = list(rest)
            col = rest.pop()
            del rest[len(rest) - len((extra or {}).get(col, [])):]
            j = col % chain
            tl = rest.pop(0) if j > 0 else None
            tr = rest.pop(0) if j < chain - 1 else None
            cobj, cemit = (rest.pop(10), rest.pop(10)) if conv is not None else (None, None)
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
                if conv is not None:
                    for _ in range_(ws[i_a + A_WORDS]):  # (a conv step: the region's other objects, then its D objects)
                        for _ in range_(ws[i_a + A_WORDS + 1]):
                            x = x_f.acquire(1)
                            cobj(x, ws)
                            x_f.release(1)
                        for _ in range_(ws[i_a + A_NE]):
                            d = d_f.acquire(1)
                            cemit(ws, d)
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
                                            *k_conv, aws[c], aws[c - 1] if j > 0 else aws[c],
                                            *(extra or {}).get(c, []), c],
                                  tile=Tile(c, row), stack_size=stack, dynamic_objfifo_lowering=True, while_true=False))
        return workers, d_fifos


def page_entries(p: str, conv: bool) -> dict[str, tuple[str, ...]]:
    """The attention cores' pages (``paging.link``'s ``pages``, page ids from 1 in this order): the query's preparation,
    the attention over the cache, the conv job's."""
    return {"prep": (f"{p}_pgprep",), "main": (f"{p}_pgmain",), **({"conv": (f"{p}_pgconv",)} if conv else {})}


def page_calls(p: str, conv: bool) -> dict[str, tuple[str, ...]]:
    """The resident stubs' calls of the pages' entries (``paging.link``'s ``calls``)."""
    out = {f"{p}_{k}": (f"{p}_pgprep",) for k in ("aq0", "aq", "aqpan", "akv")}
    out |= {f"{p}_{k}": (f"{p}_pgmain",) for k in ("aobj", "arec", "aend", "aemit")}
    return out | ({f"{p}_{k}": (f"{p}_pgconv",) for k in ("cobj", "cemit")} if conv else {})


# ----------------------------------------------------------------------
# Host side: the RoPE tables, the score constant
# ----------------------------------------------------------------------


def rope_tables(inv_freq: np.ndarray, positions: list[int] | range, rot: int) -> np.ndarray:
    """(SLOTS, 2 rot) fp32: per slot [cos | sin] of the rotated pairs' angles position inv_freq[i], i < rot (unused slots
    zero)."""
    out = np.zeros((SLOTS, 2 * rot), np.float32)
    for s, pos in enumerate(positions):
        th = pos * np.asarray(inv_freq)[:rot]
        out[s, :rot], out[s, rot:] = np.cos(th), np.sin(th)
    return out


def score_bits(c: float) -> tuple[int, int]:
    """The bf16 hi, lo bits of the score constant c log2(e) (scores q k c in log2 units)."""
    hi, lo = bf16_hilo(c / math.log(2))
    return bf16_bits(hi), bf16_bits(lo)
