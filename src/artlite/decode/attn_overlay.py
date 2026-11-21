"""The skeleton of the attention models' decode overlays (``AttentionOverlay``: Llama / Qwen3, Gemma 4, LFM3): a layer's
attention on the attention cores of the decode design (``decode.attn``), its caches and rings in KV, the token objects
and the step transaction's patches. A model supplies what differs (below); everything else is here.

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (per cache kind and
sub-step its positions and cache ranges, the RoPE tables) and the slots' residual slices (F_LOADR), then run the phases:
GEMV phases (``_gemv``: the input fills, ``_outputs``: the C drains) and operator phases of the attention cores
(``OPS``: ``_operator``; ``att``: ``_attention``) in the order of ``phases``. A column's weights between two operator
phases are one run of the (column-major) image: one A task (``_weight_run``).

The KV caches are described as data: ``kinds`` (a cache kind: head dim, positions per group, groups per KV head, a
sliding window's guard of groups before the first), ``uses`` (per attention layer its kind, the layer whose cache it
reads - its own, or a shared one - and the key of its open records in ``vring``), ``CHAIN`` (columns per KV head). They give
the offsets (``kv_base``, ``ring``), ``cache_rows``, ``read_kv``, the patch tables and the token objects' words. Per KV
head, a layer's cache is linear groups of ``G`` = kp + 1 objects of T bytes (``decode.kv``), a sliding-window cache's
after its guard; then per layer and KV head a ring of two slots of T bytes with the exact V values of the open records
(``decode.vring``).

A step is built once per pattern of sub-steps (and LM head or not: ``_build``) and patched per step (``_set_parts``):
the token objects, each column's range of each layer's cache per sub-step (A: address, length), a chain end's tail
objects (A: the ring slots of the sub-step's open records, or after a prefill the V panels of the first: record mode,
``_ring_modes``) and the record drains (D: address, length): ``decode.step``.

Several sequences in a step (``decode.step.Part``; a model with ``SEQ``): a sequence's slots are sub-steps of their
own (``_join``), each with its sequence's caches and rings (their buffers' addresses), its own positions (the token
object's, the slots' RoPE rows) and its sequence's record mode; the cores run a sub-step as in a step of that sequence
alone, so every slot's numbers are those of a step of one, and the step's time is the weights' stream once plus a pass
over each sequence's caches.

What a model supplies: its ``phases`` and their flags (``_flags``), the layout of its buffers (``_plan``: with
``_plan_caches`` and, for post-norm models, ``_plan_headers``), its GEMV outputs (``_outputs``), what the attention
objects hold (``_attention_scale``, ``_qk_weights``, ``_load_layer`` for other layers), the sub-steps (``subs``,
``_kind_subs``), the RoPE tables of a kind (``_rope``) and the words of its own in the token objects
(``_token_words``).
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from ..gguf_model import GGUFFile
from ..operators.design import COLS
from ..overlay import Sequence, Transaction, WeightBuffer
from . import kv
from .attn import (
    F_ATT,
    M_AHD,
    M_AKIND,
    M_ANH,
    M_AOWN,
    M_ATOK,
    M_ATREC,
    M_AWIN,
    M_POS,
    M_QC_HI,
    M_QC_LO,
    M_TOK_A,
    M_TOK_G0,
    M_TOK_NA,
    Attn,
    U,
)
from .model import CacheRows, InputRegion, ModelOverlay, Snapshot, host_bytes
from .protocol import (
    C_LEN,
    F_POST,
    F_TOK,
    F_VOUT,
    M_FLAGS,
    M_NASKIP,
    M_NS,
    M_SKIP,
    M_TOK_KIND,
    M_TOK_NG,
    META,
    SLOTS,
    XB,
    XPAY,
    packed_slots,
)
from .step import (
    Fill,
    FillTable,
    Part,
    PartTable,
    PatchTables,
    PositionWords,
    Rec,
    RecTable,
    StepPlan,
    Tail,
    TailTable,
    slots_of,
)
from .vring import PAIR, substeps, tails


@dataclass(frozen=True, kw_only=True)
class CacheKind:
    """One kind of KV cache of an overlay (e.g. a sliding-window one and a full one): ``hd`` the head dim of its layers,
    ``pg`` the positions of a group, ``groups`` the groups of a KV head's cache (a ``guard`` included: the groups before
    the first position's, which a sliding window's prefill attention reads from there on), ``window`` the attention
    window in positions (0: none), ``wrap`` the groups of a sliding-window cache kept as a ring (group g at slot g % wrap,
    its guard a copy of the slots before slot 0: ``slot``, ``fills``; 0: linear)."""

    hd: int
    pg: int
    groups: int
    guard: int = 0
    window: int = 0
    wrap: int = 0

    def slot(self, g: int) -> int:
        """The slot of group g (a ring: g % wrap)."""
        return g % self.wrap if self.wrap else g

    def at(self, off: int, gb: int) -> int:
        """A byte offset from group 0 of a KV head's linear cache (groups of gb bytes) -> its offset in this cache."""
        return (off // gb) % self.wrap * gb + off % gb if self.wrap else off

    def fills(self, g0: int, n: int, G: int) -> tuple[tuple[int, int], ...]:
        """The A fills (first object, objects) of a column's range of the cache (groups g0 .., n of them; G objects per
        group): one (``kv.fill``), or of a ring two, split where the range wraps, else before its last object (the
        transaction's tasks the same at every position); an empty range: two objects the core skips."""
        if not self.wrap:
            return (kv.fill(g0, n, G - 1),)
        if n == 0:
            return (0, 1), (0, 1)
        a = g0 % self.wrap
        assert n <= self.wrap, (n, self.wrap)
        if a + n <= self.wrap:
            return (G * a, G * n - 1), (G * (a + n) - 1, 1)
        return (G * a, G * (self.wrap - a)), (0, G * (n - self.wrap + a))


@dataclass(frozen=True, kw_only=True)
class CacheUse:
    """How a layer's attention uses a cache: ``kind`` (an index into ``kinds``), ``source`` the layer whose cache it reads
    (itself, or the cache of a layer before it), ``ring`` the key in ``vring`` of the open records its cache's ring holds."""

    kind: int
    source: int
    ring: int


class AttentionOverlay(ModelOverlay):
    """The decode overlay of an attention model (module docstring). ``kv_k``: the caches' K precision ("single": one bfp16
    block per 8 dims, "hilo": hi + lo blocks).

    A model's ``__init__`` sets, after ``super().__init__``: ``at`` (the attention cores' shapes), ``op``, ``kinds`` and
    ``uses``, ``q_heads`` (query heads per KV head), ``max_positions``, ``period``, then calls its ``_plan``."""

    CHAIN: ClassVar[int] = 1  # columns per KV head
    OPS: ClassVar[tuple[str, ...]] = ("att",)  # the phases of the operator cores (the others are GEMV phases)

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None, vocab: int | None, kv_k: str,
                 weights: WeightBuffer | None, image: object = None, sequences: int = 1):
        self.kp = kv.kv_precision(kv_k)
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights, image=image, sequences=sequences)
        self.G = self.kp + 1  # (cache objects per group)
        self.at: Attn
        self.kinds: tuple[CacheKind, ...]
        self.uses: dict[int, CacheUse]
        self.q_heads: int
        self.toks: tuple[int, ...] = ()  # (the token objects' offsets in ACT: per transaction buffer, ``AHEAD``)
        self.att_reg: dict[tuple[int, int], int] = {}  # (layer, KV head) -> its q and K / V objects' region in ACT
        self.kv_off: dict[int, int] = {}  # own-cache layer -> its KV heads' caches in KV
        self.ring_off: dict[int, int] = {}  # own-cache layer -> its KV heads' rings in KV
        self._trec: dict[tuple[int, int], int] = {}  # (M_ATREC as the F_ATT objects hold it, per layer and first slot)
        self.final = 0  # (a post-norm model: the LM header's updated residual in ACT)
        self.next_header: dict[tuple[int, str], tuple[int, str] | None] = {}
        self.v_to: dict[tuple[int, str], tuple[int, str]] = {}

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def kv_heads(self) -> int:
        return COLS // self.CHAIN

    @property
    def _tok_bytes(self) -> int:
        return len(self.kinds) * SLOTS * XB

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order (``lm`` False: without the LM head)."""
        raise NotImplementedError

    def _plan_caches(self, own: list[int], *, pad: int) -> None:
        """The caches of the own-cache layers ``own`` in KV, their rings after them, then ``pad`` bytes (the prefill's slices
        past the last layer's); ``rings``."""
        take, kvh = self._take, self.kv_heads
        self.kv_off = {L: take("KV", kvh * self.kv_bytes(L)) for L in own}
        r0 = self._layout["KV"]
        self.ring_off = {L: take("KV", kvh * 2 * self.T) for L in own}  # (per KV head: 2 slots)
        self.rings = (r0, self._layout["KV"] - r0)
        take("KV", pad)

    def _plan_headers(self) -> None:
        """A post-norm model's headers in order (``F_POST`` phases) and the ``F_VOUT`` phase that feeds each one."""
        gemv = [x for x in self.phases() if x[1] not in self.OPS]
        heads = [x for x in gemv if self._flags(*x) & F_POST]
        self.next_header = dict(itertools.pairwise(heads)) | {heads[-1]: None}
        self.v_to = {x: next(y for y in gemv[i + 1:] if self._flags(*y) & F_POST)
                     for i, x in enumerate(gemv) if self._flags(*x) & F_VOUT}

    @property
    def first_header(self) -> tuple[int, str]:
        """The step's first header (its r slots: x from the host)."""
        return next(x for x in self.phases() if x[1] not in self.OPS and self._flags(*x) & F_POST)

    def kv_bytes(self, L: int) -> int:
        """Bytes of a KV head's cache of own-cache layer L (a sliding-window cache: with its guard)."""
        return self.kinds[self.uses[L].kind].groups * self.G * self.T

    def kv_base(self, L: int, h: int) -> int:
        """Byte offset in KV of group 0 of KV head h's cache of own-cache layer L."""
        return self.kv_off[L] + h * self.kv_bytes(L) + self.kinds[self.uses[L].kind].guard * self.G * self.T

    def ring(self, L: int, h: int) -> int:
        """Byte offset in KV of KV head h's ring of own-cache layer L: 2 slots of T bytes, slot r & 1 the V values of record
        r ([d-block][position 8][8] bf16: ``decode.vring.ring_rows``)."""
        return self.ring_off[L] + 2 * h * self.T

    def cache_rows(self) -> list[CacheRows]:
        """(linear caches: a ring's groups are a snapshot's, ``snapshot``)"""
        out = []
        for k, kind in enumerate(self.kinds):
            if kind.wrap:
                continue
            offs = [self.kv_base(L, h) for L in self.kv_off if self.uses[L].kind == k for h in range(self.kv_heads)]
            if offs:
                out.append(CacheRows(np.array(offs, np.int64), self.G * self.T, kind.pg))
        return out

    def ranges(self, kind: int, p: int) -> list[tuple[int, int]]:
        """The chain's columns' (first group, groups) of a cache kind for a sub-step whose slots are in the group of position
        p: those of its group (a sliding-window cache: from the guard's groups before it on, the window of any of its
        positions), the same for each KV head in every step."""
        k = self.kinds[kind]
        g = p // k.pg
        return kv.ranges(first=max(0, g - k.guard) if k.guard else 0, g=g, chain=self.CHAIN)

    def v_offset(self, kind: int, p: int) -> int:
        """Byte offset from group 0 of a KV head's cache of a kind of the V panels of record p >> 3."""
        return kv.v_offset(p, hd=self.kinds[kind].hd, a_bytes=self.T, kp=self.kp)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The token objects' metadata words, the layers' objects'."""
        act = self.dev.view("ACT")
        self._trec.clear()
        for tok in self.toks:
            for t in range(len(self.kinds) * SLOTS):
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_TOK_KIND] = F_TOK, t
                act[tok + t * XB:][:4 * META] = m.view(np.uint8)
        for L in self.layers:
            self._load_layer(L)

    def _load_layer(self, L: int) -> None:
        """The static objects of layer L (the attention layers': ``_load_attention``)."""
        if L in self.uses:
            self._load_attention(L)

    def _attention_scale(self, L: int) -> tuple[int, int]:
        """The score constant of layer L's attention (bf16 hi, lo bits: ``decode.attn.score_bits``)."""
        raise NotImplementedError

    def _qk_weights(self, L: int) -> dict[str, np.ndarray]:
        """The q / k norms' weights of layer L (hd fp32 as bytes, "q" and "k"), after the payloads the qkv drains write:
        q_norm after a q object's heads, k_norm after a K / V object's [K | V]; none: {}."""
        return {}

    def _sinks(self, L: int, h: int) -> np.ndarray | None:
        """(``Attn.sinks``) The sinks of KV head h's query heads in layer L times log2 e (fp32), after a q object's heads
        and q_norm weights; None: no sinks."""
        return None

    def _load_attention(self, L: int) -> None:
        """The metadata words of layer L's KV heads' q and K / V objects: q object a's the F_ATT object of the sub-step from a
        on (static but M_ATREC: ``_tails``); a K / V object's (a layer with its own cache) M_NASKIP = -1 - its token t: after
        it the main cores skip the cache objects of the sub-step that ends at slot a. A layer without its own cache has no
        K / V objects: its q object's M_NASKIP is that."""
        act, use = self.dev.view("ACT"), self.uses[L]
        kind, own = self.kinds[use.kind], use.source == L
        hi, lo = self._attention_scale(L)
        w = self._qk_weights(L)
        for h in range(self.kv_heads):
            for a in range(SLOTS):
                t = SLOTS * use.kind + a  # (the token object of the sub-step from slot a on)
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_NASKIP], m[M_SKIP] = F_ATT, 0 if own else -1 - t, 0
                m[M_AHD], m[M_AKIND], m[M_AOWN] = kind.hd, use.kind, int(own)
                m[M_QC_HI], m[M_QC_LO], m[M_AWIN] = hi, lo, kind.window
                m[M_ANH], m[M_ATOK] = self.q_heads, t
                o = self.att_reg[(L, h)] + a * XB
                act[o:o + 4 * META] = m.view(np.uint8)
                if w:
                    act[o + 4 * META + 2 * self.q_heads * kind.hd:][:w["q"].size] = w["q"]
                sk = self._sinks(L, h)
                if sk is not None:
                    at = o + 4 * META + 2 * self.q_heads * kind.hd + (w["q"].size if w else 0)
                    act[at:at + 4 * sk.size] = np.asarray(sk, np.float32).view(np.uint8)
                if own:
                    k = np.zeros(META, np.int32)
                    k[M_NASKIP] = -1 - t
                    o = self.att_reg[(L, h)] + (SLOTS + a) * XB
                    act[o:o + 4 * META] = k.view(np.uint8)
                    if w:
                        act[o + 4 * META + 4 * kind.hd:][:w["k"].size] = w["k"]

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _post_to(self, seq: Sequence, c: int, off: int, e0: int, n: int, S: int, part: int) -> None:
        """n values per slot of column c (C objects (j, slot)) -> the v (part 0) or r (1) slots of elements e0 .. of a
        header's input at ACT offset off ([k block][v | r][slot SP][8])."""
        nobj, sp = n // C_LEN, packed_slots(S)
        seq.drain("C", c, self.dev.address("ACT", off + XB + XPAY + 4 * sp * e0 + 16 * sp * part), S * 2 * C_LEN,
                  (4, 8, 8 * sp, 4), nobj, (8 * sp * 8, nobj))

    def _header_to_r(self, seq: Sequence, st: StepPlan, phase: tuple[int, str], c: int) -> None:
        """A header's updated residual slice of column c (into the next header's r slots, or the LM header's ``final``)."""
        h8 = self.H // COLS
        nxt = self.next_header.get(phase)
        self._post_to(seq, c, self.inp[nxt][0].off if nxt is not None else self.final, h8 * c, h8, st.S, 1)

    def _vout(self, seq: Sequence, st: StepPlan, phase: tuple[int, str], c: int) -> None:
        """An F_VOUT phase's outputs of column c: its rows v into the v slots of the next header's input, then its
        partial sums of squares."""
        h8, tgt = self.H // COLS, self.inp[self.v_to[phase]][0].off
        self._post_to(seq, c, tgt, h8 * c, h8, st.S, 0)
        seq.drain("C", c, self.dev.address("ACT", tgt + XB + 2 * C_LEN * c), 2 * C_LEN)

    def _outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """The C drains of a GEMV phase (after its last group)."""
        raise NotImplementedError

    def _gemv(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        for gi, rg in enumerate(self.inp[phase]):
            if gi == 0:
                seq.sync()  # (its inputs' producers)
            self._inputs(seq, rg, st.S)
        self._outputs(seq, st, phase)

    def _operator(self, seq: Sequence, st: StepPlan, L: int, name: str, weights: list[tuple[int, str]]) -> None:
        """An operator phase of layer L (``OPS``); ``weights``: the GEMV phases after it up to the next operator (one A
        run)."""
        assert name == "att", name
        self._attention(seq, st, L, weights)

    def _kind_subs(self, subs: tuple, kind: int) -> tuple[tuple[int, int], ...]:
        """The sub-steps (first slot, slots) of a cache kind in a step's ``subs``."""
        return subs

    def _attention(self, seq: Sequence, st: StepPlan, L: int, weights: list[tuple[int, str]]) -> None:
        """The attention of layer L, per sub-step (a layer with its own cache: each after the last one's records are in
        DDR): per column its cache range (A, patched) and a chain end's tail objects, the F_ATT object and the sub-step's q
        and K / V objects (X), the records (D, patched) and o (D, into the sub-step's slots of o's input). The weights of
        the GEMV phases after it go before the sync of its inputs (their first layout block: ``_weight_run``): the A stream
        keeps streaming them while the attention waits for them."""
        dev, T, chain = self.dev, self.T, self.CHAIN
        use = self.uses[L]
        kind, src = self.kinds[use.kind], use.source
        own = src == L
        subs = self._kind_subs(st.subs, use.kind)
        sp, nq = packed_slots(st.S), self.q_heads * kind.hd  # (nq: o's elements of a KV head)
        before = self.a_bds > 3  # (a run's first block before the X objects: ``_weight_run``)
        for u, (a, n) in enumerate(subs):
            if u and own:
                seq.sync()  # (sub-step u - 1's records in DDR: this one reads them)
            for c in range(COLS):  # (its range of the cache and a chain end's tail objects, patched per step)
                st.fills += [Fill(seq.fill("A", c, dev.address("KV", self.kv_off[src]), T), L, c, u, f)
                             for f in range(2 if kind.wrap else 1)]
                if c % chain == chain - 1:  # (one object each: the slots of its records, ``_tails``)
                    st.tails += [Tail(seq.fill("A", c, dev.address("KV", self.ring_off[src]), T), L, c, u, t)
                                 for t in range(tails(n))]
            last = u == len(subs) - 1 and weights
            if last and before:
                self._weight_run(seq, weights, blocks=slice(1))  # (a later block's after the X objects: ``_weight_run``)
            if u == 0:
                seq.sync()  # (the q and K / V objects in DDR)
            for c in range(COLS):
                reg = self.att_reg[(L, c // chain)] + a * XB
                if own:  # (its q objects, then its K / V objects: half objects, the second strip SLOTS XB on)
                    seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
                else:
                    seq.fill("X", c, dev.address("ACT", reg), n * XB)
            if last:
                self._weight_run(seq, weights, blocks=slice(1, None) if before else slice(None))
            for c in range(chain - 1, COLS, chain):
                h = c // chain
                if own:
                    for i in range(len(kv.emit_order(p=0, S=n, hd=kind.hd, a_bytes=T, kp=self.kp))):
                        st.recs.append(Rec(seq.drain("D", c, dev.address("KV", self.kv_off[src]), 2 * self.at.d_len),
                                           L, h, u, i))
                o = next(rg for rg in self._o_inputs(L) if rg.k0 <= nq * h < rg.k0 + rg.K)  # (its K group)
                at = dev.address("ACT", o.off + XB + 2 * sp * (nq * h - o.k0) + 16 * a)  # ([k block][slot sp][8])
                seq.drain("D", c, at, 2 * n * nq, None if n == sp else (4 * n, nq // 8, 4 * sp, 4 * sp * nq // 8))

    def _o_inputs(self, L: int) -> list[InputRegion]:
        """The input regions the attention of layer L writes o into (o's)."""
        return self.inp[(L, "o")]

    def _token_at(self, b: int) -> int:
        """The token objects' offset in ACT of the transaction buffer ``b``."""
        return self.toks[b]

    def _weights_first(self, seq: Sequence, ph: list[tuple[int, str]], i: int,
                       run: Callable[[int], list[tuple[int, str]]]) -> None:
        """Before the GEMV phase i of a step's ``ph``: the weight run that starts at it, if it is one's start (the first
        phase's; ``run(i)``: the GEMV phases from i to the next operator, one A fill)."""
        if i == 0:
            self._weight_run(seq, run(0))

    def _begin(self, seq: Sequence) -> None:
        """A step's first words (a model's: an MoE model's control route, ``moe.steps``)."""

    def _build(self, S: int, lm: bool, subs: tuple, b: int) -> StepPlan:
        dev = self.dev
        seq = Sequence(self.op, inputs=("X",))
        self._begin(seq)
        st = StepPlan(S, subs, tok=self._token_at(b))
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", st.tok), self._tok_bytes)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * SLOTS * XB), S * XB)
        ph = self.phases(lm)

        def run(i: int) -> list[tuple[int, str]]:  # (the GEMV phases from i to the next operator phase: one A fill)
            return ph[i:next((k for k in range(i, len(ph)) if ph[k][1] in self.OPS), len(ph))]

        for i, (L, name) in enumerate(ph):
            if name in self.OPS:
                self._operator(seq, st, L, name, run(i + 1))
                continue
            self._weights_first(seq, ph, i, run)
            self._gemv(seq, st, (L, name))
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        return st

    def _lm_phase(self, seq: Sequence) -> None:
        self._weight_run(seq, [(-1, "lm")])
        self._gemv(seq, StepPlan(1, ()), (-1, "lm"))

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def _substeps(self, p: int, S: int, hd: int) -> tuple[tuple[int, int], ...]:
        """The sub-steps (first slot, slots) of the attention of head dim ``hd`` in a step at p .. p + S - 1: within a pair of
        records (``decode.attn``: P V adds two records together; a group holds whole pairs)."""
        out = tuple(substeps(p, S, PAIR, self.at.slots(hd)))
        assert len(out) <= U, (p, S)
        return out

    def _join(self, parts: tuple[Part, ...]) -> tuple:
        """(a model whose ``subs`` are every kind's: each part's from its first slot on, a sequence per sub-step)"""
        out, a0 = [], 0
        for q in parts:
            out += [(a0 + a, n) for a, n in self.subs(q.position, q.n)]
            a0 += q.n
        return tuple(out)

    def _rope(self, kind: int, positions: list[int]) -> np.ndarray:
        """The RoPE tables ((SLOTS, 2 rot) fp32, ``decode.attn.rope_tables``) of a cache kind's layers for the slots'
        ``positions``."""
        raise NotImplementedError

    def _token_words(self, st: StepPlan, parts: tuple[Part, ...]) -> None:
        """The words of the model's own in the token objects of a step of ``parts`` (before they are flushed)."""

    def _token_objects(self, st: StepPlan, parts: tuple[Part, ...]) -> PositionWords:
        """The step's token objects (per cache kind and sub-step: its positions, the columns' ranges, the step's RoPE
        tables; each slot's object copies the table, one of no sub-step has no objects)."""
        chain, act = self.CHAIN, self.dev.view("ACT")
        seq, pos, head = slots_of(parts)
        K = len(self.kinds)
        pu = np.zeros((K, SLOTS, 2), np.int64)
        fo = np.zeros((K, SLOTS, chain, 2, 2), np.int64)
        ro = np.zeros((K, SLOTS, 6, 3), np.int64)
        su = np.zeros((K, SLOTS), np.int64)
        first = np.zeros((K, SLOTS), bool)
        for k, kind in enumerate(self.kinds):
            cs = self._rope(k, pos).view(np.uint8).reshape(-1)
            for a in range(SLOTS):
                o = st.tok + (SLOTS * k + a) * XB
                act[o + 4 * META:o + 4 * META + cs.size] = cs
                m = act[o:o + 4 * META].view(np.int32)
                m[M_NS], m[M_TOK_NG:M_TOK_NG + COLS], m[M_TOK_NA:M_TOK_NA + COLS] = 0, 0, 0
            for u, (a, n) in enumerate(self._kind_subs(st.subs, k)):
                p = pos[a]  # (the sub-step's first position: its slots are one sequence's, consecutive)
                r = self.ranges(k, p)
                o = st.tok + (SLOTS * k + a) * XB
                m = act[o:o + 4 * META].view(np.int32)
                me = act[o + (n - 1) * XB:][:4 * META].view(np.int32)  # (the token of its last slot)
                m[M_NS], m[M_POS], m[M_TOK_A] = n, p, a
                fills = [kind.fills(g0, ng, self.G) for g0, ng in r]
                for c in range(COLS):  # (every KV head's columns: the same ranges)
                    g0, ng = r[c % chain]
                    no = sum(x[1] for x in fills[c % chain]) + (tails(n) if c % chain == chain - 1 else 0)
                    m[M_TOK_G0 + c], m[M_TOK_NA + c], me[M_TOK_NG + c] = g0 * kind.pg, self.G * ng | no << 16, no
                pu[k, u], su[k, u], first[k, u] = (p, n), seq[a], head[a]
                for j, fj in enumerate(fills):
                    fo[k, u, j, :len(fj)] = fj
                gb = self.G * self.T
                rec = [(b, kind.at(off, gb) if b == 0 else off, nb)
                       for b, off, nb in kv.records(p=p, S=n, hd=kind.hd, a_bytes=self.T, kp=self.kp)]
                ro[k, u, :len(rec)] = rec
        return PositionWords(pu, fo, ro, su, first)

    def _patch_tables(self, st: StepPlan) -> PatchTables:
        """A step's cache fills, tail fills and record drains as arrays."""
        uses, chain = self.uses, self.CHAIN
        f = np.array([(x.at, uses[x.layer].kind, x.sub, x.column % chain, x.piece,
                       self.kv_base(uses[x.layer].source, x.column // chain)) for x in st.fills],
                     np.int64).reshape(-1, 6)
        layers = tuple(dict.fromkeys(x.layer for x in st.tails))
        t = np.array([(x.at, uses[x.layer].kind, x.sub, x.obj, self.kv_base(uses[x.layer].source, x.column // chain),
                       self.ring(uses[x.layer].source, x.column // chain), uses[x.layer].source == x.layer,
                       layers.index(x.layer)) for x in st.tails], np.int64).reshape(-1, 8)
        r = np.array([(x.handle[0], uses[x.layer].kind, x.sub, x.i, self.kv_base(x.layer, x.head),
                       self.ring(x.layer, x.head)) for x in st.recs], np.int64).reshape(-1, 6)
        b = np.array(st.part_bds, np.int64).reshape(-1, 4)
        return PatchTables(FillTable(f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5]),
                           TailTable(t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5], t[:, 6].astype(bool),
                                     t[:, 7], layers),
                           RecTable(r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4:6]),
                           PartTable(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))

    def tail_mode(self, src: int, *, before: bool, seq: int | None = None) -> bool:
        """Whether a step's tail objects of a layer attending to the cache of ``src`` are the open record's V panels (the
        prefill wrote it, and the step did not run ``src``'s attention before the layer's: ``before`` False), not its ring
        slots; of sequence ``seq`` (None: the selected one)."""
        vring = self.vring if seq is None else self._vrings[seq]
        return self.uses[src].ring in vring.from_prefill and not before

    def _ring_modes(self, seq: int | None = None) -> dict[int, tuple[int, bool]]:
        """Per attention layer the step runs (from ``start`` on): (the mode of its tail objects, ``tail_mode``; whether the
        step runs the layer's cache's before it: a layer that shares a cache of a layer after ``start``); of sequence
        ``seq`` (None: the selected one)."""
        out = {}
        for L, use in self.uses.items():
            if L >= self.start:
                before = use.source != L and use.source >= self.start
                out[L] = (int(self.tail_mode(use.source, before=before, seq=seq)), before)
        return out

    def _tails(self, st: StepPlan, parts: tuple[Part, ...], words: PositionWords, kv0: np.ndarray, ahead: bool) -> bool:
        """The step's tail fills (per sub-step a chain end's objects: the ring slots of the records of its first and last
        positions, or after a prefill for a sequence's first sub-step's first the record's V panels: of a layer with its
        own cache, the others from the ring) and the F_ATT objects' M_ATREC. ``words``: the token objects' (per cache
        kind and sub-step its first position, slots and sequence); ``kv0``: per sequence the address of its KV.
        ``ahead``: False when M_ATREC changes (the running step reads those objects)."""
        tt, T, pu = st.patch.tails, self.T, words.pu
        seq, _, head = slots_of(parts)
        modes = {q.seq: self._ring_modes(q.seq) for q in parts}
        seq += [seq[-1]] * (SLOTS - len(seq))  # (a slot past the step's: as the last part's)
        head += [False] * (SLOTS - len(head))
        wanted = [(L, a, m if head[a] or self.uses[L].source != L else 0)
                  for a in range(SLOTS) for L, (m, _) in modes[seq[a]].items()]
        changes = [(L, a, ma) for L, a, ma in wanted if self._trec.get((L, a), 0) != ma]
        if ahead and changes:
            return False
        mode = np.zeros((self.sequences, len(tt.layers)), bool)
        for s, md in modes.items():
            mode[s] = [md[L][0] for L in tt.layers]
        own = words.su[tt.kind, tt.sub]  # (the sub-step's sequence)
        rec = mode[own, tt.layer] & (tt.obj == 0) & (words.first[tt.kind, tt.sub] | ~tt.own)
        q0, n = pu[tt.kind, tt.sub, 0], pu[tt.kind, tt.sub, 1]
        vrec = np.array([[self.kinds[k].at(self.v_offset(k, int(x)), self.G * T) for x in row]
                         for k, row in enumerate(pu[:, :, 0])], np.int64)[tt.kind, tt.sub]
        slot = np.where(tt.obj == 0, q0 >> 3, (q0 + n - 1) >> 3) & 1
        address = kv0[own] + np.where(rec, tt.base + vrec, tt.ring + slot * T)
        self._bd_words(st.tr.words, tt.at, address, np.full(len(rec), T, np.int64))
        for q in parts:
            for L, (m, before) in modes[q.seq].items():
                if not m and not before:  # (from the ring: the rows before p, or those of the step's positions too)
                    use = self.uses[L]
                    self._vrings[q.seq].check(use.ring, q.position, q.position if use.source == L else q.position + q.n)
        act = self.dev.view("ACT")
        for L, a, ma in changes:  # (the F_ATT objects of the sub-steps from slot a on)
            for h in range(self.kv_heads):
                off = self.att_reg[(L, h)] + a * XB
                act[off:off + 4 * META].view(np.int32)[M_ATREC] = ma
                self.dev.flush("ACT", off, 4 * META)
            self._trec[(L, a)] = ma
        return True

    def _ran(self, p: int, S: int, seq: int | None = None) -> None:
        """After a step: the rings of the own-cache layers it ran hold the rows of its open records."""
        vring = self.vring if seq is None else self._vrings[seq]
        for key in {use.ring for L, use in self.uses.items() if L >= self.start and use.source == L}:
            vring.ran(key, p, S)

    def _set_position(self, st: StepPlan, p: int, ahead: bool = False) -> bool:
        """(the selected sequence's S slots at p)"""
        assert st.subs == self.subs(p, st.S), (p, st.S, st.subs)
        return self._set_parts(st, (Part(self.seq, p, st.S),), ahead)

    def _set_parts(self, st: StepPlan, parts: tuple[Part, ...], ahead: bool = False) -> bool:
        """The token objects (per cache kind and sub-step: its positions, the columns' ranges, the step's RoPE tables; st's
        copy) and the step's cache transfers, each sub-step's in its sequence's KV; a part's transfers of its sequence's
        own objects (``StepPlan.part_bds``) in its KV."""
        dev, T = self.dev, self.T
        assert all(0 <= q.position and q.position + q.n <= self.max_positions for q in parts), parts
        assert len(parts) == 1 or st.subs == self._join(parts), (parts, st.subs)
        words = self._token_objects(st, parts)
        self._token_words(st, parts)
        dev.flush("ACT", st.tok, self._tok_bytes)
        if st.patch is None:
            st.patch = self._patch_tables(st)
        t, kv0 = st.patch, np.zeros(self.sequences, np.int64)
        for q in parts:
            kv0[q.seq] = dev.address(self.seq_name("KV", q.seq), own=True)
        f = words.fo[t.fills.kind, t.fills.sub, t.fills.place, t.fills.piece]  # (first object, objects)
        self._bd_words(st.tr.words, t.fills.at, kv0[words.su[t.fills.kind, t.fills.sub]] + t.fills.base + f[:, 0] * T,
                       f[:, 1] * T)
        if t.parts.at.size:
            seqs = np.array([q.seq for q in parts], np.int64)
            self._bd_words(st.tr.words, t.parts.at, kv0[seqs[t.parts.part]] + t.parts.offset, t.parts.nbytes)
        if not self._tails(st, parts, words, kv0, ahead):
            return False
        # (the record drains: from the KV head's group 0 or ring, offset, bytes)
        r = words.ro[t.recs.kind, t.recs.sub, t.recs.i]
        base = np.take_along_axis(t.recs.base, r[:, :1], 1)[:, 0]
        self._bd_words(st.tr.words, t.recs.at, kv0[words.su[t.recs.kind, t.recs.sub]] + base + r[:, 1], r[:, 2])
        st.tr.flush()
        return True

    # ------------------------------------------------------------------
    # Readbacks
    # ------------------------------------------------------------------

    def read_kv(self, layer: int, positions: list[int] | range) -> tuple[np.ndarray, np.ndarray]:
        """K, V (kv heads, n, head_dim) fp32 as cached for the positions of an own-cache layer, as the cores store them (K
        rotated, normed as the model's attention norms it)."""
        kind = self.kinds[self.uses[layer].kind]
        pos = np.asarray(positions)
        G, T = self.G, self.T
        g0, g1 = int(pos.min()) // kind.pg, int(pos.max()) // kind.pg + 1  # (the groups that hold the positions)
        assert not kind.wrap or g1 - g0 <= kind.wrap, "positions a ring still holds"
        slots = np.array([kind.slot(g) for g in range(g0, g1)])
        self.dev.fetch("KV", self.kv_off[layer], self.kv_heads * self.kv_bytes(layer))
        ks, vs = [], []
        for h in range(self.kv_heads):
            b = self.kv_base(layer, h)
            groups = self.dev.view("KV")[b:b + (int(slots.max()) + 1) * G * T].reshape(-1, G, T)[slots]
            k, v = kv.kv_from_groups(groups, kind.hd)
            ks.append(k[pos - g0 * kind.pg])
            vs.append(v[pos - g0 * kind.pg])
        return np.stack(ks), np.stack(vs)

    # ------------------------------------------------------------------
    # Rings (``CacheKind.wrap``)
    # ------------------------------------------------------------------

    @property
    def wraps(self) -> bool:
        """Whether a cache is a ring (the sequence then goes back only to a snapshot: ``snapshot``)."""
        return any(k.wrap for k in self.kinds)

    def _ring_groups(self, position: int) -> list[tuple[int, CacheKind, int, int]]:
        """Per own-cache layer of a ring: (layer, kind, first group, end group) of the groups that the windows of
        ``position`` and later read (its group and the guard's before it)."""
        out = []
        for L in self.kv_off:
            kind = self.kinds[self.uses[L].kind]
            if kind.wrap:
                g1 = position // kind.pg + 1
                out.append((L, kind, max(0, g1 - 1 - kind.guard), g1))
        return out

    def snapshot(self, position: int) -> Snapshot:
        """(and of each ring the groups its windows read from ``position`` on: ``extra["rings"]``)"""
        snap = super().snapshot(position)
        gb, kvv, parts = self.G * self.T, self.dev.view("KV"), []
        for L, kind, g0, g1 in self._ring_groups(position):
            a = host_bytes((self.kv_heads, g1 - g0, gb))
            for h in range(self.kv_heads):
                for i, g in enumerate(range(g0, g1)):
                    o = self.kv_base(L, h) + kind.slot(g) * gb
                    self.dev.fetch("KV", o, gb)
                    a[h, i] = kvv[o:o + gb]
            parts.append((L, g0, a))
        if parts:
            snap.extra["rings"] = parts
        return snap

    def restore(self, snapshot: Snapshot) -> None:
        """(the rings' groups too)"""
        super().restore(snapshot)
        gb, kvv = self.G * self.T, self.dev.view("KV")
        for L, g0, a in snapshot.extra.get("rings", ()):
            kind = self.kinds[self.uses[L].kind]
            for h in range(self.kv_heads):
                for i in range(a.shape[1]):
                    o = self.kv_base(L, h) + kind.slot(g0 + i) * gb
                    kvv[o:o + gb] = a[h, i]
                    self.dev.flush("KV", o, gb)

    def mirror(self, p0: int) -> None:
        """Before a prefill chunk at p0 (a group start) whose window reaches back over a ring's wrap (its first group in
        the ring's first ``guard`` slots): each ring's guard as a copy of its last slots, so that the chunk's attention
        reads the window's history before slot 0 in one run (``CacheKind.wrap``)."""
        gb, kvv = self.G * self.T, self.dev.view("KV")
        for L in self.kv_off:
            kind = self.kinds[self.uses[L].kind]
            g0 = p0 // kind.pg
            if not kind.wrap or g0 < kind.wrap or g0 % kind.wrap >= kind.guard:
                continue
            n = kind.guard * gb
            for h in range(self.kv_heads):
                b = self.kv_base(L, h)
                self.dev.fetch("KV", b + kind.wrap * gb - n, n)
                kvv[b - n:b] = kvv[b + kind.wrap * gb - n:b + kind.wrap * gb]
                self.dev.flush("KV", b - n, n)

    def residual(self, slot: int = 0) -> np.ndarray:
        """The last step's residual stream after the last layer (the LM header's update), bf16 values as fp32."""
        off, sp = self.final + XB + XPAY, packed_slots(self._last_s)
        self.dev.fetch("ACT", off, 4 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 4 * sp * self.H].view(np.uint16).reshape(-1, 2, sp, 8)[:, 1, slot]
        return np.ascontiguousarray(v.reshape(-1)).view(bfloat16).astype(np.float32)

    def residuals(self, S: int) -> np.ndarray:
        """``residual`` of the last step's first S slots (S, hidden): copies."""
        off, sp = self.final + XB + XPAY, packed_slots(self._last_s)
        self.dev.fetch("ACT", off, 4 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 4 * sp * self.H].view(np.uint16).reshape(-1, 2, sp, 8)[:, 1, :S]
        return np.ascontiguousarray(v.transpose(1, 0, 2)).reshape(S, self.H).view(bfloat16).astype(np.float32)


__all__ = ["AttentionOverlay", "CacheKind", "CacheUse"]
