"""A step's attention layers (``attn256``) on the decode overlay: the transfers of attn_in and the attention per
sub-step (``AttentionSteps.build``) and what the step's position patches into them (``set_position``).

The position patches three things, each a named part: the F_ATT objects (``att_words``: per sub-step and column the words of
the attention core, pure arithmetic on the attention's shape; their angles), the cache and ring reads (``patch_fills``, ``patch_tails``:
the BDs of ``AttnStep.fills`` and ``AttnStep.tails``) and the record drains (``patch_drains``). A sub-step's records are 8
positions; four make a group (a K and a V object) of the cache, a ring of two slots holds the open records' V rows
(``decode.vring``). Each record is written where the transfer is queued and read where the position patches it, by name.

A windowed spec (``attn256.AttnSpec.window``: Spark-X's) has two layer kinds (``WindowOverlay.layer_kind``): a full
layer's caches are linear, a sliding-window layer's a ring of NR groups (``WindowOverlay.ring_groups``: group g at slot
g % NR), whose window's groups (from ``attn256.window_words``' first group) a step reads in two fills, "ra" and "rb":
split where they wrap the ring, else before the last object (the transaction's tasks are the same at every position).
Each kind's F_ATT objects take their words (a sliding-window layer's window and masks) and their angles (``cs`` per kind)
per step; the step's masks and K lanes too (``attn256.step_words``: no branches).

Several sequences in a step (one pass; ``decode.step.Part``, ``set_parts``): a pass per part, in order - an F_ATT object
of its own (``AttnShape.u`` of them per layer and column), whose words are those of a step of the part's slots alone at
its position, the part's q and raw K, V objects (from its first slot on), its sequence's cache, ring and records (the
patches take each pass's KV buffer) -, its o's octets drained into the staged x of the output phase from the pass's
first slot on (``attn256``'s "Several passes"; an overlay of several sequences stages that x for every step). A step of
one part runs as a step of that sequence. A step's transaction does not follow how its slots are dealt: the transfers
that do (``Span``: a pass's q objects, raw K, V objects and o) are patched with each pass's first slot and slots, so
there is one transaction per number of passes (``pass_subs``).
"""

import dataclasses
import time
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import numpy as np
from ml_dtypes import bfloat16

from ..operators.bfp16 import block_values
from ..operators.design import COLS
from ..overlay import Sequence, Transaction
from . import attn256 as attn
from .attn256 import M_ANA, M_G0, M_GATT, M_GJ, M_NTAIL, M_POS, M_SA, M_SPLIT
from .protocol import AB_ID, C_LEN, M_NASKIP, M_NREL, M_NS, M_SKIP, META, XB
from .step import Part
from .vring import VRing

# a step's cache fills: the whole cache, the sub-step's group, a split column's run; a ring's (a windowed spec's
# sliding-window layers: the window's groups in two fills, split where they wrap the ring, else before the last object)
FILLS = ("all", "new", "run", "ra", "rb")


class Fill(NamedTuple):
    """A cache fill: its BD word index, the layer, the column, the sub-step and which groups (``FILLS``)."""

    at: int
    layer: int
    column: int
    sub: int
    kind: str


class Tail(NamedTuple):
    """A tail fill (a ring's open record): its BD word index, the layer, the column, the sub-step, the tail object."""

    at: int
    layer: int
    column: int
    sub: int
    tail: int


class Drain(NamedTuple):
    """A record drain: its BD handle, the layer, the column, the sub-step and which of the sub-step's drains it is (a
    chain's first column's K records 0, 1; its last column's V panels and rows of record 0, then of record 1)."""

    handle: list[int]
    layer: int
    column: int
    sub: int
    index: int


class Span(NamedTuple):
    """A transfer of a pass that follows its slots (several sequences, one pass): its BD word index, the pass, the
    ACT offset of its first slot's bytes for a pass from slot 0 on, the bytes per slot (its address moves by them per
    first slot), and its bytes: per slot (0), or as built (1: o's octets)."""

    at: int
    sub: int
    base: int
    unit: int
    fixed: int


def pass_subs(k: int) -> tuple[tuple[int, int, int], ...]:
    """(one pass) The sub-steps of a step of k sequences' slots as its transaction's key: a pass each, whatever its
    slots (``Span``: patched)."""
    return tuple((u, 1, 0) for u in range(k))


# how a windowed spec's sliding-window layers attend in a step (``AttnStep.mode``): "step" (row s at p + s: its window and
# the slots up to its own), "block" (a drafter's block: every row the window before the first slot's and the step's slots
# whole), "insert" (records only: the step's first group streamed, no window, the rows' outputs for a sink)
MODES = ("step", "block", "insert")


@dataclasses.dataclass(eq=False)
class AttnStep:
    """The transaction of a step of S slots (its attentions' sub-steps ``subs``: first slot, slots, how the chains divide
    their caches' groups; ``b``: which of its two buffers; ``fatt``: its F_ATT objects' fills as (BD word index, layer,
    column, sub-step): the second buffer's step is a copy that reads that buffer's objects; ``mode``: ``MODES``) with the
    records of the BDs that its position patches: ``fills``, ``tails``, ``drains`` and their arrays (``patch``)."""

    S: int
    subs: tuple[tuple[int, int, int], ...]
    tr: Transaction | None = None
    b: int = 0
    mode: str = "step"  # (a windowed spec's sliding-window layers: ``MODES``)
    fatt: list[tuple[int, int, int, int]] = dataclasses.field(default_factory=list)
    fills: list[Fill] = dataclasses.field(default_factory=list)
    tails: list[Tail] = dataclasses.field(default_factory=list)
    drains: list[Drain] = dataclasses.field(default_factory=list)
    spans: list[Span] = dataclasses.field(default_factory=list)
    patch: dict[str, np.ndarray] | None = None  # (``AttentionSteps.patch_tables``)


class AttnOverlay(Protocol):
    """What ``AttentionSteps`` takes of its decode overlay (``decode.model.ModelOverlay`` with the attention's layout:
    qwen35's ``QwenOverlay``): per attention layer L its F_ATT objects (``fatt``: L -> ACT offset, ``fatt_object``,
    ``fobj``: their first words [buffer][layer][column][sub-step]), q objects (``q_object``), the slots' raw K, V objects
    (``rawkv``), KV caches (``kv_off``, ``ash.head_bytes`` per KV head) and rings (``ring``)."""

    dev: Any
    ash: attn.AttnShape
    T: int
    SLOTS: int
    max_positions: int
    a_bds: int
    inp: dict
    fatt: dict[int, int]
    rawkv: dict[int, int]
    kv_off: dict[int, int]
    fobj: np.ndarray
    vring: VRing
    _vrings: list[VRing]
    seq: int
    seq_name: Callable[[str, int], str]
    _weights: Callable[..., None]
    _inputs: Callable[..., None]
    _weight_run: Callable[..., None]
    _sp: Callable[..., int]
    _bd_words: Callable[..., None]
    _once: Callable[[float], None]

    def x_slot(self, phase: tuple[int, str], e: int, S: int) -> tuple[int, int, int]: ...

    def staged(self, phase: tuple[int, str]) -> bool: ...

    def ring(self, L: int, h: int) -> int: ...

    def fatt_object(self, L: int, c: int, u: int, b: int = 0) -> int: ...

    def q_object(self, L: int, c: int, s: int) -> int: ...

    def subs(self, p: int, S: int) -> tuple[tuple[int, int, int], ...]: ...


class WindowOverlay(AttnOverlay, Protocol):
    """(a windowed spec, ``attn256.AttnSpec.window``) What ``AttentionSteps`` also takes of the overlay: a layer's kind
    ("full", "swa") and the groups of its caches' ring (0: linear, ``ash.head_bytes`` per KV head)."""

    def layer_kind(self, L: int) -> str: ...

    def ring_groups(self, L: int) -> int: ...


F_WORDS = {"pos": M_POS, "ns": M_NS, "sa": M_SA, "skip": M_SKIP, "split": M_SPLIT, "nrel": M_NREL, "ntail": M_NTAIL,
           "naskip": M_NASKIP, "g0": M_G0, "gatt": M_GATT, "gj": M_GJ, "ana": M_ANA}  # (an F_ATT object's words, by name)
F_LINES = np.arange(0, 4 * META, 64)  # (the cache lines of an F_ATT object's words)
F_ATT_AT = np.array(list(F_WORDS.values()))
NO_END = 1 << 20  # a core's end of attention groups: none
RECORD_STRIDE = 72  # bytes: a record's place in its pair in a panel (two records in 144 B)
# the columns of a patch table (``AttentionSteps.patch_tables``): BD word index, sub-step, the layer's index among the
# attention layers, the column's place in its chain, byte offsets in KV of group 0 of its KV head and of its ring, and
# a fill's kind (``FILLS``), a tail's object, a drain's index
AT, SUB, LAYER, PLACE, BASE, RING, ARG = range(7)


def att_words(ash: attn.AttnShape, subs: tuple[tuple[int, int, int], ...], p: int, *, first: int = 0
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The attention cores' words of a step at p with sub-steps ``subs`` (``F_WORDS``, per sub-step and column), the groups
    each kind of fill reads (per sub-step, ``FILLS``, place in a chain: first group, past the last) and per sub-step its first
    position and slots; ``first``: (one pass) the first group streamed (a sliding-window layer's window's,
    ``attn256.window_words``)."""
    nph, U = ash.nph, len(subs)
    w = np.zeros((U, COLS, len(F_WORDS)), np.int32)  # (per sub-step and column)
    gr = np.zeros((U, len(FILLS), nph, 2), np.int64)  # (a fill's groups, per place in a chain)
    su = np.zeros((U, 2), np.int64)  # (a sub-step's first position, slots)
    for u, (a, n, split) in enumerate(subs):
        q, g = p + a, (p + a) // attn.POS_PER_GROUP
        # (groups through the last slot's; split: chain core k's attention groups b[k] .. b[k + 1] - 1)
        G = (q + n - 1) // attn.POS_PER_GROUP + 1
        b = ash.bounds(g) if split else [0] * (nph + 1)
        su[u] = q, n
        for c in range(COLS):
            k = c % nph
            nt = ash.records(n) if not split or k == nph - 1 else 0
            o = attn.OBJS
            if not split:  # (a chain's first column: then group g's K and V objects, for its K records; one pass: none)
                na, g0, ga, gj = o * (G - first) + nt + (2 if k == 0 and not ash.one else 0), first, NO_END, 0
            elif k == 0:  # (then group g's K and V objects for its K records)
                na, g0, ga, gj = o * b[1] + 2, 0, b[1], g
            else:
                na, g0, ga, gj = o * (b[k + 1] - b[k]) + nt, b[k], NO_END, 0
            ana, na = na, 0 if ash.one and k else na  # (one pass: the chain's cores on the first column's stream)
            nq = 3 if split & 3 == attn.SPLIT_ROUND else nph + 1 if split else 2  # (per slot the X objects after F_ATT)
            sw = ash.round_word(split, c)
            v = {"pos": q, "ns": n, "sa": 0, "skip": nq * n, "split": sw, "nrel": ash.n_rel(c, sw, n), "ntail": nt,
                 "naskip": na, "g0": g0, "gatt": ga, "gj": gj, "ana": ana}
            w[u, c] = [v[k] for k in F_WORDS]  # (by name: a word of F_WORDS without a value is a KeyError)
        for k in range(nph):
            gr[u, :, k] = (first, G), (g, g + 1), (b[k], b[k + 1]), (first, G), (first, G)
    return w, gr, su


class AttentionSteps:
    """The attention layers' transfers and patches of the decode overlay ``ov``."""

    def __init__(self, ov: AttnOverlay):
        self.ov = ov
        # (a windowed spec: ``WindowOverlay``'s kinds and rings; an overlay without attention layers has no shape)
        self.window = ov.ash.spec.window if ov.ash is not None else 0

    def kind(self, L: int) -> str:
        """A layer's kind ("full", or a windowed spec's "swa")."""
        return self.ov.layer_kind(L) if self.window else "full"

    def ring_groups(self, L: int) -> int:
        """The groups of a layer's caches' ring (0: linear)."""
        return self.ov.ring_groups(L) if self.window else 0

    def head_bytes(self, L: int) -> int:
        """Bytes of a KV head's cache of layer L."""
        nr = self.ring_groups(L)
        return nr * attn.OBJS * self.ov.T if nr else self.ov.ash.head_bytes

    # ------------------------------------------------------------------
    # The transfers
    # ------------------------------------------------------------------

    def build(self, seq: Sequence, st: AttnStep, L: int, S: int, run: list[tuple[int, str]] | None = None, *,
              phase: str | None = "attn_in", out: str = "attn_out") -> None:
        """attn_in and the attention, per sub-step (each after the last one's records are in DDR: it reads them);
        ``run``: the weights after it (a column-major image), after the last sub-step's cache reads. ``phase``: the
        phase of the attention's input (another than attn_in: K and V alone, a drafter's injection; its rows' q objects
        as they are; None: the caller's phases filled the q objects and the raw K / V objects, a PARO checkpoint's
        attn_q, attn_k, attn_v), ``out``: the phase whose x takes o (``x_slot``)."""
        if phase is not None:
            self._input_phase(seq, L, S, phase)
        self._sub_steps(seq, st, L, S, run, out)

    def _input_phase(self, seq: Sequence, L: int, S: int, phase: str) -> None:
        """The attention's input phase: its weights and input, its outputs into the q objects and the raw K / V
        objects."""
        ov = self.ov
        dev, ash = ov.dev, ov.ash
        qkv = phase == "attn_in"
        gh = ash.spec.gate == "head" and qkv  # (the heads' gate logits after K and V: one C object, its first QC values)
        ov._weights(seq, (L, phase), 0)
        seq.sync()
        ov._inputs(seq, ov.inp[(L, phase)][0], S, (L, phase))
        for c in range(COLS):  # [q + gate | K part | V part (| gates)] per slot -> its q object, the slot's raw K, V object
            q0 = ov.q_object(L, c, 0) + 4 * META
            if qkv:
                self.q_drain(seq, L, S, c)
            for part in (0, 1):
                self.kv_drain(seq, L, S, c, part)
            if gh:  # (the q object's QC gates after its heads' q)
                seq.drain("C", c, dev.address("ACT", q0 + 2 * ash.qc * attn.HD), S * 2 * C_LEN,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))

    def q_drain(self, seq: Sequence, L: int, S: int, c: int) -> None:
        """Column c's q rows (with their gates) of S slots -> its q objects."""
        ov, ash = self.ov, self.ov.ash
        seq.drain("C", c, ov.dev.address("ACT", ov.q_object(L, c, 0) + 4 * META),
                  S * ash.qc * 2 * (1 if ash.spec.gate == "head" else 2) * attn.HD, (C_LEN // 2, S, XB // 4, C_LEN // 2))

    def kv_drain(self, seq: Sequence, L: int, S: int, c: int, part: int) -> None:
        """Column c's K (part 0) or V (part 1) rows of S slots -> the slots' raw K / V objects (its part of a KV head)."""
        ov = self.ov
        per = ov.ash.kvh * attn.HD // COLS
        h, e0 = divmod(per * c, attn.HD)
        raw = ov.rawkv[L] + 4 * META
        seq.drain("C", c, ov.dev.address("ACT", raw + 2 * (2 * attn.HD * h + attn.HD * part + e0)), S * 2 * per,
                  (C_LEN // 2, S, XB // 4, C_LEN // 2))

    def _sub_steps(self, seq: Sequence, st: AttnStep, L: int, S: int, run: list[tuple[int, str]] | None,
                   out: str) -> None:
        """The attention's sub-steps (``build``)."""
        ov = self.ov
        ash, nph = ov.ash, ov.ash.nph
        nf = 2 if self.ring_groups(L) else 1  # (the cache's fills of a one-pass column: ``_cache``)
        for u, (a, n, split) in enumerate(st.subs):
            rnd = split & 3 == attn.SPLIT_ROUND  # (a round: both chains on its KV head, the cache shared 4 columns apart)
            last = u == len(st.subs) - 1 and run
            if not split & attn.SPLIT_KV:  # (a sub-step's second round: issued with its first)
                if (u and not ash.one) or rnd:  # (one pass: sub-step u is another sequence's pass, which reads none of them)
                    seq.sync()  # (sub-step u - 1's records in DDR; a shared stream: every column's objects before it through)
                self._cache(seq, st, L, u)
                if last and not rnd and ov.a_bds > nf + attn.TAILS:  # (a later block's after the X objects: ``_weight_run``)
                    ov._weight_run(seq, run, blocks=slice(1))
                if u == 0 and not rnd:
                    seq.sync()  # (attn_in's q, K, V in DDR)
                self._inputs(seq, st, L, u)
                if rnd:  # (the second round, on the same streams after the first's objects: its inputs, then the tasks
                    # that wait for the first round's to finish (a channel queues 4): the second's tails, the drains)
                    self._inputs(seq, st, L, u + 1)
                    self._cache(seq, st, L, u + 1)
                if last and not rnd:
                    ov._weight_run(seq, run, blocks=slice(1 if ov.a_bds > nf + attn.TAILS else 0, None))
                self._outputs(seq, st, L, S, u, out)
                if rnd:
                    self._outputs(seq, st, L, S, u + 1, out)
            if rnd and u == len(st.subs) - 1:  # (then the columns' own objects: they keep no order with a shared stream's)
                seq.wait_fills("A", tuple(range(nph)))
                if run:
                    ov._weight_run(seq, run)

    def _cache(self, seq: Sequence, st: AttnStep, L: int, u: int) -> None:
        """The cache of the column's KV head (address, length: ``set_position``), its ring (one pass: the chain's first
        column's stream, to both cores; a round: the first chain's streams, the round's KV head, cc its column in that
        head's chain)."""
        ov = self.ov
        dev, ash, T, nph = ov.dev, ov.ash, ov.T, ov.ash.nph
        _, n, split = st.subs[u]
        rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
        for c in range(0, COLS, nph if ash.one else 1):
            if rnd and c >= nph:  # (its objects: the column 4 away's)
                continue
            cc = kh * nph + c if rnd else c
            head = dev.address("KV", ov.kv_off[L] + cc // nph * self.head_bytes(L))
            pk = {"packet": AB_ID + ash.a_bcast.index((c, c + nph))} if rnd else {}
            kinds = (("ra", "rb") if self.ring_groups(L) else ("all",)) if not split else \
                ("run", "new") if c % nph == 0 else ("run",)
            for kind in kinds:  # (split: the chain's first column its run, then the new positions' group's K and V objects)
                st.fills.append(Fill(seq.fill("A", c, head, attn.OBJS * T, **pk), L, cc, u, kind))
            if not split or ash.role(c) == attn.ROLE_V:  # (a column that attends to the newest group: the tails)
                st.tails += [Tail(seq.fill("A", c, dev.address("KV", ov.ring(L, cc // nph)), T, **pk), L, cc, u, t)
                             for t in range(ash.records(n))]
            if not ash.one and not split and c % nph == 0:  # (then the new positions' group again, for their records)
                st.fills.append(Fill(seq.fill("A", c, head, attn.OBJS * T), L, c, u, "new"))

    def _inputs(self, seq: Sequence, st: AttnStep, L: int, u: int) -> None:
        """The F_ATT object, the q objects, the slots' raw K, V objects."""
        ov = self.ov
        dev, nph = ov.dev, ov.ash.nph
        a, n, split = st.subs[u]
        rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
        for c in range(COLS):
            st.fatt.append((seq.fill("X", c, dev.address("ACT", ov.fatt_object(L, c, u, st.b)), XB), L, c, u))
            if split:  # (the q objects of the chain's columns in order; a round: of its half of the KV head's)
                q0, nc = (kh * nph + 2 * (c // nph), 2) if rnd else (c - c % nph, nph)
                seq.fill("X", c, dev.address("ACT", ov.q_object(L, q0, a)), nc * n * XB,
                         (XB // 8, 2 * n, XB // 8, ov.SLOTS * XB // 4))
            else:
                at = seq.fill("X", c, dev.address("ACT", ov.q_object(L, c, a)), n * XB)
                self._span(st, at, u, ov.q_object(L, c, 0), XB)
            at = seq.fill("X", c, dev.address("ACT", ov.rawkv[L] + a * XB), n * XB)
            self._span(st, at, u, ov.rawkv[L], XB)

    def _span(self, st: AttnStep, at: int, u: int, base: int, unit: int, *, fixed: bool = False) -> None:
        """(an overlay of several sequences, one pass) Pass u's transfer at BD word ``at`` follows its slots (``Span``)."""
        if self.ov.ash.one and self.ov.ash.u > 1:
            st.spans.append(Span(at, u, base, unit, int(fixed)))

    def _outputs(self, seq: Sequence, st: AttnStep, L: int, S: int, u: int, out: str = "attn_out") -> None:
        """o (the column's heads in the x of phase ``out`` at the sub-step's slots; split: the chain's last column's of the
        chain's, a round's of its half), then the records (addresses: ``set_position``)."""
        ov = self.ov
        dev, ash, nph, qc = ov.dev, ov.ash, ov.ash.nph, ov.ash.qc
        a, n, split = st.subs[u]
        rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
        for c in range(COLS):
            role = attn.ROLE_O if ash.round_word(split, c) & attn.SPLIT_NOREC else ash.role(c)
            if not split or c % nph == nph - 1:
                h0, nh = (((kh * nph + 2 * (c // nph)) * qc, 2 * qc) if rnd else
                          (qc * (c - nph + 1), nph * qc) if split else (qc * c, qc))
                sp = ov._sp(S, (L, out))
                at, left, _ = ov.x_slot((L, out), attn.HD * h0, S)
                assert left >= 4 * sp * 128 * nh
                sw, m = 4 * sp, sp if ash.one else n  # (words of an x k block: [slot sp][8] bf16; the slots out)
                if ov.staged((L, out)):  # (a pass's octets from its first slot on: a k block's into its cell)
                    hd = seq.drain("C", c, dev.address("ACT", at + 16 * a), 16 * m * 32 * nh,
                                   (sw, 32 * nh, 2 * sw, 2 * sw * 32 * nh))
                    self._span(st, hd[0], u, at, 16, fixed=True)
                else:
                    seq.drain("C", c, dev.address("ACT", at + 16 * a), 16 * m * 32 * nh,
                              None if m == sp else (4 * m, 32 * nh, sw, sw * 32 * nh))
            nk = attn.REC_K if ash.one else 2 * attn.REC_K  # (one pass: a K record; sub-steps: its pair)
            nr = ash.records(n)
            sizes = {attn.ROLE_K: (nk,) * nr, attn.ROLE_V: (attn.REC_V, attn.ROWS_B) * nr, attn.ROLE_O: ()}[role]
            for k, nb in enumerate(sizes):  # (a record's panels into its pair: V_SCATTER)
                dims = attn.V_SCATTER if nb == attn.REC_V else None  # (REC_K == REC_V)
                hd = seq.drain("C", c, dev.address("KV", ov.kv_off[L]), nb, dims)
                st.drains.append(Drain(hd, L, c, u, k))

    # ------------------------------------------------------------------
    # The patches of a step's position
    # ------------------------------------------------------------------

    def patch_tables(self, st: AttnStep) -> dict[str, np.ndarray]:
        """A step's cache fills, tail fills and record drains as arrays (columns ``AT`` .. ``ARG``), and the cache lines of
        their BDs."""
        ov, ash = self.ov, self.ov.ash
        nph = ash.nph
        ix = {L: i for i, L in enumerate(ov.fatt)}

        def where(L: int, c: int) -> tuple[int, int, int, int]:
            return ix[L], c % nph, ov.kv_off[L] + c // nph * self.head_bytes(L), ov.ring(L, c // nph)

        f = np.array([(r.at, r.sub, *where(r.layer, r.column), FILLS.index(r.kind)) for r in st.fills],
                     np.int64).reshape(-1, 7)
        t = np.array([(r.at, r.sub, *where(r.layer, r.column), r.tail) for r in st.tails], np.int64).reshape(-1, 7)
        r = np.array([(r.handle[0], r.sub, *where(r.layer, r.column), r.index) for r in st.drains],
                     np.int64).reshape(-1, 7)
        s = np.array(st.spans, np.int64).reshape(-1, 5)  # (``Span``'s columns)
        at = np.concatenate([x[:, AT] for x in (f, t, r, s)] + [x[:, AT] + 2 for x in (f, t, r, s)])
        out = {"f": f, "t": t, "r": r, "s": s, "lines": np.unique(4 * at & ~63)}
        if self.window:  # (per attention layer, by its index: its ring's groups, whether it is a sliding-window layer)
            out["nr"] = np.array([self.ring_groups(L) for L in ov.fatt], np.int64)
            out["swa"] = np.array([self.kind(L) == "swa" for L in ov.fatt])
        return out

    def set_position(self, st: AttnStep, p: int, cs: np.ndarray | dict[str, np.ndarray]) -> None:
        """The F_ATT objects (per sub-step: its positions, cache objects, rotations) and the step's cache and ring
        transfers. ``cs``: its positions' angles (S, 64) [cos 32 | sin 32] fp32 (``attn256.rope_cs``); a windowed spec's,
        per layer kind the payload of the F_ATT objects (S, n) uint8 (a slot's angles as the core reads them:
        ``attn256.AttnSpec``)."""
        ov = self.ov
        S = st.S
        assert 0 <= p and p + S <= ov.max_positions, (p, S)
        assert st.subs == ov.subs(p, S), (p, S, st.subs)
        self._rings_from_prefill(p)
        for L in ov.fatt:
            ov.vring.check(L, p, p)
        if self.window:
            gr, su = self._window_words(st, (Part(ov.seq, p, S),), cs)
        else:
            w, gr, su = att_words(ov.ash, st.subs, p)
            self._write_words(st, w, cs)
        self._patch(st, gr, su, ov.dev.address("KV"))

    def set_parts(self, st: AttnStep, parts: tuple[Part, ...], cs: np.ndarray | dict[str, np.ndarray]) -> None:
        """(one pass) ``set_position`` of a step of several sequences' slots, a pass per part (``st.subs``: each
        part's first slot and its slots): its F_ATT objects by its own position, its cache, ring and record transfers
        in its sequence's KV. ``cs``: the angles of the step's slots, in the parts' order."""
        ov = self.ov
        assert ov.ash.one and st.mode == "step" and len(parts) == len(st.subs), (parts, st.subs)
        assert sum(q.n for q in parts) == st.S, (parts, st.S)
        for q in parts:
            assert 0 <= q.position and q.position + q.n <= ov.max_positions, parts
            self._rings_from_prefill(q.position, q.seq)
            for L in ov.fatt:
                ov._vrings[q.seq].check(L, q.position, q.position)
        gr, su = self._window_words(st, parts, cs) if self.window else self._pass_words(st, parts, cs)
        kv0 = np.array([ov.dev.address(ov.seq_name("KV", q.seq), own=True) for q in parts], np.int64)
        if st.patch is None:
            st.patch = self.patch_tables(st)
        s = st.patch["s"]  # (the transfers that follow each pass's slots: its first slot, its slots)
        n = np.array([q.n for q in parts], np.int64)
        a = np.cumsum(n) - n
        ov._bd_words(st.tr.words, s[:, 0], ov.dev.address("ACT") + s[:, 2] + a[s[:, 1]] * s[:, 3],
                     np.where(s[:, 4] == 1, st.tr.words[s[:, 0]].astype(np.int64) * 4, n[s[:, 1]] * s[:, 3]))
        self._patch(st, gr, su, kv0)

    def _patch(self, st: AttnStep, gr: np.ndarray, su: np.ndarray, kv0: int | np.ndarray) -> None:
        """The step's cache and ring fills and record drains (``gr``, ``su``: the F_ATT words' groups and sub-steps);
        ``kv0``: the address of KV, or per sub-step of its sequence's."""
        if st.patch is None:
            st.patch = self.patch_tables(st)
        t, words = st.patch, st.tr.words

        def at(rows: np.ndarray) -> int | np.ndarray:
            return kv0 if np.ndim(kv0) == 0 else kv0[rows[:, SUB]]

        self.patch_fills(words, t["f"], gr, at(t["f"]), t)
        self.patch_tails(words, t["t"], su, at(t["t"]))
        self.patch_drains(words, t["r"], su, at(t["r"]), t)
        st.tr.flush_lines(t["lines"])

    def _pass_words(self, st: AttnStep, parts: tuple[Part, ...], cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(one pass, no window) The F_ATT objects' words of a step of ``parts``, a pass each: those of a step of the
        part's slots alone at its position (a chain: no branch) and its angles; the fills' groups and the
        sub-steps."""
        ov = self.ov
        U = len(parts)
        ws, grs, sus = [], [], []
        a32 = ov.dev.view("ACT").view(np.int32)
        fo = ov.fobj[st.b][:, :, :U]  # ([layer][column][sub-step])
        for u, q in enumerate(parts):
            w, gr, su = att_words(ov.ash, ((0, q.n, 0),), q.position)
            ws.append(w[0]), grs.append(gr[0]), sus.append(su[0])
            _, amask, lanes = attn.step_words(q.position, q.n)
            f = fo[:, :, u]
            a32[f[..., None] + attn.M_AMASK + np.arange(4)] = amask.view(np.int32)
            a32[f + attn.M_ALANE], a32[f + attn.M_BRANCH] = lanes, 0
        self._write_words(st, np.stack(ws), cs, parts)
        return np.stack(grs), np.stack(sus)

    def _write_words(self, st: AttnStep, w: np.ndarray, cs: np.ndarray, parts: tuple[Part, ...] | None = None) -> None:
        """The F_ATT objects' words ``w`` (per sub-step and column) and the sub-steps' slots' angles ``cs``, flushed;
        ``parts``: (several sequences) the passes' slots are the parts' (else the sub-steps')."""
        ov = self.ov
        U = len(st.subs)
        spans = [(a, n) for a, n, _ in st.subs]
        if parts is not None:
            n = [q.n for q in parts]
            spans = [(sum(n[:u]), n[u]) for u in range(U)]
        act = ov.dev.view("ACT")
        fo = ov.fobj[st.b][:, :, :U]  # (the F_ATT objects' first words of st's buffer: [layer][column][sub-step])
        a32 = act.view(np.int32)
        a32[fo[..., None] + F_ATT_AT] = w[np.arange(U), np.arange(COLS)[:, None]]
        q0 = META + 2 * attn.HD  # (the sub-step's slots' angles, in words)
        af = act.view(np.float32)
        for u, (a, n) in enumerate(spans):
            af[fo[..., u, None] + q0 + np.arange(64 * n)] = cs[a:a + n].reshape(-1)
        lines = np.concatenate([np.arange(0, 192, 64),
                                4 * q0 + np.arange(0, 4 * 64 * max(n for _, n in spans) + 64, 64)])
        ov.dev.flush_at("ACT", 4 * fo[..., None] + lines)

    def _window_words(self, st: AttnStep, parts: tuple[Part, ...], cs: dict[str, np.ndarray]
                      ) -> tuple[np.ndarray, np.ndarray]:
        """(a windowed spec, one pass) The F_ATT objects' words of a step of ``parts`` (one: the step's slots at its
        position), a pass each: the pass's (``attn256.step_words`` of its slots at its position: no branches), each
        layer kind's (a sliding-window layer's from its window's first group, its masks: ``attn256.window_words``;
        M_KIND: the overlay's, static) and payloads (``cs``: the pass's slots' rows), flushed; the fills' groups per
        kind (gr: [kind 0 full, 1 swa] then ``att_words``'), the sub-steps."""
        ov = self.ov
        U = len(parts)
        assert ov.ash.one and U == len(st.subs), (parts, st.subs)
        a32, a8 = ov.dev.view("ACT").view(np.int32), ov.dev.view("ACT")
        fo = ov.fobj[st.b][:, :, :U]  # ([layer][column][sub-step])
        kinds = np.array([self.kind(L) == "swa" for L in ov.fatt])
        assert not ov.ash.spec.branch and (st.mode == "step" or kinds.all()), st.mode
        assert st.mode == "step" or U == 1, (st.mode, parts)
        grs: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        sus = []
        none = (0, -1, np.zeros(4, np.uint32), np.zeros(2, np.uint32))
        a = 0
        for u, q in enumerate(parts):
            p, n = q.position, q.n
            f = fo[:, :, u]  # ([layer][column])
            # (a step without branches: its rows' masks, its slots' K lanes)
            _, amask, lanes = attn.step_words(p, n, block=st.mode == "block")
            a32[f[..., None] + attn.M_AMASK + np.arange(4)] = amask.view(np.int32)
            a32[f + attn.M_ALANE] = lanes
            for swa in (False, True):
                if not swa:
                    first, wrec, wm, late = none
                elif st.mode == "insert":  # (the records' groups alone)
                    first, wrec, wm, late = (p >> 5, *none[1:])
                else:
                    first, wrec, wm, late = attn.window_words(p, self.window, block=st.mode == "block")
                w, gr, su = att_words(ov.ash, ((0, n, 0),), p, first=first)
                grs[swa].append(gr[0])
                sel = f[kinds == swa]  # ([layer][column])
                if not sel.size:
                    continue
                a32[sel[..., None] + F_ATT_AT] = w[0]
                if swa:
                    a32[sel + attn.M_WREC] = wrec
                    a32[sel[..., None] + attn.M_WMASK + np.arange(4)] = wm.view(np.int32)
                    a32[sel[..., None] + attn.M_WLATE + np.arange(2)] = late.view(np.int32)
                    a32[sel + attn.M_WHOLE] = int(st.mode == "block")
                pay = np.ascontiguousarray(cs["swa" if swa else "full"][a:a + n], np.uint8).reshape(-1)
                for o in 4 * sel.reshape(-1) + 4 * META:  # (each object's payload: a slice, not an index per byte)
                    a8[o:o + pay.size] = pay
                lines = np.concatenate([F_LINES, 4 * META + np.arange(0, pay.size + 64, 64)])
                ov.dev.flush_at("ACT", 4 * sel[..., None] + lines)
            sus.append(su[0])
            a += n
        return np.stack([np.stack(g) for g in grs]), np.stack(sus)

    def patch_fills(self, words: np.ndarray, f: np.ndarray, gr: np.ndarray, kv0: int, t: dict[str, np.ndarray]) -> None:
        """The cache fills: the KV head's groups of the fill's kind (an object of a split column's new group: 2 T; a
        ring's groups g0 .. G - 1 at their slots g % NR: "ra" up to the ring's end and "rb" from its start where they
        wrap it, else "ra" all but the last object and "rb" that one). ``gr``: a windowed spec's per layer kind."""
        T, O = self.ov.T, attn.OBJS
        g = gr[t["swa"][f[:, LAYER]].astype(np.int64), f[:, SUB], f[:, ARG], f[:, PLACE]] if self.window else \
            gr[f[:, SUB], f[:, ARG], f[:, PLACE]]
        start, n = O * g[:, 0], np.where(f[:, ARG] == 1, 2, O * (g[:, 1] - g[:, 0]))  # (objects)
        if self.window and (ring := f[:, ARG] >= FILLS.index("ra")).any():
            nr = t["nr"][f[:, LAYER]]
            m = np.maximum(nr, 1)
            a, b = g[:, 0] % m, (g[:, 1] - 1) % m + 1
            assert (g[ring, 1] - g[ring, 0] <= nr[ring]).all()
            wrap, ra, rb = b <= a, f[:, ARG] == FILLS.index("ra"), f[:, ARG] == FILLS.index("rb")
            start = np.select([ra, rb & ~wrap, rb], [O * a, O * b - 1, 0], start)
            n = np.select([ra & ~wrap, ra, rb & ~wrap, rb], [O * (b - a) - 1, O * (m - a), 1, O * b], n)
        self.ov._bd_words(words, f[:, AT], kv0 + f[:, BASE] + start * T, n * T)

    def patch_tails(self, words: np.ndarray, t: np.ndarray, su: np.ndarray, kv0: int) -> None:
        """The ring slots of the sub-step's first and last records."""
        T = self.ov.T
        r = (su[t[:, SUB], 0] + np.where(t[:, ARG], su[t[:, SUB], 1] - 1, 0)) >> 3
        self.ov._bd_words(words, t[:, AT], kv0 + t[:, RING] + (r & 1) * T, np.full(len(r), T, np.int64))

    def patch_drains(self, words: np.ndarray, d: np.ndarray, su: np.ndarray, kv0: int, t: dict[str, np.ndarray]) -> None:
        """The record drains: the first column's K records, the last's V panels (into their pairs) and rows, of record 0, 1
        (a ring's group r // 4 at its slot)."""
        ov = self.ov
        ash, T = ov.ash, ov.T
        v = d[:, PLACE] == ash.nph - 1
        r = (su[d[:, SUB], 0] + np.where(np.where(v, d[:, ARG] >= 2, d[:, ARG] > 0), su[d[:, SUB], 1] - 1, 0)) >> 3
        head, j = d[:, BASE] + attn.OBJS * self._slot(r // 4, d[:, LAYER], t) * T, r % 4
        kind = np.where(v, 1 + d[:, ARG] % 2, 0)  # (K, V panels, rows)
        pair = j // 2 * 2 * attn.REC_K  # (a record in its pair; sub-steps: the K records' pairs whole)
        off = np.choose(kind, [head + pair + (j % 2 * RECORD_STRIDE if ash.one else 0),
                               head + T + pair + j % 2 * RECORD_STRIDE, d[:, RING] + (r & 1) * T])
        nk = attn.REC_K if ash.one else 2 * attn.REC_K
        ov._bd_words(words, d[:, AT], kv0 + off, np.array([nk, attn.REC_V, attn.ROWS_B])[kind])

    def _slot(self, g: np.ndarray, layer: np.ndarray, t: dict[str, np.ndarray]) -> np.ndarray:
        """The slots of groups g of the attention layers of index ``layer`` (a ring's: g % NR)."""
        if not self.window:
            return g
        nr = t["nr"][layer]
        return np.where(nr > 0, g % np.maximum(nr, 1), g)

    def _rings_from_prefill(self, p: int, seq: int | None = None) -> None:
        """Before a step at p (of sequence ``seq``; None: the selected one), of each attention layer whose caches a
        prefill wrote last (it writes no ring): its open record (p >> 3) as rows into its ring slot, from the record's V
        panels in the cache (their values are bf16 values: exact), so that a step reads every open record from the
        ring."""
        ov = self.ov
        vring = ov.vring if seq is None else ov._vrings[seq]
        layers = [L for L in ov.fatt if L in vring.from_prefill]
        if not layers:
            return
        t0 = time.perf_counter()
        r, dev, ash, T = p >> 3, ov.dev, ov.ash, ov.T
        # (the sequence's own KV buffer; the selected one's: under its plain name)
        name, own = ("KV", False) if seq is None else (ov.seq_name("KV", seq), True)
        if p % 8:
            kv = dev.view(name, own=own)
            for L in layers:
                nr = self.ring_groups(L)
                g = (r // 4) % nr if nr else r // 4
                for h in range(ash.kvh):  # (the record in its pair [d-block][record 2] of the group's V object)
                    a = ov.kv_off[L] + h * self.head_bytes(L) + attn.OBJS * g * T + T + (r % 4 // 2) * 2 * attn.REC_V
                    dev.fetch(name, a, 2 * attn.REC_V, own=own)
                    v = block_values(kv[a:a + 2 * attn.REC_V].reshape(attn.ND, 2, 8, 9)[:, r % 2])  # ([d-block][dim][position])
                    o = ov.ring(L, h) + (r & 1) * T
                    kv[o:o + attn.ROWS_B] = np.ascontiguousarray(v.transpose(0, 2, 1), bfloat16).view(np.uint8).reshape(-1)
                    dev.flush(name, o, attn.ROWS_B, own=own)
        for L in layers:
            vring.slots.setdefault(L, {})[r & 1] = (r, p)
            vring.from_prefill.discard(L)
        ov._once(t0)
