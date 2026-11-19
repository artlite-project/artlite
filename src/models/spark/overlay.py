"""A Spark-X model on the decode overlay of 8 slots (``spark/decode.py``): buffers, the transaction of a step of S <= 8
token slots, the step.

``SparkOverlay`` lays out the weight image (``image.SparkImage``, packed at ``load``), the norm weights (NORM:
contiguous bf16; a region-major image's phases read them as overlapping A objects, a column-major one's from its own
data regions), every phase's input (ACT: per K group [metadata object | x slot-interleaved], ``decode.protocol``; a
producer drains its outputs straight into its consumer's x), the residual-slice objects (F_LOADR), the attention layers'
objects (per column its F_ATT object, then per column its q objects; the raw K / V objects) and KV caches (KV,
``decode.kv``'s d-block-major layout in groups of 32 positions: a full layer's per KV head linear up to
``max_positions``, a sliding-window layer's a ring of ``ring_groups`` groups, group g at slot g % NR, which holds its
window and a prefill chunk; then per layer and KV head a ring of the open V records' values, ``decode.vring``), and the
logits ([slot][vocab] bf16).

A step is one dispatch of the transaction of its slot count (built at its first use), patched per step with its
position (``decode.attn256_steps``: the F_ATT objects' words, a sliding-window layer's window and masks, the angles of
each kind; the cache and ring reads, a ring's window in two fills; the record drains). While a step runs the host patches
the next one (S slots at the next position) into a second transaction buffer that reads F_ATT objects of its own
(``ModelOverlay.AHEAD``: 0.7 / 1.0 ms of the 1.7B's / 4B's step at 1K that the NPU otherwise waits for; ``fatt2``, the
second buffer's transactions copies of the first's, made at their first use: ``_transaction``). Per step the columns
load the slots' residual slices (F_LOADR), then per layer

- attn_in (normed: ``attn_norm``; q and the heads' gate logits to the q objects, the column's parts of K / V to the raw
  objects) | the attention (one pass: the cache through A, a KV head's chain of columns; o into attn_out's x, the
  records into the cache) | attn_out (into the residual);
- gate_up (normed: ``ffn_norm``; gelu(gate) up) | down (into the residual);

and the LM head (normed: ``output_norm``) -> LOGITS. A column-major image (the 4B) streams a column's weights between two
attentions as one run (``ModelOverlay._weight_run``), a region-major one (the 1.7B) per phase and K group.

Several sequences (``sequences``: each one's caches and rings in a KV buffer of its own, ``ModelOverlay.select``): a
step's slots may be several sequences' (``decode.step.Part``): each attention then runs a pass per sequence over that
sequence's cache, its rows drained into the sequence's slots of attn_out's staged x (``decode.attn256_steps``' "Several
sequences"), and every slot's logits are those of a step of one of its sequence, bit for bit.
"""

import dataclasses

import numpy as np
from ml_dtypes import bfloat16

from artlite import txn
from artlite.decode import attn256 as attn
from artlite.decode import phases as mp
from artlite.decode.attn256 import F_ATT, KIND_FULL, KIND_SWA, M_KIND, M_WREC
from artlite.decode.attn256_steps import AttentionSteps, AttnStep, pass_subs
from artlite.decode.model import CacheRows, ModelOverlay, Snapshot, StepKey, host_bytes, paged_formats
from artlite.decode.protocol import (
    F_EMITR,
    F_GU,
    F_NORM,
    F_RESID,
    M_FLAGS,
    M_SP,
    META,
    NX_PER_NORM,
    OBJ_K,
    XB,
    interleave,
)
from artlite.decode.step import Part
from artlite.gguf_model import GGUFFile
from artlite.model_config import SparkConfig
from artlite.operators.design import COLS
from artlite.operators.util import cos_sin_rows
from artlite.overlay import Sequence, Transaction, WeightBuffer

from .decode import SparkDecodeOperator, attn_spec
from .image import NORMS, SparkImage

FLAGS = {"attn_in": F_NORM, "attn_out": F_RESID | F_EMITR, "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR,
         "lm": F_NORM}
CHUNK = 2048  # a prefill chunk's positions at most (a sliding-window layer's ring holds its window and a chunk)


def ring_groups(window: int) -> int:
    """The groups of a sliding-window layer's ring: its window and a prefill chunk, each from any position in its group,
    and the group after them."""
    return (window + CHUNK) // attn.POS_PER_GROUP + 2


class SparkOverlay(ModelOverlay):
    """A Spark-X GGUF model on the decode overlay of 8 slots. ``layers``: the layers to run (default: all; tests run
    slices), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the full layers' caches' capacity,
    ``weights``: the image's buffer allocated ahead (its image's layers and vocabulary), ``formats``, ``gu``: tile
    formats and gate / up rows per tile the design runs besides the image's (a drafter's: ``dspark.DSparkOverlay.needs``),
    ``head``: the LM rows with a drafter's head (``SparkImage``), ``image``: an image laid out already
    (``ModelOverlay``'s), ``sequences``: the sequences whose caches the device holds (``ModelOverlay.select``)."""

    CONFIG, IMAGE = SparkConfig, SparkImage
    NORMED = frozenset(NORMS)
    AHEAD, AHEAD_PREPARED = True, False
    SEQ = ("KV",)

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, weights: WeightBuffer | None = None, formats: tuple[str, ...] = (),
                 gu: tuple[int, ...] = (), head: bool = False, image: SparkImage | None = None, sequences: int = 1):
        if image is None and weights is None:
            image = SparkImage(gguf, layers=layers, vocab=vocab, head=head)
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights, image=image, sequences=sequences)
        cfg, img = self.cfg, self.image
        if sequences > 8:
            raise ValueError(f"{sequences} sequences: a step has 8 slots, a sequence each at most")
        fmts = img.formats() | set(formats)
        gu = {R for r in img.regions if r.name == "gate_up" for b in r.blocks for R in (b.R, b.R2) if R} | set(gu)
        self.ash = attn.AttnShape(cfg, max_positions, a_bytes=img.tile_bytes, slots=8, spec=attn_spec(cfg),
                                  passes=sequences)
        self.max_positions = self.ash.max_positions
        self.nr = ring_groups(cfg.sliding_window)
        self.period = 1  # (one pass: a step's transaction is the same at every position)
        gu_win = gu != {16} or img.split
        # (four formats with the gate / up windows' frame: every one a page, ``paged_formats``' wide; bartowski's 4B
        # Q4_K_M, whose Q5_K page overflows the slot beside a resident Q4_K by 1.2 KB)
        self.op = SparkDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, attn=self.ash,
                                      eps=cfg.rms_norm_eps, paged=paged_formats(gguf, fmts, wide=gu_win),
                                      gu_win=gu_win, split=img.split, tails=any(r.half for r in img.regions),
                                      tail_pairs=any(b.pt for r in img.regions for b in r.blocks))
        self.attn_steps = AttentionSteps(self)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def layer_kind(self, L: int) -> str:
        """A layer's kind ("full", "swa")."""
        return self.cfg.kinds[L]

    def ring_groups(self, L: int) -> int:
        """The groups of layer L's caches' ring (a full layer's linear caches: 0)."""
        return self.nr if self.cfg.kinds[L] == "swa" else 0

    def _flags(self, L: int, name: str) -> int:
        return FLAGS[name]

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        out = super()._groups(L, name, S)
        for g in out:  # (the input's slots per k block, the host's per phase)
            g.meta[M_SP] = self._sp(S, (L, name))
        return out

    def _sp(self, S: int, phase: tuple[int, str] | None = None) -> int:
        """(attn_out's input: the attention's rows of whole octets)"""
        return self.SLOTS if phase is not None and phase[1] == "attn_out" else super()._sp(S, phase)

    def _phases(self) -> list[tuple[int, str]]:
        return [(L, n) for L in self.layers for n in ("attn_in", "attn_out", "gate_up", "down")] + [(-1, "lm")]

    def _input(self, L: int) -> tuple[int, str]:
        """The phase that reads layer L's input (L = -1: the LM head)."""
        return (-1, "lm") if L < 0 else (L, "attn_in")

    def _plan(self) -> None:
        """Byte offsets in ACT, NORM, KV."""
        self._layout = dict.fromkeys(("ACT", "NORM", "KV"), 0)
        take, ash, T = self._take, self.ash, self.T
        self.loadr = take("ACT", COLS * self.SLOTS * XB)  # [column][slot]
        for phase in self._phases():  # (several sequences: attn_out's x staged, the attention's passes write it)
            self._add_inputs(phase, staged=phase[1] == "attn_out" and ash.u > 1)
        self.norm_off: dict[tuple[int, str], int] = {}
        self.n_norm = -(-self.H // (NX_PER_NORM * OBJ_K))  # norm-weight A objects of a phase
        nw = 2 * NX_PER_NORM * OBJ_K * (self.n_norm - 1) + T  # (the last object's extent)
        for L in self.layers:
            for name in ("attn_norm", "ffn_norm"):
                self.norm_off[(L, name)] = take("NORM", nw)
        self.norm_off[(-1, "output_norm")] = take("NORM", nw)
        self.fatt, self.fatt2, self.rawkv, self.kv_off, self.ring_off = {}, {}, {}, {}, {}
        for L in self.layers:  # ([column][F_ATT], then [column][q objects SLOTS]; the second buffer's F_ATT; the raw K / V
            # objects; the caches)
            self.fatt[L] = take("ACT", COLS * (ash.u + self.SLOTS) * XB)
            self.fatt2[L] = take("ACT", COLS * ash.u * XB)
            self.rawkv[L] = take("ACT", self.SLOTS * XB)
            self.kv_off[L] = take("KV", ash.kvh * self.attn_steps.head_bytes(L))
        # (the F_ATT objects' first words: [transaction buffer][layer][column][sub-step])
        self.fobj = np.array([[[[self.fatt_object(L, c, u, b) // 4 for u in range(ash.u)] for c in range(COLS)]
                               for L in self.fatt] for b in range(2)], np.int64)
        r0 = self._layout["KV"]
        for L in self.layers:  # (per KV head two slots: ``ring``)
            self.ring_off[L] = take("KV", ash.kvh * 2 * T)
        self.rings = (r0, self._layout["KV"] - r0)
        take("KV", (CHUNK // attn.POS_PER_GROUP + 1) * attn.OBJS * T)  # (reads past the last cache's end)
        self._sizes()

    def ring(self, L: int, h: int) -> int:
        """Byte offset in KV of KV head h's ring of attention layer L: 2 slots of T bytes, slot r & 1 the V values of
        record r ([d-block][position 8][8] bf16: ``decode.vring``)."""
        return self.ring_off[L] + 2 * h * self.T

    def fatt_object(self, L: int, c: int, u: int, b: int = 0) -> int:
        """ACT offset of column c's F_ATT object of sub-step u of layer L (b: of the second transaction buffer)."""
        return (self.fatt2 if b else self.fatt)[L] + (c * self.ash.u + u) * XB

    def q_object(self, L: int, c: int, s: int) -> int:
        """ACT offset of column c's q object of slot s of layer L."""
        return self.fatt[L] + (COLS * self.ash.u + c * self.SLOTS + s) * XB

    @property
    def a_bds(self) -> int:
        """The host's BDs on A."""
        return getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _f32(self, name: str) -> np.ndarray:
        return np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

    def _load(self) -> None:
        """The norm weights (NORM), the attention layers' F_ATT objects (their kind; a full layer's: no window)."""
        norm = self.dev.view("NORM")
        for (L, name), off in self.norm_off.items():
            w = self._f32("output_norm.weight" if L < 0 else f"blk.{L}.{name}.weight").astype(bfloat16)
            norm[off:off + 2 * self.H] = w.view(np.uint8)
        self.dev.flush("NORM")
        act = self.dev.view("ACT")
        for L in self.fatt:
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_KIND], m[M_WREC] = F_ATT, KIND_SWA if self.cfg.kinds[L] == "swa" else KIND_FULL, -1
            for b in range(2):
                for c in range(COLS):
                    for u in range(self.ash.u):
                        o = self.fatt_object(L, c, u, b)
                        act[o:o + 4 * META] = m.view(np.uint8)

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _weights(self, seq: Sequence, phase: tuple[int, str], gi: int) -> None:
        """The A fills of a phase's K group gi: its norm-weight objects (the first group of a normed phase), its tiles
        (a column-major image: none, ``_runs``)."""
        if self.image.column_major:
            return
        dev, (L, name) = self.dev, phase
        region = self.image.region(L, name)
        gr = mp.groups(region, FLAGS[name], 1)[gi]
        for c in range(COLS):
            if gi == 0 and FLAGS[name] & F_NORM:  # (object n: T bytes from weight 4096 n on)
                key = (-1, "output_norm") if L < 0 else (L, NORMS[name])
                seq.fill("A", c, dev.address("NORM", self.norm_off[key]), self.n_norm * self.T,
                         (self.T // 16, 4, self.T // 16, NX_PER_NORM * OBJ_K // 2))
            for o, n, dims, rep, it in mp.a_reads(region, c, gr):
                seq.fill("A", c, dev.address("W", o), n, dims, rep, it)

    def _phase(self, seq: Sequence, phase: tuple[int, str], S: int, to: tuple[int, str] | str | None) -> None:
        """A phase of the step: per K group its weights and inputs (after the drains so far: its producers'); its
        outputs (the last group's) -> ``to``."""
        name = phase[1]
        h8, ffn = self.H // COLS, self.cfg.intermediate_size // COLS
        for gi, rg in enumerate(self.inp[phase]):
            self._weights(seq, phase, gi)
            if gi == 0:
                seq.sync()
            self._inputs(seq, rg, S, phase)
        if to is None:
            return
        for c in range(COLS):
            if name == "gate_up":
                self._values_to(seq, c, to, e0=ffn * c, n=ffn, S=S)
            elif name == "lm":
                self._logits_out(seq, c, S)
            else:  # the residual
                self._values_to(seq, c, to, e0=h8 * c, n=h8, S=S, whole=True)

    def _runs(self, lm: bool) -> list[list[tuple[int, str]]]:
        """A column-major step's weight runs: its phases up to the first attention's attn_in, then from each attention's
        attn_out up to the next one's attn_in (or the end)."""
        out: list[list[tuple[int, str]]] = [[]]
        for phase in self._phases():
            if phase[1] == "lm" and not lm:
                break
            if phase[1] == "attn_out":
                out.append([])
            out[-1].append(phase)
        return [r for r in out if r]

    def _build(self, S: int, lm: bool, subs: tuple[tuple[int, int, int], ...], b: int) -> AttnStep:
        """The transaction of a step of S slots (``lm``: with the LM head; ``subs``: the attention's one pass)."""
        dev = self.dev
        seq = Sequence(self.op, inputs=("X",))
        st = AttnStep(S, subs)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * self.SLOTS * XB), S * XB)
        runs = self._runs(lm) if self.image.column_major else []
        if runs:
            self._weight_run(seq, runs[0])
        for i, L in enumerate(self.layers):
            run = next((r for r in runs if r[0] == (L, "attn_out")), None)
            self.attn_steps.build(seq, st, L, S, run)
            self._phase(seq, (L, "attn_out"), S, (L, "gate_up"))
            nxt = self._input(self.layers[i + 1]) if i + 1 < len(self.layers) else (-1, "lm")
            self._phase(seq, (L, "gate_up"), S, (L, "down"))
            self._phase(seq, (L, "down"), S, nxt)
        if lm:
            self._phase(seq, (-1, "lm"), S, "LOGITS")
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        st.patch = self.attn_steps.patch_tables(st)
        return st

    def subs(self, p: int, S: int) -> tuple[tuple[int, int, int], ...]:
        """The attention's one pass: (first slot, slots, no split)."""
        return ((0, S, 0),)

    def _join(self, parts: tuple[Part, ...]) -> tuple[tuple[int, int, int], ...]:
        """(a pass per part: one transaction for every way to deal the step's slots, ``attn256_steps.pass_subs``; a
        step of one part: that sequence's)"""
        return self.subs(parts[0].position, parts[0].n) if len(parts) == 1 else pass_subs(len(parts))

    def _joint_subs(self, S: int) -> set[tuple]:
        """(a pattern per number of sequences in the step)"""
        return {pass_subs(k) for k in range(2, min(S, self.sequences) + 1)}

    def _transaction(self, key: StepKey) -> AttnStep:
        """(the second buffer's: a copy of the first's whose F_ATT fills read that buffer's objects)"""
        if key.b and key not in self._steps:
            st = self._transaction(key._replace(b=0))
            words = np.array(st.tr.words)
            at = np.array([f[0] for f in st.fatt])
            objs = np.array([self.fatt_object(L, c, u, 1) for _, L, c, u in st.fatt])
            self._bd_words(words, at, self.dev.address("ACT") + objs, np.full(len(at), XB, np.int64))
            self._steps[key] = dataclasses.replace(st, tr=Transaction(self.dev, words), b=1)
        return super()._transaction(key)

    def _set_position(self, st: AttnStep, p: int, ahead: bool = False) -> bool:
        """The attention's F_ATT objects and cache, ring and record transfers of a step at p (``AttentionSteps``), with
        its slots' angles: a full layer's 32 pairs fp32 [cos | sin], a sliding-window layer's 128 bf16. ``ahead``: not
        when a prefill's caches still owe the rings their rows (KV, which the running step reads)."""
        if ahead and self.vring.from_prefill & set(self.fatt):
            return False
        self.attn_steps.set_position(st, p, self._cs(np.arange(p, p + st.S)))
        return True

    def _cs(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """The F_ATT objects' payloads of slots at positions ``pos``, per layer kind (``_set_position``)."""
        cfg = self.cfg
        return {"full": cos_sin_rows(cfg.rope_theta(pos, "full")).view(np.uint8),
                "swa": cos_sin_rows(cfg.rope_theta(pos, "swa")).astype(bfloat16).view(np.uint8)}

    def _set_parts(self, st: AttnStep, parts: tuple[Part, ...], ahead: bool = False) -> bool:
        """``_set_position`` of a step of several sequences' slots: a pass per part, by its sequence and position
        (``AttentionSteps.set_parts``)."""
        if ahead and any(self._vrings[q.seq].from_prefill & set(self.fatt) for q in parts):
            return False
        pos = np.concatenate([np.arange(q.position, q.position + q.n) for q in parts])
        self.attn_steps.set_parts(st, parts, self._cs(pos))
        return True

    def _lm_phase(self, seq: Sequence) -> None:
        if self.image.column_major:
            self._weight_run(seq, [(-1, "lm")])
        self._phase(seq, (-1, "lm"), 1, "LOGITS")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, position: int | tuple[Part, ...], *, logits: bool = True) -> np.ndarray | None:
        """One step of S = len(x) <= 8 token slots: x (S, hidden) the first layer's inputs (embeddings) at positions
        position .. position + S - 1 of the selected sequence (the positions before are in the caches), or of several
        sequences' slots: ``position`` their parts (``decode.step.Part``: x's rows in the parts' order, each
        sequence's at its own next positions). Returns the slots' logits (S, vocab) bf16 (a view of the NPU's buffer,
        valid until the next step), or without ``logits`` None."""
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= self.SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._write_loadr(x)
        first = self._input(self.layers[0])
        self._write_input(self.inp[first][0], interleave(x, self._sp(S, first)).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def _ran(self, p: int, S: int, seq: int | None = None) -> None:
        vring = self.vring if seq is None else self._vrings[seq]
        for L in self.fatt:
            vring.ran(L, p, S)

    # ------------------------------------------------------------------
    # A sequence's state
    # ------------------------------------------------------------------

    def cache_rows(self) -> list[CacheRows]:
        """(the full layers' caches, per layer and KV head: groups of 32 positions, 2 objects each; the rings:
        ``snapshot``)"""
        rows = [self.kv_off[L] + h * self.ash.head_bytes for L in self.kv_off if not self.ring_groups(L)
                for h in range(self.ash.kvh)]
        return [CacheRows(np.array(rows, np.int64), attn.OBJS * self.T, attn.POS_PER_GROUP)] if rows else []

    @property
    def window(self) -> int:
        """The sliding-window layers' window (positions before a row)."""
        return self.cfg.sliding_window

    def _window_groups(self, position: int) -> range:
        """The groups of the sliding-window layers' window before ``position`` and of the open records at it."""
        return range(max(position - self.window, 0) // attn.POS_PER_GROUP, position // attn.POS_PER_GROUP + 1)

    def snapshot(self, position: int) -> Snapshot:
        """(and the sliding-window layers' rings' groups of the window before ``position``: [layer][KV head][group]
        [2 T] from ``_window_groups``' first)"""
        out = super().snapshot(position)
        swa = [L for L in self.kv_off if self.ring_groups(L)]
        gs = self._window_groups(position)
        gb, kv = attn.OBJS * self.T, self.dev.view("KV")
        a = host_bytes((len(swa), self.ash.kvh, len(gs), gb))
        for i, L in enumerate(swa):
            for h in range(self.ash.kvh):
                base = self.kv_off[L] + h * self.attn_steps.head_bytes(L)
                for j, g in enumerate(gs):
                    o = base + (g % self.nr) * gb
                    self.dev.fetch("KV", o, gb)
                    a[i, h, j] = kv[o:o + gb]
        out.extra["window"] = (gs.start, a)
        return out

    def restore(self, snapshot: Snapshot) -> None:
        super().restore(snapshot)
        g0, a = snapshot.extra["window"]
        swa = [L for L in self.kv_off if self.ring_groups(L)]
        gb, kv = attn.OBJS * self.T, self.dev.view("KV")
        for i, L in enumerate(swa):
            for h in range(self.ash.kvh):
                base = self.kv_off[L] + h * self.attn_steps.head_bytes(L)
                for j in range(a.shape[2]):
                    o = base + ((g0 + j) % self.nr) * gb
                    kv[o:o + gb] = a[i, h, j]
                    self.dev.flush("KV", o, gb)

    def tap_rows(self, taps: tuple[int, ...], n: int) -> np.ndarray:
        """The x of the layers ``taps``' first phase (``tap_x``) of the last step's first n slots: (n, taps, K) bf16
        bits, copies (a DSpark drafter's injection of a sequence's rows, ``dspark.DSparkOverlay.run``)."""
        out = []
        for L in taps:
            rg = self.inp[self._input(L)][0]
            sp = self._sp(self._last_s, self._input(L))
            off = rg.off + XB
            self.dev.fetch("ACT", off, 2 * sp * rg.K)
            v = self.dev.view("ACT")[off:off + 2 * sp * rg.K].view(np.uint16).reshape(-1, sp, 8)[:, :n]
            out.append(v.transpose(1, 0, 2).reshape(n, rg.K))
        return np.stack(out, 1)

    def tap_x(self, L: int) -> int:
        """ACT offset of the x of layer L's first phase (L = -1: the LM head): the residual stream that layer read, as
        [hidden / 8][slot][8] bf16 for the slots ``_sp`` of its step."""
        return self.inp[self._input(L)][0].off + XB

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L (-1: the LM head) read in the last step, of a slot (bf16 values, fp32)."""
        off, sp = self.tap_x(L), self._sp(self._last_s, self._input(L))
        self.dev.fetch("ACT", off, 2 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)[:, slot].reshape(-1)
        return v.view(bfloat16).astype(np.float32)
