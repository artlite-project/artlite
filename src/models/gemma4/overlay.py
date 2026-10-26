"""A Gemma 4 model on its decode overlay (``gemma4/decode.py``): buffers, the transaction of a step of S <= 4 token
slots, the step.

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (``attn``: per
layer kind and sub-step its positions and cache ranges, the RoPE tables) and the slots' residual slices (F_LOADR: the
embeddings, fp32), then per layer L (column c, KV head h = c / 4):

- qkv: the input norm (layer 0: of x; later layers: a header, below), the rows of query head c and the column's
  quarter of [K_h; V_h] -> the KV head's attention region [q objects SLOTS | raw K / V objects SLOTS];
- att: the attention cores (``gemma4/attn.py``) on the cache of layer ``kv_source(L)``, per sub-step (``subs``: its
  q and K / V objects, the first q object its F_ATT object) -> o (the chain ends) into o's input, the new positions'
  records (column 4 h + 3) into the cache;
- o -> F_VOUT; gate_up: header (post_attention_norm) + ffn_norm, gelu(gate) up -> down's input (2 K groups);
  down -> F_VOUT; gate: header (post_ffw_norm, no input norm), gelu(y) pli -> proj's input (column slots of 64);
  proj -> F_VOUT;

and the LM head: header (the last layer's post norm and output scale) + output_norm -> LOGITS [slot][vocab] bf16.

The residual stream (``decode.design``: post-norm updates): each column keeps its slice in fp32; an F_VOUT phase (the
output projections) leaves its rows v in fp32 (pend) and drains them (bf16) with its partial sums of squares into the
next header's input region [meta | sums | x: [k block][v | r][slot SP][8]]; a header's main core updates its slice
exactly, (res + v / rms(v) w_post) scale, and drains it (bf16) into the next header's r slots (the first header's
from the host: x), while its GEMV input is (r + v / rms(v) w_post) w1 from those bf16 copies. Norm-weight objects (in
the image, before their phase's tiles: ``image.GemmaImage.norms``) of a header: [w1 2048 | w_post 2048] per 8 input
objects.

Phase inputs are packed (``decode.design.packed_slots``): SP = 1, 2, 4 slots per k block in a step of S = 1, 2, 3 / 4
slots, so a step of one token reads a quarter of the 4-slot layout's input bytes.

The step's transaction is built once per S and pattern of sub-steps (and LM head or not) and patched per step: each
column's range of each layer's cache per sub-step (A: address, length), a chain end's tail objects (A: the ring slots
of the sub-step's open records, or after a prefill the V panels of the first: ``tail_mode``) and the record drains of
column 4 h + 3 (D: address, length). A column's weights between two attentions are one run of the (column-major)
image: one A task.

Each own-cache layer keeps per KV head a ring of two slots (``ring``, after the caches) with the exact V values of its
open records (``attn``); the host tracks which records they hold (``decode.vring.VRing``: a step continues from the
last one's positions, at most its own back, as the generation loop's rejected drafts do).
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

from artlite.decode import kv
from artlite.decode.design import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_PLI,
    F_POST,
    F_PZERO,
    F_TOK,
    F_VOUT,
    M_FLAGS,
    M_NASKIP,
    M_NS,
    M_SCALE,
    M_SKIP,
    M_TOK_KIND,
    M_TOK_NG,
    META,
    SLOTS,
    XB,
    XPAY,
    hidden_meta,
    interleave,
    packed_slots,
    post_region_bytes,
)
from artlite.decode.model import InputRegion, ModelOverlay
from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig
from artlite.operators.design import COLS
from artlite.operators.util import bf16_bits, bf16_value
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import attn as AT
from .decode import GemmaDecodeOperator
from .image import TILE_BYTES, GemmaImage

PHASES = ("qkv", "att", "o", "gate_up", "down", "gate", "proj")
GUARD = 16  # cache groups before a sliding-window cache's first one (the prefill attention reads its window's 16
# groups before a position from there on, also before position 0: positions it masks)
KV_PAD = 2048  # positions past the caches (the last layer's): the prefill's sub-chunk slices of a cache
D_BYTES = 256  # an attention core's D object


@dataclass
class _Step:
    """The transaction of a step of S slots (its attentions' sub-steps per layer kind: ``subs``) and its words that
    follow the position."""

    S: int
    subs: tuple[tuple[tuple[int, int], ...], ...]  # (per layer kind, 0 SWA, 1 full: (first slot, slots) per sub-step)
    tr: Transaction | None = None
    fills: list[tuple[int, int, int, int]] = field(default_factory=list)  # (BD word index, layer, column, sub-step)
    # (BD word index, layer, column, sub-step, tail object)
    tails: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    # (handle, layer, KV head, sub-step, i-th record)
    recs: list[tuple[list[int], int, int, int, int]] = field(default_factory=list)
    patch: dict[str, np.ndarray] | None = None  # (the lists as arrays: _patch_tables)


class GemmaOverlay(ModelOverlay):
    """A Gemma 4 GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices; the
    first one reads x), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the full-attention caches'
    capacity, ``kv_k``: the caches' K precision ("hilo": bfp16 hi + lo blocks; "single": one block per 8 dims),
    ``weights``: the image's buffer allocated ahead."""

    CONFIG, IMAGE = GemmaConfig, GemmaImage

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "hilo", weights: WeightBuffer | None = None):
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights)
        cfg, img = self.cfg, self.image
        assert self.T == TILE_BYTES
        if kv_k not in ("hilo", "single"):
            raise ValueError(f"kv_k: {kv_k!r} (hilo, single)")
        self.at = at = AT.GemmaAttn(cfg, kp=2 if kv_k == "hilo" else 1)
        self.G = at.kp + 1  # (cache objects per group)
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        self.op = GemmaDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, eps=cfg.rms_norm_eps,
                                      attn=at)
        pg_s, pg_f = at.pg[256], at.pg[512]
        assert GUARD * pg_s == cfg.sliding_window
        self.n_full = -(-max_positions // pg_f)
        self.max_positions = self.n_full * pg_f
        self.n_swa = GUARD + -(-self.max_positions // pg_s)  # (a sliding-window cache: linear, after its guard)
        self.period = max(at.pg.values())  # (the steps' sub-steps repeat with the groups)
        self._pli_idx: np.ndarray | None = None
        self._trec: dict[tuple[int, int], int] = {}  # (M_ATREC as the F_ATT objects hold it, per layer and first slot)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order, from layer ``start`` on (``lm`` False: up to the last attention that writes a
        cache)."""
        out = [(L, n) for L in self.layers if L >= self.start for n in PHASES
               if n not in ("gate", "proj") or self.cfg.per_layer_dim]
        if not lm:
            own = [L for L in self.layers if self.cfg.kv_source(L) == L]
            return out[:out.index((own[-1], "att")) + 1] if own else []
        return [*out, (-1, "lm")]

    def _flags(self, L: int, name: str) -> int:
        # (a step from a later layer on: its qkv header zeroes pend, the update it adds: ``step(start=)``)
        return {"qkv": F_NORM | (F_POST if L != self.layers[0] else 0) | (F_PZERO if L == self.start != self.layers[0]
                                                                         else 0), "o": F_VOUT | F_PZERO | F_EMITR,
                "gate_up": F_POST | F_NORM | F_GU, "down": F_VOUT | F_PZERO | F_EMITR, "gate": F_POST | F_PLI,
                "proj": F_VOUT | F_PZERO | F_EMITR, "lm": F_POST | F_NORM}[name]

    def _scale(self, L: int, name: str) -> float:
        sc = self.image.norms(L, name)[2]
        return self._scalar(sc) if sc else 1.0

    def _plan(self) -> None:
        """Byte offsets in ACT, NORM, KV; the headers' chain (each one's v and r sources)."""
        cfg, H = self.cfg, self.H
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        self.tok = take("ACT", AT.TOKENS * XB)
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name != "att":
                self._add_inputs((L, name), post=bool(self._flags(L, name) & F_POST))
        self.final = take("ACT", post_region_bytes(H))  # (the LM header's updated residual)
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token SMAX kind + a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in self.layers for h in range(AT.KV_HEADS)}
        self.kv_off: dict[int, int] = {}
        for L in self.layers:
            if cfg.kv_source(L) == L:
                self.kv_off[L] = take("KV", AT.KV_HEADS * self._kv_bytes(L))
        for L in self.layers:
            assert cfg.kv_source(L) in self.kv_off, f"layer {L} attends to the cache of layer {cfg.kv_source(L)}"
        self.ring_off = {L: take("KV", AT.KV_HEADS * 2 * self.T) for L in self.kv_off}  # (per KV head: 2 slots)
        take("KV", KV_PAD // min(self.at.pg.values()) * self.G * self.T)
        self._sizes()
        # the headers in order; v: the F_VOUT phase that feeds each one
        heads = [(L, n) for L, n in self.phases() if n != "att" and self._flags(L, n) & F_POST]
        self.next_header = dict(itertools.pairwise(heads)) | {heads[-1]: None}
        self.v_to: dict[tuple[int, str], tuple[int, str]] = {}
        seq = [x for x in self.phases() if x[1] != "att"]
        for i, (L, n) in enumerate(seq):
            if self._flags(L, n) & F_VOUT:
                self.v_to[(L, n)] = next(x for x in seq[i + 1:] if self._flags(*x) & F_POST)

    @property
    def first_header(self) -> tuple[int, str]:
        """The step's first header (its r slots: x from the host)."""
        return next(x for x in self.phases() if x[1] != "att" and self._flags(*x) & F_POST)

    def _kv_bytes(self, L: int) -> int:
        """Bytes of a KV head's cache of (own-cache) layer L (a sliding-window cache: with its guard)."""
        return (self.n_swa if self.cfg.swa[L] else self.n_full) * self.G * self.T

    def kv_base(self, L: int, h: int) -> int:
        """Byte offset in KV of group 0 of KV head h's cache of (own-cache) layer L."""
        return self.kv_off[L] + h * self._kv_bytes(L) + (GUARD * self.G * self.T if self.cfg.swa[L] else 0)

    def ring(self, L: int, h: int) -> int:
        """Byte offset in KV of KV head h's ring of (own-cache) layer L: 2 slots of T bytes, slot r & 1 the V values of
        record r ([d-block][position 8][8] bf16: ``decode.vring.ring_rows``)."""
        return self.ring_off[L] + 2 * h * self.T

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The token objects' metadata words, the attention layers' objects'."""
        act = self.dev.view("ACT")
        for t in range(AT.TOKENS):
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_TOK_KIND] = F_TOK, t
            act[self.tok + t * XB:][:4 * META] = m.view(np.uint8)
        for L in self.layers:
            self._load_attention(L)

    def _scalar(self, name: str) -> float:
        w = np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)
        if float(w.min()) != float(w.max()):
            raise NotImplementedError(f"{name}: a norm weight per dimension (the attention cores scale the scores by "
                                      f"one constant per layer)")
        return float(w[0])

    def _load_attention(self, L: int) -> None:
        """The metadata words of layer L's KV heads' q and K / V objects: q object a's the F_ATT object of the sub-step
        from a on (static but M_ATREC: ``_tails``). The main core takes the objects one at a time, and after a
        sub-step's last (its last slot's K / V object, or of a layer without them its q object) the cache's objects:
        M_NASKIP = -1 - the token of the slot, whose M_TOK_NG holds those of the sub-step that ends there."""
        cfg, act = self.cfg, self.dev.view("ACT")
        src = cfg.kv_source(L)
        hi, lo = AT.score_constant(self._scalar(f"blk.{L}.attn_q_norm.weight"),
                                   self._scalar(f"blk.{src}.attn_k_norm.weight"))
        kind = int(not cfg.swa[L])
        for h in range(AT.KV_HEADS):
            for a in range(SLOTS):
                t = AT.SMAX * kind + a
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_NASKIP], m[M_SKIP] = AT.F_ATT, -1 - t, 0
                m[AT.M_AHD], m[AT.M_AKIND], m[AT.M_AOWN] = cfg.head_dim(L), kind, int(src == L)
                m[AT.M_QC_HI], m[AT.M_QC_LO], m[AT.M_AWIN] = hi, lo, cfg.sliding_window if cfg.swa[L] else 0
                m[AT.M_ANH], m[AT.M_ATOK] = AT.QH, t
                if src == L:  # (its K / V object last)
                    m[M_NASKIP] = 0
                    kv = np.zeros(META, np.int32)
                    kv[M_NASKIP] = -1 - t
                    o = self.att_reg[(L, h)] + (SLOTS + a) * XB
                    act[o:o + 4 * META] = kv.view(np.uint8)
                o = self.att_reg[(L, h)] + a * XB
                act[o:o + 4 * META] = m.view(np.uint8)

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _meta(self, L: int, name: str, rg: InputRegion, m: np.ndarray) -> np.ndarray:
        """(the residual's sizes; a header's output scale)"""
        hidden_meta(m, self.H)
        if rg.post:
            m[M_SCALE] = bf16_bits(self._scale(L, name))
        return m

    def _post_to(self, seq: Sequence, c: int, off: int, e0: int, n: int, S: int, part: int) -> None:
        """n values per slot of column c (C objects (j, slot)) -> the v (part 0) or r (1) slots of elements e0 .. of a
        header's input at ACT offset off ([k block][v | r][slot SP][8])."""
        nobj, sp = n // C_LEN, packed_slots(S)
        seq.drain("C", c, self.dev.address("ACT", off + XB + XPAY + 4 * sp * e0 + 16 * sp * part), S * 2 * C_LEN,
                  (4, 8, 8 * sp, 4), nobj, (8 * sp * 8, nobj))

    def _outputs(self, seq: Sequence, st: _Step, phase: tuple[int, str]) -> None:
        """The C drains of a GEMV phase (after its last group): a header's updated residual slice (into the next
        header's r slots), then its outputs."""
        L, name = phase
        S, cfg, dev, H = st.S, self.cfg, self.dev, self.H
        h8 = H // COLS
        nxt = self.next_header.get(phase) if self._flags(L, name) & F_POST else None
        for c in range(COLS):
            if self._flags(L, name) & F_POST:
                off = self.inp[nxt][0].off if nxt is not None else self.final
                self._post_to(seq, c, off, h8 * c, h8, S, 1)
            if name == "qkv":
                hd, h, j = cfg.head_dim(L), c // AT.QH, c % AT.QH
                reg = self.att_reg[(L, h)]  # ([q objects SLOTS | raw K / V objects SLOTS])
                seq.drain("C", c, dev.address("ACT", reg + 4 * META + 2 * hd * j), S * 2 * hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
                if cfg.kv_source(L) == L:
                    seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 4 * META + hd * j), S * hd,
                              (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif self._flags(L, name) & F_VOUT:
                tgt = self.inp[self.v_to[phase]][0].off
                self._post_to(seq, c, tgt, h8 * c, h8, S, 0)
                seq.drain("C", c, dev.address("ACT", tgt + XB + 2 * C_LEN * c), 2 * C_LEN)  # (its partial sums)
            elif name == "gate_up":
                ffn = cfg.intermediate_size // COLS
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "gate":
                self._values_to(seq, c, (L, "proj"), e0=64 * c, n=C_LEN, S=S)
            elif name == "lm":
                self._logits_out(seq, c, S)

    def _gemv(self, seq: Sequence, st: _Step, phase: tuple[int, str]) -> None:
        for gi, rg in enumerate(self.inp[phase]):
            if gi == 0:
                seq.sync()  # (its inputs' producers)
            self._inputs(seq, rg, st.S)
        self._outputs(seq, st, phase)

    def _attention(self, seq: Sequence, st: _Step, L: int, weights: list[tuple[int, str]]) -> None:
        """The attention of layer L, per sub-step (``attn.substeps``; a layer with its own cache: each after the last
        one's records are in DDR): per column its cache range (A, patched), the F_ATT object and the sub-step's q and
        K / V objects (X), the records (D, patched) and o (D, into the sub-step's slots of o's input). The weights of
        the GEMV phases after it go before the sync of its inputs: the A stream keeps streaming them while the
        attention waits for them."""
        cfg, dev, S, T = self.cfg, self.dev, st.S, self.T
        hd, src, own = cfg.head_dim(L), cfg.kv_source(L), cfg.kv_source(L) == L
        subs = st.subs[0 if cfg.swa[L] else 1]
        o, sp, nk = self.inp[(L, "o")][0], packed_slots(S), AT.QH * hd // 8  # (o: the KV head's k blocks)
        for u, (a, n) in enumerate(subs):
            if u and own:
                seq.sync()  # (sub-step u - 1's records in DDR: this one reads them)
            for c in range(COLS):  # (its range of the cache and a chain end's tail objects, patched per step)
                st.fills.append((seq.fill("A", c, dev.address("KV", self.kv_off[src]), T), L, c, u))
                if c % AT.QH == AT.QH - 1:  # (one object each: the slots of its records, ``_tails``)
                    st.tails += [(seq.fill("A", c, dev.address("KV", self.ring_off[src]), T), L, c, u, t)
                                 for t in range(AT.tails(n))]
            if u == len(subs) - 1 and weights:
                self._weight_run(seq, weights)
            if u == 0:
                seq.sync()  # (the q and K / V objects in DDR)
            for c in range(COLS):
                h, j = c // AT.QH, c % AT.QH
                reg = self.att_reg[(L, h)] + a * XB
                if own:  # (its q objects, then its K / V objects: half objects, the second strip SLOTS XB on)
                    seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
                else:
                    seq.fill("X", c, dev.address("ACT", reg), n * XB)
                if j < AT.QH - 1:
                    continue
                if own:
                    for i in range(len(kv.emit_order(p=0, S=n, hd=hd, a_bytes=T, kp=self.at.kp))):
                        hdl = seq.drain("D", c, dev.address("KV", self.kv_off[src]), D_BYTES)
                        st.recs.append((hdl, L, h, u, i))
                at = dev.address("ACT", o.off + XB + 2 * sp * 8 * nk * h + 16 * a)  # ([k block][slot sp][8])
                seq.drain("D", c, at, 16 * n * nk, None if n == sp else (4 * n, nk, 4 * sp, 4 * sp * nk))

    def _build(self, S: int, lm: bool, subs: tuple) -> _Step:
        dev = self.dev
        seq = Sequence(self.op, rolling=True, inputs=("X",))
        st = _Step(S, subs)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.tok), AT.TOKENS * XB)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * SLOTS * XB), S * XB)
        ph = self.phases(lm)

        def run(i: int) -> list[tuple[int, str]]:  # (the GEMV phases from i to the next attention: one A fill)
            return ph[i:next((k for k in range(i, len(ph)) if ph[k][1] == "att"), len(ph))]

        for i, (L, name) in enumerate(ph):
            if name == "att":
                self._attention(seq, st, L, run(i + 1))
                continue
            if i == 0:
                self._weight_run(seq, run(0))
            self._gemv(seq, st, (L, name))
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        return st

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """The attentions' sub-steps (first slot, slots) of a step at p .. p + S - 1 per layer kind (0 SWA, 1 full):
        within a pair of records (``attn``: P V adds two records together; a group holds whole pairs)."""
        out = tuple(tuple(AT.substeps(p, S, 16, self.at.slots(hd))) for hd in (256, 512))
        assert max(len(x) for x in out) <= AT.U, (p, S)
        return out

    def _ranges(self, swa: bool, p: int) -> list[tuple[int, int]]:
        """The columns' (first group, groups) of a layer kind's cache for a sub-step whose slots are in the group of
        position p: those of its group (a sliding-window cache: from the GUARD groups before it on, the window of any
        of its positions), the same for each of them in every step."""
        g = p // self.at.pg[256 if swa else 512]
        return kv.ranges(first=max(0, g - GUARD) if swa else 0, g=g, chain=AT.QH)

    def _patch_tables(self, st: _Step) -> dict[str, np.ndarray]:
        """A step's cache fills, tail fills and record drains as arrays: BD word index, layer kind (0 SWA, 1 full), the
        sub-step, the column's place in its chain / the record's index, and the byte offsets in KV of group 0 of the
        layer's KV head (and of its ring); a tail fill's source layer, whether the step runs it."""
        cfg = self.cfg
        f = np.array([(at, 0 if cfg.swa[L] else 1, u, c % AT.QH, self.kv_base(cfg.kv_source(L), c // AT.QH))
                      for at, L, c, u in st.fills], np.int64).reshape(-1, 5)
        t = np.array([(at, 0 if cfg.swa[L] else 1, self.kv_base(cfg.kv_source(L), c // AT.QH),
                       self.ring(cfg.kv_source(L), c // AT.QH), cfg.kv_source(L), cfg.kv_source(L) == L,
                       cfg.kv_source(L) >= self.start, k, u) for at, L, c, u, k in st.tails], np.int64).reshape(-1, 9)
        r = np.array([(hdl[0], 0 if cfg.swa[L] else 1, u, i, self.kv_base(L, h), self.ring(L, h))
                      for hdl, L, h, u, i in st.recs], np.int64).reshape(-1, 6)
        return {"f_at": f[:, 0], "f_kind": f[:, 1], "f_u": f[:, 2], "f_j": f[:, 3], "f_base": f[:, 4],
                "t_at": t[:, 0], "t_kind": t[:, 1], "t_base": t[:, 2], "t_ring": t[:, 3], "t_src": t[:, 4],
                "t_own": t[:, 5].astype(bool), "t_run": t[:, 6].astype(bool), "t_k": t[:, 7], "t_u": t[:, 8],
                "r_at": r[:, 0], "r_kind": r[:, 1], "r_u": r[:, 2], "r_i": r[:, 3], "r_base": r[:, 4:6]}

    def _vrec(self, p: int, swa: bool) -> int:
        """Byte offset from group 0 of a layer's KV head of the V panels of record p >> 3."""
        return kv.v_offset(p, hd=256 if swa else 512, a_bytes=self.T, kp=self.at.kp)

    def tail_mode(self, src: int, *, before: bool) -> bool:
        """Whether a step's tail objects of a layer attending to the cache of ``src`` are the open record's V panels
        (the prefill wrote it, and the step did not run ``src``'s attention before the layer's: ``before`` False), not
        its ring slots."""
        return src in self.vring.from_prefill and not before

    def _tails(self, st: _Step, p: int, pu: np.ndarray) -> None:
        """The step's tail fills (per sub-step a chain end's objects: the ring slots of the records of its first and
        last positions, or for the first the record's V panels) and the F_ATT objects' M_ATREC. pu: per layer kind and
        sub-step (first position, slots)."""
        t, T, S = st.patch, self.T, st.S
        rec = np.array([self.tail_mode(int(src), before=bool(run and not own)) and (u == 0 or not own)
                        for src, own, run, u in zip(t["t_src"], t["t_own"], t["t_run"], t["t_u"])], bool)
        rec &= t["t_k"] == 0  # (record mode: the first sub-step's of a layer with its own cache, the others from the ring)
        q0, n = pu[t["t_kind"], t["t_u"], 0], pu[t["t_kind"], t["t_u"], 1]
        vrec = np.array([[self._vrec(int(x), kind == 0) for x in row] for kind, row in enumerate(pu[:, :, 0])],
                        np.int64)[t["t_kind"], t["t_u"]]
        slot = np.where(t["t_k"] == 0, q0 >> 3, (q0 + n - 1) >> 3) & 1
        address = self.dev.address("KV") + np.where(rec, t["t_base"] + vrec, t["t_ring"] + slot * T)
        self._bd_words(st.tr.words, t["t_at"], address, np.full(len(rec), T, np.int64))
        cfg, act = self.cfg, self.dev.view("ACT")
        for L in self.layers:
            if L < self.start:
                continue
            src = cfg.kv_source(L)
            before = src != L and src >= self.start
            m = int(self.tail_mode(src, before=before))
            if not m and not before:  # (from the ring: the rows before p, or those of the step's positions too)
                self.vring.check(src, p, p if src == L else p + S)
            for a in range(SLOTS):  # (the F_ATT objects of the sub-steps from slot a on)
                ma = m if a == 0 or src != L else 0
                if self._trec.get((L, a)) != ma:
                    for h in range(AT.KV_HEADS):
                        off = self.att_reg[(L, h)] + a * XB
                        act[off:off + 4 * META].view(np.int32)[AT.M_ATREC] = ma
                        self.dev.flush("ACT", off, 4 * META)
                    self._trec[(L, a)] = ma

    def _ran(self, p: int, S: int) -> None:
        """After a step: the rings of the own-cache layers it ran hold the rows of its open records."""
        for L in self.layers:
            if L >= self.start and self.cfg.kv_source(L) == L:
                self.vring.ran(L, p, S)

    def _set_position(self, st: _Step, p: int) -> None:
        """The token objects (per layer kind and sub-step: its positions, the columns' ranges, the step's RoPE tables)
        and the step's cache transfers."""
        S, cfg, dev, T = st.S, self.cfg, self.dev, self.T
        assert 0 <= p and p + S <= self.max_positions, (p, S)
        assert st.subs == self.subs(p, S), (p, S, st.subs)
        act = dev.view("ACT")
        pu = np.zeros((2, AT.U, 2), np.int64)  # (per kind and sub-step: first position, slots)
        fo = np.zeros((2, AT.U, AT.QH, 2), np.int64)  # (per kind, sub-step and column of the chain: its A fill)
        ro = np.zeros((2, AT.U, 6, 3), np.int64)  # (per kind, sub-step and i-th record: base, offset, bytes)
        for kind, swa in ((0, True), (1, False)):
            hd = 256 if swa else 512
            pg = self.at.pg[hd]
            cs = AT.rope_tables(cfg, list(range(p, p + S)), self.at.rot, swa=swa).view(np.uint8).reshape(-1)
            for a in range(SLOTS):  # (each token object copies the step's table; one of no sub-step: no objects)
                o = self.tok + (AT.SMAX * kind + a) * XB
                act[o + 4 * META:o + 4 * META + cs.size] = cs
                m = act[o:o + 4 * META].view(np.int32)
                m[M_NS], m[M_TOK_NG:M_TOK_NG + COLS], m[AT.M_TOK_NA:AT.M_TOK_NA + COLS] = 0, 0, 0
            for u, (a, n) in enumerate(st.subs[kind]):
                r = self._ranges(swa, p + a)
                o = self.tok + (AT.SMAX * kind + a) * XB
                m = act[o:o + 4 * META].view(np.int32)
                me = act[o + (n - 1) * XB:][:4 * META].view(np.int32)  # (the token of its last slot)
                m[M_NS], m[AT.M_POS], m[AT.M_TOK_A] = n, p + a, a
                for c in range(COLS):  # (every KV head's columns: the same ranges)
                    g0, ng = r[c % AT.QH]
                    no = kv.fill(g0, ng, self.at.kp)[1] + (AT.tails(n) if c % AT.QH == AT.QH - 1 else 0)
                    m[AT.M_TOK_G0 + c], m[AT.M_TOK_NA + c], me[M_TOK_NG + c] = g0 * pg, self.G * ng | no << 16, no
                pu[kind, u] = p + a, n
                fo[kind, u] = [kv.fill(*rj, self.at.kp) for rj in r]
                rec = kv.records(p=p + a, S=n, hd=256 if swa else 512, a_bytes=T, kp=self.at.kp)
                ro[kind, u, :len(rec)] = rec
        dev.flush("ACT", self.tok, AT.TOKENS * XB)
        if st.patch is None:
            st.patch = self._patch_tables(st)
        t, kv0 = st.patch, dev.address("KV")
        f = fo[t["f_kind"], t["f_u"], t["f_j"]]  # (first object, objects)
        self._bd_words(st.tr.words, t["f_at"], kv0 + t["f_base"] + f[:, 0] * T, f[:, 1] * T)
        self._tails(st, p, pu)
        # the record drains (from the KV head's group 0 or ring, offset, bytes)
        r = ro[t["r_kind"], t["r_u"], t["r_i"]]
        base = np.take_along_axis(t["r_base"], r[:, :1], 1)[:, 0]
        self._bd_words(st.tr.words, t["r_at"], kv0 + base + r[:, 1], r[:, 2])
        st.tr.flush()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: torch.Tensor, pli: torch.Tensor | None, position: int, *, logits: bool = True,
             start: int | None = None) -> torch.Tensor | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the scaled embeddings, pli (S, layers, per_layer_dim)
        the per-layer inputs, at positions position .. position + S - 1 (the positions before are cached). Returns the
        raw logits (S, vocab) bf16 (before the soft cap; a view of the NPU's buffer, valid until the next step), or
        without ``logits`` None (only the caches are updated). ``start``: the step from that layer on, x its input (e.g.
        a prompt's last token after a prefill chunk: ``GemmaPrefill.residual``): its qkv phase is a header whose update
        adds nothing (F_PZERO: pend = 0; v and the partial sums 0: 1 / rms(v) stays finite) to res = x / (the previous
        layer's output scale), so the header's residual is x."""
        x = x.reshape(-1, self.H).float()
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        dev, H = self.dev, self.H
        start = self._mode(start)
        if self._metas_s != (S, start):
            self._metas(S)
        act = dev.view("ACT")
        if start != self.layers[0]:  # (res: what the header scales back to x, with the scale's bf16 bits as M_SCALE)
            x = x / bf16_value(self._scale(start, "qkv"))
        xf = x.numpy()
        self._write_loadr(xf)
        sp = packed_slots(S)
        xi = interleave(xf, sp).view(np.uint8).reshape(-1)
        if start == self.layers[0]:  # (its qkv: a plain input)
            self._write_input(self.inp[(start, "qkv")][0], xi)
        else:  # (its qkv: the first header, v and the partial sums 0)
            act[self.inp[(start, "qkv")][0].off + XB:][:XPAY] = 0
        h0 = self.inp[self.first_header][0].off + XB + XPAY  # (the first header's r slots: x)
        hx = act[h0:h0 + 4 * sp * H].reshape(H // 8, 2, 16 * sp)
        hx[:, 1] = xi.reshape(H // 8, 16 * sp)
        if start != self.layers[0]:
            hx[:, 0] = 0
        dev.flush("ACT", h0 - XPAY, XPAY + 4 * sp * H)
        if self.cfg.per_layer_dim and pli is not None:
            self._write_pli(pli.reshape(S, -1, self.cfg.per_layer_dim))
        return self._run(S, position, logits)

    def _write_pli(self, pli: torch.Tensor) -> None:
        """Each layer's gate meta object: [slot 4][column 8][32] bf16 of the slots' per-layer inputs."""
        S, d = pli.shape[0], self.cfg.per_layer_dim
        b = pli.bfloat16().view(torch.int16).numpy().view(np.uint16)  # (S, layers, ple)
        offs = [self.inp[(L, "gate")][0].off + 4 * META for L in self.layers]
        if self._pli_idx is None:  # (per layer its slots' elements in ACT, as uint16 indices)
            self._pli_idx = (np.array(offs)[:, None, None] // 2 + d * np.arange(SLOTS)[None, :, None]
                             + np.arange(d)[None, None, :])
        self.dev.view("ACT").view(np.uint16)[self._pli_idx[:, :S]] = b[:, self.layers].transpose(1, 0, 2)
        for off in offs:
            self.dev.flush("ACT", off, 2 * SLOTS * d)

    def read_kv(self, layer: int, positions: list[int] | range) -> tuple[torch.Tensor, torch.Tensor]:
        """K, V (kv heads, n, head_dim) fp32 as cached for the positions of an own-cache layer: K normed and rotated
        (without the k_norm weight), V normed."""
        cfg = self.cfg
        hd = cfg.head_dim(layer)
        pos = np.asarray(positions)
        n = int(pos.max()) // self.at.pg[hd] + 1  # (the groups that hold the positions)
        self.dev.fetch("KV", self.kv_off[layer], AT.KV_HEADS * self._kv_bytes(layer))
        ks, vs = [], []
        for h in range(AT.KV_HEADS):
            b = self.kv_base(layer, h)
            k, v = kv.kv_from_groups(self.dev.view("KV")[b:b + n * self.G * self.T].reshape(n, self.G, self.T), hd)
            ks.append(k[pos])
            vs.append(v[pos])
        return torch.from_numpy(np.stack(ks)), torch.from_numpy(np.stack(vs))

    def residual(self, slot: int = 0) -> torch.Tensor:
        """The last step's residual stream after the last layer (the LM header's update), bf16 values as fp32."""
        off, sp = self.final + XB + XPAY, packed_slots(self._last_s)
        self.dev.fetch("ACT", off, 4 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 4 * sp * self.H].view(np.uint16).reshape(-1, 2, sp, 8)[:, 1, slot]
        return torch.from_numpy(v.reshape(-1).copy()).view(torch.bfloat16).float()


__all__ = ["GemmaOverlay"]
