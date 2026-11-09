"""An LFM3 model on its decode overlay (``lfm/decode.py``): buffers, the transaction of a step of S <= 4 token slots,
the step.

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (per layer kind and
sub-step its positions and cache ranges, the RoPE tables; token object 0 also the step's slots and the conv shift) and
the slots' residual slices (F_LOADR: the embeddings, fp32), then per layer L (column c, KV head h = c / 2, j = c % 2):

- conv layers: in (the input norm: layer 0 of x, later layers a header, below) -> the rows B, C, x of the column's 256
  channels into its conv region (``lfm.conv``); conv (the operator cores: the region's taps, history and rows on X ->
  y into out's input, the new history over the old, on D); out -> F_VOUT;
- attention layers: qkv (a header) -> the KV head's attention region [q objects SLOTS | raw K / V objects SLOTS]: the
  column's two query heads at heads 2 j, 2 j + 1 of each q object, its halves of K_h and V_h (dims 64 j ..) in the raw
  K / V objects, its two heads' gate logits at the q objects' ``Attn.goff`` + 64 j; att (the operator cores, a chain of
  2 per KV head, per sub-step ``subs``) -> o into o's input, the new positions' records (column 2 h + 1) into the
  cache; o -> F_VOUT;
- gate_up: header (post_attention_norm) + ffn_norm, silu(gate) up -> down's input (2 K groups); down -> F_VOUT;

and the LM head: header (the last layer's post_ffw_norm) + output_norm -> LOGITS [slot][vocab] bf16.

The residual stream (``decode.design``: post-norm updates, Gemma's form, output scale 1): each column keeps its slice in
fp32; an F_VOUT phase leaves its rows v in fp32 (pend) and drains them (bf16) with its partial sums of squares into the
next header's input region [meta | sums | x: [k block][v | r][slot SP][8]]; a header's main core updates its slice
exactly, res + v / rms(v) w_post, and drains it (bf16) into the next header's r slots (the first header's from the host:
x), while its GEMV input is (r + v / rms(v) w_post) w1 from those bf16 copies.

The KV caches (KV): per attention layer and KV head linear groups of kp + 1 objects of the tile size T (``decode.kv``:
64 positions per group at head_dim 128), a sliding-window layer's after a guard of the window's groups (``GUARD``: the
prefill reads a window before a position from there on); then per layer and KV head a ring of two slots of T bytes with
the exact V values of the open records (``decode.vring``).

The conv regions (ACT, per conv layer and column, ``lfm.conv``) hold the layer's taps and its history: the 6 rows before
the last step's first token and that step's rows; a step takes its window from the accepted count of the last one on
(``M_CSHIFT``: its first position minus the last step's), so the host patches nothing per conv layer; a step that does
not continue the last one (a reset, a restore, after a prefill) starts from a history whose base is right before it
(``conv_at``).

The step's transaction is built once per S and pattern of sub-steps (and LM head or not) and patched per step: each
column's range of each layer's cache per sub-step (A: address, length), a chain end's tail objects (A: the ring slots of
the sub-step's open records, or after a prefill the V panels of the first: ``tail_mode``) and the record drains of
column 2 h + 1 (D: address, length). A column's weights between two operator steps are one run of the (column-major)
image: one A task.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

from artlite.decode import kv
from artlite.decode.attn import (
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
    SMAX,
    Attn,
    U,
    rope_tables,
    score_bits,
    substeps,
    tails,
)
from artlite.decode.design import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
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
from artlite.model_config import LfmConfig
from artlite.operators.design import COLS
from artlite.operators.util import bf16_bits
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import conv as CV
from .decode import LfmDecodeOperator
from .image import CH, LfmImage

PHASES = {"conv": ("in", "conv", "out", "gate_up", "down"), "attn": ("qkv", "att", "o", "gate_up", "down")}
OPS = ("att", "conv")  # (the operator cores' phases)
KV_K = {"single": 1, "hilo": 2}  # K precision of the caches (``kv_k``): K panels per record
CHAIN = 2  # columns per KV head
TOKENS = 2 * SMAX  # token objects: t = SMAX kind + a, of the sub-step from slot a on (kind 0 SWA, 1 full)
KV_PAD = 4096  # positions past the caches (the last layer's): the prefill's slices of a cache
KINDS = ("swa", "full")


@dataclass
class _Step:
    """The transaction of a step of S slots (its attentions' sub-steps: ``subs``) and its words that follow the
    position."""

    S: int
    subs: tuple[tuple[int, int], ...]  # (first slot, slots) per sub-step (both kinds alike: one head dim)
    tr: Transaction | None = None
    b: int = 0
    fills: list[tuple[int, int, int, int]] = field(default_factory=list)  # (BD word index, layer, column, sub-step)
    # (BD word index, layer, column, sub-step, tail object)
    tails: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    # (handle, layer, KV head, sub-step, i-th record)
    recs: list[tuple[list[int], int, int, int, int]] = field(default_factory=list)
    patch: dict[str, np.ndarray] | None = None  # (the lists as arrays: _patch_tables)


class LfmOverlay(ModelOverlay):
    """An LFM3 GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices; the first
    one reads x), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the caches' capacity, ``kv_k``: the
    caches' K precision ("single": one bfp16 block per 8 dims, "hilo": hi + lo blocks), ``weights``: the image's buffer
    allocated ahead."""

    CONFIG, IMAGE = LfmConfig, LfmImage

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "single", weights: WeightBuffer | None = None):
        if kv_k not in KV_K:
            raise ValueError(f"kv_k: {kv_k!r} (one of {', '.join(KV_K)})")
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights)
        cfg, img = self.cfg, self.image
        assert cfg.num_key_value_heads * CHAIN == COLS and cfg.conv_width == CV.WIDTH
        self.hd, self.nh, self.kp = cfg.head_dim, cfg.q_per_kv, KV_K[kv_k]
        self.at = at = Attn(hds=(self.hd,), rot=(cfg.rot("swa"), cfg.rot("full")), chain=CHAIN, heads=self.nh,
                            kp=self.kp, qk_norm=True, qk_weights=True, eps=cfg.rms_norm_eps, a_bytes=self.T, gate=True,
                            conv=CV.ConvJob())
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        gu = {b.R for L in self.layers for b in img.region(L, "gate_up").blocks}
        self.op = LfmDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                    eps=cfg.rms_norm_eps, attn=at, gu_win=gu != {16})
        self.G = at.kp + 1  # (cache objects per group)
        self.rec = kv.Records(self.hd, self.kp)
        self.pg = at.pg[self.hd]  # positions per cache group
        assert cfg.sliding_window % self.pg == 0
        self.guard = cfg.sliding_window // self.pg  # (groups before a sliding-window cache's first one)
        self.n_full = -(-max_positions // self.pg)
        self.max_positions = self.n_full * self.pg
        self.n_swa = self.guard + self.n_full
        self.period = 16  # (the steps' sub-steps repeat with the record pairs)
        self._trec: dict[tuple[int, int], int] = {}  # (M_ATREC as the F_ATT objects hold it, per layer and first slot)
        self.inv_freq = {k: cfg.rope_inv_freq(k) for k in KINDS}
        self.conv_p, self.conv_s = 0, 0  # (the conv histories: the last step's first position and slots: ``conv_at``)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def kind(self, L: int) -> str:
        return self.cfg.kinds[L]

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order (``lm`` False: without the LM head)."""
        out = [(L, n) for L in self.layers for n in PHASES["conv" if self.kind(L) == "conv" else "attn"]]
        return out + ([(-1, "lm")] if lm else [])

    def _flags(self, L: int, name: str) -> int:
        if name in ("in", "qkv"):
            return F_NORM | (F_POST if L != self.layers[0] else 0)
        return {"out": F_VOUT | F_PZERO | F_EMITR, "o": F_VOUT | F_PZERO | F_EMITR, "down": F_VOUT | F_PZERO | F_EMITR,
                "gate_up": F_POST | F_NORM | F_GU, "lm": F_POST | F_NORM}[name]

    def _plan(self) -> None:
        """Byte offsets in ACT and KV; the headers' chain (each one's v and r sources)."""
        H = self.H
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        self.tok = take("ACT", TOKENS * XB)
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name not in OPS:
                self._add_inputs((L, name), post=bool(self._flags(L, name) & F_POST))
        self.final = take("ACT", post_region_bytes(H))  # (the LM header's updated residual)
        att = [L for L in self.layers if self.kind(L) != "conv"]
        convs = [L for L in self.layers if self.kind(L) == "conv"]
        kvh = self.cfg.num_key_value_heads
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token SMAX kind + a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in att for h in range(kvh)}
        self.conv_reg = {(L, c): take("ACT", CV.region_bytes()) for L in convs for c in range(COLS)}
        self.kv_off = {L: take("KV", kvh * self._kv_bytes(L)) for L in att}
        r0 = self._layout["KV"]
        self.ring_off = {L: take("KV", kvh * 2 * self.T) for L in att}  # (per KV head: 2 slots)
        self.rings = (r0, self._layout["KV"] - r0)
        take("KV", KV_PAD // self.pg * self.G * self.T)
        self._sizes()
        # the headers in order; v: the F_VOUT phase that feeds each one
        gemv = [x for x in self.phases() if x[1] not in OPS]
        heads = [x for x in gemv if self._flags(*x) & F_POST]
        self.next_header = dict(itertools.pairwise(heads)) | {heads[-1]: None}
        self.v_to = {x: next(y for y in gemv[i + 1:] if self._flags(*y) & F_POST)
                     for i, x in enumerate(gemv) if self._flags(*x) & F_VOUT}

    @property
    def first_header(self) -> tuple[int, str]:
        """The step's first header (its r slots: x from the host)."""
        return next(x for x in self.phases() if x[1] not in OPS and self._flags(*x) & F_POST)

    def _kv_bytes(self, L: int) -> int:
        """Bytes of a KV head's cache of attention layer L (a sliding-window cache: with its guard)."""
        return (self.n_swa if self.kind(L) == "swa" else self.n_full) * self.G * self.T

    def kv_base(self, L: int, h: int) -> int:
        """Byte offset in KV of group 0 of KV head h's cache of attention layer L."""
        return self.kv_off[L] + h * self._kv_bytes(L) + (self.guard * self.G * self.T if self.kind(L) == "swa" else 0)

    def cache_rows(self) -> list[tuple[np.ndarray, int, int]]:
        """(the sliding-window layers' caches, then the full layers')"""
        kvh = self.cfg.num_key_value_heads
        return [(np.array([self.kv_base(L, h) for L in self.kv_off if self.kind(L) == k for h in range(kvh)]),
                 self.G * self.T, self.pg) for k in KINDS if any(self.kind(L) == k for L in self.kv_off)]

    def ring(self, L: int, h: int) -> int:
        """Byte offset in KV of KV head h's ring of attention layer L: 2 slots of T bytes, slot r & 1 the V values of
        record r ([d-block][position 8][8] bf16: ``decode.vring.ring_rows``)."""
        return self.ring_off[L] + 2 * h * self.T

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The token objects' metadata words, the attention layers' objects', the conv regions' (their taps; their
        histories zero)."""
        act = self.dev.view("ACT")
        for t in range(TOKENS):
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_TOK_KIND] = F_TOK, t
            act[self.tok + t * XB:][:4 * META] = m.view(np.uint8)
        for L in self.layers:
            if self.kind(L) == "conv":
                self._load_conv(L)
            else:
                self._load_attention(L)

    def _load_attention(self, L: int) -> None:
        """The metadata words of layer L's KV heads' q and K / V objects: q object a's the F_ATT object of the sub-step
        from a on (static but M_ATREC: ``_tails``); K / V object a's M_NASKIP = -1 - its token: after it the main cores
        skip the cache objects of the sub-step that ends at slot a. The q / k norms' weights (hd fp32) after the
        payloads the qkv drains write: q_norm after a q object's heads, k_norm after a K / V object's [K | V]."""
        act, cfg = self.dev.view("ACT"), self.cfg
        hi, lo = score_bits(1 / np.sqrt(self.hd))
        kind = KINDS.index(self.kind(L))
        w = {n: np.asarray(self.gguf.tensor(f"blk.{L}.attn_{n}_norm.weight").dequantize(), np.float32).reshape(-1)
             .view(np.uint8) for n in ("q", "k")}
        for h in range(cfg.num_key_value_heads):
            for a in range(SLOTS):
                t = SMAX * kind + a
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_NASKIP], m[M_SKIP] = F_ATT, 0, 0
                m[M_AHD], m[M_AKIND], m[M_AOWN] = self.hd, kind, 1
                m[M_QC_HI], m[M_QC_LO], m[M_AWIN] = hi, lo, cfg.sliding_window if kind == 0 else 0
                m[M_ANH], m[M_ATOK] = self.nh, t
                o = self.att_reg[(L, h)] + a * XB
                act[o:o + 4 * META] = m.view(np.uint8)
                act[o + 4 * META + 2 * self.nh * self.hd:][:w["q"].size] = w["q"]
                k = np.zeros(META, np.int32)
                k[M_NASKIP] = -1 - t
                o = self.att_reg[(L, h)] + (SLOTS + a) * XB
                act[o:o + 4 * META] = k.view(np.uint8)
                act[o + 4 * META + 4 * self.hd:][:w["k"].size] = w["k"]

    def _load_conv(self, L: int) -> None:
        """A conv layer's regions: the first object's metadata (F_CONV: the other cores skip the region), the taps, a
        zero history."""
        act = self.dev.view("ACT")
        w = np.asarray(self.gguf.tensor(f"blk.{L}.shortconv.conv.weight").dequantize(), np.float32)
        taps = CV.tap_objects(w.reshape(self.H, CV.WIDTH))
        for c in range(COLS):
            o = self.conv_reg[(L, c)]
            reg = act[o:o + CV.region_bytes()]
            reg[:] = 0
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_SKIP] = CV.F_CONV, CV.OBJS - 1
            reg[:4 * META] = m.view(np.uint8)
            reg[CV.payload(0):CV.payload(0) + XPAY] = taps[c, :XPAY]
            reg[CV.payload(1):CV.payload(1) + 3 * CV.TAP] = taps[c, XPAY:]

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _meta(self, L: int, name: str, rg: InputRegion, m: np.ndarray) -> np.ndarray:
        """(the residual's sizes; a header's output scale: 1)"""
        hidden_meta(m, self.H)
        if rg.post:
            m[M_SCALE] = bf16_bits(1.0)
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
        S, dev, H = st.S, self.dev, self.H
        h8 = H // COLS
        nxt = self.next_header.get(phase) if self._flags(L, name) & F_POST else None
        for c in range(COLS):
            if self._flags(L, name) & F_POST:
                off = self.inp[nxt][0].off if nxt is not None else self.final
                self._post_to(seq, c, off, h8 * c, h8, S, 1)
            if name == "qkv":  # (per slot: q of heads 2 j, 2 j + 1, K and V halves, the gate logits)
                h, j = divmod(c, CHAIN)
                reg = self.att_reg[(L, h)] + 4 * META
                qh = self.nh // CHAIN
                pat = (C_LEN // 2, S, XB // 4, C_LEN // 2)
                seq.drain("C", c, dev.address("ACT", reg + 2 * self.hd * qh * j), S * 2 * qh * self.hd, pat)
                half = self.hd // CHAIN
                seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 2 * half * j), S * 2 * 2 * C_LEN,
                          (C_LEN // 2, S, XB // 4, self.hd // 2))
                seq.drain("C", c, dev.address("ACT", reg + 2 * (self.at.goff + 64 * j)), S * 2 * C_LEN, pat)
            elif name == "in":  # (B, C, x of the column's channels into its conv region)
                o = self.conv_reg[(L, c)]
                for off, nbytes, dims in CV.rows_drains(S):
                    seq.drain("C", c, dev.address("ACT", o + off), nbytes, dims)
            elif self._flags(L, name) & F_VOUT:
                tgt = self.inp[self.v_to[phase]][0].off
                self._post_to(seq, c, tgt, h8 * c, h8, S, 0)
                seq.drain("C", c, dev.address("ACT", tgt + XB + 2 * C_LEN * c), 2 * C_LEN)  # (its partial sums)
            elif name == "gate_up":
                ffn = self.cfg.intermediate_size // COLS
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "lm":
                self._logits_out(seq, c, S)

    def _gemv(self, seq: Sequence, st: _Step, phase: tuple[int, str]) -> None:
        for gi, rg in enumerate(self.inp[phase]):
            if gi == 0:
                seq.sync()  # (its inputs' producers)
            self._inputs(seq, rg, st.S)
        self._outputs(seq, st, phase)

    def _attention(self, seq: Sequence, st: _Step, L: int, weights: list[tuple[int, str]]) -> None:
        """The attention of layer L, per sub-step (each after the last one's records are in DDR): per column its cache
        range (A, patched), the F_ATT object and the sub-step's q and K / V objects (X), the records (D, patched) and o
        (D, into the sub-step's slots of o's input). The weights of the GEMV phases after it go before the sync of its
        inputs (their first layout block: ``_weight_run``): the A stream keeps streaming them while the attention waits
        for them."""
        dev, S, T = self.dev, st.S, self.T
        o, sp, nk = self.inp[(L, "o")][0], packed_slots(S), self.nh * self.hd // 8  # (o: the KV head's k blocks)
        for u, (a, n) in enumerate(st.subs):
            if u:
                seq.sync()  # (sub-step u - 1's records in DDR: this one reads them)
            for c in range(COLS):  # (its range of the cache and a chain end's tail objects, patched per step)
                st.fills.append((seq.fill("A", c, dev.address("KV", self.kv_off[L]), T), L, c, u))
                if c % CHAIN == CHAIN - 1:  # (one object each: the slots of its records, ``_tails``)
                    st.tails += [(seq.fill("A", c, dev.address("KV", self.ring_off[L]), T), L, c, u, t)
                                 for t in range(tails(n))]
            last = u == len(st.subs) - 1 and weights
            if last:
                self._weight_run(seq, weights, blocks=slice(1))  # (a later block's after the X objects: ``_weight_run``)
            if u == 0:
                seq.sync()  # (the q and K / V objects in DDR)
            for c in range(COLS):  # (its q objects, then its K / V objects: half objects, the second strip SLOTS XB on)
                reg = self.att_reg[(L, c // CHAIN)] + a * XB
                seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
            if last:
                self._weight_run(seq, weights, blocks=slice(1, None))
            for c in range(CHAIN - 1, COLS, CHAIN):
                h = c // CHAIN
                for i in range(len(kv.emit_order(p=0, S=n, hd=self.hd, a_bytes=T, kp=self.kp))):
                    hdl = seq.drain("D", c, dev.address("KV", self.kv_off[L]), CV.D_BYTES)
                    st.recs.append((hdl, L, h, u, i))
                at = dev.address("ACT", o.off + XB + 2 * sp * 8 * nk * h + 16 * a)  # ([k block][slot sp][8])
                seq.drain("D", c, at, 16 * n * nk, None if n == sp else (4 * n, nk, 4 * sp, 4 * sp * nk))

    def _conv(self, seq: Sequence, st: _Step, L: int, weights: list[tuple[int, str]]) -> None:
        """The conv of layer L (``lfm.conv``): per column its region (X, after the in phase's drains), y into out's input
        and the new history over the old (D). The weights of the GEMV phases after it go first (A streams them while
        the operator cores run the conv)."""
        dev, S = self.dev, st.S
        if weights:
            self._weight_run(seq, weights)
        seq.sync()  # (the in phase's rows in the regions)
        out, sp = self.inp[(L, "out")][0], packed_slots(S)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.conv_reg[(L, c)]), CV.region_bytes())
        for c in range(COLS):
            o = self.conv_reg[(L, c)]
            seq.drain("D", c, dev.address("ACT", out.off + XB + 2 * sp * CH * c), CV.y_bytes(S))
            for off, nbytes in CV.hist_drains(S):
                seq.drain("D", c, dev.address("ACT", o + off), nbytes)

    def _build(self, S: int, lm: bool, subs: tuple) -> _Step:
        dev = self.dev
        seq = Sequence(self.op, rolling=True, inputs=("X",))
        st = _Step(S, subs)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.tok), TOKENS * XB)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * SLOTS * XB), S * XB)
        ph = self.phases(lm)

        def run(i: int) -> list[tuple[int, str]]:  # (the GEMV phases from i to the next operator step: one A fill)
            return ph[i:next((k for k in range(i, len(ph)) if ph[k][1] in OPS), len(ph))]

        for i, (L, name) in enumerate(ph):
            if name == "att":
                self._attention(seq, st, L, run(i + 1))
                continue
            if name == "conv":
                self._conv(seq, st, L, run(i + 1))
                continue
            if i == 0:
                self._weight_run(seq, run(0))
            self._gemv(seq, st, (L, name))
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        return st

    def _lm_phase(self, seq: Sequence) -> None:
        self._weight_run(seq, [(-1, "lm")])
        self._gemv(seq, _Step(1, ()), (-1, "lm"))

    def logits_of(self, x: torch.Tensor) -> torch.Tensor:
        """(its header's input: r = x, v and its partial sums zero, so the header's update adds nothing and its norm is
        the output norm of x, as a step's)"""
        rg = self.inp[(-1, "lm")][0]
        if self._metas_s != (1, self.start):
            self._metas(1)
        if self._tr_lm is None:
            seq = Sequence(self.op, rolling=True, inputs=("X",))
            self._lm_phase(seq)
            seq.sync(fills=True)
            self._tr_lm = Transaction(self.dev, seq.words())
        sp = packed_slots(1)
        act, o = self.dev.view("ACT"), rg.off + XB
        act[o:o + XPAY] = 0
        hx = act[o + XPAY:o + XPAY + 4 * sp * self.H].reshape(self.H // 8, 2, 16 * sp)
        hx[:, 0] = 0
        hx[:, 1] = interleave(x.reshape(1, -1).float().numpy(), sp).view(np.uint8).reshape(self.H // 8, 16 * sp)
        self.dev.flush("ACT", o, XPAY + 4 * sp * self.H)
        self.ready()
        self.dev.dispatch(self._tr_lm)
        return self._logits(1)[0]

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        """The attentions' sub-steps (first slot, slots) of a step at p .. p + S - 1 (both kinds: one head dim): within a
        pair of records (``decode.attn``: P V adds two records together; a group holds whole pairs)."""
        out = tuple(substeps(p, S, 16, self.at.slots(self.hd)))
        assert len(out) <= U, (p, S)
        return out

    def _ranges(self, kind: int, p: int) -> list[tuple[int, int]]:
        """The chain's (first group, groups) of a layer kind's cache for a sub-step whose slots are in the group of
        position p (a sliding-window cache: from the guard's groups before it on, the window of any of its positions)."""
        g = p // self.pg
        return kv.ranges(first=max(0, g - self.guard) if kind == 0 else 0, g=g, chain=CHAIN)

    def _patch_tables(self, st: _Step) -> dict[str, np.ndarray]:
        """A step's cache fills, tail fills and record drains as arrays: BD word index, layer kind (0 SWA, 1 full), the
        sub-step, the column's place in its chain / the record's index, and the byte offsets in KV of group 0 of the
        layer's KV head (and of its ring)."""
        k = {L: KINDS.index(self.kind(L)) for L in self.kv_off}
        f = np.array([(at, k[L], u, c % CHAIN, self.kv_base(L, c // CHAIN)) for at, L, c, u in st.fills],
                     np.int64).reshape(-1, 5)
        t = np.array([(at, k[L], self.kv_base(L, c // CHAIN), self.ring(L, c // CHAIN), tk, u)
                      for at, L, c, u, tk in st.tails], np.int64).reshape(-1, 6)
        r = np.array([(hdl[0], k[L], u, i, self.kv_base(L, h), self.ring(L, h)) for hdl, L, h, u, i in st.recs],
                     np.int64).reshape(-1, 6)
        return {"f_at": f[:, 0], "f_kind": f[:, 1], "f_u": f[:, 2], "f_j": f[:, 3], "f_base": f[:, 4],
                "t_at": t[:, 0], "t_kind": t[:, 1], "t_base": t[:, 2], "t_ring": t[:, 3], "t_k": t[:, 4],
                "t_u": t[:, 5], "r_at": r[:, 0], "r_kind": r[:, 1], "r_u": r[:, 2], "r_i": r[:, 3],
                "r_base": r[:, 4:6]}

    def _tails(self, st: _Step, p: int, pu: np.ndarray) -> None:
        """The step's tail fills (per sub-step a chain end's objects: the ring slots of the records of its first and
        last positions, or after a prefill for the first sub-step's first the record's V panels) and the F_ATT objects'
        M_ATREC. pu: per sub-step (first position, slots)."""
        t, T = st.patch, self.T
        from_prefill = np.array([L in self.vring.from_prefill for L in self.kv_off], bool)
        lay = {L: i for i, L in enumerate(self.kv_off)}
        rec_l = np.array([from_prefill[lay[L]] for _, L, _, _, _ in st.tails], bool)
        rec = rec_l & (t["t_k"] == 0) & (t["t_u"] == 0)  # (record mode: the first sub-step's first tail)
        q0, n = pu[t["t_u"], 0], pu[t["t_u"], 1]
        vrec = np.array([kv.v_offset(int(x), hd=self.hd, a_bytes=T, kp=self.kp) for x in pu[:, 0]],
                        np.int64)[t["t_u"]]
        slot = np.where(t["t_k"] == 0, q0 >> 3, (q0 + n - 1) >> 3) & 1
        address = self.dev.address("KV") + np.where(rec, t["t_base"] + vrec, t["t_ring"] + slot * T)
        self._bd_words(st.tr.words, t["t_at"], address, np.full(len(rec), T, np.int64))
        act = self.dev.view("ACT")
        for L in self.kv_off:
            m = int(L in self.vring.from_prefill)
            if not m:
                self.vring.check(L, p, p)
            for a in range(SLOTS):  # (the F_ATT objects of the sub-steps from slot a on: the first one's record mode)
                ma = m if a == 0 else 0
                if self._trec.get((L, a)) != ma:
                    for h in range(self.cfg.num_key_value_heads):
                        off = self.att_reg[(L, h)] + a * XB
                        act[off:off + 4 * META].view(np.int32)[M_ATREC] = ma
                        self.dev.flush("ACT", off, 4 * META)
                    self._trec[(L, a)] = ma

    def _ran(self, p: int, S: int) -> None:
        """After a step: the rings hold the rows of its open records; the conv histories its rows."""
        for L in self.kv_off:
            self.vring.ran(L, p, S)
        self.conv_p, self.conv_s = p, S

    def conv_at(self, p: int) -> None:
        """The conv histories' base is the 6 rows before position p (a prefill wrote them, a restore put them back): the
        next step continues at p with shift 0."""
        self.conv_p, self.conv_s = p, 0

    def conv_base(self, p: int) -> None:
        """Before a prefill chunk at p: each conv region's rows of the 6 positions before p moved to its base (rows 0 ..
        5, where the chunk's conv reads its history), as a step at p shifts them on its cores."""
        sh = self._cshift(p)
        if sh:
            act, dev = self.dev.view("ACT"), self.dev
            for o in self.conv_reg.values():
                h, t = o + CV.payload(2), o + CV.payload(1) + 3 * CV.TAP  # (rows 0 .. 7, rows 8, 9)
                dev.fetch("ACT", h, 8 * CV.ROW)
                dev.fetch("ACT", t, 2 * CV.ROW)
                rows = np.concatenate([act[h:h + 8 * CV.ROW], act[t:t + 2 * CV.ROW]]).reshape(CV.HIST, CV.ROW)
                act[h:h + 6 * CV.ROW] = rows[sh:sh + 6].reshape(-1)
                dev.flush("ACT", h, 6 * CV.ROW)
        self.conv_at(p)

    def _cshift(self, p: int) -> int:
        """The conv shift of a step at p: the last step's tokens before p (a step continues from the last one's
        positions, at most its own back, as a generation loop's rejected drafts do)."""
        sh = p - self.conv_p
        assert 0 <= sh <= self.conv_s or (sh == 0 and self.conv_s == 0), \
            f"a step at {p} after the conv histories of a step at {self.conv_p} ({self.conv_s} slots)"
        return sh

    def _set_position(self, st: _Step, p: int, ahead: bool = False) -> bool:
        """The token objects (per layer kind and sub-step: its positions, the columns' ranges, the step's RoPE tables;
        token object 0: the step's slots and conv shift) and the step's cache transfers."""
        S, cfg, dev, T = st.S, self.cfg, self.dev, self.T
        if ahead:
            return False  # (the conv shift follows the step that runs now)
        assert 0 <= p and p + S <= self.max_positions, (p, S)
        assert st.subs == self.subs(p, S), (p, S, st.subs)
        act = dev.view("ACT")
        pu = np.zeros((U, 2), np.int64)  # (per sub-step: first position, slots)
        fo = np.zeros((2, U, CHAIN, 2), np.int64)  # (per kind, sub-step and column of the chain: its A fill)
        ro = np.zeros((U, 6, 3), np.int64)  # (per sub-step and i-th record: base, offset, bytes)
        for kind, kn in enumerate(KINDS):
            cs = rope_tables(self.inv_freq[kn], range(p, p + S), cfg.rot(kn)).view(np.uint8).reshape(-1)
            for a in range(SLOTS):  # (each token object copies the step's table; one of no sub-step: no objects)
                o = self.tok + (SMAX * kind + a) * XB
                act[o + 4 * META:o + 4 * META + cs.size] = cs
                m = act[o:o + 4 * META].view(np.int32)
                m[M_NS], m[M_TOK_NG:M_TOK_NG + COLS], m[M_TOK_NA:M_TOK_NA + COLS] = 0, 0, 0
            for u, (a, n) in enumerate(st.subs):
                r = self._ranges(kind, p + a)
                o = self.tok + (SMAX * kind + a) * XB
                m = act[o:o + 4 * META].view(np.int32)
                me = act[o + (n - 1) * XB:][:4 * META].view(np.int32)  # (the token of its last slot)
                m[M_NS], m[M_POS], m[M_TOK_A] = n, p + a, a
                for c in range(COLS):  # (every KV head's columns: the same ranges)
                    g0, ng = r[c % CHAIN]
                    no = kv.fill(g0, ng, self.kp)[1] + (tails(n) if c % CHAIN == CHAIN - 1 else 0)
                    m[M_TOK_G0 + c], m[M_TOK_NA + c], me[M_TOK_NG + c] = g0 * self.pg, self.G * ng | no << 16, no
                fo[kind, u] = [kv.fill(*rj, self.kp) for rj in r]
                if kind == 0:
                    pu[u] = p + a, n
                    rec = kv.records(p=p + a, S=n, hd=self.hd, a_bytes=T, kp=self.kp)
                    ro[u, :len(rec)] = rec
        m0 = act[self.tok:self.tok + 4 * META].view(np.int32)
        m0[CV.M_TOK_S], m0[CV.M_CSHIFT] = S, self._cshift(p)
        dev.flush("ACT", self.tok, TOKENS * XB)
        if st.patch is None:
            st.patch = self._patch_tables(st)
        t, kv0 = st.patch, dev.address("KV")
        f = fo[t["f_kind"], t["f_u"], t["f_j"]]  # (first object, objects)
        self._bd_words(st.tr.words, t["f_at"], kv0 + t["f_base"] + f[:, 0] * T, f[:, 1] * T)
        self._tails(st, p, pu)
        r = ro[t["r_u"], t["r_i"]]  # (the record drains: from the KV head's group 0 or ring, offset, bytes)
        base = np.take_along_axis(t["r_base"], r[:, :1], 1)[:, 0]
        self._bd_words(st.tr.words, t["r_at"], kv0 + base + r[:, 1], r[:, 2])
        st.tr.flush()
        return True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self.conv_at(0)
        if self.dev is not None:  # (zero histories: the rows before position 0)
            act = self.dev.view("ACT")
            for o in self.conv_reg.values():
                act[o + CV.payload(2):o + CV.payload(2) + XPAY] = 0
                act[o + CV.payload(1) + 3 * CV.TAP:o + CV.payload(1) + XPAY] = 0
            if self.conv_reg:
                lo, hi = min(self.conv_reg.values()), max(self.conv_reg.values()) + CV.region_bytes()
                self.dev.flush("ACT", lo, hi - lo)

    def step(self, x: torch.Tensor, position: int, *, logits: bool = True) -> torch.Tensor | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the embeddings at positions position .. position + S
        - 1 (the positions before are cached). Returns the logits (S, vocab) bf16 (a view of the NPU's buffer, valid
        until the next step), or without ``logits`` None (only the caches are updated)."""
        x = x.reshape(-1, self.H).float()
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        act = self.dev.view("ACT")
        xf = x.numpy()
        self._write_loadr(xf)
        sp = packed_slots(S)
        xi = interleave(xf, sp).view(np.uint8).reshape(-1)
        first = self.phases()[0]
        self._write_input(self.inp[first][0], xi)  # (the first layer's input: a plain one)
        h0 = self.inp[self.first_header][0].off + XB + XPAY  # (the first header's r slots: x)
        hx = act[h0:h0 + 4 * sp * self.H].reshape(self.H // 8, 2, 16 * sp)
        hx[:, 1] = xi.reshape(self.H // 8, 16 * sp)
        self.dev.flush("ACT", h0, 4 * sp * self.H)
        return self._run(S, position, logits)

    def read_kv(self, layer: int, positions: list[int] | range) -> tuple[torch.Tensor, torch.Tensor]:
        """K, V (kv heads, n, head_dim) fp32 as cached for the positions of an attention layer: K normed (with its
        weights) and rotated, V as it is."""
        pos = np.asarray(positions)
        n, kvh = int(pos.max()) // self.pg + 1, self.cfg.num_key_value_heads
        self.dev.fetch("KV", self.kv_off[layer], kvh * self._kv_bytes(layer))
        ks, vs = [], []
        for h in range(kvh):
            b = self.kv_base(layer, h)
            k, v = kv.kv_from_groups(self.dev.view("KV")[b:b + n * self.G * self.T].reshape(n, self.G, self.T), self.hd)
            ks.append(k[pos])
            vs.append(v[pos])
        return torch.from_numpy(np.stack(ks)), torch.from_numpy(np.stack(vs))

    def snapshot(self, position: int) -> dict[str, object]:
        """(and the conv histories: each region's rows, the last step's position and slots, ``conv_at``)"""
        out = super().snapshot(position)
        rows = np.zeros((len(self.conv_reg), CV.HIST * CV.ROW), np.uint8)
        act = self.dev.view("ACT")
        for i, o in enumerate(self.conv_reg.values()):
            for a, b, n in self._hist_parts(o):
                self.dev.fetch("ACT", a, n)
                rows[i, b:b + n] = act[a:a + n]
        out["conv"] = (rows, self.conv_p, self.conv_s)
        return out

    def restore(self, snapshot: dict[str, object]) -> None:
        super().restore(snapshot)
        rows, self.conv_p, self.conv_s = snapshot["conv"]
        act = self.dev.view("ACT")
        for i, o in enumerate(self.conv_reg.values()):
            for a, b, n in self._hist_parts(o):
                act[a:a + n] = rows[i, b:b + n]
                self.dev.flush("ACT", a, n)

    @staticmethod
    def _hist_parts(o: int) -> tuple[tuple[int, int, int], ...]:
        """A conv region's history rows at ACT offset o: (offset in ACT, offset in the rows, bytes) of rows 0 .. 7 and 8, 9."""
        return (o + CV.payload(2), 0, 8 * CV.ROW), (o + CV.payload(1) + 3 * CV.TAP, 8 * CV.ROW, 2 * CV.ROW)

    def hidden(self, L: int = -1, slot: int = 0) -> torch.Tensor:
        """The residual stream the LM head read in the last step (L = -1), of a slot (bf16 values, fp32): its updated
        residual (``residual``), the output norm's input."""
        assert L == -1, "the LM head's input only"
        return self.residual(slot)

    def read_conv(self, layer: int) -> torch.Tensor:
        """A conv layer's history (HIST rows, hidden) as the regions hold it (bf16 values as fp32): rows 0 .. 5 the base
        of the last step, then its rows."""
        out = []
        for c in range(COLS):
            o = self.conv_reg[(layer, c)]
            self.dev.fetch("ACT", o, CV.region_bytes())
            reg = self.dev.view("ACT")[o:o + CV.region_bytes()]
            rows = np.concatenate([reg[CV.payload(2):CV.payload(2) + 8 * CV.ROW],
                                   reg[CV.payload(1) + 3 * CV.TAP:CV.payload(1) + 3 * CV.TAP + 2 * CV.ROW]])
            out.append(rows.view(np.uint16).reshape(CV.HIST, CH))
        return torch.from_numpy(np.concatenate(out, 1).copy()).view(torch.bfloat16).float()

    def op_out(self, L: int, slot: int = 0) -> torch.Tensor:
        """The operator's output of layer L in the last step, of a slot (bf16 values, fp32): the attention's o (heads
        hd, gated) or the conv's y (hidden), the output projection's input."""
        rg, sp = self.inp[(L, "out" if self.kind(L) == "conv" else "o")][0], packed_slots(self._last_s)
        self.dev.fetch("ACT", rg.off + XB, 2 * sp * rg.K)
        v = self.dev.view("ACT")[rg.off + XB:rg.off + XB + 2 * sp * rg.K].view(np.uint16).reshape(-1, sp, 8)[:, slot]
        return torch.from_numpy(v.reshape(-1).copy()).view(torch.bfloat16).float()

    def residual(self, slot: int = 0) -> torch.Tensor:
        """The last step's residual stream after the last layer (the LM header's update), bf16 values as fp32."""
        off, sp = self.final + XB + XPAY, packed_slots(self._last_s)
        self.dev.fetch("ACT", off, 4 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 4 * sp * self.H].view(np.uint16).reshape(-1, 2, sp, 8)[:, 1, slot]
        return torch.from_numpy(v.reshape(-1).copy()).view(torch.bfloat16).float()


__all__ = ["KV_K", "KV_PAD", "LfmOverlay"]
