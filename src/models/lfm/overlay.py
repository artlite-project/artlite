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

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.attn import Attn, rope_tables, score_bits
from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.model import InputRegion, Snapshot
from artlite.decode.protocol import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_POST,
    F_PZERO,
    F_VOUT,
    M_FLAGS,
    M_SCALE,
    M_SKIP,
    META,
    SLOTS,
    XB,
    XPAY,
    hidden_meta,
    interleave,
    packed_slots,
    post_region_bytes,
)
from artlite.decode.step import StepPlan
from artlite.decode.vring import PAIR
from artlite.gguf_model import GGUFFile
from artlite.model_config import LfmConfig
from artlite.operators.design import COLS
from artlite.operators.util import bf16_bits
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import conv as CV
from .decode import LfmDecodeOperator
from .image import CH, LfmImage

PHASES = {"conv": ("in", "conv", "out", "gate_up", "down"), "attn": ("qkv", "att", "o", "gate_up", "down")}
CHAIN = 2  # columns per KV head
KV_PAD = 4096  # positions past the caches (the last layer's): the prefill's slices of a cache
KINDS = ("swa", "full")  # the cache kinds: token object t = SLOTS kind + a, of the sub-step from slot a on


class LfmOverlay(AttentionOverlay):
    """An LFM3 GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices; the first
    one reads x), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the caches' capacity, ``kv_k``: the
    caches' K precision ("single": one bfp16 block per 8 dims, "hilo": hi + lo blocks), ``weights``: the image's buffer
    allocated ahead."""

    CONFIG, IMAGE = LfmConfig, LfmImage
    CHAIN = CHAIN
    OPS = ("att", "conv")  # (the operator cores' phases)

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "single", weights: WeightBuffer | None = None):
        super().__init__(gguf, layers=layers, vocab=vocab, kv_k=kv_k, weights=weights)
        cfg, img = self.cfg, self.image
        assert cfg.num_key_value_heads * CHAIN == COLS and cfg.conv_width == CV.WIDTH
        self.hd, self.nh = cfg.head_dim, cfg.q_per_kv
        self.q_heads = self.nh
        self.at = at = Attn(hds=(self.hd,), rot=(cfg.rot("swa"), cfg.rot("full")), chain=CHAIN, heads=self.nh,
                            kp=self.kp, qk_norm=True, qk_weights=True, eps=cfg.rms_norm_eps, a_bytes=self.T, gate=True,
                            conv=CV.ConvJob())
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        gu = {b.R for L in self.layers for b in img.region(L, "gate_up").blocks}
        self.op = LfmDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                    eps=cfg.rms_norm_eps, attn=at, gu_win=gu != {16})
        self.pg = at.pg[self.hd]  # positions per cache group
        assert cfg.sliding_window % self.pg == 0
        self.guard = cfg.sliding_window // self.pg  # (groups before a sliding-window cache's first one)
        full = -(-max_positions // self.pg)
        self.max_positions = full * self.pg
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=self.guard + full, guard=self.guard,
                                window=cfg.sliding_window),
                      CacheKind(hd=self.hd, pg=self.pg, groups=full))
        self.uses = {L: CacheUse(kind=KINDS.index(self.kind(L)), source=L, ring=L)
                     for L in self.layers if self.kind(L) != "conv"}
        self.period = PAIR
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
        self.toks = (take("ACT", len(KINDS) * SLOTS * XB),)
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name not in self.OPS:
                self._add_inputs((L, name), post=bool(self._flags(L, name) & F_POST))
        self.final = take("ACT", post_region_bytes(H))  # (the LM header's updated residual)
        att = [L for L in self.layers if self.kind(L) != "conv"]
        convs = [L for L in self.layers if self.kind(L) == "conv"]
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token SLOTS kind + a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in att for h in range(self.kv_heads)}
        self.conv_reg = {(L, c): take("ACT", CV.region_bytes()) for L in convs for c in range(COLS)}
        self._plan_caches(att, pad=KV_PAD // self.pg * self.G * self.T)
        self._sizes()
        self._plan_headers()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_layer(self, L: int) -> None:
        """(a conv layer's regions: their taps, their histories zero)"""
        if self.kind(L) == "conv":
            self._load_conv(L)
        else:
            self._load_attention(L)

    def _attention_scale(self, L: int) -> tuple[int, int]:
        return score_bits(1 / np.sqrt(self.hd))

    def _qk_weights(self, L: int) -> dict[str, np.ndarray]:
        """(the q / k norms' weights per dim)"""
        return {n: np.asarray(self.gguf.tensor(f"blk.{L}.attn_{n}_norm.weight").dequantize(), np.float32).reshape(-1)
                .view(np.uint8) for n in ("q", "k")}

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

    def _outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """The C drains of a GEMV phase (after its last group): a header's updated residual slice (into the next
        header's r slots), then its outputs."""
        L, name = phase
        S, dev = st.S, self.dev
        flags = self._flags(L, name)
        for c in range(COLS):
            if flags & F_POST:
                self._header_to_r(seq, st, phase, c)
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
            elif flags & F_VOUT:
                self._vout(seq, st, phase, c)
            elif name == "gate_up":
                ffn = self.cfg.intermediate_size // COLS
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "lm":
                self._logits_out(seq, c, S)

    def _operator(self, seq: Sequence, st: StepPlan, L: int, name: str, weights: list[tuple[int, str]]) -> None:
        if name == "conv":
            self._conv(seq, st, L, weights)
        else:
            super()._operator(seq, st, L, name, weights)

    def _conv(self, seq: Sequence, st: StepPlan, L: int, weights: list[tuple[int, str]]) -> None:
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

    def logits_of(self, x: np.ndarray) -> np.ndarray:
        """(its header's input: r = x, v and its partial sums zero, so the header's update adds nothing and its norm is
        the output norm of x, as a step's)"""
        rg = self.inp[(-1, "lm")][0]
        if self._metas_s != (1, self.start):
            self._metas(1)
        if self._tr_lm is None:
            seq = Sequence(self.op, inputs=("X",))
            self._lm_phase(seq)
            seq.sync(fills=True)
            self._tr_lm = Transaction(self.dev, seq.words())
        sp = packed_slots(1)
        act, o = self.dev.view("ACT"), rg.off + XB
        act[o:o + XPAY] = 0
        hx = act[o + XPAY:o + XPAY + 4 * sp * self.H].reshape(self.H // 8, 2, 16 * sp)
        hx[:, 0] = 0
        hx[:, 1] = interleave(np.asarray(x, np.float32).reshape(1, -1), sp).view(np.uint8).reshape(self.H // 8, 16 * sp)
        self.dev.flush("ACT", o, XPAY + 4 * sp * self.H)
        self.ready()
        self.dev.dispatch(self._tr_lm)
        return self._logits(1)[0]

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        """(both kinds: one head dim)"""
        return self._substeps(p, S, self.hd)

    def _rope(self, kind: int, p: int, S: int) -> np.ndarray:
        kn = KINDS[kind]
        return rope_tables(self.inv_freq[kn], range(p, p + S), self.cfg.rot(kn))

    def _token_words(self, st: StepPlan, p: int) -> None:
        """Token object 0: the step's slots and conv shift."""
        m0 = self.dev.view("ACT")[st.tok:st.tok + 4 * META].view(np.int32)
        m0[CV.M_TOK_S], m0[CV.M_CSHIFT] = st.S, self._cshift(p)

    def _ran(self, p: int, S: int) -> None:
        """(and the conv histories: their rows)"""
        super()._ran(p, S)
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
                (h, _, nh), (t, _, nt) = CV.history_parts(o)  # (rows 0 .. 7, rows 8, 9)
                dev.fetch("ACT", h, nh)
                dev.fetch("ACT", t, nt)
                rows = np.concatenate([act[h:h + nh], act[t:t + nt]]).reshape(CV.HIST, CV.ROW)
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

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self.conv_at(0)
        if self.dev is not None:  # (zero histories: the rows before position 0)
            act = self.dev.view("ACT")
            for o in self.conv_reg.values():
                for a, _, n in CV.history_parts(o):
                    act[a:a + n] = 0
                    self.dev.flush("ACT", a, n)

    def step(self, x: np.ndarray, position: int, *, logits: bool = True) -> np.ndarray | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the embeddings at positions position .. position + S
        - 1 (the positions before are cached). Returns the logits (S, vocab) bf16 (a view of the NPU's buffer, valid
        until the next step), or without ``logits`` None (only the caches are updated)."""
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        act = self.dev.view("ACT")
        xf = x
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

    def snapshot(self, position: int) -> Snapshot:
        """(and the conv histories: each region's rows, the last step's position and slots, ``conv_at``)"""
        out = super().snapshot(position)
        rows = np.zeros((len(self.conv_reg), CV.HIST * CV.ROW), np.uint8)
        act = self.dev.view("ACT")
        for i, o in enumerate(self.conv_reg.values()):
            for a, b, n in CV.history_parts(o):
                self.dev.fetch("ACT", a, n)
                rows[i, b:b + n] = act[a:a + n]
        out.extra["conv"] = (rows, self.conv_p, self.conv_s)
        return out

    def restore(self, snapshot: Snapshot) -> None:
        super().restore(snapshot)
        rows, self.conv_p, self.conv_s = snapshot.extra["conv"]
        act = self.dev.view("ACT")
        for i, o in enumerate(self.conv_reg.values()):
            for a, b, n in CV.history_parts(o):
                act[a:a + n] = rows[i, b:b + n]
                self.dev.flush("ACT", a, n)

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream the LM head read in the last step (L = -1), of a slot (bf16 values, fp32): its updated
        residual (``residual``), the output norm's input."""
        assert L == -1, "the LM head's input only"
        return self.residual(slot)

    def read_conv(self, layer: int) -> np.ndarray:
        """A conv layer's history (HIST rows, hidden) as the regions hold it (bf16 values as fp32): rows 0 .. 5 the base
        of the last step, then its rows."""
        out = []
        for c in range(COLS):
            parts = CV.history_parts(self.conv_reg[(layer, c)])
            for a, _, n in parts:
                self.dev.fetch("ACT", a, n)
            act = self.dev.view("ACT")
            rows = np.concatenate([act[a:a + n] for a, _, n in parts])
            out.append(rows.view(np.uint16).reshape(CV.HIST, CH))
        return np.ascontiguousarray(np.concatenate(out, 1)).view(bfloat16).astype(np.float32)

    def lm_input(self, slot: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
        """The LM header's input of the last step, of a slot (a step without the LM head too: the last layer's FFN drains
        v and its partial sums there, its header r): v, r (bf16 values as fp32) and v's sum of squares (the columns'
        partial sums, fp32: the header's 1 / rms(v)). Its update r + v / rms(v) w_post, output-normed, is the LM head's
        input (``models.lfm.lfm.NpuLfm.audio_step``)."""
        rg, sp = self.inp[(-1, "lm")][0], packed_slots(self._last_s)
        o = rg.off + XB
        self.dev.fetch("ACT", o, XPAY + 4 * sp * self.H)
        act = self.dev.view("ACT")
        ps = act[o:o + 4 * COLS * SLOTS * 8].view(np.float32).reshape(COLS, SLOTS, 8)  # ([column][slot][8 lanes])
        x = act[o + XPAY:o + XPAY + 4 * sp * self.H].view(np.uint16).reshape(self.H // 8, 2, sp, 8)[:, :, slot]
        v, r = (np.ascontiguousarray(x[:, i]).reshape(-1).view(bfloat16).astype(np.float32) for i in range(2))
        return v, r, float(ps[:, slot].sum(dtype=np.float32))


__all__ = ["KV_PAD", "LfmOverlay"]
