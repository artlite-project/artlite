"""A Gemma 4 model on its decode overlay (``gemma4/decode.py``): buffers, the transaction of a step of S <= 4 token
slots, the step.

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (``attn``: per
layer kind and sub-step its positions and cache ranges, the RoPE tables) and the slots' residual slices (F_LOADR: the
embeddings, fp32); E2B / E4B then run the per-layer projection of the embeddings ("ple": column c's rows the 32 c .. of
each layer's 256, ``image``), its rows drained as p into the gate phases' metadata objects [p: slot][256] (one BD per
column: the gate input regions sit ``gate_pitch`` apart; e, the host's per-layer token rows sqrt(256 / 2), in the
other half), from which each gate phase completes its column's per-layer inputs (``decode.protocol.F_PLI``; a step
whose tokens' p the host kept, ``read_p``, writes them instead and runs without the phase: ``ple``); then per layer L
(column c, KV head h = c / 4):

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

Phase inputs are packed (``decode.protocol.packed_slots``): SP = 1, 2, 4 slots per k block in a step of S = 1, 2, 3 / 4
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

from collections.abc import Callable

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.model import InputRegion
from artlite.decode.protocol import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_PLI,
    F_POST,
    F_PZERO,
    F_VOUT,
    M_SCALE,
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
from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig
from artlite.operators.design import COLS
from artlite.operators.util import bf16_bits, bf16_value, rup
from artlite.overlay import Sequence, WeightBuffer

from . import attn as AT
from .decode import GemmaDecodeOperator
from .image import PLE, TILE_BYTES, GemmaImage

PHASES = ("qkv", "att", "o", "gate_up", "down", "gate", "proj")
PLE_PHASE = (-1, "ple")  # (E2B / E4B: the per-layer projection, the step's first phase)
GUARD = 16  # cache groups before a sliding-window cache's first one (the prefill attention reads its window's 16
# groups before a position from there on, also before position 0: positions it masks)
KV_PAD = 2048  # positions past the caches (the last layer's): the prefill's sub-chunk slices of a cache


class GemmaOverlay(AttentionOverlay):
    """A Gemma 4 GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices; the
    first one reads x), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the full-attention caches'
    capacity, ``kv_k``: the caches' K precision ("hilo": bfp16 hi + lo blocks; "single": one block per 8 dims),
    ``weights``: the image's buffer allocated ahead. Two cache kinds (sliding-window: 0, full: 1); a layer without K / V
    projections reads the cache of the last layer of its kind that has them."""

    CONFIG, IMAGE = GemmaConfig, GemmaImage
    CHAIN = AT.QH

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "hilo", weights: WeightBuffer | None = None):
        super().__init__(gguf, layers=layers, vocab=vocab, kv_k=kv_k, weights=weights)
        cfg, img = self.cfg, self.image
        assert self.T == TILE_BYTES
        self.at = at = AT.GemmaAttn(cfg, kp=self.kp)
        self.q_heads = AT.QH
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        self.op = GemmaDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, eps=cfg.rms_norm_eps,
                                      attn=at)
        pg_s, pg_f = at.pg[256], at.pg[512]
        assert GUARD * pg_s == cfg.sliding_window
        n_full = -(-max_positions // pg_f)
        self.max_positions = n_full * pg_f
        n_swa = GUARD + -(-self.max_positions // pg_s)  # (a sliding-window cache: linear, after its guard)
        self.kinds = (CacheKind(hd=256, pg=pg_s, groups=n_swa, guard=GUARD, window=cfg.sliding_window),
                      CacheKind(hd=512, pg=pg_f, groups=n_full))
        self.uses = {L: CacheUse(kind=0 if cfg.swa[L] else 1, source=cfg.kv_source(L), ring=cfg.kv_source(L))
                     for L in self.layers}
        self.period = max(at.pg.values())  # (the steps' sub-steps repeat with the groups)
        self._e_idx: np.ndarray | None = None
        self.ple = True  # (the step runs the per-layer projection; else the host wrote p: ``step(p=)``)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order, from layer ``start`` on (``lm`` False: up to the last attention that writes a
        cache)."""
        out = [(L, n) for L in self.layers if L >= self.start for n in PHASES
               if n not in ("gate", "proj") or self.cfg.per_layer_dim]
        out = [PLE_PHASE, *out] if self.cfg.per_layer_dim and self.ple else out
        if not lm:
            own = [L for L in self.layers if self.cfg.kv_source(L) == L]
            return out[:out.index((own[-1], "att")) + 1] if own else []
        return [*out, (-1, "lm")]

    def _flags(self, L: int, name: str) -> int:
        # (a step from a later layer on: its qkv header zeroes pend, the update it adds: ``step(start=)``)
        return {"qkv": F_NORM | (F_POST if L != self.layers[0] else 0) | (F_PZERO if L == self.start != self.layers[0]
                                                                         else 0), "o": F_VOUT | F_PZERO | F_EMITR,
                "gate_up": F_POST | F_NORM | F_GU, "down": F_VOUT | F_PZERO | F_EMITR, "gate": F_POST | F_PLI,
                "proj": F_VOUT | F_PZERO | F_EMITR, "lm": F_POST | F_NORM, "ple": 0}[name]

    def _variant(self) -> tuple:
        return (self.start, self.ple)

    def prepare(self, S: int, logits: bool = True, start: int | None = None) -> None:
        """(the steps from the first layer: with and without the per-layer projection)"""
        both = self.cfg.per_layer_dim and start in (None, self.layers[0])
        for self.ple in (True, False) if both else (True,):
            super().prepare(S, logits, start)
        self.ple = True

    def _scale(self, L: int, name: str) -> float:
        sc = self.image.norms(L, name)[2]
        return self._scalar(sc) if sc else 1.0

    def _plan(self) -> None:
        """Byte offsets in ACT, NORM, KV; the headers' chain (each one's v and r sources)."""
        cfg, H = self.cfg, self.H
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        self.toks = (take("ACT", AT.TOKENS * XB),)
        self.loadr = take("ACT", COLS * SLOTS * XB)
        if cfg.per_layer_dim:  # (the gate inputs gate_pitch apart: the per-layer projection's drains, ``_outputs``)
            for L in self.layers:
                self._add_inputs((L, "gate"), post=True)
            offs = [self.inp[(L, "gate")][0].off for L in self.layers]
            self.gate_pitch = rup(post_region_bytes(H))
            assert offs == [offs[0] + i * self.gate_pitch for i in range(len(offs))]
            if len(offs) % 2:
                take("ACT", self.gate_pitch)  # (the padding layer's rows: a sink)
        for L, name in self.phases():
            if name not in ("att", "gate"):
                self._add_inputs((L, name), post=bool(self._flags(L, name) & F_POST))
        self.final = take("ACT", post_region_bytes(H))  # (the LM header's updated residual)
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token SLOTS kind + a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in self.layers for h in range(self.kv_heads)}
        own = [L for L in self.layers if cfg.kv_source(L) == L]
        for L in self.layers:
            assert cfg.kv_source(L) in own, f"layer {L} attends to the cache of layer {cfg.kv_source(L)}"
        self._plan_caches(own, pad=KV_PAD // min(self.at.pg.values()) * self.G * self.T)
        self._sizes()
        self._plan_headers()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _scalar(self, name: str) -> float:
        w = np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)
        if float(w.min()) != float(w.max()):
            raise NotImplementedError(f"{name}: a norm weight per dimension (the attention cores scale the scores by "
                                      f"one constant per layer)")
        return float(w[0])

    def _attention_scale(self, L: int) -> tuple[int, int]:
        return AT.score_constant(self._scalar(f"blk.{L}.attn_q_norm.weight"),
                                 self._scalar(f"blk.{self.cfg.kv_source(L)}.attn_k_norm.weight"))

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _meta(self, L: int, name: str, rg: InputRegion, m: np.ndarray) -> np.ndarray:
        """(the residual's sizes; a header's output scale)"""
        hidden_meta(m, self.H)
        if rg.post:
            m[M_SCALE] = bf16_bits(self._scale(L, name))
        return m

    def _outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """The C drains of a GEMV phase (after its last group): a header's updated residual slice (into the next
        header's r slots), then its outputs."""
        L, name = phase
        S, cfg, dev = st.S, self.cfg, self.dev
        flags = self._flags(L, name)
        for c in range(COLS):
            if flags & F_POST:
                self._header_to_r(seq, st, phase, c)
            if name == "qkv":
                hd, h, j = cfg.head_dim(L), c // AT.QH, c % AT.QH
                reg = self.att_reg[(L, h)]  # ([q objects SLOTS | raw K / V objects SLOTS])
                seq.drain("C", c, dev.address("ACT", reg + 4 * META + 2 * hd * j), S * 2 * hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
                if cfg.kv_source(L) == L:
                    seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 4 * META + hd * j), S * hd,
                              (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif flags & F_VOUT:
                self._vout(seq, st, phase, c)
            elif name == "gate_up":
                ffn = cfg.intermediate_size // COLS
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "gate":
                self._values_to(seq, c, (L, "proj"), e0=64 * c, n=C_LEN, S=S)
            elif name == "ple":  # (C object j: the 32 rows of layers 2 j, 2 j + 1 -> their gate inputs' p [slot][256])
                pairs, pw = -(-len(self.layers) // 2), self.gate_pitch // 4
                g0 = self.inp[(self.layers[0], "gate")][0].off + 4 * META + 2 * PLE * c
                seq.drain("C", c, dev.address("ACT", g0), S * 2 * C_LEN, (PLE // 2, 2, pw, 2 * PLE * COLS // 4), pairs,
                          (2 * pw, pairs))
            elif name == "lm":
                self._logits_out(seq, c, S)

    def _weights_first(self, seq: Sequence, ph: list[tuple[int, str]], i: int,
                       run: Callable[[int], list[tuple[int, str]]]) -> None:
        if ph[i] == PLE_PHASE:  # (a run of its own: a step from a later layer on skips the layers between)
            self._weight_run(seq, [PLE_PHASE])
        elif i == 0 or ph[i - 1] == PLE_PHASE:
            self._weight_run(seq, run(i))

    def logits_of(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("the LM head after a prefill chunk runs as a step from its last layer on: step(start=)")

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """The attentions' sub-steps (first slot, slots) of a step at p .. p + S - 1 per cache kind (0 SWA, 1 full)."""
        return tuple(self._substeps(p, S, kind.hd) for kind in self.kinds)

    def _kind_subs(self, subs: tuple, kind: int) -> tuple[tuple[int, int], ...]:
        return subs[kind]

    def passes(self, p: int, S: int) -> int:
        return max(len(x) for x in self.subs(p, S))

    def _rope(self, kind: int, p: int, S: int) -> np.ndarray:
        return AT.rope_tables(self.cfg, list(range(p, p + S)), self.at.rot, swa=kind == 0)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, e: np.ndarray | None, position: int, *, logits: bool = True,
             start: int | None = None, emb: np.ndarray | None = None, p: np.ndarray | None = None
             ) -> np.ndarray | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the scaled embeddings, e (S, layers, per_layer_dim)
        the model's per-layer token rows times sqrt(per_layer_dim / 2) (``GemmaHost.layer_rows``; E2B / E4B), at positions
        position .. position + S - 1 (the positions before are cached). Returns the raw logits (S, vocab) bf16 (before
        the soft cap; a view of the NPU's buffer, valid until the next step), or without ``logits`` None (only the
        caches are updated). ``start``: the step from that layer on, x its input (e.g. a prompt's last token after a
        prefill chunk: ``GemmaPrefill.residual``; ``emb``: the scaled embeddings, the per-layer projection's input):
        its qkv phase is a header whose update adds nothing (F_PZERO: pend = 0; v and the partial sums 0: 1 / rms(v)
        stays finite) to res = x / (the previous layer's output scale), so the header's residual is x. ``p``: the
        tokens' per-layer projections (S, the overlay's layers, per_layer_dim) bf16 bits as an earlier step left them
        (``read_p``): written instead of the projection's phase."""
        emb = np.asarray(x if emb is None else emb, np.float32).reshape(-1, self.H)
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        dev, H = self.dev, self.H
        start = self._mode(start)
        if self._metas_s != (S, start):
            self._metas(S)
        act = dev.view("ACT")
        if start != self.layers[0]:  # (res: what the header scales back to x, with the scale's bf16 bits as M_SCALE)
            x = x / bf16_value(self._scale(start, "qkv"))
        xf = x
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
        self.ple = p is None or not self.cfg.per_layer_dim or start != self.layers[0]
        if self.cfg.per_layer_dim:
            if self.ple:
                self._write_input(self.inp[PLE_PHASE][0], interleave(emb, sp).view(np.uint8).reshape(-1))
            self._write_pe(np.asarray(e, np.float32).reshape(S, -1, self.cfg.per_layer_dim), None if self.ple else p)
        return self._run(S, position, logits)

    def _pe_index(self) -> np.ndarray:
        """(layers, SLOTS, per_layer_dim) uint16 indices in ACT of the gate meta objects' p (e: SLOTS d after)."""
        if self._e_idx is None:
            d = self.cfg.per_layer_dim
            offs = np.array([self.inp[(L, "gate")][0].off + 4 * META for L in self.layers])
            self._e_idx = offs[:, None, None] // 2 + d * np.arange(SLOTS)[None, :, None] + np.arange(d)[None, None, :]
        return self._e_idx

    def _write_pe(self, e: np.ndarray, p: np.ndarray | None) -> None:
        """Each layer's gate meta object: e [slot 4][per_layer_dim] bf16 after p (``decode.protocol.F_PLI``; e: the model's
        layers) and p (bf16 bits, the overlay's layers) when given."""
        S, d = e.shape[0], self.cfg.per_layer_dim
        idx, act = self._pe_index()[:, :S], self.dev.view("ACT").view(np.uint16)
        act[idx + SLOTS * d] = e[:, self.layers].astype(bfloat16).view(np.uint16).transpose(1, 0, 2)
        if p is not None:
            act[idx] = np.asarray(p, np.uint16).transpose(1, 0, 2)
        lo = 0 if p is not None else 2 * SLOTS * d  # (p and e of the slots, or e)
        for L in self.layers:
            self.dev.flush("ACT", self.inp[(L, "gate")][0].off + 4 * META + lo, 2 * SLOTS * d + 2 * S * d - lo)

    def read_p(self, S: int) -> np.ndarray:
        """The last step's per-layer projections (S, the overlay's layers, per_layer_dim) as bf16 bits (it ran the
        projection: ``ple``)."""
        d = self.cfg.per_layer_dim
        for L in self.layers:
            self.dev.fetch("ACT", self.inp[(L, "gate")][0].off + 4 * META, 2 * S * d)
        return self.dev.view("ACT").view(np.uint16)[self._pe_index()[:, :S]].transpose(1, 0, 2)


__all__ = ["GemmaOverlay"]
