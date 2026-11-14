"""A Llama 3.x or Qwen3 model on its decode overlay (``llama/decode.py``): buffers, the transaction of a step of S <= 4
token slots, the step (Qwen3: its q / k norms' weights in the attention's q and K / V objects, ``_load_attention``).

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (per sub-step its
positions and the cache's range, the RoPE tables) and the slots' residual slices (F_LOADR: the embeddings, fp32), then
per layer L (column c = KV head c):

- qkv: the input norm, the rows of the column's query heads and of K_c, V_c -> the KV head's attention region [q
  objects SLOTS | raw K / V objects SLOTS] (per slot [head nh][hd] and [K hd | V hd] bf16);
- att: the attention cores (``decode.attn``) on the layer's cache, per sub-step (``subs``: its q and K / V objects,
  the first q object its F_ATT object) -> o into o's packed input, the new positions' records into the cache;
- o: into the residual (F_RESID), its bf16 copy out (F_EMITR) as gate_up's input; gate_up: ffn_norm, silu(gate) up ->
  down's input (its K groups); down: into the residual, its copy out as the next layer's qkv input;

and the LM head: output_norm -> LOGITS [slot][vocab] bf16 (the vocabulary padded to whole C objects per column: the
padding rows' logits are cut off, ``ModelOverlay._logits``).

The KV caches (KV): per layer and KV head linear groups of kp + 1 objects of the tile size T (``decode.kv``: kp K
objects, the V object; kp = 2, K as bfp16 hi + lo, or 1, single bfp16: ``kv_k``), 128 positions per group at
head_dim 64, 64 at 128 for objects of ~9.3 KB; then per layer and KV head a ring of two slots of T bytes with the
exact V values of the open records (``decode.vring``). Phase inputs are packed (``decode.protocol.packed_slots``).

A hidden size that is not a whole number of K chunks (Qwen3-Embedding-4B's 2560) pads the K of the phases that read
the residual stream (qkv, gate_up, the LM head) with zero super-blocks (``decode.image``); their inputs' padding k
must be zero too, since the helper's norm squares every input element: a step of more slots writes a layout whose
real k reach into the padding of the fewer-slot layouts, which the next step of fewer slots clears first
(``_clear_pads``).

The step's transaction is built once per S and pattern of sub-steps (and LM head or not) and patched per step: the
token objects, each column's range of each layer's cache per sub-step (A: address, length), its tail objects (A: the
ring slots of the sub-step's open records, or after a prefill the V panels of the first: record mode) and the record
drains (D: address, length). A column's weights between two attentions are one run of the (column-major) image: one A
task.
"""

import math

import numpy as np

from artlite.decode import gemv
from artlite.decode.attn import Attn, rope_tables, score_bits
from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.model import paged_formats
from artlite.decode.protocol import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_RESID,
    META,
    SLOTS,
    XB,
    interleave,
    packed_slots,
)
from artlite.decode.step import StepPlan
from artlite.decode.vring import PAIR
from artlite.gguf_model import GGUFFile
from artlite.model_config import LlamaConfig
from artlite.operators.design import COLS
from artlite.overlay import Sequence, WeightBuffer

from .decode import LlamaDecodeOperator
from .image import LlamaImage

PHASES = ("qkv", "att", "o", "gate_up", "down")
FLAGS = {"qkv": F_NORM, "o": F_RESID | F_EMITR, "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM}
KV_PAD = 4096  # positions past the caches (the last layer's): the prefill's slices of a cache (its largest chunk's
# runtime offsets range over the caches and this pad, the range every chunk size's parameter takes: ``prefill.MAX_T``)


def check_formats(image: LlamaImage) -> set[str]:
    """The image's tile formats; NotImplementedError for those the decode does not read (yet)."""
    fmts = {b.fmt for r in image.regions for b in r.blocks}
    missing = sorted(f for f in fmts if not gemv.supported(f, image.nsb))
    if missing:
        raise NotImplementedError(f"{', '.join(missing)} tiles: the decode GEMV has no kernels for them")
    return fmts


class LlamaOverlay(AttentionOverlay):
    """A Llama GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices),
    ``vocab``: LM rows (default: the padded vocabulary), ``max_positions``: the caches' capacity, ``kv_k``: the caches'
    K precision ("single": one bfp16 block per 8 dims, "hilo": hi + lo blocks), ``weights``: the image's buffer
    allocated ahead. One cache kind, one column per KV head, one ring state for every layer (key 0 of ``vring``: every
    step runs every layer)."""

    CONFIG, IMAGE = LlamaConfig, LlamaImage
    AHEAD = True
    NORMED = frozenset({"qkv", "gate_up", "lm"})

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "single", weights: WeightBuffer | None = None):
        super().__init__(gguf, layers=layers, vocab=vocab, kv_k=kv_k, weights=weights)
        cfg, img = self.cfg, self.image
        fmts = check_formats(img)
        self.hd, self.nh = cfg.head_dim, cfg.q_per_kv
        self.q_heads = self.nh
        self.at = at = Attn(hds=(self.hd,), rot=(self.hd // 2,), chain=1, heads=self.nh, kp=self.kp,
                            qk_norm=cfg.qk_norm, qk_weights=cfg.qk_norm, eps=cfg.rms_norm_eps, a_bytes=self.T)
        gu = {b.R for L in self.layers for b in img.region(L, "gate_up").blocks}
        self.op = LlamaDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                      eps=cfg.rms_norm_eps, attn=at, gu_win=gu != {16}, paged=paged_formats(gguf, fmts))
        self.pg = at.pg[self.hd]  # positions per cache group
        groups = -(-max_positions // self.pg)
        self.max_positions = groups * self.pg
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=groups),)
        self.uses = {L: CacheUse(kind=0, source=L, ring=0) for L in self.layers}
        self.period = PAIR
        self.inv_freq = cfg.rope_inv_freq()
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order (``lm`` False: without the LM head)."""
        return [(L, n) for L in self.layers for n in PHASES] + ([(-1, "lm")] if lm else [])

    def _flags(self, L: int, name: str) -> int:
        return FLAGS[name]

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        # (token t: of the sub-step from slot t on; two copies, one per transaction buffer: ``AHEAD``)
        self.toks = (take("ACT", SLOTS * XB), take("ACT", SLOTS * XB))
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name != "att":
                self._add_inputs((L, name))
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in self.layers for h in range(self.kv_heads)}
        # (past the last layer's caches: the prefill's slices)
        self._plan_caches(self.layers, pad=2 * KV_PAD // self.pg * self.G * self.T)
        self._sizes()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _attention_scale(self, L: int) -> tuple[int, int]:
        return score_bits(1 / math.sqrt(self.hd))

    def _qk_weights(self, L: int) -> dict[str, np.ndarray]:
        """(Qwen3's q / k norms: per dim weights)"""
        if not self.cfg.qk_norm:
            return {}
        return {n: np.asarray(self.gguf.tensor(f"blk.{L}.attn_{n}_norm.weight").dequantize(), np.float32).reshape(-1)
                .view(np.uint8) for n in ("q", "k")}

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        L, name = phase
        S, dev, h8, ffn = st.S, self.dev, self.H // COLS, self.cfg.intermediate_size // COLS
        nq = self.nh * self.hd
        for c in range(COLS):
            if name == "qkv":  # [q nh hd | K hd | V hd] per slot -> its q object, its raw K / V object
                reg = self.att_reg[(L, c)]
                seq.drain("C", c, dev.address("ACT", reg + 4 * META), S * 2 * nq, (C_LEN // 2, S, XB // 4, C_LEN // 2))
                seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 4 * META), S * 4 * self.hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name == "o":
                self._values_to(seq, c, (L, "gate_up"), e0=h8 * c, n=h8, S=S)
            elif name == "gate_up":
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "down":
                i = self.layers.index(L)
                nxt = (self.layers[i + 1], "qkv") if i + 1 < len(self.layers) else (-1, "lm")
                self._values_to(seq, c, nxt, e0=h8 * c, n=h8, S=S)
            else:
                self._logits_out(seq, c, S)

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        return self._substeps(p, S, self.hd)

    def _rope(self, kind: int, p: int, S: int) -> np.ndarray:
        return rope_tables(self.inv_freq, range(p, p + S), self.hd // 2)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, position: int, *, logits: bool = True) -> np.ndarray | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the embeddings at positions position .. position + S
        - 1 (the positions before are cached). Returns the logits (S, vocab) bf16 (a view of the NPU's buffer, valid
        until the next step), or without ``logits`` None (only the caches are updated)."""
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        xf = x
        self._write_loadr(xf)
        self._write_input(self.inp[(self.layers[0], "qkv")][0], interleave(xf, packed_slots(S)).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L's qkv phase (-1: the LM head) read in the last step, of a slot (its bf16
        copy, as fp32)."""
        return self._read_input(self.inp[(L, "lm" if L < 0 else "qkv")][0], slot, self.H)

    def attn_out(self, L: int, slot: int = 0) -> np.ndarray:
        """The attention's output o (heads hd) of layer L in the last step, of a slot (bf16 values, fp32)."""
        rg = self.inp[(L, "o")][0]
        return self._read_input(rg, slot, rg.K)


__all__ = ["KV_PAD", "LlamaOverlay", "check_formats"]
