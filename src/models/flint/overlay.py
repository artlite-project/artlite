"""A Flint model on its decode overlay (``flint/decode.py``): buffers, the transaction of a step of S <= 4 token slots,
the step.

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (per layer kind and
sub-step its positions and cache ranges, the RoPE tables of the sliding-window layers) and the slots' residual slices
(F_LOADR: the embeddings, fp32), then per layer L (column c, KV head h = c / 2, j = c % 2):

- qkv (normed: ``attn_norm``) -> the KV head's attention region [q objects SLOTS | raw K / V objects SLOTS]: the column's
  four query heads at heads 4 j .. 4 j + 3 of each q object, its halves of K_h and V_h (dims 64 j ..) in the raw K / V
  objects; att (the operator cores, a chain of 2 per KV head, one sub-step of up to 4 slots: ``Attn`` of 32 query rows)
  -> o into o's input (its two K groups of 2048: KV heads 0, 1 and 2, 3), the new positions' records (column 2 h + 1) into
  the cache; o (F_RESID, F_EMITR) -> the FFN's input;
- the dense layers (0, 1): gate_up (normed: ``ffn_norm``), silu(gate) up -> down's input (3 K groups); down (F_RESID,
  F_EMITR) -> the next layer's qkv;
- the MoE layers: moe_in (normed: ``ffn_norm``; the shared expert's gate / up, the router's logits) | the route object |
  sdown (the shared expert's down, ungated, into the residual) | the expert object | the combine -> the next layer's qkv
  (``moe.steps``, ``decode.design``'s "MoE": the operator phase "moe");

and the LM head (normed: ``output_norm``) -> LOGITS [slot][vocab] bf16.

The KV caches (KV): per layer and KV head linear groups of kp + 1 objects of the tile size T (``decode.kv``: 64 positions per
group at head_dim 128), a sliding-window layer's after a guard of the window's groups (8: keys p - 512 .. p are in the 8
groups before p's at most); then per layer and KV head a ring of two slots of T bytes with the exact V values of the open
records (``decode.vring``). The full layers have no positional encoding (their kind's rotated pairs: none).

The step's transaction is built once per S and pattern of sub-steps (and LM head or not) and patched per step: each
column's range of each layer's cache per sub-step (A: address, length), a chain end's tail objects (A: the ring slots of
the sub-step's open records, or after a prefill the V panels of the first) and the record drains of column 2 h + 1 (D:
address, length). A column's weights between two operator phases are one run of the (column-major) image: one A task;
an MoE layer's run ends with its sdown (the array pushes its experts' slabs next), the next one goes after the array's
last packet.
"""

import math
import os

import numpy as np

from artlite.decode import gemv
from artlite.decode import phases as mp
from artlite.decode.attn import Attn, rope_tables, score_bits
from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.head import HeadView
from artlite.decode.model import InputRegion
from artlite.decode.protocol import (
    B_F32,
    B_GU,
    B_PAIR,
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_RESID,
    M_XC0,
    META,
    SLOTS,
    XB,
    XPAY,
    interleave,
    packed_slots,
)
from artlite.decode.step import StepPlan
from artlite.decode.vring import PAIR
from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig
from artlite.moe import route as MR
from artlite.moe.steps import CP_BYTES, MoeTransfers
from artlite.operators import cdo
from artlite.operators.design import COLS
from artlite.overlay import Sequence, WeightBuffer

from .decode import FlintDecodeOperator
from .image import FlintImage

PHASES = {"dense": ("qkv", "att", "o", "gate_up", "down"), "moe": ("qkv", "att", "o", "moe_in", "sdown", "moe")}
FLAGS = {"qkv": F_NORM, "o": F_RESID | F_EMITR, "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM,
         "moe_in": F_NORM, "sdown": F_RESID}
MOE_BLOCKS = {"sgu": B_GU, "router": B_PAIR | B_F32}  # (moe_in's blocks: ``decode.design``)
CHAIN = 2  # columns per KV head
KV_PAD = 8192  # positions past the caches (the last layer's): the prefill's slices of a cache
KINDS = ("swa", "full")  # the cache kinds: token object t = SLOTS kind + a, of the sub-step from slot a on
FORMATS = frozenset({"Q4_1_64", "BFP16"})  # the tile formats the overlay runs (the official file's and its router's)


class FlintOverlay(AttentionOverlay):
    """A Flint GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices; the first
    one reads x), ``vocab``: LM rows (default: the padded vocabulary), ``max_positions``: the caches' capacity, ``kv_k``:
    the caches' K precision ("single": one bfp16 block per 8 dims, "hilo": hi + lo blocks), ``weights``: the image's
    buffer allocated ahead; ``head``: the LM rows laid out with a DFlash2 drafter's head (``FlintImage``; a
    ``decode.dflash.overlay.DFlashTarget``)."""

    CONFIG, IMAGE = FlintConfig, FlintImage
    KV_PAD = KV_PAD
    CHAIN = CHAIN
    OPS = ("att", "moe")  # (the operator cores' phases; "moe": the MoE transfers after moe_in and sdown)
    NORMED = frozenset({"qkv", "gate_up", "moe_in", "lm"})

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "hilo", weights: WeightBuffer | None = None, head: bool = False):
        image = FlintImage(gguf, layers=layers, vocab=vocab, head=True) if head and weights is None else None
        super().__init__(gguf, layers=layers, vocab=vocab, kv_k=kv_k, weights=weights, image=image)
        cfg, img = self.cfg, self.image
        assert cfg.num_key_value_heads * CHAIN == COLS
        fmts = img.formats()
        missing = sorted(f for f in fmts if not gemv.supported(f, img.nsb))
        if missing:
            raise NotImplementedError(f"{', '.join(missing)} tiles: the decode GEMV has no kernels for them")
        other = sorted(fmts - FORMATS)
        if other:  # (unsloth's UD-Q4_K_XL: a layer step on its Q8_0 tiles of 8 rows does not end; its 6 formats need the
            # main cores' kernels as pages beside the attention cores' own)
            raise NotImplementedError(f"{gguf.path}: {', '.join(other)} tiles: Flint runs the official int4 file "
                                      f"(model_config.flint_gguf(); its Q4_1 pairs as Q4_1_64), not yet other files")
        self.hd, self.nh = cfg.head_dim, cfg.q_per_kv
        self.q_heads = self.nh
        self.at = at = Attn(hds=(self.hd,), rot=(cfg.rot("swa"), cfg.rot("full")), chain=CHAIN, heads=self.nh,
                            kp=self.kp, qk_norm=True, qk_weights=True, eps=cfg.rms_norm_eps, a_bytes=self.T, sinks=True,
                            moe=True)
        self.op = FlintDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                      eps=cfg.rms_norm_eps, attn=at, experts=cfg.n_experts, used=cfg.n_experts_used)
        self.pg = at.pg[self.hd]  # positions per cache group
        self.guard = -(-(cfg.sliding_window - 1) // self.pg)  # (groups before a sliding-window cache's first one)
        full = -(-max_positions // self.pg)
        self.max_positions = full * self.pg
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=self.guard + full, guard=self.guard,
                                window=cfg.sliding_window),
                      CacheKind(hd=self.hd, pg=self.pg, groups=full))
        self.uses = {L: CacheUse(kind=KINDS.index(cfg.kinds[L]), source=L, ring=L) for L in self.layers}
        self.period = PAIR
        self.inv_freq = {k: cfg.rope_inv_freq(k) for k in KINDS}
        self.moe_steps = MoeTransfers(self)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order (``lm`` False: without the LM head)."""
        out = [(L, n) for L in self.layers for n in PHASES["moe" if self.cfg.moe_layer(L) else "dense"]]
        return out + ([(-1, "lm")] if lm else [])

    def _flags(self, L: int, name: str) -> int:
        return FLAGS[name]

    def _input(self, L: int) -> tuple[int, str]:
        """The phase that reads layer L's input (L = -1: the LM head)."""
        return (-1, "lm") if L < 0 else (L, "qkv")

    def _next(self, L: int) -> tuple[int, str]:
        """The phase that reads layer L's output."""
        i = self.layers.index(L)
        return self._input(self.layers[i + 1] if i + 1 < len(self.layers) else -1)

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        self.toks = (take("ACT", len(KINDS) * SLOTS * XB),)
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name not in self.OPS:
                self._add_inputs((L, name))
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token SLOTS kind + a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in self.layers for h in range(self.kv_heads)}
        # (per MoE layer its route object, per column its routing's objects and expert object; PART)
        moe = [L for L in self.layers if self.cfg.moe_layer(L)]
        self.xch = {L: take("ACT", XB) for L in moe}
        self.cp = {L: take("ACT", COLS * (CP_BYTES + XB)) for L in moe}
        self.part = take("ACT", COLS * self.part_col)
        self._plan_caches(self.layers, pad=KV_PAD // self.pg * self.G * self.T)
        self._sizes()

    @property
    def part_col(self) -> int:
        """PART's bytes per column: the combine's metadata object, then per slot and half of the column's slice the 8
        columns' partials of it ([slot][half][column][hidden / 16] fp32 on the slot's grid: one object's payload each,
        ``decode.design``'s "MoE")."""
        assert self.H * 2 == XPAY, "a half of a slot's slice of the 8 columns in one X object's payload"
        return XB + 2 * SLOTS * XPAY

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """(the control route's registers free: ``moe.route.check_route``)"""
        MR.check_route(cdo.values(os.path.dirname(self.op._program.insts_path)), range(COLS))
        super()._load()

    def _attention_scale(self, L: int) -> tuple[int, int]:
        return score_bits(1 / math.sqrt(self.hd))

    def drafter_view(self) -> HeadView | None:
        """(a DFlash2 drafter's head: the first ``draft_cols`` rows of each column of the LM region)"""
        return HeadView(self.image, self.image.draft_cols) if self.image.draft_cols else None

    def _qk_weights(self, L: int) -> dict[str, np.ndarray]:
        """(the q / k norms' weights per dim)"""
        return {n: np.asarray(self.gguf.tensor(f"blk.{L}.attn_{n}_norm.weight").dequantize(), np.float32).reshape(-1)
                .view(np.uint8) for n in ("q", "k")}

    def _sinks(self, L: int, h: int) -> np.ndarray:
        """(KV head h's query heads 8 h .. 8 h + 7: their sinks times log2 e)"""
        sk = np.asarray(self.gguf.tensor(f"blk.{L}.attn_sinks.weight").dequantize(), np.float64).reshape(-1)
        return (sk[self.nh * h:self.nh * (h + 1)] / math.log(2)).astype(np.float32)

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        return mp.groups(self.image.region(L, name), FLAGS[name], S, block_flags=MOE_BLOCKS if name == "moe_in" else None,
                         slots=self.SLOTS)

    def _meta(self, L: int, name: str, rg: InputRegion, m: np.ndarray) -> np.ndarray:
        if name == "sdown":  # (its panels in the side chunk: the experts' gate / up read moe_in's)
            m[M_XC0] = self.op.max_g - 1
        return m

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """(and per MoE layer its route object's and expert objects' templates, the combine's)"""
        return super()._meta_writes(S) + self.moe_steps.meta_writes(S)

    def _weights(self, seq: Sequence, phase: tuple[int, str], gi: int) -> None:
        """(``moe.steps``: a column-major image's tiles come with the runs)"""

    def _begin(self, seq: Sequence) -> None:
        self.moe_steps.route(seq)  # (the control packets' route; the routing's shim locks)

    def _gemv(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        if phase[1] in ("moe_in", "sdown"):  # (their inputs and outputs: the MoE operator phase's, ``moe.steps``)
            return
        super()._gemv(seq, st, phase)

    def _operator(self, seq: Sequence, st: StepPlan, L: int, name: str, weights: list[tuple[int, str]]) -> None:
        if name == "moe":
            self.moe_steps.build(seq, L, st.S, self._next(L), weights or None)
        else:
            super()._operator(seq, st, L, name, weights)

    def _outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """The C drains of a GEMV phase (after its last group)."""
        L, name = phase
        S, dev, h8, ffn = st.S, self.dev, self.H // COLS, self.cfg.intermediate_size // COLS
        for c in range(COLS):
            if name == "qkv":  # (per slot: q of heads 4 j .. 4 j + 3, then the K and V halves)
                h, j = divmod(c, CHAIN)
                reg = self.att_reg[(L, h)] + 4 * META
                qh = self.nh // CHAIN
                seq.drain("C", c, dev.address("ACT", reg + 2 * self.hd * qh * j), S * 2 * qh * self.hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
                half = self.hd // CHAIN
                seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 2 * half * j), S * 2 * 2 * C_LEN,
                          (C_LEN // 2, S, XB // 4, self.hd // 2))
            elif name == "o":
                self._values_to(seq, c, (L, "moe_in" if self.cfg.moe_layer(L) else "gate_up"), e0=h8 * c, n=h8, S=S)
            elif name == "gate_up":
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "down":
                self._values_to(seq, c, self._next(L), e0=h8 * c, n=h8, S=S)
            else:
                self._logits_out(seq, c, S)

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        """(both kinds: one head dim)"""
        return self._substeps(p, S, self.hd)

    def _rope(self, kind: int, p: int, S: int) -> np.ndarray:
        kn = KINDS[kind]
        return rope_tables(self.inv_freq[kn], range(p, p + S), self.cfg.rot(kn))

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
        self._write_loadr(x)
        self._write_input(self.inp[(self.layers[0], "qkv")][0], interleave(x, packed_slots(S)).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L's qkv phase (-1: the LM head) read in the last step, of a slot (its bf16
        copy, as fp32)."""
        return self._read_input(self.inp[self._input(L)][0], slot, self.H)

    def ffn_in(self, L: int, slot: int = 0) -> np.ndarray:
        """The residual stream after layer L's attention (its FFN's input, bf16 as fp32) in the last step, of a slot."""
        return self._read_input(self.inp[(L, "moe_in" if self.cfg.moe_layer(L) else "gate_up")][0], slot, self.H)

    def attn_out(self, L: int, slot: int = 0) -> np.ndarray:
        """The attention's output o (heads hd) of layer L in the last step, of a slot (bf16 values, fp32)."""
        return np.concatenate([self._read_input(rg, slot, rg.K) for rg in self.inp[(L, "o")]])


__all__ = ["KV_PAD", "FlintOverlay"]
