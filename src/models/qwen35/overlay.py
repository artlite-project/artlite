"""A qwen35 model on the multi-slot decode overlay (``qwen35/decode.py``): buffers, the transaction of a step of S
token slots, the step.

``QwenOverlay`` lays out the weight image (``image.QwenImage``, packed at ``load``), the norm-weight objects (NORM),
every phase's input (ACT: per K group [metadata object | x slot-interleaved], ``decode.protocol.region_bytes``; a producer
drains its outputs straight into its consumer's x), the residual-slice objects (F_LOADR), the GDN layers' objects (GOBJ:
per column [header | conv q / k | K | per pair of v heads: conv v, deltas], a chunkwise core's groups: ``gdn``) and states (GSTATE: per column its
panels, as the prefill reads them), the attention layers' objects (per column an F_ATT object per sub-step, then per
column its q objects; the raw K / V objects) and KV caches (KV, ``decode.kv``'s d-block-major layout, and per KV head a
ring of the open V records' values: ``decode.vring``), and the logits ([slot][vocab] bf16).

A step is one dispatch of the transaction of its slot count and the attention's sub-steps (``subs``; built on first
use), patched per step with the positions (the KV cache and ring reads, the record drains, the F_ATT objects) and the
GDN layers' lazy-commit words (``gdn``). The overlay owns the layout, the generic phases and the step; the transfers of a
layer kind and what the position patches into them are a part each: ``gdn_steps`` (GDN layers), ``attn_steps`` (attention
layers), ``moe_steps`` (MoE FFNs), over ``steps``' records; the lazy commit's and chunkwise blocks' host state is
``cursor.GdnCursor`` (``cur``), the conv states' copies between the decode and the prefill ``conv_state.ConvHandoff`` (``conv``).

The prefill (``qwen35/prefill.py``) runs on these buffers: the GDN states (GSTATE: per column its panels), the KV
caches, the norm weights (NORM: contiguous bf16; the decode reads them as overlapping A objects), and the conv states
in GHDR (``conv_state``: per GDN layer and column a slot of ``hslot`` bytes, the 3 rows x CH bf16 at ``CS_OFF``): the host
copies them there from the conv objects' histories before a prompt chunk (``ConvHandoff.for_prefill``) and back as their
base rows before the next decode step. Per step
the columns load the slots' residual slices (F_LOADR), then per layer

- GDN layer: gdn_in (normed; F_MSG: its rows to the GDN cores) | the GDN step (``gdn``) | gdn_out;
- attention layer: attn_in (normed; q + gate to the F_ATT regions, the column's parts of K / V to the raw objects) | the
  attention (the cache through A; o and the records relayed through C, ``attn``'s chains) | attn_out;
- gate_up | down (its K groups); an MoE layer (``QwenConfig.moe``) instead: moe_in (normed: the shared expert's gate /
  up, the router's logits, the shared gate's logit) | the route object (XCH: the logits of every column; the helper
  core's routing: ``moe.route``; its control packets out to CP, streamed back into the shim's TileControl: the
  expert slab's BDs pushed by the array) | sdown (the shared expert's down x its gate, into the residual) | the expert
  object (CP, after the routing's drain: the column's items; its sub-phases on the main core: gate / up, down into acc)
  | the combine (PART: the columns' partials of the slice, into the residual, then out) (``decode.design``'s "MoE");

and the LM head (normed) -> LOGITS. The transfers go out in rounds over the columns (a phase's weights, inputs,
outputs): a channel holds 4 queued tasks, and the oldest one that a fifth waits for must not need a later round. A
phase's inputs wait for the drains before them (``Sequence.sync``: its producers' outputs are in DDR), its weights go
out before that wait (they stream into the memtile meanwhile). A column-major image (``image.QwenImage``) streams a
column's weights between two attentions (their caches on A) as one run, its phases' norm-weight objects in it; a
region-major one per phase and K group (the norm-weight objects from NORM).
"""

import itertools
import math
import os

import numpy as np
from ml_dtypes import bfloat16

from artlite import txn
from artlite.decode import gemv
from artlite.decode import phases as mp
from artlite.decode.model import CacheRows, ModelOverlay, Snapshot, host_bytes, paged_formats
from artlite.decode.protocol import (
    B_F32,
    B_GU,
    B_KEEP,
    B_PAIR,
    F_EMITR,
    F_GU,
    F_MSG,
    F_NORM,
    F_RESID,
    F_WS,
    M_FLAGS,
    M_SKIP,
    M_SP,
    M_XC0,
    MAX_K,
    META,
    NX_PER_NORM,
    OBJ_K,
    XB,
    XPAY,
    interleave,
)
from artlite.decode.vring import substeps
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.moe import route as MR
from artlite.operators import cdo
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import attn, gdn
from .attn_steps import AttentionSteps
from .conv_state import HSLOT, ConvHandoff, check_slot
from .cursor import GdnCursor
from .decode import QwenDecodeOperator
from .gdn_steps import GdnTransfers
from .image import NORMS, QwenImage, gdn_v_heads, tensor_f32
from .moe_steps import CP_BYTES, MoeTransfers
from .steps import Step
from .words import F_ATT, F_GDN, F_GIN, M_ALANE, M_AMASK, M_BRANCH, M_HPAN

FLAGS = {"eh": F_RESID | F_EMITR,  # (a drafter's first phase: ``draft``)
         "gdn_in": F_NORM | F_MSG, "attn_in": F_NORM, "gdn_out": F_RESID | F_EMITR, "attn_out": F_RESID | F_EMITR,
         "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM, "moe_in": F_NORM, "sdown": F_RESID | F_WS}
MOE_BLOCKS = {"sgu": B_GU, "router": B_PAIR | B_F32, "sgate": B_PAIR | B_KEEP}  # (moe_in's blocks: ``decode.design``)
MOE_MAX_K = 2048  # (an MoE model's K groups at most: ``image.payload``)
KV_PAD = 4096 - 256  # positions past the caches (the last layer's): the prefill's sub-chunk slices


def slots_of(cfg: QwenConfig, fmts: set[str], drafts: bool) -> int:
    """A step's token slots at most: 8 for a dense model of a KV head per 2 columns whose formats the decode GEMV runs at
    8 slots (``decode.gemv``), either Q4_K and Q6_K only (no slower at S 1, S 4 3% faster) or with ``drafts`` (a drafter's
    steps of several tokens): other formats' designs of 8 slots are slower at S 1 (the 9B's Q5_K +1.4%; Swift 1.5's IQ
    formats, their tables' layout with one X object, +2%), which only the drafts pay for (Swift 1.5's MTP layer at 1K:
    +10% on prose). Else 4: an MoE design has no broadcast input, a KV head per 4 columns makes the one-pass attention
    slower at S 1."""
    return 8 if (not cfg.moe and COLS // cfg.num_key_value_heads == 2 and all(gemv.supported(f, 4, 8) for f in fmts)
                 and (fmts <= {"Q4_K", "Q6_K"} or drafts)) else 4


def chunk_of(cfg: QwenConfig) -> bool:
    """Whether a model on a design of 4 slots runs the GDN core chunkwise (``gdn``'s "Chunkwise"): an MoE model, whose
    drafter's steps of 2 .. 4 tokens gain 1.5-4.5% for 0.4-0.8% at S 1. The token-by-token core stays the better one for a model
    without a drafter, which only runs S 1."""
    return bool(cfg.moe)


def _text_rope(n: int) -> np.ndarray:
    return np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1)


class QwenOverlay(ModelOverlay):
    """A qwen35 GGUF model on the multi-slot decode overlay. ``layers``: the layers to run (default: all; tests run
    slices), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the KV caches' capacity, ``weights``: the
    image's buffer allocated ahead (its image's layers and vocabulary), ``formats``: tile formats the design runs
    besides the image's (a drafter's: ``draft``; past three formats, all but the model's largest as code pages),
    ``gu``: the rows per tile of a drafter's gate / up tiles besides the image's (``draft.draft_gate_up``: gate / up
    tiles of other than 16 rows take ``decode.design``'s windows), ``image``: an image laid out already
    (``ModelOverlay``'s: a drafter's view of its model's).

    ``rope`` (max_positions, 3): the M-RoPE position (T, H, W) of each cache slot, which the step's and the prefill's
    rotations take; the slot's index (text from position 0) until a model writes others (an image's tokens, the
    text after it: ``models.qwen35``), again after ``reset``."""

    CONFIG, IMAGE = QwenConfig, QwenImage
    NORMED = frozenset(NORMS)

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, weights: WeightBuffer | None = None, formats: tuple[str, ...] = (),
                 image: QwenImage | None = None, slots: int | None = None, gu: frozenset[int] = frozenset(),
                 drafts: bool | None = None):
        """slots: a step's token slots at most (default ``slots_of``, ``drafts``: whether a drafter's steps of several
        tokens run, default: the file has an MTP layer of its own)."""
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights, image=image)
        cfg, img = self.cfg, self.image
        fmts = {b.fmt for r in img.regions + img.slabs for b in r.blocks} | set(formats)
        gu_win = img.gu_rows() | gu != {16}  # (``decode.design``'s gate / up windows)
        slots = slots or slots_of(cfg, fmts, bool(cfg.nextn_layers) if drafts is None else drafts)
        att = [cfg.attention[L] for L in self.layers]
        self.gsh = gdn.GdnShape(cfg, slots=slots, chunk=chunk_of(cfg)) if not all(att) else None
        self.ash = attn.AttnShape(cfg, max_positions, a_bytes=img.tile_bytes, slots=slots) if any(att) else None
        self.max_positions = self.ash.max_positions if self.ash else max_positions
        self.rope = _text_rope(self.max_positions)
        self.period = 4 * attn.POS_PER_GROUP  # (the steps' sub-steps repeat from group 3 on: ``subs``)
        self.moe = cfg.moe
        self.op = QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=img.tile_bytes,
                                      eps=cfg.rms_norm_eps, gdn=self.gsh, attn=self.ash,
                                      moe=(cfg.n_experts, cfg.n_experts_used) if cfg.moe else None,
                                      max_k=MOE_MAX_K if cfg.moe else MAX_K,
                                      paged=paged_formats(gguf, fmts, wide=cfg.moe and (max(img.R[f] for f in fmts) > 16
                                                                                        or gu_win)),
                                      slots=slots, bcast=slots == 8, gu_win=gu_win,
                                      pairs=not cfg.moe and any(b.pairs for r in img.regions for b in r.blocks))
        self.g8 = self.gsh is not None and slots == 8  # (the GDN core of 8 slots: ``gdn``'s "8 slots")
        self.g4 = self.gsh is not None and slots == 4 and self.gsh.chunk  # (chunkwise of 4: ``gdn``'s "Chunkwise", "4 slots")
        self.blocks = self.g4 or self.g8  # (a chunkwise core: the states at the ends of blocks of 8 positions, ``GdnCursor.half``)
        self.flags = FLAGS | ({"gdn_in": F_NORM | F_GIN} if slots == 8 else {})
        self.cur = GdnCursor()  # (the GDN layers' lazy commit and blocks)
        self.ends = True  # (8 slots) whether the step's chain ends its block: its state panels drain (``_variant``)
        self.gdn_steps, self.attn_steps, self.moe_steps = GdnTransfers(self), AttentionSteps(self), MoeTransfers(self)
        self.conv = ConvHandoff(self)  # (the conv states between the decode and the prefill)
        self._order = None if img.order is None else self._id_runs(img.order)  # (``_logits``)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        out = mp.groups(self.image.region(L, name), self.flags[name], S, block_flags=MOE_BLOCKS if name == "moe_in" else None,
                        f32_ab=name == "gdn_in" and self.g8, slots=self.SLOTS)
        if self.op.packed and self.SLOTS == 8:  # (the input's slots per k block, the host's per phase)
            for g in out:
                g.meta[M_SP] = self._sp(S, (L, name))
        return out

    def _sp(self, S: int, phase: tuple[int, str] | None = None) -> int:
        """(a design of 8 slots: gdn_out's input of 8, the GDN cores' outputs whole heads of 8 slots; attn_out's, one pass:
        an octet's rows)"""
        whole = phase is not None and (phase[1] == "gdn_out" or (phase[1] == "attn_out" and self.ash.one))
        return self.SLOTS if whole else super()._sp(S, phase)

    def _meta(self, L: int, name: str, rg: object, m: np.ndarray) -> np.ndarray:
        if name == "sdown":  # (its panels in the side chunk: the expert's gate / up reads moe_in's)
            m[M_XC0] = self.op.max_g - 1
        return m

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """(and an MoE model's: per layer its route object's and expert objects' templates, the combine's)"""
        out = super()._meta_writes(S)
        return out + self.moe_steps.meta_writes(S) if self.moe else out

    def _phases(self) -> list[tuple[int, str]]:
        cfg = self.cfg
        ffn = ("moe_in", "sdown") if self.moe else ("gate_up", "down")
        out = [(L, name) for L in self.layers for name in
               (("attn_in", "attn_out") if cfg.attention[L] else ("gdn_in", "gdn_out")) + ffn]
        return [*out, (-1, "lm")]

    def _input(self, L: int) -> tuple[int, str]:
        """The phase that reads layer L's input (L = -1: the LM head)."""
        return (L, "lm") if L < 0 else (L, "attn_in" if self.cfg.attention[L] else "gdn_in")

    def _plan(self) -> None:
        """Byte offsets in ACT, NORM, GOBJ, GSTATE, KV."""
        cfg = self.cfg
        self._layout = dict.fromkeys(("ACT", "NORM", "GOBJ", "GSTATE", "KV", "GHDR"), 0)
        take = self._take
        self.loadr = take("ACT", COLS * self.SLOTS * XB)  # [column][slot]
        for L, name in self._phases():  # (gdn_out: the GDN pads' sink past each x)
            self._add_inputs((L, name), extra=2 * self.SLOTS if name == "gdn_out" else 0)
        self.norm_off: dict[tuple[int, str], int] = {}
        self.n_norm = -(-self.H // (NX_PER_NORM * OBJ_K))  # norm-weight A objects of a phase
        nw = 2 * NX_PER_NORM * OBJ_K * (self.n_norm - 1) + self.T  # (the last object's extent)
        for L in self.layers:
            for name in ("attn_norm", "post_attention_norm"):
                self.norm_off[(L, name)] = take("NORM", nw)
        self.norm_off[(-1, "output_norm")] = take("NORM", nw)
        self.gobj, self.gstate, self.fatt, self.rawkv, self.kv_off, self.ghdr, self.ring_off = {}, {}, {}, {}, {}, {}, {}
        self._gobj0: dict[int, np.ndarray] = {}  # (``reset``: a GDN layer's objects as loaded)
        self.hslot = HSLOT
        if self.gsh is not None:
            check_slot(self.gsh.ch)  # (the conv state: the prefill's drain)
            self.npair = self.gsh.hc // 2
            self.gcol = 256 + XPAY * (self.gsh.n_slots if self.gsh.chunk else 5 + 3 * self.npair)
            self.ps = self.gsh.hc * 16 * gdn.PANEL
            take("GSTATE", 256)  # (the first panel object's header bytes)
        for L in self.layers:
            if cfg.attention[L]:  # ([column][F_ATT per sub-step], then [column][q objects SLOTS])
                self.fatt[L] = take("ACT", COLS * (self.ash.u + self.SLOTS) * XB)
                self.rawkv[L] = take("ACT", self.SLOTS * XB)
                self.kv_off[L] = take("KV", self.ash.kvh * self.ash.head_bytes)
            else:
                self.gobj[L] = take("GOBJ", COLS * self.gcol)
                self.gstate[L] = take("GSTATE", COLS * self.ps)
                self.ghdr[L] = take("GHDR", COLS * HSLOT)
        self.ghalf = 0
        if self.blocks:  # (the states twice: a block's base, and the state at its end, ``GdnCursor.half``)
            self.ghalf = take("GSTATE", self._layout["GSTATE"] - 256) - 256
        self.xch, self.cp = {}, {}
        if self.moe:  # (per MoE layer its route object, per column its routing's objects and expert object; PART)
            for L in self.layers:
                self.xch[L] = take("ACT", XB)
                self.cp[L] = take("ACT", COLS * (CP_BYTES + XB))
            self.part = take("ACT", COLS * self.part_col)
        self.fobj = np.array([[[(self.fatt[L] + (c * self.ash.u + u) * XB) // 4 for u in range(self.ash.u)]
                                for c in range(COLS)] for L in self.fatt], np.int64)  # ([layer][column][sub-step], words)
        self.gin = np.array([self.inp[(L, "gdn_in")][0].off // 4 for L in self.gobj], np.int64)
        self._branch: tuple | None = None  # (the step words as the metadata objects hold them: ``_set_branch``; None: unknown)
        r0 = self._layout["KV"]
        for L in self.kv_off:  # (per KV head two slots: ``ring``)
            self.ring_off[L] = take("KV", self.ash.kvh * 2 * self.T)
        self.rings = (r0, self._layout["KV"] - r0)
        if self.ash is not None:  # (the prefill's slices past the end; a step's second record; a tail fill's T bytes)
            take("KV", (KV_PAD // attn.POS_PER_GROUP + 1) * attn.OBJS * self.T)
        self._sizes()

    @property
    def part_col(self) -> int:
        """PART's bytes per column: the combine's metadata object, then per slot and half of the column's slice the 8
        columns' partials of it ([slot][half][column][hidden / 16] fp32 on the slot's grid: one object's payload each,
        ``decode.design``'s "MoE")."""
        assert self.H * 2 == XPAY, "a half of a slot's slice of the 8 columns in one X object's payload"
        return XB + 2 * self.SLOTS * XPAY

    def ring(self, L: int, h: int) -> int:
        """Byte offset in KV of KV head h's ring of attention layer L: 2 slots of T bytes, slot r & 1 the V values of
        record r ([d-block][position 8][8] bf16: ``decode.vring``)."""
        return self.ring_off[L] + 2 * h * self.T

    def fatt_object(self, L: int, c: int, u: int) -> int:
        """ACT offset of column c's F_ATT object of sub-step u of layer L."""
        return self.fatt[L] + (c * self.ash.u + u) * XB

    def q_object(self, L: int, c: int, s: int) -> int:
        """ACT offset of column c's q object of slot s of layer L."""
        return self.fatt[L] + (COLS * self.ash.u + c * self.SLOTS + s) * XB

    def gdn_payload(self, L: int, c: int, k: int) -> int:
        """GOBJ offset of layer L's column c's object k's payload (k: conv q / k 0 .. 3, K 4, pair p 5 + 3 p ..; 8 slots:
        its payload slot, ``gdn``'s "8 slots")."""
        return self.gobj[L] + c * self.gcol + 256 + XPAY * k

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The norm weights (NORM), the attention layers' F_ATT objects; (MoE) the control route's registers free."""
        if self.moe:
            MR.check_route(cdo.values(os.path.dirname(self.op._program.insts_path)), range(COLS))
        norm = self.dev.view("NORM")
        for (L, name), off in self.norm_off.items():
            w = self._f32("output_norm.weight" if L < 0 else f"blk.{L}.{name}.weight").astype(bfloat16)
            norm[off:off + 2 * self.H] = w.view(np.uint8)
        self.dev.flush("NORM")
        for L in self.fatt:
            self._load_attention(L)

    def _f32(self, name: str) -> np.ndarray:
        return tensor_f32(self.gguf, name)

    def _load_attention(self, L: int) -> None:
        act = self.dev.view("ACT")
        qn = self._f32(f"blk.{L}.attn_q_norm.weight") * np.float32(math.log2(math.e) / 16)
        norms = np.concatenate([qn, self._f32(f"blk.{L}.attn_k_norm.weight")])
        for c in range(COLS):
            for u in range(self.ash.u):
                o = self.fatt_object(L, c, u)
                m = np.zeros(META, np.int32)
                m[M_FLAGS] = F_ATT
                act[o:o + 4 * META] = m.view(np.uint8)
                act[o + 4 * META:o + 4 * META + norms.nbytes] = norms.view(np.uint8)

    def _load_gdn(self, L: int) -> None:
        """Layer L's GOBJ: the header, the conv objects (weights, zero histories, pads), zero K and delta objects."""
        sh, cfg = self.gsh, self.cfg
        gobj = self.dev.view("GOBJ")
        blk = self.image.region(L, "gdn_in").block("qkv")
        cw = np.asarray(self.gguf.tensor(f"blk.{L}.ssm_conv1d.weight").raw, np.float32)
        dt, A = self._f32(f"blk.{L}.ssm_dt.bias"), self._f32(f"blk.{L}.ssm_a")
        norm = self._f32(f"blk.{L}.ssm_norm.weight")
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_SKIP] = F_GDN, sh.n_objects - 1
        if self.g8:  # (the helper's requests per step: ``gdn``'s ``hgdn``, its heads' and the step's end)
            m[M_HPAN] = sh.hc + 3
        for c in range(COLS):
            vh = gdn_v_heads(cfg, c)
            rows = blk.rows[c][blk.rows[c] >= 0]
            o = self.gobj[L] + c * self.gcol
            gobj[o:o + self.gcol] = 0
            gobj[o:o + 4 * META] = m.view(np.uint8)
            if sh.chunk:  # (the header's or the first P object's dt, A, ssm_norm; per conv block its weights)
                h = self.gdn_payload(L, c, gdn.H_SLOT) + (gdn.H_PRM if self.g8 else gdn.P_PRM)
                gobj[h:h + 640] = gdn.head_payload(dt[vh], A[vh], norm)
                for b, w in enumerate(gdn.conv_weights(cw[rows])):
                    a = self.gdn_payload(L, c, 0) + sh.weights_at(b)
                    gobj[a:a + w.size] = w
                continue
            for b, pay in enumerate(gdn.conv_payloads(sh, cw[rows], dt[vh], A[vh], norm)):
                k = b if b < 4 else 5 + 3 * ((b - 4) // 2) + (b - 4) % 2
                gobj[self.gdn_payload(L, c, k):self.gdn_payload(L, c, k) + XPAY] = pay

    def reset(self) -> None:
        """A new sequence: the GDN layers' conv histories, replay objects and states zeroed (a layer's objects as
        ``_load_gdn`` wrote them first: their conv weights take it ~0.5 ms a layer). A non-causal model's none of them:
        it never decodes, and each of its prompt chunks (position 0) starts from zero states itself (``QwenPrefill.run``:
        a scan that zeroes them, or ``zero_state`` first; d3-lite's 18.9 MB, ~0.8 ms of a short readout)."""
        super().reset()
        if self.cfg.causal:
            self.zero_state()
            dev = self.dev
            gobj, n = dev.view("GOBJ"), COLS * self.gcol if self.gobj else 0
            for L, o in self.gobj.items():
                if L in self._gobj0:
                    gobj[o:o + n] = self._gobj0[L]
                else:
                    self._load_gdn(L)
                    self._gobj0[L] = gobj[o:o + n].copy()
            dev.flush("GOBJ")
        self.cur.reset()
        self.reset_rope()

    def reset_rope(self) -> None:
        """The M-RoPE positions of text from position 0."""
        self.rope[:] = _text_rope(self.max_positions)

    def forget_branch(self) -> None:
        """The next step writes its step words again (``_set_branch``)."""
        self._branch = None

    @property
    def a_bds(self) -> int:
        """The host's BDs on A."""
        return getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)

    def zero_state(self) -> None:
        """The GDN layers' states (GSTATE) zero."""
        self.dev.view("GSTATE")[:] = 0
        self.dev.flush("GSTATE")

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

    def x_slot(self, phase: tuple[int, str], e: int, S: int) -> tuple[int, int, int]:
        """(ACT offset of input element e's k block in a phase's x of a step of S slots, the bytes left in its K group's x,
        the K group's x bytes of a design's slots: where a sink past it starts)."""
        sp = self._sp(S, phase)
        for rg in self.inp[phase]:
            if rg.k0 <= e < rg.k0 + rg.K:
                return rg.off + XB + 2 * sp * (e - rg.k0), 2 * sp * (rg.k0 + rg.K - e), 2 * self.SLOTS * rg.K
        raise ValueError(e)

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
                self._values_to(seq, c, to, e0=h8 * c, n=h8, S=S, whole=self.SLOTS == 8)

    def _runs(self, lm: bool) -> list[list[tuple[int, str]]]:
        """A column-major step's weight runs: its phases up to the first attention's attn_in, then from each
        attention's attn_out up to the next one's attn_in (or the end); an MoE layer's run ends with its sdown (the
        array pushes its experts' slabs next)."""
        out: list[list[tuple[int, str]]] = [[]]
        for phase in self._phases():
            if phase[1] == "lm" and not lm:
                break
            if phase[1] == "attn_out":
                out.append([])
            out[-1].append(phase)
            if phase[1] == "sdown":
                out.append([])
        return [r for r in out if r]

    def _build(self, S: int, lm: bool, subs: tuple[tuple[int, int, int], ...], b: int) -> Step:
        """The transaction of a step of S slots (``lm``: with the LM head; ``subs``: the attention's sub-steps; ``b``: its
        buffer, one: no ``AHEAD``)."""
        dev, cfg = self.dev, self.cfg
        seq = Sequence(self.op, inputs=("X",))
        st = Step(S, subs)
        if self.moe:  # (the control packets' route; the routing's shim locks)
            self.moe_steps.route(seq)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * self.SLOTS * XB), S * XB)
        runs = self._runs(lm) if self.image.column_major else []
        if runs:
            self._weight_run(seq, runs[0])
        self._head(seq, S)
        ffn = "moe_in" if self.moe else "gate_up"
        for i, L in enumerate(self.layers):
            if cfg.attention[L]:
                run = next((r for r in runs if r[0] == (L, "attn_out")), None)
                self.attn_steps.build(seq, st, L, S, run)
                self._phase(seq, (L, "attn_out"), S, (L, ffn))
            else:
                self.gdn_steps.build(seq, st, L, S)
                self._phase(seq, (L, "gdn_out"), S, (L, ffn))
            nxt = self._input(self.layers[i + 1]) if i + 1 < len(self.layers) else (-1, "lm")
            if self.moe:
                self.moe_steps.build(seq, L, S, nxt, next((r for r in runs if r[0] == nxt), None))
                continue
            self._phase(seq, (L, "gate_up"), S, (L, "down"))
            self._phase(seq, (L, "down"), S, nxt)
        if lm:
            self._phase(seq, (-1, "lm"), S, "LOGITS")
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        return st

    def _head(self, seq: Sequence, S: int) -> None:
        """Phases before the first layer's (a drafter's eh: ``draft``)."""

    def _variant(self) -> tuple:
        """(a step of an 8-slot design: whether its chain ends its block, ``ends``: only then its state panels drain)"""
        return (self.start, self.ends)

    def prepare(self, S: int, logits: bool = True, start: int | None = None) -> None:
        """(8 slots: the steps that end their block and those that do not)"""
        for ends in (True, False) if self.g8 else (True,):
            self.ends = ends
            super().prepare(S, logits, start)
        self.ends = True

    def patch_ahead(self, S: int, p: int) -> None:
        """(a step of a chain)"""
        if self.g8:
            self.ends = p % 8 + S >= 8
        super().patch_ahead(S, p)

    def passes(self, p: int, S: int) -> int:
        """The passes over the KV caches of a step at p .. p + S - 1 (a sub-step's two rounds: one, a KV head each)."""
        return len({(a, n) for a, n, _ in self.subs(p, S)})

    def subs(self, p: int, S: int) -> tuple[tuple[int, int, int], ...]:
        """The attention's sub-steps of a step at p .. p + S - 1 (``decode.vring.substeps``), a round each: (first slot,
        slots, how the chains divide the cache: by its group and slots alone, ``attn.AttnShape.split_of``)."""
        if self.ash is None:
            return ()
        if self.ash.one:
            return ((0, S, False),)
        return tuple((a, n, split) for a, n in substeps(p, S, attn.POS_PER_GROUP, self.ash.sub)
                     for split in self.ash.split_of((p + a) // attn.POS_PER_GROUP, n))

    def _set_position(self, st: Step, p: int) -> None:
        """A step's words that follow its position: a chunkwise core's state transfers (``GdnTransfers.patch_panels``),
        the attention's F_ATT objects and cache, ring and record transfers (``AttentionSteps.set_position``)."""
        if self.blocks:
            self.gdn_steps.patch_panels(st, p)
        if self.ash is not None:
            self.attn_steps.set_position(st, p)

    @property
    def can_branch(self) -> bool:
        """Whether its steps of 8 slots may branch (``step``)."""
        return self.SLOTS == 8 and self.ash is not None and self.ash.one and self.gsh is not None and self.gsh.slots == 8

    def _metas(self, S: int) -> None:
        """(the step words as the metadata objects left them: ``_set_branch`` writes them at the next step)"""
        super()._metas(S)
        self._branch = None

    def _branch_angles(self, p: int, L: int) -> None:
        """The F_ATT objects' angles of a step's last L slots at p (one pass: the step's slots' rows) from ``rope``."""
        c0, s0 = attn.rope_cs(self.cfg, self.rope[p + 8 - L:p + 8])
        at = self.fobj[:, :, 0, None] + META + 2 * attn.HD + 64 * (8 - L) + np.arange(64 * L)
        self.dev.view("ACT").view(np.float32)[at] = np.concatenate([c0, s0], 1).reshape(-1)
        self.dev.flush_at("ACT", np.unique(4 * at.ravel() & ~63))

    def _set_branch(self, branch: tuple[int, int] | None, p: int, S: int) -> None:
        """A step's words that follow its slots' positions and its branch (parent c, its L slots at p + c + 1 .., inside the 8
        positions of slot c's: ``step``): per F_ATT object (one pass) ``attn.step_words``; the gdn_in objects' branch word (8
        slots: c + 1 | L << 8, or 0). Written when they change."""
        c, L = branch or (-1, 0)
        key = (p % 8, S, c, L)
        if key == self._branch:
            return
        word = (c + 1) | L << 8 if branch else 0
        a32 = self.dev.view("ACT").view(np.int32)
        lines = []
        if self.ash is not None and self.ash.one:
            aword, w, lanes = attn.step_words(p, S, branch)
            fo = self.fobj[:, :, 0].ravel()
            a32[fo + M_BRANCH] = aword
            a32[fo[:, None] + M_AMASK + np.arange(4)] = w.view(np.int32)
            a32[fo + M_ALANE] = lanes
            lines.append(4 * (fo + M_BRANCH))
        if self.g8:
            a32[self.gin + M_BRANCH] = word
            lines.append(4 * (self.gin + M_BRANCH))
        if lines:
            self.dev.flush_at("ACT", np.unique(np.concatenate(lines) & ~63))
        self._branch = key

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, position: int, *, logits: bool = True, commit: bool = True,
             branch: tuple[int, int] | None = None) -> np.ndarray | None:
        """One step of S = len(x) <= SLOTS (4; 8 where ``slots_of``) token slots: x (S, hidden) the first layer's inputs (embeddings) at positions
        position .. position + S - 1 (the positions before are in the caches, the GDN states hold the committed ones and
        the previous step's accepted tokens: ``accept``). ``commit``: the GDN states after the step's tokens are written
        (all of them must then be accepted, ``accept(S)``), else the step's tokens are replayed by the next step as far
        as accepted (a chunkwise core of 4: the states are written at the ends of blocks of 8 positions whatever
        ``commit``, which then only says that all S are accepted: ``gdn``'s "Chunkwise"). ``branch`` (parent, L) (``can_branch``, a step of 8): its last L slots follow slot parent instead of slot 7 - L,
        at position + parent + 1 .. inside the 8 positions of slot parent's (position % 8 + parent + L <= 7: its attention
        records and GDN block; a tree of two branches: ``_set_branch``); None: a chain. Returns the slots' logits
        (S, vocab) bf16 (a view of the NPU's buffer, valid until the next step), or without ``logits`` None."""
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= self.SLOTS
        cur = self.cur
        assert cur.committed or cur.accepted, "accept() the previous lazy step's tokens first"
        assert branch is None or (S == 8 and self.can_branch and not commit and branch[1] in (1, 2)
                                  and 0 <= branch[0] and position % 8 + branch[0] + branch[1] <= 7), (position, S, branch)
        self.conv.for_decode(position)
        if branch is not None:  # (after a prefill's text rope; a step patched ahead: its branch slots' angles again)
            c, L = branch
            self.rope[position + 8 - L:position + 8] = self.rope[position + c + 1:position + c + 1 + L]
            if self._ahead is not None:
                self._branch_angles(position, L)
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._set_branch(branch, position, S)
        if self.blocks:  # (the block's lanes before the step; the step's words do not depend on commit)
            assert position == cur.gnext, (position, cur.gnext)
            self.ends = not self.g8 or position % 8 + S - (branch[1] if branch else 0) >= 8
            cur.gpos = position
            self.gdn_steps.set_words(nrep=position % 8, cshift=cur.cshift, commit=False)
        else:
            self.gdn_steps.set_words(nrep=cur.nrep, cshift=cur.cshift, commit=commit)
        cur.committed = commit
        self._write_loadr(x)
        first = self._input(self.layers[0])
        self._write_input(self.inp[first][0], interleave(x, self._sp(S, first)).view(np.uint8).reshape(-1))
        out = self._run(S, position, logits)
        cur.accepted = False
        if commit:  # (all S accepted)
            if self.blocks:
                cur.advance(S)
            else:
                cur.nrep, cur.cshift = 0, S
        return out

    def _ran(self, p: int, S: int) -> None:
        for L in self.fatt:
            self.vring.ran(L, p, S)

    def _ring_state(self) -> tuple:
        """(and whether a prefill's conv states wait in GHDR: the next step resets ``rope``)"""
        return (*super()._ring_state(), self.cur.conv_in_ghdr)

    @staticmethod
    def _id_runs(order: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """The buffer of ``_logits``'s copy and an LM order's runs of consecutive rows (id, its end, row)."""
        cuts = [0, *(np.flatnonzero(np.diff(order) != 1) + 1).tolist(), len(order)]
        return np.empty((8, len(order)), np.uint16), [(a, b, int(order[a])) for a, b in itertools.pairwise(cuts)]

    def _logits(self, S: int) -> np.ndarray:
        """(an image whose LM rows are not in id order, ``QwenImage.order``: in id order, the ids' rows, in a buffer valid
        until the next dispatch as the NPU's; an MTP drafter's table rows left out)"""
        if self._order is None:
            return super()._logits(S)
        self.dev.fetch("LOGITS", 0, 2 * S * self.vocab)
        x = self.dev.view("LOGITS")[:2 * S * self.vocab].view(np.uint16).reshape(S, self.vocab)
        out, runs = self._order
        for a, b, r in runs:
            out[:S, a:b] = x[:, r:r + b - a]
        return out[:S].view(bfloat16)

    def _lm_phase(self, seq: Sequence) -> None:
        if self.image.column_major:
            self._weight_run(seq, [(-1, "lm")])
        self._phase(seq, (-1, "lm"), 1, "LOGITS")

    def accept(self, n: int) -> None:
        """The previous step's first n tokens are accepted (the next step's positions start after them): its GDN
        layers replay them (unless it committed: then n must be all of them) and shift their conv histories (a chunkwise
        core of 4: the next step recomputes its block's lanes from the histories, ``GdnCursor.advance``)."""
        cur = self.cur
        assert 0 <= n <= self._last_s and not cur.accepted
        if cur.committed:
            assert n == self._last_s, "a committing step's tokens are all accepted"
        elif self.blocks:
            cur.advance(n)
        else:
            cur.nrep, cur.cshift = n, n
        cur.accepted = True

    def cache_rows(self) -> list[CacheRows]:
        """(per attention layer and KV head: groups of 32 positions, 2 objects each)"""
        if self.ash is None:
            return []
        rows = [self.kv_off[L] + h * self.ash.head_bytes for L in self.kv_off for h in range(self.ash.kvh)]
        return [CacheRows(np.array(rows), attn.OBJS * self.T, attn.POS_PER_GROUP)]

    def snapshot(self, position: int) -> Snapshot:
        """(and the GDN layers' states, conv objects with their replay objects, and conv states of the prefill, the lazy
        commit's cursor, ``rope``)"""
        out = super().snapshot(position)
        extra = out.extra
        if self.gsh is not None:
            for name in ("GSTATE", "GOBJ", "GHDR"):
                self.dev.fetch(name)
                v = self.dev.view(name)[self._gspan()] if name == "GSTATE" else self.dev.view(name)
                extra[name] = host_bytes(v.shape)
                extra[name][:] = v
        extra["cursor"] = self.cur.copy()
        extra["rope"] = self.rope.copy()
        return out

    def restore(self, snapshot: Snapshot) -> None:
        super().restore(snapshot)
        self.cur.assign(snapshot.extra["cursor"])
        if self.gsh is not None:
            for name in ("GSTATE", "GOBJ", "GHDR"):
                self.dev.view(name)[self._gspan() if name == "GSTATE" else slice(None)] = snapshot.extra[name]
                self.dev.flush(name)
        self.rope[:] = snapshot.extra["rope"]

    def _gspan(self) -> slice:
        """GSTATE's bytes a snapshot keeps: its states (chunkwise: the half that the next position's block starts
        from, the other is the next block's scratch; a prefill's next position: its half)."""
        if not self.blocks:
            return slice(None)
        cur = self.cur
        h = cur.gcur if cur.conv_in_ghdr else cur.half(cur.gnext)
        return slice(256 + h * self.ghalf, 256 + (h + 1) * self.ghalf)

    def tap_x(self, L: int) -> int:
        """ACT offset of the x of layer L's first phase (L = -1: the LM head): the residual stream that layer read, as
        [hidden / 8][slot][8] bf16 for the slots ``_sp`` of its step (the DFlash2 drafter reads it in place)."""
        return self.inp[self._input(L)][0].off + XB

    def move_tap_slots(self, L: int, S: int, src: list[int]) -> None:
        """After a step of S slots: layer L's tap of the slots ``src`` moved to its first slots."""
        sp = self._sp(S)
        off, n = self.tap_x(L), self.H * sp * 2
        self.dev.fetch("ACT", off, n)
        x = self.dev.view("ACT")[off:off + n].view(np.uint16).reshape(self.H // 8, sp, 8)
        x[:, :len(src)] = x[:, src].copy()
        self.dev.flush("ACT", off, n)

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L (-1: the LM head) read in the last step, of a slot (bf16 values, fp32)."""
        off, sp = self.tap_x(L), self._sp(self._last_s, self._input(L))
        self.dev.fetch("ACT", off, 2 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)[:, slot].reshape(-1)
        return v.view(bfloat16).astype(np.float32)

