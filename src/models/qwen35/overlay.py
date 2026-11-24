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
layers), ``moe.steps`` (MoE FFNs), over ``steps``' records; the lazy commit's and chunkwise blocks' host state is
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

Several sequences (``sequences``; a dense model on the design of 8 slots, which a model of several takes wherever
its formats run on one: the 9B's too): each has its KV caches and its GDN layers'
objects, states and conv states in buffers of its own (``SEQ``; ``ModelOverlay.select``), its lazy commit's cursor and
its ``rope``. A step's slots may be several sequences' (``decode.step.Part``), with no kernel of its own: each attention
runs a pass per sequence (``decode.attn256_steps``' "Several sequences") and each GDN layer a chain per sequence
(``gdn_steps``' "Several sequences": its words in its header object, its rows drained into its objects, its outputs
into its slots of gdn_out's staged x; a PARO checkpoint's: rot_g's, and the attention's passes write rot_o's), and every
slot's numbers are those of a step of its sequence alone.
"""

import dataclasses
import itertools
import math
import os
from collections.abc import Mapping

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode import attn256 as attn
from artlite.decode import gemv
from artlite.decode import phases as mp
from artlite.decode.attn256 import F_ATT, M_ALANE, M_AMASK, M_BRANCH
from artlite.decode.attn256_steps import AttentionSteps, pass_subs
from artlite.decode.design import PG_ONE_CHANNEL
from artlite.decode.model import CacheRows, ModelOverlay, Snapshot, StepKey, host_bytes, paged_formats
from artlite.decode.protocol import (
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
from artlite.decode.screen import Screen, Screened
from artlite.decode.step import Part
from artlite.decode.vring import substeps
from artlite.gguf_model import GGUFFile, tensor_f32
from artlite.model_config import QwenConfig
from artlite.moe import route as MR
from artlite.moe.steps import CP_BYTES, MOE_BLOCKS, MoeTransfers
from artlite.operators import cdo
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import gdn
from .conv_state import HSLOT, ConvHandoff, check_slot
from .cursor import GdnCursor
from .decode import QwenDecodeOperator
from .gdn_steps import GdnTransfers
from .image import NORMS, QwenImage, gdn_v_heads
from .paro_steps import ParoSteps, gdn_in_rows
from .steps import PanelTable, Step
from .words import F_GDN, F_GIN, M_HPAN

FLAGS = {"eh": F_RESID | F_EMITR,  # (a drafter's first phase: ``draft``)
         "gdn_in": F_NORM | F_MSG, "attn_in": F_NORM, "gdn_out": F_RESID | F_EMITR, "attn_out": F_RESID | F_EMITR,
         "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM, "moe_in": F_NORM, "sdown": F_RESID | F_WS}
MOE_MAX_K = 2048  # (an MoE model's K groups at most: ``image.payload``)
KV_PAD = 4096 - 256  # positions past the caches (the last layer's): the prefill's sub-chunk slices


def slots_of(cfg: QwenConfig, fmts: set[str], drafts: bool) -> int:
    """A step's token slots at most: 8 for a dense model of a KV head per 2 columns whose formats the decode GEMV runs at
    8 slots (``decode.gemv``), either Q4_K and Q6_K only (no slower at S 1, S 4 3% faster) or with ``drafts`` (a drafter's
    steps of several tokens) or of more formats than a design of 4 slots pages (``decode.design.PG_ONE_CHANNEL``: unsloth's
    dynamic quantizations, 12): other formats' designs of 8 slots are slower at S 1 (the 9B's Q5_K +1.4%; Swift 1.5's IQ
    formats, their tables' layout with one X object, +2%), which only the drafts pay for (Swift 1.5's MTP layer at 1K:
    +10% on prose). Else 4: an MoE design has no broadcast input, a KV head per 4 columns makes the one-pass attention
    slower at S 1."""
    return 8 if (runs_eight(cfg, fmts) and COLS // cfg.num_key_value_heads == 2
                 and (fmts <= {"Q4_K", "Q6_K"} or drafts or len(fmts) > PG_ONE_CHANNEL)) else 4


def runs_eight(cfg: QwenConfig, fmts: set[str]) -> bool:
    """Whether a model's formats run on the design of 8 slots: a dense model whose formats' GEMV does (what a model of
    several sequences takes, whatever ``slots_of`` prefers for one: a GDN chain per sequence is that design's)."""
    return not cfg.moe and all(gemv.supported(f, 4, 8) for f in fmts)


def chunk_of(cfg: QwenConfig) -> bool:
    """Whether a model on a design of 4 slots runs the GDN core chunkwise (``gdn``'s "Chunkwise"): an MoE model, whose
    drafter's steps of 2 .. 4 tokens gain 1.5-4.5% for 0.4-0.8% at S 1. The token-by-token core stays the better one for a model
    without a drafter, which only runs S 1."""
    return bool(cfg.moe)


def _text_rope(n: int) -> np.ndarray:
    return np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1)


class QwenOverlay(ParoSteps, ModelOverlay):
    """A qwen35 GGUF model on the multi-slot decode overlay. ``layers``: the layers to run (default: all; tests run
    slices), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the KV caches' capacity, ``weights``: the
    image's buffer allocated ahead (its image's layers and vocabulary), ``formats``: tile formats the design runs
    besides the image's (a drafter's: ``draft``; past three formats, all but the model's largest as code pages),
    ``gu``: the rows per tile of a drafter's gate / up tiles besides the image's (``draft.draft_gate_up``: gate / up
    tiles of other than 16 rows take ``decode.design``'s windows), ``image``: an image laid out already
    (``ModelOverlay``'s: a drafter's view of its model's), ``sequences``: the sequences whose caches and GDN states the
    device holds (module docstring; more than one: the design of 8 slots, a dense model's).

    ``rope`` (max_positions, 3): the M-RoPE position (T, H, W) of each cache slot of the selected sequence, which the
    step's and the prefill's rotations take; the slot's index (text from position 0) until a model writes others (an
    image's tokens, the text after it: ``models.qwen35``), again after ``reset``."""

    CONFIG, IMAGE = QwenConfig, QwenImage
    NORMED = frozenset(NORMS)
    # a design of 4 slots patches the next step's position while a step runs (``ModelOverlay.AHEAD``: the attention layers'
    # words, angles and transfers, 0.37 ms of the 35B-A3B's 36 ms step that the NPU otherwise waits for): each transaction
    # buffer reads F_ATT objects of its own (``fatt2``), the second buffer's transactions are copies of the first's, made
    # at their first use (``_transaction``). Not after a step that ``patch_ahead`` patched (a drafter's round: the next
    # step is the next round's, of other slots); a drafter's own steps (``draft``) follow none of their own; a design of
    # 8 slots writes its steps' branch words between them.
    STEPS_AHEAD = True
    AHEAD_PREPARED = False
    KV_PAD = KV_PAD  # (``decode.dflash.overlay.DFlashTarget``)
    SEQ = ("KV", "GOBJ", "GSTATE", "GHDR")

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, weights: WeightBuffer | None = None, formats: tuple[str, ...] = (),
                 image: QwenImage | None = None, slots: int | None = None, gu: frozenset[int] = frozenset(),
                 drafts: bool | None = None, sequences: int = 1, taps: tuple[int, ...] = ()):
        """slots: a step's token slots at most (default ``slots_of``, ``drafts``: whether a drafter's steps of several
        tokens run, default: the file has an MTP layer of its own); taps: a DFlash2 drafter's (a PARO checkpoint's: the
        layers whose input goes out as a plain copy too, ``paro_steps``; the image's ``taps``)."""
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights, image=image, sequences=sequences)
        cfg, img = self.cfg, self.image
        self.paro = img.paro  # (a PARO checkpoint's rotated inputs: ``paro_steps``)
        self.taps = tuple(L for L in taps if L in self.layers) if self.paro else ()
        assert set(self.taps) <= set(img.taps), (self.taps, img.taps)
        if self.paro and (slots not in (None, 4, 8) or (slots == 8 and cfg.moe)):
            raise ValueError("a PARO checkpoint: a design of 4 slots (a dense model's: 8)")
        fmts = img.formats() | set(formats)
        gu_rows = img.gu_rows() | gu  # (``decode.design``'s gate / up windows: tiles of gate / up phases of other rows; a
        # PARO checkpoint has none, its gate and up phases of their own)
        gu_win = bool(gu_rows) and gu_rows != {16} or img.split
        # (several sequences: the design of 8 slots where the file's formats run on one)
        # (a PARO checkpoint: as a model with drafts, its design of 8 slots no slower at S 1: the 27B int5's 3.52 tok/s
        # against 3.47 on 4)
        slots = slots or (8 if sequences > 1 and runs_eight(cfg, fmts) else
                          slots_of(cfg, fmts, self.paro or (bool(cfg.nextn_layers) if drafts is None else drafts)))
        if sequences > 1 and (slots != 8 or cfg.moe):  # (a GDN chain per sequence: the core whose rows come through DDR)
            raise ValueError(f"{sequences} sequences: a dense model whose formats run on the design of 8 slots "
                             f"(``runs_eight``: its tile formats, a screened LM head's plane among them)")
        if sequences > 8:
            raise ValueError(f"{sequences} sequences: a step has 8 slots, a sequence each at most")
        att = [cfg.attention[L] for L in self.layers]
        self.gsh = gdn.GdnShape(cfg, slots=slots, chunk=chunk_of(cfg), parts=self.paro and slots == 4) \
            if not all(att) else None
        # (a PARO checkpoint's: the gates in the attention core, the main core's program memory for the rotations and a
        # page of the formats: the 27B's int5)
        self.ash = attn.AttnShape(cfg, max_positions, a_bytes=img.tile_bytes, slots=slots, passes=sequences,
                                  core_gates=self.paro) if any(att) else None
        self.max_positions = self.ash.max_positions if self.ash else max_positions
        self.AHEAD = self.STEPS_AHEAD and slots == 4 and self.ash is not None
        # (the positions whose ``rope`` rows the last position patch took, and the rows it took: ``_ring_state``)
        self._rope_used = (0, 0, b"")
        self._round = False  # (``patch_ahead`` patched the next step: ``_run``)
        self._kept_rows: np.ndarray | None = None  # (the last step's rows of its caller's, where not its first: ``hiddens``)
        self._ropes = [_text_rope(self.max_positions) for _ in range(sequences)]  # (per sequence: ``rope``)
        self.period = 4 * attn.POS_PER_GROUP  # (the steps' sub-steps repeat from group 3 on: ``subs``)
        self.moe = cfg.moe
        tiles = {f: sum(len(b.rows[0]) for r in img.regions for b in r.blocks if b.fmt == f) for f in fmts}
        paged = paged_formats(gguf, fmts, wide=cfg.moe and max(img.R[f] for f in fmts) > 16, rot=self.paro,
                              nbytes=tiles if self.paro else None)  # (PARO: its image's rows, not the file's F16 tensors)
        if slots == 8 and not paged and len(fmts) > 2:  # (a design of 8 slots holds two formats' kernels: Qwen3.5 9B's
            # three overflow the main core's program by 0.7 KB, and one of them beside a page slot leaves the slot
            # 0.1 KB short of a page)
            paged = tuple(sorted(fmts))
        self.op = QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=img.tile_bytes,
                                      **({"rot": True} if self.paro else {}),
                                      eps=cfg.rms_norm_eps, gdn=self.gsh, attn=self.ash,
                                      moe=(cfg.n_experts, cfg.n_experts_used) if cfg.moe else None,
                                      max_k=MOE_MAX_K if cfg.moe else MAX_K,
                                      paged=paged,
                                      slots=slots, bcast=slots == 8, gu_win=gu_win, split=img.split,
                                      pairs=not cfg.moe and any(b.pairs for r in img.regions for b in r.blocks),
                                      tails=any(r.half for r in img.regions),
                                      tail_pairs=any(b.pt for r in img.regions for b in r.blocks))
        self.g8 = self.gsh is not None and slots == 8  # (the GDN core of 8 slots: ``gdn``'s "8 slots")
        self.g4 = self.gsh is not None and slots == 4 and self.gsh.chunk  # (chunkwise of 4: ``gdn``'s "Chunkwise", "4 slots")
        self.blocks = self.g4 or self.g8  # (a chunkwise core: the states at the ends of blocks of 8 positions, ``GdnCursor.half``)
        self.flags = FLAGS | ({"gdn_in": F_NORM | F_GIN} if slots == 8 else {})
        self._curs = [GdnCursor() for _ in range(sequences)]  # (the GDN layers' lazy commit and blocks, per sequence: ``cur``)
        # (chunkwise) whether the step's chain ends its block: its state panels drain (``_variant``); a step of several
        # sequences: every chain does (``_step_parts``)
        self.ends = True
        self.gdn_steps, self.attn_steps, self.moe_steps = GdnTransfers(self), AttentionSteps(self), MoeTransfers(self)
        self.conv = ConvHandoff(self)  # (the conv states between the decode and the prefill)
        if img.screen:
            self.screen = Screen(gguf, img.lm, self._f32("output_norm.weight"), cfg.rms_norm_eps,
                                 min(self.vocab, cfg.vocab_size))
        self._plan()

    @property
    def rope(self) -> np.ndarray:
        """The selected sequence's M-RoPE positions (class docstring)."""
        return self._ropes[self.seq]

    @property
    def cur(self) -> GdnCursor:
        """The selected sequence's lazy commit and blocks (``cursor.GdnCursor``)."""
        return self._curs[self.seq]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        if self.paro:
            return self.paro_groups(L, name, S)
        out = mp.groups(self.image.region(L, name), self.flags[name], S, block_flags=MOE_BLOCKS if name == "moe_in" else None,
                        f32_ab=name == "gdn_in" and self.g8, slots=self.SLOTS)
        if self.op.packed and self.SLOTS == 8:  # (the input's slots per k block, the host's per phase)
            for g in out:
                g.meta[M_SP] = self._sp(S, (L, name))
        return out

    def _sp(self, S: int, phase: tuple[int, str] | None = None) -> int:
        """(a design of 8 slots: gdn_out's input of 8, the GDN cores' outputs whole heads of 8 slots; attn_out's, one pass:
        an octet's rows)"""
        # (a PARO checkpoint's: rot_g's, rot_o's, which the cores' outputs feed)
        name = phase[1] if phase is not None else ""
        whole = name == ("rot_g" if self.paro else "gdn_out") or (
            name == ("rot_o" if self.paro else "attn_out") and self.ash is not None and self.ash.one)
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
        if self.paro:
            return self.paro_phases()
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
        if self.paro:  # (rot_down's padding groups' outputs: ``decode.rotated.sink_out``)
            self.rot_sink = take("ACT", 2 * self.SLOTS * QwenImage.down_slice(self.cfg))
        for L, name in self._phases():  # (the GDN step's outputs' x (gdn_out's, PARO's rot_g): the GDN pads' sink past
            # it; several sequences: the x that the attention's passes and the GDN chains write staged)
            sink = 2 * self.SLOTS if name == ("rot_g" if self.paro else "gdn_out") else 0
            staged = self.sequences > 1 and name in (("rot_o", "rot_g") if self.paro else ("attn_out", "gdn_out"))
            if self.rot_kind(name) is not None:
                self._add_rot_inputs((L, name), extra=sink, staged=staged)
                continue
            self._add_inputs((L, name), extra=sink, staged=staged)
        for L in self.taps:  # (a DFlash2 drafter's taps: plain copies, no phase reads them)
            self._add_rot_inputs((L, "tap"))
        self.norm_off: dict[tuple[int, str], int] = {}
        self.n_norm = -(-self.H // (NX_PER_NORM * OBJ_K))  # norm-weight A objects of a phase
        nw = 2 * NX_PER_NORM * OBJ_K * (self.n_norm - 1) + self.T  # (the last object's extent)
        for L in self.layers:
            for name in ("attn_norm", "post_attention_norm"):
                self.norm_off[(L, name)] = take("NORM", nw)
        self.norm_off[(-1, "output_norm")] = take("NORM", nw)
        self.gobj, self.gstate, self.fatt, self.rawkv, self.kv_off, self.ghdr, self.ring_off = {}, {}, {}, {}, {}, {}, {}
        self.fatt2: dict[int, int] = {}  # (``AHEAD``: per attention layer the second transaction buffer's F_ATT objects)
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
                if self.AHEAD:
                    self.fatt2[L] = take("ACT", COLS * self.ash.u * XB)
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
            if self.paro:  # (per layer its combine's metadata objects: the next layer's copies, ``paro_steps.comb_meta``)
                self.comb = {L: take("ACT", COLS * XB) for L in self.layers}
        # (the F_ATT objects' first words: [transaction buffer][layer][column][sub-step])
        self.fobj = np.array([[[[self.fatt_object(L, c, u, b) // 4 for u in range(self.ash.u)] for c in range(COLS)]
                               for L in self.fatt] for b in range(2 if self.AHEAD else 1)], np.int64)
        # (the gdn_in metadata objects' first words, whose step words the host writes: a PARO checkpoint's gin_q and gin_z)
        self.gin = np.array([self.inp[(L, n)][0].off // 4 for L in self.gobj
                             for n in (("gin_q", "gin_z") if self.paro else ("gdn_in",))], np.int64)
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

    def fatt_object(self, L: int, c: int, u: int, b: int = 0) -> int:
        """ACT offset of column c's F_ATT object of sub-step u of layer L (b: of the second transaction buffer, ``AHEAD``)."""
        return (self.fatt2 if b else self.fatt)[L] + (c * self.ash.u + u) * XB

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
            if self.paro and L >= 0:  # (the layers' norm weights are in their consumers' rotations: the prefill's norms
                # read ones)
                w = np.ones(self.H, bfloat16)
            else:
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
        for b, c, u in itertools.product(range(len(self.fobj)), range(COLS), range(self.ash.u)):
            o = self.fatt_object(L, c, u, b)
            m = np.zeros(META, np.int32)
            m[M_FLAGS] = F_ATT
            act[o:o + 4 * META] = m.view(np.uint8)
            act[o + 4 * META:o + 4 * META + norms.nbytes] = norms.view(np.uint8)

    def _load_gdn(self, L: int) -> None:
        """Layer L's GOBJ: the header, the conv objects (weights, zero histories, pads), zero K and delta objects."""
        sh, cfg = self.gsh, self.cfg
        gobj = self.dev.view("GOBJ")
        rows_qkv = gdn_in_rows(self.image, L)
        cw = np.asarray(self.gguf.tensor(f"blk.{L}.ssm_conv1d.weight").raw, np.float32)
        dt, A = self._f32(f"blk.{L}.ssm_dt.bias"), self._f32(f"blk.{L}.ssm_a")
        norm = self._f32(f"blk.{L}.ssm_norm.weight")
        m = np.zeros(META, np.int32)
        # (several sequences: the header also gives the core its chain's words, ``GdnTransfers.set_words``)
        m[M_FLAGS], m[M_SKIP] = F_GDN | (F_GIN if self.sequences > 1 else 0), sh.n_objects - 1
        if self.g8:  # (the helper's requests per step: ``gdn``'s ``hgdn``, its heads' and the step's end)
            m[M_HPAN] = sh.hc + 3
        for c in range(COLS):
            vh = gdn_v_heads(cfg, c)
            rows = rows_qkv[c][rows_qkv[c] >= 0]
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

    def load(self, *, threads: int = 0, also: Mapping[str, int] | None = None) -> None:
        """(every sequence's GDN objects and zero states)"""
        super().load(threads=threads, also=also)
        for s in range(1, self.sequences):
            self.select(s)
            self.reset()
        self.select(0)

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
        if self.paro:  # (``paro_steps``: an MoE layer's moe_in and sdown)
            return self.paro_weights(seq, phase, gi, True)
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
        if self.paro:
            return self.paro_build(S, lm, subs, b)
        dev, cfg = self.dev, self.cfg
        seq = Sequence(self.op, inputs=("X",))
        st = Step(S, subs, b=b)
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
        # (the patch tables with the build, not at the step's first run: a round's first step of a kind would take ~10 ms
        # longer, and a drafter's planner counts a round's time)
        if self.ash is not None:
            st.patch = self.attn_steps.patch_tables(st)
        if st.panels:
            st.table = PanelTable.of(st.panels)
        return st

    def _head(self, seq: Sequence, S: int) -> None:
        """Phases before the first layer's (a drafter's eh: ``draft``)."""

    def _variant(self) -> tuple:
        """(a step of a chunkwise GDN core: whether its chain ends its block, ``ends``: only then its state panels
        drain)"""
        return (self.start, self.ends)

    def _variant_ahead(self, p: int, S: int) -> tuple:
        return (self.start, not self.blocks or p % 8 + S >= 8)

    def _transaction(self, key: StepKey) -> Step:
        """(built as the key's variant: ``_build`` reads ``ends``, which is the running step's when the next one's is
        made ahead; the second buffer's: a copy of the first's whose F_ATT fills read that buffer's objects, 1 ms instead
        of a build's ~100)"""
        if key.b and key not in self._steps:
            st = self._transaction(key._replace(b=0))
            words = np.array(st.tr.words)
            if st.fatt:
                at = np.array([f[0] for f in st.fatt])
                objs = np.array([self.fatt_object(L, c, u, 1) for _, L, c, u in st.fatt])
                self._bd_words(words, at, self.dev.address("ACT") + objs, np.full(len(at), XB, np.int64))
            self._steps[key] = dataclasses.replace(st, tr=Transaction(self.dev, words), b=1)
            return self._steps[key]
        ends, self.ends = self.ends, key.variant[1]
        try:
            return super()._transaction(key)
        finally:
            self.ends = ends

    def prepare(self, S: int, logits: bool = True, start: int | None = None) -> None:
        """(chunkwise: the steps that end their block and those that do not)"""
        for ends in (True, False) if self.blocks else (True,):
            self.ends = ends
            super().prepare(S, logits, start)
        self.ends = True

    def prepare_joint(self) -> None:
        """(several sequences) Build the transactions of the steps of several parts (``_step_parts``: all the design's
        slots) and their patch tables: one per number of sequences in the step."""
        self.ends = True
        for k in range(2, self.sequences + 1):
            st = self._transaction(self._key(self.SLOTS, True, pass_subs(k)))
            self.gdn_steps.chain_table(st)
            if self.ash is not None and st.patch is None:
                st.patch = self.attn_steps.patch_tables(st)

    def patch_ahead(self, S: int, p: int) -> None:
        """(a step of a chain)"""
        if self.blocks:
            self.ends = p % 8 + S >= 8
        super().patch_ahead(S, p)
        self._round = True

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

    def _join(self, parts: tuple[Part, ...]) -> tuple[tuple[int, int, int], ...]:
        """(a pass of each attention and a chain of each GDN layer per part: one transaction for every way to deal
        the step's slots, ``decode.attn256_steps.pass_subs``; a step of one part: that sequence's)"""
        return self.subs(parts[0].position, parts[0].n) if len(parts) == 1 else pass_subs(len(parts))

    def _joint_subs(self, S: int) -> set[tuple]:
        """(built at their first use: a transaction per number of sequences and set of chains that end their block)"""
        return set()

    def _set_position(self, st: Step, p: int, ahead: bool = False) -> bool:
        """A step's words that follow its position: a chunkwise core's state transfers (``GdnTransfers.patch_panels``),
        the attention's F_ATT objects and cache, ring and record transfers (``AttentionSteps.set_position``). ``ahead``
        (``AHEAD``: while the step before runs, st of the other transaction buffer): not when a prefill's caches still
        owe the rings their rows (KV, which the running step reads); its positions' ``rope`` rows are taken as text's that
        follows the row before them (what a model writes before a generation's step; ``rope`` itself may hold a prompt's
        rows already and is left alone): the step is patched again when ``rope`` holds others then (``_ahead_state``)."""
        rope = self.rope[p:p + st.S]
        if ahead:
            if self.vring.from_prefill & set(self.fatt):
                return False
            rope = np.repeat(int(self.rope[p - 1].max()) + 1 + np.arange(st.S, dtype=self.rope.dtype)[:, None], 3, 1)
        if self.sequences > 1:
            return self._set_parts(st, (Part(self.seq, p, st.S),))
        if self.blocks:
            self.gdn_steps.patch_panels(st, p)
        if self.ash is not None:
            self.attn_steps.set_position(st, p, np.concatenate(attn.rope_cs(self.cfg, rope), 1))
        self._rope_used = (p, p + st.S, rope.tobytes())
        return True

    def _set_parts(self, st: Step, parts: tuple[Part, ...], ahead: bool = False) -> bool:
        """``_set_position`` of a step of several sequences' slots (or of one sequence of an overlay of several): each
        GDN chain's transfers in its sequence's buffers and its slots (``GdnTransfers.patch_chains``), each
        attention's passes (``AttentionSteps.set_parts``) with their sequences' ``rope`` rows."""
        assert not ahead and self.sequences > 1, parts
        if self.gobj:
            self.gdn_steps.patch_chains(st, parts)
        if self.ash is not None:
            rope = np.concatenate([self._ropes[q.seq][q.position:q.position + q.n] for q in parts])
            self.attn_steps.set_parts(st, parts, np.concatenate(attn.rope_cs(self.cfg, rope), 1))
        self._branch = None  # (the F_ATT objects' step words are the passes': ``_set_branch`` writes its own again)
        return True

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
        at = self.fobj[0][:, :, 0, None] + META + 2 * attn.HD + 64 * (8 - L) + np.arange(64 * L)
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
            fo = self.fobj[0][:, :, 0].ravel()
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

    def step(self, x: np.ndarray, position: int | tuple[Part, ...], *, logits: bool = True, commit: bool = True,
             branch: tuple[int, int] | None = None) -> np.ndarray | None:
        """One step of S = len(x) <= SLOTS (4; 8 where ``slots_of``) token slots: x (S, hidden) the first layer's inputs (embeddings) at positions
        position .. position + S - 1 (the positions before are in the caches, the GDN states hold the committed ones and
        the previous step's accepted tokens: ``accept``). ``commit``: the GDN states after the step's tokens are written
        (all of them must then be accepted, ``accept(S)``), else the step's tokens are replayed by the next step as far
        as accepted (a chunkwise core of 4: the states are written at the ends of blocks of 8 positions whatever
        ``commit``, which then only says that all S are accepted: ``gdn``'s "Chunkwise"). ``branch`` (parent, L) (``can_branch``, a step of 8): its last L slots follow slot parent instead of slot 7 - L,
        at position + parent + 1 .. inside the 8 positions of slot parent's (position % 8 + parent + L <= 7: its attention
        records and GDN block; a tree of two branches: ``_set_branch``); None: a chain. Several sequences' slots:
        ``position`` their parts (``decode.step.Part``: x's rows in the parts' order, each sequence's at its own next
        positions; no state is committed: each sequence's accepted tokens through ``accept(n, seq)``). Returns the
        slots' logits (S, vocab) bf16 (a view of the NPU's buffer, valid until the next step), or without ``logits``
        None."""
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= self.SLOTS
        if isinstance(position, tuple):
            return self._step_parts(x, position, logits)
        assert branch is None or self.sequences == 1, "a branching step: a model of one sequence"
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
            self.ends = position % 8 + S - (branch[1] if branch else 0) >= 8
            cur.gpos = position
            self.gdn_steps.set_words(nrep=position % 8, cshift=cur.cshift, commit=False)
            if self.sequences > 1:  # (the chain's words: in its sequence's headers)
                self.gdn_steps.set_chain_words((Part(self.seq, position, S),))
        else:
            self.gdn_steps.set_words(nrep=cur.nrep, cshift=cur.cshift, commit=commit)
        cur.committed = commit
        self._kept_rows = None
        self._write_loadr(x)
        if not self.paro:  # (PARO: rot_in's copies of the residual)
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

    def _step_parts(self, x: np.ndarray, parts: tuple[Part, ...], logits: bool) -> np.ndarray | None:
        """``step`` of several sequences' slots (chunkwise cores: the states are written at the ends of the chains'
        blocks; each sequence's accepted tokens: ``accept``). A step of several parts has one transaction per number
        of parts (``prepare_joint`` builds them all: one takes 0.4 s per part on the 27B): it runs all the design's
        slots, those nobody asked for as one part's further slots (its last row again: drafts that nobody accepts; a
        step of 8 slots takes 2% more than one of a slot), and each of its chains ends its block, the core taking it
        there (``GdnTransfers.set_chain_words``): its state's update into the half the next block starts from, which
        that block's real end writes again (0.7 ms a chain of the 27B's step of 325 ms)."""
        S = x.shape[0]
        assert self.sequences > 1 and sum(q.n for q in parts) == S, parts
        keep = np.arange(S)  # (the caller's rows in the step's)
        pad = self.SLOTS - S if len(parts) > 1 else 0
        fit = [i for i, q in enumerate(parts) if q.position + q.n + pad <= self.max_positions] if pad else []
        if fit:
            i = fit[-1]  # (the last part: the caller's rows stay the step's first)
            q, e = parts[i], sum(r.n for r in parts[:i + 1])
            parts = (*parts[:i], Part(q.seq, q.position, q.n + pad), *parts[i + 1:])
            x = np.concatenate([x[:e], np.repeat(x[e - 1:e], pad, 0), x[e:]])
            keep = np.concatenate([np.arange(e), np.arange(e + pad, S + pad)])
            S += pad
        curs = [self._curs[q.seq] for q in parts]
        assert all(c.committed or c.accepted for c in curs), "accept() the previous lazy step's tokens first"
        before = self.seq
        for q, cur in zip(parts, curs, strict=True):  # (after a prefill: the sequence's conv states into its objects)
            if cur.conv_in_ghdr:
                self.select(q.seq)
                self.conv.for_decode(q.position)
        self.select(before)
        if self._metas_s != (S, self.start):
            self._metas(S)
        for q, cur in zip(parts, curs, strict=True):
            assert q.position == cur.gnext or not self.gobj, (q, cur.gnext)
            cur.gpos, cur.committed = q.position, False
        self.ends = len(parts) > 1 or parts[0].position % 8 + parts[0].n >= 8
        self._kept_rows = keep
        self.gdn_steps.set_chain_words(parts)
        self._write_loadr(x)
        if not self.paro:  # (PARO: rot_in's copies of the residual)
            first = self._input(self.layers[0])
            self._write_input(self.inp[first][0], interleave(x, self._sp(S, first)).view(np.uint8).reshape(-1))
        out = self._run(S, parts, logits)
        for cur in curs:
            cur.accepted = False
        if out is None:
            return None
        sel = slice(0, len(keep)) if keep[-1] == len(keep) - 1 else keep  # (the caller's rows: a view, else a copy)
        return out.rows(sel) if isinstance(out, Screened) else out[sel]

    def _run(self, S: int, position: int | tuple[Part, ...], logits: bool) -> np.ndarray | None:
        """(a step that ``patch_ahead`` patched patches none ahead itself: ``STEPS_AHEAD``)"""
        ahead, self.AHEAD = self.AHEAD, self.AHEAD and not self._round
        self._round = False
        try:
            return super()._run(S, position, logits)
        finally:
            self.AHEAD = ahead

    def _ran(self, p: int, S: int, seq: int | None = None) -> None:
        vring = self.vring if seq is None else self._vrings[seq]
        for L in self.fatt:
            vring.ran(L, p, S)

    def _ring_state(self, seq: int | None = None) -> tuple:
        """(and whether a prefill's conv states wait in GHDR; the GSTATE halves of the blocks, ``GdnCursor.half``; the
        ``rope`` rows the last position patch took)"""
        a, b, _ = self._rope_used
        s = self.seq if seq is None else seq
        cur = self._curs[s]
        return (*super()._ring_state(seq), cur.conv_in_ghdr, cur.gh, cur.gp, self._ropes[s][a:b].tobytes())

    def _ahead_state(self) -> tuple:
        """(the ``rope`` rows the patch took in place of those ``rope`` holds now)"""
        return (*self._ring_state()[:-1], self._rope_used[2])

    def _lm_phase(self, seq: Sequence) -> None:
        if self.paro:  # (its F_PSQ input: ``paro_logits_input``)
            self.paro_phase(seq, Step(1, ()), (-1, "lm"))
            return
        if self.image.column_major:
            self._weight_run(seq, [(-1, "lm")])
        self._phase(seq, (-1, "lm"), 1, "LOGITS")

    def logits_of(self, x: np.ndarray) -> np.ndarray:
        """(PARO: the LM head's input as its F_PSQ copy, ``paro_logits_input``)"""
        if not self.paro:
            return super().logits_of(x)
        if self._metas_s != (1, self.start):
            self._metas(1)
        if self._tr_lm is None:
            seq = Sequence(self.op, inputs=("X",))
            self._lm_phase(seq)
            seq.sync(fills=True)
            self._tr_lm = Transaction(self.dev, seq.words())
        self._clear_pads(self._sp(1))
        self.paro_logits_input(x)
        self.ready()
        self.dev.dispatch(self._tr_lm)
        return self._logits(1)[0]

    def accept(self, n: int, seq: int | None = None) -> None:
        """The previous step's first n tokens (of sequence ``seq``; None: the selected one) are accepted (the next
        step's positions start after them): its GDN layers replay them (unless it committed: then n must be all of
        them) and shift their conv histories (a chunkwise core: the next step recomputes its block's lanes from the
        histories, ``GdnCursor.advance``)."""
        cur = self.cur if seq is None else self._curs[seq]
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

    def drafter_view(self) -> object:
        """(a DFlash2 drafter's head: the MTP block's view of the LM rows, ``QwenImage.mtp_view``; ``DFlashTarget``)"""
        return self.image.mtp_view(self.cfg) if self.image.draft_cols else None

    def hiddens(self, n: int) -> np.ndarray:
        """``hidden(-1, .)`` of the first n rows the last step's caller gave it (n, hidden) fp32: copies (a step of
        several sequences' slots: the rows in the caller's order, whatever slots ran them)."""
        off, sp = self.tap_x(-1), self._sp(self._last_s, self._input(-1))
        self.dev.fetch("ACT", off, 2 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)
        rows = np.arange(n) if self._kept_rows is None else self._kept_rows[:n]
        h = np.ascontiguousarray(v[:, rows].transpose(1, 0, 2)).reshape(n, self.H).view(bfloat16).astype(np.float32)
        return h / self._lm_w() if self.paro else h

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L (-1: the LM head) read in the last step, of a slot (bf16 values, fp32; a PARO
        checkpoint's, of the LM head only: its copy over the output norm's bf16 weights, ``paro_steps``)."""
        assert not self.paro or L == -1, "a PARO checkpoint's layers read rotated copies"
        off, sp = self.tap_x(L), self._sp(self._last_s, self._input(L))
        self.dev.fetch("ACT", off, 2 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)[:, slot].reshape(-1)
        h = v.view(bfloat16).astype(np.float32)
        return h / self._lm_w() if self.paro else h

    def _lm_x(self, S: int) -> np.ndarray:
        """(PARO: the LM head's copy over the output norm's weights, ``hidden``)"""
        x = super()._lm_x(S)
        return x / self._lm_w() if self.paro else x

    def _lm_w(self) -> np.ndarray:
        """(PARO) The output norm's weights as the LM head's copy carries them (bf16; zeros as ones: a dead channel)."""
        w = self._f32("output_norm.weight").astype(bfloat16).astype(np.float32)
        return np.where(w != 0, w, 1)

