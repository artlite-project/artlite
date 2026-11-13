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
GDN layers' lazy-commit words (``gdn``).

The prefill (``qwen35/prefill.py``) runs on these buffers: the GDN states (GSTATE: per column its panels), the KV
caches, the norm weights (NORM: contiguous bf16; the decode reads them as overlapping A objects), and the conv states
in GHDR (its layout: per GDN layer and column a slot of ``hslot`` bytes, the 3 rows x CH bf16 at ``CS_OFF``): the host
copies them there from the conv objects' histories before a prompt chunk (``conv_for_prefill``) and back as their
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
import time

import numpy as np
from ml_dtypes import bfloat16

from artlite import txn
from artlite.decode import gemv
from artlite.decode import phases as mp
from artlite.decode.model import ModelOverlay, host_bytes, paged_formats
from artlite.decode.protocol import (
    AB_ID,
    B_F32,
    B_GU,
    B_KEEP,
    B_PAIR,
    C_LEN,
    F_COMB,
    F_EMITR,
    F_GU,
    F_MAIN,
    F_MSG,
    F_NORM,
    F_RESID,
    F_ROUTE,
    F_WS,
    M_FLAGS,
    M_NASKIP,
    M_NREL,
    M_NS,
    M_NX,
    M_RBASE,
    M_RSTR,
    M_RTDN,
    M_RTGU,
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
from artlite.operators import paging
from artlite.operators.bfp16 import block_values
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import attn, gdn
from .decode import QwenDecodeOperator
from .image import NORMS, QwenImage, gdn_v_heads
from .words import (
    F_ATT,
    F_GDN,
    F_GIN,
    M_ALANE,
    M_AMASK,
    M_ANA,
    M_BRANCH,
    M_COMMIT,
    M_CSHIFT,
    M_G0,
    M_GATT,
    M_GJ,
    M_HPAN,
    M_KSHIFT,
    M_NREP,
    M_NTAIL,
    M_POS,
    M_SA,
    M_SPLIT,
)

FLAGS = {"eh": F_RESID | F_EMITR,  # (a drafter's first phase: ``draft``)
         "gdn_in": F_NORM | F_MSG, "attn_in": F_NORM, "gdn_out": F_RESID | F_EMITR, "attn_out": F_RESID | F_EMITR,
         "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM, "moe_in": F_NORM, "sdown": F_RESID | F_WS}
MOE_BLOCKS = {"sgu": B_GU, "router": B_PAIR | B_F32, "sgate": B_PAIR | B_KEEP}  # (moe_in's blocks: ``decode.design``)
MOE_MAX_K = 2048  # (an MoE model's K groups at most: ``image.payload``)
CP_BYTES = 128 * MR.RBUF_OBJECTS  # (a column's routing objects in CP; the expert object starts 256 B before their end)
KV_PAD = 4096 - 256  # positions past the caches (the last layer's): the prefill's sub-chunk slices
CS_OFF = 4 * META + 640  # GHDR: the conv state's bytes in a column's slot
HSLOT = 12288  # GHDR: a column's slot (the prefill's conv-state drain writes 8 KB from CS_OFF)
DIMS_OBJ = (XB // 8, 2, XB // 8, XPAY // 4)  # X objects of XB bytes, XPAY apart (the overlapping regions)
FILLS = ("all", "new", "run")  # a step's cache fills: the whole cache, the sub-step's group, a split column's run
F_WORDS = {"pos": M_POS, "ns": M_NS, "sa": M_SA, "skip": M_SKIP, "split": M_SPLIT, "nrel": M_NREL, "ntail": M_NTAIL,
           "naskip": M_NASKIP, "g0": M_G0, "gatt": M_GATT, "gj": M_GJ, "ana": M_ANA}  # (an F_ATT object's words, by name)
F_ATT_AT = np.array(list(F_WORDS.values()))


class _Step:
    """The transaction of a step of S slots (its attentions' sub-steps ``subs``: first slot, slots, whether the
    chains divide their caches' groups) and its words that follow the position: the KV cache and ring reads (A: their
    address and length) and the record drains (C: their address)."""

    def __init__(self, S: int, subs: tuple[tuple[int, int, int], ...]):
        self.S, self.subs = S, subs
        self.tr: Transaction | None = None
        # (BD word index, layer, column, sub-step, which groups: "all", the column's "run" of a split, "new": the
        # sub-step's group)
        self.fills: list[tuple[int, int, int, int, str]] = []
        self.tails: list[tuple[int, int, int, int, int]] = []  # (BD word index, layer, column, sub-step, tail object)
        # (BD handle, layer, column, sub-step, i-th record drain: a chain's first column's K records 0, 1; its last
        # column's V panels and rows of record 0, then of record 1)
        self.drains: list[tuple[list[int], int, int, int, int]] = []
        self.patch: dict[str, np.ndarray] | None = None  # (``_patch_tables``)
        # (chunkwise: its state panels' fills and drains (BD word index or drain handle, GSTATE offset in a half, bytes,
        # whether a drain), the half they are patched for: ``_gstate_at``)
        self.gst: list[tuple[int | list[int], int, int, bool]] | tuple[np.ndarray, ...] = []
        self.ghalf = -1


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
        self.blocks = self.g4 or self.g8  # (a chunkwise core: the states at the ends of blocks of 8 positions, ``_half``)
        self.flags = FLAGS | ({"gdn_in": F_NORM | F_GIN} if slots == 8 else {})
        self._a_bds = getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)  # (the host's BDs on A)
        self._conv_in_ghdr = False
        self._cvi: tuple[np.ndarray, np.ndarray] | None = None
        self._committed, self._accepted = True, True
        # (chunkwise) the GDN state's half: the block of position _gp starts in half _gh (``_half``); the step's
        # position, the next one; a prefill's half; the k / v histories' and gates' first row (8 after a block's end)
        self._gp = self._gh = self._gpos = self._gnext = self._gcur = self.kshift = 0
        self._ends = True  # (8 slots) whether the step's chain ends its block: its state panels drain (``_variant``)
        self._order = None if img.order is None else self._id_runs(img.order)  # (``_logits``)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _flags(self, L: int, name: str) -> int:
        return self.flags[name]

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
        if not self.moe:
            return out
        img = self.image
        for L in self.layers:
            gu, dn = img.region(L, "exp_gu"), img.region(L, "exp_down")
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS] = F_MAIN | F_ROUTE, S
            a = self.dev.address("W", gu.offset)
            m[M_RBASE:M_RBASE + 2] = np.array([a & 0xFFFFFFFF, a >> 32], np.uint32).view(np.int32)
            m[M_RSTR], m[M_RTGU], m[M_RTDN] = gu.col_stride, gu.n_rt * gu.n_kc, dn.n_rt * dn.n_kc
            assert gu.col_stride % 4 == 0 and 2 * self.cfg.n_experts * gu.col_stride < 1 << 32
            out.append((self.xch[L], m.view(np.uint8)))
            tpl = np.concatenate([mp.groups(gu, F_GU, S)[0].meta, mp.groups(dn, 0, S)[0].meta]).view(np.uint8)
            for c in range(COLS):  # (the expert object's payload: the gate / up and down templates)
                out.append((self.cp[L] + c * (CP_BYTES + XB) + CP_BYTES, tpl))
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_NS], m[M_NX] = F_MAIN | F_COMB, S, 2 * S
        out += [(self.part + c * self.part_col, m.view(np.uint8)) for c in range(COLS)]
        return out

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
            assert CS_OFF + 2 * 3 * self.gsh.ch <= HSLOT, (self.gsh.ch, CS_OFF, HSLOT)  # (the conv state: the prefill's drain)
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
        if self.blocks:  # (the states twice: a block's base, and the state at its end, ``_half``)
            self.ghalf = take("GSTATE", self._layout["GSTATE"] - 256) - 256
        self.xch, self.cp = {}, {}
        if self.moe:  # (per MoE layer its route object, per column its routing's objects and expert object; PART)
            for L in self.layers:
                self.xch[L] = take("ACT", XB)
                self.cp[L] = take("ACT", COLS * (CP_BYTES + XB))
            self.part = take("ACT", COLS * self.part_col)
        self._fobj = np.array([[[(self.fatt[L] + (c * self.ash.u + u) * XB) // 4 for u in range(self.ash.u)]
                                for c in range(COLS)] for L in self.fatt], np.int64)  # ([layer][column][sub-step], words)
        self._gin = np.array([self.inp[(L, "gdn_in")][0].off // 4 for L in self.gobj], np.int64)
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

    def _fatt(self, L: int, c: int, u: int) -> int:
        """ACT offset of column c's F_ATT object of sub-step u of layer L."""
        return self.fatt[L] + (c * self.ash.u + u) * XB

    def _q(self, L: int, c: int, s: int) -> int:
        """ACT offset of column c's q object of slot s of layer L."""
        return self.fatt[L] + (COLS * self.ash.u + c * self.SLOTS + s) * XB

    def _gpay(self, L: int, c: int, k: int) -> int:
        """GOBJ offset of layer L's column c's object k's payload (k: conv q / k 0 .. 3, K 4, pair p 5 + 3 p ..; 8 slots:
        its payload slot, ``gdn``'s "8 slots")."""
        return self.gobj[L] + c * self.gcol + 256 + XPAY * k

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The norm weights (NORM), the attention layers' F_ATT objects; (MoE) the control route's registers free."""
        if self.moe:
            MR.check_route(paging.cdo_values(os.path.dirname(self.op._program.insts_path)), range(COLS))
        norm = self.dev.view("NORM")
        for (L, name), off in self.norm_off.items():
            w = self._f32("output_norm.weight" if L < 0 else f"blk.{L}.{name}.weight").astype(bfloat16)
            norm[off:off + 2 * self.H] = w.view(np.uint8)
        self.dev.flush("NORM")
        for L in self.fatt:
            self._load_attention(L)

    def _f32(self, name: str) -> np.ndarray:
        return np.asarray(self.gguf.tensor(name).raw, np.float32).reshape(-1).copy()

    def _load_attention(self, L: int) -> None:
        act = self.dev.view("ACT")
        qn = self._f32(f"blk.{L}.attn_q_norm.weight") * np.float32(math.log2(math.e) / 16)
        norms = np.concatenate([qn, self._f32(f"blk.{L}.attn_k_norm.weight")])
        for c in range(COLS):
            for u in range(self.ash.u):
                o = self._fatt(L, c, u)
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
                h = self._gpay(L, c, gdn.H_SLOT) + (gdn.H_PRM if self.g8 else gdn.P_PRM)
                gobj[h:h + 640] = gdn.head_payload(dt[vh], A[vh], norm)
                for b, w in enumerate(gdn.conv_weights(cw[rows])):
                    a = self._gpay(L, c, 0) + sh.weights_at(b)
                    gobj[a:a + w.size] = w
                continue
            for b, pay in enumerate(gdn.conv_payloads(sh, cw[rows], dt[vh], A[vh], norm)):
                k = b if b < 4 else 5 + 3 * ((b - 4) // 2) + (b - 4) % 2
                gobj[self._gpay(L, c, k):self._gpay(L, c, k) + XPAY] = pay

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
        self.nrep, self.cshift = 0, 0
        self._gp = self._gh = self._gpos = self._gnext = self._gcur = self.kshift = 0
        self._a_bds = getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)  # (the host's BDs on A)
        self._conv_in_ghdr = False
        self.rope[:] = _text_rope(self.max_positions)

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

    def _x_at(self, phase: tuple[int, str], e: int, S: int) -> tuple[int, int]:
        """(ACT offset of input element e's k block in a phase's x of a step of S slots, the bytes left in its K
        group's x)."""
        sp = self._sp(S, phase)
        for rg in self.inp[phase]:
            if rg.k0 <= e < rg.k0 + rg.K:
                return rg.off + XB + 2 * sp * (e - rg.k0), 2 * sp * (rg.k0 + rg.K - e)
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

    def _gdn(self, seq: Sequence, L: int, S: int) -> None:
        """gdn_in and the GDN step."""
        if self.g8:
            self._gdn8(seq, L, S)
            return
        if self.g4:
            self._gdn4(seq, L, S)
            return
        dev, MG, sh = self.dev, gdn, self.gsh
        self._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "gdn_in")][0], S, (L, "gdn_in"))
        for c in range(COLS):
            seq.fill("X", c, dev.address("GOBJ", self.gobj[L] + c * self.gcol), 5 * XB, DIMS_OBJ)
        for c in range(COLS):
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, 0) + 2048), 2 * MG.D_BYTES, (512, 2, 1024, 2048))
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, 4)), MG.D_BYTES)
        for p in range(self.npair):
            for c in range(COLS):
                seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, 5 + 3 * p) - 256), 3 * XB, DIMS_OBJ)
                seq.fill("X", c, dev.address("GSTATE", self.gstate[L] + c * self.ps + 32 * p * MG.PANEL - 256), 32 * XB,
                         DIMS_OBJ)
            for c in range(COLS):
                h0 = gdn_v_heads(self.cfg, c)[2 * p]
                seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, 5 + 3 * p) + 2048), MG.D_BYTES, (512, 2, 1024, 1024))
                seq.drain("D", c, dev.address("GSTATE", self.gstate[L] + c * self.ps + 32 * p * MG.PANEL), 32 * MG.PANEL)
                seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, 7 + 3 * p)), MG.D_BYTES)
                # the pair's outputs (2 KB) at its heads in gdn_out's x, its padding (2 KB) in the sink past that K
                # group's x (its x bytes on)
                at, left = self._x_at((L, "gdn_out"), gdn.D * h0, S)
                xg = next(8 * rg.K for rg in self.inp[(L, "gdn_out")] if rg.k0 <= gdn.D * h0 < rg.k0 + rg.K)
                assert left >= 2048
                seq.drain("D", c, dev.address("ACT", at), MG.D_BYTES, (512, 2, xg // 4, 1024))
        assert sh.ch == (4 + sh.hc) * gdn.D

    def _gdn8(self, seq: Sequence, L: int, S: int) -> None:
        """gdn_in and the GDN step of an 8-slot design (``gdn``'s "8 slots"): gdn_in's rows per slot into the q blocks' second
        objects, the k heads' K objects and the v heads' R objects (v, then z), ab into the header (a sync), then per k head
        its group and per v head its group and panels (``_gdn_units``)."""
        dev, sh, hc = self.dev, self.gsh, self.gsh.hc
        X = XPAY // 4  # (words)
        self._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "gdn_in")][0], S, (L, "gdn_in"))
        # (a block's objects (half, slot) as [half][slot 8][64]: half 0's S rows, then half 1's 1 KB on; per execution its
        # pair of blocks of the two k heads, or the hc v blocks; then the next: k after q, z after v)
        hb = 2 * 64 * 8 // 4  # (words between the halves)
        for c in range(COLS):
            pay = lambda k, c=c: dev.address("GOBJ", self._gpay(L, c, k))
            q0 = pay(sh.kh_slot(0) + 1) + 3 * 2 * gdn.D  # (q0's rows, past the history's rows 8 .. 10)
            seq.drain("C", c, q0, 2 * 2 * S * 2 * C_LEN, (S * C_LEN // 2, 2, hb, gdn.KH_SPAN * X), 2,
                      ((pay(sh.kh_slot(0) + 4) - q0) // 4, 2))  # q, then k rows (the K objects)
            v0 = pay(sh.v_slot(0) + 2)
            seq.drain("C", c, v0, hc * 2 * S * 2 * C_LEN, (S * C_LEN // 2, 2, hb, gdn.V_SPAN * X), 2,
                      (2048 // 4, 2))  # v rows, then z (the R objects' halves)
            seq.drain("C", c, pay(gdn.H_SLOT) + gdn.H_AB, 2 * S * C_LEN)  # ab
        seq.sync()
        vh = [gdn_v_heads(self.cfg, c) for c in range(COLS)]
        last = hc - 1  # (the last v head the core runs: k head 1's last)

        def kh_x(kh: int, c: int) -> None:  # (k head 0's group after the header)
            g0 = sh.kh_slot(kh) - (1 if kh == 0 else 0)
            seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, g0) - 256), (gdn.KH_SPAN + 1 - kh) * XB, DIMS_OBJ)

        def kh_d(kh: int, c: int) -> None:  # (the q history from the q A object's second half; the k history from the k A
            # object's, over the k B object, its last 2 KB into the K object's taken rows)
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, sh.kh_slot(kh)) + 2048), gdn.D_BYTES)
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, sh.kh_slot(kh) + 2) + 2048), 2 * gdn.D_BYTES)

        def v_xa(h: int, c: int) -> None:
            seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, sh.v_slot(h)) - 256), gdn.V_SPAN * XB, DIMS_OBJ)

        def v_xp(h: int, c: int) -> None:  # (from the block's half; the step's position patches them: ``_gstate_at``)
            o = self.gstate[L] + c * self.ps + 16 * h * gdn.PANEL - 256
            self._gst.append((seq.fill("X", c, dev.address("GSTATE", o), 16 * XB, DIMS_OBJ), o, 16 * XB, False))

        def v_dp(h: int, c: int) -> None:  # (into the other half: a step that ends its block)
            o = self.gstate[L] + c * self.ps + 16 * h * gdn.PANEL
            self._gst.append((seq.drain("D", c, dev.address("GSTATE", o), 16 * gdn.PANEL), o, 16 * gdn.PANEL, True))

        def v_dh(h: int, c: int) -> None:  # (its history from the A object's second half, over the B object, its last 2 KB into
            # the R object's taken v rows)
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, sh.v_slot(h)) + 2048), 2 * gdn.D_BYTES)

        def v_do(h: int, c: int) -> None:  # (its gated outputs (2 KB) at its head in gdn_out's x, the block's gates after them
            # (2 KB): of the last head into the header, else into the sink past that K group's x)
            at, left = self._x_at((L, "gdn_out"), gdn.D * vh[c][h], S)
            xg = next(2 * self.SLOTS * rg.K for rg in self.inp[(L, "gdn_out")] if rg.k0 <= gdn.D * vh[c][h] < rg.k0 + rg.K)
            assert left >= 2048
            if h == last:
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES // 2)
                seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, gdn.H_SLOT) + gdn.H_DN), gdn.D_BYTES // 2)
            else:
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES, (512, 2, xg // 4, 512))

        self._gdn_units(hc, ((kh_x,), (kh_d,)), ((v_xa, v_xp), (v_dp,) * self._ends + (v_dh, v_do)))

    @staticmethod
    def _gdn_units(hc: int, k: tuple[tuple, tuple], v: tuple[tuple, tuple]) -> None:
        """A chunkwise core's transfers in its order: per k head its unit (fills k[0], drains k[1]: task(head, column)), then
        its v heads' (v[0], v[1]); each unit's fills ahead of the drains of the one before (a channel's fifth task waits for
        its oldest: the core then has its next objects queued while the firmware waits for drains; at most 4 tasks a unit on
        a channel), in rounds of one task over the columns (a column's tasks in a row would leave the last column's first one
        ~10 us behind the first column's, at every unit): a head's panels drain before the next head's panels fill, whose wait
        ends only when this head's panels are in."""
        units = [u for kh in range(2) for u in [(*k, kh)] + [(*v, h) for h in range(kh, hc, 2)]]
        for i in range(len(units) + 1):
            fills = [(f, units[i][2]) for f in units[i][0]] if i < len(units) else []
            drains = [(d, units[i - 1][2]) for d in units[i - 1][1]] if i else []
            for pair in itertools.zip_longest(fills, drains):
                for task, unit in filter(None, pair):
                    for c in range(COLS):
                        task(unit, c)

    def _gdn4(self, seq: Sequence, L: int, S: int) -> None:
        """gdn_in and the GDN step of a chunkwise core of 4 slots (``gdn``'s "Chunkwise", "4 slots"): its rows on M, per k head its
        group and per v head its group and panels."""
        dev, sh = self.dev, self.gsh
        X = XPAY // 4  # (words)
        self._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "gdn_in")][0], S, (L, "gdn_in"))
        vh = [gdn_v_heads(self.cfg, c) for c in range(COLS)]

        def kh_x(kh: int, c: int) -> None:
            seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, sh.kh_slot(kh)) - 256), gdn.KH_SPAN4 * XB, DIMS_OBJ)

        def kh_d(kh: int, c: int) -> None:  # (the P and K objects, a W object apart)
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, sh.kh_slot(kh))), 2 * gdn.D_BYTES,
                      (gdn.D_BYTES // 8, 2, gdn.D_BYTES // 8, 2 * X))

        def v_xa(h: int, c: int) -> None:
            seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, sh.v_slot(h)) - 256), gdn.V_SPAN4 * XB, DIMS_OBJ)

        def v_xp(h: int, c: int) -> None:  # (from the block's half; the step's position patches them: ``_gstate_at``)
            o = self.gstate[L] + c * self.ps + 16 * h * gdn.PANEL - 256
            self._gst.append((seq.fill("X", c, dev.address("GSTATE", o), 16 * XB, DIMS_OBJ), o, 16 * XB, False))

        def v_dp(h: int, c: int) -> None:  # (into the other half)
            o = self.gstate[L] + c * self.ps + 16 * h * gdn.PANEL
            self._gst.append((seq.drain("D", c, dev.address("GSTATE", o), 16 * gdn.PANEL), o, 16 * gdn.PANEL, True))

        def v_do(h: int, c: int) -> None:  # (its gated outputs (1 KB) at its head in gdn_out's x, 1 KB of padding into the sink past
            # that K group's x)
            at, left = self._x_at((L, "gdn_out"), gdn.D * vh[c][h], S)
            xg = next(2 * self.SLOTS * rg.K for rg in self.inp[(L, "gdn_out")] if rg.k0 <= gdn.D * vh[c][h] < rg.k0 + rg.K)
            assert left >= 2 * self.SLOTS * gdn.D
            seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES // 2, (gdn.D_BYTES // 16, 2, xg // 4, 1))

        def v_dh(h: int, c: int) -> None:  # (its history, then its deltas: the A object's second half and the U object after it)
            seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, sh.v_slot(h)) + 2048), 3 * gdn.D_BYTES // 2)

        self._gdn_units(sh.hc, ((kh_x,), (kh_d,)), ((v_xa, v_xp), (v_dp, v_do, v_dh)))

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

    def _attention(self, seq: Sequence, st: _Step, L: int, S: int, run: list[tuple[int, str]] | None = None) -> None:
        """attn_in and the attention, per sub-step (each after the last one's records are in DDR: it reads them);
        ``run``: the weights after it (a column-major image), after the last sub-step's cache reads."""
        dev, ash, T = self.dev, self.ash, self.T
        qc, nph = ash.qc, ash.nph
        per = ash.kvh * attn.HD // COLS  # (K and V rows of a column: its part of KV head per c // HD)
        self._weights(seq, (L, "attn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "attn_in")][0], S, (L, "attn_in"))
        for c in range(COLS):  # [q + gate | K part | V part] per slot -> its q object, the slot's raw K, V object
            h, e0 = divmod(per * c, attn.HD)
            q0 = self._q(L, c, 0) + 4 * META
            seq.drain("C", c, dev.address("ACT", q0), S * qc * 2 * 2 * attn.HD, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            raw = self.rawkv[L] + 4 * META
            for part in (0, 1):
                seq.drain("C", c, dev.address("ACT", raw + 2 * (2 * attn.HD * h + attn.HD * part + e0)), S * 2 * per,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
        def cache(u: int) -> None:  # the cache of the column's KV head (address, length: _set_position), its ring (one
            # pass: the chain's first column's stream, to both cores; a round: the first chain's streams, the round's KV
            # head, cc its column in that head's chain)
            _, n, split = st.subs[u]
            rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
            for c in range(0, COLS, nph if ash.one else 1):
                if rnd and c >= nph:  # (its objects: the column 4 away's)
                    continue
                cc = kh * nph + c if rnd else c
                head = dev.address("KV", self.kv_off[L] + cc // nph * ash.head_bytes)
                pk = {"packet": AB_ID + ash.a_bcast.index((c, c + nph))} if rnd else {}
                kinds = ("all",) if not split else ("run", "new") if c % nph == 0 else ("run",)
                for kind in kinds:  # (split: the chain's first column its run, then the new positions' group's K and V objects)
                    st.fills.append((seq.fill("A", c, head, attn.OBJS * T, **pk), L, cc, u, kind))
                if not split or ash.role(c) == attn.ROLE_V:  # (a column that attends to the newest group: the tails)
                    st.tails += [(seq.fill("A", c, dev.address("KV", self.ring(L, cc // nph)), T, **pk), L, cc, u, t)
                                 for t in range(ash.records(n))]
                if not ash.one and not split and c % nph == 0:  # (then the new positions' group again, for their records)
                    st.fills.append((seq.fill("A", c, head, attn.OBJS * T), L, c, u, "new"))

        def inputs(u: int) -> None:  # the F_ATT object, the q objects, the slots' raw K, V objects
            a, n, split = st.subs[u]
            rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self._fatt(L, c, u)), XB)
                if split:  # (the q objects of the chain's columns in order; a round: of its half of the KV head's)
                    q0, nc = (kh * nph + 2 * (c // nph), 2) if rnd else (c - c % nph, nph)
                    seq.fill("X", c, dev.address("ACT", self._q(L, q0, a)), nc * n * XB,
                             (XB // 8, 2 * n, XB // 8, self.SLOTS * XB // 4))
                else:
                    seq.fill("X", c, dev.address("ACT", self._q(L, c, a)), n * XB)
                seq.fill("X", c, dev.address("ACT", self.rawkv[L] + a * XB), n * XB)

        def outputs(u: int) -> None:  # o (the column's heads in attn_out's x at the sub-step's slots; split: the chain's
            # last column's of the chain's, a round's of its half), then the records (addresses: _set_position)
            a, n, split = st.subs[u]
            rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
            for c in range(COLS):
                role = attn.ROLE_O if ash.round_word(split, c) & attn.SPLIT_NOREC else ash.role(c)
                if not split or c % nph == nph - 1:
                    h0, nh = ((kh * nph + 2 * (c // nph)) * qc, 2 * qc) if rnd else (qc * (c - nph + 1), nph * qc) if split else (qc * c, qc)
                    sp = self._sp(S, (L, "attn_out"))
                    at, left = self._x_at((L, "attn_out"), attn.HD * h0, S)
                    assert left >= 4 * sp * 128 * nh
                    sw, m = 4 * sp, sp if ash.one else n  # (words of an x k block: [slot sp][8] bf16; the slots out)
                    seq.drain("C", c, dev.address("ACT", at + 16 * a), 16 * m * 32 * nh,
                              None if m == sp else (4 * m, 32 * nh, sw, sw * 32 * nh))
                nk = attn.REC_K if ash.one else 2 * attn.REC_K  # (one pass: a K record; sub-steps: its pair)
                nr = ash.records(n)
                sizes = {attn.ROLE_K: (nk,) * nr, attn.ROLE_V: (attn.REC_V, attn.ROWS_B) * nr, attn.ROLE_O: ()}[role]
                for k, nb in enumerate(sizes):  # (a record's panels into its pair: V_SCATTER)
                    dims = attn.V_SCATTER if nb == attn.REC_V else None  # (REC_K == REC_V)
                    hd = seq.drain("C", c, dev.address("KV", self.kv_off[L]), nb, dims)
                    st.drains.append((hd, L, c, u, k))

        for u, (a, n, split) in enumerate(st.subs):
            rnd = split & 3 == attn.SPLIT_ROUND  # (a round: both chains on its KV head, the cache shared 4 columns apart)
            last = u == len(st.subs) - 1 and run
            if not split & attn.SPLIT_KV:  # (a sub-step's second round: issued with its first)
                if u or rnd:
                    seq.sync()  # (sub-step u - 1's records in DDR; a shared stream: every column's objects before it through)
                cache(u)
                if last and not rnd and self._a_bds > 1 + attn.TAILS:  # (a later block's after the X objects: ``_weight_run``)
                    self._weight_run(seq, run, blocks=slice(1))
                if u == 0 and not rnd:
                    seq.sync()  # (attn_in's q, K, V in DDR)
                inputs(u)
                if rnd:  # (the second round, on the same streams after the first's objects: its inputs, then the tasks
                    # that wait for the first round's to finish (a channel queues 4): the second's tails, the drains)
                    inputs(u + 1)
                    cache(u + 1)
                if last and not rnd:
                    self._weight_run(seq, run, blocks=slice(1 if self._a_bds > 1 + attn.TAILS else 0, None))
                outputs(u)
                if rnd:
                    outputs(u + 1)
            if rnd and u == len(st.subs) - 1:  # (then the columns' own objects: they keep no order with a shared stream's)
                seq.wait_fills("A", tuple(range(nph)))
                if run:
                    self._weight_run(seq, run)

    def _moe(self, seq: Sequence, L: int, S: int, to: tuple[int, str], run: list[tuple[int, str]] | None) -> None:
        """An MoE layer (module docstring) after its residual's drains into moe_in's x: moe_in, the route object (the
        control packets' round trip: the drain releases LOCK_CP, the control task acquires it; the array's A task),
        sdown, the expert object (after the routing's drain), then ``run``'s weights (after the array's last packet:
        LOCK_DONE), the partials into PART, the combine; its residual -> ``to``."""
        dev, h8 = self.dev, self.H // COLS
        self._weights(seq, (L, "moe_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "moe_in")][0], S)
        for c in range(COLS):  # (per slot: the shared expert's h, then the router's 32 logits (fp32))
            self._values_to(seq, c, (L, "sdown"), e0=64 * c, n=64, S=S)
            seq.drain("C", c, dev.address("ACT", self.xch[L] + 4 * META + 128 * c), S * 128, (32, S, 256, 32))
        seq.sync()
        for c in range(COLS):
            cp = self.cp[L] + c * (CP_BYTES + XB)
            seq.fill("X", c, dev.address("ACT", self.xch[L]), XB)
            seq.drain("C", c, dev.address("ACT", cp), CP_BYTES, release=(MR.LOCK_CP, 1))
            seq.fill("X", c, dev.address("ACT", cp), 4 * MR.PACKET_WORDS, packet=False, acquire=(MR.LOCK_CP, 1))
            seq.device_task("A", c)
        self._weights(seq, (L, "sdown"), 0)
        rg = self.inp[(L, "sdown")][0]  # (its x [h | h]: the first input object twice)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", rg.off), 2 * XB, DIMS_OBJ)
            seq.fill("X", c, dev.address("ACT", rg.off + XPAY), XB)
        seq.sync()
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.cp[L] + c * (CP_BYTES + XB) + CP_BYTES - 4 * META), XB)
        for c in range(COLS):  # (the array's packets are through: later host pushes on A keep their order)
            lock = txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * MR.LOCK_DONE)
            seq.txn.maskpoll32(lock, 1, 0x3F)
            seq.txn.write32(lock, 0)
        if run:
            self._weight_run(seq, run)
        pc = self.part_col
        for c in range(COLS):  # (acc's objects, per 128 rows k, slot, 4 of 32 rows -> PART[k / 2][slot][k % 2][c] ..)
            seq.drain("C", c, dev.address("ACT", self.part + XB + 2 * h8 * c), 4 * S * h8, (h8 // 2, S, XPAY // 2, XPAY // 4),
                      COLS, (pc // 4, COLS))
        seq.sync()
        for c in range(COLS):  # (the partials are in DDR: the experts' tasks are complete)
            seq.device_done("A", c)
            seq.fill("X", c, dev.address("ACT", self.part + c * pc), (1 + 2 * S) * XB, DIMS_OBJ)
        for c in range(COLS):
            self._values_to(seq, c, to, e0=h8 * c, n=h8, S=S)

    def _build(self, S: int, lm: bool, subs: tuple[tuple[int, int, int], ...]) -> _Step:
        """The transaction of a step of S slots (``lm``: with the LM head; ``subs``: the attention's sub-steps)."""
        dev, cfg = self.dev, self.cfg
        seq = Sequence(self.op, inputs=("X",))
        st = _Step(S, subs)
        self._gst = st.gst  # (``_gdn4``'s state transfers)
        if self.moe:  # (the control packets' route; the routing's shim locks)
            for c in range(COLS):
                MR.control_route(seq.txn, c)
                for lk in (MR.LOCK_CP, MR.LOCK_DONE):
                    seq.txn.write32(txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * lk), 0)
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
                self._attention(seq, st, L, S, run)
                self._phase(seq, (L, "attn_out"), S, (L, ffn))
            else:
                self._gdn(seq, L, S)
                self._phase(seq, (L, "gdn_out"), S, (L, ffn))
            nxt = self._input(self.layers[i + 1]) if i + 1 < len(self.layers) else (-1, "lm")
            if self.moe:
                self._moe(seq, L, S, nxt, next((r for r in runs if r[0] == nxt), None))
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
        """(a step of an 8-slot design: whether its chain ends its block, ``_ends``: only then its state panels drain)"""
        return (self.start, self._ends)

    def prepare(self, S: int, logits: bool = True, start: int | None = None) -> None:
        """(8 slots: the steps that end their block and those that do not)"""
        for ends in (True, False) if self.g8 else (True,):
            self._ends = ends
            super().prepare(S, logits, start)
        self._ends = True

    def patch_ahead(self, S: int, p: int) -> None:
        """(a step of a chain)"""
        if self.g8:
            self._ends = p % 8 + S >= 8
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

    def _patch_tables(self, st: _Step) -> dict[str, np.ndarray]:
        """A step's cache fills, tail fills and record drains as arrays: BD word index, sub-step, the layer's index among
        the attention layers, the column's place in its chain, byte offsets in KV of group 0 of its KV head and of its
        ring, a fill's kind (``FILLS``), a tail's object, a drain's index."""
        ash, nph = self.ash, self.ash.nph
        ix = {L: i for i, L in enumerate(self.fatt)}

        def where(L: int, c: int) -> tuple[int, int, int, int]:
            return ix[L], c % nph, self.kv_off[L] + c // nph * ash.head_bytes, self.ring(L, c // nph)

        f = np.array([(at, u, *where(L, c), FILLS.index(kind)) for at, L, c, u, kind in st.fills], np.int64).reshape(-1, 7)
        t = np.array([(at, u, *where(L, c), k) for at, L, c, u, k in st.tails], np.int64).reshape(-1, 7)
        r = np.array([(hd[0], u, *where(L, c), k) for hd, L, c, u, k in st.drains], np.int64).reshape(-1, 7)
        at = np.concatenate([x[:, 0] for x in (f, t, r)] + [x[:, 0] + 2 for x in (f, t, r)])
        return {"f": f, "t": t, "r": r, "lines": np.unique(4 * at & ~63)}

    def _half(self, p: int) -> int:
        """(chunkwise) The GSTATE half that holds the state at the start of position p's block (``gdn``'s "Chunkwise":
        blocks of 8 positions; a step reads its block's half and writes the other, the state at the block's end
        when the step reaches it): from the half _gh of _gp's block, the other at each block after."""
        return (self._gh + p // 8 - self._gp // 8) & 1

    def _gstate_at(self, st: _Step, p: int) -> None:
        """(chunkwise) A step's state transfers for position p: the panels from p's block's half, into the other."""
        h = self._half(p)
        if st.ghalf == h:
            return
        if not isinstance(st.gst, tuple):  # (as arrays: BD word index, offset, bytes, drain; their lines)
            at = np.array([g[0][0] if g[3] else g[0] for g in st.gst], np.int64)
            st.gst = (at, *(np.array([g[k] for g in st.gst]) for k in (1, 2, 3)), np.unique(np.concatenate([4 * at, 4 * at + 8]) & ~63))
        at, off, nb, drain, lines = st.gst
        self._bd_words(st.tr.words, at, self.dev.address("GSTATE") + off + np.where(drain, 1 - h, h) * self.ghalf, nb)
        st.tr.flush_lines(lines)
        st.ghalf = h

    def _set_position(self, st: _Step, p: int) -> None:
        """The F_ATT objects (per sub-step: its positions, cache objects, rotations) and the step's cache and ring
        transfers (and a chunkwise core of 4's state transfers: ``_gstate_at``)."""
        if self.blocks:
            self._gstate_at(st, p)
        if self.ash is None:
            return
        S = st.S
        assert 0 <= p and p + S <= self.max_positions, (p, S)
        assert st.subs == self.subs(p, S), (p, S, st.subs)
        dev, ash, T = self.dev, self.ash, self.T
        nph, U = ash.nph, len(st.subs)
        act = dev.view("ACT")
        c0, s0 = attn.rope_cs(self.cfg, self.rope[p:p + S])
        cs = np.concatenate([c0, s0], 1)
        self._rings_from_prefill(p)
        for L in self.fatt:
            self.vring.check(L, p, p)
        w = np.zeros((U, COLS, len(F_WORDS)), np.int32)  # (per sub-step and column)
        gr = np.zeros((U, len(FILLS), nph, 2), np.int64)  # (a fill's groups, per place in a chain)
        su = np.zeros((U, 2), np.int64)  # (a sub-step's first position, slots)
        for u, (a, n, split) in enumerate(st.subs):
            q, g = p + a, (p + a) // attn.POS_PER_GROUP
            G = (q + n - 1) // attn.POS_PER_GROUP + 1  # (groups through the last slot's; split: chain core k's attention groups b[k] .. b[k + 1] - 1)
            b = ash.bounds(g) if split else [0] * (nph + 1)
            su[u] = q, n
            for c in range(COLS):
                k = c % nph
                nt = ash.records(n) if not split or k == nph - 1 else 0
                o = attn.OBJS
                if not split:  # (a chain's first column: then group g's K and V objects, for its K records; one pass: none)
                    na, g0, ga, gj = o * G + nt + (2 if k == 0 and not ash.one else 0), 0, 1 << 20, 0
                elif k == 0:  # (then group g's K and V objects for its K records)
                    na, g0, ga, gj = o * b[1] + 2, 0, b[1], g
                else:
                    na, g0, ga, gj = o * (b[k + 1] - b[k]) + nt, b[k], 1 << 20, 0
                ana, na = na, 0 if ash.one and k else na  # (one pass: the chain's cores on the first column's stream)
                nq = 3 if split & 3 == attn.SPLIT_ROUND else nph + 1 if split else 2  # (per slot the X objects after F_ATT)
                sw = ash.round_word(split, c)
                v = {"pos": q, "ns": n, "sa": 0, "skip": nq * n, "split": sw, "nrel": ash.n_rel(c, sw, n), "ntail": nt, "naskip": na,
                     "g0": g0, "gatt": ga, "gj": gj, "ana": ana}
                w[u, c] = [v[k] for k in F_WORDS]  # (by name: a word of F_WORDS without a value is a KeyError)
            for k in range(nph):
                gr[u, :, k] = (0, G), (g, g + 1), (b[k], b[k + 1])
        fo = self._fobj[:, :, :U]  # (the F_ATT objects' first words: [layer][column][sub-step])
        a32 = act.view(np.int32)
        a32[fo[..., None] + F_ATT_AT] = w[np.arange(U), np.arange(COLS)[:, None]]
        q0 = META + 2 * attn.HD  # (the sub-step's slots' angles, in words)
        af = act.view(np.float32)
        for u, (a, n, _) in enumerate(st.subs):
            af[fo[..., u, None] + q0 + np.arange(64 * n)] = cs[a:a + n].reshape(-1)
        lines = np.concatenate([np.arange(0, 192, 64), 4 * q0 + np.arange(0, 4 * 64 * max(n for _, n, _ in st.subs) + 64, 64)])
        dev.flush_at("ACT", 4 * fo[..., None] + lines)
        if st.patch is None:
            st.patch = self._patch_tables(st)
        t, kv0, words = st.patch, dev.address("KV"), st.tr.words
        f = t["f"]  # (at, u, layer, k, base, ring, kind)
        g = gr[f[:, 1], f[:, 6], f[:, 3]]
        o = attn.OBJS
        self._bd_words(words, f[:, 0], kv0 + f[:, 4] + o * g[:, 0] * T, np.where(f[:, 6] == 1, 2, o * (g[:, 1] - g[:, 0])) * T)
        f = t["t"]  # (the ring slots of the sub-step's first and last records)
        r = (su[f[:, 1], 0] + np.where(f[:, 6], su[f[:, 1], 1] - 1, 0)) >> 3
        self._bd_words(words, f[:, 0], kv0 + f[:, 5] + (r & 1) * T, np.full(len(r), T, np.int64))
        f = t["r"]  # (the first column's K records; the last's V panels (into their pairs) and rows, of record 0, 1)
        v = f[:, 3] == nph - 1
        r = (su[f[:, 1], 0] + np.where(np.where(v, f[:, 6] >= 2, f[:, 6] > 0), su[f[:, 1], 1] - 1, 0)) >> 3
        head, j = f[:, 4] + o * (r // 4) * T, r % 4
        kind = np.where(v, 1 + f[:, 6] % 2, 0)  # (K, V panels, rows)
        pair = j // 2 * 2 * attn.REC_K  # (a record in its pair; sub-steps: the K records' pairs whole)
        off = np.choose(kind, [head + pair + (j % 2 * 72 if ash.one else 0), head + T + pair + j % 2 * 72, f[:, 5] + (r & 1) * T])
        nk = attn.REC_K if ash.one else 2 * attn.REC_K
        self._bd_words(words, f[:, 0], kv0 + off, np.array([nk, attn.REC_V, attn.ROWS_B])[kind])
        st.tr.flush_lines(t["lines"])

    def _rings_from_prefill(self, p: int) -> None:
        """Before a step at p, of each attention layer whose caches a prefill wrote last (it writes no ring): its open
        record (p >> 3) as rows into its ring slot, from the record's V panels in the cache (their values are bf16 values:
        exact), so that a step reads every open record from the ring."""
        layers = [L for L in self.fatt if L in self.vring.from_prefill]
        if not layers:
            return
        t0 = time.perf_counter()
        r, dev, ash, T = p >> 3, self.dev, self.ash, self.T
        if p % 8:
            kv = dev.view("KV")
            for L in layers:
                for h in range(ash.kvh):  # (the record in its pair [d-block][record 2] of the group's V object)
                    a = self.kv_off[L] + h * ash.head_bytes + attn.OBJS * (r // 4) * T + T + (r % 4 // 2) * 2 * attn.REC_V
                    dev.fetch("KV", a, 2 * attn.REC_V)
                    v = block_values(kv[a:a + 2 * attn.REC_V].reshape(attn.ND, 2, 8, 9)[:, r % 2])  # ([d-block][dim][position])
                    o = self.ring(L, h) + (r & 1) * T
                    kv[o:o + attn.ROWS_B] = np.ascontiguousarray(v.transpose(0, 2, 1), bfloat16).view(np.uint8).reshape(-1)
                    dev.flush("KV", o, attn.ROWS_B)
        for L in layers:
            self.vring.slots.setdefault(L, {})[r & 1] = (r, p)
            self.vring.from_prefill.discard(L)
        self._once(t0)

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
        at = self._fobj[:, :, 0, None] + META + 2 * attn.HD + 64 * (8 - L) + np.arange(64 * L)
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
            fo = self._fobj[:, :, 0].ravel()
            a32[fo + M_BRANCH] = aword
            a32[fo[:, None] + M_AMASK + np.arange(4)] = w.view(np.int32)
            a32[fo + M_ALANE] = lanes
            lines.append(4 * (fo + M_BRANCH))
        if self.g8:
            a32[self._gin + M_BRANCH] = word
            lines.append(4 * (self._gin + M_BRANCH))
        if lines:
            self.dev.flush_at("ACT", np.unique(np.concatenate(lines) & ~63))
        self._branch = key

    def _set_gdn(self, *, nrep: int, cshift: int, commit: bool) -> None:
        """The GDN layers' gdn_in metadata words of the lazy commit (chunkwise: the step's lanes in its block before
        it, the q history's base row, the k / v histories' and the gates' first row)."""
        if self.gobj:
            a32 = self.dev.view("ACT").view(np.int32)
            a32[self._gin + M_NREP], a32[self._gin + M_CSHIFT], a32[self._gin + M_COMMIT] = nrep, cshift, int(commit)
            a32[self._gin + M_KSHIFT] = self.kshift
            self.dev.flush_at("ACT", 4 * (self._gin + M_NREP) & ~63)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, position: int, *, logits: bool = True, commit: bool = True,
             branch: tuple[int, int] | None = None) -> np.ndarray | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the first layer's inputs (embeddings) at positions
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
        assert self._committed or self._accepted, "accept() the previous lazy step's tokens first"
        assert branch is None or (S == 8 and self.can_branch and not commit and branch[1] in (1, 2)
                                  and 0 <= branch[0] and position % 8 + branch[0] + branch[1] <= 7), (position, S, branch)
        self._conv_for_decode(position)
        if branch is not None:  # (after a prefill's text rope; a step patched ahead: its branch slots' angles again)
            c, L = branch
            self.rope[position + 8 - L:position + 8] = self.rope[position + c + 1:position + c + 1 + L]
            if self._ahead is not None:
                self._branch_angles(position, L)
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._set_branch(branch, position, S)
        if self.blocks:  # (the block's lanes before the step; the step's words do not depend on commit)
            assert position == self._gnext, (position, self._gnext)
            self._ends = not self.g8 or position % 8 + S - (branch[1] if branch else 0) >= 8
            self._gpos = position
            self._set_gdn(nrep=position % 8, cshift=self.cshift, commit=False)
        else:
            self._set_gdn(nrep=self.nrep, cshift=self.cshift, commit=commit)
        self._committed = commit
        xf = x
        self._write_loadr(xf)
        first = self._input(self.layers[0])
        self._write_input(self.inp[first][0], interleave(xf, self._sp(S, first)).view(np.uint8).reshape(-1))
        out = self._run(S, position, logits)
        self._accepted = False
        if commit:  # (all S accepted)
            if self.blocks:
                self._advance(S)
            else:
                self.nrep, self.cshift = 0, S
        return out

    def _advance(self, m: int) -> None:
        """(chunkwise) The last step's first m tokens accepted: the next step's histories' rows (past its block's end:
        the next block's, 8 rows on) and q base row."""
        self.kshift = 8 if self._gpos % 8 + m >= 8 else 0
        self.cshift = m
        self._gnext = self._gpos + m

    def _ran(self, p: int, S: int) -> None:
        for L in self.fatt:
            self.vring.ran(L, p, S)

    def _ring_state(self) -> tuple:
        """(and whether a prefill's conv states wait in GHDR: the next step resets ``rope``)"""
        return (*super()._ring_state(), self._conv_in_ghdr)

    def _conv_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """bf16 indices of the GDN layers' conv states: (GOBJ history rows 0 .. 2 of every block, GHDR's [row][channel])
        in the same order, and which are a k or v block's; the decode's base rows are cshift rows past the GOBJ ones
        (chunkwise: a k or v block's, the block's lanes before the position past the history's first row,
        ``_kv_rows``)."""
        if getattr(self, "_cvi", None) is None:
            ch = self.gsh.ch
            r, j = np.divmod(np.arange(3 * ch), ch)
            b = j // gdn.D
            k = np.where(b < 4, b, 5 + 3 * ((b - 4) // 2) + (b - 4) % 2)
            src, dst = [], []
            at = XPAY * k + 2048
            if self.gsh.chunk:  # (8 slots: from the A object's second half on, through the B object's)
                at = np.array([self.gsh.hist_at(int(i)) for i in b])
            for L in self.gobj:
                for c in range(COLS):
                    src.append(self._gpay(L, c, 0) + at + 2 * (gdn.D * r + j % gdn.D))
                    dst.append(self.ghdr[L] + c * HSLOT + CS_OFF + 2 * np.arange(3 * ch))
            kv = np.tile((b >= 2) & self.blocks, len(self.gobj) * COLS)
            self._cvi = (np.concatenate(src) // 2, np.concatenate(dst) // 2, kv)
        return self._cvi

    def _rows_at(self, kv: np.ndarray, p: int) -> np.ndarray:
        """bf16 offsets of the conv states before position p from their blocks' history rows 0 .. 2 (``_conv_index``)."""
        return gdn.D * np.where(kv, self.kshift + p % 8, self.cshift)

    def conv_for_prefill(self) -> None:
        """Before a prompt chunk: the decode's conv states (the base rows of the next step) into GHDR. The decode's GDN
        states must hold every accepted token (no pending replay)."""
        if self.gsh is None or self._conv_in_ghdr:
            return
        assert self.nrep == 0, "a prefill after a lazy step: commit its accepted tokens first"
        assert not self.blocks or self._gnext % 8 == 0, "a prefill from a GDN block's start"
        dev = self.dev
        src, dst, kv = self._conv_index()
        dev.fetch("GOBJ")
        dev.view("GHDR").view(np.uint16)[dst] = dev.view("GOBJ").view(np.uint16)[src + self._rows_at(kv, self._gnext)]
        dev.flush("GHDR")
        self._gcur = self._half(self._gnext)
        self._conv_in_ghdr = True
        self._branch = None  # (a prompt chunk in between: the next step writes its step words)

    def conv_fresh(self) -> None:
        """Before a sequence's first prompt chunk (at position 0: no conv history, its zero rows the chunk's own,
        ``QwenPrefill._conv_zero``): GHDR's conv states count as current (the chunk's GdnOut writes them) without
        ``conv_for_prefill``'s copy (its fetch of GOBJ: ~0.8 ms on MiniCPM-V)."""
        if self.gsh is None:
            return
        assert self.nrep == 0, "a prefill after a lazy step: commit its accepted tokens first"
        self._gcur = self._half(0)
        self._conv_in_ghdr = True
        self._branch = None

    def _conv_for_decode(self, p: int) -> None:
        """Before a decode step at p after a prefill: GHDR's conv states into the conv objects' base rows (chunkwise:
        the decode's blocks from the prefill's state at p, in its half; the block's lanes before p, none of the sequence's,
        take zero gates: no change of the state, so that blocks start at multiples of 8 and prefills at a block's start)."""
        if not self._conv_in_ghdr:
            return
        t0 = time.perf_counter()
        dev = self.dev
        self.nrep, self.cshift, self.kshift = 0, 0, 0
        src, dst, kv = self._conv_index()
        dev.fetch("GHDR")
        g16 = dev.view("GOBJ").view(np.uint16)
        g16[src + self._rows_at(kv, p)] = dev.view("GHDR").view(np.uint16)[dst]
        if self.blocks:
            self._gp, self._gh, self._gnext = p, self._gcur, p
            gobj = dev.view("GOBJ")
            for L in self.gobj:
                for c in range(COLS):
                    for a in ([self._gpay(L, c, gdn.H_SLOT) + gdn.H_DN] if self.g8 else
                              [self._gpay(L, c, self.gsh.kh_slot(kh)) + gdn.P_D for kh in range(2)]):
                        gobj[a:a + 64 * (p % 8)] = 0
        dev.flush("GOBJ")
        self._a_bds = getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)  # (the host's BDs on A)
        self._conv_in_ghdr = False
        self.rope[:] = _text_rope(self.max_positions)
        self._once(t0)

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
        core of 4: the next step recomputes its block's lanes from the histories, ``_advance``)."""
        assert 0 <= n <= self._last_s and not self._accepted
        if self._committed:
            assert n == self._last_s, "a committing step's tokens are all accepted"
        elif self.blocks:
            self._advance(n)
        else:
            self.nrep, self.cshift = n, n
        self._accepted = True

    def cache_rows(self) -> list[tuple[np.ndarray, int, int]]:
        """(per attention layer and KV head: groups of 32 positions, 2 objects each)"""
        if self.ash is None:
            return []
        rows = [self.kv_off[L] + h * self.ash.head_bytes for L in self.kv_off for h in range(self.ash.kvh)]
        return [(np.array(rows), attn.OBJS * self.T, attn.POS_PER_GROUP)]

    def snapshot(self, position: int) -> dict[str, object]:
        """(and the GDN layers' states, conv objects with their replay objects, and conv states of the prefill, the lazy
        commit's fields, ``rope``)"""
        out = super().snapshot(position)
        if self.gsh is not None:
            for name in ("GSTATE", "GOBJ", "GHDR"):
                self.dev.fetch(name)
                v = self.dev.view(name)[self._gspan()] if name == "GSTATE" else self.dev.view(name)
                out[name] = host_bytes(v.shape)
                out[name][:] = v
        out["fields"] = (self.nrep, self.cshift, self._conv_in_ghdr, self._committed, self._accepted, self._gp, self._gh,
                         self._gpos, self._gnext, self._gcur, self.kshift)
        out["rope"] = self.rope.copy()
        return out

    def restore(self, snapshot: dict[str, object]) -> None:
        super().restore(snapshot)
        (self.nrep, self.cshift, self._conv_in_ghdr, self._committed, self._accepted, self._gp, self._gh, self._gpos,
         self._gnext, self._gcur, self.kshift) = snapshot["fields"]
        if self.gsh is not None:
            for name in ("GSTATE", "GOBJ", "GHDR"):
                self.dev.view(name)[self._gspan() if name == "GSTATE" else slice(None)] = snapshot[name]
                self.dev.flush(name)
        self.rope[:] = snapshot["rope"]

    def _gspan(self) -> slice:
        """GSTATE's bytes a snapshot keeps: its states (chunkwise: the half that the next position's block starts
        from, the other is the next block's scratch; a prefill's next position: its half)."""
        if not self.blocks:
            return slice(None)
        h = self._gcur if self._conv_in_ghdr else self._half(self._gnext)
        return slice(256 + h * self.ghalf, 256 + (h + 1) * self.ghalf)

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L (-1: the LM head) read in the last step, of a slot (bf16 values, fp32)."""
        off, sp = self.inp[self._input(L)][0].off + XB, self._sp(self._last_s, self._input(L))
        self.dev.fetch("ACT", off, 2 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)[:, slot].reshape(-1)
        return v.view(bfloat16).astype(np.float32)


__all__ = ["FLAGS", "F_EMITR", "F_GU", "F_MSG", "F_RESID", "QwenOverlay"]
