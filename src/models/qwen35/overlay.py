"""A qwen35 model on the multi-slot decode overlay (``qwen35/decode.py``): buffers, the transaction of a step of S
token slots, the step.

``QwenOverlay`` lays out the weight image (``image.QwenImage``, packed at ``load``), the norm-weight objects (NORM),
every phase's input (ACT: per K group [metadata object | x slot-interleaved], ``decode.design.region_bytes``; a producer
drains its outputs straight into its consumer's x), the residual-slice objects (F_LOADR), the GDN layers' objects (GOBJ:
per column [header | conv q / k | K | per pair of v heads: conv v, deltas], ``gdn``) and states (GSTATE: per column its
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

import math
import os

import numpy as np
import torch

from artlite import txn
from artlite.decode import gemv
from artlite.decode import phases as mp
from artlite.decode.design import (
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
from artlite.decode.model import ModelOverlay, host_bytes, paged_formats
from artlite.decode.vring import substeps
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.moe import route as MR
from artlite.operators import paging
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction, WeightBuffer

from . import attn, gdn
from .decode import (
    M_ANA,
    M_ATREC,
    M_COMMIT,
    M_CSHIFT,
    M_G0,
    M_GATT,
    M_GJ,
    M_NREP,
    M_NTAIL,
    M_POS,
    M_SA,
    M_SPLIT,
    QwenDecodeOperator,
)
from .image import NORMS, QwenImage, gdn_v_heads

FLAGS = {"eh": F_RESID | F_EMITR,  # (a drafter's first phase: ``draft``)
         "gdn_in": F_NORM | F_MSG, "attn_in": F_NORM, "gdn_out": F_RESID | F_EMITR, "attn_out": F_RESID | F_EMITR,
         "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM, "moe_in": F_NORM, "sdown": F_RESID | F_WS}
MOE_BLOCKS = {"sgu": B_GU, "router": B_PAIR | B_F32, "sgate": B_PAIR | B_KEEP}  # (moe_in's blocks: ``decode.design``)
MOE_MAX_K = 2048  # (an MoE model's K groups at most: ``image.payload``)
CP_BYTES = 128 * MR.RBUF_OBJECTS  # (a column's routing objects in CP; the expert object starts 256 B before their end)
KV_PAD = 4096 - 256  # positions past the caches (the last layer's): the prefill's sub-chunk slices
CS_OFF = 4 * META + 640  # GHDR: the conv state's bytes in a column's slot (prefill.CS_OFF)
HSLOT = 12288  # GHDR: a column's slot (the prefill's conv-state drain writes 8 KB from CS_OFF)
DIMS_OBJ = (XB // 8, 2, XB // 8, XPAY // 4)  # X objects of XB bytes, XPAY apart (the overlapping regions)
FILLS = ("all", "new", "run")  # a step's cache fills: the whole cache, the sub-step's group, a split column's run
F_WORDS = np.array([M_POS, M_NS, M_SA, M_SKIP, M_SPLIT, M_NREL, M_NTAIL, M_NASKIP, M_G0, M_GATT, M_GJ, M_ANA])  # (an F_ATT object's)


class _Step:
    """The transaction of a step of S slots (its attentions' sub-steps ``subs``: first slot, slots, whether the
    chains divide their caches' groups) and its words that follow the position: the KV cache and ring reads (A: their
    address and length) and the record drains (C: their address)."""

    def __init__(self, S: int, subs: tuple[tuple[int, int, bool], ...]):
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


def slots_of(cfg: QwenConfig, fmts: set[str]) -> int:
    """A step's token slots at most: 8 for a dense model with its own MTP block (its drafter's chains go deep: the
    verify's slots pay where the drafter is right) whose formats the decode GEMV runs at 8 slots (``decode.gemv``),
    else 4."""
    return 8 if cfg.nextn_layers == 1 and not cfg.moe and all(gemv.supported(f, 4, 8) for f in fmts) else 4


def _text_rope(n: int) -> np.ndarray:
    return np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1)


class QwenOverlay(ModelOverlay):
    """A qwen35 GGUF model on the multi-slot decode overlay. ``layers``: the layers to run (default: all; tests run
    slices), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the KV caches' capacity, ``weights``: the
    image's buffer allocated ahead (its image's layers and vocabulary), ``formats``: tile formats the design runs
    besides the image's (a drafter's: ``draft``; past three formats, all but the model's largest as code pages),
    ``image``: an image laid out already (``ModelOverlay``'s: a drafter's view of its model's).

    ``rope`` (max_positions, 3): the M-RoPE position (T, H, W) of each cache slot, which the step's and the prefill's
    rotations take; the slot's index (text from position 0) until a model writes others (an image's tokens, the
    text after it: ``models.qwen35``), again after ``reset``."""

    CONFIG, IMAGE = QwenConfig, QwenImage

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, weights: WeightBuffer | None = None, formats: tuple[str, ...] = (),
                 image: QwenImage | None = None, slots: int | None = None):
        """slots: a step's token slots at most (default ``slots_of``)."""
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights, image=image)
        cfg, img = self.cfg, self.image
        fmts = {b.fmt for r in img.regions + img.slabs for b in r.blocks} | set(formats)
        slots = slots or slots_of(cfg, fmts)
        att = [cfg.attention[L] for L in self.layers]
        self.gsh = gdn.GdnShape(cfg, slots=slots) if not all(att) else None
        self.ash = attn.AttnShape(cfg, max_positions, a_bytes=img.tile_bytes, slots=slots) if any(att) else None
        self.max_positions = self.ash.max_positions if self.ash else max_positions
        self.rope = _text_rope(self.max_positions)
        self.period = 4 * attn.POS_PER_GROUP  # (the steps' sub-steps repeat from group 3 on: ``subs``)
        self.moe = cfg.moe
        self.op = QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=img.tile_bytes,
                                      eps=cfg.rms_norm_eps, gdn=self.gsh, attn=self.ash,
                                      moe=(cfg.n_experts, cfg.n_experts_used) if cfg.moe else None,
                                      max_k=MOE_MAX_K if cfg.moe else MAX_K,
                                      paged=paged_formats(gguf, fmts, wide=cfg.moe and max(img.R[f] for f in fmts) > 16),
                                      slots=slots, bcast=slots == 8)
        self.g8 = self.gsh is not None and slots == 8  # (the GDN core of 8 slots: ``gdn``'s "8 slots")
        self.flags = FLAGS | ({"gdn_in": F_NORM | gdn.F_GIN} if slots == 8 else {})
        self._a_bds = getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)  # (the host's BDs on A)
        self._conv_in_ghdr = False
        self._cvi: tuple[np.ndarray, np.ndarray] | None = None
        self._committed, self._accepted = True, True
        self._order = None if img.order is None else torch.from_numpy(img.order)  # (``_logits``)
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
        m[M_FLAGS], m[M_NS], m[M_NX] = F_MAIN | F_COMB, S, S
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
        self.hslot = HSLOT
        if self.gsh is not None:
            self.npair = self.gsh.hc // 2
            self.gcol = 256 + XPAY * (self.gsh.n_slots if self.g8 else 5 + 3 * self.npair)
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
        self.xch, self.cp = {}, {}
        if self.moe:  # (per MoE layer its route object, per column its routing's objects and expert object; PART)
            for L in self.layers:
                self.xch[L] = take("ACT", XB)
                self.cp[L] = take("ACT", COLS * (CP_BYTES + XB))
            self.part = take("ACT", COLS * self.part_col)
        self._fobj = np.array([[[(self.fatt[L] + (c * self.ash.u + u) * XB) // 4 for u in range(self.ash.u)]
                                for c in range(COLS)] for L in self.fatt], np.int64)  # ([layer][column][sub-step], words)
        self._gin = np.array([self.inp[(L, "gdn_in")][0].off // 4 for L in self.gobj], np.int64)
        r0 = self._layout["KV"]
        for L in self.kv_off:  # (per KV head two slots: ``ring``)
            self.ring_off[L] = take("KV", self.ash.kvh * 2 * self.T)
        self.rings = (r0, self._layout["KV"] - r0)
        if self.ash is not None:  # (the prefill's slices past the end; a step's second record; a tail fill's T bytes)
            take("KV", (KV_PAD // attn.POS_PER_GROUP + 1) * attn.OBJS * self.T)
        self._sizes()

    @property
    def part_col(self) -> int:
        """PART's bytes per column: the combine's metadata object, then per slot the 8 columns' partials of the
        column's slice ([slot][column][hidden / 8] bf16: one object's payload per slot)."""
        assert self.H * 2 == XPAY, "a slot's partials of a slice in one X object's payload"
        return XB + self.SLOTS * XPAY

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
            MR.check_route(paging.cdo_writes(os.path.dirname(self.op._program.insts_path)), range(COLS))
        norm = self.dev.view("NORM")
        for (L, name), off in self.norm_off.items():
            w = torch.from_numpy(self._f32("output_norm.weight" if L < 0 else f"blk.{L}.{name}.weight")).bfloat16()
            norm[off:off + 2 * self.H] = w.view(torch.uint8).numpy()
        self.dev.flush("NORM")
        for L in self.fatt:
            self._load_attention(L)

    def _f32(self, name: str) -> np.ndarray:
        return np.asarray(self.gguf.tensor(name).raw, np.float32).reshape(-1).copy()

    def _load_attention(self, L: int) -> None:
        act = self.dev.view("ACT")
        qn = self._f32(f"blk.{L}.attn_q_norm.weight") * (np.float32(math.log2(math.e) / 16) if self.ash.one else 1)
        norms = np.concatenate([qn, self._f32(f"blk.{L}.attn_k_norm.weight")])
        for c in range(COLS):
            for u in range(self.ash.u):
                o = self._fatt(L, c, u)
                m = np.zeros(META, np.int32)
                m[M_FLAGS] = attn.F_ATT
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
        m[M_FLAGS], m[M_SKIP] = gdn.F_GDN, sh.n_objects - 1
        if self.g8:  # (the helper's heads per step: ``gdn``'s ``hgdn``)
            m[gdn.M_HPAN] = sh.hc + 2
        for c in range(COLS):
            vh = gdn_v_heads(cfg, c)
            rows = blk.rows[c][blk.rows[c] >= 0]
            o = self.gobj[L] + c * self.gcol
            gobj[o:o + self.gcol] = 0
            gobj[o:o + 4 * META] = m.view(np.uint8)
            if self.g8:  # (the header's dt, A, ssm_norm; per conv block its weights)
                h = self._gpay(L, c, gdn.H_SLOT) + gdn.H_PRM
                gobj[h:h + 640] = gdn.head_payload(dt[vh], A[vh], norm)
                for b, w in enumerate(gdn.conv_weights(cw[rows])):
                    a = self._gpay(L, c, sh.block_slot(b))
                    gobj[a:a + w.size] = w
                continue
            for b, pay in enumerate(gdn.conv_payloads(sh, cw[rows], dt[vh], A[vh], norm)):
                k = b if b < 4 else 5 + 3 * ((b - 4) // 2) + (b - 4) % 2
                gobj[self._gpay(L, c, k):self._gpay(L, c, k) + XPAY] = pay

    def reset(self) -> None:
        """A new sequence: the GDN layers' conv histories, replay objects and states, the KV caches zeroed."""
        super().reset()
        dev = self.dev
        dev.view("GSTATE")[:] = 0
        dev.flush("GSTATE")
        for L in self.gobj:
            self._load_gdn(L)
        dev.flush("GOBJ")
        self.nrep, self.cshift = 0, 0
        self._a_bds = getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)  # (the host's BDs on A)
        self._conv_in_ghdr = False
        self.rope[:] = _text_rope(self.max_positions)

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
                self._values_to(seq, c, to, e0=h8 * c, n=h8, S=S)

    def _gdn(self, seq: Sequence, L: int, S: int) -> None:
        """gdn_in and the GDN step."""
        if self.g8:
            self._gdn8(seq, L, S)
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
        """gdn_in and the GDN step of an 8-slot design (``gdn``'s "8 slots"): gdn_in's rows per slot into the conv
        blocks' second objects, z into the v heads' Z objects, ab into the header (a sync), then per k head its group
        and per v head its group and panels, each round over the columns before its drains (the D objects that the
        core's later X objects wait for)."""
        dev, sh, hc = self.dev, self.gsh, self.gsh.hc
        X = XPAY // 4  # (words)
        self._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "gdn_in")][0], S, (L, "gdn_in"))
        # (a block's objects (half, slot) as [half][slot 8][64]: half 0's S rows, then half 1's 1 KB on; per execution its
        # pair of blocks of the two k heads, or the 6 v blocks; then the next: k after q, z after v)
        hb = 2 * 64 * 8 // 4  # (words between the halves)
        for c in range(COLS):
            pay = lambda k, c=c: dev.address("GOBJ", self._gpay(L, c, k))
            b0 = pay(sh.block_slot(0) + 1) + 3 * 2 * gdn.D  # (q0's rows)
            seq.drain("C", c, b0, 2 * 2 * S * 2 * C_LEN, (S * C_LEN // 2, 2, hb, gdn.KH_SPAN * X), 2,
                      ((pay(sh.block_slot(2)) - pay(sh.block_slot(0))) // 4, 2))  # q, then k rows
            v0 = pay(sh.block_slot(4) + 1) + 3 * 2 * gdn.D
            seq.drain("C", c, v0, hc * 2 * S * 2 * C_LEN, (S * C_LEN // 2, 2, hb, gdn.V_SPAN * X), 2,
                      ((pay(sh.v_slot(0) + 3) - v0) // 4, 2))  # v rows, then z
            seq.drain("C", c, pay(gdn.H_SLOT) + gdn.H_AB, 2 * S * C_LEN)  # ab
        seq.sync()
        vh = [gdn_v_heads(self.cfg, c) for c in range(COLS)]
        last = hc - 1  # (the last v head the core runs: k head 1's last)

        def kh_x(kh: int, c: int) -> None:  # (k head 0's group after the header)
            g0 = sh.kh_slot(kh) - (1 if kh == 0 else 0)
            seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, g0) - 256), (gdn.KH_SPAN + 1 - kh) * XB, DIMS_OBJ)

        def kh_d(kh: int, c: int) -> None:  # (the q history, the step's K object, the k history: 6 KB apart)
            a = dev.address("GOBJ", self._gpay(L, c, sh.kh_slot(kh)) + 2048)
            seq.drain("D", c, a, 3 * gdn.D_BYTES, (gdn.D_BYTES // 8, 2, gdn.D_BYTES // 8, (2 * XPAY - 2048) // 4))

        def v_x(h: int, c: int) -> None:  # (its group, its panels)
            seq.fill("X", c, dev.address("GOBJ", self._gpay(L, c, sh.v_slot(h)) - 256), gdn.V_SPAN * XB, DIMS_OBJ)
            p0 = dev.address("GSTATE", self.gstate[L] + c * self.ps + 16 * h * gdn.PANEL - 256)
            seq.fill("X", c, p0, 16 * XB, DIMS_OBJ)

        def v_d(h: int, c: int) -> None:
            v = self._gpay(L, c, sh.v_slot(h))
            p0 = dev.address("GSTATE", self.gstate[L] + c * self.ps + 16 * h * gdn.PANEL)
            seq.drain("D", c, p0, 16 * gdn.PANEL)
            seq.drain("D", c, dev.address("GOBJ", v + 2048), 2 * gdn.D_BYTES,  # its history, its deltas
                      (gdn.D_BYTES // 8, 2, gdn.D_BYTES // 8, (2 * XPAY - 2048) // 4))
            # its gated outputs (2 KB) at its head in gdn_out's x, the step's d per token after them (2 KB): of the last
            # head into the header, else into the sink past that K group's x
            at, left = self._x_at((L, "gdn_out"), gdn.D * vh[c][h], S)
            xg = next(2 * self.SLOTS * rg.K for rg in self.inp[(L, "gdn_out")] if rg.k0 <= gdn.D * vh[c][h] < rg.k0 + rg.K)
            assert left >= 2048
            if h == last:
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES // 2)
                seq.drain("D", c, dev.address("GOBJ", self._gpay(L, c, gdn.H_SLOT) + gdn.H_DOLD), gdn.D_BYTES // 2)
            else:
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES, (512, 2, xg // 4, 512))

        # the core's units in its order (per k head its group, then its v heads), each unit's fills one unit ahead of
        # the drains of the one before (a channel's fifth task waits for its oldest: the core then has its next objects
        # queued while the firmware waits for drains)
        units = [u for kh in range(2) for u in [(kh_x, kh_d, kh)] + [(v_x, v_d, h) for h in range(kh, hc, 2)]]
        for i in range(len(units) + 1):
            if i < len(units):
                for c in range(COLS):
                    units[i][0](units[i][2], c)
            if i:
                for c in range(COLS):
                    units[i - 1][1](units[i - 1][2], c)

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
        for u, (a, n, split) in enumerate(st.subs):
            if u:
                seq.sync()  # (sub-step u - 1's records in DDR)
            for c in range(0, COLS, nph if ash.one else 1):  # the cache of the column's KV head (address, length:
                # _set_position), its ring (one pass: the chain's first column's stream, to both cores)
                head = dev.address("KV", self.kv_off[L] + c // nph * ash.head_bytes)
                kinds = ("all",) if not split else ("run", "new") if c % nph == 0 else ("run",)
                for kind in kinds:  # (split: the chain's first column its run, then the new positions' group's K and V objects)
                    st.fills.append((seq.fill("A", c, head, attn.OBJS * T), L, c, u, kind))
                if not split or ash.role(c) == attn.ROLE_V:  # (a column that attends to the newest group: the tails)
                    st.tails += [(seq.fill("A", c, dev.address("KV", self.ring(L, c // nph)), T), L, c, u, t)
                                 for t in range(attn.TAILS)]
                if not ash.one and not split and c % nph == 0:  # (then the new positions' group again, for their records)
                    st.fills.append((seq.fill("A", c, head, attn.OBJS * T), L, c, u, "new"))
            last = u == len(st.subs) - 1 and run
            if last and self._a_bds > 1 + attn.TAILS:  # (a later block's after the X objects: ``_weight_run``)
                self._weight_run(seq, run, blocks=slice(1))
            if u == 0:
                seq.sync()  # (attn_in's q, K, V in DDR)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self._fatt(L, c, u)), XB)
                if split:  # (the q objects of the chain's columns, in order)
                    seq.fill("X", c, dev.address("ACT", self._q(L, c - c % nph, a)), nph * n * XB,
                             (XB // 8, 2 * n, XB // 8, self.SLOTS * XB // 4))
                else:
                    seq.fill("X", c, dev.address("ACT", self._q(L, c, a)), n * XB)
                seq.fill("X", c, dev.address("ACT", self.rawkv[L] + a * XB), n * XB)
            if last:
                self._weight_run(seq, run, blocks=slice(1 if self._a_bds > 1 + attn.TAILS else 0, None))
            for c in range(COLS):  # o (the column's heads in attn_out's x at the sub-step's slots; split: the chain's
                # last column's of the chain's), then the records (addresses: _set_position)
                role = ash.role(c)
                if not split or role == attn.ROLE_V:
                    h0, nh = (qc * (c - nph + 1), nph * qc) if split else (qc * c, qc)
                    sp = self._sp(S, (L, "attn_out"))
                    at, left = self._x_at((L, "attn_out"), attn.HD * h0, S)
                    assert left >= 4 * sp * 128 * nh
                    sw, m = 4 * sp, sp if ash.one else n  # (words of an x k block: [slot sp][8] bf16; the slots out)
                    seq.drain("C", c, dev.address("ACT", at + 16 * a), 16 * m * 32 * nh,
                              None if m == sp else (4 * m, 32 * nh, sw, sw * 32 * nh))
                nk = attn.REC_K if ash.one else 2 * attn.REC_K  # (one pass: a K record; sub-steps: its pair)
                sizes = {attn.ROLE_K: (nk,) * 2, attn.ROLE_V: (attn.REC_V, attn.ROWS_B) * 2, attn.ROLE_O: ()}[role]
                for k, nb in enumerate(sizes):  # (a record's panels into its pair: V_SCATTER)
                    dims = attn.V_SCATTER if nb == attn.REC_V else None  # (REC_K == REC_V)
                    hd = seq.drain("C", c, dev.address("KV", self.kv_off[L]), nb, dims)
                    st.drains.append((hd, L, c, u, k))

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
        for c in range(COLS):  # (acc's objects [64 rows j][slot] -> PART[j / 4][slot][c][j % 4 ..])
            seq.drain("C", c, dev.address("ACT", self.part + XB + 2 * h8 * c), 4 * S * 128, (32, S, XPAY // 4, 32),
                      COLS, (pc // 4, COLS))
        seq.sync()
        for c in range(COLS):  # (the partials are in DDR: the experts' tasks are complete)
            seq.device_done("A", c)
            seq.fill("X", c, dev.address("ACT", self.part + c * pc), (1 + S) * XB, DIMS_OBJ)
        for c in range(COLS):
            self._values_to(seq, c, to, e0=h8 * c, n=h8, S=S)

    def _build(self, S: int, lm: bool, subs: tuple[tuple[int, int, bool], ...]) -> _Step:
        """The transaction of a step of S slots (``lm``: with the LM head; ``subs``: the attention's sub-steps)."""
        dev, cfg = self.dev, self.cfg
        seq = Sequence(self.op, rolling=True, inputs=("X",))
        st = _Step(S, subs)
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

    def subs(self, p: int, S: int) -> tuple[tuple[int, int, bool], ...]:
        """The attention's sub-steps of a step at p .. p + S - 1 (``decode.vring.substeps``): (first slot, slots,
        whether the chains divide the cache: by its group alone)."""
        if self.ash is None:
            return ()
        if self.ash.one:
            return ((0, S, False),)
        out = []
        for a, n in substeps(p, S, attn.POS_PER_GROUP, self.ash.sub):
            out.append((a, n, attn.AttnShape.splits((p + a) // attn.POS_PER_GROUP)))
        return tuple(out)

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

    def _set_position(self, st: _Step, p: int) -> None:
        """The F_ATT objects (per sub-step: its positions, cache objects, rotations) and the step's cache and ring
        transfers."""
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
        rec = np.array([L in self.vring.from_prefill for L in self.fatt])
        for L, r in zip(self.fatt, rec):
            if not r:
                self.vring.check(L, p, p)
        w = np.zeros((U, nph, len(F_WORDS)), np.int32)  # (per sub-step and column of its chain)
        gr = np.zeros((U, len(FILLS), nph, 2), np.int64)  # (a fill's groups)
        su = np.zeros((U, 2), np.int64)  # (a sub-step's first position, slots)
        for u, (a, n, split) in enumerate(st.subs):
            q, g = p + a, (p + a) // attn.POS_PER_GROUP
            G = (q + n - 1) // attn.POS_PER_GROUP + 1  # (groups through the last slot's; split: chain core k's attention groups b[k] .. b[k + 1] - 1)
            b = ash.bounds(g) if split else [0] * (nph + 1)
            su[u] = q, n
            for k in range(nph):
                nt = attn.TAILS if not split or ash.role(k) == attn.ROLE_V else 0
                o = attn.OBJS
                if not split:  # (a chain's first column: then group g's K and V objects, for its K records; one pass: none)
                    na, g0, ga, gj = o * G + nt + (2 if k == 0 and not ash.one else 0), 0, 1 << 20, 0
                elif k == 0:  # (then group g's K and V objects for its K records)
                    na, g0, ga, gj = o * b[1] + 2, 0, b[1], g
                else:
                    na, g0, ga, gj = o * (b[k + 1] - b[k]) + nt, b[k], 1 << 20, 0
                ana, na = na, 0 if ash.one and k else na  # (one pass: the chain's cores on the first column's stream)
                w[u, k] = q, n, 0, (nph + 1 if split else 2) * n, split, ash.n_rel(k, split, n), nt, na, g0, ga, gj, ana
                gr[u, :, k] = (0, G), (g, g + 1), (b[k], b[k + 1])
        fo = self._fobj[:, :, :U]  # (the F_ATT objects' first words: [layer][column][sub-step])
        a32 = act.view(np.int32)
        a32[fo[..., None] + F_WORDS] = w[np.arange(U), np.arange(COLS)[:, None] % nph]
        a32[fo + M_ATREC] = 0
        a32[fo[rec, :, 0] + M_ATREC] = 1
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
        f = t["t"]  # (the ring slots of the sub-step's first and last records, or the first's V panels in record mode)
        r = (su[f[:, 1], 0] + np.where(f[:, 6], su[f[:, 1], 1] - 1, 0)) >> 3
        trec = rec[f[:, 2]] & (f[:, 1] == 0) & (f[:, 6] == 0)  # (its pair of records)
        off = np.where(trec, f[:, 4] + o * (r // 4) * T + T + (r % 4 // 2) * 2 * attn.REC_V, f[:, 5] + (r & 1) * T)
        self._bd_words(words, f[:, 0], kv0 + off, np.full(len(off), T, np.int64))
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

    def _set_gdn(self, *, nrep: int, cshift: int, commit: bool) -> None:
        """The GDN layers' gdn_in metadata words of the lazy commit."""
        if self.gobj:
            a32 = self.dev.view("ACT").view(np.int32)
            a32[self._gin + M_NREP], a32[self._gin + M_CSHIFT], a32[self._gin + M_COMMIT] = nrep, cshift, int(commit)
            self.dev.flush_at("ACT", 4 * (self._gin + M_NREP) & ~63)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: torch.Tensor, position: int, *, logits: bool = True, commit: bool = True) -> torch.Tensor | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the first layer's inputs (embeddings) at positions
        position .. position + S - 1 (the positions before are in the caches, the GDN states hold the committed ones and
        the previous step's accepted tokens: ``accept``). ``commit``: the GDN states after the step's tokens are written
        (all of them must then be accepted, ``accept(S)``), else the step's tokens are replayed by the next step as far
        as accepted. Returns the slots' logits (S, vocab) bf16 (a view of the NPU's buffer, valid until the next step),
        or without ``logits`` None."""
        x = x.reshape(-1, self.H).float()
        S = x.shape[0]
        assert 1 <= S <= self.SLOTS
        assert self._committed or self._accepted, "accept() the previous lazy step's tokens first"
        self._conv_for_decode()
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._set_gdn(nrep=self.nrep, cshift=self.cshift, commit=commit)
        self._committed = commit
        xf = x.numpy()
        self._write_loadr(xf)
        first = self._input(self.layers[0])
        self._write_input(self.inp[first][0], interleave(xf, self._sp(S, first)).view(np.uint8).reshape(-1))
        out = self._run(S, position, logits)
        self._accepted = False
        if commit:  # (all S accepted)
            self.nrep, self.cshift = 0, S
        return out

    def _ran(self, p: int, S: int) -> None:
        for L in self.fatt:
            self.vring.ran(L, p, S)

    def _conv_index(self) -> tuple[np.ndarray, np.ndarray]:
        """bf16 indices of the GDN layers' conv states: (GOBJ history rows 0 .. 2 of every block, GHDR's [row][channel])
        in the same order; the decode's base rows are cshift rows past the GOBJ ones."""
        if getattr(self, "_cvi", None) is None:
            ch = self.gsh.ch
            r, j = np.divmod(np.arange(3 * ch), ch)
            b = j // gdn.D
            k = np.where(b < 4, b, 5 + 3 * ((b - 4) // 2) + (b - 4) % 2)
            src, dst = [], []
            if self.g8:  # (the history's rows from the A object's second half on, through the B object's)
                k = np.array([self.gsh.block_slot(int(i)) for i in b])
            for L in self.gobj:
                for c in range(COLS):
                    src.append(self._gpay(L, c, 0) + XPAY * k + 2048 + 2 * (gdn.D * r + j % gdn.D))
                    dst.append(self.ghdr[L] + c * HSLOT + CS_OFF + 2 * np.arange(3 * ch))
            self._cvi = (np.concatenate(src) // 2, np.concatenate(dst) // 2)
        return self._cvi

    def conv_for_prefill(self) -> None:
        """Before a prompt chunk: the decode's conv states (the base rows of the next step) into GHDR. The decode's GDN
        states must hold every accepted token (no pending replay)."""
        if self.gsh is None or self._conv_in_ghdr:
            return
        assert self.nrep == 0, "a prefill after a lazy step: commit its accepted tokens first"
        dev = self.dev
        src, dst = self._conv_index()
        dev.fetch("GOBJ")
        dev.view("GHDR").view(np.uint16)[dst] = dev.view("GOBJ").view(np.uint16)[src + gdn.D * self.cshift]
        dev.flush("GHDR")
        self._conv_in_ghdr = True

    def _conv_for_decode(self) -> None:
        """Before a decode step after a prefill: GHDR's conv states into the conv objects' base rows."""
        if not self._conv_in_ghdr:
            return
        dev = self.dev
        src, dst = self._conv_index()
        dev.fetch("GHDR")
        dev.view("GOBJ").view(np.uint16)[src] = dev.view("GHDR").view(np.uint16)[dst]
        dev.flush("GOBJ")
        self.nrep, self.cshift = 0, 0
        self._a_bds = getattr(self.op, "BD_COUNT", {}).get("A", txn.QUEUE_DEPTH)  # (the host's BDs on A)
        self._conv_in_ghdr = False
        self.rope[:] = _text_rope(self.max_positions)

    def _logits(self, S: int) -> torch.Tensor:
        """(an image whose LM rows are not in id order, ``QwenImage.order``: a copy in id order, the ids' rows; an MTP
        drafter's table rows left out)"""
        if self._order is None:
            return super()._logits(S)
        self.dev.fetch("LOGITS", 0, 2 * S * self.vocab)
        x = torch.from_numpy(self.dev.view("LOGITS")[:2 * S * self.vocab]).view(torch.bfloat16).reshape(S, self.vocab)
        return x[:, self._order]

    def _lm_phase(self, seq: Sequence) -> None:
        if self.image.column_major:
            self._weight_run(seq, [(-1, "lm")])
        self._phase(seq, (-1, "lm"), 1, "LOGITS")

    def accept(self, n: int) -> None:
        """The previous step's first n tokens are accepted (the next step's positions start after them): its GDN
        layers replay them (unless it committed: then n must be all of them) and shift their conv histories."""
        assert 0 <= n <= self._last_s and not self._accepted
        if self._committed:
            assert n == self._last_s, "a committing step's tokens are all accepted"
        else:
            self.nrep, self.cshift = n, n
        self._accepted = True

    def cache_rows(self) -> list[tuple[np.ndarray, int, int]]:
        """(per attention layer and KV head: groups of 32 positions, 3 objects each)"""
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
                out[name] = host_bytes(self.dev.view(name).shape)
                out[name][:] = self.dev.view(name)
        out["fields"] = (self.nrep, self.cshift, self._conv_in_ghdr, self._committed, self._accepted)
        out["rope"] = self.rope.copy()
        return out

    def restore(self, snapshot: dict[str, object]) -> None:
        super().restore(snapshot)
        if self.gsh is not None:
            for name in ("GSTATE", "GOBJ", "GHDR"):
                self.dev.view(name)[:] = snapshot[name]
                self.dev.flush(name)
        self.nrep, self.cshift, self._conv_in_ghdr, self._committed, self._accepted = snapshot["fields"]
        self.rope[:] = snapshot["rope"]

    def hidden(self, L: int = -1, slot: int = 0) -> torch.Tensor:
        """The residual stream that layer L (-1: the LM head) read in the last step, of a slot (bf16 values, fp32)."""
        off, sp = self.inp[self._input(L)][0].off + XB, self._sp(self._last_s, self._input(L))
        self.dev.fetch("ACT", off, 2 * sp * self.H)
        v = self.dev.view("ACT")[off:off + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)[:, slot].reshape(-1)
        return torch.from_numpy(v.copy()).view(torch.bfloat16).float()


__all__ = ["FLAGS", "F_EMITR", "F_GU", "F_MSG", "F_RESID", "QwenOverlay"]
