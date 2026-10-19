"""A qwen35 model on the multi-slot decode overlay (``qwen35/decode.py``): buffers, the transaction of a step of S
token slots, the step.

``QwenOverlay`` lays out the weight image (``image.QwenImage``, packed at ``load``), the norm-weight objects (NORM),
every phase's input (ACT: per K group [metadata object | x slot-interleaved], ``decode.region_bytes``; a producer
drains its outputs straight into its consumer's x), the residual-slice objects (F_LOADR), the GDN layers' objects (GOBJ:
per column [header | conv q / k | K | per pair of v heads: conv v, deltas], ``gdn``) and states (GSTATE: per column its
panels, as the prefill reads them), the attention layers' objects (per column an F_ATT object per sub-step, then per
column its q objects; the raw K / V objects) and KV caches (KV, ``overlay.kv``'s layout, and per KV head a ring of
the open V records' values: ``overlay.vring``), and the logits ([slot][vocab] bf16).

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
- gate_up | down (its K groups);

and the LM head (normed) -> LOGITS. The transfers go out in rounds over the columns (a phase's weights, inputs,
outputs): a channel holds 4 queued tasks, and the oldest one that a fifth waits for must not need a later round. A
phase's inputs wait for the drains before them (``Sequence.sync``: its producers' outputs are in DDR), its weights go
out before that wait (they stream into the memtile meanwhile). A column-major image (``image.QwenImage``) streams a
column's weights between two attentions (their caches on A) as one run, its phases' norm-weight objects in it; a
region-major one per phase and K group (the norm-weight objects from NORM).
"""

import numpy as np
import torch

from ... import txn
from ...gguf_model import GGUFFile
from ...model_config import QwenConfig
from ..design import COLS
from ..overlay import phases as mp
from ..overlay.decode import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_MSG,
    F_NORM,
    F_RESID,
    M_FLAGS,
    M_NASKIP,
    M_NREL,
    M_NS,
    M_SKIP,
    META,
    NX_PER_NORM,
    OBJ_K,
    SLOTS,
    XB,
    XPAY,
    interleave,
)
from ..overlay.model import ModelOverlay
from ..overlay.runtime import Sequence, Transaction, WeightBuffer
from ..overlay.vring import substeps, tails
from . import attn, gdn
from .decode import (
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

FLAGS = {"gdn_in": F_NORM | F_MSG, "attn_in": F_NORM, "gdn_out": F_RESID | F_EMITR, "attn_out": F_RESID | F_EMITR,
         "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM}
KV_PAD = 2048 - 256  # positions past the caches (the last layer's): the prefill's sub-chunk slices
CS_OFF = 4 * META + 640  # GHDR: the conv state's bytes in a column's slot (prefill.CS_OFF)
HSLOT = 12288  # GHDR: a column's slot (the prefill's conv-state drain writes 8 KB from CS_OFF)
DIMS_OBJ = (XB // 8, 2, XB // 8, XPAY // 4)  # X objects of XB bytes, XPAY apart (the overlapping regions)


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


def _text_rope(n: int) -> np.ndarray:
    return np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1)


class QwenOverlay(ModelOverlay):
    """A qwen35 GGUF model on the multi-slot decode overlay. ``layers``: the layers to run (default: all; tests run
    slices), ``vocab``: LM rows (default: the vocabulary), ``max_positions``: the KV caches' capacity, ``weights``: the
    image's buffer allocated ahead (its image's layers and vocabulary).

    ``rope`` (max_positions, 3): the M-RoPE position (T, H, W) of each cache slot, which the step's and the prefill's
    rotations take; the slot's index (text from position 0) until a model writes others (an image's tokens, the
    text after it: ``models.qwen35``), again after ``reset``."""

    CONFIG, IMAGE = QwenConfig, QwenImage

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, weights: WeightBuffer | None = None):
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights)
        cfg, img = self.cfg, self.image
        att = [cfg.attention[L] for L in self.layers]
        self.gsh = gdn.GdnShape(cfg) if not all(att) else None
        self.ash = attn.AttnShape(cfg, max_positions, a_bytes=img.tile_bytes) if any(att) else None
        self.max_positions = self.ash.max_positions if self.ash else max_positions
        self.rope = _text_rope(self.max_positions)
        self.period = 4 * attn.POS_PER_GROUP  # (the steps' sub-steps repeat from group 3 on: ``subs``)
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        self.op = QwenDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=img.tile_bytes,
                                      eps=cfg.rms_norm_eps, gdn=self.gsh, attn=self.ash)
        self._conv_in_ghdr = False
        self._cvi: tuple[np.ndarray, np.ndarray] | None = None
        self._committed, self._accepted = True, True
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _flags(self, L: int, name: str) -> int:
        return FLAGS[name]

    def _phases(self) -> list[tuple[int, str]]:
        cfg = self.cfg
        out = [(L, name) for L in self.layers for name in
               (("attn_in", "attn_out") if cfg.attention[L] else ("gdn_in", "gdn_out")) + ("gate_up", "down")]
        return [*out, (-1, "lm")]

    def _input(self, L: int) -> tuple[int, str]:
        """The phase that reads layer L's input (L = -1: the LM head)."""
        return (L, "lm") if L < 0 else (L, "attn_in" if self.cfg.attention[L] else "gdn_in")

    def _plan(self) -> None:
        """Byte offsets in ACT, NORM, GOBJ, GSTATE, KV."""
        cfg = self.cfg
        self._layout = dict.fromkeys(("ACT", "NORM", "GOBJ", "GSTATE", "KV", "GHDR"), 0)
        take = self._take
        self.loadr = take("ACT", COLS * SLOTS * XB)  # [column][slot]
        for L, name in self._phases():  # (gdn_out: the GDN pads' sink past each x)
            self._add_inputs((L, name), extra=2 * SLOTS if name == "gdn_out" else 0)
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
            self.gcol = 256 + XPAY * (5 + 3 * self.npair)
            self.ps = self.gsh.hc * 16 * gdn.PANEL
            take("GSTATE", 256)  # (the first panel object's header bytes)
        for L in self.layers:
            if cfg.attention[L]:  # ([column][F_ATT per sub-step], then [column][q objects SLOTS])
                self.fatt[L] = take("ACT", COLS * (attn.U + SLOTS) * XB)
                self.rawkv[L] = take("ACT", SLOTS * XB)
                self.kv_off[L] = take("KV", self.ash.kvh * self.ash.head_bytes)
            else:
                self.gobj[L] = take("GOBJ", COLS * self.gcol)
                self.gstate[L] = take("GSTATE", COLS * self.ps)
                self.ghdr[L] = take("GHDR", COLS * HSLOT)
        for L in self.kv_off:  # (per KV head two slots: ``ring``)
            self.ring_off[L] = take("KV", self.ash.kvh * 2 * self.T)
        if self.ash is not None:  # (the prefill's slices past the end; a step's second record; a tail fill's T bytes)
            take("KV", (KV_PAD // attn.POS_PER_GROUP + 1) * 3 * self.T)
        self._sizes()

    def ring(self, L: int, h: int) -> int:
        """Byte offset in KV of KV head h's ring of attention layer L: 2 slots of T bytes, slot r & 1 the V values of
        record r ([d-block][position 8][8] bf16: ``overlay.vring``)."""
        return self.ring_off[L] + 2 * h * self.T

    def _fatt(self, L: int, c: int, u: int) -> int:
        """ACT offset of column c's F_ATT object of sub-step u of layer L."""
        return self.fatt[L] + (c * attn.U + u) * XB

    def _q(self, L: int, c: int, s: int) -> int:
        """ACT offset of column c's q object of slot s of layer L."""
        return self.fatt[L] + (COLS * attn.U + c * SLOTS + s) * XB

    def _gpay(self, L: int, c: int, k: int) -> int:
        """GOBJ offset of layer L's column c's object k's payload (k: conv q / k 0 .. 3, K 4, pair p 5 + 3 p ..)."""
        return self.gobj[L] + c * self.gcol + 256 + XPAY * k

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The norm weights (NORM), the attention layers' F_ATT objects."""
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
        norms = np.concatenate([self._f32(f"blk.{L}.attn_q_norm.weight"), self._f32(f"blk.{L}.attn_k_norm.weight")])
        for c in range(COLS):
            for u in range(attn.U):
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
        for c in range(COLS):
            vh = gdn_v_heads(cfg, c)
            rows = blk.rows[c][blk.rows[c] >= 0]
            o = self.gobj[L] + c * self.gcol
            gobj[o:o + self.gcol] = 0
            gobj[o:o + 4 * META] = m.view(np.uint8)
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

    def _x_at(self, phase: tuple[int, str], e: int) -> tuple[int, int]:
        """(ACT offset of input element e's k block in a phase's x, the bytes left in its K group's x)."""
        for rg in self.inp[phase]:
            if rg.k0 <= e < rg.k0 + rg.K:
                return rg.off + XB + 8 * (e - rg.k0), 8 * (rg.k0 + rg.K - e)
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
            self._inputs(seq, rg, S)
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
        dev, MG, sh = self.dev, gdn, self.gsh
        self._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "gdn_in")][0], S)
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
                at, left = self._x_at((L, "gdn_out"), gdn.D * h0)
                xg = next(8 * rg.K for rg in self.inp[(L, "gdn_out")] if rg.k0 <= gdn.D * h0 < rg.k0 + rg.K)
                assert left >= 2048
                seq.drain("D", c, dev.address("ACT", at), MG.D_BYTES, (512, 2, xg // 4, 1024))
        assert sh.ch == (4 + sh.hc) * gdn.D

    def _runs(self, lm: bool) -> list[list[tuple[int, str]]]:
        """A column-major step's weight runs: its phases up to the first attention's attn_in, then from each
        attention's attn_out up to the next one's attn_in (or the end)."""
        out: list[list[tuple[int, str]]] = [[]]
        for phase in self._phases():
            if phase[1] == "lm" and not lm:
                break
            if phase[1] == "attn_out":
                out.append([])
            out[-1].append(phase)
        return [r for r in out if r]

    def _attention(self, seq: Sequence, st: _Step, L: int, S: int, run: list[tuple[int, str]] | None = None) -> None:
        """attn_in and the attention, per sub-step (each after the last one's records are in DDR: it reads them);
        ``run``: the weights after it (a column-major image), after the last sub-step's cache reads."""
        dev, ash, T = self.dev, self.ash, self.T
        qc, nph = ash.qc, ash.nph
        per = ash.kvh * attn.HD // COLS  # (K and V rows of a column: its part of KV head per c // HD)
        self._weights(seq, (L, "attn_in"), 0)
        seq.sync()
        self._inputs(seq, self.inp[(L, "attn_in")][0], S)
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
            for c in range(COLS):  # the cache of the column's KV head (address, length: _set_position), its ring
                kinds = ("all",) if not split else ("run", "new") if c % nph == 0 else ("run",)
                for kind in kinds:  # (split: the chain's first column its run, then the new positions' for their records)
                    at = seq.fill("A", c, dev.address("KV", self.kv_off[L] + c // nph * ash.head_bytes), 3 * T)
                    st.fills.append((at, L, c, u, kind))
                if not split or ash.role(c) == attn.ROLE_V:  # (a column that attends to the newest group: the tails)
                    st.tails += [(seq.fill("A", c, dev.address("KV", self.ring(L, c // nph)), T), L, c, u, t)
                                 for t in range(tails(n))]
            if u == len(st.subs) - 1 and run:
                self._weight_run(seq, run)
            if u == 0:
                seq.sync()  # (attn_in's q, K, V in DDR)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self._fatt(L, c, u)), XB)
                if split:  # (the q objects of the chain's columns, in order)
                    seq.fill("X", c, dev.address("ACT", self._q(L, c - c % nph, a)), nph * n * XB,
                             (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
                else:
                    seq.fill("X", c, dev.address("ACT", self._q(L, c, a)), n * XB)
                seq.fill("X", c, dev.address("ACT", self.rawkv[L] + a * XB), n * XB)
            for c in range(COLS):  # o (the column's heads in attn_out's x at the sub-step's slots; split: the chain's
                # last column's of the chain's), then the records (addresses: _set_position)
                role = ash.role(c)
                if not split or role == attn.ROLE_V:
                    h0, nh = (qc * (c - nph + 1), nph * qc) if split else (qc * c, qc)
                    at, left = self._x_at((L, "attn_out"), attn.HD * h0)
                    assert left >= 2048 * nh
                    seq.drain("C", c, dev.address("ACT", at + 16 * a), 16 * n * 32 * nh,
                              None if n == SLOTS else (4 * n, 32 * nh, 16, 16 * 32 * nh))
                sizes = {attn.ROLE_K: (attn.REC_K,) * 2, attn.ROLE_V: (attn.REC_V, attn.ROWS_B) * 2, attn.ROLE_O: ()}[role]
                for k, nb in enumerate(sizes):
                    hd = seq.drain("C", c, dev.address("KV", self.kv_off[L]), nb)
                    st.drains.append((hd, L, c, u, k))

    def _build(self, S: int, lm: bool, subs: tuple[tuple[int, int, bool], ...]) -> _Step:
        """The transaction of a step of S slots (``lm``: with the LM head; ``subs``: the attention's sub-steps)."""
        dev, cfg = self.dev, self.cfg
        seq = Sequence(self.op, rolling=True, inputs=("X",))
        st = _Step(S, subs)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * SLOTS * XB), S * XB)
        runs = self._runs(lm) if self.image.column_major else []
        if runs:
            self._weight_run(seq, runs[0])
        for i, L in enumerate(self.layers):
            if cfg.attention[L]:
                run = next((r for r in runs if r[0] == (L, "attn_out")), None)
                self._attention(seq, st, L, S, run)
                self._phase(seq, (L, "attn_out"), S, (L, "gate_up"))
            else:
                self._gdn(seq, L, S)
                self._phase(seq, (L, "gdn_out"), S, (L, "gate_up"))
            self._phase(seq, (L, "gate_up"), S, (L, "down"))
            nxt = self._input(self.layers[i + 1]) if i + 1 < len(self.layers) else (-1, "lm")
            self._phase(seq, (L, "down"), S, nxt)
        if lm:
            self._phase(seq, (-1, "lm"), S, "LOGITS")
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        return st

    def subs(self, p: int, S: int) -> tuple[tuple[int, int, bool], ...]:
        """The attention's sub-steps of a step at p .. p + S - 1 (``overlay.vring.substeps``): (first slot, slots,
        whether the chains divide the cache: by its group alone)."""
        if self.ash is None:
            return ()
        return tuple((a, n, attn.AttnShape.splits((p + a) // attn.POS_PER_GROUP))
                     for a, n in substeps(p, S, attn.POS_PER_GROUP, attn.SUB))

    def _set_position(self, st: _Step, p: int) -> None:
        """The F_ATT objects (per sub-step: its positions, cache objects, rotations) and the step's cache and ring
        transfers."""
        if self.ash is None:
            return
        S = st.S
        assert 0 <= p and p + S <= self.max_positions, (p, S)
        assert st.subs == self.subs(p, S), (p, S, st.subs)
        dev, ash, T = self.dev, self.ash, self.T
        nph = ash.nph
        act = dev.view("ACT")
        cs = np.zeros((SLOTS, 64), np.float32)
        for s in range(S):
            cs[s] = np.concatenate(attn.rope_cs(self.cfg, self.rope[p + s]))
        groups, rec_mode = [], {}
        for L in self.fatt:
            rec_mode[L] = L in self.vring.from_prefill
            if not rec_mode[L]:
                self.vring.check(L, p, p)
        for u, (a, n, split) in enumerate(st.subs):
            q, g = p + a, (p + a) // attn.POS_PER_GROUP
            G = g + 1  # (groups through g; split: chain core k's attention groups b[k] .. b[k + 1] - 1)
            b = ash.bounds(g) if split else [0] * (nph + 1)
            groups.append({(c, "all"): (0, G) for c in range(COLS)} | {(c, "new"): (g, g + 1) for c in range(COLS)}
                          | {(c, "run"): (b[c % nph], b[c % nph + 1]) for c in range(COLS)})
            for L in self.fatt:
                for c in range(COLS):
                    o = self._fatt(L, c, u)
                    m = act[o:o + 4 * META].view(np.int32)
                    k = c % nph
                    nt = tails(n) if not split or ash.role(c) == attn.ROLE_V else 0
                    m[M_POS], m[M_NS], m[M_SA], m[M_SKIP], m[M_SPLIT] = q, n, a, (nph + 1 if split else 2) * n, int(split)
                    m[M_NREL], m[M_NTAIL], m[M_ATREC] = ash.n_rel(c, split, n), nt, int(rec_mode[L] and u == 0)
                    if not split:
                        m[M_NASKIP], m[M_G0], m[M_GATT], m[M_GJ] = 3 * G + nt, 0, 1 << 20, 0
                    elif k == 0:  # (then group g for its K records)
                        m[M_NASKIP], m[M_G0], m[M_GATT], m[M_GJ] = 3 * (b[1] + 1), 0, b[1], g
                    else:
                        m[M_NASKIP], m[M_G0], m[M_GATT], m[M_GJ] = 3 * (b[k + 1] - b[k]) + nt, b[k], 1 << 20, 0
                    q0 = o + 4 * META + 4 * 2 * attn.HD
                    act[q0:q0 + cs.nbytes] = cs.view(np.uint8).reshape(-1)
        for L in self.fatt:
            dev.flush("ACT", self.fatt[L], COLS * attn.U * XB)
        w = st.tr.words
        for at, L, c, u, kind in st.fills:
            g0, g1 = groups[u][(c, kind)]
            w[at:at + 3] = txn.bd_words(dev.address("KV", self.kv_off[L] + c // nph * ash.head_bytes + 3 * g0 * T),
                                        3 * (g1 - g0) * T)
        for at, L, c, u, t in st.tails:  # (the ring slots of the sub-step's first and last records, or the first's V
            a, n, _ = st.subs[u]  # panels in record mode)
            r = (p + a + (n - 1 if t else 0)) >> 3
            if rec_mode[L] and u == 0 and t == 0:
                off = self.kv_off[L] + c // nph * ash.head_bytes + 3 * (r // 4) * T + 2 * T + r % 4 * attn.REC_V
            else:
                off = self.ring(L, c // nph) + (r & 1) * T
            w[at:at + 3] = txn.bd_words(dev.address("KV", off), T)
        for hd, L, c, u, k in st.drains:
            a, n, _ = st.subs[u]
            v = ash.role(c) == attn.ROLE_V
            r = (p + a + (n - 1 if (k >= 2 if v else k) else 0)) >> 3
            g, j = r // 4, r % 4
            head = self.kv_off[L] + c // nph * ash.head_bytes + 3 * g * T
            if not v:
                off, nb = head + j // 2 * T + j % 2 * attn.REC_K, attn.REC_K
            elif k % 2 == 0:
                off, nb = head + 2 * T + j * attn.REC_V, attn.REC_V
            else:
                off, nb = self.ring(L, c // nph) + (r & 1) * T, attn.ROWS_B
            w[hd[0]:hd[0] + 3] = txn.bd_words(dev.address("KV", off), nb)
        st.tr.flush()

    def _set_gdn(self, *, nrep: int, cshift: int, commit: bool) -> None:
        """The GDN layers' gdn_in metadata words of the lazy commit."""
        act = self.dev.view("ACT")
        for L in self.gobj:
            off = self.inp[(L, "gdn_in")][0].off
            m = act[off:off + 4 * META].view(np.int32)
            m[M_NREP], m[M_CSHIFT], m[M_COMMIT] = nrep, cshift, int(commit)
            self.dev.flush("ACT", off, 4 * META)

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
        assert 1 <= S <= SLOTS
        assert self._committed or self._accepted, "accept() the previous lazy step's tokens first"
        self._conv_for_decode()
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._set_gdn(nrep=self.nrep, cshift=self.cshift, commit=commit)
        self._committed = commit
        xf = x.numpy()
        self._write_loadr(xf)
        self._write_input(self.inp[self._input(self.layers[0])][0], interleave(xf).view(np.uint8).reshape(-1))
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
        self._conv_in_ghdr = False
        self.rope[:] = _text_rope(self.max_positions)

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

    def hidden(self, L: int = -1, slot: int = 0) -> torch.Tensor:
        """The residual stream that layer L (-1: the LM head) read in the last step, of a slot (bf16 values, fp32)."""
        off = self.inp[self._input(L)][0].off + XB
        self.dev.fetch("ACT", off, 8 * self.H)
        v = self.dev.view("ACT")[off:off + 8 * self.H].view(np.uint16).reshape(-1, SLOTS, 8)[:, slot].reshape(-1)
        return torch.from_numpy(v.copy()).view(torch.bfloat16).float()


__all__ = ["FLAGS", "F_EMITR", "F_GU", "F_MSG", "F_RESID", "QwenOverlay"]
