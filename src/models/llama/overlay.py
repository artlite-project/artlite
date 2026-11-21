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

A PARO checkpoint (``llama.image``'s regions; the decode design's "Rotated inputs"): the step starts with rot_in (the
embeddings' rotated copies for the first layer's q, k, v, each with the columns' partial sums of squares), then per layer
q, k, v (F_PSQ: each on its own rotated input), att (o into rot_o's input), rot_o (each column rotates o of its heads into
o's input), o (its residual's copies for gate and up), gate and up (F_PSQ; their rows into rot_down's input, [gate SP |
up SP] per k block), rot_down (each column's act(gate) up, rotated, into down's input), down (the next layer's q, k, v
copies; the last layer's residual as it is, for the LM head).

The KV caches (KV): per layer and KV head linear groups of kp + 1 objects of the tile size T (``decode.kv``: kp K
objects, the V object; kp = 2, K as bfp16 hi + lo, or 1, single bfp16: ``kv_k``), 128 positions per group at
head_dim 64, 64 at 128 for objects of ~9.3 KB; then per layer and KV head a ring of two slots of T bytes with the
exact V values of the open records (``decode.vring``). Phase inputs are packed (``decode.protocol.packed_slots``).

A hidden size half a chunk short of whole K chunks (the Qwen3-4B family's 2560) makes the phases that read the residual
stream (qkv, gate_up, the LM head) half regions (``llama.image``): their inputs are the hidden size, the helper copies
the last half chunk's x panels into its half of the chunk (``decode.design``'s ``tails``), whose tiles hold zero
super-blocks there or a pair's second row tile (``tail_pairs``). A hidden size of another part of a chunk would pad
those phases' K with zero super-blocks (``decode.image``); their inputs' padding k must be zero too, since the helper's
norm squares every input element: a step of more slots writes a layout whose real k reach into the padding of the
fewer-slot layouts, which the next step of fewer slots clears first (``_clear_pads``).

The step's transaction is built once per S and pattern of sub-steps (and LM head or not) and patched per step: the
token objects, each column's range of each layer's cache per sub-step (A: address, length), its tail objects (A: the
ring slots of the sub-step's open records, or after a prefill the V panels of the first: record mode) and the record
drains (D: address, length). A column's weights between two attentions are one run of the (column-major) image: one A
task.

Several sequences (``sequences``: each one's caches and rings in a KV buffer of its own, ``ModelOverlay.select``): a
step's slots may be theirs (``step``'s parts), a sequence's slots sub-steps of their own. Every slot's logits are
those of a step of one of its sequence, bit for bit. Four sequences of one token take 15.0 / 39.5 / 86.7 ms where a
step of one takes 12.6 / 34.1 / 79.5 (Llama 3.2 1B / 3B, 3.1 8B at 1K positions): 3.4x - 3.7x the tokens per second;
at 16K positions 35.1 / 110.7 ms against 17.7 / 51.8 (1B / 3B: 2.0x / 1.9x), each sequence's own pass over its caches.
"""

import math

import numpy as np

from artlite.decode import gemv
from artlite.decode import phases as mp
from artlite.decode.attn import Attn, rope_tables, score_bits
from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.design import ROT_GROUP
from artlite.decode.model import InputRegion, paged_formats
from artlite.decode.protocol import (
    C_LEN,
    DIMS_OBJ,
    F_EMITR,
    F_GU,
    F_MAIN,
    F_NORM,
    F_PSQ,
    F_RACT,
    F_RESID,
    F_ROTE,
    F_ROTX,
    M_FLAGS,
    M_NRA,
    M_NRC,
    M_NS,
    M_NT,
    M_NX,
    M_RGN,
    META,
    SLOTS,
    XB,
    XPAY,
    interleave,
    packed_slots,
    region_bytes,
    rot_input_fill,
    rot_region_bytes,
)
from artlite.decode.screen import Screen
from artlite.decode.step import Part, StepPlan
from artlite.decode.vring import PAIR
from artlite.gguf_model import GGUFFile
from artlite.model_config import LlamaConfig
from artlite.operators.design import COLS
from artlite.overlay import Sequence, WeightBuffer

from .decode import LlamaDecodeOperator
from .image import LlamaImage

PHASES = ("qkv", "att", "o", "gate_up", "down")
FLAGS = {"qkv": F_NORM, "o": F_RESID | F_EMITR, "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM}
# (a PARO checkpoint: module docstring) the phases of a layer, their flags (the last layer's down: F_ROTE off), each
# rotation's data region (``llama.image``) and the phases its copies go to
PARO_PHASES = ("q", "k", "v", "att", "rot_o", "o", "gate", "up", "rot_down", "down")
PARO_FLAGS = {"rot_in": F_EMITR | F_ROTE, "q": F_PSQ, "k": F_PSQ, "v": F_PSQ, "rot_o": F_ROTX,
              "o": F_RESID | F_EMITR | F_ROTE, "gate": F_PSQ, "up": F_PSQ, "rot_down": F_ROTX | F_RACT,
              "down": F_RESID | F_EMITR | F_ROTE, "lm": F_NORM}
ROT_DATA = {"rot_in": "rot_in", "rot_o": "rot_o", "o": "o_rot", "rot_down": "rot_down", "down": "down_rot"}
KV_PAD = 4096  # positions past the caches (the last layer's): the prefill's slices of a cache (its largest chunk's
# runtime offsets range over the caches and this pad, the range every chunk size's parameter takes: ``prefill.MAX_T``)


def check_formats(image: LlamaImage) -> set[str]:
    """The image's tile formats; NotImplementedError for those the decode does not read (yet)."""
    fmts = image.formats()
    missing = sorted(f for f in fmts if not gemv.supported(f, image.nsb))
    if missing:
        raise NotImplementedError(f"{', '.join(missing)} tiles: the decode GEMV has no kernels for them")
    return fmts


class LlamaOverlay(AttentionOverlay):
    """A Llama GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices),
    ``vocab``: LM rows (default: the padded vocabulary), ``max_positions``: the caches' capacity, ``kv_k``: the caches'
    K precision ("single": one bfp16 block per 8 dims, "hilo": hi + lo blocks), ``weights``: the image's buffer
    allocated ahead, ``image``: an image laid out already (``ModelOverlay``'s; default ``LlamaImage``'s layout),
    ``sequences``: the sequences whose caches the device holds (``ModelOverlay.select``). One cache kind, one column
    per KV head, one ring state for every layer (key 0 of ``vring``: every step runs every layer)."""

    CONFIG, IMAGE = LlamaConfig, LlamaImage
    AHEAD = True
    SEQ = ("KV",)
    NORMED = frozenset({"qkv", "gate_up", "lm"})

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "single", weights: WeightBuffer | None = None,
                 image: LlamaImage | None = None, sequences: int = 1):
        super().__init__(gguf, layers=layers, vocab=vocab, kv_k=kv_k, weights=weights, image=image, sequences=sequences)
        cfg, img = self.cfg, self.image
        fmts = check_formats(img)
        self.hd, self.nh = cfg.head_dim, cfg.q_per_kv
        self.q_heads = self.nh
        self.at = at = Attn(hds=(self.hd,), rot=(self.hd // 2,), chain=1, heads=self.nh, kp=self.kp,
                            qk_norm=cfg.qk_norm, qk_weights=cfg.qk_norm, eps=cfg.rms_norm_eps, a_bytes=self.T)
        gu = {R for L in self.layers for b in img.region(L, "gate_up").blocks for R in (b.R, b.R2) if R} \
            if not img.paro else {16}
        self.paro = img.paro
        if self.paro:  # (gate and up: phases of their own)
            assert self.H % 1024 == 0 and self.H // COLS % ROT_GROUP == 0, \
                f"a PARO checkpoint of hidden size {self.H}: whole K chunks and rotation groups per column"
        self.op = LlamaDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                      eps=cfg.rms_norm_eps, attn=at, gu_win=gu != {16} or img.split,
                                      paged=paged_formats(gguf, fmts), split=img.split,
                                      tails=any(r.half for r in img.regions),
                                      tail_pairs=any(b.pt for r in img.regions for b in r.blocks),
                                      **({"rot": True} if self.paro else {}))  # (the GGUF designs' keys as they were)
        self.pg = at.pg[self.hd]  # positions per cache group
        groups = -(-max_positions // self.pg)
        self.max_positions = groups * self.pg
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=groups),)
        self.uses = {L: CacheUse(kind=0, source=L, ring=0) for L in self.layers}
        self.period = PAIR
        self.inv_freq = cfg.rope_inv_freq()
        if img.screen:
            norm = np.asarray(gguf.tensor("output_norm.weight").raw, np.float32).reshape(-1)
            self.screen = Screen(gguf, img.lm, norm, cfg.rms_norm_eps, min(self.vocab, cfg.vocab_size))
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order (``lm`` False: without the LM head)."""
        if self.paro:
            return [(self.layers[0], "rot_in")] + [(L, n) for L in self.layers for n in PARO_PHASES] + \
                ([(-1, "lm")] if lm else [])
        return [(L, n) for L in self.layers for n in PHASES] + ([(-1, "lm")] if lm else [])

    def _flags(self, L: int, name: str) -> int:
        if self.paro:
            last = L == self.layers[-1] and name == "down"
            return PARO_FLAGS[name] & ~(F_ROTE if last else 0)
        return FLAGS[name]

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        # (token t: of the sub-step from slot t on; two copies, one per transaction buffer: ``AHEAD``)
        self.toks = (take("ACT", SLOTS * XB), take("ACT", SLOTS * XB))
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name == "att":
                continue
            if self.paro and name in ("rot_in", "q", "k", "v", "rot_o", "gate", "up", "rot_down"):
                self._add_paro_inputs((L, name))
            else:
                self._add_inputs((L, name))
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token a)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in self.layers for h in range(self.kv_heads)}
        # (past the last layer's caches: the prefill's slices)
        self._plan_caches(self.layers, pad=2 * KV_PAD // self.pg * self.G * self.T)
        self._sizes()

    def _add_paro_inputs(self, phase: tuple[int, str]) -> None:
        """(PARO) The input region of a rotated phase (module docstring): rot_in's metadata object; an F_PSQ phase's
        [metadata object | partial sums object | x] (one K group); rot_o's (o of every head) and rot_down's ([gate | up]
        per k block: x of twice the slots) [metadata object | x]."""
        L, name = phase
        if name == "rot_in":
            regs = [InputRegion(self._take("ACT", XB), 0, 0)]
        elif name == "rot_o":
            K = self.nh * self.hd * COLS
            regs = [InputRegion(self._take("ACT", region_bytes(K)), 0, K)]
        elif name == "rot_down":
            K = self.cfg.intermediate_size
            regs = [InputRegion(self._take("ACT", region_bytes(K, 2 * SLOTS)), 0, K)]
        else:
            r = self.image.region(L, name)
            assert len(mp.group_chunks(r)) == 1, f"{name}: one K group"
            regs = [InputRegion(self._take("ACT", rot_region_bytes(r.K)), 0, r.K, x0=XB + XPAY)]
        self.inp[phase] = regs

    def _rot_objects(self, L: int, name: str) -> int:
        """(PARO) The rotation objects of a phase per column (its data region's: ``ROT_DATA``), 0 without."""
        r = ROT_DATA.get(name)
        return self.image.rot_groups(L, r) if r is not None and (L, r) in self.image._index else 0

    def _rot_slice(self, name: str) -> tuple[int, int]:
        """(PARO, rot_o / rot_down) A column's slice of the input: (channels, x's slots per k block for S = 4)."""
        return (self.nh * self.hd, 1) if name == "rot_o" else (self.cfg.intermediate_size // COLS, 2)

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        if not self.paro or name in ("lm",):
            return super()._groups(L, name, S)
        flags, per = self._flags(L, name), 4 if S == 1 else (2 if S == 2 else 1)  # (per: strips of an input object)
        nrot = self._rot_objects(L, name)
        if name in ("rot_in", "rot_o", "rot_down"):  # (no tiles: the rotation objects only)
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS], m[M_NT], m[M_NRA] = F_MAIN | flags, S, nrot, nrot
            if name == "rot_in":
                m[M_NRC] = 3
            else:  # (the column's slice as X objects: counted as the other cores' strips, ``decode.attn``)
                n, mul = self._rot_slice(name)
                m[M_RGN] = n // ROT_GROUP
                m[M_NX] = -(-2 * mul * packed_slots(S) * n // XPAY) * per
            return [mp.Group(0, 0, m, 0, 0)]
        out = super()._groups(L, name, S)
        for i, g in enumerate(out):
            if flags & F_PSQ:  # (the partial sums object counted in: the other cores skip it)
                g.meta[M_NX] += per
            last = i + 1 == len(out)
            if not last:
                g.meta[M_FLAGS] &= ~F_ROTE
            elif flags & F_ROTE:
                g.meta[M_NRA], g.meta[M_NRC] = nrot, 2 if name == "o" else 3
                g.meta[M_NT] += nrot
        return out

    def _gemv(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        if not self.paro:
            return super()._gemv(seq, st, phase)
        name = phase[1]
        if name not in ("k", "v", "up"):  # (k, v, up: their inputs came with q's / gate's, after its sync)
            seq.sync()  # (its inputs' producers)
        sp = packed_slots(st.S)
        for rg in self.inp[phase]:
            if name == "rot_in":  # (the metadata object: the copies come from the residual)
                for c in range(COLS):
                    seq.fill("X", c, self.dev.address("ACT", rg.off), XB)
            elif name in ("rot_o", "rot_down"):  # (the metadata object, then the column's slice)
                n, mul = self._rot_slice(name)
                nobj = -(-2 * mul * sp * n // XPAY)
                for c in range(COLS):
                    seq.fill("X", c, self.dev.address("ACT", rg.off), XB)
                    at = rg.off + XB + 2 * mul * sp * n * c - (XB - XPAY)  # (object i: the XB bytes ending with the
                    # slice's i-th XPAY bytes)
                    seq.fill("X", c, self.dev.address("ACT", at), nobj * XB, DIMS_OBJ)
            elif rg.x0 != XB:  # (F_PSQ: the partial sums object after the metadata object)
                for c in range(COLS):
                    seq.fill("X", c, self.dev.address("ACT", rg.off), *rot_input_fill(rg.K, sp))
            else:
                self._inputs(seq, rg, st.S, phase)
        self._outputs(seq, st, phase)

    def _x_to(self, seq: Sequence, c: int, phase: tuple[int, str], *, e0: int, n: int, S: int, mul: int = 1,
              so: int = 0) -> None:
        """(PARO) n output values per slot of column c (objects (j, slot) of C_LEN values) -> the elements e0 .. e0 + n
        of a phase's x ([k block][slot mul sp][8]: rot_down's [gate | up]; ``so``: from slot so on), per K group."""
        sp = packed_slots(S)
        for rg in self.inp[phase]:
            a, b = max(e0, rg.k0), min(e0 + n, rg.k0 + rg.K)
            j = a
            while j < b:  # (repeated BDs: at most 64 iterations)
                m = min(b - j, 64 * C_LEN)
                assert (j - rg.k0) % C_LEN == 0 and m % C_LEN == 0, (j, rg.k0, m)
                nobj, spx = m // C_LEN, mul * sp
                seq.drain("C", c, self.dev.address("ACT", rg.off + rg.x0 + 2 * spx * (j - rg.k0) + 16 * so), S * 2 * C_LEN,
                          (4, 8, 4 * spx, 4), nobj, (4 * spx * 8, nobj))
                j += m

    def _rot_copies(self, seq: Sequence, c: int, S: int, to: list[tuple[int, str]]) -> None:
        """(PARO) An F_ROTE emission of column c: per copy its partial sums object, then its x (the consumer's)."""
        h8 = self.H // COLS
        for ph in to:
            seq.drain("C", c, self.dev.address("ACT", self.inp[ph][0].off + XB + 2 * C_LEN * c), 2 * C_LEN)
            self._x_to(seq, c, ph, e0=h8 * c, n=h8, S=S)

    def _paro_outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        L, name = phase
        S, dev = st.S, self.dev
        ffn, nq = self.cfg.intermediate_size // COLS, self.nh * self.hd
        i = self.layers.index(L) if L >= 0 else -1
        nxt = self.layers[i + 1] if 0 <= i < len(self.layers) - 1 else None
        for c in range(COLS):
            reg = self.att_reg[(L, c)] if name in ("q", "k", "v") else 0
            if name == "rot_in":
                self._rot_copies(seq, c, S, [(L, n) for n in "qkv"])
            elif name == "q":  # -> its q object per slot
                seq.drain("C", c, dev.address("ACT", reg + 4 * META), S * 2 * nq, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name in ("k", "v"):  # -> the K / V part of its K / V object per slot
                at = reg + SLOTS * XB + 4 * META + (2 * self.hd if name == "v" else 0)
                seq.drain("C", c, dev.address("ACT", at), S * 2 * self.hd, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name == "rot_o":
                self._x_to(seq, c, (L, "o"), e0=nq * c, n=nq, S=S)
            elif name == "o":
                self._rot_copies(seq, c, S, [(L, "gate"), (L, "up")])
            elif name in ("gate", "up"):
                self._x_to(seq, c, (L, "rot_down"), e0=ffn * c, n=ffn, S=S, mul=2, so=packed_slots(S) if name == "up" else 0)
            elif name == "rot_down":
                self._x_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "down" and nxt is not None:
                self._rot_copies(seq, c, S, [(nxt, n) for n in "qkv"])
            elif name == "down":
                self._values_to(seq, c, (-1, "lm"), e0=self.H // COLS * c, n=self.H // COLS, S=S)
            else:
                self._logits_out(seq, c, S)

    def _o_inputs(self, L: int) -> list[InputRegion]:
        return self.inp[(L, "rot_o")] if self.paro else super()._o_inputs(L)

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
        if self.paro:
            return self._paro_outputs(seq, st, phase)
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

    def _rope(self, kind: int, positions: list[int]) -> np.ndarray:
        return rope_tables(self.inv_freq, positions, self.hd // 2)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, position: int | tuple[Part, ...], *, logits: bool = True) -> np.ndarray | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the embeddings at positions position .. position + S
        - 1 of the selected sequence (the positions before are cached), or of several sequences' slots: ``position``
        their parts (``decode.step.Part``: x's rows in the parts' order, each sequence's at its own next positions).
        Returns the logits (S, vocab) bf16 (a view of the NPU's buffer, valid until the next step), or without
        ``logits`` None (only the caches are updated)."""
        x = np.asarray(x, np.float32).reshape(-1, self.H)
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        xf = x
        self._write_loadr(xf)
        if not self.paro:  # (PARO: rot_in's copies of the residual)
            self._write_input(self.inp[(self.layers[0], "qkv")][0], interleave(xf, packed_slots(S)).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def hidden(self, L: int = -1, slot: int = 0) -> np.ndarray:
        """The residual stream that layer L's qkv phase (-1: the LM head) read in the last step, of a slot (its bf16
        copy, as fp32)."""
        return self._read_input(self.inp[(L, "lm" if L < 0 else "qkv")][0], slot, self.H)

    def attn_out(self, L: int, slot: int = 0) -> np.ndarray:
        """The attention's output o (heads hd) of layer L in the last step, of a slot (bf16 values, fp32)."""
        rg = self._o_inputs(L)[0]
        return self._read_input(rg, slot, rg.K)


__all__ = ["KV_PAD", "LlamaOverlay", "check_formats"]
