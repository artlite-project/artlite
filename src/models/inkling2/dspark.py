"""Inkling 2's DSpark drafter (``artlite.model_config.InklingDSparkConfig``; unsloth's ``dspark-*.gguf``, vLLM's
``models/inkling2/nvidia/dspark.py``; the reference: ``tests/inkling2/dspark_ref.py``) on the NPU. It has a separate
design and hardware context (``dspark_image.DSparkDecodeOperator``: 4 slots on fp32 states, the model's tile size) and
keeps its weights in a separate buffer (``dspark_image.DSparkImage``). Its head is the model's LM head plane, read in
place from the model's weight buffer (adopted as "MW"), and it keeps caches for each of the model's sequences.

A step of the drafter's overlay (``DSparkOverlay``) is one of two kinds:

- an injection (``inject``: ``step`` without logits): S <= 4 positions of the model's sequence from their taps (the
  residual entering the model's visits ``cfg.taps``, the last one its final normed state: ``InklingOverlay.tap_rows``,
  ``Screen.input``), mixed per layer on the host (``mixes``: ``dspark_image.mix_weights``; a few K flops a row, beside
  the screen's own work on the same states), the states 1 + L of the state array: per layer "ikv" (normed by its
  ``ctx_norm``: K_ctx, V_ctx) and the attention cores' records of the positions (K RMS-normed by ``attn_k_ctx_norm``
  and rotated; o into a sink);
- a block (``block``: ``step`` with logits): 4 rows [the anchor's normed embedding, the mask rows 0 .. 2] at P .. P + 3
  (state 0: the residual, fp32 in DDR and the helpers' memory), per layer qkv (``attn_norm``), the attention (q / k
  RMS-normed by ``attn_q_norm`` / ``attn_k_norm``, rotated; the block mask: every row the window's positions before P on a
  local layer, all of them on the full one, and the block whole), o (cur += its rows: the F_INK step without a conv),
  gate_up (``ffn_norm``, SiLU), down (three K groups, cur += each), then jh (``output_norm``: ``joint_proj``'s h columns,
  the Markov head's per-row term) and lm (``output_norm``: the model's head plane, its screened logits). The block's
  records at P .. P + 3 stay in the caches until the next injection writes the kept positions over them.

A round's dispatch (``round``) is both: the injection of the positions before P, then the block at P, in one transaction
(``_round_plans``: two step plans, the block's token objects the second set); the state array holds state 0 (the block's
rows, laid out for 4 slots) first, then the injection's states laid out for its own slots (``_state_at``), and the
phases' metadata objects are those of an injection of S slots and the block of 4 (``_meta_writes``).

A block of 4 rows (the trained block is 8: [anchor, mask x 7]) is the drafts a round verifies (3: the model's steps of
4); its drafts 1 .. 3 are those of the block of 8 (S5's measurement: 1.84 vs 1.79 accepted on code, 0.96 vs 0.97 on
chat).

The caches: per layer and KV head linear groups (``decode.kv``), the local layers' a ring of ``WRAP`` groups past that
many groups of capacity (``CacheKind.wrap``), the full layer's the capacity. An injection's attention reads and writes
its layer's cache under a separate layer index (L + ``INJ``: its K / V objects carry ``attn_k_ctx_norm``), using the
same buffers and ring of open records (``kv_base``, ``ring``). Every step is one sub-step (``subs``: the block mask
needs all its rows at once; its cache range to the last position's group, ``ranges``).

The Markov chain (``MarkovChain``) runs on the host over each row's candidates: the top 64 of the block's screened logits
and the top 64 of them plus ``head . b_ho`` (a static bias per token, ``static_bias``), each candidate's logit exact
(unsloth's form: ``Screen.exact`` of h_i + W_ho latent + b_ho, the base logit and the bias in one row product; the
mainline form: ``Screen.exact`` of h_i plus the candidate's ``markov_w2`` row times the latent and its bias); S5's
measurement: the same drafts as the whole vocabulary's (2.18 vs 2.20 accepted of 7)."""

import math
import time
from collections.abc import Callable

import numpy as np
from ml_dtypes import bfloat16

from artlite import _native
from artlite.decode import kv
from artlite.decode import phases as mp
from artlite.decode.attn import Attn, rope_tables, score_bits
from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.model import InputRegion
from artlite.decode.protocol import (
    C_LEN,
    F_GU,
    F_INK,
    F_NORM,
    F_RESID,
    F_STATES,
    M_CSET,
    M_INKN,
    M_NR,
    M_NX,
    META,
    SLOTS,
    XB,
    XPAY,
    packed_slots,
    region_bytes,
)
from artlite.decode.screen import Screen
from artlite.decode.step import Fill, Rec, StepPlan, Tail
from artlite.decode.vring import tails
from artlite.gguf_model import GGUFFile, tensor_f32
from artlite.model_config import InklingDSparkConfig
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction

from .dspark_image import DOWN_K, DSparkDecodeOperator, DSparkImage, ctx_norm, mask_rows, mix_weights

INJ = 10  # an injection's layer index: L + INJ (module docstring)
WRAP = 32  # groups of a local layer's ring past that many groups of capacity (its window's 16, a block's 2 and more)
N_DOWN = 3  # (the down's K groups: ``dspark_image.DOWN_K``)
STATES = 4  # (the state array's states: 0 the block's residual, 1 + L layer L's injection input)
BLOCK = ("qkv", "att", "o", "gate_up", *(f"down{g}" for g in range(N_DOWN)))
FLAGS = {"qkv": F_NORM, "ikv": F_NORM, "gate_up": F_NORM | F_GU, "jh": F_NORM, "lm": F_NORM, "o": F_RESID | F_INK} | \
    {f"down{g}": F_RESID | F_INK for g in range(N_DOWN)}
INK_OBJ = {"o": ("inko", "inkoq")} | {f"down{g}": (f"inkd{g}", f"inkdq{g}") for g in range(N_DOWN)}  # (an F_INK step's)


class DSparkOverlay(AttentionOverlay):
    """The DSpark drafter ``gguf`` of the model on overlay ``target`` (``InklingOverlay``; module docstring): ``load``
    after the target's (its device and weight buffer); ``inject``, ``block`` and ``round`` run steps. ``wrap``: the local
    layers' ring (groups; default ``WRAP``; the model's prefill writes the records of its chunks: its ring,
    ``prefill.InklingPrefill``); ``pad``: positions the caches' buffer extends past them (the prefill's slices)."""

    CONFIG, IMAGE = InklingDSparkConfig, DSparkImage
    CHAIN = 1
    SEQ = ("KV",)
    upto = 0  # (tests: a block's first phases only)
    _injecting = False  # (an injection's step: ``ranges``)

    def __init__(self, target: object, gguf: GGUFFile, *, wrap: int | None = None, pad: int = 0):
        cfg = InklingDSparkConfig.from_gguf(gguf)
        tc = target.cfg
        if ((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim) != (tc.hidden_size, tc.num_attention_heads,
                                                                     tc.num_key_value_heads, tc.head_dim)
                or cfg.vocab_size != tc.vocab_size or cfg.taps[-1] != len(tc.visits)):
            raise ValueError(f"{gguf.path}: the drafter of another model than {target.gguf.path}")
        image = DSparkImage(gguf, cfg, tile_bytes=target.T)
        image.vocab = target.vocab
        self.target = target
        super().__init__(gguf, layers=list(range(cfg.layers)), vocab=None, kv_k="hilo", weights=None, image=image,
                         sequences=target.sequences)
        target.followers.append(self)
        self.hd, self.q_heads = cfg.head_dim, cfg.heads // cfg.kv_heads
        self.at = at = Attn(hds=(self.hd,), rot=(self.hd // 2,), chain=1, heads=self.q_heads, kp=self.kp, qk_norm=True,
                            qk_weights=True, eps=cfg.rms_norm_eps, a_bytes=self.T, states=True, block=True)
        lm = target.image.region(-1, "lm")
        fmts = image.formats() | set(lm.blocks[0].formats)
        R = image.R | target.image.R
        self.op = DSparkDecodeOperator(hidden=self.H, rows={f: R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                       eps=cfg.rms_norm_eps, attn=at)
        self.pg = at.pg[self.hd]
        full = -(-target.max_positions // self.pg)
        self.max_positions = full * self.pg
        guard = -(-cfg.window // self.pg)
        wrap = (WRAP if full > WRAP else 0) if wrap is None else wrap
        assert not wrap or guard + 2 <= wrap < full, (wrap, guard, full)
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=guard + (wrap or full), guard=guard, window=cfg.window,
                                wrap=wrap),
                      CacheKind(hd=self.hd, pg=self.pg, groups=full))
        kinds = {L: 0 if cfg.local[L] else 1 for L in self.layers}
        self.uses = {L: CacheUse(kind=kinds[L], source=L, ring=L) for L in self.layers} | \
            {L + INJ: CacheUse(kind=kinds[L], source=L + INJ, ring=L) for L in self.layers}
        self.period = 1
        self.inv_freq = 1.0 / np.power(np.float64(cfg.rope_base), np.arange(0, self.hd, 2) / self.hd)
        self.alpha = mix_weights(gguf, cfg)  # (the taps' mix per layer, then the Markov head's)
        self.mix_norms = np.stack([tensor_f32(gguf, ctx_norm(cfg, L)) for L in range(cfg.layers + 1)])
        self.mask = mask_rows(gguf, cfg)[:SLOTS - 1]
        self.w_out = tensor_f32(gguf, "output_norm.weight")
        self.screen: Screen | None = None
        if target.screen is not None:  # (the head's exact rows: the model's, against the drafter's output norm)
            self.screen = target.screen
        self.pad = pad
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """A block's phases (``lm``), an injection's (without)."""
        if not lm:
            return [(L + INJ, n) for L in self.layers for n in ("ikv", "att")]
        out = [(L, n) for L in self.layers for n in BLOCK] + [(-1, "jh"), (-1, "lm")]
        return out[:self.upto] if self.upto else out

    def _flags(self, L: int, name: str) -> int:
        return FLAGS[name]

    def _region(self, L: int, name: str) -> object:
        """A phase's image region (an injection's ikv: its layer's; lm: the model's)."""
        if name == "lm":
            return self.target.image.region(-1, "lm")
        return self.image.region(L % INJ if L >= 0 else L, name)

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        return mp.groups(self._region(L, name), self._flags(L, name), S, slots=self.SLOTS)

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        # (the token objects; a round's block after its injection: the second set)
        self.toks = tuple(take("ACT", len(self.kinds) * SLOTS * XB) for _ in range(2))
        self.loadr = take("ACT", COLS * SLOTS * XB)
        self.sa = take("ACT", XB + STATES * self.state_bytes(SLOTS)) + XB  # ([state][k block][slot sp][8] fp32)
        self.meta = {}
        for ph in self.phases(True) + self.phases(False):
            L, name = ph
            if name in ("att", "o") or name.startswith("down"):
                continue
            self.meta[ph] = take("ACT", XB)
        self.att_reg = {(L, h): take("ACT", 2 * SLOTS * XB) for L in self.uses for h in range(self.kv_heads)}
        for L in self.layers:  # (o, down's: plain inputs; an injection's o: a sink)
            self.inp[(L, "o")] = [InputRegion(take("ACT", region_bytes(self.cfg.heads * self.hd)), 0,
                                              self.cfg.heads * self.hd)]
            for g in range(N_DOWN):
                self.inp[(L, f"down{g}")] = [InputRegion(take("ACT", region_bytes(DOWN_K)), 0, DOWN_K)]
        self.sink = InputRegion(take("ACT", region_bytes(self.cfg.heads * self.hd)), 0, self.cfg.heads * self.hd)
        self.zero = take("ACT", XB)  # (an F_INK step's history object: plain, unread)
        self.junk = take("ACT", (3 + SLOTS) * 4 * self.H // COLS + 512)  # (its window's and partial sums' drains)
        self.jh = take("ACT", COLS * SLOTS * 2 * 2 * C_LEN)  # ([column][C object 2][slot] of C_LEN bf16)
        self._plan_caches(self.layers, pad=self.pad // self.pg * self.G * self.T)
        self._sizes()

    def state_bytes(self, sp: int) -> int:
        """A state's bytes in the state array of a step of sp packed slots (``InklingOverlay.state_bytes``)."""
        return -(-4 * sp * self.H // XPAY) * XPAY

    def _state_at(self, i: int, S: int) -> int:
        """State i's offset in ACT (module docstring): state 0 (a block's rows, 4 slots) first, then the injection's
        states 1 + L of a step of S slots."""
        return self.sa if i == 0 else self.sa + self.state_bytes(SLOTS) + (i - 1) * self.state_bytes(packed_slots(S))

    def kv_base(self, L: int, h: int) -> int:
        return super().kv_base(L % INJ, h)

    def ring(self, L: int, h: int) -> int:
        return super().ring(L % INJ, h)

    def kv_bytes(self, L: int) -> int:
        return super().kv_bytes(L % INJ)

    def ranges(self, kind: int, p: int, n: int = 1) -> list[tuple[int, int]]:
        """(the cache range of a sub-step of n slots from p: to its last position's group, ``subs``; an injection's: its
        positions' groups alone, whose records it writes: its attention's output goes nowhere)"""
        k = self.kinds[kind]
        first = p if self._injecting else p - k.window if k.window else 0
        return kv.ranges(first=max(0, first // k.pg), g=(p + n - 1) // k.pg, chain=self.CHAIN)

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        return ((0, S),)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, *, threads: int = 0) -> None:
        """The design in a separate hardware context on the model's device, the image packed, the model's weight buffer
        adopted ("MW"), the static objects written."""
        img, tgt = self.image, self.target
        self.dev = dev = OverlayDevice(self.op, device=tgt.dev.device)
        dev.alloc_all({"W": img.nbytes, **self.sizes,
                       **{self.seq_name(b, s): self.sizes[b] for b in self.SEQ for s in range(1, self.sequences)}})
        dev.alias = self._alias()
        if img.pack(dev.view("W"), threads=threads):
            dev.flush("W")
        dev.adopt("MW", tgt.dev.buffers["W"])
        act = dev.view("ACT")
        act[:] = 0
        for c in range(COLS):
            for s in range(SLOTS):
                act[self.loadr + (c * SLOTS + s) * XB:][:4 * META] = mp.loadr_meta(s).view(np.uint8)
        self._load()
        dev.flush("ACT")
        if (store := self.op.page_store()) is not None:
            dev.set_prologue(self.op.page_prologue(store))
        self.gguf.release()
        for s in reversed(range(self.sequences)):
            AttentionOverlay.select(self, s)
            self.reset()
        AttentionOverlay.select(self, tgt.seq)
        self.static_bias()

    def _load(self) -> None:
        """(the token objects' and both kinds of attention objects' words)"""
        super()._load()
        for L in self.layers:
            self._load_attention(L + INJ)

    def _attention_scale(self, L: int) -> tuple[int, int]:
        return score_bits(1 / math.sqrt(self.hd))

    def _qk_weights(self, L: int) -> dict[str, np.ndarray]:
        """(a block's q_norm, k_norm; an injection's k: attn_k_ctx_norm)"""
        b = f"blk.{L % INJ}."
        k = b + ("attn_k_ctx_norm.weight" if L >= INJ else "attn_k_norm.weight")
        return {"q": tensor_f32(self.gguf, b + "attn_q_norm.weight").view(np.uint8),
                "k": tensor_f32(self.gguf, k).view(np.uint8)}

    def reset(self) -> None:
        super().reset()
        self.next = 0  # (the next position to inject)

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """(an injection of S slots and the block of 4: the GEMV phases on states: one state, coefficient 1; o, down's:
        F_INK, plain)"""
        out = []
        for (L, name), off in self.meta.items():
            n = S if name == "ikv" else SLOTS
            gr = self._groups(L, name, n)
            assert len(gr) == 1, (L, name)
            m = gr[0].meta.copy()
            m[0] |= F_STATES
            m[M_NR], m[M_NX], m[M_CSET] = 1, self.state_bytes(packed_slots(n)) // XPAY, -1
            out.append((off, m.view(np.uint8)))
        for L in self.layers:
            for name in ("o", *(f"down{g}" for g in range(N_DOWN))):
                rg = self.inp[(L, name)][0]
                gr = self._groups(L, name, SLOTS)
                assert len(gr) == 1
                m = gr[0].meta.copy()
                m[M_INKN] = 1 << 12  # (cur: state 0; one queries object)
                out.append((rg.off, m.view(np.uint8)))
        return out

    def _build(self, S: int, lm: bool, subs: tuple, b: int) -> StepPlan:
        self._lm = lm
        return super()._build(S, lm, subs, b)

    def _round_plans(self, S: int) -> tuple[StepPlan, StepPlan]:
        """The step plans of a round's dispatch (module docstring): an injection of S slots (the first token objects),
        then the block (the second ones), one transaction (built at its first use)."""
        key = ("round", S)
        if key not in self._steps:
            self._ran_keys.discard(key)
            self._steps[key] = self._build_round(S)
        return self._steps[key]

    def _build_round(self, S: int) -> tuple[StepPlan, StepPlan]:
        """(``AttentionOverlay._build`` of the injection's phases and the block's: the block's token objects and helper
        rows once the injection's records are in DDR)"""
        dev = self.dev
        seq = Sequence(self.op, inputs=("X",))
        inj = StepPlan(S, self.subs(0, S), tok=self.toks[0])
        blk = StepPlan(SLOTS, self.subs(0, SLOTS), tok=self.toks[1])
        ph = self.phases(False) + self.phases(True)
        n = len(self.phases(False))

        def run(i: int) -> list[tuple[int, str]]:
            return ph[i:next((k for k in range(i, len(ph)) if ph[k][1] in self.OPS), len(ph))]

        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", inj.tok), self._tok_bytes)
        for i, (L, name) in enumerate(ph):
            st = inj if i < n else blk
            if i == n:
                seq.sync()
                for c in range(COLS):
                    seq.fill("X", c, dev.address("ACT", blk.tok), self._tok_bytes)
                AttentionOverlay._fill_loadr(self, seq, SLOTS)
            if name in self.OPS:
                self._operator(seq, st, L, name, run(i + 1))
                continue
            self._weights_first(seq, ph, i, run)
            self._gemv(seq, st, (L, name))
        seq.sync(fills=True)
        inj.tr = blk.tr = Transaction(dev, seq.words())
        return inj, blk

    def prepare_rounds(self) -> None:
        """The transactions of every round's dispatch, the block's alone and the injections' alone (a round builds them
        at its first use: in the time the drafter's planner measures)."""
        for S in range(1, SLOTS + 1):
            self._round_plans(S)
            self.prepare(S, logits=False)
        self.prepare(SLOTS, logits=True)

    def _fill_loadr(self, seq: Sequence, S: int) -> None:
        """(a block's: its rows as state 0 of the helpers)"""
        if self._lm:
            super()._fill_loadr(seq, S)

    def _weight_run(self, seq: Sequence, phases: list[tuple[int, str]], **kw: object) -> None:
        """(consecutive phases: one run of the image per column, from the first one's norm-weight objects to the last one's
        F_INK objects; lm: its norm-weight objects, then the model's tiles from MW)"""
        img, tgt = self.image, self.target.image
        lm = phases[-1][1] == "lm"
        if lm:
            phases = phases[:-1]
        if phases:
            (L0, n0), (L1, n1) = phases[0], phases[-1]
            L0, L1 = (L0 % INJ if L0 >= 0 else L0), (L1 % INJ if L1 >= 0 else L1)
            r0 = img.region(L0, n0 + "_norm") if (L0, n0 + "_norm") in img._index else img.region(L0, n0)
            r1 = img.region(L1, INK_OBJ.get(n1, (n1,))[-1])
            if lm:
                r1 = img.region(-1, "lm_norm")
            self._fills(seq, "W", img.runs(r0, r1))
        elif lm:
            r = img.region(-1, "lm_norm")
            self._fills(seq, "W", img.runs(r, r))
        if lm:
            r = tgt.region(-1, "lm")
            self._fills(seq, "MW", tgt.runs(r, r))

    def _fills(self, seq: Sequence, buf: str, runs: list[tuple[object, int]]) -> None:
        for r, n in runs:
            for c in range(COLS):
                seq.fill("A", c, self.dev.address(buf, r.offset + c * r.col_stride), n)

    def _weights_first(self, seq: Sequence, ph: list[tuple[int, str]], i: int, run: object) -> None:
        if i == 0:
            self._weight_run(seq, run(0))

    def _gemv(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """A GEMV phase: on states (its metadata object, the state's objects), or o / down's (their input, the F_INK
        step's history object); its outputs."""
        L, name = phase
        dev, S, sp = self.dev, st.S, packed_slots(st.S)
        ST = self.state_bytes(sp)
        seq.sync()
        if phase in self.meta:
            o = self._state_at(1 + L - INJ if name == "ikv" else 0, S)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self.meta[phase]), XB)
                seq.fill("X", c, dev.address("ACT", o - (XB - XPAY)), XB, (XB // 8, 2, XB // 8, ST // 4),
                         repeat=ST // XPAY, iteration=(XPAY // 4, ST // XPAY))
        else:
            self._inputs(seq, self.inp[phase][0], S)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self.zero), XB)
        self._outputs(seq, st, phase)

    def _outputs(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        L, name = phase
        S, dev, sp = st.S, self.dev, packed_slots(st.S)
        h8, nq, ffn = self.H // COLS, self.q_heads * self.hd, self.cfg.ffn // COLS
        row = 4 * h8
        for c in range(COLS):
            if name == "qkv":  # [q nh hd | K hd | V hd] per slot -> its q object, its K / V object
                reg = self.att_reg[(L, c)]
                seq.drain("C", c, dev.address("ACT", reg + 4 * META), S * 2 * nq, (C_LEN // 2, S, XB // 4, C_LEN // 2))
                seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 4 * META), S * 4 * self.hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name == "ikv":  # [K hd | V hd] per slot -> the injection's K / V object
                reg = self.att_reg[(L, c)]
                seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 4 * META), S * 4 * self.hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name == "o" or name.startswith("down"):  # (F_INK: the window, cur's slice, the partial sums)
                seq.drain("C", c, dev.address("ACT", self.junk), (3 + S) * row)
                seq.drain("C", c, dev.address("ACT", self.sa + c * row * sp), row * sp)
                seq.drain("C", c, dev.address("ACT", self.junk), 512)
            elif name == "gate_up":
                for g in range(N_DOWN):
                    self._values_to(seq, c, (L, f"down{g}"), e0=ffn * c - DOWN_K * g, n=ffn, S=S)
            elif name == "jh":
                seq.drain("C", c, dev.address("ACT", self.jh + c * SLOTS * 4 * C_LEN), S * 4 * C_LEN)
            else:
                self._logits_out(seq, c, S)

    def _values_to(self, seq: Sequence, c: int, phase: tuple[int, str], *, e0: int, n: int, S: int,
                   stream: str = "C", whole: bool = False) -> None:
        """(a phase's outputs e0 .. e0 + n of column c that fall in its input: the others none)"""
        rg = self.inp[phase][0]
        a, b = max(e0, 0), min(e0 + n, rg.K)
        if a < b:
            sp = packed_slots(S)
            j = a
            while j < b:
                m = min(b - j, 64 * C_LEN)
                nobj = m // C_LEN
                seq.drain(stream, c, self.dev.address("ACT", rg.off + rg.x0 + 2 * sp * j), S * 2 * C_LEN,
                          (4, 8, 4 * sp, 4), nobj, (4 * sp * 8, nobj))
                j += m

    def _o_inputs(self, L: int) -> list[InputRegion]:
        return [self.sink] if L >= INJ else self.inp[(L, "o")]

    def _attention(self, seq: Sequence, st: StepPlan, L: int, weights: list[tuple[int, str]]) -> None:
        """(``AttentionOverlay._attention`` of one sub-step, the injection's layers on their layer's buffers)"""
        dev, T = self.dev, self.T
        use = self.uses[L]
        kind = self.kinds[use.kind]
        (a, n), = st.subs
        sp, nq = packed_slots(st.S), self.q_heads * kind.hd
        for c in range(COLS):
            st.fills += [Fill(seq.fill("A", c, dev.address("KV", self.kv_off[L % INJ]), T), L, c, 0, f)
                         for f in range(2 if kind.wrap else 1)]
            st.tails += [Tail(seq.fill("A", c, dev.address("KV", self.ring_off[L % INJ]), T), L, c, 0, t)
                         for t in range(tails(n))]
        seq.sync()
        for c in range(COLS):
            reg = self.att_reg[(L, c)] + a * XB
            seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
        if weights:
            self._weight_run(seq, weights)
        o = self._o_inputs(L)[0]
        for i in range(len(kv.emit_order(p=0, S=n, hd=kind.hd, a_bytes=T, kp=self.kp))):  # (round-robin over the
            # columns: a wait for a column's oldest task holds back no other column's, ``InklingOverlay._attention``)
            for c in range(COLS):
                st.recs.append(Rec(seq.drain("D", c, dev.address("KV", self.kv_off[L % INJ]), 2 * self.at.d_len),
                                   L, c, 0, i))
        for c in range(COLS):
            at = dev.address("ACT", o.off + o.x0 + 2 * sp * nq * c + 16 * a)
            seq.drain("D", c, at, 2 * n * nq, None if n == sp else (4 * n, nq // 8, 4 * sp, 4 * sp * nq // 8))

    def _rope(self, kind: int, positions: list[int]) -> np.ndarray:
        return rope_tables(self.inv_freq, positions, self.hd // 2)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def write_states(self, x: np.ndarray, first: int) -> None:
        """States first .. first + n - 1 of the state array (n, S, hidden) fp32 (zero past the hidden; ``_state_at``);
        state 0 (a block's rows) also as the helpers' F_LOADR objects."""
        n, S, H = x.shape
        sp, act = packed_slots(S), self.dev.view("ACT")
        ST = self.state_bytes(sp)
        for b in range(n):
            xs = np.zeros((ST // (4 * 8 * sp), sp, 8), np.float32)
            xs[:H // 8, :S] = x[b].reshape(S, H // 8, 8).transpose(1, 0, 2)
            o = self._state_at(first + b, S)
            act[o:o + ST] = xs.reshape(-1).view(np.uint8)
            self.dev.flush("ACT", o, ST)
        if first == 0:
            self._write_loadr(x[0])

    def _rows(self, anchor: np.ndarray) -> np.ndarray:
        """A block's rows (1, 4, hidden) fp32: the anchor's normed embedding (hidden,), then the mask rows."""
        return np.concatenate([np.asarray(anchor, np.float32)[None], self.mask])[None]

    def inject(self, taps: np.ndarray, q: int) -> None:
        """The positions q .. q + S - 1 of the selected sequence from their taps (S, taps, hidden) fp32 (S <= 4)."""
        S = taps.shape[0]
        assert 1 <= S <= SLOTS and q == self.next, (S, q, self.next)
        if self._metas_s != (S, self.start):
            self._metas(S)
        self.write_states(self.mixes(taps)[:len(self.layers)], 1)
        self._injecting = True
        try:
            self._run(S, q, False)
        finally:
            self._injecting = False
        self.next = q + S

    def block(self, anchor: np.ndarray, P: int, *, meanwhile: Callable[[], None] | None = None) -> None:
        """A block at P (the anchor's normed embedding (hidden,) fp32, then the mask rows): its logits and jh.
        ``meanwhile``: host work while it runs (the model's next step patched ahead)."""
        if not self._metas_s[0]:  # (the block's are those of every S: ``_meta_writes``)
            self._metas(SLOTS)
        self.write_states(self._rows(anchor), 0)
        self._run(SLOTS, P, True, meanwhile=meanwhile)

    def round(self, taps: np.ndarray, q: int, anchor: np.ndarray, *,
              meanwhile: Callable[[], None] | None = None) -> None:
        """A round's dispatch (module docstring): the positions q .. q + S - 1 from their taps (S, taps, hidden) fp32
        (S <= 4: ``inject``), then the block at q + S (``block``), in one transaction. ``meanwhile``: host work while
        it runs."""
        S = taps.shape[0]
        P = q + S
        assert 1 <= S <= SLOTS and q == self.next, (S, q, self.next)
        if self._metas_s != (S, self.start):
            self._metas(S)
        self.write_states(self.mixes(taps)[:len(self.layers)], 1)
        self.write_states(self._rows(anchor), 0)
        self._clear_pads(self._sp(SLOTS))
        key = ("round", S)
        before, t = self._once_t, time.perf_counter()
        inj, blk = self._round_plans(S)
        self._injecting = True
        try:
            self._set_position(inj, q)
        finally:
            self._injecting = False
        self._ran(q, S)  # (the block's patch: the rings with the injection's records)
        self._set_position(blk, P)
        # (one-time work: as ``_run``'s)
        self.once_s = before + (time.perf_counter() - t if key not in self._ran_keys else self._once_t - before)
        self._once_t = 0.0
        self._ran_keys.add(key)
        self.ready()
        self.dev.submit(blk.tr)
        try:
            self._ran(P, SLOTS)
            if meanwhile is not None:
                meanwhile()
        finally:
            self.dev.wait()
        self._last_s = SLOTS
        self.next = P

    def mixes(self, taps: np.ndarray) -> np.ndarray:
        """The taps' mixes (layers + 1, S, hidden) fp32: per layer its injection's input (unnormed: its phase norms it),
        then the Markov head's f_rnn (normed by its ``ctx_norm``)."""
        x = np.einsum("lt,sth->lsh", self.alpha, taps.astype(np.float32))
        f = x[-1].astype(np.float64)
        x[-1] = (f / np.sqrt((f * f).mean(-1, keepdims=True) + self.cfg.rms_norm_eps) * self.mix_norms[-1]).astype(np.float32)
        return x

    def h(self) -> np.ndarray:
        """The block's rows normed by ``output_norm`` (4, hidden) fp32 (state 0 after the last down)."""
        sp, ST = SLOTS, self.state_bytes(SLOTS)
        self.dev.fetch("ACT", self.sa, ST)
        x = self.dev.view("ACT")[self.sa:self.sa + ST].view(np.float32).reshape(-1, sp, 8)[:self.H // 8]
        x = x.transpose(1, 0, 2).reshape(sp, self.H).astype(np.float64)
        return (x / np.sqrt((x * x).mean(-1, keepdims=True) + self.cfg.rms_norm_eps) * self.w_out).astype(np.float32)

    def jh_rows(self) -> np.ndarray:
        """The block's jh outputs (4, 3 rank) fp32: ``joint_proj``'s h columns times each row's h."""
        self.dev.fetch("ACT", self.jh, COLS * SLOTS * 4 * C_LEN)
        per = 3 * self.cfg.rank // COLS
        v = self.dev.view("ACT")[self.jh:self.jh + COLS * SLOTS * 4 * C_LEN].view(bfloat16).reshape(COLS, 2, SLOTS, C_LEN)
        v = v.transpose(0, 2, 1, 3).reshape(COLS, SLOTS, 2 * C_LEN)  # (C objects (j, slot): j-major)
        return v[:, :, :per].astype(np.float32).transpose(1, 0, 2).reshape(SLOTS, -1)

    def _logits(self, S: int) -> np.ndarray:
        """(the block's screened logits as the NPU's: ``MarkovChain`` takes its candidates' exact ones)"""
        return self._npu_logits(S)

    def screened(self) -> np.ndarray:
        """The block's screened logits (4, LM rows) bf16 (a view of the NPU's buffer; the head plane's)."""
        return self._npu_logits(SLOTS)

    def static_bias(self) -> None:
        """``head . b_ho`` per token (``MarkovChain``'s candidates): the mainline form's ``markov_w2`` bias, else from
        the head's exact rows."""
        if self.cfg.mainline:
            self.fb = tensor_f32(self.gguf, "markov_w2.bias")
            return
        b = tensor_f32(self.gguf, "dspark_hidden_out.bias")
        self.fb = self.screen.exact(np.arange(self.cfg.vocab_size), b) if self.screen is not None else None


__all__ = ["INJ", "DSparkOverlay"]


class MarkovChain:
    """The drafter's Markov head on the host (module docstring): ``walk`` the drafts of a block (its rows' screened logits,
    normed states and jh outputs), f_rnn the round's last committed position's (``DSparkOverlay.mixes``)."""

    CAND = 64  # candidates per row from each of the two rankings (module docstring)

    def __init__(self, gguf: GGUFFile, cfg: InklingDSparkConfig, screen: Screen, fb: np.ndarray):
        R, H = cfg.rank, cfg.hidden
        t = gguf.tensor("joint_proj.weight")
        wj = np.asarray(t.raw).view(bfloat16).reshape(t.rows, t.cols).astype(np.float32)
        self.ws, self.wp, self.wf = (np.ascontiguousarray(wj[:, :R]), np.ascontiguousarray(wj[:, R:2 * R]),
                                     np.ascontiguousarray(wj[:, 2 * R + H:]))
        self.bj = tensor_f32(gguf, "joint_proj.bias")
        self.who = self.bho = None  # (unsloth's form: the latent's projection into the hidden; mainline: markov_w2)
        if not cfg.mainline:
            t = gguf.tensor("dspark_hidden_out.weight")
            self.who = np.asarray(t.raw).view(bfloat16).reshape(t.rows, t.cols).astype(np.float32)
            self.bho = tensor_f32(gguf, "dspark_hidden_out.bias")
        self.gguf, self.cfg, self.screen, self.fb = gguf, cfg, screen, fb
        self.chains: list[float] = []  # (``walk``: per draft its chain's probability)

    def resident(self) -> None:
        """markov_w1's pages read ahead (a row per draft; markov_w2's: a row per candidate)."""
        self.gguf.willneed("markov_w1.weight")
        if self.cfg.mainline:
            self.gguf.willneed("markov_w2.weight")

    def context(self, f_rnn: np.ndarray) -> np.ndarray:
        """The round's term of every draft (3 rank,) fp32 (the bias and ``joint_proj``'s f columns times f_rnn, the
        round's last committed position's context: ``DSparkOverlay.mixes``), while the block runs. Not a BLAS matmul:
        its threads, asleep while the host waits for the NPU, took up to ~5 ms to wake for it (``vecdot``: 0.25 ms)."""
        return self.bj + np.vecdot(self.wf, f_rnn)

    def walk(self, screened: np.ndarray, h: np.ndarray, jh: np.ndarray, fc: np.ndarray, anchor: int, n: int,
             stop: set[int] | frozenset[int] = frozenset()) -> list[int]:
        """Up to n drafts after the anchor token, ending with a token of ``stop``: screened (rows, vocab) the block's
        screened logits, h (rows, hidden) its rows normed, jh (rows, 3 rank) their jh outputs (row i + 1 guesses draft
        i), fc the round's ``context``; ``chains``: per draft the probability of its chain (each draft's softmax over
        its candidates, softcapped)."""
        R, cap, sc = self.cfg.rank, self.cfg.softcap, self.screen
        nv = len(self.fb)
        ctx = fc + jh[1:n + 1]
        state, prev = np.zeros(R, np.float32), anchor
        out, self.chains, chain = [], [], 1.0
        for i in range(n):
            w1 = self.gguf.rows("markov_w1.weight", [prev], prefetch=False)[0]
            raw = ctx[i] + self.ws @ state + self.wp @ w1
            gate = 1 / (1 + np.exp(-raw[:R]))
            state = (gate * state + (1 - gate) * np.tanh(raw[R:2 * R])).astype(np.float32)
            lat = np.tanh(raw[2 * R:])
            est = screened[i + 1, :nv].astype(np.float32) * sc.scale[:nv]
            c = np.union1d(_native.top_logits(est, self.CAND)[1], _native.top_logits(est + self.fb, self.CAND)[1])
            if self.who is not None:  # (unsloth's form: the latent's projection added to the head's input)
                col = sc.exact(c.astype(np.int64), (h[i + 1] + self.who @ lat + self.bho).astype(np.float32))
            else:  # (mainline: the candidates' markov_w2 rows, its bias fb)
                w2 = self.gguf.rows("markov_w2.weight", c, prefetch=False)
                col = sc.exact(c.astype(np.int64), h[i + 1]) + w2 @ lat + self.fb[c]
            if cap:
                col = np.tanh(col / cap) * cap
            a = int(np.argmax(col))
            e = np.exp(col - col[a])
            chain *= float(1 / e.sum())
            prev = int(c[a])
            out.append(prev)
            self.chains.append(chain)
            if prev in stop:
                break
        return out


__all__ = ["INJ", "DSparkOverlay", "MarkovChain"]
