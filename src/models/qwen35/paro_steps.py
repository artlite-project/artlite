"""A PARO checkpoint's steps on the qwen35 decode overlay (``image``'s PARO layout; the decode design's "Rotated
inputs", ``decode.rotated``): the phases, their flags and kinds, and the step's build. Per GDN layer gin_q, gin_z (each
F_PSQ on its rotated copy of the residual, F_MSG: the GDN core's message in two parts, ``words.M_MPART``), the GDN step
(its outputs into rot_g's input, a column's v heads in a row), rot_g (each column's v heads rotated into gdn_out's
input), gdn_out (gate's and up's copies); per attention layer attn_q, attn_k, attn_v (F_PSQ: into the q objects and the
raw K / V objects), the attention (o into rot_o's input), rot_o, attn_out; then gate, up (their rows into rot_down's
input), rot_down (act(gate) up of each column's slice of down's input: the ffn in whole rotation groups a column, zero
past it), down (the next layer's copies; the last layer's residual into the LM head's input). The first layer's copies
come from rot_in (the residual slices the columns load). Phases whose inputs came with an earlier phase's (gin_z,
attn_k, attn_v, up) take no sync. A design of 4 slots (the 4-lane rounds, ``kmath.rotate``), one sequence."""

from typing import TYPE_CHECKING

import numpy as np

from artlite.decode import phases as mp
from artlite.decode.protocol import (
    C_LEN,
    F_EMITR,
    F_MSG,
    F_PSQ,
    F_RACT,
    F_RESID,
    F_ROTE,
    F_ROTX,
    XB,
)
from artlite.decode.rotated import RotatedOverlay
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction

from .image import QwenImage, gdn_v_heads
from .steps import PanelTable, Step
from .words import M_MPART, MP_CONT, MP_MORE

if TYPE_CHECKING:
    from .overlay import QwenOverlay

FLAGS = {"rot_in": F_EMITR | F_ROTE, "gin_q": F_PSQ | F_MSG, "gin_z": F_PSQ | F_MSG, "attn_q": F_PSQ, "attn_k": F_PSQ,
         "attn_v": F_PSQ, "rot_g": F_ROTX, "rot_o": F_ROTX, "gdn_out": F_RESID | F_EMITR | F_ROTE,
         "attn_out": F_RESID | F_EMITR | F_ROTE, "gate": F_PSQ, "up": F_PSQ, "rot_down": F_ROTX | F_RACT,
         "down": F_RESID | F_EMITR | F_ROTE}
KINDS = {"rot_in": "in", "gin_q": "psq", "gin_z": "psq", "attn_q": "psq", "attn_k": "psq", "attn_v": "psq",
         "gate": "psq", "up": "psq", "rot_g": "rotx", "rot_o": "rotx", "rot_down": "rotx"}
DATA = {"rot_in": "rot_in", "rot_g": "rot_g", "rot_o": "rot_o", "gdn_out": "gdn_out_rot", "attn_out": "attn_out_rot",
        "rot_down": "rot_down", "down": "down_rot"}  # (each phase's rotation objects: ``image``'s data regions)
NOSYNC = frozenset({"gin_z", "attn_k", "attn_v", "up"})
MIXER = {True: ("attn_q", "attn_k", "attn_v", "rot_o", "attn_out"), False: ("gin_q", "gin_z", "rot_g", "gdn_out")}
FFN = ("gate", "up", "rot_down", "down")


class ParoSteps(RotatedOverlay):
    """The PARO part of ``overlay.QwenOverlay`` (module docstring)."""

    paro = False
    image: QwenImage

    # ------------------------------------------------------------------
    # The phases
    # ------------------------------------------------------------------

    def paro_phases(self: "QwenOverlay") -> list[tuple[int, str]]:
        out = [(self.layers[0], "rot_in")]
        for L in self.layers:
            out += [(L, n) for n in MIXER[bool(self.cfg.attention[L])] + FFN]
        return [*out, (-1, "lm")]

    def paro_flags(self: "QwenOverlay", L: int, name: str) -> int:
        """(the last layer's down: its residual plain, into the LM head's normed input)"""
        return FLAGS[name] & ~(F_ROTE if (L, name) == (self.layers[-1], "down") else 0)

    def copies_of(self: "QwenOverlay", L: int) -> list[tuple[int, str]]:
        """The phases that take layer L's input: its rotated copies."""
        return [(L, n) for n in (("attn_q", "attn_k", "attn_v") if self.cfg.attention[L] else ("gin_q", "gin_z"))]

    def rot_kind(self, name: str) -> str | None:
        return KINDS.get(name) if self.paro else None

    def rot_slice(self: "QwenOverlay", name: str) -> tuple[int, int]:
        cfg = self.cfg
        if name == "rot_g":  # (a column's v heads)
            return len(gdn_v_heads(cfg, 0)) * cfg.ssm_state, 1
        if name == "rot_o":  # (a column's q heads)
            return cfg.num_attention_heads // COLS * cfg.head_dim, 1
        return QwenImage.down_slice(cfg), 2

    def rot_objects(self: "QwenOverlay", L: int, name: str) -> int:
        d = DATA.get(name)
        return self.image.rot_groups(L, d) if d is not None and self.image.rot_region(L, d) else 0

    def rot_copies(self: "QwenOverlay", L: int, name: str) -> int:
        if name in ("gdn_out", "attn_out"):
            return 2
        i = self.layers.index(L)
        return len(self.copies_of(L if name == "rot_in" else self.layers[i + 1]))

    def paro_groups(self: "QwenOverlay", L: int, name: str, S: int) -> list[mp.Group]:
        flags = self.paro_flags(L, name)
        plain = [] if self.rot_kind(name) in ("in", "rotx") else mp.groups(self.image.region(L, name), flags, S,
                                                                           slots=self.SLOTS)
        out = self._rot_groups(L, name, S, flags, plain)
        if name in ("gin_q", "gin_z"):  # (the GDN core's message in two parts: gin_q's qkv rows, then gin_z's z and ab)
            # (the qkv rows the main core emits: the block's valid ones, not the padding of its last row tile)
            nq = -(-int((self.image.region(L, "gin_q").block("qkv").rows[0] >= 0).sum()) // C_LEN)
            out[0].meta[M_MPART] = nq | MP_MORE if name == "gin_q" else (self.gsh.n_msg - nq) | MP_CONT
        return out

    def gdn_x(self: "QwenOverlay", L: int, c: int, h: int) -> tuple[tuple[int, str], int]:
        """(phase, element) of column c's v head h's outputs (its h-th) in the GDN step's output x: gdn_out's (the head's
        place), a PARO checkpoint's rot_g's (the column's heads in a row)."""
        d = self.cfg.ssm_state
        if self.paro:
            return (L, "rot_g"), self.rot_slice("rot_g")[0] * c + d * h
        return (L, "gdn_out"), d * gdn_v_heads(self.cfg, c)[h]

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def paro_weights(self: "QwenOverlay", seq: Sequence, phase: tuple[int, str], gi: int, last: bool) -> None:
        """The A fills of a phase's K group gi: its tiles (a rotated phase of tiles), then after the last group's its
        rotation objects (``DATA``)."""
        L, name = phase
        dev = self.dev
        if self.rot_kind(name) not in ("in", "rotx"):
            region = self.image.region(L, name)
            gr = mp.groups(region, self.paro_flags(L, name), 1)[gi]
            for c in range(COLS):
                for o, n, dims, rep, it in mp.a_reads(region, c, gr):
                    seq.fill("A", c, dev.address("W", o), n, dims, rep, it)
        d = DATA.get(name)
        if last and d is not None and self.image.rot_region(L, d):
            r = self.image.region(L, d)
            for c in range(COLS):
                seq.fill("A", c, dev.address("W", r.offset + c * r.col_stride), r.data)

    def paro_phase(self: "QwenOverlay", seq: Sequence, st: Step, phase: tuple[int, str]) -> None:
        """A phase of the step: per K group its weights and inputs (the first after a sync: its producers' outputs in
        DDR; not ``NOSYNC``'s), then its outputs (``_outputs``)."""
        name = phase[1]
        if self.rot_kind(name) is not None:  # (one K group, or none)
            self.paro_weights(seq, phase, 0, True)
            if name not in NOSYNC:
                seq.sync()
            if not self._rot_inputs(seq, st, phase):
                self._outputs(seq, st, phase)
            return
        regs = self.inp[phase]
        for gi, rg in enumerate(regs):
            self.paro_weights(seq, phase, gi, gi + 1 == len(regs))
            if gi == 0:
                seq.sync()
            self._inputs(seq, rg, st.S, phase)
        self._outputs(seq, st, phase)

    def _outputs(self: "QwenOverlay", seq: Sequence, st: Step, phase: tuple[int, str]) -> None:
        """A PARO phase's outputs (module docstring): F_ROTE copies, x drains into the next phase's input, the q objects
        and the raw K / V objects; gin_q's and gin_z's go to the GDN cores (F_MSG)."""
        L, name = phase
        S, cfg = st.S, self.cfg
        h8, ffn = self.H // COLS, cfg.intermediate_size // COLS
        i = self.layers.index(L) if L >= 0 else -1
        nxt = self.layers[i + 1] if 0 <= i < len(self.layers) - 1 else None
        for c in range(COLS):
            if name == "rot_in":
                self.emit_copies(seq, c, S, self.copies_of(L))
            elif name == "attn_q":
                self.attn_steps.q_drain(seq, L, S, c)
            elif name in ("attn_k", "attn_v"):
                self.attn_steps.kv_drain(seq, L, S, c, int(name == "attn_v"))
            elif name == "rot_g":  # (the column's v heads at their places in gdn_out's x, a drain per run of consecutive
                # heads: a rotx phase's drains are queued before its inputs, at most 4 a shim channel; the 27B's 6 heads a
                # column in 3 runs)
                heads = gdn_v_heads(cfg, c)
                runs = np.split(heads, np.flatnonzero(np.diff(heads) != 1) + 1)
                assert len(runs) <= 4, runs
                for r in runs:
                    self.x_to(seq, c, (L, "gdn_out"), e0=cfg.ssm_state * int(r[0]), n=cfg.ssm_state * len(r), S=S)
            elif name == "rot_o":
                n = self.rot_slice(name)[0]
                self.x_to(seq, c, (L, "attn_out"), e0=n * c, n=n, S=S)
            elif name in ("gdn_out", "attn_out"):
                self.emit_copies(seq, c, S, [(L, "gate"), (L, "up")])
            elif name in ("gate", "up"):
                self.x_to(seq, c, (L, "rot_down"), e0=ffn * c, n=ffn, S=S, mul=2,
                          so=self._sp(S) if name == "up" else 0)
            elif name == "rot_down":  # (past down's K, the padding groups' outputs: into the sink)
                last = self.inp[(L, "down")][-1]
                n, K = self.rot_slice(name)[0], last.k0 + last.K  # (down's whole K: its K groups')
                real = max(0, min(n, K - n * c))
                self.x_to(seq, c, (L, "down"), e0=n * c, n=real, S=S)
                self.sink_out(seq, c, n - real, S)
            elif name == "down" and nxt is not None:
                self.emit_copies(seq, c, S, self.copies_of(nxt))
            elif name == "down":
                self._values_to(seq, c, (-1, "lm"), e0=h8 * c, n=h8, S=S)

    def paro_build(self: "QwenOverlay", S: int, lm: bool, subs: tuple[tuple[int, int, int], ...], b: int) -> Step:
        """The transaction of a step of S slots (``QwenOverlay._build``'s)."""
        dev = self.dev
        seq = Sequence(self.op, inputs=("X",))
        st = Step(S, subs, b=b)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * self.SLOTS * XB), S * XB)
        self.paro_phase(seq, st, (self.layers[0], "rot_in"))
        for L in self.layers:
            if self.cfg.attention[L]:
                for name in ("attn_q", "attn_k", "attn_v"):
                    self.paro_phase(seq, st, (L, name))
                self.attn_steps.build(seq, st, L, S, phase=None, out="rot_o")
            else:
                self.gdn_steps.build(seq, st, L, S)  # (gin_q, gin_z: ``gdn_in``)
            for name in MIXER[bool(self.cfg.attention[L])][-2:] + FFN:
                self.paro_phase(seq, st, (L, name))
        if lm:
            self._phase(seq, (-1, "lm"), S, "LOGITS")
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        if self.ash is not None:
            st.patch = self.attn_steps.patch_tables(st)
        if st.panels:
            st.table = PanelTable.of(st.panels)
        return st


def gdn_in_rows(image: QwenImage, L: int) -> np.ndarray:
    """A GDN layer's qkv rows per column (gdn_in's block; a PARO checkpoint's gin_q's)."""
    return image.region(L, "gin_q" if image.paro else "gdn_in").block("qkv").rows
