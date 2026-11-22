"""A PARO checkpoint's steps on the qwen35 decode overlay (``image``'s PARO layout; the decode design's "Rotated
inputs", ``decode.rotated``): the phases, their flags and kinds, and the step's build. Per GDN layer gin_q, gin_z (each
F_PSQ on its rotated copy of the residual, F_MSG: the GDN core's message in two parts, ``words.M_MPART``), the GDN step
(its outputs into rot_g's input, a column's v heads in a row), rot_g (each column's v heads rotated into gdn_out's
input), gdn_out (gate's and up's copies); per attention layer attn_q, attn_k, attn_v (F_PSQ: into the q objects and the
raw K / V objects), the attention (o into rot_o's input), rot_o, attn_out; then gate, up (their rows into rot_down's
input), rot_down (act(gate) up of each column's slice of down's input: the ffn in whole rotation groups a column, zero
past it), down (the next layer's copies; the last layer's residual into the LM head's input). The first layer's copies
come from rot_in (the residual slices the columns load), the LM head's from the last layer's down (an identity copy
over the output norm, ``image``). An MoE layer instead of gate .. down: moe_in (F_PSQ on its copy) and sdown, the route,
expert and combine objects of ``moe.steps`` (``MoeRot``: ``moe_in_fill``, the combine's metadata per layer and its
copies). Phases whose inputs came with an earlier phase's (gin_z, attn_k, attn_v, up) take no sync. A design of 4 slots
(the 4-lane rounds, ``kmath.rotate``), one sequence."""

from typing import TYPE_CHECKING

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode import phases as mp
from artlite.decode.protocol import (
    C_LEN,
    F_COMB,
    F_EMITR,
    F_MAIN,
    F_MSG,
    F_PSQ,
    F_RACT,
    F_RESID,
    F_ROTE,
    F_ROTX,
    F_WS,
    M_FLAGS,
    M_NRA,
    M_NRC,
    M_NS,
    M_NT,
    M_NX,
    META,
    XB,
    XPAY,
    interleave,
)
from artlite.decode.rotated import RotatedOverlay
from artlite.moe.steps import MOE_BLOCKS
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction

from .image import QwenImage, gdn_v_heads
from .steps import PanelTable, Step
from .words import M_MPART, MP_J0, MP_LAST

if TYPE_CHECKING:
    from .overlay import QwenOverlay

FLAGS = {"rot_in": F_EMITR | F_ROTE, "gin_q": F_PSQ | F_MSG, "gin_z": F_PSQ | F_MSG, "attn_q": F_PSQ, "attn_k": F_PSQ,
         "attn_v": F_PSQ, "rot_g": F_ROTX, "rot_o": F_ROTX, "gdn_out": F_RESID | F_EMITR | F_ROTE,
         "attn_out": F_RESID | F_EMITR | F_ROTE, "gate": F_PSQ, "up": F_PSQ, "rot_down": F_ROTX | F_RACT,
         "down": F_RESID | F_EMITR | F_ROTE, "moe_in": F_PSQ, "sdown": F_RESID | F_WS, "lm": F_PSQ}
KINDS = {"rot_in": "in", "gin_q": "psq", "gin_z": "psq", "attn_q": "psq", "attn_k": "psq", "attn_v": "psq",
         "gate": "psq", "up": "psq", "rot_g": "rotx", "rot_o": "rotx", "rot_down": "rotx", "moe_in": "psq", "lm": "psq"}
DATA = {"rot_in": "rot_in", "rot_g": "rot_g", "rot_o": "rot_o", "gdn_out": "gdn_out_rot", "attn_out": "attn_out_rot",
        "rot_down": "rot_down", "down": "down_rot"}  # (each phase's rotation objects: ``image``'s data regions)
NOSYNC = frozenset({"gin_z", "attn_k", "attn_v", "up"})
MIXER = {True: ("attn_q", "attn_k", "attn_v", "rot_o", "attn_out"), False: ("gin_q", "gin_z", "rot_g", "gdn_out")}
FFN = ("gate", "up", "rot_down", "down")
MOE_FFN = ("moe_in", "sdown")  # (an MoE layer's phases: its route object, experts and combine, ``moe.steps``)


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
            out += [(L, n) for n in MIXER[bool(self.cfg.attention[L])] + (MOE_FFN if self.moe else FFN)]
        return [*out, (-1, "lm")]

    def paro_flags(self: "QwenOverlay", L: int, name: str) -> int:
        return FLAGS[name]

    def copies_of(self: "QwenOverlay", L: int) -> list[tuple[int, str]]:
        """The phases that take layer L's input: its rotated copies (L = -1: the LM head's, the output norm's weights
        without a rotation)."""
        if L < 0:
            return [(-1, "lm")]
        return [(L, n) for n in (("attn_q", "attn_k", "attn_v") if self.cfg.attention[L] else ("gin_q", "gin_z"))]

    def next_layer(self: "QwenOverlay", L: int) -> int:
        """The layer after L (-1: the LM head)."""
        i = self.layers.index(L)
        return self.layers[i + 1] if i + 1 < len(self.layers) else -1

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
        if name in ("gdn_out", "attn_out"):  # (gate's and up's; an MoE layer's moe_in's)
            return 1 if self.moe else 2
        return len(self.copies_of(L if name == "rot_in" else self.next_layer(L)))

    def paro_groups(self: "QwenOverlay", L: int, name: str, S: int) -> list[mp.Group]:
        flags = self.paro_flags(L, name)
        plain = [] if self.rot_kind(name) in ("in", "rotx") else mp.groups(
            self.image.region(L, name), flags, S, slots=self.SLOTS, block_flags=MOE_BLOCKS if name == "moe_in" else None)
        out = self._rot_groups(L, name, S, flags, plain)
        if name in ("gin_q", "gin_z"):  # (the GDN core's message in two parts: gin_q's qkv rows, then gin_z's z and ab)
            # (the qkv rows the main core emits: the block's valid ones, not the padding of its last row tile)
            nq = -(-int((self.image.region(L, "gin_q").block("qkv").rows[0] >= 0).sum()) // C_LEN)
            out[0].meta[M_MPART] = S * nq if name == "gin_q" else S * (self.gsh.n_msg - nq) | MP_LAST | nq << MP_J0
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
        ffn = cfg.intermediate_size // COLS
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
                self.emit_copies(seq, c, S, [(L, "moe_in")] if self.moe else [(L, "gate"), (L, "up")])
            elif name in ("gate", "up"):
                self.x_to(seq, c, (L, "rot_down"), e0=ffn * c, n=ffn, S=S, mul=2,
                          so=self._sp(S) if name == "up" else 0)
            elif name == "rot_down":  # (past down's K, the padding groups' outputs: into the sink)
                last = self.inp[(L, "down")][-1]
                n, K = self.rot_slice(name)[0], last.k0 + last.K  # (down's whole K: its K groups')
                real = max(0, min(n, K - n * c))
                self.x_to(seq, c, (L, "down"), e0=n * c, n=real, S=S)
                self.sink_out(seq, c, n - real, S)
            elif name == "down":
                self.emit_copies(seq, c, S, self.copies_of(self.next_layer(L)))
            elif name == "lm":
                self._logits_out(seq, c, S)

    def paro_build(self: "QwenOverlay", S: int, lm: bool, subs: tuple[tuple[int, int, int], ...], b: int) -> Step:
        """The transaction of a step of S slots (``QwenOverlay._build``'s)."""
        dev = self.dev
        seq = Sequence(self.op, inputs=("X",))
        st = Step(S, subs, b=b)
        if self.moe:  # (the control packets' route; the routing's shim locks)
            self.moe_steps.route(seq)
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
            for name in MIXER[bool(self.cfg.attention[L])][-2:] + (() if self.moe else FFN):
                self.paro_phase(seq, st, (L, name))
            if self.moe:  # (its outputs: the combine's, ``comb_out``)
                self.moe_steps.build(seq, L, S, (-1, "lm"), None, rot=self, st=st)
        if lm:
            self.paro_phase(seq, st, (-1, "lm"))
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        if self.ash is not None:
            st.patch = self.attn_steps.patch_tables(st)
        if st.panels:
            st.table = PanelTable.of(st.panels)
        return st


    # ------------------------------------------------------------------
    # An MoE layer's rotations (``moe.steps.MoeTransfers.build``'s ``rot``)
    # ------------------------------------------------------------------

    def moe_in_fill(self: "QwenOverlay", seq: Sequence, st: Step, L: int) -> None:
        """moe_in's input fills: its F_PSQ copy."""
        self._rot_inputs(seq, st, (L, "moe_in"))

    def comb_rotations(self: "QwenOverlay", seq: Sequence, L: int) -> None:
        """The combine's A fills: its rotation objects (the next layer's copies of the residual), after the experts'."""
        r = self.image.region(L, DATA["down"])
        for c in range(COLS):
            seq.fill("A", c, self.dev.address("W", r.offset + c * r.col_stride), r.data)

    def comb_meta(self: "QwenOverlay", L: int, S: int) -> np.ndarray:
        """The combine's metadata object of layer L: the next layer's copies (F_ROTE: M_NRC of them, M_NRA rotation
        objects on A; the last layer's: the LM head's)."""
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_NS], m[M_NX] = F_MAIN | F_COMB | F_ROTE, S, 2 * S
        m[M_NRA] = m[M_NT] = self.rot_objects(L, "down")
        m[M_NRC] = len(self.copies_of(self.next_layer(L)))
        return m

    def comb_out(self: "QwenOverlay", seq: Sequence, c: int, L: int, S: int) -> None:
        """Column c's outputs of the combine: the next layer's copies (the last layer's: the LM head's)."""
        self.emit_copies(seq, c, S, self.copies_of(self.next_layer(L)))

    def paro_logits_input(self: "QwenOverlay", x: np.ndarray) -> None:
        """(``logits_of``) The LM head's F_PSQ input of a last layer's output x (hidden): the partial sums object (x's
        sum of squares in column 0's first lane of slot 0) and x' = bf16(x bf16(w)), w the output norm's weights (the
        copy the decode's rote makes)."""
        rg = self.inp[(-1, "lm")][0]
        x = np.asarray(x, np.float32).reshape(-1)
        q = np.zeros(XPAY // 4, np.float32)
        q[0] = np.sum(x.astype(np.float64) ** 2)
        w = self._f32("output_norm.weight").astype(bfloat16).astype(np.float32)
        xi = interleave((x * w).reshape(1, -1), self._sp(1)).view(np.uint8).reshape(-1)
        act = self.dev.view("ACT")
        act[rg.off + XB:rg.off + XB + XPAY] = q.view(np.uint8)
        act[rg.off + rg.x0:rg.off + rg.x0 + xi.size] = xi
        self.dev.flush("ACT", rg.off + XB, rg.x0 - XB + xi.size)


def gdn_in_rows(image: QwenImage, L: int) -> np.ndarray:
    """A GDN layer's qkv rows per column (gdn_in's block; a PARO checkpoint's gin_q's)."""
    return image.region(L, "gin_q" if image.paro else "gdn_in").block("qkv").rows
