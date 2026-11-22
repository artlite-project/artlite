"""A PARO checkpoint's steps on the Gemma decode overlay (``image``'s PARO layout; the decode design's "Rotated inputs"
with R phases, ``decode.design``): the phases, their regions, metadata and transfers.

Gemma's post-norm updates (``overlay``'s headers) become R phases: per update a region of the header's layout
(``hdr``: [metadata object | the producer's partial sums object | x: [k block][v | r][slot SP][8]], v the output
projection's rows, r the residual before the update), which one R phase per consuming projection reads, each column its
contiguous rotation groups (``image.R_GROUPS``) in sub-phases of one X object (``r_subs``), with metadata objects of its
own per column (the groups and objects differ): n = (r + v w_post / rms(v)) scale (1 / rms(v) from the producer's
partial sums object, which an update's first R phase reads), u = n w' (the copy's norm weight
folded into its rotation objects), rotated, out into the consumer's x. The first R phase of an update (``UPD``) also
updates the residual slice exactly (``decode.design``'s wpupd) and drains it as r into the next update's region; each
drains its copy's partial sums (the slice's res^2) into its consumer's partial sums object (an F_PSQ input: 1 / rms of
the updated residual), pgate's into the sink (the per-layer input gate reads the residual itself: no 1 / rms).

Per layer L: rq, rk, rv (KV-shared layers: rq) -> q, k, v (F_PSQ: into the attention region's q and raw K / V objects:
k and v a KV head's quarters a column), att (o into rot_o's input), rot_o (each column its head), o (F_VOUT: v and its
partial sums into the gu region), rgate, rup -> gate, up (F_PSQ: into rot_down's [gate | up]), rot_down (gelu(gate) up
of the column's 1280, rotated), down (F_VOUT -> the pg region), rpgate -> pgate (gelu(y) pli: F_PLI, its 32 rows a
column into rot_proj's 64-slot input), rot_proj (column c rotates proj's group c % 2 of the slots' first 32 channels
into element 128 c of proj's input: its 4 row groups' copies of the 256), proj (F_VOUT -> the next layer's qkv region);
then rlm (the last layer's update, an identity copy over the output norm) -> lm. The step's first layer's qkv region holds
its input as r (the embeddings, or a prefill chunk's residual over the output scale before; v and its partial sums zero:
n = r). Inputs that came with an earlier phase's (rk, rv, k, v, rup, up)
take no sync."""

import itertools
from typing import TYPE_CHECKING

import numpy as np

from artlite.decode import phases as mp
from artlite.decode.design import ROT_GROUP
from artlite.decode.model import InputRegion
from artlite.decode.protocol import (
    C_LEN,
    F_EMITR,
    F_MAIN,
    F_PLI,
    F_PSQ,
    F_PZERO,
    F_RACT,
    F_ROTX,
    F_RPOST,
    F_RPS,
    F_RSLOT,
    F_RUPD,
    F_VOUT,
    M_FLAGS,
    M_NRA,
    M_NS,
    M_NT,
    M_NX,
    M_RGN,
    M_SCALE,
    META,
    SLOTS,
    XB,
    XPAY,
    hidden_meta,
    packed_slots,
    post_region_bytes,
    region_bytes,
)
from artlite.decode.rotated import Fill, RotatedOverlay
from artlite.decode.step import StepPlan
from artlite.operators.design import COLS
from artlite.operators.util import bf16_bits, rup
from artlite.overlay import Sequence

from . import attn as AT
from .image import R_GROUPS, GemmaImage

if TYPE_CHECKING:
    from .overlay import GemmaOverlay

R_OF = {"rq": "q", "rk": "k", "rv": "v", "rgate": "gate", "rup": "up", "rpgate": "pgate", "rlm": "lm"}  # (the consumer)
UPD = frozenset({"rq", "rgate", "rpgate", "rlm"})  # (an update's first R phase: F_RUPD)
HDR = {"rq": "qkv", "rk": "qkv", "rv": "qkv", "rgate": "gu", "rup": "gu", "rpgate": "pg", "rlm": "lm"}  # (its region)
FLAGS = {"q": F_PSQ, "k": F_PSQ, "v": F_PSQ, "gate": F_PSQ, "up": F_PSQ, "lm": F_PSQ, "pgate": F_PLI, "rot_o": F_ROTX,
         "rot_down": F_ROTX | F_RACT, "rot_proj": F_ROTX | F_RSLOT, "o": F_VOUT | F_PZERO | F_EMITR,
         "down": F_VOUT | F_PZERO | F_EMITR, "proj": F_VOUT | F_PZERO | F_EMITR, "ple": 0}
KINDS = {"q": "psq", "k": "psq", "v": "psq", "gate": "psq", "up": "psq", "lm": "psq", "rot_o": "rotx",
         "rot_down": "rotx", "rot_proj": "rotx"}
DATA = {"rot_o": "rot_o", "rot_down": "rot_down", "rot_proj": "rot_proj"}  # (rotx phases' rotation objects)
NOSYNC = frozenset({"rk", "rv", "k", "v", "rup", "up"})
LAYER = ("rq", "rk", "rv", "q", "k", "v", "att", "rot_o", "o", "rgate", "rup", "gate", "up", "rot_down", "down",
         "rpgate", "pgate", "rot_proj", "proj")
SUBS = 2  # (an R phase's sub-phases at most: 3 groups, 2 an X object at SP 4)
R_META = (SUBS - 1) * XPAY + XB  # (bytes of a column's metadata objects of an R phase: XPAY apart, ``r_fills``)


class GemmaParo(RotatedOverlay):
    """The PARO part of ``overlay.GemmaOverlay`` (module docstring)."""

    paro = False
    image: GemmaImage

    # ------------------------------------------------------------------
    # The phases
    # ------------------------------------------------------------------

    def paro_phases(self: "GemmaOverlay", lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order, from layer ``start`` on (``lm`` False: up to the last attention that writes a
        cache)."""
        cfg = self.cfg
        out = [(-1, "ple")] if cfg.per_layer_dim and self.ple else []
        for L in self.layers[self.layers.index(self.start):]:
            own = cfg.kv_source(L) == L
            out += [(L, n) for n in LAYER if (own or n not in ("rk", "rv", "k", "v"))
                    and (cfg.per_layer_dim or n not in ("rpgate", "pgate", "rot_proj", "proj"))]
        if not lm:
            own = [L for L in self.layers if cfg.kv_source(L) == L]
            return out[:out.index((own[-1], "att")) + 1] if own else []
        return [*out, (-1, "rlm"), (-1, "lm")]

    def paro_flags(self: "GemmaOverlay", L: int, name: str) -> int:
        return F_ROTX | F_RPOST if name in R_OF else FLAGS[name]

    def rot_kind(self: "GemmaOverlay", name: str) -> str | None:
        return KINDS.get(name) if self.paro else None

    def rot_slice(self: "GemmaOverlay", L: int, name: str) -> tuple[int, int]:
        if name == "rot_o":  # (a column's head)
            return self.cfg.head_dim(L), 1
        if name == "rot_down":
            return self.cfg.intermediate_size // COLS, 2
        assert name == "rot_proj", name
        return ROT_GROUP, 2  # (a group of the 64-slot input: two k blocks' room a k block, as [gate | up]'s)

    def rot_start(self: "GemmaOverlay", L: int, name: str, c: int) -> int:
        return ROT_GROUP * (c % 2) if name == "rot_proj" else super().rot_start(L, name, c)

    def rot_objects(self: "GemmaOverlay", L: int, name: str) -> int:
        d = DATA.get(name)
        return self.image.rot_groups(L, d) if d is not None else 0

    def r_subs(self: "GemmaOverlay", c: int, S: int) -> list[tuple[int, int]]:
        """Column c's sub-phases of an R phase of S slots: (first group of its slice, groups), one X object each."""
        per = XPAY // (2 * 2 * packed_slots(S) * ROT_GROUP)  # (groups of an X object: [v | r] per k block)
        n = R_GROUPS[c]
        return [(g, min(per, n - g)) for g in range(0, n, per)]

    def next_layer(self: "GemmaOverlay", L: int) -> int:
        i = self.layers.index(L)
        return self.layers[i + 1] if i + 1 < len(self.layers) else -1

    def hdr_of(self: "GemmaOverlay", phase: tuple[int, str]) -> int:
        """The ACT offset of the update region an R phase reads (``HDR``)."""
        L, name = phase
        return self.hdr[(L, HDR[name])]

    def upd_to(self: "GemmaOverlay", phase: tuple[int, str]) -> int:
        """The ACT offset of the region an update's residual goes into (its next update's; rlm's: ``final``)."""
        L, name = phase
        if name == "rq":
            return self.hdr[(L, "gu")]
        if name == "rgate":
            return self.hdr[(L, "pg")] if self.cfg.per_layer_dim else self.hdr[(self.next_layer(L), "qkv")]
        if name == "rpgate":
            nxt = self.next_layer(L)
            return self.hdr[(nxt, "qkv")] if nxt >= 0 else self.hdr[(-1, "lm")]
        assert name == "rlm", name
        return self.final

    def vout_to(self: "GemmaOverlay", phase: tuple[int, str]) -> int:
        """The ACT offset of the update region an F_VOUT phase's rows and partial sums go into."""
        L, name = phase
        if name == "o":
            return self.hdr[(L, "gu")]
        if name == "down" and self.cfg.per_layer_dim:
            return self.hdr[(L, "pg")]
        nxt = self.next_layer(L)
        return self.hdr[(nxt, "qkv")] if nxt >= 0 else self.hdr[(-1, "lm")]

    def r_scale(self: "GemmaOverlay", L: int, name: str) -> float:
        """An R phase's update's output scale: the layer before L's (rq, rk, rv; rlm: the last layer's), else 1."""
        if name in ("rq", "rk", "rv") and L != self.layers[0]:
            prev = self.layers[self.layers.index(L) - 1]
        elif name == "rlm":
            prev = self.layers[-1]
        else:
            return 1.0
        sc = self.image._post_of(prev)[1]
        return self._scalar(sc) if sc else 1.0

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def paro_plan(self: "GemmaOverlay") -> None:
        """The inputs of the phases, the update regions (``hdr``), the R phases' metadata objects (``rmeta``)."""
        cfg, H = self.cfg, self.H
        take = self._take
        if cfg.per_layer_dim:  # (pgate's inputs gate_pitch apart: the per-layer projection's drains, ``_outputs``)
            for L in self.layers:
                self._add_inputs((L, "pgate"))
            offs = [self.inp[(L, "pgate")][0].off for L in self.layers]
            self.gate_pitch = rup(region_bytes(self.image.region(self.layers[0], "pgate").K_in))
            assert offs == [offs[0] + i * self.gate_pitch for i in range(len(offs))]
            if len(offs) % 2:
                take("ACT", self.gate_pitch)  # (the padding layer's rows: a sink)
        self.hdr = {}
        self.rmeta: dict[tuple[int, str], int] = {}
        reads: dict[tuple[int, str], list[tuple[int, str]]] = {}  # (an update region's R phases)
        for L, name in self.phases():
            if name in R_OF:
                reads.setdefault((L, HDR[name]), []).append((L, name))
        hdrs = [(L, h) for L in self.layers for h in (("qkv", "gu", "pg") if cfg.per_layer_dim else ("qkv", "gu"))]
        for key in [*hdrs, (-1, "lm")]:  # (its R phases' metadata objects before it: ``r_fills``)
            for phase in reads.get(key, ()):
                self.rmeta[phase] = take("ACT", COLS * R_META)
            self.hdr[key] = take("ACT", post_region_bytes(H))
        self.final = take("ACT", post_region_bytes(H))  # (rlm's updated residual)
        for L, name in self.phases():
            if name in R_OF:
                continue
            if name == "rot_proj":  # (the 64-slot input: C objects of a column's 32 rows and 32 zeros)
                self.inp[(L, name)] = [InputRegion(take("ACT", region_bytes(COLS * 64)), 0, COLS * 64, x0=XB)]
            elif self.rot_kind(name) is not None:
                self._add_rot_inputs((L, name))
            elif name not in ("att", "pgate"):
                self._add_inputs((L, name))
        self.rot_sink = take("ACT", 2 * SLOTS * C_LEN)  # (pgate's copies' partial sums: no reader)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def paro_groups(self: "GemmaOverlay", L: int, name: str, S: int) -> list[mp.Group]:
        """The metadata of a phase's input regions (an R phase's: ``r_meta``)."""
        flags = self.paro_flags(L, name)
        if name == "rot_proj":
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS], m[M_NX], m[M_RGN] = F_MAIN | flags, S, SLOTS // packed_slots(S), 1
            m[M_NT] = m[M_NRA] = 1
            return [mp.Group(0, 0, m, 0, 0)]
        plain = [] if self.rot_kind(name) == "rotx" else mp.groups(self.image.region(L, name), flags, S, slots=SLOTS)
        return self._rot_groups(L, name, S, flags, plain) if self.rot_kind(name) else plain

    def r_meta(self: "GemmaOverlay", L: int, name: str, S: int) -> list[tuple[int, np.ndarray]]:
        """An R phase's metadata objects of S slots: (ACT offset, bytes) per column and sub-phase (module docstring):
        the first sub-phase's F_RPS, F_RUPD (an update's first: its partial sums object follows, counted in M_NX), F_PZERO
        (the step's first layer's: no update, pend zero, ``overlay.GemmaOverlay.step``); per sub-phase its groups and their rotation objects, the last one's
        padding object (a column of fewer than the most groups) counted in M_NT."""
        sp = packed_slots(S)
        per = SLOTS // sp  # (strips of an X object)
        nmax = -(-max(R_GROUPS) // 2)
        out = []
        for c in range(COLS):
            subs = self.r_subs(c, S)
            for i, (g0, ng) in enumerate(subs):
                m = np.zeros(META, np.int32)
                f = F_MAIN | F_ROTX | F_RPOST
                if i == 0:
                    f |= F_RPS | (F_RUPD if name in UPD else 0) | (F_PZERO if L == self.start and name == "rq" else 0)
                nra = -(-(g0 + ng) // 2) - g0 // 2  # (its groups' objects: two a 9216-byte object, from group g0)
                pad = nmax - self.image.r_objects(c) if i + 1 == len(subs) else 0
                m[M_FLAGS], m[M_NS], m[M_NX], m[M_RGN] = f, S, per * (2 if f & F_RUPD else 1), ng
                m[M_NRA], m[M_NT] = nra, nra + pad
                hidden_meta(m, self.H)
                m[M_SCALE] = bf16_bits(self.r_scale(L, name))
                out.append((self.r_meta_at((L, name), c, i), m.view(np.uint8)))
            for i in range(len(subs), SUBS):  # (unread, zeros: every S writes the same bytes, ``_metas``)
                out.append((self.r_meta_at((L, name), c, i), np.zeros(4 * META, np.uint8)))
        return out

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def paro_gemv(self: "GemmaOverlay", seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """A phase's transfers (its weights are the column's run, ``_weight_run``): after a sync (not ``NOSYNC``'s), an R
        or rotx phase's drains, then its fills; a phase of tiles' fills, then its drains."""
        name = phase[1]
        if name not in NOSYNC:
            seq.sync()  # (its inputs' producers)
        if name in R_OF:
            self.fills_and_drains(seq, st, phase, self.r_fills(phase, st.S))
        elif name == "rot_proj":  # (column c: proj's group c % 2, 256 elements of the 64-slot input)
            rg = self.inp[phase][0]
            sp = packed_slots(st.S)
            self.fills_and_drains(seq, st, phase, [[rg.off, rg.off + rg.x0 + 2 * sp * 2 * ROT_GROUP * (c % 2) - (XB - XPAY)]
                                                   for c in range(COLS)])
        elif self.rot_kind(name) is not None:
            if not self._rot_inputs(seq, st, phase):
                self._outputs(seq, st, phase)
        else:
            for rg in self.inp[phase]:
                self._inputs(seq, rg, st.S, phase)
            self._outputs(seq, st, phase)

    def r_meta_at(self: "GemmaOverlay", phase: tuple[int, str], c: int, i: int) -> int:
        """The ACT offset of column c's metadata object of an R phase's sub-phase i (a column's XPAY apart, as its X
        objects are: ``r_fills``; the objects' tails, which nobody reads, under the next ones)."""
        return self.rmeta[phase] + c * R_META + i * XPAY

    def r_fills(self: "GemmaOverlay", phase: tuple[int, str], S: int) -> list[list[Fill]]:
        """An R phase's fills per column: per sub-phase its metadata object, an update's first R phase's first then the
        partial sums object, then its X object (the XB bytes ending with its groups' [v | r]). Two objects a task where
        the second lies past the first (a BD of two strip groups that far apart: the metadata objects lie before their
        update region), the sub-phases after the first a repetition of one (their objects XPAY apart)."""
        sp, h = packed_slots(S), self.hdr_of(phase)
        x = h + XB + XPAY

        def pair(a: int, b: int, n: int = 1) -> Fill:
            assert b > a, (a, b)
            return Fill(a, 2 * XB, (XB // 8, 2, XB // 8, (b - a) // 4), n, (XPAY // 4, n))

        out = []
        for c in range(COLS):
            subs = self.r_subs(c, S)
            X = [x + 2 * 2 * sp * ROT_GROUP * (sum(R_GROUPS[:c]) + g0) - (XB - XPAY) for g0, _ in subs]
            assert all(b - a == XPAY for a, b in itertools.pairwise(X)), X
            m0 = self.r_meta_at(phase, c, 0)
            f = [Fill(m0), pair(h + XPAY, X[0])] if phase[1] in UPD else [pair(m0, X[0])]
            if len(subs) > 1:
                f.append(pair(self.r_meta_at(phase, c, 1), X[1], len(subs) - 1))
            out.append(f)
        return out

    def paro_outputs(self: "GemmaOverlay", seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """A PARO phase's drains (module docstring)."""
        L, name = phase
        S, cfg, dev = st.S, self.cfg, self.dev
        h8 = self.H // COLS
        for c in range(COLS):
            if name in R_OF:
                if name in UPD:  # (the updated residual slice: the next update's r)
                    self._post_to(seq, c, self.upd_to(phase), h8 * c, h8, S, 1)
                cons = (L if name != "rlm" else -1, R_OF[name])
                if name == "rpgate":  # (no 1 / rms: its partial sums into the sink)
                    seq.drain("C", c, dev.address("ACT", self.rot_sink), 2 * C_LEN)
                else:
                    seq.drain("C", c, dev.address("ACT", self.inp[cons][0].off + XB + 2 * C_LEN * c), 2 * C_LEN)
                g0 = sum(R_GROUPS[:c])
                self.x_to(seq, c, cons, e0=ROT_GROUP * g0, n=ROT_GROUP * R_GROUPS[c], S=S)
            elif name in ("q", "k", "v"):
                hd, h, j = cfg.head_dim(L), c // AT.QH, c % AT.QH
                reg = self.att_reg[(L, h)]  # ([q objects SLOTS | raw K / V objects SLOTS])
                if name == "q":
                    at, n = reg + 4 * META + 2 * hd * j, hd
                else:  # (KV head h's quarter j of K, or of V after K in the raw K / V object)
                    at, n = reg + SLOTS * XB + 4 * META + (2 * hd if name == "v" else 0) + hd // 2 * j, hd // 4
                seq.drain("C", c, dev.address("ACT", at), S * 2 * n, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name == "rot_o":
                n = self.rot_slice(L, name)[0]
                self.x_to(seq, c, (L, "o"), e0=n * c, n=n, S=S)
            elif name in ("o", "down", "proj"):
                tgt = self.vout_to(phase)
                self._post_to(seq, c, tgt, h8 * c, h8, S, 0)
                seq.drain("C", c, dev.address("ACT", tgt + XB + 2 * C_LEN * c), 2 * C_LEN)
            elif name in ("gate", "up"):
                ffn = cfg.intermediate_size // COLS
                self.x_to(seq, c, (L, "rot_down"), e0=ffn * c, n=ffn, S=S, mul=2, so=packed_slots(S) if name == "up" else 0)
            elif name == "rot_down":
                n = self.rot_slice(L, name)[0]
                self.x_to(seq, c, (L, "down"), e0=n * c, n=n, S=S)
            elif name == "pgate":  # (its 32 rows and 32 zeros: column c's slot of 64)
                self.x_to(seq, c, (L, "rot_proj"), e0=64 * c, n=C_LEN, S=S)
            elif name == "rot_proj":  # (group c % 2 of copy c // 2 of proj's x)
                self._values_to(seq, c, (L, "proj"), e0=ROT_GROUP * c, n=ROT_GROUP, S=S)
            elif name == "ple":
                self.ple_out(seq, c, S)
            elif name == "lm":
                self._logits_out(seq, c, S)
