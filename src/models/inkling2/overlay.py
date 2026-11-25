"""An Inkling 2 model on its decode overlay (``inkling2/decode.py``): buffers, the transaction of a step of S <= 4 token
slots, the step. The overlay's "layers" are the visits (``InklingConfig.visits``: 48 a token, the loop core's layers 3
times); each visit has its own KV cache and raw ring, and reads its physical layer's regions of the image
(``inkling2.image``).

Per visit v of physical layer L (column c holding KV head c):

- q, k, v (``decode.protocol.F_STATES``: fp32 state inputs, the helper's mixes; normed by the layer's attention norms):
  per slot the column's q object ([q 3 x 128 | gates 3 x 128 | r 3 x 16 | sinks 3 fp32] bf16, ``decode.attn``'s INK_*)
  and raw K / V object ([K 128 | K deltas 4 | V 128 | V deltas 4] fp32) in the visit's attention region;
- att: the attention cores (``decode.attn.Attn(ink=True)``): per sub-step the raw ring (A: [position 8][K | V] fp32,
  the history of the K / V convs; out whole on D), the cache range and tail objects (A), the q and raw objects (X), an E
  object per cache group within the relative extent (X: [rel 16][128] bf16 of the layer's reversed table, ``ink_rel``,
  each a window of the group's distances), the records and o (D: o into the o phase's input).

The KV caches: per visit and KV head linear groups (``decode.kv``), a sliding-window visit's after its window's guard
and, past ``WRAP`` groups of capacity, a ring of ``WRAP`` groups (``CacheKind.wrap``: its range read in two fills);
the rings of open records (``decode.vring``), the raw rings (zero at a sequence's start: the convs' history before
position 0). The token objects carry tau per slot (the full layers' factor of q and r, ``InklingConfig.tau``) where
the other models' carry RoPE tables.

The residual stream: per step the state array (ACT, [state][k block][slot sp][8] fp32, ``INK_STATES``: the normed
embedding, the block states, cur in state n of its visit, ``cur_states``) holds what the mixes read; the helpers keep
each column's slices of the states (``decode.protocol.F_INK``), the first visit's loaded by F_LOADR objects. o (F_RESID
| F_INK): W_o's rows, the post conv over the visit's history window (KV: per visit, conv and column the inputs of the
positions base .. base + 6, ``HIST``; read at the step's position - 3 - base, ``_set_parts``; written whole: the 3 rows
read and the step's), cur (or a new block state) into the state array, the partial sums of the FFN's mix.

The FFN (``ffn``; ``decode.design``'s "MoE" with an expert's units, ``moe.route.render(units=)``): router (F_STATES |
F_COEF | F_NORM: the FFN mix; a column's 16 fp32 logits per slot into the route object, the C objects' other halves past
it), moe (the operator phase: the route object and the routing's round trip, whose tables after the logits ``_load``
writes, ``unit_table``; the expert object, its templates per unit kind; the partials into PART; the combine of each
visit's metadata, then its residual step over the FFN's history window with ``inkc`` and the next visit's ``inkq``:
cur's slice, the partial sums of the next visit's attention mix). A visit's phases read their regions as runs of
adjacent ones (``PHASE_REGIONS``, ``_weight_run``).

The loop's passes after the first (``Visit.messages``): per stream (q, k, v, ffn) the controller's down in two halves
(d<s>a: the stream's mix of states 0 .. n, d<s>b: the message, the mix of the bank the pass reads; each normed without
weights, F_COEF of the message's queries from the previous combine's partial sums "ma" / "mb"), their 8 columns' g halves
into the stream's g object, the blend (``protocol.F_BLEND``: tanh(F g), tanh(G g) of the column's rows, cur and msg from
the helper's states and banks) into the stream's blended slot of the state array, which the stream's GEMV phase reads as
its one state (``BLENDED``, ``_mix``). A pass's bank (its mid-core residual after layer 11, its output after layer 17)
is a second copy of those combines' cur (``bank_of``: the state array's slots from ``BANK0``, the helper's after its
states), the next pass's messages its mixes.

The state array's slots (``SA_SLOTS``): 0 .. 4 the states, 5 .. 7 the blended inputs, 8 .. 11 the banks.

The host writes the first visit's input states and, a loop visit's, the bank its messages read (``write_states``: the
state array, the F_LOADR objects, the partial sums of its mixes); the later visits' come from the steps of the ones
before (S3's next pieces: the whole step with the output's mix and the LM head).
"""

import math
import os
from typing import Self

import numpy as np
from ml_dtypes import bfloat16

from artlite import txn
from artlite.decode import kv
from artlite.decode import phases as mp
from artlite.decode.attn import (
    INK_ED0,
    INK_EG,
    INK_QC,
    INK_RC,
    INK_TAPS,
    M_AEXT,
    M_AFULL,
    Attn,
    ink_rel,
    score_bits,
)
from artlite.decode.attn_overlay import AttentionOverlay, CacheKind, CacheUse
from artlite.decode.design import BLEND_OBJECTS
from artlite.decode.model import InputRegion, Snapshot, host_bytes
from artlite.decode.protocol import (
    B_F32,
    B_PAIR,
    C_LEN,
    DIMS_OBJ,
    F_BLEND,
    F_COEF,
    F_COMB,
    F_GU,
    F_INK,
    F_LOADR,
    F_MAIN,
    F_NORM,
    F_RESID,
    F_ROUTE,
    F_STATES,
    INK_BANKS,
    INK_STATES,
    M_BLEND,
    M_CSET,
    M_EGA,
    M_EGRP,
    M_ETAIL,
    M_FLAGS,
    M_INKN,
    M_NMIX,
    M_NQ,
    M_NR,
    M_NS,
    M_NT,
    M_NX,
    M_RBASE,
    M_RSTR,
    M_SLOT,
    M_TOK_NG,
    M_TOK_PARTS,
    META,
    SLOTS,
    XB,
    XPAY,
    packed_slots,
    region_bytes,
)
from artlite.decode.screen import Screen, Screened
from artlite.decode.step import Fill, Part, PartBD, Rec, StepPlan, Tail
from artlite.decode.vring import PAIR, tails
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig
from artlite.moe import route as MR
from artlite.moe.steps import CP_BYTES, MOE_BLOCKS
from artlite.operators import cdo
from artlite.operators.design import COLS
from artlite.overlay import Sequence, WeightBuffer

from .decode import InklingDecodeOperator
from .image import EXPERT_UNITS, InklingImage

KINDS = ("swa", "full")
FORMATS = frozenset({"Q8_0", "BFP16", "NVFP4", "Q8_0H"})  # (the formats the overlay runs so far)
PHASES = ("q", "k", "v", "att", "o")  # (a visit's attention phases)
FFN_PHASES = ("router", "moe")  # (its FFN's: the router's GEMV, the MoE operator phase)
RING = 9216  # (a raw ring's bytes: one A object, its first 8 KB the ring)
WRAP = 32  # (groups of a sliding-window cache's ring: 2048 positions, its window's 17 groups and the next one a step's drafts
# write; decode steps only, no prefill chunks to hold)
E_ROWS = 17  # (E_rev's rows: a zero row, then the 16 of the table)
BLOCK_FLAGS = dict.fromkeys(("sink", "kd", "vd"), B_PAIR | B_F32) | {"k": B_F32, "v": B_F32}
HIST_ROWS = 8  # (a history window's rows: 3 + S <= 7 written, a read's object past its 3)
KV_PAD = 8192  # positions past the caches (the last visit's): the prefill's slices of a cache (``prefill.InklingPrefill``)
CTL = ("q", "k", "v", "ffn")  # (a loop visit's controlled streams, ``InklingConfig.CTRL_STREAMS``)
# (the image's regions a phase reads, in the A stream's order)
PHASE_REGIONS = {"q": ("q_norm", "q"), "k": ("k_norm", "k"), "v": ("v_norm", "v"), "o": ("o", "inkf", "inkfq"),
                 "router": ("router_norm", "router")} | \
    {f"d{c}{h}": (f"d{c}{h}_norm", f"d{c}{h}") for c in CTL for h in "ab"} | {f"b{c}": (f"ctl_{c}",) for c in CTL} | \
    {"lm": ("lm_norm", "lm")}
BLENDED = {"q": 5, "k": 6, "v": 7, "ffn": 5}  # (the state array's slots of a loop visit's blended inputs)
BANK0 = 8  # (the state array's first bank slot: bank w's states 8 + 2 w, 9 + 2 w)
SA_SLOTS = BANK0 + 2 * INK_BANKS  # (the state array's slots)
UNIT_KINDS = (512, 128)  # (the expert units' shapes, their templates' kinds: ``moe.route.UNIT_KINDS``)


def cur_states(cfg: InklingConfig) -> list[int]:
    """Per visit the state n of the state array that holds cur when its attention input is mixed (states 0 .. n - 1: the
    completed block states; a block visit's attention output starts state n + 1): ``ref.Model.walk``'s blocks."""
    n, out, prelude = 0, [], 0
    for v in cfg.visits:
        if v.loop >= 0 and v.layer == cfg.core_start:  # (a pass restarts from the prelude's output and block states)
            if v.loop == 0:
                prelude = n
            n = prelude
        out.append(n)
        n += int(v.block)
    return out


def psums(states: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The partial sums object's payload a producer writes for a mix ((COLS, 4 slots, 32) fp32, ``protocol.F_COEF``: per
    column the states' sums of squares over its channels at lanes b, their dots with the queries at 8 + 8 j + b) of states
    (n, S, hidden) and queries (J, hidden)."""
    n, S, H = states.shape
    h8 = H // COLS
    out = np.zeros((COLS, 4, 32), np.float32)
    for c in range(COLS):
        x = states[..., h8 * c:h8 * (c + 1)].astype(np.float64)
        q = queries[:, h8 * c:h8 * (c + 1)].astype(np.float64)
        out[c, :S, :n] = (x * x).sum(-1).T
        for j in range(len(q)):
            out[c, :S, 8 + 8 * j:8 + 8 * j + n] = (x @ q[j]).T
    return out


class Subs(tuple):
    """A step's attention sub-steps (first slot, slots) and its parts' slots (``parts``; one part: a step of one sequence):
    the key of its transaction, which takes a history window per part (the residual stream's step, ``_o_phase``)."""

    parts: tuple[int, ...]

    def __new__(cls, subs: tuple, parts: tuple[int, ...]) -> Self:
        t = super().__new__(cls, subs)
        t.parts = tuple(parts)
        return t

    def __eq__(self, other: object) -> bool:
        return tuple.__eq__(self, other) and getattr(other, "parts", None) == self.parts

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((tuple(self), self.parts))


def pow2(n: int) -> int:
    """The state objects of a k range for a mix of n states (``protocol.M_NR``: 1, 2, 4 or 8)."""
    return 1 << (n - 1).bit_length()


class InklingOverlay(AttentionOverlay):
    """An Inkling GGUF model's visits on its decode overlay (module docstring). ``visits``: the visits to run (default all;
    tests run slices), ``max_positions``: the caches' capacity, ``kv_k``: the caches' K precision, ``ffn``: the visits'
    FFNs (False: their attention and o phases only), ``paged``: the formats whose kernels are pages (default: every
    format, which the main and helper cores' memory does not hold with the residual stream's work), ``wrap``: the
    sliding-window caches' ring (groups; 0: linear; default ``WRAP`` past that many groups of capacity)."""

    CONFIG, IMAGE = InklingConfig, InklingImage
    CHAIN = 1
    SEQ = ("KV",)  # (a sequence's caches, raw rings, history windows and open records)
    OPS = ("att", "moe")
    LOADS = INK_STATES + 2 * INK_BANKS  # (F_LOADR objects per slot at most: the states, the banks)

    def __init__(self, gguf: GGUFFile, *, visits: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "hilo", weights: WeightBuffer | None = None, ffn: bool = True,
                 paged: tuple[str, ...] | None = None, wrap: int | None = None, sequences: int = 1):
        cfg = InklingConfig.from_gguf(gguf)
        vs = list(range(len(cfg.visits))) if visits is None else list(visits)
        self.vis = {v: cfg.visits[v] for v in vs}
        self.ffn = ffn
        # (the FFN's combine reads the next visit's queries: its layer's "inkq", the output's after the last visit)
        self.after = vs[-1] + 1
        # (the whole model's last visit: then the output's mix, its norm and the LM head)
        self.lm_ok = ffn and self.after == len(cfg.visits)
        layers = {x.layer for x in self.vis.values()} | \
            ({cfg.visits[self.after].layer} if ffn and not self.lm_ok else set())
        image = None if weights is not None else InklingImage(gguf, cfg, layers=sorted(layers), vocab=vocab)
        super().__init__(gguf, layers=vs, vocab=vocab, kv_k=kv_k, weights=weights, image=image, sequences=sequences)
        img = self.image
        regions = ("q", "k", "v", "o") + (("router",) if ffn else ())
        fmts = {b.fmt for L in img.layers for name in regions for b in img.region(L, name).blocks}
        if ffn:
            fmts |= {f for r in img.slabs for b in r.blocks for f in b.formats}
        if self.lm_ok:
            fmts |= set(img.region(-1, "lm").blocks[0].formats)
        if fmts - FORMATS:
            raise NotImplementedError(f"{', '.join(sorted(fmts - FORMATS))} tiles")
        self.hd, self.q_heads = cfg.head_dim, cfg.q_per_kv
        moe = (cfg.n_experts, cfg.n_experts_used) if ffn else None
        self.at = at = Attn(hds=(self.hd,), rot=(0,), chain=1, heads=self.q_heads, kp=self.kp, v_norm=True,
                            eps=cfg.rms_norm_eps, a_bytes=self.T, ink=True, moe=ffn,
                            acc=4 * (SLOTS * self.H + 32) if ffn else 0)
        self.op = InklingDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                        eps=cfg.rms_norm_eps, attn=at, moe=moe,
                                        paged=paged,
                                        situ=(cfg.situ_beta, cfg.situ_linear_beta))
        # (the experts' gate / up: tail groups of 16-row tiles or of 8-row ones in pairs, whole [8 gate | 8 up] windows)
        assert all(r.qt and r.qg % 2 == 0 for r in img.slabs if r.name.startswith("exp_gu")) and img.gu_rows() <= {8, 16}
        self.pg = at.pg[self.hd]
        self.guard = -(-(cfg.sliding_window - 1) // self.pg)
        full = -(-max_positions // self.pg)
        self.max_positions = full * self.pg
        wrap = (WRAP if full > WRAP else 0) if wrap is None else wrap
        # (a ring holds the window's groups and the group after them: a step's drafts write there, kept positions' windows
        # never)
        assert not wrap or self.guard + 2 <= wrap < full, (wrap, self.guard, full)
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=self.guard + (wrap or full), guard=self.guard,
                                window=cfg.sliding_window, wrap=wrap),
                      CacheKind(hd=self.hd, pg=self.pg, groups=full))
        self.uses = {v: CacheUse(kind=KINDS.index(x.kind), source=v, ring=v) for v, x in self.vis.items()}
        self.period = PAIR
        self.K = img.region(img.layers[0], "q").K_in  # (an attention input's K: the hidden)
        self.h8 = self.H // COLS
        self.HIST = HIST_ROWS * 4 * self.h8  # (a history window's bytes)
        self.nst = dict(zip(range(len(cfg.visits)), cur_states(cfg)))
        self.n0 = self.nst[vs[0]]  # (the first visit's cur: its states 0 .. n0 come from the host)
        self._hbase = [-3] * sequences  # (per sequence: its history windows' first position)
        self._sa_sp = 0  # (the packed slots of the state array's layout whose padding is zero: ``write_states``)
        self._eskip: dict[tuple[int, int], tuple] = {}  # ((visit, slot) -> its raw K / V objects' EGRP words: ``_set_parts``)
        if self.lm_ok and img.screen:  # (the screened head's host side: the exact logits of its candidates)
            norm = np.asarray(gguf.tensor("output_norm.weight").dequantize(), np.float32).reshape(-1)
            self.screen = Screen(gguf, img.lm, norm, cfg.rms_norm_eps, min(self.vocab, cfg.vocab_size))
        self._plan()

    def layer(self, v: int) -> int:
        """The physical layer of visit v (-1: the LM head's)."""
        return -1 if v < 0 else self.vis[v].layer

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = False) -> list[tuple[int, str]]:
        out = []
        for v in self.layers:
            msg = self.vis[v].messages  # (a loop visit after the first pass: its streams' controllers)
            ctl = [n for c in "qkv" for n in (f"d{c}a", f"d{c}b", f"b{c}")] if msg else []
            out += [(v, n) for n in ctl + list(PHASES)]
            if self.ffn:
                out += [(v, n) for n in (["dffna", "dffnb", "bffn"] if msg else []) + list(FFN_PHASES)]
        if lm and self.lm_ok:
            out.append((-1, "lm"))
        return out

    def _mix(self, v: int, name: str) -> tuple[int, int, int, tuple | None]:
        """Visit v's F_STATES phase ``name``: (its states' first slot in the state array, M_NR, the coefficient set to mix
        by: -1 one state, coefficient 1; F_COEF's (partial sums object, its queries, the mix's states) or None). A loop
        visit's GEMV phases read their blended inputs, its controllers' down halves the mix (states 0 .. n) and the
        message (the bank its pass reads); a set's F_COEF writes it and the stream's ones after it."""
        if v < 0:  # (the LM head: the output's mix of the last visit's states, one query)
            n = self.nst[self.layers[-1]] + int(self.vis[self.layers[-1]].block)
            return 0, pow2(n + 1), 0, ((self.after, "a"), 1, n + 1)
        vis = self.vis[v]
        n = self.nst[v] + (int(vis.block) if name in ("router", "dffna") else 0)
        j = 0 if vis.kind == "swa" else None
        if name in ("q", "k", "v", "router"):
            stream = "ffn" if name == "router" else name
            if vis.messages:
                return BLENDED[stream], 1, -1, None
            if name == "router":
                return 0, pow2(n + 1), 0, ((v, "f"), 1, n + 1)
            if not vis.queries:
                return 0, 1, -1, None
            return 0, pow2(n + 1), "qkv".index(name) if j is None else j, ((v, "a"), vis.queries, n + 1) if name == "q" else None
        stream, half = name[1:-1], name[-1]
        if half == "a":
            if stream == "ffn":
                return 0, pow2(n + 1), 0, ((v, "f"), 1, n + 1)
            return (0, pow2(n + 1), "qkv".index(stream) if j is None else j,
                    ((v, "a"), vis.queries, n + 1) if stream == "q" else None)
        b = BANK0 + 2 * ((vis.loop - 1) % 2)
        if stream == "ffn":
            return b, 2, 6, ((v, "mb"), 1, 2)
        return b, 2, 3 + "qkv".index(stream), ((v, "ma"), 3, 2) if stream == "q" else None

    def queries_after(self, v: int) -> tuple[int, int]:
        """(layer, queries) of the mix that visit v's FFN output feeds: the next visit's attention input, or after the last
        visit the output's (layer -1)."""
        nxt = v + 1
        if nxt >= len(self.cfg.visits):
            return -1, 1
        x = self.cfg.visits[nxt]
        return x.layer, x.queries

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        self.toks = (take("ACT", len(KINDS) * SLOTS * XB),)
        self.loadr = take("ACT", COLS * self.LOADS * SLOTS * XB)  # ([column][state b][slot s] F_LOADR objects)
        # the state array ([state][k block][slot sp][8] fp32, after the XB - XPAY bytes its objects' fill reads before
        # it); per visit: the q, k, v phases' metadata objects, the partial sums objects of its attention's and FFN's
        # mixes, the attention region [q objects | raw K / V objects] per KV head, o's input
        self.sa = take("ACT", XB + SA_SLOTS * self.state_bytes(SLOTS)) + XB
        self.meta = {(v, n): take("ACT", XB) for v in self.layers for n in ("q", "k", "v")}
        self.ps = {(v, m): take("ACT", XB) for v in self.layers for m in "af"}
        # (the message mixes' partial sums of a loop visit after the first pass: queries 0 .. 2, 3)
        self.ps |= {(v, m): take("ACT", XB) for v in self.layers if self.vis[v].messages for m in ("ma", "mb")}
        # (the visit after the last: its attention mix's, or the output's, of the last FFN's output)
        self.ps |= {(self.after, m): take("ACT", XB) for m in ("a", "ma", "mb")}
        if self.lm_ok:
            self.meta[(-1, "lm")] = take("ACT", XB)
        # per loop visit after the first pass and stream: its controller's down halves' metadata objects, their g object
        # (and the dumps of their C objects' other strips past it: ``_gemv``), its blend's metadata object
        msg = [v for v in self.layers if self.vis[v].messages]
        self.meta |= {(v, f"d{c}{h}"): take("ACT", XB) for v in msg for c in (CTL if self.ffn else CTL[:3]) for h in "ab"}
        self.gobj = {(v, c): take("ACT", XB + 8192) for v in msg for c in CTL}
        self.bmeta = {(v, c): take("ACT", XB) for v in msg for c in CTL}
        self.att_reg = {(v, h): take("ACT", 2 * SLOTS * XB) for v in self.layers for h in range(self.kv_heads)}
        self.inp = {(v, "o"): [InputRegion(take("ACT", region_bytes(3 * 1024)), 0, 3 * 1024)] for v in self.layers}
        self.rel = {L: take("ACT", E_ROWS * 2 * 1280) for L in self.image.layers}  # (E_rev per layer, ``ink_rel``)
        self._plan_caches(self.layers, pad=KV_PAD // self.pg * self.G * self.T)  # (the prefill's slices past the last cache)
        self.raw = {v: self._take("KV", self.kv_heads * RING) for v in self.layers}
        self.hist = {(v, i): self._take("KV", COLS * self.HIST) for v in self.layers for i in range(2)}  # (i: 0 attention's)
        if self.ffn:  # per visit the router's metadata object, its route object (and the router drain's other halves:
            # ``_gemv``), per column its routing's objects and expert object; the combine's objects (PART)
            self.meta |= {(v, "router"): take("ACT", XB) for v in self.layers}
            self.xch = {v: take("ACT", XB + 2048) for v in self.layers}
            self.cp = {v: take("ACT", COLS * (CP_BYTES + XB)) for v in self.layers}
            self.comb = {v: take("ACT", COLS * XB) for v in self.layers}  # (per visit its combine's metadata objects)
            self.part = take("ACT", COLS * self.part_col)
        self._sizes()

    @property
    def part_col(self) -> int:
        """PART's bytes per column: an object's header, then per slot and third of the column's slice the 8
        columns' partials of it ([slot][third][column][96] fp32 on the slot's grid: one object's payload each,
        ``decode.design``'s "MoE")."""
        return XB + 3 * SLOTS * XPAY

    def state_bytes(self, sp: int) -> int:
        """A state's bytes in the state array of a step of sp packed slots: its k ranges' (the hidden's whole objects of
        1024 / sp k: ``protocol.F_STATES``; zero past the hidden)."""
        return -(-4 * sp * self.K // XPAY) * XPAY

    def state_k(self, sp: int) -> int:
        """A state's k in the state array of a step of sp packed slots (the hidden's, padded to whole k ranges)."""
        return self.state_bytes(sp) // (4 * sp)

    def raw_ring(self, v: int, h: int) -> int:
        """Byte offset in KV of visit v's KV head h's raw ring."""
        return self.raw[v] + h * RING

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.ffn:  # (the control route's registers free: ``moe.route.check_route``)
            MR.check_route(cdo.values(os.path.dirname(self.op._program.insts_path)), range(COLS))
        super()._load()
        act, g, cfg = self.dev.view("ACT"), self.gguf, self.cfg
        act[self.loadr:self.loadr + COLS * self.LOADS * SLOTS * XB] = 0
        for v in self.layers if self.ffn else ():  # (the routing's tables after the logits: ``moe.route.render(units=)``)
            b, E = f"blk.{self.layer(v)}.", cfg.n_experts
            bias = self._expert_vec(b + "exp_probs_b.bias").astype(np.float32)
            dsc = self._expert_vec(b + "ffn_down_exps.scale") * cfg.expert_weights_scale
            gsc = self._expert_vec(b + "ffn_gate_up_exps.scale").astype(np.float32)
            pay = act[self.xch[v] + 4 * META:self.xch[v] + XB].view(np.float32)
            pay[4 * E:5 * E], pay[5 * E:6 * E], pay[6 * E:7 * E] = bias, dsc.astype(np.float32), gsc
            pay[7 * E:7 * E + 4 * len(EXPERT_UNITS)].view(np.int32)[:] = self.unit_table(self.layer(v)).reshape(-1)
        for L, off in self.rel.items():
            E = np.asarray(g.tensor(f"blk.{L}.attn_rel_bias.weight").dequantize(), np.float32).reshape(-1, cfg.rel_length).T
            act[off:off + E_ROWS * 2 * 1280] = ink_rel(E).view(np.uint8).reshape(-1)
        for s in range(1, self.sequences):  # (the other sequences' convs' state: zero, as ``reset`` leaves the selected one's)
            kv = self.dev.view(self.seq_name("KV", s))
            for o, n in self._conv_regions():
                kv[o:o + n] = 0
            self.dev.flush(self.seq_name("KV", s))

    def reset(self) -> None:
        """(and the raw rings and history windows: the convs' history before position 0 is zero)"""
        super().reset()
        self._eskip.clear()
        self._hbase[self.seq] = -3
        if self.dev is not None:
            kv = self.dev.view("KV")
            for v in self.layers:
                kv[self.raw[v]:self.raw[v] + self.kv_heads * RING] = 0
                for i in range(2):
                    kv[self.hist[(v, i)]:self.hist[(v, i)] + COLS * self.HIST] = 0
            self.dev.flush("KV")

    def _conv_regions(self) -> list[tuple[int, int]]:
        """(offset, bytes) in KV of each visit's raw rings and history windows: the convs' state."""
        return [(self.raw[v], self.kv_heads * RING) for v in self.layers] + \
            [(self.hist[(v, i)], COLS * self.HIST) for v in self.layers for i in range(2)]

    def snapshot(self, position: int) -> Snapshot:
        """(and the convs' state: the raw rings, the history windows and their first position)"""
        snap = super().snapshot(position)
        kv, parts = self.dev.view("KV"), []
        for o, n in self._conv_regions():
            self.dev.fetch("KV", o, n)
            a = host_bytes((n,))
            a[:] = kv[o:o + n]
            parts.append(a)
        snap.extra["conv"] = (parts, self._hbase[self.seq])
        return snap

    def restore(self, snapshot: Snapshot) -> None:
        """(the convs' state too)"""
        super().restore(snapshot)
        parts, self._hbase[self.seq] = snapshot.extra["conv"]
        kv = self.dev.view("KV")
        for (o, n), a in zip(self._conv_regions(), parts, strict=True):
            kv[o:o + n] = a
            self.dev.flush("KV", o, n)

    def _ran(self, p: int, S: int, seq: int | None = None) -> None:
        """(and the history windows: rows p - 3 .. p + S - 1)"""
        super()._ran(p, S, seq)
        self._hbase[self.seq if seq is None else seq] = p - 3

    def _ring_state(self, seq: int | None = None) -> tuple:
        return super()._ring_state(seq) + (self._hbase[self.seq if seq is None else seq],)

    def _expert_vec(self, name: str) -> np.ndarray:
        """A layer's per-expert vector (n_experts,) float64; a scale the file has not (a layer of unscaled experts) ones."""
        if name not in self.gguf:
            return np.ones(self.cfg.n_experts)
        return np.asarray(self.gguf.tensor(name).dequantize(), np.float64).reshape(-1)

    def unit_table(self, L: int) -> np.ndarray:
        """Layer L's units (``EXPERT_UNITS``) for the routing (len, 4) int32: per unit the offset of its bytes in an
        expert's run, its work (channels / 128), its A objects (gate / up and down tiles), its templates' kind (0: 512
        channels, 1: 128)."""
        img, off = self.image, self.image.unit_offsets(L)
        out = []
        for u, (_, n) in enumerate(EXPERT_UNITS):
            gu, dn = img.region(L, f"exp_gu{u}"), img.region(L, f"exp_dn{u}")
            out.append((off[u], n // 128, gu.n_t + dn.n_t, UNIT_KINDS.index(n)))
        return np.array(out, np.int32)

    def _attention_scale(self, L: int) -> tuple[int, int]:
        return score_bits(1 / math.sqrt(self.hd))

    def _load_attention(self, L: int) -> None:
        """(and Inkling's: the F_ATT objects' words of the layer's kind and extent, the clip scales after their payload,
        the conv taps after the raw K / V objects')"""
        super()._load_attention(L)
        act, g, cfg = self.dev.view("ACT"), self.gguf, self.cfg
        P, nq, hd = self.layer(L), self.q_heads, self.hd
        b = f"blk.{P}."
        qc = np.asarray(g.tensor(b + "attn_q_clip.weight").dequantize(), np.float32).reshape(-1, hd)
        rc = np.asarray(g.tensor(b + "attn_rel_clip.weight").dequantize(), np.float32).reshape(-1, cfg.rel_length)
        kt = np.asarray(g.tensor(b + "attn_k_conv.weight").dequantize(), np.float32).reshape(-1, cfg.conv_taps).T
        vt = np.asarray(g.tensor(b + "attn_v_conv.weight").dequantize(), np.float32).reshape(-1, cfg.conv_taps).T
        full = cfg.kinds[P] == "full"
        for h in range(self.kv_heads):
            heads = slice(nq * h, nq * (h + 1))
            qcb = qc[heads].astype(bfloat16).view(np.uint8).reshape(-1)
            rcb = rc[heads].astype(bfloat16).view(np.uint8).reshape(-1)
            taps = np.ascontiguousarray(np.concatenate([kt[:, hd * h:hd * (h + 1)], vt[:, hd * h:hd * (h + 1)]]).astype(bfloat16)).view(np.uint8)
            for a in range(SLOTS):
                o = self.att_reg[(L, h)] + a * XB
                m = act[o:o + 4 * META].view(np.int32)
                m[M_AFULL], m[M_AEXT] = int(full), cfg.rel_extent
                act[o + 4 * META + 2 * INK_QC:][:qcb.size] = qcb
                act[o + 4 * META + 2 * INK_RC:][:rcb.size] = rcb
                k = self.att_reg[(L, h)] + (SLOTS + a) * XB
                act[k + 4 * META + INK_TAPS:][:taps.size] = taps.reshape(-1)

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """(the q, k, v phases': the depth mix of the visit's states 0 .. n (fp32 state objects; the q phase's partial sums
        object: the coefficient sets, one per query), visit 0's normed embedding alone; o's: F_INK)"""
        out = []
        ranges = self.state_bytes(packed_slots(S)) // XPAY
        for (v, name), off in self.meta.items():  # (every phase on states: ``_mix``)
            flags = MOE_BLOCKS if name == "router" else BLOCK_FLAGS if name in ("q", "k", "v") else \
                None if name == "lm" else {name: B_PAIR | B_F32}
            gr = mp.groups(self.image.region(self.layer(v), name), F_NORM, S, block_flags=flags)
            assert len(gr) == 1
            m = gr[0].meta.copy()
            _, nr, cset, coef = self._mix(v, name)
            m[0] |= F_STATES
            m[M_NR], m[M_NX], m[M_CSET] = nr, ranges * nr, cset
            if coef is not None:
                m[0] |= F_COEF
                m[M_NQ], m[M_NMIX] = coef[1:]
            out.append((off, m.view(np.uint8)))
        for (v, c), off in self.bmeta.items():  # (a blend: the g object, its rows' A objects)
            vis = self.vis[v]
            j, mset = self._mix(v, f"d{c}a")[2], self._mix(v, f"d{c}b")[2]
            n = self.nst[v] + (int(vis.block) if c == "ffn" else 0)
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS], m[M_NX], m[M_NT] = F_MAIN | F_BLEND, S, 1, 2 * BLEND_OBJECTS
            m[M_BLEND] = j | mset << 4 | n << 8 | ((vis.loop - 1) % 2) << 12
            out.append((off, m.view(np.uint8)))
        for v in self.layers:
            rg = self.inp[(v, "o")][0]
            gr = mp.groups(self.image.region(self.layer(v), "o"), F_RESID | F_INK, S)
            assert len(gr) == 1
            m = gr[0].meta.copy()
            m[M_INKN] = self.nst[v] | int(self.vis[v].block) << 4 | 1 << 12  # (the FFN mix's query: inkfq)
            out.append((rg.off, m.view(np.uint8)))
        for v in self.layers if self.ffn else ():
            out += self._ffn_meta(v, S, ranges)
        return out

    def _ffn_meta(self, v: int, S: int, ranges: int) -> list[tuple[int, np.ndarray]]:
        """Visit v's FFN objects for S slots: the router's metadata (the FFN mix of states 0 .. n1, one query: F_COEF), the
        route object's words, the expert objects' templates (gate / up and down per kind, the units' kinds), the
        combine's metadata objects (F_COMB | F_INK: then the residual stream's step with ``inkc`` and the next mix's
        queries)."""
        L, img = self.layer(v), self.image
        n1 = self.nst[v] + int(self.vis[v].block)
        out = []
        gu0 = img.region(L, "exp_gu0")
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_NS] = F_MAIN | F_ROUTE, S
        a = self.dev.address("W", gu0.offset)
        m[M_RBASE:M_RBASE + 2] = np.array([a & 0xFFFFFFFF, a >> 32], np.uint32).view(np.int32)
        m[M_RSTR] = gu0.col_stride
        out.append((self.xch[v], m.view(np.uint8)))
        tpl = []
        for k in range(len(UNIT_KINDS)):
            u = next(i for i, (_, n) in enumerate(EXPERT_UNITS) if n == UNIT_KINDS[k])
            tpl += [mp.groups(img.region(L, f"exp_gu{u}"), F_GU, S)[0].meta, mp.groups(img.region(L, f"exp_dn{u}"), 0, S)[0].meta]
        kinds = np.zeros(16, np.int32)
        kinds[:len(EXPERT_UNITS)] = [UNIT_KINDS.index(n) for _, n in EXPERT_UNITS]
        tb = np.concatenate([*tpl, kinds]).view(np.uint8)
        for c in range(COLS):  # (the expert object's payload)
            out.append((self.cp[v] + c * (CP_BYTES + XB) + CP_BYTES, tb))
        _, nq = self.queries_after(v)
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_NS], m[M_NX] = F_MAIN | F_COMB | F_INK, S, 3 * S
        m[M_INKN] = n1 | nq << 12 | self._bank_words(v)
        out += [(self.comb[v] + c * XB, m.view(np.uint8)) for c in range(COLS)]
        return out

    def _weight_run(self, seq: Sequence, phases: list[tuple[int, str]], *, lead: tuple[tuple[int, str], ...] = (),
                    blocks: slice = slice(None), **kw: object) -> None:
        """(a visit's phases read its physical layer's regions, ``PHASE_REGIONS``, after ``lead``'s: (layer, region) the
        combine reads first, its taps and the next mix's queries; each stretch of adjacent regions one run of the
        column-major image per column, ``TileImage.runs``)"""
        img = self.image
        regions = [img.region(*r) for r in lead]
        regions += [img.region(self.layer(v), r) for v, n in phases if n in PHASE_REGIONS for r in PHASE_REGIONS[n]]
        if not regions:
            return
        index = {id(r): i for i, r in enumerate(img.regions)}
        runs, first = [], 0
        for i in range(1, len(regions) + 1):
            if i == len(regions) or index[id(regions[i])] != index[id(regions[i - 1])] + 1:
                runs += img.runs(regions[first], regions[i - 1])
                first = i
        for r, nb in runs[blocks]:
            for c in range(COLS):
                seq.fill("A", c, self.dev.address("W", r.offset + c * r.col_stride), nb)

    def _gemv(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """A visit's q, k or v phase: its metadata object and its input state's objects (X), its outputs (C) into the
        attention region's q or raw K / V objects."""
        v, name = phase
        if name == "o":
            self._o_phase(seq, st, v)
            return
        if name[0] == "b":
            self._blend(seq, st, v, name[1:])
            return
        dev, S, sp = self.dev, st.S, packed_slots(st.S)
        ST = self.state_bytes(sp)
        first, nr, _, coef = self._mix(v, name)
        out = {"d": f"g{name[1:-1]}", "r": "route", "l": "logits"}.get(name[0], "att")
        self._sync(seq, {f"sa{i}" for i in range(first, first + nr)} | ({f"ps{coef[0]}"} if coef else set()), {out})
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.meta[phase]), XB)
            if coef is not None:  # (the partial sums object of its mix)
                seq.fill("X", c, dev.address("ACT", self.ps[coef[0]]), XB)
            seq.fill("X", c, dev.address("ACT", self.sa + first * ST - (XB - XPAY)), nr * XB, (XB // 8, 2, XB // 8, ST // 4),
                     repeat=ST // XPAY, iteration=(XPAY // 4, ST // XPAY))
        if name == "lm":
            for c in range(COLS):
                self._logits_out(seq, c, S)
            return
        if name[0] == "d":  # (a controller's down half: per slot 8 fp32 of g, the first quarter of a C object: the other
            # three past the g object)
            h = "ab".index(name[-1])
            for c in range(COLS):
                seq.drain("C", c, dev.address("ACT", self.gobj[(v, name[1:-1])] + 4 * META + 128 * (8 * h + c)), S * 128,
                          (8, 4, 512, 8))
            return
        if name == "router":  # (per slot 16 fp32 logits, the first half of a C object: the other half past the route
            # object, ``xch``)
            for c in range(COLS):
                seq.drain("C", c, dev.address("ACT", self.xch[v] + 4 * META + 64 * c), S * 128,
                          (16, 2, (XB - 4 * META) // 4, 128))
            return
        r = self.image.region(self.layer(v), name)
        n = -(-self.outputs(r) // C_LEN)  # (C objects per slot)
        for c in range(COLS):  # (object j of slot s -> the slot's object, 128 j bytes into its payload)
            obj = self.att_reg[(v, c)] + (SLOTS * XB if name != "q" else 0) + 4 * META
            off = {"q": 0, "k": 0, "v": 4 * 160}[name]
            seq.drain("C", c, dev.address("ACT", obj + off), n * S * 2 * C_LEN, (C_LEN // 2, S, XB // 4, C_LEN // 2))

    def _blend(self, seq: Sequence, st: StepPlan, v: int, c: str) -> None:
        """Visit v's blend of stream c (``protocol.F_BLEND``): its metadata and g object (X), the rows (A: the weight run),
        the blended slice (C) into the state array's slot of the stream (``BLENDED``)."""
        dev, sp = self.dev, packed_slots(st.S)
        self._sync(seq, {f"g{c}"}, {f"sa{BLENDED[c]}"})  # (g's halves in DDR)
        for col in range(COLS):
            seq.fill("X", col, dev.address("ACT", self.bmeta[(v, c)]), XB)
            seq.fill("X", col, dev.address("ACT", self.gobj[(v, c)]), XB)
        row = 4 * self.h8
        for col in range(COLS):
            seq.drain("C", col, dev.address("ACT", self.sa + BLENDED[c] * self.state_bytes(sp) + col * row * sp), row * sp)

    def _o_phase(self, seq: Sequence, st: StepPlan, v: int) -> None:
        """Visit v's o (F_RESID | F_INK): its input (the attention's o), its post conv's history window (X, patched per
        step: ``_set_parts``); out (C) the window (the 3 rows read, the step's), cur's slice of the state array, the
        column's partial sums of the FFN's mix."""
        dev, S, sp = self.dev, st.S, packed_slots(st.S)
        n1 = self.nst[v] + int(self.vis[v].block)
        self._sync(seq, {"o"}, {f"sa{n1}", f"ps{(v, 'f')}", "hist"})
        self._inputs(seq, self.inp[(v, "o")][0], S)
        self._windows(seq, st, v, 0)
        row = 4 * self.h8
        for c in range(COLS):
            seq.drain("C", c, dev.address("ACT", self.sa + n1 * self.state_bytes(sp) + c * row * sp), row * sp)
            seq.drain("C", c, dev.address("ACT", self.ps[(v, "f")] + (XB - XPAY) + 512 * c), 512)

    def _windows(self, seq: Sequence, st: StepPlan, v: int, i: int, *, fills: bool = True, drains: bool = True) -> None:
        """Visit v's history window i (0: the attention's post conv, 1: the FFN's) of each part of the step (its sequence's,
        in its KV: the residual stream's step per part): per column its X object (patched per step: ``_set_parts``) and
        its drain, the 3 rows read and the part's (``PartBD``)."""
        dev, row = self.dev, 4 * self.h8
        parts = st.subs.parts
        hf = st.__dict__.setdefault("hfills", [])
        for c in range(COLS) if fills else ():
            for q in range(len(parts)):
                o = self.hist[(v, i)] + c * self.HIST
                hf.append((seq.fill("X", c, dev.address("KV", o) - (XB - XPAY), XB), v, i, c, q))
        for c in range(COLS) if drains else ():
            for q, n in enumerate(parts):
                o = self.hist[(v, i)] + c * self.HIST
                st.part_bds.append(PartBD(seq.drain("C", c, dev.address("KV", o), (3 + n) * row)[0], q, o, (3 + n) * row))

    def _operator(self, seq: Sequence, st: StepPlan, L: int, name: str, weights: list[tuple[int, str]]) -> None:
        if name == "moe":
            self._moe(seq, st, L, weights)
        else:
            super()._operator(seq, st, L, name, weights)
        seq.ink_dirty = None  # (its outputs: the next phase syncs)

    @staticmethod
    def _sync(seq: Sequence, reads: set[str], writes: set[str]) -> None:
        """Before a GEMV phase's transfers: a sync only when it reads what a phase since the last one writes (``reads``,
        ``writes``: the state array's slots "sa<i>", partial sums objects "ps<key>", g objects "g<stream>", ...; after an
        operator phase, which syncs itself, all of them). Its fills queue behind the phase before, whose outputs drain
        first on C: the cores take the phases in order."""
        dirty = getattr(seq, "ink_dirty", None)
        if dirty is None or dirty & reads:
            seq.sync()
            dirty = set()
        seq.ink_dirty = dirty | writes

    def _begin(self, seq: Sequence) -> None:
        """(an FFN's MoE: the control packets' route, the routing's shim locks)"""
        if self.ffn:
            for c in range(COLS):
                MR.control_route(seq.txn, c)
                for lk in (MR.LOCK_CP, MR.LOCK_DONE):
                    seq.txn.write32(txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * lk), 0)

    def _moe(self, seq: Sequence, st: StepPlan, v: int, weights: list[tuple[int, str]]) -> None:
        """Visit v's MoE after its router: the route object (the routing on every column's helper, its control packets'
        round trip into the shim: the expert slabs' BDs the array pushes), the expert object (its units' sub-phases),
        the A run from the combine's taps on (the next visit's queries, the GEMV phases after it: ``weights``), the
        experts' partials into PART, the combine (F_COMB | F_INK: the partials into a zeroed res, then the residual
        stream's step over the FFN's history window); out the window, cur's slice, the next mix's partial sums."""
        dev, S, sp = self.dev, st.S, packed_slots(st.S)
        L = self.layer(v)
        seq.sync()  # (the logits in the route object)
        for c in range(COLS):
            cp = self.cp[v] + c * (CP_BYTES + XB)
            seq.fill("X", c, dev.address("ACT", self.xch[v]), XB)
            seq.drain("C", c, dev.address("ACT", cp), CP_BYTES, release=(MR.LOCK_CP, 1))
            seq.fill("X", c, dev.address("ACT", cp), 4 * MR.PACKET_WORDS, packet=False, acquire=(MR.LOCK_CP, 1))
            seq.device_task("A", c)
        seq.sync()
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.cp[v] + c * (CP_BYTES + XB) + CP_BYTES - 4 * META), XB)
        for c in range(COLS):  # (the array's packets are through: later host pushes on A keep their order)
            lock = txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * MR.LOCK_DONE)
            seq.txn.maskpoll32(lock, 1, 0x3F)
            seq.txn.write32(lock, 0)
        la, _ = self.queries_after(v)
        self._weight_run(seq, weights, lead=((L, "inkc"), (la, "inkq")))
        pc = self.part_col
        for c in range(COLS):  # (acc's objects per destination column, slot, third, 3 of 32 rows -> PART)
            seq.drain("C", c, dev.address("ACT", self.part + XB + 384 * c), S * 3 * 384,
                      (96, 3, XPAY // 4, 3 * XPAY // 4), COLS, (pc // 4, COLS))
        seq.sync()
        for c in range(COLS):  # (the partials are in DDR: the experts' tasks are complete)
            seq.device_done("A", c)
            seq.fill("X", c, dev.address("ACT", self.comb[v] + c * XB), XB)  # (the visit's metadata, the partials)
            seq.fill("X", c, dev.address("ACT", self.part + c * pc + XPAY), 3 * S * XB, DIMS_OBJ)
        self._windows(seq, st, v, 1, drains=False)
        n1 = self.nst[v] + int(self.vis[v].block)
        row = 4 * self.h8
        bk, nxt = self.bank_of(v), v + 1 < len(self.cfg.visits) and self.cfg.visits[v + 1].messages
        ST = self.state_bytes(sp)
        self._windows(seq, st, v, 1, fills=False)
        for c in range(COLS):
            seq.drain("C", c, dev.address("ACT", self.sa + n1 * ST + c * row * sp), row * sp)
            seq.drain("C", c, dev.address("ACT", self.ps[(v + 1, "a")] + (XB - XPAY) + 512 * c), 512)
            if bk is not None:  # (cur again: its bank's copy)
                seq.drain("C", c, dev.address("ACT", self.sa + (BANK0 + 2 * bk[0] + bk[1]) * ST + c * row * sp), row * sp)
            if nxt:  # (the next visit's message mixes' partial sums)
                for m in ("ma", "mb"):
                    seq.drain("C", c, dev.address("ACT", self.ps[(v + 1, m)] + (XB - XPAY) + 512 * c), 512)

    def bank_of(self, v: int) -> tuple[int, int] | None:
        """(bank, state) visit v's combine writes cur into: a pass's (not the last's) mid-core residual (after the core's
        first block) and output, which the next pass's messages mix (``ref.Model.walk``'s bank), or None."""
        vis, cfg = self.cfg.visits[v], self.cfg
        if not 0 <= vis.loop < cfg.passes - 1:
            return None
        mid = cfg.core_start + cfg.block_size - 1
        return (vis.loop % 2, 0) if vis.layer == mid else (vis.loop % 2, 1) if vis.layer == cfg.core_end - 1 else None

    def _bank_words(self, v: int) -> int:
        """(the combine's M_INKN past the queries) its bank's copy (``bank_of``), the next visit's message mixes' partial
        sums (``protocol.M_INKN``)."""
        out = 0
        if (bk := self.bank_of(v)) is not None:
            out |= 1 << 16 | bk[0] << 17 | bk[1] << 18
        if v + 1 < len(self.cfg.visits) and (nxt := self.cfg.visits[v + 1]).messages:
            out |= 4 << 20 | ((nxt.loop - 1) % 2) << 24
        return out

    def _fill_loadr(self, seq: Sequence, S: int) -> None:
        """(the first visit's states 0 .. n0 and the bank its messages read, each slot's: ``write_states``)"""
        n = (self.n0 + 1 + 2 * int(self.vis[self.layers[0]].messages)) * packed_slots(S)
        for c in range(COLS):
            seq.fill("X", c, self.dev.address("ACT", self.loadr + c * self.LOADS * SLOTS * XB), n * XB)

    @staticmethod
    def outputs(r: object) -> int:
        """A q / k / v phase's outputs per slot (bf16 elements: an fp32 row, ``BLOCK_FLAGS``' B_F32, two; a plane pair's
        fp32 sum, two)."""
        return sum(nv * (2 if BLOCK_FLAGS.get(b.name) == B_F32 else 1) for b, nv in zip(r.blocks, mp.valid_rows(r)))

    def _attention(self, seq: Sequence, st: StepPlan, L: int, weights: list[tuple[int, str]]) -> None:
        """Visit L's attention, per sub-step (each after the last one's records are in DDR, the first after the q and raw
        K / V objects): per column the sub-step's q and raw K / V objects and its E objects (X; their window and count
        patched per step: ``_set_parts``), the raw ring and its range of the cache (A; the range patched per step), the raw
        ring back (D), the tail objects (A, patched), the records (D, patched) and o into o's input (D), then the weights
        of the GEMV phases after it. A host wait on A (a task waiting for one of the stream's 3 BDs) must find the
        outputs it depends on drained: the attention core holds the raw ring while it sends it back, and a range of more
        cache objects than the A stream's memtile FIFO holds completes its fill only as the cores take them (a step from
        position 192 on, 14 A objects a sub-step against 11, deadlocked otherwise); so the raw ring's drain goes before
        the tails' fills (a ring's range in two fills: the tails' then wait for the raw ring's), the other drains before
        the weights."""
        dev, T = self.dev, self.T
        use = self.uses[L]
        kind = self.kinds[use.kind]
        subs = self._kind_subs(st.subs, use.kind)
        sp, nq = packed_slots(st.S), self.q_heads * kind.hd
        rel = self.rel[self.layer(L)]
        efills = st.__dict__.setdefault("efills", [])
        starts = np.cumsum((0, *st.subs.parts))
        for u, (a, n) in enumerate(subs):
            q = int(np.searchsorted(starts, a, side="right")) - 1  # (the sub-step's part: its sequence's raw rings)
            seq.sync()  # (sub-step u - 1's records and raw ring in DDR, or the q and raw K / V objects)
            for c in range(COLS):
                reg = self.att_reg[(L, c)] + a * XB
                seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
                efills.append((seq.fill("X", c, dev.address("ACT", rel), XB, (64, E_ROWS, 640, 32)), L, u))
            for c in range(COLS):
                st.part_bds.append(PartBD(seq.fill("A", c, dev.address("KV", self.raw_ring(L, c)), RING), q,
                                          self.raw_ring(L, c), RING))
                st.fills += [Fill(seq.fill("A", c, dev.address("KV", self.kv_off[L]), T), L, c, u, f)
                             for f in range(2 if kind.wrap else 1)]
            for c in range(COLS):
                st.part_bds.append(PartBD(seq.drain("D", c, dev.address("KV", self.raw_ring(L, c)), 8192)[0], q,
                                          self.raw_ring(L, c), 8192))
            for c in range(COLS):
                st.tails += [Tail(seq.fill("A", c, dev.address("KV", self.ring_off[L]), T), L, c, u, t)
                             for t in range(tails(n))]
            o = self.inp[(L, "o")][0]
            for c in range(COLS):
                for i in range(len(kv.emit_order(p=0, S=n, hd=kind.hd, a_bytes=T, kp=self.kp))):
                    st.recs.append(Rec(seq.drain("D", c, dev.address("KV", self.kv_off[L]), 2 * self.at.d_len), L, c, u, i))
                at = dev.address("ACT", o.off + o.x0 + 2 * sp * nq * c + 16 * a)
                seq.drain("D", c, at, 2 * n * nq, None if n == sp else (4 * n, nq // 8, 4 * sp, 4 * sp * nq // 8))
            if u == len(subs) - 1 and weights:
                self._weight_run(seq, weights)

    def _set_parts(self, st: StepPlan, parts: tuple[Part, ...], ahead: bool = False) -> bool:
        """(and the E objects: per visit and sub-step those of its range's last INK_EG groups, ``decode.attn``'s
        ``abegin``: the fill's window and count; the count in M_SKIP of the sub-step's last raw K / V object, which the
        main cores skip before the sub-step's A objects. A count that changes: not ahead, the running step reads those
        objects)"""
        pos = [q.position + i for q in parts for i in range(q.n)]
        counts = {}
        for k in range(len(self.kinds)):
            for a, n in self._kind_subs(st.subs, k):
                g0, ng = self.ranges(k, pos[a])[0]
                counts[(k, a, n)] = (g0, ng, min(ng, INK_EG))
        skips = {(k, a + n - 1): (ne, tails(n)) for (k, a, n), (_, _, ne) in counts.items()}
        changed = [(v, s) for v in self.layers for s in range(SLOTS)
                   if self._eskip.get((v, s)) != skips.get((self.uses[v].kind, s), (0, 0))]
        if ahead and changed:
            return False
        if not super()._set_parts(st, parts, ahead):
            return False
        act = self.dev.view("ACT")
        for v, s in changed:
            ne, nt = skips.get((self.uses[v].kind, s), (0, 0))
            for h in range(self.kv_heads):
                off = self.att_reg[(v, h)] + (SLOTS + s) * XB
                m = act[off:off + 4 * META].view(np.int32)
                m[M_EGRP], m[M_EGA], m[M_ETAIL] = ne, self.G, nt
                self.dev.flush("ACT", off, 4 * META)
            self._eskip[(v, s)] = (ne, nt)
        at, addr, nb = [], [], []
        rows = []  # (per part: its sequence's KV and the window's rows of positions p - 3 ..)
        for q in parts:
            a = q.position - 3 - self._hbase[q.seq]
            assert 0 <= a <= HIST_ROWS - 4, (q, self._hbase[q.seq])
            rows.append((self.seq_name("KV", q.seq), a))
        for x, v, i, c, q in st.__dict__.get("hfills", ()):
            at.append(x)
            buf, a = rows[q]
            addr.append(self.dev.address(buf, self.hist[(v, i)] + c * self.HIST + a * 4 * self.h8, own=True) - (XB - XPAY))
            nb.append(XB)
        for x, L, u in st.__dict__.get("efills", ()):
            use = self.uses[L]
            a, n = self._kind_subs(st.subs, use.kind)[u]
            p = pos[a]
            g0, ng, ne = counts[(use.kind, a, n)]
            ge = g0 + ng - ne  # (the first group with an E object)
            j0 = (INK_ED0 - (p + n - 1 - self.pg * ge)) & ~1  # (its window's first entry of E_rev: even, word-aligned)
            at.append(x)
            addr.append(self.dev.address("ACT", self.rel[self.layer(L)]) + 2 * j0)
            nb.append(ne * XB)
        if at:
            self._bd_words(st.tr.words, np.array(at, np.int64), np.array(addr, np.int64), np.array(nb, np.int64))
            st.tr.flush()
        return True

    def _token_words(self, st: StepPlan, parts: tuple[Part, ...]) -> None:
        """(the A objects the main cores skip after a sub-step's last raw K / V object: its raw ring and its range's groups
        before the first with an E object; the rest by the object's EGRP words, ``_set_parts``)"""
        act = self.dev.view("ACT")
        pos = [q.position + i for q in parts for i in range(q.n)]
        act[st.tok:][:4 * META].view(np.int32)[M_TOK_PARTS] = len(parts) | sum(q.n << 4 * (j + 1) for j, q in enumerate(parts))
        for k in range(len(self.kinds)):
            for a, n in self._kind_subs(st.subs, k):
                _, ng = self.ranges(k, pos[a])[0]
                me = act[st.tok + (SLOTS * k + a + n - 1) * XB:][:4 * META].view(np.int32)
                me[M_TOK_NG:M_TOK_NG + COLS] = 1 + self.G * (ng - min(ng, INK_EG))

    def subs(self, p: int, S: int) -> Subs:
        return Subs(self._substeps(p, S, self.hd), (S,))

    def _join(self, parts: tuple[Part, ...]) -> Subs:
        return Subs(super()._join(parts), tuple(q.n for q in parts))

    def _rope(self, kind: int, positions: list[int]) -> np.ndarray:
        """(the token objects' payload: tau of the slots, fp32; 1 for the sliding-window kind)"""
        out = np.ones(SLOTS, np.float32)
        if KINDS[kind] == "full":
            out[:len(positions)] = self.cfg.tau(positions)
        return out

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def write_states(self, states: np.ndarray, bank: np.ndarray | None = None) -> None:
        """The first visit's states 0 .. n0 (n0 + 1, S, hidden) fp32 and, a loop visit after the first pass, the bank its
        messages mix (2, S, hidden): into the state array ([k block][slot sp][8], zero past the hidden), the F_LOADR
        objects (each slot's column slices) and the partial sums objects of its mixes."""
        n, S, H = states.shape
        v = self.layers[0]
        vis = self.vis[v]
        assert n == self.n0 + 1 and (bank is not None) == vis.messages, (n, self.n0)
        w = (vis.loop - 1) % 2
        slots = list(range(n)) + ([BANK0 + 2 * w, BANK0 + 2 * w + 1] if vis.messages else [])
        loads = list(range(n)) + ([INK_STATES + 2 * w, INK_STATES + 2 * w + 1] if vis.messages else [])
        sp, act, h8 = packed_slots(S), self.dev.view("ACT"), self.h8
        x = np.zeros((len(slots), sp, H), np.float32)
        x[:n, :S] = states
        if vis.messages:
            x[n:, :S] = bank
        ST = self.state_bytes(sp)
        if sp != self._sa_sp:  # (another layout: every state's padding past the hidden zero, which the NPU's drains
            # leave as they find it and the mixes' sums of squares read; a step of other slots put values there)
            for b in range(SA_SLOTS):
                act[self.sa + b * ST + 4 * H * sp:self.sa + (b + 1) * ST] = 0
            self._sa_sp = sp
        for b, slot in enumerate(slots):
            xs = np.zeros((self.state_k(sp) // 8, sp, 8), np.float32)
            xs[:H // 8] = x[b].reshape(sp, H // 8, 8).transpose(1, 0, 2)
            act[self.sa + slot * ST:self.sa + (slot + 1) * ST] = xs.reshape(-1).view(np.uint8)
        self.dev.flush("ACT", self.sa, SA_SLOTS * ST)
        lr = act[self.loadr:self.loadr + COLS * self.LOADS * SLOTS * XB].reshape(COLS, self.LOADS * SLOTS, XB)
        for b, idx in enumerate(loads):
            for s in range(sp):
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_SLOT], m[M_INKN] = F_LOADR | F_MAIN, s, idx
                lr[:, b * sp + s, :4 * META] = m.view(np.uint8)
                lr[:, b * sp + s, 4 * META:4 * META + 4 * h8] = x[b, s].reshape(COLS, h8).view(np.uint8)
        self.dev.flush("ACT", self.loadr, COLS * self.LOADS * SLOTS * XB)
        L = self.layer(v)
        parts = []
        if vis.queries:
            q = np.asarray(self.gguf.tensor(f"blk.{L}.attn_res_query.weight").dequantize(), np.float32).reshape(-1, H)
            parts.append(("a", x[:n], q))
        if vis.messages:
            mq = np.asarray(self.cfg.msg_queries(self.gguf, L), np.float32)
            parts += [("ma", x[n:], mq[:3]), ("mb", x[n:], mq[3:])]
        for m, xs, q in parts:
            ps = psums(xs, q)
            act[self.ps[(v, m)] + XB - XPAY:][:ps.nbytes] = ps.reshape(-1).view(np.uint8)
            self.dev.flush("ACT", self.ps[(v, m)], XB)

    def step(self, states: np.ndarray, position: int | tuple[Part, ...], bank: np.ndarray | None = None, *,
             logits: bool = False) -> np.ndarray | Screened | None:
        """One step of S slots at position .. position + S - 1 (several sequences' slots: their parts, ``decode.step.Part``):
        states (n0 + 1, S, hidden) the first visit's input states,
        bank (2, S, hidden) the bank its messages mix (``write_states``); ``logits`` (the whole model): the LM head's
        (``Screened``: the logits before the softcap), else None."""
        S = states.shape[1]
        assert 1 <= S <= SLOTS and (not logits or self.lm_ok)
        if self._metas_s != (S, self.start):
            self._metas(S)
        self.write_states(states, bank)
        return self._run(S, position, logits)

    def _lm_x(self, S: int) -> np.ndarray:
        """(the screened head's input: the output's mix of the last visit's states, from the state array; the head norms
        it, ``Screen.input``)"""
        n = self._mix(-1, "lm")[3][2]
        x = np.stack([self.state(b, S) for b in range(n)]).astype(np.float64)  # (n, S, H)
        q = np.asarray(self.gguf.tensor("output_res_query.weight").dequantize(), np.float64).reshape(-1)
        xs = x / np.sqrt((x * x).mean(-1, keepdims=True) + InklingConfig.MIX_EPS)
        sc = xs @ q / np.sqrt(self.H)
        w = np.exp(sc - sc.max(0))
        return np.einsum("ns,nsh->sh", w / w.sum(0), xs).astype(np.float32)

    def state(self, b: int, S: int) -> np.ndarray:
        """State b of the last step's state array (S, hidden) fp32."""
        sp, ST = packed_slots(S), self.state_bytes(packed_slots(S))
        self.dev.fetch("ACT", self.sa + b * ST, ST)
        x = self.dev.view("ACT")[self.sa + b * ST:self.sa + (b + 1) * ST].view(np.float32).reshape(self.state_k(sp) // 8, sp, 8)
        return x[:self.H // 8, :S].transpose(1, 0, 2).reshape(S, self.H).copy()

    def psums(self, v: int, m: str) -> np.ndarray:
        """Visit v's partial sums object of its attention's (m "a") or FFN's ("f") mix (COLS, 4, 32) fp32."""
        off = self.ps[(v, m)] + XB - XPAY
        self.dev.fetch("ACT", off, XPAY)
        return self.dev.view("ACT")[off:off + XPAY].view(np.float32).reshape(COLS, 4, 32).copy()

    def history(self, v: int, i: int, rows: int) -> np.ndarray:
        """Visit v's history window i (0: the attention's post conv) as the last step wrote it: rows (rows, hidden) fp32."""
        out = np.zeros((rows, self.H), np.float32)
        for c in range(COLS):
            off = self.hist[(v, i)] + c * self.HIST
            self.dev.fetch("KV", off, rows * 4 * self.h8)
            out[:, self.h8 * c:self.h8 * (c + 1)] = self.dev.view("KV")[off:off + rows * 4 * self.h8].view(np.float32).reshape(rows, self.h8)
        return out

    def attn_out(self, v: int, S: int) -> np.ndarray:
        """Visit v's attention output o of the last step (S, heads x head_dim), times the gates (bf16 values, fp32)."""
        rg, sp = self.inp[(v, "o")][0], packed_slots(S)
        n = 2 * sp * rg.K
        self.dev.fetch("ACT", rg.off + rg.x0, n)
        o = self.dev.view("ACT")[rg.off + rg.x0:rg.off + rg.x0 + n].view(bfloat16).reshape(rg.K // 8, sp, 8)
        return o[:, :S].transpose(1, 0, 2).reshape(S, rg.K).astype(np.float32)
