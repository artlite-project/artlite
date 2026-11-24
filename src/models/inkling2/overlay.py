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

The KV caches: per visit and KV head linear groups (``decode.kv``), a sliding-window visit's after its window's guard;
the rings of open records (``decode.vring``), the raw rings (zero at a sequence's start: the convs' history before
position 0). The token objects carry tau per slot (the full layers' factor of q and r, ``InklingConfig.tau``) where
the other models' carry RoPE tables.

The residual stream: per step the state array (ACT, [state][k block][slot sp][8] fp32, ``INK_STATES``: the normed
embedding, the block states, cur in state n of its visit, ``cur_states``) holds what the mixes read; the helpers keep
each column's slices of the states (``decode.protocol.F_INK``), the first visit's loaded by F_LOADR objects. o (F_RESID
| F_INK): W_o's rows, the post conv over the visit's history window (KV: per visit, conv and column the inputs of the
positions base .. base + 6, ``HIST``; read at the step's position - 3 - base, ``_set_parts``; written whole: the 3 rows
read and the step's), cur (or a new block state) into the state array, the partial sums of the FFN's mix.

The host writes the first visit's input states (``write_states``: the state array, the F_LOADR objects, the partial sums
of its attention mix) until the overlay's own phases make them (the FFN, the controllers: S3's next pieces).
"""

import math

import numpy as np
from ml_dtypes import bfloat16

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
from artlite.decode.model import InputRegion
from artlite.decode.protocol import (
    B_F32,
    B_PAIR,
    C_LEN,
    F_COEF,
    F_INK,
    F_LOADR,
    F_MAIN,
    F_NORM,
    F_RESID,
    F_STATES,
    INK_STATES,
    M_CSET,
    M_EGA,
    M_EGRP,
    M_ETAIL,
    M_FLAGS,
    M_INKN,
    M_NMIX,
    M_NQ,
    M_NR,
    M_NX,
    M_SLOT,
    M_TOK_NG,
    META,
    SLOTS,
    XB,
    XPAY,
    packed_slots,
    region_bytes,
)
from artlite.decode.step import Fill, Part, Rec, StepPlan, Tail
from artlite.decode.vring import PAIR, tails
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig
from artlite.operators.design import COLS
from artlite.overlay import Sequence, WeightBuffer

from .decode import InklingDecodeOperator
from .image import InklingImage

KINDS = ("swa", "full")
FORMATS = frozenset({"Q8_0", "BFP16"})  # (the formats the overlay runs so far: the attention's)
PHASES = ("q", "k", "v", "att", "o")  # (a visit's phases so far)
RING = 9216  # (a raw ring's bytes: one A object, its first 8 KB the ring)
E_ROWS = 17  # (E_rev's rows: a zero row, then the 16 of the table)
BLOCK_FLAGS = dict.fromkeys(("sink", "kd", "vd"), B_PAIR | B_F32) | {"k": B_F32, "v": B_F32}
HIST_ROWS = 8  # (a history window's rows: 3 + S <= 7 written, a read's object past its 3)


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


def pow2(n: int) -> int:
    """The state objects of a k range for a mix of n states (``protocol.M_NR``: 1, 2, 4 or 8)."""
    return 1 << (n - 1).bit_length()


class InklingOverlay(AttentionOverlay):
    """An Inkling GGUF model's visits on its decode overlay (module docstring). ``visits``: the visits to run (default all;
    tests run slices), ``max_positions``: the caches' capacity, ``kv_k``: the caches' K precision."""

    CONFIG, IMAGE = InklingConfig, InklingImage
    CHAIN = 1
    OPS = ("att",)

    def __init__(self, gguf: GGUFFile, *, visits: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "hilo", weights: WeightBuffer | None = None):
        cfg = InklingConfig.from_gguf(gguf)
        vs = list(range(len(cfg.visits))) if visits is None else list(visits)
        self.vis = {v: cfg.visits[v] for v in vs}
        image = None if weights is not None else InklingImage(gguf, cfg, layers=sorted({x.layer for x in self.vis.values()}),
                                                             vocab=vocab)
        super().__init__(gguf, layers=vs, vocab=vocab, kv_k=kv_k, weights=weights, image=image)
        img = self.image
        fmts = {b.fmt for L in img.layers for name in ("q", "k", "v") for b in img.region(L, name).blocks}
        if fmts - FORMATS:
            raise NotImplementedError(f"{', '.join(sorted(fmts - FORMATS))} attention tiles")
        self.hd, self.q_heads = cfg.head_dim, cfg.q_per_kv
        self.at = at = Attn(hds=(self.hd,), rot=(0,), chain=1, heads=self.q_heads, kp=self.kp, v_norm=True,
                            eps=cfg.rms_norm_eps, a_bytes=self.T, ink=True)
        self.op = InklingDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                        eps=cfg.rms_norm_eps, attn=at)
        self.pg = at.pg[self.hd]
        self.guard = -(-(cfg.sliding_window - 1) // self.pg)
        full = -(-max_positions // self.pg)
        self.max_positions = full * self.pg
        self.kinds = (CacheKind(hd=self.hd, pg=self.pg, groups=self.guard + full, guard=self.guard,
                                window=cfg.sliding_window),
                      CacheKind(hd=self.hd, pg=self.pg, groups=full))
        self.uses = {v: CacheUse(kind=KINDS.index(x.kind), source=v, ring=v) for v, x in self.vis.items()}
        self.period = PAIR
        self.K = img.region(img.layers[0], "q").K  # (an attention input's K: the hidden padded to whole chunks)
        self.h8 = self.H // COLS
        self.HIST = HIST_ROWS * 4 * self.h8  # (a history window's bytes)
        self.nst = dict(zip(range(len(cfg.visits)), cur_states(cfg)))
        self.n0 = self.nst[vs[0]]  # (the first visit's cur: its states 0 .. n0 come from the host)
        self._hbase = -3  # (the history windows' first position)
        self._eskip: dict[tuple[int, int], tuple] = {}  # ((visit, slot) -> its raw K / V objects' EGRP words: ``_set_parts``)
        self._plan()

    def layer(self, v: int) -> int:
        """The physical layer of visit v."""
        return self.vis[v].layer

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = False) -> list[tuple[int, str]]:
        return [(v, n) for v in self.layers for n in PHASES]

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        self.toks = (take("ACT", len(KINDS) * SLOTS * XB),)
        self.loadr = take("ACT", COLS * INK_STATES * SLOTS * XB)  # ([column][state b][slot s] F_LOADR objects)
        # the state array ([state][k block][slot sp][8] fp32, after the XB - XPAY bytes its objects' fill reads before
        # it); per visit: the q, k, v phases' metadata objects, the partial sums objects of its attention's and FFN's
        # mixes, the attention region [q objects | raw K / V objects] per KV head, o's input
        self.sa = take("ACT", XB + INK_STATES * 4 * SLOTS * self.K) + XB
        self.meta = {(v, n): take("ACT", XB) for v in self.layers for n in ("q", "k", "v")}
        self.ps = {(v, m): take("ACT", XB) for v in self.layers for m in "af"}
        self.att_reg = {(v, h): take("ACT", 2 * SLOTS * XB) for v in self.layers for h in range(self.kv_heads)}
        self.inp = {(v, "o"): [InputRegion(take("ACT", region_bytes(3 * 1024)), 0, 3 * 1024)] for v in self.layers}
        self.rel = {L: take("ACT", E_ROWS * 2 * 1280) for L in self.image.layers}  # (E_rev per layer, ``ink_rel``)
        self._plan_caches(self.layers, pad=0)
        self.raw = {v: self._take("KV", self.kv_heads * RING) for v in self.layers}
        self.hist = {(v, i): self._take("KV", COLS * self.HIST) for v in self.layers for i in range(2)}  # (i: 0 attention's)
        self._sizes()

    def state_bytes(self, sp: int) -> int:
        """A state's bytes in the state array of a step of sp packed slots."""
        return 4 * sp * self.K

    def raw_ring(self, v: int, h: int) -> int:
        """Byte offset in KV of visit v's KV head h's raw ring."""
        return self.raw[v] + h * RING

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        super()._load()
        act, g, cfg = self.dev.view("ACT"), self.gguf, self.cfg
        act[self.loadr:self.loadr + COLS * INK_STATES * SLOTS * XB] = 0
        for L, off in self.rel.items():
            E = np.asarray(g.tensor(f"blk.{L}.attn_rel_bias.weight").dequantize(), np.float32).reshape(-1, cfg.rel_length).T
            act[off:off + E_ROWS * 2 * 1280] = ink_rel(E).view(np.uint8).reshape(-1)

    def reset(self) -> None:
        """(and the raw rings and history windows: the convs' history before position 0 is zero)"""
        super().reset()
        self._eskip.clear()
        self._hbase = -3
        if self.dev is not None:
            kv = self.dev.view("KV")
            for v in self.layers:
                kv[self.raw[v]:self.raw[v] + self.kv_heads * RING] = 0
                for i in range(2):
                    kv[self.hist[(v, i)]:self.hist[(v, i)] + COLS * self.HIST] = 0
            self.dev.flush("KV")

    def _ran(self, p: int, S: int, seq: int | None = None) -> None:
        """(and the history windows: rows p - 3 .. p + S - 1)"""
        super()._ran(p, S, seq)
        self._hbase = p - 3

    def _ring_state(self, seq: int | None = None) -> tuple:
        return super()._ring_state(seq) + (self._hbase,)

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
        for (v, name), off in self.meta.items():
            vis, n = self.vis[v], self.nst[v]
            gr = mp.groups(self.image.region(self.layer(v), name), F_NORM, S, block_flags=BLOCK_FLAGS)
            assert len(gr) == 1
            m = gr[0].meta.copy()
            m[0] |= F_STATES
            m[M_NR] = pow2(n + 1)
            m[M_NX] = ranges * m[M_NR]
            m[M_CSET] = -1 if not vis.queries else (0 if vis.kind == "swa" else "qkv".index(name))
            if vis.queries and name == "q":
                m[0] |= F_COEF
                m[M_NQ], m[M_NMIX] = vis.queries, n + 1
            out.append((off, m.view(np.uint8)))
        for v in self.layers:
            rg = self.inp[(v, "o")][0]
            gr = mp.groups(self.image.region(self.layer(v), "o"), F_RESID | F_INK, S)
            assert len(gr) == 1
            m = gr[0].meta.copy()
            m[M_INKN] = self.nst[v] | int(self.vis[v].block) << 4 | 1 << 8  # (the FFN mix's query after the taps)
            out.append((rg.off, m.view(np.uint8)))
        return out

    def _weight_run(self, seq: Sequence, phases: list[tuple[int, str]], **kw: object) -> None:
        """(a visit's phases read its physical layer's regions)"""
        gemv = [(self.layer(v), n) for v, n in phases if n not in self.OPS]
        if gemv and gemv[-1][1] == "o":  # (and the residual stream's object after its tiles)
            gemv.append((gemv[-1][0], "inkf"))
        if gemv:
            super()._weight_run(seq, gemv, **kw)

    def _gemv(self, seq: Sequence, st: StepPlan, phase: tuple[int, str]) -> None:
        """A visit's q, k or v phase: its metadata object and its input state's objects (X), its outputs (C) into the
        attention region's q or raw K / V objects."""
        v, name = phase
        if name == "o":
            self._o_phase(seq, st, v)
            return
        dev, S, sp = self.dev, st.S, packed_slots(st.S)
        seq.sync()
        ST = self.state_bytes(sp)
        nr = pow2(self.nst[v] + 1)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.meta[phase]), XB)
            if name == "q" and self.vis[v].queries:  # (the partial sums object of its mix)
                seq.fill("X", c, dev.address("ACT", self.ps[(v, "a")]), XB)
            seq.fill("X", c, dev.address("ACT", self.sa - (XB - XPAY)), nr * XB, (XB // 8, 2, XB // 8, ST // 4),
                     repeat=ST // XPAY, iteration=(XPAY // 4, ST // XPAY))
        r = self.image.region(self.layer(v), name)
        n = -(-self.outputs(r) // C_LEN)  # (C objects per slot)
        for c in range(COLS):  # (object j of slot s -> the slot's object, 128 j bytes into its payload)
            obj = self.att_reg[(v, c)] + (SLOTS * XB if name != "q" else 0) + 4 * META
            off = {"q": 0, "k": 0, "v": 4 * 160}[name]
            seq.drain("C", c, dev.address("ACT", obj + off), n * S * 2 * C_LEN, (C_LEN // 2, S, XB // 4, C_LEN // 2))

    def _o_phase(self, seq: Sequence, st: StepPlan, v: int) -> None:
        """Visit v's o (F_RESID | F_INK): its input (the attention's o), its post conv's history window (X, patched per
        step: ``_set_parts``); out (C) the window (the 3 rows read, the step's), cur's slice of the state array, the
        column's partial sums of the FFN's mix."""
        dev, S, sp = self.dev, st.S, packed_slots(st.S)
        seq.sync()
        self._inputs(seq, self.inp[(v, "o")][0], S)
        hf = st.__dict__.setdefault("hfills", [])
        for c in range(COLS):
            hf.append((seq.fill("X", c, dev.address("KV", self.hist[(v, 0)] + c * self.HIST) - (XB - XPAY), XB), v, 0, c))
        n1 = self.nst[v] + int(self.vis[v].block)
        row = 4 * self.h8
        for c in range(COLS):
            seq.drain("C", c, dev.address("KV", self.hist[(v, 0)] + c * self.HIST), (3 + S) * row)
            seq.drain("C", c, dev.address("ACT", self.sa + n1 * self.state_bytes(sp) + c * row * sp), row * sp)
            seq.drain("C", c, dev.address("ACT", self.ps[(v, "f")] + (XB - XPAY) + 512 * c), 512)

    def _fill_loadr(self, seq: Sequence, S: int) -> None:
        """(the first visit's states 0 .. n0, each slot's: ``write_states``)"""
        n = (self.n0 + 1) * packed_slots(S)
        for c in range(COLS):
            seq.fill("X", c, self.dev.address("ACT", self.loadr + c * INK_STATES * SLOTS * XB), n * XB)

    @staticmethod
    def outputs(r: object) -> int:
        """A q / k / v phase's outputs per slot (bf16 elements: an fp32 row, ``BLOCK_FLAGS``' B_F32, two; a plane pair's
        fp32 sum, two)."""
        return sum(nv * (2 if BLOCK_FLAGS.get(b.name) == B_F32 else 1) for b, nv in zip(r.blocks, mp.valid_rows(r)))

    def _attention(self, seq: Sequence, st: StepPlan, L: int, weights: list[tuple[int, str]]) -> None:
        """Visit L's attention, per sub-step (each after the last one's records are in DDR): per column the raw ring, its
        range of the cache and the tail objects (A; the range and tails patched per step), the sub-step's q and raw K / V
        objects and its E objects (X; their window and count patched per step: ``_set_parts``), the raw ring back, the
        records (D, patched) and o into o's input (D). The weights of the GEMV phases after it go before the sync of its
        inputs (``AttentionOverlay._attention``)."""
        dev, T = self.dev, self.T
        use = self.uses[L]
        kind = self.kinds[use.kind]
        subs = self._kind_subs(st.subs, use.kind)
        sp, nq = packed_slots(st.S), self.q_heads * kind.hd
        before = self.a_bds > 3
        rel = self.rel[self.layer(L)]
        efills = st.__dict__.setdefault("efills", [])
        for u, (a, n) in enumerate(subs):
            if u:
                seq.sync()  # (sub-step u - 1's records and raw ring in DDR: this one reads them)
            for c in range(COLS):
                seq.fill("A", c, dev.address("KV", self.raw_ring(L, c)), RING)
                st.fills += [Fill(seq.fill("A", c, dev.address("KV", self.kv_off[L]), T), L, c, u, f)
                             for f in range(2 if kind.wrap else 1)]
                st.tails += [Tail(seq.fill("A", c, dev.address("KV", self.ring_off[L]), T), L, c, u, t)
                             for t in range(tails(n))]
            last = u == len(subs) - 1 and weights
            if last and before:
                self._weight_run(seq, weights, blocks=slice(1))
            if u == 0:
                seq.sync()  # (the q and raw K / V objects in DDR)
            for c in range(COLS):
                reg = self.att_reg[(L, c)] + a * XB
                seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
                efills.append((seq.fill("X", c, dev.address("ACT", rel), XB, (64, E_ROWS, 640, 32)), L, u))
            if last:
                self._weight_run(seq, weights, blocks=slice(1, None) if before else slice(None))
            o = self.inp[(L, "o")][0]
            for c in range(COLS):
                seq.drain("D", c, dev.address("KV", self.raw_ring(L, c)), 8192)
                for i in range(len(kv.emit_order(p=0, S=n, hd=kind.hd, a_bytes=T, kp=self.kp))):
                    st.recs.append(Rec(seq.drain("D", c, dev.address("KV", self.kv_off[L]), 2 * self.at.d_len), L, c, u, i))
                at = dev.address("ACT", o.off + o.x0 + 2 * sp * nq * c + 16 * a)
                seq.drain("D", c, at, 2 * n * nq, None if n == sp else (4 * n, nq // 8, 4 * sp, 4 * sp * nq // 8))

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
        a = pos[0] - 3 - self._hbase  # (the history windows' rows of positions p - 3 ..)
        assert 0 <= a <= HIST_ROWS - 4, (pos[0], self._hbase)
        for x, v, i, c in st.__dict__.get("hfills", ()):
            at.append(x)
            addr.append(self.dev.address("KV", self.hist[(v, i)] + c * self.HIST + a * 4 * self.h8) - (XB - XPAY))
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
        for k in range(len(self.kinds)):
            for a, n in self._kind_subs(st.subs, k):
                _, ng = self.ranges(k, pos[a])[0]
                me = act[st.tok + (SLOTS * k + a + n - 1) * XB:][:4 * META].view(np.int32)
                me[M_TOK_NG:M_TOK_NG + COLS] = 1 + self.G * (ng - min(ng, INK_EG))

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        return self._substeps(p, S, self.hd)

    def _rope(self, kind: int, positions: list[int]) -> np.ndarray:
        """(the token objects' payload: tau of the slots, fp32; 1 for the sliding-window kind)"""
        out = np.ones(SLOTS, np.float32)
        if KINDS[kind] == "full":
            out[:len(positions)] = self.cfg.tau(positions)
        return out

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def write_states(self, states: np.ndarray) -> None:
        """The first visit's states 0 .. n0 (n0 + 1, S, hidden) fp32: into the state array ([k block][slot sp][8], zero past
        the hidden), the F_LOADR objects (each slot's column slices) and the partial sums object of its attention's mix."""
        n, S, H = states.shape
        assert n == self.n0 + 1, (n, self.n0)
        sp, act, h8 = packed_slots(S), self.dev.view("ACT"), self.h8
        x = np.zeros((n, sp, H), np.float32)
        x[:, :S] = states
        ST = self.state_bytes(sp)
        xs = np.zeros((n, self.K // 8, sp, 8), np.float32)
        xs[:, :H // 8] = x.reshape(n, sp, H // 8, 8).transpose(0, 2, 1, 3)
        act[self.sa:self.sa + n * ST] = xs.reshape(-1).view(np.uint8)
        self.dev.flush("ACT", self.sa, n * ST)
        lr = act[self.loadr:self.loadr + COLS * INK_STATES * SLOTS * XB].reshape(COLS, INK_STATES * SLOTS, XB)
        for b in range(n):
            for s in range(sp):
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_SLOT], m[M_INKN] = F_LOADR | F_MAIN, s, b
                lr[:, b * sp + s, :4 * META] = m.view(np.uint8)
                lr[:, b * sp + s, 4 * META:4 * META + 4 * h8] = x[b, s].reshape(COLS, h8).view(np.uint8)
        self.dev.flush("ACT", self.loadr, COLS * INK_STATES * SLOTS * XB)
        v = self.layers[0]
        if self.vis[v].queries:
            L = self.layer(v)
            q = np.asarray(self.gguf.tensor(f"blk.{L}.attn_res_query.weight").dequantize(), np.float32).reshape(-1, H)
            ps = psums(x, q)
            act[self.ps[(v, "a")] + XB - XPAY:][:ps.nbytes] = ps.reshape(-1).view(np.uint8)
            self.dev.flush("ACT", self.ps[(v, "a")], XB)

    def step(self, states: np.ndarray, position: int) -> None:
        """One step of S slots at position .. position + S - 1: states (n0 + 1, S, hidden) the first visit's input states
        (``write_states``)."""
        S = states.shape[1]
        assert 1 <= S <= SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        self.write_states(states)
        self._run(S, position, False)

    def state(self, b: int, S: int) -> np.ndarray:
        """State b of the last step's state array (S, hidden) fp32."""
        sp, ST = packed_slots(S), self.state_bytes(packed_slots(S))
        self.dev.fetch("ACT", self.sa + b * ST, ST)
        x = self.dev.view("ACT")[self.sa + b * ST:self.sa + (b + 1) * ST].view(np.float32).reshape(self.K // 8, sp, 8)
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
