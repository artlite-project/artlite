"""The prefill of an Inkling 2 model: a prompt chunk of T tokens through the visits on the decode overlay's buffers
(``overlay.InklingOverlay``: the weight image, the visits' KV caches, raw rings and history windows), so that the decode
continues from what the chunk leaves; the prompt's last token runs as a decode step (``ChunkPrefill.plan``).

The residual stream (fp32 token rows, the prefill's buffers): E (the normed embedding), P (the prelude's output), M0 /
X0 and M1 / X1 (a pass's mid-core block state and output: the next pass's bank, alternating), CD (the coda's cur). A
visit's states for its mixes are the block states before it and cur (``ref.Model.walk``'s order of buffers: a block
visit's input becomes a block state, its output starts a new cur; a pass restarts from P and the prelude's blocks).

Per visit v of physical layer L (``prefill.ink_stream``, ``prefill.attn``, ``prefill.gemm``), in sub-chunks of ``SUB``
tokens where the operator works row by row:

- the attention inputs: visit 0 its embedding rows (bf16, EB); else the depth mix of its states with the layer's query
  (1, a full layer 3: x_q, x_k, x_v; ``InkMixOperator``) -> XM rows; a loop visit after the first pass: per stream s
  (q, k, v) its message (the mix of the bank with the message query s) MS, the controller's down halves (the norms
  without weights -> DP panels, the GEMMs d<s>a / d<s>b -> GA / GB), the blend (``InkBlendOperator``) -> XC rows;
- the norms (the layer's attention norms; ``PrefillNormOperator``) -> XN panels, the q / k / v GEMMs -> GQ / GK / GV;
- the prep (``prefill.attn.ink.InkPrepOperator``: QS, the visit's records, its raw ring; in parts of up to 512 tokens), the
  attention (``AttnOperator(ink=True)``; a chunk past 2048 tokens in items of 2048 at their own positions, after every
  prep part) -> A, the gate (``InkGateOperator``), o -> Y;
- the post conv and the residual (``InkResidOperator``: the chunk's rows at once, the visit's history window 0);
- the FFN: the FFN mix (its states and cur after the attention; a loop visit's controller of the stream "ffn") -> XM0
  (XC0), its norm -> XN0, the MoE (``moe.InkMoE``: the router, then a piece from the gathered norm on, the host's
  routing before it) -> DL, the FFN's post conv into cur (history window 1).

``ffn`` False: the visits' attention parts only (the decode overlay's ``ffn=False`` slices: tests).
"""

import functools
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

from artlite.model_config import InklingConfig
from artlite.moe.expert_gemm import CHUNK
from artlite.operators.design import COLS
from artlite.operators.util import arg_bytes
from artlite.prefill import attn as PA
from artlite.prefill import gemm as PG
from artlite.prefill.attn.ink import RING_BYTES, InkPrepOperator, ink_shape
from artlite.prefill.chunks import ChunkPrefill
from artlite.prefill.ink_stream import InkBlendOperator, InkGateOperator, InkMixOperator, InkResidOperator
from artlite.prefill.norm import PrefillNormOperator

from .moe import InkMoE
from .overlay import HIST_ROWS, KV_PAD, InklingOverlay

SUB = 256  # tokens per sub-chunk (the row operators' and GEMMs' T)
SMALL = (64, 128)  # chunk sizes of operators of their own
ATTN_T = 2048  # an attention item's tokens at most (its 64 passes of 32 queries; a larger chunk's in parts)
# chunk sizes (the experts' weights once per chunk: a chunk of 8192 tokens moves 0.61 of a 2048-token one's expert chunks
# per token, its experts' rows 82% valid)
SIZES = SMALL + tuple(range(SUB, 1024 + 1, SUB)) + (1536, 2048, 4096, 6144, 8192)
PREP_T = 512  # tokens of a prep item at most (its chained fills)
CG = 64  # positions of a cache group (the overlay's ``pg``)
CTL = ("q", "k", "v")  # a loop visit's controlled attention streams
# the time model's constants (``layer_seconds``; fitted to chunks of 1024 .. 8192 tokens at p0 0 and a 2048-token one at
# 30720, Strix Halo): a ConfigureOp; per sub-chunk a visit's attention side (its GEMMs, mixes, norms, prep, gate, post
# convs) and a loop visit's controllers; per cache group an attention pass streams, the near part's (the relative bias:
# the cores bind) and a full layer's far part's (~52 GB/s); the experts per expert chunk of the busiest column; per token
# the gathered norm and the combine, the host's routing (and per visit its fixed part)
CFG_S = 0.12e-3
SUB_S, CTL_S = 2.1e-3, 2.6e-3
NEAR_S, FAR_S = 7.5e-6, 4.2e-6
CHUNK_S, MOE_TOK_S, ROUTE_TOK_S, ROUTE_S = 1.29e-3, 2.6e-6, 0.7e-6, 1.5e-3


class InklingPrefill(ChunkPrefill):
    """The prefill of an overlay's visits in chunks of the sizes ``sizes`` (module docstring); ``run`` runs one."""

    SUB, SMALL, MAX_T, NAME, KV_PAD = SUB, SMALL, SIZES[-1], "inkling_prefill", KV_PAD
    MIN_TOKENS = 3  # (the post convs' windows: the chunk's last 3 valid tokens' inputs)
    CFG_S = CFG_S

    def __init__(self, overlay: InklingOverlay, *, sizes: tuple[int, ...] = SIZES, ffn: bool = True):
        ov = overlay
        assert not ffn or ov.ffn, "the overlay's FFNs"
        super().__init__(overlay, tuple(t for t in sizes if t % ov.pg == 0))
        cfg = self.cfg
        self.ffn = ffn
        self.layers = list(ov.layers)  # (the visits: ``ChunkPrefill``'s layers)
        self.kv_layers = list(self.layers)
        self.vis = {v: cfg.visits[v] for v in self.layers}
        self.shape = ink_shape(ov.T, ov.kp)
        self.group = self.shape.pg
        self.guard = PA.mode.swa_groups(cfg.sliding_window, self.shape.pg)
        self.wrap = ov.kinds[0].wrap  # (the sliding-window caches' ring: a chunk within one turn of it, ``fits``)
        assert self.guard == ov.guard and self.shape.pg == ov.pg == CG, (self.guard, self.shape.pg)
        self.walk = self._walk()
        self._gemms: dict = {}
        self._starts: dict[int, list[int]] = {}
        self._marks: dict[int, list[tuple[int, int, str]]] = {}  # (per size: (visit, the item its host step precedes, XF))
        self.moe = InkMoE(self.img, layers=sorted({self.vis[v].layer for v in self.layers}), sizes=self.sizes, sub=SUB,
                          eps=cfg.rms_norm_eps) if ffn else None
        self.sz = {t: self._size_ops(t) for t in sorted({self._t(T) for T in self.sizes})}
        self.kinds = sorted({x.kind for x in self.vis.values()})
        self.attn = {(k, T): self._attn(k, T) for k in self.kinds for T in self.sizes if T <= ATTN_T}
        # (a larger chunk's parts: part 0 the ATTN_T design, the others with parameters of their own)
        self.parts = {(k, j): self.attn[(k, ATTN_T)] if j == 0 else self._attn(k, ATTN_T, j)
                      for k in self.kinds for j in range(self.sizes[-1] // ATTN_T) if self.sizes[-1] > ATTN_T}
        self.preps = {(k, T): self._preps(k, T) for k in self.kinds for T in self.sizes}
        self.resid = {(T, b): InkResidOperator(T=T, block=b, y_pitch=COLS * self.sz[self._t(T)].d_pitch,
                                               y_col=self.sz[self._t(T)].d_pitch, hist_rows=HIST_ROWS)
                      for T in self.sizes for b in (False, True)}
        # (the FFN's: its delta DL as rows)
        self.fresid = {T: InkResidOperator(T=T, block=False, y_pitch=self.H, y_col=self.H // COLS, hist_rows=HIST_ROWS)
                       for T in self.sizes} if ffn else {}
        self._rows()
        self._library()

    # ------------------------------------------------------------------
    # The visits' states
    # ------------------------------------------------------------------

    def _walk(self) -> dict[int, SimpleNamespace]:
        """Per visit its states' buffers (``ref.Model.walk``): ``blocks`` (the block states before it), ``cur``, ``new``
        (a block visit's output state), ``bank`` (a pass after the first: the previous pass's [M, X])."""
        cfg = self.cfg
        cur, blocks, out = "E", [], {}
        prelude: tuple[str, list[str]] | None = None
        mid = cfg.core_start + cfg.block_size
        for v, x in enumerate(cfg.visits):
            bank = None
            if x.loop >= 0 and x.layer == cfg.core_start:
                if x.loop == 0:
                    prelude = cur, list(blocks)
                else:
                    w = (x.loop - 1) % 2
                    bank = [f"M{w}", f"X{w}"]
                    cur, blocks = prelude[0], list(prelude[1])
            if x.loop >= 1:
                bank = [f"M{(x.loop - 1) % 2}", f"X{(x.loop - 1) % 2}"]
            new = None
            if x.block:
                new = "P" if x.layer == 0 else f"M{x.loop % 2}" if x.layer == cfg.core_start else \
                    f"X{x.loop % 2}" if x.layer == mid else "CD"
            out[v] = SimpleNamespace(blocks=list(blocks), cur=cur, new=new, bank=bank)
            if x.block:
                blocks.append(cur)
                cur = new
        return out

    # ------------------------------------------------------------------
    # The operators
    # ------------------------------------------------------------------

    @staticmethod
    def _t(T: int) -> int:
        return min(T, SUB)

    def _attn(self, kind: str, T: int, part: int = 0) -> PA.AttnOperator:
        """The attention of a kind over T tokens (``part``: a larger chunk's, at its own position): sliding-window caches
        after their guard; full ones read 16 groups before group 0 (the near part's, skipped there: the argument starts
        that far before the cache)."""
        s, ov = self.shape, self.ov
        kv0 = self.guard * s.group
        first = next(v for v in self.layers if self.vis[v].kind == kind)
        return PA.AttnOperator(T=T, shape=s, kv_bytes=ov.kv_bytes(first), kv_pad=self.pad, sub=self._pt(T),
                               swa=self.cfg.sliding_window if kind == "swa" else 0, kv0=kv0, ink=True, part=part,
                               wrap=self.wrap if kind == "swa" else 0)

    def _attns(self, kind: str, T: int) -> list[PA.AttnOperator]:
        """The attention items of a chunk of size T (one, or its parts of ATTN_T tokens)."""
        return [self.attn[(kind, T)]] if T <= ATTN_T else [self.parts[(kind, j)] for j in range(T // ATTN_T)]

    @staticmethod
    def _pt(T: int) -> int:
        """Tokens of a chunk's prep part (the attention's items): the chunk up to ``PREP_T``, else ``PREP_T`` (768: half of
        it)."""
        return T if T <= PREP_T else PREP_T if T % PREP_T == 0 else T // 2

    def _preps(self, kind: str, T: int) -> list[InkPrepOperator]:
        """The prep of a kind over a chunk of T tokens: one item up to ``PREP_T`` tokens, else parts (``_pt``)."""
        pt = self._pt(T)
        assert pt <= PREP_T and pt % self.shape.pg == 0, T
        v = next(v for v in self.layers if self.vis[v].kind == kind)
        plans, ov = self.sz[self._t(T)].plans, self.ov
        L = self.vis[v].layer
        return [InkPrepOperator(T=pt, shape=self.shape, lq=plans[(L, "q")].layout, lk=plans[(L, "k")].layout,
                                lv=plans[(L, "v")].layout, full=kind == "full", kv_bytes=ov.kv_bytes(v),
                                tau_positions=ov.max_positions + self.MAX_T, kv_pad=self.pad, eps=self.cfg.rms_norm_eps,
                                kv0=self.guard * self.shape.group, part=k, wrap=self.wrap if kind == "swa" else 0)
                for k in range(T // pt)]

    @property
    def pad(self) -> int:
        """Bytes the attention's and the prep's KV arguments reach past a visit's caches (the overlay's padding)."""
        return KV_PAD // self.shape.pg * self.shape.group

    def _size_ops(self, t: int) -> SimpleNamespace:
        """The operators of sub-chunks of t tokens: the GEMM plans, the norms, the mixes (per state count), the
        controllers' blends, the gates."""
        cfg, img, H = self.cfg, self.img, self.H
        dp = PG.d_pitch(img, t)
        plans = {}
        for v in self.layers:
            L = self.vis[v].layer
            names = ["q", "k", "v", "o"] + ([f"d{s}{h}" for s in self.ctl for h in "ab"] if self.vis[v].messages else [])
            for n in names:
                if (L, n) not in plans:
                    plans[(L, n)] = PG.plan_phase(img, img.region(L, n), t, pitch=dp if n == "o" else 0, ops=self._gemms)
        eps = cfg.rms_norm_eps
        cols = 4 if t < SUB else COLS
        mixes = {n: InkMixOperator(T=t, n=n, H=H) for n in range(1, 8)}
        lq = plans[(self.vis[self.layers[0]].layer, "q")].layout
        dl = next((p.layout for (_, n), p in plans.items() if n == "dqa"), None)
        return SimpleNamespace(
            plans=plans, d_pitch=dp, mixes=mixes,
            norm=PrefillNormOperator(T=t, add=False, E=H, eps=eps, cols=cols),
            norm1=PrefillNormOperator(T=t, add=False, E=H, eps=InklingConfig.MIX_EPS, cols=cols),
            gate=InkGateOperator(T=t, lq_pitch=lq.pitch, gate_off=lq.off["gate"]),
            blend=InkBlendOperator(T=t, d_pitch=dl.pitch, w_stride=img.region(self._loop_layer(), "ctl_q").col_stride // 2)
            if dl is not None else None)

    @property
    def ctl(self) -> tuple[str, ...]:
        """A loop visit's controlled streams."""
        return CTL + (("ffn",) if self.ffn else ())

    def _loop_layer(self) -> int:
        return next(self.vis[v].layer for v in self.layers if self.vis[v].messages)

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers; the static buffers' layouts (``_bind``)."""
        H = self.H
        t = max(self.sz)
        o = self.sz[t]
        lq = next(p.layout for (_, n), p in o.plans.items() if n == "q")
        lk = max((p.layout for (_, n), p in o.plans.items() if n == "k"), key=lambda lay: lay.N)
        lv = max((p.layout for (_, n), p in o.plans.items() if n == "v"), key=lambda lay: lay.N)
        self.rb = {s: 4 * H for s in ("E", "P", "M0", "X0", "M1", "X1", "CD")}
        self.rb |= {"EB": 2 * H, "XM0": 2 * H, "XM1": 2 * H, "XM2": 2 * H, "MS": 2 * H, "XC0": 2 * H, "XC1": 2 * H,
                    "XC2": 2 * H, "XN0": 9 * H // 8, "XN1": 9 * H // 8, "XN2": 9 * H // 8, "DP": 9 * H // 8,
                    "GQ": 2 * lq.N, "GK": 2 * lk.N, "GV": 2 * lv.N, "A": 9 * COLS * 3 * 128 // 8,
                    "Y": 2 * COLS * o.d_pitch, "QS": COLS * self.shape.tgu // 8}
        dl = next((p.layout for (_, n), p in o.plans.items() if n == "dqa"), None)
        if dl is not None:
            self.rb |= {"GA": 2 * dl.N, "GB": 2 * dl.N}
        if self.moe is not None:
            self.rb |= self.moe.rows(t)
        # static: per physical layer its norm weights (NP: [attn or q, k, v inputs, ones, ffn] x H bf16), its queries (QR: the
        # attention's 3 rows (one used by a sliding-window layer), the FFN's, the 4 messages' x H fp32), its prep's PRM,
        # its E table (EB), its post convs' taps (TP: [2][COLS][4][C] bf16); TAU per position
        phys = sorted({x.layer for x in self.vis.values()})
        self.phys = {L: i for i, L in enumerate(phys)}
        prep = next(iter(self.preps.values()))[0]
        self.np_w, self.qr_w, self.prm_w = 5 * 2 * H, 8 * 4 * H, 2 * COLS * prep.prm_w
        self.e_w = {k: 2 * self._attns(k, self.sizes[-1])[0].n_e for k in self.kinds}
        self.tp_w = 2 * 2 * COLS * 4 * (H // COLS)

    def _np(self, L: int, j: int) -> str:
        o = self.phys[L] * self.np_w + j * 2 * self.H
        return f"NP[{o}:{o + 2 * self.H}]"

    def _qr(self, L: int, j: int) -> str:
        o = self.phys[L] * self.qr_w + j * 4 * self.H
        return f"QR[{o}:{o + 4 * self.H}]"

    def _prm(self, L: int, op: InkPrepOperator) -> str:
        o = self.phys[L] * self.prm_w
        return f"PRM[{o}:{o + arg_bytes(op, 4)}]"

    def _eb(self, L: int, op: PA.AttnOperator) -> str:
        o = self.phys[L] * max(self.e_w.values())
        return f"ET[{o}:{o + arg_bytes(op, 3)}]"

    def _tp(self, L: int, i: int, op: InkResidOperator) -> str:
        o = self.phys[L] * self.tp_w + i * self.tp_w // 2
        return f"TP[{o}:{o + arg_bytes(op, 2 if op.block else 3)}]"

    def _runlist(self, S: int) -> list[tuple]:
        return self._runlist_of(S * SUB)

    def _runlist_of(self, T: int, *, packed: bool = False, tree: bool = False) -> list[tuple]:
        assert not packed and not tree, "Inkling: no packed chunks"
        ov, sl = self.ov, self._sl
        t = self._t(T)
        S, o = T // t, self.sz[t]

        def tok(buf: str, j: int, op: object, *, i: int, row: int = 0, extra: int = 0) -> str:
            return sl(buf, j * t * (row or self.rb[buf]) + extra, op, i=i)

        rl: list[tuple] = []
        starts, marks = [], []
        for v in self.layers:
            starts.append(len(rl))
            x, w = self.vis[v], self.walk[v]
            L = x.layer
            full = x.kind == "full"
            states = w.blocks + [w.cur]
            inputs = []  # (per stream: its rows' buffer, its norm weight's index)
            if not x.queries:  # (layer 0: the normed embedding itself)
                inputs = [("EB", 0)]
            else:
                for j in range(3 if full else 1):
                    mx = o.mixes[len(states)]
                    rl += [(mx, *(tok(s_, jj, mx, i=i) for i, s_ in enumerate(states)), self._qr(L, j),
                            tok(f"XM{j}", jj, mx, i=len(states) + 1)) for jj in range(S)]
                    inputs.append((f"XM{j}", j if full else 0))
            if x.messages:  # (per stream its message, the controller's down halves, the blend)
                inputs = [(self._control(rl, o, S, tok, v, s, inputs[si if full else 0][0], f"XC{si}"), si if full else 0)
                          for si, s in enumerate(CTL)]
            nm = o.norm
            for j, (src, ni) in enumerate(inputs):
                rl += [(nm, tok(src, jj, nm, i=0), self._np(L, ni), tok(f"XN{j}", jj, nm, i=2)) for jj in range(S)]
            gr = {}  # (per GEMM output its rows' bytes: the visit's own layout)
            for n, g_buf in (("q", "GQ"), ("k", "GK"), ("v", "GV")):
                xn = f"XN{min('qkv'.index(n), len(inputs) - 1)}"
                gr[g_buf] = 2 * o.plans[(L, n)].layout.N
                rl += [(g.op, tok(xn, jj, g.op, i=0), self._w(g), tok(g_buf, jj, g.op, i=2, row=gr[g_buf]))
                       for g in o.plans[(L, n)].gemms for jj in range(S)]
            ring = f"KV[{ov.raw[v]}:{ov.raw[v] + COLS * RING_BYTES}]"
            for b, prep in enumerate(self.preps[(x.kind, T)]):  # (its parts in order: each reads the ring before it)
                pt = prep.T
                rl.append((prep, *(tok(b_, 0, prep, i=i, extra=b * pt * gr[b_]) for i, b_ in enumerate(("GQ", "GK", "GV"))),
                           sl("TAU", 0, prep, i=3), self._prm(L, prep),
                           tok("QS", 0, prep, i=5, extra=b * pt * self.rb["QS"]), self._kv(v, prep, 6), ring,
                           sl("JUNK", 0, prep, i=8)))
            for j, at in enumerate(self._attns(x.kind, T)):  # (its parts after every prep part: the chunk's records)
                rl.append((at, sl("QS", j * at.T * self.rb["QS"], at, i=0), self._kv(v, at, 1),
                           sl("A", j * at.T * self.rb["A"], at, i=2), self._eb(L, at)))
            gt = o.gate
            rl += [(gt, tok("A", jj, gt, i=0), tok("GQ", jj, gt, i=1, row=gr["GQ"]), tok("A", jj, gt, i=2)) for jj in range(S)]
            (g,) = o.plans[(L, "o")].gemms
            rl += [(g.op, tok("A", jj, g.op, i=0), self._w(g), tok("Y", jj, g.op, i=2)) for jj in range(S)]
            rs = self.resid[(T, bool(x.block))]
            hist = f"KV[{ov.hist[(v, 0)]}:{ov.hist[(v, 0)] + arg_bytes(rs, 2 if rs.block else 3)}]"
            dst = w.new if x.block else w.cur
            args = [sl("Y", 0, rs, i=0)] + ([] if rs.block else [sl(w.cur, 0, rs, i=1)])
            rl.append((rs, *args, self._tp(L, 0, rs), hist, sl(dst, 0, rs, i=len(args) + 2)))
            if self.ffn:
                marks.append(self._ffn(rl, o, S, tok, v, T))
        self._starts[T] = [*starts, len(rl)]
        self._marks[T] = marks
        return rl

    def _control(self, rl: list[tuple], o: SimpleNamespace, S: int, tok: object, v: int, s: str, xs: str, out: str) -> str:
        """Visit v's controller of stream s on the rows xs (the bank's message with the stream's query, the down halves
        on the norms without weights, the blend) into the rows ``out``; returns ``out``."""
        w, L = self.walk[v], self.vis[v].layer
        bl, mx2 = o.blend, o.mixes[2]
        si = self.ctl.index(s)
        rl += [(mx2, tok(w.bank[0], jj, mx2, i=0), tok(w.bank[1], jj, mx2, i=1), self._qr(L, 4 + si),
                tok("MS", jj, mx2, i=3)) for jj in range(S)]
        for h, src in (("a", xs), ("b", "MS")):
            nm = o.norm1
            rl += [(nm, tok(src, jj, nm, i=0), self._np(L, 3), tok("DP", jj, nm, i=2)) for jj in range(S)]
            p = o.plans[(L, f"d{s}{h}")]
            rl += [(g.op, tok("DP", jj, g.op, i=0), self._w(g), tok(f"G{h.upper()}", jj, g.op, i=2))
                   for g in p.gemms for jj in range(S)]
        reg = self.img.region(L, f"ctl_{s}")
        wsl = f"W[{reg.offset}:{reg.offset + arg_bytes(bl, 4)}]"
        rl += [(bl, tok("GA", jj, bl, i=0), tok("GB", jj, bl, i=1), tok(xs, jj, bl, i=2), tok("MS", jj, bl, i=3),
                wsl, tok(out, jj, bl, i=5)) for jj in range(S)]
        return out

    def _ffn(self, rl: list[tuple], o: SimpleNamespace, S: int, tok: object, v: int, T: int) -> tuple[int, int, str]:
        """Visit v's FFN (module docstring) after its attention's residual; returns its host step's mark (the visit,
        the item it precedes, the FFN input's rows)."""
        x, w, sl = self.vis[v], self.walk[v], self._sl
        L, t = x.layer, self._t(T)
        cur = w.new if x.block else w.cur  # (cur after the attention)
        states = w.blocks + ([w.cur] if x.block else []) + [cur]
        mx = o.mixes[len(states)]
        rl += [(mx, *(tok(s_, jj, mx, i=i) for i, s_ in enumerate(states)), self._qr(L, 3),
                tok("XM0", jj, mx, i=len(states) + 1)) for jj in range(S)]
        xf = self._control(rl, o, S, tok, v, "ffn", "XM0", "XC0") if x.messages else "XM0"
        nm = o.norm
        rl += [(nm, tok(xf, jj, nm, i=0), self._np(L, 4), tok("XN0", jj, nm, i=2)) for jj in range(S)]
        rp = self.moe.router(L, t)
        rl += [(g.op, tok("XN0", jj, g.op, i=0), self._w(g), tok("LG", jj, g.op, i=2)) for g in rp.gemms for jj in range(S)]
        mark = (v, len(rl), xf)
        rl += self.moe.items(L, T, sl=lambda b, off, op, i: sl(b, off, op, i=i), norm=self._np(L, 4))
        rs = self.fresid[T]
        hist = f"KV[{self.ov.hist[(v, 1)]}:{self.ov.hist[(v, 1)] + arg_bytes(rs, 3)}]"
        rl.append((rs, sl("DL", 0, rs, i=0), sl(cur, 0, rs, i=1), self._tp(L, 1, rs), hist, sl(cur, 0, rs, i=4)))
        return mark

    def _kv(self, v: int, op: object, i: int) -> str:
        """Visit v's caches as the argument i of the prep / attention: from its kv0 before the cache (a full visit's 16
        groups: the near part's reads before group 0, skipped)."""
        ov = self.ov
        full = self.vis[v].kind == "full"
        o = ov.kv_off[v] - (self.guard * self.shape.group if full else 0)
        return f"KV[{o}:{o + arg_bytes(op, i)}]"

    def _layer_starts(self, T: int) -> list[int]:
        return self._starts[T]

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        for name, r in self.rb.items():  # (the GEMMs' padding rows past the last sub-chunk)
            if name in own:
                own[name] = max(own[name], (self.sizes[-1] + 64) * r)
        n = len(self.phys)
        own["NP"] = max(own.get("NP", 0), n * self.np_w)
        own["QR"] = max(own.get("QR", 0), n * self.qr_w)
        own["PRM"] = max(own.get("PRM", 0), n * self.prm_w)
        own["ET"] = max(own.get("ET", 0), n * max(self.e_w.values()))
        own["TP"] = max(own.get("TP", 0), n * self.tp_w)
        for name, b in (self.moe.fixed() if self.moe else {}).items():
            own[name] = max(own.get(name, 0), b)

    def _bind(self) -> None:
        """The norm weights (NP), queries (QR), the preps' constants (PRM), the E tables (EB), the post convs' taps (TP),
        tau per position (TAU)."""
        dev, H, g, cfg = self.ov.dev, self.H, self.ov.gguf, self.cfg
        C = H // COLS

        def vec(name: str) -> np.ndarray:
            return np.asarray(g.tensor(name).dequantize(), np.float32).reshape(-1)

        npb, qr = dev.view("NP"), dev.view("QR")
        prm, eb, tp = dev.view("PRM"), dev.view("ET"), dev.view("TP")
        prep = next(iter(self.preps.values()))[0]
        for L, i in self.phys.items():
            b = f"blk.{L}."
            full = cfg.kinds[L] == "full"
            norms = [b + f"attn_{n}_inp_norm.weight" for n in "qkv"] if full else [b + "attn_norm.weight"] * 3
            ws = [vec(n) for n in norms] + [np.ones(H, np.float32), vec(b + "ffn_norm.weight")]
            for j, wv in enumerate(ws):  # (index 3: ones, the controllers' downs' norms; 4: the FFN's)
                o = i * self.np_w + j * 2 * H
                npb[o:o + 2 * H] = wv.astype(bfloat16).view(np.uint8)
            q = np.zeros((8, H), np.float32)
            if L > 0:
                rq = vec(b + "attn_res_query.weight").reshape(-1, H)
                q[:len(rq)] = rq
            q[3] = vec(b + "ffn_res_query.weight")
            if cfg.core_start <= L < cfg.core_end:
                q[4:8] = cfg.msg_queries(g, L)
            qr[i * self.qr_w:(i + 1) * self.qr_w] = q.view(np.uint8).reshape(-1)
            qc = vec(b + "attn_q_clip.weight").reshape(-1, 128)
            rc = vec(b + "attn_rel_clip.weight").reshape(-1, cfg.rel_length)
            kt = vec(b + "attn_k_conv.weight").reshape(-1, cfg.conv_taps).T
            vt = vec(b + "attn_v_conv.weight").reshape(-1, cfg.conv_taps).T
            pb = prep.prm(qc, rc, kt, vt)
            prm[i * self.prm_w:i * self.prm_w + pb.size] = pb
            E = vec(b + "attn_rel_bias.weight").reshape(-1, cfg.rel_length).T  # (16, extent)
            k = "full" if full else "swa"
            at = self._attns(k, self.sizes[-1])[0] if k in self.kinds else next(iter(self.attn.values()))
            e = at.e_table(E).view(np.uint8)
            eb[i * max(self.e_w.values()):][:e.size] = e
            for k, name in enumerate((b + "attn_post_conv.weight", b + "ffn_post_conv.weight")):
                taps = vec(name).reshape(-1, cfg.conv_taps).T  # (4, H)
                t4 = np.ascontiguousarray(np.stack([taps[:, C * c:C * (c + 1)] for c in range(COLS)]).astype(bfloat16))
                o = i * self.tp_w + k * self.tp_w // 2
                tp[o:o + t4.nbytes] = t4.view(np.uint8).reshape(-1)
        tau = cfg.tau(np.arange(self.own["TAU"] // 4)).astype(np.float32)
        dev.view("TAU")[:tau.nbytes] = tau.view(np.uint8)
        for name in ("NP", "QR", "PRM", "ET", "TP", "TAU"):
            dev.flush(name)
        # (QS: the INK unit's slot of a query block's fourth row block, zero where nothing writes; DS: the combine's ds
        # rows, zero but the weights the host writes)
        for name in ("QS", "DS") if self.moe else ("QS",):
            dev.view(name)[:] = 0
            dev.flush(name)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def fits(self, T: int, p0: int) -> bool:
        """(and within one turn of the sliding-window caches' ring: its window's history before the chunk is the ring's
        or its guard's, ``AttentionOverlay.mirror``)"""
        w, pg = self.wrap, self.shape.pg
        return super().fits(T, p0) and (not w or (p0 // pg) % w + -(-T // pg) <= w)

    @staticmethod
    def ring(max_positions: int) -> int:
        """The sliding-window caches' ring (groups; 0: linear) of an overlay of ``max_positions`` that a prefill runs on:
        two of the largest chunk (chunks of it start at its multiples, each within one turn: ``fits``), linear where the
        capacity holds no more (33K positions: 3.3 GB of caches against 5.6 GB linear; 128K: 6 GB against 22)."""
        w = 2 * SIZES[-1] // CG
        return w if -(-max_positions // CG) > w else 0

    def params(self, T: int, p0: int, n: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0 of n valid tokens."""
        out = {}
        for k in self.kinds:
            for prep in self.preps[(k, T)]:
                out |= prep.params(p0, n)
            for j, at in enumerate(self._attns(k, T)):
                out |= at.params(p0 + j * at.T)
        a = p0 - 3 - self.ov._hbase[self.ov.seq]  # (the conv histories' first row in the windows)
        out |= self.resid[(T, False)].params(n, a)
        return out

    def run(self, e: np.ndarray, p0: int, T: int = 0) -> None:
        """A chunk of size T (default: the smallest that holds e) at p0: e (n, hidden) the tokens' normed embeddings
        (the first visit's input state). Leaves the visits' K / V of the positions in the caches, the raw rings and the
        history windows the decode continues from, the last visit's output in its state."""
        ov, dev = self.ov, self.ov.dev
        n = e.shape[0]
        T = self._size(n, p0, T)
        H = self.H
        if "E" in self.own:  # (the mixes' state 0; a slice of visit 0 alone reads only EB)
            ev = dev.view("E")[:4 * T * H].view(np.float32).reshape(T, H)
            ev[:n], ev[n:] = e, 0
            dev.flush("E", 0, 4 * T * H)
        eb = dev.view("EB")[:2 * T * H].view(bfloat16).reshape(T, H)
        eb[:n], eb[n:] = e.astype(bfloat16), 0
        dev.flush("EB", 0, 2 * T * H)
        ov.mirror(p0)  # (a ring's window before the chunk across its wrap: the guard)
        hooks = {i: functools.partial(self.moe.host_route, self.vis[v].layer, n=n, T=T, t=self._t(T), xf=xf, dev=dev,
                                      lib=self.lib) for v, i, xf in self._marks.get(T, ())}
        # (the host's routing reads the router's logits: after the piece that ends with the router)
        self._launch(T, p0, self.params(T, p0, n), hooks or None, synced=tuple(hooks))
        dev.stale = True  # (the chunk's configurations replaced the attention cores' code pages)
        ov._hbase[ov.seq] = p0 + n - 3  # (the windows' rows 0 .. 2: the chunk's last 3 valid tokens')

    def state(self, name: str, i: int, n: int) -> np.ndarray:
        """Rows i .. i + n - 1 (n, hidden) fp32 of a state buffer the last chunk left."""
        dev, H = self.ov.dev, self.H
        dev.fetch(name, 4 * i * H, 4 * n * H)
        return dev.view(name)[4 * i * H:4 * (i + n) * H].view(np.float32).reshape(n, H).copy()

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def layer_seconds(self, L: int, p0: int, S: float = 1) -> float:
        """(module constants: per sub-chunk the attention side and a loop visit's controllers, the ConfigureOps, the
        attention by the cache groups its passes stream, the MoE by its busiest column's expert chunks: the experts hit
        by T tokens' routing, or ~0.62 of them plus a chunk per CHUNK pairs (the routing statistics of 8192 essay tokens:
        the busiest column within ~6%), per token the gathered norm, the combine and the routing)"""
        x = self.vis[L]
        T = S * SUB
        out = S * (SUB_S + (CTL_S if x.messages else 0)) + self.layer_configs(L) * CFG_S + self._attn_seconds(x.kind, p0, T)
        if self.moe is not None:
            E, k = self.cfg.n_experts, self.cfg.n_experts_used
            chunks = max(E * (1 - (1 - k / E) ** T), 0.62 * E + T * k / CHUNK)
            out += chunks / COLS * CHUNK_S + T * (MOE_TOK_S + ROUTE_TOK_S) + ROUTE_S
        return out

    def _attn_seconds(self, kind: str, p0: int, T: float) -> float:
        """A visit's attention over T tokens at p0: per pass of 32 queries its near part's groups (the window and the
        pass's own; a full layer's also the item's groups before the pass) and a full layer's far part's (from position
        0 to 16 groups before the item: at least one)."""
        pg, items = self.shape.pg, max(1, round(T / ATTN_T))
        t = T / items  # (an item's tokens)
        out = 0.0
        for j in range(items):
            passes, ng = t / 32, t / pg
            if kind == "swa":
                out += passes * (self.guard + 1) * NEAR_S
            else:
                far = max(1.0, (p0 + j * t) / pg - self.guard)
                out += passes * ((self.guard + 1 + (ng - 1) / 2) * NEAR_S + far * FAR_S)
        return out

    def layer_configs(self, L: int) -> int:
        return 12 + (16 if self.vis[L].messages else 0) + (10 if self.ffn else 0)


__all__ = ["SIZES", "InklingPrefill"]
