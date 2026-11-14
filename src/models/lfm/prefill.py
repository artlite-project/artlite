"""The prefill of an LFM3 model: a prompt chunk of up to ``MAX_T`` tokens through every layer on the decode overlay's
buffers (``overlay.LfmOverlay``: the weight image, the KV caches, the conv layers' regions), so the decode continues from
the caches and conv histories the chunk leaves; the prompt's last token's logits from the LM head alone on the chunk's
last output row (``LfmOverlay.logits_of``).

A chunk of T tokens (``SIZES``: ``SMALL``, multiples of 256) runs layer-major in sub-chunks of ``SUB`` tokens, of
``BIG`` where T is a multiple of it (the input projections in their ``SUB``-token items; a ``SMALL`` size: one of its
own): each operator of a layer runs over the sub-chunks back to back on buffer slices a sub-chunk apart (a ConfigureOp
per layer and operator), the attention preps in blocks of ``PREP_SIZES``, the norms in blocks of ``NORM_SIZES``, the
conv and the flash attention once over the whole chunk (QAD-Q4_0, 32K tokens: 6.75 s in 256-token sub-chunks, 6.33 s
with BIG). The residual stream x is fp32 (PX, ``prefill.norm.ResidualNormOperator``: N0 measured a bf16 one 35% worse
on chats). Per layer L (the decode's order):

- conv layers: in GEMM (``prefill.gemm``) of xn -> G ([B | C | x] per column); the conv (``conv_prefill``: the window
  over the chunk's tokens, from the layer's conv regions' history, back into them) -> A (out's panels); out GEMM -> Y;
- attention layers: qkv GEMM of xn -> G; the attention prep of the layer's kind and qkv layout (``prefill.attn``: Q
  units and gates into QS, the chunk's K / V records into the layer's cache; sliding-window caches after their guard;
  Q4_K_M's layers with a planar Q6_K v place their gate logits elsewhere), the flash
  attention over the chunk (two columns per KV head on one broadcast cache stream; a sliding-window pass streams its
  window) -> A; o GEMM -> Y;
- norm ``post``: x += norm(Y) post_attention_norm, xn = norm(x) ffn_norm; gate_up GEMMs with the SiLU epilogue -> HP;
  down GEMMs in two K parts -> F; norm ``post``: x += norm(F) post_ffw_norm, xn = norm(x) of the next layer's attn_norm
  (the last layer: x, ``hidden``).

The first layer's input norm (``in``) starts the chunk. The control code is assembled on the host from one library ELF
and runs in pieces of whole layers (``prefill.chunks.ChunkPrefill``).

A non-causal model (``LfmConfig.causal`` False: d1's decision models) reads a prompt in one chunk from position 0: its
full layers' attention sees the prompt both ways as a state and questions after it (``run``'s ``sets``: the state's
tokens every token, a question's the state's and its own; ``prefill.attn.AttnOperator(causal=False)``, the bounds in
SEG), its sliding-window and conv layers stay causal, the conv from zero histories.

Chunks start at a cache group (64 positions) and hold at least ``MIN_TOKENS`` valid tokens (the conv history's 6 rows).
Before a chunk the host moves each conv region's history to its base (the 6 rows before the chunk: ``LfmOverlay.conv_base``);
the chunk's conv leaves its last 6 valid rows there (``LfmOverlay.conv_at``). The operator cores' code pages are reloaded
before the next decode step (the chunk's configurations replace them: ``OverlayDevice.stale``).
"""

import itertools
from dataclasses import dataclass

import numpy as np

from artlite.operators.design import COLS, MAX_BD_STEP_WORDS
from artlite.operators.util import arg_bytes
from artlite.prefill import attn as PA
from artlite.prefill import gemm as PG
from artlite.prefill.chunks import ChunkPrefill, norm_blocks
from artlite.prefill.norm import ResidualNormOperator, residual_params

from . import conv as CV
from .conv_prefill import ConvPrefillOperator
from .overlay import KV_PAD, LfmOverlay

SUB = 256  # tokens per sub-chunk (the operators' T)
BIG = 1024  # the sub-chunk of the chunks made of them (out / o, gate_up, down once per 1024 tokens: their weights
# stream a quarter as often; in and qkv's items stay SUB, as Llama's: a 1024-token input projection is no faster), or
# half of it where a block's down parts would lie too far apart for the norm (``D_STRIDE``: Q4_K_M's planar Q6_K downs
# at a pitch of 288)
PREP_SIZES = (1024, 512)  # tokens per attention prep item where a chunk is made of them
NORM_SIZES = (1024, 512)  # tokens per norm item where a chunk holds them (its fill and drain once per item; those whose
# down parts lie within D_STRIDE)
D_STRIDE = 2 * MAX_BD_STEP_WORDS  # bf16 between a norm block's down parts at most (a shim BD dimension's step;
# ``ResidualNormOperator`` refuses more)
SMALL = (64, 128)  # chunk sizes of operators of their own (short prompts: fewer passes)
SIZES = SMALL + tuple(range(SUB, 2049, SUB)) + (3072, 4096)
MAX_T = SIZES[-1]
DOWN_CHUNKS = 4  # K chunks per down part (2 parts of 4 of the ffn's 8: each part's A strip stays in the memtile, each
# sums 64 k-tiles in its bf16 C tile; the norm adds the parts in fp32)
INPUTS = ("in", "qkv")  # (the input projections: SUB-token items)
# the time model's constants (``layer_seconds``; fitted to QAD-Q4_0's chunk times, T = 64 .. 4096 at p0 = 0 .. 28672:
# rms 5%, within 13%; a sliding-window pass and the conv fit to nothing beside the weights): a ConfigureOp, per
# sub-chunk a byte of a layer's weights, a cache record a full attention pass streams
CFG_S, WEIGHT_S, RECORD_S = 0.097e-3, 3.75e-11, 0.288e-6


def plan_gemm(img: object, region: object, T: int, *, pitch: int = 0, ops: dict[str, object] | None = None
              ) -> PG.PhasePlan:
    """A GEMM phase of the prefill (``prefill.gemm.plan_phase``): gate / up with the SiLU epilogue, down in K parts of
    ``DOWN_CHUNKS`` chunks."""
    return PG.plan_down_parts(img, region, T, pitch=pitch, ops=ops, act="silu", down_chunks=DOWN_CHUNKS)


@dataclass
class _Ops:
    """The operators of a sub-chunk size: the GEMM plans, the norms, the attention preps (per kind, qkv layout and
    block size)."""

    plans: dict[tuple[int, str], PG.PhasePlan]
    qt: int  # (the input projections' tokens per item)
    pitch: dict[str, int]  # (the input projections' column pitch: in, qkv)
    d_pitch: int  # (the output projections')
    f_part: int  # (bytes of a down part's C)
    n_in: ResidualNormOperator
    norms: dict[int, tuple[ResidualNormOperator, ResidualNormOperator]]  # (per block size: after o / out, after down)
    preps: dict[tuple[str, tuple, int], PA.PrepOperator]


class LfmPrefill(ChunkPrefill):
    """The prefill of an overlay (its layers) in chunks of the sizes ``sizes`` (those within the overlay's caches); ``run``
    runs one."""

    SUB, SMALL, MAX_T, NAME, KV_PAD = SUB, SMALL, MAX_T, "lfm_prefill", KV_PAD
    SHARED_KV_RANGE = True  # (the attention's P_FO: one range for every size)
    CFG_S, WEIGHT_S = CFG_S, WEIGHT_S
    MIN_TOKENS = CV.WIDTH - 1

    def __init__(self, overlay: LfmOverlay, *, sizes: tuple[int, ...] = SIZES, precise: bool = True):
        """``precise``: the attention's P as hi + lo panels (``prefill.attn.Shape.pp``: the P rounding's error gone;
        N0's largest prefill rounding on chats)."""
        ov = overlay
        s = PA.Shape(ov.hd, ov.nh // 2, ov.kp, ov.T, 1, 2 if precise else 1, per=2, gate=True)
        super().__init__(overlay, tuple(t for t in sizes if t % ov.pg == 0))
        self.layers = ov.layers
        self.kv_layers = [L for L in self.layers if ov.kind(L) != "conv"]
        self.shape = s
        self.group = s.pg
        self.cs_rows = ov.max_positions + MAX_T
        self.pad = KV_PAD // s.pg * s.group  # (the KV arguments past a layer's caches: the overlay's padding)
        # (the preps': a sub-chunk's groups, so that P_KV's one range, a sub-chunk's prep's, reaches a SMALL chunk at
        # the caches' end; their slices start a block's groups apart, within the rings and KV_PAD past the caches)
        self.prep_pad = SUB // s.pg * s.group
        kinds = sorted({ov.kind(L) for L in self.kv_layers})
        self.kv_bytes = {ov.kind(L): ov.kv_bytes(L) for L in self.kv_layers}  # (per KV head, a guard included)
        self.kv0 = {"swa": ov.guard * s.group, "full": 0}
        self.win = {"swa": self.cfg.sliding_window, "full": 0}
        self._starts: dict[int, list[int]] = {}  # (per chunk size: its layers' first items, ``_layer_starts``)
        self._norm_ops: dict[tuple, ResidualNormOperator] = {}  # (one object per distinct norm)
        self.big = BIG if BIG * COLS * PG.d_pitch(self.img, BIG) <= D_STRIDE else BIG // 2
        gemms: dict[str, object] = {}  # (one object per distinct GEMM)
        self.ops = {t: self._ops(t, gemms) for t in sorted({self._t(T) for T in self.sizes})}
        self.plans = self.ops[SUB].plans if SUB in self.ops else self.ops[max(self.ops)].plans
        self.nc = not self.cfg.causal  # (the full layers' attention non-causal: d1)
        self.attn = {(k, T): PA.AttnOperator(T=T, shape=s, kv_bytes=self.kv_bytes[k], kv_pad=self.pad, sub=self._pt(T),
                                             swa=self.win[k], kv0=self.kv0[k], causal=k != "full" or not self.nc)
                     for k in kinds for T in self.sizes}
        self.conv = {T: ConvPrefillOperator(T=T, pitch=self.ops[self._t(T)].pitch["in"]) for T in self.sizes}
        self._rows()
        self._library()

    def _t(self, T: int) -> int:
        """The sub-chunk of a chunk of size T: ``big`` where the chunk is made of them, else SUB (a short chunk:
        itself)."""
        return self.big if T % self.big == 0 else min(T, SUB)

    @staticmethod
    def _pt(T: int) -> int:
        """The prep's block of a chunk of size T: the largest of PREP_SIZES the chunk is made of, else a sub-chunk."""
        return next((n for n in PREP_SIZES if T % n == 0), min(T, SUB))

    def _phases(self, L: int) -> tuple[str, ...]:
        """Layer L's GEMM phases."""
        return ("in", "out", "gate_up", "down") if self.ov.kind(L) == "conv" else ("qkv", "o", "gate_up", "down")

    def _ops(self, t: int, gemms: dict[str, object]) -> _Ops:
        """The operators of sub-chunks of t tokens (the input projections' of qt: SUB in BIG ones)."""
        ov, cfg, img, H = self.ov, self.cfg, self.img, self.H
        qt = SUB if t == self.big else t
        dp = PG.d_pitch(img, t)
        pitch = {n: PG.phase_pitch(img, n, qt) for n in INPUTS if any(r.name == n for r in img.regions)}
        plans = {(L, n): plan_gemm(img, img.region(L, n), qt if n in INPUTS else t, ops=gemms,
                                   pitch=dp if n in PG.OUT else pitch.get(n, 0))
                 for L in self.layers for n in self._phases(L)}
        parts = {plans[(L, "down")].parts for L in self.layers}
        assert parts == {2}, parts
        f_part = max(arg_bytes(g.op, 2) for (L, n), p in plans.items() if n == "down" for g in p.gemms)
        for n, pt in pitch.items():  # (the conv and the preps read every layer's rows at one pitch)
            assert {p.layout.pitch for (L, m), p in plans.items() if m == n} == {pt}, n
        eps = cfg.rms_norm_eps
        n_in = ResidualNormOperator(T=t, E=H, mode="in", eps=eps)
        big = max(T for T in self.sizes if self._t(T) == t)
        norms = {n: self._norms(n, dp) for n in {t, *(m for m in NORM_SIZES if t <= m <= big)}
                 if n * COLS * dp <= D_STRIDE}
        assert t in norms, (t, dp)
        preps = {}  # (a layer's qkv layout places its blocks: Q4_K_M's layers with a planar Q6_K v put g 32 columns on)
        for L in self.kv_layers:
            k, lay = ov.kind(L), plans[(L, "qkv")].layout
            for pt in {self._pt(T) for T in self.sizes if self._t(T) == t}:
                if (k, lay.key, pt) not in preps:
                    preps[(k, lay.key, pt)] = PA.PrepOperator(
                        T=pt, shape=self.shape, layout=lay, kv_bytes=self.kv_bytes[k], cs_positions=self.cs_rows,
                        kv_pad=self.prep_pad, norm=True, eps=eps, kv0=self.kv0[k])
        return _Ops(plans, qt, pitch, dp, f_part, n_in, norms, preps)

    def _norms(self, n: int, dp: int) -> tuple[ResidualNormOperator, ResidualNormOperator]:
        """The norms of a block of n tokens: after o / out (Y), after down (F: its two parts n tokens apart)."""
        eps, H = self.cfg.rms_norm_eps, self.H

        def norm(deltas: int) -> ResidualNormOperator:
            key = (n, dp, deltas)
            if key not in self._norm_ops:
                self._norm_ops[key] = ResidualNormOperator(T=n, E=H, mode="post", eps=eps, d_pitch=dp, deltas=deltas,
                                                           d_stride=n * COLS * dp if deltas > 1 else 0)
            return self._norm_ops[key]

        return norm(1), norm(2)

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers (the sub-chunk sizes' layouts: at most the largest); the norms'
        parameter objects ([8][H] bf16 each), the q / k norms' weights (NW)."""
        H, s = self.H, self.shape
        pitch = max(p for o in self.ops.values() for p in o.pitch.values())
        dp = max(o.d_pitch for o in self.ops.values())
        self.rb = {"PX": 4 * H, "PXN": 9 * H // 8, "G": 2 * COLS * pitch, "QS": COLS * s.tgu // 8,
                   "A": 9 * COLS * s.qh * s.hd // 8, "Y": 2 * COLS * dp,
                   "HP": 9 * -(-self.cfg.intermediate_size // self.img.kc) * self.img.kc // 8}
        self.np_off = {"in": 0}
        k = 1
        for L in self.layers:
            for n in ("o", "d"):
                self.np_off[(L, n)] = k * 16 * H
                k += 1
        self.np_bytes = k * 16 * H
        self.nw = PA.PrepOperator.nw_bytes(s.hd)
        self.nw_off = {L: i * self.nw for i, L in enumerate(self.kv_layers)}

    def _np(self, key: object, op: object) -> str:
        o = self.np_off[key]
        return f"NP[{o}:{o + arg_bytes(op, 0)}]"

    def _runlist(self, S: int) -> list[tuple]:
        return self._runlist_of(S * SUB)

    def _runlist_of(self, T: int, *, packed: bool = False, tree: bool = False) -> list[tuple]:
        """(the sub-chunks of ``_t``: a size below SUB one of its own operators; the input projections in items of
        ``qt`` tokens, the preps in blocks of ``_pt``, the norms in blocks of ``norm_blocks``)"""
        assert not packed and not tree, "LFM3: no packed chunks yet"
        t = self._t(T)
        o, S, qt, pt = self.ops[t], T // t, self.ops[t].qt, self._pt(T)
        ov, s, sl, rb = self.ov, self.shape, self._sl, self.rb

        def tok(buf: str, a: int, op: object, *, i: int, row: int = 0) -> str:  # (from token a on)
            return sl(buf, a * (row or rb[buf]), op, i=i)

        ry = 2 * COLS * o.d_pitch  # (bytes per token of Y and F at these operators' pitch: a norm block's rows)
        # norm blocks; F per block [part][its tokens], then the down GEMMs' spill (their C rows past t tokens)
        blocks = norm_blocks(T, t, [m for m in NORM_SIZES if m in o.norms])
        spill = o.f_part - t * ry
        f_base = list(itertools.accumulate((2 * nt * ry + spill for _, nt in blocks), initial=0))

        def f_at(tk: int, part: int) -> int:  # (token tk's row of a down part in F)
            k = max(i for i, (a, _) in enumerate(blocks) if a <= tk)
            return f_base[k] + (part * blocks[k][1] + tk - blocks[k][0]) * ry

        rl: list[tuple] = [(o.n_in, self._np("in", o.n_in), tok("PX", j * t, o.n_in, i=1),
                            tok("PXN", j * t, o.n_in, i=2)) for j in range(S)]
        starts = []
        for L in self.layers:
            starts.append(len(rl))
            kind = ov.kind(L)
            plan = o.plans[(L, "in" if kind == "conv" else "qkv")]
            rg = 2 * COLS * plan.layout.pitch
            rl += [(g.op, tok("PXN", j * qt, g.op, i=0), self._w(g), tok("G", j * qt, g.op, i=2, row=rg))
                   for g in plan.gemms for j in range(T // qt)]
            if kind == "conv":
                cv = self.conv[T]
                h0 = ov.conv_reg[(L, 0)]
                rl.append((cv, sl("G", 0, cv, i=0), f"ACT[{h0}:{h0 + arg_bytes(cv, 1)}]", sl("A", 0, cv, i=2)))
            else:
                prep, attn = o.preps[(kind, plan.layout.key, pt)], self.attn[(kind, T)]
                kv0, kvb = ov.kv_off[L], arg_bytes(prep, 3)
                gb = pt // s.pg * s.group  # (a block's cache groups)
                cs = "CSS" if kind == "swa" else "CSF"
                nw = f"NW[{self.nw_off[L]}:{self.nw_off[L] + self.nw}]"
                rl += [(prep, tok("G", b * pt, prep, i=0, row=rg), sl(cs, b * pt * 4 * s.hd, prep, i=1),
                        tok("QS", b * pt, prep, i=2), f"KV[{kv0 + b * gb}:{kv0 + b * gb + kvb}]", nw)
                       for b in range(T // pt)]
                seg = [sl("SEG", 0, attn, i=3)] if attn.nc else []  # (non-causal: its rows' keys, ``run``)
                rl.append((attn, sl("QS", 0, attn, i=0), f"KV[{kv0}:{kv0 + arg_bytes(attn, 1)}]", sl("A", 0, attn, i=2),
                           *seg))
            (g,) = o.plans[(L, "out" if kind == "conv" else "o")].gemms
            rl += [(g.op, tok("A", j * t, g.op, i=0), self._w(g), tok("Y", j * t, g.op, i=2, row=ry)) for j in range(S)]
            for a, nt in blocks:
                n = o.norms[nt][0]
                rl.append((n, self._np((L, "o"), n), tok("PX", a, n, i=1), tok("Y", a, n, i=2, row=ry),
                           tok("PX", a, n, i=3), tok("PXN", a, n, i=4)))
            rl += [(g.op, tok("PXN", j * t, g.op, i=0), self._w(g), tok("HP", j * t, g.op, i=2))
                   for g in o.plans[(L, "gate_up")].gemms for j in range(S)]
            rl += [(g.op, sl("HP", j * t * rb["HP"] + 2 * g.a_off, g.op, i=0), self._w(g),
                    sl("F", f_at(j * t, g.part), g.op, i=2)) for g in o.plans[(L, "down")].gemms for j in range(S)]
            for (a, nt), base in zip(blocks, f_base, strict=False):
                n = o.norms[nt][1]
                rl.append((n, self._np((L, "d"), n), tok("PX", a, n, i=1), sl("F", base, n, i=2), tok("PX", a, n, i=3),
                           tok("PXN", a, n, i=4)))
        self._starts[T] = [*starts, len(rl)]
        return rl

    def _layer_starts(self, T: int) -> list[int]:
        return self._starts[T]

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        for name, r in self.rb.items():  # (the GEMMs' padding rows past the last sub-chunk)
            own[name] = max(own.get(name, 0), (self.sizes[-1] + 64) * r)
        own["F"] = own["F"] + max(o.f_part for o in self.ops.values())

    def _bind(self) -> None:
        """The norms' parameter objects (NP), the q / k norms' weights (NW), the cos / sin tables (CSS, CSF)."""
        ov, dev, H = self.ov, self.ov.dev, self.H
        npb = dev.view("NP").view(np.uint16)

        def vec(name: str) -> np.ndarray:
            return np.asarray(ov.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

        def put(key: object, wp: str | None, w1: str | None) -> None:
            o = self.np_off[key] // 2
            v = residual_params(None if wp is None else vec(wp), None if w1 is None else vec(w1), 1.0, H)
            npb[o:o + v.size] = v.reshape(-1)

        put("in", None, f"blk.{self.layers[0]}.attn_norm.weight")
        for i, L in enumerate(self.layers):
            b = f"blk.{L}."
            put((L, "o"), b + "post_attention_norm.weight", b + "ffn_norm.weight")
            put((L, "d"), b + "post_ffw_norm.weight", f"blk.{self.layers[i + 1]}.attn_norm.weight"
                if i + 1 < len(self.layers) else "output_norm.weight")
        dev.flush("NP")
        nw = dev.view("NW")
        for L, o in self.nw_off.items():
            w = np.concatenate([vec(f"blk.{L}.attn_{n}_norm.weight") for n in "qk"]).view(np.uint8)
            nw[o:o + w.size] = w
        dev.flush("NW")
        for kind, name in (("swa", "CSS"), ("full", "CSF")):
            inv = np.zeros(self.shape.hd // 2)
            f = self.cfg.rope_inv_freq(kind)
            inv[:f.size] = f  # (the pairs past rot: cos 1, sin 0)
            cs = PA.cs_table(inv, self.cs_rows).view(np.uint8).reshape(-1)
            dev.view(name)[:cs.size] = cs
            dev.flush(name)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def params(self, T: int, p0: int, n: int, sets: tuple[int, ...] = ()) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0 with n valid tokens (non-causal: ``sets``, its
        attention's bounds)."""
        out = ConvPrefillOperator.params(n)
        for (k, _, pt), prep in self.ops[self._t(T)].preps.items():
            if pt == self._pt(T):
                at = self.attn[(k, T)]
                out |= prep.params(p0) | (at.set_params(sets) if at.nc else at.params(p0))
        return out

    def run(self, x: np.ndarray, p0: int, T: int = 0, *, sets: tuple[int, ...] | None = None) -> None:
        """A chunk of size T (default: the smallest that holds x): x (n, hidden) the embeddings of positions p0 .. p0 +
        n - 1 (the caches and conv histories hold the positions before). Leaves the attention layers' K / V of the
        positions in the caches and the conv histories at p0 + n. A non-causal model's prompt: at p0 0 (zero
        histories), ``sets`` its bounds (``NpuModel.outputs``; default (n,): every token sees every token)."""
        n, dev = x.shape[0], self.ov.dev
        T = self._size(n, p0, T)
        assert n >= self.MIN_TOKENS, n
        if self.nc:
            sets = tuple(sets or (n,))
            assert p0 == 0 and sets[-1] == n, (p0, sets, n)
            seg = self.attn[("full", T)].set_seg(sets)
            dev.view("SEG")[:seg.size] = seg
            dev.flush("SEG", 0, seg.size)
        else:
            assert sets is None, "a causal model's prompt sees what is before it"
        self._stage_px(x, T, np.float32)
        self.ov.conv_base(p0)
        self._launch(T, p0, self.params(T, p0, n, sets or ()))
        self.ov.conv_at(p0 + n)
        dev.stale = True  # (the chunk's configurations replaced the operator cores' code pages)

    def hidden(self, i: int, n: int = 1) -> np.ndarray:
        """Rows i .. i + n - 1 (n, hidden) fp32 of the last chunk's output residual (the LM head's input before its
        norm: ``LfmOverlay.logits_of``)."""
        return self._px_rows(i, n, np.float32)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def attention_seconds(self, L: int, p0: int, S: float) -> float:
        """Estimated time of layer L's flash attention over a chunk of S sub-chunks at p0 beyond its layer's weights'
        term: a full layer's passes by the cache records they stream (from position 0 to their queries' group)."""
        if self.ov.kind(L) != "full":
            return 0.0
        at, s = self.attn[("full", round(S * SUB))], self.shape
        if at.nc:  # (every pass the chunk's groups: a prompt that fills it)
            return RECORD_S * at.n_pass * at.T // s.pg * s.rg
        q, r = divmod(at.n_pass, at.m)
        recs = (at.n_pass * (p0 // s.pg + 1) + at.m * q * (q - 1) // 2 + q * r) * s.rg
        return RECORD_S * recs

    def layer_seconds(self, L: int, p0: int, S: float = 1) -> float:
        return self._layer_static(L, S) + self.attention_seconds(L, p0, S)

    def layer_configs(self, L: int) -> int:
        """(its GEMM objects, the operator (conv, or prep and attention), the norms)"""
        gemms = len({id(g.op) for name in self._phases(L) for g in self.plans[(L, name)].gemms})
        return gemms + (1 if self.ov.kind(L) == "conv" else 2) + 2
