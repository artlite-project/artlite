"""The prefill of a Flint model: a prompt chunk of up to ``MAX_T`` tokens through every layer on the decode overlay's
buffers (``overlay.FlintOverlay``: the weight image, the KV caches), so that the decode continues from the caches the
chunk leaves; the prompt's last token's logits from the LM head alone on the chunk's last output row
(``ModelOverlay.logits_of``).

A chunk of T tokens (``SIZES``: ``SMALL``, multiples of 256 up to 2048, then multiples of 2048 up to 8192) runs
layer-major in sub-chunks of ``SUB`` tokens (a ``SMALL`` size: one of its own): each operator of a layer runs over the
sub-chunks back to back on buffer slices a sub-chunk apart (a ConfigureOp per layer and operator), the attention preps in
blocks of ``PREP_SIZES``, the flash attention once over a chunk of up to ``ATTN_T`` tokens (its 64 passes of 32 queries),
a larger chunk's in parts of ``ATTN_T`` (``AttnOperator(part=)``: each at its own position, its history from the cache,
which the preps filled with the whole chunk's records first). A chunk streams the experts' weights once: from 2048
tokens on the expert GEMMs are weight-bound, so a larger chunk takes them for more tokens. The residual stream x is bf16 (PX,
``prefill.norm.PrefillNormOperator``; the MoE block gathers its rows). Per layer L (the decode's order):

- qkv GEMM (``prefill.gemm``) of xn -> G; the attention prep of the layer's kind (``prefill.attn``: q / k RMS-normed with
  their weights per dim, rotated by the kind's cos / sin table: a sliding-window layer's NeoX RoPE on all 64 pairs, a
  full layer's none (cos 1, sin 0); Q units into QS, the chunk's K / V records into the layer's cache, a
  sliding-window cache after its guard), the flash attention over the chunk (two columns per KV head on one broadcast
  cache stream, 4 query heads each; the layer's learned sinks in SK, added to each row's sum at a pass's end; a
  sliding-window pass streams its window of 513 positions) -> A; o GEMM -> Y;
- a dense layer (0, 1): norm ``add`` (x += Y, xn = norm(x) ffn_norm), gate_up GEMMs with the SiLU epilogue -> HP, down
  GEMMs in K parts -> F, norm ``add`` (x += F, xn = norm(x) of the next layer's attn_norm);
- an MoE layer: the norm with lo panels (x += Y, PXM = norm(x) ffn_norm as hi + lo panels), the MoE block
  (``moe.prefill.MoEPrefill``: the router, the shared expert ungated, the host's routing between pieces (the top 6 by
  sigmoid, renormalized), the gathered norm, the experts' GEMMs, the combine -> DL), norm ``add`` (x += DL, xn = norm(x)
  of the next layer's attn_norm).

The first layer's input norm (``in``) starts the chunk. The control code is assembled on the host from one library ELF
and runs in pieces of whole layers (``prefill.chunks.ChunkPrefill``), an MoE layer in three (the host routes while the
shared expert runs, writes the experts' control streams while the gathered norm runs).

Chunks start at a cache group (64 positions); the padding tokens after the valid ones write cache records that later
tokens overwrite. The attention cores' code pages are reloaded before the next decode step (the chunk's configurations
replace them: ``OverlayDevice.stale``).
"""

import functools
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.dflash.overlay import DFlashOverlay
from artlite.moe.prefill import MoEPrefill
from artlite.operators.design import COLS
from artlite.operators.util import arg_bytes
from artlite.prefill import attn as PA
from artlite.prefill import gemm as PG
from artlite.prefill.chunks import PIECE_BUDGET_S, ChunkPrefill
from artlite.prefill.dflash import DFlashPrefill
from artlite.prefill.drafter import DrafterPrefill
from artlite.prefill.norm import PrefillNormOperator

from .overlay import KINDS, KV_PAD, FlintOverlay

SUB = 256  # tokens per sub-chunk (the operators' T)
SMALL = (64, 128)  # chunk sizes of operators of their own (short prompts: fewer passes)
ATTN_T = 2048  # a flash attention item's tokens at most (its 64 passes of 32 queries)
SIZES = SMALL + tuple(range(SUB, ATTN_T + 1, SUB)) + (4096, 6144, 8192)  # chunk sizes (a flash attention design per size
# and kind up to ATTN_T; larger ones: its parts)
MAX_T = SIZES[-1]
PREP_SIZES = (1024, 512)  # tokens per attention prep item where a chunk is made of them
PHASES = {"dense": ("qkv", "o", "gate_up", "down"), "moe": ("qkv", "o")}  # (the GEMM phases: an MoE layer's FFN is
# ``moe``'s)
# the time model's constants (``layer_seconds``): a ConfigureOp, per sub-chunk a byte of a layer's weights, a cache record
# a flash attention pass streams; an MoE layer's experts per byte of their weights and of their rows
CFG_S, WEIGHT_S, RECORD_S = 0.111e-3, 3.69e-11, 0.346e-6
MOE_WEIGHT_S, MOE_ROW_S = 1 / 38e9, 1 / 38e9
# a drafter's records (``DrafterPrefill.seconds``): a GEMM item, a norm per hidden unit, the prep per q head (qwen35's fits)
GEMM_S, NORM_S, PREP_S = 0.056e-3, 43e-9, 17e-6


class FlintPrefill(ChunkPrefill):
    """The prefill of an overlay (its layers) in chunks of the sizes ``sizes`` (those within the overlay's caches);
    ``run`` runs one; ``dflash``: a DFlash2 drafter whose rows of the prompt a chunk writes (``prefill.dflash``). The
    attention's Q and P are single panels (``prefill.attn.Shape``'s ``qp``, ``pp``): their hi + lo changed the next-token
    NLL after the prompt by nothing measurable (six 1K prompts, two of 32K: within an MoE model's noise; Flint's q / k
    norm weights reach 7.5, Qwen3's ~100) and cost the full layers' attention 1.2x at 30K; K follows the caches'
    ``kv_k``."""

    SUB, SMALL, MAX_T, NAME, KV_PAD = SUB, SMALL, MAX_T, "flint_prefill", KV_PAD
    SHARED_KV_RANGE = True  # (the attention's P_FO: one range for every size)
    CFG_S, WEIGHT_S = CFG_S, WEIGHT_S

    def __init__(self, overlay: FlintOverlay, *, sizes: tuple[int, ...] = SIZES, dflash: DFlashOverlay | None = None):
        ov = overlay
        s = PA.Shape(ov.hd, ov.nh // 2, ov.kp, ov.T, 1, 1, per=2, sink=True)
        pq = 32 * s.blocks[0]  # (the flash attention's queries per pass: 64 passes a chunk at most)
        assert pq * 64 >= ATTN_T
        super().__init__(overlay, tuple(t for t in sizes if t % ov.pg == 0 and (t <= ATTN_T or t % ATTN_T == 0)))
        cfg, img = self.cfg, self.img
        self.layers = ov.layers
        self.kv_layers = list(self.layers)  # (every layer its own cache and ring)
        self.shape = s
        self.group = s.pg
        self.cs_rows = ov.max_positions + MAX_T
        self.pad = KV_PAD // s.pg * s.group  # (the attention's KV arguments past a layer's caches: the overlay's padding)
        self.prep_pad = SUB // s.pg * s.group  # (the preps': a sub-chunk's groups, as ``LfmPrefill``'s)
        self.kv_bytes = {cfg.kinds[L]: ov.kv_bytes(L) for L in self.layers}  # (per KV head, a guard included)
        self.kv0 = {"swa": ov.guard * s.group, "full": 0}
        self.win = {"swa": cfg.sliding_window, "full": 0}
        self.wrap = {"swa": ov.kinds[0].wrap, "full": 0}  # (a ring's chunks: within one turn of it, ``fits``)
        kinds = sorted(self.kv_bytes)
        moe = [L for L in self.layers if cfg.moe_layer(L)]
        self.moe = MoEPrefill(img, layers=moe, sizes=self.sizes, sub=SUB, eps=cfg.rms_norm_eps, gating="sigmoid") \
            if moe else None
        self._moe_marks: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
        self._starts: dict[int, list[int]] = {}
        self._gemms: dict[tuple, object] = {}  # (one object per distinct GEMM)
        self._k_parts: dict[tuple, int] = {}  # (per region shape and size: output projections in resident K parts)
        self.sz = {t: self._size_ops(t) for t in sorted({self._t(T) for T in self.sizes})}
        self.plans = self.sz[SUB].plans if SUB in self.sz else self.sz[max(self.sz)].plans
        self.drafters: list[DrafterPrefill] = [DFlashPrefill(dflash, self)] if dflash is not None else []
        self.attn = {(k, T): self._attn(k, T) for k in kinds for T in self.sizes if T <= ATTN_T}
        # (a larger chunk's parts: part 0 the ATTN_T design, the others with parameters of their own)
        self.parts = {(k, j): self.attn[(k, ATTN_T)] if j == 0 and (k, ATTN_T) in self.attn else self._attn(k, ATTN_T, j)
                      for k in kinds for j in range(self.sizes[-1] // ATTN_T) if self.sizes[-1] > ATTN_T}
        self._rows()
        self._library()

    # ------------------------------------------------------------------
    # The operators
    # ------------------------------------------------------------------

    def _attn(self, kind: str, T: int, part: int = 0) -> PA.AttnOperator:
        """The flash attention of a kind over T tokens (``part``: a larger chunk's)."""
        return PA.AttnOperator(T=T, shape=self.shape, kv_bytes=self.kv_bytes[kind], kv_pad=self.pad, sub=self._pt(T),
                               swa=self.win[kind], kv0=self.kv0[kind], part=part, wrap=self.wrap[kind])

    def _attns(self, kind: str, T: int) -> list[PA.AttnOperator]:
        """The flash attention items of a chunk of size T (one, or its parts of ATTN_T tokens)."""
        return [self.attn[(kind, T)]] if T <= ATTN_T else [self.parts[(kind, j)] for j in range(T // ATTN_T)]

    @staticmethod
    def _t(T: int) -> int:
        """The sub-chunk of a chunk of size T (a ``SMALL`` size: itself)."""
        return min(T, SUB)

    @staticmethod
    def _pt(T: int) -> int:
        """The prep's block of a chunk of size T: the largest of PREP_SIZES the chunk is made of, else a sub-chunk."""
        return next((n for n in PREP_SIZES if T % n == 0), min(T, SUB))

    def _phases(self, L: int) -> tuple[str, ...]:
        """Layer L's GEMM phases (an MoE layer's FFN: ``moe``)."""
        return PHASES["moe" if self.cfg.moe_layer(L) else "dense"]

    def _size_ops(self, t: int) -> SimpleNamespace:
        """The operators of sub-chunks of t tokens: the GEMM plans (qkv at its layout's pitch, the output projections at
        ``d_pitch``, in resident K parts where those cost less: ``prefill.gemm.resident_k_part``), the norms, the
        attention preps (per kind, qkv layout and block size)."""
        cfg, img, H = self.cfg, self.img, self.H
        dp, qp = PG.d_pitch(img, t), PG.phase_pitch(img, "qkv", t)

        def plan(L: int, name: str) -> PG.PhasePlan:
            region = img.region(L, name)
            pitch = dp if name in PG.OUT else qp if name == "qkv" else 0
            key = (name, region.K, region.kc_group, tuple(b.formats for b in region.blocks), t)
            if key not in self._k_parts:
                self._k_parts[key] = PG.resident_k_part(img, region, t, pitch=pitch)
            return PG.plan_phase(img, region, t, pitch=pitch, ops=self._gemms, k_part=self._k_parts[key])

        plans = {(L, n): plan(L, n) for L in self.layers for n in self._phases(L)}
        assert all(plans[(L, "o")].parts == 1 for L in self.layers), "o in one part (the MoE block's norm adds one)"
        parted = [g.op for p in plans.values() if p.parts > 1 for g in p.gemms]
        f_part = max(arg_bytes(op, 2) for op in parted) if parted else 0
        eps, cols = cfg.rms_norm_eps, 4 if t < SUB else COLS  # (a SMALL size's norms on half the columns)
        kh = -(-H // img.kc) * img.kc  # (the norms' panel rows: the hidden size padded to whole K chunks)
        out = kh if kh != H else 0
        d = dp if dp != H // COLS else 0
        preps = {}
        for L in self.layers:
            k, lay = cfg.kinds[L], plans[(L, "qkv")].layout
            for pt in {self._pt(T) for T in self.sizes if self._t(T) == t}:
                if (k, lay.key, pt) not in preps:
                    preps[(k, lay.key, pt)] = PA.PrepOperator(
                        T=pt, shape=self.shape, layout=lay, kv_bytes=self.kv_bytes[k], cs_positions=self.cs_rows,
                        kv_pad=self.prep_pad, norm=True, eps=eps, kv0=self.kv0[k], wrap=self.wrap[k])
        return SimpleNamespace(
            plans=plans, d_pitch=dp, f_part=f_part, rb_y=2 * COLS * dp, preps=preps,
            pn0=PrefillNormOperator(T=t, add=False, E=H, eps=eps, out_pitch=out, cols=cols),
            pn=PrefillNormOperator(T=t, E=H, eps=eps, d_pitch=d, out_pitch=out, cols=cols),
            # (an MoE layer's delta: rows of H)
            pnm=PrefillNormOperator(T=t, E=H, eps=eps, out_pitch=out, cols=cols) if d else None,
            pnd={P: PrefillNormOperator(T=t, E=H, eps=eps, d_pitch=d, deltas=P, d_stride=f_part // 2, out_pitch=out,
                                        cols=cols)
                 for P in sorted({p.parts for p in plans.values() if p.parts > 1})})

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers; the norms' weights (NP: H bf16 each), the q / k norms' weights (NW),
        the sinks (SK)."""
        cfg, H, s = self.cfg, self.H, self.shape
        kh = -(-H // self.img.kc) * self.img.kc
        pitch = max(PG.phase_pitch(self.img, "qkv", t) for t in self.sz)
        self.rb = {"PX": 2 * H, "PXN": 9 * kh // 8, "G": 2 * COLS * pitch, "QS": COLS * s.qh * s.qunit // 8,
                   "A": 9 * COLS * s.qh * s.hd // 8, "Y": 2 * COLS * max(o.d_pitch for o in self.sz.values())}
        self.rb["F"] = self.rb["Y"]
        dense = [L for L in self.layers if not cfg.moe_layer(L)]
        if dense:  # (down's K: the dense FFN padded to whole K chunks)
            self.rb["HP"] = 9 * self.img.region(dense[0], "down").K // 8
        if self.moe:
            self.rb |= self.moe.rows()
        for d in self.drafters:
            self.rb |= d.rows()
        self.np_off = {"in": 0}
        k = 1
        for L in self.layers:
            for n in ("o", "d"):
                self.np_off[(L, n)] = k * 2 * H
                k += 1
        self.nw = PA.PrepOperator.nw_bytes(s.hd)
        self.sk = 2 * next(iter((self.attn | self.parts).values())).n_sk  # (bytes of a layer's SK argument)

    def _np(self, key: object) -> str:
        o = self.np_off[key]
        return f"NP[{o}:{o + 2 * self.H}]"

    def _runlist(self, S: int) -> list[tuple]:
        return self._runlist_of(S * SUB)

    def _layer_starts(self, T: int) -> list[int]:
        return self._starts[T]

    def _runlist_of(self, T: int, *, packed: bool = False, tree: bool = False) -> list[tuple]:
        """(sub-chunks of ``_t`` tokens: a ``SMALL`` size one of its own operators; the layers' first items in
        ``_starts``, the MoE layers' piece marks in ``_moe_marks``)"""
        assert not packed and not tree, "Flint: no packed chunks"
        ov, cfg, s, sl = self.ov, self.cfg, self.shape, self._sl
        t = self._t(T)
        S, o, pt = T // t, self.sz[t], self._pt(T)

        def tok(buf: str, j: int, op: object, *, i: int, row: int = 0, extra: int = 0) -> str:
            return sl(buf, j * t * (row or self.rb[buf]) + extra, op, i=i)

        ry, fp = o.rb_y, o.f_part
        rl: list[tuple] = [(o.pn0, tok("PX", j, o.pn0, i=0), self._np("in"), tok("PXN", j, o.pn0, i=2))
                           for j in range(S)]
        for d in self.drafters:
            rl += d.start_items(T, tok)
        starts, marks = [], []
        for i, L in enumerate(self.layers):
            starts.append(len(rl))
            for d in self.drafters:  # (PX: the layer's input)
                rl += d.tap_items(L, T, tok)
            kind = cfg.kinds[L]
            p_in = o.plans[(L, "qkv")]
            rg = 2 * COLS * p_in.layout.pitch
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("G", j, g.op, i=2, row=rg))
                   for g in p_in.gemms for j in range(S)]
            prep = o.preps[(kind, p_in.layout.key, pt)]
            kv0, kvb = ov.kv_off[L], arg_bytes(prep, 3)
            gb = pt // s.pg * s.group  # (a block's cache groups)
            cs = "CSS" if kind == "swa" else "CSF"
            nw = f"NW[{i * self.nw}:{(i + 1) * self.nw}]"
            rl += [(prep, tok("G", 0, prep, i=0, row=rg, extra=b * pt * rg), sl(cs, b * pt * 4 * s.hd, prep, i=1),
                    tok("QS", 0, prep, i=2, extra=b * pt * self.rb["QS"]), f"KV[{kv0 + b * gb}:{kv0 + b * gb + kvb}]", nw)
                   for b in range(T // pt)]
            rl += [(at, sl("QS", j * at.T * self.rb["QS"], at, i=0), f"KV[{kv0}:{kv0 + arg_bytes(at, 1)}]",
                    sl("A", j * at.T * self.rb["A"], at, i=2), f"SK[{i * self.sk}:{(i + 1) * self.sk}]")
                   for j, at in enumerate(self._attns(kind, T))]
            (g,) = o.plans[(L, "o")].gemms
            rl += [(g.op, tok("A", j, g.op, i=0), self._w(g), tok("Y", j, g.op, i=2, row=ry)) for j in range(S)]
            nxt = self._np((L, "d"))
            if cfg.moe_layer(L):  # (the norm with lo panels, the MoE block, the residual norm adding its delta)
                ln = self.moe.ln[t]
                rl += [(ln, tok("PX", j, ln, i=0), tok("Y", j, ln, i=1, row=ry), self._np((L, "o")),
                        tok("PX", j, ln, i=3), tok("PXM", j, ln, i=4)) for j in range(S)]
                items, at = self.moe.items(L, T, tok=lambda b, j, op, i, row: tok(b, j, op, i=i, row=row),
                                           sl=lambda b, off, op, i: sl(b, off, op, i=i), w=self._w,
                                           norm=self._np((L, "o")))
                marks.append((L, tuple(len(rl) + k for k in at)))
                rl += items
                pm = o.pnm or o.pn
                rl += [(pm, tok("PX", j, pm, i=0), tok("DL", j, pm, i=1), nxt, tok("PX", j, pm, i=3),
                        tok("PXN", j, pm, i=4)) for j in range(S)]
                continue
            rl += [(o.pn, tok("PX", j, o.pn, i=0), tok("Y", j, o.pn, i=1, row=ry), self._np((L, "o")),
                    tok("PX", j, o.pn, i=3), tok("PXN", j, o.pn, i=4)) for j in range(S)]
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("HP", j, g.op, i=2))
                   for g in o.plans[(L, "gate_up")].gemms for j in range(S)]
            down = o.plans[(L, "down")]
            if down.parts == 1:
                (g,) = down.gemms
                rl += [(g.op, tok("HP", j, g.op, i=0), self._w(g), tok("F", j, g.op, i=2, row=ry)) for j in range(S)]
                rl += [(o.pn, tok("PX", j, o.pn, i=0), tok("F", j, o.pn, i=1, row=ry), nxt, tok("PX", j, o.pn, i=3),
                        tok("PXN", j, o.pn, i=4)) for j in range(S)]
            else:  # (sub-chunk j's part k at (P j + k) f_part)
                P, pn = down.parts, o.pnd[down.parts]
                rl += [(g.op, tok("HP", j, g.op, i=0, extra=2 * g.a_off), self._w(g),
                        sl("F", (P * j + g.part) * fp, g.op, i=2)) for g in down.gemms for j in range(S)]
                rl += [(pn, tok("PX", j, pn, i=0), sl("F", P * j * fp, pn, i=1), nxt, tok("PX", j, pn, i=3),
                        tok("PXN", j, pn, i=4)) for j in range(S)]
        for d in self.drafters:
            rl += d.end_items(T, tok, lambda b, off, op, i: sl(b, off, op, i=i))
        self._starts[T] = [*starts, len(rl)]
        self._moe_marks[T] = marks
        return rl

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        for name, r in self.rb.items():  # (the GEMMs' padding rows past the last sub-chunk)
            if name in own:
                own[name] = max(own[name], (self.sizes[-1] + 64) * r)

    def _external(self) -> set[str]:
        return set().union(*(d.external() for d in self.drafters))

    def _buffers(self) -> dict[str, object]:
        return super()._buffers() | {k: v for d in self.drafters for k, v in d.buffers().items()}

    def _bind(self) -> None:
        """The norms' weights (NP), the q / k norms' weights (NW), the sinks (SK), the cos / sin tables (CSS, CSF); the
        drafters' static buffers."""
        ov, dev, H = self.ov, self.ov.dev, self.H

        def vec(name: str) -> np.ndarray:
            return np.asarray(ov.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

        npb = dev.view("NP").view(np.uint16)

        def put(key: object, name: str) -> None:
            o = self.np_off[key] // 2
            npb[o:o + H] = vec(name).astype(bfloat16).view(np.uint16)

        put("in", f"blk.{self.layers[0]}.attn_norm.weight")
        for i, L in enumerate(self.layers):
            put((L, "o"), f"blk.{L}.ffn_norm.weight")
            put((L, "d"), f"blk.{self.layers[i + 1]}.attn_norm.weight" if i + 1 < len(self.layers)
                else "output_norm.weight")
        dev.flush("NP")
        nw, sk = dev.view("NW"), dev.view("SK")
        at = next(iter((self.attn | self.parts).values()))
        for i, L in enumerate(self.layers):
            w = np.concatenate([vec(f"blk.{L}.attn_{n}_norm.weight") for n in "qk"]).view(np.uint8)
            nw[i * self.nw:i * self.nw + w.size] = w
            sk[i * self.sk:(i + 1) * self.sk] = at.sinks(vec(f"blk.{L}.attn_sinks.weight"))
        dev.flush("NW")
        dev.flush("SK")
        for kind, name in zip(KINDS, ("CSS", "CSF"), strict=True):
            if name not in self.own:
                continue
            inv = np.zeros(self.shape.hd // 2)
            f = self.cfg.rope_inv_freq(kind)
            inv[:f.size] = f  # (the pairs past rot: cos 1, sin 0)
            cs = PA.cs_table(inv, self.cs_rows).view(np.uint8).reshape(-1)
            dev.view(name)[:cs.size] = cs
            dev.flush(name)
        for d in self.drafters:
            d.static(dev, ov)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def params(self, T: int, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0."""
        out = self.moe.params(T) if self.moe else {}
        for d in self.drafters:
            out |= d.params(T, p0)
        for (k, _, pt), prep in self.sz[self._t(T)].preps.items():
            if pt == self._pt(T):
                out |= prep.params(p0)
                for j, at in enumerate(self._attns(k, T)):
                    out |= at.params(p0 + j * at.T)
        return out

    def fits(self, T: int, p0: int) -> bool:
        """(and within one turn of a sliding-window ring: its window's history before the chunk is the ring's or its
        guard's, ``FlintOverlay.mirror``)"""
        w = self.wrap["swa"]
        return super().fits(T, p0) and (not w or (p0 // self.group) % w + T // self.group <= w)

    def run(self, x: np.ndarray, p0: int, T: int = 0) -> None:
        """A chunk of size T (default: the smallest that holds x): x (n, hidden) the embeddings of positions p0 .. p0 +
        n - 1 (the caches hold the positions before). Leaves the layers' K / V of the positions in the caches and the
        last layer's output in PX (``hidden``)."""
        n, dev = x.shape[0], self.ov.dev
        T = self._size(n, p0, T)
        self._stage_px(x, T, bfloat16)
        self.ov.mirror(p0)
        for d in self.drafters:
            d.before(dev, p0, T)
        hooks = {}
        for L, (_, i_e, i_g) in self._moe_marks.get(T, ()):
            hooks[i_e] = functools.partial(self.moe.host_route, L, n=n, T=T, dev=dev, lib=self.lib)
            hooks[i_g] = functools.partial(self.moe.host_streams, L, n=n, T=T, dev=dev, lib=self.lib)
        self._launch(T, p0, self.params(T, p0), hooks or None)
        dev.stale = True  # (the chunk's configurations replaced the attention cores' code pages)
        for d in self.drafters:
            d.after(dev, p0, n, T)

    def _splits(self, T: int, p0: int, budget: float = PIECE_BUDGET_S) -> list[int]:
        """(an MoE layer: pieces from its shared expert and from its experts on: ``moe``)"""
        out = super()._splits(T, p0, budget)
        return sorted({*out, *(i for _, at in self._moe_marks.get(T, ()) for i in at)})

    def hidden(self, i: int, n: int = 1) -> np.ndarray:
        """Rows i .. i + n - 1 (n, hidden) of the last chunk's output residual (bf16 values, fp32; the LM head's input
        before its norm)."""
        return self._px_rows(i, n, bfloat16)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def attention_seconds(self, L: int, p0: int, S: float) -> float:
        """Estimated time of layer L's flash attention over a chunk of S sub-chunks at p0: per pass the cache records it
        streams (a full layer's: from position 0 to its queries' group; a sliding-window layer's: its window's groups
        and its own), RECORD_S each."""
        T = round(S * SUB)
        kind, s, recs = self.cfg.kinds[L], self.shape, 0
        for j, at in enumerate(self._attns(kind, T)):
            p = p0 + j * at.T
            if kind == "swa":
                g = PA.mode.swa_groups(self.win["swa"], s.pg)
                recs += sum(min(g, (p + b * at.pq) // s.pg) + 1 for b in range(at.n_pass)) * s.rg
            else:
                q, r = divmod(at.n_pass, at.m)  # (sum over the passes b of b // m)
                recs += (at.n_pass * (p // s.pg + 1) + at.m * q * (q - 1) // 2 + q * r) * s.rg
        return RECORD_S * recs

    def layer_seconds(self, L: int, p0: int, S: float = 1) -> float:
        """(per sub-chunk, or the fraction T / SUB of one, the GEMMs by the bytes of their weights, WEIGHT_S each (the
        norms and the prep included); an MoE layer's router and shared expert likewise, its experts by
        ``MoEPrefill.layer_seconds``; the flash attention; CFG_S per ConfigureOp)"""
        s = self._layer_static(L, S) + self.attention_seconds(L, p0, S)
        if self.cfg.moe_layer(L):
            s += S * WEIGHT_S * sum(self.img.region(L, n).nbytes for n in ("moe_in", "sdown"))
            s += self.moe.layer_seconds(L, max(round(S * SUB), 1), weight_s=MOE_WEIGHT_S, row_s=MOE_ROW_S)
        if L == self.layers[-1]:  # (the drafters' records after the last layer)
            for d in self.drafters:
                s += d.seconds(S, weight_s=WEIGHT_S, gemm_s=GEMM_S, norm_s=NORM_S, prep_s=PREP_S, cfg_s=CFG_S)
        return s

    def layer_configs(self, L: int) -> int:
        """(its GEMM objects, the prep and attention, the norms; an MoE layer's block: 7)"""
        gemms = len({id(g.op) for name in self._phases(L) for g in self.plans[(L, name)].gemms})
        return gemms + 2 + 2 + (7 if self.cfg.moe_layer(L) else 0)


__all__ = ["SIZES", "FlintPrefill"]
