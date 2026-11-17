"""The prefill of a Spark-X model: a prompt chunk of up to ``MAX_T`` tokens through every layer on the decode overlay's
buffers (``overlay.SparkOverlay``: the weight image, the full layers' KV caches and the sliding-window layers' rings), so
that the decode continues from the caches the chunk leaves; the prompt's last token's logits from the LM head alone on
the chunk's last output row (``SparkOverlay.logits_of``).

A chunk of T tokens (``SIZES``: ``SMALL``, multiples of ``SUB`` up to 2048: the rings hold a window and a chunk,
``overlay.ring_groups``) runs layer-major in sub-chunks of ``SUB`` tokens (a ``SMALL`` size: one of its own): each
operator of a layer runs over the sub-chunks back to back on buffer slices a sub-chunk apart. The residual stream x is
fp32 (PX, ``prefill.norm.ResidualNormOperator``: N0 measured a bf16 one 66% worse on the 4B's chats), its panels in rows
of the GEMMs' K (the 4B's 2560 in 3072: ``out_pitch``). Per layer L:

- attn_in GEMMs of xn -> G (per column [q | k part | v part | gate rows], ``plan_gemm``); the attention prep of the
  layer's kind per sub-chunk (``prefill.attn256.AttnPrepOperator``: Q slots with the gates into QS; a full layer's
  records into its cache, a sliding-window layer's into the staging S after the window's history), the attention over
  the chunk (``PrefillAttnOperator``: a sliding-window layer's from S, the history copied in from its ring first and
  the chunk's records into the ring last) -> A; attn_out GEMMs -> Y;
- norm ``add``: x += Y, xn = norm(x) ffn_norm; gate_up GEMMs with GELU's erf form -> HP; down GEMMs in two K parts -> F;
  norm ``add``: x += F's parts, xn = norm(x) of the next layer's attn_norm (the last layer: output_norm, unused: the LM
  head norms its own input).

The first layer's input norm (``in``) starts the chunk. The control code is assembled on the host from one library ELF
and runs in pieces of whole layers (``prefill.chunks.ChunkPrefill``).
"""

import numpy as np

from artlite.operators.design import COLS
from artlite.operators.util import arg_bytes
from artlite.prefill import attn256 as AP
from artlite.prefill import gemm as PG
from artlite.prefill.chunks import ChunkPrefill
from artlite.prefill.norm import ResidualNormOperator, residual_params

from .overlay import CHUNK, SparkOverlay

DOWN_PARTS = 2  # (the down projection's K parts: each part's bf16 C tile sums half the k-tiles, the residual add sums
# the parts in fp32: the 4B's 160 k-tiles as 2 x 80; ``prefill.norm.ResidualNormOperator`` sums 1, 2 or 4)
SUB = 256  # tokens per sub-chunk (the operators' T)
SMALL = (64, 128)  # chunk sizes of operators of their own (short prompts)
SIZES = SMALL + tuple(range(SUB, CHUNK + 1, SUB))
MAX_T = SIZES[-1]
# the time model's constants (``layer_seconds``; to fit to measured chunks): a ConfigureOp, per sub-chunk a byte of a
# layer's weights, a cache record a full attention pass streams, a sliding-window pass
CFG_S, WEIGHT_S, RECORD_S, SWA_PASS_S = 0.1e-3, 3.75e-11, 0.29e-6, 80e-6


def plan_gemm(img: object, region: object, T: int, *, pitch: int = 0, ops: dict[tuple, object] | None = None,
              k_parts: dict[tuple, int] | None = None) -> PG.PhasePlan:
    """A GEMM phase of the prefill (``prefill.gemm.plan_phase``): gate / up with GELU's erf form; the down in
    ``DOWN_PARTS`` K parts; attn_out, whose A strips would be refetched per pass, in the parts whose strips stay in the
    memtile (``prefill.gemm.resident_k_part``). ``k_parts``: the choices per region shape, kept across calls."""
    if region.name == "down":  # (two parts: of whole chunk groups, the 4B's 5 + 5; else of chunks, the 1.7B's 3 + 4)
        g = region.kc_group
        k = -(-region.n_kc // DOWN_PARTS // g) * g
        return PG.plan_phase(img, region, T, pitch=pitch, ops=ops, act="gelu_erf", k_part=k)
    k = 0
    if region.name in PG.OUT:
        cache = {} if k_parts is None else k_parts
        key = (region.name, region.K, region.kc_group, tuple(b.fmt for b in region.blocks), T, pitch, k)
        if key not in cache:
            cache[key] = PG.resident_k_part(img, region, T, pitch=pitch, k_part=k)
        k = cache[key]
    return PG.plan_phase(img, region, T, pitch=pitch, ops=ops, act="gelu_erf", k_part=k)


class SparkPrefill(ChunkPrefill):
    """The prefill of an overlay (its layers) in chunks of the sizes ``sizes`` (those within the overlay's caches);
    ``run`` runs one."""

    SUB, SMALL, MAX_T, NAME = SUB, SMALL, MAX_T, "spark_prefill"
    KV_PAD = CHUNK  # (the overlay's padding past its last cache: a full layer's prep slices a sub-chunk's groups on)
    CFG_S, WEIGHT_S = CFG_S, WEIGHT_S

    def __init__(self, overlay: SparkOverlay, *, sizes: tuple[int, ...] = SIZES):
        ov = overlay
        super().__init__(overlay, sizes)
        self.layers = ov.layers
        self.kv_layers = list(ov.layers)
        cfg = self.cfg
        self.window = cfg.sliding_window
        self.wg = self.window // 32
        self.stage = self.wg + MAX_T // 32  # (the staging's groups per KV head: one layout for every size)
        self.kmax = max(r.K for r in self.img.regions if r.name == "attn_in")  # (the panels' K: the hidden padded)
        self._norm_ops: dict[tuple, ResidualNormOperator] = {}
        self._k_parts: dict[tuple, int] = {}
        self._gemms: dict[tuple, object] = {}
        self._preps: dict[tuple, AP.AttnPrepOperator] = {}
        self.plans = {t: {(L, n): self._plan(L, n, t) for L in self.layers for n in self._phases(L)}
                      for t in sorted({self._t(T) for T in self.sizes})}
        self.d_pitch = {t: PG.d_pitch(self.img, t) for t in self.plans}
        self.attn = {(k, T): AP.PrefillAttnOperator(
            T=T, cfg=cfg, spec=ov.ash.spec, window=self.window if k == "swa" else 0, ring=ov.nr if k == "swa" else 0,
            head_bytes=ov.nr * AP.GROUP if k == "swa" else ov.ash.head_bytes, stage=self.stage if k == "swa" else 0)
            for k in ("full", "swa") for T in self.sizes}
        self.cs_rows = ov.max_positions + MAX_T
        self._starts: dict[int, list[int]] = {}  # (per chunk size: its layers' first items)
        self._rows()
        self._library()

    def _t(self, T: int) -> int:
        """The sub-chunk of a chunk of size T (a ``SMALL`` size: itself)."""
        return min(T, SUB)

    def _phases(self, L: int) -> tuple[str, ...]:
        return ("attn_in", "attn_out", "gate_up", "down")

    def _plan(self, L: int, name: str, t: int) -> PG.PhasePlan:
        pitch = PG.d_pitch(self.img, t) if name in PG.OUT else PG.phase_pitch(self.img, name, t) \
            if name == "attn_in" else 0
        plan = plan_gemm(self.img, self.img.region(L, name), t, pitch=pitch, ops=self._gemms, k_parts=self._k_parts)
        assert plan.parts in (1, 2, 4), (L, name, plan.parts)
        return plan

    def _prep(self, L: int, t: int) -> AP.AttnPrepOperator:
        """Layer L's prep of a sub-chunk of t tokens (one object per layer kind and attn_in layout)."""
        ov, kind = self.ov, self.ov.layer_kind(L)
        lay = self.plans[t][(L, "attn_in")].layout
        key = (kind, lay.key, t)
        if key not in self._preps:
            swa = kind == "swa"
            self._preps[key] = AP.AttnPrepOperator(
                T=t, cfg=self.cfg, layout=lay, cs_positions=self.cs_rows, spec=ov.ash.spec, kind=kind,
                head_bytes=self.stage * AP.GROUP if swa else ov.ash.head_bytes, hist=self.wg if swa else 0)
        return self._preps[key]

    def _norm(self, t: int, mode: str, deltas: int, stride: int = 0) -> ResidualNormOperator:
        """The residual norm of a sub-chunk of t tokens (``in``, or ``add`` of ``deltas`` K parts ``stride`` bytes
        apart)."""
        dp = self.d_pitch[t]
        key = (t, mode, deltas, stride)
        if key not in self._norm_ops:
            self._norm_ops[key] = ResidualNormOperator(
                T=t, E=self.H, mode=mode, eps=self.cfg.rms_norm_eps, d_pitch=dp if mode != "in" else 0, deltas=deltas,
                d_stride=stride // 2 if deltas > 1 else 0, out_pitch=self.kmax)
        return self._norm_ops[key]

    @staticmethod
    def _c_slot(plan: PG.PhasePlan) -> int:
        """Bytes of an output projection's C slot per sub-chunk and K part: its GEMMs' C (their rows past the sub-chunk
        too: padded token passes), in whole 64-byte lines."""
        return -(-max(arg_bytes(g.op, 2) for g in plan.gemms) // 64) * 64

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers (the sub-chunk sizes' layouts: at most the largest); the norms'
        parameter objects ([8][H] bf16 each)."""
        H, s = self.H, self.ov.ash
        pitch = max(p.layout.pitch for o in self.plans.values() for (L, n), p in o.items() if n == "attn_in")
        ffn_k = max(r.K for r in self.img.regions if r.name == "down")
        self.rb = {"PX": 4 * H, "PXN": 9 * self.kmax // 8, "G": 2 * COLS * pitch, "QS": COLS * s.qc * AP.SLOT // 8,
                   "A": 9 * COLS * s.qc * AP.HD // 8, "HP": 9 * ffn_k // 8}  # (Y, F: C slots, ``_c_slot``)
        split = [r for r in self.img.regions if r.name == "gate_up" and r.blocks[0].fmt2]
        if split:  # (split gate / up blocks: their GEMMs' rows, ``prefill.gemm``)
            self.rb["GU"] = max(2 * PG.gu_columns(self.img, r) for r in split)
        self.np_off = {"in": 0}
        k = 1
        for L in self.layers:
            for n in ("o", "d"):
                self.np_off[(L, n)] = k * 16 * H
                k += 1

    def _np(self, key: object, op: object) -> str:
        o = self.np_off[key]
        return f"NP[{o}:{o + arg_bytes(op, 0)}]"

    def _runlist(self, S: int) -> list[tuple]:
        return self._runlist_of(S * SUB)

    def _runlist_of(self, T: int, *, packed: bool = False, tree: bool = False) -> list[tuple]:
        assert not packed and not tree, "Spark-X: no packed chunks"
        t = self._t(T)
        S, plans, ov, sl, rb = T // t, self.plans[t], self.ov, self._sl, self.rb

        def tok(buf: str, a: int, op: object, *, i: int, row: int = 0) -> str:  # (from token a on)
            return sl(buf, a * (row or rb[buf]), op, i=i)

        n_in = self._norm(t, "in", 1)
        rl: list[tuple] = [(n_in, self._np("in", n_in), tok("PX", j * t, n_in, i=1), tok("PXN", j * t, n_in, i=2))
                           for j in range(S)]
        self._starts[T] = starts = []
        for L in self.layers:
            starts.append(len(rl))
            kind = ov.layer_kind(L)
            p_in = plans[(L, "attn_in")]
            rg = 2 * COLS * p_in.layout.pitch
            rl += [(g.op, tok("PXN", j * t, g.op, i=0), self._w(g), tok("G", j * t, g.op, i=2, row=rg))
                   for g in p_in.gemms for j in range(S)]
            prep, attn = self._prep(L, t), self.attn[(kind, T)]
            kv0 = ov.kv_off[L]
            qs_g = COLS * prep.qc * prep.tg_slots * AP.SLOT  # (bytes of a token group's Q slots: QS pass-major)
            grp = t // 32 * AP.GROUP  # (a sub-chunk's cache groups)
            cs, crow = ("CSS", 2 * prep.cs_row) if kind == "swa" else ("CSF", 2 * prep.cs_row)
            if kind == "swa":  # (the prep's records into the staging, the attention's ring and staging)
                kvb = arg_bytes(prep, 3)
                rl += [(prep, tok("G", j * t, prep, i=0, row=rg), sl(cs, j * t * crow, prep, i=1),
                        sl("QS", j * t // 8 * qs_g, prep, i=2), f"S[{j * grp}:{j * grp + kvb}]") for j in range(S)]
                rl.append((attn, sl("QS", 0, attn, i=0), f"KV[{kv0}:{kv0 + arg_bytes(attn, 1)}]", sl("A", 0, attn, i=2),
                           sl("S", 0, attn, i=3)))
            else:
                hb = arg_bytes(prep, 3)
                rl += [(prep, tok("G", j * t, prep, i=0, row=rg), sl(cs, j * t * crow, prep, i=1),
                        sl("QS", j * t // 8 * qs_g, prep, i=2), f"KV[{kv0 + j * grp}:{kv0 + j * grp + hb}]")
                       for j in range(S)]
                rl.append((attn, sl("QS", 0, attn, i=0), f"KV[{kv0}:{kv0 + arg_bytes(attn, 1)}]", sl("A", 0, attn, i=2)))
            # (Y and F: per sub-chunk j and K part p a slot of the GEMMs' C at (parts j + p) cb; the norm takes the parts)
            for name, src, dst, key in (("attn_out", "A", "Y", "o"), ("down", "HP", "F", "d")):
                plan = plans[(L, name)]
                cb, k = self._c_slot(plan), plan.parts
                if name == "attn_out":
                    rl += [(g.op, tok("A", j * t, g.op, i=0), self._w(g), sl("Y", (k * j + g.part) * cb, g.op, i=2))
                           for g in plan.gemms for j in range(S)]
                else:
                    gu = plans[(L, "gate_up")]
                    rl += [(g.op, tok("PXN", j * t, g.op, i=0), self._w(g), tok(g.c or "HP", j * t, g.op, i=2))
                           for g in gu.gemms for j in range(S)]
                    if gu.act is not None:  # (a split block: its GEMMs' rows into the down's panels)
                        rl += [(gu.act, tok("GU", j * t, gu.act, i=0), tok("HP", j * t, gu.act, i=1)) for j in range(S)]
                    rl += [(g.op, sl("HP", j * t * rb["HP"] + 2 * g.a_off, g.op, i=0), self._w(g),
                            sl("F", (k * j + g.part) * cb, g.op, i=2)) for g in plan.gemms for j in range(S)]
                n = self._norm(t, "add", k, cb)
                rl += [(n, self._np((L, key), n), tok("PX", j * t, n, i=1), sl(dst, k * j * cb, n, i=2),
                        tok("PX", j * t, n, i=3), tok("PXN", j * t, n, i=4)) for j in range(S)]
        starts.append(len(rl))
        return rl

    def _layer_starts(self, T: int) -> list[int]:
        return self._starts[T]

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        for name, r in self.rb.items():  # (the GEMMs' padding rows past the last sub-chunk)
            own[name] = max(own.get(name, 0), (self.sizes[-1] + 64) * r)
        own["NP"] = max(own.get("NP", 0), max(self.np_off.values()) + 16 * self.H)
        own["S"] = max(own.get("S", 0), self.ov.ash.kvh * self.stage * AP.GROUP)

    def _bind(self) -> None:
        """The norms' parameter objects (NP), the cos / sin tables (CSF: a full layer's, CSS: a sliding-window
        layer's)."""
        ov, dev, H = self.ov, self.ov.dev, self.H
        npb = dev.view("NP").view(np.uint16)

        def vec(name: str) -> np.ndarray:
            return np.asarray(ov.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

        def put(key: object, w1: str) -> None:
            o = self.np_off[key] // 2
            v = residual_params(None, vec(w1), 1.0, H)
            npb[o:o + v.size] = v.reshape(-1)

        put("in", f"blk.{self.layers[0]}.attn_norm.weight")
        for i, L in enumerate(self.layers):
            put((L, "o"), f"blk.{L}.ffn_norm.weight")
            put((L, "d"), f"blk.{self.layers[i + 1]}.attn_norm.weight" if i + 1 < len(self.layers)
                else "output_norm.weight")
        dev.flush("NP")
        for kind, name in (("swa", "CSS"), ("full", "CSF")):
            cs = AP.cs_table(self.cfg, self.cs_rows, kind).view(np.uint8).reshape(-1)
            dev.view(name)[:cs.size] = cs
            dev.flush(name)
        dev.view("S")[:] = 0  # (a short history's groups before position 0: junk the attention masks; finite)
        dev.flush("S")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def params(self, T: int, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0."""
        t, out = self._t(T), {}
        for L in self.layers:
            kind = self.ov.layer_kind(L)
            out |= self._prep(L, t).params(p0) | self.attn[(kind, T)].params(p0)
        return out

    def run(self, x: np.ndarray, p0: int, T: int = 0) -> None:
        """A chunk of size T (default: the smallest that holds x): x (n, hidden) the embeddings of positions p0 .. p0 +
        n - 1 (the caches hold the positions before). Leaves the layers' K / V of the positions in the caches and
        rings."""
        n = x.shape[0]
        T = self._size(n, p0, T)
        self._stage_px(x, T, np.float32)
        self._launch(T, p0, self.params(T, p0))
        self.ov.dev.stale = True  # (the chunk's configurations replaced the overlay's code pages: reloaded before a step)

    def hidden(self, i: int, n: int = 1) -> np.ndarray:
        """Rows i .. i + n - 1 (n, hidden) fp32 of the last chunk's output residual (the LM head's input before its
        norm: ``SparkOverlay.logits_of``)."""
        return self._px_rows(i, n, np.float32)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def attention_seconds(self, L: int, p0: int, S: float) -> float:
        """Estimated time of layer L's attention over a chunk of S sub-chunks at p0 beyond its layer's weights' term: a
        full layer's passes by the cache records they stream, a sliding-window layer's by its passes."""
        at = self.attn[(self.ov.layer_kind(L), max(self.SMALL[0], round(S * SUB)))]
        if at.window:
            return SWA_PASS_S * at.T // (32 * at.qg)
        n = at.T // (32 * at.qg)
        groups = n * (p0 // 32 + at.qg) + at.qg * n * (n - 1) // 2
        return RECORD_S * groups * 8 * self.ov.ash.kvh

    def layer_seconds(self, L: int, p0: int, S: float = 1) -> float:
        return self._layer_static(L, S) + self.attention_seconds(L, p0, S)

    def layer_configs(self, L: int) -> int:
        """(its GEMM objects, the prep and the attention, the norms)"""
        t = max(self.plans)
        gemms = len({id(g.op) for name in self._phases(L) for g in self.plans[t][(L, name)].gemms})
        return gemms + 2 + 2
