"""The prefill of a Llama 3.x model: a prompt chunk of up to ``MAX_T`` tokens through every layer on the decode
overlay's buffers (``overlay.LlamaOverlay``: the weight image, the KV caches), so the decode continues from the caches
the chunk leaves; the prompt's last token's logits from the LM head alone (``ModelOverlay.logits_of``).

A chunk of T tokens (``SIZES``: ``SMALL``, multiples of 256) runs layer-major in sub-chunks of ``SUB`` tokens, of
``BIG`` where T is a multiple of it (qkv in its ``SUB``-token items): each operator of a layer runs over the sub-chunks
back to back on buffer slices a sub-chunk apart (consecutive runs of one operator share its configuration: a ConfigureOp
per layer and operator), the attention prep in blocks of ``PREP_SIZES``, the flash attention once over the whole chunk.
The residual stream x is bf16 (PX, ``prefill.norm.PrefillNormOperator``: for Llama the same KL as an fp32 residual at
half the norms' bytes), its norms over blocks of up to 2048 tokens (``NORM_SIZES``). Per layer L (the decode's order):

- qkv GEMMs (``prefill.gemm``) of xn (the norm's bfp16 panels) -> G; the attention prep (``prefill.attn``: Q units
  into QS, the chunk's K / V records into the layer's cache: slices of the cache and the cos / sin table shifted by
  the sub-chunk's positions), the flash attention over the chunk -> A (the O GEMM's panels);
- O GEMM -> Y; norm ``add``: x += Y, xn = norm(x) ffn_norm;
- gate_up GEMMs with the SiLU epilogue -> HP; down GEMMs -> F; norm ``add``: x += F, xn = norm(x) of the next layer's
  attn_norm (the last layer: x; the LM head reads its last valid row through the overlay).

The first layer's input norm (``in``) starts the chunk. The control code is assembled on the host from one library ELF
and runs in pieces of whole layers (``prefill.chunks.ChunkPrefill``). A GEMM writes whole MAC tiles (GemmQd: 288 rows
for 256 tokens): its padding rows land on the next sub-chunk's first rows, which the next sub-chunk's GEMM writes
afterwards (the last's in the buffers' padding).

Chunks start at a cache group (64 positions at head_dim 128, 128 at 64: the attention's cache reads and writes are whole
groups); the padding tokens after the valid ones write cache records that later tokens overwrite.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from ..design import COLS
from ..prefill import attn as PA
from ..prefill import gemm as PG
from ..prefill.chunks import ChunkPrefill
from ..prefill.norm import PrefillNormOperator
from ..util import arg_bytes
from .overlay import KV_PAD, LlamaOverlay

SUB = 256  # tokens per sub-chunk (the operators' T)
# the time model's constants (``layer_seconds``; fitted to the chunk times of 1B Q4_0 and 3B Q4_K_M, T = 64 .. 4096 at
# p0 = 0 and 4096: within ~5%, a chunk of 64 ~20% below): a ConfigureOp, per sub-chunk a byte of a layer's weights, a
# cache record a flash attention pass streams
CFG_S, WEIGHT_S, RECORD_S = 0.143e-3, 3.69e-11, 0.346e-6
BIG = 1024  # the sub-chunk of the chunks made of them (the GEMMs but qkv once per 1024 tokens: fewer items and fills;
# qkv's items stay SUB: a 1024-token GemmQd is slower, a GemmQ no faster)
PREP_SIZES = (1024, 512)  # tokens per attention prep item where a chunk is made of them
NORM_SIZES = (2048, 1024, 512)  # tokens per norm item where a chunk holds them (its fill and drain once per item)
SMALL = (64, 128)  # chunk sizes of operators of their own (short prompts: fewer passes)
# chunk sizes of whole cache groups (a flash attention design each); past 2048 in whole BIG sub-chunks (long prompts:
# half the ConfigureOps per token of 2048-token chunks)
SIZES = SMALL + tuple(range(SUB, 2049, SUB)) + (3072, 4096)
MAX_T = SIZES[-1]
PHASES = ("qkv", "o", "gate_up", "down")
NORMS = ("o", "d")  # a layer's residual adds (after o, after down)
DOWN_CHUNKS = 4  # K chunks per down part (1B / 3B: 2 parts of 4; each part's A strip stays in the memtile, each sums 64
# k-tiles in its bf16 C tile; the norm adds the parts in fp32)


def _key(lay: PG.Layout) -> tuple:
    return lay.pitch, tuple(sorted(lay.off.items()))


def plan_gemm(img: object, region: object, T: int, *, pitch: int = 0, ops: dict[str, object] | None = None,
              down_chunks: int = DOWN_CHUNKS, k_parts: dict[tuple, int] | None = None) -> PG.PhasePlan:
    """A GEMM phase of the prefill (``prefill.gemm.plan_phase``): gate / up with the SiLU epilogue; down in K parts of
    ``down_chunks`` chunks where its chunk groups divide into them (1B, 3B: 2 x 4; its C tile's bf16 sums over fewer
    k-tiles: 3B KL(same) 0.00071 -> 0.00031 on a chat prompt); an output projection whose A strips would still be
    refetched per pass in the parts whose strips stay in the memtile, where they cost less
    (``prefill.gemm.resident_k_part``: 8B's Q4_K down in 4, 3B's Q6_K down at 1024 tokens in 4). ``k_parts``: the
    choices per region shape, kept across calls."""
    k = down_chunks if region.name == "down" and region.kc_group % down_chunks == 0 and \
        region.n_kc // down_chunks in (2, 4) else 0
    if region.name in PG.OUT:
        cache = {} if k_parts is None else k_parts
        key = (region.name, region.K, region.kc_group, tuple(b.fmt for b in region.blocks), T, pitch, k)
        if key not in cache:
            cache[key] = PG.resident_k_part(img, region, T, pitch=pitch, k_part=k)
        k = cache[key]
    return PG.plan_phase(img, region, T, pitch=pitch, ops=ops, k_part=k)


@dataclass
class _Ops:
    """The operators of a sub-chunk size: the GEMM plans, the norms, the attention preps (one per qkv layout)."""

    plans: dict[tuple[int, str], PG.PhasePlan]
    qt: int  # (qkv's tokens per item)
    pitch: int  # (the qkv layouts' column pitch)
    d_pitch: int  # (the output projections')
    parts: int  # (down's K parts: the most of any layer, F's slots per norm block)
    f_part: int  # (bytes of a down part's C)
    n_in: PrefillNormOperator
    # (per norm block size: after o, after down per K part count)
    norms: dict[int, tuple[PrefillNormOperator, dict[int, PrefillNormOperator]]]
    preps: dict[tuple, PA.PrepOperator]


class LlamaPrefill(ChunkPrefill):
    """The prefill of an overlay (its layers) in chunks of the sizes ``sizes`` (those within the overlay's caches and
    of whole cache groups); ``run`` runs one."""

    SUB, SMALL, MAX_T, NAME = SUB, SMALL, MAX_T, "llama_prefill"

    def __init__(self, overlay: LlamaOverlay, *, sizes: tuple[int, ...] = SIZES):
        ov = overlay
        s = PA.Shape(ov.hd, ov.nh, ov.kp, ov.T)
        pq = 32 * s.blocks[0]  # (the flash attention's queries per pass: at most 64 passes a chunk, 8B's 2048 tokens)
        super().__init__(overlay, tuple(t for t in sizes if t % ov.pg == 0 and t // pq <= 64))
        self.layers = ov.layers
        self.kv_layers = [0]  # (the overlay's rings: one state for every layer, key 0)
        self.shape = s
        self.group = s.pg
        self.cs_rows = ov.max_positions + MAX_T
        self.pad = KV_PAD // s.pg * s.group  # (the KV arguments past a layer's caches: the overlay's padding)
        self._static_s: dict[tuple[int, float], float] = {}
        self._norm_ops: dict[tuple, PrefillNormOperator] = {}  # (one object per distinct norm)
        self._k_parts: dict[tuple, int] = {}  # (the output projections' K parts per region shape: plan_gemm)
        self._starts: dict[int, list[int]] = {}  # (per chunk size: its layers' first items, ``_layer_starts``)
        gemms: dict[str, object] = {}  # (one object per distinct GEMM)
        self.ops = {t: self._ops(t, gemms) for t in sorted({self._t(T) for T in self.sizes})}
        self.plans = self.ops[SUB].plans if SUB in self.ops else self.ops[max(self.ops)].plans
        self.attn = {T: PA.AttnOperator(T=T, shape=s, kv_bytes=ov.kv_bytes, kv_pad=self.pad, sub=self._pt(T))
                     for T in self.sizes}
        self._rows()
        self._library()

    @staticmethod
    def _t(T: int) -> int:
        """The sub-chunk of a chunk of size T: BIG where the chunk is made of them, else SUB (a short chunk: itself)."""
        return BIG if T % BIG == 0 else min(T, SUB)

    def _ops(self, t: int, gemms: dict[str, object]) -> _Ops:
        """The operators of sub-chunks of t tokens (qkv's of qt: SUB in BIG ones)."""
        ov, cfg, img, H = self.ov, self.cfg, self.img, self.H
        qt = SUB if t == BIG else t
        pitch, dp = PG.phase_pitch(img, "qkv", qt), PG.d_pitch(img, t)
        plans = {(L, name): plan_gemm(img, img.region(L, name), qt if name == "qkv" else t, ops=gemms,
                                      pitch=dp if name in PG.OUT else pitch if name == "qkv" else 0,
                                      k_parts=self._k_parts)
                 for L in self.layers for name in PHASES}
        down = [plans[(L, "down")] for L in self.layers]
        parts = max(d.parts for d in down)
        assert all(p.parts == 1 for (_, name), p in plans.items() if name != "down")
        f_part = max(arg_bytes(g.op, 2) for d in down for g in d.gemms)
        preps: dict[tuple, PA.PrepOperator] = {}  # (per qkv layout and prep block size)
        for L in self.layers:
            lay = plans[(L, "qkv")].layout
            for pt in {self._pt(T) for T in self.sizes if self._t(T) == t}:
                if (_key(lay), pt) not in preps:
                    preps[(_key(lay), pt)] = PA.PrepOperator(T=pt, shape=self.shape, layout=lay, kv_bytes=ov.kv_bytes,
                                                             cs_positions=self.cs_rows, kv_pad=self.pad)
        o = _Ops(plans, qt, pitch, dp, parts, f_part,
                 PrefillNormOperator(T=t, add=False, E=H, eps=cfg.rms_norm_eps, cols=self._norm_cols(t)), {}, preps)
        for n in {t, *(m for m in NORM_SIZES if t <= m <= max(T for T in self.sizes if self._t(T) == t))}:
            o.norms[n] = self._norms(n, o)
        return o

    @staticmethod
    def _pt(T: int) -> int:
        """The prep's block of a chunk of size T: the largest of PREP_SIZES the chunk is made of, else a sub-chunk."""
        return next((n for n in PREP_SIZES if T % n == 0), min(T, SUB))

    def _norms(self, n: int, o: _Ops) -> tuple[PrefillNormOperator, dict[int, PrefillNormOperator]]:
        """The norms of a block of n tokens (after o, after down per K part count: its parts n tokens of F apart)."""
        eps, H, dp = self.cfg.rms_norm_eps, self.H, o.d_pitch
        cols = self._norm_cols(n)

        def norm(deltas: int) -> PrefillNormOperator:
            key = (n, dp, deltas, cols)
            if key not in self._norm_ops:
                self._norm_ops[key] = PrefillNormOperator(T=n, E=H, eps=eps, d_pitch=dp, deltas=deltas, cols=cols,
                                                          d_stride=n * COLS * dp if deltas > 1 else 0)
            return self._norm_ops[key]

        return norm(1), {P: norm(P) for P in {o.plans[(L, "down")].parts for L in self.layers}}

    @staticmethod
    def _norm_cols(n: int) -> int:
        """A norm block's cores: half the columns for short blocks (in-process A/B at T = 64 / 128: 4 faster than 2 and
        than 8)."""
        return 4 if n <= 256 else COLS

    @staticmethod
    def _blocks(T: int, t: int, o: _Ops) -> list[tuple[int, int]]:
        """A chunk's norm blocks (first token, tokens): the largest of NORM_SIZES that fit, then sub-chunks of t tokens."""
        out, a = [], 0
        while a < T:
            n = next((m for m in NORM_SIZES if m in o.norms and T - a >= m), t)
            out.append((a, n))
            a += n
        return out

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers (the sub-chunk size's layouts: at most the largest); the norms'
        parameter objects."""
        cfg, H, s = self.cfg, self.H, self.shape
        pitch, dp = max(o.pitch for o in self.ops.values()), max(o.d_pitch for o in self.ops.values())
        self.rb = {"PX": 2 * H, "PXN": 9 * H // 8, "G": 2 * COLS * pitch, "QS": COLS * s.qh * s.qunit // 8,
                   "A": 9 * COLS * s.qh * s.hd // 8, "Y": 2 * COLS * dp, "HP": 9 * cfg.intermediate_size // 8}
        self.np_off = {"in": 0}  # (the norms' weights: H bf16 each)
        k = 1
        for L in self.layers:
            for n in NORMS:
                self.np_off[(L, n)] = k * 2 * H
                k += 1
        self.np_bytes = k * 2 * H

    def _np(self, key: object) -> str:
        o = self.np_off[key]
        return f"NP[{o}:{o + 2 * self.H}]"

    def _runlist(self, S: int) -> list[tuple]:
        return self._runlist_of(S * SUB)

    def _runlist_of(self, T: int) -> list[tuple]:
        """(the sub-chunks of ``_t``: a size below SUB one of its own operators)"""
        t = self._t(T)
        return self._ops_runlist(T // t, self.ops[t])

    def _layer_starts(self, T: int) -> list[int]:
        return self._starts[T]

    def _ops_runlist(self, S: int, o: _Ops) -> list[tuple]:
        """The runlist of S sub-chunks of the operators ``o`` (their layers' first items in ``_starts``)."""
        ov, s, sl = self.ov, self.shape, self._sl
        t = o.n_in.T  # (a sub-chunk's tokens)

        def tok(buf: str, j: int, op: object, *, i: int, row: int = 0, extra: int = 0) -> str:
            return sl(buf, j * t * (row or self.rb[buf]) + extra, op, i=i)

        rb, P = self.rb, o.parts
        ry = 2 * COLS * o.d_pitch  # (bytes per token of Y and F at these operators' pitch: a norm block's rows)
        rg = 2 * COLS * o.pitch  # (of G: a prep block's)
        # norm blocks; F per block [part][its tokens], then the down GEMMs' spill (their C rows past t tokens)
        blocks = self._blocks(S * t, t, o)
        spill = o.f_part - t * ry
        f_base = list(itertools.accumulate((P * nt * ry + spill for _, nt in blocks), initial=0))

        def f_at(tk: int, part: int) -> int:  # (token tk's row of a down part in F)
            k = max(i for i, (a, _) in enumerate(blocks) if a <= tk)
            return f_base[k] + (part * blocks[k][1] + tk - blocks[k][0]) * ry

        rl: list[tuple] = [(o.n_in, tok("PX", j, o.n_in, i=0), self._np("in"), tok("PXN", j, o.n_in, i=2))
                           for j in range(S)]
        starts = []
        for L in self.layers:
            starts.append(len(rl))
            qt = o.qt
            rl += [(g.op, sl("PXN", j * qt * rb["PXN"], g.op, i=0), self._w(g), sl("G", j * qt * rg, g.op, i=2))
                   for g in o.plans[(L, "qkv")].gemms for j in range(S * t // qt)]
            pt = self._pt(S * t)  # (the prep's blocks of the chunk)
            prep, attn = o.preps[(_key(o.plans[(L, "qkv")].layout), pt)], self.attn[S * t]
            kv0, kvb = ov.kv_off[L], arg_bytes(prep, 3)
            pg = pt // s.pg * s.group
            rl += [(prep, tok("G", 0, prep, i=0, row=rg, extra=b * pt * rg), sl("CS", b * pt * 4 * s.hd, prep, i=1),
                    tok("QS", 0, prep, i=2, extra=b * pt * rb["QS"]), f"KV[{kv0 + b * pg}:{kv0 + b * pg + kvb}]")
                   for b in range(S * t // pt)]
            rl.append((attn, sl("QS", 0, attn, i=0), f"KV[{kv0}:{kv0 + arg_bytes(attn, 1)}]", sl("A", 0, attn, i=2)))
            (g,) = o.plans[(L, "o")].gemms
            rl += [(g.op, tok("A", j, g.op, i=0), self._w(g), tok("Y", j, g.op, i=2, row=ry)) for j in range(S)]
            for a, nt in blocks:
                n = o.norms[nt][0]
                rl.append((n, tok("PX", 0, n, i=0, extra=a * rb["PX"]), tok("Y", 0, n, i=1, extra=a * ry),
                           self._np((L, "o")), tok("PX", 0, n, i=3, extra=a * rb["PX"]),
                           tok("PXN", 0, n, i=4, extra=a * rb["PXN"])))
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("HP", j, g.op, i=2))
                   for g in o.plans[(L, "gate_up")].gemms for j in range(S)]
            rl += [(g.op, tok("HP", j, g.op, i=0, extra=2 * g.a_off), self._w(g),
                    sl("F", f_at(j * t, g.part), g.op, i=2)) for g in o.plans[(L, "down")].gemms for j in range(S)]
            for (a, nt), base in zip(blocks, f_base):
                n = o.norms[nt][1][o.plans[(L, "down")].parts]
                rl.append((n, tok("PX", 0, n, i=0, extra=a * rb["PX"]), sl("F", base, n, i=1), self._np((L, "d")),
                           tok("PX", 0, n, i=3, extra=a * rb["PX"]), tok("PXN", 0, n, i=4, extra=a * rb["PXN"])))
        self._starts[S * t] = [*starts, len(rl)]
        return rl

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        own["F"] = own["F"] + max(o.f_part for o in self.ops.values())

    def _bind(self) -> None:
        """The norms' parameter objects (NP), the cos / sin table (CS)."""
        dev, H = self.ov.dev, self.H
        npb = dev.view("NP").view(np.uint16)

        def put(key: object, w1: str) -> None:
            o = self.np_off[key] // 2
            w = torch.from_numpy(np.asarray(self.ov.gguf.tensor(w1).dequantize(), np.float32).reshape(-1))
            npb[o:o + H] = w.bfloat16().view(torch.int16).numpy().view(np.uint16)

        put("in", f"blk.{self.layers[0]}.attn_norm.weight")
        for i, L in enumerate(self.layers):
            put((L, "o"), f"blk.{L}.ffn_norm.weight")
            put((L, "d"), f"blk.{self.layers[i + 1]}.attn_norm.weight" if i + 1 < len(self.layers)
                else "output_norm.weight")
        dev.flush("NP")
        cs = PA.cs_table(self.cfg.rope_inv_freq(), self.cs_rows).view(np.uint8).reshape(-1)
        dev.view("CS")[:cs.size] = cs
        dev.flush("CS")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def params(self, T: int, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0."""
        prep = next(p for (_, pt), p in self.ops[self._t(T)].preps.items() if pt == self._pt(T))
        return prep.params(p0) | self.attn[T].params(p0)

    def run(self, x: torch.Tensor, p0: int, T: int = 0) -> None:
        """A chunk of size T (default: the smallest that holds x): x (n, hidden) the embeddings of positions p0 .. p0 +
        n - 1 (the caches hold the positions before). Leaves the layers' K / V of the positions in the caches and the
        last layer's output in PX (``hidden``)."""
        n, dev, H = x.shape[0], self.ov.dev, self.H
        T = self._size(n, p0, T)
        px = dev.view("PX")[:2 * T * H].view(np.uint16).reshape(T, H)
        px[:n] = x.bfloat16().view(torch.int16).numpy().view(np.uint16)
        px[n:] = 0
        dev.flush("PX", 0, 2 * T * H)
        self._launch(T, p0, self.params(T, p0))

    def hidden(self, i: int) -> torch.Tensor:
        """Row i (hidden,) of the last chunk's output residual (bf16 values; the LM head's input before its norm)."""
        H, dev = self.H, self.ov.dev
        dev.fetch("PX", 2 * i * H, 2 * H)
        row = dev.view("PX")[2 * i * H:2 * (i + 1) * H].view(np.int16).copy()
        return torch.from_numpy(row).view(torch.bfloat16).float()

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def attention_seconds(self, p0: int, S: float) -> float:
        """Estimated time of a layer's flash attention over a chunk of S sub-chunks at p0: per pass the cache records
        it streams (from position 0 to its queries' group), RECORD_S each."""
        s, at = self.shape, self.attn[round(S * SUB)]
        q, r = divmod(at.n_pass, at.m)  # (sum over the passes b of b // m)
        recs = (at.n_pass * (p0 // s.pg + 1) + at.m * q * (q - 1) // 2 + q * r) * s.rg
        return RECORD_S * recs

    def layer_seconds(self, L: int, p0: int, S: float = 1) -> float:
        """(per sub-chunk, or the fraction T / SUB of one, the GEMMs by the bytes of their weights, WEIGHT_S each (the
        norms and the prep included); the flash attention (``attention_seconds``); CFG_S per ConfigureOp)"""
        return self._layer_static(L, S) + self.attention_seconds(p0, S)

    def _layer_static(self, L: int, S: float) -> float:
        """(``layer_seconds`` but the attention's; cached: the planner asks for every layer and size)"""
        key = (L, S)
        if key not in self._static_s:
            wbytes = sum(self.img.region(L, name).nbytes for name in PHASES)
            self._static_s[key] = S * WEIGHT_S * wbytes + CFG_S * self.layer_configs(L)
        return self._static_s[key]

    def layer_configs(self, L: int) -> int:
        """(its GEMM objects, the prep and attention, the norms)"""
        return len({id(g.op) for name in PHASES for g in self.plans[(L, name)].gemms}) + 2 + 2
