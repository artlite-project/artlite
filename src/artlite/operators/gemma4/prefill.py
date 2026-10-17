"""The prefill of a Gemma 4 model: a prompt chunk of up to ``MAX_T`` tokens through the layers that own a KV cache, on
the decode overlay's buffers (``overlay.GemmaOverlay``: the weight image, the KV caches), so the decode continues from
the caches the chunk leaves. The layers that share a cache (E4B's 24 .. 41) only matter for a prompt's last token: a
decode step from the last own-cache layer on (its input: ``residual``) finishes it, with the LM head
(``NpuGemma.forward``).

A chunk of T tokens (``SIZES``: multiples of 288) runs layer-major in S = T / ``SUB`` sub-chunks: each operator of a
layer runs over the sub-chunks back to back on buffer slices ``SUB`` tokens apart (consecutive runs of one operator
share its configuration: a ConfigureOp per layer and operator), the flash attention once over the whole chunk.
The residual stream x is fp32 (PX, ``prefill.norm.ResidualNormOperator``). Per layer L (the decode's order):

- qkv GEMM (``prefill.gemm``) of xn (the norm's bfp16 panels) -> G; the attention prep of the layer kind
  (``attn_prefill``: Q panels into QS, the chunk's K / V records into the layer's cache), the flash attention over the
  chunk -> A (the O GEMM's panels);
- O GEMM -> Y; norm ``post``: x += norm(Y) post_attention_norm, xn = norm(x) ffn_norm;
- gate_up GEMM with the GELU epilogue -> HP; down GEMM in four K parts -> F; norm ``plain``: x += norm(F)
  post_ffw_norm, xn = x (the per-layer input gate's input);
- gate GEMM -> GY; ``prefill_ple``: gelu(GY) pli -> HPL (the host's per-layer inputs of the layer: PLI); proj GEMM ->
  P; norm ``post`` with the output scale: x = (x + norm(P) post_norm) layer_output_scale, xn = norm(x) of the next
  layer's attn_norm.

The last layer stops after its attention prep (its cache is all the prefill leaves). The first layer's input norm
(``in``) starts the chunk.

The control code is assembled on the host from one library ELF and runs in pieces of whole layers
(``prefill.chunks.ChunkPrefill``). A sub-chunk is the GEMMs' whole MAC tiles: 3 MAC rows of 96 tokens.

Precision: layer 23's caches end within 2-3% (K / V rms) of the decode path's, ~4% / 6% of the fp32 model's on long
prompts (the decode path: ~1% / 1.6%). The GEMMs' single bfp16 inputs, their weights as bfp16 panels (the Q4_0 scale
rounded into 8-bit mantissas) and the MAC loop's bf16 C tile between k-tiles cost about equally (a CPU simulation of
the rounding points); the down GEMM's K parts halve its C tile's share; the attention's P as single bfp16 panels
would double the error (hence hi + lo).
"""

from concurrent.futures import Future

import numpy as np
import torch

from ..design import COLS
from ..prefill import gemm as PG
from ..prefill.chunks import CONFIG_S, ChunkPrefill
from ..prefill.norm import ResidualNormOperator, residual_params
from ..util import arg_bytes
from . import attn_prefill as AP
from .overlay import GUARD, KV_PAD, GemmaOverlay
from .prefill_ple import PleOperator

SUB = 288  # tokens per sub-chunk (the operators' T: the GEMMs' whole passes of 3 MAC rows x 96 tokens)
SIZES = tuple(range(SUB, 2049, SUB))  # chunk sizes (a flash attention design each per layer kind)
MAX_T = SIZES[-1]
PAD_ROWS = 64  # rows past a chunk's buffers (a GEMM's padding rows past its tokens, if any)
PHASES = ("qkv", "o", "gate_up", "down", "gate", "proj")
DOWN_CHUNKS = 5  # K chunks per down part (4 of E4B's 20: each part's A strip stays in the memtile, each sums 40
# k-tiles in its bf16 C tile; the norm adds the parts in fp32)
NORMS = ("o", "d", "p")  # a layer's residual updates (after o, down, proj)


def plan_gemm(img: object, region: object, T: int, *, pitch: int = 0, ops: dict[str, object] | None = None
              ) -> PG.PhasePlan:
    """A GEMM phase of the prefill (``prefill.gemm.plan_phase``): gate / up with the GELU epilogue, down in K parts of
    ``DOWN_CHUNKS`` chunks."""
    return PG.plan_phase(img, region, T, pitch=pitch, ops=ops, act="gelu",
                         k_part=DOWN_CHUNKS if region.name == "down" else 0)


class GemmaPrefill(ChunkPrefill):
    """The prefill of an overlay's own-cache layers in chunks of the sizes ``sizes`` (those within the overlay's
    caches); ``run`` runs one."""

    SUB, MAX_T, NAME = SUB, MAX_T, "gemma_prefill"

    def __init__(self, overlay: GemmaOverlay, *, sizes: tuple[int, ...] = SIZES):
        super().__init__(overlay, sizes)
        ov, cfg, img, H = overlay, self.cfg, self.img, self.H
        assert cfg.per_layer_dim == COLS * 32, "per-layer inputs: 32 per column"
        own = [L for L in ov.layers if cfg.kv_source(L) == L]
        assert own == ov.layers[:len(own)], "the own-cache layers first"
        self.layers = self.kv_layers = own
        self.d_pitch = dp = PG.d_pitch(img, SUB)
        self.pitch = {}  # (the qkv layout's pitch per head_dim)
        for L in own:
            hd = cfg.head_dim(L)
            r = img.region(L, "qkv")
            self.pitch[hd] = max(self.pitch.get(hd, 0), plan_gemm(img, r, SUB).layout.pitch)
        gemms: dict[str, object] = {}  # (one object per distinct GEMM)
        self.plans = {(L, name): plan_gemm(img, img.region(L, name), SUB, ops=gemms, pitch=dp if name in PG.OUT else
                                           self.pitch[cfg.head_dim(L)] if name == "qkv" else 0)
                      for L in own for name in PHASES}
        down = [self.plans[(L, "down")] for L in own]
        self.parts = down[0].parts
        assert all(d.parts == self.parts for d in down) and self.parts in (1, 2, 4)
        self.f_part = max(arg_bytes(g.op, 2) for d in down for g in d.gemms)  # (a down part's C)
        gl = self.plans[(own[0], "gate")].layout
        assert gl.pitch == 64 and gl.off["gate"] == 0, gl
        eps = cfg.rms_norm_eps
        self.n_in = ResidualNormOperator(T=SUB, E=H, mode="in", eps=eps)
        self.n_o = ResidualNormOperator(T=SUB, E=H, mode="post", eps=eps, d_pitch=dp)
        self.n_d = ResidualNormOperator(T=SUB, E=H, mode="plain", eps=eps, d_pitch=dp, deltas=self.parts,
                                     d_stride=self.f_part // 2 if self.parts > 1 else 0)
        self.n_p = ResidualNormOperator(T=SUB, E=H, mode="post", eps=eps, d_pitch=dp, scale=True)
        self.ple = PleOperator(T=SUB)
        self.hds = sorted({cfg.head_dim(L) for L in own})
        self.kind = {hd: AP.kind_of(cfg, hd, ov.at.kp) for hd in self.hds}
        self.cs_rows = ov.max_positions + MAX_T
        self.kv_bytes = {cfg.head_dim(L): ov._kv_bytes(L) for L in own}
        pad = KV_PAD // 16 * ov.G * AP.OBJ  # (the KV arguments past a layer's caches: the overlay's padding at the end)
        self.prep = {hd: AP.GemmaPrepOperator(T=SUB, cfg=cfg, hd=hd, layout=self.plans[(next(
            L for L in own if cfg.head_dim(L) == hd), "qkv")].layout, kv_bytes=self.kv_bytes[hd],
            cs_positions=self.cs_rows, kv_pad=pad, kp=ov.at.kp) for hd in self.hds}
        self.attn = {(hd, T): AP.GemmaAttnOperator(T=T, cfg=cfg, hd=hd, kv_bytes=self.kv_bytes[hd], kv_pad=pad,
                                                   kp=ov.at.kp) for hd in self.hds for T in self.sizes}
        self._rows()
        self._library()

    # ------------------------------------------------------------------
    # The runlists
    # ------------------------------------------------------------------

    def _rows(self) -> None:
        """Bytes per token of the chunk's buffers."""
        cfg, H = self.cfg, self.H
        hd = max(self.hds)
        self.rb = {"PX": 4 * H, "PXN": 9 * H // 8, "G": 2 * COLS * max(self.pitch.values()), "A": 9 * COLS * hd // 8,
                   "Y": 2 * COLS * self.d_pitch, "HP": 9 * cfg.intermediate_size // 8, "GY": 2 * COLS * 64,
                   "HPL": 9 * COLS * 64 // 8, "P": 2 * COLS * self.d_pitch, "PLI": 2 * cfg.per_layer_dim,
                   "QS": COLS * max(k.qp for k in self.kind.values()) // 8}
        self.np_off = {}  # (the norms' parameter objects: [8][H] bf16 each)
        self.np_off["in"] = 0
        k = 1
        for L in self.layers[:-1]:
            for n in NORMS:
                self.np_off[(L, n)] = k * 16 * H
                k += 1
        self.np_bytes = k * 16 * H
        self.prm_off = {L: i * 4096 for i, L in enumerate(self.layers)}  # (the preps' parameter objects)

    def _np(self, key: object, op: object) -> str:
        o = self.np_off[key]
        return f"NP[{o}:{o + arg_bytes(op, 0)}]"

    def _runlist(self, S: int) -> list[tuple]:
        ov, cfg, rb = self.ov, self.cfg, self.rb
        sl, tok = self._sl, self._tok
        T, fp = S * SUB, self.f_part
        rl: list[tuple] = [(self.n_in, self._np("in", self.n_in), tok("PX", j, self.n_in, i=1),
                            tok("PXN", j, self.n_in, i=2)) for j in range(S)]
        for i, L in enumerate(self.layers):
            hd = cfg.head_dim(L)
            kd = self.kind[hd]
            rg, rq, ra = 2 * COLS * self.pitch[hd], COLS * kd.qp // 8, 9 * hd  # (the kind's G, QS, A rows)
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("G", j, g.op, i=2, row=rg))
                   for g in self.plans[(L, "qkv")].gemms for j in range(S)]
            prep, attn = self.prep[hd], self.attn[(hd, T)]
            kv0, kvb = ov.kv_off[L], arg_bytes(prep, 4)
            grp = SUB // kd.pg * kd.group  # (a sub-chunk's cache groups)
            cs = "CSS" if kd.swa else "CSF"
            prm = f"PRM[{self.prm_off[L]}:{self.prm_off[L] + arg_bytes(prep, 2)}]"
            rl += [(prep, tok("G", j, prep, i=0, row=rg), sl(cs, j * SUB * 8 * kd.rot, prep, i=1), prm,
                    tok("QS", j, prep, i=3, row=rq), f"KV[{kv0 + j * grp}:{kv0 + j * grp + kvb}]") for j in range(S)]
            if i == len(self.layers) - 1:  # (the last layer: its cache only)
                break
            rl.append((attn, sl("QS", 0, attn, i=0), f"KV[{kv0}:{kv0 + kvb}]", sl("A", 0, attn, i=2)))
            (g,) = self.plans[(L, "o")].gemms
            rl += [(g.op, tok("A", j, g.op, i=0, row=ra), self._w(g), tok("Y", j, g.op, i=2)) for j in range(S)]
            n = self.n_o
            rl += [(n, self._np((L, "o"), n), tok("PX", j, n, i=1), tok("Y", j, n, i=2), tok("PX", j, n, i=3),
                    tok("PXN", j, n, i=4)) for j in range(S)]
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("HP", j, g.op, i=2))
                   for g in self.plans[(L, "gate_up")].gemms for j in range(S)]
            P = self.parts  # (sub-chunk j's down parts at (P j + part) fp)
            rl += [(g.op, tok("HP", j, g.op, i=0, extra=2 * g.a_off), self._w(g), sl("F", (P * j + g.part) * fp, g.op, i=2))
                   for g in self.plans[(L, "down")].gemms for j in range(S)]
            n = self.n_d
            rl += [(n, self._np((L, "d"), n), tok("PX", j, n, i=1), sl("F", P * j * fp, n, i=2), tok("PX", j, n, i=3),
                    tok("PXN", j, n, i=4)) for j in range(S)]
            (g,) = self.plans[(L, "gate")].gemms
            rl += [(g.op, tok("PXN", j, g.op, i=0), self._w(g), tok("GY", j, g.op, i=2)) for j in range(S)]
            pli = self.layers.index(L) * (MAX_T + PAD_ROWS) * rb["PLI"]
            rl += [(self.ple, tok("GY", j, self.ple, i=0), tok("PLI", j, self.ple, i=1, extra=pli),
                    tok("HPL", j, self.ple, i=2)) for j in range(S)]
            (g,) = self.plans[(L, "proj")].gemms
            rl += [(g.op, tok("HPL", j, g.op, i=0), self._w(g), tok("P", j, g.op, i=2)) for j in range(S)]
            n = self.n_p
            rl += [(n, self._np((L, "p"), n), tok("PX", j, n, i=1), tok("P", j, n, i=2), tok("PX", j, n, i=3),
                    tok("PXN", j, n, i=4)) for j in range(S)]
        return rl

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _own_sizes(self, own: dict[str, int]) -> None:
        for name, r in self.rb.items():  # (the GEMMs' padding rows past the last sub-chunk)
            own[name] = max(own.get(name, 0), (MAX_T + PAD_ROWS) * r) + (
                (len(self.layers) - 1) * (MAX_T + PAD_ROWS) * r if name == "PLI" else 0)
        own["F"] = own["F"] + self.f_part

    def _bind(self) -> None:
        """The norms' parameter objects (NP), the preps' (PRM), the cos / sin tables (CSS, CSF)."""
        ov, dev, cfg, img, H = self.ov, self.ov.dev, self.cfg, self.img, self.H
        npb = dev.view("NP").view(np.uint16)

        def put(key: object, wp: str | None, w1: str | None, scale: float) -> None:
            o = self.np_off[key] // 2
            v = residual_params(None if wp is None else img._vec(wp), None if w1 is None else img._vec(w1), scale, H)
            npb[o:o + v.size] = v.reshape(-1)

        put("in", None, f"blk.{self.layers[0]}.attn_norm.weight", 1.0)
        for i, L in enumerate(self.layers[:-1]):
            b = f"blk.{L}."
            put((L, "o"), b + "post_attention_norm.weight", b + "ffn_norm.weight", 1.0)
            put((L, "d"), b + "post_ffw_norm.weight", None, 1.0)
            sc = b + "layer_output_scale.weight"
            put((L, "p"), b + "post_norm.weight", f"blk.{self.layers[i + 1]}.attn_norm.weight",
                ov._scalar(sc) if sc in ov.gguf else 1.0)
        dev.flush("NP")
        prm = dev.view("PRM")
        for L, o in self.prm_off.items():
            c = AP.score_constant(ov._scalar(f"blk.{L}.attn_q_norm.weight"), ov._scalar(f"blk.{L}.attn_k_norm.weight"))
            v = AP.prm_object(self.kind[cfg.head_dim(L)], c).view(np.uint8)
            prm[o:o + v.size] = v
        dev.flush("PRM")
        for hd, name in ((256, "CSS"), (512, "CSF")):
            if hd in self.kind:
                cs = AP.cs_table(cfg, self.kind[hd], self.cs_rows).view(np.uint8).reshape(-1)
                dev.view(name)[:cs.size] = cs
                dev.flush(name)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def params(self, T: int, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk of size T at position p0."""
        out = {}
        for hd in self.hds:  # (the operators of the library: e.g. the last layer's kind has no attention there)
            out |= self.prep[hd].params(p0, GUARD if self.kind[hd].swa else 0)
            if self.attn[(hd, T)] in self.lib:
                out |= self.attn[(hd, T)].params(p0)
        return out

    def residual(self, i: int) -> torch.Tensor:
        """Row i of the last chunk's residual stream (hidden,) fp32 where the chunk leaves it: the input of the last
        layer (which only writes its cache here). A decode step from that layer on (``GemmaOverlay.step(start=)``)
        finishes the token: a prompt's last token without the layers before."""
        H, dev = self.H, self.ov.dev
        dev.fetch("PX", 4 * i * H, 4 * H)
        return torch.from_numpy(dev.view("PX")[4 * i * H:4 * (i + 1) * H].view(np.float32).copy())

    def _write_pli(self, pli: torch.Tensor, n: int, T: int) -> None:
        dev, d = self.ov.dev, self.cfg.per_layer_dim
        stride = (MAX_T + PAD_ROWS) * d  # (a layer's rows, bf16)
        pl = dev.view("PLI").view(np.uint16)
        b = pli.reshape(n, -1, d)[:, :len(self.layers)].bfloat16().view(torch.int16).numpy().view(np.uint16)
        for i in range(len(self.layers) - 1):
            pl[i * stride:i * stride + n * d] = b[:, i].reshape(-1)
            pl[i * stride + n * d:i * stride + T * d] = 0
        dev.flush("PLI")

    def run(self, x: torch.Tensor, pli: torch.Tensor | Future, p0: int, T: int = 0) -> None:
        """A chunk of size T (default: the smallest that holds x): x (n, hidden) the scaled embeddings and pli (n,
        layers, per_layer_dim) the per-layer inputs (of the first layers at least: those of the prefill) of positions
        p0 .. p0 + n - 1 (the caches hold the positions before). Leaves the own-cache layers' K / V of the positions in
        the caches. pli may be a Future: the chunk starts without it, and its first PLE item waits for it (a new
        piece: the host computes the per-layer inputs while layer 0's first operators run)."""
        n, dev, H = x.shape[0], self.ov.dev, self.H
        T = self._size(n, p0, T)
        px = dev.view("PX")[:4 * T * H].view(np.float32).reshape(T, H)
        px[:n] = x.float().numpy()
        px[n:] = 0
        dev.flush("PX", 0, 4 * T * H)
        hooks = None
        if isinstance(pli, Future):  # (a piece from the first PLE item on)
            i = next(i for i, (op, *_) in enumerate(self.runlists[T]) if op is self.ple)
            hooks = {i: lambda: self._write_pli(pli.result(), n, T)}
        else:
            self._write_pli(pli, n, T)
        self._launch(T, p0, self.params(T, p0), hooks)

    # ------------------------------------------------------------------
    # Time estimates
    # ------------------------------------------------------------------

    def attention_seconds(self, L: int, p0: int, S: int) -> float:
        """Estimated time of layer L's flash attention over a chunk of S sub-chunks at p0: the KV bytes its passes
        stream at ~40 GB/s (a sliding-window pass: 17 groups; a full one: every group from position 0) plus ~0.3 ms per
        sub-chunk."""
        kd = self.kind[self.cfg.head_dim(L)]
        T = S * SUB
        if kd.swa:
            groups = T // 32 * AP.WINDOW_GROUPS
        else:
            groups = sum(p0 // 16 + 2 * b + 2 for b in range(T // 32))
        return 2 * groups * kd.group / 40e9 + 0.3e-3 * S

    def layer_seconds(self, L: int, p0: int, S: int = 1) -> float:
        """(per sub-chunk the GEMMs by the bytes of their weights (~45 ms per GB: Gemma's tiles stream a third more
        than they hold) plus ~0.1 ms each, ~0.15 ms per norm and op; the flash attention (``attention_seconds``);
        CONFIG_S per ConfigureOp)"""
        wbytes = sum(self.img.region(L, name).nbytes for name in PHASES)
        s = S * (0.045e-9 * wbytes + 0.1e-3 * len(PHASES) + 0.15e-3 * 5)
        return s + self.attention_seconds(L, p0, S) + CONFIG_S * self.layer_configs(L)

    def layer_configs(self, L: int) -> int:
        if L == self.layers[-1]:
            return len({id(g.op) for g in self.plans[(L, "qkv")].gemms}) + 1
        gemms = {id(g.op) for name in PHASES for g in self.plans[(L, name)].gemms}
        return len(gemms) + 2 + 3 + 1

    def layer_entries(self, L: int, S: int) -> int:
        if L == self.layers[-1]:
            return S * (len(self.plans[(L, "qkv")].gemms) + 1)
        gemms = sum(len(self.plans[(L, name)].gemms) for name in PHASES)
        return S * (gemms + 1 + 3 + 1) + 1
