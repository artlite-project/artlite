"""Host model of the Gemma decode overlay's arithmetic (operators/gemma4: the shared decode overlay with post-norm
residual updates, the attention cores): fp32 math on the dequantized weights with the kernels' rounding points.

- main cores: the residual stream in fp32 (each column's slice), updated exactly at every header: (r + v / rms(v)
  w_post) scale with v the sub-layer's fp32 rows; a header's GEMV input from the bf16 copies, (bf16(r) + bf16(v) /
  rms(v) w_post) w1, scaled by 1 / rms of those values (``header``); every other GEMV output rounded to bf16;
  gelu(bf16(gate)) up, gelu(bf16(y)) pli. (Not modelled: the MMUL panels keep 14 bits below the largest of 8
  elements; the sums of 32 products are rounded to bf16 before their scale.)
- attention cores: q, k normed and rotated in fp32 (fp32 cos / sin), k cached as bfp16 hi + lo blocks (or one), v normed to
  bf16 and cached as bfp16 blocks per dim over the 8 positions of a record, encoded from the bf16 values of its
  positions up to the attending one (all 8 once it is past the record), exact softmax of the scores (q_norm k_norm
  scale), o rounded to bf16. (Not modelled: the Q panels' 14 bits, the P panels, exp2's bf16-level polynomial.)

``GemmaNpuRef`` runs a list of layers of a GGUF file (and a truncated LM head) that way, one token after the other.
"""

import numpy as np
import torch

from artlite.operators import bfp16

from .gemma_cpu import GemmaCpu, gelu


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """1 / rms over the last dimension (fp32)."""
    return torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + eps).float()


def hilo_values(x: torch.Tensor) -> torch.Tensor:
    """x (..., 8 n) as two bfp16ebs8 blocks per 8 elements (hi + lo)."""
    a = np.ascontiguousarray(x.numpy(), np.float32)
    hi = bfp16.values(a)
    return torch.from_numpy(hi + bfp16.values(a - hi))


def rope(y: torch.Tensor, cs: tuple[torch.Tensor, torch.Tensor], rot: int) -> torch.Tensor:
    """The pairs (i, i + hd / 2) of i < rot rotated (fp32)."""
    c, s = cs
    half = y.shape[-1] // 2
    out = y.clone()
    a, b = y[..., :rot], y[..., half:half + rot]
    out[..., :rot], out[..., half:half + rot] = a * c - b * s, a * s + b * c
    return out


class AttentionRef:
    """One layer's cache and attention as the attention cores compute them (``overlay.attn``): ``store`` a position's K,
    V projections, ``attend`` a query. ``norm``: q, k, v RMS-normed (Gemma; Llama: none); ``kp``: K as hi + lo bfp16
    blocks (2) or one block (1) per 8 dims."""

    def __init__(self, *, head_dim: int, heads: int, kv_heads: int, eps: float, window: int, rot: int, norm: bool = True,
                 kp: int = 2):
        self.hd, self.heads, self.nkv, self.eps, self.window, self.rot = head_dim, heads, kv_heads, eps, window, rot
        self.norm, self.kp = norm, kp
        self.K = torch.zeros(kv_heads, 0, head_dim)  # (normed,) rotated (without the k_norm weight), as cached
        self.V = torch.zeros(kv_heads, 0, head_dim)  # as cached: the records' bfp16 panels
        self.rows = torch.zeros(kv_heads, 0, head_dim)  # (normed,) bf16 (the attention cores' ring)

    def _n(self, x: torch.Tensor) -> torch.Tensor:
        return x * inv_rms(x, self.eps) if self.norm else x

    def store(self, k: torch.Tensor, v: torch.Tensor, cs: tuple[torch.Tensor, torch.Tensor]) -> None:
        p, hd = self.K.shape[1], self.hd
        k, v = k.view(self.nkv, hd), v.view(self.nkv, hd)
        kr = rope(self._n(k), cs, self.rot)
        kn = hilo_values(kr) if self.kp == 2 else torch.from_numpy(bfp16.values(np.ascontiguousarray(kr.numpy())))
        self.K = torch.cat([self.K, kn[:, None]], 1)
        self.rows = torch.cat([self.rows, bf(self._n(v))[:, None]], 1)
        r0 = p // 8 * 8  # the record of the position: its V panels encoded from its rows so far (what p sees of it)
        rec = np.zeros((self.nkv, 8, hd), np.float32)
        rec[:, :p + 1 - r0] = self.rows[:, r0:].numpy()
        rec = bfp16.values(np.ascontiguousarray(rec.transpose(0, 2, 1))).transpose(0, 2, 1)
        self.V = torch.cat([self.V[:, :r0], torch.from_numpy(np.ascontiguousarray(rec[:, :p + 1 - r0]))], 1)

    def attend(self, q: torch.Tensor, cs: tuple[torch.Tensor, torch.Tensor], c: float, p: int) -> torch.Tensor:
        """The attention of q (heads hd,) at position p (the positions up to p cached) -> (heads hd,) bf16-valued."""
        q = q.view(self.heads, self.hd)
        qn = rope(self._n(q), cs, self.rot)
        lo = max(0, p - self.window + 1) if self.window else 0
        per = self.heads // self.nkv
        out = []
        for h in range(self.heads):
            K, V = self.K[h // per, lo:p + 1].double(), self.V[h // per, lo:p + 1].double()
            out.append(torch.softmax((qn[h].double() * c) @ K.T, -1) @ V)
        return bf(torch.stack(out).float().reshape(-1))


class GemmaNpuRef(GemmaCpu):
    """The layers ``layers`` of a Gemma GGUF file (as a model of their own: the overlay's ``layers``) and the first
    ``vocab`` LM rows with the overlay's rounding points. ``run`` takes the tokens from position 0 and returns the raw
    logits (T, vocab) (before the soft cap), with sub-layer results in ``trace``."""

    def __init__(self, gguf: object, layers: list[int], vocab: int, *, kp: int = 2):
        super().__init__(gguf)
        self.layers, self.vocab, self.kp = layers, vocab, kp

    def mv(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """W x in fp32 (the GEMV's rows before their rounding)."""
        return x @ self.mat(name).T

    def header(self, r: torch.Tensor, v: torch.Tensor, wp: torch.Tensor, scale: float, w1: torch.Tensor | None,
               norm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A header phase: (the updated residual (fp32), its GEMV input x, the outputs' scale) from r, v (fp32)."""
        eps = self.cfg.rms_norm_eps
        iv = inv_rms(v, eps)
        r_new = (r + v * iv * wp) * scale
        approx = bf(r) + bf(v) * iv * wp
        x = approx * (w1 if w1 is not None else 1.0)
        return r_new, x, (inv_rms(bf(approx), eps) if norm else torch.ones(()))

    def run(self, tokens: list[int], trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        cfg, eps, T = self.cfg, self.cfg.rms_norm_eps, len(tokens)
        trace = {} if trace is None else trace
        hds = (cfg.head_dim_swa, cfg.head_dim_full)
        f = np.asarray(cfg.rope_freq_factors) if cfg.rope_freq_factors is not None else np.ones(hds[1] // 2)
        rot = {True: hds[0] // 2, False: int((f[:hds[1] // 2] < 1e10).sum())}
        x = self.embed(tokens)
        pli = bf(self.per_layer_inputs(tokens, x))
        r, v = x.clone(), None
        prev: tuple[torch.Tensor, float] | None = None  # (post norm weight, scale) of the update that ends a layer
        self.caches: dict[int, AttentionRef] = {}

        def scal(name: str) -> float:
            return float(self.mat(name).reshape(-1)[0]) if name in self.gguf else 1.0

        for layer in self.layers:
            b = f"blk.{layer}."
            hd, swa, src = cfg.head_dim(layer), cfg.swa[layer], cfg.kv_source(layer)
            w1 = self.mat(b + "attn_norm.weight")
            if prev is None:
                xin, sc = bf(r) * w1, inv_rms(bf(r), eps)
            else:
                r, xin, sc = self.header(r, v, prev[0], prev[1], w1, True)
            q = trace.setdefault(f"q.{layer}", bf(self.mv(b + "attn_q.weight", xin) * sc))
            th = torch.from_numpy(np.outer(np.arange(T), cfg.rope_inv_freq(swa=swa)[:rot[swa]]).astype(np.float32))
            if src == layer:
                k = bf(self.mv(b + "attn_k.weight", xin) * sc)
                vv = bf(self.mv(b + "attn_v.weight", xin) * sc)
                self.caches[layer] = AttentionRef(head_dim=hd, heads=cfg.num_attention_heads, kv_heads=cfg.kv_heads[layer],
                                                  eps=eps, window=cfg.sliding_window if swa else 0, rot=rot[swa],
                                                  kp=self.kp)
            cache = self.caches[src]
            c = scal(b + "attn_q_norm.weight") * scal(f"blk.{src}.attn_k_norm.weight")
            att = []
            for t in range(T):  # the cache as it stood at each token
                cs = (torch.cos(th[t]), torch.sin(th[t]))
                if src == layer:
                    cache.store(k[t], vv[t], cs)
                att.append(cache.attend(q[t], cs, c, t))
            att = trace.setdefault(f"attn.{layer}", torch.stack(att))
            v = trace.setdefault(f"o.{layer}", self.mv(b + "attn_output.weight", att))
            r, xin, sc = self.header(r, v, self.mat(b + "post_attention_norm.weight"), 1.0, self.mat(b + "ffn_norm.weight"),
                                     True)
            gate, up = self.mv(b + "ffn_gate.weight", xin) * sc, self.mv(b + "ffn_up.weight", xin) * sc
            h = trace.setdefault(f"h.{layer}", bf(gelu(bf(gate)) * up))
            v = trace.setdefault(f"down.{layer}", self.mv(b + "ffn_down.weight", h))
            if cfg.per_layer_dim:
                r, xin, _ = self.header(r, v, self.mat(b + "post_ffw_norm.weight"), 1.0, None, False)
                g = bf(bf(gelu(bf(self.mv(b + "inp_gate.weight", xin)))) * pli[:, layer])
                v = trace.setdefault(f"proj.{layer}", self.mv(b + "proj.weight", g))
                prev = (self.mat(b + "post_norm.weight"), scal(b + "layer_output_scale.weight"))
            else:
                prev = (self.mat(b + "post_ffw_norm.weight"), scal(b + "layer_output_scale.weight"))
        r, xin, sc = self.header(r, v, prev[0], prev[1], self.mat("output_norm.weight"), True)
        trace["r"] = r
        lm = self.gguf.rows("token_embd.weight", list(range(self.vocab)))
        return bf((xin @ lm.T) * sc)
