"""Host model of the Llama decode overlay's arithmetic (operators/llama: the shared decode overlay with pre-norm residual
updates, the attention cores): fp32 math on the dequantized weights with the kernels' rounding points.

- main cores: the residual stream in fp32 (each column's slice), the output projections' rows added exactly; a normed
  phase's input is the residual's bf16 copy times the norm weight (bf16), its rows scaled by 1 / rms of that copy and
  rounded to bf16; silu(bf16(gate)) rounded to bf16, times up, rounded to bf16. (Not modelled: the MMUL panels keep 14
  bits below the largest of 8 elements; the sums of 32 products are rounded to bf16 before their scale.)
- attention cores (``gemma_ref.AttentionRef`` without norms): q, k rotated in fp32 (fp32 cos / sin), k cached as one
  bfp16 block (kp 1) or hi + lo blocks (kp 2) per 8 dims, v cached as bfp16 blocks per dim over the 8 positions of a
  record, encoded from the bf16 values of its positions up to the attending one, exact softmax of the scores q k /
  sqrt(head_dim), o rounded to bf16. (Not modelled: the Q panels' 14 bits, the P panels, exp2's bf16-level polynomial.)

``LlamaNpuRef`` runs a list of layers of a GGUF file (and a truncated LM head) that way, one token after the other.
"""

import math

import numpy as np
import torch

from artlite.gguf_model import GGUFFile, unpermute_rows
from artlite.model_config import LlamaConfig
from artlite.operators import bfp16
from artlite.operators.formats import has_tile_format, tile_format

from .gemma_ref import AttentionRef, bf, inv_rms


class LlamaNpuRef:
    """The layers ``layers`` of a Llama GGUF file (as a model of their own: the overlay's ``layers``) and the first
    ``vocab`` LM rows with the overlay's rounding points; ``kp``: the caches' K panels per record. ``run`` takes tokens
    after the cached positions (``caches``: per layer an ``AttentionRef``) and returns the logits (T, vocab), with
    sub-layer results in ``trace``."""

    def __init__(self, gguf: GGUFFile, layers: list[int], vocab: int, *, kp: int = 1):
        self.gguf, self.layers, self.vocab, self.kp = gguf, layers, vocab, kp
        self.cfg = cfg = LlamaConfig.from_gguf(gguf)
        self._mats: dict[str, torch.Tensor] = {}
        self.caches = {L: AttentionRef(head_dim=cfg.head_dim, heads=cfg.num_attention_heads,
                                       kv_heads=cfg.num_key_value_heads, eps=cfg.rms_norm_eps, window=0,
                                       rot=cfg.head_dim // 2, norm=False, kp=kp) for L in layers}

    def mat(self, name: str) -> torch.Tensor:
        """A tensor dequantized (fp32; BF16 / F16 rounded to the BFP16 tiles' blocks), Q / K rows in rotate-half order
        (as the image's)."""
        if name not in self._mats:
            t = self.gguf.tensor(name)
            w = t.dequantize().float()
            if has_tile_format(t.qtype) and tile_format(t.qtype) == "BFP16":
                w = torch.from_numpy(bfp16.values(w.numpy()))
            cfg = self.cfg
            if name.endswith("attn_q.weight"):
                w = w[torch.from_numpy(unpermute_rows(cfg.num_attention_heads, w.shape[0]))]
            elif name.endswith("attn_k.weight"):
                w = w[torch.from_numpy(unpermute_rows(cfg.num_key_value_heads, w.shape[0]))]
            self._mats[name] = w
        return self._mats[name]

    def embed(self, tokens: list[int]) -> torch.Tensor:
        """The tokens' embedding rows (T, hidden) fp32."""
        return self.gguf.rows("token_embd.weight", tokens)

    def _normed(self, r: torch.Tensor, w: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A normed phase's input from the residual r: (x w, 1 / rms(x)) of its bf16 copy x."""
        x = bf(r)
        return x * bf(self.mat(w).reshape(-1)), inv_rms(x, self.cfg.rms_norm_eps)

    def run(self, tokens: list[int], position: int = 0, trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        cfg, T = self.cfg, len(tokens)
        trace = {} if trace is None else trace
        hd = cfg.head_dim
        r = self.embed(tokens)
        th = torch.from_numpy(np.outer(np.arange(position, position + T), cfg.rope_inv_freq()).astype(np.float32))
        c = 1 / math.sqrt(hd)
        for L in self.layers:
            b = f"blk.{L}."
            xin, sc = self._normed(r, b + "attn_norm.weight")
            q = bf((xin @ self.mat(b + "attn_q.weight").T) * sc)
            k = bf((xin @ self.mat(b + "attn_k.weight").T) * sc)
            v = bf((xin @ self.mat(b + "attn_v.weight").T) * sc)
            cache = self.caches[L]
            att = []
            for t in range(T):  # (the cache as it stood at each token)
                cs = (torch.cos(th[t]), torch.sin(th[t]))
                cache.store(k[t], v[t], cs)
                att.append(cache.attend(q[t], cs, c, position + t))
            att = trace.setdefault(f"attn.{L}", torch.stack(att))
            r = r + att @ self.mat(b + "attn_output.weight").T
            xin, sc = self._normed(r, b + "ffn_norm.weight")
            gate = bf((xin @ self.mat(b + "ffn_gate.weight").T) * sc)
            up = (xin @ self.mat(b + "ffn_up.weight").T) * sc
            h = trace.setdefault(f"h.{L}", bf(bf(torch.nn.functional.silu(gate)) * up))
            r = r + h @ self.mat(b + "ffn_down.weight").T
        trace["r"] = bf(r)
        xin, sc = self._normed(r, "output_norm.weight")
        lm = "output.weight" if "output.weight" in self.gguf else "token_embd.weight"
        return bf((xin @ self.gguf.rows(lm, list(range(self.vocab))).T) * sc)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative rms error (inf when a is not finite: a NaN would compare as no error)."""
    if not torch.isfinite(a.float()).all():
        return float("inf")
    return float((a.float() - b.float()).pow(2).sum().sqrt() / b.float().pow(2).sum().sqrt())
