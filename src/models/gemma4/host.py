"""What the host computes for Gemma 4 on the NPU: the scaled token embeddings and the per-layer inputs (E2B / E4B),
from the GGUF file's tensors (llama.cpp's ``src/models/gemma4.cpp``).

``GemmaHost`` reads matrices dequantized to fp32 on first use, embedding rows by a lookup in the mapped file
(``GGUFFile.rows``). Its ``mm`` and ``norm`` are methods so that models of other arithmetic can override them (the fp32
reference of the whole model, tests/gemma4/cpu.py, derives from it).
"""

import math

import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """x / sqrt(mean(x^2) + eps) (* w) over the last dimension."""
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Final logit soft-capping: cap * tanh(logits / cap) (monotonic)."""
    return torch.tanh(logits / cap) * cap if cap else logits


class GemmaHost:
    """The tensors of a GGUF file the host reads, and the inputs it computes from them."""

    def __init__(self, gguf: GGUFFile, *, threads: int = 8):
        self.gguf = gguf
        self.cfg = GemmaConfig.from_gguf(gguf)
        self._mats: dict[str, torch.Tensor] = {}
        self._bf16: dict[str, torch.Tensor] = {}  # (bf16 copies of BF16 matrices: per_layer_inputs(bf16=True))
        self._threads = threads

    def mat(self, name: str) -> torch.Tensor:
        """The dequantized matrix (rows, cols) fp32, or vector (n,)."""
        if name not in self._mats:
            t = self.gguf.tensor(name)
            w = t.dequantize()
            self._mats[name] = w.reshape(-1) if t.rows == 1 else w
        return self._mats[name]

    def mm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """x (T, K) @ W^T."""
        return x @ self.mat(name).T

    def norm(self, x: torch.Tensor, name: str | None) -> torch.Tensor:
        """RMSNorm over the last dimension with the weight vector ``name`` (None: no weight)."""
        return rms_norm(x, None if name is None else self.mat(name), self.cfg.rms_norm_eps)

    def embed(self, tokens: list[int]) -> torch.Tensor:
        """(T, hidden): the token embeddings * sqrt(hidden)."""
        return self.gguf.rows("token_embd.weight", tokens) * math.sqrt(self.cfg.hidden_size)

    def per_layer_inputs(self, tokens: list[int], x: torch.Tensor, *, bf16: bool = False,
                         layers: int = 0) -> torch.Tensor | None:
        """(T, layers, per_layer_dim): (RMSNorm(per_layer_model_proj @ x / sqrt(hidden)) +
        sqrt(per_layer_dim) * per_layer_token_embd[token]) / sqrt(2). bf16: the projection as a bf16 matmul (its
        weight is BF16 in the file, x rounded to bf16: ~0.2% of the projection; 0.5 instead of 4.4 ms per token on
        Zen 5): the NPU app's inputs, which go to the NPU as bf16; ``layers``: of the first that many layers only (a
        prompt chunk's: the layers that own a cache)."""
        cfg = self.cfg
        if not cfg.per_layer_dim:
            return None
        T, n, d = len(tokens), layers or cfg.num_hidden_layers, cfg.per_layer_dim
        name = "per_layer_model_proj.weight"
        if bf16:  # (fused: rms(p / sqrt(hidden)) = rms(p) with eps * hidden, by vector_norm; the rows' dequant in torch
            # ops: 20 instead of 73 ms unfused for the 24 own-cache layers of 1031 tokens on Zen 5)
            if name not in self._bf16:
                self._bf16[name] = self.gguf.tensor(name).dequantize().bfloat16()
            # (one row as two: torch's 1-row product rounds otherwise than a row of any larger one, and the app keeps
            # a token's inputs from the first batch it was in)
            xb = x.bfloat16() if T > 1 else x.bfloat16().repeat(2, 1)
            proj = (xb @ self._bf16[name][:n * d].T)[:T].float().view(T, n, d)
            inv = torch.linalg.vector_norm(proj, dim=-1, keepdim=True).mul_(1 / math.sqrt(d)).pow_(2)
            inv = inv.add_(cfg.rms_norm_eps * cfg.hidden_size).rsqrt_()
            y = proj.mul_(inv).mul_(self.mat("per_layer_proj_norm.weight"))
            tok = self.gguf.rows("per_layer_token_embd.weight", tokens, n * d).view(T, n, d)
            return torch.add(y, tok, alpha=math.sqrt(d)).mul_(1 / math.sqrt(2.0))
        proj = (self.mm(name, x) / math.sqrt(cfg.hidden_size)).view(T, n, d)
        proj = self.norm(proj, "per_layer_proj_norm.weight")
        tok = self.gguf.rows("per_layer_token_embd.weight", tokens).view(T, n, d) * math.sqrt(d)
        return (proj + tok) / math.sqrt(2.0)
