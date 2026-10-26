"""Host model of the decode overlay's attention cores (``artlite.decode.attn``: Gemma's and Llama's), and the roundings
the models' host models share: q, k (RMS-normed: Gemma) rotated in fp32 (fp32 cos / sin), k cached as bfp16 hi + lo
blocks (or one), v (normed to bf16: Gemma) cached as bfp16 blocks per dim over the 8 positions of a record, encoded
from the bf16 values of its positions up to the attending one (all 8 once it is past the record), exact softmax of
the scores, o rounded to bf16. (Not modelled: the Q panels' 14 bits, the P panels, exp2's bf16-level polynomial.)
"""

import numpy as np
import torch

from artlite.operators import bfp16


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
    """One layer's cache and attention as the attention cores compute them (``decode.attn``): ``store`` a position's K,
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
