"""References of LFM3-VL's vision encoder (the mmproj of projector ``lfm3vl``: LFM3-VL-2B's, d1-2B's): the encoder in
torch, and llama.cpp goldens through libmtmd (``tests.minicpmv.ref.golden``: the dump tool runs any mmproj; its
embeddings ``embd(0)`` are the image chunk's rows, the merged tokens in raster order).

The encoder (``Encoder``): llama.cpp's ``clip_graph_lfm3vl`` (``tools/mtmd/models/lfm3vl.cpp``) and transformers'
``Lfm3VlVisionModel`` + ``Lfm3VlMultiModalProjector`` on the mmproj's tensors for an image's pixels (3, H, W): the
patchifier (a linear map of a patch's pixels: two temporal kernels, on a still image summed, on a video's pair of frames
one each; no bias), RMSNorm
(``v.embedding_norm``), 14 layers [RMSNorm -> operator | RMSNorm -> SwiGLU 1024 -> 2560 -> 1024] with residuals, the
operator a gated depthwise 3 x 3 convolution over the patch grid (``conv2d_in_proj`` 1024 -> [B; C; x], y = C conv(B x)
with zero padding, ``conv2d_out_proj``) or bidirectional attention (16 query heads, 8 KV heads of 64; q / k RMSNorm per
head; the axial 2-D RoPE: dims i, i + 32 of a head turned by y theta^(-i / 16) for i < 16 and by x theta^(-(i - 16) /
16) above, theta 1e6), no norm after the layers, then the projector: LayerNorm (``mm.input_norm``, eps 1e-6) per
patch, the 2 x 2 merge (the window's patches (0, 0), (0, 1), (1, 0), (1, 1)) -> mm.0 -> GELU (erf) -> mm.2. In
float64 (patches in raster order), or with ``llama=True`` llama.cpp's choices: a matmul against BF16 weights takes its
activations as bf16.
"""

import math

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig

from ..minicpmv.ref import VisionGolden, _bf16, golden

__all__ = ["Encoder", "VisionGolden", "golden", "rope_angles"]

THETA = 1e6  # (the vision RoPE's base: llama.cpp's vision_rope_theta for lfm3vl)


def rope_angles(ph: int, pw: int, head_dim: int = 64, base: float = THETA) -> np.ndarray:
    """The axial RoPE's angles (ph pw patches in raster order, head_dim / 2) float64: pair i (dims i, i + head_dim / 2)
    turns by y base^(-4 i / head_dim) for i < head_dim / 4, by x base^(-4 (i - head_dim / 4) / head_dim) above."""
    q = head_dim // 4
    f = base ** (-np.arange(q, dtype=np.float64) / q)
    y, x = np.divmod(np.arange(ph * pw), pw)
    return np.concatenate([y[:, None] * f, x[:, None] * f], 1)


def windows(ph: int, pw: int) -> np.ndarray:
    """The 2 x 2 merge's gather indices (raster order of the ph x pw grid): per window (raster order of the merged
    grid) its patches (0, 0), (0, 1), (1, 0), (1, 1) -> (ph pw / 4, 4)."""
    i, j = np.meshgrid(np.arange(ph // 2), np.arange(pw // 2), indexing="ij")
    return np.stack([((2 * i + r) * pw + (2 * j + c)).reshape(-1) for r in (0, 1) for c in (0, 1)], 1)


class Encoder:
    """LFM3-VL's vision encoder from its mmproj file (module docstring); ``llama``: llama.cpp's CPU rounding."""

    def __init__(self, mmproj: str, *, llama: bool = False):
        self.g = g = GGUFFile(mmproj)
        self.cfg = cfg = ClipConfig.from_gguf(g)
        assert cfg.projector == "lfm3vl", cfg.projector
        self.hidden, self.heads, self.kv_heads, self.hd = cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim
        self.layers, self.patch, self.eps = cfg.layers, cfg.patch_size, cfg.eps
        self.llama = llama
        self.dtype = torch.float32 if llama else torch.float64
        self._w: dict[str, torch.Tensor] = {}

    def w(self, name: str) -> torch.Tensor:
        """A tensor as (rows, cols) of self.dtype (cached)."""
        if name not in self._w:
            t = self.g.tensor(name)
            self._w[name] = torch.from_numpy(t.dequantize().reshape(t.rows, t.cols)).to(self.dtype)
        return self._w[name]

    def vec(self, name: str) -> torch.Tensor:
        return self.w(name).reshape(-1)

    def is_conv(self, il: int) -> bool:
        return f"v.blk.{il}.conv2d_in_proj.weight" in self.g

    def _mm(self, x: torch.Tensor, name: str, bias: str | None = None) -> torch.Tensor:
        w = self.w(name)
        if self.llama and self.g.tensor_info[name][1] == "BF16":  # (ggml's BF16 dot products: bf16 activations)
            x = _bf16(x)
        y = x @ w.T
        return y + self.vec(bias) if bias else y

    def _rms(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.vec(name)

    def _rope(self, x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """x (heads, n, d) turned by ang (n, d / 2): dims i, i + d / 2 a pair (rotate_half)."""
        h = self.hd // 2
        c, s = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
        a, b = x[..., :h], x[..., h:]
        return torch.cat([a * c - b * s, b * c + a * s], -1)

    def _attn(self, x: torch.Tensor, p: str, ang: torch.Tensor) -> torch.Tensor:
        n, hd = x.shape[0], self.hd
        q = self._mm(x, p + "attn_q.weight").reshape(n, self.heads, hd)
        k = self._mm(x, p + "attn_k.weight").reshape(n, self.kv_heads, hd)
        v = self._mm(x, p + "attn_v.weight").reshape(n, self.kv_heads, hd)
        q, k = self._rms(q, p + "attn_q_norm.weight"), self._rms(k, p + "attn_k_norm.weight")
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        q, k = self._rope(q, ang), self._rope(k, ang)
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
        s = q @ k.transpose(1, 2) / math.sqrt(hd)
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(n, self.heads * hd)
        return self._mm(o, p + "attn_out.weight")

    def conv(self, g: torch.Tensor, p: str, ph: int, pw: int) -> torch.Tensor:
        """The depthwise 3 x 3 convolution (zero padding) of g (ph pw, hidden) over the patch grid (raster order)."""
        w = self.w(p + "conv2d_spatial.weight").reshape(self.hidden, 1, 3, 3)
        img = g.T.reshape(1, self.hidden, ph, pw)
        out = torch.nn.functional.conv2d(img, w, padding=1, groups=self.hidden)
        return out.reshape(self.hidden, ph * pw).T

    def _conv(self, x: torch.Tensor, p: str, ph: int, pw: int) -> torch.Tensor:
        H = self.hidden
        bcx = self._mm(x, p + "conv2d_in_proj.weight")
        b, c, xx = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
        return self._mm(c * self.conv(b * xx, p, ph, pw), p + "conv2d_out_proj.weight")

    def _layer(self, x: torch.Tensor, il: int, ang: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        p = f"v.blk.{il}."
        h = self._rms(x, p + "ln1.weight")
        x = x + (self._conv(h, p, ph, pw) if self.is_conv(il) else self._attn(h, p, ang))
        h = self._rms(x, p + "ln2.weight")
        g = self._mm(h, p + "ffn_gate.weight")
        h = self._mm(g * torch.sigmoid(g) * self._mm(h, p + "ffn_up.weight"), p + "ffn_down.weight")
        return x + h

    def embed(self, pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
        """The patchifier of pixels (normalized, llama.cpp's inp_raw), RMS-normed: (ph pw, hidden) in raster order. An
        image (3, H, W) takes both temporal kernels, a video's pair of frames (2, 3, H, W) kernel t on frame t."""
        P, H = self.patch, self.hidden
        x = torch.as_tensor(np.asarray(pixels), dtype=self.dtype)
        k0, k1 = (self.w(n).reshape(H, 3, P, P) for n in ("v.patch_embd.weight", "v.patch_embd.weight.1"))
        conv = lambda img, k: torch.nn.functional.conv2d(img[None], k, stride=P)[0]
        e = conv(x, k0 + k1) if x.ndim == 3 else conv(x[0], k0) + conv(x[1], k1)
        return self._rms(e.reshape(H, -1).T, "v.embedding_norm.weight")

    def forward(self, pixels: np.ndarray | torch.Tensor, trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Pixels (3, H, W) of an image or (2, 3, H, W) of a video's pair (normalized) -> the embeddings (ph pw / 4,
        projection_dim), the merged grid in raster order. ``trace``: the intermediates ("embedding_norm",
        "layer_out-<il>", "proj_input_norm" in raster order of the patches, "merged" (windows, 4 hidden))."""
        P = self.patch
        ph, pw = np.asarray(pixels).shape[-2] // P, np.asarray(pixels).shape[-1] // P
        tr = trace if trace is not None else {}
        x = self.embed(pixels)
        tr["embedding_norm"] = x
        ang = torch.from_numpy(rope_angles(ph, pw, self.hd))
        for il in range(self.layers):
            x = self._layer(x, il, ang, ph, pw)
            tr[f"layer_out-{il}"] = x
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + 1e-6) * self.vec("mm.input_norm.weight") + self.vec("mm.input_norm.bias")
        tr["proj_input_norm"] = x
        m = x[torch.from_numpy(windows(ph, pw))].reshape(-1, 4 * self.hidden)
        tr["merged"] = m
        h = self._mm(m, "mm.0.weight", "mm.0.bias")
        h = 0.5 * h * (1 + torch.erf(h / math.sqrt(2)))
        return self._mm(h, "mm.2.weight", "mm.2.bias")
