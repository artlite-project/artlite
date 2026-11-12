"""References of Qwen3-VL's vision encoder (the mmproj of projector ``qwen3vl_merger``: Qwen3.5 / 3.6 / 3.8's): the
encoder in torch, and llama.cpp goldens through libmtmd (``tests.minicpmv.ref.golden``: the dump tool runs any
mmproj; its embeddings ``embd(0)`` are the image chunk's rows, the merged tokens in raster order).

The encoder (``Encoder``): llama.cpp's ``clip_graph_qwen3vl`` (``tools/mtmd/models/qwen3vl.cpp``) and transformers'
``Qwen3VLVisionModel`` on the mmproj's tensors for an image's pixels (3, H, W): the patch embedding (a Conv3d of two
temporal kernels on a still image: both kernels on the same pixels, summed), the learned 48 x 48 position embedding
interpolated bilinearly (align corners) to the patch grid, the ViT layers (LayerNorm with bias, a fused q | k | v
projection, the 2-D RoPE of the vision mode: dims i, i + 36 of a head turned by y s^i for i < 18 and by x s^(i - 18)
for 18 <= i < 36, s = 10000^(-1/18); 16 heads of 72, GELU tanh), post_ln per patch, the 2 x 2 merge (the window's
patches (0, 0), (0, 1), (1, 0), (1, 1)) -> mm.0 -> GELU (transformers' ``nn.GELU``: erf; llama.cpp: tanh) -> mm.2.
In float64 (patches in raster order: the attention does not depend on the tokens' order, only the merge does), or
with ``llama=True`` llama.cpp's choices: a matmul against BF16 weights takes its activations as bf16, GELU tanh
through ggml's fp16 table, the merger's GELU tanh too.
"""

import math

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig

from ..minicpmv.ref import VisionGolden, _bf16, _gelu_table, _gelu_tanh, golden

__all__ = ["Encoder", "NpuEncoder", "VisionGolden", "golden", "pos_rows", "rope_angles"]


def rope_angles(ph: int, pw: int, head_dim: int = 72, base: float = 10000.0) -> np.ndarray:
    """The 2-D RoPE's angles (ph pw patches in raster order, head_dim / 2) float64: pair i (dims i, i + head_dim / 2)
    turns by y base^(-4 i / head_dim) for i < head_dim / 4, by x base^(-4 (i - head_dim / 4) / head_dim) above."""
    q = head_dim // 4
    f = base ** (-np.arange(q, dtype=np.float64) / q)
    y, x = np.divmod(np.arange(ph * pw), pw)
    return np.concatenate([y[:, None] * f, x[:, None] * f], 1)


def pos_rows(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """The learned position embedding (n x n rows of hidden) at a ph x pw patch grid (raster order): bilinear, align
    corners (the grid's corners at the table's), float64."""
    n = round(math.sqrt(table.shape[0]))
    t = torch.as_tensor(np.asarray(table, np.float64)).reshape(n, n, -1).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(t, size=(ph, pw), mode="bilinear", align_corners=True)[0]
    return out.permute(1, 2, 0).reshape(ph * pw, -1).numpy()


class Encoder:
    """Qwen3-VL's vision encoder from its mmproj file (module docstring); ``llama``: llama.cpp's CPU rounding."""

    def __init__(self, mmproj: str, *, llama: bool = False):
        self.g = g = GGUFFile(mmproj)
        self.cfg = cfg = ClipConfig.from_gguf(g)
        assert cfg.projector == "qwen3vl_merger", cfg.projector
        self.hidden, self.heads, self.hd = cfg.hidden, cfg.heads, cfg.head_dim
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

    def _mm(self, x: torch.Tensor, name: str, bias: str | None = None) -> torch.Tensor:
        w = self.w(name)
        if self.llama and self.g.tensor_info[name][1] == "BF16":  # (ggml's BF16 dot products: bf16 activations)
            x = _bf16(x)
        y = x @ w.T
        return y + self.w(bias).reshape(-1) if bias else y

    def _ln(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.w(prefix + ".weight").reshape(-1) \
            + self.w(prefix + ".bias").reshape(-1)

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_table(x) if self.llama else _gelu_tanh(x)

    def _rope(self, x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """x (heads, n, d) turned by ang (n, d / 2): dims i, i + d / 2 a pair (rotate_half)."""
        h = self.hd // 2
        c, s = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
        a, b = x[..., :h], x[..., h:]
        return torch.cat([a * c - b * s, b * c + a * s], -1)

    def _attn(self, x: torch.Tensor, p: str, ang: torch.Tensor) -> torch.Tensor:
        n, H = x.shape[0], self.hidden
        qkv = self._mm(x, p + "attn_qkv.weight", p + "attn_qkv.bias")
        q, k, v = (qkv[:, i * H:(i + 1) * H].reshape(n, self.heads, self.hd).transpose(0, 1) for i in range(3))
        q, k = self._rope(q, ang), self._rope(k, ang)
        s = q @ k.transpose(1, 2) / math.sqrt(self.hd)
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(n, H)
        return self._mm(o, p + "attn_out.weight", p + "attn_out.bias")

    def _layer(self, x: torch.Tensor, il: int, ang: torch.Tensor) -> torch.Tensor:
        p = f"v.blk.{il}."
        x = x + self._attn(self._ln(x, p + "ln1"), p, ang)
        h = self._ln(x, p + "ln2")
        h = self._mm(self._gelu(self._mm(h, p + "ffn_up.weight", p + "ffn_up.bias")), p + "ffn_down.weight",
                     p + "ffn_down.bias")
        return x + h

    @staticmethod
    def windows(ph: int, pw: int) -> np.ndarray:
        """The 2 x 2 merge's gather indices (raster order of the ph x pw grid): per window (raster order of the merged
        grid) its patches (0, 0), (0, 1), (1, 0), (1, 1) -> (ph pw / 4, 4)."""
        i, j = np.meshgrid(np.arange(ph // 2), np.arange(pw // 2), indexing="ij")
        return np.stack([((2 * i + r) * pw + (2 * j + c)).reshape(-1) for r in (0, 1) for c in (0, 1)], 1)

    def embed(self, pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
        """The patch embedding of pixels (3, H, W) (normalized, llama.cpp's inp_raw) plus the learned positions:
        (ph pw, hidden) in raster order."""
        P, H = self.patch, self.hidden
        x = torch.as_tensor(np.asarray(pixels), dtype=self.dtype)
        ph, pw = x.shape[1] // P, x.shape[2] // P
        kern = (self.w("v.patch_embd.weight") + self.w("v.patch_embd.weight.1")).reshape(H, 3, P, P)
        e = torch.nn.functional.conv2d(x[None], kern, stride=P)[0]
        x = e.reshape(H, -1).T + self.w("v.patch_embd.bias").reshape(-1)
        return x + torch.from_numpy(pos_rows(self.w("v.position_embd.weight").numpy(), ph, pw)).to(self.dtype)

    def forward(self, pixels: np.ndarray | torch.Tensor, trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Pixels (3, H, W) (normalized) -> the image's embeddings (ph pw / 4, projection_dim), the merged grid in
        raster order. ``trace``: the intermediates ("inp_pos_emb", "layer_out-<il>", "post_ln" in raster order of
        the patches, "merged" (windows, 4 hidden))."""
        P = self.patch
        ph, pw = np.asarray(pixels).shape[1] // P, np.asarray(pixels).shape[2] // P
        tr = trace if trace is not None else {}
        x = self.embed(pixels)
        tr["inp_pos_emb"] = x
        ang = torch.from_numpy(rope_angles(ph, pw, self.hd))
        for il in range(self.layers):
            x = self._layer(x, il, ang)
            tr[f"layer_out-{il}"] = x
        x = self._ln(x, "v.post_ln")
        tr["post_ln"] = x
        m = x[torch.from_numpy(self.windows(ph, pw))].reshape(-1, 4 * self.hidden)
        tr["merged"] = m
        h = self._mm(m, "mm.0.weight", "mm.0.bias")
        h = self._gelu(h) if self.llama else 0.5 * h * (1 + torch.erf(h / math.sqrt(2)))
        return self._mm(h, "mm.2.weight", "mm.2.bias")


def _bfp(x: torch.Tensor) -> torch.Tensor:
    """bfp16ebs8 blocks of 8 along the last axis (RNE)."""
    from artlite.operators import bfp16
    n = x.shape[-1]
    xp = torch.nn.functional.pad(x, (0, -n % 8)) if n % 8 else x
    return torch.from_numpy(bfp16.values(xp.to(torch.float32).numpy())).to(x.dtype)[..., :n]


class NpuEncoder(Encoder):
    """The encoder at the rounding points of the vision overlay (``models.qwen3vl.encoder``; float64 between them): a
    GEMM's A as bfp16 panels and its weights bfp16, C over the whole K from the bf16 bias plus a bf16 residual, rounded
    once (bf16 C tiles: the residual stream; bfp16 panels: a GEMM's or the attention's input); Q and K rounded to bf16,
    turned by bf16 cos / sin (the products summed in fp32, rounded to bf16), then bfp16 per head's 8 dims of the
    epilogue's layout (dims 0 .. d / 2 - 1 | d / 2 .. d - 1, each padded to 40); LayerNorm statistics S1, S2 of the bf16 values
    in fp32; GELU of bf16 values to bf16; attention with P = 2^(s - max) in bf16 as bfp16 panels, l their sums, o / l as
    panels. The keys' blocks of 8 (P's and V^T's) are the NPU's key panels: ``vblocks``, per block its 8 patches (-1:
    a padding slot). ``forward``: the patch embedding to bf16, every layer, post_ln's panels, the merge and the merger
    (hi + lo weights, GELU erf of bf16 values to bf16 panels, bf16 out)."""

    def __init__(self, mmproj: str):
        super().__init__(mmproj)
        self._wq: dict[str, torch.Tensor] = {}

    def wq(self, name: str) -> torch.Tensor:
        if name not in self._wq:
            self._wq[name] = _bfp(self.w(name))
        return self._wq[name]

    def gemm(self, a: torch.Tensor, name: str, bias: str | None, res: torch.Tensor | None = None,
             rows: slice = slice(None)) -> torch.Tensor:
        """a: the A panels' values -> C + bias + residual of the weight rows ``rows``, unrounded."""
        c = a @ self.wq(name)[rows].T
        if bias:
            c = c + _bf16(self.w(bias).reshape(-1)[rows])
        return c if res is None else c + res

    def ln(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        """x: bf16 values -> the LayerNorm's bfp16 panels."""
        x32 = x.to(torch.float32)
        n = x.shape[-1]
        mean = x32.sum(-1, keepdim=True) / n
        var = torch.clamp((x32 * x32).sum(-1, keepdim=True) / n - mean * mean, min=0)
        rstd = 1 / torch.sqrt(var + self.eps)
        y = (x - mean.to(x.dtype)) * rstd.to(x.dtype) * self.w(prefix + ".weight").reshape(-1) \
            + self.w(prefix + ".bias").reshape(-1)
        return _bfp(y)

    def _turn(self, c: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """C (n, heads d) of Q or K -> the RoPE epilogue's panels' values (heads, n, d)."""
        n, h = c.shape[0], self.hd // 2
        x = _bf16(c).reshape(n, self.heads, 2, h).transpose(0, 1)
        cs, sn = (_bf16(f(ang)).to(c.dtype) for f in (torch.cos, torch.sin))
        a, b = x[:, :, 0], x[:, :, 1]
        out = torch.stack([_bf16(a * cs - b * sn), _bf16(b * cs + a * sn)], 2)  # (heads, n, 2, d / 2)
        out = torch.nn.functional.pad(out, (0, 40 - h))  # (the epilogue's blocks: d / 2 + zeros to 40 per half)
        return _bfp(out).reshape(self.heads, n, 2, 40)[..., :h].reshape(self.heads, n, self.hd)

    def attention(self, a: torch.Tensor, p: str, ang: torch.Tensor, vblocks: np.ndarray) -> torch.Tensor:
        """a: the LayerNorm's panels (n, hidden) -> o / l panels (n, hidden); vblocks: the tokens of each V^T block of 8
        (padding: -1)."""
        n, H, d = a.shape[0], self.hidden, self.hd
        w = p + "attn_qkv.weight"
        b = p + "attn_qkv.bias"
        q = self._turn(self.gemm(a, w, b, rows=slice(0, H)) * (math.log2(math.e) / math.sqrt(d)), ang)
        k = self._turn(self.gemm(a, w, b, rows=slice(H, 2 * H)), ang)
        v = self.gemm(a, w, b, rows=slice(2 * H, 3 * H)).reshape(n, self.heads, d).permute(1, 2, 0)  # (h, d, n)
        ko = torch.from_numpy(vblocks.reshape(-1))  # (the keys in the NPU's blocks of 8; padding: -1)
        ok = ko >= 0
        vq = _bfp(torch.cat([v, torch.zeros_like(v[..., :1])], -1)[..., ko])  # (h, d, slots)
        s = q @ k[:, ko[ok]].transpose(1, 2)  # (log2 units)
        e = torch.zeros(self.heads, n, len(ko), dtype=s.dtype)
        e[..., ok] = torch.exp2(s - s.amax(-1, keepdim=True))
        pr = _bfp(_bf16(e))
        o = (pr @ vq.transpose(1, 2)) / pr.sum(-1, keepdim=True)
        return _bfp(o.transpose(0, 1).reshape(n, H))

    def layer(self, x: torch.Tensor, il: int, ang: torch.Tensor, vblocks: np.ndarray) -> torch.Tensor:
        """Layer il on the residual x (bf16 values)."""
        p = f"v.blk.{il}."
        o = self.attention(self.ln(x, p + "ln1"), p, ang, vblocks)
        x = _bf16(self.gemm(o, p + "attn_out.weight", p + "attn_out.bias", x))
        h = self.gemm(self.ln(x, p + "ln2"), p + "ffn_up.weight", p + "ffn_up.bias")
        h = _bfp(_bf16(_gelu_tanh(_bf16(h))))
        return _bf16(self.gemm(h, p + "ffn_down.weight", p + "ffn_down.bias", x))

    def _hilo(self, name: str) -> torch.Tensor:
        w = self.w(name)
        hi = _bfp(w)
        return hi + _bfp(w - hi)

    def forward(self, pixels: np.ndarray | torch.Tensor, vblocks: np.ndarray,  # type: ignore[override]
                trace: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Pixels (3, H, W) (normalized) -> the embeddings at the NPU's rounding points (class docstring); vblocks: the
        NPU's key panels (patches in raster order). ``trace``: "layer_out-<il>" (each layer's weights dropped after it:
        the float64 copies of all 27 take ~4 GB)."""
        P = self.patch
        ph, pw = np.asarray(pixels).shape[1] // P, np.asarray(pixels).shape[2] // P
        tr = trace if trace is not None else {}
        ang = torch.from_numpy(rope_angles(ph, pw, self.hd))
        x = _bf16(self.embed(pixels))
        for il in range(self.layers):
            x = self.layer(x, il, ang, vblocks)
            tr[f"layer_out-{il}"] = x
            for cache in (self._w, self._wq):
                for k in [k for k in cache if k.startswith(f"v.blk.{il}.")]:
                    del cache[k]
        m = self.ln(x, "v.post_ln")[torch.from_numpy(self.windows(ph, pw))].reshape(-1, 4 * self.hidden)
        h = _bf16(m @ self._hilo("mm.0.weight").T + _bf16(self.w("mm.0.bias").reshape(-1)))
        h = _bfp(_bf16(0.5 * h * (1 + torch.erf(h / math.sqrt(2)))))
        return _bf16(h @ self._hilo("mm.2.weight").T + _bf16(self.w("mm.2.bias").reshape(-1)))
