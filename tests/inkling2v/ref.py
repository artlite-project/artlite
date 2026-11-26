"""References of Inkling 2's vision encoder (the mmproj of projector ``inkling2v``): the encoder in torch, and llama.cpp
goldens through libmtmd (``tests.goldens.media.golden``: its embeddings ``embd(0)`` are the image chunk's rows, the
merged tokens in raster order).

The encoder (``Encoder``): llama.cpp's ``mtmd_graph_inkling2v`` (``tools/mtmd/models/inkling2v.cpp``, ``build_vit``) and
transformers' ``Inkling2VisionModel`` on the mmproj's tensors for an image's pixels (3, H, W) (``pixels``: padded to
multiples of 40 with -1 on the 0 .. 255 scale, normalized by 0.5 / 0.5): the patch embedding (a conv of 20 x 20, no
bias), the learned 32 x 32 position embedding interpolated bilinearly (corners not aligned: ``pos_rows``), a class token
(``class_embd`` + ``class_position_embd``) before the patches, pre_ln, 23 layers (LayerNorm with bias, q / k / v with
biases, the 2-D RoPE of interleaved pairs: a head's pairs (2 i, 2 i + 1) for i < 16 turned by (x + 1) s^i, the others
by (y + 1) s^(i - 16), s = 10000^(-1 / 16), the class token at (0, 0); 16 heads of 64, the attention over every token,
its output and the MLP's (GELU erf) times the layer scales ``ls1`` / ``ls2``), the class token dropped, each 2 x 2
window's patches (0, 0), (0, 1), (1, 0), (1, 1) merged (4 hidden) -> mm.1 -> GELU (erf) -> mm.2. In float64 (tokens:
the class token, then the patches in raster order).
"""

import math

import numpy as np
import torch

from artlite.hf_model import open_mmproj
from artlite.model_config import ClipConfig
from models.inkling2v.host import PAD, TOKEN_PX

from ..goldens.media import VisionGolden, golden
from ..numerics import bf16_like

__all__ = ["Encoder", "NpuEncoder", "VisionGolden", "golden", "pixels", "pos_rows", "rope_angles"]


def pixels(px: np.ndarray) -> np.ndarray:
    """uint8 pixels (H, W, 3) -> the encoder's input (3, H', W') float32 padded to multiples of 40 (llama.cpp's
    inp_raw: (p - 127.5) / 127.5, the padding p = -1)."""
    h, w = px.shape[:2]
    out = np.full((3, -(-h // TOKEN_PX) * TOKEN_PX, -(-w // TOKEN_PX) * TOKEN_PX), PAD, np.float32)
    out[:, :h, :w] = px.transpose(2, 0, 1)
    return (out - np.float32(127.5)) / np.float32(127.5)


def rope_angles(ph: int, pw: int, head_dim: int = 64, base: float = 10000.0) -> np.ndarray:
    """The 2-D RoPE's angles (the class token, then ph pw patches in raster order; head_dim / 2 pairs) float64: pair i
    (dims 2 i, 2 i + 1) turns by (x + 1) base^(-4 i / head_dim) for i < head_dim / 4, by (y + 1) base^(-4 (i -
    head_dim / 4) / head_dim) above; the class token by 0."""
    q = head_dim // 4
    f = base ** (-np.arange(q, dtype=np.float64) / q)
    y, x = np.divmod(np.arange(ph * pw), pw)
    ang = np.concatenate([(x + 1)[:, None] * f, (y + 1)[:, None] * f], 1)
    return np.concatenate([np.zeros((1, 2 * q)), ang])


def pos_rows(table: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """The learned position embedding (n x n rows of hidden) at a ph x pw patch grid (raster order): bilinear, the
    corners not aligned (torch's ``align_corners=False``, llama.cpp's GGML_SCALE_MODE_BILINEAR), float64."""
    n = round(math.sqrt(table.shape[0]))
    t = torch.as_tensor(np.asarray(table, np.float64)).reshape(n, n, -1).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(t, size=(ph, pw), mode="bilinear", align_corners=False)[0]
    return out.permute(1, 2, 0).reshape(ph * pw, -1).numpy()


class Encoder:
    """Inkling 2's vision encoder from its mmproj file (module docstring)."""

    def __init__(self, mmproj: str):
        self.g = g = open_mmproj(mmproj)
        self.cfg = cfg = ClipConfig.from_gguf(g)
        assert cfg.projector == "inkling2v", cfg.projector
        self.hidden, self.heads, self.hd = cfg.hidden, cfg.heads, cfg.head_dim
        self.layers, self.patch, self.eps = cfg.layers, cfg.patch_size, cfg.eps
        self.dtype = torch.float64
        self._w: dict[str, torch.Tensor] = {}

    def w(self, name: str) -> torch.Tensor:
        """A tensor as (rows, cols) of self.dtype (cached)."""
        if name not in self._w:
            t = self.g.tensor(name)
            self._w[name] = torch.from_numpy(t.dequantize().reshape(t.rows, t.cols)).to(self.dtype)
        return self._w[name]

    def vec(self, name: str) -> torch.Tensor:
        return self.w(name).reshape(-1)

    def _mm(self, x: torch.Tensor, name: str, bias: str | None = None) -> torch.Tensor:
        y = x @ self.w(name).T
        return y + self.vec(bias) if bias else y

    def _ln(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.vec(prefix + ".weight") + self.vec(prefix + ".bias")

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

    def _rope(self, x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """x (heads, n, d) turned by ang (n, d / 2): dims 2 i, 2 i + 1 a pair."""
        c, s = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack([a * c - b * s, b * c + a * s], -1).flatten(-2)

    def _attn(self, x: torch.Tensor, p: str, ang: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = (self._mm(x, p + f"attn_{r}.weight", p + f"attn_{r}.bias").reshape(n, self.heads, self.hd)
                   .transpose(0, 1) for r in "qkv")
        q, k = self._rope(q, ang), self._rope(k, ang)
        s = q @ k.transpose(1, 2) / math.sqrt(self.hd)
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(n, self.hidden)
        return self._mm(o, p + "attn_out.weight", p + "attn_out.bias")

    def _layer(self, x: torch.Tensor, il: int, ang: torch.Tensor) -> torch.Tensor:
        p = f"v.blk.{il}."
        x = x + self._attn(self._ln(x, p + "ln1"), p, ang) * self.vec(p + "ls1.weight")
        h = self._ln(x, p + "ln2")
        h = self._mm(self._gelu(self._mm(h, p + "ffn_up.weight", p + "ffn_up.bias")), p + "ffn_down.weight",
                     p + "ffn_down.bias")
        return x + h * self.vec(p + "ls2.weight")

    @staticmethod
    def windows(ph: int, pw: int) -> np.ndarray:
        """The 2 x 2 merge's gather indices (raster order of the ph x pw grid): per window (raster order of the merged
        grid) its patches (0, 0), (0, 1), (1, 0), (1, 1) -> (ph pw / 4, 4)."""
        i, j = np.meshgrid(np.arange(ph // 2), np.arange(pw // 2), indexing="ij")
        return np.stack([((2 * i + r) * pw + (2 * j + c)).reshape(-1) for r in (0, 1) for c in (0, 1)], 1)

    def embed(self, pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
        """The class token and the patch embedding of pixels (3, H, W) (normalized, padded: llama.cpp's inp_raw) plus
        the learned positions: (1 + ph pw, hidden), the patches in raster order."""
        P, H = self.patch, self.hidden
        x = torch.as_tensor(np.asarray(pixels), dtype=self.dtype)
        ph, pw = x.shape[1] // P, x.shape[2] // P
        e = torch.nn.functional.conv2d(x[None], self.w("v.patch_embd.weight").reshape(H, 3, P, P), stride=P)[0]
        e = e.reshape(H, -1).T + torch.from_numpy(pos_rows(self.w("v.position_embd.weight").numpy(), ph, pw))
        cls = self.vec("v.class_embd") + self.vec("v.class_position_embd.weight")
        return torch.cat([cls[None], e])

    def forward(self, pixels: np.ndarray | torch.Tensor, trace: dict[str, torch.Tensor] | None = None,
                layers: int | None = None) -> torch.Tensor:
        """Pixels (3, H, W) (normalized, padded) -> the image's embeddings (ph pw / 4, projection_dim), the merged grid in
        raster order. ``trace``: the intermediates ("embeddings", "pre_ln", "layer_out-<il>": the class token first,
        then the patches in raster order; "merged" (windows, 4 hidden)); ``layers``: only the first ones (the trace's
        tensors), None."""
        P = self.patch
        ph, pw = np.asarray(pixels).shape[1] // P, np.asarray(pixels).shape[2] // P
        tr = trace if trace is not None else {}
        x = self.embed(pixels)
        tr["embeddings"] = x
        x = self._ln(x, "v.pre_ln")
        tr["pre_ln"] = x
        ang = torch.from_numpy(rope_angles(ph, pw, self.hd))
        for il in range(self.layers if layers is None else layers):
            x = self._layer(x, il, ang)
            tr[f"layer_out-{il}"] = x
        if layers is not None:
            return x
        m = x[1:][torch.from_numpy(self.windows(ph, pw))].reshape(-1, 4 * self.hidden)
        tr["merged"] = m
        return self._mm(self._gelu(self._mm(m, "mm.1.weight")), "mm.2.weight")


def _bfp(x: torch.Tensor) -> torch.Tensor:
    """bfp16ebs8 blocks of 8 along the last axis (RNE)."""
    from artlite.operators import bfp16
    n = x.shape[-1]
    xp = torch.nn.functional.pad(x, (0, -n % 8)) if n % 8 else x
    return torch.from_numpy(bfp16.values(xp.to(torch.float32).numpy())).to(x.dtype)[..., :n]


def _hilo(w: torch.Tensor) -> torch.Tensor:
    hi = _bfp(w)
    return hi + _bfp(w - hi)


class NpuEncoder(Encoder):
    """The encoder at the rounding points of the vision overlay (``models.inkling2v.encoder``; float64 between them): a
    GEMM's A as bfp16 panels and its weights bfp16 (the layer scales and the queries' scale folded in first), C over
    the whole K from the bf16 bias plus a bf16 residual, rounded once (bf16 C tiles: the residual stream; bfp16 panels:
    a GEMM's or the attention's input); the patch embedding and the merger hi + lo weights on exact pixels / bfp16
    panels of the residual; pre_ln to bf16 values (``x0``: "bfp16", its bfp16 panels instead); Q and K rounded to bf16,
    turned by bf16 cos / sin (the products summed in fp32, rounded to bf16), then bfp16 per 8 pairs' halves (the
    epilogue's layout); LayerNorm statistics of the bf16 values in fp32; GELU of bf16 values to bf16; attention with P
    = 2^(s - max) in bf16 as bfp16 panels, l their sums, o / l as panels. ``vblocks``: the keys' blocks of 8 (P's and
    V^T's: the NPU's key panels), per block its 8 tokens (the class token 0, patch i at 1 + i; -1: a padding slot)."""

    def __init__(self, mmproj: str, x0: str = "bf16"):
        super().__init__(mmproj)
        self.x0 = x0
        self._wq: dict[str, torch.Tensor] = {}

    def wq(self, name: str, scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """bfp16 weights of ``name`` times ``scale`` per row."""
        if name not in self._wq:
            s = scale if isinstance(scale, float) else scale[:, None]
            self._wq[name] = _bfp(self.w(name) * s)
        return self._wq[name]

    def ln(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        """x: bf16 values -> the LayerNorm's values (bf16 twice: x a + b, then times gamma plus beta)."""
        x32 = x.to(torch.float32)
        n = x.shape[-1]
        mean = x32.sum(-1, keepdim=True) / n
        var = torch.clamp((x32 * x32).sum(-1, keepdim=True) / n - mean * mean, min=0)
        rstd = 1 / torch.sqrt(var + self.eps)
        v = bf16_like((x - mean.to(x.dtype)) * rstd.to(x.dtype))
        return bf16_like(v * self.vec(prefix + ".weight") + self.vec(prefix + ".bias"))

    def _turn(self, c: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """C (n, heads d) of Q or K -> the RoPE epilogue's panels' values (heads, n, d)."""
        n, h = c.shape[0], self.hd // 2
        x = bf16_like(c).reshape(n, self.heads, h, 2).transpose(0, 1)
        cs, sn = (bf16_like(f(ang)).to(c.dtype) for f in (torch.cos, torch.sin))
        a, b = x[..., 0], x[..., 1]
        out = torch.stack([bf16_like(a * cs - b * sn), bf16_like(b * cs + a * sn)], 2)  # (heads, n, 2, d / 2)
        out = torch.nn.functional.pad(out, (0, 40 - h))  # (the epilogue's blocks: d / 2 + zeros to 40 per half)
        out = _bfp(out).reshape(self.heads, n, 2, 40)[..., :h]
        return out.transpose(2, 3).reshape(self.heads, n, self.hd)

    def attention(self, a: torch.Tensor, p: str, ang: torch.Tensor, vblocks: np.ndarray) -> torch.Tensor:
        """a: LN1's panels (n, hidden) -> o / l panels (n, hidden)."""
        n, H, d = a.shape[0], self.hidden, self.hd
        qs = math.log2(math.e) / math.sqrt(d)
        g = lambda r, s=1.0: a @ self.wq(p + f"attn_{r}.weight", s).T + bf16_like(self.vec(p + f"attn_{r}.bias") * s)
        q, k = self._turn(g("q", qs), ang), self._turn(g("k"), ang)
        v = g("v").reshape(n, self.heads, d).permute(1, 2, 0)  # (h, d, n)
        ko = torch.from_numpy(vblocks.reshape(-1))  # (the keys in the NPU's blocks of 8; padding: -1)
        ok = ko >= 0
        vq = _bfp(torch.cat([v, torch.zeros_like(v[..., :1])], -1)[..., ko])  # (h, d, slots)
        s = q @ k[:, ko[ok]].transpose(1, 2)  # (log2 units)
        e = torch.zeros(self.heads, n, len(ko), dtype=s.dtype)
        e[..., ok] = torch.exp2(s - s.amax(-1, keepdim=True))
        pr = _bfp(bf16_like(e))
        o = (pr @ vq.transpose(1, 2)) / pr.sum(-1, keepdim=True)
        return _bfp(o.transpose(0, 1).reshape(n, H))

    def gemm(self, a: torch.Tensor, name: str, bias: str, res: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return a @ self.wq(name, scale).T + bf16_like(self.vec(bias) * scale) + res

    def layer(self, x: torch.Tensor, il: int, ang: torch.Tensor, vblocks: np.ndarray) -> torch.Tensor:
        """Layer il on the residual x (bf16 values)."""
        p = f"v.blk.{il}."
        o = self.attention(_bfp(self.ln(x, p + "ln1")), p, ang, vblocks)
        x = bf16_like(self.gemm(o, p + "attn_out.weight", p + "attn_out.bias", x, self.vec(p + "ls1.weight")))
        h = _bfp(self.ln(x, p + "ln2")) @ self.wq(p + "ffn_up.weight").T + bf16_like(self.vec(p + "ffn_up.bias"))
        h = _bfp(bf16_like(self._gelu(bf16_like(h))))
        return bf16_like(self.gemm(h, p + "ffn_down.weight", p + "ffn_down.bias", x, self.vec(p + "ls2.weight")))

    def forward(self, pixels: np.ndarray | torch.Tensor, vblocks: np.ndarray,  # type: ignore[override]
                trace: dict[str, torch.Tensor] | None = None, layers: int | None = None) -> torch.Tensor:
        """Pixels (3, H, W) (normalized, padded) -> the embeddings at the NPU's rounding points (class docstring);
        ``trace``: "pre_ln", "layer_out-<il>" (each layer's weights dropped after it); ``layers``: only the first ones
        (the trace's tensors), None."""
        P, H = self.patch, self.hidden
        ph, pw = np.asarray(pixels).shape[1] // P, np.asarray(pixels).shape[2] // P
        tr = trace if trace is not None else {}
        ang = torch.from_numpy(rope_angles(ph, pw, self.hd))
        x = torch.as_tensor(np.asarray(pixels), dtype=self.dtype)
        e = torch.nn.functional.conv2d(x[None], _hilo(self.w("v.patch_embd.weight")).reshape(H, 3, P, P), stride=P)[0]
        pos = torch.from_numpy(pos_rows(self.w("v.position_embd.weight").numpy(), ph, pw))
        cls = self.vec("v.class_embd") + self.vec("v.class_position_embd.weight")
        x = bf16_like(torch.cat([bf16_like(cls)[None], e.reshape(H, -1).T + bf16_like(pos)]))
        x = self.ln(x, "v.pre_ln")
        if self.x0 == "bfp16":
            x = _bfp(x)
        tr["pre_ln"] = x
        for il in range(self.layers if layers is None else layers):
            x = self.layer(x, il, ang, vblocks)
            tr[f"layer_out-{il}"] = x
            for cache in (self._w, self._wq):
                for k in [k for k in cache if k.startswith(f"v.blk.{il}.")]:
                    del cache[k]
        if layers is not None:
            return x
        m = _bfp(x[1:])[torch.from_numpy(self.windows(ph, pw))].reshape(-1, 4 * H)
        h = bf16_like(m @ _hilo(self.w("mm.1.weight")).T)
        h = _bfp(bf16_like(self._gelu(h)))
        return bf16_like(h @ _hilo(self.w("mm.2.weight")).T)
