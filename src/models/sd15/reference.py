"""Stable Diffusion 1.5's networks in fp32 torch, from a diffusers directory's safetensors: the CLIP text encoder
(transformers' ``CLIPTextModel``), the UNet (diffusers' ``UNet2DConditionModel``) and the VAE decoder (the decoding
half of ``AutoencoderKL``), with SD 1.x's configuration. The CPU backend, and the reference of the NPU's parts.

- Text encoder: 12 pre-norm transformer layers (768 wide, 12 heads, causal, quick-GELU MLP of 3072) over token and
  position embeddings, then the final layer norm.
- UNet on 64 x 64 latents: conv_in (4 -> 320); down blocks of 320 / 640 / 1280 / 1280 channels, each 2 resnets
  (the first three with a spatial transformer after each resnet) and a stride-2 conv (not the last); the middle
  block (resnet, transformer, resnet); up blocks mirroring them with 3 resnets each, the input concatenated with the
  matching down activation, and a 2x nearest upsampling + conv (not the last); GroupNorm(32) + SiLU, conv_out
  (320 -> 4). A resnet: GroupNorm + SiLU, 3x3 conv, + its time bias (``host``), GroupNorm + SiLU, 3x3 conv, plus
  the input (through a 1x1 conv where the channels change). A spatial transformer: GroupNorm, 1x1 conv, then on
  the pixels as tokens: self-attention, cross-attention on the text (K, V from the 77 x 768 context), GEGLU
  feed-forward (each pre-LayerNorm with a residual, 8 heads), 1x1 conv, plus the input.
- VAE decoder: latents / 0.18215, post_quant_conv (1x1), conv_in (4 -> 512), the middle block (resnet,
  single-head self-attention on the 64 x 64 pixels, resnet), up blocks of 512 / 512 / 256 / 128 channels with 3
  resnets each and a 2x upsampling + conv (not the last), GroupNorm + SiLU, conv_out (128 -> 3): 512 x 512 pixels.
"""

import json
import os

import torch
import torch.nn.functional as F

from .host import resnet_time_biases
from .weights import load_safetensors

Weights = dict[str, torch.Tensor]


def _config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def conv(w: Weights, p: str, x: torch.Tensor, *, stride: int = 1) -> torch.Tensor:
    """The convolution ``p`` (weight, bias): k x k, padding k // 2."""
    k = w[p + "weight"]
    return F.conv2d(x, k, w[p + "bias"], stride=stride, padding=k.shape[-1] // 2)


def linear(w: Weights, p: str, x: torch.Tensor) -> torch.Tensor:
    """The linear layer ``p`` (weight as 2-D or 1x1 conv, optional bias)."""
    return F.linear(x, w[p + "weight"].flatten(1), w.get(p + "bias"))


def group_norm(w: Weights, p: str, x: torch.Tensor, eps: float, *, silu: bool = False) -> torch.Tensor:
    y = F.group_norm(x, 32, w[p + "weight"], w[p + "bias"], eps)
    return F.silu(y) if silu else y


def layer_norm(w: Weights, p: str, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), w[p + "weight"], w[p + "bias"], eps)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, *, causal: bool = False) -> torch.Tensor:
    """Multi-head attention of q (B, N, C) on k, v (B, M, C), scaled by head_dim^-1/2."""
    b, n, c = q.shape
    q, k, v = (t.view(b, -1, heads, c // heads).transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(b, n, c)


def resnet(w: Weights, p: str, x: torch.Tensor, *, eps: float, time_bias: torch.Tensor | None = None) -> torch.Tensor:
    h = conv(w, p + "conv1.", group_norm(w, p + "norm1.", x, eps, silu=True))
    if time_bias is not None:
        h = h + time_bias[:, None, None]
    h = conv(w, p + "conv2.", group_norm(w, p + "norm2.", h, eps, silu=True))
    if p + "conv_shortcut.weight" in w:
        x = conv(w, p + "conv_shortcut.", x)
    return x + h


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    b, c, hh, ww = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, hh * ww, c)


def _to_pixels(h: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    b, c, hh, ww = like.shape
    return h.reshape(b, hh, ww, c).permute(0, 3, 1, 2)


class ClipText:
    """The CLIP text encoder: ``self(ids)``, ids (B, 77) int64 -> last hidden states (B, 77, 768)."""

    def __init__(self, directory: str):
        self.cfg = _config(os.path.join(directory, "config.json"))
        self.w = load_safetensors(os.path.join(directory, "model.safetensors"))

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        w, heads, pre = self.w, self.cfg["num_attention_heads"], "text_model."
        h = w[pre + "embeddings.token_embedding.weight"][ids] + w[pre + "embeddings.position_embedding.weight"][None]
        for i in range(self.cfg["num_hidden_layers"]):
            p = f"{pre}encoder.layers.{i}."
            x = layer_norm(w, p + "layer_norm1.", h)
            a = p + "self_attn."
            o = attention(linear(w, a + "q_proj.", x), linear(w, a + "k_proj.", x), linear(w, a + "v_proj.", x), heads,
                          causal=True)
            h = h + linear(w, a + "out_proj.", o)
            x = linear(w, p + "mlp.fc1.", layer_norm(w, p + "layer_norm2.", h))
            h = h + linear(w, p + "mlp.fc2.", x * torch.sigmoid(1.702 * x))
        return layer_norm(w, pre + "final_layer_norm.", h)


class UNet:
    """The UNet: ``self(x, t, context)``, latents x (B, 4, 64, 64) (scaled to unit variance), a fractional timestep,
    the text context (B, 77, 768) -> the predicted noise (B, 4, 64, 64)."""

    def __init__(self, directory: str):
        self.cfg = _config(os.path.join(directory, "config.json"))
        self.w = load_safetensors(os.path.join(directory, "diffusion_pytorch_model.safetensors"))

    def transformer(self, p: str, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        w, heads, q = self.w, self.cfg["attention_head_dim"], p + "transformer_blocks.0."  # (SD 1.x: the heads)
        h = _to_tokens(conv(w, p + "proj_in.", group_norm(w, p + "norm.", x, 1e-6)))
        n = layer_norm(w, q + "norm1.", h)
        a = q + "attn1."
        o = attention(linear(w, a + "to_q.", n), linear(w, a + "to_k.", n), linear(w, a + "to_v.", n), heads)
        h = h + linear(w, a + "to_out.0.", o)
        n = layer_norm(w, q + "norm2.", h)
        a = q + "attn2."
        o = attention(linear(w, a + "to_q.", n), linear(w, a + "to_k.", context), linear(w, a + "to_v.", context),
                      heads)
        h = h + linear(w, a + "to_out.0.", o)
        hidden, gate = linear(w, q + "ff.net.0.proj.", layer_norm(w, q + "norm3.", h)).chunk(2, dim=-1)
        h = h + linear(w, q + "ff.net.2.", hidden * F.gelu(gate))
        return conv(w, p + "proj_out.", _to_pixels(h, x)) + x

    def __call__(self, x: torch.Tensor, t: float, context: torch.Tensor) -> torch.Tensor:
        w, cfg, eps = self.w, self.cfg, self.cfg["norm_eps"]
        biases = resnet_time_biases(w, t)

        def res(p: str, h: torch.Tensor) -> torch.Tensor:
            return resnet(w, p, h, eps=eps, time_bias=biases[p])

        h = conv(w, "conv_in.", x)
        skips = [h]
        last = len(cfg["down_block_types"]) - 1
        for i, kind in enumerate(cfg["down_block_types"]):
            for j in range(cfg["layers_per_block"]):
                h = res(f"down_blocks.{i}.resnets.{j}.", h)
                if kind.startswith("CrossAttn"):
                    h = self.transformer(f"down_blocks.{i}.attentions.{j}.", h, context)
                skips.append(h)
            if i < last:
                h = conv(w, f"down_blocks.{i}.downsamplers.0.conv.", h, stride=2)
                skips.append(h)
        h = res("mid_block.resnets.0.", h)
        h = self.transformer("mid_block.attentions.0.", h, context)
        h = res("mid_block.resnets.1.", h)
        for i, kind in enumerate(cfg["up_block_types"]):
            for j in range(cfg["layers_per_block"] + 1):
                h = res(f"up_blocks.{i}.resnets.{j}.", torch.cat([h, skips.pop()], dim=1))
                if kind.startswith("CrossAttn"):
                    h = self.transformer(f"up_blocks.{i}.attentions.{j}.", h, context)
            if i < last:
                h = conv(w, f"up_blocks.{i}.upsamplers.0.conv.", F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return conv(w, "conv_out.", group_norm(w, "conv_norm_out.", h, eps, silu=True))


class VaeDecoder:
    """The VAE decoder: ``self(latents)``, the sampler's latents (B, 4, 64, 64) -> images (B, 3, 512, 512) in about
    [-1, 1] (not clamped)."""

    EPS = 1e-6

    def __init__(self, directory: str):
        self.cfg = _config(os.path.join(directory, "config.json"))
        self.scale = self.cfg.get("scaling_factor", 0.18215)
        self.w = load_safetensors(os.path.join(directory, "diffusion_pytorch_model.safetensors"),
                                  ("decoder.", "post_quant_conv."))

    def attention(self, p: str, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        q, k, v, out = ("query.", "key.", "value.", "proj_attn.") if p + "query.weight" in w else \
            ("to_q.", "to_k.", "to_v.", "to_out.0.")  # (the names of older / newer diffusers files)
        h = _to_tokens(group_norm(w, p + "group_norm.", x, self.EPS))
        o = attention(linear(w, p + q, h), linear(w, p + k, h), linear(w, p + v, h), 1)
        return _to_pixels(linear(w, p + out, o), x) + x

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        w, eps = self.w, self.EPS
        h = conv(w, "decoder.conv_in.", conv(w, "post_quant_conv.", latents / self.scale))
        h = resnet(w, "decoder.mid_block.resnets.0.", h, eps=eps)
        h = self.attention("decoder.mid_block.attentions.0.", h)
        h = resnet(w, "decoder.mid_block.resnets.1.", h, eps=eps)
        blocks = len(self.cfg["up_block_types"])
        for i in range(blocks):
            for j in range(self.cfg["layers_per_block"] + 1):
                h = resnet(w, f"decoder.up_blocks.{i}.resnets.{j}.", h, eps=eps)
            if i < blocks - 1:
                h = conv(w, f"decoder.up_blocks.{i}.upsamplers.0.conv.",
                         F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return conv(w, "decoder.conv_out.", group_norm(w, "decoder.conv_norm_out.", h, eps, silu=True))
