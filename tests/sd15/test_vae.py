"""sd15's VAE decoder (``models.sd15.vae``) on the NPU: its resnet blocks, upsamplers and mid attention per level
vs torch."""

import pytest
import torch
import torch.nn.functional as F

from artlite.vision import layout as L
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import DTensor
from artlite.vision.levels import ImageLevel
from models.sd15 import vae

from ..numerics import rel_native


def q(x: torch.Tensor) -> torch.Tensor:
    """bfp16ebs8 rounding along the last axis (blocks of 8)."""
    return torch.from_numpy(L.unblocks(L.blocks(x.float().reshape(*x.shape[:-1], -1, 8)))).reshape(x.shape)


def qconv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """torch's conv on the operands as the NPU holds them (bfp16 x and w over the input channels, bf16 bias)."""
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    wq = q(w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return F.conv2d(xq, wq, None if b is None else b.bfloat16().float(), padding=1)


def _rand_resnet(p: str, cin: int, cout: int) -> dict:
    g = torch.Generator().manual_seed(5)
    w = {p + "norm1.weight": 1 + 0.2 * torch.randn(cin, generator=g), p + "norm1.bias": 0.2 * torch.randn(cin, generator=g),
         p + "conv1.weight": torch.randn(cout, cin, 3, 3, generator=g) / (9 * cin) ** 0.5,
         p + "conv1.bias": 0.1 * torch.randn(cout, generator=g),
         p + "norm2.weight": 1 + 0.2 * torch.randn(cout, generator=g), p + "norm2.bias": 0.2 * torch.randn(cout, generator=g),
         p + "conv2.weight": torch.randn(cout, cout, 3, 3, generator=g) / (9 * cout) ** 0.5,
         p + "conv2.bias": 0.1 * torch.randn(cout, generator=g)}
    if cin != cout:
        w[p + "conv_shortcut.weight"] = torch.randn(cout, cin, 1, 1, generator=g) / cin ** 0.5
        w[p + "conv_shortcut.bias"] = 0.1 * torch.randn(cout, generator=g)
    return w


def _input(bd: Builder, x: torch.Tensor, lv: ImageLevel, name: str) -> DTensor:
    ch = vae.chans(x.shape[1])
    t = DTensor(lv, ch.P, name)
    vae.write_dtensor(bd, t, x, ch)
    vae.with_stats(bd, t, ch, name=name + ".st")
    return t


@pytest.mark.parametrize("H, cin, cout", [(64, 512, 512), (128, 512, 512), (256, 512, 256),
                                         (512, 256, 128), (512, 128, 128)])
def test_resnet(npu: VisionDevice, H: int, cin: int, cout: int) -> None:
    """The decoder's ResnetBlock2D on a level (GN1 + SiLU, conv1, GN2 + SiLU, conv2 + x or the 1x1 shortcut; the
    halves' statistics joined): within 1.5e-2 of the fp32 reference."""
    from . import ref as reference
    torch.manual_seed(2)
    p, lv = f"r{H}.{cin}.{cout}.", vae.LEVELS[H]
    w = _rand_resnet(p, cin, cout)
    x = torch.randn(1, cin, H, H)
    bd = Builder(npu)
    out = vae.resnet(bd, w, p, _input(bd, x, lv, p + "x"), cin=vae.chans(cin), cout=vae.chans(cout), eps=1e-6,
                     name=p + "y")
    bd.op({}, {})
    npu.run(bd.transaction())
    err = rel_native(vae.read_dtensor(npu, out, vae.chans(cout)), reference.resnet(w, p, x, eps=1e-6))
    print(f"H {H} {cin} -> {cout}: rel err {err:.3e}")
    assert err < 1.5e-2


@pytest.mark.parametrize("H, C", [(64, 512), (256, 256)])
def test_upsample(npu: VisionDevice, H: int, C: int) -> None:
    """The decoder's upsampler (nearest 2x in the drains into the next level's image, then its 3x3 conv): within 8e-3
    of torch on the bfp16 operands."""
    torch.manual_seed(3)
    p, lv = f"u{H}.", vae.LEVELS[H]
    x = torch.randn(1, C, H, H)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": torch.randn(C) * 0.1}
    bd = Builder(npu)
    ch = vae.chans(C)
    xd = DTensor(lv, ch.P, p + "x")
    vae.write_dtensor(bd, xd, x, ch)
    y = vae.upsample(bd, w, p, xd, ch=ch, name=p + "y")
    bd.op({}, {})
    npu.run(bd.transaction())
    up = F.interpolate(x.bfloat16().float(), scale_factor=2.0, mode="nearest")
    err = rel_native(vae.read_dtensor(npu, y, ch), qconv(up, w[p + "conv.weight"], w[p + "conv.bias"]))
    print(f"H {H} -> {2 * H}: rel err {err:.3e}")
    assert err < 8e-3


def test_mid_attention(npu: VisionDevice) -> None:
    """The decoder's mid-block attention on the 64 level (GroupNorm, q / k / v, the d-512 attention over all 4096
    pixels, proj_attn + x): within 1.5e-2 of the fp32 reference."""
    from . import ref as reference
    torch.manual_seed(6)
    p, C, lv = "m.", 512, vae.LEVELS[64]
    g = torch.Generator().manual_seed(8)
    w = {p + "group_norm.weight": 1 + 0.2 * torch.randn(C, generator=g), p + "group_norm.bias": 0.2 * torch.randn(C, generator=g)}
    for n in ("to_q.", "to_k.", "to_v.", "to_out.0."):
        w[p + n + "weight"] = torch.randn(C, C, generator=g) / C ** 0.5
        w[p + n + "bias"] = 0.1 * torch.randn(C, generator=g)
    x = torch.randn(1, C, 64, 64)
    bd = Builder(npu)
    out = vae.mid_attention(bd, w, p, _input(bd, x, lv, "ma.x"), eps=1e-6, name="ma")
    bd.op({}, {})
    npu.run(bd.transaction())
    ref = reference.VaeDecoder.attention(type("R", (), {"w": w, "EPS": 1e-6})(), p, x)
    err = rel_native(vae.read_dtensor(npu, out, vae.chans(C)), ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2
