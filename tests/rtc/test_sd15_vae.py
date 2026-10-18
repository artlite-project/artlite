"""sd15_v2's VAE decoder (``operators.sd15_v2.vae``) on the NPU: its convs, norms and blocks per level vs torch."""

import os

import pytest
import torch
import torch.nn.functional as F

from artlite.operators.design import COLS
from artlite.operators.sd15_v2 import layout as L
from artlite.operators.sd15_v2 import vae
from artlite.operators.sd15_v2.build import Builder, Level
from artlite.operators.sd15_v2.runtime import V2Device
from artlite.operators.sd15_v2.unet import DTensor


@pytest.fixture(scope="module")
def v2() -> V2Device:
    return V2Device()


def q(x: torch.Tensor) -> torch.Tensor:
    """bfp16ebs8 rounding along the last axis (blocks of 8)."""
    return torch.from_numpy(L.unblocks(L.blocks(x.float().reshape(*x.shape[:-1], -1, 8)))).reshape(x.shape)


def qconv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """torch's conv on the operands as the NPU holds them (bfp16 x and w over the input channels, bf16 bias)."""
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    wq = q(w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return F.conv2d(xq, wq, None if b is None else b.bfloat16().float(), padding=1)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y - ref).norm() / ref.norm())


def read(v2: V2Device, y: DTensor, ch: vae.Chans) -> torch.Tensor:
    """A DTensor's real channels as (1, C, H, H)."""
    lv = y.lv
    bands = [v2.read(y.band(b), COLS * y.nbytes).reshape(COLS, -1) for b in range(lv.nb)]
    t = torch.stack([torch.cat([L.from_ctiles(bands[b][c], lv.T, ch.P, 64) for b in range(lv.nb)]) for c in range(COLS)])
    return vae.from_tokens(ch.unpad(t), lv)


@pytest.mark.parametrize("H, Cin, N", [(64, 512, 512), (128, 512, 512), (256, 256, 256), (512, 128, 256),
                                      (512, 128, 128)])
def test_conv(v2: V2Device, H: int, Cin: int, N: int) -> None:
    """A 3x3 conv (padding 1) of every band of a level (64: row tiles; 128: even / odd groups; 256, 512: wide rows, an
    op per half row) from its DDR image, the halves joined at the seam: within 8e-3 of torch on the bfp16 operands."""
    torch.manual_seed(1)
    lv = vae.LEVELS[H]
    x = torch.randn(1, Cin, H, H)
    w, b = torch.randn(N, Cin, 3, 3) / (9 * Cin) ** 0.5, torch.randn(N) * 0.1
    bd = Builder(v2)
    img = vae.VImage(Cin, f"x{H}", lv)
    vae.write(bd, img, x)
    ch = vae.chans(N)
    y = vae.conv_layer(bd, img, w, b, ch=ch, name=f"c{H}", stats=False)
    bd.op({}, {})
    v2.run(bd.transaction())
    err = _rel(read(v2, y, ch), qconv(x, w, b))
    print(f"H {H}: rel err {err:.3e}")
    assert err < 8e-3


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


def _input(bd: Builder, x: torch.Tensor, lv: Level, name: str) -> DTensor:
    ch = vae.chans(x.shape[1])
    t = DTensor(lv, ch.P, name)
    vae.write_dtensor(bd, t, x, ch)
    vae.with_stats(bd, t, ch, name=name + ".st")
    return t


@pytest.mark.parametrize("H, cin, cout", [(64, 512, 512), (128, 512, 512), (256, 512, 256), (256, 256, 256),
                                         (512, 256, 128), (512, 128, 128)])
def test_resnet(v2: V2Device, H: int, cin: int, cout: int) -> None:
    """The decoder's ResnetBlock2D on a level (GN1 + SiLU, conv1, GN2 + SiLU, conv2 + x or the 1x1 shortcut; the
    halves' statistics joined): within 1.5e-2 of the fp32 reference."""
    from models.sd15 import reference
    torch.manual_seed(2)
    p, lv = f"r{H}.{cin}.{cout}.", vae.LEVELS[H]
    w = _rand_resnet(p, cin, cout)
    x = torch.randn(1, cin, H, H)
    bd = Builder(v2)
    out = vae.resnet(bd, w, p, _input(bd, x, lv, p + "x"), cin=vae.chans(cin), cout=vae.chans(cout), eps=1e-6,
                     name=p + "y")
    bd.op({}, {})
    v2.run(bd.transaction())
    err = _rel(read(v2, out, vae.chans(cout)), reference.resnet(w, p, x, eps=1e-6))
    print(f"H {H} {cin} -> {cout}: rel err {err:.3e}")
    assert err < 1.5e-2


@pytest.mark.parametrize("H, C", [(64, 512), (128, 512), (256, 256)])
def test_upsample(v2: V2Device, H: int, C: int) -> None:
    """The decoder's upsampler (nearest 2x in the drains into the next level's image, then its 3x3 conv): within 8e-3
    of torch on the bfp16 operands."""
    torch.manual_seed(3)
    p, lv = f"u{H}.", vae.LEVELS[H]
    x = torch.randn(1, C, H, H)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": torch.randn(C) * 0.1}
    bd = Builder(v2)
    ch = vae.chans(C)
    xd = DTensor(lv, ch.P, p + "x")
    vae.write_dtensor(bd, xd, x, ch)
    y = vae.upsample(bd, w, p, xd, ch=ch, name=p + "y")
    bd.op({}, {})
    v2.run(bd.transaction())
    up = F.interpolate(x.bfloat16().float(), scale_factor=2.0, mode="nearest")
    err = _rel(read(v2, y, ch), qconv(up, w[p + "conv.weight"], w[p + "conv.bias"]))
    print(f"H {H} -> {2 * H}: rel err {err:.3e}")
    assert err < 8e-3


@pytest.mark.slow
def test_decoder(v2: V2Device) -> None:
    """SD 1.5's VAE decoder (its mid attention on the host) on the sampler's latents of a prompt: within 2e-2 of the
    fp32 decoder (pixels)."""
    from artlite.model_config import sd15_dir
    from models.sd15 import reference
    path = os.path.join(sd15_dir(), "vae")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    ref = reference.VaeDecoder(path)
    dec = vae.Decoder(v2, ref.w, scale=ref.scale, eps=ref.EPS)
    torch.manual_seed(7)
    lat = torch.randn(1, 4, 64, 64)  # (the sampler's latents: about unit variance)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = dec(lat, lambda h: ref.attention("decoder.mid_block.attentions.0.", h).float())
    err = _rel(y, ref(lat))
    print(f"rel err {err:.3e}")
    assert err < 2e-2
