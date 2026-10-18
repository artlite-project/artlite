"""sd15's VAE decoder (``operators.sd15.vae``) on the NPU: its convs, norms and blocks per level vs torch."""

import os

import pytest
import torch
import torch.nn.functional as F

from artlite.operators.design import COLS
from artlite.operators.sd15 import layout as L
from artlite.operators.sd15 import vae
from artlite.operators.sd15.build import Builder, Level
from artlite.operators.sd15.runtime import Sd15Device
from artlite.operators.sd15.unet import DTensor


@pytest.fixture(scope="module")
def npu() -> Sd15Device:
    return Sd15Device()


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


@pytest.mark.parametrize("H, Cin, N", [(64, 512, 512), (128, 512, 512), (256, 256, 256), (512, 128, 256),
                                      (512, 128, 128)])
def test_conv(npu: Sd15Device, H: int, Cin: int, N: int) -> None:
    """A 3x3 conv (padding 1) of every band of a level (64: row tiles; 128: even / odd groups; 256, 512: wide rows, an
    op per half row) from its DDR image, the halves joined at the seam: within 8e-3 of torch on the bfp16 operands."""
    torch.manual_seed(1)
    lv = vae.LEVELS[H]
    x = torch.randn(1, Cin, H, H)
    w, b = torch.randn(N, Cin, 3, 3) / (9 * Cin) ** 0.5, torch.randn(N) * 0.1
    bd = Builder(npu)
    img = vae.VImage(Cin, f"x{H}", lv)
    vae.write(bd, img, x)
    ch = vae.chans(N)
    y = vae.conv_layer(bd, img, w, b, ch=ch, name=f"c{H}", stats=False)
    bd.op({}, {})
    npu.run(bd.transaction())
    err = _rel(vae.read_dtensor(npu, y, ch), qconv(x, w, b))
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
def test_resnet(npu: Sd15Device, H: int, cin: int, cout: int) -> None:
    """The decoder's ResnetBlock2D on a level (GN1 + SiLU, conv1, GN2 + SiLU, conv2 + x or the 1x1 shortcut; the
    halves' statistics joined): within 1.5e-2 of the fp32 reference."""
    from models.sd15 import reference
    torch.manual_seed(2)
    p, lv = f"r{H}.{cin}.{cout}.", vae.LEVELS[H]
    w = _rand_resnet(p, cin, cout)
    x = torch.randn(1, cin, H, H)
    bd = Builder(npu)
    out = vae.resnet(bd, w, p, _input(bd, x, lv, p + "x"), cin=vae.chans(cin), cout=vae.chans(cout), eps=1e-6,
                     name=p + "y")
    bd.op({}, {})
    npu.run(bd.transaction())
    err = _rel(vae.read_dtensor(npu, out, vae.chans(cout)), reference.resnet(w, p, x, eps=1e-6))
    print(f"H {H} {cin} -> {cout}: rel err {err:.3e}")
    assert err < 1.5e-2


@pytest.mark.parametrize("H, C", [(64, 512), (128, 512), (256, 256)])
def test_upsample(npu: Sd15Device, H: int, C: int) -> None:
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
    err = _rel(vae.read_dtensor(npu, y, ch), qconv(up, w[p + "conv.weight"], w[p + "conv.bias"]))
    print(f"H {H} -> {2 * H}: rel err {err:.3e}")
    assert err < 8e-3


@pytest.mark.slow
def test_decoder(npu: Sd15Device) -> None:
    """SD 1.5's VAE decoder (one dispatch) on random latents: within 2e-2 of the fp32 decoder (pixels)."""
    from artlite.model_config import sd15_dir
    from models.sd15 import reference
    path = os.path.join(sd15_dir(), "vae")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    ref = reference.VaeDecoder(path)
    dec = vae.Decoder(npu, ref.w, scale=ref.scale, eps=ref.EPS)
    torch.manual_seed(7)
    lat = torch.randn(1, 4, 64, 64)  # (the sampler's latents: about unit variance)
    err = _rel(dec(lat), ref(lat))
    print(f"rel err {err:.3e}")
    assert err < 2e-2


def test_attention512(npu: Sd15Device) -> None:
    """The mid block's single-head attention of d 512 (role v) for a band of 1024 queries (16 query panels a column, 4 a
    core row) over 1024 keys: within 1.5e-2 of softmax attention on the bfp16 operands."""
    import numpy as np

    from artlite.operators.sd15.seq import ColOps, MemTask
    torch.manual_seed(5)
    nq, nk, d = 8 * 128, 1024, 512
    qs = torch.randn(nq, d) * (1.4426950408889634 / d ** 0.5) * 2
    k, v = torch.randn(nk, d), torch.randn(nk, d)
    bd = Builder(npu)
    kv = bd.image("a512.kv", vae.kv512_image(k, v))
    Qa, Oa = 0, 64 * 16 * 72
    qimg = np.concatenate([L.panels(qs[128 * c:128 * (c + 1)]) for c in range(COLS)])
    qa = bd.image("a512.q", qimg)
    bd.op({c: ColOps(fills=[MemTask([{"addr": Qa, "nwords": qimg.size // 32}], ddr=(qa + c * qimg.size // 8, qimg.size // 8,
                                                                                      None))]) for c in range(COLS)}, {})
    at = vae.Attention512(np=16, q=Qa, o=Oa, kv=kv, nkeys=nk)
    bd.op({c: at.col_ops() for c in range(COLS)}, at.row_ops(bd.image("a512.cmd", at.command())))
    ob = 80 * 16 * 72
    out = bd.ddr("a512.o", COLS * ob)
    bd.op({c: ColOps(drains=[MemTask([{"addr": Oa, "nwords": ob // 4}], ddr=(out + c * ob, ob, None))]) for c in range(COLS)},
          {})
    npu.run(bd.transaction())
    raw = npu.read("a512.o", COLS * ob).reshape(COLS, 80 * 16 * 72)
    o = torch.cat([L.from_panels(raw[c][:64 * 16 * 72], 128, d) for c in range(COLS)])
    qq, kq, vq = q(qs), q(k), q(v)
    p = torch.exp2(qq @ kq.T - (qq @ kq.T).max(-1, keepdim=True).values)
    ref = (p @ vq) / p.sum(-1, keepdim=True)
    err = _rel(o, ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


def test_mid_attention(npu: Sd15Device) -> None:
    """The decoder's mid-block attention on the 64 level (GroupNorm, q / k / v, the d-512 attention over all 4096
    pixels, proj_attn + x): within 1.5e-2 of the fp32 reference."""
    from models.sd15 import reference
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
    err = _rel(vae.read_dtensor(npu, out, vae.chans(C)), ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2
