"""The SD 1.x UNet on the vision overlay (``models.sd15``: the blocks of ``unet`` and ``level64``, the step of
``model``), run by host-built transactions against torch on random data."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from artlite.operators.design import COLS
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.build import Builder, Level
from artlite.vision.device import VisionDevice
from artlite.vision.layers import DTensor, Tensor

from ..vision.test_ops import _rel, from_il_tokens, from_level_tokens, il_tokens, level_tokens, q, qconv_w


@pytest.fixture(scope="module")
def npu() -> VisionDevice:
    return VisionDevice()


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


@pytest.mark.parametrize("H, cin, cout", [(8, 1280, 1280), (16, 640, 1280), (16, 1280, 1280)])
def test_resnet(npu: VisionDevice, H: int, cin: int, cout: int) -> None:
    """diffusers' ResnetBlock2D on the level (GN1 + SiLU, conv1 + time bias, GN2 + SiLU, conv2 + x or the 1x1
    shortcut; the input's GroupNorm records by a copy op): within 1.5e-2 of the fp32 reference."""
    from artlite.vision.layers import copy_stats
    from models.sd15 import reference
    from models.sd15.unet import resnet
    torch.manual_seed(6)
    p = "r."
    w = _rand_resnet(p, cin, cout)
    tb = 0.3 * torch.randn(cout)
    x = torch.randn(2, cin, H, H)
    lv = Level(H)
    bd = Builder(npu)
    xt = Tensor(lv, cin, bd.mem.alloc(lv.ct(cin)))
    xt_cols = level_tokens(x, lv.R, lv.Lg)
    img = bd.image("x_in", np.concatenate([L.ctiles(xt_cols[c], lv.m) for c in range(COLS)]))
    assert img
    bd.op(bd.load("x_in", xt.addr, lv.ct(cin)), {})
    xt.rec = copy_stats(bd, xt, name="x.st", st=(cin // 32, 0))
    out = resnet(bd, w, p, xt, cout=cout, time_bias=tb, eps=1e-5, stats=None)
    bd.op(bd.spill("y_out", out.addr, lv.ct(cout)), {})
    npu.run(bd.transaction())
    yb = npu.read("y_out", COLS * lv.ct(cout)).reshape(COLS, -1)
    y = from_level_tokens(torch.stack([L.from_ctiles(yb[c], lv.m, cout, lv.m) for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, x, eps=1e-5, time_bias=tb)
    assert _rel(y, ref) < 1.5e-2


@pytest.mark.slow
@pytest.mark.parametrize("cin, cout", [(640, 640), (320, 640)])
def test_resnet_bands(npu: VisionDevice, cin: int, cout: int) -> None:
    """ResnetBlock2D on the banded 32x32 level (x and the output in DDR per band, h parked; the shortcut where cin !=
    cout): within 1.5e-2 of the fp32 reference."""
    from artlite.vision.layers import DTensor, copy_stats
    from models.sd15 import reference
    from models.sd15.unet import resnet_bands
    torch.manual_seed(13)
    p, H = "r.", 32
    w = _rand_resnet(p, cin, cout)
    tb = 0.3 * torch.randn(cout)
    x = torch.randn(2, cin, H, H)
    lv = Level(H, 2)
    bd = Builder(npu)
    xc = level_tokens(x, lv.R, lv.Lg)
    xd = DTensor(lv, cin, "x_in", rec=[])
    for b in range(2):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    for b in range(2):
        t = xd.load(bd, b)
        xd.rec += copy_stats(bd, t, name=f"x.st{b}", st=(cin // 32, 0))
        bd.mem.release(t.addr, t.nbytes)
    out = resnet_bands(bd, w, p, xd, cout=cout, time_bias=tb, eps=1e-5, stats=None, name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    yb = [npu.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, cout, lv.m) for b in range(2)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, x, eps=1e-5, time_bias=tb)
    err = _rel(y, ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


def _il_input(bd: Builder, x: torch.Tensor, name: str, st: tuple[int, int]) -> DTensor:
    """A 64x64 DTensor of x (2, C, 64, 64) in DDR per band, with GroupNorm records for st (copy ops)."""
    from artlite.vision.layers import copy_stats
    from models.sd15.level64 import LV64
    lv, C = LV64, x.shape[1]
    xc = il_tokens(x, lv.R)
    xd = DTensor(lv, C, name, rec=[])
    for b in range(lv.nb):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    for c0 in range(0, C, 320):  # (parts of 320 channels: a band's C tiles and panels in the memtile)
        for b in range(lv.nb):
            t = xd.load(bd, b, c0, 320)
            xd.rec += copy_stats(bd, t, name=f"{name}.st{c0}", st=(st[0], st[1] + c0), band=b)
            bd.mem.release(t.addr, t.nbytes)
    return xd


def _il_output(npu: VisionDevice, y: DTensor) -> torch.Tensor:
    lv = y.lv
    yb = [npu.read(y.band(b), COLS * y.nbytes).reshape(COLS, -1) for b in range(lv.nb)]
    return from_il_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, y.C, lv.m) for b in range(lv.nb)])
                                       for c in range(COLS)]), lv.R)


@pytest.mark.slow
def test_resnet64(npu: VisionDevice) -> None:
    """ResnetBlock2D 320 -> 320 on the 64x64 level (``level64``: x and the output in DDR per band, the convs' input rows
    from DDR panel images): within 1.5e-2 of the fp32 reference."""
    from models.sd15 import reference
    from models.sd15.level64 import resnet64
    torch.manual_seed(15)
    p, C = "r.", 320
    w = _rand_resnet(p, C, C)
    tb = 0.3 * torch.randn(C)
    x = torch.randn(2, C, 64, 64)
    bd = Builder(npu)
    xd = _il_input(bd, x, "x_in", (C // 32, 0))
    out = resnet64(bd, w, p, xd, cout=C, time_bias=tb, eps=1e-5, stats=None, name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    err = _rel(_il_output(npu, out), reference.resnet(w, p, x, eps=1e-5, time_bias=tb))
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


@pytest.mark.slow
@pytest.mark.parametrize("ch, cs", [(1280, 640), (640, 320)])
def test_resnet_cat_bands(npu: VisionDevice, ch: int, cs: int) -> None:
    """The up path's resnet on [h | skip] on the banded 32x32 level (K parts of at most 640 channels, partial sums
    parked in DDR): within 1.5e-2 of the fp32 reference."""
    from artlite.vision.layers import DTensor
    from models.sd15 import reference
    from models.sd15.unet import resnet_cat_bands, skip_bands
    torch.manual_seed(14)
    p, H, cout = "r.", 32, 640
    w = _rand_resnet(p, ch + cs, cout)
    tb = 0.3 * torch.randn(cout)
    hx, sk = torch.randn(2, ch, H, H), torch.randn(2, cs, H, H)
    lv = Level(H, 2)
    bd = Builder(npu)

    def ddr(x: torch.Tensor, name: str) -> DTensor:
        xc = level_tokens(x, lv.R, lv.Lg)
        t = DTensor(lv, x.shape[1], name)
        for b in range(2):
            bd.image(t.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
        return t
    C = ch + cs
    hd = skip_bands(bd, ddr(hx, "h_in"), name="h", st=(C // 32, 0))
    sd = skip_bands(bd, ddr(sk, "s_in"), name="s", st=(C // 32, ch))
    out = resnet_cat_bands(bd, w, p, hd, sd, cout=cout, time_bias=tb, eps=1e-5, stats=None, name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    yb = [npu.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, cout, lv.m) for b in range(2)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, torch.cat([hx, sk], 1), eps=1e-5, time_bias=tb)
    err = _rel(y, ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


def test_downsample_bands(npu: VisionDevice) -> None:
    """The stride-2 downsampler of the banded 32x32 level (x in DDR per band) into 16x16 C tiles: within 8e-3 of torch
    on the bfp16-rounded input."""
    from artlite.vision.layers import DTensor
    from models.sd15.unet import downsample_bands
    torch.manual_seed(15)
    C, p, H = 640, "s.", 32
    x = torch.randn(2, C, H, H)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": 0.1 * torch.randn(C)}
    lv = Level(H, 2)
    bd = Builder(npu)
    xc = level_tokens(x, lv.R, lv.Lg)
    xd = DTensor(lv, C, "x_in")
    for b in range(2):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    out = downsample_bands(bd, w, p, xd)
    y = _level_output(npu, bd, out, "y_out")
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), stride=2, padding=1)
    assert _rel(y, ref) < 8e-3, _rel(y, ref)


@pytest.mark.slow
def test_upsample_bands(npu: VisionDevice) -> None:
    """The nearest 2x upsampler with its conv from 16x16 C tiles (1280 channels) into the banded 32x32 level (K parts,
    output halves, partial sums in DDR): within 8e-3 of torch on the bfp16-rounded input."""
    from models.sd15.unet import upsample_bands
    torch.manual_seed(16)
    C, p = 1280, "s."
    x = torch.randn(2, C, 16, 16)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": 0.1 * torch.randn(C)}
    bd = Builder(npu)
    xt = _level_input(bd, Level(16), x, "x_in")
    out = upsample_bands(bd, w, p, xt, stats=None, name="u")
    bd.op({}, {})
    npu.run(bd.transaction())
    lv = out.lv
    yb = [npu.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, C, lv.m) for b in range(2)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    xq = F.interpolate(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), scale_factor=2.0, mode="nearest")
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), padding=1)
    assert _rel(y, ref) < 8e-3, _rel(y, ref)


def _rand_transformer(p: str, C: int = 1280, cd: int = 768) -> dict:
    g = torch.Generator().manual_seed(7)
    q = p + "transformer_blocks.0."

    def lin(n: int, k: int, bias: bool = True, conv: bool = False) -> tuple:
        wt = torch.randn(n, k, generator=g) / k ** 0.5
        return (wt[:, :, None, None] if conv else wt), (0.1 * torch.randn(n, generator=g) if bias else None)

    w = {}
    for name, n, k, bias, conv in (("proj_in", C, C, True, True), ("proj_out", C, C, True, True)):
        w[p + name + ".weight"], w[p + name + ".bias"] = lin(n, k, bias, conv)
    for nm in (p + "norm.", q + "norm1.", q + "norm2.", q + "norm3."):
        w[nm + "weight"], w[nm + "bias"] = 1 + 0.2 * torch.randn(C, generator=g), 0.2 * torch.randn(C, generator=g)
    for a in ("attn1", "attn2"):
        kin = C if a == "attn1" else cd
        w[q + a + ".to_q.weight"] = lin(C, C)[0]
        w[q + a + ".to_k.weight"] = lin(C, kin)[0]
        w[q + a + ".to_v.weight"] = lin(C, kin)[0]
        w[q + a + ".to_out.0.weight"], w[q + a + ".to_out.0.bias"] = lin(C, C)
    w[q + "ff.net.0.proj.weight"], w[q + "ff.net.0.proj.bias"] = lin(8 * C, C)
    w[q + "ff.net.2.weight"], w[q + "ff.net.2.bias"] = lin(C, 4 * C)
    return w


@pytest.mark.parametrize("H", [8, 16])
def test_transformer(npu: VisionDevice, H: int) -> None:
    """SD 1.x's Transformer2DModel on the level (GN, proj_in, self-attention, cross-attention on 77 context keys (the
    host's K / V slots), GEGLU feed-forward in two halves, proj_out + x; 16x16: x spilled to DDR): within 2e-2 of the
    fp32 reference."""
    import types

    from artlite.vision.layers import copy_stats
    from models.sd15 import reference
    from models.sd15.unet import transformer
    torch.manual_seed(8)
    p, C = "t.", 1280
    w = _rand_transformer(p)
    x = torch.randn(2, C, H, H)
    ctx = torch.randn(2, 77, 768)
    q = p + "transformer_blocks.0.attn2."
    lv = Level(H)
    bd = Builder(npu)
    xt = Tensor(lv, C, bd.mem.alloc(lv.ct(C)))
    xc = level_tokens(x, lv.R, lv.Lg)
    bd.image("x_in", np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load("x_in", xt.addr, lv.ct(C)), {})
    xt.rec = copy_stats(bd, xt, name="x.st", st=(C // 32, 0))
    kv = bd.ddr("kv_self", 16 * lv.np * 2 * D.Q)
    cks = L.kv_slots(ctx @ w[q + "to_k.weight"].T, ctx @ w[q + "to_v.weight"].T)
    ckv = bd.image("kv_ctx", cks)
    out = transformer(bd, w, p, xt, kv=kv, ctx_kv=ckv, nctx=77, stats=None, spill=H == 16)
    bd.op(bd.spill("y_out", out.addr, lv.ct(C)), {})
    npu.run(bd.transaction())
    yb = npu.read("y_out", COLS * lv.ct(C)).reshape(COLS, -1)
    y = from_level_tokens(torch.stack([L.from_ctiles(yb[c], lv.m, C, lv.m) for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.UNet.transformer(types.SimpleNamespace(w=w, cfg={"attention_head_dim": 8}), p, x, ctx)
    assert _rel(y, ref) < 2e-2


@pytest.mark.slow
def test_transformer_bands(npu: VisionDevice) -> None:
    """The transformer on a banded level (32x32, 640 channels, d 80: 2 bands, x and the output in DDR, h and Q of band 0
    parked): within 2e-2 of the fp32 reference."""
    import types

    from artlite.vision.layers import DTensor, copy_stats
    from models.sd15 import reference
    from models.sd15.unet import transformer_bands
    torch.manual_seed(11)
    p, C, H = "t.", 640, 32
    w = _rand_transformer(p, C)
    x = torch.randn(2, C, H, H)
    ctx = torch.randn(2, 77, 768)
    q = p + "transformer_blocks.0.attn2."
    lv = Level(H, 2)
    bd = Builder(npu)
    xc = level_tokens(x, lv.R, lv.Lg)  # (COLS, 2 T, C): band b = tokens [T b, T (b + 1))
    xd = DTensor(lv, C, "x_in")
    for b in range(2):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    xd.rec = []
    for b in range(2):  # (the input's records: a copy op per band)
        t = xd.load(bd, b)
        xd.rec += copy_stats(bd, t, name=f"x.st{b}", st=(C // 32, 0))
        bd.mem.release(t.addr, t.nbytes)
    from artlite.vision.ops import Attention
    kv = bd.ddr("kv_self", Attention(np=lv.np, q=0, o=0, kv=0, d=C // 8, nbands=2).kv_bytes())
    ckv = bd.image("kv_ctx", L.kv_slots(ctx @ w[q + "to_k.weight"].T, ctx @ w[q + "to_v.weight"].T, d=C // 8))
    out = transformer_bands(bd, w, p, xd, kv=kv, ctx_kv=ckv, nctx=77, stats=None, name="y")
    bd.op({}, {})  # (the last band's deferred drain)
    npu.run(bd.transaction())
    yb = [npu.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(2)]
    yc = torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, C, lv.m) for b in range(2)]) for c in range(COLS)])
    y = from_level_tokens(yc, lv.R, lv.Lg)
    ref = reference.UNet.transformer(types.SimpleNamespace(w=w, cfg={"attention_head_dim": 8}), p, x, ctx)
    err = _rel(y, ref)
    print(f"rel err {err:.3e}")
    assert err < 2e-2


def _b32_input(bd: Builder, x: torch.Tensor, name: str, st: tuple[int, int]) -> DTensor:
    """A 32x32 DTensor of x (2, C, 32, 32) in DDR per band (``Level(32, 2)``), with GroupNorm records for st."""
    from artlite.vision.layers import copy_stats
    lv, C = Level(32, 2), x.shape[1]
    xc = level_tokens(x, lv.R, lv.Lg)
    xd = DTensor(lv, C, name, rec=[])
    for b in range(lv.nb):
        bd.image(xd.band(b), np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
    for b in range(lv.nb):
        t = xd.load(bd, b)
        xd.rec += copy_stats(bd, t, name=f"{name}.st{b}", st=st, band=b)
        bd.mem.release(t.addr, t.nbytes)
    return xd


@pytest.mark.slow
def test_down64(npu: VisionDevice) -> None:
    """The 64x64 level's stride-2 conv (320 -> 320) into the 32x32 level's bands (``level64.down64``): within 8e-3 of
    torch on the bfp16-rounded operands."""
    from models.sd15.level64 import down64
    torch.manual_seed(17)
    C = 320
    x = torch.randn(2, C, 64, 64)
    w = {"s.conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, "s.conv.bias": 0.1 * torch.randn(C)}
    bd = Builder(npu)
    xd = _il_input(bd, x, "x_in", (C // 32, 0))
    out = down64(bd, w, "s.", xd, stats=(C // 32, 0), name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    lv = out.lv
    yb = [npu.read(out.band(b), COLS * out.nbytes).reshape(COLS, -1) for b in range(lv.nb)]
    y = from_level_tokens(torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, C, lv.m) for b in range(lv.nb)])
                                       for c in range(COLS)]), lv.R, lv.Lg)
    ref = F.conv2d(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), qconv_w(w["s.conv.weight"]),
                   w["s.conv.bias"].bfloat16().float(), padding=1, stride=2)
    err = _rel(y, ref)
    assert err < 8e-3, err


@pytest.mark.slow
def test_up64(npu: VisionDevice) -> None:
    """Nearest 2x of a 32x32 level's bands (640 channels) and the conv (640 -> 640, K parts and output halves of 320)
    into the 64x64 level (``level64.up64``): within 8e-3 of torch on the bfp16-rounded operands."""
    from models.sd15.level64 import up64
    torch.manual_seed(18)
    C = 640
    x = torch.randn(2, C, 32, 32)
    w = {"s.conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, "s.conv.bias": 0.1 * torch.randn(C)}
    bd = Builder(npu)
    xd = _b32_input(bd, x, "x_in", (C // 32, 0))
    out = up64(bd, w, "s.", xd, stats=(30, 0), name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    xu = F.interpolate(q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), scale_factor=2.0, mode="nearest")
    ref = F.conv2d(xu, qconv_w(w["s.conv.weight"]), w["s.conv.bias"].bfloat16().float(), padding=1)
    err = _rel(_il_output(npu, out), ref)
    assert err < 8e-3, err


@pytest.mark.slow
@pytest.mark.parametrize("ch", [640, 320])
def test_resnet_cat64(npu: VisionDevice, ch: int) -> None:
    """The up path's resnet on [h | skip] (skip 320) -> 320 on the 64x64 level (K parts of 320): within 1.5e-2 of the
    fp32 reference."""
    from models.sd15 import reference
    from models.sd15.level64 import resnet_cat64
    torch.manual_seed(19)
    p, cs, cout = "r.", 320, 320
    w = _rand_resnet(p, ch + cs, cout)
    tb = 0.3 * torch.randn(cout)
    h, sk = torch.randn(2, ch, 64, 64), torch.randn(2, cs, 64, 64)
    gs = (ch + cs) // 32
    bd = Builder(npu)
    hd = _il_input(bd, h, "h_in", (gs, 0))
    sd = _il_input(bd, sk, "s_in", (gs, ch))
    out = resnet_cat64(bd, w, p, hd, sd, cout=cout, time_bias=tb, eps=1e-5, stats=None, name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    ref = reference.resnet(w, p, torch.cat([h, sk], 1), eps=1e-5, time_bias=tb)
    err = _rel(_il_output(npu, out), ref)
    print(f"rel err {err:.3e}")
    assert err < 1.5e-2


@pytest.mark.slow
def test_transformer64(npu: VisionDevice) -> None:
    """The transformer on the 64x64 level (320 channels, d 40 on role c: 4 bands, x and the output in DDR, h and Q of
    bands 0 .. 2 parked): within 2e-2 of the fp32 reference."""
    import types

    from artlite.vision.ops import Attention
    from models.sd15 import reference
    from models.sd15.level64 import LV64
    from models.sd15.unet import transformer_bands
    torch.manual_seed(16)
    p, C = "t.", 320
    w = _rand_transformer(p, C)
    x = torch.randn(2, C, 64, 64)
    ctx = torch.randn(2, 77, 768)
    q = p + "transformer_blocks.0.attn2."
    bd = Builder(npu)
    xd = _il_input(bd, x, "x_in", (C // 32, 0))
    kv = bd.ddr("kv_self64", Attention(np=LV64.np, q=0, o=0, kv=0, d=40, nbands=LV64.nb).kv_bytes())
    ckv = bd.image("kv_ctx40", L.kv_slots(ctx @ w[q + "to_k.weight"].T, ctx @ w[q + "to_v.weight"].T, d=40))
    out = transformer_bands(bd, w, p, xd, kv=kv, ctx_kv=ckv, nctx=77, stats=None, name="y")
    bd.op({}, {})
    npu.run(bd.transaction())
    ref = reference.UNet.transformer(types.SimpleNamespace(w=w, cfg={"attention_head_dim": 8}), p, x, ctx)
    err = _rel(_il_output(npu, out), ref)
    print(f"rel err {err:.3e}")
    assert err < 2e-2


def _level_input(bd: Builder, lv: Level, x: torch.Tensor, name: str) -> Tensor:
    """x (2, C, H, H) as the level's C tiles: a DDR image, loaded into the memtile; the Tensor (no records)."""
    C = x.shape[1]
    t = Tensor(lv, C, bd.mem.alloc(lv.ct(C)))
    xc = level_tokens(x, lv.R, lv.Lg)
    bd.image(name, np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load(name, t.addr, lv.ct(C)), {})
    return t


def _level_output(npu: VisionDevice, bd: Builder, t: Tensor, name: str) -> torch.Tensor:
    bd.op(bd.spill(name, t.addr, t.nbytes), {})
    npu.run(bd.transaction())
    lv = t.lv
    yb = npu.read(name, COLS * t.nbytes).reshape(COLS, -1)
    return from_level_tokens(torch.stack([L.from_ctiles(yb[c], lv.m, t.C, lv.m) for c in range(COLS)]), lv.R, lv.Lg)


@pytest.mark.parametrize("H, ch, cs", [(8, 1280, 1280), (16, 1280, 1280), (16, 1280, 640)])
def test_resnet_cat(npu: VisionDevice, H: int, ch: int, cs: int) -> None:
    """The up path's resnet on the concatenation [h | skip] (the skip from DDR; GN over both, conv1 and the shortcut in
    two K-chunks): within 1.5e-2 of the fp32 reference."""
    from artlite.vision.layers import copy_stats, free
    from models.sd15 import reference
    from models.sd15.unet import make_skip, resnet_cat
    torch.manual_seed(9)
    p, cout = "r.", 1280
    w = _rand_resnet(p, ch + cs, cout)
    tb = 0.3 * torch.randn(cout)
    h, sk = torch.randn(2, ch, H, H), torch.randn(2, cs, H, H)
    lv = Level(H)
    bd = Builder(npu)
    st = _level_input(bd, lv, sk, "s_in")
    skip = make_skip(bd, st, name="skip", st=((ch + cs) // 32, ch))
    free(bd, st)
    ht = _level_input(bd, lv, h, "h_in")
    ht.rec = copy_stats(bd, ht, name="h.st", st=((ch + cs) // 32, 0))
    out = resnet_cat(bd, w, p, ht, skip, cout=cout, time_bias=tb, eps=1e-5, stats=None)
    y = _level_output(npu, bd, out, "y_out")
    ref = reference.resnet(w, p, torch.cat([h, sk], 1), eps=1e-5, time_bias=tb)
    assert _rel(y, ref) < 1.5e-2


@pytest.mark.parametrize("kind", ["down", "up"])
def test_resample(npu: VisionDevice, kind: str) -> None:
    """The stride-2 downsampler (16x16 -> 8x8) and the nearest 2x upsampler with its conv (8x8 -> 16x16): within 8e-3
    of torch on the bfp16-rounded input."""
    from models.sd15.unet import downsample, upsample
    torch.manual_seed(10)
    C, p = 1280, "s."
    H = 16 if kind == "down" else 8
    x = torch.randn(2, C, H, H)
    w = {p + "conv.weight": torch.randn(C, C, 3, 3) / (9 * C) ** 0.5, p + "conv.bias": 0.1 * torch.randn(C)}
    bd = Builder(npu)
    xt = _level_input(bd, Level(H), x, "x_in")
    out = (downsample if kind == "down" else upsample)(bd, w, p, xt, stats=None)
    y = _level_output(npu, bd, out, "y_out")
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    if kind == "up":
        xq = F.interpolate(xq, scale_factor=2.0, mode="nearest")
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), stride=2 if kind == "down" else 1,
                   padding=1)
    assert _rel(y, ref) < 8e-3


@pytest.mark.slow
def test_unet(npu: VisionDevice) -> None:
    """SD 1.5's UNet step (``model.UNet``: every level on the design, one dispatch) on random latents at t 500 with a
    random context: within 1.5e-2 of the fp32 reference; a second run the same."""
    import time

    from artlite.model_config import sd15_dir
    from models import is_diffusers
    from models.sd15 import reference
    from models.sd15.host import resnet_time_biases
    from models.sd15.model import UNet
    path = sd15_dir()
    if not is_diffusers(path):
        pytest.skip(f"{path}: not present")
    u = reference.UNet(f"{path}/unet")
    torch.manual_seed(0)
    x, ctx = torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768)
    t0 = time.perf_counter()
    net = UNet(npu, u.w, eps=u.cfg["norm_eps"])
    print(f"UNet: {net.bd.s.n} ops, built in {time.perf_counter() - t0:.1f} s")
    net.set_time(resnet_time_biases(u.w, 500.0))
    net.set_context(ctx)
    y = net(x)
    ts = []
    for _ in range(3):
        t1 = time.perf_counter()
        y2 = net(x)
        ts.append(time.perf_counter() - t1)
    err = _rel(y, u(x, 500.0, ctx))
    print(f"UNet: rel err {err:.3e}; {np.median(ts) * 1e3:.1f} ms per step")
    assert torch.equal(y, y2)
    assert err < 1.5e-2
