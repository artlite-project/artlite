"""The SD 1.x UNet on the vision overlay (``models.sd15``: the blocks of ``unet`` and ``level64``, the step of
``model``), run by host-built transactions against torch on random data."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from artlite.operators.design import COLS
from artlite.vision import design as D
from artlite.vision import layout as L
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor
from artlite.vision.levels import ImageLevel

from ..numerics import rel_native
from ..vision.ops_ref import from_level_tokens, level_tokens, q, qconv_w


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


@pytest.mark.parametrize("H, cin, cout", [(8, 1280, 1280), (16, 640, 1280)])
def test_resnet(npu: VisionDevice, H: int, cin: int, cout: int) -> None:
    """diffusers' ResnetBlock2D on the level (GN1 + SiLU, conv1 + time bias, GN2 + SiLU, conv2 + x or the 1x1
    shortcut; the input's GroupNorm records by a copy op): within 1.5e-2 of the fp32 reference."""
    from models.sd15.layers import copy_stats
    from models.sd15.unet import resnet

    from . import ref as reference
    torch.manual_seed(6)
    p = "r."
    w = _rand_resnet(p, cin, cout)
    tb = 0.3 * torch.randn(cout)
    x = torch.randn(2, cin, H, H)
    lv = ImageLevel(H)
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
    y = from_level_tokens(torch.stack([torch.from_numpy(L.from_ctiles(yb[c], lv.m, cout, lv.m)) for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.resnet(w, p, x, eps=1e-5, time_bias=tb)
    assert rel_native(y, ref) < 1.5e-2


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

    from models.sd15.layers import copy_stats
    from models.sd15.unet import transformer

    from . import ref as reference
    torch.manual_seed(8)
    p, C = "t.", 1280
    w = _rand_transformer(p)
    x = torch.randn(2, C, H, H)
    ctx = torch.randn(2, 77, 768)
    q = p + "transformer_blocks.0.attn2."
    lv = ImageLevel(H)
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
    y = from_level_tokens(torch.stack([torch.from_numpy(L.from_ctiles(yb[c], lv.m, C, lv.m)) for c in range(COLS)]), lv.R, lv.Lg)
    ref = reference.UNet.transformer(types.SimpleNamespace(w=w, cfg={"attention_head_dim": 8}), p, x, ctx)
    assert rel_native(y, ref) < 2e-2


def _level_input(bd: Builder, lv: ImageLevel, x: torch.Tensor, name: str) -> Tensor:
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
    return from_level_tokens(torch.stack([torch.from_numpy(L.from_ctiles(yb[c], lv.m, t.C, lv.m)) for c in range(COLS)]), lv.R, lv.Lg)


@pytest.mark.parametrize("H, ch, cs", [(8, 1280, 1280), (16, 1280, 640)])
def test_resnet_cat(npu: VisionDevice, H: int, ch: int, cs: int) -> None:
    """The up path's resnet on the concatenation [h | skip] (the skip from DDR; GN over both, conv1 and the shortcut in
    two K-chunks): within 1.5e-2 of the fp32 reference."""
    from artlite.vision.layers import free
    from models.sd15.layers import copy_stats
    from models.sd15.unet import make_skip, resnet_cat

    from . import ref as reference
    torch.manual_seed(9)
    p, cout = "r.", 1280
    w = _rand_resnet(p, ch + cs, cout)
    tb = 0.3 * torch.randn(cout)
    h, sk = torch.randn(2, ch, H, H), torch.randn(2, cs, H, H)
    lv = ImageLevel(H)
    bd = Builder(npu)
    st = _level_input(bd, lv, sk, "s_in")
    skip = make_skip(bd, st, name="skip", st=((ch + cs) // 32, ch))
    free(bd, st)
    ht = _level_input(bd, lv, h, "h_in")
    ht.rec = copy_stats(bd, ht, name="h.st", st=((ch + cs) // 32, 0))
    out = resnet_cat(bd, w, p, ht, skip, cout=cout, time_bias=tb, eps=1e-5, stats=None)
    y = _level_output(npu, bd, out, "y_out")
    ref = reference.resnet(w, p, torch.cat([h, sk], 1), eps=1e-5, time_bias=tb)
    assert rel_native(y, ref) < 1.5e-2


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
    xt = _level_input(bd, ImageLevel(H), x, "x_in")
    out = (downsample if kind == "down" else upsample)(bd, w, p, xt, stats=None)
    y = _level_output(npu, bd, out, "y_out")
    xq = q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    if kind == "up":
        xq = F.interpolate(xq, scale_factor=2.0, mode="nearest")
    ref = F.conv2d(xq, qconv_w(w[p + "conv.weight"]), w[p + "conv.bias"].bfloat16().float(), stride=2 if kind == "down" else 1,
                   padding=1)
    assert rel_native(y, ref) < 8e-3

