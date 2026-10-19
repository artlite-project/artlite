"""MiniCPM-V 4.7's vision encoder on sd15's design (artlite.operators.vision.minicpmv) against the encoder at the same
rounding points on the CPU (minicpmv_ref.NpuEncoder), on real activations: the float64 encoder's residual after the
ViT merger of the astronaut's overview and of its mirror image (two entries of 16 x 16 tokens)."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig
from artlite.operators.sd15 import layout as L
from artlite.operators.sd15.build import Builder, Level
from artlite.operators.sd15.runtime import Sd15Device
from artlite.operators.sd15.unet import Tensor
from artlite.operators.vision import minicpmv as M
from models.qwen35 import minicpmv as MV

from . import minicpmv_ref as V
from .test_minicpmv_ref import IMAGE, MMPROJ
from .test_sd15_unet import from_level_tokens, level_tokens

COLS = 8


@pytest.fixture(scope="module")
def npu() -> Sd15Device:
    return Sd15Device()


@pytest.fixture(scope="module")
def enc() -> V.Encoder:
    if not os.path.exists(MMPROJ):
        pytest.skip(f"{MMPROJ} missing")
    return V.Encoder(MMPROJ)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def _entries(enc: V.Encoder, il: int) -> list[torch.Tensor]:
    """The two entries' residuals (256, 1152) before layer il (> the merger's)."""
    ov = MV.slice_image(MV.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][0]
    out = []
    for px in (ov, np.ascontiguousarray(ov[:, :, ::-1])):
        tr: dict[str, torch.Tensor] = {}
        enc.forward(px, None, tr)
        out.append(tr[f"layer_out-{il - 1}"])
    return out


def _level_in(x: list[torch.Tensor]) -> torch.Tensor:
    """Two entries (256 tokens, 1152) -> per column its tokens (COLS, 64, 1280): an entry's token t as pixel (t // 16,
    t % 16) of sd15's 16 x 16 level."""
    img = torch.stack([torch.nn.functional.pad(e.float(), (0, M.C - M.H)).T.reshape(M.C, 16, 16) for e in x])
    return level_tokens(img, 2, 2)


def _level_out(y: torch.Tensor) -> list[torch.Tensor]:
    img = from_level_tokens(y, 2, 2)  # (2, C, 16, 16)
    return [img[i].reshape(M.C, 256).T for i in range(2)]


def test_layer(npu: Sd15Device, enc: V.Encoder) -> None:
    """One layer (10) on two entries: within 3e-3 of the CPU at the NPU's rounding points (measured 1.5e-3: their orders
    of summation, exp2, GELU's table; both 1.7e-3 from float64), the pad channels zero."""
    il = 10
    xs = [V._bf16(x) for x in _entries(enc, il)]
    sim = V.NpuEncoder(MMPROJ)
    want = [sim.layer(x, il) for x in xs]
    w = M.layer_weights(lambda n: sim.w(n).float(), il)
    lv = Level(16)
    bd = Builder(npu)
    xc = _level_in(xs)
    xt = Tensor(lv, M.C, bd.mem.alloc(lv.ct(M.C)))
    bd.image("x_in", np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load("x_in", xt.addr, lv.ct(M.C)), {})
    rec = M.ln_records(bd, xt, name="x.rec")
    kv = bd.ddr("kv", M.kv_bytes(lv))
    y, _ = M.layer(bd, w, f"l{il}", xt, rec, kv=kv)
    bd.op(bd.spill("y_out", y.addr, lv.ct(M.C)), {})
    npu.run(bd.transaction())
    yb = npu.read("y_out", COLS * lv.ct(M.C)).reshape(COLS, -1)
    got = _level_out(torch.stack([L.from_ctiles(yb[c], lv.m, M.C, lv.m) for c in range(COLS)]))
    for g, e, x in zip(got, want, xs, strict=True):
        exact = enc._layer(x.double(), il)
        print(f"NPU vs CPU at its rounding points {_rel(g[:, :M.H], e):.2e}; vs float64 {_rel(g[:, :M.H], exact):.2e} "
              f"(the CPU's {_rel(e, exact):.2e})")
        assert float(g[:, M.H:].abs().max()) == 0
        assert _rel(g[:, :M.H], e) < 3e-3
