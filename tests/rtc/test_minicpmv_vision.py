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
    """The two entries' residuals (256, 1152) before layer il (> the merger's: layer 7's the merger's output)."""
    ov = MV.slice_image(MV.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][0]
    out = []
    for px in (ov, np.ascontiguousarray(ov[:, :, ::-1])):
        tr: dict[str, torch.Tensor] = {}
        enc.forward(px, None, tr)
        out.append(tr["vit_merger_ds_out" if il == enc.insert + 1 else f"layer_out-{il - 1}"])
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
    """One layer (10) on two entries: as far from the float64 layer as the CPU at the NPU's rounding points (both 1.7e-3;
    1.5e-3 apart: their orders of summation, exp2, GELU's table), the pad channels zero."""
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
        assert _rel(g[:, :M.H], exact) < 1.1 * _rel(e, exact)


@pytest.mark.slow  # (20 layers' weights converted on the host)
def test_layers(npu: Sd15Device, enc: V.Encoder) -> None:
    """The 20 layers after the ViT merger (7 .. 26) in one dispatch on two entries: as far from the float64 layers as
    the CPU at the NPU's rounding points (the two apart by ~1e-2 after 20 layers: each layer's 1.5e-3); the dispatch's
    time."""
    import time

    xs = [V._bf16(x) for x in _entries(enc, 7)]
    sim = V.NpuEncoder(MMPROJ)
    want, exact = list(xs), [x.double() for x in xs]
    for il in range(7, 27):
        want = [sim.layer(x, il) for x in want]
        exact = [enc._layer(x, il) for x in exact]
    lv = Level(16)
    bd = Builder(npu)
    xc = _level_in(xs)
    x = Tensor(lv, M.C, bd.mem.alloc(lv.ct(M.C)))
    bd.image("x_in", np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load("x_in", x.addr, lv.ct(M.C)), {})
    rec = M.ln_records(bd, x, name="x.rec")
    kv = bd.ddr("kv", M.kv_bytes(lv))
    for il in range(7, 27):
        x, rec = M.layer(bd, M.layer_weights(lambda n: sim.w(n).float(), il), f"l{il}", x, rec, kv=kv)
    bd.op(bd.spill("y_out", x.addr, lv.ct(M.C)), {})
    tr = bd.transaction()
    npu.run(tr)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        npu.run(tr)
        ts.append(time.perf_counter() - t0)
    yb = npu.read("y_out", COLS * lv.ct(M.C)).reshape(COLS, -1)
    got = _level_out(torch.stack([L.from_ctiles(yb[c], lv.m, M.C, lv.m) for c in range(COLS)]))
    for g, e, x in zip(got, want, exact, strict=True):
        npu_err, cpu_err = _rel(g[:, :M.H], x), _rel(e, x)
        print(f"20 layers on 2 x 256 tokens: {1e3 * min(ts):.2f} ms ({1e3 * min(ts) / 20:.3f} ms per layer); from "
              f"float64 {npu_err:.2e} (the CPU's {cpu_err:.2e}), NPU vs CPU {_rel(g[:, :M.H], e):.2e}")
        assert npu_err < 1.1 * cpu_err


def _bands_in(bd: Builder, x: list[torch.Tensor], lv: Level, name: str) -> M.Bands:
    """Two entries (1024 tokens, 1152) as a banded 32 x 32 level's residual in DDR (their records by a copy op per
    band)."""
    img = torch.stack([torch.nn.functional.pad(e.float(), (0, M.C - M.H)).T.reshape(M.C, 32, 32) for e in x])
    xc = level_tokens(img, lv.R, lv.Lg)  # (COLS, nb T, C): band b = tokens [T b, T (b + 1))
    bands = M.Bands(lv, name)
    for b in range(lv.nb):
        bd.image(f"{name}.in.{b}", np.concatenate([L.ctiles(xc[c, lv.T * b:lv.T * (b + 1)], lv.m) for c in range(COLS)]))
        t = Tensor(lv, M.C, bd.mem.alloc(lv.ct(M.C)))
        bd.op(bd.load(f"{name}.in.{b}", t.addr, t.nbytes), {})
        bands.store(bd, b, t, M.ln_records(bd, t, name=f"{name}.rec{b}"))
    bd.op({}, {})
    return bands


def _bands_out(npu: Sd15Device, y: M.Bands) -> list[torch.Tensor]:
    lv = y.lv
    yb = [npu.read(y.x.band(b), COLS * lv.ct(M.C)).reshape(COLS, -1) for b in range(lv.nb)]
    yc = torch.stack([torch.cat([L.from_ctiles(yb[b][c], lv.T, M.C, lv.m) for b in range(lv.nb)]) for c in range(COLS)])
    img = from_level_tokens(yc, lv.R, lv.Lg)
    return [img[i].reshape(M.C, 1024).T for i in range(2)]


def test_layer_bands(npu: Sd15Device, enc: V.Encoder) -> None:
    """A layer before the ViT merger (3) on two entries of 32 x 32 tokens, the level in 4 bands: as far from the float64
    layer as the CPU at the NPU's rounding points (both ~8.6e-3), the pad channels zero."""
    il = 3
    ov = MV.slice_image(MV.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][0]
    xs = []
    for px in (ov, np.ascontiguousarray(ov[:, :, ::-1])):
        tr: dict[str, torch.Tensor] = {}
        enc.forward(px, None, tr)
        xs.append(V._bf16(tr[f"layer_out-{il - 1}"]))
    sim = V.NpuEncoder(MMPROJ)
    want = [sim.layer(x, il) for x in xs]
    lv = Level(32, 4)
    bd = Builder(npu)
    x = _bands_in(bd, xs, lv, "x")
    out = M.Bands(lv, "y")
    M.layer_bands(bd, M.layer_weights(lambda n: sim.w(n).float(), il), f"l{il}", x, out,
                  kv=bd.ddr("kv", M.kv_bytes(lv)))
    bd.op({}, {})  # (the last band's drains)
    npu.run(bd.transaction())
    for g, e, xx in zip(_bands_out(npu, out), want, xs, strict=True):
        exact = enc._layer(xx.double(), il)
        print(f"NPU vs CPU at its rounding points {_rel(g[:, :M.H], e):.2e}; vs float64 {_rel(g[:, :M.H], exact):.2e} "
              f"(the CPU's {_rel(e, exact):.2e})")
        assert float(g[:, M.H:].abs().max()) == 0
        assert _rel(g[:, :M.H], exact) < 1.1 * _rel(e, exact)
