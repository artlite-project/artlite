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
from artlite.operators.sd15.seq import ColOps, MemTask
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


def _bands_read(npu: Sd15Device, y: M.Bands) -> torch.Tensor:
    """A banded level's residual -> (nb, COLS, 64, C)."""
    lv = y.lv
    return torch.stack([torch.stack([L.from_ctiles(npu.read(y.x.band(b), COLS * lv.ct(M.C)).reshape(COLS, -1)[c], 64, M.C,
                                                   64) for c in range(COLS)]) for b in range(lv.nb)])


@pytest.mark.slow  # (7 layers' weights converted on the host)
def test_before_merger(npu: Sd15Device, enc: V.Encoder) -> None:
    """The patch embedding and layers 0 .. 6 on two entries of different sizes (the astronaut's overview, 32 x 32
    patches: 64 groups; its first slice, 24 x 44: 66 groups, so 8 bands, the overview's padding masked): each as far
    from the float64 encoder as the CPU at the NPU's rounding points."""
    ents = MV.slice_image(MV.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][:2]
    sim = V.NpuEncoder(MMPROJ)
    entries = [M.Entry(e.shape[1] // 14, e.shape[2] // 14) for e in ents]
    lv = M.vit_level(4 * max(e.bands for e in entries))
    trs, ref = [], []
    for e in ents:
        tr, tf = {}, {}
        sim.forward(e, None, tr)
        enc.forward(e, None, tf)
        trs.append(tr)
        ref.append(tf)
    pos = [V._bf16(sim.w("v.position_embd.weight")[torch.from_numpy(sim.positions(e.ph, e.pw))]) for e in entries]
    bd = Builder(npu)
    x = M.patch_embed(bd, M.patch_weights(lambda n: sim.w(n).float()), entries, [M.patches(e) for e in ents],
                      [torch.nn.functional.pad(p.float(), (0, M.C - M.H)) for p in pos], lv, "x0")
    kv = bd.ddr("kv", M.kv_bytes(lv))
    valid = M.key_valid(entries, lv.nb, "patch")
    for il in range(7):
        y = M.Bands(lv, f"x{il + 1}")
        M.layer_bands(bd, M.layer_weights(lambda n: sim.w(n).float(), il), f"l{il}", x, y, kv=kv, valid=valid)
        x = y
    bd.op({}, {})
    npu.run(bd.transaction())
    slots = [e.patch_slots() for e in entries]
    for key, name in (("x0", "pos_embed"), ("x7", "layer_out-6")):
        got = M.unpack(_bands_read(npu, M.Bands(lv, key)), slots)
        for g, tr, tf in zip(got, trs, ref, strict=True):
            npu_err, cpu_err = _rel(g[:, :M.H], tf[name]), _rel(tr[name], tf[name])
            print(f"{name}: from float64 {npu_err:.2e} (the CPU's {cpu_err:.2e}), NPU vs CPU {_rel(g[:, :M.H], tr[name]):.2e}")
            assert float(g[:, M.H:].abs().max()) == 0
            assert npu_err < 1.1 * cpu_err + 1e-4


def test_window_attention(npu: Sd15Device) -> None:
    """The ViT merger's window attention on random Q, K, V of 4 sources for two token panels per column: within 1.2e-2
    of softmax attention per window and head on the bfp16 operands (Q, K blocked along the dims, V along the 8 windows
    of a panel), P in bf16 and the bfp16 output."""
    from .test_sd15_unet import q as bq

    torch.manual_seed(5)
    n = COLS * 16  # (windows: per column 2 panels of 8)
    x = torch.randn(3, 4, n, M.C)  # (Q, K, V; source; window; channel)
    x[0] *= M.LOG2E / M.HD ** 0.5 * 2
    stride = M.C // 8 * 2 * 72  # (a source's [kb][2][72])
    QA, KA, VA, OA = 0, 4 * stride + 8192, 8 * stride + 16384, 12 * stride + 24576
    cols = [slice(c * 16, (c + 1) * 16) for c in range(COLS)]
    bd = Builder(npu)
    ops = {}
    for t, (base, f) in enumerate(((QA, L.panels), (KA, L.panels), (VA, L.vt_panels))):
        img = bd.image(f"in{t}", np.concatenate([np.concatenate([f(x[t, s, sl]) for s in range(4)]) for sl in cols]))
        ops[t] = {c: ColOps(fills=[MemTask([{"addr": base, "nwords": stride}], ddr=(img + c * 4 * stride, 4 * stride, None))])
                  for c in range(COLS)}
    bd.op(bd.merge(*ops.values()), {})
    wa = M.WindowAttention(q=QA, k=KA, vt=VA, o=OA, stride=stride)
    cmd = bd.image("wcmd", wa.command())
    bd.op({c: wa.col_ops() for c in range(COLS)}, wa.row_ops(cmd))
    bd.op(bd.spill("o", OA, 4 * stride), {})
    npu.run(bd.transaction())
    ob = npu.read("o", COLS * 4 * stride).reshape(COLS, 4, stride)
    o = torch.stack([torch.cat([L.from_panels(ob[c, s], 16, M.C) for c in range(COLS)]) for s in range(4)])  # (s, n, C)
    qq, kk = bq(x[0]), bq(x[1])
    vv = torch.stack([torch.cat([bq(x[2, s, sl].T).T for sl in cols]) for s in range(4)])  # (V^T: blocks over 8 windows)
    ref = torch.zeros(4, n, M.C)
    for h in range(M.HEADS):
        ch = slice(M.HP * h, M.HP * (h + 1))
        sc = torch.einsum("iwd,jwd->wij", qq[:, :, ch], kk[:, :, ch])
        pr = torch.exp2(sc - sc.amax(-1, keepdim=True))
        pr = pr / pr.sum(-1, keepdim=True)
        ref[:, :, ch] = torch.einsum("wij,jwd->iwd", pr, vv[:, :, ch])
    err = _rel(o, ref)
    print(f"window attention: rel err {err:.2e}")
    assert err < 1.2e-2


def _bands_from(bd: Builder, rows: list[torch.Tensor], slots: list, lv: M.VitLevel, name: str) -> M.Bands:
    """Two entries' rows (n_i, 1152) at their slots as a banded level's residual in DDR (records by a copy op per
    band)."""
    tiles = M.pack([torch.nn.functional.pad(r.float(), (0, M.C - M.H)) for r in rows], slots, lv.nb)
    bands = M.Bands(lv, name)
    for b in range(lv.nb):
        bd.image(f"{name}.in.{b}", np.concatenate([L.ctiles(tiles[b, c], 64) for c in range(COLS)]))
        t = Tensor(lv, M.C, bd.mem.alloc(lv.ct(M.C)))
        bd.op(bd.load(f"{name}.in.{b}", t.addr, t.nbytes), {})
        bands.store(bd, b, t, M.ln_records(bd, t, name=f"{name}.rec{b}"))
    bd.op({}, {})
    return bands


def _before_merger(sim: V.NpuEncoder, px: np.ndarray) -> torch.Tensor:
    """The CPU at the NPU's rounding points through layer 6 (raster order)."""
    tr: dict[str, torch.Tensor] = {}
    P, H = sim.patch, sim.hidden
    x = torch.as_tensor(np.asarray(px), dtype=sim.dtype)
    ph, pw = x.shape[1] // P, x.shape[2] // P
    cols = x.reshape(3, ph, P, pw, P).permute(1, 3, 0, 2, 4).reshape(ph * pw, 3 * P * P)
    c = V._bfp(cols) @ V._bfp(sim.w("v.patch_embd.weight").reshape(H, -1)).T + V._bf16(sim.w("v.patch_embd.bias").reshape(-1))
    x = V._bf16(c + V._bf16(sim.w("v.position_embd.weight")[torch.from_numpy(sim.positions(ph, pw))]))
    for il in range(sim.insert + 1):
        x = sim.layer(x, il)
    del tr
    return x


@pytest.mark.slow  # (the CPU encoder's first 7 layers, the merger's weights converted)
def test_merger(npu: Sd15Device, enc: V.Encoder) -> None:
    """The ViT merger on two entries (the astronaut's overview, 32 x 32 patches; its first slice, 24 x 44: 2 bands)
    from the CPU's layer-6 output (at the NPU's rounding points): as far from the float64 merger of that input as the
    CPU's merger."""
    ents = MV.slice_image(MV.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][:2]
    sim = V.NpuEncoder(MMPROJ)
    entries = [M.Entry(e.shape[1] // 14, e.shape[2] // 14) for e in ents]
    nbp = max(e.bands for e in entries)
    xs = [_before_merger(sim, e) for e in ents]
    want = [sim.vit_merger(x, e.ph, e.pw) for x, e in zip(xs, entries, strict=True)]
    exact = [enc.vit_merger(x.double(), e.ph, e.pw) for x, e in zip(xs, entries, strict=True)]
    bd = Builder(npu)
    x = _bands_from(bd, xs, [e.patch_slots() for e in entries], M.vit_level(4 * nbp), "x")
    out = M.Bands(M.vit_level(nbp), "y")
    M.merger(bd, M.merger_weights(lambda n: sim.w(n).float()), "m", x, out)
    bd.op({}, {})
    npu.run(bd.transaction())
    got = M.unpack(_bands_read(npu, out), [e.window_slots() for e in entries])
    for g, e, ex in zip(got, want, exact, strict=True):
        npu_err, cpu_err = _rel(g[:, :M.H], ex), _rel(e, ex)
        print(f"merger: from float64 {npu_err:.2e} (the CPU's {cpu_err:.2e}), NPU vs CPU {_rel(g[:, :M.H], e):.2e}")
        assert float(g[:, M.H:].abs().max()) == 0
        assert npu_err < 1.1 * cpu_err


@pytest.mark.slow  # (the CPU encoders, every weight converted)
def test_encoder(npu: Sd15Device, enc: V.Encoder) -> None:
    """The whole encoder on two entries (the astronaut's overview, 32 x 32 patches; its first slice, 24 x 44) in one
    dispatch: each one's embeddings as far from the float64 encoder's as the CPU at the NPU's rounding points; the
    dispatch's time."""
    import time

    ents = MV.slice_image(MV.load_image(IMAGE), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][:2]
    sim = V.NpuEncoder(MMPROJ)
    entries = [M.Entry(e.shape[1] // 14, e.shape[2] // 14) for e in ents]
    want = [sim.forward(e) for e in ents]
    exact = [enc.forward(e) for e in ents]
    pos = [torch.nn.functional.pad(V._bf16(sim.w("v.position_embd.weight")[torch.from_numpy(sim.positions(e.ph, e.pw))]
                                           ).float(), (0, M.C - M.H)) for e in entries]
    bd = Builder(npu)
    imgs = M.encode(bd, M.weights(lambda n: sim.w(n).float(), sim.layers, sim.insert), entries,
                    [M.patches(e) for e in ents], pos)
    tr = bd.transaction()
    npu.run(tr)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        npu.run(tr)
        ts.append(time.perf_counter() - t0)
    got = M.embeddings(npu, imgs, entries)
    print(f"encoder on 2 entries: {1e3 * min(ts):.1f} ms")
    for g, e, ex in zip(got, want, exact, strict=True):
        npu_err, cpu_err = _rel(g, ex), _rel(e, ex)
        print(f"embeddings: from float64 {npu_err:.2e} (the CPU's {cpu_err:.2e}), NPU vs CPU {_rel(g, e):.2e}")
        assert npu_err < 1.1 * cpu_err
