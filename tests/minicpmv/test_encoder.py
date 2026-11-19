"""MiniCPM-V 4.7's vision encoder on the vision overlay (models.minicpmv.encoder) against the encoder at the same
rounding points on the CPU (ref.NpuEncoder), on real activations: the float64 encoder's residual after the
ViT merger of the astronaut's overview and of its mirror image (two entries of 16 x 16 tokens)."""


import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, minicpmv_mmproj
from artlite.vision import layout as L
from artlite.vision import vit
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import Tensor
from artlite.vision.levels import ImageLevel
from artlite.vision.seq import ColOps, MemTask
from models.minicpmv import encoder as M
from models.minicpmv import host as MV

from ..common import ASTRONAUT
from ..model_files import require
from ..numerics import bf16_like, rel
from ..vision.ops_ref import from_level_tokens, level_tokens
from . import ref as V
from .files import MMPROJ

COLS = 8


@pytest.fixture(scope="module")
def enc() -> V.Encoder:
    require(MMPROJ)
    return V.Encoder(MMPROJ)


def _entries(enc: V.Encoder, il: int) -> list[torch.Tensor]:
    """The two entries' residuals (256, 1152) before layer il (> the merger's: layer 7's the merger's output)."""
    ov = MV.slice_image(MV.load_image(ASTRONAUT), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][0]
    out = []
    for px in (ov, np.ascontiguousarray(ov[:, :, ::-1])):
        tr: dict[str, torch.Tensor] = {}
        enc.forward(px, None, tr)
        out.append(tr["vit_merger_ds_out" if il == enc.insert + 1 else f"layer_out-{il - 1}"])
    return out


def _level_in(x: list[torch.Tensor]) -> torch.Tensor:
    """Two entries (256 tokens, 1152) -> per column its tokens (COLS, 64, 1280): an entry's token t as pixel (t // 16,
    t % 16) of the vision overlay's 16 x 16 level."""
    img = torch.stack([torch.nn.functional.pad(e.float(), (0, M.C - M.H)).T.reshape(M.C, 16, 16) for e in x])
    return level_tokens(img, 2, 2)


def _level_out(y: torch.Tensor) -> list[torch.Tensor]:
    img = from_level_tokens(y, 2, 2)  # (2, C, 16, 16)
    return [img[i].reshape(M.C, 256).T for i in range(2)]


def test_plan() -> None:
    """An image's entries as dispatches: those of at most 64 groups in pairs, the others and an odd one alone (joint:
    both images' keys one sequence, each image's queries over all of them)."""
    sizes = [(32, 32), (24, 44), (32, 32), (24, 40), (28, 40)]  # (64, 66, 64, 60, 70 groups)
    entries = [M.Entry(h, w) for h, w in sizes]
    assert M.plan(entries) == [[0, 2], [3], [1], [4]]
    pair, lone = M.place([entries[0], entries[3]]), M.place([entries[1]])
    assert [(e.img, e.joint) for e in pair] == [(0, False), (1, False)] and lone[0].joint
    v0, v1 = M.key_valid(pair, M.PRE.nb, "patch")
    assert v0.sum() == 64 * 16 and v1.sum() == 60 * 16
    j0, j1 = M.key_valid(lone, M.PRE.nb, "patch")
    assert np.array_equal(j0, j1) and j0.size == 2 * v0.size and j0.sum() == 66 * 16
    with pytest.raises(AssertionError):
        M.place([entries[0], entries[1]])  # (66 groups: not in one image)
    with pytest.raises(AssertionError):
        M.Entry(32, 68)  # (136 groups)


def test_layer(npu: VisionDevice, enc: V.Encoder) -> None:
    """One layer (10) on two entries: as far from the float64 layer as the CPU at the NPU's rounding points (both 1.7e-3;
    1.5e-3 apart: their orders of summation, exp2, GELU's table), the pad channels zero."""
    il = 10
    xs = [bf16_like(x) for x in _entries(enc, il)]
    sim = V.NpuEncoder(MMPROJ)
    want = [sim.layer(x, il) for x in xs]
    w = M.layer_weights(lambda n: sim.w(n).float(), il)
    lv = ImageLevel(16)
    bd = Builder(npu)
    xc = _level_in(xs)
    xt = Tensor(lv, M.C, bd.mem.alloc(lv.ct(M.C)))
    bd.image("x_in", np.concatenate([L.ctiles(xc[c], lv.m) for c in range(COLS)]))
    bd.op(bd.load("x_in", xt.addr, lv.ct(M.C)), {})
    rec = M.ln_records(bd, xt, name="x.rec")
    kv = bd.ddr("kv", M.kv_bytes(lv))
    y, _ = M.layer(bd, w, f"l{il}", xt, rec, kv_ddr=kv, keys=M.Keys("attn"))
    bd.op(bd.spill("y_out", y.addr, lv.ct(M.C)), {})
    npu.run(bd.transaction())
    yb = npu.read("y_out", COLS * lv.ct(M.C)).reshape(COLS, -1)
    got = _level_out(torch.stack([torch.from_numpy(L.from_ctiles(yb[c], lv.m, M.C, lv.m)) for c in range(COLS)]))
    for g, e, x in zip(got, want, xs, strict=True):
        exact = enc._layer(x.double(), il)
        print(f"NPU vs CPU at its rounding points {rel(g[:, :M.H], e):.2e}; vs float64 {rel(g[:, :M.H], exact):.2e} "
              f"(the CPU's {rel(e, exact):.2e})")
        assert float(g[:, M.H:].abs().max()) == 0
        assert rel(g[:, :M.H], exact) < 1.1 * rel(e, exact)


def _bands_in(bd: Builder, x: list[torch.Tensor], lv: ImageLevel, name: str) -> M.Bands:
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


def _bands_out(npu: VisionDevice, y: M.Bands) -> list[torch.Tensor]:
    lv = y.lv
    yb = [npu.read(y.x.band(b), COLS * lv.ct(M.C)).reshape(COLS, -1) for b in range(lv.nb)]
    yc = torch.stack([torch.cat([torch.from_numpy(L.from_ctiles(yb[b][c], lv.T, M.C, lv.m)) for b in range(lv.nb)]) for c in range(COLS)])
    img = from_level_tokens(yc, lv.R, lv.Lg)
    return [img[i].reshape(M.C, 1024).T for i in range(2)]


def test_layer_bands(npu: VisionDevice, enc: V.Encoder) -> None:
    """A layer before the ViT merger (3) on two entries of 32 x 32 tokens, the level in 4 bands: as far from the float64
    layer as the CPU at the NPU's rounding points (both ~8.6e-3), the pad channels zero."""
    il = 3
    ov = MV.slice_image(MV.load_image(ASTRONAUT), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0][0]
    xs = []
    for px in (ov, np.ascontiguousarray(ov[:, :, ::-1])):
        tr: dict[str, torch.Tensor] = {}
        enc.forward(px, None, tr)
        xs.append(bf16_like(tr[f"layer_out-{il - 1}"]))
    sim = V.NpuEncoder(MMPROJ)
    want = [sim.layer(x, il) for x in xs]
    lv = ImageLevel(32, 4)
    bd = Builder(npu)
    x = _bands_in(bd, xs, lv, "x")
    out = M.Bands(lv, "y")
    M.layer_bands(bd, M.layer_weights(lambda n: sim.w(n).float(), il), f"l{il}", x, out,
                  kv_ddr=bd.ddr("kv", M.kv_bytes(lv)), keys=M.Keys("attn"))
    bd.op({}, {})  # (the last band's drains)
    npu.run(bd.transaction())
    for g, e, xx in zip(_bands_out(npu, out), want, xs, strict=True):
        exact = enc._layer(xx.double(), il)
        print(f"NPU vs CPU at its rounding points {rel(g[:, :M.H], e):.2e}; vs float64 {rel(g[:, :M.H], exact):.2e} "
              f"(the CPU's {rel(e, exact):.2e})")
        assert float(g[:, M.H:].abs().max()) == 0
        assert rel(g[:, :M.H], exact) < 1.1 * rel(e, exact)


def _pixels(case: str) -> list[np.ndarray]:
    """A dispatch's entries: "pair" the astronaut's overview (32 x 32 patches: 64 groups) and its first slice cut to
    24 x 40 (60 groups: the overview's padding masked); "joint" that slice whole (24 x 44: 66 groups, alone)."""
    ents = MV.slice_image(MV.load_image(ASTRONAUT), ClipConfig.from_gguf(GGUFFile(MMPROJ)))[0]
    return [ents[0], np.ascontiguousarray(ents[1][:, :, :40 * 14])] if case == "pair" else [ents[1]]


def test_window_attention(npu: VisionDevice) -> None:
    """The ViT merger's window attention on random Q, K, V of 4 sources for two token panels per column: within 1.2e-2
    of softmax attention per window and head on the bfp16 operands (Q, K blocked along the dims, V along the 8 windows
    of a panel), P in bf16 and the bfp16 output."""
    from ..vision.ops_ref import q as bq

    torch.manual_seed(5)
    n = COLS * 16  # (windows: per column 2 panels of 8)
    x = torch.randn(3, 4, n, M.C)  # (Q, K, V; source; window; channel)
    x[0] *= L.LOG2E / M.HD ** 0.5 * 2
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
    o = torch.stack([torch.cat([torch.from_numpy(L.from_panels(ob[c, s], 16, M.C)) for c in range(COLS)]) for s in range(4)])  # (s, n, C)
    qq, kk = bq(x[0]), bq(x[1])
    vv = torch.stack([torch.cat([bq(x[2, s, sl].T).T for sl in cols]) for s in range(4)])  # (V^T: blocks over 8 windows)
    ref = torch.zeros(4, n, M.C)
    for h in range(M.HEADS):
        ch = slice(vit.HP * h, vit.HP * (h + 1))
        sc = torch.einsum("iwd,jwd->wij", qq[:, :, ch], kk[:, :, ch])
        pr = torch.exp2(sc - sc.amax(-1, keepdim=True))
        pr = pr / pr.sum(-1, keepdim=True)
        ref[:, :, ch] = torch.einsum("wij,jwd->iwd", pr, vv[:, :, ch])
    err = rel(o, ref)
    print(f"window attention: rel err {err:.2e}")
    assert err < 1.2e-2


@pytest.mark.slow  # (the CPU encoders, every weight converted)
@pytest.mark.parametrize("compression", [16, 4])
def test_encoder(npu: VisionDevice, compression: int) -> None:
    """The whole encoder (``Encoder``: its pair and joint transactions; the 16x file, or the 4x one without the ViT
    merger) on the entries of both of ``_pixels``' cases: their embeddings' median per-token error from the float64
    encoder's on average within 1.1x of the CPU's at the NPU's rounding points, each entry's within 1.2x (not the rms:
    a few fragile tokens hold most of an entry's squared error, any small difference changing them by up to ~90%, so
    an entry's rms is a draw of 0.8-1.9x the CPU's); each dispatch's time."""
    import time

    path = minicpmv_mmproj(compression)
    require(path)
    sim, enc = V.NpuEncoder(path), V.Encoder(path)
    pos = bf16_like(sim.w("v.position_embd.weight"))
    npe = M.Encoder(npu, M.weights(lambda n: sim.w(n).float(), sim.layers, sim.insert, sim.cfg.mean, sim.cfg.std), pos,
                    patch=sim.patch)
    errs = []
    for case in ("pair", "joint"):
        ents = _pixels(case)
        got = npe(ents)
        tr = npe.tr[case == "joint"]
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            npu.run(tr)
            ts.append(time.perf_counter() - t0)
        print(f"encoder ({compression}x), {case} of {len(ents)}: {1e3 * min(ts):.1f} ms")
        for g, e in zip(got, ents, strict=True):
            ex, want = enc.forward(e), sim.forward(e)
            med = [float(((torch.as_tensor(a).double() - ex).norm(dim=1) / ex.norm(dim=1)).median()) for a in (g, want)]
            print(f"embeddings: from float64 median {med[0]:.2e} (the CPU's {med[1]:.2e}), rms {rel(g, ex):.2e} (the "
                  f"CPU's {rel(want, ex):.2e}), NPU vs CPU {rel(g, want):.2e}")
            errs.append(med)
    npu_err, cpu_err = np.mean(errs, 0)
    assert npu_err < 1.1 * cpu_err and all(a < 1.2 * b for a, b in errs), errs
