"""Qwen3-VL's vision encoder on the vision overlay (models.qwen3vl.encoder) against the float64 encoder (ref.Encoder)
and the CPU at the NPU's rounding points (ref.NpuEncoder): the level's layout (no NPU), one layer as far from the
float64 layer as the CPU at the NPU's rounding points, a program in pieces bit for bit the same as in one dispatch,
whole images of 1 .. 9 bands."""

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig
from artlite.operators.design import COLS
from artlite.vision import layout as L
from artlite.vision.build import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import alloc
from artlite.vision.vit import Bands, C, H, Keys, kv_bytes, ln_records, vit_level
from models.qwen3vl import encoder as E
from models.qwen3vl import host as Q
from models.sequence import load_image

from . import ref as V
from .test_ref import IMAGE, MMPROJ


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


@pytest.mark.parametrize("gh, gw", [(11, 11), (16, 16), (18, 20), (64, 64), (1, 8), (3, 5), (24, 48)])
def test_layout(gh: int, gw: int) -> None:
    """Every patch in one real slot; the joint key sequence's valid keys whole groups of 64 but the last live one;
    ``Layout.merged`` the inverse of the windows' tile slots; the layer pieces' renames."""
    lay = E.Layout(gh, gw)
    _, _, real = lay.slot_windows()
    sp = lay.slot_patches()
    assert np.array_equal(np.sort(sp[real]), np.arange(4 * lay.windows))
    groups = lay.keys().reshape(-1, 64)
    live = groups.any(1)
    n = int(np.flatnonzero(live)[-1]) + 1
    assert groups[:n - 1].all() and not live[n:].any() and groups.sum() == 4 * lay.windows
    # (a key's slot: band, column, token panel 2 s + p, lane; valid where the slot is real)
    nb = lay.nb
    p, b, c, k, lane = np.meshgrid(*(np.arange(x) for x in (2, nb, COLS, 4, 8)), indexing="ij")
    assert np.array_equal(real[b, c, (2 * k + p) * 8 + lane].reshape(-1), lay.keys())
    # (the MLP's tiles: quad q's token panel 2 (band % 4) + p of the windows; each window's embedding its index)
    nq = -(-nb // 4)
    q, cc, t = np.meshgrid(np.arange(nq), np.arange(COLS), np.arange(64), indexing="ij")
    j, ln_ = np.divmod(t, 8)
    win = (((j % 2) * nb + 4 * q + j // 2) * COLS + cc) * 8 + ln_
    assert np.array_equal(lay.merged(win[..., None].astype(np.float32))[:, 0], np.arange(lay.windows))
    nm = E.Names("q")
    assert nm.rename(3)("q.l2.k") == "q.l5.k" and nm.rename(3)("q.xa.4") == "q.xb.4"
    assert nm.rename(2)("q.xa.rec.1") == "q.xa.rec.1" and nm.rename(1)("q.park") == "q.park"
    assert E.layers_per_piece(1) == 27 and E.layers_per_piece(32) == 1


@pytest.fixture(scope="module")
def npu() -> VisionDevice:
    return VisionDevice()


@pytest.fixture(scope="module")
def mm() -> tuple:
    if not os.path.exists(MMPROJ):
        pytest.skip(f"{MMPROJ} missing")
    g = GGUFFile(MMPROJ)

    def get(n: str) -> torch.Tensor:
        t = g.tensor(n)
        return t.dequantize().reshape(t.rows, t.cols).float()
    return get, ClipConfig.from_gguf(g)


def _pixels(w: int, h: int) -> np.ndarray:
    return Q.image_pixels(load_image(IMAGE).resize((w, h)))


@pytest.mark.parametrize("size, il", [((352, 352), 10), ((640, 576), 20)], ids=["nb1-l10", "nb3-l20"])
def test_layer(npu: VisionDevice, mm: tuple, size: tuple[int, int], il: int) -> None:
    """One layer (its input the float64 encoder's, bf16) on the level of an image (1 / 3 bands: their keys one dense
    sequence): as far from the float64 layer as the CPU at the NPU's rounding points (~4e-3 / ~8e-3; ~3e-3 apart: their
    orders of summation, exp2, GELU's table, the softmax's reference max), the pad channels zero."""
    get, clip = mm
    px = _pixels(*size)
    lay = E.Layout(px.shape[0] // 32, px.shape[1] // 32)
    enc = V.Encoder(MMPROJ)
    enc.layers = il
    tr: dict[str, torch.Tensor] = {}
    enc.forward(Q.normalized(px, clip.mean, clip.std), tr)
    xin = tr[f"layer_out-{il - 1}"].to(torch.bfloat16).double()
    ph, pw = 2 * lay.gh, 2 * lay.gw
    ang = torch.from_numpy(V.rope_angles(ph, pw))
    exact = enc._layer(xin, il, ang).numpy()
    nb, lv, nm = lay.nb, vit_level(lay.nb), E.Names(f"t{il}")
    sp = lay.slot_patches().reshape(nb, COLS, 64)
    _, _, real = lay.slot_windows()
    bd = Builder(npu)
    kv = bd.ddr(f"{nm.p}.kv", kv_bytes(lv, Keys("", joint=True, dense=True)))
    xr = np.zeros((ph * pw, C), np.float32)
    xr[:, :H] = xin.numpy()
    x0 = Bands(lv, nm.x(il))
    for b in range(nb):  # (the input's bands and their LayerNorm records)
        bd.image(f"{nm.p}.in.{b}", np.concatenate([L.ctiles(torch.from_numpy(xr[sp[b, c]]), 64) for c in range(COLS)]))
        t = alloc(bd, lv, C)
        bd.op(bd.load(f"{nm.p}.in.{b}", t.addr, t.nbytes), {})
        x0.store(bd, b, t, ln_records(bd, t, name=f"{nm.p}.in.rec"))
    bd.op({}, {})
    E.layers(bd, {il: E.layer_weights(get, il)}, nm, nb, range(il, il + 1), kv)
    bd.op({}, {})
    host = E.Encoder.__new__(E.Encoder)  # (the image's RoPE tables and key masks: ``Encoder.inputs``)
    host.nm, host.pos, host._inputs = nm, get("v.position_embd.weight").numpy(), OrderedDict()
    for n, d in E.Encoder.inputs(host, lay, px).items():
        if n.startswith((f"{nm.p}.rope", f"{nm.p}.attn")):
            npu.alloc(n, d.size)
            npu.write(n, d)
    npu.run(bd.transaction())
    rows = np.zeros((ph * pw, C), np.float32)
    for b in range(nb):
        raw = npu.read(f"{nm.x(il + 1)}.{b}", COLS * lv.ct(C)).reshape(COLS, -1)
        for c in range(COLS):
            v = L.from_ctiles(raw[c], 64, C, 64).numpy()
            rows[sp[b, c][real[b, c]]] = v[real[b, c]]
    sim = V.NpuEncoder(MMPROJ).layer(xin, il, ang, np.where(real, sp, -1).reshape(-1, 8)).numpy()
    got = rows[:, :H]
    print(f"NPU vs float64 {_rel(got, exact):.2e} (the CPU at its rounding points {_rel(sim, exact):.2e}); NPU vs CPU "
          f"{_rel(got, sim):.2e}")
    assert float(np.abs(rows[:, H:]).max()) == 0
    assert _rel(got, exact) < 1.1 * _rel(sim, exact)


@pytest.fixture(scope="module")
def encoder(npu: VisionDevice, mm: tuple) -> E.Encoder:
    get, clip = mm
    return E.Encoder(npu, E.weights(get, clip.layers, clip.projection_dim), get("v.position_embd.weight").numpy(),
                     max_bands=9)


def test_pieces(npu: VisionDevice, mm: tuple, encoder: E.Encoder, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image of 3 bands in pieces of 5 layers (at odd layers too: the residual's images swapped; the rest in a piece
    of 2) gives bit for bit the embeddings of its program in one dispatch of all 27 layers."""
    get, clip = mm
    px = _pixels(640, 576)
    assert E.layers_per_piece(3) == 5 and E.Layout(18, 20).nb == 3
    got = encoder(px)
    monkeypatch.setattr(E, "layers_per_piece", lambda nb, layers=27: layers)
    one = E.Encoder(npu, E.weights(get, clip.layers, clip.projection_dim), get("v.position_embd.weight").numpy(),
                    max_bands=3, name="q1")
    assert len(one.schedule(3)) == 3 and len(encoder.schedule(3)) == 8
    assert np.array_equal(one(px), got)


@pytest.mark.parametrize("size", [(352, 352), (512, 512), (1536, 768)], ids=["nb1", "nb2", "nb9"])
def test_encoder(mm: tuple, encoder: E.Encoder, size: tuple[int, int]) -> None:
    """An image of 1, 2 and 9 bands (4608 patches: 72 key groups): the embeddings within 9% of the float64 encoder's (the
    27 layers' roundings: ~6-8%; the language model's KL against llama.cpp's own run at llama.cpp's level:
    test_npu.py), the same at a second run."""
    _, clip = mm
    px = _pixels(*size)
    got = encoder(px)
    want = V.Encoder(MMPROJ).forward(Q.normalized(px, clip.mean, clip.std)).numpy()
    print(f"{size}: {got.shape[0]} tokens, rel {_rel(got, want):.3f}")
    assert got.shape == want.shape and _rel(got, want) < 0.09
    assert np.array_equal(encoder(px), got)
