"""LFM3-VL's vision encoder on the vision overlay (models.lfm3vl.encoder) against the float64 encoder (ref.Encoder):
each layer alone from the float64 input (its error a small part of its own contribution), the edge masks of the
depthwise conv, pieces of layers against one dispatch (bit for bit), whole images of 1, 2 and 9 bands per merged
token."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig
from artlite.operators.design import COLS
from artlite.overlay import Transaction
from artlite.vision import layout as L
from artlite.vision import windows as W
from artlite.vision.device import VisionDevice
from artlite.vision.layers import alloc
from artlite.vision.vit import Bands, ln_records
from artlite.vision.weights import gguf_getter
from models.lfm3vl import encoder as E
from models.lfm3vl import host as Q
from models.sequence import load_image

from ..numerics import rel
from . import ref as V
from .test_ref import IMAGE, MMPROJ


@pytest.fixture(scope="module")
def npu() -> VisionDevice:
    return VisionDevice()


@pytest.fixture(scope="module")
def mm() -> tuple:
    if not os.path.exists(MMPROJ):
        pytest.skip(f"{MMPROJ} missing")
    g = GGUFFile(MMPROJ)
    clip = ClipConfig.from_gguf(g)
    return gguf_getter(g), clip, E.kinds_of(g.tensor_info, clip.layers)


def _pixels(w: int, h: int) -> np.ndarray:
    return Q.image_pixels(load_image(IMAGE).resize((w, h), Image.Resampling.BICUBIC), min_tokens=1)


def test_masks() -> None:
    """The conv's edge masks against the patch grid: a tap is valid where its neighbour patch is in the image (real
    windows only), its window's lanes all set."""
    gh, gw, nb = 3, 50, 2
    m = W.dw_masks(gh, gw, nb, 1)
    assert m.shape == (COLS, 4, 2, 4, 9, 2) and (m == m[:, :1]).all()
    for c in range(COLS):
        for p in range(2):
            for j, w in enumerate(W.windows_of(nb, 1, c, p)):
                wy, wx = divmod(int(w), gw)
                for q in range(4):
                    for t in range(9):
                        y, x = 2 * wy + q // 2 + t // 3 - 1, 2 * wx + q % 2 + t % 3 - 1
                        ok = w < gh * gw and 0 <= y < 2 * gh and 0 <= x < 2 * gw
                        byte = (int(m[c, 0, p, q, t, j // 4]) >> (8 * (j % 4))) & 0xFF
                        assert byte == (0xFF if ok else 0), (c, p, j, q, t)


@pytest.fixture(scope="module")
def encoder(npu: VisionDevice, mm: tuple) -> E.Encoder:
    get, clip, kinds = mm
    return E.Encoder(npu, E.weights(get, kinds, clip.projection_dim), max_bands=9)


@pytest.mark.parametrize("il, size", [(0, (512, 512)), (2, (512, 512)), (3, (640, 576)), (12, (352, 352)), (4, (480, 480))])
def test_layer(npu: VisionDevice, encoder: E.Encoder, il: int, size: tuple[int, int]) -> None:
    """One layer (a conv: 0, 3, an attention: 2, 12, 4) from the float64 encoder's input (bf16) on the level of an
    image (2, 3, 1, 2 bands; 352 and 480 with padding windows): its error ~1-2% of the layer's own contribution (the
    bfp16 operands of its GEMMs), every token within 6%, the pads zero. The image's inputs are written after the
    layer's build (which writes the level's attention command without key masks)."""
    px = _pixels(*size)
    lay = W.Layout(px.shape[0] // 32, px.shape[1] // 32)
    ph, pw, nb = 2 * lay.gh, 2 * lay.gw, lay.nb
    ref = V.Encoder(MMPROJ)
    x = ref.embed(Q.normalized(px))
    ang = torch.from_numpy(V.rope_angles(ph, pw, E.HD))
    for i in range(il):
        x = ref._layer(x, i, ang, ph, pw)
    xin = x.to(torch.bfloat16).double()
    want = ref._layer(xin, il, ang, ph, pw).numpy()
    sp = lay.slot_patches().reshape(nb, COLS, 64)
    real = lay.slot_windows()[2].reshape(nb, COLS, 64)
    lv = E.level(nb)
    xr = np.zeros((ph * pw, E.C), np.float32)
    xr[:, :E.H] = xin.float().numpy()
    bd = encoder.bd.fork()
    x0 = Bands(lv, encoder.nm.x(il))
    for b in range(nb):
        bd.image(f"t.in.{il}.{nb}.{b}", np.concatenate([L.ctiles(torch.from_numpy(xr[sp[b, c]]), 64) for c in range(COLS)]))
        t = alloc(bd, lv, E.C)
        bd.op(bd.load(f"t.in.{il}.{nb}.{b}", t.addr, t.nbytes), {})
        x0.store(bd, b, t, ln_records(bd, t, name=f"t.in.{il}.rec"))
    bd.op({}, {})
    encoder.net.kv = encoder.kv
    encoder.net.layer(bd, il, nb, encoder.gmax)
    words = bd.finish()
    for n, d in encoder.inputs(lay, px).items():  # (after the build: it writes the level's attention command)
        npu.write(n, d)
    for idx, tag in bd.s.tags:
        a = (int(words[idx]) | int(words[idx + 1]) << 32) + encoder.shift(tag, lay)
        words[idx], words[idx + 1] = a & 0xFFFFFFFF, a >> 32
    npu.run(Transaction(npu.dev, words))
    rows = np.zeros((ph * pw, E.C), np.float32)
    for b in range(nb):
        raw = npu.read(f"{encoder.nm.x(il + 1)}.{b}", COLS * lv.ct(E.C)).reshape(COLS, -1)
        for c in range(COLS):
            v = L.from_ctiles(raw[c], 64, E.C, 64)
            rows[sp[b, c][real[b, c]]] = v[real[b, c]]
    got, d = rows[:, :E.H], want - xin.numpy()
    tok = np.linalg.norm(got - want, axis=1) / np.linalg.norm(d, axis=1)
    print(f"layer {il} ({encoder.kinds[il]}, {nb} bands): {rel(got - xin.numpy(), d):.2e} of its contribution, "
          f"tokens median {np.median(tok):.3f} max {tok.max():.3f}")
    assert float(np.abs(rows[:, E.H:]).max()) == 0
    assert rel(got - xin.numpy(), d) < 0.025 and tok.max() < 0.06


def test_pieces(npu: VisionDevice, mm: tuple, encoder: E.Encoder, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image of 3 bands in pieces (the leading conv layers, then pieces of [attention, conv] pairs run at every
    pair: the residual's images swapped at odd distances) gives bit for bit the embeddings of its program in one
    dispatch of all 14 layers."""
    get, clip, kinds = mm
    px = _pixels(640, 576)
    assert W.Layout(18, 20).nb == 3
    sched = encoder.schedule(3)
    assert len(sched) > 3 and len({i for i, _ in sched}) < len(sched)  # (a piece run at several layers)
    got = encoder(px)
    monkeypatch.setattr(E.Encoder, "layers_per_piece", lambda self, nb: self.nl)
    one = E.Encoder(npu, E.weights(get, kinds, clip.projection_dim), max_bands=3, name="l1")
    assert len(one.schedule(3)) == 3
    assert np.array_equal(one(px), got)


def _token_errors(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(got, np.float64) - want, axis=1) / np.linalg.norm(want, axis=1)


@pytest.mark.parametrize("size", [(352, 352), (512, 512), (1152, 1024)])
def test_encoder(mm: tuple, encoder: E.Encoder, size: tuple[int, int]) -> None:
    """An image of 1, 2 and 9 bands against the float64 encoder per merged token: median within 5%, p90 within 12%
    (a few tokens further: patches whose residual cancels to a fraction of their neighbours' before the projector's
    LayerNorm, ``encoder``'s docstring); the same embeddings again."""
    px = _pixels(*size)
    got = encoder(px)
    want = V.Encoder(MMPROJ).forward(Q.normalized(px)).numpy()
    e = _token_errors(got, want)
    print(f"{size}: per merged token median {np.median(e):.4f}, p90 {np.quantile(e, 0.9):.4f}, max {e.max():.3f}; "
          f"total {rel(got, want):.4f}")
    assert np.median(e) < 0.05 and np.quantile(e, 0.9) < 0.12
    assert np.array_equal(encoder(px), got)
