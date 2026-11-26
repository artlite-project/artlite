"""Inkling 2's vision encoder on the vision overlay (models.inkling2v.encoder) against the float64 encoder (ref.Encoder)
and the CPU at the overlay's rounding points (ref.NpuEncoder): the layout with its class token (no NPU), the patch
embedding and pre_ln of a padded image (the padding pixels' corrections, pre_ln's bf16 C tiles), one layer as far from
the float64 layer as the CPU at the overlay's rounding points, a program in pieces bit for bit the same as in one
dispatch, whole images of 2 and 5 bands (per-token statistics: as close as the CPU at the overlay's rounding points)."""

from collections import OrderedDict

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, inkling_gguf, inkling_mmproj_of
from artlite.operators.design import COLS
from artlite.vision import layout as L
from artlite.vision import windows as W
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import alloc
from artlite.vision.vit import Bands, kv_bytes, ln_records, vit_level
from artlite.vision.weights import gguf_getter
from models.inkling2v import encoder as E
from models.inkling2v import host as H
from models.sequence import load_image

from ..common import ASTRONAUT
from ..model_files import require
from ..numerics import rel, rel_rows
from . import ref

MODEL = inkling_gguf()


@pytest.mark.parametrize("gh, gw", [(13, 13), (16, 8), (16, 16), (20, 26), (1, 1), (3, 5), (64, 64)])
def test_layout(gh: int, gw: int) -> None:
    """The class token at patch 0 of window ``windows`` (an image of 128 k windows takes a band more): the keys every
    patch's and the class token's, whole groups of 64 but the last live one (``ops.Attention(dense=True)``); its slot's
    panels zeros (the constant input too), every other slot's constant input 1; its rows the class token's."""
    lay = W.Layout(gh, gw, cls=True)
    assert lay.nb == -(-(gh * gw + 1) // 128)
    keys = lay.keys()
    groups = keys.reshape(-1, 64)
    live = groups.any(1)
    n = int(np.flatnonzero(live)[-1]) + 1
    assert groups[:n - 1].all() and not live[n:].any() and keys.sum() == 4 * gh * gw + 1
    b, c, t = lay.cls_slot()
    p, lane = divmod(t, 8)
    assert keys[((((p * lay.nb + b) * COLS + c) * 4) * 8) + lane] and not lay.slot_windows()[2][b, c, t]
    px = np.random.default_rng(0).integers(0, 256, (40 * gh, 40 * gw, 3), np.uint8)
    pan = W.patch_panels(lay, px, "t", kp=E.KP, one=E.ONE, patch=20)[f"t.pix.{b}"].reshape(COLS, -1)
    v = L.from_panels(pan[c], 64, E.KP)
    assert not v[t].any() and (np.delete(v[:, E.ONE], t) == 1).all()
    rows = W.band_rows(lay, np.ones((4 * gh * gw, 80), np.float32), "r", np.full(80, 2, np.float32))
    r = L.from_ctiles(rows[f"r.{b}"].reshape(COLS, -1)[c], 64, 80, 64)
    assert (r[t] == 2).all() and (np.delete(r, t, 0) == 1).all()


@pytest.fixture(scope="module")
def mm() -> tuple:
    mmproj = inkling_mmproj_of(MODEL)
    require(mmproj)
    g = GGUFFile(mmproj)
    c = ClipConfig.from_gguf(g)
    vt = E.tower(c.hidden, c.heads, c.ffn, c.eps)
    return mmproj, E.weights(gguf_getter(g), c.layers, c.projection_dim, vt), vt


def _vblocks(lay: W.Layout) -> np.ndarray:
    """The NPU's key panels (blocks of 8 slots) as ``ref.NpuEncoder``'s tokens: the class token 0, patch i 1 + i, -1 a
    padding slot."""
    _, _, real = lay.slot_windows()
    b, c, t = lay.cls_slot()
    tok = np.where(real, 1 + lay.slot_patches(), -1)
    tok[b, c, t] = 0
    return tok.reshape(-1, 8)


def _rows(npu: VisionDevice, name: str, lay: W.Layout, C: int) -> np.ndarray:
    """A banded residual's DDR images -> the class token's row, then the patches' (raster order)."""
    lv = vit_level(lay.nb)
    sp, (_, _, real) = lay.slot_patches().reshape(lay.nb, COLS, 64), lay.slot_windows()
    out = np.zeros((1 + 4 * lay.windows, C), np.float32)
    cb, cc, ct = lay.cls_slot()
    for b in range(lay.nb):
        raw = npu.read(f"{name}.{b}", COLS * lv.ct(C)).reshape(COLS, -1)
        for c in range(COLS):
            v = L.from_ctiles(raw[c], 64, C, 64)
            out[1 + sp[b, c][real[b, c]]] = v[real[b, c]]
            if (b, c) == (cb, cc):
                out[0] = v[ct]
    return out


def _host(mm: tuple, nm: W.Names) -> E.Encoder:
    """An encoder's host side alone (``Encoder.inputs``: the size's rows, the pixels' panels)."""
    host = E.Encoder.__new__(E.Encoder)
    host.nm, host.w, host.vt, host._inputs = nm, mm[1], mm[2], OrderedDict()
    return host


def test_patch(npu: VisionDevice, mm: tuple) -> None:
    """The patch embedding and pre_ln of an image of 470 x 375 pixels (480 x 400 padded: its padding pixels' corrections
    in the position rows) on 2 bands against the float64 pre_ln: as close as the CPU at the overlay's rounding points
    (bf16 x0: ~2e-3), the class token's row too; the pad channels zero."""
    mmproj, w, vt = mm
    im = load_image(ASTRONAUT).resize((470, 375), Image.Resampling.BICUBIC)
    px = np.asarray(im, np.uint8)
    lay = W.Layout(10, 12, cls=True)
    nm = W.Names("tp")
    bd = Builder(npu)
    E.patch_embed(bd, w, nm, lay.nb, vt)
    bd.op({}, {})
    for n, d in _host(mm, nm).inputs(lay, px).items():
        if n.startswith((f"{nm.p}.pix", f"{nm.p}.pos")):
            npu.write(n, d)
    npu.run(bd.transaction())
    got = _rows(npu, nm.x(0), lay, vt.C)
    tr: dict[str, torch.Tensor] = {}
    ref.Encoder(mmproj).forward(ref.pixels(px), tr, layers=0)
    sim: dict[str, torch.Tensor] = {}
    ref.NpuEncoder(mmproj).forward(ref.pixels(px), _vblocks(lay), sim, layers=0)
    want = tr["pre_ln"].numpy()
    print(f"pre_ln: NPU vs float64 {rel(got[:, :vt.H], want):.2e} (the CPU at its rounding points "
          f"{rel(sim['pre_ln'], want):.2e}), class token {rel(got[0, :vt.H], want[0]):.2e}")
    assert float(np.abs(got[:, vt.H:]).max()) == 0
    assert rel(got[:, :vt.H], want) < 1.2 * rel(sim["pre_ln"], want) and rel(got[0, :vt.H], want[0]) < 4e-3


@pytest.mark.parametrize("size, il", [((512, 512), 0), ((800, 600), 12)], ids=["nb2-l0", "nb3-l12"])
def test_layer(npu: VisionDevice, mm: tuple, size: tuple[int, int], il: int) -> None:
    """One layer (its input the float64 encoder's, bf16) on the level of an image (2 / 3 bands: their keys one dense
    sequence with the class token's): as far from the float64 layer as the CPU at the overlay's rounding points, the
    pad channels zero."""
    mmproj, w, vt = mm
    px = H.image_pixels(load_image(ASTRONAUT).resize(size))
    lay = W.Layout(-(-px.shape[0] // 40), -(-px.shape[1] // 40), cls=True)
    enc = ref.Encoder(mmproj)
    tr: dict[str, torch.Tensor] = {}
    enc.forward(ref.pixels(px), tr, layers=il)
    xin = (tr[f"layer_out-{il - 1}"] if il else tr["pre_ln"]).to(torch.bfloat16).double()
    ph, pw = 2 * lay.gh, 2 * lay.gw
    ang = torch.from_numpy(ref.rope_angles(ph, pw))
    exact = enc._layer(xin, il, ang).numpy()
    nb, lv, nm = lay.nb, vit_level(lay.nb), W.Names(f"tl{il}")
    sp = lay.slot_patches().reshape(nb, COLS, 64)
    cb, cc, ct = lay.cls_slot()
    bd = Builder(npu)
    kv = bd.ddr(f"{nm.p}.kv", kv_bytes(lv, E.keys_of("", vt)))
    xr = np.zeros((1 + ph * pw, vt.C), np.float32)
    xr[:, :vt.H] = xin.numpy()
    x0 = Bands(lv, nm.x(il), vt.C)
    for b in range(nb):  # (the input's bands and their LayerNorm records)
        rows = xr[1 + sp[b]]
        if b == cb:
            rows[cc, ct] = xr[0]
        bd.image(f"{nm.p}.in.{b}", np.concatenate([L.ctiles(rows[c], 64) for c in range(COLS)]))
        t = alloc(bd, lv, vt.C)
        bd.op(bd.load(f"{nm.p}.in.{b}", t.addr, t.nbytes), {})
        x0.store(bd, b, t, ln_records(bd, t, name=f"{nm.p}.in.rec", C=vt.C))
    bd.op({}, {})
    E.layers(bd, {il: w[il]}, nm, nb, range(il, il + 1), kv, vt)
    bd.op({}, {})
    for n, d in _host(mm, nm).inputs(lay, px).items():
        if n.startswith((f"{nm.p}.rope", f"{nm.p}.attn")):
            npu.alloc(n, d.size)
            npu.write(n, d)
    npu.run(bd.transaction())
    got = _rows(npu, nm.x(il + 1), lay, vt.C)
    sim = ref.NpuEncoder(mmproj).layer(xin, il, ang, _vblocks(lay)).numpy()
    print(f"NPU vs float64 {rel(got[:, :vt.H], exact):.2e} (the CPU at its rounding points {rel(sim, exact):.2e}); NPU "
          f"vs CPU {rel(got[:, :vt.H], sim):.2e}")
    assert float(np.abs(got[:, vt.H:]).max()) == 0
    assert rel(got[:, :vt.H], exact) < 1.1 * rel(sim, exact)


@pytest.fixture(scope="module")
def encoder(npu: VisionDevice, mm: tuple) -> E.Encoder:
    return E.Encoder(npu, mm[1], max_bands=5)


def test_pieces(npu: VisionDevice, mm: tuple, encoder: E.Encoder, monkeypatch: pytest.MonkeyPatch) -> None:
    """An image of 3 bands in pieces of several layers (at odd layers too: the residual's images swapped) gives bit
    for bit the embeddings of its program in one dispatch of all 23 layers."""
    px = H.image_pixels(load_image(ASTRONAUT).resize((800, 600)))
    assert E.layers_per_piece(3) < 23 and encoder.layout(px).nb == 3
    got = encoder(px)
    monkeypatch.setattr(E, "layers_per_piece", lambda nb, layers=23: layers)
    one = E.Encoder(npu, mm[1], max_bands=3, name="i1")
    assert len(one.schedule(3)) == 3 and len(encoder.schedule(3)) > 3
    assert np.array_equal(one(px), got)


@pytest.mark.parametrize("size", [(512, 512), pytest.param((1024, 768), marks=pytest.mark.slow)], ids=["nb2", "nb5"])
def test_encoder(mm: tuple, encoder: E.Encoder, size: tuple[int, int]) -> None:
    """An image of 2 and 5 bands (the astronaut: padded to 520; 1024 x 768: the plan's 520 tokens) per merged token as
    close to the float64 encoder as the CPU at the overlay's rounding points over all of it (``ref.NpuEncoder``: the
    median within 1.15x, p90 within 1.3x of the CPU's), the same at a second run."""
    mmproj = mm[0]
    im = load_image(ASTRONAUT)
    px = H.image_pixels(im if size == im.size else im.resize(size, Image.Resampling.BICUBIC))
    got = encoder(px)
    assert np.array_equal(encoder(px), got)
    pix = ref.pixels(px)
    want = ref.Encoder(mmproj).forward(pix).numpy()
    sim = ref.NpuEncoder(mmproj).forward(pix, _vblocks(encoder.layout(px))).numpy()
    e, es = rel_rows(np.asarray(got, np.float64), want), rel_rows(np.asarray(sim, np.float64), want)
    med, p90 = (np.median(e), np.median(es)), (np.percentile(e, 90), np.percentile(es, 90))
    print(f"{size}: {got.shape[0]} tokens, per token median {med[0]:.4f} (the CPU's {med[1]:.4f}), p90 {p90[0]:.4f} "
          f"({p90[1]:.4f}), max {e.max():.3f} ({es.max():.3f}); sum {rel(got, want):.4f} ({rel(sim, want):.4f})")
    assert got.shape == want.shape and med[0] < 1.15 * med[1] and p90[0] < 1.3 * p90[1]
