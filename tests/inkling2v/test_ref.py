"""Inkling 2's image sizes, pixels and rows (models/inkling2v/host.py, encoder.py's host side) and the torch encoder
(tests/inkling2v/ref.py) against llama.cpp's libmtmd (no NPU): the encoder input's size and pixels identical (the
padding -1 on the 0 .. 255 scale; a JPEG's within its decoders' differences), the image chunk's tokens and positions;
the position rows, the padding pixels' corrections, the RoPE rows; the encoder: a layer fed llama.cpp's input gives
llama.cpp's output, the whole within its own sensitivity across 23 layers (llama.cpp's im2col rounds the pixels to
fp16: ~3e-3 at pre_ln, ~4% at the end)."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.model_config import inkling_gguf, inkling_mmproj_of
from models.inkling2v import encoder as E
from models.inkling2v import host as H
from models.sequence import load_image

from ..common import ASTRONAUT
from ..model_files import require
from ..numerics import rel
from . import ref

MODEL = inkling_gguf()
PROMPT = "<|message_user|><__media__><|end_message|><|message_user|><|content_text|>Describe the image.<|end_message|>"
CASES = {"astronaut": None, "noise-300x210": (300, 210), "noise-2750x2410": (2750, 2410)}  # (padded; resized: 4209
# merged tokens padded, over the cap)


def _mmproj() -> str:
    mmproj = inkling_mmproj_of(MODEL)
    require(MODEL, mmproj)
    return mmproj


def _image(case: str, tmp: str) -> str:
    if CASES[case] is None:
        return ASTRONAUT
    w, h = CASES[case]
    path = os.path.join(tmp, f"{case}.png")  # (noise: the resampling exact at every pixel)
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)).save(path)
    return path


def test_sizes() -> None:
    """The image as it is under the cap (its grid padded up), else llama.cpp's multiples of 40 under it."""
    assert H.target_size(512, 512) == (512, 512) and H.grid(np.zeros((512, 512, 3))) == H.Grid(13, 13)
    assert H.target_size(2560, 2560) == (2560, 2560)  # (64 x 64 = 4096 tokens: the cap itself)
    assert H.target_size(2750, 2410) == (2720, 2360)  # (69 x 61 = 4209 tokens padded: 68 x 59 = 4012)
    assert H.target_size(4032, 3024) == (2920, 2200)  # (73 x 55 = 4015 tokens)
    assert H.target_size(2563, 2557) == (2560, 2560)  # (65 x 64 padded: rounded 64 x 64, the cap)
    assert H.target_size(1000, 700, max_tokens=256) == (760, 520)  # (25 x 18 padded: 19 x 13 under 256)
    for w, h in [(4032, 3024), (8000, 300), (33, 2000), (1280, 720), (2563, 2557)]:
        tw, th = H.target_size(w, h)
        assert H.grid(np.zeros((th, tw, 3))).tokens <= H.MAX_TOKENS, (w, h, tw, th)


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.slow) if c == "noise-2750x2410" else c for c in CASES])
def test_pixels(case: str, tmp_path: str) -> None:
    """The encoder's input against llama.cpp's inp_raw, the image chunk's tokens and positions."""
    mmproj = _mmproj()
    path = _image(case, str(tmp_path))
    g = ref.golden(MODEL, mmproj, path, PROMPT)
    px = H.image_pixels(load_image(path))
    got, ll = ref.pixels(px), g.tensor("inp_raw", 0)
    # identical; a JPEG's decoded pixels within 3 levels (Pillow's libjpeg-turbo vs llama.cpp's stb_image); a resized
    # one's within Pillow's and llama.cpp's bicubic filters' rounding
    tol = {"astronaut": 3.01 / 127.5, "noise-2750x2410": 1.01 / 127.5}.get(case, 1e-6)
    assert got.shape == ll.shape and np.abs(got - ll).max() <= tol, (got.shape, ll.shape)
    grid = H.grid(px)
    text, image = g.chunks[:2]
    assert text["text"][-1] == 200005 and image["image"] == grid.tokens and image["n_pos"] == grid.tokens
    o = len(text["text"])
    assert np.array_equal(np.asarray(image["pos"])[:, 0], o + np.arange(grid.tokens))


def test_rows() -> None:
    """The host's rows against the reference's: the position rows (float32, llama.cpp's interpolation) at the
    reference's float64 within 1e-6, the padding pixels' corrections the patch weights' sums over them, the RoPE rows
    the reference's angles; the RoPE heads' rows keep every q . k."""
    mmproj = _mmproj()
    enc = ref.Encoder(mmproj)
    table = enc.w("v.position_embd.weight").numpy()
    for ph, pw in ((26, 26), (32, 32), (40, 52), (4, 90)):
        assert rel(E.pos_rows(table.astype(np.float32), ph, pw), ref.pos_rows(table, ph, pw)) < 1e-6, (ph, pw)
    wp = enc.w("v.patch_embd.weight").numpy().reshape(-1, 3, 20, 20)
    sums = E.patch_sums(wp)
    h, w = 95, 113  # (padded to 120 x 120: the last patch rows / columns partly or wholly padding)
    corr = E.pad_rows(sums, h, w).reshape(6, 6, -1)
    pad = np.ones((120, 120), bool)
    pad[:h, :w] = False
    for i, j in ((0, 0), (4, 2), (5, 3), (1, 5), (4, 5), (5, 5)):
        m = pad[20 * i:20 * i + 20, 20 * j:20 * j + 20]
        want = -wp[:, :, m].sum(-1).sum(1) / 127.5
        assert np.allclose(corr[i, j], want, rtol=1e-12, atol=1e-12), (i, j)
    vt = E.tower(1024, 16, 4096, 1e-5)
    rows, cls = E.rope_rows(6, 8, vt)
    ang = ref.rope_angles(6, 8)
    assert np.array_equal(cls[:32], np.ones(32)) and not cls[40:].any()
    assert np.allclose(rows[:, :32], np.cos(ang[1:]), atol=1e-7) and np.allclose(rows[:, 40:72], np.sin(ang[1:]), atol=1e-7)
    q, k = np.random.default_rng(0).standard_normal((2, 1024))
    qi, ki = (E.rope_heads(x, vt) for x in (q, k))
    assert np.isclose(qi @ ki, q @ k) and E.rope_index(range(1), 64, 80)[[0, 1, 40, 41, 31, 71, 32]].tolist() == \
        [0, 2, 1, 3, 62, 63, -1]


@pytest.mark.slow  # (the whole encoder on the CPU, in float64)
def test_encoder() -> None:
    """The reference on llama.cpp's pixels: the embeddings (class token, patches, positions) and pre_ln at llama.cpp's
    (its im2col's fp16 pixels: ~3e-3 after pre_ln's mean), layer 0 fed llama.cpp's pre_ln its output to 1e-6, the
    embeddings within the network's sensitivity (~4%)."""
    mmproj = _mmproj()
    g = ref.golden(MODEL, mmproj, ASTRONAUT, PROMPT, names="embeddings|pre_ln|layer_out")
    enc = ref.Encoder(mmproj)
    tr: dict[str, torch.Tensor] = {}
    out = enc.forward(g.tensor("inp_raw", 0), tr).numpy()
    assert rel(tr["embeddings"], g.tensor("embeddings", 0)) < 1.5e-3 and rel(tr["pre_ln"], g.tensor("pre_ln", 0)) < 5e-3
    ang = torch.from_numpy(ref.rope_angles(26, 26))
    x0 = enc._layer(torch.from_numpy(g.tensor("pre_ln", 0).astype(np.float64)), 0, ang)
    assert rel(x0, g.tensor("layer_out", 0, 0)) < 1e-6
    assert rel(out, g.embd(0)) < 0.06
