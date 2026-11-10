"""LFM3-VL's image sizes and pixels (models/lfm3vl/host.py) and the torch encoder (tests/lfm3vl/ref.py) against
llama.cpp's libmtmd (no NPU): the encoder input's size and pixels (llama.cpp resizes with torchvision's int16
weights, Pillow with its own: a level apart at most; a JPEG's within its decoders' differences), the image chunk's
tokens and 1-D positions; the encoder's embeddings within llama.cpp's own distance from float64 (~1%)."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, lfm3vl_gguf, lfm3vl_mmproj
from models.lfm3vl import host as Q
from models.sequence import load_image

from . import ref as V

IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "sd15-astronaut.jpg")
MODEL = lfm3vl_gguf()
MMPROJ = lfm3vl_mmproj()
PROMPT = "<__media__>Describe the image."
CASES = {"astronaut": None, "noise-300x200": (300, 200), "noise-1536x1024": (1536, 1024), "noise-1000x700": (1000, 700)}
VISION_START = 125009  # (<|vision_start|>)


def _files() -> None:
    for path in (MODEL, MMPROJ):
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")


def _image(case: str, tmp: str) -> str:
    if CASES[case] is None:
        return IMAGE
    w, h = CASES[case]
    path = os.path.join(tmp, f"{case}.png")
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)).save(path)
    return path


def test_sizes() -> None:
    """Multiples of 32 near the aspect, within the token bounds: transformers' smart_resize on a few shapes."""
    assert Q.target_size(512, 512) == (512, 512)
    assert Q.target_size(1000, 700) == (992, 704)  # (31.25 -> 31, 21.875 -> 22)
    assert Q.target_size(4032, 3024) == (2336, 1760)  # (73 x 55 = 4015 tokens: scaled to the 4096 bound)
    assert Q.target_size(16, 16) == (256, 256)  # (64 tokens at least: 8 x 8)
    assert Q.target_size(208, 208) == (256, 256)  # (6.5 -> 6 (half to even): 36 tokens -> 8 x 8)
    for w, h in [(4032, 3024), (8000, 300), (33, 2000), (1280, 720)]:
        tw, th = Q.target_size(w, h)
        assert tw % 32 == th % 32 == 0 and Q.MIN_TOKENS <= tw * th // 1024 <= Q.MAX_TOKENS, (w, h, tw, th)


def test_config() -> None:
    _files()
    c = ClipConfig.from_gguf(GGUFFile(MMPROJ))
    assert (c.projector, c.hidden, c.heads, c.kv_heads, c.ffn, c.layers, c.patch_size, c.merge) == \
        ("lfm3vl", 1024, 16, 8, 2560, 14, 16, 2)
    assert (c.min_pixels, c.max_pixels, c.resize) == (65536, 16777216, "bicubic")
    assert c.min_pixels // 1024 == Q.MIN_TOKENS


@pytest.mark.parametrize("case", list(CASES))
def test_pixels(case: str, tmp_path: str) -> None:
    """The encoder's input against llama.cpp's inp_raw, the image chunk's tokens and positions."""
    _files()
    path = _image(case, str(tmp_path))
    g = V.golden(MODEL, MMPROJ, path, PROMPT)
    px = Q.normalized(Q.image_pixels(load_image(path)))
    ll = g.tensor("inp_raw", 0)
    # the resize's weights: ~0.6% of the values a level apart, a few in a million two (noise); a JPEG's decoded pixels
    # within 3 more (libjpeg-turbo vs stb_image)
    d = np.rint(np.abs(px - ll) * 127.5) if px.shape == ll.shape else None
    assert d is not None and d.max() <= (5 if case == "astronaut" else 2), (px.shape, ll.shape, d and d.max())
    assert case == "astronaut" or (d > 0).mean() < 0.01
    grid = Q.Grid(px.shape[1] // 32, px.shape[2] // 32)
    text, image = g.chunks[:2]
    assert text["text"][-1] == VISION_START and image["image"] == grid.tokens and image["n_pos"] == grid.tokens
    o = len(text["text"])
    assert np.array_equal(np.asarray(image["pos"])[:, 0], o + np.arange(grid.tokens))


@pytest.mark.slow  # (the whole encoder on the CPU, in float64)
def test_encoder() -> None:
    _files()
    g = V.golden(MODEL, MMPROJ, IMAGE, PROMPT)
    out = V.Encoder(MMPROJ).forward(g.tensor("inp_raw", 0))
    emb = g.embd(0)
    assert float(torch.linalg.norm(out - torch.from_numpy(emb).double()) / np.linalg.norm(emb)) < 2e-2
