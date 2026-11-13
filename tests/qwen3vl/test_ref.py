"""Qwen3-VL's image sizes and pixels (models/qwen3vl/host.py) and the torch encoder (tests/qwen3vl/ref.py) against
llama.cpp's libmtmd (no NPU): the encoder input's size and pixels identical (a JPEG's within its decoders'
differences), the image chunk's tokens and positions; the encoder's embeddings within its own sensitivity across 27
layers (llama.cpp's activations bf16 against float64: ~1.5%)."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import MODELS_DIR, ClipConfig, qwen_gguf
from models.qwen3vl import host as Q
from models.sequence import load_image

from ..goldens.media import golden
from . import ref as V

IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "sd15-astronaut.jpg")
MODEL = qwen_gguf("Q4_K_M", "35B-A3B")
MMPROJ = os.path.join(MODELS_DIR, "lmstudio-community", "Qwen3.6-35B-A3B-GGUF", "mmproj-Qwen3.6-35B-A3B-BF16.gguf")
PROMPT = "<__media__>Describe the image."
# (the sizes keep the image's aspect: llama.cpp before its PR #30736 letterboxed the others)
CASES = {"astronaut": None, "noise-300x200": (300, 200), "noise-1536x1024": (1536, 1024)}


def _files() -> None:
    for path in (MODEL, MMPROJ):
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")


def _image(case: str, tmp: str) -> str:
    if CASES[case] is None:
        return IMAGE
    w, h = CASES[case]
    path = os.path.join(tmp, f"{case}.png")  # (noise: the resampling exact at every pixel)
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)).save(path)
    return path


def test_sizes() -> None:
    """Multiples of 32 near the aspect, within the token bounds: llama.cpp's choices on a few shapes."""
    assert Q.target_size(512, 512) == (512, 512)
    assert Q.target_size(1000, 700) == (992, 704)  # (31.25 -> 31, 21.875 -> 22)
    assert Q.target_size(4032, 3024) == (2336, 1760)  # (73 x 55 = 4015 tokens: scaled to the 4096 bound)
    assert Q.target_size(16, 16) == (96, 96)  # (8 tokens at least: 3 x 3)
    assert Q.target_size(48, 16) == (160, 64)  # (rounds to 64 x 32: 2 tokens -> 5 x 2)
    for w, h in [(4032, 3024), (8000, 300), (33, 2000), (1280, 720)]:
        tw, th = Q.target_size(w, h)
        assert tw % 32 == th % 32 == 0 and Q.MIN_TOKENS <= tw * th // 1024 <= Q.MAX_TOKENS, (w, h, tw, th)


@pytest.mark.parametrize("case", list(CASES))
def test_pixels(case: str, tmp_path: str) -> None:
    """The encoder's input against llama.cpp's inp_raw, the image chunk's tokens and positions."""
    _files()
    path = _image(case, str(tmp_path))
    g = golden(MODEL, MMPROJ, path, PROMPT)  # (inp_raw: the image of entry 0 and 1, the two kernels' convs)
    clip = ClipConfig.from_gguf(GGUFFile(MMPROJ))
    px = Q.normalized(Q.image_pixels(load_image(path)), clip.mean, clip.std)
    ll = g.tensor("inp_raw", 0)
    # identical; a JPEG's decoded pixels within 3 levels (Pillow's libjpeg-turbo vs llama.cpp's stb_image)
    tol = 3.01 / 127.5 if case == "astronaut" else 1e-6
    assert px.shape == ll.shape and np.abs(px - ll).max() <= tol, (px.shape, ll.shape)
    grid = Q.Grid(px.shape[1] // 32, px.shape[2] // 32)
    text, image = g.chunks[:2]
    assert text["text"][-1] == 248053 and image["image"] == grid.tokens and image["n_pos"] == max(grid.gh, grid.gw)
    pos, nxt = grid.positions()
    o = len(text["text"])  # (llama.cpp's [t, x, y, z] from the chunk's first position)
    assert np.array_equal(np.asarray(image["pos"])[:, :3], (pos + o)[:, [0, 2, 1]]) and nxt == image["n_pos"]


@pytest.mark.slow  # (the whole encoder on the CPU, in float64)
def test_encoder() -> None:
    _files()
    g = golden(MODEL, MMPROJ, IMAGE, PROMPT)
    out = V.Encoder(MMPROJ).forward(g.tensor("inp_raw", 0))
    emb = g.embd(0)
    assert float(torch.linalg.norm(out - torch.from_numpy(emb).double()) / np.linalg.norm(emb)) < 2e-2
