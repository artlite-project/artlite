"""MiniCPM-V 4.7's vision references (tests/minicpmv/ref.py) and slicing (models/minicpmv/host.py) against
llama.cpp's libmtmd on an image (no NPU): the entries' pixels identical (where llama.cpp's float grid choice agrees
with the official; a JPEG's within its decoders' differences), the image chunk's tokens; the float64 torch encoder per
entry (the overview, the slices: within the encoder's own sensitivity across 27 layers and two merges, ~0.5-1%), with
the separator rows between the entries."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, minicpmv_gguf
from models.minicpmv import host as M

from ..common import ASTRONAUT
from ..goldens.media import VisionGolden, golden
from ..model_files import require
from ..numerics import rel
from . import ref as V
from .files import MMPROJ, NAMES


@pytest.fixture(scope="module")
def gold() -> VisionGolden:
    require(minicpmv_gguf(), MMPROJ)
    return golden(minicpmv_gguf(), MMPROJ, ASTRONAUT, "<__media__>Describe the image.", names=NAMES)


@pytest.mark.parametrize("case", ["astronaut", "square-1024", "small"])
def test_slices(case: str, tmp_path: str) -> None:
    require(minicpmv_gguf(), MMPROJ)
    path = ASTRONAUT
    if case != "astronaut":  # (noise: the resampling exact at every pixel)
        w, h = (1024, 1024) if case == "square-1024" else (300, 200)
        path = os.path.join(tmp_path, f"{case}.png")
        Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)).save(path)
    g = golden(minicpmv_gguf(), MMPROJ, path, "<__media__>Describe the image.",
                 names=NAMES if case == "astronaut" else "")  # (the astronaut's: the gold fixture's)
    entries, grid = M.slice_image(M.load_image(path), ClipConfig.from_gguf(GGUFFile(MMPROJ)))
    ll = [g.tensor("inp_raw", k) for k in range(g.entries)]
    # identical; a JPEG's decoded pixels within 3 levels (Pillow's libjpeg-turbo vs llama.cpp's stb_image)
    tol = 3.01 / 127.5 if case == "astronaut" else 0
    same = lambda a, b: a.shape == b.shape and np.abs(a - b).max() <= tol
    assert same(entries[0], ll[0])  # (the overview)
    if case == "square-1024":  # (llama.cpp's float tie: 3 x 2 slices of 392 x 560 instead of 2 x 3 of 560 x 336)
        assert grid.grid == (2, 3) and [e.shape for e in ll[1:]] == [(3, 392, 560)] * 6
        return
    assert len(entries) == len(ll) and all(same(a, b) for a, b in zip(entries, ll, strict=True))
    text, image = (c for c in g.chunks[:2])
    assert text["text"][-1] == M.IMAGE and image["image"] == len(grid.layout(newline_after=False)) - 1


@pytest.mark.slow
def test_entries(gold: VisionGolden) -> None:
    enc = V.Encoder(MMPROJ)
    emb = gold.embd(0)
    image = next(c for c in gold.chunks if "image" in c)
    assert image["image"] == emb.shape[0]
    n = gold.entries
    sep = enc.w("v.tok_embd_sep").numpy()
    off = 0
    for k in range(n):
        tr: dict[str, torch.Tensor] = {}
        out = enc.forward(gold.tensor("inp_raw", k), None, tr).numpy()
        assert rel(out, emb[off:off + out.shape[0]]) < 1.5e-2, k
        assert rel(tr["post_ln"], gold.tensor("post_ln", k)) < 1e-2, k
        off += out.shape[0]
        # the separators after the entry: </image> (and <slice>) after the overview, </slice> (\n) <slice> between
        # the slices, </slice> after the last
        place = ("ov_slice" if n > 1 else "ov") if k == 0 else "last" if k == n - 1 else None
        rows: list[int] = []
        while off < emb.shape[0] and (j := next((i for i in range(4) if np.array_equal(emb[off], sep[i])), None)) is not None:
            rows.append(j)
            off += 1
        if place is not None:
            assert rows == V.SUFFIX_ROWS[place], (k, rows)
        else:
            assert rows in (V.SUFFIX_ROWS["slice"], V.SUFFIX_ROWS["row_end"]), (k, rows)
    assert off == emb.shape[0]
