"""MiniCPM-V 4.7's vision references (tests/rtc/minicpmv_ref.py) and slicing (models/qwen35/minicpmv.py) against
llama.cpp's libmtmd on an image (no NPU): the entries' pixels identical (where llama.cpp's float grid choice agrees
with the official; a JPEG's within its decoders' differences), the image chunk's tokens; the torch encoder op by op
in llama.cpp's rounding mode (each op fed llama.cpp's own input: within its fp32 summation order and a few fp16 table
ULPs) and per entry (the overview, the slices: within the encoder's own sensitivity across 27 layers and two merges,
~0.5-1%), with the separator rows between the entries."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import MODELS_DIR, ClipConfig, minicpmv_gguf
from models.qwen35 import minicpmv as M

from . import minicpmv_ref as V

IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "images", "sd15-astronaut.jpg")
MMPROJ = os.path.join(MODELS_DIR, "openbmb", "MiniCPM-V-4.7-1B-gguf", "mmproj-model-f16.gguf")
NAMES = ("pos_embed|layer_inp_normed|Qcur|attn_out|ffn_inp_normed|ffn_up_b|ffn_gelu|ffn_out|layer_out|"
         "vit_merger_ds_out|post_ln|merger_out")


def _files() -> None:
    for path in (minicpmv_gguf(), MMPROJ):
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")


@pytest.fixture(scope="module")
def gold() -> V.VisionGolden:
    _files()
    return V.golden(minicpmv_gguf(), MMPROJ, IMAGE, "<__media__>Describe the image.", names=NAMES)


@pytest.mark.parametrize("case", ["astronaut", "square-1024", "small"])
def test_slices(case: str, tmp_path: str) -> None:
    _files()
    path = IMAGE
    if case != "astronaut":  # (noise: the resampling exact at every pixel)
        w, h = (1024, 1024) if case == "square-1024" else (300, 200)
        path = os.path.join(tmp_path, f"{case}.png")
        Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)).save(path)
    g = V.golden(minicpmv_gguf(), MMPROJ, path, "<__media__>Describe the image.",
                 names=NAMES if case == "astronaut" else "pos_embed")  # (the astronaut's: test_ops' golden)
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


def _rel(a: torch.Tensor | np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.slow  # (the whole encoder on the CPU, in float32)
def test_ops(gold: V.VisionGolden) -> None:
    enc = V.Encoder(MMPROJ, llama=True)
    t = lambda name, il=None: torch.from_numpy(gold.tensor(name, 0, il).copy())
    tr: dict[str, torch.Tensor] = {}
    enc.forward(gold.tensor("inp_raw", 0), None, tr)
    assert _rel(tr["pos_embed"], gold.tensor("pos_embed", 0)) < 1e-6
    p = "v.blk.0."
    assert _rel(enc._ln(t("pos_embed"), p + "ln1"), gold.tensor("layer_inp_normed", 0, 0)) < 1e-6
    x = t("layer_inp_normed", 0)
    q = enc._mm(x, p + "attn_q.weight", p + "attn_q.bias")
    assert _rel(q, gold.tensor("Qcur", 0, 0).reshape(q.shape)) < 1e-6
    assert _rel(enc._attn(x, p), gold.tensor("attn_out", 0, 0)) < 1e-5
    assert _rel(enc._mm(t("ffn_inp_normed", 0), p + "ffn_up.weight", p + "ffn_up.bias"),
                gold.tensor("ffn_up_b", 0, 0)) < 1e-6
    assert _rel(enc._gelu(t("ffn_up_b", 0)), gold.tensor("ffn_gelu", 0, 0)) < 1e-5
    assert _rel(enc._mm(t("ffn_gelu", 0), p + "ffn_down.weight", p + "ffn_down.bias"), gold.tensor("ffn_out", 0, 0)) < 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("llama", [True, False], ids=["llama", "f64"])
def test_entries(gold: V.VisionGolden, llama: bool) -> None:
    enc = V.Encoder(MMPROJ, llama=llama)
    emb = gold.embd(0)
    image = next(c for c in gold.chunks if "image" in c)
    assert image["image"] == emb.shape[0]
    n = gold.entries
    sep = enc.w("v.tok_embd_sep").numpy()
    off = 0
    for k in range(n):
        tr: dict[str, torch.Tensor] = {}
        out = enc.forward(gold.tensor("inp_raw", k), None, tr).numpy()
        assert out.shape[0] == gold.tensor("merger_out", k).shape[0]
        assert _rel(out, emb[off:off + out.shape[0]]) < 1.5e-2, k
        assert _rel(tr["post_ln"], gold.tensor("post_ln", k)) < 1e-2, k
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
