"""LFM3-VL's videos (``models.lfm3vl.host.VideoPlan``: frames in pairs, one temporal patch each): the float64 encoder of
a pair (``ref.Encoder`` on its two frames: kernel t on frame t; an odd last frame alone, both kernels on it) against
llama.cpp's pair embeddings (libmtmd), within llama.cpp's own distance from float64 as on images. On LFM3-VL-2B's BF16
tower: llama.cpp's CPU runs a Q8_0 file's matmuls on 8-bit activations, 3-11% from float64 (a still image's 3-4%), too
far to check a pair's inputs by (its kernels swapped: 40-50%). No NPU."""

import os

import numpy as np
import pytest
import torch

from artlite.model_config import lfm3vl_gguf, lfm3vl_mmproj
from models import video as VD
from models.lfm3vl import host as H

from ..common import video_file
from . import ref as V

MODEL, MMPROJ = lfm3vl_gguf(), lfm3vl_mmproj("BF16")
PROMPT = "<|im_start|>user\n<|vision_start|><__media__><|vision_end|>Describe the video.<|im_end|>\n"


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("videos"))


def pair_pixels(path: str, heard: bool = False) -> list[np.ndarray]:
    """A video's pairs as the encoder's inputs: (2, 3, H, W) per pair, (3, H, W) for an odd last frame (normalized)."""
    data = VD.read(path)
    info = VD.probe(data)
    plan = H.VideoPlan.of(info.n_frames, info.fps, info.width, info.height, heard=heard)
    fr = VD.frames(data, info, plan.indices, (plan.width, plan.height))
    out = []
    for p in plan.pairs(len(fr)):
        x = np.stack([H.normalized(fr[f]) for f in p])
        out.append(x if len(p) == 2 else x[0])
    return out


@pytest.mark.parametrize("name", ["odd-25fps", "astronaut"])
def test_pair_reference(videos: str, name: str) -> None:
    """Each pair's (and an odd last frame's) float64 embeddings within 3% of llama.cpp's (measured 1.0 - 2.4%; a 512 x
    512 image's 1.0%)."""
    for p in (MODEL, MMPROJ):
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")
    path = video_file(videos, name)
    g = V.golden(MODEL, MMPROJ, path, PROMPT)
    media = [c for c in g.chunks if "text" not in c]
    images = [k for k, c in enumerate(media) if "image" in c]
    pairs = pair_pixels(path)
    assert len(images) == len(pairs)
    enc = V.Encoder(MMPROJ)
    errs = []
    for k, x in zip(images, pairs, strict=True):
        want = g.embd(k)
        got = enc.forward(x)
        errs.append(float(torch.linalg.norm(got - torch.from_numpy(want).double()) / np.linalg.norm(want)))
    print(f"{name}: pairs vs llama.cpp {[round(e, 4) for e in errs]}")
    assert max(errs) < 3e-2
