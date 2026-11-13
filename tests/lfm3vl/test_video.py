"""LFM3-VL's videos (``models.lfm3vl.host.VideoPlan``: frames in pairs, one temporal patch each): the float64 encoder of
a pair (``ref.Encoder`` on its two frames: kernel t on frame t; an odd last frame alone, both kernels on it) against
llama.cpp's pair embeddings (libmtmd), within llama.cpp's own distance from float64 as on images. On LFM3-VL-2B's BF16
tower: llama.cpp's CPU runs a Q8_0 file's matmuls on 8-bit activations, 3-11% from float64 (a still image's 3-4%), too
far to check a pair's inputs by (its kernels swapped: 40-50%). The NPU's pairs (``models.lfm3vl.npu.NpuVision.pairs``)
against the float64 encoder."""

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


@pytest.mark.parametrize("mmproj", ["BF16", "omni"])
def test_npu_pairs(videos: str, mmproj: str) -> None:
    """The NPU's pairs (``NpuVision.pairs``: the pair's patch GEMM, KP 1600, then the image program's layers) of two
    videos (an odd last frame; 14 pairs of 510 tokens) within the tower's error from float64 as on images: the median
    per merged token under 6% (measured 3.3 - 5.2%; still frames of the same size 4.2 - 4.3%)."""
    from artlite.model_config import lfm3omni_mmproj
    from models.lfm3vl.npu import NpuVision
    mm = MMPROJ if mmproj == "BF16" else lfm3omni_mmproj()
    if not os.path.exists(mm):
        pytest.skip(f"{mm} missing")
    v, enc = NpuVision(mm), V.Encoder(mm)
    for name in ("odd-25fps", "astronaut"):
        data = VD.read(video_file(videos, name))
        info = VD.probe(data)
        plan = H.VideoPlan.of(info.n_frames, info.fps, info.width, info.height, heard=True)
        fr = VD.frames(data, info, plan.indices, (plan.width, plan.height))
        out = v.pairs(fr)
        assert len(out) == len(plan.pairs()) and all(e.shape[0] == plan.grid.tokens for e in out)
        for e, x in zip(out[:3], pair_pixels(video_file(videos, name), heard=True), strict=False):
            ref = enc.forward(x).numpy()
            med = float(np.median(np.linalg.norm(e - ref, axis=1) / np.linalg.norm(ref, axis=1)))
            print(f"{name}: a pair's median per token {med:.4f}")
            assert med < 0.06


@pytest.mark.slow
def test_app_video(videos: str) -> None:
    """LFM3-VL-2B with a video (frames only: its mmproj has no speech encoder): the app's prompt (pairs on the NPU)
    and a greedy answer that describes it."""
    from artlite.model_config import lfm3vl_gguf
    from models.lfm.lfm import Lfm
    m = Lfm(lfm3vl_gguf(), mmproj=MMPROJ)
    npu = m.load(capacity=8192)
    msgs = [{"role": "user", "content": [{"type": "video", "video": video_file(videos, "astronaut")},
                                         {"type": "text", "text": "What is in the video? One sentence."}]}]
    prompt = m.chat(msgs)
    text = m.tokenizer.decode([t for t in npu.generate(npu.forward(prompt), 60, m.sampling(), prompt)
                               if t not in m.eos_token_ids])
    print(f"{len(prompt)} tokens: {text!r}")
    assert "horse" in text.lower() and "astronaut" in text.lower()
