"""LFM3-VL's videos (``models.lfm3vl.host.VideoPlan``: frames in pairs, one temporal patch each): the float64 encoder of
a pair (``ref.Encoder`` on its two frames: kernel t on frame t; an odd last frame alone, both kernels on it) against
llama.cpp's pair embeddings (libmtmd), within llama.cpp's own distance from float64 as on images. On LFM3-VL-2B's BF16
tower: llama.cpp's CPU runs a Q8_0 file's matmuls on 8-bit activations, 3-11% from float64 (a still image's 3-4%), too
far to check a pair's inputs by (its kernels swapped: 40-50%). The NPU's pairs (``models.lfm3vl.npu.NpuVision.pairs``)
against the float64 encoder."""


import numpy as np
import pytest
import torch

from artlite.model_config import lfm3vl_gguf, lfm3vl_mmproj
from models import video as VD
from models.lfm3vl import host as H

from ..common import video_file
from ..goldens.media import golden
from ..models import require
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
    require(MODEL, MMPROJ)
    path = video_file(videos, name)
    g = golden(MODEL, MMPROJ, path, PROMPT)
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
    require(mm)
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


@pytest.mark.parametrize("size", [(224, 256), (384, 448), (320, 320)])
def test_packed_pairs(size: tuple[int, int]) -> None:
    """Two pairs in one run (``Layout.two``: one a half, their own keys and conv neighbours) give each pair's embeddings
    bit for bit as a run each (``Encoder.packs``: pairs of 56 tokens, a band for two; of 168, 3 bands for two; 100
    tokens do not pack), with an odd frame and a lone last pair (run with itself)."""
    from models.lfm3vl.npu import NpuVision
    require(MMPROJ)
    v = NpuVision(MMPROJ)
    enc = v.encoder
    frames = np.random.default_rng(0).integers(0, 256, (7, *size, 3), np.uint8)  # (3 pairs and an odd frame)
    packs = enc.packs(size[0] // 32, size[1] // 32)
    assert packs == (size != (320, 320))
    got = v.pairs(frames)
    enc.packs = lambda gh, gw: False
    want = v.pairs(frames)
    assert len(got) == len(want) == 4 and all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


def test_program_cache() -> None:
    """A program is cached for later processes only where it takes the cached state's images: a pair's pixel panels
    are allocated after the state (by the first pair program built), so no pair program is cached, whichever size
    comes first (a later process restoring the state has no such image); the image programs they take are."""
    from models.lfm3vl.npu import NpuVision
    require(MMPROJ)
    enc = NpuVision(MMPROJ).encoder
    for k in [k for k in enc.programs if isinstance(k, tuple)]:
        del enc.programs[k]
    cached = {}
    enc.built = lambda key, pieces, same: cached.__setitem__(key, same)
    for nb in (2, 1, 3):
        enc.program(nb, pair=True)
    pairs = {k: v for k, v in cached.items() if isinstance(k, tuple)}  # (an image program a pair's took: cached)
    assert any(".pixp." in n for n in enc.npu._images) and pairs == dict.fromkeys([("pair", 2), ("pair", 1),
                                                                                  ("pair", 3)], False)


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
