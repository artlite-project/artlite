"""LFM3-Audio's speech generator on the NPU's vision overlay (models/lfm3a/depth.py, codec.py) against the float64
generator (tests/lfm3a/gen_ref.py) on llama.cpp's sampled speech codes and hidden states: the depthformer's passes
(per-codebook KL, as far from float64 as about twice llama.cpp's own: the bfp16 weights and GEMM inputs), the codec's
samples (PCM SNR within 3 dB of a bfloat16 codec's: transformers' dtype) and its streaming (calls of 4 frames = the codes
at once, bit for bit: whole 8-key blocks), the ISTFT (= the reference's)."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.vision.device import VisionDevice
from artlite.vision.weights import gguf_getter
from models.lfm3a import codec as CD
from models.lfm3a import depth as DP
from models.lfm3a import speech as S

from . import gen_ref as R
from .test_gen_ref import MMPROJ, MODEL, TEXT, kl

TEXTS = [TEXT, "The quick brown fox jumps over the lazy dog, and then it runs back into the forest.",
         "Speech synthesis on a neural processing unit is a good test of small matrix programs."]


@pytest.fixture(scope="module")
def gen() -> tuple[DP.Depth, CD.Codec]:
    for path in (MODEL, MMPROJ):
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")
    g = GGUFFile(MMPROJ)
    get = gguf_getter(g)
    npu = VisionDevice()
    mask = g.tensor("a.gen.code.mask_embd").dequantize().reshape(-1)
    dp = DP.Depth(npu, DP.weights(get, 3), lambda r: g.rows("a.gen.code.embd.weight", r), mask)
    cd = CD.Codec(npu, CD.weights(get, CD.kinds_of(g.tensor_info, 8)))
    return dp, cd


def golden(i: int = 0, n: int = 40) -> R.GenGolden:
    return R.golden(MODEL, MMPROJ, prompt=S.tts_prompt(GGUFFile(MODEL), TEXTS[i]), gen=n, seed=1 + i)


def test_depth(gen: tuple[DP.Depth, CD.Codec]) -> None:
    """A frame's 3 passes (a random order: 4 codebooks each, the earlier ones' codes revealed) of 8 frames' hidden
    states: per-codebook KL of the NPU's logits from float64 within 2.5x llama.cpp's own (measured 2.1x; median)."""
    dp, _ = gen
    ref = R.Depth(MMPROJ)
    g = golden()
    rng = np.random.default_rng(0)
    ks, lks = [], []
    for f in (0, 3, 5, 9, 14, 19, 25, 30):
        h = g.hg(f)
        dp.begin(h)
        order = rng.permutation(R.N_CB)
        for p in range(3):
            c = -np.ones(R.N_CB, np.int32)
            c[order[:4 * p]] = g.codes[f, order[:4 * p]]
            cbs = list(order[4 * p:4 * p + 4])
            r = ref.logits(h, c)[cbs]
            ks.append(kl(r, dp.logits(c, cbs)))
            lks.append(kl(r, R.golden(MODEL, MMPROJ, h=h, passes=c[None]).logits(0)[cbs]))
    ks, lks = np.concatenate(ks), np.concatenate(lks)
    print(f"KL from float64: NPU median {np.median(ks):.2e} (p90 {np.percentile(ks, 90):.2e}), llama.cpp {np.median(lks):.2e}")
    assert np.median(ks) < 2.5 * np.median(lks) and np.percentile(ks, 90) < 2.5 * np.percentile(lks, 90)


def test_codec(gen: tuple[DP.Depth, CD.Codec]) -> None:
    """3 clips' codes (llama.cpp's, 40-70 frames) streamed in calls of 4 frames: PCM SNR from float64 within 3 dB of a
    bfloat16 codec's (over the clips; measured 26.7 vs 28.9 dB), the log magnitudes' rms error below 0.035."""
    _, cd = gen
    ref, refb = R.Codec(R.Weights(MMPROJ)), R.Codec(MMPROJ, torch.bfloat16)
    tot = np.zeros(3)
    for i in range(len(TEXTS)):
        codes = golden(i, 120).codes
        rf = ref.frames(codes)
        cd.reset()
        ist = S.Istft()
        fs, ys = [], []
        for k in range(0, len(codes), 4):
            f = cd.frames(codes[k:k + 4])
            fs.append(f)
            ys.append(ist.push(f))
        f, y = np.concatenate(fs), np.concatenate(ys + [ist.flush()])
        yr, yb = R.pcm(ref, codes), R.pcm(refb, codes)
        assert np.sqrt(((rf[:, :S.NBINS] - f[:, :S.NBINS]) ** 2).mean()) < 0.035
        tot += [(yr ** 2).sum(), ((yr - y) ** 2).sum(), ((yr - yb) ** 2).sum()]
    s_npu, s_bf16 = 10 * np.log10(tot[0] / tot[1]), 10 * np.log10(tot[0] / tot[2])
    print(f"PCM SNR from float64: NPU {s_npu:.1f} dB, bfloat16 codec {s_bf16:.1f} dB")
    assert s_npu > s_bf16 - 3


def test_codec_stream(gen: tuple[DP.Depth, CD.Codec]) -> None:
    """The codec in calls of 4 frames (a band's runs with more of its rows, two bands and a carry: 50 frames = 300
    positions) = the codes at once, bit for bit; the ISTFT of NPU frames = the reference's (float64 vs float32 out)."""
    _, cd = gen
    codes = golden(1, 120).codes[:50]
    cd.reset()
    whole = cd.frames(codes)
    cd.reset()
    parts = np.concatenate([cd.frames(codes[k:k + 4]) for k in range(0, len(codes), 4)])
    assert whole.shape == (300, CD.NOUT) and np.array_equal(whole, parts)
    ist, rist = S.Istft(), R.Istft()
    y = np.concatenate([ist.push(whole[:100]), ist.push(whole[100:]), ist.flush()])
    yr = np.concatenate([rist.push(whole[:100]), rist.push(whole[100:]), rist.flush()])
    assert y.size == yr.size == 300 * R.HOP and np.abs(y - yr).max() < 1e-5 * np.abs(yr).max()
