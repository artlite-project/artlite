"""LFM3-Audio's speech output on the host (models/lfm3a/speech.py) and the float64 generator (tests/lfm3a/gen_ref.py)
against llama.cpp's libmtmd (no NPU): the generator's metadata, the TTS prompts' text and tokens, the FSQ table, the
depthformer's logits (llama.cpp's own distance from float64: its Q8_0 activations) and feedback rows, the codec's
samples of speech codes (llama.cpp's SNR from float64, and a bfloat16 codec's: transformers' dtype), the codec run in
chunks equal to the codes at once, and the code sampler's penalty window."""

import math

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import GenConfig
from models.lfm3a import speech as S
from models.tokenizer import BpeTokenizer

from ..llama_cpp import llama_tokenize
from ..models import require
from ..numerics import kl_rows
from . import gen_ref as R
from .files import MMPROJ, MODEL, TEXT


def snr(ref: np.ndarray, x: np.ndarray) -> float:
    return float(10 * np.log10((ref ** 2).sum() / ((ref - x) ** 2).sum()))


@pytest.fixture(scope="module")
def weights() -> R.Weights:
    require(MODEL, MMPROJ)
    return R.Weights(MMPROJ)


@pytest.fixture(scope="module")
def tts() -> R.GenGolden:
    """llama.cpp's TTS of TEXT: the prompt's hidden state, 40 frames sampled (seed 1) and each frame's hidden state."""
    require(MODEL, MMPROJ)
    return R.golden(MODEL, MMPROJ, prompt=S.tts_prompt(GGUFFile(MODEL), TEXT), gen=40, seed=1)


def test_config() -> None:
    require(MODEL, MMPROJ)
    c = GenConfig.from_gguf(GGUFFile(MMPROJ))
    assert (c.projector, c.projection_dim, c.hidden, c.ffn, c.layers, c.heads, c.kv_heads, c.top_k, c.penalty_last_n) == \
        ("lfm3a_gen", 2048, 1024, 2816, 3, 32, 8, 128, 4096)
    assert c.head_dim == 32 and math.isclose(c.eps, 1e-5, rel_tol=1e-6)
    assert math.isclose(c.temp, 1.0) and math.isclose(c.penalty_repeat, 1.1, rel_tol=1e-6)


@pytest.mark.parametrize("language, voice, system", [
    ("en", "female", "Perform TTS in American English using a female voice."),
    ("de", "male", "Perform TTS in German using a male voice."),
    ("EN-GB", "Female", "Perform TTS in British English using a female voice."),
    ("Klingon", "male", "Perform TTS in Klingon using a male voice."),
])
def test_tts_prompt(language: str, voice: str, system: str) -> None:
    require(MODEL, MMPROJ)
    g = GGUFFile(MODEL)
    text = S.tts_prompt(g, TEXT, language=language, voice=voice)
    assert text == (f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{TEXT}<|im_end|>\n"
                    "<|im_start|>assistant\n<|audio_start|>")
    assert BpeTokenizer(g).encode(text) == llama_tokenize(MODEL, text)
    with pytest.raises(ValueError, match="voices"):
        S.tts_prompt(g, TEXT, voice="robot")


def test_fsq(weights: R.Weights) -> None:
    tab = weights("a.gen.wav.quant.first.codebook.weight").numpy()
    assert np.abs(tab - R.fsq_latents(np.arange(R.N_CODES - 1))).max() < 1e-7


def test_depth(weights: R.Weights, tts: R.GenGolden) -> None:
    """A frame's passes (every code masked, 4 and 8 revealed in llama.cpp's sampled frames' orders) on the prompt's
    hidden state and later frames': per-codebook KL of llama.cpp's logits from float64 (its Q8_0 activations), the
    feedback rows."""
    d = R.Depth(weights)
    codes = np.asarray(tts.codes)
    assert len(codes) >= 20
    rng = np.random.default_rng(0)
    passes, hs = [], []
    for f in (0, 5, 19):
        order = rng.permutation(R.N_CB)
        for n in (0, 4, 8):
            c = -np.ones(R.N_CB, np.int32)
            c[order[:n]] = codes[f, order[:n]]
            passes.append(c)
            hs.append(tts.hg(f))
    kls = []
    for c, h in zip(passes, hs, strict=True):
        g = R.golden(MODEL, MMPROJ, h=h, passes=c[None])
        kls.append(kl_rows(d.logits(h, c), g.logits(0)))
    kls = np.array(kls)
    print(f"KL(fp64 || llama.cpp) per codebook: median {np.median(kls):.2e}, max {kls.max():.2e}")
    assert np.median(kls) < 1e-3 and kls.max() < 5e-3
    for f in (0, 7):
        fb = d.feedback(codes[f])
        assert np.abs(fb - tts.fb(f"g{f}")).max() <= 1e-5 * np.abs(fb).max()
        assert np.abs(fb - S.feedback(GGUFFile(MMPROJ), codes[f])).max() <= 1e-5 * np.abs(fb).max()


def test_codec(weights: R.Weights, tts: R.GenGolden) -> None:
    """The codec's samples of llama.cpp's sampled speech codes: llama.cpp (in calls of 4 frames) against float64 (SNR,
    its Q8_0 activations), a bfloat16 codec's (transformers' dtype: the NPU's yardstick); float64 in calls of 4 frames
    and of 1 = the codes at once."""
    codes = np.asarray(tts.codes)
    c = R.Codec(weights)
    y = R.pcm(c, codes)
    assert y.size == R.FRAME_STACK * R.HOP * len(codes)
    for chunk in (1, 4):
        assert np.abs(R.pcm(c, codes, chunk) - y).max() < 1e-12
    g = R.golden(MODEL, MMPROJ, frames=codes, chunk=4)
    yl = g.pcm
    assert yl.size == y.size and g.pcm_calls[-1] == R.TRIM
    s_llama = snr(y, yl)
    s_bf16 = snr(y, R.pcm(R.Codec(MMPROJ, torch.bfloat16), codes))
    print(f"codec SNR vs float64: llama.cpp {s_llama:.1f} dB, bfloat16 {s_bf16:.1f} dB")
    assert s_llama > 20 and s_bf16 > 15


def test_sampler() -> None:
    """The penalty's window: a codebook's codes of the last ``penalty_last_n`` frames, the older ones forgotten; top-k
    keeps the k largest; temperature 0 is the argmax."""
    require(MODEL, MMPROJ)
    cfg = GenConfig.from_gguf(GGUFFile(MMPROJ))
    s = S.CodeSampler(cfg.__class__(**{**cfg.__dict__, "penalty_last_n": 3}), seed=0)
    for f in range(5):
        s.accept(np.full(S.N_CB, f, np.int32))
    assert (s.counts[:, :2] == 0).all() and (s.counts[:, 2:5] == 1).all() and s.counts.sum() == 3 * S.N_CB
    lg = np.zeros(S.N_CODES)
    lg[[7, 8]] = 50
    lg[4] = 60  # (penalized: 60 / 1.1 < 55)
    picks = {s.sample(lg, 0) for _ in range(200)}
    assert picks <= {4, 7, 8} and {7, 8} <= picks
    g = S.CodeSampler(cfg.__class__(**{**cfg.__dict__, "temp": 0.0}), seed=0)
    assert g.sample(lg, 0) == 4
    k = S.CodeSampler(cfg.__class__(**{**cfg.__dict__, "top_k": 2, "penalty_repeat": 1.0}), seed=0)
    assert {k.sample(lg + np.arange(S.N_CODES) * 1e-3, 0) for _ in range(100)} <= {4, 8}
