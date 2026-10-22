"""qwen35 models on the NPU (models.qwen35, operators/qwen35/overlay.py) against llama.cpp's decode
(ubatch 1) of the same GGUF file (golden tensors and logits, tests/rtc/qwen35_ref.py):

- a slice of four layers in the middle of the model (GDN layers and a full-attention layer), fed llama.cpp's residual
  stream before it token by token: the residual after every layer;
- the whole model, teacher-forced over an essay: KL of the next-token distributions and argmax agreement against
  llama.cpp's decode, within llama.cpp's own noise: it quantizes the activations of every projection to Q8_K, and its
  prompt (one graph) and decode runs of the text differ by a KL of 0.0026 (9B) / 0.051 (27B: a few tokens of this
  text near a tie flip, in both runs and on the NPU).
"""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.qwen35.qwen import NpuQwen

from . import qwen35_ref as ref
from .common import essay

pytestmark = pytest.mark.slow  # whole GGUF models: run with --run-slow

def _path(size: str) -> str:
    path = ref.qwen_gguf("Q4_K_M", size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    return path


def _tokens(path: str, n: int, name: str = "founders.txt") -> list[int]:
    return ref.llama_tokenize(path, essay(name))[:n]


@pytest.mark.parametrize("size", ["9B", "27B", "1B"])
def test_slice(size: str) -> None:
    from artlite.operators.qwen35.overlay import QwenOverlay

    path = _path(size)
    g = GGUFFile(path)
    cfg = ref.QwenConfig.from_gguf(g)
    La = cfg.attention_layers[len(cfg.attention_layers) // 2]
    layers = list(range(La - 3, La + 1))
    n = 34
    tokens = _tokens(path, n, "apple.txt")
    G = ref.golden(path, tokens, ubatch=1, layers=(layers[0] - 1, *layers), names="l_out", logits="none")
    ov = QwenOverlay(g, layers=layers, vocab=512, max_positions=64)
    ov.load(release=False)
    errs = np.zeros((n, len(layers)))
    for t in range(n):
        ov.step(torch.from_numpy(np.asarray(G["l_out", layers[0] - 1][t], np.float32).copy()), t, logits=False)
        for i, L in enumerate(layers):
            got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1).numpy()
            errs[t, i] = ref.rel_err(got, np.asarray(G["l_out", L][t], np.float32))
    print(f"{size} layers {layers}: mean error per layer {errs.mean(0)}, max {errs.max():.3f}")
    assert errs.max() < 4e-2 and errs.mean(0).max() < 1.5e-2


def _kl(want: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    return (lw.exp() * (lw - lg)).sum(-1)


@pytest.mark.parametrize("size", ["9B", "27B", "1B", "35B-A3B"])
def test_teacher_forced(size: str) -> None:
    path = _path(size)
    tokens = _tokens(path, 128)
    dec = torch.from_numpy(np.asarray(ref.golden(path, tokens, ubatch=1, logits="f32").logits, np.float32).copy())
    pro = torch.from_numpy(np.asarray(ref.golden(path, tokens, ubatch=0, logits="f32").logits, np.float32).copy())
    model = NpuQwen(GGUFFile(path), capacity=256)
    got = model.forced_logits(tokens)
    kl, kl_llama = float(_kl(dec, got).mean()), float(_kl(dec, pro).mean())
    agree = float((dec.argmax(-1) == got.argmax(-1)).float().mean())
    agree_llama = float((dec.argmax(-1) == pro.argmax(-1)).float().mean())
    print(f"{size}: KL {kl:.4f} (llama.cpp prompt vs decode {kl_llama:.4f}), argmax agreement {agree:.3f} "
          f"({agree_llama:.3f}) over {len(tokens)} tokens")
    # (an MoE model's routing flips near-tied experts: its argmax agreement moves by a few of the 128 tokens)
    tol = 0.05 if model.cfg.moe else 0.02
    assert kl < 1.5 * kl_llama and agree > agree_llama - tol


@pytest.mark.parametrize("size", ["9B", "27B", "1B", "35B-A3B"])
def test_prefill(size: str) -> None:
    """The app's prompt path: 300 tokens through the prefill (one chunk of 512 tokens, 212 of them padding), the
    last one's logits from the LM head alone, then teacher-forced decode steps continuing from the prefill's caches
    and states; the next-token distributions against llama.cpp's split run (the prompt in one graph, then one token per
    graph), next to llama.cpp's own decode-only run against it."""
    path = _path(size)
    P, K = 300, 24
    tokens = _tokens(path, P + K)
    split = torch.from_numpy(np.asarray(ref.golden(path, tokens, split=P, logits="f32").logits, np.float32)[P - 1:P + K - 1].copy())
    dec = torch.from_numpy(np.asarray(ref.golden(path, tokens, ubatch=1, logits="f32").logits, np.float32)[P - 1:P + K - 1].copy())
    model = NpuQwen(GGUFFile(path), capacity=4096)
    got = torch.stack([model.forward(tokens[:P])] + [model.step(t) for t in tokens[P:P + K - 1]])
    kl, kl_llama = _kl(split, got), _kl(split, dec)
    agree = float((split.argmax(-1) == got.argmax(-1)).float().mean())
    agree_llama = float((split.argmax(-1) == dec.argmax(-1)).float().mean())
    print(f"{size}: KL vs llama.cpp's split run: prompt's last token {float(kl[0]):.4f} (llama.cpp decode-only "
          f"{float(kl_llama[0]):.4f}), mean over {K} positions {float(kl.mean()):.4f} ({float(kl_llama.mean()):.4f}); "
          f"argmax agreement {agree:.3f} ({agree_llama:.3f})")
    assert float(kl.mean()) < 1.5 * float(kl_llama.mean()) + 2e-3 and agree > agree_llama - 0.05
