"""Flint on the NPU (models.flint) against llama.cpp's runs of the same GGUF file (logits, tests/goldens/text.py), the
whole model:

- teacher-forced decode steps over an essay: KL of the next-token distributions and argmax agreement against llama.cpp's
  decode (ubatch 1), next to llama.cpp's own prompt run against it (its graphs differ: Q8_1 activations, and 21-24% of
  its tokens' top-6 expert sets differ between the two runs);
- the app's prompt path: the prompt through the prefill, the last token's logits from the LM head alone, then
  teacher-forced decode steps from the prefill's caches, against llama.cpp's split run (the prompt in one graph, then
  one token per graph), next to llama.cpp's decode-only run against it.
"""

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import flint_gguf
from models.flint.flint import NpuFlint

from ..common import essay, essays
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from ..numerics import kl_rows

pytestmark = pytest.mark.slow  # whole GGUF models: run with --run-slow


def _path() -> str:
    path = flint_gguf()
    require(path)
    return path


def _tokens(path: str, n: int) -> list[int]:
    """An essay's first n tokens (past ~1500: the corpus's)."""
    out = llama_tokenize(path, essay("founders.txt") if n < 1500 else essays()[:8 * n])[:n]
    assert len(out) == n
    return out


def test_teacher_forced() -> None:
    """(the agreement over the tokens whose top two llama.cpp's decode has at least 1 apart: an MoE model's routing flips
    near-tied experts, then the next token where its top two are near ties)"""
    path = _path()
    tokens = _tokens(path, 128)
    dec = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=1, logits="f32").logits, np.float32).copy())
    pro = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=0, logits="f32").logits, np.float32).copy())
    model = NpuFlint(GGUFFile(path), capacity=256, prefill=False)
    got = torch.from_numpy(model.forced_logits(tokens))
    kl, kl_llama = float(kl_rows(dec, got).mean()), float(kl_rows(dec, pro).mean())
    top2 = torch.topk(dec, 2).values
    sure = top2[:, 0] - top2[:, 1] >= 1
    agree = float((dec.argmax(-1) == got.argmax(-1))[sure].float().mean())
    agree_llama = float((dec.argmax(-1) == pro.argmax(-1))[sure].float().mean())
    print(f"KL {kl:.4f} (llama.cpp prompt vs decode {kl_llama:.4f}), argmax agreement {agree:.3f} ({agree_llama:.3f}) "
          f"over {int(sure.sum())} of {len(tokens)} tokens")
    assert kl < 1.5 * kl_llama and agree > agree_llama - 0.05


@pytest.mark.parametrize("P", [300, 2100])
def test_prefill(P: int) -> None:
    """P tokens through the prefill (one chunk of 512, 212 of them padding; or a chunk of 2048 and decode steps), then
    24 teacher-forced decode steps; each position's KL capped at 0.5 before the mean (an ill-conditioned position's
    rounding moves its distribution far: P 300's tenth token, where llama.cpp's own runs differ by 0.33, the decode
    steps' path by 0.78, the prefill's by 2.6; the other positions' mean 0.009, decode steps' 0.004, llama.cpp's 0.005:
    the prefill's bfp16 GEMM operands)."""
    path = _path()
    K = 24
    tokens = _tokens(path, P + K)
    split = torch.from_numpy(np.asarray(golden(path, tokens, split=P, logits="f32").logits, np.float32)[P - 1:P + K - 1]
                             .copy())
    dec = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=1, logits="f32").logits, np.float32)[P - 1:P + K - 1]
                           .copy())
    model = NpuFlint(GGUFFile(path), capacity=4096)
    got = torch.from_numpy(np.stack([model.forward(tokens[:P])] + [model.step(t) for t in tokens[P:P + K - 1]]))
    kl, kl_llama = kl_rows(split, got), kl_rows(split, dec)
    top2 = torch.topk(split, 2).values
    sure = top2[:, 0] - top2[:, 1] >= 1  # (as ``test_teacher_forced``: a near tie flips on either side's rounding)
    agree = float((split.argmax(-1) == got.argmax(-1))[sure].float().mean())
    agree_llama = float((split.argmax(-1) == dec.argmax(-1))[sure].float().mean())
    print(f"P {P}: KL vs llama.cpp's split run: prompt's last token {float(kl[0]):.4f} (llama.cpp decode-only "
          f"{float(kl_llama[0]):.4f}), mean over {K} positions {float(kl.mean()):.4f} ({float(kl_llama.mean()):.4f}); "
          f"argmax agreement {agree:.3f} ({agree_llama:.3f}) over {int(sure.sum())} of {K} tokens")
    # (one clear token may flip: P 300's prompt's last token flips on the decode steps' path too)
    assert float(kl.clamp(max=0.5).mean()) < 1.5 * float(kl_llama.clamp(max=0.5).mean()) + 2e-3 and \
        agree * int(sure.sum()) >= agree_llama * int(sure.sum()) - 1
