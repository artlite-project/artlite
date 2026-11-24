"""Inkling 2 on the NPU (``models.inkling2.inkling.NpuInkling``), the whole model:

- teacher-forced decode steps over an essay against llama.cpp's runs of the same GGUF file (logits,
  ``tests/goldens/text.py``): KL of the next-token distributions and argmax agreement against its decode (ubatch 1),
  next to its own prompt run against it (its graphs differ: Q8_0 activations, and the routing of near-tied experts);
- a checkpoint: the steps after a restore give the logits they gave before it, bit for bit (the convs' state and the
  sliding-window caches back with it).
"""

import numpy as np
import pytest
import torch

from artlite.model_config import inkling_gguf
from models.inkling2.inkling import Inkling

from ..common import essay
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require
from ..numerics import kl_rows

pytestmark = pytest.mark.slow  # whole GGUF models: run with --run-slow


def _path() -> str:
    path = inkling_gguf()
    require(path)
    return path


def test_teacher_forced() -> None:
    """(the agreement over the tokens whose top two llama.cpp's decode has at least 1 apart, as Flint's: a mixture of
    experts flips near-tied experts on either side's rounding)"""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:128]
    dec = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=1, logits="f32").logits, np.float32).copy())
    pro = torch.from_numpy(np.asarray(golden(path, tokens, ubatch=0, logits="f32").logits, np.float32).copy())
    model = Inkling(path).load(capacity=1024)
    got = torch.from_numpy(model.forced_logits(tokens))
    kl, kl_llama = float(kl_rows(dec, got).mean()), float(kl_rows(dec, pro).mean())
    top2 = torch.topk(dec, 2).values
    sure = top2[:, 0] - top2[:, 1] >= 1
    agree = float((dec.argmax(-1) == got.argmax(-1))[sure].float().mean())
    agree_llama = float((dec.argmax(-1) == pro.argmax(-1))[sure].float().mean())
    print(f"KL {kl:.4f} (llama.cpp prompt vs decode {kl_llama:.4f}), argmax agreement {agree:.3f} ({agree_llama:.3f}) "
          f"over {int(sure.sum())} of {len(tokens)} tokens")
    assert kl < 1.5 * kl_llama and agree > agree_llama - 0.05


def test_checkpoint() -> None:
    """A checkpoint at position 37 (inside a cache group; the conv rings and history windows mid-way), 11 steps after
    it, a restore and the same steps again; then a prompt from position 0 again after a reset."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:48]
    model = Inkling(path).load(capacity=1024)
    model.prompt(tokens[:37])
    cp = model.checkpoint()
    first = [np.asarray(model.step(t), np.float32).copy() for t in tokens[37:]]
    model.restore(37, cp)
    again = [np.asarray(model.step(t), np.float32).copy() for t in tokens[37:]]
    for i, (a, b) in enumerate(zip(first, again, strict=True)):
        assert np.array_equal(a, b), i
    model.reset()
    model.prompt(tokens[:37])
    assert np.array_equal(np.asarray(model.step(tokens[37]), np.float32), first[0])
