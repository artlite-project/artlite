"""Inkling 2 on the NPU (``models.inkling2.inkling.NpuInkling``), the whole model:

- teacher-forced decode steps over an essay against llama.cpp's runs of the same GGUF file (logits,
  ``tests/goldens/text.py``): KL of the next-token distributions and argmax agreement against its decode (ubatch 1),
  next to its own prompt run against it (its graphs differ: Q8_0 activations, and the routing of near-tied experts);
- a checkpoint: the steps after a restore give the logits they gave before it, bit for bit (the convs' state and the
  sliding-window caches back with it);
- steps of S tokens against steps of one, a rollback of rejected drafts, greedy text with the context's drafts: bit for
  bit, the same text.
"""

import numpy as np
import pytest
import torch

from artlite.model_config import inkling_gguf
from models.inkling2.inkling import Inkling
from models.sampling import Sampling

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


def test_steps_of_s() -> None:
    """Steps of S = 2, 3, 4 tokens give every position the logits steps of one give, bit for bit (positions 0 .. 79:
    across the first cache group, the history windows and the raw rings at every offset)."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:80]
    model = Inkling(path).load(capacity=1024)
    one = model.forced_logits(tokens, batch=1)
    for S in (2, 3, 4):
        got = model.forced_logits(tokens, batch=S)
        bad = np.flatnonzero((got != one).any(-1))
        assert not bad.size, (S, bad[:8])


def test_rollback() -> None:
    """A step of a token and drafts that are rejected, then the steps after the kept tokens (``SpeculativeMixin``'s
    round: the position back to after them), give the logits the steps that never ran the drafts give, bit for bit: the
    raw rings, the history windows' offset and the open records follow the position."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:48]
    model = Inkling(path).load(capacity=1024)
    want = model.forced_logits(tokens, batch=1)
    model.reset()
    model.prompt(tokens[:20])
    got = []
    i = 20
    for keep, wrong in ((1, [7, 8, 9]), (2, [11, 12]), (3, [13]), (1, []), (4, [])):
        run = tokens[i:i + keep] + wrong
        rows = model.cap(np.asarray(model.steps(run)))
        got += list(rows[:keep])
        model.position = i + keep  # (as ``SpeculativeMixin._kept``: the rejected drafts' positions forgotten)
        i += keep
    while i < len(tokens):
        got.append(model.cap(np.asarray(model.steps([tokens[i]])))[0])
        i += 1
    got = np.stack(got)
    bad = np.flatnonzero((got != want[20:]).any(-1))
    assert not bad.size, bad[:8]


def test_drafts() -> None:
    """Greedy text with the context's drafts is the text without them, and the drafts are kept: an essay's first 300
    tokens, then its first 100 again (the model goes on repeating it, as the drafts guess)."""
    path = _path()
    tokens = llama_tokenize(path, essay("founders.txt"))[:300]
    prompt = tokens + tokens[:100]
    model = Inkling(path).load(capacity=1024)
    texts = []
    for drafts in (0, 3):
        model.reset()
        logits = model.prompt(prompt, logits=True)
        model.lookup = drafts > 0
        texts.append(list(model.generate(logits, 64, Sampling(greedy=True), prompt, drafts=drafts)))
    assert texts[0] == texts[1]
    assert model.accepted_ctx > 0.5 * model.drafted_ctx > 0, (model.drafted_ctx, model.accepted_ctx)
