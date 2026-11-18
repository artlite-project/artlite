"""Flint's sliding-window caches as rings (``FlintOverlay(wrap=)``, ``decode.attn_overlay.CacheKind.wrap``) against linear
ones, bit for bit (only the caches' addresses differ):

- decode steps of 1 .. 4 slots on two sliding-window layers across the ring's wraps (16 groups: 1024 positions), and a
  snapshot's restore after them;
- the whole model (slow): a prompt through the prefill whose chunks start at the ring's slot 0 (the guard a copy of
  its last groups: ``mirror``) and its middle, decode steps across a wrap, a continuation whose window reaches back over
  the wrap, a checkpoint's restore.
"""

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import flint_gguf
from models.flint.flint import NpuFlint
from models.flint.overlay import FlintOverlay
from models.sampling import Sampling
from models.tokenizer import BpeTokenizer

from ..common import essay
from ..model_files import require
from .test_decode import STEPS, VOCAB, _run

WRAP = 16  # (groups: 1024 positions)


@pytest.fixture(scope="module")
def gguf() -> GGUFFile:
    path = flint_gguf()
    require(path)
    return GGUFFile(path)


def test_flint_ring_steps(gguf: GGUFFile) -> None:
    ids = BpeTokenizer(gguf).encode(essay("apple.txt"))[:2300]
    x = gguf.rows("token_embd.weight", np.asarray(ids)).astype(np.float32)
    out = {}
    for wrap in (0, WRAP):
        ov = FlintOverlay(gguf, layers=[2, 3], vocab=VOCAB, max_positions=2560, wrap=wrap)
        ov.load()
        out[wrap] = _run(ov, x, STEPS)
        if wrap:  # (a snapshot inside the second turn, restored after the third's start)
            ov.reset()
            _run(ov, x[:1200], STEPS)
            snap = ov.snapshot(1200)
            for p in range(1200, 2200, 4):
                ov.step(x[p:p + 4], p, logits=False)
            ov.restore(snap)
            again = np.stack([np.asarray(ov.step(x[p:p + 1], p), np.float32)[0] for p in range(1200, 1300)])
            assert np.array_equal(again, out[wrap][1200:1300])
        del ov
    assert np.array_equal(out[WRAP], out[0])


@pytest.mark.slow
def test_flint_ring_prefill(gguf: GGUFFile) -> None:
    """(chunks of up to 512 tokens: the same plan with either cache; the prompt's chunks at slots 0 and 8, its last 52
    tokens as steps across the wrap at 2048, 60 tokens generated, a continuation from 2176: slot 2, inside the guard)"""
    tk = BpeTokenizer(gguf)
    text = tk.encode(essay("apple.txt"))
    first, more = text[:2100], text[2100:2700]
    out = {}
    for wrap in (0, WRAP):
        m = NpuFlint(gguf, capacity=3072, sizes=(64, 128, 256, 512), wrap=wrap)
        m.lookup = False
        m.reset()
        lg = [np.asarray(m.forward(first), np.float32).copy()]
        gen = list(m.generate(lg[0], 60, Sampling(greedy=True), first))
        lg.append(np.asarray(m.forward(more), np.float32).copy())
        p, cp = m.position, m.checkpoint()
        after = list(m.generate(lg[1], 20, Sampling(greedy=True), more))
        m.restore(p, cp)
        assert list(m.generate(lg[1], 20, Sampling(greedy=True), more)) == after
        out[wrap] = (lg, gen, after)
        del m
    (la, ga, aa), (lb, gb, ab) = out[0], out[WRAP]
    assert ga == gb and aa == ab
    assert all(np.array_equal(a, b) for a, b in zip(la, lb, strict=True))
