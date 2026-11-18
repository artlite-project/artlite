"""The LFM3 decode overlay (models/lfm: ``LfmOverlay``) on the NPU:

- slices of layers with a truncated vocabulary against the fp32 CPU model (``cpu.LfmCpu``) on the same weights: five
  layers of all kinds (conv, conv, full, conv, sliding-window) over 300 positions (past the window of 256) in steps of
  1 .. 4 tokens, K cached as single or hi + lo blocks: logits, the K / V caches, a conv layer's history;
- steps of S slots give each slot the logits of a step of one, bit for bit, also where a step continues from fewer
  slots than the last one ran (rejected drafts: the conv shift, the rings).
The whole model is gated by test_prefill.py's teacher-forced test (prefill, then decode steps).
"""

import numpy as np
import pytest
import torch

from models.lfm.overlay import LfmOverlay

from ..common import to_torch
from ..numerics import rel_finite
from . import ref

VOCAB = 2048
STEPS = (1, 4, 3, 2, 4, 4, 1, 3, 3, 2, 1, 4)  # (step sizes, repeated: every alignment to the records and their pairs)
TOL = 0.03  # (relative rms error of the logits: the GEMV's panels and bf16 sums, the caches' blocks; up to ~1.6% here)


def _run(ov: LfmOverlay, x: torch.Tensor, steps: tuple[int, ...]) -> torch.Tensor:
    """The logits of every position of x, in steps of the given sizes (repeated) from position 0."""
    out = torch.zeros(x.shape[0], ov.vocab)
    p, i = 0, 0
    while p < x.shape[0]:
        n = min(steps[i % len(steps)], x.shape[0] - p)
        i += 1
        out[p:p + n] = to_torch(ov.step(x[p:p + n], p)).float()
        p += n
    return out


@pytest.mark.parametrize("kv_k", ["single", "hilo"], ids=["five", "five-hilo"])
def test_lfm_overlay_layers(kv_k: str) -> None:
    g = ref.gguf()
    layers, n = [0, 1, 2, 3, 4], 300
    ids = ref.essay_ids(n)
    cpu = ref.cpu.LfmCpu(g, layers=layers, vocab=VOCAB)
    state = ref.cpu.LfmState()
    want = cpu.head(cpu.run(ids, state))
    x = cpu.emb[torch.tensor(ids)]
    ov = LfmOverlay(g, layers=layers, vocab=VOCAB, max_positions=512, kv_k=kv_k)
    ov.load()
    got = _run(ov, x, STEPS)
    worst = {"logits": max(rel_finite(got[p], want[p]) for p in range(n))}
    for L in layers:
        if ov.kind(L) == "conv":  # (the history: the last step's base and its rows of B x)
            rows = torch.cat([torch.zeros(6, ov.H), cpu_rows(cpu, ids, L)])  # (zeros before position 0)
            worst[f"conv.{L}"] = rel_finite(ov.read_conv(L)[:6 + ov.conv_s], rows[n - ov.conv_s:n + 6])
        else:
            k, v = ov.read_kv(L, range(n))
            worst[f"K.{L}"] = rel_finite(k, state.keys[L])
            worst[f"V.{L}"] = rel_finite(v, state.values[L])
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < TOL for v in worst.values()), worst


def cpu_rows(cpu: ref.cpu.LfmCpu, ids: list[int], L: int) -> torch.Tensor:
    """The CPU model's rows of B x (positions, hidden) of conv layer L (from position 0)."""
    rows: list[torch.Tensor] = []
    conv = cpu.conv

    def keep(layer: int, bx: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        if layer == L:
            rows.append(bx.clone())
        return conv(layer, bx, hist)

    cpu.conv = keep
    try:
        cpu.run(ids, ref.cpu.LfmState())
    finally:
        cpu.conv = conv
    return torch.cat(rows)


def test_lfm_overlay_steps() -> None:
    """Steps of 1 .. 4 slots give the logits of steps of one, bit for bit; so do steps that continue from fewer slots
    than the last one ran (a generation's rejected drafts: the slots past the accepted ones hold other tokens)."""
    g = ref.gguf()
    layers = [0, 1, 2, 3, 4]
    ids = ref.essay_ids(140)
    ov = LfmOverlay(g, layers=layers, vocab=VOCAB, max_positions=512)
    ov.load()
    cpu = ref.cpu.LfmCpu(g, layers=[], vocab=0)
    x = cpu.emb[torch.tensor(ids)]
    one = _run(ov, x, (1,))
    ov.reset()
    assert torch.equal(_run(ov, x, STEPS), one)
    ov.reset()
    rng = np.random.default_rng(0)
    p = 0
    while p < len(ids):  # (each step: up to 4 slots, the accepted ones the text's, then other tokens)
        n = min(int(rng.integers(1, 5)), len(ids) - p)
        acc = int(rng.integers(1, n + 1))
        xs = x[p:p + n].clone()
        xs[acc:] = cpu.emb[torch.from_numpy(rng.integers(0, 120000, n - acc))]
        got = to_torch(ov.step(xs, p)).float()
        assert torch.equal(got[:acc], one[p:p + acc]), p
        p += acc
