"""Sampling on the host: llama.cpp's penalties (``models.common.Penalties``) on a few candidates against their
definition on the whole vocabulary, and the generation loops' window of recent tokens (``NpuModel.generate``, the
overlay models' bf16 logits with a soft cap)."""

import collections

import pytest
import torch

from models.common import NpuModel, OverlayModel, Penalties, Sampling

VOCAB = 4096


def softcap(x: torch.Tensor) -> torch.Tensor:
    return 30.0 * torch.tanh(x / 30.0)


def reference(logits: torch.Tensor, tokens: list[int], s: Sampling) -> torch.Tensor:
    """llama.cpp's penalties sampler on all the logits (fp32): the last ``repeat_last_n`` of the tokens (an image's
    rows, negative, left out)."""
    x = logits.float().clone()
    recent = [t for t in tokens if t >= 0][-s.repeat_last_n:] if s.penalized else []
    for t, c in collections.Counter(recent).items():
        x[t] = x[t] * s.repeat_penalty if x[t] <= 0 else x[t] / s.repeat_penalty
        x[t] -= c * s.frequency_penalty + s.presence_penalty
    return x


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-cap"])
@pytest.mark.parametrize("s", [
    Sampling(top_k=8, repeat_penalty=1.3, repeat_last_n=16),
    Sampling(top_k=1, repeat_penalty=1.1, repeat_last_n=64, presence_penalty=0.5, frequency_penalty=0.3),
    Sampling(top_k=40, repeat_penalty=0.7, repeat_last_n=24, presence_penalty=-2.0),  # (penalties that raise)
    Sampling(top_k=8, repeat_last_n=0, repeat_penalty=1.5),  # (none)
], ids=["repeat", "greedy-all", "raise", "off"])
def test_penalties(s: Sampling, bf16: bool) -> None:
    """The k largest penalized logits at each step of a sequence: a prompt with an image's rows, then tokens accepted
    one by one (the window full, its oldest leaving), among them the largest logits' tokens and negative ones; the
    same logits in another order (a drafter's: ``rows``, ``where``)."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(VOCAB, generator=g) * 4
    rows = torch.randperm(VOCAB, generator=g)  # (logit j: token rows[j])
    where = torch.empty_like(rows)
    where[rows] = torch.arange(VOCAB)
    order = torch.argsort(logits, descending=True).tolist()
    prompt = [order[3], -1, -1, order[0], order[-1], *torch.randint(VOCAB, (40,), generator=g).tolist(), order[1]]
    p = Penalties(s, prompt)
    tokens = list(prompt)
    for step in range(30):
        raw = logits.bfloat16() if bf16 else logits
        v, i = p.top(raw, s.top_k, softcap if bf16 else None)
        want = reference(softcap(raw.float()) if bf16 else raw, tokens, s)
        close = {"rtol": 1e-6, "atol": 1e-6, "msg": f"step {step}"}
        torch.testing.assert_close(v, torch.topk(want, s.top_k).values, **close)  # (the k largest; ties: any of them)
        torch.testing.assert_close(want[i], v, **close)
        pv, pi = p.top(raw[rows], s.top_k, softcap if bf16 else None, rows=rows, where=where)
        torch.testing.assert_close(pv, v, **close)
        torch.testing.assert_close(want[pi], pv, **close)
        t = order[step % 5] if step % 3 else int(i[0])
        p.accept(t)
        tokens.append(t)
        logits = logits.roll(1)


class Steps(NpuModel):
    """A model whose logits after any token prefer 5, then 6, then 7 (by 0.1), the others far below."""

    eos_token_ids = (0,)
    capacity = 64

    def __init__(self) -> None:
        self.position = 0

    def step(self, token: int) -> torch.Tensor:
        self.position += 1
        out = torch.full((16,), -5.0)
        out[5:8] = torch.tensor([2.0, 1.9, 1.8])
        return out


class OverlaySteps(OverlayModel):
    """The same logits as an overlay's (bf16, from ``steps``) under a soft cap."""

    eos_token_ids = (0,)
    capacity = 64

    def __init__(self) -> None:
        self.position = 0

    def cap(self, logits: torch.Tensor) -> torch.Tensor:
        return softcap(logits)

    def steps(self, tokens: list[int], *, logits: bool = True) -> torch.Tensor:
        self.position += len(tokens)
        return Steps().step(0).bfloat16()[None].repeat(len(tokens), 1)


@pytest.mark.parametrize("model", [Steps, OverlaySteps])
def test_generate(model: type[NpuModel]) -> None:
    """Greedy: the same token without penalties; with them, the next best once the window (2 tokens) holds the best,
    from the prompt's tokens on."""
    npu = model()
    first = Steps().step(0)
    greedy = Sampling(greedy=True)
    assert list(npu.generate(first, 6, greedy)) == [5] * 6
    s = Sampling(greedy=True, repeat_penalty=1.5, repeat_last_n=2)
    assert list(npu.generate(first, 6, s)) == [5, 6, 7, 5, 6, 7]
    assert list(npu.generate(first, 4, s, prompt=[5, 6])) == [7, 5, 6, 7]
    assert list(npu.generate(first, 4, s, prompt=[5, 6, -1])) == [7, 5, 6, 7]  # (an image's row: not a token)
    with pytest.raises(ValueError, match="repeat_penalty"):
        Sampling(repeat_penalty=0.0)
    with pytest.raises(ValueError, match="repeat_last_n"):
        Sampling(repeat_last_n=-1)
