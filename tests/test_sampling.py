"""Sampling on the host: the native top-k of a vocabulary's logits (``models.common.top_logits``, ``max_prob``), llama.cpp's
penalties (``models.common.Penalties``) on a few candidates against their definition on the whole vocabulary, the
choice among the candidates (``Sampling.pick``), and the generation loops' window of recent tokens
(``NpuModel.generate``, the overlay models' bf16 logits with a soft cap)."""

import collections

import ml_dtypes
import numpy as np
import pytest
import torch

from artlite import _native
from models.common import NpuModel, OverlayModel, Penalties, Sampling, max_prob, seed, top_logits

VOCAB = 4096
CAP = 30.0


def bf16(x: np.ndarray) -> np.ndarray:
    """fp32 values rounded to bf16, as their bits (an overlay's logits)."""
    return x.astype(ml_dtypes.bfloat16).view(np.uint16)


def values(x: np.ndarray) -> np.ndarray:
    """The fp32 values of logits (bf16 bits or fp32)."""
    return (x.astype(np.uint32) << 16).view(np.float32) if x.dtype == np.uint16 else x


def softcap(x: np.ndarray) -> np.ndarray:
    return (np.tanh(x / CAP) * CAP).astype(np.float32)


def reference(logits: np.ndarray, tokens: list[int], s: Sampling) -> np.ndarray:
    """llama.cpp's penalties sampler on all the logits (fp32): the last ``repeat_last_n`` of the tokens (an image's
    rows, negative, left out)."""
    x = logits.astype(np.float32).copy()
    recent = [t for t in tokens if t >= 0][-s.repeat_last_n:] if s.penalized else []
    for t, c in collections.Counter(recent).items():
        x[t] = x[t] * s.repeat_penalty if x[t] <= 0 else x[t] / s.repeat_penalty
        x[t] -= c * s.frequency_penalty + s.presence_penalty
    return x


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("n", [1, 31, 33, 129, 4096, 128256, 262144])
def test_top_logits(n: int, dtype: str) -> None:
    """The k largest of logits (sizes past whole vectors, many equal values, -0 and +0) as a stable sort by value
    orders them (equal ones: the lower index first), with excluded logits (``at`` without penalties ranks them as the
    others), a soft cap and the logits' tokens (``rows``)."""
    rng = np.random.default_rng(n)
    for ties in (False, True):
        x = (rng.standard_normal(n) * 3).astype(np.float32)
        if ties:
            x = np.round(x)  # (-0 among them)
        x = bf16(x) if dtype == "bf16" else x
        v = values(x)
        order = np.lexsort((np.arange(n), -v.astype(np.float64)))  # (-0 == +0: the lower index first)
        rows = rng.permutation(n).astype(np.int32)
        for k in (1, 8, 64, 300):
            got_v, got_i = top_logits(x, k)
            want = order[:k]
            np.testing.assert_array_equal(got_i, want, err_msg=f"k {k}, ties {ties}")
            np.testing.assert_array_equal(got_v, v[want])
            got_v, got_i = top_logits(x, k, CAP, rows=rows)
            np.testing.assert_array_equal(got_i, rows[want])
            np.testing.assert_allclose(got_v, softcap(v[want]), rtol=1e-6, atol=1e-6)
            at = order[1:4].astype(np.int32)  # (three of the largest: penalties of 1 keep their values)
            got_v, got_i = _native.top_logits(x, k, 0.0, at, np.ones(len(at), np.float32))
            np.testing.assert_array_equal(got_v, v[want])  # (equal values: the others first)
            np.testing.assert_array_equal(v[got_i], got_v)


def test_max_prob() -> None:
    """A vocabulary's largest bf16 logit (the first of equal ones) and its softmax probability."""
    rng = np.random.default_rng(0)
    for n in (5, 100, 128256, 262144):
        x = bf16((rng.standard_normal(n) * 4).astype(np.float32))
        x[n // 2] = x[n // 3] = bf16(np.array([30.0], np.float32))[0]
        v = values(x).astype(np.float64)
        i, p = max_prob(x)
        assert i == n // 3
        assert p == pytest.approx(1 / np.exp(v - v.max()).sum(), rel=1e-4)


def test_pick() -> None:
    """The choice among candidates: softmax at the temperature, top_p's smallest prefix, min_p, the draw u; and the
    draws of a seed."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = np.sort(rng.standard_normal(64).astype(np.float32) * 2)[::-1].copy()
        temp, top_p, min_p, u = rng.uniform(0.3, 1.5), rng.choice([1.0, 0.95, 0.5]), rng.choice([0.0, 0.05]), rng.random()
        p = np.exp(v.astype(np.float64) / temp - v[0] / temp)
        p /= p.sum()
        keep = (np.cumsum(p) - p < top_p) & (p >= min_p * p[0])
        keep[0] = True
        p = np.where(keep, p, 0.0)
        want = int(np.searchsorted(np.cumsum(p / p.sum()), u, side="right"))
        assert _native.pick(v, temp, top_p, min_p, u) == want
    s, ids = Sampling(temp=1.0, top_k=64), np.arange(100, 164, dtype=np.int32)
    seed(7)
    first = [s.pick(v, ids) for _ in range(20)]
    seed(7)
    assert [s.pick(v, ids) for _ in range(20)] == first
    assert len(set(first)) > 1


@pytest.mark.parametrize("dtype", ["fp32", "bf16-cap"])
@pytest.mark.parametrize("s", [
    Sampling(top_k=8, repeat_penalty=1.3, repeat_last_n=16),
    Sampling(top_k=1, repeat_penalty=1.1, repeat_last_n=64, presence_penalty=0.5, frequency_penalty=0.3),
    Sampling(top_k=40, repeat_penalty=0.7, repeat_last_n=24, presence_penalty=-2.0),  # (penalties that raise)
    Sampling(top_k=8, repeat_last_n=0, repeat_penalty=1.5),  # (none)
], ids=["repeat", "greedy-all", "raise", "off"])
def test_penalties(s: Sampling, dtype: str) -> None:
    """The k largest penalized logits at each step of a sequence: a prompt with an image's rows, then tokens accepted
    one by one (the window full, its oldest leaving), among them the largest logits' tokens and negative ones; the
    same logits in another order (a drafter's: ``rows``, ``where``)."""
    rng = np.random.default_rng(1)
    logits = (rng.standard_normal(VOCAB) * 4).astype(np.float32)
    rows = rng.permutation(VOCAB).astype(np.int32)  # (logit j: token rows[j])
    where = np.empty_like(rows)
    where[rows] = np.arange(VOCAB, dtype=np.int32)
    order = np.argsort(-logits, kind="stable").tolist()
    prompt = [order[3], -1, -1, order[0], order[-1], *rng.integers(VOCAB, size=40).tolist(), order[1]]
    p = Penalties(s, prompt)
    tokens = list(prompt)
    cap = CAP if dtype == "bf16-cap" else 0.0
    for step in range(30):
        raw = bf16(logits) if cap else logits
        v, i = p.top(raw, s.top_k, cap)
        want = reference(softcap(values(raw)) if cap else raw, tokens, s)
        close = {"rtol": 1e-6, "atol": 1e-6, "err_msg": f"step {step}"}
        np.testing.assert_allclose(v, np.sort(want)[::-1][:s.top_k], **close)  # (the k largest; ties: any of them)
        np.testing.assert_allclose(want[i], v, **close)
        pv, pi = p.top(raw[rows], s.top_k, cap, rows=rows, where=where)
        np.testing.assert_allclose(pv, v, **close)
        np.testing.assert_allclose(want[pi], pv, **close)
        t = order[step % 5] if step % 3 else int(i[0])
        p.accept(t)
        tokens.append(t)
        logits = np.roll(logits, 1)


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
    softcap = CAP

    def __init__(self) -> None:
        self.position = 0

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
