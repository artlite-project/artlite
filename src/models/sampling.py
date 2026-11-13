"""How the next token is chosen: ``Sampling`` (temperature, top-k, top-p, min-p, penalties), llama.cpp's penalties
sampler over a generation (``Penalties``), a constraint on the tokens (``Mask``), and the native top-k of a
vocabulary's logits (``top_logits``, ``argmax``, ``max_prob``). The draws come from one generator (``seed``)."""

import collections
import copy
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ml_dtypes import bfloat16

from artlite import _native

from .sequence import MEDIA, token_ids

LogitsLike = np.ndarray  # a vocabulary's logits: bf16 (``ml_dtypes.bfloat16``: an overlay's) or fp32
_rng = random.Random()  # (sampling's draws: ``seed``)


def seed(n: int | None = None) -> None:
    """Seed sampling's draws (None: from the operating system's randomness)."""
    _rng.seed(n)


@dataclass(frozen=True)
class Sampling:
    """How the next token is chosen: the logits penalized as llama.cpp's penalties sampler does (``Penalties``: the
    tokens among the last ``repeat_last_n`` of the prompt and the generated ones), among the ``top_k`` largest of them,
    softmax at temperature ``temp``, then the smallest prefix of them reaching probability ``top_p``, of those the ones
    with at least ``min_p`` times the largest probability; the argmax when ``greedy`` or ``temp`` <= 0."""

    temp: float = 1.0
    top_k: int = 64
    top_p: float = 1.0
    greedy: bool = False
    min_p: float = 0.0
    repeat_penalty: float = 1.0  # a recent token's logit divided by it (multiplied when <= 0); 1: none
    repeat_last_n: int = 64  # the recent tokens (0: no penalties)
    presence_penalty: float = 0.0  # subtracted from a recent token's logit
    frequency_penalty: float = 0.0  # subtracted from a recent token's logit per occurrence
    banned: tuple[int, ...] = ()  # tokens never chosen (their logits -inf: LFM3-Audio's text-only output)

    def __post_init__(self) -> None:
        if not 0 < self.repeat_penalty < math.inf:
            raise ValueError(f"repeat_penalty {self.repeat_penalty}: greater than 0 (1: none)")
        if self.repeat_last_n < 0:
            raise ValueError(f"repeat_last_n {self.repeat_last_n}: at least 0 (0: no penalties)")

    @property
    def k(self) -> int:
        """The candidates a choice needs (1: the argmax)."""
        return 1 if self.greedy or self.temp <= 0 else self.top_k

    @property
    def penalized(self) -> bool:
        """Whether the penalties change logits."""
        return self.repeat_last_n > 0 and (self.repeat_penalty != 1.0 or self.presence_penalty != 0.0
                                           or self.frequency_penalty != 0.0)

    def pick(self, values: np.ndarray, ids: np.ndarray) -> int:
        """The next token from the ``k`` largest logits (values descending, fp32) and their token ids
        (``_native.pick``; the draw: ``seed``)."""
        if self.k == 1:
            return int(ids[0])
        return int(ids[_native.pick(values, self.temp, self.top_p, self.min_p, _rng.random())])

    def probs(self, values: np.ndarray) -> np.ndarray:
        """The probabilities ``pick`` draws from (of its candidates, in their order; 0: filtered out)."""
        if self.k == 1:
            return np.eye(1, len(values), dtype=np.float32)[0]
        q = np.exp((values - values[0]) / np.float64(self.temp))
        q /= q.sum()
        keep = (np.concatenate([[0.0], np.cumsum(q)[:-1]]) < self.top_p) & (q >= self.min_p * q[0])
        keep[0] = True
        q = np.where(keep if self.top_p < 1.0 or self.min_p > 0.0 else True, q, 0.0)
        return (q / q.sum()).astype(np.float32)


def top_logits(logits: LogitsLike, k: int, cap: float = 0.0, *, rows: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """The k largest of a vocabulary's logits after a soft cap (``cap`` > 0: cap tanh(l / cap)), fp32, descending
    (equal ones: the lower index first), and their token ids (int32; ``rows``: the token of each logit, else its index):
    one pass over the logits (``_native.top_logits``)."""
    return _native.top_logits(logit_bits(logits), k, cap, rows=rows)


def argmax(logits: LogitsLike) -> int:
    """The index of the largest logit (the first of equal ones)."""
    return int(_native.top_logits(logit_bits(logits), 1)[1][0])


def logit_bits(logits: LogitsLike) -> np.ndarray:
    """bf16 logits as their bits (uint16: what the native code takes), fp32 as they are."""
    return logits.view(np.uint16) if logits.dtype == bfloat16 else logits


def max_prob(logits: LogitsLike) -> tuple[int, float]:
    """The largest of bf16 logits (its index: the first of equal ones) and its probability (softmax)."""
    return _native.max_prob(logit_bits(logits))


# a generation's observer of each choice (``NpuModel.generate``'s ``probs``): the logits the token was chosen from (a view
# valid until the next dispatch), the token, the model's soft cap of them, the candidates the sampler drew from
Probs = Callable[[LogitsLike, int, float, tuple[np.ndarray, np.ndarray]], None]


class Mask(Protocol):
    """A constraint on a generation's tokens (a tool call's name: ``server.tools.NameMask``)."""

    def allowed(self) -> np.ndarray | None:
        """The tokens the next one may be (sorted ids), or None: any."""
        ...

    def accept(self, token: int) -> None:
        """The sequence's next token."""
        ...


class Penalties:
    """llama.cpp's penalties sampler over a generation: the tokens among the last ``repeat_last_n`` of the prompt
    (``prompt``: its ids; an image's rows, negative, are none) and of the generated ones (``accept``) have their logits
    divided by ``repeat_penalty`` (multiplied when <= 0), less ``presence_penalty`` and ``frequency_penalty`` times
    their count. Nothing when the sampling has no penalties (``Sampling.penalized``). ``mask``: the candidates are only
    the tokens it allows (it follows the accepted tokens too)."""

    def __init__(self, sampling: Sampling, prompt: Sequence[int] = (), mask: Mask | None = None):
        self.sampling = sampling
        self.n = sampling.repeat_last_n if sampling.penalized else 0
        self.last: collections.deque[int] = collections.deque()
        self.counts: collections.Counter[int] = collections.Counter()
        self.mask: Mask | None = None
        tail: list[int] = []
        for t in reversed(token_ids(prompt)):
            if len(tail) == self.n:
                break
            if t != MEDIA:
                tail.append(int(t))
        for t in reversed(tail):
            self.accept(t)
        self.mask = mask

    @property
    def active(self) -> bool:
        return self.n > 0

    def copy(self) -> "Penalties":
        """Another window of the same tokens (a draft's: without the mask)."""
        out = copy.copy(self)
        out.last, out.counts, out.mask = self.last.copy(), self.counts.copy(), None
        return out

    def accept(self, token: int) -> None:
        """The sequence's next token."""
        if self.mask is not None:
            self.mask.accept(token)
        if not self.n:
            return
        self.last.append(token)
        self.counts[token] += 1
        if len(self.last) > self.n:
            old = self.last.popleft()
            self.counts[old] -= 1
            if not self.counts[old]:
                del self.counts[old]

    def top(self, logits: LogitsLike, k: int, cap: float = 0.0, *, rows: np.ndarray | None = None,
            where: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` largest penalized logits (fp32, descending) and their token ids, of a token's logits (vocab,), fp32
        or the overlay's bf16 (``top_logits``; ``cap``: the model's soft cap); ``rows``, ``where``: the token of each
        logit and the logit of each token (logits in another order: a drafter's). The candidates: the recent tokens
        and the ``k`` largest logits of the others; with the mask, the tokens it allows (logits in the vocabulary's
        order); never the sampling's ``banned`` ones."""
        ban = self.sampling.banned
        if not ban:
            return self._top(logits, k, cap, rows=rows, where=where)
        v, i = self._top(logits, k + len(ban), cap, rows=rows, where=where)
        keep = ~np.isin(i, ban)
        return v[keep][:k], i[keep][:k]

    def _top(self, logits: LogitsLike, k: int, cap: float = 0.0, *, rows: np.ndarray | None = None,
             where: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        allowed = self.mask.allowed() if self.mask is not None else None
        if allowed is not None:
            return self._top_of(logits, k, cap, allowed)
        if not self.counts:
            return top_logits(logits, k, cap, rows=rows)
        ids = np.fromiter(self.counts, np.int32, len(self.counts))
        s, count = self.sampling, np.fromiter(self.counts.values(), np.float32, len(ids))
        return _native.top_logits(logit_bits(logits), k, cap, ids if where is None else where[ids], count, s.repeat_penalty,
                                  s.presence_penalty, s.frequency_penalty, rows)

    def _top_of(self, logits: LogitsLike, k: int, cap: float, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` largest penalized logits of the ``allowed`` tokens (sorted ids) and their ids."""
        sub = logits[allowed]
        k = min(k, len(allowed))
        if not self.counts:
            return top_logits(sub, k, cap, rows=allowed)
        ids = np.fromiter(self.counts, np.int32, len(self.counts))
        count = np.fromiter(self.counts.values(), np.float32, len(ids))
        at = np.minimum(np.searchsorted(allowed, ids), len(allowed) - 1)
        hit = allowed[at] == ids
        s = self.sampling
        return _native.top_logits(logit_bits(sub), k, cap, at[hit].astype(np.int32), count[hit], s.repeat_penalty,
                                  s.presence_penalty, s.frequency_penalty, allowed)
