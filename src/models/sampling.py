"""How the next token is chosen: ``Sampling`` (temperature, top-k, top-p, min-p, penalties), llama.cpp's penalties
sampler over a generation (``Penalties``), a constraint on the tokens (``Mask``), and the native top-k of a
vocabulary's logits (``top_logits``, ``argmax``, ``max_prob``). The draws come from a ``Sampling``'s own generator
(``rng``: a job's, where several generate at once), else from the process's (``seed``)."""

import collections
import copy
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from ml_dtypes import bfloat16

from artlite import _native
from artlite.decode.screen import SEEDS, ScreenedRow

from .sequence import MEDIA, token_ids

# a vocabulary's logits: bf16 (``ml_dtypes.bfloat16``: an overlay's) or fp32, or a screened head's row (``taken``)
LogitsLike = np.ndarray | ScreenedRow
_rng = random.Random()  # (the draws of a ``Sampling`` without a generator of its own: ``seed``)


def seed(n: int | None = None) -> None:
    """Seed the process's draws (None: from the operating system's randomness)."""
    _rng.seed(n)


TOP_K_ALL = 1000  # the candidates of top_k <= 0 (llama.cpp: all tokens; past these the probabilities are negligible)


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
    # the generator of its draws (None: the process's, ``seed``): one per job, so that a job's draws are its seed's
    # whatever generates beside it
    rng: random.Random | None = field(default=None, compare=False, repr=False)

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
        (``_native.pick``; the draw: ``rng``'s, else the process's)."""
        if self.k == 1:
            return int(ids[0])
        draw = (_rng if self.rng is None else self.rng).random()
        return int(ids[_native.pick(values, self.temp, self.top_p, self.min_p, draw)])

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
    one pass over the logits (``_native.top_logits``); a screened head's row: its own (``ScreenedRow.top``)."""
    if isinstance(logits, ScreenedRow) and rows is None:
        values, ids = logits.top(k)
        return _capped(values, cap), ids
    return _native.top_logits(logit_bits(logits), k, cap, rows=rows)


def _capped(values: np.ndarray, cap: float) -> np.ndarray:
    """Logits after a soft cap (``cap`` > 0: cap tanh(l / cap)), fp32."""
    return (np.tanh(values / np.float32(cap)) * np.float32(cap)).astype(np.float32) if cap > 0 else values


def argmax(logits: LogitsLike) -> int:
    """The index of the largest logit (the first of equal ones)."""
    if isinstance(logits, ScreenedRow):
        return int(logits.top(1)[1][0])
    return int(_native.top_logits(logit_bits(logits), 1)[1][0])


def logit_bits(logits: LogitsLike) -> np.ndarray:
    """bf16 logits as their bits (uint16: what the native code takes), fp32 as they are (a screened head's row: as
    an array, ``ScreenedRow.dense``)."""
    if isinstance(logits, ScreenedRow):
        return logits.dense()
    return logits.view(np.uint16) if logits.dtype == bfloat16 else logits


def max_prob(logits: LogitsLike) -> tuple[int, float]:
    """The largest of a vocabulary's logits (its index: the first of equal ones) and its probability (softmax)."""
    if isinstance(logits, ScreenedRow):
        x = logits.dense(1)
        i = int(np.argmax(x))
        return i, float(1.0 / np.exp(x.astype(np.float64) - float(x[i])).sum())
    return _native.max_prob(logit_bits(logits))


# a generation's observer of each choice (``NpuModel.generate``'s ``probs``): the logits the token was chosen from (a view
# valid until the next dispatch), the token, the model's soft cap of them, the candidates the sampler drew from
Probs = Callable[[LogitsLike, int, float, tuple[np.ndarray, np.ndarray]], None]


class Mask(Protocol):
    """A constraint on a generation's tokens (a grammar's: ``structured.GrammarMask``, which also forks for a drafter,
    cuts drafts, gives forced tokens and computes masks ahead: ``speculative.SpeculativeMixin.generate``)."""

    def bits(self) -> np.ndarray | None:
        """The tokens the next one may be as a vocabulary bitmask (uint32: bit i % 32 of word i // 32 for token i,
        llguidance's), or None: any."""
        ...

    def accept(self, token: int) -> None:
        """The sequence's next token."""
        ...

    def first_allowed(self, candidates: Sequence[int]) -> int:
        """The index of the first of ``candidates`` the next token may be, or -1 (a drafter's guesses, best first)."""
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
        """Another window of the same tokens (a drafter's guesses': the mask forked where it forks, ``GrammarMask.fork``:
        they follow its grammar; else without it)."""
        out = copy.copy(self)
        fork = getattr(self.mask, "fork", None)
        out.last, out.counts, out.mask = self.last.copy(), self.counts.copy(), fork() if fork is not None else None
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
        bits = self.mask.bits() if self.mask is not None else None
        if bits is not None:
            out = self._top_masked(logits, k, cap, bits, rows=rows, where=where)
            if len(out[0]):
                return out
            # (no candidate: a mask that allows none of these logits; the grammar's mask follows the next token)
        if not self.counts:
            return top_logits(logits, k, cap, rows=rows)
        ids = np.fromiter(self.counts, np.int32, len(self.counts))
        s, count = self.sampling, np.fromiter(self.counts.values(), np.float32, len(ids))
        if isinstance(logits, ScreenedRow) and rows is None and where is None:
            return self._top_screened(logits, k, cap, ids, count)
        return _native.top_logits(logit_bits(logits), k, cap, ids if where is None else where[ids], count, s.repeat_penalty,
                                  s.presence_penalty, s.frequency_penalty, rows)

    def _top_screened(self, logits: ScreenedRow, k: int, cap: float, ids: np.ndarray, count: np.ndarray,
                      bits: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``_native.top_logits``' candidates of a screened head's row: the k largest logits of the other tokens (a mask
        of all but the recent ones: the rows of a choice of k, not of k + their number), then the recent tokens' ``ids``
        (``count`` occurrences each) penalized, the k largest of all (equal ones: in that order); every one of them
        exact. ``bits``: of the tokens a mask allows (the recent ones too)."""
        s = self.sampling
        if bits is not None:
            ok = (bits[ids >> 5] >> (ids & 31).astype(np.uint32)) & 1 == 1
            ids, count = ids[ok], count[ok]
        others = np.full(-(-len(logits) // 32), 0xFFFFFFFF, np.uint32) if bits is None else bits.copy()
        np.bitwise_and.at(others, ids >> 5, ~(np.uint32(1) << (ids & 31).astype(np.uint32)))
        v, i = logits.top(k, others)
        v = _capped(v, cap)
        r = _capped(logits.at(ids), cap)
        r = np.where(r <= 0, r * np.float32(s.repeat_penalty), r / np.float32(s.repeat_penalty)) \
            - np.float32(s.presence_penalty) - count * np.float32(s.frequency_penalty)
        v, i = np.concatenate([v, r.astype(np.float32)]), np.concatenate([i, ids])
        order = np.argsort(-v, kind="stable")[:k]
        return v[order], i[order]

    def _top_masked(self, logits: LogitsLike, k: int, cap: float, bits: np.ndarray, *, rows: np.ndarray | None,
                    where: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` largest penalized logits of the tokens a mask allows (``bits``: a vocabulary bitmask) and their ids,
        in one pass over the logits (``_native.top_logits``' mask); a screened row's exact (``ScreenedRow.top``'s
        mask: as exact as an unmasked choice). ``rows``, ``where``: logits in another order (``top``)."""
        s = self.sampling
        ids = np.fromiter(self.counts, np.int32, len(self.counts))
        count = np.fromiter(self.counts.values(), np.float32, len(ids))
        if isinstance(logits, ScreenedRow) and rows is None and where is None:
            allowed = np.empty(SEEDS, np.int32)
            few = _native.mask_ids(bits, len(logits), allowed)
            if not few:  # (none: the caller's fallback)
                return np.empty(0, np.float32), np.empty(0, np.int32)
            if few <= len(allowed):  # (a few tokens allowed: their rows, exact)
                allowed = allowed[:few]
                sub = logits.at(allowed)
                at = np.searchsorted(allowed, ids)
                hit = (at < few) & (allowed[np.minimum(at, few - 1)] == ids)
                return _native.top_logits(sub, k, cap, at[hit].astype(np.int32), count[hit], s.repeat_penalty,
                                          s.presence_penalty, s.frequency_penalty, allowed)
            return self._top_screened(logits, k, cap, ids, count, bits)
        x = logit_bits(logits)
        if rows is not None:  # (a mask of the logits' order)
            bits = pack_mask(np.unpackbits(bits.view(np.uint8), bitorder="little")[rows], (len(x) + 31) // 32)
        elif len(x) > 32 * len(bits):  # (logits past the vocabulary's tokens: none allowed)
            bits = np.concatenate([bits, np.zeros((len(x) + 31) // 32 - len(bits), np.uint32)])
        at = ids if where is None else where[ids]
        return _native.top_logits(x, k, cap, at if len(at) else None, count if len(at) else None, s.repeat_penalty,
                                  s.presence_penalty, s.frequency_penalty, rows, bits)


def pack_mask(allowed: np.ndarray, words: int) -> np.ndarray:
    """A vocabulary bitmask (``Mask.bits``: ``words`` uint32 words) of a bool per token."""
    out = np.zeros(words * 4, np.uint8)
    b = np.packbits(allowed, bitorder="little")[:words * 4]
    out[:len(b)] = b
    return out.view(np.uint32)
