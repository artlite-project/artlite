"""A screened LM head: the NPU reads the head's high plane (``formats.q6h``: a Q6_K head's scales and the top 2 bits of
its codes, 39% of its bytes), and the host computes the logits that can matter exactly, from the model file's own rows
(``GGUFFile.rows``, as an embedding lookup).

The plane's logit of row i is the exact one plus e_i = sum_k (d sc)_ik (l_ik - 8) g_k (l: the codes' low 4 bits, g: the
head's input, the normed hidden state), of deviation sigma_i = ``SIGMA`` A_i |g| with A_i the rms of the row's scales
d sc (a number per row, from the file's scales at load): ~0.7 of a logit on Qwen3.6-35B-A3B. A logit lies within
``Z`` sigma_i of its screened value: its bounds. The rows computed exactly for the k largest logits
(``Screen.candidates``; the passes over the vocabulary: ``_native.screen_*``, which read the logits and bounds and write
nothing):

1. those whose upper bound reaches the largest lower bound: the largest logit M is among them (an argmax: ~9 rows);
2. for k > 1, those whose upper bound reaches both M - ``WINDOW`` (a logit further below the largest weighs e^-WINDOW
   of it) and the k-th largest of step 1's exact logits, ``ROWS`` at most (the highest upper bounds): the k largest
   logits, as far as they lie in the window (a sampler's 20 candidates: ~70 rows).

A row is only left out when its upper bound is below another row's exact value or lower bound: a miss takes an error
beyond ``Z`` sigma (4 sigma lost 1 of 200 steps' 20th candidates on a generation of the 35B-A3B, 5 none).

``Screened`` is a step's logits as the overlays return them, a ``ScreenedRow`` per slot: ``top(k)`` its k largest
logits (what ``models.sampling`` takes: fp32 sums of the file's weights against g, closer to the model than the cores'
bf16 logits, whose argmax differs from the exact one's in ~0.5% of a generation's steps), ``at`` the exact logits of
given tokens (a mask's), and as an array (vocab,) fp32: those exact, every other logit its screened value, below them.
"""

import sys
from collections.abc import Callable

import numpy as np
from ml_dtypes import bfloat16

from .. import _native
from ..formats import q6h
from ..gguf_model import GGUFFile

SIGMA = 5.0  # deviation of a code's low 4 bits from 8 (uniform: 4.64; Qwen3.6-35B-A3B's head: 4.86, their mean 7.1)
Z = 5.0  # a logit's bounds: Z sigma_i around its screened value
K = 64  # the largest logits that are exact when a row is taken as an array (a sampler's own k: ``ScreenedRow.top``)
WINDOW = 10.0  # ... down to this far below the largest (e^-10: 5e-5 of its probability at temperature 1)
ROWS = 2048  # step 2's rows at most (a flat distribution: the highest upper bounds)
IDS = 16384  # the rows a native pass collects (past them: numpy's)
BLOCK = 210  # a Q6_K block: 128 B of low nibbles, 64 B of high crumbs, 16 int8 scales, the fp16 d


def screens(qtype: str) -> bool:
    """Whether an LM head of this GGUF type can be screened (its high plane has a tile format)."""
    return qtype == q6h.SOURCE


class Screen:
    """The host side of a screened LM head: tensor ``name`` of the file (its rows the token ids'), normed by ``norm``
    (hidden,) fp32 with ``eps``; ``n``: the vocabulary's rows."""

    def __init__(self, gguf: GGUFFile, name: str, norm: np.ndarray, eps: float, n: int):
        t = gguf.tensor(name)
        assert screens(t.qtype) and n <= t.rows, (name, t.qtype, n)
        self.gguf, self.name, self.eps, self.n = gguf, name, eps, n
        self.norm = np.ascontiguousarray(norm, np.float32)
        blocks = t.blocks(BLOCK)
        self.bound = np.empty(n, np.float32)  # Z sigma_i per unit |g|
        for a in range(0, n, 8192):  # (the scales alone: 18 B of a block's 210)
            b = blocks[a:min(a + 8192, n)]
            d = np.ascontiguousarray(b[..., 208:210]).view(np.float16)[..., 0].astype(np.float32)
            sc = np.ascontiguousarray(b[..., 192:208]).view(np.int8).astype(np.float32)
            self.bound[a:a + 8192] = np.sqrt(((d[..., None] * sc) ** 2).mean((1, 2)))
        self.bound *= np.float32(SIGMA * Z)
        self._ids = np.empty(IDS, np.int32)
        self._pool: list[np.ndarray] = []  # (``_row``)
        self.rows = 0  # (rows computed exactly so far)
        self.taken = 0  # (choices so far: ``candidates``)

    def input(self, x: np.ndarray) -> np.ndarray:
        """The head's input g (hidden,) fp32 of a slot's residual x (the bf16 values the head's phase read): x w / rms(x)."""
        x = np.asarray(x, np.float64)
        return (x * (1.0 / np.sqrt(np.dot(x, x) / x.size + self.eps)) * self.norm).astype(np.float32)

    def exact(self, ids: np.ndarray, g: np.ndarray) -> np.ndarray:
        """The exact logits (len(ids),) fp32 of the tokens ``ids`` against the head's input g."""
        self.rows += len(ids)
        return self.gguf.rows(self.name, ids, prefetch=False) @ g

    def _above(self, bits: np.ndarray, gnorm: float, thr: float) -> np.ndarray:
        """The rows whose upper bound reaches thr, ascending (int32; a copy)."""
        n = _native.screen_above(bits, self.bound, gnorm, thr, self._ids)
        if n <= len(self._ids):
            return self._ids[:n].copy()
        return np.flatnonzero(bits.view(bfloat16).astype(np.float32) + self.bound * np.float32(gnorm) >= thr).astype(np.int32)

    def candidates(self, screened: np.ndarray, g: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The rows computed exactly for the k largest logits of a slot (the module docstring's two steps), from its
        screened logits (n,) bf16 and the head's input g: their tokens (int32, in no order) and exact logits (fp32)."""
        bits, gnorm = screened.view(np.uint16), float(np.linalg.norm(g))
        ids = self._above(bits, gnorm, _native.screen_lower(bits, self.bound, gnorm))
        e = self.exact(ids, g)
        if k > 1:
            thr = float(e.max()) - WINDOW
            if len(e) >= k:
                thr = max(thr, float(np.partition(e, -k)[-k]))
            more = self._above(bits, gnorm, thr)
            more = more[~np.isin(more, ids, assume_unique=True)]
            if len(more) > ROWS:
                up = screened[more].astype(np.float32) + self.bound[more] * np.float32(gnorm)
                more = np.sort(more[np.argpartition(up, -ROWS)[-ROWS:]])
            ids, e = np.concatenate([ids, more]), np.concatenate([e, self.exact(more, g)])
        self.taken += 1
        return ids, e

    def top(self, screened: np.ndarray, g: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The k largest logits of a slot, fp32, descending (equal ones: the lower token first), and their tokens
        (int32): the exact ones (``candidates``), then, where fewer than k lie in the window, the largest screened ones
        of the other tokens, below them."""
        ids, e = self.candidates(screened, g, k)
        order = np.lexsort((ids, -e))[:k]
        v, i = e[order], ids[order]
        if len(i) < min(k, self.n):
            sv, si = _native.top_logits(screened.view(np.uint16), min(k + len(ids), self.n))
            keep = ~np.isin(si, ids)
            sv, si = sv[keep][:k - len(i)], si[keep][:k - len(i)]
            v = np.concatenate([v, np.minimum(sv, np.nextafter(e.min(), np.float32(-np.inf)))])
            i = np.concatenate([i, si])
        return v, i

    def logits(self, screened: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
        """A slot's logits (n,) fp32: the k largest exact (``candidates``), the others their screened values, below
        the k-th of those."""
        ids, e = self.candidates(screened, g, k)
        floor = np.partition(e, -k)[-k] if len(e) >= k else e.min()
        a = self._row()
        _native.screen_dense(screened.view(np.uint16), float(np.nextafter(np.float32(floor), np.float32(-np.inf))), a)
        a[ids] = e
        return a

    def _row(self) -> np.ndarray:
        """A buffer (n,) fp32 for a row: one of the pool's that nothing else holds any more, else a new one."""
        for buf in self._pool:
            if sys.getrefcount(buf) == 3:  # (the pool's, this loop's, getrefcount's)
                return buf
        self._pool.append(np.empty(self.n, np.float32))
        return self._pool[-1]


class ScreenedRow:
    """A slot's logits of a screened head: ``top(k)`` its k largest (exact), ``at`` the exact logits of given tokens,
    ``dense(k)`` and as an array (vocab,) fp32 with the k (``K``) largest exact. Computed when taken (``Screened``)."""

    dtype = np.dtype(np.float32)
    ndim = 1

    def __init__(self, screen: Screen, screened: np.ndarray, x: np.ndarray, alive: Callable[[], bool]):
        self.screen, self.screened, self.x, self.alive = screen, screened, x, alive
        self.shape = screened.shape
        self._g: np.ndarray | None = None
        self._top: tuple[int, np.ndarray, np.ndarray] | None = None
        self._dense: tuple[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def _input(self) -> np.ndarray:
        if not self.alive():
            raise RuntimeError("a screened step's logits are taken before the overlay's next dispatch of logits")
        if self._g is None:
            self._g = self.screen.input(self.x)
        return self._g

    def top(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The k largest logits, fp32, descending (equal ones: the lower token first), and their tokens (int32)."""
        if self._top is None or self._top[0] < k:
            self._top = (k, *self.screen.top(self.screened, self._input(), k))
        return self._top[1][:k], self._top[2][:k]

    def at(self, ids: np.ndarray) -> np.ndarray:
        """The exact logits (len(ids),) fp32 of the tokens ``ids``."""
        return self.screen.exact(np.asarray(ids), self._input())

    def dense(self, k: int = K) -> np.ndarray:
        """(vocab,) fp32: the k largest logits exact (those within the window), the others screened, below them."""
        if self._dense is None or self._dense[0] < k:
            self._dense = (k, self.screen.logits(self.screened, self._input(), k))
        return self._dense[1]

    def __getitem__(self, i: int | np.ndarray) -> np.ndarray:
        return self.dense()[i]

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        out = self.dense()
        return out if dtype is None else out.astype(dtype)


class Screened:
    """A step's logits of a screened head: a ``ScreenedRow`` per slot (``[i]``; as an array: (slots, vocab) fp32).
    ``screened``: the NPU's logits (slots, vocab) bf16, a view of the overlay's buffer: a row is taken before the next
    dispatch of logits (``alive``); ``x``: the slots' residuals (slots, hidden) the head read."""

    dtype = np.dtype(np.float32)
    ndim = 2

    def __init__(self, screen: Screen, screened: np.ndarray, x: np.ndarray, alive: Callable[[], bool]):
        assert screened.dtype == bfloat16 and len(x) == len(screened)
        self.shape = screened.shape
        self._rows = [ScreenedRow(screen, s, xi, alive) for s, xi in zip(screened, x, strict=True)]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int | slice) -> ScreenedRow | np.ndarray:
        if isinstance(i, slice):
            return np.stack([r.dense() for r in self._rows[i]])
        return self._rows[i]

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        out = self[:]
        return out if dtype is None else out.astype(dtype)
