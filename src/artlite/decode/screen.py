"""A screened LM head: the NPU reads the head's plane (``PLANES``: a Q6_K head's high plane, ``formats.q6h``: its scales
and the top 2 bits of its codes, 39% of its bytes; an F16 / BF16 head's Q4_0 plane, quantized from its rows at load:
28% of its bytes), and the host computes the logits that can matter exactly, from the model file's own rows (read in
place, as an embedding lookup).

With a row's weights d sc (q - 32) = a (p + r), p the plane's part of a code and r the rest (its low 4 bits less 8), the
plane's logit of row i is sum a p g against the exact sum a (p + r) g (g: the head's input, the normed hidden state).
Per row, from the file's codes at load (``_native.screen_stats``): rho_i = sum a^2 r (p + r) / sum a^2 (p + r)^2, the
share of a logit the plane lacks in the mean (~0.083: a row's largest logits are 8% short, 2 - 3 logits of a token's
best), and what is left once that is put back, a deviation sigma_i = sqrt(mean a^2 (r - rho (p + r))^2) |g| / (1 - rho)
(~0.75 of a logit on Qwen3.6-35B-A3B; a Q4_0 plane's: rho ~0, and its largest 32-value block's deviation of what it
rounds away, ``_native.screen_stats_plane``: the input's outlier channels put a row's error in a few blocks). So a logit lies within ``Z`` sigma_i of its estimate, the screened value / (1 -
rho_i): its bounds (of a generation's 200 K largest logits the furthest lay 4.7 sigma off; without rho the largest
logits lie 4 - 6 sigma low). The rows computed exactly for the k largest logits (``Screen.candidates``; the passes
over the vocabulary: ``_native.screen_*``, which read the logits and the rows' numbers and write nothing):

1. the ``SEEDS`` largest screened ones (2 k where that is more): the largest of their exact logits, m, is a lower
   bound of the largest logit, their k-th largest one of the k-th;
2. those whose upper bound reaches m (and the largest lower bound): the largest logit M is among them (an argmax: a
   few rows more);
3. for k > 1, those whose upper bound reaches both M - ``WINDOW`` (a logit further below the largest weighs e^-WINDOW
   of it) and the k-th largest of the exact logits so far: the k largest logits, as far as they lie in the window (a
   sampler's 20 candidates: ~100 rows in all). Steps 2 and 3: ``ROWS`` at most each (a flat distribution: the highest
   upper bounds).

A row is only left out when its upper bound is below another row's exact value or lower bound.

A mask (a constraint's tokens: ``models.sampling.Mask``) restricts every step to the tokens it allows: the k largest of
those are as exact as an unmasked choice's.

``Screened`` is a step's logits as the overlays return them, a ``ScreenedRow`` per slot: ``top(k)`` its k largest
logits (what ``models.sampling`` takes: fp32 sums of the file's weights against g, closer to the model than the cores'
bf16 logits, whose argmax differs from the exact one's in ~0.5% of a generation's steps), ``at`` the exact logits of
given tokens (a mask's), ``dense(k)`` (vocab,) fp32 with those exact and every other logit its estimate, below them (a
row to keep or to sum: the tail's ~0.7 costs a whole distribution ~0.004 nats of KL), and as an array every logit exact
(``exact``: the whole head on the host, ~0.15 us per row and 1024 columns: tests and quality tools).
"""

from collections.abc import Callable

import numpy as np
from ml_dtypes import bfloat16

from .. import _native, settings
from ..formats import q6h
from ..gguf_model import GGUFFile

Z = 6.0  # a logit's bounds: Z sigma_i around its estimate
K = 64  # the largest logits that are exact when a row is taken as an array (a sampler's own k: ``ScreenedRow.top``)
WINDOW = 10.0  # ... down to this far below the largest (e^-10: 5e-5 of its probability at temperature 1)
SEEDS = 16  # step 1's rows
ROWS = 2048  # the rows of step 2, of step 3 at most
IDS = 16384  # the rows a native pass collects (past them: numpy's)
FILE_FORMATS = 3  # a screened head: in files of this many tile formats at most (``screened``)


PLANES = {q6h.SOURCE: "Q6_KH", "F16": "Q4_0", "BF16": "Q4_0"}  # an LM head's GGUF type -> its plane's tile format
HALF = ("F16", "BF16")  # (heads whose plane the image quantizes at load: ``derived``)


def screens(qtype: str) -> bool:
    """Whether an LM head of this GGUF type can be screened (its plane has a tile format)."""
    return qtype in PLANES


def plane(qtype: str) -> str:
    """The tile format of an LM head's plane."""
    return PLANES[qtype]


def derived(fmt: str, qtype: str) -> bool:
    """Whether tiles of format ``fmt`` of a tensor of GGUF type ``qtype`` are a half-precision head's Q4_0 plane: rows
    the image quantizes at load (``_native.q4_0_rows``), the packer packing those."""
    return qtype in HALF and fmt == PLANES[qtype]


def screened(qtype: str, formats: set[str]) -> bool:
    """Whether a model's image screens its LM head of GGUF type ``qtype`` (``settings.LM_SCREEN``): a type with a plane, in
    a file of up to ``FILE_FORMATS`` tile formats (``formats``: those of its other matrices). Each paged format's dispatch
    takes 64 B of the main cores' page slot, and Occamy's seven leave its largest page that short with an eighth."""
    return settings.LM_SCREEN.get() and screens(qtype) and len(formats | {PLANES[qtype]}) <= FILE_FORMATS


def half_rows(t: object) -> np.ndarray:
    """A half-precision tensor's rows as (rows, cols) uint16 words (a view of its mapping)."""
    return np.asarray(t.raw).view(np.uint16).reshape(t.rows, t.cols)


def plane_rows(t: object, threads: int = 0) -> np.ndarray:
    """A half-precision head's Q4_0 plane as GGUF rows (rows, cols / 32 * 18) uint8 (``_native.q4_0_rows``)."""
    out = np.empty((t.rows, t.cols // 32 * 18), np.uint8)
    _native.q4_0_rows(half_rows(t), t.qtype == "BF16", t.cols, out, threads)
    return out


class Screen:
    """The host side of a screened LM head: tensor ``name`` of the file (its rows the token ids'), normed by ``norm``
    (hidden,) fp32 with ``eps``; ``n``: the vocabulary's rows."""

    def __init__(self, gguf: GGUFFile, name: str, norm: np.ndarray, eps: float, n: int):
        t = gguf.tensor(name)
        assert screens(t.qtype) and n <= t.rows, (name, t.qtype, n)
        self.half = t.qtype in HALF  # (a half-precision head: rows of words, bf16 or fp16)
        self.bf16 = t.qtype == "BF16"
        self.raw = half_rows(t) if self.half else t.raw  # (the file's rows: its mapping)
        self.eps, self.n = eps, n
        self._populate: Callable[[], None] | None = lambda: gguf.populate(name)  # (before the first rows: ``exact``)
        self.norm = np.ascontiguousarray(norm, np.float32)
        # (per row: 1 / (1 - rho_i), and Z sigma_i per unit of |g|)
        self.scale, self.bound = np.empty(n, np.float32), np.empty(n, np.float32)
        if self.half:
            _native.screen_stats_plane(self.raw, self.bf16, plane_rows(t)[:n], t.cols, self.scale, self.bound)
        else:
            _native.screen_stats(t.raw, t.cols, self.scale, self.bound)
        self.bound *= np.float32(Z)
        self.rows = 0  # (rows computed exactly so far)
        self.taken = 0  # (choices so far: ``candidates``)

    def input(self, x: np.ndarray) -> np.ndarray:
        """The head's input g (hidden,) fp32 of a slot's residual x (the bf16 values the head's phase read): x w / rms(x)."""
        x = np.asarray(x, np.float64)
        return (x * (1.0 / np.sqrt(np.dot(x, x) / x.size + self.eps)) * self.norm).astype(np.float32)

    def exact(self, ids: np.ndarray, g: np.ndarray) -> np.ndarray:
        """The exact logits (len(ids),) fp32 of the tokens ``ids`` against the head's input g: the file's rows, read in
        place (``_native.screen_exact``: ~0.1 us per 1024 columns of a row)."""
        if self._populate is not None:  # (once, after the loader gave the file's pages back: ``GGUFFile.release``)
            self._populate()
            self._populate = None
        self.rows += len(ids)
        if self.half:
            return _native.screen_exact_half(self.raw, self.bf16, np.ascontiguousarray(ids, np.int32), g)
        return _native.screen_exact(self.raw, np.ascontiguousarray(ids, np.int32), g)

    def _above(self, bits: np.ndarray, gnorm: float, thr: float, mask: np.ndarray | None = None) -> np.ndarray:
        """The rows whose upper bound reaches thr (``mask``: of the allowed ones), ascending (int32; a copy)."""
        ids = np.empty(IDS, np.int32)  # (the call's own: several sequences' choices run on threads of their own)
        n = _native.screen_above(bits, self.scale, self.bound, gnorm, thr, ids, mask)
        if n <= len(ids):
            return ids[:n].copy()
        up = self._upper(bits, gnorm) >= thr
        if mask is not None:
            up &= np.unpackbits(mask.view(np.uint8), count=self.n, bitorder="little").astype(bool)
        return np.flatnonzero(up).astype(np.int32)

    def _upper(self, bits: np.ndarray, gnorm: float, ids: np.ndarray | slice = slice(None)) -> np.ndarray:
        """The upper bounds of the rows ``ids`` (fp32)."""
        return bits[ids].view(bfloat16).astype(np.float32) * self.scale[ids] + self.bound[ids] * np.float32(gnorm)

    def candidates(self, screened: np.ndarray, g: np.ndarray, k: int, mask: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        """The rows computed exactly for the k largest logits of a slot (the module docstring's steps), from its
        screened logits (n,) bf16 and the head's input g: their tokens (int32, in no order) and exact logits (fp32).
        ``mask``: of the tokens a vocabulary bitmask allows (``models.sampling.Mask.bits``): every step among them, the
        k largest of those as exact as an unmasked choice's."""
        bits, gnorm = screened.view(np.uint16), float(np.linalg.norm(g))
        ids = _native.top_logits(bits, min(max(SEEDS, 2 * k), self.n), mask=mask)[1]
        e = self.exact(ids, g)
        if not len(e):
            return ids, e
        thr = max(float(e.max()), _native.screen_lower(bits, self.scale, self.bound, gnorm, mask))
        ids, e = self._more(bits, gnorm, thr, ids, e, g, mask)
        if k > 1:
            thr = float(e.max()) - WINDOW
            if len(e) >= k:
                thr = max(thr, float(np.partition(e, -k)[-k]))
            ids, e = self._more(bits, gnorm, thr, ids, e, g, mask)
        self.taken += 1
        return ids, e

    def _more(self, bits: np.ndarray, gnorm: float, thr: float, ids: np.ndarray, e: np.ndarray, g: np.ndarray,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(ids, e) and the rows whose upper bound reaches thr (``ROWS`` at most: the highest upper bounds; ``mask``: of
        the allowed ones), exact."""
        more = self._above(bits, gnorm, thr, mask)
        more = more[~np.isin(more, ids)]
        if len(more) > ROWS:
            more = np.sort(more[np.argpartition(self._upper(bits, gnorm, more), -ROWS)[-ROWS:]])
        if not len(more):
            return ids, e
        return np.concatenate([ids, more]), np.concatenate([e, self.exact(more, g)])

    def top(self, screened: np.ndarray, g: np.ndarray, k: int, mask: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
        """The k largest logits of a slot, fp32, descending (equal ones: the lower token first), and their tokens
        (int32): the exact ones (``candidates``), then, where fewer than k lie in the window, the largest screened ones
        of the other tokens (their estimates), below them (``mask``: of the allowed tokens)."""
        ids, e = self.candidates(screened, g, k, mask)
        if not len(ids):
            return e, ids
        order = np.lexsort((ids, -e))[:k]
        v, i = e[order], ids[order]
        if len(i) < min(k, self.n):
            sv, si = _native.top_logits(screened.view(np.uint16), min(k + len(ids), self.n), mask=mask)
            keep = ~np.isin(si, ids)
            sv, si = sv[keep][:k - len(i)], si[keep][:k - len(i)]
            v = np.concatenate([v, np.minimum(sv * self.scale[si], np.nextafter(e.min(), np.float32(-np.inf)))])
            i = np.concatenate([i, si])
        return v, i

    def logits(self, screened: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
        """A slot's logits (n,) fp32: the k largest exact (``candidates``), the others their estimates, below the k-th
        of those."""
        ids, e = self.candidates(screened, g, k)
        floor = np.partition(e, -k)[-k] if len(e) >= k else e.min()
        a = np.empty(self.n, np.float32)
        _native.screen_dense(screened.view(np.uint16), self.scale,
                             float(np.nextafter(np.float32(floor), np.float32(-np.inf))), a)
        a[ids] = e
        return a


class ScreenedRow:
    """A slot's logits of a screened head: ``top(k)`` its k largest (exact), ``at`` the exact logits of given tokens,
    ``dense(k)`` (vocab,) fp32 with the k (``K``) largest exact, ``exact()`` and as an array every logit exact.
    Computed when taken (``Screened``)."""

    dtype = np.dtype(np.float32)
    ndim = 1

    def __init__(self, screen: Screen, screened: np.ndarray, x: np.ndarray, alive: Callable[[], bool]):
        self.screen, self.screened, self.x, self.alive = screen, screened, x, alive
        self.shape = screened.shape
        self._g: np.ndarray | None = None
        self._top: tuple[int, np.ndarray, np.ndarray] | None = None
        self._dense: tuple[int, np.ndarray] | None = None
        self._exact: np.ndarray | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def _input(self) -> np.ndarray:
        if not self.alive():
            raise RuntimeError("a screened step's logits are taken before the overlay's next dispatch of logits")
        if self._g is None:
            self._g = self.screen.input(self.x)
        return self._g

    def top(self, k: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The k largest logits, fp32, descending (equal ones: the lower token first), and their tokens (int32);
        ``mask``: of the tokens a vocabulary bitmask allows (``models.sampling.Mask.bits``), as exact."""
        if mask is not None:
            return self.screen.top(self.screened, self._input(), k, mask)
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

    def exact(self) -> np.ndarray:
        """(vocab,) fp32: every logit exact (the whole head against g on the host)."""
        if self._exact is None:
            self._exact = self.screen.exact(np.arange(len(self), dtype=np.int32), self._input())
        return self._exact

    def astype(self, dtype: np.dtype) -> np.ndarray:
        return self.exact().astype(dtype)

    def __getitem__(self, i: int | np.ndarray) -> np.ndarray | np.float32:
        """A token's exact logit (a chosen token's, for its log-probability), or those of an array of tokens."""
        if self._exact is not None:
            return self._exact[i]
        return self.at(np.atleast_1d(i))[0] if np.ndim(i) == 0 else self.at(np.asarray(i))

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        out = self.exact()
        return out if dtype is None else out.astype(dtype)


class Screened:
    """A step's logits of a screened head: a ``ScreenedRow`` per slot (``[i]``; ``dense()`` and as an array: (slots,
    vocab) fp32 of the rows' ``dense()`` / ``exact()``).
    ``screened``: the NPU's logits (slots, vocab) bf16, a view of the overlay's buffer: a row is taken before the next
    dispatch of logits (``alive``); ``x``: the slots' residuals (slots, hidden) the head read."""

    dtype = np.dtype(np.float32)
    ndim = 2

    def __init__(self, screen: Screen, screened: np.ndarray, x: np.ndarray, alive: Callable[[], bool]):
        assert screened.dtype == bfloat16 and len(x) == len(screened)
        self.shape = screened.shape
        self._of = (screen, screened, x)
        self._rows = [ScreenedRow(screen, s, xi, alive) for s, xi in zip(screened, x, strict=True)]

    def take(self, a: int, b: int) -> "Screened":
        """Slots a .. b - 1 over copies of the NPU's rows: valid past the overlay's next dispatch of logits (several
        sequences' step: each one's rows, read while the next step runs)."""
        screen, screened, x = self._of
        return Screened(screen, screened[a:b].copy(), x[a:b], lambda: True)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int | slice) -> ScreenedRow | np.ndarray:
        if isinstance(i, slice):
            return np.stack([r.exact() for r in self._rows[i]])
        return self._rows[i]

    def dense(self) -> np.ndarray:
        return np.stack([r.dense() for r in self._rows])

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        out = self[:]
        return out if dtype is None else out.astype(dtype)
