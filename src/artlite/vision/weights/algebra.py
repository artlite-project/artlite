"""The vision networks' weights as index maps on sources: ``Wt``.

A ``Wt`` is a weight matrix of parts of sources (``Source``: a tensor of a GGUF or safetensors file, read in place from
the file's memory map; a small matrix the host computes at every load, ``computed``; the identity, ``eye``): per part
the source row of each of its rows, the source column of each of its columns (-1: zero) and a scale. The networks'
weight transforms (heads and channels padded, the RoPE heads' rows, the queries' scale, a GEGLU's halves, a conv's
input slices, a sum of column blocks) are index maps on them (``take``, ``cat``, ``Wt.__getitem__``, ``*``), so a
weight is never materialized on the host. Torch tensors take the same helpers (the tests' weights): ``Builder.gemm``
takes either, a tensor as a source of memory of its own.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

KINDS = ("F32", "F16", "BF16", "Q8_0", "EYE")


@dataclass(eq=False)
class Source:
    """A matrix of rows x cols elements of ``kind`` ("F32", "F16", "BF16", "Q8_0" (blocks of 32: fp16 d, 32 int8), or
    "EYE": the identity, no data): ``raw`` (rows, row bytes) uint8; ``file``: the file's path and its whole mapping
    (None: memory of its own). ``name``: unique among a model's sources (a cached build's jobs refer to it)."""

    name: str
    kind: str
    rows: int
    cols: int
    raw: np.ndarray | None = None
    file: tuple[str, np.ndarray] | None = None

    def values(self) -> np.ndarray:
        """The elements as float32 (rows, cols)."""
        if self.kind == "EYE":
            return np.eye(self.rows, self.cols, dtype=np.float32)
        r = self.raw
        if self.kind == "F32":
            return r.view(np.float32).copy()
        if self.kind == "F16":
            return r.view(np.float16).astype(np.float32)
        if self.kind == "BF16":
            return (r.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        b = r.reshape(self.rows, -1, 34)  # (Q8_0: d q, one float32 product)
        d = np.ascontiguousarray(b[..., :2]).view(np.float16).astype(np.float32)
        return (d * np.ascontiguousarray(b[..., 2:]).view(np.int8).astype(np.float32)).reshape(self.rows, self.cols)


class Part(NamedTuple):
    source: Source
    rows: np.ndarray  # (N,) int64 source rows, -1: zero
    cols: np.ndarray  # (C inner,) int64 source columns, -1: zero
    scale: float


def _index(key: object, n: int) -> np.ndarray:
    """A row or channel selection (a slice, or indices: a sequence or array) as int64 indices into n."""
    if isinstance(key, slice):
        return np.arange(n, dtype=np.int64)[key]
    idx = np.asarray(key, np.int64).reshape(-1)
    assert ((idx >= 0) & (idx < n)).all(), "indices in range (no negative ones)"
    return idx


class Wt:
    """A weight matrix of ``shape`` (N, C, *spatial: its columns C x prod(spatial), channel-major), the sum of its parts
    (each zero where its row or column map is -1; parts do not overlap)."""

    __slots__ = ("parts", "shape")

    def __init__(self, parts: tuple[Part, ...], shape: tuple[int, ...]):
        self.parts = tuple(p for p in parts if (p.rows >= 0).any() and (p.cols >= 0).any())
        self.shape = tuple(int(s) for s in shape)

    @classmethod
    def of(cls, src: Source, shape: tuple[int, ...] | None = None) -> Wt:
        """The whole of a source (shape: its rows and columns, or a factoring of the columns)."""
        shape = shape or (src.rows, src.cols)
        assert shape[0] == src.rows and math.prod(shape[1:]) == src.cols, (src.name, shape)
        return cls((Part(src, np.arange(src.rows), np.arange(src.cols), 1.0),), shape)

    @property
    def inner(self) -> int:
        return math.prod(self.shape[2:])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def float(self) -> Wt:
        return self

    def take_rows(self, idx: np.ndarray) -> Wt:
        """Rows idx (-1: zero rows)."""
        idx = np.asarray(idx, np.int64)
        assert idx.max(initial=-1) < self.shape[0], (idx.max(), self.shape)
        return Wt(tuple(p._replace(rows=np.where(idx >= 0, p.rows[idx], -1)) for p in self.parts),
                  (len(idx),) + self.shape[1:])

    def take_cols(self, idx: np.ndarray) -> Wt:
        """Channels (dim 1) idx (-1: zero channels), each its inner columns."""
        idx = np.asarray(idx, np.int64)
        assert idx.max(initial=-1) < self.shape[1], (idx.max(), self.shape)
        k = self.inner
        flat = np.where(idx[:, None] >= 0, idx[:, None] * k + np.arange(k), -1).reshape(-1)
        return Wt(tuple(p._replace(cols=np.where(flat >= 0, p.cols[flat], -1)) for p in self.parts),
                  (self.shape[0], len(idx)) + self.shape[2:])

    def __getitem__(self, key: object) -> Wt:
        """Rows (``w[rows]``), channels (``w[:, ch]``), spatial positions (``w[:, :, y, x]``: ints)."""
        key = key if isinstance(key, tuple) else (key,)
        w = self
        if not (isinstance(key[0], slice) and key[0] == slice(None)):
            w = w.take_rows(_index(key[0], w.shape[0]))
        if len(key) > 1 and not (isinstance(key[1], slice) and key[1] == slice(None)):
            w = w.take_cols(_index(key[1], w.shape[1]))
        if len(key) > 2:  # (fixed spatial positions: the columns of them)
            sp = w.shape[2:]
            assert len(key) == 2 + len(sp) and all(isinstance(i, int) for i in key[2:]), key
            at = int(np.ravel_multi_index(tuple(key[2:]), sp))
            k = w.inner
            sel = np.arange(w.shape[1]) * k + at
            w = Wt(tuple(p._replace(cols=p.cols[sel]) for p in w.parts), w.shape[:2])
        return w

    def flatten(self, start_dim: int = 1) -> Wt:
        assert start_dim == 1
        return Wt(self.parts, (self.shape[0], math.prod(self.shape[1:])))

    def __mul__(self, s: float) -> Wt:
        return Wt(tuple(p._replace(scale=float(np.float32(p.scale) * np.float32(s))) for p in self.parts), self.shape)

    __rmul__ = __mul__

    @staticmethod
    def cat(ws: list[Wt], dim: int = 0) -> Wt:
        """Rows (dim 0) or columns (dim 1, of 2-D matrices) of the matrices in order."""
        if dim == 0:
            n, parts = sum(w.shape[0] for w in ws), []
            at = 0
            for w in ws:
                assert w.shape[1:] == ws[0].shape[1:], [x.shape for x in ws]
                for p in w.parts:
                    rows = np.full(n, -1, np.int64)
                    rows[at:at + w.shape[0]] = p.rows
                    parts.append(p._replace(rows=rows))
                at += w.shape[0]
            return Wt(_merge(parts), (n,) + ws[0].shape[1:])
        assert dim == 1 and all(w.ndim == 2 and w.shape[0] == ws[0].shape[0] for w in ws), [x.shape for x in ws]
        k, parts, at = sum(w.shape[1] for w in ws), [], 0
        for w in ws:
            for p in w.parts:
                cols = np.full(k, -1, np.int64)
                cols[at:at + w.shape[1]] = p.cols
                parts.append(p._replace(cols=cols))
            at += w.shape[1]
        return Wt(_merge(parts), (ws[0].shape[0], k))

    def numpy(self) -> np.ndarray:
        """The values (float32, ``shape``): the reference of what ``Slots`` packs."""
        if len(self.parts) == 1 and (p := self.parts[0]).scale == 1.0 and \
                (len(p.rows), len(p.cols)) == (p.source.rows, p.source.cols) and \
                (p.rows == np.arange(len(p.rows))).all() and (p.cols == np.arange(len(p.cols))).all():
            return p.source.values().reshape(self.shape)  # (a whole source)
        out = np.zeros((self.shape[0], self.shape[1] * self.inner), np.float32)
        for p in self.parts:
            v = p.source.values()
            r, c = p.rows >= 0, p.cols >= 0
            block = v[np.ix_(p.rows[r], p.cols[c])]
            out[np.ix_(r, c)] += block if p.scale == 1.0 else block * np.float32(p.scale)
        return out.reshape(self.shape)

    @property
    def key(self) -> tuple:
        """A fingerprint of the matrix (equal for the same parts of the same sources)."""
        return (self.shape,) + tuple((p.source.name, p.scale, hashlib.blake2b(p.rows.tobytes(), digest_size=8).digest(),
                                      hashlib.blake2b(p.cols.tobytes(), digest_size=8).digest()) for p in self.parts)

    def sources(self) -> list[Source]:
        return [p.source for p in self.parts]


def _merge(parts: list[Part]) -> tuple[Part, ...]:
    """Parts of the same source and scale as one where they have the same columns and rows that do not overlap, or the
    same rows and columns that do not overlap."""
    out: list[Part] = []
    for p in parts:
        for i, q in enumerate(out):
            if q.source is not p.source or q.scale != p.scale:
                continue
            if np.array_equal(q.cols, p.cols) and not ((q.rows >= 0) & (p.rows >= 0)).any():
                out[i] = q._replace(rows=np.where(p.rows >= 0, p.rows, q.rows))
                break
            if np.array_equal(q.rows, p.rows) and not ((q.cols >= 0) & (p.cols >= 0)).any():
                out[i] = q._replace(cols=np.where(p.cols >= 0, p.cols, q.cols))
                break
        else:
            out.append(p)
    return tuple(out)


def memory(x: np.ndarray, name: str | None = None) -> Wt:
    """A matrix of memory of its own (float32 values; name: a computed weight's, unique among the model's sources;
    None: by the values' digest, for tests: not resolvable in a cached build)."""
    a = np.ascontiguousarray(x, np.float32)
    shape = a.shape if a.ndim > 1 else (1, a.size)
    rows, cols = shape[0], math.prod(shape[1:])
    name = name or "mem:" + hashlib.blake2b(a.tobytes(), digest_size=12).hexdigest()
    return Wt.of(Source(name, "F32", rows, cols, a.reshape(rows, cols).view(np.uint8)), shape)


def computed(name: str, x: np.ndarray) -> Wt:
    """A weight the host computes from the model's tensors at every load (``name``: unique among its sources)."""
    return memory(x, name)


def eye(n: int) -> Wt:
    """The n x n identity."""
    return Wt.of(Source(f"eye:{n}", "EYE", n, n))


def as_wt(x: Wt | np.ndarray) -> Wt:
    return x if isinstance(x, Wt) else memory(x)


def values(x: Wt | np.ndarray) -> np.ndarray:
    """A matrix's values as float32 numpy (for the host's computed weights)."""
    if isinstance(x, Wt):
        return x.numpy()
    return np.asarray(x, np.float32)


def flat(x: Wt | np.ndarray) -> Wt | np.ndarray:
    """A matrix as (rows, the rest): a conv's (N, C, kh, kw) as (N, C kh kw)."""
    return x.flatten(1) if isinstance(x, Wt) else np.asarray(x).reshape(x.shape[0], -1)


def f32(x: Wt | np.ndarray) -> Wt | np.ndarray:
    """A matrix as it is (a Wt), an array's values as float32."""
    return x if isinstance(x, Wt) else np.asarray(x, np.float32)


def take(x: Wt | np.ndarray, idx: np.ndarray, dim: int = 0) -> Wt | np.ndarray:
    """Entries idx of x along dim 0 (rows) or 1 (channels), -1: zeros (a Wt's maps; an array's copy)."""
    idx = np.asarray(idx, np.int64)
    if isinstance(x, Wt):
        return x.take_rows(idx) if dim == 0 else x.take_cols(idx)
    x = np.asarray(x)
    shape = list(x.shape)
    shape[dim] = len(idx)
    out = np.zeros(shape, x.dtype)
    at = [slice(None)] * x.ndim
    at[dim] = np.flatnonzero(idx >= 0)
    out[tuple(at)] = np.take(x, idx[idx >= 0], axis=dim)
    return out


def cat(xs: list[Wt | np.ndarray], dim: int = 0) -> Wt | np.ndarray:
    """The concatenation of arrays; matrices of which any is a Wt as one Wt (arrays as memory of their own)."""
    if not any(isinstance(x, Wt) for x in xs):
        return np.concatenate([np.asarray(x) for x in xs], dim)
    return Wt.cat([as_wt(x) for x in xs], dim)


def pad_index(n: int, size: int) -> np.ndarray:
    """0 .. n - 1 with those past size -1 (a zero-padded dimension of n entries over size)."""
    i = np.arange(n, dtype=np.int64)
    return np.where(i < size, i, -1)
