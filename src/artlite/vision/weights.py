"""The vision networks' weights (``Wt``) and their packing into the overlay's B slots (``Slots``).

A ``Wt`` is a weight matrix of parts of sources (``Source``: a tensor of a GGUF or safetensors file, read in place from
the file's memory map; a small matrix the host computes at every load, ``computed``; the identity, ``eye``): per part
the source row of each of its rows, the source column of each of its columns (-1: zero) and a scale. The networks'
weight transforms (heads and channels padded, the RoPE heads' rows, the queries' scale, a GEGLU's halves, a conv's
input slices, a sum of column blocks) are index maps on them (``take``, ``cat``, ``Wt.__getitem__``, ``*``), so a
weight is never materialized on the host. Torch tensors take the same helpers (the tests' weights): ``Builder.gemm``
takes either, a tensor as a source of memory of its own.

``Builder.gemm`` turns a GEMM's weights into jobs (``Slots.add``: per core row its passes' rows, the runs of its
k-tiles' columns: a conv's taps in its B slots' order, hilo's planes), which ``_native.pack_slots`` packs (the end of a
build, and every restore of a cached build: the cache keeps the jobs, never the weights): threads over a pass's 16
k-tiles, the files read into the page cache ahead of them, AVX-512 bfp16 blocks, non-temporal stores into the device's
buffers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .. import _native
from . import design as D

if TYPE_CHECKING:
    from ..gguf_model import GGUFFile
    from .device import VisionDevice
    from .ops import Gemm

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


class Sources(Mapping):
    """Sources by name, each resolved on first use (``resolve``: the Source of a name, or None): a cached build's
    restore takes only those its jobs read."""

    def __init__(self, resolve: Callable[[str], Source | None]):
        self._resolve, self._got = resolve, {}

    def __getitem__(self, name: str) -> Source:
        if name not in self._got:
            self._got[name] = self._resolve(name)
        if self._got[name] is None:
            raise KeyError(name)
        return self._got[name]

    def __iter__(self) -> Iterator[str]:
        return (n for n, s in self._got.items() if s is not None)

    def __len__(self) -> int:
        return sum(s is not None for s in self._got.values())


def gguf_sources(g: GGUFFile, computed: dict[str, Callable[[], Wt]]) -> Sources:
    """A GGUF file's tensors (``gguf_matrix``: by their sources' names) and a model's computed weights (by name: the
    function that computes the Wt) as ``Sources``."""
    prefix = os.path.basename(g.path) + ":"

    def resolve(name: str) -> Source | None:
        if name in computed:
            return computed[name]().parts[0].source
        t = name.removeprefix(prefix)
        return gguf_matrix(g, t).parts[0].source if name.startswith(prefix) and t in g.tensor_info else None
    return Sources(resolve)


def sources(w: object) -> dict[str, Source]:
    """The sources of every Wt in w (dicts, lists and tuples of them, other values skipped) by name."""
    out: dict[str, Source] = {}
    stack = [w]
    while stack:
        x = stack.pop()
        if isinstance(x, Wt):
            for s in x.sources():
                out.setdefault(s.name, s)
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list | tuple):
            stack.extend(x)
    return out


# -- model files

_GGUF_KINDS = KINDS[:4]


def gguf_matrix(g: GGUFFile, name: str) -> Wt:
    """A GGUF tensor as a matrix (rows, cols) of its file: F32, F16, BF16 or Q8_0 (the overlay's GEMMs read bfp16, so a
    4-bit matrix would grow on every load: refused)."""
    t = g.tensor(name)
    sname = f"{os.path.basename(g.path)}:{name}"
    if t.qtype not in _GGUF_KINDS:
        raise NotImplementedError(f"{g.path}: {name} is {t.qtype}: the NPU's media encoders take mmproj files of Q8_0, "
                                  f"BF16, F16 or F32 weights (use the model's Q8_0 mmproj)")
    raw = t.raw if t.raw.dtype == np.uint8 else t.raw.view(np.uint8)
    return Wt.of(Source(sname, t.qtype, t.rows, t.cols, raw.reshape(t.rows, -1), (g.path, g.data)))


def gguf_getter(g: GGUFFile) -> Callable[[str], Wt | np.ndarray]:
    """A GGUF file's tensors by name: matrices (2 dims and more: ``gguf_matrix``) and vectors as float32 (1, cols)
    arrays."""
    def get(name: str) -> Wt | np.ndarray:
        if len(g.tensor_info[name][0]) > 1:
            return gguf_matrix(g, name)
        t = g.tensor(name)
        return t.dequantize().reshape(1, t.cols)
    return get


_ST_KINDS = {"F32": ("F32", np.float32), "F16": ("F16", np.float16), "BF16": ("BF16", np.uint16)}


def safetensors(path: str, prefixes: tuple[str, ...] = (), *, tag: str = "") -> dict[str, Wt | np.ndarray]:
    """A safetensors file's tensors (``prefixes``: only those whose names start with one of them) read from its memory
    map: matrices (2 dims and more) as Wt (sources named ``tag:<name>``, tag default the file's name), vectors as
    float32 arrays."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    (n,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(bytes(data[8:8 + n]))
    header.pop("__metadata__", None)
    tag, base, out = tag or os.path.basename(path), 8 + n, {}
    for name, e in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        if prefixes and not name.startswith(prefixes):
            continue
        kind, dt = _ST_KINDS[e["dtype"]]
        begin, end = e["data_offsets"]
        shape = tuple(e["shape"])
        raw = data[base + begin:base + end]
        if raw.size != math.prod(shape) * np.dtype(dt).itemsize:
            raise ValueError(f"{path}: {name}: a byte range that does not fit the shape")
        if len(shape) < 2:
            v = raw.view(dt)
            out[name] = (v.astype(np.uint32) << 16).view(np.float32) if kind == "BF16" else v.astype(np.float32)
            continue
        out[name] = Wt.of(Source(f"{tag}:{name}", kind, shape[0], math.prod(shape[1:]), raw.reshape(shape[0], -1),
                                 (path, data)), shape)
    return out


# -- the jobs

def _runs(c: np.ndarray) -> np.ndarray:
    """A job part's output columns' sources (int64, a multiple of 8 of them; -1: zero) as runs (R, 4) int32 [first
    column, columns, source column, stride]: arithmetic 8-column blocks merged where they continue each other, the
    columns of other blocks one by one."""
    b = c.reshape(-1, 8)
    st = b[:, 1] - b[:, 0]
    arith = (b >= 0).all(1) & (np.diff(b, axis=1) == st[:, None]).all(1) & (st >= 0)
    i = np.flatnonzero(arith)
    cont = np.zeros(len(i), bool)
    cont[1:] = (i[1:] == i[:-1] + 1) & (st[i[1:]] == st[i[:-1]]) & (b[i[1:], 0] == b[i[:-1], 0] + 8 * st[i[:-1]])
    first = np.flatnonzero(~cont)
    n = np.diff(np.append(first, len(i))) * 8
    runs = [np.stack([8 * i[first], n, b[i[first], 0], st[i[first]]], 1)]
    mixed = np.flatnonzero(~arith & (b >= 0).any(1))
    if len(mixed):  # (columns of other blocks one by one)
        k = (8 * mixed[:, None] + np.arange(8)).reshape(-1)
        k = k[c[k] >= 0]
        runs.append(np.stack([k, np.ones_like(k), c[k], np.ones_like(k)], 1))
    out = np.concatenate(runs).astype(np.int64)
    assert out[:, 2].max(initial=0) < 2 ** 31
    return out[np.argsort(out[:, 0], kind="stable")].astype(np.int32)


def _columns(g: Gemm, w: Wt) -> np.ndarray:
    """The matrix columns of a Gemm's k-tiles in its B slots' order (hilo: of the hi plane), -1 past the matrix: a
    conv's per chunk of 64 input channels its taps (``ConvSpec.tap_order``), each the chunk's channels."""
    half = (g.nkt // 2 if g.hilo else g.nkt) * D.KT
    if not g.conv:
        k = w.shape[1] * w.inner
        assert k <= half, (w.shape, g.nkt)
        return pad_index(half, k)
    assert w.shape[1:] == (g.K, 3, 3), (w.shape, g.K)
    ch = np.arange(64)
    return np.concatenate([(64 * c + ch) * 9 + 3 * dy + dx for c in range(g.nch) for dy, dx in g.conv.tap_order(c)])


class Slots:
    """Weight jobs of the vision overlay's DDR images (``add``: a Gemm's core rows; ``pack``: the pending ones), and the
    table of every job packed (``table`` / ``restore``: a cached build's)."""

    def __init__(self):
        self.pending: list[tuple] = []  # (image, offset, npo, nkt, hilo, [(source, scale, rows int32, runs int32)])
        self.done: list[tuple] = []

    def add(self, g: Gemm, w: Wt, at: dict[int, tuple[str, int]]) -> None:
        """Jobs of a Gemm's weights w (its rows ``Gemm.weight_rows``) into its core rows' places (row -> image, byte
        offset)."""
        cols = _columns(g, w)
        parts = [(p, _runs(np.where(cols >= 0, p.cols[np.maximum(cols, 0)], -1))) for p in w.parts]
        parts = [(p, r) for p, r in parts if len(r)]
        for row, (image, offset) in at.items():
            idx = g.weight_rows(row)
            assert idx.max() < w.shape[0], (idx.max(), w.shape)
            self.pending.append((image, offset, g.npo, g.nkt, g.hilo,
                                 [(p.source, p.scale, np.where(p.rows[idx] >= 0, p.rows[idx], -1).astype(np.int32), r)
                                  for p, r in parts]))

    def pack(self, npu: VisionDevice, *, threads: int = 0) -> None:
        """Pack the pending jobs into the device's images."""
        if self.pending:
            jobs, self.pending = self.pending, []
            _pack(_flat(jobs, npu._images.__getitem__), npu.dev.view, threads=threads)
            self.done += jobs

    def pack_to(self, out: dict[str, np.ndarray]) -> None:
        """Pack the pending jobs into host arrays (uint8) by image name."""
        jobs, self.pending = self.pending, []
        _pack(_flat(jobs, lambda im: (im, 0, out[im].size)), out.__getitem__)

    def table(self, npu: VisionDevice) -> dict:
        """The packed jobs as arrays, by their sources' names and their buffers' (a cached build's: ``restore``)."""
        t = _flat(self.done, npu._images.__getitem__)
        anon = [s.name for s in t["sources"] if s.name.startswith("mem:")]
        assert not anon, f"weights of memory without a name (not resolvable in a cached build): {anon[:3]}"
        t["sources"] = [(s.name, s.kind, s.rows, s.cols) for s in t["sources"]]
        return t

    @staticmethod
    def resolve(table: dict, sources: dict[str, Source]) -> list[Source] | None:
        """A ``table``'s sources among the model's (eye: by its name); None: one missing or of another kind or shape."""
        res = []
        for name, kind, rows, cols in table["sources"]:
            s = sources.get(name) or (Source(name, "EYE", rows, cols) if kind == "EYE" else None)
            if s is None or (s.kind, s.rows, s.cols) != (kind, rows, cols):
                return None
            res.append(s)
        return res

    @staticmethod
    def restore(npu: VisionDevice, table: dict, sources: list[Source], *, threads: int = 0) -> None:
        """Pack a ``table``'s jobs from its sources (``resolve``) into a device of the table's layout."""
        _pack(table | {"sources": sources}, npu.dev.view, threads=threads)


def _flat(jobs: list[tuple], place: Callable[[str], tuple[str, int, int]]) -> dict:
    """Jobs as ``_native.pack_slots``' arrays (place: an image's buffer, offset in it and bytes)."""
    src: dict[str, int] = {}
    srcs: list[Source] = []
    bufs: dict[str, int] = {}
    runs_at: dict[int, int] = {}
    jt, pt, scales, rows, runs = [], [], [], [], []
    nrow = nrun = 0
    for image, offset, npo, nkt, hilo, parts in jobs:
        chunk, off, n = place(image)
        assert offset + npo * nkt * D.Q <= n, (image, offset, npo, nkt, n)
        jt.append((bufs.setdefault(chunk, len(bufs)), off + offset, npo, nkt, int(hilo), len(pt), len(parts)))
        for s, scale, r, u in parts:
            if s.name not in src:
                src[s.name] = len(srcs)
                srcs.append(s)
            if id(u) not in runs_at:
                runs_at[id(u)] = nrun
                runs.append(u)
                nrun += len(u)
            pt.append((src[s.name], nrow, runs_at[id(u)], len(u)))
            scales.append(scale)
            rows.append(r)
            nrow += len(r)
    return {"sources": srcs, "buffers": list(bufs), "jobs": np.array(jt, np.int64).reshape(-1, 7),
            "parts": np.array(pt, np.int64).reshape(-1, 4), "scales": np.array(scales, np.float32),
            "rows": np.concatenate(rows) if rows else np.zeros(0, np.int32),
            "runs": np.concatenate(runs) if runs else np.zeros((0, 4), np.int32)}


def _pack(t: dict, view: Callable[[str], np.ndarray], *, threads: int = 0) -> None:
    files: dict[str, int] = {}
    fl, ts = [], []
    for s in t["sources"]:
        f = -1
        if s.file is not None:
            if s.file[0] not in files:
                files[s.file[0]] = len(fl)
                fl.append(s.file)
            f = files[s.file[0]]
        ts.append((s.kind, s.raw, s.rows, s.cols, f))
    _native.pack_slots([view(b) for b in t["buffers"]], ts, t["jobs"], t["parts"], t["scales"],
                       np.ascontiguousarray(t["rows"], np.int32), np.ascontiguousarray(t["runs"], np.int32), fl,
                       threads=threads, release=True)
