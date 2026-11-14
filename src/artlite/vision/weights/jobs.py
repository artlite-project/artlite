"""The weights' jobs: per core row of a GEMM its passes' rows and the runs of its k-tiles' columns (``Slots.add``: a conv's
taps in its B slots' order, hilo's planes), which ``_native.pack_slots`` packs (the end of a build, and every restore of
a cached build: the cache keeps the jobs, never the weights): threads over a pass's 16 k-tiles, the files read into the
page cache ahead of them, AVX-512 bfp16 blocks, non-temporal stores into the device's buffers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ... import _native
from .. import design as D
from .algebra import Source, Wt, pad_index

if TYPE_CHECKING:
    from ..device import VisionDevice
    from ..ops import Gemm

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
            pack_arrays(job_arrays(jobs, npu.span), npu.dev.view, threads=threads)
            self.done += jobs

    def table(self, npu: VisionDevice) -> dict:
        """The packed jobs as arrays, by their sources' names and their buffers' (a cached build's: ``restore``)."""
        t = job_arrays(self.done, npu.span)
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
        pack_arrays(table | {"sources": sources}, npu.dev.view, threads=threads)


def job_arrays(jobs: list[tuple], place: Callable[[str], tuple[str, int, int]]) -> dict:
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


def pack_arrays(t: dict, view: Callable[[str], np.ndarray], *, threads: int = 0) -> None:
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
