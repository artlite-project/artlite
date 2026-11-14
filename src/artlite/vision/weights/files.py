"""The weights' sources: the model files' tensors (GGUF, safetensors: read in place from their memory maps) as ``Wt``s
and the by-name tables of a model's sources (``Sources``, ``sources``) that a cached build's restore resolves."""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from .algebra import KINDS, Source, Wt

if TYPE_CHECKING:
    from ...gguf_model import GGUFFile

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
