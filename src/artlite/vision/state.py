"""A model's built state on the vision overlay's design, cached in a directory per model file: the device's layout, what its
builds wrote besides the weights (commands, norm parameters, tables), the weight jobs and the models' own state (its
transactions with their shim addresses as relocations). Restored, a load skips the builds and packs the weights from the
model's files (``weights.Slots``), as a build does; the cache never holds a copy of the weights."""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from collections.abc import Callable, Mapping
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from ..operators.toolchain import CACHE_DIR
from .device import VisionDevice
from .sources import SRC, source_files
from .weights import Slots, Source

if TYPE_CHECKING:
    from .builder import Built

FORMAT = 1  # (the entry's layout: in its key and in its state's dict)


def state_dir(kind: str, path: str, modules: tuple[str, ...], files: list[str]) -> str:
    """The cache directory of a model's built state: ``CACHE_DIR/kind/<path's digest>/<key>``, the key covering the
    toolchain, the code that builds the images and transactions (the files ``sources.source_files(*modules)`` derives
    from the modules' imports: the loader's modules, named by the caller) and the model's ``files`` (size and mtime)."""
    h = hashlib.sha256(f"{FORMAT} {version('llvm-aie')}".encode())
    for f in source_files(*modules):
        h.update(os.path.relpath(f, SRC).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)} {st.st_size} {st.st_mtime_ns}".encode())
    return os.path.join(CACHE_DIR, kind, hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:12], h.hexdigest()[:16])


def load_state(cache: str, sources: Mapping[str, Source], *,
               accept: Callable[[dict], bool] | None = None) -> tuple[VisionDevice, dict] | None:
    """A device restored from ``save_state``'s directory (the host's writes again, the weights packed from
    ``sources``: the model's, ``weights.sources`` or ``weights.Sources``, on a thread until the first dispatch:
    ``VisionDevice.pack_behind``) and the saved state; None: no entry, one whose state ``accept`` refuses, or one whose
    weights' sources the model lacks."""
    try:
        with open(os.path.join(cache, "state.pkl"), "rb") as f:
            st = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError):
        return None  # (no entry, or one of another code's classes)
    if not isinstance(st, dict) or st.get("format") != FORMAT or not {"layout", "writes", "slots", "state"} <= st.keys() \
            or (accept is not None and not accept(st["state"])):
        return None
    res = [Slots.resolve(t, sources) for t in st["slots"]]
    if any(r is None for r in res):
        return None
    npu = VisionDevice()
    npu.restore(st["layout"])
    for name, off, data in st["writes"]:
        npu.write(name, np.frombuffer(data, np.uint8), off)
    npu.pack_behind(lambda: [Slots.restore(npu, t, r) for t, r in zip(st["slots"], res, strict=True)])
    return npu, st["state"]


def save_state(cache: str, npu: VisionDevice, built: list[Built], state: dict) -> None:
    """The built state of a device (what ``built`` wrote: the host's byte ranges, without their trailing zeros, and the
    weight jobs) and the models' ``state`` into the cache directory (written to a sibling, then renamed); the model
    file's older entries removed."""
    root, tmp = os.path.dirname(cache), f"{cache}.{os.getpid()}"
    npu.packed()
    writes = []
    for b in built:
        for name, ranges in b.writes.items():
            v = npu.view(name)
            for off, n in sorted(ranges.items()):
                nz = np.flatnonzero(v[off:off + n])
                if len(nz):
                    writes.append((name, off, v[off:off + nz[-1] + 1].tobytes()))
    st = {"format": FORMAT, "layout": npu.layout(), "writes": writes, "slots": [b.slots.table(npu) for b in built], "state": state}
    os.makedirs(tmp)
    with open(os.path.join(tmp, "state.pkl"), "wb") as f:
        pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
    for e in os.listdir(root):
        if e != os.path.basename(tmp):
            shutil.rmtree(os.path.join(root, e), ignore_errors=True)
    os.replace(tmp, cache)
