"""The vision overlay on the NPU (``VisionDevice``): its hardware context, DDR buffers, the roles' pages (``paging``: one
image of two B slots per role, the same for every core), and dispatches of ``seq.Seq`` programs."""

from __future__ import annotations

import glob
import hashlib
import os
import pickle
import shutil
import threading
from collections.abc import Callable, Mapping
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from ..operators import paging
from ..operators.base import CACHE_DIR, PACKAGE_DIR
from ..operators.design import COLS, ROWS
from ..overlay import OverlayDevice, Transaction
from . import core as K
from . import design as D
from .seq import Seq
from .weights import Slots, Source

if TYPE_CHECKING:
    from .build import Built


def role_pages(work_dir: str) -> paging.Pager:
    """The roles' pages (``core.ROLES``, entries ``sd15_ext_<role>``, called from the main loop ``sd15_run``; "ln" also
    ``sd15_geglu`` and ``sd15_gelu``, "d" ``sd15_swiglu`` and ``sd15_prod``, the GEMM's; "r"'s entry the GEMM's RoPE
    epilogue) of the compiled design in ``work_dir``, the same on every core."""
    p = K.P
    more = {"ln": (f"{p}_geglu", f"{p}_gelu"), "d": (f"{p}_swiglu", f"{p}_prod")}
    pages = {r: (f"{p}_ext_{r}",) + more.get(r, ()) for r in K.ROLES}
    return paging.link(work_dir, pages=pages, pager=f"{p}_pg_self",
                       calls={f"{p}_run": tuple(f"{p}_ext_{r}" for r in K.ROLES),
                              f"{p}_gemm": (f"{p}_geglu", f"{p}_gelu", f"{p}_ext_r", f"{p}_swiglu", f"{p}_prod")},
                       roots=(f"{p}_main",), stack=D.STACK, cores=[(c, r + 2) for c in range(COLS) for r in range(ROWS)],
                       state=D.L1["ext"], wake=D.L_WAKE)


class VisionDevice:
    """The compiled design and its DDR images by name (``alloc`` / ``write`` / ``read`` / ``address``), sub-allocated
    from a few large host buffers (``CHUNK`` bytes each: a dispatch lists every buffer, and the driver's command holds
    some 500)."""

    CHUNK = 64 << 20

    def __init__(self, design: D.VisionDesign | None = None):
        self.design = design or D.VisionDesign()
        self.dev = OverlayDevice(self.design)
        self._packing: tuple[threading.Thread, list] | None = None  # (``pack_behind``)
        self._images: dict[str, tuple[str, int, int]] = {}  # (name -> chunk, offset, bytes)
        self._top = self.CHUNK  # (the bytes used of the newest chunk)
        self.pager = role_pages(os.path.dirname(self.dev._prog.insts_path))
        nop = np.zeros(D.Q // 4, np.int32)
        nop[K.C_KIND] = K.K_NOP
        self.alloc("_nop", D.Q)
        self.write("_nop", nop.view(np.uint8))
        self.pages = {}
        for role in K.ROLES:
            pg = self.pager.slots(role, words=D.Q // 4, n=2)
            self.alloc(f"_page_{role}", pg.nbytes)
            self.write(f"_page_{role}", pg)
            self.pages[role] = self.address(f"_page_{role}")

    def has(self, name: str) -> bool:
        return name in self._images

    def alloc(self, name: str, nbytes: int) -> np.ndarray:
        """A DDR image of at least nbytes (an existing one of that name if it is large enough; else a new one), zeroed
        when new; returns its host view."""
        nbytes = -(-nbytes // 4096) * 4096
        if name in self._images and self._images[name][2] >= nbytes:
            return self._view(name)
        if nbytes > self.CHUNK:  # (a buffer of its own)
            chunk = f"_big{len(self.dev.buffers)}"
            self.dev.alloc(chunk, nbytes)
            self._images[name] = (chunk, 0, nbytes)
            return self._view(name)
        if self._top + nbytes > self.CHUNK:
            self.dev.alloc(f"_chunk{len(self.dev.buffers)}", self.CHUNK)
            self._chunk, self._top = f"_chunk{len(self.dev.buffers) - 1}", 0
        self._images[name] = (self._chunk, self._top, nbytes)
        self._top += nbytes
        return self._view(name)

    def _view(self, name: str) -> np.ndarray:
        chunk, off, n = self._images[name]
        return self.dev.view(chunk)[off:off + n]

    def view(self, name: str) -> np.ndarray:
        """An image's host bytes (written there: then ``flush``)."""
        return self._view(name)

    def flush(self, name: str) -> None:
        chunk, off, n = self._images[name]
        self.dev.flush(chunk, off, n)

    def write(self, name: str, data: np.ndarray, offset: int = 0) -> None:
        """Host bytes -> an image (flushed)."""
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        chunk, off, n = self._images[name]
        assert offset + data.size <= n, (name, offset, data.size, n)
        self.dev.view(chunk)[off + offset:off + offset + data.size] = data
        self.dev.flush(chunk, off + offset, data.size)

    def read(self, name: str, nbytes: int | None = None, offset: int = 0) -> np.ndarray:
        self.packed()
        chunk, off, n = self._images[name]
        nbytes = n - offset if nbytes is None else nbytes
        self.dev.fetch(chunk, off + offset, nbytes)
        return self.dev.view(chunk)[off + offset:off + offset + nbytes].copy()

    def address(self, name: str, offset: int = 0) -> int:
        chunk, off, _ = self._images[name]
        return self.dev.address(chunk, off + offset)

    def seq(self) -> Seq:
        return Seq(nop=self.address("_nop"), pages=self.pages)

    def transaction(self, seq: Seq) -> Transaction:
        return Transaction(self.dev, seq.finish())

    def run(self, tr: Transaction) -> None:
        self.packed()
        self.dev.dispatch(tr)

    def pack_behind(self, pack: Callable[[], None]) -> None:
        """Run ``pack`` (weight images: ``weights.Slots``, the native packer without the GIL) on a thread while the
        caller goes on; the next dispatch waits for it (``packed``)."""
        self.packed()
        err: list = []

        def run() -> None:
            try:
                pack()
            except BaseException as e:  # noqa: BLE001 (raised by ``packed``)
                err.append(e)
        t = threading.Thread(target=run, name="vision-pack")
        t.start()
        self._packing = (t, err)

    def packed(self) -> None:
        """Until ``pack_behind``'s packing is done (its error raised here)."""
        if self._packing is not None:
            (t, err), self._packing = self._packing, None
            t.join()
            if err:
                raise err[0]

    def layout(self) -> dict:
        """The DDR layout: the host buffers in creation order (name, bytes), per image its buffer, offset and bytes
        (``restore`` rebuilds it on another device)."""
        return {"buffers": [(n, self.dev.view(n).nbytes) for n in self.dev.buffers], "images": dict(self._images),
                "chunk": getattr(self, "_chunk", None), "top": self._top}

    def restore(self, layout: dict) -> None:
        """``layout()`` of a device built from the same design: the buffers this one lacks allocated in that order (the
        same names), the images where they were (their bytes zero: the owner writes them)."""
        for n, nbytes in layout["buffers"]:
            if n not in self.dev.buffers:
                self.dev.alloc(n, nbytes)
        self._images = dict(layout["images"])
        self._chunk, self._top = layout["chunk"], layout["top"]

    def relocations(self, addrs: list[tuple[int, int]]) -> np.ndarray:
        """Shim BD addresses of a transaction (``txn.Txn.addrs``) as [word index, buffer number, offset] rows: where
        ``relocate`` writes them for another device's buffers."""
        names = list(self.dev.buffers)
        base = np.array([self.dev.address(n) for n in names], np.int64)
        order = np.argsort(base)
        a = np.array([x for _, x in addrs], np.int64)
        k = order[np.searchsorted(base[order], a, side="right") - 1]
        size = np.array([self.dev.view(n).nbytes for n in names], np.int64)
        assert ((a >= base[k]) & (a < base[k] + size[k])).all(), "a shim address outside the buffers"
        return np.stack([np.array([i for i, _ in addrs], np.int64), k, a - base[k]], 1)

    def relocate(self, words: np.ndarray, rel: np.ndarray) -> np.ndarray:
        """words with the addresses of ``relocations`` rows rewritten for this device's buffers (a copy)."""
        base = np.array([self.dev.address(n) for n in self.dev.buffers], np.int64)
        a = base[rel[:, 1]] + rel[:, 2]
        out = words.copy()
        out[rel[:, 0]] = (a & 0xFFFFFFFF).astype(np.uint32)
        out[rel[:, 0] + 1] = (a >> 32).astype(np.uint32)
        return out

    def refs(self, addrs: list[tuple[int, int]]) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Shim BD addresses of a transaction (``txn.Txn.addrs``) by image: their word indices, the images' names and per
        address its image (an index into them) and offset in it (``resolve`` rewrites them for other images: another
        layer's of the same layout)."""
        names = list(self._images)
        base = np.array([self.address(n) for n in names], np.int64)
        size = np.array([self._images[n][2] for n in names], np.int64)
        order = np.argsort(base)
        a = np.array([x for _, x in addrs], np.int64)
        k = order[np.searchsorted(base[order], a, side="right") - 1]
        assert ((a >= base[k]) & (a < base[k] + size[k])).all(), "a shim address outside the images"
        used, inv = np.unique(k, return_inverse=True)
        return np.array([i for i, _ in addrs], np.int64), [names[j] for j in used], np.stack([inv, a - base[k]], 1)

    def resolve(self, words: np.ndarray, refs: tuple, rename: Callable[[str], str]) -> None:
        """Rewrite in place the shim addresses of ``refs`` (``refs``) in words for the images ``rename`` maps theirs to
        (each of at least the bytes the old one had)."""
        idx, names, at = refs
        base = np.array([self.address(rename(n)) for n in names], np.int64)
        a = base[at[:, 0]] + at[:, 1]
        words[idx] = (a & 0xFFFFFFFF).astype(np.uint32)
        words[idx + 1] = (a >> 32).astype(np.uint32)

    def warm(self) -> None:
        """An empty dispatch: the first one after new host buffers takes ~0.15 s more than later ones (a model's load
        ends with it, so that its first step does not)."""
        self.run(self.transaction(self.seq()))


# A model's built state on the design (the device's layout, what its builds wrote besides the weights: commands, norm
# parameters, tables; the weight jobs; its transactions with their shim addresses as relocations) cached in a directory
# per model file: restored, a load skips the builds and packs the weights from the model's files (``weights.Slots``),
# as a build does; the cache never holds a copy of the weights.

def state_dir(kind: str, path: str, sources: tuple[str, ...], files: list[str]) -> str:
    """The cache directory of a model's built state: ``CACHE_DIR/kind/<path's digest>/<key>``, the key covering the
    toolchain, the code that builds the images and transactions (``sources``: files or directories of the src tree)
    and the model's ``files`` (size and mtime)."""
    src = os.path.dirname(PACKAGE_DIR)  # (the artlite and models packages' directory)
    h = hashlib.sha256(version("llvm-aie").encode())
    for s in sources:
        p = os.path.join(src, s)
        for f in sorted(glob.glob(os.path.join(p, "*.py"))) if os.path.isdir(p) else [p]:
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
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    if not isinstance(st, dict) or "slots" not in st or (accept is not None and not accept(st["state"])):
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
    st = {"layout": npu.layout(), "writes": writes, "slots": [b.slots.table(npu) for b in built], "state": state}
    os.makedirs(tmp)
    with open(os.path.join(tmp, "state.pkl"), "wb") as f:
        pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
    for e in os.listdir(root):
        if e != os.path.basename(tmp):
            shutil.rmtree(os.path.join(root, e), ignore_errors=True)
    os.replace(tmp, cache)
