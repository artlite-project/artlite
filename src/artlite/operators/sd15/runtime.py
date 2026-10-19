"""sd15's design on the NPU (``Sd15Device``): its hardware context, DDR buffers, the roles' pages (``paging``: one
image of two B slots per role, the same for every core), and dispatches of ``seq.Seq`` programs."""

import os

import numpy as np

from .. import paging
from ..design import COLS, ROWS
from ..overlay.runtime import OverlayDevice, Transaction
from . import core as K
from . import design as D
from .seq import Seq


def role_pages(work_dir: str) -> paging.Pager:
    """The roles' pages (``core.ROLES``, entries ``sd15_ext_<role>``, called from the main loop ``sd15_run``; "ln" also
    ``sd15_geglu`` and ``sd15_gelu``, the GEMM's) of the compiled design in ``work_dir``, the same on every core."""
    p = K.P
    pages = {r: (f"{p}_ext_{r}",) + ((f"{p}_geglu", f"{p}_gelu") if r == "ln" else ()) for r in K.ROLES}
    return paging.link(work_dir, pages=pages, pager=f"{p}_pg_self",
                       calls={f"{p}_run": tuple(f"{p}_ext_{r}" for r in K.ROLES), f"{p}_gemm": (f"{p}_geglu", f"{p}_gelu")},
                       roots=(f"{p}_main",), stack=D.STACK, cores=[(c, r + 2) for c in range(COLS) for r in range(ROWS)],
                       state=D.L1["ext"], wake=D.L_WAKE)


class Sd15Device:
    """The compiled design and its DDR images by name (``alloc`` / ``write`` / ``read`` / ``address``), sub-allocated
    from a few large host buffers (``CHUNK`` bytes each: a dispatch lists every buffer, and the driver's command holds
    some 500)."""

    CHUNK = 64 << 20

    def __init__(self, design: D.Sd15Design | None = None):
        self.design = design or D.Sd15Design()
        self.dev = OverlayDevice(self.design)
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
        self.dev.dispatch(tr)

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

    def warm(self) -> None:
        """An empty dispatch: the first one after new host buffers takes ~0.15 s more than later ones (a model's load
        ends with it, so that its first step does not)."""
        self.run(self.transaction(self.seq()))
