"""The vision overlay's DDR images by name (``ImageStore``): sub-allocated from a few large host buffers of an overlay's
device, written and read by name, laid out the same on another device (``layout`` / ``restore``), and their addresses in
a transaction's shim BDs found by image and rewritten for other images (``refs`` / ``resolve``: another layer's of the
same layout)."""

from collections.abc import Callable

import numpy as np

from ..overlay import OverlayDevice


class ImageStore:
    """DDR images by name (``alloc`` / ``write`` / ``read`` / ``address``) in the host buffers of ``dev``, ``CHUNK``
    bytes each (a dispatch lists every buffer, and the driver's command holds some 500; an image larger than a chunk gets
    a buffer of its own)."""

    CHUNK = 64 << 20

    def __init__(self, dev: OverlayDevice):
        self.dev = dev
        self._images: dict[str, tuple[str, int, int]] = {}  # (name -> chunk, offset, bytes)
        self._chunk: str | None = None  # (the newest chunk's buffer)
        self._top = self.CHUNK  # (the bytes used of the newest chunk)

    def names(self) -> list[str]:
        """The images' names in allocation order."""
        return list(self._images)

    def span(self, name: str) -> tuple[str, int, int]:
        """An image's host buffer, its offset in it and its bytes."""
        return self._images[name]

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
            self._chunk, self._top = f"_chunk{len(self.dev.buffers)}", 0
            self.dev.alloc(self._chunk, self.CHUNK)
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
        """An image's bytes, fetched from the device."""
        chunk, off, n = self._images[name]
        nbytes = n - offset if nbytes is None else nbytes
        self.dev.fetch(chunk, off + offset, nbytes)
        return self.dev.view(chunk)[off + offset:off + offset + nbytes].copy()

    def address(self, name: str, offset: int = 0) -> int:
        chunk, off, _ = self._images[name]
        return self.dev.address(chunk, off + offset)

    def layout(self) -> dict:
        """The DDR layout: the host buffers in creation order (name, bytes), per image its buffer, offset and bytes
        (``restore`` rebuilds it on another device)."""
        return {"buffers": [(n, self.dev.view(n).nbytes) for n in self.dev.buffers], "images": dict(self._images),
                "chunk": self._chunk, "top": self._top}

    def restore(self, layout: dict) -> None:
        """``layout()`` of a device built from the same design: the buffers this one lacks allocated in that order (the
        same names), the images where they were (their bytes zero: the owner writes them)."""
        for n, nbytes in layout["buffers"]:
            if n not in self.dev.buffers:
                self.dev.alloc(n, nbytes)
        self._images = dict(layout["images"])
        self._chunk, self._top = layout["chunk"], layout["top"]

    def refs(self, addrs: list[tuple[int, int]]) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Shim BD addresses of a transaction (``txn.Txn.addrs``) by image: their word indices, the images' names and per
        address its image (an index into them) and offset in it (``resolve`` rewrites them for other images: another
        layer's of the same layout)."""
        names = self.names()
        base = np.array([self.address(n) for n in names], np.int64)
        size = np.array([self._images[n][2] for n in names], np.int64)
        order = np.argsort(base)
        a = np.array([x for _, x in addrs], np.int64)
        k = order[np.searchsorted(base[order], a, side="right") - 1]
        assert ((a >= base[k]) & (a <= base[k] + size[k])).all(), "a shim address outside the images"  # (<=: an end)
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
