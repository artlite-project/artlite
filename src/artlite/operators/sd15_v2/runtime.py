"""sd15_v2's design on the NPU (``V2Device``): its hardware context, DDR buffers, and dispatches of ``seq.Seq``
programs."""

import numpy as np

from ..overlay.runtime import OverlayDevice, Transaction
from . import core as K
from . import design as D
from .seq import Seq


class V2Device:
    """The compiled design and its DDR images by name (``alloc`` / ``write`` / ``read`` / ``address``), sub-allocated
    from a few large host buffers (``CHUNK`` bytes each: a dispatch lists every buffer, and the driver's command holds
    some 500)."""

    CHUNK = 64 << 20

    def __init__(self, design: D.V2Design | None = None, *, device: object = None):
        """device: an ``artlite.Device`` to share (a library's whose entry the design is: its transactions run as the
        library's raw items, ``fused.program.RawTxn``)."""
        self.design = design or D.V2Design()
        self.dev = OverlayDevice(self.design, device=device)
        self._images: dict[str, tuple[str, int, int]] = {}  # (name -> chunk, offset, bytes)
        self._top = self.CHUNK  # (the bytes used of the newest chunk)
        nop = np.zeros(D.Q // 4, np.int32)
        nop[K.C_KIND] = K.K_NOP
        self.alloc("_nop", D.Q)
        self.write("_nop", nop.view(np.uint8))

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
        return Seq(nop=self.address("_nop"))

    def transaction(self, seq: Seq) -> Transaction:
        return Transaction(self.dev, seq.finish())

    def run(self, tr: Transaction) -> None:
        self.dev.dispatch(tr)
