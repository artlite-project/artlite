"""sd15_v2's design on the NPU (``V2Device``): its hardware context, DDR buffers, and dispatches of ``seq.Seq``
programs."""

import numpy as np

from ..overlay.runtime import OverlayDevice, Transaction
from . import core as K
from . import design as D
from .seq import Seq


class V2Device:
    """The compiled design and its host buffers (``alloc`` / ``write`` / ``read`` by name)."""

    def __init__(self, design: D.V2Design | None = None, *, device: object = None):
        """device: an ``artlite.Device`` to share (a library's whose entry the design is: its transactions run as the
        library's raw items, ``fused.program.RawTxn``)."""
        self.design = design or D.V2Design()
        self.dev = OverlayDevice(self.design, device=device)
        nop = np.zeros(D.Q // 4, np.int32)
        nop[K.C_KIND] = K.K_NOP
        self.dev.alloc("_nop", D.Q)[:] = nop.view(np.uint8)
        self.dev.flush("_nop")

    def alloc(self, name: str, nbytes: int) -> np.ndarray:
        """A host buffer of at least nbytes (an existing one of that name if it is large enough; else a new one)."""
        nbytes = -(-nbytes // 4096) * 4096
        b = self.dev.buffers.get(name)
        if b is not None and b.size >= nbytes:
            return self.dev.view(name)
        self.dev.buffers.pop(name, None)
        return self.dev.alloc(name, nbytes)

    def write(self, name: str, data: np.ndarray, offset: int = 0) -> None:
        """Host bytes -> a buffer (flushed)."""
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self.dev.view(name)[offset:offset + data.size] = data
        self.dev.flush(name, offset, data.size)

    def read(self, name: str, nbytes: int | None = None, offset: int = 0) -> np.ndarray:
        n = self.dev.view(name).size - offset if nbytes is None else nbytes
        self.dev.fetch(name, offset, n)
        return self.dev.view(name)[offset:offset + n].copy()

    def address(self, name: str, offset: int = 0) -> int:
        return self.dev.address(name, offset)

    def seq(self) -> Seq:
        return Seq(nop=self.dev.address("_nop"))

    def transaction(self, seq: Seq) -> Transaction:
        return Transaction(self.dev, seq.finish())

    def run(self, tr: Transaction) -> None:
        self.dev.dispatch(tr)
