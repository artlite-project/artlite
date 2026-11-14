"""The vision overlay on the NPU (``VisionDevice``): its hardware context, DDR images (``images``), the roles' pages
(``paging``: one image of two B slots per role, the same for every core), and dispatches of ``seq.Seq`` programs."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np

from ..operators import paging
from ..operators.design import COLS, ROWS
from ..overlay import OverlayDevice, Transaction
from . import core as K
from . import design as D
from .images import ImageStore
from .roles import ROLES, SPECS
from .seq import Seq


def role_pages(work_dir: str) -> paging.Pager:
    """The roles' pages (``roles.SPECS``: entries ``sd15_ext_<role>``, then the role's ``more``, called from the main loop
    ``sd15_run``; the GEMM calls the epilogues named by ``RoleSpec.gemm_calls``) of the compiled design in ``work_dir``,
    the same on every core."""
    p = K.P
    pages = {r: (f"{p}_ext_{r}", *(f"{p}_{e}" for e in SPECS[r].more)) for r in ROLES}
    gemm_calls = dict.fromkeys(f"{p}_{e}" for r in ROLES for e in SPECS[r].gemm_calls)
    return paging.link(work_dir, pages=pages, pager=f"{p}_pg_self",
                       calls={f"{p}_run": tuple(f"{p}_ext_{r}" for r in ROLES), f"{p}_gemm": tuple(gemm_calls)},
                       roots=(f"{p}_main",), stack=D.STACK, cores=[(c, r + 2) for c in range(COLS) for r in range(ROWS)],
                       state=D.L1["ext"], wake=D.L_WAKE)


class VisionDevice(ImageStore):
    """The compiled design, its DDR images by name (``ImageStore``), the roles' pages and the dispatches of ``seq.Seq``
    programs."""

    def __init__(self, design: D.VisionDesign | None = None):
        self.design = design or D.VisionDesign()
        super().__init__(OverlayDevice(self.design))
        self._packing: Future[None] | None = None  # (``pack_behind``)
        self.pager = role_pages(os.path.dirname(self.dev._prog.insts_path))
        nop = np.zeros(D.Q // 4, np.int32)
        nop[K.C_KIND] = K.K_NOP
        self.alloc("_nop", D.Q)
        self.write("_nop", nop.view(np.uint8))
        self.pages = {}
        for role in ROLES:
            pg = self.pager.slots(role, words=D.Q // 4, n=2)
            self.alloc(f"_page_{role}", pg.nbytes)
            self.write(f"_page_{role}", pg)
            self.pages[role] = self.address(f"_page_{role}")

    def read(self, name: str, nbytes: int | None = None, offset: int = 0) -> np.ndarray:
        """An image's bytes, fetched from the device (after the packing in progress)."""
        self.packed()
        return super().read(name, nbytes, offset)

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
        done: Future[None] = Future()

        def run() -> None:
            try:
                pack()
                done.set_result(None)
            except BaseException as e:  # noqa: BLE001 (raised by ``packed``)
                done.set_exception(e)
        threading.Thread(target=run, name="vision-pack").start()
        self._packing = done

    def packed(self) -> None:
        """Until ``pack_behind``'s packing is done (its error raised here)."""
        if self._packing is not None:
            done, self._packing = self._packing, None
            done.result()

    def warm(self) -> None:
        """An empty dispatch: the first one after new host buffers takes ~0.15 s more than later ones (a model's load
        ends with it, so that its first step does not)."""
        self.run(self.transaction(self.seq()))
