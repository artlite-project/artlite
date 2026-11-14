"""The GDN layers' conv states between the decode and the prefill.

The decode keeps a conv state as history rows of the GDN core's objects (GOBJ), the prefill reads and writes it in GHDR: per
GDN layer and column a slot of ``HSLOT`` bytes, the 3 rows x ``ch`` channels bf16 at ``CS_OFF`` (the prefill's conv-state drain
writes 8 KB from there; ``check_slot``). ``ConvHandoff`` copies them across: before a prompt chunk the decode's states (the base
rows of the next step) into GHDR (``for_prefill``), before the next decode step GHDR's into the conv objects' base rows
(``for_decode``). Where they are, and which half of a chunkwise core's states the prefill continues from, the overlay's
``GdnCursor`` says.
"""

import time
from typing import TYPE_CHECKING

import numpy as np

from artlite.decode.protocol import META, XPAY
from artlite.operators.design import COLS

from . import gdn

if TYPE_CHECKING:
    from .overlay import QwenOverlay

CS_OFF = 4 * META + 640  # GHDR: the conv state's bytes in a column's slot
HSLOT = 12288  # GHDR: a column's slot (the prefill's conv-state drain writes 8 KB from CS_OFF)


def check_slot(ch: int) -> None:
    """The conv state of a column's ``ch`` channels fits its GHDR slot."""
    assert CS_OFF + 2 * 3 * ch <= HSLOT, (ch, CS_OFF, HSLOT)


class ConvHandoff:
    """The conv states' copies between the decode overlay ``ov`` (its GDN layers' objects and GHDR) and the prefill."""

    def __init__(self, ov: "QwenOverlay"):
        self.ov = ov
        self._index: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _conv_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """bf16 indices of the GDN layers' conv states: (GOBJ history rows 0 .. 2 of every block, GHDR's [row][channel])
        in the same order, and which are a k or v block's; the decode's base rows are cshift rows past the GOBJ ones
        (chunkwise: a k or v block's, the block's lanes before the position past the history's first row,
        ``_rows_at``)."""
        if self._index is None:
            ov = self.ov
            ch = ov.gsh.ch
            r, j = np.divmod(np.arange(3 * ch), ch)
            b = j // gdn.D
            k = np.where(b < 4, b, 5 + 3 * ((b - 4) // 2) + (b - 4) % 2)
            src, dst = [], []
            at = XPAY * k + 2048
            if ov.gsh.chunk:  # (8 slots: from the A object's second half on, through the B object's)
                at = np.array([ov.gsh.hist_at(int(i)) for i in b])
            for L in ov.gobj:
                for c in range(COLS):
                    src.append(ov.gdn_payload(L, c, 0) + at + 2 * (gdn.D * r + j % gdn.D))
                    dst.append(ov.ghdr[L] + c * HSLOT + CS_OFF + 2 * np.arange(3 * ch))
            kv = np.tile((b >= 2) & ov.blocks, len(ov.gobj) * COLS)
            self._index = (np.concatenate(src) // 2, np.concatenate(dst) // 2, kv)
        return self._index

    def _rows_at(self, kv: np.ndarray, p: int) -> np.ndarray:
        """bf16 offsets of the conv states before position p from their blocks' history rows 0 .. 2 (``_conv_index``)."""
        cur = self.ov.cur
        return gdn.D * np.where(kv, cur.kshift + p % 8, cur.cshift)

    def for_prefill(self) -> None:
        """Before a prompt chunk: the decode's conv states (the base rows of the next step) into GHDR. The decode's GDN
        states must hold every accepted token (no pending replay)."""
        ov, cur = self.ov, self.ov.cur
        if ov.gsh is None or cur.conv_in_ghdr:
            return
        assert cur.nrep == 0, "a prefill after a lazy step: commit its accepted tokens first"
        assert not ov.blocks or cur.gnext % 8 == 0, "a prefill from a GDN block's start"
        dev = ov.dev
        src, dst, kv = self._conv_index()
        dev.fetch("GOBJ")
        dev.view("GHDR").view(np.uint16)[dst] = dev.view("GOBJ").view(np.uint16)[src + self._rows_at(kv, cur.gnext)]
        dev.flush("GHDR")
        cur.gcur = cur.half(cur.gnext)
        cur.conv_in_ghdr = True
        ov.forget_branch()  # (a prompt chunk in between: the next step writes its step words)

    def fresh(self) -> None:
        """Before a sequence's first prompt chunk (at position 0: no conv history, its zero rows the chunk's own,
        ``QwenPrefill._conv_zero``): GHDR's conv states count as current (the chunk's GdnOut writes them) without
        ``for_prefill``'s copy (its fetch of GOBJ: ~0.8 ms on MiniCPM-V)."""
        ov, cur = self.ov, self.ov.cur
        if ov.gsh is None:
            return
        assert cur.nrep == 0, "a prefill after a lazy step: commit its accepted tokens first"
        cur.gcur = cur.half(0)
        cur.conv_in_ghdr = True
        ov.forget_branch()

    def for_decode(self, p: int) -> None:
        """Before a decode step at p after a prefill: GHDR's conv states into the conv objects' base rows (chunkwise:
        the decode's blocks from the prefill's state at p, in its half; the block's lanes before p, none of the sequence's,
        take zero gates: no change of the state, so that blocks start at multiples of 8 and prefills at a block's start)."""
        ov, cur = self.ov, self.ov.cur
        if not cur.conv_in_ghdr:
            return
        t0 = time.perf_counter()
        dev = ov.dev
        cur.nrep, cur.cshift, cur.kshift = 0, 0, 0
        src, dst, kv = self._conv_index()
        dev.fetch("GHDR")
        g16 = dev.view("GOBJ").view(np.uint16)
        g16[src + self._rows_at(kv, p)] = dev.view("GHDR").view(np.uint16)[dst]
        if ov.blocks:
            cur.gp, cur.gh, cur.gnext = p, cur.gcur, p
            gobj = dev.view("GOBJ")
            for L in ov.gobj:
                for c in range(COLS):
                    for a in ([ov.gdn_payload(L, c, gdn.H_SLOT) + gdn.H_DN] if ov.g8 else
                              [ov.gdn_payload(L, c, ov.gsh.kh_slot(kh)) + gdn.P_D for kh in range(2)]):
                        gobj[a:a + 64 * (p % 8)] = 0
        dev.flush("GOBJ")
        cur.conv_in_ghdr = False
        ov.reset_rope()
        ov._once(t0)
