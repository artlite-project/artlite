"""A step's GDN layers on the decode overlay: the transfers of gdn_in and the GDN step (``GdnTransfers.build``: the
token-by-token core, the chunkwise core of 4 slots, the one of 8 slots: ``gdn``'s cores) and what the position and the lazy
commit patch into them (``patch_panels``: a chunkwise core's state panels, ``set_words``: gdn_in's commit words).

A chunkwise core's transfers are units in an order (``units``); the state panels of its v heads are the transfers the position
patches: the block's half of GSTATE (``QwenOverlay.cur``).
"""

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from artlite.decode.protocol import C_LEN, DIMS_OBJ, XB, XPAY
from artlite.operators.design import COLS
from artlite.overlay import Sequence

from . import gdn
from .image import gdn_v_heads
from .steps import PanelTable, StatePanel, Step
from .words import M_COMMIT, M_CSHIFT, M_KSHIFT, M_NREP

if TYPE_CHECKING:
    from .overlay import QwenOverlay

Task = Callable[[int, int], None]  # a unit's transfer: (head, column)


def units(hc: int, k: tuple[tuple[Task, ...], tuple[Task, ...]], v: tuple[tuple[Task, ...], tuple[Task, ...]]) -> None:
    """A chunkwise core's transfers in its order: per k head its unit (fills k[0], drains k[1]: task(head, column)), then
    its v heads' (v[0], v[1]); each unit's fills ahead of the drains of the one before (a channel's fifth task waits for
    its oldest: the core then has its next objects queued while the firmware waits for drains; at most 4 tasks a unit on
    a channel), in rounds of one task over the columns (a column's tasks in a row would leave the last column's first one
    ~10 us behind the first column's, at every unit): a head's panels drain before the next head's panels fill, whose wait
    ends only when this head's panels are in."""
    order = [u for kh in range(2) for u in [(*k, kh)] + [(*v, h) for h in range(kh, hc, 2)]]
    for i in range(len(order) + 1):
        fills = [(f, order[i][2]) for f in order[i][0]] if i < len(order) else []
        drains = [(d, order[i - 1][2]) for d in order[i - 1][1]] if i else []
        for pair in itertools.zip_longest(fills, drains):
            for task, unit in filter(None, pair):
                for c in range(COLS):
                    task(unit, c)


class GdnTransfers:
    """The GDN layers' transfers and patches of the decode overlay ``ov``."""

    def __init__(self, ov: "QwenOverlay"):
        self.ov = ov

    def build(self, seq: Sequence, st: Step, L: int, S: int) -> None:
        """gdn_in and the GDN step of layer L."""
        if self.ov.g8:
            self._chunkwise(seq, st, L, S, wide=True)
        elif self.ov.g4:
            self._chunkwise(seq, st, L, S, wide=False)
        else:
            self._token(seq, L, S)

    def _gdn_in(self, seq: Sequence, L: int, S: int) -> None:
        """gdn_in's weights and input."""
        ov = self.ov
        ov._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        ov._inputs(seq, ov.inp[(L, "gdn_in")][0], S, (L, "gdn_in"))

    def _token(self, seq: Sequence, L: int, S: int) -> None:
        """The token-by-token core: its objects whole, the conv q / k and K objects, then per pair of v heads."""
        ov = self.ov
        dev, sh = ov.dev, ov.gsh
        self._gdn_in(seq, L, S)
        for c in range(COLS):
            seq.fill("X", c, dev.address("GOBJ", ov.gobj[L] + c * ov.gcol), 5 * XB, DIMS_OBJ)
        for c in range(COLS):
            seq.drain("D", c, dev.address("GOBJ", ov.gdn_payload(L, c, 0) + 2048), 2 * gdn.D_BYTES, (512, 2, 1024, 2048))
            seq.drain("D", c, dev.address("GOBJ", ov.gdn_payload(L, c, 4)), gdn.D_BYTES)
        for p in range(ov.npair):
            for c in range(COLS):
                seq.fill("X", c, dev.address("GOBJ", ov.gdn_payload(L, c, 5 + 3 * p) - 256), 3 * XB, DIMS_OBJ)
                seq.fill("X", c, dev.address("GSTATE", ov.gstate[L] + c * ov.ps + 32 * p * gdn.PANEL - 256), 32 * XB,
                         DIMS_OBJ)
            for c in range(COLS):
                h0 = gdn_v_heads(ov.cfg, c)[2 * p]
                seq.drain("D", c, dev.address("GOBJ", ov.gdn_payload(L, c, 5 + 3 * p) + 2048), gdn.D_BYTES,
                          (512, 2, 1024, 1024))
                seq.drain("D", c, dev.address("GSTATE", ov.gstate[L] + c * ov.ps + 32 * p * gdn.PANEL), 32 * gdn.PANEL)
                seq.drain("D", c, dev.address("GOBJ", ov.gdn_payload(L, c, 7 + 3 * p)), gdn.D_BYTES)
                # the pair's outputs (2 KB) at its heads in gdn_out's x, its padding (2 KB) in the sink past that K
                # group's x (its x bytes on)
                at, left, xg = ov.x_slot((L, "gdn_out"), gdn.D * h0, S)
                assert left >= 2048
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES, (512, 2, xg // 4, 1024))
        assert sh.ch == (4 + sh.hc) * gdn.D

    def _chunkwise(self, seq: Sequence, st: Step, L: int, S: int, *, wide: bool) -> None:
        """gdn_in and the GDN step of a chunkwise core (``gdn``'s "Chunkwise", "4 slots" and "8 slots" (``wide``)). 4 slots:
        gdn_in's rows on M, per k head its group and per v head its group and panels. 8 slots: gdn_in's rows per slot into the q
        blocks' second objects, the k heads' K objects and the v heads' R objects (v, then z), ab into the header (a sync),
        then per k head its group and per v head its group and panels (``units``)."""
        ov = self.ov
        dev, sh, hc = ov.dev, ov.gsh, ov.gsh.hc

        def pay(c: int, k: int) -> int:
            return dev.address("GOBJ", ov.gdn_payload(L, c, k))

        self._gdn_in(seq, L, S)
        if wide:
            self._rows_wide(seq, L, S, pay)
        vh = [gdn_v_heads(ov.cfg, c) for c in range(COLS)]
        last = hc - 1  # (the last v head the core runs: k head 1's last)
        kh_span, v_span = (gdn.KH_SPAN, gdn.V_SPAN) if wide else (gdn.KH_SPAN4, gdn.V_SPAN4)

        def kh_x(kh: int, c: int) -> None:
            if wide:  # (k head 0's group after the header)
                g0 = sh.kh_slot(kh) - (1 if kh == 0 else 0)
                seq.fill("X", c, pay(c, g0) - 256, (kh_span + 1 - kh) * XB, DIMS_OBJ)
            else:
                seq.fill("X", c, pay(c, sh.kh_slot(kh)) - 256, kh_span * XB, DIMS_OBJ)

        def kh_d(kh: int, c: int) -> None:
            if wide:  # (the q history from the q A object's second half; the k history from the k A object's, over the k B
                # object, its last 2 KB into the K object's taken rows)
                seq.drain("D", c, pay(c, sh.kh_slot(kh)) + 2048, gdn.D_BYTES)
                seq.drain("D", c, pay(c, sh.kh_slot(kh) + 2) + 2048, 2 * gdn.D_BYTES)
            else:  # (the P and K objects, a W object apart)
                seq.drain("D", c, pay(c, sh.kh_slot(kh)), 2 * gdn.D_BYTES,
                          (gdn.D_BYTES // 8, 2, gdn.D_BYTES // 8, 2 * (XPAY // 4)))

        def v_xa(h: int, c: int) -> None:
            seq.fill("X", c, pay(c, sh.v_slot(h)) - 256, v_span * XB, DIMS_OBJ)

        def v_xp(h: int, c: int) -> None:  # (from the block's half; the step's position patches them: ``patch_panels``)
            o = ov.gstate[L] + c * ov.ps + 16 * h * gdn.PANEL - 256
            st.panels.append(StatePanel(seq.fill("X", c, dev.address("GSTATE", o), 16 * XB, DIMS_OBJ), o, 16 * XB, False))

        def v_dp(h: int, c: int) -> None:  # (into the other half; 8 slots: a step that ends its block)
            o = ov.gstate[L] + c * ov.ps + 16 * h * gdn.PANEL
            st.panels.append(StatePanel(seq.drain("D", c, dev.address("GSTATE", o), 16 * gdn.PANEL), o,
                                        16 * gdn.PANEL, True))

        def v_dh(h: int, c: int) -> None:
            if wide:  # (its history from the A object's second half, over the B object, its last 2 KB into the R object's
                # taken v rows)
                seq.drain("D", c, pay(c, sh.v_slot(h)) + 2048, 2 * gdn.D_BYTES)
            else:  # (its history, then its deltas: the A object's second half and the U object after it)
                seq.drain("D", c, pay(c, sh.v_slot(h)) + 2048, 3 * gdn.D_BYTES // 2)

        def v_do(h: int, c: int) -> None:
            at, left, xg = ov.x_slot((L, "gdn_out"), gdn.D * vh[c][h], S)
            if not wide:  # (its gated outputs (1 KB) at its head in gdn_out's x, 1 KB of padding into the sink past that K
                # group's x)
                assert left >= 2 * ov.SLOTS * gdn.D
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES // 2, (gdn.D_BYTES // 16, 2, xg // 4, 1))
                return
            # (its gated outputs (2 KB) at its head in gdn_out's x, the block's gates after them (2 KB): of the last head
            # into the header, else into the sink past that K group's x)
            assert left >= 2048
            if h == last:
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES // 2)
                seq.drain("D", c, pay(c, gdn.H_SLOT) + gdn.H_DN, gdn.D_BYTES // 2)
            else:
                seq.drain("D", c, dev.address("ACT", at), gdn.D_BYTES, (512, 2, xg // 4, 512))

        if wide:
            units(hc, ((kh_x,), (kh_d,)), ((v_xa, v_xp), (v_dp,) * ov.ends + (v_dh, v_do)))
        else:
            units(hc, ((kh_x,), (kh_d,)), ((v_xa, v_xp), (v_dp, v_do, v_dh)))

    def _rows_wide(self, seq: Sequence, L: int, S: int, pay: Callable[[int, int], int]) -> None:
        """An 8-slot core's rows of gdn_in: per slot into the q blocks' second objects, the k heads' K objects and the v
        heads' R objects (v, then z), ab into the header, then a sync."""
        sh, hc = self.ov.gsh, self.ov.gsh.hc
        X = XPAY // 4  # (words)
        # (a block's objects (half, slot) as [half][slot 8][64]: half 0's S rows, then half 1's 1 KB on; per execution its
        # pair of blocks of the two k heads, or the hc v blocks; then the next: k after q, z after v)
        hb = 2 * 64 * 8 // 4  # (words between the halves)
        for c in range(COLS):
            q0 = pay(c, sh.kh_slot(0) + 1) + 3 * 2 * gdn.D  # (q0's rows, past the history's rows 8 .. 10)
            seq.drain("C", c, q0, 2 * 2 * S * 2 * C_LEN, (S * C_LEN // 2, 2, hb, gdn.KH_SPAN * X), 2,
                      ((pay(c, sh.kh_slot(0) + 4) - q0) // 4, 2))  # q, then k rows (the K objects)
            v0 = pay(c, sh.v_slot(0) + 2)
            seq.drain("C", c, v0, hc * 2 * S * 2 * C_LEN, (S * C_LEN // 2, 2, hb, gdn.V_SPAN * X), 2,
                      (2048 // 4, 2))  # v rows, then z (the R objects' halves)
            seq.drain("C", c, pay(c, gdn.H_SLOT) + gdn.H_AB, 2 * S * C_LEN)  # ab
        seq.sync()

    def patch_panels(self, st: Step, p: int) -> None:
        """(chunkwise) A step's state transfers for position p: the panels from p's block's half, into the other."""
        ov = self.ov
        h = ov.cur.half(p)
        if st.half == h:
            return
        if st.table is None:
            st.table = PanelTable.of(st.panels)
        t = st.table
        ov._bd_words(st.tr.words, t.at, ov.dev.address("GSTATE") + t.offset + np.where(t.drain, 1 - h, h) * ov.ghalf,
                     t.nbytes)
        st.tr.flush_lines(t.lines)
        st.half = h

    def set_words(self, *, nrep: int, cshift: int, commit: bool) -> None:
        """The GDN layers' gdn_in metadata words of the lazy commit (chunkwise: the step's lanes in its block before
        it, the q history's base row, the k / v histories' and the gates' first row)."""
        ov = self.ov
        if ov.gobj:
            a32 = ov.dev.view("ACT").view(np.int32)
            a32[ov.gin + M_NREP], a32[ov.gin + M_CSHIFT], a32[ov.gin + M_COMMIT] = nrep, cshift, int(commit)
            a32[ov.gin + M_KSHIFT] = ov.cur.kshift
            ov.dev.flush_at("ACT", 4 * (ov.gin + M_NREP) & ~63)
