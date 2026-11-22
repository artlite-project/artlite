"""A step's GDN layers on the decode overlay: the transfers of gdn_in and the GDN step (``GdnTransfers.build``: the
token-by-token core, the chunkwise core of 4 slots, the one of 8 slots: ``gdn``'s cores) and what the position and the lazy
commit patch into them (``patch_panels``: a chunkwise core's state panels, ``set_words``: gdn_in's commit words).

A chunkwise core's transfers are units in an order (``units``); the state panels of its v heads are the transfers the position
patches: the block's half of GSTATE (``QwenOverlay.cur``).

Several sequences (an overlay of several, 8 slots: ``_chains``): a GDN layer runs a chain per sequence of the step (a
step of one sequence: one chain), each the core's step of that sequence's slots alone, on the kernels of one sequence:

- its words (its slots, its block's lanes before it, its histories' base rows) reach the core in the metadata of its
  sequence's header object, which has ``F_GIN`` beside ``F_GDN`` (``gdn.entry8``'s ``gbegin`` takes the step's words
  where it finds ``F_GIN``: after the gdn_in phase's own object, whose words the header's replace);
- gdn_in's rows are drained per chain into its sequence's objects, the chain's first slot's as the objects' first (a
  drain per group of the stream's objects - the q and k blocks' halves, the v heads' v and z halves, ab - and chain);
- its objects, histories and states are its sequence's (GOBJ, GSTATE: every transfer's address patched per step,
  ``patch_chains``; the state panels from its block's half);
- its outputs, whole heads of 8 slots, go into gdn_out's staged x from the chain's first slot on
  (``decode.protocol.staged_region_bytes``), the block's gates after them into the sink or, the last head's, into the
  sequence's header.

A step of several chains has one transaction per number of chains: the rows' drains and the outputs' are patched with
the chains' slots, and every chain ends its block (``QwenOverlay._step_parts``; a step of one chain: where its slots
do, as a model of one sequence).
"""

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from artlite.decode.attn256 import M_BRANCH
from artlite.decode.protocol import C_LEN, DIMS_OBJ, M_NS, STAGED_X, XB, XPAY, staged_cell
from artlite.decode.step import Part
from artlite.operators.design import COLS
from artlite.overlay import Sequence

from . import gdn
from .image import gdn_v_heads
from .steps import ChainBD, PanelTable, StatePanel, Step
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
        if self.ov.sequences > 1:
            self._chains(seq, st, L, S)
        elif self.ov.g8:
            self._chunkwise(seq, st, L, S, wide=True)
        elif self.ov.g4:
            self._chunkwise(seq, st, L, S, wide=False)
        else:
            self._token(seq, L, S, st)

    def _gdn_in(self, seq: Sequence, L: int, S: int, st: Step | None = None) -> None:
        """gdn_in's weights and input (a PARO checkpoint's: gin_q's and gin_z's, ``paro_steps``)."""
        ov = self.ov
        if ov.paro:
            for name in ("gin_q", "gin_z"):
                ov.paro_phase(seq, st, (L, name))
            return
        ov._weights(seq, (L, "gdn_in"), 0)
        seq.sync()
        ov._inputs(seq, ov.inp[(L, "gdn_in")][0], S, (L, "gdn_in"))

    def _token(self, seq: Sequence, L: int, S: int, st: Step | None = None) -> None:
        """The token-by-token core: its objects whole, the conv q / k and K objects, then per pair of v heads."""
        ov = self.ov
        dev, sh = ov.dev, ov.gsh
        self._gdn_in(seq, L, S, st)
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
                seq.drain("D", c, dev.address("GOBJ", ov.gdn_payload(L, c, 5 + 3 * p) + 2048), gdn.D_BYTES,
                          (512, 2, 1024, 1024))
                seq.drain("D", c, dev.address("GSTATE", ov.gstate[L] + c * ov.ps + 32 * p * gdn.PANEL), 32 * gdn.PANEL)
                seq.drain("D", c, dev.address("GOBJ", ov.gdn_payload(L, c, 7 + 3 * p)), gdn.D_BYTES)
                # the pair's outputs (2 KB) at its heads in gdn_out's x (``gdn_x``), its padding (2 KB) in the sink past
                # that K group's x (its x bytes on)
                at, left, xg = ov.x_slot(*ov.gdn_x(L, c, 2 * p), S)
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

        self._gdn_in(seq, L, S, st)
        if wide:
            self._rows_wide(seq, L, S, pay)
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

        def v_dp(h: int, c: int) -> None:  # (into the other half; of a step that ends its block)
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
            at, left, xg = ov.x_slot(*ov.gdn_x(L, c, h), S)
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
            units(hc, ((kh_x,), (kh_d,)), ((v_xa, v_xp), (v_dp,) * ov.ends + (v_do, v_dh)))

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

    def _chains(self, seq: Sequence, st: Step, L: int, S: int) -> None:
        """(an overlay of several sequences, 8 slots) gdn_in and a GDN chain per sequence of the step (module docstring
        "Several sequences"): gdn_in's rows per chain into its objects' first slots (a sync), then per chain its units
        as ``_chunkwise``'s, every transfer in a sequence's buffer recorded (``Step.chains``: ``patch_chains``)."""
        ov = self.ov
        dev, sh, hc = ov.dev, ov.gsh, ov.gsh.hc
        chains = [(a, n) for a, n, _ in st.subs] if len(st.subs) > 1 else [(0, S)]
        assert ov.ends or len(chains) == 1, "a step of several sequences: every chain ends its block"
        gobj, gstate = dev.address("GOBJ"), dev.address("GSTATE")
        row = 2 * C_LEN  # (bytes of a slot's object of gdn_in's rows)

        def pay(c: int, k: int) -> int:
            return ov.gdn_payload(L, c, k)

        def rec(at: int, u: int, kind: int, offset: int) -> None:
            st.chains.append(ChainBD(at, u, kind, offset))

        self._gdn_in(seq, L, S)

        def dests(c: int) -> list[int]:  # (the stream's groups in its order: a block's objects (half, slot); then ab)
            out = [pay(c, sh.kh_slot(kh) + 1) + 3 * 2 * gdn.D + 8 * row * half for kh in range(2) for half in range(2)]
            out += [pay(c, sh.kh_slot(kh) + 4) + 8 * row * half for kh in range(2) for half in range(2)]
            out += [pay(c, sh.v_slot(h) + 2) + 2048 * z + 8 * row * half for z in range(2) for h in range(hc)
                    for half in range(2)]
            return [*out, pay(c, gdn.H_SLOT) + gdn.H_AB]

        # (per group each chain's slots, in rounds over the columns: a column's drains in a row would hold the others'
        # rows back, a channel queues 4)
        for ds in zip(*(dests(c) for c in range(COLS)), strict=True):
            for u, (_, n) in enumerate(chains):
                for c, d in enumerate(ds):
                    rec(seq.drain("C", c, gobj + d, n * row)[0], u, ChainBD.ROWS, d)
        seq.sync()
        vh = [gdn_v_heads(ov.cfg, c) for c in range(COLS)]
        last = hc - 1  # (the last v head the core runs: k head 1's last)
        cell = staged_cell(ov.SLOTS)
        for u, (a, n) in enumerate(chains):
            def kh_x(kh: int, c: int, u: int = u) -> None:  # (k head 0's group after the header)
                o = pay(c, sh.kh_slot(kh) - (1 if kh == 0 else 0)) - 256
                rec(seq.fill("X", c, gobj + o, (gdn.KH_SPAN + 1 - kh) * XB, DIMS_OBJ), u, ChainBD.GOBJ, o)

            def kh_d(kh: int, c: int, u: int = u) -> None:
                for o, nb in ((pay(c, sh.kh_slot(kh)) + 2048, gdn.D_BYTES), (pay(c, sh.kh_slot(kh) + 2) + 2048, 2 * gdn.D_BYTES)):
                    rec(seq.drain("D", c, gobj + o, nb)[0], u, ChainBD.GOBJ, o)

            def v_xa(h: int, c: int, u: int = u) -> None:
                o = pay(c, sh.v_slot(h)) - 256
                rec(seq.fill("X", c, gobj + o, gdn.V_SPAN * XB, DIMS_OBJ), u, ChainBD.GOBJ, o)

            def v_xp(h: int, c: int, u: int = u) -> None:  # (from the block's half)
                o = ov.gstate[L] + c * ov.ps + 16 * h * gdn.PANEL - 256
                rec(seq.fill("X", c, gstate + o, 16 * XB, DIMS_OBJ), u, ChainBD.PANEL, o)

            def v_dp(h: int, c: int, u: int = u) -> None:  # (into the other half; of a chain that ends its block)
                o = ov.gstate[L] + c * ov.ps + 16 * h * gdn.PANEL
                rec(seq.drain("D", c, gstate + o, 16 * gdn.PANEL)[0], u, ChainBD.UPDATE, o)

            def v_dh(h: int, c: int, u: int = u) -> None:
                o = pay(c, sh.v_slot(h)) + 2048
                rec(seq.drain("D", c, gobj + o, 2 * gdn.D_BYTES)[0], u, ChainBD.GOBJ, o)

            def v_do(h: int, c: int, u: int = u, a: int = a) -> None:
                # (its gated outputs (2 KB: [k block 16][slot 8][8]) into the cells of its head in gdn_out's staged x from the
                # chain's first slot on; the block's gates after them (2 KB): of the last head into the header, else into
                # the sink past that K group's x)
                e = gdn.D * vh[c][h]
                rg = next(r for r in ov.inp[(L, "gdn_out")] if r.k0 <= e < r.k0 + r.K)
                x = rg.off + STAGED_X + cell * ((e - rg.k0) // 8)
                kb = cell // 8  # (words of a k block)
                at = seq.drain("D", c, dev.address("ACT", x + 16 * a), gdn.D_BYTES // 2, (kb, gdn.D // 8, 2 * kb, 2 * kb * gdn.D // 8))[0]
                rec(at, u, ChainBD.OUT, x)
                if h == last:
                    o = pay(c, gdn.H_SLOT) + gdn.H_DN
                    rec(seq.drain("D", c, gobj + o, gdn.D_BYTES // 2)[0], u, ChainBD.GOBJ, o)
                else:
                    sink = rg.off + STAGED_X + cell * (rg.K // 8) + 2 * ov.SLOTS * (e - rg.k0)
                    seq.drain("D", c, dev.address("ACT", sink), gdn.D_BYTES // 2)

            units(hc, ((kh_x,), (kh_d,)), ((v_xa, v_xp), (v_dp,) * ov.ends + (v_dh, v_do)))

    def patch_chains(self, st: Step, parts: tuple[Part, ...]) -> None:
        """(an overlay of several sequences) A step's GDN chains for its parts: every transfer of chain u in the buffers
        of part u's sequence (the state panels from its block's half, their updates into the other), its rows' drains of
        its slots' bytes, its outputs' from its first slot's place on."""
        ov = self.ov
        key = tuple((q.seq, q.n, ov._curs[q.seq].half(q.position)) for q in parts)
        if st.chained == key or not st.chains:
            return
        (t, lines), dev = self.chain_table(st), ov.dev
        at, u, kind, off = t.T
        n = np.array([q.n for q in parts], np.int64)
        a = np.cumsum(n) - n
        half = np.array([ov._curs[q.seq].half(q.position) for q in parts], np.int64)
        base = np.array([[dev.address(ov.seq_name(b, q.seq), own=True) for q in parts] for b in ("GOBJ", "GSTATE")], np.int64)
        addr = np.select(
            [kind == ChainBD.OUT, kind == ChainBD.PANEL, kind == ChainBD.UPDATE],
            [dev.address("ACT") + off + 16 * a[u], base[1][u] + off + half[u] * ov.ghalf, base[1][u] + off + (1 - half[u]) * ov.ghalf],
            base[0][u] + off)
        nbytes = np.where(kind == ChainBD.ROWS, n[u] * 2 * C_LEN, st.tr.words[at].astype(np.int64) * 4)
        ov._bd_words(st.tr.words, at, addr, nbytes)
        st.tr.flush_lines(lines)
        st.chained = key

    @staticmethod
    def chain_table(st: Step) -> tuple[np.ndarray, np.ndarray]:
        """A step's recorded transfers as ``patch_chains`` takes them ((word index, chain, kind, offset) rows; the
        lines their words lie in)."""
        if st.chain_table is None:
            t = np.array(st.chains, np.int64).reshape(-1, 4)
            st.chain_table = (t, np.unique(np.concatenate([4 * t[:, 0], 4 * t[:, 0] + 8]) & ~63))
        return st.chain_table

    def set_chain_words(self, parts: tuple[Part, ...]) -> None:
        """(an overlay of several sequences) The chains' words in their sequences' header objects (per GDN layer and
        column: its slots, its block's lanes before it, the q history's base row, the k / v histories' and the gates'
        first row; a chain: no branch). A chain's slots in a step of several: its part's, or those up to its block's
        end where the part ends before it: the core then runs the lanes past the part's as drafts that nobody
        accepts, on the rows its objects hold, and writes the block's update (``QwenOverlay._step_parts``)."""
        ov = self.ov
        if not ov.gobj:
            return
        hdr = np.array([(ov.gobj[L] + c * ov.gcol) // 4 for L in ov.gobj for c in range(COLS)], np.int64)
        for q in parts:
            cur, name = ov._curs[q.seq], ov.seq_name("GOBJ", q.seq)
            g32 = ov.dev.view(name, own=True).view(np.int32)
            ns = q.n if len(parts) == 1 else max(q.n, 8 - q.position % 8)
            g32[hdr + M_NS], g32[hdr + M_NREP], g32[hdr + M_CSHIFT] = ns, q.position % 8, cur.cshift
            g32[hdr + M_KSHIFT], g32[hdr + M_COMMIT], g32[hdr + M_BRANCH] = cur.kshift, 0, 0
            words = np.array([M_NS, M_NREP, M_CSHIFT, M_COMMIT, M_KSHIFT, M_BRANCH])
            ov.dev.buffers[name].sync_lines(np.unique(4 * (hdr[:, None] + words) & ~63))

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
