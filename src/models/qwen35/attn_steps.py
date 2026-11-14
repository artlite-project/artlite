"""A step's attention layers on the decode overlay: the transfers of attn_in and the attention per sub-step
(``AttentionSteps.build``) and what the step's position patches into them (``set_position``).

The position patches three things, each a named part: the F_ATT objects (``att_words``: per sub-step and column the words of
the attention core, pure arithmetic on the attention's shape; their angles), the cache and ring reads (``patch_fills``, ``patch_tails``:
the BDs of ``Step.fills`` and ``Step.tails``) and the record drains (``patch_drains``). A sub-step's records are 8 positions;
four make a group (a K and a V object) of the cache, a ring of two slots holds the open records' V rows
(``decode.vring``).
"""

import time
from typing import TYPE_CHECKING

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.protocol import AB_ID, C_LEN, M_NASKIP, M_NREL, M_NS, M_SKIP, META, XB
from artlite.operators.bfp16 import block_values
from artlite.operators.design import COLS
from artlite.overlay import Sequence

from . import attn
from .steps import FILLS, Drain, Fill, Step, Tail
from .words import M_ANA, M_G0, M_GATT, M_GJ, M_NTAIL, M_POS, M_SA, M_SPLIT

if TYPE_CHECKING:
    from .overlay import QwenOverlay

F_WORDS = {"pos": M_POS, "ns": M_NS, "sa": M_SA, "skip": M_SKIP, "split": M_SPLIT, "nrel": M_NREL, "ntail": M_NTAIL,
           "naskip": M_NASKIP, "g0": M_G0, "gatt": M_GATT, "gj": M_GJ, "ana": M_ANA}  # (an F_ATT object's words, by name)
F_ATT_AT = np.array(list(F_WORDS.values()))
NO_END = 1 << 20  # a core's end of attention groups: none
RECORD_STRIDE = 72  # bytes: a record's place in its pair in a panel (two records in 144 B)
# the columns of a patch table (``AttentionSteps.patch_tables``): BD word index, sub-step, the layer's index among the
# attention layers, the column's place in its chain, byte offsets in KV of group 0 of its KV head and of its ring, and
# a fill's kind (``FILLS``), a tail's object, a drain's index
AT, SUB, LAYER, PLACE, BASE, RING, ARG = range(7)


def att_words(ash: attn.AttnShape, subs: tuple[tuple[int, int, int], ...], p: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The attention cores' words of a step at p with sub-steps ``subs`` (``F_WORDS``, per sub-step and column), the groups
    each kind of fill reads (per sub-step, ``FILLS``, place in a chain: first group, past the last) and per sub-step its first
    position and slots."""
    nph, U = ash.nph, len(subs)
    w = np.zeros((U, COLS, len(F_WORDS)), np.int32)  # (per sub-step and column)
    gr = np.zeros((U, len(FILLS), nph, 2), np.int64)  # (a fill's groups, per place in a chain)
    su = np.zeros((U, 2), np.int64)  # (a sub-step's first position, slots)
    for u, (a, n, split) in enumerate(subs):
        q, g = p + a, (p + a) // attn.POS_PER_GROUP
        # (groups through the last slot's; split: chain core k's attention groups b[k] .. b[k + 1] - 1)
        G = (q + n - 1) // attn.POS_PER_GROUP + 1
        b = ash.bounds(g) if split else [0] * (nph + 1)
        su[u] = q, n
        for c in range(COLS):
            k = c % nph
            nt = ash.records(n) if not split or k == nph - 1 else 0
            o = attn.OBJS
            if not split:  # (a chain's first column: then group g's K and V objects, for its K records; one pass: none)
                na, g0, ga, gj = o * G + nt + (2 if k == 0 and not ash.one else 0), 0, NO_END, 0
            elif k == 0:  # (then group g's K and V objects for its K records)
                na, g0, ga, gj = o * b[1] + 2, 0, b[1], g
            else:
                na, g0, ga, gj = o * (b[k + 1] - b[k]) + nt, b[k], NO_END, 0
            ana, na = na, 0 if ash.one and k else na  # (one pass: the chain's cores on the first column's stream)
            nq = 3 if split & 3 == attn.SPLIT_ROUND else nph + 1 if split else 2  # (per slot the X objects after F_ATT)
            sw = ash.round_word(split, c)
            v = {"pos": q, "ns": n, "sa": 0, "skip": nq * n, "split": sw, "nrel": ash.n_rel(c, sw, n), "ntail": nt,
                 "naskip": na, "g0": g0, "gatt": ga, "gj": gj, "ana": ana}
            w[u, c] = [v[k] for k in F_WORDS]  # (by name: a word of F_WORDS without a value is a KeyError)
        for k in range(nph):
            gr[u, :, k] = (0, G), (g, g + 1), (b[k], b[k + 1])
    return w, gr, su


class AttentionSteps:
    """The attention layers' transfers and patches of the decode overlay ``ov``."""

    def __init__(self, ov: "QwenOverlay"):
        self.ov = ov

    # ------------------------------------------------------------------
    # The transfers
    # ------------------------------------------------------------------

    def build(self, seq: Sequence, st: Step, L: int, S: int, run: list[tuple[int, str]] | None = None) -> None:
        """attn_in and the attention, per sub-step (each after the last one's records are in DDR: it reads them);
        ``run``: the weights after it (a column-major image), after the last sub-step's cache reads."""
        ov = self.ov
        dev, ash = ov.dev, ov.ash
        nph = ash.nph
        per = ash.kvh * attn.HD // COLS  # (K and V rows of a column: its part of KV head per c // HD)
        ov._weights(seq, (L, "attn_in"), 0)
        seq.sync()
        ov._inputs(seq, ov.inp[(L, "attn_in")][0], S, (L, "attn_in"))
        for c in range(COLS):  # [q + gate | K part | V part] per slot -> its q object, the slot's raw K, V object
            h, e0 = divmod(per * c, attn.HD)
            q0 = ov.q_object(L, c, 0) + 4 * META
            seq.drain("C", c, dev.address("ACT", q0), S * ash.qc * 2 * 2 * attn.HD, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            raw = ov.rawkv[L] + 4 * META
            for part in (0, 1):
                seq.drain("C", c, dev.address("ACT", raw + 2 * (2 * attn.HD * h + attn.HD * part + e0)), S * 2 * per,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
        for u, (a, n, split) in enumerate(st.subs):
            rnd = split & 3 == attn.SPLIT_ROUND  # (a round: both chains on its KV head, the cache shared 4 columns apart)
            last = u == len(st.subs) - 1 and run
            if not split & attn.SPLIT_KV:  # (a sub-step's second round: issued with its first)
                if u or rnd:
                    seq.sync()  # (sub-step u - 1's records in DDR; a shared stream: every column's objects before it through)
                self._cache(seq, st, L, u)
                if last and not rnd and ov.a_bds > 1 + attn.TAILS:  # (a later block's after the X objects: ``_weight_run``)
                    ov._weight_run(seq, run, blocks=slice(1))
                if u == 0 and not rnd:
                    seq.sync()  # (attn_in's q, K, V in DDR)
                self._inputs(seq, st, L, u)
                if rnd:  # (the second round, on the same streams after the first's objects: its inputs, then the tasks
                    # that wait for the first round's to finish (a channel queues 4): the second's tails, the drains)
                    self._inputs(seq, st, L, u + 1)
                    self._cache(seq, st, L, u + 1)
                if last and not rnd:
                    ov._weight_run(seq, run, blocks=slice(1 if ov.a_bds > 1 + attn.TAILS else 0, None))
                self._outputs(seq, st, L, S, u)
                if rnd:
                    self._outputs(seq, st, L, S, u + 1)
            if rnd and u == len(st.subs) - 1:  # (then the columns' own objects: they keep no order with a shared stream's)
                seq.wait_fills("A", tuple(range(nph)))
                if run:
                    ov._weight_run(seq, run)

    def _cache(self, seq: Sequence, st: Step, L: int, u: int) -> None:
        """The cache of the column's KV head (address, length: ``set_position``), its ring (one pass: the chain's first
        column's stream, to both cores; a round: the first chain's streams, the round's KV head, cc its column in that
        head's chain)."""
        ov = self.ov
        dev, ash, T, nph = ov.dev, ov.ash, ov.T, ov.ash.nph
        _, n, split = st.subs[u]
        rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
        for c in range(0, COLS, nph if ash.one else 1):
            if rnd and c >= nph:  # (its objects: the column 4 away's)
                continue
            cc = kh * nph + c if rnd else c
            head = dev.address("KV", ov.kv_off[L] + cc // nph * ash.head_bytes)
            pk = {"packet": AB_ID + ash.a_bcast.index((c, c + nph))} if rnd else {}
            kinds = ("all",) if not split else ("run", "new") if c % nph == 0 else ("run",)
            for kind in kinds:  # (split: the chain's first column its run, then the new positions' group's K and V objects)
                st.fills.append(Fill(seq.fill("A", c, head, attn.OBJS * T, **pk), L, cc, u, kind))
            if not split or ash.role(c) == attn.ROLE_V:  # (a column that attends to the newest group: the tails)
                st.tails += [Tail(seq.fill("A", c, dev.address("KV", ov.ring(L, cc // nph)), T, **pk), L, cc, u, t)
                             for t in range(ash.records(n))]
            if not ash.one and not split and c % nph == 0:  # (then the new positions' group again, for their records)
                st.fills.append(Fill(seq.fill("A", c, head, attn.OBJS * T), L, c, u, "new"))

    def _inputs(self, seq: Sequence, st: Step, L: int, u: int) -> None:
        """The F_ATT object, the q objects, the slots' raw K, V objects."""
        ov = self.ov
        dev, nph = ov.dev, ov.ash.nph
        a, n, split = st.subs[u]
        rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", ov.fatt_object(L, c, u)), XB)
            if split:  # (the q objects of the chain's columns in order; a round: of its half of the KV head's)
                q0, nc = (kh * nph + 2 * (c // nph), 2) if rnd else (c - c % nph, nph)
                seq.fill("X", c, dev.address("ACT", ov.q_object(L, q0, a)), nc * n * XB,
                         (XB // 8, 2 * n, XB // 8, ov.SLOTS * XB // 4))
            else:
                seq.fill("X", c, dev.address("ACT", ov.q_object(L, c, a)), n * XB)
            seq.fill("X", c, dev.address("ACT", ov.rawkv[L] + a * XB), n * XB)

    def _outputs(self, seq: Sequence, st: Step, L: int, S: int, u: int) -> None:
        """o (the column's heads in attn_out's x at the sub-step's slots; split: the chain's last column's of the chain's, a
        round's of its half), then the records (addresses: ``set_position``)."""
        ov = self.ov
        dev, ash, nph, qc = ov.dev, ov.ash, ov.ash.nph, ov.ash.qc
        a, n, split = st.subs[u]
        rnd, kh = split & 3 == attn.SPLIT_ROUND, (split & attn.SPLIT_KV) >> 2
        for c in range(COLS):
            role = attn.ROLE_O if ash.round_word(split, c) & attn.SPLIT_NOREC else ash.role(c)
            if not split or c % nph == nph - 1:
                h0, nh = (((kh * nph + 2 * (c // nph)) * qc, 2 * qc) if rnd else
                          (qc * (c - nph + 1), nph * qc) if split else (qc * c, qc))
                sp = ov._sp(S, (L, "attn_out"))
                at, left, _ = ov.x_slot((L, "attn_out"), attn.HD * h0, S)
                assert left >= 4 * sp * 128 * nh
                sw, m = 4 * sp, sp if ash.one else n  # (words of an x k block: [slot sp][8] bf16; the slots out)
                seq.drain("C", c, dev.address("ACT", at + 16 * a), 16 * m * 32 * nh,
                          None if m == sp else (4 * m, 32 * nh, sw, sw * 32 * nh))
            nk = attn.REC_K if ash.one else 2 * attn.REC_K  # (one pass: a K record; sub-steps: its pair)
            nr = ash.records(n)
            sizes = {attn.ROLE_K: (nk,) * nr, attn.ROLE_V: (attn.REC_V, attn.ROWS_B) * nr, attn.ROLE_O: ()}[role]
            for k, nb in enumerate(sizes):  # (a record's panels into its pair: V_SCATTER)
                dims = attn.V_SCATTER if nb == attn.REC_V else None  # (REC_K == REC_V)
                hd = seq.drain("C", c, dev.address("KV", ov.kv_off[L]), nb, dims)
                st.drains.append(Drain(hd, L, c, u, k))

    # ------------------------------------------------------------------
    # The patches of a step's position
    # ------------------------------------------------------------------

    def patch_tables(self, st: Step) -> dict[str, np.ndarray]:
        """A step's cache fills, tail fills and record drains as arrays (columns ``AT`` .. ``ARG``), and the cache lines of
        their BDs."""
        ov, ash = self.ov, self.ov.ash
        nph = ash.nph
        ix = {L: i for i, L in enumerate(ov.fatt)}

        def where(L: int, c: int) -> tuple[int, int, int, int]:
            return ix[L], c % nph, ov.kv_off[L] + c // nph * ash.head_bytes, ov.ring(L, c // nph)

        f = np.array([(r.at, r.sub, *where(r.layer, r.column), FILLS.index(r.kind)) for r in st.fills],
                     np.int64).reshape(-1, 7)
        t = np.array([(r.at, r.sub, *where(r.layer, r.column), r.tail) for r in st.tails], np.int64).reshape(-1, 7)
        r = np.array([(r.handle[0], r.sub, *where(r.layer, r.column), r.index) for r in st.drains],
                     np.int64).reshape(-1, 7)
        at = np.concatenate([x[:, AT] for x in (f, t, r)] + [x[:, AT] + 2 for x in (f, t, r)])
        return {"f": f, "t": t, "r": r, "lines": np.unique(4 * at & ~63)}

    def set_position(self, st: Step, p: int) -> None:
        """The F_ATT objects (per sub-step: its positions, cache objects, rotations) and the step's cache and ring
        transfers."""
        ov = self.ov
        S = st.S
        assert 0 <= p and p + S <= ov.max_positions, (p, S)
        assert st.subs == ov.subs(p, S), (p, S, st.subs)
        c0, s0 = attn.rope_cs(ov.cfg, ov.rope[p:p + S])
        cs = np.concatenate([c0, s0], 1)
        self._rings_from_prefill(p)
        for L in ov.fatt:
            ov.vring.check(L, p, p)
        w, gr, su = att_words(ov.ash, st.subs, p)
        self._write_words(st, w, cs)
        if st.patch is None:
            st.patch = self.patch_tables(st)
        t, kv0, words = st.patch, ov.dev.address("KV"), st.tr.words
        self.patch_fills(words, t["f"], gr, kv0)
        self.patch_tails(words, t["t"], su, kv0)
        self.patch_drains(words, t["r"], su, kv0)
        st.tr.flush_lines(t["lines"])

    def _write_words(self, st: Step, w: np.ndarray, cs: np.ndarray) -> None:
        """The F_ATT objects' words ``w`` (per sub-step and column) and the sub-steps' slots' angles ``cs``, flushed."""
        ov = self.ov
        U = len(st.subs)
        act = ov.dev.view("ACT")
        fo = ov.fobj[:, :, :U]  # (the F_ATT objects' first words: [layer][column][sub-step])
        a32 = act.view(np.int32)
        a32[fo[..., None] + F_ATT_AT] = w[np.arange(U), np.arange(COLS)[:, None]]
        q0 = META + 2 * attn.HD  # (the sub-step's slots' angles, in words)
        af = act.view(np.float32)
        for u, (a, n, _) in enumerate(st.subs):
            af[fo[..., u, None] + q0 + np.arange(64 * n)] = cs[a:a + n].reshape(-1)
        lines = np.concatenate([np.arange(0, 192, 64),
                                4 * q0 + np.arange(0, 4 * 64 * max(n for _, n, _ in st.subs) + 64, 64)])
        ov.dev.flush_at("ACT", 4 * fo[..., None] + lines)

    def patch_fills(self, words: np.ndarray, f: np.ndarray, gr: np.ndarray, kv0: int) -> None:
        """The cache fills: the KV head's groups of the fill's kind (an object of a split column's new group: 2 T)."""
        T = self.ov.T
        g = gr[f[:, SUB], f[:, ARG], f[:, PLACE]]
        self.ov._bd_words(words, f[:, AT], kv0 + f[:, BASE] + attn.OBJS * g[:, 0] * T,
                          np.where(f[:, ARG] == 1, 2, attn.OBJS * (g[:, 1] - g[:, 0])) * T)

    def patch_tails(self, words: np.ndarray, t: np.ndarray, su: np.ndarray, kv0: int) -> None:
        """The ring slots of the sub-step's first and last records."""
        T = self.ov.T
        r = (su[t[:, SUB], 0] + np.where(t[:, ARG], su[t[:, SUB], 1] - 1, 0)) >> 3
        self.ov._bd_words(words, t[:, AT], kv0 + t[:, RING] + (r & 1) * T, np.full(len(r), T, np.int64))

    def patch_drains(self, words: np.ndarray, d: np.ndarray, su: np.ndarray, kv0: int) -> None:
        """The record drains: the first column's K records, the last's V panels (into their pairs) and rows, of record 0, 1."""
        ov = self.ov
        ash, T = ov.ash, ov.T
        v = d[:, PLACE] == ash.nph - 1
        r = (su[d[:, SUB], 0] + np.where(np.where(v, d[:, ARG] >= 2, d[:, ARG] > 0), su[d[:, SUB], 1] - 1, 0)) >> 3
        head, j = d[:, BASE] + attn.OBJS * (r // 4) * T, r % 4
        kind = np.where(v, 1 + d[:, ARG] % 2, 0)  # (K, V panels, rows)
        pair = j // 2 * 2 * attn.REC_K  # (a record in its pair; sub-steps: the K records' pairs whole)
        off = np.choose(kind, [head + pair + (j % 2 * RECORD_STRIDE if ash.one else 0),
                               head + T + pair + j % 2 * RECORD_STRIDE, d[:, RING] + (r & 1) * T])
        nk = attn.REC_K if ash.one else 2 * attn.REC_K
        ov._bd_words(words, d[:, AT], kv0 + off, np.array([nk, attn.REC_V, attn.ROWS_B])[kind])

    def _rings_from_prefill(self, p: int) -> None:
        """Before a step at p, of each attention layer whose caches a prefill wrote last (it writes no ring): its open
        record (p >> 3) as rows into its ring slot, from the record's V panels in the cache (their values are bf16 values:
        exact), so that a step reads every open record from the ring."""
        ov = self.ov
        layers = [L for L in ov.fatt if L in ov.vring.from_prefill]
        if not layers:
            return
        t0 = time.perf_counter()
        r, dev, ash, T = p >> 3, ov.dev, ov.ash, ov.T
        if p % 8:
            kv = dev.view("KV")
            for L in layers:
                for h in range(ash.kvh):  # (the record in its pair [d-block][record 2] of the group's V object)
                    a = ov.kv_off[L] + h * ash.head_bytes + attn.OBJS * (r // 4) * T + T + (r % 4 // 2) * 2 * attn.REC_V
                    dev.fetch("KV", a, 2 * attn.REC_V)
                    v = block_values(kv[a:a + 2 * attn.REC_V].reshape(attn.ND, 2, 8, 9)[:, r % 2])  # ([d-block][dim][position])
                    o = ov.ring(L, h) + (r & 1) * T
                    kv[o:o + attn.ROWS_B] = np.ascontiguousarray(v.transpose(0, 2, 1), bfloat16).view(np.uint8).reshape(-1)
                    dev.flush("KV", o, attn.ROWS_B)
        for L in layers:
            ov.vring.slots.setdefault(L, {})[r & 1] = (r, p)
            ov.vring.from_prefill.discard(L)
        ov._once(t0)
