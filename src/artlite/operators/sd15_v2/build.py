"""Building sd15_v2 programs: a dispatch's ops (``seq.Seq``) with their DDR images (commands, weights, tables), the
memtile allocation (the same addresses in every column) and the GroupNorm records' routes.

Tensors of a level (``Level``): C tiles (bf16, ``layout.ct_bytes`` per 80 channels: a residual stream, a norm's
input) or panels (bfp16 [kb][token panel]: a GEMM's / conv's input); in the memtile at an address of every column, or
in DDR as the columns' images of that memtile region (``spill`` / ``load``: one contiguous transfer per column).

GroupNorm records (``Records``): a producer op's per-core records ([core][OCH] in the memtile) drained to DDR [col][core]
[256 B]; an apply's core sums those of the producer rows whose 16-group window is its own (``ops.gn_window``), read from
DDR by its B stream (a shim gather of every column's).
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from ..design import COLS, ROWS
from . import core as K
from . import design as D
from . import layout as L
from .ops import Apply, Gemm, gn_window
from .runtime import V2Device
from .seq import B_BDS, ColOps, MemTask, RowOps

TBIAS, TBIAS_BYTES = "_tbias", 64 * 1024  # (the time-biased convs' command biases: [op][row][passes 80 bf16])


@dataclass(frozen=True)
class Level:
    """A level's geometry per column: R image rows of Lg groups of 8 lanes, both images; its tokens in ``nb`` bands (the
    memtile holds one band's tensors: 32x32 two of 2 rows, 64x64 four), a band's in tiles of m tokens (64: 4 groups of a
    row; 8x8: 16)."""

    H: int
    nb: int = 1

    @property
    def R(self) -> int:
        return self.H // COLS

    @property
    def Lg(self) -> int:
        return self.H // 8

    @property
    def T(self) -> int:  # (a band's tokens per column)
        return 16 * self.R * self.Lg // self.nb

    @property
    def m(self) -> int:
        return min(self.T, 64)

    @property
    def ntile(self) -> int:
        return self.T // self.m

    @property
    def np(self) -> int:  # (token panels per kb)
        return self.T // 8

    def ct(self, C: int) -> int:
        """Bytes of a C-tile tensor of C channels (one band)."""
        return C // 80 * self.ntile * L.ct_bytes(self.m)

    def pn(self, C: int) -> int:
        """Bytes of a panel tensor of C channels (one band)."""
        return C // 8 * self.np * L.PANEL


class Mem:
    """First-fit allocation of the columns' memtile bytes [0, size) (64-byte aligned)."""

    def __init__(self, size: int = D.HEAP):
        self.free = [(0, size)]
        self.size = size

    def alloc(self, n: int, *, top: bool = False) -> int:
        """n bytes from the first free range that holds them (top: the end of the last one: small buffers that live
        across ops, away from the large tensors)."""
        n = -(-n // 64) * 64
        for i in (reversed(range(len(self.free))) if top else range(len(self.free))):
            a, s = self.free[i]
            if s >= n:
                if top:
                    self.free[i] = (a, s - n)
                    a += s - n
                else:
                    self.free[i] = (a + n, s - n)
                if self.free[i][1] == 0:
                    del self.free[i]
                return a
        raise MemoryError(f"memtile: {n} B not free ({self.free})")

    def release(self, a: int, n: int) -> None:
        n = -(-n // 64) * 64
        self.free.append((a, n))
        self.free.sort()
        merged = []
        for b, s in self.free:
            if merged and merged[-1][0] + merged[-1][1] == b:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((b, s))
        self.free = merged


@dataclass
class Records:
    """A producer's GroupNorm records in DDR ([col][core][256 B] at ``addr``): its channels [ch_off, ch_off + 4 nr)
    (nr per core row) in the space of groups of ``gsize``."""

    addr: int
    ch_off: int
    nr: int
    gsize: int

    def rows_for(self, c0: int, n: int) -> list[int]:
        """The producer rows whose records an apply core of channels [c0, c0 + n) sums: of its window, with channels
        of its groups (a group may straddle two producers)."""
        w = gn_window(c0, n, self.gsize)
        lo, hi = c0 // self.gsize * self.gsize, -(-(c0 + n) // self.gsize) * self.gsize
        return [r for r in range(ROWS) if gn_window(self.ch_off + r * self.nr, self.nr, self.gsize) == w
                and self.ch_off + r * self.nr < hi and lo < self.ch_off + (r + 1) * self.nr]


@dataclass
class Builder:
    """A dispatch under construction: ``ops`` in order, DDR images by name (``image``), the memtile (``mem``), the
    activation table each core row holds."""

    v2: V2Device
    mem: Mem = field(default_factory=Mem)

    def __post_init__(self) -> None:
        self.s = self.v2.seq()
        self.tab: dict[int, str | None] = dict.fromkeys(range(ROWS))
        self.tables: dict[str, np.ndarray] = {}
        self.zero = self.mem.alloc(4096)  # (zeros: a conv's halo rows at the image's edges)
        self.tb_hooks: list = []  # (time biases: (resnet prefix, offset in TBIAS, the bias rows [row][passes 80]))
        self.tb_used = 0
        self._zfill = False
        self._tab_runs: dict[int, tuple] = {}  # (row -> a table's B run, taken before the row's next op: ``table``)
        self._rec_names: dict[str, int] = {}  # (records' DDR images by name: their uses)
        self._images: dict[str, int] = {}  # (DDR image name -> its data's hash)
        # (drains for the next op: records, deferred spills; the memtile bytes released after it; records?)
        self._drains: list[tuple[dict[int, ColOps], list[tuple[int, int]], bool]] = []

    # -- DDR images
    def image(self, name: str, data: np.ndarray) -> int:
        """A persistent DDR buffer holding ``data``; returns its device address. A name may be imaged again with the
        same data only (ops of several bands share a layer's images: the dispatch reads what was written last)."""
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        h = hash(data.tobytes())
        assert self._images.setdefault(name, h) == h, f"DDR image {name!r} written again with other data"
        self.v2.alloc(name, data.size)
        self.v2.write(name, data)
        return self.v2.address(name)

    def ddr(self, name: str, nbytes: int) -> int:
        self.v2.alloc(name, nbytes)
        return self.v2.address(name)

    # -- ops
    def op(self, cols: dict[int, ColOps], rows: dict[int, RowOps], *, barrier: bool = False, nolock: bool = False,
           sync: bool = False) -> int:
        """Push an op (``seq.Seq.op``) with the drains pending since the last op (``records``, ``defer``) merged into
        its drains; sync: DDR synced after its memtile tasks, before its B runs (a GroupNorm apply reading them)."""
        if not self._zfill:  # (the zero region, once)
            self._zfill = True
            z = self.image("_zeros", np.zeros(COLS * 4096, np.uint8))
            zc = {c: ColOps(fills=[MemTask([{"addr": self.zero, "nwords": 1024}], ddr=(z + 4096 * c, 4096, None))])
                  for c in range(COLS)}
            self.s.op(zc, {})
        if self._tab_runs:  # (tables first: their K_TABLE commands before the op's; an op of their own where a row's
            # B chain would not take one more run, a parity NOP included)
            tabs = {r: RowOps(runs=[run], b_objects=2, parity=False) for r, run in self._tab_runs.items()}
            self._tab_runs = {}
            if any(len(rows[r].runs) + 2 > len(B_BDS) for r in tabs if r in rows):
                self.s.op({}, tabs)
            else:
                rows = dict(rows)
                for r, t in tabs.items():
                    ro = rows.get(r, RowOps(runs=[], b_objects=0, parity=False))
                    rows[r] = RowOps(runs=t.runs + ro.runs, b_objects=ro.b_objects + 2, parity=ro.parity)
        pending, self._drains = self._drains, []
        if sync:  # (an op synced for its records: at most two drain tasks a column, the rest with the next op)
            take, keep, nd = [], [], max((len(co.drains) for co in cols.values()), default=0)
            for d in sorted(pending, key=lambda d: not d[2]):
                k = max(len(co.drains) for co in d[0].values())
                (take if d[2] or nd + k <= 2 else keep).append(d)
                nd += k if d in take else 0
            pending, self._drains = take, keep
        if pending:
            cols = self.merge(cols, *(c for c, _, _ in pending))
        n = self.s.op(cols, rows, barrier=barrier, nolock=nolock, sync=sync)
        for _, rel, _ in pending:  # (the drains' memtile bytes: later ops' outputs come after these drains)
            for a, nb in rel:
                self.mem.release(a, nb)
        return n

    def defer(self, cols: dict[int, ColOps], release: list[tuple[int, int]] = ()) -> None:
        """Drains (``spill``) with the next op (not synced: ``op``); release: memtile (addr, bytes) freed after it."""
        self._drains.append((cols, list(release), False))

    def load(self, name: str, addr: int, nbytes: int) -> dict[int, ColOps]:
        """Fills of DDR image ``name`` ([col][nbytes]) into the memtile at addr (merge into an op)."""
        base = self.v2.address(name)
        return {c: ColOps(fills=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(base + c * nbytes, nbytes, None))])
                for c in range(COLS)}

    def spill(self, name: str, addr: int, nbytes: int) -> dict[int, ColOps]:
        """Drains of the memtile bytes [addr, addr + nbytes) into DDR image ``name`` ([col][nbytes])."""
        base = self.ddr(name, COLS * nbytes)
        return {c: ColOps(drains=[MemTask([{"addr": addr, "nwords": nbytes // 4}], ddr=(base + c * nbytes, nbytes, None))])
                for c in range(COLS)}

    @staticmethod
    def merge(*ops: dict[int, ColOps]) -> dict[int, ColOps]:
        out: dict[int, ColOps] = {}
        for o in ops:
            for c, co in o.items():
                m = out.setdefault(c, ColOps())
                m.fills += co.fills
                m.drains += co.drains
                for r, ts in co.sends.items():
                    m.sends.setdefault(r, []).extend(ts)
                for r, ts in co.recvs.items():
                    m.recvs.setdefault(r, []).extend(ts)
        return out

    def table(self, name: str, tab: np.ndarray, rows: tuple[int, ...] = (0, 1, 2, 3)) -> None:
        """Make the core rows ``rows`` hold activation table ``name`` (a K_TABLE command where they do not, on their B
        streams before the next op's)."""
        need = [r for r in rows if self.tab[r] != name]
        if not need:
            return
        if name not in self.tables:
            cmd = np.zeros(D.Q // 4, np.int32)
            cmd[K.C_KIND] = K.K_TABLE
            slot = np.zeros(D.Q, np.uint8)
            slot[:tab.size] = tab
            self.tables[name] = self.image(f"_tab_{name}", np.concatenate([cmd.view(np.uint8), slot]))
        for r in need:
            self._tab_runs[r] = (self.tables[name], 2 * D.Q)
            self.tab[r] = name

    def gemm(self, g: Gemm, w: torch.Tensor | None, b: torch.Tensor | None, *, name: str, barrier: bool = False,
             extra: dict[int, ColOps] | None = None, inputs_ready: bool = False, tbias: str | None = None) -> None:
        """A GEMM / conv / copy op (``extra``: fills / drains merged into it; ``name``'s image: per row its command,
        then its weights). A conv is a barrier (its halo rows: the neighbours'), unless inputs_ready (complete before an
        earlier op: its later passes); it clobbers the activation tables. tbias: the resnet whose time bias b includes:
        the commands' biases on the B streams from TBIAS (rewritten per step: ``tb_hooks``)."""
        cmds = [g.command(b, r) for r in range(ROWS)]
        ws = [g.weights(w, r) for r in range(ROWS)] if g.nkt else [np.zeros(0, np.uint8)] * ROWS
        per = D.Q + ws[0].size  # (per row its command, then its weights: one B run)
        addr = self.image(name, np.concatenate([x for r in range(ROWS) for x in (cmds[r], ws[r])]))
        rows = {r: g.row_ops(addr + r * per) for r in range(ROWS)}
        if tbias:  # (the first run in three: the header, the biases from TBIAS, the rest)
            nb, hb = 160 * g.npo, 4 * K.C_BIAS
            if not self.tb_used:
                self.ddr(TBIAS, TBIAS_BYTES)
            off = self.tb_used
            self.tb_used += ROWS * nb
            assert self.tb_used <= TBIAS_BYTES
            tb = self.v2.address(TBIAS)
            for r, ro in rows.items():
                (a, n), *more = ro.runs  # (more: the weights again per further tile)
                self.v2.write(TBIAS, g.bias_bytes(b, r), off + r * nb)
                ro.runs[:] = [(a, hb), (tb + off + r * nb, nb), (a + hb + nb, n - hb - nb), *more]
            self.tb_hooks.append((tbias, off, torch.cat([g._rows(r) for r in range(ROWS)]).numpy()))
        cols = {c: g.col_ops(c) for c in range(COLS)}
        if extra:
            cols = self.merge(cols, extra)
        self.op(cols, rows, barrier=barrier or (bool(g.conv) and not inputs_ready), nolock=inputs_ready)
        if g.conv:
            self.tab = dict.fromkeys(range(ROWS))

    def records(self, rec: int, *, name: str, ch_off: int, nr: int, gsize: int) -> Records:
        """Drain a producer's GroupNorm records (memtile ``rec`` [core][OCH], 4 OCH bytes: released by the drain's op)
        to DDR, with the next op; the consumer's B stream reads them after ``Seq.ddr_sync`` (``apply`` syncs). Each call
        its own DDR image (``name``, numbered from its second use: the bands of one layer)."""
        k = self._rec_names[name] = self._rec_names.get(name, -1) + 1
        base = self.ddr(f"{name}.{k}" if k else name, COLS * ROWS * 256)
        cols = {c: ColOps(drains=[MemTask([{"addr": rec, "nwords": 256, "dims": [(64, 1), (ROWS, D.OCH // 4)]}],
                                          ddr=(base + c * 1024, 1024, None))]) for c in range(COLS)}
        self._drains.append((cols, [(rec, 4 * D.OCH)], True))
        return Records(base, ch_off, nr, gsize)

    def apply(self, a: Apply, gamma: torch.Tensor | None, beta: torch.Tensor | None, *, name: str,
              records: list[Records] = (), extra: dict[int, ColOps] | None = None) -> None:
        """An apply op (GroupNorm: its cores' records gathered from ``records``, after a DDR sync)."""
        runs, cmds, gbs, slots = {}, [], [], {}
        for i, r in enumerate(a.rows):
            rr, nrec = [], 0
            if a.gn:  # (per producer rows' gather: strips of their records in every column, split at slot ends)
                c0, n = a.ch0(r), a.nblk * 80
                for rs in records:
                    sel = rs.rows_for(c0, n)
                    assert sel == list(range(sel[0], sel[-1] + 1)) if sel else True, sel
                    for col in range(COLS) if sel else ():
                        rr.append((rs.addr + col * 1024 + sel[0] * 256, len(sel) * 256))
                        nrec += len(sel)
                assert nrec, (c0, n)
                rr = self._slots(rr)
            runs[r] = rr
            slots[r] = -(-nrec // K.REC_SLOT)
            cmds.append(a.command(r, nrec))
            gbs.append(a.gamma_beta(gamma, beta, r))
        sync = a.gn and any(r for _, _, r in self._drains)  # (records drained with this op: synced before its B runs)
        if a.gn and not sync:
            self.s.ddr_sync()
        addr = self.image(name, np.concatenate([x for i in range(len(a.rows)) for x in (cmds[i], gbs[i])]))
        rows = {r: a.row_ops(addr + 2 * i * D.Q, runs[r], slots[r]) for i, r in enumerate(a.rows)}
        cols = {c: a.col_ops() for c in range(COLS)}
        self.op(self.merge(cols, extra) if extra else cols, rows, sync=sync)

    def _slots(self, strips: list[tuple[int, int]]) -> list[tuple]:
        """Record strips (device address, bytes) packed into B slots of ``core.REC_SLOT`` records (a strip cut at a
        slot's end; each slot padded to Q), runs of equal strips 1 KB apart merged into one shim gather."""
        cap = K.REC_SLOT * 256
        slots, cur, used = [], [], 0
        for a, n in strips:
            while n:
                k = min(n, cap - used)
                cur.append((a, k))
                a, n, used = a + k, n - k, used + k
                if used == cap:
                    slots.append(cur)
                    cur, used = [], 0
        if cur:
            slots.append(cur)
        out = []
        for sl in slots:
            i = 0
            while i < len(sl):
                a, n = sl[i]
                j = i + 1
                while j < len(sl) and sl[j][1] == n and sl[j][0] == a + 1024 * (j - i):
                    j += 1
                out.append((a, n) if j == i + 1 else (a, n * (j - i), (n // 4, j - i, 256, 2048)))
                i = j
            used = sum(n for _, n in sl)
            if used < D.Q:
                out.append((self.zeros_q(), D.Q - used))
        return out

    def zeros_q(self) -> int:
        if "_zq" not in self.v2.dev.buffers:
            self.image("_zq", np.zeros(D.Q, np.uint8))
        return self.v2.address("_zq")

    def transaction(self):
        return self.v2.transaction(self.s)
