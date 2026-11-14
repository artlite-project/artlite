"""Building the vision overlay's programs: a dispatch's ops (``seq.Seq``) with their DDR images (commands, weights,
tables), the memtile allocation (the same addresses in every column) and the GroupNorm records' routes.

Tensors of a level (``levels.Level``): C tiles (bf16, ``layout.ct_bytes`` per 80 channels: a residual stream, a norm's
input) or panels (bfp16 [kb][token panel]: a GEMM's / conv's input); in the memtile at an address of every column, or
in DDR as the columns' images of that memtile region (``Builder.drain`` / ``fill``: one contiguous transfer per column).

GroupNorm records (``Records``): a producer op's per-core records ([core][OCH] in the memtile) drained to DDR [col][core]
[256 B]; an apply's core sums those of the producer rows whose 16-group window is its own (``ops.gn_window``), read from
DDR by its B stream (a shim gather of every column's).

A GEMM's weights go into its DDR image as jobs (``weights.Slots``), packed by the native packer when the dispatch is
finished; what the host wrote besides them (commands, norm parameters, tables) is ``Built.writes``: a model's cached
build keeps both (``state.save_state``), never the weights.
"""

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..operators.design import COLS, ROWS
from ..overlay import Transaction
from . import core as K
from . import design as D
from .device import VisionDevice
from .memtile import Mem
from .ops import ZERO_BYTES, Apply, Gemm, gn_window
from .ops.common import limit
from .roles import SPECS
from .seq import B_BDS, ColOps, MemTask, RowOps
from .weights import Slots, Wt, as_wt

TBIAS, TBIAS_BYTES = "_tbias", 64 * 1024  # (the time-biased convs' command biases: [op][row][passes 80 bf16])


def _transfer(base: int, col: int, image_bytes: int, off: int, mem: int, nbytes: int, mdims: list | None,
              sdims: tuple | None) -> MemTask:
    """One column's DDR <-> memtile task: ``nbytes`` of the memtile at ``mem`` (walk ``mdims``) and of the DDR image at
    ``base`` ([col][image_bytes]) from ``off`` in the column's (shim dims ``sdims``)."""
    bd = {"addr": mem, "nwords": nbytes // 4} | ({"dims": mdims} if mdims else {})
    return MemTask([bd], ddr=(base + col * image_bytes + off, nbytes, sdims))


@dataclass(frozen=True)
class TimeBias:
    """A resnet's time-biased conv: the offset of its biases in TBIAS and its weight rows [row][passes 80] (the host
    rewrites them per step: ``models.sd15.model.UNet.set_time``)."""

    resnet: str
    offset: int
    rows: np.ndarray


@dataclass
class Records:
    """A producer's GroupNorm records in DDR ([col][core][256 B] at ``addr``): its channels [ch_off, ch_off + 4 nr)
    (nr per core row) in the space of groups of ``gsize``."""

    addr: int
    ch_off: int
    nr: int
    gsize: int
    split: int = 1  # (core rows of a channel slice: ``ops.Gemm`` tsplit)

    def rows_for(self, c0: int, n: int) -> list[int]:
        """The producer rows whose records an apply core of channels [c0, c0 + n) sums: of its window, with channels
        of its groups (a group may straddle two producers)."""
        w = gn_window(c0, n, self.gsize)
        lo, hi = c0 // self.gsize * self.gsize, -(-(c0 + n) // self.gsize) * self.gsize
        a = [self.ch_off + r // self.split * self.nr for r in range(ROWS)]
        return [r for r in range(ROWS) if gn_window(a[r], self.nr, self.gsize) == w and a[r] < hi and lo < a[r] + self.nr]


@dataclass
class Built:
    """What a model's builds wrote into the device's DDR images (``Builder.built``): the host's byte ranges (commands,
    norm parameters, tables; image -> {offset: bytes}) and the weight jobs."""

    writes: dict[str, dict[int, int]]
    slots: Slots


@dataclass
class Builder:
    """A dispatch under construction: ``ops`` in order, DDR images by name (``image``), the memtile (``mem``), the
    activation table each core row holds. reuse: an image the device holds already (a restored model's: its weights,
    commands) is taken by its name as it is, neither converted nor written again."""

    npu: VisionDevice
    mem: Mem = field(default_factory=Mem)
    reuse: bool = False

    def __post_init__(self) -> None:
        self.s = self.npu.seq()
        self.tab: dict[int, str | None] = dict.fromkeys(range(ROWS))
        self.tables: dict[str, np.ndarray] = {}
        self.zero = self.mem.alloc(ZERO_BYTES)  # (zeros: a conv's halo rows at the image's edges)
        self.tb_hooks: list[TimeBias] = []  # (the time-biased convs: ``TimeBias``)
        self.tb_used = 0
        self._zfill = False
        self._tab_runs: dict[int, tuple] = {}  # (row -> a table's B run, taken before the row's next op: ``table``)
        self._rec_names: dict[str, int] = {}  # (records' DDR images by name: their uses)
        self._rec_img: dict[str, tuple] = {}  # (records' DDR images by name: the current one's address, its bands)
        self._images: dict[str, int] = {}  # (DDR image name -> its data's hash)
        self._ddr: dict[str, int] = {}  # (DDR image name -> its bytes)
        self._gemm_images: dict[str, tuple] = {}  # (a GEMM's image name -> (its commands' hash, its weights), address, row bytes)
        self._dirty: set[str] = set()  # (DDR images drained into since the last DDR sync)
        self.slots = Slots()  # (the GEMMs' weight jobs: packed by ``finish``)
        self._writes: dict[str, dict[int, int]] = {}  # (the host's writes of DDR images: ``Built.writes``)
        # (drains for the next op: records, deferred spills; the memtile bytes released after it; records?)
        self._drains: list[tuple[dict[int, ColOps], list[tuple[int, int]], bool]] = []

    def fork(self) -> "Builder":
        """A builder of another dispatch on the same device that shares this one's DDR images (weights, commands,
        tables: written once, the same names in both)."""
        b = Builder(self.npu, reuse=self.reuse)
        b.tables, b._images, b._ddr, b._gemm_images = self.tables, self._images, self._ddr, self._gemm_images
        b.slots, b._writes = self.slots, self._writes
        return b

    def built(self) -> Built:
        """What this builder and its forks wrote (``Built``)."""
        return Built(self._writes, self.slots)

    # -- DDR images
    def image(self, name: str, data: np.ndarray) -> int:
        """A persistent DDR buffer holding ``data``; returns its device address. A name may be imaged again with the
        same data only (ops of several bands share a layer's images: the dispatch reads what was written last)."""
        if self.reuse and self.npu.has(name):
            return self.npu.address(name)
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        h = hash(data.tobytes())
        assert self._images.setdefault(name, h) == h, f"DDR image {name!r} written again with other data"
        self.ddr(name, data.size)
        self.put(name, data)
        return self.npu.address(name)

    def put(self, name: str, data: np.ndarray, offset: int = 0) -> None:
        """Host bytes into DDR image ``name`` at offset (a build's: ``Built.writes``)."""
        self.npu.write(name, data, offset)
        w = self._writes.setdefault(name, {})
        w[offset] = max(w.get(offset, 0), data.size)

    def ddr(self, name: str, nbytes: int) -> int:
        """The device address of DDR image ``name`` of nbytes (a name this dispatch took with fewer bytes is an error:
        ops pushed so far read that image)."""
        assert self._ddr.setdefault(name, nbytes) >= nbytes, f"DDR image {name!r} of {self._ddr[name]} B asked for {nbytes} B"
        self.npu.alloc(name, nbytes)
        return self.npu.address(name)

    # -- ops
    def op(self, cols: dict[int, ColOps], rows: dict[int, RowOps], *, barrier: bool = False, nolock: bool = False,
           sync: bool = False) -> int:
        """Push an op (``seq.Seq.op``) with the drains pending since the last op (``records``, ``defer``) merged into
        its drains; sync: DDR synced after its memtile tasks, before its B runs (a GroupNorm apply reading them)."""
        for r, ro in rows.items():  # (a role's ops that use the table buffer: d 40's attention scores)
            if ro.role and SPECS[ro.role].clobbers_table:
                self.tab[r] = None
        if not self._zfill:  # (the zero region, once)
            self._zfill = True
            z = self.image("_zeros", np.zeros(COLS * ZERO_BYTES, np.uint8))
            zc = {c: ColOps(fills=[MemTask([{"addr": self.zero, "nwords": ZERO_BYTES // 4}], ddr=(z + ZERO_BYTES * c, ZERO_BYTES, None))])
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
                    rows[r] = dataclasses.replace(ro, runs=t.runs + ro.runs, b_objects=ro.b_objects + 2)
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

    def flush(self) -> None:
        """The drains pending (``defer``) pushed with an op of their own, their memtile bytes released."""
        if self._drains:
            self.op({}, {})

    def ddr_sync(self) -> None:
        """Every drain pushed so far written to DDR (``seq.Seq.ddr_sync``)."""
        self.s.ddr_sync()
        self._dirty.clear()

    def written(self, name: str) -> None:
        """DDR image ``name`` is drained into (a later fill of it syncs first: ``reading``)."""
        self._dirty.add(name)

    def reading(self, name: str) -> None:
        """A fill of DDR image ``name`` follows: if a drain wrote it since the last sync, that drain pushed (pending
        deferred drains with an op of their own) and DDR synced (a fill's shim read is not ordered after a drain's
        writes)."""
        if name in self._dirty:
            if self._drains:
                self.op({}, {})
            self.ddr_sync()

    def fill(self, name: str, image_bytes: int, *, mem: int, nbytes: int, off: int = 0, mdims: list | None = None,
             sdims: tuple | None = None) -> dict[int, ColOps]:
        """Per column a fill of ``nbytes`` of DDR image ``name`` ([col][image_bytes], from ``off`` in the column's) into
        the memtile at ``mem`` (merge into an op): the memtile walk ``mdims`` (``ops.dims`` form), the shim's ``sdims``
        (``txn.shim_bd``)."""
        base = self.ddr(name, COLS * image_bytes)
        self.reading(name)
        return {c: ColOps(fills=[_transfer(base, c, image_bytes, off, mem, nbytes, mdims, sdims)]) for c in range(COLS)}

    def drain(self, name: str, image_bytes: int, *, mem: int, nbytes: int, off: int = 0, mdims: list | None = None,
              sdims: tuple | None = None) -> dict[int, ColOps]:
        """Per column a drain of the memtile bytes [mem, mem + nbytes) into DDR image ``name`` ([col][image_bytes], at
        ``off`` in the column's); ``mdims``, ``sdims``: ``fill``'s."""
        base = self.ddr(name, COLS * image_bytes)
        self.written(name)
        return {c: ColOps(drains=[_transfer(base, c, image_bytes, off, mem, nbytes, mdims, sdims)]) for c in range(COLS)}

    def load(self, name: str, addr: int, nbytes: int) -> dict[int, ColOps]:
        """Fills of DDR image ``name`` ([col][nbytes]) into the memtile at addr (merge into an op)."""
        return self.fill(name, nbytes, mem=addr, nbytes=nbytes)

    def spill(self, name: str, addr: int, nbytes: int) -> dict[int, ColOps]:
        """Drains of the memtile bytes [addr, addr + nbytes) into DDR image ``name`` ([col][nbytes])."""
        return self.drain(name, nbytes, mem=addr, nbytes=nbytes)

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

    def gemm(self, g: Gemm, w: Wt | np.ndarray | None, b: np.ndarray | None, *, name: str, barrier: bool = False,
             extra: dict[int, ColOps] | None = None, inputs_ready: bool = False, tbias: str | None = None,
             wname: str = "") -> None:
        """A GEMM / conv / copy op (``extra``: fills / drains merged into it; ``name``'s image: per row its command,
        then its weights: jobs, packed by ``finish``; wname: the weights in an image of that name, shared by a layer's
        command variants). A conv is a barrier (its halo rows: the neighbours'), unless inputs_ready (complete before an
        earlier op: its later passes); it clobbers the activation tables. tbias: the resnet whose time bias b includes:
        the commands' biases on the B streams from TBIAS (rewritten per step: ``tb_hooks``)."""
        assert not (wname and tbias)
        cmds = [g.command(b, r) for r in range(ROWS)]
        assert w is not None or not g.nkt, f"{name}: a GEMM of {g.nkt} k-tiles without weights"
        w = None if w is None or not g.nkt else as_wt(w)
        rows = self._weight_image(g, w, cmds, name, wname, key=(hash(b"".join(c.tobytes() for c in cmds)),
                                                                 None if w is None else w.key))
        if tbias:
            self._time_bias(g, b, rows, tbias)
        co = g.col_ops(0) if g.col_free else None  # (one ColOps for every column: its tasks' words built once)
        cols = {c: co or g.col_ops(c) for c in range(COLS)}
        if extra:
            cols = self.merge(cols, extra)
        self.op(cols, rows, barrier=barrier or (g.conv is not None and not inputs_ready), nolock=inputs_ready)
        if g.conv is not None:
            self.tab = dict.fromkeys(range(ROWS))

    def _weight_image(self, g: Gemm, w: Wt | None, cmds: list[np.ndarray], name: str, wname: str, *, key: tuple
                      ) -> dict[int, RowOps]:
        """The DDR images of a GEMM's commands and weights (jobs, packed by ``finish``) and its rows' B streams: the
        weights in the image ``wname`` (shared by a layer's command variants: ``name`` holds the commands), or per row
        its command then its weights in ``name`` (a layer's bands: its image again, not converted again; a restored
        model's: as it is)."""
        wb = g.npo * g.nkt * D.Q  # (a row's weight bytes: its passes' k-tiles of B slots)
        if wname:
            addr = self.image(name, np.concatenate(cmds))
            if wname not in self._gemm_images:
                wa = self.ddr(wname, ROWS * wb)
                self.slots.add(g, w, {r: (wname, r * wb) for r in range(ROWS)})
                self._gemm_images[wname] = (key[1], wa, wb)
            assert self._gemm_images[wname][0] == key[1], f"weights image {wname!r} of other weights"
            wa, wb = self._gemm_images[wname][1:]
            return {r: g.row_ops(addr + r * D.Q, wa + r * wb) for r in range(ROWS)}
        per = D.Q + wb  # (per row its command, then its weights: one B run)
        if self._gemm_images.get(name, (None,))[0] == key:  # (a layer's bands: its image again, not converted again)
            addr = self._gemm_images[name][1]
        elif self.reuse and self.npu.has(name):
            addr = self.npu.address(name)
        else:
            assert name not in self._gemm_images, f"DDR image {name!r} written again with other data"
            addr = self.ddr(name, ROWS * per)
            for r in range(ROWS):
                self.put(name, cmds[r], r * per)
            if w is not None:
                self.slots.add(g, w, {r: (name, r * per + D.Q) for r in range(ROWS)})
            self._gemm_images[name] = (key, addr, per)
        return {r: g.row_ops(addr + r * per) for r in range(ROWS)}

    def _time_bias(self, g: Gemm, b: np.ndarray, rows: dict[int, RowOps], resnet: str) -> None:
        """The rows' first run in three: the command's header, the biases from TBIAS (rewritten per step), the rest
        (``tb_hooks``)."""
        nb, hb = 160 * g.npo, 4 * K.C_BIAS
        if not self.tb_used:
            self.ddr(TBIAS, TBIAS_BYTES)
        off = self.tb_used
        self.tb_used += ROWS * nb
        limit("the time-biased convs' biases", self.tb_used, TBIAS_BYTES, "the TBIAS image")
        tb = self.npu.address(TBIAS)
        for r, ro in rows.items():
            (a, n), *more = ro.runs  # (more: the weights again per further tile)
            self.npu.write(TBIAS, g.bias_bytes(b, r), off + r * nb)
            ro.runs[:] = [(a, hb), (tb + off + r * nb, nb), (a + hb + nb, n - hb - nb), *more]
        self.tb_hooks.append(TimeBias(resnet, off, np.concatenate([g.weight_rows(r) for r in range(ROWS)])))

    def records(self, rec: int, *, name: str, ch_off: int, nr: int, gsize: int, band: int = 0, nbands: int = 1) -> Records:
        """Drain a producer's GroupNorm records (memtile ``rec`` [core][OCH], 4 OCH bytes: released by the drain's op)
        to DDR, with the next op; the consumer's B stream reads them after ``Seq.ddr_sync`` (``apply`` syncs). A layer's
        bands' records in one DDR image [band][col][core][256 B] (a band the current one holds already opens a new
        one: ``name``, numbered from its second), so that a consumer gathers a producer's strips of every band in one
        shim BD."""
        img, held = self._rec_img.get(name, (0, {band}))
        if band in held:
            k = self._rec_names[name] = self._rec_names.get(name, -1) + 1
            img, held = self.ddr(f"{name}.{k}" if k else name, nbands * COLS * ROWS * 256), set()
        held.add(band)
        self._rec_img[name] = img, held
        base = img + band * COLS * ROWS * 256
        cols = {c: ColOps(drains=[_transfer(base, c, 1024, 0, rec, 1024, [(64, 1), (ROWS, D.OCH // 4)], None)])
                for c in range(COLS)}
        self._drains.append((cols, [(rec, 4 * D.OCH)], True))
        return Records(base, ch_off, nr, gsize)

    def apply(self, a: Apply, gamma: np.ndarray | None, beta: np.ndarray | None, *, name: str,
              records: list[Records] = (), extra: dict[int, ColOps] | None = None) -> None:
        """An apply op (GroupNorm: its cores' records gathered from ``records``, after a DDR sync)."""
        runs, cmds, gbs, slots = {}, [], [], {}
        for i, r in enumerate(a.rows):
            rr, nrec = [], 0
            if a.gn:  # (per producer rows' gather: strips of their records in every column)
                c0, n = a.ch0(r), a.nreal or a.nblk * 80
                for rs in sorted(records, key=lambda rs: -len(rs.rows_for(c0, n))):  # (the larger strips first: fewer
                    # gathers cut at slot ends)
                    sel = rs.rows_for(c0, n)
                    assert sel == list(range(sel[0], sel[-1] + 1)) if sel else True, sel
                    for col in range(COLS) if sel else ():
                        rr.append((rs.addr + col * 1024 + sel[0] * 256, len(sel) * 256))
                assert rr, (c0, n)
                rr, ns = self._slots(rr)
                nrec = ns * K.REC_SLOT  # (whole slots: their zero records add nothing)
            runs[r] = rr
            slots[r] = -(-nrec // K.REC_SLOT)
            cmds.append(a.command(r, nrec))
            gbs.append(a.gamma_beta(gamma, beta, r))
        sync = a.gn and any(r for _, _, r in self._drains)  # (records drained with this op: synced before its B runs)
        if a.gn and not sync:
            self.ddr_sync()
        addr = self.image(name, np.concatenate([x for i in range(len(a.rows)) for x in (cmds[i], gbs[i])]))
        rows = {r: a.row_ops(addr + 2 * i * D.Q, runs[r], slots[r]) for i, r in enumerate(a.rows)}
        co = a.col_ops()
        cols = {c: co for c in range(COLS)}
        self.op(self.merge(cols, extra) if extra else cols, rows, sync=sync)

    def _slots(self, strips: list[tuple[int, int]]) -> tuple[list[tuple], int]:
        """Record strips (device address, bytes) packed whole into B slots of ``core.REC_SLOT`` records (each slot
        filled up with zero records, then padded to Q), runs of equal strips of the columns (1 KB apart) and bands (8
        KB) merged into one shim gather. Returns the B runs and the slots."""
        cap = K.REC_SLOT * 256
        slots, used = [[]], 0
        for a, n in strips:
            assert n <= cap
            if used + n > cap:
                slots.append([])
                used = 0
            slots[-1].append((a, n))
            used += n
        out = []
        for sl in slots:
            i = 0
            while i < len(sl):
                a, n = sl[i]
                j = i + 1
                while j < len(sl) and sl[j][1] == n and sl[j][0] == a + 1024 * ((j - i) % COLS) + 8192 * ((j - i) // COLS):
                    j += 1
                if j - i > COLS:  # (whole groups of the columns' strips: a shim BD's length)
                    j = i + (j - i) // COLS * COLS
                out.append((a, n) if j == i + 1 else (a, n * (j - i), (n // 4, min(j - i, COLS), 256, 2048)))
                i = j
            used = sum(n for _, n in sl)
            if used < D.Q:
                out.append((self.zeros_q(), D.Q - used))
        return out, len(slots)

    def zeros(self, nbytes: int) -> int:
        """The device address of nbytes of zeros in DDR (one image of the largest size asked)."""
        self.npu.alloc("_zddr", nbytes)
        return self.npu.address("_zddr")

    def zeros_q(self) -> int:
        if not self.npu.has("_zq"):
            self.image("_zq", np.zeros(D.Q, np.uint8))
        return self.npu.address("_zq")

    def finish(self) -> np.ndarray:
        """The dispatch's transaction words (``seq.Seq.finish``), the drains still pending pushed first; the weight jobs
        still pending packed."""
        if self._drains:
            self.op({}, {})
        self.slots.pack(self.npu)
        return self.s.finish()

    def transaction(self):
        return Transaction(self.npu.dev, self.finish())
