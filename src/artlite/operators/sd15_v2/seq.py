"""Host-built transactions of sd15_v2's design (``design``): ops in program order, handed over by locks.

Per column an op is a set of memtile tasks: receives (S2MM r <- core r's outputs), sends (MM2S r -> core r), fills
(DDR -> S2MM 4) and drains (MM2S 4 -> DDR); per row a run of B-stream reads (shim -> the row's cores). The handover:
- every producer task (receive, fill, drain) releases the column's ready lock X by 1 when it completes;
- op n's gate (one BD on MM2S 5, looped back to the column's own S2MM 5) acquires X >= the producers of op n - 1, then
  releases op n's lock G[n % R] by the executions of op n's consumers (sends, fills, drains: one acquire of G per
  execution). One release has one consumer (a lock shared by tasks of two ops on different channels deadlocks: the
  later one takes the earlier one's units); within an op the consumers share G (each takes exactly its count, so any
  order completes), and R exceeds the ops in flight;
- so the command processor never waits for an op: only for queue depth (4 tasks per channel: before a fifth, the op
  of the oldest must be complete), cross-column data (``barrier``: the previous op complete in every column, e.g.
  conv halo rows from the neighbours, DDR round trips) and the end. "Op u complete everywhere" = the gate of op u + 1
  fired in every column (its producers done; its sends, B reads and older ops before them): one sync of the 8
  columns' gate tokens (~0.36 us per token).
- an op's command is in B slot 0, so that its k-tile i is in B slot (i + 1) % 2 (a k-tile of an A object in region 2
  + i % 2: ``design.R23``): per row the host tracks the B objects and puts a NOP command before an op (``RowOps.parity``)
  when they are odd (and at the end: the cores' positions persist across dispatches).
- core roles (``core.ROLES``): an op's role commands on a row (``RowOps.role``) need the row's cores in that role;
  ``role`` switches rows (their program memory rewritten while they wait for a command), the end restores the
  design's (``design.ROLE``).
"""

import collections
from dataclasses import dataclass, field

import numpy as np

from ... import txn
from ..design import COLS, ROWS
from . import design as D

R_LOCKS = 6  # rotating gate locks G (> ops in flight)
LOCK_X, LOCK_G0 = 0, 1  # memtile lock ids
SHIM_BD_BASE = {(txn.MM2S, 0): 0, (txn.S2MM, 0): 8}  # (4 BDs each; the B streams' MM2S 1: B_BDS)
B_BDS = (4, 5, 6, 7, 12, 13, 14, 15)  # (MM2S 1: a pool for the B runs' chains, the role switches', S2MM 1 unused)


@dataclass
class MemTask:
    """A memtile task: its chain of BDs (``txn.mem_bd_words`` keywords without locks and links: the first BD takes
    the task's acquire, the last its release), runs; a fill's / drain's DDR side (device address, bytes per run, shim
    dims[, shim iteration (step words, count)]: the shim task runs as often)."""

    chain: list[dict]
    repeat: int = 1
    ddr: tuple | None = None


@dataclass
class ColOps:
    """An op's tasks in one column: sends / receives by core row (in order), fills, drains."""

    sends: dict[int, list[MemTask]] = field(default_factory=dict)
    recvs: dict[int, list[MemTask]] = field(default_factory=dict)
    fills: list[MemTask] = field(default_factory=list)
    drains: list[MemTask] = field(default_factory=list)


@dataclass
class RowOps:
    """An op's B stream in one row: (device address, bytes[, shim dims]) runs, the B objects each of its cores takes;
    parity: the op's command must be in B slot 0 (its k-tiles' B slots matter: m-64 GEMMs, convs and attention); role:
    the role whose code its command runs (None: shared code)."""

    runs: list[tuple]
    b_objects: int
    parity: bool = True
    role: str | None = None


class Seq:
    """A dispatch's transaction. ``nop``: the device address of a NOP command (one B slot); ctrl: per (role, core row)
    the device address and bytes of its switch's control packets (``runtime.role_packets``)."""

    def __init__(self, *, nop: int, ctrl: dict | None = None):
        self.t = txn.Txn()
        self.nop = nop
        self.ctrl = ctrl or {}
        self.role_of = list(D.ROLE)  # (per core row its role)
        self._row_op = [-1] * ROWS  # (per core row the last op with commands there)
        self.n = 0  # ops pushed
        self.frontier = 0  # every task of the ops before it is complete
        self._q: dict[tuple, collections.deque] = {}  # (row 0 / 1, col, dir, ch) -> ops of its pending tasks
        self._k: dict[tuple, int] = {}  # tasks per shim channel (BD rotation)
        self._kbd: dict[tuple, int] = {}  # BD rotation per shim channel (the B runs)
        self._chains: dict[tuple, dict] = {}  # per memtile channel: chain -> (its BD ids, their words) of the chains its BDs hold
        self._lru = [[-1] * txn.MEM_BDS for _ in range(COLS)]  # (per column, per memtile BD: its last use)
        self._tick = 0
        self._held: list[list[list[int] | None]] = [[None] * txn.MEM_BDS for _ in range(COLS)]  # (its words)
        self._user: list[list[int | None]] = [[None] * txn.MEM_BDS for _ in range(COLS)]  # (the op of its last task)
        self._b_user: dict[tuple[int, int], int] = {}  # (col, shim BD of the B runs) -> the op of its last task
        self._gates: dict[int, list[int]] = {}  # op -> per column its gate's push (for tokens)
        self._prod = [0] * COLS  # producers pushed since the column's last gate
        self._par = [0] * ROWS  # per row: B objects, mod 2
        self._drains: dict[int, int] = {}  # col -> its last shim drain's push (waited for at the end)
        for c in range(COLS):
            for d in (txn.S2MM, txn.MM2S):
                for ch in range(6):
                    self.t.mem_token_controller(c, d, ch)
                for ch in range(2):
                    self.t.token_controller(c, d, ch)
            for lk in range(LOCK_X, LOCK_G0 + R_LOCKS):  # (the handover locks: not the design's, no reset of their own)
                self.t.lock_set(c, 1, lk, 0)

    # -- completion
    def complete(self, u: int) -> None:
        """Wait until every task of op u (and earlier) is complete in every column."""
        if u < self.frontier:
            return
        assert u + 1 < self.n, f"op {u} has no later gate"
        for c, op in enumerate(self._gates[u + 1]):
            self.t.set_token(op)
        self.t.sync(col=0, direction=txn.MM2S, channel=5, row=1, ncol=COLS)
        self.frontier = u + 1

    def _slot(self, key: tuple, op: int) -> int:
        """A queue slot on a channel for a task of ``op`` (before a fifth pending task: the oldest's op complete, or
        the oldest itself when it is of this op: its own token); returns the channel's task count (shim BD
        rotation). The caller records the push (``_pushed``)."""
        q = self._q.setdefault(key, collections.deque())
        while q and q[0][0] < self.frontier:
            q.popleft()
        while len(q) >= txn.QUEUE_DEPTH:
            u, push = q.popleft()
            if u < op:
                self.complete(u)
            else:
                row, col, d, ch = key
                self.t.set_token(push)
                self.t.sync(col=col, direction=d, channel=ch, row=row)
        q.append((op, -1))
        k = self._k.get(key, 0)
        self._k[key] = k + 1
        return k

    def _pushed(self, key: tuple, push: int) -> None:
        q = self._q[key]
        q[-1] = (q[-1][0], push)

    def _task(self, col: int, d: int, ch: int, op: int, chain: list[list[int]], ck: tuple, repeat: int) -> int:
        """A memtile task: its chain on BDs of the channel's pool (``design.mem_bd_pool``), linked: a chain the channel ran
        before reuses its BDs while they still hold it, else the pool's least recently used BDs (rewritten: their last
        user complete; never one of this op's; ck: the chain as a key). Returns the push's operation number."""
        key = (1, col, d, ch)
        self._slot(key, op)

        def linked(ids: list[int]) -> list[list[int]]:
            out = []
            for i, words in enumerate(chain):
                words = list(words)
                if i + 1 < len(chain):
                    words[1] |= 1 << 19 | ids[i + 1] << 20
                out.append(words)
            return out
        cache = self._chains.setdefault(key, {})
        hit = cache.get(ck)
        users, held, lru = self._user[col], self._held[col], self._lru[col]
        if hit is not None and all(held[b] == w for b, w in zip(*hit, strict=True)):
            ids = hit[0]
        else:
            pool = D.mem_bd_pool(ch)
            free = sorted((b for b in pool if users[b] != op), key=lru.__getitem__)
            assert len(free) >= len(chain), f"op {op}: more BDs on memtile channel {(d, ch)} than its pool has free"
            ids = free[:len(chain)]
            words_l = linked(ids)
            for bid, words in zip(ids, words_l, strict=True):
                if held[bid] != words:
                    if (u := users[bid]) is not None:
                        self.complete(u)
                    self.t.mem_bd(col, bid, words)
                    held[bid] = words
            for k in [k for k, (i, _) in cache.items() if not set(i).isdisjoint(ids)]:
                del cache[k]
            cache[ck] = (ids, words_l)
        for bid in ids:
            users[bid] = op
            self._tick += 1
            lru[bid] = self._tick
        push = self.t.mem_push(col, d, ch, ids[0], repeat=repeat)
        self._pushed(key, push)
        return push

    def _gate(self, col: int, op: int, words: list[int]) -> int:
        key = (1, col, txn.MM2S, 5)
        self._slot(key, op)
        bid = D.GATE_BDS[op % len(D.GATE_BDS)]
        if self._held[col][bid] != words:
            if (u := self._user[col][bid]) is not None:
                self.complete(u)
            self.t.mem_bd(col, bid, words)
            self._held[col][bid] = words
        self._user[col][bid] = op
        push = self.t.mem_push(col, txn.MM2S, 5, bid)
        self._pushed(key, push)
        return push

    def _b_chain(self, col: int, op: int, runs: list[tuple]) -> None:
        """A row's B runs as tasks of chained BDs on shim MM2S 1 (BDs from ``B_BDS`` in turn; a BD rewritten only when
        its last user is complete): more runs than BDs go as several chains, each after the previous one completed
        (its slots are taken as they arrive: the wait cannot block the cores)."""
        key = (0, col, txn.MM2S, 1)
        for c0 in range(0, len(runs), len(B_BDS)):
            chunk = runs[c0:c0 + len(B_BDS)]
            if c0:  # (the previous chain complete: its BDs free)
                self.t.set_token(self._q[key][-1][1])
                self.t.sync(col=col, direction=txn.MM2S, channel=1)
                self._q[key].pop()
            self._slot(key, op)
            k = self._kbd.get(key, 0)
            ids = [B_BDS[(k + i) % len(B_BDS)] for i in range(len(chunk))]
            self._kbd[key] = k + len(chunk)
            for i, (bid, run) in enumerate(zip(ids, chunk, strict=True)):
                if (u := self._b_user.get((col, bid))) is not None and u != op:
                    self.complete(u)
                self.t.shim_bd(col, bid, address=run[0], nbytes=run[1], dims=run[2] if len(run) > 2 else None,
                               next_bd=ids[i + 1] if i + 1 < len(ids) else None)
                self._b_user[(col, bid)] = op
            n = self.t.n_ops
            self.t.shim_push(col, txn.MM2S, 1, ids[0])
            self._pushed(key, n)

    def _shim(self, col: int, d: int, ch: int, op: int, address: int, nbytes: int, dims: tuple | None,
              iteration: tuple[int, int] | None = None, *, repeat: int = 1) -> int:
        key = (0, col, d, ch)
        k = self._slot(key, op) % txn.QUEUE_DEPTH
        bd = SHIM_BD_BASE[(d, ch)] + k
        self.t.shim_bd(col, bd, address=address, nbytes=nbytes, dims=dims, iteration=iteration)
        n = self.t.n_ops
        self.t.shim_push(col, d, ch, bd, repeat=repeat)
        self._pushed(key, n)
        return n

    # -- ops
    def op(self, cols: dict[int, ColOps], rows: dict[int, RowOps], *, barrier: bool = False, nolock: bool = False,
           sync: bool = False) -> int:
        """Push an op; returns its index. barrier: its sends start only once the previous op is complete in every
        column (the command processor waits for its gates; the sends then take no lock: e.g. a conv's, more
        executions than a lock counts). nolock: its sends take no lock without that wait (inputs complete before an
        earlier op: a conv's later passes). sync: ``ddr_sync`` after its memtile tasks, before its B runs (the B runs
        read what its drains write)."""
        free_sends = barrier or nolock
        n = self.n
        self.n += 1
        g = self._gates[n] = []
        lock_g = txn.mem_lock(LOCK_G0 + n % R_LOCKS)
        x = txn.mem_lock(LOCK_X)
        for c in range(COLS):
            co = cols.get(c, ColOps())
            execs = (0 if free_sends else sum(t.repeat for ts in co.sends.values() for t in ts)) + \
                sum(t.repeat for t in co.fills + co.drains)
            assert execs < 64 and self._prod[c] < 64, (execs, self._prod[c])
            words = txn.mem_bd_words(addr=D.GATE_SCRATCH, nwords=1,
                                     acquire=(x, self._prod[c], True) if self._prod[c] else None,
                                     release=(lock_g, execs) if execs else None)
            g.append(self._gate(c, n, words))
            self._prod[c] = 0
        if barrier and n:
            self.complete(n - 1)

        memo: dict[tuple, tuple] = {}  # ((task, acquire, release) -> its BD words: a task object of several columns)

        def chain(t: MemTask, acquire: bool, release: bool) -> tuple:
            key = (id(t), acquire, release)
            if key not in memo:
                words = [txn.mem_bd_words(**bd, acquire=(lock_g, 1, True) if acquire and i == 0 else None,
                                          release=(x, 1) if release and i == len(t.chain) - 1 else None)
                         for i, bd in enumerate(t.chain)]
                memo[key] = (words, tuple(tuple(w) for w in words))
            return memo[key]

        for r, ro in rows.items():
            assert not ro.role or ro.role == self.role_of[r], f"op {n}: row {r} is in role {self.role_of[r]}, not {ro.role}"
            self._row_op[r] = n

        def b_runs() -> None:
            for r, ro in sorted(rows.items()):
                nop = self._par[r] and ro.parity
                runs = ([(self.nop, D.Q)] if nop else []) + list(ro.runs)
                self._b_chain(D.B_SHIM[r], n, runs)
                self._par[r] = (self._par[r] + nop + ro.b_objects) % 2

        # (per column its channels' task lists in program order; an op with more tasks on a channel than its queue
        # holds (a residual conv's passes: chunks, residual, ...) takes its B runs first and issues its tasks in rounds
        # over the columns and channels, so that a queue-depth wait only waits for work every core can do)
        lists = []
        for c in range(COLS):
            co = cols.get(c, ColOps())
            for r, ts in sorted(co.recvs.items()):
                lists.append([("recv", c, r, t) for t in ts])
            for r, ts in sorted(co.sends.items()):
                lists.append([("send", c, r, t) for t in ts])
            lists.append([("fill", c, 4, t) for t in co.fills])
            lists.append([("drain", c, 4, t) for t in co.drains])
        deep = max((len(ls) for ls in lists), default=0) > 2
        assert not (deep and sync)
        if deep:
            b_runs()
        order = [e for k in range(max((len(ls) for ls in lists), default=0)) for ls in lists if k < len(ls)
                 for e in [ls[k]]] if deep else [e for ls in lists for e in ls]
        for kind, c, ch, t in order:
            if kind == "recv":
                self._task(c, txn.S2MM, ch, n, *chain(t, False, True), t.repeat)
                self._prod[c] += t.repeat
            elif kind == "send":
                self._task(c, txn.MM2S, ch, n, *chain(t, not free_sends, False), t.repeat)
            elif kind == "fill":
                self._task(c, txn.S2MM, 4, n, *chain(t, True, True), t.repeat)
                self._prod[c] += t.repeat
                self._shim(c, txn.MM2S, 0, n, *t.ddr, repeat=t.repeat)
            else:
                self._task(c, txn.MM2S, 4, n, *chain(t, True, True), t.repeat)
                self._prod[c] += t.repeat
                self._drains[c] = self._shim(c, txn.S2MM, 0, n, *t.ddr, repeat=t.repeat)
        if sync:
            self.ddr_sync()
        if not deep:
            b_runs()
        return n

    def role(self, rows: tuple[int, ...], role: str) -> None:
        """Core rows ``rows`` in role ``role`` for the ops pushed from now: where a row is in another one, once every op
        with commands there is complete (an empty op pushed for its gate when the last is the newest: the cores wait for
        a command, their instruction fetch stalled; a program-memory write that meets a fetch fails, AM027), its cores'
        program memory from the shared code's end and their ext words rewritten by control packets: per row a BD of one
        task on shim ``design.CTRL_SHIM`` MM2S 1 (the row's broadcast), ending with the marker word ext[1] := 1, cleared
        before and polled in every core."""
        rows = tuple(r for r in rows if self.role_of[r] != role)
        if not rows:
            return
        u = max(self._row_op[r] for r in rows)
        if u >= self.frontier:
            if u + 1 >= self.n:
                self.op({}, {})
            self.complete(u)
        mark = [txn.tile_address(c, r + 2, D.L1["ext"] + 4) for r in rows for c in range(COLS)]
        for a in mark:
            self.t.write32(a, 0)
        for i, r in enumerate(rows):
            addr, nbytes = self.ctrl[(role, r)]
            self.t.shim_bd(D.CTRL_SHIM, B_BDS[i], address=addr, nbytes=nbytes,
                           next_bd=B_BDS[i + 1] if i + 1 < len(rows) else None)
        self.t.shim_push(D.CTRL_SHIM, txn.MM2S, 1, B_BDS[0])
        for a in mark:
            self.t.maskpoll32(a, 1, 0xFFFFFFFF)
        for r in rows:
            self.role_of[r] = role

    def ddr_sync(self) -> None:
        """Wait until every drain pushed so far has written DDR (before a fill or B stream reads what it wrote)."""
        for c, op in sorted(self._drains.items()):  # (a channel completes in order: its last drain covers the others)
            self.t.set_token(op)
            self.t.sync(col=c, direction=txn.S2MM, channel=0)
        self._drains.clear()

    def finish(self) -> np.ndarray:
        """The transaction: the cores' B parity and roles restored, every op complete, every drain written to DDR."""
        # (a last op: its gate makes the previous op's completion observable; the rows' parity NOPs)
        self.op({}, {r: RowOps(runs=[], b_objects=0) for r in range(ROWS) if self._par[r]})
        for role in sorted(set(D.ROLE)):
            self.role(tuple(r for r in range(ROWS) if D.ROLE[r] == role), role)
        self.complete(self.n - 2)
        self.ddr_sync()
        return self.t.words()
