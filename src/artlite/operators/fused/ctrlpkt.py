"""A library configuration's tile writes as control packets (``Loader``): the shim DMA streams each tile's writes,
packets of up to 4 words with their target address, to the tile's TileControl port, instead of the firmware executing
them one by one (a blockwrite costs ~1.5 us per KB, any operation ~0.1 us: a Llama 1B chunk of 128 tokens wrote 6.7 MB
of program memory in 10 of its 43 ms, and ran ~850 operations per GEMM configuration).

``split`` takes out of a configuration (after its PDI load: the array reset) per core and memory tile its writes to
program memory, data memory (static tables: a GDN operator's SiLU table copies, 100 KB on 32 cores, took ~150 us of
blockwrites), DMA BDs, DMA channels (start queues included), locks and the core's control but its enable, in their
order (masked writes as the values they leave); the transaction keeps the stream switches, the shim, the cores'
enables, masked writes to memory and the contents of the marker's word (written after the poll). The DMA channels then
start before the stream switches are configured: every task waits for a lock or for data, and none comes before the
cores run and the program's shim tasks.

The loader routes, per column with packets, the shim's MM2S channel 0 (its mux South 3) up the column: circuit through
the shim (North 0), at each tile on the way a split in packet mode (its own packets, its controller id as mlir-aie
assigns them, to its TileControl port, the rest North 0; the memtile in circuit mode when it takes none). Each tile's
packets are a BD of one chained shim task per column (TLAST between tiles), the column's last tile's end with a write of
``MARK_VALUE`` to a memory word (cleared before) the transaction polls: a stream delivers its packets in order, so all
of the column's writes are done before anything else of the configuration runs. The configuration's own stream switch
writes then zero the loader's ports (``prune``'s merged port blockwrites; the ones it does not cover in the
prelude); its slots stay, inert unless the design puts that slave port in packet mode (then zeroed too). A tile's memory
holds nothing of the design before its configuration: the marker costs none.

A configuration takes the loader where its estimated cost (``worth``: the measured costs of a transaction's
operations) is below the transaction's for the same writes. ``ARTLITE_PM_PACKETS=0`` keeps every write in the
transaction.

On an array the previous design leaves running (``reused``: a configuration without the reset, ``program``), each
column's packets go as two tasks: first every tile of the column quiesced (``_quiesce``: its core held in reset, its
DMA channels held in reset while every lock is written 63, which grants any acquire a core or channel left pending -
the reset of either does not cancel it, and it would take the next design's first release of that lock - then 0, the
channels let go; BDs ``QUIESCE_BD0`` on), then the design's packets as above; the route writes each tile's whole master
range (no port the previous design enabled fans the packets out), and the slots of every slave the design puts in
packet mode are zeroed.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ... import txn
from ...npu_regs import (
    CORE_CONTROL_RESET,
    CORE_TILE,
    MEM_TILE,
    SHIM_MUX,
    SHIM_MUX_SOUTH3_DMA,
    SHIM_SLAVE_SOUTH3,
    SHIM_TILE,
    SWITCH_ENABLE,
    SWITCH_PACKET,
    tile_map,
)

ROW_ID = {1: 26, 2: 27, 3: 29, 4: 30, 5: 31}  # (a tile's controller id per row: its packets' stream id)
# (the memory word the column's packets end with, data memory's last: memtile, core tile)
MARK = {1: MEM_TILE.data_bytes - 4, 2: CORE_TILE.data_bytes - 4}
MARK_VALUE = 0x5A5A0001
BD0 = 11  # (the shim BDs of a column's chained task, one per tile: free during a configuration)
QUIESCE_BD0 = 6  # (``reused``: the shim BDs of a column's quiescing task)
# a transaction's operations (s): a single write, a blockwrite (and per byte), a poll that passes; packets per byte
# through a column's stream (measured on a Llama 1B chunk: 129 configurations)
WRITE_S, BLOCKWRITE_S, BLOCK_BYTE_S, POLL_S, STREAM_BYTE_S = 0.133e-6, 0.101e-6, 1.5e-9, 0.075e-6, 0.25e-9

# (the registers a tile's packets take, per row kind: core tile - program memory, DMA BDs, DMA channels, locks, core
# control, data memory; memtile - DMA BDs and channels, locks, data memory)
_MOVED = {2: (CORE_TILE.program, CORE_TILE.bds, CORE_TILE.dma, CORE_TILE.locks, CORE_TILE.control,
              (0, CORE_TILE.data_bytes)),
          1: ((MEM_TILE.bds[0], MEM_TILE.dma[1]), MEM_TILE.locks, (0, MEM_TILE.data_bytes))}
_DATA_END = {2: CORE_TILE.data_bytes, 1: MEM_TILE.data_bytes}  # (a tile's data memory: from 0)
_CORE_CONTROL = CORE_TILE.control[0]
_MEM_SOUTH0 = 7  # (the memtile's slave port number of South 0)
_SHIM_SOUTH3 = 5  # (the shim's slave port number of South 3)
LOCKS = {2: (CORE_TILE.locks[0], 16), 1: (MEM_TILE.locks[0], 64)}  # (lock values, one per 16 B: first, count)
LOCK_FLUSH = 63  # (a lock value that grants any pending acquire)


def enabled() -> bool:
    """Whether a library configuration's writes may go as control packets (``ARTLITE_PM_PACKETS=0``: none), read at each
    assembly."""
    return os.environ.get("ARTLITE_PM_PACKETS", "1") != "0"


Writes = dict[tuple[int, int], list[tuple[int, np.ndarray]]]  # ((col, row) -> [(tile-local address, words)])


def parity(word: int) -> int:
    """The word with its odd parity bit (31)."""
    return word | (0 if word.bit_count() % 2 else 1 << 31)


def slot(stream_id: int, arbiter: int, mask: int = 0x1F) -> int:
    """A slave port slot matching the stream ids equal to ``stream_id`` in the bits ``mask`` to an arbiter, msel 0."""
    return (stream_id << 24) | (mask << 16) | (1 << 8) | arbiter


def _north(own: int, up: list[int]) -> list[int]:
    """Slots (arbiter 1) matching the ids ``up`` but not ``own``: per bit, chosen greedily, the ids that differ from own
    in it (a slave port has 4 slots: a memtile's own and 4 ids above would not fit one each)."""
    out, left = [], set(up)
    while left:
        b = max(range(5), key=lambda b: sum(((i ^ own) >> b) & 1 for i in left))
        out.append(slot((~own >> b & 1) << b, 1, 1 << b))
        left = {i for i in left if not ((i ^ own) >> b) & 1}
    return out


def _parity_v(words: np.ndarray) -> np.ndarray:
    """``parity`` of each word of an array."""
    w = np.asarray(words, np.uint32)
    c = w ^ (w >> 16)
    c ^= c >> 8
    c ^= c >> 4
    c ^= c >> 2
    c ^= c >> 1
    return w | ((~c & 1) << 31).astype(np.uint32)


def packets(stream_id: int, writes: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """A tile's writes ((tile-local address, words)) as control packets: per packet the stream header (``stream_id``:
    the route's packet id), the control header (write, words - 1, address), up to 4 words (mlir-aie's encoding) in one
    16-byte line (words past its end wrap to its start)."""
    head = parity(stream_id)
    out = []
    for addr, data in writes:
        d = np.asarray(data, np.uint32)
        if k := min(d.size, -addr // 4 % 4):  # (the words up to the next line)
            out.append(np.array([head, parity(((k - 1) << 20) | (addr & 0xFFFFF)), *d[:k].tolist()], np.uint32))
            addr, d = addr + 4 * k, d[k:]
        q, r = divmod(d.size, 4)
        if q:  # (whole packets of 4 words)
            blk = np.empty((q, 6), np.uint32)
            blk[:, 0] = head
            blk[:, 1] = _parity_v(3 << 20 | (addr + 16 * np.arange(q, dtype=np.int64)) & 0xFFFFF)
            blk[:, 2:] = d[:4 * q].reshape(q, 4)
            out.append(blk.reshape(-1))
        if r:
            out.append(np.array([head, parity(((r - 1) << 20) | ((addr + 16 * q) & 0xFFFFF)), *d[4 * q:].tolist()],
                                np.uint32))
    return np.concatenate(out) if out else np.zeros(0, np.uint32)


def split(cfg: np.ndarray) -> tuple[np.ndarray, Writes]:
    """A configuration (after its PDI load) without the writes control packets take (per core and memory tile its
    program memory, data memory but the marker's word, DMA BDs, DMA channels, locks, the core's control but its enable:
    writes, blockwrites, masked writes as the values they leave; not masked writes to data memory, whose old values are
    unknown), and those per tile in their order."""
    rest, moved = [], defaultdict(list)
    value, big = {}, []  # (the moved writes: per word of short ones (seq, value); long ones (seq, address, words))

    def last(reg: int, default: int) -> int:
        """The value the moved writes left at reg (a masked write's old value)."""
        seq, v = value.get(reg, (-1, default))
        for q, r, words in reversed(big):
            if q < seq:
                break
            if r <= reg < r + 4 * len(words):
                return int(words[(reg - r) // 4])
        return v

    ops = txn.op_offsets(cfg)
    for seq, ((p, op), e) in enumerate(zip(ops, [q for q, _ in ops[1:]] + [len(cfg)])):
        reg = int(cfg[p + 2]) if op in (txn.OP_WRITE, txn.OP_BLOCKWRITE, txn.OP_MASKWRITE) else 0
        row, off = min((reg >> 20) & 0x1F, 2), reg & 0xFFFFF
        n = (e - p - 4) if op == txn.OP_BLOCKWRITE else 1
        if not row or not any(a <= off and off + 4 * n <= b for a, b in _MOVED[row]) or \
                (op == txn.OP_MASKWRITE and row == 2 and off == _CORE_CONTROL and int(cfg[p + 4]) & int(cfg[p + 5]) & 1) or \
                (op == txn.OP_MASKWRITE and off < _DATA_END[row]):
            rest.append(cfg[p:e])  # (not a tile's, the core's enable, or a masked write to memory)
            continue
        if op == txn.OP_MASKWRITE:
            v, m = int(cfg[p + 4]), int(cfg[p + 5])
            old = last(reg, CORE_CONTROL_RESET if row == 2 and off == _CORE_CONTROL else 0)
            words = np.array([(old & ~m) | (v & m)], np.uint32)
        else:
            words = cfg[p + 4:e] if op == txn.OP_BLOCKWRITE else cfg[p + 4:p + 5]
        if off <= MARK[row] < off + 4 * len(words):  # (the marker's word, data memory's last: after the poll)
            rest.append(np.array([txn.OP_WRITE, 0, (reg & ~0xFFFFF) | MARK[row], 0, words[-1], 24], np.uint32))
            words = words[:-1]
            if not len(words):
                continue
        if len(words) > 16:  # (program and data memory)
            big.append((seq, reg, words))
        else:
            for i, w in enumerate(words):
                value[reg + 4 * i] = (seq, int(w))
        moved[(reg >> 25, (reg >> 20) & 0x1F)].append((off, words))
    return (np.concatenate(rest) if rest else np.zeros(0, np.uint32)), dict(moved)


@dataclass
class Loader:
    """The control packets of a configuration's writes (``split``'s) and the transaction around them."""

    writes: Writes
    reused: bool = False  # (the array not reset since the previous design: each tile quiesced first, whole ranges)
    columns: dict[int, list[int]] = field(init=False)  # (column -> its rows with packets, ascending)
    streams: dict[tuple[int, int], np.ndarray] = field(init=False)  # ((col, row) -> its packets)
    quiesce: dict[tuple[int, int], np.ndarray] = field(init=False)  # ((col, row) -> its quiescing packets: reused)

    def __post_init__(self) -> None:
        cols = defaultdict(list)
        for c, r in sorted(self.writes):
            cols[c].append(r)
        self.columns = dict(cols)
        self.streams = {}
        for c, rows in self.columns.items():
            for r in rows:
                w = self.writes[(c, r)]
                if r == rows[-1]:  # (the column's last: its marker)
                    w = [*w, (MARK[min(r, 2)], np.array([MARK_VALUE], np.uint32))]
                self.streams[(c, r)] = packets(ROW_ID[r], w)
        self.quiesce = {(c, r): packets(ROW_ID[r], _quiesce(r)) for c, rows in self.columns.items() for r in rows
                        if self.reused}

    def routes(self) -> list[tuple[int, list[int], str]]:
        """The loader's writes (absolute address, words, kind: "mux", "ports", "slots"): per column the shim mux and
        the circuit through the shim; at each tile up to the column's top one with packets its slave South 0 in packet
        mode, slots (its own id to arbiter 0, the ids above to arbiter 1: ``_north``) and masters (TileControl on arbiter 0, North
        0 on arbiter 1; the ports between them 0, as after the reset; ``reused``: the tile's whole master range), the
        memtile in circuit mode without packets."""
        a, out = txn.tile_address, []
        for c, rows in self.columns.items():
            out += [(a(c, 0, SHIM_MUX), [SHIM_MUX_SOUTH3_DMA], "mux"),
                    (a(c, 0, SHIM_SLAVE_SOUTH3), [SWITCH_ENABLE], "ports"),
                    self._masters(c, 0, {SHIM_TILE.north0_master: SWITCH_ENABLE | _SHIM_SOUTH3})]
            for r in range(1, rows[-1] + 1):
                t = tile_map(r)
                slave, slots0, ctrl, north = t.south0_slave, t.south0_slots, t.tile_ctrl_master, t.north0_master
                if r == 1 and r not in rows:
                    out += [(a(c, 1, slave), [SWITCH_ENABLE], "ports"),
                            self._masters(c, 1, {north: SWITCH_ENABLE | _MEM_SOUTH0})]
                    continue
                up = [ROW_ID[q] for q in rows if q > r]
                slots = ([slot(ROW_ID[r], 0)] if r in rows else []) + _north(ROW_ID[r], up)
                masters = {}
                if r in rows:
                    masters[ctrl] = SWITCH_ENABLE | SWITCH_PACKET | (1 << 3)  # (arbiter 0, msel 0; the header kept)
                if up:
                    masters[north] = SWITCH_ENABLE | SWITCH_PACKET | (1 << 3) | 1
                out += [(a(c, r, slave), [SWITCH_ENABLE | SWITCH_PACKET], "ports"), (a(c, r, slots0), slots, "slots"),
                        self._masters(c, r, masters)]
        return out

    def _masters(self, c: int, r: int, values: dict[int, int]) -> tuple[int, list[int], str]:
        """A route tile's master port writes: from the first of ``values`` (tile-local register -> value) to the last,
        the ports between 0 (``reused``: the whole master range)."""
        masters = tile_map(r).masters
        lo, hi = (masters[0], masters[1] - 4) if self.reused else (min(values), max(values))
        return txn.tile_address(c, r, lo), [values.get(reg, 0) for reg in range(lo, hi + 4, 4)], "ports"

    def ports(self) -> list[int]:
        """The stream switch port registers the routes enable (``prune`` zeroes them)."""
        return [addr + 4 * i for addr, words, kind in self.routes() if kind == "ports" for i, w in enumerate(words) if w]

    def slots(self, rest: np.ndarray) -> list[int]:
        """The slot registers to zero: a tile's whose slave South 0 the design (``rest``) puts in packet mode
        (``reused``: every slot of each slave port it puts in packet mode)."""
        packet = set()
        for p, op in txn.op_offsets(rest):
            if op not in (txn.OP_WRITE, txn.OP_MASKWRITE, txn.OP_BLOCKWRITE):
                continue
            reg = int(rest[p + 2])
            row = min(txn.address_row(reg), 2)
            n = int(rest[p + 3]) // 4 - 4 if op == txn.OP_BLOCKWRITE else 1
            s0, s1 = tile_map(row).slaves
            for i, v in enumerate(rest[p + 4:p + 4 + n]):
                off = (reg & 0xFFFFF) + 4 * i
                south0 = row and off == tile_map(row).south0_slave
                if int(v) & SWITCH_PACKET and (s0 <= off < s1 if self.reused else south0):
                    packet.add((reg & ~0xFFFFF, (off - s0) // 4))
        if self.reused:
            return sorted(t | (tile_map((t >> 20) & 0x1F).slot0 + 16 * k + 4 * i) for t, k in packet for i in range(4))
        tiles = {t for t, _ in packet}
        return [addr + 4 * i for addr, words, kind in self.routes()
                if kind == "slots" and addr & ~0xFFFFF in tiles for i in range(len(words))]

    def prelude(self, zeros: list[int]) -> tuple[np.ndarray, int, list[tuple[int, np.ndarray]]]:
        """The transaction between the PDI load and the rest of the configuration: the routes, the columns' markers
        cleared, per column a chained task of its tiles' packets (addresses 0: ``place``; ``reused``: after one of their
        quiescing packets), the markers polled, then the registers ``zeros`` (the loader's ports the rest does not zero,
        slots) written 0. Returns its words (no header), operation count and per BD the word of its address and its
        packets."""
        t, a = txn.Txn(), txn.tile_address
        for addr, words, _ in self.routes():
            t.blockwrite(addr, words) if len(words) > 1 else t.write32(addr, words[0])
        for c, rows in self.columns.items():
            t.write32(a(c, rows[-1], MARK[min(rows[-1], 2)]), 0)
        at = []
        for c, rows in self.columns.items():
            tasks = ((QUIESCE_BD0, self.quiesce), (BD0, self.streams)) if self.reused else ((BD0, self.streams),)
            for bd0, streams in tasks:
                for k, r in enumerate(rows):
                    bd, last = bd0 + k, k == len(rows) - 1
                    w = t.blockwrite(a(c, 0, txn.SHIM_BD0 + txn.BD_BYTES * bd),
                                     [*txn.bd_words(0, 4 * len(streams[(c, r)])), 0, 3 << 30, 2 << 24, 0,
                                      (1 << 25) | (0 if last else (1 << 26) | ((bd + 1) << 27))])
                    at.append((w + 1, streams[(c, r)]))  # (its address low word: the blockwrite's data from w + 4)
                t.shim_push(c, txn.MM2S, 0, bd0)
        for c, rows in self.columns.items():
            t.maskpoll32(a(c, rows[-1], MARK[min(rows[-1], 2)]), MARK_VALUE, 0xFFFFFFFF)
        for lo, hi in _runs(sorted(zeros)):
            t.blockwrite(lo, [0] * ((hi - lo) // 4 + 1)) if hi > lo else t.write32(lo, 0)
        return t.words()[4:], t.n_ops, at


def _quiesce(row: int) -> list[tuple[int, np.ndarray]]:
    """A reused tile's first writes (``Loader(reused=True)``): its core in reset, its DMA channels in reset while every
    lock is written ``LOCK_FLUSH`` (granting any pending acquire) and then 0, the channels let go."""
    k = min(row, 2)
    base, n = LOCKS[k]
    dma_ctrl = tile_map(k).channel_ctrl
    one = lambda v: np.array([v], np.uint32)
    return ([(_CORE_CONTROL, one(CORE_CONTROL_RESET))] if k == 2 else []) + [(r, one(2)) for r in dma_ctrl] + \
        [(base + 16 * i, one(v)) for v in (LOCK_FLUSH, 0) for i in range(n)] + [(r, one(0)) for r in dma_ctrl]


def _runs(regs: list[int]) -> list[tuple[int, int]]:
    """Consecutive registers (4 bytes apart) as (first, last) runs."""
    out: list[tuple[int, int]] = []
    for r in regs:
        if out and r == out[-1][1] + 4:
            out[-1] = (out[-1][0], r)
        else:
            out.append((r, r))
    return out


def worth(loader: Loader) -> bool:
    """Whether the loader costs less than the transaction's operations for the same writes (by the measured costs)."""
    writes = [d for w in loader.writes.values() for _, d in w]
    by_txn = sum(WRITE_S if len(d) == 1 else BLOCKWRITE_S + 4 * len(d) * BLOCK_BYTE_S for d in writes)
    by_packets = (len(loader.routes()) * (BLOCKWRITE_S + WRITE_S) / 2
                  + len(loader.streams) * BLOCKWRITE_S + len(loader.columns) * (2 * WRITE_S + POLL_S)
                  + max(sum(4 * len(loader.streams[(c, r)]) for r in rows) for c, rows in loader.columns.items())
                  * STREAM_BYTE_S)
    return by_packets < by_txn
