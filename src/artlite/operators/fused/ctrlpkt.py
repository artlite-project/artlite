"""A library configuration's tile writes as control packets (``Loader``): the shim DMA streams each tile's writes,
packets of up to 4 words with their target address, to the tile's TileControl port, instead of the firmware executing
them one by one (a blockwrite costs ~1.5 us per KB, any operation ~0.1 us: a Llama 1B chunk of 128 tokens wrote 6.7 MB
of program memory in 10 of its 43 ms, and ran ~850 operations per GEMM configuration).

``split`` takes out of a configuration (after its PDI load: the array reset) per core and memory tile its writes to
program memory, DMA BDs, DMA channels (start queues included), locks and the core's control but its enable, in their
order (masked writes as the values they leave); the transaction keeps the stream switches, the shim, the cores'
enables and memory contents (a marker may share their word). The DMA channels then start before the stream switches
are configured: every task waits for a lock or for data, and none comes before the cores run and the program's shim
tasks.

The loader routes, per column with packets, the shim's MM2S channel 0 (its mux South 3) up the column: circuit through
the shim (North 0), at each tile on the way a split in packet mode (its own packets, its controller id as mlir-aie
assigns them, to its TileControl port, the rest North 0; the memtile in circuit mode when it takes none). Each tile's
packets are a BD of one chained shim task per column (TLAST between tiles), the column's last tile's end with a write of
``MARK_VALUE`` to a memory word (cleared before) the transaction polls: a stream delivers its packets in order, so all
of the column's writes are done before anything else of the configuration runs. The configuration's own stream switch
writes then zero the loader's ports (``program._prune``'s merged port blockwrites; the ones it does not cover in the
prelude); its slots stay, inert unless the design puts that slave port in packet mode (then zeroed too). A tile's memory
holds nothing of the design before its configuration: the marker costs none.

A configuration takes the loader where its estimated cost (``worth``: the measured costs of a transaction's
operations) is below the transaction's for the same writes. ``ARTLITE_PM_PACKETS=0`` keeps every write in the
transaction.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ... import txn

ENABLED = os.environ.get("ARTLITE_PM_PACKETS", "1") != "0"
ROW_ID = {1: 26, 2: 27, 3: 29, 4: 30, 5: 31}  # (a tile's controller id per row: its packets' stream id)
MARK = {1: 0x7FFFC, 2: 0xFFFC}  # (the memory word the column's packets end with: memtile, core tile)
MARK_VALUE = 0x5A5A0001
BD0 = 11  # (the shim BDs of a column's chained task, one per tile: free during a configuration)
# a transaction's operations (s): a single write, a blockwrite (and per byte), a poll that passes; packets per byte
# through a column's stream (measured on a Llama 1B chunk: 129 configurations)
WRITE_S, BLOCKWRITE_S, BLOCK_BYTE_S, POLL_S, STREAM_BYTE_S = 0.133e-6, 0.101e-6, 1.5e-9, 0.075e-6, 0.25e-9

# (the registers a tile's packets take, per row kind: core tile - program memory, DMA BDs, DMA channels, locks, core
# control; memtile - DMA BDs, DMA channels, locks)
_MOVED = {2: ((0x20000, 0x24000), (0x1D000, 0x1D200), (0x1DE00, 0x1DE20), (0x1F000, 0x1F100), (0x32000, 0x32004)),
          1: ((0xA0000, 0xA0660), (0xC0000, 0xC0400))}
_CORE_CONTROL, _CORE_CONTROL_RESET = 0x32000, 0x2  # (its value after a reset: in reset, disabled)
# (stream switch registers: slave South 0, its slots, masters TileControl .. North 0, the slave port number of South 0)
_SWITCH = {2: (0x3F114, 0x3F250, 0x3F00C, 0x3F034), 1: (0xB011C, 0xB0270, 0xB0018, 0xB002C)}
_MEM_SOUTH0 = 7
_SHIM_MUX, _SHIM_MUX_SOUTH3_DMA = 0x1F000, 0x400
_SHIM_SLAVE_SOUTH3, _SHIM_MASTER_NORTH0, _SHIM_SOUTH3 = 0x3F114, 0x3F030, 5  # (South 3: slave port 5)
_ENABLE, _PACKET = 1 << 31, 1 << 30

Writes = dict[tuple[int, int], list[tuple[int, np.ndarray]]]  # ((col, row) -> [(tile-local address, words)])


def _parity(word: int) -> int:
    """The word with its odd parity bit (31)."""
    return word | (0 if word.bit_count() % 2 else 1 << 31)


def _slot(stream_id: int, arbiter: int, mask: int = 0x1F) -> int:
    """A slave port slot matching the stream ids equal to ``stream_id`` in the bits ``mask`` to an arbiter, msel 0."""
    return (stream_id << 24) | (mask << 16) | (1 << 8) | arbiter


def _north(own: int, up: list[int]) -> list[int]:
    """Slots (arbiter 1) matching the ids ``up`` but not ``own``: per bit, chosen greedily, the ids that differ from own
    in it (a slave port has 4 slots: a memtile's own and 4 ids above would not fit one each)."""
    out, left = [], set(up)
    while left:
        b = max(range(5), key=lambda b: sum(((i ^ own) >> b) & 1 for i in left))
        out.append(_slot((~own >> b & 1) << b, 1, 1 << b))
        left = {i for i in left if not ((i ^ own) >> b) & 1}
    return out


def packets(stream_id: int, writes: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """A tile's writes ((tile-local address, words)) as control packets: per packet the stream header (``stream_id``:
    the route's packet id), the control header (write, words - 1, address), up to 4 words (mlir-aie's encoding)."""
    head = _parity(stream_id)
    out = []
    for addr, data in writes:
        for i in range(0, len(data), 4):
            d = data[i:i + 4]
            out += [head, _parity(((len(d) - 1) << 20) | ((addr + 4 * i) & 0xFFFFF)), *(int(v) for v in d)]
    return np.array(out, np.uint32)


def split(cfg: np.ndarray) -> tuple[np.ndarray, Writes]:
    """A configuration (after its PDI load) without the writes control packets take (per core and memory tile its
    program memory, DMA BDs, DMA channels, locks, the core's control but its enable: writes, blockwrites, masked writes
    as the values they leave), and those per tile in their order."""
    rest, moved, value = [], defaultdict(list), {}
    ops = txn.op_offsets(cfg)
    for (p, op), e in zip(ops, [q for q, _ in ops[1:]] + [len(cfg)]):
        reg = int(cfg[p + 2]) if op in (txn.OP_WRITE, txn.OP_BLOCKWRITE, txn.OP_MASKWRITE) else 0
        row, off = min((reg >> 20) & 0x1F, 2), reg & 0xFFFFF
        n = (e - p - 4) if op == txn.OP_BLOCKWRITE else 1
        if not row or not any(a <= off and off + 4 * n <= b for a, b in _MOVED[row]) or \
                (op == txn.OP_MASKWRITE and row == 2 and off == _CORE_CONTROL and int(cfg[p + 4]) & int(cfg[p + 5]) & 1):
            rest.append(cfg[p:e])  # (not a tile's, or the core's enable)
            continue
        if op == txn.OP_MASKWRITE:
            v, m = int(cfg[p + 4]), int(cfg[p + 5])
            old = value.get(reg, _CORE_CONTROL_RESET if row == 2 and off == _CORE_CONTROL else 0)
            words = np.array([(old & ~m) | (v & m)], np.uint32)
        else:
            words = cfg[p + 4:e] if op == txn.OP_BLOCKWRITE else cfg[p + 4:p + 5]
        for i, w in enumerate(words):
            value[reg + 4 * i] = int(w)
        moved[(reg >> 25, (reg >> 20) & 0x1F)].append((off, words))
    return (np.concatenate(rest) if rest else np.zeros(0, np.uint32)), dict(moved)


@dataclass
class Loader:
    """The control packets of a configuration's writes (``split``'s) and the transaction around them."""

    writes: Writes
    columns: dict[int, list[int]] = field(init=False)  # (column -> its rows with packets, ascending)
    streams: dict[tuple[int, int], np.ndarray] = field(init=False)  # ((col, row) -> its packets)

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

    def routes(self) -> list[tuple[int, list[int], str]]:
        """The loader's writes (absolute address, words, kind: "mux", "ports", "slots"): per column the shim mux and
        the circuit through the shim; at each tile up to the column's top one with packets its slave South 0 in packet
        mode, slots (its own id to arbiter 0, the ids above to arbiter 1: ``_north``) and masters (TileControl on arbiter 0, North
        0 on arbiter 1; the ports between them 0, as after the reset), the memtile in circuit mode without packets."""
        a, out = txn.tile_address, []
        for c, rows in self.columns.items():
            out += [(a(c, 0, _SHIM_MUX), [_SHIM_MUX_SOUTH3_DMA], "mux"), (a(c, 0, _SHIM_SLAVE_SOUTH3), [_ENABLE], "ports"),
                    (a(c, 0, _SHIM_MASTER_NORTH0), [_ENABLE | _SHIM_SOUTH3], "ports")]
            for r in range(1, rows[-1] + 1):
                slave, slots0, ctrl, north = _SWITCH[min(r, 2)]
                if r == 1 and r not in rows:
                    out += [(a(c, 1, slave), [_ENABLE], "ports"), (a(c, 1, north), [_ENABLE | _MEM_SOUTH0], "ports")]
                    continue
                up = [ROW_ID[q] for q in rows if q > r]
                slots = ([_slot(ROW_ID[r], 0)] if r in rows else []) + _north(ROW_ID[r], up)
                masters = [0] * ((north - ctrl) // 4 + 1)
                if r in rows:
                    masters[0] = _ENABLE | _PACKET | (1 << 3)  # (arbiter 0, msel 0; the header kept)
                if up:
                    masters[-1] = _ENABLE | _PACKET | (1 << 3) | 1
                lo = 0 if r in rows else len(masters) - 1
                hi = len(masters) - 1 if up else 0
                out += [(a(c, r, slave), [_ENABLE | _PACKET], "ports"), (a(c, r, slots0), slots, "slots"),
                        (a(c, r, ctrl + 4 * lo), masters[lo:hi + 1], "ports")]
        return out

    def ports(self) -> list[int]:
        """The stream switch port registers the routes enable (``program._prune`` zeroes them)."""
        return [addr + 4 * i for addr, words, kind in self.routes() if kind == "ports" for i, w in enumerate(words) if w]

    def slots(self, rest: np.ndarray) -> list[int]:
        """The slot registers to zero: a tile's whose slave South 0 the design (``rest``) puts in packet mode."""
        packet = set()
        for p, op in txn.op_offsets(rest):
            if op in (txn.OP_WRITE, txn.OP_MASKWRITE):
                reg = int(rest[p + 2])
                row = (reg >> 20) & 0x1F
                if row and reg & 0xFFFFF == _SWITCH[min(row, 2)][0] and int(rest[p + 4]) & _PACKET:
                    packet.add(reg & ~0xFFFFF)
        return [addr + 4 * i for addr, words, kind in self.routes()
                if kind == "slots" and addr & ~0xFFFFF in packet for i in range(len(words))]

    def prelude(self, zeros: list[int]) -> tuple[np.ndarray, int, list[tuple[int, tuple[int, int]]]]:
        """The transaction between the PDI load and the rest of the configuration: the routes, the columns' markers
        cleared, per column a chained task of its tiles' packets (addresses 0: ``place``), the markers polled, then the
        registers ``zeros`` (the loader's ports the rest does not zero, slots) written 0. Returns its words (no
        header), operation count and per tile the word of its BD's address."""
        t, a = txn.Txn(), txn.tile_address
        for addr, words, _ in self.routes():
            t.blockwrite(addr, words) if len(words) > 1 else t.write32(addr, words[0])
        for c, rows in self.columns.items():
            t.write32(a(c, rows[-1], MARK[min(rows[-1], 2)]), 0)
        at = []
        for c, rows in self.columns.items():
            for k, r in enumerate(rows):
                bd, last = BD0 + k, k == len(rows) - 1
                w = t.blockwrite(a(c, 0, txn.SHIM_BD0 + txn.BD_BYTES * bd),
                                 [*txn.bd_words(0, 4 * len(self.streams[(c, r)])), 0, 3 << 30, 2 << 24, 0,
                                  (1 << 25) | (0 if last else (1 << 26) | ((bd + 1) << 27))])
                at.append((w + 1, (c, r)))  # (its address low word: the blockwrite's data from w + 4, after the header)
            t.shim_push(c, txn.MM2S, 0, BD0)
        for c, rows in self.columns.items():
            t.maskpoll32(a(c, rows[-1], MARK[min(rows[-1], 2)]), MARK_VALUE, 0xFFFFFFFF)
        for lo, hi in _runs(sorted(zeros)):
            t.blockwrite(lo, [0] * ((hi - lo) // 4 + 1)) if hi > lo else t.write32(lo, 0)
        return t.words()[4:], t.n_ops, at


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
