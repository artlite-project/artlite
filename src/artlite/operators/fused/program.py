"""Programs assembled on the host from a library ELF's control program (``Library``; no device here).

aiecc lowers a library's runlist to one block per entry: a ConfigureOp (an expanded PDI load: the array reset and its
design's configuration, ending with the cores' enables) and a RunOp. A program lists items, each an entry with the
device addresses of its arguments; ``Assembler.assemble`` writes the items' RunOps with those addresses, an entry's
configuration only where the previous item is of another design:

- the configurations' PDI loads alternate between the library's two (the firmware skips the load, and with it the
  array reset, of the PDI it loaded last);
- a RunOp again right after its own design: the form with a poll for a free task-queue slot before each shim task
  push (``Run.guarded``);
- memory-tile BDs the registers already hold since the last configuration left out (``Run.emit``);
- the configurations without the writes their PDI load's array reset makes redundant, a tile's stream switch ports
  as blockwrites (``_prune``: 57% fewer configuration operations in a Llama prefill chunk, -6% time at 1K tokens);
- the cores' program memory by control packets where they cost less than blockwrites (``ctrlpkt``: a Llama 1B chunk
  of 128 tokens 43 -> 35 ms), from a packet buffer the library allocates (``Assembler.place``).
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

import numpy as np

from ... import txn
from . import ctrlpkt
from .elf import ShimBd

# Bytes of a program's piece (a dispatch: ``Library.run``): two instruction BOs of this size share the device's 64 MB
# heap with the library's PDIs
PIECE_BYTES = 24 << 20

Item = tuple[int, Sequence[int]]  # (entry, the device addresses of its arguments)
Slot = tuple[int, int, int, int]  # (scratch start, end, entry, argument): where the ELF placed an entry's argument

_CORE_CONTROL = 0x32000  # a core's control register (enable: bit 0)
_CORE_CONTROL_RESET = 0x2  # (its value after a reset: Reset set, Enable clear; AM025)
_CORE_DMA_CTRL = (0x1DE00, 0x1DE08, 0x1DE10, 0x1DE18)  # a core's DMA channel controls (Reset: bit 1)
_START_QUEUES = {0: (*txn.QUEUE[txn.S2MM], *txn.QUEUE[txn.MM2S]), 1: tuple(0xA0604 + 8 * c for c in range(12)),
                 2: (0x1DE04, 0x1DE0C, 0x1DE14, 0x1DE1C)}  # (shim, memory tile, core: DMA start queues)
_CORE_LOCKS, _MEM_LOCKS = (0x1F000, 0x1F100), (0xC0000, 0xC0400)  # lock values (one per 16 B; 0 after a reset)
# (stream switch port configurations, master and slave ranges of the shim, memory and core tiles: contiguous, 0 after
# a reset; AM025)
_SWITCH_PORTS = {0: ((0x3F000, 0x3F058), (0x3F100, 0x3F15C)), 1: ((0xB0000, 0xB0044), (0xB0100, 0xB0148)),
                 2: ((0x3F000, 0x3F05C), (0x3F100, 0x3F164))}
_MEM_TILE_BDS = (0xA0000, 0xA0600)  # a memory tile's BD registers (48 BDs of 8 words)
# (word 6: Iteration_Current 28:23, which the DMA advances each time it loads the BD, Iteration_Wrap 22:17,
# Iteration_Stepsize 16:0: a BD with an iteration does not keep its written value, AM029 DMA_BD0_6)
_ITER_WORD, _ITER_MASK = 6, (1 << 23) - 1
_QUEUE_STATUS = {q: s for d in txn.QUEUE for q, s in zip(txn.QUEUE[d], txn.QUEUE_STATUS[d])}
_QUEUE_FULL = txn.QUEUE_DEPTH << 20  # (Task_Queue_Size, bits 22:20)


def _row(reg: int) -> int:
    return (reg >> 20) & 0x1F


@dataclass(eq=False)
class Run:
    """A RunOp's control code: its words, its operation count, and the word of each data BD's address (bits 31:2;
    the next word's low half: bits 47:32)."""

    words: np.ndarray
    n_ops: int
    bd_words: np.ndarray

    @functools.cached_property
    def ops(self) -> list[tuple[int, int]]:
        return txn.op_offsets(self.words)

    @functools.cached_property
    def guarded(self) -> Run:
        """This RunOp as it runs again right after its own design: a poll for a free slot in the channel's task queue
        before each shim task push (the previous run's unawaited tasks may still be queued, and a push onto a full
        queue is dropped; aiecc inserts these polls where its queue model cannot rule that out)."""
        parts, cuts, last = [], [], 0
        for p, op in self.ops:
            reg = int(self.words[p + 2]) if op == txn.OP_WRITE else 0
            if op != txn.OP_WRITE or _row(reg) != 0 or (status := _QUEUE_STATUS.get(reg & 0xFFFFF)) is None:
                continue
            parts += [self.words[last:p],
                      np.array([txn.OP_MASKPOLL, 0, (reg & ~0xFFFFF) | status, 0, 0, _QUEUE_FULL, 28], np.uint32)]
            cuts.append(p)
            last = p
        parts.append(self.words[last:])
        shift = 7 * np.searchsorted(np.asarray(cuts, np.int64), self.bd_words, side="right")
        return Run(np.concatenate(parts), self.n_ops + len(cuts), self.bd_words + shift)

    @functools.cached_property
    def _bd_units(self) -> _BdUnits:
        return _bd_units(self.words, self.ops)

    def emit(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """(words, BD address words, operation count) of this RunOp without its memory-tile BD writes whose registers
        already hold their values. ``state``: per column and BD register word, the value since the last configuration
        (-1: unknown); updated."""
        units = self._bd_units
        if not len(units.start):
            return self.words, self.bd_words, self.n_ops
        held = (state[units.col, units.reg] == units.val) & (units.val >= 0)
        state[units.col[units.last], units.reg[units.last]] = units.val[units.last]
        if units.dup:
            return self.words, self.bd_words, self.n_ops
        drop = np.minimum.reduceat(held, units.first).astype(bool)
        if not drop.any():
            return self.words, self.bd_words, self.n_ops
        keep = np.ones(len(self.words), bool)
        for s, e in zip(units.start[drop], units.end[drop]):
            keep[s:e] = False
        shift = np.cumsum(~keep)
        return self.words[keep], self.bd_words - shift[self.bd_words], self.n_ops - int(units.nops[drop].sum())


@dataclass(eq=False)
class _BdUnits:
    """A run's memory-tile BD writes (each BD: its blockwrite and the masked writes of its fields right after it): per
    unit its word range and operation count; their registers (column, word index in the BD block) and the values the
    unit leaves (-1: a BD with an iteration, whose word 6 the DMA advances: never left out, unknown after); ``dup``: a
    register written by two units (the run is then kept whole)."""

    start: np.ndarray
    end: np.ndarray
    nops: np.ndarray
    first: np.ndarray  # (each unit's first entry in col / reg / val)
    col: np.ndarray
    reg: np.ndarray
    val: np.ndarray
    dup: bool
    last: np.ndarray  # (per register written, its last entry: the value the run leaves)


def _bd_units(run: np.ndarray, ops: list[tuple[int, int]]) -> _BdUnits:
    ends = [p for p, _ in ops[1:]] + [len(run)]
    start, end, nops, first, col, reg, val = [], [], [], [], [], [], []
    k = 0
    while k < len(ops):
        p, op = ops[k]
        addr = int(run[p + 2]) if op == txn.OP_BLOCKWRITE else 0
        if op != txn.OP_BLOCKWRITE or _row(addr) != 1 or not _MEM_TILE_BDS[0] <= addr & 0xFFFFF < _MEM_TILE_BDS[1]:
            k += 1
            continue
        n = ends[k] - p - 4
        vals = {addr + 4 * i: int(v) for i, v in enumerate(run[p + 4:ends[k]])}
        j = k + 1  # (the writes into the BD right after its blockwrite)
        while (j < len(ops) and ops[j][1] in (txn.OP_WRITE, txn.OP_MASKWRITE)
               and addr <= int(run[ops[j][0] + 2]) < addr + 4 * n):
            q = ops[j][0]
            r, v = int(run[q + 2]), int(run[q + 4])
            m = int(run[q + 5]) if ops[j][1] == txn.OP_MASKWRITE else 0xFFFFFFFF
            vals[r] = (vals[r] & ~m & 0xFFFFFFFF) | (v & m)
            j += 1
        start.append(p)
        end.append(ends[j - 1])
        nops.append(j - k)
        first.append(len(col))
        iterated = vals.get(addr + 4 * _ITER_WORD, 0) & _ITER_MASK
        for r, v in vals.items():
            col.append(r >> 25)
            reg.append(((r & 0xFFFFF) - _MEM_TILE_BDS[0]) // 4)
            val.append(-1 if iterated else v)
        k = j
    keys = list(zip(col, reg))
    last = sorted({key: i for i, key in enumerate(keys)}.values())
    a = functools.partial(np.array, dtype=np.int64)
    return _BdUnits(a(start), a(end), a(nops), a(first), a(col), a(reg), a(val), len(set(keys)) != len(keys), a(last))


@dataclass(eq=False)
class Block:
    """An entry of the library: its configuration (words 0 .. 3: its PDI load), operation count and RunOp, the
    argument of each data BD and its byte offset in it. ``design``: the first entry with the same configuration and
    packets (operators that share one array design and differ in their runtime sequences; designs that differ only in
    program memory differ only in their packets). ``packets``: the control packets
    of its program memory (``ctrlpkt``) and the configuration word of each one's shim BD address (``Assembler.place``)."""

    config: np.ndarray
    n_config: int
    run: Run
    bd_arg: np.ndarray
    bd_rel: np.ndarray
    design: int
    packets: list[tuple[int, np.ndarray]] = field(default_factory=list)


def _prune(cfg: np.ndarray, zero: Collection[int] = ()) -> tuple[np.ndarray, list[int]]:
    """A configuration (after its PDI load: an array reset) with fewer operations (each costs the firmware ~0.1 us):
    without the writes the reset makes redundant (lock values written the value they hold, 0 after the reset, before
    the tile's DMA starts; the core DMA channels' reset pulses, set then cleared before the channel starts;
    Core_Control writes that leave its value, 0x2 after the reset: in reset, disabled), and a tile's stream switch port
    configurations before any DMA starts as one blockwrite per master and slave range (the ports it leaves out 0, their
    reset value). ``zero``: port registers a loader set before (``ctrlpkt``), zeroed by those blockwrites; returns the
    ones they do not cover."""
    ops = txn.op_offsets(cfg)
    ends = [p for p, _ in ops[1:]] + [len(cfg)]
    drop: set[int] = set()
    locks: dict[int, int] = {}
    started: set[int] = set()  # (tiles whose DMA a start-queue write started)
    pulse: dict[int, int] = {}  # (channel control register -> the op of its pending reset set)
    control: dict[int, int] = {}
    ports: dict[tuple[int, int], tuple[list[int], dict[int, int]]] = {}  # ((tile, range) -> (ops, {register: value}))
    masked: set[int] = set()  # (tiles with a port mask-written: left as they are)
    zeros: dict[tuple[int, int], set[int]] = {}
    for reg in zero:
        rng = next(r for r in _SWITCH_PORTS[min(_row(reg), 2)] if r[0] <= reg & 0xFFFFF < r[1])
        zeros.setdefault((reg & ~0xFFFFF, rng[0]), set()).add(reg)
    for k, (p, op) in enumerate(ops):
        if op not in (txn.OP_WRITE, txn.OP_MASKWRITE):
            continue
        reg, v = int(cfg[p + 2]), int(cfg[p + 4])
        tile, off, row = reg & ~0xFFFFF, reg & 0xFFFFF, _row(reg)
        if op == txn.OP_WRITE:
            if off in _START_QUEUES.get(min(row, 2), ()):
                started.add(tile)
            lock = (row >= 2 and _CORE_LOCKS[0] <= off < _CORE_LOCKS[1]) or (row == 1 and _MEM_LOCKS[0] <= off < _MEM_LOCKS[1])
            if lock and off % 16 == 0 and tile not in started:
                if locks.get(reg, 0) == v:
                    drop.add(k)
                locks[reg] = v
            rng = next((r for r in _SWITCH_PORTS.get(min(row, 2), ()) if r[0] <= off < r[1]), None)
            if rng is not None and not started:
                ks, vals = ports.setdefault((tile, rng[0]), ([], {}))
                ks.append(k)
                vals[reg] = v
            continue
        m = int(cfg[p + 5])
        if any(r[0] <= off < r[1] for r in _SWITCH_PORTS.get(min(row, 2), ())):
            masked.add(tile)
        if row >= 2 and off == _CORE_CONTROL:
            old = control.get(reg, _CORE_CONTROL_RESET)
            new = (old & ~m) | (v & m)
            if new == old:
                drop.add(k)
            control[reg] = new
        elif row >= 2 and off in _CORE_DMA_CTRL and m == 2 and tile not in started:
            if v & 2:
                pulse[reg] = k
            elif reg in pulse:
                drop |= {pulse.pop(reg), k}
    merged = {}
    for key, (ks, vals) in ports.items():
        tile = key[0]
        if tile in masked or (len(ks) < 2 and key not in zeros):
            continue
        vals = dict.fromkeys(zeros.pop(key, ()), 0) | vals
        drop.update(ks[1:])
        k, lo, hi = ks[0], min(vals), max(vals)
        words = np.zeros((hi - lo) // 4 + 1, np.uint32)
        for r, v in vals.items():
            words[(r - lo) // 4] = v
        merged[k] = np.concatenate([np.array([txn.OP_BLOCKWRITE, 0, lo, 4 * (4 + len(words))], np.uint32), words])
    left = sorted(r for z in zeros.values() for r in z)
    if not drop and not merged:
        return cfg, left
    keep = [merged[k] if k in merged else cfg[p:e] for k, ((p, _), e) in enumerate(zip(ops, ends)) if k not in drop]
    return np.concatenate(keep), left


def _cut(code: np.ndarray, bds: Sequence[ShimBd], slots: Sequence[Slot]
         ) -> tuple[np.ndarray, np.ndarray, list[Block], list[np.ndarray]]:
    """A library ELF's control program (patched, 32-bit words) -> its header, its preamble (the operations before the
    first PDI load: the scratchpad's creation), one ``Block`` per entry and the two PDI loads its configurations
    alternate between. ``bds``: its data BDs, ``slots``: where the ELF placed each entry's arguments."""
    ops = [(p + 4, op) for p, op in txn.op_offsets(code[4:])]
    starts = [p for p, _ in ops] + [len(code)]
    loads = [i for i, (_, op) in enumerate(ops) if op == txn.OP_LOAD_PDI] + [len(ops)]
    if loads[-2] == len(ops) - 1:  # (a PDI load closing the control code, after the last entry: not an entry)
        ops, starts = ops[:-1], starts[:-1]
        loads[-2:] = [len(ops)]
    assert all(op == txn.OP_CREATE_SCRATCHPAD for _, op in ops[:loads[0]]), "operations before the first PDI load"
    slot_starts = np.array([s[0] for s in slots])
    order = np.argsort(slot_starts)
    bd_at = np.array([bd.offset // 4 + 1 for bd in bds], np.int64)  # (the address word)
    bd_start = np.array([bd.start for bd in bds], np.int64)
    designs: dict[bytes, int] = {}  # configuration (after its PDI load) and its packets -> the first entry with them
    blocks: list[Block] = []
    for e, (a, b) in enumerate(itertools.pairwise(loads)):
        enables = [i for i in range(a, b) if ops[i][1] == txn.OP_MASKWRITE
                   and code[starts[i] + 2] & 0xFFFFF == _CORE_CONTROL and code[starts[i] + 4] & 1]
        assert enables, f"block {e}: no core enables"
        r = enables[-1] + 1  # (the RunOp: after the configuration's core enables)
        lo, mid, hi = starts[a], starts[r], starts[b]
        assert not ((bd_at >= lo) & (bd_at < mid)).any(), f"block {e}: a data BD in the configuration"
        sel = (bd_at >= mid) & (bd_at < hi)
        k = order[np.searchsorted(slot_starts[order], bd_start[sel], side="right") - 1]
        assert all(slots[i][2] == e for i in k) and (bd_start[sel] < [slots[i][1] for i in k]).all(), \
            f"block {e}: BD arguments"
        cfg, loader = code[lo + 4:mid], None
        if ctrlpkt.ENABLED:
            without, pm = ctrlpkt.split(cfg)
            if pm and ctrlpkt.worth(candidate := ctrlpkt.Loader(pm)):
                cfg, loader = without, candidate
        rest, left = _prune(cfg, loader.ports() if loader is not None else ())
        packets = []
        if loader is not None:
            pre, _, at = loader.prelude(left + loader.slots(rest))
            rest = np.concatenate([pre, rest])
            packets = [(4 + w, loader.streams[tile]) for w, tile in at]
        config = np.concatenate([code[lo:lo + 4], rest])
        blocks.append(Block(config, 1 + len(txn.op_offsets(rest)), Run(code[mid:hi].copy(), b - r, bd_at[sel] - mid),
                            np.array([slots[i][3] for i in k], np.int64), bd_start[sel] - slot_starts[k],
                            designs.setdefault(b"".join([config[4:].tobytes(), *(p.tobytes() for _, p in packets)]), e),
                            packets))
    loadpdi = {tuple(int(v) for v in blk.config[:4]) for blk in blocks}
    assert len(loadpdi) == 2, f"configurations alternate between two PDI loads: {len(loadpdi)}"
    return code[:4].copy(), code[4:starts[loads[0]]].copy(), blocks, [np.array(v, np.uint32) for v in sorted(loadpdi)]


class Assembler:
    """Programs of a library ELF's blocks (``_cut``)."""

    def __init__(self, code: np.ndarray, bds: Sequence[ShimBd], slots: Sequence[Slot]):
        self.header, self.preamble, self.blocks, self.loadpdi = _cut(code, bds, slots)
        self.parity = 1  # (the PDI load of the next configuration; the first: the second, as aiecc starts)
        self._offsets: dict[bytes, int] = {}  # (each distinct packet stream's byte offset in the packet buffer)
        self.packet_bytes = 0
        for blk in self.blocks:
            for _, p in blk.packets:
                if (key := p.tobytes()) not in self._offsets:
                    self._offsets[key] = self.packet_bytes
                    self.packet_bytes += len(key)

    def place(self, view: np.ndarray, address: int) -> None:
        """Write the configurations' control packets into the packet buffer (``view``: its uint32 words, at the device
        address ``address``) and their shim BDs' addresses into the configurations."""
        for data, off in self._offsets.items():
            view[off // 4:off // 4 + len(data) // 4] = np.frombuffer(data, np.uint32)
        for blk in self.blocks:
            for w, p in blk.packets:
                a = address + self._offsets[p.tobytes()]
                blk.config[w], blk.config[w + 1] = a & 0xFFFFFFFF, a >> 32

    def _configure(self, e: int, prev: int | None) -> bool:
        """Whether item ``e`` after item ``prev`` takes its design's configuration."""
        return prev is None or self.blocks[prev].design != self.blocks[e].design

    def assemble(self, items: Sequence[Item]) -> np.ndarray:
        """A control program (a TXN) of the items; advances ``parity``."""
        parts, n_ops, prev = [self.header, self.preamble], len(self.preamble) // 4, None  # (CREATE_SCRATCHPAD: 4 words)
        state = np.full((64, (_MEM_TILE_BDS[1] - _MEM_TILE_BDS[0]) // 4), -1, np.int64)  # (per column: the BDs held)
        for e, addrs in items:
            blk = self.blocks[e]
            configure = self._configure(e, prev)
            if configure:
                cfg = blk.config.copy()
                cfg[:4] = self.loadpdi[self.parity]
                self.parity ^= 1
                parts.append(cfg)
                n_ops += blk.n_config
                state[:] = -1
            words, bd_words, n = (blk.run if configure else blk.run.guarded).emit(state)
            words = words.copy()
            addr = np.asarray(addrs, np.int64)[blk.bd_arg] + blk.bd_rel
            words[bd_words] = (addr & 0xFFFFFFFC).astype(np.uint32)
            words[bd_words + 1] = (words[bd_words + 1] & 0xFFFF0000) | (addr >> 32).astype(np.uint32)
            parts.append(words)
            n_ops += n
            prev = e
        out = np.concatenate(parts)
        out[2], out[3] = n_ops, 4 * out.size
        return out

    def bounds(self, items: Sequence[Item], splits: Collection[int]) -> list[int]:
        """The item bounds of a program's pieces: ``splits``, and a cut at the last change of design before a piece
        would exceed PIECE_BYTES."""
        def nbytes(i: int, prev: int | None) -> int:  # (as ``assemble`` emits it, all memory-tile BDs written)
            blk = self.blocks[items[i][0]]
            if self._configure(items[i][0], prev):
                return 4 * (blk.config.size + blk.run.words.size)
            return 4 * blk.run.guarded.words.size

        empty = 4 * (4 + self.preamble.size)
        forced = set(splits) | {len(items)}
        bounds, size, prev, last_change = [0], empty, None, 0
        for i, (e, _) in enumerate(items):
            if i in forced and i:
                bounds.append(i)
                size, prev = empty, None
            n = nbytes(i, prev)
            if prev is None or self.blocks[e].design != self.blocks[prev].design:
                last_change = i
            if size + n > PIECE_BYTES and i > bounds[-1]:
                cut = last_change if last_change > bounds[-1] else i
                bounds.append(cut)
                size, prev = empty, None
                for j in range(cut, i + 1):  # (the items moved into the new piece)
                    size += nbytes(j, prev)
                    prev = items[j][0]
                continue
            size += n
            prev = e
        bounds.append(len(items))
        return sorted(set(bounds))
