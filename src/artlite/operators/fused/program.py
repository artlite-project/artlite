"""Programs assembled on the host from a library ELF's control program (``Library``; no device here).

aiecc lowers a library's runlist to one block per entry: a ConfigureOp (an expanded PDI load: the array reset and its
design's configuration, ending with the cores' enables) and a RunOp. A program lists items, each an entry with the
device addresses of its arguments; ``Assembler.assemble`` writes the items' RunOps with those addresses, an entry's
configuration only where the previous item is of another design:

- the configurations' PDI loads alternate between the library's two (the firmware skips the load, and with it the
  array reset, of the PDI it loaded last); a piece's first loads the other one first (skipped after the previous
  piece: 0.25 us), so that its own load resets whatever ran before (a piece assembled and never dispatched, one a
  TDR stopped, another context's, a fresh context: neither PDI loaded yet);
- a RunOp again right after its own design: the form with a poll for a free task-queue slot before each shim task
  push (``Run.guarded``);
- memory-tile BDs the registers already hold since the last configuration left out (``Run.emit``);
- the configurations without the writes their PDI load's array reset makes redundant, a tile's stream switch ports
  as blockwrites (``_prune``: 57% fewer configuration operations in a Llama prefill chunk, -6% time at 1K tokens);
- the cores' program and data memory by control packets where they cost less than blockwrites (``ctrlpkt``: a Llama
  1B chunk of 128 tokens 43 -> 35 ms), from a packet buffer the library allocates (``Assembler.place``);
- a configuration after another of the same piece without the array reset (``Quiet``: the PDI loaded last again, which
  the firmware skips, 0.25 against 37 us; a 64-token chunk of Qwen3 0.6B or qwen35 0.8B -7 to -8 ms): the previous
  design's cores outside the next one held in reset, its cores that wait on a neighbour's locks too and those locks
  flushed (``locks``), each tile of the design quiesced by its packets (``ctrlpkt.Loader(reused=True)``). The reset
  stays where a design could still disturb the next one (``Assembler._quiet``: a stale stream switch master on a slave
  the next design uses, a memory tile DMA of the previous one on a neighbour's locks, a channel of the next one that
  sends before any lock release), at a piece's first configuration (another context may have run before it), and
  everywhere with ``ARTLITE_PDI_RESET=1``.
"""

from __future__ import annotations

import functools
import itertools
import os
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ... import txn
from . import ctrlpkt
from .elf import ShimBd

if TYPE_CHECKING:
    from .locks import Census

# Bytes of a program's piece (a dispatch: ``Library.run``) at most: the process's two piece BOs share its device heap
# with the PDIs (smaller on a heap that cannot grow: ``library.Pieces``)
PIECE_BYTES = 24 << 20
REUSE = os.environ.get("ARTLITE_PDI_RESET", "0") != "1"  # (configurations without the array reset: ``Quiet``)

Item = tuple[int, Sequence[int]]  # (entry, the device addresses of its arguments)
Slot = tuple[int, int, int, int]  # (scratch start, end, entry, argument): where the ELF placed an entry's argument

_CORE_CONTROL = 0x32000  # a core's control register (enable: bit 0)
_CORE_CONTROL_RESET = 0x2  # (its value after a reset: Reset set, Enable clear; AM025)
_CORE_DMA_CTRL = (0x1DE00, 0x1DE08, 0x1DE10, 0x1DE18)  # a core's DMA channel controls (Reset: bit 1)
_START_QUEUES = {0: (*txn.QUEUE[txn.S2MM], *txn.QUEUE[txn.MM2S]), 1: tuple(0xA0604 + 8 * c for c in range(12)),
                 2: (0x1DE04, 0x1DE0C, 0x1DE14, 0x1DE1C)}  # (shim, memory tile, core: DMA start queues)
_CORE_LOCKS, _MEM_LOCKS = (0x1F000, 0x1F100), (0xC0000, 0xC0400)  # lock values (one per 16 B; 0 after a reset)
# (stream switch port configurations, master and slave ranges of the shim, memory and core tiles: contiguous, 0 after
# a reset)
_SWITCH_PORTS = ctrlpkt.SWITCH_PORTS
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
    ports: dict[int, dict[int, int]] = field(default_factory=dict)  # (``_ports`` of its configuration)
    quiet: Quiet | None = None


@dataclass(eq=False)
class Quiet:
    """An entry's configuration on the array its predecessor left running, without the array reset (words 0 .. 3: a
    PDI load, the one loaded last when assembled: ``ctrlpkt.Loader(reused=True)``), its operation count and packets
    (``Block.packets``'s form), and what ``Assembler._quiet`` checks: its tiles with packets and cores, its port
    configurations (``_ports``), whether a channel it starts can send before any lock release (``_early``: the previous
    design's routes would take the data), whether a memory tile DMA of it acquires a neighbour's lock (``_far``: that
    acquire would stay pending after it), and per core tile the locks its DMA BDs take."""

    config: np.ndarray
    n_config: int
    packets: list[tuple[int, np.ndarray]]
    tiles: frozenset[tuple[int, int]]
    cores: frozenset[tuple[int, int]]
    ports: dict[int, dict[int, int]]
    early: bool
    far: bool
    dma_locks: dict[tuple[int, int], set[int]]


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


def _ports(words: np.ndarray) -> dict[int, dict[int, int]]:
    """The stream switch port configurations a transaction leaves: per tile (its base address) register -> value."""
    out: dict[int, dict[int, int]] = {}
    for p, op in txn.op_offsets(words):
        if op not in (txn.OP_WRITE, txn.OP_BLOCKWRITE, txn.OP_MASKWRITE):
            continue
        reg = int(words[p + 2])
        tile, off = reg & ~0xFFFFF, reg & 0xFFFFF
        n = int(words[p + 3]) // 4 - 4 if op == txn.OP_BLOCKWRITE else 1
        for i, v in enumerate(words[p + 4:p + 4 + n]):
            if any(a <= off + 4 * i < b for a, b in _SWITCH_PORTS[min(_row(reg), 2)]):
                regs = out.setdefault(tile, {})
                if op == txn.OP_MASKWRITE:
                    m = int(words[p + 5])
                    v = (regs.get(off, 0) & ~m) | (int(v) & m)
                regs[off + 4 * i] = int(v)
    return out


def _values(writes: list[tuple[int, np.ndarray]]) -> dict[int, int]:
    """The register values a tile's moved writes (``ctrlpkt.split``'s) leave: tile-local address -> value (its BDs,
    channels and locks: not memory)."""
    out = {}
    for addr, words in writes:
        if addr >= 0x1D000:
            out.update((addr + 4 * i, int(w)) for i, w in enumerate(words))
    return out


def _signed7(v: int) -> int:
    return v - 128 if v & 0x40 else v


def _early(values: dict[tuple[int, int], dict[int, int]]) -> bool:
    """Whether an MM2S channel the configuration starts (memory and core tiles) can send before any lock release: its
    first BD acquires no lock, or one whose initial value already grants it (or a neighbour memory tile's)."""
    for (c, r), v in values.items():
        if r == 1:
            starts = [v[q] & 0x3F for q in (0xA0634 + 8 * k for k in range(6)) if q in v]
            bds = [(v.get(0xA001C + 32 * bd, 0) >> 15 & 1, v.get(0xA001C + 32 * bd, 0)) for bd in starts]
            locks = [(on, w & 0xFF, _signed7(w >> 8 & 0x7F)) for on, w in bds]
            base, own = 0xC0000, range(64, 128)
        else:
            starts = [v[q] & 0xF for q in (0x1DE14, 0x1DE1C) if q in v]
            bds = [(v.get(0x1D014 + 32 * bd, 0) >> 12 & 1, v.get(0x1D014 + 32 * bd, 0)) for bd in starts]
            locks = [(on, (w & 0xF) + 64, _signed7(w >> 5 & 0x7F)) for on, w in bds]
            base, own = 0x1F000, range(64, 80)
        for on, lock, value in locks:
            if not on or lock not in own:
                return True
            init = v.get(base + 16 * (lock - 64), 0)
            if init >= -value if value <= 0 else init == value:
                return True
    return False


def _far(values: dict[tuple[int, int], dict[int, int]], run: Run) -> bool:
    """Whether a memory tile BD (of the configuration or its RunOp) acquires or releases a neighbour memory tile's lock
    (ids 0-63 west, 128-191 east)."""
    words = [w for (_, r), v in values.items() if r == 1
             for a, w in v.items() if _MEM_TILE_BDS[0] <= a < _MEM_TILE_BDS[1] and a % 32 == 28]
    for p, op in run.ops:
        reg = int(run.words[p + 2]) if op in (txn.OP_WRITE, txn.OP_BLOCKWRITE) else 0
        if _row(reg) == 1 and _MEM_TILE_BDS[0] <= reg & 0xFFFFF < _MEM_TILE_BDS[1]:
            n = int(run.words[p + 3]) // 4 - 4 if op == txn.OP_BLOCKWRITE else 1
            words += [int(w) for i, w in enumerate(run.words[p + 4:p + 4 + n]) if ((reg & 0xFFFFF) + 4 * i) % 32 == 28]
    return any(w >> 31 and ((w >> 15 & 1 and not 64 <= w & 0xFF < 128)
                            or (w >> 24 & 0x7F and not 64 <= w >> 16 & 0xFF < 128)) for w in words)


def _dma_locks(values: dict[tuple[int, int], dict[int, int]]) -> dict[tuple[int, int], set[int]]:
    """Per core tile, the locks its valid DMA BDs acquire or release."""
    out = {}
    for t, v in values.items():
        if t[1] >= 2:
            ws = [v.get(0x1D014 + 32 * bd, 0) for bd in range(16)]
            out[t] = {k for w in ws if w >> 25 & 1
                      for k, on in ((w & 0xF, w >> 12 & 1), (w >> 13 & 0xF, w >> 18 & 0x7F)) if on}
    return out


def _quiet(load: np.ndarray, without: np.ndarray, pm: ctrlpkt.Writes, run: Run) -> Quiet:
    """An entry's ``Quiet`` (``without``, ``pm``: ``ctrlpkt.split`` of its configuration)."""
    loader = ctrlpkt.Loader(pm, reused=True)
    rest, left = _prune(without, loader.ports())
    pre, _, at = loader.prelude(left + loader.slots(rest))
    rest = np.concatenate([pre, rest])
    values = {t: _values(w) for t, w in pm.items()}
    return Quiet(np.concatenate([load, rest]), 1 + len(txn.op_offsets(rest)), [(4 + w, p) for w, p in at],
                 frozenset(pm), frozenset(t for t in pm if t[1] >= 2), _ports(rest), _early(values),
                 _far(values, run), _dma_locks(values))


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
        cfg, loader, pm = code[lo + 4:mid], None, None
        if ctrlpkt.ENABLED:
            without, pm = ctrlpkt.split(cfg)
            if pm and ctrlpkt.worth(candidate := ctrlpkt.Loader(pm)):
                cfg, loader = without, candidate
        rest, left = _prune(cfg, loader.ports() if loader is not None else ())
        packets = []
        if loader is not None:
            pre, _, at = loader.prelude(left + loader.slots(rest))
            rest = np.concatenate([pre, rest])
            packets = [(4 + w, p) for w, p in at]
        config = np.concatenate([code[lo:lo + 4], rest])
        run = Run(code[mid:hi].copy(), b - r, bd_at[sel] - mid)
        blocks.append(Block(config, 1 + len(txn.op_offsets(rest)), run,
                            np.array([slots[i][3] for i in k], np.int64), bd_start[sel] - slot_starts[k],
                            designs.setdefault(b"".join([config[4:].tobytes(), *(p.tobytes() for _, p in packets)]), e),
                            packets, _ports(rest), _quiet(code[lo:lo + 4], without, pm, run) if pm else None))
    loadpdi = {tuple(int(v) for v in blk.config[:4]) for blk in blocks}
    assert len(loadpdi) == 2, f"configurations alternate between two PDI loads: {len(loadpdi)}"
    return code[:4].copy(), code[4:starts[loads[0]]].copy(), blocks, [np.array(v, np.uint32) for v in sorted(loadpdi)]


class Assembler:
    """Programs of a library ELF's blocks (``_cut``)."""

    def __init__(self, code: np.ndarray, bds: Sequence[ShimBd], slots: Sequence[Slot], census: Census | None = None):
        """``census``: the library's cores' acquires of a neighbour's lock (``locks.census``; None: every
        configuration with the array reset)."""
        self.header, self.preamble, self.blocks, self.loadpdi = _cut(code, bds, slots)
        # (the PDI load of the next configuration with the reset, the other most likely the firmware's last; the first:
        # the second, as aiecc starts)
        self.parity = 1
        self.item_ops: list[int] = []  # (the last ``assemble``'s operation count through the end of each of its items)
        self.reuse = REUSE and census is not None  # (configurations without the array reset: ``Quiet``)
        # (per entry: its cores that acquire a neighbour's lock, and those (tile, lock))
        self._neighbours = {e: (frozenset(t for t, _, _ in v), frozenset((lt, k) for _, lt, k in v))
                            for e, v in (census or {}).items()}
        self._pre: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        self._offsets: dict[bytes, int] = {}  # (each distinct packet stream's byte offset in the packet buffer)
        self.packet_bytes = 0
        for blk in self.blocks:
            for _, p in [*blk.packets, *(blk.quiet.packets if blk.quiet is not None else ())]:
                if (key := p.tobytes()) not in self._offsets:
                    self._offsets[key] = self.packet_bytes
                    self.packet_bytes += len(key)

    def place(self, view: np.ndarray, address: int) -> None:
        """Write the configurations' control packets into the packet buffer (``view``: its uint32 words, at the device
        address ``address``) and their shim BDs' addresses into the configurations."""
        for data, off in self._offsets.items():
            view[off // 4:off // 4 + len(data) // 4] = np.frombuffer(data, np.uint32)
        for cfg, packets in [(b.config, b.packets) for b in self.blocks] + \
                            [(b.quiet.config, b.quiet.packets) for b in self.blocks if b.quiet is not None]:
            for w, p in packets:
                a = address + self._offsets[p.tobytes()]
                cfg[w], cfg[w + 1] = a & 0xFFFFFFFF, a >> 32

    def _configure(self, e: int, prev: int | None) -> bool:
        """Whether item ``e`` after item ``prev`` takes its design's configuration."""
        return prev is None or self.blocks[prev].design != self.blocks[e].design

    def _before(self, p: int, e: int) -> tuple[np.ndarray, int]:
        """The transaction before entry ``e``'s ``Quiet`` after entry ``p``'s (words, operation count): ``p``'s cores
        outside ``e``'s tiles held in reset, and its cores that acquire a neighbour's lock (and those locks' tiles'),
        then each of those locks written ``ctrlpkt.LOCK_FLUSH`` (granting the acquire such a core left pending: neither
        its reset cancels it nor, when the lock's tile comes first, that tile's own quiescing) and 0."""
        if (p, e) not in self._pre:
            qp, qe = self.blocks[p].quiet, self.blocks[e].quiet
            crossers, locks = self._neighbours.get(p, ((), ()))
            stop = (qp.cores - qe.tiles) | set(crossers) | {t for t, _ in locks if t in qp.cores}
            t = txn.Txn()
            for c, r in sorted(stop):
                t.write32(txn.tile_address(c, r, _CORE_CONTROL), _CORE_CONTROL_RESET)
            for (c, r), k in sorted(locks):
                for v in (ctrlpkt.LOCK_FLUSH, 0):
                    t.write32(txn.tile_address(c, r, ctrlpkt.LOCKS[2][0] + 16 * k), v)
            self._pre[(p, e)] = (t.words()[4:], t.n_ops)
        return self._pre[(p, e)]

    def _quiet(self, p: int | None, e: int, ports: dict[int, dict[int, int]]) -> bool:
        """Whether entry ``e``'s configuration after entry ``p``'s goes without the array reset (``ports``: the stream
        switch port configurations since the last reset): not where ``p`` leaves an acquire pending that no flush
        reaches (a memory tile DMA's on a neighbour's lock, a core's on a neighbour's lock its tile's DMA also takes),
        where a channel of ``e`` sends before any lock release (the previous routes would take it), or where a master
        port enabled since the last reset that ``e`` does not configure would take a stream of ``e`` (its slave enabled)
        or packets (in packet mode)."""
        if not self.reuse or p is None or self.blocks[p].quiet is None or self.blocks[e].quiet is None:
            return False
        qp, qe = self.blocks[p].quiet, self.blocks[e].quiet
        _, locks = self._neighbours.get(p, ((), ()))
        if qp.far or qe.early or any(k in qp.dma_locks.get(t, ()) for t, k in locks):
            return False
        for tile, regs in qe.ports.items():
            old = ports.get(tile)
            if not old:
                continue
            (m0, m1), (s0, _) = _SWITCH_PORTS[min(_row(tile), 2)]
            for reg, v in old.items():
                if m0 <= reg < m1 and v >> 31 and reg not in regs and \
                        (v >> 30 & 1 or regs.get(s0 + 4 * (v & 0x7F), old.get(s0 + 4 * (v & 0x7F), 0)) >> 31):
                    return False
        return True

    def assemble(self, items: Sequence[Item]) -> np.ndarray:
        """A control program (a TXN) of the items; advances ``parity``; ``item_ops`` holds where each item ends."""
        parts, n_ops, prev = [self.header, self.preamble], len(self.preamble) // 4, None  # (CREATE_SCRATCHPAD: 4 words)
        self.item_ops = ends = []
        state = np.full((64, (_MEM_TILE_BDS[1] - _MEM_TILE_BDS[0]) // 4), -1, np.int64)  # (per column: the BDs held)
        ports: dict[int, dict[int, int]] = {}  # (the port configurations since the last array reset, per tile)
        for e, addrs in items:
            blk = self.blocks[e]
            configure = self._configure(e, prev)
            if configure and self._quiet(prev, e, ports):
                pre, n_pre = self._before(prev, e)
                parts += [self.loadpdi[self.parity ^ 1], pre, blk.quiet.config[4:]]  # (the PDI loaded last: skipped)
                n_ops += blk.quiet.n_config + n_pre
                for tile, regs in blk.quiet.ports.items():
                    ports.setdefault(tile, {}).update(regs)
                state[:] = -1
            elif configure:
                cfg = blk.config.copy()
                cfg[:4] = self.loadpdi[self.parity]
                if prev is None:  # (a piece's first: the other PDI loaded first, so that this load is never skipped)
                    parts.append(self.loadpdi[self.parity ^ 1])
                    n_ops += 1
                self.parity ^= 1
                parts.append(cfg)
                n_ops += blk.n_config
                ports = {tile: dict(regs) for tile, regs in blk.ports.items()}
                state[:] = -1
            words, bd_words, n = (blk.run if configure else blk.run.guarded).emit(state)
            words = words.copy()
            addr = np.asarray(addrs, np.int64)[blk.bd_arg] + blk.bd_rel
            if addr.size and (addr.min() < 0 or addr.max() >= 1 << 48):
                raise txn.HardwareLimitError(f"item {len(ends)} (entry {e}): BD addresses {int(addr.min()):#x} .. "
                                             f"{int(addr.max()):#x} do not fit a BD's 48 bits")
            words[bd_words] = (addr & 0xFFFFFFFC).astype(np.uint32)
            words[bd_words + 1] = (words[bd_words + 1] & 0xFFFF0000) | (addr >> 32).astype(np.uint32)
            parts.append(words)
            n_ops += n
            ends.append(n_ops)
            prev = e
        out = np.concatenate(parts)
        out[2], out[3] = n_ops, 4 * out.size
        return out

    def bounds(self, items: Sequence[Item], splits: Collection[int], limit: int = PIECE_BYTES) -> list[int]:
        """The item bounds of a program's pieces: ``splits``, and a cut at the last change of design before a piece
        would exceed ``limit`` bytes."""
        def nbytes(i: int, prev: int | None) -> int:  # (as ``assemble`` emits it, all memory-tile BDs written)
            e = items[i][0]
            blk = self.blocks[e]
            if prev is None:  # (two PDI loads, then the configuration)
                return 4 * (4 + blk.config.size + blk.run.words.size)
            if self._configure(e, prev):
                quiet = blk.quiet.config.size + self._before(prev, e)[0].size \
                    if self.reuse and blk.quiet is not None and self.blocks[prev].quiet else 0
                return 4 * (max(blk.config.size, quiet) + blk.run.words.size)
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
            if size + n > limit and i > bounds[-1]:
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
