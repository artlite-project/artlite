"""The RunOp of a library entry (``Run``): its control code, as it runs again right after its own design (``guarded``)
and without the memory-tile BD writes the registers already hold (``emit``)."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ... import txn
from ...npu_regs import MEM_TILE

_QUEUE_STATUS = {q: s for d in txn.QUEUE for q, s in zip(txn.QUEUE[d], txn.QUEUE_STATUS[d])}
_QUEUE_FULL = txn.QUEUE_DEPTH << 20  # (Task_Queue_Size, bits 22:20)


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
            if op != txn.OP_WRITE or txn.address_row(reg) != 0 or (status := _QUEUE_STATUS.get(reg & 0xFFFFF)) is None:
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
        if op != txn.OP_BLOCKWRITE or txn.address_row(addr) != 1 or not MEM_TILE.bds[0] <= addr & 0xFFFFF < MEM_TILE.bds[1]:
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
        iterated = vals.get(addr + 4 * MEM_TILE.iteration_word, 0) & MEM_TILE.iteration_mask
        for r, v in vals.items():
            col.append(r >> 25)
            reg.append(((r & 0xFFFFF) - MEM_TILE.bds[0]) // 4)
            val.append(-1 if iterated else v)
        k = j
    keys = list(zip(col, reg))
    last = sorted({key: i for i, key in enumerate(keys)}.values())
    a = functools.partial(np.array, dtype=np.int64)
    return _BdUnits(a(start), a(end), a(nops), a(first), a(col), a(reg), a(val), len(set(keys)) != len(keys), a(last))
