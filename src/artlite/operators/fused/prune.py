"""A library configuration with fewer operations (``prune``): each costs the firmware ~0.1 us."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from ... import txn
from ...npu_regs import CORE_CONTROL_RESET, CORE_TILE, MEM_TILE, tile_map

_START_QUEUES = {0: (*txn.QUEUE[txn.S2MM], *txn.QUEUE[txn.MM2S]), 1: MEM_TILE.start_queues,
                 2: CORE_TILE.start_queues}  # (shim, memory tile, core: DMA start queues)
_CORE_CONTROL = CORE_TILE.control[0]
_CORE_DMA_CTRL = CORE_TILE.channel_ctrl  # (a core's DMA channel controls: Reset is bit 1)


def prune(cfg: np.ndarray, zero: Collection[int] = ()) -> tuple[np.ndarray, list[int]]:
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
        rng = next(r for r in tile_map(txn.address_row(reg)).ports if r[0] <= reg & 0xFFFFF < r[1])
        zeros.setdefault((reg & ~0xFFFFF, rng[0]), set()).add(reg)
    for k, (p, op) in enumerate(ops):
        if op not in (txn.OP_WRITE, txn.OP_MASKWRITE):
            continue
        reg, v = int(cfg[p + 2]), int(cfg[p + 4])
        tile, off, row = reg & ~0xFFFFF, reg & 0xFFFFF, txn.address_row(reg)
        if op == txn.OP_WRITE:
            if off in _START_QUEUES.get(min(row, 2), ()):
                started.add(tile)
            lock = (row >= 2 and CORE_TILE.locks[0] <= off < CORE_TILE.locks[1]) or (row == 1 and MEM_TILE.locks[0] <= off < MEM_TILE.locks[1])
            if lock and off % 16 == 0 and tile not in started:
                if locks.get(reg, 0) == v:
                    drop.add(k)
                locks[reg] = v
            rng = next((r for r in tile_map(row).ports if r[0] <= off < r[1]), None)
            if rng is not None and not started:
                ks, vals = ports.setdefault((tile, rng[0]), ([], {}))
                ks.append(k)
                vals[reg] = v
            continue
        m = int(cfg[p + 5])
        if any(r[0] <= off < r[1] for r in tile_map(row).ports):
            masked.add(tile)
        if row >= 2 and off == _CORE_CONTROL:
            old = control.get(reg, CORE_CONTROL_RESET)
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
