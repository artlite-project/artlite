"""What a configuration leaves running, for the decision to load it without the array reset (``Quiet``): its stream
switch port configurations, whether a channel it starts can send before any lock release (``early``), whether a memory
tile DMA of it acquires or releases a neighbour's lock (``far``), and the locks its core tiles' DMA BDs take."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import txn
from ...npu_regs import BD_STRIDE, CORE_TILE, MEM_TILE, TileMap, tile_map
from . import ctrlpkt
from .bdelide import Run
from .prune import prune


@dataclass(eq=False)
class Quiet:
    """An entry's configuration on the array its predecessor left running, without the array reset (words 0 .. 3: a
    PDI load, the one loaded last when assembled: ``ctrlpkt.Loader(reused=True)``), its operation count and packets
    (``Block.packets``'s form), and what ``Assembler._quiet`` checks: its tiles with packets and cores, its port
    configurations (``ports``), whether a channel it starts can send before any lock release (``early``: the previous
    design's routes would take the data), whether a memory tile DMA of it acquires a neighbour's lock (``far``: that
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


def ports(words: np.ndarray) -> dict[int, dict[int, int]]:
    """The stream switch port configurations a transaction leaves: per tile (its base address) register -> value."""
    out: dict[int, dict[int, int]] = {}
    for p, op in txn.op_offsets(words):
        if op not in (txn.OP_WRITE, txn.OP_BLOCKWRITE, txn.OP_MASKWRITE):
            continue
        reg = int(words[p + 2])
        tile, off = reg & ~0xFFFFF, reg & 0xFFFFF
        n = int(words[p + 3]) // 4 - 4 if op == txn.OP_BLOCKWRITE else 1
        for i, v in enumerate(words[p + 4:p + 4 + n]):
            if any(a <= off + 4 * i < b for a, b in tile_map(txn.address_row(reg)).ports):
                regs = out.setdefault(tile, {})
                if op == txn.OP_MASKWRITE:
                    m = int(words[p + 5])
                    v = (regs.get(off, 0) & ~m) | (int(v) & m)
                regs[off + 4 * i] = int(v)
    return out


def values(writes: list[tuple[int, np.ndarray]]) -> dict[int, int]:
    """The register values a tile's moved writes (``ctrlpkt.split``'s) leave: tile-local address -> value (its BDs,
    channels and locks: not memory)."""
    out = {}
    for addr, words in writes:
        if addr >= CORE_TILE.bds[0]:
            out.update((addr + 4 * i, int(w)) for i, w in enumerate(words))
    return out


def _first_bds(v: dict[int, int], t: TileMap) -> list[int]:
    """The BDs the MM2S channels start (their start queues' BD ids) among the register values ``v``."""
    return [v[q] & t.start_bd_mask for q in t.start_queues[t.s2mm_channels:] if q in v]


def early(vals: dict[tuple[int, int], dict[int, int]]) -> bool:
    """Whether an MM2S channel the configuration starts (memory and core tiles) can send before any lock release: its
    first BD acquires no lock, or one whose initial value already grants it (or a neighbour memory tile's)."""
    for (c, r), v in vals.items():
        t = tile_map(r)
        word = t.lock_word.word
        for bd in _first_bds(v, t):
            f = t.bd_locks(v.get(t.bd_register(bd, word), 0))
            if not f.acq_enable or f.acq_id not in t.own_locks:
                return True
            init = v.get(t.own_lock_register(f.acq_id), 0)
            if init >= -f.acq_value if f.acq_value <= 0 else init == f.acq_value:
                return True
    return False


def far(vals: dict[tuple[int, int], dict[int, int]], run: Run) -> bool:
    """Whether a memory tile BD (of the configuration or its RunOp) acquires or releases a neighbour memory tile's lock
    (ids 0-63 west, 128-191 east)."""
    t, word = MEM_TILE, MEM_TILE.lock_word.word
    stride = BD_STRIDE
    words = [w for (_, r), v in vals.items() if r == 1
             for a, w in v.items() if t.bds[0] <= a < t.bds[1] and (a - t.bds[0]) % stride == 4 * word]
    for p, op in run.ops:
        reg = int(run.words[p + 2]) if op in (txn.OP_WRITE, txn.OP_BLOCKWRITE) else 0
        if txn.address_row(reg) == 1 and t.bds[0] <= reg & 0xFFFFF < t.bds[1]:
            n = int(run.words[p + 3]) // 4 - 4 if op == txn.OP_BLOCKWRITE else 1
            words += [int(w) for i, w in enumerate(run.words[p + 4:p + 4 + n])
                      if ((reg & 0xFFFFF) + 4 * i - t.bds[0]) % stride == 4 * word]
    for w in words:
        f = t.bd_locks(w)
        if f.valid and ((f.acq_enable and f.acq_id not in t.own_locks) or (f.rel_value and f.rel_id not in t.own_locks)):
            return True
    return False


def dma_locks(vals: dict[tuple[int, int], dict[int, int]]) -> dict[tuple[int, int], set[int]]:
    """Per core tile, the locks its valid DMA BDs acquire or release."""
    t, out = CORE_TILE, {}
    for tile, v in vals.items():
        if tile[1] >= 2:
            fields = [t.bd_locks(v.get(t.bd_register(bd, t.lock_word.word), 0)) for bd in range(t.n_bds)]
            out[tile] = {k for f in fields if f.valid
                         for k, on in ((f.acq_id, f.acq_enable), (f.rel_id, f.rel_value)) if on}
    return out


def quiet(load: np.ndarray, without: np.ndarray, pm: ctrlpkt.Writes, run: Run) -> Quiet:
    """An entry's ``Quiet`` (``without``, ``pm``: ``ctrlpkt.split`` of its configuration)."""
    loader = ctrlpkt.Loader(pm, reused=True)
    rest, left = prune(without, loader.ports())
    pre, _, at = loader.prelude(left + loader.slots(rest))
    rest = np.concatenate([pre, rest])
    vals = {t: values(w) for t, w in pm.items()}
    return Quiet(np.concatenate([load, rest]), 1 + len(txn.op_offsets(rest)), [(4 + w, p) for w, p in at],
                 frozenset(pm), frozenset(t for t in pm if t[1] >= 2), ports(rest), early(vals), far(vals, run),
                 dma_locks(vals))
