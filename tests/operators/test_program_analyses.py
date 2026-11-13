"""The analyses the library program assembler runs on a configuration (operators/fused/: ``prune``, ``quiet``,
``bdelide``) and the pieces a library keeps (``PieceCache``), on hand-built transactions (no NPU)."""

import numpy as np
import pytest

from artlite import txn
from artlite.npu_regs import CORE_TILE, MEM_TILE
from artlite.operators.fused import ctrlpkt, program, quiet
from artlite.operators.fused.library import PieceCache, _Piece
from artlite.operators.fused.prune import prune


def ops(cfg: np.ndarray) -> list[tuple[int, int, int]]:
    """(operation, register, first value) of each operation of a configuration."""
    return [(op, int(cfg[p + 2]), int(cfg[p + 4])) for p, op in txn.op_offsets(cfg)]


def core(reg: int) -> int:
    return txn.tile_address(0, 2, reg)


def test_prune_locks_and_ports() -> None:
    """A lock written the value it holds (0 after the reset) is dropped, the stream switch ports a tile's writes set
    before any DMA starts become one blockwrite per range (the ports between them 0)."""
    t = txn.Txn()
    lock = CORE_TILE.locks[0]
    for value in (0, 3, 3):
        t.write32(core(lock), value)
    north, south = CORE_TILE.north0_master, CORE_TILE.south0_slave
    t.write32(core(north), 0x80000001)
    t.write32(core(north + 8), 0x80000002)
    t.write32(core(south), 0x80000000)
    t.write32(core(CORE_TILE.start_queues[2]), 1)  # (the DMA starts: later port writes stay single writes)
    t.write32(core(north + 4), 0x80000003)
    cfg = t.words()[4:]
    rest, left = prune(cfg)
    got = ops(rest)
    assert left == [] and [r for _, r, _ in got].count(core(lock)) == 1 and (txn.OP_WRITE, core(lock), 3) in got
    merged = [(op, r) for op, r, _ in got if op == txn.OP_BLOCKWRITE]
    assert merged == [(txn.OP_BLOCKWRITE, core(north))]
    p, _ = next(x for x in txn.op_offsets(rest) if int(rest[x[0] + 2]) == core(north))
    assert list(rest[p + 4:p + 7]) == [0x80000001, 0, 0x80000002]
    assert (txn.OP_WRITE, core(south), 0x80000000) in got and (txn.OP_WRITE, core(north + 4), 0x80000003) in got
    assert len(got) < len(ops(cfg))


def test_prune_core_control() -> None:
    """A core control write that leaves its value after the reset (Reset set, Enable clear) is dropped; the enable stays."""
    t = txn.Txn()
    t.maskwrite32(core(CORE_TILE.control[0]), 2, 3)
    t.maskwrite32(core(CORE_TILE.control[0]), 1, 1)
    rest, _ = prune(t.words()[4:])
    assert [(op, v) for op, _, v in ops(rest)] == [(txn.OP_MASKWRITE, 1)]


def bd(lock: int, *, enable: bool = True, value: int = 1) -> dict[int, int]:
    """A core tile's BD 0 with the acquire of ``lock`` (``value``: the value it needs)."""
    return {CORE_TILE.bd_register(0, 5): 1 << 25 | enable << 12 | (value & 0x7F) << 5 | lock}


def test_quiet_early() -> None:
    """A channel that sends before any lock release: its BD acquires no lock, a lock that is not the tile's own, or one
    whose value grants it already."""
    start = {CORE_TILE.start_queues[2]: 0}  # (the first MM2S channel starts BD 0)
    tile = (0, 2)
    assert not quiet.early({tile: {**start, **bd(3), CORE_TILE.own_lock_register(3): 0}})  # (waits for a release)
    assert quiet.early({tile: {**start, **bd(3), CORE_TILE.own_lock_register(3): 1}})  # (value 1: granted)
    assert quiet.early({tile: {**start, **bd(3, enable=False)}})  # (no acquire)
    assert not quiet.early({tile: bd(3)})  # (no channel starts)
    mem = {MEM_TILE.start_queues[6]: 2, MEM_TILE.bd_register(2, 7): 1 << 31 | 1 << 15 | 1 << 8 | 70}
    assert not quiet.early({(0, 1): mem}) and quiet.early({(0, 1): {**mem, MEM_TILE.own_lock_register(70): 1}})
    west = {**mem, MEM_TILE.bd_register(2, 7): 1 << 31 | 1 << 15 | 1 << 8 | 10}  # (a neighbour's lock)
    assert quiet.early({(0, 1): west})


class NoRun:
    ops = ()


def test_quiet_far() -> None:
    """A memory tile BD that acquires or releases a neighbour memory tile's lock (ids 0-63, 128-191)."""
    reg = MEM_TILE.bd_register
    own = 1 << 31 | 1 << 15 | 70
    assert not quiet.far({(0, 1): {reg(3, 7): own}}, NoRun())
    assert quiet.far({(0, 1): {reg(3, 7): 1 << 31 | 1 << 15 | 10}}, NoRun())  # (acquire of the west's)
    assert quiet.far({(0, 1): {reg(3, 7): 1 << 31 | 1 << 24 | 130 << 16}}, NoRun())  # (release of the east's)
    assert not quiet.far({(0, 1): {reg(3, 7): 1 << 24 | 130 << 16}}, NoRun())  # (not a valid BD)
    assert not quiet.far({(0, 1): {reg(3, 7): 1 << 31 | 130 << 16 | 70}}, NoRun())  # (no release value: no release)
    assert not quiet.far({(0, 2): {reg(3, 7): 1 << 31 | 1 << 15 | 10}}, NoRun())  # (a core tile's: other registers)
    # the BD of a RunOp's blockwrite
    run = txn.Txn()
    run.blockwrite(txn.tile_address(0, 1, reg(2, 0)), [0] * 7 + [1 << 31 | 1 << 15 | 10])
    words = run.words()[4:]
    assert quiet.far({}, program.Run(words, 1, np.zeros(0, np.int64)))


def test_quiet_dma_locks() -> None:
    w = 1 << 25 | 4 << 13 | 2 << 18 | 1 << 12 | 9  # (valid; acquires 9, releases 4)
    got = quiet.dma_locks({(0, 2): {CORE_TILE.bd_register(1, 5): w, CORE_TILE.bd_register(2, 5): w & ~(1 << 25) | 7},
                           (0, 1): {}})
    assert got == {(0, 2): {9, 4}}


def test_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARTLITE_PDI_RESET", raising=False)
    monkeypatch.delenv("ARTLITE_PM_PACKETS", raising=False)
    assert ctrlpkt.enabled() and program.quiet_enabled()
    monkeypatch.setenv("ARTLITE_PM_PACKETS", "0")
    assert not ctrlpkt.enabled() and not program.quiet_enabled()
    monkeypatch.setenv("ARTLITE_PM_PACKETS", "1")
    monkeypatch.setenv("ARTLITE_PDI_RESET", "1")
    assert ctrlpkt.enabled() and not program.quiet_enabled()


def test_piece_cache() -> None:
    cache = PieceCache(100)
    items = [(0, [1])]
    piece = _Piece(items, np.zeros(10, np.uint32), 1, [3])
    assert cache.get(items, 0, 1, 0) is None
    cache.put(0, 1, 0, piece)
    assert cache.get(items, 0, 1, 0) is piece and cache.get(items, 0, 1, 1) is None and cache.get(list(items), 0, 1, 0) is None
    cache.put(0, 1, 1, _Piece(items, np.zeros(20, np.uint32), 0, [3]))  # (40 + 80 B: over the limit: the first is dropped)
    assert cache.get(items, 0, 1, 0) is None and cache.get(items, 0, 1, 1) is not None
    assert cache.end(items, 0, 1) is None
    cache.note_end(items, 0, 1, 0.5)
    cache.note_end(items, 0, 1, 0.7)
    assert cache.end(items, 0, 1) == 0.5 and cache.end(list(items), 0, 1) is None
