"""The AIE2P array's registers: the one place that knows their addresses (tile registers, then performance counters).

Tile registers (``TileMap``: ``CORE_TILE``, ``MEM_TILE``, ``SHIM_TILE``, ``tile_map(row)``): where a tile kind's locks,
DMA BD file, DMA channels, program and data memory, core control and stream switch ports are, and the fields of a BD's
lock word (``BdLocks``) that the program assembler reads. The shim's DMA BDs, queues and the words of a BD are
``txn``'s. ``tests/test_npu_regs.py`` compares every constant here with aie-rt's header. Offsets are within a tile
(``txn.tile_address`` adds its column and row).

Performance counters:

The map is aie-rt's ``xaie2pgbl_params.h``, not AM029's (Versal's layout: its core offsets differ, its memtile and
shim have 6 counters). A counter module has up to 4 counters; a start / stop control register serves two of them, and
the counters' value registers follow each other 4 bytes apart. Reading an address past a module's counters (no
register there, e.g. 0x91030) wedges the hardware context: ``CounterModule.address`` refuses such an index.

- ``CORE``: a core tile's core module (4 counters), ``MEM``: its memory module (the 64 KB data memory with locks and
  DMAs, 2 counters), ``MEMTILE``: the memory tile's module (row 1, 4), ``SHIM``: the shim tile's PL module (row 0, 2).
- ``window_counter_writes``: the control writes of the event0 / event1 windows of a kernel (``event0()`` /
  ``event1()``) that the timing builds of the prefill and MoE operators program.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

SHIM_ROW = 0  # the tile rows of the NPU's array (the compute rows follow the memory tile's)
MEMTILE_ROW = 1

EVENT_TRUE = 1  # (a core event that is always on: a counter between two of them counts all cycles)
EVENT_INSTR_0 = 33  # the kernel's event0()
EVENT_INSTR_1 = 34  # event1()
EVENT_LOCK_STALL = 26  # (core module)
EVENT_MEMORY_STALL = 23


# --- tile registers ---------------------------------------------------------------------------------------------------

LOCK_STRIDE = 16  # a lock's value register: 16 B apart
BD_STRIDE = 32  # a DMA BD: 8 words
CORE_CONTROL_RESET = 0x2  # (a core's control register after a reset: Reset set, Enable clear; AM025)
SWITCH_ENABLE, SWITCH_PACKET = 1 << 31, 1 << 30  # a stream switch port configuration's enable and packet-mode bits


class BdLocks(NamedTuple):
    """The lock fields of a DMA BD (its acquire and release): ``acq_value`` signed, ``rel_value`` 0: no release."""

    valid: bool
    acq_enable: bool
    acq_id: int
    acq_value: int
    rel_id: int
    rel_value: int


class _LockWord(NamedTuple):
    """Where a BD's lock fields are: its word, and the least significant bits of each (ids: width, values: 7 bits)."""

    word: int
    valid: int
    acq_enable: int
    acq_id_width: int
    acq_value: int
    rel_id: int
    rel_value: int


@dataclass(frozen=True)
class TileMap:
    """The registers of one tile kind (a row's: ``tile_map``); ranges are [first register, end)."""

    name: str
    locks: tuple[int, int] | None  # lock value registers, ``LOCK_STRIDE`` apart
    own_locks: range | None  # the ids a BD's lock fields give this tile's own locks (a memory tile's: 64-127)
    bds: tuple[int, int] | None  # the DMA BD file, ``BD_STRIDE`` a BD
    iteration_word: int | None  # a BD's word with its iteration, whose count the DMA advances (its mask below)
    start_bd_mask: int  # the BD id bits of a channel's start queue
    dma: tuple[int, int] | None  # the DMA channel registers, 8 B a channel: control, start queue (S2MM, then MM2S)
    s2mm_channels: int
    program: tuple[int, int] | None  # a core's program memory
    control: tuple[int, int] | None  # a core's control register (enable: bit 0)
    data_bytes: int | None  # the data memory, from 0
    masters: tuple[int, int]  # the stream switch's master and slave port configurations, one word a port
    slaves: tuple[int, int]
    slot0: int  # the slave slots: 4 words a slave port, in the ports' order
    south0_slave: int | None  # its slave South 0's configuration
    south0_slots: int | None  # its slots
    tile_ctrl_master: int | None  # its master TileControl
    north0_master: int  # its master North 0
    lock_word: _LockWord | None
    iteration_mask: int = 0

    @property
    def channel_ctrl(self) -> tuple[int, ...]:
        """The DMA channel control registers (Reset: bit 1), S2MM first; the start queue is 4 above each."""
        lo, hi = self.dma
        return tuple(range(lo, hi, 8))

    @property
    def ports(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """The master and slave port configuration ranges (contiguous, 0 after a reset)."""
        return self.masters, self.slaves

    @property
    def n_bds(self) -> int:
        return (self.bds[1] - self.bds[0]) // BD_STRIDE

    @property
    def start_queues(self) -> tuple[int, ...]:
        return tuple(r + 4 for r in self.channel_ctrl)

    @property
    def mm2s_ctrl(self) -> int:
        """The first MM2S channel's control register."""
        return self.dma[0] + 8 * self.s2mm_channels

    def own_lock_register(self, lock: int) -> int:
        """The value register of this tile's own lock ``lock`` (its id in a BD's lock fields)."""
        return self.locks[0] + LOCK_STRIDE * (lock - self.own_locks.start)

    def bd_register(self, bd: int, word: int) -> int:
        return self.bds[0] + BD_STRIDE * bd + 4 * word

    def bd_locks(self, w: int) -> BdLocks:
        """The lock fields of the BD word ``w`` (word ``lock_word.word`` of the BD)."""
        f = self.lock_word
        signed = lambda v: v - 128 if v & 0x40 else v  # (7 bits)
        return BdLocks(bool(w >> f.valid & 1), bool(w >> f.acq_enable & 1), w & ((1 << f.acq_id_width) - 1),
                       signed(w >> f.acq_value & 0x7F), w >> f.rel_id & ((1 << f.acq_id_width) - 1),
                       w >> f.rel_value & 0x7F)


CORE_TILE = TileMap(
    "core", locks=(0x1F000, 0x1F100), own_locks=range(16), bds=(0x1D000, 0x1D200), iteration_word=None, start_bd_mask=0xF,
    dma=(0x1DE00, 0x1DE20), s2mm_channels=2, program=(0x20000, 0x24000), control=(0x32000, 0x32004), data_bytes=0x10000,
    masters=(0x3F000, 0x3F05C), slaves=(0x3F100, 0x3F164), slot0=0x3F200, south0_slave=0x3F114, south0_slots=0x3F250,
    tile_ctrl_master=0x3F00C, north0_master=0x3F034, lock_word=_LockWord(5, 25, 12, 4, 5, 13, 18))
MEM_TILE = TileMap(
    "memtile", locks=(0xC0000, 0xC0400), own_locks=range(64, 128), bds=(0xA0000, 0xA0600), iteration_word=6, start_bd_mask=0x3F,
    dma=(0xA0600, 0xA0660), s2mm_channels=6, program=None, control=None, data_bytes=0x80000,
    masters=(0xB0000, 0xB0044), slaves=(0xB0100, 0xB0148), slot0=0xB0200, south0_slave=0xB011C, south0_slots=0xB0270,
    tile_ctrl_master=0xB0018, north0_master=0xB002C, lock_word=_LockWord(7, 31, 15, 8, 8, 16, 24),
    iteration_mask=(1 << 23) - 1)  # (word 6: Iteration_Current 28:23, which the DMA advances, Wrap 22:17, Stepsize 16:0)
SHIM_TILE = TileMap(
    "shim", locks=None, own_locks=None, bds=None, iteration_word=None, start_bd_mask=0, dma=None, s2mm_channels=2, program=None,
    control=None, data_bytes=None, masters=(0x3F000, 0x3F058), slaves=(0x3F100, 0x3F15C), slot0=0x3F200,
    south0_slave=None, south0_slots=None, tile_ctrl_master=None, north0_master=0x3F030, lock_word=None)
SHIM_MUX, SHIM_MUX_SOUTH3_DMA = 0x1F000, 0x400  # (the shim's mux: South 3 from its DMA; the loader's route)
SHIM_SLAVE_SOUTH3 = 0x3F114  # (the slave port South 3's configuration; its port number is 5)


def tile_map(row: int) -> TileMap:
    """The register map of the tiles of row ``row`` (0: shim, 1: memory tile, above: core tiles)."""
    return (SHIM_TILE, MEM_TILE, CORE_TILE)[min(row, 2)]


@dataclass(frozen=True)
class CounterModule:
    name: str  # "core" | "mem" | "mt" | "sh"
    events: str  # the module's event enum in ``aie.utils.trace.events``
    ctrl: tuple[int, ...]  # control registers (start and stop events), each for two counters
    base: int  # counter 0's value register
    max_counters: int
    tiles: str  # the class of tiles that carry the module: "core" (core tiles), "mt" (row 1), "sh" (row 0)

    def address(self, index: int) -> int:
        """The value register of counter ``index``."""
        if not 0 <= index < self.max_counters:
            raise ValueError(f"the {self.name} module has {self.max_counters} counters, not a counter {index}")
        return self.base + 4 * index

    def ctrl_writes(self, pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """(register, word) of the control registers that program the counters ``pairs[i] = (start, stop)`` events."""
        if len(pairs) > self.max_counters:
            raise ValueError(f"the {self.name} module has {self.max_counters} counters, not {len(pairs)}")
        return [
            (self.ctrl[group], ctrl_word(*pairs[2 * group : 2 * group + 2]))
            for group in range((len(pairs) + 1) // 2)
        ]


CORE = CounterModule("core", "CoreEvent", (0x31500, 0x31504), 0x31520, 4, "core")
MEM = CounterModule("mem", "MemEvent", (0x11000,), 0x11020, 2, "core")
MEMTILE = CounterModule("mt", "MemTileEvent", (0x91000, 0x91004), 0x91020, 4, "mt")
SHIM = CounterModule("sh", "ShimTileEvent", (0x31000,), 0x31020, 2, "sh")
MODULES = {module.name: module for module in (CORE, MEM, MEMTILE, SHIM)}


def ctrl_word(first: tuple[int, int], second: tuple[int, int] = (0, 0)) -> int:
    """A control register's word: the (start, stop) events of its two counters."""
    return first[0] | first[1] << 8 | second[0] << 16 | second[1] << 24


def window_counter_writes(*, stall: int = EVENT_LOCK_STALL, memory_events: Sequence[int] = ()) -> list[tuple[int, int]]:
    """(register, word) writes that program a core's counters around its event0 / event1 windows: counter 0 the
    windows' cycles, 1 the ``stall`` event's cycles, 2 all cycles, 3 the windows. ``memory_events`` (one or two): the
    memory module's counters 0 and 1 count them (pulses) or their cycles (levels)."""
    writes = [
        (CORE.ctrl[0], ctrl_word((EVENT_INSTR_0, EVENT_INSTR_1), (stall, stall))),
        (CORE.ctrl[1], ctrl_word((EVENT_TRUE, EVENT_TRUE), (EVENT_INSTR_0, EVENT_INSTR_0))),
    ]
    if memory_events:
        first, last = memory_events[0], memory_events[-1]
        writes.append((MEM.ctrl[0], ctrl_word((first, first), (last, last))))
    return writes
