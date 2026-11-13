"""The AIE2P performance-counter registers: the one place that knows their addresses.

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

SHIM_ROW = 0  # the tile rows of the NPU's array (the compute rows follow the memory tile's)
MEMTILE_ROW = 1

EVENT_TRUE = 1  # (a core event that is always on: a counter between two of them counts all cycles)
EVENT_INSTR_0 = 33  # the kernel's event0()
EVENT_INSTR_1 = 34  # event1()
EVENT_LOCK_STALL = 26  # (core module)
EVENT_MEMORY_STALL = 23


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
