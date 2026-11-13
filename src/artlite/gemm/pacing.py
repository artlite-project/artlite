"""Pacing: the PACE word, and the paced cores' idle (``GemmQOperator(pace=)``, ``GemmQdOperator(pace=)``).

Some laptops' power delivery asserts PROCHOT while the array runs MMULs and streams weights at a prefill GEMM's rate (a
Krackan Point laptop: every Q4_K GEMM item, on any power source, within about a millisecond; the whole SoC then runs
at its lowest clocks, the GEMMs 2.3x slower). A paced design reads the scratchpad parameter ``PACE`` (a level per
class, ``PACE_BITS`` each) on the cores that set its rate (``pace_cores``): before each acquire of its input the core
idles ``PACE_IDLE[level]`` sixteenths of the cycles it worked since the previous acquire returned (the tile's timer),
and the array follows it through its FIFOs: duty 1 .. 0.38, with every core's current and the weight stream's. Level 0
runs the unpaced code (``prefill.pace``: the levels a machine needs, by timing).

A class is a GEMM's role (``PACE_ROLES``: the prefill's ``gemm.WHICH`` values) and its tile kind (planar or not).
"""

from collections.abc import Callable

from aie.dialects import arith
from aie.dialects.aie import T as IrT
from aie.iron.controlflow import else_, if_

PACE = "gq_pace"  # the scratchpad parameter of paced designs: PACE_BITS per class, class c at bit PACE_BITS c
PACE_BITS = 3
PACE_ROLES = ("qkv", "o", "gu", "down")  # a class per role and tile kind
PACE_CLASSES = tuple(f"{r}{k}" for r in PACE_ROLES for k in ("", ".planar"))  # class 2 i + planar
PACE_IDLE = (0, 2, 4, 6, 9, 13, 18, 26)  # sixteenths of its work a paced core idles per level: duty 1 .. 0.38
PACE_CAP = 1 << 16  # cycles: a longer interval between two acquires holds a pause (between dispatches), not work


def pace_class(which: str, planar_tiles: bool) -> int:
    """The pace class of a prefill GEMM of role ``which`` (``PACE_ROLES``) on planar tiles or not."""
    return 2 * PACE_ROLES.index(which) + int(planar_tiles)


def pace_word(levels: dict[int, int]) -> int:
    """The ``PACE`` parameter of the levels per class (0 .. len(PACE_IDLE) - 1)."""
    assert all(0 <= c < len(PACE_CLASSES) and 0 <= v < len(PACE_IDLE) for c, v in levels.items()), levels
    return sum(v << PACE_BITS * c for c, v in levels.items())


def pace_levels(word: int) -> dict[int, int]:
    """The nonzero levels per class of a ``PACE`` word."""
    m = (1 << PACE_BITS) - 1
    return {c: word >> PACE_BITS * c & m for c in range(len(PACE_CLASSES)) if word >> PACE_BITS * c & m}


class Paced:
    """A paced core's input FIFO (``pc``: the PACE parameter's handle, the core's timestamp buffer, the kernels
    ``_pace0`` / ``_pace1``; ``cls``): each acquire idles first (its level's share of the work since the previous one
    returned) and notes when it returned."""

    def __init__(self, h: object, pc: tuple, cls: int):
        self.h = h
        prm, self.st, self.k0, self.k1 = pc
        self.level = arith.andi(arith.shrui(prm.read(), arith.constant(IrT.i32(), PACE_BITS * cls)),
                                arith.constant(IrT.i32(), (1 << PACE_BITS) - 1))

    def acquire(self, n: int) -> object:
        self.k1(self.st, self.level)
        x = self.h.acquire(n)
        self.k0(self.st)
        return x

    def release(self, n: int) -> None:
        self.h.release(n)


def run_paced(pc: tuple | None, cls: int | None, body: Callable[[object], None], h: object) -> None:
    """A worker's item: ``body`` of its input FIFO handle h; on the core ``pc`` is given to (``_pace_args``) in two
    copies, the one of the paced handle at a nonzero level (level 0 runs the unpaced code: a branch per item)."""
    if pc is None:
        body(h)
        return
    w = Paced(h, pc, cls)
    with if_(arith.cmpi(arith.CmpIPredicate.ne, w.level, arith.constant(IrT.i32(), 0))) as branch:
        body(w)
    with else_(branch):
        body(h)
