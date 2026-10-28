"""Pacing's host side (``gemm.gemm_q``'s PACE word, ``prefill.pace``): the word's encoding, ``ARTLITE_PACE``, and the
calibration's search on simulated machines (a paced run's time 1 / duty; a throttle that multiplies a class's time
while its duty is above a trip point)."""

from collections.abc import Callable

import pytest

from artlite.gemm.gemm_q import PACE_BITS, PACE_CLASSES, PACE_IDLE, pace_class, pace_levels, pace_word
from artlite.prefill import pace as P


def test_word() -> None:
    levels = {pace_class("gu", False): 3, pace_class("qkv", True): 7, pace_class("down", False): 1}
    w = pace_word(levels)
    assert pace_levels(w) == levels and w < 1 << 30  # (the scratchpad keeps 30 bits)
    assert len(PACE_CLASSES) * PACE_BITS <= 30 and len(PACE_IDLE) == 1 << PACE_BITS
    assert P.describe(w) == "qkv.planar 7, gu 3, down 1"
    assert pace_levels(0) == {} and P.describe(0) == ""


def test_parse() -> None:
    assert P.parse("") is None and P.parse("auto") is None and P.parse("recalibrate") is None
    assert P.parse("off") == {}
    assert P.parse("2") == dict.fromkeys(range(len(PACE_CLASSES)), 2)
    assert P.parse("gu=3, qkv.planar=1") == {pace_class("gu", False): 3, pace_class("qkv", True): 1}
    assert P.parse("9") == dict.fromkeys(range(len(PACE_CLASSES)), len(PACE_IDLE) - 1)
    with pytest.raises(ValueError, match="<class>=<level>"):
        P.parse("gate_up=3")


def _machine(base: dict[int, float], trip: dict[int, float], slow: float = 2.3
             ) -> tuple[Callable[[list[int] | None, int], float], list]:
    """measure() of a machine: class c's items take base[c] / duty, times ``slow`` while the duty is above trip[c]
    (no throttle: trip 1)."""
    calls = []

    def measure(classes: list[int] | None, word: int) -> float:
        calls.append((classes, word))
        levels = pace_levels(word)
        total = 0.0
        for c in classes or sorted(base):
            duty = 16 / (16 + PACE_IDLE[levels.get(c, 0)])
            total += base[c] / duty * (slow if duty > trip[c] else 1.0)
        return total

    return measure, calls


def test_search_unthrottled() -> None:
    """Nothing throttles: the quick check of the heaviest class, every level 0."""
    gu, qkv = pace_class("gu", False), pace_class("qkv", False)
    measure, calls = _machine({gu: 36.0, qkv: 9.0}, {gu: 1.0, qkv: 1.0})
    levels, table = P.search([gu, qkv], measure)
    assert levels == {} and len(calls) == 1 + len(P.QUICK) and table["quick"]["class"] == "gu"


def test_search_throttled() -> None:
    """Each class's fastest level is the first below its trip duty; a class that never trips stays at 0."""
    gu, qkv, down6 = pace_class("gu", False), pace_class("qkv", False), pace_class("down", True)
    measure, calls = _machine({gu: 36.0, qkv: 9.0, down6: 16.0}, {gu: 0.75, qkv: 0.95, down6: 1.0})
    levels, table = P.search([gu, down6, qkv], measure)
    duty = [16 / (16 + i) for i in PACE_IDLE]
    assert levels == {gu: next(L for L, d in enumerate(duty) if d <= 0.75),
                      qkv: next(L for L, d in enumerate(duty) if d <= 0.95)}
    assert table["classes"]["down.planar"]["level"] == 0 and "all" in table
    assert (None, pace_word(levels)) in calls  # (confirmed on every class's items together)


def test_search_unconfirmed() -> None:
    """Classes that gain alone but not together (the throttle follows the whole program): every level 0."""
    gu, qkv = pace_class("gu", False), pace_class("qkv", False)
    measure, _ = _machine({gu: 36.0, qkv: 9.0}, {gu: 0.75, qkv: 0.95})

    def together(classes: list[int] | None, word: int) -> float:
        return measure(classes, word) * (2.0 if classes is None and word else 1.0)

    levels, table = P.search([gu, qkv], together)
    assert levels == {} and table["all"]["s"]["levels"] > table["all"]["s"]["level 0"]
