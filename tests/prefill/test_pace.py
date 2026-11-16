"""Pacing's host side (``gemm.pacing``'s PACE word, ``prefill.pace``): the word's encoding, ``ARTLITE_PACE``, and the
calibration's search on simulated machines (a paced run's time 1 / duty; a throttle that multiplies a class's time
while its duty is above a trip point)."""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from artlite.gemm.pacing import PACE_BITS, PACE_CLASSES, PACE_IDLE, PACE_ROLES, pace_class, pace_levels, pace_word
from artlite.operators import bfp16
from artlite.prefill import gemm as PG
from artlite.prefill import pace as P


def test_word() -> None:
    levels = {pace_class("gu", False): 3, pace_class("qkv", True): 7, pace_class("down", False): 1}
    w = pace_word(levels)
    assert pace_levels(w) == levels and w < 1 << 30  # (the scratchpad keeps 30 bits)
    assert len(PACE_CLASSES) * PACE_BITS <= 30 and len(PACE_IDLE) == 1 << PACE_BITS
    assert P.describe(w) == "qkv.planar 7, gu 3, down 1"
    assert pace_levels(0) == {} and P.describe(0) == ""


def test_every_gemm_role_is_a_pace_role() -> None:
    assert set(PG.WHICH.values()) == set(PACE_ROLES)


def test_parse() -> None:
    assert P.parse("") is None and P.parse("auto") is None and P.parse("recalibrate") is None
    assert P.parse("off") == {}
    assert P.parse("2") == dict.fromkeys(range(len(PACE_CLASSES)), 2)
    assert P.parse("gu=3, qkv.planar=1") == {pace_class("gu", False): 3, pace_class("qkv", True): 1}
    assert P.parse("9") == dict.fromkeys(range(len(PACE_CLASSES)), len(PACE_IDLE) - 1)
    with pytest.raises(ValueError, match="<class>=<level>"):
        P.parse("gate_up=3")


def _machine(base: dict[int, float], trip: dict[int, float], slow: float = 2.3, bound: dict[int, float] | None = None
             ) -> tuple[Callable[[list[int] | None, int], float], list]:
    """measure() of a machine: class c's items take base[c] / duty, times ``slow`` while the duty is above trip[c]
    (no throttle: trip 1); ``bound``: a class's floor (its time when the paced core is not what bounds it, as a planar
    GemmQd's dequantizing cores: pacing hardly changes it)."""
    calls = []

    def measure(classes: list[int] | None, word: int) -> float:
        calls.append((classes, word))
        levels = pace_levels(word)
        total = 0.0
        for c in classes or sorted(base):
            duty = 16 / (16 + PACE_IDLE[levels.get(c, 0)])
            total += max(base[c] / duty, (bound or {}).get(c, 0.0)) * (slow if duty > trip[c] else 1.0)
        return total

    return measure, calls


def test_search_unthrottled() -> None:
    """Nothing throttles: each class at level 0 and the probe levels, every level 0, a note that says so."""
    gu, qkv = pace_class("gu", False), pace_class("qkv", False)
    measure, calls = _machine({gu: 36.0, qkv: 9.0}, {gu: 1.0, qkv: 1.0})
    notes: list[str] = []
    levels, table = P.search([gu, qkv], measure, notes.append)
    assert levels == {} and len(calls) == 2 * (1 + len(P.PROBE)) and "all" not in table
    assert all(e["level"] == 0 for e in table["classes"].values())
    assert len(notes) == 1 and "unthrottled" in notes[0] and "unpaced" in notes[0]


def test_search_largest_never_trips() -> None:
    """A class with as much work as the largest, first, never trips and pacing hardly changes its time (Llama 3.1 8B
    Q4_K_M's Q6_K down, Qwen3.8 27B Q4_K_M's Q6_K qkv: a planar GemmQd's dequantizing cores bound it); the others trip
    at full rate: they are found and paced, it stays at 0."""
    gu, qkv, down6 = pace_class("gu", False), pace_class("qkv", False), pace_class("down", True)
    measure, _ = _machine({gu: 118.0, qkv: 6.2, down6: 30.0}, {gu: 0.95, qkv: 0.95, down6: 1.0},
                          bound={down6: 117.0})
    notes: list[str] = []
    levels, table = P.search([down6, gu, qkv], measure, notes.append)
    assert levels == {gu: 1, qkv: 1} and table["classes"]["down.planar"]["level"] == 0
    assert "throttles" in notes[0] and "down.planar" not in notes[0] and notes[-1] == "paced: qkv 1, gu 1"


def test_search_deep_trip() -> None:
    """A stricter board: level 1 still trips, the probe at level 5 finds the throttle, the scan the fastest level."""
    gu = pace_class("gu", False)
    measure, _ = _machine({gu: 36.0}, {gu: 0.7})
    levels, _ = P.search([gu], measure)
    duty = [16 / (16 + i) for i in PACE_IDLE]
    assert levels == {gu: next(L for L, d in enumerate(duty) if d <= 0.7)}


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


def test_search_deeper_together() -> None:
    """Classes unthrottled alone at their levels that trip together (a trip duty lower for the whole program): every
    class one level deeper."""
    gu, qkv = pace_class("gu", False), pace_class("qkv", False)
    alone, _ = _machine({gu: 36.0, qkv: 9.0}, {gu: 0.9, qkv: 0.9})
    together, _ = _machine({gu: 36.0, qkv: 9.0}, {gu: 0.85, qkv: 0.85})

    def measure(classes: list[int] | None, word: int) -> float:
        return (together if classes is None else alone)(classes, word)

    levels, table = P.search([gu, qkv], measure)
    duty = [16 / (16 + i) for i in PACE_IDLE]
    first = next(L for L, d in enumerate(duty) if d <= 0.9)
    assert duty[first] > 0.85 and levels == {gu: first + 1, qkv: first + 1}
    assert table["all"]["s"]["deeper"] < table["all"]["s"]["levels"]


def test_pick() -> None:
    """A class's items: one of each distinct GEMM first, then round robin, each GEMM's items one after the other."""
    a, b, c = [("a", i) for i in range(5)], [("b", i) for i in range(3)], [("c", 0)]
    assert P.pick([a, b, c], 4) == [("a", 0), ("a", 1), ("b", 0), ("c", 0)]
    assert P.pick([a, b, c], 2) == [("a", 0), ("b", 0), ("c", 0)]  # (every GEMM at least once)
    assert P.pick([c], 4) == [("c", 0)] and P.pick([a], 4) == a[:4]


def test_item_params() -> None:
    """A whole-chunk GEMM's item reads its token chunks (an MoE layer's router: ``moe_t``), in ``mc_unit`` tokens
    or the cores' own chunks; other GEMMs' read none."""
    from types import SimpleNamespace

    from artlite.gemm.gemm_q import GemmQOperator
    from artlite.moe import gemm as MG

    def op(**kw: object) -> SimpleNamespace:
        return SimpleNamespace(**{"mc_param": "", "M": 1024, "mc_unit": 0, "nrows": 2, "m": 64, **kw})

    assert GemmQOperator.item_params(op(mc_param=MG.CHUNK_PARAM, mc_unit=MG.CHUNK_UNIT)) == {MG.CHUNK_PARAM: 4}
    assert GemmQOperator.item_params(op(mc_param="n")) == {"n": 8} and GemmQOperator.item_params(op()) == {}


def test_activations() -> None:
    """Random activation panels: valid bfp16ebs8 blocks of N(0, 1) values, any length."""
    b = P.activations(9 * 4096 + 5)
    assert b.dtype == np.uint8 and b.size == 9 * 4096 + 5
    v = bfp16.block_values(b[:9 * 4096].reshape(-1, 9))
    assert abs(float(v.std()) - 1) < 0.05 and abs(float(v.mean())) < 0.05 and np.count_nonzero(v) > 0.99 * v.size


def test_items_buffers() -> None:
    """The calibration's dispatches are given the buffers its items address (the weights, its two scratch buffers,
    allocated together), not every buffer of the prefill: on the XRT backend a command lists at most 5 XRT buffers."""
    listed: list[set[str]] = []

    class Buffer:
        def __init__(self, size: int) -> None:
            self.size = size

        def write(self, data: np.ndarray) -> None:
            assert data.size == self.size

        def sync_to_device(self) -> None:
            pass

    class Device:
        def alloc_host_buffers(self, sizes: list[int]) -> list[Buffer]:
            return [Buffer(n) for n in sizes]

    class Library:
        def items(self, runlist: list, buffers: dict) -> list:
            listed.append(set(buffers))
            return runlist

    op = SimpleNamespace(pace=pace_class("qkv", False), mc_param=False, M=256, K=256, N=256, item_params=dict,
                         get_arg_spec=lambda: [SimpleNamespace(nbytes=n) for n in (1024, 2048, 4096)])
    pf = SimpleNamespace(sizes=(256,), runlists={256: [(op, "PXN[0:1024]", "W[0:2048]", "G[0:4096]")]},
                         ov=SimpleNamespace(dev=SimpleNamespace(device=Device())), lib=Library(),
                         _buffers=lambda: dict.fromkeys(("W", "KV", "GSTATE", "F", "PXN", "G")))
    _, _, buffers, T = P._items(pf)
    assert T == 256 and set(buffers) == {"W", "PACE_A", "PACE_C"} and listed == [{"W", "PACE_A", "PACE_C"}]
    assert (buffers["PACE_A"].size, buffers["PACE_C"].size) == (1024, 4096)
