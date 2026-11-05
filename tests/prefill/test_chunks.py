"""``ChunkPrefill.pack`` (the packed chunks' planner) on synthetic estimates, no NPU: a chunk of T tokens costs a fixed
part (its ConfigureOps, the weights' pass) plus per token, and per token squared (the attention)."""

import itertools
import random
from types import SimpleNamespace

import pytest

from artlite.prefill.chunks import ChunkPrefill, Packing


class _Fake(ChunkPrefill):
    SUB, MAX_T, MIN_TOKENS, NAME = 256, 4096, 3, "fake"

    def __init__(self, sizes: tuple[int, ...], *, fixed: float = 1.0, tok: float = 0.01, sq: float = 0.0,
                 span: int = 4096, unit: int = 1):
        self.ov = SimpleNamespace(max_positions=1 << 16, SLOTS=4)
        self.sizes, self.pack_sizes, self.tree_sizes = sizes, sizes, sizes
        self.group, self.span, self.unit, self.pace, self._est, self._last_pack = 32, span, unit, None, {}, None
        self.fixed, self.tok, self.sq = fixed, tok, sq

    def _fits_budget(self, T: int, p0: int) -> bool:
        return True

    def chunk_seconds(self, T: int, p0: int) -> float:
        return self.fixed + self.tok * T + self.sq * T * T

    def packed_seconds(self, T: int, p0: int) -> float:
        return self.chunk_seconds(T, p0)

    def seg_unit(self, p0: int) -> int:
        return self.unit if p0 else 1


SIZES = (64, 128, 256, 512, 1024, 2048)


def _check(pf: _Fake, lengths: list[int], plan: Packing, p0: int = 0) -> None:
    """Every input once; each bin within its size (segments rounded up to the unit, the last one not); the estimate
    the sum of its parts and never above every input alone."""
    seen = sorted([i for _, b in plan.bins for i in b] + plan.alone)
    assert seen == list(range(len(lengths)))
    u = pf.seg_unit(p0)
    for T, b in plan.bins:
        assert len(b) > 1 and T in pf.packed_sizes(p0)
        assert sum(-(-lengths[i] // u) * u for i in b) <= T
        assert max(lengths[i] for i in b) <= pf.span
    parts = sum(pf.packed_seconds(T, p0) for T, _ in plan.bins) + sum(pf.alone_seconds(lengths[i], p0) for i in plan.alone)
    assert plan.seconds == pytest.approx(parts)
    assert plan.sequential == pytest.approx(sum(pf.alone_seconds(n, p0) for n in lengths))
    assert plan.seconds <= plan.sequential + 1e-12


@pytest.mark.parametrize("sq", [0.0, 1e-5, 1e-4])
@pytest.mark.parametrize("p0", [0, 256])
def test_never_above_sequential(sq: float, p0: int) -> None:
    """Many requests (lengths from 3 to 700, 1 to 12 inputs; a ConfigureOp-bound and an attention-bound cost model):
    never above every input alone, and every input exactly once."""
    pf = _Fake(SIZES, sq=sq, span=512, unit=16)
    rng = random.Random(5)
    for _ in range(200):
        lengths = [rng.randint(3, 700) for _ in range(rng.randint(1, 12))]
        _check(pf, lengths, pf.pack(lengths, p0), p0)


def test_ffd() -> None:
    """First-fit-decreasing: 300 + 200 + 100 + 50 tokens in one 1024-token chunk when the chunks' fixed cost dominates
    (four alone: four fixed parts); the sizes' smallest that holds them."""
    pf = _Fake(SIZES, fixed=1.0, tok=1e-4)
    plan = pf.pack([100, 300, 50, 200], 0)
    _check(pf, [100, 300, 50, 200], plan)
    assert plan.bins == [(1024, [0, 1, 2, 3])] and plan.alone == []
    plan = pf.pack([100, 120, 30], 0)  # (250 tokens: one chunk of 256)
    assert plan.bins == [(256, [0, 1, 2])]


def test_alone_when_not_faster() -> None:
    """A chunk is packed only where its estimate is below its inputs alone: per-token costs only (no fixed part),
    nothing gains from packing, everything runs alone."""
    pf = _Fake(SIZES, fixed=0.0, tok=1e-3)
    lengths = [64, 64, 128]
    plan = pf.pack(lengths, 0)
    _check(pf, lengths, plan)
    assert plan.bins == [] and plan.alone == [0, 1, 2]


def test_span_and_min_tokens() -> None:
    """An input longer than ``span`` runs alone (the packed attention's window); a chunk of fewer than MIN_TOKENS valid
    tokens does not run packed."""
    pf = _Fake(SIZES, span=512)
    lengths = [600, 100, 100]
    plan = pf.pack(lengths, 0)
    _check(pf, lengths, plan)
    assert 0 in plan.alone and plan.bins == [(256, [1, 2])]
    plan = pf.pack([1, 1], 0)
    assert plan.bins == [] and plan.alone == [0, 1]


def test_smaller_sizes() -> None:
    """Each bin's inputs also in smaller chunks: with a quadratic attention cost, 8 inputs of 250 tokens run as four
    512-token chunks rather than one 2048-token chunk (FFD fills the largest size first)."""
    pf = _Fake(SIZES, fixed=1.0, tok=1e-4, sq=4e-6)
    lengths = [250] * 8
    plan = pf.pack(lengths, 0)
    _check(pf, lengths, plan)
    assert sorted(T for T, _ in plan.bins) == [512] * 4
    one = pf.packed_seconds(2048, 0)
    assert plan.seconds < one


def test_seg_unit() -> None:
    """After a prefix (a recurrent model's tree chunks) segments start at multiples of ``seg_unit``: 3 x 20 tokens,
    each rounded up to 32, take a 128-token chunk; from position 0 a 64-token one."""
    pf = _Fake(SIZES, fixed=1.0, tok=1e-4, unit=16)
    assert pf.pack([20, 20, 20], 0).bins == [(64, [0, 1, 2])]
    plan = pf.pack([20, 20, 20], 256)
    _check(pf, [20, 20, 20], plan, 256)
    assert plan.bins == [(128, [0, 1, 2])]


def test_plan_kept() -> None:
    """The last plan is kept (the engine estimates a request, then runs it): the same object for the same request, a
    new one for another."""
    pf = _Fake(SIZES)
    a = pf.pack([100, 50], 64)
    assert pf.pack([100, 50], 64) is a
    assert pf.pack([100, 50], 0) is not a


def test_exhaustive_small() -> None:
    """Every request of 4 inputs from a few lengths: the plan against the best partition into packed chunks and inputs
    alone (first-fit-decreasing is not optimal in general; on these the optimum, 1.000x)."""
    pf = _Fake(SIZES, fixed=1.0, tok=1e-3, sq=2e-6, span=1024)

    def best(lengths: list[int]) -> float:
        out = sum(pf.alone_seconds(n, 0) for n in lengths)
        idx = list(range(len(lengths)))
        for r in range(2, len(lengths) + 1):
            for c in itertools.combinations(idx, r):
                f = sum(lengths[i] for i in c)
                T = next((T for T in SIZES if T >= f), None)
                if T is None:
                    continue
                rest = [lengths[i] for i in idx if i not in c]
                out = min(out, pf.packed_seconds(T, 0) + (best(rest) if rest else 0.0))
        return out

    worst = 1.0
    for lengths in itertools.combinations_with_replacement([30, 100, 250, 500, 900], 4):
        plan = pf.pack(list(lengths), 0)
        _check(pf, list(lengths), plan)
        worst = max(worst, plan.seconds / best(list(lengths)))
    print(f"FFD vs the best packed / alone partition: at most {worst:.3f}x")
    assert worst < 1.01
