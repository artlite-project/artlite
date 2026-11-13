"""The cost model: cycles of a loop entry and of a call, from the schedules and trip counts."""

from __future__ import annotations

from dataclasses import dataclass

from .report import KernelReport, LoopReport, fmt_int


@dataclass(frozen=True)
class _Cycles:
    value: int
    bound: bool = False  # branches: every path counted, an upper bound

    def __add__(self, other: _Cycles) -> _Cycles:
        return _Cycles(self.value + other.value, self.bound or other.bound)


class CostModel:
    """Cycle estimates (a bundle is a cycle) of loops and of calls to the kernel's functions.

    One iteration of a loop: its II (its own bundles on all paths for a branch
    loop), all iterations of its inner loops (their prologue and epilogue blocks
    are the enclosing loop's own blocks) and its calls. A call: the function's
    code outside loops, its loops and its calls. None when a trip count or a
    callee (outside the kernel, recursion) is unknown."""

    def __init__(self, report: KernelReport) -> None:
        self.report = report
        headers = {(loop.function, loop.header) for loop in report.loops if loop.header}
        self.children: dict[tuple[str, str | None], list[LoopReport]] = {}
        for loop in report.loops:
            parent = loop.parent if (loop.function, loop.parent) in headers else None
            self.children.setdefault((loop.function, parent), []).append(loop)
        self.calls_of: dict[str, _Cycles | None] = {}

    def nested(self, loop: LoopReport) -> list[LoopReport]:
        return self.children.get((loop.function, loop.header), []) if loop.header else []

    def function(self, name: str) -> _Cycles | None:
        """One call of a function of the kernel."""
        if name not in self.calls_of:
            self.calls_of[name] = None  # (while it is being evaluated: recursion is unknown)
            outside = self.report.outside_loops.get(name)
            if outside is not None:
                bundles, callees, branches = outside
                parts = [self.steady(loop, calls=True) for loop in self.children.get((name, None), [])]
                parts += [self.function(callee) for callee in callees]
                if None not in parts:
                    self.calls_of[name] = sum(parts, _Cycles(bundles, branches))
        return self.calls_of[name]

    def has_calls(self, loop: LoopReport) -> bool:
        return bool(loop.callees) or any(self.has_calls(child) for child in self.nested(loop))

    def calls(self, loop: LoopReport) -> _Cycles | None:
        """The calls of one iteration."""
        parts = [self.function(callee) for callee in loop.callees]
        return None if None in parts else sum(parts, _Cycles(0))

    def terms(self, loop: LoopReport, *, calls: bool) -> tuple[list[str], _Cycles | None]:
        """One iteration: its terms as text (II or body, inner loops, calls) and their sum."""
        body = loop.ii if loop.ii is not None else loop.body_bundles
        terms = [fmt_int(body)]
        parts: list[_Cycles | None] = [None if body is None else _Cycles(body, loop.branches)]
        nested = [self.steady(child, calls=calls) for child in self.nested(loop)]
        if nested:
            terms.append("inner loops" if None in nested else str(sum(cycles.value for cycles in nested)))
            parts += nested
        if calls and loop.callees:
            cycles = self.calls(loop)
            terms.append("calls" if cycles is None else str(cycles.value))
            parts.append(cycles)
        return terms, None if None in parts else sum(parts, _Cycles(0))

    def steady(self, loop: LoopReport, *, calls: bool) -> _Cycles | None:
        """All iterations of one entry into the loop."""
        iters = loop.effective_iters()
        _, iteration = self.terms(loop, calls=calls)
        if iters is None or iteration is None:
            return None
        return _Cycles(iters * iteration.value, iteration.bound or exits_early(loop))


def exits_early(loop: LoopReport) -> bool:
    """The loop may leave before its trip count: the count is a maximum (a ZOL's count is exact)."""
    return loop.hw_iters is None and (loop.early_exit or (loop.branches and loop.num_blocks == 1))


def loop_cycles(report: KernelReport, loop: LoopReport) -> tuple[int, bool] | None:
    """Cycles of one entry into the loop, as its printed cost (prologue + iterations x (II or body, inner loops,
    calls) + epilogue; without the calls when a callee's cost is unknown), and whether that is an upper bound
    (branches, an early exit). None without a trip count."""
    costs = CostModel(report)
    _, iteration = costs.terms(loop, calls=True)
    if iteration is None:
        _, iteration = costs.terms(loop, calls=False)
    iters = loop.effective_iters()
    if iters is None or iteration is None:
        return None
    total = loop.prologue_cycles + iters * iteration.value + (loop.epilogue_bundles or 0)
    return total, iteration.bound or exits_early(loop)


def call_cycles(report: KernelReport, function: str) -> tuple[int, bool] | None:
    """Cycles of one call of a function of the kernel (its code outside loops, its loops and its calls) and
    whether that is an upper bound. None when a trip count or a callee is unknown."""
    cycles = CostModel(report).function(function)
    return None if cycles is None else (cycles.value, cycles.bound)
