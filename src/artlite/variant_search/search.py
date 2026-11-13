"""The search: variants compiled in parallel and measured, with the result types."""

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import parallel_compile
from ..measure_ii import KernelReport, LoopReport, measure
from ..measure_ii.demangle import demangle_short
from ..measure_ii.report import tree_order
from .pipeliner_log import PipelinerTrace, compiler_drift, debug_compiler, trace_postpipeliner
from .variants import Variant, render_variant


@dataclass
class RegionResult:
    """An event window of the NPU counters: per window, and its static cycles from measure_ii."""

    static: int | None = None
    cycles: float | None = None  # 33:34 / 33:33
    windows: float | None = None  # per dispatch and core
    memory_stall: float | None = None  # 23:23 per window
    lock_stall: float | None = None  # 26:26 per dispatch and core (also the host's counter readback)
    error: str | None = None


@dataclass
class NpuResult:
    median_us: float | None = None
    p10_us: float | None = None
    regions: dict[str, RegionResult] = field(default_factory=dict)
    error: str | None = None


@dataclass
class VariantResult:
    variant: Variant
    source: Path | None = None  # the compiled file
    kernel_source: str = ""
    clang_cmd: list[str] = field(default_factory=list)
    pragma_lines: list[int] = field(default_factory=list)
    same_as: str | None = None  # an earlier variant with the identical source and flags (its report is shared)
    report: KernelReport | None = None
    error: str | None = None
    traces: dict[tuple[str, int], PipelinerTrace] | None = None
    trace_error: str | None = None
    npu: NpuResult | None = None


@dataclass
class SearchResult:
    factory: Callable[..., Any]
    kwargs: dict[str, Any]
    functions: list[str]  # target function names (substrings)
    totals: list[str]  # functions reported by cycles per call only
    out_dir: Path
    variants: list[VariantResult]
    debug_compiler: str | None = None
    drift: str | None = None  # how the debug build differs from the wheel (compiler_drift), when traces were taken


def factory_label(factory: Callable[..., Any], kwargs: Mapping[str, Any]) -> str:
    return f"{getattr(factory, '__qualname__', repr(factory))}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"


def function_symbols(report: KernelReport, name: str) -> list[str]:
    """The function symbols of a report whose demangled name contains name, in assembly order."""
    names = [*report.functions, *(loop.function for loop in report.loops)]
    return [symbol for symbol in dict.fromkeys(names) if name in demangle_short(symbol)]


def target_loops(report: KernelReport, symbol: str) -> list[LoopReport]:
    """The loops of a function in nesting order (depth first, siblings by source line)."""
    return tree_order([loop for loop in report.loops if loop.function == symbol])


def _wants_trace(result: VariantResult, functions: Sequence[str]) -> bool:
    """A target loop of one block stayed unpipelined (the post-pipeliner tried and failed, or did not try)."""
    report = result.report
    if report is None:
        return False
    return any(
        loop.sched == "none" and loop.num_blocks == 1
        for name in functions
        for symbol in function_symbols(report, name)
        for loop in target_loops(report, symbol)
    )


def search(
    factory: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    functions: Sequence[str],
    variants: Sequence[Variant],
    totals: Sequence[str] = (),
    jobs: int | None = None,
    out_dir: str | Path | None = None,
    why: bool = False,
) -> SearchResult:
    """Render the variants of factory(**kwargs) (an ArtcOperator, or a function returning kernel source) and
    measure them in parallel (jobs at a time, default ``parallel_compile.jobs()``). functions: the target functions (name substrings) that take the pragmas and get a
    table per loop (a name as compiled matches without its hash: a patch renames what it changes, ``kernel_source``);
    totals: other functions reported by their cycles per call (e.g. the kernel entry point).
    The post-pipeliner trace (debug llvm-aie) is taken for variants with an unpipelined one-block target loop, or
    for all with why. A patch that does not apply raises ValueError before anything compiles."""
    from ..operators.base import CACHE_DIR

    functions = [re.sub(r"_[0-9a-f]{8}$", "", f) for f in functions]
    totals = [re.sub(r"_[0-9a-f]{8}$", "", f) for f in totals]
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"variant names must be unique and not empty: {names}")
    rendered = [render_variant(factory, kwargs, variant, functions=functions) for variant in variants]
    builds = [(r.source, r.clang_cmd) for r in rendered]
    key = hashlib.sha256(repr((factory_label(factory, kwargs), builds)).encode()).hexdigest()[:12]
    out = Path(out_dir) if out_dir is not None else Path(CACHE_DIR) / "variant_search" / key
    results = []
    by_source: dict[tuple[str, tuple[str, ...]], VariantResult] = {}
    for variant, item in zip(variants, rendered):
        directory = out / re.sub(r"[^\w.+-]", "_", variant.name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "kernel.cc"
        path.write_text(item.source)
        result = VariantResult(variant, path, item.kernel_source, item.clang_cmd, item.pragma_lines)
        first = by_source.setdefault((item.source, tuple(item.clang_cmd)), result)
        if first is not result:
            result.same_as = first.variant.name
        results.append(result)

    def compile_one(result: VariantResult) -> None:
        assert result.source is not None
        report = measure(result.source, clang_cmd=result.clang_cmd, verbose=False)
        if report.returncode:
            errors = [line for line in report.diagnostics if "error:" in line] or report.issues
            result.error = f"compile failed: {errors[0] if errors else report.returncode}"
        result.report = report

    unique = [result for result in results if result.same_as is None]
    parallel_compile.parallel_map(compile_one, unique, limit=jobs)
    compiler = debug_compiler()
    wanted = [result for result in unique if result.error is None and (why or _wants_trace(result, functions))]
    drift = compiler_drift(compiler) if wanted and compiler is not None else None
    if wanted and compiler is None:
        for result in wanted:
            result.trace_error = "no debug build of llvm-aie (set ARTLITE_DEBUG_CLANG to an assertions clang++)"
    elif wanted:
        assert compiler is not None

        def trace_one(result: VariantResult) -> None:
            assert result.source is not None
            result.traces, result.trace_error = trace_postpipeliner(
                result.source, result.clang_cmd, compiler=compiler
            )

        parallel_compile.parallel_map(trace_one, wanted, limit=jobs)
    firsts = {result.variant.name: result for result in unique}
    for result in results:
        if result.same_as is not None:
            first = firsts[result.same_as]
            result.report, result.error = first.report, first.error
            result.traces, result.trace_error = first.traces, first.trace_error
    return SearchResult(factory, dict(kwargs), list(functions), list(totals), out, results, compiler, drift)
