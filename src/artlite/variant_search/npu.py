"""NPU runs of the variants: one process per build, the time of a dispatch and the event-window counters."""

import ast
import functools
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import parallel_compile
from ..measure_ii import KernelReport, call_cycles
from ..profiling import ENV_PREFIX
from .search import NpuResult, RegionResult, SearchResult, VariantResult, function_symbols
from .variants import Variant, render_variant

COUNTERS = "33:34,33:33,23:23,26:26"  # window cycles, windows, memory stall, lock stall
_LABELS = {
    "INSTR_EVENT_0->INSTR_EVENT_1 cycles": "cycles",
    "INSTR_EVENT_0->INSTR_EVENT_0 count": "windows",
    "MEMORY_STALL->MEMORY_STALL count": "memory",
    "LOCK_STALL->LOCK_STALL count": "lock",
}


def _importable(factory: Callable[..., Any]) -> str:
    """module:qualname of the factory, checked to import back to it (a worker process rebuilds the operator)."""
    path = f"{factory.__module__}:{factory.__qualname__}"
    module, _, qualname = path.partition(":")
    try:
        found = functools.reduce(getattr, qualname.split("."), importlib.import_module(module))
    except (ImportError, AttributeError):
        found = None
    if found is not factory:
        raise ValueError(f"the NPU runs need an importable operator class, not {path}")
    return path


def parse_regions(specs: Sequence[str]) -> dict[str, list[tuple[str, int]]]:
    """NAME[=FN[*N][+FN[*N]...]]: the event window of the operator's win parameter NAME, whose static cycles are
    N calls of each FN (a function name substring)."""
    regions: dict[str, list[tuple[str, int]]] = {}
    for spec in specs:
        name, _, terms = spec.partition("=")
        regions[name] = []
        for term in filter(None, terms.split("+")):
            fn, _, count = term.partition("*")
            regions[name].append((fn, int(count or 1)))
    return regions


def _static(report: KernelReport, terms: Sequence[tuple[str, int]]) -> int | None:
    total = 0
    for name, count in terms:
        symbols = function_symbols(report, name)
        cycles = call_cycles(report, symbols[0]) if len(symbols) == 1 else None
        if cycles is None:
            return None
        total += count * cycles[0]
    return total if terms else None


def _read_counters(directory: Path) -> dict[str, list[list[int]]]:
    """Per counter (cycles / windows / memory / lock) the samples of each tile, the first one dropped."""
    samples: dict[str, dict[tuple[int, int], list[int]]] = {}
    for path in directory.glob("*_counters.json"):
        events = json.loads(path.read_text())
        labels = {(event["pid"], event["tid"]): event["args"]["name"] for event in events
                  if event.get("ph") == "M" and event.get("name") == "thread_name"}
        for event in events:
            if event.get("ph") != "C":
                continue
            label = _LABELS.get(labels.get((event["pid"], event["tid"]), event.get("name", "")))
            if label is not None:
                samples.setdefault(label, {}).setdefault((event["pid"], event["tid"]), []).append(
                    int(event["args"]["value"])
                )
    return {label: [values[1:] for values in tiles.values()] for label, tiles in samples.items()}


def region_result(directory: Path) -> RegionResult:
    counters = _read_counters(directory)
    if not counters.get("windows") or not any(counters["windows"]):
        return RegionResult(error="no counter samples")
    windows = sum(map(sum, counters["windows"]))
    samples = max(len(values) for values in counters["windows"])
    tiles = sum(1 for values in counters["windows"] if sum(values))
    if not windows:
        return RegionResult(error="no event windows (does the kernel mark this region?)")
    return RegionResult(
        cycles=sum(map(sum, counters.get("cycles", []))) / windows,
        windows=windows / samples / max(tiles, 1),
        memory_stall=sum(map(sum, counters.get("memory", []))) / windows if "memory" in counters else None,
        lock_stall=sum(map(sum, counters.get("lock", []))) / samples / max(tiles, 1) if "lock" in counters else None,
    )


@dataclass
class _Build:
    result: VariantResult
    region: str | None  # None: the timing build
    directory: Path
    spec: Path | None = None
    error: str | None = None


def _worker_env(build: _Build) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(ENV_PREFIX) and key != "ARTLITE_MEASURE_II"}
    if build.region is not None:
        env.update({ENV_PREFIX: "counters", f"{ENV_PREFIX}_DIR": str(build.directory / "profile"),
                    f"{ENV_PREFIX}_COUNTERS": COUNTERS, f"{ENV_PREFIX}_SAMPLE_EVERY": "1"})
    return env


def run_npu(
    result: SearchResult,
    *,
    reps: int = 30,
    counter_reps: int = 6,
    regions: Mapping[str, Sequence[tuple[str, int]]] | None = None,
    win_param: str = "win",
    jobs: int | None = None,
    compile_timeout: float = 1800,
) -> None:
    """Time each variant on the NPU and read its event-window counters per region (the operator's win_param set
    to the region name), filling VariantResult.npu. Every build compiles first in its own process (in parallel,
    with a timeout: no NPU context yet), then runs in its own process, one at a time, without a timeout (a
    process that holds an NPU context must not be killed)."""
    path = _importable(result.factory)
    if isinstance(result.factory(**result.kwargs), str):
        raise TypeError("the NPU runs need an operator, not a function returning kernel source")
    regions = dict(regions or {})
    builds: list[_Build] = []
    for item in result.variants:
        item.npu = NpuResult()
        if item.error is not None:
            item.npu.error = item.error
            continue
        if item.same_as is not None:
            continue
        for region in [None, *regions]:
            directory = item.source.parent / ("npu" if region is None else f"npu_{region}")  # type: ignore[union-attr]
            build = _Build(item, region, directory)
            builds.append(build)
            directory.mkdir(parents=True, exist_ok=True)
            variant = item.variant
            if region is not None:
                variant = Variant(variant.name, variant.pragma, variant.subs, variant.defines,
                                  {**variant.params, win_param: region}, variant.loops)
            try:
                rendered = render_variant(result.factory, result.kwargs, variant, functions=result.functions)
            except (ValueError, TypeError) as exc:
                build.error = str(exc)
                continue
            (directory / "kernel_src.cc").write_text(rendered.kernel_source)
            build.spec = directory / "spec.json"
            build.spec.write_text(json.dumps({
                "factory": path,
                "kwargs": repr({**result.kwargs, **variant.params}),
                "source": str(directory / "kernel_src.cc"),
                "reps": reps if region is None else counter_reps,
            }))

    def compile_one(build: _Build) -> None:
        if build.spec is None:
            return
        try:
            process = subprocess.run(
                [sys.executable, "-m", "artlite.variant_search", "--worker", str(build.spec), "--compile-only"],
                env=_worker_env(build), capture_output=True, text=True, timeout=compile_timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            build.error = f"compile timed out after {compile_timeout:.0f} s"
            return
        if process.returncode:
            build.error = "compile failed: " + (process.stderr.strip().splitlines() or ["?"])[-1]

    parallel_compile.parallel_map(compile_one, builds, limit=jobs)
    for build in builds:  # one NPU job at a time, each in its own process
        npu = build.result.npu
        assert npu is not None
        report = build.result.report
        static = _static(report, regions[build.region]) if build.region is not None and report else None
        if build.error is None and build.spec is not None:
            if build.region is not None:
                for old in (build.directory / "profile").glob("*_counters.json"):
                    old.unlink()
            print(f"  npu: {build.result.variant.name}{f' [{build.region}]' if build.region else ''} ...",
                  file=sys.stderr, flush=True)
            process = subprocess.run(  # no timeout: never kill a process that holds an NPU context
                [sys.executable, "-m", "artlite.variant_search", "--worker", str(build.spec)],
                env=_worker_env(build), capture_output=True, text=True, check=False,
            )
            if process.returncode:
                build.error = "run failed: " + (process.stderr.strip().splitlines() or ["?"])[-1]
            elif build.region is None:
                line = next((ln for ln in process.stdout.splitlines() if ln.startswith("RESULT ")), None)
                if line is None:
                    build.error = "no result from the worker"
                else:
                    times = sorted(json.loads(line[len("RESULT "):])["times"])
                    npu.median_us = statistics.median(times) * 1e6
                    npu.p10_us = times[int(0.1 * (len(times) - 1))] * 1e6
            else:
                npu.regions[build.region] = region_result(build.directory / "profile")
                npu.regions[build.region].static = static
        if build.error is not None:
            if build.region is None:
                npu.error = build.error
            else:
                npu.regions[build.region] = RegionResult(static=static, error=build.error)
    for item in result.variants:
        if item.same_as is not None:
            first = next(other for other in result.variants if other.variant.name == item.same_as)
            item.npu = first.npu


def run_worker(spec_path: str, *, compile_only: bool) -> None:
    """Rebuild the operator with the variant's source, compile it, and (unless compile_only) dispatch it."""
    spec = json.loads(Path(spec_path).read_text())
    module, _, qualname = spec["factory"].partition(":")
    factory = functools.reduce(getattr, qualname.split("."), importlib.import_module(module))
    operator = factory(**ast.literal_eval(spec["kwargs"]))
    source = Path(spec["source"]).read_text()
    operator.get_kernel_src = lambda: source  # (the instance attribute shadows the method)
    operator.compile()
    if compile_only:
        return
    import numpy as np

    import artlite

    device = operator.device  # opens the HW context
    rng = np.random.default_rng(0)
    buffers = []
    for arg in operator.get_arg_spec():
        if arg.direction == "in":
            n = math.prod(arg.shape)
            if arg.dtype.kind == "f" or arg.dtype == artlite.bfloat16:
                data = (rng.standard_normal(n, np.float32) * 0.5).astype(arg.dtype)
            else:
                data = rng.integers(0, 100, n).astype(arg.dtype)
            buffers.append(artlite.from_numpy(data.reshape(arg.shape), device))
        else:
            buffers.append(artlite.empty(arg.shape, dtype=arg.dtype, device=device))
    for _ in range(2):
        operator.dispatch(*buffers)
    times = []
    for _ in range(spec["reps"]):
        start = time.perf_counter()
        operator.dispatch(*buffers)
        times.append(time.perf_counter() - start)
    print("RESULT " + json.dumps({"times": times}), flush=True)
