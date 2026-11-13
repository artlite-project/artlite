"""The printed ranking: a table per loop of each target function, the NPU's timings."""

import re
from collections.abc import Sequence
from typing import Any

from ..measure_ii import SLOTS, KernelReport, LoopReport, SlotCounts, call_cycles, count_slots, loop_cycles
from ..measure_ii.demangle import demangle_short
from ..measure_ii.report import loop_where
from .npu import COUNTERS
from .patch import POST
from .pipeliner_log import PipelinerTrace
from .search import SearchResult, VariantResult, factory_label, function_symbols, target_loops


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], *, indent: str = "  ") -> list[str]:
    """Columns padded to their widths; the first column left-aligned, the others right-aligned."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    out = []
    for row in [header, *rows]:
        cells = [cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)]
        out.append(indent + "  ".join(cells).rstrip())
    return out


def _cycles_text(value: tuple[int, bool] | None) -> str:
    return "-" if value is None else f"{'<=' if value[1] else ''}{value[0]}"


def _pragma_name(pragma: str | None) -> str:
    """The shorthand of a variant's pragma (the inverse of pragma_text)."""
    if pragma is None:
        return ""
    names = []
    for macro in re.findall(r"[A-Za-z_]\w*(?:\([^)]*\))?", pragma):
        match = re.fullmatch(r"AIE_TRY_INITIATION_INTERVAL\((\d+)\)", macro)
        names.append("post" if macro == POST else f"ii{match.group(1)}" if match else macro)
    return "+".join(names) or "none"


def _loop_row(result: VariantResult, loop: LoopReport) -> tuple[list[str], SlotCounts]:
    report = result.report
    assert report is not None
    slots = count_slots(loop.asm_body)
    binding = set(slots.binding)
    sched = loop.sched if loop.num_blocks == 1 else f"none/{loop.num_blocks}bb"
    if loop.hw_iters is not None:
        peeled = loop.ns - 1 if loop.sched != "none" else 0
        iters = f"{loop.hw_iters}+{peeled}" if peeled else str(loop.hw_iters)
    else:
        iters = "-" if loop.source_trip is None else f"{loop.source_trip}s"
    spills = f"{loop.loop_spills}/{loop.loop_reloads}" if loop.loop_spills or loop.loop_reloads else "-"
    stack = report.stack_bytes.get(loop.function)
    pragma = _pragma_name(result.variant.pragma) if loop.kernel_line in result.pragma_lines else ""
    return [
        _row_name(result),
        pragma,
        sched,
        "-" if loop.ii is None else str(loop.ii),
        str(loop.ns) if loop.sched != "none" else "-",
        iters,
        _cycles_text(loop_cycles(report, loop)),
        "-" if loop.body_bundles is None else str(loop.body_bundles),
        spills,
        "-" if stack is None else str(stack),
        *[f"{slots.count(slot)}{'*' if slot in binding else ''}" for slot in SLOTS],
        _port_text(slots, binding),
        f"{slots.bound} {'/'.join(sorted(binding))}" if binding else "-",
    ], slots


def _port_text(slots: SlotCounts, binding: set[str]) -> str:
    """The busiest shared port: 'rm 46*' (binding: above every slot)."""
    busiest = slots.busiest_port
    if busiest is None:
        return "-"
    port, count = busiest
    return f"{port} {count}{'*' if port in binding else ''}"


def _rank_key(result: VariantResult, value: tuple[int, bool] | None, ii_value: int | None) -> tuple[Any, ...]:
    return (result.error is not None, value is None, value[0] if value else 0, ii_value or 1 << 30)


def _trace_for(result: VariantResult, loop: LoopReport) -> PipelinerTrace | None:
    if result.traces is None or loop.sched_block is None:
        return None
    return result.traces.get((loop.function, loop.sched_block))


def print_report(result: SearchResult) -> None:
    """Print the summary table (cycles per call of the functions) and one ranked table per target loop."""
    print(f"\nvariant search: {factory_label(result.factory, result.kwargs)}")
    print(f"  targets: {', '.join(result.functions)}; sources and reports in {result.out_dir}")
    for item in result.variants:
        if item.error:
            print(f"  ✗ {item.variant.name}: {item.error}")
    ok = [(item, item.report) for item in result.variants if item.report is not None and item.error is None]
    if not ok:
        return
    names = list(dict.fromkeys(
        demangle_short(symbol)
        for name in [*result.functions, *result.totals]
        for _, report in ok
        for symbol in function_symbols(report, name)
    ))
    rows = []
    for item, report in ok:
        symbols = {demangle_short(symbol): symbol for symbol in report.functions}
        cells = [_cycles_text(call_cycles(report, symbols[name])) if name in symbols else "-" for name in names]
        stack = max((report.stack_bytes.get(symbol, 0) for symbol in report.functions), default=0)
        code = sum(block.bytes for blocks in report.layout.values() for block in blocks)
        rows.append([_row_name(item), *cells, str(stack), str(code)])
    print("\ncycles per call (calls included; <=: every branch counted)")
    for line in _table(["variant", *names, "max stack", "code B"], rows):
        print(line)
    for name in result.functions:
        symbols = list(dict.fromkeys(symbol for _, report in ok for symbol in function_symbols(report, name)))
        if not symbols:
            print(f"\n{name}: no such function in the compiled code (inlined?)")
        for symbol in symbols:
            loops_of = [(item, report, target_loops(report, symbol)) for item, report in ok]
            count = max(len(loops) for _, _, loops in loops_of)
            if not count:
                print(f"\n{demangle_short(symbol)}: no loops")
            for index in range(count):
                _print_loop_table(symbol, index, count, loops_of)
    if result.debug_compiler is None and any(item.trace_error for item, _ in ok):
        print("\n  (post-pipeliner traces need a debug build of llvm-aie: set ARTLITE_DEBUG_CLANG)")
    if result.drift:
        print(f"\n  ⚠ post-pipeliner traces may differ from the shipped schedule: {result.drift}")
    if any(item.npu is not None for item in result.variants):
        _print_npu(result)


def _row_name(item: VariantResult) -> str:
    return item.variant.name + (f" (= {item.same_as})" if item.same_as else "")


def _loop_notes(item: VariantResult, loop: LoopReport, slots: SlotCounts) -> list[str]:
    """Detail lines of a variant's loop: the binding slot's ops, the post-pipeliner trace, calls."""
    notes = [f"{slot} {slots.pressure[slot]}: {slots.describe(slot)}" for slot in slots.binding]
    if slots.ops.get("?"):
        notes.append(f"unknown mnemonics: {', '.join(slots.ops['?'])}")
    trace = _trace_for(item, loop)
    if trace is not None:
        described = trace.describe()
        notes += [f"post-pipeliner (debug llvm-aie): {described[0]}", *described[1:]]
        if trace.final is not None and loop.sched == "none":
            notes.append(f"  ⚠ the debug build pipelines it at II {trace.final.ii}: it differs from the wheel")
    elif item.trace_error and loop.sched == "none" and loop.num_blocks == 1:
        notes.append(f"post-pipeliner: {item.trace_error}")
    elif loop.sched == "none" and loop.num_blocks > 1:
        reasons = "; ".join(loop.prereject_reasons) or "branches"
        notes.append(f"{loop.num_blocks} blocks: never pipelined ({reasons})")
    if loop.loop_calls:
        notes.append(f"calls per iteration: {', '.join(loop.loop_calls)}")
    return notes


def _print_loop_table(
    symbol: str, index: int, count: int, loops_of: list[tuple[VariantResult, KernelReport, list[LoopReport]]]
) -> None:
    entries = [(item, report, loops[index]) for item, report, loops in loops_of if index < len(loops)]
    item0, _, first = entries[0]
    where = loop_where(entries[0][1], first) if first.kernel_line or first.source_ref else ""
    print(
        f"\n{demangle_short(symbol)} loop {index + 1}/{count}: {first.block}{where}, depth {first.depth} "
        f"(in {item0.variant.name}); ranked by cycles per entry"
    )
    entries.sort(key=lambda entry: _rank_key(entry[0], loop_cycles(entry[1], entry[2]), entry[2].ii))
    header = ["variant", "pragma", "sched", "II", "NS", "iters", "cycles", "body", "spill", "stack", *SLOTS, "port",
              "res bound"]
    rows, details = [], []
    for item, _, loop in entries:
        row, slots = _loop_row(item, loop)
        rows.append(row)
        if item.same_as is None:
            details.append((item.variant.name, _loop_notes(item, loop, slots)))
    for line in _table(header, rows):
        print(line)
    for name, notes in details:
        for k, note in enumerate(notes):
            print(f"  {name if k == 0 else ' ' * len(name)}{':' if k == 0 else ' '} {note}")


def _fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _print_npu(result: SearchResult) -> None:
    items = [item for item in result.variants if item.npu is not None]
    print("\nNPU: one dispatch, each variant in its own process (median, p10 over the reps)")
    rows = [[item.variant.name, _fmt(item.npu.median_us, 1), _fmt(item.npu.p10_us, 1), item.npu.error or ""]  # type: ignore[union-attr]
            for item in sorted(items, key=lambda item: (item.npu.median_us is None, item.npu.median_us or 0))]  # type: ignore[union-attr]
    for line in _table(["variant", "median us", "p10 us", ""], rows):
        print(line)
    regions = list(dict.fromkeys(region for item in items for region in item.npu.regions))  # type: ignore[union-attr]
    for region in regions:
        print(f"\nNPU region {region}: cycles per event window (counters {COUNTERS}) vs static (measure_ii)")
        rows = []
        for item in items:
            found = item.npu.regions.get(region)  # type: ignore[union-attr]
            if found is None:
                continue
            ratio = found.cycles / found.static if found.cycles and found.static else None
            rows.append([
                item.variant.name, _fmt(found.static and float(found.static)), _fmt(found.cycles),
                "-" if ratio is None else f"{ratio:.2f}", _fmt(found.memory_stall), _fmt(found.windows),
                _fmt(found.lock_stall), found.error or "",
            ])
        for line in _table(["variant", "static", "measured", "ratio", "mem stall", "windows/disp",
                            "lock stall/disp", ""], rows):
            print(line)
    if regions:
        print("  (lock stall per dispatch and core includes the host's counter readback between dispatches)")
