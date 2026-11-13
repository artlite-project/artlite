"""What the report warns about: calls, spills, schedules, annotations."""

from .costs import CostModel
from .demangle import demangle_short
from .report import KernelReport, LoopReport


def calls_text(loop: LoopReport, costs: CostModel) -> str:
    """The calls of one iteration with the cycles of each."""
    counts: dict[str, int] = {}
    for callee in loop.callees:
        counts[callee] = counts.get(callee, 0) + 1
    parts = []
    for callee, count in counts.items():
        cycles = costs.function(callee)
        cost = "cost unknown" if cycles is None else f"{'<= ' if cycles.bound else ''}{cycles.value} cy"
        parts.append(f"{demangle_short(callee)}{f' x{count}' if count > 1 else ''} ({cost}{' each' * (count > 1)})")
    return ", ".join(parts)


def spill_text(loop: LoopReport) -> str:
    """The loop body's spill code, and the register allocator's count, with their units."""
    parts = []
    if loop.loop_spills:
        parts.append(f"{loop.loop_spills} spill store(s) ({loop.spill_bytes} B)")
    if loop.loop_reloads:
        parts.append(f"{loop.loop_reloads} reload load(s) ({loop.reload_bytes} B)")
    text = ", ".join(parts)
    if loop.regalloc_spills is not None:
        text += (
            f"; the register allocator: {loop.regalloc_spills} spill(s), {loop.regalloc_reloads} reload(s) of "
            "registers, inner loops included (a register wider than 64 B moves in 64-byte parts)"
        )
    return text


def collect_issues(report: KernelReport) -> None:
    costs = CostModel(report)
    for loop in report.loops:
        if loop.callees:
            report.issues.append(
                f"{loop.name}: {len(loop.callees)} call(s) per iteration ({calls_text(loop, costs)}); "
                "not in the bundle count"
            )
        if loop.loop_spills or loop.loop_reloads:
            report.issues.append(f"{loop.name}: spill code in the loop body: {spill_text(loop)}")
        if loop.sched == "pre-RA" and loop.ns > 4:
            report.issues.append(
                f"{loop.name}: pre-RA schedule with high stage count NS={loop.ns} "
                "(spill risk; consider AIE_PREPARE_FOR_POSTPIPELINING)"
            )
        if (
            loop.sched != "none"
            and loop.body_bundles is not None
            and loop.ii is not None
            and loop.body_bundles != loop.ii
        ):
            report.issues.append(
                f"{loop.name}: final II={loop.ii} but loop body has "
                f"{loop.body_bundles} bundles"
            )
    if report.missing_bank_annotations or report.missing_bank_remarks:
        report.issues.append(
            f"missing memory-bank annotations "
            f"(stderr warnings={report.missing_bank_annotations}, remarks={report.missing_bank_remarks}); "
            "annotate pointers with __aie_dm_resource_* to fix"
        )
    if report.pass_failed_warnings:
        report.issues.append(
            f"{len(report.pass_failed_warnings)} -Wpass-failed warning(s): a loop pragma was dropped"
        )
    for note in report.regalloc_spill_notes:
        report.issues.append(note)
    has_loop_blocks = any("for." in key.split("::")[-1] for key in report.blocks)
    if (
        not report.loops
        and has_loop_blocks
        and not report.parse_error
        and not report.returncode
    ):
        report.issues.append("loops present but no pipeline remarks found")
