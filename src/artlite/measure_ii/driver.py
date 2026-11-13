"""`measure`: compile a kernel and assemble its report."""

import hashlib
import shlex
from pathlib import Path

import yaml

from .compilation import compile_kernel, parse_stderr, resolve_clang_cmd
from .issues import collect_issues
from .match import enrich_from_assembly
from .remarks import parse_yaml_remarks
from .render import print_report
from .report import KernelReport, loop_name
from .trips import parse_source_trip, template_arguments

_REPORT_CACHE: dict[str, KernelReport] = {}


def measure(
    source: str | Path,
    *,
    clang_cmd: list[str] | None = None,
    verbose: bool = True,
    trip_count: int = 32,
) -> KernelReport:
    """Compile a kernel and report its final AIE loop schedules.

    Args:
        source: C++ source file.
        clang_cmd: full clang++ argument list (defaults to the Artlite set).
        verbose: print the formatted report.
        trip_count: fallback trip count used to evaluate loop costs when
            neither the ZOL iteration count nor the source for-header yields
            a trip count.
    """
    if trip_count < 0:
        raise ValueError("trip_count must be non-negative")
    source = Path(source)
    resolved_clang_cmd = resolve_clang_cmd(clang_cmd)

    # The same generated kernel.cc text is often re-measured for operators
    # that share a kernel but differ in MLIR; run the full report only once.
    source_key = hashlib.sha256(
        source.read_bytes() + repr(resolved_clang_cmd).encode()
    ).hexdigest()
    if source_key in _REPORT_CACHE:
        first = _REPORT_CACHE[source_key]
        if verbose:
            print(
                f"\nmeasure_ii: {source.absolute()} is textually identical to "
                f"{first.source.absolute()} (report above); skipping duplicate"
            )
        return first

    yaml_path = source.with_suffix(source.suffix + ".yaml")
    compilation = compile_kernel(
        source,
        clang_cmd=resolved_clang_cmd,
        yaml_path=yaml_path,
    )
    clean_path = source.with_suffix(source.suffix + ".s")
    annotated_path = source.with_suffix(source.suffix + ".annotated.s")
    report = KernelReport(
        source=source,
        cmd=shlex.join(compilation.annotated_command or compilation.clean_command),
        commands=[
            shlex.join(compilation.clean_command),
            *(
                [shlex.join(compilation.annotated_command)]
                if compilation.annotated_command
                else []
            ),
        ],
        returncode=compilation.returncode,
        yaml_path=yaml_path,
        clean_assembly=clean_path,
        annotated_assembly=annotated_path,
        trip_count=trip_count,
    )

    stderr_info = parse_stderr(compilation.stderr)
    report.missing_bank_annotations = stderr_info["missing_bank_annotations"]
    report.warnings = stderr_info["warnings"]
    report.pass_failed_warnings = stderr_info["pass_failed_warnings"]
    report.diagnostics = stderr_info["diagnostics"]

    if compilation.returncode:
        errors = [d for d in report.diagnostics if "error:" in d]
        report.issues.append(
            f"compiler failed with return code {compilation.returncode}"
            + "".join(f"\n      {e}" for e in errors[:5])
        )
        if verbose:
            print_report(report)
        return report

    if not yaml_path.exists():
        report.parse_error = "compiler completed without creating an optimization record"
        report.issues.append(report.parse_error)
    else:
        try:
            parsed = parse_yaml_remarks(yaml_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            report.parse_error = f"could not parse optimization record: {exc}"
            report.issues.append(report.parse_error)
        else:
            parsed.source = source
            parsed.cmd = report.cmd
            parsed.commands = report.commands
            parsed.returncode = report.returncode
            parsed.yaml_path = yaml_path
            parsed.clean_assembly = clean_path
            parsed.annotated_assembly = annotated_path
            parsed.trip_count = trip_count
            parsed.missing_bank_annotations += report.missing_bank_annotations
            parsed.warnings = report.warnings
            parsed.pass_failed_warnings = report.pass_failed_warnings
            parsed.diagnostics = report.diagnostics
            report = parsed

    if report.parse_error is None:
        enrich_from_assembly(report)
    for loop in report.loops:
        if loop.kernel_line is not None and loop.hw_iters is None:
            loop.source_trip = parse_source_trip(
                source, loop.kernel_line, template_arguments(source, loop.kernel_line, loop.function)
            )
    _mark_shared_blocks(report)
    collect_issues(report)

    if report.returncode == 0:
        _REPORT_CACHE[source_key] = report
    if verbose:
        print_report(report)
    return report


def _mark_shared_blocks(report: KernelReport) -> None:
    """Consecutive loops: the block between them is the first one's epilogue and the second one's prologue (the
    final remarks count it in both); it counts in the first one's cost only."""
    epilogues = {
        (loop.function, loop.epilogue_block): loop for loop in report.loops if loop.epilogue_block
    }
    for loop in report.loops:
        other = epilogues.get((loop.function, loop.prologue_block)) if loop.prologue_block else None
        if other is not None and loop.prologue_bundles:
            loop.prologue_of = loop_name(report, other)
            other.epilogue_of = loop_name(report, loop)
