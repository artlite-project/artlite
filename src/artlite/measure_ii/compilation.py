"""The two compiles of a kernel (clean, and with optimization remarks) and the compiler's stderr."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REMARK_PASSES = (
    "pipeliner|postpipeliner|aie-hardware-loops|aie-asm-printer|"
    "aie-multi-slot-pseudo|asm-printer|regalloc|prologepilog|loop-vectorize"
)


@dataclass
class _Compilation:
    clean_command: list[str]
    annotated_command: list[str] | None
    clean_stdout: str
    clean_stderr: str
    clean_returncode: int
    annotated_stdout: str = ""
    annotated_stderr: str = ""
    annotated_returncode: int = 0

    @property
    def stderr(self) -> str:
        return self.clean_stderr + self.annotated_stderr

    @property
    def returncode(self) -> int:
        return self.clean_returncode or self.annotated_returncode


COMPILE_TIMEOUT = 120  # seconds (a fused library's translation unit takes ~30 s, ~45 s with 16 compiles at once)


def _run_compiler(command: list[str]) -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        def text(b: bytes | str | None) -> str:  # (the exception's output is bytes even with text=True)
            return b.decode(errors="replace") if isinstance(b, bytes) else b or ""

        return text(exc.stdout), text(exc.stderr) + f"compiler timed out after {COMPILE_TIMEOUT} seconds\n", 124
    return result.stdout or "", result.stderr or "", result.returncode


def compile_kernel(
    source: Path,
    *,
    clang_cmd: list[str],
    yaml_path: Path,
) -> _Compilation:
    clean_path = source.with_suffix(source.suffix + ".s")
    annotated_path = source.with_suffix(source.suffix + ".annotated.s")
    for path in (clean_path, annotated_path, yaml_path):
        path.unlink(missing_ok=True)

    clean_command = [*clang_cmd, "-S", str(source), "-o", str(clean_path)]
    clean_stdout, clean_stderr, clean_returncode = _run_compiler(clean_command)
    compilation = _Compilation(
        clean_command=clean_command,
        annotated_command=None,
        clean_stdout=clean_stdout,
        clean_stderr=clean_stderr,
        clean_returncode=clean_returncode,
    )
    if clean_returncode:
        return compilation

    annotated_command = [
        *clang_cmd,
        "-S",
        str(source),
        "-Wpass-failed",
        "-Rpass=pipeliner|postpipeliner",
        "-Rpass-missed=pipeliner|postpipeliner|aie-multi-slot-pseudo",
        "-Rpass-analysis=pipeliner|aie-hardware-loops|aie-asm-printer",
        "-fsave-optimization-record=yaml",
        f"-foptimization-record-file={yaml_path}",
        f"-foptimization-record-passes={_REMARK_PASSES}",
        "-o",
        str(annotated_path),
    ]
    annotated_stdout, annotated_stderr, annotated_returncode = _run_compiler(
        annotated_command
    )
    compilation.annotated_command = annotated_command
    compilation.annotated_stdout = annotated_stdout
    compilation.annotated_stderr = annotated_stderr
    compilation.annotated_returncode = annotated_returncode
    return compilation


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_stderr(stderr: str) -> dict[str, Any]:
    """Parse compiler warnings and remark diagnostics not represented in YAML."""
    clean = _ANSI_ESCAPE.sub("", stderr)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    warnings = [line for line in lines if re.search(r"\bwarning:", line)]
    diagnostics = [
        line for line in lines if re.search(r"\b(?:remark|warning|error):", line)
    ]
    pass_failed_warnings = [
        line
        for line in warnings
        if "pass-failed" in line or "transformation might be disabled" in line
    ]
    return {
        "missing_bank_annotations": len(re.findall(r"No memory bank assigned", clean)),
        "warnings": warnings,
        "pass_failed_warnings": pass_failed_warnings,
        "diagnostics": diagnostics,
    }


def resolve_clang_cmd(clang_cmd: list[str] | None) -> list[str]:
    if clang_cmd is not None:
        return list(clang_cmd)
    from artlite.operators.base import AIE_TARGET, CLANG_CMD_COMMON

    return [*CLANG_CMD_COMMON, f"--target={AIE_TARGET}"]
