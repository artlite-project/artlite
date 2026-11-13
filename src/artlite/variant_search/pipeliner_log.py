"""The post-pipeliner's trace: llvm-aie's debug log parsed per loop, and the debug build that writes it."""

import os
import re
import subprocess
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

POSTPIPELINER_MAX_II = 60  # llvm-aie's -aie-postpipeliner-maxii default: no larger II is tried
POSTPIPELINER_MAX_TRY_II = 20  # -aie-postpipeliner-maxtry-ii default: IIs tried from ResMII up


def debug_compiler() -> str | None:
    """An assertions build of llvm-aie's clang++ (-debug-only works): $ARTLITE_DEBUG_CLANG, else the build
    directory of the llvm-aie checkout next to Artlite's."""
    path = os.environ.get("ARTLITE_DEBUG_CLANG") or str(
        Path(__file__).resolve().parents[3] / "llvm-aie" / "build" / "bin" / "clang++"
    )
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


_AIE_PATHS = (":(top)llvm/lib/Target/AIE", ":(top)llvm/lib/CodeGen", ":(top)llvm/include/llvm/CodeGen")


def compiler_drift(compiler: str) -> str | None:
    """How the debug build's llvm-aie differs from the wheel's (traces of other code than the shipped schedule's):
    None when both are one commit; else the commits and, when the build's checkout has both, the commits between
    them that touch the AIE backend or CodeGen."""
    from ..gen_isa_tables import clang_commit
    from ..operators.base import CLANGPP

    debug, wheel = clang_commit(compiler), clang_commit(CLANGPP)
    if debug is None or wheel is None:
        return f"llvm-aie commits unknown (debug {debug or '?'}, wheel {wheel or '?'})"
    if debug.startswith(wheel) or wheel.startswith(debug):
        return None
    text = f"debug llvm-aie {debug[:12]} != the wheel's {wheel[:12]}"

    def git(*args: str) -> str | None:
        try:
            process = subprocess.run(["git", "-C", str(Path(compiler).resolve().parent), *args], capture_output=True,
                                     text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return process.stdout if process.returncode == 0 else None

    ahead, behind = git("rev-list", "--count", f"{wheel}..{debug}"), git("rev-list", "--count", f"{debug}..{wheel}")
    if ahead is None or behind is None:
        return text + " (the commits are not both in the build's checkout)"
    changes = git("log", "--format=%h %s", f"{wheel}...{debug}", "--", *_AIE_PATHS) or ""
    subjects = changes.splitlines()
    text += f": {int(ahead)} commit(s) ahead, {int(behind)} behind; {len(subjects)} touch the AIE backend / CodeGen"
    return text + "".join(f"\n    {line}" for line in subjects[:5]) + ("\n    ..." if len(subjects) > 5 else "")


@dataclass
class _Attempt:
    ii: int
    mode: str  # Physical / Virtual
    rec_mii: int | None = None
    scarce_mii: int | None = None
    scarce_rejected: bool = False
    ran: bool = False  # the strategies ran (II >= RecMII, ScarceRegMII)
    strategies: int = 0  # failed strategy runs
    latency: Counter[int] = field(default_factory=Counter)  # modulo node (SU of the first iteration) -> events
    resources: int = 0
    regalloc: int = 0
    success: bool = False


@dataclass
class PipelinerTrace:
    """The post-pipeliner's search for one loop block, from llvm-aie's -debug-only output."""

    function: str  # symbol
    block: int  # the scheduler's block number (LoopReport.sched_block)
    res_mii: int | None = None
    target_ii: int | None = None  # AIE_TRY_INITIATION_INTERVAL
    not_candidate: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sus: dict[int, str] = field(default_factory=dict)  # SU number -> opcode
    chains: dict[int, list[int]] = field(default_factory=dict)  # modulo node -> its last LastEarliestPusher chain
    attempts: list[_Attempt] = field(default_factory=list)

    @property
    def final(self) -> _Attempt | None:
        return next((attempt for attempt in self.attempts if attempt.success), None)

    def _su(self, number: int, *, short: bool = False) -> str:
        """An SU with its opcode; a node of the second iteration (number >= the loop's instructions) is SUk'."""
        count = len(self.sus)
        base = number % count if count else number
        opcode = self.sus.get(base, "?")
        mark = "'" if count and number >= count else ""
        return f"SU{base}{mark} {opcode.split('_')[0] if short else opcode}"

    def describe(self, *, max_ii: int = POSTPIPELINER_MAX_II) -> list[str]:
        """What the post-pipeliner tried and why each II failed."""
        if self.not_candidate:
            return [f"not a post-pipeliner candidate: {'; '.join(self.not_candidate)}"]
        if self.res_mii is None:
            return ["no post-pipeliner record"]
        head = f"ResMII {self.res_mii}" + (f", TargetII {self.target_ii}" if self.target_ii else "")
        if not self.attempts:
            if self.res_mii > max_ii:
                return [f"{head} > {max_ii} (-aie-postpipeliner-maxii): no II tried"]
            return [f"{head}: no II tried" + (f" ({'; '.join(self.notes)})" if self.notes else "")]
        tried = sorted({attempt.ii for attempt in self.attempts})
        final = self.final
        head += f"; tried II {tried[0]}..{tried[-1]}"
        if final is not None:
            head += f" -> II {final.ii} ({final.mode})"
        elif tried[-1] >= max_ii:
            head += f": no schedule; stopped at -aie-postpipeliner-maxii={max_ii}"
        elif len(tried) >= POSTPIPELINER_MAX_TRY_II:
            head += f": no schedule; stopped after -aie-postpipeliner-maxtry-ii={POSTPIPELINER_MAX_TRY_II} IIs"
        else:
            head += ": no schedule"
        lines = [head + (f" ({'; '.join(self.notes)})" if self.notes else "")]
        for mode in dict.fromkeys(attempt.mode for attempt in self.attempts):
            attempts = [attempt for attempt in self.attempts if attempt.mode == mode]
            bounds = [f"RecMII {_values(attempt.rec_mii for attempt in attempts)}"]
            if any(attempt.scarce_mii is not None for attempt in attempts):
                bounds.append(f"ScarceRegMII {_values(attempt.scarce_mii for attempt in attempts)}")
            failed = [attempt for attempt in attempts if not attempt.success]
            reasons = []
            below_rec = [attempt.ii for attempt in failed if not attempt.ran and not attempt.scarce_rejected]
            if below_rec:
                reasons.append(f"II {_span(below_rec)} < RecMII")
            below_scarce = [attempt.ii for attempt in failed if attempt.scarce_rejected]
            if below_scarce:
                reasons.append(f"II {_span(below_scarce)} < ScarceRegMII")
            ran = [attempt for attempt in failed if attempt.ran]
            if ran:
                events = [
                    f"{name} {count}"
                    for name, count in (
                        ("latency not met", sum(sum(attempt.latency.values()) for attempt in ran)),
                        ("out of resources", sum(attempt.resources for attempt in ran)),
                        ("register allocation failed", sum(attempt.regalloc for attempt in ran)),
                    )
                    if count
                ]
                reasons.append(
                    f"II {_span([attempt.ii for attempt in ran])}: {sum(attempt.strategies for attempt in ran)} "
                    f"strategy runs failed" + (f" ({', '.join(events)})" if events else "")
                )
            lines.append(f"  {mode}: {', '.join(bounds)}" + (f"; {'; '.join(reasons)}" if reasons else ""))
        latency = sum((attempt.latency for attempt in self.attempts if not attempt.success), Counter())
        for node, count in latency.most_common(2):
            chain = [self._su(su, short=True) for su in self.chains.get(node, [])]
            if len(chain) > 10:
                chain = [*chain[:4], "...", *chain[-5:]]
            lines.append(
                f"  latency not met {count}x: {self._su(node)}'s next-iteration copy (') is pushed past its II slot"
                + (f" by {' -> '.join(chain)}" if chain else "")
            )
        return lines


def _values(values: Iterable[int | None]) -> str:
    found = sorted({value for value in values if value is not None})
    return "-" if not found else str(found[0]) if len(found) == 1 else f"{found[0]}..{found[-1]}"


def _span(iis: Sequence[int]) -> str:
    values = sorted(set(iis))
    if values == list(range(values[0], values[-1] + 1)):
        return str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}"
    return ",".join(map(str, values))


_FREQUENT = ("PostPipelined SU", "  Trying SU", "  Pushed ", "Available:", "SU(")
_NOT_CANDIDATE = re.compile(
    r"^ PostPipeliner: (No ZOL|No fallthrough preheader|Loop contains lock instruction|No tripcount def|"
    r"No min tripcount|min tripcount < 2)"
)
_LATENCY = re.compile(r"^Latency not met for SU(\d+) in cycle -?\d+ \(Earliest=-?\d+ ModuloNode=SU(\d+)\)")


def parse_postpipeliner_log(lines: Iterable[str]) -> dict[tuple[str, int], PipelinerTrace]:
    """Post-pipeliner traces per (function symbol, scheduler block) from clang's stderr with
    -debug-only=postpipeliner,postpipeliner-summary,sched-blocks."""
    traces: dict[tuple[str, int], PipelinerTrace] = {}
    function, block = "", -1
    trace: PipelinerTrace | None = None
    attempt: _Attempt | None = None
    rec_mii: int | None = None
    listing = False
    chain: list[int] | None = None
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(_FREQUENT):
            continue
        if chain is not None:
            match = re.match(r"^  --> SU(\d+) @", line)
            if match:
                chain.append(int(match.group(1)))
                continue
            chain = None
        if listing:  # the loop's instructions by SU number, printed before the first attempt's RecMII
            match = re.match(r"^(\d+) (.*)$", line)
            if match:
                if trace is not None:
                    opcode = next((word for word in match.group(2).split() if word[:1].isupper()), "?")
                    trace.sus.setdefault(int(match.group(1)), opcode)
                continue
        if ">> enterFunction " in line:
            function = line.split(">> enterFunction ", 1)[1].strip()
            continue
        match = re.search(r"(?:>> enterBlock|<< leaveBlock) (\d+)", line)
        if match:
            block = int(match.group(1))
            continue
        match = _NOT_CANDIDATE.match(line)
        if match:
            candidate = traces.setdefault((function, block), PipelinerTrace(function, block))
            if match.group(1) not in candidate.not_candidate:
                candidate.not_candidate.append(match.group(1))
            continue
        if "PostPipeliner: TargetII=" in line:  # (printed before the loop's ResMII)
            traces.setdefault((function, block), PipelinerTrace(function, block)).target_ii = int(
                line.rsplit("=", 1)[1]
            )
            continue
        if line.startswith("PostPipeliner: ResMII="):
            trace = PipelinerTrace(function, block, res_mii=int(line.split("=", 1)[1]))
            previous = traces.get((function, block))
            if previous is not None:
                trace.target_ii = previous.target_ii
            traces[(function, block)] = trace
            attempt, listing = None, True
            continue
        if trace is None:
            continue
        if line.startswith("RecMII="):
            rec_mii, listing = int(line.split("=", 1)[1]), False
            continue
        match = re.match(r"^digraph PostPipeliner_II(\d+)_(\w+) \{", line)
        if match:
            attempt = _Attempt(int(match.group(1)), match.group(2), rec_mii=rec_mii)
            trace.attempts.append(attempt)
            continue
        if line.startswith("PostPipeliner: MinTripCount insufficient"):
            if "trip count too small for the stages" not in trace.notes:
                trace.notes.append("trip count too small for the stages")
            continue
        if attempt is None:
            continue
        if line.startswith("ScarceRegMII="):
            attempt.scarce_mii = int(line.split("=", 1)[1])
        elif "Scarce register MII=" in line and "does not fit" in line:
            attempt.scarce_rejected = True
        elif line.startswith("-- MinLength="):
            attempt.ran = True
        elif line.startswith("Latency not met for SU"):
            match = _LATENCY.match(line)
            if match:
                node = int(match.group(2))
                attempt.latency[node] += 1
                chain = trace.chains[node] = []
        elif line == "Out of resources":
            attempt.resources += 1
        elif line == "PostPipeliner: Register allocation failed":
            attempt.regalloc += 1
        elif re.match(r"^ +Strategy \S+ failed$", line):
            attempt.strategies += 1
        elif line == "PostPipeliner: Success":
            attempt.success = attempt.ran = True
    return traces


def trace_postpipeliner(
    source: Path, clang_cmd: Sequence[str], *, compiler: str, timeout: float = 1800
) -> tuple[dict[tuple[str, int], PipelinerTrace], str | None]:
    """Compile with the debug build and parse its post-pipeliner trace (streamed: the log has millions of
    lines). Returns the traces and an error (None when the compile worked)."""
    command = [
        compiler,
        *clang_cmd[1:],
        "-S",
        str(source),
        "-o",
        str(source.with_suffix(".dbg.s")),
        "-mllvm",
        "-debug-only=postpipeliner,postpipeliner-summary,sched-blocks",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    timer = threading.Timer(timeout, process.kill)  # (compile only: no NPU context to protect)
    timer.start()
    tail: list[str] = []

    def lines() -> Iterable[str]:
        assert process.stderr is not None
        for line in process.stderr:
            if "error:" in line or "Unknown command line argument" in line:
                tail.append(line.strip())
            yield line

    try:
        traces = parse_postpipeliner_log(lines())
        returncode = process.wait()
    finally:
        timer.cancel()
    if returncode:
        return traces, f"debug compile failed ({returncode}): {'; '.join(tail[:2])}"
    if not traces:
        return traces, "no post-pipeliner output (not an assertions build of llvm-aie?)"
    return traces, None
