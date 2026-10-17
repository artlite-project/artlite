"""Kernel variant search: compile named variants of an operator's kernel and rank each loop of the target
functions.

A variant is the operator's rendered kernel with any of: a loop pragma on the target functions' loops and those of
the always_inline helpers they call (``post`` = AIE_PREPARE_FOR_POSTPIPELINING, ``none``, ``iiN`` =
AIE_TRY_INITIATION_INTERVAL(N), any text),
source substitutions (each must match exactly once), replaced function definitions, other operator (Jinja)
parameters. All variants compile in parallel with measure_ii; per loop of each target function one table ranks
them by cycles per entry: II, stages, iterations, cycles (prologue and epilogue included), body bundles, spill
code, stack, the ops per issue slot of the loop body in the assembly and the binding slot. For an unpipelined
loop (or every loop with ``why``) a debug build of llvm-aie gives the post-pipeliner's ResMII / RecMII and why
each II failed (the report says when that build's commit is not the wheel's). With ``npu``, each variant also
runs on the NPU, one process per build, one at a time: the time of one dispatch and the event-window counters of
named regions next to their static cycles.

    python -m artlite.variant_search 'tests.rtc.mmv.MmvOperator(fmt="Q5_K", n_tiles=4)' \\
        --fn _q5k --pragmas post,none,ii44 [--sub NAME OLD NEW] [--define NAME FN @def.cc] \\
        [--param NAME key=value] [--variants FILE.py] [--why] [--npu --region codes=_q5k]

OLD / NEW / definitions starting with ``@`` are read from that file. Python: ``search(...)``, ``print_report``,
``run_npu``.
"""

import argparse
import ast
import functools
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import runpy
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import parallel_compile
from .measure_ii import (
    SLOTS,
    KernelReport,
    LoopReport,
    SlotCounts,
    _closing,
    _demangle_short,
    _loop_where,
    _statement_end,
    _Token,
    _tokenize,
    _tree_order,
    call_cycles,
    count_slots,
    loop_cycles,
    measure,
)

# -- Source patching -------------------------------------------------------------------------------------------

POST = "AIE_PREPARE_FOR_POSTPIPELINING"
_PIPELINING = {"AIE_PREPARE_FOR_POSTPIPELINING", "AIE_TRY_INITIATION_INTERVAL"}
_QUALIFIERS = {"const", "noexcept", "override", "final", "volatile", "&", "&&"}


def ii(n: int) -> str:
    return f"AIE_TRY_INITIATION_INTERVAL({n})"


def pragma_text(spec: str) -> str | None:
    """A pragma shorthand: base / keep = as rendered (None), none = no pipelining pragma, post = POST, iiN =
    AIE_TRY_INITIATION_INTERVAL(N), joined by + for several (post+ii44); anything else is the pragma text."""
    if "+" in spec and "(" not in spec:
        return " ".join(filter(None, (pragma_text(part) for part in spec.split("+"))))
    if spec in ("base", "keep"):
        return None
    if spec == "none":
        return ""
    if spec == "post":
        return POST
    match = re.fullmatch(r"ii(\d+)", spec)
    return ii(int(match.group(1))) if match else spec


def _is_loop_macro(text: str) -> bool:
    return text == "_Pragma" or re.fullmatch(r"AIE_[A-Z0-9_]+", text) is not None


def _opening(tokens: list[_Token], index: int) -> int:
    """Index of the bracket opening the one at tokens[index]."""
    depth = 0
    for position in range(index, -1, -1):
        text = tokens[position].text
        if text in (")", "]", "}"):
            depth += 1
        elif text in ("(", "[", "{"):
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("unbalanced brackets")


@dataclass(frozen=True)
class _Definition:
    name: str
    start: int  # token index of its first token (template header, attributes, return type)
    body: int  # token index of the body's opening brace
    end: int  # token index of the body's closing brace


_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "sizeof", "decltype", "alignof", "static_assert"}


def _definitions(tokens: list[_Token], name: str) -> list[_Definition]:
    """The function definitions whose name contains name ("": all)."""
    found = []
    for index, token in enumerate(tokens[:-1]):
        if (
            name not in token.text
            or token.text in _KEYWORDS
            or not re.fullmatch(r"[A-Za-z_]\w*", token.text)
            or tokens[index + 1].text != "("
        ):
            continue
        after = _closing(tokens, index + 1) + 1
        while after < len(tokens) and tokens[after].text in _QUALIFIERS:
            after += 1
        if after >= len(tokens) or tokens[after].text != "{":
            continue  # a call or a declaration
        start = index
        while start > 0 and tokens[start - 1].text not in (";", "}", "{"):
            start -= 1
        found.append(_Definition(token.text, start, after, _closing(tokens, after)))
    return found


@dataclass(frozen=True)
class _Loop:
    keyword: int  # token index of for / while / do
    end: int  # token index of its last token
    prefix: tuple[tuple[int, int], ...]  # token spans of the AIE_* macros in front of it

    def macros(self, tokens: list[_Token]) -> list[str]:
        return [tokens[first].text for first, _ in self.prefix]

    def rolled(self, tokens: list[_Token]) -> bool:
        return "AIE_LOOP_UNROLL_FULL" not in self.macros(tokens)


def _loops(tokens: list[_Token], definition: _Definition) -> list[_Loop]:
    """The loop statements of a function body, in source order."""
    loops = []
    do_whiles: set[int] = set()
    for index in range(definition.body + 1, definition.end):
        if tokens[index].text not in ("for", "while", "do") or index in do_whiles:
            continue
        end = _statement_end(tokens, index)
        if tokens[index].text == "do":
            do_whiles.add(_statement_end(tokens, index + 1) + 1)
        prefix: list[tuple[int, int]] = []
        position = index
        while position > definition.body + 1:
            last = position - 1
            if tokens[last].text == ")":
                first = _opening(tokens, last)
                if not _is_loop_macro(tokens[first - 1].text):
                    break
                prefix.insert(0, (first - 1, last))
                position = first - 1
            elif _is_loop_macro(tokens[last].text):
                prefix.insert(0, (last, last))
                position = last
            else:
                break
        loops.append(_Loop(index, end, tuple(prefix)))
    return loops


def _offsets(text: str) -> Callable[[_Token], int]:
    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    return lambda token: line_starts[token.line - 1] + token.column - 1


def _calls(tokens: list[_Token], start: int, end: int, names: Iterable[str]) -> set[str]:
    """The names called between token indices start and end (name( or name<...>()."""
    names = set(names)
    return {
        tokens[index].text
        for index in range(start, end)
        if tokens[index].text in names and tokens[index + 1].text in ("(", "<") and tokens[index - 1].text != "."
    }


def _helpers(tokens: list[_Token], definitions: Sequence[_Definition]) -> list[_Definition]:
    """The always_inline functions the definitions call, also through each other, in source order: their loops are
    compiled into the targets."""
    inline: dict[str, list[_Definition]] = {}
    for definition in _definitions(tokens, ""):
        if any(token.text == "always_inline" for token in tokens[definition.start : definition.body]):
            inline.setdefault(definition.name, []).append(definition)
    found: dict[_Definition, None] = {}
    pending = list(definitions)
    while pending:
        definition = pending.pop()
        for name in _calls(tokens, definition.body + 1, definition.end, inline):
            for helper in inline[name]:
                if helper not in found and helper not in definitions:
                    found[helper] = None
                    pending.append(helper)
    return sorted(found, key=lambda definition: definition.start)


def _target_loops(
    source: str, functions: Sequence[str], loops: Sequence[int] | None
) -> tuple[list[_Token], list[_Loop]]:
    """The loops that take a pragma: the rolled loops with the given indices (in source order over the target
    functions, then over the always_inline helpers they call), by default the innermost rolled loops (a loop that
    calls a helper with rolled loops is not). A helper's pragma applies to every caller of the helper."""
    tokens = _tokenize(source)
    definitions = [definition for name in functions for definition in _definitions(tokens, name)]
    if not definitions:
        raise ValueError(f"no definition of {', '.join(functions)} in the kernel source")
    definitions = sorted(set(definitions), key=lambda definition: definition.start)
    helpers = _helpers(tokens, definitions)
    rolled = [
        loop for definition in [*definitions, *helpers] for loop in _loops(tokens, definition) if loop.rolled(tokens)
    ]
    if loops is not None:
        if any(not 0 <= k < len(rolled) for k in loops):
            raise ValueError(f"loop index out of range: {len(rolled)} rolled loop(s) in {', '.join(functions)} and "
                             "the always_inline helpers they call")
        return tokens, [rolled[k] for k in loops]
    looped = {  # the helpers with rolled loops, also in the helpers they call
        helper.name
        for helper in helpers
        if any(loop.rolled(tokens) for other in [helper, *_helpers(tokens, [helper])] for loop in _loops(tokens, other))
    }
    innermost = [
        loop
        for loop in rolled
        if not any(loop.keyword < other.keyword <= loop.end for other in rolled)
        and not _calls(tokens, loop.keyword, loop.end, looped)
    ]
    if not innermost:
        raise ValueError(f"no rolled loop in {', '.join(functions)}")
    return tokens, innermost


def apply_pragma(source: str, functions: Sequence[str], pragma: str, *, loops: Sequence[int] | None = None) -> str:
    """The source with the pipelining pragmas (AIE_PREPARE_FOR_POSTPIPELINING, AIE_TRY_INITIATION_INTERVAL) of
    the target loops (see _target_loops) replaced by pragma ("" removes them); other pragmas stay."""
    tokens, targets = _target_loops(source, functions, loops)
    offset = _offsets(source)
    edits: list[tuple[int, int, str]] = []
    for loop in targets:
        for first, last in loop.prefix:
            if tokens[first].text not in _PIPELINING:
                continue
            start, end = offset(tokens[first]), offset(tokens[last]) + len(tokens[last].text)
            line_start = source.rfind("\n", 0, start) + 1
            line_end = source.find("\n", end)
            line_end = len(source) if line_end < 0 else line_end
            if not source[line_start:start].strip() and not source[end:line_end].strip():
                start, end = line_start, min(line_end + 1, len(source))  # the macro's own line
            else:
                while end < len(source) and source[end] in " \t":
                    end += 1
            edits.append((start, end, ""))
        if pragma:
            at = offset(tokens[loop.keyword])
            indent = source[source.rfind("\n", 0, at) + 1 : at]
            edits.append((at, at, f"{pragma}\n{indent}" if not indent.strip() else f"{pragma} "))
    for start, end, text in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        source = source[:start] + text + source[end:]
    return source


def pragma_lines(source: str, functions: Sequence[str], *, loops: Sequence[int] | None = None) -> list[int]:
    """The lines of the loop statements a pragma goes on (see apply_pragma)."""
    tokens, targets = _target_loops(source, functions, loops)
    return [tokens[loop.keyword].line for loop in targets]


def substitute(source: str, old: str, new: str) -> str:
    """The source with old replaced by new; old must occur exactly once."""
    count = source.count(old)
    if count != 1:
        raise ValueError(f"substitution text matches {count} times (must match once): {old[:80]!r}")
    return source.replace(old, new)


def replace_function(source: str, name: str, definition: str) -> str:
    """The source with the definition of the function whose name contains name (exactly one) replaced."""
    tokens = _tokenize(source)
    found = _definitions(tokens, name)
    if len(found) != 1:
        raise ValueError(f"{len(found)} definitions of a function named *{name}* (must be one)")
    offset = _offsets(source)
    start, end = offset(tokens[found[0].start]), offset(tokens[found[0].end]) + 1
    return source[:start] + definition.strip() + source[end:]


# -- Variants --------------------------------------------------------------------------------------------------


@dataclass
class Variant:
    """A named kernel variant: patches applied to the rendered source in this order: parameters (the operator's
    or the render function's keyword arguments), function definitions (name substring -> new definition),
    substitutions (old, new), the pragma on the target functions' loops (None: as rendered)."""

    name: str
    pragma: str | None = None
    subs: list[tuple[str, str]] = field(default_factory=list)
    defines: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    # the rolled loops (indices: the targets' in source order, then their always_inline helpers') that take the
    # pragma; None: the innermost ones
    loops: list[int] | None = None


@dataclass
class _Rendered:
    source: str  # the compiled file: '#include "operators.h"' + the patched kernel source
    kernel_source: str  # the patched kernel source (what get_kernel_src returns)
    clang_cmd: list[str]
    pragma_lines: list[int]  # in the compiled file


_INCLUDE = '#include "operators.h"\n\n'  # (as ArtcOperator.compile prepends it)


def render_variant(
    factory: Callable[..., Any], kwargs: Mapping[str, Any], variant: Variant, *, functions: Sequence[str]
) -> _Rendered:
    """Render the operator (or the render function) with the variant's parameters and apply its patches."""
    from .operators.base import AIE_TARGET, CLANG_CMD_COMMON

    made = factory(**{**kwargs, **variant.params})
    if isinstance(made, str):
        source, flags = made, []
    else:
        source, flags = made.get_kernel_src(), list(made.clang_flags)
    try:
        for name, definition in variant.defines.items():
            source = replace_function(source, name, definition)
        for old, new in variant.subs:
            source = substitute(source, old, new)
        lines: list[int] = []
        if variant.pragma is not None:
            source = apply_pragma(source, functions, variant.pragma, loops=variant.loops)
            lines = [line + _INCLUDE.count("\n") for line in pragma_lines(source, functions, loops=variant.loops)]
    except ValueError as exc:
        raise ValueError(f"variant {variant.name}: {exc}") from None
    return _Rendered(_INCLUDE + source, source, [*CLANG_CMD_COMMON, f"--target={AIE_TARGET}", *flags], lines)


# -- Post-pipeliner trace (debug build of llvm-aie) ------------------------------------------------------------

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
    from .gen_isa_tables import clang_commit
    from .operators.base import CLANGPP

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


# -- Search ----------------------------------------------------------------------------------------------------


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
    same_as: str | None = None  # an earlier variant with the identical source (its report is shared)
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


def _label(factory: Callable[..., Any], kwargs: Mapping[str, Any]) -> str:
    return f"{getattr(factory, '__qualname__', repr(factory))}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"


def _symbols(report: KernelReport, name: str) -> list[str]:
    """The function symbols of a report whose demangled name contains name, in assembly order."""
    names = [*report.functions, *(loop.function for loop in report.loops)]
    return [symbol for symbol in dict.fromkeys(names) if name in _demangle_short(symbol)]


def target_loops(report: KernelReport, symbol: str) -> list[LoopReport]:
    """The loops of a function in nesting order (depth first, siblings by source line)."""
    return _tree_order([loop for loop in report.loops if loop.function == symbol])


def _wants_trace(result: VariantResult, functions: Sequence[str]) -> bool:
    """A target loop of one block stayed unpipelined (the post-pipeliner tried and failed, or did not try)."""
    report = result.report
    if report is None:
        return False
    return any(
        loop.sched == "none" and loop.num_blocks == 1
        for name in functions
        for symbol in _symbols(report, name)
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
    table per loop; totals: other functions reported by their cycles per call (e.g. the kernel entry point).
    The post-pipeliner trace (debug llvm-aie) is taken for variants with an unpipelined one-block target loop, or
    for all with why. A patch that does not apply raises ValueError before anything compiles."""
    from .operators.base import CACHE_DIR

    names = [variant.name for variant in variants]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"variant names must be unique and not empty: {names}")
    rendered = [render_variant(factory, kwargs, variant, functions=functions) for variant in variants]
    key = hashlib.sha256(repr((_label(factory, kwargs), [r.source for r in rendered])).encode()).hexdigest()[:12]
    out = Path(out_dir) if out_dir is not None else Path(CACHE_DIR) / "variant_search" / key
    results = []
    by_source: dict[str, VariantResult] = {}
    for variant, item in zip(variants, rendered):
        directory = out / re.sub(r"[^\w.+-]", "_", variant.name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "kernel.cc"
        path.write_text(item.source)
        result = VariantResult(variant, path, item.kernel_source, item.clang_cmd, item.pragma_lines)
        first = by_source.setdefault(item.source, result)
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


# -- Report ----------------------------------------------------------------------------------------------------


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
    print(f"\nvariant search: {_label(result.factory, result.kwargs)}")
    print(f"  targets: {', '.join(result.functions)}; sources and reports in {result.out_dir}")
    for item in result.variants:
        if item.error:
            print(f"  ✗ {item.variant.name}: {item.error}")
    ok = [(item, item.report) for item in result.variants if item.report is not None and item.error is None]
    if not ok:
        return
    names = list(dict.fromkeys(
        _demangle_short(symbol)
        for name in [*result.functions, *result.totals]
        for _, report in ok
        for symbol in _symbols(report, name)
    ))
    rows = []
    for item, report in ok:
        symbols = {_demangle_short(symbol): symbol for symbol in report.functions}
        cells = [_cycles_text(call_cycles(report, symbols[name])) if name in symbols else "-" for name in names]
        stack = max((report.stack_bytes.get(symbol, 0) for symbol in report.functions), default=0)
        code = sum(block.bytes for blocks in report.layout.values() for block in blocks)
        rows.append([_row_name(item), *cells, str(stack), str(code)])
    print("\ncycles per call (calls included; <=: every branch counted)")
    for line in _table(["variant", *names, "max stack", "code B"], rows):
        print(line)
    for name in result.functions:
        symbols = list(dict.fromkeys(symbol for _, report in ok for symbol in _symbols(report, name)))
        if not symbols:
            print(f"\n{name}: no such function in the compiled code (inlined?)")
        for symbol in symbols:
            loops_of = [(item, report, target_loops(report, symbol)) for item, report in ok]
            count = max(len(loops) for _, _, loops in loops_of)
            if not count:
                print(f"\n{_demangle_short(symbol)}: no loops")
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
    where = _loop_where(entries[0][1], first) if first.kernel_line or first.source_ref else ""
    print(
        f"\n{_demangle_short(symbol)} loop {index + 1}/{count}: {first.block}{where}, depth {first.depth} "
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


# -- NPU -------------------------------------------------------------------------------------------------------

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
        symbols = _symbols(report, name)
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


def _region_result(directory: Path) -> RegionResult:
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
           if not key.startswith("ARTLITE_PROFILE") and key != "ARTLITE_MEASURE_II"}
    if build.region is not None:
        env.update(ARTLITE_PROFILE="counters", ARTLITE_PROFILE_DIR=str(build.directory / "profile"),
                   ARTLITE_PROFILE_COUNTERS=COUNTERS, ARTLITE_PROFILE_SAMPLE_EVERY="1")
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
                npu.regions[build.region] = _region_result(build.directory / "profile")
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


def _worker(spec_path: str, *, compile_only: bool) -> None:
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
    import torch

    import artlite

    device = operator.device  # opens the HW context
    generator = torch.Generator().manual_seed(0)
    buffers = []
    for arg in operator.get_arg_spec():
        if arg.direction == "in":
            n = math.prod(arg.shape)
            if arg.dtype.is_floating_point:
                data = (torch.randn(n, generator=generator) * 0.5).to(arg.dtype)
            else:
                data = torch.randint(0, 100, (n,), generator=generator).to(arg.dtype)
            buffers.append(artlite.from_torch(data.reshape(arg.shape), device))
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


# -- CLI -------------------------------------------------------------------------------------------------------


def parse_operator(expression: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    """'Name(key=literal, ...)': Name from artlite.operators (also a submodule's attribute: overlay.gemv.render) or a
    dotted import path."""
    from . import operators

    call = ast.parse(expression, mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise ValueError(f"expected Name(key=value, ...), got {expression!r}")
    parts = ast.unparse(call.func).split(".")

    def module(name: str) -> Any:
        try:
            return importlib.import_module(name) if importlib.util.find_spec(name) else None
        except ImportError:
            return None

    # (the longest prefix that is a submodule of artlite.operators: overlay.gemv.render)
    sub = next((i for i in range(len(parts), 0, -1) if module(".".join([operators.__name__, *parts[:i]])) is not None), 0)
    if hasattr(operators, parts[0]):
        base: Any = getattr(operators, parts[0])
        rest = parts[1:]
    elif sub:
        base, rest = module(".".join([operators.__name__, *parts[:sub]])), parts[sub:]
    else:
        index = next((i for i in range(len(parts), 0, -1) if module(".".join(parts[:i])) is not None), 0)
        if not index:
            raise ValueError(f"cannot resolve {ast.unparse(call.func)}")
        base, rest = module(".".join(parts[:index])), parts[index:]
    factory = functools.reduce(getattr, rest, base)
    return factory, {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg}


def _text(argument: str) -> str:
    return Path(argument[1:]).read_text() if argument.startswith("@") else argument


def _cli_variants(args: argparse.Namespace) -> list[Variant]:
    variants: dict[str, Variant] = {}

    def get(name: str) -> Variant:
        return variants.setdefault(name, Variant(name))

    if args.variants:
        loaded = runpy.run_path(args.variants)["VARIANTS"]
        for variant in loaded.values() if isinstance(loaded, Mapping) else loaded:
            variants[variant.name] = variant
    for token in filter(None, args.pragmas.split(",")):
        get(token).pragma = pragma_text(token)
    for name, text in args.pragma:
        get(name).pragma = pragma_text(text)
    for name, old, new in args.sub:
        get(name).subs.append((_text(old), _text(new)))
    for name, function, definition in args.define:
        get(name).defines[function] = _text(definition)
    for name, assignment in args.param:
        key, _, value = assignment.partition("=")
        try:
            get(name).params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            get(name).params[key] = value
    if args.loop:
        for variant in variants.values():
            variant.loops = args.loop
    if "base" not in variants:
        variants = {"base": Variant("base", loops=args.loop or None), **variants}
    return list(variants.values())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m artlite.variant_search",
        description="Compile kernel variants of an operator and rank the loops of target functions.",
    )
    parser.add_argument("operator", nargs="?", help="Name(key=value, ...): an operator class of artlite.operators "
                        "or a function returning kernel source (overlay.gemv.render)")
    parser.add_argument("--fn", action="append", default=[], help="target function (name substring); repeatable")
    parser.add_argument("--total", action="append", default=[], help="also report this function's cycles per call")
    parser.add_argument("--pragmas", default="", help="variants named by pragma shorthands: base,post,none,ii44")
    parser.add_argument("--pragma", nargs=2, action="append", default=[], metavar=("VARIANT", "PRAGMA"))
    parser.add_argument("--sub", nargs=3, action="append", default=[], metavar=("VARIANT", "OLD", "NEW"),
                        help="replace OLD (must match once) by NEW; @file reads the text")
    parser.add_argument("--define", nargs=3, action="append", default=[], metavar=("VARIANT", "FN", "DEFINITION"),
                        help="replace the definition of function FN; @file reads it")
    parser.add_argument("--param", nargs=2, action="append", default=[], metavar=("VARIANT", "KEY=VALUE"),
                        help="an operator keyword argument of the variant (a Python literal)")
    parser.add_argument("--variants", metavar="FILE.py", help="a file defining VARIANTS (a list of Variant)")
    parser.add_argument("--loop", type=int, action="append", help="pragmas go on these rolled loops (index in "
                        "source order over the target functions, then the always_inline helpers they call) instead "
                        "of the innermost ones")
    parser.add_argument("--why", action="store_true", help="post-pipeliner traces for every variant (debug llvm-aie)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="concurrent compiles (default: ARTLITE_COMPILE_JOBS, else the CPU count)")
    parser.add_argument("--out", help="directory for the sources and reports")
    parser.add_argument("--npu", action="store_true", help="also time each variant on the NPU")
    parser.add_argument("--reps", type=int, default=30, help="timed dispatches per variant")
    parser.add_argument("--region", action="append", default=[], metavar="NAME[=FN[*N]+...]",
                        help="counter windows of the operator's win=NAME build, static cycles from N calls of FN")
    parser.add_argument("--win-param", default="win", help="the operator parameter that selects an event window")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--compile-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker, compile_only=args.compile_only)
        return
    if not args.operator or not args.fn:
        parser.error("an operator and at least one --fn are required")
    factory, kwargs = parse_operator(args.operator)
    result = search(factory, kwargs, functions=args.fn, variants=_cli_variants(args), totals=args.total,
                    jobs=args.jobs, out_dir=args.out, why=args.why)
    if args.npu:
        run_npu(result, reps=args.reps, regions=parse_regions(args.region), win_param=args.win_param,
                jobs=args.jobs)
    print_report(result)


if __name__ == "__main__":
    main()
