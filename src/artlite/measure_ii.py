"""Measure AIE loop schedules from LLVM optimization records and assembly.

Loop model (verified against llvm-aie sources and the emitted assembly):

- The loops are the final code's loops. The asm printer's loop comments
  (``=>This Loop Header: Depth=N``, ``Parent Loop``, ``in Loop: Header=``)
  give every loop header, its nesting and its blocks, including multi-block
  branch loops that no remark names by block. The assembly read is the one of
  the remarks compile: ``-fsave-optimization-record`` implies line tables, so
  its ``.loc`` directives give each bundle's source line (its code is the
  clean compile's).
- The per-loop *final* result is a single ``!Passed`` remark
  ``Pass: pipeliner, Name: schedule`` with a ``Pipeliner`` argument, emitted by
  ``AIEInterBlockScheduling::emitLoopRemarks`` after both pipeliners and
  register allocation have run, for single-block loops only.
  ``Pipeliner: postpipeliner`` = post-RA pipelined (II/NS from the schedule),
  ``Pipeliner: prepipeliner`` = pre-RA pipelined (II = actual loop-region
  length after code generation, NS from metadata), ``Pipeliner: loop-aware``
  = the loop stayed unpipelined (II = body length). Prologue/Epilogue block
  names and bundle counts are included for all three cases. The remark names
  the loop block ``bb.N.<IR name>``; block numbers can change before the asm
  printer runs, and a pre-RA pipelined loop's prologue keeps its IR name, so
  the remark goes to the loop header the asm printer reports with that IR
  name, II and NS, else to the loop of that IR name, in layout order.
- The pre-RA MachinePipeliner remarks (``canPipelineLoop``, ``schedule``)
  know a loop only by its statement's source location
  (``MachineLoop::getStartLoc``); every loop gets at least one.
  ``!Analysis ... "Schedule found with Initiation Interval"`` is the pre-RA
  *search* result: a plan made before code generation and register
  allocation, often shorter than the generated loop; reported as a note,
  never as the schedule. ``!Passed ... "Pipelined succesfully!"`` only says
  the plan was applied. Rejections (``Not a single basic block: N``, ``The
  loop structure is not supported``, ``Unable to find schedule``) matter only
  for loops that stayed unpipelined: the post-RA pipeliner may still succeed.
- These source loops are matched with the final loops: a pre-RA schedule
  names its loop block; otherwise the source lines of a loop's own bundles
  (the loop statement, its extent parsed from the source, that contains most
  of them, consistently with the nesting), then the lines of its preheader
  (the guard of the loop statement), then elimination. A source loop that
  matches no final loop is reported as unattributed.
- A zero-overhead loop (ZOL) starts at the label written to ``ls``
  (``aie-hardware-loops !Analysis`` reports the same for loops with a single
  loop-control block). ``aie-asm-printer !Analysis`` reports each block's
  BundleCount/ByteCount per region, in layout order, and on a scheduled loop
  block's first region the II and NS of its final remark.
- Hardware-loop trip counts are recovered from the assembly: a ZOL is set up
  by ``add.nc lc, rX, #imm`` and executes exactly ``lc`` times. The register
  value is read with start-of-bundle semantics (a ``mov`` to rX in the same
  bundle is not visible to the ``add.nc``). For pipelined loops the pipeliner
  peels ``NS-1`` iterations into prologue+epilogue, so the source trip count
  is ``lc + (NS-1)``.
- ``regalloc`` ``(Loop)SpillReloadCopies`` remarks count spilled and reloaded
  registers per loop statement (inner loops included) and per function; each
  run of the allocator emits them again, the last run counts all register
  classes. The assembly's ``N-byte [Folded] Spill/Reload`` comments count
  instructions: a register wider than 64 bytes moves in 64-byte parts (a
  2048-bit accumulator spill is 4 stores).
- Costs count a bundle as a cycle. A loop's iteration adds its inner loops'
  iterations (their prologue and epilogue blocks are the enclosing loop's own
  blocks) and its calls, a call costing the callee's code outside loops, its
  loops and its calls. Branches count every path (an upper bound). The
  remark's prologue and epilogue are whole blocks (the preheader, the exit):
  between two consecutive loops one block is the first one's epilogue and the
  second one's prologue; it counts in the first loop's cost only.
- The asm printer's comment on a ``.loc`` lists the inlining chain of the
  bundle (``vector.hpp:1098:29 @[ kernel_lib.h:41:27 @[ kernel.cc:315:9 ] ]``):
  the source matching uses the innermost frame in a loop statement, so a
  helper's loop inlined from a header (whose own bundles come from deeper
  headers) still finds its statement, and the frame in the measured file
  after it is the call site.
"""

from __future__ import annotations

import bisect
import hashlib
import re
import shlex
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from . import aie2p_isa


class _LLVMLoader(yaml.SafeLoader):
    pass


def _construct_tagged(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, MappingNode):
        value = loader.construct_mapping(node, deep=True)
    elif isinstance(node, SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, ScalarNode):
        value = loader.construct_scalar(node)
    else:
        value = loader.construct_object(node, deep=True)
    if isinstance(value, dict):
        value["_tag"] = f"!{tag_suffix}"
    return value


# YAML 1.1 reads plain scalars with digit separators as integers: the
# hardware-loop ID 0_1 would load as 1 (and 1_0 as 10). LLVM writes its
# integers in plain decimal, so only those resolve to int.
_LLVMLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:int"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_LLVMLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^[-+]?(?:0|[1-9][0-9]*)$"), list("-+0123456789")
)
_LLVMLoader.add_multi_constructor("!", _construct_tagged)


def _sanitise_yaml(text: str) -> str:
    """Normalize tabs used as YAML indentation without changing quoted values."""
    return re.sub(
        r"^[ \t]+",
        lambda match: match.group(0).replace("\t", "    "),
        text,
        flags=re.MULTILINE,
    )


def _normalise_block(block: str) -> str:
    return re.sub(r"^bb\.\d+\.", "", block.strip())


# -- Data classes -------------------------------------------------------------


@dataclass
class LoopReport:
    """One loop, described by its final (post-everything) compiler state."""

    function: str = ""
    block: str = ""  # IR name of the loop header (normalized, no bb.N. prefix)
    mbb: int | None = None  # machine basic block number of the header
    # the header's number in the final remark (bb.N): the scheduler's, as llvm-aie's debug output names it
    sched_block: int | None = None
    sched: str = "none"  # "pre-RA" | "post-RA" | "none"
    ii: int | None = None  # final II (post-RA) or final loop-body length (pre-RA)
    ns: int = 1
    prologue_bundles: int | None = None
    epilogue_bundles: int | None = None
    prologue_block: str = ""  # the remark's label of the prologue block (bb.N.name)
    epilogue_block: str = ""
    # the prologue block is another loop's epilogue (the block between consecutive loops): its bundles count in
    # that loop's cost, not here; the loop as printed ("for.body @kernel.cc:40")
    prologue_of: str = ""
    epilogue_of: str = ""  # the loop whose prologue is this loop's epilogue
    body_bundles: int | None = None  # the loop's own blocks (not its inner loops'), ground truth from aie-asm-printer
    body_bytes: int | None = None
    zol: bool | None = None
    hw_iters: int | None = None  # ZOL executions recovered from assembly (lc)
    kernel_line: int | None = None  # loop-statement line in the compiled source
    source_trip: int | None = None  # trip count parsed from the source for-header
    source_ref: str = ""  # "file:line" of the loop statement (also when in a header)
    call_site: str = ""  # an inlined helper's loop of another file: "kernel.cc:L" of the call in the measured file
    search_ii: int | None = None  # stale pre-RA search plan, for reference
    prereject_reasons: list[str] = field(default_factory=list)
    search_mii_note: str | None = None  # "MII too large: X > limit" if reported
    # spill code in the loop's own blocks, as asm instructions: a spill of a
    # 2048-bit accumulator is four 64-byte stores (regalloc counts it once)
    loop_spills: int = 0
    loop_reloads: int = 0
    spill_bytes: int = 0
    reload_bytes: int = 0
    # the register allocator's count for the loop statement, its inner loops included
    regalloc_spills: int | None = None
    regalloc_reloads: int | None = None
    # callees of `jl` (and tail calls: `j` to a function) inside the loop block (soft-float helpers such as
    # __mulsf3/__gtsf2, division, noinline functions): each costs a call,
    # its body and the delay slots, none of which the II/bundle count shows
    loop_calls: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)  # their symbols
    header: str = ""  # asm label of the header block (".LBBf_n"); "" when not located in the assembly
    depth: int = 1  # nesting depth in the final code
    parent: str | None = None  # header label of the enclosing loop
    inner: list[str] = field(default_factory=list)  # header labels of the directly nested loops
    num_blocks: int = 1  # blocks of the loop, those of its inner loops included
    # conditional jumps besides the back edge (an if, a break): not every bundle runs every iteration
    branches: bool = False
    early_exit: bool = False  # a break, return or goto in the loop statement: source_trip is a maximum
    # the bundles of the loop's own blocks in the assembly, each a list of its instructions
    asm_body: list[list[str]] = field(default_factory=list, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return self.function, self.block

    @property
    def trip_count(self) -> int | None:
        """Best estimate of source-level trip count (peeling undone)."""
        if self.hw_iters is None:
            return self.source_trip
        if self.sched in {"pre-RA", "post-RA"} and self.ns > 1:
            return self.hw_iters + self.ns - 1
        return self.hw_iters

    @property
    def name(self) -> str:
        return f"{_demangle_short(self.function)}::{_block_label(self.block)}"

    def effective_iters(self) -> int | None:
        """Steady-state loop executions feeding the cost model."""
        if self.hw_iters is not None:
            return self.hw_iters
        iters = self.source_trip
        if iters is None:
            return None
        if self.sched in {"pre-RA", "post-RA"} and self.ns > 1:
            # Source-derived trips: the pipeliner peels NS-1 iterations.
            return max(0, iters - (self.ns - 1))
        return iters

    @property
    def prologue_cycles(self) -> int:
        """The prologue's bundles in this loop's cost: 0 when they count as another loop's epilogue."""
        return 0 if self.prologue_of else self.prologue_bundles or 0

    def estimated_cycles(self) -> int | None:
        """P + iterations*II + E if every term is known (P: prologue_cycles)."""
        iters = self.effective_iters()
        if iters is None or self.ii is None:
            return None
        return self.prologue_cycles + iters * self.ii + (self.epilogue_bundles or 0)

    def cost_formula(self) -> str:
        prologue = _prologue_text(self)
        epilogue = "-" if self.epilogue_bundles is None else str(self.epilogue_bundles)
        iters = self.effective_iters()
        return f"{prologue} + {'iters' if iters is None else iters}*{self.ii} + {epilogue}"


@dataclass
class BlockInfo:
    bundles: int
    bytes: int
    regions: int = 1
    region_bundles: list[int] = field(default_factory=list)
    region_bytes: list[int] = field(default_factory=list)
    region_indices: set[int] = field(default_factory=set)
    name: str = ""  # IR name of the block
    schedule: tuple[int, int] | None = None  # a loop block of the layout: the (II, NS) the post-RA scheduler committed


@dataclass
class HardwareLoop:
    function: str
    block: str
    loop_id: str = ""
    zero_overhead: bool | None = None


@dataclass(eq=False)
class SourceLoop:
    """A loop of the pre-RA pipeliner remarks, known by its statement's source location."""

    function: str
    path: str  # DebugLoc file as the remark spells it
    line: int
    column: int = 0
    end_line: int | None = None  # last line of the loop statement, parsed from the source
    scope: tuple[int, int] | None = None  # lines of the function body around the statement
    exits: bool = False  # a break, return or goto in its body: the header's trip count is a maximum
    trip: int | None = None  # constant trip count of its for-header
    num_blocks: int | None = None  # rejected as "Not a single basic block: N"
    reasons: list[str] = field(default_factory=list)  # pre-RA rejection reasons
    search_ii: int | None = None  # "Schedule found with Initiation Interval"
    search_block: str = ""  # IR name of the block that schedule is for
    mii_note: str | None = None
    pipelined: bool = False  # "Pipelined succesfully!"
    regalloc: tuple[int, int] | None = None  # (spills, reloads) of registers, inner loops included

    @property
    def file(self) -> str:
        return Path(self.path).name

    @property
    def ref(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class KernelReport:
    source: Path
    cmd: str | None = None
    commands: list[str] = field(default_factory=list)
    returncode: int = 0
    yaml_path: Path | None = None
    clean_assembly: Path | None = None
    annotated_assembly: Path | None = None
    loops: list[LoopReport] = field(default_factory=list)
    blocks: dict[str, BlockInfo] = field(default_factory=dict)
    layout: dict[str, list[BlockInfo]] = field(default_factory=dict)  # per function: its blocks in layout order
    functions: list[str] = field(default_factory=list)  # in assembly order
    hardware_loops: list[HardwareLoop] = field(default_factory=list)
    source_loops: dict[str, list[SourceLoop]] = field(default_factory=dict)
    unattributed_loops: list[LoopReport] = field(default_factory=list)  # source loops no final loop matched
    # per function, its blocks outside loops: (bundles, callee symbols, has conditional branches)
    outside_loops: dict[str, tuple[int, list[str], bool]] = field(default_factory=dict)
    stack_bytes: dict[str, int] = field(default_factory=dict)
    regalloc_spill_notes: list[str] = field(default_factory=list)
    missing_bank_annotations: int = 0
    missing_bank_remarks: int = 0
    warnings: list[str] = field(default_factory=list)
    pass_failed_warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    parse_error: str | None = None
    trip_count: int = 32  # fallback trip count for loops with unknown lc

    @property
    def pipelined_loops(self) -> list[LoopReport]:
        return [loop for loop in self.loops if loop.sched != "none"]

    @property
    def best_ii(self) -> int | None:
        return min((loop.ii for loop in self.pipelined_loops if loop.ii is not None), default=None)

    @property
    def best_post_ra_ii(self) -> int | None:
        values = [
            loop.ii
            for loop in self.pipelined_loops
            if loop.sched == "post-RA" and loop.ii is not None
        ]
        return min(values, default=None)


# -- Demangling ---------------------------------------------------------------

_demangle_cache: dict[str, str] = {}


def _demangle_tool() -> str | None:
    try:
        from artlite.operators.base import CLANGPP

        llvm_cxxfilt = Path(CLANGPP).with_name("llvm-cxxfilt")
        if llvm_cxxfilt.exists():
            return str(llvm_cxxfilt)
    except (ImportError, AttributeError, SystemExit):
        pass
    return shutil.which("llvm-cxxfilt") or shutil.which("c++filt")


def _demangle(mangled: str) -> str:
    if mangled in _demangle_cache:
        return _demangle_cache[mangled]
    tool = _demangle_tool()
    demangled = mangled
    if tool:
        try:
            result = subprocess.run(
                [tool, mangled],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                demangled = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    _demangle_cache[mangled] = demangled
    return demangled


# `(.exit186.7)`: the suffix of a clone or of an inlined function's exit block
_CLONE_SUFFIX = re.compile(r"\s+\((\.[^()]+)\)$")
# operator names hold brackets and spaces the signature scan must not see
_OPERATOR_NAME = re.compile(
    r"\boperator(?:\(\)|\[\]|<=>|->\*|<<=|>>=|->|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|[-+*/%^&|~!=<>,]=?"
    r'|""\s*\w+|\s+[^()]+?(?=\())'
)
_QUALIFIERS = re.compile(r"(?:\s*(?:const|volatile|&&|&|noexcept))*\s*")


def _top_level(text: str) -> list[tuple[int, str]]:
    """(index, char) of the characters outside parentheses and template arguments."""
    out = []
    parens = angles = 0
    for index, char in enumerate(text):
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif parens == 0 and char == "<":
            angles += 1
        elif parens == 0 and char == ">":
            angles -= 1
        elif parens == 0 and angles == 0:
            out.append((index, char))
    return out


def _function_name(demangled: str) -> str:
    """Qualified name of a demangled function, with its template arguments.

    Drops the return type, the parameters, cv/ref qualifiers and the anonymous
    namespace: ``void (anonymous namespace)::srec<32, 4>(unsigned char const*)``
    is ``srec<32, 4>``. A clone suffix stays (``f.cold``)."""
    suffix = ""
    clone = _CLONE_SUFFIX.search(demangled)
    if clone:
        demangled, suffix = demangled[: clone.start()], clone.group(1)
    operators: list[str] = []

    def protect(match: re.Match[str]) -> str:
        operators.append(match.group(0))
        return f"\0{len(operators) - 1}\0"

    text = _OPERATOR_NAME.sub(protect, demangled)
    # the parameter list: the last top-level (...) that only qualifiers follow
    parens = angles = 0
    name_end = len(text)
    group_start = 0
    for index, char in enumerate(text):
        if char == "(":
            if parens == 0 and angles == 0:
                group_start = index
            parens += 1
        elif char == ")":
            parens -= 1
            if parens == 0 and angles == 0 and _QUALIFIERS.fullmatch(text, index + 1):
                name_end = group_start
                break
        elif parens == 0 and char == "<":
            angles += 1
        elif parens == 0 and char == ">":
            angles -= 1
    name = text[:name_end]
    # a return type (template functions) ends at the last top-level space
    spaces = [index for index, char in _top_level(name) if char == " "]
    if spaces:
        name = name[spaces[-1] + 1 :]
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub("\0(\\d+)\0", lambda match: operators[int(match.group(1))], name)
    return name + suffix


def _unqualified(name: str) -> str:
    """Last component of a qualified name, without its template arguments."""
    separators = [index for index, char in _top_level(name) if char == ":"]
    if separators:
        name = name[separators[-1] + 1 :]
    if name.endswith(">") and "<" in name and not name.startswith("operator"):
        name = name[: name.index("<")]
    return name


def _demangle_short(mangled: str) -> str:
    if not mangled.startswith("_Z"):
        return mangled
    full = _demangle(mangled)
    if full == mangled:
        return mangled
    return _function_name(full) or mangled


def _block_label(block: str) -> str:
    """A block name for printing: an inlined function's exit block ``_ZN3aie...3popEv.exit186.7`` is
    ``pop.exit186.7``."""
    if not block.startswith("_Z"):
        return block
    full = _demangle(block)
    if full == block:
        return block
    clone = _CLONE_SUFFIX.search(full)
    suffix = clone.group(1) if clone else ""
    name = _function_name(full[: clone.start()] if clone else full)
    return (_unqualified(name) or name) + suffix


# -- Compile ------------------------------------------------------------------

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


def _run_compiler(command: list[str]) -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "compiler timed out after 120 seconds\n"
        return stdout, stderr, 124
    return result.stdout or "", result.stderr or "", result.returncode


def _compile_kernel(
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


# -- Parse YAML ---------------------------------------------------------------


def _args_to_dict(
    args: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Flatten LLVM's list-of-single-key dictionaries without losing repeats."""
    if isinstance(args, Mapping):
        args = [args]
    out: dict[str, Any] = {}
    for item in args or []:
        if not isinstance(item, Mapping):
            continue
        for key, value in item.items():
            key = str(key)
            if key not in out:
                out[key] = value
            elif isinstance(out[key], list):
                out[key].append(value)
            else:
                out[key] = [out[key], value]
    return out


def _arg_values(args: Mapping[str, Any], key: str) -> list[Any]:
    value = args.get(key)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _arg_text(args: Mapping[str, Any], key: str) -> str:
    return " ".join(str(value) for value in _arg_values(args, key))


def _first_text(args: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        for value in _arg_values(args, key):
            text = str(value).strip()
            if text:
                return text
    return ""


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, list):
        value = value[-1] if value else None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _arg_int(
    args: Mapping[str, Any], key: str, default: int | None = None
) -> int | None:
    values = _arg_values(args, key)
    return _as_int(values[-1], default) if values else default


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def _add_block_remark(
    report: KernelReport, *, function: str, args: Mapping[str, Any], seen: set[tuple[str, int]]
) -> None:
    """One aie-asm-printer region (in layout order): region 0 starts a block, the others extend it.

    ``report.blocks`` sums the blocks of one IR name (clones keep the name)."""
    values = _arg_values(args, "BasicBlock")
    block = _normalise_block(str(values[0])) if values else ""
    bundles = _arg_int(args, "BundleCount")
    offset = _arg_int(args, "Offset")
    if bundles is None or (offset is not None and (function, offset) in seen):
        return  # (a repeated remark)
    if offset is not None:
        seen.add((function, offset))
    byte_count = _arg_int(args, "ByteCount", 0) or 0
    region = _arg_int(args, "RegionIndex")
    layout = report.layout.setdefault(function, [])
    if not (region and layout and layout[-1].name == block):
        layout.append(BlockInfo(bundles=0, bytes=0, regions=0, name=block))
        ii = _arg_int(args, "II")
        if ii is not None:
            layout[-1].schedule = (ii, _arg_int(args, "NS", 1) or 1)
    targets = [layout[-1]]
    if block:
        targets.append(report.blocks.setdefault(f"{function}::{block}", BlockInfo(0, 0, regions=0, name=block)))
    for info in targets:
        info.bundles += bundles
        info.bytes += byte_count
        info.regions += 1
        info.region_bundles.append(bundles)
        info.region_bytes.append(byte_count)
        if region is not None:
            info.region_indices.add(region)


def _load_yaml_documents(yaml_path: Path) -> list[dict[str, Any]]:
    text = _sanitise_yaml(yaml_path.read_text())
    try:
        documents = yaml.load_all(text, Loader=_LLVMLoader)
        return [doc for doc in documents if isinstance(doc, dict)]
    except yaml.YAMLError:
        fallback = text.replace("\t", "    ")
        documents = yaml.load_all(fallback, Loader=_LLVMLoader)
        return [doc for doc in documents if isinstance(doc, dict)]


def _source_ref(doc: Mapping[str, Any]) -> str:
    debug_loc = doc.get("DebugLoc")
    if not isinstance(debug_loc, Mapping):
        return ""
    file = str(debug_loc.get("File", ""))
    line = debug_loc.get("Line")
    if not file or not line:
        return ""
    return f"{Path(file).name}:{line}"


def _debug_location(doc: Mapping[str, Any]) -> tuple[str, int, int] | None:
    """(file, line, column) of a remark."""
    debug_loc = doc.get("DebugLoc")
    if not isinstance(debug_loc, Mapping):
        return None
    file = str(debug_loc.get("File", ""))
    line = _as_int(debug_loc.get("Line"))
    if not file or not line:
        return None
    return file, line, _as_int(debug_loc.get("Column"), 0) or 0


_GENERIC_REJECTS = {"Failed to pipeline loop"}


def _final_remark_loop(function: str, pass_name: str, args: Mapping[str, Any]) -> LoopReport:
    """The loop of an AIEInterBlockScheduling final remark."""
    pipeliner = _first_text(args, "Pipeliner").lower()
    sched = {"postpipeliner": "post-RA", "prepipeliner": "pre-RA"}.get(
        pipeliner, "post-RA" if pass_name == "postpipeliner" else "none"
    )
    block = _first_text(args, "Loop", "BasicBlock", "Block")
    number = re.match(r"^bb\.(\d+)(?:\.|$)", block)
    return LoopReport(
        function=function,
        block="" if number and number.end() == len(block) else _normalise_block(block) or "<unknown>",
        mbb=int(number.group(1)) if number else None,
        sched_block=int(number.group(1)) if number else None,
        sched=sched,
        ns=_arg_int(args, "NS", 1) or 1,
        # For unpipelined loops the IBS II field is the body length.
        ii=_arg_int(args, "II"),
        prologue_bundles=_arg_int(args, "PrologueBundles"),
        epilogue_bundles=_arg_int(args, "EpilogueBundles"),
        prologue_block=_first_text(args, "Prologue"),
        epilogue_block=_first_text(args, "Epilogue"),
    )


def _note_pipeliner_remark(
    report: KernelReport,
    *,
    open_loops: dict[tuple[str, str, int, int], SourceLoop],
    doc: Mapping[str, Any],
    args: Mapping[str, Any],
) -> None:
    """Add a pre-RA pipeliner remark to the source loop at its location.

    A loop's remarks are consecutive and end with a terminal one; a remark at
    the same location after that belongs to another loop (a loop of an inlined
    function, a versioned copy)."""
    function = str(doc.get("Function", "") or "")
    location = _debug_location(doc)
    if not function or location is None:
        return
    key = (function, *location)
    loop = open_loops.get(key)
    if loop is None:
        loop = SourceLoop(function=function, path=location[0], line=location[1], column=location[2])
        report.source_loops.setdefault(function, []).append(loop)
        open_loops[key] = loop
    tag = str(doc.get("_tag", ""))
    text = _arg_text(args, "String").strip()
    done = False
    if str(doc.get("Name", "")) == "canPipelineLoop":
        if tag == "!Missed":  # "Failed to pipeline loop"
            done = True
        elif text:
            reason = text
            num_blocks = _arg_int(args, "NumBlocks")
            if num_blocks is not None:
                loop.num_blocks = num_blocks
                reason = f"{text.rstrip(': ').rstrip()}: {num_blocks}"
            if reason not in loop.reasons:
                loop.reasons.append(reason)
    elif "Initiation Interval" in text and "Schedule found" in text:
        loop.search_ii = _arg_int(args, "II")
        loop.search_block = _normalise_block(_first_text(args, "BasicBlock"))
    elif "Minimal Initiation Interval too large" in text:
        loop.mii_note = (
            f"pre-RA search cap exceeded: MII={_arg_int(args, 'MII')} > limit {_arg_int(args, 'SwpMaxMii')}"
        )
        done = True
    elif tag == "!Passed":  # "Pipelined succesfully!"
        loop.pipelined = True
        done = True
    else:  # no schedule found, or one not worth applying
        reason = text or "pipeline schedule failed"
        if reason not in loop.reasons:
            loop.reasons.append(reason)
        done = True
    if done:
        open_loops.pop(key, None)


def parse_yaml_remarks(yaml_path: Path) -> KernelReport:
    """Parse LLVM/AIE optimization records.

    ``report.loops`` gets the loops of the final scheduling remarks (single-block
    loops); ``report.source_loops`` the loops of the pre-RA pipeliner remarks. The
    assembly turns them into the final code's loop tree (``_enrich_from_assembly``)."""
    report = KernelReport(source=yaml_path)
    documents = _load_yaml_documents(yaml_path)
    open_loops: dict[tuple[str, str, int, int], SourceLoop] = {}
    block_offsets: set[tuple[str, int]] = set()
    regalloc_loops: dict[tuple[str, str, int, int], list[tuple[int, int]]] = {}
    regalloc_functions: dict[str, tuple[int, int, str]] = {}

    for doc in documents:
        pass_name = str(doc.get("Pass", ""))
        name = str(doc.get("Name", ""))
        function = str(doc.get("Function", "") or "")
        args = _args_to_dict(doc.get("Args"))

        # Final per-loop result from AIEInterBlockScheduling (has Pipeliner key).
        if (
            pass_name in {"pipeliner", "postpipeliner"}
            and name == "schedule"
            and ("Pipeliner" in args or pass_name == "postpipeliner")
        ):
            loop = _final_remark_loop(function, pass_name, args)
            loop.source_ref = _source_ref(doc)
            report.loops.append(loop)
            continue

        # Pre-RA MachinePipeliner: rejections and search plans, by source location.
        if pass_name == "pipeliner" and name in {"canPipelineLoop", "schedule"}:
            _note_pipeliner_remark(report, open_loops=open_loops, doc=doc, args=args)
            continue

        # ZOL status.
        if pass_name == "aie-hardware-loops" and name == "analysis":
            zero_overhead = _as_bool(_first_text(args, "Zero-Overhead-Loop"))
            for block in _arg_values(args, "BasicBlock"):
                report.hardware_loops.append(
                    HardwareLoop(
                        function=function,
                        block=_normalise_block(str(block)),
                        loop_id=_first_text(args, "LoopID"),
                        zero_overhead=zero_overhead,
                    )
                )

        # Ground-truth block sizes.
        if pass_name == "aie-asm-printer" and name == "analysis":
            _add_block_remark(report, function=function, args=args, seen=block_offsets)

        if pass_name == "prologepilog" and name == "StackSize":
            stack_bytes = _arg_int(args, "NumStackBytes")
            if stack_bytes is not None and function:
                report.stack_bytes[function] = stack_bytes

        # Register allocation: spills and reloads of registers (the last run of the allocator counts all
        # register classes); a loop's count includes its inner loops.
        if pass_name == "regalloc" and "SpillReloadCopies" in name and function:
            counts = (_arg_int(args, "NumSpills", 0) or 0, _arg_int(args, "NumReloads", 0) or 0)
            location = _debug_location(doc)
            if not name.startswith("Loop"):
                regalloc_functions[function] = (*counts, _source_ref(doc))
            elif location is not None:
                regalloc_loops.setdefault((function, *location), []).append(counts)

        if pass_name == "aie-multi-slot-pseudo" and name == "missing-memory-bank":
            report.missing_bank_remarks += 1

    _attach_regalloc(report, regalloc_loops)
    for function, (spills, reloads, ref) in regalloc_functions.items():
        if spills or reloads:
            where = f" @{ref}" if ref else ""
            report.regalloc_spill_notes.append(
                f"{_demangle_short(function)} (function{where}): the register allocator spilled "
                f"{spills}, reloaded {reloads} register(s)"
            )
    return report


def _attach_regalloc(
    report: KernelReport, loops_at: dict[tuple[str, str, int, int], list[tuple[int, int]]]
) -> None:
    """Give the source loops the register allocator's counts at their location.

    Each run of the allocator reports every loop with copies, spills or reloads,
    inner loops first like the pipeliner: the last run's remarks, one per loop at
    that location. A loop only the allocator reported becomes a source loop."""
    for (function, path, line, column), counts in loops_at.items():
        loops = [
            loop
            for loop in report.source_loops.get(function, [])
            if (loop.path, loop.line, loop.column) == (path, line, column)
        ]
        if not loops:
            loops = [SourceLoop(function=function, path=path, line=line, column=column)]
            report.source_loops.setdefault(function, []).extend(loops)
        if len(loops) == 1 or len(counts) % len(loops) == 0:
            for loop, count in zip(loops, counts[-len(loops) :]):
                loop.regalloc = count


# -- Source loop statements ---------------------------------------------------


@dataclass(frozen=True)
class _Token:
    text: str
    line: int
    column: int  # 1-based, as in DebugLocs


_TOKEN = re.compile(
    r"(?P<skip>\s+|//[^\n]*|/\*.*?\*/)"
    r'|(?P<literal>[uUL8]*R"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)"'
    r'|[uUL8]*"(?:\\.|[^"\\\n])*"|[uUL8]*\'(?:\\.|[^\'\\\n])+\'|\.?\d(?:\'?[\w.]|[eEpP][-+])*)'
    r"|(?P<word>[A-Za-z_]\w*)"
    r"|(?P<punct>.)",
    re.DOTALL,
)
_OPENING = {"(", "[", "{"}
_CLOSING = {")", "]", "}"}
_STATEMENT_KEYWORDS = {"for", "while", "do", "if", "switch", "{"}


def _tokenize(text: str) -> list[_Token]:
    """C++ tokens without comments and preprocessor directives."""
    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    tokens = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        assert match is not None  # (punct matches any character)
        position = match.end()
        if match.lastgroup == "skip":
            continue
        line = bisect.bisect_right(line_starts, match.start())
        line_start = line_starts[line - 1]
        if match.group() == "#" and not text[line_start : match.start()].strip():
            while True:  # a directive: up to the end of its (continued) line
                newline = text.find("\n", position)
                if newline < 0:
                    position = len(text)
                    break
                position = newline + 1
                end = newline - 1 if newline and text[newline - 1] == "\r" else newline
                if not (end and text[end - 1] == "\\"):
                    break
            continue
        tokens.append(_Token(match.group(), line, match.start() - line_start + 1))
    return tokens


def _closing(tokens: list[_Token], index: int) -> int:
    """Index of the bracket closing tokens[index]."""
    depth = 0
    for position in range(index, len(tokens)):
        text = tokens[position].text
        if text in _OPENING:
            depth += 1
        elif text in _CLOSING:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("unbalanced brackets")


def _pragma_like(text: str) -> bool:
    """AIE_LOOP_* and _Pragma(...) in front of a statement."""
    return text == "_Pragma" or re.fullmatch(r"[A-Z][A-Z0-9_]*", text) is not None


def _statement_end(tokens: list[_Token], index: int) -> int:
    """Index of the last token of the statement starting at tokens[index]."""
    text = tokens[index].text
    if text == "{":
        return _closing(tokens, index)
    if text in {"for", "while", "switch"}:
        return _statement_end(tokens, _closing(tokens, index + 1) + 1)
    if text == "if":
        condition = index + 2 if tokens[index + 1].text == "constexpr" else index + 1
        end = _statement_end(tokens, _closing(tokens, condition) + 1)
        if end + 1 < len(tokens) and tokens[end + 1].text == "else":
            return _statement_end(tokens, end + 2)
        return end
    if text == "do":
        body_end = _statement_end(tokens, index + 1)  # then: while (...) ;
        return _closing(tokens, body_end + 2) + 1
    if _pragma_like(text):
        following = index + 1
        if tokens[following].text == "(":
            following = _closing(tokens, following) + 1
        if tokens[following].text in _STATEMENT_KEYWORDS or _pragma_like(tokens[following].text):
            return _statement_end(tokens, following)
    if text == "[" and tokens[index + 1].text == "[":  # [[attributes]]
        return _statement_end(tokens, _closing(tokens, index) + 1)
    depth = 0
    for position in range(index, len(tokens)):
        text = tokens[position].text
        if text in _OPENING:
            depth += 1
        elif text in _CLOSING:
            depth -= 1
            if depth < 0:
                break
        elif text == ";" and depth == 0:
            return position
    raise ValueError("statement without an end")


@dataclass(frozen=True)
class _Statement:
    """A loop statement of a source file."""

    line: int  # of its keyword
    column: int
    end_line: int
    exits: bool = False  # a break, return or goto leaves it before its condition does
    scope: tuple[int, int] | None = None  # lines of the function body around it


_CONTROL = {"if", "for", "while", "switch", "catch", "constexpr"}
_FUNCTION_QUALIFIERS = {"const", "noexcept", "override", "final", "volatile", "&", "&&"}


def _function_bodies(tokens: list[_Token]) -> list[tuple[int, int]]:
    """Token ranges of the function (and lambda) bodies: a brace after a parameter list and its qualifiers, not
    after a control statement's condition."""
    bodies = []
    for index, token in enumerate(tokens):
        if token.text != "{" or index == 0:
            continue
        before = index - 1
        while before > 0 and tokens[before].text in _FUNCTION_QUALIFIERS:
            before -= 1
        if tokens[before].text != ")":
            continue
        depth, opening = 0, before
        while opening >= 0:  # the parenthesis it closes
            text = tokens[opening].text
            depth += text == ")"
            depth -= text == "("
            if depth == 0:
                break
            opening -= 1
        if opening <= 0 or tokens[opening - 1].text in _CONTROL:
            continue
        try:
            bodies.append((index, _closing(tokens, index)))
        except ValueError:
            continue
    return bodies


def _loop_statements(tokens: list[_Token]) -> list[_Statement]:
    """The loop statements of a file."""
    spans = []  # token ranges of the loops and switches: a break leaves the innermost one
    closing: set[int] = set()  # the `while` of a do statement
    for index, token in enumerate(tokens):
        if token.text not in {"for", "while", "do", "switch"} or index in closing:
            continue
        try:
            end = _statement_end(tokens, index)
            if token.text == "do":
                closing.add(_statement_end(tokens, index + 1) + 1)
        except (ValueError, IndexError):
            continue
        spans.append((index, end))
    exits: set[int] = set()
    for index, token in enumerate(tokens):
        if token.text == "break":
            around = [span for span in spans if span[0] < index <= span[1]]
            if around:
                exits.add(max(around)[0])
        elif token.text in {"return", "goto"}:
            exits.update(start for start, end in spans if start < index <= end)
    bodies = _function_bodies(tokens)

    def scope(start: int) -> tuple[int, int] | None:
        around = [body for body in bodies if body[0] < start <= body[1]]
        if not around:
            return None
        first, last = max(around)  # (the innermost: a lambda in a function)
        return tokens[first].line, tokens[last].line

    return [
        _Statement(tokens[start].line, tokens[start].column, tokens[end].line, start in exits, scope(start))
        for start, end in spans
        if tokens[start].text != "switch"
    ]


def _loop_statement(statements: list[_Statement], line: int, column: int) -> _Statement | None:
    """The loop statement of a remark location.

    The location is the statement's keyword, unless the loop's preheader has
    no location: then it is the header's branch, inside the statement."""
    for statement in statements:
        if statement.line == line and (statement.column == column or not column):
            return statement
    around = [
        statement
        for statement in statements
        if (statement.line, statement.column) <= (line, column) and line <= statement.end_line
    ]
    return max(around, key=lambda statement: (statement.line, statement.column), default=None)


def _around(statements: list[_Statement], loop: SourceLoop) -> _Statement | None:
    """The innermost loop statement strictly around a located source loop."""
    if loop.end_line is None:
        return None
    around = [
        statement
        for statement in statements
        if (statement.line, statement.column) < (loop.line, loop.column) and statement.end_line >= loop.end_line
    ]
    return max(around, key=lambda statement: (statement.line, statement.column), default=None)


def _locate(loop: SourceLoop, statement: _Statement) -> None:
    loop.line, loop.column, loop.end_line, loop.exits, loop.scope = (
        statement.line,
        statement.column,
        statement.end_line,
        statement.exits,
        statement.scope,
    )


def _read_statements(loops: list[SourceLoop], *, source: Path) -> None:
    """Locate each source loop's statement: keyword, last line and constant trip count.

    An outer loop whose body starts with an inner loop can report the inner loop
    statement's location (its preheader's branch has it): of the loops at one
    statement, one with more blocks encloses the others and takes the loop
    statement around them (copies of one loop have the same block count)."""
    statements_by_path: dict[str, list[_Statement] | None] = {}
    for loop in loops:
        if loop.path not in statements_by_path:
            try:
                text = _source_path(loop.path, source=source).read_text(errors="replace")
                statements_by_path[loop.path] = _loop_statements(_tokenize(text))
            except OSError:
                statements_by_path[loop.path] = None
        statements = statements_by_path[loop.path]
        statement = _loop_statement(statements, loop.line, loop.column) if statements is not None else None
        if statement is not None:
            _locate(loop, statement)
    at_statement: dict[tuple[str, int, int], list[SourceLoop]] = {}
    for loop in loops:
        if loop.end_line is not None:
            at_statement.setdefault((loop.path, loop.line, loop.column), []).append(loop)
    for (path, _, _), group in at_statement.items():
        sizes = sorted({loop.num_blocks or 1 for loop in group})
        for loop in group:
            for _ in range(sizes.index(loop.num_blocks or 1)):
                around = _around(statements_by_path[path] or [], loop)
                if around is None:
                    break
                _locate(loop, around)
    for loop in loops:
        if statements_by_path[loop.path] is not None:
            loop.trip = _parse_source_trip(_source_path(loop.path, source=source), loop.line)


def _source_path(path: str, *, source: Path) -> Path:
    """A DebugLoc file: the measured source when it names it relatively (as the compile command did)."""
    candidate = Path(path)
    return source if candidate.name == source.name and not candidate.is_absolute() else candidate


# -- Assembly -----------------------------------------------------------------

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_FUNCTION_START = re.compile(r"^([\w.$]+):.*// @(.+?)\s*$")
# calls: `jl`, and tail calls (an unconditional `j` to a function symbol, not a local `.L` label)
_CALL = re.compile(r"\b(?:jl\s+#?([\w.$]+)|j\s+#([A-Za-z_][\w$]*))")
_ASM_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
_ASM_LOC = re.compile(r"^\.loc\s+(\d+)\s+(\d+)(?:\s+(\d+))?")
_ASM_FRAME = re.compile(r"(\S+?):(\d+):\d+")  # a frame of the inlining chain in a .loc's comment
_ASM_BLOCK = re.compile(r"^(?:(\.LBB\d+_(\d+)):|//\s*%bb\.(\d+):)\s*(?://\s*%(\S+))?")
_ASM_HEADER = re.compile(r"=>\s*This (?:Inner )?Loop Header: Depth=(\d+)")
_ASM_PARENT = re.compile(r"Parent Loop BB\d+_(\d+) Depth=(\d+)")
_ASM_IN_LOOP = re.compile(r"in Loop: Header=BB\d+_(\d+) Depth=(\d+)")
_ASM_LOOP_START = re.compile(r"\bls,\s*#?(\.L\w+)")
_ASM_BRANCH = re.compile(r"\bj(?:n?z|nzd)\s+[^;]*?#(\.L\w+)")  # conditional jumps
# The asm printer's comment on a spill/reload instruction; the comments of one
# bundle's instructions run together ("64-byte Folded Reload64-byte Folded Spill").
_SPILL_COMMENT = re.compile(r"(?:(\d+)-byte|Unknown-size) (?:Folded )?(Spill|Reload)")


@dataclass
class _AsmBlock:
    number: int  # machine basic block number
    name: str  # IR name, "" if none
    label: str | None  # ".LBBf_n" when labelled
    loop: int | None = None  # header number of the innermost loop containing the block
    header: bool = False
    depth: int = 0
    parent: int | None = None  # a header's enclosing loop (header number)
    bundles: list[tuple[list[str], list[str]]] = field(default_factory=list)  # ([labels], [slots])
    # per bundle with a location: its (file, line) and the call sites it is inlined at, innermost first
    locs: list[tuple[tuple[str, int], ...]] = field(default_factory=list)
    spills: int = 0  # spill stores (asm instructions)
    reloads: int = 0
    spill_bytes: int = 0
    reload_bytes: int = 0
    calls: list[str] = field(default_factory=list)  # callee symbols
    jumps: list[str] = field(default_factory=list)  # targets of its conditional jumps
    gotos: list[str] = field(default_factory=list)  # targets of its unconditional jumps to local labels
    ends: bool = False  # an unconditional jump or a return: no fall-through

    @property
    def ref(self) -> str:
        return self.label or f"%bb.{self.number}"


@dataclass
class _AsmFunction:
    name: str
    blocks: list[_AsmBlock] = field(default_factory=list)
    loop_starts: set[str] = field(default_factory=set)  # labels written to ls: the ZOL headers


@dataclass
class _AsmLoop:
    header: int  # block number
    index: int  # layout index of the header block
    depth: int
    parent: int | None
    children: list[int] = field(default_factory=list)
    own: list[int] = field(default_factory=list)  # layout indices of the blocks not in an inner loop
    blocks: list[int] = field(default_factory=list)  # layout indices of all its blocks


def _note_block_comment(block: _AsmBlock, comment: str) -> None:
    """Loop structure and IR name from a comment line of a block (or of its label line)."""
    header = _ASM_HEADER.search(comment)
    parent = _ASM_PARENT.search(comment)
    in_loop = _ASM_IN_LOOP.search(comment)
    name = re.fullmatch(r"\s*%(\S+)\s*", comment)
    if header:
        block.header, block.loop, block.depth = True, block.number, int(header.group(1))
    elif parent:
        block.parent = int(parent.group(1))  # outermost first: the last one is the enclosing loop
    elif in_loop:
        block.loop, block.depth = int(in_loop.group(1)), int(in_loop.group(2))
    elif name and not block.name and not block.bundles:
        block.name = _normalise_block(name.group(1))


def _inlining_chain(file: str, line: int, comment: str) -> tuple[tuple[str, int], ...]:
    """A .loc's location and the call sites it is inlined at, innermost first, from the asm printer's comment
    (``a.hpp:1098:29 @[ kernel_lib.h:41:27 @[ kernel.cc:315:9 ] ]``); call sites of line 0 are dropped."""
    frames = [(Path(path).name, int(number)) for path, number in _ASM_FRAME.findall(comment)]
    if not frames or frames[0] != (file, line):
        return ((file, line),)
    return tuple(frame for frame in frames if frame[1])


def _parse_assembly(text: str) -> dict[str, _AsmFunction]:
    """Blocks of each function: bundles, source lines, loop structure, spills and calls."""
    files: dict[str, str] = {}
    functions: dict[str, _AsmFunction] = {}
    function: _AsmFunction | None = None
    block: _AsmBlock | None = None
    location: tuple[tuple[str, int], ...] | None = None
    labels: list[str] = []
    for raw in text.splitlines():
        start = _FUNCTION_START.match(raw)
        if start:
            function = functions.setdefault(start.group(1), _AsmFunction(start.group(1)))
            block, location, labels = None, None, []
            continue
        file_directive = _ASM_FILE.match(raw)
        if file_directive:
            files[file_directive.group(1)] = Path(file_directive.group(3) or file_directive.group(2)).name
            continue
        if function is None:
            continue
        if raw.startswith(".Lfunc_end") or "-- End function" in raw:
            function = None
            continue
        block_start = _ASM_BLOCK.match(raw)
        if block_start:
            number = int(block_start.group(2) or block_start.group(3))
            block = _AsmBlock(number, _normalise_block(block_start.group(4) or ""), block_start.group(1))
            function.blocks.append(block)
            # (the location stays: the asm printer emits a .loc when it changes, in text order)
            labels = [block_start.group(1)] if block_start.group(1) else []
            # a block without IR name has its loop comment on the label line
            _note_block_comment(block, raw[block_start.end() :].partition("//")[2])
            continue
        if block is None:
            continue
        code, _, comment = raw.partition("//")
        code = code.strip()
        if not code:
            _note_block_comment(block, comment)
            continue
        loc = _ASM_LOC.match(code)
        if loc:
            line = int(loc.group(2))
            location = _inlining_chain(files.get(loc.group(1), ""), line, comment) if line else None
            continue
        label = re.match(r"^(\.L\w+|bb\w*):\s*(.*)$", code)
        while label:
            labels.append(label.group(1))
            code = label.group(2).strip()
            label = re.match(r"^(\.L\w+|bb\w*):\s*(.*)$", code)
        if not code or code.startswith("."):
            continue
        slots = [slot.strip() for slot in code.split(";") if slot.strip()]
        block.bundles.append((labels, slots))
        labels = []
        if location is not None:
            block.locs.append(location)
        loop_start = _ASM_LOOP_START.search(code)
        if loop_start:
            function.loop_starts.add(loop_start.group(1))
        call = _CALL.search(code)
        if call:
            block.calls.append(call.group(1) or call.group(2))
        block.jumps.extend(_ASM_BRANCH.findall(code))
        for slot in slots:
            mnemonic, *operands = slot.split(None, 1)
            if mnemonic in ("j", "ret"):
                block.ends = True
                target = re.match(r"#(\.L\w+)", operands[0].strip()) if operands else None
                if target:
                    block.gotos.append(target.group(1))
        for size, kind in _SPILL_COMMENT.findall(comment):
            if kind == "Spill":
                block.spills += 1
                block.spill_bytes += int(size or 0)
            else:
                block.reloads += 1
                block.reload_bytes += int(size or 0)
    return functions


def _asm_loops(function: _AsmFunction) -> dict[int, _AsmLoop]:
    """The loop tree of a function, by header block number."""
    loops = {
        block.number: _AsmLoop(block.number, index, block.depth, block.parent)
        for index, block in enumerate(function.blocks)
        if block.header
    }
    for index, block in enumerate(function.blocks):
        if block.loop in loops:
            loops[block.loop].own.append(index)
    for loop in loops.values():
        if loop.parent in loops:
            loops[loop.parent].children.append(loop.header)

    def collect(header: int) -> list[int]:
        loop = loops[header]
        return [*loop.own, *(index for child in loop.children for index in collect(child))]

    for header, loop in loops.items():
        loop.blocks = sorted(collect(header))
    return loops


_NO_RESULT = re.compile(r"(?:v?st|j|ret|nop|event|done)")  # stores and jumps: the first operand is read


# the registers a call may return a value in (llvm-aie RetCC_AIE2P): code that reads another register after a call
# relies on the callee preserving it (the compiler keeps values in caller-saved registers across calls it knows)
_RETURN_REGISTERS = {*(f"r{n}" for n in range(8)), *(f"l{n}" for n in range(8)), *(f"p{n}" for n in range(6))}
_DELAY_SLOTS = 5  # bundles after a jump or call that run before it takes effect


def _extract_zol_iters(
    bundles: list[tuple[list[str], list[str]]],
    entry: Mapping[str, int | None] | None = None,
) -> tuple[list[dict[str, int | None]], list[tuple[int, str, int]], dict[str, int | None]]:
    """Register values per bundle from the entry values, the `add.nc lc` setups, and the values at the end.

    Tracks simple immediate defs with start-of-bundle read semantics
    (validated against simulated store counts on the zero kernels); a call
    clobbers the return-value registers after its delay slots."""
    values_by_index: list[dict[str, int | None]] = []
    add_ncs: list[tuple[int, str, int]] = []
    values: dict[str, int | None] = dict(entry or {})
    clobber: int | None = None  # the bundle after a call's delay slots
    for _index, (_labels, slots) in enumerate(bundles):
        if clobber == _index:
            values = {register: value for register, value in values.items() if register not in _RETURN_REGISTERS}
            clobber = None
        writes: dict[str, int | None] = {}
        for slot in slots:
            match = re.match(r"mov[a-z]*\s+(\w+), #(-?\d+)$", slot)
            if match:
                writes[match.group(1)] = int(match.group(2))
                continue
            match = re.match(r"add\.nc\s+lc, (\w+), #(-?\d+)", slot)
            if match:
                add_ncs.append((_index, match.group(1), int(match.group(2))))
                continue
            # Any other write to a tracked register (the first operand of an
            # instruction that is no store or jump: lshl, mul, lda, ...)
            # invalidates our model.
            mnemonic, *operands = slot.split(None, 1)
            if mnemonic == "jl":
                clobber = _index + 1 + _DELAY_SLOTS
            target = operands[0].split(",", 1)[0].strip() if operands else ""
            if not _NO_RESULT.match(mnemonic) and target in values and target not in writes:
                writes[target] = None
        # add.nc reads registers with start-of-bundle semantics.
        values_by_index.append(dict(values))
        values.update(writes)
    if clobber is not None:
        values = {register: value for register, value in values.items() if register not in _RETURN_REGISTERS}
    return values_by_index, add_ncs, values


def _meet(states: Sequence[Mapping[str, int | None]]) -> dict[str, int | None]:
    """The register values all the states agree on."""
    first, *rest = states
    return {
        register: value
        for register, value in first.items()
        if value is not None and all(state.get(register) == value for state in rest)
    }


def _block_entries(function: _AsmFunction, loops: dict[int, _AsmLoop]) -> list[dict[str, int | None]]:
    """The register values at the start of each block: the meet over its predecessors (jumps, fall-through, a
    ZOL's back edge from the block of its end label), to a fixed point."""
    blocks = function.blocks
    index_of = {block.label: index for index, block in enumerate(blocks) if block.label}
    preds: list[set[int]] = [set() for _ in blocks]
    for index, block in enumerate(blocks):
        for target in [*block.jumps, *block.gotos]:
            if target in index_of:
                preds[index_of[target]].add(index)
        if not block.ends and index + 1 < len(blocks):
            preds[index + 1].add(index)
    for loop in loops.values():
        if blocks[loop.index].label in function.loop_starts:  # (a ZOL: no jump back)
            ends = [index for index in loop.blocks
                    if any(label.startswith(".L_LEnd") for labels, _ in blocks[index].bundles for label in labels)]
            preds[loop.index].update(ends or loop.blocks)
    exits: list[dict[str, int | None] | None] = [None] * len(blocks)  # None: not reached yet
    entries: list[dict[str, int | None]] = [{} for _ in blocks]
    for _ in range(4 * len(blocks) + 4):
        changed = False
        for index, block in enumerate(blocks):
            reached = [state for state in (exits[pred] for pred in preds[index]) if state is not None]
            if index and preds[index] and not reached:
                continue
            entries[index] = _meet(reached) if index and reached else {}
            state = _extract_zol_iters(block.bundles, entries[index])[2]
            if state != exits[index]:
                exits[index], changed = state, True
        if not changed:
            break
    return entries


def _zol_iterations(function: _AsmFunction, loops: dict[int, _AsmLoop]) -> dict[int, int | None]:
    """ZOL executions (lc) per loop header number.

    The lc setup precedes its loop: it binds to the first later bundle that
    starts a ZOL header block. Register values flow along the control flow
    (_block_entries): an if-chain's branch sees the values set before its
    jump, not those of the blocks laid out between."""
    zol = [
        block.header and block.number in loops and (block.label in function.loop_starts or not function.loop_starts)
        for block in function.blocks
    ]
    entries = _block_entries(function, loops)
    out: dict[int, int | None] = {}
    for index, block in enumerate(function.blocks):
        values_by_index, add_ncs, _ = _extract_zol_iters(block.bundles, entries[index])
        for position, register, imm in add_ncs:
            target = next(
                (
                    function.blocks[later].number
                    for later in range(index, len(function.blocks))
                    if zol[later] and (later > index or position == 0)
                ),
                None,
            )
            if target is None:
                continue
            base = values_by_index[position].get(register)
            count = (base + imm) if base is not None else None
            out[target] = count if target not in out or out[target] == count else None
    return out


def _place_final_remarks(
    function: _AsmFunction,
    loops: dict[int, _AsmLoop],
    remarks: list[LoopReport],
    schedules: Mapping[int, tuple[int, int]],
) -> tuple[dict[int, LoopReport], list[LoopReport]]:
    """Final remarks per loop header: first the header block the asm printer gives the remark's IR name, II and NS
    (``schedules``, per header), then the loop of the remark's IR block name, in layout order."""
    placed: dict[int, LoopReport] = {}
    leftover = []
    headers = sorted(loops, key=lambda header: (loops[header].blocks != [loops[header].index], loops[header].index))
    pending = []
    for remark in remarks:
        header = next((header for header in headers if header not in placed
                       and schedules.get(header) == (remark.ii, remark.ns)
                       and function.blocks[loops[header].index].name == remark.block), None)
        if header is None:
            pending.append(remark)
        else:
            placed[header] = remark
    for remark in pending:  # (in layout order)
        if remark.block:
            options = [
                header
                for header in headers
                if header not in placed and function.blocks[loops[header].index].name == remark.block
            ]
        else:
            options = [header for header in headers if header not in placed and header == remark.mbb]
        if options:
            placed[options[0]] = remark
        else:
            leftover.append(remark)
    return placed, leftover


def _contains(outer: SourceLoop, inner: SourceLoop) -> bool | None:
    """Whether outer's statement strictly contains inner's; None when unknown."""
    if outer.file != inner.file or outer.end_line is None or inner.end_line is None:
        return None
    if outer.scope is not None and inner.scope is not None and outer.scope != inner.scope:
        return None  # (a loop of an inlined function inside a caller's loop)
    return (outer.line, outer.column) < (inner.line, inner.column) and inner.end_line <= outer.end_line


def _match_source_loops(
    function: _AsmFunction,
    loops: dict[int, _AsmLoop],
    candidates: list[SourceLoop],
    *,
    trips: dict[int, int],
) -> dict[int, SourceLoop]:
    """Match the final loops (by header number) with the pre-RA source loops.

    ``trips``: the trip counts of the ZOLs (peeling undone)."""
    matched: dict[int, SourceLoop] = {}
    free = list(candidates)
    relatives: dict[int, tuple[set[int], set[int]]] = {}
    for header, loop in loops.items():
        ancestors = set()
        parent = loop.parent
        while parent in loops and parent not in ancestors:
            ancestors.add(parent)
            parent = loops[parent].parent
        descendants = set()
        pending = list(loop.children)
        while pending:
            child = pending.pop()
            if child not in descendants:
                descendants.add(child)
                pending.extend(loops[child].children)
        relatives[header] = ancestors, descendants

    def consistent(header: int, candidate: SourceLoop) -> bool:
        """The final loop tree and the statement nesting agree."""
        ancestors, descendants = relatives[header]
        for other, taken in matched.items():
            contains, within = _contains(candidate, taken), _contains(taken, candidate)
            if other in descendants:
                if contains is False:
                    return False
            elif other in ancestors:
                if within is False:
                    return False
            elif contains or within:
                return False
        return True

    def take(header: int, candidate: SourceLoop) -> None:
        matched[header] = candidate
        free[:] = [loop for loop in free if loop is not candidate]

    # A pre-RA schedule names its loop block.
    for candidate in list(free):
        if candidate.search_block:
            options = [
                header
                for header in loops
                if header not in matched and function.blocks[loops[header].index].name == candidate.search_block
            ]
            if len(options) == 1:
                take(options[0], candidate)

    located = [loop for loop in candidates if loop.end_line is not None]
    frames: dict[tuple[tuple[str, int], ...], tuple[str, int]] = {}

    def frame(chain: tuple[tuple[str, int], ...]) -> tuple[str, int]:
        """The innermost frame of a bundle's inlining chain in a loop statement of the function (the frames after
        it are call sites); the innermost frame when none is."""
        if chain not in frames:
            frames[chain] = next(
                (
                    (file, line)
                    for file, line in chain
                    if any(loop.file == file and loop.line <= line <= (loop.end_line or 0) for loop in located)
                ),
                chain[0],
            )
        return frames[chain]

    def own_lines(header: int) -> list[tuple[str, int]]:
        return [frame(chain) for index in loops[header].own for chain in function.blocks[index].locs]

    def preheader_lines(header: int) -> list[tuple[str, int]]:
        """Lines of the blocks laid out before the header in the enclosing loop (the loop statement's guard)."""
        out = []
        index = loops[header].index - 1
        while index >= 0 and len(out) < 64 and loops[header].index - index <= 3:
            block = function.blocks[index]
            if block.loop != loops[header].parent:
                break
            out.extend(frame(chain) for chain in block.locs)
            index -= 1
        return out

    def within(candidate: SourceLoop, line: int) -> bool:
        return candidate.line <= line <= (candidate.end_line or candidate.line)

    def at(candidate: SourceLoop, line: int) -> bool:
        return line == candidate.line

    order = sorted(loops, key=lambda header: (-loops[header].depth, loops[header].index))
    for header in order:  # inner loops first
        if header in matched:
            continue
        single = len(loops[header].blocks) == 1
        evidence: tuple[tuple[list[tuple[str, int]], Callable[[SourceLoop, int], bool], bool], ...] = (
            (own_lines(header), within, True),
            (preheader_lines(header), at, True),
            (own_lines(header), within, False),
        )
        for tier, (lines, hit, strict) in enumerate(evidence):
            options = []
            for candidate in free:
                if strict and single and candidate.num_blocks is not None and candidate.num_blocks > 1:
                    continue  # (a one-block loop is not a statement rejected as several blocks, unless nothing else
                    # fits: its own code can lie in the enclosing statement before its own)
                if tier == 1 and (
                    single != (candidate.num_blocks is None)
                    or any(_contains(candidate, other) for other in free if other is not candidate)
                ):
                    continue  # a preheader lies in the enclosing loop: it also has that loop's statement lines
                hits = sum(1 for file, line in lines if file == candidate.file and hit(candidate, line))
                if hits and consistent(header, candidate):
                    size = (candidate.end_line - candidate.line) if candidate.end_line is not None else 1 << 30
                    options.append(((single == (candidate.num_blocks is None), hits, -size), candidate))
            if options:
                take(header, max(options, key=lambda option: option[0])[1])
                break

    # No source lines (all inlined code): a ZOL's trip count against the constant bound of a for-header.
    for header in order:
        trip = trips.get(header)
        if header in matched or trip is None:
            continue
        options = [loop for loop in free if loop.trip == trip and consistent(header, loop)]
        rivals = [other for other in order if other != header and other not in matched and trips.get(other) == trip]
        if len(options) == 1 and not rivals:
            take(header, options[0])

    # Branch loops: the block count of a "Not a single basic block" rejection, when it singles one out.
    for header in order:
        count = len(loops[header].blocks)
        if header in matched or count == 1:
            continue
        options = [loop for loop in free if loop.num_blocks == count and consistent(header, loop)]
        rivals = [
            other for other in order if other != header and other not in matched and len(loops[other].blocks) == count
        ]
        if len(options) == 1 and not rivals:
            take(header, options[0])

    # Elimination: a loop with one consistent source loop that no other loop can take.
    changed = True
    while changed:
        changed = False
        rest = [header for header in order if header not in matched]
        options = {header: [loop for loop in free if consistent(header, loop)] for header in rest}
        for header in rest:
            if len(options[header]) == 1:
                candidate = options[header][0]
                if not any(any(loop is candidate for loop in options[other]) for other in rest if other != header):
                    take(header, candidate)
                    changed = True
                    break
    return matched


def _statement_fallback(
    function: _AsmFunction,
    loops: dict[int, _AsmLoop],
    matched: dict[int, SourceLoop],
    statements: list[_Statement],
    file: str,
) -> dict[int, _Statement]:
    """Loop statements for the final loops no pre-RA remark names (e.g. an outer loop the pre-RA pipeliner never
    looked at): the innermost unmatched statement of the measured file around all the loop's own lines, outside
    the statements of the loops matched inside it."""
    taken = {(loop.line, loop.column) for loop in matched.values() if loop.file == file}
    out: dict[int, _Statement] = {}
    for header in sorted(loops, key=lambda header: -loops[header].depth):  # inner loops first
        if header in matched:
            continue
        lines = [chain[0][1] for index in loops[header].own for chain in function.blocks[index].locs
                 if chain[0][0] == file]
        if not lines:
            continue
        inner = [
            (loop.line, loop.end_line)
            for child, loop in matched.items()
            if child in _descendants(loops, header) and loop.file == file and loop.end_line is not None
        ]
        options = [
            statement
            for statement in statements
            if (statement.line, statement.column) not in taken
            and statement.line <= min(lines)
            and max(lines) <= statement.end_line
            and all(statement.line < line and end <= statement.end_line for line, end in inner
                    if statement.scope is None or statement.scope[0] <= line <= statement.scope[1])
        ]
        if options:
            statement = max(options, key=lambda statement: (statement.line, statement.column))
            out[header] = statement
            taken.add((statement.line, statement.column))
    return out


def _descendants(loops: dict[int, _AsmLoop], header: int) -> set[int]:
    out: set[int] = set()
    pending = list(loops[header].children)
    while pending:
        child = pending.pop()
        if child not in out:
            out.add(child)
            pending.extend(loops[child].children)
    return out


def _call_site(chains: list[tuple[tuple[str, int], ...]], loop: SourceLoop, file: str) -> str:
    """Where an inlined helper's loop (a statement of another file) is called in the measured file: the most
    frequent frame of that file after the frame in the loop's statement ("kernel.cc:L"), "" when none."""
    sites: Counter[int] = Counter()
    for chain in chains:
        for index, (name, line) in enumerate(chain):
            if name == loop.file and loop.line <= line <= (loop.end_line or loop.line):
                site = next((number for other, number in chain[index + 1 :] if other == file), None)
                if site is not None:
                    sites[site] += 1
                break
    return f"{file}:{sites.most_common(1)[0][0]}" if sites else ""


def _attribute(loop: LoopReport, source_loop: SourceLoop, *, source: Path) -> None:
    """Copy a source loop's location and pre-RA findings to its final loop."""
    loop.kernel_line = source_loop.line if source_loop.file == source.name else None
    loop.source_ref = source_loop.ref
    loop.search_ii = source_loop.search_ii
    loop.search_mii_note = source_loop.mii_note
    loop.prereject_reasons = [reason for reason in source_loop.reasons if reason not in _GENERIC_REJECTS]
    loop.early_exit = source_loop.exits
    if source_loop.regalloc is not None:
        loop.regalloc_spills, loop.regalloc_reloads = source_loop.regalloc


def _remark_only_loop(report: KernelReport, loop: LoopReport) -> LoopReport:
    """A final remark not located in the assembly: sizes by block name, ZOL from the hardware-loop remarks."""
    block_info = report.blocks.get(f"{loop.function}::{loop.block}")
    if block_info is not None:
        loop.body_bundles = block_info.bundles
        loop.body_bytes = block_info.bytes
    flags = [hw.zero_overhead for hw in report.hardware_loops if (hw.function, hw.block) == loop.key]
    if flags:
        loop.zol = all(bool(flag) for flag in flags)
    loop.source_ref = ""
    return loop


def _enrich_from_assembly(report: KernelReport) -> None:
    """Replace the remark loops by the final code's loop tree and add what the assembly shows."""
    path = report.annotated_assembly
    if path is None or not path.exists():
        path = report.clean_assembly
    functions = _parse_assembly(path.read_text(errors="replace")) if path is not None and path.exists() else {}
    report.functions = [name for name, function in functions.items() if function.blocks]
    for loops in report.source_loops.values():
        _read_statements(loops, source=report.source)

    remarks: dict[str, list[LoopReport]] = {}
    for loop in report.loops:
        remarks.setdefault(loop.function, []).append(loop)
    statements: list[_Statement] | None = None  # (of the measured file, parsed when a loop needs them)
    out: list[LoopReport] = []
    unattributed: list[SourceLoop] = []
    for name in [*report.functions, *(name for name in remarks if name not in functions)]:
        function = functions.get(name, _AsmFunction(name))
        loops = _asm_loops(function)
        layout = report.layout.get(name, [])
        paired = len(layout) == len(function.blocks) and all(
            info.name == block.name for info, block in zip(layout, function.blocks)
        )
        schedules = {header: layout[loop.index].schedule for header, loop in loops.items()
                     if paired and layout[loop.index].schedule is not None}
        placed, leftover = _place_final_remarks(function, loops, remarks.get(name, []), schedules)
        iterations = _zol_iterations(function, loops)
        trips: dict[int, int] = {}
        for header, count in iterations.items():
            if count is not None:
                remark = placed.get(header)
                peeled = remark.ns - 1 if remark is not None and remark.sched != "none" else 0
                trips[header] = count + peeled
        matched = _match_source_loops(function, loops, report.source_loops.get(name, []), trips=trips)
        if len(matched) < len(loops):
            if statements is None:
                try:
                    statements = _loop_statements(_tokenize(report.source.read_text(errors="replace")))
                except OSError:
                    statements = []
            for header, statement in _statement_fallback(function, loops, matched, statements,
                                                          report.source.name).items():
                fallback = SourceLoop(function=name, path=report.source.name, line=statement.line)
                _locate(fallback, statement)
                fallback.trip = _parse_source_trip(report.source, statement.line)
                matched[header] = fallback
        for header, asm_loop in sorted(loops.items(), key=lambda item: item[1].index):
            block = function.blocks[asm_loop.index]
            loop = placed.get(header) or LoopReport(function=name, block=block.name or f"<{block.ref}>")
            loop.mbb = header
            loop.header = block.ref
            loop.depth = max(asm_loop.depth, 1)
            loop.parent = function.blocks[loops[asm_loop.parent].index].ref if asm_loop.parent in loops else None
            loop.inner = [function.blocks[loops[child].index].ref for child in asm_loop.children]
            loop.num_blocks = len(asm_loop.blocks)
            own = [function.blocks[index] for index in asm_loop.own]
            if paired:
                loop.body_bundles = sum(layout[index].bundles for index in asm_loop.own)
                loop.body_bytes = sum(layout[index].bytes for index in asm_loop.own)
            else:
                loop.body_bundles = sum(len(own_block.bundles) for own_block in own)
            if header not in placed and loop.num_blocks == 1:
                loop.ii = loop.body_bundles  # (as the final remark has it for an unpipelined loop)
            loop.zol = block.label in function.loop_starts
            if header in iterations:
                loop.hw_iters = iterations[header]
                loop.zol = True
            loop.loop_spills = sum(own_block.spills for own_block in own)
            loop.loop_reloads = sum(own_block.reloads for own_block in own)
            loop.spill_bytes = sum(own_block.spill_bytes for own_block in own)
            loop.reload_bytes = sum(own_block.reload_bytes for own_block in own)
            loop.asm_body = [slots for own_block in own for _, slots in own_block.bundles]
            loop.callees = [call for own_block in own for call in own_block.calls]
            loop.loop_calls = [_demangle_short(call) for call in loop.callees]
            loop.branches = any(target != block.label for own_block in own for target in own_block.jumps)
            loop.source_ref = ""
            if header in matched:
                _attribute(loop, matched[header], source=report.source)
                if matched[header].file != report.source.name:
                    chains = [chain for own_block in own for chain in own_block.locs]
                    loop.call_site = _call_site(chains, matched[header], report.source.name)
            out.append(loop)
        out.extend(_remark_only_loop(report, loop) for loop in leftover)
        if function.blocks:
            in_loops = {index for asm_loop in loops.values() for index in asm_loop.blocks}
            outside = [index for index in range(len(function.blocks)) if index not in in_loops]
            report.outside_loops[name] = (
                sum(layout[index].bundles if paired else len(function.blocks[index].bundles) for index in outside),
                [call for index in outside for call in function.blocks[index].calls],
                any(function.blocks[index].jumps for index in outside),
            )
        taken = set(map(id, matched.values()))
        unattributed.extend(loop for loop in report.source_loops.get(name, []) if id(loop) not in taken)
    for name, source_loops in report.source_loops.items():
        if name not in report.functions and name not in remarks:
            unattributed.extend(source_loops)
    report.loops = out
    report.unattributed_loops = []
    for source_loop in unattributed:
        loop = LoopReport(function=source_loop.function)
        _attribute(loop, source_loop, source=report.source)
        report.unattributed_loops.append(loop)


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


# -- Public API ---------------------------------------------------------------


def _resolve_clang_cmd(clang_cmd: list[str] | None) -> list[str]:
    if clang_cmd is not None:
        return list(clang_cmd)
    from artlite.operators.base import AIE_TARGET, CLANG_CMD_COMMON

    return [*CLANG_CMD_COMMON, f"--target={AIE_TARGET}"]


_FOR_HEADER = re.compile(
    r"\bfor\s*\(\s*(?:[\w:]+\s+)*(\w+)\s*=\s*([^;]+?)\s*;"  # [type] i = A;
    r"\s*(\w+)\s*(<=|<|!=)\s*([^;]+?)\s*;"  # i < B;
    r"\s*(?:\+\+\s*(\w+)|(\w+)\s*\+\+|(\w+)\s*\+=\s*([^)]+?))\s*\)"  # ++i | i++ | i += S
)


def _constant(expr: str) -> int | None:
    """Value of a constant integer expression (C semantics)."""
    expr = re.sub(r"(?<=\d)[uUlL]+\b", "", expr)
    if not re.fullmatch(r"[\d\s+\-*/()]+", expr):
        return None
    try:
        value = eval(expr.replace("/", "//"), {"__builtins__": {}}, {})
    except (SyntaxError, ZeroDivisionError, NameError, TypeError):
        return None
    return value if isinstance(value, int) else None


def _template_arguments(source: Path, line_no: int, function: str) -> dict[str, str]:
    """Values of the template parameters of the function (an instance: `name<256, 4>`, demangled) whose
    definition contains line line_no: {"N": "256", ...}; empty when unknown."""
    name = _function_name(_demangle(function)) if function else ""
    if "<" not in name:
        return {}
    arguments = [text.strip() for _, text in _split_top_level(name[name.index("<") + 1 : name.rindex(">")])]
    try:
        tokens = _tokenize(source.read_text(errors="replace"))
    except OSError:
        return {}
    bodies = [body for body in _function_bodies(tokens) if tokens[body[0]].line <= line_no <= tokens[body[1]].line]
    if not bodies:
        return {}
    start = max(bodies)[0]
    index = start  # back to the `template <...>` in front of the declaration
    while index > 0 and tokens[index].text not in {";", "}"} and tokens[index].text != "template":
        index -= 1
    if tokens[index].text != "template" or tokens[index + 1].text != "<":
        return {}
    depth, position, parameters, current = 0, index + 1, [], []
    while position < start:
        text = tokens[position].text
        position += 1
        if text == "<":
            depth += 1
            if depth == 1:
                continue
        elif text == ">":
            depth -= 1
            if depth == 0:
                parameters.append(current)
                break
        elif text == "," and depth == 1:
            parameters.append(current)
            current = []
            continue
        current.append(text)
    names = []
    for parameter in parameters:
        if "=" in parameter:
            parameter = parameter[: parameter.index("=")]
        names.append(parameter[-1] if parameter and re.fullmatch(r"[A-Za-z_]\w*", parameter[-1]) else "")
    return {key: value for key, value in zip(names, arguments) if key and re.fullmatch(r"-?\d+[uUlL]*", value)}


def _split_top_level(text: str) -> list[tuple[int, str]]:
    """Comma-separated parts of a template argument list, outside nested brackets."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append((start, text[start:index]))
            start = index + 1
    parts.append((start, text[start:]))
    return parts


def _parse_source_trip(source: Path, line_no: int, constants: Mapping[str, str] | None = None) -> int | None:
    """Parse a constant trip count from a `for (i = A; i < B; i += S)` header (also `<=`, `!=`, `i++`);
    constants: names to substitute (template parameters of the function instance)."""
    try:
        lines = source.read_text(errors="replace").splitlines()
    except OSError:
        return None
    if not 1 <= line_no <= len(lines):
        return None
    # The for-header may spill across lines; join three and try.
    text = " ".join(lines[line_no - 1 : line_no + 2])
    match = _FOR_HEADER.search(text)
    if not match or match.start() >= len(lines[line_no - 1]):
        return None
    var, init, cond_var, op, bound, pre_inc, post_inc, step_var, step = match.groups()
    if not var == cond_var == (pre_inc or post_inc or step_var):
        return None
    for name, value in (constants or {}).items():
        init, bound = (re.sub(rf"\b{name}\b", f"({value})", text) for text in (init, bound))
        step = None if step is None else re.sub(rf"\b{name}\b", f"({value})", step)
    start, end = _constant(init), _constant(bound)
    stride = 1 if step is None else _constant(step)
    if start is None or end is None or stride is None or stride <= 0:
        return None
    if op == "<=":
        end += 1
    elif op == "!=" and (end - start) % stride:
        return None
    return max(0, -(-(end - start) // stride))


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
    resolved_clang_cmd = _resolve_clang_cmd(clang_cmd)

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
    compilation = _compile_kernel(
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
            _print_report(report)
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
        _enrich_from_assembly(report)
    for loop in report.loops:
        if loop.kernel_line is not None and loop.hw_iters is None:
            loop.source_trip = _parse_source_trip(
                source, loop.kernel_line, _template_arguments(source, loop.kernel_line, loop.function)
            )
    _mark_shared_blocks(report)
    _collect_issues(report)

    if report.returncode == 0:
        _REPORT_CACHE[source_key] = report
    if verbose:
        _print_report(report)
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
            loop.prologue_of = _loop_name(report, other)
            other.epilogue_of = _loop_name(report, loop)


@dataclass(frozen=True)
class _Cycles:
    value: int
    bound: bool = False  # branches: every path counted, an upper bound

    def __add__(self, other: _Cycles) -> _Cycles:
        return _Cycles(self.value + other.value, self.bound or other.bound)


class _CostModel:
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
        terms = [_fmt_int(body)]
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
        return _Cycles(iters * iteration.value, iteration.bound or _exits_early(loop))


def _exits_early(loop: LoopReport) -> bool:
    """The loop may leave before its trip count: the count is a maximum (a ZOL's count is exact)."""
    return loop.hw_iters is None and (loop.early_exit or (loop.branches and loop.num_blocks == 1))


def loop_cycles(report: KernelReport, loop: LoopReport) -> tuple[int, bool] | None:
    """Cycles of one entry into the loop, as its printed cost (prologue + iterations x (II or body, inner loops,
    calls) + epilogue; without the calls when a callee's cost is unknown), and whether that is an upper bound
    (branches, an early exit). None without a trip count."""
    costs = _CostModel(report)
    _, iteration = costs.terms(loop, calls=True)
    if iteration is None:
        _, iteration = costs.terms(loop, calls=False)
    iters = loop.effective_iters()
    if iters is None or iteration is None:
        return None
    total = loop.prologue_cycles + iters * iteration.value + (loop.epilogue_bundles or 0)
    return total, iteration.bound or _exits_early(loop)


def call_cycles(report: KernelReport, function: str) -> tuple[int, bool] | None:
    """Cycles of one call of a function of the kernel (its code outside loops, its loops and its calls) and
    whether that is an upper bound. None when a trip count or a callee is unknown."""
    cycles = _CostModel(report).function(function)
    return None if cycles is None else (cycles.value, cycles.bound)


def _calls_text(loop: LoopReport, costs: _CostModel) -> str:
    """The calls of one iteration with the cycles of each."""
    counts: dict[str, int] = {}
    for callee in loop.callees:
        counts[callee] = counts.get(callee, 0) + 1
    parts = []
    for callee, count in counts.items():
        cycles = costs.function(callee)
        cost = "cost unknown" if cycles is None else f"{'<= ' if cycles.bound else ''}{cycles.value} cy"
        parts.append(f"{_demangle_short(callee)}{f' x{count}' if count > 1 else ''} ({cost}{' each' * (count > 1)})")
    return ", ".join(parts)


def _spill_text(loop: LoopReport) -> str:
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


def _collect_issues(report: KernelReport) -> None:
    costs = _CostModel(report)
    for loop in report.loops:
        if loop.callees:
            report.issues.append(
                f"{loop.name}: {len(loop.callees)} call(s) per iteration ({_calls_text(loop, costs)}); "
                "not in the bundle count"
            )
        if loop.loop_spills or loop.loop_reloads:
            report.issues.append(f"{loop.name}: spill code in the loop body: {_spill_text(loop)}")
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


# -- Issue slots and shared ports --------------------------------------------

# The issue slot and the shared ports of each AIE2P assembly mnemonic: aie2p_isa, generated from llvm-aie's TableGen
# sources (python -m artlite.gen_isa_tables after a toolchain upgrade). Mnemonics of several slots (mov, mov.nb, vmov,
# j, jl) are told apart by their operands in slot_of; nops are not counted. A lng (long) op takes the alu and the mv
# slot of its bundle.
#
# The ports: a loop's II is at least the number of its ops on each, which the per-slot counts do not show (e.g.
# vshuffle and the conversion pushes / flushes of a bfp16 output stream all read L_RM_PORT: a dequant loop of 30
# shuffles and 8 panel pushes runs at II 46 with its mv slot at 30).
SLOTS = tuple(aie2p_isa.SLOT_MNEMONICS)  # lda ldb st vec mv alu lng
_SLOT_OF = {mnemonic: slot for slot, text in aie2p_isa.SLOT_MNEMONICS.items() for mnemonic in text.split()}
PORTS = tuple(aie2p_isa.PORT_UNITS)
_PORTS_OF: dict[str, list[tuple[str, int]]] = {}  # mnemonic -> [(port, cycles)]
for _port, _text in aie2p_isa.PORT_MNEMONICS.items():
    for _word in _text.split():
        _mnemonic, _, _cycles = _word.partition("*")
        _PORTS_OF.setdefault(_mnemonic, []).append((_port, int(_cycles or 1)))


def slot_of(instruction: str) -> str | None:
    """The issue slot of an assembly instruction: one of SLOTS, None for a nop, "?" for an unknown mnemonic."""
    mnemonic, *rest = instruction.split(None, 1)
    operands = rest[0].strip() if rest else ""
    if mnemonic.startswith("nop"):
        return None
    if mnemonic in ("mov", "mov.nb"):
        if ".flt2fx" in operands or ".fx2flt" in operands:
            return "alu"
        if re.match(r"ms\b", operands):
            return "st"
        if mnemonic == "mov.nb" or re.search(r",\s*ss$", operands):
            return "lda"
        return "mv"
    if mnemonic == "vmov":
        if re.match(r"mcd\b", operands):
            return "st"
        return "lda" if re.search(r"\bscd\b", operands) else "mv"
    if mnemonic in ("j", "jl"):
        return "lng" if "#" in operands else "alu"
    return _SLOT_OF.get(mnemonic, "?")


@dataclass
class SlotCounts:
    """The ops of a loop body (one iteration) per issue slot, and per shared port (PORTS)."""

    ops: dict[str, Counter[str]]  # slot -> mnemonic -> count ("?": unknown mnemonics)
    bundles: int = 0
    port_ops: dict[str, Counter[str]] = field(default_factory=dict)  # port -> mnemonic -> cycles

    def count(self, slot: str) -> int:
        if slot in PORTS:
            return sum(self.port_ops.get(slot, Counter()).values())
        return sum(self.ops.get(slot, Counter()).values())

    @property
    def pressure(self) -> dict[str, int]:
        """Bundles each slot and port needs per iteration: a lng op occupies the alu and the mv slot."""
        lng = self.count("lng")
        slots = {slot: self.count(slot) + (lng if slot in ("alu", "mv") else 0) for slot in SLOTS if slot != "lng"}
        return slots | {port: self.count(port) for port in PORTS}

    @property
    def bound(self) -> int:
        """The resource bound of the II: the busiest slot or port."""
        return max(self.pressure.values(), default=0)

    @property
    def binding(self) -> list[str]:
        bound = self.bound
        return [slot for slot, count in self.pressure.items() if bound and count == bound]

    @property
    def busiest_port(self) -> tuple[str, int] | None:
        counts = [(self.count(port), port) for port in PORTS if self.count(port)]
        return (max(counts)[1], max(counts)[0]) if counts else None

    def describe(self, slot: str) -> str:
        """The slot's (or port's) ops, most frequent first: 'vshuffle 16, vband 6'."""
        ops = self.port_ops.get(slot, Counter()) if slot in PORTS else self.ops.get(slot, Counter())
        if slot in ("alu", "mv"):
            ops = ops + self.ops.get("lng", Counter())
        return ", ".join(f"{mnemonic} {count}" for mnemonic, count in ops.most_common())


def count_slots(bundles: Sequence[Sequence[str]]) -> SlotCounts:
    """Per-slot and per-port op counts of assembly bundles (each a list of instructions)."""
    counts = SlotCounts(ops={}, bundles=len(bundles))
    for bundle in bundles:
        for instruction in bundle:
            slot = slot_of(instruction)
            if slot is not None:
                mnemonic = instruction.split(None, 1)[0]
                counts.ops.setdefault(slot, Counter())[mnemonic] += 1
                for port, cycles in _PORTS_OF.get(mnemonic, ()):
                    counts.port_ops.setdefault(port, Counter())[mnemonic] += cycles
    return counts


def _resources_text(loop: LoopReport) -> str:
    """The resource bound of a loop body's II (the busiest slot or shared port) with its ops, and the busy slots."""
    slots = count_slots(loop.asm_body)
    if not slots.bound:
        return ""
    binding = (  # (the ops of one or two binding resources; a small bound ties many)
        "; ".join(f"{name}: {slots.describe(name)}" for name in slots.binding)
        if len(slots.binding) <= 2
        else ", ".join(slots.binding)
    )
    busy = ", ".join(f"{slot} {count}" for slot, count in slots.pressure.items() if slot in SLOTS and count)
    unknown = f"; unknown mnemonics: {', '.join(slots.ops['?'])}" if slots.ops.get("?") else ""
    return f"res bound {slots.bound} ({binding}); slots {busy}{unknown}"


# -- Printing -----------------------------------------------------------------


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _loop_where(report: KernelReport, loop: LoopReport) -> str:
    """ @file:line of the loop statement (an inlined helper's loop of another file: and its call site), @?."""
    if loop.kernel_line is not None:
        return f" @{report.source.name}:{loop.kernel_line}"
    if loop.source_ref:
        return f" @{loop.source_ref}" + (f" via {loop.call_site}" if loop.call_site else "")
    return " @?"


def _loop_name(report: KernelReport, loop: LoopReport) -> str:
    return f"{_block_label(loop.block)}{_loop_where(report, loop)}"


def _prologue_text(loop: LoopReport) -> str:
    """The prologue term of a cost: (N) when it counts as another loop's epilogue."""
    if loop.prologue_bundles is None:
        return "-"
    return f"({loop.prologue_bundles})" if loop.prologue_of else str(loop.prologue_bundles)


def _cost_text(report: KernelReport, loop: LoopReport, costs: _CostModel) -> str:
    """Cycles per entry into the loop: P + iters*(iteration) + E; a branch loop: iters*(iteration).

    An iteration: the II (a branch loop: its own bundles on all paths), its inner
    loops and its calls (excluded when a callee's cost is unknown)."""
    tags = []
    terms, iteration = costs.terms(loop, calls=True)
    if costs.has_calls(loop):
        bare_terms, bare = costs.terms(loop, calls=False)
        if iteration is None and bare is not None:
            terms, iteration = bare_terms, bare
            tags.append("excluding calls")
        elif iteration is not None:
            tags.append("with calls")
    if loop.branches and loop.num_blocks > 1:
        tags.append("all paths")
    if _exits_early(loop):
        tags.append("may exit early")
    if loop.prologue_of:
        tags.append(f"prologue shared: counted in {loop.prologue_of}")
    if loop.epilogue_of:
        tags.append(f"epilogue shared: the prologue of {loop.epilogue_of}")
    per_iteration = terms[0] if len(terms) == 1 else f"({' + '.join(terms)})"
    bound = "<= " if (iteration is not None and iteration.bound) or _exits_early(loop) else ""

    def formula(count: str) -> str:
        if loop.ii is None:
            return f"{count}*{per_iteration}"
        return f"{_prologue_text(loop)} + {count}*{per_iteration} + {_fmt_int(loop.epilogue_bundles)}"

    def total(count: int, value: int) -> int:
        return loop.prologue_cycles + count * value + (loop.epilogue_bundles or 0)

    iters = loop.effective_iters()
    if iters is not None and iteration is not None:
        tags.insert(0, f"iters from {'ZOL' if loop.hw_iters is not None else 'src'}")
        return f"cost {bound}{formula(str(iters))} = {total(iters, iteration.value)} cy [{'; '.join(tags)}]"
    text = f"cost(iters) {bound or '= '}{formula('iters')}"
    if iters is None and iteration is not None and report.trip_count:
        text += f"; @{report.trip_count} iters = {total(report.trip_count, iteration.value)} cy"
    return text + (f" [{'; '.join(tags)}]" if tags else "")


def _tree_order(loops: list[LoopReport]) -> list[LoopReport]:
    """Loops as a depth-first walk of the nesting, siblings by source line."""
    headers = {loop.header for loop in loops if loop.header}
    children: dict[str | None, list[LoopReport]] = {}
    for loop in loops:
        children.setdefault(loop.parent if loop.parent in headers else None, []).append(loop)
    for siblings in children.values():
        siblings.sort(key=lambda item: (item.kernel_line is None, item.kernel_line or 0, item.source_ref, item.mbb or 0))
    order: list[LoopReport] = []
    seen: set[int] = set()

    def visit(parent: str | None) -> None:
        for loop in children.get(parent, []):
            if id(loop) not in seen:
                seen.add(id(loop))
                order.append(loop)
                if loop.header:
                    visit(loop.header)

    visit(None)
    order.extend(loop for loop in loops if id(loop) not in seen)
    return order


def _print_loop(report: KernelReport, loop: LoopReport, costs: _CostModel) -> None:
    pad = "    " + "  " * (max(loop.depth, 1) - 1)
    where = _loop_where(report, loop)
    label = _block_label(loop.block)
    if label != loop.block and loop.header:
        label += f" ({loop.header})"  # (the shortened name does not find the block in the assembly)
    if loop.sched == "none":
        reasons = "; ".join(loop.prereject_reasons)
        if loop.ii is None and loop.num_blocks > 1:
            nested = costs.nested(loop)
            detail = f"not pipelined, {loop.num_blocks - sum(child.num_blocks for child in nested)} blocks"
            if nested:
                detail += f" + {len(nested)} inner loop{'s' if len(nested) > 1 else ''}"
            detail += f", body {_fmt_int(loop.body_bundles)} bundles"
        else:
            detail = f"not pipelined, body {_fmt_int(loop.body_bundles or loop.ii)} bundles"
        if loop.zol is True:
            detail += ", ZOL"
            if loop.hw_iters is not None:
                detail += f", {loop.hw_iters} iters"
        elif loop.zol is False:
            detail += ", no ZOL"
        if reasons:
            detail += f" ({reasons})"
        print(f"{pad}{label}{where}: {detail}")
        print(f"{pad}  {_cost_text(report, loop, costs)}")
        _print_resources(loop, pad=pad)
        _print_loop_warnings(loop, costs, pad=pad)
        return
    zol = {True: "ZOL", False: "no ZOL", None: "ZOL?"}[loop.zol]
    ns_text = str(loop.ns) if loop.ns > 1 else "1 (no overlap)"
    head = (
        f"{pad}{label}{where}: {loop.sched} pipelined, "
        f"II={loop.ii}, NS={ns_text}, {zol}"
    )
    if loop.hw_iters is not None:
        head += f", {loop.hw_iters} iters"
        if loop.trip_count is not None and loop.trip_count != loop.hw_iters:
            head += f" (trip {loop.trip_count}: {loop.ns - 1} peeled)"
    elif loop.source_trip is not None:
        head += f", trip {loop.source_trip} (from source)"
    print(head)
    body = f"body {_fmt_int(loop.body_bundles)} bundles"
    if loop.body_bytes is not None:
        body += f" ({loop.body_bytes} B)"
    if (
        loop.body_bundles is not None
        and loop.ii is not None
        and loop.body_bundles != loop.ii
    ):
        body += f" != II={loop.ii}!"
    print(f"{pad}  {_cost_text(report, loop, costs)}; {body}")
    _print_resources(loop, pad=pad)
    overhead = loop.prologue_cycles + (loop.epilogue_bundles or 0)
    iters = loop.effective_iters()
    if iters is not None and loop.ii is not None:
        steady = iters * loop.ii
        if overhead > max(steady, 8):
            print(
                f"{pad}  note: ramp+drain ({overhead} cy) exceeds the steady-state "
                f"loop time ({steady} cy) -- the pipeline barely pays off at this trip count"
            )
    if loop.search_ii is not None and loop.search_ii != loop.ii:
        print(
            f"{pad}  note: compiler pre-RA search found II={loop.search_ii}, "
            "but the generated loop has the II above"
        )
    if loop.search_mii_note:
        print(f"{pad}  note: {loop.search_mii_note}")
    _print_loop_warnings(loop, costs, pad=pad)


def _print_resources(loop: LoopReport, *, pad: str) -> None:
    """A one-block loop's resource bound: what its II cannot go below (calls and branches aside)."""
    text = _resources_text(loop) if loop.num_blocks == 1 else ""
    if text:
        print(f"{pad}  {text}")


def _print_loop_warnings(loop: LoopReport, costs: _CostModel, *, pad: str) -> None:
    if loop.loop_spills or loop.loop_reloads:
        print(f"{pad}  ⚠ spill code in the loop body: {_spill_text(loop)}")
    if loop.callees:
        print(f"{pad}  ⚠ calls per iteration (not in the bundle count): {_calls_text(loop, costs)}")


def _print_report(report: KernelReport) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Source:    {report.source.absolute()}")
    if report.clean_assembly:
        print(f"  Assembly:  {report.clean_assembly.absolute()}")
    if report.yaml_path:
        print(f"  Remarks:   {report.yaml_path.absolute()}")

    if report.parse_error:
        print(f"  ✗ {report.parse_error}")

    loops_by_function: dict[str, list[LoopReport]] = {}
    for loop in report.loops:
        loops_by_function.setdefault(loop.function, []).append(loop)
    unattributed: dict[str, list[LoopReport]] = {}
    for loop in report.unattributed_loops:
        unattributed.setdefault(loop.function, []).append(loop)
    # Functions in assembly order; then those known only from remarks (e.g. straight-line).
    functions = list(report.functions)
    for function in [*loops_by_function, *unattributed, *report.layout, *(key.split("::")[0] for key in report.blocks)]:
        if function not in functions:
            functions.append(function)
    names = {function: _demangle_short(function) for function in functions}
    for function in functions:
        if list(names.values()).count(names[function]) > 1:  # overloads: with their parameters
            names[function] = _demangle(function)
    costs = _CostModel(report)

    for function in functions:
        stack = report.stack_bytes.get(function)
        stack_text = f", stack {_fmt_int(stack)} B" if stack is not None else ""
        totals = report.layout.get(function) or [
            block for key, block in report.blocks.items() if key.split("::")[0] == function
        ]
        code_text = ""
        if totals:
            code_text = (
                f", code {sum(b.bundles for b in totals)} bundles "
                f"/ {sum(b.bytes for b in totals)} B"
            )
        print(f"\n  {names[function]}{stack_text}{code_text}")
        loops = loops_by_function.get(function, [])
        lost = unattributed.get(function, [])
        if not loops and not lost:
            print("    (no loops; straight-line code)")
            continue
        for loop in _tree_order(loops):
            _print_loop(report, loop, costs)
        # Pre-RA loops that no loop of the final code could be tied to.
        for loop in sorted(lost, key=lambda item: (item.kernel_line is None, item.kernel_line or 0, item.source_ref)):
            found = "; ".join(loop.prereject_reasons)
            if loop.search_ii is not None:
                found = "; ".join(filter(None, [found, f"search found II={loop.search_ii}"]))
            print(
                f"    loop{_loop_where(report, loop)}: unattributed (matches no loop of the final code)"
                + (f"; pre-RA: {found}" if found else "")
            )

    seen_issues: set[str] = set()
    for issue in report.issues:
        if issue not in seen_issues:
            seen_issues.add(issue)
            print(f"  ⚠ {issue}")
    print(f"{'=' * 60}")
