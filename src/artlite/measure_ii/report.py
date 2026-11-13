"""The report's data: loops, blocks, source loops, the kernel report, and how a loop is named and ordered."""

from dataclasses import dataclass, field
from pathlib import Path

from .demangle import block_label, demangle_short


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
        return f"{demangle_short(self.function)}::{block_label(self.block)}"

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
        prologue = prologue_text(self)
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


def fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def loop_where(report: KernelReport, loop: LoopReport) -> str:
    """ @file:line of the loop statement (an inlined helper's loop of another file: and its call site), @?."""
    if loop.kernel_line is not None:
        return f" @{report.source.name}:{loop.kernel_line}"
    if loop.source_ref:
        return f" @{loop.source_ref}" + (f" via {loop.call_site}" if loop.call_site else "")
    return " @?"


def loop_name(report: KernelReport, loop: LoopReport) -> str:
    return f"{block_label(loop.block)}{loop_where(report, loop)}"


def prologue_text(loop: LoopReport) -> str:
    """The prologue term of a cost: (N) when it counts as another loop's epilogue."""
    if loop.prologue_bundles is None:
        return "-"
    return f"({loop.prologue_bundles})" if loop.prologue_of else str(loop.prologue_bundles)


def tree_order(loops: list[LoopReport]) -> list[LoopReport]:
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
