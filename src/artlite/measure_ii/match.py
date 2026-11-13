"""Matching the pre-RA source loops with the final code's loops, and the loops' assembly facts."""

from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

from ..cxx_scan import Statement, loop_statements, tokenize
from .asm import AsmFunction, AsmLoop, asm_loops, parse_assembly, zol_iterations
from .demangle import demangle_short
from .remarks import GENERIC_REJECTS
from .report import KernelReport, LoopReport, SourceLoop
from .statements import locate, read_statements
from .trips import parse_source_trip


def _place_final_remarks(
    function: AsmFunction,
    loops: dict[int, AsmLoop],
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
    function: AsmFunction,
    loops: dict[int, AsmLoop],
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
    function: AsmFunction,
    loops: dict[int, AsmLoop],
    matched: dict[int, SourceLoop],
    statements: list[Statement],
    file: str,
) -> dict[int, Statement]:
    """Loop statements for the final loops no pre-RA remark names (e.g. an outer loop the pre-RA pipeliner never
    looked at): the innermost unmatched statement of the measured file around all the loop's own lines, outside
    the statements of the loops matched inside it."""
    taken = {(loop.line, loop.column) for loop in matched.values() if loop.file == file}
    out: dict[int, Statement] = {}
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


def _descendants(loops: dict[int, AsmLoop], header: int) -> set[int]:
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
    loop.prereject_reasons = [reason for reason in source_loop.reasons if reason not in GENERIC_REJECTS]
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


def enrich_from_assembly(report: KernelReport) -> None:
    """Replace the remark loops by the final code's loop tree and add what the assembly shows."""
    path = report.annotated_assembly
    if path is None or not path.exists():
        path = report.clean_assembly
    functions = parse_assembly(path.read_text(errors="replace")) if path is not None and path.exists() else {}
    report.functions = [name for name, function in functions.items() if function.blocks]
    for loops in report.source_loops.values():
        read_statements(loops, source=report.source)

    remarks: dict[str, list[LoopReport]] = {}
    for loop in report.loops:
        remarks.setdefault(loop.function, []).append(loop)
    statements: list[Statement] | None = None  # (of the measured file, parsed when a loop needs them)
    out: list[LoopReport] = []
    unattributed: list[SourceLoop] = []
    for name in [*report.functions, *(name for name in remarks if name not in functions)]:
        function = functions.get(name, AsmFunction(name))
        loops = asm_loops(function)
        layout = report.layout.get(name, [])
        paired = len(layout) == len(function.blocks) and all(
            info.name == block.name for info, block in zip(layout, function.blocks)
        )
        schedules = {header: layout[loop.index].schedule for header, loop in loops.items()
                     if paired and layout[loop.index].schedule is not None}
        placed, leftover = _place_final_remarks(function, loops, remarks.get(name, []), schedules)
        iterations = zol_iterations(function, loops)
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
                    statements = loop_statements(tokenize(report.source.read_text(errors="replace")))
                except OSError:
                    statements = []
            for header, statement in _statement_fallback(function, loops, matched, statements,
                                                          report.source.name).items():
                fallback = SourceLoop(function=name, path=report.source.name, line=statement.line)
                locate(fallback, statement)
                fallback.trip = parse_source_trip(report.source, statement.line)
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
            loop.loop_calls = [demangle_short(call) for call in loop.callees]
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
