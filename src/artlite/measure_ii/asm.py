"""The assembly: functions, blocks and their loop comments, the final code's loops, zero-overhead loop trip counts."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .remarks import normalise_block

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
class AsmFunction:
    name: str
    blocks: list[_AsmBlock] = field(default_factory=list)
    loop_starts: set[str] = field(default_factory=set)  # labels written to ls: the ZOL headers


@dataclass
class AsmLoop:
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
        block.name = normalise_block(name.group(1))


def inlining_chain(file: str, line: int, comment: str) -> tuple[tuple[str, int], ...]:
    """A .loc's location and the call sites it is inlined at, innermost first, from the asm printer's comment
    (``a.hpp:1098:29 @[ kernel_lib.h:41:27 @[ kernel.cc:315:9 ] ]``); call sites of line 0 are dropped."""
    frames = [(Path(path).name, int(number)) for path, number in _ASM_FRAME.findall(comment)]
    if not frames or frames[0] != (file, line):
        return ((file, line),)
    return tuple(frame for frame in frames if frame[1])


def parse_assembly(text: str) -> dict[str, AsmFunction]:
    """Blocks of each function: bundles, source lines, loop structure, spills and calls."""
    files: dict[str, str] = {}
    functions: dict[str, AsmFunction] = {}
    function: AsmFunction | None = None
    block: _AsmBlock | None = None
    location: tuple[tuple[str, int], ...] | None = None
    labels: list[str] = []
    for raw in text.splitlines():
        start = _FUNCTION_START.match(raw)
        if start:
            function = functions.setdefault(start.group(1), AsmFunction(start.group(1)))
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
            block = _AsmBlock(number, normalise_block(block_start.group(4) or ""), block_start.group(1))
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
            location = inlining_chain(files.get(loc.group(1), ""), line, comment) if line else None
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


def asm_loops(function: AsmFunction) -> dict[int, AsmLoop]:
    """The loop tree of a function, by header block number."""
    loops = {
        block.number: AsmLoop(block.number, index, block.depth, block.parent)
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


def _block_entries(function: AsmFunction, loops: dict[int, AsmLoop]) -> list[dict[str, int | None]]:
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


def zol_iterations(function: AsmFunction, loops: dict[int, AsmLoop]) -> dict[int, int | None]:
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
