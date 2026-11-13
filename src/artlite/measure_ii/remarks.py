"""The optimization record's YAML: the remarks of the final loops and of the pre-RA pipeliner."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from .demangle import demangle_short
from .report import BlockInfo, HardwareLoop, KernelReport, LoopReport, SourceLoop


class LLVMLoader(yaml.SafeLoader):
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
LLVMLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:int"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
LLVMLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int", re.compile(r"^[-+]?(?:0|[1-9][0-9]*)$"), list("-+0123456789")
)
LLVMLoader.add_multi_constructor("!", _construct_tagged)


def _sanitise_yaml(text: str) -> str:
    """Normalize tabs used as YAML indentation without changing quoted values."""
    return re.sub(
        r"^[ \t]+",
        lambda match: match.group(0).replace("\t", "    "),
        text,
        flags=re.MULTILINE,
    )


def normalise_block(block: str) -> str:
    return re.sub(r"^bb\.\d+\.", "", block.strip())


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
    block = normalise_block(str(values[0])) if values else ""
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
        documents = yaml.load_all(text, Loader=LLVMLoader)
        return [doc for doc in documents if isinstance(doc, dict)]
    except yaml.YAMLError:
        fallback = text.replace("\t", "    ")
        documents = yaml.load_all(fallback, Loader=LLVMLoader)
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


GENERIC_REJECTS = {"Failed to pipeline loop"}


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
        block="" if number and number.end() == len(block) else normalise_block(block) or "<unknown>",
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
        loop.search_block = normalise_block(_first_text(args, "BasicBlock"))
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
    assembly turns them into the final code's loop tree (``enrich_from_assembly``)."""
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
                        block=normalise_block(str(block)),
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
                f"{demangle_short(function)} (function{where}): the register allocator spilled "
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
