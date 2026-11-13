"""The printed report."""

from .costs import CostModel, exits_early
from .demangle import block_label, demangle, demangle_short
from .issues import calls_text, spill_text
from .report import KernelReport, LoopReport, fmt_int, loop_where, prologue_text, tree_order
from .slots import SLOTS, count_slots


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


def _cost_text(report: KernelReport, loop: LoopReport, costs: CostModel) -> str:
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
    if exits_early(loop):
        tags.append("may exit early")
    if loop.prologue_of:
        tags.append(f"prologue shared: counted in {loop.prologue_of}")
    if loop.epilogue_of:
        tags.append(f"epilogue shared: the prologue of {loop.epilogue_of}")
    per_iteration = terms[0] if len(terms) == 1 else f"({' + '.join(terms)})"
    bound = "<= " if (iteration is not None and iteration.bound) or exits_early(loop) else ""

    def formula(count: str) -> str:
        if loop.ii is None:
            return f"{count}*{per_iteration}"
        return f"{prologue_text(loop)} + {count}*{per_iteration} + {fmt_int(loop.epilogue_bundles)}"

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


def _print_loop(report: KernelReport, loop: LoopReport, costs: CostModel) -> None:
    pad = "    " + "  " * (max(loop.depth, 1) - 1)
    where = loop_where(report, loop)
    label = block_label(loop.block)
    if label != loop.block and loop.header:
        label += f" ({loop.header})"  # (the shortened name does not find the block in the assembly)
    if loop.sched == "none":
        reasons = "; ".join(loop.prereject_reasons)
        if loop.ii is None and loop.num_blocks > 1:
            nested = costs.nested(loop)
            detail = f"not pipelined, {loop.num_blocks - sum(child.num_blocks for child in nested)} blocks"
            if nested:
                detail += f" + {len(nested)} inner loop{'s' if len(nested) > 1 else ''}"
            detail += f", body {fmt_int(loop.body_bundles)} bundles"
        else:
            detail = f"not pipelined, body {fmt_int(loop.body_bundles or loop.ii)} bundles"
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
    body = f"body {fmt_int(loop.body_bundles)} bundles"
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


def _print_loop_warnings(loop: LoopReport, costs: CostModel, *, pad: str) -> None:
    if loop.loop_spills or loop.loop_reloads:
        print(f"{pad}  ⚠ spill code in the loop body: {spill_text(loop)}")
    if loop.callees:
        print(f"{pad}  ⚠ calls per iteration (not in the bundle count): {calls_text(loop, costs)}")


def print_report(report: KernelReport) -> None:
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
    names = {function: demangle_short(function) for function in functions}
    for function in functions:
        if list(names.values()).count(names[function]) > 1:  # overloads: with their parameters
            names[function] = demangle(function)
    costs = CostModel(report)

    for function in functions:
        stack = report.stack_bytes.get(function)
        stack_text = f", stack {fmt_int(stack)} B" if stack is not None else ""
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
        for loop in tree_order(loops):
            _print_loop(report, loop, costs)
        # Pre-RA loops that no loop of the final code could be tied to.
        for loop in sorted(lost, key=lambda item: (item.kernel_line is None, item.kernel_line or 0, item.source_ref)):
            found = "; ".join(loop.prereject_reasons)
            if loop.search_ii is not None:
                found = "; ".join(filter(None, [found, f"search found II={loop.search_ii}"]))
            print(
                f"    loop{loop_where(report, loop)}: unattributed (matches no loop of the final code)"
                + (f"; pre-RA: {found}" if found else "")
            )

    seen_issues: set[str] = set()
    for issue in report.issues:
        if issue not in seen_issues:
            seen_issues.add(issue)
            print(f"  ⚠ {issue}")
    print(f"{'=' * 60}")
