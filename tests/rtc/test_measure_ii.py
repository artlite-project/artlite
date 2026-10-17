"""artlite.measure_ii: names, loops and their source lines, costs (compiles with Peano, no NPU)."""

import contextlib
import io
import re
from pathlib import Path

import pytest
import yaml

from artlite.measure_ii import (
    KernelReport,
    LoopReport,
    SourceLoop,
    _asm_loops,
    _CostModel,
    _demangle_short,
    _function_name,
    _inlining_chain,
    _LLVMLoader,
    _loop_statements,
    _parse_assembly,
    _parse_source_trip,
    _read_statements,
    _template_arguments,
    _tokenize,
    _zol_iterations,
    loop_cycles,
    measure,
    parse_yaml_remarks,
)

KERNEL = """#include "operators.h"

#define WS(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

__attribute__((noinline)) void sink(float *__restrict y)
{
    aie::store_v(y, aie::zeros<float, 16>());
}

// two instantiations: a loop over h around a loop over b
template <int N>
__attribute__((noinline)) void scale(const float *__restrict x, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < 2; h++) {  // loop h
        AIE_LOOP_NO_UNROLL
        for (int b = 0; b < N; b++)  // loop b
            aie::store_v(y + 16 * b, aie::add(aie::load_v<16>(x + 16 * b), aie::load_v<16>(y + 16 * b)));
        x += 16 * N;
    }
}

// a branch around a call: a loop of several blocks
__attribute__((noinline)) void masked(const int *__restrict m, float *__restrict y, int n)
{
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < n; i++) {  // loop i
        if (m[i])
            sink(y + 16 * i);
    }
}

// an early exit around a call: the header's trip count is a maximum
__attribute__((noinline)) void until(const int *__restrict m, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < 6; i++) {  // loop u
        if (m[i] < 0)
            break;
        sink(y + 16 * i);
    }
}

// the outer loop's remark has the location of the inner loop that starts its body
__attribute__((noinline)) void inv(int32_t *__restrict ws, float *__restrict out, int hilo)
{
    const float *__restrict s = WS(float, 27200);
    float *__restrict tmp = WS(float, 26176);  // (free between phases)
    AIE_LOOP_NO_UNROLL
    for (int h = 0; h < 2; h++) {
        aie::accum<accfloat, 16> ms(al::mulff<16>(aie::load_v<16>(s + 16 * h), aie::broadcast<float, 16>(0.000390625f)));
        ms = aie::add(ms, aie::accum<accfloat, 16>(aie::broadcast<float, 16>(9.999999974752427e-07f)));
        aie::store_v(tmp + 16 * h, aie::invsqrt(ms.template to_vector<float>()));
    }
    AIE_LOOP_NO_UNROLL
    for (int sg = 0; sg < 4; sg++) {  // loop sg
        aie::vector<float, 64> r;
        for (int j = 0; j < 8; j++)  // loop j
            r.insert(j, aie::broadcast<float, 8>(tmp[8 * sg + j]));
        if (hilo) {
            bfloat16 *__restrict o = (bfloat16 *)out;
            const aie::vector<bfloat16, 64> h = aie::accum<accfloat, 64>(r).template to_vector<bfloat16>();
            aie::store_v(o + 64 * sg, h);
            aie::store_v(o + 256 + 64 * sg, aie::accum<accfloat, 64>(aie::sub(r, aie::accum<accfloat, 64>(h).template to_vector<float>())).template to_vector<bfloat16>());
        } else {
            aie::store_v(out + 64 * sg, r);
        }
    }
}

// a loop bounded by a template parameter around a hardware loop (no ZOL: the trip count from the source)
template <int N>
__attribute__((noinline)) void tiles(const float *__restrict x, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int c = 0; c < N; c += 2) {  // loop c
        AIE_LOOP_NO_UNROLL
        for (int b = 0; b < 4; b++)
            aie::store_v(y + 16 * b, aie::load_v<16>(x + 16 * b));
        x += 64;
        y += 64;
    }
}

// a loop of an inlined helper inside its caller's loop (the helper's statement is outside the caller's)
[[clang::always_inline]] inline void rows(const float *__restrict x, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 4; t++) {  // loop t
        aie::store_v(y, aie::add(aie::load_v<16>(x), aie::load_v<16>(y)));
        x += 16;
        y += 16;
    }
}

__attribute__((noinline)) void caller(const float *__restrict x, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int p = 0; p < 3; p++)  // loop p
        rows(x + 64 * p, y + 64 * p);
}

// kernel_lib.h's loops inlined into an if-chain: their own code comes from deeper headers (aie_api), the statement
// and the call site from the inlining chains of the .loc directives
__attribute__((noinline)) void init(float *__restrict y, const float *__restrict x, int k)
{
    if (k == 0) {
        al::copy<512, 16>(y, x);  // copy 0
    } else if (k == 1) {
        al::copy<192, 16>(y + 512, x);  // copy 1
    } else {
        al::copy<64, 16>(y + 1024, x);  // copy 2
    }
}

// consecutive loops: the block between them is the first one's epilogue and the second one's prologue
__attribute__((noinline)) void two(const float *__restrict x, float *__restrict y, float *__restrict z)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int b = 0; b < 16; b++)  // loop first
        aie::store_v(y + 16 * b, aie::add(aie::load_v<16>(x + 16 * b), aie::load_v<16>(y + 16 * b)));
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int b = 0; b < 16; b++)  // loop second
        aie::store_v(z + 16 * b, aie::sub(aie::load_v<16>(x + 16 * b), aie::load_v<16>(z + 16 * b)));
}

}  // namespace

extern "C" {

void straight(float *__restrict y)
{
    aie::store_v(y, aie::zeros<float, 16>());
}

void entry(const float *__restrict x, float *__restrict y, const int *__restrict m, int n, int32_t *__restrict ws)
{
    scale<4>(x, y);
    scale<8>(x, y);
    masked(m, y, n);
    until(m, y);
    inv(ws, y, n);
    tiles<6>(x, y);
    caller(x, y);
    init(y, x, n);
    two(x, y, y + 256);
    AIE_LOOP_NO_UNROLL
    for (int k = 0; k < 32; k += 2)  // loop k
        aie::store_v(y + 16 * k, aie::load_v<16>(x + 16 * k));
}
}
"""


def _line(marker: str) -> int:
    return next(number for number, text in enumerate(KERNEL.splitlines(), 1) if text.endswith(f"// {marker}"))


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[KernelReport, str]:
    """The report of KERNEL and its printout."""
    source = tmp_path_factory.mktemp("measure_ii") / "kernel.cc"
    source.write_text(KERNEL)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        report = measure(source, verbose=True)
    assert report.returncode == 0, report.diagnostics
    return report, printed.getvalue()


def _loops(report: KernelReport, name: str) -> list[LoopReport]:
    return [loop for loop in report.loops if _demangle_short(loop.function) == name]


def _printed_loop(printed: str, head: str) -> list[str]:
    """The printed lines of the loop whose first line contains head: it and its detail lines."""
    lines = printed.splitlines()
    start = next(index for index, line in enumerate(lines) if head in line)
    indent = len(lines[start]) - len(lines[start].lstrip())
    rest = []
    for line in lines[start + 1 :]:
        if len(line) - len(line.lstrip()) <= indent or not line.strip():
            break
        rest.append(line.strip())
    return [lines[start].strip(), *rest]


def test_function_names(measured: tuple[KernelReport, str]) -> None:
    """Anonymous-namespace functions are demangled; template arguments tell instantiations apart."""
    _, printed = measured
    lines = printed.splitlines()
    headers = {line.split(", stack")[0].strip(): index for index, line in enumerate(lines) if ", stack" in line}
    assert set(headers) == {"sink", "scale<4>", "scale<8>", "masked", "until", "inv", "tiles<6>", "caller", "init",
                            "two", "entry", "straight"}, headers
    for name in ("sink", "straight"):
        assert lines[headers[name] + 1] == "    (no loops; straight-line code)"


def test_nested_loops(measured: tuple[KernelReport, str]) -> None:
    """The outer loop of several blocks is listed, the inner loop has its own line and trip count; the
    outer loop's cost adds the inner loop's iterations (its prologue and epilogue are the outer loop's own)."""
    report, _ = measured
    costs = _CostModel(report)
    for n in (4, 8):
        loops = {loop.kernel_line: loop for loop in _loops(report, f"scale<{n}>")}
        assert set(loops) == {_line("loop h"), _line("loop b")}, loops
        outer, inner = loops[_line("loop h")], loops[_line("loop b")]
        assert outer.num_blocks > 1 and outer.sched == "none" and outer.depth == 1
        assert inner.depth == 2 and inner.parent == outer.header and inner.header in outer.inner
        assert inner.trip_count == n and inner.ii is not None and inner.hw_iters is not None
        cycles = costs.steady(outer, calls=True)
        assert cycles is not None and cycles.value == 2 * (outer.body_bundles + inner.hw_iters * inner.ii)


def test_multi_block_loop(measured: tuple[KernelReport, str]) -> None:
    """A loop with a branch has no final scheduling remark: it is found in the assembly. Its call is
    listed under it, with the callee's cycles."""
    report, printed = measured
    (loop,) = _loops(report, "masked")
    assert loop.kernel_line == _line("loop i")
    assert loop.num_blocks > 1 and loop.sched == "none" and loop.zol is False and loop.branches
    assert loop.loop_calls == ["sink"]
    lines = _printed_loop(printed, f"@kernel.cc:{_line('loop i')}:")
    assert re.fullmatch(r"⚠ calls per iteration \(not in the bundle count\): sink \(\d+ cy\)", lines[-1]), lines


def test_early_exit_and_calls(measured: tuple[KernelReport, str]) -> None:
    """A break makes the for-header's trip count a maximum; the cost includes the calls."""
    report, printed = measured
    (loop,) = _loops(report, "until")
    assert loop.kernel_line == _line("loop u") and loop.early_exit and loop.source_trip == 6
    _, cost, calls = _printed_loop(printed, f"@kernel.cc:{_line('loop u')}:")
    assert cost.startswith("cost <= 6*(") and "with calls" in cost and "may exit early" in cost, cost
    assert calls.startswith("⚠ calls per iteration") and "sink (" in calls


def test_merged_location(measured: tuple[KernelReport, str]) -> None:
    """Both loops of inv report the inner loop's location: the outer one gets its own statement."""
    report, printed = measured
    loops = {loop.kernel_line: loop for loop in _loops(report, "inv")}
    outer, inner = loops[_line("loop sg")], loops[_line("loop j")]
    assert outer.source_trip == 4 and inner.source_trip == 8
    assert inner.parent == outer.header and outer.num_blocks > inner.num_blocks > 1
    assert "unattributed" not in printed and "@?" not in printed


def test_plain_loop(measured: tuple[KernelReport, str]) -> None:
    """A loop of the entry function; a one-block loop's resource bound (the busiest slot or shared port) is
    printed under its cost."""
    report, printed = measured
    (loop,) = _loops(report, "entry")
    assert loop.kernel_line == _line("loop k") and loop.trip_count == 16
    assert f"@kernel.cc:{_line('loop k')}: " in printed
    assert not report.unattributed_loops and "@?" not in printed
    assert all(loop.kernel_line is not None or loop.call_site for loop in report.loops)
    lines = _printed_loop(printed, f"@kernel.cc:{_line('loop k')}: ")
    assert re.fullmatch(r"res bound \d+ \(\w+: .+\); slots .+", lines[2]), lines


def test_inlined_header_loops(measured: tuple[KernelReport, str]) -> None:
    """A header's helper loops inlined into an if-chain: each is the helper's statement, called at its line."""
    report, printed = measured
    loops = _loops(report, "init")
    assert len(loops) == 3, loops
    assert {loop.source_ref for loop in loops} == {loops[0].source_ref} and loops[0].source_ref.startswith(
        "kernel_lib.h:"
    )
    assert {loop.call_site for loop in loops} == {f"kernel.cc:{_line(f'copy {k}')}" for k in range(3)}
    for loop in loops:
        assert f"@{loop.source_ref} via {loop.call_site}: " in printed


def test_shared_block(measured: tuple[KernelReport, str]) -> None:
    """The block between consecutive loops is the first one's epilogue and the second one's prologue: it counts
    in the first one's cost only, the second one prints it in parentheses."""
    report, printed = measured
    loops = {loop.kernel_line: loop for loop in _loops(report, "two")}
    first, second = loops[_line("loop first")], loops[_line("loop second")]
    assert first.sched != "none" and second.sched != "none"
    assert second.prologue_block == first.epilogue_block and second.prologue_bundles == first.epilogue_bundles
    assert second.prologue_of == f"{first.block} @kernel.cc:{_line('loop first')}"
    assert first.epilogue_of == f"{second.block} @kernel.cc:{_line('loop second')}"
    assert second.prologue_cycles == 0 and first.prologue_cycles == first.prologue_bundles
    iters = second.effective_iters()
    assert iters is not None and second.ii is not None
    assert loop_cycles(report, second) == (iters * second.ii + (second.epilogue_bundles or 0), False)
    lines = _printed_loop(printed, f"@kernel.cc:{_line('loop second')}: ")
    assert lines[1].startswith(f"cost ({second.prologue_bundles}) + ") and "prologue shared: counted in" in lines[1]


def test_schedules_on_headers(measured: tuple[KernelReport, str]) -> None:
    """The asm printer reports each scheduled loop's II and NS on its header block, where its final remark goes."""
    report, _ = measured
    scheduled = sorted((function, info.name, info.schedule) for function, blocks in report.layout.items()
                       for info in blocks if info.schedule is not None)
    placed = sorted((loop.function, loop.block, (loop.ii, loop.ns)) for loop in report.loops
                    if loop.sched_block is not None)
    assert scheduled and placed == scheduled


def test_hardware_loop_ids(measured: tuple[KernelReport, str]) -> None:
    """Nested hardware-loop IDs stay strings (YAML 1.1 reads 0_0 as the integer 0)."""
    report, _ = measured
    assert "0_0" in {hw.loop_id for hw in report.hardware_loops}


def test_template_and_inlined_loops(measured: tuple[KernelReport, str]) -> None:
    """A for-header bounded by a template parameter takes the instance's argument (tiles<6>: 3 iterations); the
    loop around an inlined helper's loop is still its caller's statement (the nesting check spans one function)."""
    report, _ = measured
    (outer_c, _) = sorted(_loops(report, "tiles<6>"), key=lambda loop: loop.depth)
    assert outer_c.kernel_line == _line("loop c") and outer_c.trip_count == 3
    loops = {loop.kernel_line: loop for loop in _loops(report, "caller")}
    assert set(loops) == {_line("loop p"), _line("loop t")}
    assert loops[_line("loop p")].trip_count == 3 and loops[_line("loop t")].depth == 2


def test_names_and_trips_offline(tmp_path: Path) -> None:
    assert _function_name("void (anonymous namespace)::srec<32, 4>(unsigned char const*, float*)") == "srec<32, 4>"
    assert _function_name("(anonymous namespace)::record(int*, int, int)") == "record"
    assert _function_name("aie::vector_elem_ref<float, 32u>::operator float() const (.exit)") == (
        "aie::vector_elem_ref<float, 32u>::operator float.exit"
    )
    assert yaml.load("LoopID: 0_1\nLine: 12\n", Loader=_LLVMLoader) == {"LoopID": "0_1", "Line": 12}
    source = tmp_path / "trip.cc"
    source.write_text("for (unsigned j = 0; j < 8; j += 2) {}\nfor (int i = s + 1; i < 32; i++) {}\n")
    assert _parse_source_trip(source, 1) == 4
    assert _parse_source_trip(source, 2) is None
    source.write_text("template <int N, typename T, unsigned M = 2>\nvoid f(T *p) {\n"
                      "    for (int c = 0; c < N; c += M) {}\n}\n")
    args = _template_arguments(source, 3, "_Z1fILi8EfLj4EEvPT0_")  # (f<8, float, 4u>)
    assert args == {"N": "8", "M": "4u"}
    assert _parse_source_trip(source, 3, args) == 2


def test_statements_offline(tmp_path: Path) -> None:
    """Early exits of loop statements; the loop with more blocks at a shared location is the enclosing one."""
    text = (
        "for (int i = 0; i < 5; i++) {\n    if (x) break;\n    for (int j = 0; j < 3; j++) { if (y) break; }\n}\n"
        "for (int k = 0; k < 2; k++) { switch (k) { case 0: break; } }\n"
    )
    assert {statement.line: statement.exits for statement in _loop_statements(_tokenize(text))} == {
        1: True,
        3: True,
        5: False,
    }
    source = tmp_path / "k.cc"
    source.write_text("void f() {\n    for (int sg = 0; sg < 4; sg++) {\n        for (int j = 0; j < 8; j++)\n"
                      "            g(j);\n    }\n}\n")
    inner = SourceLoop(function="f", path="k.cc", line=3, column=9, num_blocks=4)
    outer = SourceLoop(function="f", path="k.cc", line=3, column=9, num_blocks=9)
    _read_statements([inner, outer], source=source)
    assert (inner.line, inner.end_line, inner.trip) == (3, 4, 8)
    assert (outer.line, outer.end_line, outer.trip) == (2, 5, 4)


def test_spill_counts_offline(tmp_path: Path) -> None:
    """Spill code in the assembly counts instructions (the comments of a bundle run together); the
    register allocator counts registers, its last run is complete."""
    asm = (
        "f:                                      // @f\n"
        ".LBB0_1:                                // %for.body\n"
        "                                        // =>This Inner Loop Header: Depth=1\n"
        "\tvlda\t bmll2, [sp, #-512];\t\tvst\t bmhl0, [sp, #-384] // 64-byte Folded Reload64-byte Folded Spill\n"
        "\tst\t r8, [sp, #-64]                  // 4-byte Folded Spill\n"
        "\tjnz\t r0, #.LBB0_1\n"
        ".Lfunc_end0:\n"
    )
    (block,) = _parse_assembly(asm)["f"].blocks
    assert (block.spills, block.spill_bytes, block.reloads, block.reload_bytes) == (2, 68, 1, 64)
    assert block.header and block.jumps == [".LBB0_1"]

    def remark(tag: str, pass_name: str, name: str, line: int, args: str) -> str:
        return (
            f"--- !{tag}\nPass: {pass_name}\nName: {name}\nDebugLoc: {{ File: k.cc, Line: {line}, Column: 5 }}\n"
            f"Function: f\nArgs:\n{args}...\n"
        )

    records = tmp_path / "k.cc.yaml"
    records.write_text(
        remark("Analysis", "pipeliner", "canPipelineLoop", 3, "  - String: 'Not a single basic block: '\n"
               "  - NumBlocks: '3'\n")
        + remark("Missed", "pipeliner", "canPipelineLoop", 3, "  - String: Failed to pipeline loop\n")
        + remark("Missed", "regalloc", "LoopSpillReloadCopies", 3, "  - NumVRCopies: '2'\n")
        + remark("Missed", "regalloc", "SpillReloadCopies", 1, "  - NumVRCopies: '2'\n")
        + remark("Missed", "regalloc", "LoopSpillReloadCopies", 3, "  - NumSpills: '1'\n  - NumReloads: '2'\n")
        + remark("Missed", "regalloc", "SpillReloadCopies", 1, "  - NumSpills: '3'\n  - NumReloads: '4'\n")
    )
    report = parse_yaml_remarks(records)
    (loop,) = report.source_loops["f"]
    assert loop.num_blocks == 3 and loop.regalloc == (1, 2)
    assert report.regalloc_spill_notes == ["f (function @k.cc:1): the register allocator spilled 3, reloaded 4 register(s)"]


def test_inlining_chain_offline() -> None:
    """A .loc's comment lists the call sites it is inlined at, innermost first; a comment of another location (or
    none) leaves the .loc's own."""
    comment = (" .venv/x/aie_api/vector.hpp:1098:29 @[ src/artlite/operators/kernel_lib.h:41:27 @[ "
               "/tmp/k/kernel.cc:0:3 @[ /tmp/k/kernel.cc:315:9 ] ] ]")
    assert _inlining_chain("vector.hpp", 1098, comment) == (("vector.hpp", 1098), ("kernel_lib.h", 41),
                                                             ("kernel.cc", 315))
    assert _inlining_chain("kernel.cc", 12, comment) == (("kernel.cc", 12),)
    assert _inlining_chain("kernel.cc", 12, "") == (("kernel.cc", 12),)


def _zol_counts(asm: str) -> dict[int, int | None]:
    (function,) = _parse_assembly(asm).values()
    return _zol_iterations(function, _asm_loops(function))


def test_zol_count_dataflow_offline() -> None:
    """Register values flow along the jumps: an if-chain's branch sees the lc register set before its jump, not
    the writes of the branch laid out between; a call clobbers the return-value registers, not the others."""
    asm = (
        "f:                                      // @f\n"
        "// %bb.0:                               // %entry\n"
        "\tmova\tr1, #2\n\teq\t r2, r0, r1\n\tjnz\t r2, #.LBB0_2\n\tnop\n"
        "// %bb.1:                               // %if.else\n"
        "\tmova\tr1, #1\n\teq\t r1, r0, r1\n\tj\t#.LBB0_4\n\tnop\n"
        ".LBB0_2:                                // %if.then\n"
        "\tadd.nc\tlc, r1, #-1;\t\tmovxm\tls, #.LBB0_3\n\tmovxm\tle, #.L_LEnd0\n"
        ".LBB0_3:                                // %for.body\n"
        "                                        // =>This Inner Loop Header: Depth=1\n"
        "\tnop\n.L_LEnd0:\n\tvst\t x2, [p1], #64\n"
        ".LBB0_4:                                // %if.end\n"
        "\tret\tlr\n"
        ".Lfunc_end0:\n"
    )
    assert _zol_counts(asm) == {3: 1}
    call = "\tjl\t#g\n" + "\tnop\n" * 5
    for register, count in (("r0", None), ("r16", 7)):
        asm = (
            "f:                                      // @f\n"
            "// %bb.0:                               // %entry\n"
            f"\tmova\t{register}, #8\n{call}\tadd.nc\tlc, {register}, #-1;\t\tmovxm\tls, #.LBB0_1\n"
            ".LBB0_1:                                // %for.body\n"
            "                                        // =>This Inner Loop Header: Depth=1\n"
            "\tnop\n"
            ".Lfunc_end0:\n"
        )
        assert _zol_counts(asm) == {1: count}, register


def test_tail_calls_offline() -> None:
    """A tail call (an unconditional `j` to a function symbol) is a call; a jump to a local label is not."""
    asm = (
        "f:                                      // @f\n"
        "// %bb.0:                               // %entry\n"
        "\tnopa\t;\t\tjl\t#g\n"
        "\tnopa\t;\t\tj\t#_ZN12_GLOBAL__N_11hEv;\t\tnopv\t\n"
        "\tj\t#.LBB0_2\n"
        ".Lfunc_end0:\n"
    )
    (block,) = _parse_assembly(asm)["f"].blocks
    assert block.calls == ["g", "_ZN12_GLOBAL__N_11hEv"]
