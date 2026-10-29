"""artlite.variant_search: slot counts, variant patches, post-pipeliner traces, the report (compiles with Peano,
no NPU)."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import pytest

from artlite import aie2p_isa, gen_isa_tables
from artlite.measure_ii import SLOTS, slot_of
from artlite.operators.base import CLANGPP
from artlite.variant_search import (
    POST,
    Variant,
    _cli_variants,
    _region_result,
    apply_pragma,
    compiler_drift,
    count_slots,
    ii,
    parse_operator,
    parse_postpipeliner_log,
    parse_regions,
    pragma_lines,
    pragma_text,
    print_report,
    replace_function,
    search,
    substitute,
)

SOURCE = """namespace {

template <int R>
__attribute__((noinline)) void pfx_codes(const float *__restrict x, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < R; t++) {  // outer
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int b = 0; b < 8; b++) {  // inner
            AIE_LOOP_UNROLL_FULL
            for (int s = 0; s < 4; s++)  // unrolled
                y[s] += x[s];
            x += 4;
        }
    }
}

__attribute__((noinline)) void pfx_other(float *__restrict y)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 8; i++)  // other
        y[i] = 0;
}

}  // namespace

extern "C" void entry(const float *x, float *y)
{
    pfx_codes<4>(x, y);
    pfx_other(y);
}
"""


def _line(text: str, marker: str) -> str:
    return next(line for line in text.splitlines() if line.endswith(f"// {marker}"))


def _before(text: str, marker: str) -> str:
    """The line before the one ending with the marker."""
    lines = text.splitlines()
    return lines[lines.index(_line(text, marker)) - 1]


def test_slot_of() -> None:
    """The compiler's slot of each mnemonic; operands decide for mov, vmov, j; nops are not counted."""
    assert slot_of("vshuffle\tx11, x8, x2, r1") == "mv"
    assert slot_of("vups.4x\tdm2, x3, s0, upssign0") == "mv"
    assert slot_of("vsrs.4x\tx9, dm2, s1, srssign0") == "st"
    assert slot_of("movs\tp2, p4") == "st"
    assert slot_of("vldb.unpack\t x3, unpacksign0, [p6], #32") == "ldb"
    assert slot_of("vlda.2d\t x10, [p5], d1") == "lda"
    assert slot_of("vmac.f\tdm0, dm0, y3, y4, r2") == "vec"
    assert slot_of("vst.conv.bf16.fp32\t cml4, [p3], #64") == "st"
    assert slot_of("mov\ts3, r3") == "mv"
    assert slot_of("mov\tr1, ss") == "lda"
    assert slot_of("mov\tms, r2") == "st"
    assert slot_of("mov\tr1, r2.flt2fx, s3") == "alu"
    assert slot_of("vmov\tx1, x2") == "mv"
    assert slot_of("vmov\tmcd, x2") == "st"
    assert slot_of("jl\t#memcpy") == "lng"
    assert slot_of("jl\tp0") == "alu"
    assert slot_of("movxm\tp0, #.LCPI0_0") == "lng"
    assert slot_of("nopv") is None and slot_of("nop") is None
    assert slot_of("vfrobnicate\tx1") == "?"


def test_isa_tables() -> None:
    """slot_of tells every mnemonic of several slots apart by its operands (a new one would be "?")."""
    assert set(aie2p_isa.SLOT_MNEMONICS) == set(SLOTS)
    for mnemonic, slots in aie2p_isa.SHARED_MNEMONICS.items():
        assert mnemonic in ("mov", "mov.nb", "vmov", "j", "jl"), f"{mnemonic} ({slots}): add it to slot_of"


def test_isa_tables_current() -> None:
    """aie2p_isa.py is what gen_isa_tables generates for the wheel's llvm-aie commit (python -m
    artlite.gen_isa_tables after a toolchain upgrade). Needs an llvm-aie checkout with its llvm-tblgen."""
    tblgen = gen_isa_tables.LLVM_AIE / "build" / "bin" / "llvm-tblgen"
    if not tblgen.exists():
        pytest.skip(f"no llvm-tblgen at {tblgen}")
    try:
        text = gen_isa_tables.generate()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert text == gen_isa_tables.TABLES.read_text(), "stale: run python -m artlite.gen_isa_tables"


def test_compiler_drift() -> None:
    """The wheel's commit is in its --version; its own compiler does not drift from itself."""
    commit = gen_isa_tables.clang_commit(CLANGPP)
    assert commit is not None and len(commit) == 40
    assert compiler_drift(CLANGPP) is None


def test_count_slots() -> None:
    """Per-slot counts of a loop body; a lng op takes the alu and the mv slot; the busiest slot binds."""
    body = [
        ["vlda\t bmlh1, [p2, #64]", "paddb\t [p6], m2", "nops", "nopx", "mov\ts3, r3", "vmac.f\tdm0, dm0, y3, y4, r2"],
        ["vldb.unpack\t x3, unpacksign0, [p6], #32", "vband\t x8, x8, x0"],
        ["vshuffle\tx8, x9, x2, r0"],
        ["movxm\tp0, #.LCPI0_0", "vsrs.4x\tx9, dm2, s1, srssign0"],
    ]
    slots = count_slots(body)
    assert slots.bundles == 4
    assert {slot: slots.count(slot) for slot in ("lda", "ldb", "st", "vec", "mv", "alu", "lng")} == {
        "lda": 1, "ldb": 2, "st": 1, "vec": 1, "mv": 3, "alu": 0, "lng": 1,
    }
    assert slots.pressure["mv"] == 4 and slots.pressure["alu"] == 1
    assert slots.bound == 4 and slots.binding == ["mv"]
    assert slots.describe("mv") == "mov 1, vband 1, vshuffle 1, movxm 1"
    assert slots.count("rm") == 1 and slots.count("qwb") == 1 and slots.count("qwa") == 1 and slots.count("ada") == 1


def test_count_ports() -> None:
    """Ops of different slots that share a port: shuffles (mv) and a bfp16 stream's conversion pushes and flushes
    (st) all read L_RM_PORT, so the port binds above both slots (a dequant loop at II 46 with mv 30, st 16)."""
    body = [["vshuffle\tx8, x9, x2, r0", "vst.push.576.conv.bfp16ebs8.fp32\t dm1, [p2, sf, r26]"],
            ["vshuffle\tx3, x9, x2, r1", "vst.flush.512.conv\t [p2, sf, r26]"],
            ["vshuffle\tx4, x9, x2, r1"]]
    slots = count_slots(body)
    assert slots.count("mv") == 3 and slots.count("st") == 2
    assert slots.count("rm") == 5 and slots.count("wm") == 2
    assert slots.bound == 5 and slots.binding == ["rm"]
    assert slots.busiest_port == ("rm", 5)
    assert slots.describe("rm") == "vshuffle 3, vst.push.576.conv.bfp16ebs8.fp32 1, vst.flush.512.conv 1"


def test_lock_port() -> None:
    """A lock op holds the core's lock-request unit 4 cycles: an acquire + release loop (an ObjectFifo drain) is
    bound at 8 by it, not by its scalar ops."""
    slots = count_slots([["acq\t#53, r1"], ["add\tr0, r0, #-1"], ["jnz\tr0, #0x0"], ["rel\t#52, r2"]])
    assert slots.count("alu") == 3 and slots.count("lck") == 8  # (jnz: lng)
    assert slots.bound == 8 and slots.binding == ["lck"]


def test_pragma_variants() -> None:
    """A pragma replaces the pipelining pragma of the target's innermost rolled loops, other pragmas stay;
    unrolled loops, calls and other functions are left alone."""
    patched = apply_pragma(SOURCE, ["_codes"], ii(44))
    assert _before(patched, "inner").strip() == "AIE_TRY_INITIATION_INTERVAL(44)"
    assert POST not in _before(patched, "inner") and patched.count(POST) == 1  # (pfx_other's)
    assert _before(patched, "outer").strip() == "AIE_LOOP_NO_UNROLL"
    assert _before(patched, "unrolled").strip() == "AIE_LOOP_UNROLL_FULL"
    assert _before(patched, "other").strip() == POST
    assert _before(patched, "inner").startswith("        AIE_")  # (the loop's indentation)
    removed = apply_pragma(SOURCE, ["_codes"], "")
    assert len(removed.splitlines()) == len(SOURCE.splitlines()) - 1
    assert _before(removed, "inner").strip() == "for (int t = 0; t < R; t++) {  // outer"
    # the outer loop (rolled loop 0): its NO_UNROLL stays, the pragma goes between it and the loop
    outer = apply_pragma(SOURCE, ["_codes"], POST, loops=[0])
    assert _before(outer, "outer").strip() == POST and outer.count(POST) == 3
    assert pragma_text("post+ii44") == f"{POST} {ii(44)}" and pragma_text("none") == "" and pragma_text("base") is None
    with pytest.raises(ValueError, match="no definition"):
        apply_pragma(SOURCE, ["_missing"], POST)
    with pytest.raises(ValueError, match="out of range"):
        apply_pragma(SOURCE, ["_codes"], POST, loops=[2])


HELPERS = """namespace {

[[clang::always_inline]] inline void pfx_row(const float *__restrict x, float *__restrict y)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int s = 0; s < 8; s++)  // helper
        y[s] += x[s];
}

[[clang::always_inline]] inline void pfx_pair(const float *__restrict x, float *__restrict y)
{
    pfx_row(x, y);
    pfx_row(x + 8, y);
}

__attribute__((noinline)) void pfx_rows(const float *__restrict x, float *__restrict y)
{
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 4; t++) {  // caller
        pfx_pair(x + 16 * t, y);
    }
    for (int t = 0; t < 4; t++)  // own
        y[t] = 0;
}

}  // namespace
"""


def test_helper_pragmas() -> None:
    """The loops of the always_inline helpers a target calls (also through other helpers) take pragmas: indices
    count the target's rolled loops first; a loop calling a helper with loops is not innermost."""
    patched = apply_pragma(HELPERS, ["_rows"], ii(9))
    assert _before(patched, "helper").strip() == ii(9) and _before(patched, "own").strip() == ii(9)
    assert _before(patched, "caller").strip() == "AIE_LOOP_NO_UNROLL"
    caller = apply_pragma(HELPERS, ["_rows"], POST, loops=[0])
    assert _before(caller, "caller").strip() == POST
    helper = apply_pragma(HELPERS, ["_rows"], ii(9), loops=[2])
    assert _before(helper, "helper").strip() == ii(9) and POST not in _before(helper, "own")
    (line,) = pragma_lines(helper, ["_rows"], loops=[2])
    assert helper.splitlines()[line - 1].endswith("// helper")
    with pytest.raises(ValueError, match="3 rolled loop.*always_inline helpers"):
        apply_pragma(HELPERS, ["_rows"], POST, loops=[3])


def test_substitutions() -> None:
    """A substitution must match exactly once; a definition is replaced whole (template header included), a
    call is no definition."""
    assert "y[s] -= x[s]" in substitute(SOURCE, "y[s] += x[s]", "y[s] -= x[s]")
    with pytest.raises(ValueError, match="matches 0 times"):
        substitute(SOURCE, "y[s] *= x[s]", "")
    with pytest.raises(ValueError, match="matches 2 times"):
        substitute(SOURCE, "AIE_PREPARE_FOR_POSTPIPELINING", "")
    definition = "template <int R>\nvoid pfx_codes(const float *x, float *y) {}\n"
    replaced = replace_function(SOURCE, "_codes", definition)
    assert "// outer" not in replaced and definition + "\n" in replaced
    assert "pfx_codes<4>(x, y);" in replaced and "// other" in replaced
    with pytest.raises(ValueError, match="2 definitions"):
        replace_function(SOURCE, "pfx_", "")


LOG = """\
>> enterFunction _ZN12_GLOBAL__N_19pfx_codesILi4EEEvPKfPf
  >> enterBlock 1 Loop FixPointIter=0 II=0
 PostPipeliner: TargetII=6
  << leaveBlock 1
PostPipeliner: ResMII=5
  >> enterBlock 1 Loop FixPointIter=3 II=5
0 renamable $x0, renamable $p0 = VLDA_dmw_lda_w renamable $p0(tied-def 1), 64
1 renamable $x1 = VSHUFFLE_vec_shuffle_x renamable $x0, renamable $x0, renamable $r0
2 VST_dmw_sts_w renamable $x1, renamable $p1
RecMII=7
digraph PostPipeliner_II5_Physical {
}
RecMII=3
digraph PostPipeliner_II5_Virtual {
}
ScarceRegMII=2
-- MinLength=9
Starting Config_100_1_0_NodeNum
Latency not met for SU4 in cycle 6 (Earliest=8 ModuloNode=SU1)
  --> SU0 @0
  --> SU1 @2
  --> SU4 @0
  Try to delay SU1 with TweakedEarliest= 1
    Strategy Config_100_1_0_NodeNum failed
Out of resources
PostPipeliner: Register allocation failed
   Register allocation failed
    Strategy Config_100_1_0_Latest failed
=== II=5 Failed ===
PostPipeliner: No schedule found
RecMII=7
digraph PostPipeliner_II6_Physical {
}
-- MinLength=9
PostPipeliner: Physical mode - no allocation needed
PostPipeliner: Success
>> enterFunction _ZN12_GLOBAL__N_19pfx_otherEPf
  >> enterBlock 2 Loop FixPointIter=0 II=0
 PostPipeliner: No ZOL
  >> enterBlock 3 Loop FixPointIter=0 II=0
PostPipeliner: ResMII=61
"""


def test_postpipeliner_log() -> None:
    """Per loop block: ResMII, each II's attempts per mode and why they failed, the latency chains with their
    instructions, the TargetII printed before the ResMII, loops that are no candidate or not tried."""
    traces = parse_postpipeliner_log(io.StringIO(LOG))
    codes = traces[("_ZN12_GLOBAL__N_19pfx_codesILi4EEEvPKfPf", 1)]
    assert codes.res_mii == 5 and codes.target_ii == 6 and len(codes.sus) == 3
    assert [(a.ii, a.mode, a.rec_mii, a.ran, a.success) for a in codes.attempts] == [
        (5, "Physical", 7, False, False), (5, "Virtual", 3, True, False), (6, "Physical", 7, True, True),
    ]
    virtual = codes.attempts[1]
    assert (virtual.scarce_mii, virtual.strategies, virtual.resources, virtual.regalloc) == (2, 2, 1, 1)
    assert dict(virtual.latency) == {1: 1} and codes.chains[1] == [0, 1, 4]
    head, physical, virtual_line, latency = codes.describe()
    assert head == "ResMII 5, TargetII 6; tried II 5..6 -> II 6 (Physical)"
    assert physical == "  Physical: RecMII 7; II 5 < RecMII"
    assert virtual_line == (
        "  Virtual: RecMII 3, ScarceRegMII 2; II 5: 2 strategy runs failed "
        "(latency not met 1, out of resources 1, register allocation failed 1)"
    )
    assert "SU1 VSHUFFLE_vec_shuffle_x" in latency and "SU0 VLDA -> SU1 VSHUFFLE -> SU1' VSHUFFLE" in latency
    other = "_ZN12_GLOBAL__N_19pfx_otherEPf"
    assert traces[(other, 2)].describe() == ["not a post-pipeliner candidate: No ZOL"]
    assert traces[(other, 3)].describe() == ["ResMII 61 > 60 (-aie-postpipeliner-maxii): no II tried"]


def test_cli_parsing(tmp_path: Path) -> None:
    from artlite.decode import gemv
    from artlite.gemm.gemm_q import GemmQOperator

    assert parse_operator('gemm.gemm_q.GemmQOperator(fmt="Q4_K", M=64)') == (GemmQOperator, {"fmt": "Q4_K", "M": 64})
    assert parse_operator('decode.gemv.render(p="v", fmts=["Q5_K"])') == (
        gemv.render, {"p": "v", "fmts": ["Q5_K"]},
    )
    (tmp_path / "new.cc").write_text("NEW\n")
    args = argparse.Namespace(
        variants=None, pragmas="post,ii44", pragma=[["x", "none"]], sub=[["x", "OLD", "@" + str(tmp_path / "new.cc")]],
        define=[], param=[["y", "nsb=8"], ["y", "win=codes"]], loop=None,
    )
    variants = _cli_variants(args)
    assert [variant.name for variant in variants] == ["base", "post", "ii44", "x", "y"]
    assert variants[3].pragma == "" and variants[3].subs == [("OLD", "NEW\n")]
    assert variants[4].params == {"nsb": 8, "win": "codes"} and variants[4].pragma is None
    assert parse_regions(["codes=_q5_codes*4+_q5_scale", "tile"]) == {
        "codes": [("_q5_codes", 4), ("_q5_scale", 1)], "tile": [],
    }


def test_counter_windows(tmp_path: Path) -> None:
    """Counters JSON of a profiled build: the first sample is dropped; cycles and memory stall per window."""
    events = [{"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "tile(0,2)"}}]
    labels = ["INSTR_EVENT_0->INSTR_EVENT_1 cycles", "INSTR_EVENT_0->INSTR_EVENT_0 count",
              "MEMORY_STALL->MEMORY_STALL count", "LOCK_STALL->LOCK_STALL count"]
    events += [{"name": "thread_name", "ph": "M", "pid": 1, "tid": k + 1, "args": {"name": name}}
               for k, name in enumerate(labels)]
    for sample, values in enumerate([(9999, 1, 9, 9), (3300, 10, 100, 500), (3300, 10, 80, 700)]):
        events += [{"name": name, "ph": "C", "ts": sample, "pid": 1, "tid": k + 1, "args": {"value": value}}
                   for k, (name, value) in enumerate(zip(labels, values))]
    (tmp_path / "Op_1234_counters.json").write_text(json.dumps(events))
    region = _region_result(tmp_path)
    assert (region.cycles, region.windows, region.memory_stall, region.lock_stall) == (330, 10, 9, 600)


KERNEL = """namespace {

__attribute__((noinline)) void vs_scale(const float *__restrict x, float *__restrict y)
{
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int b = 0; b < 16; b++)  // loop b
        aie::store_v(y + 16 * b, aie::add(aie::load_v<16>(x + 16 * b), aie::load_v<16>(y + 16 * b)));
}

}  // namespace

extern "C" void vs_entry(const float *__restrict x, float *__restrict y)
{
    vs_scale(x, y);
}
"""


def test_search_report(tmp_path: Path) -> None:
    """Variants of a kernel compiled in parallel: one row per variant in the loop's table, ranked by cycles; a
    variant whose source equals an earlier one is measured once."""
    variants = [Variant("base"), Variant("post", pragma=POST), Variant("none", pragma=""),
                Variant("sub", subs=[("aie::add(", "aie::sub(")])]
    result = search(lambda: KERNEL, {}, functions=["vs_scale"], variants=variants, totals=["vs_entry"],
                    out_dir=tmp_path)
    by_name = {item.variant.name: item for item in result.variants}
    assert all(item.error is None for item in result.variants), [item.error for item in result.variants]
    assert by_name["post"].same_as == "base" and by_name["post"].report is by_name["base"].report
    assert by_name["base"].pragma_lines == []
    for name in ("post", "none"):  # (lines of the compiled file, the loop statement)
        (line,) = by_name[name].pragma_lines
        assert by_name[name].source.read_text().splitlines()[line - 1].endswith("// loop b")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_report(result)
    text = printed.getvalue()
    assert "vs_scale loop 1/1: " in text and "vs_entry" in text.split("cycles per call")[1].splitlines()[1]
    lines = text.split("vs_scale loop 1/1")[1].splitlines()
    header = lines[1].split()
    assert header[:7] == ["variant", "pragma", "sched", "II", "NS", "iters", "cycles"]
    rows = {line.split()[0]: line for line in lines[2:6]}
    assert set(rows) == {"base", "post", "none", "sub"} and "(= base)" in rows["post"]
    base = by_name["base"].report
    assert base is not None
    (loop,) = [loop for loop in base.loops if "vs_scale" in loop.function]
    assert loop.asm_body and len(loop.asm_body) == loop.body_bundles
    assert f" {loop.ii} " in rows["base"]


class _Flagged:
    """An operator stand-in whose parameter only changes its compiler flags."""

    def __init__(self, flags: tuple[str, ...] = ()):
        self.clang_flags = list(flags)

    def get_kernel_src(self) -> str:
        return KERNEL


def test_search_flags(tmp_path: Path) -> None:
    """Variants of one source with different compiler flags are compiled each; equal flags share a report."""
    variants = [Variant("base"), Variant("same", params={"flags": ()}),
                Variant("flag", params={"flags": ("-mllvm", "--enable-aie-zol-without-minitercount=false")})]
    result = search(_Flagged, {}, functions=["vs_scale"], variants=variants, out_dir=tmp_path)
    by_name = {item.variant.name: item for item in result.variants}
    assert all(item.error is None for item in result.variants), [item.error for item in result.variants]
    assert by_name["same"].same_as == "base" and by_name["flag"].same_as is None
    assert by_name["flag"].clang_cmd[-1] == "--enable-aie-zol-without-minitercount=false"
