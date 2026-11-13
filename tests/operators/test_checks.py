"""The build-time checks (operators/checks.py) on canned disassembly, a stack usage file and small designs (no NPU, no
toolchain), and the disassembly parse they share (operators/objdump.py)."""

import pytest

from artlite.operators import checks, objdump

# a kernel object: ``entry`` calls ``mid`` (which calls ``leaf``, ``__mulsf3`` and rounds) and ``setter`` (sets the
# rounding mode, then calls ``leaf``); ``leaf`` rounds; ``self_call`` calls itself and takes a lock; the core's control
# code ``core_1_2`` reserves a stack frame
DISASSEMBLY = """
kernel.o:	file format elf32-aie2p

Disassembly of section .text.entry:

00000000 <entry>:
       0: 	paddxm	[sp], #0x40
       2: 	jl	#0x0
		00000002:  R_AIE_15	mid
       6: 	jl	#0x0
		00000006:  R_AIE_15	setter
       a: 	jl	#0x0
		0000000a:  R_AIE_15	.text.leaf

00000040 <mid>:
       0: 	jl	#0x0
		00000000:  R_AIE_15	leaf
       4: 	jl	#0x0
		00000004:  R_AIE_15	__mulsf3
       8: 	vconv.bf16.fp32	wl0, bml0

00000080 <setter>:
       0: 	mov	crrnd, r0
       2: 	jl	#0x0
		00000002:  R_AIE_15	leaf

000000c0 <leaf>:
       0: 	vst.srs.s16.s32	[p0], bmh0, s0
       4: 	ret	lr

00000100 <self_call>:
       0: 	acq	#0x34, r8
       2: 	jnz	r1, #0x0
       6: 	jl	#0x0
		00000006:  R_AIE_15	self_call

00000140 <core_1_2>:
       0: 	paddxm	[sp], #0x80
"""
SU = "kernel.cc:3:6:entry\t16\tstatic\nkernel.cc:9:6:mid\t32\tstatic\nkernel.cc:12:6:setter\t8\tstatic\n" \
     "kernel.cc:20:6:leaf\t48\tstatic\nkernel.cc:30:6:self_call\t24\tstatic\n"


@pytest.fixture
def obj(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    """A kernel object whose disassembly is the canned text (and its stack usage file)."""
    path = str(tmp_path / "kernel.o")
    with open(path, "wb"):
        pass
    with open(str(tmp_path / "kernel.su"), "w") as f:
        f.write(SU)
    monkeypatch.setattr(objdump, "_run", lambda paths, cwd: DISASSEMBLY)
    return path


def test_parse() -> None:
    f = objdump.parse(DISASSEMBLY)
    assert list(f) == ["entry", "mid", "setter", "leaf", "self_call", "core_1_2"]
    assert f["entry"].relocs == ("mid", "setter", ".text.leaf") and f["entry"].frame == 0x40
    assert f["mid"].rounds and not f["mid"].sets_rounding and f["setter"].sets_rounding and f["leaf"].rounds
    assert f["self_call"].acquires == {0x34} and [j.split("\t", 1)[1] for j in f["self_call"].jumps] == ["jnz\tr1, #0x0", "jl\t#0x0"]
    assert f["core_1_2"].frame == 0x80 and f["leaf"].frame is None and not f["leaf"].jumps


def test_parse_objects() -> None:
    """The heads of an output of several objects split it."""
    text = "a.o:\tfile format elf32-aie2p\n\n00000000 <x>:\n 0: acq #5, r0\nb.o:\tfile format elf32-aie2p\n\n" \
           "00000000 <y>:\n 0: acq #6, r0\n"
    parts = {}
    heads = list(objdump._FILE.finditer(text))
    for m, n in zip(heads, heads[1:] + [None]):
        parts[m[1]] = objdump.parse(text[m.end():n.start() if n else len(text)])
    assert parts["a.o"]["x"].acquires == {5} and parts["b.o"]["y"].acquires == {6}


def test_stack_usage(obj: str) -> None:
    su = obj[:-1] + "su"
    use = checks.stack_usage(obj, su)
    # entry (16) + mid (32) + leaf (48); __mulsf3 has no frame record: 64
    assert use["entry"] == 16 + 32 + 64 and use["setter"] == 8 + 48 and use["leaf"] == 48
    assert use["self_call"] == 24  # (its own call is not a callee)
    loop = checks.stack_usage(obj, su, calls={"leaf": {"mid"}})  # (recursion: counted once, not forever)
    assert loop["leaf"] == 48 + 32 + 64 and loop["mid"] == 32 + max(48 + 32, 64)  # (a cycle's frames once, in any order)
    more = checks.stack_usage(obj, su, calls={"leaf": {"setter"}})  # (an indirect call: leaf -> setter -> leaf)
    assert more["leaf"] == 48 + 8 + 48
    ind = checks.stack_usage(obj, su, indirect=[(r"^(ent)ry$", r"^s[a-z_]+$")])  # (entry calls every function s...)
    assert ind["entry"] == 16 + max(32 + 64, 8 + 48, 24 + 24)


def test_rounding_and_libcalls(obj: str) -> None:
    assert checks.unset_rounding(obj, {"entry"}) == {"entry": ["leaf", "mid"]}  # (leaf through mid: nobody sets it)
    assert checks.unset_rounding(obj, {"setter"}) == {}
    assert checks.unset_rounding(obj, {"nope"}) == {}
    assert checks.library_arithmetic(obj) == {"mid": {"__mulsf3"}}


def test_core_frames(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    prj = tmp_path / "design.mlir.prj" / "objects_main_core_1_2"
    prj.mkdir(parents=True)
    (prj / "objects_main_core_1_2.o").write_bytes(b"")
    monkeypatch.setattr(objdump, "_run", lambda paths, cwd: f"{paths[0]}:\tfile format elf32-aie2p\n" + DISASSEMBLY)
    assert checks.core_frames(str(tmp_path)) == {"1,2": 0x80}
    assert checks.core_frames(str(tmp_path / "none")) == {}


SOURCE = """
namespace {
// a comment mentioning void k_ok(int a)
void k_ok(int32_t *__restrict a, const int16_t *b, int32_t n) {
    k_other(a, n);
}
int k_bad_ret(int32_t *a) { return 0; }
void k_arity(int32_t *a) {}
void k_scalar(int32_t *a, int16_t n) {}
void k_ptr(int32_t n) {}
}
"""
DECLS = {"k_ok": ["memref<8xi32>", "memref<8xi16>", "i32"], "k_bad_ret": ["memref<8xi32>"],
         "k_arity": ["memref<8xi32>", "i32"], "k_scalar": ["memref<8xi32>", "i32"], "k_ptr": ["memref<8xi32>"],
         "k_elsewhere": ["i32"]}


def test_signatures() -> None:
    errors = checks._check_signatures(SOURCE, DECLS)
    assert len(errors) == 4, errors
    assert "k_bad_ret: kernel entry points must return void" in errors[0]
    assert "k_arity: 1 parameters in the source, 2 in the design" in errors[1]
    assert "k_scalar arg 1: design i32 (4 B) vs source 'int16_t n' (2 B)" in errors[2]
    assert "k_ptr arg 0: design memref<8xi32> vs source 'int32_t n'" in errors[3]
    assert checks._check_signatures(SOURCE, {"k_ok": DECLS["k_ok"]}) == []


MLIR = """module {
  aie.device(npu2) {
    func.func private @k(memref<16xi32>, i32)
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "b"} : memref<16xi32>
    %c = aie.core(%t) {
      %v = arith.constant 1 : i32
      func.call @k(%b, %v) : (memref<16xi32>, i32) -> ()
      aie.end
    } {stack_size = 256 : i32}
    %t2 = aie.tile(1, 2)
    %c2 = aie.core(%t2) {
      %v = arith.constant 1 : i32
      func.call @k(%b, %v) : (memref<16xi32>, i32) -> ()
      aie.end
    }
  }
}"""


def test_design() -> None:
    decls, cores = checks._design(MLIR)
    assert decls == {"k": ["memref<16xi32>", "i32"]}
    assert cores == [("0,2", 256, {"k"}), ("1,2", checks.DEFAULT_STACK, {"k"})]


def test_analyze(obj: str, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors (a core whose deepest call and own frames exceed its stack) and warnings (rounding, library calls), and
    ``ARTLITE_CHECKS=strict`` makes the warnings errors."""
    mlir = MLIR.replace("@k", "@entry").replace("memref<16xi32>, i32", "").replace("(%b, %v)", "()").replace(
        "(memref<16xi32>, i32) -> ()", "() -> ()")
    errors, warnings = checks.analyze(source=None, mlir_text=mlir, obj_path=obj)
    # entry needs 16 + 32 + 64 = 112 B, + the core's frame (128: no per-core objects) + 64 = 304 B > 256
    assert len(errors) == 1 and "core (0,2)" in errors[0] and "need 304 B of stack" in errors[0], errors
    assert len(warnings) == 2 and warnings[0].startswith("kernels that round without setting") and "mid" in warnings[0]
    assert "scalar arithmetic through runtime library calls" in warnings[1]
    monkeypatch.setenv("ARTLITE_CHECKS", "strict")
    with pytest.raises(checks.DesignCheckError, match="round without setting"):
        checks.check(source=None, mlir_text=mlir.replace("256", "4096"), obj_path=obj, label="T")
    monkeypatch.setenv("ARTLITE_CHECKS", "")
    checks.check(source=None, mlir_text=mlir.replace("256", "4096"), obj_path=obj, label="T")
    monkeypatch.setenv("ARTLITE_SKIP_CHECKS", "1")
    checks.check(source=None, mlir_text=mlir, obj_path=obj, label="T")
