"""Build-time checks of an operator's kernels against its design (run by every compile).

- Stack: the deepest kernel call of each core (``-fstack-usage`` frames plus the
  call graph of ``kernel.o``) and the frame of the core's own function (aiecc's
  control code, on the same stack: 64 or 128 B) must fit the core's ``stack_size``.
  An overflow is silent on the device and corrupts the neighbouring buffer (a
  kernel that needed exactly its core's stack_size zeroed parts of a workspace).
- Signatures: every kernel the design declares (``func.func private @name(...)``)
  must be defined in the kernel source with a ``void`` return, the same argument
  count, pointers for memrefs and scalars of the same width (an ``int`` return
  once shifted IRON's scalar argument registers).
- Rounding: every kernel entry whose code rounds (vector conversions and SRS
  stores), itself or in a function it calls, sets the rounding mode
  (``aie::set_rounding``: the entry or a function on the path to the rounding
  one); the mode is per-core state that survives kernel calls, so relying on
  another kernel to set it breaks when that kernel is absent. Reported as a
  warning.
- Library arithmetic: calls of compiler-rt arithmetic helpers (the core has no scalar floating
  point: a scalar float product is ``__mulsf3``; a signed division by a constant is ``__divsi3``):
  slow, and each helper costs program memory. Reported as a warning; vectorize the math (e.g.
  ``al::inv_rms``) or use shifts / ``al::div_small``.

Errors raise ``DesignCheckError``; ``ARTLITE_SKIP_CHECKS=1`` skips the checks.
"""

import glob
import os
import re
import subprocess
import sys
from collections.abc import Iterator

from .base import PEANO_DIR

OBJDUMP = os.path.join(PEANO_DIR, "bin", "llvm-objdump")
_UNKNOWN_FRAME = 64  # stack of a callee without a frame record (runtime library helpers)
CORE_FRAME = 128  # a core's own frame when its object is not kept (the largest measured)
MAIN_INIT_FRAME = 64  # the crt's _main_init frame below main's (paddxm [sp], #0x40 before it calls main)
_SCALAR_BYTES = {"int32_t": 4, "uint32_t": 4, "int": 4, "unsigned": 4, "float": 4, "int16_t": 2, "uint16_t": 2,
                 "short": 2, "int8_t": 1, "uint8_t": 1, "char": 1, "int64_t": 8, "uint64_t": 8, "bool": 1}
_MLIR_BYTES = {"i32": 4, "ui32": 4, "si32": 4, "f32": 4, "i16": 2, "i8": 1, "ui8": 1, "i64": 8, "index": 8, "f64": 8}
_LIBCALL = re.compile(r"^__(?:(?:add|sub|mul|div|neg|pow)[sdt]f[23]|(?:eq|ne|lt|le|gt|ge|unord|cmp)[sdt]f2|"
                      r"float(?:un)?[sdt]i[sdt]f|fix(?:uns)?[sdt]f[sdt]i|extend[sdt]f[sdt]f2|trunc[sdt]f[sdt]f2|"
                      r"(?:u?div|u?mod|mul)[sd]i3)$")
_ROUNDING = re.compile(r"\b(vst\.conv|vconv\.bf16|vconv\.bfp16|vsrs|vst\.srs|vst\.push\.conv|vst\.flush\.conv)")


class DesignCheckError(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get("ARTLITE_SKIP_CHECKS", "0") != "1"


def _disassembly(obj_path: str) -> dict[str, list[str]]:
    """Function name -> its disassembly lines with relocations (``llvm-objdump -dr``)."""
    out = subprocess.run([OBJDUMP, "-dr", obj_path], capture_output=True, text=True, check=True).stdout
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if m:
            if not m.group(1).startswith("."):
                cur = m.group(1)
                funcs.setdefault(cur, [])
            continue
        if cur is not None:
            funcs[cur].append(line)
    return funcs


def stack_usage(obj_path: str, su_path: str, *, calls: dict[str, set[str]] | None = None) -> dict[str, int]:
    """Deepest stack (bytes) of every function of kernel.o: its frame plus the deepest callee (``calls``: more calls,
    by function name: indirect ones)."""
    own: dict[str, int] = {}
    with open(su_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                own[parts[0].rsplit(":", 1)[-1]] = int(parts[1])
    graph: dict[str, set[str]] = {}
    for fn, lines in _disassembly(obj_path).items():
        graph[fn] = {m.group(1) for ln in lines if (m := re.search(r"R_AIE_\d+\s+([A-Za-z_]\w*)$", ln))
                     and not m.group(1).startswith(".") and m.group(1) != fn}
    for fn, more in (calls or {}).items():
        graph[fn] = graph.get(fn, set()) | more
    memo: dict[str, int] = {}

    def depth(fn: str, seen: frozenset[str]) -> int:
        if fn in memo:
            return memo[fn]
        if fn in seen:  # recursion: count the frame once
            return own.get(fn, _UNKNOWN_FRAME)
        d = own.get(fn, _UNKNOWN_FRAME) + max((depth(c, seen | {fn}) for c in graph.get(fn, ())), default=0)
        memo[fn] = d
        return d

    return {fn: depth(fn, frozenset()) for fn in set(own) | set(graph)}


def core_frames(design_dir: str) -> dict[str, int]:
    """'col,row' -> the stack frame of each core's own function (aiecc's control code: ``paddxm [sp], #N`` at its
    start), from the per-core objects aiecc keeps in design.mlir.prj ({} when it kept none)."""
    objs = sorted(glob.glob(os.path.join(design_dir, "design.mlir.prj", "objects_main_core_*", "*.o")))
    if not objs:
        return {}
    out = subprocess.run([OBJDUMP, "-d", *objs], capture_output=True, text=True, check=True).stdout
    frames: dict[str, int] = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <core_(\d+)_(\d+)>:$", line)
        if m:
            cur = f"{m.group(1)},{m.group(2)}"
        elif cur is not None and cur not in frames and (f := re.search(r"paddxm\s+\[sp\],\s*#(0x[0-9a-f]+|\d+)", line)):
            frames[cur] = int(f.group(1), 0)
    return frames


def rounding_functions(obj_path: str) -> dict[str, tuple[bool, bool]]:
    """Function -> (rounds: has conversion / SRS instructions, sets_mode: writes crrnd), own code only."""
    return {fn: (any(_ROUNDING.search(ln) for ln in lines), any("crrnd" in ln and "mov" in ln for ln in lines))
            for fn, lines in _disassembly(obj_path).items()}


def unset_rounding(obj_path: str, entries: set[str]) -> dict[str, list[str]]:
    """Entry -> the functions of its call tree (``kernel.o``'s relocations) that round while neither they nor a
    function on the path from the entry set the rounding mode."""
    funcs = _disassembly(obj_path)
    rnd = rounding_functions(obj_path)
    calls = {fn: {m.group(1) for ln in lines if (m := re.search(r"R_AIE_\d+\s+([\w.$]+)$", ln)) and m.group(1) in funcs}
             for fn, lines in funcs.items()}
    out: dict[str, list[str]] = {}
    for e in sorted(entries & set(funcs)):
        lax, seen, todo = set(), set(), [e]
        while todo:  # (functions reached with the mode unset so far)
            f = todo.pop()
            if f in seen:
                continue
            seen.add(f)
            rounds, sets = rnd.get(f, (False, False))
            if sets:
                continue
            if rounds:
                lax.add(f)
            todo += calls.get(f, ())
        if lax:
            out[e] = sorted(lax)
    return out


def library_arithmetic(obj_path: str) -> dict[str, set[str]]:
    """Function -> the compiler-rt arithmetic helpers it calls, own code only."""
    out: dict[str, set[str]] = {}
    for fn, lines in _disassembly(obj_path).items():
        calls = {m.group(1) for ln in lines if (m := re.search(r"R_AIE_\d+\s+(__\w+)$", ln)) and _LIBCALL.match(m.group(1))}
        if calls:
            out[fn] = calls
    return out


def _walk(op: object) -> Iterator[object]:
    op = getattr(op, "operation", op)
    yield op
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield from _walk(inner)


def _design(mlir_text: str) -> tuple[dict[str, list[str]], list[tuple[str, int, set[str]]]]:
    """(kernel declarations: name -> arg types, cores: (tile, stack_size, called kernels))."""
    from aie import ir
    from aie.extras.context import mlir_mod_ctx

    decls: dict[str, list[str]] = {}
    cores: list[tuple[str, int, set[str]]] = []
    with mlir_mod_ctx() as ctx:
        module = ir.Module.parse(mlir_text, ctx.context)
        for op in _walk(module.operation):
            if op.name == "func.func" and not list(op.regions[0].blocks):
                fty = ir.FunctionType(ir.TypeAttr(op.attributes["function_type"]).value)
                decls[ir.StringAttr(op.attributes["sym_name"]).value] = (
                    [str(t) for t in fty.inputs] + [f"-> {t}" for t in fty.results])
            elif op.name == "aie.core":
                stack = ir.IntegerAttr(op.attributes["stack_size"]).value if "stack_size" in op.attributes else 1024
                tile = op.operands[0].owner.attributes
                where = (f"{ir.IntegerAttr(tile['col']).value},{ir.IntegerAttr(tile['row']).value}"
                         if "col" in tile and "row" in tile else "unplaced")
                callees = {ir.FlatSymbolRefAttr(o.attributes["callee"]).value for o in _walk(op) if o.name == "func.call"}
                cores.append((where, stack, callees))
    return decls, cores


def _c_signature(source: str, name: str) -> tuple[str, list[str]] | None:
    """(return type, parameter texts) of the definition of ``name`` in the kernel source."""
    m = re.search(r"([A-Za-z_][\w:<>]*)\s*[\*&]?\s+" + re.escape(name) + r"\s*\(", source)
    if not m:
        return None
    i = m.end()
    depth, j = 1, i
    while depth and j < len(source):
        depth += {"(": 1, ")": -1}.get(source[j], 0)
        j += 1
    params, level, cur = [], 0, ""
    for ch in source[i:j - 1]:
        level += {"<": 1, ">": -1, "(": 1, ")": -1}.get(ch, 0)
        if ch == "," and level == 0:
            params.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip() and cur.strip() != "void":
        params.append(cur.strip())
    return m.group(1), params


def _check_signatures(source: str, decls: dict[str, list[str]]) -> list[str]:
    errors = []
    for name, types in decls.items():
        sig = _c_signature(source, name)
        if sig is None:
            continue  # defined elsewhere (e.g. a library object)
        ret, params = sig
        if any(t.startswith("-> ") for t in types) or ret != "void":
            errors.append(f"{name}: kernel entry points must return void (source: {ret})")
            continue
        if len(params) != len(types):
            errors.append(f"{name}: {len(params)} parameters in the source, {len(types)} in the design ({types})")
            continue
        for k, (p, t) in enumerate(zip(params, types)):
            is_ptr = "*" in p or "[" in p
            if t.startswith("memref") != is_ptr:
                errors.append(f"{name} arg {k}: design {t} vs source '{p}'")
            elif not is_ptr:
                ctype = p.replace("const", "").split()[0] if p.split() else ""
                want = _MLIR_BYTES.get(t)
                got = _SCALAR_BYTES.get(ctype)
                if want and got and want != got:
                    errors.append(f"{name} arg {k}: design {t} ({want} B) vs source '{p}' ({got} B)")
    return errors


def check(*, source: str | None, mlir_text: str, obj_path: str | None, label: str) -> None:
    """Run the checks for one compiled design (``obj_path``: kernel.o, with kernel.su next to it)."""
    if not enabled():
        return
    decls, cores = _design(mlir_text)
    errors = _check_signatures(source, decls) if source else []
    su = obj_path[: -len(".o")] + ".su" if obj_path else None
    if obj_path and os.path.exists(obj_path) and su and os.path.exists(su):
        usage = stack_usage(obj_path, su)
        frames = core_frames(os.path.dirname(obj_path))
        for tile, stack, callees in cores:
            own = frames.get(tile, CORE_FRAME) + MAIN_INIT_FRAME
            need = max((usage.get(c, 0) for c in callees), default=0) + own
            if need > stack:
                deep = max(callees, key=lambda c: usage.get(c, 0))
                errors.append(f"core ({tile}): {deep} and the core's own frame ({own} B) need {need} B of stack, "
                              f"the core has stack_size {stack}")
        called = set().union(*(c for _, _, c in cores)) if cores else set()
        lax = unset_rounding(obj_path, called)
        if lax:
            sys.stderr.write(f"{label}: kernels that round without setting the rounding mode: "
                             + "; ".join(f"{e} ({', '.join(f for f in fs if f != e) or 'itself'})" if fs != [e] else e
                                         for e, fs in lax.items()) + "\n")
        lib = library_arithmetic(obj_path)
        if lib:
            sys.stderr.write(f"{label}: scalar arithmetic through runtime library calls (slow, program memory): "
                             + "; ".join(f"{fn}: {', '.join(sorted(c))}" for fn, c in sorted(lib.items())) + "\n")
    if errors:
        raise DesignCheckError(f"{label}: " + "; ".join(errors))
