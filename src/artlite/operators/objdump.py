"""The disassembly of AIE2P objects (``llvm-objdump``), parsed once per object into what the build-time checks, the code
pager's check and the neighbour-lock census read: per function its relocations (callees and data), the lock ids it
acquires, its jumps, whether its code rounds or sets the rounding mode, and its stack frame.

Instructions that follow a label starting with "." (a section's) belong to the function before it.
"""

import functools
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import toolchain

_HEADER = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
_FILE = re.compile(r"^(\S+):\s+file format", re.MULTILINE)
_RELOC = re.compile(r"R_AIE_\d+\s+(\S+)$")
_ACQUIRE = re.compile(r"\bacq(?:\.\w+)?\s+#(0x[0-9a-f]+|\d+)")
_JUMP = re.compile(r"\b(?:j|jl|jz|jnz)\s")
_ROUNDING = re.compile(r"\b(vst\.conv|vconv\.bf16|vconv\.bfp16|vsrs|vst\.srs|vst\.push\.conv|vst\.flush\.conv)")
_FRAME = re.compile(r"paddxm\s+\[sp\],\s*#(0x[0-9a-f]+|\d+)")
_ARGS = 16384  # characters of object paths per llvm-objdump (Windows' command line: 32767 at most)


@dataclass(frozen=True)
class Function:
    """What a function's disassembly says."""

    relocs: tuple[str, ...]  # the symbols its relocations name (calls, data), in order
    acquires: frozenset[int]  # the immediates of its ``acq`` instructions (lock ids)
    jumps: tuple[str, ...]  # its jump instructions' lines
    rounds: bool  # it has vector conversions or SRS stores
    sets_rounding: bool  # it writes the rounding mode register
    frame: int | None  # the bytes its first ``paddxm [sp], #N`` reserves (its stack frame)


def parse(text: str) -> dict[str, Function]:
    """The functions of one object's ``llvm-objdump -d -r`` text."""
    lines: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            if not m.group(1).startswith("."):
                cur = m.group(1)
                lines.setdefault(cur, [])
            continue
        if cur is not None:
            lines[cur].append(line)
    out = {}
    for name, body in lines.items():
        frame = next((int(m.group(1), 0) for ln in body if (m := _FRAME.search(ln))), None)
        out[name] = Function(
            relocs=tuple(m.group(1) for ln in body if (m := _RELOC.search(ln))),
            acquires=frozenset(int(m.group(1), 0) for ln in body for m in _ACQUIRE.finditer(ln)),
            jumps=tuple(ln.strip() for ln in body if _JUMP.search(ln)),
            rounds=any(_ROUNDING.search(ln) for ln in body),
            sets_rounding=any("crrnd" in ln and "mov" in ln for ln in body),
            frame=frame)
    return out


def _run(paths: list[str], cwd: str | None) -> str:
    return subprocess.run([toolchain.peano_tool("llvm-objdump"), "-d", "-r", "--no-show-raw-insn", "--triple=aie2p",
                           *paths], cwd=cwd, capture_output=True, text=True, check=True).stdout


@functools.lru_cache(maxsize=4)
def _functions(path: str, mtime_ns: int, size: int) -> dict[str, Function]:
    return parse(_run([path], None))


def functions(path: str) -> dict[str, Function]:
    """The functions of the object or ELF ``path`` (parsed once while the file is unchanged: a build's checks, the
    stack's and the rounding's, read one parse)."""
    st = os.stat(path)
    return _functions(path, st.st_mtime_ns, st.st_size)


def functions_of(paths: list[str], *, cwd: str) -> dict[str, dict[str, Function]]:
    """Each of ``paths``' functions (relative to ``cwd``, without spaces: the file heads of the output name them): one
    llvm-objdump for each chunk of the paths that fits a command line."""
    out: dict[str, dict[str, Function]] = {}
    parts: list[list[str]] = [[]]
    size = 0
    for p in paths:
        if parts[-1] and size + len(p) >= _ARGS:
            parts.append([])
            size = 0
        parts[-1].append(p)
        size += len(p) + 1
    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        for text in ex.map(lambda part: _run(part, cwd), [p for p in parts if p]):
            heads = list(_FILE.finditer(text))
            for m, n in zip(heads, heads[1:] + [None]):
                out[m[1]] = parse(text[m.end():n.start() if n else len(text)])
    return out
