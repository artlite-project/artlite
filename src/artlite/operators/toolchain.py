"""The build toolchain: where Peano (llvm-aie) and aiecc (mlir-aie) are, the command lines of clang++ and aiecc, and the
subprocess runners of the builds (``build``).

Nothing here needs the toolchain at import: its locations resolve at the first build that asks for them (cached) and a
missing tool raises ``ToolchainError`` there, so host-only code (the kernel source splitter, the check parsers, the
import sweep of the tests) runs without it.
"""

import functools
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from importlib.util import find_spec

from .. import backend, portable

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # (the artlite package)
INCLUDE_DIR = os.path.join(PACKAGE_DIR, "include")  # the kernels' headers
OPERATORS_H = os.path.join(INCLUDE_DIR, "operators.h")

CACHE_DIR = os.environ.get(
    "ARTLITE_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "artlite"),
)

# AIE target triple for Peano cross-compiler: aie2p = AIE-ML v2 (NPU2: Strix Point, Strix Halo, Krackan)
AIE_TARGET = "aie2p-none-unknown-elf"

# llc flags for the cores' own code (the worker loops aiecc generates from a design; the kernels are clang++'s): a loop
# without a known minimum trip count (an object fifo's skip loop: acquire, release) as a branch loop, not a
# zero-overhead loop padded with full NOP bundles (~32 instead of ~180 B each: the Gemma decode overlay's attention
# cores' loop 1840 -> 1408 B, the main cores' 2400 -> 2192). aiecc passes them on (-Xllc).
CORE_LLC_FLAGS = ("--enable-aie-zol-without-minitercount=false",)


class ToolchainError(RuntimeError):
    """A tool of the build toolchain is missing."""


def _package_dir(wheel: str) -> str:
    spec = find_spec(wheel)
    if spec is None or not spec.submodule_search_locations:
        raise ToolchainError(f"{wheel} is not installed (the build toolchain: see docs/install.md)")
    return spec.submodule_search_locations[0]


@functools.cache
def peano_dir() -> str:
    """The llvm-aie wheel's directory (Peano)."""
    return _package_dir("llvm-aie")


@functools.cache
def mlir_aie_dir() -> str:
    """The mlir-aie wheel's directory."""
    return _package_dir("mlir_aie")


def peano_tool(name: str) -> str:
    """A Peano binary's path."""
    return os.path.join(peano_dir(), "bin", name + portable.EXE)


def clangpp() -> str:
    """Peano's clang++."""
    return peano_tool("clang++")


@functools.cache
def aiecc() -> str:
    """The aiecc executable: on PATH, then the interpreter's scripts directory (an unactivated venv, an IDE's test
    discovery, runs its python without it)."""
    path = shutil.which("aiecc") or shutil.which("aiecc", path=sysconfig.get_path("scripts"))
    if not path:
        raise ToolchainError("aiecc not found in PATH. Please check your installation.")
    return path


def clang_cmd_common() -> list[str]:
    """The kernels' clang++ command without a target."""
    mlir_aie = mlir_aie_dir()
    return [
        clangpp(),
        "-O2",
        "-std=c++20",
        "-Wno-unknown-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DNDEBUG",
        "-I" + INCLUDE_DIR,
        "-I" + os.path.join(mlir_aie, "include"),
        "-I" + os.path.join(mlir_aie, "aie_runtime_lib", "AIE2P"),
    ]


def _tool_env() -> dict[str, str] | None:
    """The environment of the build tools (None: this process's). aiecc packs a full ELF with aiebu-asm, which
    mlir-aie's Windows wheels lack: XRT's joins PATH."""
    if not portable.WINDOWS or shutil.which("aiebu-asm") or not (tool := backend.xrt_tool("aiebu-asm")):
        return None
    return {**os.environ, "PATH": os.path.dirname(tool) + os.pathsep + os.environ.get("PATH", "")}


def run(cmd: list[str], cwd: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a build tool; a nonzero exit raises with its stderr."""
    if os.environ.get("ARTLITE_LOG_COMMANDS", "0") == "1":
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=capture, check=False, env=_tool_env())
    if r.returncode != 0:
        stderr = r.stderr if capture else ""
        raise RuntimeError(
            f"Command failed (rc={r.returncode}): {' '.join(cmd)}\n{stderr}"
        )
    return r


def run_aiecc(cmd: list[str], *, cwd: str, link_files: Callable[[], None]) -> None:
    """Run aiecc while ``link_files`` builds what its cores link (kernel.o, then its checks): with --await-link-files
    aiecc lowers the design meanwhile and reads no linked file until a line arrives on its stdin (the Llama prefill
    library builds in 26 s, 37 s with kernel.o built first). Its output goes to a file: a pipe nobody reads
    meanwhile would stall it."""
    cmd = [*cmd, "--await-link-files"]
    if os.environ.get("ARTLITE_LOG_COMMANDS", "0") == "1":
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    with tempfile.TemporaryFile("w+") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=log, stderr=log, text=True, env=_tool_env())
        try:
            link_files()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        proc.communicate("\n")
        if proc.returncode != 0:
            log.seek(0)
            raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{log.read()}")


def cache_key(*args: object) -> str:
    """A cache key of ``args`` (their repr)."""
    return hashlib.sha256(repr(args).encode()).hexdigest()[:16]
