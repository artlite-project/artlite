"""Builds: a rendered design (``BuildSpec``) into its cache directory, its artifacts published whole.

``build`` compiles the kernels (Peano: ``kernel.cc`` -> ``kernel.o``) and the design (``aiecc``: ``design.mlir`` ->
``main.pdi`` + ``insts.bin``, or a fused design's full ELF) into ``CACHE_DIR/<kind>_<key>/``, ``BuildSpec.cache_key`` the
only key. Builds of one key serialize on a lock file (the later ones find the cache); distinct keys run in parallel.

An artifact a later build takes for cached is never a partial one:

- a single file a tool writes (``kernel.o`` with its ``kernel.su``, the precompiled header) is written under a temporary
  name and renamed whole once the tool succeeded;
- aiecc writes its outputs and its project directory in place: ``IN_PROGRESS`` exists in the directory from before it runs
  until it and its checks succeeded, and a directory that has it is built again (a build that failed, or one killed, leaves
  it).
"""

import glob
import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version

from .. import portable, profiling
from . import checks, toolchain
from .toolchain import AIE_TARGET, CACHE_DIR, CORE_LLC_FLAGS, INCLUDE_DIR, OPERATORS_H

IN_PROGRESS = ".building"


class BuildKind(StrEnum):
    RTC = "rtc"  # an ArtcOperator: main.pdi + insts.bin
    FUSED = "fused"  # a FusedOperator or Library: a full ELF


def profile_key(cfg: profiling.ProfileConfig | None) -> tuple | None:
    """What a profiling configuration changes in a design (``BuildSpec.profile_key``: part of the cache key)."""
    return None if cfg is None else cfg.cache_key()


@dataclass(frozen=True)
class BuildSpec:
    """A rendered design: everything a build needs, as plain data (picklable, hashable), so that
    rendering (Python, aie.iron: one thread) and building (``build``: subprocesses, any thread or
    process) can happen apart (``artlite.parallel_compile``).

    kind: the artifacts the build makes (``BuildKind``);
    label: the operator class or the fused ELF's name (checks, messages; a fused ELF's file name);
    source: the kernel translation unit (``kernel_source.translation_unit``), None without kernels;
    mlir: the design, its kernel names resolved (``KernelSource.rename``)."""

    kind: BuildKind
    label: str
    source: str | None
    mlir: str
    clang_flags: tuple[str, ...] = ()
    profile_key: tuple | None = None  # rtc: the profiling configuration the design was rendered with
    expand_load_pdis: bool = False  # fused
    aiecc_flags: tuple[str, ...] = ()  # ``ArtcOperator.AIECC_FLAGS``
    indirect_calls: tuple[tuple[str, str], ...] = ()  # ``ArtcOperator.INDIRECT_CALLS`` (a check's: not in the key)

    @property
    def clang_cmd(self) -> list[str]:
        return [*toolchain.clang_cmd_common(), f"--target={AIE_TARGET}", *self.clang_flags]

    @property
    def outputs(self) -> tuple[str, ...]:
        """The files of the build directory the build makes."""
        return ("main.pdi", "insts.bin") if self.kind is BuildKind.RTC else (f"{self.label}.elf",)

    def cache_key(self, pch_path: str) -> str:
        """The build's key: everything its artifacts depend on (``pch_path``: the precompiled header, a key of the
        compile command and the headers)."""
        if self.kind is BuildKind.FUSED:
            return toolchain.cache_key(self.source, self.mlir, pch_path, version("mlir_aie"), version("llvm_aie"),
                                       self.expand_load_pdis, CORE_LLC_FLAGS, *self.clang_flags, *self.aiecc_flags)
        return toolchain.cache_key(self.source, *self.clang_flags, self.mlir, pch_path, self.profile_key,
                                   CORE_LLC_FLAGS, *self.aiecc_flags)


def buildable(spec: BuildSpec) -> bool:
    """Whether this process builds ``spec`` as rendered: the build takes the profiling configuration (counters, trace
    lowering, intermediates) from its process, so a spec of another one (a planning subprocess's) only where they
    agree."""
    return spec.kind is not BuildKind.RTC or spec.profile_key == profile_key(profiling.get_config())


def _tmp_name() -> str:
    return f"{os.getpid()}_{threading.get_ident()}"


def compile_pch(*, clang_cmd: list[str]) -> str:
    """The precompiled header of ``operators.h`` for ``clang_cmd`` (built once per key and header contents)."""
    operators_pch_dir = os.path.join(CACHE_DIR, "operators")
    os.makedirs(operators_pch_dir, exist_ok=True)

    # -mllvm codegen args are rejected by the c++-header action; they don't
    # affect the PCH contents, so drop them (only) from the PCH compile.
    # Note: multiple -aie-* options may follow a single -mllvm, so strip
    # every -aie-* token, not just the one directly after -mllvm.
    pch_cmd: list[str] = []
    for arg in clang_cmd:
        if arg == "-mllvm" or arg.startswith("-aie-"):
            continue
        pch_cmd.append(arg)

    aiecc_mtime = os.path.getmtime(toolchain.aiecc())
    clangpp_mtime = os.path.getmtime(toolchain.clangpp())
    # operators.h and the headers it includes (kernel_lib.h): contents and modification times
    headers = sorted(glob.glob(os.path.join(INCLUDE_DIR, "*.h")))
    header_keys = []
    for path in headers:
        with open(path, "rb") as f:
            header_keys += [os.path.basename(path), f.read(), os.path.getmtime(path)]
    pch_hash = toolchain.cache_key(
        *clang_cmd,
        *header_keys,
        version("mlir_aie"),
        version("llvm_aie"),
        aiecc_mtime,
        clangpp_mtime,
    )
    operators_pch_path = os.path.join(operators_pch_dir, f"{pch_hash}.pch")
    # Lock around the exists-check AND the write: a fast-path read of a
    # partially-written .pch from a parallel process crashes clang.
    with portable.file_lock(operators_pch_path + ".lock"):
        if not os.path.exists(operators_pch_path):
            tmp = os.path.join(operators_pch_dir, f"{pch_hash}.{_tmp_name()}.tmp.pch")
            try:
                toolchain.run([*pch_cmd, "-x", "c++-header", OPERATORS_H, "-o", tmp])
                os.replace(tmp, operators_pch_path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    return operators_pch_path


def compile_kernel(
    *,
    operators_pch_path: str,
    work_dir: str,
    source_code: str,
    clang_cmd: list[str],
    ignore_cache: bool = False,
    measure: bool = True,
) -> None:
    """Compile kernel source to kernel.o in *work_dir*, with its kernel.su (per-function stack frames: the stack check,
    ``checks``); ``measure``: the ARTLITE_MEASURE_II report."""

    src_path = os.path.join(work_dir, "kernel.cc")

    with open(src_path, "w") as f:
        f.write(source_code)

    if measure and os.environ.get("ARTLITE_MEASURE_II", "0") == "1":
        from artlite import measure_ii

        measure_ii.measure(src_path, clang_cmd=clang_cmd, verbose=True)

    obj_path = os.path.join(work_dir, "kernel.o")
    if not ignore_cache and os.path.exists(obj_path):
        return

    # kernel.su is named after the object: both under one temporary name, the .su first (kernel.o is what makes the
    # compile cached)
    stem = os.path.join(work_dir, f"kernel.{_tmp_name()}.tmp")
    try:
        toolchain.run([*clang_cmd, "-fstack-usage", "-include-pch", operators_pch_path, "-c", src_path,
                       "-o", stem + ".o"])
        if os.path.exists(stem + ".su"):
            os.replace(stem + ".su", os.path.join(work_dir, "kernel.su"))
        os.replace(stem + ".o", obj_path)
    finally:
        for ext in (".o", ".su"):
            if os.path.exists(stem + ext):
                os.remove(stem + ext)

    if os.environ.get("ARTLITE_GENERATE_ASM", "0") == "1":
        asm_path = f"{src_path}.s"
        toolchain.run([*clang_cmd, "-S", src_path, "-o", asm_path])


def _kernel_and_checks(spec: BuildSpec, *, work_dir: str, pch_path: str, ignore_cache: bool, measure: bool) -> None:
    """kernel.o of ``spec`` in ``work_dir`` (when it has kernels), then the checks of ``operators/checks.py``."""
    if spec.source:
        compile_kernel(operators_pch_path=pch_path, clang_cmd=spec.clang_cmd, work_dir=work_dir,
                       source_code=spec.source, ignore_cache=ignore_cache, measure=measure)
    checks.check(source=spec.source, mlir_text=spec.mlir, label=spec.label,
                 obj_path=os.path.join(work_dir, "kernel.o") if spec.source else None, indirect=spec.indirect_calls)


@contextmanager
def _in_progress(work_dir: str) -> Iterator[None]:
    """``IN_PROGRESS`` in ``work_dir`` while the body runs; removed only when it succeeds."""
    with open(os.path.join(work_dir, IN_PROGRESS), "w"):
        pass
    yield
    os.remove(os.path.join(work_dir, IN_PROGRESS))


def _write_design(spec: BuildSpec, mlir_path: str, work_dir: str) -> None:
    """design.mlir of the build (an rtc one's with the profiling transforms applied)."""
    if spec.kind is BuildKind.FUSED:
        with open(mlir_path, "w") as f:
            f.write(spec.mlir)
        return
    profiling_cfg = profiling.get_config()
    final_text = spec.mlir
    if profiling_cfg is not None and profiling_cfg.mode == "counters":
        final_text, _ = profiling.inject_perf_counters(spec.mlir, profiling_cfg.counter_specs, work_dir)
    with open(mlir_path, "w") as f:
        f.write(final_text)
    if profiling_cfg is not None and profiling_cfg.mode == "trace":
        pre_lowered = profiling.pre_lower_trace(mlir_path, profiling_cfg, work_dir)
        if pre_lowered != mlir_path:
            shutil.copy(pre_lowered, mlir_path)


def _aiecc_cmd(spec: BuildSpec, *, aiecc_jobs: int | None, work_dir: str, mlir_path: str) -> list[str]:
    # --unified: every core of a device lowered once (default: a clone of the design per core); the same outputs, aiecc
    # -55% (the Llama prefill library 52 -> 21 s, the SD 1.5 UNet program 22 -> 8 s)
    cmd = [toolchain.aiecc(), *([f"-j{aiecc_jobs}"] if aiecc_jobs else []), "--unified",
           f"--peano={toolchain.peano_dir()}", *(f"-Xllc={flag}" for flag in CORE_LLC_FLAGS), *spec.aiecc_flags]
    if spec.kind is BuildKind.FUSED:
        elf_path = os.path.join(work_dir, spec.outputs[0])
        return [*cmd, *(["--expand-load-pdis"] if spec.expand_load_pdis else []), "--get-full-elf",
                "--get-scratchpad-parameters", "--full-elf-name", os.path.abspath(elf_path),
                os.path.abspath(mlir_path)]
    cmd += ["--get-pdi", "--get-npu-insts", "--pdi-name=main.pdi", "--npu-insts-name=insts.bin"]
    if profiling.get_config() is not None:
        # Persist design.mlir.prj intermediates (notably input_with_addresses.mlir) for trace parsing.
        cmd.append("--dump-intermediates")
    return [*cmd, mlir_path]


def build(spec: BuildSpec, *, aiecc_jobs: int | None = None, measure: bool = True) -> str:
    """Build ``spec`` into its cache directory (skipped when cached) and return the directory.
    aiecc_jobs: aiecc's compile threads (default: its own choice);
    measure: print the ARTLITE_MEASURE_II report."""
    ignore_cache = os.environ.get("ARTLITE_IGNORE_CACHE", "0") == "1"
    clang_cmd = spec.clang_cmd
    pch_path = compile_pch(clang_cmd=clang_cmd)
    if not buildable(spec):
        raise RuntimeError(f"{spec.label}: rendered with profiling {spec.profile_key}, this process has "
                           f"{profile_key(profiling.get_config())}")
    key = spec.cache_key(pch_path)
    name = f"{spec.kind.value}_{key}"
    work_dir = os.path.join(CACHE_DIR, name)
    os.makedirs(work_dir, exist_ok=True)
    mlir_path = os.path.join(work_dir, "design.mlir")

    # Under a per-key lock: concurrent compilation of an identical operator (same cache dir) from parallel
    # tests/processes otherwise produces corrupted kernel.o / design.mlir.prj artifacts.
    with portable.file_lock(os.path.join(CACHE_DIR, f"{name}.lock")):
        redo = (ignore_cache or os.path.exists(os.path.join(work_dir, IN_PROGRESS))
                or not all(os.path.exists(os.path.join(work_dir, f)) for f in spec.outputs))
        if redo:
            with _in_progress(work_dir):
                _write_design(spec, mlir_path, work_dir)
                cmd = _aiecc_cmd(spec, aiecc_jobs=aiecc_jobs, work_dir=work_dir, mlir_path=mlir_path)
                toolchain.run_aiecc(cmd, cwd=work_dir, link_files=lambda: _kernel_and_checks(
                    spec, work_dir=work_dir, pch_path=pch_path, ignore_cache=ignore_cache, measure=measure))
        else:
            if spec.kind is BuildKind.FUSED or not os.path.exists(mlir_path):
                _write_design(spec, mlir_path, work_dir)
            if spec.source:
                compile_kernel(operators_pch_path=pch_path, clang_cmd=clang_cmd, work_dir=work_dir,
                               source_code=spec.source, ignore_cache=ignore_cache, measure=measure)
    return work_dir
