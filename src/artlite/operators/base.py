from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, NamedTuple, Self

import torch
from jinja2 import Environment, StrictUndefined

from .. import backend, portable, profiling
from . import kmath
from .kernel_source import KernelSource, translation_unit

if TYPE_CHECKING:
    from aie.ir import Module

    from .. import Device, Program
    from .._native import Buffer

# Toolchain paths

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # (the artlite package)
INCLUDE_DIR = os.path.join(PACKAGE_DIR, "include")  # the kernels' headers
OPERATORS_H = os.path.join(INCLUDE_DIR, "operators.h")

CACHE_DIR = os.environ.get(
    "ARTLITE_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "artlite"),
)

PEANO_DIR = find_spec("llvm-aie").submodule_search_locations[0]
MLIR_AIE_DIR = find_spec("mlir_aie").submodule_search_locations[0]
MLIR_AIE_INCLUDE = os.path.join(MLIR_AIE_DIR, "include")


def peano_tool(name: str) -> str:
    """A Peano binary's path."""
    return os.path.join(PEANO_DIR, "bin", name + portable.EXE)


CLANGPP = peano_tool("clang++")
# PATH, then the interpreter's scripts directory: an unactivated venv (an IDE's test discovery) runs its python without it
AIECC = shutil.which("aiecc") or shutil.which("aiecc", path=sysconfig.get_path("scripts"))
if not AIECC:
    sys.exit("ERROR: aiecc not found in PATH. Please check your installation.")

AIE_RUNTIME_LIB_INCLUDE = os.path.join(MLIR_AIE_DIR, "aie_runtime_lib", "AIE2P")

CLANG_CMD_COMMON = [
    CLANGPP,
    "-O2",
    "-std=c++20",
    "-Wno-unknown-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
    "-Wno-deprecated-declarations",
    "-D__AIE_API_AIE_ADF_HPP__",
    "-DNDEBUG",
    "-I" + INCLUDE_DIR,
    f"-I{MLIR_AIE_INCLUDE}",
    f"-I{AIE_RUNTIME_LIB_INCLUDE}",
]

# AIE target triple for Peano cross-compiler: aie2p = AIE-ML v2 (NPU2: Strix Point, Strix Halo, Krackan)
AIE_TARGET = "aie2p-none-unknown-elf"

# llc flags for the cores' own code (the worker loops aiecc generates from a design; the kernels are clang++'s): a loop
# without a known minimum trip count (an object fifo's skip loop: acquire, release) as a branch loop, not a zero-overhead
# loop padded with full NOP bundles (~32 instead of ~180 B each: the Gemma decode overlay's attention cores' loop 1840 ->
# 1408 B, the main cores' 2400 -> 2192). aiecc passes them on (-Xllc).
CORE_LLC_FLAGS = ("--enable-aie-zol-without-minitercount=false",)


def _tool_env() -> dict[str, str] | None:
    """The environment of the build tools (None: this process's). aiecc packs a full ELF with aiebu-asm, which mlir-aie's
    Windows wheels lack: XRT's joins PATH."""
    if not portable.WINDOWS or shutil.which("aiebu-asm") or not (tool := backend.xrt_tool("aiebu-asm")):
        return None
    return {**os.environ, "PATH": os.path.dirname(tool) + os.pathsep + os.environ.get("PATH", "")}


# Shared Jinja environment for all kernel template strings.
# trim_blocks removes the newline after a block tag; lstrip_blocks strips
# leading whitespace before a block tag. Both keep the generated C++ clean.
# StrictUndefined: a variable the render call does not pass is an error, not an empty string.
# km: the math functions generated per contract (kmath.py), in every template.
JINJA_ENV = Environment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
JINJA_ENV.globals["km"] = kmath


# Internal helpers


def _run(cmd: list[str], cwd: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    if os.environ.get("ARTLITE_LOG_COMMANDS", "0") == "1":
        print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=capture, check=False, env=_tool_env())
    if r.returncode != 0:
        stderr = r.stderr if capture else ""
        raise RuntimeError(
            f"Command failed (rc={r.returncode}): {' '.join(cmd)}\n{stderr}"
        )
    return r


def _run_aiecc(cmd: list[str], *, cwd: str, link_files: Callable[[], None]) -> None:
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


def _cache_key(*args: object) -> str:
    return hashlib.sha256(repr(args).encode()).hexdigest()[:16]


def profile_key(cfg: profiling.ProfileConfig | None) -> tuple | None:
    """What a profiling configuration changes in a design (``BuildSpec.profile_key``: part of the cache key): trace
    parameters shape the MLIR, counters are injected by the build."""
    if cfg is None:
        return None
    if cfg.mode == "trace":
        return "trace", cfg.trace_size, str(cfg.trace_events), cfg.trace_tiles, cfg.trace_lateral, cfg.trace_distribute
    return "counters", cfg.counter_specs


def buildable(spec: BuildSpec) -> bool:
    """Whether this process builds ``spec`` as rendered: the build takes the profiling configuration (counters, trace
    lowering, intermediates) from its process, so a spec of another one (a planning subprocess's) only where they
    agree."""
    return spec.kind != "rtc" or spec.profile_key == profile_key(profiling.get_config())


def _compile_pch(*, clang_cmd: list[str]) -> str:
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

    aiecc_mtime = os.path.getmtime(AIECC)
    clangpp_mtime = os.path.getmtime(CLANGPP)
    # operators.h and the headers it includes (kernel_lib.h): contents and modification times
    headers = sorted(glob.glob(os.path.join(INCLUDE_DIR, "*.h")))
    header_keys = []
    for path in headers:
        with open(path, "rb") as f:
            header_keys += [os.path.basename(path), f.read(), os.path.getmtime(path)]
    pch_hash = _cache_key(
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
            _run([*pch_cmd, "-x", "c++-header", OPERATORS_H, "-o", operators_pch_path])

    return operators_pch_path


def _compile_kernel(
    *,
    operators_pch_path: str,
    work_dir: str,
    source_code: str,
    clang_cmd: list[str],
    ignore_cache: bool = False,
    measure: bool = True,
) -> None:
    """Compile kernel source to kernel.o in *work_dir* (``measure``: the ARTLITE_MEASURE_II report)."""

    src_path = os.path.join(work_dir, "kernel.cc")

    with open(src_path, "w") as f:
        f.write(source_code)

    if measure and os.environ.get("ARTLITE_MEASURE_II", "0") == "1":
        from artlite import measure_ii

        measure_ii.measure(src_path, clang_cmd=clang_cmd, verbose=True)

    obj_path = os.path.join(work_dir, "kernel.o")
    if not ignore_cache and os.path.exists(obj_path):
        return

    # -fstack-usage: kernel.su (per-function frames) for the stack check (checks.py)
    _run(
        [*clang_cmd, "-fstack-usage", "-include-pch", operators_pch_path, "-c", src_path, "-o", obj_path]
    )

    if os.environ.get("ARTLITE_GENERATE_ASM", "0") == "1":
        asm_path = f"{src_path}.s"
        _run([*clang_cmd, "-S", src_path, "-o", asm_path])


class ArgDirection(StrEnum):
    IN = "in"
    OUT = "out"


class ArgSpec(NamedTuple):
    """Describes one buffer argument for an RTC operator.

    The ``shape`` is the *logical* shape used for allocation sizing.
    At dispatch time only the total element count (``math.prod(shape)``)
    and ``dtype`` are validated — callers may pass tensors with any
    shape as long as numel and dtype match.
    """

    direction: ArgDirection
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class BuildSpec:
    """A rendered design: everything a build needs, as plain data (picklable, hashable), so that
    rendering (Python, aie.iron: one thread) and building (``build``: subprocesses, any thread or
    process) can happen apart (``artlite.parallel_compile``).

    kind: "rtc" (an ArtcOperator: main.pdi + insts.bin) or "fused" (a FusedOperator: a full ELF);
    label: the operator class or the fused ELF's name (checks, messages; a fused ELF's file name);
    source: the kernel translation unit (``kernel_source.translation_unit``), None without kernels;
    mlir: the design, its kernel names resolved (``KernelSource.rename``)."""

    kind: str
    label: str
    source: str | None
    mlir: str
    clang_flags: tuple[str, ...] = ()
    profile_key: tuple | None = None  # rtc: the profiling configuration the design was rendered with
    expand_load_pdis: bool = False  # fused

    @property
    def clang_cmd(self) -> list[str]:
        return [*CLANG_CMD_COMMON, f"--target={AIE_TARGET}", *self.clang_flags]


def build(spec: BuildSpec, *, aiecc_jobs: int | None = None, measure: bool = True) -> str:
    """Build ``spec`` into its cache directory (skipped when cached) and return the directory.
    Builds of one cache key serialize on a lock file (the later ones find the cache); distinct
    keys run in parallel. aiecc_jobs: aiecc's compile threads (default: its own choice);
    measure: print the ARTLITE_MEASURE_II report."""
    ignore_cache = os.environ.get("ARTLITE_IGNORE_CACHE", "0") == "1"
    clang_cmd = spec.clang_cmd
    pch_path = _compile_pch(clang_cmd=clang_cmd)
    # --unified: every core of a device lowered once (default: a clone of the design per core); the same outputs, aiecc
    # -55% (the Llama prefill library 52 -> 21 s, the SD 1.5 UNet program 22 -> 8 s)
    aiecc_opts = [*([f"-j{aiecc_jobs}"] if aiecc_jobs else []), "--unified", f"--peano={PEANO_DIR}",
                  *(f"-Xllc={flag}" for flag in CORE_LLC_FLAGS)]
    if spec.kind == "fused":
        key = _cache_key(spec.source, spec.mlir, pch_path, version("mlir_aie"), version("llvm_aie"),
                         spec.expand_load_pdis, CORE_LLC_FLAGS)
        work_dir = os.path.join(CACHE_DIR, f"fused_{key}")
        os.makedirs(work_dir, exist_ok=True)
        elf_path = os.path.join(work_dir, f"{spec.label}.elf")
        with portable.file_lock(os.path.join(CACHE_DIR, f"fused_{key}.lock")):
            mlir_path = os.path.join(work_dir, "design.mlir")
            with open(mlir_path, "w") as f:
                f.write(spec.mlir)
            if not os.path.exists(elf_path) or ignore_cache:
                _run_aiecc([AIECC, *aiecc_opts, *(["--expand-load-pdis"] if spec.expand_load_pdis else []),
                            "--get-full-elf", "--get-scratchpad-parameters", "--full-elf-name",
                            os.path.abspath(elf_path), os.path.abspath(mlir_path)], cwd=work_dir,
                           link_files=lambda: _kernel_and_checks(spec, work_dir=work_dir, pch_path=pch_path,
                                                                 ignore_cache=ignore_cache, measure=measure))
            elif spec.source:
                _compile_kernel(operators_pch_path=pch_path, clang_cmd=clang_cmd, work_dir=work_dir,
                                source_code=spec.source, ignore_cache=ignore_cache, measure=measure)
        return work_dir

    assert spec.kind == "rtc", spec.kind
    profiling_cfg = profiling.get_config()
    if not buildable(spec):
        raise RuntimeError(f"{spec.label}: rendered with profiling {spec.profile_key}, this process has "
                           f"{profile_key(profiling_cfg)}")
    key = _cache_key(spec.source, *spec.clang_flags, spec.mlir, pch_path, spec.profile_key, CORE_LLC_FLAGS)
    work_dir = os.path.join(CACHE_DIR, f"rtc_{key}")
    os.makedirs(work_dir, exist_ok=True)
    pdi_path = os.path.join(work_dir, "main.pdi")
    insts_path = os.path.join(work_dir, "insts.bin")
    mlir_path = os.path.join(work_dir, "design.mlir")

    # Steps 1-3 under a per-key lock: concurrent compilation of an
    # identical operator (same cache dir) from parallel tests/processes
    # otherwise produces corrupted kernel.o / design.mlir.prj artifacts.
    with portable.file_lock(os.path.join(CACHE_DIR, f"rtc_{key}.lock")):
        # Step 1: write MLIR (with profiling transforms applied)
        if not os.path.exists(mlir_path) or ignore_cache:
            final_text = spec.mlir
            if profiling_cfg is not None and profiling_cfg.mode == "counters":
                final_text, _ = profiling.inject_perf_counters(spec.mlir, profiling_cfg.counter_specs, work_dir)
            with open(mlir_path, "w") as f:
                f.write(final_text)
            if profiling_cfg is not None and profiling_cfg.mode == "trace":
                pre_lowered = profiling.pre_lower_trace(mlir_path, profiling_cfg, work_dir)
                if pre_lowered != mlir_path:
                    shutil.copy(pre_lowered, mlir_path)

        # Step 2: compile MLIR -> PDI + insts, the C++ kernel -> kernel.o meanwhile
        if not (os.path.exists(pdi_path) and os.path.exists(insts_path)) or ignore_cache:
            aiecc_cmd = [AIECC, *aiecc_opts, "--get-pdi", "--get-npu-insts", "--pdi-name=main.pdi",
                         "--npu-insts-name=insts.bin"]
            if profiling_cfg is not None:
                # Persist design.mlir.prj intermediates (notably
                # input_with_addresses.mlir) for trace parsing.
                aiecc_cmd.append("--dump-intermediates")
            aiecc_cmd.append(mlir_path)
            _run_aiecc(aiecc_cmd, cwd=work_dir,
                       link_files=lambda: _kernel_and_checks(spec, work_dir=work_dir, pch_path=pch_path,
                                                             ignore_cache=ignore_cache, measure=measure))
        elif spec.source:
            _compile_kernel(operators_pch_path=pch_path, clang_cmd=clang_cmd, work_dir=work_dir,
                            source_code=spec.source, ignore_cache=ignore_cache, measure=measure)
    return work_dir


def _kernel_and_checks(spec: BuildSpec, *, work_dir: str, pch_path: str, ignore_cache: bool, measure: bool) -> None:
    """kernel.o of ``spec`` in ``work_dir`` (when it has kernels), then the checks of ``operators/checks.py``."""
    from . import checks

    if spec.source:
        _compile_kernel(operators_pch_path=pch_path, clang_cmd=spec.clang_cmd, work_dir=work_dir,
                        source_code=spec.source, ignore_cache=ignore_cache, measure=measure)
    checks.check(source=spec.source, mlir_text=spec.mlir, label=spec.label,
                 obj_path=os.path.join(work_dir, "kernel.o") if spec.source else None)


class ArtcOperator:
    """Base class for Artlite RTC operators.

    Subclasses override:
      - ``get_kernel_src()`` -> C++ source code string, its namespace-scope names under ``kernel_prefix(stem)``
      - ``get_mlir_artifact()``  -> MLIR module, its kernels declared under the same names
      - ``get_arg_spec()``       -> list of ArgSpec(direction, shape, dtype)

    The build names the kernels by their code (``kernel_source``): ``kernel_source()`` resolves the source,
    ``mlir_text()`` the design.

    Usage::

        # Simple (one-shot):
        op = MyOperator(size=1024).compile()
        output = op.run(input_tensor)

        # Pipeline (explicit device sharing, zero-copy between ops):
        op_a = OpA(...).compile()
        op_b = OpB(...).compile(device=op_a.device)
        buf = artlite.from_torch(x, op_a.device)
        tmp = artlite.empty(..., device=op_a.device)
        out = artlite.empty(..., device=op_a.device)
        op_a.dispatch(buf, tmp)
        op_b.dispatch(tmp, out)
        result = out.to_torch()
    """

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        self = super().__new__(cls)
        self._ctor = (args, kwargs)  # (the render manifest's key: ``render_cache``)
        return self

    def __init__(self, *, clang_flags: list[str] | None = None):
        self.clang_flags = list(clang_flags or [])
        self._program: Program | None = None

    @property
    def device(self) -> Device:
        """The Device backing this operator (available after compile)."""
        if self._program is None:
            raise RuntimeError("operator not compiled yet; call compile() first")
        self._program._ensure_loaded()
        return self._program._device

    def get_kernel_src(self) -> str | None:
        """Return C++ source code for the kernel.

        Override in subclasses.  Default: None (no external kernels).
        """
        return None

    def kernel_source(self) -> KernelSource | None:
        """The kernel source with its names resolved (rendered once per operator object)."""
        if "_kernel_source" not in self.__dict__:
            raw = self.get_kernel_src()
            self._kernel_source = None if raw is None else KernelSource(raw, label=self.kernel_label(),
                                                                        detail=self.kernel_detail())
        return self._kernel_source

    def kernel_label(self) -> str:
        """The operator's section header in a translation unit: its class and file."""
        mod = sys.modules[type(self).__module__].__file__ or ""
        top = os.path.dirname(PACKAGE_DIR)  # (artlite's and models' directory)
        where = os.path.relpath(mod, top) if mod.startswith((os.path.join(top, "artlite", ""),
                                                             os.path.join(top, "models", ""))) else os.path.basename(mod)
        return f"{type(self).__name__} ({where})"

    def kernel_detail(self) -> str:
        """What tells the operator's sections in a fused ELF's translation unit apart: its scalar constructor
        arguments."""
        def plain(v: object) -> str | None:
            if v is None or isinstance(v, bool | int):
                return repr(v)
            if isinstance(v, float):
                return f"{v:g}"
            return repr(v) if isinstance(v, str) and len(v) <= 24 else None
        args, kwargs = self.__dict__.get("_ctor", ((), {}))
        shown = [t for t in map(plain, args) if t is not None]
        return ", ".join(shown + [f"{k}={t}" for k, v in kwargs.items() if (t := plain(v)) is not None])

    def mlir_text(self) -> str:
        """The design's MLIR, its kernel names resolved (rendered once per operator object)."""
        if "_mlir_text" not in self.__dict__:
            text = str(self.get_mlir_artifact())
            ks = self.kernel_source()
            self._mlir_text = ks.rename(text) if ks is not None else text
        return self._mlir_text

    def get_mlir_artifact(self) -> Module:
        """Return MLIR module for this operator.

        Must be overridden in subclasses.
        """
        raise NotImplementedError

    def get_arg_spec(self) -> list[ArgSpec]:
        """Return list of ArgSpec declaring buffer directions, shapes, dtypes.

        Must be overridden in subclasses.
        """
        raise NotImplementedError

    def build_spec(self) -> BuildSpec:
        """Render the operator: its kernel source and MLIR design (``compile`` = this + ``build``)."""
        profiling_cfg = profiling.get_config()
        if profiling_cfg is not None and profiling_cfg.mode == "trace":
            profiling.install_iron_hook()

        ks = self.kernel_source()
        source_code = None if ks is None else translation_unit([ks])

        return BuildSpec(kind="rtc", label=type(self).__name__, source=source_code, mlir=self.mlir_text(),
                         clang_flags=tuple(self.clang_flags), profile_key=profile_key(profiling_cfg))

    def compile(self, device: Device | None = None) -> ArtcOperator:
        """Compile the operator and prepare for dispatch (cached on disk, the render too: ``render_cache``; several
        operators at once: ``artlite.parallel_compile.compile_all``).

        Args:
            device: Optional artlite.Device for shared buffer allocation.
                    If None, a Device is created on first use.

        Returns:
            self (for chaining).
        """
        from .. import parallel_compile

        parallel_compile.compile_all([self], device=device)
        return self

    def _mark_planned(self) -> ArtcOperator:
        """Planning mode (``parallel_compile.Precompiler``): nothing built, any use of the program raises."""
        from .. import parallel_compile

        self._program = parallel_compile.PLANNED
        return self

    def _attach(self, work_dir: str, device: Device | None = None) -> ArtcOperator:
        """Bind the built artifacts in ``work_dir`` (``build``) for dispatch."""
        import artlite

        pdi_path = os.path.join(work_dir, "main.pdi")
        insts_path = os.path.join(work_dir, "insts.bin")
        self._program = artlite.Program(
            pdi_path=pdi_path,
            insts_path=insts_path,
            arg_specs=self.get_arg_spec(),
            device=device,
        )
        profiling_cfg = profiling.get_config()
        if profiling_cfg is not None:
            with open(os.path.join(work_dir, "design.mlir")) as f:
                final_text = f.read()
            tiles_by_class = profiling.instrumented_tiles_of(
                final_text, profiling_cfg.counter_specs
            )
            self._program.profiling_attach(
                profiling_cfg,
                name=f"{type(self).__name__}_{os.path.basename(work_dir)[len('rtc_'):][:8]}",
                tiles_by_class=tiles_by_class,
                work_dir=work_dir,
            )
        # Do not load program here! Compilation is parallel, and attempt to load here
        # causes RuntimeError: shim_xdna: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed
        # due to multiple contexts being created in parallel
        return self

    def dispatch(self, *buffers: Buffer) -> None:
        """Dispatch pre-allocated device Buffers (no host sync).

        Args:
            *buffers: artlite.Buffer instances in MLIR sequence order.
        """
        if self._program is None:
            self.compile()
        self._program.dispatch(*buffers)

    def run(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Compile (if needed), dispatch, and return output tensors.

        Input tensors are synced to device, output buffers are auto-allocated
        from ``get_arg_spec()``, and results are read back to host.
        See ``Program.run`` for validation semantics (numel + dtype).

        Args:
            *inputs: torch tensors for each "in" arg in the arg spec.

        Returns:
            Single output tensor, or tuple of tensors if multiple outputs.
        """
        if self._program is None:
            self.compile()
        return self._program.run(*inputs)
