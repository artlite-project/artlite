"""The operator base class and its argument specs: an operator renders its kernel source and its array design
(``build_spec``); ``build`` builds them."""

from __future__ import annotations

import math
import os
import sys
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, Self

import numpy as np

from .. import profiling
from .build import BuildKind, BuildSpec, profile_key
from .kernel_source import KernelSource, translation_unit
from .toolchain import PACKAGE_DIR

if TYPE_CHECKING:
    from aie.ir import Module
    from numpy.typing import DTypeLike

    from .. import Device, Program
    from .._native import Buffer


class ArgDirection(StrEnum):
    IN = "in"
    OUT = "out"


class _ArgSpec(NamedTuple):
    direction: ArgDirection
    shape: tuple[int, ...]
    dtype: np.dtype


class ArgSpec(_ArgSpec):
    """Describes one buffer argument for an RTC operator.

    The ``shape`` is the *logical* shape used for allocation sizing.
    At dispatch time only the total element count (``math.prod(shape)``)
    and ``dtype`` (any numpy dtype-like, kept as ``np.dtype``) are validated —
    callers may pass arrays with any shape as long as size and dtype match.
    """

    __slots__ = ()

    def __new__(cls, direction: ArgDirection | str, shape: tuple[int, ...], dtype: DTypeLike) -> Self:
        return super().__new__(cls, direction, shape, np.dtype(dtype))

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


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
        buf = artlite.from_numpy(x, op_a.device)
        tmp = artlite.empty(..., device=op_a.device)
        out = artlite.empty(..., device=op_a.device)
        op_a.dispatch(buf, tmp)
        op_b.dispatch(tmp, out)
        result = out.numpy()
    """

    AIECC_FLAGS: tuple[str, ...] = ()  # (aiecc options of the design)
    # (calls the stack check cannot see in the object: (pattern of a caller's linked name, template of its callees' names
    # expanded with the caller's match), ``checks.stack_usage``)
    INDIRECT_CALLS: tuple[tuple[str, str], ...] = ()

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

        return BuildSpec(kind=BuildKind.RTC, label=type(self).__name__, source=source_code, mlir=self.mlir_text(),
                         clang_flags=tuple(self.clang_flags), profile_key=profile_key(profiling_cfg),
                         aiecc_flags=self.AIECC_FLAGS, indirect_calls=self.INDIRECT_CALLS)

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

    def run(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Compile (if needed), dispatch, and return output arrays.

        Input arrays are copied to device buffers, output buffers are auto-allocated
        from ``get_arg_spec()``, and results are read back to host.
        See ``Program.run`` for validation semantics (size + dtype).

        Args:
            *inputs: numpy arrays for each "in" arg in the arg spec.

        Returns:
            Single output array, or tuple of arrays if multiple outputs.
        """
        if self._program is None:
            self.compile()
        return self._program.run(*inputs)
