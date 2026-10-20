"""The common part of the fused ELFs (``FusedOperator``, ``Library``): a runlist of operators compiled into one full
ELF (``design.render``; cached on disk), and on the NPU the ELF's PDIs, its control program's scratchpad (runtime
parameters) and its hardware context, created at the first dispatch.
"""

import os
import sys
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, NamedTuple, Self

import numpy as np

from ... import txn
from ..._native import ParameterScratchpad
from ...elf_parser import CTRL_SCRATCHPAD, DataBd, ParsedElf, parse_elf, patch_control_code, pdi_sections
from ..base import XCLBIN_KERNEL, BuildSpec, _file_lock, pdi_xclbin
from ..design import COLS
from ..kernel_source import translation_unit
from . import design, layout
from .layout import BufferLayout, Runlist, distinct, scratch_segments

SPACES = ("input", "output", "scratch")  # (main's arguments: input, output, then a segment of scratch each)


class ShimBd(NamedTuple):
    """A shim DMA BD of the control program that transfers to or from a buffer space."""

    offset: int  # byte offset of the BD in the control program
    space: str
    start: int  # byte offset in the space where the transfer starts
    words: int  # the transfer's length in 32-bit words


class FusedElf:
    LIBRARY = False  # (design.render: operators with equal devices share one, a ConfigureOp per entry)

    def __init__(self, *, name: str, runlist: Runlist, inputs: Sequence[str] = (), outputs: Sequence[str] = (),
                 sizes: Mapping[str, int] | None = None, external: Collection[str] = (),
                 expand_load_pdis: bool = True):
        """``expand_load_pdis=False`` keeps each ConfigureOp a firmware PDI load instead of an array reset and the
        configuration's writes inlined."""
        self.name = name
        self.runlist = list(runlist)
        self.layout = BufferLayout(self.runlist, inputs=inputs, outputs=outputs, sizes=sizes or {}, external=external)
        self.segments = scratch_segments((self.layout.span(n) for _, *names in self.runlist for n in names),
                                         layout.SEGMENT_BYTES)
        self.expand_load_pdis = expand_load_pdis
        self.elf_path: str | None = None
        self._loaded = False
        self._lock = threading.Lock()
        self._device: Any = None
        self._ctx: Any = None
        self._kern: Any = None
        self._params: ParameterScratchpad | None = None

    @property
    def operators(self) -> list:
        """The distinct operators of the runlist, in order."""
        return distinct(op for op, *_ in self.runlist)

    def render_args(self) -> tuple:
        """What the render depends on besides the runlist and its operators (``render_cache``'s key)."""
        return self.expand_load_pdis, layout.SEGMENT_BYTES

    @property
    def artifacts(self) -> tuple[str, ...]:
        """The files of a build directory this ELF attaches."""
        return (f"{self.name}.elf",)

    # ------------------------------------------------------------------
    # Compilation (``parallel_compile.compile_all``: build_spec, then a build's _attach)
    # ------------------------------------------------------------------

    @property
    def compiled(self) -> bool:
        return self.elf_path is not None

    def compile(self) -> Self:
        """Build the ELF (cached on disk, the render too: ``render_cache``; several at once:
        ``parallel_compile.compile_all``)."""
        from ... import parallel_compile

        parallel_compile.compile_all([self])
        return self

    def build_spec(self) -> BuildSpec:
        """The kernels (one translation unit, each definition once: ``kernel_source``; the operators agree on their
        compile flags) and the fused design."""
        ops = self.operators
        flags = {tuple(op.clang_flags) for op in ops}
        assert len(flags) <= 1, f"{self.name}: the operators' clang_flags differ ({sorted(flags)})"
        sources = {ks.raw: ks for op in ops if (ks := op.kernel_source()) is not None}
        arg_bytes = [self.layout.space_bytes["input"], self.layout.space_bytes["output"],
                     *(end - start for start, end in self.segments)]
        entries = [(op, [self._arg(n) for n in names]) for op, *names in self.runlist]
        mlir = design.render(entries, arg_bytes=arg_bytes, library=self.LIBRARY)
        return BuildSpec(kind="fused", label=self.name,
                         source=translation_unit(sources.values()) if sources else None, mlir=mlir,
                         clang_flags=flags.pop() if flags else (), expand_load_pdis=self.expand_load_pdis)

    def _arg(self, name: str) -> design.Arg:
        """Where a buffer (or slice) lies in main's arguments."""
        s = self.layout.span(name)
        if s.space != "scratch":
            return design.Arg(SPACES.index(s.space), s.offset, s.nbytes)
        k = next(k for k, (a, b) in enumerate(self.segments) if a <= s.offset and s.offset + s.nbytes <= b)
        return design.Arg(2 + k, s.offset - self.segments[k][0], s.nbytes)

    def _mark_planned(self) -> Self:
        """Planning mode (``parallel_compile.Precompiler``): nothing built (``load`` raises)."""
        return self

    def _attach(self, work_dir: str, device: object = None) -> Self:
        """Bind the ELF built in ``work_dir`` (``device``: compile_all's interface, unused)."""
        self.elf_path = os.path.join(work_dir, f"{self.name}.elf")
        return self

    # ------------------------------------------------------------------
    # On the NPU
    # ------------------------------------------------------------------

    @property
    def device(self) -> Any:
        """The Device backing this ELF (``load`` first)."""
        assert self._device is not None, f"{self.name}: load() first"
        return self._device

    def load(self) -> Self:
        """Allocate the ELF's BOs and patch its control program (the hardware context follows at the first dispatch).
        Idempotent and thread-safe (two threads racing here would each allocate BOs and write into a discarded one)."""
        if not self._loaded:
            from ... import parallel_compile

            parallel_compile.no_device()
            with self._lock:
                if not self._loaded:
                    assert self.compiled, f"{self.name}: compile() first"
                    self._load(parse_elf(self.elf_path))
                    self._loaded = True
        return self

    def _load(self, elf: ParsedElf) -> None:
        """The PDIs (DEV BOs: the firmware reads them) and the scratchpad; subclasses add their control program."""
        import artlite

        if elf.partition_columns not in (None, COLS):
            raise ValueError(f"{self.elf_path}: built for a partition of {elf.partition_columns} columns")
        dev = self._device = artlite.Device()
        self._first_pdi = np.frombuffer(elf.pdi_map[min(elf.pdi_map)].data, np.uint8)  # (configures the context)
        if sys.platform == "win32":  # (XRT gives a buffer its device address only once a hardware context exists)
            self._open_context()
        self._pdis = {}
        for sym, section in pdi_sections(elf).items():
            bo = self._pdis[sym] = dev.alloc_insts_buffer(section.size)
            bo.write(np.frombuffer(section.data, np.uint8).copy())
            bo.sync_to_device()
        size = next((s.size for s in elf.symbols if s.name == CTRL_SCRATCHPAD), 0)
        self._scratchpad = dev.alloc_insts_buffer(size) if size else None

    def _control_program(self, elf: ParsedElf, address: Callable[[ShimBd], int]) -> tuple[np.ndarray, list[ShimBd]]:
        """The main control program (32-bit words) with its relocations resolved, each data BD's transfer at the
        device address ``address(bd)``; and its data BDs in instruction order."""
        bds = []

        def at(bd: DataBd) -> int:
            if not bd.arg.isdigit() or int(bd.arg) >= 2 + len(self.segments):
                raise ValueError(f"shim DMA relocation of an unknown sequence argument {bd.arg!r}")
            i = int(bd.arg)
            s = ShimBd(bd.offset, SPACES[min(i, 2)], bd.start + (self.segments[i - 2][0] if i >= 2 else 0), bd.words)
            bds.append(s)
            return address(s)

        code, _ = patch_control_code(elf, address=at, pdis={k: self._paddr(bo) for k, bo in self._pdis.items()},
                                     scratchpad=self._paddr(self._scratchpad) if self._scratchpad is not None else None)
        words = np.frombuffer(bytes(code), np.uint32)
        if sys.platform == "win32":  # (run as a transaction: Kernel.dispatch_elf)
            words = txn.without_ddr_patches(words)
        return words, sorted(bds)

    def _paddr(self, bo: Any) -> int:
        """A BO's device address in this ELF's context (Windows: each context has its own; elsewhere none is open yet
        and the address is the same in all)."""
        return bo.paddr if self._ctx is None else bo.paddr_in(self._ctx)

    def set_params(self, values: Mapping[str, int]) -> None:
        """Write runtime parameters (``aie.iron.ScratchpadParameter``: int32 by name) for the next dispatch."""
        if not values:
            return
        if self._params is None:
            self.load()
            path = os.path.join(os.path.dirname(self.elf_path), "params.txt")
            if self._scratchpad is None or not os.path.exists(path):
                raise ValueError(f"{self.name}: no runtime parameters ({sorted(values)})")
            self._params = ParameterScratchpad(self._scratchpad.map().view(np.uint32), path)
        for name, value in values.items():
            self._params.write(name, np.int32(value))
        self._params.sync(self._scratchpad)

    def _kernel(self) -> Any:
        """The hardware context and its Kernel, created at the first dispatch: a context that sits idle (weight
        uploads, other ELFs compiling) only adds firmware work to every runtime resume of the NPU."""
        if self._kern is None:
            import artlite

            self.load()
            with self._lock:
                if self._kern is None:
                    if self._ctx is None:
                        self._open_context()
                    self._kern = artlite.Kernel(self._device, self._ctx, self._kernel_name)
        return self._kern

    def _open_context(self) -> None:
        """The hardware context (the Kernel refers to it): on Windows from an xclbin of the first PDI, beside the ELF.
        The control program runs there as a transaction (``Kernel.dispatch_elf``), its PDI loads included."""
        image, self._kernel_name = self._first_pdi, "main"
        if sys.platform == "win32":
            path = os.path.splitext(self.elf_path)[0] + ".xclbin"
            with _file_lock(path + ".lock"):
                if not os.path.exists(path):
                    pdi_xclbin(image.tobytes(), path)
            image, self._kernel_name = np.fromfile(path, np.uint8), XCLBIN_KERNEL
        self._ctx = self._device.create_context(image, self._kernel_name)
