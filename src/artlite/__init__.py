"""artlite - Lightweight NPU runtime for AMD Ryzen AI.

Provides Device, Tensor, and Program classes for NPU dispatch.
Compilation is handled by the ``operators`` package (ArtcOperator subclasses).

Simple usage::

    op = SomeOperator(...).compile()
    output = op.run(input_tensor)

Pipeline usage (batch dispatch, explicit sync)::

    dev = artlite.Device()
    op_a = OpA(...).compile(device=dev)
    op_b = OpB(...).compile(device=dev)

    x = artlite.from_torch(x_torch, dev)
    tmp = artlite.empty(x.shape, dtype=x.dtype, device=dev)
    y = artlite.empty(x.shape, dtype=x.dtype, device=dev)

    op_a.dispatch(x, tmp)  # no host sync
    op_b.dispatch(tmp, y)  # tmp stays on device
    result = y.to_torch()  # one sync at the end
"""

import math
import os
import time

import numpy as np
import torch

from . import profiling
from .backend import _native  # (first: the backend ARTLITE_BACKEND selects is artlite._native)
from .operators.base import ArgDirection, ArgSpec

# Re-export low-level C++ bindings for advanced usage
Kernel = _native.Kernel
Error = _native.Error
ExecutableError = _native.ExecutableError
DispatchError = _native.DispatchError
HeapFullError = _native.HeapFullError
Device = _native.Device


def _get_buffer_nbytes(size: tuple, dtype: torch.dtype) -> int:
    return math.prod(size) * torch.tensor([], dtype=dtype).element_size()


# numpy dtype usable to reinterpret the raw byte buffer for each torch dtype.
# All shape/dtype juggling MUST happen at the numpy level, followed by a
# single torch.from_numpy() hop: torch tensors whose provenance contains a
# torch-level shape-changing op (reshape/shape-view) are torn down only after
# nanobind's Py_AtExit leak check, which then falsely reports the underlying
# Buffer/Device instances as leaked.  (Dtype-only torch .view() hops are
# teardown-safe, so bfloat16 is handled via uint16 at the numpy level plus
# one torch dtype view.)
_TORCH_TO_NUMPY: dict[torch.dtype, np.dtype] = {
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.float16: np.dtype(np.float16),
    torch.int8: np.dtype(np.int8),
    torch.uint8: np.dtype(np.uint8),
    torch.int16: np.dtype(np.int16),
    torch.uint16: np.dtype(np.uint16),
    torch.int32: np.dtype(np.int32),
    torch.uint32: np.dtype(np.uint32),
    torch.int64: np.dtype(np.int64),
    torch.bool: np.dtype(np.bool_),
}


def _map_torch(raw: _native.Buffer, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Zero-copy torch view of a raw device buffer with the given shape/dtype."""
    view_as_uint16 = dtype == torch.bfloat16
    np_dtype = np.dtype(np.uint16) if view_as_uint16 else _TORCH_TO_NUMPY[dtype]
    arr = raw.map().view(np_dtype).reshape(shape)
    t = torch.from_numpy(arr)
    if view_as_uint16:
        t = t.view(torch.bfloat16)
    return t


class Tensor:
    """Device-resident tensor with explicit sync control.

    Created via ``from_torch()`` or ``empty()``.  Data stays on device until
    ``.to_torch()`` is called.
    """

    def __init__(self, size: tuple, *, dtype: torch.dtype, device: Device):
        nbytes = _get_buffer_nbytes(size, dtype)
        # alloc_host_buffer returns zeroed memory, so we don't need to zero it
        self._raw = device.alloc_host_buffer(nbytes)
        self.shape = size
        self.dtype = dtype
        self.device = device
        self._synced_to_device = True

    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return _get_buffer_nbytes(self.shape, self.dtype)

    def sync_to_device(self) -> None:
        """Ensure host data is visible to the NPU."""
        if not self._synced_to_device:
            self._raw.sync_to_device()
            self._synced_to_device = True

    def sync_from_device(self) -> None:
        """Ensure NPU results are visible to the host."""
        self._raw.sync_from_device()

    def to_torch(self) -> torch.Tensor:
        """Read data back from device and return a torch tensor."""
        self.sync_from_device()
        return _map_torch(self._raw, self.shape, self.dtype)


def empty(
    size: tuple, *, dtype: torch.dtype = torch.bfloat16, device: Device
) -> "Tensor":
    return Tensor(size=size, dtype=dtype, device=device)


def from_torch(t: torch.Tensor, dev: Device) -> Tensor:
    """A new device Tensor holding a copy of ``t`` (synced to the device)."""
    src = t.contiguous().view(torch.uint8).reshape(-1)
    tensor = Tensor(t.shape, dtype=t.dtype, device=dev)
    torch.from_numpy(tensor._raw.map()).copy_(src)
    tensor._synced_to_device = False
    tensor.sync_to_device()

    return tensor


class Program:
    """A compiled AIE program ready for dispatch on the NPU.

    Two usage modes:

    **Simple (auto-sync)**::

        prog = SomeOperator(...).compile()
        output = prog.run(input_tensor)

    **Pipeline (explicit sync, tensor reuse)**::

        dev = artlite.Device()
        prog_a = OpA(...).compile(device=dev)
        prog_b = OpB(...).compile(device=dev)
        buf = from_torch(x, dev)
        tmp = artlite.empty(buf.shape, dtype=buf.dtype, device=dev)
        out = artlite.empty(..., device=dev)
        prog_a.dispatch(buf, tmp)     # no host sync
        prog_b.dispatch(tmp, out)     # tmp stays on device
        result = out.to_torch()       # one sync at the end
    """

    def __init__(
        self,
        *,
        pdi_path: str,
        insts_path: str,
        arg_specs: list[ArgSpec],
        device: Device | None = None,
    ):
        self.pdi_path = pdi_path
        self.insts_path = insts_path
        self.arg_specs = arg_specs
        self._device = device
        self._exe = None
        self._kern = None
        self._ctx = None
        self._insts_buf = None
        # Profiling state (set by profiling_attach when ARTLITE_PROFILE on)
        self._prof_cfg = None
        self._prof_name = None
        self._prof_tiles_by_class: dict[str, list[tuple[int, int]]] = {}
        self._prof_work_dir = None
        self._profilers: list = []
        self._trace_raw = None
        self._trace_seq = 0
        self._prof_dispatches = 0

    def profiling_attach(
        self,
        cfg: profiling.ProfileConfig,
        *,
        name: str,
        tiles_by_class: dict[str, list[tuple[int, int]]],
        work_dir: str,
    ) -> None:
        """Enable NPU profiling output for this Program (called by
        ArtcOperator.compile when ARTLITE_PROFILE is set).

        tiles_by_class maps "core"/"mt"/"sh" to instrumented tile lists; each
        class gets a Profiler with the matching counter-module readout
        ("core" class = core + memory modules, "mt" = memtile, "sh" = shim)."""
        self._prof_cfg = cfg
        self._prof_name = name
        self._prof_tiles_by_class = tiles_by_class
        self._prof_work_dir = work_dir

    def __repr__(self) -> str:
        return f"Program(pdi_path={self.pdi_path}, insts_path={self.insts_path}, arg_specs={self.arg_specs})"

    def _ensure_loaded(self) -> None:
        """The Device and the instruction buffer (no hardware context yet, see ``_kernel``)."""
        if self._insts_buf is not None:
            return
        from . import parallel_compile

        parallel_compile.no_device()
        if self._device is None:
            self._device = Device()
        dev = self._device

        self._exe = _native.Executable.from_files(self.pdi_path, self.insts_path)
        insts = self._exe.entry().asm_instr_runlist[0]
        self._insts_buf = dev.alloc_insts_buffer(insts.nbytes)
        self._insts_buf.map()[:] = insts.view(np.uint8)
        self._insts_buf.sync_to_device()

    def _kernel(self) -> Kernel:
        """The hardware context and its Kernel, created at the first dispatch: until then
        nothing in the firmware depends on this program (runtime resumes re-create every open
        context)."""
        if self._kern is not None:
            return self._kern
        self._ensure_loaded()
        dev = self._device
        self._ctx = dev.create_context(self._exe)
        self._kern = _native.Kernel(dev, self._ctx)

        cfg = self._prof_cfg
        if cfg is not None:
            if cfg.mode == "trace":
                self._trace_raw = dev.alloc_host_buffer(cfg.trace_size)
                self._trace_raw.map()[:] = 0
                self._trace_raw.sync_to_device()
            else:
                _PROF_CLASS_MODULES = {"core": ("core", "mem"), "mt": ("mt",), "sh": ("sh",)}
                for cls, tiles in self._prof_tiles_by_class.items():
                    readout = cfg.counter_readout(_PROF_CLASS_MODULES[cls])
                    if not tiles or not readout:
                        continue
                    suffix = "" if cls == "core" else f"_{cls}"
                    out = os.path.join(
                        cfg.out_dir, f"{self._prof_name}{suffix}_counters.json"
                    )
                    prof = _native.Profiler(
                        self._ctx,
                        [(int(c), int(r)) for c, r in tiles],
                        readout,
                        out,
                    )
                    profiling.register_profiler(prof)
                    self._profilers.append(prof)
        return self._kern

    def dispatch(self, *buffers: Tensor) -> None:
        """Dispatch the program on the NPU with pre-allocated Buffers.

        No host<->device sync is performed.

        Args:
            *buffers: Buffer instances in MLIR sequence order.
        """
        kern = self._kernel()
        raw_bufs = [b._raw for b in buffers]
        if self._prof_cfg is not None and self._prof_cfg.mode == "trace":
            raw_bufs.append(self._trace_raw)
        if not self._profilers:
            kern.dispatch(self._insts_buf, raw_bufs)
        else:
            # host tracks: wall time, the driver's busy time (mailbox send to response; reading
            # fdinfo stays outside the timed region) and the H clock the driver reports
            busy0 = self._device.busy_ns()
            t0 = time.perf_counter()
            kern.dispatch(self._insts_buf, raw_bufs)
            wall = time.perf_counter() - t0
            busy = self._device.busy_ns() - busy0
            self._prof_dispatches += 1
            if self._prof_dispatches % self._prof_cfg.sample_every == 0:
                self._profilers[0].record("dispatch wall us", wall * 1e6)
                self._profilers[0].record("device busy us", busy / 1e3)
                self._profilers[0].record("H clock MHz", self._device.clocks()[1])
                for prof in self._profilers:
                    prof.sample()
        if self._prof_cfg is not None and self._prof_cfg.mode == "trace":
            # The hardware keeps writing trace packets past the previous run's
            # data, so every dump is cumulative; the last dump is the complete
            # timeline. Write to a single file per program.
            path = os.path.join(self._prof_cfg.out_dir, f"{self._prof_name}.ttbin")
            _native.dump_buffer(self._trace_raw, path)
            if self._trace_seq == 0:
                profiling.register_trace_dump(path, self._prof_work_dir)
            self._trace_seq += 1

    def run(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Convenience: allocate outputs, sync, dispatch, read back.

        Each input tensor is validated against the corresponding ``ArgSpec``:
        the **dtype** must match exactly, and the **total element count**
        (``numel()``) must equal ``math.prod(spec.shape)``.  The actual
        tensor shape may differ (e.g. flattened or transposed views are fine).

        Output buffers are allocated using the shapes from ``arg_specs``
        and returned after device-to-host sync.

        Args:
            *inputs: One ``torch.Tensor`` per ``ArgDirection.IN`` entry
                in ``self.arg_specs``.

        Returns:
            A single output tensor when there is one ``ArgDirection.OUT``
            spec, otherwise a tuple of tensors.
        """

        input_specs = [
            (shape, dtype)
            for direction, shape, dtype in self.arg_specs
            if direction == ArgDirection.IN
        ]
        output_specs = [
            (shape, dtype)
            for direction, shape, dtype in self.arg_specs
            if direction == ArgDirection.OUT
        ]

        if len(inputs) != len(input_specs):
            raise ValueError(
                f"Expected {len(input_specs)} input tensors, got {len(inputs)}"
            )

        for i, (buf, (expected_shape, expected_dtype)) in enumerate(
            zip(inputs, input_specs)
        ):
            if buf.dtype != expected_dtype:
                raise ValueError(
                    f"Input {i}: dtype mismatch — expected {expected_dtype}, got {buf.dtype}"
                )
            expected_numel = math.prod(expected_shape)
            actual_numel = buf.numel()
            if actual_numel != expected_numel:
                raise ValueError(
                    f"Input {i}: element count mismatch — "
                    f"expected {expected_numel} elements (shape {expected_shape}), "
                    f"got {actual_numel} elements (shape {tuple(buf.shape)})"
                )
        self._ensure_loaded()
        dev = self._device

        in_bufs = [from_torch(t, dev) for t in inputs]
        out_bufs = [
            empty(shape, dtype=dtype, device=dev) for shape, dtype in output_specs
        ]

        self.dispatch(*in_bufs, *out_bufs)

        results = [b.to_torch() for b in out_bufs]
        return results[0] if len(results) == 1 else tuple(results)
