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

    x = artlite.from_numpy(x_np, dev)
    tmp = artlite.empty(x.shape, dtype=x.dtype, device=dev)
    y = artlite.empty(x.shape, dtype=x.dtype, device=dev)

    op_a.dispatch(x, tmp)  # no host sync
    op_b.dispatch(tmp, y)  # tmp stays on device
    result = y.numpy()  # one sync at the end
"""

import math
import os
import time

# numpy's OpenBLAS on one thread (read as numpy loads it): the host's products are a few rows (a decision head, a pooling),
# which a pool of threads turns into busy-waiting cores (Kev-4B's head, 512 x 2560 against a question's rows: 11.5 CPUs
# spinning while the NPU worked, 62 W instead of ~25)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import ml_dtypes
import numpy as np
from numpy.typing import DTypeLike

from . import profiling, txn
from .backend import _native  # (first: the backend ARTLITE_BACKEND selects is artlite._native)
from .operators.base import ArgDirection, ArgSpec

# Re-export low-level C++ bindings for advanced usage
Kernel = _native.Kernel
Error = _native.Error
ExecutableError = _native.ExecutableError
DispatchError = _native.DispatchError
HeapFullError = _native.HeapFullError
Device = _native.Device

bfloat16 = np.dtype(ml_dtypes.bfloat16)


class Tensor:
    """Device-resident tensor with explicit sync control.

    Created via ``from_numpy()`` or ``empty()``.  Data stays on device until
    ``.numpy()`` is called.
    """

    def __init__(self, shape: tuple[int, ...], *, dtype: DTypeLike, device: Device):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        # alloc_host_buffer returns zeroed memory, so we don't need to zero it
        self._raw = device.alloc_host_buffer(self.nbytes)
        self.device = device
        self._synced_to_device = True

    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel() * self.dtype.itemsize

    def sync_to_device(self) -> None:
        """Ensure host data is visible to the NPU."""
        if not self._synced_to_device:
            self._raw.sync_to_device()
            self._synced_to_device = True

    def sync_from_device(self) -> None:
        """Ensure NPU results are visible to the host."""
        self._raw.sync_from_device()

    def numpy(self) -> np.ndarray:
        """The NPU's data: synced from the device, a zero-copy view of the buffer (writes to it reach the device
        only through ``from_numpy``'s copy or an explicit ``_raw.sync_to_device()``)."""
        self.sync_from_device()
        return self._raw.map().view(self.dtype)[:self.numel()].reshape(self.shape)


def empty(shape: tuple[int, ...], *, dtype: DTypeLike = bfloat16, device: Device) -> Tensor:
    return Tensor(shape, dtype=dtype, device=device)


def from_numpy(a: np.ndarray, dev: Device) -> Tensor:
    """A new device Tensor holding a copy of ``a`` (synced to the device)."""
    a = np.asarray(a)
    tensor = Tensor(a.shape, dtype=a.dtype, device=dev)
    tensor._raw.map()[:tensor.nbytes] = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
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
        buf = from_numpy(x, dev)
        tmp = artlite.empty(buf.shape, dtype=buf.dtype, device=dev)
        out = artlite.empty(..., device=dev)
        prog_a.dispatch(buf, tmp)     # no host sync
        prog_b.dispatch(tmp, out)     # tmp stays on device
        result = out.numpy()          # one sync at the end
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
        self._arg_bytes = [math.prod(shape) * np.dtype(dtype).itemsize for _, shape, dtype in arg_specs]
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
        if insts.view(np.uint32)[4] & 0xFF == txn.OP_CREATE_SCRATCHPAD:  # (the firmware fails it: ERROR (state 5))
            raise RuntimeError(f"{self.insts_path}: the design reads scratchpad parameters, which only a library or "
                               "fused program has (prefill GEMMs: their unpaced twin, pace None)")
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
            *buffers: Buffer instances in MLIR sequence order, each at least its ArgSpec's
                size (the NPU writes past a smaller one into the host memory after it).
        """
        for i, (b, nbytes) in enumerate(zip(buffers, self._arg_bytes)):
            if b._raw.size < nbytes:
                raise ValueError(f"Buffer {i}: {b._raw.size} bytes, its ArgSpec {nbytes}")
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

    def run(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Convenience: allocate outputs, sync, dispatch, read back.

        Each input array is validated against the corresponding ``ArgSpec``:
        the **dtype** must match exactly, and the **total element count**
        (``size``) must equal ``math.prod(spec.shape)``.  The actual
        array shape may differ (e.g. flattened or transposed views are fine).

        Output buffers are allocated using the shapes from ``arg_specs``
        and returned after device-to-host sync.

        Args:
            *inputs: One array per ``ArgDirection.IN`` entry
                in ``self.arg_specs``.

        Returns:
            A single output array (a view of its device buffer) when there is
            one ``ArgDirection.OUT`` spec, otherwise a tuple of arrays.
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
                f"Expected {len(input_specs)} input arrays, got {len(inputs)}"
            )

        for i, (buf, (expected_shape, expected_dtype)) in enumerate(
            zip(inputs, input_specs)
        ):
            if np.dtype(buf.dtype) != np.dtype(expected_dtype):
                raise ValueError(
                    f"Input {i}: dtype mismatch — expected {expected_dtype}, got {buf.dtype}"
                )
            expected_numel = math.prod(expected_shape)
            actual_numel = buf.size
            if actual_numel != expected_numel:
                raise ValueError(
                    f"Input {i}: element count mismatch — "
                    f"expected {expected_numel} elements (shape {expected_shape}), "
                    f"got {actual_numel} elements (shape {tuple(buf.shape)})"
                )
        self._ensure_loaded()
        dev = self._device

        in_bufs = [from_numpy(t, dev) for t in inputs]
        out_bufs = [
            empty(shape, dtype=dtype, device=dev) for shape, dtype in output_specs
        ]

        self.dispatch(*in_bufs, *out_bufs)

        results = [b.numpy() for b in out_bufs]
        return results[0] if len(results) == 1 else tuple(results)
