"""``FusedOperator``: a runlist of operators compiled into one ELF and dispatched at once, on named buffers in three
host BOs (input, output, scratch: ``layout``).

Usage::

    fused = FusedOperator(name="silu_gelu", runlist=[(silu, "in", "t"), (gelu, "t", "out")], input_args=["in"],
                          output_args=["out"]).compile()
    fused.get_buffer("in").copy_(x)
    fused.sync_inputs()
    fused()
    fused.sync_outputs()
    y = fused.get_buffer("out").clone()

Lifecycle: ``compile()`` (no device) -> ``load()`` (the BOs and the patched control program; implicit at the first
buffer access or dispatch) -> host writes, syncs, dispatches. Every buffer the runlist reads before an operator writes
it must be an input, external, declared ``uninitialized`` or written by the host (``get_buffer``) before the first
dispatch: a forgotten weight upload is an error, not silent zeros.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ... import txn
from ...elf_parser import ParsedElf
from .elf import FusedElf, ShimBd
from .layout import Runlist, Span, parse_name


class FusedOperator(FusedElf):
    def __init__(self, *, name: str, runlist: Runlist, input_args: Sequence[str], output_args: Sequence[str],
                 buffer_sizes: Mapping[str, int] | None = None, expand_load_pdis: bool = True,
                 external_buffers: Mapping[str, tuple[Any, int]] | None = None,
                 uninitialized: Sequence[str] | None = None):
        """``external_buffers``: scratch buffers in another host BO, name -> (BO, byte offset in it), e.g. the weights
        a model's prefill and decode ELFs share: they need a size, are neither allocated nor synced here (their owner
        does both), and the BDs that address them address that BO. ``uninitialized``: buffers read before any operator
        writes them whose contents do not matter (e.g. an input the kernel ignores in this configuration)."""
        self.input_args, self.output_args = list(input_args), list(output_args)
        self.buffer_sizes = dict(buffer_sizes or {})
        self.external_buffers = dict(external_buffers or {})
        assert all(n in self.buffer_sizes for n in self.external_buffers), "external buffers need sizes"
        super().__init__(name=name, runlist=runlist, inputs=input_args, outputs=output_args, sizes=self.buffer_sizes,
                         external=self.external_buffers, expand_load_pdis=expand_load_pdis)
        self.uninitialized = set(uninitialized or ())
        unknown = self.uninitialized - {parse_name(n)[0] for _, *names in self.runlist for n in names}
        assert not unknown, f"uninitialized buffers not in the runlist: {sorted(unknown)}"
        self._host_written: set[str] = set()
        self._checked = False  # (the host writes, at the first dispatch)
        self._views: dict[str, torch.Tensor] = {}

    def render_args(self) -> tuple:
        return (*super().render_args(), self.input_args, self.output_args, self.buffer_sizes,
                list(self.external_buffers))

    def _reads_before_writes(self) -> list[str]:
        """Buffers some operator reads before any operator writes them, but the inputs, the external and the
        uninitialized ones: the host must write them."""
        written = set(self.input_args) | set(self.external_buffers) | self.uninitialized
        need: list[str] = []
        for op, *names in self.runlist:
            specs = op.get_arg_spec()
            bases = [parse_name(n)[0] for n in names]
            need += [b for spec, b in zip(specs, bases)
                     if spec.direction == "in" and b not in written and b not in need]
            written.update(b for spec, b in zip(specs, bases) if spec.direction == "out")
        return need

    def _check_host_writes(self) -> None:
        if self._checked:
            return
        if missing := [n for n in self._reads_before_writes() if n not in self._host_written]:
            raise RuntimeError(f"{self.name}: buffers read by the runlist but never written by the host: "
                               f"{missing[:8]}{' ...' if len(missing) > 8 else ''} (write them through get_buffer, "
                               f"or declare them uninitialized)")
        self._checked = True

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self, elf: ParsedElf) -> None:
        """The data BOs (SHMEM: PASID-based DMA; the external buffers live in other BOs) and the control program with
        every BD address patched, in an instruction BO."""
        super()._load(elf)
        dev, sizes = self._device, self.layout.space_bytes
        self._in_bo = dev.alloc_host_buffer(max(sizes["input"], 2))
        self._out_bo = dev.alloc_host_buffer(max(sizes["output"], 2))
        self._ext = []  # (scratch start, end, BO, byte offset in the BO)
        for name, (bo, offset) in self.external_buffers.items():
            s = self.layout.spans[name]
            assert s.space == "scratch", f"external buffer {name} must be a scratch buffer"
            self._ext.append((s.offset, s.offset + s.nbytes, bo, offset))
        self._scratch_bo = dev.alloc_host_buffer(max(min((e[0] for e in self._ext), default=sizes["scratch"]), 2))
        code, self._bds = self._control_program(elf, lambda bd: txn.device_address(*self._bo_at(bd.space, bd.start)))
        self._insts_bo = dev.alloc_insts_buffer(code.nbytes)
        np.copyto(self._insts_bo.map(), code.view(np.uint8))
        self._insts_bo.sync_to_device()
        # (the BOs a dispatch uses, for the drivers that pin a command's BOs for the job)
        self._dispatch_bufs = [self._in_bo, self._out_bo, self._scratch_bo, *self._pdis.values()]
        for *_, bo, _ in self._ext:
            if all(bo is not b for b in self._dispatch_bufs):
                self._dispatch_bufs.append(bo)

    def _bo_at(self, space: str, offset: int) -> tuple[Any, int]:
        """(BO, byte offset in it) of a byte of a space."""
        if space == "scratch":
            for start, end, bo, boff in self._ext:
                if start <= offset < end:
                    return bo, boff + offset - start
        return {"input": self._in_bo, "output": self._out_bo, "scratch": self._scratch_bo}[space], offset

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def span(self, name: str) -> Span:
        """Where a buffer (or ``"NAME[a:b]"``) lies: its space, byte offset and size."""
        return self.layout.span(name)

    def buffer_bo(self, name: str) -> tuple[Any, int]:
        """(host BO, byte offset) of a buffer, e.g. for another ELF's ``external_buffers``."""
        self.load()
        return self._bo_at(*self.span(name)[:2])

    def get_buffer(self, name: str) -> torch.Tensor:
        """A bf16 view of a buffer in its host BO (writes reach the device at the next sync). Counts as a host write
        of the buffer."""
        self.load()
        self._host_written.add(parse_name(name)[0])
        if name not in self._views:
            s = self.span(name)
            bo, offset = self._bo_at(s.space, s.offset)
            view = np.frombuffer(bo.map(), np.uint8, s.nbytes, offset)
            self._views[name] = torch.from_numpy(view).view(torch.bfloat16)
        return self._views[name]

    def sync_inputs(self) -> None:
        """Flush the input and scratch BOs to the device."""
        self.load()
        self._in_bo.sync_to_device()
        self._scratch_bo.sync_to_device()

    def sync_input_only(self) -> None:
        """Flush the input BO to the device."""
        self.load()
        self._in_bo.sync_to_device()

    def sync_scratch_from_device(self, length: int | None = None, offset: int = 0) -> None:
        """Pull the scratch BO (or ``length`` bytes from ``offset``) from the device."""
        self.load()
        if length is None:
            self._scratch_bo.sync_from_device()
        else:
            self._scratch_bo.sync_from_device(length, offset)

    def sync_outputs(self) -> None:
        """Pull the output BO from the device."""
        self.load()
        self._out_bo.sync_from_device()

    # ------------------------------------------------------------------
    # The control program's shim BDs
    # ------------------------------------------------------------------

    def shim_bds(self) -> list[ShimBd]:
        """The data BDs of the control program, in instruction order."""
        self.load()
        return list(self._bds)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(self, *, spin_after_s: float | None = None, nap_s: float = 0.0005) -> None:
        """Dispatch the runlist (no syncs; ``set_params`` first for runtime parameters).

        ``spin_after_s``: sleep this long, then busy-poll the completion instead of blocking in the driver. Keeps
        the host core awake (after a blocking wait of several ms the first host work on the result runs several
        times slower); use slightly less than the expected run time."""
        kern = self._kernel()
        self._check_host_writes()
        if spin_after_s is None:
            kern.dispatch_elf(self._insts_bo, self._dispatch_bufs)
            return
        kern.dispatch_elf_async(self._insts_bo, self._dispatch_bufs)
        if spin_after_s > 0:  # (short naps keep the core in shallow idle states: one long sleep slows the next work)
            end = time.perf_counter() + spin_after_s
            while time.perf_counter() < end and not kern.dispatch_poll():
                time.sleep(nap_s)
        while not kern.dispatch_poll():
            pass
        kern.dispatch_wait()  # (retires the command; raises DispatchError on a failure)

    def run(self, *, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """One shot: write ``inputs`` (by buffer name), sync every data BO (overlapping input and output names
        work), dispatch, return the outputs by name."""
        self.load()
        for name, tensor in inputs.items():
            self.get_buffer(name).copy_(tensor.view(torch.bfloat16).flatten())
        for bo in (self._in_bo, self._out_bo, self._scratch_bo):
            bo.sync_to_device()
        self()
        self._in_bo.sync_from_device()
        self._out_bo.sync_from_device()
        return {name: self.get_buffer(name).clone() for name in self.output_args}
