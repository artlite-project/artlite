"""One shim task from a chain of BDs: the runtime sequence's ``chained_fill``.

A shim channel queues at most ``SHIM_QUEUE`` tasks and a BD iterates at most ``MAX_BD_ITER`` times, so the streams
that read several pieces per pass (a GEMM's weights per K chunk, an attention's Q, history and chunk) are written as one
task whose BDs, each iterating over the passes, are chained: the stream is [pass][BD][d2][d1][d0].
"""

from dataclasses import dataclass

from aie.dialects.aie import EndOp, bds, next_bd
from aie.dialects.aiex import dma_configure_task_for, dma_start_task, shim_dma_bd
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import TaskGroup
from aie.iron.runtime._context import active_sequence
from aie.iron.runtime.dmatask import DMATask
from aie.iron.runtime.endpoint import RuntimeEndpoint

from .design import MAX_BD_ITER, TILE_BDS


@dataclass(frozen=True)
class ChainPart:
    """A BD of ``chained_fill`` reading another runtime buffer than the fill's, or with a scratchpad parameter: the
    element offset added to the tap's, or the transfer's length in passes of the tap (as ``ObjectFifoHandle.fill``),
    or in ``length_unit`` elements (a tap of one pass per unit: [1, 2, d1, d0] with length_unit d1 d0 keeps d2's stride,
    which a d2 of size 1 loses)."""

    src: object
    tap: TensorAccessPattern
    offset_parameter: object = None
    length_parameter: object = None
    length_unit: int | None = None


def chained_fill(prod: object, src: object, taps: list[TensorAccessPattern | ChainPart], group: TaskGroup, *,
                 wait: bool = False, repeat_parameter: object = None) -> None:
    """A runtime-sequence fill (or drain: a consumer handle) of ``prod`` (a shim ObjectFifoHandle) from ``src`` as ONE
    task whose BD chain reads ``taps`` in order, the task repeated taps[0].sizes[0] times: every tap is
    [repeat, d2, d1, d0] with the repeat as its BD's iteration (all taps the same repeat count), so
    the stream is [repeat][tap][d2][d1][d0]. A ``ChainPart`` reads its own buffer and parameters. Enrolled in
    ``group`` (freed when it finishes; ``wait``: awaited). ``repeat_parameter``: a ScratchpadParameter n, the task
    runs n + 1 times (at most taps[0].sizes[0])."""
    rt = active_sequence()._runtime
    rp = rt.register_parameter(repeat_parameter)
    parts = [t if isinstance(t, ChainPart) else ChainPart(src, t) for t in taps]
    parts = [ChainPart(p.src, p.tap, rt.register_parameter(p.offset_parameter),
                       rt.register_parameter(p.length_parameter), p.length_unit) for p in parts]
    reps = parts[0].tap.sizes[0]
    assert all(len(p.tap.sizes) == 4 and p.tap.sizes[0] == reps for p in parts) and reps <= MAX_BD_ITER
    assert len(parts) <= TILE_BDS, f"a chain of {len(parts)} BDs: a shim tile's DMA has {TILE_BDS}"

    class _Chain(DMATask):
        def resolve(self, loc: object = None, ip: object = None) -> None:
            task = dma_configure_task_for(self._alloc, repeat_count=reps - 1, issue_token=wait)
            with bds(task) as bd:
                for i, p in enumerate(parts):
                    with bd[i]:
                        shim_dma_bd(p.src.op, tap=p.tap, offset_parameter=p.offset_parameter,
                                    length_parameter=p.length_parameter, length_unit=p.length_unit,
                                    transfer_len=0 if p.length_parameter is not None else None)
                        if i + 1 < len(parts):
                            next_bd(bd[i + 1])
                        else:
                            EndOp()
            self._task = task
            dma_start_task(task, repeat_parameter=rp)

    if prod._endpoint is None:
        prod.endpoint = RuntimeEndpoint(prod._shim_tile)
    active_sequence().note_fifo(prod)
    active_sequence().emit_transfer(_Chain(prod.name, src, tap=parts[0].tap, task_group=group, wait=wait), group)
