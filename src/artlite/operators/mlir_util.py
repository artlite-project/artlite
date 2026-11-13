"""Helpers for walking an aie MLIR module: its operations, its device and the device's runtime sequence."""

from collections.abc import Iterator

from aie.dialects import aie
from aie.ir import Module


def walk(op: object) -> Iterator[object]:
    """``op`` and every operation nested in it, depth first."""
    op = getattr(op, "operation", op)
    yield op
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield from walk(inner)


def device_ops(module: Module) -> list[aie.DeviceOp]:
    """The module's ``aie.device`` operations."""
    return [o for o in module.body.operations if isinstance(o, aie.DeviceOp)]


def device_op(module: Module) -> aie.DeviceOp:
    """The module's first ``aie.device`` operation."""
    return next(o for o in module.body.operations if isinstance(o, aie.DeviceOp))


def runtime_sequence(device: aie.DeviceOp) -> object:
    """The device's first ``aie.runtime_sequence`` operation."""
    return next(o for o in device.body_region.blocks[0].operations if o.operation.name == "aie.runtime_sequence")
