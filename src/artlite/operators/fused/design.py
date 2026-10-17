"""The MLIR design of a fused ELF: its operators' devices side by side in one module, and a device ``main`` whose
runtime sequence runs the entries in order. Per entry a ConfigureOp of its operator's device (when the operator
changes; in a library at every entry) holds a RunOp of the operator's runtime sequence on views of main's arguments
(bf16 subviews, cast to the operator's argument types).

In a library, operators whose devices are equal apart from their runtime sequences share the first one's device (each
keeps its sequence, renamed): one array design compiled, not one per operator.
"""

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import ml_dtypes
import numpy as np

from ..base import ArtcOperator
from .layout import distinct

_ELEM = 2  # bytes of main's argument elements (bf16)


class Arg(NamedTuple):
    """Where an operator's runtime-sequence argument lies in main's arguments."""

    index: int  # main's argument
    offset: int  # bytes
    nbytes: int


def without_sequences(mlir: str) -> str:
    """A design's MLIR text without its runtime sequences (each from its ``aie.runtime_sequence`` line to the closing
    brace at that line's indentation)."""
    out, skip = [], None
    for line in mlir.splitlines():
        if skip is not None:
            if line == skip:
                skip = None
            continue
        if "aie.runtime_sequence" in line and line.rstrip().endswith("{"):
            skip = line[:len(line) - len(line.lstrip())] + "}"
            continue
        out.append(line)
    return "\n".join(out)


def render(entries: Sequence[tuple[ArtcOperator, Sequence[Arg]]], *, arg_bytes: Sequence[int],
           library: bool) -> str:
    """The fused MLIR of ``entries`` (an operator and where its arguments lie, per run). ``arg_bytes``: the sizes of
    main's arguments."""
    from aie.extras.context import mlir_mod_ctx

    ops = distinct(op for op, _ in entries)
    names = {id(op): f"op{i}_{type(op).__name__}" for i, op in enumerate(ops)}
    texts = {names[id(op)]: op.mlir_text() for op in ops}
    device = _device(texts)
    designs: dict[str, str] = {}
    host = {n: designs.setdefault(without_sequences(t), n) if library else n for n, t in texts.items()}
    with mlir_mod_ctx() as ctx:
        sequences = _merge(ctx, texts, host)
        _main(device, [(names[id(op)], args) for op, args in entries], arg_bytes, host, sequences, library)
        return str(ctx.module)


def _device(texts: dict[str, str]) -> Any:
    """The devices' common type (each design: one device with a runtime sequence)."""
    from aie import ir
    from aie.dialects import aie
    from aie.extras.context import mlir_mod_ctx

    device = None
    for name, text in texts.items():
        with mlir_mod_ctx() as ctx:
            module = ir.Module.parse(text, ctx.context)  # (kept: its operations live as long as it)
            devices = [o for o in module.body.operations if isinstance(o, aie.DeviceOp)]
            if len(devices) != 1:
                raise ValueError(f"{name}: expected one device operation, got {len(devices)}")
            if device is None:
                device = devices[0].device
            elif str(device) != str(devices[0].device):
                raise ValueError(f"{name}: device {devices[0].device}, not {device}")
            if not any(o.operation.name == "aie.runtime_sequence" for o in devices[0].body_region.blocks[0].operations):
                raise ValueError(f"{name}: no runtime sequence")
    return device


def _merge(ctx: Any, texts: dict[str, str], host: dict[str, str]) -> dict[str, Any]:
    """Move the designs into the fused module: module-scope declarations once each, each device under its operator's
    name, or (an operator sharing another's device) only its runtime sequence into the host device, renamed. Returns
    each operator's runtime sequence."""
    from aie import ir
    from aie.dialects import aie

    declared: dict[str, str] = {}  # symbol -> its declaration (two designs of one operator declare the same parameters)
    devices: dict[str, Any] = {}
    sequences: dict[str, Any] = {}
    for name, text in texts.items():
        parsed = ir.Module.parse(text, ctx.context)  # (in the fused context: symbol references resolve)
        for op in list(parsed.body.operations):
            if isinstance(op, aie.DeviceOp):
                continue
            attrs = op.operation.attributes
            key = {n: str(attrs[n]) for n in attrs}.get("sym_name")
            text_of = repr((op.operation.name, sorted((n, str(attrs[n])) for n in attrs)))  # (not where it prints)
            if key is not None and key in declared:
                assert declared[key] == text_of, f"{key}: different declarations ({declared[key]}, {text_of})"
                continue
            if key is not None:
                declared[key] = text_of
            ctx.module.body.append(op)
        device = next(op for op in parsed.body.operations if isinstance(op, aie.DeviceOp))
        sequence = next(o for o in device.body_region.blocks[0].operations
                        if o.operation.name == "aie.runtime_sequence")
        sequences[name] = sequence
        if host[name] == name:
            device.sym_name = ir.StringAttr.get(name)
            ctx.module.body.append(device)
            devices[name] = device
            continue
        static = [o for o in device.body_region.blocks[0].operations if o.operation.name != "aie.runtime_sequence"]
        target = devices[host[name]]
        target_static = [o for o in target.body_region.blocks[0].operations
                         if o.operation.name != "aie.runtime_sequence"]
        for a, b in zip(static, target_static, strict=True):  # (its tiles, buffers, locks -> the host's equal ones)
            for ra, rb in zip(a.operation.results, b.operation.results, strict=True):
                ra.replace_all_uses_with(rb)
        sequence.operation.attributes["sym_name"] = ir.StringAttr.get(f"sequence_{name}")
        body = list(target.body_region.blocks[0].operations)
        if body[-1].operation.name == "aie.end":
            sequence.operation.move_before(body[-1].operation)
        else:
            sequence.operation.move_after(body[-1].operation)
        device.operation.erase()
    return sequences


def _main(device: Any, entries: list[tuple[str, Sequence[Arg]]], arg_bytes: Sequence[int], host: dict[str, str],
          sequences: dict[str, Any], library: bool) -> None:
    """The device ``main``: its runtime sequence configures and runs the entries."""
    from aie import ir
    from aie.dialects import aie, aiex, memref

    @aie.device(device)
    def main():
        @aiex.runtime_sequence(*[np.ndarray[(max(n, _ELEM) // _ELEM,), np.dtype[ml_dtypes.bfloat16]]
                                 for n in arg_bytes])
        def sequence(*args):
            last = None
            for name, op_args in entries:
                if name != last or library:
                    body = aiex.ConfigureOp(ir.FlatSymbolRefAttr.get(host[name])).body.blocks.append()
                    last = name
                types = [a.type for a in sequences[name].body.blocks[0].arguments]
                with ir.InsertionPoint(body):
                    views = [_view(args[a.index], a, ty, memref) for a, ty in zip(op_args, types, strict=True)]
                    seq = "sequence" if host[name] == name else f"sequence_{name}"
                    aiex.RunOp(ir.FlatSymbolRefAttr.get(seq), views)


def _view(buf: Any, arg: Arg, ty: Any, memref: Any) -> Any:
    """An argument of an operator's sequence: the bytes ``arg`` of ``buf`` as the operator's memref type ``ty``."""
    from aie import ir

    target = ir.MemRefType(ty)
    shape = [target.shape[j] for j in range(target.rank)]
    nbytes = math.prod(shape) * _elem_bytes(target.element_type)
    # (<=: operators that read different sizes of one buffer, e.g. a norm writes 2048 bf16 of what another reads 8192)
    assert nbytes <= arg.nbytes, f"an argument of {nbytes} bytes in a buffer of {arg.nbytes}"
    strides = [math.prod(shape[j + 1:]) for j in range(len(shape))]
    sv = memref.subview(buf, [arg.offset // _ELEM], [arg.nbytes // _ELEM], [1])
    return memref.reinterpret_cast(result=ir.MemRefType.get(shape, target.element_type), source=sv, offsets=[],
                                   sizes=[], strides=[], static_offsets=[0], static_sizes=shape, static_strides=strides)


def _elem_bytes(t: Any) -> int:
    from aie import ir

    if isinstance(t, ir.IntegerType):
        return t.width // 8
    if isinstance(t, ir.BF16Type):
        return 2
    if isinstance(t, ir.F32Type):
        return 4
    raise TypeError(f"unsupported MLIR element type: {t}")
