"""Explicit L1 addresses for a design's core buffers.

Core data memory behaves like four 16 KB regions of two interleaved banks
(``CONFLICT_DM_BANK_2i/2i+1`` count region i). aiecc places buffers without
knowing which ones a hot loop streams together, and two streams in one region
conflict every cycle (GemmQ's MAC: 222 of 900 cycles per k-tile). IRON's
``ObjectFifo`` has no placement option, so ``pin_l1_buffers`` edits the resolved
module: ``aie-objectfifo-split`` turns every FIFO into per-tile pools, a pinned
pool gets its own ``aie.buffer``s with an ``address`` (``aie-objectfifo-allocate``
keeps buffers a pool already names; aiecc's rerun of the split is a no-op), and a
worker ``Buffer`` gets an ``address`` attribute. aiecc places everything else
(stack, other buffers) in the gaps. The resulting addresses are in
``design.mlir.prj/checked_bank_placement.mlir`` of the RTC cache directory (kept with ``ARTLITE_PROFILE``, which
makes aiecc keep its intermediates).
"""

import math
from collections.abc import Callable

from aie import ir
from aie.dialects import aie
from aie.dialects._aie_enum_gen import WireBundle
from aie.passmanager import PassManager

from ..txn import HardwareLimitError

L1_BYTES = 0x10000  # a core's data memory
L1_ALIGN = 64  # a 128-byte vector load needs 64-byte alignment: an unaligned address is silently rounded down


def pin_l1_buffers(
    module: ir.Module,
    *,
    fifo: Callable[[str, int], list[int] | None],
    buffer: Callable[[str], int | None] | None = None,
) -> int:
    """Pin core-tile buffers of ``module`` in place; returns the number of FIFO pools pinned.

    fifo(fifo name, depth): byte addresses of the pool's buffers on a core tile (None: leave
    it to aiecc); buffer(symbol name): byte address of a worker buffer (None: leave it). Per core tile the pinned
    buffers must be ``L1_ALIGN``-aligned, inside L1 and pairwise disjoint (HardwareLimitError naming them)."""
    pinned = 0
    placed: dict[tuple[int, int], list[tuple[str, int, int]]] = {}  # core tile -> (name, address, bytes) pinned
    with module.context, ir.Location.unknown():
        PassManager.parse("builtin.module(aie.device(aie-place-tiles,aie-objectfifo-split))").run(module.operation)
        device = next(o for o in module.body.operations if isinstance(o, aie.DeviceOp))
        i32 = ir.IntegerType.get_signless(32)
        for op in list(device.regions[0].blocks[0].operations):
            if isinstance(op, aie.ObjectFifoPoolOp) and "fifoName" in op.attributes:
                if not op.tile.owner.opview.is_core_tile():
                    continue
                depth = ir.IntegerAttr(op.attributes["depth"]).value
                addrs = fifo(ir.StringAttr(op.attributes["fifoName"]).value, depth)
                if addrs is None:
                    continue
                if len(addrs) != depth:
                    raise HardwareLimitError(f"FIFO {ir.StringAttr(op.attributes['fifoName']).value}: depth {depth}, "
                                             f"{len(addrs)} pinned addresses {addrs}")
                base = ir.StringAttr(op.attributes["sym_name"]).value.removesuffix("__pool")
                names = []
                nbytes = _nbytes(ir.TypeAttr(op.attributes["elemType"]).value)
                for i, addr in enumerate(addrs):
                    placed.setdefault(_tile(op), []).append((f"{base}__buf{i}", addr, nbytes))
                    aie.BufferOp(ir.TypeAttr(op.attributes["elemType"]).value, op.tile, sym_name=f"{base}__buf{i}",
                                 address=addr, ip=ir.InsertionPoint(op))
                    names.append(f"{base}__buf{i}")
                op.attributes["buffers"] = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(n) for n in names])
                pinned += 1
            elif buffer is not None and isinstance(op, aie.BufferOp) and "sym_name" in op.attributes:
                addr = buffer(ir.StringAttr(op.attributes["sym_name"]).value)
                if addr is not None:
                    op.attributes["address"] = ir.IntegerAttr.get(i32, addr)
                    if op.tile.owner.opview.is_core_tile():
                        placed.setdefault(_tile(op), []).append(
                            (ir.StringAttr(op.attributes["sym_name"]).value, addr, _nbytes(op.results[0].type)))
    check_placement(placed)
    return pinned


def _tile(op: ir.Operation) -> tuple[int, int]:
    tile = op.tile.owner.opview
    return tile.col.value, tile.row.value


def _nbytes(memref: ir.Type) -> int:
    """The bytes of a static memref (0 where its element type has no width)."""
    t = ir.MemRefType(memref)
    return math.prod(t.shape) * getattr(t.element_type, "width", 0) // 8


def check_placement(placed: dict[tuple[int, int], list[tuple[str, int, int]]]) -> None:
    """Per core tile, the pinned buffers (name, address, bytes) are ``L1_ALIGN``-aligned, inside L1 and pairwise
    disjoint; HardwareLimitError naming the buffers."""
    for tile, buffers in placed.items():
        end, last = 0, None
        for name, addr, nbytes in sorted(buffers, key=lambda b: b[1]):
            if addr % L1_ALIGN:
                raise HardwareLimitError(f"tile {tile}: pinned buffer {name} at {addr:#x} is not {L1_ALIGN}-byte aligned")
            if addr < 0 or addr + nbytes > L1_BYTES:
                raise HardwareLimitError(f"tile {tile}: pinned buffer {name} [{addr:#x}, {addr + nbytes:#x}) is outside "
                                         f"L1 (0 .. {L1_BYTES:#x})")
            if addr < end:
                raise HardwareLimitError(f"tile {tile}: pinned buffers {last[0]} [{last[1]:#x}, {end:#x}) and {name} "
                                         f"[{addr:#x}, {addr + nbytes:#x}) overlap")
            end, last = addr + nbytes, (name, addr)


def packet_flow(module: ir.Module, pkt_id: int, src: tuple[int, int], channel: int,
                dsts: list[tuple[int, int, int]]) -> None:
    """A packet flow from tile ``src``'s DMA ``channel`` to ``dsts`` (column, row, DMA channel) in a placed ``module``
    (``pin_l1_buffers``); its tiles exist."""
    with module.context, ir.Location.unknown():
        device = next(o for o in module.body.operations if isinstance(o, aie.DeviceOp))
        block = device.regions[0].blocks[0]
        tiles = {(t.col.value, t.row.value): t for t in block.operations if isinstance(t, aie.TileOp)}
        aie.packetflow(pkt_id=pkt_id, source=tiles[src], source_port=WireBundle.DMA, source_channel=channel,
                       dests=[{"dest": tiles[d[:2]], "port": WireBundle.DMA, "channel": d[2]} for d in dsts],
                       keep_pkt_header=False, ip=ir.InsertionPoint.at_block_terminator(block))
