"""The ring variant's memory tiles (``AttnOperator(ring=)``): buffers and tasks of their own in place of the forwards of
streams A and B, written into the design's module after ``pin_l1_buffers``."""

from typing import TYPE_CHECKING

from aie import ir
from aie.dialects import aie
from aie.dialects.aiex import dma_configure_task_for, dma_free_task, dma_start_task

from ...operators.design import COLS
from ...operators.mlir_util import device_op
from .shape import Shape

if TYPE_CHECKING:
    from .attention import AttnOperator

MT_BYTES = 0x80000 - 8192  # a memory tile's for the ring variant's buffers


def mt_bytes(shape: Shape, *, ring: int, window: int, qb: int) -> int:
    """Bytes of a column's memory tile the ring variant takes: the held history and the ring (K and V), the pass's
    Q units and SEG, the output join's objects."""
    s = shape
    return ((ring + window + 1) * 3 * s.used + (2 * qb * s.qh * s.qp + 2) * s.unit + 2 * 4 * qb * s.nd * 72)


def ring_groups(shape: Shape, window: int, qb: int) -> int:
    """The most history groups (the chunk's first included) a ring variant's memory tiles hold beside its ring."""
    s = shape
    rest = (window + 1) * 3 * s.used + (2 * qb * s.qh * s.qp + 2) * s.unit + 2 * 4 * qb * s.nd * 72
    return (MT_BYTES - rest) // (3 * s.used)


def ring_memtiles(attn: "AttnOperator", module: ir.Module) -> None:
    """The ring variant's memory tiles (``pin_l1_buffers``' split module): per column the forwards of streams A and B
    replaced by buffers of their own (the held history and a ring of win + 1 groups after it, the pass's Q units
    and SEG) and tasks the runtime sequence starts once per dispatch (a runtime BD: three dimensions and its
    iteration). A writes the history, then each pass's group into slot b % R (a group's K records only: the fills
    leave out the padding); it sends the history and the whole ring every pass (the cores skip the slots outside
    their window). B writes its history, then per pass the Q units and SEG, then the group's V, each object of two
    units re-laid as stream B's walk sends it (la_v's to_stream, inverted: a BD per object); it sends them in order.
    A slot is written again once the pass before the last that reads it ended (credits: 2). The tasks end with the
    dispatch's last pass (freed after the drains)."""

    s, R, hold, npg = attn.shape, attn.win + 1, attn.ring, attn.n_pass
    ob = s.unit  # (bf16 of an object of two units)
    ks, vs = s.used, s.used // 2  # (bf16 of a group's K, V)
    opg = vs // ob  # (V objects per group)
    nqs = (2 * attn.qb * s.qh * s.qp + 2) * ob // 2  # (bf16 of a pass's Q units and SEG)
    walk = [(4, 72), (s.nd // 2, 288), (72, 1)]  # (an object's cache order [slice][d-block] into la_v's stream order)
    with module.context, ir.Location.unknown():
        dev = device_op(module)
        blk = dev.regions[0].blocks[0]
        ops = {ir.StringAttr(o.attributes["sym_name"]).value: o for o in blk.operations
               if "sym_name" in o.attributes}
        tiles = {(o.col.value, o.row.value): o.result for o in blk.operations if isinstance(o, aie.TileOp)}
        routes = [o for o in blk.operations if isinstance(o, aie.RouteOp)]
        seq = next(o for o in blk.operations if isinstance(o, aie.RuntimeSequenceOp))
        bf = ir.BF16Type.get()
        ends: dict[tuple[int, str], str] = {}
        locks: dict[tuple[int, str], object] = {}
        bufs: dict[tuple[int, str], object] = {}
        ip = ir.InsertionPoint(seq.operation)
        for c in range(COLS):
            mt = tiles[(c, 1)]
            for src, fwd, st in ((f"la_ka{c}", f"la_k{c}", "a"), (f"la_vb{c}", f"la_v{c}", "b")):
                for name in (f"{src}__cons__pool", f"{src}__cons__dma", f"{fwd}__prod__dma"):
                    ops[name].erase()
                for d in ("in", "out"):
                    ends[(c, st + d)] = f"rg_{st}{d}{c}"
                    aie.route_endpoint(ends[(c, st + d)], mt, aie.WireBundle.DMA, ip=ip)
                for r in routes:
                    dst = [ir.FlatSymbolRefAttr(a).value for a in ir.ArrayAttr(r.attributes["destinations"])]
                    srcs = [ir.FlatSymbolRefAttr(a).value for a in ir.ArrayAttr(r.attributes["sources"])]
                    if dst == [f"{src}__cons__dma"]:
                        r.attributes["destinations"] = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(ends[(c, st + "in")])])
                    if srcs == [f"{fwd}__prod__dma"]:
                        r.attributes["sources"] = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(ends[(c, st + "out")])])
            for name, n in (("k", (hold + R) * ks), ("q", nqs), ("v", (hold + R) * vs)):
                bufs[(c, name)] = aie.BufferOp(ir.MemRefType.get([n], bf), mt, sym_name=f"rg_{name}{c}", ip=ip).result
            for name, init in (("k_ready", 0), ("k_free", 2), ("q_ready", 0), ("q_free", 1), ("q_mid", 0),
                               ("v_ready", 0), ("v_free", 2), ("v_mid", 0)):
                locks[(c, name)] = aie.lock(mt, init=init, sym_name=f"rg_{name}{c}", ip=ip)
        acq, rel = aie.LockAction.AcquireGreaterEqual, aie.LockAction.Release
        tasks = []

        def task(end: str, parts: list[tuple], reps: int) -> None:
            """A memory tile task of ``reps`` runs of the BD chain ``parts``: (buffer, offset, length, dims or None,
            the lock acquired and the lock released or None, iteration (steps, stride) or None)."""
            t = dma_configure_task_for(end, repeat_count=reps - 1)
            with aie.bds(t) as bd:
                for i, (buf, off, n, dims, lk, it) in enumerate(parts):
                    with bd[i]:
                        if lk is not None:
                            aie.use_lock(lk[0], acq, 1)
                        kw = {"sizes": [d for d, _ in dims], "strides": [st for _, st in dims]} if dims else {}
                        op = aie.dma_bd(buf, offset=off, transfer_len=n, **kw)
                        if it:
                            op = op if isinstance(op, ir.OpView) else op.owner
                            op.attributes["iteration"] = ir.Attribute.parse(
                                f"#aie.bd_iteration<size = {it[0]}, stride = {it[1]}, current = 0>")
                        if lk is not None:
                            aie.use_lock(lk[1], rel, 1)
                        if i + 1 < len(parts):
                            aie.next_bd(bd[i + 1])
                        else:
                            aie.EndOp()
            dma_start_task(t)
            tasks.append(t)

        first = seq.regions[0].blocks[0]
        with ir.InsertionPoint.at_block_begin(first):
            for c in range(COLS):
                B = lambda n, c=c: bufs[(c, n)]
                L = lambda n, c=c: locks[(c, n)]
                task(ends[(c, "ain")], [(B("k"), 0, hold * ks, None, None, None)], 1)
                task(ends[(c, "ain")], [(B("k"), hold * ks, ks, None, (L("k_free"), L("k_ready")), (R, ks))], npg)
                task(ends[(c, "aout")], [(B("k"), 0, (hold + R) * ks, None, (L("k_ready"), L("k_free")), None)], npg)
                task(ends[(c, "bin")], [(B("v"), 0, ob, walk, None, (hold * opg, ob))], hold * opg)
                qs = [(B("q"), k * ob, ob, walk, None, None) for k in range(nqs // ob)]
                qs[0] = (*qs[0][:4], (L("q_free"), L("q_mid")), None)
                qs[-1] = (*qs[-1][:4], (L("q_mid"), L("q_ready")), None)
                vv = [(B("v"), hold * vs + k * ob, ob, walk, None, (R, vs)) for k in range(opg)]
                if opg == 1:
                    vv[0] = (*vv[0][:4], (L("v_free"), L("v_ready")), (R, vs))
                else:
                    vv[0] = (*vv[0][:4], (L("v_free"), L("v_mid")), (R, vs))
                    vv[-1] = (*vv[-1][:4], (L("v_mid"), L("v_ready")), (R, vs))
                task(ends[(c, "bin")], qs + vv, npg)
                task(ends[(c, "bout")], [(B("q"), 0, nqs, None, (L("q_ready"), L("q_free")), None),
                                         (B("v"), 0, (hold + R) * vs, None, (L("v_ready"), L("v_free")), None)], npg)
        with ir.InsertionPoint(first):  # (after the drains' awaits: the tasks are done by then)
            for t in tasks:
                dma_free_task(t)

