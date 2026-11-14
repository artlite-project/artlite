"""A step's MoE layers on the decode overlay: the transfers of an MoE layer's FFN (``MoeTransfers.build``) and its
metadata objects (``meta_writes``: per layer the route object's and expert objects' templates, the combine's).

An MoE layer (``decode.design``'s "MoE") after its residual's drains into moe_in's x: moe_in (normed: the shared expert's gate
/ up, the router's logits, the shared gate's logit) | the route object (XCH: the logits of every column; the helper core's
routing: ``moe.route``; its control packets out to CP, streamed back into the shim's TileControl: the expert slab's BDs
pushed by the array) | sdown (the shared expert's down x its gate, into the residual) | the expert object (CP, after the
routing's drain: the column's items; its sub-phases on the main core: gate / up, down into acc) | the combine (PART: the
columns' partials of the slice, into the residual, then out).
"""

from typing import TYPE_CHECKING

import numpy as np

from artlite import txn
from artlite.decode import phases as mp
from artlite.decode.protocol import (
    DIMS_OBJ,
    F_COMB,
    F_GU,
    F_MAIN,
    F_ROUTE,
    M_FLAGS,
    M_NS,
    M_NX,
    M_RBASE,
    M_RSTR,
    M_RTDN,
    M_RTGU,
    META,
    XB,
    XPAY,
)
from artlite.moe import route as MR
from artlite.operators.design import COLS
from artlite.overlay import Sequence

if TYPE_CHECKING:
    from .overlay import QwenOverlay

CP_BYTES = 128 * MR.RBUF_OBJECTS  # (a column's routing objects in CP; the expert object starts 256 B before their end)


class MoeTransfers:
    """The MoE layers' transfers and metadata of the decode overlay ``ov``."""

    def __init__(self, ov: "QwenOverlay"):
        self.ov = ov

    def meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """Per layer its route object's and expert objects' templates, the combine's."""
        ov = self.ov
        out = []
        for L in ov.layers:
            gu, dn = ov.image.region(L, "exp_gu"), ov.image.region(L, "exp_down")
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS] = F_MAIN | F_ROUTE, S
            a = ov.dev.address("W", gu.offset)
            m[M_RBASE:M_RBASE + 2] = np.array([a & 0xFFFFFFFF, a >> 32], np.uint32).view(np.int32)
            m[M_RSTR], m[M_RTGU], m[M_RTDN] = gu.col_stride, gu.n_rt * gu.n_kc, dn.n_rt * dn.n_kc
            assert gu.col_stride % 4 == 0 and 2 * ov.cfg.n_experts * gu.col_stride < 1 << 32
            out.append((ov.xch[L], m.view(np.uint8)))
            tpl = np.concatenate([mp.groups(gu, F_GU, S)[0].meta, mp.groups(dn, 0, S)[0].meta]).view(np.uint8)
            for c in range(COLS):  # (the expert object's payload: the gate / up and down templates)
                out.append((ov.cp[L] + c * (CP_BYTES + XB) + CP_BYTES, tpl))
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_NS], m[M_NX] = F_MAIN | F_COMB, S, 2 * S
        out += [(ov.part + c * ov.part_col, m.view(np.uint8)) for c in range(COLS)]
        return out

    def route(self, seq: Sequence) -> None:
        """A step's start: the control packets' route and the routing's shim locks."""
        for c in range(COLS):
            MR.control_route(seq.txn, c)
            for lk in (MR.LOCK_CP, MR.LOCK_DONE):
                seq.txn.write32(txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * lk), 0)

    def build(self, seq: Sequence, L: int, S: int, to: tuple[int, str], run: list[tuple[int, str]] | None) -> None:
        """An MoE layer after its residual's drains into moe_in's x: moe_in, the route object (the control packets' round
        trip: the drain releases LOCK_CP, the control task acquires it; the array's A task), sdown, the expert object (after
        the routing's drain), then ``run``'s weights (after the array's last packet: LOCK_DONE), the partials into PART, the
        combine; its residual -> ``to``."""
        ov = self.ov
        dev, h8 = ov.dev, ov.H // COLS
        ov._weights(seq, (L, "moe_in"), 0)
        seq.sync()
        ov._inputs(seq, ov.inp[(L, "moe_in")][0], S)
        for c in range(COLS):  # (per slot: the shared expert's h, then the router's 32 logits (fp32))
            ov._values_to(seq, c, (L, "sdown"), e0=64 * c, n=64, S=S)
            seq.drain("C", c, dev.address("ACT", ov.xch[L] + 4 * META + 128 * c), S * 128, (32, S, 256, 32))
        seq.sync()
        for c in range(COLS):
            cp = ov.cp[L] + c * (CP_BYTES + XB)
            seq.fill("X", c, dev.address("ACT", ov.xch[L]), XB)
            seq.drain("C", c, dev.address("ACT", cp), CP_BYTES, release=(MR.LOCK_CP, 1))
            seq.fill("X", c, dev.address("ACT", cp), 4 * MR.PACKET_WORDS, packet=False, acquire=(MR.LOCK_CP, 1))
            seq.device_task("A", c)
        ov._weights(seq, (L, "sdown"), 0)
        rg = ov.inp[(L, "sdown")][0]  # (its x [h | h]: the first input object twice)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", rg.off), 2 * XB, DIMS_OBJ)
            seq.fill("X", c, dev.address("ACT", rg.off + XPAY), XB)
        seq.sync()
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", ov.cp[L] + c * (CP_BYTES + XB) + CP_BYTES - 4 * META), XB)
        for c in range(COLS):  # (the array's packets are through: later host pushes on A keep their order)
            lock = txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * MR.LOCK_DONE)
            seq.txn.maskpoll32(lock, 1, 0x3F)
            seq.txn.write32(lock, 0)
        if run:
            ov._weight_run(seq, run)
        pc = ov.part_col
        for c in range(COLS):  # (acc's objects, per 128 rows k, slot, 4 of 32 rows -> PART[k / 2][slot][k % 2][c] ..)
            seq.drain("C", c, dev.address("ACT", ov.part + XB + 2 * h8 * c), 4 * S * h8,
                      (h8 // 2, S, XPAY // 2, XPAY // 4), COLS, (pc // 4, COLS))
        seq.sync()
        for c in range(COLS):  # (the partials are in DDR: the experts' tasks are complete)
            seq.device_done("A", c)
            seq.fill("X", c, dev.address("ACT", ov.part + c * pc), (1 + 2 * S) * XB, DIMS_OBJ)
        for c in range(COLS):
            ov._values_to(seq, c, to, e0=h8 * c, n=h8, S=S)
