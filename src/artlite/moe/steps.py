"""A step's MoE layers on the decode overlay (qwen35moe's, flint's): the transfers of an MoE layer's FFN
(``MoeTransfers.build``) and its metadata objects (``meta_writes``: per layer the route object's and expert objects'
templates, the combine's).

An MoE layer (``decode.design``'s "MoE") after its residual's drains into moe_in's x: moe_in (normed: the shared expert's gate
/ up, the router's logits, a gated shared expert's gate logit) | the route object (XCH: the logits of every column; the
helper core's routing: ``moe.route``; its control packets out to CP, streamed back into the shim's TileControl: the expert
slab's BDs pushed by the array) | sdown (the shared expert's down, times its gate where it has one, into the residual) |
the expert object (CP, after the routing's drain: the column's items; its sub-phases on the main core: gate / up, down into
acc) | the combine (PART: the columns' partials of the slice, into the residual, then out).

sdown's input is the shared expert's h (Fs = ``shared_ffn``: Fs / 8 values of each column) in the side chunk: at Fs = kc
its one chunk; at Fs = kc / 2 (the 35B-A3B's 512, row groups of 2) [h | h], the first input object twice (the design's
unpacked 4-slot inputs).
"""

from typing import Any, Protocol

import numpy as np

from .. import txn
from ..decode import phases as mp
from ..decode.protocol import (
    B_F32,
    B_GU,
    B_KEEP,
    B_PAIR,
    DIMS_OBJ,
    F_COMB,
    F_GU,
    F_MAIN,
    F_ROUTE,
    M_FLAGS,
    M_NRA,
    M_NS,
    M_NT,
    M_NX,
    M_RBASE,
    M_RSTR,
    M_RTDN,
    M_RTGU,
    META,
    XB,
    XPAY,
)
from ..operators.design import COLS
from ..overlay import Sequence
from . import route as MR

CP_BYTES = 128 * MR.RBUF_OBJECTS  # (a column's routing objects in CP; the expert object starts 256 B before their end)
# moe_in's blocks' B_* bits (``decode.design``'s "MoE"; ``phases.groups``' block_flags): the shared expert's gate / up, the
# router's planes (their fp32 sums out), a gated shared expert's gate row (its logit kept)
MOE_BLOCKS = {"sgu": B_GU, "router": B_PAIR | B_F32, "sgate": B_PAIR | B_KEEP}


class MoeOverlay(Protocol):
    """What the transfers take of a model's decode overlay (``decode.model.ModelOverlay``): its device, hidden size,
    config (``n_experts``, ``shared_ffn``), image, the buffers per MoE layer (``xch``: its route object, by layer: the MoE
    layers; ``cp``: per column
    its routing's objects and expert object; ``part``: the combine's objects, ``part_col`` bytes per column), the phases'
    input regions (``inp``) and its fills (``_weights``: a region-major image's A fills of a phase, ``_inputs``,
    ``_values_to``, ``_weight_run``)."""

    dev: Any
    H: int
    cfg: Any
    image: Any
    xch: dict[int, int]
    cp: dict[int, int]
    part: int
    part_col: int
    inp: dict

    def _weights(self, seq: Sequence, phase: tuple[int, str], gi: int) -> None: ...

    def _inputs(self, seq: Sequence, rg: Any, S: int, phase: tuple[int, str] | None = None) -> None: ...

    def _values_to(self, seq: Sequence, c: int, phase: tuple[int, str], *, e0: int, n: int, S: int) -> None: ...

    def _weight_run(self, seq: Sequence, phases: list[tuple[int, str]]) -> None: ...


class MoeRot(Protocol):
    """A rotated overlay's part of an MoE layer (``decode.rotated``; qwen35's ``paro_steps``): moe_in's input fills (its
    rotated copy), the combine's metadata objects (``comb``: per layer, ACT offsets of the columns' XB each; ``comb_meta``
    their words), its rotation objects' A fills and its outputs (the next layer's copies)."""

    comb: dict[int, int]

    def moe_in_fill(self, seq: Sequence, st: Any, L: int) -> None: ...

    def comb_rotations(self, seq: Sequence, L: int) -> None: ...

    def comb_meta(self, L: int, S: int) -> np.ndarray: ...

    def comb_out(self, seq: Sequence, c: int, L: int, S: int) -> None: ...


class MoeTransfers:
    """The MoE layers' transfers and metadata of the decode overlay ``ov``."""

    def __init__(self, ov: MoeOverlay):
        self.ov = ov

    def meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """Per layer its route object's and expert objects' templates, the combine's."""
        ov = self.ov
        out = []
        rot = getattr(ov, "comb", None)  # (rotated experts: per layer its combine's metadata, ``MoeRot``)
        for L in ov.xch:  # (the MoE layers)
            gu, dn = ov.image.region(L, "exp_gu"), ov.image.region(L, "exp_down")
            hr = int((L, "exp_rot") in ov.image._index)  # (a half's rotation object after its gate / up tiles)
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NS] = F_MAIN | F_ROUTE, S
            a = ov.dev.address("W", gu.offset)
            m[M_RBASE:M_RBASE + 2] = np.array([a & 0xFFFFFFFF, a >> 32], np.uint32).view(np.int32)
            m[M_RSTR], m[M_RTGU], m[M_RTDN] = gu.col_stride, gu.n_rt * gu.n_kc + hr, dn.n_rt * dn.n_kc
            assert gu.col_stride % 4 == 0 and 2 * ov.cfg.n_experts * gu.col_stride < 1 << 32
            out.append((ov.xch[L], m.view(np.uint8)))
            mg = mp.groups(gu, F_GU, S)[0].meta
            mg[M_NRA] = hr  # (counted in M_NT: ``decode.design``'s hrot)
            mg[M_NT] += hr
            tpl = np.concatenate([mg, mp.groups(dn, 0, S)[0].meta]).view(np.uint8)
            for c in range(COLS):  # (the expert object's payload: the gate / up and down templates)
                out.append((ov.cp[L] + c * (CP_BYTES + XB) + CP_BYTES, tpl))
            if rot is not None:
                out += [(rot[L] + c * XB, ov.comb_meta(L, S).view(np.uint8)) for c in range(COLS)]
        if rot is None:
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

    def build(self, seq: Sequence, L: int, S: int, to: tuple[int, str], run: list[tuple[int, str]] | None, *,
              rot: "MoeRot | None" = None, st: Any = None) -> None:
        """An MoE layer after its residual's drains into moe_in's x: moe_in, the route object (the control packets' round
        trip: the drain releases LOCK_CP, the control task acquires it; the array's A task), sdown, the expert object (after
        the routing's drain), then ``run``'s weights (after the array's last packet: LOCK_DONE), the partials into PART, the
        combine; its residual -> ``to``. ``rot``: rotated inputs (``MoeRot``; st the step's plan): moe_in's input its
        copy, the combine's rotation objects after the experts' A objects, its metadata the layer's, its outputs the next
        layer's copies."""
        ov = self.ov
        dev, h8, hs = ov.dev, ov.H // COLS, ov.cfg.shared_ffn // COLS  # (hs: the shared expert's h of a column)
        assert ov.image.kc in (ov.cfg.shared_ffn, 2 * ov.cfg.shared_ffn), ov.cfg.shared_ffn
        ov._weights(seq, (L, "moe_in"), 0)
        seq.sync()
        if rot is not None:
            rot.moe_in_fill(seq, st, L)
        else:
            ov._inputs(seq, ov.inp[(L, "moe_in")][0], S)
        for c in range(COLS):  # (per slot: the shared expert's h, then the router's 32 logits (fp32))
            ov._values_to(seq, c, (L, "sdown"), e0=hs * c, n=hs, S=S)
            seq.drain("C", c, dev.address("ACT", ov.xch[L] + 4 * META + 128 * c), S * 128, (32, S, 256, 32))
        seq.sync()
        for c in range(COLS):
            cp = ov.cp[L] + c * (CP_BYTES + XB)
            seq.fill("X", c, dev.address("ACT", ov.xch[L]), XB)
            seq.drain("C", c, dev.address("ACT", cp), CP_BYTES, release=(MR.LOCK_CP, 1))
            seq.fill("X", c, dev.address("ACT", cp), 4 * MR.PACKET_WORDS, packet=False, acquire=(MR.LOCK_CP, 1))
            seq.device_task("A", c)
        ov._weights(seq, (L, "sdown"), 0)
        rg = ov.inp[(L, "sdown")][0]
        if 2 * ov.cfg.shared_ffn == ov.image.kc:  # (its x [h | h]: the first input object twice)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", rg.off), 2 * XB, DIMS_OBJ)
                seq.fill("X", c, dev.address("ACT", rg.off + XPAY), XB)
        else:
            ov._inputs(seq, rg, S, (L, "sdown"))
        seq.sync()
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", ov.cp[L] + c * (CP_BYTES + XB) + CP_BYTES - 4 * META), XB)
        for c in range(COLS):  # (the array's packets are through: later host pushes on A keep their order)
            lock = txn.tile_address(c, 0, txn.SHIM_LOCK0 + 16 * MR.LOCK_DONE)
            seq.txn.maskpoll32(lock, 1, 0x3F)
            seq.txn.write32(lock, 0)
        if rot is not None:
            rot.comb_rotations(seq, L)
        if run:
            ov._weight_run(seq, run)
        pc = ov.part_col
        for c in range(COLS):  # (acc's objects, per 128 rows k, slot, 4 of 32 rows -> PART[k / 2][slot][k % 2][c] ..)
            seq.drain("C", c, dev.address("ACT", ov.part + XB + 2 * h8 * c), 4 * S * h8,
                      (h8 // 2, S, XPAY // 2, XPAY // 4), COLS, (pc // 4, COLS))
        seq.sync()
        for c in range(COLS):  # (the partials are in DDR: the experts' tasks are complete)
            seq.device_done("A", c)
            if rot is not None:  # (the layer's metadata object, then the partials: objects XPAY apart, ``DIMS_OBJ``)
                seq.fill("X", c, dev.address("ACT", ov.comb[L] + c * XB), XB, DIMS_OBJ)
                seq.fill("X", c, dev.address("ACT", ov.part + c * pc + XPAY), 2 * S * XB, DIMS_OBJ)
            else:
                seq.fill("X", c, dev.address("ACT", ov.part + c * pc), (1 + 2 * S) * XB, DIMS_OBJ)
        for c in range(COLS):
            if rot is not None:
                rot.comb_out(seq, c, L, S)
            else:
                ov._values_to(seq, c, to, e0=h8 * c, n=h8, S=S)
