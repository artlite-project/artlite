"""Q6_K tiles (210 B per row and super-block): d [row][sb] bf16 (padded to 32 B) | sc planes [sb pair p][row]
[32 B]: int8 [sb 2][h][i] (sub-block 2 i + h) | 2 half-super-block slices [ql step 2h: R x 32 B | ql step 2h + 1:
R x 32 B | qh: R x 32 B]: ql nibbles of a step, qh nibble l = c(step 2h) + 4 c(step 2h + 1) of lane l (c = high 2
bits). Every part is planes of 32 B per row: the prefill GEMM's memtile gathers a k-tile's words of all rows with
one 4-dimensional pattern.

Prefill: GemmQd's planar gather (``_hcp``, ``_hdr_d`` once per K chunk, ``_hdr_sb``, ``_dq``)."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from .common import (
    DqKernels,
    Slot,
    TileFormat,
    _bf16_bytes,
    _nibbles,
    _word_order,
    collected_header,
    dq_kernel,
)

# GemmQd: the planar kernels
_GEMM = templating.source("artlite/formats/q6k.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``)
_MMV = templating.source("artlite/formats/q6k.mmv.cc.jinja")


class Q6KSlot(Slot):
    formats = (TileFormat("Q6_K", 2, 210, 16, 1, 192, 16, planes=6, plane_major=True, header_pad32=True,
                          ws_planes=2, dq_row_cycles=12.5, mmv="q6k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q6_K": _GEMM, "mmv/Q6_K": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        hs = 2 * nsb  # half super-blocks
        p = f["q"][:, _word_order(nsb)].reshape(R, nsb, 4, 64)  # [row][sb][s][lane]
        ql = _nibbles((p & 15).reshape(R, -1)).reshape(R, hs, 2, 32)  # [row][hs][step 2h, 2h + 1][lanes]
        c = p >> 4
        qh = _nibbles((c[:, :, 0::2] + 4 * c[:, :, 1::2]).reshape(R, -1)).reshape(R, hs, 1, 32)  # [row][hs][lanes]
        sc = f["sc"].reshape(R, nsb, 8, 2).transpose(0, 1, 3, 2)  # [row][sb][h][i] = sc[2 i + h]
        d = np.zeros(-(-2 * nsb * R // 32) * 32, np.uint8)
        d[:2 * nsb * R] = _bf16_bytes(f["d"]).reshape(-1)
        planes = np.ascontiguousarray(sc).view(np.uint8).reshape(R, nsb // 2, 32).transpose(1, 0, 2)  # [pair][row][32]
        halves = np.concatenate([ql, qh], axis=2).transpose(1, 2, 0, 3)  # [hs][ql 2h, ql 2h + 1, qh][row][32 B]
        return [d, planes, halves]

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the super-block's scales [kt][h][ii][g] bf16, then d [sb][g] fp32
        return {"ws": (1, 32 * op.n + 4 * op.nsb * op.n), "hb": (1, 4 * 4 * op._hdr_slot_words)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty)
        chunk = kernel("hdr_d", [hb_ty, ws_ty])  # d of the rows, once per K chunk
        return DqKernels(hdr=hdr, sb=sb, chunk=chunk, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True,
                         unit=False)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        """Where the header unit's parts land after the permutation ([h][j][row] chunks of 32 B = hj planes):
        d chunks n_d at rows R - n_d .. of hj_d, sc planes from hj_sc0 on."""
        R, nsb = op.R, op.nsb
        n_d = -(-2 * nsb * R // 32)
        assert n_d <= R and (6 * R - n_d - nsb // 2 * R) * 32 == op.hdr_skip, (n_d, op.hdr_skip)
        return {"n_d": n_d, "hj_d": 5 - nsb // 2, "hj_sc0": 6 - nsb // 2}

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/trq", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R6 = self.format(fmt).rows(tile_bytes, nsb)
        assert R6 <= 16, R6
        dpad = -(-2 * nsb * R6 // 32) * 32
        return {"R6": R6, "q6_dpad": dpad, "q6_codes": dpad + nsb // 2 * 32 * R6}


SLOT = Q6KSlot()
