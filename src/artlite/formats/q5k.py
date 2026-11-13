"""Q5_K tiles (176 B per row and super-block): scale planes [sb pair p][row][8 words]: sc lo nibbles of
super-blocks 2p, 2p + 1 | mn lo nibbles | high crumbs (nibble i = c_sc + 4 c_mn of group i) | d bf16 x 2 | -dmin
bf16 x 2 | codes [sb][plane 5][row][32 B]: the nibbles of step s (plane s), then qh (nibble of lane l: bit s = the
high bit of step s). Planes of 32 B per row, as Q6_K's: the prefill's dequantizing core gathers both with one
memtile pattern.

Prefill: GemmQd's planar gather (``_hcp``, ``_hdr_sb``, ``_expand`` per k-tile into the ``ex`` buffer, ``_dq``)."""

from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

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

# GemmQd: the 6-bit scale expansion (the shared plane8) and the planar kernels
_GEMM = templating.source("artlite/formats/q5k.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``)
_MMV = templating.source("artlite/formats/q5k.mmv.cc.jinja")


class Q5KSlot(Slot):
    formats = (TileFormat("Q5_K", 6, 176, 16, 1, 160, 16, planes=5, plane_major=True, ws_planes=2,
                          dq_row_cycles=10.5, mmv="q5k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q5_K": _GEMM, "mmv/Q5_K": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        p = f["q"][:, _word_order(nsb)].reshape(R, nsb, 4, 64)  # [row][sb][s][lane]
        ql = _nibbles((p & 15).reshape(R, -1)).reshape(R, nsb, 4, 32)
        hb = (p >> 4).astype(np.uint8)
        qh = _nibbles((hb[:, :, 0] | hb[:, :, 1] << 1 | hb[:, :, 2] << 2 | hb[:, :, 3] << 3).reshape(R, -1))
        codes = np.concatenate([ql, qh.reshape(R, nsb, 1, 32)], axis=2).transpose(1, 2, 0, 3)  # [sb][plane][row][32]
        sc, mn = f["sc"].reshape(R, nsb, 8), f["mn"].reshape(R, nsb, 8)

        def lo(v: np.ndarray) -> np.ndarray:  # the low nibbles: [row][pair][sb][4 B]
            return _nibbles((v & 15).reshape(R, -1)).reshape(R, nsb // 2, 2, 4)

        cr = _nibbles(((sc >> 4) | (mn >> 4) << 2).reshape(R, -1)).reshape(R, nsb // 2, 2, 4)
        dd = _bf16_bytes(f["d"]).reshape(R, nsb // 2, 2, 2).reshape(R, nsb // 2, 1, 4)
        dm = _bf16_bytes(-f["dmin"]).reshape(R, nsb // 2, 1, 4)
        rec = np.concatenate([lo(sc), lo(mn), cr, dd, dm], axis=2).reshape(R, nsb // 2, 32)  # words 0..7
        return [rec.transpose(1, 0, 2), codes]

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        return True  # (the scale planes)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the super-block's scales [group 8][A | B][g] bf16; hb: the 4 header slots; ex: the k-tile's scales
        # expanded per MAC lane [16-row block][word 2][A 2 x 64 bf16 | C 2 x 64 fp32]
        return {"ws": (1, 32 * op.n), "hb": (1, 4 * 4 * op._hdr_slot_words), "ex": (1, 96 * op.n)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty)
        # (the k-tile's expanded scales: a buffer of the dequantizing core)
        ex_ty = np.ndarray[(self.qd_scratch(op)["ex"][1] // 2,), np.dtype[bfloat16]]
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [ex_ty]), collect=True,
                         unit=False, ex_ty=ex_ty)

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/trq", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R5 = self.format(fmt).rows(tile_bytes, nsb)
        assert R5 <= 16, R5
        return {"R5": R5, "q5_codes": nsb // 2 * 32 * R5}


SLOT = Q5KSlot()
