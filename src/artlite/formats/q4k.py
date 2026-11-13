"""Q4_K tiles (144 B per row and super-block): per-row scale records (``pack``) | nibbles.

Prefill: 16-row tiles take the tile-major permutation (GemmQ: ``_hcp`` collects the header slots in the staging
buffer, ``_hdr_q4k`` expands them per K chunk; GemmQd: ``_hdr_sb`` per super-block; 1024-column tiles read their
1 KB header as a slice-sized unit from 1 KB before the tile, ``hdr_skip``)."""

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
    sb_major,
)

# GemmQ / GemmQd: the records' expansion (with the shared 6-bit plane8), then the tile-major dequant
_GEMM = templating.source("artlite/formats/q4k.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``)
_MMV = templating.source("artlite/formats/q4k.mmv.cc.jinja")


class Q4KSlot(Slot):
    formats = (TileFormat("Q4_K", 3, 144, 16, 8, 128, 16, planes=4, ws_planes=2, mmv="q4k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q4_K": _GEMM, "mmv/Q4_K": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        g = 8 * nsb  # groups of the chunk (g = 8 sb + i)

        def six(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            v = v.reshape(R, g)
            c = v >> 4
            return _nibbles(v & 15), _nibbles(c[:, :g // 2] + 4 * c[:, g // 2:])

        (sl, sh), (ml, mh) = six(f["sc"]), six(f["mn"])
        # every vector load naturally aligned: the nibbles first, then the smaller parts
        per_row = np.concatenate([sl, ml, sh, mh, _bf16_bytes(f["d"]), _bf16_bytes(-f["dmin"])], axis=1)
        return [per_row, sb_major(_nibbles(f["q"][:, _word_order(nsb)]), R, nsb)]

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        return rows == 16

    def gemm_q_header(self, op: object, kernel: object, B_l1_ty: object, ws_ty: object,
                      st_ty: object) -> tuple[object, object]:
        # header slots collected in the staging buffer, then expanded
        return kernel("hcp", [B_l1_ty, st_ty, np.int32]), kernel("hdr_q4k", [st_ty, ws_ty])

    gemm_q_collects = True

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        # header units collected, then expanded per super-block
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty)
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True, unit=False)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)


SLOT = Q4KSlot()
