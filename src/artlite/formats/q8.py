"""Q8_0 tiles (272 bytes per row and super-block) as super-block units: per super-block and row group of 8 a
sub-unit [codes [group i][step s][row 8][8 B] (2048 B) | d [row 8][i 8] bf16 (128 B)] (2176 B, 32-byte aligned), the
codes as the decode GEMV's MMUL panels (codes 8 s .. 8 s + 7 of group i of the 8 rows: one 64-byte B panel each, no
transposition in the kernel). A unit is a super-block's sub-units of the tile's row groups: 1024-column tiles (Qwen's
GDN z) have 8 rows, 512-column ones (Gemma's LM head and per-layer projection) 16.

Decode (``decode.gemv``): Gemma's tiles by the row split (a core per row group), Qwen's z by the super-block split (as
the K-quants': pageable).

Prefill (``gemm/Q8_0``, GemmQd on units: ``Slot.unit_bytes``): the weight object's sub-units of the pass's row groups
in order; per sub-unit d expanded per lane ([row][k] = d[row]), per panel the codes to fp32 (exact) times d (exact),
rounded once into bfp16 (bfp16(bf16(d) q))."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from .common import DqKernels, Slot, TileFormat, _bf16_bytes, dq_kernel, i32

# the decode GEMV (``decode.gemv``; the scale tables: ``mmv/dtab``)
_MMV = templating.source("artlite/formats/q8.mmv.cc.jinja")


# GemmQd (prefill) on super-block units: the pass's 8-row groups
_GEMM = templating.source("artlite/formats/q8.gemm.cc.jinja")


class Q8Slot(Slot):
    formats = (TileFormat("Q8_0", 9, 272, 16, 8, 256, 16, planes=8, dq_row_cycles=4.0, mmv="q80"),)
    sources: ClassVar[dict[str, str]] = {"mmv/Q8_0": _MMV, "gemm/Q8_0": _GEMM}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R % 8 == 0, R
        q = np.asarray(f["q"], np.int8).reshape(R // 8, 8, nsb, 8, 4, 8)  # [g][r][sb][i][s][8]
        codes = q.transpose(2, 0, 3, 4, 1, 5).view(np.uint8).reshape(nsb, R // 8, 2048)  # [sb][g][i][s][r][8]
        d = _bf16_bytes(f["d"]).reshape(R // 8, 8, nsb, 16).transpose(2, 0, 1, 3).reshape(nsb, R // 8, 128)  # [sb][g][r][i]
        return [np.concatenate([codes, d], 2).reshape(nsb, -1)]

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return 272 * rows

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (whole 16-row blocks; two weight objects of up to 8 sub-units in the dequantizing core's L1: 34 KB)
        return tiles * rows % 16 == 0 and tiles * rows <= 64

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the scale vectors of the unit ([row group][i][64] bf16); hb unused
        return {"ws": (1, 128 * op.TR), "hb": (1, 64)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R % 8 == 0 and op.halves == 1 and op.n == op.TR, "Q8_0: passes of whole 16-row blocks"
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), units=True)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        # (a tile's unit: its sub-units back to back, 2176 B each)
        assert op.unit_pitch == 272 * op.R, op.unit_pitch
        return {"UP": 2176, "T8": op.TR // 8}

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return True

    def mmv_split(self, fmt: str, nsb: int) -> int:
        # (512-column tiles: each core one row group of every super-block; 1024: 8 rows, the super-block split)
        return 16 if nsb == 2 else 0

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/dtab", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb, self.mmv_split(fmt, nsb))
        assert R == (16 if nsb == 2 else 8), (R, nsb)
        return {"R80": R}


SLOT = Q8Slot()
