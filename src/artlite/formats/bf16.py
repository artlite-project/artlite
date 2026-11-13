"""BFP16 tiles (288 bytes per row and super-block) of a BF16 or F16 tensor, its values rounded once to bfp16ebs8: 8 rows
= 32 MMUL panels per super-block [k8][row][exp, 8 int8]. (BF16 and F16 tensors have no tile format of their own:
``tile_format``.) Of an F32 tensor (an MoE router's rows), two planes per row: hi = bfp16(w) and lo = bfp16(w - hi)
(``decode.image.Block``), whose sum is w to ~3e-5.

Prefill: GemmQ's tile-major permutation delivers them as MMUL B panels (no dequant). Decode: the panels against the x
panels as they are (``_MMV``)."""

from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from ..operators import bfp16, templating
from .common import Slot, TileFormat

# GemmQ: no dequant (the tile-major Q4 dequant renders unused)
_GEMM = templating.source("artlite/formats/bf16.gemm.cc.jinja")

# the decode GEMV (``decode.gemv``): the tile's panels are MMUL B operands
_MMV = templating.source("artlite/formats/bf16.mmv.cc.jinja")


def _values(w: np.ndarray) -> np.ndarray:
    """A BF16 tensor's bf16 bit patterns (uint16) or an F16 / F32 tensor's values -> float32."""
    if w.dtype in (np.float16, np.float32):
        return w.astype(np.float32)
    return np.ascontiguousarray(w).view(bfloat16).astype(np.float32)


class Bf16Slot(Slot):
    formats = (TileFormat("BFP16", 5, 288, 8, 8, 288, 0, fixed_rows=8, mmv="bfp"),)
    field = "w"
    sources: ClassVar[dict[str, str]] = {"gemm/BFP16": _GEMM, "mmv/BFP16": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        # 8 rows: MMUL panels [k8][row][exp, 8 int8] of the rows' values
        e, m = bfp16.blocks(_values(f["w"]).reshape(R, 32 * nsb, 8))  # (R, k8), (R, k8, 8)
        blk = np.concatenate([e[..., None], m.view(np.uint8)], axis=-1)  # (R, k8, 9)
        return [np.ascontiguousarray(blk.transpose(1, 0, 2))]

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        return rows == 8

    def tile_major_dims(self, op: object) -> tuple[list, list]:
        # 8 tiles x [k8 32][8-row panel 18 words] per slice; S2MM writes [kt][tile][k8 8][18], MM2S reads k-tile
        # objects [16-row block][k8][8-row half][panel] = the MAC pop order
        return [(8, 144), (4, 1152), (144, 1)], [(16, 288), (8, 18), (2, 144), (18, 1)]

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return True


SLOT = Bf16Slot()
