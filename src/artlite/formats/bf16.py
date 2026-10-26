"""BFP16 tiles (288 bytes per row and super-block) of a BF16 or F16 tensor, its values rounded once to bfp16ebs8: 8 rows
= 32 MMUL panels per super-block [k8][row][exp, 8 int8]. (BF16 and F16 tensors have no tile format of their own:
``tile_format``.) Of an F32 tensor (an MoE router's rows), two planes per row: hi = bfp16(w) and lo = bfp16(w - hi)
(``decode.image.Block``), whose sum is w to ~3e-5.

Prefill: GemmQ's tile-major permutation delivers them as MMUL B panels (no dequant). Decode: the panels against the x
panels as they are (``_MMV``)."""

from typing import ClassVar

import numpy as np
import torch

from ..operators import bfp16
from .common import Slot, TileFormat

# GemmQ: no dequant (the tile-major Q4 dequant renders unused)
_GEMM = r"""{% include "gemm/dq_tile_major" %}
"""

# the decode GEMV (``decode.gemv``): the tile's panels are MMUL B operands
_MMV = r"""// ---- BFP16 (8 rows): the panels [k8 {{ 32 * NSB }}][row 8][exp, 8 int8] (a bfp16 block per row and 8 k, formats/bf16.py);
// core H: super-blocks {{ NSB // 2 }} H .. (k8 {{ 16 * NSB }} H ..). y[g 0] += the hi and lo rows' sums of X W^T, in fp32 (two
// chains: even and odd k8)
template <int H>
__attribute__((noinline)) void {{ p }}_bfp(uint8_t *__restrict a, const uint8_t *__restrict xp,
                                           const uint8_t *__restrict xs, float *__restrict y, uint8_t *__restrict scr)
{
    al::BInR<aie_dm_resource::a> xv((const bfp16ebs8 *)xp);
    al::BInR<aie_dm_resource::b> wv((const bfp16ebs8 *)(a + {{ 16 * NSB * PANEL }} * H));
    v64accfloat C0 = mul_8x8_8x8T(xv.pop(), 1, wv.pop(), 1);
    v64accfloat C1 = mul_8x8_8x8T(xv.pop(), 1, wv.pop(), 1);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int k = 1; k < {{ 8 * NSB }}; k++) {
        C0 = mac_8x8_8x8T(xv.pop(), 1, wv.pop(), 1, C0);
        C1 = mac_8x8_8x8T(xv.pop(), 1, wv.pop(), 1, C1);
    }
    {{ p }}_yadd(y, aie::add(aie::accum<accfloat, 64>(C0), aie::accum<accfloat, 64>(C1)));
}
"""


def _values(w: np.ndarray) -> np.ndarray:
    """A BF16 tensor's bf16 bit patterns (uint16) or an F16 / F32 tensor's values -> float32."""
    if w.dtype in (np.float16, np.float32):
        return w.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).view(torch.bfloat16).float().numpy()


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

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return True


SLOT = Bf16Slot()
