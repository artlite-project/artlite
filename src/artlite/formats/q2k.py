"""Q2_K tiles (shared layout and prefill kernels: ``k23``).

Code words per super-block (16 words, 2 planes): word j = sub-block j, nibble b = c(16 j + b) + 4 c(16 j + 8 + b).
Header words per row (20 B per super-block): [sb][sc h 0, sc h 1, mn h 0, mn h 1]
(nibble s: sub-block 8 h + s) | [sb]: bf16 d, bf16 dmin."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import k23
from .common import Slot, TileFormat


def header_words(nsb: int) -> int:
    """Header words per row."""
    return 5 * nsb


# GemmQd: the planar dequant (the header kernels: gemm/k23)
_GEMM = templating.source("artlite/formats/q2k.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``; the realignment and the scaling: ``mmv/k23``)
_MMV = templating.source("artlite/formats/q2k.mmv.cc.jinja")

class Q2KSlot(k23.PlanarK23, Slot):
    formats = (TileFormat("Q2_K", 7, 84, 32, 1, 64, 20, planes=2,
                          word_major=True, dq_row_cycles=9.0, mmv="q2k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q2_K": _GEMM, "mmv/Q2_K": _MMV}
    ws_bytes = 512  # (per 8-row block: A, B of 16 sub-blocks)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        hw = np.zeros((R, header_words(nsb)), np.uint32)
        h = np.stack([f["sc"], f["mn"]], axis=2).reshape(R, nsb, 2, 2, 8)  # [sb][sc | mn][h][s]
        hw[:, :4 * nsb] = k23.words(h).reshape(R, 4 * nsb)
        hw[:, 4 * nsb:5 * nsb] = k23.bf16_u16(f["d"]) | (k23.bf16_u16(f["dmin"]) << 16)
        c = f["q"].reshape(R, nsb, 16, 2, 8)  # [j][a][b]
        return k23.tile_parts(hw, k23.words(c[..., 0, :] + 4 * c[..., 1, :]), R, nsb)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        v = super().mmv_vars(fmt, tile_bytes, nsb)
        assert 24 < v["R2"] <= 32, v  # (4 row groups: the kernels' layout)
        return {**v, "NW2": header_words(nsb)}


SLOT = Q2KSlot()
