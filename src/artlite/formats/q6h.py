"""Q6_K's high plane ("Q6_KH"): tiles of a Q6_K matrix's scales and the top 2 bits c = q >> 4 of its codes, in Q2_K's
layout and bytes (``q2k``: 84 B per row and super-block against Q6_K's 210). They are the matrix at 2 bits a weight,

    w ~ 16 d sc (c - 1.5)        (Q6_K: w = d sc (q - 32); q = 16 c + its low 4 bits, taken as 8),

a lossless part of it (nothing is rounded but d, to bf16 as in every K-quant's tiles): the weights of a screening pass
(``models.screen``: an LM head's logits within ~0.7 of the exact ones; the rows that can matter are then computed
exactly on the host from the file's own rows).

Header words per row (20 B per super-block): [sb][4 words: the int8 sc of sub-blocks 4 w .. 4 w + 3, a byte each] |
[sb]: bf16 (16 d). Code words per super-block (16 words): word j = sub-block j, nibble b = c(16 j + b) + 4 c(16 j +
8 + b): Q2_K's code words, read by its loops (``mmv/q2mm``). Decode only: no prefill GEMM reads these tiles."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import k23, q2k
from .common import Slot, TileFormat

SOURCE = "Q6_K"  # the GGUF type whose tensors these tiles are packed from (``pack``'s fields)

# the decode GEMV (``decode.gemv``; the realignment and the scaling: ``mmv/k23``, the sub-block sums: ``mmv/q2mm``)
_MMV = templating.source("artlite/formats/q6h.mmv.cc.jinja")


def fields(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The plane's own fields of a Q6_K tensor's (``QTensor.fields``): d' = 16 d, sc, c = q >> 4; w ~ d' sc (c - 1.5)."""
    return {"d": 16 * f["d"], "sc": f["sc"], "q": f["q"] >> 4}


class Q6HSlot(Slot):
    formats = (TileFormat("Q6_KH", 19, 84, 32, 1, 64, 20, planes=2, word_major=True, mmv="q6h"),)
    sources: ClassVar[dict[str, str]] = {"mmv/Q6_KH": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        """(f: the Q6_K tensor's fields)"""
        f = fields(f)
        hw = np.zeros((R, q2k.header_words(nsb)), np.uint32)
        sc = np.ascontiguousarray(f["sc"].reshape(R, nsb, 4, 4)).view(np.uint8)  # [sb][word][byte]
        hw[:, :4 * nsb] = (sc.astype(np.uint32) << (8 * np.arange(4, dtype=np.uint32))).sum(-1, dtype=np.uint32).reshape(R, 4 * nsb)
        hw[:, 4 * nsb:5 * nsb] = k23.bf16_u16(f["d"])
        c = f["q"].reshape(R, nsb, 16, 2, 8)  # [j][a][b]
        return k23.tile_parts(hw, k23.words(c[..., 0, :] + 4 * c[..., 1, :]), R, nsb)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/q2mm", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        """(Q2_K's variables: its tiles' rows and words, which ``mmv/q2mm`` takes)"""
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert 24 < R <= 32, (fmt, R)  # (4 row groups: the kernels' layout)
        return {"R2": R, "G2": -(-R // 8), "NW2": q2k.header_words(nsb)}


SLOT = Q6HSlot()
