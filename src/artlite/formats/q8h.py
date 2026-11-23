"""Q8_0's high plane ("Q8_0H"): tiles of a Q8_0 matrix's scales and the high nibbles h = q >> 4 of its int8 codes, in
Q4_0's layout and bytes (``q4``: 144 B per row and super-block against Q8_0's 272, 53%). They are the matrix at 4 bits a
weight,

    w ~ 16 d (h + 0.5)        (Q8_0: w = d q; q = 16 h + its low 4 bits l, taken as 7.5),

a lossless part of it (nothing is rounded but d, to bf16 as in every tile; 16 d is bf16(d) times 16): the weights of a
screened LM head (``decode.screen``: Inkling's Q8_0 head, 492 MB -> 261 MB a token; the rows that can matter computed
exactly on the host from the file's own rows).

Tile: Q4_0's (d' = 16 d [row][sb][i] bf16 | the codes c = h + 8 in [0, 15] as Q4_0's nibbles [sb][row][128 B]); its
GEMV is Q4_0's (``mmv/q4mm``, ``mmv/q4s``, ``mmv/dtab``) with the codes centered at 7.5 instead of 8 (C = -7.5 XS_i +
sum c X: the -7.5 e_i panels in registers). Decode only: no prefill GEMM reads these tiles."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import q4
from .common import Slot, TileFormat

SOURCE = "Q8_0"  # the GGUF type whose tensors these tiles are packed from (``pack``'s fields)

# the decode GEMV (``decode.gemv``; Q4_0's parts: the scale tables ``mmv/dtab``, the codes' transposition ``mmv/q4s``,
# their MMUL ``mmv/q4mm``)
_MMV = templating.source("artlite/formats/q8h.mmv.cc.jinja")


def fields(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The plane's own fields of a Q8_0 tensor's (``QTensor.fields``), as Q4_0's: d' = 16 d, codes c = (q >> 4) + 8;
    w ~ d' (c - 7.5)."""
    return {"d": 16 * np.asarray(f["d"], np.float32), "q": (np.asarray(f["q"], np.int8) >> 4).astype(np.uint8) + 8}


class Q8HSlot(Slot):
    formats = (TileFormat("Q8_0H", 24, 144, 16, 2, 128, 16, planes=4, mmv="q80h"),)
    sources: ClassVar[dict[str, str]] = {"mmv/Q8_0H": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        """(f: the Q8_0 tensor's fields)"""
        return q4.SLOT.pack("Q4_0", fields(f), R, nsb)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/dtab", "mmv/q4mm", "mmv/q4s", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R <= 16, (fmt, R)
        return {"R8H": R}


SLOT = Q8HSlot()
