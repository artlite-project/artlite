"""The AWQ family's tiles: asymmetric integer codes with an integer zero and a scale per group along K, w = s (q - z)
(a ParoQuant checkpoint's linears, ``artlite.paro``, their K in the storage order of their rotation's tan form).

P4_128 (144 B per row and super-block: Q4_K's tile, word-major) holds 4-bit codes in groups of 128: the header words
[word][row][4 B] of a row's 4 nsb words, words 0 .. nsb - 1: s, nsb .. 2 nsb - 1: B = bf16(-s z), 2 nsb .. 3 nsb - 1:
V hi, 3 nsb .. 4 nsb - 1: V lo (V = s (8 - z), exact in fp32 as bf16 hi + lo; each word a bf16 pair, groups 2 w', 2 w'
+ 1), then Q4_K's code words per super-block ([i 8][s 4][row][4 B]). s are the checkpoint's fp16 scales in bf16 (every
format's rule). The decode GEMV runs Q4_K's code MMUL (``mmv/q4kmm``, centered at 8) with s per sub-block as its table
and adds the offsets XS V (Q4_1's form: V as bfp16 hi + lo panels); the prefill GEMMs read the tiles as Q4_K's
tile-major ones, the header unit expanded into A = s, B planes of the dequant (w = A q + B).

The format fills Q4_K's 9216-byte object with 16 rows (an image holds objects of at least 9216 bytes: BFP16's fixed
8 rows, the F16 tensors of a PARO checkpoint): a header of 1 KB per tile costs no bytes a smaller one would save.
"""

from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from ..operators import templating
from .common import TileFormat, _bf16_bytes, _nibbles, _word_order
from .q4k import Q4KSlot

# GemmQ / GemmQd: the header unit's expansion into A and B, then Q4_K's tile-major dequant
_GEMM = templating.source("artlite/formats/awq.gemm.cc.jinja")

# the decode GEMV (``decode.gemv``): the table and the offsets (the codes: ``mmv/q4kmm``)
_MMV = templating.source("artlite/formats/awq.mmv.cc.jinja")


def header(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """A row's header bytes (R, 16 nsb) of scales s (fp32 of fp16) and zeros z (R, 2 nsb groups): [s | B | V hi | V lo]
    bf16."""
    sb = np.asarray(s, np.float32).astype(bfloat16).astype(np.float32)
    v = sb * (8 - z.astype(np.float32))  # (exact: 8 bits times 4)
    vh = v.astype(bfloat16).astype(np.float32)
    return np.concatenate([_bf16_bytes(sb), _bf16_bytes(-sb * z.astype(np.float32)), _bf16_bytes(vh),
                           _bf16_bytes(v - vh)], axis=1)


class AwqSlot(Q4KSlot):
    formats = (TileFormat("P4_128", 21, 144, 16, 8, 128, 16, planes=4, ws_planes=2, word_major=True, mmv="p4"),)
    sources: ClassVar[dict[str, str]] = {"gemm/P4_128": _GEMM, "mmv/P4_128": _MMV}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        per_row = header(f["s"].reshape(R, 2 * nsb), f["z"].reshape(R, 2 * nsb))
        codes = _nibbles(f["q"][:, _word_order(nsb)]).reshape(R, nsb, 4, 8, 4)  # [row][sb][s][i][4 B]
        return [per_row.reshape(R, 4 * nsb, 4).transpose(1, 0, 2), codes.transpose(1, 3, 2, 0, 4)]

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        return rows == 16 and kc == 1024

    def gemm_q_header(self, op: object, kernel: object, B_l1_ty: object, ws_ty: object,
                      st_ty: object) -> tuple[object, object]:
        return kernel("hcp", [B_l1_ty, st_ty, np.int32]), kernel("hdr_p4", [st_ty, ws_ty])

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/q4kmm", f"mmv/{fmt}")


SLOT = AwqSlot()
