"""The AWQ family's tiles: asymmetric integer codes with an integer zero and a scale per group along K, w = s (q - z)
(a ParoQuant checkpoint's linears, ``artlite.paro``, their K in the storage order of their rotation's tan form).
P4_128 (int4) on Q4_K's tile, P5_128 (int5) on Q5_K's.

P4_128 (144 B per row and super-block: Q4_K's tile, word-major) holds 4-bit codes in groups of 128: the header words
[word][row][4 B] of a row's 4 nsb words, words 0 .. nsb - 1: s, nsb .. 2 nsb - 1: B = bf16(-s z), 2 nsb .. 3 nsb - 1:
V hi, 3 nsb .. 4 nsb - 1: V lo (V = s (8 - z), exact in fp32 as bf16 hi + lo; each word a bf16 pair, groups 2 w', 2 w'
+ 1), then Q4_K's code words per super-block ([i 8][s 4][row][4 B]). s are the checkpoint's fp16 scales in bf16 (every
format's rule). The decode GEMV runs Q4_K's code MMUL (``mmv/q4kmm``, centered at 8) with s per sub-block as its table
and adds the offsets XS V (Q4_1's form: V as bfp16 hi + lo panels); the prefill GEMMs read the tiles as Q4_K's
tile-major ones, the header unit expanded into A = s, B planes of the dequant (w = A q + B).

The format fills Q4_K's 9216-byte object with 16 rows (an image holds objects of at least 9216 bytes: BFP16's fixed
8 rows, the F16 tensors of a PARO checkpoint): a header of 1 KB per tile costs no bytes a smaller one would save.

P5_128 (176 B per row and super-block: Q5_K's tile, row-major) holds 5-bit codes in groups of 128: Q5_K's scale
records hold the AWQ words instead, [sb pair p][row][8 words]: s (super-blocks 2 p, 2 p + 1), B = bf16(-s z) (2 p, 2 p +
1), V hi (2 p, 2 p + 1), V lo (2 p, 2 p + 1), V = s (16 - z) (each word the bf16 pair of the super-block's two groups),
then Q5_K's code planes ([sb][plane 5][row][32 B]). The decode GEMV runs Q5_K's code passes (``mmv/Q5_K``: the codes
centered at 16) with s per sub-block as the table and adds XS V; the prefill's GemmQd expands the words into Q5_K's A = s,
B planes (w = A q + B) and dequantizes as Q5_K.
"""

from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

from ..operators import templating
from .common import TileFormat, _bf16_bytes, _nibbles, _word_order
from .q4k import Q4KSlot
from .q5k import Q5KSlot, q5k_codes

# GemmQ / GemmQd: the header unit's expansion into A and B, then Q4_K's tile-major dequant
_GEMM = templating.source("artlite/formats/awq.gemm.cc.jinja")

# the decode GEMV (``decode.gemv``): the table and the offsets (the codes: ``mmv/q4kmm``)
_MMV = templating.source("artlite/formats/awq.mmv.cc.jinja")

# P5_128: GemmQd's header expansion (Q5_K's dequant: ``gemm/q5k_dq``); the decode GEMV's table and offsets (the codes:
# ``mmv/Q5_K``'s passes)
_GEMM5 = templating.source("artlite/formats/awq5.gemm.cc.jinja")
_MMV5 = templating.source("artlite/formats/awq5.mmv.cc.jinja")


def header(s: np.ndarray, z: np.ndarray, center: int = 8) -> np.ndarray:
    """A row's header bytes (R, 16 nsb) of scales s (fp32 of fp16) and zeros z (R, 2 nsb groups): [s | B | V hi | V lo]
    bf16, V = s (center - z) (int4: 8, int5: 16)."""
    sb = np.asarray(s, np.float32).astype(bfloat16).astype(np.float32)
    v = sb * (center - z.astype(np.float32))  # (exact: 8 bits times 5)
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


class Awq5Slot(Q5KSlot):
    formats = (TileFormat("P5_128", 22, 176, 16, 1, 160, 16, planes=5, plane_major=True, ws_planes=2,
                          dq_row_cycles=10.5, mmv="p5"),)
    sources: ClassVar[dict[str, str]] = {"gemm/P5_128": _GEMM5, "mmv/P5_128": _MMV5}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        h = header(f["s"].reshape(R, 2 * nsb), f["z"].reshape(R, 2 * nsb), 16).reshape(R, 4, nsb // 2, 2, 4)
        rec = h.transpose(2, 0, 1, 3, 4).reshape(nsb // 2, R, 32)  # ([pair][row][word: s 2 | B 2 | V hi 2 | V lo 2])
        return [rec, q5k_codes(f["q"], R, nsb)]

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/trq", "mmv/Q5_K", f"mmv/{fmt}")


SLOT = AwqSlot()
SLOT5 = Awq5Slot()
