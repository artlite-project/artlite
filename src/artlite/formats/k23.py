"""Shared parts of the Q2_K and Q3_K tile formats (``q2k``, ``q3k``; this slot has no formats of its own).

Tiles of up to 32 rows, stored word-major ([word][row][4 B]: a row group's word is one 32-byte load), the header words
first, then each super-block's code words. Sub-block j = 8 h + s of a super-block (half h, slot s) is k = 16 j + 8 a +
b (a: its two blocks of 8, b = 0 .. 7).

Prefill (GemmQd's planar gather, ``gemm/k23``): k-tile objects [word][row] (a k-tile is 4 Q2_K / 6 Q3_K words), the
header word-major too (row-direct, ``hdr_direct``) in slice-sized units from before the tile: ``_hcp`` collects them,
``_hdr_sb`` turns a super-block's scales into bf16 per row and sub-block (an MMUL against I), ``_dq`` expands them per
lane (``al::expand8``) and computes
w = S (128 + q) + C, rounded once to bfp16 (Q2_K 9, Q3_K 10 cycles per row and k-tile)."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from .common import DqKernels, Slot, _bf16_bytes, collected_header, dq_kernel


def words(nib: np.ndarray) -> np.ndarray:
    """(..., 8) nibbles -> (...) uint32 words, nibble j at bits 4 j."""
    return (nib.astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32))).sum(-1, dtype=np.uint32)


def bf16_u16(x: np.ndarray) -> np.ndarray:
    """fp32 -> the bits of its bf16 (as uint32)."""
    return _bf16_bytes(x[..., None]).view(np.uint16)[..., 0].astype(np.uint32)


def tile_parts(hw: np.ndarray, w: np.ndarray, R: int, nsb: int) -> list[np.ndarray]:
    """The parts of a tile: the header words (R, nw) and the code words (R, nsb, words), word-major."""
    codes = np.ascontiguousarray(w.reshape(R, nsb, -1).transpose(1, 2, 0))  # [sb][word][row]
    return [np.ascontiguousarray(hw.T).view(np.uint8), codes.view(np.uint8)]


# GemmQd (prefill): the planar kernels both formats share (``gemm/Q2_K``, ``gemm/Q3_K`` include them)
_GEMM = templating.source("artlite/formats/k23.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``): a row group's word (8 rows, 32 B) is 4-byte aligned only (R rows of 4 B per word):
# the header words a core reads come into its scratch (``_k23al``), the code words are realigned in the MMUL loops
# (``_k23w``)
_MMV = templating.source("artlite/formats/k23.mmv.cc.jinja")

class K23Slot(Slot):
    sources: ClassVar[dict[str, str]] = {"gemm/k23": _GEMM, "mmv/k23": _MMV}


SLOT = K23Slot()


class PlanarK23:
    """GemmQd hooks of the Q2_K / Q3_K slots (their ``gemm/<fmt>`` kernels include ``gemm/k23``): header units read
    through the planar gather (row-direct: word-major), ``_hcp`` collects them, ``_hdr_sb`` computes a super-block's
    scales (``ws_bytes`` per 8-row block); and their decode GEMV's hooks (``mmv/k23``: the realignment)."""

    ws_bytes = 0

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R <= 32, (fmt, R)
        return {f"R{fmt[1]}": R, f"G{fmt[1]}": -(-R // 8)}

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        return True

    def _words(self, op: object) -> dict[str, int]:
        """Words per k-tile W and per slice cw, header words nw, skipped words before the header (whole k-tile slots
        are not collected) and hpad, the junk words hb starts with."""
        f = self.format(op.fmt)
        W, nw = 2 * f.planes, f.header_bytes(1, op.nsb) // 4
        skip = op.hdr_skip // (4 * op.R)
        return {"W": W, "cw": 4 * W, "nw": nw, "skip": skip, "hpad": skip % W}

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        w = self._words(op)
        # ws: the super-block's scales; hb: the header words [word][g]
        return {"ws": (1, self.ws_bytes * op.n // 8), "hb": (1, 4 * (w["nw"] + w["hpad"]) * op.n)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty, unit=True)
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True, unit=True)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        return self._words(op)
