"""Q3_K tiles (shared layout and prefill kernels: ``k23``).

Code words per super-block (24 words, 3 planes): words 3 p + t of quad p (k 32 p .. 32 p + 31: blocks t = 0 .. 3,
sub-blocks 2 p, 2 p + 1): nibbles b of A = q(t 0) + 8 q3_0, B = q(t 1) + 8 q3_1, C = q(t 2) + 8 q3_2, with
q(t 3) = q3_0 + 2 q3_1 + 4 q3_2 (signed unpack: B & 16 = 16 q3_1, C & 32 = 32 q3_2).
Header words per row (14 B per super-block): [sb][h 2]: the low nibbles of the 6-bit
scales u = sc + 32 of sub-blocks 8 h .. 8 h + 7 | [sb]: the high crumbs, nibble s = c(s) + 4 c(s + 8) | [sb pair]:
bf16 d of both."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import k23
from .common import DqKernels, Slot, TileFormat, collected_header, dq_kernel


def header_words(nsb: int) -> int:
    """Header words per row."""
    return (7 * nsb + 1) // 2


# GemmQd: the planar dequant (the header kernels: gemm/k23)
_GEMM = templating.source("artlite/formats/q3k.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``; the realignment and the scaling: ``mmv/k23``)
_MMV = templating.source("artlite/formats/q3k.mmv.cc.jinja")

class Q3KSlot(k23.PlanarK23, Slot):
    formats = (TileFormat("Q3_K", 8, 110, 32, 1, 96, 14, planes=3,
                          word_major=True, dq_row_cycles=10.0, mmv="q3k"),)
    sources: ClassVar[dict[str, str]] = {"gemm/Q3_K": _GEMM, "mmv/Q3_K": _MMV}
    ws_bytes = 256  # (per 8-row block: A of 16 sub-blocks)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # (cb: a dq call's code bytes and scales per lane, 1 KB per 16-row block and quad)
        return {**super().qd_scratch(op), "ex": (1, 128 * op.nm)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty, unit=True)
        cb_ty = np.ndarray[(self.qd_scratch(op)["ex"][1],), np.dtype[np.uint8]]
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [cb_ty]), collect=True, unit=True,
                         ex_ty=cb_ty)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        hw = np.zeros((R, header_words(nsb)), np.uint32)
        u = (f["sc"].astype(np.int16) + 32).astype(np.uint8).reshape(R, nsb, 16)
        hw[:, :2 * nsb] = k23.words((u & 15).reshape(R, nsb, 2, 8)).reshape(R, 2 * nsb)
        cr = u >> 4
        hw[:, 2 * nsb:3 * nsb] = k23.words(cr[..., :8] + 4 * cr[..., 8:])
        d = k23.bf16_u16(f["d"])  # (R, nsb)
        hw[:, 3 * nsb:3 * nsb + nsb // 2] = d[:, 0::2] | (d[:, 1::2] << 16)
        c = f["q"].reshape(R, nsb, 8, 4, 8)  # [p][t][b]
        q3 = c[..., 3, :]
        nib = np.stack([c[..., t, :] + 8 * ((q3 >> t) & 1) for t in range(3)], axis=3)  # (R, nsb, p, t, b)
        return k23.tile_parts(hw, k23.words(nib), R, nsb)

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        v = super().mmv_vars(fmt, tile_bytes, nsb)
        assert 16 < v["R3"] <= 24, v  # (3 row groups: the kernels' layout)
        return {**v, "NW3": header_words(nsb)}


SLOT = Q3KSlot()
