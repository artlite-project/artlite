"""Q4_0, Q4_1 and Q4_1_64 tiles (bytes per row and super-block):
  Q4_0 (144): d [row][sb][i] bf16 | nibbles [row][s][i] words
  Q4_1 (160): d | m [row][sb][i] bf16 | nibbles
  Q4_1_64 (144): [row][sb][d of the 4 group pairs | m of the 4] bf16 | nibbles (Q4_0's)

Q4_1_64 is a Q4_1 tensor whose blocks come in pairs of equal (d, m) exact in bf16: an affine INT4 of groups of 64 (MLX's
quantization, Flint's official file) stored as GGUF's Q4_1 of 32; one (d, m) per pair loses nothing and the tile is
Q4_0's size. The packer checks every pair (``_native.pack_tiles`` throws on the first that differs); ``pairs64``
checks a tensor's first blocks to choose the format.

Prefill: tiles of 16 rows x 2048 or 1024 columns take the tile-major permutation (GemmQ, GemmQd; ``_hdr`` expands the
scale header: 1024 columns read it as a slice-sized unit, Q4_0's from 1 KB before the tile), any other (14 or 32
rows, or 512 columns) GemmQd's planar gather (the nibbles as 4 planes
[row][step][32 B]; Gemma's 512-column Q4_0 tiles of 32 rows: one header unit per 4 K chunks, ``GemmQOperator.hg``)."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from .common import (
    KC,
    DqKernels,
    Slot,
    TileFormat,
    _bf16_bytes,
    _nibbles,
    _word_order,
    collected_header,
    dq_kernel,
    sb_major,
)

# GemmQ / GemmQd: the scale header's de-interleave and the tile-major dequant, or the planar kernels
_GEMM = templating.source("artlite/formats/q4.gemm.cc.jinja")


# the decode GEMV (``decode.gemv``; the scale tables: ``mmv/dtab``): the codes' MMUL (``mmv/q4mm``), Gemma's row-split
# Q4_0 tiles of 512 columns, the super-block split of 1024 (its transposition: ``mmv/q4s``)
_MM = templating.source("artlite/formats/q4.mm.cc.jinja")

_Q4S = templating.source("artlite/formats/q4.q4s.cc.jinja")

_MMV = templating.source("artlite/formats/q4.mmv.cc.jinja")

_MMV41 = templating.source("artlite/formats/q4.mmv41.cc.jinja")


_MMV41H = templating.source("artlite/formats/q4.mmv41h.cc.jinja")

SOURCE_64 = "Q4_1"  # the GGUF type Q4_1_64 tiles are packed from


def pairs64(raw: np.ndarray, blocks: int = 1 << 16) -> bool:
    """Whether a Q4_1 tensor's first blocks (raw: its rows of 20-byte blocks) come in pairs (2 i, 2 i + 1) of equal
    (d, m), every d and m exact in bf16 (Q4_1_64's condition; the packer checks all of them)."""
    b = np.ascontiguousarray(raw).view(np.uint8).reshape(-1, 20)[:blocks - blocks % 2]
    dm = b[:, :4].copy().view(np.float16).reshape(-1, 2, 2)
    f = dm.astype(np.float32)
    return bool(np.array_equal(dm[:, 0].view(np.uint16), dm[:, 1].view(np.uint16))
                and not (f.view(np.uint32) & 0xFFFF).any())


class Q4Slot(Slot):
    formats = (
        TileFormat("Q4_0", 0, 144, 16, 2, 128, 16, planes=4, ws_planes=1, dq_row_cycles=6.0, mmv="q40"),
        TileFormat("Q4_1", 1, 160, 16, 2, 128, 32, planes=4, ws_planes=2, dq_row_cycles=6.0, mmv="q41"),
        TileFormat("Q4_1_64", 20, 144, 16, 2, 128, 16, planes=4, ws_planes=2, dq_row_cycles=6.0, mmv="q41h"),
    )

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        if fmt == "Q4_1_64":
            d, m = f["d"].reshape(R, nsb, 4, 2), f["m"].reshape(R, nsb, 4, 2)
            assert np.array_equal(d[..., 0], d[..., 1]) and np.array_equal(m[..., 0], m[..., 1]), "Q4_1 not in pairs"
            hdr = np.concatenate([d[..., 0], m[..., 0]], axis=2).reshape(R, 8 * nsb)
            return [_bf16_bytes(hdr), sb_major(_nibbles(f["q"][:, _word_order(nsb)]), R, nsb)]
        parts = [_bf16_bytes(f["d"])]
        if fmt == "Q4_1":
            parts.append(_bf16_bytes(f["m"]))
        parts.append(sb_major(_nibbles(f["q"][:, _word_order(nsb)]), R, nsb))
        return parts

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return not self.tile_major(fmt, rows, kc)

    def tile_major(self, fmt: str, rows: int, kc: int) -> bool:
        return rows == 16 and kc in (KC, KC // 2)

    def hdr_direct(self, fmt: str, nsb: int) -> bool:
        return nsb == 8  # (headers of 2048 columns; a half-slice header's row pairs are not)

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        if not op.planar:
            return super().qd_scratch(op)
        # ws: the super-block's scales [group 8][d | m][g] bf16; hb: the header units' slots
        return {"ws": (1, 32 * op.n), "hb": (1, op.hdr_units * 4 * 4 * op.b_l1_words)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        if not op.planar:
            return super().qd_kernels(op, kernel, B_l1_ty=B_l1_ty, ws_ty=ws_ty, hb_ty=hb_ty, D_ty=D_ty)
        # the slots of every header unit collected, then expanded per super-block
        hdr, sb = collected_header(kernel, B_l1_ty, ws_ty, hb_ty, unit=True, chunk=op.hg > 1)
        return DqKernels(hdr=hdr, sb=sb, dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty), collect=True, unit=True)

    def qd_template_vars(self, op: object) -> dict[str, int]:
        if op.planar and op.nsb == 2:  # (the header rows of a 512-column tile: whole unit rows of 128 B, 32-row blocks)
            assert op.hdr_skip % 128 == 0 and op.R % 32 == 0, (op.hdr_skip, op.R)
            assert op.hg == 1 or (op.R == 32 and op.us == 2 and op.fmt == "Q4_0"), (op.hg, op.R, op.us)
        return {"hrow0": op.hdr_skip // 128, "hg": op.hg}

    sources: ClassVar[dict[str, str]] = {"gemm/Q4_0": _GEMM, "gemm/Q4_1": _GEMM, "gemm/Q4_1_64": _GEMM, "mmv/q4mm": _MM,
                                         "mmv/q4s": _Q4S, "mmv/Q4_0": _MMV, "mmv/Q4_1": _MMV41, "mmv/Q4_1_64": _MMV41H}

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4 or (fmt == "Q4_0" and nsb == 2)

    def mmv_split(self, fmt: str, nsb: int) -> int:
        return 32 if nsb == 2 else 0  # (512 columns: each core 16 rows of both super-blocks)

    def mmv_helper_y(self, fmt: str, nsb: int) -> int:
        return 64 if nsb == 2 else 0

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/dtab", "mmv/q4mm", *(("mmv/q4s",) if nsb == 4 else ()), f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, int]:
        R = self.format(fmt).rows(tile_bytes, nsb, self.mmv_split(fmt, nsb))
        assert R == 32 if nsb == 2 else R <= 16, (fmt, nsb, R)
        return {{"Q4_0": "R40", "Q4_1": "R41", "Q4_1_64": "R41H"}[fmt]: R}


SLOT = Q4Slot()
