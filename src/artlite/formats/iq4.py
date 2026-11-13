"""IQ4_NL and IQ4_XS tiles (R = 8 or 16 rows in row groups of 8; IQ4_NL 144, IQ4_XS 136 B per row and super-block,
the GGUF blocks' sizes): per super-block a unit of ``unit_bytes`` (R 16: IQ4_NL 2304 B, IQ4_XS 2176 B; 4 per
1024-column tile, nothing else in the tile) holding

- the codes [32-k group i 8][8-k step s 4][row group g][row 8][4 B]: a word = the 8 codes of k 32 i + 8 s .. of a row,
  byte b = codes 2 b (low nibble) and 2 b + 1 (high): an 8 x 8 weight panel's 32 bytes in a row;
- IQ4_NL: d [i 8][row] bf16;
- IQ4_XS: the 6-bit scales ls as a low plane (nibbles [i 8][row]: ls & 15) and a high plane (nibbles [i' 4][row]:
  (ls_i' >> 4) | (ls_i'+4 >> 4) << 2), then d [row] bf16;

nibbles in pairs per byte (the first one low), in the order of the list.

w = d v[q] (IQ4_NL: d per 32-k group), w = d (ls - 32) v[q] (IQ4_XS), v = ``gguf_model.IQ4_VALUES`` (16 int8 values).

Decode (``mmv/iq4``): a weight panel (8 rows x 8 k) is its 32 code bytes looked up by the gather unit in a table of the
256 byte values' value pairs [v[lo], v[hi]] (``pair_words``: 16-bit entries, 4x16 gathers, ``common.TAB``): 8 gathers
per panel, two panels per iteration (II 20, bound by the gathers' DM_ADA port); then the MMUL against the x panels, per
32-k group the rows' sums c, and y += hilo(sum_i c_i (ls_i - 32)) d (IQ4_XS: the integer scales exact in bf16) or
y += sum_i c_i d_i (IQ4_NL: bf16 d).

Prefill (``gemm/IQ4_NL``, ``gemm/IQ4_XS``: GemmQd on super-block units, a pass of whole 16-row tiles, MMUL row block t =
tile t): per k-tile and tile the 8 k-steps' 64 code bytes each by 16 gathers of a table of the byte values' bf16 pairs
(``bf16_pair_words``: 32-bit entries, 4x32), w = S v rounded once into bfp16, S = d (IQ4_NL) or bf16(d (ls - 32))
(IQ4_XS: per unit, ``_sb``)."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import k23
from .common import TAB, DqKernels, Slot, TileFormat, dq_kernel, i32


def _pairs(nib: np.ndarray) -> np.ndarray:
    """(..., 2 n) nibbles -> (..., n) bytes, the first of a pair low."""
    return (nib[..., 0::2] | nib[..., 1::2] << 4).astype(np.uint8)


def codes(q: np.ndarray, R: int, nsb: int) -> np.ndarray:
    """(R, 256 nsb) codes -> (nsb, 128 R) bytes [sb][i][s][g][r][4 B] (module docstring)."""
    G = R // 8
    b = _pairs(q.reshape(R, nsb, 8, 4, 8))  # [row][sb][i][s][4 B]
    return np.ascontiguousarray(b.reshape(G, 8, nsb, 8, 4, 4).transpose(2, 3, 4, 0, 1, 5)).reshape(nsb, -1)


def pair_words() -> np.ndarray:
    """The decode's table (256 uint32 = 1 KB): entry b = [v[b & 15], v[b >> 4]] (int8 pair) at byte 4 b, i.e. per group of
    16 entries their 32 bytes twice."""
    from ..gguf_model import IQ4_VALUES

    v = IQ4_VALUES.view(np.uint8).astype(np.uint16)
    e = v[np.arange(256) & 15] | v[np.arange(256) >> 4] << 8
    return np.concatenate([e.reshape(16, 16)] * 2, 1).reshape(-1).view(np.uint32)


TAB_PREFILL = 2048  # (the prefill's table copies)


def bf16_pair_words() -> np.ndarray:
    """The prefill's table (512 uint32 = 2 KB): entry b = [bf16(v[b & 15]), bf16(v[b >> 4])] at byte 8 b, i.e. per group
    of 8 entries their 32 bytes twice (4x32 gathers)."""
    from ..gguf_model import IQ4_VALUES

    v = (IQ4_VALUES.astype(np.float32).view(np.uint32) >> 16).astype(np.uint32)
    e = v[np.arange(256) & 15] | v[np.arange(256) >> 4] << 16
    return np.concatenate([e.reshape(32, 8)] * 2, 1).reshape(-1)


# GemmQd (prefill) on super-block units of whole 16-row tiles
_GEMM = templating.source("artlite/formats/iq4.gemm.cc.jinja")

# the decode GEMV (``decode.gemv``; the 4x16 gathers: ``mmv/g16``, the scaling by d: ``mmv/k23``)
_MMV = templating.source("artlite/formats/iq4.mmv.cc.jinja")

_NL = templating.source("artlite/formats/iq4.nl.cc.jinja")

_XS = templating.source("artlite/formats/iq4.xs.cc.jinja")


class Iq4Slot(Slot):
    formats = (TileFormat("IQ4_NL", 11, 144, 16, 8, 144, 0, planes=4, unit_align=64, dq_row_cycles=9.0, mmv="iq4nl"),
               TileFormat("IQ4_XS", 12, 136, 16, 8, 136, 0, planes=4, unit_align=64, dq_row_cycles=9.0,
                          mmv="iq4xs"))

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R % 8 == 0, R
        parts = [codes(f["q"], R, nsb)]
        if fmt == "IQ4_NL":
            d = k23.bf16_u16(f["d"]).astype(np.uint16).reshape(R, nsb, 8).transpose(1, 2, 0)  # [sb][i][row]
            parts.append(np.ascontiguousarray(d).view(np.uint8).reshape(nsb, -1))
        else:
            ls = f["ls"].reshape(R, nsb, 8).transpose(1, 2, 0)  # [sb][i][row]
            hi = ls >> 4
            parts += [_pairs((ls & 15).reshape(nsb, -1)), _pairs((hi[:, :4] | hi[:, 4:] << 2).reshape(nsb, -1)),
                      np.ascontiguousarray(k23.bf16_u16(f["d"]).astype(np.uint16).T).view(np.uint8)]
        units = np.concatenate(parts, axis=1)
        assert units.shape[1] == self.format(fmt).unit_bytes(R), (fmt, R, units.shape)
        return [units.reshape(-1)]

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return self.format(fmt).unit_bytes(rows)

    sources: ClassVar[dict[str, str]] = {"mmv/iq4": _MMV, "mmv/IQ4_NL": _NL, "mmv/IQ4_XS": _XS, "gemm/IQ4_NL": _GEMM,
                                         "gemm/IQ4_XS": _GEMM}

    # GemmQd (prefill): super-block units of whole 16-row tiles

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 8 units in the dequantizing core's L1: 36 KB; MMUL columns <= 128)
        return tiles <= 8

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: IQ4_XS's scales S [i][n] bf16; the bf16 pair table's two copies (``gemm_qd``: two regions)
        return {"ws": (1, 16 * op.n if op.fmt == "IQ4_XS" else 64), "hb": (1, 64), "tab": (1, TAB_PREFILL),
                "tabb": (1, TAB_PREFILL)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == 16 and op.unit_pitch % 64 == 0, (op.R, op.unit_pitch)
        tab_ty = np.ndarray[(TAB_PREFILL // 4,), np.dtype[np.uint32]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, bf16_pair_words()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        return {"UP": op.unit_pitch}

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/g16", "mmv/iq4", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == 16, R  # (2 row groups: the kernels' layout)
        return {"R4": R, "TAB": TAB}

    def mmv_table(self, fmt: str) -> np.ndarray:
        return pair_words()


SLOT = Iq4Slot()
