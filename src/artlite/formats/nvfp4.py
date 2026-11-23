"""NVFP4 tiles (R = 8 or 16 rows in row groups of 8; 144 B per row and super-block, the GGUF blocks' size: 4 blocks of
64 k, each 4 UE4M3 scale bytes and 32 code bytes): IQ4_NL's tile geometry (``formats.iq4``), per super-block a unit of
144 R bytes (R 16: 2304 B, 4 per 1024-column tile, nothing else in the tile) holding

- the codes [32-k group i 8][8-k step s 4][row group g][row 8][4 B], as IQ4_NL's (a word = the 8 codes of k 32 i + 8 s ..
  of a row, byte b = codes 2 b (low nibble) and 2 b + 1 (high));
- the scale codes [16-k group j 16][row] u8: the UE4M3 scale's value d (ggml's ``ggml_ue4m3_to_fp32``: half the UE4M3
  value, 0x7F as 0, bit 7 ignored; ``gguf_model.UE4M3_HALF``) recoded losslessly: 0 for d = 0, else b = (E + 11) 8 + f
  for d = 2^E (1 + f / 8) (E in -10 .. 7: every UE4M3 value, subnormals included, has 4 significant bits), so that the
  bf16 of the bits b << 4 times 2^116 is d (``scale_codes``; one multiply in the kernels, zero included).

w = d v[q], v = ``gguf_model.MXFP4_VALUES`` (16 int8 values: twice E2M1's), d per 16-k group.

Decode (``mmv/NVFP4``): IQ4_NL's weight panels (the gather table of the byte values' value pairs, ``iq4.pair_words`` of
MXFP4's values; ``mmv/iq4w``), then the MMUL against the x panels, per 16-k group the rows' sums c (two k-steps: twice
IQ4_NL's chains), and y += sum_j c_j d_j (bf16 d: the products exact; ``_nv_sc``), the scales converted per super-block
(``_nv_ss``).

Prefill (``gemm/NVFP4``, GemmQd on super-block units, IQ4_NL's kernels: ``iq4.gemm.cc.jinja``): per unit the scales of
the pass's rows as bf16 [j][n] (``_sb``), per k-tile and tile the 8 k-steps' 64 code bytes each by 16 gathers of a table
of the byte values' bf16 pairs (``iq4.bf16_pair_words`` of MXFP4's values), w = d v rounded once into bfp16."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import iq4
from .common import TAB, DqKernels, Slot, TileFormat, dq_kernel, i32

SCALE_MUL = 2.0 ** 116  # (a scale code's bf16 bits b << 4 times this: its value)
_SCALE_MUL_CXX = "0x1p116f"  # (the kernels')


def scale_codes() -> np.ndarray:
    """(256,) uint8: the scale code of each UE4M3 byte (module docstring)."""
    from ..gguf_model import UE4M3_HALF

    d = UE4M3_HALF.astype(np.float64)
    m, e = np.frexp(np.where(d > 0, d, 1.0))  # (d = m 2^e, m in [0.5, 1): E = e - 1, 1 + f / 8 = 2 m)
    f = np.round((2 * m - 1) * 8)
    assert np.all((2 * m - 1) * 8 == f) and np.all((e - 1 >= -10) & (e - 1 <= 7))
    return np.where(d > 0, (e - 1 + 11) * 8 + f, 0).astype(np.uint8)


def scale_values(b: np.ndarray) -> np.ndarray:
    """The values of scale codes (as the kernels compute them: bf16 bits b << 4, times 2^116), float32."""
    bits = (b.astype(np.uint32) << 4) << 16
    return (bits.view(np.float32).astype(np.float64) * SCALE_MUL).astype(np.float32)


def pair_words() -> np.ndarray:
    """The decode's gather table (``iq4.pair_words`` of MXFP4's values)."""
    from ..gguf_model import MXFP4_VALUES

    return iq4.pair_words(MXFP4_VALUES)


def bf16_pair_words() -> np.ndarray:
    """The prefill's gather table (``iq4.bf16_pair_words`` of MXFP4's values)."""
    from ..gguf_model import MXFP4_VALUES

    return iq4.bf16_pair_words(MXFP4_VALUES)


# the decode GEMV (``decode.gemv``; the gathers: ``mmv/iq4w``, ``mmv/g16``)
_MMV = templating.source("artlite/formats/nvfp4.mmv.cc.jinja")


class Nvfp4Slot(Slot):
    formats = (TileFormat("NVFP4", 23, 144, 16, 8, 144, 0, planes=4, unit_align=64, dq_row_cycles=9.6, mmv="nvfp4"),)
    sources: ClassVar[dict[str, str]] = {"mmv/NVFP4": _MMV, "gemm/NVFP4": iq4.GEMM}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R % 8 == 0, R
        sc = scale_codes()[f["sc"]].reshape(R, nsb, 16).transpose(1, 2, 0)  # [sb][j][row]
        units = np.concatenate([iq4.codes(f["q"], R, nsb), np.ascontiguousarray(sc).reshape(nsb, -1)], axis=1)
        assert units.shape[1] == self.format(fmt).unit_bytes(R), (fmt, R, units.shape)
        return [units.reshape(-1)]

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return self.format(fmt).unit_bytes(rows)

    # GemmQd (prefill): super-block units of whole 16-row tiles (IQ4_NL's)

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        return tiles <= 8

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: the unit's scales d [j 16][n] bf16; the bf16 pair table's two copies (``gemm_qd``: two regions)
        return {"ws": (1, 32 * op.n), "hb": (1, 64), "tab": (1, iq4.TAB_PREFILL), "tabb": (1, iq4.TAB_PREFILL)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        assert op.R == 16 and op.unit_pitch % 64 == 0, (op.R, op.unit_pitch)
        tab_ty = np.ndarray[(iq4.TAB_PREFILL // 4,), np.dtype[np.uint32]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [tab_ty, tab_ty]), units=True,
                         tables=(tab_ty, bf16_pair_words()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        return {"UP": op.unit_pitch, "SCALE_MUL": _SCALE_MUL_CXX}

    # the decode GEMV

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/g16", "mmv/iq4w", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == 16, R  # (2 row groups: the kernels' layout)
        return {"R4": R, "TAB": TAB, "SCALE_MUL": _SCALE_MUL_CXX}

    def mmv_table(self, fmt: str) -> np.ndarray:
        return pair_words()


SLOT = Nvfp4Slot()
