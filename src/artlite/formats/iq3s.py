"""IQ3_S tiles (R = 17 .. 20 rows in 3 row groups of 8, the third of r2 = R - 16 rows; 110 B per row and super-block,
the GGUF block's size): per super-block a unit of ``UNIT`` B (4 per 1024-column tile, nothing else in the tile) holding

- A (1536 B): the grid indices' low bytes as records [32-k group i 8][panel pair h 2][8-k panel l' 2] of 48 B: rows 0
  .. 7 (row r's two indices of panel l = 2 h + l' at 2 r, 2 r + 1), rows 8 .. 15, rows 16 .. R - 1 (2 r2 B), zeros, then
  at byte 40 the panels' 9th index bits, a 16-bit mask per row group g (bit 2 r + j: row 8 g + r's index j), zeros;
- B (768 B): the signs as records [i 8][panel l 4] of 24 B: rows 0 .. 7's sign bytes (bit j: value 8 l + j of group i
  negative), rows 8 .. 15, rows 16 .. R - 1, zeros, then at byte 20 a 4-byte slot of C;
- C (the 32 slots of B, slot n in record n): slot r < R row r's scale bytes (the GGUF block's: groups 2 j, 2 j + 1 in
  byte j's low and high nibble), slot R + s rows 2 s and 2 s + 1's d (bf16), zeros past them.

So each weight panel (8 rows x 8 k: 16 indices) has its index bytes in 16 contiguous bytes and its signs in 8 (rows past
R: other bytes, whose panel rows the kernels drop). w = d (1 + 2 sc) q, q = the 9-bit index's grid value
(``gguf_model.iq3s_grid``: 4 values per index, in {1, 3, .., 15}) negated where its sign is set.

Decode (``mmv/IQ3_S``): per super-block the panels' 9th bits as bytes 0 / 128 (``_iq3s_hb``: per 4 records their 12
masks as 3 x 64 bits), then per 4 panels (64 index bytes) the gather offsets 4 (qs + 256 qh) on the MMUL (``mul_elem`` +
``mac_elem``; a select from a mask in the gather loop leaves it unpipelined), 4 gathers of 16-bit entries per panel
(``grid_halves``: the grid's 4 values as nibbles, 2 KB per copy at ``common.TAB``), the nibbles unpacked, the signs by
``vaddsub8`` from the tile's masks; the MMUL and the scaling as IQ3_XXS's (``mmv/iq3c``), the scales 1 + 2 sc and d
from C's slots (``_iq3s_ss``).

Prefill (``gemm/IQ3_S``: GemmQd on super-block units, IQ3_XXS's structure): the pass's rows compacted (MMUL row q = tile
q / R's row q % R); per unit the scales S = bf16((1 + 2 sc) d) and the signs of the compacted rows; per k-tile the index
lists compacted (the low bytes and the 9th bits as bytes, 4 lists per iteration so that every window's shift is a
constant: the weight object, its records' k-tiles and the units are 64-byte aligned), per 4 panels the gather offsets 4 lo
+ 8 hb on the MMUL (the decode's form), the codes 64 +- g by 4x16 gathers of the nibble grid (unpacked), the panels w = S
(128 + u) - 192 S rounded once (as IQ3_XXS's ``_iq3s``)."""

from typing import ClassVar

import numpy as np

from ..operators import templating
from . import k23
from .common import TAB, TAB_BYTES, DqKernels, Slot, TileFormat, dq_kernel, i32
from .iq3 import Iq3Slot

UNIT = 2304  # bytes of a super-block's unit (A + B)


def grid_halves() -> np.ndarray:
    """The grid as 16-bit entries for the decode's 4x16 gathers (``TAB_BYTES`` / 4 uint32): entry i (its 4 values,
    one nibble each, value j in nibble j) at byte 4 i of a table whose 32-byte groups of 16 entries are stored twice."""
    from ..gguf_model import iq3s_grid

    g = iq3s_grid().astype(np.uint16)
    e = g[:, 0] | g[:, 1] << 4 | g[:, 2] << 8 | g[:, 3] << 12
    return np.concatenate([e.reshape(32, 16)] * 2, 1).reshape(-1).view(np.uint32)


_MMV = templating.source("artlite/formats/iq3s.mmv.cc.jinja")


# GemmQd (prefill) on super-block units: the pass's rows compacted (as IQ3_XXS's)
_GEMM = templating.source("artlite/formats/iq3s.gemm.cc.jinja")


class Iq3sSlot(Slot):
    formats = (TileFormat("IQ3_S", 13, 110, 20, 1, 110, 0, planes=3, unit_fixed=UNIT, dq_row_cycles=12.0, mmv="iq3s"),)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert 16 < R <= 20, R
        r2 = R - 16
        idx = np.zeros((24, nsb, 8, 2, 2, 2), np.uint16)  # [row][sb][i][h][l'][j]
        idx[:R] = f["idx"].reshape(R, nsb, 8, 2, 2, 2)
        a = np.zeros((nsb, 8, 2, 2, 48), np.uint8)
        low = (idx & 0xFF).astype(np.uint8).reshape(3, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6)  # [..][g][r][j]
        a[..., :32] = low[:, :, :, :, :2].reshape(nsb, 8, 2, 2, 32)
        a[..., 32:32 + 2 * r2] = low[:, :, :, :, 2, :r2].reshape(nsb, 8, 2, 2, -1)
        hi = (idx >> 8).astype(np.uint16).reshape(3, 8, nsb, 8, 2, 2, 2).transpose(2, 3, 4, 5, 0, 1, 6)
        mask = (hi.reshape(nsb, 8, 2, 2, 3, 16) << np.arange(16, dtype=np.uint16)).sum(-1, dtype=np.uint16)
        a[..., 40:46] = mask.view(np.uint8).reshape(nsb, 8, 2, 2, 6)
        sg = np.zeros((24, nsb, 8, 4), np.uint8)  # [row][sb][i][l]
        sg[:R] = f["signs"].reshape(R, nsb, 8, 4)
        b = np.zeros((nsb, 32, 24), np.uint8)
        s = sg.transpose(1, 2, 3, 0).reshape(nsb, 32, 24)  # [sb][i l][row]
        b[..., :16 + r2] = s[..., :16 + r2]
        c = np.zeros((nsb, 32, 4), np.uint8)
        sc = f["sc"].reshape(R, nsb, 4, 2)
        c[:, :R] = (sc[..., 0] | sc[..., 1] << 4).transpose(1, 0, 2)
        d = np.zeros((2 * -(-R // 2), nsb), np.uint16)
        d[:R] = k23.bf16_u16(f["d"]).astype(np.uint16)
        c[:, R:R + d.shape[0] // 2] = np.ascontiguousarray(d.T).view(np.uint8).reshape(nsb, -1, 4)
        b[..., 20:] = c
        return [np.concatenate([a.reshape(nsb, -1), b.reshape(nsb, -1)], 1).reshape(-1)]

    def unit_bytes(self, fmt: str, rows: int, nsb: int) -> int:
        return UNIT

    sources: ClassVar[dict[str, str]] = {"mmv/IQ3_S": _MMV, "gemm/IQ3_S": _GEMM}

    # GemmQd (prefill): super-block units, rows compacted

    def planar(self, fmt: str, rows: int, kc: int) -> bool:
        return True

    def pass_tiles(self, fmt: str, rows: int, tiles: int) -> bool:
        # (two weight objects of up to 7 units in the dequantizing core's L1: 32 KB, a gate / up pass of a whole SiLU
        # group of 128 rows; MMUL columns <= 128, in whole 32-row blocks: the dequant's 4-panel iterations)
        return tiles == 7 or tiles < 7 and -(-rows * tiles // 16) * 16 % 32 == 0

    def _ws(self, op: object) -> dict[str, int]:
        """Byte offsets in the prefill dequant's scratch (``gemm/IQ3_S``'s): WL the index lists' low bytes, then their 9th
        bits (LS bytes each: whole 64-byte blocks of 32 rows), at WU = WL between k-tiles the scale words (``_sb``); its
        size."""
        LS = Iq3Slot.list_bytes(-(-op.n // 32) * 32)
        WU = WL = 24 * op.n  # (_sb's scratch while no k-tile's lists are live)
        return {"WU": WU, "WL": WL, "LS": LS, "size": WL + max(16 * LS, 4 * op.n)}

    def qd_scratch(self, op: object) -> dict[str, tuple[int, int]]:
        # ws: S, a k-tile's signs, the lists (``_ws``); hb: d; ex: a dq call's codes; the grid's two copies (in two 16 KB
        # regions: ``gemm_qd``)
        return {"ws": (1, self._ws(op)["size"]), "hb": (1, max(2 * op.n, 64)), "ex": (1, 64 * op.nm),
                "tab": (1, TAB_BYTES), "tabb": (1, TAB_BYTES)}

    def qd_kernels(self, op: object, kernel: object, *, B_l1_ty: object, ws_ty: object, hb_ty: object,
                   D_ty: object) -> DqKernels:
        # (_sbp: d pairs; _iq3sg: 32 rows per iteration)
        assert 16 < op.R <= 20 and op.R % 2 == 0 and op.n % 32 == 0 and op.nm % 32 == 0, (op.R, op.n, op.nm)
        # (the constant windows of _prep and _sbg: the weight objects and the scratch 64-byte aligned, as the dequantizing
        # core's layouts place every buffer)
        assert op.unit_pitch % 64 == 0, op.unit_pitch
        tab_ty = np.ndarray[(TAB_BYTES // 4,), np.dtype[np.uint32]]
        ex_ty = np.ndarray[(64 * op.nm,), np.dtype[np.uint8]]
        return DqKernels(hdr=None, sb=kernel("sb", [B_l1_ty, ws_ty, hb_ty, i32]),
                         dq=dq_kernel(op, kernel, B_l1_ty, ws_ty, D_ty, [ex_ty, tab_ty, tab_ty]), units=True,
                         ex_ty=ex_ty, tables=(tab_ty, grid_halves()))

    def qd_template_vars(self, op: object) -> dict[str, object]:
        R, T = op.R, op.tiles

        def seg(q0: int, q1: int) -> list[tuple[int, int, int]]:  # (tile, first row, end) of compacted rows q0 .. q1
            return [(t, max(q0, R * t), min(q1, R * (t + 1))) for t in range(q0 // R, min(T, -(-q1 // R)))]

        def mask(lo: int, hi: int) -> int:  # (bytes lo .. hi - 1 of 64)
            return ((1 << hi) - 1) & ~((1 << lo) - 1)

        return {"UP": op.unit_pitch, "seg": seg, "mask": mask, **self._ws(op)}

    def mmv_slots(self, fmt: str) -> tuple[int, ...]:
        return (4, 8)

    def mmv_supports(self, fmt: str, nsb: int) -> bool:
        return nsb == 4  # (two super-blocks per core)

    def mmv_parts(self, fmt: str, nsb: int) -> tuple[str, ...]:
        return ("mmv/k23", "mmv/g16", "mmv/iq3c", f"mmv/{fmt}")

    def mmv_vars(self, fmt: str, tile_bytes: int, nsb: int) -> dict[str, object]:
        R = self.format(fmt).rows(tile_bytes, nsb)
        assert R == 20, R  # (d of rows 2 s, 2 s + 1 at slot R + s: 16-byte aligned for the kernels' loads)
        return {"R3S": R, "TAB": TAB, "UNIT": UNIT}

    def mmv_table(self, fmt: str) -> np.ndarray:
        return grid_halves()


SLOT = Iq3sSlot()
