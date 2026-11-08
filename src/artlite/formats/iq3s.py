"""IQ3_S tiles (R = 17 .. 20 rows in 3 row groups of 8, the third of r2 = R - 16 rows; 110 B per row and super-block,
the GGUF block's size): per super-block a unit of ``UNIT`` B (4 per 1024-column tile, nothing else in the tile) holding

- A (1536 B): the grid indices' low bytes as records [32-k group i 8][panel pair h 2][8-k panel l' 2] of 48 B: rows 0
  .. 7 (row r's two indices of panel l = 2 h + l' at 2 r, 2 r + 1), rows 8 .. 15, rows 16 .. R - 1 (2 r2 B), zeros, then
  at byte 42 the panels' 9th index bits, a 16-bit mask per row group g (bit 2 r + j: row 8 g + r's index j);
- B (768 B): the signs as records [i 8][panel l 4] of 24 B: rows 0 .. 7's sign bytes (bit j: value 8 l + j of group i
  negative), rows 8 .. 15, rows 16 .. R - 1, zeros, then at byte 20 a 4-byte slot of C;
- C (the 32 slots of B, slot n in record n): slot r < R row r's scale bytes (the GGUF block's: groups 2 j, 2 j + 1 in
  byte j's low and high nibble), slot R + s rows 2 s and 2 s + 1's d (bf16), zeros past them.

So each weight panel (8 rows x 8 k: 16 indices) has its index bytes in 16 contiguous bytes and its signs in 8 (rows past
R: other bytes, whose panel rows the kernels drop). w = d (1 + 2 sc) q, q = the 9-bit index's grid value
(``gguf_model.iq3s_grid``: 4 values per index, in {1, 3, .., 15}) negated where its sign is set."""

import numpy as np

from . import k23
from .common import Slot, TileFormat

UNIT = 2304  # bytes of a super-block's unit (A + B)


def grid_halves() -> np.ndarray:
    """The grid as 16-bit entries for the decode's 4x16 gathers (``TAB_BYTES`` / 4 uint32): entry i (its 4 values,
    one nibble each, value j in nibble j) at byte 4 i of a table whose 32-byte groups of 16 entries are stored twice."""
    from ..gguf_model import iq3s_grid

    g = iq3s_grid().astype(np.uint16)
    e = g[:, 0] | g[:, 1] << 4 | g[:, 2] << 8 | g[:, 3] << 12
    return np.concatenate([e.reshape(32, 16)] * 2, 1).reshape(-1).view(np.uint32)


TAB_BYTES = 2048


class Iq3sSlot(Slot):
    formats = (TileFormat("IQ3_S", 13, 110, 20, 1, 110, 0, planes=3, unit_fixed=UNIT, mmv="iq3s"),)

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
        a[..., 42:48] = mask.view(np.uint8).reshape(nsb, 8, 2, 2, 6)
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


SLOT = Iq3sSlot()
