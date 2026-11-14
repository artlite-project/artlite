"""IQ2_XXS, IQ2_XS and IQ2_S tiles (66, 74 and 82 B per row and super-block, the GGUF blocks' sizes; R <= 24 rows in 3
row groups of 8, rows past R zero): per super-block a unit of ``UNIT`` B (4 per 1024-column tile, nothing else in the
tile) holding, in panel order p = [32-k group i 8][8-k panel l 4][row group g 3] (96 panels of 8 rows x 8 k: row 8 g + r's
8-k block 4 i + l, one grid entry):

- LO (at 0, 768 B): [p][r 8] the grid indices' low bytes;
- SG (at 768, 768 B): [p][r 8] the block's signs: IQ2_XXS and IQ2_XS the prefix XOR c of its 7-bit sign field (c_j = f_0 ^
  .. ^ f_j: the 8 signs, the 8th the parity of the field's, are c ^ (c << 1)), IQ2_XXS with bit l of its 32-group's scale
  ls at bit 7; IQ2_S the 8 signs (bit j: value j negative);
- SC (at 1536, 192 B; IQ2_XS, IQ2_S): [i][g][r 8] the GGUF scales byte (16-group 2 i's scale in the low nibble, 2 i + 1's
  in the high);
- HB (at 1728, 192 B; IQ2_XS, IQ2_S): [half q 2][bit b 2][word m 12] u32: bit n of word (q, b, m) = bit 8 + b of the
  index of entry 32 m + n of the half's panels (i in 4 q .. 4 q + 3, in panel order);
- D (at 1920, 48 B): d [row] bf16.

w = d (2 s + 1) / 8 q, s the 16-group's scale (IQ2_XXS: its 32-group's), q = the index's grid value (``gguf_model.iq2_grid``:
8 values per index, in {8, 25, 43}) negated where its sign is set.

The grids as gather tables (``grid_words``): entry i = [A | B] (16 bits), A's bit k = value k of the entry >= 25, B's =
value k is 43, so that g = 43 where B, else 25 where A, else 8: an 8x8 panel is 8 entries of 4x16 gathers, their A and B
bytes as 64-bit masks by one byte transpose (``T8_8x8``), g by two selects, the signs by a masked subtract. IQ2_XXS 1 KB
and IQ2_XS 2 KB per table copy (``common.TAB``: each core half's two copies); IQ2_S's 4 KB, its halves (index bit 9) one in
each copy slot (``grid_words``' rows)."""

from typing import ClassVar

import numpy as np

from .common import Slot, TileFormat, _bf16_bytes

UNIT = 2304  # bytes of a super-block's unit
ROWS = 24  # rows of a tile (3 row groups)
LO, SG, SC, HB, D = 0, 768, 1536, 1728, 1920  # the unit's parts (module docstring)


def grid_words(fmt: str) -> np.ndarray:
    """The grid as gather tables (uint32 words, ``Slot.mmv_table``): entry i = A | B << 8 (module docstring) at byte 4 i of
    a table whose 32-byte groups of 16 entries are stored twice; IQ2_S: (2, 512) words, its halves (entries 512 h ..) in
    copy h."""
    from ..gguf_model import iq2_grid

    g = iq2_grid(fmt).astype(np.int32)
    bits = (1 << np.arange(8)).astype(np.uint16)
    e = ((g >= 25) * bits).sum(-1, dtype=np.uint16) | (((g == 43) * bits).sum(-1, dtype=np.uint16) << 8)
    tables = e.reshape(-1, 512) if e.size > 512 else e.reshape(1, -1)
    out = np.stack([np.concatenate([t.reshape(-1, 16)] * 2, 1).reshape(-1).view(np.uint32) for t in tables])
    return out if len(out) > 1 else out[0]


def prefix_xor7(f: np.ndarray) -> np.ndarray:
    """7-bit sign fields -> their prefix XOR c (uint8, bit j = f_0 ^ .. ^ f_j)."""
    c = f.astype(np.uint8)
    for k in (1, 2, 4):
        c ^= c << k
    return c & 0x7F


def panel_order(a: np.ndarray, nsb: int) -> np.ndarray:
    """(24, nsb, 8 i, 4 l) per-row values of the 8-k blocks -> (nsb, 96 panels x 8 rows) in panel order."""
    return a.reshape(3, 8, nsb, 8, 4).transpose(2, 3, 4, 0, 1).reshape(nsb, 768)


class Iq2Slot(Slot):
    formats = (TileFormat("IQ2_XXS", 14, 66, ROWS, 1, 66, 0, planes=2, unit_fixed=UNIT),
               TileFormat("IQ2_XS", 15, 74, ROWS, 1, 74, 0, planes=2, unit_fixed=UNIT),
               TileFormat("IQ2_S", 16, 82, ROWS, 1, 82, 0, planes=2, unit_fixed=UNIT))
    sources: ClassVar[dict[str, str]] = {}

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert R <= ROWS, R
        idx = np.zeros((ROWS, nsb, 8, 4), np.uint16)
        idx[:R] = f["idx"].reshape(R, nsb, 8, 4)
        sg = np.zeros((ROWS, nsb, 8, 4), np.uint8)
        sg[:R] = f["signs"].reshape(R, nsb, 8, 4)
        units = np.zeros((nsb, UNIT), np.uint8)
        units[:, LO:LO + 768] = panel_order(idx & 0xFF, nsb)
        if fmt != "IQ2_S":
            sg = prefix_xor7(sg)
        if fmt == "IQ2_XXS":
            ls = np.zeros((ROWS, nsb, 8), np.uint8)
            ls[:R] = f["ls"].reshape(R, nsb, 8)
            sg |= ((ls[..., None] >> np.arange(4, dtype=np.uint8)) & 1) << 7
        units[:, SG:SG + 768] = panel_order(sg, nsb)
        if fmt != "IQ2_XXS":
            sc = np.zeros((ROWS, nsb, 8, 2), np.uint8)
            sc[:R] = f["sc"].reshape(R, nsb, 8, 2)
            units[:, SC:SC + 192] = (sc[..., 0] | sc[..., 1] << 4).reshape(3, 8, nsb, 8).transpose(2, 3, 0, 1).reshape(nsb, 192)
            hb = np.stack([panel_order((idx >> (8 + b)) & 1, nsb) for b in range(2)], 1)  # (nsb, b, 768)
            words = (hb.reshape(nsb, 2, 2, 12, 32).astype(np.uint32) << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)
            units[:, HB:HB + 192] = np.ascontiguousarray(words.transpose(0, 2, 1, 3)).view(np.uint8).reshape(nsb, 192)
        d = np.zeros((ROWS, nsb), np.float32)
        d[:R] = f["d"]
        units[:, D:D + 48] = _bf16_bytes(np.ascontiguousarray(d.T)[..., None]).reshape(nsb, 48)
        return [units.reshape(-1)]


SLOT = Iq2Slot()
