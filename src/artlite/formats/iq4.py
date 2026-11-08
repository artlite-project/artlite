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
"""

import numpy as np

from . import k23
from .common import Slot, TileFormat


def _pairs(nib: np.ndarray) -> np.ndarray:
    """(..., 2 n) nibbles -> (..., n) bytes, the first of a pair low."""
    return (nib[..., 0::2] | nib[..., 1::2] << 4).astype(np.uint8)


def codes(q: np.ndarray, R: int, nsb: int) -> np.ndarray:
    """(R, 256 nsb) codes -> (nsb, 128 R) bytes [sb][i][s][g][r][4 B] (module docstring)."""
    G = R // 8
    b = _pairs(q.reshape(R, nsb, 8, 4, 8))  # [row][sb][i][s][4 B]
    return np.ascontiguousarray(b.reshape(G, 8, nsb, 8, 4, 4).transpose(2, 3, 4, 0, 1, 5)).reshape(nsb, -1)


class Iq4Slot(Slot):
    formats = (TileFormat("IQ4_NL", 11, 144, 16, 8, 144, 0, planes=4, unit_align=64, mmv="iq4nl"),
               TileFormat("IQ4_XS", 12, 136, 16, 8, 136, 0, planes=4, unit_align=64, mmv="iq4xs"))

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


SLOT = Iq4Slot()
