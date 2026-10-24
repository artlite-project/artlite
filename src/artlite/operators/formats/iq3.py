"""IQ3_XXS tiles (98 B per row and super-block, the GGUF block's size), word-major as Q3_K's (``k23``): header words
[sb pair][row]: bf16 d of both super-blocks, then per super-block 24 code words [k-tile kt 4][word j 6][row]: j < 4 the
grid indices of k 64 kt + 16 j .. + 15 (byte b: the index of k + 4 b .. + 3, as the GGUF block's qs), j = 4, 5 the
signs and scale of the 32-k group i = 2 kt + j - 4: byte l = its 7-bit sign field l (k 32 i + 8 l ..; the 8th sign:
their parity) | bit l of its scale ls at bit 7.

w = d (2 ls + 1) / 4 q, q = the index's grid value (``gguf_model.iq3xxs_grid``: 4 values per index, in {4 .. 62})
negated where its sign is set."""

import numpy as np

from . import k23
from .common import Slot, TileFormat


def ss_words(signs: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """(R, n groups x 4) 7-bit sign fields, (R, n) scales -> (R, n) uint32 words: byte l = field l | bit l of ls << 7."""
    s = signs.reshape(*ls.shape, 4).astype(np.uint32)
    bits = (ls[..., None].astype(np.uint32) >> np.arange(4, dtype=np.uint32)) & 1
    return ((s | bits << 7) << (8 * np.arange(4, dtype=np.uint32))).sum(-1, dtype=np.uint32)


class Iq3Slot(Slot):
    formats = (TileFormat("IQ3_XXS", 10, 98, 32, 1, 96, 2, planes=3, word_major=True),)

    def pack(self, fmt: str, f: dict[str, np.ndarray], R: int, nsb: int) -> list[np.ndarray]:
        assert nsb % 2 == 0, nsb
        d = k23.bf16_u16(f["d"])  # (R, nsb)
        hw = d[:, 0::2] | (d[:, 1::2] << 16)
        idx = np.ascontiguousarray(f["idx"]).reshape(R, nsb, 4, 4, 4).view(np.uint32)[..., 0]  # [sb][kt][j] (bytes LE)
        ss = ss_words(f["signs"], f["ls"]).reshape(R, nsb, 4, 2)  # [sb][kt][j - 4]
        return k23.tile_parts(hw, np.concatenate([idx, ss], -1), R, nsb)


SLOT = Iq3Slot()
