"""The DFlash2 drafter's weights (``dflash.DFlashOverlay``): its matrices' tile regions in a ``TileImage``."""

import numpy as np

from artlite.decode.image import Region, TileImage, column_rows, gate_up_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import DFlashConfig
from artlite.operators.design import COLS

from .attn import OBJ_BYTES
from .dflash_core import GROUPS, QC
from .image import KC, payload


def conv_rows(c: int) -> np.ndarray:
    """A conv projection's rows of column c: [side 2][tap 2] the column's GROUPS conv groups."""
    return np.concatenate([q * COLS * GROUPS + GROUPS * c + np.arange(GROUPS) for q in range(4)])


class DFlashImage(TileImage):
    """The drafter's weights, its matrices' tiles (the selector's tables stay in the file, read by row): fc, per layer ain,
    aqkv ([q | K | V] of the column's 4 query heads and KV head), aout, fin, gu, down, then gate (``selector_hidden``). An injection's K and V
    are aqkv's rows (``ikv``: a view of its tiles)."""

    def __init__(self, gguf: GGUFFile, cfg: DFlashConfig):
        self.cfg = cfg
        H, I, d = cfg.hidden, cfg.ffn, cfg.head_dim
        super().__init__(gguf, kc=KC, payload=payload(H), tile_bytes=OBJ_BYTES, vocab=0)
        self._add(-2, "fc", [("fc", ("fc.weight",), column_rows(H))], len(cfg.taps) * H)
        q = np.stack([QC * d * c + np.arange(QC * d) for c in range(COLS)])
        kv = np.stack([d * c + np.arange(d) for c in range(COLS)])
        conv = np.stack([conv_rows(c) for c in range(COLS)])
        for L in range(cfg.layers):
            b = f"blk.{L}."
            self._add(L, "ain", [("dyn", (b + "attn_conv_proj.weight",), conv)], H)
            self._add(L, "aqkv", [("q", (b + "attn_q.weight",), q), ("k", (b + "attn_k.weight",), kv), ("v", (b + "attn_v.weight",), kv)], H)
            self._add(L, "aout", [("out", (b + "attn_output.weight",), column_rows(H))], cfg.heads * d)
            self._add(L, "fin", [("dyn", (b + "ffn_conv_proj.weight",), conv)], H)
            R = self.R[self.fmt(b + "ffn_gate.weight")]
            self._add(L, "gu", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
            self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
        self._add(-1, "gate", [("gate", ("selector_hidden.weight",), column_rows(cfg.rank))], H)
        self._place()
        self.ikv: dict[int, Region] = {}
        for L in range(cfg.layers):
            r = self.region(L, "aqkv")
            blocks = r.blocks[1:]
            rt = r.blocks[0].tiles
            self.ikv[L] = Region(L, "ikv", blocks, r.n_kc, r.kc_group, r.tile_bytes, r.kc,
                                 offset=r.offset + rt * r.n_kc * r.tile_bytes, col_stride=r.col_stride)

    def formats(self) -> set[str]:
        return {b.fmt for r in self.regions for b in r.blocks}
