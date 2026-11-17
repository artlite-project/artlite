"""The DSpark drafter's weights (``dspark.DSparkOverlay``): its matrices' tiles in a ``TileImage`` of Spark's (``image``:
K chunks of 1024 columns, tiles of ``decode.attn256.OBJ_BYTES``; the BF16 files' matrices as BFP16 tiles), region-major.

- "fc" of layer i (tap i): fc's columns of tap i (the file's ``fc.weight`` read as rows of hidden columns: row 5 r + i
  of the view is row r's tap i, ``TileImage.view``), rows hidden / 8 c .., each normed by its row of ``fc_norm``;
- per layer: "attn_in" (blocks "q", "k", "v", "g": Spark's ``image.attn_rows``), "attn_out", "gate_up" (GeGLU's gate and
  up rows), "down", as the model's; an injection's K and V are attn_in's rows (``ikv``: a view of the "k" and "v" blocks'
  tiles, normed by ``enc.output_norm``).

The head is the model's (``decode.head.HeadView`` of its image: ``head``, set by the overlay)."""

import numpy as np

from artlite.decode.attn256 import OBJ_BYTES
from artlite.decode.image import Region, TileImage, column_rows, gate_up_rows
from artlite.decode.protocol import MAX_K
from artlite.gguf_model import GGUFFile
from artlite.model_config import DSparkConfig
from artlite.operators.design import COLS

from .image import KC

NORMS = {"attn_in": "attn_norm", "gate_up": "ffn_norm"}  # (a layer's normed phases; fc's: fc_norm, ikv's: enc, lm's: output)


def attn_rows(cfg: DSparkConfig) -> dict[str, np.ndarray]:
    """(COLS, n) rows of attn_in's blocks "q", "k", "v" (``attn_q``, ``attn_k``, ``attn_v``) and "g" (``attn_gate``):
    the column's query heads, its part of KV head c // NPH, its heads' gates (Spark's ``image.attn_rows``)."""
    qc, per = cfg.heads // COLS, cfg.kv_heads * cfg.head_dim // COLS
    c = np.arange(COLS)[:, None]
    return {"q": qc * cfg.head_dim * c + np.arange(qc * cfg.head_dim)[None], "k": per * c + np.arange(per)[None],
            "v": per * c + np.arange(per)[None], "g": qc * c + np.arange(qc)[None]}


class DSparkImage(TileImage):
    """A DSpark file's weights (module docstring). ``head``: the model's head region the drafter's LM phase reads."""

    head: Region | None = None

    def __init__(self, gguf: GGUFFile, cfg: DSparkConfig):
        self.cfg = cfg
        H, I = cfg.hidden, cfg.ffn
        self.layers = list(range(cfg.layers))
        super().__init__(gguf, kc=KC, payload=MAX_K, tile_bytes=OBJ_BYTES, vocab=0)
        assert H % 256 == 0 and gguf.tensor("fc.weight").cols == len(cfg.taps) * H
        self.view("fc.weight", H // 256)
        n = len(cfg.taps)
        for i in range(n):
            self._add(i, "fc", [("fc", ("fc.weight",), column_rows(H) * n + i)], H)
        rows = attn_rows(cfg)
        for L in self.layers:
            b = f"blk.{L}."
            self._add(L, "attn_in", [(k, (b + f"attn_{k}.weight" if k != "g" else b + "attn_gate.weight",), rows[k])
                                     for k in ("q", "k", "v", "g")], H)
            self._add(L, "attn_out", [("out", (b + "attn_output.weight",), column_rows(H))], cfg.heads * cfg.head_dim)
            R = self.R[self.fmt(b + "ffn_gate.weight")]
            self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
            self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
        self._place()
        self.ikv: dict[int, Region] = {}
        for L in self.layers:
            r = self.region(L, "attn_in")
            assert r.kc_group == r.n_kc and [b.name for b in r.blocks] == ["q", "k", "v", "g"]
            q = r.blocks[0].tiles
            self.ikv[L] = Region(L, "ikv", r.blocks[1:3], r.n_kc, r.kc_group, r.tile_bytes, r.kc,
                                 offset=r.offset + q * r.n_kc * r.tile_bytes, col_stride=r.col_stride)

    def region(self, layer: int, name: str) -> Region:
        if name == "ikv":
            return self.ikv[layer]
        if (layer, name) == (-1, "lm"):
            assert self.head is not None, "the model's head: DSparkOverlay"
            return self.head
        return super().region(layer, name)

    def gu_rows(self) -> set[int]:
        """The rows per tile of the gate / up tiles (the model's design runs them: ``SparkOverlay(gu=)``)."""
        return {b.R for r in self.regions if r.name == "gate_up" for b in r.blocks}
