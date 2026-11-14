"""The weight image of an LFM3 model (``decode.image.TileImage``): the file's tile formats over K chunks of 1024 columns
(``KC``), objects of the size that streams the fewest bytes for the file's formats (``formats.choose_tile_bytes``). The
decode's GEMV (``decode.gemv``) and the prefill's GEMMs read the tiles in place.

Column-major (as ``llama.image``): a column's regions in step order, each normed phase's norm-weight objects (data
region "<name>_norm") before its tiles, so that a decode step streams a column's weights between two operator cores
(attention, conv) as one run.

Per-column rows (layer L, column c; an attention layer's KV head h = c / 2, j = c % 2):

- conv layers: "in" (``shortconv.in_proj``) = the column's 256 channels 256 c .. of each of B, C and x (in_proj's rows
  [B; C; x]: blocks "b", "c", "x"), "out" (``shortconv.out_proj``): rows hidden / 8 c ..;
- attention layers: "qkv" = block "q", the rows of query heads 4 h + 2 j, 4 h + 2 j + 1, blocks "k" and "v", half j of
  K_h's rows and of V_h's (dims 64 j ..: K and V may be of two formats), block "g", the two heads' rows of
  ``attn_gate`` (padded to a row tile); "o" (``attn_output``): rows hidden / 8 c ..; the rows of Q and K are the
  file's (NeoX pairs: dims i, i + head_dim / 2);
- every layer: "gate_up" (``decode.image.gate_up_rows``), "down": rows hidden / 8 c .., its K (the ffn, 7680) padded to
  whole K chunks with zero super-blocks (8192);
- "lm": rows vocab / 8 c .. of ``token_embd`` (the tied LM head; 128000: 250 C objects per column).

Normed phases ("in", "qkv", "gate_up", "lm") are headers of the post-norm residual update (``decode.protocol.F_POST``:
the sandwich norms), their norm-weight objects [w1 | w_post] per 8 input objects: "in" / "qkv" (w1 = ``attn_norm``,
w_post = the previous layer's ``post_ffw_norm``; the first layer's: a plain input, w1 alone), "gate_up" (``ffn_norm``,
``post_attention_norm``), "lm" (``output_norm``, the last layer's ``post_ffw_norm``).
"""

import numpy as np

from artlite.decode.image import NormedImage, column_rows, gate_up_rows
from artlite.decode.protocol import MAX_K, NX_PER_NORM, POST_K
from artlite.gguf_model import GGUFFile
from artlite.model_config import LfmConfig
from artlite.operators.design import COLS

KC = 1024  # K chunk of a tile
CH = 256  # conv channels per column (hidden / COLS)
NORMED = ("in", "qkv", "gate_up", "lm")


def qkv_rows(cfg: LfmConfig) -> dict[str, np.ndarray]:
    """(COLS, n) rows of the qkv phase's blocks: "q" of ``attn_q``, "k" / "v" of ``attn_k`` / ``attn_v``, "g" of
    ``attn_gate``."""
    hd, nkv = cfg.head_dim, cfg.num_key_value_heads
    per = COLS // nkv  # (columns per KV head)
    qh = cfg.q_per_kv // per  # (query heads per column)
    h, j = np.arange(COLS) // per, np.arange(COLS) % per
    heads = (cfg.q_per_kv * h + qh * j)[:, None] + np.arange(qh)[None]  # (COLS, qh)
    q = (hd * heads[..., None] + np.arange(hd)).reshape(COLS, -1)
    half = (hd * h + hd // per * j)[:, None] + np.arange(hd // per)[None]
    return {"q": q, "k": half, "v": half, "g": heads}


class LfmImage(NormedImage):
    """The weight image of an LFM3 GGUF file (layers: the layers to lay out, default all; vocab: LM rows, default the
    vocabulary; tile_bytes: the object size, default chosen from the file's formats)."""

    def __init__(self, gguf: GGUFFile, cfg: LfmConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0):
        cfg = cfg or LfmConfig.from_gguf(gguf)
        H = cfg.hidden_size
        super().__init__(gguf, cfg, layers=layers, kc=KC, payload=MAX_K, tile_bytes=tile_bytes,
                         vocab=cfg.vocab_size if vocab is None else vocab)
        assert H == COLS * CH and cfg.num_key_value_heads * 2 == COLS and self.vocab % (COLS * 64) == 0, cfg
        assert NX_PER_NORM * POST_K >= H, "a header's [w1 | w_post] in one norm-weight object per 8 input objects"
        for L in self.layers:
            self._layer(L)
        self._norm(-1, "lm")
        self._add(-1, "lm", [("lm", ("token_embd.weight",), column_rows(self.vocab))], H)
        self._place()

    # ------------------------------------------------------------------
    # Norm-weight objects
    # ------------------------------------------------------------------

    def norms(self, L: int, name: str) -> tuple[str, str | None]:
        """(w1, w_post) tensor names of a normed phase; w_post None: a plain normed input (the slice's first layer),
        else a header (``decode.protocol.F_POST``)."""
        b = f"blk.{L}."
        if name in ("in", "qkv"):
            if L == self.layers[0]:
                return b + "attn_norm.weight", None
            return b + "attn_norm.weight", f"blk.{self.layers[self.layers.index(L) - 1]}.post_ffw_norm.weight"
        if name == "gate_up":
            return b + "ffn_norm.weight", b + "post_attention_norm.weight"
        assert name == "lm", name
        return "output_norm.weight", f"blk.{self.layers[-1]}.post_ffw_norm.weight"

    # ------------------------------------------------------------------
    # Regions
    # ------------------------------------------------------------------

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, I = cfg.hidden_size, cfg.intermediate_size
        if cfg.kinds[L] == "conv":
            ch = CH * np.arange(COLS)[:, None] + np.arange(CH)[None]
            self._norm(L, "in")
            self._add(L, "in", [(n, (b + "shortconv.in_proj.weight",), ch + H * i) for i, n in enumerate("bcx")], H)
            self._add(L, "out", [("out", (b + "shortconv.out_proj.weight",), column_rows(H))], H)
        else:
            rows = qkv_rows(cfg)
            self._norm(L, "qkv")
            self._add(L, "qkv", [(n, (b + f"attn_{n}.weight",), rows[n]) for n in ("q", "k", "v")]
                      + [("g", (b + "attn_gate.weight",), rows["g"])], H)
            self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], cfg.num_attention_heads * cfg.head_dim)
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._norm(L, "gate_up")
        self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
