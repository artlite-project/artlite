"""The weight image of a Llama 3.x model (``decode.image.TileImage``): the file's tile formats over K chunks of 1024
columns (``KC``, every model and format: 4 super-blocks of 256 columns), objects of the size that streams the fewest
bytes for the file's formats (``formats.choose_tile_bytes``: 9280 for Q4_K_M, Q4_K_S, Q5_K_M and Q4_0).

Column-major (as ``gemma4.image``): a column's regions in step order, each normed phase's norm-weight objects (data
region "<name>_norm": the RMSNorm weights, ``decode.design.norm_objects``) before its tiles, so that a decode step
streams a column's weights between two attentions as one run (8B: in two layout blocks of layers, each under 3 GiB, the
prefill GEMMs' weight arguments within 4 GB: ``decode.image.BLOCK_BYTES``).

Per-column rows (layer L, column c = KV head c, qh query heads per KV head):

- "qkv": block "q" = the rows of query heads qh c .. qh c + qh - 1, "k" and "v" = KV head c's. llama.cpp's converter
  interleaves a head's Q / K rows by rotary pair: the blocks take them in rotate-half order (pair i = dims i,
  i + head_dim / 2; ``gguf_model.unpermute_rows``), the other models' RoPE;
- "o" (``attn_output``), "down": rows hidden / 8 c ..; "gate_up": ``decode.image.gate_up_rows``;
- "lm": rows vocab / 8 c .. of ``output`` (the tied ``token_embd`` in 1B and 3B), the vocabulary padded to whole C
  objects per column (128256 -> 128512; padding rows -1: zero tiles).

down's K chunks go in groups of the overlay's K group (``decode.design.MAX_K``: 1B / 3B 2 groups of 4, 8B 7 of 2).
"""

import numpy as np

from artlite.decode.design import C_LEN, MAX_K, NX_PER_NORM, OBJ_K, norm_objects
from artlite.decode.image import TileImage, column_rows, gate_up_rows
from artlite.gguf_model import GGUFFile, unpermute_rows
from artlite.model_config import LlamaConfig
from artlite.operators.design import COLS

KC = 1024  # K chunk of a tile
NORMS = {"qkv": "attn_norm", "gate_up": "ffn_norm"}  # a layer's normed phases: their RMSNorm weights


class LlamaImage(TileImage):
    """The weight image of a Llama GGUF file (layers: the layers to lay out, default all; vocab: LM rows, default the
    padded vocabulary; tile_bytes: the object size, default chosen from the file's formats)."""

    def __init__(self, gguf: GGUFFile, cfg: LlamaConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0):
        self.cfg = cfg = cfg or LlamaConfig.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        H, I = cfg.hidden_size, cfg.intermediate_size
        if vocab is None:
            vocab = -(-cfg.vocab_size // (COLS * C_LEN)) * COLS * C_LEN
        super().__init__(gguf, kc=KC, payload=MAX_K, tile_bytes=tile_bytes, vocab=vocab, column_major=True)
        assert cfg.num_key_value_heads == COLS, f"one KV head per column: {cfg.num_key_value_heads} KV heads"
        assert H % KC == 0 and I % KC == 0 and I % (COLS * 64) == 0 and self.vocab % COLS == 0
        self.n_norm = -(-H // (NX_PER_NORM * OBJ_K))  # norm-weight objects of a normed phase
        for L in self.layers:
            self._layer(L)
        lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        rows = column_rows(self.vocab)
        self._norm(-1, "lm")
        self._add(-1, "lm", [("lm", (lm,), np.where(rows < cfg.vocab_size, rows, -1))], H)
        self._place()

    def _norm(self, L: int, name: str) -> None:
        self._add_data(L, name + "_norm", self.n_norm * self.tile_bytes)

    def norm_objects(self, L: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's RMSNorm weights (bf16)."""
        w = self.gguf.tensor("output_norm.weight" if L < 0 else f"blk.{L}.{NORMS[name]}.weight").dequantize()
        return norm_objects(w.numpy().reshape(-1), self.tile_bytes).view(np.uint8)

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, I, hd, nh = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_attention_heads
        q = unpermute_rows(nh, nh * hd).reshape(COLS, cfg.q_per_kv * hd)
        k = unpermute_rows(COLS, COLS * hd).reshape(COLS, hd)
        self._norm(L, "qkv")
        self._add(L, "qkv", [("q", (b + "attn_q.weight",), q), ("k", (b + "attn_k.weight",), k),
                             ("v", (b + "attn_v.weight",), column_rows(COLS * hd))], H)
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], nh * hd)
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._norm(L, "gate_up")
        self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
