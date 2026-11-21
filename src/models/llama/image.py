"""The weight image of a Llama 3.x or Qwen3 model (``decode.image.TileImage``): the file's tile formats over K chunks of
1024 columns (``KC``, every model and format: 4 super-blocks of 256 columns), objects of the size that streams the
fewest bytes for the file's formats (``formats.choose_tile_bytes``: 9280 for Q4_K_M, Q4_K_S, Q5_K_M and Q4_0, 9216 for
Q8_0 alone).

A hidden size half a chunk short of whole chunks (the Qwen3-4B family's 2560: qkv, gate_up and the LM head, ``TAILS``)
makes those regions ``decode.image.Region.half`` regions (``tails``): their inputs are the hidden size, the prefill's
GEMMs read 2.5 of a row tile's 3 tiles. A model that decodes (``pairs``) holds gate / up's 16-row Q4_K / Q4_0 blocks
(which the prefill reads in pass pairs: ``prefill.gemm.pair_tails``) and the LM head's (which no prefill reads) in pairs
of row tiles that share the tile of the last 512 columns; a readout model's (its prompts only) and the other blocks'
tiles keep that tile's second half as zero super-blocks. An ffn of part of a chunk (the 4B's 9728: down) is padded to
whole chunks with zero super-blocks.

Column-major (as ``gemma4.image``): a column's regions in step order, each normed phase's norm-weight objects (data
region "<name>_norm": the RMSNorm weights, ``decode.image.NormedImage``) before its tiles, so that a decode step
streams a column's weights between two attentions as one run (8B: in two layout blocks of layers, each under 3 GiB, the
prefill GEMMs' weight arguments within 4 GB: ``decode.image.BLOCK_BYTES``).

Per-column rows (layer L, column c = KV head c, qh query heads per KV head):

- "qkv": block "q" = the rows of query heads qh c .. qh c + qh - 1, "k" and "v" = KV head c's. llama.cpp's converter
  interleaves a Llama head's Q / K rows by rotary pair: the blocks take them in rotate-half order (pair i = dims i,
  i + head_dim / 2; ``gguf_model.unpermute_rows``), the other models' RoPE (Qwen3's rows are in that order:
  ``LlamaConfig.neox``);
- "o" (``attn_output``), "down": rows hidden / 8 c ..; "gate_up": ``decode.image.gate_up_rows``;
- "lm": rows vocab / 8 c .. of ``output`` (the tied ``token_embd`` in 1B and 3B), the vocabulary padded to whole C
  objects per column (128256 -> 128512; padding rows -1: zero tiles).

down's K chunks go in groups of the overlay's K group (``decode.protocol.MAX_K``: 1B / 3B 2 groups of 4, 8B 7 of 2).

A PARO checkpoint's linears (``paro``: their inputs rotated, the decode design's "Rotated inputs") take inputs of their
own: per layer the regions "q", "k", "v", "o", "gate", "up", "down" (each one block; q / k / v / gate / up without norm
weights: a norm's weight is in its consumers' rotations), and the columns' rotation objects (``protocol.rot_objects``:
column c's groups of each rotation, in the order the cores rotate them) in data regions on its A run: "rot_in" (before
the first layer: its q, k, v copies of the residual, rotated from the embeddings), "rot_o" (the attention's output: o's
rotation of the column's heads), "o_rot" (after o's tiles: gate's and up's copies of the residual slice), "rot_down"
(after up: down's rotation of the column's h), "down_rot" (after down: the next layer's q, k, v copies; none after the
last, whose residual goes to the LM head as it is).
"""

import numpy as np

from artlite.decode.image import NormedImage, column_rows, gate_up_rows
from artlite.decode.protocol import C_LEN, MAX_K
from artlite.decode.rotated import RotatedImage, column_groups
from artlite.decode.screen import plane, screened
from artlite.gguf_model import GGUFFile, unpermute_rows
from artlite.model_config import LlamaConfig
from artlite.operators.design import COLS
from artlite.paro import ParoTensor
from artlite.prefill.gemm import pair_tails

KC = 1024  # K chunk of a tile
NORMS = {"qkv": "attn_norm", "gate_up": "ffn_norm"}  # a layer's normed phases: their RMSNorm weights
TAILS = frozenset({"qkv", "gate_up", "lm"})  # (``tails``: the regions of the hidden's input)


class LlamaImage(RotatedImage, NormedImage):
    """The weight image of a Llama GGUF file (layers: the layers to lay out, default all; vocab: LM rows, default the
    padded vocabulary; tile_bytes: the object size, default chosen from the file's formats; tails: the hidden's regions
    half regions where it is half a chunk short, pairs: their pairs of row tiles, module docstring)."""

    def __init__(self, gguf: GGUFFile, cfg: LlamaConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0, tails: bool = True, pairs: bool = True):
        cfg = cfg or LlamaConfig.from_gguf(gguf)
        H, I = cfg.hidden_size, cfg.intermediate_size
        if vocab is None:
            vocab = -(-cfg.vocab_size // (COLS * C_LEN)) * COLS * C_LEN
        self.paired = pairs
        L0 = (list(range(cfg.num_hidden_layers)) if layers is None else layers)[0]
        self.paro = isinstance(gguf.tensor(f"blk.{L0}.attn_q.weight"), ParoTensor)  # (module docstring)
        super().__init__(gguf, cfg, layers=layers, kc=KC, payload=MAX_K, tile_bytes=tile_bytes, vocab=vocab,
                         tails=TAILS if tails else frozenset())
        assert cfg.num_key_value_heads == COLS, f"one KV head per column: {cfg.num_key_value_heads} KV heads"
        assert H % (COLS * 64) == 0 and I % (COLS * 64) == 0 and self.vocab % COLS == 0
        self._rot_init()
        for L in self.layers:
            if self.paro:
                self._paro_layer(L)
            else:
                self._layer(L)
        self.lm = lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        rows = column_rows(self.vocab)
        self._norm(-1, "lm")
        # (a Q6_K head as its high plane, an F16 / BF16 one as a Q4_0 plane, the logits that matter exactly on the host:
        # ``decode.screen``)
        self.screen = screened(self.qtype(lm), self.formats())
        self._add(-1, "lm", [("lm", (lm,), np.where(rows < cfg.vocab_size, rows, -1), *((plane(self.qtype(lm)),) if self.screen else ()))], H)
        self._place()

    def pairs(self, region: str, block: str, fmt: str, R: int) -> bool:
        """(``paired``) The LM head's (the prefill reads none), the regions' the prefill reads in pairs
        (``pair_tails``)."""
        return self.paired and (region == "lm" or pair_tails(region, fmt, R))

    def norms(self, L: int, name: str) -> tuple[str, None]:
        """(w1, None) tensor names of a normed phase: plain inputs (pre-norm residual updates)."""
        return "output_norm.weight" if L < 0 else f"blk.{L}.{NORMS[name]}.weight", None

    def data_objects(self, layer: int, name: str) -> np.ndarray:
        if self.rot_region(layer, name):
            return self._rot_objs[(layer, name)]
        return super().data_objects(layer, name)

    def _paro_layer(self, L: int) -> None:
        """A PARO layer's regions (module docstring)."""
        cfg, b = self.cfg, f"blk.{L}."
        H, I, hd, nh = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_attention_heads
        q = (np.arange(nh * hd) if cfg.neox else unpermute_rows(nh, nh * hd)).reshape(COLS, cfg.q_per_kv * hd)
        k = (np.arange(COLS * hd) if cfg.neox else unpermute_rows(COLS, COLS * hd)).reshape(COLS, hd)
        h8 = H // COLS
        if L == self.layers[0]:
            self.rot_data(L, "rot_in", self._qkv_rot(L))
        self._add(L, "q", [("q", (b + "attn_q.weight",), q)], H)
        self._add(L, "k", [("k", (b + "attn_k.weight",), k)], H)
        self._add(L, "v", [("v", (b + "attn_v.weight",), column_rows(COLS * hd))], H)
        self.rot_data(L, "rot_o", [(b + "attn_output.weight", None, column_groups(0, cfg.q_per_kv * hd))])
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], nh * hd)
        g = self._vec(f"{b}ffn_norm.weight")
        self.rot_data(L, "o_rot", [(b + "ffn_gate.weight", g, column_groups(0, h8)), (b + "ffn_up.weight", g, column_groups(0, h8))])
        self._add(L, "gate", [("gate", (b + "ffn_gate.weight",), column_rows(I))], H)
        self._add(L, "up", [("up", (b + "ffn_up.weight",), column_rows(I))], H)
        self.rot_data(L, "rot_down", [(b + "ffn_down.weight", None, column_groups(0, I // COLS))])
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
        i = self.layers.index(L)
        if i + 1 < len(self.layers):
            self.rot_data(L, "down_rot", self._qkv_rot(self.layers[i + 1]))

    def tensor_name(self, L: int, name: str) -> str:
        """(PARO) The tensor of a layer's region."""
        return f"blk.{L}." + {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate",
                              "up": "ffn_up", "down": "ffn_down"}[name] + ".weight"

    def _qkv_rot(self, L: int) -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
        """Layer L's q, k, v copies of the residual (their rotations, the input norm's weight folded in)."""
        g, h8 = self._vec(f"blk.{L}.attn_norm.weight"), self.cfg.hidden_size // COLS
        return [(f"blk.{L}.attn_{n}.weight", g, column_groups(0, h8)) for n in "qkv"]

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, I, hd, nh = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, cfg.num_attention_heads
        q = (np.arange(nh * hd) if cfg.neox else unpermute_rows(nh, nh * hd)).reshape(COLS, cfg.q_per_kv * hd)
        k = (np.arange(COLS * hd) if cfg.neox else unpermute_rows(COLS, COLS * hd)).reshape(COLS, hd)
        self._norm(L, "qkv")
        self._add(L, "qkv", [("q", (b + "attn_q.weight",), q), ("k", (b + "attn_k.weight",), k),
                             ("v", (b + "attn_v.weight",), column_rows(COLS * hd))], H)
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], nh * hd)
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._norm(L, "gate_up")
        self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
