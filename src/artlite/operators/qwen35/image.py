"""The weight image of a qwen35 model (``overlay.image.TileImage``): Q4_K, Q5_K, Q6_K tiles of 1024 columns.

Per-column rows (the columns' work is independent up to each output projection):

- GDN layers ("gdn_in"): column c holds q / k heads {2c, 2c + 1} (``QwenConfig.ssm_k_heads`` / 8 per column) and
  the v heads j with j % k_heads in that set (27B: 6 heads, 9B: 4), in increasing j: its rows of ``attn_qkv``
  ([q heads | k heads | v heads] of the column), of ``attn_gate`` (z of its v heads) and one tile of
  ``ssm_alpha`` | ``ssm_beta`` (its v heads' rows). The conv channels, the delta-rule state and the gated norm
  of a v head stay in its column; the column's GDN output (its v heads, 128 each) lands at the heads' places
  of the ``ssm_out`` input (pairs of consecutive heads: whole Q4_K super-blocks).
- attention layers ("attn_in"): column c holds q heads 3c .. (27B; 9B: 2 per column) with their gate rows
  (``attn_q``: per head [q 256 | gate 256]), and rows 128 c .. of ``attn_k`` and of ``attn_v`` (half a KV head).
- "gate_up": ``overlay.image.gate_up_rows``; "down", "gdn_out" (``ssm_out``), "attn_out" (``attn_output``): rows
  hidden / 8 c ..; "lm": rows vocab / 8 c .. of ``output``, the vocabulary padded to whole C objects per column
  (MiniCPM-V's 248094 -> 248320; padding rows -1: zero tiles).

down's K is the ffn padded to whole K chunks (MiniCPM-V's 3584 -> 4096: zero super-blocks, ``overlay.image``). The
activation objects' payload (``payload``: the K groups) is the hidden state rounded up to whole 4 KB state panels of the
GDN cores, at least 4096 elements (27B 6144, 9B and MiniCPM-V 4096): down takes its K chunks in groups (27B 1 of 17, 9B
4 of 12, MiniCPM-V 4 of 4).
"""

import numpy as np

from ...gguf_model import GGUFFile
from ...model_config import QwenConfig
from ..design import COLS
from ..overlay.decode import C_LEN
from ..overlay.image import TileImage, column_rows, gate_up_rows

KC = 1024  # K chunk of a tile (5120 = 5 chunks, 4096 = 4)


def payload(hidden: int) -> int:
    """bf16 elements of an activation object's payload: the hidden state, rounded up to whole 4 KB panels (the
    GDN state objects carry whole panels), at least 4096."""
    return max(-(-hidden // 2048) * 2048, 4096)


def gdn_v_heads(cfg: QwenConfig, c: int) -> list[int]:
    """The v heads of column c (increasing): those whose q / k head (j % k_heads) is one of the column's."""
    kpc = cfg.ssm_k_heads // COLS
    return [j for j in range(cfg.ssm_v_heads) if j % cfg.ssm_k_heads // kpc == c]


class QwenImage(TileImage):
    """The weight image of a qwen35 GGUF file."""

    def __init__(self, gguf: GGUFFile, cfg: QwenConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0):
        """layers: the layers to lay out (default: all); vocab: LM rows (default: the padded vocabulary; tests take
        fewer)."""
        self.cfg = cfg = cfg or QwenConfig.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        H, I = cfg.hidden_size, cfg.intermediate_size
        if vocab is None:
            vocab = -(-cfg.vocab_size // (COLS * C_LEN)) * COLS * C_LEN
        super().__init__(gguf, kc=KC, payload=payload(H), tile_bytes=tile_bytes, vocab=vocab)
        assert H % KC == 0 and H % (8 * COLS) == 0 and I % (64 * COLS) == 0
        assert cfg.ssm_k_heads % COLS == 0 and cfg.num_attention_heads % COLS == 0
        assert cfg.num_key_value_heads * cfg.head_dim % (COLS * 64) == 0
        assert self.vocab % COLS == 0
        for layer in self.layers:
            self._layer(layer)
        lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        rows = column_rows(self.vocab)
        self._add(-1, "lm", [("lm", (lm,), np.where(rows < cfg.vocab_size, rows, -1))], H)
        self._place()

    def _layer(self, layer: int) -> None:
        cfg, b = self.cfg, f"blk.{layer}."
        H, I, cols = cfg.hidden_size, cfg.intermediate_size, np.arange(COLS)[:, None]
        if cfg.attention[layer]:
            qh, hd = cfg.num_attention_heads // COLS, cfg.head_dim
            q = 2 * qh * hd * cols + np.arange(2 * qh * hd)[None]  # (per head [q | gate])
            kv = (cfg.num_key_value_heads * hd // COLS) * cols + np.arange(cfg.num_key_value_heads * hd // COLS)[None]
            self._add(layer, "attn_in", [("q", (b + "attn_q.weight",), q), ("k", (b + "attn_k.weight",), kv),
                                         ("v", (b + "attn_v.weight",), kv)], H)
            self._add(layer, "attn_out", [("out", (b + "attn_output.weight",), column_rows(H))],
                      cfg.num_attention_heads * hd)
        else:
            d, hk = cfg.ssm_state, cfg.ssm_k_heads
            kpc = hk // COLS
            qkv, z, ab = [], [], []
            for c in range(COLS):
                vh = gdn_v_heads(cfg, c)
                qk = np.arange(kpc * d) + kpc * d * c
                v = np.concatenate([2 * hk * d + j * d + np.arange(d) for j in vh])
                qkv.append(np.concatenate([qk, hk * d + qk, v]))
                z.append(np.concatenate([j * d + np.arange(d) for j in vh]))
                ab.append(np.concatenate([vh, cfg.ssm_v_heads + np.asarray(vh)]))  # (alpha | beta stacked)
            self._add(layer, "gdn_in", [("qkv", (b + "attn_qkv.weight",), np.stack(qkv)),
                                        ("z", (b + "attn_gate.weight",), np.stack(z)),
                                        ("ab", (b + "ssm_alpha.weight", b + "ssm_beta.weight"), np.stack(ab))], H)
            self._add(layer, "gdn_out", [("out", (b + "ssm_out.weight",), column_rows(H))], cfg.ssm_v_heads * d)
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._add(layer, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(layer, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
