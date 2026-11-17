"""The weight image of a Spark-X model (``decode.image.TileImage``): the file's tile formats over K chunks of 1024 columns
(``KC``) in objects of ``decode.attn256.OBJ_BYTES`` (the overlay's: a weight tile and a KV cache object are one size),
read in place by the decode's GEMV (``decode.gemv``) and the prefill's GEMMs.

Per-column rows (layer L, column c; NPH = 8 / KV heads columns per KV head, QC = heads / 8 query heads per column):

- "attn_in" (the attention's input, normed by ``attn_norm``; the fused ``attn_qkv``'s rows [q | k | v], or ``attn_q``,
  ``attn_k``, ``attn_v``): block "q", query heads QC c .. (their 256 rows each, as stored: NeoX pairs), blocks "k" and
  "v", rows 256 / NPH (c % NPH) .. of KV head c // NPH (the 4B's halves, the 1.7B's quarters: qwen35's layout, the
  attention core's ``decode.attn256``), block "g", the QC heads' rows of ``attn_gate`` (one row tile); an F32 gate
  (bartowski's files) as its BF16 copy (``_own``: one BFP16 plane, as the official files' BF16 gates);
- "attn_out" (``attn_output``), "down": rows hidden / 8 c ..; "gate_up" (normed by ``ffn_norm``):
  ``decode.image.gate_up_rows`` (GeGLU's gate and up rows);
- "lm" (normed by ``output_norm``): rows vocab / 8 c .. of ``output``, or of the tied ``token_embd``; with a DSpark
  drafter's head (``head``: ``decode.head``) per column its share of the head's fixed tokens, of its table's, then of
  the other ids (``order``: the LM row of each token id).

The 4B's hidden 2560 (attn_in, gate_up, lm: ``TAILS``) is half a chunk short of 3: those regions are
``decode.image.Region.half`` regions (``tails``), gate / up's 16-row Q4_K / Q4_0 blocks (the prefill's GEMMs read them in
pass pairs) and the LM head's (the prefill reads none) holding their row tiles in pairs that share the tile of the last
512 columns (``pairs``), attn_in's tiles padded with zero super-blocks (its outputs, of 11 or 16 rows a tile, would leave
the prefill's GEMMs in strided runs); ``tails=False``: all padded. The 1.7B's ffn 6656 (down) is padded to whole chunks
with zero super-blocks (``decode.image``). Column-major (each normed phase's norm-weight objects, data region
"<name>_norm", before its tiles) when every region's decode K groups are whole chunk groups of the image (the 4B), else
region-major (the 1.7B: its down's 7 chunks run as 5 + 2)."""

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.attn256 import OBJ_BYTES
from artlite.decode.head import HEAD_ROWS, TABLE_ROWS, draft_cols, head_rows, head_tokens, lm_order
from artlite.decode.image import TileImage, column_rows, gate_up_rows
from artlite.decode.phases import group_chunks
from artlite.decode.protocol import MAX_K, NX_PER_NORM, OBJ_K, norm_objects
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import SparkConfig
from artlite.operators.design import COLS
from artlite.prefill.gemm import pair_tails

KC = 1024  # K chunk of a tile
NORMS = {"attn_in": "attn_norm", "gate_up": "ffn_norm", "lm": "output_norm"}
TAILS = frozenset(NORMS)  # (``tails``: the regions of the hidden's input, ``decode.image.Region.half``)


def attn_rows(cfg: SparkConfig, *, fused: bool) -> dict[str, np.ndarray]:
    """(COLS, n) rows of attn_in's blocks "q", "k", "v" (of the fused ``attn_qkv``: [q | k | v]; else of ``attn_q``,
    ``attn_k``, ``attn_v``) and "g" (of ``attn_gate``)."""
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qc, per = nh // COLS, nkv * hd // COLS
    c = np.arange(COLS)[:, None]
    k0, v0 = (nh * hd, (nh + nkv) * hd) if fused else (0, 0)
    return {"q": qc * hd * c + np.arange(qc * hd)[None], "k": k0 + per * c + np.arange(per)[None],
            "v": v0 + per * c + np.arange(per)[None], "g": qc * c + np.arange(qc)[None]}


def contiguous(image: TileImage) -> bool:
    """Whether the decode's K groups of every region (``phases.group_chunks``) are whole chunk groups of the image."""
    return all(g == r.kc_group for r in image.regions for _, g in group_chunks(r))


class SparkImage(TileImage):
    """The weight image of a spark2_5 GGUF file (layers: the layers to lay out, default all; vocab: LM rows, default the
    vocabulary; head: the LM rows laid out with a drafter's head, ``draft_cols`` rows per column; tails: the hidden's
    regions half regions, module docstring)."""

    order: np.ndarray | None = None  # (the LM row of each token id when they are not in id order)
    draft_cols = 0

    def __init__(self, gguf: GGUFFile, cfg: SparkConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, head: bool = False, tails: bool = True):
        self.cfg = cfg = cfg or SparkConfig.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        H = cfg.hidden_size
        self.n_ids = cfg.vocab_size if vocab is None else vocab  # (the token ids of the LM rows: 0 .. ids - 1)
        super().__init__(gguf, kc=KC, payload=MAX_K, tile_bytes=OBJ_BYTES, vocab=self.n_ids,
                         tails=TAILS if tails else frozenset())
        assert cfg.num_attention_heads % COLS == 0 and cfg.num_key_value_heads * cfg.head_dim % (COLS * 64) == 0, cfg
        assert H % (64 * COLS) == 0 and self.vocab % (COLS * 64) == 0, cfg
        for L in self.layers:
            self._layer(L)
        self.lm = "token_embd.weight" if cfg.tied else "output.weight"
        rows = column_rows(self.vocab)
        if head:
            self.draft_cols = draft_cols(self.n_ids, self.R[self.fmt(self.lm)], HEAD_ROWS)
            F = COLS * self.draft_cols - TABLE_ROWS
            self.fixed, self.table = (head_tokens(gguf, F) if self.n_ids == cfg.vocab_size else
                                      (np.arange(F), np.arange(F, F + TABLE_ROWS)))
            self.vocab = self.n_ids + TABLE_ROWS
            rows = head_rows(self.n_ids, self.fixed, self.table)
            self.order = lm_order(rows, self.draft_cols)
        self._add(-1, "lm", [("lm", (self.lm,), rows)], H)
        self.n_norm = -(-H // (NX_PER_NORM * OBJ_K))  # norm-weight objects of a normed phase
        self.column_major = contiguous(self)
        if self.column_major:  # (each normed phase's norm-weight objects before its tiles)
            regions, self.regions = self.regions, []
            for r in regions:
                if r.name in NORMS:
                    self._add_data(r.layer, r.name + "_norm", self.n_norm * self.tile_bytes)
                self.regions.append(r)
        self._place()

    def pairs(self, region: str, block: str, fmt: str, R: int) -> bool:
        """The LM head's (the prefill reads none), the regions' the prefill reads in pairs (``pair_tails``)."""
        return region == "lm" or pair_tails(region, fmt, R)

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's RMSNorm weights (bf16)."""
        w = self.gguf.tensor("output_norm.weight" if layer < 0 else f"blk.{layer}.{NORMS[name]}.weight").dequantize()
        return norm_objects(w.reshape(-1), self.tile_bytes).view(np.uint8)

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, I = cfg.hidden_size, cfg.intermediate_size
        fused = b + "attn_qkv.weight" in self.gguf
        rows = attn_rows(cfg, fused=fused)
        gate = b + "attn_gate.weight"
        if self.qtype(gate) == "F32":  # (its BF16 copy: one BFP16 plane, module docstring)
            t = self.gguf.tensor(gate)
            raw = np.asarray(t.raw, np.float32).reshape(t.rows, t.cols).astype(bfloat16).view(np.uint16)
            self._own[gate] = QTensor(name=gate, qtype="BF16", rows=t.rows, cols=t.cols, raw=raw)
        qkv = [(n, (b + "attn_qkv.weight",) if fused else (b + f"attn_{n}.weight",), rows[n]) for n in "qkv"]
        self._add(L, "attn_in", [*qkv, ("g", (gate,), rows["g"])], H)
        self._add(L, "attn_out", [("out", (b + "attn_output.weight",), column_rows(H))],
                  cfg.num_attention_heads * cfg.head_dim)
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
