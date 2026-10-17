"""The weight image of a Gemma 4 model (``overlay.image.TileImage``): Q4_0 tiles of 32 rows and Q8_0 tiles of 16 rows
over K chunks of 512 columns (``KC``: E4B's K are 2560, 2048, 4096 and 10240), objects of 9216 bytes (``TILE_BYTES``:
one head_dim-512 K record of the KV cache, which streams through the same A objects). The decode's GEMV
(``overlay.gemv``) and the prefill's GEMMs read the tiles in place.

Column-major: a column's regions in step order, each normed phase's norm-weight objects (data region "<name>_norm",
``norms``) before its tiles, so that a decode step streams a column's weights between two attentions as one run.

Per-column rows (layer L, column c, KV head h = c / 4, j = c % 4; hd: the layer's head_dim):

- "qkv": block "q" = the rows of query head c (hd); a layer with its own KV cache also block "kv" = the j-th quarter of
  the stack [K_h; V_h] (j 0, 1: K_h's halves; 2, 3: V_h's): the column's outputs land in the KV head's q objects and
  raw K / V objects (``gemma4.overlay``);
- "o" (``attn_output``), "down", "proj": rows hidden / 8 c ..; "gate_up": ``overlay.image.gate_up_rows``; "gate"
  (``inp_gate``): rows per_layer / 8 c ..;
- "proj" reads its input (per_layer values) in column slots of 64: the gate phase's column c drains its 32 outputs and
  32 zeros to elements 64 c ..; its tiles hold ``proj``'s columns 32 c .. at 64 c .. and zero weights between (the
  re-laid-out copy ``_slotted``: GGUF blocks of 32 columns, a zero block after each);
- "lm": rows vocab / 8 c .. of the LM head (``token_embd``, tied).
"""

import numpy as np
import torch

from ...gguf_model import GGUFFile, QTensor
from ...model_config import GemmaConfig
from ..design import COLS
from ..overlay import gemv
from ..overlay.decode import MAX_K, NX_PER_NORM, OBJ_K, POST_K
from ..overlay.image import TileImage, column_rows, gate_up_rows

KC = 512  # K chunk of a tile
TILE_BYTES = 9216
FORMATS = ("Q4_0", "Q8_0")  # (the decode GEMV's row-split tiles at KC)
SLOT = 64  # the proj input's column slots


def kv_rows(hd: int) -> np.ndarray:
    """(COLS, hd / 2) rows of the stack [K; V] (2 KV heads: K rows 0 .. 2 hd - 1, V rows 2 hd ..) of the columns."""
    out = []
    for c in range(COLS):
        h, j = divmod(c, 4)
        base = (2 * hd if j >= 2 else 0) + h * hd + (j % 2) * hd // 2
        out.append(base + np.arange(hd // 2))
    return np.stack(out)


class GemmaImage(TileImage):
    """The weight image of a Gemma 4 GGUF file (layers: the layers to lay out, default all; vocab: LM rows, default
    the vocabulary)."""

    def __init__(self, gguf: GGUFFile, cfg: GemmaConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None):
        self.cfg = cfg = cfg or GemmaConfig.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        H = cfg.hidden_size
        super().__init__(gguf, kc=KC, payload=MAX_K, tile_bytes=TILE_BYTES,
                         vocab=cfg.vocab_size if vocab is None else vocab, max_rows=gemv.row_splits(KC // 256),
                         column_major=True)
        assert self.vocab % (COLS * 16) == 0 and H % (COLS * 64) == 0
        for L in self.layers:
            self._layer(L)
        lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        self._norm(-1, "lm")
        self._add(-1, "lm", [("lm", (lm,), column_rows(self.vocab))], H)
        self._place()

    # ------------------------------------------------------------------
    # Norm-weight objects
    # ------------------------------------------------------------------

    def _post_of(self, L: int) -> tuple[str, str | None]:
        """(post norm, output scale) of the update that ends layer L."""
        b = f"blk.{L}."
        scale = b + "layer_output_scale.weight"
        return (b + ("post_norm.weight" if self.cfg.per_layer_dim else "post_ffw_norm.weight"),
                scale if scale in self.gguf else None)

    def norms(self, L: int, name: str) -> tuple[str | None, str | None, str | None]:
        """(w1, w_post, scale) tensor names of a normed phase (None: ones / no scale); w_post None: a plain normed
        input (the slice's first layer), else a header (``overlay.decode.F_POST``)."""
        b = f"blk.{L}."
        if name == "qkv":
            if L == self.layers[0]:
                return b + "attn_norm.weight", None, None
            return (b + "attn_norm.weight", *self._post_of(self.layers[self.layers.index(L) - 1]))
        if name == "gate_up":
            return b + "ffn_norm.weight", b + "post_attention_norm.weight", None
        if name == "gate":
            return None, b + "post_ffw_norm.weight", None
        assert name == "lm", name
        return ("output_norm.weight", *self._post_of(self.layers[-1]))

    def n_norm(self, L: int, name: str) -> int:
        """Norm-weight objects of a phase: [w1 4096] (a plain input) or [w1 2048 | w_post 2048] (a header)."""
        per = NX_PER_NORM * (OBJ_K if self.norms(L, name)[1] is None else POST_K)
        return -(-self.cfg.hidden_size // per)

    def _norm(self, L: int, name: str) -> None:
        self._add_data(L, name + "_norm", self.n_norm(L, name) * self.tile_bytes)

    def _vec(self, name: str | None, fill: float = 1.0) -> np.ndarray:
        if name is None:
            return np.full(self.cfg.hidden_size, fill, np.float32)
        return np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

    def norm_objects(self, L: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's norm-weight objects (bf16)."""
        w1n, wpn, _ = self.norms(L, name)
        H, n, T = self.cfg.hidden_size, self.n_norm(L, name), self.tile_bytes
        if wpn is None:
            parts = np.zeros((1, n * NX_PER_NORM * OBJ_K), np.float32)
            parts[0, :H] = self._vec(w1n)
        else:
            parts = np.zeros((2, n * NX_PER_NORM * POST_K), np.float32)
            parts[0, :H], parts[1, :H] = self._vec(w1n), self._vec(wpn)
        objs = parts.reshape(len(parts), n, -1).transpose(1, 0, 2).reshape(n, -1)  # [object][w1 | w_post]
        b = torch.from_numpy(np.ascontiguousarray(objs)).bfloat16().view(torch.int16).numpy().view(np.uint8)
        out = np.zeros((n, T), np.uint8)
        out[:, :b.shape[1]] = b
        return out

    def fmt(self, name: str) -> str:
        f = super().fmt(name)
        if f not in FORMATS:
            raise NotImplementedError(f"{name}: {f} tiles (the Gemma decode reads {', '.join(FORMATS)} of {KC} columns)")
        return f

    def _slotted(self, name: str) -> str:
        """A copy of a matrix of COLS x 32 columns with a zero GGUF block after each block of 32 (its columns 32 c ..
        at 2 x 32 c ..: the column slots of its input)."""
        t = self.gguf.tensor(name)
        assert t.cols == COLS * 32 and t.qtype in ("Q4_0", "Q8_0"), (name, t.cols, t.qtype)
        bb = 18 if t.qtype == "Q4_0" else 34
        raw = np.zeros((t.rows, 2 * COLS, bb), np.uint8)
        raw[:, 0::2] = np.asarray(t.raw).reshape(t.rows, COLS, bb)
        self._own[name] = QTensor(name=name, qtype=t.qtype, rows=t.rows, cols=COLS * SLOT, raw=raw.reshape(t.rows, -1))
        return name

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, I, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim(L)
        q = hd * np.arange(COLS)[:, None] + np.arange(hd)[None]
        parts = [("q", (b + "attn_q.weight",), q)]
        if cfg.kv_source(L) == L:
            parts.append(("kv", (b + "attn_k.weight", b + "attn_v.weight"), kv_rows(hd)))
        self._norm(L, "qkv")
        self._add(L, "qkv", parts, H)
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], COLS * hd)
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._norm(L, "gate_up")
        self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
        if cfg.per_layer_dim:
            assert cfg.per_layer_dim == COLS * 32, "per-layer inputs: 32 per column"
            self._norm(L, "gate")
            self._add(L, "gate", [("gate", (b + "inp_gate.weight",), column_rows(cfg.per_layer_dim))], H)
            self._add(L, "proj", [("proj", (self._slotted(b + "proj.weight"),), column_rows(H))], COLS * SLOT)
