"""The weight image of a Flint model (``decode.image.NormedImage``): the file's tile formats over K chunks of 1024 columns
(``KC``); the official int4 file's Q4_1 tensors as Q4_1_64 tiles (``formats.q4``: its blocks pair up, one (d, m) per
64 weights, Q4_0's 144 B per row and super-block), the router's F32 rows as BFP16 planes: objects of 9216 B (16 rows of
Q4_1_64, BFP16's 8).

Column-major (as ``llama.image``): a column's regions in step order, each normed phase's norm-weight objects (data
region "<name>_norm") before its tiles, so that a decode step streams a column's weights between two attentions as one
run; the experts' slabs after them (``moe.image``).

Per-column rows (layer L, column c; KV head h = c / 2, j = c % 2: a KV head's attention on two columns):

- "qkv": block "q" = the rows of query heads 8 h + 4 j .. 8 h + 4 j + 3, blocks "k" and "v" half j of K_h's rows and of
  V_h's (dims 64 j ..); the rows of Q and K are the file's (NeoX pairs: dims i, i + head_dim / 2);
- "o" (``attn_output``): rows hidden / 8 c .., K 4096 in two K groups of the activation objects' 2048;
- the dense layers (``FlintConfig.n_dense``): "gate_up" (``decode.image.gate_up_rows``), "down": rows hidden / 8 c ..;
- the MoE layers: ``moe.image``'s "moe_in" (normed by ``ffn_norm``; the shared expert's gate / up, the router's planes;
  no shared gate: Flint's shared expert is ungated), "sdown" (K 1024: plain tiles), the expert slabs;
- "lm": rows vocab / 8 c .. of ``output``, the vocabulary padded to whole C objects per column (150144 -> 150528;
  padding rows -1: zero tiles).

The activation objects' payload is the hidden state's 2048 (an MoE design's: room for the side chunk next to the
experts' partials, ``decode.design``'s "MoE").
"""

import numpy as np

from artlite.decode.head import TABLE_ROWS, draft_cols, head_rows, head_tokens, lm_order
from artlite.decode.image import NormedImage, column_rows, gate_up_rows
from artlite.decode.protocol import C_LEN
from artlite.formats import FORMATS, q4
from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig
from artlite.moe.image import MoeImage
from artlite.operators.design import COLS

KC = 1024  # K chunk of a tile
HEAD_SIZE = 32768  # the DFlash2 drafter's head's fixed rows (``decode.head``): its drafts as many and as often kept as with 80000
# (code, math on the Flint-DFlash2 Q4_K_M), its round's reads 56 MB fewer (+2% decode on code)
NORMS = {"qkv": "attn_norm", "gate_up": "ffn_norm", "moe_in": "ffn_norm"}  # a layer's normed phases: their RMSNorm


def qkv_rows(cfg: FlintConfig) -> dict[str, np.ndarray]:
    """(COLS, n) rows of the qkv phase's blocks: "q" of ``attn_q``, "k" / "v" of ``attn_k`` / ``attn_v``."""
    hd, nkv = cfg.head_dim, cfg.num_key_value_heads
    per = COLS // nkv  # (columns per KV head)
    qh = cfg.q_per_kv // per  # (query heads per column)
    h, j = np.arange(COLS) // per, np.arange(COLS) % per
    heads = (cfg.q_per_kv * h + qh * j)[:, None] + np.arange(qh)[None]  # (COLS, qh)
    q = (hd * heads[..., None] + np.arange(hd)).reshape(COLS, -1)
    half = (hd * h + hd // per * j)[:, None] + np.arange(hd // per)[None]
    return {"q": q, "k": half, "v": half}


class FlintImage(MoeImage, NormedImage):
    """The weight image of a Flint GGUF file (layers: the layers to lay out, default all; vocab: LM rows, default the
    padded vocabulary; tile_bytes: the object size, default BFP16's: the router's planes; head: the LM rows laid out
    with a DFlash2 drafter's head, ``draft_cols`` rows per column, ``decode.head``: then ``order`` holds each token id's
    LM row)."""

    order: np.ndarray | None = None
    draft_cols = 0

    def __init__(self, gguf: GGUFFile, cfg: FlintConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0, head: bool = False, head_size: int = HEAD_SIZE):
        cfg = cfg or FlintConfig.from_gguf(gguf)
        H, nkv = cfg.hidden_size, cfg.num_key_value_heads
        if vocab is None:
            vocab = -(-cfg.vocab_size // (COLS * C_LEN)) * COLS * C_LEN
        self._q4_64: dict[str, bool] = {}
        # (the router's BFP16 planes fix the object size: 8 rows)
        super().__init__(gguf, cfg, layers=layers, kc=KC, payload=H,
                         tile_bytes=tile_bytes or FORMATS["BFP16"].tile_bytes(KC // 256), vocab=vocab)
        assert H == 2 * KC and COLS % nkv == 0 and cfg.q_per_kv % (COLS // nkv) == 0, cfg
        assert cfg.head_dim % (COLS // nkv * 64) == 0 and self.vocab % COLS == 0, cfg
        for L in self.layers:
            self._layer(L)
        self.lm = "output.weight"
        rows = column_rows(self.vocab)
        if head:  # (the ids: the padded vocabulary's, its padding rows empty; a truncated one in tests)
            n_ids = self.vocab
            self.draft_cols = draft_cols(n_ids, self.R[self.fmt(self.lm)], head_size)
            F = COLS * self.draft_cols - TABLE_ROWS
            self.fixed, self.table = (head_tokens(gguf, F) if n_ids >= cfg.vocab_size else
                                      (np.arange(F), np.arange(F, F + TABLE_ROWS)))
            self.vocab = n_ids + TABLE_ROWS
            rows = head_rows(n_ids, self.fixed, self.table)
            self.order = lm_order(rows, self.draft_cols)[:min(n_ids, cfg.vocab_size)]  # (the logits: the real ids')
        self._norm(-1, "lm")
        self._add(-1, "lm", [("lm", (self.lm,), np.where(rows < cfg.vocab_size, rows, -1))], H)
        self._place()

    def fmt(self, name: str) -> str:
        """(a Q4_1 tensor: Q4_1_64, its blocks checked to pair up, module docstring)"""
        f = super().fmt(name)
        if f != "Q4_1":
            return f
        if name not in self._q4_64:
            self._q4_64[name] = q4.pairs64(self.tensor(name).raw, 4096)
        if not self._q4_64[name]:
            raise NotImplementedError(f"{self.gguf.path}: {name}'s Q4_1 blocks are not in pairs of equal (d, m) (the "
                                      "official int4 file's group-64 INT4): no Q4_1_64 tiles")
        return "Q4_1_64"

    def norms(self, L: int, name: str) -> tuple[str, None]:
        """(w1, None) tensor names of a normed phase: plain inputs (pre-norm residual updates)."""
        return "output_norm.weight" if L < 0 else f"blk.{L}.{NORMS[name]}.weight", None

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, I = cfg.hidden_size, cfg.intermediate_size
        rows = qkv_rows(cfg)
        self._norm(L, "qkv")
        self._add(L, "qkv", [(n, (b + f"attn_{n}.weight",), rows[n]) for n in ("q", "k", "v")], H)
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], cfg.num_attention_heads * cfg.head_dim)
        if cfg.moe_layer(L):
            self._norm(L, "moe_in")
            self._moe(L, gated=False)
            return
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._norm(L, "gate_up")
        self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)

    def pack(self, out: np.ndarray, *, threads: int = 0) -> bool:
        """(the router's tensors as their copies times w / bf16(w), ``moe.image``; dropped after)"""
        names = []
        for L in self.layers:
            if self.cfg.moe_layer(L):
                names += self.scaled_planes(f"blk.{L}.ffn_norm.weight", self.router_planes(L))
        try:
            return super().pack(out, threads=threads)
        finally:
            for name in names:
                del self._own[name]
