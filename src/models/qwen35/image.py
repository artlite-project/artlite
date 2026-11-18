"""The weight image of a qwen35 model (``decode.image.TileImage``): Q4_K, Q5_K, Q6_K tiles of 1024 columns.

Per-column rows (the columns' work is independent up to each output projection):

- GDN layers ("gdn_in"): column c holds q / k heads {2c, 2c + 1} (``QwenConfig.ssm_k_heads`` / 8 per column) and
  the v heads j with j % k_heads in that set (27B: 6 heads, 9B: 4), in increasing j: its rows of ``attn_qkv``
  ([q heads | k heads | v heads] of the column), of ``attn_gate`` (z of its v heads) and one tile of
  ``ssm_alpha`` | ``ssm_beta`` (its v heads' rows: block "ab"; of two tile formats, an F32 alpha's (Occamy's): blocks
  "alpha", its BFP16 planes (``decode.image.Block``: per head its hi and lo rows, whose outputs the decode adds,
  packed times w / bf16(w) as the router's below, w the layer's ``attn_norm``), and "beta"). The conv channels, the delta-rule state and the gated norm
  of a v head stay in its column; the column's GDN output (its v heads, 128 each) lands at the heads' places
  of the ``ssm_out`` input (pairs of consecutive heads: whole Q4_K super-blocks).
- attention layers ("attn_in"): column c holds q heads 3c .. (27B; 9B: 2 per column) with their gate rows
  (``attn_q``: per head [q 256 | gate 256]), and rows 128 c .. of ``attn_k`` and of ``attn_v`` (half a KV head).
- "gate_up": ``decode.image.gate_up_rows``; "down", "gdn_out" (``ssm_out``), "attn_out" (``attn_output``): rows
  hidden / 8 c ..; "lm": rows vocab / 8 c .. of ``output``, the vocabulary padded to whole C objects per column
  (MiniCPM-V's 248094 -> 248320; padding rows -1: zero tiles).

An MoE layer (``qwen35moe``: ``QwenConfig.moe``) has instead of gate_up and down the regions of ``moe.image``: "moe_in"
(normed by ``post_attention_norm``: the shared expert's gate / up, the router's BFP16 planes, the shared expert's gate
row "sgate"), "sdown" (its K 512 as row groups of 2 per tile row) and the expert slabs "exp_gu", "exp_down".

A file with a multi-token-prediction block of its own (Qwen3.8: ``QwenConfig.nextn_layers``, the block at layer
``num_hidden_layers``) holds it after the LM head: "eh" (``nextn.eh_proj``, rows hidden / 8 c .. over K 2 hidden), then
the block's attention layer and its FFN as a layer's regions; only its drafter's steps stream them
(``draft.QwenDraft``). Its drafter's head is the first ``draft_cols`` rows of each column of the model's "lm" region
(``mtp_view``): the head's fixed tokens (``head_tokens``: the special tokens and the first ids), then ``TABLE_ROWS`` /
8 rows of a table of the sequence's own tokens (the drafter's alone: they repeat other rows), then the column's other
tokens; ``order``: the LM row of each token id (the model's logits, ``QwenOverlay._logits``).

Column-major (``decode.image``: a column's regions in step order, each normed phase's norm-weight objects before its
tiles, data region "<name>_norm") when every region's decode K groups are whole chunk groups of the image
(``contiguous``: the 9B, MiniCPM-V), so that a decode step streams a column's weights between two attentions as one run;
else region-major (the 27B: its output projections' 6 chunks run as 3 + 3, its down's 17 chunk-major in groups of 5).

An input half a chunk short of whole chunks in one K group (``decode.image.Region.half`` regions, ``tails``; a dense
model's) is the hidden size where it is so (Kev-4B's 2560: gdn_in, attn_in, gate_up, the LM head) and down's ffn
(MiniCPM-V's and d3-lite's 3584 in 4 chunks): those phases' inputs are of that K, the prefill's GEMMs read 2.5 of 3 (3.5
of 4) of a row tile's tiles; a model that decodes (``pairs``) holds gate / up's 16-row Q4_K / Q4_0 blocks (the prefill
reads them in pass pairs: ``prefill.gemm.pair_tails``) and the LM head's (no prefill reads it) in pairs of row tiles
that share the tile of the last 512 columns, a readout model's (its prompts only) and the other blocks' tiles keep that
tile's second half as zero super-blocks. A down of another K of part of a chunk is padded to whole K chunks with zero
super-blocks (``decode.image``). The activation objects' payload (``payload``: the K groups) is the hidden state rounded
up to whole 4 KB state panels of the GDN cores, at least 4096 elements (27B 6144, 9B and MiniCPM-V 4096): down takes its
K chunks in groups (27B 1 of 17, 9B 4 of 12, MiniCPM-V 4 of 4); an MoE model's is the hidden state's (Qwen3.6-35B-A3B
2048: gdn_out and attn_out in groups of 2 chunks), so that the helper cores' x panels have room for the MoE's side chunk
next to its experts' partials (``decode.design``'s "MoE").
"""

import numpy as np

from artlite.decode.attn256 import OBJ_BYTES
from artlite.decode.head import (
    DFLASH_HEAD_ROWS,
    HEAD_ROWS,
    TABLE_ROWS,
    draft_cols,
    head_region,
    head_rows,
    head_tokens,
    lm_order,
)
from artlite.decode.image import Block, Region, TileImage, column_rows, gate_up_rows
from artlite.decode.phases import group_chunks
from artlite.decode.protocol import C_LEN, NX_PER_NORM, OBJ_K, norm_objects
from artlite.decode.screen import screened
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.moe.image import MoeImage
from artlite.operators.design import COLS
from artlite.prefill.gemm import pair_tails

KC = 1024  # K chunk of a tile (5120 = 5 chunks, 4096 = 4)
NORMS = {"gdn_in": "attn_norm", "attn_in": "attn_norm", "gate_up": "post_attention_norm", "moe_in": "post_attention_norm",
         "lm": "output_norm"}
# (``tails``: a dense model's regions of the hidden's input, and down's: the ffn half a chunk short in one K group)
TAILS = frozenset({"gdn_in", "attn_in", "gate_up", "down", "lm"})


def payload(hidden: int, moe: bool = False) -> int:
    """bf16 elements of an activation object's payload: the hidden state, rounded up to whole 4 KB panels (the
    GDN state objects carry whole panels), at least 4096 (an MoE model's: the hidden state's, module docstring)."""
    return -(-hidden // 2048) * 2048 if moe else max(-(-hidden // 2048) * 2048, 4096)


def gdn_v_heads(cfg: QwenConfig, c: int) -> list[int]:
    """The v heads of column c (increasing): those whose q / k head (j % k_heads) is one of the column's."""
    kpc = cfg.ssm_k_heads // COLS
    return [j for j in range(cfg.ssm_v_heads) if j % cfg.ssm_k_heads // kpc == c]


def contiguous(region: Region) -> bool:
    """Whether the decode's K groups of a region (``phases.group_chunks``) are whole chunk groups of the image: the
    tiles stream in the image's order."""
    return all(g == region.kc_group for _, g in group_chunks(region))


class QwenImage(MoeImage, TileImage):
    """The weight image of a qwen35 GGUF file. ``draft_cols``: its MTP block's drafter's head rows per column (0: no
    block laid out; module docstring)."""

    order: np.ndarray | None = None  # (the LM row of each token id when they are not in id order)
    draft_cols = 0
    mtp_layer = -1
    SCREEN = True  # (the model's own LM head may be screened: ``screen``)

    def __init__(self, gguf: GGUFFile, cfg: QwenConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0, mtp: bool = True, tails: bool = True,
                 pairs: bool = True):
        """layers: the layers to lay out (default: all); vocab: LM rows (default: the padded vocabulary; tests take
        fewer: the ids below it, ``n_ids``, and with an MTP block the drafter's table besides); mtp: the MTP block's regions
        (False: the drafter head's rows alone, a DFlash2 drafter reads them and not the block); tails: a dense model's
        regions of the hidden's input half regions where it is half a chunk short, pairs: their pairs of row tiles
        (module docstring)."""
        self.cfg = cfg = cfg or QwenConfig.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        H, I = cfg.hidden_size, cfg.intermediate_size
        self.n_ids = cfg.vocab_size if vocab is None else vocab  # (the token ids of the LM rows: 0 .. ids - 1)
        if vocab is None:
            vocab = -(-cfg.vocab_size // (COLS * C_LEN)) * COLS * C_LEN
        self.paired = pairs
        # (the overlay's objects: ``OBJ_BYTES`` whatever the file's formats; a file of Q8_0 alone would choose 9792)
        super().__init__(gguf, kc=KC, payload=payload(H, cfg.moe), tile_bytes=tile_bytes or OBJ_BYTES, vocab=vocab,
                         tails=TAILS if tails and not cfg.moe else frozenset())
        assert H % (64 * COLS) == 0 and I % (64 * COLS) == 0
        assert cfg.ssm_k_heads % COLS == 0 and cfg.num_attention_heads % COLS == 0
        assert cfg.num_key_value_heads * cfg.head_dim % (COLS * 64) == 0
        assert self.vocab % COLS == 0
        for layer in self.layers:
            self._layer(layer)
        trunk = all(contiguous(r) for r in self.regions)
        lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        if cfg.nextn_layers == 1 and not cfg.moe and not trunk:  # (its head: a view of a region-major image's rows)
            self.draft_cols = self._draft_cols(self.R[self.fmt(lm)], HEAD_ROWS if mtp else DFLASH_HEAD_ROWS)
        if self.draft_cols:
            F = COLS * self.draft_cols - TABLE_ROWS
            self.fixed, self.table = (head_tokens(gguf, F) if self.n_ids == cfg.vocab_size else
                                      (np.arange(F), np.arange(F, F + TABLE_ROWS)))
            self.vocab = self.n_ids + TABLE_ROWS
        self.lm = lm
        # (the LM head as its high plane, the logits that matter exactly on the host: ``decode.screen``. Not a dense
        # model's with an MTP block, whatever rows of it the image holds: its drafters read the head's rows in place
        # (``draft_cols``), and its design of 8 slots has no L1 for tiles of more than 16 rows next to two X objects,
        # ``decode.design``'s "Designs of 8 slots")
        self.screen = self.SCREEN and not (cfg.nextn_layers and not cfg.moe) and screened(self.qtype(lm), self.formats())
        self._add(-1, "lm", [("lm", (lm,), self.head_rows(), *(("Q6_KH",) if self.screen else ()))], H)
        if self.draft_cols:
            self.order = self._order()
        if self.draft_cols and mtp:
            self.mtp_layer = L = cfg.num_hidden_layers
            self._add(L, "eh", [("eh", (f"blk.{L}.nextn.eh_proj.weight",), column_rows(H))], 2 * H)
            self._layer(L, attention=True)
        self.n_norm = -(-H // (NX_PER_NORM * OBJ_K))  # norm-weight objects of a normed phase
        self.column_major = all(contiguous(r) for r in self.regions)
        if self.column_major:  # (each normed phase's norm-weight objects before its tiles)
            regions, self.regions = self.regions, []
            for r in regions:
                if r.name in NORMS:
                    self._add_data(r.layer, r.name + "_norm", self.n_norm * self.tile_bytes)
                self.regions.append(r)
        self._place()

    def pairs(self, region: str, block: str, fmt: str, R: int) -> bool:
        """(``paired``) The LM head's (the prefill reads none), the regions' the prefill reads in pairs
        (``pair_tails``)."""
        return self.paired and (region == "lm" or pair_tails(region, fmt, R))

    def _draft_cols(self, R: int, rows: int) -> int:
        """The drafter head's rows per column (``head.draft_cols``)."""
        return draft_cols(self.n_ids, R, rows)

    def head_rows(self) -> np.ndarray:
        """The LM rows' tokens (COLS, vocab / COLS): column c's are vocab / 8 c .. (-1: a padding row); with an MTP
        block (``draft_cols``) its share of the drafter head's fixed tokens, of the table's, then of the other ids."""
        if not self.draft_cols:
            rows = column_rows(self.vocab)
            return np.where(rows < self.cfg.vocab_size, rows, -1)
        return head_rows(self.n_ids, self.fixed, self.table)

    def _order(self) -> np.ndarray:
        """The LM row of each token id (``n_ids``,): its fixed or other row (the table's repeat them)."""
        return lm_order(self.head_rows(), self.draft_cols)

    def gu_rows(self) -> set[int]:
        """The rows per tile of the gate / up phases' tiles: the dense layers' gate_up (a split block's both), an MoE
        layer's shared expert (moe_in's "sgu") and experts (exp_gu)."""
        return {R for r in self.regions + self.slabs for b in r.blocks
                if r.name in ("gate_up", "exp_gu") or (r.name, b.name) == ("moe_in", "sgu") for R in (b.R, b.R2) if R}

    def mtp_view(self, cfg: QwenConfig) -> "MtpView":
        """The image of the MTP block's drafter (``cfg``: its config, the block as a layer): ``MtpView``."""
        assert self.draft_cols, "a file with an MTP block of its own"
        return MtpView(self, cfg)

    def pack(self, out: np.ndarray, *, threads: int = 0) -> bool:
        """(an MoE model's router tensors and an F32 alpha as their copies times w / bf16(w), module docstring; dropped
        after)"""
        names = []
        for L in self.layers:
            b = f"blk.{L}."
            if self.cfg.moe:
                names += self.scaled_planes(b + "post_attention_norm.weight", self.router_planes(L))
            if not self.cfg.attention[L] and any(blk.name == "alpha" for blk in self.region(L, "gdn_in").blocks):
                names += self.scaled_planes(b + "attn_norm.weight", (b + "ssm_alpha.weight",))
        try:
            return super().pack(out, threads=threads)
        finally:
            for name in names:
                del self._own[name]

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's RMSNorm weights (bf16)."""
        w = self.gguf.tensor("output_norm.weight" if layer < 0 else f"blk.{layer}.{NORMS[name]}.weight").dequantize()
        return norm_objects(w.reshape(-1), self.tile_bytes).view(np.uint8)

    def _layer(self, layer: int, attention: bool | None = None) -> None:
        """(``attention``: whether it is a full-attention layer, default ``cfg.attention``'s: an MTP block's is)"""
        cfg, b = self.cfg, f"blk.{layer}."
        H, I, cols = cfg.hidden_size, cfg.intermediate_size, np.arange(COLS)[:, None]
        if cfg.attention[layer] if attention is None else attention:
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
            alpha, beta = b + "ssm_alpha.weight", b + "ssm_beta.weight"
            ab = np.stack(ab)
            if self.fmt(alpha) == self.fmt(beta):
                abp = [("ab", (alpha, beta), ab)]
            else:  # (an F32 alpha: its planes [hi, lo] per head, module docstring; their sums: the design's B_PAIR)
                assert self.gguf.tensor_info[alpha][1] == "F32", (alpha, self.fmt(alpha), self.fmt(beta))
                a = ab[:, :ab.shape[1] // 2]
                abp = [("alpha", (alpha,), np.stack([2 * a, 2 * a + 1], -1).reshape(COLS, -1)),
                       ("beta", (beta,), ab[:, ab.shape[1] // 2:] - cfg.ssm_v_heads)]
            self._add(layer, "gdn_in", [("qkv", (b + "attn_qkv.weight",), np.stack(qkv)),
                                        ("z", (b + "attn_gate.weight",), np.stack(z)), *abp], H)
            self._add(layer, "gdn_out", [("out", (b + "ssm_out.weight",), column_rows(H))], cfg.ssm_v_heads * d)
        if cfg.moe:
            self._moe(layer, gated=True)
            return
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._add(layer, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(layer, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)


class MtpView:
    """The image an MTP block's drafter runs on (``QwenImage.mtp_view``): the block's regions in the model's image (its
    eh, attention layer and FFN) and the drafter's head, the first ``draft_cols`` rows of each column of the model's
    "lm" region (its fixed tokens, then its table's: ``head_rows``); in the model's buffer (``draft.QwenDraft``)."""

    order = None
    column_major = False
    screen = False  # (the drafter reads its head's rows as they are)

    def __init__(self, image: QwenImage, cfg: QwenConfig):
        self.model, self.cfg, self.gguf = image, cfg, image.gguf
        self.slabs: list[Region] = []
        self.layers = [image.mtp_layer]
        self.tile_bytes, self.R, self.kc, self.n_norm = image.tile_bytes, image.R, image.kc, image.n_norm
        self.fixed, self.table = image.fixed, image.table
        P = image.draft_cols
        head = head_region(image.region(-1, "lm"), P)
        self.regions = [r for r in image.regions if r.layer == image.mtp_layer]
        self.vocab = COLS * P
        self._index = {(r.layer, r.name): r for r in self.regions} | {(-1, "lm"): head}

    def region(self, layer: int, name: str) -> Region:
        return self._index[(layer, name)]

    gu_rows = QwenImage.gu_rows
    formats = TileImage.formats
    split = TileImage.split

    def head_rows(self) -> np.ndarray:
        """The head's tokens (COLS, draft_cols): per column its fixed tokens, then its table's (the model's LM rows)."""
        return self.region(-1, "lm").block("lm").rows

    def repack(self, out: np.ndarray, r: Region, b: Block, col: int, rt: int) -> list[tuple[int, int]]:
        return self.model.repack(out, r, b, col, rt)
