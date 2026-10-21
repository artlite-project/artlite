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

An MoE layer (``qwen35moe``: ``QwenConfig.moe``) has instead of gate_up and down:

- "moe_in" (normed by ``post_attention_norm``): "sgu", the shared expert's gate / up rows (``gate_up_rows``);
  "router", the router's rows of the column's experts E / 8 c .. (the F32 ``ffn_gate_inp`` as BFP16 planes,
  ``overlay.image.Block``: per expert its hi and lo rows, whose outputs the decode adds: the logits' order of
  llama.cpp's fp32 up to ~3e-5 of a weight); "sgate", the shared expert's gate row (``ffn_gate_inp_shexp``: its hi and
  lo planes) in every column. Both F32 tensors are packed times w / bf16(w), w the layer's ``post_attention_norm``
  (``pack``): the decode's norm multiplies x by bf16 weights (``overlay.decode.norm_objects``), the logits need w's
  fp32 (with the bf16 weights alone they are 8e-4 of their largest off, against 1.3e-5);
- "sdown": the shared expert's down, rows hidden / 8 c .., its K (512) as row groups of 2 per tile row
  (``overlay.image.group_rows``);
- the experts as slab regions (``TileImage.slabs``; half k = 2 e + h of expert e: its outputs h F / 2 .. of the expert's
  F = ``expert_ffn``): "exp_gu", the half's gate / up rows (``gate_up_rows`` over the 3-D ``ffn_gate_exps`` /
  ``ffn_up_exps``: expert e's rows e F ..), then "exp_down", the down's rows (all hidden of them) over the half's K
  (F / 2 columns, h F / 2 .., of ``ffn_down_exps``' rows e hidden ..) as row groups of kc / (F / 2) (4 for F 512).
  Expert e's two halves are one run of the image (``Region.col_stride`` apart: half 2 e + 1 right after half 2 e):
  a decode column streams a whole expert, or one half of one.

Column-major (``overlay.image``: a column's regions in step order, each normed phase's norm-weight objects before its
tiles, data region "<name>_norm") when every region's decode K groups are whole chunk groups of the image
(``contiguous``: the 9B, MiniCPM-V), so that a decode step streams a column's weights between two attentions as one run;
else region-major (the 27B: its output projections' 6 chunks run as 3 + 3, its down's 17 chunk-major in groups of 5).

down's K is the ffn padded to whole K chunks (MiniCPM-V's 3584 -> 4096: zero super-blocks, ``overlay.image``). The
activation objects' payload (``payload``: the K groups) is the hidden state rounded up to whole 4 KB state panels of the
GDN cores, at least 4096 elements (27B 6144, 9B and MiniCPM-V 4096): down takes its K chunks in groups (27B 1 of 17, 9B
4 of 12, MiniCPM-V 4 of 4); an MoE model's is the hidden state's (Qwen3.6-35B-A3B 2048: gdn_out and attn_out in groups
of 2 chunks), so that the helper cores' x panels have room for the MoE's side chunk next to its experts' partials
(``overlay.decode``'s "MoE").
"""

import numpy as np
import torch

from ...gguf_model import GGUFFile, QTensor
from ...model_config import QwenConfig
from ..design import COLS
from ..overlay.decode import C_LEN, NX_PER_NORM, OBJ_K, norm_objects
from ..overlay.image import Region, TileImage, column_rows, gate_up_rows, group_rows
from ..overlay.phases import group_chunks

KC = 1024  # K chunk of a tile (5120 = 5 chunks, 4096 = 4)
NORMS = {"gdn_in": "attn_norm", "attn_in": "attn_norm", "gate_up": "post_attention_norm", "moe_in": "post_attention_norm",
         "lm": "output_norm"}


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


class QwenImage(TileImage):
    """The weight image of a qwen35 GGUF file."""

    def __init__(self, gguf: GGUFFile, cfg: QwenConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tile_bytes: int = 0):
        """layers: the layers to lay out (default: all); vocab: LM rows (default: the padded vocabulary; tests take
        fewer)."""
        self.cfg = cfg = cfg or QwenConfig.from_gguf(gguf)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self._expert_rows: dict[tuple[str, int], np.ndarray] = {}  # (every layer's slab blocks share them)
        H, I = cfg.hidden_size, cfg.intermediate_size
        if vocab is None:
            vocab = -(-cfg.vocab_size // (COLS * C_LEN)) * COLS * C_LEN
        super().__init__(gguf, kc=KC, payload=payload(H, cfg.moe), tile_bytes=tile_bytes, vocab=vocab)
        assert H % KC == 0 and H % (8 * COLS) == 0 and I % (64 * COLS) == 0
        assert not cfg.moe or (cfg.n_experts % COLS == 0 and cfg.shared_ffn % (64 * COLS) == 0
                               and KC % cfg.shared_ffn == 0 and KC % (cfg.expert_ffn // 2) == 0
                               and cfg.expert_ffn % 128 == 0)
        assert cfg.ssm_k_heads % COLS == 0 and cfg.num_attention_heads % COLS == 0
        assert cfg.num_key_value_heads * cfg.head_dim % (COLS * 64) == 0
        assert self.vocab % COLS == 0
        for layer in self.layers:
            self._layer(layer)
        lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        rows = column_rows(self.vocab)
        self._add(-1, "lm", [("lm", (lm,), np.where(rows < cfg.vocab_size, rows, -1))], H)
        self.n_norm = -(-H // (NX_PER_NORM * OBJ_K))  # norm-weight objects of a normed phase
        self.column_major = all(contiguous(r) for r in self.regions)
        if self.column_major:  # (each normed phase's norm-weight objects before its tiles)
            regions, self.regions = self.regions, []
            for r in regions:
                if r.name in NORMS:
                    self._add_data(r.layer, r.name + "_norm", self.n_norm * self.tile_bytes)
                self.regions.append(r)
        self._place()

    def pack(self, out: np.ndarray, *, threads: int = 0, release: bool = True) -> bool:
        """(an MoE model's router tensors as their copies times w / bf16(w), module docstring; dropped after)"""
        names = []
        for L in self.layers if self.cfg.moe else []:
            w = self.gguf.tensor(f"blk.{L}.post_attention_norm.weight").dequantize().numpy().reshape(-1)
            bw = torch.from_numpy(w).bfloat16().float().numpy()
            ratio = np.where(bw != 0, w / np.where(bw != 0, bw, 1), 1).astype(np.float32)
            for name in (f"blk.{L}.ffn_gate_inp.weight", f"blk.{L}.ffn_gate_inp_shexp.weight"):
                t = self.gguf.tensor(name)
                raw = (np.asarray(t.raw, np.float32).reshape(t.rows, t.cols) * ratio).astype(np.float32)
                self._own[name] = QTensor(name=name, qtype="F32", rows=t.rows, cols=t.cols, raw=raw)
                names.append(name)
        try:
            return super().pack(out, threads=threads, release=release)
        finally:
            for name in names:
                del self._own[name]

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's RMSNorm weights (bf16)."""
        w = self.gguf.tensor("output_norm.weight" if layer < 0 else f"blk.{layer}.{NORMS[name]}.weight").dequantize()
        return norm_objects(w.numpy().reshape(-1), self.tile_bytes).view(np.uint8)

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
        if cfg.moe:
            self._moe(layer)
            return
        R = self.R[self.fmt(b + "ffn_gate.weight")]
        self._add(layer, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
        self._add(layer, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)

    def _moe(self, layer: int) -> None:
        """An MoE layer's regions (module docstring)."""
        cfg, b = self.cfg, f"blk.{layer}."
        H, E, F, Fs = cfg.hidden_size, cfg.n_experts, cfg.expert_ffn, cfg.shared_ffn
        mine = E // COLS * np.arange(COLS)[:, None] + np.arange(E // COLS)[None]  # (the column's experts)
        planes = (b + "ffn_gate_inp.weight", b + "ffn_gate_inp_shexp.weight")  # (rows 0 .. E - 1, E)
        R = self.R[self.fmt(b + "ffn_gate_shexp.weight")]
        self._add(layer, "moe_in", [("sgu", (b + "ffn_gate_shexp.weight", b + "ffn_up_shexp.weight"),
                                     gate_up_rows(Fs, R)),
                                    ("router", planes, np.stack([2 * mine, 2 * mine + 1], -1).reshape(COLS, -1)),
                                    ("sgate", planes, np.tile([2 * E, 2 * E + 1], (COLS, 1)))], H)
        R = self.R[self.fmt(b + "ffn_down_shexp.weight")]
        self._add(layer, "sdown", [("sdown", (b + "ffn_down_shexp.weight",), group_rows(column_rows(H), R, KC // Fs))],
                  Fs, G=KC // Fs)
        gu, down = (b + "ffn_gate_exps.weight", b + "ffn_up_exps.weight"), b + "ffn_down_exps.weight"
        self._add_slab(layer, "exp_gu", [("gu", gu, self._slab_rows("gu", self.R[self.fmt(gu[0])]))], H)
        self.view(down, F // 512)  # (a row of the view: a half's K)
        self._add_slab(layer, "exp_down", [("down", (down,), self._slab_rows("down", self.R[self.fmt(down)]))], F // 2,
                       G=2 * KC // F)

    def _slab_rows(self, name: str, R: int) -> np.ndarray:
        """(2 E, n) the rows of the expert halves' "gu" or "down" block of R-row tiles (module docstring)."""
        if (name, R) not in self._expert_rows:
            cfg = self.cfg
            H, E, F = cfg.hidden_size, cfg.n_experts, cfg.expert_ffn
            if name == "gu":
                rows = gate_up_rows(E * F, R, 2 * E)
            else:  # (rows of the view: half h of the file's row e H + o at 2 (e H + o) + h)
                k = np.arange(2 * E)[:, None]
                rows = group_rows(2 * (H * (k // 2) + np.arange(H)[None]) + k % 2, R, 2 * KC // F)
            self._expert_rows[(name, R)] = rows
        return self._expert_rows[(name, R)]
