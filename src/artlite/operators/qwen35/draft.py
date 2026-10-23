"""The multi-token-prediction drafter of a qwen35moe model (an MTP file: ``model_config.qwen_mtp_of``; llama.cpp's
``graph_mtp`` in ``src/models/qwen35moe.cpp``) on the model's decode overlay: a draft step is one dispatch of the
model's design (its formats include the drafter's: ``QwenOverlay(formats=)``) and hardware context, with the
drafter's weights, inputs and KV cache in buffers of its own.

The block (the file's layer L = the model's layer count): x0 = eh_proj [enorm(emb(t_q)) | hnorm(h_{q-1})], then a
full-attention layer on its own cache and an MoE layer, both as the model's (``QwenOverlay``'s phases on the drafter's
image: attn_in, the attention, attn_out, moe_in, the routing, sdown, the experts, the combine), then the head over the
residual normed by ``nextn.shared_head_norm``: the first ``HEAD_ROWS`` rows of the file's ``output.weight`` (the ids
follow the tokenizer's BPE merges, a frequency order: 98-99% of essay and code tokens). The row at position q (token
t_q, the model's output-normed hidden state of q - 1) writes its cache record at q and predicts t_{q+1}; the next
draft's h is its own residual normed by shared_head_norm (``hidden_state``).

Per step of S slots (``step``): the host writes the eh phase's input [enorm(emb) | hnorm(h)] (K 2 hidden: two K
groups) and zero residual slices; the eh phase (F_RESID | F_EMITR) adds eh_proj's rows into them and sends the
residual to attn_in. A generation round after a verify step at q .. q + S - 1 that accepted a tokens: one step of
a + 1 slots at q + 1 (the accepted tokens' rows, from the verify's hidden states, and the new token's: the first
draft) rewrites the rows the last round's drafts wrote, then one-slot steps (the next drafts).
"""

import dataclasses

import numpy as np
import torch

from ...gguf_model import GGUFFile
from ...model_config import QwenConfig
from ..overlay.decode import interleave, norm_objects
from ..overlay.image import column_rows
from ..overlay.runtime import Sequence
from .image import QwenImage
from .overlay import QwenOverlay

HEAD_ROWS = 65536  # the draft head's rows: the first ids of the vocabulary


def mtp_config(gguf: GGUFFile) -> QwenConfig:
    """The drafter file's config with its block as a layer of its own: layer ``num_hidden_layers`` (an attention layer)
    after the trunk's (which the file does not hold)."""
    cfg = QwenConfig.from_gguf(gguf)
    if not cfg.moe or cfg.nextn_layers != 1:
        raise NotImplementedError(f"{gguf.path}: a qwen35moe file with one MTP block (moe {cfg.moe}, blocks "
                                  f"{cfg.nextn_layers})")
    return dataclasses.replace(cfg, num_hidden_layers=cfg.num_hidden_layers + 1, attention=(*cfg.attention, True),
                               nextn_layers=0)


class DraftImage(QwenImage):
    """The drafter's weights: per its layer "eh" (``nextn.eh_proj``, rows hidden / 8 c .. over K 2 hidden), then the
    layer's regions (an attention layer and its MoE: ``QwenImage``'s), the head's first ``vocab`` rows normed by
    ``nextn.shared_head_norm``; tiles of the model image's 9280 bytes (16 rows of Q4_0)."""

    def __init__(self, gguf: GGUFFile, cfg: QwenConfig, *, layers: list[int], vocab: int | None = None):
        super().__init__(gguf, cfg, layers=layers, vocab=vocab, tile_bytes=9280)

    def _layer(self, layer: int) -> None:
        H = self.cfg.hidden_size
        self._add(layer, "eh", [("eh", (f"blk.{layer}.nextn.eh_proj.weight",), column_rows(H))], 2 * H)
        super()._layer(layer)

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        if layer >= 0:
            return super().norm_objects(layer, name)
        w = self.gguf.tensor(f"blk.{self.layers[0]}.nextn.shared_head_norm.weight").dequantize()
        return norm_objects(w.numpy().reshape(-1), self.tile_bytes).view(np.uint8)


class _DraftConfig:
    from_gguf = staticmethod(mtp_config)


class QwenDraft(QwenOverlay):
    """The drafter ``gguf`` (a qwen35moe MTP file) of the model on ``target`` (whose design runs the drafter's formats:
    ``QwenOverlay(formats=)``); ``vocab``: the head's rows. ``load`` after the target's (its program and context)."""

    CONFIG, IMAGE = _DraftConfig, DraftImage

    def __init__(self, target: QwenOverlay, gguf: GGUFFile, *, vocab: int = HEAD_ROWS):
        cfg = mtp_config(gguf)
        tc = target.cfg
        if (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.n_experts,
                cfg.n_experts_used, cfg.expert_ffn, cfg.shared_ffn) != (
                tc.hidden_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.n_experts,
                tc.n_experts_used, tc.expert_ffn, tc.shared_ffn):
            raise ValueError(f"{gguf.path}: the drafter of another model than {target.gguf.path}")
        self.L = cfg.num_hidden_layers - 1
        super().__init__(gguf, layers=[self.L], vocab=vocab, max_positions=target.max_positions)
        assert vars(self.ash) == vars(target.ash), "the drafter's attention: the design's"
        fmts = {b.fmt for r in self.image.regions + self.image.slabs for b in r.blocks}
        if not fmts <= set(target.op.rows):
            raise ValueError(f"the design's formats {sorted(target.op.rows)} lack the drafter's {sorted(fmts)}: "
                             f"QwenOverlay(formats=)")
        self.target, self.op = target, target.op  # (the model's design and hardware context: ``load``)

    def _phases(self) -> list[tuple[int, str]]:
        return [(self.L, "eh"), *super()._phases()]

    def _input(self, L: int) -> tuple[int, str]:
        return (L, "eh") if L == self.L else super()._input(L)

    def _head(self, seq: Sequence, S: int) -> None:
        self._phase(seq, (self.L, "eh"), S, (self.L, "attn_in"))

    def _f32(self, name: str) -> np.ndarray:
        return super()._f32(f"blk.{self.L}.nextn.shared_head_norm.weight" if name == "output_norm.weight" else name)

    def inputs(self, emb: torch.Tensor, h: torch.Tensor) -> np.ndarray:
        """The eh phase's input (S, 2 hidden) fp32: [enorm(emb) | hnorm(h)] of the slots' token embeddings and the
        hidden states before them (S, hidden each)."""
        eps, b = self.cfg.rms_norm_eps, f"blk.{self.L}.nextn."

        def norm(x: torch.Tensor, w: str) -> np.ndarray:
            x = x.float().numpy().astype(np.float32)
            r = 1 / np.sqrt((x * x).mean(-1, keepdims=True, dtype=np.float32) + np.float32(eps))
            return x * r * self._f32(b + w)

        return np.concatenate([norm(emb, "enorm.weight"), norm(h, "hnorm.weight")], -1)

    def step(self, emb: torch.Tensor, h: torch.Tensor, position: int, *, logits: bool = True
             ) -> torch.Tensor | None:
        """A step of S = len(emb) <= 4 rows at positions position .. position + S - 1 (their cache records): emb (S,
        hidden) their tokens' embeddings, h (S, hidden) the hidden states of the positions before them. Returns the
        head's logits (S, vocab) bf16 (a view of the NPU's buffer, valid until the next step), or None."""
        x = self.inputs(emb.reshape(-1, self.H), h.reshape(-1, self.H))
        S = x.shape[0]
        assert 1 <= S <= self.SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._write_loadr(np.zeros((S, self.H), np.float32))
        for rg in self.inp[(self.L, "eh")]:
            self._write_input(rg, interleave(x[:, rg.k0:rg.k0 + rg.K]).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def hidden_state(self, slot: int = 0) -> torch.Tensor:
        """The next draft's h of a slot of the last step: its residual before the head, normed by shared_head_norm
        (hidden,) fp32."""
        x = self.hidden(-1, slot).numpy()
        eps = np.float32(self.cfg.rms_norm_eps)
        r = 1 / np.sqrt((x * x).mean(dtype=np.float32) + eps)
        return torch.from_numpy(x * r * self._f32(f"blk.{self.L}.nextn.shared_head_norm.weight"))


__all__ = ["HEAD_ROWS", "DraftImage", "QwenDraft", "mtp_config"]
