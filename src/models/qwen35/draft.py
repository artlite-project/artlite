"""The multi-token-prediction drafter of a qwen35 / qwen35moe model on the model's decode overlay (llama.cpp's
``graph_mtp`` in ``src/models/qwen35moe.cpp`` / ``qwen35.cpp``): a draft step is one dispatch of the model's design and
hardware context, with the drafter's inputs and KV cache in separate buffers. Its block comes either from an MTP file
(``model_config.qwen_mtp_of``: Qwen3.6-35B-A3B's; the design runs the file's formats too, ``QwenOverlay(formats=)``),
with its weights in a separate buffer, or from the model file itself (Qwen3.8-27B: ``QwenImage``'s MTP block, which
the drafter streams from the model's buffer, its head the first rows of the model's: ``QwenImage.mtp_view``).

The block (layer L = the model's layer count): x0 = eh_proj [enorm(emb(t_q)) | hnorm(h_{q-1})], then a full-attention
layer with a separate KV cache and an FFN (an MoE layer, or a dense one), both as the model's (``QwenOverlay``'s phases
on the drafter's image: attn_in, the attention, attn_out, then moe_in, the routing, sdown, the experts, the combine; or
gate_up, down), then the head over the residual normed by ``nextn.shared_head_norm``: rows of the file's
``output.weight``, ``HeadTable.ids`` the token of each logit. ``HEAD_ROWS`` fixed rows: the special tokens (``head_tokens``: the turn's end, tool calls, a thought's end) and
the first ids (the tokenizer's BPE merges, a frequency order: 98.6% of essay tokens, 97.9% of agents' generated ones);
then a table of ``TABLE_ROWS`` of the sequence's own tokens past them (``HeadTable.see``: the most recently seen;
identifiers of the code an agent reads leave 0.2% of its tokens out of the head, against 2.1% without; text in other
scripts, CJK or Cyrillic, keeps 6-17% out: tokens it had not used before). The row at position q (token t_q, the model's
output-normed hidden state of q - 1) writes its cache record at q and predicts t_{q+1}; the next draft's h is this row's
residual normed by shared_head_norm (``hidden_state``).

Per step of S slots (``step``): the host writes the eh phase's input [enorm(emb) | hnorm(h)] (K 2 hidden: two K
groups) and zero residual slices; the eh phase (F_RESID | F_EMITR) adds eh_proj's rows into them and sends the
residual to attn_in. A generation round after a verify step at q .. q + S - 1 that accepted a tokens: one step of
a + 1 slots at q + 1 (the accepted tokens' rows, from the verify's hidden states, and the new token's: the first
draft) rewrites the rows the last round's drafts wrote, then one-slot steps (the next drafts).
"""

import dataclasses

import numpy as np

from artlite.decode.attn256 import OBJ_BYTES
from artlite.decode.head import HEAD_ROWS, TABLE_ROWS, HeadTable, head_tokens
from artlite.decode.image import column_rows
from artlite.decode.protocol import interleave, norm_objects
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.operators.design import COLS
from artlite.overlay import Sequence

from .image import QwenImage
from .overlay import QwenOverlay


def mtp_config(gguf: GGUFFile) -> QwenConfig:
    """The config of a file with one MTP block (an MTP file, or a model file that holds its own block), with that block
    as a separate layer: layer ``num_hidden_layers`` (an attention layer) after the trunk's layers."""
    cfg = QwenConfig.from_gguf(gguf)
    if cfg.nextn_layers != 1:
        raise NotImplementedError(f"{gguf.path}: a file with one MTP block, not {cfg.nextn_layers}")
    return dataclasses.replace(cfg, num_hidden_layers=cfg.num_hidden_layers + 1, attention=(*cfg.attention, True),
                               nextn_layers=0)


class DraftImage(QwenImage):
    """The drafter's weights: per its layer "eh" (``nextn.eh_proj``, rows hidden / 8 c .. over K 2 hidden), then the
    layer's regions (an attention layer and its MoE: ``QwenImage``'s), the head normed by ``nextn.shared_head_norm``:
    ``vocab`` rows (default ``HEAD_ROWS`` + ``TABLE_ROWS``), per column its share of the fixed tokens (``head_tokens``)
    then of the table's; tiles of the model image's 9280 bytes (16 rows of Q4_0)."""

    SCREEN = False  # (a draft's logits: its head's rows as they are)
    ROT_IN = False  # (rotated: its layer's copies come from its eh phase)

    def __init__(self, gguf: GGUFFile, cfg: QwenConfig, *, layers: list[int], vocab: int | None = None,
                 rotated: bool = False):
        """rotated: the drafter of a PARO checkpoint (its design's phases: ``QwenImage``'s ``rotated``, the eh phase's
        copies of the residual for the layer's q, k, v, "eh_rot")."""
        self.fixed, self.table = head_tokens(gguf, (vocab or HEAD_ROWS + TABLE_ROWS) - TABLE_ROWS)
        super().__init__(gguf, cfg, layers=layers, vocab=len(self.fixed) + len(self.table), tile_bytes=OBJ_BYTES,
                         rotated=rotated)

    def head_rows(self) -> np.ndarray:
        """(per column: its fixed tokens, then its table's)"""
        return np.concatenate([self.fixed.reshape(COLS, -1), self.table.reshape(COLS, -1)], 1)

    def _layer(self, layer: int) -> None:
        H = self.cfg.hidden_size
        self._add(layer, "eh", [("eh", (f"blk.{layer}.nextn.eh_proj.weight",), column_rows(H))], 2 * H)
        super()._layer(layer)

    def _paro_layer(self, layer: int) -> None:
        H = self.cfg.hidden_size
        self._add(layer, "eh", [("eh", (f"blk.{layer}.nextn.eh_proj.weight",), column_rows(H))], 2 * H)
        self.rot_data(layer, "eh_rot", self._in_rot(layer))
        super()._paro_layer(layer)

    def _vec(self, name: str) -> np.ndarray:
        """(the head's norm: the block's shared_head_norm)"""
        return super()._vec(f"blk.{self.layers[0]}.nextn.shared_head_norm.weight" if name == "output_norm.weight" else name)

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        if layer >= 0:
            return super().norm_objects(layer, name)
        w = self.gguf.tensor(f"blk.{self.layers[0]}.nextn.shared_head_norm.weight").dequantize()
        return norm_objects(w.reshape(-1), self.tile_bytes).view(np.uint8)


def _draft_image(gguf: GGUFFile) -> DraftImage:
    cfg = mtp_config(gguf)
    return DraftImage(gguf, cfg, layers=[cfg.num_hidden_layers - 1])


def draft_formats(gguf: GGUFFile) -> tuple[str, ...]:
    """The tile formats of a drafter file's image: its model's design runs them (``QwenOverlay(formats=)``)."""
    return tuple(sorted(_draft_image(gguf).formats()))


def draft_gate_up(gguf: GGUFFile) -> frozenset[int]:
    """The rows per tile of a drafter file's gate / up tiles: its model's design takes them (``QwenOverlay(gu=)``)."""
    return frozenset(_draft_image(gguf).gu_rows())


class _DraftConfig:
    from_gguf = staticmethod(mtp_config)


class QwenDraft(QwenOverlay):
    """The drafter ``gguf`` (an MTP file; None: the block of the model's own file, ``QwenImage.mtp_view``) of the model
    on ``target`` (whose design runs the drafter's formats: ``QwenOverlay(formats=)``; a PARO checkpoint's: its block
    laid out as a PARO layer of identity rotations, ``DraftImage``'s ``rotated``, its steps ``paro_steps``'). ``load``
    after the target's (its program and context; the model's own block: its weight buffer). ``table``: its head's tokens. A target of
    several sequences: a cache of each, the selected one the target's (``ModelOverlay.followers``); the table is
    one, of every sequence's tokens."""

    CONFIG, IMAGE = _DraftConfig, DraftImage
    STEPS_AHEAD = False  # (its steps follow the model's rounds, not each other)
    ROT_IN = False  # (a PARO checkpoint's drafter: its layer's copies from its eh phase, ``paro_steps``)

    def __init__(self, target: QwenOverlay, gguf: GGUFFile | None = None):
        own = gguf is None
        gguf = target.gguf if own else gguf
        cfg = mtp_config(gguf)
        tc = target.cfg
        if (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.n_experts,
                cfg.n_experts_used, cfg.expert_ffn, cfg.shared_ffn) != (
                tc.hidden_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.n_experts,
                tc.n_experts_used, tc.expert_ffn, tc.shared_ffn):
            raise ValueError(f"{gguf.path}: the drafter of another model than {target.gguf.path}")
        self.L = cfg.num_hidden_layers - 1
        image = target.image.mtp_view(cfg) if own else (DraftImage(gguf, cfg, layers=[self.L], rotated=True)
                                                         if target.paro else None)
        super().__init__(gguf, layers=[self.L], max_positions=target.max_positions, image=image, slots=target.SLOTS,
                         sequences=target.sequences)
        target.followers.append(self)  # (its cache of each of the model's sequences: selected with the model's)
        self.w_of = target if own else None
        assert vars(self.ash) == vars(target.ash), "the drafter's attention: the design's"
        fmts = self.image.formats()
        if not fmts <= set(target.op.rows):
            raise ValueError(f"the design's formats {sorted(target.op.rows)} lack the drafter's {sorted(fmts)}: "
                             f"QwenOverlay(formats=draft_formats(gguf))")
        if self.image.gu_rows() != {16} and not target.op.gu_win:
            raise ValueError(f"the drafter's gate / up tiles of {sorted(self.image.gu_rows())} rows need the design's "
                             f"windows: QwenOverlay(gu=draft_gate_up(gguf))")
        self.target, self.op = target, target.op  # (the model's design and hardware context: ``load``)
        self.table = HeadTable(self.image, cfg.vocab_size, lambda: self.dev)

    def _phases(self) -> list[tuple[int, str]]:
        return [(self.L, "eh"), *super()._phases()]

    def _input(self, L: int) -> tuple[int, str]:
        return (L, "eh") if L == self.L else super()._input(L)

    def _head(self, seq: Sequence, S: int) -> None:
        self._phase(seq, (self.L, "eh"), S, (self.L, "attn_in"))

    def _f32(self, name: str) -> np.ndarray:
        return super()._f32(f"blk.{self.L}.nextn.shared_head_norm.weight" if name == "output_norm.weight" else name)

    def inputs(self, emb: np.ndarray, h: np.ndarray) -> np.ndarray:
        """The eh phase's input (S, 2 hidden) fp32: [enorm(emb) | hnorm(h)] of the slots' token embeddings and the
        hidden states before them (S, hidden each)."""
        eps, b = self.cfg.rms_norm_eps, f"blk.{self.L}.nextn."

        def norm(x: np.ndarray, w: str) -> np.ndarray:
            x = np.asarray(x, np.float32)
            r = 1 / np.sqrt((x * x).mean(-1, keepdims=True, dtype=np.float32) + np.float32(eps))
            return x * r * self._f32(b + w)

        return np.concatenate([norm(emb, "enorm.weight"), norm(h, "hnorm.weight")], -1)

    def step(self, emb: np.ndarray, h: np.ndarray, position: int, *, logits: bool = True
             ) -> np.ndarray | None:
        """A step of S = len(emb) <= SLOTS rows at positions position .. position + S - 1 (their cache records): emb (S,
        hidden) their tokens' embeddings, h (S, hidden) the hidden states of the positions before them. Returns the
        head's logits (S, vocab) bf16 (the tokens ``ids``; a view of the NPU's buffer, valid until the next step), or
        None."""
        x = self.inputs(emb.reshape(-1, self.H), h.reshape(-1, self.H))
        S = x.shape[0]
        assert 1 <= S <= self.SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        self._set_branch(None, position, S)  # (its attention's rows' masks and K lanes)
        self._write_loadr(np.zeros((S, self.H), np.float32))
        for rg in self.inp[(self.L, "eh")]:
            self._write_input(rg, interleave(x[:, rg.k0:rg.k0 + rg.K], self._sp(S, (self.L, "eh"))).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def hidden_state(self, slot: int = 0) -> np.ndarray:
        """The next draft's h of a slot of the last step: its residual before the head, normed by shared_head_norm
        (hidden,) fp32."""
        x = self.hidden(-1, slot)
        eps = np.float32(self.cfg.rms_norm_eps)
        r = 1 / np.sqrt((x * x).mean(dtype=np.float32) + eps)
        return x * r * self._f32(f"blk.{self.L}.nextn.shared_head_norm.weight")


__all__ = ["HEAD_ROWS", "TABLE_ROWS", "DraftImage", "HeadTable", "QwenDraft", "draft_formats", "draft_gate_up", "head_tokens",
           "mtp_config"]
