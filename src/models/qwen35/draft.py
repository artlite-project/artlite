"""The multi-token-prediction drafter of a qwen35 / qwen35moe model on the model's decode overlay (llama.cpp's
``graph_mtp`` in ``src/models/qwen35moe.cpp`` / ``qwen35.cpp``): a draft step is one dispatch of the model's design and
hardware context, with the drafter's inputs and KV cache in buffers of its own. Its block comes from an MTP file
(``model_config.qwen_mtp_of``: Qwen3.6-35B-A3B's; the design runs the file's formats too, ``QwenOverlay(formats=)``),
its weights in a buffer of their own, or from the model file itself (Qwen3.8-27B: ``QwenImage``'s MTP block, which
the drafter streams from the model's buffer, its head the first rows of the model's: ``QwenImage.mtp_view``).

The block (layer L = the model's layer count): x0 = eh_proj [enorm(emb(t_q)) | hnorm(h_{q-1})], then a full-attention
layer on its own cache and an FFN (an MoE layer, or a dense one), both as the model's (``QwenOverlay``'s phases on the
drafter's image: attn_in, the attention, attn_out, then moe_in, the routing, sdown, the experts, the combine; or
gate_up, down), then the head over the residual normed by ``nextn.shared_head_norm``: rows of the file's
``output.weight``, ``QwenDraft.ids`` the token of each logit. ``HEAD_ROWS`` fixed rows: the special tokens (``head_tokens``: the turn's end, tool calls, a thought's end) and
the first ids (the tokenizer's BPE merges, a frequency order: 98.6% of essay tokens, 97.9% of agents' generated ones);
then a table of ``TABLE_ROWS`` of the sequence's own tokens past them (``QwenDraft.see``: the most recently seen;
identifiers of the code an agent reads leave 0.2% of its tokens out of the head, against 2.1% without; text in other
scripts, CJK or Cyrillic, keeps 6-17% out: tokens it had not used before). The row at position q (token t_q, the model's
output-normed hidden state of q - 1) writes its cache record at q and predicts t_{q+1}; the next draft's h is its own
residual normed by shared_head_norm (``hidden_state``).

Per step of S slots (``step``): the host writes the eh phase's input [enorm(emb) | hnorm(h)] (K 2 hidden: two K
groups) and zero residual slices; the eh phase (F_RESID | F_EMITR) adds eh_proj's rows into them and sends the
residual to attn_in. A generation round after a verify step at q .. q + S - 1 that accepted a tokens: one step of
a + 1 slots at q + 1 (the accepted tokens' rows, from the verify's hidden states, and the new token's: the first
draft) rewrites the rows the last round's drafts wrote, then one-slot steps (the next drafts).
"""

import dataclasses
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import torch

from artlite.decode.design import interleave, norm_objects
from artlite.decode.image import column_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.operators.design import COLS
from artlite.overlay import Sequence

from .attn import OBJ_BYTES
from .image import HEAD_ROWS, TABLE_ROWS, QwenImage, head_tokens
from .overlay import QwenOverlay


def mtp_config(gguf: GGUFFile) -> QwenConfig:
    """The config of a file with one MTP block (an MTP file, or a model's own) with its block as a layer of its own:
    layer ``num_hidden_layers`` (an attention layer) after the trunk's."""
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

    def __init__(self, gguf: GGUFFile, cfg: QwenConfig, *, layers: list[int], vocab: int | None = None):
        self.fixed, self.table = head_tokens(gguf, (vocab or HEAD_ROWS + TABLE_ROWS) - TABLE_ROWS)
        super().__init__(gguf, cfg, layers=layers, vocab=len(self.fixed) + len(self.table), tile_bytes=OBJ_BYTES)

    def head_rows(self) -> np.ndarray:
        """(per column: its fixed tokens, then its table's)"""
        return np.concatenate([self.fixed.reshape(COLS, -1), self.table.reshape(COLS, -1)], 1)

    def _layer(self, layer: int) -> None:
        H = self.cfg.hidden_size
        self._add(layer, "eh", [("eh", (f"blk.{layer}.nextn.eh_proj.weight",), column_rows(H))], 2 * H)
        super()._layer(layer)

    def norm_objects(self, layer: int, name: str) -> np.ndarray:
        if layer >= 0:
            return super().norm_objects(layer, name)
        w = self.gguf.tensor(f"blk.{self.layers[0]}.nextn.shared_head_norm.weight").dequantize()
        return norm_objects(w.numpy().reshape(-1), self.tile_bytes).view(np.uint8)


def _formats(img: QwenImage) -> set[str]:
    return {b.fmt for r in img.regions + img.slabs for b in r.blocks}


def draft_formats(gguf: GGUFFile) -> tuple[str, ...]:
    """The tile formats of a drafter file's image: its model's design runs them (``QwenOverlay(formats=)``)."""
    cfg = mtp_config(gguf)
    return tuple(sorted(_formats(DraftImage(gguf, cfg, layers=[cfg.num_hidden_layers - 1]))))


class _DraftConfig:
    from_gguf = staticmethod(mtp_config)


class QwenDraft(QwenOverlay):
    """The drafter ``gguf`` (an MTP file; None: the block of the model's own file, ``QwenImage.mtp_view``) of the model
    on ``target`` (whose design runs the drafter's formats: ``QwenOverlay(formats=)``). ``load`` after the target's
    (its program and context; the model's own block: its weight buffer). ``ids`` (vocab,): the token of each of the
    head's logits; ``table_rows``: the logits of the table (``see``)."""

    CONFIG, IMAGE = _DraftConfig, DraftImage

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
        super().__init__(gguf, layers=[self.L], max_positions=target.max_positions,
                         image=target.image.mtp_view(cfg) if own else None)
        self.w_of = target if own else None
        assert vars(self.ash) == vars(target.ash), "the drafter's attention: the design's"
        fmts = _formats(self.image)
        if not fmts <= set(target.op.rows):
            raise ValueError(f"the design's formats {sorted(target.op.rows)} lack the drafter's {sorted(fmts)}: "
                             f"QwenOverlay(formats=draft_formats(gguf))")
        self.target, self.op = target, target.op  # (the model's design and hardware context: ``load``)
        img = self.image
        self.ids = img.head_rows().reshape(-1)
        self._per = self.vocab // COLS  # (a column's logits)
        self.table_rows = np.flatnonzero(np.arange(self.vocab) % self._per >= len(img.fixed) // COLS)
        self._fixed = np.zeros(cfg.vocab_size, bool)
        self._fixed[img.fixed] = True
        idle = sorted(zip(self.ids[self.table_rows].tolist(), self.table_rows.tolist()), reverse=True)
        self._lru = OrderedDict(idle)  # token -> its logit, least recently seen first (idle: the last ids first)

    def see(self, ids: Iterable[int]) -> None:
        """Tokens of the sequence in order (an image's rows, negative: none): the table holds the ``TABLE_ROWS`` most
        recently seen ones outside the fixed rows (the least recently seen leaves), the head's rows of those that
        enter packed into the weight image (the next step's)."""
        a = np.fromiter(ids, np.int64)
        a = a[(a >= 0) & (a < len(self._fixed))]
        a = a[~self._fixed[a]]
        if not a.size:
            return
        u, last = np.unique(a[::-1], return_index=True)
        order = u[np.argsort(-last)][-len(self.table_rows):].tolist()  # (the last ones, by their last occurrence)
        lru = self._lru
        for t in order:
            if t in lru:
                lru.move_to_end(t)  # (they stay: the others leave first)
        new = [t for t in order if t not in lru]
        for t in new:
            lru[t] = lru.popitem(last=False)[1]
        for t in order:
            lru.move_to_end(t)
        if new:
            self._write([lru[t] for t in new], new)

    def _write(self, rows: list[int], tokens: list[int]) -> None:
        """The head's logits ``rows`` are the ``tokens``' now: their row tiles packed again, flushed."""
        r = self.image.region(-1, "lm")
        b = r.block("lm")
        tiles = set()
        for i, t in zip(rows, tokens, strict=True):
            c, j = divmod(i, self._per)
            self.ids[i] = b.rows[c, j] = t
            tiles.add((c, j // (b.R * b.G)))
        w = self.dev.view("W")
        for c, rt in sorted(tiles):
            for off, n in self.image.repack(w, r, b, c, rt):
                self.dev.flush("W", off, n)

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
        head's logits (S, vocab) bf16 (the tokens ``ids``; a view of the NPU's buffer, valid until the next step), or
        None."""
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


__all__ = ["HEAD_ROWS", "TABLE_ROWS", "DraftImage", "QwenDraft", "draft_formats", "head_tokens", "mtp_config"]
