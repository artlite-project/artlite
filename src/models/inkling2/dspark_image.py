"""Inkling 2's DSpark drafter on the NPU: its design and weight image (the overlay: ``inkling2.dspark``).

``DSparkDecodeOperator``: the decode design of 4 slots on fp32 states with quarter tails (Inkling's: ``decode.design``'s
``states``, ``qtails``), no MoE; the attention cores with q / k weights, NeoX RoPE and the block mask (``decode.attn``'s
``block``: every row of a step sees all its positions and the window's before them); the residual stream's F_INK steps
without the conv (``INK_PLAIN``: cur += the phase's rows). Its formats: the drafter's (Q8_0), BFP16 (``jh``) and the
model's head plane (Q8_0H: its LM region, read in place in the model's weight buffer), as pages.

``DSparkImage`` (``decode.image.NormedImage``; K chunks of 1024, the model's tile size: an input of the hidden 2304 is a
``Region.qt`` region), column-major; column c holds KV head c (query heads 3 c .. 3 c + 2) and the residual's channels
288 c ... A block's regions first (the phases between two attentions one run of the image each: one A task, ``decode
.attn_overlay``'s ``_weight_run``), then an injection's. Per layer L:

- "qkv" (``attn_norm``): blocks "q" (``attn_q`` rows 384 c ..), "k", "v" (``attn_k`` / ``attn_v`` rows 128 c ..);
- "o" (``attn_output`` rows 288 c .., K 3072), then "inko", "inkoq": the F_INK step's taps and queries objects (zero:
  ``INK_PLAIN`` reads no taps, the partial sums go nowhere);
- "gate_up" (``ffn_norm``): [8 gate | 8 up] windows of the column's 1152 outputs;
- "down<g>" (g < 3: ``ffn_down`` rows 288 c .., its columns 3072 g .. 3072 g + 3071: a K group each, one F_INK step each),
  then "inkd<g>", "inkdq<g>";

then "jh" (``output_norm``): the columns of ``joint_proj`` that the block's rows enter (its h part, 512 .. 2815: rows
96 c ..; BF16 as BFP16 tiles), and "lm_norm": the LM phase's norm-weight objects (``output_norm``; its tiles: the
model's); then per layer "ikv" (``blk.L.dflare_norm``: an injection's input, the taps' mix of the layer): "k", "v"
(``attn_k_ctx`` / ``attn_v_ctx`` rows 128 c ..)."""

from dataclasses import replace

import numpy as np

from artlite.decode.attn import Attn
from artlite.decode.attn_operator import AttentionDecodeOperator
from artlite.decode.image import NormedImage, column_rows, gate_up_rows
from artlite.decode.protocol import MAX_K
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import InklingDSparkConfig
from artlite.operators.design import COLS

from .decode import MAX_K as INK_MAX_K
from .image import KC

DOWN_K = 3072  # (a down region's K: the K group of a phase, ``decode`` MAX_K)
RANK_PARTS = 2  # (``joint_proj``'s columns before the h part: the state's and the previous token's, rank each)


class DSparkDecodeOperator(AttentionDecodeOperator):
    """The drafter's design (module docstring): ``rows``: {tile format: rows per tile} of tiles of ``tile_bytes`` (the
    model's); ``attn``: the attention cores (``Attn(block=True, states=True)``)."""

    PREFIX = "dsk"
    CHAIN = 1
    INK_PLAIN = True
    PAGES = True  # (the attention cores' program: q / k norms, RoPE, the states' skips and the block mask overflow 16 KB)
    STACK = 2304  # (the main core's: ``collect`` over a gate / up group's windows, as Inkling's)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float, attn: Attn,
                 paged: tuple[str, ...] | None = None):
        assert attn.block and attn.states, attn
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu",
                         packed=True, tok=True, max_k=INK_MAX_K, paged=tuple(sorted(rows)) if paged is None else paged,
                         states=True, qtails=True)


class DSparkImage(NormedImage):
    """The drafter's weights (module docstring) of a file ``gguf``, in tiles of ``tile_bytes`` (the model's)."""

    def __init__(self, gguf: GGUFFile, cfg: InklingDSparkConfig, *, tile_bytes: int):
        super().__init__(gguf, cfg, layers=list(range(cfg.layers)), kc=KC, payload=MAX_K, tile_bytes=tile_bytes,
                         vocab=0, qtails=True)
        H, hd, I = cfg.hidden, cfg.head_dim, cfg.ffn
        nq = cfg.heads // cfg.kv_heads
        assert cfg.kv_heads == COLS and H % (8 * COLS) == 0 and I % DOWN_K == 0 and nq == 3, cfg
        cols = np.arange(COLS)[:, None]
        heads = nq * cols + np.arange(nq)[None]
        q_rows = (hd * heads[..., None] + np.arange(hd)).reshape(COLS, -1)
        kv_rows = hd * cols + np.arange(hd)[None]
        for L in self.layers:
            b = f"blk.{L}."
            self._norm(L, "qkv")
            self._add(L, "qkv", [("q", (b + "attn_q.weight",), q_rows), ("k", (b + "attn_k.weight",), kv_rows),
                                 ("v", (b + "attn_v.weight",), kv_rows)], H)
            self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], cfg.heads * hd)
            self._zero(L, "inko")
            self._zero(L, "inkoq")
            self._norm(L, "gate_up")
            R = self.R[self.fmt(b + "ffn_gate.weight")]
            self._add(L, "gate_up", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
            for g in range(I // DOWN_K):
                self._add(L, f"down{g}", [("down", (b + "ffn_down.weight",), column_rows(H))], DOWN_K)
                r = self.regions[-1]
                r.blocks[0] = replace(r.blocks[0], kslice=(DOWN_K * g, DOWN_K))
                self._zero(L, f"inkd{g}")
                self._zero(L, f"inkdq{g}")
        self._norm(-1, "jh")
        self._add(-1, "jh", [("jh", (self._h_cols(),), column_rows(3 * cfg.rank))], H)
        self._norm(-1, "lm")
        for L in self.layers:  # (an injection's: after the block's, whose phases' regions are then one run each)
            b = f"blk.{L}."
            self._norm(L, "ikv")
            self._add(L, "ikv", [("k", (b + "attn_k_ctx.weight",), kv_rows), ("v", (b + "attn_v_ctx.weight",), kv_rows)], H)
        self._place()

    def _zero(self, L: int, name: str) -> None:
        """A data region of one zero object (the F_INK step's taps or queries: ``INK_PLAIN``)."""
        self._add_data(L, name, self.tile_bytes)

    def _h_cols(self) -> str:
        """``joint_proj``'s h columns (3 rank, hidden) as a BF16 tensor of their own (``_own``)."""
        name, cfg = "joint_proj.weight", self.cfg
        key = f"{name}[h]"
        if key not in self._own:
            t = self.gguf.tensor(name)
            c0 = RANK_PARTS * cfg.rank
            raw = np.asarray(t.raw).view(np.uint16).reshape(t.rows, t.cols)[:, c0:c0 + cfg.hidden]
            self._own[key] = QTensor(name=key, qtype="BF16", rows=t.rows, cols=cfg.hidden, raw=np.ascontiguousarray(raw))
        return key

    def norms(self, L: int, name: str) -> tuple[str | None, None]:
        if L < 0:
            return "output_norm.weight", None
        return {"qkv": f"blk.{L}.attn_norm.weight", "ikv": f"blk.{L}.dflare_norm.weight",
                "gate_up": f"blk.{L}.ffn_norm.weight"}[name], None

    def data_objects(self, layer: int, name: str) -> np.ndarray:
        if name.startswith("ink"):
            return np.zeros((COLS, 1, self.tile_bytes), np.uint8)
        return super().data_objects(layer, name)


__all__ = ["DOWN_K", "DSparkDecodeOperator", "DSparkImage"]
