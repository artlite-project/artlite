"""An MoE layer's regions in a weight image (``decode.image.TileImage``; the qwen35moe and flint models'), as a mixin
of the model's image class (``MoeImage``). Per MoE layer:

- "moe_in" (normed by the layer's FFN norm): "sgu", the shared expert's gate / up rows (``decode.image.gate_up_rows``);
  "router", the router's rows of the column's experts E / 8 c .. (the F32 or BF16 ``ffn_gate_inp`` as BFP16 planes,
  ``decode.image.Block``: per expert its hi and lo rows, whose outputs the decode adds: the logits' order of llama.cpp's
  fp32 up to ~3e-5 of a weight); with a gated shared expert "sgate", its gate row (``ffn_gate_inp_shexp``: its hi and lo
  planes) in every column. The router's tensors are packed times w / bf16(w), w the FFN norm's weights
  (``scaled_planes``): the decode's norm multiplies x by bf16 weights (``decode.protocol.norm_objects``), the logits need
  w's fp32 (with the bf16 weights alone they are 8e-4 of their largest off, against 1.3e-5);
- "sdown": the shared expert's down, rows hidden / 8 c .., its K (Fs) as row groups of kc / Fs per tile row
  (``decode.image.group_rows``; one chunk at Fs = kc: plain tiles);
- the experts as slab regions (``TileImage.slabs``; half k = 2 e + h of expert e: its outputs h F / 2 .. of the expert's
  F = ``expert_ffn``): "exp_gu", the half's gate / up rows (``gate_up_rows`` over the 3-D ``ffn_gate_exps`` /
  ``ffn_up_exps``: expert e's rows e F ..), then "exp_down", the down's rows (all hidden of them) over the half's K
  (F / 2 columns, h F / 2 .., of ``ffn_down_exps``' rows e hidden ..) as row groups of kc / (F / 2) (4 for F 512).
  Expert e's two halves are one run of the image (``Region.col_stride`` apart: half 2 e + 1 right after half 2 e):
  a decode column streams a whole expert, or one half of one.
"""

from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from ..decode.image import column_rows, gate_up_rows, group_rows
from ..gguf_model import QTensor
from ..operators.design import COLS


class MoeImage:
    """An image's MoE layers (``_moe``) and their router planes at pack time (``scaled_planes``). The model's image
    holds ``cfg`` (``hidden_size``, ``n_experts``, ``expert_ffn``, ``shared_ffn``) and ``kc``."""

    cfg: Any
    kc: int

    def qtype(self, name: str) -> str:
        """(the router's tensors: F32, as ``scaled_planes`` packs them)"""
        return "F32" if name in self.__dict__.get("_planes", ()) else super().qtype(name)

    def _moe(self, layer: int, *, gated: bool) -> None:
        """An MoE layer's regions (module docstring); gated: the shared expert's sigmoid gate row ("sgate")."""
        cfg, b = self.cfg, f"blk.{layer}."
        H, E, F, Fs, KC = cfg.hidden_size, cfg.n_experts, cfg.expert_ffn, cfg.shared_ffn, self.kc
        assert E % COLS == 0 and Fs % (64 * COLS) == 0 and KC % Fs == 0 and KC % (F // 2) == 0 and F % 128 == 0, cfg
        mine = E // COLS * np.arange(COLS)[:, None] + np.arange(E // COLS)[None]  # (the column's experts)
        planes = (b + "ffn_gate_inp.weight", *((b + "ffn_gate_inp_shexp.weight",) if gated else ()))  # (rows 0 .. E)
        self.__dict__.setdefault("_planes", set()).update(planes)
        R = self.R[self.fmt(b + "ffn_gate_shexp.weight")]
        self._add(layer, "moe_in", [("sgu", (b + "ffn_gate_shexp.weight", b + "ffn_up_shexp.weight"),
                                     gate_up_rows(Fs, R)),
                                    ("router", planes, np.stack([2 * mine, 2 * mine + 1], -1).reshape(COLS, -1)),
                                    *([("sgate", planes, np.tile([2 * E, 2 * E + 1], (COLS, 1)))] if gated else [])], H)
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
        cache = self.__dict__.setdefault("_expert_rows", {})  # (every layer's slab blocks share them)
        if (name, R) not in cache:
            cfg = self.cfg
            H, E, F = cfg.hidden_size, cfg.n_experts, cfg.expert_ffn
            if name == "gu":
                rows = gate_up_rows(E * F, R, 2 * E)
            else:  # (rows of the view: half h of the file's row e H + o at 2 (e H + o) + h)
                k = np.arange(2 * E)[:, None]
                rows = group_rows(2 * (H * (k // 2) + np.arange(H)[None]) + k % 2, R, 2 * self.kc // F)
            cache[(name, R)] = rows
        return cache[(name, R)]

    def router_planes(self, layer: int) -> tuple[str, ...]:
        """An MoE layer's router tensors (``scaled_planes``)."""
        b = f"blk.{layer}."
        return tuple(t for t in (b + "ffn_gate_inp.weight", b + "ffn_gate_inp_shexp.weight") if t in self._planes)

    def scaled_planes(self, norm: str, tensors: tuple[str, ...]) -> list[str]:
        """Substitute F32 copies of ``tensors`` (F32 or BF16 matrices read as BFP16 planes) times w / bf16(w), w the
        norm ``norm``'s weights (``_own``, until the caller drops them): the names substituted."""
        w = self.gguf.tensor(norm).dequantize().reshape(-1)
        bw = w.astype(bfloat16).astype(np.float32)
        ratio = np.where(bw != 0, w / np.where(bw != 0, bw, 1), 1).astype(np.float32)
        for name in tensors:
            t = self.gguf.tensor(name)
            raw = (np.asarray(t.dequantize(), np.float32).reshape(t.rows, t.cols) * ratio).astype(np.float32)
            self._own[name] = QTensor(name=name, qtype="F32", rows=t.rows, cols=t.cols, raw=raw)
        return list(tensors)
