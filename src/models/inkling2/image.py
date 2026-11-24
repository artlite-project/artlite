"""The weight image of an Inkling 2 model (``decode.image.NormedImage``): the file's tile formats over K chunks of 1024
columns (``KC``), objects of 9216 B (Q8_0's 8 rows, NVFP4's 16, BFP16's 8); a K of no whole number of chunks (the
hidden 2304, the controllers' 4608) is padded to the next one (zero super-blocks past the matrix).

Column-major: a column's regions of a physical layer in a visit's order (the loop core's 3 passes stream the same
regions), each normed phase's norm-weight objects before its tiles, the experts' slabs after the layers. Column c
holds KV head c's attention (its query heads 3 c .. 3 c + 2) and the residual's channels 288 c .. (``H8``). Per layer
L:

- "inkq" (L > 0; and the LM head's): the ink core's queries object (``ink_queries``): the column's slices of the
  layer's depth-mix queries of its attention input (1, or 3 on a full layer), then of its 4 message queries (loop layers),
  fp32 [query][288]: the visit before reads it for the partial sums of this visit's mixes;
- loop layers: "dq", "dk", "dv" (the controllers' down of the q, k, v streams: 64 rows of K 4608, rows 8 c .. of each
  column as two BFP16 planes: B_PAIR) and "ctl_q", "ctl_k", "ctl_v" (the ink core's: per stream the column's rows of the
  forget and gain matrices, bf16 [288][64] each, 4 objects each: ``ctl_objects``);
- "q" (normed: ``attn_norm``, a full layer's ``attn_q_inp_norm``): blocks "q" (``attn_q`` rows 384 c ..), "gate"
  (``attn_gate`` rows 384 c ..), "r" (``attn_rel`` rows 48 c ..), "sink" (``attn_sink_proj`` rows 3 c .., two planes);
- "k" (``attn_norm`` / ``attn_k_inp_norm``): "k" (``attn_k`` rows 128 c ..), a full layer's "kd" (``attn_k_conv_proj``
  rows 4 c .., two planes); "v" likewise;
- "o" (``attn_output`` rows 288 c .., K 3072);
- "inkf": the ink core's object of the o phase: the FFN mix's query, the attention's post-conv taps (``ink_objects``);
- loop layers: "dffn", "ctl_ffn" (the FFN stream's controller);
- "router" (``ffn_norm``): the router's rows of the column's experts 16 c .., two planes (fp32 logits: B_PAIR);
- "inkc": the ink core's object of the combine: the FFN's post-conv taps.

The BF16 tensors read as two planes (sink, the tap deltas) are substituted by F32 copies of theirs (exact: ``_own``).

The experts (``EXPERT_UNITS``): an expert's h (``expert_ffn`` 1152) in units of 512, 512 and 128 channels; unit u of
expert e has its gate / up rows ("exp_gu<u>": groups of 64 outputs, 8 x [8 gate | 8 up] each, over K 2304) and its
down's rows over the unit's K slice ("exp_dn<u>": every hidden row of the expert, ``Block.kslice``, as row groups of
kc / 512 = 2 or kc / 256 = 4 rows, the last unit's 128 columns zero-padded to 256). The six slab regions back to back
per expert: expert e's units are one run of the image.

"lm": rows vocab / 8 c .. of ``output`` (the vocabulary padded to whole C objects per column; rows past the file's
vocabulary zero), screened (``decode.screen``: the Q8_0 head's high plane, Q8_0H).
"""

from dataclasses import replace

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode.image import NormedImage, column_rows, f32_planes, group_rows
from artlite.decode.protocol import C_LEN, MAX_K
from artlite.decode.screen import plane, screened
from artlite.formats import FORMATS, gu_tiles
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import InklingConfig
from artlite.operators.design import COLS

KC = 1024  # K chunk of a tile
EXPERT_UNITS = ((0, 512), (512, 512), (1024, 128))  # an expert's units: (first channel of h, channels)
CTL_RANK = 64  # (the controllers' gate_rank: ``InklingConfig.gate_rank``)
PLANES = ("sink", "kd", "vd")  # the blocks of BF16 tensors read as two planes


def unit_gu_rows(E: int, F: int, h0: int, n: int, R: int) -> np.ndarray:
    """(E, rows) of ``ffn_gate_up_exps`` (expert e's rows e 2 F .., gate then up) of a unit: its outputs h0 .. h0 + n in
    groups of 64, each 8 x [8 gate | 8 up] rows in ``gu_tiles(R)`` tiles (-1: padding rows)."""
    g = n // 64
    out = np.full((E, g, gu_tiles(R) * R), -1, np.int64)
    first = 2 * F * np.arange(E)[:, None] + h0 + 64 * np.arange(g)  # (E, g): a group's first gate row
    out[:, :, :128] = (first[..., None, None, None] + 8 * np.arange(8)[:, None, None] + F * np.arange(2)[:, None]
                       + np.arange(8)).reshape(E, g, 128)
    return out.reshape(E, -1)


def unit_k(n: int) -> int:
    """The K a unit's down tiles take (its n columns padded to whole super-blocks)."""
    return -(-n // 256) * 256


class InklingImage(NormedImage):
    """The weight image of an Inkling 2 GGUF file (module docstring; layers: the physical layers to lay out, default
    all; vocab: LM rows, default the padded vocabulary)."""

    def __init__(self, gguf: GGUFFile, cfg: InklingConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None):
        cfg = cfg or InklingConfig.from_gguf(gguf)
        H = cfg.hidden_size
        if vocab is None:
            vocab = -(-cfg.vocab_padded // (COLS * C_LEN)) * COLS * C_LEN
        super().__init__(gguf, cfg, layers=layers, kc=KC, payload=MAX_K,
                         tile_bytes=FORMATS["BFP16"].tile_bytes(KC // 256), vocab=vocab)
        assert cfg.num_key_value_heads == COLS and H % (8 * COLS) == 0 and cfg.gate_rank == CTL_RANK, cfg
        assert sum(n for _, n in EXPERT_UNITS) == cfg.expert_ffn, cfg.expert_ffn
        self.h8 = H // COLS
        for L in self.layers:
            self._layer(L)
        self._ink(-1, "inkq", 1)
        for L in self.layers:
            self._experts(L)
        self.lm = lm = "output.weight"
        self._norm(-1, "lm")
        # (the Q8_0 head as its high plane, the logits that matter exactly on the host: ``decode.screen``)
        self.screen = screened(self.qtype(lm), self.formats(), opt_in=True)
        rows = column_rows(self.vocab)
        self._add(-1, "lm", [("lm", (lm,), np.where(rows < cfg.vocab_size, rows, -1),
                              *((plane(self.qtype(lm)),) if self.screen else ()))], H)
        self._place()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def loop_layer(self, L: int) -> bool:
        """Whether layer L is the loop core's (its visits of passes 2, 3 have controllers)."""
        return self.cfg.core_start <= L < self.cfg.core_end

    def queries(self, L: int) -> int:
        """The depth-mix queries of layer L's attention input (0: layer 0's, the normed embedding; L = -1: the output's)."""
        if L < 0:
            return 1
        return 0 if L == 0 else (1 if self.cfg.kinds[L] == "swa" else 3)

    def _ink(self, L: int, name: str, n: int) -> None:
        """A data region of n objects of the ink core."""
        self._add_data(L, name, n * self.tile_bytes)

    def _layer(self, L: int) -> None:
        cfg, b = self.cfg, f"blk.{L}."
        H, hd, swa = cfg.hidden_size, cfg.head_dim, cfg.kinds[L] == "swa"
        nq, rl = cfg.q_per_kv, cfg.rel_length
        cols = np.arange(COLS)[:, None]
        if L > 0:
            self._ink(L, "inkq", 1)
        loop = self.loop_layer(L)
        if loop:
            for s in "qkv":
                self._ctl_down(L, s)
            for s in "qkv":
                self._ink(L, f"ctl_{s}", 2 * self.ctl_objects())
        heads = (nq * cols + np.arange(nq)[None])  # (COLS, 3)
        q_rows = (hd * heads[..., None] + np.arange(hd)).reshape(COLS, -1)
        self._norm(L, "q")
        self._add(L, "q", [("q", (b + "attn_q.weight",), q_rows), ("gate", (b + "attn_gate.weight",), q_rows),
                           ("r", (b + "attn_rel.weight",), rl * nq * cols + np.arange(rl * nq)[None]),
                           ("sink", (self._f32(b + "attn_sink_proj.weight"),), f32_planes(heads, 2))], H)
        for s in "kv":
            parts = [(s, (b + f"attn_{s}.weight",), hd * cols + np.arange(hd)[None])]
            if not swa:  # (the taps' per-token deltas: conv_taps per KV head)
                taps = cfg.conv_taps
                parts.append((s + "d", (self._f32(b + f"attn_{s}_conv_proj.weight"),),
                              f32_planes(taps * cols + np.arange(taps)[None], 2)))
            self._norm(L, s)
            self._add(L, s, parts, H)
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], cfg.num_attention_heads * hd)
        self._ink(L, "inkf", 1)
        if loop:
            self._ctl_down(L, "ffn")
            self._ink(L, "ctl_ffn", 2 * self.ctl_objects())
        E = cfg.n_experts
        mine = E // COLS * cols + np.arange(E // COLS)[None]  # (the column's experts)
        self._norm(L, "router")
        self._add(L, "router", [("router", (b + "ffn_gate_inp.weight",), f32_planes(mine, 2))], H)
        self._ink(L, "inkc", 1)

    def _ctl_down(self, L: int, s: str) -> None:
        """A loop layer's controller down of stream s (module docstring): rows CTL_RANK / 8 c .. as planes."""
        r = CTL_RANK // COLS
        name = self.cfg.ctrl_tensor(L, s, "down")
        self._add(L, f"d{s}", [(f"d{s}", (name,), f32_planes(r * np.arange(COLS)[:, None] + np.arange(r)[None], 2))],
                  2 * self.cfg.hidden_size)

    def _experts(self, L: int) -> None:
        """Layer L's expert slabs (module docstring)."""
        cfg, b = self.cfg, f"blk.{L}."
        E, F, H = cfg.n_experts, cfg.expert_ffn, cfg.hidden_size
        gu, dn = b + "ffn_gate_up_exps.weight", b + "ffn_down_exps.weight"
        down = H * np.arange(E)[:, None] + np.arange(H)[None]  # (E, H): expert e's rows of the down
        for u, (h0, n) in enumerate(EXPERT_UNITS):
            self._add_slab(L, f"exp_gu{u}", [("gu", (gu,), unit_gu_rows(E, F, h0, n, self.R[self.fmt(gu)]))], H)
            K, R = unit_k(n), self.R[self.fmt(dn)]
            self._add_slab(L, f"exp_dn{u}", [("down", (dn,), group_rows(down, R, KC // K))], K, G=KC // K)
            r = self.slabs[-1]
            r.blocks[0] = replace(r.blocks[0], kslice=(h0, n))

    def _f32(self, name: str) -> str:
        """A BF16 tensor read as two planes: its F32 copy (exact; ``_own``)."""
        if name not in self._own:
            t = self.gguf.tensor(name)
            raw = np.ascontiguousarray(np.asarray(t.dequantize(), np.float32).reshape(t.rows, t.cols))
            self._own[name] = QTensor(name=name, qtype="F32", rows=t.rows, cols=t.cols, raw=raw)
        return name

    def norms(self, L: int, name: str) -> tuple[str, None]:
        """(w1, None): the RMSNorm of a normed phase's input (module docstring)."""
        if L < 0:
            return "output_norm.weight", None
        if name == "router":
            return f"blk.{L}.ffn_norm.weight", None
        if self.cfg.kinds[L] == "swa":
            return f"blk.{L}.attn_norm.weight", None
        return f"blk.{L}.attn_{name}_inp_norm.weight", None

    # ------------------------------------------------------------------
    # The ink core's objects
    # ------------------------------------------------------------------

    def ctl_objects(self) -> int:
        """Objects of a controller matrix's column rows (bf16 [288][64])."""
        return -(-self.h8 * CTL_RANK * 2 // self.tile_bytes)

    def _slices(self, v: np.ndarray, dtype: type) -> np.ndarray:
        """(COLS, n, h8): each column's slices of the vectors v (n, hidden)."""
        v = np.asarray(v, np.float32).reshape(-1, self.cfg.hidden_size)
        return v.reshape(v.shape[0], COLS, self.h8).transpose(1, 0, 2).astype(dtype)

    def ink_queries(self, L: int) -> np.ndarray:
        """(COLS, n, h8) fp32: the column slices of the queries of layer L's "inkq" (the attention mix's, then the
        messages'; L = -1: the output's)."""
        g, b = self.gguf, f"blk.{L}."
        if L < 0:
            return self._slices(g.tensor("output_res_query.weight").dequantize(), np.float32)
        qs = [np.asarray(g.tensor(b + "attn_res_query.weight").dequantize(), np.float32).reshape(-1, self.cfg.hidden_size)]
        if self.loop_layer(L):
            qs.append(self.cfg.msg_queries(g, L))
        return self._slices(np.concatenate(qs), np.float32)

    def ink_objects(self, L: int, name: str) -> np.ndarray:
        """(COLS, objects, tile_bytes) uint8: each column's data of an ink region (module docstring)."""
        g, b, T = self.gguf, f"blk.{L}.", self.tile_bytes
        if name == "inkq":
            parts = [self.ink_queries(L).reshape(COLS, -1).view(np.uint8)]
        elif name == "inkf":
            q = self._slices(g.tensor(b + "ffn_res_query.weight").dequantize(), np.float32)
            taps = self._slices(g.tensor(b + "attn_post_conv.weight").dequantize(), bfloat16)
            parts = [q.reshape(COLS, -1).view(np.uint8), taps.reshape(COLS, -1).view(np.uint8)]
        elif name == "inkc":
            parts = [self._slices(g.tensor(b + "ffn_post_conv.weight").dequantize(), bfloat16).reshape(COLS, -1)
                     .view(np.uint8)]
        else:  # (ctl_<stream>: the forget rows, then the gain rows)
            s = name.removeprefix("ctl_")
            n = self.ctl_objects()
            out = np.zeros((COLS, 2, n * T), np.uint8)
            for i, part in enumerate(("forget", "gain")):
                w = np.asarray(g.tensor(self.cfg.ctrl_tensor(L, s, part)).dequantize(), np.float32)
                w = w.reshape(self.cfg.hidden_size, CTL_RANK).astype(bfloat16).view(np.uint8)
                out[:, i, :self.h8 * CTL_RANK * 2] = w.reshape(COLS, -1)
            return out.reshape(COLS, -1, T)
        data = np.concatenate(parts, 1)
        out = np.zeros((COLS, -(-data.shape[1] // T) * T), np.uint8)
        out[:, :data.shape[1]] = data
        return out.reshape(COLS, -1, T)

    def data_objects(self, layer: int, name: str) -> np.ndarray:
        if name.startswith(("ink", "ctl_")):
            return self.ink_objects(layer, name)
        return super().data_objects(layer, name)

    def gu_rows(self) -> set[int]:
        """(the experts' gate / up: their units' rows per tile)"""
        return {b.R for r in self.slabs for b in r.blocks if r.name.startswith("exp_gu")}

    def unit_offsets(self, L: int) -> list[int]:
        """Byte offsets of an expert's units in its run of layer L (and the run's bytes last)."""
        rs = [r for r in self.slabs if r.layer == L]
        out = [0]
        for u in range(len(EXPERT_UNITS)):
            out.append(out[-1] + sum(r.col_bytes for r in rs if r.name in (f"exp_gu{u}", f"exp_dn{u}")))
        return out


__all__ = ["EXPERT_UNITS", "KC", "InklingImage", "unit_gu_rows", "unit_k"]
