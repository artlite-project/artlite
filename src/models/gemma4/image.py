"""The weight image of a Gemma 4 model (``decode.image.TileImage``): Q4_0 tiles of 32 rows and Q8_0 tiles of 16 rows
over K chunks of 512 columns (``KC``: E4B's K are 2560, 2048, 4096 and 10240), objects of 9216 bytes (``TILE_BYTES``:
one head_dim-512 K record of the KV cache, which streams through the same A objects). The decode's GEMV
(``decode.gemv``) and the prefill's GEMMs read the tiles in place.

Column-major: a column's regions in step order, each normed phase's norm-weight objects (data region "<name>_norm",
``norms``) before its tiles, so that a decode step streams a column's weights between two attentions as one run.

Per-column rows (layer L, column c, KV head h = c / 4, j = c % 4; hd: the layer's head_dim):

- "qkv": block "q" = the rows of query head c (hd); a layer with its own KV cache also block "kv" = the j-th quarter of
  the stack [K_h; V_h] (j 0, 1: K_h's halves; 2, 3: V_h's): the column's outputs land in the KV head's q objects and
  raw K / V objects (``gemma4.overlay``);
- "o" (``attn_output``), "down", "proj": rows hidden / 8 c ..; "gate_up": ``decode.image.gate_up_rows``; "gate"
  (``inp_gate``): rows per_layer / 8 c ..;
- "proj" reads its input (per_layer values) in column slots of 64: the gate phase's column c drains its 32 outputs and
  32 zeros to elements 64 c ..; its tiles hold ``proj``'s columns 32 c .. at 64 c .. and zero weights between (the
  re-laid-out copy ``_slotted``: GGUF blocks of 32 columns, a zero block after each);
- "lm": rows vocab / 8 c .. of the LM head (``token_embd``, tied);
- "ple" (layer -1, before layer 0: the step's first phase), E2B / E4B: the per-layer projection
  (``per_layer_model_proj``, BF16 in the files: a Q8_0 copy, ``_ple``), column c's rows the 32 c .. of each layer's
  per_layer_dim (the per-layer inputs its gate phases take), an even number of layers (a padding one: zero rows); the
  gate phases' first norm-weight object holds the column's 32 per_layer_proj_norm weights / sqrt(2) after [w1 | w_post]
  as bf16 hi [32] | lo [32] (``PLW``).
"""

import dataclasses
import math

import numpy as np
from ml_dtypes import bfloat16

from artlite import _native
from artlite.decode import gemv
from artlite.decode.design import ROT_GROUP as ROT
from artlite.decode.design import RPL_AT, RWP_AT, RWS_AT
from artlite.decode.image import NormedImage, column_rows, gate_up_rows, group_rows
from artlite.decode.protocol import MAX_K, NX_PER_NORM, POST_K
from artlite.decode.rotated import RotatedImage, column_groups
from artlite.decode.screen import plane, screened
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import GemmaConfig
from artlite.operators.design import COLS
from artlite.paro import ParoTensor
from artlite.prefill.gemm import pair_tails

KC = 512  # K chunk of a tile
KC_PARO = 1024  # (a PARO checkpoint's: P4_128 tiles of 16 rows; the hidden's inputs half regions)
TILE_BYTES = 9216
FORMATS = ("Q4_0", "Q8_0")  # (the decode GEMV's row-split tiles at KC)
PARO_FORMATS = ("P4_128", "Q4_0", "Q8_0")  # (the linears, the LM head's plane, the per-layer projection's copy)
PARO_TAILS = frozenset({"ple", "q", "k", "v", "gate", "up", "pgate", "lm"})  # (the regions of the hidden's input)
R_GROUPS = (3, 3, 3, 3, 2, 2, 2, 2)  # (PARO) the rotation groups of a column in an R phase (of the hidden's 20)
SLOT = 64  # the proj input's column slots
PLE = 32  # per-layer inputs per column (per_layer_dim / COLS)
PLW = 4 * NX_PER_NORM * POST_K  # (bytes) where a gate phase's first norm-weight object holds the column's PLE weights
PLE_Q8 = "per_layer_model_proj.weight/Q8_0"  # (the per-layer projection's copy: ``_ple``)


def kv_quarters(hd: int) -> np.ndarray:
    """(COLS, hd / 4) rows of K (or V: 2 KV heads, 2 hd rows) of the columns (PARO: k and v phases of their own): column
    4 h + j's are KV head h's quarter j."""
    c = np.arange(COLS)[:, None]
    return (c // 4) * hd + (c % 4) * (hd // 4) + np.arange(hd // 4)[None]


def kv_rows(hd: int) -> np.ndarray:
    """(COLS, hd / 2) rows of the stack [K; V] (2 KV heads: K rows 0 .. 2 hd - 1, V rows 2 hd ..) of the columns."""
    out = []
    for c in range(COLS):
        h, j = divmod(c, 4)
        base = (2 * hd if j >= 2 else 0) + h * hd + (j % 2) * hd // 2
        out.append(base + np.arange(hd // 2))
    return np.stack(out)


class GemmaImage(RotatedImage, NormedImage):
    """The weight image of a Gemma 4 GGUF file or PARO checkpoint (layers: the layers to lay out, default all; vocab: LM
    rows, default the vocabulary)."""

    screen = False  # (PARO: the tied F16 head as its Q4_0 plane, ``decode.screen``)

    def __init__(self, gguf: GGUFFile, cfg: GemmaConfig | None = None, *, layers: list[int] | None = None,
                 vocab: int | None = None, tails: bool = True):
        """tails: a PARO checkpoint's regions of the hidden's input as half regions (``PARO_TAILS``; False: padded to
        whole chunks, the tests' reference)."""
        cfg = cfg or GemmaConfig.from_gguf(gguf)
        H = cfg.hidden_size
        L0 = 0 if layers is None else layers[0]
        q0 = f"blk.{L0}.attn_q.weight"
        self.paro = q0 in gguf.tensor_info and isinstance(gguf.tensor(q0), ParoTensor)
        kc = KC_PARO if self.paro else KC
        self._rot_init()
        super().__init__(gguf, cfg, layers=layers, kc=kc, payload=MAX_K, tile_bytes=TILE_BYTES,
                         vocab=cfg.vocab_size if vocab is None else vocab, max_rows=gemv.row_splits(kc // 256),
                         tails=PARO_TAILS if self.paro and tails else frozenset())
        assert self.vocab % (COLS * 16) == 0 and H % (COLS * 64) == 0
        if cfg.per_layer_dim:
            assert cfg.per_layer_dim == COLS * PLE, "per-layer inputs: 32 per column"
            self._add(-1, "ple", [("ple", (self._ple(),), self.ple_rows())], H)
        for L in self.layers:
            if self.paro:
                self._paro_layer(L)
            else:
                self._layer(L)
        lm = "output.weight" if "output.weight" in gguf else "token_embd.weight"
        if self.paro:  # (the LM head's copy: the last layer's post-norm update, the output norm's weights, no rotation)
            self.screen = screened(self.qtype(lm), self.formats())
            self._r_data(-1, "rlm", "", self._vec("output_norm.weight"), self._post_of(self.layers[-1])[0])
            self._add(-1, "lm", [("lm", (lm,), column_rows(self.vocab), *((plane(self.qtype(lm)),) if self.screen else ()))],
                      H)
        else:
            self._norm(-1, "lm")
            self._add(-1, "lm", [("lm", (lm,), column_rows(self.vocab))], H)
        if self.paro and not self.formats() <= set(PARO_FORMATS):
            raise NotImplementedError(f"{', '.join(sorted(self.formats() - set(PARO_FORMATS)))} tiles (a PARO checkpoint's "
                                      f"Gemma decode reads {', '.join(PARO_FORMATS)} of {KC_PARO} columns)")
        self._place()

    def pairs(self, region: str, block: str, fmt: str, R: int) -> bool:
        """(PARO) The LM head's (the prefill reads none), the regions' the prefill reads in pairs (``pair_tails``)."""
        return region == "lm" or pair_tails(region, fmt, R)

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
        input (the slice's first layer), else a header (``decode.protocol.F_POST``)."""
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

    def norm_objects(self, L: int, name: str) -> np.ndarray:
        """(n_norm, tile_bytes) uint8: the phase's norm-weight objects (bf16); a gate phase's first object also holds the
        column's own per-layer projection norm weights (``PLW``): (COLS, n_norm, tile_bytes)."""
        out = super().norm_objects(L, name)
        if name != "gate":
            return out
        assert out.shape[1] >= PLW + 4 * PLE
        per = np.repeat(out[None], COLS, 0)  # (each column's own: its PLE weights)
        per[:, 0, PLW:PLW + 4 * PLE] = self._plw()
        return per

    def _plw(self) -> np.ndarray:
        """(COLS, 4 PLE) uint8: per column its PLE per-layer projection norm weights / sqrt(2) as bf16 hi [32] | lo [32]
        (``PLW``'s)."""
        w = (self._vec("per_layer_proj_norm.weight") / np.float32(math.sqrt(2.0))).astype(np.float32)
        hi = w.astype(bfloat16)
        lo = (w - hi.astype(np.float32)).astype(bfloat16)
        return np.stack([hi.reshape(COLS, PLE), lo.reshape(COLS, PLE)], 1).view(np.uint8).reshape(COLS, -1)

    def _ple(self) -> str:
        """The per-layer projection as a Q8_0 copy (``_native.quantize_q8_0``: ggml's rounding)."""
        t = self.gguf.tensor("per_layer_model_proj.weight")
        assert t.qtype in ("BF16", "F16"), t.qtype
        raw = np.empty((t.rows, t.cols // 32 * 34), np.uint8)
        src = t.raw if t.qtype == "BF16" else t.dequantize().astype(bfloat16).view(np.uint16)  # (a PARO checkpoint's F16:
        # its bf16 rounding first)
        _native.quantize_q8_0(np.ascontiguousarray(src), raw)
        self._own[PLE_Q8] = QTensor(name=PLE_Q8, qtype="Q8_0", rows=t.rows, cols=t.cols, raw=raw)
        return PLE_Q8

    def pack(self, out: np.ndarray, *, threads: int = 0) -> bool:
        """(once: the per-layer projection's copy is dropped after it)"""
        dirty = super().pack(out, threads=threads)
        if PLE_Q8 in self._own:
            t = self._own[PLE_Q8]
            assert t.raw.size, "the image is packed once"
            self._own[PLE_Q8] = dataclasses.replace(t, raw=t.raw[:0])
        return dirty

    def ple_rows(self) -> np.ndarray:
        """(COLS, PLE n) rows of the per-layer projection: column c's the PLE c .. of each of the n layers (an even
        number: a padding layer of -1 after an odd count)."""
        d = self.cfg.per_layer_dim
        rows = [L * d + PLE * np.arange(COLS)[:, None] + np.arange(PLE)[None] for L in self.layers]
        if len(rows) % 2:
            rows.append(np.full((COLS, PLE), -1, np.int64))
        return np.concatenate(rows, 1)

    def fmt(self, name: str) -> str:
        f = super().fmt(name)
        if f not in FORMATS and not self.paro:  # (a PARO image's: ``formats``, its head's plane not its own)
            raise NotImplementedError(f"{name}: {f} tiles (the Gemma decode reads {', '.join(FORMATS)} of {KC} columns)")
        return f

    def data_objects(self, layer: int, name: str) -> np.ndarray:
        if self.rot_region(layer, name):
            return self._rot_objs[(layer, name)]
        return super().data_objects(layer, name)

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

    # ------------------------------------------------------------------
    # PARO
    # ------------------------------------------------------------------

    @staticmethod
    def r_groups(c: int) -> tuple[int, int]:
        """(PARO) Column c's rotation groups of an R phase: (first, count), contiguous (``R_GROUPS``)."""
        return sum(R_GROUPS[:c]), R_GROUPS[c]

    def _r_data(self, L: int, name: str, tensor: str, gamma: np.ndarray, post: str | None) -> None:
        """(PARO) The data region of an R phase (``gemma4.paro_steps``): per column the rotation objects of its groups
        (``r_groups``; a column of fewer: a padding group, its object unread) of ``tensor``'s rotation ("": the identity)
        with gamma folded in, two a 9216-byte object; each object w_post (``post``: None, zeros) of its two groups at
        ``RWP_AT``, the column's first w_post of its residual slice at ``RWS_AT`` and its per-layer projection norm's
        weights at ``RPL_AT`` (bf16, ``_plw``)."""
        H, h8 = self.cfg.hidden_size, self.cfg.hidden_size // COLS
        assert H == ROT * sum(R_GROUPS), (H, R_GROUPS)
        n = max(R_GROUPS)
        groups = []
        for c in range(COLS):
            g0, ng = self.r_groups(c)
            groups.append(np.array([g0 + i if i < ng else -1 for i in range(n)]))
        self.rot_data(L, name, [(tensor, gamma, groups)])
        wp = (self._vec(post) if post else np.zeros(H, np.float32)).astype(bfloat16).view(np.uint8)
        objs = self._rot_objs[(L, name)]  # (COLS, objects, tile_bytes)
        assert RWS_AT + 2 * h8 <= RPL_AT, h8
        pl = self._plw() if self.cfg.per_layer_dim else None
        for c in range(COLS):
            g0, ng = self.r_groups(c)
            for k in range(-(-ng // 2)):
                a, b = 2 * ROT * (g0 + 2 * k), 2 * ROT * min(g0 + 2 * k + 2, g0 + ng)
                objs[c, k, RWP_AT:RWP_AT + b - a] = wp[a:b]
            objs[c, 0, RWS_AT:RWS_AT + 2 * h8] = wp[2 * h8 * c:2 * h8 * (c + 1)]
            if pl is not None:  # (the pgate phase's per-layer inputs, ``decode.design``'s wpupd)
                objs[c, 0, RPL_AT:RPL_AT + 4 * PLE] = pl[c]

    def r_objects(self, c: int) -> int:
        """(PARO) The rotation A objects of column c in an R phase (``_r_data``): its groups'."""
        return -(-R_GROUPS[c] // 2)

    def _paro_layer(self, L: int) -> None:
        """A PARO checkpoint's layer (``gemma4.paro_steps``: the step's phases): per layer the R phases' data and the
        linears' regions in step order."""
        cfg, b = self.cfg, f"blk.{L}."
        H, I, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim(L)
        own = cfg.kv_source(L) == L
        i = self.layers.index(L)
        post = None if i == 0 else self._post_of(self.layers[i - 1])[0]  # (layer 0: the embeddings, no update)
        g = self._vec(b + "attn_norm.weight")
        for n in ("q", "k", "v") if own else ("q",):
            self._r_data(L, "r" + n, f"{b}attn_{n}.weight", g, post)
        self._add(L, "q", [("q", (b + "attn_q.weight",), hd * np.arange(COLS)[:, None] + np.arange(hd)[None])], H)
        if own:
            self._add(L, "k", [("k", (b + "attn_k.weight",), kv_quarters(hd))], H)
            self._add(L, "v", [("v", (b + "attn_v.weight",), kv_quarters(hd))], H)
        self.rot_data(L, "rot_o", [(b + "attn_output.weight", None, column_groups(0, hd))])
        self._add(L, "o", [("o", (b + "attn_output.weight",), column_rows(H))], COLS * hd)
        g = self._vec(b + "ffn_norm.weight")
        for n in ("gate", "up"):
            self._r_data(L, "r" + n, f"{b}ffn_{n}.weight", g, b + "post_attention_norm.weight")
        self._add(L, "gate", [("gate", (b + "ffn_gate.weight",), column_rows(I))], H)
        self._add(L, "up", [("up", (b + "ffn_up.weight",), column_rows(I))], H)
        self.rot_data(L, "rot_down", [(b + "ffn_down.weight", None, column_groups(0, I // COLS))])
        self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
        if cfg.per_layer_dim:  # (the per-layer input gate on the residual itself: no norm weight, no 1 / rms)
            self._r_data(L, "rpgate", b + "inp_gate.weight", None, b + "post_ffw_norm.weight")
            self._add(L, "pgate", [("pgate", (b + "inp_gate.weight",), column_rows(cfg.per_layer_dim))], H)
            assert cfg.per_layer_dim == 2 * ROT, "proj's input: two rotation groups"
            self.rot_data(L, "rot_proj", [(b + "proj.weight", None, [np.array([c % 2]) for c in range(COLS)])])
            self._add(L, "proj", [("proj", (b + "proj.weight",), group_rows(column_rows(H), 16, self.kc // ROT // 2))],
                      cfg.per_layer_dim, G=self.kc // ROT // 2)  # (K 256: row groups of 4 a tile row)

