"""The multi-token-prediction drafter of a Gemma 4 model (a ``gemma4-assistant`` GGUF file; llama.cpp's
``src/models/gemma4-assistant.cpp``) on the model's decode overlay (``gemma4/decode.py``): a draft step is one dispatch
of the same design, with the drafter's weights in a buffer of their own and its layers attending to the model's caches.

A draft step predicts the token after a token t at position p from t's embedding (the model's, scaled) and a hidden
state h (the model's output-normed hidden state of position p - 1, or the one the previous draft step left): pre =
W_pre [x | h] (the drafter's width, 256), its layers (q only: they attend to the model's caches of its last
sliding-window and its last full layer at the positions before p - p itself is not cached -, o, gelu(gate) up, down,
the model's post-norm residual updates), the output norm, the LM head over the vocabulary and the post-projection (the
next step's h, back to the model's width). Every step of a draft runs at the same position p.

On the overlay (the model's phases: ``gemma4/overlay.py``):

- the residual: 32 values per column (``decode.design.M_H8``); a narrow emit's C object holds them and 32 zeros, drained
  into the slot-1 positions next to them (which must be zero);
- pre (K 5120: [x | h], one row tile per column) adds into the residual (F_RESID, after F_LOADR objects of zeros); o
  and down (K 1024 / 2048, 32 rows per column) are F_VOUT phases as in the model; q, gate_up and the LM head are
  headers (q of layer 0 one whose update adds nothing, as the model's ``step(start=)``);
- the matrices of K = 256 (q, gate / up, the LM head and the post-projection) as row pairs (``F_KPAIR``): a tile row of
  512 columns holds two rows of the matrix (super-block 0: row a, 1: row b), and the header runs as two slots of the
  token, slot 0 = [x | 0] (row a's product), slot 1 = [0 | x] (row b's): no padded weight bytes (the LM head is 37.7
  MB). ``pair_rows`` chooses a, b so that the C objects ([slot 0's 64 | slot 1's 64] per 128 outputs) come in the
  outputs' order; the LM head pairs consecutive rows instead (a view of the file's matrix: ``logit_rows``);
- the attention: per KV head its 4 columns over the cache (one chain) with both of its query heads (``attn.M_ANH``),
  the masks of position p - 1 (the window W - 1), no K / V insert;
- per step the host writes x (and h for a draft's first step: the post-projection's drain writes the next one) and,
  at a new position, the token objects and the cache fills (``_position``).
"""

from dataclasses import dataclass

import numpy as np
import torch

from artlite.decode import gemv, kv
from artlite.decode import phases as mp
from artlite.decode.design import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_KPAIR,
    F_NORM,
    F_POST,
    F_PZERO,
    F_RESID,
    F_TOK,
    F_VOUT,
    M_FLAGS,
    M_NASKIP,
    M_NS,
    M_SCALE,
    M_SKIP,
    M_TOK_KIND,
    M_TOK_NG,
    MAX_K,
    META,
    NX_PER_NORM,
    POST_K,
    XB,
    XPAY,
    hidden_meta,
    input_fill,
    post_input_fill,
    post_region_bytes,
    region_bytes,
)
from artlite.decode.image import TileImage, column_rows
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import GemmaConfig
from artlite.operators.design import COLS
from artlite.operators.util import bf16_bits, rup
from artlite.overlay import Sequence, Transaction

from ..common import argmax
from . import attn as AT
from .image import FORMATS, KC, TILE_BYTES
from .overlay import GemmaOverlay

PAIR = 2 * C_LEN  # outputs of a K-pair phase per pair of C objects (slot 0's C_LEN, then slot 1's)
FLAGS = {"pre": F_RESID | F_EMITR, "q": F_POST | F_NORM | F_KPAIR, "o": F_VOUT | F_PZERO | F_EMITR,
         "gate_up": F_POST | F_NORM | F_GU | F_KPAIR, "down": F_VOUT | F_PZERO | F_EMITR, "lm": F_POST | F_NORM | F_KPAIR}
LAYER = ("q", "att", "o", "gate_up", "down")


def pair_rows(n: int, R: int, gu: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The virtual rows (tile rows, R per tile) of a K-pair block with n outputs per column (a multiple of PAIR): (a, b,
    up) - the output of super-block 0 (slot 0) and of super-block 1 (slot 1), and (gu) whether the row is an up row.
    The C objects of a tile group (the tiles of C_LEN outputs per slot) are slot 0's outputs PAIR g .., then slot 1's
    PAIR g + C_LEN ..; gu (F_GU): 16 tile rows [8 gate | 8 up] give 8 outputs per slot, n tile rows per column."""
    v = np.arange(n if gu else n // 2)
    per = R // 2 if gu else R  # (outputs of a tile per slot)
    t, r = v // R, v % R
    g, h = t // (C_LEN // per), t % (C_LEN // per)
    a = PAIR * g + per * h + (8 * (r // 16) + r % 8 if gu else r)
    return a, a + C_LEN, (r % 16 >= 8) & gu


class DraftImage(TileImage):
    """The drafter's weights (``gguf``: the gemma4-assistant file of the model ``cfg``), column-major in step order:
    pre, then per layer q (its norm-weight objects first), o, gate_up (norm-weight objects first), down, then the LM
    head (norm-weight objects first): the post-projection's rows (padded to PAIR per column), then the vocabulary's."""

    def __init__(self, gguf: GGUFFile, cfg: GemmaConfig):
        if gguf.arch != "gemma4-assistant":
            raise ValueError(f"{gguf.path}: architecture {gguf.arch!r}, expected gemma4-assistant")
        m = gguf.arch_meta
        self.cfg, H = cfg, cfg.hidden_size
        self.d = d = int(m("embedding_length"))
        self.heads, self.ffn = int(m("attention.head_count")), int(m("feed_forward_length"))
        self.swa = [bool(x) for x in m("attention.sliding_window_pattern")]
        self.hd = [int(m("attention.key_length_swa")) if s else int(m("attention.key_length")) for s in self.swa]
        self.eps = float(m("attention.layer_norm_rms_epsilon"))
        dims = {True: cfg.head_dim_swa, False: cfg.head_dim_full}
        if (int(m("embedding_length_out")) != H or self.hd != [dims[s] for s in self.swa] or d // COLS != 32
                or self.heads % AT.KV_HEADS or self.heads // AT.KV_HEADS > AT.QH or abs(self.eps - cfg.rms_norm_eps) > 1e-12
                or (float(m("rope.freq_base")), float(m("rope.freq_base_swa", 1e4))) != (cfg.rope_base_full, cfg.rope_base_swa)
                or self.ffn % (COLS * PAIR)):
            raise NotImplementedError(f"{gguf.path}: a drafter of other shapes than the overlay's (width {d}, heads "
                                      f"{self.heads}, head dims {self.hd}, ffn {self.ffn})")
        assert gguf.tensor_info["token_embd.weight"][0][1] == cfg.vocab_size, "the drafter's vocabulary"
        super().__init__(gguf, kc=KC, payload=MAX_K, tile_bytes=TILE_BYTES, vocab=cfg.vocab_size,
                         max_rows=gemv.row_splits(KC // 256), column_major=True)
        self._add(-1, "pre", [("pre", ("nextn.pre_projection.weight",), column_rows(d))], 2 * H)
        for il in range(len(self.swa)):
            b, nq = f"blk.{il}.", self.heads * self.hd[il]
            self._norm(il, "q")
            self._add(il, "q", [self._pairs("q", (b + "attn_q.weight",), nq)], KC)
            self._add(il, "o", [("o", (b + "attn_output.weight",), column_rows(d))], nq)
            self._norm(il, "gate_up")
            self._add(il, "gate_up", [self._pairs("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), self.ffn)], KC)
            self._add(il, "down", [("down", (b + "ffn_down.weight",), column_rows(d))], self.ffn)
        self._norm(-1, "lm")
        self._add(-1, "lm", [self._pairs("post", ("nextn.post_projection.weight",), H), self._lm_pairs()], KC)
        self._place()

    def fmt(self, name: str) -> str:
        f = super().fmt(name)
        if f not in FORMATS:
            raise NotImplementedError(f"{name}: {f} tiles (the drafter reads {', '.join(FORMATS)})")
        return f

    def _pairs(self, block: str, names: tuple[str, ...], n: int) -> tuple[str, tuple[str, ...], np.ndarray]:
        """A K-pair block of the matrix ``names`` (one, or gate and up: their stack) with n outputs (n / 8 per column,
        padded to PAIR with zero rows; ``pair_rows``), as a re-laid-out copy (``_own``): a virtual row = the GGUF
        blocks of rows a and b."""
        ts = [self.gguf.tensor(t) for t in names]
        assert all(t.cols == POST_K and t.qtype == ts[0].qtype and t.rows == n for t in ts), names
        R, gu = self.R[self.fmt(names[0])], len(names) == 2
        per = n // COLS
        a, b, up = pair_rows(-(-per // PAIR) * PAIR, R, gu)
        rows = np.stack([np.concatenate([np.where(o < per, per * c + o + n * up, -1) for c in range(COLS)])
                         for o in (a, b)], 1)  # [column's virtual row][super-block]: a row of the stack, -1: zeros
        raw = np.concatenate([t.raw for t in ts]) if gu else ts[0].raw
        out = np.zeros((rows.shape[0], 2, raw.shape[1]), np.uint8)
        for k in range(2):
            out[rows[:, k] >= 0, k] = raw[rows[rows[:, k] >= 0, k]]
        self._own[names[0]] = QTensor(name=names[0], qtype=ts[0].qtype, rows=rows.shape[0], cols=2 * POST_K,
                                      raw=out.reshape(rows.shape[0], -1))
        return block, (names[0],), column_rows(rows.shape[0])

    def _lm_pairs(self) -> tuple[str, tuple[str, ...], np.ndarray]:
        """The LM head (token_embd) as row pairs (2 v, 2 v + 1): a view of the file's matrix (``logit_rows``)."""
        t = self.gguf.tensor("token_embd.weight")
        assert t.cols == POST_K and t.rows % (COLS * PAIR) == 0
        self._own[t.name] = QTensor(name=t.name, qtype=t.qtype, rows=t.rows // 2, cols=2 * POST_K,
                                    raw=t.raw.reshape(t.rows // 2, -1))
        return "lm", (t.name,), column_rows(t.rows // 2)

    def norms(self, L: int, name: str) -> tuple[str | None, str | None, str | None]:
        """(w1, w_post, scale) tensor names of a header (None: ones / no scale): q of layer 0 adds nothing."""
        b, prev = f"blk.{L}.", f"blk.{L - 1 if L >= 0 else len(self.swa) - 1}."
        if name == "q":
            return (b + "attn_norm.weight", *((prev + "post_ffw_norm.weight", prev + "layer_output_scale.weight")
                                              if L > 0 else (None, None)))
        if name == "gate_up":
            return b + "ffn_norm.weight", b + "post_attention_norm.weight", None
        assert name == "lm", name
        return "output_norm.weight", prev + "post_ffw_norm.weight", prev + "layer_output_scale.weight"

    def _norm(self, L: int, name: str) -> None:
        self._add_data(L, name + "_norm", self.tile_bytes)

    def _vec(self, name: str | None) -> np.ndarray:
        if name is None:
            return np.ones(self.d, np.float32)
        return np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)

    def norm_objects(self, L: int, name: str) -> np.ndarray:
        """(tile_bytes,) uint8: a header's norm-weight object [w1 | w_post] (NX_PER_NORM POST_K bf16 each), each as its
        weights for both strips of the K pair."""
        w1, wp, _ = self.norms(L, name)
        parts = np.zeros((2, NX_PER_NORM * POST_K), np.float32)
        for i, w in enumerate((w1, wp)):
            parts[i, :2 * self.d] = np.tile(self._vec(w), 2)
        b = torch.from_numpy(parts.reshape(-1)).bfloat16().view(torch.int16).numpy().view(np.uint8)
        out = np.zeros(self.tile_bytes, np.uint8)
        out[:b.size] = b
        return out

    def logit_rows(self) -> np.ndarray:
        """(vocab,) the vocabulary row of each LM output in a step's logits: per column (vocab / 8 outputs) and group
        of PAIR outputs the even rows of 64 pairs, then their odd rows."""
        f = np.arange(self.vocab)
        per = self.vocab // COLS
        c, rem = f // per, f % per
        v = per // 2 * c + C_LEN * (rem // PAIR) + rem % C_LEN  # (the virtual row)
        return 2 * v + rem // C_LEN % 2


@dataclass
class _Phase:
    """A draft phase's input region in DACT."""

    name: str
    layer: int
    off: int
    K: int
    S: int  # its slots (2: a K pair)


class GemmaDraft:
    """The drafter ``gguf`` of the model on ``overlay`` (``load`` after the overlay's: its device and caches)."""

    def __init__(self, overlay: GemmaOverlay, gguf: GGUFFile):
        self.ov, self.gguf, self.cfg = overlay, gguf, overlay.cfg
        assert overlay.layers == list(range(self.cfg.num_hidden_layers)), "a drafter needs the model's caches"
        self.image = img = DraftImage(gguf, self.cfg)
        fmts = {b.fmt for r in img.regions for b in r.blocks}
        assert fmts <= set(overlay.op.rows), f"the design's tile formats {set(overlay.op.rows)} lack {fmts}"
        self.d, self.vocab, self.T, self.H = img.d, img.vocab, TILE_BYTES, self.cfg.hidden_size
        self.src = [self.cfg.kv_layers - (2 if s else 1) for s in img.swa]  # (the model's caches the layers attend to)
        self.nq = img.heads // AT.KV_HEADS
        self.rows = img.logit_rows().astype(np.int32)
        self._pos = -1
        self._tr: Transaction | None = None
        self._fills: list[tuple[int, int, int]] = []  # (BD word index, layer, column) of the cache fills
        self._tails: list[tuple[int, int, int]] = []  # (the same) of the chain ends' tail fills
        self._tkey: tuple | None = None  # (the position and record modes the tail fills were patched for)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self) -> list[tuple[int, str]]:
        return [(-1, "pre"), *[(il, n) for il in range(len(self.image.swa)) for n in LAYER], (-1, "lm")]

    def _plan(self) -> None:
        """Byte offsets in DACT; the headers' chain (whose v and r each phase's outputs are)."""
        img = self.image
        n = 0

        def take(nbytes: int) -> int:
            nonlocal n
            n += rup(nbytes)
            return n - rup(nbytes)

        self.tok = take(2 * XB)
        self.loadr = take(COLS * 2 * XB)
        self.inp: dict[tuple[int, str], _Phase] = {}
        self.att: dict[tuple[int, int], int] = {}
        for L, name in self.phases():
            if name == "att":
                for h in range(AT.KV_HEADS):  # (its q object, whose metadata words are the F_ATT object)
                    self.att[(L, h)] = take(XB)
                continue
            K = img.region(L, name).K
            kp = bool(FLAGS[name] & F_KPAIR)
            self.inp[(L, name)] = _Phase(name, L, take(post_region_bytes(K) if kp else region_bytes(K)), K, 2 if kp else 1)
        self.nbytes = n
        seq = [x for x in self.phases() if x[1] != "att"]
        heads = [x for x in seq if FLAGS[x[1]] & F_POST]
        self.r_to = {x: heads[heads.index(x) + 1] if x != heads[-1] else None for x in heads}  # (a header's update)
        self.r_to[(-1, "pre")] = heads[0]
        self.v_to = {x: next(y for y in seq[i + 1:] if FLAGS[y[1]] & F_POST)
                     for i, x in enumerate(seq) if FLAGS[x[1]] & F_VOUT}

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, *, threads: int = 0) -> None:
        """Pack the drafter's weights into a buffer of the overlay's device, write the static objects, build the
        transaction."""
        dev = self.ov.dev
        assert dev is not None, "load the overlay first"
        if self.image.pack(dev.alloc("DW", self.image.nbytes), threads=threads, release=False):
            dev.flush("DW")
        self.image._own.clear()  # (the re-laid-out copies)
        dev.alloc("DACT", self.nbytes)
        dev.alloc("DLOGITS", 2 * self.vocab)
        act = dev.view("DACT")
        for kind in (0, 1):  # (the first sub-step's token of each layer kind)
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_TOK_KIND], m[M_NS] = F_TOK, AT.SMAX * kind, 1
            act[self.tok + kind * XB:][:4 * META] = m.view(np.uint8)
        for c in range(COLS):
            for s in range(2):
                act[self.loadr + (2 * c + s) * XB:][:4 * META] = mp.loadr_meta(s).view(np.uint8)
        for (L, name), ph in self.inp.items():
            (gr,) = mp.groups(self.image.region(L, name), FLAGS[name] | (F_PZERO if (L, name) == (0, "q") else 0), ph.S)
            m = gr.meta
            hidden_meta(m, self.d)
            if FLAGS[name] & F_POST:
                sc = self.image.norms(L, name)[2]
                m[M_SCALE] = bf16_bits(self._scalar(sc) if sc else 1.0)
            act[ph.off:ph.off + 4 * META] = m.view(np.uint8)
        for (L, h), off in self.att.items():
            swa = self.image.swa[L]
            hi, lo = AT.score_constant(self._scalar(f"blk.{L}.attn_q_norm.weight", self.gguf),
                                       self.ov._scalar(f"blk.{self.src[L]}.attn_k_norm.weight"))
            m = np.zeros(META, np.int32)
            m[M_FLAGS], m[M_NASKIP], m[M_SKIP] = AT.F_ATT, -1 - AT.SMAX * int(not swa), 0
            m[AT.M_AHD], m[AT.M_AKIND], m[AT.M_AOWN] = self.image.hd[L], int(not swa), 0
            m[AT.M_QC_HI], m[AT.M_QC_LO], m[AT.M_ATOK] = hi, lo, AT.SMAX * int(not swa)
            m[AT.M_AWIN], m[AT.M_ANH] = self.cfg.sliding_window - 1 if swa else 0, self.nq
            act[off:off + 4 * META] = m.view(np.uint8)
        dev.flush("DACT")
        self._tr = self._build()

    def _scalar(self, name: str, gguf: GGUFFile | None = None) -> float:
        w = np.asarray((gguf or self.gguf).tensor(name).dequantize(), np.float32).reshape(-1)
        if float(w.min()) != float(w.max()):
            raise NotImplementedError(f"{name}: a weight per dimension (the overlay takes one number)")
        return float(w[0])

    # ------------------------------------------------------------------
    # The transaction of a draft step
    # ------------------------------------------------------------------

    def _narrow(self, seq: Sequence, c: int, off: int, part: int, repeat: int = 1) -> None:
        """Column c's C object of 32 values and 32 zeros -> the v (part 0) or r (1) slots of its elements of a header's
        input at DACT offset off (slot 0's, then slot 1's: zeros); ``repeat`` 2: a K pair's second object (zeros) into
        the strip-1 area (unread: the helper mirrors strip 0)."""
        a = self.ov.dev.address("DACT", off + XB + XPAY + 8 * 32 * c + 32 * part)
        seq.drain("C", c, a, 2 * C_LEN, (4, 4, 16, 4), repeat, (512, 2) if repeat == 2 else None)

    def _outputs(self, seq: Sequence, phase: tuple[int, str]) -> None:
        """A phase's C drains, in its emit order."""
        L, name = phase
        dev, H, img = self.ov.dev, self.H, self.image
        nxt = self.r_to.get(phase)
        for c in range(COLS):
            if name == "pre":
                self._narrow(seq, c, self.inp[nxt].off, 1)
                continue
            if FLAGS[name] & F_POST:  # (the updated residual: into the next header's r slots)
                self._narrow(seq, c, self.inp[nxt].off if nxt else self.inp[phase].off + 2048, 1, 2)
            if name == "q":
                n = img.heads * img.hd[L] // COLS
                h, j = c // AT.QH, c % AT.QH
                seq.drain("C", c, dev.address("DACT", self.att[(L, h)] + 4 * META + 2 * n * j), 2 * n)
            elif FLAGS[name] & F_VOUT:
                tgt = self.inp[self.v_to[phase]].off
                self._narrow(seq, c, tgt, 0)
                seq.drain("C", c, dev.address("DACT", tgt + XB + 2 * C_LEN * c), 2 * C_LEN)  # (its partial sums)
            elif name == "gate_up":
                per = img.ffn // COLS
                seq.drain("C", c, dev.address("DACT", self.inp[(L, "down")].off + XB + 2 * per * c), 2 * per)
            elif name == "lm":
                pre, per = self.inp[(-1, "pre")].off, H // COLS
                npad = -(-per // PAIR) * PAIR
                seq.drain("C", c, dev.address("DACT", pre + XB + 2 * (H + per * c)), 2 * per)  # (h of the next step)
                seq.drain("C", c, dev.address("DACT", pre + XB + 4 * H + 2 * (npad - per) * c), 2 * (npad - per))
                nv = self.vocab // COLS
                seq.drain("C", c, dev.address("DLOGITS", 2 * nv * c), 2 * nv)

    def _attention(self, seq: Sequence, L: int, weights: list[tuple[int, str]]) -> None:
        """Layer L's attention: per column its range of the model's cache (A, patched per position), the KV head's
        region (X); o from the chain's end (D)."""
        dev, hd = self.ov.dev, self.image.hd[L]
        for c in range(COLS):
            at = seq.fill("A", c, dev.address("KV", self.ov.kv_off[self.src[L]]), self.T)
            self._fills.append((at, L, c))
            if c % AT.QH == AT.QH - 1:  # (the chain's end: the tail objects, one fill each)
                self._tails += [(seq.fill("A", c, dev.address("KV", self.ov.ring_off[self.src[L]]), self.T), L, c)
                                for _ in range(AT.tails(1))]
        if weights:
            self.ov._weight_run(seq, weights, image=self.image, buf="DW")
        seq.sync()  # (the q objects in DDR)
        o = self.inp[(L, "o")]
        for c in range(COLS):
            h = c // AT.QH
            seq.fill("X", c, dev.address("DACT", self.att[(L, h)]), XB)
            if c % AT.QH == AT.QH - 1:
                seq.drain("D", c, dev.address("DACT", o.off + XB + 2 * hd * self.nq * h), 2 * hd * self.nq)

    def _build(self) -> Transaction:
        dev = self.ov.dev
        seq = Sequence(self.ov.op, rolling=True, inputs=("X",))
        for c in range(COLS):
            seq.fill("X", c, dev.address("DACT", self.tok), 2 * XB)
        for c in range(COLS):
            seq.fill("X", c, dev.address("DACT", self.loadr + 2 * c * XB), 2 * XB)
        ph = self.phases()

        def run(i: int) -> list[tuple[int, str]]:  # (the GEMV phases from i to the next attention: one A fill)
            return ph[i:next((k for k in range(i, len(ph)) if ph[k][1] == "att"), len(ph))]

        for i, (L, name) in enumerate(ph):
            if name == "att":
                self._attention(seq, L, run(i + 1))
                continue
            if i == 0:
                self.ov._weight_run(seq, run(0), image=self.image, buf="DW")
            p = self.inp[(L, name)]
            seq.sync()  # (its inputs' producers)
            nbytes, dims = post_input_fill(p.K, p.S) if p.S == 2 else input_fill(p.K, p.S)
            for c in range(COLS):
                seq.fill("X", c, dev.address("DACT", p.off), nbytes, dims)
            self._outputs(seq, (L, name))
        seq.sync(fills=True)
        return Transaction(dev, seq.words())

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _position(self, pos: int) -> None:
        """The token objects (the masks at pos - 1, RoPE at pos, the columns' ranges) and the cache fills."""
        ov, cfg, dev, T = self.ov, self.cfg, self.ov.dev, self.T
        act = dev.view("DACT")
        fills = {}
        for kind, swa in ((0, True), (1, False)):
            pg = ov.at.pg[256 if swa else 512]
            r = ov._ranges(swa, pos - 1)
            m = act[self.tok + kind * XB:][:4 * META].view(np.int32)
            m[AT.M_POS] = pos - 1
            for c in range(COLS):
                g0, n = r[c % AT.QH]
                no = kv.fill(g0, n, ov.at.kp)[1] + (AT.tails(1) if c % AT.QH == AT.QH - 1 else 0)
                m[AT.M_TOK_G0 + c], m[AT.M_TOK_NA + c], m[M_TOK_NG + c] = g0 * pg, (ov.at.kp + 1) * n | no << 16, no
                fills[(kind, c)] = kv.fill(g0, n, ov.at.kp)
            cs = AT.rope_tables(cfg, [pos], ov.at.rot, swa=swa)
            act[self.tok + kind * XB + 4 * META:][:cs.nbytes] = cs.view(np.uint8).reshape(-1)
        dev.flush("DACT", self.tok, 2 * XB)
        kv0 = dev.address("KV")
        f = np.array([(at, *fills[(0 if self.image.swa[L] else 1, c)], ov.kv_base(self.src[L], c // AT.QH))
                      for at, L, c in self._fills], np.int64)
        ov._bd_words(self._tr.words, f[:, 0], kv0 + f[:, 3] + f[:, 1] * T, f[:, 2] * T)
        self._pos = pos

    def _tail_fills(self, pos: int) -> bool:
        """The chain ends' tail fills of the open record of position pos - 1 (``GemmaOverlay.tail_mode``: its ring
        slots, or its V panels as the prefill left them) and the F_ATT objects' M_ATREC; False: as they were."""
        ov, T, dev = self.ov, self.T, self.ov.dev
        rec = {src: ov.tail_mode(src, before=False) for src in set(self.src)}
        key = (pos, tuple(sorted(rec.items())))
        if key == self._tkey:
            return False
        for src, m in rec.items():
            if not m:
                ov.vring.check(src, pos - 1, pos)
        kv0, act = dev.address("KV"), dev.view("DACT")
        for at, L, c in self._tails:  # (the record of pos - 1)
            src, h = self.src[L], c // AT.QH
            a = ov.kv_base(src, h) + ov._vrec(pos - 1, bool(self.image.swa[L])) if rec[src] else \
                ov.ring(src, h) + ((pos - 1) >> 3 & 1) * T
            ov._bd_words(self._tr.words, np.array([at]), np.array([kv0 + a]), np.array([T]))
        for (L, h), off in self.att.items():
            act[off:off + 4 * META].view(np.int32)[AT.M_ATREC] = int(rec[self.src[L]])
            dev.flush("DACT", off, 4 * META)
        self._tkey = key
        return True

    def step(self, x: torch.Tensor, h: torch.Tensor | None, pos: int) -> torch.Tensor:
        """A draft step for the token at position ``pos`` (not cached; the positions before are): x (hidden,) its
        scaled embedding, h (hidden,) the model's output-normed hidden state of position pos - 1, or None: the hidden
        state the previous step left (the steps of a draft run at the position of its first token). Returns the raw
        logits of the next token in the LM's output order (vocab,) bf16 (a view of the NPU's buffer; ``token``)."""
        assert 1 <= pos <= self.ov.max_positions, pos
        dev = self.ov.dev
        act = dev.view("DACT")
        pre = self.inp[(-1, "pre")].off + XB
        xh = torch.cat([x.reshape(-1), h.reshape(-1)]) if h is not None else x.reshape(-1)
        act[pre:pre + 2 * xh.numel()] = xh.bfloat16().view(torch.int16).numpy().view(np.uint8)
        dev.flush("DACT", pre, 2 * xh.numel())
        moved = pos != self._pos
        if moved:
            self._position(pos)
        if self._tail_fills(pos) or moved:
            self._tr.flush()
        self.ov.ready()
        dev.dispatch(self._tr)
        dev.fetch("DLOGITS")
        return torch.from_numpy(dev.view("DLOGITS")).view(torch.bfloat16)

    def token(self, logits: torch.Tensor) -> int:
        """The most likely token of a step's logits."""
        return int(self.rows[argmax(logits)])

    def reset(self) -> None:
        """(the token objects and the tail fills are rewritten at the next step)"""
        self._pos = -1
        self._tkey = None


def output_norm(r: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """The model's output-normed hidden state of a residual r (hidden,): what a draft starts from (bf16)."""
    r = r.float()
    return (r * torch.rsqrt(r.pow(2).mean() + eps) * w).bfloat16()


__all__ = ["DraftImage", "GemmaDraft", "output_norm", "pair_rows"]
