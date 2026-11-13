"""The DFlash2 block drafter of a qwen35 model (GGUF architecture ``dflash``, llama.cpp's ``src/models/dflash.cpp`` and
``common/speculative.cpp``'s draft-dflash) on a decode overlay of its own (``dflash_core``): one dispatch per round, in its own
hardware context, the model's verify between two of them. A dispatch after another hardware context costs ~1.9 ms, a verify too:
~3.7 ms a round, against 10 ms of each MTP draft.

A dispatch holds up to two parts:

- the injection of the verify's kept rows. The drafter's cache takes the rows of the model's sequence: fc over the rows' taps (the
  model's residual before layers ``DFlashConfig.taps``, which are the x of those layers' first phases: its ACT holds them after
  its step, read in place, 5 K groups into the main cores' residual), bf16 g, normed by enc.output_norm; per layer K (RMS-normed,
  rotated) and V of g, into the open object (``dflash_core``), a closed record into the ring.
- the block at position p: the anchor token and 7 mask tokens, non-causal. Per layer, the conv core holding the residual:

  ain (x normed: the conv projection's rows dyn) | conv-in -> aqkv (q, K, V) | the attention over the window and the block | aout
  (o) | conv-out, into the residual | fin (x normed: the FFN's conv projection) | conv-in -> gu (gate / up) | down | conv-out

  then, normed by output_norm, gate (``selector_hidden``, the selector's 256 dims) and lm: the head, the first rows of the model's
  LM head per column in the model's weight buffer (``QwenImage.mtp_view``: the fixed tokens and the table of the sequence's own,
  ``draft.HeadTable``).

The selector runs on the host (``Selector``): per position the top 16 logits, their rows of the file's ``selector_successor`` /
``selector_predecessor`` (dequantized on demand from the mapped file, read ahead after the load), a greedy walk; in a generation's
rounds also a sibling branch where the first draft is uncertain: two tokens from the first position's second candidate on, which the
round's step runs in its last two slots, branching off the round's token (``NpuQwen._sibling``).

The main cores' residual is a scratch here: aout's, down's and fc's K groups sum into it (zero-loaded objects first) and go out
as whole x; the phases need no kernel of their own.

A sequence's state (``snapshot``, ``restore``) holds the ring itself, and the KV records of the prompt cache none of the drafter's: the
ring keeps a window and a chunk's padding, so a position a few hundred back is already losing its window, while a restore in place, the
prompt cache's usual one, goes back over a whole answer.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode import phases as mp
from artlite.decode.design import (
    BC_COL,
    BC_ID,
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_RESID,
    M_FLAGS,
    M_NASKIP,
    M_NREL,
    M_NS,
    M_SKIP,
    M_SP,
    META,
    NX_PER_NORM,
    OBJ_K,
    XB,
    XPAY,
    input_fill,
)
from artlite.decode.image import Region, TileImage, column_rows, gate_up_rows
from artlite.decode.model import InputRegion, ModelOverlay, host_bytes
from artlite.gguf_model import GGUFFile
from artlite.model_config import DFlashConfig
from artlite.operators.design import COLS
from artlite.overlay import OverlayDevice, Sequence, Transaction

from ..common import top_logits
from .attn import OBJ_BYTES
from .dflash_core import (
    CV_IN,
    CV_LOAD,
    CV_OUT,
    D_BYTES,
    DYN_ROWS,
    F_ATT,
    F_CV,
    GROUPS,
    H8,
    HD,
    M_ANOBJ,
    M_CVNE,
    M_CVOP,
    M_G0,
    M_INJ,
    M_KEEP,
    M_POS,
    ND,
    NE_BLOCK,
    NE_INJ,
    OPEN_BYTES,
    PAIR,
    POS_PER_GROUP,
    QC,
    REC_K,
    XK,
    DFlashOperator,
    DFlashShape,
)
from .draft import HeadTable
from .image import KC, payload
from .overlay import KV_PAD

FLAGS = {"fc": F_RESID | F_EMITR, "ikv": F_NORM, "ain": F_NORM, "aqkv": 0, "aout": F_RESID | F_EMITR, "fin": F_NORM, "gu": F_GU,
         "down": F_RESID | F_EMITR, "gate": F_NORM, "lm": F_NORM}
DIMS_OBJ = (XB // 8, 2, XB // 8, XPAY // 4)  # X objects of XB bytes, XPAY apart (the overlapping regions)
SCATTER = (18, ND, 36, 36 * ND)  # a record's drain into its pair (words): ND panels 144 B apart
NE = XK * 2 * 8 * 8 // D_BYTES  # D objects of a column's x
Y_OBJS = -(-XK * 128 // XPAY)  # X objects of a column's y
NOBJ_R = 5  # X objects of a column's res


def conv_rows(c: int) -> np.ndarray:
    """A conv projection's rows of column c: [side 2][tap 2] the column's GROUPS conv groups."""
    return np.concatenate([q * COLS * GROUPS + GROUPS * c + np.arange(GROUPS) for q in range(4)])


class DFlashImage(TileImage):
    """The drafter's weights, its matrices' tiles (the selector's tables stay in the file, read by row): fc, per layer ain,
    aqkv ([q | K | V] of the column's 4 query heads and KV head), aout, fin, gu, down, then gate (``selector_hidden``). An injection's K and V
    are aqkv's rows (``ikv``: a view of its tiles)."""

    def __init__(self, gguf: GGUFFile, cfg: DFlashConfig):
        self.cfg = cfg
        H, I, d = cfg.hidden, cfg.ffn, cfg.head_dim
        super().__init__(gguf, kc=KC, payload=payload(H), tile_bytes=OBJ_BYTES, vocab=0)
        self._add(-2, "fc", [("fc", ("fc.weight",), column_rows(H))], len(cfg.taps) * H)
        q = np.stack([QC * d * c + np.arange(QC * d) for c in range(COLS)])
        kv = np.stack([d * c + np.arange(d) for c in range(COLS)])
        conv = np.stack([conv_rows(c) for c in range(COLS)])
        for L in range(cfg.layers):
            b = f"blk.{L}."
            self._add(L, "ain", [("dyn", (b + "attn_conv_proj.weight",), conv)], H)
            self._add(L, "aqkv", [("q", (b + "attn_q.weight",), q), ("k", (b + "attn_k.weight",), kv), ("v", (b + "attn_v.weight",), kv)], H)
            self._add(L, "aout", [("out", (b + "attn_output.weight",), column_rows(H))], cfg.heads * d)
            self._add(L, "fin", [("dyn", (b + "ffn_conv_proj.weight",), conv)], H)
            R = self.R[self.fmt(b + "ffn_gate.weight")]
            self._add(L, "gu", [("gu", (b + "ffn_gate.weight", b + "ffn_up.weight"), gate_up_rows(I, R))], H)
            self._add(L, "down", [("down", (b + "ffn_down.weight",), column_rows(H))], I)
        self._add(-1, "gate", [("gate", ("selector_hidden.weight",), column_rows(cfg.rank))], H)
        self._place()
        self.ikv: dict[int, Region] = {}
        for L in range(cfg.layers):
            r = self.region(L, "aqkv")
            blocks = r.blocks[1:]
            rt = r.blocks[0].tiles
            self.ikv[L] = Region(L, "ikv", blocks, r.n_kc, r.kc_group, r.tile_bytes, r.kc,
                                 offset=r.offset + rt * r.n_kc * r.tile_bytes, col_stride=r.col_stride)

    def formats(self) -> set[str]:
        return {b.fmt for r in self.regions for b in r.blocks}


class Selector:
    """DFlash2's selector on the host (llama.cpp's ``build_dflash2_selector`` and its greedy walk): the drafts of a block from its logits and
    gates. Of the candidates' scores only the chosen predecessor's row is needed: per position the rows of its top_k candidates in
    ``selector_successor`` and of the predecessor in ``selector_predecessor``, read for all positions at once (the predecessors': the
    anchor's and every candidate's but the last position's)."""

    TABLES = ("selector_successor.weight", "selector_predecessor.weight")
    SIBLING = 2.0  # a sibling's probability against this times its chain's (``walk``)

    def __init__(self, gguf: GGUFFile, cfg: DFlashConfig):
        self.gguf, self.top_k = gguf, cfg.top_k
        self.sibling: list[int] | None = None

    def resident(self) -> None:
        """The tables' pages read ahead (after the drafter's load released the file's): ~200 scattered rows a round."""
        for name in self.TABLES:
            self.gguf.willneed(name)

    def walk(self, logits: np.ndarray, gates: np.ndarray, anchor: int, ids: np.ndarray, n: int, stop: set[int], *,
             sibling: int = 0) -> list[int]:
        """Up to n drafts after the anchor token (logits (block, vocab): rows of the tokens ``ids``), ending with a token of ``stop``.
        ``sibling`` > 0: also ``self.sibling``, a branch of that many tokens from the first position's second candidate on (its
        walk) when that candidate's probability exceeds ``SIBLING`` times the n drafts' (the probabilities: the lattice's softmax
        over a position's candidates), else None: the drafts that a step gives their slots are the last ones, kept with the
        chain's probability, the branch where the first draft is not (prose: a third of first drafts at 32K, of which the
        second candidate is right in a third to a half)."""
        self.sibling = None
        if n <= 0:
            return []
        k = self.top_k
        top = [top_logits(logits[i], k, rows=ids) for i in range(1, n + 1)]
        c = np.stack([c for _, c in top])
        succ = self.gguf.rows(self.TABLES[0], c.reshape(-1), prefetch=False).reshape(n, k, -1)
        pred = self.gguf.rows(self.TABLES[1], np.concatenate([[anchor], c[:-1].reshape(-1)]), prefetch=False)
        out: list[int] = []
        j = 0  # (the predecessor's row)
        chain, p2, b = 1.0, 0.0, 0
        for i in range(n):
            sc = succ[i] @ (pred[j] * gates[i + 1]) + top[i][0]
            a = int(np.argmax(sc))
            if sibling:
                e = np.exp(sc - sc[a])
                chain /= e.sum()
                if i == 0:
                    e[a] = 0.0
                    b = int(np.argmax(e))
                    p2 = e[b] * chain
            out.append(int(c[i, a]))
            if out[-1] in stop:
                break
            j = 1 + i * k + a
        if sibling and len(out) == n and p2 > self.SIBLING * chain:
            self.sibling, j = [int(c[0, b])], 1 + b
            for i in range(1, sibling):
                a = int(np.argmax(succ[i] @ (pred[j] * gates[i + 1]) + top[i][0]))
                self.sibling.append(int(c[i, a]))
                j = 1 + i * k + a
        return out


class _Step:
    """A dispatch's transaction and its words that follow the positions: the ring's fills (BD word index, layer, column, run) and an
    injection's record drains (BD word index, layer, column, 0: K, 1: V)."""

    def __init__(self, S: int, block: bool):
        self.S, self.block = S, block
        self.tr: Transaction | None = None
        self.runs: list[tuple[int, int, int, int]] = []
        self.recs: list[tuple[int, int, int, int]] = []


class DFlashOverlay(ModelOverlay):
    """The drafter ``gguf`` of the model on overlay ``target``. ``load`` after the target's (its device, weight and activation
    buffers); ``step`` runs a dispatch."""

    CONFIG = DFlashConfig
    IMAGE = DFlashImage
    SLOTS_OF = 8  # (the model's step: the drafter's injections read its slots' x)

    def __init__(self, target: object, gguf: GGUFFile):
        self.target = target
        self.cfg = cfg = DFlashConfig.from_gguf(gguf)
        tc = target.cfg
        if (cfg.hidden != tc.hidden_size or len(cfg.taps) * cfg.hidden != gguf.tensor("fc.weight").cols
                or max(cfg.taps) >= tc.num_hidden_layers):
            raise ValueError(f"{gguf.path}: the drafter of another model than {target.gguf.path}")
        if not target.image.draft_cols or target.SLOTS != self.SLOTS_OF:
            raise ValueError(f"{gguf.path}: a DFlash2 drafter goes with a model of steps of {self.SLOTS_OF} tokens and a drafter head "
                             f"(an MTP block), not {target.gguf.path}")
        self.shape = DFlashShape(cfg)
        self.image = DFlashImage(gguf, cfg)
        view = target.image.mtp_view(target.cfg)
        self.table = HeadTable(view, tc.vocab_size, lambda: target.dev)
        self.head = view.region(-1, "lm")
        self.image.vocab = self.head.n_rt * self.head.blocks[0].R * COLS
        super().__init__(gguf, layers=list(range(cfg.layers)), vocab=None, weights=None, image=self.image)
        fmts = self.image.formats() | {self.head.blocks[0].fmt}
        self.op = DFlashOperator(hidden=cfg.hidden, rows={f: self.image.R[f] for f in sorted(fmts)}, shape=self.shape, eps=cfg.rms_norm_eps)
        self.flags = FLAGS
        self.n_norm = -(-cfg.hidden // (NX_PER_NORM * OBJ_K))
        self.selector = Selector(gguf, cfg)
        self._cs: dict[int, np.ndarray] = {}  # (``_angles``)
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _flags(self, L: int, name: str) -> int:
        return self.flags[name]

    def region(self, L: int, name: str) -> Region:
        if name == "ikv":
            return self.image.ikv[L]
        return self.head if name == "lm" else self.image.region(L, name)

    def _groups(self, L: int, name: str, S: int) -> list[mp.Group]:
        out = mp.groups(self.region(L, name), self.flags[name], S, slots=self.SLOTS)
        for g in out:
            g.meta[M_SP] = self._sp(S)
        return out

    def _input(self, L: int) -> tuple[int, str]:
        raise NotImplementedError

    def _add_meta(self, phase: tuple[int, str]) -> None:
        """A phase that reads another's x: its K groups' metadata objects only."""
        r = self.region(*phase)
        self.inp[phase] = [InputRegion(self._take("ACT", XB), k0 * r.kc, g * r.kc) for k0, g in mp.group_chunks(r)]

    def _add_inputs(self, phase: tuple[int, str], *, post: bool = False, extra: int = 0) -> None:
        r = self.region(*phase)
        assert not post
        regs = []
        for k0, g in mp.group_chunks(r):
            K = g * r.kc
            regs.append(InputRegion(self._take("ACT", XB + 2 * self.SLOTS * K + extra), k0 * r.kc, K))
        self.inp[phase] = regs

    def _plan(self) -> None:
        """Byte offsets in ACT (the objects), NORM, KV (per layer and KV head the ring, then the open objects)."""
        cfg, T = self.cfg, self.T
        self._layout = dict.fromkeys(("ACT", "NORM", "KV", "GATE"), 0)
        take = self._take
        self.loadr = take("ACT", COLS * self.SLOTS * XB)
        self._add_meta((-2, "fc"))
        self._add_inputs((0, "ikv"))
        for L in range(1, cfg.layers):
            self._add_meta((L, "ikv"))
        for L in range(cfg.layers):
            for name in ("ain", "aqkv", "aout", "fin", "gu", "down"):
                self._add_inputs((L, name))
        self._add_inputs((-1, "gate"))
        self._add_meta((-1, "lm"))
        nw = 2 * NX_PER_NORM * OBJ_K * (self.n_norm - 1) + T
        self.norm_off = {name: take("NORM", nw) for name in ("enc", "output")}
        for L in range(cfg.layers):
            self.norm_off[(L, "attn")] = take("NORM", nw)
            self.norm_off[(L, "ffn")] = take("NORM", nw)
        self.cload = take("ACT", COLS * (XB + NOBJ_R * XPAY))  # [column][F_CV object | res]
        # (per layer and branch, per column: conv-in's object, dyn's object; conv-out's object | y)
        self.cvin = take("ACT", cfg.layers * 2 * COLS * 2 * XB)
        self.cvy = take("ACT", cfg.layers * 2 * COLS * (XB + Y_OBJS * XPAY))
        self.fatt = take("ACT", 2 * 2 * XB)  # ([block | injection][F_ATT object | rope object])
        self.norms = take("ACT", cfg.layers * XB)  # (a layer's q_norm, k_norm)
        self.qobj = take("ACT", cfg.layers * COLS * self.SLOTS * XB)  # [layer][column][slot]: q, K, V
        self.kvobj = take("ACT", cfg.layers * COLS * self.SLOTS * XB)  # [layer][column][slot]: K, V of an injection
        self.ring_bytes = (self.shape.ring + 1 + self.shape.over) * T  # (+ 1: an empty run's object; a chunk's writes past the ring)
        self.ring_off = take("KV", cfg.layers * COLS * self.ring_bytes)
        self.open_off = take("KV", cfg.layers * COLS * T)
        self.sink = take("KV", 2 * PAIR)  # (the records no injection closes)
        take("KV", KV_PAD // POS_PER_GROUP * T)  # (the prefill's sub-chunk slices of the last layer's rings run past them:
        # ``dflash_prefill``'s DFlashPrepOperator, a sub-chunk's rings from its first object on)
        take("GATE", 2 * COLS * self.SLOTS * C_LEN)
        self._sizes()

    def _ring(self, L: int, c: int) -> int:
        """KV offset of layer L's KV head c's ring of A objects."""
        return self.ring_off + (L * COLS + c) * self.ring_bytes

    def _open(self, L: int, c: int) -> int:
        return self.open_off + (L * COLS + c) * self.T

    def _cvin(self, L: int, b: int, c: int) -> int:
        """ACT offset of the conv-in object (b: 0 attention, 1 FFN), dyn's object after it."""
        return self.cvin + ((L * 2 + b) * COLS + c) * 2 * XB

    def _cvy(self, L: int, b: int, c: int) -> int:
        """ACT offset of the conv-out object, its y's objects overlapping it."""
        return self.cvy + ((L * 2 + b) * COLS + c) * (XB + Y_OBJS * XPAY)

    def _q(self, L: int, c: int) -> int:
        return self.qobj + (L * COLS + c) * self.SLOTS * XB

    def _kv(self, L: int, c: int) -> int:
        return self.kvobj + (L * COLS + c) * self.SLOTS * XB

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, *, threads: int = 0, release: bool = True) -> None:
        """The design in its own hardware context on the target's device, the image packed, the target's buffers adopted, the static
        objects written."""
        img, dev0 = self.image, self.target.dev
        self.dev = dev = OverlayDevice(self.op, device=dev0.device)
        if img.pack(dev.alloc("W", img.nbytes), threads=threads, release=release):
            dev.flush("W")
        self.selector.resident()
        dev.adopt("MW", dev0.buffers["W"])
        dev.adopt("MACT", dev0.buffers["ACT"])
        for name, n in self.sizes.items():
            dev.alloc(name, n)
        self._load()
        dev.flush("ACT")
        self.reset()

    def _f32(self, name: str) -> np.ndarray:
        return np.asarray(self.gguf.tensor(name).raw, np.float32).reshape(-1).copy()

    def _base(self, name: str) -> np.ndarray:
        """A conv's base (2 side, 2 tap, hidden) bf16 bits."""
        x = self._f32(name).astype(bfloat16).view(np.uint16)
        return x.reshape(2, 2, self.cfg.hidden)

    def _load(self) -> None:
        cfg = self.cfg
        norm, act = self.dev.view("NORM"), self.dev.view("ACT")

        def bf(x: np.ndarray) -> np.ndarray:
            return x.astype(bfloat16).view(np.uint8)

        for key, name in (("enc", "enc.output_norm.weight"), ("output", "output_norm.weight")):
            norm[self.norm_off[key]:][:2 * self.H] = bf(self._f32(name))
        for L in range(cfg.layers):
            b = f"blk.{L}."
            for key, name in (("attn", "attn_norm"), ("ffn", "ffn_norm")):
                norm[self.norm_off[(L, key)]:][:2 * self.H] = bf(self._f32(b + name + ".weight"))
            q = self._f32(b + "attn_q_norm.weight") * np.float32(np.log2(np.e) / np.sqrt(HD))
            act[self.norms + L * XB + 4 * META:][:4 * 2 * HD] = np.concatenate([q, self._f32(b + "attn_k_norm.weight")]).view(np.uint8)
            for k, (w, base) in enumerate((("attn_norm", "attn_conv_base"), ("ffn_norm", "ffn_conv_base"))):
                wb = self._f32(b + w + ".weight").astype(bfloat16).view(np.uint16).reshape(COLS, H8)
                bs = self._base(b + base)
                for c in range(COLS):
                    o = self._cvin(L, k, c)
                    m = np.zeros(META, np.int32)
                    m[M_FLAGS], m[M_CVOP], m[M_SKIP], m[M_CVNE] = F_CV, CV_IN, 1, NE
                    act[o:o + 4 * META] = m.view(np.uint8)
                    p0 = np.concatenate([wb[c], bs[0, 0, H8 * c:H8 * (c + 1)], bs[0, 1, H8 * c:H8 * (c + 1)]])
                    act[o + 4 * META:][:p0.nbytes] = p0.view(np.uint8)
                    o = self._cvy(L, k, c)
                    m[M_CVOP], m[M_SKIP] = CV_OUT, Y_OBJS
                    act[o:o + 4 * META] = m.view(np.uint8)
                    p1 = np.concatenate([bs[1, 0, H8 * c:H8 * (c + 1)], bs[1, 1, H8 * c:H8 * (c + 1)]])
                    act[o + 4 * META:][:p1.nbytes] = p1.view(np.uint8)
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_CVOP], m[M_SKIP], m[M_CVNE] = F_CV, CV_LOAD, NOBJ_R, NE
        for c in range(COLS):
            act[self.cload + c * (XB + NOBJ_R * XPAY):][:4 * META] = m.view(np.uint8)
        for s in range(self.SLOTS):
            for c in range(COLS):
                act[self.loadr + (c * self.SLOTS + s) * XB:][:4 * META] = mp.loadr_meta(s).view(np.uint8)
        for (L, name), regs in self.inp.items():  # (a block's phases: S 8, static)
            if name not in ("fc", "ikv"):
                for rg, gr in zip(regs, self._groups(L, name, self.SLOTS), strict=True):
                    act[rg.off:rg.off + 4 * META] = gr.meta.view(np.uint8)

    def reset(self) -> None:
        """A new sequence: no cached positions."""
        self.dev.view("KV")[:] = 0
        self.dev.flush("KV")
        self._metas_s = (0, 0)
        self.next = 0  # (the next position to inject)

    # ------------------------------------------------------------------
    # The transaction
    # ------------------------------------------------------------------

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """(an injection's phases: its S)"""
        return [(rg.off, gr.meta.view(np.uint8)) for (L, name), regs in self.inp.items() if name in ("fc", "ikv")
                for rg, gr in zip(regs, self._groups(L, name, S), strict=True)]

    def _weights(self, seq: Sequence, phase: tuple[int, str], gi: int, norm: object = None) -> None:
        """The A fills of a phase's K group gi: its norm-weight objects (``norm``: NORM's key), its tiles."""
        dev, (L, name) = self.dev, phase
        region = self.region(L, name)
        gr = mp.groups(region, FLAGS[name], 1, slots=self.SLOTS)[gi]
        buf = "MW" if name == "lm" else "W"
        for c in range(COLS):
            if gi == 0 and norm is not None:  # (object n: T bytes from weight 4096 n on)
                seq.fill("A", c, dev.address("NORM", self.norm_off[norm]), self.n_norm * self.T,
                         (self.T // 16, 4, self.T // 16, NX_PER_NORM * OBJ_K // 2))
            for o, n, dims, rep, it in mp.a_reads(region, c, gr):
                seq.fill("A", c, dev.address(buf, o), n, dims, rep, it)

    def _bc(self, seq: Sequence, meta: int, K: int, sp: int, x: int | None = None) -> None:
        """A phase's input on the broadcast stream: its [metadata object | x] at ACT offset ``meta``, or x elsewhere (ACT offset x)."""
        nbytes, dims = input_fill(K, sp, self.SLOTS)
        dev = self.dev
        if x is None:
            seq.fill("X", BC_COL, dev.address("ACT", meta), nbytes, dims, packet=BC_ID)
            return
        seq.fill("X", BC_COL, dev.address("ACT", meta), XB, packet=BC_ID)
        seq.fill("X", BC_COL, dev.address("ACT", x + XB - 4 * META), nbytes - XB, dims, packet=BC_ID)

    def _d_to(self, seq: Sequence, phase: tuple[int, str]) -> None:
        """The conv core's x -> a phase's x."""
        rg = self.inp[phase][0]
        for c in range(COLS):
            seq.drain("D", c, self.dev.address("ACT", rg.off + XB + 2 * self.SLOTS * H8 * c), 2 * self.SLOTS * H8)

    def _x_drains(self, seq: Sequence, c: int, phase: tuple[int, str], S: int) -> None:
        """Column c's residual out of the main core (F_EMITR) -> a phase's x."""
        self._values_to(seq, c, phase, e0=H8 * c, n=H8, S=S, whole=True)

    def _inject(self, seq: Sequence, st: _Step) -> None:
        """fc over the rows' taps, g normed, per layer K and V, the attention core's records."""
        dev, S, cfg, tgt = self.dev, st.S, self.cfg, self.target
        sp = self._sp(S)
        for c in range(COLS):  # (the main cores' residual zero; fc's K groups local: nothing drained yet)
            seq.fill("X", c, dev.address("ACT", self.loadr + c * self.SLOTS * XB), S * XB)
        phase = (-2, "fc")
        for gi, rg in enumerate(self.inp[phase]):
            self._weights(seq, phase, gi)
            nbytes, dims = input_fill(rg.K, sp, self.SLOTS)
            tap = tgt.inp[tgt._input(cfg.taps[gi])][0].off + XB - 4 * META
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", rg.off), XB)
                seq.fill("X", c, dev.address("MACT", tap), nbytes - XB, dims)
        for c in range(COLS):
            self._x_drains(seq, c, (0, "ikv"), S)
        g0 = self.inp[(0, "ikv")][0]
        for L in range(cfg.layers):
            self._weights(seq, (L, "ikv"), 0, "enc")
            seq.sync()
            rg = self.inp[(L, "ikv")][0]
            self._bc(seq, rg.off, rg.K, sp, None if L == 0 else g0.off)
            for c in range(COLS):  # [K | V] per slot
                seq.drain("C", c, dev.address("ACT", self._kv(L, c) + 4 * META), S * 2 * 2 * HD, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            seq.sync()
            for c in range(COLS):
                seq.fill("A", c, dev.address("KV", self._open(L, c)), self.T)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self.fatt + 2 * XB), 2 * XB)
                seq.fill("X", c, dev.address("ACT", self.norms + L * XB), XB)
                seq.fill("X", c, dev.address("ACT", self._kv(L, c)), S * XB)
            for c in range(COLS):
                seq.drain("C", c, dev.address("KV", self._open(L, c)), OPEN_BYTES)
                st.recs.append((seq.drain("C", c, dev.address("KV", self.sink), REC_K, SCATTER)[0], L, c, 0))
                st.recs.append((seq.drain("C", c, dev.address("KV", self.sink + PAIR), REC_K, SCATTER)[0], L, c, 1))

    def _conv_in(self, seq: Sequence, L: int, b: int, to: tuple[int, str]) -> None:
        """After a phase's rows: the conv core's objects; its x -> ``to``."""
        seq.sync()
        for c in range(COLS):
            seq.fill("X", c, self.dev.address("ACT", self._cvin(L, b, c)), 2 * XB)
        self._d_to(seq, to)

    def _rows_to(self, seq: Sequence, phase: tuple[int, str], L: int, b: int) -> None:
        """A normed phase's rows (dyn) -> its conv object."""
        for c in range(COLS):
            seq.drain("C", c, self.dev.address("ACT", self._cvin(L, b, c) + XB + 4 * META), 2 * self.SLOTS * DYN_ROWS,
                      (C_LEN // 2, self.SLOTS, DYN_ROWS // 2, C_LEN // 2))

    def _phase_in(self, seq: Sequence, phase: tuple[int, str], norm: object = None) -> None:
        """A phase's weights, its inputs after the drains so far."""
        for gi, rg in enumerate(self.inp[phase]):
            self._weights(seq, phase, gi, norm)
            if gi == 0:
                seq.sync()
            self._bc(seq, rg.off, rg.K, 8)

    def _conv_out(self, seq: Sequence, L: int, b: int, to: tuple[int, str]) -> None:
        """An output phase's x (F_EMITR) in DDR: conv-out; x -> ``to``."""
        seq.sync()
        for c in range(COLS):
            seq.fill("X", c, self.dev.address("ACT", self._cvy(L, b, c)), (1 + Y_OBJS) * XB, DIMS_OBJ)
        self._d_to(seq, to)

    def _y_drains(self, seq: Sequence, L: int, b: int) -> None:
        for c in range(COLS):
            seq.drain("C", c, self.dev.address("ACT", self._cvy(L, b, c) + XB), 2 * self.SLOTS * H8)

    def _block(self, seq: Sequence, st: _Step) -> None:
        dev, cfg, S = self.dev, self.cfg, self.SLOTS
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.cload + c * (XB + NOBJ_R * XPAY)), (1 + NOBJ_R) * XB, DIMS_OBJ)
        self._d_to(seq, (0, "ain"))
        for L in range(cfg.layers):
            self._phase_in(seq, (L, "ain"), (L, "attn"))
            self._rows_to(seq, (L, "ain"), L, 0)
            self._conv_in(seq, L, 0, (L, "aqkv"))
            self._phase_in(seq, (L, "aqkv"))
            for c in range(COLS):
                seq.drain("C", c, dev.address("ACT", self._q(L, c) + 4 * META), S * 2 * (QC + 2) * HD, (C_LEN // 2, S, XB // 4, C_LEN // 2))
            seq.sync()
            for c in range(COLS):
                seq.fill("A", c, dev.address("KV", self._open(L, c)), self.T)
                for run in range(2):
                    st.runs.append((seq.fill("A", c, dev.address("KV", self._ring(L, c)), self.T), L, c, run))
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self.loadr + c * S * XB), S * XB)  # (aout's sum: zero first)
                seq.fill("X", c, dev.address("ACT", self.fatt), 2 * XB)
                seq.fill("X", c, dev.address("ACT", self.norms + L * XB), XB)
                seq.fill("X", c, dev.address("ACT", self._q(L, c)), S * XB)
            rg = self.inp[(L, "aout")][0]
            for c in range(COLS):  # o: the column's heads in aout's x
                seq.drain("C", c, dev.address("ACT", rg.off + XB + 2 * S * QC * HD * c), 2 * S * QC * HD)
            self._phase_in(seq, (L, "aout"))
            self._y_drains(seq, L, 0)
            self._conv_out(seq, L, 0, (L, "fin"))
            self._phase_in(seq, (L, "fin"), (L, "ffn"))
            self._rows_to(seq, (L, "fin"), L, 1)
            seq.sync()
            for c in range(COLS):  # (down's sum: zero first; the conv objects' drains cover these fills)
                seq.fill("X", c, dev.address("ACT", self.loadr + c * S * XB), S * XB)
            for c in range(COLS):
                seq.fill("X", c, dev.address("ACT", self._cvin(L, 1, c)), 2 * XB)
            self._d_to(seq, (L, "gu"))
            self._phase_in(seq, (L, "gu"))
            for c in range(COLS):
                self._values_to(seq, c, (L, "down"), e0=cfg.ffn // COLS * c, n=cfg.ffn // COLS, S=S)
            self._phase_in(seq, (L, "down"))
            self._y_drains(seq, L, 1)
            self._conv_out(seq, L, 1, (L + 1, "ain") if L + 1 < cfg.layers else (-1, "gate"))
        self._phase_in(seq, (-1, "gate"), "output")
        for c in range(COLS):  # (32 rows of each slot's C object)
            seq.drain("C", c, dev.address("GATE", c * S * 2 * C_LEN), S * 2 * C_LEN)
        seq.sync()
        self._weights(seq, (-1, "lm"), 0, "output")
        rg = self.inp[(-1, "gate")][0]
        self._bc(seq, self.inp[(-1, "lm")][0].off, self.H, 8, rg.off)
        for c in range(COLS):
            self._logits_out(seq, c, S)

    def _build(self, S: int, block: bool, subs: tuple = ()) -> _Step:
        """The transaction of a dispatch: an injection of a verify's S rows (0: none), the block."""
        seq = Sequence(self.op, rolling=True, inputs=("X",))
        st = _Step(S, block)
        if S:
            self._inject(seq, st)
        if block:
            self._block(seq, st)
        seq.sync(fills=True)
        st.tr = Transaction(self.dev, seq.words())
        st.patch = self._tables(st)
        return st

    def _tables(self, st: _Step) -> dict[str, np.ndarray]:
        r = np.array(st.runs, np.int64).reshape(-1, 4)
        k = np.array(st.recs, np.int64).reshape(-1, 4)
        at = np.concatenate([r[:, 0], k[:, 0], r[:, 0] + 2, k[:, 0] + 2])
        return {"r": r, "k": k, "lines": np.unique(4 * at & ~63)}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _fatt(self, kind: int, p: int, S: int, *, keep: int = 0, nobj: int = 0, g0: int = 0) -> None:
        """The F_ATT and rope objects of a dispatch's block (kind 0) or injection (1)."""
        act = self.dev.view("ACT")
        o = self.fatt + kind * 2 * XB
        m = act[o:o + 4 * META].view(np.int32)
        m[:] = 0
        m[M_FLAGS], m[M_NS], m[M_POS], m[M_INJ], m[M_KEEP], m[M_ANOBJ], m[M_G0] = F_ATT, S, p, kind, keep, nobj, g0
        m[M_SKIP], m[M_NASKIP], m[M_NREL] = 2 + S, 1 + nobj, NE_INJ if kind else NE_BLOCK
        cs = np.zeros((self.SLOTS, HD), np.float32)
        cs[:S] = self._angles(p, S)
        act[o + XB + 4 * META:][:cs.nbytes] = cs.view(np.uint8).reshape(-1)
        self.dev.flush("ACT", o, 2 * XB)

    def _angles(self, p: int, S: int) -> np.ndarray:
        """[cos | sin] (S, HD) fp32 of positions p .. p + S - 1, from tables of 64 positions (``rope_theta``'s loop over a head's pairs
        costs ~0.1 ms a call: a round's two dispatches read two or three tables)."""
        out = []
        for c in range(p >> 6, ((p + S - 1) >> 6) + 1):
            if c not in self._cs:
                if len(self._cs) > 8:
                    self._cs.clear()
                th = self.cfg.rope_theta(np.arange(64 * c, 64 * c + 64)).astype(np.float64)
                self._cs[c] = np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)
            out.append(self._cs[c])
        return np.concatenate(out)[p & 63:(p & 63) + S]

    def _window(self, p: int) -> tuple[list[tuple[int, int]], int]:
        """A block at p: the ring's two runs (first object's slot, objects; an empty run the pad object) and the first object."""
        G = self.shape.ring
        ro = p >> 3
        first = max(p - self.shape.window + 1, 0) // POS_PER_GROUP
        last = (ro - 1) >> 2 if ro else 0
        n, a = last - first + 1, first % G
        n1 = min(n, G - a)
        return [(a, n1), (0, n - n1) if n > n1 else (G, 1)], first

    def _patch(self, st: _Step, *, p: int | None, rec: tuple[int, int] | None) -> int:
        """The ring's fills of a block at p; an injection's record drains (rec: (record, its object in the ring), or None: into the sink).
        Returns the ring's objects."""
        words = st.tr.words
        kv0, T = self.dev.address("KV"), self.T
        nobj = 0
        t = st.patch
        if p is not None and len(t["r"]):
            runs, _ = self._window(p)
            nobj = sum(n for _, n in runs)
            r = t["r"]
            base = self.ring_off + (r[:, 1] * COLS + r[:, 2]) * self.ring_bytes
            a = np.array(runs)[r[:, 3]]
            self._bd_words(words, r[:, 0], kv0 + base + a[:, 0] * T, a[:, 1] * T)
        k = t["k"]
        if len(k):
            if rec is None:
                off = self.sink + k[:, 3] * PAIR
            else:
                r0, g = rec
                base = self.ring_off + (k[:, 1] * COLS + k[:, 2]) * self.ring_bytes
                off = base + g * T + k[:, 3] * 2 * PAIR + ((r0 & 3) >> 1) * PAIR + (r0 & 1) * 72
            self._bd_words(words, k[:, 0], kv0 + off, np.full(len(off), REC_K, np.int64))
        st.tr.flush_lines(t["lines"])
        return nobj

    def _transaction_of(self, S: int, block: bool) -> _Step:
        key = (S, block)
        if key not in self._steps:
            self._steps[key] = self._build(S, block)
        return self._steps[key]

    def prepare(self, S: int, logits: bool = True, start: int | None = None) -> None:
        """The transactions of an injection of S rows (0: none) with the block and of one alone."""
        self._transaction_of(S, True)
        if S:
            self._transaction_of(S, False)
            self._meta_of(S)

    def step(self, S: int, keep: int, q: int, x: np.ndarray | None, p: int, *, meanwhile: Callable[[], None] | None = None) -> None:
        """A dispatch: the injection of the first ``keep`` rows at q .. of the model's last step of S slots (S 0: none), then with rows x
        (8, hidden) fp32 (the anchor's and the mask token's embeddings; None: no block) the block at p. The results: ``logits``,
        ``gates``. ``meanwhile``: host work run while the dispatch runs (the model's next step patched ahead)."""
        assert not S or q == self.next, (q, self.next)
        st = self._transaction_of(S, x is not None)
        rec = None
        if S:
            if self._metas_s[0] != S:
                self._metas(S)
            self.next = q + keep
            rc = (self.next >> 3) - 1
            if rc >= q >> 3:
                rec = (rc, (rc >> 2) % self.shape.ring)
        nobj = self._patch(st, p=p if x is not None else None, rec=rec)
        if S:
            self._fatt(1, q, S, keep=keep)
        if x is not None:
            _, first = self._window(p)
            self._fatt(0, p, 8, nobj=nobj, g0=first)
            self._write_rows(x)
        self.ready()
        if meanwhile is None:
            self.dev.dispatch(st.tr)
            return
        self.dev.submit(st.tr)
        try:
            meanwhile()
        finally:
            self.dev.wait()

    def _write_rows(self, x: np.ndarray) -> None:
        """The block's rows (8, hidden) fp32 -> the conv cores' res ([column][k block][slot 8][8])."""
        act = self.dev.view("ACT")
        n = x.reshape(8, COLS, XK, 8).transpose(1, 2, 0, 3).astype(np.float32)
        for c in range(COLS):
            o = self.cload + c * (XB + NOBJ_R * XPAY)
            act[o + XB:][:n[c].nbytes] = n[c].view(np.uint8).reshape(-1)
        self.dev.flush("ACT", self.cload, COLS * (XB + NOBJ_R * XPAY))

    def snapshot(self, position: int) -> dict[str, Any]:
        """What continuing from ``position`` needs of the drafter (the module docstring): the ring's objects of the positions before it,
        the open objects, the next position."""
        assert position == self.next, (position, self.next)
        dev, kv, T = self.dev, self.dev.view("KV"), self.T
        n, heads = min(-(-position // POS_PER_GROUP), self.shape.ring), self.cfg.layers * COLS
        ring, opened = host_bytes((heads, n, T)), host_bytes((heads * T,))
        for i in range(heads):
            o = self.ring_off + i * self.ring_bytes
            dev.fetch("KV", o, n * T)
            ring[i] = kv[o:o + n * T].reshape(n, T)
        dev.fetch("KV", self.open_off, heads * T)
        opened[:] = kv[self.open_off:self.open_off + heads * T]
        return {"ring": ring, "open": opened, "next": self.next}

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Back to a ``snapshot``."""
        dev, kv, T = self.dev, self.dev.view("KV"), self.T
        ring, opened = snapshot["ring"], snapshot["open"]
        n = ring.shape[1]
        for i in range(ring.shape[0]):
            o = self.ring_off + i * self.ring_bytes
            kv[o:o + n * T] = ring[i].reshape(-1)
            dev.flush("KV", o, n * T)
        kv[self.open_off:self.open_off + opened.size] = opened
        dev.flush("KV", self.open_off, opened.size)
        self.next = snapshot["next"]

    def logits(self) -> np.ndarray:
        """The block's head logits (8, vocab) bf16 (a view of the NPU's buffer; [slot][the head's rows])."""
        self.dev.fetch("LOGITS")
        return self.dev.view("LOGITS").view(bfloat16).reshape(8, self.vocab)

    def gates(self) -> np.ndarray:
        """The block's selector gates (8, rank) fp32."""
        self.dev.fetch("GATE")
        g = self.dev.view("GATE").view(np.uint16).reshape(COLS, 8, C_LEN)
        per = self.cfg.rank // COLS
        return g[:, :, :per].astype(np.uint16).view(bfloat16).astype(np.float32).transpose(1, 0, 2).reshape(8, -1)


__all__ = ["DFlashImage", "DFlashOverlay", "Selector", "conv_rows"]
