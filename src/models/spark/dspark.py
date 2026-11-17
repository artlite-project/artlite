"""The DSpark block drafter of a Spark-X model (``artlite.model_config.DSparkConfig``; XHToken's llama.cpp fork's
``src/models/dflash.cpp``) on the model's decode overlay: the model's design and hardware context (``SparkOverlay.op``,
which runs the drafter's BFP16 tiles and gate / up rows: ``needs``), buffers of its own, its weights in a buffer of their
own (``dspark_image.DSparkImage``), its head the first rows of each column of the model's LM head, read in place
(``decode.head``: the model's image laid out with them, ``SparkOverlay(head=True)``). One dispatch a round (``run``):

- the injection of the model's last step's rows: their taps (the residual entering the layers ``taps``: the x of
  those layers' attn_in in the model's ACT, read in place) -> fc of each tap (normed by its row of ``fc_norm``)
  into the main cores' residual (zero-loaded first), emitted as g; per layer ikv (g normed by ``enc.output_norm``: K, V)
  and the attention core's records of the rows' positions into the layer's ring (``AttnStep`` mode "insert": the step's
  groups streamed, no window, o into a sink);
- the block at P: rows [anchor, mask x 6] at P .. P + 6 (the token embeddings), per layer attn_in | the attention (mode
  "block": every row the ``window`` positions before P and the block whole) | attn_out | gate_up | down, then the head
  (normed by ``output_norm``) -> logits [7][the head's rows]; the block's records at P .. P + 6 stay in the rings until
  the next injection writes the kept rows over them (a block's slots past the kept rows: the next block's).

An injection writes the records of all S slots of the model's step; those past the kept rows are the next block's
positions, which it writes again. The caches: per layer and KV head a ring of ``nr`` groups (group g at g % nr: the
block's window and its own groups; ``decode.attn256_steps``' "ra" / "rb" fills), the open records' rings
(``decode.vring``); a sequence's state (``snapshot``) holds the window's groups before a position, as the model's
sliding-window layers'. Transactions per (injected slots, block) are built at their first use and patched per round: the
two parts' F_ATT objects are the two buffers of ``AttentionSteps`` (b 0: the injection's, 1: the block's)."""

import dataclasses
from collections.abc import Callable

import numpy as np
from ml_dtypes import bfloat16

from artlite.decode import attn256 as attn
from artlite.decode import phases as mp
from artlite.decode.attn256 import F_ATT, KIND_SWA, M_KIND, M_WREC
from artlite.decode.attn256_steps import AttentionSteps, AttnStep
from artlite.decode.head import HeadTable, HeadView
from artlite.decode.model import InputRegion, ModelOverlay
from artlite.decode.protocol import (
    BC_COL,
    BC_ID,
    F_EMITR,
    F_GU,
    F_NORM,
    F_RESID,
    M_FLAGS,
    META,
    NX_PER_NORM,
    OBJ_K,
    XB,
    input_fill,
    interleave,
    packed_slots,
    region_bytes,
)
from artlite.gguf_model import GGUFFile
from artlite.model_config import DSparkConfig
from artlite.operators.design import COLS
from artlite.operators.util import cos_sin_rows
from artlite.overlay import OverlayDevice, Sequence, Transaction

from .dspark_image import DSparkImage
from .overlay import SparkOverlay

FLAGS = {"ikv": F_NORM, "attn_in": F_NORM, "attn_out": F_RESID | F_EMITR, "gate_up": F_NORM | F_GU,
         "down": F_RESID | F_EMITR, "lm": F_NORM}
RING_PAD = 4  # the rings' groups past a window's


@dataclasses.dataclass(eq=False)
class Round:
    """The transaction of a round: the injection's attention step (S slots; None: no injection) and the block's."""

    tr: Transaction
    inj: AttnStep | None
    blk: AttnStep | None


def needs(gguf: GGUFFile) -> dict[str, object]:
    """What the model's overlay takes for a DSpark file's drafter (``SparkOverlay``'s arguments): its tile formats,
    gate / up rows per tile and the drafter's head in the LM rows."""
    img = DSparkImage(gguf, DSparkConfig.from_gguf(gguf))
    return {"formats": tuple(sorted(img.formats())), "gu": tuple(sorted(img.gu_rows())), "head": True}


class DSparkOverlay(SparkOverlay):
    """The DSpark drafter ``gguf`` of the model on overlay ``target`` (module docstring): ``load`` after the target's
    (its program and context, its weight and activation buffers); ``run`` a round; ``table``: its head's tokens."""

    CONFIG, IMAGE = DSparkConfig, DSparkImage
    NORMED = frozenset({"ikv"})  # (g's x: the injection's slots vary; the block's inputs are of 7 slots always)
    AHEAD = False

    def __init__(self, target: SparkOverlay, gguf: GGUFFile):
        cfg = DSparkConfig.from_gguf(gguf)
        tc = target.cfg
        if ((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim) != (tc.hidden_size, tc.num_attention_heads,
                                                                     tc.num_key_value_heads, tc.head_dim)
                or cfg.vocab_size != tc.vocab_size or max(cfg.taps) >= tc.num_hidden_layers):
            raise ValueError(f"{gguf.path}: the drafter of another model than {target.gguf.path}")
        if not target.image.draft_cols:
            raise ValueError("the model's LM rows without the drafter's head: SparkOverlay(**dspark.needs(gguf))")
        image = DSparkImage(gguf, cfg)
        self.head = HeadView(target.image, target.image.draft_cols)
        image.head = self.head.region(-1, "lm")
        image.vocab = self.head.vocab
        ModelOverlay.__init__(self, gguf, layers=list(range(cfg.layers)), vocab=None, weights=None, image=image)
        self.target, self.op, self.ash = target, target.op, target.ash
        if not image.formats() <= set(self.op.rows) or (image.gu_rows() != {16} and not self.op.gu_win):
            raise ValueError("the model's design lacks the drafter's tiles: SparkOverlay(**dspark.needs(gguf))")
        self.B = cfg.block
        assert self.B <= self.SLOTS
        self.max_positions = target.max_positions + self.B
        self.nr = cfg.window // attn.POS_PER_GROUP + RING_PAD
        self.period = 1
        self.attn_steps = AttentionSteps(self)
        self.table = HeadTable(self.head, tc.vocab_size, lambda: target.dev)
        self._rounds: dict[tuple[int, bool], Round] = {}
        for L in cfg.taps:  # (a tap: one K group, the fc phases' own)
            assert len(target.inp[(L, "attn_in")]) == 1
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def layer_kind(self, L: int) -> str:
        return "swa"

    @property
    def window(self) -> int:
        """(a block's: the positions before its anchor)"""
        return self.cfg.window

    def ring_groups(self, L: int) -> int:
        return self.nr

    def _flags(self, L: int, name: str) -> int:
        if name == "fc":  # (tap L's: the last one emits the sum)
            return F_NORM | F_RESID | (F_EMITR if L == len(self.cfg.taps) - 1 else 0)
        return FLAGS[name]

    def _sp(self, S: int, phase: tuple[int, str] | None = None) -> int:
        """(o's: the attention's rows of whole octets)"""
        return self.SLOTS if phase is not None and phase[1] in ("attn_out", "sink") else packed_slots(S)

    def _add_meta(self, phase: tuple[int, str]) -> None:
        """A phase that reads another's x (one K group): its metadata object only."""
        r = self.image.region(*phase)
        (k0, g), = mp.group_chunks(r)
        self.inp[phase] = [InputRegion(self._take("ACT", XB), k0 * r.kc, g * r.kc)]

    def _plan(self) -> None:
        """Byte offsets in ACT, NORM, KV."""
        cfg, ash, T = self.cfg, self.ash, self.T
        self._layout = dict.fromkeys(("ACT", "NORM", "KV"), 0)
        take = self._take
        self.loadr = take("ACT", COLS * self.SLOTS * XB)  # (the block's residual slices: [column][slot])
        self.zero = take("ACT", COLS * self.SLOTS * XB)  # (the injection's: zero)
        for i in range(len(cfg.taps)):
            self._add_meta((i, "fc"))
            tg = self.target.inp[(cfg.taps[i], "attn_in")][0]
            assert self.inp[(i, "fc")][0].K == tg.K, "fc's K group: the tap's x as the model's attn_in reads it"
        self._add_inputs((0, "ikv"))
        for L in self.layers[1:]:
            self._add_meta((L, "ikv"))
        for L in self.layers:
            for name in ("attn_in", "attn_out", "gate_up", "down"):
                self._add_inputs((L, name))
        self._add_inputs((-1, "lm"))
        K = cfg.heads * cfg.head_dim
        sink = take("ACT", region_bytes(K, self.SLOTS))  # (the injection's o)
        for L in self.layers:
            self.inp[(L, "sink")] = [InputRegion(sink, 0, K)]
        self.n_norm = -(-self.H // (NX_PER_NORM * OBJ_K))
        nw = 2 * NX_PER_NORM * OBJ_K * (self.n_norm - 1) + T
        keys = [("fc", i) for i in range(len(cfg.taps))] + ["enc", "output"]
        keys += [(L, n) for L in self.layers for n in ("attn_norm", "ffn_norm")]
        self.norm_off = {k: take("NORM", nw) for k in keys}
        self.fatt, self.fatt2, self.rawkv, self.kv_off, self.ring_off = {}, {}, {}, {}, {}
        for L in self.layers:  # (as the model's: [column][F_ATT] of the injection, [column][q objects]; the block's F_ATT)
            self.fatt[L] = take("ACT", COLS * (ash.u + self.SLOTS) * XB)
            self.fatt2[L] = take("ACT", COLS * ash.u * XB)
            self.rawkv[L] = take("ACT", self.SLOTS * XB)
            self.kv_off[L] = take("KV", ash.kvh * self.attn_steps.head_bytes(L))
        self.fobj = np.array([[[[self.fatt_object(L, c, u, b) // 4 for u in range(ash.u)] for c in range(COLS)]
                               for L in self.fatt] for b in range(2)], np.int64)
        r0 = self._layout["KV"]
        for L in self.layers:
            self.ring_off[L] = take("KV", ash.kvh * 2 * T)
        self.rings = (r0, self._layout["KV"] - r0)
        take("KV", attn.OBJS * T)
        self._sizes()
        rg = self.inp[(self.layers[0], "ikv")][0]  # (``_clear_pads``: g's x alone, the later layers' ikv read it)
        self._pads = [(rg.off, self.H, rg.K)] if rg.K > self.H else []

    def _norm_key(self, L: int, name: str) -> object:
        if name == "fc":
            return ("fc", L)
        return {"ikv": "enc", "lm": "output", "attn_in": (L, "attn_norm"), "gate_up": (L, "ffn_norm")}[name]

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, *, threads: int = 0) -> None:
        """The buffers on the model's device and context, the image packed, the model's weight and activation buffers
        adopted ("MW", "MACT"), the static objects written."""
        img, tgt = self.image, self.target
        self.dev = dev = OverlayDevice(self.op, device=tgt.dev.device)
        dev.alloc_all({"W": img.nbytes, **self.sizes})
        if img.pack(dev.view("W"), threads=threads):
            dev.flush("W")
        dev.adopt("MW", tgt.dev.buffers["W"])
        dev.adopt("MACT", tgt.dev.buffers["ACT"])
        act = dev.view("ACT")
        for base in (self.loadr, self.zero):
            for c in range(COLS):
                for s in range(self.SLOTS):
                    act[base + (c * self.SLOTS + s) * XB:][:4 * META] = mp.loadr_meta(s).view(np.uint8)
        self._load()
        dev.flush("ACT")
        if (store := self.op.page_store()) is not None:
            dev.set_prologue(self.op.page_prologue(store))
        self._w_out = self._f32("output_norm.weight")
        self.gguf.release()
        self.reset()

    def _load(self) -> None:
        """The norm weights (NORM), the F_ATT objects (sliding-window layers)."""
        norm, cfg = self.dev.view("NORM"), self.cfg
        fc_norm = self._f32("fc_norm.weight").reshape(len(cfg.taps), self.H)
        for key, off in self.norm_off.items():
            if isinstance(key, tuple) and key[0] == "fc":
                w = fc_norm[key[1]]
            elif isinstance(key, tuple):
                w = self._f32(f"blk.{key[0]}.{key[1]}.weight")
            else:
                w = self._f32({"enc": "enc.output_norm.weight", "output": "output_norm.weight"}[key])
            norm[off:off + 2 * self.H] = w.astype(bfloat16).view(np.uint8)
        self.dev.flush("NORM")
        act = self.dev.view("ACT")
        m = np.zeros(META, np.int32)
        m[M_FLAGS], m[M_KIND], m[M_WREC] = F_ATT, KIND_SWA, -1
        for L in self.fatt:
            for b in range(2):
                for c in range(COLS):
                    o = self.fatt_object(L, c, 0, b)
                    act[o:o + 4 * META] = m.view(np.uint8)

    # ------------------------------------------------------------------
    # The transaction of a round
    # ------------------------------------------------------------------

    def _meta_writes(self, S: int) -> list[tuple[int, np.ndarray]]:
        """(the injection's phases: S slots; the block's: B)"""
        out = []
        for (L, name), regs in self.inp.items():
            if name == "sink":
                continue
            s = S if name in ("fc", "ikv") else self.B
            out += [(rg.off, gr.meta.view(np.uint8)) for rg, gr in zip(regs, self._groups(L, name, s), strict=True)]
        return out

    def _weights(self, seq: Sequence, phase: tuple[int, str], gi: int) -> None:
        """The A fills of a phase's K group gi: its norm-weight objects (the first group of a normed phase), its tiles
        (the head's: the model's buffer)."""
        dev, (L, name) = self.dev, phase
        region, flags = self.image.region(L, name), self._flags(L, name)
        gr = mp.groups(region, flags, 1)[gi]
        buf = "MW" if name == "lm" else "W"
        for c in range(COLS):
            if gi == 0 and flags & F_NORM:
                seq.fill("A", c, dev.address("NORM", self.norm_off[self._norm_key(L, name)]), self.n_norm * self.T,
                         (self.T // 16, 4, self.T // 16, NX_PER_NORM * OBJ_K // 2))
            for o, n, dims, rep, it in mp.a_reads(region, c, gr):
                seq.fill("A", c, dev.address(buf, o), n, dims, rep, it)

    def _inputs(self, seq: Sequence, rg: InputRegion, S: int, phase: tuple[int, str] | None = None) -> None:
        """(a later layer's ikv: its metadata object, then layer 0's x)"""
        if phase is not None and phase[1] == "ikv" and phase[0] != self.layers[0]:
            nbytes, dims = input_fill(rg.K, self._sp(S, phase), self.SLOTS)
            x = self.inp[(self.layers[0], "ikv")][0].off
            seq.fill("X", BC_COL, self.dev.address("ACT", rg.off), XB, packet=BC_ID)
            seq.fill("X", BC_COL, self.dev.address("ACT", x + XB - 4 * META), nbytes - XB, dims, packet=BC_ID)
            return
        super()._inputs(seq, rg, S, phase)

    def _inject(self, seq: Sequence, st: AttnStep) -> None:
        """fc over the S rows' taps (a phase per tap, normed; into the zero residual), g -> ikv, per layer K, V and the
        records."""
        dev, S, cfg = self.dev, st.S, self.cfg
        sp = self._sp(S)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.zero + c * self.SLOTS * XB), S * XB)
        for i, tap in enumerate(cfg.taps):
            phase = (i, "fc")
            rg = self.inp[phase][0]
            self._weights(seq, phase, 0)
            nbytes, dims = input_fill(rg.K, sp, self.SLOTS)
            x = self.target.tap_x(tap) - 4 * META
            for c in range(COLS):  # (the dispatch's first phases: per column, after local objects)
                seq.fill("X", c, dev.address("ACT", rg.off), XB)
                seq.fill("X", c, dev.address("MACT", x), nbytes - XB, dims)
        h8 = self.H // COLS
        for c in range(COLS):
            self._values_to(seq, c, (self.layers[0], "ikv"), e0=h8 * c, n=h8, S=S, whole=True)
        for L in self.layers:
            self.attn_steps.build(seq, st, L, S, phase="ikv", out="sink")

    def _block(self, seq: Sequence, st: AttnStep) -> None:
        """The block's rows through the layers and the head."""
        dev, B = self.dev, st.S
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * self.SLOTS * XB), B * XB)
        for i, L in enumerate(self.layers):
            self.attn_steps.build(seq, st, L, B)
            self._phase(seq, (L, "attn_out"), B, (L, "gate_up"))
            self._phase(seq, (L, "gate_up"), B, (L, "down"))
            nxt = (self.layers[i + 1], "attn_in") if i + 1 < len(self.layers) else (-1, "lm")
            self._phase(seq, (L, "down"), B, nxt)
        self._phase(seq, (-1, "lm"), B, "LOGITS")

    def _round(self, S: int, block: bool) -> Round:
        """The transaction of a round (built at its first use)."""
        key = (S, block)
        if key not in self._rounds:
            seq = Sequence(self.op, inputs=("X",))
            inj = AttnStep(S, ((0, S, 0),), b=0, mode="insert") if S else None
            blk = AttnStep(self.B, ((0, self.B, 0),), b=1, mode="block") if block else None
            if inj is not None:
                self._inject(seq, inj)
            if blk is not None:
                self._block(seq, blk)
            seq.sync(fills=True)
            tr = Transaction(self.dev, seq.words())
            for st in (inj, blk):
                if st is not None:
                    st.tr = tr
                    st.patch = self.attn_steps.patch_tables(st)
            self._rounds[key] = Round(tr, inj, blk)
        return self._rounds[key]

    def _angles(self, p: int, n: int) -> dict[str, np.ndarray]:
        """The F_ATT objects' payload of positions p .. p + n - 1 (bf16 [cos | sin] of all pairs)."""
        return {"swa": cos_sin_rows(self.cfg.rope_theta(np.arange(p, p + n))).astype(bfloat16).view(np.uint8)}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def block_rows(self, anchor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The block's inputs (B, hidden) fp32 of the anchor's and the mask token's embeddings (hidden each)."""
        return np.stack([anchor, *[mask] * (self.B - 1)]).astype(np.float32)

    def run(self, S: int, q: int, x: np.ndarray | None, P: int, *, meanwhile: Callable[[], None] | None = None
            ) -> None:
        """One round's dispatch: the injection of the model's last step (its S slots at q .., whose taps its ACT holds;
        0: none) and, with x (B, hidden) the block's inputs (``block_rows``), the block at P (its logits and h:
        ``logits``, ``h``). ``meanwhile``: host work while it runs."""
        rnd = self._round(S, x is not None)
        s = S or self._metas_s[0] or 1  # (no injection: its phases' objects as they are)
        if self._metas_s != (s, self.start):
            self._metas(s)
        if S:
            self._clear_pads(self._sp(S))
        if x is not None:
            x = np.asarray(x, np.float32).reshape(self.B, self.H)
            self._write_loadr(x)
            first = (self.layers[0], "attn_in")
            self._write_input(self.inp[first][0], interleave(x, self._sp(self.B, first)).view(np.uint8).reshape(-1))
        if rnd.inj is not None:
            self.attn_steps.set_position(rnd.inj, q, self._angles(q, S))
            for L in self.layers:
                self.vring.ran(L, q, S)
        if rnd.blk is not None:
            self.attn_steps.set_position(rnd.blk, P, self._angles(P, self.B))
            for L in self.layers:
                self.vring.ran(L, P, self.B)
        self.dev.submit(rnd.tr)
        try:
            if meanwhile is not None:
                meanwhile()
        finally:
            self.dev.wait()

    def logits(self) -> np.ndarray:
        """The last block's logits (B, head rows) bf16 (the tokens ``table.ids``; a view of the NPU's buffer)."""
        return self._npu_logits(self.B)

    def h(self) -> np.ndarray:
        """The last block's rows normed by ``output_norm`` (B, hidden) fp32 (the confidence head's input)."""
        x = self._lm_x(self.B)
        r = 1 / np.sqrt((x * x).mean(-1, keepdims=True, dtype=np.float32) + np.float32(self.cfg.rms_norm_eps))
        return x * r * self._w_out

    def _f32(self, name: str) -> np.ndarray:
        return np.asarray(self.gguf.tensor(name).dequantize(), np.float32).reshape(-1)
