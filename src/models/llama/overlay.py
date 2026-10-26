"""A Llama 3.x model on its decode overlay (``llama/decode.py``): buffers, the transaction of a step of S <= 4 token
slots, the step.

Per step (one dispatch; the weights stream once for all slots) the columns read the token objects (per sub-step its
positions and the cache's range, the RoPE tables) and the slots' residual slices (F_LOADR: the embeddings, fp32), then
per layer L (column c = KV head c):

- qkv: the input norm, the rows of the column's query heads and of K_c, V_c -> the KV head's attention region [q
  objects SLOTS | raw K / V objects SLOTS] (per slot [head nh][hd] and [K hd | V hd] bf16);
- att: the attention cores (``decode.attn``) on the layer's cache, per sub-step (``subs``: its q and K / V objects,
  the first q object its F_ATT object) -> o into o's packed input, the new positions' records into the cache;
- o: into the residual (F_RESID), its bf16 copy out (F_EMITR) as gate_up's input; gate_up: ffn_norm, silu(gate) up ->
  down's input (its K groups); down: into the residual, its copy out as the next layer's qkv input;

and the LM head: output_norm -> LOGITS [slot][vocab] bf16 (the vocabulary padded to whole C objects per column: the
padding rows' logits are cut off, ``ModelOverlay._logits``).

The KV caches (KV): per layer and KV head linear groups of kp + 1 objects of the tile size T (``decode.kv``: kp K
objects, the V object; kp = 2, K as bfp16 hi + lo, or 1, single bfp16: ``kv_k``), 128 positions per group at
head_dim 64, 64 at 128 for objects of ~9.3 KB; then per layer and KV head a ring of two slots of T bytes with the
exact V values of the open records (``decode.vring``). Phase inputs are packed (``decode.design.packed_slots``).

The step's transaction is built once per S and pattern of sub-steps (and LM head or not) and patched per step: the
token objects, each column's range of each layer's cache per sub-step (A: address, length), its tail objects (A: the
ring slots of the sub-step's open records, or after a prefill the V panels of the first: record mode) and the record
drains (D: address, length). A column's weights between two attentions are one run of the (column-major) image: one A
task.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from artlite.decode import gemv, kv
from artlite.decode.attn import (
    F_ATT,
    M_AHD,
    M_AKIND,
    M_ANH,
    M_AOWN,
    M_ATOK,
    M_ATREC,
    M_AWIN,
    M_POS,
    M_QC_HI,
    M_QC_LO,
    M_TOK_A,
    M_TOK_G0,
    M_TOK_NA,
    SMAX,
    Attn,
    U,
    rope_tables,
    score_bits,
    substeps,
    tails,
)
from artlite.decode.design import (
    C_LEN,
    F_EMITR,
    F_GU,
    F_NORM,
    F_RESID,
    F_TOK,
    M_FLAGS,
    M_NASKIP,
    M_NS,
    M_SKIP,
    M_TOK_KIND,
    M_TOK_NG,
    META,
    SLOTS,
    XB,
    interleave,
    packed_slots,
)
from artlite.decode.model import ModelOverlay, paged_formats
from artlite.gguf_model import GGUFFile
from artlite.model_config import LlamaConfig
from artlite.operators.design import COLS
from artlite.overlay import Sequence, Transaction, WeightBuffer

from .decode import LlamaDecodeOperator
from .image import LlamaImage

PHASES = ("qkv", "att", "o", "gate_up", "down")
FLAGS = {"qkv": F_NORM, "o": F_RESID | F_EMITR, "gate_up": F_NORM | F_GU, "down": F_RESID | F_EMITR, "lm": F_NORM}
KV_K = {"single": 1, "hilo": 2}  # K precision of the caches (``kv_k``): K panels per record
KV_PAD = 2048  # positions past the caches (the last layer's): the prefill's sub-chunk slices of a cache


def check_formats(image: LlamaImage) -> set[str]:
    """The image's tile formats; NotImplementedError for those the decode does not read (yet)."""
    fmts = {b.fmt for r in image.regions for b in r.blocks}
    missing = sorted(f for f in fmts if not gemv.supported(f, image.nsb))
    if missing:
        raise NotImplementedError(f"{', '.join(missing)} tiles: the decode GEMV has no kernels for them")
    return fmts


@dataclass
class _Step:
    """The transaction of a step of S slots (its attentions' sub-steps: ``subs``) and its words that follow the
    position."""

    S: int
    subs: tuple[tuple[int, int], ...]  # (first slot, slots) per sub-step
    tr: Transaction | None = None
    tok: int = 0  # (its token objects' offset in ACT: one of ``toks``)
    b: int = 0  # (its buffer: ``ModelOverlay._transaction``)
    fills: list[tuple[int, int, int, int]] = field(default_factory=list)  # (BD word index, layer, column, sub-step)
    # (BD word index, layer, column, sub-step, tail object)
    tails: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    # (handle, layer, column, sub-step, i-th record)
    recs: list[tuple[list[int], int, int, int, int]] = field(default_factory=list)
    patch: dict[str, np.ndarray] | None = None  # (the lists as arrays: _patch_tables)


class LlamaOverlay(ModelOverlay):
    """A Llama GGUF model on its decode overlay. ``layers``: the layers to run (default: all; tests run slices),
    ``vocab``: LM rows (default: the padded vocabulary), ``max_positions``: the caches' capacity, ``kv_k``: the caches'
    K precision ("single": one bfp16 block per 8 dims, "hilo": hi + lo blocks), ``weights``: the image's buffer
    allocated ahead."""

    CONFIG, IMAGE = LlamaConfig, LlamaImage
    AHEAD = True

    def __init__(self, gguf: GGUFFile, *, layers: list[int] | None = None, vocab: int | None = None,
                 max_positions: int = 4096, kv_k: str = "single", weights: WeightBuffer | None = None):
        if kv_k not in KV_K:
            raise ValueError(f"kv_k: {kv_k!r} (one of {', '.join(KV_K)})")
        super().__init__(gguf, layers=layers, vocab=vocab, weights=weights)
        cfg, img = self.cfg, self.image
        fmts = check_formats(img)
        self.hd, self.nh, self.kp = cfg.head_dim, cfg.q_per_kv, KV_K[kv_k]
        self.at = at = Attn(hds=(self.hd,), rot=(self.hd // 2,), chain=1, heads=self.nh, kp=self.kp,
                            eps=cfg.rms_norm_eps, a_bytes=self.T)
        gu = {b.R for L in self.layers for b in img.region(L, "gate_up").blocks}
        self.op = LlamaDecodeOperator(hidden=self.H, rows={f: img.R[f] for f in sorted(fmts)}, tile_bytes=self.T,
                                      eps=cfg.rms_norm_eps, attn=at, gu_win=gu != {16}, paged=paged_formats(gguf, fmts))
        self.rec = kv.Records(self.hd, self.kp)
        self.pg = at.pg[self.hd]  # positions per cache group
        self.n_groups = -(-max_positions // self.pg)
        self.max_positions = self.n_groups * self.pg
        self.period = 16  # (the steps' sub-steps repeat with the record pairs)
        self._trec = 0  # (M_ATREC as the F_ATT objects of slot 0 hold it)
        self.inv_freq = cfg.rope_inv_freq()
        self._plan()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def phases(self, lm: bool = True) -> list[tuple[int, str]]:
        """The step's phases in order (``lm`` False: without the LM head)."""
        return [(L, n) for L in self.layers for n in PHASES] + ([(-1, "lm")] if lm else [])

    def _flags(self, L: int, name: str) -> int:
        return FLAGS[name]

    def _plan(self) -> None:
        """Byte offsets in ACT and KV."""
        self._layout = dict.fromkeys(("ACT", "KV"), 0)
        take = self._take
        # (token t: of the sub-step from slot t on; two copies, one per transaction buffer: ``AHEAD``)
        self.toks = (take("ACT", SMAX * XB), take("ACT", SMAX * XB))
        self.loadr = take("ACT", COLS * SLOTS * XB)
        for L, name in self.phases():
            if name != "att":
                self._add_inputs((L, name))
        # per layer and KV head [q objects SLOTS | raw K / V objects SLOTS]; q object a's metadata words: the F_ATT
        # object of the sub-step from slot a on (its token a)
        self.att_reg = {(L, c): take("ACT", 2 * SLOTS * XB) for L in self.layers for c in range(COLS)}
        self.kv_bytes = self.n_groups * self.rec.objects * self.T  # (a KV head's cache of a layer)
        self.kv_off = {L: take("KV", COLS * self.kv_bytes) for L in self.layers}
        self.ring_off = {L: take("KV", COLS * 2 * self.T) for L in self.layers}  # (per KV head: 2 slots)
        take("KV", 2 * KV_PAD // self.pg * self.rec.objects * self.T)  # (the prefill's slices past the last layer's)
        self._sizes()

    def kv_base(self, L: int, c: int) -> int:
        """Byte offset in KV of group 0 of KV head c's cache of layer L."""
        return self.kv_off[L] + c * self.kv_bytes

    def ring(self, L: int, c: int) -> int:
        """Byte offset in KV of KV head c's ring of layer L: 2 slots of T bytes, slot r & 1 the V values of record r
        ([d-block][position 8][8] bf16: ``decode.vring.ring_rows``)."""
        return self.ring_off[L] + 2 * c * self.T

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """The token objects' metadata words, the attention layers' objects'."""
        act = self.dev.view("ACT")
        for tok in self.toks:
            for t in range(SMAX):
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_TOK_KIND] = F_TOK, t
                act[tok + t * XB:][:4 * META] = m.view(np.uint8)
        for L in self.layers:
            self._load_attention(L)

    def _load_attention(self, L: int) -> None:
        """The metadata words of layer L's q and K / V objects: q object a's the F_ATT object of the sub-step from a on
        (static but M_ATREC: ``_tails``); K / V object a's M_NASKIP = -1 - a: after it the main core skips the cache
        objects of token a (those of the sub-step that ends at slot a, else none)."""
        act = self.dev.view("ACT")
        hi, lo = score_bits(1 / math.sqrt(self.hd))
        for c in range(COLS):
            for a in range(SLOTS):
                m = np.zeros(META, np.int32)
                m[M_FLAGS], m[M_NASKIP], m[M_SKIP] = F_ATT, 0, 0
                m[M_AHD], m[M_AKIND], m[M_AOWN] = self.hd, 0, 1
                m[M_QC_HI], m[M_QC_LO], m[M_AWIN] = hi, lo, 0
                m[M_ANH], m[M_ATOK] = self.nh, a
                o = self.att_reg[(L, c)] + a * XB
                act[o:o + 4 * META] = m.view(np.uint8)
                m = np.zeros(META, np.int32)
                m[M_NASKIP] = -1 - a
                o = self.att_reg[(L, c)] + (SLOTS + a) * XB
                act[o:o + 4 * META] = m.view(np.uint8)

    # ------------------------------------------------------------------
    # The transaction of a step
    # ------------------------------------------------------------------

    def _outputs(self, seq: Sequence, phase: tuple[int, str], S: int) -> None:
        """The C drains of a GEMV phase (after its last group)."""
        L, name = phase
        dev, h8, ffn = self.dev, self.H // COLS, self.cfg.intermediate_size // COLS
        nq = self.nh * self.hd
        for c in range(COLS):
            if name == "qkv":  # [q nh hd | K hd | V hd] per slot -> its q object, its raw K / V object
                reg = self.att_reg[(L, c)]
                seq.drain("C", c, dev.address("ACT", reg + 4 * META), S * 2 * nq, (C_LEN // 2, S, XB // 4, C_LEN // 2))
                seq.drain("C", c, dev.address("ACT", reg + SLOTS * XB + 4 * META), S * 4 * self.hd,
                          (C_LEN // 2, S, XB // 4, C_LEN // 2))
            elif name == "o":
                self._values_to(seq, c, (L, "gate_up"), e0=h8 * c, n=h8, S=S)
            elif name == "gate_up":
                self._values_to(seq, c, (L, "down"), e0=ffn * c, n=ffn, S=S)
            elif name == "down":
                i = self.layers.index(L)
                nxt = (self.layers[i + 1], "qkv") if i + 1 < len(self.layers) else (-1, "lm")
                self._values_to(seq, c, nxt, e0=h8 * c, n=h8, S=S)
            else:
                self._logits_out(seq, c, S)

    def _gemv(self, seq: Sequence, phase: tuple[int, str], S: int) -> None:
        for gi, rg in enumerate(self.inp[phase]):
            if gi == 0:
                seq.sync()  # (its inputs' producers)
            self._inputs(seq, rg, S)
        self._outputs(seq, phase, S)

    def _attention(self, seq: Sequence, st: _Step, L: int, weights: list[tuple[int, str]]) -> None:
        """The attention of layer L, per sub-step (each after the last one's records are in DDR): per column its cache
        range and tail objects (A, patched), the F_ATT object and the sub-step's q and K / V objects (X), the records
        (D, patched) and o (D, into the sub-step's slots of o's input). The weights of the GEMV phases after it go before
        the sync of its inputs: the A stream keeps streaming them while the attention waits for them."""
        dev, S, T = self.dev, st.S, self.T
        o, sp, nk = self.inp[(L, "o")][0], packed_slots(S), self.nh * self.hd // 8  # (o: the KV head's k blocks)
        for u, (a, n) in enumerate(st.subs):
            if u:
                seq.sync()  # (sub-step u - 1's records in DDR: this one reads them)
            for c in range(COLS):  # (its range of the cache and its tail objects, patched per step: ``_tails``)
                st.fills.append((seq.fill("A", c, dev.address("KV", self.kv_off[L]), T), L, c, u))
                st.tails += [(seq.fill("A", c, dev.address("KV", self.ring_off[L]), T), L, c, u, t)
                             for t in range(tails(n))]
            if u == len(st.subs) - 1 and weights:
                self._weight_run(seq, weights)
            if u == 0:
                seq.sync()  # (the q and K / V objects in DDR)
            for c in range(COLS):  # (its q objects, then its K / V objects: half objects, the second strip SLOTS XB on)
                reg = self.att_reg[(L, c)] + a * XB
                seq.fill("X", c, dev.address("ACT", reg), 2 * n * XB, (XB // 8, 2 * n, XB // 8, SLOTS * XB // 4))
                for i in range(len(kv.emit_order(p=0, S=n, hd=self.hd, a_bytes=T, kp=self.kp))):
                    hdl = seq.drain("D", c, dev.address("KV", self.kv_off[L]), 2 * self.at.d_len)
                    st.recs.append((hdl, L, c, u, i))
                at = dev.address("ACT", o.off + XB + 2 * sp * 8 * nk * c + 16 * a)  # ([k block][slot sp][8])
                seq.drain("D", c, at, 16 * n * nk, None if n == sp else (4 * n, nk, 4 * sp, 4 * sp * nk))

    def _build(self, S: int, lm: bool, subs: tuple) -> _Step:
        dev = self.dev
        seq = Sequence(self.op, rolling=True, inputs=("X",))
        st = _Step(S, subs, tok=self.toks[self._b])
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", st.tok), SMAX * XB)
        for c in range(COLS):
            seq.fill("X", c, dev.address("ACT", self.loadr + c * SLOTS * XB), S * XB)
        ph = self.phases(lm)

        def run(i: int) -> list[tuple[int, str]]:  # (the GEMV phases from i to the next attention: one A fill)
            return ph[i:next((k for k in range(i, len(ph)) if ph[k][1] == "att"), len(ph))]

        for i, (L, name) in enumerate(ph):
            if name == "att":
                self._attention(seq, st, L, run(i + 1))
                continue
            if i == 0:
                self._weight_run(seq, run(0))
            self._gemv(seq, (L, name), S)
        seq.sync(fills=True)
        st.tr = Transaction(dev, seq.words())
        return st

    def _lm_phase(self, seq: Sequence) -> None:
        self._weight_run(seq, [(-1, "lm")])
        self._gemv(seq, (-1, "lm"), 1)

    # ------------------------------------------------------------------
    # The per-step words
    # ------------------------------------------------------------------

    def subs(self, p: int, S: int) -> tuple[tuple[int, int], ...]:
        """The attention's sub-steps (first slot, slots) of a step at p .. p + S - 1: within a pair of records
        (``decode.attn``: P V adds two records together; a group holds whole pairs)."""
        out = tuple(substeps(p, S, 16, self.at.slots(self.hd)))
        assert len(out) <= U, (p, S)
        return out

    def _patch_tables(self, st: _Step) -> dict[str, np.ndarray]:
        """A step's cache fills, tail fills and record drains as arrays: BD word index, the sub-step, the tail object /
        the record's index, the byte offsets in KV of group 0 of the layer's KV head (and of its ring)."""
        f = np.array([(at, u, self.kv_base(L, c)) for at, L, c, u in st.fills], np.int64).reshape(-1, 3)
        t = np.array([(at, u, k, self.kv_base(L, c), self.ring(L, c)) for at, L, c, u, k in st.tails],
                     np.int64).reshape(-1, 5)
        r = np.array([(hdl[0], u, i, self.kv_base(L, c), self.ring(L, c)) for hdl, L, c, u, i in st.recs],
                     np.int64).reshape(-1, 5)
        return {"f_at": f[:, 0], "f_u": f[:, 1], "f_base": f[:, 2], "t_at": t[:, 0], "t_u": t[:, 1], "t_k": t[:, 2],
                "t_base": t[:, 3], "t_ring": t[:, 4], "r_at": r[:, 0], "r_u": r[:, 1], "r_i": r[:, 2],
                "r_base": r[:, 3:5]}

    def _tails(self, st: _Step, p: int, pu: np.ndarray, ahead: bool) -> bool:
        """The step's tail fills (per sub-step and column: the ring slots of the records of its first and last
        positions, or after a prefill for the first sub-step's first the record's V panels) and the F_ATT objects'
        M_ATREC. pu: per sub-step (first position, slots). (Every step runs every layer: one ring state for all, key
        0 of ``vring``.) ``ahead``: False when M_ATREC changes (the running step reads those objects)."""
        t, T, m = st.patch, self.T, int(0 in self.vring.from_prefill)
        if ahead and self._trec != m:
            return False
        rec = (t["t_u"] == 0) & (t["t_k"] == 0) & bool(m)  # (record mode: the first sub-step's first tail)
        q0, n = pu[t["t_u"], 0], pu[t["t_u"], 1]
        vrec = np.array([kv.v_offset(int(x), hd=self.hd, a_bytes=T, kp=self.kp) for x in pu[:, 0]],
                        np.int64)[t["t_u"]]
        slot = np.where(t["t_k"] == 0, q0 >> 3, (q0 + n - 1) >> 3) & 1
        address = self.dev.address("KV") + np.where(rec, t["t_base"] + vrec, t["t_ring"] + slot * T)
        self._bd_words(st.tr.words, t["t_at"], address, np.full(len(rec), T, np.int64))
        if not m:
            self.vring.check(0, p, p)  # (the rows before p from the rings)
        if self._trec != m:  # (the F_ATT objects of the sub-steps from slot 0 on: the others' stay 0)
            act = self.dev.view("ACT")
            for L in self.layers:
                for c in range(COLS):
                    off = self.att_reg[(L, c)]
                    act[off:off + 4 * META].view(np.int32)[M_ATREC] = m
                    self.dev.flush("ACT", off, 4 * META)
            self._trec = m
        return True

    def _ran(self, p: int, S: int) -> None:
        """After a step: the rings hold the rows of its open records."""
        self.vring.ran(0, p, S)

    def _set_position(self, st: _Step, p: int, ahead: bool = False) -> bool:
        """The token objects (per sub-step: its positions, the cache's range, the step's RoPE tables; st's copy) and
        the step's cache transfers."""
        S, dev, T, pg = st.S, self.dev, self.T, self.pg
        assert 0 <= p and p + S <= self.max_positions, (p, S)
        assert st.subs == self.subs(p, S), (p, S, st.subs)
        act = dev.view("ACT")
        pu = np.zeros((U, 2), np.int64)  # (per sub-step: first position, slots)
        fo = np.zeros((U, 2), np.int64)  # (per sub-step: the A fill of the range, first object and objects)
        ro = np.zeros((U, 6, 3), np.int64)  # (per sub-step and i-th record: base, offset, bytes)
        cs = rope_tables(self.inv_freq, range(p, p + S), self.hd // 2).view(np.uint8).reshape(-1)
        for a in range(SLOTS):  # (each token object copies the step's table; one of no sub-step: no objects)
            o = st.tok + a * XB
            act[o + 4 * META:o + 4 * META + cs.size] = cs
            m = act[o:o + 4 * META].view(np.int32)
            m[M_NS], m[M_TOK_NG:M_TOK_NG + COLS], m[M_TOK_NA:M_TOK_NA + COLS] = 0, 0, 0
        for u, (a, n) in enumerate(st.subs):
            (g0, ng), = kv.ranges(first=0, g=(p + a) // pg, chain=1)
            o = st.tok + a * XB
            m = act[o:o + 4 * META].view(np.int32)
            me = act[o + (n - 1) * XB:][:4 * META].view(np.int32)  # (the token of its last slot)
            fo[u] = kv.fill(g0, ng, self.kp)
            no = int(fo[u, 1]) + tails(n)
            m[M_NS], m[M_POS], m[M_TOK_A] = n, p + a, a
            m[M_TOK_G0:M_TOK_G0 + COLS] = g0 * pg
            m[M_TOK_NA:M_TOK_NA + COLS] = self.rec.objects * ng | no << 16
            me[M_TOK_NG:M_TOK_NG + COLS] = no
            pu[u] = p + a, n
            rec = kv.records(p=p + a, S=n, hd=self.hd, a_bytes=T, kp=self.kp)
            ro[u, :len(rec)] = rec
        dev.flush("ACT", st.tok, SMAX * XB)
        if st.patch is None:
            st.patch = self._patch_tables(st)
        t, kv0 = st.patch, dev.address("KV")
        f = fo[t["f_u"]]
        self._bd_words(st.tr.words, t["f_at"], kv0 + t["f_base"] + f[:, 0] * T, f[:, 1] * T)
        if not self._tails(st, p, pu, ahead):
            return False
        r = ro[t["r_u"], t["r_i"]]  # (the record drains: from the KV head's group 0 or ring, offset, bytes)
        base = np.take_along_axis(t["r_base"], r[:, :1], 1)[:, 0]
        self._bd_words(st.tr.words, t["r_at"], kv0 + base + r[:, 1], r[:, 2])
        st.tr.flush()
        return True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def step(self, x: torch.Tensor, position: int, *, logits: bool = True) -> torch.Tensor | None:
        """One step of S = len(x) <= 4 token slots: x (S, hidden) the embeddings at positions position .. position + S
        - 1 (the positions before are cached). Returns the logits (S, vocab) bf16 (a view of the NPU's buffer, valid
        until the next step), or without ``logits`` None (only the caches are updated)."""
        x = x.reshape(-1, self.H).float()
        S = x.shape[0]
        assert 1 <= S <= SLOTS
        if self._metas_s != (S, self.start):
            self._metas(S)
        xf = x.numpy()
        self._write_loadr(xf)
        self._write_input(self.inp[(self.layers[0], "qkv")][0], interleave(xf, packed_slots(S)).view(np.uint8).reshape(-1))
        return self._run(S, position, logits)

    def read_kv(self, layer: int, positions: list[int] | range) -> tuple[torch.Tensor, torch.Tensor]:
        """K, V (kv heads, n, head_dim) fp32 as cached for the positions: K rotated, V as it is."""
        pos = np.asarray(positions)
        n, G, T = int(pos.max()) // self.pg + 1, self.rec.objects, self.T  # (the groups that hold the positions)
        self.dev.fetch("KV", self.kv_off[layer], COLS * self.kv_bytes)
        ks, vs = [], []
        for c in range(COLS):
            b = self.kv_base(layer, c)
            k, v = kv.kv_from_groups(self.dev.view("KV")[b:b + n * G * T].reshape(n, G, T), self.hd)
            ks.append(k[pos])
            vs.append(v[pos])
        return torch.from_numpy(np.stack(ks)), torch.from_numpy(np.stack(vs))

    def hidden(self, L: int = -1, slot: int = 0) -> torch.Tensor:
        """The residual stream that layer L's qkv phase (-1: the LM head) read in the last step, of a slot (its bf16
        copy, as fp32)."""
        rg, sp = self.inp[(L, "lm" if L < 0 else "qkv")][0], packed_slots(self._last_s)
        self.dev.fetch("ACT", rg.off + XB, 2 * sp * self.H)
        v = self.dev.view("ACT")[rg.off + XB:rg.off + XB + 2 * sp * self.H].view(np.uint16).reshape(-1, sp, 8)[:, slot]
        return torch.from_numpy(v.reshape(-1).copy()).view(torch.bfloat16).float()

    def attn_out(self, L: int, slot: int = 0) -> torch.Tensor:
        """The attention's output o (heads hd) of layer L in the last step, of a slot (bf16 values, fp32)."""
        rg, sp = self.inp[(L, "o")][0], packed_slots(self._last_s)
        self.dev.fetch("ACT", rg.off + XB, 2 * sp * rg.K)
        v = self.dev.view("ACT")[rg.off + XB:rg.off + XB + 2 * sp * rg.K].view(np.uint16).reshape(-1, sp, 8)[:, slot]
        return torch.from_numpy(v.reshape(-1).copy()).view(torch.bfloat16).float()


__all__ = ["KV_K", "KV_PAD", "LlamaOverlay", "check_formats"]
