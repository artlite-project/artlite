"""Inkling's attention prep (``InkPrepOperator``): a chunk of T tokens of one visit, from the q / k / v GEMMs' outputs
(``prefill.gemm``'s layouts of ``models.inkling2.image``'s regions) to the Inkling attention's operands
(``AttnOperator(ink=True)``) and the visit's K / V records in the decode's cache, so the decode continues from them.

Per column c (KV head c, query heads 3 c .. 3 c + 2), core (c, 2) the queries, (c, 4) K, (c, 5) V:

- Q: per 8 tokens and head q times the head's clip scales, L2-clipped to sqrt(512), times tau (full layers: the
  position's, ``InklingConfig.tau``, from the TAU table) and the score constant (log2 units: 1 / sqrt(hd) / ln 2) ->
  the head's Q unit (hi + lo panels); r times its clip scales, clipped to 8, times tau and log2 e -> its R panels (hi, lo:
  the relative bias's operands); the sink logits times log2 e;
- K: the raw K rows (the k GEMM's) through the causal conv of 4 taps (the 3 positions before p0 from the visit's raw
  ring, the decode's: slot = position % 8, p0 % 8 = 0; full layers: plus each token's tap deltas), / sqrt(5) (full: / 3),
  L2-clipped to sqrt(512) -> the cache's K records; V likewise, RMS-normed -> the V records. Each keeps the raw rows of
  the last 8 valid positions (``n``: the chunk's valid tokens, ``P_N``; padding tokens' records are junk that later
  tokens overwrite) and writes them back as the ring at the chunk's end.

QS per column and token group: ``QS_UNITS`` units of the attention's stream B: the 3 heads' Q units, then a unit of
INK rows (rows of 72 bf16: per head R hi, R lo, then the sinks [head][token] fp32; the attention's cores run row blocks
in pairs, a query block's fourth row block zero: its Q units' slot holds them).

The PRM argument per column (``prm``): its KV head's K and V taps [4][hd], its query heads' q clip [3][hd] and r clip
[3][16] scales (bf16). The K and V cores take [old ring K | K taps | old ring V | V taps] once per chunk (stream B,
split by the memtile), the Q core the clips with each token group's rows (the memtile's six output channels: three
for A, two for B, one for the records' join). A prep item takes at most 512 tokens (64 token groups: a chained fill's
repeats; its fills and drains use 15 of a shim tile's 16 BDs). The ring object at the end goes through the records' join as one more group object, its
ring parts into RING (strided: [slot][K | V]), the rest into JUNK."""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ...operators import templating
from ...operators.base import ArgSpec, ArtcOperator
from ...operators.chain import ChainPart, chained_fill
from ...operators.design import COLS, MAX_BD_ITER
from ...operators.kernel_source import kernel_prefix
from ...operators.util import Attr, regions
from ..gemm import Layout
from .shape import P_KV, P_KVR, Shape

_PREP = templating.load("artlite/prefill/attn/ink_prep.cc.jinja")

P_TAU = "pi_tau"  # bf16 offset of the chunk's first position in the TAU table (2 p0)
P_N = "pi_n"  # the chunk's valid tokens
HEADS = 3  # query heads per column (per KV head)
QS_UNITS = 4  # stream B units per token group and column: 3 Q units, 1 of INK rows
INK_ROWS = 7  # (rows of 72 bf16: per head R hi, R lo; the sinks)
RING_BYTES = 9216  # a raw ring's bytes in the decode's KV buffer ([slot 8][K | V] fp32, then 1 KB unused)


def ink_shape(a_bytes: int, kp: int = 2) -> Shape:
    """The Inkling attention's shape: head_dim 128, 4 row blocks a query block (3 heads and a zero one), hi + lo Q."""
    return Shape(128, 4, kp, a_bytes, qp=2, pp=1, per=1)


class InkPrepOperator(ArtcOperator):
    """The prep of T tokens of a visit (module docstring). Runtime sequence: GQ, GK, GV (the q, k, v GEMMs' outputs in
    their layouts ``lq``, ``lk``, ``lv``), TAU (per position tau fp32, as bf16 pairs; ``tau_positions`` of them), PRM
    (``prm``), QS (out), KV (the visit's cache: COLS KV heads of ``kv_bytes``, ``kv_pad`` past them), RING (the visit's
    raw rings, ``RING_BYTES`` a KV head: read, then written), JUNK (scratch: the ring object's other bytes).
    Parameters ``params``. ``full``: a full-attention layer (tap deltas, tau, the conv's 1 / 3); ``kv0``: bytes of a KV
    head's cache before its group 0 (a sliding-window cache's guard); ``wrap``: the cache a ring of that many groups."""

    STACK = 1024
    Q_STACK = 1536  # (the Q core's: the q kernel's frame and the clip's)
    pfx = kernel_prefix("pik")

    def __init__(self, *, T: int, shape: Shape, lq: Layout, lk: Layout, lv: Layout, full: bool, kv_bytes: int,
                 tau_positions: int, kv_pad: int = 0, eps: float = 1e-6, kv0: int = 0, wrap: int = 0):
        s = shape
        assert s.hd == 128 and s.qh == 4 and s.qp == 2 and s.per == 1 and T % s.pg == 0 and T // 8 <= MAX_BD_ITER
        assert {"q", "r", "sink"} <= set(lq.off) and "k" in lk.off and "v" in lv.off
        assert (lk.off.get("kd", 128) == 128 and lv.off.get("vd", 128) == 128) and full == ("kd" in lk.off)
        self.T, self.shape, self.lq, self.lk, self.lv, self.full = T, s, lq, lk, lv, full
        self.kv_bytes, self.tau_positions, self.kv_pad, self.eps, self.kv0, self.wrap = \
            kv_bytes, tau_positions, kv_pad, eps, kv0, wrap
        self.nq = -(-(lq.off["sink"] + 2 * HEADS) // 32) * 32  # (a Q row's columns read: q .. the sinks, whole 64 B)
        assert self.nq <= lq.pitch and lq.off["q"] == 0
        self.pk = -(-(128 + (8 if full else 0)) // 32) * 32  # (a K / V row's: the values, the deltas' fp32 pairs)
        assert self.pk <= lk.pitch and self.pk <= lv.pitch
        super().__init__()

    @property
    def unit(self) -> int:
        """bf16 of a stream B unit."""
        return self.shape.unit // 2

    @property
    def tg_qs(self) -> int:
        """bf16 of a column's token group in QS."""
        return QS_UNITS * self.unit

    def workspaces(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        hd = self.shape.hd
        q = regions([("row", 4 * hd), ("stage", 32 * hd), ("rst", 512)])
        k = regions([("row", 4 * hd), ("stage", 32 * hd), ("taps", 8 * hd), ("hist", 32 * hd), ("st", 64)])
        v = regions([("row", 4 * hd), ("vb", 16 * hd), ("taps", 8 * hd), ("hist", 32 * hd), ("st", 64)])
        return q, k, v

    def get_kernel_src(self) -> str:
        s = self.shape
        wq, wk, wv = (Attr(w) for w in self.workspaces())
        return _PREP.render(p=self.pfx, HD=s.hd, ND=s.nd, KP=s.kp, KREC=s.rec.k_bytes, QUNIT=s.qunit, NQ=self.nq,
                            PK=self.pk, OFF_R=self.lq.off["r"], OFF_S=self.lq.off["sink"], FULL=self.full,
                            SCALE=repr(float(np.float32(s.scale))), CSC=repr(float(np.float32(1 / 3 if self.full else 5 ** -0.5))),
                            eps=repr(self.eps), wq=wq, wk=wk, wv=wv)

    # ------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------

    @property
    def prm_w(self) -> int:
        """bf16 of a column's PRM part: K taps, V taps [4][hd], q clip [3][hd], r clip [3][16] (then padding to 32)."""
        return 2 * 4 * 128 + 3 * 128 + 64

    def prm(self, qclip: np.ndarray, rclip: np.ndarray, ktaps: np.ndarray, vtaps: np.ndarray) -> np.ndarray:
        """The PRM argument (uint8) of a layer's q clip (heads, hd), r clip (heads, 16), K / V taps (4, KV heads x hd)."""
        out = np.zeros((COLS, self.prm_w), bfloat16)
        for c in range(COLS):
            out[c, :512] = ktaps[:, 128 * c:128 * (c + 1)].reshape(-1)
            out[c, 512:1024] = vtaps[:, 128 * c:128 * (c + 1)].reshape(-1)
            out[c, 1024:1408] = qclip[HEADS * c:HEADS * (c + 1)].reshape(-1)
            out[c, 1408:1456] = rclip[HEADS * c:HEADS * (c + 1)].reshape(-1)
        return out.view(np.uint8).reshape(-1)

    @property
    def n_qs(self) -> int:
        return COLS * self.T // 8 * self.tg_qs

    @property
    def n_kv(self) -> int:
        return (self.shape.kvh * self.kv_bytes + self.kv_pad) // 2

    @property
    def n_junk(self) -> int:
        """bf16 of JUNK: the ring object's K part past its ring rows."""
        return self.shape.kp * self.shape.used // 2 - 2048

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (T * self.lq.N,), bfloat16), ArgSpec("in", (T * self.lk.N,), bfloat16),
                ArgSpec("in", (T * self.lv.N,), bfloat16), ArgSpec("in", (2 * self.tau_positions,), bfloat16),
                ArgSpec("in", (COLS * self.prm_w,), bfloat16), ArgSpec("out", (self.n_qs,), bfloat16),
                ArgSpec("out", (self.n_kv,), bfloat16), ArgSpec("out", (COLS * RING_BYTES // 2,), bfloat16),
                ArgSpec("out", (self.n_junk,), bfloat16)]

    def params(self, p0: int, n: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 (a group start) of n valid tokens."""
        s = self.shape
        assert p0 % s.pg == 0 and 1 <= n <= self.T and p0 + self.T <= self.tau_positions
        g = p0 // s.pg
        if self.wrap:
            assert g % self.wrap + self.T // s.pg <= self.wrap, (p0, self.T, "a chunk across the ring's wrap")
            kv = {P_KVR: s.group // 2 * (g % self.wrap)}
        else:
            kv = {P_KV: s.group // 2 * g}
        return kv | {P_TAU: 2 * p0, P_N: n}

    # ------------------------------------------------------------------
    # Design
    # ------------------------------------------------------------------

    def get_mlir_artifact(self) -> Module:
        p, T, s = self.pfx, self.T, self.shape
        nq, pk = self.nq, self.pk
        bf = np.dtype[bfloat16]
        wq, wk, wv = self.workspaces()
        qa_ty = np.ndarray[(8 * nq + 448 + 16,), bf]  # 8 Q rows, the clips, their tau
        kva_ty = np.ndarray[(8 * pk,), bf]  # 8 K or V rows
        kvb_ty = np.ndarray[(2048 + 512,), bf]  # the old ring's rows, the taps
        qo_ty = np.ndarray[(3 * self.unit + INK_ROWS * 72,), bf]
        kg_ty = np.ndarray[(s.kp * s.used // 2,), bf]
        vg_ty = np.ndarray[(s.used // 2,), bf]
        wsq, wsk, wsv = (np.ndarray[(w["size"] // 4,), np.dtype[np.float32]] for w in (wq, wk, wv))
        k_q = Kernel(f"{p}_q", "kernel.o", [qa_ty, wsq, qo_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [kva_ty, wsk, kg_ty, np.int32, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [kva_ty, wsv, vg_ty, np.int32, np.int32])
        k_kprm = Kernel(f"{p}_kprm", "kernel.o", [kvb_ty, wsk])
        k_vprm = Kernel(f"{p}_vprm", "kernel.o", [kvb_ty, wsv])
        k_kring = Kernel(f"{p}_kring", "kernel.o", [wsk, kg_ty])
        k_vring = Kernel(f"{p}_vring", "kernel.o", [wsv, vg_ty])
        p_tau, p_n = ScratchpadParameter(P_TAU, np.int32), ScratchpadParameter(P_N, np.int32)
        p_kv = ScratchpadParameter(P_KVR if self.wrap else P_KV, np.int32)
        n_tg, rg, n_grp = T // 8, s.rg, T // s.pg

        # per column and token group [Q rows, clips, tau | K rows | V rows], split in the memtile
        sizes = [8 * nq + 448 + 16, 8 * pk, 8 * pk]
        offs = [sum(sizes[:i]) for i in range(3)]
        a3 = [ObjectFifo(np.ndarray[(sum(sizes),), bf], name=f"li_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split(offs, tile=Tile(c, 1), depths=[2] * 3, obj_types=[qa_ty, kva_ty, kva_ty],
                                 names=[f"li_a{c}_{i}" for i in range(3)]) for c in range(COLS)]
        # the chunk's constants [old ring K | K taps | old ring V | V taps], split
        b3 = [ObjectFifo(np.ndarray[(2 * 2560,), bf], name=f"li_b{c}", depth=1) for c in range(COLS)]
        b1 = [b3[c].cons().split([0, 2560], tile=Tile(c, 1), depths=[1] * 2, obj_types=[kvb_ty, kvb_ty],
                                 names=[f"li_b{c}_{i}" for i in range(2)]) for c in range(COLS)]
        o3 = [ObjectFifo(qo_ty, name=f"li_o{c}", depth=2) for c in range(COLS)]
        g3 = [ObjectFifo(np.ndarray[(s.objects * s.used // 2,), bf], name=f"li_g{c}", depth=2) for c in range(COLS)]
        g1 = [g3[c].prod().join([0, s.kp * s.used // 2], tile=Tile(c, 1), depths=[2, 2], obj_types=[kg_ty, vg_ty],
                                names=[f"li_k{c}", f"li_v{c}"]) for c in range(COLS)]

        def core_q(a_h, o_h, fn, ws):
            for _ in forever_():
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, ws, out)
                    o_h.release(1)
                    a_h.release(1)

        def core_kv(a_h, b_h, o_h, fn, fprm, fring, ws, pn):
            for _ in forever_():
                b = b_h.acquire(1)
                fprm(b, ws)
                b_h.release(1)
                n = pn.read()
                for _ in range_(n_grp):
                    out = o_h.acquire(1)
                    for r in range_(rg):
                        x = a_h.acquire(1)
                        fn(x, ws, out, r, n)
                        a_h.release(1)
                    o_h.release(1)
                out = o_h.acquire(1)  # (the ring)
                fring(ws, out)
                o_h.release(1)

        workers = []
        for c in range(COLS):
            workers.append(Worker(core_q, [a1[c][0].cons(), o3[c].prod(), k_q, Buffer(wsq, name=f"li_wq{c}")],
                                  tile=Tile(c, 2), stack_size=self.Q_STACK, dynamic_objfifo_lowering=True, while_true=False))
            for i, (fn, fp, fr, wty, nm) in enumerate(((k_k, k_kprm, k_kring, wsk, "k"), (k_v, k_vprm, k_vring, wsv, "v"))):
                workers.append(Worker(core_kv, [a1[c][1 + i].cons(), b1[c][i].cons(), g1[c][i].prod(), fn, fp, fr,
                                                Buffer(wty, name=f"li_w{nm}{c}"), p_n],
                                      tile=Tile(c, 4 + i), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        lq, lk, lv = self.lq, self.lk, self.lv
        n_qs, n_kv, uw, tq = self.n_qs, self.n_kv, self.unit, self.tg_qs
        hb, g2, o2, ow = self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2
        k0b, rb, pw = self.kv0 // 2, RING_BYTES // 2, self.prm_w
        n_junk = self.n_junk

        def seq(GQ, GK, GV, TAU, PRM, QS, KV, RING, JUNK, ah, bh, oh, gh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                for t0 in range(0, n_tg, MAX_BD_ITER):  # (a chained fill repeats at most MAX_BD_ITER times)
                    nt = min(MAX_BD_ITER, n_tg - t0)
                    chained_fill(ah[c], GQ, [
                        ChainPart(GQ, TensorAccessPattern((T * lq.N,), c * lq.pitch + 8 * t0 * lq.N, [nt, 1, 8, nq],
                                                          [8 * lq.N, 0, lq.N, 1])),
                        ChainPart(PRM, TensorAccessPattern((COLS * pw,), c * pw + 1024, [nt, 1, 1, 448], [0, 0, 0, 1])),
                        ChainPart(TAU, TensorAccessPattern((2 * self.tau_positions,), 16 * t0, [nt, 1, 1, 16],
                                                           [16, 0, 0, 1]), offset_parameter=p_tau),
                        ChainPart(GK, TensorAccessPattern((T * lk.N,), c * lk.pitch + 8 * t0 * lk.N, [nt, 1, 8, pk],
                                                          [8 * lk.N, 0, lk.N, 1])),
                        ChainPart(GV, TensorAccessPattern((T * lv.N,), c * lv.pitch + 8 * t0 * lv.N, [nt, 1, 8, pk],
                                                          [8 * lv.N, 0, lv.N, 1]))], tg)
                r0 = c * rb  # (the KV head's ring: [slot 8][K 128 | V 128] fp32 = 512 bf16 a slot)
                chained_fill(bh[c], RING, [
                    ChainPart(RING, TensorAccessPattern((COLS * rb,), r0, [1, 1, 8, 256], [0, 0, 512, 1])),
                    ChainPart(PRM, TensorAccessPattern((COLS * pw,), c * pw, [1, 1, 1, 512], [0, 0, 0, 1])),
                    ChainPart(RING, TensorAccessPattern((COLS * rb,), r0 + 256, [1, 1, 8, 256], [0, 0, 512, 1])),
                    ChainPart(PRM, TensorAccessPattern((COLS * pw,), c * pw + 512, [1, 1, 1, 512], [0, 0, 0, 1]))],
                    tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * n_tg * tq, [1, n_tg, 1, 3 * uw + INK_ROWS * 72],
                                                    [0, tq, 0, 1]), group=tg, wait=True)
                # per group its objects' records (rows of 576 elements: a shim BD's inner dimension <= 1023 words), then the
                # ring object: its K rows into the ring's K halves, its V rows into the V halves
                gh[c].drain(KV, TensorAccessPattern((n_kv,), c * hb + k0b, [n_grp, s.objects, ow // 576, 576],
                                                    [g2, o2, 576, 1]), offset_parameter=p_kv, group=tg, wait=True)
                chained_fill(gh[c], RING, [
                    ChainPart(RING, TensorAccessPattern((COLS * rb,), r0, [1, 1, 8, 256], [0, 0, 512, 1])),
                    ChainPart(JUNK, TensorAccessPattern((n_junk,), 0, [1, 1, 1, n_junk], [0, 0, 0, 1])),
                    ChainPart(RING, TensorAccessPattern((COLS * rb,), r0 + 256, [1, 1, 8, 256], [0, 0, 512, 1])),
                    ChainPart(JUNK, TensorAccessPattern((n_junk,), 0, [1, 1, 1, ow - 2048], [0, 0, 0, 1]))],
                    tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * lq.N,), bf], np.ndarray[(T * lk.N,), bf], np.ndarray[(T * lv.N,), bf],
                           np.ndarray[(2 * self.tau_positions,), bf], np.ndarray[(COLS * pw,), bf],
                           np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(COLS * rb,), bf],
                           np.ndarray[(n_junk,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(g3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


__all__ = ["HEADS", "INK_ROWS", "P_N", "P_TAU", "QS_UNITS", "RING_BYTES", "InkPrepOperator", "ink_shape"]
