"""The attention of a prefill on the decode's KV record format (``decode.kv``), one column per KV head (Llama, Qwen3): a
T-token chunk at position p0 of one layer, from the qkv GEMMs' output G (``prefill.gemm``: per column c [q of its qh
heads | k | v] of KV head c) to the O GEMM's A panels, with the chunk's K / V records in the decode's cache, so the
decode continues from them. A KV head's cache is groups of kp + 1 objects of the image's tile size (kp K objects of
records of 8 positions, kp = 2: hi + lo panels, 1: single; the V object), linear in the positions; a chunk starts at a
group (``Shape.pg`` positions: 128 at head_dim 64, 64 at 128). RoPE on the pairs (i, i + hd / 2) (the image's Q / K rows
in rotate-half order), scores in log2 units: q k log2(e) / sqrt(hd).

``PrepOperator``: per column c, cores (c, 2) and (c, 3) the Q heads (the first half of the column's heads, the rest),
(c, 4) K, (c, 5) V. Per 8 tokens a Q core turns each of its heads' rows into the attention's operands: q rotated (fp32)
times the score constant, as single bfp16 panels of 8 queries x 8 dims (ND panels: a Q unit; ``Shape.qp`` 2: hi then
lo panels, 2 ND); QS holds per token group and column the qh heads' units. Qwen3's q and k are RMS-normed per head and
times their weights per dim first (``norm``: the layer's weights the cores' first object; its weights reach ~100, so a
row's dims span ~1e5 within a block: Q hi + lo, KL 0.0005-0.0023 of Qwen3 0.6B less on the CPU model with the
attention's roundings). The K core writes K rotated (fp32) as the cache's K
records (kp per 8 dims: hi + lo, or single panels), the V core V (bf16) as V records (per 8 dims a panel of blocks over
the 8 positions); each fills a group's K objects (V object) in L1 and the memtile joins them into the group's objects
(one transfer per group: a column's shim has two output channels). cos / sin from a table of every position
(``cs_table``); the chunk's rows and cache groups by scratchpad parameters (``PrepOperator.params``).

``AttnOperator``: per column c 4 cores, passes of 32 QB queries; core r takes the QB query blocks of 8 from block QB r
of the pass for each of the column's heads (QB qh row blocks), streams the KV head's cache groups from position 0 up to
the pass's and writes their outputs as the O GEMM's A panels. Two streams per column, each through its memtile and
broadcast to the column's 4 cores (a core's two input channels): A the K records (a single-K cache: its K objects,
each unit's two records interleaved per d-block; hi + lo: each group's [K | K] objects, the memtile skipping the first
one's padding), B the pass's Q units ahead of the V objects, both in objects of two units read per d half (a core keeps
its own Q units). A unit is 144 ND bytes: a hi + lo K record (two single ones), two V records, two row blocks' Q units
(hi + lo Q: one row block's). Per chunk of CR records (QB, CR: the most that L1 holds, ``Shape.blocks``):

- the scores on the MMUL, the accumulators starting at the rows' bias: per pair of row blocks one loop of two panel
  streams (Q | K0, K1: four chains, MMUL-bound; hi + lo K: a loop for K hi, then one for K lo; hi + lo Q: each Q panel
  pair's products into the same accumulators, Q lo against K hi only). A single-K chunk is one call on both its K units
  (the pairs' and units' loop pipelined as an outer loop: an iteration's setup under the previous one's MMULs and
  stores); each tile's epilogue writes the softmax's terms v = max(bf16(A), 129), f = bf16(A - v) (bf16, ``vf``: the
  exp's range reduction where the vector slot is idle) and a flag for the softmax when some v reaches 208;
- the softmax (``soft``, from the chunk's first ``pv``) with a running maximum per row, raised (o, l rescaled) only when
  the chunk's maximum exceeds it by 2^16 (qwen35's ``attn_prefill``: m is only the reference of 2^(s - m)); masks only
  on a diagonal chunk; P = 2^(s - m) times 2^64 (o / l cancels it) as 2^(v - 128) 2^f (km.exp2's steps), clamped at
  2^-63 of that (masked lanes v 129, f -1/2), as single bfp16 panels (half the P V MMULs of hi + lo), or hi + lo
  (``Shape.pp`` 2: the output then within its own bfp16 rounding, against 1.13-1.39x of it; LFM3's and the readout
  models'). Known error, cause open: the keys below a row's reference weigh ~0.4% too much
  (test_llama_npu_teacher_forced[Q3_K_L-1B-hilo] fails);
- o += P V per pair of row blocks (each V panel popped once for both: two MMUL chains; both panel streams on load slot B,
  o's accumulator loads on A: MMUL-bound), and the row sums l += P J on the MMUL (J a ones panel: no accumulator in the
  exp loop). Where L1 holds them o is in two copies a region apart, P V reading one and writing the other (in place,
  its loads and stores meet in one region's banks: ~60 cycles a call); ``l1_layout`` puts every hot loop's streams in
  distinct regions;

chunks past every one of the core's queries are skipped (their objects consumed). At the pass's end o / l per head as
the O GEMM's A panels. QB = 2 (64-query passes: half the cache re-reads) where L1 holds it (1B, 3B), else 1 (8B).
For 2048 tokens at p0 = 8K per layer: Llama's and Qwen3's shapes are cache-bound (3B 13.2 ms, 12.5 with the compute
compiled out); LFM3's broadcast design (``Shape.per`` 2) binds on its cores: 9.6 ms with hi + lo P (~450 cycles per
record and core: scores 124, softmax 110, P V 180, the rest calls and locks), 8.2 with single P, 6.2 with the compute
compiled out.
"""

import itertools
import math
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
from aie.dialects.aiex import npu_write32
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

from ..decode.kv import Records
from ..gemm.gemm_q import ChainPart, chained_fill
from ..operators import templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS, plus_i32, range32, stream_walk
from ..operators.kernel_source import kernel_prefix
from ..operators.kmath import ACC_C
from ..operators.l1_layout import pin_l1_buffers
from ..operators.util import Attr, bf16_hilo, regions
from .gemm import Layout

# scratchpad parameters
P_CS = "pa_cs"  # bf16 offset of the chunk's rows in the cos / sin table (2 hd p0)
P_KV = "pa_kv"  # bf16 offset of the chunk's first cache group in a KV head's cache
P_P8 = "pa_p8"  # p0 / 8 (the attention cores' masks)
P_FG = "pa_fg"  # the cache groups every pass streams first (p0 / pg + 1: the history and the chunk's first)
P_FQ = "pa_fq"  # their chunks of CR records (the cores' count)
P_FO = "pa_fo"  # bf16 offset of the chunk's second group
P_G0 = "pa_g0"  # bf16 offset of the chunk's first group (``AttnOperator(ring=)``)
P_FH = "pa_fh"  # the history's chunks of CR records (``AttnOperator(ring=)``: the cores')
P_HK = "pa_hk"  # the chunk's first keys every row sees (packed: a lead, ``AttnOperator.params``)
P_SW = "pa_sw"  # (sliding window) bf16 offset of the chunk's first cache group from a KV head's group 0
L1 = 0x10000 - 512  # a core's data memory for its buffers (a margin for aiecc's)
MT_BYTES = 0x80000 - 8192  # a memory tile's for the ring variant's buffers (``AttnOperator(ring=)``)


@dataclass(frozen=True)
class Shape:
    """The attention's shapes: head_dim, query heads per column, K panels per record, the cache's object bytes, Q panels
    per row block (``qp``: 1 single bfp16, 2 hi + lo), P panels per row block and record (``pp``: 1, 2), columns per KV
    head (``per``: 1, Llama's; 2, LFM3's: the KV head's K and V halves in the two columns' G slices), a sigmoid output
    gate per head (``gate``: its logits the G slices' block "g", per token group the column's heads' gates after their
    Q units in QS)."""

    hd: int
    qh: int
    kp: int
    a_bytes: int
    qp: int = 1
    pp: int = 1
    per: int = 1
    gate: bool = False

    @cached_property
    def kvh(self) -> int:  # KV heads (of the array's columns)
        return COLS // self.per

    @cached_property
    def gw(self) -> int:  # bytes of a token group's gates of a column (8 tokens x qh bf16)
        return 16 * self.qh if self.gate else 0

    @cached_property
    def tgu(self) -> int:  # bytes of a token group of a column in QS: its heads' Q units, its gates
        return self.qh * self.qunit + self.gw

    @cached_property
    def nd(self) -> int:
        return self.hd // 8

    @cached_property
    def rec(self) -> Records:
        return Records(self.hd, self.kp)

    @cached_property
    def unit(self) -> int:  # bytes of a stream unit (module docstring)
        return 144 * self.nd

    @cached_property
    def qunit(self) -> int:  # bytes of a row block's Q panels (8 queries x 8 dims per panel; hi + lo: [hi ND | lo ND])
        return 72 * self.nd * self.qp

    @cached_property
    def rg(self) -> int:  # records per cache group
        return self.rec.rg(self.a_bytes)

    @cached_property
    def pg(self) -> int:  # positions per cache group
        return 8 * self.rg

    @cached_property
    def objects(self) -> int:  # cache objects per group
        return self.kp + 1

    @cached_property
    def used(self) -> int:  # bytes of an object's records (the rest padding)
        return self.rec.kper(self.a_bytes) * self.rec.k_bytes

    @cached_property
    def group(self) -> int:  # bytes of a cache group
        return self.objects * self.a_bytes

    @cached_property
    def scale(self) -> float:  # the score constant (log2 units)
        return 1 / math.sqrt(self.hd) / math.log(2)

    def l1(self, qb: int, cr: int) -> int:
        """L1 bytes of an attention core at qb row blocks per head and chunks of cr records (``AttnOperator``)."""
        rows = qb * self.qh
        ws = regions(AttnOperator.regions(rows, cr))["size"]
        return (AttnOperator.STACK + 4 * self.unit + 2 * qb * self.nd * 72 + rows * self.qunit + rows * self.nd * 256
                + ws + 72 * rows * cr * self.pp)

    @cached_property
    def blocks(self) -> tuple[int, int]:
        """(QB, CR): the most row blocks per head that fit L1 (2 or 1), chunks of 4 records."""
        return next((qb, 4) for qb in (2, 1) if self.l1(qb, 4) <= L1)


def cs_rows(inv_freq: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """(n, hd) fp32: per position [cos hd / 2 | sin hd / 2] of the pairs' angles position inv_freq[i] (the decode's
    ``decode.attn.rope_tables``)."""
    th = np.outer(np.asarray(positions, np.float64), np.asarray(inv_freq, np.float64))
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


def cs_table(inv_freq: np.ndarray, n: int) -> np.ndarray:
    """``cs_rows`` of the positions 0 .. n - 1."""
    return cs_rows(inv_freq, np.arange(n))


_PREP = templating.load("artlite/prefill/attn.prep.cc.jinja")


class PrepOperator(ArtcOperator):
    """The attention prep of T tokens (module docstring). Runtime sequence: G (the qkv GEMM's output, ``layout``), CS
    (``cs_table`` of every position, as bf16 pairs), QS (per token group and column the heads' Q units; a sub-chunk's
    part of a chunk's), KV (the layer's cache: COLS KV heads of ``kv_bytes`` each, ``kv_pad`` bytes past them); with
    ``norm`` (Qwen3: q and k RMS-normed per head with weights per dim before RoPE, ``eps``) NW, the layer's [q_norm |
    k_norm] fp32 (``nw_bytes``: the B stream's first object, before the cos / sin rows). Parameters ``params``."""

    STACK = 1024
    pfx = kernel_prefix("pap")

    def __init__(self, *, T: int, shape: Shape, layout: Layout, kv_bytes: int, cs_positions: int, kv_pad: int = 0,
                 norm: bool = False, eps: float = 1e-6, kv0: int = 0):
        """``kv0``: the bytes of a KV head's cache before its group 0 (a sliding-window cache's guard)."""
        s = shape
        assert T % s.pg == 0 and T // s.pg <= 64 and {"q", "k", "v"} <= set(layout.off)
        assert s.qh in (2, 3, 4) and s.used % 1152 == 0 and s.per in (1, 2)
        assert s.per == 1 or (s.qh == 2 and s.hd % (2 * 64) == 0), "two columns per KV head: a column's two heads"
        assert not s.gate or (s.per == 2 and "g" in layout.off), "the gates: the cores two columns of a KV head free"
        self.T, self.shape, self.layout, self.kv_bytes, self.cs_positions = T, s, layout, kv_bytes, cs_positions
        self.kv_pad, self.norm, self.eps, self.kv0 = kv_pad, norm, eps, kv0
        self.nq = [-(-s.qh // 2), s.qh // 2]  # (the Q cores' heads)
        super().__init__()

    @staticmethod
    def nw_bytes(hd: int) -> int:
        """Bytes of a layer's NW object (the B stream's objects: 8 positions' cos / sin; its weights 8 hd)."""
        return 8 * 2 * hd * 2

    def workspace(self) -> dict[str, int]:
        hd = self.shape.hd
        return regions([("row", 4 * hd), ("stage", 32 * hd)] + ([("nrm", 8 * hd)] if self.norm else []))

    def get_kernel_src(self) -> str:
        p = self.pfx
        s = self.shape
        return _PREP.render(p=p, HD=s.hd, ND=s.nd, KP=s.kp, QP=s.qp, KREC=s.rec.k_bytes, QUNIT=s.qunit,
                            qc=bf16_hilo(s.scale), w=Attr(self.workspace()), NORM=self.norm, eps=repr(self.eps),
                            GATE=s.gate, QH=s.qh)

    @property
    def n_cs(self) -> int:
        return self.cs_positions * 2 * self.shape.hd

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument: [column][token group][head][Q unit] (with gates: each token group's then)."""
        return self.T // 8 * COLS * self.shape.tgu // 2

    @property
    def n_kv(self) -> int:
        return (self.shape.kvh * self.kv_bytes + self.kv_pad) // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        nw = [ArgSpec("in", (self.nw_bytes(self.shape.hd) // 2,), bfloat16)] if self.norm else []
        return [ArgSpec("in", (self.T * self.layout.N,), bfloat16), ArgSpec("in", (self.n_cs,), bfloat16),
                ArgSpec("out", (self.n_qs,), bfloat16), ArgSpec("out", (self.n_kv,), bfloat16), *nw]

    def params(self, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 (a group start)."""
        s = self.shape
        assert p0 % s.pg == 0 and p0 + self.T <= self.cs_positions
        return {P_CS: 2 * s.hd * p0, P_KV: s.group // 2 * (p0 // s.pg)}

    def get_mlir_artifact(self) -> Module:
        if self.shape.per == 2:
            return self._artifact2()
        p, T, lay, s = self.pfx, self.T, self.layout, self.shape
        hd, N, pitch, nq = s.hd, lay.N, lay.pitch, self.nq
        bf = np.dtype[bfloat16]
        q_ty = [np.ndarray[(8 * n * hd,), bf] for n in nq]  # 8 rows of a Q core's heads
        row_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of K or V
        cs_ty = np.ndarray[(8 * 2 * hd,), bf]  # 8 positions x [cos | sin] fp32
        qo_ty = [np.ndarray[(n * s.qunit // 2,), bf] for n in nq]
        kg_ty = np.ndarray[(s.kp * s.used // 2,), bf]  # a group's K objects (their records)
        vg_ty = np.ndarray[(s.used // 2,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_q = Kernel(f"{p}_q0", "kernel.o", [q_ty[0], cs_ty, ws_ty, qo_ty[0], np.int32, np.int32])
        k_q1 = Kernel(f"{p}_q1", "kernel.o", [q_ty[1], cs_ty, ws_ty, qo_ty[1], np.int32, np.int32])
        k_k = Kernel(f"{p}_k", "kernel.o", [row_ty, cs_ty, ws_ty, kg_ty, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [row_ty, vg_ty, np.int32])
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty]) if self.norm else None
        p_cs, p_kv = ScratchpadParameter(P_CS, np.int32), ScratchpadParameter(P_KV, np.int32)
        n_tg, rg, n_grp = T // 8, s.rg, T // s.pg

        # per column and token group [Q core 0's rows | Q core 1's | K rows | V rows], split in the memtile
        sizes = [8 * nq[0] * hd, 8 * nq[1] * hd, 8 * hd, 8 * hd]
        offs = [sum(sizes[:i]) for i in range(4)]
        a3 = [ObjectFifo(np.ndarray[(sum(sizes),), bf], name=f"lp_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split(offs, tile=Tile(c, 1), depths=[2] * 4, obj_types=[*q_ty, row_ty, row_ty],
                                 names=[f"lp_a{c}_{i}" for i in range(4)]) for c in range(COLS)]
        b = [ObjectFifo(cs_ty, name=f"lp_b{c}", depth=2) for c in range(COLS)]
        # the Q cores' units joined per token group; the K and V cores' group objects joined per group
        o3 = [ObjectFifo(np.ndarray[(s.qh * s.qunit // 2,), bf], name=f"lp_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([0, nq[0] * s.qunit // 2], tile=Tile(c, 1), depths=[2, 2], obj_types=qo_ty,
                                names=[f"lp_o{c}_{i}" for i in range(2)]) for c in range(COLS)]
        g3 = [ObjectFifo(np.ndarray[(s.objects * s.used // 2,), bf], name=f"lp_g{c}", depth=2) for c in range(COLS)]
        g1 = [g3[c].prod().join([0, s.kp * s.used // 2], tile=Tile(c, 1), depths=[2, 2], obj_types=[kg_ty, vg_ty],
                                names=[f"lp_k{c}", f"lp_v{c}"]) for c in range(COLS)]

        def prm(b_h, fn, ws):  # (norm: the layer's weights, the B stream's first object)
            if fn is not None:
                w = b_h.acquire(1)
                fn(w, ws)
                b_h.release(1)

        def core_q(a_h, b_h, o_h, fn, ws, n, k_prm):
            for _ in forever_():
                prm(b_h, k_prm, ws)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    for h in range(n):
                        fn(x, cs, ws, out, h, n)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        def core_k(a_h, b_h, o_h, fn, ws, k_prm):
            for _ in forever_():
                prm(b_h, k_prm, ws)
                for _ in range_(n_grp):
                    out = o_h.acquire(1)
                    for r in range_(rg):
                        x = a_h.acquire(1)
                        cs = b_h.acquire(1)
                        fn(x, cs, ws, out, r)
                        b_h.release(1)
                        a_h.release(1)
                    o_h.release(1)

        def core_v(a_h, o_h, fn):
            for _ in forever_():
                for _ in range_(n_grp):
                    out = o_h.acquire(1)
                    for r in range_(rg):
                        x = a_h.acquire(1)
                        fn(x, out, r)
                        a_h.release(1)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for i, fn in enumerate((k_q, k_q1)):
                workers.append(Worker(core_q, [a1[c][i].cons(), b[c].cons(), o1[c][i].prod(), fn,
                                               Buffer(ws_ty, name=f"lp_ws{c}_{i}"), nq[i], k_prm],
                                      tile=Tile(c, 2 + i), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            workers.append(Worker(core_k, [a1[c][2].cons(), b[c].cons(), g1[c][0].prod(), k_k,
                                           Buffer(ws_ty, name=f"lp_ws{c}_k"), k_prm],
                                  tile=Tile(c, 4), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                  while_true=False))
            workers.append(Worker(core_v, [a1[c][3].cons(), g1[c][1].prod(), k_v], tile=Tile(c, 5),
                                  stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv, n_qs = self.n_cs, self.n_kv, self.n_qs
        uw, qw = s.qunit // 2, s.qh * s.qunit // 2  # (bf16: a Q unit, a column's units of a token group)
        hb, g2, o2, ow = self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2

        n_nw = self.nw_bytes(hd) // 2

        def seq(G, CS, QS, KV, *rest):
            *nw, ah, bh, oh, gh = rest
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                for t0 in range(0, n_tg, 64):  # (a chained fill repeats at most 64 times: one per 64 token groups)
                    q0, k0 = c * pitch + lay.off["q"] + 8 * t0 * N, c * pitch + lay.off["k"] + 8 * t0 * N
                    nt = min(64, n_tg - t0)
                    chained_fill(ah[c], G, [
                        TensorAccessPattern((T * N,), q0, [nt, 1, 8, nq[0] * hd], [8 * N, 0, N, 1]),
                        TensorAccessPattern((T * N,), q0 + nq[0] * hd, [nt, 1, 8, nq[1] * hd], [8 * N, 0, N, 1]),
                        TensorAccessPattern((T * N,), k0, [nt, 2, 8, hd], [8 * N, lay.off["v"] - lay.off["k"], N, 1])],
                        tg)
                cs = TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 2 * hd], [0, 0, 0, 1])
                if nw:
                    chained_fill(bh[c], CS, [ChainPart(nw[0], TensorAccessPattern((n_nw,), 0, [1, 1, 1, n_nw], [0, 0, 0, 1])),
                                             ChainPart(CS, cs, offset_parameter=p_cs)], tg)
                else:
                    bh[c].fill(CS, cs, offset_parameter=p_cs, group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * n_tg * qw, [1, n_tg, s.qh, uw], [0, qw, uw, 1]),
                            group=tg, wait=True)
                # per group its objects' records (rows of 576 elements: a shim BD's inner dimension <= 1023 words)
                gh[c].drain(KV, TensorAccessPattern((n_kv,), c * hb, [n_grp, s.objects, ow // 576, 576],
                                                    [g2, o2, 576, 1]), offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(n_qs,), bf],
                           np.ndarray[(n_kv,), bf], *([np.ndarray[(n_nw,), bf]] if self.norm else []),
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(g3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


    def _artifact2(self) -> Module:
        """Two columns per KV head (``Shape.per``): column c = 2 h + j takes its two heads' Q (cores (c, 2), (c, 3)); KV head
        h's K records from both columns' K halves on core (2 h, 4), its V records on (2 h + 1, 5); each column's gates
        (``Shape.gate``: sigmoid of its heads' logits, after its Q units in QS) on its free core ((2 h, 5), (2 h + 1, 4))."""
        p, T, lay, s = self.pfx, self.T, self.layout, self.shape
        hd, N, pitch, qh = s.hd, lay.N, lay.pitch, s.qh
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of a Q core's head
        row_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of K or V (both halves)
        gi_ty = np.ndarray[(8 * qh,), bf]  # 8 rows of the column's gate logits
        cs_ty = np.ndarray[(8 * 2 * hd,), bf]
        qo_ty = np.ndarray[(s.qunit // 2,), bf]
        go_ty = np.ndarray[(s.gw // 2,), bf]
        kg_ty = np.ndarray[(s.kp * s.used // 2,), bf]
        vg_ty = np.ndarray[(s.used // 2,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_q = [Kernel(f"{p}_q{i}", "kernel.o", [q_ty, cs_ty, ws_ty, qo_ty, np.int32, np.int32]) for i in range(2)]
        k_k = Kernel(f"{p}_k", "kernel.o", [row_ty, cs_ty, ws_ty, kg_ty, np.int32])
        k_v = Kernel(f"{p}_v", "kernel.o", [row_ty, vg_ty, np.int32])
        k_g = Kernel(f"{p}_g", "kernel.o", [gi_ty, go_ty]) if s.gate else None
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty]) if self.norm else None
        p_cs, p_kv = ScratchpadParameter(P_CS, np.int32), ScratchpadParameter(P_KV, np.int32)
        n_tg, rg, n_grp = T // 8, s.rg, T // s.pg
        gate = s.gate

        def parts(j: int) -> list[tuple[object, int]]:  # (per column the memtile object's pieces: rows 2 .. 5)
            kv = [(row_ty, 8 * hd)]
            g = [(gi_ty, 8 * qh)] if gate else []
            return [(q_ty, 8 * hd), (q_ty, 8 * hd), *(kv + g if j == 0 else g + kv)]

        a1, o1, kvo = {}, {}, {}
        for c in range(COLS):
            j = c % 2
            pc = parts(j)
            offs = [sum(n for _, n in pc[:i]) for i in range(len(pc))]
            a3 = ObjectFifo(np.ndarray[(sum(n for _, n in pc),), bf], name=f"lp_a{c}", depth=2)
            a1[c] = (a3, a3.cons().split(offs, tile=Tile(c, 1), depths=[2] * len(pc), obj_types=[t for t, _ in pc],
                                         names=[f"lp_a{c}_{i}" for i in range(len(pc))]))
            o3 = ObjectFifo(np.ndarray[(s.tgu // 2,), bf], name=f"lp_o{c}", depth=2)
            jo = [0, s.qunit // 2] + ([qh * s.qunit // 2] if gate else [])
            o1[c] = (o3, o3.prod().join(jo, tile=Tile(c, 1), depths=[2] * len(jo),
                                        obj_types=[qo_ty, qo_ty] + ([go_ty] if gate else []),
                                        names=[f"lp_o{c}_{i}" for i in range(len(jo))]))
            kvo[c] = ObjectFifo(kg_ty if j == 0 else vg_ty, name=f"lp_g{c}", depth=2)
        b = [ObjectFifo(cs_ty, name=f"lp_b{c}", depth=2) for c in range(COLS)]

        def prm(b_h, fn, ws):
            if fn is not None:
                w = b_h.acquire(1)
                fn(w, ws)
                b_h.release(1)

        def core_q(a_h, b_h, o_h, fn, ws, k_prm):
            for _ in forever_():
                prm(b_h, k_prm, ws)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, cs, ws, out, 0, 1)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        def core_k(a_h, b_h, o_h, fn, ws, k_prm):
            for _ in forever_():
                prm(b_h, k_prm, ws)
                for _ in range_(n_grp):
                    out = o_h.acquire(1)
                    for r in range_(rg):
                        x = a_h.acquire(1)
                        cs = b_h.acquire(1)
                        fn(x, cs, ws, out, r)
                        b_h.release(1)
                        a_h.release(1)
                    o_h.release(1)

        def core_v(a_h, o_h, fn):
            for _ in forever_():
                for _ in range_(n_grp):
                    out = o_h.acquire(1)
                    for r in range_(rg):
                        x = a_h.acquire(1)
                        fn(x, out, r)
                        a_h.release(1)
                    o_h.release(1)

        def core_g(a_h, o_h, fn):
            for _ in forever_():
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, out)
                    o_h.release(1)
                    a_h.release(1)

        workers = []
        for c in range(COLS):
            j, sp, jn = c % 2, a1[c][1], o1[c][1]
            for i in range(2):
                workers.append(Worker(core_q, [sp[i].cons(), b[c].cons(), jn[i].prod(), k_q[i],
                                               Buffer(ws_ty, name=f"lp_ws{c}_{i}"), k_prm],
                                      tile=Tile(c, 2 + i), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            if j == 0:
                workers.append(Worker(core_k, [sp[2].cons(), b[c].cons(), kvo[c].prod(), k_k,
                                               Buffer(ws_ty, name=f"lp_ws{c}_k"), k_prm],
                                      tile=Tile(c, 4), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            else:
                workers.append(Worker(core_v, [sp[3 if gate else 2].cons(), kvo[c].prod(), k_v], tile=Tile(c, 5),
                                      stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
            if gate:
                workers.append(Worker(core_g, [sp[3 if j == 0 else 2].cons(), jn[2].prod(), k_g],
                                      tile=Tile(c, 5 if j == 0 else 4), stack_size=2 * self.STACK,  # (kmath's sigmoid)
                                      dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv, n_qs, n_nw = self.n_cs, self.n_kv, self.n_qs, self.nw_bytes(hd) // 2
        tw = s.tgu // 2  # (bf16 of a column's token group in QS)
        hb, g2, o2, ow, k0b = self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2, self.kv0 // 2

        def seq(G, CS, QS, KV, *rest):
            *nw, ah, bh, oh, gh = rest
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                h, j = divmod(c, 2)
                for t0 in range(0, n_tg, 64):  # (a chained fill repeats at most 64 times: one per 64 token groups)
                    nt = min(64, n_tg - t0)
                    q0 = c * pitch + lay.off["q"] + 8 * t0 * N
                    kv = TensorAccessPattern((T * N,), 2 * h * pitch + lay.off["k" if j == 0 else "v"] + 8 * t0 * N,
                                             [nt, 8, 2, hd // 2], [8 * N, N, pitch, 1])  # (both columns' halves)
                    pieces = [TensorAccessPattern((T * N,), q0, [nt, 1, 8, hd], [8 * N, 0, N, 1]),
                              TensorAccessPattern((T * N,), q0 + hd, [nt, 1, 8, hd], [8 * N, 0, N, 1])]
                    g = [TensorAccessPattern((T * N,), c * pitch + lay.off["g"] + 8 * t0 * N, [nt, 1, 8, qh],
                                             [8 * N, 0, N, 1])] if gate else []
                    chained_fill(ah[c], G, pieces + ([kv] + g if j == 0 else g + [kv]), tg)
                cs = TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 2 * hd], [0, 0, 0, 1])
                if nw:
                    chained_fill(bh[c], CS, [ChainPart(nw[0], TensorAccessPattern((n_nw,), 0, [1, 1, 1, n_nw], [0, 0, 0, 1])),
                                             ChainPart(CS, cs, offset_parameter=p_cs)], tg)
                else:
                    bh[c].fill(CS, cs, offset_parameter=p_cs, group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * n_tg * tw, [1, 1, n_tg, tw], [0, 0, tw, 1]),
                            group=tg, wait=True)
                if j == 0:  # (the KV head's K objects of each group; its V object: column 2 h + 1's)
                    pat = TensorAccessPattern((n_kv,), h * hb + k0b, [n_grp, s.kp, ow // 576, 576], [g2, o2, 576, 1])
                else:
                    pat = TensorAccessPattern((n_kv,), h * hb + k0b + s.kp * o2, [1, n_grp, ow // 576, 576],
                                              [0, g2, 576, 1])
                gh[c].drain(KV, pat, offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(n_qs,), bf],
                           np.ndarray[(n_kv,), bf], *([np.ndarray[(n_nw,), bf]] if self.norm else []),
                           [a1[c][0].prod(tile=Tile(c, 0)) for c in range(COLS)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [o1[c][0].cons(tile=Tile(c, 0)) for c in range(COLS)],
                           [kvo[c].cons(tile=Tile(c, 0)) for c in range(COLS)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_ATTN = templating.load("artlite/prefill/attn.cc.jinja")


class AttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 (module docstring) on the cache that already holds the chunk's K and V
    (``PrepOperator``). Column c = KV head c (query heads qh c ..), core row r; in pass b the core takes the query
    blocks 4 QB b + QB r .. of each of the column's heads. Runtime sequence: QS (the prep's of sub-chunks of ``sub``
    tokens, each [column][token group][head][unit]), KV (the layer's cache: COLS KV heads of ``kv_bytes``, ``kv_pad``
    past them), A (T x heads hd as bfp16 panels [T / 8][heads hd / 8]). Parameters ``params``.

    Packed (``window`` W > 0: several inputs, segments, in the chunk): each row sees the history (the cache before p0)
    and its segment's keys up to itself, from its segment's first key lo (the SEG argument: ``seg``, an object per pass
    on stream B ahead of the V objects). Pass b streams the history and the chunk's groups max(1, D - W + 1) .. D (D:
    its queries' group; a window fixed by the design: no runtime parameters), so a segment spans at most
    ``max_segment`` positions; key chunks below every one of a core's rows' segments are skipped, those below some
    masked (-1e30). It reads at the DDR port's rate (each pass the history and its window again).

    The ring variant (``ring`` H > 0: hi + lo K, a pass per cache group, a history of at most H groups) reads each group
    once: per column the memory tile holds H groups from position 0 (the history's, the rest skipped) and a ring of W + 1
    of the chunk's groups (slot b % (W + 1) group b), and sends every pass the history and the whole ring
    (``_ring_memtiles``); the cores skip the slots outside their window (the chunk's groups max(0, D - W + 1) .. D), so
    a pass takes its window's key chunks in the slots' order (o, l the same sums in another order: not bit-identical to
    the window variant). A key chunk the cores skip still crosses their input streams: the ring wins from 2048 tokens.

    Non-causal (``causal`` False: LFM3's d1 decision models' full-attention layers, a prompt in one chunk at position 0):
    every pass streams the prompt's cache groups (the causal design's history part, from group 0: ``set_params``);
    row q sees the keys below the state's end S (``P_HK``) and its own, lo .. hi - 1 (the SEG argument, ``set_seg``: an
    object per pass after the GATE object, per row lo then hi, int16): the state's rows and the padding's every key of
    the prompt, a question's rows the state's and their question's. Key chunks no row of a core sees are skipped, those
    some row sees in part masked (-1e30)."""

    STACK = 2048
    pfx = kernel_prefix("paa")
    # stats builds: core counters 0: cycles in the region, 1: lock (memory) stall cycles, 2: all cycles, 3: windows
    # (AM029 Performance_Control0 / 1)
    CTRL: ClassVar[tuple[tuple[int, int], ...]] = ((0x31500, 33 | 34 << 8 | 26 << 16 | 26 << 24),
                                                   (0x31504, 1 | 1 << 8 | 33 << 16 | 33 << 24))
    COUNTER0 = 0x31520

    def __init__(self, *, T: int, shape: Shape, kv_bytes: int, kv_pad: int = 0, sub: int = 0,
                 ablate: tuple[str, ...] = (), window: int = 0, ring: int = 0, swa: int = 0, kv0: int = 0,
                 bcast: bool = True, causal: bool = True):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv": wrong results); "ev_<region>"
        (qk, soft, pv, raise, chunk: grp to the second pv) marks a region with event0 / event1 and the sequence programs the
        cores' performance counters (``CTRL``; "memstall": memory instead of lock stalls in counter 1). ``window``: the
        packed variant's window of cache groups; ``ring``: its history's groups held in the memory tiles (class
        docstring); ``swa``: a sliding window of that many positions (whole cache groups: each pass streams the window's
        groups before its own and its own, from ``kv0`` bytes of a KV head's cache, its guard, before group 0); ``bcast``
        (two columns per KV head, ``Shape.per``): the KV head's streams read once, by column 2 h's shim, and broadcast by
        its memtile to the cores of both columns (each taking its column's Q units and gates of both), else each column
        reads them (twice the cache's bytes: a reference); ``causal`` False: the non-causal design (class docstring)."""
        s = shape
        self.qb, self.cr = s.blocks
        self.pq = 32 * self.qb  # queries per pass
        self.m = s.pg // self.pq  # passes per cache group
        self.sub = sub or T
        assert T % s.pg == 0 and T // self.pq <= 64 and s.rg % self.cr == 0 and self.m >= 1
        assert self.qb * s.qh % 2 == 0 and T % self.sub == 0 and self.sub % self.pq == 0, "row blocks in pairs"
        assert 4 * self.qb * s.qh % 4 == 0  # (a pass's Q units: whole B objects of 4)
        assert window >= 0 and self.pq <= 72, "a pass's segment starts in its object's first 72 values (stream B's walk)"
        self.T, self.shape, self.kv_bytes, self.kv_pad, self.ablate = T, s, kv_bytes, kv_pad, tuple(ablate)
        self.win = window
        self.n_pass = T // self.pq
        self.ring = ring
        self.swa, self.kv0 = swa, kv0
        self.bc = shape.per == 2 and bcast  # (one stream per KV head, broadcast to its columns' cores)
        self.nc = not causal
        assert not self.nc or (not window and not ring and not swa and T < 32768), "non-causal: full layers, int16 keys"
        assert not self.bc or (shape.gate and not window and not ring), "the broadcast: LFM3's shapes"
        assert not swa or (swa % s.pg == 0 and not window and not ring and kv0 >= swa // s.pg * s.group), swa
        if ring:
            assert window and self.m == 1 and s.kp == 2 and s.used % (2 * s.unit) == 0, "the ring: hi + lo K, a pass a group"
            assert self.mt_bytes <= MT_BYTES, (self.mt_bytes, "a column's memory tile")
        self.oc = 2 if self._layout2() is not None else 1  # (o's copies: ``l1_layout``)
        super().__init__()

    @staticmethod
    def regions(rows: int, cr: int, qb: int = 0, nc: int = 0) -> list[tuple[str, int]]:
        """The workspace's regions at ``rows`` row blocks and chunks of ``cr`` records (the kernels' comment); ``qb``: the
        packed variant's query blocks (their rows' segment starts); ``nc``: the non-causal variant's (their rows' keys)."""
        return [("sc", 256 * rows * cr), ("bias", 256 * rows), ("l", 256 * rows), ("vm", 192), ("st", 64)] + (
            [("lo", 128 * qb + 128)] if qb else []) + ([("lh", 32 * nc)] if nc else [])

    OS = 0x4000 // 4  # (floats from o's first copy to its second: a region apart)

    @property
    def pp_bytes(self) -> int:
        """Bytes of the P panels of a chunk (the kernels' pp)."""
        return 72 * self.qb * self.shape.qh * self.cr * self.shape.pp

    @property
    def o_bytes(self) -> int:
        """Bytes of one copy of o."""
        return 256 * self.shape.nd * self.qb * self.shape.qh

    def workspace(self) -> dict[str, int]:
        gt = [("gt", 32 * self.qb * self.shape.qh)] if self.shape.gate else []
        return regions(self.regions(self.qb * self.shape.qh, self.cr, self.qb if self.win else 0,
                                    self.qb if self.nc else 0) + gt)

    @property
    def mt_bytes(self) -> int:
        """Bytes of a column's memory tile the ring variant takes: the held history and the ring (K and V), the pass's
        Q units and SEG, the output join's objects."""
        s = self.shape
        return ((self.ring + self.win + 1) * 3 * s.used + (2 * self.qb * s.qh * s.qp + 2) * s.unit
                + 2 * 4 * self.qb * s.nd * 72)

    @staticmethod
    def ring_groups(shape: Shape, window: int, qb: int) -> int:
        """The most history groups (the chunk's first included) a ring variant's memory tiles hold beside its ring."""
        s = shape
        rest = (window + 1) * 3 * s.used + (2 * qb * s.qh * s.qp + 2) * s.unit + 2 * 4 * qb * s.nd * 72
        return (MT_BYTES - rest) // (3 * s.used)

    @property
    def max_segment(self) -> int:
        """The packed variant's longest segment: its window holds its keys wherever in a cache group it starts."""
        return (self.win - 1) * self.shape.pg + 1 if self.win else self.T

    def get_kernel_src(self) -> str:
        p = self.pfx
        s = self.shape
        tri = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j > q)
        tril = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j <= q)
        w = self.workspace()
        cur = f" + {self.OS} * AW(int32_t, {w['st']})[15]" if self.oc == 2 else ""
        return _ATTN.render(p=p, w=Attr(w), ND=s.nd, QB=self.qb, QH=s.qh, RB=self.qb * s.qh, CR=self.cr,
                            OC=self.oc, OS=self.OS, CUR=cur, c=ACC_C,
                            KP=s.kp, QP=s.qp, PP=s.pp, M=self.m, QUNIT=s.qunit, tri=f"{tri:016X}", ablate=self.ablate,
                            WIN=self.win, RG=s.rg, RING=self.ring, R=self.win + 1, GQ=s.rg // self.cr,
                            HBC=self.ring * s.rg // self.cr, SWA=self.swa > 0, WR=self.swa // 8,
                            WG=self.swa // s.pg, tril=f"{tril:016X}", GATE=s.gate, NC=self.nc)

    @property
    def n_qs(self) -> int:
        return self.T // 8 * COLS * self.shape.tgu // 2

    @property
    def n_kv(self) -> int:
        return (self.shape.kvh * self.kv_bytes + self.kv_pad) // 2

    @property
    def n_a(self) -> int:
        return self.T * COLS * self.shape.qh * self.shape.hd * 9 // 16

    @property
    def n_seg(self) -> int:
        """bf16 of the SEG argument (packed): an object of two units per pass."""
        return self.n_pass * self.shape.unit

    def get_arg_spec(self) -> list[ArgSpec]:
        seg = [ArgSpec("in", (self.n_seg,), bfloat16)] if self.win or self.nc else []
        return [ArgSpec("in", (self.n_qs,), bfloat16), ArgSpec("in", (self.n_kv,), bfloat16),
                ArgSpec("out", (self.n_a,), bfloat16), *seg]

    def seg(self, lo: np.ndarray) -> np.ndarray:
        """The SEG argument's bytes (uint8) of a packed chunk: ``lo`` (T,) per token the chunk-relative first key of its
        segment (a padding token's: itself) -> per pass its rows' as int16 at its object's start."""
        assert self.win and lo.shape == (self.T,) and (lo <= np.arange(self.T)).all() and (lo >= 0).all()
        out = np.zeros((self.n_pass, self.shape.unit), np.int16)
        out[:, :self.pq] = lo.reshape(self.n_pass, self.pq)
        return out.view(np.uint8).reshape(-1)

    def set_seg(self, sets: tuple[int, ...]) -> np.ndarray:
        """The SEG argument's bytes (uint8) of a non-causal prompt of n = sets[-1] tokens, a state and questions after it
        (``sets``: bounds S, e1, .., en, as ``NpuModel.outputs`` takes them) -> per pass its rows' lo (values 0 ..) and
        hi (values 72 ..: stream B's walk puts them 288 bf16 into the unit) as int16: the state's rows and the padding's
        0, n; question k's e(k-1), ek."""
        assert self.nc and len(sets) >= 1 and sets[0] >= 1 and list(sets) == sorted(sets) and sets[-1] <= self.T, sets
        n = sets[-1]
        lo, hi = np.zeros(self.T, np.int16), np.full(self.T, n, np.int16)
        for a, b in itertools.pairwise(sets):
            lo[a:b], hi[a:b] = a, b
        out = np.zeros((self.n_pass, self.shape.unit), np.int16)
        out[:, :self.pq] = lo.reshape(self.n_pass, self.pq)
        out[:, 72:72 + self.pq] = hi.reshape(self.n_pass, self.pq)
        return out.view(np.uint8).reshape(-1)

    def set_params(self, sets: tuple[int, ...]) -> dict[str, int]:
        """The scratchpad parameters of a non-causal prompt (``set_seg``'s ``sets``): its cache groups every pass streams
        from position 0, the state's end."""
        s = self.shape
        g = -(-sets[-1] // s.pg)
        return {P_P8: 0, P_FG: g, P_FQ: g * s.rg // self.cr, P_HK: sets[0]}

    def params(self, p0: int, lead: int = 0) -> dict[str, int]:
        """The scratchpad parameters of a chunk at p0 (packed: ``lead``, its first keys every row sees: 0 or a cache
        group, a prompt every segment continues; its rows a segment of their own)."""
        s = self.shape
        assert not self.nc, "non-causal: set_params"
        assert p0 % s.pg == 0 and lead in (0, s.pg) and (self.win or not lead)
        if self.swa:
            return {P_P8: p0 // 8, P_SW: s.group // 2 * (p0 // s.pg)}
        fg = p0 // s.pg + 1
        if self.ring:
            assert fg - 1 + lead // s.pg <= self.ring, (p0, lead, self.ring, "a history the memory tiles hold")
            return {P_P8: p0 // 8, P_FH: (fg - 1 + lead // s.pg) * s.rg // self.cr, P_G0: s.group // 2 * (fg - 1),
                    P_HK: lead}
        out = {P_P8: p0 // 8, P_FG: fg, P_FQ: fg * s.rg // self.cr} | ({P_HK: lead} if self.win else {})
        if self.n_pass > self.m:  # (a pass streams the chunk's groups after its first)
            out[P_FO] = s.group // 2 * fg
        return out

    def get_mlir_artifact(self) -> Module:
        p, s, qb, cr, m = self.pfx, self.shape, self.qb, self.cr, self.m
        nd, qh, rb = s.nd, s.qh, qb * s.qh
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        u_ty = np.ndarray[(s.unit // 2,), bf]
        q_ty = np.ndarray[(rb * s.qunit // 2,), bf]
        o_ty = np.ndarray[(qb * nd * 36,), bf]  # a head's A panels of the core's blocks
        om_ty = np.ndarray[(4 * qb * nd * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), f32]
        wo_ty = np.ndarray[((self.OS if self.oc == 2 else 0) + 64 * nd * rb,), f32]
        pp_ty = np.ndarray[(self.pp_bytes,), np.dtype[np.uint8]]
        win = self.win
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty, np.int32, np.int32, np.int32] + [np.int32] * (
            2 if win else 1 if self.nc else 0))
        k_seg = Kernel(f"{p}_seg", "kernel.o", [u_ty, ws_ty, np.int32]) if win or self.nc else None
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [u_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, u_ty, u_ty, ws_ty, np.int32, np.int32] if s.kp == 1 else
                      [q_ty, u_ty, ws_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [u_ty, ws_ty, wo_ty, pp_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty, np.int32])
        k_gsel = Kernel(f"{p}_gsel", "kernel.o", [u_ty, ws_ty, np.int32]) if s.gate else None
        p_p8, p_fq = ScratchpadParameter(P_P8, np.int32), ScratchpadParameter(P_FH if self.ring else P_FQ, np.int32)
        p_fg, p_fo = ScratchpadParameter(P_FG, np.int32), ScratchpadParameter(P_FO, np.int32)
        p_sw = ScratchpadParameter(P_SW, np.int32)
        n_pass, ku, gq = self.n_pass, cr * s.kp // 2, s.rg // cr

        # stream A: the K records (hi + lo: a group's [K obj | padding | K obj], the memtile reads the objects' records)
        ilv = stream_walk(s.unit // 2, [(nd, 36), (2, nd * 36), (36, 1)])  # (per d-block the unit's two halves)
        if self.ring:  # (a group's K records without the padding; the memory tiles' own buffers: ``_ring_memtiles``)
            ka_ty = np.ndarray[(s.used,), bf]
            walk = stream_walk(s.used, [(s.used // 576, 576), (576, 1)])
        elif s.kp == 2:
            ka_ty = np.ndarray[((s.a_bytes + s.used) // 2,), bf]
            walk = stream_walk((s.a_bytes + s.used) // 2, [(2, s.a_bytes // 2), (s.used // 1152, 576), (576, 1)],
                               pad=[(0, 0)] * 3)
        else:  # (a unit's two records interleaved per d-block)
            ka_ty, walk = u_ty, ilv
        scols = list(range(0, COLS, s.per)) if self.bc else list(range(COLS))  # (the columns whose shims stream)
        a3 = {c: ObjectFifo(ka_ty, name=f"la_ka{c}", depth=2) for c in scols}
        # (single K: a core takes a chunk's two K units at once, its two L1 objects; the memtile's end one deeper, so
        # the cores' depth is the stated 2, not the acquire's 3)
        a1 = {c: a3[c].cons().forward(tile=Tile(c, 1), obj_type=u_ty, depth=3 if s.kp == 1 else 2, name=f"la_k{c}",
                                      to_stream=walk) for c in scols}
        # stream B: objects of two units, four slices of nd panels (4 V records; 4 Q units: two pairs of row blocks)
        # -> per d half [d-block pair][slice][d-block 2]
        vwalk = stream_walk(s.unit, [(2, nd // 2 * 36), (nd // 4, 72), (4, nd * 36), (72, 1)])
        b3 = {c: ObjectFifo(np.ndarray[(s.unit,), bf], name=f"la_vb{c}", depth=2) for c in scols}
        b1 = {c: b3[c].cons().forward(tile=Tile(c, 1), obj_type=u_ty, depth=2, name=f"la_v{c}", to_stream=vwalk)
              for c in scols}
        o3 = [ObjectFifo(om_ty, name=f"la_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([qb * nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"la_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, bb, nq, seg=None,
                 hk=None, full=False, gsel=None):
            """(a core's share of a pass: its Q units of the pass's nq, its segments and lead (packed), the history, its output;
            ``full``: the packed window's last groups only, ``win`` of them; the ring: the held history's and the ring's
            chunks, every pass)"""
            for i in range_(nq):
                u = b_h.acquire(1)
                qsel(u, q, i, me)
                b_h.release(1)
            if gsel is not None:  # (the pass's GATE object: two units, the first read)
                u = b_h.acquire(1)
                gsel[0](u, ws, gsel[1])
                b_h.release(1)
                b_h.acquire(1)
                b_h.release(1)
            if seg is not None:  # (the pass's SEG object: two units, the first read)
                u = b_h.acquire(1)
                seg(u, ws, row)
                b_h.release(1)
                b_h.acquire(1)
                b_h.release(1)
            if win:
                init(ws, wo, p8.read(), bg, bb, fq.read(), hk.read())
            elif self.nc:
                init(ws, wo, p8.read(), bg, bb, hk.read())
            else:
                init(ws, wo, p8.read(), bg, bb)
            if self.swa:  # (the window's groups and the pass's)
                n = gq * (self.swa // s.pg + 1)
            else:
                n = fq.read()  # (read after this dispatch's first object: the host wrote it before the fills)
            if self.swa or self.nc:  # (non-causal: the prompt's groups, every pass)
                pass
            elif self.ring:
                n = gq * (self.ring + win + 1)
            elif full:
                n = plus_i32(n, gq * win)
            else:
                for _ in range(gq):
                    n = n + bg
            for t in range32(n):  # (a chunk: ``qk`` of its first K unit starts it, ``pv`` of its first V half the softmax)
                if s.kp == 1:  # (its two K units at once)
                    k = a_h.acquire(2)
                    qk(q, k[0], k[1], ws, t, row)
                    a_h.release(2)
                else:
                    for kk in range(ku):
                        k = a_h.acquire(1)
                        qk(q, k, ws, kk, t, row)
                        a_h.release(1)
                for uu in range(2):
                    v = b_h.acquire(1)
                    pv(v, ws, wo, pp, uu)
                    b_h.release(1)
            for h in range(qh):
                o = o_h.acquire(1)
                fin(ws, wo, o, h)
                o_h.release(1)

        npg = n_pass // m  # (the chunk's groups)

        def core(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, *extra):
            seg, gs = (extra[:-2], extra[-2:]) if s.gate else (extra, None)
            if self.ring:  # (every pass the held history and the whole ring)
                for _ in forever_():
                    for bg in range_(npg):
                        turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, 0,
                             2 * rb * s.qp, *seg)
                return
            for _ in forever_():
                # (packed: the passes whose window is every group of the chunk after its first, then the rest's last
                # win)
                for bg in range_(min(npg, win + 1) if win else npg):
                    for bb in range(m):  # (the pass's units to every core: each keeps its own)
                        turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, bb,
                             2 * rb * s.qp * (s.per if self.bc else 1), *seg, gsel=gs)
                if win and npg > win + 1:
                    for bg in range_(win + 1, npg):
                        for bb in range(m):
                            turn(a_h, b_h, o_h, init, qsel, qk, pv, fin, ws, wo, pp, q, p8, fq, me, row, bg, bb,
                                 2 * rb * s.qp, *seg, full=True)

        def src(c: int) -> int:  # (the column whose streams column c's cores take)
            return c - c % s.per if self.bc else c

        def jb(c: int) -> int:  # (column c's place among a broadcast's columns)
            return c % s.per if self.bc else 0

        workers = [Worker(core, [a1[src(c)].cons(depth=2), b1[src(c)].cons(), o1[c][r].prod(), k_init, k_qsel, k_qk, k_pv,
                                 k_fin, Buffer(ws_ty, name=f"la_ws{c}_{r}"),
                                 Buffer(wo_ty, name=f"la_wo{c}_{r}"), Buffer(pp_ty, name=f"la_pp{c}_{r}"),
                                 Buffer(q_ty, name=f"la_q{c}_{r}"), p_p8, p_fq,
                                 rb * (r + 4 * jb(c)), r,
                                 *([k_seg, ScratchpadParameter(P_HK, np.int32)] if win or self.nc else []),
                                 *([k_gsel, r + 4 * jb(c)] if s.gate else [])],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        n_qs, n_kv, n_a, NH = self.n_qs, self.n_kv, self.n_a, COLS * qh
        uw, hb, g2, o2, ow = s.qunit // 2, self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2
        rows, sg = ow // 576, self.sub // 8  # (a sub-chunk's token groups)
        if s.kp == 2:  # (a group's [K obj | padding | K obj])
            n2 = (s.a_bytes + s.used) // 2
            k_tap = [1, 1, 16, n2 // 16], [0, g2, n2 // 16, 1]
        else:
            k_tap = [1, 1, rows, 576], [0, g2, 576, 1]

        n_seg = self.n_seg

        def seq(QS, KV, A, *rest):
            *seg, ah, bh, oh = rest  # (packed: SEG)

            def kv_part(off: int, n: int = 0, *, sizes: list[int], strides: list[int]) -> ChainPart:
                """A KV head's groups from ``off`` (bf16): the history's (P_FG groups, n = 0) or n of the chunk's from
                P_FO on (sizes [1, 1, d1, d0]: a group's part)."""
                if n:
                    return ChainPart(KV, TensorAccessPattern((n_kv,), off, [1, n, *sizes[2:]], strides),
                                     offset_parameter=p_fo)
                return ChainPart(KV, TensorAccessPattern((n_kv,), off, [1, 2, *sizes[2:]], strides),
                                 length_parameter=p_fg, length_unit=sizes[2] * sizes[3])

            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):
                ctrl = self.CTRL
                if "memstall" in self.ablate:
                    ctrl = ((ctrl[0][0], 33 | 34 << 8 | 23 << 16 | 23 << 24), ctrl[1])
                mc = [int(a[3:]) for a in self.ablate if a.startswith("mc_")]  # (memory-module events, e.g. bank conflicts)
                if mc:
                    ctrl = (*ctrl, (0x11000, mc[0] | mc[0] << 8 | mc[-1] << 16 | mc[-1] << 24))
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in ctrl:
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):  # pass b: token groups 4 qb b .. of the column's heads (a pass's 4 cores joined)
                oh[c].drain(A, TensorAccessPattern((n_a,), c * qh * nd * 36, [n_pass, qh, 4 * qb, nd * 36],
                                                   [4 * qb * NH * nd * 36, nd * 36, NH * nd * 36, 1]),
                            group=tg, wait=True)
            if self.ring:
                ring_fills(QS, KV, seg[0], ah, bh, tg)
                return
            # (a pass's fills: at most three in flight, within the shim tiles' BDs; packed or gated: two, their SEG / GATE
            # parts more)
            inflight = []
            tw, gw2, wg = s.tgu // 2, s.gw // 2, self.swa // s.pg  # (QS's token groups; a group's gates; window groups)
            for b in range(n_pass):
                tp = TaskGroup()
                ng = b // m  # (the chunk's groups after its first that the pass streams; packed: the last win of them)
                j0, nw = (max(1, ng - win + 1), min(ng, win)) if win else (1, 0 if self.nc else ng)
                t0 = 4 * qb * b  # (the pass's first token group: its sub-chunk, the group in it)
                q0 = t0 // sg * sg * COLS * tw + t0 % sg * tw
                for i, c in enumerate(scols):
                    h = c // s.per  # (the column's KV head)
                    k0, v0 = h * hb + self.kv0 // 2, h * hb + self.kv0 // 2 + s.kp * o2
                    if self.swa:  # (the window's groups before the pass's and its own, from the guard on)
                        w0 = (ng - wg) * g2
                        parts = [ChainPart(KV, TensorAccessPattern((n_kv,), k0 + w0, [1, wg + 1, *k_tap[0][2:]],
                                                                   [0, *k_tap[1][1:]]), offset_parameter=p_sw)]
                    else:
                        parts = [kv_part(k0, sizes=k_tap[0], strides=k_tap[1])]
                        if nw:
                            parts.append(kv_part(k0 + (j0 - 1) * g2, nw, sizes=k_tap[0], strides=k_tap[1]))
                    chained_fill(ah[i], KV, parts, tp, wait=True)
                    qc, nc = q0 + c * sg * tw, s.per if self.bc else 1  # (the stream's columns: their Q units, gates)
                    if s.gate:  # (the pass's Q units, then its GATE object: its token groups' gates, padding to 2 units)
                        parts = [ChainPart(QS, TensorAccessPattern((n_qs,), qc, [1, nc, 4 * qb, qh * uw],
                                                                   [0, sg * tw, tw, 1])),
                                 ChainPart(QS, TensorAccessPattern((n_qs,), qc + qh * uw, [1, nc, 4 * qb, gw2],
                                                                   [0, sg * tw, tw, 1])),
                                 ChainPart(QS, TensorAccessPattern((n_qs,), 0, [1, 1, 1, s.unit - nc * 4 * qb * gw2],
                                                                   [0, 0, 0, 1]))]
                    else:
                        parts = [ChainPart(QS, TensorAccessPattern((n_qs,), qc, [1, 1, 4 * qb * qh, uw], [0, 0, uw, 1]))]
                    if seg:
                        parts.append(ChainPart(seg[0], TensorAccessPattern((n_seg,), b * s.unit,
                                                                           [1, 1, s.unit // 576, 576], [0, 0, 576, 1])))
                    if self.swa:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), v0 + w0, [1, wg + 1, rows, 576],
                                                                       [0, g2, 576, 1]), offset_parameter=p_sw))
                    else:
                        parts.append(kv_part(v0, sizes=[1, 1, rows, 576], strides=[0, g2, 576, 1]))
                        if nw:
                            parts.append(kv_part(v0 + (j0 - 1) * g2, nw, sizes=[1, 1, rows, 576],
                                                 strides=[0, g2, 576, 1]))
                    chained_fill(bh[i], KV, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > (1 if win or s.gate else 2):  # (a GATE object: two parts more)
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        def ring_fills(QS: object, KV: object, SEG: object, ah: list, bh: list, tg: TaskGroup) -> None:
            """The ring variant's fills: per column the held history (groups from position 0, the chunk's first among
            them) once, then pass b's group b (stream A: its K; B: the pass's Q units, SEG, its V), each read once."""
            p_g0, hold = ScratchpadParameter(P_G0, np.int32), self.ring
            hist = TaskGroup()
            for c in range(COLS):  # (a group's K: its two objects' records, without the first one's padding)
                k0, v0 = c * hb, c * hb + s.kp * o2
                chained_fill(ah[c], KV, [TensorAccessPattern((n_kv,), k0, [hold, 2, rows, 576], [g2, o2, 576, 1])],
                             hist, wait=True)
                chained_fill(bh[c], KV, [TensorAccessPattern((n_kv,), v0, [hold, 1, rows, 576], [g2, 0, 576, 1])], hist,
                             wait=True)
            inflight = [hist]
            for b in range(n_pass):
                tp = TaskGroup()
                t0 = 4 * qb * b
                q0 = t0 // sg * sg * COLS * qh * uw + t0 % sg * qh * uw
                for c in range(COLS):
                    k0, v0 = c * hb + b * g2, c * hb + s.kp * o2 + b * g2
                    chained_fill(ah[c], KV, [ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 2, rows, 576],
                                                                               [0, o2, 576, 1]), offset_parameter=p_g0)],
                                 tp, wait=True)
                    chained_fill(bh[c], KV, [
                        ChainPart(QS, TensorAccessPattern((n_qs,), q0 + c * sg * qh * uw, [1, 1, 4 * qb * qh, uw],
                                                          [0, 0, uw, 1])),
                        ChainPart(SEG, TensorAccessPattern((n_seg,), b * s.unit, [1, 1, s.unit // 576, 576],
                                                           [0, 0, 576, 1])),
                        ChainPart(KV, TensorAccessPattern((n_kv,), v0, [1, 1, rows, 576], [0, g2, 576, 1]),
                                  offset_parameter=p_g0)], tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 2:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           *([np.ndarray[(n_seg,), bf]] if win or self.nc else []),
                           [a3[c].prod(tile=Tile(c, 0)) for c in scols],
                           [b3[c].prod(tile=Tile(c, 0)) for c in scols],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        if self.ring:
            self._ring_memtiles(module)
        return module

    def _ring_memtiles(self, module: Module) -> None:
        """The ring variant's memory tiles (``pin_l1_buffers``' split module): per column the forwards of streams A and B
        replaced by buffers of their own (the held history and a ring of win + 1 groups after it, the pass's Q units
        and SEG) and tasks the runtime sequence starts once per dispatch (a runtime BD: three dimensions and its
        iteration). A writes the history, then each pass's group into slot b % R (a group's K records only: the fills
        leave out the padding); it sends the history and the whole ring every pass (the cores skip the slots outside
        their window). B writes its history, then per pass the Q units and SEG, then the group's V, each object of two
        units re-laid as stream B's walk sends it (la_v's to_stream, inverted: a BD per object); it sends them in order.
        A slot is written again once the pass before the last that reads it ended (credits: 2). The tasks end with the
        dispatch's last pass (freed after the drains)."""
        from aie import ir
        from aie.dialects import aie
        from aie.dialects.aiex import dma_configure_task_for, dma_free_task, dma_start_task

        s, R, hold, npg = self.shape, self.win + 1, self.ring, self.n_pass
        ob = s.unit  # (bf16 of an object of two units)
        ks, vs = s.used, s.used // 2  # (bf16 of a group's K, V)
        opg = vs // ob  # (V objects per group)
        nqs = (2 * self.qb * s.qh * s.qp + 2) * ob // 2  # (bf16 of a pass's Q units and SEG)
        walk = [(4, 72), (s.nd // 2, 288), (72, 1)]  # (an object's cache order [slice][d-block] into la_v's stream order)
        with module.context, ir.Location.unknown():
            dev = next(o for o in module.body.operations if isinstance(o, aie.DeviceOp))
            blk = dev.regions[0].blocks[0]
            ops = {ir.StringAttr(o.attributes["sym_name"]).value: o for o in blk.operations
                   if "sym_name" in o.attributes}
            tiles = {(o.col.value, o.row.value): o.result for o in blk.operations if isinstance(o, aie.TileOp)}
            routes = [o for o in blk.operations if isinstance(o, aie.RouteOp)]
            seq = next(o for o in blk.operations if isinstance(o, aie.RuntimeSequenceOp))
            bf = ir.BF16Type.get()
            ends: dict[tuple[int, str], str] = {}
            locks: dict[tuple[int, str], object] = {}
            bufs: dict[tuple[int, str], object] = {}
            ip = ir.InsertionPoint(seq.operation)
            for c in range(COLS):
                mt = tiles[(c, 1)]
                for src, fwd, st in ((f"la_ka{c}", f"la_k{c}", "a"), (f"la_vb{c}", f"la_v{c}", "b")):
                    for name in (f"{src}__cons__pool", f"{src}__cons__dma", f"{fwd}__prod__dma"):
                        ops[name].erase()
                    for d in ("in", "out"):
                        ends[(c, st + d)] = f"rg_{st}{d}{c}"
                        aie.route_endpoint(ends[(c, st + d)], mt, aie.WireBundle.DMA, ip=ip)
                    for r in routes:
                        dst = [ir.FlatSymbolRefAttr(a).value for a in ir.ArrayAttr(r.attributes["destinations"])]
                        srcs = [ir.FlatSymbolRefAttr(a).value for a in ir.ArrayAttr(r.attributes["sources"])]
                        if dst == [f"{src}__cons__dma"]:
                            r.attributes["destinations"] = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(ends[(c, st + "in")])])
                        if srcs == [f"{fwd}__prod__dma"]:
                            r.attributes["sources"] = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(ends[(c, st + "out")])])
                for name, n in (("k", (hold + R) * ks), ("q", nqs), ("v", (hold + R) * vs)):
                    bufs[(c, name)] = aie.BufferOp(ir.MemRefType.get([n], bf), mt, sym_name=f"rg_{name}{c}", ip=ip).result
                for name, init in (("k_ready", 0), ("k_free", 2), ("q_ready", 0), ("q_free", 1), ("q_mid", 0),
                                   ("v_ready", 0), ("v_free", 2), ("v_mid", 0)):
                    locks[(c, name)] = aie.lock(mt, init=init, sym_name=f"rg_{name}{c}", ip=ip)
            acq, rel = aie.LockAction.AcquireGreaterEqual, aie.LockAction.Release
            tasks = []

            def task(end: str, parts: list[tuple], reps: int) -> None:
                """A memory tile task of ``reps`` runs of the BD chain ``parts``: (buffer, offset, length, dims or None,
                the lock acquired and the lock released or None, iteration (steps, stride) or None)."""
                t = dma_configure_task_for(end, repeat_count=reps - 1)
                with aie.bds(t) as bd:
                    for i, (buf, off, n, dims, lk, it) in enumerate(parts):
                        with bd[i]:
                            if lk is not None:
                                aie.use_lock(lk[0], acq, 1)
                            kw = {"sizes": [d for d, _ in dims], "strides": [st for _, st in dims]} if dims else {}
                            op = aie.dma_bd(buf, offset=off, transfer_len=n, **kw)
                            if it:
                                op = op if isinstance(op, ir.OpView) else op.owner
                                op.attributes["iteration"] = ir.Attribute.parse(
                                    f"#aie.bd_iteration<size = {it[0]}, stride = {it[1]}, current = 0>")
                            if lk is not None:
                                aie.use_lock(lk[1], rel, 1)
                            if i + 1 < len(parts):
                                aie.next_bd(bd[i + 1])
                            else:
                                aie.EndOp()
                dma_start_task(t)
                tasks.append(t)

            first = seq.regions[0].blocks[0]
            with ir.InsertionPoint.at_block_begin(first):
                for c in range(COLS):
                    B = lambda n, c=c: bufs[(c, n)]
                    L = lambda n, c=c: locks[(c, n)]
                    task(ends[(c, "ain")], [(B("k"), 0, hold * ks, None, None, None)], 1)
                    task(ends[(c, "ain")], [(B("k"), hold * ks, ks, None, (L("k_free"), L("k_ready")), (R, ks))], npg)
                    task(ends[(c, "aout")], [(B("k"), 0, (hold + R) * ks, None, (L("k_ready"), L("k_free")), None)], npg)
                    task(ends[(c, "bin")], [(B("v"), 0, ob, walk, None, (hold * opg, ob))], hold * opg)
                    qs = [(B("q"), k * ob, ob, walk, None, None) for k in range(nqs // ob)]
                    qs[0] = (*qs[0][:4], (L("q_free"), L("q_mid")), None)
                    qs[-1] = (*qs[-1][:4], (L("q_mid"), L("q_ready")), None)
                    vv = [(B("v"), hold * vs + k * ob, ob, walk, None, (R, vs)) for k in range(opg)]
                    if opg == 1:
                        vv[0] = (*vv[0][:4], (L("v_free"), L("v_ready")), (R, vs))
                    else:
                        vv[0] = (*vv[0][:4], (L("v_free"), L("v_mid")), (R, vs))
                        vv[-1] = (*vv[-1][:4], (L("v_mid"), L("v_ready")), (R, vs))
                    task(ends[(c, "bin")], qs + vv, npg)
                    task(ends[(c, "bout")], [(B("q"), 0, nqs, None, (L("q_ready"), L("q_free")), None),
                                             (B("v"), 0, (hold + R) * vs, None, (L("v_ready"), L("v_free")), None)], npg)
            with ir.InsertionPoint(first):  # (after the drains' awaits: the tasks are done by then)
                for t in tasks:
                    dma_free_task(t)

    def _layout2(self) -> dict[str, list[int]] | None:
        """The layout of o in two copies (``l1_layout``), None where it does not fit."""
        s, R = self.shape, 0x4000
        rows, u = self.qb * s.qh, s.unit
        q, out, ws = rows * s.qunit, self.qb * s.nd * 72, self.workspace()["size"]
        r0 = self.STACK + q + 2 * out
        if self.o_bytes > R or r0 + self.pp_bytes > R or 3 * R + ws + 4 * u > L1:
            return None
        return {"la_q": [self.STACK], "la_or": [self.STACK + q, self.STACK + q + out], "la_pp": [r0], "la_wo": [R],
                "la_ws": [3 * R], "la_k": [3 * R + ws, 3 * R + ws + u], "la_v": [3 * R + ws + 2 * u, 3 * R + ws + 3 * u]}

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (the stack at 0). With o in two copies (``oc`` 2, where L1 holds them): the
        stack, Q, the output buffers and P in the first region, o's copies in the second and third (P V reads one and
        writes the other: a region each), the workspace and the K and V objects in the last; every hot loop's streams
        then lie in distinct regions (Q K: Q | K; the softmax: sc | P; P V: P | V | o | o'). Else the stack, Q and the
        output buffers (touched by ``fin`` only) in the first (Q K streams Q against K, whose objects sit with the
        workspace in the last), P there too where it fits, O from the second on (P V: O against P and V in stream B's
        objects after O)."""
        if self.oc == 2:
            return self._layout2()
        s, R = self.shape, 0x4000
        rows, u = self.qb * s.qh, s.unit
        q, wo, ws, out = rows * s.qunit, 64 * s.nd * rows * 4, self.workspace()["size"], self.qb * s.nd * 72
        lay = {"la_q": [self.STACK], "la_or": [self.STACK + q, self.STACK + q + out], "la_wo": [R]}
        r0 = self.STACK + q + 2 * out
        if r0 + self.pp_bytes <= R:
            lay["la_pp"], pp = [r0], 0
        else:
            pp = self.pp_bytes  # (P after the workspace)
        pos = R + wo
        lay["la_v"], pos = [pos, pos + u], pos + 2 * u
        top = -(-pos // R) * R  # (the workspace from the next region on, stream A's objects after it)
        if top + ws + pp + 2 * u > L1:
            lay["la_k"], pos = [pos, pos + u], pos + 2 * u
            top = max(pos, L1 - ws - pp)
            lay["la_ws"] = [top]
        else:
            lay["la_ws"], lay["la_k"] = [top], [top + ws + pp, top + ws + pp + u]
        if pp:
            lay["la_pp"] = [top + ws]
        assert self.STACK + q + 2 * out <= R and max(v[-1] + n for v, n in ((lay["la_ws"], ws + pp), (lay["la_k"], u))) <= L1, lay
        return lay
