"""The attention of a prefill on the decode's KV record format (``overlay.kv``), one column per KV head (Llama): a
T-token chunk at position p0 of one layer, from the qkv GEMMs' output G (``prefill.gemm``: per column c [q of its qh
heads | k | v] of KV head c) to the O GEMM's A panels, with the chunk's K / V records in the decode's cache, so the
decode continues from them. A KV head's cache is groups of kp + 1 objects of the image's tile size (kp K objects of
records of 8 positions, kp = 2: hi + lo panels, 1: single; the V object), linear in the positions; a chunk starts at a
group (``Shape.pg`` positions: 128 at head_dim 64, 64 at 128). RoPE on the pairs (i, i + hd / 2) (the image's Q / K rows
in rotate-half order), scores in log2 units: q k log2(e) / sqrt(hd).

``PrepOperator``: per column c, cores (c, 2) and (c, 3) the Q heads (the first half of the column's heads, the rest),
(c, 4) K, (c, 5) V. Per 8 tokens a Q core turns each of its heads' rows into the attention's operands: q rotated (fp32)
times the score constant, as single bfp16 panels of 8 queries x 8 dims (ND panels: a Q unit); QS holds per token group
and column the qh heads' units. The K core writes K rotated (fp32) as the cache's K
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
its own Q units). A unit is 144 ND bytes: a hi + lo K record (two single ones), two V records, two row blocks' Q units.
Per chunk of CR records (QB, CR: the most that L1 holds, ``Shape.blocks``):

- the scores on the MMUL, the accumulators starting at the rows' bias: per pair of row blocks one loop of two panel
  streams (Q | K0, K1: four chains, MMUL-bound; hi + lo K: a loop for K hi, then one for K lo);
- the softmax with a running maximum per row, raised (o, l rescaled) only when the chunk's maximum exceeds it by 2^16
  (qwen35's ``attn_prefill``; the chunk's maxima in bf16: m is only the reference of 2^(s - m)); masks only on a
  diagonal chunk; P = 2^(s - m) times 2^64 (o / l cancels it) by ``km.exp2``, clamped at 2^-63 of that (masked scores
  -1e30), as single bfp16 panels (half the P V MMULs of hi + lo). Known error, cause open: the keys below a row's
  reference weigh ~0.4% too much (test_llama_npu_teacher_forced[Q3_K_L-1B-hilo] fails);
- o += P V per pair of row blocks (each V panel popped once for both: two MMUL chains), and the row sums l += P J on
  the MMUL (J a ones panel: no accumulator in the exp2 loop);

chunks past every one of the core's queries are skipped (their objects consumed). At the pass's end o / l per head as
the O GEMM's A panels. QB = 2 (64-query passes: half the cache re-reads) where L1 holds it (1B, 3B), else 1 (8B).
Cache-bound at long contexts:
3B 15.1 ms per layer for 2048 tokens at p0 = 8K, 13.1 with the compute compiled out (the history re-read per pass at
~52 GB/s); ~740 cycles per record and core.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
import torch
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

from ..base import JINJA_ENV, ArgSpec, ArtcOperator
from ..design import COLS, range32, stream_walk
from ..gemm.gemm_q import ChainPart, chained_fill
from ..kernel_source import kernel_prefix
from ..l1_layout import pin_l1_buffers
from ..overlay.kv import Records
from ..util import Attr, bf16_hilo, regions
from .gemm import Layout

# scratchpad parameters
P_CS = "pa_cs"  # bf16 offset of the chunk's rows in the cos / sin table (2 hd p0)
P_KV = "pa_kv"  # bf16 offset of the chunk's first cache group in a KV head's cache
P_P8 = "pa_p8"  # p0 / 8 (the attention cores' masks)
P_FG = "pa_fg"  # the cache groups every pass streams first (p0 / pg + 1: the history and the chunk's first)
P_FQ = "pa_fq"  # their chunks of CR records (the cores' count)
P_FO = "pa_fo"  # bf16 offset of the chunk's second group
L1 = 0x10000 - 512  # a core's data memory for its buffers (a margin for aiecc's)


@dataclass(frozen=True)
class Shape:
    """The attention's shapes: head_dim, query heads per KV head (a column's), K panels per record, the cache's object
    bytes."""

    hd: int
    qh: int
    kp: int
    a_bytes: int

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
    def qunit(self) -> int:  # bytes of a row block's Q panels (single bfp16: 8 queries x 8 dims per panel)
        return 72 * self.nd

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
                + ws)

    @cached_property
    def blocks(self) -> tuple[int, int]:
        """(QB, CR): the most row blocks per head that fit L1 (2 or 1), chunks of 4 records."""
        return next((qb, 4) for qb in (2, 1) if self.l1(qb, 4) <= L1)


def cs_table(inv_freq: np.ndarray, n: int) -> np.ndarray:
    """(n, hd) fp32: per position [cos hd / 2 | sin hd / 2] of the pairs' angles position inv_freq[i] (the decode's
    ``overlay.attn.rope_tables``)."""
    th = np.outer(np.arange(n, dtype=np.float64), np.asarray(inv_freq, np.float64))
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


_PREP = JINJA_ENV.from_string(r"""
// The prep cores' workspace (bytes): row (HD fp32) | stage [db ND][row 8][8] fp32
#define PW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// x (HD bf16) -> row (fp32), the pairs (i, i + HD / 2) rotated by cs [cos HD / 2 | sin HD / 2]
__attribute__((noinline)) void {{ p }}_rope(const bfloat16 *__restrict x, const float *__restrict cs, float *__restrict row)
{
    for (int i = 0; i < {{ HD }}; i += 32)
        aie::store_v(row + i, aie::accum<accfloat, 32>(aie::load_v<32>(x + i)).template to_vector<float>());
    for (int i = 0; i < {{ HD // 2 }}; i += 32) {
        const aie::vector<float, 32> a = aie::load_v<32>(row + i), b = aie::load_v<32>(row + {{ HD // 2 }} + i);
        const aie::vector<float, 32> c = aie::load_v<32>(cs + i), s = aie::load_v<32>(cs + {{ HD // 2 }} + i);
        aie::accum<accfloat, 32> o0, o1;
        al::rot2(a, b, c, s, o0, o1);
        aie::store_v(row + i, o0.template to_vector<float>());
        aie::store_v(row + {{ HD // 2 }} + i, o1.template to_vector<float>());
    }
}

// row (HD fp32) -> row r of the stage [db][r][8]
__attribute__((noinline)) void {{ p }}_scatter(const float *__restrict row, float *__restrict st, int r)
{
    float *__restrict d = st + 8 * r;
    for (int db = 0; db < {{ ND }}; db++) {
        aie::store_v(d, aie::load_v<8>(row));
        row += 8;
        d += 64;
    }
}

// the stage -> [hi panels ND | lo panels ND]
__attribute__((noinline)) void {{ p }}_hilo(const float *__restrict st, uint8_t *__restrict out)
{
    const al::BV nI = al::neg_eye();
    al::BOut<> ho((bfp16ebs8 *)out);
    al::BOut<> lo((bfp16ebs8 *)(out + {{ 72 * ND }}));
    for (int db = 0; db < {{ ND }}; db++) {
        al::BV h, l;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
        ho.push(h);
        lo.push(l);
        st += 64;
    }
}

// 8 tokens of head h of the core's nq (rows of nq HD bf16), cs (8 x [cos | sin] fp32) -> unit h of out: the Q panels
// (single bfp16) of q rotated, times the score constant
__attribute__((noinline)) void {{ p }}_qh(const bfloat16 *__restrict q, const bfloat16 *__restrict cs_, float *__restrict ws,
                                          bfloat16 *__restrict out, int32_t h, int32_t nq)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    for (int r = 0; r < 8; r++) {
        {{ p }}_rope(q + {{ HD }} * (nq * r + h), cs + {{ HD }} * r, row);
        for (int i = 0; i < {{ HD }}; i += 64)
            aie::store_v(row + i, al::mul_hilo<64, {{ qc[0] }}f, {{ qc[1] }}f>(aie::load_v<64>(row + i)));
        {{ p }}_scatter(row, st, r);
    }
    al::BOut<> o((bfp16ebs8 *)((uint8_t *)out + {{ QUNIT }} * h));
    for (int db = 0; db < {{ ND }}; db++) {
        o.push(aie::accum<accfloat, 64>(aie::load_v<64>(st)).template to_vector<bfp16ebs8>());
        st += 64;
    }
}

}  // namespace

extern "C" {

{% for i in range(2) %}
// the Q core {{ i }}'s (its own buffer sizes)
void {{ p }}_q{{ i }}(const bfloat16 *__restrict q, const bfloat16 *__restrict cs, float *__restrict ws, bfloat16 *__restrict out,
                int32_t h, int32_t nq)
{
    {{ p }}_qh(q, cs, ws, out, h, nq);
}

{% endfor %}
// 8 tokens of the KV head's K (rows of HD bf16), cs -> record r of the group's K objects in out
void {{ p }}_k(const bfloat16 *__restrict k, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out,
               int32_t r)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    for (int t = 0; t < 8; t++) {
        {{ p }}_rope(k + {{ HD }} * t, cs + {{ HD }} * t, row);
        {{ p }}_scatter(row, st, t);
    }
    uint8_t *__restrict rec = (uint8_t *)out + {{ KREC }} * r;
{% if KP == 2 %}
    {{ p }}_hilo(st, rec);
{% else %}
    al::BOut<> o((bfp16ebs8 *)rec);
    for (int db = 0; db < {{ ND }}; db++) {
        o.push(aie::accum<accfloat, 64>(aie::load_v<64>(st)).template to_vector<bfp16ebs8>());
        st += 64;
    }
{% endif %}
}

// 8 tokens of the KV head's V (rows of HD bf16) -> record r of the group's V object in out: per 8 dims a panel of
// blocks over the positions
void {{ p }}_v(const bfloat16 *__restrict v, bfloat16 *__restrict out, int32_t r)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BOut<> o((bfp16ebs8 *)((uint8_t *)out + {{ 72 * ND }} * r));
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        aie::vector<bfloat16, 64> pd;  // [position][dim]
        AIE_LOOP_UNROLL_FULL
        for (int t = 0; t < 8; t++)
            pd.insert(t, aie::load_v<8>(v + {{ HD }} * t + 8 * db));
        o.push(aie::accum<accfloat, 64>(aie::transpose(pd, 8, 8)).template to_vector<bfp16ebs8>());
    }
}

}  // extern "C"
""")


class PrepOperator(ArtcOperator):
    """The attention prep of T tokens (module docstring). Runtime sequence: G (the qkv GEMM's output, ``layout``), CS
    (``cs_table`` of every position, as bf16 pairs), QS (per token group and column the heads' Q units; a sub-chunk's
    part of a chunk's), KV (the layer's cache: COLS KV heads of ``kv_bytes`` each, ``kv_pad`` bytes past them).
    Parameters ``params``."""

    STACK = 1024
    pfx = kernel_prefix("pap")

    def __init__(self, *, T: int, shape: Shape, layout: Layout, kv_bytes: int, cs_positions: int, kv_pad: int = 0):
        s = shape
        assert T % s.pg == 0 and T // s.pg <= 64 and {"q", "k", "v"} <= set(layout.off)
        assert s.qh in (3, 4) and s.used % 1152 == 0
        self.T, self.shape, self.layout, self.kv_bytes, self.cs_positions = T, s, layout, kv_bytes, cs_positions
        self.kv_pad = kv_pad
        self.nq = [-(-s.qh // 2), s.qh // 2]  # (the Q cores' heads)
        super().__init__()

    def workspace(self) -> dict[str, int]:
        hd = self.shape.hd
        return regions([("row", 4 * hd), ("stage", 32 * hd)])

    def get_kernel_src(self) -> str:
        p = self.pfx
        s = self.shape
        return _PREP.render(p=p, HD=s.hd, ND=s.nd, KP=s.kp, KREC=s.rec.k_bytes, QUNIT=s.qunit, qc=bf16_hilo(s.scale),
                            w=Attr(self.workspace()))

    @property
    def n_cs(self) -> int:
        return self.cs_positions * 2 * self.shape.hd

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument: [column][token group][head][Q unit]."""
        return self.T // 8 * COLS * self.shape.qh * self.shape.qunit // 2

    @property
    def n_kv(self) -> int:
        return (COLS * self.kv_bytes + self.kv_pad) // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.T * self.layout.N,), torch.bfloat16), ArgSpec("in", (self.n_cs,), torch.bfloat16),
                ArgSpec("out", (self.n_qs,), torch.bfloat16), ArgSpec("out", (self.n_kv,), torch.bfloat16)]

    def params(self, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 (a group start)."""
        s = self.shape
        assert p0 % s.pg == 0 and p0 + self.T <= self.cs_positions
        return {P_CS: 2 * s.hd * p0, P_KV: s.group // 2 * (p0 // s.pg)}

    def get_mlir_artifact(self) -> Module:
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

        def core_q(a_h, b_h, o_h, fn, ws, n):
            for _ in forever_():
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    for h in range(n):
                        fn(x, cs, ws, out, h, n)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        def core_k(a_h, b_h, o_h, fn, ws):
            for _ in forever_():
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
                                               Buffer(ws_ty, name=f"lp_ws{c}_{i}"), nq[i]],
                                      tile=Tile(c, 2 + i), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
            workers.append(Worker(core_k, [a1[c][2].cons(), b[c].cons(), g1[c][0].prod(), k_k,
                                           Buffer(ws_ty, name=f"lp_ws{c}_k")],
                                  tile=Tile(c, 4), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                  while_true=False))
            workers.append(Worker(core_v, [a1[c][3].cons(), g1[c][1].prod(), k_v], tile=Tile(c, 5),
                                  stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv, n_qs = self.n_cs, self.n_kv, self.n_qs
        uw, qw = s.qunit // 2, s.qh * s.qunit // 2  # (bf16: a Q unit, a column's units of a token group)
        hb, g2, o2, ow = self.kv_bytes // 2, s.group // 2, s.a_bytes // 2, s.used // 2

        def seq(G, CS, QS, KV, ah, bh, oh, gh):
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
                bh[c].fill(CS, TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 2 * hd], [0, 0, 0, 1]),
                           offset_parameter=p_cs, group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * n_tg * qw, [1, n_tg, s.qh, uw], [0, qw, uw, 1]),
                            group=tg, wait=True)
                # per group its objects' records (rows of 576 elements: a shim BD's inner dimension <= 1023 words)
                gh[c].drain(KV, TensorAccessPattern((n_kv,), c * hb, [n_grp, s.objects, ow // 576, 576],
                                                    [g2, o2, 576, 1]), offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(n_qs,), bf],
                           np.ndarray[(n_kv,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(g3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_ATTN = JINJA_ENV.from_string(r"""
// The attention core's buffers: wo = o [row block RB][db ND][8 x 8] fp32 (row block j QH + h: block j of head h; pairs
// of row blocks 2 p, 2 p + 1); q: per pair its Q panels [d-block pair ND / 2][hi 2 p, lo 2 p, hi 2 p + 1, lo 2 p + 1]
// [d-block 2]; ws (bytes): sc [pair][record CR][row block 2][8 x 8] fp32 (A = s + 192 - m: s the scores in log2 units,
// lane 8 q + k; m the row's reference maximum), bias [row block][64] (192 - m), l [row block][64] (P's row sums, in
// every lane of the row), pp [pair][record][row block 2] P panels, vm [64] bf16 (the chunk's maximum A per lane), st
// [16] int32 (the chunk's first record relative to
// the core's first block, skip, the pass's first record)
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

{# 2^(A - 128) of 32 lanes (A below 256: 2^(s - m) times 2^64, which o / l cancels), 2^1 below 128.5 (masked -1e30 too) #}
{% set exp2 = km.exp2(32, bias=192, shift=64, inline=True) %}
{{ exp2.src }}

namespace {

// P = 2^(A - 128) of the chunk's panels s (``exp2``: 2^(s - m) times 2^64) as bfp16 panels into pp, in their order
__attribute__((noinline)) void {{ p }}_ppan(const float *__restrict s, uint8_t *__restrict pp)
{
    al::BOut<> po((bfp16ebs8 *)pp);
    AIE_TRY_INITIATION_INTERVAL(20)  // (post-RA II 25; the default pre-RA schedule: 38, spilling)
    for (int i = 0; i < {{ RB * CR }}; i++) {
        const aie::accum<accfloat, 32> p0 = {{ exp2 }}(aie::accum<accfloat, 32>(aie::load_v<32>(s)));
        const aie::accum<accfloat, 32> p1 = {{ exp2 }}(aie::accum<accfloat, 32>(aie::load_v<32>(s + 32)));
        po.push(aie::concat(p0, p1).template to_vector<bfp16ebs8>());
        s += 64;
    }
}

// A diagonal chunk's masks, per row block b (its query block b / QH) and record k: rho = rho0 - b / QH + k, past the
// queries (rho > 0): all; rho = 0: keys after the query (lane 8 q + i, i > q)
__attribute__((noinline, minsize)) void {{ p }}_mask(float *__restrict sc, int32_t rho0)
{
    const aie::vector<float, 64> ninf = aie::broadcast<float, 64>(-1e30f);
    AIE_LOOP_NO_UNROLL
    for (int b = 0; b < {{ RB }}; b++) {
        float *__restrict s = sc + {{ 128 * CR }} * (b >> 1) + 64 * (b & 1);
        const int r = rho0 - (b >= {{ QH }}{% for j in range(2, QB) %} + b >= {{ QH * j }}{% endfor %});  // (block b / QH)
{% for k in range(CR) %}
        {
            const uint64_t all = -(uint64_t)(r + {{ k }} > 0), diag = -(uint64_t)(r + {{ k }} == 0);
            const aie::mask<64> mk = aie::mask<64>::from_uint64(all | (diag & 0x{{ tri }}ull));
            aie::store_v(s + {{ 128 * k }}, aie::select(aie::load_v<64>(s + {{ 128 * k }}), ninf, mk));
        }
{% endfor %}
    }
}

// Row blocks with a row whose chunk maximum exceeds its reference by 2^16 (rare, and at a pass's first chunk): each row
// whose maximum (bf16: an integer from 128 on) exceeds its reference takes it, m' - m = d >= 0 an integer (< 64: A <
// 256): o, l scaled by 2^-d = vexp2(64 - d) 2^-64 (vexp2 is exact at non-negative integers only), the bias and the
// chunk's A lowered by d (per half row block: 4 rows in 32 lanes)
__attribute__((noinline, minsize)) void {{ p }}_raise(float *__restrict ws, float *__restrict wo)
{
    for (int b = 0; b < {{ RB }}; b++) {
        for (int hf = 0; hf < 2; hf++) {
            float *__restrict s = AW(float, {{ w.sc }}) + {{ 128 * CR }} * (b >> 1) + 64 * (b & 1) + 32 * hf;
            aie::vector<bfloat16, 32> v = aie::accum<accfloat, 32>(aie::load_v<32>(s)).template to_vector<bfloat16>();
            for (int k = 1; k < {{ CR }}; k++)
                v = aie::max(v, aie::accum<accfloat, 32>(aie::load_v<32>(s + 128 * k)).template to_vector<bfloat16>());
            if (aie::lt(v, aie::broadcast<bfloat16, 32>(208.0f)).full())
                continue;
            const aie::vector<float, 32> am = al::max8(aie::accum<accfloat, 32>(v).template to_vector<float>());
            const aie::vector<float, 32> d = aie::max(aie::sub(am, aie::broadcast<float, 32>(192.0f)), aie::zeros<float, 32>());
            const aie::vector<bfloat16, 32> a = aie::mul(al::exp2_acc<32>(aie::accum<accfloat, 32>(
                aie::sub(aie::broadcast<float, 32>(64.0f), d))), aie::broadcast<bfloat16, 32>(0x1p-64f)).template to_vector<bfloat16>();
            float *__restrict o = wo + {{ 64 * ND }} * b + 32 * hf;
            for (int db = 0; db < {{ ND }}; db++)
                aie::store_v(o + 64 * db, al::mulc<32>(aie::load_v<32>(o + 64 * db), a).template to_vector<float>());
            float *__restrict pl = AW(float, {{ w.l }}) + 64 * b + 32 * hf, *__restrict pb = AW(float, {{ w.bias }}) + 64 * b + 32 * hf;
            aie::store_v(pl, al::mulc<32>(aie::load_v<32>(pl), a).template to_vector<float>());
            aie::store_v(pb, aie::sub(aie::load_v<32>(pb), d));
            for (int k = 0; k < {{ CR }}; k++)
                aie::store_v(s + 128 * k, aie::sub(aie::load_v<32>(s + 128 * k), d));
        }
    }
}

// o += P V of a V half (v: [d-block pair ND / 4][record CR][d-block 2]) for each pair of row blocks: per (pair, d-block
// pair) four accumulators (2 row blocks x 2 d-blocks) over the chunk's records, o read and written once (ol, os the
// same panels as distinct restrict parameters, each read before it is written, so the loop pipelines)
__attribute__((noinline)) void {{ p }}_pvh(const uint8_t *__restrict pp, const uint8_t *__restrict v, const float *__restrict ol_,
                                         float *__restrict os_)
{
    const float __aie_dm_resource_c *__restrict ol = (const float __aie_dm_resource_c *)ol_;
    float __aie_dm_resource_d *__restrict os = (float __aie_dm_resource_d *)os_;
    al::BInR<aie_dm_resource::a> ps((const bfp16ebs8 *)pp);
    al::BInR<aie_dm_resource::b> vs((const bfp16ebs8 *)v);
    dims_2d_t pat_p({{ ND // 4 - 1 }}, {{ -2 * CR }}, 0);  // (the pair's P per d-block pair, then the next pair's)
    dims_2d_t pat_v({{ ND // 4 - 1 }}, 0, {{ -ND // 4 * 2 * CR }});  // (the half's V once per pair)
    int cl = 0, cs = 0;
    for (int it = 0; it < {{ RB // 2 * ND // 4 }}; it++) {
        al::MM8 c00(aie::accum<accfloat, 64>(aie::load_v<64>(ol)));
        al::MM8 c01(aie::accum<accfloat, 64>(aie::load_v<64>(ol + 64)));
        al::MM8 c10(aie::accum<accfloat, 64>(aie::load_v<64>(ol + {{ 64 * ND }})));
        al::MM8 c11(aie::accum<accfloat, 64>(aie::load_v<64>(ol + {{ 64 * ND + 64 }})));
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < {{ CR }}; k++) {
            const al::BV P0 = ps.pop(), P1 = ps.pop(), V0 = vs.pop(), V1 = vs.pop();
            c00.mac(P0, aie::op_transpose(V0));
            c01.mac(P0, aie::op_transpose(V1));
            c10.mac(P1, aie::op_transpose(V0));
            c11.mac(P1, aie::op_transpose(V1));
        }
        aie::store_v(os, c00.template to_vector<float>());
        aie::store_v(os + 64, c01.template to_vector<float>());
        aie::store_v(os + {{ 64 * ND }}, c10.template to_vector<float>());
        aie::store_v(os + {{ 64 * ND + 64 }}, c11.template to_vector<float>());
        ol = add_2d_ptr(ol, {{ 128 * ND - 128 * (ND // 4 - 1) }}, {{ ND // 4 - 1 }}, cl, 128);
        os = add_2d_ptr(os, {{ 128 * ND - 128 * (ND // 4 - 1) }}, {{ ND // 4 - 1 }}, cs, 128);
        ps.seek(pat_p);
        vs.seek(pat_v);
    }
}

// l += P J of each pair of row blocks over the chunk's records (J a ones panel: every lane of a row its row sum)
__attribute__((noinline)) void {{ p }}_lsum(const uint8_t *__restrict pp, const float *__restrict ll, float *__restrict ls)
{
    const al::BV J = aie::accum<accfloat, 64>(aie::broadcast<float, 64>(1.0f)).template to_vector<bfp16ebs8>();
    al::BInR<aie_dm_resource::a> ps((const bfp16ebs8 *)pp);
    for (int pr = 0; pr < {{ RB // 2 }}; pr++) {
        al::MM8 e0(aie::accum<accfloat, 64>(aie::load_v<64>(ll)));
        al::MM8 e1(aie::accum<accfloat, 64>(aie::load_v<64>(ll + 64)));
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < {{ CR }}; k++) {
            const al::BV P0 = ps.pop(), P1 = ps.pop();
            e0.mac(P0, aie::op_transpose(J));
            e1.mac(P1, aie::op_transpose(J));
        }
        aie::store_v(ls, e0.template to_vector<float>());
        aie::store_v(ls + 64, e1.template to_vector<float>());
        ll += 128;
        ls += 128;
    }
}

}  // namespace

extern "C" {

// A pass's start: o = 0, the reference m = -48 (bias 240: P = 2^(s + 48), a chunk's maximum above -32 raises it; rows
// whose scores all stay below -111 would read 2^-63 for each), l = 0; the pass's first record p0 / 8 + {{ 4 * QB }}
// (M bg + bb)
__attribute__((minsize)) void {{ p }}_init(float *__restrict ws, float *__restrict wo, int32_t p8, int32_t bg, int32_t bb)
{
{% if "ev_q" in ablate %}
    event1();
{% endif %}
    al::zero<{{ RB * 64 * ND }}, 64>(wo);
    for (int b = 0; b < {{ RB }}; b++)
        aie::store_v(AW(float, {{ w.bias }}) + 64 * b, aie::broadcast<float, 64>(240.0f));
    al::zero<{{ RB * 64 }}, 64>(AW(float, {{ w.l }}));
    AW(int32_t, {{ w.st }})[2] = p8 + {{ 4 * QB }} * ({{ M }} * bg + bb);
}

// Chunk t of the pass's stream (records {{ CR }} t ..): its first record relative to the core's first query block
// (rho0: block j's record is rho0 - j), skipped when past every one of the core's queries
__attribute__((minsize)) void {{ p }}_grp(float *__restrict ws, int32_t t, int32_t r)
{
{% if "ev_chunk" in ablate %}
    event0();
{% endif %}
    int32_t *__restrict st = AW(int32_t, {{ w.st }});
    st[0] = {{ CR }} * t - st[2] - {{ QB }} * r;
    st[1] = st[0] - {{ QB - 1 }} > 0;
}

// The pass's L1 unit i (of {{ 2 * RB }}: d half i % 2 of the Q units 4 (i / 2) ..: per d-block pair 4 slices [d-block 2], a
// pair of row blocks the first two slices, the next pair the others): the core's own pairs (row blocks me .. me + {{ RB - 1 }})
// into q (per pair [d half][d-block pair][4 panels])
__attribute__((minsize)) void {{ p }}_qsel(const bfloat16 *__restrict u, bfloat16 *__restrict q, int32_t i, int32_t me)
{
{% if "ev_q" in ablate %}
    if (i == 0)
        event0();
{% endif %}
    for (int pp = 0; pp < 2; pp++) {
        const int pr = 2 * (i >> 1) + pp - (me >> 1);  // (the core's pair)
        if (pr < 0 || pr >= {{ RB // 2 }})
            continue;
        const bfloat16 *__restrict a = u + 144 * pp;
        bfloat16 *__restrict d = q + {{ QUNIT }} * pr + {{ QUNIT // 2 }} * (i & 1);
        for (int j = 0; j < {{ ND // 4 }}; j++) {
            al::copy<144, 16>(d, a);
            a += 288;
            d += 144;
        }
    }
}

// Scores of the chunk's K unit kk for each pair of row blocks, A = s + bias (the accumulators start at the rows' bias)
// into sc, the chunk's maximum A per lane into vm{% if KP == 1 %}: the unit's two records (per d-block their panels, interleaved by the memtile); per pair one loop
// over d-block pairs of two panel streams (Q: 4 panels, K: 4), four accumulators (2 row blocks x 2 records){% else %}: a record [hi panels | lo panels]; per pair a loop over d-block pairs for K hi, then one for K lo, two
// accumulators (the row blocks){% endif %}

void {{ p }}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict k, float *__restrict ws, int32_t kk)
{
{% if "qk" in ablate %}
    return;
{% endif %}
    if (AW(int32_t, {{ w.st }})[1])
        return;
{% if "ev_qk" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);  // (the maxima's bf16)
    const float *__restrict bias = AW(float, {{ w.bias }});
{% if KP == 1 %}
    float *__restrict sc = AW(float, {{ w.sc }}) + 256 * kk;
{% else %}
    float *__restrict sc = AW(float, {{ w.sc }}) + 128 * kk;
{% endif %}
    aie::vector<bfloat16, 64> vm = aie::load_v<64>(AW(bfloat16, {{ w.vm }}));
    if (kk == 0)
        vm = aie::broadcast<bfloat16, 64>(-1e30f);
    AIE_LOOP_NO_UNROLL
    for (int pr = 0; pr < {{ RB // 2 }}; pr++) {
        const uint8_t *__restrict qa = (const uint8_t *)q + {{ 2 * QUNIT }} * pr;
        const aie::accum<accfloat, 64> b0(aie::load_v<64>(bias + 128 * pr)), b1(aie::load_v<64>(bias + 128 * pr + 64));
        float *__restrict s = sc + {{ 128 * CR }} * pr;
{% if KP == 1 %}
        al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
        al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)k);
        al::MM8 c00(b0), c01(b0), c10(b1), c11(b1);  // (row block 2 pr + e, record 2 kk + r)
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ ND // 2 }}; j++) {  // (d-blocks 2 j, 2 j + 1)
            const al::BV K0 = ks.pop(), K1 = ks.pop(), K0n = ks.pop(), K1n = ks.pop();
{% for e in (0, 1) %}
            {
                const al::BV Q = qs.pop(), Qn = qs.pop();
                c{{ e }}0.mac(Q, aie::op_transpose(K0));
                c{{ e }}1.mac(Q, aie::op_transpose(K1));
                c{{ e }}0.mac(Qn, aie::op_transpose(K0n));
                c{{ e }}1.mac(Qn, aie::op_transpose(K1n));
            }
{% endfor %}
        }
        aie::store_v(s, c00.template to_vector<float>());
        aie::store_v(s + 64, c10.template to_vector<float>());
        aie::store_v(s + 128, c01.template to_vector<float>());
        aie::store_v(s + 192, c11.template to_vector<float>());
        vm = aie::max(vm, aie::max(aie::max(c00.template to_vector<bfloat16>(), c01.template to_vector<bfloat16>()),
                                   aie::max(c10.template to_vector<bfloat16>(), c11.template to_vector<bfloat16>())));
{% else %}
        al::MM8 c0(b0), c1(b1);
{% for half in range(2) %}
        {
            al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)qa);
            al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)((const uint8_t *)k + {{ 72 * ND * half }}));
            AIE_PREPARE_FOR_POSTPIPELINING
            for (int j = 0; j < {{ ND // 2 }}; j++) {
                const al::BV K = ks.pop(), Kn = ks.pop();
{% for e in (0, 1) %}
                {
                    const al::BV Q = qs.pop(), Qn = qs.pop();
                    c{{ e }}.mac(Q, aie::op_transpose(K));
                    c{{ e }}.mac(Qn, aie::op_transpose(Kn));
                }
{% endfor %}
            }
        }
{% endfor %}
        aie::store_v(s, c0.template to_vector<float>());
        aie::store_v(s + 64, c1.template to_vector<float>());
        vm = aie::max(vm, aie::max(c0.template to_vector<bfloat16>(), c1.template to_vector<bfloat16>()));
{% endif %}
    }
    aie::store_v(AW(bfloat16, {{ w.vm }}), vm);
{% if "ev_qk" in ablate %}
    event1();
{% endif %}
}

// The chunk's softmax terms: masks on a diagonal chunk; rows whose maximum exceeds their reference by 2^16 raise it
// (``raise``: when any A of the chunk reaches 208, masked ones too); P = 2^(A - 192) as bfp16 panels (``ppan``)
void {{ p }}_soft(float *__restrict ws, float *__restrict wo)
{
{% if "soft" in ablate %}
    return;
{% endif %}
    const int32_t *__restrict st = AW(int32_t, {{ w.st }});
    if (st[1])
        return;
{% if "ev_soft" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict sc = AW(float, {{ w.sc }});
    if (st[0] + {{ CR - 1 }} >= 0)  // (the chunk reaches the first block's queries)
        {{ p }}_mask(sc, st[0]);
    if (!aie::lt(aie::load_v<64>(AW(bfloat16, {{ w.vm }})), aie::broadcast<bfloat16, 64>(208.0f)).full())
        {{ p }}_raise(ws, wo);
    {{ p }}_ppan(sc, AW(uint8_t, {{ w.pp }}));
{% if "ev_soft" in ablate %}
    event1();
{% endif %}
}

// o += P V of the chunk's V half u (d-blocks {{ ND // 2 }} u ..: ``pvh``); with the first, l += P J
void {{ p }}_pv(const bfloat16 *__restrict v, float *__restrict ws, float *__restrict wo, int32_t u)
{
{% if "pv" in ablate %}
    return;
{% endif %}
    if (AW(int32_t, {{ w.st }})[1]) {
{% if "ev_chunk" in ablate %}
        if (u)
            event1();
{% endif %}
        return;
    }
{% if "ev_pv" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict o = wo + {{ 32 * ND }} * u;
    {{ p }}_pvh(AW(uint8_t, {{ w.pp }}), (const uint8_t *)v, o, o);
    if (u == 0)
        {{ p }}_lsum(AW(uint8_t, {{ w.pp }}), AW(float, {{ w.l }}), AW(float, {{ w.l }}));
{% if "ev_pv" in ablate %}
    event1();
{% endif %}
{% if "ev_chunk" in ablate %}
    if (u)
        event1();
{% endif %}
}

// Head h's output: o / l of its QB row blocks as the O GEMM's A panels (per block ND panels of 8 queries x 8 dims)
__attribute__((minsize)) void {{ p }}_fin(float *__restrict ws, float *__restrict wo, bfloat16 *__restrict out, int32_t h)
{
{% if "ev_fin" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BOut<> po((bfp16ebs8 *)out);
    for (int j = 0; j < {{ QB }}; j++) {
        const int b = {{ QH }} * j + h;
        const aie::vector<float, 8> ls = al::group8<false>(aie::load_v<64>(AW(float, {{ w.l }}) + 64 * b));  // (8 l)
        const aie::vector<float, 16> inv = aie::inv(aie::concat(ls, aie::broadcast<float, 8>(1.0f)));
        const aie::vector<float, 64> iv = al::bcast8(aie::mul(inv.template extract<8>(0), 8.0f).template to_vector<float>());
        const float *__restrict o = wo + {{ 64 * ND }} * b;
        for (int db = 0; db < {{ ND }}; db++) {
            po.push(aie::accum<accfloat, 64>(al::mulff<64>(aie::load_v<64>(o), iv)).template to_vector<bfp16ebs8>());
            o += 64;
        }
    }
{% if "ev_fin" in ablate %}
    event1();
{% endif %}
}

}  // extern "C"
""")


class AttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 (module docstring) on the cache that already holds the chunk's K and V
    (``PrepOperator``). Column c = KV head c (query heads qh c ..), core row r; in pass b the core takes the query
    blocks 4 QB b + QB r .. of each of the column's heads. Runtime sequence: QS (the prep's of sub-chunks of ``sub``
    tokens, each [column][token group][head][unit]), KV (the layer's cache: COLS KV heads of ``kv_bytes``, ``kv_pad``
    past them), A (T x heads hd as bfp16 panels [T / 8][heads hd / 8]). Parameters ``params``."""

    STACK = 2048
    pfx = kernel_prefix("paa")
    # stats builds: core counters 0: cycles in the region, 1: lock (memory) stall cycles, 2: all cycles, 3: windows
    # (AM029 Performance_Control0 / 1)
    CTRL: ClassVar[tuple[tuple[int, int], ...]] = ((0x31500, 33 | 34 << 8 | 26 << 16 | 26 << 24),
                                                   (0x31504, 1 | 1 << 8 | 33 << 16 | 33 << 24))
    COUNTER0 = 0x31520

    def __init__(self, *, T: int, shape: Shape, kv_bytes: int, kv_pad: int = 0, sub: int = 0,
                 ablate: tuple[str, ...] = ()):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv": wrong results); "ev_<region>"
        (qk, soft, pv, chunk: grp to the second pv) marks a region with event0 / event1 and the sequence programs the
        cores' performance counters (``CTRL``; "memstall": memory instead of lock stalls in counter 1)."""
        s = shape
        self.qb, self.cr = s.blocks
        self.pq = 32 * self.qb  # queries per pass
        self.m = s.pg // self.pq  # passes per cache group
        self.sub = sub or T
        assert T % s.pg == 0 and T // self.pq <= 64 and s.rg % self.cr == 0 and self.m >= 1
        assert self.qb * s.qh % 2 == 0 and T % self.sub == 0 and self.sub % self.pq == 0, "row blocks in pairs"
        assert 4 * self.qb * s.qh % 4 == 0  # (a pass's Q units: whole B objects of 4)
        self.T, self.shape, self.kv_bytes, self.kv_pad, self.ablate = T, s, kv_bytes, kv_pad, tuple(ablate)
        self.n_pass = T // self.pq
        super().__init__()

    @staticmethod
    def regions(rows: int, cr: int) -> list[tuple[str, int]]:
        """The workspace's regions at ``rows`` row blocks and chunks of ``cr`` records (the kernels' comment)."""
        return [("sc", 256 * rows * cr), ("bias", 256 * rows), ("l", 256 * rows), ("pp", 72 * rows * cr), ("vm", 128),
                ("st", 64)]

    def workspace(self) -> dict[str, int]:
        return regions(self.regions(self.qb * self.shape.qh, self.cr))

    def get_kernel_src(self) -> str:
        p = self.pfx
        s = self.shape
        tri = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j > q)
        return _ATTN.render(p=p, w=Attr(self.workspace()), ND=s.nd, QB=self.qb, QH=s.qh, RB=self.qb * s.qh, CR=self.cr,
                            KP=s.kp, M=self.m, QUNIT=s.qunit, tri=f"{tri:016X}", ablate=self.ablate)

    @property
    def n_qs(self) -> int:
        return self.T // 8 * COLS * self.shape.qh * self.shape.qunit // 2

    @property
    def n_kv(self) -> int:
        return (COLS * self.kv_bytes + self.kv_pad) // 2

    @property
    def n_a(self) -> int:
        return self.T * COLS * self.shape.qh * self.shape.hd * 9 // 16

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.n_qs,), torch.bfloat16), ArgSpec("in", (self.n_kv,), torch.bfloat16),
                ArgSpec("out", (self.n_a,), torch.bfloat16)]

    def params(self, p0: int) -> dict[str, int]:
        s = self.shape
        assert p0 % s.pg == 0
        fg = p0 // s.pg + 1
        out = {P_P8: p0 // 8, P_FG: fg, P_FQ: fg * s.rg // self.cr}
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
        wo_ty = np.ndarray[(64 * nd * rb,), f32]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty, np.int32, np.int32, np.int32])
        k_grp = Kernel(f"{p}_grp", "kernel.o", [ws_ty, np.int32, np.int32])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [u_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, u_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, wo_ty])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [u_ty, ws_ty, wo_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty, np.int32])
        p_p8, p_fq = ScratchpadParameter(P_P8, np.int32), ScratchpadParameter(P_FQ, np.int32)
        p_fg, p_fo = ScratchpadParameter(P_FG, np.int32), ScratchpadParameter(P_FO, np.int32)
        n_pass, ku, gq = self.n_pass, cr * s.kp // 2, s.rg // cr

        # stream A: the K records (hi + lo: a group's [K obj | padding | K obj], the memtile reads the objects' records)
        ilv = stream_walk(s.unit // 2, [(nd, 36), (2, nd * 36), (36, 1)])  # (per d-block the unit's two halves)
        if s.kp == 2:
            ka_ty = np.ndarray[((s.a_bytes + s.used) // 2,), bf]
            walk = stream_walk((s.a_bytes + s.used) // 2, [(2, s.a_bytes // 2), (s.used // 1152, 576), (576, 1)],
                               pad=[(0, 0)] * 3)
        else:  # (a unit's two records interleaved per d-block)
            ka_ty, walk = u_ty, ilv
        a3 = [ObjectFifo(ka_ty, name=f"la_ka{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().forward(tile=Tile(c, 1), obj_type=u_ty, depth=2, name=f"la_k{c}", to_stream=walk)
              for c in range(COLS)]
        # stream B: objects of two units, four slices of nd panels (4 V records; 4 Q units: two pairs of row blocks)
        # -> per d half [d-block pair][slice][d-block 2]
        vwalk = stream_walk(s.unit, [(2, nd // 2 * 36), (nd // 4, 72), (4, nd * 36), (72, 1)])
        b3 = [ObjectFifo(np.ndarray[(s.unit,), bf], name=f"la_vb{c}", depth=2) for c in range(COLS)]
        b1 = [b3[c].cons().forward(tile=Tile(c, 1), obj_type=u_ty, depth=2, name=f"la_v{c}", to_stream=vwalk)
              for c in range(COLS)]
        o3 = [ObjectFifo(om_ty, name=f"la_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([qb * nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"la_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def turn(a_h, b_h, o_h, init, grp, qsel, qk, soft, pv, fin, ws, wo, q, p8, fq, me, row, bg, bb, nq):
            """(a core's share of a pass: its Q units of the pass's nq, the history, its output)"""
            for i in range_(nq):
                u = b_h.acquire(1)
                qsel(u, q, i, me)
                b_h.release(1)
            init(ws, wo, p8.read(), bg, bb)
            n = fq.read()  # (read after this dispatch's first object: the host wrote it before the fills)
            for _ in range(gq):
                n = n + bg
            for t in range32(n):
                grp(ws, t, row)
                for kk in range(ku):
                    k = a_h.acquire(1)
                    qk(q, k, ws, kk)
                    a_h.release(1)
                soft(ws, wo)
                for uu in range(2):
                    v = b_h.acquire(1)
                    pv(v, ws, wo, uu)
                    b_h.release(1)
            for h in range(qh):
                o = o_h.acquire(1)
                fin(ws, wo, o, h)
                o_h.release(1)

        def core(a_h, b_h, o_h, init, grp, qsel, qk, soft, pv, fin, ws, wo, q, p8, fq, me, row):
            for _ in forever_():
                for bg in range_(n_pass // m):
                    for bb in range(m):  # (the pass's units to every core: each keeps its own)
                        turn(a_h, b_h, o_h, init, grp, qsel, qk, soft, pv, fin, ws, wo, q, p8, fq, me, row, bg, bb, 2 * rb)

        workers = [Worker(core, [a1[c].cons(), b1[c].cons(), o1[c][r].prod(), k_init, k_grp, k_qsel, k_qk, k_soft,
                                 k_pv, k_fin, Buffer(ws_ty, name=f"la_ws{c}_{r}"), Buffer(wo_ty, name=f"la_wo{c}_{r}"),
                                 Buffer(q_ty, name=f"la_q{c}_{r}"), p_p8, p_fq, rb * r, r],
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

        def seq(QS, KV, A, ah, bh, oh):
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
            inflight = []  # (a pass's fills: at most three in flight, within the shim tiles' BDs)
            for b in range(n_pass):
                tp = TaskGroup()
                ng = b // m  # (the chunk's groups after its first that the pass streams)
                t0 = 4 * qb * b  # (the pass's first token group: its sub-chunk, the group in it)
                q0 = t0 // sg * sg * COLS * qh * uw + t0 % sg * qh * uw
                for c in range(COLS):
                    k0, v0 = c * hb, c * hb + s.kp * o2
                    parts = [kv_part(k0, sizes=k_tap[0], strides=k_tap[1])]
                    if ng:
                        parts.append(kv_part(k0, ng, sizes=k_tap[0], strides=k_tap[1]))
                    chained_fill(ah[c], KV, parts, tp, wait=True)
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0 + c * sg * qh * uw, [1, 1, 4 * qb * qh, uw],
                                                               [0, 0, uw, 1])),
                             kv_part(v0, sizes=[1, 1, rows, 576], strides=[0, g2, 576, 1])]
                    if ng:
                        parts.append(kv_part(v0, ng, sizes=[1, 1, rows, 576], strides=[0, g2, 576, 1]))
                    chained_fill(bh[c], KV, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 2:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        return module

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (the stack at 0): the stack, Q and the output buffers (touched by ``fin``
        only) in the first (Q K streams Q against K, whose objects sit with the workspace in the last), O from the
        second on (P V: O against P in the workspace and V in stream B's objects after O)."""
        s, R = self.shape, 0x4000
        rows, u = self.qb * s.qh, s.unit
        q, wo, ws, out = rows * s.qunit, 64 * s.nd * rows * 4, self.workspace()["size"], self.qb * s.nd * 72
        lay = {"la_q": [self.STACK], "la_or": [self.STACK + q, self.STACK + q + out], "la_wo": [R]}
        pos = R + wo
        lay["la_v"], pos = [pos, pos + u], pos + 2 * u
        top = -(-pos // R) * R  # (the workspace from the next region on, stream A's objects after it)
        if top + ws + 2 * u > L1:
            lay["la_k"], pos = [pos, pos + u], pos + 2 * u
            top = max(pos, L1 - ws)
            lay["la_ws"] = [top]
        else:
            lay["la_ws"], lay["la_k"] = [top], [top + ws, top + ws + u]
        assert self.STACK + q + 2 * out <= R and max(v[-1] + n for v, n in ((lay["la_ws"], ws), (lay["la_k"], u))) <= L1, lay
        return lay
