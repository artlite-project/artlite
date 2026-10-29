"""The prefill of the qwen35 full-attention layers: a T-token chunk at position p0 (a multiple of 32) from the attn_in
GEMMs' output G (``prefill.gemm``: per column [q + gate of its QC heads | K part | V part]: KV head h's dims in the NPH
= 8 / KV heads columns NPH h ..) to the attn_out GEMM's A.

``AttnPrepOperator``: per column c, cores 2 .. 1 + QC one per q head (the column's heads QC c ..), core 2 + QC the
column's KV role. Per 8 tokens a Q core turns its head's rows into the attention's operands: q / rms(q) q_norm, the
NeoX rotation of dims 0 .. 63, x log2(e) / 16 (scores in log2 units), as bfp16 panels of 8 queries x 8 dims, per d-block
hi then lo (4608 B), then sigmoid(gate) (256 bf16 per query, padded to 4608 B: per d-block 144 B as a Q panel pair).
QS holds per column and token group [the column's heads' Q panels | their sigmoid(gate)] (``SLOT`` bytes per head each).
The 2 x KV heads jobs of the KV cores (``AttnPrepOperator.kv_job``: one every 8 / (2 KV heads) columns; 9B / 27B every
column, MiniCPM-V every other): K head h (rms(k) k_norm, rotated) as hi + lo records, then V head h as records of blocks
over the 8 positions, in the decode's cache layout (``decode.kv``, d-block major: a K record [d-block][hi | lo], two V
records [d-block][record 2]): the chunk's groups from p0 / 32 on (scratchpad
parameter ``QA_KV``), so the decode continues from them. cos / sin of the
tokens' positions from a table (``cs_table``; parameter ``QA_CS``: the row of the chunk's first token): the static rows
of every text position, or rows the host writes past them for a chunk of image tokens (M-RoPE: ``cs_rows``).

``PrefillAttnOperator``: per column 4 cores, passes of 32 QG queries (QG = 2 when a column holds one q head, the L1 of
the 9B's two heads: half the history's re-streams, else 1); core r takes the QG token groups QG r .. of the pass for
each of the column's heads (QC QG row panels: an O of QC QG x 8 KB) and streams the cache groups up to the pass's (the
KV head's stream broadcast to the 4 NPH cores of its columns; with two row panels per core and 4 columns per KV head on
two streams, ``two_streams``: one stream's memtile channel bound the history's re-streams): per K record the scores on
the MMUL (Q hi + lo against K hi + lo, three products: one loop over the record's [d-block][K hi | K lo] panels, the Q
slots' likewise; P V over the V object's pairs of records [d-block][record 2]), per group of 32 positions the softmax (a
row's maximum is raised, rescaling o and l, only when the group's exceeds it by 2^16: the per-dimension norm weights
leave scores unbounded), P = 2^(s - m) by the vexp2 exp2 (vexp2 builds an exact 2^n; 0.59 ULP) as bfp16 panels, o += P
V; at the pass's end o / l sigmoid(gate) as the attn_out GEMM's bfp16 A panels. Within 0.7% of the exact attention of
its own operands, ~1% rms of llama.cpp's attn_gated.

Time (T = 512, per layer alone, at p0 = 0 / 4K / 16K): the 27B 1.38 / 6.89 / 22.1 ms (its compute compiled out at
16K: 20.6, the KV re-streams of the 32-query passes at the DDR ceiling), the 1B 0.97 / 2.47 / 7.23 (5.99: the cores'
QK binds), the 35B-A3B (two streams) 1.12 / 3.44 / 10.6 (8.8).
"""

import math

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

from artlite.gemm.gemm_q import ChainPart, chained_fill
from artlite.model_config import QwenConfig
from artlite.operators import kmath
from artlite.operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from artlite.operators.design import COLS, range32, stream_walk
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.l1_layout import pin_l1_buffers
from artlite.operators.util import Attr, bf16_hilo, regions
from artlite.prefill.gemm import Layout

from . import attn as AT

HD = AT.HD
ND = AT.ND
SLOT = AT.REC_K  # bytes per head of a token group's Q panels (and of its sigmoid(gate): 4096 B used)
GROUP = 3 * AT.OBJ_BYTES  # bytes of a cache group (32 positions: two K objects, a V object)
# Scratchpad parameters. QA_KV and QA_OFF are one per operator size ("qa_kv<T>"): a parameter's range is the tightest
# of the operators that read it, and a smaller chunk may start where a larger one would overrun the cache.
QA_CS = "qa_cs"  # bf16 offset of the chunk's rows in the cos / sin table (128 p0)
QA_KV = "qa_kv"  # bf16 offset of the chunk's first cache group (GROUP / 2 * p0 / 32)
QA_N = "qa_n"  # p0 / 32 + 1 (cache groups every attention pass streams first; the cores' count)
QA_OFF = "qa_off"  # bf16 offset of the chunk's second cache group
PW = 2 * 512  # bytes of the attention softmax's scratch (P and f)
# (timing builds, ablate "ev_<kernel>": counter 0 the window's cycles, 1 lock stall, 2 all cycles, 3 the windows)
COUNTERS = ((0x31500, 33 | 34 << 8 | 26 << 16 | 26 << 24), (0x31504, 1 | 1 << 8 | 33 << 16 | 33 << 24))


def pass_groups(cfg: QwenConfig, T: int) -> int:
    """QG: the token groups per core and head in a 32 QG-token attention pass (2 when a column holds one q head and T
    allows, the L1 of the 9B's two heads: half the history's re-streams); QS holds the slots pass by pass."""
    return 2 if cfg.num_attention_heads == COLS and T % 64 == 0 else 1


def two_streams(cfg: QwenConfig, T: int) -> bool:
    """Whether the prefill attention of a T-token chunk takes each KV head's cache on two streams (two row panels per
    core: one q head per column and QG = 2, or two q heads; a KV head's 4 columns at least, each stream from two
    shims)."""
    qc = cfg.num_attention_heads // COLS
    return COLS // cfg.num_key_value_heads >= 4 and qc * pass_groups(cfg, T) == 2


def part_walk() -> TensorAccessPattern:
    """The two-stream attention's (``two_streams``) memtile walk of one shim channel's part of a stream object (the two
    parts make the object): d-blocks 8 i .. of a stream's half of a K object's two records ([d-block][hi | lo] each),
    of a V object's two pairs of records or of two slots ([piece 2][d-block 8][144 B]) -> [d-block][piece][144 B]. (One
    stream's objects are a K record, a V pair or a slot: as they lie in DDR.)"""
    return stream_walk(SLOT // 4, [(ND // 4, 72), (2, SLOT // 8), (72, 1)])


def cs_rows(cfg: QwenConfig, positions: np.ndarray | range) -> np.ndarray:
    """(n, 64) fp32: per position (text, or M-RoPE (T, H, W)) [cos 32 | sin 32] of the rotated pairs
    (``QwenConfig.rope_theta``)."""
    th = cfg.rope_theta(positions).astype(np.float64)
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


def cs_table(cfg: QwenConfig, n: int) -> np.ndarray:
    """(n, 64) fp32: ``cs_rows`` of the text positions 0 .. n - 1."""
    return cs_rows(cfg, range(n))


_PREP = JINJA_ENV.from_string(r"""
// The prep cores' workspace (bytes): nrm [q_norm | k_norm] fp32, row (256 fp32), stage [db 32][row 8][8] fp32
#define PW(T, off) ((T *)((uint8_t *)ws + (off)))

{% set sigmoid = km.table_pass(km.SIGMOID) %}
{{ sigmoid.src }}

namespace {

// x (256 bf16) -> row = x / rms(x) w (fp32), dims 0 .. 63 rotated by cs [cos 32 | sin 32] (NeoX pairs i, i + 32)
__attribute__((noinline)) void {{ p }}_nr(const bfloat16 *__restrict x, const float *__restrict w,
                                          const float *__restrict cs, float *__restrict row)
{
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    for (int i = 0; i < {{ HD }}; i += 64)
        ss = aie::mac_square(ss, aie::load_v<64>(x + i));
    const float inv = al::inv_rms<{{ HD }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    const aie::vector<float, 64> iv = aie::broadcast<float, 64>(inv);
    for (int i = 0; i < {{ HD }}; i += 64) {
        const aie::vector<float, 64> xf = aie::accum<accfloat, 64>(aie::load_v<64>(x + i)).template to_vector<float>();
        aie::store_v(row + i, al::mulff<64>(al::mulff<64>(xf, iv), aie::load_v<64>(w + i)));
    }
    const aie::vector<float, 32> a = aie::load_v<32>(row), b = aie::load_v<32>(row + 32);
    const aie::vector<float, 32> c = aie::load_v<32>(cs), s = aie::load_v<32>(cs + 32);
    aie::store_v(row, aie::sub(al::mulff<32>(a, c), al::mulff<32>(b, s)));
    aie::store_v(row + 32, aie::add(al::mulff<32>(a, s), al::mulff<32>(b, c)));
}

// row (256 fp32) -> row r of the stage [db][r][8]
__attribute__((noinline)) void {{ p }}_scatter(const float *__restrict row, float *__restrict st, int r)
{
    float *__restrict d = st + 8 * r;
    for (int db = 0; db < {{ ND }}; db++) {
        aie::store_v(d, aie::load_v<8>(row));
        row += 8;
        d += 64;
    }
}

// stage -> per d-block the hi panel, then the lo panel (a panel: the 8 rows' blocks of 8 dims; Q and the K records:
// one stream for the attention's MMUL)
__attribute__((noinline)) void {{ p }}_hiloi(const float *__restrict st, uint8_t *__restrict out)
{
    const al::BV nI = al::neg_eye();
    al::BOut<> o((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        al::BV h, l;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
        o.push(h);
        o.push(l);
        st += 64;
    }
}

}  // namespace

extern "C" {

// The first object: q_norm, k_norm (256 fp32 each); q_norm times the scores' scale log2(e) / 16 (the rotation is
// linear: the scale before it)
void {{ p }}_prm(const bfloat16 *__restrict n, float *__restrict ws)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::copy<{{ 2 * HD }}, 16>(PW(float, {{ w.nrm }}), (const float *)n);
    for (int i = 0; i < {{ HD }}; i += 64)
        aie::store_v(PW(float, {{ w.nrm }}) + i, al::mul_hilo<64, {{ qs[0] }}f, {{ qs[1] }}f>(aie::load_v<64>(PW(float, {{ w.nrm }}) + i)));
}

// 8 tokens of a q head: q (rows [q 256 | gate 256] bf16), cs (8 x [cos 32 | sin 32] fp32) -> out: per d-block the
// Q panels hi, lo of q / rms(q) q_norm log2(e) / 16, rotated
void {{ p }}_q(const bfloat16 *__restrict q, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    for (int r = 0; r < 8; r++) {
        {{ p }}_nr(q + {{ 2 * HD }} * r, PW(float, {{ w.nrm }}), cs + 64 * r, row);
        {{ p }}_scatter(row, st, r);
    }
    {{ p }}_hiloi(st, (uint8_t *)out);
}

// sigmoid(gate) of the 8 tokens in the panel order of the attention's output: per 8 dims db a block [query 8][8] and 8
// zeros (72 values: the slot's layout of a Q panel pair), sg. The gates clamped to the table's range into the stage
// (bf16 [8][256]), the table pass in place (kmath.SIGMOID: within 0.54 bf16 ULP), then the blocks
void {{ p }}_sg(const bfloat16 *__restrict q, float *__restrict ws, bfloat16 *__restrict sg)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 32> hi = aie::broadcast<bfloat16, 32>({{ tmax }}f), lo = aie::broadcast<bfloat16, 32>(-{{ tmax }}f);
    bfloat16 *__restrict st = (bfloat16 *)PW(float, {{ w.stage }});
    const bfloat16 *__restrict g = q + {{ HD }};
    bfloat16 *__restrict t = st;
    for (int r = 0; r < 8; r++) {
        for (int i = 0; i < {{ HD }}; i += 32)
            aie::store_v(t + i, aie::min(aie::max(aie::load_v<32>(g + i), lo), hi));
        g += {{ 2 * HD }};
        t += {{ HD }};
    }
    {{ sigmoid }}<{{ 8 * HD }}>(st, st, st, st);
    const bfloat16 *__restrict ps = st;
    bfloat16 *__restrict po = sg;
    for (int db = 0; db < {{ ND }}; db++) {
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++)
            aie::store_v(po + 8 * r, aie::load_v<8>(ps + {{ HD }} * r));
        aie::store_v(po + 64, aie::zeros<bfloat16, 8>());
        ps += 8;
        po += 72;
    }
}

// 8 tokens of a KV head's K (rows [half 0 | half 1] bf16), cs -> its cache record [d-block][hi | lo]
void {{ p }}_k(const bfloat16 *__restrict k, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    for (int r = 0; r < 8; r++) {
        {{ p }}_nr(k + {{ HD }} * r, PW(float, {{ w.nrm }}) + {{ HD }}, cs + 64 * r, row);
        {{ p }}_scatter(row, st, r);
    }
    {{ p }}_hiloi(st, (uint8_t *)out);
}

// The V panel of d-block db of 8 tokens' rows (bf16): per dim of the 8 a block over the positions (the rows' inserts
// unrolled: a rolled r makes insert(r, ..) a switch, ~26K cycles per record; unrolled II 14, 478)
[[clang::always_inline]] inline al::BV {{ p }}_vpan(const bfloat16 *__restrict v, int db)
{
    aie::vector<bfloat16, 64> pd;  // [position][dim]
    AIE_LOOP_UNROLL_FULL
    for (int r = 0; r < 8; r++)
        pd.insert(r, aie::load_v<8>(v + {{ HD }} * r + 8 * db));
    return aie::accum<accfloat, 64>(aie::transpose(pd, 8, 8)).template to_vector<bfp16ebs8>();
}

// 8 tokens of a KV head's V (rows bf16) -> record ``half`` of a pair of cache records ([d-block][record 2], out): the
// first's panels into the workspace, then per d-block the first's and the second's
void {{ p }}_v(const bfloat16 *__restrict v, const bfloat16 *__restrict cs_, float *__restrict ws,
               bfloat16 *__restrict out, int32_t half)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    if (!half) {
        al::BOut<> o((bfp16ebs8 *)ws);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int db = 0; db < {{ ND }}; db++)
            o.push({{ p }}_vpan(v, db));
        return;
    }
    al::BIn<> a((const bfp16ebs8 *)ws);
    al::BOut<> o((bfp16ebs8 *)out);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        o.push(a.pop());
        o.push({{ p }}_vpan(v, db));
    }
}

}  // extern "C"
""")


class AttnPrepOperator(ArtcOperator):
    """The attention prep of a chunk (module docstring). Runtime sequence: G (attn_in's output, ``layout``), CS (the
    cos / sin table of ``cs_positions`` rows, as bf16 pairs), NRM (q_norm | k_norm fp32), QS (per column, token group
    and head a ``SLOT``, pass-major: [32 QG-token pass][column][token group 4 QG][Q | gate][head], ``pass_groups``;
    a sub-chunk's part of a chunk's QS contiguous),
    KV (one layer's cache: the KV heads' of ``attn.AttnShape.head_bytes``). Parameters: ``QA_CS`` = 128 x the table row
    of the first token (default p0), ``QA_KV`` (T's) = GROUP / 2 * p0 / 32 (``params``)."""

    STACK = 3072
    pfx = kernel_prefix("qap")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, head_bytes: int, cs_positions: int):
        self.kvh = cfg.num_key_value_heads
        assert cfg.head_dim == HD and cfg.rope_dims == 2 * AT.ROT and COLS % (2 * self.kvh) == 0
        assert T % 32 == 0 and T // 8 <= 64, "whole cache groups; a BD repeats at most 64 times"
        assert {"q", "k", "v"} <= set(layout.off)
        self.T, self.cfg, self.layout = T, cfg, layout
        self.nph, self.per = COLS // self.kvh, self.kvh * HD // COLS  # (a KV head's columns, its dims in one)
        self.qc = cfg.num_attention_heads // COLS
        assert 1 <= self.qc <= 3, "a core per q head and the KV core in a column"
        self.head_bytes, self.cs_positions = head_bytes, cs_positions
        self.qg = pass_groups(cfg, T)
        assert T % (32 * self.qg) == 0
        super().__init__()

    def workspace(self) -> dict[str, int]:
        return {"nrm": 0, "row": 8 * HD, "stage": 12 * HD, "size": 44 * HD}

    def get_kernel_src(self) -> str:
        p = self.pfx
        qs = bf16_hilo(math.log2(math.e) / 16)
        return _PREP.render(p=p, HD=HD, ND=ND, w=Attr(self.workspace()), eps=repr(self.cfg.rms_norm_eps), qs=qs,
                            AT_REC_K=AT.REC_K, tmax=repr(kmath.SIGMOID.t_max))

    def kv_job(self, c: int) -> tuple[str, int] | None:
        """Column c's KV core's job: ("k" or "v", the KV head), or None (one every COLS / (2 KV heads) columns)."""
        step = COLS // (2 * self.kvh)
        if c % step:
            return None
        j = c // step
        return ("k" if j < self.kvh else "v"), j % self.kvh

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.cs_positions * 128,), torch.bfloat16),
                ArgSpec("in", (2 * HD * 2,), torch.bfloat16),
                ArgSpec("out", (self.n_qs,), torch.bfloat16),
                ArgSpec("out", (self.kvh * self.head_bytes // 2,), torch.bfloat16)]

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument."""
        return COLS * self.T // 8 * self.qc * SLOT

    def params(self, p0: int, cs_row: int | None = None) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 whose tokens' cos / sin start at table row ``cs_row``
        (default p0)."""
        r = p0 if cs_row is None else cs_row
        assert p0 % 32 == 0 and 0 <= r and r + self.T <= self.cs_positions
        return {QA_CS: 128 * r, f"{QA_KV}{self.T}": GROUP // 2 * (p0 // 32)}

    def get_mlir_artifact(self) -> Module:
        p, T, qc, lay = self.pfx, self.T, self.qc, self.layout
        N, pitch = lay.N, lay.pitch
        bf = np.dtype[bfloat16]
        row_ty = np.ndarray[(8 * 2 * HD,), bf]  # 8 rows of [q | gate]
        kv_ty = np.ndarray[(8 * HD,), bf]  # 8 rows of a KV head
        in_ty = np.ndarray[(qc * 8 * 2 * HD + 8 * HD,), bf]
        inq_ty = np.ndarray[(qc * 8 * 2 * HD,), bf]  # (a column without a KV job)
        jobs = {c: self.kv_job(c) for c in range(COLS) if self.kv_job(c) is not None}
        cs_ty = np.ndarray[(8 * 128,), bf]  # 8 positions x 64 fp32 (the norms: 512 fp32)
        slot_ty = np.ndarray[(SLOT // 2,), bf]
        qs_ty = np.ndarray[(qc * SLOT // 2,), bf]
        kr_ty = np.ndarray[(AT.REC_K // 2,), bf]
        vr_ty = np.ndarray[(AT.REC_V,), bf]  # (a pair of V records)
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty])
        k_q = Kernel(f"{p}_q", "kernel.o", [row_ty, cs_ty, ws_ty, slot_ty])
        k_sg = Kernel(f"{p}_sg", "kernel.o", [row_ty, ws_ty, slot_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [kv_ty, cs_ty, ws_ty, kr_ty])
        k_v = Kernel(f"{p}_v", "kernel.o", [kv_ty, cs_ty, ws_ty, vr_ty, np.int32])
        p_cs, p_kv = ScratchpadParameter(QA_CS, np.int32), ScratchpadParameter(f"{QA_KV}{T}", np.int32)
        n_tg = T // 8

        a3 = [ObjectFifo(in_ty if c in jobs else inq_ty, name=f"qp_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split([8 * 2 * HD * h for h in range(qc + (c in jobs))], tile=Tile(c, 1),
                                 depths=[2] * (qc + (c in jobs)), obj_types=[row_ty] * qc + [kv_ty] * (c in jobs),
                                 names=[f"qp_ah{c}_{h}" for h in range(qc + (c in jobs))]) for c in range(COLS)]
        b = [ObjectFifo(cs_ty, name=f"qp_b{c}", depth=2) for c in range(COLS)]
        o3 = [ObjectFifo(qs_ty, name=f"qp_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([SLOT // 2 * h for h in range(qc)], tile=Tile(c, 1), depths=[2] * qc,
                                obj_types=[slot_ty] * qc, names=[f"qp_oh{c}_{h}" for h in range(qc)])
              for c in range(COLS)]
        k = {c: ObjectFifo(kr_ty if jobs[c][0] == "k" else vr_ty, name=f"qp_k{c}", depth=2) for c in jobs}

        def core(a_h, b_h, o_h, prm, fn, ws, sg=None):
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    o = o_h.acquire(1)
                    fn(x, cs, ws, o)
                    o_h.release(1)
                    if sg is not None:  # (a Q core: then the token group's sigmoid(gate))
                        o = o_h.acquire(1)
                        sg(x, ws, o)
                        o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        def core_q(a_h, b_h, o_h, prm, fn, sg, ws):
            core(a_h, b_h, o_h, prm, fn, ws, sg)

        def core_v(a_h, b_h, o_h, prm, fn, ws):  # (two token groups' records per object)
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg // 2):
                    o = o_h.acquire(1)
                    for half in range(2):
                        x = a_h.acquire(1)
                        cs = b_h.acquire(1)
                        fn(x, cs, ws, o, half)
                        b_h.release(1)
                        a_h.release(1)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for h in range(qc + (c in jobs)):
                ws = Buffer(ws_ty, name=f"qp_ws{c}_{h}")
                if h < qc:
                    fn, args = core_q, [a1[c][h].cons(), b[c].cons(), o1[c][h].prod(), k_prm, k_q, k_sg, ws]
                else:
                    fn, args = (core, k_k) if jobs[c][0] == "k" else (core_v, k_v)
                    args = [a1[c][h].cons(), b[c].cons(), k[c].prod(), k_prm, args, ws]
                workers.append(Worker(fn, args, tile=Tile(c, 2 + h), stack_size=self.STACK,
                                      dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv = self.cs_positions * 128, self.kvh * self.head_bytes // 2
        nph, per = self.nph, self.per
        n_qs, blk = self.n_qs, 4 * self.qg * qc * SLOT  # (bf16 of a column's slots in a pass)
        hb, g2, o2 = self.head_bytes // 2, GROUP // 2, AT.OBJ_BYTES // 2

        def seq(G, CS, NRM, QS, KV, ah, bh, oh, khs):
            kh = dict(zip(k, khs))
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                taps = [TensorAccessPattern((T * N,), c * pitch + lay.off["q"], [n_tg, qc, 8, 2 * HD],
                                            [8 * N, 2 * HD, N, 1])]
                if c in jobs:  # (the KV head's rows: its dims in its NPH columns)
                    kind, h = jobs[c]
                    taps.append(TensorAccessPattern((T * N,), nph * h * pitch + lay.off[kind], [n_tg, 8, nph, per],
                                                    [8 * N, N, pitch, 1]))
                chained_fill(ah[c], G, taps, tg)
                bh[c].fill(NRM, TensorAccessPattern((2 * HD * 2,), 0, [1, 1, 1, 2 * HD * 2], [0, 0, 0, 1]), group=tg)
                bh[c].fill(CS, TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 128], [0, 0, 0, 1]), offset_parameter=p_cs,
                           group=tg)
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * blk, [1, n_tg // (4 * self.qg), blk // 576, 576],
                                                    [0, COLS * blk, 576, 1]), group=tg, wait=True)
                if c not in jobs:
                    continue
                kind, h = jobs[c]
                # (rows of 576 elements: a shim BD's inner dimension is at most 1023 words)
                if kind == "k":  # records 4 i .. 4 i + 3 of the chunk: K objects 3 (g + i), 3 (g + i) + 1, two records each
                    tap = TensorAccessPattern((n_kv,), h * hb, [T // 32, 2, 8, 576], [g2, o2, 576, 1])
                else:  # V object 3 (g + i) + 2: its 4 records
                    tap = TensorAccessPattern((n_kv,), h * hb + 2 * o2, [1, T // 32, 8, 576], [0, g2, 576, 1])
                kh[c].drain(KV, tap, offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(2 * HD * 2,), bf],
                           np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in k.items()]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_ATTN = JINJA_ENV.from_string(r"""
// The attention core's workspace (bytes, PrefillAttnOperator.workspace()): per row panel (R = QC QG: head h = i / QG,
// token group QG r + i % QG of the pass) sc [panel][record 4][8 x 8] fp32 (A = s + 192 - m: the scores s in log2
// units against the row's maximum; after the softmax its first 288 B the P panels), m, b, l [panel][64] (lane 8 q + k:
// query q's running maximum, 192 - m, the partial sums), o [panel][db 32][8 x 8] fp32{{ ", ob its second copy (P V
// reads one and writes the other: o after a group)" if OB }}; pw (the softmax's scratch, its own buffer): P [8][32],
// f [8][32] bf16 (or a rescale's row maxima [64] fp32)
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

{% set exp2, e192 = km.exp2(64, inline=True), km.exp2(32, bias=192, below="bounded") %}
{{ exp2.src }}
{{ e192.parts.reduce.src }}
{{ e192.parts.poly.src }}

namespace {

// Rows whose maximum grew: o, l scaled by 2^(m - mn), m = mn = max(m, rm) (rm: the rows' group maxima)
__attribute__((noinline)) void {{ p }}_rescale(float *__restrict o, float *__restrict pl, float *__restrict pm,
                                               const float *__restrict prm)
{
    const aie::vector<float, 64> m = aie::load_v<64>(pm);
    const aie::vector<float, 64> mn = aie::max(m, aie::load_v<64>(prm));
    const aie::vector<float, 64> a = {{ exp2 }}(aie::sub(m, mn));
    for (int db = 0; db < {{ ND }}; db++)
        aie::store_v(o + 64 * db, al::mulff<64>(aie::load_v<64>(o + 64 * db), a));
    aie::store_v(pl, al::mulff<64>(aie::load_v<64>(pl), a));
    aie::store_v(pm, mn);
}

// Masked keys of a diag group (row panel token group t, record rec = t % 4; dd < 0: the group lies after the panel's
// queries): -16384 (P = 0 against any real row maximum)
__attribute__((noinline)) void {{ p }}_mask(float *__restrict s, int32_t dd, int32_t rec)
{
    const aie::mask<64> tri = aie::mask<64>::from_uint64(0x{{ tri }}ull);  // lane 8 q + j: j > q
    const aie::vector<float, 64> ninf = aie::broadcast<float, 64>(-16384.0f);
    for (int k = 0; k < 4; k++) {
        if (dd < 0 || k > rec)
            aie::store_v(s + 64 * k, ninf);
        else if (k == rec)
            aie::store_v(s + 64 * k, aie::select(aie::load_v<64>(s + 64 * k), ninf, tri));
    }
}

// pass 1 of P = 2^(s - m) over a head's 4 panels of A = s + 192 - m, 32 lanes a step: the first step of kmath.exp2
// (P = 2^(round(s - m) + 64) exactly, f = s - m - round(s - m); masked keys -16384: P ~ 0) -> pP, pF; whether
// every bf16(A) <= 208 (s <= m + 16)
[[clang::always_inline]] inline bool {{ p }}_pexp1(const float *__restrict sl, bfloat16 *__restrict pP,
                                                   bfloat16 *__restrict pF)
{
    aie::vector<bfloat16, 32> mx = aie::broadcast<bfloat16, 32>(128.0f);  // (the clamp's constant)
    AIE_LOOP_UNROLL(2)
    for (int t = 0; t < 8; t++) {
        aie::accum<accfloat, 32> A(aie::load_v<32>(sl));
        mx = aie::max(mx, A.template to_vector<bfloat16>());
        aie::vector<bfloat16, 32> P, f;
        {{ e192.parts.reduce }}(A, P, f);
        aie::store_v(pF, f);
        aie::store_v(pP, P);
        sl += 32;
        pP += 32;
        pF += 32;
    }
    return aie::lt(mx, aie::broadcast<bfloat16, 32>(209.0f)).full();
}

__attribute__((noinline)) bool {{ p }}_pass1(const float *__restrict s, float *__restrict pw)
{
    return {{ p }}_pexp1(s, (bfloat16 *)pw, (bfloat16 *)pw + 256);
}

// pass 2, a panel a step: the second step of kmath.exp2 (2^-64 P 2^f, bias 192): the panel's P as bfp16 into pp, l += P
__attribute__((noinline)) void {{ p }}_pass2(const float *__restrict pw, float *__restrict pl, uint8_t *__restrict pp)
{
    const bfloat16 *__restrict pP = (const bfloat16 *)pw;
    const bfloat16 *__restrict pF = (const bfloat16 *)pw + 256;
    al::BOut<> po((bfp16ebs8 *)pp);
    aie::accum<accfloat, 32> l0(aie::load_v<32>(pl)), l1(aie::load_v<32>(pl + 32));
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 4; t++) {
        aie::accum<accfloat, 32> r[2];
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 2; u++) {
            r[u] = {{ e192.parts.poly }}(aie::load_v<32>(pP + 32 * u), aie::load_v<32>(pF + 32 * u));
        }
        l0 = aie::add(l0, r[0]);
        l1 = aie::add(l1, r[1]);
        po.push(aie::concat(r[0], r[1]).template to_vector<bfp16ebs8>());
        pP += 64;
        pF += 64;
    }
    aie::store_v(pl, l0.template to_vector<float>());
    aie::store_v(pl + 32, l1.template to_vector<float>());
}

// A head's row maxima rise: rm = the group's row maxima of s = A - b, then o, l scaled by 2^(m - mn), m = mn =
// max(m, rm), b = 192 - mn, A += the change of b
__attribute__((noinline)) void {{ p }}_raise(float *__restrict s, float *__restrict o, float *__restrict pl,
                                             float *__restrict pm, float *__restrict pb, float *__restrict prm)
{
    const aie::vector<float, 64> v = aie::max(aie::max(aie::load_v<64>(s), aie::load_v<64>(s + 64)),
                                              aie::max(aie::load_v<64>(s + 128), aie::load_v<64>(s + 192)));
    const aie::vector<float, 64> b = aie::load_v<64>(pb);
    aie::store_v(prm, aie::sub(al::bcast8(al::group8<true>(v)), b));
    {{ p }}_rescale(o, pl, pm, prm);
    const aie::vector<float, 64> bn = aie::sub(aie::broadcast<float, 64>(192.0f), aie::load_v<64>(pm));
    const aie::vector<float, 64> d = aie::sub(bn, b);
    aie::store_v(pb, bn);
    for (int k = 0; k < 4; k++)
        aie::store_v(s + 64 * k, aie::add(aie::load_v<64>(s + 64 * k), d));
}

// a head's o += P0 V0 + P1 V1 (v: per d-block the two records' panels, interleaved by the memtile; o in place: ol, os
// the same panels as distinct restrict parameters, each read before it is written, so the loop pipelines)
template <int NB>
[[clang::always_inline]] inline void {{ p }}_pvh(const al::BV &P0, const al::BV &P1, const uint8_t *__restrict v,
                                                 const float *__restrict ol_, float *__restrict os_)
{
    al::BInR<aie_dm_resource::a> sv((const bfp16ebs8 *)v);
    const float __aie_dm_resource_c *__restrict ol = (const float __aie_dm_resource_c *)ol_;
    float __aie_dm_resource_d *__restrict os = (float __aie_dm_resource_d *)os_;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < NB; db++) {
        al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(ol)));
        c.mac(P0, aie::op_transpose(sv.pop()));
        c.mac(P1, aie::op_transpose(sv.pop()));
        aie::store_v(os, c.template to_vector<float>());
        ol += 64;
        os += 64;
    }
}

}  // namespace

extern "C" {

// A pass's start: o = 0, m = -1e30, b = 0 (the first group raises m), l = 0
void {{ p }}_init(float *__restrict ws)
{
    al::zero<{{ R * 2048 }}, 64>(AW(float, {{ w.o }}));
    float *__restrict m = AW(float, {{ w.m }});
    for (int i = 0; i < {{ R }}; i++)
        aie::store_v(m + 64 * i, aie::broadcast<float, 64>(-1e30f));
    al::zero<{{ R * 64 }}, 64>(AW(float, {{ w.b }}));
    al::zero<{{ R * 64 }}, 64>(AW(float, {{ w.l }}));
}

{% if TWO %}
// Scores of K object k2 (records 2 k2, 2 k2 + 1; per d-block their hi and lo panels: [d-block 32][record 2][hi | lo],
// the memtile interleaves the object's records, the two streams' objects one after the other) for both row panels
// (q: [d-block 32][panel 2][hi | lo]): per panel and record b + Q hi K hi + Q lo K hi + Q hi K lo, one MMUL chain each;
// every K panel popped once for both row panels, every Q panel once for both records (12 MMULs per d-block: their bound)
void {{ p }}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict kr, float *__restrict ws, int32_t k2)
{
{% if "qk" in ablate %}
    return;
{% endif %}
{% if "ev_qk" in ablate %}
    event0();
{% endif %}
    float *__restrict sc = AW(float, {{ w.sc }}) + 128 * k2;
    const float *__restrict pb = AW(float, {{ w.b }});
    const aie::accum<accfloat, 64> b0(aie::load_v<64>(pb)), b1(aie::load_v<64>(pb + 64));
    al::MM8 a00(b0), a01(b0), a10(b1), a11(b1);  // (row panel, record)
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)q);
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)kr);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        const al::BV K0h = ks.pop(), K0l = ks.pop(), K1h = ks.pop(), K1l = ks.pop();
        const al::BV q0h = qs.pop(), q0l = qs.pop(), q1h = qs.pop(), q1l = qs.pop();
        a00.mac(q0h, aie::op_transpose(K0h));
        a01.mac(q0h, aie::op_transpose(K1h));
        a10.mac(q1h, aie::op_transpose(K0h));
        a11.mac(q1h, aie::op_transpose(K1h));
        a00.mac(q0l, aie::op_transpose(K0h));
        a01.mac(q0l, aie::op_transpose(K1h));
        a10.mac(q1l, aie::op_transpose(K0h));
        a11.mac(q1l, aie::op_transpose(K1h));
        a00.mac(q0h, aie::op_transpose(K0l));
        a01.mac(q0h, aie::op_transpose(K1l));
        a10.mac(q1h, aie::op_transpose(K0l));
        a11.mac(q1h, aie::op_transpose(K1l));
    }
    aie::store_v(sc, a00.template to_vector<float>());
    aie::store_v(sc + 64, a01.template to_vector<float>());
    aie::store_v(sc + 256, a10.template to_vector<float>());
    aie::store_v(sc + 320, a11.template to_vector<float>());
{% if "ev_qk" in ablate %}
    event1();
{% endif %}
}
{% else %}
// Scores of K record k (8 positions; per d-block its hi and lo panels: the memtile interleaves the record's halves)
// for each row panel: b + Q hi K hi + Q lo K hi + Q hi K lo (q: per panel per d-block the Q panels hi, lo), one loop
// of two panel streams and three MMUL chains
void {{ p }}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict kr, float *__restrict ws, int32_t k)
{
{% if "qk" in ablate %}
    return;
{% endif %}
{% if "ev_qk" in ablate %}
    event0();
{% endif %}
    float *__restrict sc = AW(float, {{ w.sc }}) + 64 * k;
    const float *__restrict pb = AW(float, {{ w.b }});
    for (int h = 0; h < {{ R }}; h++) {
        al::MM8 a(aie::accum<accfloat, 64>(aie::load_v<64>(pb + 64 * h))), b(aie::zeros<accfloat, 64>()),
            c(aie::zeros<accfloat, 64>());
        al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)((const uint8_t *)q + {{ SLOT }} * h));
        al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *)kr);
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int db = 0; db < {{ ND }}; db++) {
            const al::BV qh = qs.pop(), ql = qs.pop(), Kh = ks.pop(), Kl = ks.pop();
            a.mac(qh, aie::op_transpose(Kh));
            b.mac(ql, aie::op_transpose(Kh));
            c.mac(qh, aie::op_transpose(Kl));
        }
        aie::store_v(sc + 256 * h, aie::add(aie::add(a.to_accum(), c.to_accum()), b.to_accum()).template to_vector<float>());
    }
{% if "ev_qk" in ablate %}
    event1();
{% endif %}
}

{% endif %}

// The group's softmax terms, diag groups before the pass's last: per row panel (its token group t = QG r + i % QG of
// the pass, in the pass's group t / 4) on its diagonal group keys after the query masked (record k vs record t % 4),
// the groups after it wholly (a panel's first groups are never masked: they lie before every pass's queries); a row's
// maximum m is raised (o, l rescaled, pass 1 again) on the pass's first group (g = 0) and when the group's exceeds it
// by 2^16; P = 2^(s - m) as bfp16 panels, l += P
void {{ p }}_soft(float *__restrict ws, float *__restrict pw, int32_t diag, int32_t r, int32_t g)
{
{% if "soft" in ablate %}
    return;
{% endif %}
{% if "ev_soft" in ablate %}
    event0();
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    for (int h = 0; h < {{ R }}; h++) {
        float *__restrict s = AW(float, {{ w.sc }}) + 256 * h;
        const int t = {{ QG }} * r + h % {{ QG }}, dd = diag - ({{ QG - 1 }} - (t >> 2));
        if (dd <= 0)
            {{ p }}_mask(s, dd, t & 3);
        float *__restrict pm = AW(float, {{ w.m }}) + 64 * h;
        float *__restrict pl = AW(float, {{ w.l }}) + 64 * h;
        if (!{{ p }}_pass1(s, pw) || g == 0) {
            {{ p }}_raise(s, AW(float, {{ w.o }}) + 2048 * h, pl, pm, AW(float, {{ w.b }}) + 64 * h, pw);
            {{ p }}_pass1(s, pw);
        }
        {{ p }}_pass2(pw, pl, (uint8_t *)s);
    }
{% if "ev_soft" in ablate %}
    event1();
{% endif %}
}

{% if TWO %}
// o += P V of the group's 4 V records for both row panels (v: [d-block 32][record 4], the memtile interleaves the
// object's records, the two streams' objects one after the other): every V panel popped once for both panels; o in
// place (ol, os the same panels as distinct restrict parameters, each read before it is written)
[[clang::always_inline]] inline void {{ p }}_pv2(const uint8_t *__restrict v, const uint8_t *__restrict pp,
                                                 const float *__restrict ol_, float *__restrict os_)
{
    al::BIn<> p0((const bfp16ebs8 *)pp), p1((const bfp16ebs8 *)(pp + 1024));
    const al::BV P0 = p0.pop(), P1 = p0.pop(), P2 = p0.pop(), P3 = p0.pop();
    const al::BV Q0 = p1.pop(), Q1 = p1.pop(), Q2 = p1.pop(), Q3 = p1.pop();
    al::BInR<aie_dm_resource::a> sv((const bfp16ebs8 *)v);
    const float __aie_dm_resource_c *__restrict o0 = (const float __aie_dm_resource_c *)ol_;
    const float __aie_dm_resource_c *__restrict o1 = (const float __aie_dm_resource_c *)(ol_ + 2048);
    float __aie_dm_resource_d *__restrict s0 = (float __aie_dm_resource_d *)os_;
    float __aie_dm_resource_d *__restrict s1 = (float __aie_dm_resource_d *)(os_ + 2048);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(o0))), d(aie::accum<accfloat, 64>(aie::load_v<64>(o1)));
        const al::BV V0 = sv.pop(), V1 = sv.pop(), V2 = sv.pop(), V3 = sv.pop();
        c.mac(P0, aie::op_transpose(V0));
        d.mac(Q0, aie::op_transpose(V0));
        c.mac(P1, aie::op_transpose(V1));
        d.mac(Q1, aie::op_transpose(V1));
        c.mac(P2, aie::op_transpose(V2));
        d.mac(Q2, aie::op_transpose(V2));
        c.mac(P3, aie::op_transpose(V3));
        d.mac(Q3, aie::op_transpose(V3));
        aie::store_v(s0, c.template to_vector<float>());
        aie::store_v(s1, d.template to_vector<float>());
        o0 += 64;
        o1 += 64;
        s0 += 64;
        s1 += 64;
    }
}

void {{ p }}_pv(const bfloat16 *__restrict v, float *__restrict ws)
{
{% if "pv" in ablate %}
    return;
{% endif %}
{% if "ev_pv" in ablate %}
    event0();
{% endif %}
    {{ p }}_pv2((const uint8_t *)v, AW(uint8_t, {{ w.sc }}), AW(float, {{ w.o }}), AW(float, {{ w.o }}));
{% if "ev_pv" in ablate %}
    event1();
{% endif %}
}
{% else %}
// o += P V of two of the group's V records (v: records 2 half, 2 half + 1, a panel per d-block of blocks over the
// positions){{ "; o -> ob, then ob -> o: loads and stores in different regions" if OB }}
void {{ p }}_pv(const bfloat16 *__restrict v, float *__restrict ws, int32_t half)
{
{% if "pv" in ablate %}
    return;
{% endif %}
{% if "ev_pv" in ablate %}
    event0();
{% endif %}
{% if OB %}
    const float *__restrict src = AW(float, {{ w.o }}) + (half ? {{ w.ob - w.o }} / 4 : 0);
    float *__restrict dst = AW(float, {{ w.o }}) + (half ? 0 : {{ w.ob - w.o }} / 4);
{% else %}
    const float *__restrict src = AW(float, {{ w.o }});
    float *__restrict dst = AW(float, {{ w.o }});
{% endif %}
    for (int h = 0; h < {{ R }}; h++) {
        al::BIn<> ps((const bfp16ebs8 *)(AW(uint8_t, {{ w.sc }}) + 1024 * h + 144 * half));
        const al::BV P0 = ps.pop(), P1 = ps.pop();
        const float *__restrict ol = src + 2048 * h;
        float *__restrict os = dst + 2048 * h;
        {{ p }}_pvh<{{ ND }}>(P0, P1, (const uint8_t *)v, ol, os);
    }
{% if "ev_pv" in ablate %}
    event1();
{% endif %}
}

{% endif %}

{% if not TWO %}
// A pass's slot i of the KV head's cores (one object: [d-block 32][72 x 2]): the core's own (me .. me + R - 1) into q
// (its R slots: Q panels at the pass's start, then sigmoid(gate) at its end)
void {{ p }}_qsel(const bfloat16 *__restrict a, bfloat16 *__restrict q, int32_t i, int32_t me)
{
    const unsigned k = (unsigned)(i - me);
    if (k < {{ R }}u)
        al::copy<{{ SLOT // 2 }}, 32>(q + {{ SLOT // 2 }} * k, a);
}
{% else %}
// A pass's slot pair i of the KV head's cores (slots 2 i, 2 i + 1, d-blocks 0 .. 15 in a, the rest in b's object after
// it: [d-block 32][slot 2][72 x 2]): the core's own (pair me) into q (Q panels at the pass's start, then
// sigmoid(gate) at its end)
void {{ p }}_qsel(const bfloat16 *__restrict a, bfloat16 *__restrict q, int32_t i, int32_t me)
{
    if (i == me)
        al::copy<{{ SLOT }}, 32>(q, a);
}

{% endif %}

// Row panel h's output: o / l sigmoid(gate) as the attn_out GEMM's 32 A panels (8 queries x 8 dims; sg: the row panels'
// gate slots)
void {{ p }}_fin(float *__restrict ws, const bfloat16 *__restrict sg, int32_t h, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<float, 8> ls = al::group8<false>(aie::load_v<64>(AW(float, {{ w.l }}) + 64 * h));
    const aie::vector<float, 16> inv = aie::inv(aie::concat(ls, aie::broadcast<float, 8>(1.0f)));
    const aie::vector<float, 64> iv = al::bcast8(inv.template extract<8>(0));
{% if TWO %}
    const bfloat16 *__restrict g = sg + 72 * h;  // (per d-block and panel [query 8][8] and 8 zeros)
{% else %}
    const bfloat16 *__restrict g = sg + {{ SLOT // 2 }} * h;  // (per d-block [query 8][8] and 8 zeros)
{% endif %}
    const float *__restrict o = AW(float, {{ w.o }}) + 2048 * h;
    al::BOut<> po((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        aie::vector<bfloat16, 64> gv;
        AIE_LOOP_UNROLL_FULL
        for (int q = 0; q < 8; q++)
            gv.insert(q, aie::load_v<8>(g + {{ 144 if TWO else 72 }} * db + 8 * q));
        const aie::vector<float, 64> y = al::mulff<64>(aie::load_v<64>(o + 64 * db), iv);
        po.push(al::mulc<64>(y, gv).template to_vector<bfp16ebs8>());
    }
}

}  // extern "C"
""")


class PrefillAttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 (module docstring) on the cache that already holds the chunk's K and V
    (``AttnPrepOperator``). Column c (q heads QC c .., KV head c // NPH), core row r; passes of 32 QG queries (``qg``):
    in pass b the core takes the QG token groups 4 QG b + QG r .. of each of the column's heads (QC QG row panels),
    streams the cache groups 0 .. p0 / 32 + QG (b + 1) - 1 (the KV head's stream broadcast to the 4 NPH cores of its
    columns) and writes their outputs as the attn_out GEMM's A panels. Runtime sequence: QS (the prep's), KV (one layer's cache), A (T x heads x 256 as bfp16
    panels [T / 8][heads x 32]). Parameters ``QA_N`` = p0 / 32 + 1 (the history groups every pass streams first, and the
    cores' trip count), ``QA_OFF`` (T's): the bf16 offset of the chunk's second group (``params``)."""

    STACK = 3072
    pfx = kernel_prefix("qaa")

    def __init__(self, *, T: int, cfg: QwenConfig, head_bytes: int, ablate: tuple[str, ...] = ()):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv": wrong results); "ev_qk",
        "ev_soft", "ev_pv": an event window around each call of the kernel, the cores' counters programmed (``COUNTERS``;
        two streams only)."""
        self.kvh = cfg.num_key_value_heads
        assert cfg.head_dim == HD and COLS % self.kvh == 0
        assert T % 32 == 0
        self.T, self.cfg, self.head_bytes = T, cfg, head_bytes
        self.nph = COLS // self.kvh
        self.ablate = tuple(ablate)
        self.qc = cfg.num_attention_heads // COLS
        self.qg = pass_groups(cfg, T)  # (query groups per core and head)
        self.R = self.qc * self.qg  # (row panels)
        self.H = cfg.num_attention_heads
        # (two row panels per core, a KV head's 4 columns: each KV head's cache on two streams, half of every object
        # each, so a core takes it on both input channels; its Q and gates ride ahead and behind on them; else one
        # stream, Q apart)
        self.two = two_streams(cfg, T)
        self.ob = self.R <= 2 and not self.two  # (o's second copy fits: regions 2 and 3)
        super().__init__()

    def workspace(self) -> dict[str, int]:
        R = self.R
        sizes = [("sc", R * 1024), ("m", R * 256), ("b", R * 256), ("l", R * 256), ("o", R * 8192)]
        return regions(sizes + [("ob", R * 8192)] if self.ob else sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        tri = sum(1 << (8 * q + k) for q in range(8) for k in range(8) if k > q)
        return _ATTN.render(p=p, w=Attr(self.workspace()), R=self.R, QG=self.qg, HD=HD, ND=ND, SLOT=SLOT,
                            tri=f"{tri:016X}", ablate=self.ablate, TWO=self.two, OB=self.ob)

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (COLS * T // 8 * 2 * self.qc * SLOT // 2,), torch.bfloat16),
                ArgSpec("in", (self.kvh * self.head_bytes // 2,), torch.bfloat16),
                ArgSpec("out", (T * self.H * HD * 9 // 16,), torch.bfloat16)]

    def params(self, p0: int) -> dict[str, int]:
        assert p0 % 32 == 0
        return {QA_N: p0 // 32 + 1, f"{QA_OFF}{self.T}": GROUP // 2 * (p0 // 32 + 1)}

    def get_mlir_artifact(self) -> Module:
        if self.two:
            return self._two_artifact()
        p, T, qc, H, qg, R = self.pfx, self.T, self.qc, self.H, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        kv_ty = np.ndarray[(AT.REC_K // 2,), bf]  # a K record, two V records, or a slot
        o_ty = np.ndarray[(ND * 36,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * ND * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        pw_ty = np.ndarray[(PW // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [kv_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, kv_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, pw_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [kv_ty, ws_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(f"{QA_OFF}{T}", np.int32)
        n_pass, nph, n_kvh = T // (32 * qg), self.nph, self.kvh
        assert nph >= 2, "a KV head's stream from two shims of its columns"
        n_q = nph * 4 * qg * qc  # (a pass's slots of a KV head's cores, one object each)

        # the KV head's stream from the shims of columns NPH h and NPH h + 1 (d-blocks 16 i .. of every piece from part
        # i: a K record, a V pair, a slot), joined by its memtile: one panel stream in the kernels' loops
        kv = [ObjectFifo(kv_ty, name=f"pa_kv{h}", depth=2) for h in range(n_kvh)]
        kv3 = [kv[h].prod().join([0, SLOT // 4], tile=Tile(nph * h, 1), depths=[2, 2],
                                 obj_types=[np.ndarray[(SLOT // 4,), bf]] * 2, names=[f"pa_ks{h}_{i}" for i in range(2)])
               for h in range(n_kvh)]
        o3 = [ObjectFifo(om_ty, name=f"pa_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([ND * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"pa_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(kv_h, o_h, init, qsel, qk, soft, pv, fin, ws, pw, qb, n_p, row, me):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(n_q):
                        a = kv_h.acquire(1)
                        qsel(a, qb, i, me)
                        kv_h.release(1)
                    init(ws)
                    # read after this dispatch's first object: the host wrote it before the fills
                    n = n_p.read() + b * qg + (qg - 1)
                    for g in range32(n):
                        for k in range(4):
                            kr = kv_h.acquire(1)
                            qk(qb, kr, ws, k)
                            kv_h.release(1)
                        soft(ws, pw, n - 1 - g, row, g)
                        for half in range(2):
                            v = kv_h.acquire(1)
                            pv(v, ws, half)
                            kv_h.release(1)
                    for i in range_(n_q):  # (the gates into the Q slots' buffer)
                        a = kv_h.acquire(1)
                        qsel(a, qb, i, me)
                        kv_h.release(1)
                    for h in range(R):
                        o = o_h.acquire(1)
                        fin(ws, qb, h, o)
                        o_h.release(1)

        workers = [Worker(core, [kv[c // nph].cons(), o1[c][r].prod(), k_init, k_qsel, k_qk, k_soft, k_pv, k_fin,
                                 Buffer(ws_ty, name=f"pa_ws{c}_{r}"), Buffer(pw_ty, name=f"pa_pw{c}_{r}"),
                                 Buffer(q_ty, name=f"pa_qb{c}_{r}"), p_n, r, (c % nph * 4 * qg + qg * r) * qc],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        n_qs = COLS * T // 8 * qc * SLOT
        n_kv, n_a = n_kvh * self.head_bytes // 2, T * H * HD * 9 // 16
        hb, sl, blk = self.head_bytes // 2, SLOT // 2, 4 * qg * qc * SLOT  # (blk: a column's slots in a pass)

        def seq(QS, KV, A, kvh, oh):
            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):  # (the cores' counters: window cycles, lock stall)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in COUNTERS:
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):
                # (a joined object: the 4 rows' panel i; QG > 1 (one head): token groups i, QG + i, .. of the pass)
                # (passes in runs of at most 64: a BD repeats at most 64 times)
                g_st = ND * 36 if qg == 1 else H * ND * 36
                for b0 in range(0, n_pass, 64):
                    oh[c].drain(A, TensorAccessPattern((n_a,), c * qc * ND * 36 + b0 * 4 * qg * H * ND * 36,
                                                       [min(64, n_pass - b0), R, 4, ND * 36],
                                                       [4 * qg * H * ND * 36, g_st, qg * H * ND * 36, 1]), group=tg,
                                wait=True)
            inflight = []  # (a pass: one task of up to four BDs per part, at most two passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                more = qg * b + qg - 1  # (the chunk's groups after its first through the pass's)
                for (h, i), f in zip(((h, i) for h in range(n_kvh) for i in range(2)), kvh, strict=True):
                    # (part i: d-blocks 16 i .. of every piece (a cache object's two); the pass's slots of the head's
                    # columns [column NPH][token group 4 QG][head] one after the other in QS, a token group [Q |
                    # sigmoid(gate)][head])
                    q0, k0 = (b * COLS + nph * h) * blk + i * SLOT // 4, h * hb + i * SLOT // 4
                    q_dims, q_str = [1, nph * 4 * qg, qc, SLOT // 4], [0, qc * SLOT, sl, 1]
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0, q_dims, q_str)),
                             ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 3, 2, SLOT // 4],
                                                               [0, 4640, SLOT // 2, 1]), length_parameter=p_n)]
                    if more:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 3 * more, 2, SLOT // 4],
                                                                       [0, 4640, SLOT // 2, 1]),
                                               offset_parameter=p_off))
                    parts.append(ChainPart(QS, TensorAccessPattern((n_qs,), q0 + qc * sl, q_dims, q_str)))
                    chained_fill(f, QS, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 1:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [kv3[h][i].prod(tile=Tile(nph * h + i, 0)) for h in range(n_kvh) for i in range(2)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        return module

    def _two_artifact(self) -> Module:
        """Two row panels per core (``two_streams``: QG token groups of one q head, or a token group of QC = 2
        heads): stream s of KV head h carries d-blocks 16 s .. 16 s + 15 of every cache object (a K object's two
        records, hi and lo panels; a V object's two pairs of records) from two shims (columns NPH h + 2 s and the next:
        d-blocks 8 i .. of each from part i) through its memtile (column NPH h + 2 s: the pieces interleaved per
        d-block, ``part_walk``), broadcast to the head's 4 NPH cores, which take the two on their two input channels:
        with one stream the shim channel and each core's input bound the history's re-streams (the 1B at 16K: 24.5
        GB/s for its 2 KV heads), with one shim channel per stream ~40 GB/s for the 4.
        Per pass each stream carries first the halves of the pass's Q slots of the head's cores ([column NPH][token
        group 4 QG][head QC], from the pass-major QS), two slots an object, then the history and the chunk's groups
        before the pass's, then the halves of their sigmoid(gate) slots: each core copies its own pair (``qsel``)."""
        p, T, H, qc, qg, R = self.pfx, self.T, self.H, self.qc, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        h_ty = np.ndarray[(SLOT // 2,), bf]  # (half a K or V object, the halves of two slots)
        o_ty = np.ndarray[(ND * 36,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * ND * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        pw_ty = np.ndarray[(PW // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [h_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, h_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, pw_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [h_ty, ws_ty])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(f"{QA_OFF}{T}", np.int32)
        n_pass, nph, n_kvh = T // (32 * qg), self.nph, self.kvh
        n_q = nph * 4 * qg * qc  # (a pass's slots of a KV head's cores)

        streams = {(h, s): nph * h + 2 * s for h in range(n_kvh) for s in range(2)}
        # (each stream from the shims of columns NPH h + 2 s and the next, d-blocks 8 i .. of every piece from part i,
        # joined by its memtile: one shim channel each paced the history's re-streams)
        kv = {hs: ObjectFifo(h_ty, name=f"pa_k{'ab'[hs[1]]}{hs[0]}", depth=2, to_stream=part_walk()) for hs in streams}
        kv3 = {hs: kv[hs].prod().join([0, SLOT // 4], tile=Tile(c, 1), depths=[2, 2],
                                      obj_types=[np.ndarray[(SLOT // 4,), bf]] * 2,
                                      names=[f"pa_ks{hs[0]}_{hs[1]}_{i}" for i in range(2)])
               for hs, c in streams.items()}
        o3 = [ObjectFifo(om_ty, name=f"pa_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([ND * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"pa_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(a_h, b_h, o_h, init, qsel, qk, soft, pv, fin, ws, pw, qb, n_p, row, me):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(n_q // 2):
                        a = a_h.acquire(1)
                        b_h.acquire(1)  # (b's object follows a's: l1_layout)
                        qsel(a, qb, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    init(ws)
                    # read after this dispatch's first object: the host wrote it before the fills
                    n = n_p.read() + b * qg + (qg - 1)
                    for g in range32(n):
                        for k in range(2):
                            ka = a_h.acquire(1)
                            b_h.acquire(1)
                            qk(qb, ka, ws, k)
                            a_h.release(1)
                            b_h.release(1)
                        soft(ws, pw, n - 1 - g, row, g)
                        va = a_h.acquire(1)
                        b_h.acquire(1)
                        pv(va, ws)
                        a_h.release(1)
                        b_h.release(1)
                    for i in range_(n_q // 2):  # (the gates into the Q slots' buffer)
                        a = a_h.acquire(1)
                        b_h.acquire(1)
                        qsel(a, qb, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    for h in range(R):
                        o = o_h.acquire(1)
                        fin(ws, qb, h, o)
                        o_h.release(1)

        workers = []
        for c in range(COLS):
            h = c // nph
            for r in range(4):
                workers.append(Worker(core, [kv[h, 0].cons(), kv[h, 1].cons(), o1[c][r].prod(), k_init, k_qsel, k_qk,
                                             k_soft, k_pv, k_fin, Buffer(ws_ty, name=f"pa_ws{c}_{r}"),
                                             Buffer(pw_ty, name=f"pa_pw{c}_{r}"),
                                             Buffer(q_ty, name=f"pa_qb{c}_{r}"), p_n, r, (c % nph) * 4 + r],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs = COLS * T // 8 * qc * SLOT
        n_kv, n_a = n_kvh * self.head_bytes // 2, T * H * HD * 9 // 16
        hb, sl = self.head_bytes // 2, SLOT // 2
        blk = 4 * qg * qc * SLOT  # (a column's slots in a pass: per token group [its heads' Q | their sigmoid(gate)])

        def seq(QS, KV, A, kvh, oh):
            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):  # (the cores' counters: window cycles, lock stall)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in COUNTERS:
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):
                g_st = ND * 36 if qg == 1 else H * ND * 36  # (the row panels: heads, or token groups)
                for b0 in range(0, n_pass, 64):  # (runs of at most 64 passes)
                    oh[c].drain(A, TensorAccessPattern((n_a,), c * qc * ND * 36 + b0 * 4 * qg * H * ND * 36,
                                                       [min(64, n_pass - b0), R, 4, ND * 36],
                                                       [4 * qg * H * ND * 36, g_st, qg * H * ND * 36, 1]),
                                group=tg, wait=True)
            inflight = []  # (a pass: one task of up to four BDs per stream, at most two passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                more = qg * b + qg - 1  # (the chunk's groups after its first through the pass's)
                for ((h, s), i), f in zip(((hs, i) for hs in streams for i in range(2)), kvh, strict=True):
                    # (part i: d-blocks 16 s + 8 i .. of every piece (a cache object's two); QS pass-major: the pass's
                    # slots of the head's columns [column NPH][token group 4 QG] one after the other, a token group's [Q
                    # | sigmoid(gate)] of its QC heads)
                    q0 = (b * COLS + nph * h) * blk + (2 * s + i) * SLOT // 8  # (its quarter of the slots)
                    k0, run = h * hb + s * SLOT // 4 + i * SLOT // 8, SLOT // 8
                    q_dims, q_str = [1, nph * 4 * qg, qc, SLOT // 8], [0, qc * SLOT, SLOT // 2, 1]
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0, q_dims, q_str)),
                             ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 3, 2, run], [0, 4640, SLOT // 2, 1]),
                                       length_parameter=p_n)]
                    if more:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), k0, [1, 3 * more, 2, run],
                                                                       [0, 4640, SLOT // 2, 1]),
                                               offset_parameter=p_off))
                    parts.append(ChainPart(QS, TensorAccessPattern((n_qs,), q0 + qc * sl, q_dims, q_str)))
                    chained_fill(f, QS, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 1:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [kv3[hs][i].prod(tile=Tile(c + i, 0)) for hs, c in streams.items() for i in range(2)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        return module

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (aiecc puts the stack at 0): 0 the K / V records, an output buffer and the
        softmax's scratch, 1 the Q panels held through a pass, the other output buffer (and with o's second copy the
        scores, m, l), 2 and 3 o (its two copies, else the workspace): QK streams Q (1) against K (0), PV V (0) from one
        o copy into the other (two streams: their objects in regions 0 and 1). The 9B: all 64 KB, the 27B 61."""
        kv, o, q, w = AT.REC_K, ND * 72, self.R * SLOT, self.workspace()
        pw = self.STACK + 2 * kv + o
        if self.two:  # (the two streams' objects i in region i, b's after a's: the kernels read both from a's pointer; the
            # Q slots, then the gates, and the scores in region 2, o in 3)
            lay = {"pa_ka": [self.STACK, 0x4000], "pa_kb": [self.STACK + kv, 0x4000 + kv],
                   "pa_or": [self.STACK + 2 * kv, 0x4000 + 2 * kv], "pa_qb": [0x8000], "pa_pw": [pw],
                   "pa_ws": [0x10000 - w["size"]]}
            assert pw + PW <= 0x4000 and 0x4000 + 2 * kv + o <= 0x8000 and 0x8000 + q <= 0x10000 - w["size"]
            assert 0x10000 - w["size"] + w["o"] == 0xC000
            return lay
        else:
            lay = {"pa_kv": [self.STACK, self.STACK + kv], "pa_or": [self.STACK + 2 * kv, 0x4000 + q], "pa_qb": [0x4000]}
        ws = 0x8000 - w["o"] if self.ob else 0x8000
        lay |= {"pa_pw": [pw], "pa_ws": [ws]}
        assert pw + PW <= 0x4000 and 0x4000 + q + o <= ws and ws + w["size"] <= 0x10000
        return lay
