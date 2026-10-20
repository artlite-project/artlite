"""The prefill of the qwen35 full-attention layers: a T-token chunk at position p0 (a multiple of 32) from the attn_in
GEMMs' output G (``prefill.gemm``: per column [q + gate of its QC heads | K part | V part]: KV head h's dims in the NPH
= 8 / KV heads columns NPH h ..) to the attn_out GEMM's A.

``AttnPrepOperator``: per column c, cores 2 .. 1 + QC one per q head (the column's heads QC c ..), core 2 + QC the
column's KV role. Per 8 tokens a Q core turns its head's rows into the attention's operands: q / rms(q) q_norm, the
NeoX rotation of dims 0 .. 63, x log2(e) / 16 (scores in log2 units), as bfp16 panels of 8 queries x 8 dims, per d-block
hi then lo (4608 B), then sigmoid(gate) (256 bf16 per query, padded to 4608 B). QS holds per column and token group
[the column's heads' Q panels | their sigmoid(gate)] (``SLOT`` bytes per head each).
The 2 x KV heads jobs of the KV cores (``AttnPrepOperator.kv_job``: one every 8 / (2 KV heads) columns; 9B / 27B every
column, MiniCPM-V every other): K head h (rms(k) k_norm, rotated) as hi + lo records, then V head h as records of blocks
over the 8 positions, in the decode's cache layout (``overlay.kv``): the chunk's groups from p0 / 32 on (scratchpad
parameter ``QA_KV``), so the decode continues from them. cos / sin of the
tokens' positions from a table (``cs_table``; parameter ``QA_CS``: the row of the chunk's first token): the static rows
of every text position, or rows the host writes past them for a chunk of image tokens (M-RoPE: ``cs_rows``).

``PrefillAttnOperator``: per column 4 cores, passes of 32 QG queries (QG = 2 when a column holds one q head, the L1 of
the 9B's two heads: half the history's re-streams, else 1); core r takes the QG token groups QG r .. of the pass for
each of the column's heads (QC QG row panels: an O of QC QG x 8 KB) and streams the cache groups up to the pass's (the
KV head's stream broadcast to the 4 NPH cores of its columns): per K record the scores on the MMUL (Q hi + lo against K hi + lo, three
products: one loop, the KV head's memtile re-lays each record per d-block, [K hi | K lo] and the V object's two
records), per group of 32 positions the softmax (a row's maximum is raised, rescaling o and l, only when the group's
exceeds it by 2^16: the per-dimension norm weights leave scores unbounded), P = 2^(s - m) by the vexp2 exp2 (vexp2
builds an exact 2^n; 0.59 ULP) as bfp16 panels, o += P V; at the pass's end o / l sigmoid(gate) as the attn_out GEMM's
bfp16 A panels. Within 0.7% of the exact attention of its own operands, ~1% rms of llama.cpp's attn_gated.

Time (T = 512, per layer alone, at p0 = 0 / 4K / 16K): the 27B 1.38 / 6.89 / 22.1 ms (its compute compiled out at
16K: 20.6, the KV re-streams of the 32-query passes at the DDR ceiling), the 1B 0.97 / 2.47 / 7.23 (5.99: the cores'
QK binds).
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

from ...model_config import QwenConfig
from .. import act_lut
from ..base import JINJA_ENV, ArgSpec, ArtcOperator
from ..design import COLS, range32, stream_walk
from ..gemm.gemm_q import ChainPart, chained_fill
from ..kernel_source import kernel_prefix
from ..l1_layout import pin_l1_buffers
from ..prefill.gemm import Layout
from ..util import Attr, bf16_hilo, regions
from . import attn as AT

HD = AT.HD
ND = AT.ND
SLOT = AT.REC_K  # bytes per head of a token group's Q panels (and of its sigmoid(gate): 4096 B used)
GROUP = 3 * 9280  # bytes of a cache group (32 positions: two K objects, a V object)
QA_CS = "qa_cs"  # scratchpad parameter: bf16 offset of the chunk's rows in the cos / sin table (128 p0)
QA_KV = "qa_kv"  # scratchpad parameter: bf16 offset of the chunk's first cache group (GROUP / 2 * p0 / 32)
QA_N = "qa_n"  # scratchpad parameter: p0 / 32 + 1 (cache groups every attention pass streams first; the cores' count)
QA_OFF = "qa_off"  # scratchpad parameter: bf16 offset of the chunk's second cache group
PW = 2 * 512  # bytes of the attention softmax's scratch (P and f)
# (timing builds, ablate "ev_<kernel>": counter 0 the window's cycles, 1 lock stall, 2 all cycles, 3 the windows)
COUNTERS = ((0x31500, 33 | 34 << 8 | 26 << 16 | 26 << 24), (0x31504, 1 | 1 << 8 | 33 << 16 | 33 << 24))


def slot_walk() -> TensorAccessPattern:
    """QS's slots in DDR: a slot's 32 pairs of 36 values (a d-block's Q panels hi, lo; or its gate block) as [half 2]
    [value 2][d-block 16][36] (the attention's walks, ``half_walk``, read a half's d-blocks back in order): written so by
    the prep's join from the cores' [d-block][72]."""
    return TensorAccessPattern((SLOT // 2,), 0, [2, ND // 2, 2, 36], [SLOT // 4, 36, SLOT // 8, 1])


def half_walk(n: int = 1) -> TensorAccessPattern:
    """A memtile's walk of n halves of slots, or of K / V objects (``AT.REC_K`` / 2 bytes each: [value 2][d-block
    16][36], a K record's hi then lo panels, or a V object's two records): [half][d-block][value][36]."""
    return stream_walk(n * SLOT // 4, [(n, SLOT // 4), (ND // 2, 36), (2, SLOT // 8), (36, 1)])


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

namespace {

{{ sigmoid }}

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

// stage -> the 32 hi panels, then the 32 lo panels (a panel: the 8 rows' blocks of 8 dims)
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

// stage -> per d-block the hi panel, then the lo panel (Q: one stream for the attention's MMUL)
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
// (bf16 [8][256]), the table pass in place (act_lut.SIGMOID: within 0.54 bf16 ULP), then the blocks
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
    {{ lut }}_sigmoid_tab<{{ 8 * HD }}>(st, st, st, st);
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

// 8 tokens of a KV head's K (rows [half 0 | half 1] bf16), cs -> its cache record [hi panels | lo panels]
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
    {{ p }}_hilo(st, (uint8_t *)out);
}

// 8 tokens of a KV head's V (rows bf16) -> its cache record: per 8 dims a panel of blocks over the positions
void {{ p }}_v(const bfloat16 *__restrict v, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BOut<> o((bfp16ebs8 *)out);
    // (the rows' inserts unrolled: a rolled r makes insert(r, ..) a switch, ~26K cycles per call; unrolled II 14, 478)
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        aie::vector<bfloat16, 64> pd;  // [position][dim]
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++)
            pd.insert(r, aie::load_v<8>(v + {{ HD }} * r + 8 * db));
        o.push(aie::accum<accfloat, 64>(aie::transpose(pd, 8, 8)).template to_vector<bfp16ebs8>());
    }
}

}  // extern "C"
""")


class AttnPrepOperator(ArtcOperator):
    """The attention prep of a chunk (module docstring). Runtime sequence: G (attn_in's output, ``layout``), CS (the
    cos / sin table of ``cs_positions`` rows, as bf16 pairs), NRM (q_norm | k_norm fp32), QS (per column, token group
    and head a ``SLOT``; ``qs_T``: the tokens of a column's part, default T: a sub-chunk's part of a longer chunk's QS),
    KV (one layer's cache: the KV heads' of ``attn.AttnShape.head_bytes``). Parameters: ``QA_CS`` = 128 x the table row
    of the first token (default p0), ``QA_KV`` = GROUP / 2 * p0 / 32 (``params``)."""

    STACK = 3072
    pfx = kernel_prefix("qap")

    def __init__(self, *, T: int, cfg: QwenConfig, layout: Layout, head_bytes: int, cs_positions: int,
                 qs_T: int = 0):
        self.kvh = cfg.num_key_value_heads
        assert cfg.head_dim == HD and cfg.rope_dims == 2 * AT.ROT and COLS % (2 * self.kvh) == 0
        assert T % 32 == 0 and T // 8 <= 64, "whole cache groups; a BD repeats at most 64 times"
        assert {"q", "k", "v"} <= set(layout.off)
        self.T, self.cfg, self.layout = T, cfg, layout
        self.nph, self.per = COLS // self.kvh, self.kvh * HD // COLS  # (a KV head's columns, its dims in one)
        self.qc = cfg.num_attention_heads // COLS
        assert 1 <= self.qc <= 3, "a core per q head and the KV core in a column"
        self.head_bytes, self.cs_positions = head_bytes, cs_positions
        self.qs_T = qs_T or T
        assert self.qs_T >= T and self.qs_T % 8 == 0
        super().__init__()

    def workspace(self) -> dict[str, int]:
        return {"nrm": 0, "row": 8 * HD, "stage": 12 * HD, "size": 44 * HD}

    def get_kernel_src(self) -> str:
        p = self.pfx
        qs = bf16_hilo(math.log2(math.e) / 16)
        return _PREP.render(p=p, HD=HD, ND=ND, w=Attr(self.workspace()), eps=repr(self.cfg.rms_norm_eps), qs=qs,
                            AT_REC_K=AT.REC_K, sigmoid=act_lut.SIGMOID.render(), lut=act_lut.LUT,
                            tmax=repr(act_lut.SIGMOID.t_max))

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
        """bf16 of the QS argument: the columns' parts ``qs_T`` tokens apart, T tokens of the last."""
        g = 2 * self.qc * SLOT // 2  # (a token group's)
        return ((COLS - 1) * self.qs_T // 8 + self.T // 8) * g

    def params(self, p0: int, cs_row: int | None = None) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 whose tokens' cos / sin start at table row ``cs_row``
        (default p0)."""
        r = p0 if cs_row is None else cs_row
        assert p0 % 32 == 0 and 0 <= r and r + self.T <= self.cs_positions
        return {QA_CS: 128 * r, QA_KV: GROUP // 2 * (p0 // 32)}

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
        vr_ty = np.ndarray[(AT.REC_V // 2,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty])
        k_q = Kernel(f"{p}_q", "kernel.o", [row_ty, cs_ty, ws_ty, slot_ty])
        k_sg = Kernel(f"{p}_sg", "kernel.o", [row_ty, ws_ty, slot_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [kv_ty, cs_ty, ws_ty, kr_ty])
        k_v = Kernel(f"{p}_v", "kernel.o", [kv_ty, cs_ty, ws_ty, vr_ty])
        p_cs, p_kv = ScratchpadParameter(QA_CS, np.int32), ScratchpadParameter(QA_KV, np.int32)
        n_tg = T // 8

        a3 = [ObjectFifo(in_ty if c in jobs else inq_ty, name=f"qp_a{c}", depth=2) for c in range(COLS)]
        a1 = [a3[c].cons().split([8 * 2 * HD * h for h in range(qc + (c in jobs))], tile=Tile(c, 1),
                                 depths=[2] * (qc + (c in jobs)), obj_types=[row_ty] * qc + [kv_ty] * (c in jobs),
                                 names=[f"qp_ah{c}_{h}" for h in range(qc + (c in jobs))]) for c in range(COLS)]
        b = [ObjectFifo(cs_ty, name=f"qp_b{c}", depth=2) for c in range(COLS)]
        o3 = [ObjectFifo(qs_ty, name=f"qp_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([SLOT // 2 * h for h in range(qc)], tile=Tile(c, 1), depths=[2] * qc,
                                obj_types=[slot_ty] * qc, names=[f"qp_oh{c}_{h}" for h in range(qc)],
                                from_stream=[slot_walk()] * qc) for c in range(COLS)]
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

        workers = []
        for c in range(COLS):
            for h in range(qc + (c in jobs)):
                ws = Buffer(ws_ty, name=f"qp_ws{c}_{h}")
                if h < qc:
                    fn, args = core_q, [a1[c][h].cons(), b[c].cons(), o1[c][h].prod(), k_prm, k_q, k_sg, ws]
                else:
                    fn, args = core, [a1[c][h].cons(), b[c].cons(), k[c].prod(), k_prm,
                                      k_k if jobs[c][0] == "k" else k_v, ws]
                workers.append(Worker(fn, args, tile=Tile(c, 2 + h), stack_size=self.STACK,
                                      dynamic_objfifo_lowering=True, while_true=False))
        n_cs, n_kv = self.cs_positions * 128, self.kvh * self.head_bytes // 2
        nph, per = self.nph, self.per
        n_qs, qs_col = self.n_qs, self.qs_T // 8 * 2 * qc * SLOT // 2
        hb, g2, o2 = self.head_bytes // 2, GROUP // 2, 9280 // 2

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
                oh[c].drain(QS, TensorAccessPattern((n_qs,), c * qs_col, [1, 1, 1, n_tg * qc * SLOT], [0, 0, 0, 1]),
                            group=tg, wait=True)
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

namespace {

// Rows whose maximum grew: o, l scaled by 2^(m - mn), m = mn = max(m, rm) (rm: the rows' group maxima)
__attribute__((noinline)) void {{ p }}_rescale(float *__restrict o, float *__restrict pl, float *__restrict pm,
                                               const float *__restrict prm)
{
    const aie::vector<float, 64> m = aie::load_v<64>(pm);
    const aie::vector<float, 64> mn = aie::max(m, aie::load_v<64>(prm));
    const aie::vector<float, 64> a = al::exp2f<64>(aie::max(aie::sub(m, mn), aie::broadcast<float, 64>(-126.0f)));
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

// pass 1 of P = 2^(s - m) over a head's 4 panels of A = s + 192 - m, 32 lanes a step: v = bf16(A) (an integer for
// 128 <= A < 256), f = bf16(A - v), P = 2^(max(v, 129) - 192) exactly (vexp2 of max(v, 129) - 128: the 2^-64 in pass
// 2's constants; below 129 2^-63, f small: P ~ 0) -> pP, pF; whether every v <= 208 (s <= m + 16)
[[clang::always_inline]] inline bool {{ p }}_pexp1(const float *__restrict sl, bfloat16 *__restrict pP,
                                                   bfloat16 *__restrict pF)
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f), lo = aie::broadcast<bfloat16, 32>(129.0f);
    aie::vector<bfloat16, 32> mx = lo;
    AIE_LOOP_UNROLL(2)
    for (int t = 0; t < 8; t++) {
        aie::accum<accfloat, 32> A(aie::load_v<32>(sl));
        const aie::vector<bfloat16, 32> v0 = A.template to_vector<bfloat16>();
        mx = aie::max(mx, v0);
        aie::store_v(pF, aie::msc(A, v0, one).template to_vector<bfloat16>());
        aie::store_v(pP, al::exp2_acc<32>(aie::mac(aie::mul(aie::max(v0, lo), one), aie::broadcast<bfloat16, 32>(-128.0f),
                                                   one)));
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

// pass 2, a panel a step: 2^f = 1 + c1 f + f^2 (c2 + c3 f) (c1 fitted first, then c2, c3 with its rounding: 0.59 bf16
// ULP), times P: the panel's P as bfp16 into pp, l += P
__attribute__((noinline)) void {{ p }}_pass2(const float *__restrict pw, float *__restrict pl, uint8_t *__restrict pp)
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const bfloat16 *__restrict pP = (const bfloat16 *)pw;
    const bfloat16 *__restrict pF = (const bfloat16 *)pw + 256;
    al::BOut<> po((bfp16ebs8 *)pp);
    aie::accum<accfloat, 32> l0(aie::load_v<32>(pl)), l1(aie::load_v<32>(pl + 32));
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int t = 0; t < 4; t++) {
        aie::accum<accfloat, 32> r[2];
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 2; u++) {
            const aie::vector<bfloat16, 32> P = aie::load_v<32>(pP + 32 * u), f = aie::load_v<32>(pF + 32 * u);
            const aie::vector<bfloat16, 32> hi = aie::mac(aie::mul(f, aie::broadcast<bfloat16, 32>(0.0625f * 0x1p-64f)),
                                                          aie::broadcast<bfloat16, 32>(0.2412109375f * 0x1p-64f), one)
                                                     .template to_vector<bfloat16>();
            const aie::vector<bfloat16, 32> p1 = aie::mul(P, f).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 32> p2 = aie::mul(p1, f).template to_vector<bfloat16>();
            r[u] = aie::mul(P, aie::broadcast<bfloat16, 32>(0x1p-64f));
            r[u] = aie::mac(r[u], p1, aie::broadcast<bfloat16, 32>(0.69140625f * 0x1p-64f));
            r[u] = aie::mac(r[u], p2, hi);
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

// Scores of K record k (8 positions; per d-block its hi and lo panels: the memtile interleaves the record's halves{{ "; the two streams' objects one after the other" if TWO }})
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

// o += P V of two of the group's V records (v: records 2 half, 2 half + 1, a panel per d-block of blocks over the
// positions{{ "; the two streams' objects one after the other" if TWO }}){{ "; o -> ob, then ob -> o: loads and stores in different regions" if OB }}
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

{% if TWO %}
// A pass's slot i of the KV head's cores (its halves a, b, one per stream: [d-block 16][72] each): the core's own (me ..
// me + R - 1) into q (its R slots: Q panels at the pass's start, then sigmoid(gate) at its end)
void {{ p }}_qsel(const bfloat16 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict q, int32_t i,
                  int32_t me)
{
    const unsigned k = (unsigned)(i - me);
    if (k >= {{ R }}u)
        return;
    bfloat16 *__restrict d = q + {{ SLOT // 2 }} * k;
    al::copy<{{ SLOT // 4 }}, 32>(d, a);
    al::copy<{{ SLOT // 4 }}, 32>(d + {{ SLOT // 4 }}, b);
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
    const bfloat16 *__restrict g = sg + {{ SLOT // 2 }} * h;  // (per d-block [query 8][8] and 8 zeros)
    const float *__restrict o = AW(float, {{ w.o }}) + 2048 * h;
    al::BOut<> po((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        aie::vector<bfloat16, 64> gv;
        AIE_LOOP_UNROLL_FULL
        for (int q = 0; q < 8; q++)
            gv.insert(q, aie::load_v<8>(g + 72 * db + 8 * q));
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
    cores' trip count), ``QA_OFF``: the bf16 offset of the chunk's second group (``params``)."""

    STACK = 3072
    pfx = kernel_prefix("qaa")

    def __init__(self, *, T: int, cfg: QwenConfig, head_bytes: int, ablate: tuple[str, ...] = ()):
        """``ablate`` (timing experiments): kernel bodies compiled out ("qk", "soft", "pv": wrong results); "ev_qk",
        "ev_soft", "ev_pv": an event window around each call of the kernel, the cores' counters programmed (``COUNTERS``;
        two streams only)."""
        self.kvh = cfg.num_key_value_heads
        assert cfg.head_dim == HD and COLS % self.kvh == 0
        assert T % 32 == 0 and T // 32 <= 64
        self.T, self.cfg, self.head_bytes = T, cfg, head_bytes
        self.nph = COLS // self.kvh
        self.ablate = tuple(ablate)
        self.qc = cfg.num_attention_heads // COLS
        self.qg = 2 if self.qc == 1 and T % 64 == 0 else 1  # (query groups per core and head)
        self.R = self.qc * self.qg  # (row panels)
        self.H = cfg.num_attention_heads
        # (one q head per column: each KV head's cache on two streams, half of every object each, so a core takes it on
        # both input channels; its Q and gates ride ahead and behind on them; more q heads: one stream, Q apart)
        self.two = self.qc == 1 and self.nph >= 2
        self.ob = self.R <= 2  # (o's second copy fits: regions 2 and 3)
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
        return {QA_N: p0 // 32 + 1, QA_OFF: GROUP // 2 * (p0 // 32 + 1)}

    def get_mlir_artifact(self) -> Module:
        if self.two:
            return self._two_artifact()
        p, T, qc, H, qg, R = self.pfx, self.T, self.qc, self.H, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        qm_ty = np.ndarray[(4 * R * SLOT // 2,), bf]
        kv_ty = np.ndarray[(AT.REC_K // 2,), bf]  # a K record, or two V records
        o_ty = np.ndarray[(ND * 36,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * ND * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        pw_ty = np.ndarray[(PW // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, kv_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, pw_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [kv_ty, ws_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(QA_OFF, np.int32)
        n_pass = T // (32 * qg)

        q3 = [ObjectFifo(qm_ty, name=f"pa_q{c}", depth=2) for c in range(COLS)]
        q1 = [q3[c].cons().split([R * SLOT // 2 * r for r in range(4)], tile=Tile(c, 1), depths=[1] * 4,
                                 obj_types=[q_ty] * 4, names=[f"pa_qr{c}_{r}" for r in range(4)],
                                 to_stream=[half_walk(2 * R)] * 4) for c in range(COLS)]
        # the KV head's records through its memtile, each re-laid per d-block ([hi | lo] of a K record, the two records
        # of a V object): one panel stream in the kernels' loops
        nph, n_kvh = self.nph, self.kvh
        kv3 = [ObjectFifo(kv_ty, name=f"pa_ks{h}", depth=2) for h in range(n_kvh)]
        ilv = stream_walk(AT.REC_K // 2, [(ND, 36), (2, ND * 36), (36, 1)])
        kv = [kv3[h].cons().forward(tile=Tile(nph * h, 1), depth=2, name=f"pa_kv{h}", to_stream=ilv)
              for h in range(n_kvh)]
        o3 = [ObjectFifo(om_ty, name=f"pa_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([ND * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"pa_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(q_h, kv_h, o_h, init, qk, soft, pv, fin, ws, pw, n_p, row):
            for _ in forever_():
                for b in range_(n_pass):
                    q = q_h.acquire(1)
                    init(ws)
                    # read after this dispatch's first object: the host wrote it before the fills
                    n = n_p.read() + b * qg + (qg - 1)
                    for g in range32(n):
                        for k in range(4):
                            kr = kv_h.acquire(1)
                            qk(q, kr, ws, k)
                            kv_h.release(1)
                        soft(ws, pw, n - 1 - g, row, g)
                        for half in range(2):
                            v = kv_h.acquire(1)
                            pv(v, ws, half)
                            kv_h.release(1)
                    q_h.release(1)
                    sg = q_h.acquire(1)
                    for h in range(R):
                        o = o_h.acquire(1)
                        fin(ws, sg, h, o)
                        o_h.release(1)
                    q_h.release(1)

        workers = [Worker(core, [q1[c][r].cons(), kv[c // nph].cons(), o1[c][r].prod(), k_init, k_qk, k_soft, k_pv, k_fin,
                                 Buffer(ws_ty, name=f"pa_ws{c}_{r}"), Buffer(pw_ty, name=f"pa_pw{c}_{r}"), p_n, r],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        n_qs = COLS * T // 8 * 2 * qc * SLOT // 2
        n_kv, n_a = n_kvh * self.head_bytes // 2, T * H * HD * 9 // 16
        hb, sl = self.head_bytes // 2, SLOT // 2

        def seq(QS, KV, A, qh, kvh, oh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                base = c * n_qs // COLS  # [token group][Q panels | sigmoid(gate)][head][SLOT]; pass b: groups 4 QG b ..
                taps = [TensorAccessPattern((n_qs,), base + part * qc * sl, [n_pass, 4 * qg, 2 * qc, sl // 2],
                                            [4 * qg * 2 * qc * sl, 2 * qc * sl, sl // 2, 1]) for part in range(2)]
                chained_fill(qh[c], QS, taps, tg)
                # (a joined object: the 4 rows' panel i; QG > 1 (one head): token groups i, QG + i, .. of the pass)
                g_st = ND * 36 if qg == 1 else H * ND * 36
                oh[c].drain(A, TensorAccessPattern((n_a,), c * qc * ND * 36, [n_pass, R, 4, ND * 36],
                                                   [4 * qg * H * ND * 36, g_st, qg * H * ND * 36, 1]), group=tg,
                            wait=True)
            inflight = []  # (a pass's fills hold 2 of the shim tile's 16 BDs: at most two passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                more = qg * b + qg - 1  # (the chunk's groups after its first through the pass's)
                for h in range(n_kvh):  # the history (QA_N groups), then the chunk's groups before the pass's
                    kvh[h].fill(KV, TensorAccessPattern((n_kv,), h * hb, [1, 3, 4, 1152], [0, 4640, 1152, 1]),
                                length_parameter=p_n, group=tp, wait=not more)
                    if more:
                        kvh[h].fill(KV, TensorAccessPattern((n_kv,), h * hb, [1, 3 * more, 4, 1152],
                                                            [0, 4640, 1152, 1]),
                                    offset_parameter=p_off, group=tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 1:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(q3)],
                           [f.prod(tile=Tile(nph * h, 0)) for h, f in enumerate(kv3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        module = Program(NPU2(), rt, workers=workers).resolve_program()
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for k, v in lay.items() if name.startswith(k)), None),
                       buffer=lambda name: next((v[0] for k, v in lay.items() if name.startswith(k)), None))
        return module

    def _two_artifact(self) -> Module:
        """One q head per column: stream s of KV head h carries d-blocks 16 s .. 16 s + 15 of every cache object (a K
        record's hi and lo panels, a V object's two records) from its own shim through its own memtile (the column
        NPH h + 2 s: re-laid per d-block, ``half_walk``), broadcast to the head's 4 NPH cores, which take the two on
        their two input channels: with one stream the shim channel and each core's input bound the history's
        re-streams (the 1B at 16K: 24.5 GB/s for its 2 KV heads). Per pass each stream carries first the halves of the
        pass's Q slots of the head's cores ([column NPH][token group 4 QG], from QS), then the history and the chunk's
        groups before the pass's, then the halves of their sigmoid(gate) slots: each core copies its own (``qsel``)."""
        p, T, H, qg, R = self.pfx, self.T, self.H, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        h_ty = np.ndarray[(SLOT // 4,), bf]  # (half a K record, a V object's half, half a slot)
        o_ty = np.ndarray[(ND * 36,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * ND * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        pw_ty = np.ndarray[(PW // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [h_ty, h_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, h_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, pw_ty, np.int32, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [h_ty, ws_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(QA_OFF, np.int32)
        n_pass, nph, n_kvh = T // (32 * qg), self.nph, self.kvh
        n_q = nph * 4 * qg  # (a pass's slots of a KV head's cores)

        streams = {(h, s): nph * h + 2 * s for h in range(n_kvh) for s in range(2)}
        kv3 = {hs: ObjectFifo(h_ty, name=f"pa_ks{hs[0]}_{hs[1]}", depth=2) for hs in streams}
        kv = {hs: kv3[hs].cons().forward(tile=Tile(c, 1), depth=2, name=f"pa_k{'ab'[hs[1]]}{hs[0]}",
                                         to_stream=half_walk()) for hs, c in streams.items()}
        o3 = [ObjectFifo(om_ty, name=f"pa_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([ND * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"pa_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(a_h, b_h, o_h, init, qsel, qk, soft, pv, fin, ws, pw, qb, n_p, row, me):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(n_q):
                        a = a_h.acquire(1)
                        bb = b_h.acquire(1)
                        qsel(a, bb, qb, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    init(ws)
                    # read after this dispatch's first object: the host wrote it before the fills
                    n = n_p.read() + b * qg + (qg - 1)
                    for g in range32(n):
                        for k in range(4):
                            ka = a_h.acquire(1)
                            b_h.acquire(1)  # (b's object follows a's: l1_layout)
                            qk(qb, ka, ws, k)
                            a_h.release(1)
                            b_h.release(1)
                        soft(ws, pw, n - 1 - g, row, g)
                        for half in range(2):
                            va = a_h.acquire(1)
                            b_h.acquire(1)
                            pv(va, ws, half)
                            a_h.release(1)
                            b_h.release(1)
                    for i in range_(n_q):  # (the gates into the Q slots' buffer)
                        a = a_h.acquire(1)
                        bb = b_h.acquire(1)
                        qsel(a, bb, qb, i, me)
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
                                             Buffer(q_ty, name=f"pa_qb{c}_{r}"), p_n, r, (c % nph) * 4 * qg + qg * r],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs = COLS * T // 8 * 2 * SLOT // 2
        n_kv, n_a = n_kvh * self.head_bytes // 2, T * H * HD * 9 // 16
        hb, sl, half = self.head_bytes // 2, SLOT // 2, SLOT // 4
        qs_col = T // 8 * 2 * sl  # (bf16 of a column's part of QS: per token group [Q slot | gate slot])

        def seq(QS, KV, A, kvh, oh):
            sync_parameters()
            if any(a.startswith("ev_") for a in self.ablate):  # (the cores' counters: window cycles, lock stall)
                for c in range(COLS):
                    for r in range(4):
                        for reg, val in COUNTERS:
                            npu_write32(reg, val, column=c, row=r + 2)
            tg = TaskGroup()
            for c in range(COLS):
                oh[c].drain(A, TensorAccessPattern((n_a,), c * ND * 36, [n_pass, R, 4, ND * 36],
                                                   [4 * qg * H * ND * 36, H * ND * 36, qg * H * ND * 36, 1]),
                            group=tg, wait=True)
            inflight = []  # (a pass: one task of up to four BDs per stream, at most two passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                more = qg * b + qg - 1  # (the chunk's groups after its first through the pass's)
                for (h, s), f in zip(streams, kvh, strict=True):
                    q0 = nph * h * qs_col + 4 * qg * b * 2 * sl + s * half  # (the pass's token groups of the head's columns)
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), q0, [1, nph, 4 * qg, half], [0, qs_col, 2 * sl, 1])),
                             ChainPart(KV, TensorAccessPattern((n_kv,), h * hb + s * SLOT // 8, [1, 3, 4, SLOT // 8],
                                                               [0, 4640, SLOT // 4, 1]), length_parameter=p_n)]
                    if more:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), h * hb + s * SLOT // 8,
                                                                       [1, 3 * more, 4, SLOT // 8],
                                                                       [0, 4640, SLOT // 4, 1]),
                                               offset_parameter=p_off))
                    parts.append(ChainPart(QS, TensorAccessPattern((n_qs,), q0 + sl, [1, nph, 4 * qg, half],
                                                                   [0, qs_col, 2 * sl, 1])))
                    chained_fill(f, QS, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 1:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [kv3[hs].prod(tile=Tile(c, 0)) for hs, c in streams.items()],
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
        o copy into the other. The 1B and 9B: all 64 KB, the 27B 61."""
        kv, o, q, w = AT.REC_K, ND * 72, self.R * SLOT, self.workspace()
        pw = self.STACK + 2 * kv + o
        if self.two:  # (the two streams' half objects in region 0, object i of b after object i of a: the kernels read
            # both from a's pointer; the Q slots, then the gates, in region 1)
            h = kv // 2
            lay = {"pa_ka": [self.STACK, self.STACK + 2 * h], "pa_kb": [self.STACK + h, self.STACK + 3 * h],
                   "pa_or": [self.STACK + 4 * h, 0x4000 + q], "pa_qb": [0x4000]}
        else:
            lay = {"pa_kv": [self.STACK, self.STACK + kv], "pa_or": [self.STACK + 2 * kv, 0x4000 + q], "pa_qr": [0x4000]}
        ws = 0x8000 - w["o"] if self.ob else 0x8000
        lay |= {"pa_pw": [pw], "pa_ws": [ws]}
        assert pw + PW <= 0x4000 and 0x4000 + q + o <= ws and ws + w["size"] <= 0x10000
        return lay
