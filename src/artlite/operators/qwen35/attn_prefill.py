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
positions from a table of every position (``cs_table``; parameter ``QA_CS``).

``PrefillAttnOperator``: per column 4 cores, passes of 32 QG queries (QG = 2 when a column holds one q head, the L1 of
the 9B's two heads: half the history's re-streams, else 1); core r takes the QG token groups QG r .. of the pass for
each of the column's heads (QC QG row panels: an O of QC QG x 8 KB) and streams the cache groups up to the pass's (the
KV head's stream broadcast to the 4 NPH cores of its columns): per K record the scores on the MMUL (Q hi + lo against K hi + lo, three
products: one loop, the KV head's memtile re-lays each record per d-block, [K hi | K lo] and the V object's two
records), per group of 32 positions the softmax (a row's maximum is raised, rescaling o and l, only when the group's
exceeds it by 2^16: the per-dimension norm weights leave scores unbounded), P = 2^(s - m) by the vexp2 exp2 (vexp2
builds an exact 2^n; 0.59 ULP) as bfp16 panels, o += P V; at the pass's end o / l sigmoid(gate) as the attn_out GEMM's
bfp16 A panels. Within 0.7% of the exact attention of its own operands, ~1% rms of llama.cpp's attn_gated.

Time (27B, T = 512, per layer alone): attention 1.30 / 7.93 ms at
p0 = 0 / 4K (9.77 at 4K without the interleave; its compute compiled out: 5.86, the KV re-streams of the 32-query
passes at the DDR ceiling).
"""

import math

import numpy as np
import torch
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
from ..base import JINJA_ENV, ArgSpec, ArtcOperator
from ..design import COLS, range32, stream_walk
from ..gemm.gemm_q import chained_fill
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


def cs_table(cfg: QwenConfig, n: int) -> np.ndarray:
    """(n, 64) fp32: per position [cos 32 | sin 32] of the rotated pairs (``QwenConfig.rope_theta``)."""
    th = cfg.rope_theta(list(range(n))).astype(np.float64)
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


_PREP = JINJA_ENV.from_string(r"""
// The prep cores' workspace (bytes): nrm [q_norm | k_norm] fp32, row (256 fp32), stage [db 32][row 8][8] fp32
#define PW(T, off) ((T *)((uint8_t *)ws + (off)))

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

// The first object: q_norm, k_norm (256 fp32 each)
void {{ p }}_prm(const bfloat16 *__restrict n, float *__restrict ws)
{
    al::copy<{{ 2 * HD }}, 16>(PW(float, {{ w.nrm }}), (const float *)n);
}

// 8 tokens of a q head: q (rows [q 256 | gate 256] bf16), cs (8 x [cos 32 | sin 32] fp32) -> out: per d-block the
// Q panels hi, lo of q / rms(q) q_norm, rotated, x log2(e) / 16
void {{ p }}_q(const bfloat16 *__restrict q, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    for (int r = 0; r < 8; r++) {
        {{ p }}_nr(q + {{ 2 * HD }} * r, PW(float, {{ w.nrm }}), cs + 64 * r, row);
        for (int i = 0; i < {{ HD }}; i += 64)
            aie::store_v(row + i, al::mul_hilo<64, {{ qs[0] }}f, {{ qs[1] }}f>(aie::load_v<64>(row + i)));
        {{ p }}_scatter(row, st, r);
    }
    {{ p }}_hiloi(st, (uint8_t *)out);
}

// sigmoid(gate) of the 8 tokens = 1 / (1 + 2^(-gate log2 e)) -> sg (8 x 256 bf16)
void {{ p }}_sg(const bfloat16 *__restrict q, bfloat16 *__restrict sg)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<float, 32> one = aie::broadcast<float, 32>(1.0f);
    for (int r = 0; r < 8; r++) {
        const bfloat16 *__restrict g = q + {{ 2 * HD }} * r + {{ HD }};
        for (int i = 0; i < {{ HD }}; i += 32) {
            const aie::vector<float, 32> gv = aie::accum<accfloat, 32>(aie::load_v<32>(g + i)).template to_vector<float>();
            const aie::vector<float, 32> e = al::exp2f<32>(al::log2e<32>(aie::sub(aie::zeros<float, 32>(), gv)));
            aie::store_v(sg + {{ HD }} * r + i,
                         aie::accum<accfloat, 32>(aie::inv(aie::add(e, one))).template to_vector<bfloat16>());
        }
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
    ``cs_table`` of every position, as bf16 pairs), NRM (q_norm | k_norm fp32), QS (per column, token group and head
    a ``SLOT``; ``qs_T``: the tokens of a column's part, default T: a sub-chunk's part of a longer chunk's QS), KV (one
    layer's cache: the KV heads' of ``attn.AttnShape.head_bytes``). Parameters: ``QA_CS`` = 128 p0, ``QA_KV`` = GROUP /
    2 * p0 / 32 (``params``)."""

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
                            AT_REC_K=AT.REC_K)

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

    def params(self, p0: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0."""
        assert p0 % 32 == 0 and p0 + self.T <= self.cs_positions
        return {QA_CS: 128 * p0, QA_KV: GROUP // 2 * (p0 // 32)}

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
        k_sg = Kernel(f"{p}_sg", "kernel.o", [row_ty, slot_ty])
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
                                obj_types=[slot_ty] * qc, names=[f"qp_oh{c}_{h}" for h in range(qc)]) for c in range(COLS)]
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
                        sg(x, o)
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
// token group QG r + i % QG of the pass) o [panel][db 32][8 x 8] fp32, sc [panel][record 4][8 x 8] fp32 (scores, log2
// units), m, l [panel][64] (lane 8 q + k: query q's running maximum / its partial sums), pp [panel][record] P panels,
// rm (a panel's group maxima)
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// 2^y of 32 lanes, y = A - 192 (|y| < 64; A an fp32 accumulator): n = round(y) by the bf16 rounding of A, f = A - (192 +
// n), vexp2 of n + 64 (exact: a non-negative integer; the 2^-64 is in the constants), 2^f = 1 + c1 f + f^2 (c2 + c3 f)
// (c1 fitted first, then c2, c3 with its rounding: 0.59 bf16 ULP)
[[clang::always_inline]] inline aie::accum<accfloat, 32> {{ p }}_exp2(aie::accum<accfloat, 32> A)
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::vector<bfloat16, 32> v = A.template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> f = aie::msc(A, v, one).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> P = al::exp2_acc<32>(aie::mac(aie::mul(v, one), aie::broadcast<bfloat16, 32>(-128.0f), one));
    const aie::vector<bfloat16, 32> hi = aie::mac(aie::mul(f, aie::broadcast<bfloat16, 32>(0.0625f * 0x1p-64f)),
                                                  aie::broadcast<bfloat16, 32>(0.2412109375f * 0x1p-64f), one)
                                             .template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> pf = aie::mul(P, f).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> pf2 = aie::mul(pf, f).template to_vector<bfloat16>();
    aie::accum<accfloat, 32> r = aie::mul(P, aie::broadcast<bfloat16, 32>(0x1p-64f));
    r = aie::mac(r, pf, aie::broadcast<bfloat16, 32>(0.69140625f * 0x1p-64f));
    return aie::mac(r, pf2, hi);
}

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

// P = 2^(s - m) of a head's 4 score panels, clamped at 2^-63 (A = max(s + 192 - m, 129)), in place (sl, ss: the
// scores as distinct restrict parameters, each read before it is written); l += P
__attribute__((noinline)) void {{ p }}_pexp(const float *__restrict sl, float *__restrict ss, const float *__restrict pm,
                                            float *__restrict pl)
{
    const aie::vector<float, 32> b0 = aie::sub(aie::broadcast<float, 32>(192.0f), aie::load_v<32>(pm));
    const aie::vector<float, 32> b1 = aie::sub(aie::broadcast<float, 32>(192.0f), aie::load_v<32>(pm + 32));
    const aie::vector<float, 32> lo = aie::broadcast<float, 32>(129.0f);
    aie::accum<accfloat, 32> l0(aie::load_v<32>(pl)), l1(aie::load_v<32>(pl + 32));
    for (int k = 0; k < 4; k++) {
        const aie::accum<accfloat, 32> p0 = {{ p }}_exp2(aie::accum<accfloat, 32>(aie::max(aie::add(aie::load_v<32>(sl), b0), lo)));
        const aie::accum<accfloat, 32> p1 = {{ p }}_exp2(aie::accum<accfloat, 32>(aie::max(aie::add(aie::load_v<32>(sl + 32), b1), lo)));
        l0 = aie::add(l0, p0);
        l1 = aie::add(l1, p1);
        aie::store_v(ss, p0.template to_vector<float>());
        aie::store_v(ss + 32, p1.template to_vector<float>());
        sl += 64;
        ss += 64;
    }
    aie::store_v(pl, l0.template to_vector<float>());
    aie::store_v(pl + 32, l1.template to_vector<float>());
}

// a head's 4 P panels (fp32) -> bfp16 panels
__attribute__((noinline)) void {{ p }}_ppack(const float *__restrict s, uint8_t *__restrict pp)
{
    al::BOut<> po((bfp16ebs8 *)pp);
    for (int k = 0; k < 4; k++) {
        po.push(aie::accum<accfloat, 64>(aie::load_v<64>(s)).template to_vector<bfp16ebs8>());
        s += 64;
    }
}

}  // namespace

extern "C" {

// A pass's start: o = 0, m = -1e30, l = 0
void {{ p }}_init(float *__restrict ws)
{
    al::zero<{{ R * 2048 }}, 64>(AW(float, {{ w.o }}));
    float *__restrict m = AW(float, {{ w.m }});
    for (int i = 0; i < {{ R }}; i++)
        aie::store_v(m + 64 * i, aie::broadcast<float, 64>(-1e30f));
    al::zero<{{ R * 64 }}, 64>(AW(float, {{ w.l }}));
}

// Scores of K record k (8 positions; per d-block its hi and lo panels: the memtile interleaves the record's halves)
// for each row panel: Q hi K hi + Q lo K hi + Q hi K lo (q: per panel per d-block the Q panels hi, lo), one loop of
// two panel streams and three MMUL chains
void {{ p }}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict kr, float *__restrict ws, int32_t k)
{
{% if "qk" in ablate %}
    return;
{% endif %}
    float *__restrict sc = AW(float, {{ w.sc }}) + 64 * k;
    for (int h = 0; h < {{ R }}; h++) {
        al::MM8 a(aie::zeros<accfloat, 64>()), b(aie::zeros<accfloat, 64>()), c(aie::zeros<accfloat, 64>());
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
}

// The group's softmax terms, diag groups before the pass's last: per row panel (its token group t = QG r + i % QG of
// the pass, in the pass's group t / 4) on its diagonal group keys after the query masked (record k vs record t % 4),
// the groups after it wholly (a panel's first groups are never masked: they lie before every pass's queries); a row's
// maximum m is raised (o, l rescaled) only when its group maximum exceeds it by 2^16; P = 2^(s - m) as bfp16 panels,
// l += P
void {{ p }}_soft(float *__restrict ws, int32_t diag, int32_t r)
{
{% if "soft" in ablate %}
    return;
{% endif %}
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::mask<64> tri = aie::mask<64>::from_uint64(0x{{ tri }}ull);  // lane 8 q + j: j > q
    const aie::vector<float, 64> ninf = aie::broadcast<float, 64>(-1e30f);
    for (int h = 0; h < {{ R }}; h++) {
        float *__restrict s = AW(float, {{ w.sc }}) + 256 * h;
        const int t = {{ QG }} * r + h % {{ QG }}, rec = t & 3, dd = diag - ({{ QG - 1 }} - (t >> 2));
        if (dd <= 0) {
            for (int k = 0; k < 4; k++) {
                if (dd < 0 || k > rec)
                    aie::store_v(s + 64 * k, ninf);
                else if (k == rec)
                    aie::store_v(s + 64 * k, aie::select(aie::load_v<64>(s + 64 * k), ninf, tri));
            }
        }
        const aie::vector<float, 64> v = aie::max(aie::max(aie::load_v<64>(s), aie::load_v<64>(s + 64)),
                                                  aie::max(aie::load_v<64>(s + 128), aie::load_v<64>(s + 192)));
        float *__restrict pm = AW(float, {{ w.m }}) + 64 * h;
        float *__restrict pl = AW(float, {{ w.l }}) + 64 * h;
        // (each lane against its row's maximum: the rows' group maxima only when one is raised, rarely)
        if (!aie::lt(aie::sub(v, aie::load_v<64>(pm)), aie::broadcast<float, 64>(16.0f)).full()) {
            float *__restrict prm = AW(float, {{ w.rm }});
            aie::store_v(prm, al::bcast8(al::group8<true>(v)));
            {{ p }}_rescale(AW(float, {{ w.o }}) + 2048 * h, pl, pm, prm);
        }
        {{ p }}_pexp(s, s, pm, pl);
        {{ p }}_ppack(s, AW(uint8_t, {{ w.pp }}) + 288 * h);
    }
}

// a head's o += P0 V0 + P1 V1 (v: per d-block the two records' panels, interleaved by the memtile; o in place: ol, os
// the same panels as distinct restrict parameters, each read before it is written, so the loop pipelines)
[[clang::always_inline]] inline void {{ p }}_pvh(const al::BV &P0, const al::BV &P1, const uint8_t *__restrict v,
                                                 const float *__restrict ol_, float *__restrict os_)
{
    al::BInR<aie_dm_resource::a> sv((const bfp16ebs8 *)v);
    const float __aie_dm_resource_c *__restrict ol = (const float __aie_dm_resource_c *)ol_;
    float __aie_dm_resource_d *__restrict os = (float __aie_dm_resource_d *)os_;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(ol)));
        c.mac(P0, aie::op_transpose(sv.pop()));
        c.mac(P1, aie::op_transpose(sv.pop()));
        aie::store_v(os, c.template to_vector<float>());
        ol += 64;
        os += 64;
    }
}

// o += P V of two of the group's V records (v: records 2 half, 2 half + 1, a panel per d-block of blocks over the
// positions)
void {{ p }}_pv(const bfloat16 *__restrict v, float *__restrict ws, int32_t half)
{
{% if "pv" in ablate %}
    return;
{% endif %}
    for (int h = 0; h < {{ R }}; h++) {
        al::BIn<> ps((const bfp16ebs8 *)(AW(uint8_t, {{ w.pp }}) + 288 * h + 144 * half));
        const al::BV P0 = ps.pop(), P1 = ps.pop();
        float *__restrict o = AW(float, {{ w.o }}) + 2048 * h;
        {{ p }}_pvh(P0, P1, (const uint8_t *)v, o, o);
    }
}

// Row panel h's output: o / l sigmoid(gate) as the attn_out GEMM's 32 A panels (8 queries x 8 dims; sg: [8][256] bf16)
void {{ p }}_fin(float *__restrict ws, const bfloat16 *__restrict sg, int32_t h, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<float, 8> ls = al::group8<false>(aie::load_v<64>(AW(float, {{ w.l }}) + 64 * h));
    const aie::vector<float, 16> inv = aie::inv(aie::concat(ls, aie::broadcast<float, 8>(1.0f)));
    const aie::vector<float, 64> iv = al::bcast8(inv.template extract<8>(0));
    const bfloat16 *__restrict g = sg + {{ SLOT // 2 }} * h;
    const float *__restrict o = AW(float, {{ w.o }}) + 2048 * h;
    al::BOut<> po((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        aie::vector<bfloat16, 64> gv;
        AIE_LOOP_UNROLL_FULL
        for (int q = 0; q < 8; q++)
            gv.insert(q, aie::load_v<8>(g + {{ HD }} * q + 8 * db));
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
        """``ablate`` (timing experiments, wrong results): kernel bodies compiled out ("qk", "soft", "pv")."""
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
        super().__init__()

    def workspace(self) -> dict[str, int]:
        R = self.R
        sizes = [("o", R * 8192), ("sc", R * 1024), ("m", R * 256), ("l", R * 256), ("pp", R * 288), ("rm", 256)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        tri = sum(1 << (8 * q + k) for q in range(8) for k in range(8) if k > q)
        return _ATTN.render(p=p, w=Attr(self.workspace()), R=self.R, QG=self.qg, HD=HD, ND=ND, SLOT=SLOT,
                            tri=f"{tri:016X}", ablate=self.ablate)

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (COLS * T // 8 * 2 * self.qc * SLOT // 2,), torch.bfloat16),
                ArgSpec("in", (self.kvh * self.head_bytes // 2,), torch.bfloat16),
                ArgSpec("out", (T * self.H * HD * 9 // 16,), torch.bfloat16)]

    def params(self, p0: int) -> dict[str, int]:
        assert p0 % 32 == 0
        return {QA_N: p0 // 32 + 1, QA_OFF: GROUP // 2 * (p0 // 32 + 1)}

    def get_mlir_artifact(self) -> Module:
        p, T, qc, H, qg, R = self.pfx, self.T, self.qc, self.H, self.qg, self.R
        bf = np.dtype[bfloat16]
        q_ty = np.ndarray[(R * SLOT // 2,), bf]
        qm_ty = np.ndarray[(4 * R * SLOT // 2,), bf]
        kv_ty = np.ndarray[(AT.REC_K // 2,), bf]  # a K record, or two V records
        o_ty = np.ndarray[(ND * 36,), bf]  # 32 A panels
        om_ty = np.ndarray[(4 * ND * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, kv_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, np.int32, np.int32])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [kv_ty, ws_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, q_ty, np.int32, o_ty])
        p_n, p_off = ScratchpadParameter(QA_N, np.int32), ScratchpadParameter(QA_OFF, np.int32)
        n_pass = T // (32 * qg)

        q3 = [ObjectFifo(qm_ty, name=f"pa_q{c}", depth=2) for c in range(COLS)]
        q1 = [q3[c].cons().split([R * SLOT // 2 * r for r in range(4)], tile=Tile(c, 1), depths=[1] * 4,
                                 obj_types=[q_ty] * 4, names=[f"pa_qr{c}_{r}" for r in range(4)]) for c in range(COLS)]
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

        def core(q_h, kv_h, o_h, init, qk, soft, pv, fin, ws, n_p, row):
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
                        soft(ws, n - 1 - g, row)
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
                                 Buffer(ws_ty, name=f"pa_ws{c}_{r}"), p_n, r],
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
                       buffer=lambda name: lay["pa_ws"][0] if name.startswith("pa_ws") else None)
        return module

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (aiecc puts the stack at 0): 0 the K / V records and an output buffer,
        1 the Q panels held through a pass and the other output buffer, 2 and 3 the workspace (O, the scores, P): QK
        streams Q (1) against K (0), PV V (0) into O (2, 3). 27B: 61 of 64 KB."""
        kv, o, q = AT.REC_K, ND * 72, self.R * SLOT
        lay = {"pa_kv": [self.STACK, self.STACK + kv], "pa_or": [self.STACK + 2 * kv, 0x4000 + q], "pa_qr": [0x4000],
               "pa_ws": [0x8000]}
        assert self.STACK + 2 * kv + o <= 0x4000 and 0x4000 + q + o <= 0x8000
        assert 0x8000 + self.workspace()["size"] <= 0x10000
        return lay
