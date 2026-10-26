"""The attention of the Gemma prefill: a T-token chunk at position p0 (a multiple of 32) of one layer, from the qkv
GEMMs' output G (``prefill.gemm``: per column c [q of head c (hd) | the column's quarter of [K_h; V_h], h = c / 4,
own-cache layers]) to the O GEMM's A panels, with the K / V records in the decode's cache layout (``decode.kv``:
per KV head groups of 3 objects of 9216 B, [K obj | K obj | V obj]; a sliding-window cache after its guard,
``overlay.GUARD``), so the decode continues from them. Head dims: 256 (sliding-window layers: every pair (i, i + 128)
rotated, theta 10K, a window of 512 positions) and 512 (full layers: pairs (i, i + 256) of i < 64, theta 1M);
q / k RMS-normed per head with the product of their constant weights (and log2 e) as one score constant (the decode's
``attn.score_constant``), v RMS-normed without a weight; scores unscaled.

``GemmaPrepOperator``: per column c core (c, 2) turns head c's q rows into the attention's operands, per 8 tokens:
q / rms(q), rotated (fp32), times the score constant, as bfp16 panels of 8 queries x 8 dims, per d-block hi then lo
(QS: per token group and column; head_dim 512: pass-major, [pass][KV head][token group][column], and per half of the
d-blocks hi panels then lo, which the attention's memtile interleaves). Cores (4 h, 3) and (4 h + 2, 3) write KV head
h's K (normed, rotated, hi + lo panel records) and V (bf16(v / rms(v)), per d-block a panel of blocks over the 8 positions) into the cache: the chunk's
groups from p0 on (scratchpad parameters ``params``). cos / sin of the positions from a table of every position
(``cs_table``). The score constant comes as the first object of the cos / sin stream (``prm_object``).

``GemmaAttnOperator``: per column c (query head c) 4 cores, passes of 32 queries; core r takes queries 8 r .. of the
pass and streams the KV head's cache (broadcast to the 16 cores of the head's 4 columns; head_dim 512: two streams, one
per input channel of each core, each with half of every cache object: ``GemmaAttnOperator._full_artifact``): per K
record the scores on the MMUL (Q hi + lo against K hi + lo: three products), per group the softmax with a running maximum
(qwen35's ``attn_prefill``: raised only by 2^16; masked scores far below a floor, so a row without keys in a group
adds nothing), P = 2^(s - m) as bfp16 hi + lo panels (single ones cost 2x the cache error by layer 23), o += P V; at the pass's end o / l as the O GEMM's A panels.
Sliding-window passes read 17 groups of 32 positions (the window's 16 before the pass and the pass's own: one BD
iterated per pass); groups before position 0 (the guard) are skipped, the window's first and the pass's own masked
per record. Full passes read every group from position 0 (the history's length a parameter, as qwen35's).
"""

import math
from dataclasses import dataclass

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

from artlite.gemm.gemm_q import ChainPart, chained_fill
from artlite.model_config import GemmaConfig
from artlite.operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from artlite.operators.design import COLS, range32, stream_walk
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.l1_layout import pin_l1_buffers
from artlite.operators.util import Attr, regions
from artlite.prefill.gemm import Layout

from . import attn as AT

OBJ = 9216  # a cache object
QH = AT.QH  # query heads per KV head
WINDOW_GROUPS = 17  # groups a sliding-window pass streams (the window's 16 before it, its own)
# scratchpad parameters
QA_CS = {256: "g4_cs_s", 512: "g4_cs_f"}  # bf16 offset of the chunk's rows in the cos / sin table
QA_KV = {256: "g4_kv_s", 512: "g4_kv_f"}  # bf16 offset of the chunk's first cache group (a sliding-window cache: after
# its guard)
QA_SW = "g4_sw"  # sliding-window attention: bf16 offset of the window of the chunk's first pass (p0 / 32 groups)
QA_FN = "g4_fn"  # full attention: the history's groups every pass streams first (p0 / 16 + 2)
QA_FO = "g4_fo"  # full attention: bf16 offset of the group after them
QA_P8 = "g4_p8"  # p0 / 8 (the full attention cores' masks)
QA_P32 = "g4_p32"  # p0 / 32 (the sliding-window attention cores' masks)


@dataclass(frozen=True)
class Kind:
    """A layer kind's attention shapes."""

    hd: int
    rot: int  # rotated pairs (i, i + hd / 2), i < rot
    kp: int = 2  # K panels per record (2: hi + lo, 1: single)

    @property
    def nd(self) -> int:
        return self.hd // 8

    @property
    def swa(self) -> bool:
        return self.hd == 256

    @property
    def pg(self) -> int:  # positions per cache group
        return 32 if self.swa else 16

    @property
    def rec_k(self) -> int:  # bytes of a K record (8 positions: hi | lo panels, or single ones)
        return 72 * self.kp * self.nd

    @property
    def group(self) -> int:  # bytes of a cache group (kp K objects, the V object)
        return (self.kp + 1) * OBJ

    @property
    def unit(self) -> int:  # bytes of the stream's unit: a hi + lo K record, two single ones, a V object's two records
        return 144 * self.nd

    @property
    def qp(self) -> int:  # bytes of a token group's Q panels
        return 144 * self.nd

    @property
    def recs(self) -> int:  # records per group
        return self.pg // 8

    @property
    def k_units(self) -> int:  # K units per group
        return self.pg // 8 if self.kp == 2 else self.pg // 16

    @property
    def v_units(self) -> int:  # V units (two records) per group
        return self.pg // 16


def kind_of(cfg: GemmaConfig, hd: int, kp: int = 2) -> Kind:
    return Kind(hd, AT.GemmaAttn(cfg).rot[0 if hd == 256 else 1], kp)


def cs_table(cfg: GemmaConfig, kind: Kind, n: int) -> np.ndarray:
    """(n, 2 rot) fp32: per position [cos rot | sin rot] of the kind's rotated pairs."""
    inv = cfg.rope_inv_freq(swa=kind.swa)[:kind.rot]
    th = np.outer(np.arange(n, dtype=np.float64), inv)
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


def prm_object(kind: Kind, c: float) -> np.ndarray:
    """The prep's parameter object (one cos / sin object of 8 positions, fp32): the score constant first."""
    out = np.zeros(8 * 2 * kind.rot, np.float32)
    out[0] = c
    return out


_PREP = JINJA_ENV.from_string(r"""
// The prep cores' workspace (bytes): c [64] fp32 (the score constant), row (HD fp32), stage [db ND][row 8][8] fp32,
// vn [8][HD] bf16
#define PW(T, off) ((T *)((uint8_t *)ws + (off)))

namespace {

// x (HD bf16) -> row = x / rms(x) (fp32), the pairs (i, i + HD / 2) of i < ROT rotated by cs [cos ROT | sin ROT]
__attribute__((noinline)) void {{ p }}_nr(const bfloat16 *__restrict x, const float *__restrict cs, float *__restrict row)
{
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    for (int i = 0; i < {{ HD }}; i += 64)
        ss = aie::mac_square(ss, aie::load_v<64>(x + i));
    const float inv = al::inv_rms<{{ HD }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    aie::vector<bfloat16, 64> ih, il;  // (x bf16: x 1 / rms as its two products with 1 / rms's parts)
    al::bf16_hilo<64>(inv, ih, il);
    for (int i = 0; i < {{ HD }}; i += 64) {
        const aie::vector<bfloat16, 64> xv = aie::load_v<64>(x + i);
        aie::store_v(row + i, aie::mac(aie::mul(xv, ih), xv, il).template to_vector<float>());
    }
{% if ROT %}
    for (int i = 0; i < {{ ROT }}; i += 32) {
        const aie::vector<float, 32> a = aie::load_v<32>(row + i), b = aie::load_v<32>(row + {{ HD // 2 }} + i);
        const aie::vector<float, 32> c = aie::load_v<32>(cs + i), s = aie::load_v<32>(cs + {{ ROT }} + i);
        aie::accum<accfloat, 32> o0, o1;
        al::rot2(a, b, c, s, o0, o1);
        aie::store_v(row + i, o0.template to_vector<float>());
        aie::store_v(row + {{ HD // 2 }} + i, o1.template to_vector<float>());
    }
{% endif %}
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

}  // namespace

extern "C" {

// The first object of the cos / sin stream: the score constant (Q cores)
void {{ p }}_prm(const bfloat16 *__restrict n, float *__restrict ws)
{
    aie::store_v(PW(float, {{ w.c }}), aie::broadcast<float, 64>(((const float *)n)[0]));
}

// 8 tokens of the q head (rows of HD bf16), cs (8 x [cos | sin] fp32) -> out: the Q panels of q / rms(q), rotated,
// times the score constant: {{ "per half of the d-blocks its hi panels, then its lo (the attention's memtile interleaves them)" if HALVES else "per d-block hi, lo" }}
void {{ p }}_q(const bfloat16 *__restrict q, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    const aie::vector<float, 64> cv = aie::load_v<64>(PW(float, {{ w.c }}));
    for (int r = 0; r < 8; r++) {
        {{ p }}_nr(q + {{ HD }} * r, cs + {{ 2 * ROT }} * r, row);
        for (int i = 0; i < {{ HD }}; i += 64)
            aie::store_v(row + i, al::mulff<64>(aie::load_v<64>(row + i), cv));
        {{ p }}_scatter(row, st, r);
    }
    const al::BV nI = al::neg_eye();
{% if HALVES %}
    for (int s = 0; s < 2; s++) {
        al::BOut<> ho((bfp16ebs8 *)((uint8_t *)out + {{ 72 * ND }} * s));
        al::BOut<> lo((bfp16ebs8 *)((uint8_t *)out + {{ 72 * ND }} * s + {{ 36 * ND }}));
        for (int db = 0; db < {{ ND // 2 }}; db++) {
            al::BV h, l;
            al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
            ho.push(h);
            lo.push(l);
            st += 64;
        }
    }
{% else %}
    al::BOut<> o((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        al::BV h, l;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
        o.push(h);
        o.push(l);
        st += 64;
    }
{% endif %}
}

// 8 tokens of a KV head's K (rows of HD bf16), cs -> its cache record {{ "[hi panels | lo panels]" if KP == 2 else "(single panels)" }}
void {{ p }}_k(const bfloat16 *__restrict k, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const float *__restrict cs = (const float *)cs_;
    float *__restrict row = PW(float, {{ w.row }});
    float *__restrict st = PW(float, {{ w.stage }});
    for (int r = 0; r < 8; r++) {
        {{ p }}_nr(k + {{ HD }} * r, cs + {{ 2 * ROT }} * r, row);
        {{ p }}_scatter(row, st, r);
    }
{% if KP == 2 %}
    const al::BV nI = al::neg_eye();
    al::BOut<> ho((bfp16ebs8 *)out);
    al::BOut<> lo((bfp16ebs8 *)((uint8_t *)out + {{ 72 * ND }}));
    for (int db = 0; db < {{ ND }}; db++) {
        al::BV h, l;
        al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(st)), nI, h, l);
        ho.push(h);
        lo.push(l);
        st += 64;
    }
{% else %}
    al::BOut<> o((bfp16ebs8 *)out);
    for (int db = 0; db < {{ ND }}; db++) {
        o.push(aie::accum<accfloat, 64>(aie::load_v<64>(st)).template to_vector<bfp16ebs8>());
        st += 64;
    }
{% endif %}
}

// 8 tokens of a KV head's V (rows of HD bf16) -> its cache record: bf16(v / rms(v)), per 8 dims a panel of blocks
// over the positions
void {{ p }}_v(const bfloat16 *__restrict v, const bfloat16 *__restrict cs_, float *__restrict ws, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    float *__restrict row = PW(float, {{ w.row }});
    bfloat16 *__restrict vn = PW(bfloat16, {{ w.vn }});
    for (int r = 0; r < 8; r++) {
        {{ p }}_nr0(v + {{ HD }} * r, row);
        for (int i = 0; i < {{ HD }}; i += 32)
            aie::store_v(vn + {{ HD }} * r + i, aie::accum<accfloat, 32>(aie::load_v<32>(row + i)).template to_vector<bfloat16>());
    }
    al::BOut<> o((bfp16ebs8 *)out);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND }}; db++) {
        aie::vector<bfloat16, 64> pd;  // [position][dim]
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++)
            pd.insert(r, aie::load_v<8>(vn + {{ HD }} * r + 8 * db));
        o.push(aie::accum<accfloat, 64>(aie::transpose(pd, 8, 8)).template to_vector<bfp16ebs8>());
    }
}

}  // extern "C"
""")

_NR0 = r"""
namespace {
// x (HD bf16) -> row = x / rms(x) (fp32), no rotation (V)
__attribute__((noinline)) void {{ p }}_nr0(const bfloat16 *__restrict x, float *__restrict row)
{
    aie::accum<accfloat, 64> ss = aie::zeros<accfloat, 64>();
    for (int i = 0; i < {{ HD }}; i += 64)
        ss = aie::mac_square(ss, aie::load_v<64>(x + i));
    const float inv = al::inv_rms<{{ HD }}>(aie::reduce_add(ss.template to_vector<float>()), {{ eps }}f);
    aie::vector<bfloat16, 64> ih, il;  // (x bf16: x 1 / rms as its two products with 1 / rms's parts)
    al::bf16_hilo<64>(inv, ih, il);
    for (int i = 0; i < {{ HD }}; i += 64) {
        const aie::vector<bfloat16, 64> xv = aie::load_v<64>(x + i);
        aie::store_v(row + i, aie::mac(aie::mul(xv, ih), xv, il).template to_vector<float>());
    }
}
}  // namespace
"""


class GemmaPrepOperator(ArtcOperator):
    """The attention prep of T tokens of one layer kind (module docstring). Runtime sequence: G (the qkv GEMM's
    output, ``layout``), CS (the kind's ``cs_table`` of every position, as bf16 pairs), PRM (``prm_object``), QS (per
    token group and column the Q panels, head_dim 512 pass-major: ``n_qs``; a sub-chunk's part of a chunk's), KV (the
    layer's cache: 2 KV heads of ``kv_bytes`` each, from the first head's guard). Parameters ``params``."""

    STACK = 2048
    pfx = kernel_prefix("g4ap")

    def __init__(self, *, T: int, cfg: GemmaConfig, hd: int, layout: Layout, kv_bytes: int, cs_positions: int,
                 kv_pad: int = 0, kp: int = 2):
        """``kp``: the cache's K panels per record (2: hi + lo, 1: single)."""
        assert cfg.num_attention_heads == COLS and all(k == COLS // QH for k in cfg.kv_heads)
        self.kind = kind_of(cfg, hd, kp)
        assert T % 32 == 0 and T // 8 <= 64 and {"q", "kv"} <= set(layout.off)
        self.T, self.cfg, self.layout, self.kv_bytes, self.cs_positions = T, cfg, layout, kv_bytes, cs_positions
        self.kv_pad = kv_pad
        super().__init__()

    def workspace(self) -> dict[str, int]:
        """Byte offsets: c (the score constant, 64 fp32), row (hd fp32), stage (Q / K cores: [nd][8][8] fp32) or vn (V
        cores: [8][hd] bf16)."""
        hd = self.kind.hd
        return {"c": 0, "row": 256, "stage": 256 + 4 * hd, "vn": 256 + 4 * hd, "size": 256 + 36 * hd}

    def get_kernel_src(self) -> str:
        p = self.pfx
        k = self.kind
        args = {"p": p, "HD": k.hd, "ND": k.nd, "ROT": k.rot, "eps": repr(self.cfg.rms_norm_eps), "KP": k.kp}
        return JINJA_ENV.from_string(_NR0).render(**args) + _PREP.render(w=Attr(self.workspace()), HALVES=not k.swa,
                                                                         **args)

    @property
    def n_cs(self) -> int:
        return self.cs_positions * 4 * self.kind.rot

    @property
    def n_qs(self) -> int:
        """bf16 of the QS argument: [token group][column][Q panels] (head_dim 512: [pass][KV head][token group]
        [column])."""
        return self.T // 8 * COLS * self.kind.qp // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        k = self.kind
        return [ArgSpec("in", (self.T * self.layout.N,), torch.bfloat16),
                ArgSpec("in", (self.n_cs,), torch.bfloat16),
                ArgSpec("in", (8 * 4 * k.rot,), torch.bfloat16),
                ArgSpec("out", (self.n_qs,), torch.bfloat16),
                ArgSpec("out", ((COLS // QH * self.kv_bytes + self.kv_pad) // 2,), torch.bfloat16)]

    def params(self, p0: int, guard: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk at position p0 (``guard``: the cache's groups before group 0)."""
        k = self.kind
        assert p0 % 32 == 0 and p0 + self.T <= self.cs_positions
        return {QA_CS[k.hd]: 4 * k.rot * p0, QA_KV[k.hd]: k.group // 2 * (guard + p0 // k.pg)}

    def get_mlir_artifact(self) -> Module:
        p, T, lay, k = self.pfx, self.T, self.layout, self.kind
        hd, N, pitch = k.hd, lay.N, lay.pitch
        bf = np.dtype[bfloat16]
        row_ty = np.ndarray[(8 * hd,), bf]  # 8 rows of a head
        cs_ty = np.ndarray[(8 * 4 * k.rot,), bf]  # 8 positions x [cos | sin] fp32 (the parameters: the score constant)
        qp_ty = np.ndarray[(k.qp // 2,), bf]
        kr_ty = np.ndarray[(k.rec_k // 2,), bf]
        vr_ty = np.ndarray[(36 * k.nd,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [cs_ty, ws_ty])
        k_q = Kernel(f"{p}_q", "kernel.o", [row_ty, cs_ty, ws_ty, qp_ty])
        k_k = Kernel(f"{p}_k", "kernel.o", [row_ty, cs_ty, ws_ty, kr_ty])
        k_v = Kernel(f"{p}_v", "kernel.o", [row_ty, cs_ty, ws_ty, vr_ty])
        p_cs, p_kv = ScratchpadParameter(QA_CS[hd], np.int32), ScratchpadParameter(QA_KV[hd], np.int32)
        n_tg = T // 8
        roles = {QH * h: "k" for h in range(COLS // QH)} | {QH * h + 2: "v" for h in range(COLS // QH)}

        # per column [q rows | the role's K or V rows] per token group, split in the memtile
        a3, a1 = [], []
        for c in range(COLS):
            if c in roles:
                f = ObjectFifo(np.ndarray[(2 * 8 * hd,), bf], name=f"gp_a{c}", depth=2)
                a1.append(f.cons().split([0, 8 * hd], tile=Tile(c, 1), depths=[2, 2], obj_types=[row_ty, row_ty],
                                         names=[f"gp_aq{c}", f"gp_ak{c}"]))
            else:
                f = ObjectFifo(row_ty, name=f"gp_a{c}", depth=2)
                a1.append([f])
            a3.append(f)
        b = [ObjectFifo(cs_ty, name=f"gp_b{c}", depth=2) for c in range(COLS)]
        od = 2 if k.swa else 1  # (output objects: head_dim 512's would not fit L1 twice)
        o = [ObjectFifo(qp_ty, name=f"gp_o{c}", depth=od) for c in range(COLS)]
        kv = {c: ObjectFifo(kr_ty if r == "k" else vr_ty, name=f"gp_k{c}", depth=od) for c, r in roles.items()}

        def core(a_h, b_h, o_h, prm, fn, ws):
            for _ in forever_():
                n = b_h.acquire(1)
                prm(n, ws)
                b_h.release(1)
                for _ in range_(n_tg):
                    x = a_h.acquire(1)
                    cs = b_h.acquire(1)
                    out = o_h.acquire(1)
                    fn(x, cs, ws, out)
                    o_h.release(1)
                    b_h.release(1)
                    a_h.release(1)

        workers = []
        for c in range(COLS):
            workers.append(Worker(core, [a1[c][0].cons(), b[c].cons(), o[c].prod(), k_prm, k_q,
                                         Buffer(ws_ty, name=f"gp_ws{c}")],
                                  tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False))
            if c in roles:
                workers.append(Worker(core, [a1[c][1].cons(), b[c].cons(), kv[c].prod(), k_prm,
                                             k_k if roles[c] == "k" else k_v, Buffer(ws_ty, name=f"gp_wk{c}")],
                                      tile=Tile(c, 3), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_cs, n_kv, n_qs = self.n_cs, (COLS // QH * self.kv_bytes + self.kv_pad) // 2, self.n_qs
        qh = k.qp // 2  # (a token group's Q panels of a column, bf16: rows of 1152 for a shim BD)
        hb, g2 = self.kv_bytes // 2, k.group // 2
        n_grp = T // k.pg

        def seq(G, CS, PRM, QS, KV, ah, bh, oh, kh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                taps = [TensorAccessPattern((T * N,), c * pitch + lay.off["q"], [n_tg, 1, 8, hd], [8 * N, 0, N, 1])]
                if c in roles:  # (the KV head's halves: this column's and the next one's kv blocks)
                    taps.append(TensorAccessPattern((T * N,), c * pitch + lay.off["kv"], [n_tg, 8, 2, hd // 2],
                                                    [8 * N, N, pitch, 1]))
                chained_fill(ah[c], G, taps, tg)
                bh[c].fill(PRM, TensorAccessPattern((8 * 4 * k.rot,), 0, [1, 1, 1, 8 * 4 * k.rot], [0, 0, 0, 1]),
                           group=tg)
                bh[c].fill(CS, TensorAccessPattern((n_cs,), 0, [1, 1, 1, T * 4 * k.rot], [0, 0, 0, 1]),
                           offset_parameter=p_cs, group=tg)
                if k.swa:
                    q_tap = TensorAccessPattern((n_qs,), c * qh, [1, n_tg, qh // 1152, 1152], [0, COLS * qh, 1152, 1])
                else:  # (pass-major: [pass][KV head][token group of the pass][column of the head])
                    q_tap = TensorAccessPattern((n_qs,), (4 * QH * (c // QH) + c % QH) * qh,
                                                [T // 32, 4, qh // 1152, 1152], [4 * COLS * qh, QH * qh, 1152, 1])
                oh[c].drain(QS, q_tap, group=tg, wait=True)
                if c in roles:
                    h = c // QH
                    if roles[c] == "k":  # a group's K records: its first kp objects
                        tap = TensorAccessPattern((n_kv,), h * hb, [1, n_grp, 8 * k.kp, OBJ // 16],
                                                  [0, g2, OBJ // 16, 1])
                    else:  # its V records: the last object
                        tap = TensorAccessPattern((n_kv,), h * hb + k.kp * OBJ // 2, [1, n_grp, 8, OBJ // 16],
                                                  [0, g2, OBJ // 16, 1])
                    kh[list(kv).index(c)].drain(KV, tap, offset_parameter=p_kv, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(n_cs,), bf], np.ndarray[(8 * 4 * k.rot,), bf],
                           np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o)],
                           [f.cons(tile=Tile(c, 0)) for c, f in kv.items()]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


_ATTN = JINJA_ENV.from_string(r"""
// The attention core's buffers: wo = o [db ND][8 x 8] fp32; ws (bytes): sc [record RG][8 x 8] fp32 (scores, log2
// units), m, l [64] (lane 8 q + k: query q's running maximum / its partial sums), pp [record] P panels hi, lo, rm (the group
// maxima), st [16] int32 (the group's first record relative to the core's queries, skip)
#define AW(T, off) ((T *)((uint8_t *)ws + (off)))

{# P = 2^(A - 192) (pexp clamps A at 129 itself: 2^-63) #}
{% set exp2, e192 = km.exp2(64, inline=True), km.exp2(32, bias=192, below="none", inline=True) %}
{{ exp2.src }}
{{ e192.src }}

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

// P = 2^(s - m) of the group's RG score panels, clamped at 2^-63 (A = max(s + 192 - m, 129)), in place (sl, ss: the
// scores as distinct restrict parameters, each read before it is written); l += P
__attribute__((noinline)) void {{ p }}_pexp(const float *__restrict sl, float *__restrict ss, const float *__restrict pm,
                                            float *__restrict pl)
{
    const aie::vector<float, 32> b0 = aie::sub(aie::broadcast<float, 32>(192.0f), aie::load_v<32>(pm));
    const aie::vector<float, 32> b1 = aie::sub(aie::broadcast<float, 32>(192.0f), aie::load_v<32>(pm + 32));
    const aie::vector<float, 32> lo = aie::broadcast<float, 32>(129.0f);
    aie::accum<accfloat, 32> l0(aie::load_v<32>(pl)), l1(aie::load_v<32>(pl + 32));
    for (int k = 0; k < {{ RG }}; k++) {
        const aie::accum<accfloat, 32> p0 = {{ e192 }}(aie::accum<accfloat, 32>(aie::max(aie::add(aie::load_v<32>(sl), b0), lo)));
        const aie::accum<accfloat, 32> p1 = {{ e192 }}(aie::accum<accfloat, 32>(aie::max(aie::add(aie::load_v<32>(sl + 32), b1), lo)));
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

// o of QB query blocks += P0 V0 + P1 V1, P as hi + lo panels (v: per d-block the two records' panels, interleaved by
// the memtile, popped once for every block{{ "; d-blocks 0 .. ND / 2 - 1 in v, the rest in v1: the two streams" if not SWA }};
// o in place: the ol, os pairs the same panels as distinct restrict parameters, each read before it is written, so
// the loop pipelines)
[[clang::always_inline]] inline void {{ p }}_pvh(const al::BV &P0, const al::BV &L0, const al::BV &P1, const al::BV &L1,
{% if QB == 2 %}
                                                 const al::BV &Q0, const al::BV &M0, const al::BV &Q1, const al::BV &M1,
{% endif %}
                                                 const uint8_t *__restrict v, {{ "const uint8_t *__restrict v1, " if not SWA }}const float *__restrict ol_,
                                                 float *__restrict os_{{ ", const float *__restrict ol1_, float *__restrict os1_" if QB == 2 }})
{
    const float __aie_dm_resource_c *__restrict ol = (const float __aie_dm_resource_c *)ol_;
    float __aie_dm_resource_d *__restrict os = (float __aie_dm_resource_d *)os_;
{% if QB == 2 %}
    const float __aie_dm_resource_c *__restrict ol1 = (const float __aie_dm_resource_c *)ol1_;
    float __aie_dm_resource_d *__restrict os1 = (float __aie_dm_resource_d *)os1_;
{% endif %}
{% for src in (["v"] if SWA else ["v", "v1"]) %}
    {
    al::BInR<aie_dm_resource::a> sv((const bfp16ebs8 *){{ src }});
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND if SWA else ND // 2 }}; db++) {
        const al::BV V0 = sv.pop(), V1 = sv.pop();
        al::MM8 c(aie::accum<accfloat, 64>(aie::load_v<64>(ol)));
{% if QB == 2 %}
        c.mac(P0, aie::op_transpose(V0));
        c.mac(L0, aie::op_transpose(V0));
        c.mac(P1, aie::op_transpose(V1));
        c.mac(L1, aie::op_transpose(V1));
        aie::store_v(os, c.template to_vector<float>());
        ol += 64;
        os += 64;
        al::MM8 c1(aie::accum<accfloat, 64>(aie::load_v<64>(ol1)));
        c1.mac(Q0, aie::op_transpose(V0));
        c1.mac(M0, aie::op_transpose(V0));
        c1.mac(Q1, aie::op_transpose(V1));
        c1.mac(M1, aie::op_transpose(V1));
        aie::store_v(os1, c1.template to_vector<float>());
        ol1 += 64;
        os1 += 64;
{% else %}
        al::MM8 d(aie::zeros<accfloat, 64>());  // (the lo products: a second chain)
        c.mac(P0, aie::op_transpose(V0));
        d.mac(L0, aie::op_transpose(V0));
        c.mac(P1, aie::op_transpose(V1));
        d.mac(L1, aie::op_transpose(V1));
        aie::store_v(os, aie::add(c.to_accum(), d.to_accum()).template to_vector<float>());
        ol += 64;
        os += 64;
{% endif %}
    }
    }
{% endfor %}
}

}  // namespace

extern "C" {

// A pass's start: o = 0, m = a floor far above the masked scores and below any real one, l = 0 (every query block)
void {{ p }}_init(float *__restrict ws, float *__restrict wo)
{
    al::zero<{{ QB * 64 * ND }}, 64>(wo);
    for (int b = 0; b < {{ QB }}; b++)
        aie::store_v(AW(float, {{ w.m }}) + 64 * b, aie::broadcast<float, 64>(-1e4f));
    al::zero<{{ QB * 64 }}, 64>(AW(float, {{ w.l }}));
}

// The group g of the pass: its first record relative to the core's first query block (rho: record 0 + rho holds its
// queries; block b's: rho - b), skipped when every position is before 0. {{ "Sliding window: rho = 4 g - 64 - %d r, x = p0 / 32 + %d b (skip: g + x < 16)" % (QB, QB) if SWA else "Full: rho = 2 g - x - r, x = p0 / 8 + 4 b" }}
void {{ p }}_grp(float *__restrict ws, int32_t g, int32_t x, int32_t r)
{
    int32_t *__restrict st = AW(int32_t, {{ w.st }});
{% if SWA %}
    st[0] = 4 * g - 64 - {{ QB }} * r;
    st[1] = g + x < 16;
{% else %}
    st[0] = 2 * g - x - r;
    st[1] = 0;
{% endif %}
}

{% if not SWA %}
// A pass's Q object i of the KV head's 16 (its halves a, b, one per stream): the core's own (i == me) into q
void {{ p }}_qsel(const bfloat16 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict q, int32_t i,
                  int32_t me)
{
    if (i != me)
        return;
    for (int j = 0; j < {{ ND * 72 // 2 }}; j += 32) {
        aie::store_v(q + j, aie::load_v<32>(a + j));
        aie::store_v(q + {{ ND * 72 // 2 }} + j, aie::load_v<32>(b + j));
    }
}

{% endif %}
{% if KP == 2 %}
// Scores of K record k (8 positions; per d-block its hi and lo panels, interleaved by the memtile{{ "; d-blocks
// 0 .. ND / 2 - 1 in kr, the rest in kl: the two streams" if not SWA }})
// for the QB query blocks: Q hi K hi, then Q lo K hi + Q hi K lo (q: per d-block {{ "the blocks' " if QB == 2 }}Q panels
// hi, lo), {{ "one loop" if SWA else "a loop per stream object" }} of two panel streams
void {{ p }}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict kr, {{ "const bfloat16 *__restrict kl, " if not SWA }}float *__restrict ws, int32_t k)
{
{% if "qk" in ablate %}
    return;
{% endif %}
    if (AW(int32_t, {{ w.st }})[1])
        return;
    float *__restrict sc = AW(float, {{ w.sc }}) + 64 * k;
    al::MM8 a(aie::zeros<accfloat, 64>()), b(aie::zeros<accfloat, 64>());
{% if QB == 2 %}
    al::MM8 a1(aie::zeros<accfloat, 64>()), b1(aie::zeros<accfloat, 64>());
{% endif %}
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)q);
{% for src in (["kr"] if SWA else ["kr", "kl"]) %}
    {
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *){{ src }});
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND if SWA else ND // 2 }}; db++) {
        const al::BV qh = qs.pop(), ql = qs.pop(), Kh = ks.pop(), Kl = ks.pop();
        a.mac(qh, aie::op_transpose(Kh));
        b.mac(ql, aie::op_transpose(Kh));
        b.mac(qh, aie::op_transpose(Kl));
{% if QB == 2 %}
        const al::BV qh1 = qs.pop(), ql1 = qs.pop();
        a1.mac(qh1, aie::op_transpose(Kh));
        b1.mac(ql1, aie::op_transpose(Kh));
        b1.mac(qh1, aie::op_transpose(Kl));
{% endif %}
    }
    }
{% endfor %}
    aie::store_v(sc, aie::add(a.to_accum(), b.to_accum()).template to_vector<float>());
{% if QB == 2 %}
    aie::store_v(sc + {{ 64 * RG }}, aie::add(a1.to_accum(), b1.to_accum()).template to_vector<float>());
{% endif %}
}

{% else %}
// Scores of K unit k (records 2 k, 2 k + 1; per d-block their panels, interleaved by the memtile{{ "; d-blocks
// 0 .. ND / 2 - 1 in kr, the rest in kl: the two streams" if not SWA }})
// for the QB query blocks: Q hi K + Q lo K (q: per d-block {{ "the blocks' " if QB == 2 }}Q panels hi, lo),
// {{ "one loop" if SWA else "a loop per stream object" }} of two panel streams
void {{ p }}_qk(const bfloat16 *__restrict q, const bfloat16 *__restrict kr, {{ "const bfloat16 *__restrict kl, " if not SWA }}float *__restrict ws, int32_t k)
{
{% if "qk" in ablate %}
    return;
{% endif %}
    if (AW(int32_t, {{ w.st }})[1])
        return;
    float *__restrict sc = AW(float, {{ w.sc }}) + 128 * k;
    al::MM8 a0(aie::zeros<accfloat, 64>()), a1(aie::zeros<accfloat, 64>());
{% if QB == 2 %}
    al::MM8 c0(aie::zeros<accfloat, 64>()), c1(aie::zeros<accfloat, 64>());
{% endif %}
    al::BInR<aie_dm_resource::a> qs((const bfp16ebs8 *)q);
{% for src in (["kr"] if SWA else ["kr", "kl"]) %}
    {
    al::BInR<aie_dm_resource::b> ks((const bfp16ebs8 *){{ src }});
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int db = 0; db < {{ ND if SWA else ND // 2 }}; db++) {
        const al::BV qh = qs.pop(), ql = qs.pop(), K0 = ks.pop(), K1 = ks.pop();
        a0.mac(qh, aie::op_transpose(K0));
        a1.mac(qh, aie::op_transpose(K1));
        a0.mac(ql, aie::op_transpose(K0));
        a1.mac(ql, aie::op_transpose(K1));
{% if QB == 2 %}
        const al::BV qh1 = qs.pop(), ql1 = qs.pop();
        c0.mac(qh1, aie::op_transpose(K0));
        c1.mac(qh1, aie::op_transpose(K1));
        c0.mac(ql1, aie::op_transpose(K0));
        c1.mac(ql1, aie::op_transpose(K1));
{% endif %}
    }
    }
{% endfor %}
    aie::store_v(sc, a0.template to_vector<float>());
    aie::store_v(sc + 64, a1.template to_vector<float>());
{% if QB == 2 %}
    aie::store_v(sc + {{ 64 * RG }}, c0.template to_vector<float>());
    aie::store_v(sc + {{ 64 * RG + 64 }}, c1.template to_vector<float>());
{% endif %}
}

{% endif %}
// The group's softmax terms, per query block b: per record its mask (relative record rho = st[0] - b + k: past the
// queries (rho > 0) or {{ "before their window (rho < -64): all; rho = 0: keys after the query; rho = -64: keys up to the query's" if SWA else "rho > 0: all; rho = 0: keys after the query" }}); a row's maximum is raised (o, l
// rescaled) only when its group maximum exceeds it by 2^16; P = 2^(s - m) as bfp16 hi + lo panels, l += P
void {{ p }}_soft(float *__restrict ws, float *__restrict wo)
{
{% if "soft" in ablate %}
    return;
{% endif %}
    const int32_t *__restrict st = AW(int32_t, {{ w.st }});
    if (st[1])
        return;
    aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::mask<64> tri = aie::mask<64>::from_uint64(0x{{ tri }}ull);  // lane 8 q + j: j > q
    const aie::vector<float, 64> ninf = aie::broadcast<float, 64>(-1e30f);
    AIE_LOOP_NO_UNROLL
    for (int bq = 0; bq < {{ QB }}; bq++) {
        float *__restrict s = AW(float, {{ w.sc }}) + {{ 64 * RG }} * bq;
        const int rho0 = st[0] - bq;
        for (int k = 0; k < {{ RG }}; k++) {
            const int rho = rho0 + k;
            if (rho > 0{{ " || rho < -64" if SWA }})
                aie::store_v(s + 64 * k, ninf);
            else if (rho == 0)
                aie::store_v(s + 64 * k, aie::select(aie::load_v<64>(s + 64 * k), ninf, tri));
{% if SWA %}
            else if (rho == -64)
                aie::store_v(s + 64 * k, aie::select(ninf, aie::load_v<64>(s + 64 * k), tri));
{% endif %}
        }
{% if RG == 4 %}
        const aie::vector<float, 64> v = aie::max(aie::max(aie::load_v<64>(s), aie::load_v<64>(s + 64)),
                                                  aie::max(aie::load_v<64>(s + 128), aie::load_v<64>(s + 192)));
{% else %}
        const aie::vector<float, 64> v = aie::max(aie::load_v<64>(s), aie::load_v<64>(s + 64));
{% endif %}
        float *__restrict pm = AW(float, {{ w.m }}) + 64 * bq;
        float *__restrict pl = AW(float, {{ w.l }}) + 64 * bq;
        // (each lane against its row's maximum: the rows' group maxima only when one is raised, rarely)
        if (!aie::lt(aie::sub(v, aie::load_v<64>(pm)), aie::broadcast<float, 64>(16.0f)).full()) {
            float *__restrict prm = AW(float, {{ w.rm }});
            aie::store_v(prm, al::bcast8(al::group8<true>(v)));
            {{ p }}_rescale(wo + {{ 64 * ND }} * bq, pl, pm, prm);
        }
        {{ p }}_pexp(s, s, pm, pl);
        const al::BV nI = al::neg_eye();
        al::BOut<> po((bfp16ebs8 *)(AW(uint8_t, {{ w.pp }}) + {{ 144 * RG }} * bq));
        for (int k = 0; k < {{ RG }}; k++) {
            al::BV h, l;
            al::split_hilo(aie::accum<accfloat, 64>(aie::load_v<64>(s)), nI, h, l);
            po.push(h);
            po.push(l);
            s += 64;
        }
    }
}

// o += P V of the group's V unit u (records 2 u, 2 u + 1: a panel per d-block of blocks over the positions{{ "; v1: the second record" if not SWA }})
void {{ p }}_pv(const bfloat16 *__restrict v, {{ "const bfloat16 *__restrict v1, " if not SWA }}float *__restrict ws, float *__restrict wo, int32_t u)
{
{% if "pv" in ablate %}
    return;
{% endif %}
    if (AW(int32_t, {{ w.st }})[1])
        return;
    al::BIn<> ps((const bfp16ebs8 *)(AW(uint8_t, {{ w.pp }}) + 288 * u));
    const al::BV P0 = ps.pop(), L0 = ps.pop(), P1 = ps.pop(), L1 = ps.pop();
{% if QB == 2 %}
    al::BIn<> ps1((const bfp16ebs8 *)(AW(uint8_t, {{ w.pp }}) + {{ 144 * RG }} + 288 * u));
    const al::BV Q0 = ps1.pop(), M0 = ps1.pop(), Q1 = ps1.pop(), M1 = ps1.pop();
    {{ p }}_pvh(P0, L0, P1, L1, Q0, M0, Q1, M1, (const uint8_t *)v, wo, wo, wo + {{ 64 * ND }}, wo + {{ 64 * ND }});
{% else %}
    {{ p }}_pvh(P0, L0, P1, L1, (const uint8_t *)v, {{ "(const uint8_t *)v1, " if not SWA }}wo, wo);
{% endif %}
}

// The pass's output: o / l of each query block as the O GEMM's ND A panels (8 queries x 8 dims)
void {{ p }}_fin(float *__restrict ws, float *__restrict wo, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    al::BOut<> po((bfp16ebs8 *)out);
    for (int bq = 0; bq < {{ QB }}; bq++) {
        const aie::vector<float, 8> ls = al::group8<false>(aie::load_v<64>(AW(float, {{ w.l }}) + 64 * bq));
        const aie::vector<float, 16> inv = aie::inv(aie::concat(ls, aie::broadcast<float, 8>(1.0f)));
        const aie::vector<float, 64> iv = al::bcast8(inv.template extract<8>(0));
        const float *__restrict o = wo + {{ 64 * ND }} * bq;
        for (int db = 0; db < {{ ND }}; db++) {
            po.push(aie::accum<accfloat, 64>(al::mulff<64>(aie::load_v<64>(o), iv)).template to_vector<bfp16ebs8>());
            o += 64;
        }
    }
}

}  // extern "C"
""")


class GemmaAttnOperator(ArtcOperator):
    """Causal attention of a T-token chunk at p0 of one layer kind (module docstring) on the cache that already holds
    the chunk's K and V (``GemmaPrepOperator``). Column c (query head c, KV head c / 4), core row r; passes of 32
    queries: in pass b the core takes the queries 32 b + 8 r .., streams the KV head's cache (broadcast to the 16 cores
    of its columns; head_dim 512 on both input channels) and writes their outputs as the O GEMM's A panels. Runtime
    sequence: QS (the prep's), KV (the
    layer's cache, from the first head's guard), A (T x 8 hd as bfp16 panels [T / 8][hd]). Parameters ``params``."""

    STACK = 3072
    pfx = kernel_prefix("g4aa")

    def __init__(self, *, T: int, cfg: GemmaConfig, hd: int, kv_bytes: int, kv_pad: int = 0, kp: int = 2,
                 ablate: tuple[str, ...] = ()):
        """``kv_pad``: bytes of the KV argument past the last head's cache (the walks of a chunk near the cache's end
        may reach past its tail: their parameters' range); ``kp``: the cache's K panels per record (2: hi + lo, 1:
        single); ``ablate`` (timing experiments, wrong results): kernel bodies compiled out ("qk", "soft", "pv")."""
        assert cfg.num_attention_heads == COLS and all(k == COLS // QH for k in cfg.kv_heads)
        self.kind = kind_of(cfg, hd, kp)
        assert not self.kind.swa or cfg.sliding_window == (WINDOW_GROUPS - 1) * 32
        # query blocks of 8 per core and pass (head_dim 256: two, 64-query passes; 512: one, L1)
        self.qb = 2 if self.kind.swa else 1
        self.pq = 32 * self.qb  # queries per pass
        self.n_pass = -(-T // self.pq)  # (a last pass past T: its queries' panels and outputs in the buffers' padding)
        assert T % 32 == 0 and self.n_pass <= 64
        self.wg = WINDOW_GROUPS + self.qb - 1  # (the groups a sliding-window pass streams)
        self.T, self.cfg, self.kv_bytes, self.kv_pad, self.ablate = T, cfg, kv_bytes, kv_pad, tuple(ablate)
        super().__init__()

    def workspace(self) -> dict[str, int]:
        rg, qb = self.kind.recs, self.qb
        sizes = [("sc", 256 * rg * qb), ("m", 256 * qb), ("l", 256 * qb), ("pp", 144 * rg * qb), ("rm", 256), ("st", 64)]
        return regions(sizes)

    def get_kernel_src(self) -> str:
        p = self.pfx
        k = self.kind
        tri = sum(1 << (8 * q + j) for q in range(8) for j in range(8) if j > q)
        return _ATTN.render(p=p, w=Attr(self.workspace()), ND=k.nd, RG=k.recs, KP=k.kp, SWA=k.swa, QB=self.qb,
                            tri=f"{tri:016X}", ablate=self.ablate)

    @property
    def n_qs(self) -> int:
        return COLS * self.n_pass * self.pq // 8 * self.kind.qp // 2

    def get_arg_spec(self) -> list[ArgSpec]:
        hd = self.kind.hd
        return [ArgSpec("in", (self.n_qs,), torch.bfloat16),
                ArgSpec("in", ((COLS // QH * self.kv_bytes + self.kv_pad) // 2,), torch.bfloat16),
                ArgSpec("out", (self.n_pass * self.pq * COLS * hd * 9 // 16,), torch.bfloat16)]

    def params(self, p0: int) -> dict[str, int]:
        assert p0 % 32 == 0
        if self.kind.swa:
            return {QA_P32: p0 // 32, QA_SW: self.kind.group // 2 * (p0 // 32)}
        return {QA_P8: p0 // 8, QA_FN: p0 // 16 + 2, QA_FO: self.kind.group // 2 * (p0 // 16 + 2)}

    def get_mlir_artifact(self) -> Module:
        if not self.kind.swa:
            return self._full_artifact()
        p, k, qb = self.pfx, self.kind, self.qb
        nd, hd = k.nd, k.hd
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        q_ty = np.ndarray[(qb * k.qp // 2,), bf]  # the core's query blocks' Q panels
        qm_ty = np.ndarray[(4 * qb * k.qp // 2,), bf]
        kv_ty = np.ndarray[(k.unit // 2,), bf]  # a K record, or two V records
        o_ty = np.ndarray[(qb * nd * 36,), bf]  # ND A panels per query block
        om_ty = np.ndarray[(4 * qb * nd * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), f32]
        wo_ty = np.ndarray[(64 * nd * qb,), f32]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty])
        k_grp = Kernel(f"{p}_grp", "kernel.o", [ws_ty, np.int32, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, kv_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, wo_ty])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [kv_ty, ws_ty, wo_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty])
        p_p8 = ScratchpadParameter(QA_P32, np.int32)
        p_sw = ScratchpadParameter(QA_SW, np.int32)
        n_pass, wg = self.n_pass, self.wg
        n_kv_heads = COLS // QH

        q3 = [ObjectFifo(qm_ty, name=f"ga_q{c}", depth=2) for c in range(COLS)]
        # (two query blocks: their Q panels interleaved per d-block, [db][block][hi | lo], one stream in the kernel)
        qw = k.qp // 2
        ilq = stream_walk(qb * qw, [(nd, 72), (qb, qw), (72, 1)])
        q1 = [q3[c].cons().split([qb * qw * r for r in range(4)], tile=Tile(c, 1), depths=[1] * 4,
                                 obj_types=[q_ty] * 4, names=[f"ga_qr{c}_{r}" for r in range(4)], to_stream=[ilq] * 4)
              for c in range(COLS)]
        # the KV head's units through its memtile, each re-laid per d-block ([hi | lo] of a K record, the two records
        # of a V unit): one panel stream in the kernels' loops (the stream of a head reads ~6 GB/s through one shim
        # channel: the bound of a sliding-window pass; a unit's halves through two channels joined in the memtile would
        # need 7 of its 6 input channels, with the column's Q split and output join)
        kv3 = [ObjectFifo(kv_ty, name=f"ga_ks{h}", depth=2) for h in range(n_kv_heads)]
        ilv = stream_walk(k.unit // 2, [(nd, 36), (2, nd * 36), (36, 1)])
        kv = [kv3[h].cons().forward(tile=Tile(QH * h, 1), depth=2, name=f"ga_kv{h}", to_stream=ilv)
              for h in range(n_kv_heads)]
        o3 = [ObjectFifo(om_ty, name=f"ga_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([qb * nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"ga_or{c}_{r}" for r in range(4)]) for c in range(COLS)]
        ku, vu = k.k_units, k.v_units

        def core(q_h, kv_h, o_h, init, grp, qk, soft, pv, fin, ws, wo, p8, row):
            for _ in forever_():
                for b in range_(n_pass):
                    q = q_h.acquire(1)
                    init(ws, wo)
                    x = p8.read() + b + b  # (p0 / 32 + 2 b; read after the first object: written before the fills)
                    for g in range_(wg):
                        grp(ws, g, x, row)
                        for kk in range(ku):
                            kr = kv_h.acquire(1)
                            qk(q, kr, ws, kk)
                            kv_h.release(1)
                        soft(ws, wo)
                        for u in range(vu):
                            v = kv_h.acquire(1)
                            pv(v, ws, wo, u)
                            kv_h.release(1)
                    q_h.release(1)
                    o = o_h.acquire(1)
                    fin(ws, wo, o)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for r in range(4):
                workers.append(Worker(core, [q1[c][r].cons(), kv[c // QH].cons(), o1[c][r].prod(), k_init, k_grp, k_qk,
                                             k_soft, k_pv, k_fin, Buffer(ws_ty, name=f"ga_ws{c}_{r}"),
                                             Buffer(wo_ty, name=f"ga_wo{c}_{r}"), p_p8, r],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs, n_kv, n_a = self.n_qs, (n_kv_heads * self.kv_bytes + self.kv_pad) // 2, n_pass * self.pq * COLS * hd * 9 // 16
        hb, g2, uw = self.kv_bytes // 2, k.group // 2, k.unit // 2  # (bf16: a head's cache, a group, a unit)

        def seq(QS, KV, A, qh, kvh, oh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):  # pass b: token groups 4 qb b .. of the column's head
                qh[c].fill(QS, TensorAccessPattern((n_qs,), c * qw, [1, n_pass * 4 * qb, qw // 1152, 1152],
                                                   [0, COLS * qw, 1152, 1]), group=tg)
                oh[c].drain(A, TensorAccessPattern((n_a,), c * nd * 36, [n_pass, 4 * qb, nd, 36],
                                                   [4 * qb * COLS * nd * 36, COLS * nd * 36, 36, 1]), group=tg, wait=True)
            # per pass the window's 16 groups before it and its own (one BD iterated per pass): their units' halves
            for h in range(n_kv_heads):
                kvh[h].fill(KV, TensorAccessPattern((n_kv,), h * hb, [n_pass, 1, wg * k.group // k.unit, uw],
                                                    [qb * g2, 0, uw, 1]), offset_parameter=p_sw, group=tg)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(q3)],
                           [f.prod(tile=Tile(QH * h, 0)) for h, f in enumerate(kv3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return self._pinned(Program(NPU2(), rt, workers=workers).resolve_program())

    def _full_artifact(self) -> Module:
        """Head_dim 512: a core's two input channels each carry half of the KV head's cache stream, as a fused attention
        whose tiles take Q + K on one inbound path and V on the other (arXiv 2609.21264), balanced: stream s of a KV
        head carries d-blocks 32 s .. 32 s + 31 of every cache object (a K record's hi and lo panels, a V object's two
        records), from its own shim through its own memtile (which interleaves the two parts per d-block),
        broadcast to the head's 16 cores; the kernels run a loop per stream object. With one stream of
        whole objects, each core's 8 B / cycle input channel bounds the layer (every position's 1728 bytes to each core
        once per 32-query pass). A pass's 16 Q objects ([token group of the pass][column of the head], the prep's
        pass-major QS, per half [hi | lo] panels) ride ahead of its cache objects, their halves on the two streams;
        each core copies its own."""
        p, k = self.pfx, self.kind
        nd, hd = k.nd, k.hd
        bf, f32 = np.dtype[bfloat16], np.dtype[np.float32]
        half = k.unit // 4  # (bf16 of a stream object: half a K record, one V record, half a Q object)
        h_ty = np.ndarray[(half,), bf]
        q_ty = np.ndarray[(k.qp // 2,), bf]  # the core's Q panels
        o_ty = np.ndarray[(nd * 36,), bf]  # ND A panels
        om_ty = np.ndarray[(4 * nd * 36,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), f32]
        wo_ty = np.ndarray[(64 * nd,), f32]
        k_init = Kernel(f"{p}_init", "kernel.o", [ws_ty, wo_ty])
        k_grp = Kernel(f"{p}_grp", "kernel.o", [ws_ty, np.int32, np.int32, np.int32])
        k_qsel = Kernel(f"{p}_qsel", "kernel.o", [h_ty, h_ty, q_ty, np.int32, np.int32])
        k_qk = Kernel(f"{p}_qk", "kernel.o", [q_ty, h_ty, h_ty, ws_ty, np.int32])
        k_soft = Kernel(f"{p}_soft", "kernel.o", [ws_ty, wo_ty])
        k_pv = Kernel(f"{p}_pv", "kernel.o", [h_ty, h_ty, ws_ty, wo_ty, np.int32])
        k_fin = Kernel(f"{p}_fin", "kernel.o", [ws_ty, wo_ty, o_ty])
        p_p8 = ScratchpadParameter(QA_P8, np.int32)
        p_fn = ScratchpadParameter(QA_FN, np.int32)
        p_fo = ScratchpadParameter(QA_FO, np.int32)
        n_pass, n_kv_heads, ku = self.n_pass, COLS // QH, k.k_units
        assert self.qb == 1 and k.v_units == 1 and k.qp // 2 == 2 * half

        streams = {(h, s): QH * h + 2 * s for h in range(n_kv_heads) for s in range(2)}  # (its shim's and memtile's
        # column)
        kv3 = {hs: ObjectFifo(h_ty, name=f"ga_ks{hs[0]}_{hs[1]}", depth=2) for hs in streams}
        ilv = stream_walk(half, [(nd // 2, 36), (2, half // 2), (36, 1)])
        kv = {hs: kv3[hs].cons().forward(tile=Tile(c, 1), depth=2, name=f"ga_k{'ab'[hs[1]]}{hs[0]}", to_stream=ilv)
              for hs, c in streams.items()}
        o3 = [ObjectFifo(om_ty, name=f"ga_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([nd * 36 * r for r in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                                obj_types=[o_ty] * 4, names=[f"ga_or{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(a_h, b_h, o_h, init, grp, qsel, qk, soft, pv, fin, ws, wo, q, p8, fn, me, row):
            for _ in forever_():
                for b in range_(n_pass):
                    for i in range_(4 * QH):
                        q0 = a_h.acquire(1)
                        q1 = b_h.acquire(1)
                        qsel(q0, q1, q, i, me)
                        a_h.release(1)
                        b_h.release(1)
                    init(ws, wo)
                    n = fn.read() + b + b  # (the history's groups, then the chunk's two per pass before this one)
                    x = p8.read() + b + b + b + b
                    for g in range32(n):
                        grp(ws, g, x, row)
                        for kk in range(ku):
                            kh = a_h.acquire(1)
                            kl = b_h.acquire(1)
                            qk(q, kh, kl, ws, kk)
                            a_h.release(1)
                            b_h.release(1)
                        soft(ws, wo)
                        v0 = a_h.acquire(1)
                        v1 = b_h.acquire(1)
                        pv(v0, v1, ws, wo, 0)
                        a_h.release(1)
                        b_h.release(1)
                    o = o_h.acquire(1)
                    fin(ws, wo, o)
                    o_h.release(1)

        workers = []
        for c in range(COLS):
            for r in range(4):
                workers.append(Worker(core, [kv[c // QH, 0].cons(), kv[c // QH, 1].cons(), o1[c][r].prod(), k_init,
                                             k_grp, k_qsel, k_qk, k_soft, k_pv, k_fin, Buffer(ws_ty, name=f"ga_ws{c}_{r}"),
                                             Buffer(wo_ty, name=f"ga_wo{c}_{r}"), Buffer(q_ty, name=f"ga_qb{c}_{r}"),
                                             p_p8, p_fn, QH * r + c % QH, r],
                                      tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        n_qs, n_kv, n_a = self.n_qs, (n_kv_heads * self.kv_bytes + self.kv_pad) // 2, n_pass * self.pq * COLS * hd * 9 // 16
        hb, ob, qw = self.kv_bytes // 2, OBJ // 2, k.qp // 2  # (bf16: a head's cache, a cache object, a Q object)

        def seq(QS, KV, A, kvh, oh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):  # pass b: token groups 4 b .. of the column's head
                oh[c].drain(A, TensorAccessPattern((n_a,), c * nd * 36, [n_pass, 4, nd, 36],
                                                   [4 * COLS * nd * 36, COLS * nd * 36, 36, 1]), group=tg, wait=True)
            inflight = []  # (a pass: one task of three BDs per stream, at most two passes in flight)
            for b in range(n_pass):
                tp = TaskGroup()
                for (h, s), f in zip(streams, kvh, strict=True):  # the pass's Q halves, the history (QA_FN groups),
                    # then the chunk's groups before the pass's: the objects' d-block halves s (of both parts)
                    parts = [ChainPart(QS, TensorAccessPattern((n_qs,), (4 * COLS * b + 4 * QH * h) * qw + s * half,
                                                               [1, 4 * QH, 2, half // 2], [0, qw, half // 2, 1])),
                             ChainPart(KV, TensorAccessPattern((n_kv,), h * hb + s * half // 2,
                                                               [1, k.kp + 1, 2, half // 2],
                                                               [0, ob, half, 1]), length_parameter=p_fn)]
                    if b:
                        parts.append(ChainPart(KV, TensorAccessPattern((n_kv,), h * hb + s * half // 2,
                                                                       [1, 2 * (k.kp + 1) * b, 2, half // 2],
                                                                       [0, ob, half, 1]),
                                               offset_parameter=p_fo))
                    chained_fill(f, KV, parts, tp, wait=True)
                inflight.append(tp)
                if len(inflight) > 1:
                    inflight.pop(0).finish()
            for tp in inflight:
                tp.finish()
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_qs,), bf], np.ndarray[(n_kv,), bf], np.ndarray[(n_a,), bf],
                           [kv3[hs].prod(tile=Tile(c, 0)) for hs, c in streams.items()],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return self._pinned(Program(NPU2(), rt, workers=workers).resolve_program())

    def _pinned(self, module: Module) -> Module:
        lay = self.l1_layout()
        pin_l1_buffers(module, fifo=lambda name, depth: next((v for kk, v in lay.items() if name.startswith(kk)), None),
                       buffer=lambda name: next((v[0] for kk, v in lay.items() if name.startswith(kk)), None))
        return module

    def l1_layout(self) -> dict[str, list[int]]:
        """Core byte addresses by 16 KB region (the stack at 0): QK streams Q against K, PV V into O. Head_dim 256 (two
        query blocks): 0 the K / V units, 1 Q and an output buffer, 2 O (16 KB), 3 the workspace and the other output
        buffer; 512: 0 and 1 stream A's and B's objects (1 also an output buffer), 2 O (16 KB), 3 Q, the workspace and
        the other output buffer."""
        k = self.kind
        u, o, q, w = k.unit, self.qb * k.nd * 72, self.qb * k.qp, -(-self.workspace()["size"] // 64) * 64
        S, R = self.STACK, 0x4000
        if k.swa:
            lay = {"ga_kv": [S, S + u], "ga_or": [R + q, 3 * R + w], "ga_qr": [R], "ga_wo": [2 * R], "ga_ws": [3 * R]}
            assert S + 2 * u <= R and R + q + o <= 2 * R and 3 * R + w + o <= 4 * R
        else:
            h = u // 2
            lay = {"ga_ka": [S, S + h], "ga_kb": [R, R + h], "ga_or": [R + 2 * h, 3 * R + q + w], "ga_wo": [2 * R],
                   "ga_qb": [3 * R], "ga_ws": [3 * R + q]}
            assert S + 2 * h <= R and R + 2 * h + o <= 2 * R and 3 * R + q + w + o <= 4 * R
        assert 64 * 4 * k.nd * self.qb <= R
        return lay


# ----------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------


def score_constant(q_norm: float, k_norm: float) -> float:
    """q_norm k_norm log2(e): the scores in log2 units (the decode: ``attn.score_constant``, as bf16 hi + lo)."""
    return q_norm * k_norm / math.log(2)
