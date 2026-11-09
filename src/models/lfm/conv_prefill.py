"""LFM3's short convolution in the prefill: y = C * conv(B * x) of a T-token chunk, between the in GEMM and the out GEMM.

``ConvPrefillOperator`` (one core per column, (c, 2): the column's 256 channels ``conv.CH``, its tokens in order: the
window runs across them): G the in GEMM's output (T rows of ``COLS`` slices of ``pitch`` columns: column c's [B 256 |
C 256 | x 256] at c pitch, ``prefill.gemm`` layout of the in phase), H the layer's conv regions of the decode overlay
(``lfm.conv``: per column its taps and its history), A y as the out GEMM's input: bfp16 panels [T / 8][2048 / 8][72 B]
(column c's channels in panel columns 32 c ..).

Per column: the region's payloads 0 .. 2 first (taps, history rows 0 .. 5: the 6 rows before the chunk, rows the host
moved to the region's base: ``LfmPrefill``), then per group of 8 tokens: bx = bf16(B x) into a window of 14 rows
[6 rows before | the group's 8], y_t = C_t sum_k (hi_k + lo_k) window[t + k] (fp32, the decode's order: k up, hi then lo;
times C as the sum's bf16 hi + lo) as single bfp16 panels; at the chunk's ``P_CN``-th token (its valid ones: a runtime
parameter) the window's last 6 rows of valid tokens are kept, and go out at the end over the region's history rows 0 ..
5, so the decode continues there with shift 0 (``LfmOverlay.conv_at``). Memory-bound: ~1.8 KB of G per token and column.
"""

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

from artlite.decode.design import XB, XPAY
from artlite.operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from artlite.operators.design import COLS
from artlite.operators.kernel_source import kernel_prefix

from . import conv as CV

P_CN = "lc_n"  # the chunk's valid tokens (its last 6 rows of bx: the history the decode continues from)
ROW = 3 * CV.CH  # a column's G row: [B | C | x]

_SRC = JINJA_ENV.from_string(r"""
extern "C" {

// The region's payloads 0 .. 2 (12 KB: taps 0 .. 3, taps 4 .. 6 and history rows 8, 9, history rows 0 .. 7; held for
// the chunk: its taps): the window's rows 0 .. 5
void {{ p }}_cinit(const bfloat16 *__restrict r, bfloat16 *__restrict win, int32_t *__restrict st, int32_t n)
{
    st[0] = 0;  // (the next group's first token)
    st[1] = n;
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ 6 * CH }}; i += 64)
        aie::store_v(win + i, aie::load_v<64>(r + {{ 2 * XPAY }} + i));
}

// 8 tokens t0 .. t0 + 7 of G (rows [B | C | x]) -> 32 panels of y (8 tokens x 8 channels); the token n - 1 among them:
// the 6 rows of bx up to it into hk (the history's output object, held for the chunk)
void {{ p }}_cgrp(const bfloat16 *__restrict g, const bfloat16 *__restrict taps, bfloat16 *__restrict win,
                  float *__restrict yt, bfloat16 *__restrict hk, bfloat16 *__restrict out, int32_t *__restrict st)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    const int t0 = st[0], n = st[1];
    st[0] = t0 + 8;
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 8; t++) {
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ CH // 64 }}; v++) {
            const aie::vector<bfloat16, 64> b = aie::load_v<64>(g + {{ ROW }} * t + 64 * v);
            const aie::vector<bfloat16, 64> x = aie::load_v<64>(g + {{ ROW }} * t + {{ 2 * CH }} + 64 * v);
            aie::store_v(win + {{ CH }} * (6 + t) + 64 * v, aie::mul(b, x).template to_vector<bfloat16>());
        }
    }
    AIE_LOOP_NO_UNROLL
    for (int t = 0; t < 8; t++) {
        AIE_LOOP_NO_UNROLL
        for (int v = 0; v < {{ CH // 64 }}; v++) {
            aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
            AIE_LOOP_NO_UNROLL
            for (int k = 0; k < 7; k++) {
                const aie::vector<bfloat16, 64> r = aie::load_v<64>(win + {{ CH }} * (t + k) + 64 * v);
                acc = aie::mac(acc, aie::load_v<64>(taps + {{ 2 * CH }} * k + 64 * v), r);
                acc = aie::mac(acc, aie::load_v<64>(taps + {{ 2 * CH }} * k + {{ CH }} + 64 * v), r);
            }
            const aie::vector<bfloat16, 64> ah = acc.template to_vector<bfloat16>();
            const aie::vector<bfloat16, 64> al = aie::accum<accfloat, 64>(
                aie::sub(acc.template to_vector<float>(), aie::accum<accfloat, 64>(ah).template to_vector<float>())).template to_vector<bfloat16>();
            const aie::vector<bfloat16, 64> c = aie::load_v<64>(g + {{ ROW }} * t + {{ CH }} + 64 * v);
            aie::store_v(yt + {{ CH }} * t + 64 * v, aie::mac(aie::mul(ah, c), al, c).template to_vector<float>());
        }
    }
    {
        al::BOut<> o((bfp16ebs8 *)out);
        AIE_LOOP_NO_UNROLL
        for (int j = 0; j < {{ CH // 8 }}; j++)
            o.push(aie::accum<accfloat, 64>(al::gather<8, 8>(yt + 8 * j, {{ CH }})).template to_vector<bfp16ebs8>());
    }
    if (n - 1 >= t0 && n - 1 < t0 + 8) {  // (window rows n - 6 - t0 + 6 .. : the 6 tokens up to n - 1)
        const bfloat16 *__restrict src = win + {{ CH }} * (n - t0);
        AIE_LOOP_NO_UNROLL
        for (int i = 0; i < {{ 6 * CH }}; i += 64)
            aie::store_v(hk + i, aie::load_v<64>(src + i));
    }
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < {{ 6 * CH }}; i += 64)  // (the next group's rows before it)
        aie::store_v(win + i, aie::load_v<64>(win + {{ 8 * CH }} + i));
}

}  // extern "C"
""")


class ConvPrefillOperator(ArtcOperator):
    """y = C conv(B x) of T tokens (module docstring). Runtime sequence: G (T x COLS pitch bf16), H (the layer's conv
    regions: COLS ``conv.region_bytes`` bytes), A (panels of T x 2048)."""

    STACK = 1024
    p = kernel_prefix("lfcv")

    def __init__(self, *, T: int, pitch: int, hidden: int = COLS * CV.CH):
        assert T % 8 == 0 and pitch >= ROW and hidden == COLS * CV.CH, (T, pitch, hidden)
        self.T, self.pitch, self.hidden = T, pitch, hidden
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, CH=CV.CH, ROW=ROW, XPAY=XPAY // 2)

    @property
    def n_g(self) -> int:
        return self.T * COLS * self.pitch

    @property
    def n_h(self) -> int:
        return COLS * CV.region_bytes() // 2

    @property
    def n_a(self) -> int:
        return self.T * self.hidden * 9 // 16

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.n_g,), torch.bfloat16), ArgSpec("inout", (self.n_h,), torch.bfloat16),
                ArgSpec("out", (self.n_a,), torch.bfloat16)]

    @staticmethod
    def params(n: int) -> dict[str, int]:
        return {P_CN: n}

    def get_mlir_artifact(self) -> Module:
        T, p, pitch, CH = self.T, self.p, self.pitch, CV.CH
        bf = np.dtype[bfloat16]
        ng = T // 8
        r_ty = np.ndarray[(3 * XPAY // 2,), bf]
        g_ty = np.ndarray[(8 * ROW,), bf]
        o_ty = np.ndarray[(CH * 8 * 9 // 16,), bf]
        h_ty = np.ndarray[(6 * CH,), bf]
        st_ty = np.ndarray[(16,), np.dtype[np.int32]]
        k_init = Kernel(f"{p}_cinit", "kernel.o", [r_ty, np.ndarray[(14 * CH,), bf], st_ty, np.int32])
        k_grp = Kernel(f"{p}_cgrp", "kernel.o", [g_ty, r_ty, np.ndarray[(14 * CH,), bf],
                                                np.ndarray[(8 * CH,), np.dtype[np.float32]], h_ty, o_ty, st_ty])
        p_n = ScratchpadParameter(P_CN, np.int32)
        rf = [ObjectFifo(r_ty, name=f"lc_r{c}", depth=1) for c in range(COLS)]
        gf = [ObjectFifo(g_ty, name=f"lc_g{c}", depth=2) for c in range(COLS)]
        of = [ObjectFifo(o_ty, name=f"lc_o{c}", depth=2) for c in range(COLS)]
        hf = [ObjectFifo(h_ty, name=f"lc_h{c}", depth=1) for c in range(COLS)]

        def core(r_f, g_f, o_f, h_f, cinit, cgrp, win, yt, st, pn):
            for _ in forever_():
                r = r_f.acquire(1)
                h = h_f.acquire(1)
                cinit(r, win, st, pn.read())
                for _ in range_(ng):
                    g = g_f.acquire(1)
                    o = o_f.acquire(1)
                    cgrp(g, r, win, yt, h, o, st)
                    o_f.release(1)
                    g_f.release(1)
                h_f.release(1)
                r_f.release(1)

        workers = [Worker(core, [rf[c].cons(), gf[c].cons(), of[c].prod(), hf[c].prod(), k_init, k_grp,
                                 Buffer(np.ndarray[(14 * CH,), bf], name=f"lc_win{c}"),
                                 Buffer(np.ndarray[(8 * CH,), np.dtype[np.float32]], name=f"lc_yt{c}"),
                                 Buffer(st_ty, name=f"lc_st{c}", initial_value=np.zeros(16, np.int32)), p_n],
                          tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS)]
        n_g, n_h, n_a = self.n_g, self.n_h, self.n_a
        rb, xb = CV.region_bytes() // 2, XB // 2
        ga = 8 * COLS * pitch  # (a token group's G elements)
        pa = 8 * self.hidden * 9 // 16  # (a token group's panels)

        def seq(G, H, A, rh, gh, oh, hh):
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                rh[c].fill(H, TensorAccessPattern((n_h,), rb * c + CV.payload(0) // 2, [1, 1, 3, XPAY // 2],
                                                  [0, 0, xb, 1]), group=tg)
                gh[c].fill(G, TensorAccessPattern((n_g,), pitch * c, [1, ng, 8, ROW], [0, ga, COLS * pitch, 1]),
                           group=tg)
                oh[c].drain(A, TensorAccessPattern((n_a,), CH * 8 * 9 // 16 * c, [1, 1, ng, CH * 8 * 9 // 16],
                                                   [0, 0, pa, 1]), group=tg, wait=True)
                hh[c].drain(H, TensorAccessPattern((n_h,), rb * c + CV.payload(2) // 2, [1, 1, 1, 6 * CH], [0, 0, 0, 1]),
                            group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_g,), bf], np.ndarray[(n_h,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(rf)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(gf)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(hf)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
