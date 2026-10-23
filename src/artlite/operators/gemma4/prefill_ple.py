"""The per-layer input of the Gemma prefill: h = gelu(bf16(y)) pli between the per-layer input gate (``inp_gate``) and
the per-layer projection (``proj``).

``PleOperator`` runs over T tokens: y the gate GEMM's output (T rows of 8 column slots of ``SLOT`` values: column c's
32 outputs at 64 c, ``prefill.gemm`` layout of the gate phase), pli the layer's per-layer inputs (T x 256 bf16, the
host's), h as proj's A: bfp16 panels [T / 8][512 / 8][72 B] in the same slots (column c's 32 values at 64 c, zeros
after them: the image's proj tiles hold zero weights there, ``gemma4.image``). GELU through kmath's table pass (a worker
buffer of the core: no static tables), bf16 like the decode's F_PLI epilogue: bf16(bf16(gelu(bf16(y))) pli).

``cols`` cores (one per column) take the token groups of 8 in turn.
"""

import numpy as np
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from .. import kmath
from ..base import JINJA_ENV, ArgSpec, ArtcOperator
from ..design import COLS
from ..kernel_source import kernel_prefix

SLOT = 64  # the gate outputs' and proj inputs' column slots
PER = 32  # values per slot

_SRC = JINJA_ENV.from_string(r"""
{% set gact = km.act(km.GELU, table_arg=True, walk=(1, SLOT), inline=True) %}
{{ gact.src }}

extern "C" {

// 8 tokens: y (8 x {{ 8 * SLOT }} bf16), pli (8 x {{ 8 * PER }} bf16) -> 64 panels (8 tokens x 8 columns) of h in the slots
void {{ p }}_ple(const bfloat16 *__restrict y, const bfloat16 *__restrict pli, bfloat16 *__restrict t,
                 const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd, bfloat16 *__restrict out)
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    // gelu of the slots' values (y's slots, {{ SLOT }} apart) into t, contiguous
    {{ gact }}<{{ 8 * 8 * PER }}>(y, t, tab_ab, tab_cd);
    {  // h = bf16(gelu) pli -> panels: per slot 4 panels of values, 4 of zeros
        const bfloat16 *__restrict pg = t;
        al::BOut<> o((bfp16ebs8 *)out);
        for (int s = 0; s < 8; s++) {
            for (int q = 0; q < 4; q++) {
                aie::vector<bfloat16, 64> g, l;
{% for r in range(8) %}
                g.insert({{ r }}, aie::load_v<8>(pg + {{ 8 * PER * r }} + {{ PER }} * s + 8 * q));
                l.insert({{ r }}, aie::load_v<8>(pli + {{ 8 * PER * r }} + {{ PER }} * s + 8 * q));
{% endfor %}
                o.push(aie::mul(g, l).template to_vector<bfp16ebs8>());
            }
            for (int q = 0; q < 4; q++)
                o.push(aie::zeros<accfloat, 64>().template to_vector<bfp16ebs8>());
        }
    }
}

}  // extern "C"
""")


class PleOperator(ArtcOperator):
    """gelu(y) pli of T tokens (module docstring). Runtime sequence: Y (T x 8 SLOT bf16), PLI (T x 8 PER bf16), A
    (panels of T x 8 SLOT)."""

    STACK = 1024
    p = kernel_prefix("g4ple")

    def __init__(self, *, T: int, cols: int = COLS):
        assert T % 8 == 0 and T >= 8 * cols and 1 <= cols <= COLS
        self.T, self.cols = T, cols
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, SLOT=SLOT, PER=PER)

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (T * COLS * SLOT,), torch.bfloat16), ArgSpec("in", (T * COLS * PER,), torch.bfloat16),
                ArgSpec("out", (T * COLS * SLOT * 9 // 16,), torch.bfloat16)]

    def get_mlir_artifact(self) -> Module:
        T, p, cols = self.T, self.p, self.cols
        n_gs = [len(range(c, T // 8, cols)) for c in range(cols)]  # (core c: token groups c, c + cols, ..)
        bf = np.dtype[bfloat16]
        y_ty = np.ndarray[(8 * COLS * SLOT,), bf]
        l_ty = np.ndarray[(8 * COLS * PER,), bf]
        t_ty = np.ndarray[(8 * COLS * PER,), bf]
        o_ty = np.ndarray[(8 * COLS * SLOT * 9 // 16,), bf]
        tab = kmath.GELU.table_words()  # (its two copies in banks 2 and 3: kmath.table_pass)
        tab_ty = np.ndarray[(tab.size,), np.dtype[np.uint32]]
        k_ple = Kernel(f"{p}_ple", "kernel.o", [y_ty, l_ty, t_ty, tab_ty, tab_ty, o_ty])
        yf = [ObjectFifo(y_ty, name=f"pl_y{c}", depth=2) for c in range(cols)]
        lf = [ObjectFifo(l_ty, name=f"pl_l{c}", depth=2) for c in range(cols)]
        of = [ObjectFifo(o_ty, name=f"pl_o{c}", depth=2) for c in range(cols)]

        def core(y_f, l_f, o_f, ple, t, ta, tb, n_g):
            for _ in forever_():
                for _ in range_(n_g):
                    y = y_f.acquire(1)
                    pli = l_f.acquire(1)
                    o = o_f.acquire(1)
                    ple(y, pli, t, ta, tb, o)
                    o_f.release(1)
                    l_f.release(1)
                    y_f.release(1)

        workers = [Worker(core, [yf[c].cons(), lf[c].cons(), of[c].prod(), k_ple, Buffer(t_ty, name=f"pl_t{c}"),
                                 Buffer(tab_ty, name=f"pl_tab{c}", initial_value=tab, mem_bank=2),
                                 Buffer(tab_ty, name=f"pl_tabb{c}", initial_value=tab, mem_bank=3), n_gs[c]],
                          tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(cols)]
        ny, nl, no = T * COLS * SLOT, T * COLS * PER, T * COLS * SLOT * 9 // 16
        g_y, g_l, g_o = 8 * COLS * SLOT, 8 * COLS * PER, 8 * COLS * SLOT * 9 // 16

        def seq(Y, PLI, A, yh, lh, oh):
            tg = TaskGroup()
            for c in range(cols):  # (the core's groups c, c + cols, ..)
                n_g = n_gs[c]
                yh[c].fill(Y, TensorAccessPattern((ny,), g_y * c, [1, n_g, g_y // 512, 512], [0, cols * g_y, 512, 1]),
                           group=tg)
                lh[c].fill(PLI, TensorAccessPattern((nl,), g_l * c, [1, n_g, g_l // 512, 512], [0, cols * g_l, 512, 1]),
                           group=tg)
                oh[c].drain(A, TensorAccessPattern((no,), g_o * c, [1, n_g, g_o // 576, 576], [0, cols * g_o, 576, 1]),
                            group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(ny,), bf], np.ndarray[(nl,), bf], np.ndarray[(no,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(yf)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(lf)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
