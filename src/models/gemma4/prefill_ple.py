"""The per-layer input of the Gemma prefill: h = gelu(bf16(y)) pli between the per-layer input gate (``inp_gate``) and
the per-layer projection (``proj``), with the per-layer inputs pli = bf16(p / rms(p) w + e) formed here, as the decode's
F_PLI epilogue forms them (``decode.design.F_PLI``).

``PleOperator`` runs over T tokens: y the gate GEMM's output (T rows of 8 column slots of ``SLOT`` values: column c's
32 outputs at 64 c, ``prefill.gemm`` layout of the gate phase); p the layer's rows of the per-layer projection of the
embeddings (the prefill's "ple" GEMM, bf16: per token column c's 32 values at c ``pitch``, the argument starting at the
layer's); e the host's per-layer token rows of the layer (T x 256 bf16, ``GemmaHost.layer_rows``); w the projection
norm's weights / sqrt 2 (bf16 hi [256] | lo [256]); h as proj's A: bfp16 panels [T / 8][512 / 8][72 B] in the same
slots (column c's 32 values at 64 c, zeros after them: the image's proj tiles hold zero weights there, ``gemma4.image``).
1 / rms(p) over a token's 256 values with eps = hidden_size rms_norm_eps (``prefill`` folds the projection's 1 / sqrt(hidden) into
the norm), GELU through kmath's table pass (a worker buffer of the core: no static tables), bf16 like the decode's:
bf16(bf16(gelu(bf16(y))) pli).

``cols`` cores (one per column, at most 4) take the token groups of 8 in turn; a core's p, e and w arrive as one
object (its memtile joins the three: a core has two input channels), y and p from its column's shim, e and w from the
shim 4 columns over (a shim has two output channels).
"""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.operators import kmath, templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.design import COLS
from artlite.operators.kernel_source import kernel_prefix

SLOT = 64  # the gate outputs' and proj inputs' column slots
PER = 32  # values per slot
D = COLS * PER  # a token's per-layer inputs

_SRC = templating.load("models/gemma4/prefill_ple.cc.jinja")


class PleOperator(ArtcOperator):
    """gelu(y) pli of T tokens (module docstring). Runtime sequence: Y (T x 8 SLOT bf16), P (T rows of 8 ``pitch`` bf16:
    column c's 32 p values at c pitch of a row), E (T x 8 PER bf16), W (2 x 8 PER bf16), A (panels of T x 8 SLOT)."""

    STACK = 1024
    p = kernel_prefix("g4ple")

    def __init__(self, *, T: int, pitch: int, eps: float, cols: int = COLS // 2):
        assert T % 8 == 0 and T >= 8 * cols and 1 <= cols <= COLS // 2 and pitch % 32 == 0
        self.T, self.pitch, self.eps, self.cols = T, pitch, eps, cols
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, SLOT=SLOT, PER=PER, D=D, eps=repr(self.eps))

    def get_arg_spec(self) -> list[ArgSpec]:
        T = self.T
        return [ArgSpec("in", (T * COLS * SLOT,), bfloat16), ArgSpec("in", (T * COLS * self.pitch,), bfloat16),
                ArgSpec("in", (T * D,), bfloat16), ArgSpec("in", (2 * D,), bfloat16),
                ArgSpec("out", (T * COLS * SLOT * 9 // 16,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        T, p, cols, pitch = self.T, self.p, self.cols, self.pitch
        n_gs = [len(range(c, T // 8, cols)) for c in range(cols)]  # (core c: token groups c, c + cols, ..)
        bf = np.dtype[bfloat16]
        y_ty = np.ndarray[(8 * COLS * SLOT,), bf]
        l_ty = np.ndarray[(8 * D,), bf]
        w_ty = np.ndarray[(2 * D,), bf]
        j_ty = np.ndarray[(18 * D,), bf]  # (p | e | w)
        s_ty = np.ndarray[(64,), np.dtype[np.float32]]
        o_ty = np.ndarray[(8 * COLS * SLOT * 9 // 16,), bf]
        tab = kmath.GELU.table_words()  # (its two copies in banks 2 and 3: kmath.table_pass)
        tab_ty = np.ndarray[(tab.size,), np.dtype[np.uint32]]
        k_ple = Kernel(f"{p}_ple", "kernel.o", [y_ty, j_ty, l_ty, l_ty, s_ty, tab_ty, tab_ty, o_ty])
        yf = [ObjectFifo(y_ty, name=f"pl_y{c}", depth=2) for c in range(cols)]
        jf = [ObjectFifo(j_ty, name=f"pl_j{c}", depth=2) for c in range(cols)]
        sub = [jf[c].prod().join([0, 8 * D, 16 * D], tile=Tile(c, 1), depths=[2] * 3, obj_types=[l_ty, l_ty, w_ty],
                                 names=[f"pl_{x}{c}" for x in "pew"]) for c in range(cols)]
        of = [ObjectFifo(o_ty, name=f"pl_o{c}", depth=2) for c in range(cols)]

        def core(y_f, j_f, o_f, ple, t, pl, ss, ta, tb, n_g):
            for _ in forever_():
                for _ in range_(n_g):
                    y = y_f.acquire(1)
                    j = j_f.acquire(1)
                    o = o_f.acquire(1)
                    ple(y, j, t, pl, ss, ta, tb, o)
                    o_f.release(1)
                    j_f.release(1)
                    y_f.release(1)

        workers = [Worker(core, [yf[c].cons(), jf[c].cons(), of[c].prod(), k_ple,
                                 Buffer(l_ty, name=f"pl_t{c}"), Buffer(l_ty, name=f"pl_l{c}"),
                                 Buffer(s_ty, name=f"pl_s{c}"),
                                 Buffer(tab_ty, name=f"pl_tab{c}", initial_value=tab, mem_bank=2),
                                 Buffer(tab_ty, name=f"pl_tabb{c}", initial_value=tab, mem_bank=3), n_gs[c]],
                          tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(cols)]
        ny, npp, ne, no = T * COLS * SLOT, T * COLS * pitch, T * D, T * COLS * SLOT * 9 // 16
        g_y, g_e, g_o = 8 * COLS * SLOT, 8 * D, 8 * COLS * SLOT * 9 // 16
        row = COLS * pitch  # (a token's row of P)

        def seq(Y, P, E, W, A, yh, ph, eh, wh, oh):  # (p, e, w: each group's object of the join)
            tg = TaskGroup()
            for c in range(cols):  # (the core's groups c, c + cols, ..)
                n_g = n_gs[c]
                yh[c].fill(Y, TensorAccessPattern((ny,), g_y * c, [1, n_g, g_y // 512, 512], [0, cols * g_y, 512, 1]),
                           group=tg)
                ph[c].fill(P, TensorAccessPattern((npp,), 8 * row * c, [n_g, 8, COLS, PER],
                                                  [8 * cols * row, row, pitch, 1]), group=tg)
                eh[c].fill(E, TensorAccessPattern((ne,), g_e * c, [1, n_g, g_e // 512, 512], [0, cols * g_e, 512, 1]),
                           group=tg)
                wh[c].fill(W, TensorAccessPattern((2 * D,), 0, [1, n_g, 1, 2 * D], [0, 0, 0, 1]), group=tg)
                oh[c].drain(A, TensorAccessPattern((no,), g_o * c, [1, n_g, g_o // 576, 576], [0, cols * g_o, 576, 1]),
                            group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(ny,), bf], np.ndarray[(npp,), bf], np.ndarray[(ne,), bf],
                           np.ndarray[(2 * D,), bf], np.ndarray[(no,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(yf)],
                           [s[0].prod(tile=Tile(c, 0)) for c, s in enumerate(sub)],
                           [s[1].prod(tile=Tile(c + COLS // 2, 0)) for c, s in enumerate(sub)],
                           [s[2].prod(tile=Tile(c + COLS // 2, 0)) for c, s in enumerate(sub)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
