"""The activation of a split gate / up block's prefill (``decode.image.Block.fmt2``: gate and up in two tile formats):
their GEMMs write plain bf16 rows into GU (``prefill.gemm``'s split plan), this operator turns them into the down
projection's A, h = bf16(act(gate) up) as bfp16 panels, the gate / up epilogue's arithmetic (``gemm_q``'s times_up,
kmath.silu_tanh, or a GELU form through its table), so a split layer's h is what one GEMM of both parts would give.

GU: T rows of ``COLS`` column slices of ``groups`` x [gate W | up W] (each part's first 64 values its group's outputs, the
rest the passes' padding rows). HP: the panels [T / 8][K / 8][72 B] (K: the down's), column c's outputs at their place
(c groups 64 ..), columns past the groups' never written. One core per column takes its 8-token blocks' objects of ``G``
groups (a 4-dimensional shim read: [token block][group chunk][token 8][2 G W]), each into G x 8 panels (a 3-dimensional
drain)."""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ..operators import kmath, templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS, MAX_BD_ITER
from ..operators.kernel_source import kernel_prefix

_SRC = templating.load("artlite/prefill/gate_up.cc.jinja")

PART = 64  # outputs of a SiLU group (each part's valid values)


def chunk_groups(groups: int) -> int:
    """Groups per object (G): the most of 4, 2, 1 that divides the column's groups."""
    return next(g for g in (4, 2, 1) if groups % g == 0)


class GateUpActOperator(ArtcOperator):
    """h = bf16(act(gate) up) of T tokens (module docstring; ``act``: silu, or a GELU form of ``kmath.TABLES``). Runtime
    sequence: GU (T x COLS ``groups`` 2 W bf16), HP (panels of T x K)."""

    STACK = 1024
    p = kernel_prefix("guact")

    def __init__(self, *, T: int, groups: int, W: int, K: int, act: str = "silu"):
        # (a shim channel's tasks: pieces of MAX_BD_ITER token blocks, at most 4 queued)
        assert T % 8 == 0 and T // 8 <= 4 * MAX_BD_ITER and W % 8 == 0 and W >= PART and K >= COLS * groups * PART, (T, W, K)
        assert act == "silu" or act in kmath.TABLES, act
        self.T, self.groups, self.W, self.K, self.act = T, groups, W, K, act
        self.G = chunk_groups(groups)
        super().__init__()

    @property
    def N(self) -> int:
        """GU's row (bf16 elements)."""
        return COLS * self.groups * 2 * self.W

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, G=self.G, W=self.W, act=self.act)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.T * self.N,), bfloat16), ArgSpec("out", (self.T * self.K * 9 // 16,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        T, G, W, N, K, p = self.T, self.G, self.W, self.N, self.K, self.p
        nb, nc = T // 8, self.groups // G  # (token blocks, group chunks)
        bf = np.dtype[bfloat16]
        x_ty = np.ndarray[(8 * 2 * G * W,), bf]
        s_ty = np.ndarray[(8 * G * 128,), bf]
        y_ty = np.ndarray[(8 * G * 64,), bf]
        o_ty = np.ndarray[(8 * G * 36,), bf]  # (8 G panels of 72 B)
        gelu = self.act != "silu"  # (the table's two copies, cold: aiecc places them)
        lut_ty = np.ndarray[(kmath.TABLES[self.act].table_words().size,), np.dtype[np.uint32]]
        k_act = Kernel(f"{p}_act", "kernel.o", [x_ty, s_ty, y_ty, o_ty, *([lut_ty] * 2 if gelu else [np.int32])])
        xf = [ObjectFifo(x_ty, name=f"ga_x{c}", depth=2) for c in range(COLS)]
        of = [ObjectFifo(o_ty, name=f"ga_o{c}", depth=2) for c in range(COLS)]

        def core(x_f, o_f, act, s, y, *luts):
            for _ in forever_():
                for _ in range_(nb * nc):
                    x = x_f.acquire(1)
                    o = o_f.acquire(1)
                    act(x, s, y, o, *(luts or (0,)))  # (silu: the kernel's opaque runtime zero)
                    o_f.release(1)
                    x_f.release(1)

        luts = [[Buffer(lut_ty, name=f"ga_{key}{c}", initial_value=kmath.TABLES[self.act].table_words())
                 for key in ("lut", "lutb")] if gelu else [] for c in range(COLS)]
        workers = [Worker(core, [xf[c].cons(), of[c].prod(), k_act, Buffer(s_ty, name=f"ga_s{c}"),
                                 Buffer(y_ty, name=f"ga_y{c}"), *luts[c]],
                          tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS)]
        P, nh = self.groups * 2 * W, T * K * 9 // 16
        pk = K // 8 * 36  # (a token block's panels: bf16 elements)

        def seq(GU, HP, xh, oh):
            tg = TaskGroup()
            for b0 in range(0, nb, MAX_BD_ITER):  # (pieces of token blocks: a BD iterates at most MAX_BD_ITER times)
                n = min(MAX_BD_ITER, nb - b0)
                for c in range(COLS):
                    xh[c].fill(GU, TensorAccessPattern((T * N,), 8 * b0 * N + c * P, [n, nc, 8, 2 * G * W],
                                                       [8 * N, 2 * G * W, N, 1]), group=tg)
                    oh[c].drain(HP, TensorAccessPattern((nh,), b0 * pk + c * self.groups * 8 * 36, [1, n, nc, G * 8 * 36],
                                                        [0, pk, G * 8 * 36, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(T * N,), bf], np.ndarray[(nh,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(xf)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
