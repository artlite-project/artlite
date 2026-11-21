"""The rotations of a PARO checkpoint's linear inputs in a prompt's prefill (``artlite.paro``, the decode's "Rotated
inputs"): a projection's input x (T tokens x K channels as the GEMMs' bfp16ebs8 panels, [T / 8][K / 8][72 B]: a norm's
x / rms, the attention's o, the activation's h) -> x' (the same panels), each 128-channel group by its rotation
(``paro.TanForm.pair_objects``: a norm's weight folded in), x' in the group's storage order (the tiles' K order).

The tokens lie in the vector lanes (u fp32 rows of 32 tokens, a row per channel): each round updates its 64 Givens
pairs (s, p) from the round's old values, no gathers (the decode's 4 slots need them: ``kmath.rotate``): u lies in the
round's pair order (pair i in rows 2 i, 2 i + 1), read in a row, each output stored where the next round reads it (II 9
per pair: ~1.1 cycles per element for 8 rounds); the partner's value as bf16 times T in bf16 on the fp32 MAC (the decode's
numerics). An item: 32 tokens of a group; round j gives core (c, r) item 32 j + 4 c + r - 2 of the group-major list
(items [group][32 tokens]), so a column's 4 cores share the round's group: its memtile broadcasts the group's object to them (once per run
of rounds of the group) and splits the column's 4 items' panels; their outputs join for the shim. Runtime sequence: (X, RO, OUT): RO the groups' objects
[group][3328 B], OUT x' as X. A norm's copies for several projections: a run per copy (the runs of one design share its
configuration)."""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from ..operators import templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.design import COLS
from ..operators.kernel_source import kernel_prefix

_SRC = templating.load("artlite/prefill/rotate.cc.jinja")

GROUP, TOK = 128, 32  # channels of a rotation group, tokens of an item
OBJ = 4608  # bytes of a group's rotation object of 8 rounds (``paro.TanForm.pair_objects``)
PANELS = TOK // 8 * GROUP // 8  # panels of an item (4 token groups x 16 channel blocks)
ROWS = 4  # cores per column (rows 2 .. 5)


class PrefillRotOperator(ArtcOperator):
    """A rotated copy of T tokens x K channels (module docstring; ``rounds``: of the tan form)."""

    STACK = 1024
    p = kernel_prefix("prot")

    def __init__(self, *, T: int, K: int, rounds: int = 8):
        G = K // GROUP
        assert K % GROUP == 0 and T % (TOK * ROWS) == 0 and (G * T // TOK) % (COLS * ROWS) == 0, (T, K)
        assert rounds % 2 == 0 and 512 * rounds + 4 * GROUP == OBJ, rounds
        self.T, self.K, self.rounds = T, K, rounds
        super().__init__()

    @property
    def n_rounds(self) -> int:
        """Rounds of the cores' program (an item per core each)."""
        return self.K // GROUP * self.T // TOK // (COLS * ROWS)

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, R=self.rounds)

    def get_arg_spec(self) -> list[ArgSpec]:
        n = self.T * self.K * 9 // 16  # (bf16 elements of the panels)
        return [ArgSpec("in", (n,), bfloat16), ArgSpec("in", (self.K // GROUP * OBJ // 2,), bfloat16),
                ArgSpec("out", (n,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        T, K, p = self.T, self.K, self.p
        G, nb = K // GROUP, T // TOK  # (groups, items per group)
        u8, f32 = np.dtype[np.uint8], np.dtype[np.float32]
        item = PANELS * 72
        x_ty, xc_ty = np.ndarray[(item,), u8], np.ndarray[(ROWS * item,), u8]
        r_ty = np.ndarray[(OBJ,), u8]
        u_ty = np.ndarray[(GROUP * TOK,), f32]
        k_rot = Kernel(f"{p}_rot", "kernel.o", [x_ty, r_ty, x_ty, u_ty, u_ty])
        n = self.n_rounds
        x3 = [ObjectFifo(xc_ty, name=f"pr_x{c}", depth=2) for c in range(COLS)]
        r3 = [ObjectFifo(r_ty, name=f"pr_r{c}", depth=2) for c in range(COLS)]
        o3 = [ObjectFifo(xc_ty, name=f"pr_o{c}", depth=2) for c in range(COLS)]
        xs = [x3[c].cons().split([item * m for m in range(ROWS)], tile=Tile(c, 1), obj_types=[x_ty] * ROWS,
                                 names=[f"pr_x{c}_{m}" for m in range(ROWS)], depths=[2] * ROWS) for c in range(COLS)]
        rs = [r3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"pr_rb{c}") for c in range(COLS)]
        os_ = [o3[c].prod().join([item * m for m in range(ROWS)], tile=Tile(c, 1), obj_types=[x_ty] * ROWS,
                                 names=[f"pr_o{c}_{m}" for m in range(ROWS)], depths=[2] * ROWS) for c in range(COLS)]

        t = max(nb // (COLS * ROWS), 1)  # (consecutive rounds of one group)

        def core(x_f, r_f, o_f, rot, ua, ub):
            for _ in forever_():
                for _ in range_(n // t):
                    r = r_f.acquire(1)
                    for _ in range_(t):
                        x, o = x_f.acquire(1), o_f.acquire(1)
                        rot(x, r, o, ua, ub)
                        o_f.release(1)
                        x_f.release(1)
                    r_f.release(1)

        workers = [Worker(core, [xs[c][m].cons(), rs[c].cons(), os_[c][m].prod(), k_rot,
                                 Buffer(u_ty, name=f"pr_ua{c}_{m}"), Buffer(u_ty, name=f"pr_ub{c}_{m}")],
                          tile=Tile(c, 2 + m), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for m in range(ROWS)]
        # (bf16 elements: the panels, a token group's row of them, an item's token group of its 16 channel blocks; the
        # rotation objects)
        nx, kp, w, no = T * K * 9 // 16, K // 8 * 36, 16 * 36, G * OBJ // 2

        def taps(c: int) -> tuple[TensorAccessPattern, TensorAccessPattern]:
            """Column c's x (and output) walk and its rotation objects' over the rounds: round j, items 32 j + 4 c ..
            (16 token groups of a group's 16 channel blocks)."""
            if nb >= COLS * ROWS:  # (a group's items over t = nb / 32 rounds: the columns' blocks of 4 items in each)
                x = TensorAccessPattern((nx,), 16 * c * kp, [G, t, 16, w], [w, 128 * kp, kp, 1])
                r = TensorAccessPattern((no,), 0, [1, 1, G, OBJ // 2], [0, 0, OBJ // 2, 1])
            else:  # (a round's items over 32 / nb groups)
                s = COLS * ROWS // nb
                x = TensorAccessPattern((nx,), 16 * c % (4 * nb) * kp + 4 * c // nb * w, [1, n, 16, w], [0, s * w, kp, 1])
                r = TensorAccessPattern((no,), 4 * c // nb * OBJ // 2, [1, 1, n, OBJ // 2], [0, 0, s * OBJ // 2, 1])
            return x, r

        def seq(X, RO, OUT, xh, rh, oh):
            tg = TaskGroup()
            for c in range(COLS):
                x, r = taps(c)
                xh[c].fill(X, x, group=tg)
                rh[c].fill(RO, r, group=tg)
                oh[c].drain(OUT, x, group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(nx,), np.dtype[bfloat16]], np.ndarray[(no,), np.dtype[bfloat16]],
                           np.ndarray[(nx,), np.dtype[bfloat16]],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(x3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(r3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
