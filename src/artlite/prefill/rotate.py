"""The rotations of a PARO checkpoint's linear inputs in a prompt's prefill (``artlite.paro``, the decode's "Rotated
inputs"): a projection's input x (T tokens x K channels as the GEMMs' bfp16ebs8 panels, [T / 8][K / 8][72 B]: a norm's
x / rms, the attention's o) -> x' (the same panels), each 128-channel group by its rotation
(``paro.TanForm.pair_objects``: a norm's weight folded in), x' in the group's storage order (the tiles' K order). The
down projection's: ``gu`` = the activation of a split gate / up block on the way (``prefill.gemm.plan_gate_up_pair``:
GU = gate and up as bf16 panels, [T / 8][K / 8][gate 64 | up 64]), x = bf16(silu(gate) up), the gate / up epilogue's
arithmetic, rotated before any rounding to bfp16.

The tokens lie in the vector lanes (u fp32 rows of ``tok`` tokens, a row per channel): each round updates its 64 Givens
pairs (s, p) from the round's old values, no gathers (the decode's 4 slots need them: ``kmath.rotate``): u lies in the
round's pair order (pair i in rows 2 i, 2 i + 1), read in a row, each output stored where the next round reads it (II 9
per pair of 32-token rows: ~1.1 cycles per element for 8 rounds); the partner's value as bf16 times T in bf16 on the fp32
MAC (the decode's numerics). An item: ``tok`` tokens of a group (32; ``gu``: 16, its 8 KB of gate / up panels fit L1
next to the state); the items in group-major order, each group's split evenly over the 8 columns, a column's run of
token groups contiguous (four shim dimensions: [group][token group][run]): column c takes the group's items 4 t c ..
4 t (c + 1) in t rounds of its 4 cores (core r: item 4 t c + 4 i + r - 2 in round i; fewer than 32 items a group: a
round's 32 items span 32 / items groups), so a column's 4 cores share the round's group: its memtile broadcasts the
group's object to them (once per run of rounds of the group) and splits the column's 4 items; their outputs join for
the shim. Where every core works on one group at a time (32 items a group or more), one shim broadcasts the objects to
all 32 cores instead (the columns' reads of them: 12% of the pass's bytes at 1024 tokens). Runtime sequence: (X, RO,
OUT): RO the groups' objects [group][``OBJ``], OUT x' as panels. A norm's ``copies`` for several projections (q, k, v;
gate, up): x read once, each item rotated per copy (RO [copy][group][OBJ], OUT [token group][copy][K / 8 panels]: a
copy's panels one row apart, whatever the prompt's size: a GEMM reads its copy as K columns of panels ``copies`` K
wide). ``rows_out``: x' as bf16 rows [T][K] (an MoE layer's copy that its experts gather, ``moe.gather``). ``rows`` (``gu`` of gate and up GEMMs that write rows, ``prefill.gemm.plan_gate_up_rows``: tiles whose passes
are not 64 rows, P5_128's 13): GU as bf16 rows [token][column block][gate | up] (a block's gate channels from its start,
its up ones ``rows[1]`` later, ``rows[0]`` a block's width): an item's L1 object [token][gate 128 | up 128], each round's
[the column's tokens][gate | up][128] a fill whose repeat is the round, a fill per column block (the groups of a block
are linear in it, not across blocks: a block's padding past its channels)."""

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

GROUP = 128  # channels of a rotation group
OBJ = 4608  # bytes of a group's rotation object of 8 rounds (``paro.TanForm.pair_objects``)
ROWS = 4  # cores per column (rows 2 .. 5)


def row_floats(gu: bool) -> int:
    """Tokens of an item (u's row: ``paro.TanForm.pair_objects``'s ``row``)."""
    return 16 if gu else 32


class PrefillRotOperator(ArtcOperator):
    """A rotated copy of T tokens x K channels (module docstring; ``rounds``: of the tan form; ``gu``: of a split gate /
    up block's GU, its activation on the way; ``bf16``: x as bf16 blocks, a norm's ``bf16`` output)."""

    STACK = 1024
    p = kernel_prefix("prot")

    def __init__(self, *, T: int, K: int, rounds: int = 8, gu: bool = False, copies: int = 1, bf16: bool = False,
                 rows: tuple[int, int] | None = None, rows_out: bool = False):
        """rows_out: x' as bf16 rows [T][K] instead of panels (an MoE layer's copy that its experts gather)."""
        G = K // GROUP
        self.tok = row_floats(gu)
        assert K % GROUP == 0 and T % (self.tok * ROWS) == 0 and (G * T // self.tok) % (COLS * ROWS) == 0, (T, K)
        assert rounds % 2 == 0 and 512 * rounds + 4 * GROUP == OBJ, rounds
        assert copies == 1 or (not gu and T // self.tok <= COLS * ROWS), "copies: a round per group"
        assert not (gu and bf16) and (rows is None or (gu and K % COLS == 0 and K // COLS % GROUP == 0
                                                      and rows[0] >= 2 * K // COLS and rows[1] >= K // COLS))
        assert not rows_out or (copies == 1 and not gu), "rows_out: one copy of a norm's x"
        self.T, self.K, self.rounds, self.gu, self.copies, self.bf16, self.rows = T, K, rounds, gu, copies, bf16, rows
        self.rows_out = rows_out
        super().__init__()

    @property
    def n_rounds(self) -> int:
        """Rounds of the cores' program (an item per core each)."""
        return self.K // GROUP * self.T // self.tok // (COLS * ROWS)

    @property
    def unit(self) -> int:
        """bf16 elements of an input panel ([gate 64 | up 64] bf16, a bf16 block, or a bfp16 panel's 72 bytes)."""
        return 128 if self.gu else 64 if self.bf16 else 36

    def get_kernel_src(self) -> str:
        return _SRC.render(p=self.p, R=self.rounds, TOK=self.tok, gu=self.gu, xb=self.bf16, gurows=self.rows is not None,
                           rows_out=self.rows_out)

    def get_arg_spec(self) -> list[ArgSpec]:
        n = self.T * self.K if self.rows_out else self.T * self.K * 9 // 16  # (bf16 elements of the rows or panels)
        nx = self.T * COLS * self.rows[0] if self.rows else self.T * self.K // 64 * self.unit
        return [ArgSpec("in", (nx,), bfloat16),
                ArgSpec("in", (self.copies * self.K // GROUP * OBJ // 2,), bfloat16),
                ArgSpec("out", (self.copies * n,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        T, K, p, tok, gu, C = self.T, self.K, self.p, self.tok, self.gu, self.copies
        G, nb = K // GROUP, T // tok  # (groups, items per group)
        u8, f32 = np.dtype[np.uint8], np.dtype[np.float32]
        ntg = tok // 8  # (token groups of an item)
        x_item, o_item = ntg * 16 * self.unit * 2, tok * 2 * GROUP if self.rows_out else ntg * 16 * 72  # (bytes)
        x_ty, xc_ty = np.ndarray[(x_item,), u8], np.ndarray[(ROWS * x_item,), u8]
        o_ty, oc_ty = np.ndarray[(o_item,), u8], np.ndarray[(ROWS * o_item,), u8]
        r_ty = np.ndarray[(OBJ,), u8]
        u_ty = np.ndarray[(GROUP * tok,), f32]
        y_ty = np.ndarray[(ntg * 1024,), np.dtype[bfloat16]]  # (gu: the gates' silu)
        k_rot = Kernel(f"{p}_rot", "kernel.o", [x_ty, r_ty, o_ty, u_ty, u_ty, *([y_ty, np.int32] if gu else [])])
        n = self.n_rounds
        x3 = [ObjectFifo(xc_ty, name=f"pr_x{c}", depth=2) for c in range(COLS)]
        bc = nb >= COLS * ROWS  # (every core on one group at a time: the objects broadcast)
        r3 = [ObjectFifo(r_ty, name="pr_r" if bc else f"pr_r{c}", depth=2) for c in range(1 if bc else COLS)]
        o3 = [ObjectFifo(oc_ty, name=f"pr_o{c}", depth=2) for c in range(COLS)]
        xs = [x3[c].cons().split([x_item * m for m in range(ROWS)], tile=Tile(c, 1), obj_types=[x_ty] * ROWS,
                                 names=[f"pr_x{c}_{m}" for m in range(ROWS)], depths=[2] * ROWS) for c in range(COLS)]
        rs = r3 * COLS if bc else [r3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"pr_rb{c}") for c in range(COLS)]
        od = 1 if self.bf16 else 2  # (bf16 blocks: 16 KB of x objects; the output, written last, drains meanwhile)
        os_ = [o3[c].prod().join([o_item * m for m in range(ROWS)], tile=Tile(c, 1), obj_types=[o_ty] * ROWS,
                                 names=[f"pr_o{c}_{m}" for m in range(ROWS)], depths=[od] * ROWS) for c in range(COLS)]

        t = max(nb // (COLS * ROWS), 1)  # (consecutive rounds of one group)

        def core(x_f, r_f, o_f, rot, ua, ub, *y):
            for _ in forever_():
                for _ in range_(n // t):
                    if C > 1:  # (t = 1: the item's copies in turn, x held)
                        x = x_f.acquire(1)
                        for _ in range(C):
                            r, o = r_f.acquire(1), o_f.acquire(1)
                            rot(x, r, o, ua, ub)
                            o_f.release(1)
                            r_f.release(1)
                        x_f.release(1)
                        continue
                    r = r_f.acquire(1)
                    for _ in range_(t):
                        x, o = x_f.acquire(1), o_f.acquire(1)
                        rot(x, r, o, ua, ub, *([y[0], 0] if gu else []))  # (gu: silu's opaque runtime zero)
                        o_f.release(1)
                        x_f.release(1)
                    r_f.release(1)

        rd = {"depth": 1} if self.rows_out else {}  # (rows out: 8 KB items; the object single-buffered, one per group)
        workers = [Worker(core, [xs[c][m].cons(), rs[c].cons(**rd), os_[c][m].prod(), k_rot,
                                 Buffer(u_ty, name=f"pr_ua{c}_{m}"), Buffer(u_ty, name=f"pr_ub{c}_{m}"),
                                 *([Buffer(y_ty, name=f"pr_y{c}_{m}")] if gu else [])],
                          tile=Tile(c, 2 + m), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for m in range(ROWS)]
        # (bf16 elements: the input, its token group's row, a group's run of it in a token group; the output's; the
        # rotation objects)
        nx, xr, xw = T * K // 64 * self.unit, K // 8 * self.unit, 16 * self.unit
        if self.rows:
            nx = T * COLS * self.rows[0]
        no, orow, ow = T * K * 9 // 16, K // 8 * 36, 16 * 36
        if self.rows_out:  # (rows: a token's K, a group's 128 of them)
            no, orow, ow = T * K, K, GROUP
        nro = C * G * OBJ // 2

        def walk(c: int, size: int, row: int, run: int, nc: int = 1, cstep: int = 0, per: int = ntg) -> TensorAccessPattern:
            """Column c's walk over the rounds of an argument of ``size`` elements in token-group rows of ``row`` (``per``
            of them an item; rows_out's output: token rows, ``tok``), a group's ``run`` of each (runs of 1024 elements: a
            shim BD's inner dimension holds at most 1023 words): [group (or round)][the column's token groups][run]; ``nc``
            copies: [round][copy, ``cstep`` apart][..][run]."""
            cut = min(run, 1024)
            if nb >= COLS * ROWS:  # (a group's items over t = nb / 32 rounds: the column's 4 t of them)
                base, dims, steps = 4 * t * per * c * row, [G, 4 * t * per], [run, row]
            else:  # (a round's items over 32 / nb groups)
                s = COLS * ROWS // nb
                base, dims, steps = 4 * c % nb * per * row + 4 * c // nb * run, [n, 4 * per], [s * run, row]
            if nc > 1:
                assert cut == run
                return TensorAccessPattern((size,), base, [dims[0], nc, dims[1], run], [steps[0], cstep, steps[1], 1])
            if cut < run:
                return TensorAccessPattern((size,), base, [*dims, run // cut, cut], [*steps, cut, 1])
            return TensorAccessPattern((size,), base, [1, *dims, run], [0, *steps, 1])

        def ro_tap(c: int) -> TensorAccessPattern:
            """The objects [copy][group] of column c's rounds (broadcast: every core's), a round's copies in turn."""
            go = G * OBJ // 2
            if bc:
                return TensorAccessPattern((nro,), 0, [1, G, C, OBJ // 2], [0, OBJ // 2, go, 1])
            s = COLS * ROWS // nb
            return TensorAccessPattern((nro,), 4 * c // nb * OBJ // 2, [1, n, C, OBJ // 2], [0, s * OBJ // 2, go, 1])

        def rows_taps(c: int) -> list[TensorAccessPattern]:
            """(rows) Column c's fills, one per column block: [round (repeat)][the column's 4 items' tokens][gate | up]
            [128] (a round's s = 32 / nb groups: the column's the s r + 4 c / nb-th)."""
            P, U = self.rows
            W, bg = COLS * P, K // COLS // GROUP  # (a row's elements; a block's groups)
            s = COLS * ROWS // nb
            e, t0 = 4 * c // nb, 4 * c % nb * tok
            out = []
            for b in range(COLS):
                r0, r1 = -(-(bg * b - e) // s), (bg * (b + 1) - 1 - e) // s + 1  # (rounds whose group is in block b)
                if r1 > r0:
                    g0 = s * r0 + e
                    out.append(TensorAccessPattern((nx,), t0 * W + b * P + GROUP * (g0 - bg * b), [r1 - r0, 4 * tok, 2, GROUP],
                                                   [s * GROUP, W, U, 1]))
            assert sum(t.sizes[0] for t in out) == n, (c, [t.sizes for t in out])
            return out

        def seq(X, RO, OUT, xh, rh, oh):
            tg = TaskGroup()
            for c in range(COLS):
                if not self.rows:
                    xh[c].fill(X, walk(c, nx, xr, xw), group=tg)
                if c < len(rh):
                    rh[c].fill(RO, ro_tap(c), group=tg)
                oh[c].drain(OUT, walk(c, no, orow, ow, per=tok) if self.rows_out else walk(c, C * no, C * orow, ow, C, orow),
                            group=tg, wait=True)
                for tap in rows_taps(c) if self.rows else ():  # (after the column's drain: past 4 queued tasks a fill
                    # waits for its first to complete, which the cores reach only while their outputs drain)
                    xh[c].fill(X, tap, group=tg)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(nx,), np.dtype[bfloat16]], np.ndarray[(nro,), np.dtype[bfloat16]],
                           np.ndarray[(C * no,), np.dtype[bfloat16]],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(x3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(r3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()
