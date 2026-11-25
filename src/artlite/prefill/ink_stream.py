"""Inkling's residual stream in the prefill: the operators between the GEMMs and the attention that carry a chunk's fp32
states (``models.inkling2.prefill``: the normed embedding, the block states, the residual cur; rows of the hidden size
H, token-major).

- ``InkMixOperator``: a depth mix (``InklingConfig``'s attention residuals) of n states with one query: per token
  r_b = rms(s_b) (eps ``MIX_EPS``), w = softmax_b(q . s_b / r_b / sqrt(H)), x = sum_b w_b s_b / r_b, as bf16 rows (the
  next norm's input, ``prefill.norm.PrefillNormOperator``). Two passes over the states: the scores, then the sum. Token t
  on column t % 8, core (t / 8) % 4: per column a chained task of a BD a state, each a token's row, repeated over the
  column's tokens; the memtile splits each 4 of them to the cores, each its token's rows twice (pass A, pass B) in chunks
  of ``CH`` channels; the query resident (hi + lo).
- ``InkResidOperator``: a post conv and the residual: per token conv = (y_t + sum_k taps_k y_(t - 3 + k)) / sqrt(5) of its
  input rows y (the o GEMM's output, the MoE combine's; bf16), the 3 positions before the chunk from the visit's history
  window (the decode's: [row 8][column slice] fp32, ``P_HA`` its first row), cur += conv (a block visit: cur' = conv in
  another state); the inputs of the chunk's last 3 valid tokens (``P_N`` = n >= 3) into the window's rows 0 .. 2 (the
  decode's next step: its window's first position p0 + n - 3). Column c its slice of the hidden (``C`` channels), one core.
- ``InkGateOperator``: the attention's output gate: o (bfp16 panels, ``prefill.attn.AttnOperator``'s A) times sigmoid of
  the gate logits (the q GEMM's "gate" block, bf16), per channel. Column c its KV head's query heads, one core.
- ``InkBlendOperator``: a loop visit's controller of a stream: g = g_a + g_b (its down's two halves, the GEMMs' fp32 sums of
  plane pairs: 8 a column), out = (1 - tanh(F g)) cur + tanh(G g) msg (cur, msg: the stream's mix and message, bf16
  rows), F and G the image's ``ctl_<stream>`` rows (bf16 [288][64] a column, read in place). Column c its slice, core r
  its 72 channels: their F, G rows resident as hi + lo panels, F g and G g on the MMUL per 8 tokens.
"""

import numpy as np
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

from ..model_config import InklingConfig
from ..operators import templating
from ..operators.base import ArgSpec, ArtcOperator
from ..operators.chain import ChainPart, chained_fill
from ..operators.design import COLS, MAX_BD_ITER, stream_walk
from ..operators.kernel_source import kernel_prefix
from ..operators.util import Attr, regions
from .attn.ink import P_N

_SRC = templating.load("artlite/prefill/ink_stream.cc.jinja")

CH = 288  # channels of a chunk (fp32: 576 bf16 elements, a shim BD row)
P_HA = "is_ha"  # bf16 offset of the conv history's first row in a column's window (the decode's window rows a .. a + 2)
TB = 4  # tokens of a resid block (L1: its inputs and cur' in pairs)


class InkMixOperator(ArtcOperator):
    """A depth mix of T tokens (module docstring): runtime sequence S0 .. S(n - 1) (the states, (T, H) fp32 as bf16 pairs),
    Q (the query, H fp32), X (out: (T, H) bf16)."""

    STACK = 1536  # (the coefficients' frame and kmath's exp2)
    p = kernel_prefix("imx")

    def __init__(self, *, T: int, n: int, H: int = 2304, eps: float = InklingConfig.MIX_EPS):
        assert 1 <= n <= 8 and T % (4 * COLS) == 0 and H % CH == 0 and 2 * H % 576 == 0
        self.T, self.n, self.H, self.eps = T, n, H, eps
        super().__init__()

    def workspace(self) -> dict[str, int]:
        return regions([("qh", 2 * self.H), ("ql", 2 * self.H), ("acc", 128 * self.n), ("cf", 64)])

    def get_kernel_src(self) -> str:
        sc = float(np.float32(1 / np.log(2) / np.sqrt(self.H)))
        return _SRC.render(op="mix", p=self.p, w=Attr(self.workspace()), H=self.H, N=self.n, CH=CH, eps=repr(self.eps),
                           SC=repr(sc))

    def get_arg_spec(self) -> list[ArgSpec]:
        T, H = self.T, self.H
        return [*(ArgSpec("in", (2 * T * H,), bfloat16) for _ in range(self.n)), ArgSpec("in", (2 * H,), bfloat16),
                ArgSpec("out", (T * H,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, H, n = self.p, self.T, self.H, self.n
        bf = np.dtype[bfloat16]
        R = 2 * H  # (bf16 elements of an fp32 row)
        x_ty = np.ndarray[(n * 2 * CH,), bf]  # a chunk of a token's states
        q_ty = np.ndarray[(R,), bf]
        row_ty = np.ndarray[(H,), bf]  # a token's output row
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_q = Kernel(f"{p}_q", "kernel.o", [q_ty, ws_ty])
        k_a = Kernel(f"{p}_a", "kernel.o", [x_ty, ws_ty, np.int32])
        k_c = Kernel(f"{p}_c", "kernel.o", [ws_ty])
        k_b = Kernel(f"{p}_b", "kernel.o", [x_ty, ws_ty, row_ty, np.int32])
        n_grp = T // (4 * COLS)  # (groups of 4 tokens a column)
        nk = H // CH

        # a group: 4 of the column's tokens (8 apart), each [state n][R]; each core its token's twice ([pass 2][chunk][state]
        # [CH fp32])
        part = n * R
        a3 = [ObjectFifo(np.ndarray[(4 * part,), bf], name=f"mx_a{c}", depth=2) for c in range(COLS)]
        walk = stream_walk(part, [(nk, 2 * CH), (n, R), (2 * CH, 1)])
        a1 = [a3[c].cons().split([r * part for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[x_ty] * 4,
                                 names=[f"mx_a{c}_{r}" for r in range(4)], to_stream=[walk] * 4, repeat_counts=[2] * 4)
              for c in range(COLS)]
        qf = [ObjectFifo(q_ty, name=f"mx_q{c}", depth=1) for c in range(COLS)]
        o3 = [ObjectFifo(np.ndarray[(4 * H,), bf], name=f"mx_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([r * H for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[row_ty] * 4,
                                names=[f"mx_o{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(a_h, q_h, o_h, fq, fa, fc, fb, ws):
            for _ in forever_():
                q = q_h.acquire(1)
                fq(q, ws)
                q_h.release(1)
                for _ in range_(n_grp):
                    for k in range_(nk):
                        x = a_h.acquire(1)
                        fa(x, ws, k)
                        a_h.release(1)
                    fc(ws)
                    out = o_h.acquire(1)
                    for k in range_(nk):
                        x = a_h.acquire(1)
                        fb(x, ws, out, k)
                        a_h.release(1)
                    o_h.release(1)

        workers = [Worker(core, [a1[c][r].cons(), qf[c].cons(), o1[c][r].prod(), k_q, k_a, k_c, k_b,
                                 Buffer(ws_ty, name=f"mx_ws{c}_{r}")],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]

        def seq(*args):
            *S, Q, X, ah, qh, oh = args
            tg = TaskGroup()
            for c in range(COLS):
                nt = T // COLS  # (the column's tokens c, c + 8, ..)
                for t0 in range(0, nt, MAX_BD_ITER):  # (a chained fill repeats at most MAX_BD_ITER times)
                    m = min(MAX_BD_ITER, nt - t0)
                    chained_fill(ah[c], S[0], [ChainPart(s_, TensorAccessPattern((2 * T * H,), (c + COLS * t0) * R,
                                                                                 [m, 1, R // 576, 576], [COLS * R, 0, 576, 1]))
                                               for s_ in S], tg)
                qh[c].fill(Q, TensorAccessPattern((R,), 0, [1, 1, R // 576, 576], [0, 0, 576, 1]), group=tg)
                # (a group's 4 rows: tokens c + 8 (4 j + r))
                oh[c].drain(X, TensorAccessPattern((T * H,), c * H, [n_grp, 4, 2, H // 2], [4 * COLS * H, COLS * H, H // 2, 1]),
                            group=tg, wait=True)
            tg.finish()

        types = [np.ndarray[(2 * T * H,), bf] for _ in range(n)] + [np.ndarray[(R,), bf], np.ndarray[(T * H,), bf]]
        rt = Runtime(seq, [*types, [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(qf)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


class InkResidOperator(ArtcOperator):
    """A post conv and the residual of T tokens (module docstring). Runtime sequence: Y (the conv's inputs: rows of
    ``y_pitch`` bf16, column c's slice at c ``y_col``), CUR (non-``block``: (T, H) fp32 as bf16 pairs), TAPS ([COLS][4][C]
    bf16), HIST (the visit's history windows: per column ``hist_rows`` rows of C fp32; read at ``P_HA``, its rows 0 .. 2
    written), OUT ((T, H) fp32: cur', in place over CUR or a block's new state). Parameters ``params``."""

    STACK = 1024
    p = kernel_prefix("irs")

    def __init__(self, *, T: int, block: bool, y_pitch: int, y_col: int, H: int = 2304, hist_rows: int = 8):
        self.C = H // COLS
        assert T % TB == 0 and T >= 8 and self.C % 32 == 0 and y_col >= self.C and y_pitch >= COLS * y_col - (y_col - self.C)
        self.T, self.block, self.y_pitch, self.y_col, self.H, self.hist_rows = T, block, y_pitch, y_col, H, hist_rows
        super().__init__()

    def workspace(self) -> dict[str, int]:
        C = self.C
        return regions([("taps", 8 * C), ("win", 16 * C), ("snap", 12 * C), ("st", 64)])

    def get_kernel_src(self) -> str:
        return _SRC.render(op="resid", p=self.p, w=Attr(self.workspace()), C=self.C, TB=TB, BLOCK=self.block,
                           SC=repr(float(np.float32(5 ** -0.5))))

    @property
    def n_hist(self) -> int:
        return COLS * self.hist_rows * 2 * self.C

    def get_arg_spec(self) -> list[ArgSpec]:
        T, H = self.T, self.H
        cur = [] if self.block else [ArgSpec("in", (2 * T * H,), bfloat16)]
        return [ArgSpec("in", (T * self.y_pitch,), bfloat16), *cur, ArgSpec("in", (COLS * 4 * self.C,), bfloat16),
                ArgSpec("out", (self.n_hist,), bfloat16), ArgSpec("out", (2 * T * H,), bfloat16)]

    def params(self, n: int, a: int) -> dict[str, int]:
        """The scratchpad parameters of a chunk of n valid tokens whose history is its windows' rows a .. a + 2."""
        assert 3 <= n <= self.T and 0 <= a <= self.hist_rows - 3, (n, a)
        return {P_N: n, P_HA: 2 * self.C * a}

    def get_mlir_artifact(self) -> Module:
        p, T, H, C = self.p, self.T, self.H, self.C
        bf = np.dtype[bfloat16]
        x_ty = np.ndarray[(TB * C * (1 if self.block else 3),), bf]  # a block's inputs (and cur)
        b_ty = np.ndarray[(4 * C + 6 * C,), bf]  # the taps, the history
        o_ty = np.ndarray[(TB * C,), np.dtype[np.float32]]
        s_ty = np.ndarray[(3 * C,), np.dtype[np.float32]]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [b_ty, ws_ty])
        k_run = Kernel(f"{p}_run", "kernel.o", [x_ty, ws_ty, o_ty, np.int32])
        k_snap = Kernel(f"{p}_snap", "kernel.o", [ws_ty, s_ty])
        p_n, p_ha = ScratchpadParameter(P_N, np.int32), ScratchpadParameter(P_HA, np.int32)
        nb = T // TB
        af = [ObjectFifo(x_ty, name=f"rs_a{c}", depth=2) for c in range(COLS)]
        bf_ = [ObjectFifo(b_ty, name=f"rs_b{c}", depth=1) for c in range(COLS)]
        of = [ObjectFifo(o_ty, name=f"rs_o{c}", depth=2) for c in range(COLS)]
        sf = [ObjectFifo(s_ty, name=f"rs_s{c}", depth=1) for c in range(COLS)]

        def core(a_h, b_h, o_h, s_h, fprm, frun, fsnap, ws, pn):
            for _ in forever_():
                b = b_h.acquire(1)
                fprm(b, ws)
                b_h.release(1)
                n = pn.read()
                for _ in range_(nb):
                    x = a_h.acquire(1)
                    out = o_h.acquire(1)
                    frun(x, ws, out, n)
                    o_h.release(1)
                    a_h.release(1)
                out = s_h.acquire(1)
                fsnap(ws, out)
                s_h.release(1)

        workers = [Worker(core, [af[c].cons(), bf_[c].cons(), of[c].prod(), sf[c].prod(), k_prm, k_run, k_snap,
                                 Buffer(ws_ty, name=f"rs_ws{c}"), p_n],
                          tile=Tile(c, 2), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS)]
        yp, yc, nh, hr = self.y_pitch, self.y_col, self.n_hist, self.hist_rows

        def seq(*args):
            Y, *cur, TAPS, HIST, OUT, ah, bh, oh, sh = args
            sync_parameters()
            tg = TaskGroup()
            for c in range(COLS):
                for b0 in range(0, nb, MAX_BD_ITER):  # (a chained fill repeats at most MAX_BD_ITER times)
                    m = min(MAX_BD_ITER, nb - b0)
                    parts = [ChainPart(Y, TensorAccessPattern((T * yp,), c * yc + TB * b0 * yp, [m, 1, TB, C],
                                                              [TB * yp, 0, yp, 1]))]
                    if cur:
                        parts.append(ChainPart(cur[0], TensorAccessPattern((2 * T * H,), 2 * C * c + TB * b0 * 2 * H,
                                                                           [m, 1, TB, 2 * C], [TB * 2 * H, 0, 2 * H, 1])))
                    chained_fill(ah[c], Y, parts, tg)
                chained_fill(bh[c], TAPS, [
                    ChainPart(TAPS, TensorAccessPattern((COLS * 4 * C,), 4 * C * c, [1, 1, 1, 4 * C], [0, 0, 0, 1])),
                    ChainPart(HIST, TensorAccessPattern((nh,), c * hr * 2 * C, [1, 1, 3, 2 * C], [0, 0, 2 * C, 1]),
                              offset_parameter=p_ha)], tg)
                oh[c].drain(OUT, TensorAccessPattern((2 * T * H,), 2 * C * c, [1, T, 1, 2 * C], [0, 2 * H, 0, 1]),
                            group=tg, wait=True)
                sh[c].drain(HIST, TensorAccessPattern((nh,), c * hr * 2 * C, [1, 1, 3, 2 * C], [0, 0, 2 * C, 1]),
                            group=tg, wait=True)
            tg.finish()

        types = [np.ndarray[(T * yp,), bf]] + ([] if self.block else [np.ndarray[(2 * T * H,), bf]]) + \
            [np.ndarray[(COLS * 4 * C,), bf], np.ndarray[(nh,), bf], np.ndarray[(2 * T * H,), bf]]
        rt = Runtime(seq, [*types, [f.prod(tile=Tile(c, 0)) for c, f in enumerate(af)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(bf_)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(sf)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


class InkGateOperator(ArtcOperator):
    """The attention output's gates of T tokens (module docstring). Runtime sequence: A (the attention's output: T x
    COLS ``heads`` hd as bfp16 panels [T / 8][COLS heads hd / 8]), G (the q GEMM's output, rows of ``lq_pitch`` COLS
    bf16, column c's gate logits from c ``lq_pitch`` + ``gate_off``), O (out: A's layout; A in place)."""

    STACK = 2048  # (kmath's sigmoid of 64 lanes)
    p = kernel_prefix("igt")

    def __init__(self, *, T: int, lq_pitch: int, gate_off: int, heads: int = 3, hd: int = 128):
        assert T % 8 == 0 and heads * hd % 8 == 0
        self.T, self.lq_pitch, self.gate_off, self.heads, self.hd = T, lq_pitch, gate_off, heads, hd
        self.np_ = heads * hd // 8  # (a column's panels of a token group)
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SRC.render(op="gate", p=self.p, NP=self.np_)

    @property
    def n_a(self) -> int:
        return self.T * COLS * self.heads * self.hd * 9 // 16

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.n_a,), bfloat16), ArgSpec("in", (self.T * COLS * self.lq_pitch,), bfloat16),
                ArgSpec("out", (self.n_a,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, npn = self.p, self.T, self.np_
        bf = np.dtype[bfloat16]
        pan = npn * 36  # (bf16 units of a column's panels of a token group)
        a_ty = np.ndarray[(pan,), bf]
        g_ty = np.ndarray[(8 * 8 * npn,), bf]
        k_run = Kernel(f"{p}_run", "kernel.o", [a_ty, g_ty, a_ty])
        n_tg = T // 8
        af = [ObjectFifo(a_ty, name=f"gt_a{c}", depth=2) for c in range(COLS)]
        gf = [ObjectFifo(g_ty, name=f"gt_g{c}", depth=2) for c in range(COLS)]
        of = [ObjectFifo(a_ty, name=f"gt_o{c}", depth=2) for c in range(COLS)]

        def core(a_h, g_h, o_h, frun):
            for _ in forever_():
                for _ in range_(n_tg):
                    a = a_h.acquire(1)
                    g = g_h.acquire(1)
                    o = o_h.acquire(1)
                    frun(a, g, o)
                    o_h.release(1)
                    g_h.release(1)
                    a_h.release(1)

        workers = [Worker(core, [af[c].cons(), gf[c].cons(), of[c].prod(), k_run], tile=Tile(c, 2), stack_size=self.STACK,
                          dynamic_objfifo_lowering=True, while_true=False) for c in range(COLS)]
        n_a, N = self.n_a, COLS * self.lq_pitch
        row = COLS * pan  # (a token group's panels of every column)

        def seq(A, G, O, ah, gh, oh):
            tg = TaskGroup()
            for c in range(COLS):
                ah[c].fill(A, TensorAccessPattern((n_a,), c * pan, [1, n_tg, 1, pan], [0, row, 0, 1]), group=tg)
                gh[c].fill(G, TensorAccessPattern((T * N,), c * self.lq_pitch + self.gate_off, [1, n_tg, 8, 8 * npn],
                                                  [0, 8 * N, N, 1]), group=tg)
                oh[c].drain(O, TensorAccessPattern((n_a,), c * pan, [1, n_tg, 1, pan], [0, row, 0, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(n_a,), bf], np.ndarray[(T * N,), bf], np.ndarray[(n_a,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(af)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(gf)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(of)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


class InkBlendOperator(ArtcOperator):
    """A controller's blend of T tokens (module docstring). Runtime sequence: GA, GB (the down halves' outputs: rows of
    COLS ``d_pitch`` bf16, column c's 8 fp32 at c ``d_pitch``), CS, MS ((T, H) bf16), W (the image's ``ctl_<stream>``
    region: per column [F 288][64] then [G 288][64] bf16, ``w_stride`` bf16 apart), OUT ((T, H) bf16)."""

    STACK = 4096  # (the run's frame: kmath's fp32 sigmoid of 64 lanes, twice)
    p = kernel_prefix("ibl")

    def __init__(self, *, T: int, d_pitch: int, w_stride: int, H: int = 2304, rank: int = 64):
        self.C = H // COLS
        assert T % 8 == 0 and self.C == 4 * 72 and rank == 64 and w_stride >= 2 * self.C * rank
        self.T, self.d_pitch, self.w_stride, self.H = T, d_pitch, w_stride, H
        super().__init__()

    def workspace(self) -> dict[str, int]:
        return regions([("wp", 2 * 9 * 16 * 72), ("gp", 16 * 72), ("c", 18 * 256)])

    def get_kernel_src(self) -> str:
        return _SRC.render(op="blend", p=self.p, wb=Attr(self.workspace()), C=self.C)

    def get_arg_spec(self) -> list[ArgSpec]:
        T, H, nd = self.T, self.H, self.T * COLS * self.d_pitch
        return [ArgSpec("in", (nd,), bfloat16), ArgSpec("in", (nd,), bfloat16), ArgSpec("in", (T * H,), bfloat16),
                ArgSpec("in", (T * H,), bfloat16), ArgSpec("in", (COLS * self.w_stride,), bfloat16),
                ArgSpec("out", (T * H,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        p, T, H, C = self.p, self.T, self.H, self.C
        bf = np.dtype[bfloat16]
        w_ty = np.ndarray[(8 * 64,), bf]  # an output tile's rows
        x_ty = np.ndarray[(2048 + 16 * C,), bf]  # a block's g halves, cur, msg
        o_ty = np.ndarray[(8 * 72,), bf]
        ws_ty = np.ndarray[(self.workspace()["size"] // 4,), np.dtype[np.float32]]
        k_prm = Kernel(f"{p}_prm", "kernel.o", [w_ty, ws_ty, np.int32])
        k_run = Kernel(f"{p}_run", "kernel.o", [x_ty, ws_ty, o_ty, np.int32])
        nb = T // 8
        part = 2 * 72 * 64  # (a core's rows: F, G)
        w3 = [ObjectFifo(np.ndarray[(4 * part,), bf], name=f"bl_w{c}", depth=1) for c in range(COLS)]
        w1 = [w3[c].cons().split([r * part for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[w_ty] * 4,
                                 names=[f"bl_w{c}_{r}" for r in range(4)]) for c in range(COLS)]
        x3 = [ObjectFifo(x_ty, name=f"bl_x{c}", depth=2) for c in range(COLS)]
        x1 = [x3[c].cons().forward(tile=Tile(c, 1), obj_type=x_ty, depth=2, name=f"bl_xm{c}") for c in range(COLS)]
        o3 = [ObjectFifo(np.ndarray[(4 * 8 * 72,), bf], name=f"bl_o{c}", depth=2) for c in range(COLS)]
        o1 = [o3[c].prod().join([r * 8 * 72 for r in range(4)], tile=Tile(c, 1), depths=[2] * 4, obj_types=[o_ty] * 4,
                                names=[f"bl_o{c}_{r}" for r in range(4)]) for c in range(COLS)]

        def core(w_h, x_h, o_h, fprm, frun, ws, r):
            for _ in forever_():
                for t in range_(18):
                    w = w_h.acquire(1)
                    fprm(w, ws, t)
                    w_h.release(1)
                for _ in range_(nb):
                    x = x_h.acquire(1)
                    o = o_h.acquire(1)
                    frun(x, ws, o, r)
                    o_h.release(1)
                    x_h.release(1)

        workers = [Worker(core, [w1[c][r].cons(), x1[c].cons(depth=1), o1[c][r].prod(), k_prm, k_run,
                                 Buffer(ws_ty, name=f"bl_ws{c}_{r}"), r],
                          tile=Tile(c, 2 + r), stack_size=self.STACK, dynamic_objfifo_lowering=True, while_true=False)
                   for c in range(COLS) for r in range(4)]
        dp, nw, nd = self.d_pitch, COLS * self.w_stride, T * COLS * self.d_pitch
        Nd = COLS * dp

        def seq(GA, GB, CS, MS, W, OUT, wh, xh, oh):
            tg = TaskGroup()
            for c in range(COLS):
                # (core r's rows: F rows 72 r .. and G rows 72 r .., a repeat each)
                chained_fill(wh[c], W, [TensorAccessPattern((nw,), c * self.w_stride, [4, 2, 8, 576],
                                                            [part // 2, 288 * 64, 576, 1])], tg)
                for b0 in range(0, nb, MAX_BD_ITER):  # (a chained fill repeats at most MAX_BD_ITER times)
                    m = min(MAX_BD_ITER, nb - b0)
                    chained_fill(xh[c], GA, [
                        ChainPart(GA, TensorAccessPattern((nd,), 8 * b0 * Nd, [m, 8, COLS, 16], [8 * Nd, Nd, dp, 1])),
                        ChainPart(GB, TensorAccessPattern((nd,), 8 * b0 * Nd, [m, 8, COLS, 16], [8 * Nd, Nd, dp, 1])),
                        ChainPart(CS, TensorAccessPattern((T * H,), c * C + 8 * b0 * H, [m, 1, 8, C], [8 * H, 0, H, 1])),
                        ChainPart(MS, TensorAccessPattern((T * H,), c * C + 8 * b0 * H, [m, 1, 8, C], [8 * H, 0, H, 1]))],
                        tg)
                oh[c].drain(OUT, TensorAccessPattern((T * H,), c * C, [nb, 4, 8, 72], [8 * H, 72, H, 1]), group=tg, wait=True)
            tg.finish()

        rt = Runtime(seq, [np.ndarray[(nd,), bf], np.ndarray[(nd,), bf], np.ndarray[(T * H,), bf],
                           np.ndarray[(T * H,), bf], np.ndarray[(nw,), bf], np.ndarray[(T * H,), bf],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(w3)],
                           [f.prod(tile=Tile(c, 0)) for c, f in enumerate(x3)],
                           [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3)]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


__all__ = ["CH", "P_HA", "TB", "InkBlendOperator", "InkGateOperator", "InkMixOperator", "InkResidOperator"]
