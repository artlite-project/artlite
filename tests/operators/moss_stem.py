"""The MOSS-Transcribe encoder stem's GEMM contract, pinned as one standalone operator.

The model's first two stages are plain 3-tap convs over time with GELU (``conv1`` 80 -> 1024 channels, stride 1;
``conv2`` 1024 -> 1024, stride 2), then the position table is added after the second GELU. This operator runs that
arithmetic as one GEMM family in the vision overlay's panel layouts, so the contracts the rest of the encoder reuses
are fixed here:

- **A, the im2col panels**: a row's 3 taps are flattened into K = 3 x C_in channels (``im2col``), K padded to whole
  k-tiles with zeros, and *one extra channel carries 1.0* (the bias below). Packed as bfp16ebs8 panels
  (``layout.panels``: [K/8][tokens/8][8 tokens][9 B]) and re-ordered into the **pop order of the MAC loop**
  (``gemm/mac.py``): [band of 16 tokens][k-tile][k-step][half][8 tokens x 8 k]. One A object = a (band, k-tile) = 16
  panels = 1152 B; the overlay's ``design.A_OBJ`` (128 tokens x 64 k = 9216 B) is exactly 8 of them.
- **W, the weights**: [N/8 output panels][K/8 k-blocks][9 B] panels of ``w`` (output channels as rows), re-ordered to
  [column of 160 outputs][k-tile][16-output block][k-step][half]. One object = a (column, k-tile) = 160 panels =
  11520 B; the overlay's 80-channel ``B`` slot (``design.Q``, 5760 B) is half of one.
- **C, the output**: the MAC's fp32 stripe [8-token row block][8-channel column block][64], 2560 values per core (16
  tokens x 160 channels) — ``layout.ctiles``' layout with this operator's N instead of 80-channel blocks. The epilogue
  rounds it to bf16, applies **GELU through its table** (``kmath.GELU_ERF``: the model's erf form) and, with ``res``,
  adds the position table (bf16, one value per (token, channel), the same blocked layout) — that add is
  ``MossPosAddOperator``, a dispatch of its own (the core's two input channels, below).
- **bias** is folded into the GEMM as the extra A channel (A's K-th channel is 1.0 for every token, W's K-th column is
  the bias), so no bias buffer exists and the products add it in fp32; the bias itself is bfp16-rounded like every
  weight (the 2 % gate absorbs that — the reference adds it in fp32).

Work split: 8 columns of 160 output channels (N = 1280 = 16 heads x 80 dims, the residual width's pad) x 4 rows of 16
tokens (M = 64); one MAC call per k-tile (``K_TILE`` = 64, the overlay's ``design.KT``) with the core's 10 blocks of
16x16 outputs in the MAC loop's own order. One dispatch is one M-token tile: the host walks a 30 s chunk's 3000
(stride 1) or 1500 (stride 2) rows in tiles of 64. The position table is **not** this operator's third input: a core has
two input DMA channels (``docs/development.md`` "NPU pitfalls"), and A + W already take both, so ``MossPosAddOperator``
(the elementwise ``y + pos`` in the same blocked layout) runs as a second dispatch.

Streams (the GEMM pattern of ``gemm/design_parts.py``): A per core row and W per column enter their memtile and are
forwarded (a broadcast: A to the row's 8 cores, W to the column's 4); the position table per core row is split in its
memtile into the row's 8 columns' slices; the cores' C stripes join in each column's memtile for one drain.

Host only: ``compile`` builds the design (nothing dispatches), and the tests check the packers and the arithmetic in
the MAC's pop order against the fp32 reference (the A2 gate: relative L2 <= 2 %).
"""

from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import NPU2, Tile
from ml_dtypes import bfloat16

from artlite.gemm.mac import render_mac
from artlite.operators import kmath, templating
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.design import COLS, MAX_BD_ITER, ROWS
from artlite.operators.kernel_source import kernel_prefix
from artlite.vision import layout as L

TOK = 16  # tokens of a core row's band
NCOL = 160  # output channels of a column
N_TILE = COLS * NCOL  # 1280: the operator's N
M_TILE = ROWS * TOK  # 64: the operator's M (one token tile)
K_TILE = 64  # k of one MAC call (the overlay's k-tile)
ACT = "gelu_erf"  # the Whisper encoder's GELU form

_JINJA = Path(__file__).resolve().with_name("moss_stem.cc.jinja")  # (test-only operator: its template next to it)
TEMPLATE = templating.JINJA_ENV.from_string(_JINJA.read_text())


# ------------------------------------------------------------------------------------------------------------- host side

def im2col(x: np.ndarray, *, stride: int = 1, taps: int = 3, pad: int = 1) -> np.ndarray:
    """(T, C) -> (T_out, taps * C) fp32: row j is x[j stride - pad, + taps) flattened tap-major (tap * C + channel),
    the taps outside x zero. T_out = (T + 2 pad - taps) // stride + 1 (3000 -> 3000 or 1500)."""
    t, c = x.shape
    t_out = (t + 2 * pad - taps) // stride + 1
    xp = np.zeros((t + 2 * pad, c), np.float32)
    xp[pad:pad + t] = np.asarray(x, np.float32)
    idx = np.arange(t_out)[:, None] * stride + np.arange(taps)[None, :]
    return np.ascontiguousarray(xp[idx].reshape(t_out, taps * c))            # [tap][channel] per row


def fold_weights(w: np.ndarray, bias: np.ndarray, *, k: int) -> np.ndarray:
    """A conv's (C_out, C_in, taps) weights and its bias -> the GEMM's W (C_out, k): tap-major rows like ``im2col``,
    the bias in the K-th column (A's is 1.0), zeros to k."""
    c_out, c_in, taps = w.shape
    real = taps * c_in
    assert k >= real + 1 and k % K_TILE == 0, (k, real)
    out = np.zeros((c_out, k), np.float32)
    out[:, :real] = np.asarray(w, np.float32).transpose(0, 2, 1).reshape(c_out, real)
    out[:, real] = bias
    return out


def bias_channel(x: np.ndarray, k: int) -> np.ndarray:
    """The im2col matrix with the bias channel (1.0 at K = taps * C_in) and zeros to k columns."""
    m, real = x.shape
    assert k >= real + 1 and k % K_TILE == 0, (k, real)
    out = np.zeros((m, k), np.float32)
    out[:, :real] = x
    out[:, real] = 1.0
    return out


def a_image(x: np.ndarray) -> np.ndarray:
    """(M, K) -> the A image's bfp16 bytes in the MAC's pop order [band][k-tile][k-step][half][72 B]."""
    m, kk = x.shape
    assert m % (2 * TOK) == 0 and kk % K_TILE == 0, (m, kk)
    p = L.panels(x).reshape(kk // 8, m // 8, L.PANEL)               # [k-block][row panel][72 B]
    nkt, ks = kk // K_TILE, K_TILE // 8
    v = p.reshape(nkt, ks, m // TOK, 2, L.PANEL)                    # [kt][k-step][band][half][72]
    return np.ascontiguousarray(v.transpose(2, 0, 1, 3, 4)).reshape(-1)


def w_image(w: np.ndarray) -> np.ndarray:
    """(N, K) -> the W image's bfp16 bytes in the MAC's pop order [column][k-tile][16-output block][k-step][half]."""
    n, kk = w.shape
    assert n % NCOL == 0 and kk % K_TILE == 0, (n, kk)
    p = L.panels(w).reshape(kk // 8, n // 8, L.PANEL)               # [k-block][output panel][72 B]
    nkt, ks = kk // K_TILE, K_TILE // 8
    v = p.reshape(nkt, ks, n // NCOL, NCOL // 16, 2, L.PANEL)       # [kt][k-step][column][block][half][72]
    return np.ascontiguousarray(v.transpose(2, 0, 3, 1, 4, 5)).reshape(-1)


def to_c_f32(x: np.ndarray, n: int = N_TILE) -> np.ndarray:
    """(M, N) -> the C stripe layout [8-token row block][8-channel column block][64], fp32 (no rounding)."""
    m, nn = x.shape
    assert m % 8 == 0 and nn == n, (m, nn, n)
    return np.ascontiguousarray(np.asarray(x, np.float32).reshape(m // 8, 8, nn // 8, 8)
                                .transpose(0, 2, 1, 3)).reshape(-1)


def to_c(x: np.ndarray, n: int = N_TILE) -> np.ndarray:
    """``to_c_f32`` rounded to bf16: what the epilogue stores."""
    return to_c_f32(x, n).astype(bfloat16)


def from_c(c: np.ndarray, m: int, n: int = N_TILE) -> np.ndarray:
    """The stripe's inverse, fp32 (bf16 values converted elementwise)."""
    return np.ascontiguousarray(np.asarray(c, np.float32).reshape(m // 8, n // 8, 8, 8)
                                .transpose(0, 2, 1, 3)).reshape(m, n)


def mac_walk(a_img: np.ndarray, w_img: np.ndarray, m: int, n: int, kk: int) -> np.ndarray:
    """The fp32 C stripe of ``MAC<16, 160>`` over the packed images, as a host model of the MAC loop's pop order:
    the tests walk it to check the packing independently of the kernel."""
    nkt, ks = kk // K_TILE, K_TILE // 8
    a = L.unblocks(np.asarray(a_img).view(np.uint8).reshape(-1, 9)).reshape(m // TOK, nkt, ks, 2, 8, 8)
    w = L.unblocks(np.asarray(w_img).view(np.uint8).reshape(-1, 9)).reshape(n // NCOL, nkt, NCOL // 16, ks, 2, 8, 8)
    c = np.zeros((m // 8, n // 8, 8, 8), np.float64)                 # [8-token block][8-channel block][row][channel]
    for col in range(n // NCOL):
        for band in range(m // TOK):
            for kt in range(nkt):
                for block in range(NCOL // 16):
                    for st in range(ks):
                        for half in range(2):
                            for bh in range(2):
                                c[2 * band + half, col * 20 + 2 * block + bh] += (
                                    a[band, kt, st, half] @ w[col, kt, block, st, bh].T)
    return c.reshape(-1)


def column_tap(m: int, n: int, col: int) -> TensorAccessPattern:
    """A column's 4 cores' stripes of the Y (or position) image: [band][2 row blocks][8*NCOL], the columns' blocks
    contiguous - ROWS times 2 row blocks of 8 * NCOL elements = 8 * n = the 10,240 an n-1280 tile's column holds."""
    return TensorAccessPattern((m * n,), ROWS * 2 * 8 * NCOL * col, [ROWS * 2 * 8 * NCOL], [1])


# ------------------------------------------------------------------------------------------------------- the operator

class MossStemOperator(ArtcOperator):
    """The stem's GEMM + GELU (module docstring). Arguments: A (M x K im2col panels, in bf16 units), W (N x K, the
    same), optionally the position table (bf16, C layout) -> Y (M x N, bf16, C layout)."""

    p = kernel_prefix("mstem")
    STACK = 2048

    def __init__(self, *, K: int, M: int = M_TILE, N: int = N_TILE):
        assert M == M_TILE and N == N_TILE, "the stem's split: 4 rows of 16 tokens, 8 columns of 160 channels"
        assert K % K_TILE == 0 and K // K_TILE <= MAX_BD_ITER, f"K {K}: whole k-tiles of {K_TILE}, at most {MAX_BD_ITER}"
        self.K, self.M, self.N = K, M, N
        self.nkt = K // K_TILE
        super().__init__()

    a_obj = TOK * K_TILE * 9 // 8    # 1152 B: a band's k-tile
    w_obj = NCOL * K_TILE * 9 // 8   # 11520 B: a column's k-tile
    c_elems = TOK * NCOL             # 2560: a core's fp32 stripe

    def act_fn(self) -> object:
        """The GELU helper (one rendering per operator: its table words and the two L1 copies; the stripe is walked as
        one group of 32-value vectors: a contiguous C stripe, not the overlay's banded one)."""
        return kmath.act(kmath.TABLES[ACT], table_arg=True)

    def get_kernel_src(self) -> str:
        p = self.p
        act = self.act_fn()
        mac = render_mac(name=f"{p}_mac_t", k=K_TILE, f32=True)
        body = TEMPLATE.render(p=p, nsub=self.c_elems // 64, c_elems=self.c_elems, act=ACT, act_t=str(act),
                               add_only=False)
        return mac + "\n" + act.src + "\n" + body + f"""
extern "C" {{
// one 16 x 160 stripe of the k-tile GEMM: A (the band's 16 panels), W (the column's 160 panels), C (fp32 stripe)
void {p}_mac(const bfloat16 *__restrict a, const bfloat16 *__restrict w, float *__restrict c, float *__restrict cd)
{{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {p}_mac_t<{TOK}, {NCOL}>((const bfp16ebs8 *)a, (const void *)w,
        (const float __aie_dm_resource_c *)c, (float __aie_dm_resource_c *)cd);
}}
}}
"""

    def get_arg_spec(self) -> list[ArgSpec]:
        spec = [ArgSpec("in", (self.M * self.K * 9 // 16,), bfloat16),
                ArgSpec("in", (self.N * self.K * 9 // 16,), bfloat16)]
        spec.append(ArgSpec("out", (self.M * self.N,), bfloat16))
        return spec

    def get_mlir_artifact(self) -> Module:
        bf = np.dtype[bfloat16]
        ai, ni, ci, mi, ki, ncol, nkt = self.a_obj, self.w_obj, self.c_elems, self.M, self.K, self.N, self.nkt
        A_l2 = np.ndarray[(ai // 2,), bf]
        W_l2 = np.ndarray[(ni // 2,), bf]
        OBJ_ty = np.ndarray[(ci,), bf]
        C_ty = np.ndarray[(ci,), np.dtype[np.float32]]
        S_ty = np.ndarray[(ci,), bf]
        A_img = np.ndarray[(mi * ki * 9 // 16,), bf]
        W_img = np.ndarray[(ncol * ki * 9 // 16,), bf]
        OUT_img = np.ndarray[(mi * ncol,), bf]
        tab_ty = np.ndarray[(kmath.TABLES[ACT].table_words().size,), np.dtype[np.uint32]]
        p = self.p
        k_zero = Kernel(f"{p}_zero", "kernel.o", [C_ty])
        k_mac = Kernel(f"{p}_mac", "kernel.o", [A_l2, W_l2, C_ty, C_ty])
        k_epi = Kernel(f"{p}_epi", "kernel.o", [C_ty, S_ty, OBJ_ty, tab_ty, tab_ty])
        # A: per core row into its memtile, broadcast to the row's 8 cores; W: per column, broadcast to its 4 cores
        a_l3 = [ObjectFifo(A_l2, name=f"ms_a{row}", depth=2) for row in range(ROWS)]
        a_l1 = [a_l3[row].cons().forward(obj_type=A_l2, name=f"ms_a1_{row}", depth=2, tile=Tile(2 * row, 1))
                for row in range(ROWS)]
        w_l3 = [ObjectFifo(W_l2, name=f"ms_w{c}", depth=2) for c in range(COLS)]
        w_l1 = [w_l3[c].cons().forward(obj_type=W_l2, name=f"ms_w1_{c}", depth=2, tile=Tile(c, 1))
                for c in range(COLS)]
        # C: the column's 4 cores' stripes join in its memtile; one drain to the Y image
        C_l2 = np.ndarray[(ROWS * ci,), bf]  # (the column's joined object: 4 stripes, 10240 bf16)
        c_l3 = [ObjectFifo(C_l2, name=f"ms_o{c}", depth=2) for c in range(COLS)]
        c_l1 = [[] for _ in range(ROWS)]
        for c in range(COLS):  # (the column's 4 stripes join once in its memtile)
            parts = c_l3[c].prod().join([ci * r for r in range(ROWS)], obj_types=[OBJ_ty] * ROWS,
                                        names=[f"ms_o1_{c}_{r}" for r in range(ROWS)], depths=[2] * ROWS,
                                        tile=Tile(c, 1))
            for r in range(ROWS):
                c_l1[r].append(parts[r].prod())

        def core(a_h, w_h, o_h, zero, mac, epi, c, s, *tabs):
            """The k-tiles into the stripe, then the GELU epilogue into the output object."""
            for _ in forever_():
                zero(c)
                for _ in range_(nkt):
                    a = a_h.acquire(1)
                    w = w_h.acquire(1)
                    mac(a, w, c, c)
                    a_h.release(1)
                    w_h.release(1)
                out = o_h.acquire(1)
                epi(c, s, out, *tabs)
                o_h.release(1)

        def seq(A, W, OUT, af, wf, cf):
            tg = TaskGroup()
            for r in range(ROWS):
                af[r].fill(A, TensorAccessPattern((mi * ki * 9 // 16,), r * nkt * ai // 2, [nkt, ai // 2],
                                                  [ai // 2, 1]), group=tg, wait=False)
            for c in range(COLS):
                wf[c].fill(W, TensorAccessPattern((ncol * ki * 9 // 16,), c * nkt * ni // 2, [nkt, ni // 2],
                                                  [ni // 2, 1]), group=tg, wait=False)
            tg.finish()
            tgd = TaskGroup()
            for c in range(COLS):
                cf[c].drain(OUT, column_tap(mi, ncol, c), group=tgd, wait=True)
            tgd.finish()

        workers = []
        for r in range(ROWS):
            for c in range(COLS):
                args = [a_l1[r].cons(), w_l1[c].cons(), c_l1[r][c], k_zero, k_mac, k_epi,
                        Buffer(C_ty, name=f"ms_c{r}_{c}"), Buffer(S_ty, name=f"ms_s{r}_{c}")]
                for key in ("ab", "cd"):
                    args.append(Buffer(tab_ty, name=f"ms_{key}{r}_{c}", initial_value=kmath.TABLES[ACT].table_words()))
                workers.append(Worker(core, args, tile=Tile(c, r + 2), stack_size=self.STACK,
                                      dynamic_objfifo_lowering=True, while_true=False))
        a_prods = [a_l3[r].prod(tile=Tile(2 * r, 0)) for r in range(ROWS)]
        w_prods = [w_l3[c].prod(tile=Tile(c, 0)) for c in range(COLS)]
        c_conses = [c_l3[c].cons(tile=Tile(c, 0)) for c in range(COLS)]
        return Program(NPU2(), Runtime(seq, [A_img, W_img, OUT_img, a_prods, w_prods, c_conses]),
                       workers=workers).resolve_program()


class MossPosAddOperator(ArtcOperator):
    """``y + pos`` of the stride-2 stem, in the C stripe layout (one dispatch: the core's second input channel).

    The core's two input DMA channels are why this is not the stem's third argument: A and W take both (module docstring).
    Streams: y and pos enter their memtiles per column and are split into the column's 4 cores' stripes; the results
    join per column for one drain.
    """

    p = kernel_prefix("mpos")
    STACK = 1024

    def __init__(self, *, M: int = M_TILE, N: int = N_TILE):
        assert M == M_TILE and N == N_TILE, "the stem's tile"
        self.M, self.N = M, N
        super().__init__()

    c_elems = TOK * NCOL

    def get_kernel_src(self) -> str:
        return TEMPLATE.render(p=self.p, nsub=self.c_elems // 64, c_elems=self.c_elems, act=ACT,
                               act_t=str(kmath.act(kmath.TABLES[ACT], table_arg=True, walk=(2, 128))), add_only=True)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.M * self.N,), bfloat16), ArgSpec("in", (self.M * self.N,), bfloat16),
                ArgSpec("out", (self.M * self.N,), bfloat16)]

    def get_mlir_artifact(self) -> Module:
        bf = np.dtype[bfloat16]
        ci, mi, ncol = self.c_elems, self.M, self.N
        OBJ_ty = np.ndarray[(ci,), bf]
        COL_ty = np.ndarray[(ROWS * ci,), bf]  # (a column's 4 stripes: [band][2 row blocks][8*NCOL])
        IMG = np.ndarray[(mi * ncol,), bf]
        p = self.p
        k_add = Kernel(f"{p}_add", "kernel.o", [COL_ty, COL_ty, OBJ_ty, np.int32])
        # y and pos: per column one memtile object, broadcast to the column's 4 cores (each reads its band); the sums
        # join per column for one drain (a split would need 4 memtile output channels per stream)
        l3 = {name: [ObjectFifo(COL_ty, name=f"ms_{name}{c}", depth=1) for c in range(COLS)] for name in ("y", "pos")}
        l1 = {name: [l3[name][c].cons().forward(obj_type=COL_ty, name=f"ms_{name}1_{c}", depth=1, tile=Tile(c, 1))
                     for c in range(COLS)] for name in ("y", "pos")}
        o_l3 = [ObjectFifo(COL_ty, name=f"ms_sum{c}", depth=1) for c in range(COLS)]
        o_l1 = [[] for _ in range(ROWS)]
        for c in range(COLS):
            subs = o_l3[c].prod().join([ci * r for r in range(ROWS)], obj_types=[OBJ_ty] * ROWS,
                                       names=[f"ms_sum1_{c}_{r}" for r in range(ROWS)], depths=[1] * ROWS,
                                       tile=Tile(c, 1))
            for r in range(ROWS):
                o_l1[r].append(subs[r].prod())

        def core(y_h, p_h, o_h, add, band):
            """The core's band of the column's two objects, added into the output stripe."""
            for _ in forever_():
                out = o_h.acquire(1)
                add(y_h.acquire(1), p_h.acquire(1), out, band)
                y_h.release(1)
                p_h.release(1)
                o_h.release(1)

        def seq(Y, POS, OUT, yc, pc, oc):
            tg = TaskGroup()
            for c in range(COLS):
                yc[c].fill(Y, column_tap(mi, ncol, c), group=tg, wait=False)
                pc[c].fill(POS, column_tap(mi, ncol, c), group=tg, wait=False)
            tg.finish()
            tgd = TaskGroup()
            for c in range(COLS):
                oc[c].drain(OUT, column_tap(mi, ncol, c), group=tgd, wait=True)
            tgd.finish()

        workers = []
        for r in range(ROWS):
            for c in range(COLS):
                workers.append(Worker(core, [l1["y"][c].cons(), l1["pos"][c].cons(), o_l1[r][c], k_add, r],
                                      tile=Tile(c, r + 2), stack_size=self.STACK,
                                      dynamic_objfifo_lowering=True, while_true=False))
        y_prods = [l3["y"][c].prod(tile=Tile(c, 0)) for c in range(COLS)]
        p_prods = [l3["pos"][c].prod(tile=Tile(c, 0)) for c in range(COLS)]
        o_conses = [o_l3[c].cons(tile=Tile(c, 0)) for c in range(COLS)]
        return Program(NPU2(), Runtime(seq, [IMG, IMG, IMG, y_prods, p_prods, o_conses]),
                       workers=workers).resolve_program()
