"""The stem operator's host side: the im2col / panel packing contracts and the A2 gate (relative L2 <= 2 %), walked in
the MAC loop's pop order instead of on the NPU (``moss_stem.py``: the operator this stage builds and compiles; nothing
dispatches).

The reference is the fp32 convolution itself (``im2col`` + the weights' matmul + the bias, what the host computes); the
device side is the same arithmetic over the bfp16ebs8 panels this operator packs, so a failure here is a packer or
layout failure, not a kernel one.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.operators.kmath.acttable import gelu_erf64
from artlite.vision import layout as L

from .moss_stem import (
    K_TILE,
    M_TILE,
    N_TILE,
    NCOL,
    TOK,
    MossPosAddOperator,
    MossStemOperator,
    a_image,
    bias_channel,
    fold_weights,
    from_c,
    im2col,
    mac_walk,
    to_c,
    to_c_f32,
    w_image,
)

MEL, CH = 80, 1024  # the front-end's mel bins and the encoder's width


def conv_ref(x: np.ndarray, w: np.ndarray, b: np.ndarray, *, stride: int) -> np.ndarray:
    """The fp32 3-tap conv the operator's GEMM reproduces: (T, C_in) -> GELU(cols @ W^T + b)."""
    cols = im2col(x, stride=stride)
    wm = w.transpose(0, 2, 1).reshape(w.shape[0], -1)
    return np.asarray(gelu_erf64(cols @ wm.T + b), np.float32)


@pytest.mark.parametrize("stride,channels,rows", [(1, MEL, 3000), (2, CH, 3000)])
def test_im2col(stride: int, channels: int, rows: int) -> None:
    """The two stems' K (3 x 80 and 3 x 1024 channels) and the rows they produce (3000 / 1500)."""
    x = np.random.default_rng(7).standard_normal((rows, channels)).astype(np.float32)
    cols = im2col(x, stride=stride)
    assert cols.shape == (rows // stride, 3 * channels)
    assert np.array_equal(cols[0, :channels], np.zeros(channels, np.float32))  # (the zero pad above row 0)
    assert np.array_equal(cols[0, channels:2 * channels], x[0])
    if stride == 2:
        assert np.array_equal(cols[1, channels:2 * channels], x[2])


@pytest.mark.parametrize("channels,stride,k", [(MEL, 1, 256), (CH, 2, 3136)])
def test_images_are_the_panels_reordered(channels: int, stride: int, k: int) -> None:
    """The A and W images hold exactly the panel bytes, only re-ordered into the MAC's pop order."""
    rng = np.random.default_rng(11)
    x = bias_channel(im2col(rng.standard_normal((M_TILE, channels)).astype(np.float32), stride=stride), k)
    w = fold_weights(rng.standard_normal((N_TILE, channels, 3)).astype(np.float32),
                     rng.standard_normal(N_TILE).astype(np.float32), k=k)
    for img, mat in ((a_image(x), x), (w_image(w), w)):
        assert img.size == L.panels(mat).size
        assert np.array_equal(np.sort(img.view(np.uint8)), np.sort(L.panels(mat).view(np.uint8)))


def quantize(x: np.ndarray) -> np.ndarray:
    """bfp16ebs8 round trip through the panel layout (the operands the device's MAC reads)."""
    m, c = x.shape
    return L.from_panels(L.panels(np.asarray(x, np.float32)), m, c)


def test_mac_walk_is_the_matmul() -> None:
    """Walking the packing in the MAC loop's order gives the quantized A @ W^T (a host model of ``MAC<16, 160>``)."""
    m, n, k = M_TILE, N_TILE, 256
    x = np.random.default_rng(21).standard_normal((m, k)).astype(np.float32)
    w = np.random.default_rng(22).standard_normal((n, k)).astype(np.float32)
    got = mac_walk(a_image(x), w_image(w), m, n, k)
    want = to_c_f32(quantize(x) @ quantize(w).T)
    assert got.shape == (m * n,)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6 * np.abs(want).max())


@pytest.mark.parametrize("channels,stride,k,t", [(MEL, 1, 256, 64), (CH, 2, 3136, 128)])
def test_gate_a2(channels: int, stride: int, k: int, t: int) -> None:
    """A2: the operator's arithmetic (bfp16 panels, the folded bias, GELU, bf16 store) within 2 % of the fp32 conv."""
    rng = np.random.default_rng(31)
    x = (rng.standard_normal((t, channels)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((N_TILE, channels, 3)) * 0.05).astype(np.float32)
    b = (rng.standard_normal(N_TILE) * 0.02).astype(np.float32)
    ref = conv_ref(x, w, b, stride=stride)                               # (t / stride, N)
    assert ref.shape[0] == M_TILE, "the gate runs on one token tile (the operator's M)"
    a = bias_channel(im2col(x, stride=stride), k)
    wm = fold_weights(w, b, k=k)
    stripe = np.asarray(gelu_erf64(mac_walk(a_image(a), w_image(wm), M_TILE, N_TILE, k)), np.float32)
    got = from_c(stripe.astype(bfloat16), M_TILE)                         # the epilogue's bf16 store (same layout)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel <= 0.02, f"{rel:.4f}"
    assert np.abs(got - ref).max() <= 0.02 * np.abs(ref).max()


def test_gate_a2_residual() -> None:
    """The stride-2 stem with the position table: the residual is one bf16 rounding on top, still inside 2 %."""
    rng = np.random.default_rng(41)
    channels, k, t = CH, 3136, 128
    x = (rng.standard_normal((t, channels)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((N_TILE, channels, 3)) * 0.05).astype(np.float32)
    b = (rng.standard_normal(N_TILE) * 0.02).astype(np.float32)
    pos = (rng.standard_normal((M_TILE, N_TILE)) * 0.5).astype(np.float32)
    y = np.asarray(gelu_erf64(mac_walk(a_image(bias_channel(im2col(x, stride=2), k)),
                                       w_image(fold_weights(w, b, k=k)), M_TILE, N_TILE, k)), np.float32)
    yq = from_c(y.astype(bfloat16), M_TILE)                             # GELU -> bf16
    out = from_c(to_c(yq + pos), M_TILE)                                # + the position table -> bf16
    ref = conv_ref(x, w, b, stride=2) + pos
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel <= 0.02, f"{rel:.4f}"


@pytest.mark.parametrize("k", [256, 3136])
def test_render(k: int) -> None:
    """The stem renders a 32-core design with the streams (A per row and W per column broadcast, the C join) and its
    argument spec matches the images the packers build."""
    op = MossStemOperator(K=k)
    shapes = [a.shape for a in op.get_arg_spec()]
    assert shapes == [(M_TILE * k * 9 // 16,), (N_TILE * k * 9 // 16,), (M_TILE * N_TILE,)]
    text = op.mlir_text()
    assert text.count("aie.core") == 32, "8 columns x 4 rows"
    for name in ("ms_a1_0", "ms_w1_0", "ms_o0", "ms_a0", "ms_w0"):
        assert name in text, name
    src = op.get_kernel_src()
    for name in ("_zero", "_mac", "_epi", "mstem_mac_t"):
        assert name in src, name


def test_pos_add_render() -> None:
    """The position add renders its own 32-core design (y and pos broadcast per column, the sums joined)."""
    op = MossPosAddOperator()
    assert [a.shape for a in op.get_arg_spec()] == [(M_TILE * N_TILE,)] * 3
    text = op.mlir_text()
    assert text.count("aie.core") == 32
    for name in ("ms_y0", "ms_pos0", "ms_sum0", "ms_y1_0", "ms_pos1_0"):
        assert name in text, name
    assert "_add" in op.get_kernel_src()


def test_k_tiles_fit_a_bd() -> None:
    """One BD iterates a channel's k-tiles: k / K_TILE <= the shim's iteration bound (48 for the stride-2 stem)."""
    for k in (256, 3136):
        assert MossStemOperator(K=k).nkt == k // K_TILE
        assert MossStemOperator(K=256).nkt <= 64


def test_c_layout() -> None:
    """``to_c`` / ``from_c``: an exact inverse up to the bf16 store, and the two runs of a core's stripe are the drain's."""
    rng = np.random.default_rng(51)
    y = rng.standard_normal((M_TILE, N_TILE)).astype(np.float32)
    assert np.array_equal(from_c(to_c(y), M_TILE), np.asarray(y.astype(bfloat16), np.float32))
    c = to_c(y)
    for r in range(M_TILE // TOK):
        for cidx in range(8):
            for h in range(2):  # (a run is one 8-token block's 20 sub-blocks of [8 rows][8 channels])
                off = (2 * r + h) * 8 * N_TILE + 20 * cidx * 64
                run = np.asarray(c[off:off + 8 * NCOL], np.float32)
                want = y[8 * (2 * r + h):8 * (2 * r + h) + 8, NCOL * cidx:NCOL * (cidx + 1)]
                assert np.allclose(run, want.reshape(8, 20, 8).transpose(1, 0, 2).reshape(-1), atol=1e-2, rtol=0.02)
