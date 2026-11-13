"""The prefill's short convolution (models/lfm/conv_prefill.py) on the NPU against its numpy model, bit for bit: random
B, C, x rows (as the in GEMM leaves them, bf16), a layer's taps of the GGUF file and a random history in the decode's
conv regions; y as the out GEMM's bfp16 panels, the history the decode continues from (the last 6 rows of B x of the
chunk's valid tokens, at the regions' rows 0 .. 5)."""

import numpy as np
import pytest
import torch

from artlite.decode.protocol import XPAY
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.fused import FusedOperator
from models.lfm import conv as CV
from models.lfm.conv_prefill import ROW, ConvPrefillOperator

from ..common import to_numpy
from ..numerics import bf16_round
from . import ref

PITCH = 768


def conv_ref(b: np.ndarray, c: np.ndarray, x: np.ndarray, hist: np.ndarray, taps: np.ndarray, n: int
             ) -> tuple[np.ndarray, np.ndarray]:
    """The kernel's arithmetic (fp32 steps): b, c, x (T, ch) bf16 values, hist (6, ch), taps (7, 2, ch) bf16 hi / lo ->
    y (T, ch) fp32 (before the panels' rounding), the last 6 rows of B x of the first n tokens."""
    bx = bf16_round(b * x)
    win = np.concatenate([hist, bx]).astype(np.float32)
    T = b.shape[0]
    acc = np.zeros_like(b, np.float32)
    for k in range(7):
        for h in range(2):
            acc = (acc + taps[k, h] * win[k:k + T]).astype(np.float32)
    ah = bf16_round(acc)
    al = bf16_round(acc - ah)
    y = (ah * c).astype(np.float32)
    y = (y + al * c).astype(np.float32)
    return y, win[n:n + 6]


@pytest.mark.parametrize(("T", "n"), [(64, 64), (256, 200), (1024, 1023)])
def test_conv_prefill(T: int, n: int) -> None:
    g = ref.gguf()
    L = 3
    w = np.asarray(g.tensor(f"blk.{L}.shortconv.conv.weight").dequantize(), np.float32).reshape(COLS * CV.CH, CV.WIDTH)
    taps = CV.tap_objects(w)  # (COLS, 7 TAP)
    op = ConvPrefillOperator(T=T, pitch=PITCH)
    fused = FusedOperator(name=f"lfcv_test{T}", runlist=[(op, "G", "H", "A")], input_args=["G", "H"],
                          output_args=["A", "H"]).compile()
    gen = torch.Generator().manual_seed(T)
    G = (torch.randn(T, COLS * PITCH, generator=gen) * 0.5).bfloat16()
    hist = (torch.randn(COLS, 6, CV.CH, generator=gen) * 0.3).bfloat16()
    H = np.zeros((COLS, CV.region_bytes()), np.uint8)
    for c in range(COLS):
        H[c, CV.payload(0):CV.payload(0) + XPAY] = taps[c, :XPAY]
        H[c, CV.payload(1):CV.payload(1) + 3 * CV.TAP] = taps[c, XPAY:]
        H[c, CV.payload(2):CV.payload(2) + 6 * CV.ROW] = hist[c].view(torch.uint8).numpy().reshape(-1)
    fused.get_buffer("G")[:] = to_numpy(G.reshape(-1))
    fused.get_buffer("H").view(np.uint8)[:] = H.reshape(-1)
    fused.sync_inputs()
    fused.set_params(op.params(n))
    fused()
    fused.sync_outputs()
    a = fused.get_buffer("A").view(np.uint8)
    y = bfp16.block_values(a.reshape(T // 8, COLS * CV.CH // 8, 8, 9)).transpose(0, 2, 1, 3).reshape(T, -1)
    h_out = fused.get_buffer("H").view(np.uint8).reshape(COLS, -1)
    Gf = G.float().numpy()
    for c in range(COLS):
        rows = Gf[:, c * PITCH:c * PITCH + ROW]
        wv = torch.from_numpy(taps[c].copy()).view(torch.bfloat16).float().numpy().reshape(7, 2, CV.CH)
        yr, hr = conv_ref(rows[:, :CV.CH], rows[:, CV.CH:2 * CV.CH], rows[:, 2 * CV.CH:], hist[c].float().numpy(), wv, n)
        want = bfp16.values(yr.reshape(T, CV.CH // 8, 8)).reshape(T, CV.CH)
        got = y[:, c * CV.CH:(c + 1) * CV.CH]
        assert np.array_equal(got, want), (c, np.abs(got - want).max())
        hg = torch.from_numpy(h_out[c, CV.payload(2):CV.payload(2) + 6 * CV.ROW].copy()).view(torch.bfloat16).float()
        assert np.array_equal(hg.numpy().reshape(6, CV.CH), hr), c
