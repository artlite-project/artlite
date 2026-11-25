"""Inkling's residual-stream operators in the prefill (``prefill.ink_stream``) on random states: the depth mix against its
fp64 model (its bf16 output rounding)."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.operators import FusedOperator
from artlite.prefill.ink_stream import InkMixOperator

from ..numerics import rel

H = 2304
MIX = [(256, 1), (256, 2), (128, 5), (64, 3)]


def _mix(states: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(n, T, H) states, (H,) query -> (T, H) the depth mix (fp64)."""
    s = states.astype(np.float64)
    r = np.sqrt((s * s).mean(-1, keepdims=True) + eps)
    sn = s / r
    sc = np.einsum("nth,h->tn", sn, q.astype(np.float64)) / np.sqrt(H)
    w = np.exp(sc - sc.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return np.einsum("tn,nth->th", w, sn)


@pytest.mark.parametrize(("T", "n"), MIX, ids=[f"T{a}-n{b}" for a, b in MIX])
def test_ink_mix(T: int, n: int) -> None:
    op = InkMixOperator(T=T, n=n)
    names = [f"S{i}" for i in range(n)]
    fused = FusedOperator(name=f"ink_mix_test_{n}", input_args=[*names, "Q"], output_args=["X"],
                          runlist=[(op, *names, "Q", "X")]).compile()
    gen = torch.Generator().manual_seed(T + n)
    # (states of different scales, as block states and a residual: the mix's norms)
    st = np.stack([torch.randn(T, H, generator=gen).numpy() * (0.5 + 2 * i) for i in range(n)]).astype(np.float32)
    q = (torch.randn(H, generator=gen) * 0.3).numpy().astype(np.float32)
    for i in range(n):
        fused.get_buffer(f"S{i}").view(np.float32)[:] = st[i].reshape(-1)
    fused.get_buffer("Q").view(np.float32)[:] = q
    fused.sync_inputs()
    fused()
    fused.sync_outputs()
    got = fused.get_buffer("X").astype(np.float32).reshape(T, H)
    want = _mix(st, q)
    floor = rel(want.astype(np.float32).astype(bfloat16).astype(np.float64), want)
    err = rel(got, want)
    print(f"mix {err:.2e} (bf16 rounding {floor:.2e})")
    assert err < 1.1 * floor


# (chunk, valid tokens, a block's restart, the history's first row in the window, the input rows' pitch and column slice)
RESID = [(256, 256, False, 2, 2304, 288), (128, 101, True, 0, 2560, 320), (64, 3, False, 4, 2304, 288),
         (2048, 2000, True, 1, 2560, 320), (1024, 1000, True, 1, 2560, 320), (1280, 1200, True, 1, 2560, 320)]  # (2048: the fills in batches, the rows as [T / 8][8])


@pytest.mark.parametrize(("T", "n", "block", "a", "yp", "yc"), RESID,
                         ids=[f"T{t}-n{n}-{'block' if b else 'add'}-a{a}" for t, n, b, a, _, _ in RESID])
def test_ink_resid(T: int, n: int, block: bool, a: int, yp: int, yc: int) -> None:
    from artlite.operators.design import COLS
    from artlite.prefill.ink_stream import InkResidOperator

    op = InkResidOperator(T=T, block=block, y_pitch=yp, y_col=yc)
    C = H // COLS
    ins = ["Y"] + ([] if block else ["CUR"]) + ["TAPS"]
    fused = FusedOperator(name=f"ink_resid_test_{int(block)}", input_args=ins, output_args=["HIST", "OUT"],
                          runlist=[(op, *ins, "HIST", "OUT" if block else "CUR")]
                          if not block else [(op, *ins, "HIST", "OUT")]).compile()
    gen = torch.Generator().manual_seed(T + n + a)
    y = (torch.randn(T, H, generator=gen) * 3).bfloat16().double().numpy()  # (bf16 values: the GEMM's output)
    taps = (torch.randn(4, H, generator=gen) * 2).bfloat16().double().numpy()
    cur = torch.randn(T, H, generator=gen).numpy().astype(np.float32)
    hist = torch.randn(COLS, 8, C, generator=gen).numpy().astype(np.float32)
    Y = np.zeros((T, yp), np.float32)
    for c in range(COLS):
        Y[:, c * yc:c * yc + C] = y[:, c * C:(c + 1) * C]
    fused.get_buffer("Y")[:] = Y.astype(bfloat16).reshape(-1)
    tp = np.zeros((COLS, 4, C), np.float32)
    for c in range(COLS):
        tp[c] = taps[:, c * C:(c + 1) * C]
    fused.get_buffer("TAPS")[:] = tp.astype(bfloat16).reshape(-1)
    if not block:
        fused.get_buffer("CUR").view(np.float32)[:] = cur.reshape(-1)
    fused.get_buffer("HIST").view(np.float32)[:] = hist.reshape(-1)
    fused.sync_inputs()
    fused.set_params(op.params(n, a))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()
    got = fused.get_buffer("OUT" if block else "CUR").view(np.float32).reshape(T, H).astype(np.float64)
    hp = np.concatenate([hist[c, a:a + 3] for c in range(COLS)], -1).astype(np.float64)  # (3, H)
    xp = np.concatenate([hp, y], 0)
    conv = (y + sum(taps[k] * xp[k:k + T] for k in range(4))) * 5 ** -0.5
    want = conv if block else cur + conv
    assert rel(got, want) < 1e-5, rel(got, want)  # (fp32 products as bf16 hi + lo: ~2^-16)
    win = fused.get_buffer("HIST").view(np.float32).reshape(COLS, 8, C)
    wrows = np.concatenate([win[c, :3] for c in range(COLS)], -1)
    assert np.array_equal(wrows, xp[n:n + 3].astype(np.float32)), "the window: the last 3 valid tokens' inputs"


@pytest.mark.parametrize("T", [256, 64])
def test_ink_gate(T: int) -> None:
    from artlite.operators import bfp16
    from artlite.operators.design import COLS
    from artlite.prefill.ink_stream import InkGateOperator

    op = InkGateOperator(T=T, lq_pitch=896, gate_off=384)
    fused = FusedOperator(name="ink_gate_test", input_args=["A", "G"], output_args=["O"],
                          runlist=[(op, "A", "G", "O")]).compile()
    gen = torch.Generator().manual_seed(T)
    o = torch.randn(T, COLS * 384, generator=gen).numpy().astype(np.float32)
    g = (torch.randn(T, COLS * 384, generator=gen) * 3).bfloat16().double().numpy()
    fused.get_buffer("A")[:] = bfp16.to_panels(o)
    G = np.zeros((T, COLS * 896), np.float32)
    for c in range(COLS):
        G[:, c * 896 + 384:c * 896 + 768] = g[:, 384 * c:384 * (c + 1)]
    fused.get_buffer("G")[:] = G.astype(bfloat16).reshape(-1)
    fused.sync_inputs()
    fused()
    fused.sync_outputs()
    got = bfp16.from_panels(fused.get_buffer("O")[:op.n_a], T, COLS * 384).astype(np.float64)
    want = bfp16.values(o).astype(np.float64) / (1 + np.exp(-g))
    floor = rel(bfp16.values(want.astype(np.float32)).astype(np.float64), want)
    err = rel(got, want)
    print(f"gate {err:.2e} (bfp16 rounding {floor:.2e})")
    assert err < 1.5 * floor


@pytest.mark.parametrize("T", [256, 64])
def test_ink_blend(T: int) -> None:
    from artlite.operators.design import COLS
    from artlite.prefill.ink_stream import InkBlendOperator

    dp, ws = 64, 2 * 288 * 64 + 1024  # (the down halves' layout pitch; a column's ctl objects and a gap)
    op = InkBlendOperator(T=T, d_pitch=dp, w_stride=ws)
    fused = FusedOperator(name="ink_blend_test", input_args=["GA", "GB", "CS", "MS", "W"], output_args=["OUT"],
                          runlist=[(op, "GA", "GB", "CS", "MS", "W", "OUT")]).compile()
    gen = torch.Generator().manual_seed(T + 1)
    ga = (torch.randn(T, 64, generator=gen) * 0.5).numpy().astype(np.float32)
    gb = (torch.randn(T, 64, generator=gen) * 0.5).numpy().astype(np.float32)
    cs = torch.randn(T, H, generator=gen).bfloat16().double().numpy()
    ms = torch.randn(T, H, generator=gen).bfloat16().double().numpy()
    F = (torch.randn(H, 64, generator=gen) * 0.3).bfloat16().double().numpy()
    G = (torch.randn(H, 64, generator=gen) * 0.3).bfloat16().double().numpy()
    for name, g in (("GA", ga), ("GB", gb)):  # (column c's 8 outputs at c pitch, fp32 as bf16 pairs)
        D = np.zeros((T, COLS * dp), np.uint16)
        for c in range(COLS):
            D[:, c * dp:c * dp + 16] = g[:, 8 * c:8 * c + 8].copy().view(np.uint16)
        fused.get_buffer(name).view(np.uint16)[:] = D.reshape(-1)
    fused.get_buffer("CS")[:] = cs.astype(bfloat16).reshape(-1)
    fused.get_buffer("MS")[:] = ms.astype(bfloat16).reshape(-1)
    W = np.zeros((COLS, ws), np.float32)
    for c in range(COLS):
        W[c, :288 * 64] = F[288 * c:288 * (c + 1)].reshape(-1)
        W[c, 288 * 64:2 * 288 * 64] = G[288 * c:288 * (c + 1)].reshape(-1)
    fused.get_buffer("W")[:] = W.astype(bfloat16).reshape(-1)
    fused.sync_inputs()
    fused()
    fused.sync_outputs()
    got = fused.get_buffer("OUT").astype(np.float64).reshape(T, H)
    g = (ga + gb).astype(np.float64)
    want = (1 - np.tanh(g @ F.T)) * cs + np.tanh(g @ G.T) * ms
    floor = rel(want.astype(np.float32).astype(bfloat16).astype(np.float64), want)
    err = rel(got, want)
    print(f"blend {err:.2e} (bf16 rounding {floor:.2e})")
    assert err < 1.2 * floor
