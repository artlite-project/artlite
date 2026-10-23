"""The prefills' norms (operators/prefill/norm.py): PrefillNormOperator ((x += delta,) RMSNorm on a bf16 residual:
Qwen's, Llama's) vs torch, ResidualNormOperator (an fp32 residual: Gemma's modes) vs its fp64 model."""

import numpy as np
import pytest
import torch

import artlite
from artlite.operators import PrefillNormOperator, bfp16
from artlite.operators.design import COLS
from artlite.operators.prefill.norm import ResidualNormOperator, residual_params

_from_panels = bfp16.from_panels


CASES = [(T, add, E, 0, 8) for E in (2048, 3072) for add in (True, False) for T in (64, 512)]
CASES += [(64, True, 2048, 264, 8), (512, True, 3072, 396, 8)]  # delta of a Q6_K down projection (11-row tiles)
CASES += [(64, True, 3072, 396, 2), (64, False, 3072, 0, 1), (128, True, 2048, 0, 2)]  # fewer cores
CASES += [(64, True, 4096, 528, 8), (512, True, 4096, 528, 8), (512, False, 4096, 0, 8), (64, True, 4096, 528, 2)]  # 8B
CASES += [(2048, True, 3072, 396, 8), (1024, True, 4096, 528, 8)]  # (Llama's norm blocks)


@pytest.mark.parametrize(("T", "add", "E", "d_pitch", "cols"), CASES,
                         ids=[f"T{T}-{'add' if a else 'noadd'}-E{E}{f'-dp{p}' if p else ''}{f'-c{c}' if c < 8 else ''}"
                              for T, a, E, p, c in CASES])
def test_prefill_norm(T: int, add: bool, E: int, d_pitch: int, cols: int):
    """d_pitch: delta's column blocks of E / 8 values at this pitch (padding never read); cols: cores."""
    g = torch.Generator().manual_seed(T + add)
    x = (torch.randn(T, E, generator=g) * 2).bfloat16()
    d = torch.randn(T, E, generator=g).bfloat16()
    w = (torch.rand(E, generator=g) + 0.5).bfloat16()
    op = PrefillNormOperator(T=T, add=add, E=E, d_pitch=d_pitch, cols=cols).compile()
    dev = op.device
    xb, wb = artlite.from_torch(x.clone().flatten(), dev), artlite.from_torch(w.clone(), dev)
    xnb = artlite.empty((T * E * 9 // 16,), dtype=torch.bfloat16, device=dev)
    dp = d_pitch or E // 8
    dpad = torch.full((T, 8, dp), 1e30).bfloat16()  # (padding: garbage)
    dpad[:, :, :E // 8] = d.view(T, 8, E // 8)
    op.dispatch(*((xb, artlite.from_torch(dpad.flatten(), dev), wb, xb, xnb) if add else (xb, wb, xnb)))
    xnb.sync_from_device()
    xs = (x.float() + d.float()).bfloat16().float() if add else x.float()
    ref = xs * torch.rsqrt((xs ** 2).mean(-1, keepdim=True) + 1e-5) * w.float()
    xn = _from_panels(xnb.to_torch(), T, E)
    rel = ((xn - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"xn rel-rms {rel}"  # bfp16ebs8 rounding: ~7e-3
    if add:
        xb.sync_from_device()
        assert torch.equal(xb.to_torch().view(T, E), (x.float() + d.float()).bfloat16())


@pytest.mark.parametrize(("T", "d_pitch"), [(64, 0), (256, 264)], ids=["T64", "T256-dp264"])
def test_prefill_norm_lo(T: int, d_pitch: int) -> None:
    """xn as hi + lo panels (``lo``: an MoE router's x, Qwen3.6-35B-A3B's hidden 2048): per token group [hi | lo]
    panels; hi + lo within 4e-5 of x / rms(x) w in fp64 (x the bf16 sum; 2.6e-5: lo's rounding, ~2^-16 of each block's
    maximum; the per-token scale 1 / rms 2.4e-6), hi the single panels' rounding."""
    E = 2048
    g = torch.Generator().manual_seed(T)
    x = (torch.randn(T, E, generator=g) * 2).bfloat16()
    d = torch.randn(T, E, generator=g).bfloat16()
    w = (torch.rand(E, generator=g) + 0.5).bfloat16()
    op = PrefillNormOperator(T=T, E=E, d_pitch=d_pitch, lo=True).compile()
    dev = op.device
    xb, wb = artlite.from_torch(x.clone().flatten(), dev), artlite.from_torch(w.clone(), dev)
    xnb = artlite.empty((2 * T * E * 9 // 16,), dtype=torch.bfloat16, device=dev)
    dp = d_pitch or E // 8
    dpad = torch.full((T, 8, dp), 1e30).bfloat16()
    dpad[:, :, :E // 8] = d.view(T, 8, E // 8)
    op.dispatch(xb, artlite.from_torch(dpad.flatten(), dev), wb, xb, xnb)
    xnb.sync_from_device()
    xs = (x.float() + d.float()).bfloat16().double()
    ref = xs * torch.rsqrt((xs ** 2).mean(-1, keepdim=True) + 1e-5) * w.double()
    both = _from_panels(xnb.to_torch(), T, 2 * E).double()  # (per token [hi | lo])
    hi, lo = both[:, :E], both[:, E:]
    assert ((hi - ref).norm() / ref.norm()).item() < 1e-2
    rel = ((hi + lo - ref).norm() / ref.norm()).item()
    scale = ((hi + lo) * ref).sum(1) / (ref * ref).sum(1) - 1  # (per token)
    print(f"hi + lo rel-rms {rel:.2e}; per-token scale rms {scale.pow(2).mean().sqrt():.2e}")
    assert rel < 4e-5, rel
    xb.sync_from_device()
    assert torch.equal(xb.to_torch().view(T, E), xs.float().bfloat16())


@pytest.mark.parametrize("T", [64, 256])
def test_prefill_norm_pre_pitch(T: int) -> None:
    """Two norms into one A of 2 E columns (``out_pitch``, ``out_col``: an MTP drafter's eh_proj input [enorm(emb) |
    hnorm(h)]): a plain one into columns 0 .., ``pre`` into E ..: (x w1) w2 / rms(x w1), x w1 rounded to bf16 (h =
    the model's output norm of its residual, then the drafter's hnorm); each within the bfp16 panels' rounding."""
    E = 2048
    g = torch.Generator().manual_seed(T)
    e, x = (torch.randn(T, E, generator=g) * 2).bfloat16(), (torch.randn(T, E, generator=g) * 3).bfloat16()
    we, w1, w2 = ((torch.rand(E, generator=g) + 0.5).bfloat16() for _ in range(3))
    a = PrefillNormOperator(T=T, add=False, E=E, out_pitch=2 * E, out_col=0).compile()
    b = PrefillNormOperator(T=T, add=False, E=E, pre=True, out_pitch=2 * E, out_col=E).compile(device=a.device)
    dev = a.device
    xn = artlite.empty((T * 2 * E * 9 // 16,), dtype=torch.bfloat16, device=dev)
    a.dispatch(artlite.from_torch(e.clone().flatten(), dev), artlite.from_torch(we.clone(), dev), xn)
    b.dispatch(artlite.from_torch(x.clone().flatten(), dev), artlite.from_torch(torch.cat([w1, w2]), dev), xn)
    xn.sync_from_device()
    both = _from_panels(xn.to_torch(), T, 2 * E).double()
    ef = e.double()
    ref_e = ef * torch.rsqrt((ef ** 2).mean(-1, keepdim=True) + 1e-5) * we.double()
    xw = (x.float() * w1.float()).bfloat16().double()
    ref_h = xw * torch.rsqrt((xw ** 2).mean(-1, keepdim=True) + 1e-5) * w2.double()
    for got, ref in ((both[:, :E], ref_e), (both[:, E:], ref_h)):
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, rel


# (T, E, d_pitch, cols, rows, deltas): Llama's and qwen35's projections in K parts
DELTAS = [(64, 5120, 660, 4, 96, 2), (256, 4096, 528, 8, 256, 2), (64, 4096, 0, 2, 64, 2),
          (1024, 3072, 396, 8, 1024, 2), (2048, 4096, 528, 8, 2048, 2), (512, 2048, 264, 8, 512, 2),
          (256, 5120, 660, 8, 288, 4), (256, 5120, 660, 8, 288, 6), (256, 4096, 528, 8, 288, 3), (64, 5120, 0, 2, 64, 5)]


@pytest.mark.parametrize(("T", "E", "d_pitch", "cols", "rows", "n"), DELTAS,
                         ids=[f"T{a}-E{b}-dp{c}-c{d}-r{e}-n{f}" for a, b, c, d, e, f in DELTAS])
def test_prefill_norm_deltas(T: int, E: int, d_pitch: int, cols: int, rows: int, n: int):
    """x += d1 + .. + dn (a projection in n K parts, each ``rows`` rows after the previous: a GEMM's padded C, a norm
    block of Llama's; two deltas at E 2048: the replay design, more: the hold design): the sum rounded once to bf16,
    bit for bit."""
    g = torch.Generator().manual_seed(T + E + n)
    x = (torch.randn(T, E, generator=g) * 2).bfloat16()
    d = torch.randn(n, T, E, generator=g).bfloat16()
    w = (torch.rand(E, generator=g) + 0.5).bfloat16()
    dp = d_pitch or E // 8
    op = PrefillNormOperator(T=T, E=E, d_pitch=d_pitch, cols=cols, deltas=n, d_stride=rows * 8 * dp).compile()
    dev = op.device
    dpad = torch.full((n, rows, 8, dp), 1e30).bfloat16()  # ([d1 | d2 | ..]; padding: garbage)
    dpad[:, :T, :, :E // 8] = d.view(n, T, 8, E // 8)
    xb = artlite.from_torch(x.clone().flatten(), dev)
    xnb = artlite.empty((T * E * 9 // 16,), dtype=torch.bfloat16, device=dev)
    op.dispatch(xb, artlite.from_torch(dpad.flatten(), dev), artlite.from_torch(w.clone(), dev), xb, xnb)
    xb.sync_from_device()
    xnb.sync_from_device()
    xs = (x.float() + d.float().sum(0)).bfloat16()
    assert torch.equal(xb.to_torch().view(T, E), xs)
    ref = xs.float() * torch.rsqrt((xs.float() ** 2).mean(-1, keepdim=True) + 1e-5) * w.float()
    xn = _from_panels(xnb.to_torch(), T, E)
    assert ((xn - ref).norm() / ref.norm()).item() < 1e-2


# (mode, E, deltas, d_pitch, scale): Gemma E4B's modes
def _residual_reference(x: torch.Tensor, d: torch.Tensor | None, w_post: torch.Tensor | None, w1: torch.Tensor | None,
                        scale: float, eps: float, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(x', xn values before the bfp16 rounding) in fp64 of fp32 x (T, E), d (T, E) the (summed) delta."""
    x = x.double()
    if mode != "in":
        dd = d.double()
        iv = torch.rsqrt(dd.bfloat16().double().pow(2).mean(-1, keepdim=True) + eps)
        x = (x + dd * iv * w_post.bfloat16().double()) * scale
    if mode == "plain":
        return x, x
    xb = x.float().bfloat16().double()
    iv = torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + eps)
    return x, xb * w1.bfloat16().double() * iv


RESIDUAL = [("in", 2560, 1, 320, False), ("post", 2560, 1, 320, False), ("post", 2560, 1, 384, True),
            ("plain", 2560, 2, 320, False), ("plain", 2560, 2, 384, False), ("plain", 2560, 4, 320, False)]


@pytest.mark.parametrize(("mode", "E", "deltas", "d_pitch", "scale"), RESIDUAL,
                         ids=[f"{m}-E{e}-d{d}-p{p}{'-s' if s else ''}" for m, e, d, p, s in RESIDUAL])
def test_residual_norm(mode: str, E: int, deltas: int, d_pitch: int, scale: bool) -> None:
    """ResidualNormOperator against its fp64 model: x' within ~2^-16 (the fp32 update), the panels within 1.1 x the
    error of the model's values rounded to bfp16. Several deltas: a down projection's K parts, ``d_stride`` apart
    (padded)."""
    T, eps = 256, 1e-6
    stride = T * COLS * d_pitch + (1024 if deltas > 1 else 0)
    op = ResidualNormOperator(T=T, E=E, mode=mode, eps=eps, d_pitch=d_pitch, deltas=deltas, d_stride=stride,
                              scale=scale).compile()
    dev = op.device
    g = torch.Generator().manual_seed(3 + E)
    x = torch.randn(T, E, generator=g) * 4
    d = (torch.randn(deltas, T, COLS, d_pitch, generator=g) * 30).bfloat16()
    wp, w1 = torch.rand(E, generator=g) + 0.5, torch.rand(E, generator=g) + 0.5
    s = 0.7128 if scale else 1.0
    prm = residual_params(wp.numpy(), w1.numpy(), s, E).view(np.int16)
    P = artlite.from_torch(torch.from_numpy(prm).view(torch.bfloat16), dev)
    X = artlite.from_torch(x.contiguous().view(torch.bfloat16).reshape(-1).clone(), dev)
    XN = artlite.from_torch(torch.zeros(T * E * 9 // 16, dtype=torch.bfloat16), dev)
    dd = d[..., :E // COLS].float().reshape(deltas, T, E).sum(0)
    if mode == "in":
        op.dispatch(P, X, XN)
    else:
        dp = torch.zeros(deltas, stride, dtype=torch.bfloat16)
        dp[:, :T * COLS * d_pitch] = d.reshape(deltas, -1)
        D = artlite.from_torch(dp.reshape(-1)[:(deltas - 1) * stride + T * COLS * d_pitch].contiguous(), dev)
        op.dispatch(P, X, D, X, XN)
    X.sync_from_device()
    XN.sync_from_device()
    want_x, want_n = _residual_reference(x, dd, wp, w1, s, eps, mode)
    got_x = X.to_torch().view(torch.float32).reshape(T, E).double()
    got_n = bfp16.from_panels(XN.to_torch(), T, E).double()
    want_q = torch.from_numpy(bfp16.values(want_n.float().numpy())).double()

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / b.norm())

    ex, en, eq = rel(got_x, want_x), rel(got_n, want_n), rel(want_q, want_n)
    print(f"x' {ex:.2e}, xn {en:.2e} (the model's values rounded to bfp16: {eq:.2e})")
    assert ex < 2e-5 and en < 1.1 * eq, (ex, en, eq)
