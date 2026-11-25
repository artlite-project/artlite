"""The prefills' norms (prefill/norm/): PrefillNormOperator ((x += delta,) RMSNorm on a bf16 residual:
Qwen's, Llama's) vs torch, ResidualNormOperator (an fp32 residual: Gemma's modes) vs its fp64 model."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

import artlite
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.prefill.norm import PrefillNormOperator, ResidualNormOperator, residual_params

from ..common import to_numpy, to_torch
from ..numerics import rel

_from_panels = bfp16.from_panels


CASES = [(64, True, 2048, 0, 8), (512, False, 3072, 0, 8),
         (512, True, 3072, 396, 8),  # delta of a Q6_K down projection (11-row tiles)
         (64, True, 3072, 396, 2), (64, False, 3072, 0, 1),  # fewer cores
         (512, True, 4096, 528, 8), (64, True, 4096, 528, 2), (512, False, 4096, 0, 8),  # 8B
         (2048, True, 3072, 396, 8)]  # (Llama's norm blocks)


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
    xb, wb = artlite.from_numpy(to_numpy(x.clone().flatten()), dev), artlite.from_numpy(to_numpy(w), dev)
    xnb = artlite.empty((T * E * 9 // 16,), dtype=bfloat16, device=dev)
    dp = d_pitch or E // 8
    dpad = torch.full((T, 8, dp), 1e30).bfloat16()  # (padding: garbage)
    dpad[:, :, :E // 8] = d.view(T, 8, E // 8)
    op.dispatch(*((xb, artlite.from_numpy(to_numpy(dpad.flatten()), dev), wb, xb, xnb) if add else (xb, wb, xnb)))
    xnb.sync_from_device()
    xs = (x.float() + d.float()).bfloat16().float() if add else x.float()
    ref = xs * torch.rsqrt((xs ** 2).mean(-1, keepdim=True) + 1e-5) * w.float()
    xn = to_torch(_from_panels(xnb.numpy(), T, E))
    rel = ((xn - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"xn rel-rms {rel}"  # bfp16ebs8 rounding: ~7e-3
    if add:
        xb.sync_from_device()
        assert torch.equal(to_torch(xb.numpy()).view(T, E), (x.float() + d.float()).bfloat16())


@pytest.mark.parametrize(("T", "E", "d_pitch"), [(256, 2048, 264), (256, 2304, -1)], ids=["T256-dp264", "T256-E2304"])
def test_prefill_norm_lo(T: int, E: int, d_pitch: int) -> None:
    """xn as hi + lo panels (``lo``: an MoE router's x; Qwen3.6-35B-A3B's hidden 2048 after a delta, the plain design;
    Inkling's 2304 without one (d_pitch -1), the replay design): per token group [hi | lo] panels; hi + lo within 4e-5 of
    x / rms(x) w in fp64 (x the bf16 sum; 2.6e-5: lo's rounding, ~2^-16 of each block's maximum; the per-token scale
    1 / rms 2.4e-6), hi the single panels' rounding."""
    add = d_pitch >= 0
    g = torch.Generator().manual_seed(T + E)
    x = (torch.randn(T, E, generator=g) * 2).bfloat16()
    d = torch.randn(T, E, generator=g).bfloat16() if add else torch.zeros(T, E).bfloat16()
    w = (torch.rand(E, generator=g) + 0.5).bfloat16()
    op = PrefillNormOperator(T=T, E=E, add=add, **({"d_pitch": d_pitch} if add else {}), lo=True).compile()
    assert op.replay == (not add)
    dev = op.device
    xb, wb = artlite.from_numpy(to_numpy(x.clone().flatten()), dev), artlite.from_numpy(to_numpy(w), dev)
    xnb = artlite.empty((2 * T * E * 9 // 16,), dtype=bfloat16, device=dev)
    if add:
        dp = d_pitch or E // 8
        dpad = torch.full((T, 8, dp), 1e30).bfloat16()
        dpad[:, :, :E // 8] = d.view(T, 8, E // 8)
        op.dispatch(xb, artlite.from_numpy(to_numpy(dpad.flatten()), dev), wb, xb, xnb)
    else:
        op.dispatch(xb, wb, xnb)
    xnb.sync_from_device()
    xs = (x.float() + d.float()).bfloat16().double()
    ref = xs * torch.rsqrt((xs ** 2).mean(-1, keepdim=True) + 1e-5) * w.double()
    both = to_torch(_from_panels(xnb.numpy(), T, 2 * E)).double()  # (per token [hi | lo])
    hi, lo = both[:, :E], both[:, E:]
    assert ((hi - ref).norm() / ref.norm()).item() < 1e-2
    rel = ((hi + lo - ref).norm() / ref.norm()).item()
    scale = ((hi + lo) * ref).sum(1) / (ref * ref).sum(1) - 1  # (per token)
    print(f"hi + lo rel-rms {rel:.2e}; per-token scale rms {scale.pow(2).mean().sqrt():.2e}")
    assert rel < 4e-5, rel
    if add:
        xb.sync_from_device()
        assert torch.equal(to_torch(xb.numpy()).view(T, E), xs.float().bfloat16())


@pytest.mark.parametrize(("T", "E"), [(64, 2048), (256, 5120)])
def test_prefill_norm_pre_pitch(T: int, E: int) -> None:
    """Two norms into one A of 2 E columns (``out_pitch``, ``out_col``: an MTP drafter's eh_proj input [enorm(emb) |
    hnorm(h)]): a plain one into columns 0 .., ``pre`` into E ..: (x w1) w2 / rms(x w1), x w1 rounded to bf16 (h =
    the model's output norm of its residual, then the drafter's hnorm); each within the bfp16 panels' rounding (E
    5120: the replay design)."""
    g = torch.Generator().manual_seed(T)
    e, x = (torch.randn(T, E, generator=g) * 2).bfloat16(), (torch.randn(T, E, generator=g) * 3).bfloat16()
    we, w1, w2 = ((torch.rand(E, generator=g) + 0.5).bfloat16() for _ in range(3))
    a = PrefillNormOperator(T=T, add=False, E=E, out_pitch=2 * E, out_col=0).compile()
    b = PrefillNormOperator(T=T, add=False, E=E, pre=True, out_pitch=2 * E, out_col=E).compile(device=a.device)
    dev = a.device
    xn = artlite.empty((T * 2 * E * 9 // 16,), dtype=bfloat16, device=dev)
    a.dispatch(artlite.from_numpy(to_numpy(e.clone().flatten()), dev), artlite.from_numpy(to_numpy(we), dev), xn)
    b.dispatch(artlite.from_numpy(to_numpy(x.clone().flatten()), dev), artlite.from_numpy(to_numpy(torch.cat([w1, w2])), dev), xn)
    xn.sync_from_device()
    both = to_torch(_from_panels(xn.numpy(), T, 2 * E)).double()
    ef = e.double()
    ref_e = ef * torch.rsqrt((ef ** 2).mean(-1, keepdim=True) + 1e-5) * we.double()
    xw = (x.float() * w1.float()).bfloat16().double()
    ref_h = xw * torch.rsqrt((xw ** 2).mean(-1, keepdim=True) + 1e-5) * w2.double()
    for got, ref in ((both[:, :E], ref_e), (both[:, E:], ref_h)):
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, rel


@pytest.mark.parametrize(("T", "add", "cols"), [(64, True, 4), (256, False, 4)])
def test_prefill_norm_padded(T: int, add: bool, cols: int) -> None:
    """E 2560 (Qwen3-Embedding-4B's hidden size: the replay design, E not a multiple of 1024) into panel rows of 3072
    columns (the GEMMs' K in whole chunks): the norm within the bfp16 panels' rounding, the padding panels unwritten."""
    E, P = 2560, 3072
    g = torch.Generator().manual_seed(T)
    x = (torch.randn(T, E, generator=g) * 2).bfloat16()
    d = torch.randn(T, E, generator=g).bfloat16()
    w = (torch.rand(E, generator=g) + 0.5).bfloat16()
    op = PrefillNormOperator(T=T, add=add, E=E, cols=cols, out_pitch=P).compile()
    dev = op.device
    xb, wb = artlite.from_numpy(to_numpy(x.clone().flatten()), dev), artlite.from_numpy(to_numpy(w), dev)
    xnb = artlite.from_numpy(to_numpy(torch.full((T * P * 9 // 16,), 7.0).bfloat16()), dev)  # (the padding: not zero)
    op.dispatch(*((xb, artlite.from_numpy(to_numpy(d.clone().flatten()), dev), wb, xb, xnb) if add else (xb, wb, xnb)))
    xnb.sync_from_device()
    xs = (x.float() + d.float()).bfloat16().float() if add else x.float()
    ref = xs * torch.rsqrt((xs ** 2).mean(-1, keepdim=True) + 1e-5) * w.float()
    out = xnb.numpy()
    xn = to_torch(_from_panels(out, T, P))
    assert ((xn[:, :E] - ref).norm() / ref.norm()).item() < 1e-2
    pad = to_torch(out).view(T // 8, P // 8, 36)[:, E // 8:]  # (the padding panels as they were)
    assert torch.equal(pad, torch.full_like(pad, 7.0))


# (T, E, d_pitch, cols, rows, deltas): Llama's and qwen35's projections in K parts
DELTAS = [(64, 5120, 660, 4, 96, 2), (256, 4096, 528, 8, 256, 2), (512, 2048, 264, 8, 512, 2),
          (256, 5120, 660, 8, 288, 4), (64, 5120, 0, 2, 64, 5)]


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
    xb = artlite.from_numpy(to_numpy(x.clone().flatten()), dev)
    xnb = artlite.empty((T * E * 9 // 16,), dtype=bfloat16, device=dev)
    op.dispatch(xb, artlite.from_numpy(to_numpy(dpad.flatten()), dev), artlite.from_numpy(to_numpy(w), dev), xb, xnb)
    xb.sync_from_device()
    xnb.sync_from_device()
    xs = (x.float() + d.float().sum(0)).bfloat16()
    assert torch.equal(to_torch(xb.numpy()).view(T, E), xs)
    ref = xs.float() * torch.rsqrt((xs.float() ** 2).mean(-1, keepdim=True) + 1e-5) * w.float()
    xn = to_torch(_from_panels(xnb.numpy(), T, E))
    assert ((xn - ref).norm() / ref.norm()).item() < 1e-2


# (T, E, d_pitch, cols, deltas): the plain design (E 2048: the output rows in two pieces), replay (two deltas too), hold
BF16 = [(256, 1024, 0, 8, 1), (64, 2048, 0, 8, 0), (256, 4096, 528, 8, 1), (256, 4096, 528, 8, 2), (256, 5120, 660, 8, 4)]


@pytest.mark.parametrize(("T", "E", "d_pitch", "cols", "n"), BF16,
                         ids=[f"T{a}-E{b}-dp{c}-c{d}-n{e}" for a, b, c, d, e in BF16])
def test_prefill_norm_bf16(T: int, E: int, d_pitch: int, cols: int, n: int) -> None:
    """xn as bf16 blocks (``bf16``: a PARO checkpoint's rotations' input, [T / 8][E / 8][8 tokens x 8 columns]) in
    each design: within bf16's rounding of x / rms(x) w, x the bf16 sum."""
    g = torch.Generator().manual_seed(T + E + n)
    x = (torch.randn(T, E, generator=g) * 2).bfloat16()
    w = (torch.rand(E, generator=g) + 0.5).bfloat16()
    dp = d_pitch or E // 8
    kw = {"deltas": n, "d_stride": T * 8 * dp} if n > 1 else {}
    op = PrefillNormOperator(T=T, E=E, d_pitch=d_pitch if n else 0, cols=cols, add=n > 0, bf16=True, **kw).compile()
    dev = op.device
    xb = artlite.from_numpy(to_numpy(x.clone().flatten()), dev)
    xnb = artlite.empty((T * E,), dtype=bfloat16, device=dev)
    xs = x.float()
    args = [xb]
    if n:
        d = torch.randn(n, T, E, generator=g).bfloat16()
        dpad = torch.full((n, T, 8, dp), 1e30).bfloat16()  # (padding: garbage)
        dpad[:, :, :, :E // 8] = d.view(n, T, 8, E // 8)
        args.append(artlite.from_numpy(to_numpy(dpad.flatten()), dev))
        xs = (x.float() + d.float().sum(0)).bfloat16().float()
    args += [artlite.from_numpy(to_numpy(w), dev), *([xb] if n else []), xnb]
    op.dispatch(*args)
    xnb.sync_from_device()
    ref = xs.double() * torch.rsqrt((xs.double() ** 2).mean(-1, keepdim=True) + 1e-5) * w.double()
    xn = to_torch(xnb.numpy()).double().view(T // 8, E // 8, 8, 8).transpose(1, 2).reshape(T, E)
    err = ((xn - ref).norm() / ref.norm()).item()
    assert err < 3e-3, err  # (bf16's rounding: ~1.6e-3)


def _residual_reference(x: torch.Tensor, d: torch.Tensor | None, w_post: torch.Tensor | None, w1: torch.Tensor | None,
                        scale: float, eps: float, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(x', xn values before the bfp16 rounding) in fp64 of fp32 x (T, E), d (T, E) the (summed) delta."""
    x = x.double()
    if mode == "add":
        x = x + d.double()
    elif mode != "in":
        dd = d.double()
        iv = torch.rsqrt(dd.bfloat16().double().pow(2).mean(-1, keepdim=True) + eps)
        x = (x + dd * iv * w_post.bfloat16().double()) * scale
    if mode == "plain":
        return x, x
    xb = x.float().bfloat16().double()
    iv = torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + eps)
    return x, xb * w1.bfloat16().double() * iv


# (mode, E, deltas, d_pitch, scale, the panels' row: 0 = E): Gemma E4B's modes, LFM3's (E 2048), Spark-X's pre-norm
# updates (add; the 4B's 2560 in rows of 3072: the GEMMs' K in whole chunks)
RESIDUAL = [("in", 2560, 1, 320, False, 0), ("post", 2560, 1, 320, False, 0), ("post", 2560, 1, 384, True, 0),
            ("plain", 2560, 2, 384, False, 0), ("plain", 2560, 4, 320, False, 0), ("post", 2048, 2, 256, False, 0),
            ("in", 2560, 1, 320, False, 3072), ("add", 2560, 2, 384, False, 3072), ("add", 2048, 2, 256, False, 0)]


@pytest.mark.parametrize(("mode", "E", "deltas", "d_pitch", "scale", "pitch"), RESIDUAL,
                         ids=[f"{m}-E{e}-d{d}-p{p}{'-s' if s else ''}{f'-o{o}' if o else ''}"
                              for m, e, d, p, s, o in RESIDUAL])
def test_residual_norm(mode: str, E: int, deltas: int, d_pitch: int, scale: bool, pitch: int) -> None:
    """ResidualNormOperator against its fp64 model: x' within ~2^-16 (the fp32 update), the panels within 1.1 x the
    error of the model's values rounded to bfp16. Several deltas: a down projection's K parts, ``d_stride`` apart
    (padded). Panels in longer rows (``pitch``): the padding panels unwritten."""
    T, eps = 256, 1e-6
    stride = T * COLS * d_pitch + (1024 if deltas > 1 else 0)
    op = ResidualNormOperator(T=T, E=E, mode=mode, eps=eps, d_pitch=d_pitch, deltas=deltas, d_stride=stride,
                              scale=scale, out_pitch=pitch).compile()
    P_ = pitch or E
    dev = op.device
    g = torch.Generator().manual_seed(3 + E)
    x = torch.randn(T, E, generator=g) * 4
    d = (torch.randn(deltas, T, COLS, d_pitch, generator=g) * 30).bfloat16()
    wp, w1 = torch.rand(E, generator=g) + 0.5, torch.rand(E, generator=g) + 0.5
    s = 0.7128 if scale else 1.0
    prm = residual_params(wp.numpy(), w1.numpy(), s, E).view(np.int16)
    P = artlite.from_numpy(prm.view(bfloat16), dev)
    X = artlite.from_numpy(to_numpy(x.contiguous().view(torch.bfloat16).reshape(-1)), dev)
    XN = artlite.from_numpy(np.full(T * P_ * 9 // 16, 7.0, bfloat16), dev)  # (the padding: not zero)
    dd = d[..., :E // COLS].float().reshape(deltas, T, E).sum(0)
    if mode == "in":
        op.dispatch(P, X, XN)
    else:
        dp = torch.zeros(deltas, stride, dtype=torch.bfloat16)
        dp[:, :T * COLS * d_pitch] = d.reshape(deltas, -1)
        D = artlite.from_numpy(to_numpy(dp.reshape(-1)[:(deltas - 1) * stride + T * COLS * d_pitch].contiguous()), dev)
        op.dispatch(P, X, D, X, XN)
    X.sync_from_device()
    XN.sync_from_device()
    want_x, want_n = _residual_reference(x, dd, wp, w1, s, eps, mode)
    got_x = to_torch(X.numpy()).view(torch.float32).reshape(T, E).double()
    got_n = to_torch(bfp16.from_panels(XN.numpy(), T, P_)[:, :E]).double()
    pad = XN.numpy().reshape(T // 8, P_ // 8, 36)[:, E // 8:]  # (the padding panels as they were)
    assert np.array_equal(pad, np.full_like(pad, 7.0))
    want_q = torch.from_numpy(bfp16.values(want_n.float().numpy())).double()

    ex, en, eq = rel(got_x, want_x), rel(got_n, want_n), rel(want_q, want_n)
    print(f"x' {ex:.2e}, xn {en:.2e} (the model's values rounded to bfp16: {eq:.2e})")
    assert ex < 2e-5 and en < 1.1 * eq, (ex, en, eq)


@pytest.mark.parametrize("mode", ["in", "post", "plain"])
def test_residual_norm_bf16(mode: str) -> None:
    """``bf16`` (a PARO Gemma's norms, w1 of ones: its weights in the rotations): xn as bf16 blocks of 8 tokens x 8
    columns in the panels' order, within bf16's rounding of the fp64 model's values (``plain``: x' itself)."""
    T, E, dp, eps = 256, 2560, 320, 1e-6
    op = ResidualNormOperator(T=T, E=E, mode=mode, eps=eps, d_pitch=dp, bf16=True).compile()
    dev = op.device
    g = torch.Generator().manual_seed(11)
    x = torch.randn(T, E, generator=g) * 4
    d = (torch.randn(1, T, COLS, dp, generator=g) * 30).bfloat16()
    wp = torch.rand(E, generator=g) + 0.5
    w1 = torch.ones(E)
    P = artlite.from_numpy(residual_params(wp.numpy(), w1.numpy(), 1.0, E).view(np.int16).view(bfloat16), dev)
    X = artlite.from_numpy(to_numpy(x.contiguous().view(torch.bfloat16).reshape(-1)), dev)
    XN = artlite.empty((T * E,), dtype=bfloat16, device=dev)
    if mode == "in":
        op.dispatch(P, X, XN)
    else:
        op.dispatch(P, X, artlite.from_numpy(to_numpy(d.reshape(-1).contiguous()), dev), X, XN)
    XN.sync_from_device()
    want_n = _residual_reference(x, d.float().reshape(T, E), wp, w1, 1.0, eps, mode)[1]
    got_n = to_torch(XN.numpy()).double().view(T // 8, E // 8, 8, 8).transpose(1, 2).reshape(T, E)
    err = rel(got_n, want_n)
    print(f"{mode}: xn {err:.2e}")
    assert err < 3e-3, err  # (bf16's rounding: ~1.6e-3)
