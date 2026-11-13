"""The comparators and roundings the tests share: relative errors (each tolerance is calibrated against one of
them, so each is a function of its own), KL divergences of logits, bf16 rounding and bf16 bits, the GELUs of the
CPU references, and the elementwise tolerance check of the stream operators.

Arguments are numpy arrays (``ml_dtypes.bfloat16`` too) or torch tensors, in any mix."""

import math

import numpy as np
import pytest
import torch

from .common import as_torch, to_torch


def _tensor(x: object) -> torch.Tensor:
    """``x`` as a CPU tensor (numpy arrays as ``to_torch``: read-only ones copied, bf16 kept)."""
    if isinstance(x, torch.Tensor):
        return x
    a = np.asarray(x)
    return to_torch(a if a.flags.writeable else a.copy())


def rel(a: object, b: object) -> float:
    """``|a - b| / |b|`` (Frobenius norms) in float64."""
    a, b = _tensor(a).double(), _tensor(b).double()
    return float((a - b).norm() / b.norm())


def rel_native(a: object, b: object) -> float:
    """``|a - b| / |b|`` in the tensors' own dtype (float32: the torch norm's rounding is part of the value the
    tolerances were calibrated on)."""
    a, b = as_torch(a), as_torch(b)
    return float((a - b).norm() / b.norm())


def rel_finite(a: object, b: object) -> float:
    """``|a - b| / |b|`` in float32 (the rms ratio), inf when ``a`` is not finite (a NaN would compare as no error)."""
    a, b = _tensor(a).float(), _tensor(b).float()
    if not torch.isfinite(a).all():
        return float("inf")
    return float((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt())


def rel_rows(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per row of two matrices: ``|got - ref| / |ref|`` (float64 norms of the rows)."""
    return np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)


def rel_max(a: object, b: object) -> float:
    """``max |a - b| / max |b|`` in float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def kl_rows(want: object, got: object) -> torch.Tensor | np.ndarray:
    """``KL(softmax(want) || softmax(got))`` per row (float64); a numpy array for two numpy arrays, else a tensor."""
    arrays = isinstance(want, np.ndarray) and isinstance(got, np.ndarray)
    lw = torch.log_softmax(_tensor(np.array(want, np.float64) if arrays else want).double(), -1)
    lg = torch.log_softmax(_tensor(np.array(got, np.float64) if arrays else got).double(), -1)
    out = (lw.exp() * (lw - lg)).sum(-1)
    return out.numpy() if arrays else out


def kl_agree(want: object, got: object) -> tuple[float, float]:
    """The mean KL of the rows and the fraction whose argmax agrees."""
    want, got = _tensor(want), _tensor(got)
    return float(kl_rows(want, got).mean()), float((want.argmax(-1) == got.argmax(-1)).float().mean())


def kl_report(name: str, want: object, got: object, *, where: str = "") -> tuple[float, float]:
    """``kl_agree`` printed as llama.cpp's comparison (``where``: which rows); returns (mean KL, agreement)."""
    want, got = _tensor(want), _tensor(got)
    k = kl_rows(want, got)
    agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
    print(f"{name}: KL vs llama.cpp{where} {float(k.mean()):.4f} (max {float(k.max()):.4f}), argmax agreement {agree:.3f} "
          f"over {len(k)} tokens")
    return float(k.mean()), agree


def bf16_round(x: object) -> np.ndarray | torch.Tensor:
    """The values of ``x`` rounded to bf16, as float32: a tensor for a tensor, else a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.bfloat16().float()
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy()


def bf16_like(x: torch.Tensor) -> torch.Tensor:
    """The values of a tensor rounded to bf16, in its own dtype."""
    return x.to(torch.bfloat16).to(x.dtype)


def from_bf16_bits(u: np.ndarray) -> np.ndarray:
    """Raw bf16 bits (uint16) -> float32 values."""
    return (np.asarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_table(x: torch.Tensor) -> torch.Tensor:
    """ggml's CPU GELU (tanh): an fp16 table of the fp16-rounded input (|x| < 10), the identity / 0 outside."""
    xh = x.to(torch.float16).to(torch.float32)
    y = gelu_tanh(xh).to(torch.float16).to(x.dtype)
    return torch.where(x <= -10, torch.zeros_like(x), torch.where(x >= 10, x, y))


def verify_close(output, reference, *, rel_tol, abs_tol, label="output"):
    """Assert elements are within tolerance.

    Uses IRON's verify_buffer formula:
      pass if abs_err < max(abs_tol, rel_tol * (|out| + |ref|))

    Handles reference in float64 (high-precision) by converting to float32.
    """
    output, reference = (to_torch(x) if isinstance(x, np.ndarray) else x for x in (output, reference))
    assert output.shape == reference.shape, f"{label}: shape mismatch {output.shape} vs {reference.shape}"
    out_f = output.float().reshape(-1)
    ref_f = reference.float().reshape(-1)  # Converts float64 -> float32 if needed
    abs_err = (out_f - ref_f).abs()
    norm = (out_f.abs() + ref_f.abs()).clamp(max=3.4e38)
    threshold = torch.maximum(
        torch.tensor(abs_tol),
        torch.tensor(rel_tol) * norm,
    )
    ok = abs_err < threshold
    n_bad = (~ok).sum().item()
    total = output.numel()
    error_rate = n_bad / total if total > 0 else 0.0

    if n_bad == 0:
        return

    bad_idx = (~ok).nonzero(as_tuple=True)[0]
    worst_abs = abs_err[bad_idx].max().item()
    rel_err = abs_err / ref_f.abs().clamp(min=1e-30)
    worst_rel = rel_err[bad_idx].max().item()
    samples = bad_idx[:5]
    details = [
        f"  [{i.item()}] out={out_f[i].item():.6f} ref={ref_f[i].item():.6f} "
        f"abs={abs_err[i].item():.6f} rel={rel_err[i].item():.6f}"
        for i in samples
    ]
    pytest.fail(
        f"{label}: {n_bad}/{total} ({error_rate:.2%}) elements exceed tolerance "
        f"(rel_tol={rel_tol}, abs_tol={abs_tol})\n"
        f"  worst abs_err={worst_abs:.6f}, worst rel_err={worst_rel:.6f}\n" + "\n".join(details)
    )

