"""Numpy models of kmath's functions as the NPU computes them, bit-exact (``test_kmath.py`` checks each variant on the
NPU), and of the AIE2P arithmetic they are built from: bf16 rounding, the accumulators' fp32 adder, kernel_lib.h's
products through bf16 hi / lo parts (``mulc``, ``mul_hilo``, ``mulff``).

    emulate(fn, x)   the results of ``fn`` (a kmath.Fn) for the arguments x (fp32, in ``fn.valid``)
    exact(fn, x)     the function ``fn`` computes, in float64

The fp32 adder (accumulator add / sub, fp32 vector add / sub, a MAC's sum) is not IEEE: it aligns the smaller operand
to the larger's exponent with 23 fractional bits (AM027), rounding it there to nearest even, adds exactly and rounds
to nearest even on a carry, so after a cancellation the sum keeps only the larger operand's grid (probed on the NPU:
every pair of a random set, ties included). A bf16 x bf16 product is exact in fp32.
"""

import numpy as np
import torch

from artlite.operators import kmath


def _f32(x: object) -> np.ndarray:
    return np.asarray(x, np.float32)


def bf16(x: object) -> np.ndarray:
    """fp32 -> bf16 rounded to nearest even (``aie::rounding_mode::conv_even``), as fp32."""
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy()


def add(x: object, y: object) -> np.ndarray:
    """x + y on the accumulators' adder (fp32 values or exact bf16 products)."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    big = np.where(np.abs(x) >= np.abs(y), x, y)
    small = np.where(np.abs(x) >= np.abs(y), y, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.exp2(np.floor(np.log2(np.abs(big))) - 23)
        s = np.where(np.isfinite(big) & (big != 0), big + np.round(small / q) * q, big + small)
    return _f32(s)


def mac(acc: object, a: object, b: object) -> np.ndarray:
    """acc + a b (a, b bf16)."""
    return add(acc, np.asarray(a, np.float64) * np.asarray(b, np.float64))


def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The bf16 hi / lo parts of fp32 x (``al::split2``)."""
    hi = bf16(x)
    return hi, bf16(add(x, -hi))


def mulc(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``al::mulc``: fp32 x times an exact bf16 c."""
    hi, lo = split(x)
    return mac(hi * c.astype(np.float64), lo, c)


def mul_hilo(x: np.ndarray, h: float, lo_c: float) -> np.ndarray:
    """``al::mul_hilo``: fp32 x times the constant h + lo_c (bf16 parts)."""
    hi, lo = split(x)
    return mac(mac(hi * np.float64(h), lo, h), hi, lo_c)


def mulff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``al::mulff``: fp32 a times fp32 b."""
    ah, al = split(a)
    bh, bl = split(b)
    return mac(mac(ah * bh.astype(np.float64), ah, bl), al, bh)


def _ldexp_bits(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """p with n added to its exponent bits (integer arithmetic, as the kernels do)."""
    bits = p.view(np.int32).astype(np.int64) + (n.astype(np.int64) << 23)
    return (bits & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def _exp2_f32(x: np.ndarray, prec: str, below: str) -> np.ndarray:
    x = _f32(x)
    if below != "int":
        x = np.maximum(x, np.float32(-127 if below == "zero" else -126))
    magic = np.float32(12582912)
    y = add(x, magic)
    f = add(x, -add(y, -magic))
    n = y.view(np.int32).astype(np.int64) - 0x4B400000
    if below == "int":
        n = np.maximum(n, -126)
    if prec == "fp32":
        t = mul_hilo(f, *kmath.LN2)
        p = np.full_like(f, kmath.TAYLOR[0])
        for c in kmath.TAYLOR[1:]:
            p = add(c, mulff(t, p))
    else:
        fb = bf16(f)
        q2 = bf16(mac(np.float32(kmath.CUBIC[0]), fb, bf16(np.float32(kmath.CUBIC[1]))))
        q1 = bf16(mac(np.float32(kmath.CUBIC[2]), fb, q2))
        p = add(1.0, mulc(f, q1))
    return _ldexp_bits(p, n)


def _exp2_acc(a: np.ndarray, below: str, above: str, shift: int) -> np.ndarray:
    k1, k2, k3 = (np.float64(bf16(np.float32(c * 2.0 ** (shift - 64)))) for c in kmath.ACC_C)
    a = _f32(a)
    v = bf16(a)
    if below == "clamp" and above == "none":  # (f from the clamped v)
        v = np.maximum(v, np.float32(129))
        f = np.maximum(bf16(mac(a, v, -1.0)), np.float32(-0.5))
    else:  # (f the residual of the unclamped v)
        f = bf16(mac(a, v, -1.0))
        if above == "clamp":
            f = np.clip(f, np.float32(-0.5), np.float32(0.5))
        if below != "none":
            v = np.maximum(v, np.float32(128))
        if above == "clamp":
            v = np.minimum(v, np.float32(255))
    p = _f32(np.exp2(v.astype(np.float64) - 128))  # (a non-negative integer power: vexp2 exact)
    hi = bf16(mac(f * k3, k2, 1.0))
    pf = bf16(p.astype(np.float64) * f)
    pf2 = bf16(pf.astype(np.float64) * f)
    return mac(mac(p * 2.0 ** (shift - 64), pf, k1), pf2, hi)


def emulate(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    """``fn``'s results for the arguments x."""
    if fn.func == "exp2":
        p = fn.params
        if p["bias"] is None:
            return _exp2_f32(x, p["prec"], p["below"])
        return _exp2_acc(x, p["below"], p["above"], p["shift"])
    raise ValueError(f"{fn.name}: no model")


def exact(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    """The function ``fn`` computes, in float64."""
    if fn.func == "exp2":
        return np.exp2(np.asarray(x, np.float64) - (fn.params["bias"] or 0) + fn.params["shift"])
    raise ValueError(f"{fn.name}: no exact function")
