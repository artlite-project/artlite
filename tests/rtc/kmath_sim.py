"""Numpy models of kmath's functions as the NPU computes them, bit-exact (``test_kmath.py`` checks each variant on the
NPU), and of the AIE2P arithmetic they are built from: bf16 rounding, the accumulators' fp32 adder, kernel_lib.h's
products through bf16 hi / lo parts (``mulc``, ``mul_hilo``, ``mulff``).

    emulate(fn, x)   the results of ``fn`` (a kmath.Fn) for the arguments x (fp32, in ``fn.valid``; bf16 values for a
                     bf16 argument)
    exact(fn, x)     the function ``fn`` computes, in float64

The fp32 adder (accumulator add / sub, fp32 vector add / sub, a MAC's sum) is not IEEE: it aligns the smaller operand
to the larger's exponent with 23 fractional bits (AM027), rounding it there to nearest even, adds exactly and rounds
to nearest even on a carry, so after a cancellation the sum keeps only the larger operand's grid (probed on the NPU:
every pair of a random set, ties included). A bf16 x bf16 product is exact in fp32. A result below 2^-126 is flushed
to zero, judged by its exact value: a product (``aie::mul``), a sum before its rounding (a MAC's: of the exact,
unflushed product), a bf16 conversion; a zero's sign is not modelled.
"""

import numpy as np
import torch

from artlite.operators import kmath


def _ftz(x: np.ndarray) -> np.ndarray:
    """x with its denormals (below 2^-126) flushed to zero."""
    return np.where(np.abs(x) < 2.0 ** -126, 0, x).astype(x.dtype)


def _f32(x: object) -> np.ndarray:
    return _ftz(np.asarray(x, np.float32))


def bf16(x: object) -> np.ndarray:
    """fp32 -> bf16 rounded to nearest even (``aie::rounding_mode::conv_even``), as fp32."""
    return _ftz(torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().numpy())


def add(x: object, y: object) -> np.ndarray:
    """x + y on the accumulators' adder (fp32 values or exact bf16 products)."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    big = np.where(np.abs(x) >= np.abs(y), x, y)
    small = np.where(np.abs(x) >= np.abs(y), y, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.exp2(np.floor(np.log2(np.abs(big))) - 23)
        s = np.where(np.isfinite(big) & (big != 0), big + np.round(small / q) * q, big + small)
    return _f32(_ftz(s))  # (flushed before it is rounded)


def mul(a: object, b: object) -> np.ndarray:
    """a b (a, b bf16: ``aie::mul`` into an fp32 accumulator)."""
    return _f32(_ftz(np.asarray(a, np.float64) * np.asarray(b, np.float64)))


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
    return mac(mul(hi, c), lo, c)


def mul_hilo(x: np.ndarray, h: float, lo_c: float) -> np.ndarray:
    """``al::mul_hilo``: fp32 x times the constant h + lo_c (its bf16 parts: a constant that is no bf16 value is
    rounded to one, as ``aie::broadcast<bfloat16>`` does)."""
    h, lo_c = (np.float64(bf16(np.float32(c))[()]) for c in (h, lo_c))
    hi, lo = split(x)
    return mac(mac(mul(hi, h), lo, h), hi, lo_c)


def mulff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``al::mulff``: fp32 a times fp32 b."""
    ah, al = split(a)
    bh, bl = split(b)
    return mac(mac(mul(ah, bh), ah, bl), al, bh)


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
    hi = bf16(mac(mul(f, k3), k2, 1.0))
    pf = bf16(mul(p, f))
    pf2 = bf16(mul(pf, f))
    return mac(mac(mul(p, 2.0 ** (shift - 64)), pf, k1), pf2, hi)


def vexp2(a: np.ndarray) -> np.ndarray:
    """``al::exp2_acc`` (the vexp2 LUT): Schraudolph's exp of a truncated to bf16, t: the bf16 bits floor(128 t) +
    16256, for t < 0 16255 - trunc(128 |t|) (one's complement; probed over every bf16 and fp32 inputs)."""
    t = (np.asarray(a, np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32).astype(np.float64)
    k = np.where(t >= 0, np.floor(128 * t), -np.trunc(128 * -t) - 1)
    return (((k + 16256).astype(np.int64) & 0xFFFF).astype(np.uint32) << 16).view(np.float32)


def log2e(x: np.ndarray) -> np.ndarray:
    """``al::log2e``: x log2(e) (bf16 hi + lo constant)."""
    return mul_hilo(x, 1.4453125, -0.00262451171875)


def _recip(x: np.ndarray, domain: str, steps: int) -> np.ndarray:
    x = _f32(x)
    if domain == "normal":
        b = x.view(np.int32).astype(np.int64)
        e = (b & 0x7F800000) - 0x3F800000
        m = (b - e).astype(np.int32).view(np.float32)
    else:
        m = x
    r = add(np.float32(1.4117647), -mulc(m, bf16(np.float32(0.47058824))))
    for _ in range(steps):
        r = mulff(r, add(2.0, -mulff(m, r)))
    if domain == "normal":
        return (r.view(np.int32).astype(np.int64) - e).astype(np.int32).view(np.float32)
    return r


def _expneg(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    x = _f32(x)
    return emulate(fn.params["exp_fn"], log2e(-np.abs(x)))


def _expm1(fn: kmath.Fn, g: np.ndarray) -> np.ndarray:
    g = _f32(g)
    p = np.full_like(g, kmath.EXPM1_C[0])
    for c in kmath.EXPM1_C[1:]:
        p = add(c, mulff(g, p))
    small = add(g, mulff(mulff(g, g), p))
    big = add(emulate(fn.params["exp_fn"], log2e(g)), -1.0)
    return np.where(g > np.float32(-0.5), small, big)


def _logistic(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    x = _f32(x)
    t = emulate(fn.params["expneg"], x)
    rf = fn.params["recip_fn"]
    if fn.func == "sigmoid":
        r = emulate(rf, add(1.0, t))
        return np.where(x < 0, mulff(t, r), r)
    r = mulff(emulate(rf, add(1.0, mulff(t, np.float32(0.5)))), np.float32(0.5))
    s = mulff(t, r)
    s2 = mulff(s, s)
    p = np.full_like(x, kmath.ATANH_C[0])
    for c in kmath.ATANH_C[1:]:
        p = add(c, mulff(s2, p))
    return add(np.maximum(x, np.float32(0)), mulff(add(s, s), p))


def _silu(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    xf = bf16(x)
    t = bf16(emulate(fn.params["exp_fn"], log2e(-np.abs(xf))))
    r = emulate(fn.params["recip_fn"], add(t, 1.0))
    return mulff(xf, np.where(xf < 0, mulff(t, r), r))


def emulate(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    """``fn``'s results for the arguments x."""
    p = fn.params
    if fn.func == "exp2":
        if p["bias"] is None:
            return _exp2_f32(x, p["prec"], p["below"])
        if p["bias"] == 0:
            a = _f32(x)
            if p["prec"] == "chord":
                return vexp2(a)
            return mac(mul(vexp2(a), kmath.CHORD2_C[0]), vexp2(add(a, 0.5)), kmath.CHORD2_C[1])
        return _exp2_acc(x, p["below"], p["above"], p["shift"])
    if fn.func == "recip":
        return _recip(x, p["domain"], p["steps"])
    models = {"expneg": _expneg, "expm1": _expm1, "sigmoid": _logistic, "softplus": _logistic, "silu": _silu}
    if fn.func in models:
        return models[fn.func](fn, x)
    raise ValueError(f"{fn.name}: no model")


def exact(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    """The function ``fn`` computes, in float64."""
    x = np.asarray(x, np.float64)
    sig = lambda v: np.where(v >= 0, 1 / (1 + np.exp(-np.abs(v))), np.exp(-np.abs(v)) / (1 + np.exp(-np.abs(v))))
    funcs = {
        "recip": lambda: 1 / x,
        "expneg": lambda: np.exp(-np.abs(x)),
        "expm1": lambda: np.expm1(x),
        "softplus": lambda: np.logaddexp(0, x),
        "sigmoid": lambda: sig(x),
        "silu": lambda: x * sig(x),
    }
    if fn.func == "exp2":
        p = fn.params
        scale = kmath.CHORD2_SCALE if p["prec"] == "chord2" else 1.0
        return scale * np.exp2(x - (p["bias"] or 0) + p["shift"])
    if fn.func in funcs:
        return funcs[fn.func]()
    raise ValueError(f"{fn.name}: no exact function")
