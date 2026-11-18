"""Numpy models of kmath's functions as the NPU computes them, bit-exact (``test_kmath.py`` checks each variant on the
NPU), and of the AIE2P arithmetic they are built from: bf16 rounding, the accumulators' fp32 adder, kernel_lib.h's
products through bf16 hi / lo parts (``mulc``, ``mul_hilo``, ``mulff``).

    emulate(fn, x)   the results of ``fn`` (a kmath.Fn) for the arguments x (fp32, in ``fn.valid``; bf16 values for a
                     bf16 argument; an array function's: its result per element)
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
    with np.errstate(over="ignore"):  # (past the fp32 range: inf, as on the NPU)
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
    """acc + a b (a, b bf16). A product larger than acc takes acc onto a grid 3 bits finer than its own (or the sum's,
    if larger), rounded to nearest even, then the sum rounded (probed: 99.9% of single MACs with cancellation, the rest
    1 ULP apart; a smaller product: ``add``, exact)."""
    acc, p = np.broadcast_arrays(np.asarray(acc, np.float64), np.asarray(a, np.float64) * np.asarray(b, np.float64))
    big = np.abs(p) > np.abs(acc)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = np.maximum(np.floor(np.log2(np.abs(p))), np.floor(np.log2(np.abs(p + acc))))
        q = np.exp2(e - 26)
        s = np.where(big & np.isfinite(p) & (p != 0), p + np.round(acc / q) * q, p + acc)
    return np.where(big, _f32(_ftz(s)), add(acc, p))


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


def _exp2_acc(a: np.ndarray, below: str, above: str, shift: int, prec: str = "bf16") -> np.ndarray:
    k1, k2, k3 = (np.float64(bf16(np.float32(c * 2.0 ** (shift - 64)))) for c in (*kmath.ACC_Q, 0.0)) if prec == "quad" \
        else (np.float64(bf16(np.float32(c * 2.0 ** (shift - 64)))) for c in kmath.ACC_C)
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
    pf = bf16(mul(p, f))
    pf2 = bf16(mul(pf, f))
    if prec == "quad":
        return mac(mac(mul(p, 2.0 ** (shift - 64)), pf, k1), pf2, k2)
    hi = bf16(mac(mul(f, k3), k2, 1.0))
    return mac(mac(mul(p, 2.0 ** (shift - 64)), pf, k1), pf2, hi)


def vexp2(a: np.ndarray) -> np.ndarray:
    """``al::exp2_acc`` (the vexp2 LUT): Schraudolph's exp of a truncated to bf16, t: k = floor(128 t), for t < 0
    -trunc(128 |t|) - 1 (one's complement), the value 2^e (1 + m / 128) of e = floor(k / 128), m = k - 128 e: the bf16
    bits k + 16256, below 2^-126 a denormal (the value rounded to nearest on its grid, at most 127 units: 2^-126 (1 -
    2^-7) at t = -126; 0 from t = -134 on; probed over every bf16 and fp32 inputs)."""
    t = (np.asarray(a, np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32).astype(np.float64)
    t = np.clip(np.nan_to_num(t, nan=0.0), -1024.0, 1024.0)  # (beyond the range: garbage anyway)
    k = np.where(t >= 0, np.floor(128 * t), -np.trunc(128 * -t) - 1).astype(np.int64)
    normal = ((((k + 16256) & 0xFFFF).astype(np.uint32)) << 16).view(np.float32)
    e = np.floor_divide(k, 128)
    den = np.minimum(np.round((128 + k - 128 * e) * np.exp2(np.minimum(e + 126, 0).astype(np.float64))), 127)  # (2^-133)
    sub = (den.astype(np.uint32) << 16).view(np.float32)
    return np.where(e >= -126, normal, sub)


_VTANH: np.ndarray | None = None  # (load_vtanh)


def load_vtanh(table: np.ndarray) -> None:
    """vtanh (``al::tanh_acc``: a LUT of the argument truncated to bf16, probed) as read on the NPU: its results for
    every bf16 input, indexed by the input's bits."""
    global _VTANH
    _VTANH = np.asarray(table, np.float32)


def vtanh(a: np.ndarray) -> np.ndarray:
    """``al::tanh_acc``: the loaded table (``load_vtanh``) at a truncated to bf16."""
    if _VTANH is None:
        raise RuntimeError("kmath_sim.vtanh: no table loaded (load_vtanh)")
    return _VTANH[(np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.int64)]


_INV: dict[str, np.ndarray] = {}  # (load_inv)


def load_inv(op: str, results: np.ndarray) -> None:
    """``aie::inv`` / ``aie::invsqrt`` (``op``: the scalar unit's functions, within 1 fp32 ULP of the rounded result,
    ~88% exact) as read on the NPU: its results for every x in [1, 4) (``inv_arguments``), by x's bits. Each depends on
    the mantissa alone (invsqrt: also on the exponent's parity), so the two binades give any normal x (probed: 2^18
    random x over 2^-120 .. 2^120, every one)."""
    _INV[op] = np.asarray(results, np.float32).view(np.uint32).reshape(2, 1 << 23)


def inv_arguments() -> np.ndarray:
    """Every fp32 in [1, 2), then in [2, 4): ``load_inv``'s arguments."""
    m = np.arange(1 << 23, dtype=np.uint32)
    return np.concatenate([m | 0x3F800000, m | 0x40000000]).view(np.float32)


def inv(x: np.ndarray, op: str = "inv") -> np.ndarray:
    """``aie::inv`` (or ``aie::invsqrt``) of normal fp32 x > 0: the loaded results (``load_inv``) at x's mantissa, the
    exponent negated (halved)."""
    if op not in _INV:
        raise RuntimeError(f"kmath_sim.inv: no {op} results loaded (load_inv)")
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    e, m = ((b >> 23) & 0xFF) - 127, b & 0x7FFFFF
    t = _INV[op]
    if op == "inv":
        r = t[0][m].astype(np.int64) - (e << 23)
    else:
        odd = e & 1
        r = np.where(odd == 1, t[1][m], t[0][m]).astype(np.int64) - ((e - odd) // 2 << 23)
    return (r & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


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


def _exp(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    """``kmath.exp``: exp_fn(x log2 e) (a bf16 x: its two products with log2 e's parts; neg: -x)."""
    if fn.params["arg"] == "float":
        return emulate(fn.params["exp_fn"], log2e(_f32(x)))
    s = -1.0 if fn.params["neg"] else 1.0
    x = bf16(x)
    return emulate(fn.params["exp_fn"], mac(mul(x, s * 1.4453125), x, -s * 0.00262451171875))


def _sigmoid_bf16(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    """``kmath.sigmoid(prec="bf16")``: 1 + e^-x (``kmath.exp``'s bf16 form), its aie::inv in bf16."""
    return bf16(inv(add(_exp(kmath.exp(fn.lanes, arg="bfloat16", neg=True, exp_fn=fn.params["exp_fn"]), x), 1.0)))


def _logistic(fn: kmath.Fn, x: np.ndarray) -> np.ndarray:
    if fn.params.get("prec") == "bf16":
        return _sigmoid_bf16(fn, x)
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


def _act(table: kmath.ActTable, x: np.ndarray, f32: bool = False) -> np.ndarray:
    """``kmath.act(table)``: t = min(|x|, t_max) (x clamped for a full table), its segment's quadratic plus relu(x)
    (``f32``: not rounded to bf16)."""
    x = bf16(x)
    tm = np.float32(table.t_max)
    v, r, r2 = table.reduce(np.clip(x, -tm, tm) if table.full else np.minimum(np.abs(x), tm))
    s = ((v - table.base) / table.seg).astype(np.int64)
    c0, c1, c2 = (_f32(table.coefficients[s, i]) for i in range(3))
    acc = mac(mac(c0, c1, r), c2, r2)
    y = acc if table.full else mac(acc, np.maximum(x, np.float32(0)), 1.0)
    return y if f32 else bf16(y)


def _silu_tanh(x: np.ndarray) -> np.ndarray:
    """``kmath.silu_tanh``: A (vtanh, the chord tail below -3), then B (its silu_small below |x| = 2.5)."""
    x = bf16(x)
    a_acc = mul(x, 0.5)
    a, t = bf16(a_acc), vtanh(a_acc)
    body = bf16(mac(a_acc, a, t))
    a1t = bf16(mac(a_acc, a, -t.astype(np.float64)))
    tail = bf16(mul(a1t, vexp2(mul(x, bf16(np.float32(1.44269504))))))
    y = np.where(x < -3, tail, body)
    s = bf16(mul(bf16(mul(x, 0.5)), bf16(mul(x, 0.5))))
    acc = mac(mul(x, 0.5), s, 0.99609375)
    acc = mac(acc, bf16(mul(s, s)), bf16(mac(np.float32(-0.30078125), s, 0.06298828125)))
    return np.where(np.abs(x) < 2.5, bf16(acc), y)


def reduce_add(v: np.ndarray) -> np.ndarray:
    """``aie::reduce_add`` of fp32 vectors (the last axis, 64 lanes or fewer: a power of two) on the accumulators'
    adder: lanes j + j + L / 2 until one is left."""
    v = _f32(v)
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        v = add(v[..., :h], v[..., h:])
    return v[..., 0]


def inv_rms(ss: np.ndarray, n: int, eps: float) -> np.ndarray:
    """``al::inv_rms<n>``: 1 / sqrt(ss / n + eps), ss / n as bf16 products (1 / n a bf16 for n a power of two, else
    al::mulff), the sum on the adder, aie::invsqrt."""
    ms = mulc(_f32(ss), bf16(np.float32(1 / n))) if n & (n - 1) == 0 else mulff(_f32(ss), np.float32(1 / n))
    return inv(add(ms, np.float32(eps)), "invsqrt")


def rms_norm(fn: kmath.Fn, x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """``kmath.rms_norm``: rows x (bf16 values [rows, n]) -> fp32 x / rms(x) (times w)."""
    n, eps = fn.params["n"], fn.params["eps"]
    x = bf16(x)
    xs = x.reshape(-1, n // 64, 64)
    ss = mul(xs[:, 0], xs[:, 0])
    for k in range(1, n // 64):
        ss = mac(ss, xs[:, k], xs[:, k])  # (mac_square: per lane, in order)
    iv = inv_rms(reduce_add(ss), n, eps)[:, None]
    ih, il = split(iv)
    y = mac(mul(x, ih), x, il)
    return mulff(y, _f32(w)) if fn.params["weight"] else y


def rot2(a: np.ndarray, b: np.ndarray, c: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``al::rot2``: a c - b s, a s + b c, each operand split once, a pair's six products in one accumulator (their
    cancellations meet ``mac``'s unmodelled cases: ~0.5% of the results 1-2 ULP off)."""
    (ah, al), (bh, bl), (ch, cl), (sh, sl) = split(a), split(b), split(c), split(s)

    def msc(acc: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return mac(acc, -np.asarray(p, np.float64), q)

    o0 = msc(msc(msc(mac(mac(mul(ah, ch), ah, cl), al, ch), bh, sh), bh, sl), bl, sh)
    o1 = mac(mac(mac(mac(mac(mul(ah, sh), ah, sl), al, sh), bh, ch), bh, cl), bl, ch)
    return o0, o1


def rope(fn: kmath.Fn, row: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """``kmath.rope``: fp32 rows [rows, n] with their pairs (i, i + dist), i < rot, rotated by cos cs[i] and sin
    cs[rot + i] (a bf16 table's values as given: bf16)."""
    rot, dist = fn.params["rot"], fn.params["dist"]
    out = _f32(row).copy()
    c, s = _f32(cs[:rot]), _f32(cs[rot:2 * rot])
    a, b = out[:, :rot], out[:, dist:dist + rot]
    if fn.params["table"] == "bf16":  # (four products: c, s bf16)
        (ah, al), (bh, bl) = split(a), split(b)
        n = lambda v: -np.asarray(v, np.float64)
        o0 = mac(mac(mac(mul(ah, c), al, c), n(bh), s), n(bl), s)
        o1 = mac(mac(mac(mul(ah, s), al, s), bh, c), bl, c)
    else:
        o0, o1 = rot2(a, b, c, s)
    out[:, :rot], out[:, dist:dist + rot] = o0, o1
    return out


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
        return _exp2_acc(x, p["below"], p["above"], p["shift"], p["prec"])
    if fn.func == "recip":
        return _recip(x, p["domain"], p["steps"])
    if fn.func == "exp":
        return _exp(fn, x)
    if fn.func == "act":
        return _act(p["table"], x, p["f32"])
    if fn.func == "silu_tanh":
        return _silu_tanh(x)
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
        "exp": lambda: np.exp(-x if fn.params.get("neg") else x),
        "silu": lambda: x * sig(x),
    }
    if fn.func == "act":
        return fn.params["table"].f64(x)
    if fn.func == "silu_tanh":
        return kmath.silu64(x)
    if fn.func == "exp2":
        p = fn.params
        scale = kmath.CHORD2_SCALE if p["prec"] == "chord2" else 1.0
        return scale * np.exp2(x - (p["bias"] or 0) + p["shift"])
    if fn.func in funcs:
        return funcs[fn.func]()
    raise ValueError(f"{fn.name}: no exact function")
