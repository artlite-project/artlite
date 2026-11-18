"""Functions built on exp2 and recip: e^x, e^-|x|, e^g - 1, softplus, sigmoid, silu (templates:
``composite.*.cc.jinja``).
A composite calls the functions a caller names (``exp_fn``, ``recip_fn``; a caller's own, e.g. one its template
already defines) or the defaults, and its name carries the choice."""

from __future__ import annotations

import numpy as np

from .exponent import exp2
from .fn import KM, LANES, Fn, bound, check, make
from .reciprocal import recip

EXPM1_C = tuple(np.float32(1) / np.float32(k) for k in (5040, 720, 120, 24, 6, 2))  # (e^g - 1 - g) / g^2, highest first
ATANH_C = tuple(np.float32(1) / np.float32(k) for k in (11, 9, 7, 5, 3, 1))  # atanh(s) / s in s^2, highest first


def _local(fn: Fn) -> str:
    return fn.name.removeprefix(KM + "_")


def _take(what: str, fn: Fn | None, default: Fn, **need: object) -> tuple[Fn, str]:
    """The function a composite calls for ``what``: ``fn`` (a caller's, e.g. one its template already defines) or
    ``default``, checked against what the composite needs (``func``, ``lanes``, ``params``; a set: any of its values);
    and the suffix it adds to the composite's name (none for the default)."""
    fn = default if fn is None else fn
    for k, v in need.items():
        have = getattr(fn, k) if k in ("func", "lanes") else fn.params.get(k)
        if have not in (v if isinstance(v, set) else {v}):
            raise ValueError(f"kmath: {what} = {fn.name}: its {k} is {have!r}, not {v!r}")
    return fn, "" if fn.name == default.name else f"__{_local(fn)}"


def exp(lanes: int, *, arg: str = "float", neg: bool = False, exp_fn: Fn | None = None, inline: bool = True) -> Fn:
    """e^x (``neg``: e^-x) of ``lanes`` lanes as ``exp_fn``(x log2 e) (an exp2 of fp32 arguments).

    ``arg`` "float": fp32 x, x log2 e through al::log2e (~2^-16 relative: e^x off by ~1.6e-5 |x|), 1.4e-3 for x in [-87,
    88), 2^-15.9 for |x| <= 1 (``exp_fn`` default: the fp32 one).

    ``arg`` "bfloat16": bf16 x, x log2 e as the two products of x with log2 e's bf16 parts in one accumulator (al::log2e's
    values: x's lo part is 0), the bf16-level exp2 below "zero" by default (a gate's e^-x: 0.4 bf16 ULP; exactly 0 where
    -x log2 e <= -127)."""
    check("lanes", lanes, LANES)
    check("arg", arg, ("float", "bfloat16"))
    if neg and arg == "float":
        raise ValueError("kmath: exp neg = True takes a bf16 x")
    bf = arg == "bfloat16"
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, below="zero", inline=True) if bf else exp2(lanes, prec="fp32"),
                   func="exp2", lanes=lanes, bias=None)
    ulp, rel = (0.4, None) if bf else (None, 1.4e-3)
    ulp, rel = (None, None) if tag else (ulp, rel)
    lo = 88.0 if not neg else -88.0
    return make(f"exp_{'b' if bf else ''}{lanes}" + ("_neg" if neg else "") + tag + ("" if inline else "_o"),
                "composite.exp.cc.jinja", inline=inline, deps=(e,), lanes=lanes, arg=arg, ret="float", func="exp",
                params={"arg": arg, "neg": neg, "exp_fn": e},
                domain=(-87.0, 88.0) if not neg else (-88.0, 87.0),
                valid=(max(e.valid[0] / 1.45, -1e38), 88.0) if not neg else (-88.0, 3e38), ulp=ulp, rel=rel,
                variables={"N": lanes, "exp2": e.name, "bf": bf, "neg": neg, "lo": lo, "bound": bound(ulp, rel)})


def expneg(lanes: int, *, exp_fn: Fn | None = None, inline: bool = True) -> Fn:
    """e^(-|x|) of ``lanes`` fp32 lanes, ``exp`` of -|x| through ``exp_fn`` (an exp2 of fp32 arguments; default: the
    fp32 one)."""
    check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"), func="exp2", lanes=lanes, bias=None,
                   below={"clamp", "zero"})
    x = exp(lanes, exp_fn=e)
    rel = None if tag else 1.4e-3
    return make(f"expneg_{lanes}{tag}" + ("" if inline else "_o"), "composite.expneg.cc.jinja", inline=inline,
                deps=(x,), lanes=lanes, arg="float", ret="float", func="expneg", params={"exp_fn": e},
                domain=(-87.0, 87.0), valid=(-1e38, 1e38), rel=rel,
                variables={"N": lanes, "exp": x.name, "bound": bound(None, rel)})


def expm1(lanes: int, *, exp_fn: Fn | None = None, inline: bool = False) -> Fn:
    """e^g - 1 of ``lanes`` fp32 lanes for g <= 0 to 2^-15.6 (``exp_fn``: an exp2 of fp32 arguments, default the
    fp32 one)."""
    check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"), func="exp2", lanes=lanes, bias=None,
                   below={"clamp", "zero"})
    x = exp(lanes, exp_fn=e)
    rel = None if tag else 2.0e-5
    return make(f"expm1_{lanes}{tag}" + ("_i" if inline else ""), "composite.expm1.cc.jinja", inline=inline,
                deps=(x,), lanes=lanes, arg="float", ret="float", func="expm1", params={"exp_fn": e},
                domain=(-87.0, 0.0), valid=(-1e38, 0.0), rel=rel,
                variables={"N": lanes, "exp": x.name, "c": [str(k) for k in EXPM1_C], "bound": bound(None, rel)})


def _logistic(func: str, lanes: int, exp_fn: Fn | None, recip_fn: Fn | None, inline: bool, rel: float) -> Fn:
    check("lanes", lanes, LANES)
    t = expneg(lanes, exp_fn=exp_fn)
    r, rtag = _take("recip_fn", recip_fn, recip(lanes, steps=3), func="recip", lanes=lanes, domain="unit")
    etag = "" if exp_fn is None else _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"))[1]
    rel = None if etag or rtag else rel
    return make(f"{func}_{lanes}{etag}{rtag}" + ("_i" if inline else ""), f"composite.{func}.cc.jinja", inline=inline,
                deps=(t, r), lanes=lanes, arg="float", ret="float", func=func, params={"expneg": t, "recip_fn": r},
                domain=(-80.0, 80.0), valid=(-1e38, 1e38), rel=rel,
                variables={"N": lanes, "expneg": t.name, "recip": r.name, "c": [str(k) for k in ATANH_C],
                           "bound": bound(None, rel)})


def softplus(lanes: int, *, exp_fn: Fn | None = None, recip_fn: Fn | None = None, inline: bool = False) -> Fn:
    """softplus(x) = log(1 + e^x) of ``lanes`` fp32 lanes: 2^-14.6 for |x| <= 1, 2^-8.8 at |x| = 80 (the error of x
    log2 e grows with |x|; ``exp_fn``: an exp2 of fp32 arguments, default the fp32 one; ``recip_fn``: a recip on [1, 2],
    default 3 steps)."""
    return _logistic("softplus", lanes, exp_fn, recip_fn, inline, 2.3e-3)


def _sigmoid_bf16(lanes: int, exp_fn: Fn | None, inline: bool) -> Fn:
    check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, below="zero", inline=True), func="exp2", lanes=lanes, bias=None,
                   below="zero")
    x = exp(lanes, arg="bfloat16", neg=True, exp_fn=e)
    ulp = None if tag else 0.74
    return make(f"sigmoid_b{lanes}{tag}" + ("_i" if inline else ""), "composite.sigmoid_bf16.cc.jinja", inline=inline,
                deps=(x,), lanes=lanes, arg="bfloat16", ret="bfloat16", func="sigmoid",
                params={"prec": "bf16", "exp_fn": e}, domain=(-87.0, 100.0), valid=(-87.0, 3e38), ulp=ulp,
                variables={"N": lanes, "exp": x.name, "bound": bound(ulp, None)})


def sigmoid(lanes: int, *, prec: str = "fp32", exp_fn: Fn | None = None, recip_fn: Fn | None = None,
            inline: bool = False) -> Fn:
    """sigmoid(x) = 1 / (1 + e^-x) of ``lanes`` lanes.

    ``prec`` "fp32": of fp32 x, as accurate as softplus (``exp_fn``, ``recip_fn``: as softplus').

    ``prec`` "bf16": of bf16 x, rounded to bf16 (a gate): 0.74 bf16 ULP for x in [-87, 100) (0.64 for |x| <= 20); e^-x
    through ``exp``'s bf16 form (``exp_fn``: an exp2 of fp32 arguments with below "zero"; default the bf16-level one,
    inlined), the reciprocal aie::inv."""
    check("prec", prec, ("fp32", "bf16"))
    if prec == "bf16":
        if recip_fn is not None:
            raise ValueError("kmath: sigmoid prec bf16 takes aie::inv, no recip_fn")
        return _sigmoid_bf16(lanes, exp_fn, inline)
    return _logistic("sigmoid", lanes, exp_fn, recip_fn, inline, 1.8e-3)


def silu(lanes: int, *, exp_fn: Fn | None = None, recip_fn: Fn | None = None, inline: bool = False) -> Fn:
    """silu(x) of ``lanes`` bf16 lanes -> fp32 (~16 cycles per element: fp32 math through bf16 products; relative
    error up to 1.7e-3 for x >= 0, 3.9e-3 below). The gather table is ~1 cycle: ``act(SILU)``'s bf16 is off by up to
    2^-8, ``act(SILU, f32=True)`` by 4.8e-4 and 6.9e-4, so a core with room for the table uses that. ``exp_fn``: an exp2 of fp32
    arguments that clamps (the argument reaches -3e38; default the bf16-level one, inlined); ``recip_fn``: a recip on
    [1, 2] (default 2 steps unrolled, inlined)."""
    check("lanes", lanes, LANES)
    e, etag = _take("exp_fn", exp_fn, exp2(lanes, inline=True), func="exp2", lanes=lanes, bias=None,
                    below={"clamp", "zero"})
    r, rtag = _take("recip_fn", recip_fn, recip(lanes, unroll=True), func="recip", lanes=lanes, domain="unit")
    x = exp(lanes, exp_fn=e)
    return make(f"silu_{lanes}{etag}{rtag}" + ("_i" if inline else ""), "composite.silu.cc.jinja", inline=inline,
                deps=(x, r), lanes=lanes, arg="bfloat16", ret="float", func="silu",
                params={"exp_fn": e, "recip_fn": r}, domain=(-80.0, 80.0), valid=(-1e38, 1e38),
                rel=None if etag or rtag else 4.5e-3,
                variables={"N": lanes, "exp": x.name, "recip": r.name})
