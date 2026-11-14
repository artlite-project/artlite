"""Functions built on exp2 and recip: e^-|x|, e^g - 1, softplus, sigmoid, silu (templates: ``composite.*.cc.jinja``).
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


def expneg(lanes: int, *, exp_fn: Fn | None = None, inline: bool = True) -> Fn:
    """e^(-|x|) of ``lanes`` fp32 lanes through ``exp_fn`` (an exp2 of fp32 arguments; default: the fp32 one)."""
    check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"), func="exp2", lanes=lanes, bias=None,
                   below={"clamp", "zero"})
    rel = None if tag else 1.4e-3
    return make(f"expneg_{lanes}{tag}" + ("" if inline else "_o"), "composite.expneg.cc.jinja", inline=inline,
                deps=(e,), lanes=lanes, arg="float", ret="float", func="expneg", params={"exp_fn": e},
                domain=(-87.0, 87.0), valid=(-1e38, 1e38), rel=rel,
                variables={"N": lanes, "exp2": e.name, "bound": bound(None, rel)})


def expm1(lanes: int, *, exp_fn: Fn | None = None, inline: bool = False) -> Fn:
    """e^g - 1 of ``lanes`` fp32 lanes for g <= 0 to 2^-15.6 (``exp_fn``: an exp2 of fp32 arguments, default the
    fp32 one)."""
    check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"), func="exp2", lanes=lanes, bias=None,
                   below={"clamp", "zero"})
    rel = None if tag else 2.0e-5
    return make(f"expm1_{lanes}{tag}" + ("_i" if inline else ""), "composite.expm1.cc.jinja", inline=inline,
                deps=(e,), lanes=lanes, arg="float", ret="float", func="expm1", params={"exp_fn": e},
                domain=(-87.0, 0.0), valid=(-1e38, 0.0), rel=rel,
                variables={"N": lanes, "exp2": e.name, "c": [str(k) for k in EXPM1_C], "bound": bound(None, rel)})


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


def sigmoid(lanes: int, *, exp_fn: Fn | None = None, recip_fn: Fn | None = None, inline: bool = False) -> Fn:
    """sigmoid(x) = 1 / (1 + e^-x) of ``lanes`` fp32 lanes, as accurate as softplus (``exp_fn``, ``recip_fn``: as
    softplus')."""
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
    return make(f"silu_{lanes}{etag}{rtag}" + ("_i" if inline else ""), "composite.silu.cc.jinja", inline=inline,
                deps=(e, r), lanes=lanes, arg="bfloat16", ret="float", func="silu",
                params={"exp_fn": e, "recip_fn": r}, domain=(-80.0, 80.0), valid=(-1e38, 1e38),
                rel=None if etag or rtag else 4.5e-3,
                variables={"N": lanes, "exp2": e.name, "recip": r.name})
