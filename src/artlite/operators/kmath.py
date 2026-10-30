"""Math functions for kernel sources, generated per contract: ``km`` in every kernel template (``JINJA_ENV``).

A function's arguments state what its caller needs (lanes, the argument's form, the precision, what arguments below
the range give, inlined or one out-of-line copy per core) and it returns an ``Fn``: the definition to render before
the first use (``{{ e.src }}``) and the name to call (``{{ e }}``):

    {% set exp2 = km.exp2(64, below="zero") %}
    {{ exp2.src }}
    ...
        const aie::vector<float, 64> e = {{ exp2 }}(aie::sub(s, m));

A contract no implementation meets is an error. The definitions render under one stem (``km``: ``kernel_source``):
equal contracts render equal text, so a fused ELF holds one copy whichever operators ask for it, and a core that
runs several of them one body. They assume ``aie::rounding_mode::conv_even``.

An ``Fn`` also carries its contract (``func``, ``params``), its domain and its error bound there (stated in its
comment): tests/operators/test_kmath.py checks them on the NPU against the variant's numpy model
(tests/operators/kmath_sim.py), bit-exact.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from jinja2 import Environment, StrictUndefined

from .kernel_source import kernel_prefix

KM = kernel_prefix("km")
_ENV = Environment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)  # (JINJA_ENV's options)
LANES = (16, 32, 64)


@dataclass(frozen=True)
class Fn:
    """A generated function: ``name`` (the placeholder name kernels call: ``{{ f }}``), ``src`` (its definition after
    those of the functions it calls, each in an anonymous namespace: ``{{ f.src }}``), its argument and result
    (``arg``, ``ret``: "float" an fp32 vector, "accum" an fp32 accumulator, "bfloat16" a bf16 vector, of ``lanes``),
    the contract that made it (``func``: the kmath function, ``params``: its arguments but lanes and inline) and what
    the tests check: ``valid`` (the arguments it is defined for), ``domain`` (those the bound holds for; ranges are [lo,
    hi)) and the bound (``ulp``: bf16 ULPs of the exact value, ``rel``: relative error). ``parts``: functions of its
    steps, for a caller that splits it (``{{ f.parts.reduce }}``)."""

    name: str
    src: str
    lanes: int
    arg: str
    ret: str
    func: str = ""
    params: dict[str, object] = field(default_factory=dict)
    domain: tuple[float, float] = (-np.inf, np.inf)
    valid: tuple[float, float] = (-np.inf, np.inf)
    ulp: float | None = None
    rel: float | None = None
    parts: dict[str, Fn] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name


def _fn(local: str, text: str, *, inline: bool, deps: tuple[Fn, ...] = (), **meta: object) -> Fn:
    """An Fn named ``km_<local>`` from its definition's template text (``{{ name }}``, ``{{ attr }}``)."""
    name = f"{KM}_{local}"
    attr = "[[clang::always_inline]] inline" if inline else "__attribute__((noinline))"
    own = "namespace {\n\n" + _ENV.from_string(text).render(name=name, attr=attr).strip("\n") + "\n\n}  // namespace\n"
    srcs = dict.fromkeys(d.src for d in deps)  # (each once, in order)
    return Fn(name=name, src="\n".join([*srcs, own]), **meta)


def _check(what: str, value: object, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"kmath: {what} = {value!r}, not one of {allowed}")


def _bound(ulp: float | None, rel: float | None) -> str:
    """A bound as the comments state it (none: the variant's is not measured, e.g. with a caller's exp_fn)."""
    if ulp is not None:
        return f"max {ulp:.2f} bf16 ULP"
    if rel is None:
        return "bound not measured"
    return f"max {rel:.1%}" if rel >= 0.01 else f"max 2^{math.log2(rel):.1f} relative"


# exp2 ---------------------------------------------------------------------------------------------------------------

# (the coefficients: also kmath_sim's)
LN2 = (0.69140625, 0.001739501953125)  # ln 2 as bf16 hi + lo
TAYLOR = tuple(np.float32(1) / np.float32(k) for k in (720, 120, 24, 6, 2, 1, 1))  # e^t to t^6, highest first
CUBIC = (0.24022650695910071, 0.0555041086648216, 0.69314718055994531)  # 2^f ~ 1 + f (c1 + f (c2 + c3 f)): c2, c3, c1
ACC_C = (0.69140625, 0.2412109375, 0.0625)  # 2^f ~ 1 + c1 f + f^2 (c2 + c3 f), bf16 (fitted in this order)
# CHORD2_SCALE 2^x ~ c1 vexp2(x) + c2 vexp2(x + 1/2): c1 = 1 (the caller's 1.0 register shared), c2 = 1 / sqrt 2 (the
# chords' errors, half a period apart, cancel: the SD 1.5 UNet's error 1.19e-2; the pair minimax for |x| <= 2, 1.28e-2)
CHORD2_C = (1.0, 0.70703125)
CHORD2_SCALE = 2.0677  # (the middle of the ratio's range for |x| <= 2)

_EXP2_F32 = r"""
// 2^x of {{ N }} fp32 lanes (kmath.exp2: {{ bound }} for x in [-126, 127.5)): n = round(x) by the 1.5 2^23 split, 2^f
// (|f| <= 1/2) by {{ "e^(f ln 2)'s Taylor polynomial of degree 6 in fp32 products (al::mulff)" if fp32 else "a cubic in bf16 products" }}, n added to the exponent bits.
// {{ below_doc }}
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<float, {{ N }}> x)
{
{% if below != "int" %}
    x = aie::max(x, aie::broadcast<float, {{ N }}>({{ "-127.0f" if below == "zero" else "-126.0f" }}));{{ "  // (-127: 1.0 with the exponent bits cleared)" if below == "zero" else "" }}
{% endif %}
    const aie::vector<float, {{ N }}> magic = aie::broadcast<float, {{ N }}>(12582912.0f);  // 1.5 2^23
    const aie::vector<float, {{ N }}> y = aie::add(x, magic);
    const aie::vector<float, {{ N }}> f = aie::sub(x, aie::sub(y, magic));
    aie::vector<int32, {{ N }}> n = aie::sub(y.template cast_to<int32>(), aie::broadcast<int32, {{ N }}>(0x4B400000));
{% if below == "int" %}
    n = aie::upshift(aie::max(n, aie::broadcast<int32, {{ N }}>(-126)), 23);
{% endif %}
{% if fp32 %}
    const aie::vector<float, {{ N }}> t = al::mul_hilo<{{ N }}, {{ ln2[0] }}f, {{ ln2[1] }}f>(f);  // (f ln 2)
    aie::vector<float, {{ N }}> p = aie::broadcast<float, {{ N }}>({{ taylor[0] }}f);
{% for c in taylor[1:] %}
    p = aie::add(aie::broadcast<float, {{ N }}>({{ c }}f), al::mulff<{{ N }}>(t, p));
{% endfor %}
{% else %}
    const aie::vector<bfloat16, {{ N }}> fb = aie::accum<accfloat, {{ N }}>(f).template to_vector<bfloat16>();
    aie::accum<accfloat, {{ N }}> q2(aie::broadcast<float, {{ N }}>({{ cubic[0] }}f));
    q2 = aie::mac(q2, fb, aie::broadcast<bfloat16, {{ N }}>({{ cubic[1] }}f));
    aie::accum<accfloat, {{ N }}> q1(aie::broadcast<float, {{ N }}>({{ cubic[2] }}f));
    q1 = aie::mac(q1, fb, q2.template to_vector<bfloat16>());
    aie::accum<accfloat, {{ N }}> pa(aie::broadcast<float, {{ N }}>(1.0f));
    const aie::vector<float, {{ N }}> p = aie::add(pa, al::mulc<{{ N }}>(f, q1.template to_vector<bfloat16>())).template to_vector<float>();
{% endif %}
    return aie::add(p.template cast_to<int32>(), {{ "n" if below == "int" else "aie::upshift(n, 23)" }}).template cast_to<float>();
}
"""

_BELOW_F32 = {
    "clamp": "Below -126: 2^-126 (masked lanes: -1e30, -inf).",
    "zero": "Exactly 0 from x <= -127 (masked lanes: -1e30, -inf); between -127 and -126 below 2^-126, inexact.",
    "int": "The exponent clamped as an integer (no fp32 max: arguments above -2^22): 2^(-126 + f) below -126.",
}


# The bias-192 form's first step per contract, each the form its callers' loops schedule best: below "clamp" alone takes
# f from the clamped v (one bf16 max each; prefill/attn's P panels: II 25); with a clamp above or "bounded", f is the
# residual of the unclamped v (a shorter chain: the vision overlay's pexp pipelines, II 23; qwen35's pass 1 II 8), and v
# is clamped at 128 (bf16(A) rounds A in [127.5, 128.5] to 128: v and f agree; vexp2(0) = 1).
_EXP2_ACC_REDUCE = r"""
{% macro reduce(N, out_P, out_f) %}
    const aie::vector<bfloat16, {{ N }}> one = aie::broadcast<bfloat16, {{ N }}>(1.0f);
{% if below == "clamp" and above == "none" %}
    const aie::vector<bfloat16, {{ N }}> v = aie::max(A.template to_vector<bfloat16>(), aie::broadcast<bfloat16, {{ N }}>(129.0f));
    {{ out_f }} = aie::max(aie::msc(A, v, one).template to_vector<bfloat16>(), aie::broadcast<bfloat16, {{ N }}>(-0.5f));
{% elif below == "none" and above == "none" %}
    const aie::vector<bfloat16, {{ N }}> v = A.template to_vector<bfloat16>();
    {{ out_f }} = aie::msc(A, v, one).template to_vector<bfloat16>();
{% else %}
    const aie::vector<bfloat16, {{ N }}> v0 = A.template to_vector<bfloat16>();
{% if above == "clamp" %}
    {{ out_f }} = aie::min(aie::max(aie::msc(A, v0, one).template to_vector<bfloat16>(), aie::broadcast<bfloat16, {{ N }}>(-0.5f)),
                  aie::broadcast<bfloat16, {{ N }}>(0.5f));
{% else %}
    {{ out_f }} = aie::msc(A, v0, one).template to_vector<bfloat16>();
{% endif %}
{% set v = "aie::max(v0, aie::broadcast<bfloat16, " ~ N ~ ">(128.0f))" if below != "none" else "v0" %}
    const aie::vector<bfloat16, {{ N }}> v = {{ "aie::min(" ~ v ~ ", aie::broadcast<bfloat16, " ~ N ~ ">(255.0f))" if above == "clamp" else v }};
{% endif %}
    {{ out_P }} = al::exp2_acc<{{ N }}>(aie::mac(aie::mul(v, one), aie::broadcast<bfloat16, {{ N }}>(-128.0f), one));
{% endmacro %}
{% macro poly(N, P, f) %}
{% set s = "" if shift else " * 0x1p-64f" %}
    const aie::vector<bfloat16, {{ N }}> hi = aie::mac(aie::mul({{ f }}, aie::broadcast<bfloat16, {{ N }}>({{ c[2] }}f{{ s }})),
                                               aie::broadcast<bfloat16, {{ N }}>({{ c[1] }}f{{ s }}), one)
                                          .template to_vector<bfloat16>();
    const aie::vector<bfloat16, {{ N }}> pf = aie::mul({{ P }}, {{ f }}).template to_vector<bfloat16>();
    const aie::vector<bfloat16, {{ N }}> pf2 = aie::mul(pf, {{ f }}).template to_vector<bfloat16>();
    aie::accum<accfloat, {{ N }}> r = aie::mul({{ P }}, {{ "one" if shift else "aie::broadcast<bfloat16, " ~ N ~ ">(0x1p-64f)" }});
    r = aie::mac(r, pf, aie::broadcast<bfloat16, {{ N }}>({{ c[0] }}f{{ s }}));
    return aie::mac(r, pf2, hi);
{% endmacro %}
"""

_EXP2_ACC = _EXP2_ACC_REDUCE + r"""
// 2^{{ "(y + 64)" if shift else "y" }} of {{ N }} lanes, A = y + 192 an fp32 accumulator (kmath.exp2: {{ bound }} for A in [128.5, 255.5)): v =
// bf16(A) = 192 + round(y) ([128, 256) is the binade of ULP 1), f = bf16(A - v); P = vexp2(v - 128) = 2^(round(y) + 64)
// exactly (vexp2 is exact at non-negative integers{{ "" if shift else "; the 2^-64 in the constants" }}), 2^f = 1 + c1 f + f^2 (c2 + c3 f) in
// bf16 products, one rounding at the end. {{ below_doc }}{{ above_doc }}
{{ attr }} aie::accum<accfloat, {{ N }}> {{ name }}(aie::accum<accfloat, {{ N }}> A)
{
    aie::vector<bfloat16, {{ N }}> f;
{{ reduce(N, "const aie::vector<bfloat16, " ~ N ~ "> P", "f") }}{{ poly(N, "P", "f") }}}
"""

_EXP2_ACC_PART_REDUCE = _EXP2_ACC_REDUCE + r"""
// {{ whole }}'s first step: P = 2^(round(y) + 64) (bf16, exact), f = y - round(y) of A = y + 192
{{ attr }} void {{ name }}(aie::accum<accfloat, {{ N }}> A, aie::vector<bfloat16, {{ N }}> &P, aie::vector<bfloat16, {{ N }}> &f)
{
{{ reduce(N, "P", "f") }}}
"""

_EXP2_ACC_PART_POLY = _EXP2_ACC_REDUCE + r"""
// {{ whole }}'s second step: {{ "P 2^f" if shift else "2^-64 P 2^f" }}
{{ attr }} aie::accum<accfloat, {{ N }}> {{ name }}(aie::vector<bfloat16, {{ N }}> P, aie::vector<bfloat16, {{ N }}> f)
{
    const aie::vector<bfloat16, {{ N }}> one = aie::broadcast<bfloat16, {{ N }}>(1.0f);
{{ poly(N, "P", "f") }}}
"""

_BELOW_ACC = {
    ("clamp", "none"): "x below -63.5 (masked lanes too, -inf included): v = 129, f = -1/2 (~2^-63.5).",
    ("clamp", "clamp"): "x below -64.5 (masked lanes too, finite): v = 128, f in [-1/2, 1/2] (~2^-64).",
    ("bounded", "none"): "x below -64.5 down to A = -2^15 (masked lanes as -16384): v = 128, f the residual (|2^x| < "
                         "2^-49, its sign not kept).",
    ("none", "none"): "A below 128.5 is the caller's to prevent.",
}


_EXP2_CHORD = r"""
// {{ "%g " % scale if scale != 1 else "" }}2^x of {{ N }} lanes, x an fp32 accumulator (kmath.exp2: {{ bound }} for x in [-2, 2], {{ wider }}):
// {{ "the chords of vexp2 at x and x + 1/2, the second times 1 / sqrt 2" if two else "vexp2" }}, Schraudolph's exp: 2^n (1 + f) of x truncated to bf16 (n + f, f
// to 7 bits; negative x in one's complement), so in steps of 2^-6 .. 1/4 from |x| = 2 .. 32
{{ attr }} aie::{{ "accum<accfloat" if two else "vector<bfloat16" }}, {{ N }}> {{ name }}(aie::accum<accfloat, {{ N }}> A)
{
{% if two %}
    return aie::mac(aie::mul(al::exp2_acc<{{ N }}>(A), aie::broadcast<bfloat16, {{ N }}>({{ c[0] }}f)),
                    al::exp2_acc<{{ N }}>(aie::add(A, aie::broadcast<float, {{ N }}>(0.5f))), aie::broadcast<bfloat16, {{ N }}>({{ c[1] }}f));
{% else %}
    return al::exp2_acc<{{ N }}>(A);
{% endif %}
}
"""


def exp2(lanes: int, *, bias: int | None = None, prec: str = "bf16", below: str = "clamp", above: str = "none",
         shift: int = 0, inline: bool = False) -> Fn:
    """2^x of ``lanes`` lanes.

    ``bias=0``: x an fp32 accumulator, through vexp2 (the LUT: x truncated to bf16), coarse: ``prec`` "chord" (one
    vexp2 -> a bf16 vector, 2^x to +6.2% / -0.6% for |x| <= 2) or "chord2" (two, weighted -> an fp32 accumulator,
    2.068 2^x to 1.2% for |x| <= 2: a softmax's P, whose scale cancels).

    ``bias=None``: x an fp32 vector -> an fp32 vector, x in [-126, 127.5); ``prec`` "bf16" (a cubic: 0.4 bf16 ULP) or
    "fp32" (degree 6 in fp32 products: 2^-16.5); ``below``: "clamp" (2^-126 below -126, masked lanes too), "zero"
    (exactly 0 from x <= -127), "int" (the exponent clamped as an integer: no fp32 max; arguments above -2^22).

    ``bias=192``: A = x + 192 an fp32 accumulator -> an fp32 accumulator, bf16-level, x in [-63.5, 63.5) (vexp2 builds
    2^round(x): no fp32 vector arithmetic, pipelines); ``below``: "clamp" (x below -63.5, masked lanes too: ~2^-63),
    "bounded" (A >= -2^15, masked lanes as -16384: |2^x| < 2^-49 below, its sign not kept; one bf16 max less) or
    "none" (the caller keeps A >= 128.5); ``above``: "none" (the caller keeps A < 255.5) or "clamp" (saturates at
    ~2^63); ``shift`` 64: 2^(x + 64) (a softmax's scale cancels; one constant register less). Its ``parts``:
    ``reduce`` (A -> P, f) and ``poly`` (P, f -> 2^x), for a caller that stores P and f between two loops."""
    _check("lanes", lanes, LANES)
    _check("bias", bias, (None, 0, 192))
    params = {"bias": bias, "prec": prec, "below": below, "above": above, "shift": shift}
    if bias == 0:
        _check("prec", prec, ("chord", "chord2"))
        _check("below", below, ("clamp",))
        _check("above", above, ("none",))
        _check("shift", shift, (0,))
        two = prec == "chord2"
        rel = 1.2e-2 if two else 6.2e-2
        wider = "2.0% for |x| <= 4, 3.6% for |x| <= 8" if two else "7.9% for |x| <= 8"
        text = _ENV.from_string(_EXP2_CHORD).render(N=lanes, two=two, c=CHORD2_C, scale=CHORD2_SCALE if two else 1,
                                                    bound=_bound(None, rel), wider=wider, name="{{ name }}",
                                                    attr="{{ attr }}")
        local = f"exp2_{prec}_{lanes}" + ("_i" if inline else "")
        return _fn(local, text, inline=inline, lanes=lanes, arg="accum",
                   ret="accum" if two else "bfloat16", func="exp2", params=params, domain=(-2.0, 2.0),
                   valid=(-120.0, 120.0), rel=rel)
    if bias is None:
        _check("prec", prec, ("bf16", "fp32"))
        _check("below", below, ("clamp", "zero", "int") if prec == "bf16" else ("clamp",))
        _check("above", above, ("none",))
        _check("shift", shift, (0,))
        fp32 = prec == "fp32"
        ulp, rel = (None, 1.1e-5) if fp32 else (0.4, None)
        local = f"exp2_v{lanes}" + ("_fp32" if fp32 else "") + ("" if below == "clamp" else f"_{below}")
        local += "_i" if inline else ""
        text = _ENV.from_string(_EXP2_F32).render(N=lanes, fp32=fp32, below=below, below_doc=_BELOW_F32[below],
                                                  bound=_bound(ulp, rel), ln2=LN2, taylor=[str(c) for c in TAYLOR],
                                                  cubic=CUBIC, name="{{ name }}", attr="{{ attr }}")
        return _fn(local, text, inline=inline, lanes=lanes, arg="float", ret="float", func="exp2", params=params,
                   domain=(-126.0, 127.5), valid=(-2.0 ** 22 if below == "int" else -np.inf, 127.5), ulp=ulp, rel=rel)
    _check("prec", prec, ("bf16",))
    _check("below", below, ("clamp", "bounded", "none"))
    _check("above", above, ("none", "clamp"))
    _check("shift", shift, (0, 64))
    local = f"exp2_a{lanes}" + {"clamp": "", "bounded": "_bounded", "none": "_unclamped"}[below]
    local += ("_sat" if above == "clamp" else "") + (f"_s{shift}" if shift else "")
    above_doc = " x from 63.5 on: v = 255, f = 1/2 (~2^63)." if above == "clamp" else ""
    kw = {"N": lanes, "below": below, "above": above, "shift": shift, "c": ACC_C}
    below_doc = _BELOW_ACC.get((below, above)) or _BELOW_ACC[below, "none"].replace("-inf included", "finite")
    text = _ENV.from_string(_EXP2_ACC).render(**kw, bound=_bound(0.26, None), below_doc=below_doc,
                                              above_doc=above_doc, name="{{ name }}", attr="{{ attr }}")
    # (arguments the result is defined for: the residual forms make a NaN of an infinite one)
    lo = {"clamp": -np.inf if above == "none" else -2.0 ** 100, "bounded": -2.0 ** 15, "none": 128.5}[below]
    fn = _fn(local + ("_i" if inline else ""), text, inline=inline, lanes=lanes, arg="accum", ret="accum", func="exp2",
             params=params, domain=(128.5, 255.5), valid=(lo, 2.0 ** 100 if above == "clamp" else 255.5), ulp=0.26)
    poly = f"exp2_a{lanes}_poly" + (f"_s{shift}" if shift else "")
    parts = {  # (the same for the inlined whole)
        "reduce": _fn(local + "_reduce", _ENV.from_string(_EXP2_ACC_PART_REDUCE).render(
            **kw, whole=local, name="{{ name }}", attr="{{ attr }}"), inline=True, lanes=lanes, arg="accum", ret=""),
        "poly": _fn(poly, _ENV.from_string(_EXP2_ACC_PART_POLY).render(
            **kw, whole=poly.replace("_poly", ""), name="{{ name }}", attr="{{ attr }}"), inline=True, lanes=lanes,
            arg="", ret="accum"),
    }
    return Fn(**{**fn.__dict__, "parts": parts})


# Reciprocal and functions built on exp2 ---------------------------------------------------------------------------

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


_RECIP = r"""
// 1 / x of {{ N }} fp32 lanes (kmath.recip: {{ bound }} for x in {{ dom }}): {{ "the mantissa's reciprocal: " if normal else "" }}a linear guess on [1, 2] (6%,
// its slope a bf16 product), {{ steps }} Newton steps in fp32 products (al::mulff){{ ", the exponent negated in the bits (no scalar division: a\n// soft-float call)" if normal else "" }}
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<float, {{ N }}> x)
{
{% if normal %}
    const aie::vector<int32, {{ N }}> b = x.template cast_to<int32>();
    const aie::vector<int32, {{ N }}> e = aie::sub(aie::bit_and(b, aie::broadcast<int32, {{ N }}>(0x7F800000)),
                                             aie::broadcast<int32, {{ N }}>(0x3F800000));
    const aie::vector<float, {{ N }}> m = aie::sub(b, e).template cast_to<float>();  // (in [1, 2))
{% set m = "m" %}
{% else %}
{% set m = "x" %}
{% endif %}
    const aie::vector<float, {{ N }}> two = aie::broadcast<float, {{ N }}>(2.0f);
    aie::vector<float, {{ N }}> r = aie::sub(aie::broadcast<float, {{ N }}>(1.4117647f),
                                     al::mulc<{{ N }}>({{ m }}, aie::broadcast<bfloat16, {{ N }}>(0.47058824f)).template to_vector<float>());
{% if unroll %}
{% for i in range(steps) %}
    r = al::mulff<{{ N }}>(r, aie::sub(two, al::mulff<{{ N }}>({{ m }}, r)));
{% endfor %}
{% else %}
    for (int i = 0; i < {{ steps }}; i++)
        r = al::mulff<{{ N }}>(r, aie::sub(two, al::mulff<{{ N }}>({{ m }}, r)));
{% endif %}
    return {{ "aie::sub(r.template cast_to<int32>(), e).template cast_to<float>()" if normal else "r" }};
}
"""

# recip's bound per steps (relative; the Newton steps square a 1/17 guess's error, the fp32 products' rounding stays)
_RECIP_REL = {2: 4.1e-5, 3: 1.5e-5}


def recip(lanes: int, *, domain: str = "unit", steps: int = 2, unroll: bool = False, inline: bool = True) -> Fn:
    """1 / x of ``lanes`` fp32 lanes to ~2^-16: ``domain`` "unit" (x in [1, 2]) or "normal" (normal x > 0: the
    mantissa's reciprocal, the exponent negated in the bits); ``steps`` Newton steps after a linear guess (2 or 3), a
    loop (each step needs the last: not pipelined; ~80 bundles less than unrolled at the same cycles in gdn's
    softplus, sigmoid) or ``unroll``ed (inlined in silu: ~7% faster)."""
    _check("lanes", lanes, LANES)
    _check("domain", domain, ("unit", "normal"))
    _check("steps", steps, (2, 3))
    rel = _RECIP_REL[steps]
    dom = (1.0, 2.0) if domain == "unit" else (2.0 ** -126, 2.0 ** 125)
    text = _ENV.from_string(_RECIP).render(N=lanes, normal=domain == "normal", steps=steps, unroll=unroll,
                                           bound=_bound(None, rel),
                                           dom="[1, 2]" if domain == "unit" else "[2^-126, 2^125]", name="{{ name }}",
                                           attr="{{ attr }}")
    local = f"recip_{lanes}" + ("_normal" if domain == "normal" else "") + f"_n{steps}" + ("_u" if unroll else "")
    local += "" if inline else "_o"
    return _fn(local, text, inline=inline, lanes=lanes, arg="float", ret="float", func="recip",
               params={"domain": domain, "steps": steps, "unroll": unroll}, domain=(dom[0], np.nextafter(dom[1], np.inf)),
               valid=(dom[0], np.nextafter(dom[1], np.inf)), rel=rel)


_EXPNEG = r"""
// e^(-|x|) of {{ N }} fp32 lanes (kmath.expneg, in (0, 1]: {{ bound }} for x in [-87, 87]; the error of x log2 e, ~2^-16
// relative, grows with |x|: 2^-15.8 for |x| <= 1)
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<float, {{ N }}> x)
{
    const aie::vector<float, {{ N }}> ax = aie::max(x, aie::sub(aie::zeros<float, {{ N }}>(), x));
    return {{ exp2 }}(al::log2e<{{ N }}>(aie::sub(aie::zeros<float, {{ N }}>(), ax)));
}
"""


def expneg(lanes: int, *, exp_fn: Fn | None = None, inline: bool = True) -> Fn:
    """e^(-|x|) of ``lanes`` fp32 lanes through ``exp_fn`` (an exp2 of fp32 arguments; default: the fp32 one)."""
    _check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"), func="exp2", lanes=lanes, bias=None,
                   below={"clamp", "zero"})
    rel = None if tag else 1.4e-3
    text = _ENV.from_string(_EXPNEG).render(N=lanes, exp2=e.name, bound=_bound(None, rel), name="{{ name }}",
                                            attr="{{ attr }}")
    return _fn(f"expneg_{lanes}{tag}" + ("" if inline else "_o"), text, inline=inline, deps=(e,), lanes=lanes,
               arg="float", ret="float", func="expneg", params={"exp_fn": e}, domain=(-87.0, 87.0),
               valid=(-1e38, 1e38), rel=rel)


_EXPM1 = r"""
// e^g - 1 of {{ N }} fp32 lanes for g <= 0 (kmath.expm1: {{ bound }} for g in [-87, 0]): its Taylor polynomial in fp32
// products below |g| = 1/2 (no cancellation), else 2^(g log2 e) - 1
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<float, {{ N }}> g)
{
    aie::vector<float, {{ N }}> p = aie::broadcast<float, {{ N }}>({{ c[0] }}f);
{% for k in c[1:] %}
    p = aie::add(aie::broadcast<float, {{ N }}>({{ k }}f), al::mulff<{{ N }}>(g, p));
{% endfor %}
    const aie::vector<float, {{ N }}> small = aie::add(g, al::mulff<{{ N }}>(al::mulff<{{ N }}>(g, g), p));
    const aie::vector<float, {{ N }}> big = aie::sub({{ exp2 }}(al::log2e<{{ N }}>(g)), aie::broadcast<float, {{ N }}>(1.0f));
    return aie::select(big, small, aie::gt(g, aie::broadcast<float, {{ N }}>(-0.5f)));
}
"""
EXPM1_C = tuple(np.float32(1) / np.float32(k) for k in (5040, 720, 120, 24, 6, 2))  # (e^g - 1 - g) / g^2, highest first


def expm1(lanes: int, *, exp_fn: Fn | None = None, inline: bool = False) -> Fn:
    """e^g - 1 of ``lanes`` fp32 lanes for g <= 0 to 2^-15.6 (``exp_fn``: an exp2 of fp32 arguments, default the
    fp32 one)."""
    _check("lanes", lanes, LANES)
    e, tag = _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"), func="exp2", lanes=lanes, bias=None,
                   below={"clamp", "zero"})
    rel = None if tag else 2.0e-5
    text = _ENV.from_string(_EXPM1).render(N=lanes, exp2=e.name, c=[str(k) for k in EXPM1_C],
                                           bound=_bound(None, rel), name="{{ name }}", attr="{{ attr }}")
    return _fn(f"expm1_{lanes}{tag}" + ("_i" if inline else ""), text, inline=inline, deps=(e,), lanes=lanes,
               arg="float", ret="float", func="expm1", params={"exp_fn": e}, domain=(-87.0, 0.0),
               valid=(-1e38, 0.0), rel=rel)


_SOFTPLUS = r"""
// softplus(x) = max(x, 0) + log1p(e^(-|x|)) of {{ N }} fp32 lanes (kmath.softplus: {{ bound }} for x in [-80, 80], 2^-14.6 for
// |x| <= 1: as kmath.expneg's): log1p(t) = 2 atanh(s), s = t / (2 + t) <= 1/3, its odd series to s^11 in fp32 products
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<float, {{ N }}> x)
{
    const aie::vector<float, {{ N }}> t = {{ expneg }}(x);
    const aie::vector<float, {{ N }}> half = aie::broadcast<float, {{ N }}>(0.5f);
    // 1 / (2 + t) = (1 / (1 + t / 2)) / 2
    const aie::vector<float, {{ N }}> r = al::mulff<{{ N }}>({{ recip }}(aie::add(aie::broadcast<float, {{ N }}>(1.0f), al::mulff<{{ N }}>(t, half))), half);
    const aie::vector<float, {{ N }}> s = al::mulff<{{ N }}>(t, r);
    const aie::vector<float, {{ N }}> s2 = al::mulff<{{ N }}>(s, s);
    aie::vector<float, {{ N }}> p = aie::broadcast<float, {{ N }}>({{ c[0] }}f);
{% for k in c[1:] %}
    p = aie::add(aie::broadcast<float, {{ N }}>({{ k }}f), al::mulff<{{ N }}>(s2, p));
{% endfor %}
    const aie::vector<float, {{ N }}> l = al::mulff<{{ N }}>(aie::add(s, s), p);
    return aie::add(aie::max(x, aie::zeros<float, {{ N }}>()), l);
}
"""
ATANH_C = tuple(np.float32(1) / np.float32(k) for k in (11, 9, 7, 5, 3, 1))  # atanh(s) / s in s^2, highest first

_SIGMOID = r"""
// sigmoid(x) of {{ N }} fp32 lanes (kmath.sigmoid: {{ bound }} for x in [-80, 80], 2^-15.0 for |x| <= 1: as kmath.expneg's): r
// or t r (x < 0), t = e^(-|x|), r = 1 / (1 + t)
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<float, {{ N }}> x)
{
    const aie::vector<float, {{ N }}> t = {{ expneg }}(x);
    const aie::vector<float, {{ N }}> r = {{ recip }}(aie::add(aie::broadcast<float, {{ N }}>(1.0f), t));
    return aie::select(r, al::mulff<{{ N }}>(t, r), aie::lt(x, aie::zeros<float, {{ N }}>()));
}
"""


def _logistic(func: str, text: str, lanes: int, exp_fn: Fn | None, recip_fn: Fn | None, inline: bool,
              rel: float) -> Fn:
    _check("lanes", lanes, LANES)
    t = expneg(lanes, exp_fn=exp_fn)
    r, rtag = _take("recip_fn", recip_fn, recip(lanes, steps=3), func="recip", lanes=lanes, domain="unit")
    etag = "" if exp_fn is None else _take("exp_fn", exp_fn, exp2(lanes, prec="fp32"))[1]
    rel = None if etag or rtag else rel
    src = _ENV.from_string(text).render(N=lanes, expneg=t.name, recip=r.name, c=[str(k) for k in ATANH_C],
                                        bound=_bound(None, rel), name="{{ name }}", attr="{{ attr }}")
    return _fn(f"{func}_{lanes}{etag}{rtag}" + ("_i" if inline else ""), src, inline=inline, deps=(t, r),
               lanes=lanes, arg="float", ret="float", func=func, params={"expneg": t, "recip_fn": r},
               domain=(-80.0, 80.0), valid=(-1e38, 1e38), rel=rel)


def softplus(lanes: int, *, exp_fn: Fn | None = None, recip_fn: Fn | None = None, inline: bool = False) -> Fn:
    """softplus(x) = log(1 + e^x) of ``lanes`` fp32 lanes: 2^-14.6 for |x| <= 1, 2^-8.8 at |x| = 80 (the error of x
    log2 e grows with |x|; ``exp_fn``: an exp2 of fp32 arguments, default the fp32 one; ``recip_fn``: a recip on [1, 2],
    default 3 steps)."""
    return _logistic("softplus", _SOFTPLUS, lanes, exp_fn, recip_fn, inline, 2.3e-3)


def sigmoid(lanes: int, *, exp_fn: Fn | None = None, recip_fn: Fn | None = None, inline: bool = False) -> Fn:
    """sigmoid(x) = 1 / (1 + e^-x) of ``lanes`` fp32 lanes, as accurate as softplus (``exp_fn``, ``recip_fn``: as
    softplus')."""
    return _logistic("sigmoid", _SIGMOID, lanes, exp_fn, recip_fn, inline, 1.8e-3)


_SILU = r"""
// silu(x) = x sigmoid(x) of {{ N }} bf16 lanes in fp32 (kmath.silu): t = e^-|x| (2^x to bf16), r = 1 / (1 + t), sigmoid = x
// >= 0 ? r : t r. Rounded to bf16: 98.7% correctly rounded, max 1.17 ULPs (x < 0, where t's bf16-level error passes
// through).
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<bfloat16, {{ N }}> x)
{
    aie::vector<float, {{ N }}> xf = aie::accum<accfloat, {{ N }}>(x).template to_vector<float>();
    aie::vector<float, {{ N }}> nax = aie::min(xf, aie::sub(aie::zeros<float, {{ N }}>(), xf));
    aie::vector<bfloat16, {{ N }}> tb = aie::accum<accfloat, {{ N }}>({{ exp2 }}(al::log2e<{{ N }}>(nax))).template to_vector<bfloat16>();
    aie::vector<float, {{ N }}> t = aie::accum<accfloat, {{ N }}>(tb).template to_vector<float>();
    aie::vector<float, {{ N }}> r = {{ recip }}(aie::add(t, aie::broadcast<float, {{ N }}>(1.0f)));
    aie::vector<float, {{ N }}> sg = aie::select(r, al::mulff<{{ N }}>(t, r), aie::lt(xf, aie::zeros<float, {{ N }}>()));
    return al::mulff<{{ N }}>(xf, sg);
}
"""


def silu(lanes: int, *, exp_fn: Fn | None = None, recip_fn: Fn | None = None, inline: bool = False) -> Fn:
    """silu(x) of ``lanes`` bf16 lanes -> fp32 (~16 cycles per element: fp32 math through bf16 products; the gather
    table's SiLU, ``act(SILU)``, is ~1 but rounds to bf16). For fp32 consumers: ~5e-4 relative for x >= 0, where the
    table's bf16 result is off by up to 2^-8 (the qwen35 decode GDN state of test_gdn_steps_27b: 3.9e-3 through the
    table, bound 2e-3). ``exp_fn``: an exp2 of fp32 arguments that clamps (the argument reaches -3e38;
    default the bf16-level one, inlined); ``recip_fn``: a recip on [1, 2] (default 2 steps unrolled, inlined)."""
    _check("lanes", lanes, LANES)
    e, etag = _take("exp_fn", exp_fn, exp2(lanes, inline=True), func="exp2", lanes=lanes, bias=None,
                    below={"clamp", "zero"})
    r, rtag = _take("recip_fn", recip_fn, recip(lanes, unroll=True), func="recip", lanes=lanes, domain="unit")
    text = _ENV.from_string(_SILU).render(N=lanes, exp2=e.name, recip=r.name, name="{{ name }}", attr="{{ attr }}")
    return _fn(f"silu_{lanes}{etag}{rtag}" + ("_i" if inline else ""), text, inline=inline, deps=(e, r), lanes=lanes,
               arg="bfloat16", ret="float", func="silu", params={"exp_fn": e, "recip_fn": r}, domain=(-80.0, 80.0),
               valid=(-1e38, 1e38), rel=None if etag or rtag else 4.5e-3)


# Activations through the gather LUT -----------------------------------------------------------------------------------
#
# AIE2P's gather (``load_4x64_lo / hi`` = ``vldb.4x64``: four 64-bit table reads per instruction, ldb slot; probed on
# the NPU): lane k of an instruction reads table ab (k % 4 < 2) or cd, its odd lanes 32 bytes further; offset bits 5:4
# select the entry of a 64-byte group, bits 3:0 are ignored. A table is stored as groups of 4 entries, each group twice
# (64 bytes per 4 segments), in two copies (ab, cd) in two 16 KB regions: 32 bytes per segment, 3.2 KB for SiLU (in one
# region every gather waits a cycle for a bank: SiLU 4227 vs 3287 cycles per 4096 values). Entries: [c2 bf16 | c1 bf16 | c0 fp32] (``shuffle(.., T16_16x4_lo)`` gives c2 | c1 per 16 lanes,
# ``T32_16x2_hi`` c0).
#
# SiLU: silu(x) = relu(x) + g(t), g(t) = silu(-t), t = min(|x|, 24) (silu(x) - silu(-x) = x). g is a table of 97
# segments of 1/4 over [0, 24], each a quadratic in the offset r from its centre: c0 + c1 r + c2 r^2 (c0 fp32, c1 and c2
# bf16; minimax over the segment's bf16 points, in units of the result's bf16 ULP):
#     v = bf16(t + 32)              t rounded to the 1/4 grid ([32, 56] is a binade of ULP 1/4)
#     r = bf16((32 - v) + t)        exact ((32 - v) first: tiny t stay exact)
#     y = bf16(c0 + c1 r + c2 bf16(r^2) + max(x, 0))
#     table byte offset = 64 v - 2048 (``bfloat16_to_int(v, 6)``: ``vfloor``)
# Max 0.59 bf16 ULP over [-24, inf), 99.84% correctly rounded (below -24: silu(-24), within 1e-9 of the exact value).
# The vtanh / vexp2 form: 6.6 ULP; ``silu``: 1.17 ULP at ~16 cycles per element. 3287 cycles per 4096 elements on the
# NPU (the vtanh / vexp2 form: 2682 per 1024), in two loops: t into y (II 2), then the table pass in place (II 22 per 32
# lanes; bound 9: 8 table reads, the shuffles; below 22 the post-pipeliner's register allocation fails). Fewer, longer
# loops ran slower: the constant vectors (1, 24, the abs mask, 0, the lanes' table pointers) take 5 of the 12 x
# registers.
#
# Sigmoid (``full``: the function itself over [-16, 16], x + 48 in the binade [32, 64] of ULP 1/4: 129 segments, no
# symmetry, x clamped to the range): within 0.6 bf16 ULP, ~1 cycle per element (an fp32 exp2 and reciprocal: ~5).


def _bf16(x: object) -> np.ndarray:
    """fp32 -> bf16 rounded to nearest even, as fp32 (finite values)."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


def _f32(x: object) -> np.ndarray:
    return np.asarray(x, np.float32)


def bf16_values() -> np.ndarray:
    """Every finite bf16 value, as float32 (bit patterns 0x8000 .. 0xFFFF, then 0x0000 .. 0x7FFF)."""
    x = (np.arange(-32768, 32768).astype(np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return x[np.isfinite(x)]


def silu64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / (1.0 + np.exp(np.minimum(-x, 700.0)))


def sigmoid64(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.minimum(-np.asarray(x, np.float64), 700.0)))


def gelu64(x: np.ndarray) -> np.ndarray:
    """gelu in its tanh form (gelu_pytorch_tanh, ggml_gelu) as x sigmoid(z), z = 2 sqrt(2 / pi) (x + 0.044715 x^3)."""
    x = np.asarray(x, np.float64)
    z = 2.0 * np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return x / (1.0 + np.exp(np.minimum(-z, 700.0)))


def gelu_erf64(x: np.ndarray) -> np.ndarray:
    """gelu in its erf form (x Phi(x): MiniCPM-V's final merger)."""
    x = np.asarray(x, np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def ulp_bf16(t: np.ndarray) -> np.ndarray:
    """The bf16 ULP at t (normal range)."""
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(t), 2.0 ** -126))) - 7)


def _minimax(cols: list[np.ndarray], f: np.ndarray, w: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Coefficients c of sum c_k col_k minimizing max |w (sum c_k col_k + fixed - f)| (Lawson's iteration)."""
    A = np.stack(cols, 1) * w[:, None]
    b = (f - fixed) * w
    u = np.full(len(f), 1.0 / len(f))
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(300):
        s = np.sqrt(u)
        c = np.linalg.lstsq(A * s[:, None], b * s, rcond=None)[0]
        e = np.abs(A @ c - b)
        if e.max() <= 1e-9 * max(np.abs(b).max(), 1e-30):
            break
        u = u * e
        u /= u.sum()
    return c


def _round_bf16(c: float) -> float:
    return float(_bf16(np.float32([c]))[0])


@dataclass(frozen=True)
class ActTable:
    """An activation f with f(x) - f(-x) = x (x sigmoid of an odd function: SiLU, GELU) as relu(x) + g(t), g(t) =
    f(-t), t = min(|x|, t_max), g a table of segments of ``seg`` over [0, t_max] (t + m in one binade of ULP seg).
    ``full``: f itself over [-t_max, t_max] (x clamped; x + m in the binade [m - t_max, m + t_max]: sigmoid).
    ``floor``: the error that counts in units of the result's ULP is at least this absolute one (GELU's tail, below
    1e-7 at t = 5, decays faster than a quadratic per segment follows; relative accuracy there is irrelevant)."""

    name: str
    f64: Callable[[np.ndarray], np.ndarray]
    seg: float
    m: float
    t_max: float
    floor: float = 0.0
    full: bool = False

    @property
    def base(self) -> float:
        """The reduced argument's binade start (its first segment's)."""
        return self.m - self.t_max if self.full else self.m

    @property
    def segs(self) -> int:
        return int((2 if self.full else 1) * self.t_max / self.seg) + 1

    @property
    def groups(self) -> int:
        return -(-self.segs // 4)

    def reduce(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The kernel's v, r, bf16(r^2) of t (bf16 values in [0, t_max])."""
        v = _bf16(_f32(self.m) + _f32(t))
        r = _bf16(_f32(_f32(self.m) - v) + _f32(t))
        return v, r, _bf16(_f32(r) * _f32(r))

    @functools.cached_property
    def coefficients(self) -> np.ndarray:
        """(segs, 3): c0 (fp32 values), c1, c2 (bf16 values) of g's segments (cached on disk: ~1 s to fit, keyed by the
        table's parameters and the fitting code)."""
        from .base import CACHE_DIR  # (base imports this module)

        code = [inspect.getsource(f) for f in (self.f64, ActTable._fit, ActTable.reduce, _minimax, _round_bf16, _bf16, _f32,
                                               bf16_values, ulp_bf16)]
        key = hashlib.sha256(repr((self.name, self.seg, self.m, self.t_max, self.floor, self.full, code)).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, "kmath", f"{self.name}_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        coef = self._fit()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.npy"
        np.save(tmp, coef)
        os.replace(tmp, path)
        return coef

    def _fit(self) -> np.ndarray:
        x = bf16_values()
        if self.full:
            t = x[(np.abs(x) <= self.t_max) & ((x == 0) | (np.abs(x) >= 2.0 ** -40))]
            v, r, r2 = self.reduce(t)
            f = self.f64(t)
            w = 1.0 / np.maximum(ulp_bf16(f), self.floor)
        else:
            t = -x[(x <= 0) & (x >= -self.t_max) & ((x == 0) | (np.abs(x) >= 2.0 ** -40))]
            v, r, r2 = self.reduce(t)
            f = self.f64(-t)
            # (the error of y = relu(x) + g on both sides)
            w = 1.0 / np.maximum(np.minimum(ulp_bf16(f), ulp_bf16(self.f64(t))), self.floor)
        s = ((v - self.base) / self.seg).astype(np.int64)
        coef = np.zeros((self.segs, 3))
        for k in range(self.segs):
            sel = (s == k) & ((t != 0) | self.full)
            rk, r2k, fk, wk = r[sel].astype(np.float64), r2[sel].astype(np.float64), f[sel], w[sel]
            ones, zero = np.ones_like(rk), np.zeros_like(fk)
            if k == 0 and not self.full:  # c0 = 0: tiny t stay relative-accurate
                c2 = _round_bf16(_minimax([rk, r2k], fk, wk, zero)[1])
                c1 = _round_bf16(_minimax([rk], fk, wk, c2 * r2k)[0])
                coef[k] = 0.0, c1, c2
                continue
            c2 = _round_bf16(_minimax([ones, rk, r2k], fk, wk, zero)[2])
            c1 = _round_bf16(_minimax([ones, rk], fk, wk, c2 * r2k)[1])
            c0 = _minimax([ones], fk, wk, c1 * rk + c2 * r2k)[0]
            coef[k] = float(np.float32(c0)), c1, c2
        return coef

    def table_words(self, coefficients: np.ndarray | None = None) -> np.ndarray:
        """One table copy as uint32 words: entries [c2 | c1 (high half), c0], groups of 4 entries stored twice (the
        gather's odd lanes read 32 bytes further); of ``coefficients`` (default: the table's)."""
        coef = self.coefficients if coefficients is None else coefficients

        def bits(c: np.ndarray) -> np.ndarray:
            return _bf16(_f32(c)).view(np.uint32) >> 16

        words = np.zeros((self.groups * 4, 2), np.uint32)
        words[:len(coef), 0] = bits(coef[:, 1]) << 16 | bits(coef[:, 2])
        words[:len(coef), 1] = _f32(coef[:, 0]).view(np.uint32)
        g = words.reshape(self.groups, 8)
        return np.concatenate([g, g], 1).reshape(-1)


SILU = ActTable("silu", silu64, 0.25, 32.0, 24.0)
# GELU: 65 segments of 1/8 over [0, 8] (g(-8) ~ -4e-21): within 0.59 bf16 ULP from x = -4 on, 1.42 in [-6, -4) (absolute
# 1.4e-7), below: within 6e-13; the tanh form through vtanh: 1.7 ULP at ~7 cycles per element
GELU = ActTable("gelu", gelu64, 0.125, 16.0, 8.0)
GELU_ERF = ActTable("gelu_erf", gelu_erf64, 0.125, 16.0, 8.0)
SIGMOID = ActTable("sigmoid", sigmoid64, 0.25, 48.0, 16.0, full=True)
TABLES = {t.name: t for t in (SILU, GELU, GELU_ERF, SIGMOID)}

def _walk(walk: tuple[int, int], ptr: str, counter: str) -> dict[str, str]:
    """A pointer's walk over x in groups of ``walk[0]`` vectors of 32 values, ``walk[1]`` values apart (a gate/up C
    tile's gate halves: (2, 128)): its counter's declaration and the step after each vector."""
    count, period = walk
    if count < 1 or period % 32 or period < 32 * count:
        raise ValueError(f"kmath: walk = {walk}: groups of whole vectors (period a multiple of 32, >= 32 x count)")
    if count == 1:
        return {"decl": "", "step": f"{ptr} += {period};"}
    return {"decl": f"    int {counter} = 0;", "step": f"{ptr} = add_2d_ptr({ptr}, {period - 32 * (count - 1)}, {count - 1}, {counter}, 32);"}


def _walk_tag(walk: tuple[int, int]) -> str:
    return "" if walk == (1, 32) else f"_w{walk[0]}x{walk[1]}"


_LUT16 = r"""
// 16 table reads: lanes 0-7 into q0, 8-15 into q1
{{ attr }} void {{ name }}(v16int32 ptr, v32bfloat16 &q0, v32bfloat16 &q1)
{
    q0 = insert(q0, 0, (v16bfloat16)load_4x64_lo(extract_v8int32(ptr, 0)));
    q0 = insert(q0, 1, (v16bfloat16)load_4x64_hi(extract_v8int32(ptr, 0)));
    q1 = insert(q1, 0, (v16bfloat16)load_4x64_lo(extract_v8int32(ptr, 1)));
    q1 = insert(q1, 1, (v16bfloat16)load_4x64_hi(extract_v8int32(ptr, 1)));
}
"""


def _lut16() -> Fn:
    return _fn("lut16", _LUT16, inline=True, lanes=16, arg="ptr", ret="void")


_TABLE_PASS = r"""
{% if not table_arg %}
alignas(64) static const uint32_t {{ '{{ name }}' }}_ab[{{ words | length }}] __attribute__((section(".aie.bank0.{{ '{{ name }}' }}"))) = { {{ words | join(", ") }} };
alignas(64) static const uint32_t {{ '{{ name }}' }}_cd[{{ words | length }}] __attribute__((section(".aie.bank1.{{ '{{ name }}' }}"))) = { {{ words | join(", ") }} };

{% endif %}
// y = c0 + c1 r + c2 r^2{{ " + max(x, 0) of t = min(|x|, %d)" % (t_max | float | int) if not full else " of t = x clamped to +-%d" % (t_max | float | int) }} (kmath.table_pass, {{ table }}'s table; in t_ and t2_: y, read before it
// is written){{ "; x: groups of %d vectors of 32, %d apart" % walk if walk != (1, 32) }}
{{ "template <int N>" if not n }}
{{ '{{ attr }}' }} void {{ '{{ name }}' }}(const bfloat16 *__restrict x, const bfloat16 *__restrict t_, const bfloat16 *__restrict t2_,
                {{ T }} *__restrict y{{ ", const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd" if table_arg }})
{
    const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
    const aie::accum<accfloat, 32> k(aie::broadcast<float, 32>({{ M }}f));
{% if table_arg %}
    const v16int32 base = ::sel(broadcast_s32((long)tab_ab - {{ bias }}), broadcast_s32((long)tab_cd - {{ bias }}), 0xCCCC);
{% else %}
    const v16int32 base = ::sel(broadcast_s32((long){{ '{{ name }}' }}_ab - {{ bias }}), broadcast_s32((long){{ '{{ name }}' }}_cd - {{ bias }}),
                                0xCCCC);
{% endif %}
{{ wx.decl }}
    const bfloat16 __aie_dm_resource_d *__restrict px = (const bfloat16 __aie_dm_resource_d *)x;
    const bfloat16 __aie_dm_resource_a *__restrict pt = (const bfloat16 __aie_dm_resource_a *)t_;
    const bfloat16 __aie_dm_resource_c *__restrict pt2 = (const bfloat16 __aie_dm_resource_c *)t2_;
    {{ T }} __aie_dm_resource_b *__restrict py = ({{ T }} __aie_dm_resource_b *)y;
    AIE_PREPARE_FOR_POSTPIPELINING AIE_LOOP_REG_REWRITE_SWPAWARE
    for (int j = 0; j < {{ n or "N" }} / 32; j++) {
        const aie::vector<bfloat16, 32> t = aie::load_v<32>(pt);
        const aie::vector<bfloat16, 32> t2 = aie::load_v<32>(pt2);
        const aie::vector<bfloat16, 32> v = aie::mac(k, t, one).template to_vector<bfloat16>();  // t on the grid
        v32bfloat16 q0, q1, q2, q3;
        {{ lut16 }}(::add(base, bfloat16_to_int(v.extract<16>(0), {{ shift }})), q0, q1);
        {{ lut16 }}(::add(base, bfloat16_to_int(v.extract<16>(1), {{ shift }})), q2, q3);
        const aie::vector<bfloat16, 32> r = aie::mac(aie::msc(k, v, one), t2, one).template to_vector<bfloat16>();
        const aie::vector<bfloat16, 32> r2 = aie::mul(r, r).template to_vector<bfloat16>();
        const v32bfloat16 s0 = ::shuffle(q0, q1, T16_16x4_lo);  // c2 | c1 of lanes 0-15
        const v32bfloat16 s1 = ::shuffle(q2, q3, T16_16x4_lo);
        aie::accum<accfloat, 32> acc;
        acc.insert(0, aie::accum<accfloat, 16>((v16accfloat)::shuffle(q0, q1, T32_16x2_hi)));  // c0
        acc.insert(1, aie::accum<accfloat, 16>((v16accfloat)::shuffle(q2, q3, T32_16x2_hi)));
        acc = aie::mac(acc, aie::vector<bfloat16, 32>(::shuffle(s0, s1, T256_2x2_hi)), r);
        acc = aie::mac(acc, aie::vector<bfloat16, 32>(::shuffle(s0, s1, T256_2x2_lo)), r2);
{% if not full %}
        acc = aie::mac(acc, aie::max(aie::load_v<32>(px), aie::zeros<bfloat16, 32>()), one);
{% endif %}
        aie::store_v(py, acc.template to_vector<{{ T }}>());
        {{ wx.step }}
        pt += 32;
        pt2 += 32;
        py += 32;
    }
}
"""

def table_pass(table: ActTable, *, table_arg: bool = False, walk: tuple[int, int] = (1, 32), n: int | None = None,
               f32: bool = False) -> Fn:
    """The table pass of ``table`` (an ActTable): a noinline template ``<N>(x, t_, t2_, y[, tab_ab, tab_cd])``, y = the
    table's quadratic of t (t_, t2_: t = min(|x|, t_max) as bf16, or x clamped for a full table; may be y) plus relu(x)
    (``f32``: y fp32, not rounded to bf16; t_, t2_ then buffers of their own).
    The table's two copies in two 16 KB regions: static, in the linker's sections of banks 0 and 1 (one section per
    table: a core links only the tables it reads), or (``table_arg``)
    the core's buffers ``tab_ab``, ``tab_cd`` (``table_words`` each; two parameters: llvm-aie gives the gathers a memory
    operand only when each base of their address vector is a pointer of its own: from one pointer and an offset of it,
    each gather orders against every load and store, and the loop does not pipeline). ``walk``: x in groups of
    ``walk[0]`` vectors of 32 values, ``walk[1]`` apart (a GEMM's gate/up C tile [gate 64 | up 64]: (2, 128)). ``n``: a plain function of n values (a multiple of 32), not a template (a kernel of a
    program-memory-patched design must be emitted where it is defined)."""
    if not isinstance(table, ActTable):
        raise TypeError(f"kmath: table = {table!r}, not an ActTable")
    shift = int(np.log2(16 / table.seg))  # 16 bytes of offset per segment
    words = [f"0x{w:08x}u" for w in table.table_words()]
    l16 = _lut16()
    kw = {"table": table.name, "lut16": l16.name, "M": repr(table.m), "shift": shift, "t_max": table.t_max,
          "full": table.full}
    if n is not None and (n <= 0 or n % 32):
        raise ValueError(f"kmath: a table pass of n = {n} values (a multiple of 32)")
    text = _ENV.from_string(_TABLE_PASS).render(**kw, words=words, bias=int(table.base * 2 ** shift), T="float" if f32 else "bfloat16",
                                                table_arg=table_arg, walk=walk, wx=_walk(walk, "px", "cx"), n=n)
    local = f"{table.name}_tab" + ("_t" if table_arg else "") + _walk_tag(walk) + (f"_n{n}" if n else "") + ("_f" if f32 else "")
    params = {"table": table, "table_arg": table_arg, "walk": walk, "n": n, "f32": f32}
    return _fn(local, text, inline=False, deps=(l16,), lanes=32, arg="ptr", ret="void", func="table_pass",
               params=params)


_ACT = r"""
// {{ what }} of N bf16 (a multiple of 32) into y (kmath.act: {{ bound }}): t = {{ t_text }} into y,
// then the table pass in place (sets the rounding mode){{ "; x: groups of %d vectors of 32, %d apart" % walk if walk != (1, 32) }}
{{ "template <int N>" if not n }}
{{ '{{ attr }}' }} void {{ '{{ name }}' }}(const bfloat16 *__restrict x, {{ "bfloat16 *__restrict t, float *__restrict y" if f32 else "bfloat16 *__restrict y" }}{{ ", const uint32_t *__restrict tab_ab, const uint32_t *__restrict tab_cd" if table_arg }})
{
    aie::set_rounding(aie::rounding_mode::conv_even);
    {
        const aie::vector<bfloat16, 32> hi = aie::broadcast<bfloat16, 32>({{ clamp }}f);
{% if full %}
        const aie::vector<bfloat16, 32> lo = aie::broadcast<bfloat16, 32>(-{{ clamp }}f);
{% endif %}
        const bfloat16 __aie_dm_resource_a *__restrict px = (const bfloat16 __aie_dm_resource_a *)x;
        bfloat16 __aie_dm_resource_b *__restrict pw = (bfloat16 __aie_dm_resource_b *){{ "t" if f32 else "y" }};
{{ wx.decl }}
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < {{ n or "N" }} / 32; j++) {
{% if full %}
            aie::store_v(pw, aie::min(aie::max(aie::load_v<32>(px), lo), hi));
{% elif scale == 2 %}
            const aie::vector<bfloat16, 32> a = aie::abs(aie::load_v<32>(px));
            aie::store_v(pw, aie::min(aie::add(a, a), hi));
{% else %}
            aie::store_v(pw, aie::min(aie::abs(aie::load_v<32>(px)), hi));
{% endif %}
            {{ wx.step }}
            pw += 32;
        }
    }
    {{ tab }}{{ "<N>" if not n }}(x, {{ "t, t" if f32 else "y, y" }}, y{{ ", tab_ab, tab_cd" if table_arg }});
}
"""

# act's bounds (bf16 ULPs of the exact value) and where they hold (below: tests/operators/test_kmath.py)
_ACT_BOUND = {"silu": (0.59, (-24.0, np.inf)), "sigmoid": (0.54, (-16.0, 16.0)), "gelu": (0.59, (-4.0, np.inf)),
              "gelu_erf": (0.59, (-4.0, np.inf))}


_ACT_REL = {"silu": 7e-4}  # (the same, relative, of ``f32`` results: tests/operators/test_kmath.py)


def act(table: ActTable, *, table_arg: bool = False, walk: tuple[int, int] = (1, 32), n: int | None = None,
        scale: int = 1, clamp: float | None = None, inline: bool = False, f32: bool = False) -> Fn:
    """``table``'s activation (an ActTable: SILU, GELU, GELU_ERF, SIGMOID) of N bf16 into y (x, y distinct): a noinline
    template ``<N>(x, y[, tab_ab, tab_cd])``, two loops (t = min(|x|, t_max) into y, then the table pass in place): ~0.8
    cycles per element (SiLU: 3287 per 4096 on the NPU). ``f32``: ``<N>(x, t, y)``, y fp32 (the quadratic not rounded to
    bf16: the table's fit, ~0.09 bf16 ULP of the exact value, where a fp32 consumer takes ``silu``'s 16 cycles per element
    otherwise), t a bf16 scratch of N values. ``table_arg``, ``walk`` (x's), ``n`` (a plain function of n
    values): ``table_pass``'s. ``scale`` 2, ``clamp``: the table fed t = min(2 |x|, clamp) (a table of another function
    on this one's grid: the vision overlay's GELU, ``vision.layers.gelu_table``; its words the caller's buffers).
    ``inline``: the clamp pass inlined into the caller (a call per few thousand values costs ~7 cycles: the PLE's
    0.25%)."""
    _check("scale", scale, (1, 2))
    clamp = table.t_max * scale if clamp is None else clamp
    if table.full and (scale, clamp) != (1, table.t_max):
        raise ValueError(f"kmath: {table.name} (a full table) takes x itself, clamped to its range")
    tab = table_pass(table, table_arg=table_arg, walk=walk, n=n, f32=f32)
    ulp, dom = _ACT_BOUND[table.name] if (scale, clamp) == (1, table.t_max) else (None, (-np.inf, np.inf))
    t_text = (f"x clamped to +-{clamp:g}" if table.full else f"min({scale} |x|, {clamp:g})" if scale > 1
              else f"min(|x|, {clamp:g})")
    bound = f"{_bound(ulp, None)} for x in [{dom[0]:g}, {dom[1]:g})" if ulp else "the caller's words"
    what = f"{table.name} through its table" if ulp else f"a table on {table.name}'s grid"
    text = _ENV.from_string(_ACT).render(what=what, table=table.name, tab=tab.name, clamp=repr(float(clamp)), full=table.full,
                                         bound=bound, t_text=t_text, scale=scale, n=n, table_arg=table_arg, walk=walk,
                                         wx=_walk(walk, "px", "cx"), f32=f32)
    local = (f"{table.name}_lut" + ("_t" if table_arg else "") + _walk_tag(walk) + (f"_n{n}" if n else "")
             + (f"_s{scale}c{clamp:g}" if (scale, clamp) != (1, table.t_max) else "") + ("_i" if inline else "")
             + ("_f" if f32 else ""))
    return _fn(local, text, inline=inline, deps=(tab,), lanes=32, arg="ptr", ret="void", func="act",
               params={"table": table, "table_arg": table_arg, "walk": walk, "n": n, "scale": scale, "clamp": clamp,
                       "inline": inline, "f32": f32},
               domain=dom, valid=(-3e38, 3e38), ulp=None if f32 else ulp, rel=_ACT_REL.get(table.name) if f32 else None)


_SILU_SMALL = r"""
// silu(x) for |x| < 2.5 as u + s (P0 + P1 s + P2 s^2), s = u^2, u = x / 2 (tanh(u) / u fitted in bf16 arithmetic); x
// again through xr (a second load of the same data, not merged with x's by the compiler: shorter lifetimes, II 10
// instead of 14-18) and xm (the select's). Returns y where |x| >= 2.5.
{{ attr }} aie::vector<bfloat16, 32> {{ name }}(aie::vector<bfloat16, 32> x, aie::vector<bfloat16, 32> xr,
                                             aie::vector<bfloat16, 32> xm, aie::vector<bfloat16, 32> y)
{
    const aie::vector<bfloat16, 32> kh = aie::broadcast<bfloat16, 32>(0.5f);
    const aie::vector<bfloat16, 32> p0 = aie::broadcast<bfloat16, 32>(0.99609375f);
    const aie::vector<bfloat16, 32> p2 = aie::broadcast<bfloat16, 32>(0.06298828125f);
    const aie::accum<accfloat, 32> p1 = aie::accum<accfloat, 32>(aie::broadcast<float, 32>(-0.30078125f));
    aie::vector<bfloat16, 32> u = aie::mul(x, kh).template to_vector<bfloat16>();
    aie::vector<bfloat16, 32> s = aie::mul(u, u).template to_vector<bfloat16>();
    aie::accum<accfloat, 32> acc = aie::mac(aie::mul(xr, kh), s, p0);
    acc = aie::mac(acc, aie::mul(s, s).template to_vector<bfloat16>(), aie::mac(p1, s, p2).template to_vector<bfloat16>());
    return aie::select(y, acc.template to_vector<bfloat16>(), aie::lt(aie::abs(xm), aie::broadcast<bfloat16, 32>(2.5f)));
}
"""

_SILU_TANH = r"""
// silu of N bf16 into y in two passes (kmath.silu_tanh: max 0.92 bf16 ULP from x = 2.5 on, 6.6 (mean 0.04) in (-2.5,
// 2.5), 7.7 in [-3, -2.5), 38 (mean 8.6) in [-24, -3); kmath.act(SILU): 0.59): A: u (1 + t), t = vtanh(u), u = x / 2,
// below x = -3 u (1 - t) 2^(x log2e) by one vexp2 chord (the tail factor u (1 - t) formed first: II 10; 14 with the
// exp2 product first); B: silu_small for |x| < 2.5 in place (II 10; the vtanh LUT alone costs 2.6x the first-token
// KL of a 1000-token prompt). One loop for A and B does not pipeline. ``zero`` is a runtime 0: B's second loads of x
// (not merged with the first) keep its lifetimes short.{{ " x: groups of %d vectors of 32, %d apart." % walk if walk != (1, 32) else "" }}
template <int N>
{{ '{{ attr }}' }} void {{ '{{ name }}' }}(const bfloat16 *__restrict x, bfloat16 *__restrict y, int32_t zero)
{
    {  // A
        const bfloat16 __aie_dm_resource_a *__restrict pg = (const bfloat16 __aie_dm_resource_a *)x;
        bfloat16 __aie_dm_resource_b *__restrict py = (bfloat16 __aie_dm_resource_b *)y;
        const aie::vector<bfloat16, 32> kh = aie::broadcast<bfloat16, 32>(0.5f);
        const aie::vector<bfloat16, 32> kl = aie::broadcast<bfloat16, 32>(1.44269504f);
        const aie::vector<bfloat16, 32> kt = aie::broadcast<bfloat16, 32>(-3.0f);
{{ wg.decl }}
        for (int j = 0; j < N / 32; j++) {  // (no pragma: post-RA II 10; post-RA only: 25)
            aie::vector<bfloat16, 32> x = aie::load_v<32>(pg);
            {{ wg.step }}
            aie::accum<accfloat, 32> a_acc = aie::mul(x, kh);
            aie::vector<bfloat16, 32> a = a_acc.template to_vector<bfloat16>();
            aie::vector<bfloat16, 32> t = al::tanh_acc<32>(a_acc);
            aie::vector<bfloat16, 32> body = aie::mac(a_acc, a, t).template to_vector<bfloat16>();
            aie::vector<bfloat16, 32> a1t = aie::msc(a_acc, a, t).template to_vector<bfloat16>();
            aie::vector<bfloat16, 32> tail = aie::mul(a1t, {{ chord }}(aie::mul(x, kl))).template to_vector<bfloat16>();
            aie::store_v(py, aie::select(body, tail, aie::lt(x, kt)));
            py += 32;
        }
    }
    {  // B
        const bfloat16 __aie_dm_resource_a *__restrict pg = (const bfloat16 __aie_dm_resource_a *)x;
        const bfloat16 __aie_dm_resource_d *__restrict py = (const bfloat16 __aie_dm_resource_d *)y;
        bfloat16 __aie_dm_resource_b *__restrict pw = (bfloat16 __aie_dm_resource_b *)y;
{{ wg.decl }}
        AIE_PREPARE_FOR_POSTPIPELINING
        for (int j = 0; j < N / 32; j++) {
            aie::store_v(pw, {{ small }}(aie::load_v<32>(pg), aie::load_v<32>(pg + zero), aie::load_v<32>(pg + zero),
                                       aie::load_v<32>(py)));
            {{ wg.step }}
            py += 32;
            pw += 32;
        }
    }
}
"""


def silu_tanh(*, walk: tuple[int, int] = (1, 32), inline: bool = True) -> Fn:
    """silu of N bf16 into y (a template ``<N>(x, y, zero)``, ``zero`` a runtime 0) in two pipelined passes through
    vtanh, one vexp2 chord and a fit for |x| < 2.5: ~0.6 cycles per element against ``act(SILU)``'s ~0.8, but up to 38 bf16
    ULP below x = -2.5 (``act``: 0.59 everywhere). ``walk``: x's walk (``table_pass``)."""
    chord = exp2(32, bias=0, prec="chord", inline=True)
    small = _fn("silu_small", _SILU_SMALL, inline=True, lanes=32, arg="bfloat16", ret="bfloat16")
    text = _ENV.from_string(_SILU_TANH).render(chord=chord.name, small=small.name, walk=walk, wg=_walk(walk, "pg", "cg"))
    local = "silu_tanh" + _walk_tag(walk) + ("_i" if inline else "")
    return _fn(local, text, inline=inline, deps=(chord, small), lanes=32, arg="ptr", ret="void", func="silu_tanh",
               params={"walk": walk, "zero": True}, domain=(2.5, 3e38), valid=(-3e38, 3e38),
               ulp=0.92)
