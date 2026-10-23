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
comment): tests/rtc/test_kmath.py checks them on the NPU against the variant's numpy model (tests/rtc/kmath_sim.py),
bit-exact.
"""

from __future__ import annotations

import math
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
    return "bound not measured" if rel is None else f"max 2^{math.log2(rel):.1f} relative"


# exp2 ---------------------------------------------------------------------------------------------------------------

# (the coefficients: also kmath_sim's)
LN2 = (0.69140625, 0.001739501953125)  # ln 2 as bf16 hi + lo
TAYLOR = tuple(np.float32(1) / np.float32(k) for k in (720, 120, 24, 6, 2, 1, 1))  # e^t to t^6, highest first
CUBIC = (0.24022650695910071, 0.0555041086648216, 0.69314718055994531)  # 2^f ~ 1 + f (c1 + f (c2 + c3 f)): c2, c3, c1
ACC_C = (0.69140625, 0.2412109375, 0.0625)  # 2^f ~ 1 + c1 f + f^2 (c2 + c3 f), bf16 (fitted in this order)

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


# The bias-192 form's first step per contract, each the form its callers' loops schedule best: below "clamp" alone
# takes f from the clamped v (one bf16 max each; prefill/attn's P panels: II 25); with a clamp above or "bounded", f
# is the residual of the unclamped v (a shorter chain: sd15's pexp pipelines, II 23; qwen35's pass 1 II 8), and v is
# clamped at 128 (bf16(A) rounds A in [127.5, 128.5] to 128: v and f agree; vexp2(0) = 1).
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


def exp2(lanes: int, *, bias: int | None = None, prec: str = "bf16", below: str = "clamp", above: str = "none",
         shift: int = 0, inline: bool = False) -> Fn:
    """2^x of ``lanes`` lanes.

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
    _check("bias", bias, (None, 192))
    params = {"bias": bias, "prec": prec, "below": below, "above": above, "shift": shift}
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
// through; tests/rtc/test_silu.py).
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
    table's SiLU, ``act_lut``, is ~1). ``exp_fn``: an exp2 of fp32 arguments that clamps (the argument reaches -3e38;
    default the bf16-level one, inlined); ``recip_fn``: a recip on [1, 2] (default 2 steps unrolled, inlined)."""
    _check("lanes", lanes, LANES)
    e, etag = _take("exp_fn", exp_fn, exp2(lanes, inline=True), func="exp2", lanes=lanes, bias=None,
                    below={"clamp", "zero"})
    r, rtag = _take("recip_fn", recip_fn, recip(lanes, unroll=True), func="recip", lanes=lanes, domain="unit")
    text = _ENV.from_string(_SILU).render(N=lanes, exp2=e.name, recip=r.name, name="{{ name }}", attr="{{ attr }}")
    return _fn(f"silu_{lanes}{etag}{rtag}" + ("_i" if inline else ""), text, inline=inline, deps=(e, r), lanes=lanes,
               arg="bfloat16", ret="float", func="silu", params={"exp_fn": e, "recip_fn": r}, domain=(-80.0, 80.0),
               valid=(-1e38, 1e38), rel=None if etag or rtag else 4.5e-3)


_GELU = r"""
// gelu(x) of 32 bf16 lanes in its tanh form (kmath.gelu), a (1 + tanh(w)), a = x / 2, w = sqrt(2 / pi) (x + 0.044715 x^3)
// (gelu_pytorch_tanh, ggml_gelu: Gemma's activation) with the vtanh LUT, bf16: w = x (C1 + C2 x^2) in fp32 through bf16
// hi / lo products (C1 = sqrt(2 / pi), C2 = 0.044715 C1), T = tanh(|w|) (al::silu_small's fit tanh(w) / w in w^2
// below |w| = 1.25, the LUT above); a (1 + T) for x >= 0, a 2^(-2 |w| log2e) (1 + T) for x < 0 (1 + tanh(w) = e^(2 w)
// (1 + |tanh w|) for w < 0: no cancellation anywhere). {{ "x loaded twice from g" if ptr else "xr: x again" }} (a second load: shorter lifetimes, as in
// al::silu_small).{{ " Valid for x >= -344 (the exp2's exponent clamped as an integer)." if int_exp else "" }}
{% if ptr %}
{{ attr }} aie::vector<bfloat16, 32> {{ name }}(const bfloat16 *__restrict g)
{
    const aie::vector<bfloat16, 32> x = aie::load_v<32>(g), xr = aie::load_v<32>(g);
{% else %}
{{ attr }} aie::vector<bfloat16, 32> {{ name }}(aie::vector<bfloat16, 32> x, aie::vector<bfloat16, 32> xr)
{
{% endif %}
    const aie::vector<float, 32> sq = aie::mul(x, x).template to_vector<float>();  // exact
    const aie::vector<float, 32> u = aie::add(aie::broadcast<float, 32>(0.7978845608f),
                                              al::mul_hilo<32, 0.03564453125f, 3.2901763916015625e-05f>(sq));
    const aie::vector<float, 32> aw = aie::abs(al::mulc<32>(u, xr).template to_vector<float>());  // |w|
    // T = tanh(|w|)
    const aie::vector<bfloat16, 32> wb = aie::accum<accfloat, 32>(aw).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> s = aie::mul(wb, wb).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> p0 = aie::broadcast<bfloat16, 32>(0.99609375f);
    const aie::vector<bfloat16, 32> p2 = aie::broadcast<bfloat16, 32>(0.06298828125f);
    const aie::accum<accfloat, 32> p1 = aie::accum<accfloat, 32>(aie::broadcast<float, 32>(-0.30078125f));
    const aie::vector<bfloat16, 32> tp = aie::mac(aie::mul(wb, p0), aie::mul(wb, s).template to_vector<bfloat16>(),
                                                  aie::mac(p1, s, p2).template to_vector<bfloat16>()).template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> tl = aie::tanh<bfloat16>(aw);
    const aie::vector<bfloat16, 32> t = aie::select(tl, tp, aie::lt(wb, aie::broadcast<bfloat16, 32>(1.25f)));
    // x >= 0: a (1 + T)
    const auto a_acc = aie::mul(x, aie::broadcast<bfloat16, 32>(0.5f));
    const aie::vector<bfloat16, 32> a = a_acc.template to_vector<bfloat16>();
    const aie::vector<bfloat16, 32> pos = aie::mac(a_acc, a, t).template to_vector<bfloat16>();
    // x < 0: a e (1 + T), e = 2^(-2 |w| log2e)
    const aie::vector<float, 32> e = {{ exp2 }}(al::mul_hilo<32, -2.890625f, 0.0052490234375f>(aw));
    const auto pe = aie::mul(a, aie::accum<accfloat, 32>(e).template to_vector<bfloat16>());
    const aie::vector<bfloat16, 32> neg = aie::mac(pe, pe.template to_vector<bfloat16>(), t).template to_vector<bfloat16>();
    return aie::select(pos, neg, aie::lt(x, aie::zeros<bfloat16, 32>()));
}
"""


def gelu(*, ptr: bool = False, exp_fn: Fn | None = None, inline: bool = False) -> Fn:
    """gelu(x) (tanh form) of 32 bf16 lanes -> bf16: called with x twice (x, xr: two loads of the same data), or with
    ``ptr`` a pointer to them (it loads them twice itself). ``exp_fn``: an exp2 of 32 fp32 lanes (default: the
    exponent clamped as an integer, inlined: x >= -344; a clamping one for any x)."""
    e, tag = _take("exp_fn", exp_fn, exp2(32, below="int", inline=True), func="exp2", lanes=32, bias=None)
    text = _ENV.from_string(_GELU).render(exp2=e.name, ptr=ptr, int_exp=e.params["below"] == "int",
                                          name="{{ name }}", attr="{{ attr }}")
    return _fn("gelu_32" + ("_p" if ptr else "") + tag + ("_i" if inline else ""), text, inline=inline, deps=(e,),
               lanes=32, arg="bfloat16", ret="bfloat16", func="gelu", params={"exp_fn": e, "ptr": ptr})
