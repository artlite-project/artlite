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
    """A bound as the comments state it."""
    return f"max {ulp:.2f} bf16 ULP" if ulp is not None else f"max 2^{math.log2(rel):.1f} relative"


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


# Activations built on exp2 -----------------------------------------------------------------------------------------

_SILU = r"""
// silu(x) = x sigmoid(x) of {{ N }} bf16 lanes in fp32 (kmath.silu): t = e^-|x| (2^x to bf16), r = 1 / (1 + t) from a linear
// guess on [1, 2] and two Newton steps, sigmoid = x >= 0 ? r : t r. Rounded to bf16: 98.7% correctly rounded, max
// 1.17 ULPs (x < 0, where t's bf16-level error passes through; tests/rtc/test_silu.py).
{{ attr }} aie::vector<float, {{ N }}> {{ name }}(aie::vector<bfloat16, {{ N }}> x)
{
    aie::vector<float, {{ N }}> xf = aie::accum<accfloat, {{ N }}>(x).template to_vector<float>();
    aie::vector<float, {{ N }}> nax = aie::min(xf, aie::sub(aie::zeros<float, {{ N }}>(), xf));
    aie::vector<bfloat16, {{ N }}> tb = aie::accum<accfloat, {{ N }}>({{ exp2 }}(al::log2e<{{ N }}>(nax))).template to_vector<bfloat16>();
    aie::vector<float, {{ N }}> t = aie::accum<accfloat, {{ N }}>(tb).template to_vector<float>();
    const aie::vector<float, {{ N }}> one = aie::broadcast<float, {{ N }}>(1.0f);
    const aie::vector<float, {{ N }}> two = aie::broadcast<float, {{ N }}>(2.0f);
    aie::vector<float, {{ N }}> d = aie::add(t, one);
    aie::vector<float, {{ N }}> r = aie::sub(aie::broadcast<float, {{ N }}>(1.4117647f),
                                     al::mulc<{{ N }}>(d, aie::broadcast<bfloat16, {{ N }}>(0.47058824f)).template to_vector<float>());
    r = al::mulff<{{ N }}>(r, aie::sub(two, al::mulff<{{ N }}>(d, r)));
    r = al::mulff<{{ N }}>(r, aie::sub(two, al::mulff<{{ N }}>(d, r)));
    aie::vector<float, {{ N }}> sg = aie::select(r, al::mulff<{{ N }}>(t, r), aie::lt(xf, aie::zeros<float, {{ N }}>()));
    return al::mulff<{{ N }}>(xf, sg);
}
"""


def silu(lanes: int, *, inline: bool = False) -> Fn:
    """silu(x) of ``lanes`` bf16 lanes -> fp32 (~16 cycles per element: fp32 math through bf16 products; the gather
    table's SiLU, ``act_lut``, is ~1)."""
    _check("lanes", lanes, LANES)
    e = exp2(lanes, inline=True)
    text = _ENV.from_string(_SILU).render(N=lanes, exp2=e.name, name="{{ name }}", attr="{{ attr }}")
    return _fn(f"silu_{lanes}" + ("_i" if inline else ""), text, inline=inline, deps=(e,), lanes=lanes,
               arg="bfloat16", ret="float", func="silu")


_GELU = r"""
// gelu(x) of 32 bf16 lanes in its tanh form (kmath.gelu), a (1 + tanh(w)), a = x / 2, w = sqrt(2 / pi) (x + 0.044715 x^3)
// (gelu_pytorch_tanh, ggml_gelu: Gemma's activation) with the vtanh LUT, bf16: w = x (C1 + C2 x^2) in fp32 through bf16
// hi / lo products (C1 = sqrt(2 / pi), C2 = 0.044715 C1), T = tanh(|w|) (al::silu_small's fit tanh(w) / w in w^2
// below |w| = 1.25, the LUT above); a (1 + T) for x >= 0, a 2^(-2 |w| log2e) (1 + T) for x < 0 (1 + tanh(w) = e^(2 w)
// (1 + |tanh w|) for w < 0: no cancellation anywhere; 2^y with the exponent clamped as an integer). xr: x again (a
// second load: shorter lifetimes, as in al::silu_small).
{{ attr }} aie::vector<bfloat16, 32> {{ name }}(aie::vector<bfloat16, 32> x, aie::vector<bfloat16, 32> xr)
{
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


def gelu(*, inline: bool = False) -> Fn:
    """gelu(x) (tanh form) of 32 bf16 lanes -> bf16; called with x twice (x, xr: two loads of the same data)."""
    e = exp2(32, below="int", inline=True)
    text = _ENV.from_string(_GELU).render(exp2=e.name, name="{{ name }}", attr="{{ attr }}")
    return _fn("gelu_32" + ("_i" if inline else ""), text, inline=inline, deps=(e,), lanes=32, arg="bfloat16",
               ret="bfloat16", func="gelu")
