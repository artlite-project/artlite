"""2^x in three forms: an fp32 vector (``exp2_f32``), through the vexp2 LUT as a chord of it (``exp2_chord``), and the
bias-192 accumulator form (``exp2_acc``); ``exp2`` picks one by ``bias``. Templates: ``exponent.*.cc.jinja``."""

from __future__ import annotations

import numpy as np

from .fn import LANES, Fn, bound, check, make

# (the coefficients: also kmath_sim's)
LN2 = (0.69140625, 0.001739501953125)  # ln 2 as bf16 hi + lo
TAYLOR = tuple(np.float32(1) / np.float32(k) for k in (720, 120, 24, 6, 2, 1, 1))  # e^t to t^6, highest first
CUBIC = (0.24022650695910071, 0.0555041086648216, 0.69314718055994531)  # 2^f ~ 1 + f (c1 + f (c2 + c3 f)): c2, c3, c1
ACC_C = (0.69140625, 0.2412109375, 0.0625)  # 2^f ~ 1 + c1 f + f^2 (c2 + c3 f), bf16 (fitted in this order)
# CHORD2_SCALE 2^x ~ c1 vexp2(x) + c2 vexp2(x + 1/2): c1 = 1 (the caller's 1.0 register shared), c2 = 1 / sqrt 2 (the
# chords' errors, half a period apart, cancel: the SD 1.5 UNet's error 1.19e-2; the pair minimax for |x| <= 2, 1.28e-2)
CHORD2_C = (1.0, 0.70703125)
CHORD2_SCALE = 2.0677  # (the middle of the ratio's range for |x| <= 2)

_BELOW_F32 = {
    "clamp": "Below -126: 2^-126 (masked lanes: -1e30, -inf).",
    "zero": "Exactly 0 from x <= -127 (masked lanes: -1e30, -inf); between -127 and -126 below 2^-126, inexact.",
    "int": "The exponent clamped as an integer (no fp32 max: arguments above -2^22): 2^(-126 + f) below -126.",
}

# The bias-192 form's first step per contract (``exponent.acc_steps.cc.jinja``'s ``reduce``), each the form its callers'
# loops schedule best: below "clamp" alone takes f from the clamped v (one bf16 max each; prefill/attn's P panels: II
# 25); with a clamp above or "bounded", f is the residual of the unclamped v (a shorter chain: the vision overlay's pexp
# pipelines, II 23; qwen35's pass 1 II 8), and v is clamped at 128 (bf16(A) rounds A in [127.5, 128.5] to 128: v and f
# agree; vexp2(0) = 1).
_BELOW_ACC = {
    ("clamp", "none"): "x below -63.5 (masked lanes too, -inf included): v = 129, f = -1/2 (~2^-63.5).",
    ("clamp", "clamp"): "x below -64.5 (masked lanes too, finite): v = 128, f in [-1/2, 1/2] (~2^-64).",
    ("bounded", "none"): "x below -64.5 down to A = -2^15 (masked lanes as -16384): v = 128, f the residual (|2^x| < "
                         "2^-49, its sign not kept).",
    ("none", "none"): "A below 128.5 is the caller's to prevent.",
}


def exp2_chord(lanes: int, *, prec: str, below: str, above: str, shift: int, inline: bool) -> Fn:
    """The bias-0 form: x an fp32 accumulator, through vexp2 (``exp2``'s arguments)."""
    check("prec", prec, ("chord", "chord2"))
    check("below", below, ("clamp",))
    check("above", above, ("none",))
    check("shift", shift, (0,))
    two = prec == "chord2"
    rel = 1.2e-2 if two else 6.2e-2
    wider = "2.0% for |x| <= 4, 3.6% for |x| <= 8" if two else "7.9% for |x| <= 8"
    local = f"exp2_{prec}_{lanes}" + ("_i" if inline else "")
    return make(local, "exponent.chord.cc.jinja", inline=inline, lanes=lanes, arg="accum",
                ret="accum" if two else "bfloat16", func="exp2",
                params={"bias": 0, "prec": prec, "below": below, "above": above, "shift": shift}, domain=(-2.0, 2.0),
                valid=(-120.0, 120.0), rel=rel,
                variables={"N": lanes, "two": two, "c": CHORD2_C, "scale": CHORD2_SCALE if two else 1,
                           "bound": bound(None, rel), "wider": wider})


def exp2_f32(lanes: int, *, prec: str, below: str, above: str, shift: int, inline: bool) -> Fn:
    """The bias-None form: x an fp32 vector (``exp2``'s arguments)."""
    check("prec", prec, ("bf16", "fp32"))
    check("below", below, ("clamp", "zero", "int") if prec == "bf16" else ("clamp",))
    check("above", above, ("none",))
    check("shift", shift, (0,))
    fp32 = prec == "fp32"
    ulp, rel = (None, 1.1e-5) if fp32 else (0.4, None)
    local = f"exp2_v{lanes}" + ("_fp32" if fp32 else "") + ("" if below == "clamp" else f"_{below}")
    local += "_i" if inline else ""
    return make(local, "exponent.f32.cc.jinja", inline=inline, lanes=lanes, arg="float", ret="float", func="exp2",
                params={"bias": None, "prec": prec, "below": below, "above": above, "shift": shift},
                domain=(-126.0, 127.5), valid=(-2.0 ** 22 if below == "int" else -np.inf, 127.5), ulp=ulp, rel=rel,
                variables={"N": lanes, "fp32": fp32, "below": below, "below_doc": _BELOW_F32[below],
                           "bound": bound(ulp, rel), "ln2": LN2, "taylor": [str(c) for c in TAYLOR],
                           "cubic": CUBIC})


def exp2_acc(lanes: int, *, prec: str, below: str, above: str, shift: int, inline: bool) -> Fn:
    """The bias-192 form: A = x + 192 an fp32 accumulator (``exp2``'s arguments); ``parts``: ``reduce``, ``poly``."""
    check("prec", prec, ("bf16",))
    check("below", below, ("clamp", "bounded", "none"))
    check("above", above, ("none", "clamp"))
    check("shift", shift, (0, 64))
    local = f"exp2_a{lanes}" + {"clamp": "", "bounded": "_bounded", "none": "_unclamped"}[below]
    local += ("_sat" if above == "clamp" else "") + (f"_s{shift}" if shift else "")
    variables = {"N": lanes, "below": below, "above": above, "shift": shift, "c": ACC_C}
    below_doc = _BELOW_ACC.get((below, above)) or _BELOW_ACC[below, "none"].replace("-inf included", "finite")
    # (arguments the result is defined for: the residual forms make a NaN of an infinite one)
    lo = {"clamp": -np.inf if above == "none" else -2.0 ** 100, "bounded": -2.0 ** 15, "none": 128.5}[below]
    poly = f"exp2_a{lanes}_poly" + (f"_s{shift}" if shift else "")
    parts = {  # (the same for the inlined whole)
        "reduce": make(local + "_reduce", "exponent.acc_reduce.cc.jinja", variables={**variables, "whole": local},
                       inline=True, lanes=lanes, arg="accum", ret=""),
        "poly": make(poly, "exponent.acc_poly.cc.jinja",
                     variables={**variables, "whole": poly.replace("_poly", "")}, inline=True, lanes=lanes, arg="",
                     ret="accum"),
    }
    return make(local + ("_i" if inline else ""), "exponent.acc.cc.jinja", inline=inline, lanes=lanes, arg="accum",
                ret="accum", func="exp2",
                params={"bias": 192, "prec": prec, "below": below, "above": above, "shift": shift},
                domain=(128.5, 255.5), valid=(lo, 2.0 ** 100 if above == "clamp" else 255.5), ulp=0.26, parts=parts,
                variables={**variables, "bound": bound(0.26, None), "below_doc": below_doc,
                           "above_doc": " x from 63.5 on: v = 255, f = 1/2 (~2^63)." if above == "clamp" else ""})


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
    check("lanes", lanes, LANES)
    check("bias", bias, (None, 0, 192))
    variant = {0: exp2_chord, None: exp2_f32, 192: exp2_acc}[bias]
    return variant(lanes, prec=prec, below=below, above=above, shift=shift, inline=inline)
