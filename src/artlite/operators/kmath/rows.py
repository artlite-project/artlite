"""Functions on a head's row (templates: ``rows.*.cc.jinja``): ``rms_norm`` (a bf16 row times 1 / rms into an fp32
row, optionally times an fp32 weight) and ``rope`` (rotations of pairs of an fp32 row's dims, in place), the attention
preps' and cores' query and key rows. A caller composes them in a function of its own:

    {% set nrm, rope = km.rms_norm(HD, eps=eps, weight=True), km.rope(HD // 2) %}
    {{ nrm.src }}
    {{ rope.src }}
    ... {{ nrm }}(x, w, row); {{ rope }}(row, cs);

One numeric form each, the cheapest of the equally accurate ones (measured with measure_ii, n = 256): the norm's x
times 1 / rms as two products of the bf16 x with 1 / rms's bf16 parts (x times the fp32 1 / rms through ``al::mulff``
gives the same values: x's lo part is 0), the sum of squares written out (a loop of 2 / 4 ``mac_square`` costs 80 / 128
B more and 10-35 cycles); the rotation ``al::rot2`` (each operand split once, a pair's output one rounding: four
``al::mulff`` are as accurate, 16 B more and two more vector operations per 32 lanes). What stays a choice trades
program memory against cycles: the lanes of the norm's loop and its pragma (with a weight, n = 256: 32 lanes and no
hardware loop 224 B / 306 cycles, 64 lanes 288 B / 189, a 64-lane hardware loop 608 B / 110; without one 32 lanes, a
hardware loop: 288 B / 72 cycles)."""

from __future__ import annotations

from .fn import Fn, check, make

PRAGMAS = {"": "", "min1": "AIE_LOOP_MIN_ITERATION_COUNT(1)"}  # (min1: no hardware loop: smaller, slower)


def _tag(eps: float) -> str:
    return f"{eps:g}".replace("-", "m").replace(".", "p").replace("+", "")


def rms_norm(n: int, *, eps: float, weight: bool = False, lanes: int = 32, loop: str = "", inline: bool = True) -> Fn:
    """out = x / rms(x) (n bf16 -> fp32; ``weight``: times w, n fp32, by ``al::mulff``): a function ``(x, [w,] out)``;
    1 / rms through ``al::inv_rms<n>`` (the scalar invsqrt) of ``aie::mac_square``'s sums in 64 lanes, then per
    ``lanes`` (32 or 64) values x times its bf16 parts (2^-16). ``loop``: that loop's pragma ("" or "min1": no hardware
    loop)."""
    check("lanes", lanes, (32, 64))
    check("loop", loop, tuple(PRAGMAS))
    if n % 64 or n <= 0:
        raise ValueError(f"kmath: rms_norm of n = {n} (a multiple of 64)")
    local = f"rms_{n}_l{lanes}" + ("_w" if weight else "") + (f"_{loop}" if loop else "") + f"_e{_tag(eps)}"
    return make(local + ("_i" if inline else ""), "rows.rms_norm.cc.jinja", inline=inline, lanes=lanes, arg="ptr",
                ret="ptr", func="rms_norm", params={"n": n, "eps": eps, "weight": weight, "lanes": lanes, "loop": loop},
                variables={"n": n, "eps": repr(float(eps)), "weight": weight, "L": lanes, "pragma": PRAGMAS[loop]})


def rope(rot: int, *, dist: int | None = None, sin: str = "after", table: str = "fp32", inline: bool = True) -> Fn:
    """The rotation of the pairs (i, i + ``dist``) of an fp32 row, i < ``rot`` (multiples of 32; ``dist``: default
    ``rot``, NeoX's halves), in place by ``al::rot2``: a function ``(row, cs)``, cos at cs + i and sin at cs + rot + i
    (``sin`` "after"), or ``(row, cs, sn)``, sin at sn + i ("ptr"); ``table`` "fp32" or "bf16" (cos, sin as bf16)."""
    dist = rot if dist is None else dist
    check("sin", sin, ("after", "ptr"))
    check("table", table, ("fp32", "bf16"))
    if rot <= 0 or rot % 32 or dist < rot or dist % 32:
        raise ValueError(f"kmath: rope of rot = {rot}, dist = {dist} (multiples of 32, dist >= rot)")
    local = (f"rope_{rot}" + (f"_d{dist}" if dist != rot else "") + ("_sp" if sin == "ptr" else "")
             + ("_tb" if table == "bf16" else "") + ("_i" if inline else ""))
    return make(local, "rows.rope.cc.jinja", inline=inline, lanes=32, arg="ptr", ret="ptr", func="rope",
                params={"rot": rot, "dist": dist, "sin": sin, "table": table},
                variables={"rot": rot, "dist": dist, "sn": sin == "ptr", "T": "bfloat16" if table == "bf16" else "float"})
