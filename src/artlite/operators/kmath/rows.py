"""Functions on a head's row (templates: ``rows.*.cc.jinja``): ``rms_norm`` (a bf16 row times 1 / rms into an fp32
row, optionally times an fp32 weight) and ``rope`` (rotations of pairs of an fp32 row's dims, in place), the attention
preps' and cores' query and key rows. A caller composes them in a function it defines:

    {% set nrm, rope = km.rms_norm(HD, eps=eps, weight=True), km.rope(HD // 2) %}
    {{ nrm.src }}
    {{ rope.src }}
    ... {{ nrm }}(x, w, row); {{ rope }}(row, cs);

One numeric form each, the cheapest of the equally accurate ones (measured with measure_ii, n = 256): the norm's x
times 1 / rms as two products of the bf16 x with 1 / rms's bf16 parts (x times the fp32 1 / rms through ``al::mulff``
gives the same values: x's lo part is 0), the sum of squares written out (a loop of 2 / 4 ``mac_square`` costs 80 / 128
B more and 10-35 cycles); the rotation ``al::rot2`` (each operand split once, a pair's output one rounding: four
``al::mulff`` are as accurate, 16 B more and two more vector operations per 32 lanes; of a bf16 table four products). What stays a choice trades
program memory against cycles: the lanes of the norm's loop and its pragma (with a weight, n = 256: 32 lanes and no
hardware loop 224 B / 306 cycles, 64 lanes 288 B / 189, a 64-lane hardware loop 608 B / 110; without one 32 lanes, a
hardware loop: 288 B / 72 cycles).

Extents known only at run time (a function shared by two kinds of layer, in a ``noinline`` / ``minsize`` wrapper):
``rms_norm``'s ``n`` and ``rope``'s ``rot`` and ``dist`` take a tuple of their values instead of a number, and then are
arguments of the function (after the others, in the order n | rot, dist). Their loops count at run time (a norm's sum
too: ``mac_square`` in a loop, 1 / n selected among the values' constants), and take the loop pragma
``AIE_LOOP_NO_UNROLL``. A plain number gives a compile-time extent: the sum is written out and the trip counts are constant
(n = 128 with a weight: 80 B less than the run-time form's). A ``minsize`` caller passes ``minsize=True``: the inlined
function's loops are compiled with that function's attributes before inlining (rotated and pipelined: the head's norm and
rotation of the attention cores 112 B larger)."""

from __future__ import annotations

from .fn import Fn, check, make

# (min1: no hardware loop: smaller, slower)
PRAGMAS = {"": "", "min1": "AIE_LOOP_HINT(enable-aie-hardware-loops, 0)"}


def _tag(eps: float) -> str:
    return f"{eps:g}".replace("-", "m").replace(".", "p").replace("+", "")


def _extent(what: str, value: int | tuple[int, ...], unit: int, *, zero: bool = False) -> tuple[int, ...]:
    """The values of an extent (a tuple of one is a number), each a multiple of ``unit``, positive (``zero``: or 0)."""
    vals = (value,) if isinstance(value, int) else tuple(value)
    if not vals or any(v % unit or v < 0 or (v == 0 and not zero) for v in vals) or max(vals) == 0:
        raise ValueError(f"kmath: {what} = {value} (multiples of {unit}{', 0 allowed' if zero else ''})")
    return vals


def rms_norm(n: int | tuple[int, ...], *, eps: float, weight: bool = False, lanes: int = 32, loop: str = "",
             inline: bool = True, minsize: bool = False) -> Fn:
    """out = x / rms(x) (n bf16 -> fp32; ``weight``: times w, n fp32, by ``al::mulff``): a function ``(x, [w,] out)``;
    1 / rms through ``al::inv_rms`` (the scalar invsqrt) of ``aie::mac_square``'s sums in 64 lanes, then per
    ``lanes`` (32 or 64) values x times its bf16 parts (2^-16). ``loop``: that loop's pragma ("" or "min1": no hardware
    loop). ``n`` a tuple of values (multiples of 64): a function ``(x, [w,] out, n)`` of the run-time n (module
    docstring). ``minsize``: for a ``minsize`` caller (``make``)."""
    check("lanes", lanes, (32, 64))
    check("loop", loop, tuple(PRAGMAS))
    ns = _extent("rms_norm n", n, 64)
    rt = len(ns) > 1
    inv_n = " : ".join(f"n == {v} ? 1.0f / {v}" for v in ns[:-1]) + (" : " if rt else "") + f"1.0f / {ns[-1]}"
    local = (f"rms_{'x'.join(map(str, ns))}_l{lanes}" + ("_w" if weight else "") + (f"_{loop}" if loop else "")
             + f"_e{_tag(eps)}" + ("_ms" if minsize else ""))
    return make(local + ("_i" if inline else ""), "rows.rms_norm.cc.jinja", inline=inline, minsize=minsize, lanes=lanes,
                arg="ptr",
                ret="ptr", func="rms_norm",
                params={"n": ns[0] if not rt else ns, "eps": eps, "weight": weight, "lanes": lanes, "loop": loop},
                variables={"n": "n" if rt else ns[0], "rt": rt, "label": f"n = {' | '.join(map(str, ns))}" if rt else ns[0],
                           "inv_n": inv_n, "eps": repr(float(eps)), "weight": weight, "L": lanes,
                           "pragma": PRAGMAS[loop] or ("AIE_LOOP_NO_UNROLL" if rt else "")})


def rope(rot: int | tuple[int, ...], *, dist: int | tuple[int, ...] | None = None, table: str = "fp32",
         inline: bool = True, minsize: bool = False) -> Fn:
    """The rotation of the pairs (i, i + ``dist``) of an fp32 row, i < ``rot`` (multiples of 32; ``dist``: default
    ``rot``, NeoX's halves), in place: a function ``(row, cs, [rot,] [dist])``, cos at cs + i and sin at cs + rot + i;
    ``table`` "fp32" (``al::rot2``) or "bf16" (a bf16 table: four products per output, a's and b's bf16 parts times c
    and s: rot2's values, its products of c's and s's lo parts 0). ``rot`` (may be 0: no rotation) or ``dist`` a tuple
    of values: the run-time argument (module docstring). ``minsize``: for a ``minsize`` caller (``make``)."""
    rots = _extent("rope rot", rot, 32, zero=True)
    dists = rots if dist is None else _extent("rope dist", dist, 32)
    check("table", table, ("fp32", "bf16"))
    if min(dists) < max(rots):
        raise ValueError(f"kmath: rope of rot = {rot}, dist = {dist} (dist >= rot)")
    rr, dr = len(rots) > 1, len(dists) > 1
    local = (f"rope_{'x'.join(map(str, rots))}" + (f"_d{'x'.join(map(str, dists))}" if dists != rots else "")
             + ("_tb" if table == "bf16" else "") + ("_ms" if minsize else "") + ("_i" if inline else ""))
    return make(local, "rows.rope.cc.jinja", inline=inline, minsize=minsize, lanes=32, arg="ptr", ret="ptr", func="rope",
                params={"rot": rots[0] if not rr else rots, "dist": dists[0] if not dr else dists, "table": table},
                variables={"rot": "rot" if rr else rots[0], "dist": "dist" if dr else dists[0], "rr": rr, "dr": dr,
                           "label": f"{' | '.join(map(str, rots))}" if rr else rots[0],
                           "dlabel": f"{' | '.join(map(str, dists))}" if dr else dists[0],
                           "T": "bfloat16" if table == "bf16" else "float",
                           "pragma": "AIE_LOOP_NO_UNROLL" if rr or dr else ""})
