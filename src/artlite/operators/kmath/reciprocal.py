"""1 / x (template: ``reciprocal.cc.jinja``)."""

from __future__ import annotations

import numpy as np

from .fn import LANES, Fn, bound, check, make

# recip's bound per steps (relative; the Newton steps square a 1/17 guess's error, the fp32 products' rounding stays)
_RECIP_REL = {2: 4.1e-5, 3: 1.5e-5}


def recip(lanes: int, *, domain: str = "unit", steps: int = 2, unroll: bool = False, inline: bool = True) -> Fn:
    """1 / x of ``lanes`` fp32 lanes to ~2^-16: ``domain`` "unit" (x in [1, 2]) or "normal" (normal x > 0: the
    mantissa's reciprocal, the exponent negated in the bits); ``steps`` Newton steps after a linear guess (2 or 3), a
    loop (each step needs the last: not pipelined; ~80 bundles less than unrolled at the same cycles in gdn's
    softplus, sigmoid) or ``unroll``ed (inlined in silu: ~7% faster)."""
    check("lanes", lanes, LANES)
    check("domain", domain, ("unit", "normal"))
    check("steps", steps, (2, 3))
    rel = _RECIP_REL[steps]
    dom = (1.0, 2.0) if domain == "unit" else (2.0 ** -126, 2.0 ** 125)
    local = f"recip_{lanes}" + ("_normal" if domain == "normal" else "") + f"_n{steps}" + ("_u" if unroll else "")
    local += "" if inline else "_o"
    return make(local, "reciprocal.cc.jinja", inline=inline, lanes=lanes, arg="float", ret="float", func="recip",
                params={"domain": domain, "steps": steps, "unroll": unroll},
                domain=(dom[0], np.nextafter(dom[1], np.inf)), valid=(dom[0], np.nextafter(dom[1], np.inf)), rel=rel,
                variables={"N": lanes, "normal": domain == "normal", "steps": steps, "unroll": unroll,
                           "bound": bound(None, rel), "dom": "[1, 2]" if domain == "unit" else "[2^-126, 2^125]"})
