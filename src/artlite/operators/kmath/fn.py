"""The generated function and its machinery: ``Fn``, the contract checks and the templates' renderer.

The templates are files of this package (``<module>.<part>.cc.jinja``), rendered once by ``make`` with the function's
``name`` and ``attr`` as ordinary variables beside the contract's. ``kmath`` is the ``km`` global of ``JINJA_ENV``
(``templating``), so this package loads its files with an environment of its own (``JINJA_ENV``'s options) and never
imports ``templating``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..kernel_source import kernel_prefix

KM = kernel_prefix("km")
LANES = (16, 32, 64)
_ENV = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))), trim_blocks=True,
                   lstrip_blocks=True, undefined=StrictUndefined)


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


def make(local: str, template: str, *, variables: dict[str, object], inline: bool, deps: tuple[Fn, ...] = (),
         **meta: object) -> Fn:
    """An Fn named ``km_<local>`` from the package's template file ``template``, rendered with ``variables`` and its
    ``name`` and ``attr`` (inlined, or one out-of-line copy per core); ``deps``: the functions it calls."""
    name = f"{KM}_{local}"
    attr = "[[clang::always_inline]] inline" if inline else "__attribute__((noinline))"
    body = _ENV.get_template(template).render(**variables, name=name, attr=attr).strip("\n")
    own = "namespace {\n\n" + body + "\n\n}  // namespace\n"
    srcs = dict.fromkeys(d.src for d in deps)  # (each once, in order)
    return Fn(name=name, src="\n".join([*srcs, own]), **meta)


def check(what: str, value: object, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"kmath: {what} = {value!r}, not one of {allowed}")


def bound(ulp: float | None, rel: float | None) -> str:
    """A bound as the comments state it (none: the variant's is not measured, e.g. with a caller's exp_fn)."""
    if ulp is not None:
        return f"max {ulp:.2f} bf16 ULP"
    if rel is None:
        return "bound not measured"
    return f"max {rel:.1%}" if rel >= 0.01 else f"max 2^{math.log2(rel):.1f} relative"
