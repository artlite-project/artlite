"""kmath's functions on the NPU (operators/kmath.py): every variant over its domain (every bf16 argument there and
fp32 ones between) and the edges its contract covers (masked lanes, clamps): bit-exact against its numpy model
(kmath_sim.py), within its error bound against the exact function."""

import itertools

import numpy as np
import pytest
import torch

from artlite.operators import kmath
from artlite.operators.kernel_source import KernelSource

from . import kmath_sim
from .elementwise import KmOperator

CASES = {  # id: the function and its arguments
    "exp2": ("exp2", {"lanes": 32}),
    "exp2-zero": ("exp2", {"lanes": 64, "below": "zero"}),
    "exp2-int": ("exp2", {"lanes": 32, "below": "int"}),
    "exp2-fp32": ("exp2", {"lanes": 16, "prec": "fp32"}),
    "exp2-b192": ("exp2", {"lanes": 32, "bias": 192}),
    "exp2-b192-unclamped": ("exp2", {"lanes": 32, "bias": 192, "below": "none"}),
    "exp2-b192-sat": ("exp2", {"lanes": 32, "bias": 192, "above": "clamp"}),
    "exp2-b192-bounded": ("exp2", {"lanes": 32, "bias": 192, "below": "bounded"}),
    "exp2-b192-s64": ("exp2", {"lanes": 32, "bias": 192, "shift": 64}),
}

EDGES = np.float32([-np.inf, -1e30, -4e6, -32000, -16384, -1000, -200, -127.5, -127.25, -127, -126.9, -126.5, -126.1, 0, 100,
                    127, 128, 128.25, 255.5, 256, 300, 1000, 16384, 1e30, np.inf])


def _arguments(fn: kmath.Fn) -> tuple[np.ndarray, np.ndarray]:
    """The domain's arguments (every bf16 there, for an accumulator argument every bf16 exponent plus the bias, and
    2^18 fp32 ones between) and the edges in ``fn.valid``."""
    lo, hi = fn.domain
    bias = np.float32(192 if fn.arg == "accum" else 0)
    b = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().numpy()
    b = b[np.isfinite(b)] + bias
    rng = np.random.default_rng(0)
    dom = np.concatenate([b[(b >= lo) & (b < hi)], rng.uniform(lo, hi, 1 << 18)]).astype(np.float32)
    return dom, EDGES[(EDGES >= fn.valid[0]) & (EDGES < fn.valid[1]) & ((EDGES < lo) | (EDGES >= hi))]


def _contracts() -> list[kmath.Fn]:
    fns = []
    for lanes, inline in itertools.product(kmath.LANES, (False, True)):
        fns += [kmath.exp2(lanes, prec="bf16", below=b, inline=inline) for b in ("clamp", "zero", "int")]
        fns += [kmath.exp2(lanes, prec="fp32", inline=inline), kmath.silu(lanes, inline=inline)]
        fns += [kmath.exp2(lanes, bias=192, below=b, above=a, shift=s, inline=inline)
                for b, a, s in itertools.product(("clamp", "bounded", "none"), ("none", "clamp"), (0, 64))]
    return fns + [kmath.gelu(inline=i) for i in (False, True)]


def test_kmath_sources() -> None:
    """Every contract renders a source whose items define its name (and its parts' names, each once per name: the
    names tell the variants apart); contracts no implementation meets are errors."""
    names: dict[str, str] = {}
    for fn in _contracts():
        for f in (fn, *fn.parts.values()):
            ks = KernelSource(f.src)
            assert f.name in ks.names, f.name
            assert names.setdefault(f.name, f.src) == f.src, f"{f.name}: two definitions"
    for bad in ({"lanes": 8}, {"lanes": 32, "prec": "fp32", "below": "zero"}, {"lanes": 32, "bias": 128}, {"lanes": 32, "shift": 64},
                {"lanes": 32, "bias": 192, "prec": "fp32"}, {"lanes": 32, "above": "clamp"}):
        with pytest.raises(ValueError):
            kmath.exp2(**bad)


@pytest.mark.parametrize("case", list(CASES))
def test_kmath(case: str) -> None:
    func, kwargs = CASES[case]
    fn = getattr(kmath, func)(**kwargs)
    dom, edges = _arguments(fn)
    x = np.concatenate([dom, edges])
    xp = np.concatenate([x, np.full(-len(x) % 1024, dom[0])])
    out = KmOperator(size=len(xp), func=func, **kwargs).compile().run(torch.from_numpy(xp)).numpy()[:len(x)]
    em = kmath_sim.emulate(fn, x)
    same = out.view(np.uint32) == em.view(np.uint32)
    assert same.all(), (f"{(~same).sum()} results differ from the emulation; at {x[~same][:4]}: {out[~same][:4]} vs "
                        f"{em[~same][:4]}")
    o, ref = out[:len(dom)].astype(np.float64), kmath_sim.exact(fn, dom)
    if fn.ulp is not None:
        e = np.abs(o - ref) / np.exp2(np.floor(np.log2(ref)) - 7)
        print(f"\n{fn.name}: max {e.max():.3f} bf16 ULP, mean {e.mean():.3f}")
        assert e.max() <= fn.ulp, f"max {e.max():.3f} bf16 ULP at {dom[np.argmax(e)]}"
    if fn.rel is not None:
        r = np.abs(o / ref - 1)
        print(f"\n{fn.name}: max rel. error {r.max():.3g}")
        assert r.max() <= fn.rel, f"max rel. error {r.max():.3g} at {dom[np.argmax(r)]}"
