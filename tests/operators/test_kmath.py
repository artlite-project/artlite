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
from .elementwise import KmOperator, VtanhOperator

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
    "exp2-chord": ("exp2", {"lanes": 32, "bias": 0, "prec": "chord"}),
    "exp2-chord2": ("exp2", {"lanes": 32, "bias": 0, "prec": "chord2"}),
    "recip-n2": ("recip", {"lanes": 32, "unroll": True}),
    "recip-n3": ("recip", {"lanes": 16, "steps": 3, "inline": False}),
    "recip-normal": ("recip", {"lanes": 16, "domain": "normal", "steps": 3, "inline": False}),
    "expneg": ("expneg", {"lanes": 16}),
    "expm1": ("expm1", {"lanes": 16}),
    "softplus": ("softplus", {"lanes": 16}),
    "sigmoid": ("sigmoid", {"lanes": 16}),
    "silu": ("silu", {"lanes": 32}),
    "act-silu": ("act", {"table": "silu"}),
    "act-sigmoid": ("act", {"table": "sigmoid"}),
    "act-gelu": ("act", {"table": "gelu"}),
    "act-gelu_erf": ("act", {"table": "gelu_erf"}),
    "silu_tanh": ("silu_tanh", {}),
}

EDGES = np.float32([-np.inf, -1e30, -4e6, -32000, -16384, -1000, -200, -127.5, -127.25, -127, -126.9, -126.5, -126.1, 0, 100,
                    127, 128, 128.25, 255.5, 256, 300, 1000, 16384, 1e30, np.inf])


def _arguments(fn: kmath.Fn) -> tuple[np.ndarray, np.ndarray]:
    """The domain's arguments (every bf16 there, for an accumulator argument every bf16 exponent plus the bias, and
    2^18 fp32 ones between: log-uniform over a wide positive range; for a bf16 argument those rounded to bf16) and the
    edges in ``fn.valid``."""
    lo, hi = fn.domain
    if fn.arg == "ptr":  # (an array function: every finite bf16 but the tiny ones whose result is denormal)
        b = kmath.bf16_values()
        b = b[(np.abs(b) >= 1e-30) | (b == 0)]
        return b[(b >= lo) & (b < hi)], b[(b < lo) | (b >= hi)]
    bias = np.float32(192 if fn.arg == "accum" else 0)
    b = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().numpy()
    b = b[np.isfinite(b)] + bias
    rng = np.random.default_rng(0)
    wide = lo > 0 and hi / lo > 1e3
    r = np.exp2(rng.uniform(np.log2(lo), np.log2(hi), 1 << 18)) if wide else rng.uniform(lo, hi, 1 << 18)
    dom = np.concatenate([b[(b >= lo) & (b < hi)], r]).astype(np.float32)
    if fn.arg == "bfloat16":
        dom = kmath_sim.bf16(dom)
    return dom, EDGES[(EDGES >= fn.valid[0]) & (EDGES < fn.valid[1]) & ((EDGES < lo) | (EDGES >= hi))]


def _contracts() -> list[kmath.Fn]:
    fns = []
    for lanes, inline in itertools.product(kmath.LANES, (False, True)):
        fns += [kmath.exp2(lanes, prec="bf16", below=b, inline=inline) for b in ("clamp", "zero", "int")]
        fns += [kmath.exp2(lanes, prec="fp32", inline=inline), kmath.silu(lanes, inline=inline)]
        fns += [kmath.exp2(lanes, bias=0, prec=p, inline=inline) for p in ("chord", "chord2")]
        fns += [kmath.recip(lanes, domain=d, steps=n, unroll=u, inline=inline)
                for d, n, u in itertools.product(("unit", "normal"), (2, 3), (False, True))]
        fns += [f(lanes, inline=inline) for f in (kmath.expneg, kmath.expm1, kmath.softplus, kmath.sigmoid)]
        fns += [kmath.sigmoid(lanes, recip_fn=kmath.recip(lanes), inline=inline),
                kmath.silu(lanes, exp_fn=kmath.exp2(lanes, below="zero"), inline=inline),
                kmath.expm1(lanes, exp_fn=kmath.exp2(lanes, inline=True), inline=inline)]
        fns += [kmath.exp2(lanes, bias=192, below=b, above=a, shift=s, inline=inline)
                for b, a, s in itertools.product(("clamp", "bounded", "none"), ("none", "clamp"), (0, 64))]
    for t in kmath.TABLES.values():
        fns += [kmath.act(t)] + [kmath.table_pass(t, table_arg=a, walk=w)
                                 for a, w in itertools.product((False, True), ((1, 32), (1, 64), (2, 128)))]
        fns += [kmath.act(t, table_arg=True, walk=(2, 128))]
    return fns + [kmath.table_pass(kmath.SILU, table_arg=True, n=640), kmath.silu_tanh(walk=(2, 128)),
                  kmath.act(kmath.SILU, table_arg=True, n=640, scale=2, clamp=16.0, walk=(10, 640))]


def test_kmath_sources() -> None:
    """Every contract renders a source whose items define its name (and its parts' names, each once per name: the
    names tell the variants apart); contracts no implementation meets are errors."""
    names: dict[str, str] = {}
    for fn in _contracts():
        for f in (fn, *fn.parts.values()):
            ks = KernelSource(f.src)
            assert f.name in ks.names, f.name
            assert names.setdefault(f.name, f.src) == f.src, f"{f.name}: two definitions"
    for bad in ({"lanes": 8}, {"lanes": 32, "prec": "fp32", "below": "zero"}, {"lanes": 32, "bias": 128},
                {"lanes": 32, "shift": 64}, {"lanes": 32, "bias": 192, "prec": "fp32"}, {"lanes": 32, "above": "clamp"},
                {"lanes": 32, "bias": 0}, {"lanes": 32, "bias": 0, "prec": "chord", "below": "zero"}):
        with pytest.raises(ValueError):
            kmath.exp2(**bad)
    for bad in (lambda: kmath.silu(32, exp_fn=kmath.exp2(32, below="int")),  # (silu's argument reaches -3e38)
                lambda: kmath.silu(32, exp_fn=kmath.exp2(16)), lambda: kmath.sigmoid(16, recip_fn=kmath.exp2(16)),
                lambda: kmath.softplus(16, recip_fn=kmath.recip(16, domain="normal")), lambda: kmath.recip(16, steps=1),
                lambda: kmath.table_pass(kmath.GELU, walk=(2, 48)), lambda: kmath.table_pass(kmath.SILU, n=48),
                lambda: kmath.silu_tanh(walk=(0, 32)), lambda: kmath.act(kmath.SILU, scale=3),
                lambda: kmath.act(kmath.SIGMOID, scale=2)):
        with pytest.raises(ValueError):
            bad()


def test_kmath_shared_dependency() -> None:
    """Two functions calling one exp2 (softplus, expm1): a source rendering both defines it once."""
    tu = KernelSource(kmath.softplus(16).src + kmath.expm1(16).src).text
    assert tu.count("__attribute__((noinline)) aie::vector<float, 16> km_exp2_v16_fp32_") == 1


@pytest.fixture(scope="module")
def vtanh_table() -> np.ndarray:
    """vtanh of every bf16 (by its bits) as the NPU computes it: the model's (``kmath_sim.load_vtanh``)."""
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return VtanhOperator(size=65536).compile().run(bits).float().numpy()


@pytest.mark.parametrize("case", list(CASES))
def test_kmath(case: str, request: pytest.FixtureRequest) -> None:
    func, kwargs = CASES[case]
    if func == "silu_tanh":
        kmath_sim.load_vtanh(request.getfixturevalue("vtanh_table"))
    fn = getattr(kmath, func)(**{k: kmath.TABLES[v] if k == "table" else v for k, v in kwargs.items()})
    dom, edges = _arguments(fn)
    x = np.concatenate([dom, edges])
    xp = np.concatenate([x, np.full(-len(x) % 1024, dom[0])])
    if fn.arg == "ptr":
        xt = torch.from_numpy(xp).bfloat16()
        out = KmOperator(size=len(xp), func=func, **kwargs).compile().run(xt).float().numpy()[:len(x)]
    else:
        out = KmOperator(size=len(xp), func=func, **kwargs).compile().run(torch.from_numpy(xp)).numpy()[:len(x)]
    em = kmath_sim.emulate(fn, x)
    same = (out == em) | (np.isnan(out) & np.isnan(em))  # (zeros of either sign alike)
    assert same.all(), (f"{(~same).sum()} results differ from the emulation; at {x[~same][:4]}: {out[~same][:4]} vs "
                        f"{em[~same][:4]}")
    o, ref = out[:len(dom)].astype(np.float64), kmath_sim.exact(fn, dom)
    keep = np.abs(ref) >= 2.0 ** -125  # (results below 2^-126 are flushed to zero, and those just above it may be)
    o, ref, dom = o[keep], ref[keep], dom[keep]
    if fn.ulp is not None:
        e = np.abs(o - ref) / np.exp2(np.floor(np.log2(np.abs(ref))) - 7)
        print(f"\n{fn.name}: max {e.max():.3f} bf16 ULP, mean {e.mean():.3f}")
        assert e.max() <= fn.ulp, f"max {e.max():.3f} bf16 ULP at {dom[np.argmax(e)]}"
    if fn.rel is not None:
        r = np.abs(o / ref - 1)
        print(f"\n{fn.name}: max rel. error {r.max():.3g}")
        assert r.max() <= fn.rel, f"max rel. error {r.max():.3g} at {dom[np.argmax(r)]}"
