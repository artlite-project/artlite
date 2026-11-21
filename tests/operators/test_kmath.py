"""kmath's functions on the NPU (operators/kmath/): every variant over its domain (every bf16 argument there and
fp32 ones between) and the edges its contract covers (masked lanes, clamps): bit-exact against its numpy model
(kmath_sim.py), within its error bound against the exact function."""

import itertools

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.operators import bfp16, kmath
from artlite.operators.kernel_source import KernelSource

from . import kmath_sim
from .elementwise import InvOperator, KmOperator, RotOperator, RowOperator, SoftmaxOperator, VtanhOperator

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
    "exp2-b192-quad": ("exp2", {"lanes": 32, "bias": 192, "below": "bounded", "prec": "quad"}),
    "exp2-chord": ("exp2", {"lanes": 32, "bias": 0, "prec": "chord"}),
    "exp2-chord2": ("exp2", {"lanes": 32, "bias": 0, "prec": "chord2"}),
    "recip-n2": ("recip", {"lanes": 32, "unroll": True}),
    "recip-n3": ("recip", {"lanes": 16, "steps": 3, "inline": False}),
    "recip-normal": ("recip", {"lanes": 16, "domain": "normal", "steps": 3, "inline": False}),
    "expneg": ("expneg", {"lanes": 16}),
    "expm1": ("expm1", {"lanes": 16}),
    "softplus": ("softplus", {"lanes": 16}),
    "sigmoid": ("sigmoid", {"lanes": 16}),
    "sigmoid-bf16": ("sigmoid", {"lanes": 32, "prec": "bf16"}),
    "exp": ("exp", {"lanes": 16}),
    "exp-bf16-neg": ("exp", {"lanes": 64, "arg": "bfloat16", "neg": True}),
    "silu": ("silu", {"lanes": 32}),
    "act-silu": ("act", {"table": "silu"}),
    "act-silu-f32": ("act", {"table": "silu", "f32": True}),
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
        fns += [f(lanes, inline=inline) for f in (kmath.expneg, kmath.expm1, kmath.softplus, kmath.sigmoid, kmath.exp)]
        fns += [kmath.exp(lanes, exp_fn=kmath.exp2(lanes, below="zero", inline=True), inline=inline)]
        fns += [kmath.sigmoid(lanes, prec="bf16", exp_fn=e, inline=inline) for e in (None, kmath.exp2(lanes, below="zero"))]
        fns += [kmath.exp(lanes, arg="bfloat16", neg=n, inline=inline) for n in (False, True)]
        fns += [kmath.sigmoid(lanes, recip_fn=kmath.recip(lanes), inline=inline),
                kmath.silu(lanes, exp_fn=kmath.exp2(lanes, below="zero"), inline=inline),
                kmath.expm1(lanes, exp_fn=kmath.exp2(lanes, inline=True), inline=inline)]
        fns += [kmath.exp2(lanes, bias=192, below=b, above=a, shift=s, inline=inline)
                for b, a, s in itertools.product(("clamp", "bounded", "none"), ("none", "clamp"), (0, 64))]
    for t in kmath.TABLES.values():
        fns += [kmath.act(t), kmath.act(t, f32=True)] + [kmath.table_pass(t, table_arg=a, walk=w)
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
def inv_results() -> dict[str, np.ndarray]:
    """aie::inv's and aie::invsqrt's results over [1, 4) as the NPU computes them: the model's (``kmath_sim.load_inv``)."""
    x = kmath_sim.inv_arguments()
    return {op: InvOperator(size=len(x), op=op).compile().run(x) for op in ("inv", "invsqrt")}


@pytest.fixture(scope="module")
def vtanh_table() -> np.ndarray:
    """vtanh of every bf16 (by its bits) as the NPU computes it: the model's (``kmath_sim.load_vtanh``)."""
    bits = np.arange(65536, dtype=np.int32).astype(np.uint16).view(bfloat16)
    return VtanhOperator(size=65536).compile().run(bits).astype(np.float32)


@pytest.mark.parametrize("case", list(CASES))
def test_kmath(case: str, request: pytest.FixtureRequest) -> None:
    func, kwargs = CASES[case]
    if func == "silu_tanh":
        kmath_sim.load_vtanh(request.getfixturevalue("vtanh_table"))
    fn = getattr(kmath, func)(**{k: kmath.TABLES[v] if k == "table" else v for k, v in kwargs.items()})
    if "aie::inv" in fn.src or "al::inv_rms" in fn.src:
        for op, r in request.getfixturevalue("inv_results").items():
            kmath_sim.load_inv(op, r)
    dom, edges = _arguments(fn)
    x = np.concatenate([dom, edges])
    xp = np.concatenate([x, np.full(-len(x) % 1024, dom[0])])
    if fn.arg == "ptr" and not fn.params.get("f32"):
        xt = xp.astype(np.float32).astype(bfloat16)
        out = KmOperator(size=len(xp), func=func, **kwargs).compile().run(xt).astype(np.float32)[:len(x)]
    else:
        out = KmOperator(size=len(xp), func=func, **kwargs).compile().run(xp)[:len(x)]
    em = kmath_sim.emulate(fn, x)
    same = (out == em) | (np.isnan(out) & np.isnan(em))  # (zeros of either sign alike)
    if fn.params.get("f32"):  # (unrounded: the accumulator's sum, which the model takes in fp32 steps)
        same |= np.abs(out - em) <= 4 * np.spacing(np.abs(em))
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


ROWS = {  # id: a row function's arguments; and for run-time extents the values they take in the run
    "rms-256-w": ("rms_norm", {"n": 256, "eps": 1e-6, "weight": True}),
    "rms-256-w-min1": ("rms_norm", {"n": 256, "eps": 1e-6, "weight": True, "loop": "min1"}),
    "rms-128-w-l64": ("rms_norm", {"n": 128, "eps": 1e-5, "weight": True, "lanes": 64}),
    "rms-512": ("rms_norm", {"n": 512, "eps": 1e-6}),
    "rms-192": ("rms_norm", {"n": 192, "eps": 1e-6}),
    "rope-32": ("rope", {"rot": 32}),
    "rope-64": ("rope", {"rot": 64}),
    "rope-64-d128": ("rope", {"rot": 64, "dist": 128}),
    "rope-128-bf16": ("rope", {"rot": 128, "table": "bf16"}),
    "rms-rt-512": ("rms_norm", {"n": (256, 512), "eps": 1e-6, "minsize": True}, {"n": 512}),
    "rms-rt-256-w-l64": ("rms_norm", {"n": (256, 512), "eps": 1e-6, "weight": True, "lanes": 64}, {"n": 256}),
    "rope-rt-64": ("rope", {"rot": (64, 32), "dist": 128}, {"rot": 64}),
    "rope-rt-32-d256": ("rope", {"rot": (64, 32), "dist": (128, 256)}, {"rot": 32, "dist": 256}),
}


@pytest.mark.parametrize("case", list(ROWS))
def test_kmath_rows(case: str, request: pytest.FixtureRequest) -> None:
    """A row function on 256 rows of random values (rows of scales 2^-20 .. 2^20): bit-exact against its model
    (kmath_sim; rope: 99% of its results, the rest within 2 ULP: the model's MACs after a cancellation), and within its
    bound of the exact result (rms_norm: 2^-15 relative per element; rope: 2^-15.8 of |a| + |b| per pair)."""
    func, kwargs, *at = ROWS[case]
    at = at[0] if at else {}
    fn = getattr(kmath, func)(**kwargs)
    for op, r in request.getfixturevalue("inv_results").items():
        kmath_sim.load_inv(op, r)
    rng = np.random.default_rng(1)
    n = at.get("n", kwargs.get("n", 256)) if func == "rms_norm" else max(256, 2 * at.get("dist", 0))  # (rows hold dist + rot)
    rows = 72 if n == 192 else 256  # (a tile of whole rows)
    scale = np.exp2(rng.uniform(-20, 20, (rows, 1)))
    x = (rng.standard_normal((rows, n)) * scale).astype(np.float32)
    if func == "rms_norm":
        x = kmath_sim.bf16(x)
        w = rng.uniform(0.25, 4, n).astype(np.float32) if kwargs.get("weight") else None
        out = RowOperator(size=rows * n, fn=fn, n=n, **({"w": w} if w is not None else {})).compile().run(
            x.astype(bfloat16).ravel()).reshape(rows, n)
        em = kmath_sim.rms_norm(fn, x, w)
        x64 = x.astype(np.float64)
        ref = x64 / np.sqrt(np.mean(x64 ** 2, axis=1, keepdims=True) + kwargs["eps"]) * (1 if w is None else w)
        keep = ref != 0
        err = np.abs(out[keep] - ref[keep]) / np.abs(ref[keep])
        bound = 2.0 ** -15
    else:
        rot = at.get("rot", kwargs["rot"])
        d = at.get("dist", fn.params["dist"])
        th = rng.uniform(0, 2 * np.pi, rot)
        cs, sn = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
        if kwargs.get("table") == "bf16":
            cs, sn = kmath_sim.bf16(cs), kmath_sim.bf16(sn)
        tables = {"cs": np.concatenate([cs, sn]).astype(bfloat16 if kwargs.get("table") == "bf16" else np.float32)}
        out = RowOperator(size=rows * n, fn=fn, n=n, rot=rot, dist=d, **tables).compile().run(x.ravel()).reshape(rows, n)
        em = kmath_sim.rope(fn, x, np.concatenate([cs, sn]), rot, d)
        a, b = x[:, :rot].astype(np.float64), x[:, d:d + rot].astype(np.float64)
        ref = x.astype(np.float64).copy()
        ref[:, :rot], ref[:, d:d + rot] = a * cs - b * sn, a * sn + b * cs
        mag = np.zeros_like(ref)
        mag[:, :rot] = mag[:, d:d + rot] = np.abs(a) + np.abs(b)
        keep = mag > 0
        err = np.abs(out - ref)[keep] / mag[keep]
        bound = 2.0 ** -15.8
    same = out == em
    if func == "rope":  # (the rotated dims: the others are copies)
        ulps = np.abs(out.astype(np.float64) - em) / np.spacing(np.abs(em)).astype(np.float64)
        assert same[keep].mean() >= 0.99 and ulps.max() <= 2, f"{same[keep].mean():.4f} as the model, max {ulps.max()} ULP apart"
    else:
        assert same.all(), f"{(~same).sum()} results differ from the model; at {np.argwhere(~same)[:4]}: {out[~same][:4]} vs {em[~same][:4]}"
    print(f"\n{fn.name}: max error {err.max():.3g}")
    assert err.max() <= bound, f"max error {err.max():.3g}"


SOFTMAX = {  # id: the exp2 of bias 192's arguments, the passes' (SoftmaxOperator's)
    "inplace-decode": ({"below": "bounded", "above": "clamp"}, {"layout": "inplace", "lsum": True, "rounding": True}),
    "split-track-lo": ({"below": "bounded"}, {"track": True, "lsum": True, "lo": True}),
    "split-quad": ({"below": "bounded", "prec": "quad"}, {}),
    "split-clamp-lsum": ({}, {"lsum": True}),
    "inplace-unclamped": ({"below": "none"}, {"layout": "inplace", "lsum": True}),
}


@pytest.mark.parametrize("case", list(SOFTMAX))
def test_kmath_softmax(case: str) -> None:
    """A softmax's two passes (``p_pass1``, ``p_pass2``) on 1024 scores A = y + 192 per run (y <= 0: s - m; y up to 20 in
    the last run: pass 1's ``track`` flag; masked scores: -16384, -1e30 where the exp2 clamps): the P panels (bfp16),
    the l sums and (``lo``) the lo panels bit-exact against their model (``kmath_sim.softmax_passes``: the exp2's parts,
    ``bfp16.blocks``' rounding, the accumulators' adder; the lo panels up to 2% of their bytes: the MMUL's subtraction, not modelled bit for bit),
    and the panels' values within 2^-6 of the block maximum of 2^y, the sums within 2^-7 of their exact value."""
    e_kw, kw = SOFTMAX[case]
    e192 = kmath.exp2(32, bias=192, **e_kw)
    n, steps = 1024, 1024 // 64
    op = SoftmaxOperator(size=n, e192=e192, **kw).compile()
    rng = np.random.default_rng(3)
    floor = -63.0 if e_kw.get("below") == "none" else -64.0  # ("none": A >= 128.5 is the caller's)
    masked = {"bounded": -16384.0, "none": None}.get(e_kw.get("below"), -1e30)
    for run in range(3):
        y = rng.uniform(floor, 0.0, (steps, 2, 32)).astype(np.float32)
        y[rng.random(y.shape) < 0.05] = 0.0
        if masked is not None and run >= 1:
            y[rng.random(y.shape) < 0.1] = masked
        if run == 2:
            y[0, 0, :4] = rng.uniform(18.0, 20.0, 4)
        a = (y + np.float32(192)).astype(np.float32)
        flat = (a if kw.get("layout", "split") == "split" else a.transpose(1, 0, 2)).reshape(-1)
        raw = op.run(flat).view(np.uint8)
        em = kmath_sim.softmax_passes(e192, a, lsum=op.lsum, lo=op.lo)
        hi = raw[:72 * steps].reshape(steps, 72)
        assert np.array_equal(hi, em["hi"]), f"run {run}: {(hi != em['hi']).sum()} panel bytes differ from the model"
        if op.lsum:
            sums = raw[op.l_off:op.l_off + 256].view(np.float32).reshape(2, 32)
            assert np.array_equal(sums, em["l"]), f"run {run}: l differs from the model at {np.argwhere(sums != em['l'])[:4]}"
        if op.lo:
            lo = raw[op.lo_off:op.lo_off + 72 * steps].reshape(steps, 72)
            assert (lo != em["lo"]).mean() <= 0.02, f"run {run}: {(lo != em['lo']).mean():.4f} of the lo bytes differ"
            x = em["r"].reshape(steps, 8, 8).astype(np.float64)
            both = bfp16.block_values(hi.reshape(steps, 8, 9)) + bfp16.block_values(lo.reshape(steps, 8, 9))
            assert (np.abs(both - x) / x.max(-1, keepdims=True)).max() <= 2.0 ** -13, f"run {run}: hi + lo"
        if op.track:
            assert bool(raw[op.flag_off:op.flag_off + 4].view(np.int32)[0]) == bool(em["below209"]), f"run {run}: track"
            # the values against the exact 2^y
        ex = np.exp2(y.astype(np.float64)).reshape(steps, 8, 8)
        hv = bfp16.block_values(hi.reshape(steps, 8, 9)).astype(np.float64)
        err = np.abs(hv - ex) / ex.max(-1, keepdims=True)
        assert err.max() <= 2.0 ** -6, f"run {run}: panel value error {err.max():.3g} of the block maximum"
        if op.lsum:
            ref = np.exp2(y.astype(np.float64)).sum(0)
            rel = np.abs(sums - ref) / ref
            assert rel.max() <= 2.0 ** -7, f"run {run}: l error {rel.max():.3g}"


@pytest.mark.parametrize("f32", [True, False], ids=["f32", "bf16"])
def test_kmath_rotate(f32: bool) -> None:
    """``kmath.rotate`` on the NPU against its model (``kmath_sim.rotate``) bit for bit, on the tan form of Qwen3-0.6B-PARO's
    rotations (random rotations without the checkpoint), and the model against the exact transform (fp64)."""
    from artlite.model_config import checkpoint
    from artlite.operators.kmath.rotate import coefs
    from artlite.paro import Rotation

    rng = np.random.default_rng(0)
    G, calls = 4, 2
    try:
        from artlite.hf_model import HfModel

        rot = HfModel(checkpoint("qwen3-0.6B-paro")).tensor("blk.2.ffn_down.weight").rot()
    except (OSError, FileNotFoundError):
        pairs = np.stack([np.concatenate([rng.permutation(128) for _ in range(24)]) for _ in range(8)])
        rot = Rotation(rng.uniform(-3, 3, (8, 3072 // 2)), pairs, rng.uniform(-2, 2, 3072), 128)
    tf = rot.tan_form()
    n = G * calls
    T = tf.T.reshape(8, -1, 128)[:, :n].transpose(1, 0, 2)  # (groups, rounds, 128)
    part = tf.partner.reshape(8, -1, 128)[:, :n].transpose(1, 0, 2)
    x = kmath_sim.bf16(rng.standard_normal((n, 128, 4)) * np.exp(rng.standard_normal((n, 1, 4))))
    w = kmath_sim.bf16(rng.uniform(0.5, 2.0, (n, 128)))
    Tb = kmath_sim.bf16(T)
    op = RotOperator(groups=G, calls=calls, f32=f32)
    inp = np.zeros((calls, op.group_ints), np.int32)
    for c in range(calls):
        g = slice(G * c, G * (c + 1))
        parts = [x[g].astype(bfloat16).view(np.uint8).reshape(-1), w[g].astype(bfloat16).view(np.uint8).reshape(-1),
                 np.concatenate([coefs(T[i], part[i]) for i in range(G * c, G * (c + 1))])]
        b = np.concatenate(parts)
        inp[c].view(np.uint8)[:b.size] = b
    out = op.compile().run(inp.reshape(-1)).reshape(calls, op.group_ints)
    got = out[:, :G * 512].view(np.float32).reshape(n, 128, 4)
    want = kmath_sim.rotate(x, w, Tb, part, f32=f32)
    assert np.array_equal(got.view(np.uint32), np.asarray(want, np.float32).view(np.uint32)), np.argwhere(got != want)[:4]
    # (against the exact rounds of the same inputs: fp64 on x w')
    exact = x.astype(np.float64) * w[..., None]
    gi = np.arange(n)[:, None]
    for r in range(8):
        exact = exact + T[:, r, :, None] * exact[gi, part[:, r]]
    err = np.sqrt(((got - exact) ** 2).mean() / (exact ** 2).mean())
    assert err < (2e-3 if f32 else 6e-3), err
