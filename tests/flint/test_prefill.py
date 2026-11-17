"""The Flint prefill (models/flint/prefill.py: ``FlintPrefill``; its GEMMs: prefill/test_gemm.py, its attention:
test_attn_prefill.py, its MoE block: moe/test_prefill.py) on the NPU:

- a slice of six layers (dense full, dense full, MoE sliding-window x 3, MoE full) against the decode path on the same
  tokens: a prompt from position 0 and after decode steps or a first prompt's chunks, past the window of 513, K hi + lo
  and single, a chunk of 2048: the caches and the last token's logits (the LM head alone on the chunk's last output row:
  ``ModelOverlay.logits_of``, or a decode step after the chunks);
- the same slice's output rows against the numpy reference (``ref.layer``, fp32 weights).

The two paths round differently, and a position whose routing has two near-tied experts can take either: the errors per
position are held in their median and 95th percentile. Against the fp32 reference the decode's caches are within 0.2-1.5%
(median), the prefill's within 0.7-4.4%: its GEMMs' operands as bfp16 panels (xn, the attention's output, the FFN's
hidden: the block of 8 around a large value loses the small ones; 2.3% of layer 2's V in a numpy model of these roundings
alone), their bf16 C tiles summed across k-tiles, the bf16 residual. The model-level gate is test_npu.py's KL against
llama.cpp.
"""

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig, flint_gguf
from models.flint.overlay import FlintOverlay
from models.flint.prefill import FlintPrefill
from models.tokenizer import BpeTokenizer

from ..common import essays
from ..model_files import require
from ..numerics import rel
from . import ref

VOCAB = 1024
LAYERS = [0, 1, 2, 3, 4, 5]
SIZES = (64, 128, 256, 512)  # (the tests' library: fewer designs to compile)
BIG = (64, 1024, 2048)  # (prep blocks of 1024; chunks of 4096 and 8192: the attention in parts of 2048)
# (a first prompt's tokens through the prefill, then steps (their sizes), then the prompt's tokens, K, sizes)
CASES = [(0, (), 300, "hilo", SIZES), (0, (4,) * 15 + (1, 3), 700, "hilo", SIZES),
         (128, (2,) + (4,) * 15 + (2,), 500, "single", SIZES), (0, (), 2100, "hilo", BIG),
         (0, (), 4200, "hilo", (64, 4096)), (192, (4,) * 16, 8300, "single", (64, 2048, 8192))]
# (the steps end at a whole cache group: a chunk starts there)
assert all((f + sum(s)) % 64 == 0 for f, s, *_ in CASES)


@pytest.fixture(scope="module")
def flint() -> tuple[GGUFFile, FlintConfig]:
    path = flint_gguf()
    require(path)
    g = GGUFFile(path)
    return g, FlintConfig.from_gguf(g)


def _x(g: GGUFFile, n: int) -> np.ndarray:
    ids = BpeTokenizer(g).encode(essays()[:16 * n])[:n]
    assert len(ids) == n
    return g.rows("token_embd.weight", np.asarray(ids)).astype(np.float32)


def _steps(ov: FlintOverlay, x: np.ndarray, p: int, sizes: tuple[int, ...]) -> int:
    for S in sizes:
        ov.step(x[p:p + S], p, logits=False)
        p += S
    return p


def _prompt(ov: FlintOverlay, pf: FlintPrefill, x: np.ndarray, p: int) -> np.ndarray:
    """x[p:] through the prefill's chunks, the tokens they leave as decode steps: the last token's logits (fp32)."""
    n = x.shape[0]
    m = 0
    for T in pf.plan(p, n - p):
        m = min(T, n - p)
        pf.run(x[p:p + m], p, T)
        p += m
    if p == n:
        return np.asarray(ov.logits_of(pf.hidden(m - 1)), np.float32)
    while p < n:
        S = min(4, n - p)
        out = np.asarray(ov.step(x[p:p + S], p), np.float32)
        p += S
    return out[-1]


def _rows(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Per position (axis -2) the relative rms error over its other axes."""
    g, w = np.moveaxis(got, -2, 0), np.moveaxis(want, -2, 0)
    return np.array([rel(a, b) for a, b in zip(g, w, strict=True)])


@pytest.mark.parametrize(("first", "steps", "n", "kv_k", "sizes"), CASES,
                         ids=[f"{f}-after{f + sum(s)}-n{n}-{k}-T{max(z)}" for f, s, n, k, z in CASES])
def test_flint_prefill_layers(flint: tuple[GGUFFile, FlintConfig], first: int, steps: tuple[int, ...], n: int,
                              kv_k: str, sizes: tuple[int, ...]) -> None:
    """The caches the prefill leaves against the decode path's (steps of 4) for the same essay tokens: per position the
    K of every layer within 3.5% (median; 0.7-3.1%, single K the most) and 5% (95th percentile), the V within 6% and 8%
    (3.4-4.8%, up to 6.7%), the last token's logits within 5% (0.8-2.7%)."""
    g, _ = flint
    p0 = first + sum(steps)
    x = _x(g, p0 + n)
    ov = FlintOverlay(g, layers=LAYERS, vocab=VOCAB, max_positions=-(-(p0 + n) // 1024) * 1024 + 1024, kv_k=kv_k)
    ov.load()
    p = 0
    while p < x.shape[0]:
        S = min(4, x.shape[0] - p)
        want_l = np.asarray(ov.step(x[p:p + S], p), np.float32)[-1]
        p += S
    pos = range(x.shape[0])
    want = {L: ov.read_kv(L, pos) for L in LAYERS}
    ov.reset()
    pf = FlintPrefill(ov, sizes=sizes).compile().bind()
    if first:
        _prompt(ov, pf, x[:first], 0)
    got_l = _prompt(ov, pf, x, _steps(ov, x, first, steps))
    worst = {"logits": (rel(got_l, want_l),) * 2}
    for L in LAYERS:
        k, v = ov.read_kv(L, pos)
        for name, a, b in (("K", k, want[L][0]), ("V", v, want[L][1])):
            e = _rows(a, b)
            worst[f"{name}.{L}"] = (float(np.median(e)), float(np.percentile(e, 95)))
    print({k: tuple(round(x, 4) for x in v) for k, v in worst.items()})
    assert worst.pop("logits")[0] < 0.05, worst
    assert all(m < (0.035 if k[0] == "K" else 0.06) and p95 < (0.05 if k[0] == "K" else 0.08)
               for k, (m, p95) in worst.items()), worst


@pytest.mark.parametrize("n", [600, 2048])
def test_flint_prefill_ref(flint: tuple[GGUFFile, FlintConfig], n: int) -> None:
    """The slice's output rows (the residual after layer 5) of a prompt from position 0 against the numpy reference:
    per row within 3% (median) and 6% (95th percentile)."""
    g, cfg = flint
    x = _x(g, n)
    h, pos = x.copy(), np.arange(n)
    for L in LAYERS:
        h = ref.layer(g, cfg, L, h, pos, q8=False)["l_out"]
    ov = FlintOverlay(g, layers=LAYERS, vocab=VOCAB, max_positions=4096)
    ov.load()
    pf = FlintPrefill(ov, sizes=BIG if n > 512 else SIZES).compile().bind()
    p = 0
    rows = []
    for T in pf.plan(0, n, steps=False):
        m = min(T, n - p)
        pf.run(x[p:p + m], p, T)
        rows.append(pf.hidden(0, m))
        p += m
    assert p == n
    e = _rows(np.concatenate(rows), h)
    print(f"rows: median {np.median(e):.4f}, p95 {np.percentile(e, 95):.4f}, max {e.max():.4f}")
    assert np.median(e) < 0.03 and np.percentile(e, 95) < 0.06
