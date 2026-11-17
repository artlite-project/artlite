"""The Flint decode overlay (models/flint: ``FlintOverlay``) on the NPU:

- slices of layers with a truncated vocabulary against the numpy reference (``ref.layer``, fp32 weights): a dense
  full-attention layer (0, no positional encoding), an MoE sliding-window layer (2), an MoE full layer (5) and layers 0 .. 5
  over 600 positions (past the window of 513) in steps of 1 .. 4 tokens, K cached as hi + lo or single blocks: the
  logits, the residual after the last layer's attention;
- steps of S slots give each slot the logits of a step of one, bit for bit (the attention's sub-steps, the MoE's partials
  summed on the slots' grid whatever the union of the step's experts), also where a step continues from fewer slots than
  the last one ran (rejected drafts).
"""

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig, flint_gguf
from models.flint.overlay import FlintOverlay
from models.tokenizer import BpeTokenizer

from ..common import essay
from ..model_files import require
from ..numerics import rel
from . import ref

VOCAB = 1024
STEPS = (1, 4, 3, 2, 4, 4, 1, 3, 3, 2, 1, 4)  # (step sizes, repeated: every alignment to the records and their pairs)
TOL = 0.03  # (relative rms error of the logits: bfp16 x panels, bf16 sums, the caches' blocks; ~1% here; a position whose
# routing flips a near-tied expert up to ~6%: the 95th percentile and the median are held)


@pytest.fixture(scope="module")
def flint() -> tuple[GGUFFile, FlintConfig]:
    path = flint_gguf()
    require(path)
    g = GGUFFile(path)
    return g, FlintConfig.from_gguf(g)


def _ids(g: GGUFFile, n: int) -> list[int]:
    return BpeTokenizer(g).encode(essay("apple.txt"))[:n]


def _run(ov: FlintOverlay, x: np.ndarray, steps: tuple[int, ...]) -> np.ndarray:
    """The logits (fp32) of every position of x, in steps of the given sizes (repeated) from position 0."""
    out = np.zeros((x.shape[0], ov.vocab), np.float32)
    p, i = 0, 0
    while p < x.shape[0]:
        n = min(steps[i % len(steps)], x.shape[0] - p)
        i += 1
        out[p:p + n] = np.asarray(ov.step(x[p:p + n], p), np.float32)
        p += n
    return out


@pytest.mark.parametrize(("layers", "n", "kv_k", "steps"),
                         [([0], 40, "hilo", (1,)), ([2], 40, "hilo", STEPS), ([5], 40, "single", STEPS),
                          ([0, 1, 2, 3, 4, 5], 600, "hilo", STEPS)],
                         ids=["dense-full", "moe-swa", "moe-full-single", "six"])
def test_flint_overlay_layers(flint: tuple[GGUFFile, FlintConfig], layers: list[int], n: int, kv_k: str,
                              steps: tuple[int, ...]) -> None:
    g, cfg = flint
    ids = _ids(g, n)
    x = g.rows("token_embd.weight", np.asarray(ids)).astype(np.float32)
    h, pos = x.copy(), np.arange(n)
    for L in layers:
        out = ref.layer(g, cfg, L, h, pos, q8=False)
        ffn_in, h = out["ffn_inp"], out["l_out"]
    _, want = ref.head_logits(g, cfg, h, np.arange(VOCAB), q8=False)
    ov = FlintOverlay(g, layers=layers, vocab=VOCAB, max_positions=1024, kv_k=kv_k)
    ov.load()
    got = _run(ov, x, steps)
    errs = np.array([rel(got[p], want[p]) for p in range(n)])
    agree = float(np.mean(got.argmax(1) == want.argmax(1)))
    p0 = n - _sizes(n, steps)[-1]  # (the last step's first slot: its residual after the last layer's attention)
    ffn = rel(ov.ffn_in(layers[-1]), ffn_in[p0])
    print(f"logits median {np.median(errs):.4f}, p95 {np.percentile(errs, 95):.4f}, max {errs.max():.4f}, argmax {agree:.3f}, "
          f"ffn_in {ffn:.4f}")
    assert np.median(errs) < TOL / 2 and np.percentile(errs, 95) < TOL and ffn < TOL and agree > 0.9, (errs, agree, ffn)


def _sizes(n: int, steps: tuple[int, ...]) -> list[int]:
    """The step sizes ``_run`` takes over n positions."""
    out, p, i = [], 0, 0
    while p < n:
        out.append(min(steps[i % len(steps)], n - p))
        p += out[-1]
        i += 1
    return out


@pytest.mark.smoke
def test_flint_overlay_steps(flint: tuple[GGUFFile, FlintConfig]) -> None:
    """Steps of 1 .. 4 slots give the logits of steps of one, bit for bit (a dense and an MoE layer); so do steps that
    continue from fewer slots than the last one ran (a generation's rejected drafts: the slots past the accepted ones hold
    other tokens)."""
    g, _ = flint
    ids = _ids(g, 60)
    x = g.rows("token_embd.weight", np.asarray(ids)).astype(np.float32)
    ov = FlintOverlay(g, layers=[1, 2], vocab=VOCAB, max_positions=512)
    ov.load()
    one = _run(ov, x, (1,))
    ov.reset()
    assert np.array_equal(_run(ov, x, STEPS), one)
    ov.reset()
    rng = np.random.default_rng(0)
    p = 0
    while p < len(ids):
        n = min(int(rng.integers(1, 5)), len(ids) - p)
        acc = int(rng.integers(1, n + 1))
        xs = x[p:p + n].copy()
        xs[acc:] = g.rows("token_embd.weight", rng.integers(0, 150000, n - acc)).astype(np.float32)
        got = np.asarray(ov.step(xs, p), np.float32)
        assert np.array_equal(got[:acc], one[p:p + acc]), p
        p += acc
