"""``prefill.rotate.PrefillRotOperator`` on the NPU against its host model bit for bit (Qwen3-0.6B-PARO's rotations, or
random ones without the checkpoint): the panels of x (``gu``: gate and up's bf16 panels, x = bf16(silu(gate) up),
kmath.silu_tanh's model) -> u = x w' (exact), the rounds on Givens pairs (the partner's bf16 value times T on the fp32
MAC), x' as hi + lo into bfp16 panels; and the model against the exact transform (fp64)."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.operators import bfp16, kmath
from artlite.paro import Rotation
from artlite.prefill.rotate import PrefillRotOperator, row_floats

from ..operators import kmath_sim as ks
from .test_gate_up import vtanh  # noqa: F401 (the fixture)


def _f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32)


def model(x: np.ndarray, tf: object) -> np.ndarray:
    """The operator's arithmetic on x (T, K) (bfp16 values) with the unit tan form tf (``kmath_sim``'s accumulator
    arithmetic): x' (T, K) as hi + lo (fp32, before the output panels' rounding)."""
    T, K = x.shape
    g = tf.group
    w = ks.bf16(tf.w)
    u = ks.mul(x.reshape(T, K // g, g), w.reshape(1, K // g, g))  # (exact: bf16 times bf16)
    pairs, coef = tf.pairs()
    cb = ks.bf16(coef)
    gi = np.arange(K // g)[:, None]
    for r in range(pairs.shape[1]):
        s, q = pairs[:, r, :, 0], pairs[:, r, :, 1]
        a, b = u[:, gi, s], u[:, gi, q]
        out = u.copy()
        out[:, gi, s] = ks.mac(a, cb[None, :, r, :, 0], ks.bf16(b))
        out[:, gi, q] = ks.mac(b, cb[None, :, r, :, 1], ks.bf16(a))
        u = out
    hi = ks.bf16(u)  # (out as hi + lo: the panels' rounding of their sum)
    return ks.add(hi, ks.bf16(ks.add(u, -hi))).astype(np.float32).reshape(T, K)


def _rotation(K: int, rng: np.random.Generator, layer: int = 1) -> Rotation:
    """Qwen3-0.6B-PARO's rotation of a K (a layer's q or down), or a random one (another K, or no checkpoint)."""
    try:
        from artlite.hf_model import HfModel
        from artlite.model_config import checkpoint

        name = {3072: f"blk.{layer}.ffn_down.weight", 1024: f"blk.{layer}.attn_q.weight"}[K]
        return HfModel(checkpoint("qwen3-0.6B-paro")).tensor(name).rot()
    except (OSError, FileNotFoundError, KeyError):
        pairs = np.stack([np.concatenate([rng.permutation(128) for _ in range(K // 128)]) for _ in range(8)])
        return Rotation(rng.uniform(-3, 3, (8, K // 2)), pairs, rng.uniform(-2, 2, K), 128)


def _gu_input(T: int, K: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """GU's panels [T / 8][K / 8][gate 64 | up 64] (bf16, [token][channel] each) and x = bf16(silu(gate) up) (T, K)."""
    gu = (rng.standard_normal((T // 8, K // 8, 2, 8, 8)) * 2).astype(bfloat16)
    g = gu[:, :, 0].astype(np.float32)
    y = np.asarray(ks.emulate(kmath.silu_tanh(walk=(2, 128)), g.reshape(-1)), np.float32).reshape(g.shape)
    h = (torch.from_numpy(y).bfloat16().float() * torch.from_numpy(gu[:, :, 1].astype(np.float32))).bfloat16().float()
    return gu.reshape(-1), h.numpy().transpose(0, 2, 1, 3).reshape(T, K)


# (tokens, channels, gu, copies): a norm's / o's input (a norm's q, k, v copies: the objects broadcast; gate, up of 256
# tokens: a column's own), Qwen3-0.6B's down (16-token items: 2 rounds of a group; 128 tokens: 4 groups a round),
# Qwen3.5-0.8B's ffn (28 groups, 256 tokens: 2 groups a round)
CASES = [(256, 1024, False, 1), (1024, 3072, False, 1), (1024, 1024, False, 3), (256, 1024, False, 2),
         (1024, 3072, True, 1), (128, 3072, True, 1), (256, 3584, True, 1)]


@pytest.mark.parametrize(("T", "K", "gu", "copies"), CASES,
                         ids=[f"{T}x{K}" + ("-gu" if gu else "") + (f"-c{c}" if c > 1 else "") for T, K, gu, c in CASES])
@pytest.mark.usefixtures("vtanh")
def test_prefill_rotate(T: int, K: int, gu: bool, copies: int) -> None:
    rng = np.random.default_rng(0)
    tfs = [_rotation(K, rng, 1 + c).tan_form(None if gu else rng.uniform(0.5, 2.0, K), unit=True) for c in range(copies)]
    if gu:
        xp, x = _gu_input(T, K, rng)
    else:
        xp = bfp16.to_panels(rng.standard_normal((T, K)).astype(np.float32) * np.exp(rng.standard_normal((T, 1))))
        x = bfp16.from_panels(xp, T, K)  # (the panels' values)
    op = PrefillRotOperator(T=T, K=K, gu=gu, copies=copies)
    ro = np.concatenate([tf.pair_objects(row_floats(gu)).reshape(-1) for tf in tfs])
    out = np.asarray(op.compile().run(xp, ro.view(bfp16.bfloat16))).reshape(copies, -1)
    for c, tf in enumerate(tfs):
        want = model(x, tf)
        got = bfp16.from_panels(out[c], T, K)
        exp = bfp16.from_panels(bfp16.to_panels(want), T, K)  # (the output panels' rounding of the model's x')
        bad = np.argwhere(got != exp)
        assert not len(bad), f"copy {c}: {len(bad)} of {got.size} values differ, first {bad[:4].tolist()}"
    exact = tfs[-1].apply(x.astype(np.float64))  # (fp64: storage order)

    def rel(v: np.ndarray) -> float:
        return float(np.sqrt(((v - exact) ** 2).mean() / (exact ** 2).mean()))

    floor = rel(bfp16.from_panels(bfp16.to_panels(exact), T, K))  # (the output panels' rounding of the exact x')
    assert rel(got) < 1.2 * floor, (rel(got), floor)  # (the rounds: the partners in bf16)
