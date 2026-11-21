"""``prefill.rotate.PrefillRotOperator`` on the NPU against its host model bit for bit (Qwen3-0.6B-PARO's rotations, or
random ones without the checkpoint): the panels of x -> u = x w' (exact), the rounds on Givens pairs (the partner's bf16
value times T on the fp32 MAC), x' as hi + lo into bfp16 panels; and the model against the exact transform (fp64)."""

import numpy as np
import pytest

from artlite.operators import bfp16
from artlite.paro import Rotation
from artlite.prefill.rotate import PrefillRotOperator

from ..operators import kmath_sim as ks


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


@pytest.mark.parametrize(("T", "K"), [(256, 1024), (1024, 3072)], ids=["256x1024", "1024x3072"])
def test_prefill_rotate(T: int, K: int) -> None:
    rng = np.random.default_rng(0)
    try:
        from artlite.hf_model import HfModel
        from artlite.model_config import checkpoint

        name = "blk.1.ffn_down.weight" if K == 3072 else "blk.1.attn_q.weight"
        rot = HfModel(checkpoint("qwen3-0.6B-paro")).tensor(name).rot()
    except (OSError, FileNotFoundError):
        pairs = np.stack([np.concatenate([rng.permutation(128) for _ in range(K // 128)]) for _ in range(8)])
        rot = Rotation(rng.uniform(-3, 3, (8, K // 2)), pairs, rng.uniform(-2, 2, K), 128)
    tf = rot.tan_form(rng.uniform(0.5, 2.0, K), unit=True)
    xp = bfp16.to_panels(rng.standard_normal((T, K)).astype(np.float32) * np.exp(rng.standard_normal((T, 1))))
    x = bfp16.from_panels(xp, T, K)  # (the panels' values)
    op = PrefillRotOperator(T=T, K=K)
    out = op.compile().run(xp, tf.pair_objects().reshape(-1).view(bfp16.bfloat16))
    want = model(x, tf)
    got = bfp16.from_panels(np.asarray(out), T, K)
    exp = bfp16.from_panels(bfp16.to_panels(want), T, K)  # (the output panels' rounding of the model's x')
    bad = np.argwhere(got != exp)
    assert not len(bad), f"{len(bad)} of {got.size} values differ, first {bad[:4].tolist()}"
    exact = tf.apply(x.astype(np.float64))  # (fp64: storage order)

    def rel(v: np.ndarray) -> float:
        return float(np.sqrt(((v - exact) ** 2).mean() / (exact ** 2).mean()))

    floor = rel(bfp16.from_panels(bfp16.to_panels(exact), T, K))  # (the output panels' rounding of the exact x')
    assert rel(got) < 1.2 * floor, (rel(got), floor)  # (the rounds: the partners in bf16)
