"""The single attention layer's host side: the panel/row packers and the gate (relative L2 <= 1 % against the exact fp32
attention of the same Q / K / V), walked in the operator's own order (``moss_attn.attn_walk``): exact per-query
maximum, fp32 exp2, P rounded once to bfp16, the row sums on V's pad column, and the finalize's division.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.vision import layout as L

from .moss_attn import (
    A_OBJ,
    CORE_HEADS,
    HD,
    HP,
    K_OBJ,
    KB,
    QT,
    V_OBJ,
    MossAttnOperator,
    attn_walk,
    b_image,
    from_out,
    k_image,
    kv_image,
    pad_keys,
    pad_rows,
    q_image,
    v_image,
)


def attn_ref(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The exact fp32 attention of (S, heads, 64) q (already scaled by 1 / sqrt(hd), as HF scales it) / k / v."""
    s = q.shape[0]
    out = np.zeros((s, q.shape[1], HD), np.float64)
    for h in range(q.shape[1]):
        sc = q[:, h, :].astype(np.float64) @ k[:, h, :].astype(np.float64).T
        sc -= sc.max(1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(1, keepdims=True)
        out[:, h, :] = p @ v[:, h, :]
    return out.astype(np.float32)


def seeded(s: int, seed: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q is pre-scaled by 1 / sqrt(hd) (HF's own order: the scale is on the queries before the scores)."""
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((s, CORE_HEADS, HD)) * scale / np.sqrt(HD)).astype(np.float32)
    k = (rng.standard_normal((s, CORE_HEADS, HD)) * scale).astype(np.float32)
    v = rng.standard_normal((s, CORE_HEADS, HD)).astype(np.float32)
    return q, k, v


def test_b_image_is_the_panels_reordered() -> None:
    """A B object holds exactly the bfp16 panel bytes of its matrix, in the MAC's pop order."""
    x = np.random.default_rng(3).standard_normal((KB, HD)).astype(np.float32)
    assert np.array_equal(np.sort(b_image(x).view(np.uint8)), np.sort(L.panels(x).view(np.uint8)))


@pytest.mark.parametrize("s", [QT, 3 * QT + 32])
def test_images(s: int) -> None:
    """Q: one A object a tile; K and V^T: one object a band (per head); KV: a band's two halves, in the core's order
    [head][tile][pass][band]. The V^T object's pad column 64 is 1.0 for every key (the row sums come from the MMUL)."""
    q, k, v = seeded(s, 5)
    qi, ki, vi = q_image(q), k_image(k), v_image(v)
    tiles, bands = -(-s // QT), -(-k.shape[0] // KB)
    assert qi.size == CORE_HEADS * tiles * A_OBJ
    assert ki.size == CORE_HEADS * bands * K_OBJ
    assert vi.size == CORE_HEADS * bands * V_OBJ
    kv = kv_image(k, v, tiles)
    assert kv.size == CORE_HEADS * tiles * 2 * bands * (K_OBJ + V_OBJ)
    # (a KV object: K's panels then the V^T's, and the V^T's row 64 the ones)
    obj = kv[:K_OBJ + V_OBJ]
    assert np.array_equal(obj[:K_OBJ], ki[:K_OBJ])
    vp0 = pad_keys(np.ascontiguousarray(v[:, 0, :]), s)[:KB]
    vt0 = np.zeros((HP, KB), np.float32)
    vt0[:HD] = vp0.T
    vt0[HD] = 1.0
    assert np.array_equal(np.sort(obj[K_OBJ:].view(np.uint8)), np.sort(L.panels(vt0).view(np.uint8)))
    assert np.allclose(L.from_panels(L.panels(vt0), HP, KB)[HD], 1.0)
    canon = np.concatenate([L.panels(np.ascontiguousarray(pad_rows(q[:, h, :], s)[t * QT:(t + 1) * QT]))
                            for h in range(CORE_HEADS) for t in range(tiles)])
    assert np.array_equal(np.sort(qi.view(np.uint8)), np.sort(canon.view(np.uint8)))


def test_q_image_rows() -> None:
    """A Q tile's object holds the head's rows of that tile (a permutation of the canonical panels, padded rows zero)."""
    s = 2 * QT + 40
    q, _k, _v = seeded(s, 7)
    qi = q_image(q)
    canon = L.panels(np.ascontiguousarray(q[:QT, 0, :]))
    assert np.array_equal(np.sort(qi[:QT * HD * 9 // 8].view(np.uint8)), np.sort(canon.view(np.uint8)))
    assert np.allclose(L.from_panels(canon, QT, HD), q[:QT, 0, :], atol=1e-2, rtol=2e-2)


@pytest.mark.parametrize("s", [2 * KB, 5 * KB + 48])
def test_gate(s: int) -> None:
    """The gate: the operator's arithmetic within 1 % of the exact fp32 attention of the same operands."""
    q, k, v = seeded(s, 11, scale=0.5)
    ref = attn_ref(q, k, v)
    got = attn_walk(q, k, v, s=s)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel <= 0.01, f"{rel:.5f}"


@pytest.mark.parametrize("s", [2 * KB, 1500])
def test_render(s: int) -> None:
    """The design renders (4 cores of 4 heads), its argument sizes are the images the packers build."""
    op = MossAttnOperator(S=s)
    shapes = [a.shape for a in op.get_arg_spec()]
    tiles, bands = op.tiles, op.bands
    kv = K_OBJ + V_OBJ
    assert shapes == [(CORE_HEADS * tiles * A_OBJ // 2,),
                      (CORE_HEADS * tiles * 2 * bands * kv // 2,),
                      (CORE_HEADS * tiles * QT * HD,)]
    text = op.mlir_text()
    assert text.count("aie.core") == CORE_HEADS
    src = op.get_kernel_src()
    for name in ("_init", "_max", "_exp", "_fin", "_qk", "_pv"):
        assert name in src, name
    assert ("_expz" in src) == bool(op.nv), "the tail band's masked P kernel only where S is not a whole band"



def test_from_out() -> None:
    """OUT image bytes -> (S, heads, 64), the padded queries dropped."""
    s = 3 * QT + 27
    _q, _k, _v = seeded(s, 13)
    tiles = -(-s // QT)
    y = np.concatenate([np.zeros(CORE_HEADS * tiles * QT * HD, np.float32).reshape(-1).astype(bfloat16)])
    assert from_out(y, CORE_HEADS, s).shape == (s, CORE_HEADS, HD)
    assert np.allclose(from_out(y, CORE_HEADS, s), 0.0)
