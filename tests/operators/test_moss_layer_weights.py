"""The layer's weight images: the overlay's B layout, the folded bias column, the LN vectors, the position rows and
the KV slots — host side only (the layer program reads these)."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.vision import layout as L

from .moss_layer_weights import (
    b_decode,
    b_image,
    bf16_decode,
    fold_bias,
    kv_slots,
    layer_images,
    ln_vectors,
    npad_of,
    pos_rows,
)

WIDTH, FFN = 1280, 4160
HEADS, HD, HP = 16, 64, 80


def seeded_layer() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {
        "q_w": (rng.standard_normal((2048, 1024)) * 0.02).astype(np.float32),
        "q_b": (rng.standard_normal(2048) * 0.01).astype(np.float32),
        "k_w": (rng.standard_normal((1024, 1024)) * 0.02).astype(np.float32),
        "v_w": (rng.standard_normal((1024, 1024)) * 0.02).astype(np.float32),
        "v_b": (rng.standard_normal(1024) * 0.01).astype(np.float32),
        "o_w": (rng.standard_normal((1024, 2048)) * 0.02).astype(np.float32),
        "o_b": (rng.standard_normal(1024) * 0.01).astype(np.float32),
        "fc1_w": (rng.standard_normal((4096, 1024)) * 0.02).astype(np.float32),
        "fc1_b": (rng.standard_normal(4096) * 0.01).astype(np.float32),
        "fc2_w": (rng.standard_normal((1024, 4096)) * 0.02).astype(np.float32),
        "fc2_b": (rng.standard_normal(1024) * 0.01).astype(np.float32),
        "ln_a_w": rng.standard_normal(1024).astype(np.float32),
        "ln_a_b": rng.standard_normal(1024).astype(np.float32),
        "ln_f_w": rng.standard_normal(1024).astype(np.float32),
        "ln_f_b": rng.standard_normal(1024).astype(np.float32),
        "pos": (rng.standard_normal((1500, 1024)) * 0.5).astype(np.float32),
    }


MATRICES = {"q": (2240, 1280, "q_w", "q_b"), "k": (1280, 1280, "k_w", None), "v": (1280, 1280, "v_w", "v_b"),
            "o": (1280, 2240, "o_w", "o_b"), "fc1": (4160, 1280, "fc1_w", "fc1_b"),
            "fc2": (1280, 4160, "fc2_w", "fc2_b")}


@pytest.mark.parametrize("key", list(MATRICES))
def test_b_image_round_trip(key: str) -> None:
    """A matrix's B image decodes back to its bfp16 values: the MAC's pop order is a permutation, the bias column
    sits at K_real, and the pad rows / columns are zero."""
    sd = seeded_layer()
    n, kk, wname, bname = MATRICES[key]
    ref = fold_bias(sd[wname], sd[bname] if bname else None, kpad=kk, npad=n)
    img = b_image(ref)
    assert img.size == n * kk * 9 // 8
    got = b_decode(img, n, kk)
    rel = np.linalg.norm(got - np.asarray(ref.astype(bfloat16), np.float32)) / np.linalg.norm(ref)
    assert rel <= 0.01, f"{rel:.4f}"
    # (the bias column and the zero pad past the real channels: K_real = the source's, rows past N are zero)
    real = sd[wname].shape[1]
    if bname:
        assert np.allclose(got[:sd[wname].shape[0], real], sd[bname], atol=1e-3)
    assert np.allclose(got[sd[wname].shape[0]:], 0.0) and np.allclose(got[:, real + 1:], 0.0)


def test_images() -> None:
    """The layer's image set: sizes, LN vectors at the residual width, the position rows in the C-tile layout."""
    sd = seeded_layer()
    imgs = layer_images(sd)
    assert set(imgs) == {"q", "k", "v", "o", "fc1", "fc2", "ln_a_g", "ln_a_b", "ln_f_g", "ln_f_b", "pos"}
    for key, (n, kk, _w, _b) in MATRICES.items():
        assert imgs[key].size == n * kk * 9 // 8
    for tag in ("ln_a", "ln_f"):
        g, b = imgs[f"{tag}_g"], imgs[f"{tag}_b"]
        assert g.dtype == np.float32 and g.size == WIDTH
        assert np.allclose(g[:1024], sd[f"{tag}_w"]) and np.allclose(g[1024:], 0.0)
        assert np.allclose(b[:1024], sd[f"{tag}_b"]) and np.allclose(b[1024:], 0.0)
    pos = bf16_decode(imgs["pos"], 1504, WIDTH)
    assert np.allclose(pos[:1500, :1024], sd["pos"], atol=1e-2, rtol=2e-2)
    assert np.allclose(pos[:1500, 1024:], 0.0)


def test_kv_slots() -> None:
    """The attention's K / V^T image of a round's 8 heads (d 80, groups of 64 keys) holds the values and is
    sized by the slot layout."""
    rng = np.random.default_rng(5)
    s = 1500
    k = rng.standard_normal((s, 8, HD)).astype(np.float32)
    v = rng.standard_normal((s, 8, HD)).astype(np.float32)
    img = kv_slots(k, v)
    assert img.dtype == np.uint8 and img.size % 9 == 0
    groups = -(-s // 64)                        # (keys padded to whole groups of 64)
    # [image][head][key group]: K (64 keys x 80 dims) then V^T (80 dims x 64 keys), each bfp16 blocks
    assert img.size == 2 * 8 * groups * 2 * 64 * 80 * 9 // 8
    assert img.size % L.PANEL == 0


def test_npad() -> None:
    """The output channels pad to the overlay's pass geometry (multiples of 320)."""
    assert [npad_of(n) for n in (1024, 2048, 4096)] == [1280, 2240, 4160]


def test_ln_vectors_pad() -> None:
    """LN's gamma / beta at the residual width: the real channels then zeros (the pads)."""
    g, b = ln_vectors(np.arange(1024, dtype=np.float32), None, WIDTH)
    assert np.array_equal(g[:1024], np.arange(1024, dtype=np.float32)) and np.allclose(g[1024:], 0.0)
    assert np.allclose(b, 0.0)


def test_pos_rows_zero_past() -> None:
    """The position table's C-tile image: its own rows, zero past them (1500 rows padded to the 16-row grid)."""
    pos = np.ones((1500, 1024), np.float32)
    img = pos_rows(pos, 1500, WIDTH)
    out = bf16_decode(img, 1504, WIDTH)
    assert np.allclose(out[:1500, :1024], 1.0) and np.allclose(out[1500:], 0.0)
