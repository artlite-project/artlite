"""The weight views the ViT-family encoders share (artlite.vision.vit: ``vec``, ``kinds_of``, ``blocks4``,
``plain_index``, ``rope_index``, ``folded_patch``) against the loops they replace, and the media sequences."""

import numpy as np

from artlite.vision import vit
from models.images import Grid
from models.qwen3vl.host import Grid as MropeGrid
from models.sequence import MEDIA, Sequence


def test_plain_and_rope_index_against_loops() -> None:
    for hd, hp, heads in ((64, 80, [3, 0, 7]), (72, 80, range(16)), (96, 160, [1, 2]), (32, 40, [5])):
        plain = [h * hd + d if d < hd else -1 for h in heads for d in range(hp)]
        assert (vit.plain_index(heads, hd, hp) == plain).all()
        rope = []
        for h in heads:
            for j in range(hp):  # (pair k: dims k and hd / 2 + k at channels k and hp / 2 + k)
                half = j % (hp // 2)
                rope.append(h * hd + (half + (0 if j < hp // 2 else hd // 2)) if half < hd // 2 else -1)
        assert (vit.rope_index(heads, hd, hp) == rope).all()
    assert (vit.Vit().head_rows() == vit.plain_index(range(vit.HEADS), vit.HD, vit.HP)).all()


def test_blocks4_kinds_vec() -> None:
    h, c = 5, 8
    v = np.arange(4 * h, dtype=np.float32)
    out = vit.blocks4(v, 0, h, c)
    assert out.shape == (4 * c,) and (out.reshape(4, c)[:, :h] == v.reshape(4, h)).all() and not out.reshape(4, c)[:, h:].any()
    m = np.arange(3 * 4 * h, dtype=np.float32).reshape(3, 4 * h)
    assert (vit.blocks4(m, 1, h, c).reshape(3, 4, c)[:, :, :h] == m.reshape(3, 4, h)).all()
    assert vit.kinds_of({"b.1.conv": 0}, 3, "b.{il}.conv") == ("attn", "conv", "attn")
    got = vit.vec(lambda n: np.array([[1.0, 2.0]]), "x", 4)
    assert got.tolist() == [1.0, 2.0, 0.0, 0.0] and got.dtype == np.float32


def test_folded_patch_is_the_normalization() -> None:
    """q = p - 128 through the folded weights = the weights on x = (p / 255 - 0.5) / 0.5 (the constant input 1)."""
    rng = np.random.default_rng(0)
    w0, w1 = rng.standard_normal((2, 6, 12))
    wt = vit.folded_patch([w0, w1], c=8, kp=40, one=30)  # (a video pair: kernel t on frame t's inputs)
    p = rng.integers(0, 256, (2, 12))
    q = np.zeros(40)
    q[:12], q[12:24], q[30] = p[0] - 128, p[1] - 128, 1
    x = (p / 255 - 0.5) / 0.5
    assert np.allclose(wt[:6] @ q, w0 @ x[0] + w1 @ x[1], atol=1e-4) and not wt[6:].any()
    one = vit.folded_patch([w0 + w1], c=8, kp=40, one=30)  # (a still image: the kernels summed)
    q = np.zeros(40)
    q[:12], q[30] = p[0] - 128, 1
    assert np.allclose(one[:6] @ q, (w0 + w1) @ x[0], atol=1e-4)


def test_media_sequences() -> None:
    rows = np.arange(12, dtype=np.float32).reshape(6, 2)
    s = Sequence.media(rows)
    assert (s.ids == MEDIA).all() and s.given.all() and s.next == 6 and (s.pos == np.arange(6)[:, None]).all()
    flat = Grid(2, 3).sequence(rows)
    assert (flat.pos == s.pos).all() and flat.next == 6
    m = MropeGrid(2, 3).sequence(rows)  # (M-RoPE: (0, r, c), the text after from max(gh, gw))
    assert m.pos.tolist() == [[0, r, c] for r in range(2) for c in range(3)] and m.next == 3
