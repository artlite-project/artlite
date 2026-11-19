"""A screened LM head's host side (``artlite.decode.screen``) on a random Q6_K matrix, no NPU: the exact rows
(``_native.screen_exact``) against the dequantized matrix, and from screened logits computed as the cores compute them
(the high plane's weights, ``formats.q6h``, rounded to bf16) the exact largest logits, the dense row, and
``models.sampling``'s choices (penalties, a mask) against those of the exact logits."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite import _native
from artlite.decode import screen as SC
from artlite.formats import q6h
from artlite.gguf_model import QTensor
from models.sampling import Penalties, Sampling, argmax, pack_mask, top_logits

from ..tensors import random_tensor

V, H = 8192, 1024


class _File:
    """What ``Screen`` takes of a GGUF file: a tensor by name."""

    def __init__(self, t: QTensor):
        self.t = t

    def tensor(self, name: str) -> QTensor:
        return self.t

    def populate(self, name: str) -> None:
        pass


@pytest.fixture(scope="module")
def head() -> tuple[SC.Screen, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """A screen of a random Q6_K head, its exact weights (V, H) and its high plane's (float64), its fields."""
    t = random_tensor("Q6_K", V, H, 7)
    f = t.fields()
    a = np.repeat((f["d"][..., None] * f["sc"].astype(np.float64)).reshape(V, -1), 16, 1)
    w = a * (f["q"].astype(np.float64) - 32)
    hf = q6h.fields(f)
    plane = np.repeat((hf["d"][..., None] * hf["sc"].astype(np.float64)).reshape(V, -1), 16, 1) * (hf["q"] - 1.5)
    norm = np.random.default_rng(1).uniform(0.5, 2.0, H).astype(np.float32)
    return SC.Screen(_File(t), "output.weight", norm, 1e-6, V), w, plane, f


def _row(head: tuple, x: np.ndarray) -> tuple[SC.ScreenedRow, np.ndarray]:
    """A screened row of the residual x as an overlay returns it, and its exact logits (float64 of the fp32 head input)."""
    sc, w, plane, _ = head
    g = sc.input(x)
    screened = (plane @ g.astype(np.float64)).astype(np.float32).astype(bfloat16)
    return SC.Screened(sc, screened[None], x[None], lambda: True)[0], w @ g.astype(np.float64)


def _inputs(head: tuple) -> list[np.ndarray]:
    """Residuals whose logits are peaked (a row's own direction: one, or several near ties) and flat (noise)."""
    sc, w = head[:2]
    rng = np.random.default_rng(2)
    out = [rng.standard_normal(H).astype(np.float32)]
    for rows, noise in (([17], 0.3), ([5, 4000, 8100], 0.5), ([123, 124], 1.0), (list(range(40, 70)), 0.4)):
        d = sum(w[r] / np.linalg.norm(w[r]) for r in rows) / sc.norm
        out.append((d * np.sqrt(H) + noise * rng.standard_normal(H)).astype(np.float32))
    return out


def test_exact(head: tuple) -> None:
    """The rows' exact logits read in place from the file's blocks: the dequantized matrix's, to fp32's rounding."""
    sc, w = head[:2]
    rng = np.random.default_rng(3)
    g = rng.standard_normal(H).astype(np.float32)
    ids = np.sort(rng.choice(V, 500, replace=False)).astype(np.int32)
    want = w[ids] @ g.astype(np.float64)
    assert np.abs(sc.exact(ids, g) - want).max() < 2e-6 * np.abs(want).max()


def test_top(head: tuple) -> None:
    """The k largest logits of a screened row are the exact ones (those within the window of the largest), in their
    order; the screened logits lie within their bounds."""
    sc = head[0]
    for x in _inputs(head):
        row, exact = _row(head, x)
        g = sc.input(x)
        est = row.screened.astype(np.float64) * sc.scale  # (the logits lie within their bounds)
        assert (np.abs(est - exact) <= sc.bound * np.linalg.norm(g) + 2.0 ** -8 * np.abs(exact)).all()
        for k in (1, 20, 64):
            v, i = top_logits(row, k)
            order = np.lexsort((np.arange(V), -exact))[:k]
            m = int((exact[order] >= exact[order[0]] - SC.WINDOW).sum())
            assert len(i) == k and np.array_equal(i[:m], order[:m]), k
            assert np.abs(v[:m] - exact[order[:m]]).max() < 1e-5 * np.abs(exact).max(), k
            # (past the window: screened values, below)
            assert (np.diff(v) <= 0).all() and len(set(i.tolist())) == k, k
            assert argmax(row) == order[0], k


def test_arrays(head: tuple) -> None:
    """A row as an array: every logit exact; ``dense``: the largest exact, the others screened and below them."""
    for x in _inputs(head):
        row, exact = _row(head, x)
        assert abs(float(row[77]) - exact[77]) < 1e-5 * np.abs(exact).max() and row._exact is None  # (one row's)
        assert np.abs(np.asarray(row) - exact).max() < 1e-5 * np.abs(exact).max()
        d = row.dense(20)
        order = np.argsort(-exact)[:20]
        m = int((exact[order] >= exact[order[0]] - SC.WINDOW).sum())
        assert np.array_equal(np.argsort(-d, kind="stable")[:m], order[:m])
        assert np.abs(d[order[:m]] - exact[order[:m]]).max() < 1e-5 * np.abs(exact).max()


def test_take(head: tuple) -> None:
    """Slots taken from a step's screened logits (``Screened.take``) are copies: their rows give the step's choices
    after the NPU's buffer holds another step's and the step's own rows are no longer valid."""
    sc, _, plane, _ = head
    xs = np.stack(_inputs(head)[:4])
    buf = np.stack([(plane @ sc.input(x).astype(np.float64)).astype(np.float32) for x in xs]).astype(bfloat16)
    alive = [True]
    step = SC.Screened(sc, buf, xs, lambda: alive[0])
    want = [top_logits(step[i], 20) for i in range(4)]
    mine = step.take(1, 3)
    buf[:], alive[0] = 0, False  # (the next dispatch)
    assert len(mine) == 2
    for i in range(2):
        v, ids = top_logits(mine[i], 20)
        assert np.array_equal(ids, want[1 + i][1]) and np.array_equal(v, want[1 + i][0])
    with pytest.raises(RuntimeError, match="before the overlay's next dispatch"):
        SC.Screened(sc, buf, xs, lambda: alive[0])[0].top(1)


def test_sampling(head: tuple) -> None:
    """``Penalties.top`` of a screened row (recent tokens penalized, a mask's tokens) against the exact logits'."""
    rng = np.random.default_rng(4)
    s = Sampling(top_k=20, repeat_penalty=1.3, presence_penalty=0.2, frequency_penalty=0.1)
    for x in _inputs(head):
        row, exact = _row(head, x)
        e32 = exact.astype(np.float32)
        recent = [int(t) for t in np.argsort(-exact)[[0, 2, 5]]] + [int(t) for t in rng.choice(V, 12)]
        for cap in (0.0, 30.0):
            p = Penalties(s, recent + recent[:2])
            v, i = p.top(row, 20, cap)
            wv, wi = p.top(e32, 20, cap)
            # (those the window holds: the logits within it of the largest one before the penalties)
            m = int(np.cumprod(e32[wi] >= e32.max() - SC.WINDOW).sum())
            assert np.array_equal(i[:m], wi[:m]) and np.allclose(v[:m], wv[:m], rtol=1e-5, atol=1e-5 * np.abs(wv).max())

        class Mask:
            def __init__(self, ids: np.ndarray):
                allowed = np.zeros(V, bool)
                allowed[ids] = True
                self.mask = pack_mask(allowed, V // 32)

            def bits(self) -> np.ndarray:
                return self.mask

            def accept(self, token: int) -> None:
                pass

        p = Penalties(s, recent, Mask(np.empty(0, np.int64)))  # (none allowed: the unmasked choice)
        assert len(p.top(row, 20)[1]) == 20
        # (a mask's candidates as exact as an unmasked choice's: a few tokens, their rows; many, the masked screen)
        order = np.argsort(-exact)
        for ids in (rng.choice(V, 10, replace=False), rng.choice(V, 300, replace=False),
                    np.concatenate([order[1:4], rng.choice(V, V // 2, replace=False)]), order[5:]):
            for s2 in (Sampling(top_k=20), s):
                p = Penalties(s2, recent, Mask(ids))
                v, i = p.top(row, 20)
                wv, wi = p.top(e32, 20)
                # (exact: a few tokens' rows all, else those the window holds below the largest allowed logit)
                m = int(np.cumprod(e32[wi] >= e32[ids].max() - SC.WINDOW).sum()) if len(ids) > 16 else len(wi)
                assert (m or s2.penalized) and np.array_equal(i[:m], wi[:m]), (len(ids), s2)
                assert not m or np.abs(v[:m] - wv[:m]).max() < 1e-5 * np.abs(wv).max()


def test_native_passes() -> None:
    """``_native.screen_lower`` / ``screen_above`` / ``screen_dense`` against numpy, at lengths off a vector's lanes."""
    rng = np.random.default_rng(5)
    for n in (1, 15, 16, 17, 1000, 4099):
        x = (10 * rng.standard_normal(n)).astype(np.float32).astype(bfloat16)
        b, sc = rng.uniform(0.1, 2.0, n).astype(np.float32), rng.uniform(1.0, 1.2, n).astype(np.float32)
        a = x.astype(np.float32) * sc
        gn = 1.7
        lo = _native.screen_lower(x.view(np.uint16), sc, b, gn)
        assert lo == pytest.approx(float((a.astype(np.float64) - b.astype(np.float64) * gn).max()), rel=1e-5)
        thr = float(np.median(a))
        ids = np.empty(8, np.int32)
        up = a.astype(np.float64) + b.astype(np.float64) * gn
        want = np.flatnonzero(up >= thr)
        got = _native.screen_above(x.view(np.uint16), sc, b, gn, thr, ids)
        edge = int((np.abs(up - thr) < 1e-4 * max(abs(thr), 1)).sum())  # (rows at the threshold to fp32's rounding)
        assert abs(got - len(want)) <= edge and (edge or np.array_equal(ids[:min(got, 8)], want[:8]))
        out = np.empty(n, np.float32)
        _native.screen_dense(x.view(np.uint16), sc, 3.0, out)
        assert np.array_equal(out, np.minimum(a, np.float32(3.0)))
        allowed = rng.random(n) < 0.3
        mask = pack_mask(allowed, (n + 31) // 32)
        lo = _native.screen_lower(x.view(np.uint16), sc, b, gn, mask)
        want_lo = float((a.astype(np.float64) - b.astype(np.float64) * gn)[allowed].max()) if allowed.any() else -np.inf
        assert lo == pytest.approx(want_lo, rel=1e-5)
        ids = np.empty(n, np.int32)
        got = _native.screen_above(x.view(np.uint16), sc, b, gn, -np.inf, ids, mask)
        assert np.array_equal(ids[:got], np.flatnonzero(allowed))


def test_stats(head: tuple) -> None:
    """``_native.screen_stats``: a row's share rho of a logit that its high plane lacks and the deviation left, from
    its codes and scales, against numpy on the tensor's fields."""
    sc, f = head[0], head[3]
    a2 = np.repeat((f["d"][..., None] * f["sc"].astype(np.float64)).reshape(V, -1) ** 2, 16, 1)
    wq, r = f["q"].astype(np.float64) - 32, (f["q"] & 15).astype(np.float64) - 8
    rho = (a2 * r * wq).sum(1) / (a2 * wq * wq).sum(1)
    nv = (a2 * (r - rho[:, None] * wq) ** 2).mean(1)
    assert np.allclose(sc.scale, 1 / (1 - rho), rtol=1e-5)
    assert np.allclose(sc.bound, SC.Z * np.sqrt(nv) / (1 - rho), rtol=1e-4)
