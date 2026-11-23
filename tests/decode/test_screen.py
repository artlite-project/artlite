"""A screened LM head's host side (``artlite.decode.screen``) on random Q6_K and Q8_0 matrices and random F16 / BF16
ones, no NPU: the exact rows (``_native.screen_exact``, ``screen_exact_q8``, ``screen_exact_half``) against the
dequantized matrix, and from screened logits computed as the cores compute them (the high plane's weights,
``formats.q6h``, ``formats.q8h``; a half-precision head's Q4_0 plane, quantized at load: ``screen.plane_rows``; rounded
to bf16) the exact largest logits, the dense row, and ``models.sampling``'s choices (penalties, a mask) against those of
the exact logits; Inkling's Q8_0 head on llama.cpp's head inputs of an essay."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite import _native
from artlite.decode import screen as SC
from artlite.formats import q6h, q8h
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


def _half_head(qtype: str) -> tuple[SC.Screen, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """A screen of a random half-precision head (``qtype`` F16, BF16), its exact weights (V, H) and its Q4_0 plane's
    (float64), no fields."""
    rng = np.random.default_rng(7)
    dt = np.float16 if qtype == "F16" else bfloat16
    w = (rng.standard_normal((V, H)) * rng.uniform(0.01, 0.05, (V, 1))).astype(dt)
    t = QTensor(name="output.weight", qtype=qtype, rows=V, cols=H, raw=w.view(np.uint16))
    q = SC.plane_rows(t).reshape(V, H // 32, 18)
    d = q[..., :2].copy().view(np.float16)[..., 0].astype(np.float64)
    codes = np.concatenate([q[..., 2:] & 15, q[..., 2:] >> 4], -1).astype(np.float64) - 8
    plane = (codes * d[..., None]).reshape(V, H)
    norm = np.random.default_rng(1).uniform(0.5, 2.0, H).astype(np.float32)
    return SC.Screen(_File(t), "output.weight", norm, 1e-6, V), w.astype(np.float64), plane, {}


@pytest.fixture(scope="module", params=["Q6_K", "Q8_0", "F16", "BF16"])
def head(request: pytest.FixtureRequest) -> tuple[SC.Screen, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """A screen of a random Q6_K or Q8_0 head, its exact weights (V, H) and its high plane's (float64, ``formats.q6h``,
    ``formats.q8h``), its fields; of a random F16 / BF16 head, with its Q4_0 plane (``_half_head``)."""
    if request.param in SC.HALF:
        return _half_head(request.param)
    if request.param == "Q8_0":
        t = random_tensor("Q8_0", V, H, 7)
        f = t.fields()
        hf = q8h.fields(f)
        w = np.repeat(f["d"].astype(np.float64), 32, 1) * f["q"]
        plane = np.repeat(hf["d"].astype(np.float64), 32, 1) * (hf["q"] - 7.5)
        norm = np.random.default_rng(1).uniform(0.5, 2.0, H).astype(np.float32)
        return SC.Screen(_File(t), "output.weight", norm, 1e-6, V), w, plane, f
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
    its codes and scales, against numpy on the tensor's fields; a half-precision head's (``screen_stats_plane``) from its
    weights and its Q4_0 plane's."""
    sc, f = head[0], head[3]
    if not f:
        w, r = head[1], head[1] - head[2]
        rho = (r * w).sum(1) / (w * w).sum(1)
        nv = ((r - rho[:, None] * w) ** 2).reshape(V, -1, 32).mean(-1).max(1)  # (the largest block's)
        assert np.allclose(sc.scale, 1 / (1 - rho), rtol=1e-5)
        assert np.allclose(sc.bound, SC.Z * np.sqrt(nv) / (1 - rho), rtol=1e-3)
        return
    if "sc" not in f:  # (Q8_0: a = d, the largest block's deviation)
        a2 = np.repeat(f["d"].astype(np.float64) ** 2, 32, 1)
        wq = f["q"].astype(np.float64)
        r = (f["q"] & 15) - 7.5
        rho = (a2 * r * wq).sum(1) / (a2 * wq * wq).sum(1)
        nv = (a2 * (r - rho[:, None] * wq) ** 2).reshape(V, -1, 32).mean(-1).max(1)
        assert np.allclose(sc.scale, 1 / (1 - rho), rtol=1e-5)
        assert np.allclose(sc.bound, SC.Z * np.sqrt(nv) / (1 - rho), rtol=1e-4)
        return
    a2 = np.repeat((f["d"][..., None] * f["sc"].astype(np.float64)).reshape(V, -1) ** 2, 16, 1)
    wq, r = f["q"].astype(np.float64) - 32, (f["q"] & 15).astype(np.float64) - 8
    rho = (a2 * r * wq).sum(1) / (a2 * wq * wq).sum(1)
    nv = (a2 * (r - rho[:, None] * wq) ** 2).mean(1)
    assert np.allclose(sc.scale, 1 / (1 - rho), rtol=1e-5)
    assert np.allclose(sc.bound, SC.Z * np.sqrt(nv) / (1 - rho), rtol=1e-4)


def test_screened_counts_file_formats() -> None:
    """A head is screened in a file of up to FILE_FORMATS tile formats, its own type counting (not its plane's): a Q4_K_M
    file's Q4_K, Q5_K, Q6_K (Qwen3.6-35B-A3B) keeps its Q6_K head screened; a fourth format does not."""
    assert SC.screened("Q6_K", {"Q4_K", "Q5_K", "Q6_K"})
    assert not SC.screened("Q6_K", {"Q4_K", "Q5_K", "Q8_0"})
    assert SC.screened("F16", {"P4_128"}) and not SC.screened("Q8_0", {"Q4_K"})
    assert SC.screened("Q8_0", {"NVFP4", "BFP16"}, opt_in=True)  # (a Q8_0 head: where the model asks, Inkling's)


def test_inkling_head() -> None:
    """Inkling's Q8_0 head (its vocabulary's 200058 rows; the Q8_0H plane 53% of the bytes) on llama.cpp's head inputs
    (``result_norm``) of 512 essay tokens: the estimates of the 1000 largest logits of every token within their bounds (over 5120 tokens:
    4.0 of the 6 sigma), each token's 20 largest logits the exact ones, the argmax llama.cpp's where its own (fp16)
    logits' best two are apart."""
    import torch

    from artlite.gguf_model import GGUFFile
    from artlite.model_config import inkling_gguf

    from ..common import essays
    from ..goldens.text import golden
    from ..llama_cpp import llama_tokenize
    from ..model_files import require

    path = inkling_gguf()
    require(path)
    g = GGUFFile(path)
    t = g.tensor("output.weight")
    G = golden(path, llama_tokenize(path, essays())[:512], layers=("g",), names="result_norm", logits="f16")
    xs, ll = np.array(G["result_norm"], np.float32), np.asarray(G.logits, np.float32)
    V = ll.shape[1]  # (the vocabulary's rows: 200058 of the head's 201024)
    sc = SC.Screen(g, "output.weight", np.ones(t.cols, np.float32), 1e-6, V)
    f = t.fields()
    hf = q8h.fields(f)
    W = torch.from_numpy(np.repeat(f["d"], 32, 1) * f["q"].astype(np.float32))[:V]
    P = torch.from_numpy(np.repeat(hf["d"], 32, 1) * (hf["q"].astype(np.float32) - 7.5))[:V]
    for c0 in range(0, len(xs), 128):
        x = torch.from_numpy(xs[c0:c0 + 128])
        exact, screened = (x @ W.T).numpy(), (x @ P.T).bfloat16().view(torch.int16).numpy().view(bfloat16)
        for j in range(len(x)):
            gv, e = xs[c0 + j], exact[j]
            z = np.abs(screened[j].astype(np.float32) * sc.scale - e) / (sc.bound * np.float32(np.linalg.norm(gv)))
            largest = np.argpartition(e, -1000)[-1000:]
            assert z[largest].max() < 1, (c0 + j, float(z[largest].max()) * SC.Z)
            v, i = sc.top(screened[j], gv, 20)
            order = np.lexsort((np.arange(V), -e))[:20]
            m = int((e[order] >= e[order[0]] - SC.WINDOW).sum())
            assert np.array_equal(i[:m], order[:m]) and np.allclose(v[:m], e[order[:m]], rtol=1e-5), c0 + j
            best = np.sort(ll[c0 + j])[-2:]
            assert best[1] - best[0] < 0.05 or i[0] == int(ll[c0 + j].argmax()), c0 + j
