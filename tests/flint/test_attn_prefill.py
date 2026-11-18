"""The prefill attention on Flint's shapes (prefill/attn/ with ``Shape(per=2, sink=True)``): two columns per KV head on
one broadcast stream (each column four query heads, the KV head's K and V from both columns' halves; no gate), q / k
RMS-normed with weights per dim, learned sinks (the SK argument), full layers without positional encoding (every pair's
cos 1, sin 0) and sliding-window layers of 513 positions (keys p - 512 .. p, the cache after a guard of 8 groups), on
random projections of a chunk at p0 after a cache the host filled: the Q units and the K / V records against their fp64
model (single panels: within their own bfp16 rounding), the output panels against the exact attention with sinks of
the cores' own operands. Flint's planes (K hi + lo, single Q and P: ``FlintPrefill``'s; its K panels interleaved per d-block
by the memtile, ``AttnOperator.kilv``) and single K."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.decode import kv as KV
from artlite.operators import FusedOperator, bfp16
from artlite.operators.design import COLS
from artlite.prefill import attn as PA
from artlite.prefill.gemm import Layout

from ..numerics import rel

A_BYTES = 9216
HD, QH = 128, 4  # (per column)
KVH = COLS // 2
WIN = 513
LAYOUT = Layout(640, {"q": 0, "k": 512, "v": 576})  # (the qkv GEMMs' layout of Flint's image, ``prefill.gemm``)
# (chunk, its position, sliding window (0: a full layer), planes K, Q, P (Flint's: K hi + lo; single K: the app's
# option))
CASES = [(512, 128, 0, (2, 1, 1)), (512, 448, WIN, (2, 1, 1)), (2048, 0, WIN, (2, 1, 1)), (64, 576, WIN, (2, 1, 1)),
         (512, 448, WIN, (1, 1, 1))]


def _q_values(units: np.ndarray, hd: int, qp: int) -> np.ndarray:
    """Q units (..., 72 hd / 8 qp) uint8 -> (..., 8 queries, hd) the values they hold (hi + lo: their sum)."""
    lead = units.shape[:-1]
    u = units.reshape(*lead, qp, 72 * hd // 8)
    v = sum(bfp16.block_values(u[..., i, :].reshape(*lead, hd // 8, 8, 9)) for i in range(qp))  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, hd)


@pytest.mark.parametrize(("T", "p0", "win", "planes"), CASES,
                         ids=[f"T{a}-p{b}-{'swa' if c else 'full'}-k{d[0]}q{d[1]}p{d[2]}" for a, b, c, d in CASES])
def test_flint_attn(T: int, p0: int, win: int, planes: tuple[int, int, int]) -> None:
    kp, qp, pp = planes
    shape = PA.Shape(HD, QH, kp, A_BYTES, qp, pp, per=2, sink=True)
    max_pos = max(1024, p0 + T)
    guard = PA.mode.swa_groups(win, shape.pg) if win else 0  # (groups before a sliding-window cache's group 0)
    n_groups = guard + max_pos // shape.pg
    kvb, kv0 = n_groups * shape.group, guard * shape.group
    prep = PA.PrepOperator(T=T, shape=shape, layout=LAYOUT, kv_bytes=kvb, cs_positions=max_pos, norm=True, eps=1e-6,
                           kv0=kv0)
    attn = PA.AttnOperator(T=T, shape=shape, kv_bytes=kvb, swa=win, kv0=kv0)
    fused = FusedOperator(name=f"fla_test_{'s' if win else 'f'}{kp}{qp}{pp}", input_args=["G", "CS", "NW", "SK"],
                          output_args=["A"], runlist=[(prep, "G", "CS", "QS", "KV", "NW"),
                                                      (attn, "QS", "KV", "A", "SK")]).compile()
    gen = torch.Generator().manual_seed(p0 + T + win)
    inv = 1e4 ** (-np.arange(0, HD, 2) / HD) if win else np.zeros(HD // 2)  # (a full layer: no RoPE)
    cs = PA.cs_table(inv, max_pos)
    fused.get_buffer("CS").view(np.uint8)[:] = cs.view(np.uint8).reshape(-1)
    wq = (torch.rand(HD, generator=gen) * 2 + 0.2).double()
    wk = (torch.rand(HD, generator=gen) * 2 + 0.2).double()
    nw = np.zeros(PA.PrepOperator.nw_bytes(HD) // 4, np.float32)
    nw[:2 * HD] = torch.cat([wq, wk]).numpy()
    fused.get_buffer("NW").view(np.uint8)[:] = nw.view(np.uint8)
    sk = (torch.rand(COLS * QH, generator=gen) * 12 - 4).numpy()  # (Flint's: -9 .. 12)
    fused.get_buffer("SK").view(np.uint8)[:] = attn.sinks(sk)
    kv = fused.get_buffer("KV").view(np.uint8)[:KVH * kvb].reshape(KVH, n_groups, shape.objects, A_BYTES)
    kv[:] = 0
    for h in range(KVH):  # (the cache before the chunk)
        if p0:
            k_h = torch.randn(p0, HD, generator=gen).numpy() * 2
            v_h = torch.randn(p0, HD, generator=gen).numpy()
            kv[h, guard:guard + p0 // shape.pg] = KV.kv_groups(k_h, v_h, A_BYTES, kp)
    q = torch.randn(T, COLS * QH, HD, generator=gen) * 2  # (head 4 c + i: column c's)
    k = torch.randn(T, KVH, HD, generator=gen) * 2
    v = torch.randn(T, KVH, HD, generator=gen)
    G = np.zeros((T, LAYOUT.N), np.float32)
    for c in range(COLS):
        h, j = divmod(c, 2)
        o = c * LAYOUT.pitch
        G[:, o + np.arange(QH * HD)] = q[:, QH * c:QH * c + QH].reshape(T, -1)
        G[:, o + LAYOUT.off["k"] + np.arange(64)] = k[:, h, 64 * j:64 * j + 64]
        G[:, o + LAYOUT.off["v"] + np.arange(64)] = v[:, h, 64 * j:64 * j + 64]
    fused.get_buffer("G")[:] = G.reshape(-1).astype(bfloat16)
    fused.sync_inputs()
    fused.set_params(prep.params(p0) | attn.params(p0))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()

    def rope(x: torch.Tensor, w: torch.Tensor) -> np.ndarray:  # x (T, ..., hd) bf16 values normed and rotated (fp64)
        y = x.bfloat16().double().numpy().copy()
        y = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-6) * w.numpy()
        c_ = cs[p0:p0 + T, :HD // 2].astype(np.float64).reshape(T, *[1] * (y.ndim - 2), HD // 2)
        s_ = cs[p0:p0 + T, HD // 2:].astype(np.float64).reshape(c_.shape)
        a, b = y[..., :HD // 2].copy(), y[..., HD // 2:].copy()
        y[..., :HD // 2], y[..., HD // 2:] = a * c_ - b * s_, a * s_ + b * c_
        return y

    qs = fused.get_buffer("QS").view(np.uint8)[:prep.n_qs * 2].reshape(COLS, T // 8, QH, shape.qunit)
    qv = _q_values(qs, HD, shape.qp).transpose(0, 2, 1, 3, 4).reshape(COLS * QH, T, HD)  # [head][query][hd]
    def floor(x: np.ndarray, planes: int) -> float:  # (the error of x's single bfp16 panels; hi + lo: 2e-4)
        return 2e-4 if planes == 2 else 1.05 * rel(bfp16.values(x.reshape(-1, x.shape[-1])).reshape(x.shape), x)

    want_q = rope(q, wq).transpose(1, 0, 2) * shape.scale
    assert rel(qv, want_q) < floor(want_q, qp)
    n = (p0 + T) // shape.pg
    Ks, Vs = {}, {}
    for h in range(KVH):
        Ks[h], Vs[h] = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[h, guard:guard + n], HD))
        assert rel(Ks[h][p0:], rope(k, wk)[:, h]) < floor(rope(k, wk)[:, h], kp), h
        assert rel(Vs[h][p0:], v[:, h].bfloat16().double().numpy()) < 1e-2, h
    A = bfp16.from_panels(fused.get_buffer("A")[:attn.n_a], T, COLS * QH * HD).reshape(T, COLS * QH, HD)
    want = np.zeros((T, COLS * QH, HD))
    kpos, qpos = np.arange(p0 + T)[None], np.arange(p0, p0 + T)[:, None]
    masked = (kpos > qpos) | ((qpos - kpos >= win) if win else False)
    for hq in range(COLS * QH):
        h = hq // 8  # (its KV head: 8 query heads each)
        sc = qv[hq].astype(np.float64) @ Ks[h].T
        sc[masked] = -np.inf
        s2 = float(np.float32(sk[hq] / np.log(2)))  # (the sink in log2 units, as the SK argument holds it)
        mx = np.maximum(sc.max(-1, keepdims=True), s2)
        pr = np.exp2(sc - mx)
        want[:, hq] = (pr @ Vs[h]) / (pr.sum(-1, keepdims=True) + np.exp2(s2 - mx))
    err = rel(A, want)
    eq = rel(bfp16.values(want.reshape(T, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"attention {err:.2e} (the exact output rounded to bfp16: {eq:.2e}); QB, CR {shape.blocks}")
    # (hi + lo P: the exact output's own rounding, 1.00x; single P adds its rounding: 1.34-1.44x after a history, 1.65-1.72x
    # at p0 0, whose first rows see few keys)
    assert err < (1.1 if pp == 2 else 1.9) * eq
