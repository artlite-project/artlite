"""The prefill attention on LFM3's shapes (prefill/attn.py with ``Shape(per=2, gate=True)``): two columns per KV head
(each column its two query heads, the KV head's K and V from both columns' halves), q / k RMS-normed with weights per
dim, a sigmoid output gate per head, full layers (32 rotated pairs: the other pairs' cos 1, sin 0) and sliding-window
layers (a window of 256 positions, the cache after a guard of its groups), on random projections of a chunk at p0 after a
cache the host filled: the K / V records against their fp64 model, the output panels against the exact attention of the
cores' own operands (their Q units, the cache they read) times the gates."""

import numpy as np
import pytest
import torch

from artlite.decode import kv as KV
from artlite.operators import FusedOperator, bfp16
from artlite.operators.design import COLS
from artlite.prefill import attn as PA
from artlite.prefill.gemm import Layout

A_BYTES = 9280
HD, QH = 128, 2  # (per column)
KVH = COLS // 2
PP = 2  # (P panels per row block and record: hi + lo)
LAYOUT = Layout(448, {"q": 0, "k": 256, "v": 320, "g": 384})  # (the qkv GEMMs' layout of LFM3's image, ``prefill.gemm``)
# (K panels per record, chunk, its position, sliding window (0: a full layer), rotated pairs)
CASES = [(1, 256, 0, 0, 32), (1, 512, 128, 0, 32), (2, 256, 64, 0, 32), (1, 256, 0, 256, 64), (1, 512, 192, 256, 64),
         (1, 1024, 512, 256, 64), (2, 512, 64, 256, 64), (1, 64, 0, 0, 32), (1, 64, 320, 256, 64)]


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _q_values(units: np.ndarray, hd: int) -> np.ndarray:
    """Q units (..., 72 hd / 8) uint8 -> (..., 8 queries, hd) the values they hold."""
    lead = units.shape[:-1]
    v = bfp16.block_values(units.reshape(*lead, hd // 8, 8, 9))  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, hd)


@pytest.mark.parametrize(("kp", "T", "p0", "win", "rot"), CASES,
                         ids=[f"kp{a}-T{b}-p{c}-{'swa' if d else 'full'}" for a, b, c, d, _ in CASES])
def test_lfm_attn(kp: int, T: int, p0: int, win: int, rot: int) -> None:
    shape = PA.Shape(HD, QH, kp, A_BYTES, 1, PP, per=2, gate=True)
    max_pos = max(1024, p0 + T)
    guard = win // shape.pg  # (groups before a sliding-window cache's group 0)
    n_groups = guard + max_pos // shape.pg
    kvb, kv0 = n_groups * shape.group, guard * shape.group
    prep = PA.PrepOperator(T=T, shape=shape, layout=LAYOUT, kv_bytes=kvb, cs_positions=max_pos, norm=True, eps=1e-5,
                           kv0=kv0)
    attn = PA.AttnOperator(T=T, shape=shape, kv_bytes=kvb, swa=win, kv0=kv0)
    fused = FusedOperator(name=f"lfa_test{kp}_{'s' if win else 'f'}", input_args=["G", "CS", "NW"], output_args=["A"],
                          runlist=[(prep, "G", "CS", "QS", "KV", "NW"), (attn, "QS", "KV", "A")]).compile()
    gen = torch.Generator().manual_seed(p0 + T + kp + win)
    base = 1e4 if win else 1e6
    inv = base ** (-np.arange(0, HD, 2) / HD)
    inv[rot:] = 0  # (pairs past rot: cos 1, sin 0)
    cs = PA.cs_table(inv, max_pos)
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(cs.view(np.uint8).reshape(-1))
    wq = (torch.rand(HD, generator=gen) * 2 + 0.2).double()
    wk = (torch.rand(HD, generator=gen) * 2 + 0.2).double()
    nw = np.zeros(PA.PrepOperator.nw_bytes(HD) // 4, np.float32)
    nw[:2 * HD] = torch.cat([wq, wk]).numpy()
    fused.get_buffer("NW").view(torch.uint8)[:] = torch.from_numpy(nw.view(np.uint8))
    kv = fused.get_buffer("KV").view(torch.uint8).numpy()[:KVH * kvb].reshape(KVH, n_groups, shape.objects, A_BYTES)
    kv[:] = 0
    for h in range(KVH):  # (the cache before the chunk)
        if p0:
            k_h = torch.randn(p0, HD, generator=gen).numpy() * 2
            v_h = torch.randn(p0, HD, generator=gen).numpy()
            kv[h, guard:guard + p0 // shape.pg] = KV.kv_groups(k_h, v_h, A_BYTES, kp)
    q = torch.randn(T, COLS * QH, HD, generator=gen) * 2  # (head 2 c + i: column c's)
    k = torch.randn(T, KVH, HD, generator=gen) * 2
    v = torch.randn(T, KVH, HD, generator=gen)
    g = torch.randn(T, COLS * QH, generator=gen) * 2
    G = np.zeros((T, LAYOUT.N), np.float32)
    for c in range(COLS):
        h, j = divmod(c, 2)
        o = c * LAYOUT.pitch
        G[:, o + np.arange(QH * HD)] = q[:, QH * c:QH * c + QH].reshape(T, -1)
        G[:, o + LAYOUT.off["k"] + np.arange(64)] = k[:, h, 64 * j:64 * j + 64]
        G[:, o + LAYOUT.off["v"] + np.arange(64)] = v[:, h, 64 * j:64 * j + 64]
        G[:, o + LAYOUT.off["g"] + np.arange(QH)] = g[:, QH * c:QH * c + QH]
    fused.get_buffer("G")[:] = torch.from_numpy(G.reshape(-1)).bfloat16()
    fused.sync_inputs()
    fused.set_params(prep.params(p0) | attn.params(p0))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()

    def rope(x: torch.Tensor, w: torch.Tensor) -> np.ndarray:  # x (T, ..., hd) bf16 values normed and rotated (fp64)
        y = x.bfloat16().double().numpy().copy()
        y = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-5) * w.numpy()
        c_ = cs[p0:p0 + T, :HD // 2].astype(np.float64).reshape(T, *[1] * (y.ndim - 2), HD // 2)
        s_ = cs[p0:p0 + T, HD // 2:].astype(np.float64).reshape(c_.shape)
        a, b = y[..., :HD // 2].copy(), y[..., HD // 2:].copy()
        y[..., :HD // 2], y[..., HD // 2:] = a * c_ - b * s_, a * s_ + b * c_
        return y

    qs = fused.get_buffer("QS").view(torch.uint8).numpy()[:prep.n_qs * 2].reshape(COLS, T // 8, shape.tgu)
    qv = _q_values(qs[:, :, :QH * shape.qunit].reshape(COLS, T // 8, QH, shape.qunit), HD)
    qv = qv.transpose(0, 2, 1, 3, 4).reshape(COLS * QH, T, HD)  # [head][query][hd]
    gv = torch.from_numpy(qs[:, :, QH * shape.qunit:].copy()).view(torch.bfloat16).float().numpy()  # [col][tg][8][QH]
    gv = gv.reshape(COLS, T // 8, 8, QH).transpose(0, 3, 1, 2).reshape(COLS * QH, T)
    want_g = torch.sigmoid(g.bfloat16().double()).numpy().T
    assert np.abs(gv - want_g).max() < 1e-2
    want_q = rope(q, wq).transpose(1, 0, 2) * shape.scale
    assert _rel(qv, want_q) < 1e-2
    n = (p0 + T) // shape.pg
    Ks, Vs = {}, {}
    for h in range(KVH):
        Ks[h], Vs[h] = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[h, guard:guard + n], HD))
        assert _rel(Ks[h][p0:], rope(k, wk)[:, h]) < (2e-4 if kp == 2 else 1e-2), h
        assert _rel(Vs[h][p0:], v[:, h].bfloat16().double().numpy()) < 1e-2, h
    A = bfp16.from_panels(fused.get_buffer("A")[:attn.n_a].clone(), T, COLS * QH * HD).numpy().reshape(T, COLS * QH, HD)
    want = np.zeros((T, COLS * QH, HD))
    kpos, qpos = np.arange(p0 + T)[None], np.arange(p0, p0 + T)[:, None]
    masked = (kpos > qpos) | ((qpos - kpos >= win) if win else False)
    for hq in range(COLS * QH):
        h = hq // 4  # (its KV head: 4 query heads each)
        sc = qv[hq].astype(np.float64) @ Ks[h].T
        sc[masked] = -np.inf
        pr = np.exp2(sc - sc.max(-1, keepdims=True))
        want[:, hq] = (pr @ Vs[h]) / pr.sum(-1, keepdims=True) * gv[hq][:, None]
    err = _rel(A, want)
    eq = _rel(bfp16.values(want.reshape(T, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"attention {err:.2e} (the exact output rounded to bfp16: {eq:.2e}); QB, CR {shape.blocks}")
    assert err < 1.5 * eq
