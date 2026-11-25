"""Inkling's prefill attention (``prefill.attn.ink``'s prep, ``AttnOperator(ink=True)``) on random projections of a chunk
at p0 after a cache and a raw ring the host filled: the Q units, R panels and sinks against their fp64 model (hi + lo
panels: within ~2^-16), the K / V records against the convs over the raw history (the ring's rows before p0), clipped /
normed, the ring written back (the last 8 valid positions' raw rows), a full layer's tap deltas and tau."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.decode import kv as KV
from artlite.operators import FusedOperator, bfp16
from artlite.operators.design import COLS
from artlite.operators.util import arg_bytes
from artlite.prefill import attn as PA
from artlite.prefill.attn.ink import HEADS, INK_ROWS, QS_UNITS, RING_BYTES, InkPrepOperator, ink_shape
from artlite.prefill.gemm import Layout

from ..numerics import rel

A_BYTES = 9216
HD = 128
LQ = Layout(896, {"q": 0, "gate": 384, "r": 768, "sink": 832})  # (Inkling's q / k / v GEMM layouts, ``prefill.gemm``)
LK = {False: Layout(128, {"k": 0}), True: Layout(192, {"k": 0, "kd": 128})}
LV = {False: Layout(128, {"v": 0}), True: Layout(192, {"v": 0, "vd": 128})}
WIN = 1024
# (chunk, position, valid tokens, a full layer)
PREP = [(512, 0, 512, False), (256, 192, 201, False), (512, 8192, 500, True), (128, 64, 3, True)]


def _q_values(units: np.ndarray, nd: int) -> np.ndarray:
    """Panels of hi + lo units (..., 2 x 72 nd) uint8 -> (..., 8 rows, 8 nd) the values they hold."""
    lead = units.shape[:-1]
    u = units.reshape(*lead, 2, 72 * nd)
    v = sum(bfp16.block_values(u[..., i, :].reshape(*lead, nd, 8, 9)) for i in range(2))  # [db][row][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, 8 * nd)


def _clip(x: np.ndarray, thr: float) -> np.ndarray:
    n = np.sqrt((x * x).sum(-1, keepdims=True))
    return x * np.minimum(1.0, thr / np.where(n > 0, n, 1.0))


def _tau(positions: np.ndarray) -> np.ndarray:
    return 1 + 0.1 * np.log(np.maximum(1.0, (positions + 1) / 8192))


@pytest.mark.parametrize(("T", "p0", "n", "full"), PREP, ids=[f"T{a}-p{b}-n{c}-{'full' if d else 'swa'}" for a, b, c, d in PREP])
def test_ink_prep(T: int, p0: int, n: int, full: bool) -> None:
    s = ink_shape(A_BYTES)
    max_pos = max(1024, p0 + T)
    guard = 0 if full else PA.mode.swa_groups(WIN, s.pg)
    n_groups = guard + max_pos // s.pg
    kvb, kv0 = n_groups * s.group, guard * s.group
    prep = InkPrepOperator(T=T, shape=s, lq=LQ, lk=LK[full], lv=LV[full], full=full, kv_bytes=kvb,
                           tau_positions=max_pos, kv0=kv0)
    fused = FusedOperator(name=f"ink_prep_test_{'f' if full else 's'}{T}", input_args=["GQ", "GK", "GV", "TAU", "PRM"],
                          output_args=["QS", "KV", "RING"],
                          runlist=[(prep, "GQ", "GK", "GV", "TAU", "PRM", "QS", "KV", "RING", "JUNK")]).compile()
    gen = torch.Generator().manual_seed(p0 + T + n)
    lk, lv = LK[full], LV[full]
    q = (torch.randn(T, COLS * HEADS, HD, generator=gen) * 0.5).bfloat16().double().numpy()
    r = (torch.randn(T, COLS * HEADS, 16, generator=gen) * 3).bfloat16().double().numpy()
    sk = (torch.randn(T, COLS * HEADS, generator=gen) * 3).float().numpy()
    k = (torch.randn(T, COLS, HD, generator=gen)).bfloat16().double().numpy()
    v = (torch.randn(T, COLS, HD, generator=gen)).bfloat16().double().numpy()
    kd = (torch.randn(T, COLS, 4, generator=gen) * 0.3).float().numpy()
    vd = (torch.randn(T, COLS, 4, generator=gen) * 0.3).float().numpy()
    qc = (torch.rand(COLS * HEADS, HD, generator=gen) + 0.5).bfloat16().double().numpy()
    rc = (torch.rand(COLS * HEADS, 16, generator=gen) + 0.5).bfloat16().double().numpy()
    kt = (torch.randn(4, COLS * HD, generator=gen) * 0.5).bfloat16().double().numpy()
    vt = (torch.randn(4, COLS * HD, generator=gen) * 0.5).bfloat16().double().numpy()
    ring = (torch.randn(COLS, 8, 2, HD, generator=gen)).float().numpy()  # (positions p0 - 8 .. p0 - 1 at slot p % 8)
    GQ = np.zeros((T, LQ.N), np.float32)
    GK, GV = np.zeros((T, lk.N), np.float32), np.zeros((T, lv.N), np.float32)
    for c in range(COLS):
        o, hs = c * LQ.pitch, slice(HEADS * c, HEADS * (c + 1))
        GQ[:, o:o + HEADS * HD] = q[:, hs].reshape(T, -1)
        GQ[:, o + 768:o + 768 + HEADS * 16] = r[:, hs].reshape(T, -1)
        GK[:, c * lk.pitch:c * lk.pitch + HD] = k[:, c]
        GV[:, c * lv.pitch:c * lv.pitch + HD] = v[:, c]
    G = {name: g.astype(bfloat16) for name, g in (("GQ", GQ), ("GK", GK), ("GV", GV))}
    for c in range(COLS):  # (fp32 values as bf16 pairs: the sinks, the deltas)
        G["GQ"].view(np.uint16).reshape(T, -1)[:, c * LQ.pitch + 832:c * LQ.pitch + 838] = \
            sk[:, HEADS * c:HEADS * (c + 1)].view(np.uint16)
        if full:
            G["GK"].view(np.uint16).reshape(T, -1)[:, c * lk.pitch + 128:c * lk.pitch + 136] = kd[:, c].view(np.uint16)
            G["GV"].view(np.uint16).reshape(T, -1)[:, c * lv.pitch + 128:c * lv.pitch + 136] = vd[:, c].view(np.uint16)
    for name, g in G.items():
        fused.get_buffer(name)[:] = g.reshape(-1)
    tau = _tau(np.arange(max_pos)).astype(np.float32)
    fused.get_buffer("TAU").view(np.uint8)[:] = tau.view(np.uint8)
    fused.get_buffer("PRM").view(np.uint8)[:] = prep.prm(qc, rc, kt, vt)
    rg = fused.get_buffer("RING").view(np.uint8).reshape(COLS, RING_BYTES)
    rg[:] = 0
    rg[:, :8192] = ring.reshape(COLS, -1).view(np.uint8)
    kvbuf = fused.get_buffer("KV").view(np.uint8)[:COLS * kvb].reshape(COLS, n_groups, s.objects, A_BYTES)
    kvbuf[:] = 0
    fused.get_buffer("QS").view(np.uint8)[:] = 0
    fused.sync_inputs()
    fused.set_params(prep.params(p0, n))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()

    # Q units, R panels, sinks
    qs = fused.get_buffer("QS").view(np.uint8)[:prep.n_qs * 2].reshape(COLS, T // 8, QS_UNITS, s.unit)
    t8 = tau[p0:p0 + T].astype(np.float64) if full else np.ones(T)
    want_q = _clip(q * qc[None], 512 ** 0.5) * t8[:, None, None] * s.scale  # (T, heads, hd)
    got_q = _q_values(qs[:, :, :HEADS], s.nd)  # (COLS, T / 8, 3, 8, hd)
    got_q = got_q.transpose(1, 3, 0, 2, 4).reshape(T, COLS * HEADS, HD)
    assert rel(got_q, want_q) < 3e-5, rel(got_q, want_q)
    ink = qs[:, :, 3, :INK_ROWS * 144]
    want_r = _clip(r * rc[None], 8.0) * t8[:, None, None] / np.log(2)
    rp = ink[:, :, :6 * 144].reshape(COLS, T // 8, HEADS, 2 * 144)  # (per head [hi 2 panels | lo 2 panels])
    got_r = _q_values(rp, 2).transpose(1, 3, 0, 2, 4).reshape(T, COLS * HEADS, 16)
    assert rel(got_r, want_r) < 3e-5, rel(got_r, want_r)
    got_s = ink[:, :, 6 * 144:6 * 144 + 96].copy().view(np.float32).reshape(COLS, T // 8, HEADS, 8)
    got_s = got_s.transpose(1, 3, 0, 2).reshape(T, COLS * HEADS)
    assert rel(got_s, sk / np.log(2)) < 1e-5

    # K / V records, the ring
    pos = np.arange(p0 - 8, p0 + T)
    for c in range(COLS):
        raw = {}
        for i, x in enumerate((k, v)):
            h = np.zeros((T + 8, HD))
            h[8:] = x[:, c]
            for j in range(8):  # (the ring's rows: positions p0 - 8 + j at slot (p0 - 8 + j) % 8)
                h[j] = ring[c, (p0 - 8 + j) % 8, i]
            raw[i] = h
        g0 = guard + p0 // s.pg
        Kc, Vc = (np.asarray(a, np.float64) for a in KV.kv_from_groups(kvbuf[c, g0:g0 + T // s.pg], HD))
        for i, (x, dlt, taps) in enumerate(((k, kd, kt), (v, vd, vt))):
            h = raw[i]
            y = h[8:].copy()
            for kk in range(4):  # (tap kk: position t - 3 + kk)
                y += taps[kk, HD * c:HD * (c + 1)] * h[8 - 3 + kk:8 - 3 + kk + T]
            if full:
                for lag in range(4):
                    y += dlt[:, c, lag:lag + 1].astype(np.float64) * h[8 - lag:8 - lag + T]
            y *= 1 / 3 if full else 5 ** -0.5
            if i == 0:
                want = _clip(y, 512 ** 0.5)
                assert rel(Kc[:n], want[:n]) < 3e-5, (c, rel(Kc[:n], want[:n]))
            else:
                want = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-6)
                assert rel(Vc[:n], want[:n]) < 1e-2, (c, rel(Vc[:n], want[:n]))
        got_ring = rg[c, :8192].copy().view(np.float32).reshape(8, 2, HD)
        for slot in range(8):  # (the last valid position at this slot)
            p_ = max(pp for pp in pos[:8 + n] if pp % 8 == slot)
            for i in range(2):
                assert np.array_equal(got_ring[slot, i], raw[i][p_ - (p0 - 8)].astype(np.float32)), (c, slot, i)


# (chunk, position, a full layer)
ATTN = [(256, 0, False), (512, 1088, False), (256, 0, True), (512, 1536, True), (128, 2048, False), (2048, 0, False)]


@pytest.mark.parametrize(("T", "p0", "full"), ATTN, ids=[f"T{a}-p{b}-{'full' if c else 'swa'}" for a, b, c in ATTN])
def test_ink_attention(T: int, p0: int, full: bool) -> None:
    """The prep and the attention of a chunk after a random cache: the output panels against the exact attention (the
    relative bias R . E[:, d] for d < 1024, sliding window 1024, the rows' sinks) of the cores' own operands (the prep's Q
    and R panels and sinks, the cache's K / V)."""
    s = ink_shape(A_BYTES)
    max_pos = max(2048, p0 + T)
    guard = PA.mode.swa_groups(WIN, s.pg)  # (a full layer's cache too: the near part reads from 16 groups before)
    n_groups = guard + max_pos // s.pg
    kvb, kv0 = n_groups * s.group, guard * s.group
    pt = min(T, 512)  # (the prep in parts of 512 tokens, the attention's items: as the prefill's)
    preps = [InkPrepOperator(T=pt, shape=s, lq=LQ, lk=LK[full], lv=LV[full], full=full, kv_bytes=kvb,
                             tau_positions=max_pos, kv0=kv0, part=k) for k in range(T // pt)]
    prep = preps[0]
    attn = PA.AttnOperator(T=T, shape=s, kv_bytes=kvb, swa=0 if full else WIN, kv0=kv0, sub=pt, ink=True)
    lk, lv = LK[full], LV[full]

    def part(buf: str, k: int, row: int, op: object, i: int) -> str:
        a = k * pt * row
        return f"{buf}[{a}:{a + arg_bytes(op, i)}]"

    fused = FusedOperator(name=f"ink_attn_test_{'f' if full else 's'}{T}",
                          input_args=["GQ", "GK", "GV", "TAU", "PRM", "E"], output_args=["A", "RING"],
                          runlist=[(pp, part("GQ", k, 2 * LQ.N, pp, 0), part("GK", k, 2 * lk.N, pp, 1),
                                    part("GV", k, 2 * lv.N, pp, 2), "TAU", "PRM", part("QS", k, 2 * pp.n_qs // pt, pp, 5),
                                    "KV", "RING", "JUNK") for k, pp in enumerate(preps)] +
                                  [(attn, "QS", "KV", "A", "E")],
                          buffer_sizes={"GQ": 2 * T * LQ.N, "GK": 2 * T * lk.N, "GV": 2 * T * lv.N}).compile()
    gen = torch.Generator().manual_seed(p0 + T + 7 * full)
    GQ = np.zeros((T, LQ.N), np.float32)
    GK, GV = np.zeros((T, lk.N), np.float32), np.zeros((T, lv.N), np.float32)
    for c in range(COLS):
        o = c * LQ.pitch
        GQ[:, o:o + HEADS * HD] = (torch.randn(T, HEADS * HD, generator=gen) * 2).numpy()
        GQ[:, o + 768:o + 768 + HEADS * 16] = (torch.randn(T, HEADS * 16, generator=gen) * 3).numpy()
        GK[:, c * lk.pitch:c * lk.pitch + HD] = (torch.randn(T, HD, generator=gen) * 2).numpy()
        GV[:, c * lv.pitch:c * lv.pitch + HD] = torch.randn(T, HD, generator=gen).numpy()
    G = {name: g.astype(bfloat16) for name, g in (("GQ", GQ), ("GK", GK), ("GV", GV))}
    sk = (torch.randn(T, COLS * HEADS, generator=gen) * 2).float().numpy()
    for c in range(COLS):
        G["GQ"].view(np.uint16).reshape(T, -1)[:, c * LQ.pitch + 832:c * LQ.pitch + 838] = \
            sk[:, HEADS * c:HEADS * (c + 1)].view(np.uint16)
    for name, g in G.items():
        fused.get_buffer(name)[:] = g.reshape(-1)
    tau = _tau(np.arange(max_pos)).astype(np.float32)
    fused.get_buffer("TAU").view(np.uint8)[:] = tau.view(np.uint8)
    qc = (torch.rand(COLS * HEADS, HD, generator=gen) + 0.5).bfloat16().double().numpy()
    rc = (torch.rand(COLS * HEADS, 16, generator=gen) + 0.5).bfloat16().double().numpy()
    kt = (torch.randn(4, COLS * HD, generator=gen) * 0.5).bfloat16().double().numpy()
    vt = (torch.randn(4, COLS * HD, generator=gen) * 0.5).bfloat16().double().numpy()
    fused.get_buffer("PRM").view(np.uint8)[:] = prep.prm(qc, rc, kt, vt)
    E = torch.randn(16, 1024, generator=gen).numpy()
    E = E / np.maximum(1.0, np.sqrt((E * E).sum(0, keepdims=True)) / 2)  # (column norms <= 2: Inkling's)
    E = E.astype(bfloat16).astype(np.float64)
    fused.get_buffer("E")[:] = attn.e_table(E)
    fused.get_buffer("RING").view(np.uint8)[:] = 0
    kv = fused.get_buffer("KV").view(np.uint8)[:COLS * kvb].reshape(COLS, n_groups, s.objects, A_BYTES)
    kv[:] = 0
    for c in range(COLS):  # (the cache before the chunk)
        if p0:
            k_h = torch.randn(p0, HD, generator=gen).numpy()
            k_h *= np.minimum(1.0, 512 ** 0.5 / np.sqrt((k_h * k_h).sum(-1, keepdims=True)))
            v_h = torch.randn(p0, HD, generator=gen).numpy()
            kv[c, guard:guard + p0 // s.pg] = KV.kv_groups(k_h, v_h, A_BYTES, 2)
    fused.get_buffer("QS").view(np.uint8)[:] = 0
    fused.sync_inputs()
    fused.set_params({k: v for pp in preps for k, v in pp.params(p0, T).items()} | attn.params(p0))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()

    qs = fused.get_buffer("QS").view(np.uint8)[:len(preps) * prep.n_qs * 2]
    qs = qs.reshape(len(preps), COLS, pt // 8, QS_UNITS, s.unit).transpose(1, 0, 2, 3, 4).reshape(COLS, T // 8, QS_UNITS, s.unit)
    qv = _q_values(qs[:, :, :HEADS], s.nd).transpose(0, 2, 1, 3, 4).reshape(COLS, HEADS, T, HD)
    ink = qs[:, :, 3, :INK_ROWS * 144]
    rv = _q_values(ink[:, :, :6 * 144].reshape(COLS, T // 8, HEADS, 288), 2).transpose(0, 2, 1, 3, 4)
    rv = rv.reshape(COLS, HEADS, T, 16)
    sv = ink[:, :, 6 * 144:6 * 144 + 96].copy().view(np.float32).reshape(COLS, T // 8, HEADS, 8)
    sv = sv.transpose(0, 2, 1, 3).reshape(COLS, HEADS, T).astype(np.float64)
    A = bfp16.from_panels(fused.get_buffer("A")[:attn.n_a], T, COLS * HEADS * HD).reshape(T, COLS, HEADS, HD)
    n = (p0 + T) // s.pg
    kpos, qpos = np.arange(p0 + T)[None], np.arange(p0, p0 + T)[:, None]
    d = qpos - kpos
    masked = (d < 0) | ((d >= WIN) if not full else False)
    want = np.zeros((T, COLS, HEADS, HD))
    for c in range(COLS):
        Kc, Vc = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[c, guard:guard + n], HD))
        for h in range(HEADS):
            sc = qv[c, h].astype(np.float64) @ Kc.T  # (log2 units: the prep's score constant)
            rb = rv[c, h].astype(np.float64) @ E  # (T, 1024): log2 units
            bias = np.where((d >= 0) & (d < 1024), np.take_along_axis(rb, np.clip(d, 0, 1023), 1), 0.0)
            sc = np.where(masked, -np.inf, sc + bias)
            mx = np.maximum(sc.max(-1, keepdims=True), sv[c, h][:, None])
            pr = np.exp2(sc - mx)
            want[:, c, h] = (pr @ Vc) / (pr.sum(-1, keepdims=True) + np.exp2(sv[c, h][:, None] - mx))
    err = rel(A, want)
    eq = rel(bfp16.values(want.reshape(T, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"attention {err:.2e} (the exact output rounded to bfp16: {eq:.2e})")
    assert err < 1.9 * eq
