"""The Llama prefill on the NPU (models/llama/prefill.py; its GEMMs: prefill/test_gemm.py, its norms:
prefill/test_norm.py):

- the attention (prefill/attn.py): ``PrepOperator`` then ``AttnOperator`` on random projections of a chunk
  at p0 after a cache the host filled, in both K formats and the models' head layouts (Qwen3's: q / k norms);
- the prefill program on 4-layer slices against the decode path (slow): the caches, the last token's output row and
  logits (Llama; Qwen3 0.6B Q4_1, Qwen3-Embedding-0.6B Q8_0 and -4B: q / k norms, all-Q8_0 GEMMs, a hidden size of
  2560 in padded K chunks).
"""

import numpy as np
import pytest
import torch

from artlite.decode import kv as KV
from artlite.operators import FusedOperator, bfp16
from artlite.operators.design import COLS
from artlite.prefill import attn as PA
from artlite.prefill.gemm import Layout

A_BYTES = 9280  # the cache objects (the Q4_K_M images' tile size)
# (head_dim, query heads per KV head, K panels per record, chunk, its position, q / k norms with weights and Q hi + lo:
# Qwen3)
ATTN = [(64, 4, 1, 256, 0, False), (64, 4, 2, 256, 128, False), (64, 4, 1, 512, 384, False),
        (128, 3, 1, 256, 0, False), (128, 3, 2, 256, 64, False), (128, 3, 1, 512, 192, False),
        (128, 4, 1, 256, 0, False), (128, 4, 2, 512, 128, False), (128, 3, 1, 1024, 64, False),
        (64, 4, 1, 128, 128, False), (128, 3, 1, 64, 0, False), (128, 3, 2, 128, 192, False),
        (128, 4, 1, 64, 64, False),  # (short chunks: 1 row)
        (128, 2, 2, 256, 0, True), (128, 2, 2, 512, 128, True), (128, 2, 1, 256, 64, True), (128, 2, 2, 64, 0, True),
        (128, 2, 2, 1024, 64, True), (128, 4, 2, 256, 64, True)]


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _q_values(units: np.ndarray, hd: int, qp: int = 1) -> np.ndarray:
    """Q units (..., 72 hd / 8 qp) uint8 -> (..., 8 queries, hd) the values they hold (qp 2: hi + lo)."""
    lead = units.shape[:-1]
    v = bfp16.block_values(units.reshape(*lead, qp, hd // 8, 8, 9)).sum(-4)  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, hd)


def _layout(hd: int, qh: int) -> Layout:
    """The qkv GEMMs' output layout (``prefill.gemm``): per column [q heads | k | v], a pitch of whole 64 columns."""
    return Layout(-(-(qh + 2) * hd // 64) * 64, {"q": 0, "k": qh * hd, "v": (qh + 1) * hd})


@pytest.mark.parametrize(("hd", "qh", "kp", "T", "p0", "norm"), ATTN,
                         ids=[f"hd{a}-qh{b}-kp{c}-T{d}-p{e}{'-norm' if f else ''}" for a, b, c, d, e, f in ATTN])
def test_attn(hd: int, qh: int, kp: int, T: int, p0: int, norm: bool) -> None:
    """The Q units and the K / V records against their fp64 model (K hi + lo within the panels' ~14 bits, Q and single
    K and V within bfp16), the output panels against the exact attention (causal) of the cores' own operands (their Q
    units, the cache they read): within 1.5 x the error of the exact output rounded to bfp16 (P as single bfp16 panels).
    ``norm``: q and k RMS-normed per head with random weights per dim (Qwen3's: up to ~100) before RoPE, Q hi + lo
    (within its ~14 bits)."""
    shape = PA.Shape(hd, qh, kp, A_BYTES, 2 if norm else 1, 2 if norm else 1)
    max_pos = max(1024, p0 + T)
    n_groups = max_pos // shape.pg
    kvb = n_groups * shape.group
    lay = _layout(hd, qh)
    prep = PA.PrepOperator(T=T, shape=shape, layout=lay, kv_bytes=kvb, cs_positions=max_pos, norm=norm)
    attn = PA.AttnOperator(T=T, shape=shape, kv_bytes=kvb)
    fused = FusedOperator(name=f"pa_test{hd}_{qh}_{kp}{'_n' if norm else ''}",
                          runlist=[(prep, "G", "CS", "QS", "KV", *(["NW"] if norm else [])), (attn, "QS", "KV", "A")],
                          input_args=["G", "CS", *(["NW"] if norm else [])], output_args=["A"]).compile()
    gen = torch.Generator().manual_seed(p0 + hd + kp)
    inv = 500000.0 ** (-np.arange(0, hd, 2) / hd)
    cs = PA.cs_table(inv, max_pos)
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(cs.view(np.uint8).reshape(-1))
    wq = (torch.rand(hd, generator=gen) * 4 + 0.5).double()
    wk = torch.exp(torch.randn(hd, generator=gen) * 1.5).double()  # (dims a thousand times apart)
    if norm:
        nw = np.zeros(PA.PrepOperator.nw_bytes(hd) // 4, np.float32)
        nw[:2 * hd] = torch.cat([wq, wk]).numpy()
        fused.get_buffer("NW").view(torch.uint8)[:] = torch.from_numpy(nw.view(np.uint8))
    kv = fused.get_buffer("KV").view(torch.uint8).numpy().reshape(COLS, n_groups, shape.objects, A_BYTES)
    kv[:] = 0
    for c in range(COLS):  # (the cache before the chunk)
        k_c = torch.randn(p0, hd, generator=gen).numpy() * 2
        v_c = torch.randn(p0, hd, generator=gen).numpy()
        if p0:
            kv[c, :p0 // shape.pg] = KV.kv_groups(k_c, v_c, A_BYTES, kp)
    q = torch.randn(T, COLS, qh, hd, generator=gen) * 2
    k = torch.randn(T, COLS, hd, generator=gen) * 2
    v = torch.randn(T, COLS, hd, generator=gen)
    G = np.zeros((T, lay.N), np.float32)
    for c in range(COLS):
        G[:, c * lay.pitch + lay.off["q"] + np.arange(qh * hd)] = q[:, c].reshape(T, -1)
        G[:, c * lay.pitch + lay.off["k"] + np.arange(hd)] = k[:, c]
        G[:, c * lay.pitch + lay.off["v"] + np.arange(hd)] = v[:, c]
    fused.get_buffer("G")[:] = torch.from_numpy(G.reshape(-1)).bfloat16()
    fused.sync_inputs()
    fused.set_params(prep.params(p0) | attn.params(p0))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()

    def rope(x: torch.Tensor, w: torch.Tensor | None = None) -> np.ndarray:  # x (T, ..., hd) bf16 values rotated (fp64)
        y = x.bfloat16().double().numpy().copy()
        if norm:  # (RMS-normed, times the weights)
            y = y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-6) * w.numpy()
        c_ = cs[p0:p0 + T, :hd // 2].astype(np.float64).reshape(T, *[1] * (y.ndim - 2), hd // 2)
        s_ = cs[p0:p0 + T, hd // 2:].astype(np.float64).reshape(c_.shape)
        a, b = y[..., :hd // 2].copy(), y[..., hd // 2:].copy()
        y[..., :hd // 2], y[..., hd // 2:] = a * c_ - b * s_, a * s_ + b * c_
        return y

    units = fused.get_buffer("QS").view(torch.uint8).numpy()[:T // 8 * COLS * qh * shape.qunit]
    qv = _q_values(units.reshape(COLS, T // 8, qh, shape.qunit), hd, shape.qp)  # [column][token group][head][query][hd]
    qv = qv.transpose(0, 2, 1, 3, 4).reshape(COLS, qh, T, hd)
    want_q = rope(q, wq).transpose(1, 2, 0, 3) * shape.scale
    assert _rel(qv, want_q) < (2e-4 if shape.qp == 2 else 1e-2)
    n = (p0 + T) // shape.pg
    Ks, Vs = {}, {}
    for c in range(COLS):
        Ks[c], Vs[c] = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[c, :n], hd))
        assert _rel(Ks[c][p0:], rope(k, wk)[:, c]) < (2e-4 if kp == 2 else 1e-2), c
        assert _rel(Vs[c][p0:], v[:, c].bfloat16().double().numpy()) < 1e-2, c
    A = bfp16.from_panels(fused.get_buffer("A")[:attn.n_a].clone(), T, COLS * qh * hd).numpy().reshape(T, COLS, qh, hd)
    want = np.zeros((T, COLS, qh, hd))
    masked = np.arange(p0 + T)[None] > np.arange(p0, p0 + T)[:, None]
    for c in range(COLS):
        for h in range(qh):
            sc = qv[c, h].astype(np.float64) @ Ks[c].T  # (log2 units)
            sc[masked] = -np.inf
            pr = np.exp2(sc - sc.max(-1, keepdims=True))
            want[:, c, h] = (pr @ Vs[c]) / pr.sum(-1, keepdims=True)
    err = _rel(A, want)
    eq = _rel(bfp16.values(want.reshape(T, -1).astype(np.float32)).reshape(want.shape), want)
    print(f"attention {err:.2e} (the exact output rounded to bfp16: {eq:.2e}); QB, CR {shape.blocks}")
    assert err < 1.5 * eq


# (head_dim, query heads per KV head, K panels per record, q / k norms (Qwen3: Q, P hi + lo), chunk, history, window,
# segments (first token, tokens): starts inside a record, a group, at 32-token chunks of keys; the longest the window's)
PACKED = [(128, 2, 2, True, 512, 0, 3, [(0, 100), (128, 64), (192, 37), (230, 122), (352, 129)]),
          (128, 2, 2, True, 1024, 128, 4, [(0, 193), (193, 7), (200, 120), (320, 190), (512, 193), (705, 300)]),
          (64, 4, 1, False, 1024, 0, 2, [(0, 129), (129, 100), (256, 64), (320, 3), (323, 129), (452, 500)]),
          (128, 4, 2, False, 512, 64, 2, [(0, 65), (96, 96), (192, 31), (223, 65), (320, 65)])]


@pytest.mark.parametrize(("hd", "qh", "kp", "norm", "T", "p0", "win", "segs"), PACKED,
                         ids=[f"hd{a}-qh{b}-kp{c}{'-norm' if d else ''}-T{e}-p{f}-w{g}" for a, b, c, d, e, f, g, _ in PACKED])
def test_attn_packed(hd: int, qh: int, kp: int, norm: bool, T: int, p0: int, win: int,
                     segs: list[tuple[int, int]]) -> None:
    """A packed chunk (``AttnOperator(window=)``: segments of the chunk, each its own rows' attention over the history
    and its keys up to the row; rotary positions from p0 per segment, the prep's cos / sin rows written past the
    table's) against the exact attention of the cores' own operands, as ``test_attn``; a segment that starts at a
    32-token chunk of keys bit-identical to the same tokens run alone (the unpacked design at p0: the history, the
    segment's keys in the same chunks, the skipped ones adding nothing) up to its last whole V record. A segment too
    long for the window: padding (its rows see only themselves)."""
    shape = PA.Shape(hd, qh, kp, A_BYTES, 2 if norm else 1, 2 if norm else 1)
    max_pos = max(1024, p0 + T)
    n_groups = max_pos // shape.pg
    kvb = n_groups * shape.group
    lay = _layout(hd, qh)
    prep = PA.PrepOperator(T=T, shape=shape, layout=lay, kv_bytes=kvb, cs_positions=max_pos + T, norm=norm)
    attn = PA.AttnOperator(T=T, shape=shape, kv_bytes=kvb, window=win)
    nw_arg = ["NW"] if norm else []
    fused = FusedOperator(name=f"pa_packed{hd}_{qh}_{kp}_{T}_{win}{'_n' if norm else ''}",
                          runlist=[(prep, "G", "CS", "QS", "KV", *nw_arg), (attn, "QS", "KV", "A", "SEG")],
                          input_args=["G", "CS", "SEG", *nw_arg], output_args=["A"]).compile()
    segs = [(a, n) for a, n in segs if n <= attn.max_segment]  # (the rest padding)
    gen = torch.Generator().manual_seed(p0 + hd + kp + T)
    inv = 500000.0 ** (-np.arange(0, hd, 2) / hd)
    pos = np.arange(p0, p0 + T)  # (rotary positions; padding: its cache position)
    lo = np.arange(T)
    for a, n in segs:
        pos[a:a + n] = p0 + np.arange(n)
        lo[a:a + n] = a
    cs = PA.cs_table(inv, max_pos)
    rows = PA.cs_table(inv, max_pos + T)[pos]
    csb = np.concatenate([cs, rows])
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(csb.view(np.uint8).reshape(-1))
    fused.get_buffer("SEG").view(torch.uint8)[:] = torch.from_numpy(attn.seg(lo))
    wq = (torch.rand(hd, generator=gen) * 4 + 0.5).double()
    wk = torch.exp(torch.randn(hd, generator=gen) * 1.5).double()
    nw = np.zeros(PA.PrepOperator.nw_bytes(hd) // 4, np.float32)
    nw[:2 * hd] = torch.cat([wq, wk]).numpy()
    if norm:
        fused.get_buffer("NW").view(torch.uint8)[:] = torch.from_numpy(nw.view(np.uint8))
    kv = fused.get_buffer("KV").view(torch.uint8).numpy().reshape(COLS, n_groups, shape.objects, A_BYTES)
    kv[:] = 0
    hist = [(torch.randn(p0, hd, generator=gen).numpy() * 2, torch.randn(p0, hd, generator=gen).numpy())
            for _ in range(COLS)]
    for c in range(COLS):
        if p0:
            kv[c, :p0 // shape.pg] = KV.kv_groups(*hist[c], A_BYTES, kp)
    q = torch.randn(T, COLS, qh, hd, generator=gen) * 2
    k = torch.randn(T, COLS, hd, generator=gen) * 2
    v = torch.randn(T, COLS, hd, generator=gen)

    def g_rows(sel: np.ndarray, n: int) -> np.ndarray:  # (the qkv output of the tokens sel, padded to n rows)
        G = np.zeros((n, lay.N), np.float32)
        for c in range(COLS):
            G[:len(sel), c * lay.pitch + lay.off["q"] + np.arange(qh * hd)] = q[sel, c].reshape(len(sel), -1)
            G[:len(sel), c * lay.pitch + lay.off["k"] + np.arange(hd)] = k[sel, c]
            G[:len(sel), c * lay.pitch + lay.off["v"] + np.arange(hd)] = v[sel, c]
        return G.reshape(-1)

    fused.get_buffer("G")[:] = torch.from_numpy(g_rows(np.arange(T), T)).bfloat16()
    fused.sync_inputs()
    fused.set_params(prep.params(p0) | {PA.P_CS: 2 * hd * max_pos} | attn.params(p0))
    fused()
    fused.sync_outputs()
    fused.sync_scratch_from_device()
    units = fused.get_buffer("QS").view(torch.uint8).numpy()[:T // 8 * COLS * qh * shape.qunit]
    qv = _q_values(units.reshape(COLS, T // 8, qh, shape.qunit), hd, shape.qp)
    qv = qv.transpose(0, 2, 1, 3, 4).reshape(COLS, qh, T, hd)
    n = (p0 + T) // shape.pg
    A = bfp16.from_panels(fused.get_buffer("A")[:attn.n_a].clone(), T, COLS * qh * hd).numpy().reshape(T, COLS, qh, hd)
    want = np.zeros((T, COLS, qh, hd))
    keys = np.arange(p0 + T)
    seen = (keys[None] < p0) | ((keys[None] >= p0 + lo[:, None]) & (keys[None] <= p0 + np.arange(T)[:, None]))
    for c in range(COLS):
        Kc, Vc = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[c, :n], hd))
        for h in range(qh):
            sc = qv[c, h].astype(np.float64) @ Kc.T
            sc[~seen] = -np.inf
            pr = np.exp2(sc - sc.max(-1, keepdims=True))
            want[:, c, h] = (pr @ Vc) / pr.sum(-1, keepdims=True)
    valid = np.concatenate([np.arange(a, a + n) for a, n in segs])
    err = _rel(A[valid], want[valid])
    eq = _rel(bfp16.values(want[valid].reshape(len(valid), -1).astype(np.float32)).reshape(want[valid].shape),
              want[valid])
    print(f"packed attention {err:.2e} (the exact output rounded to bfp16: {eq:.2e}); QB, CR {shape.blocks}")
    assert err < 1.5 * eq
    for a, n in segs:  # (each segment alone: the unpacked design, the same tokens at p0)
        if a % 32:
            continue
        Ts = -(-n // shape.pg) * shape.pg
        prep1 = PA.PrepOperator(T=Ts, shape=shape, layout=lay, kv_bytes=kvb, cs_positions=max_pos, norm=norm)
        attn1 = PA.AttnOperator(T=Ts, shape=shape, kv_bytes=kvb)
        alone = FusedOperator(name=f"pa_alone{hd}_{qh}_{kp}_{Ts}{'_n' if norm else ''}",
                              runlist=[(prep1, "G", "CS", "QS", "KV", *nw_arg), (attn1, "QS", "KV", "A")],
                              input_args=["G", "CS", *nw_arg], output_args=["A"]).compile()
        alone.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(cs.view(np.uint8).reshape(-1))
        if norm:
            alone.get_buffer("NW").view(torch.uint8)[:] = torch.from_numpy(nw.view(np.uint8))
        kv1 = alone.get_buffer("KV").view(torch.uint8).numpy().reshape(COLS, n_groups, shape.objects, A_BYTES)
        kv1[:] = 0
        for c in range(COLS):
            if p0:
                kv1[c, :p0 // shape.pg] = KV.kv_groups(*hist[c], A_BYTES, kp)
        alone.get_buffer("G")[:] = torch.from_numpy(g_rows(np.arange(a, a + n), Ts)).bfloat16()
        alone.sync_inputs()
        alone.set_params(prep1.params(p0) | attn1.params(p0))
        alone()
        alone.sync_outputs()
        A1 = bfp16.from_panels(alone.get_buffer("A")[:attn1.n_a].clone(), Ts, COLS * qh * hd).numpy()
        m = n - n % 8  # (a V record's blocks span its 8 positions: a last record shared with the next segment rounds
        # its V otherwise, so only the rows before it are bit-identical)
        A1 = A1.reshape(Ts, COLS, qh, hd)[:m]
        bad = np.flatnonzero((A1 != A[a:a + m]).any((1, 2, 3)))
        assert bad.size == 0, f"segment {a} + {n}: rows {bad.tolist()} differ from it alone"


# (GGUF quantization, model size, K format, tokens, decode steps first; "qwen3:<name>": model_config.qwen3_gguf's file)
CACHES = [("Q4_K_M", "1B", "single", 300, 0), ("Q4_K_M", "1B", "hilo", 300, 128), ("Q4_K_M", "3B", "single", 600, 64),
          ("Q4_K_S", "8B", "hilo", 300, 0), ("Q4_1", "qwen3:0.6B", "hilo", 300, 64),
          ("Q8_0", "qwen3:embedding-0.6B", "hilo", 600, 0), ("Q4_K_M", "qwen3:embedding-4B", "hilo", 300, 64)]


@pytest.mark.slow
@pytest.mark.parametrize(("quant", "size", "kv_k", "n", "p0"), CACHES, ids=[f"{s}_{q}-{k}-n{n}-p{p}" for q, s, k, n, p in CACHES])
def test_prefill_caches(quant: str, size: str, kv_k: str, n: int, p0: int) -> None:
    """The prefill program (LlamaPrefill) on a slice of 4 layers leaves the caches the decode path leaves for the same
    essay tokens, up to the rounding of the two paths' GEMM inputs (single bfp16 panels here, exact-integer weights
    against hi / lo inputs there): p0 tokens as decode steps first (a prefill continued after a decode), then n in the
    planner's chunks; the last token's output row against the decode path's LM input, and its logits."""
    import os

    from artlite.gguf_model import GGUFFile
    from artlite.model_config import qwen3_gguf
    from models.llama.overlay import LlamaOverlay
    from models.llama.prefill import LlamaPrefill
    from models.llama.tokenizer import load_tokenizer
    from models.tokenizer import QwenTokenizer

    from ..common import essay, llama_gguf

    qwen3 = size.startswith("qwen3:")
    path = qwen3_gguf(size.removeprefix("qwen3:")) if qwen3 else llama_gguf(quant, size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    layers = [0, 1, 2, 3]
    ids = (QwenTokenizer(g) if qwen3 else load_tokenizer(path)).encode(essay("apple.txt"))[:p0 + n]
    x = g.rows("token_embd.weight", ids).float()
    ov = LlamaOverlay(g, layers=layers, vocab=2048, max_positions=1024, kv_k=kv_k)
    ov.load()
    for t in range(0, p0 + n - 1, 4):  # the decode path
        ov.step(x[t:min(t + 4, p0 + n - 1)], t, logits=False)
    want_l = ov.step(x[p0 + n - 1:], p0 + n - 1).float()[0]
    want_h = ov.hidden(-1, 0)
    want = {L: ov.read_kv(L, range(p0 + n)) for L in layers}
    ov.reset()
    for t in range(0, p0, 4):
        ov.step(x[t:t + 4], t, logits=False)
    pf = LlamaPrefill(ov).compile().bind()
    t = p0
    for T in pf.plan(p0, n):
        m = min(T, p0 + n - t)
        pf.run(x[t:t + m], t, T)
        t += m
    assert t == p0 + n
    worst = {}
    for L in layers:
        k, v = ov.read_kv(L, range(p0 + n))
        worst[f"K.{L}"], worst[f"V.{L}"] = _rel(k.numpy(), want[L][0].numpy()), _rel(v.numpy(), want[L][1].numpy())
    h = pf.hidden(m - 1)
    worst["hidden"] = _rel(h.numpy(), want_h.numpy())
    worst["logits"] = _rel(ov.logits_of(h).float().numpy(), want_l.numpy())
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < 0.06 for v in worst.values()), worst
