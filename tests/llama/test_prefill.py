"""The Llama prefill on the NPU (models/llama/prefill.py; its GEMMs: prefill/test_gemm.py, its norms:
prefill/test_norm.py):

- the attention (prefill/attn.py): ``PrepOperator`` then ``AttnOperator`` on random projections of a chunk
  at p0 after a cache the host filled, in both K formats and the three models' head layouts;
- the prefill program on 4-layer slices against the decode path (slow): the caches, the last token's output row and
  logits.
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
# (head_dim, query heads per KV head, K panels per record, chunk, its position)
ATTN = [(64, 4, 1, 256, 0), (64, 4, 2, 256, 128), (64, 4, 1, 512, 384), (128, 3, 1, 256, 0), (128, 3, 2, 256, 64),
        (128, 3, 1, 512, 192), (128, 4, 1, 256, 0), (128, 4, 2, 512, 128), (128, 3, 1, 1024, 64),
        (64, 4, 1, 128, 128), (128, 3, 1, 64, 0), (128, 3, 2, 128, 192), (128, 4, 1, 64, 64)]  # (short chunks: 1 row)


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _q_values(units: np.ndarray, hd: int) -> np.ndarray:
    """Q units (..., 72 hd / 8) uint8 -> (..., 8 queries, hd) the values they hold."""
    lead = units.shape[:-1]
    v = bfp16.block_values(units.reshape(*lead, hd // 8, 8, 9))  # [db][query][8]
    return np.moveaxis(v, -3, -2).reshape(*lead, 8, hd)


def _layout(hd: int, qh: int) -> Layout:
    """The qkv GEMMs' output layout (``prefill.gemm``): per column [q heads | k | v], a pitch of whole 64 columns."""
    return Layout(-(-(qh + 2) * hd // 64) * 64, {"q": 0, "k": qh * hd, "v": (qh + 1) * hd})


@pytest.mark.parametrize(("hd", "qh", "kp", "T", "p0"), ATTN, ids=[f"hd{a}-qh{b}-kp{c}-T{d}-p{e}" for a, b, c, d, e in ATTN])
def test_attn(hd: int, qh: int, kp: int, T: int, p0: int) -> None:
    """The Q units and the K / V records against their fp64 model (K hi + lo within the panels' ~14 bits, Q and single
    K and V within bfp16), the output panels against the exact attention (causal) of the cores' own operands (their Q
    units, the cache they read): within 1.5 x the error of the exact output rounded to bfp16 (P as single bfp16 panels)."""
    shape = PA.Shape(hd, qh, kp, A_BYTES)
    max_pos = max(1024, p0 + T)
    n_groups = max_pos // shape.pg
    kvb = n_groups * shape.group
    lay = _layout(hd, qh)
    prep = PA.PrepOperator(T=T, shape=shape, layout=lay, kv_bytes=kvb, cs_positions=max_pos)
    attn = PA.AttnOperator(T=T, shape=shape, kv_bytes=kvb)
    fused = FusedOperator(name=f"pa_test{hd}_{qh}_{kp}", runlist=[(prep, "G", "CS", "QS", "KV"),
                                                                  (attn, "QS", "KV", "A")],
                          input_args=["G", "CS"], output_args=["A"]).compile()
    gen = torch.Generator().manual_seed(p0 + hd + kp)
    inv = 500000.0 ** (-np.arange(0, hd, 2) / hd)
    cs = PA.cs_table(inv, max_pos)
    fused.get_buffer("CS").view(torch.uint8)[:] = torch.from_numpy(cs.view(np.uint8).reshape(-1))
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

    def rope(x: torch.Tensor) -> np.ndarray:  # x (T, ..., hd) bf16 values rotated (fp64)
        y = x.bfloat16().double().numpy().copy()
        c_ = cs[p0:p0 + T, :hd // 2].astype(np.float64).reshape(T, *[1] * (y.ndim - 2), hd // 2)
        s_ = cs[p0:p0 + T, hd // 2:].astype(np.float64).reshape(c_.shape)
        a, b = y[..., :hd // 2].copy(), y[..., hd // 2:].copy()
        y[..., :hd // 2], y[..., hd // 2:] = a * c_ - b * s_, a * s_ + b * c_
        return y

    units = fused.get_buffer("QS").view(torch.uint8).numpy()[:T // 8 * COLS * qh * shape.qunit]
    qv = _q_values(units.reshape(COLS, T // 8, qh, shape.qunit), hd)  # [column][token group][head][query][hd]
    qv = qv.transpose(0, 2, 1, 3, 4).reshape(COLS, qh, T, hd)
    want_q = rope(q).transpose(1, 2, 0, 3) * shape.scale
    assert _rel(qv, want_q) < 1e-2
    n = (p0 + T) // shape.pg
    Ks, Vs = {}, {}
    for c in range(COLS):
        Ks[c], Vs[c] = (np.asarray(x, np.float64) for x in KV.kv_from_groups(kv[c, :n], hd))
        assert _rel(Ks[c][p0:], rope(k)[:, c]) < (2e-4 if kp == 2 else 1e-2), c
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


# (GGUF quantization, model size, K format, tokens, decode steps first)
CACHES = [("Q4_K_M", "1B", "single", 300, 0), ("Q4_K_M", "1B", "hilo", 300, 128), ("Q4_K_M", "3B", "single", 600, 64),
          ("Q4_K_S", "8B", "hilo", 300, 0)]


@pytest.mark.slow
@pytest.mark.parametrize(("quant", "size", "kv_k", "n", "p0"), CACHES, ids=[f"{s}_{q}-{k}-n{n}-p{p}" for q, s, k, n, p in CACHES])
def test_prefill_caches(quant: str, size: str, kv_k: str, n: int, p0: int) -> None:
    """The prefill program (LlamaPrefill) on a slice of 4 layers leaves the caches the decode path leaves for the same
    essay tokens, up to the rounding of the two paths' GEMM inputs (single bfp16 panels here, exact-integer weights
    against hi / lo inputs there): p0 tokens as decode steps first (a prefill continued after a decode), then n in the
    planner's chunks; the last token's output row against the decode path's LM input, and its logits."""
    import os

    from artlite.gguf_model import GGUFFile
    from models.llama.overlay import LlamaOverlay
    from models.llama.prefill import LlamaPrefill
    from models.llama.tokenizer import load_tokenizer

    from ..common import essay, llama_gguf

    path = llama_gguf(quant, size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    layers = [0, 1, 2, 3]
    ids = load_tokenizer(path).encode(essay("apple.txt"))[:p0 + n]
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
