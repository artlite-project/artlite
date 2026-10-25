"""The multi-token-prediction drafter of Qwen3.6-35B-A3B (``qwen35.draft.QwenDraft``, ggml-org's Q4_0 MTP file) and
of Occamy 1.0 (its Q4_K_M MTP file: Q4_K and Q6_K) on the decode overlay of a 2-layer slice (layers 0 and 3: the
model's design, its formats with the drafter's), against
``qwen35_ref.mtp_block`` on llama.cpp's hidden states of an essay (golden(mtp=): the rows' h and the drafter file's
token embeddings): per row the next draft's h and the head's logits, in steps of S rows from position 0. The prompt's
records (``QwenPrefill(draft=)``, ``draft_prefill``) against those of drafter steps from the same inputs."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_mtp_of
from artlite.operators.qwen35.draft import QwenDraft, draft_formats
from artlite.operators.qwen35.overlay import QwenOverlay
from artlite.operators.qwen35.prefill import QwenPrefill

from . import qwen35_ref as ref
from .common import essay

N = 100
# (per model: the router gap below which a row's 8th and 9th experts tie, h's median and largest error elsewhere;
# Occamy's drafter is Q4_K / Q6_K, whose decode tiles hold d and dmin as bf16: that alone moves h by 0.9% median)
TOL = {"35B-A3B": (0.01, 7e-3, 1.5e-2), "occamy": (0.02, 1.5e-2, 2.5e-2)}


def _files(size: str) -> tuple[str, str]:
    """The model file of a size ("35B-A3B", "occamy") and its drafter's."""
    model = ref.size_gguf(size)
    mtp = qwen_mtp_of(model) if os.path.exists(model) else None
    if mtp is None:
        pytest.skip(f"no {model} or its MTP drafter")
    return model, mtp


@pytest.fixture(scope="module", params=["35B-A3B", "occamy"])
def setup(request: pytest.FixtureRequest) -> tuple:
    model, mtp = _files(request.param)
    tokens = ref.llama_tokenize(model, essay("apple.txt"))[:N]
    G = ref.golden(model, tokens, layers=(40, "g"), names="mtp_tok_embd|result_norm", logits="none", mtp=mtp)
    emb = np.asarray(G.mtp("mtp_tok_embd", 40), np.float32).reshape(N, -1)
    hr = np.asarray(G["result_norm"], np.float32).reshape(N, -1)
    h = np.concatenate([np.zeros_like(hr[:1]), hr[:-1]])
    gm = GGUFFile(mtp)
    target = QwenOverlay(GGUFFile(model), layers=[0, 3], vocab=4096, max_positions=1024, formats=draft_formats(gm))
    target.load()
    draft = QwenDraft(target, gm)
    draft.load()
    r = ref.mtp_block(gm, QwenConfig.from_gguf(gm), emb, h, np.arange(N), head_rows=draft.vocab)
    wr = gm.tensor("blk.40.ffn_gate_inp.weight").dequantize().numpy().reshape(gm.tensor("blk.40.ffn_gate_inp.weight").rows, -1)
    p = np.sort(ref.softmax(r["mtp_attn_post_norm"] @ wr.T), -1)[:, ::-1]
    tie, *tol = TOL[request.param]
    tied = (p[:, 7] - p[:, 8]) / p[:, 7] < tie  # (rows whose 8th and 9th experts nearly tie: either may win)
    return draft, torch.from_numpy(emb), torch.from_numpy(h), r, tied, tol


@pytest.mark.parametrize("S", [1, 3, 4])
def test_draft_steps(setup: tuple, S: int) -> None:
    """Rows within 1.5% rms of the oracle's h (median 0.5%) but where the router's 8th and 9th experts tie within 1%
    (Occamy's: ``TOL``); the head's argmax the oracle's on those rows where its top two logits are 0.1 apart."""
    draft, emb, h, r, tied, (med, top) = setup
    draft.reset()
    lg, hs = [], []
    for p in range(0, N, S):
        k = min(S, N - p)
        lg.append(draft.step(emb[p:p + k], h[p:p + k], p).float().numpy().copy())
        hs += [draft.hidden_state(s).numpy() for s in range(k)]
    lg, hs = np.concatenate(lg), np.stack(hs)
    want = r["h_nextn"].astype(np.float64)
    e = np.sqrt(((hs - want) ** 2).mean(-1) / (want ** 2).mean(-1))
    assert np.median(e) < med and e[~tied].max() < top, (np.median(e), e[~tied].max())
    top2 = np.sort(r["logits"], -1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0] > 0.1) & ~tied
    assert (lg.argmax(-1) == r["logits"].argmax(-1))[clear].all()


def test_interleaved(setup: tuple) -> None:
    """The app's order of dispatches: drafter steps between the model's (lazy steps of two tokens, one accepted) on
    the same hardware context; the drafter's logits bit for bit those of drafter steps alone."""
    draft, emb, h, *_ = setup
    target, n = draft.target, 12
    draft.reset()
    alone = [draft.step(emb[p:p + 1], h[p:p + 1], p).clone() for p in range(n)]
    draft.reset()
    target.reset()
    for p in range(n):
        target.step(emb[p:p + 2], p, logits=False, commit=False)
        target.accept(1)
        assert torch.equal(draft.step(emb[p:p + 1], h[p:p + 1], p), alone[p]), p


@pytest.mark.parametrize("size", ["35B-A3B", "occamy"])
def test_draft_prefill(size: str) -> None:
    """Three 256-token chunks of an essay through the slice's prefill with the drafter's records (the carry between
    chunks: the last row's h), then the drafter's step at the next position; the same step after drafter steps of 4
    rows wrote the records from the same inputs (the embeddings, the model's h of the prefill's output rows): within
    1% rms (h ~0.67%, logits ~0.69%: bfp16 vs bf16 roundings of the records' inputs), the same argmax. The last
    chunk's carry as the host computes it after decode steps (``carry_of``, fp32) within 2% of the device's (1.2%: its
    GEMM's bfp16 panels of hnorm(h))."""
    model, mtp = _files(size)
    T, nch = 256, 3
    g, gm = GGUFFile(model), GGUFFile(mtp)
    target = QwenOverlay(g, layers=[0, 3], vocab=4096, max_positions=4096, formats=draft_formats(gm))
    draft = QwenDraft(target, gm)
    pf = QwenPrefill(target, sizes=(T,), draft=draft)
    pf.compile()
    target.load()
    draft.load()
    pf.bind()
    n = T * nch
    emb = g.rows("token_embd.weight", ref.llama_tokenize(model, essay("apple.txt"))[:n + 1]).float()
    target.rope[:n + 4] = np.arange(n + 4)[:, None]
    xs = []
    for c in range(nch):
        pf.run(emb[c * T:(c + 1) * T], c * T, T)
        xs.append(pf.hidden(0, T))
    x = torch.cat(xs)
    h = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + target.cfg.rms_norm_eps) * torch.from_numpy(
        target._f32("output_norm.weight"))
    hp = torch.cat([torch.zeros(1, draft.H), h])  # (row q: the model's h of q - 1)
    dev_carry = torch.from_numpy(pf.dp.carry.copy()).view(torch.bfloat16).float()
    pf.dp.carry_of(h[n - 1])  # (the host's, after decode steps)
    host_carry = torch.from_numpy(pf.dp.carry.copy()).view(torch.bfloat16).float()
    assert float((dev_carry - host_carry).norm() / host_carry.norm()) < 2e-2
    a = draft.step(emb[n:n + 1], hp[n:n + 1], n).float().numpy().copy()
    ha = draft.hidden_state(0).numpy()
    draft.reset()
    for p in range(0, n, 4):
        draft.step(emb[p:p + 4], hp[p:p + 4], p)
    b = draft.step(emb[n:n + 1], hp[n:n + 1], n).float().numpy().copy()
    hb = draft.hidden_state(0).numpy()
    for u, v in ((ha, hb), (a, b)):
        assert np.sqrt(((u - v) ** 2).mean() / (v ** 2).mean()) < 1e-2
    assert a.argmax() == b.argmax()
