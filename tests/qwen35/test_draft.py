"""The multi-token-prediction drafter (``qwen35.draft.QwenDraft``) of Qwen3.6-35B-A3B (ggml-org's Q4_0 MTP file), of
Occamy 1.0 (its Q4_K_M MTP file: Q4_K and Q6_K) and of Qwen3.8-27B (its file's own block, on the model's image: its
head the first rows of the model's, the vocabulary whole) on the decode overlay of a 2-layer slice (layers 0 and 3: the
model's design, its formats with the drafter's), against ``ref.mtp_block`` on llama.cpp's hidden states of an essay
(golden(mtp=): the rows' h and the drafter's token embeddings): per row the next draft's h and the head's logits, in
steps of S rows from position 0. The prompt's records (``QwenPrefill(draft=)``, ``draft_prefill``) against those of
drafter steps from the same inputs."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_mtp_of
from models.qwen35.draft import TABLE_ROWS, QwenDraft, draft_formats
from models.qwen35.overlay import QwenOverlay
from models.qwen35.prefill import QwenPrefill

from ..common import essay
from . import ref

N = 100
# (per model: the router gap below which a row's 8th and 9th experts tie, h's median and largest error elsewhere;
# Occamy's drafter is Q4_K / Q6_K, whose decode tiles hold d and dmin as bf16: that alone moves h by 0.9% median; the
# 27B's block too: 0.84% median, two rows of the essay at 2.5-2.7%)
TOL = {"35B-A3B": (0.01, 7e-3, 1.5e-2), "occamy": (0.02, 1.5e-2, 2.5e-2), "3.8-27B": (0.0, 1.5e-2, 3e-2)}
# (designs of 8 slots: x as single bfp16 rows, ``decode.gemv``; the 27B's h 1.4% median, 4.3% at most)
TOL8 = {"3.8-27B": (0.0, 2e-2, 5e-2)}
SIZES = ["35B-A3B", "occamy", "3.8-27B"]
PREFILL_TOL = {"3.8-27B": (1.5e-2, 3e-2)}  # (test_draft_prefill's records and carry, where not 1% and 2%)
PREFILL_TOL8 = {"3.8-27B": (2.5e-2, 3e-2)}  # (designs of 8 slots: the steps' records 1.7% from the prefill's)


def _files(size: str) -> tuple[str, str]:
    """The model file of a size ("35B-A3B", "occamy", "3.8-27B") and its drafter's (the model's own, with a block)."""
    model = ref.size_gguf(size)
    if not os.path.exists(model):
        pytest.skip(f"no {model}")
    mtp = model if QwenConfig.from_gguf(GGUFFile(model)).nextn_layers else qwen_mtp_of(model)
    if mtp is None:
        pytest.skip(f"no MTP drafter of {model}")
    return model, mtp


def _slice(model: str, mtp: str, max_positions: int) -> tuple[QwenOverlay, QwenDraft]:
    """Layers 0 and 3 of the model and its drafter (an MTP file's: the vocabulary's first 4096 rows; the model's own
    block: on the model's image, its head the first rows of the model's whole one)."""
    if mtp == model:
        target = QwenOverlay(GGUFFile(model), layers=[0, 3], max_positions=max_positions)
        return target, QwenDraft(target)
    gm = GGUFFile(mtp)
    target = QwenOverlay(GGUFFile(model), layers=[0, 3], vocab=4096, max_positions=max_positions,
                         formats=draft_formats(gm))
    return target, QwenDraft(target, gm)


@pytest.fixture(scope="module", params=SIZES)
def setup(request: pytest.FixtureRequest) -> tuple:
    model, mtp = _files(request.param)
    gm = GGUFFile(mtp)
    cfg = QwenConfig.from_gguf(gm)
    L = cfg.num_hidden_layers
    tokens = ref.llama_tokenize(model, essay("apple.txt"))[:N]
    G = ref.golden(model, tokens, layers=(L, "g"), names="mtp_tok_embd|result_norm", logits="none", mtp=mtp)
    emb = np.asarray(G.mtp("mtp_tok_embd", L), np.float32).reshape(N, -1)
    hr = np.asarray(G["result_norm"], np.float32).reshape(N, -1)
    h = np.concatenate([np.zeros_like(hr[:1]), hr[:-1]])
    target, draft = _slice(model, mtp, 1024)
    target.load()
    draft.load()
    r = ref.mtp_block(gm, cfg, emb, h, np.arange(N), head=draft.ids)
    tie, *tol = (TOL8 if draft.SLOTS == 8 else TOL)[request.param]
    tied = np.zeros(N, bool)
    if cfg.moe:
        t = gm.tensor(f"blk.{L}.ffn_gate_inp.weight")
        p = np.sort(ref.softmax(r["mtp_attn_post_norm"] @ t.dequantize().numpy().reshape(t.rows, -1).T), -1)[:, ::-1]
        tied = (p[:, 7] - p[:, 8]) / p[:, 7] < tie  # (rows whose 8th and 9th experts nearly tie: either may win)
    return draft, torch.from_numpy(emb), torch.from_numpy(h), r, tied, tol


@pytest.mark.parametrize("S", [1, 3, 4, 8])
def test_draft_steps(setup: tuple, S: int) -> None:
    """Rows within 1.5% rms of the oracle's h (median 0.5%) but where the router's 8th and 9th experts tie within 1%
    (Occamy's: ``TOL``); the head's argmax the oracle's on those rows where its top two logits are 0.1 apart. S 8: the
    designs of 8 slots (the 27B's)."""
    draft, emb, h, r, tied, (med, top) = setup
    if S > draft.SLOTS:
        pytest.skip(f"a design of {draft.SLOTS} slots")
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


def test_table(setup: tuple) -> None:
    """The head's table (``QwenDraft.see``), on a drafter of its own (a model's own block: the module's, whose table has
    seen nothing yet; its rows are the model's buffer): the special tokens are fixed rows; after a
    sequence of more tokens past the fixed rows than the table holds, the table holds the ``TABLE_ROWS`` seen last (by
    their last occurrence), and the sequence seen again rewrites nothing; a step's logits of the table's tokens within
    2% rms of the oracle's (its h, the file's rows of those tokens) on the rows ``test_draft_steps`` checks (8 slots:
    4%)."""
    draft, emb, h, r, tied, _ = setup
    d = draft if draft.w_of is not None else QwenDraft(draft.target, draft.gguf)  # (the model's own block: its buffer)
    if d is not draft:
        d.load()
    types = np.asarray(d.gguf.meta["tokenizer.ggml.token_type"])
    assert np.isin(np.flatnonzero((types == 3) | (types == 4)), d.ids).all()
    rare = np.setdiff1d(np.arange(70000, 248000, 211), d.ids)[:TABLE_ROWS + 100]
    seq = [*rare[:300], 5, *rare[300:], *rare[:50]]
    d.see(seq)
    assert set(d.ids[d.table_rows].tolist()) == {*rare[:50].tolist(), *rare[150:].tolist()}
    ids = d.ids.copy()
    d.see(seq)
    assert np.array_equal(d.ids, ids)
    lg = np.concatenate([d.step(emb[p:p + 4], h[p:p + 4], p).float().numpy()[:, d.table_rows] for p in range(0, N, 4)])
    w = d.gguf.rows("output.weight", d.ids[d.table_rows]).numpy().astype(np.float64)
    want = r["h_nextn"].astype(np.float64) @ w.T
    e = np.sqrt(((lg - want) ** 2).mean(-1) / (want ** 2).mean(-1))
    assert e[~tied].max() < (2e-2 if d.SLOTS == 4 else 4e-2), (np.median(e), e[~tied].max())  # (8 slots: 3.6%)


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


@pytest.mark.parametrize("size", SIZES)
def test_draft_prefill(size: str) -> None:
    """Three 256-token chunks of an essay through the slice's prefill with the drafter's records (the carry between
    chunks: the last row's h), then the drafter's step at the next position; the same step after drafter steps of SLOTS
    rows wrote the records from the same inputs (the embeddings, the model's h of the prefill's output rows): within
    1% rms (h ~0.67%, logits ~0.69%: bfp16 vs bf16 roundings of the records' inputs), the same argmax. The last
    chunk's carry as the host computes it after decode steps (``carry_of``, fp32) within 2% of the device's (1.2%: its
    GEMM's bfp16 panels of hnorm(h)). The 27B (``PREFILL_TOL``): its hnorm(h) has a dimension at ~68 times its rms, whose
    bfp16 blocks cost 2.2% of the carry (2.4%), the records 1.1%."""
    model, mtp = _files(size)
    T, nch = 256, 3
    target, draft = _slice(model, mtp, 4096)
    rec, carry = (PREFILL_TOL8 if target.SLOTS == 8 else PREFILL_TOL).get(size, (1e-2, 2e-2))
    g = target.gguf
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
    assert float((dev_carry - host_carry).norm() / host_carry.norm()) < carry
    a = draft.step(emb[n:n + 1], hp[n:n + 1], n).float().numpy().copy()
    ha = draft.hidden_state(0).numpy()
    draft.reset()
    for p in range(0, n, draft.SLOTS):
        draft.step(emb[p:p + draft.SLOTS], hp[p:p + draft.SLOTS], p)
    b = draft.step(emb[n:n + 1], hp[n:n + 1], n).float().numpy().copy()
    hb = draft.hidden_state(0).numpy()
    for u, v in ((ha, hb), (a, b)):
        assert np.sqrt(((u - v) ** 2).mean() / (v ** 2).mean()) < rec
    assert a.argmax() == b.argmax()
