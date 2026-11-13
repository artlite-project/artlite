"""The multi-token-prediction drafter (``qwen35.draft.QwenDraft``) of Qwen3.6-35B-A3B (ggml-org's Q4_0 MTP file, and its
Q8_0 one: gate / up tiles of 8 rows, row-group downs), of
Occamy 1.0 (its Q4_K_M MTP file: Q4_K and Q6_K) and of Qwen3.8-27B (its file's own block, on the model's image: its
head the first rows of the model's, the vocabulary whole) on the decode overlay of a 2-layer slice (layers 0 and 3: the
model's design, its formats with the drafter's), against ``ref.mtp_block`` on llama.cpp's hidden states of an essay
(golden(mtp=): the rows' h and the drafter's token embeddings): per row the next draft's h and the head's logits, in
steps of S rows from position 0. The prompt's records (``QwenPrefill(draft=)``, ``draft_prefill``) against those of
drafter steps from the same inputs."""


import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_mtp_of
from models.qwen35.draft import TABLE_ROWS, QwenDraft, draft_formats, draft_gate_up
from models.qwen35.overlay import QwenOverlay
from models.qwen35.prefill import QwenPrefill

from .. import np_ops
from ..common import essay
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require, size_gguf
from . import ref

N = 100
# (per model: the router gap below which a row's 8th and 9th experts may trade places, h's median and largest error;
# a row within that gap is checked against the oracle that routes it as the NPU did: either expert (the router's input
# is ~0.7% from the oracle's: the 35B-A3B Q8_0 drafter's row 64 of the essay, a 2.1% gap, takes the 9th, its h 1.0%
# from that oracle and 11% from the other). Occamy's drafter is Q4_K / Q6_K, whose decode tiles hold d and dmin as
# bf16: that alone moves h by 0.9% median; the 27B's block too: 0.84% median, two rows of the essay at 2.5-2.7%. The
# K records hold K as single bfp16: the 35B-A3B's h 0.72-0.74% median against the oracle, 0.68-0.70% against one with
# K so rounded)
TOL = {"35B-A3B": (0.03, 7.5e-3, 2e-2), "35B-A3B-Q8_0": (0.03, 7.5e-3, 2e-2), "occamy": (0.03, 1.5e-2, 3e-2),
       "3.8-27B": (0.0, 1.5e-2, 3e-2)}
# (designs of 8 slots: x as single bfp16 rows, ``decode.gemv``; the 27B's h 1.4% median, 4.3% at most)
TOL8 = {"3.8-27B": (0.0, 2e-2, 5e-2)}
SIZES = ["35B-A3B", "35B-A3B-Q8_0", "occamy", "3.8-27B"]
PREFILL_TOL = {"3.8-27B": (1.5e-2, 3e-2)}  # (test_draft_prefill's records and carry, where not 1% and 2%)
PREFILL_TOL8 = {"3.8-27B": (2.5e-2, 3e-2)}  # (designs of 8 slots: the steps' records 1.7% from the prefill's)


def _files(size: str) -> tuple[str, str]:
    """The model file of a size ("35B-A3B", "occamy", "3.8-27B"; "35B-A3B-Q8_0": its Q8_0 drafter) and its drafter's
    (the model's own, with a block)."""
    size, _, quant = size.partition("-Q")
    model = size_gguf(size)
    require(model)
    own = QwenConfig.from_gguf(GGUFFile(model)).nextn_layers
    mtp = model if own else qwen_mtp_of(model, "Q" + quant if quant else "")
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
                         formats=draft_formats(gm), gu=draft_gate_up(gm))
    return target, QwenDraft(target, gm)


@pytest.fixture(scope="module", params=SIZES)
def setup(request: pytest.FixtureRequest) -> tuple:
    model, mtp = _files(request.param)
    gm = GGUFFile(mtp)
    cfg = QwenConfig.from_gguf(gm)
    L = cfg.num_hidden_layers
    tokens = llama_tokenize(model, essay("apple.txt"))[:N]
    G = golden(model, tokens, layers=(L, "g"), names="mtp_tok_embd|result_norm", logits="none", mtp=mtp)
    emb = np.asarray(G.mtp("mtp_tok_embd", L), np.float32).reshape(N, -1)
    hr = np.asarray(G["result_norm"], np.float32).reshape(N, -1)
    h = np.concatenate([np.zeros_like(hr[:1]), hr[:-1]])
    target, draft = _slice(model, mtp, 1024)
    target.load()
    draft.load()
    r = ref.mtp_block(gm, cfg, emb, h, np.arange(N), head=draft.table.ids)
    tie, *tol = (TOL8 if draft.SLOTS == 8 else TOL)[request.param]
    refs = [r]
    if cfg.moe:  # (rows whose 8th and 9th experts nearly tie: either may win, the other oracle routes them with the 9th)
        t = gm.tensor(f"blk.{L}.ffn_gate_inp.weight")
        p = np.sort(np_ops.softmax(r["mtp_attn_post_norm"] @ t.dequantize().reshape(t.rows, -1).T), -1)[:, ::-1]
        swap = (p[:, 7] - p[:, 8]) / p[:, 7] < tie
        refs.append(ref.mtp_block(gm, cfg, emb, h, np.arange(N), head=draft.table.ids, swap=swap))
    return draft, emb, h, refs, tol


def _nearest(got: np.ndarray, wants: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per row the rms error of got relative to the nearer of the oracles' wants (the routing the NPU took), and which."""
    e = np.stack([np.sqrt(((got - w) ** 2).mean(-1) / (w ** 2).mean(-1)) for w in wants])
    return e.min(0), e.argmin(0)


@pytest.mark.parametrize("S", [1, 3, 4, 8])
def test_draft_steps(setup: tuple, S: int) -> None:
    """Rows within ``TOL`` of the oracle's h (rows whose router's 8th and 9th experts nearly tie: of the oracle that
    routes them as the NPU did); the head's argmax that oracle's on those rows where its top two logits are 0.1 apart.
    S 8: the designs of 8 slots (the 27B's)."""
    draft, emb, h, refs, (med, top) = setup
    if S > draft.SLOTS:
        pytest.skip(f"a design of {draft.SLOTS} slots")
    draft.reset()
    lg, hs = [], []
    for p in range(0, N, S):
        k = min(S, N - p)
        lg.append(draft.step(emb[p:p + k], h[p:p + k], p).astype(np.float32))
        hs += [draft.hidden_state(s) for s in range(k)]
    lg, hs = np.concatenate(lg), np.stack(hs)
    e, which = _nearest(hs, [r["h_nextn"].astype(np.float64) for r in refs])
    assert np.median(e) < med and e.max() < top, (np.median(e), e.max(), int(e.argmax()))
    want = np.choose(which[:, None], [r["logits"] for r in refs])
    top2 = np.sort(want, -1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > 0.1
    assert (lg.argmax(-1) == want.argmax(-1))[clear].all()


def test_table(setup: tuple) -> None:
    """The head's table (``HeadTable.see``), on a drafter of its own (a model's own block: the module's, whose table has
    seen nothing yet; its rows are the model's buffer): the special tokens are fixed rows; after a
    sequence of more tokens past the fixed rows than the table holds, the table holds the ``TABLE_ROWS`` seen last (by
    their last occurrence), and the sequence seen again rewrites nothing; a step's logits of the table's tokens within
    2.5% rms of the oracle's (its h, the file's rows of those tokens; a near tie's row: of the oracle that routes it as
    the NPU did, as ``test_draft_steps``) (8 slots: 4%; Occamy: its h bound, 3%)."""
    draft, emb, h, refs, (_, top) = setup
    d = draft if draft.w_of is not None else QwenDraft(draft.target, draft.gguf)  # (the model's own block: its buffer)
    if d is not draft:
        d.load()
    t = d.table
    types = np.asarray(d.gguf.meta["tokenizer.ggml.token_type"])
    assert np.isin(np.flatnonzero((types == 3) | (types == 4)), t.ids).all()
    rare = np.setdiff1d(np.arange(70000, 248000, 211), t.ids)[:TABLE_ROWS + 100]
    seq = [*rare[:300], 5, *rare[300:], *rare[:50]]
    t.see(seq)
    assert set(t.ids[t.rows].tolist()) == {*rare[:50].tolist(), *rare[150:].tolist()}
    ids = t.ids.copy()
    t.see(seq)
    assert np.array_equal(t.ids, ids)
    lg = np.concatenate([d.step(emb[p:p + 4], h[p:p + 4], p).astype(np.float32)[:, t.rows] for p in range(0, N, 4)])
    w = d.gguf.rows("output.weight", t.ids[t.rows]).astype(np.float64)
    e, _ = _nearest(lg, [r["h_nextn"].astype(np.float64) @ w.T for r in refs])
    # (8 slots: 3.6%; the h bound of ``TOL`` where larger: Occamy's row 4, a near tie routed as the oracle, h 2.1% off, its
    # logits 2.7%)
    assert e.max() < (max(2.5e-2, top) if d.SLOTS == 4 else 4e-2), (np.median(e), e.max())


def test_interleaved(setup: tuple) -> None:
    """The app's order of dispatches: drafter steps between the model's (lazy steps of two tokens, one accepted) on
    the same hardware context; the drafter's logits bit for bit those of drafter steps alone."""
    draft, emb, h, *_ = setup
    target, n = draft.target, 12
    draft.reset()
    alone = [draft.step(emb[p:p + 1], h[p:p + 1], p).copy() for p in range(n)]
    draft.reset()
    target.reset()
    for p in range(n):
        target.step(emb[p:p + 2], p, logits=False, commit=False)
        target.accept(1)
        assert np.array_equal(draft.step(emb[p:p + 1], h[p:p + 1], p).view(np.uint16), alone[p].view(np.uint16)), p


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
    emb = g.rows("token_embd.weight", llama_tokenize(model, essay("apple.txt"))[:n + 1])
    target.rope[:n + 4] = np.arange(n + 4)[:, None]
    xs = []
    for c in range(nch):
        pf.run(emb[c * T:(c + 1) * T], c * T, T)
        xs.append(pf.hidden(0, T))
    x = np.concatenate(xs)
    h = x / np.sqrt((x * x).mean(-1, keepdims=True) + target.cfg.rms_norm_eps) * target._f32("output_norm.weight")
    hp = np.concatenate([np.zeros((1, draft.H), np.float32), h])  # (row q: the model's h of q - 1)
    dev_carry = pf.dp.carry.view(bfloat16).astype(np.float32)
    pf.dp.carry_of(h[n - 1])  # (the host's, after decode steps)
    host_carry = pf.dp.carry.view(bfloat16).astype(np.float32)
    assert float(np.linalg.norm(dev_carry - host_carry) / np.linalg.norm(host_carry)) < carry
    a = draft.step(emb[n:n + 1], hp[n:n + 1], n).astype(np.float32)
    ha = draft.hidden_state(0)
    draft.reset()
    for p in range(0, n, draft.SLOTS):
        draft.step(emb[p:p + draft.SLOTS], hp[p:p + draft.SLOTS], p)
    b = draft.step(emb[n:n + 1], hp[n:n + 1], n).astype(np.float32)
    hb = draft.hidden_state(0)
    for u, v in ((ha, hb), (a, b)):
        assert np.sqrt(((u - v) ** 2).mean() / (v ** 2).mean()) < rec
    assert a.argmax() == b.argmax()
