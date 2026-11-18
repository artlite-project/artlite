"""``NpuModel.outputs`` and ``readouts`` on the NPU (qwen35: MiniCPM-V 4.7's language model; Llama 3.2 1B; Qwen3-Embedding-0.6B;
LFM3): a readout's rows are the rows the prompt path's LM head reads (a chunk's last row: the same logits through ``logits_of``,
bit for bit; any selection of rows: the same rows; a tail of decode steps: the step's own logits; a readout model's
image holds no LM head: its rows alone, no logits), and pooled embeddings
of them match llama.cpp's on the same file (``llama-embedding`` on the CPU, cached; Qwen3-Embedding's texts end with its
end token, as the file asks): cosine >= 0.999 for mean pooling, >= 0.995 for one row (last: llama.cpp's own two CPU
paths, ``llama-embedding`` and ``goldens/qwen35_dump.cpp``, are 0.998 apart on MiniCPM-V's row of a 30-token text; the NPU
0.996 - 0.999 on these files: the prefill's bfp16 weights and activations). Packed readouts (Qwen3-Embedding-0.6B,
Kev-4B, d3-lite): against each input alone. Rerank (Qwen3-Reranker-0.6B, -4B): against llama-server's scores."""

import os

import numpy as np
import pytest
import torch

from artlite.model_config import decision_gguf, lfm3_gguf, llama_gguf, minicpmv_gguf, qwen3_gguf
from models import open_model
from models.model import Model
from models.npu_model import NpuModel
from models.readouts import Readouts, pool, rows_of

from .common import cached_ref, essay, essays, model_key
from .llama_cpp import lib_identity, llama_embeddings, llama_server_requests
from .model_files import require
from .numerics import kl_rows

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

CASES = [minicpmv_gguf(), llama_gguf("Q4_K_M", "1B"), qwen3_gguf("embedding-0.6B"),
         lfm3_gguf("QAD-Q4_0")]


@pytest.fixture(scope="module", params=CASES, ids=os.path.basename)
def loaded(request: pytest.FixtureRequest) -> tuple[str, Model, NpuModel]:
    path = request.param
    require(path)
    model = open_model(path)
    return path, model, model.load(capacity=8192)


def test_rows(loaded: tuple[str, Model, NpuModel]) -> None:
    _, model, npu = loaded
    ids = model.encode(essay("addiction.txt") * 3)
    big = npu.prefill.sizes[-1]
    for n in (64, 99, big + 1, big + 2):  # (a chunk, a padded one; full chunks and a tail of 1-2 tokens)
        toks = ids[:n]
        npu.reset()
        rows = npu.outputs(toks, [0, n // 2, -1], norm=False)
        npu.reset()
        every = npu.outputs(toks, None, norm=False)
        assert np.array_equal(rows, every[[0, n // 2, n - 1]]), n
        assert np.isfinite(every).all()
        npu.reset()
        if not npu.generates:  # (a readout model's: no LM head, ``models.readouts.READOUT_VOCAB``)
            with pytest.raises(NotImplementedError):
                npu.forward(toks)
            continue
        want = torch.from_numpy(npu.forward(toks))  # (its last logits: the LM head on the chunk's last row, or a decode step's)
        got = torch.from_numpy(npu.cap(npu.overlay.logits_of(rows[-1])))  # (a screened head's as ``forward``'s)
        plan = npu.prefill.plan(0, n)
        # (a post-norm model's step row is its LM header's updated residual in bf16, not the header's sum: LFM3's)
        if plan == npu.prefill.plan(0, n, steps=False) and (sum(plan) >= n or not getattr(npu.overlay.op, "post", 0)):
            assert torch.equal(got, want), n  # (the same row through the same LM head)
        else:  # (the prompt path's tail is a decode step, the readout's a chunk: two paths, test_rewind's KL bound)
            assert got.argmax() == want.argmax() and kl_rows(want, got) < 0.002, n


# (mean pooling's bound where it is not 0.999: LFM3's rows 0.9990 - 0.9998 from the fp32 CPU model's, llama.cpp's 0.9997 -
# 0.99997, 0.9986 - 0.9997 from each other; one row 0.9967 - 0.9991 from the fp32 model's: the prefill's 8-bit GEMM
# operands)
MEAN = {lfm3_gguf("QAD-Q4_0"): 0.998}


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_embeddings(loaded: tuple[str, Model, NpuModel], pooling: str) -> None:
    path, model, npu = loaded
    text = essay("addiction.txt")
    texts = [text[:120], text[500:1500], text[2000:6000], text[6000:12000]]
    ref = cached_ref(path, f"embeddings_{pooling}", (model_key(path), texts, pooling),
                     lambda: [torch.tensor(e) for e in llama_embeddings(path, texts, pooling)], version=lib_identity())
    readouts = Readouts.of(path)
    for t, want in zip(texts, ref, strict=True):
        ids = readouts.embedding_ids(model.encode, t)
        npu.reset()
        got = torch.from_numpy(pool(npu.outputs(ids, rows_of(pooling, len(ids))), pooling))
        cos = torch.nn.functional.cosine_similarity(got.double(), want.double(), 0).item()
        assert cos >= (MEAN.get(path, 0.999) if pooling == "mean" else 0.995), (len(ids), cos)


PACKED = [qwen3_gguf("embedding-0.6B"), decision_gguf("kev"), decision_gguf("d3-lite")]


@pytest.fixture(scope="module", params=PACKED, ids=os.path.basename)
def packed(request: pytest.FixtureRequest) -> tuple[str, Model, NpuModel]:
    path = request.param
    require(path)
    model = open_model(path)
    return path, model, model.load(capacity=8192)


def test_readouts_packed(packed: tuple[str, Model, NpuModel]) -> None:
    """``NpuModel.readouts`` through packed chunks (a readout model's: the Llama path's, qwen35's from position 0 and,
    for a causal model, its tree chunks after a prefix) against each input alone (``NpuModel.readouts``' default: one
    after another): rows at any indices (negative: from an input's end) bit for bit in a tree chunk (alone's
    arithmetic: each segment's padding as alone's), else close (cosine 0.997 at least, 0.999 the median: the Llama
    path's chunk sizes differ in their GEMMs' K parts and norm splits, a non-causal chunk in its groups, and segments
    there share V blocks; the NPU's own distance from llama.cpp); and the model left at the prefix: a prompt after it
    runs as after the prefix alone."""
    path, model, npu = packed
    ids = model.encode(essay("addiction.txt") * 3)
    lengths = [20, 77, 130, 45, 300, 3]
    starts = np.cumsum([0, *lengths])[:-1]
    p = 64 if npu.causal else 0
    inputs = [ids[p + 2000 + a:p + 2000 + a + n] for a, n in zip(starts, lengths, strict=True)]
    at = [np.array([0, -1, n // 2, -2 % n]) for n in lengths]
    plan = npu.prefill.pack(lengths, p)
    assert plan.bins, (path, plan)

    def run(fn: object) -> tuple[list[np.ndarray], np.ndarray]:
        npu.reset()
        if p:
            npu.outputs(ids[:p], [-1])
        out = fn(inputs, at)
        assert npu.position == p
        after = npu.outputs(ids[p:p + 40], [-1])  # (the prefix's sequence continues)
        return out, after

    alone, after_alone = run(lambda x, r: NpuModel.readouts(npu, x, r))
    got, after = run(npu.readouts)
    same, cos = 0, []
    for g, w in zip(got, alone, strict=True):
        assert g.shape == (4, model.cfg.hidden_size)
        cos += list((g * w).sum(-1) / np.linalg.norm(g, axis=-1) / np.linalg.norm(w, axis=-1))
        same += int((g == w).all())
    print(f"{os.path.basename(path)}: plan {[(T, len(b)) for T, b in plan.bins]} alone {plan.alone}; {same} of "
          f"{len(lengths)} inputs bit-identical to alone, cosine min {min(cos):.5f} median {np.median(cos):.5f}")
    if npu.recurrent and p:
        assert same == len(lengths)
    assert min(cos) > 0.997 and np.median(cos) > 0.999
    assert np.array_equal(after, after_alone)


def test_readouts_lead(packed: tuple[str, Model, NpuModel]) -> None:
    """``NpuModel.readouts(lead=)`` (a prompt every input continues, not cached yet: the Llama path's packed chunk runs it
    as its first segment, others run it first) against the prompt run first: rows close (cosine 0.999 at least: the
    chunk's sizes differ; bit for bit where the lead runs first), the model left at the prompt's end."""
    path, model, npu = packed
    if not npu.causal:
        pytest.skip("a non-causal model reads every prompt whole")
    ids = model.encode(essay("addiction.txt") * 3)
    g = npu.group
    lengths = [20, 77, 130, 45, 300, 3]
    starts = np.cumsum([0, *lengths])[:-1]
    inputs = [ids[g + 2000 + a:g + 2000 + a + n] for a, n in zip(starts, lengths, strict=True)]
    at = [np.array([0, -1]) for _ in lengths]
    npu.reset()
    npu.outputs(ids[:g], [-1])
    want = npu.readouts(inputs, at)
    after_want = npu.outputs(ids[g:g + 40], [-1])
    npu.reset()
    got = npu.readouts(inputs, at, lead=ids[:g])
    assert npu.position == g
    after = npu.outputs(ids[g:g + 40], [-1])
    cos = [float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)) for x, y in zip(got, want, strict=True)
           for a, b in zip(x, y, strict=True)]
    print(f"{os.path.basename(path)}: lead inside {npu.lead_tokens == g}; cosine min {min(cos):.5f}")
    if npu.lead_tokens != g:
        assert all(np.array_equal(x, y) for x, y in zip(got, want, strict=True))
    assert min(cos) > 0.999
    assert np.array_equal(after, after_want)



RERANKERS = [qwen3_gguf("reranker-0.6B"), qwen3_gguf("reranker-4B")]


@pytest.mark.parametrize("path", RERANKERS, ids=os.path.basename)
def test_rerank(path: str) -> None:
    """A reranker's scores (``Readouts``: its classifier head on the last row of each pair's rerank prompt; 4B: half
    regions, no LM head) of 8 essay passages for a query, packed (``NpuModel.readouts``), against llama-server's
    ``/v1/rerank`` on the CPU (cached): the logits within 1.5 of its (the NPU 0.9 - 1.2 at most over 600 pairs of 30
    queries), the same best passage and the order alike (Spearman >= 0.9)."""
    require(path)
    model = open_model(path)
    assert not model.generates
    readouts = Readouts.of(path)
    text = essays()
    docs = [text[i:i + 1200] for i in range(0, 8 * 9000, 9000)]
    query = "What does the author say about how startups should treat their first users?"
    body = {"query": query, "documents": docs}
    ref = cached_ref(path, "rerank", (model_key(path), body),
                     lambda: llama_server_requests(path, "/v1/rerank", [body], ("--rerank",))[0], version=lib_identity())
    want = np.zeros(len(docs))
    for e in ref["results"]:
        want[e["index"]] = e["relevance_score"]
    npu = model.load(capacity=8192)
    inputs = [readouts.rerank_ids(model.tokenizer.encode, model.encode(""), query, d) for d in docs]
    rows = npu.readouts(inputs, [rows_of("rank", len(ids)) for ids in inputs])
    got = np.array([readouts.head.score(r[0]) for r in rows])

    def logit(s: np.ndarray) -> np.ndarray:
        s = np.clip(s, 1e-9, 1 - 1e-9)
        return np.log(s / (1 - s))

    d = np.abs(logit(got) - logit(want))
    rank = (np.argsort(np.argsort(got)), np.argsort(np.argsort(want)))
    rho = float(np.corrcoef(*rank)[0, 1])
    print(f"{os.path.basename(path)}: scores {np.round(got, 3)} vs {np.round(want, 3)}; |dlogit| max {d.max():.3f}, "
          f"Spearman {rho:.3f}")
    assert d.max() < 1.5 and got.argmax() == want.argmax() and rho >= 0.9
