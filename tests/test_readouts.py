"""``NpuModel.outputs`` and ``readouts`` on the NPU (qwen35: MiniCPM-V 4.7's language model; Llama 3.2 1B; Qwen3-Embedding-0.6B, -4B): a
readout's rows are the rows the prompt path's LM head reads (a chunk's last row: the same logits through ``logits_of``,
bit for bit; any selection of rows: the same rows; a tail of decode steps: the step's own logits), and pooled embeddings
of them match llama.cpp's on the same file (``llama-embedding`` on the CPU, cached; Qwen3-Embedding's texts end with its
end token, as the file asks): cosine >= 0.999 for mean pooling, >= 0.995 for one row (last: llama.cpp's own two CPU
paths, ``llama-embedding`` and ``qwen35/dump.cpp``, are 0.998 apart on MiniCPM-V's row of a 30-token text; the NPU
0.996 - 0.999 on these files: the prefill's bfp16 weights and activations). Packed readouts (Qwen3-Embedding-0.6B,
Kev-4B, d3-lite): against each input alone."""

import os

import numpy as np
import pytest
import torch

from artlite.model_config import decision_gguf, llama_gguf, minicpmv_gguf, qwen3_gguf
from models import open_model
from models.common import Model, NpuModel
from models.readouts import Readouts, pool, rows_of

from .common import cached_ref, essay, model_key
from .llama_cpp import _lib_identity, llama_embeddings

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

CASES = [minicpmv_gguf(), llama_gguf("Q4_K_M", "1B"), qwen3_gguf("embedding-0.6B"), qwen3_gguf("embedding-4B")]


@pytest.fixture(scope="module", params=CASES, ids=os.path.basename)
def loaded(request: pytest.FixtureRequest) -> tuple[str, Model, NpuModel]:
    path = request.param
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    model = open_model(path)
    return path, model, model.load(capacity=8192)


def test_rows(loaded: tuple[str, Model, NpuModel]) -> None:
    path, model, npu = loaded
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
        want = npu.forward(toks)  # (its last logits: the LM head on the chunk's last row, or a decode step's)
        got = npu.overlay.logits_of(torch.from_numpy(rows[-1])).float()
        if npu.prefill.plan(0, n) == npu.prefill.plan(0, n, steps=False):
            assert torch.equal(got, want), n  # (the same row through the same LM head)
        else:  # (the prompt path's tail is a decode step, the readout's a chunk: two paths, test_rewind's KL bound;
            # Q8_0's weights: exact codes in the decode's GEMV, bfp16 panels in the prefill's GEMMs, KL 0.0095)
            lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
            bound = 0.012 if path == qwen3_gguf("embedding-0.6B") else 0.002
            assert got.argmax() == want.argmax() and (lw.exp() * (lw - lg)).sum() < bound, n


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_embeddings(loaded: tuple[str, Model, NpuModel], pooling: str) -> None:
    path, model, npu = loaded
    text = essay("addiction.txt")
    texts = [text[:120], text[500:1500], text[2000:6000], text[6000:12000]]
    ref = cached_ref(path, f"embeddings_{pooling}", (model_key(path), texts, pooling, _lib_identity()),
                     lambda: [torch.tensor(e) for e in llama_embeddings(path, texts, pooling)])
    readouts = Readouts.of(path)
    for t, want in zip(texts, ref, strict=True):
        ids = readouts.embedding_ids(model.encode, t)
        npu.reset()
        got = torch.from_numpy(pool(npu.outputs(ids, rows_of(pooling, len(ids))), pooling))
        cos = torch.nn.functional.cosine_similarity(got.double(), want.double(), 0).item()
        assert cos >= (0.999 if pooling == "mean" else 0.995), (len(ids), cos)


PACKED = [qwen3_gguf("embedding-0.6B"), decision_gguf("kev"), decision_gguf("d3-lite")]


@pytest.fixture(scope="module", params=PACKED, ids=os.path.basename)
def packed(request: pytest.FixtureRequest) -> tuple[str, Model, NpuModel]:
    path = request.param
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
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
