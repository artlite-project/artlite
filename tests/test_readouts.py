"""``NpuModel.outputs`` on the NPU (qwen35: MiniCPM-V 4.7's language model; Llama 3.2 1B; Qwen3-Embedding-0.6B, -4B): a
readout's rows are the rows the prompt path's LM head reads (a chunk's last row: the same logits through ``logits_of``,
bit for bit; any selection of rows: the same rows; a tail of decode steps: the step's own logits), and pooled embeddings
of them match llama.cpp's on the same file (``llama-embedding`` on the CPU, cached; Qwen3-Embedding's texts end with its
end token, as the file asks): cosine >= 0.999 for mean pooling, >= 0.995 for one row (last: llama.cpp's own two CPU
paths, ``llama-embedding`` and ``qwen35/dump.cpp``, are 0.998 apart on MiniCPM-V's row of a 30-token text; the NPU
0.996 - 0.999 on these files: the prefill's bfp16 weights and activations)."""

import os

import numpy as np
import pytest
import torch

from artlite.model_config import llama_gguf, minicpmv_gguf, qwen3_gguf
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
