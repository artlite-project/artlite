"""References of the LFM3 tests: the files (``gguf``), token sequences (an essay's tokens, a chat turn with a fixed
reply), the fp32 CPU model's logits (``cpu.LfmCpu``, cached in ``tests/refs/<model>/``: ``common.cached_ref``) and
llama.cpp's CPU logits of a sequence (the golden dump tool: ``tests.goldens.text.golden`` runs any architecture)."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import d1_gguf, lfm3_gguf

from .. import common
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from . import cpu

SOURCES = [cpu.__file__]
PROMPT = "Explain how photosynthesis works in plants."
# a fixed reply (teacher forcing: the CPU and NPU models see the same tokens)
REPLY = ("Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical "
         "energy. It takes place in the chloroplasts, mainly in the leaves.")


def path(quant: str = "QAD-Q4_0") -> str:
    p = lfm3_gguf(quant)
    if not os.path.exists(p):
        pytest.skip(f"{p} missing")
    return p


def gguf(quant: str = "QAD-Q4_0") -> GGUFFile:
    return GGUFFile(path(quant))


def d1_path(quant: str = "Q4_0") -> str:
    """A d1-2B file (non-causal full layers: a decision model's prompts)."""
    p = d1_gguf(quant)
    if not os.path.exists(p):
        pytest.skip(f"{p} missing")
    return p


def essay_ids(n: int, quant: str = "QAD-Q4_0", name: str = "addiction.txt", *, path: str | None = None) -> list[int]:
    """The BOS and the first n - 1 tokens of an essay of the corpus (``path``: another LFM3 file's tokenizer)."""
    return [124894] + llama_tokenize(path or globals()["path"](quant), common.essay(name))[:n - 1]


def chat_ids(quant: str = "QAD-Q4_0") -> list[int]:
    """A chat turn (thinking off) and the fixed reply, as llama.cpp tokenizes them."""
    text = f"<|startoftext|><|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think></think>{REPLY}"
    return llama_tokenize(path(quant), text)


def cpu_logits(quant: str, ids: list[int], *, layers: list[int] | None = None, vocab: int | None = None,
               model: cpu.LfmCpu | None = None) -> torch.Tensor:
    """The fp32 CPU model's logits (len(ids), vocab) of a sequence from position 0 (cached)."""
    key = (common.model_key(path(quant)), tuple(ids), tuple(layers or ()), vocab)

    def compute() -> torch.Tensor:
        m = model or cpu.LfmCpu(gguf(quant), layers=layers, vocab=vocab)
        return m.logits(ids, slice(None)).half()

    return common.cached_ref(path(quant), "cpu_logits", key, compute, sources=SOURCES).float()


def llama_logits(quant: str, ids: list[int]) -> torch.Tensor:
    """llama.cpp's CPU logits (len(ids), vocab) of a sequence (its prompt graph)."""
    g = golden(path(quant), ids, logits="f32")
    return torch.from_numpy(np.asarray(g.logits, np.float32).copy())


