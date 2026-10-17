"""Llama on the NPU against the fp32 CPU model of the same GGUF weights (models.llama3, tests/rtc/llama_cpu.py),
teacher-forced: KL(same) of the next-token distributions measures the implementation error alone. A chat prompt runs
through the prefill (one chunk), then its continuation one token and four tokens per dispatch, in both K formats of the
KV cache; a 4800-token essay prompt (prefill chunks) and 32 tokens after it."""

import os

import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.common import auto_capacity
from models.llama3.llama import Llama, NpuLlama
from models.llama3.tokenizer import chat_prompt

from . import llama_cpu
from .common import cached_ref, essays, llama_gguf, model_key

pytestmark = pytest.mark.slow  # GGUF models + CPU references: run with --run-slow

PROMPT = "Explain how photosynthesis works in plants, step by step."
DATE = "26 Jul 2024"  # a fixed system-header date: today's would change the CPU references' cache key daily
# a fixed continuation (teacher forcing): the CPU and NPU models see the same tokens
TEXT = ("Photosynthesis is the process by which plants, algae and some bacteria convert light energy into "
        "chemical energy. It takes place in the chloroplasts, mainly in the leaves.")
FILES = [("Q4_K_M", "1B"), ("Q5_K_M", "1B"), ("Q6_K", "1B"), ("Q4_0", "1B"), ("Q4_1", "1B"), ("Q3_K_L", "1B"),
         ("BF16", "1B"), ("Q4_K_M", "3B"), ("Q5_K_M", "3B"), ("Q2_K", "3B"), ("Q3_K_M", "3B"), ("Q4_K_M", "8B"),
         ("Q4_K_S", "8B")]


def _path(quant: str, size: str) -> str:
    path = llama_gguf(quant, size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    return path


def _cpu_logits(path: str, seq: list[int], first: int) -> torch.Tensor:
    """CPU fp32 logits of positions first .. len(seq) - 1 of seq (cached in tests/rtc/refs)."""
    return cached_ref(path, "cpu_logits", (model_key(path), seq, first),
                      lambda: llama_cpu.LlamaCpu(GGUFFile(path)).logits(seq, slice(first, None)),
                      sources=[llama_cpu.__file__])


def _kl_agree(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    lr, lp = torch.log_softmax(ref.double(), -1), torch.log_softmax(got.double(), -1)
    return float((lr.exp() * (lr - lp)).sum(-1).mean()), float((ref.argmax(-1) == got.argmax(-1)).float().mean())


@pytest.mark.parametrize("kv_k", ["single", "hilo"])
@pytest.mark.parametrize(("quant", "size"), FILES)
def test_llama_npu_teacher_forced(quant: str, size: str, kv_k: str) -> None:
    path = _path(quant, size)
    model = Llama(path, kv_k=kv_k)
    prompt = model.tokenizer.encode(chat_prompt(model.gguf, [{"role": "user", "content": PROMPT}], date=DATE))
    seq = prompt + model.tokenizer.encode(TEXT)
    first = len(prompt) - 1  # (the prompt's tokens before its last run without logits)
    want = _cpu_logits(path, seq, first)
    npu = model.load(capacity=256)
    for batch in (1, 4):
        got = npu.forced_logits(seq, first, batch=batch)
        kl, agree = _kl_agree(want, got)
        print(f"{size} {quant} {kv_k}, {batch} per dispatch: KL(same) {kl:.5f}, argmax agreement {agree:.3f}")
        assert kl < 0.006 and agree > 0.9, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}"


@pytest.mark.parametrize(("quant", "size"), [("Q4_K_M", "1B"), ("Q4_K_M", "3B"), ("Q4_K_M", "8B")])
def test_llama_npu_long_prompt(quant: str, size: str) -> None:
    """A 4800-token essay prompt (prefill chunks) and 32 tokens after it, one per dispatch."""
    path = _path(quant, size)
    P, N = 4800, 32
    model = Llama(path)
    ids = model.encode(essays())[:P + N]
    want = _cpu_logits(path, ids, P - 1)
    npu: NpuLlama = model.load(capacity=auto_capacity(P + N))
    got = npu.forced_logits(ids, P - 1, batch=1)
    kl, agree = _kl_agree(want, got)
    print(f"{size} {quant} at {P}: KL(same) {kl:.5f}, argmax agreement {agree:.3f}")
    assert kl < 0.006 and agree > 0.85, f"KL(same) {kl:.5f}, argmax agreement {agree:.3f}"
