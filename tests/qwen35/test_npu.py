"""qwen35 models on the NPU (models.qwen35, models/qwen35/overlay.py) against llama.cpp's decode
(ubatch 1) of the same GGUF file (golden tensors and logits, tests/qwen35/ref.py):

- a slice of four layers in the middle of the model (GDN layers and a full-attention layer), fed llama.cpp's residual
  stream before it token by token: the residual after every layer;
- the whole model, teacher-forced over an essay: KL of the next-token distributions and argmax agreement against
  llama.cpp's decode, within llama.cpp's own noise: it quantizes the activations of every projection to Q8_K, and its
  prompt (one graph) and decode runs of the text differ by a KL of 0.0026 (9B) / 0.051 (27B: a few tokens of this
  text near a tie flip, in both runs and on the NPU).
"""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.qwen35.qwen import NpuQwen

from ..common import essay
from . import ref

pytestmark = pytest.mark.slow  # whole GGUF models: run with --run-slow

def _path(size: str) -> str:
    path = ref.size_gguf(size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    return path


def _tokens(path: str, n: int, name: str = "founders.txt") -> list[int]:
    return ref.llama_tokenize(path, essay(name))[:n]


@pytest.mark.parametrize("size", ["9B", "27B", "1B"])
def test_slice(size: str) -> None:
    from models.qwen35.overlay import QwenOverlay

    path = _path(size)
    g = GGUFFile(path)
    cfg = ref.QwenConfig.from_gguf(g)
    La = cfg.attention_layers[len(cfg.attention_layers) // 2]
    layers = list(range(La - 3, La + 1))
    n = 34
    tokens = _tokens(path, n, "apple.txt")
    G = ref.golden(path, tokens, ubatch=1, layers=(layers[0] - 1, *layers), names="l_out", logits="none")
    ov = QwenOverlay(g, layers=layers, vocab=512, max_positions=64)
    ov.load(release=False)
    errs = np.zeros((n, len(layers)))
    for t in range(n):
        ov.step(torch.from_numpy(np.asarray(G["l_out", layers[0] - 1][t], np.float32).copy()), t, logits=False)
        for i, L in enumerate(layers):
            got = ov.hidden(layers[i + 1] if i + 1 < len(layers) else -1).numpy()
            errs[t, i] = ref.rel_err(got, np.asarray(G["l_out", L][t], np.float32))
    print(f"{size} layers {layers}: mean error per layer {errs.mean(0)}, max {errs.max():.3f}")
    assert errs.max() < 4e-2 and errs.mean(0).max() < 1.5e-2


def _kl(want: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    return (lw.exp() * (lw - lg)).sum(-1)


@pytest.mark.parametrize("size", ["9B", "27B", "1B", "35B-A3B", "occamy"])
def test_teacher_forced(size: str) -> None:
    path = _path(size)
    tokens = _tokens(path, 128)
    dec = torch.from_numpy(np.asarray(ref.golden(path, tokens, ubatch=1, logits="f32").logits, np.float32).copy())
    pro = torch.from_numpy(np.asarray(ref.golden(path, tokens, ubatch=0, logits="f32").logits, np.float32).copy())
    model = NpuQwen(GGUFFile(path), capacity=256)
    got = model.forced_logits(tokens)
    kl, kl_llama = float(_kl(dec, got).mean()), float(_kl(dec, pro).mean())
    # (an MoE model's routing flips near-tied experts, and then the next token where its top two are near ties: either
    # side's rounding moves a few of the 128 (Occamy's NPU 12, all where llama.cpp's own top two lie within 0.85); its
    # agreement counts the tokens whose top two llama.cpp's decode has at least 1 apart)
    top2 = torch.topk(dec, 2).values
    sure = top2[:, 0] - top2[:, 1] >= 1 if model.cfg.moe else torch.ones(len(dec), dtype=torch.bool)
    agree = float((dec.argmax(-1) == got.argmax(-1))[sure].float().mean())
    agree_llama = float((dec.argmax(-1) == pro.argmax(-1))[sure].float().mean())
    print(f"{size}: KL {kl:.4f} (llama.cpp prompt vs decode {kl_llama:.4f}), argmax agreement {agree:.3f} "
          f"({agree_llama:.3f}) over {int(sure.sum())} of {len(tokens)} tokens")
    tol = 0.05 if model.cfg.moe else 0.02
    assert kl < 1.5 * kl_llama and agree > agree_llama - tol


@pytest.mark.parametrize("size", ["9B", "27B", "1B", "35B-A3B", "occamy"])
def test_prefill(size: str) -> None:
    """The app's prompt path: 300 tokens through the prefill (one chunk of 512 tokens, 212 of them padding), the
    last one's logits from the LM head alone, then teacher-forced decode steps continuing from the prefill's caches
    and states; the next-token distributions against llama.cpp's split run (the prompt in one graph, then one token per
    graph), next to llama.cpp's own decode-only run against it."""
    path = _path(size)
    P, K = 300, 24
    tokens = _tokens(path, P + K)
    split = torch.from_numpy(np.asarray(ref.golden(path, tokens, split=P, logits="f32").logits, np.float32)[P - 1:P + K - 1].copy())
    dec = torch.from_numpy(np.asarray(ref.golden(path, tokens, ubatch=1, logits="f32").logits, np.float32)[P - 1:P + K - 1].copy())
    model = NpuQwen(GGUFFile(path), capacity=4096)
    got = torch.stack([model.forward(tokens[:P])] + [model.step(t) for t in tokens[P:P + K - 1]])
    kl, kl_llama = _kl(split, got), _kl(split, dec)
    agree = float((split.argmax(-1) == got.argmax(-1)).float().mean())
    agree_llama = float((split.argmax(-1) == dec.argmax(-1)).float().mean())
    print(f"{size}: KL vs llama.cpp's split run: prompt's last token {float(kl[0]):.4f} (llama.cpp decode-only "
          f"{float(kl_llama[0]):.4f}), mean over {K} positions {float(kl.mean()):.4f} ({float(kl_llama.mean()):.4f}); "
          f"argmax agreement {agree:.3f} ({agree_llama:.3f})")
    assert float(kl.mean()) < 1.5 * float(kl_llama.mean()) + 2e-3 and agree > agree_llama - 0.05


@pytest.mark.parametrize("size", ["35B-A3B", "occamy"])
def test_mtp(size: str) -> None:
    """Qwen3.6-35B-A3B and Occamy 1.0 with their MTP drafters (``NpuQwen.generate``, up to 3 drafts per step): greedy
    replies to two user turns of a chat, the second after the first reply (decode steps up to a cache group, then a
    chunk from the host's carry). Every token is the argmax of the model's one-token steps over the same text, but where
    a multi-token step flips a near tie (its experts summed in another order: KL ~0.001 against steps of one here;
    Occamy's ~0.005-0.009, its steps of 4 as close to llama.cpp's decode per layer as steps of one, but the flips of its
    40 layers' routing near ties move the top two logits by up to ~0.6); most drafts are accepted in both replies (the
    drafter's cache follows the model's positions)."""
    from artlite.model_config import qwen_mtp_of
    from models.common import Sampling
    from models.qwen35.qwen import Qwen

    path = _path(size)
    if qwen_mtp_of(path) is None:
        pytest.skip(f"no MTP drafter of {path}")
    qwen = Qwen(path)
    model = qwen.load(capacity=4096)
    assert model.drafter is not None and model.DRAFTS == 3
    first = qwen.prompt(essay("apple.txt")[:4000] + "\n\nSummarize the essay above in a few paragraphs.")
    greedy, N = Sampling(temp=0.0, greedy=True), 48
    out1 = list(model.generate(model.forward(first), N, greedy))
    stats1 = (model.drafted, model.accepted)
    pos = model.position  # (the reply's last token is not cached)
    turn = ("<|im_end|>\n<|im_start|>user\nNow list the three most important arguments of the essay as short bullet "
            "points, each with one sentence of explanation, and say whether you agree with the author.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = first + out1 + qwen.tokenizer.encode(turn)
    assert model.prefill.group - pos % model.prefill.group + 3 < len(ids) - pos  # (decode steps, then a chunk)
    out2 = list(model.generate(model.forward(ids[pos:]), N, greedy))
    stats2 = (model.drafted - stats1[0], model.accepted - stats1[1])
    model.DRAFTS = 0
    model.reset()
    flips = []
    for prompt, out in ((first, out1), (ids[pos:], out2)):
        lg = model.forward(prompt)
        for i, t in enumerate(out):
            top = torch.topk(lg.float(), 2)
            if int(top.indices[0]) != t:
                flips.append(float(top.values[0] - top.values[1]))
            if i + 1 < len(out):
                lg = model.step(t)
    print(f"drafts (drafted, accepted): {stats1}, {stats2}; tokens not the argmax of one-token steps (their top-2 "
          f"gaps): {flips}")
    flips = [g for g in flips if g > 0]  # (not at exact ties of the bf16 logits: either token is their argmax)
    assert len(flips) <= 3 and all(g <= (1.0 if size == "occamy" else 0.5) for g in flips)
    assert all(a >= 0.5 * d for d, a in (stats1, stats2))
