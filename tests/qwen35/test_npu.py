"""qwen35 models on the NPU (models.qwen35, models/qwen35/overlay.py) against llama.cpp's decode
(ubatch 1) of the same GGUF file (golden tensors and logits, tests/goldens/text.py):

- the whole model: the prompt through the prefill, then teacher-forced decode steps continuing from it: KL of the
  next-token distributions and argmax agreement against llama.cpp's split run, within llama.cpp's own noise: it
  quantizes the activations of every projection to Q8_K, and its prompt (one graph) and decode runs of the text
  differ by a KL of 0.0026 (9B) / 0.051 (27B: a few tokens of this text near a tie flip, in both runs and on the NPU);
- greedy generation with the MTP drafters, and the drafter after a prefill.
"""


import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from models.qwen35.qwen import NpuQwen

from ..common import draft_gate, draft_prompts, essay
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require, size_gguf
from ..numerics import kl_rows

pytestmark = pytest.mark.slow  # whole GGUF models: run with --run-slow

def _path(size: str) -> str:
    path = size_gguf(size)
    require(path)
    return path


def _drafter(path: str) -> None:
    """Skip a model without an MTP drafter (its file's own block, or an MTP file: ``qwen_mtp_of``)."""
    from artlite.model_config import qwen_mtp_of

    if not QwenConfig.from_gguf(GGUFFile(path)).nextn_layers and qwen_mtp_of(path) is None:
        pytest.skip(f"no MTP drafter of {path}")


def _tokens(path: str, n: int) -> list[int]:
    return llama_tokenize(path, essay("founders.txt"))[:n]


@pytest.mark.parametrize("size", ["9B", "9B-lanes", "3.8-27B", "1B", "1B-lanes", "35B-A3B", "swift"])
def test_prefill(size: str) -> None:
    """The app's prompt path: 300 tokens through the prefill (one chunk of 512 tokens, 212 of them padding), the
    last one's logits from the LM head alone, then teacher-forced decode steps continuing from the prefill's caches
    and states; the next-token distributions against llama.cpp's split run (the prompt in one graph, then one token per
    graph), next to llama.cpp's own decode-only run against it. "-lanes": the model of two sequences (the design of 8
    slots, every format's kernel a code page, the screened LM head's plane among them; MiniCPM-V's: a KV head's cache
    over 4 columns in one pass), on lane 0."""
    size, lanes, _ = size.partition("-lanes")
    path = _path(size)
    P, K = 300, 24
    tokens = _tokens(path, P + K)
    split = torch.from_numpy(
        np.asarray(golden(path, tokens, split=P, logits="f32", logits_from=P - 1).logits, np.float32)[:K].copy())
    dec = torch.from_numpy(
        np.asarray(golden(path, tokens, ubatch=1, logits="f32", logits_from=P - 1).logits, np.float32)[:K].copy())
    if lanes:
        from models.qwen35.qwen import Qwen

        model = Qwen(path).load(capacity=4096, sequences=2)
        assert model.overlay.SLOTS == 8 and len(model.lanes) == 2
    else:
        model = NpuQwen(GGUFFile(path), capacity=4096)
    got = torch.from_numpy(np.stack([model.forward(tokens[:P])] + [model.step(t) for t in tokens[P:P + K - 1]]))
    kl, kl_llama = kl_rows(split, got), kl_rows(split, dec)
    agree = float((split.argmax(-1) == got.argmax(-1)).float().mean())
    agree_llama = float((split.argmax(-1) == dec.argmax(-1)).float().mean())
    print(f"{size}: KL vs llama.cpp's split run: prompt's last token {float(kl[0]):.4f} (llama.cpp decode-only "
          f"{float(kl_llama[0]):.4f}), mean over {K} positions {float(kl.mean()):.4f} ({float(kl_llama.mean()):.4f}); "
          f"argmax agreement {agree:.3f} ({agree_llama:.3f})")
    # (each position's KL capped at 0.5 before the mean: an ill-conditioned position's rounding moves its distribution far, and one
    # such position sets an uncapped mean. Qwen3.6 27B's prompt's last token: llama.cpp's two runs 0.21 apart, the NPU's
    # residual within 1.4% of llama.cpp's to layer 35, 21% at the last layer, no layer a step: KL 1.19; medians 0.0013, llama.cpp
    # 0.0015)
    assert float(kl.clamp(max=0.5).mean()) < 1.5 * float(kl_llama.clamp(max=0.5).mean()) + 2e-3 and agree > agree_llama - 0.05


@pytest.mark.parametrize("size", ["35B-A3B", "3.8-27B"])
def test_mtp(size: str) -> None:
    """Qwen3.6-35B-A3B with its MTP drafter, Qwen3.8-27B with its file's own block (``NpuQwen.generate``, up to
    SLOTS - 1 drafts per step: 3, the 27B's design 7): greedy replies to two user turns of
    a chat, the second after the first reply (decode steps up to a cache group, then a chunk from the host's carry). Every token is the argmax of the
    model's one-token steps over the same text (steps of S slots give the numbers of S steps of one; an exact tie of the
    bf16 logits takes either); most drafts are accepted (the drafter's cache follows the model's positions; up to 7
    drafts: 40%), over both replies (a reply drafts as its measured step times say: 6 drafts in one, where one more or
    less kept moves its share by 17%), and each reply that drafts keeps some."""
    from models.qwen35.qwen import Qwen
    from models.sampling import Sampling

    path = _path(size)
    _drafter(path)
    qwen = Qwen(path, spec_type="draft-mtp")
    model = qwen.load(capacity=4096)
    assert model.drafter is not None and model.DRAFTS == model.overlay.SLOTS - 1
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
    drafts, model.DRAFTS = model.DRAFTS, 0
    model.reset()
    flips = []
    for prompt, out in ((first, out1), (ids[pos:], out2)):
        lg = model.forward(prompt)
        for i, t in enumerate(out):
            top = torch.topk(torch.from_numpy(lg), 2)
            if int(top.indices[0]) != t:
                flips.append(float(top.values[0] - top.values[1]))
            if i + 1 < len(out):
                lg = model.step(t)
    print(f"drafts (drafted, accepted): {stats1}, {stats2}; tokens not the argmax of one-token steps (their top-2 "
          f"gaps): {flips}")
    assert not [g for g in flips if g > 0]  # (an exact tie of the bf16 logits: either token is their argmax)
    # (up to 7 drafts: the chains run deeper while their probability pays a slot, about half of them kept)
    d, a = stats1[0] + stats2[0], stats1[1] + stats2[1]
    assert a >= (0.5 if drafts == 3 else 0.4) * d and all(a > 0 for d, a in (stats1, stats2) if d >= 4)


@pytest.mark.parametrize("size", ["35B-A3B"])
def test_mtp_prefill(size: str) -> None:
    """The drafter after prompts through the prefill drafts as after the same prompts as decode steps
    (``common.draft_gate``: teacher-forced over the model's greedy replies): the prefill writes the drafter's records of
    the prompt's rows from its own hidden states (``qwen35.draft_prefill``), which the model's KL does not see. Tokens
    per round 1.825 vs 1.823; a prompt's differ by up to ~2%, their first guesses the argmax by up to ~7% (the decode's
    residual is fp32, its sums of a slot's experts exact, ``decode.design``'s "MoE"; the prefill's residual bf16)."""
    from models.qwen35.qwen import Qwen

    path = _path(size)
    _drafter(path)
    qwen = Qwen(path, spec_type="draft-mtp")
    got = draft_gate(qwen.load(capacity=2048), draft_prompts(qwen))
    pre, steps = got["prefill"].mean(0), got["steps"].mean(0)
    assert pre[0] > 0.985 * steps[0] and (pre[1:] > steps[1:] - 0.03).all(), (pre, steps)
