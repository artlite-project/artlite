"""``NpuModel.rewind``, the prompt cache of ``artlite-server``: a chat prompt A and a few generated tokens, then the
cache rewound to the prefix that a prompt B shares with them and the rest of B run, against B from an empty cache: the
same next token, and the logits of B's last token and of a fixed continuation within the KL of two prompt paths (the
chunk splits differ, so they are not bit-identical). B either leaves A inside its essay (the rewind lands in the
prefill's positions) or continues A's generated tokens (in the decode steps' positions)."""

import os

import pytest
import torch

from artlite.model_config import gemma_gguf, llama_gguf
from models import open_model
from models.common import Model, NpuModel, Sampling

from .common import essay

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

QUESTION_A = "Summarize this essay in two sentences."
QUESTION_B = "What is the main claim of this essay?"
# a fixed continuation (teacher forcing): both paths see the same tokens
TEXT = " The essay argues that technology makes the things we like more addictive."
# (file, prefill)
CASES = [(llama_gguf("Q4_0", "1B"), True), (llama_gguf("Q4_0", "1B"), False), (gemma_gguf("Q4_0"), True)]


def _kl(want: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    lr, lp = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    return (lr.exp() * (lr - lp)).sum(-1)


def _logits(npu: NpuModel, prompt: list[int], continuation: list[int]) -> torch.Tensor:
    """The logits of the prompt's last token and of the continuation's tokens, run after the cached positions."""
    out = [npu.forward(prompt)]
    out += [npu.step(t) for t in continuation]
    return torch.stack(out)


@pytest.fixture(scope="module", params=CASES, ids=[f"{os.path.basename(p)}-{'prefill' if f else 'steps'}"
                                                   for p, f in CASES])
def loaded(request: pytest.FixtureRequest) -> tuple[Model, NpuModel]:
    """A model file and the model on the NPU (once per case)."""
    path, prefill = request.param
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    model = open_model(path)
    return model, model.load(capacity=4096, prefill=prefill)


@pytest.mark.parametrize("where", ["prompt", "generated"])
def test_rewind(loaded: tuple[Model, NpuModel], where: str) -> None:
    model, npu = loaded
    text = essay("addiction.txt")[:3000]
    a = model.chat([{"role": "user", "content": f"{text}\n\n{QUESTION_A}"}])
    npu.reset()
    gen = list(npu.generate(npu.forward(a), 8, Sampling(greedy=True)))
    cached = (a + gen)[:npu.position]
    if where == "prompt":
        b = model.chat([{"role": "user", "content": f"{text}\n\n{QUESTION_B}"}])
    else:
        b = a + gen[:6] + model.encode(" Also, the essay")[1:]
    common = next((i for i, (x, y) in enumerate(zip(cached, b, strict=False)) if x != y), min(len(cached), len(b)))
    kept = npu.rewind(common)
    unit = npu.prefill.group if npu.prefill is not None else 8
    assert 0 < kept <= common and kept % unit == 0 and common - kept < unit, (common, kept)
    cont = model.encode(TEXT)[1:]
    got = _logits(npu, b[kept:], cont)
    npu.reset()
    want = _logits(npu, b, cont)
    kl = [round(float(x), 5) for x in _kl(want, got)]
    print(f"{model.name}: common prefix {common}, kept {kept} of {len(cached)}, B {len(b)}: KL {kl}")
    assert int(got[0].argmax()) == int(want[0].argmax())
    assert all(x < 0.002 for x in kl), kl
