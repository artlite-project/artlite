"""``NpuModel.checkpoint`` / ``restore`` and ``read_records`` / ``write_records``, the prompt cache of
``artlite-server``: a chat prompt A, a checkpoint at its end, a few generated tokens (with drafts where the model has a
drafter: its last step uncommitted), a checkpoint there and A's records copied; another prompt B and its tokens overwrite
the caches and states; then A's records written back and its checkpoints restored - at the generation's end, then at
the prompt's end (inside a cache group, as the server's next request after an answer it does not continue) - and a fixed
continuation run from each, against the same continuations without B in between: the same logits, bit for bit. The
copies' times are printed."""

import os
import time

import pytest
import torch

from artlite.model_config import gemma_gguf, llama_gguf, minicpmv_gguf, qwen_gguf
from models import open_model
from models.common import Model, NpuModel, Sampling

from .common import essay

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

# the files: a Llama, a Gemma with its drafter, qwen35 (MiniCPM-V's language model) and qwen35moe with its drafter
CASES = [llama_gguf("Q4_0", "1B"), gemma_gguf("Q4_0"), minicpmv_gguf(), qwen_gguf("Q4_K_M", "35B-A3B")]
GREEDY = Sampling(greedy=True)


@pytest.fixture(scope="module", params=CASES, ids=[os.path.basename(p) for p in CASES])
def loaded(request: pytest.FixtureRequest) -> tuple[Model, NpuModel]:
    """A model file and the model on the NPU (once per case)."""
    if not os.path.exists(request.param):
        pytest.skip(f"{request.param} missing")
    model = open_model(request.param)
    return model, model.load(capacity=4096)


def _logits(npu: NpuModel, tokens: list[int]) -> torch.Tensor:
    """The logits of each of the tokens, run one after another from the cached positions."""
    return torch.stack([npu.forward(tokens[:1])] + [npu.step(t) for t in tokens[1:]])


def _timed(fn: object, *args: object) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def test_checkpoint(loaded: tuple[Model, NpuModel]) -> None:
    model, npu = loaded
    a = model.chat([{"role": "user", "content": f"{essay('addiction.txt')[:3000]}\n\nSummarize it in two sentences."}])
    b = model.chat([{"role": "user", "content": f"{essay('apple.txt')[:2000]}\n\nWhat is its main claim?"}])
    cont = model.encode(" The essay argues that technology makes the things we like more addictive.")[1:]
    # without B in between
    npu.reset()
    gen = list(npu.generate(npu.forward(a), 8, GREEDY))
    want_end = _logits(npu, gen[-1:] + cont)  # (the last generated token: not run yet)
    npu.reset()
    npu.forward(a)
    want_prompt = _logits(npu, cont)
    # A's checkpoints and records, B, A back
    npu.reset()
    logits = npu.forward(a)
    at_prompt, t_cp = _timed(npu.checkpoint)
    assert list(npu.generate(logits, 8, GREEDY)) == gen
    n = npu.position
    at_end = npu.checkpoint()
    records, t_read = _timed(npu.read_records, 0, n)
    npu.reset()
    list(npu.generate(npu.forward(b), 8, GREEDY))
    _, t_write = _timed(npu.write_records, records, 0, n)
    _, t_restore = _timed(npu.restore, n, at_end)
    got_end = _logits(npu, gen[-1:] + cont)
    npu.restore(len(a), at_prompt)  # (in place: the caches hold A's records)
    got_prompt = _logits(npu, cont)
    mb = sum(r.nbytes for r in records) / 2**20
    print(f"{model.name}: A {len(a)} tokens, {n} cached; checkpoint {at_end.nbytes / 2**20:.1f} MiB in {t_cp * 1e3:.1f} "
          f"ms, restored in {t_restore * 1e3:.1f} ms; records {mb:.1f} MiB read in {t_read * 1e3:.1f} ms, written in "
          f"{t_write * 1e3:.1f} ms")
    for got, want in ((got_end, want_end), (got_prompt, want_prompt)):
        assert torch.equal(got, want), (got - want).abs().max()
