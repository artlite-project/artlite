"""LFM3 through the app (``models.lfm.lfm.NpuLfm``) on the NPU:

- greedy generations with the context's drafts (``drafts.Lookup``) give the tokens of steps of one, in fewer dispatches
  on a text that repeats its prompt (steps of S slots bit-identical to steps of one; a rejected draft's conv rows and
  cache records forgotten: the lazy conv shift);
- ``forward``'s last-token logits through the prefill (the LM head alone on its last row) against a prompt of decode
  steps: within the two paths' difference (KL), the same argmax.
"""

import pytest
import torch

from models import open_model
from models.common import Sampling

from ..common import essay
from . import ref

pytestmark = pytest.mark.slow  # the whole model on the NPU: run with --run-slow
GREEDY = Sampling(greedy=True)


@pytest.fixture(scope="module")
def loaded() -> tuple:
    model = open_model(ref.path())
    return model, model.load(capacity=4096)


def test_lfm_drafts(loaded: tuple) -> None:
    model, npu = loaded
    text = essay("addiction.txt")[:900]
    prompt = model.chat([{"role": "user", "content": f"Repeat this text word for word:\n\n{text}"}])
    out = {}
    for lookup in (True, False):
        npu.reset()
        npu.lookup = lookup
        d0, a0, n0 = npu.drafted_ctx, npu.accepted_ctx, npu.dispatches
        out[lookup] = list(npu.generate(npu.forward(prompt), 96, GREEDY, prompt))
        print(f"lookup {lookup}: {npu.dispatches - n0} dispatches, {npu.drafted_ctx - d0} drafts, "
              f"{npu.accepted_ctx - a0} accepted: {model.tokenizer.decode(out[lookup])[:80]!r}")
    npu.lookup = True
    assert out[True] == out[False]
    assert npu.accepted_ctx > 48


def test_lfm_forward(loaded: tuple) -> None:
    model, npu = loaded
    for n in (40, 700):  # (one SMALL chunk; a chunk of 512 and one of 256 with 188 valid)
        ids = [model.bos] + model.encode(essay("apple.txt"))[:n - 1]
        npu.reset()
        got = npu.forward(ids)
        prefill, npu.prefill = npu.prefill, None
        try:
            npu.reset()
            want = npu.forward(ids)
        finally:
            npu.prefill = prefill
        lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
        kl = float((lw.exp() * (lw - lg)).sum())
        print(f"{n} tokens: KL(decode || prefill) {kl:.5f}")
        assert got.argmax() == want.argmax() and kl < 0.01, (n, kl)
