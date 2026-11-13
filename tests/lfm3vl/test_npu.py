"""LFM3-VL-2B with an image on the NPU (models.lfm with LFM3-VL's mmproj: a ``Sequence`` prompt): an essay paragraph,
the astronaut, more essay and a question, teacher-forced over a description of the image, against llama.cpp's own
encoder and language model on the CPU (``ref.golden``). With llama.cpp's embeddings the prompt's tokens (1-D
positions) are tested on their own, through the prefill and as decode steps; the app's path (``Lfm(mmproj=)``) adds the
NPU encoder."""


import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.lfm.lfm import Lfm, NpuLfm
from models.lfm3vl import host as Q
from models.sequence import MEDIA
from models.tokenizer import BpeTokenizer, chat_prompt

from ..common import ASTRONAUT, essay
from ..goldens.media import golden
from ..models import require
from ..numerics import kl_report
from .files import MMPROJ, MODEL

pytestmark = pytest.mark.slow  # (the whole model)

ANSWER = ("The image shows an astronaut in a white spacesuit riding a brown horse across a grassy field. The horse is "
          "galloping, and the astronaut holds the reins with one hand. In the background there are rolling hills, a "
          "few trees and a clear blue sky with some clouds.")
PAD = "<|vision_start|><|image|><|vision_end|>"  # (the chat template's image; llama.cpp's mtmd adds the markers)


def _messages(image: str | None) -> list[dict[str, object]]:
    """The prompt: an essay paragraph, the image (its file: the app's image part), more essay and the question."""
    words = essay("founders.txt").split()
    before, after = " ".join(words[:90]), " ".join(words[90:300]) + "\n\nDescribe the image."
    img = {"type": "image"} | ({"image": image} if image else {})
    return [{"role": "user", "content": [{"type": "text", "text": before}, img, {"type": "text", "text": after}]}]


@pytest.fixture(scope="module")
def case() -> tuple:
    require(MODEL, MMPROJ)
    g = GGUFFile(MODEL)
    tok = BpeTokenizer(g)
    bos = g.strings("tokenizer.ggml.tokens", int(g.meta["tokenizer.ggml.bos_token_id"]))[0]
    text = chat_prompt(g, _messages(None), enable_thinking=False, bos_token=bos)
    answer = tok.encode(ANSWER)
    gold = golden(MODEL, MMPROJ, ASTRONAUT, text.replace(PAD, "<__media__>"), answer=answer)
    raw = gold.tensor("inp_raw", 0)
    grid = Q.Grid(raw.shape[1] // 32, raw.shape[2] // 32)
    head, tail = text.split(Q.IMAGE)
    seq = tok.encode(head) + grid.sequence(gold.embd(0)) + tok.encode(tail)
    want = torch.from_numpy(np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1].copy())
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * c.get("image", 0))]
    assert ids == list(seq.ids) + answer, "the prompt's tokens: llama.cpp's"
    return seq, answer, want


def test_image_prompt(case: tuple) -> None:
    """llama.cpp's embeddings at the prompt's positions: through the prefill (chunks of 256: the image across a
    boundary) and as decode steps, within the language model's own distance from llama.cpp's."""
    seq, answer, want = case
    for prefill in (True, False):
        model = NpuLfm(GGUFFile(MODEL), capacity=1024, prefill=prefill, sizes=(256,))
        got = torch.from_numpy(np.stack([model.forward(seq)] + [model.step(t) for t in answer[:-1]]))
        kl, agree = kl_report("prefill" if prefill else "steps", want, got)
        assert kl < 0.003 and agree > 0.95  # (measured 0.0005 / 0.0004)
        del model


def test_app_image_prompt(case: tuple) -> None:
    """The app's path (``Lfm(mmproj=)``: the chat prompt with the NPU encoder's embeddings) teacher-forced over the
    description: the prompt's tokens the reference's, the KL within ~2x of the language model's with llama.cpp's
    embeddings (``test_image_prompt``: 0.0005; the NPU encoder 0.0007)."""
    seq, answer, want = case
    app = Lfm(MODEL, mmproj=MMPROJ)
    prompt = app.chat(_messages(ASTRONAUT))
    assert np.array_equal(prompt.ids, seq.ids) and np.array_equal(prompt.given, seq.given)
    model = app.load(capacity=1024)
    got = torch.from_numpy(np.stack([model.forward(prompt)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = kl_report("the NPU encoder, prefill", want, got)
    assert kl < 0.004 and agree > 0.95  # (measured 0.0007)
