"""Inkling 2 with an image on the NPU (models.inkling2 with its mmproj: a ``Sequence`` prompt): an essay paragraph, the
astronaut (512 x 512: padded to 520, the padding pixels' corrections), more essay and a question, teacher-forced over
a description of the image, against llama.cpp's own encoder and language model on the CPU (``ref.golden``): the
prompt's tokens against mtmd's, the language model on llama.cpp's image rows, then the app's path
(``Inkling(mmproj=)``: the NPU encoder's embeddings through the prefill and decode steps)."""

import numpy as np
import pytest
import torch

from artlite.model_config import inkling_gguf, inkling_mmproj_of
from models.inkling2.inkling import Inkling
from models.inkling2v import host as H
from models.sequence import MEDIA, load_image

from ..common import ASTRONAUT, essay
from ..model_files import require
from ..numerics import kl_report
from . import ref

pytestmark = pytest.mark.slow  # (the whole model)

MODEL = inkling_gguf()
ANSWER = ("<|message_model|><|content_text|>The image shows an astronaut in a white spacesuit riding a chestnut horse "
          "across a flat, sandy plain. The horse is rearing up on its hind legs, its mane and tail flowing, and the "
          "astronaut holds on with one gloved hand. Behind them is a dark, cloudy sky, which gives the scene a "
          "dramatic, surreal mood.<|end_message|>")
IMAGE = "<|content_image|><|unused_200054|>"  # (the chat template's image; llama.cpp's mtmd adds <|content_image|>)


def _messages(image: str | None) -> list[dict[str, object]]:
    """The prompt: an essay paragraph, the image (its file: the app's image part), more essay and the question."""
    words = essay("founders.txt").split()
    before, after = " ".join(words[:90]), " ".join(words[90:300]) + "\n\nDescribe the image."
    img = {"type": "image"} | ({"image": image} if image else {})
    return [{"role": "user", "content": [{"type": "text", "text": before}, img, {"type": "text", "text": after}]}]


@pytest.fixture(scope="module")
def app() -> Inkling:
    mmproj = inkling_mmproj_of(MODEL)
    require(MODEL, mmproj)
    return Inkling(MODEL, mmproj=mmproj, spec_type="none")


@pytest.fixture(scope="module")
def case(app: Inkling) -> tuple:
    text = app.chat_text(_messages(None))
    answer = app.encode(ANSWER)
    gold = ref.golden(MODEL, app.vision.path, ASTRONAUT, text.replace(IMAGE, "<__media__>"), answer=answer)
    head, tail = text.split(H.IMAGE)
    seq = app.encode(head) + H.grid(H.image_pixels(load_image(ASTRONAUT))).sequence(gold.embd(0)) + app.encode(tail)
    want = torch.from_numpy(np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1].copy())
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * c.get("image", 0))]
    assert ids == list(seq.ids) + answer, "the prompt's tokens: llama.cpp's"
    return seq, answer, want


def test_image_prompt(app: Inkling, case: tuple) -> None:
    """The language model on llama.cpp's image rows (its distance from llama.cpp's own run: KL 0.0012, the argmax
    0.975 of 81 tokens), then the app's path (``Inkling(mmproj=)``: the chat prompt with the NPU encoder's embeddings,
    the same tokens) teacher-forced over the description (0.0025, 0.988: the encoder's error on top of the language
    model's)."""
    seq, answer, want = case
    model = app.load(capacity=2048)
    got = torch.from_numpy(np.stack([model.forward(seq)] + [model.step(t) for t in answer[:-1]]))
    kl_lm, agree_lm = kl_report("llama.cpp's image rows, NPU prefill", want, got)
    prompt = app.chat(_messages(ASTRONAUT))
    assert np.array_equal(prompt.ids, seq.ids) and np.array_equal(prompt.given, seq.given)
    model.reset()
    got = torch.from_numpy(np.stack([model.forward(prompt)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = kl_report("the NPU encoder, NPU prefill", want, got)
    assert kl_lm < 0.005 and agree_lm > 0.93
    assert kl < 0.008 and agree > 0.93
