"""Qwen3.6-35B-A3B with an image on the NPU (models.qwen35 with Qwen3-VL's mmproj: a ``Sequence`` prompt): an essay
paragraph, the astronaut, more essay and a question, teacher-forced over a description of the image, against
llama.cpp's own encoder and language model on the CPU (``ref.golden``): the prompt's tokens and positions (the
official M-RoPE ones, mtmd's) against llama.cpp's, then the app's path (``Qwen(mmproj=)``: the NPU encoder's
embeddings through the prefill and decode steps)."""


import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.qwen3vl import host as Q
from models.qwen35.qwen import Qwen
from models.sequence import MEDIA
from models.tokenizer import BpeTokenizer, chat_prompt

from ..common import ASTRONAUT, essay
from ..goldens.media import golden
from ..model_files import require
from ..numerics import kl_report
from .files import MMPROJ, MODEL

pytestmark = pytest.mark.slow  # (the whole model)

ANSWER = ("The image shows an astronaut in a white spacesuit riding a brown horse across a grassy field. The horse is "
          "galloping, and the astronaut holds the reins with one hand. In the background there are rolling hills, a "
          "few trees and a clear blue sky with some clouds.")
PAD = "<|vision_start|><|image_pad|><|vision_end|>"  # (the chat template's image; llama.cpp's mtmd adds the markers)


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
    text = chat_prompt(g, _messages(None), enable_thinking=False)
    answer = tok.encode(ANSWER)
    gold = golden(MODEL, MMPROJ, ASTRONAUT, text.replace(PAD, "<__media__>"), answer=answer)
    raw = gold.tensor("inp_raw", 0)
    grid = Q.Grid(raw.shape[1] // 32, raw.shape[2] // 32)
    head, tail = text.split("<|image_pad|>")
    seq = tok.encode(head) + grid.sequence(gold.embd(0)) + tok.encode(tail)
    want = torch.from_numpy(np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1].copy())
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * c.get("image", 0))]
    assert ids == list(seq.ids) + answer, "the prompt's tokens: llama.cpp's"
    return seq, answer, want


def test_app_image_prompt(case: tuple) -> None:
    """The app's path (``Qwen(mmproj=)``: the chat prompt with the NPU encoder's embeddings) teacher-forced over the
    description: the prompt's tokens and positions the reference's, the KL within the gate (the language
    model's own distance from llama.cpp's ~0.005, plus the encoder's)."""
    seq, answer, want = case
    app = Qwen(MODEL, mmproj=MMPROJ, spec_type="none")
    prompt = app.chat(_messages(ASTRONAUT))
    assert np.array_equal(prompt.ids, seq.ids) and np.array_equal(prompt.pos, seq.pos)
    assert np.array_equal(prompt.given, seq.given)
    model = app.load(capacity=1024)
    got = torch.from_numpy(np.stack([model.forward(prompt)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = kl_report("the NPU encoder, prefill", want, got)
    assert kl < 0.012 and agree > 0.93
