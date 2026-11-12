"""Qwen3.6-35B-A3B with an image on the NPU (models.qwen35 with Qwen3-VL's mmproj: a ``Sequence`` prompt): an essay
paragraph, the astronaut, more essay and a question, teacher-forced over a description of the image, against
llama.cpp's own encoder and language model on the CPU (``ref.golden``). With llama.cpp's embeddings the prompt's
tokens and positions (the official M-RoPE ones, mtmd's) are tested on their own, through the prefill and as decode
steps; the app's path (``Qwen(mmproj=)``) adds the NPU encoder."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.qwen3vl import host as Q
from models.qwen35.qwen import NpuQwen, Qwen
from models.sequence import MEDIA
from models.tokenizer import BpeTokenizer, chat_prompt

from ..common import essay
from . import ref as V
from .test_ref import IMAGE, MMPROJ, MODEL

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


def _kl_report(name: str, want: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    kl = (lw.exp() * (lw - lg)).sum(-1)
    agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
    print(f"{name}: KL vs llama.cpp {float(kl.mean()):.4f} (max {float(kl.max()):.4f}), argmax agreement {agree:.3f} "
          f"over {len(kl)} tokens")
    return float(kl.mean()), agree


@pytest.fixture(scope="module")
def case() -> tuple:
    for p in (MODEL, MMPROJ):
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")
    g = GGUFFile(MODEL)
    tok = BpeTokenizer(g)
    text = chat_prompt(g, _messages(None), enable_thinking=False)
    answer = tok.encode(ANSWER)
    gold = V.golden(MODEL, MMPROJ, IMAGE, text.replace(PAD, "<__media__>"), answer=answer)
    raw = gold.tensor("inp_raw", 0)
    grid = Q.Grid(raw.shape[1] // 32, raw.shape[2] // 32)
    head, tail = text.split("<|image_pad|>")
    seq = tok.encode(head) + grid.sequence(gold.embd(0)) + tok.encode(tail)
    want = torch.from_numpy(np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1].copy())
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * c.get("image", 0))]
    assert ids == list(seq.ids) + answer, "the prompt's tokens: llama.cpp's"
    return seq, answer, want


def test_image_prompt(case: tuple) -> None:
    """llama.cpp's embeddings at the prompt's positions: through the prefill (chunks of 256: the image across a
    boundary) and as decode steps, within the language model's own distance from llama.cpp's (KL ~0.005, argmax
    agreement ~0.97)."""
    seq, answer, want = case
    for prefill in (True, False):
        model = NpuQwen(GGUFFile(MODEL), capacity=1024, prefill=prefill, sizes=(256,))
        got = torch.from_numpy(np.stack([model.forward(seq)] + [model.step(t) for t in answer[:-1]]))
        kl, agree = _kl_report("prefill" if prefill else "steps", want, got)
        assert model.rope_next == seq.next + len(answer) - 1  # (the text positions after the image's)
        assert kl < 0.01 and agree > 0.93
        del model


def test_app_image_prompt(case: tuple) -> None:
    """The app's path (``Qwen(mmproj=)``: the chat prompt with the NPU encoder's embeddings) teacher-forced over the
    description: the prompt's tokens and positions the reference's, the KL within ~2x of the language model's with
    llama.cpp's embeddings (``test_image_prompt``; measured as close, ~0.005)."""
    seq, answer, want = case
    app = Qwen(MODEL, mmproj=MMPROJ, spec_type="none")
    prompt = app.chat(_messages(IMAGE))
    assert np.array_equal(prompt.ids, seq.ids) and np.array_equal(prompt.pos, seq.pos)
    assert np.array_equal(prompt.given, seq.given)
    model = app.load(capacity=1024)
    got = torch.from_numpy(np.stack([model.forward(prompt)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = _kl_report("the NPU encoder, prefill", want, got)
    assert kl < 0.012 and agree > 0.93
