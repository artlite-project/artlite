"""LFM3-Audio-2B with a clip on the NPU (models.lfm with LFM3-Audio's mmproj: a ``Sequence`` prompt): the ASR prompt of
a LibriSpeech clip, teacher-forced over its transcript, against llama.cpp's own encoder and language model on the CPU
(``ref.golden``). With llama.cpp's embeddings the clip's rows are tested on their own, through the prefill and as
decode steps; the app's path (``Lfm(mmproj=)``) adds the NPU encoder; its greedy transcript is the reference's."""


import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.lfm.lfm import Lfm, NpuLfm
from models.lfm3a import host as A
from models.sequence import MEDIA
from models.tokenizer import BpeTokenizer

from ..goldens.media import golden
from ..models import require
from ..numerics import kl_report
from .test_ref import MARKER, MMPROJ, MODEL, asr_text, clip

pytestmark = pytest.mark.slow  # (the whole model)

NAME = "6930-75918-0001"
ANSWER = ("The English forwarded to the French baskets of flowers, of which they had made a plentiful provision to "
          "greet the arrival of the young princess. The French, in return, invited the English to a supper, which was "
          "to be given the next day.")  # (llama.cpp's greedy transcript of the clip, CPU)


def _messages(path: str) -> list[dict[str, object]]:
    return [{"role": "system", "content": A.asr_system("en")}, {"role": "user", "content": [{"type": "audio", "audio": path}]}]


@pytest.fixture(scope="module")
def case() -> tuple:
    require(MODEL, MMPROJ)
    g = GGUFFile(MODEL)
    tok = BpeTokenizer(g)
    answer = tok.encode(ANSWER)
    gold = golden(MODEL, MMPROJ, clip(NAME), asr_text(g, marker=MARKER), answer=answer)
    head, tail = asr_text(g).split(A.AUDIO)
    seq = tok.encode(head) + A.clip_sequence(gold.embd(0)) + tok.encode(tail)
    want = torch.from_numpy(np.asarray(gold.logits, np.float32)[-len(answer) - 1:-1].copy())
    ids = [t for c in gold.chunks for t in c.get("text", [MEDIA] * c.get("audio", 0))]
    assert ids == list(seq.ids) + answer, "the prompt's tokens: llama.cpp's"
    return seq, answer, want


def test_audio_prompt(case: tuple) -> None:
    """llama.cpp's embeddings at the prompt's positions: through the prefill (chunks of 128: the clip across a
    boundary) and as decode steps, within the language model's own distance from llama.cpp's."""
    seq, answer, want = case
    for prefill in (True, False):
        model = NpuLfm(GGUFFile(MODEL), capacity=1024, prefill=prefill, sizes=(128,))
        got = torch.from_numpy(np.stack([model.forward(seq)] + [model.step(t) for t in answer[:-1]]))
        kl, agree = kl_report("prefill" if prefill else "steps", want, got)
        assert kl < 0.003 and agree > 0.95
        del model


def test_app_audio_prompt(case: tuple) -> None:
    """The app's path (``Lfm(mmproj=)``: the ASR chat prompt with the NPU encoder's embeddings) teacher-forced over the
    transcript, then greedy: the prompt's tokens the reference's, the KL within ~2x of the language model's with
    llama.cpp's embeddings, the transcript llama.cpp's."""
    seq, answer, want = case
    app = Lfm(MODEL, mmproj=MMPROJ)
    prompt = app.chat(_messages(clip(NAME)))
    assert np.array_equal(prompt.ids, seq.ids) and np.array_equal(prompt.given, seq.given)
    model = app.load(capacity=1024)
    got = torch.from_numpy(np.stack([model.forward(prompt)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = kl_report("the NPU encoder, prefill", want, got)
    assert kl < 0.004 and agree > 0.95
    model.reset()
    sampling = app.sampling()
    assert sampling.k == 1 and len(sampling.banned) == 2
    text = app.tokenizer.decode([t for t in model.generate(model.forward(prompt), 200, sampling, prompt)
                                 if t not in app.eos_token_ids])
    assert text.strip() == ANSWER
