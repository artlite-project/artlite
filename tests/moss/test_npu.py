"""MOSS-Transcribe-Diarize on the NPU (models.moss.moss.Moss: the audio tower on the vision overlay, the Qwen3 language
model on Llama's path) transcribing a LibriSpeech clip with the model's instruction, against the checkpoint computed
on the CPU: the float64 tower (tests/moss/ref.py) and the fp32 language model (tests/llama/cpu.py), teacher-forced over
the official model's transcript (transformers on the checkpoint's own code, greedy). With the reference's rows the
language model is tested on its own, as decode steps; the app's path adds the NPU tower and the prefill, and its greedy
transcript is the official one; transcribe.cpp's Q4_K_M file gives the same words."""

import numpy as np
import pytest
import torch

from artlite.hf_model import open_weights
from artlite.model_config import moss_gguf
from models.audio import load_audio
from models.moss import host as H
from models.moss.moss import Moss
from models.sequence import Sequence

from ..llama.cpu import LlamaCpu
from ..model_files import require
from ..numerics import kl_report, rel_rows
from . import ref as R
from .files import MODEL, clip

pytestmark = pytest.mark.slow  # (the whole model)

NAME = "6930-75918-0001"
# the official model's greedy transcript of the clip (transformers, fp32 on the CPU)
ANSWER = ("[0.19][S01] The english forwarded to the french baskets of flowers, of which they had made a plentiful[4.91]"
          "[4.91][S01] provision to greet the arrival of the young princess. The french, in return, invited the[10.31]"
          "[10.31][S01] english to a supper, which was to be given the next day.[14.01]")


@pytest.fixture(scope="module")
def app() -> Moss:
    require(MODEL)
    return Moss(MODEL)


@pytest.fixture(scope="module")
def case(app: Moss) -> tuple:
    x = load_audio(clip(NAME))
    prompt = app.chat(app.transcription([{"type": "audio", "audio": x}])[0])
    enc = R.Encoder(MODEL)
    rows = torch.cat([enc.forward(torch.from_numpy(H.log_mel(c)), H.tokens(min(H.CHUNK, len(x) - i * H.CHUNK)))
                      for i, c in enumerate(H.chunks(x))]).float().numpy()
    seq = Sequence(prompt.ids, prompt.given, rows, prompt.pos, prompt.next)  # (the reference's rows in their places)
    answer = app.tokenizer.encode(ANSWER)
    want = LlamaCpu(open_weights(MODEL)).logits(seq + answer, slice(len(seq) - 1, len(seq) + len(answer) - 1))
    return prompt, seq, answer, want


def test_prompt(app: Moss, case: tuple) -> None:
    """The transcription's prompt: the template's turns around the recording's span (a row per token, the tower's
    within a few percent of the reference's; the 5 s markers' digits between them) and the instruction."""
    prompt, seq, _, _ = case
    tok = app.tokenizer
    n = H.tokens(len(load_audio(clip(NAME))))
    assert int(prompt.given.sum()) == n == len(prompt.rows) and len(prompt) == len(seq)
    at = np.flatnonzero(prompt.given)
    assert prompt.ids[:at[0]].tolist() == tok.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                                                     "<|im_start|>user\n<|audio_start|>")
    assert prompt.ids[at[-1] + 1:].tolist() == tok.encode(f"<|audio_end|>\n{H.INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n")
    assert tok.decode(prompt.ids[at[0]:at[-1]][~prompt.given[at[0]:at[-1]]]) == "510"  # (14 s: the markers of 5 and 10 s)
    r = rel_rows(prompt.rows, seq.rows)
    assert np.median(r) < 0.025 and r.max() < 0.12, (np.median(r), r.max())


def test_reference_rows(app: Moss, case: tuple) -> None:
    """The reference tower's rows at the prompt's positions, as decode steps (no prefill): the language model alone."""
    _, seq, answer, want = case
    model = app.load(capacity=1024, prefill=False)
    got = torch.from_numpy(np.stack([model.forward(seq)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = kl_report("steps", want, got)
    assert kl < 0.003 and agree > 0.97


def test_transcription(app: Moss, case: tuple) -> None:
    """The app's path (the NPU tower's rows, the prefill) teacher-forced over the transcript, then greedy: the
    transcript the official model's, with and without draft tokens."""
    prompt, _, answer, want = case
    model = app.load(capacity=1024)
    got = torch.from_numpy(np.stack([model.forward(prompt)] + [model.step(t) for t in answer[:-1]]))
    kl, agree = kl_report("the NPU tower, prefill", want, got)
    assert kl < 0.004 and agree > 0.97
    for drafts in (None, 0):
        model.reset()
        text = app.tokenizer.decode([t for t in model.generate(model.forward(prompt), 200, app.sampling(), prompt,
                                                               drafts=drafts) if t not in app.eos_token_ids])
        assert text.strip() == ANSWER, drafts
    assert app.transcript(ANSWER)[0] == H.plain(ANSWER) and len(app.transcript(ANSWER)[1]) == len(H.segments(ANSWER))


def test_transcribe_cpp_file() -> None:
    """transcribe.cpp's Q4_K_M file (its language model's 4-bit tiles and screened head, its tower dequantized at load):
    the official transcript's words."""
    require(moss_gguf())
    app = Moss(moss_gguf())
    prompt = app.chat(app.transcription([{"type": "audio", "audio": load_audio(clip(NAME))}])[0])
    model = app.load(capacity=1024)
    text = app.tokenizer.decode([t for t in model.generate(model.forward(prompt), 200, app.sampling(), prompt)
                                 if t not in app.eos_token_ids])
    words = lambda t: "".join(c for c in H.plain(t).lower() if c.isalnum() or c == " ").split()
    assert words(text) == words(ANSWER) and len(H.segments(text)) == len(H.segments(ANSWER)), text
