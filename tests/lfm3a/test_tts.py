"""LFM3-Audio-2B's text to speech on the NPU (``Lfm.tts``: the language model's steps without the LM head, the mmproj's
depthformer and codec on the vision overlay) against llama.cpp's on the CPU (``gen_ref.golden``): the TTS prompt's
hidden state and, teacher-forced on llama.cpp's sampled frames, each frame's (the audio steps' hidden state from the LM
header's input), within the language model's own distance from llama.cpp's; seeded runs end with the end-of-audio frame
and their speech transcribes back to the text (the model's own speech encoder: ASR)."""

import os
import re

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from models.lfm.lfm import Lfm
from models.lfm3a import host as A
from models.lfm3a import speech as S

from . import gen_ref as R
from .test_gen_ref import MMPROJ, MODEL, TEXT

pytestmark = pytest.mark.slow  # (the whole model)


@pytest.fixture(scope="module")
def app() -> tuple[Lfm, object]:
    for p in (MODEL, MMPROJ):
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")
    m = Lfm(MODEL, mmproj=MMPROJ)
    return m, m.load(capacity=2048)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_audio_steps(app: tuple[Lfm, object]) -> None:
    """The prompt's hidden state, then llama.cpp's 40 sampled frames' feedback rows as audio steps: the frames' hidden
    states as close to llama.cpp's as the prompt's (a text prefill: the language model's own distance, ~4% after the
    output norm, KL 0.0002), none drifting (llama.cpp's Q4_K_M activations as Q8_0 on the CPU, the NPU's bf16 /
    bfp16; measured: prompt 4.2%, frames median 3.1%, max 4.2%)."""
    m, npu = app
    g = R.golden(MODEL, MMPROJ, prompt=S.tts_prompt(GGUFFile(MODEL), TEXT), gen=40, seed=1)
    ids = m.encode(S.tts_prompt(m.gguf, TEXT))
    assert ids == list(g.prompt_ids)
    npu.reset()
    h = npu.outputs(ids, at=[-1])[0]
    errs = [rel(h, g.hg(0))]
    for f, codes in enumerate(g.codes[:-1]):
        h = npu.audio_step(S.feedback(m.audio.gguf, codes))[0]
        errs.append(rel(h, g.hg(f + 1)))
    print(f"hidden states vs llama.cpp's: prompt {errs[0]:.4f}, frames median {np.median(errs[1:]):.4f} max {max(errs):.4f}")
    assert errs[0] < 0.06 and np.median(errs[1:]) <= errs[0] and max(errs[1:]) < 1.5 * errs[0]


def test_tts(app: tuple[Lfm, object]) -> None:
    """Two seeds: the speech ends with the end-of-audio frame (well before 512 frames: 2.5 - 8 frames a word), its
    samples are the frames' (1920 each, the codec's streamed chunks), the same seed gives the same samples, and the
    model's own ASR of it is the text (but punctuation and case)."""
    m, npu = app
    words = len(TEXT.split())
    outs = []
    for seed in (1, 2, 1):
        st = S.SpeechStats()
        y = np.concatenate(list(m.tts(TEXT, seed=seed, stats=st)))
        assert 2.5 * words <= st.frames <= 8 * words and y.size == st.frames * S.FRAME_SAMPLES
        outs.append(y)
    assert np.array_equal(outs[0], outs[2]) and not np.array_equal(outs[0], outs[1])
    norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()
    for y in outs[:2]:
        x = _to16k(y)
        msgs = [{"role": "system", "content": A.asr_system("en")}, {"role": "user", "content": [{"type": "audio", "audio": x}]}]
        npu.reset()
        prompt = m.chat(msgs)
        text = m.tokenizer.decode([t for t in npu.generate(npu.forward(prompt), 100, m.sampling(), prompt)
                                   if t not in m.eos_token_ids])
        print("ASR:", text)
        assert norm(text) == norm(TEXT)


def _to16k(y: np.ndarray) -> np.ndarray:
    """24 kHz samples at 16 kHz (``host.load_audio``'s resampler: a WAV through miniaudio)."""
    return A.load_audio(S.wav(y))


def test_voice_reply(app: tuple[Lfm, object]) -> None:
    """A spoken reply (interleaved text and speech, ``Lfm.voice_reply``) to a question: text and speech both come, the
    reply ends with ``<|im_end|>`` well before its limits, and the speech says the text (the model's ASR of it shares
    most of its words)."""
    m, npu = app
    st, rp = S.SpeechStats(), S.Reply()
    text, audio = [], []
    for kind, x in m.voice_reply([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
                                 seed=3, stats=st, reply=rp):
        (text if kind == "text" else audio).append(x)
    said = m.tokenizer.decode(text)
    y = np.concatenate(audio)
    print(f"reply: {said!r}; {st.frames} frames, {len(y) / S.SR:.2f} s")
    assert "paris" in said.lower() and 0 < st.frames < 2048 and y.size >= st.frames * S.FRAME_SAMPLES - S.FRAME_SAMPLES
    assert rp.ids.count(m.encode("<|audio_out|>")[0]) == len(rp.rows) == st.frames + 1  # (and the end of audio's)
    norm = lambda t: set(re.sub(r"[^a-z ]", "", t.lower()).split())
    msgs = [{"role": "system", "content": A.asr_system("en")}, {"role": "user", "content": [{"type": "audio", "audio": _to16k(y)}]}]
    npu.reset()
    prompt = m.chat(msgs)
    heard = m.tokenizer.decode([t for t in npu.generate(npu.forward(prompt), 200, m.sampling(), prompt)
                                if t not in m.eos_token_ids])
    print("ASR:", heard)
    assert len(norm(heard) & norm(said)) >= 0.7 * len(norm(said))
