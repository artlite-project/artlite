"""LFM3-Audio's speech output for the lfm3 language models (its mmproj's generator, projector ``lfm3a_gen``): the prompts,
the speech codes' sampling and the feedback rows.

A frame (80 ms of speech) is 12 codes, one per codebook, which the depthformer scores from the language model's hidden
state in 3 MaskGIT passes (``SCHEDULE``: per frame a random order of the codebooks, drawn before its passes; pass p
samples codebooks order[4 p .. 4 p + 4) and reveals them to the next). Any codebook's code 2016 (``EOA``) ends the
audio (the frame's codes all 2016). The language model's next input is the frame's feedback row: the sum over its
codebooks of the mmproj's ``a.gen.code.out_embd`` rows 2017 cb + code (``feedback``).

Sampling (``CodeSampler``, llama.cpp's ``lfm3a_gen_audio_pipeline`` and transformers' ``sample_audio_frame``; the
mmproj's ``clip.gen.audio.sampling.*``, the model card's: temperature 1.0, top-k 128, a repetition penalty of 1.1 over
4096 frames): per codebook the CTRL penalty on the codes it took in the window (a logit / penalty if positive, else
times it), the top k (ties at the k-th kept), the temperature, a draw.

Prompts (``tts_prompt``): the language model file's ``tts`` chat template, ``Perform TTS in <language> using a <voice>
voice.`` with the text in the user turn and the assistant turn opened by ``<|audio_start|>``; the languages by code
(``TTS_LANGUAGES``; any other name is used as it is), the voices female (the default) and male.

The frames' loop (``speak``): a frame's passes on the vision overlay (``depth.Depth``), its feedback row through a step
of the language model without the LM head (``models.lfm.lfm.NpuLfm.audio_step``: the next hidden state), the frames
through the codec every ``CHUNK`` frames (``codec.Codec``: whole 8-key blocks of its window attention) and the ISTFT on
the host (``codec.Istft``), the samples streamed as they come. The language model's hardware context and the vision
overlay's alternate every frame (the model's overlay marked stale after the vision ones ran).
"""

import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import GenConfig

from ..common import render_template
from .codec import Istft

N_CB, N_CODES = 12, 2017  # (codebooks a frame, codes a codebook: 2016 FSQ codes and the end of audio)
EOA = N_CODES - 1
SCHEDULE = (4, 4, 4)  # (codebooks revealed per depthformer pass)
TEXT_BLOCK, AUDIO_BLOCK = 6, 12  # (an interleaved reply's blocks: text tokens, then frames)
SR = 24000  # (the codec's sample rate)
CHUNK = 4  # (frames a codec call: 24 positions, whole 8-key blocks; 320 ms of speech)
FRAME_SAMPLES = 1920  # (a frame's samples: 80 ms)
VOICES = ("female", "male")
TTS_LANGUAGES = ("ar", "da", "de", "el", "en", "en-us", "en-gb", "es", "fi", "fr", "he", "it", "ja", "ko", "nl", "pl",
                 "pt", "sv", "tr", "zh")
AUDIO_START, MIXED_START, TEXT_END, IM_END, AUDIO_OUT = (
    "<|audio_start|>", "<|mixed_start|>", "<|text_end|>", "<|im_end|>", "<|audio_out|>")


def tts_prompt(gguf: GGUFFile, text: str, *, language: str = "en", voice: str = "female") -> str:
    """The TTS prompt's text of the language model file's ``tts`` chat template (module docstring)."""
    template = gguf.meta.get("tokenizer.chat_template.tts")
    if template is None:
        raise ValueError(f"{gguf.path}: no tts chat template (not LFM3-Audio's language model)")
    if voice.lower() not in VOICES:
        raise ValueError(f"voice {voice!r}: the model's voices are {', '.join(VOICES)}")
    bos = gguf.strings("tokenizer.ggml.tokens", gguf.meta["tokenizer.ggml.bos_token_id"])[0]
    return render_template(str(template), messages=[{"role": "user", "content": text}], language=language, voice=voice,
                           bos_token=bos)


class CodeSampler:
    """The speech codes' sampling (module docstring) with its random generator and each codebook's codes of the
    penalty's window (counts over a ring of the last ``penalty_last_n`` frames)."""

    def __init__(self, cfg: GenConfig, seed: int | None = None):
        self.temp, self.top_k = cfg.temp, cfg.top_k
        self.penalty, self.last_n = cfg.penalty_repeat, cfg.penalty_last_n
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros((N_CB, N_CODES), np.int32)
        self.ring = np.zeros((max(self.last_n, 1), N_CB), np.int32)
        self.frames = 0

    def order(self) -> np.ndarray:
        """A frame's random order of the codebooks."""
        return self.rng.permutation(N_CB)

    def sample(self, logits: np.ndarray, cb: int) -> int:
        """Codebook cb's code from its logits (N_CODES,)."""
        x = np.array(logits, np.float64)
        if self.penalty > 1 and self.last_n > 0:
            seen = self.counts[cb] > 0
            x[seen] = np.where(x[seen] > 0, x[seen] / self.penalty, x[seen] * self.penalty)
        if self.temp <= 0:
            return int(np.argmax(x))
        if 0 < self.top_k < x.size:
            x[x < np.partition(x, x.size - self.top_k)[x.size - self.top_k]] = -np.inf
        p = np.exp((x - x.max()) / self.temp)
        c = np.cumsum(p)
        return int(min(np.searchsorted(c, self.rng.random() * c[-1]), x.size - 1))

    def accept(self, codes: np.ndarray) -> None:
        """A frame's codes into the penalty's window (not the end of audio's)."""
        if self.last_n <= 0:
            return
        slot = self.frames % self.last_n
        if self.frames >= self.last_n:
            np.subtract.at(self.counts, (np.arange(N_CB), self.ring[slot]), 1)
        self.ring[slot] = codes
        np.add.at(self.counts, (np.arange(N_CB), np.asarray(codes)), 1)
        self.frames += 1


def feedback(gguf: GGUFFile, codes: np.ndarray) -> np.ndarray:
    """A frame's feedback row (hidden,) float32: the sum of its codes' ``a.gen.code.out_embd`` rows."""
    return gguf.rows("a.gen.code.out_embd.weight", np.arange(N_CB) * N_CODES + np.asarray(codes)).sum(0)


def sample_frame(depth: Any, h: np.ndarray, sampler: CodeSampler) -> np.ndarray:
    """A frame's codes (12,) from the hidden state h: ``SCHEDULE``'s passes in a random order of the codebooks (module
    docstring); all ``EOA`` where any codebook took it."""
    depth.begin(h)
    order = sampler.order()
    codes = np.full(N_CB, -1, np.int64)
    i = 0
    for n in SCHEDULE:
        cbs = order[i:i + n]
        lg = depth.logits(codes, cbs)
        for j, cb in enumerate(cbs):
            codes[cb] = sampler.sample(lg[j], int(cb))
        i += n
    if (codes == EOA).any():
        codes[:] = EOA
    return codes


@dataclass
class SpeechStats:
    """A generation's frames and seconds: the depthformer's (``t_depth``), the codec's and the ISTFT's (``t_codec``),
    the language model's steps (``t_steps``)."""

    frames: int = 0
    t_depth: float = 0.0
    t_codec: float = 0.0
    t_steps: float = 0.0
    codes: list = field(default_factory=list)  # (the frames' codes)


def speak(model: Any, audio: Any, h: np.ndarray, sampler: CodeSampler, *, max_frames: int, chunk: int = CHUNK,
          stats: SpeechStats | None = None) -> Iterator[np.ndarray]:
    """The samples (24 kHz float32 chunks) of the frames after the hidden state h until the end of audio or
    ``max_frames`` (module docstring): ``model`` the language model on the NPU (``NpuLfm``: its next position follows
    the frames), ``audio`` its ``NpuAudio`` (depth, codec, the mmproj); stats: filled as it goes."""
    st = stats if stats is not None else SpeechStats()
    audio.codec.reset()
    ist = Istft()
    pend: list[np.ndarray] = []

    def out() -> np.ndarray:
        t0 = time.perf_counter()
        y = ist.push(audio.codec.frames(np.stack(pend)))
        pend.clear()
        st.t_codec += time.perf_counter() - t0
        return y

    for f in range(max_frames):
        t0 = time.perf_counter()
        codes = sample_frame(audio.depth, h, sampler)
        st.t_depth += time.perf_counter() - t0
        if codes[0] == EOA:
            break
        sampler.accept(codes)
        st.codes.append(codes)
        st.frames += 1
        pend.append(codes)
        if len(pend) == chunk:
            yield out()
        if f + 1 == max_frames:
            break
        t0 = time.perf_counter()
        model.stale()  # (the vision overlay ran)
        h = model.audio_step(feedback(audio.gguf, codes))
        st.t_steps += time.perf_counter() - t0
    if pend:
        yield out()
    yield ist.flush()


def pcm16(samples: np.ndarray) -> bytes:
    """Samples (float32 in [-1, 1]) as 16-bit little-endian PCM (clipped)."""
    return (np.clip(np.asarray(samples, np.float32), -1, 1) * 32767).round().astype("<i2").tobytes()


def wav_header(n: int | None, sr: int = SR) -> bytes:
    """A 16-bit mono WAV header of n samples (None: a stream of unknown length, its sizes 0xFFFFFFFF)."""
    data = 0xFFFFFFFF if n is None else 2 * n
    riff = 0xFFFFFFFF if n is None else 36 + data
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", riff, b"WAVE", b"fmt ", 16, 1, 1, sr, 2 * sr, 2, 16, b"data", data)


def wav(samples: np.ndarray, sr: int = SR) -> bytes:
    """A 16-bit mono WAV file of the samples."""
    return wav_header(len(samples), sr) + pcm16(samples)
