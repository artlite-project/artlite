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

Voice replies (``interleave``, llama.cpp's ``lfm3a_gen_audio_pipeline`` and transformers' ``generate``): the system
prompt ``INTERLEAVED`` and the assistant turn opened by ``<|mixed_start|>``; blocks of ``TEXT_BLOCK`` greedy text tokens
and ``AUDIO_BLOCK`` frames in turn; ``<|text_end|>`` in a text slot ends the text (frames follow until the end of audio,
then text, as after a reply that switched to speech by itself), ``<|im_end|>`` in any text slot the turn. A text slot's
token comes from the logits of the step before it, so the audio step before a text slot runs the LM head; the others do
not. A reply's sequence (its tokens, one ``<|audio_out|>`` a frame, the frames' feedback rows: ``Reply``) is how the
conversation's next turn sees it.

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
from ..sequence import Sequence

N_CB, N_CODES = 12, 2017  # (codebooks a frame, codes a codebook: 2016 FSQ codes and the end of audio)
EOA = N_CODES - 1
SCHEDULE = (4, 4, 4)  # (codebooks revealed per depthformer pass)
TEXT_BLOCK, AUDIO_BLOCK = 6, 12  # (an interleaved reply's blocks: text tokens, then frames)
SR = 24000  # (the codec's sample rate)
CHUNK = 4  # (frames a codec call: 24 positions, whole 8-key blocks; 320 ms of speech)
N_FFT, HOP, NBINS = 1280, 320, 641  # (the ISTFT: a frame's 6 positions are hops)
TRIM = (N_FFT - HOP) // 2
FRAME_SAMPLES = 1920  # (a frame's samples: 80 ms)
VOICES = ("female", "male")
TTS_LANGUAGES = ("ar", "da", "de", "el", "en", "en-us", "en-gb", "es", "fi", "fr", "he", "it", "ja", "ko", "nl", "pl",
                 "pt", "sv", "tr", "zh")
AUDIO_START, MIXED_START, TEXT_END, IM_END, AUDIO_OUT, AUDIO_IN = (
    "<|audio_start|>", "<|mixed_start|>", "<|text_end|>", "<|im_end|>", "<|audio_out|>", "<|audio|>")
INTERLEAVED = "Respond with interleaved text and audio."  # (a voice reply's system prompt)


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


class Istft:
    """The streaming ISTFT (llama.cpp's ``mtmd_audio_streaming_istft``, transformers' ``LiquidCodecISTFT`` with its
    streaming cache): ``push(frames)`` (STFT frames (n, 1282) [log magnitude | phase]) -> the samples they complete
    (24 kHz float32): exp(m) e^(i phi) -> irfft (1280) times a periodic Hann window, overlap-add at hop 320 divided by
    the window-square envelope, the stream's first 480 dropped; ``flush()`` the rest but its last 480."""

    def __init__(self):
        self.win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
        self.w2 = self.win ** 2
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(N_FFT)
        self.env = np.zeros(N_FFT)
        self.drop = TRIM

    def push(self, frames: np.ndarray) -> np.ndarray:
        f = np.asarray(frames, np.float64)
        if not len(f):
            return np.zeros(0, np.float32)
        seg = np.fft.irfft(np.exp(f[:, :NBINS] + 1j * f[:, NBINS:]), N_FFT, axis=1) * self.win
        n = len(f)
        acc = np.zeros(N_FFT + HOP * n)
        env = np.zeros(N_FFT + HOP * n)
        acc[:N_FFT], env[:N_FFT] = self.buf, self.env
        for k in range(n):
            acc[HOP * k:HOP * k + N_FFT] += seg[k]
            env[HOP * k:HOP * k + N_FFT] += self.w2
        y = acc[:HOP * n] / np.where(env[:HOP * n] > 1e-8, env[:HOP * n], 1)
        self.buf, self.env = acc[HOP * n:], env[HOP * n:]
        k = min(self.drop, y.size)
        self.drop -= k
        return y[k:].astype(np.float32)

    def flush(self) -> np.ndarray:
        n = N_FFT - HOP
        e = self.env[:n]
        y = self.buf[:n] / np.where(e > 1e-8, e, 1)
        self.reset()
        return y[:n - TRIM].astype(np.float32)


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
    """A generation's prompt tokens (``prompt``), frames and seconds: the depthformer's (``t_depth``), the codec's and
    the ISTFT's (``t_codec``), the language model's steps (``t_steps``)."""

    prompt: int = 0
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
        h = model.audio_step(feedback(audio.gguf, codes))[0]
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


@dataclass
class Reply:
    """A voice reply's sequence as it grows (``interleave``): per position its token id (``ids``; a frame's
    ``<|audio_out|>``) and, for frames, the feedback row the model ran there (``rows``, by position); ``text`` the
    text's token ids, ``codes`` the frames' codes."""

    ids: list[int] = field(default_factory=list)
    rows: dict[int, np.ndarray] = field(default_factory=dict)
    text: list[int] = field(default_factory=list)
    codes: list[np.ndarray] = field(default_factory=list)

    def sequence(self, hidden: int) -> Sequence:
        """The positions the reply ran as a prompt's tokens: its token ids, a frame's feedback row at its
        ``<|audio_out|>`` (a conversation's next turn continues from them exactly)."""
        n = len(self.ids)
        given = np.zeros(n, bool)
        given[list(self.rows)] = True
        rows = np.stack([self.rows[i] for i in sorted(self.rows)]).astype(np.float32) if self.rows else \
            np.zeros((0, hidden), np.float32)
        return Sequence(np.asarray(self.ids, np.int64), given, rows,
                        np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1), n)


def interleave(model: Any, audio: Any, tokens: dict[str, int], logits: np.ndarray | None, sampler: CodeSampler, *,
               max_tokens: int, max_frames: int, mode: str = "mixed", h: np.ndarray | None = None,
               speech: bool = True, reply: Reply | None = None, stats: SpeechStats | None = None
               ) -> Iterator[tuple[str, Any]]:
    """A voice reply after a prompt (module docstring): ("text", token id) and ("audio", samples) as they come, until
    ``<|im_end|>``, ``max_tokens`` text tokens or ``max_frames`` frames. ``mode``: how the prompt opened the reply -
    "mixed" (``<|mixed_start|>``), "audio" (``<|audio_start|>``: speech, then text), "text" (a text reply, which may
    switch to speech by itself unless ``speech`` is false); ``logits``: the prompt's last token's (a text slot first),
    ``h``: its hidden state ("audio": a frame first); ``tokens``: the special tokens' ids by name (``speech_tokens``);
    ``reply``: filled with the reply's sequence. A speech segment's samples stream every ``CHUNK`` frames across the
    text blocks, its tail at its end of audio."""
    st = stats if stats is not None else SpeechStats()
    rp = reply if reply is not None else Reply()
    t_end, im_end, a_out = tokens[TEXT_END], tokens[IM_END], tokens[AUDIO_OUT]
    speaking, text_done, left = False, False, TEXT_BLOCK
    banned = [a_out, tokens[AUDIO_IN]] + ([] if speech else [tokens[AUDIO_START], tokens[MIXED_START]])
    ist, pend, seg = Istft(), [], False  # (seg: a speech segment open)
    audio.codec.reset()

    def out() -> np.ndarray:
        t0 = time.perf_counter()
        y = ist.push(audio.codec.frames(np.stack(pend)))
        pend.clear()
        st.t_codec += time.perf_counter() - t0
        return y

    def tail() -> np.ndarray:  # (a segment's end: its last frames, the ISTFT's held samples; the codec anew)
        y = np.concatenate([out() if pend else np.zeros(0, np.float32), ist.flush()])
        audio.codec.reset()
        return y

    ntext = 0
    while ntext < max_tokens and st.frames < max_frames and model.position < model.capacity:
        inter = mode == "mixed"
        if mode == "text" or (inter and not speaking):  # (a text slot)
            row = np.array(logits, np.float32)
            row[banned] = -np.inf
            tok = int(np.argmax(row))
            if tok == im_end or (not inter and tok == t_end):
                break
            if not inter and tok == tokens[AUDIO_START]:
                mode = "audio"
            elif not inter and tok == tokens[MIXED_START]:
                mode, left, speaking, text_done = "mixed", TEXT_BLOCK, False, False
            elif inter and tok == t_end:
                text_done, speaking = True, True
            else:
                yield "text", tok
                rp.text.append(tok)
                ntext += 1
            rp.ids.append(tok)
            t0 = time.perf_counter()
            logits = model.cap(model.steps([tok])[0])
            h = model.hidden_after()
            st.t_steps += time.perf_counter() - t0
        else:  # (a frame)
            t0 = time.perf_counter()
            codes = sample_frame(audio.depth, h, sampler)
            st.t_depth += time.perf_counter() - t0
            if codes[0] == EOA:
                if seg:
                    yield "audio", tail()
                seg, mode = False, "text"
            else:
                sampler.accept(codes)
                st.codes.append(codes)
                rp.codes.append(codes)
                st.frames += 1
                pend.append(codes)
                seg = True
                if len(pend) == CHUNK:
                    yield "audio", out()
            fb = feedback(audio.gguf, codes)
            t0 = time.perf_counter()
            model.stale()  # (the vision overlay ran)
            rp.rows[len(rp.ids)] = fb
            rp.ids.append(a_out)
            h, lg = model.audio_step(fb, logits=mode == "text" or (inter and not text_done and left == 1))
            logits = logits if lg is None else lg
            st.t_steps += time.perf_counter() - t0
        if inter and not text_done:  # (the slot counts down its block; a switching token does not)
            left -= 1
            if left == 0:
                speaking = not speaking
                left = AUDIO_BLOCK if speaking else TEXT_BLOCK
    if seg:
        yield "audio", tail()


def speech_tokens(vocab: list[str]) -> dict[str, int]:
    """The special tokens of speech output by name (a vocabulary's ids)."""
    return {t: vocab.index(t) for t in (AUDIO_START, MIXED_START, TEXT_END, IM_END, AUDIO_OUT, AUDIO_IN)}
