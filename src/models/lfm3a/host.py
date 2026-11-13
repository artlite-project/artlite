"""LFM3-Audio's speech in the lfm3 language models (its mmproj, projector ``lfm3a``): a clip's samples, its log-mel
spectrogram and rows, the ASR prompts.

Samples (``load_audio``): 16 kHz mono float32. A WAV file (PCM of 8, 16, 24 or 32 bits, IEEE float of 32 or 64,
WAVE_FORMAT_EXTENSIBLE) at 16 kHz is parsed here, its channels averaged; other containers (MP3, FLAC, Ogg Vorbis) and
rates go through the ``miniaudio`` package, the library llama.cpp's mtmd decodes and resamples with (the same
configuration: float32, one channel, 16 kHz), so they give llama.cpp's samples.

Mel (``log_mel``, transformers' ``Lfm3AudioFeatureExtractor`` and llama.cpp's ``mtmd_audio_preprocessor_lfm3a``): the
clip right-padded with zeros to whole frames (``padded_length``), frames of 400 samples every 160 (not centred) under a
periodic Hann window, the power spectrum of each (``numpy.fft.rfft``), 128 Slaney mel filters (Slaney norm, 0 .. 8000
Hz), log10 floored at -12, then (x + 4) / 4: 100 frames a second. The encoder makes one row of every 8 frames
(``rows``: 12.5 a second), the last one from the frames there are (zero padding inside the encoder's convolutions).

Prompts: offline ASR is a chat with the system turn "Perform ASR in <language>." (``asr_system``: llama.cpp's
``common_chat_get_asr_prompt``; "Perform ASR." without a language) and the clip in the user turn (the chat template's
``<|audio|>``, replaced by the clip's rows: ``audio_prompt``), answered greedily without the audio-out tokens
(``TEXT_ONLY_BANNED``).
"""

import functools
import struct
from collections.abc import Callable

import miniaudio
import numpy as np

from ..sequence import Sequence, splice

SR, N_FFT, HOP, N_MEL = 16000, 400, 160, 128
STRIDE = 8  # (mel frames per encoder row)
AUDIO = "<|audio|>"  # (the chat template's place of a clip)
TEXT_ONLY_BANNED = ("<|mixed_start|>", "<|audio_start|>")  # (the tokens that would start speech output)
LANGUAGES = {"ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
             "fi": "Finnish", "fr": "French", "he": "Hebrew", "it": "Italian", "ja": "Japanese", "ko": "Korean",
             "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "sv": "Swedish", "tr": "Turkish", "zh": "Chinese"}


def asr_system(language: str = "") -> str:
    """The ASR prompt's system turn for a language code (or name; "": the model detects the language)."""
    if not language:
        return "Perform ASR."
    return f"Perform ASR in {LANGUAGES.get(language.lower(), language)}."


# -- samples

_WAVE_PCM, _WAVE_FLOAT, _WAVE_EXT = 1, 3, 0xFFFE


def _wav(data: bytes) -> tuple[np.ndarray, int] | None:
    """A RIFF / WAVE file's samples (frames, channels) float32 and rate; None: a format this does not read."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None or len(fmt) < 16:
        return None
    tag, ch, rate, _, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == _WAVE_EXT and len(fmt) >= 26:
        tag = struct.unpack_from("<H", fmt, 24)[0]  # (the subformat GUID's first two bytes)
    if ch < 1 or align != ch * bits // 8:
        return None
    n = len(raw) // align
    raw = raw[:n * align]
    if tag == _WAVE_PCM and bits == 8:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / np.float32(128)
    elif tag == _WAVE_PCM and bits == 16:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / np.float32(32768)
    elif tag == _WAVE_PCM and bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        x = (v - ((v & 0x800000) << 1)).astype(np.float32) / np.float32(1 << 23)
    elif tag == _WAVE_PCM and bits == 32:
        x = (np.frombuffer(raw, "<i4").astype(np.float64) / (1 << 31)).astype(np.float32)
    elif tag == _WAVE_FLOAT and bits == 32:
        x = np.frombuffer(raw, "<f4").astype(np.float32)
    elif tag == _WAVE_FLOAT and bits == 64:
        x = np.frombuffer(raw, "<f8").astype(np.float32)
    else:
        return None
    return x.reshape(n, ch), rate


def _miniaudio(data: bytes) -> np.ndarray:
    try:
        d = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=SR)
    except miniaudio.DecodeError as e:
        raise ValueError(f"cannot decode the audio: {e}") from None
    return np.frombuffer(d.samples, np.float32).copy()


def load_audio(src: str | bytes) -> np.ndarray:
    """An audio file (a path or its bytes) as 16 kHz mono float32 samples (module docstring)."""
    if isinstance(src, bytes):
        data = src
    else:
        with open(src, "rb") as f:
            data = f.read()
    w = _wav(data)
    if w is not None and w[1] == SR:
        x, _ = w
        return x[:, 0].copy() if x.shape[1] == 1 else x.mean(1, dtype=np.float32)
    return _miniaudio(data)


def wav_bytes(samples: np.ndarray, rate: int = SR) -> bytes:
    """Mono float samples as a 16-bit PCM WAV file (clipped, rounded)."""
    pcm = np.clip(np.round(np.asarray(samples, np.float64) * 32768), -32768, 32767).astype("<i2").tobytes()
    head = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, _WAVE_PCM, 1, rate, 2 * rate,
                       2, 16, b"data", len(pcm))
    return head + pcm


# -- mel

def padded_length(n: int) -> int:
    """Samples after the right padding to whole frames (at least one frame)."""
    return N_FFT if n <= N_FFT else n + (-(n - N_FFT)) % HOP


def frames(n: int) -> int:
    """Mel frames of n samples."""
    return 1 + (padded_length(n) - N_FFT) // HOP


def rows(nframes: int) -> int:
    """The encoder's rows of nframes mel frames."""
    return -(-nframes // STRIDE)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, np.float64)
    return np.where(f < 1000, 3 * f / 200, 15 + np.log(np.maximum(f, 1e-9) / 1000) * 27 / np.log(6.4))


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    return np.where(m < 15, 200 * m / 3, 1000 * np.exp((m - 15) * np.log(6.4) / 27))


@functools.cache
def mel_filters() -> np.ndarray:
    """The Slaney mel filter bank (N_MEL, N_FFT / 2 + 1) float64 (transformers' ``mel_filter_bank``, norm and scale
    "slaney", 0 .. SR / 2)."""
    hz = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SR / 2), N_MEL + 2))
    fft = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    d = np.diff(hz)
    slopes = hz[None, :] - fft[:, None]
    tri = np.maximum(0, np.minimum(-slopes[:, :-2] / d[:-1], slopes[:, 2:] / d[1:]))
    return (tri * (2 / (hz[2:] - hz[:-2]))[None, :]).T


@functools.cache
def _window() -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # (periodic Hann)


def log_mel(samples: np.ndarray, *, dtype: type = np.float32, block: int = 4096) -> np.ndarray:
    """The clip's log-mel spectrogram (N_MEL, frames) (module docstring), computed in ``dtype`` (float32: the host's;
    float64: the reference's), ``block`` frames at a time."""
    x = np.asarray(samples, dtype)
    n = len(x)
    xp = np.zeros(padded_length(n), dtype)
    xp[:n] = x
    nf = frames(n)
    win, fb = _window().astype(dtype), mel_filters().astype(dtype)
    out = np.empty((N_MEL, nf), dtype)
    view = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP]
    for f0 in range(0, nf, block):
        spec = np.fft.rfft(view[f0:f0 + block] * win, axis=1)
        power = spec.real * spec.real + spec.imag * spec.imag
        mel = fb @ power.T
        out[:, f0:f0 + block] = (np.log10(np.maximum(mel, dtype(1e-12))) + 4) / 4
    return out


# -- prompts

AUDIO_PARTS = ("audio", "input_audio", "audio_url")  # (a chat message's audio parts: the template's <|audio|>)


def part_audio(part: dict) -> str | bytes | np.ndarray:
    """The clip of a chat message's audio part: ``{"type": "audio", "audio": a file, its bytes or 16 kHz samples}``,
    OpenAI's ``{"type": "input_audio", "input_audio": {"data": base64, "format": ...}}`` or ``{"type": "audio_url",
    "audio_url": {"url": a data URL}}``; ValueError: none of them."""
    import base64

    from ..sequence import data_bytes
    t = part.get("type")
    try:
        if t == "audio":
            return part["audio"]
        if t == "input_audio":
            return base64.b64decode(part["input_audio"]["data"], validate=True)
        if t == "audio_url":
            v = part["audio_url"]
            return data_bytes(v["url"] if isinstance(v, dict) else v)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"an audio part without its clip: {e}") from None
    raise ValueError(f"not an audio part: {t!r}")

def audio_prompt(text: str, clips: list[object], encode: Callable[[str], list[int]],
                 sequence: Callable[[object], Sequence]) -> Sequence:
    """A chat prompt's text with the template's ``AUDIO`` per clip -> its tokens: each placeholder becomes the clip's
    rows (``sequence(clip)``: ``clip_sequence``'s)."""
    if text.count(AUDIO) != len(clips):
        raise ValueError(f"{len(clips)} audio clip(s) for {text.count(AUDIO)} place(s) in the prompt")
    return splice(text, AUDIO, [sequence(c) for c in clips], encode)


def clip_sequence(embd: np.ndarray) -> Sequence:
    """A clip's tokens: the encoder's rows ``embd`` (rows, hidden), a position each."""
    return Sequence.media(embd)
