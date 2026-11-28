"""LFM3-Audio's speech in the lfm3 language models (its mmproj, projector ``lfm3a``): a clip's samples, its log-mel
spectrogram and rows, the ASR prompts.

Samples: ``models.audio.load_audio``'s (16 kHz mono float32).

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

from collections.abc import Callable

import numpy as np

from ..audio import N_FFT, hann, mel_spectrum
from ..sequence import Sequence, splice

HOP, N_MEL = 160, 128
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


def log_mel(samples: np.ndarray, *, dtype: type = np.float32, block: int = 4096) -> np.ndarray:
    """The clip's log-mel spectrogram (N_MEL, frames) (module docstring), computed in ``dtype`` (float32: the host's;
    float64: the reference's), ``block`` frames at a time."""
    x = np.asarray(samples, dtype)
    n = len(x)
    xp = np.zeros(padded_length(n), dtype)
    xp[:n] = x
    nf = frames(n)
    win = hann().astype(dtype)
    out = np.empty((N_MEL, nf), dtype)
    view = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP]
    for f0 in range(0, nf, block):
        spec = np.fft.rfft(view[f0:f0 + block] * win, axis=1)
        power = spec.real * spec.real + spec.imag * spec.imag
        mel = mel_spectrum(power, N_MEL)
        out[:, f0:f0 + block] = (np.log10(np.maximum(mel, dtype(1e-12))) + 4) / 4
    return out


# -- prompts

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
