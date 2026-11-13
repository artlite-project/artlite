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
"""

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import GenConfig

from ..common import render_template

N_CB, N_CODES = 12, 2017  # (codebooks a frame, codes a codebook: 2016 FSQ codes and the end of audio)
EOA = N_CODES - 1
SCHEDULE = (4, 4, 4)  # (codebooks revealed per depthformer pass)
TEXT_BLOCK, AUDIO_BLOCK = 6, 12  # (an interleaved reply's blocks: text tokens, then frames)
SR = 24000  # (the codec's sample rate)
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
