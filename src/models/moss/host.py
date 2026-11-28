"""MOSS-Transcribe-Diarize's host side (the checkpoint's ``MossTranscribeDiarizeProcessor``): a recording's chunks and
their log-mel spectrograms, its audio span in a prompt, the transcription instruction and a transcript's segments.

Chunks (``chunks``): a recording (16 kHz mono samples: ``models.audio.load_audio``) in pieces of 30 s, the last one
zero-padded; the encoder runs each chunk separately, always the padded chunk (Whisper's), and the adaptor makes a token
of every 4 encoder rows: ``tokens(n)`` = (n - 1) // 1280 + 1 of a chunk's n samples (12.5 a second, 375 a full chunk),
read from the chunk's first rows.

Mel (``log_mel``, transformers' ``WhisperFeatureExtractor``): a chunk reflect-padded by 200 samples at both ends, frames
of 400 samples every 160 under a periodic Hann window, the power spectrum of each, 80 Slaney mel filters, log10
floored 8 below the chunk's maximum, then (x + 4) / 4: 3000 frames (the 3001st dropped).

The audio span (``Span``): the chat template writes one ``<|audio_pad|>`` per recording; its place takes one pad per
token of the recording's chunks, in order (each the tower's row there: a ``Sequence``'s given row), with the elapsed
seconds as digit tokens after every ``every`` seconds' pads (5 s: after 62, 124, .. pads; the markers count 62 pads for
62.5 tokens, so they run ahead of the audio by half a token each, as the model was trained).

A transcript (``segments``) is the model's ``[start][Sxx]text[end]`` runs: times in seconds from the recording's start,
anonymous speaker tags. Its form gives draft tokens (``Drafts``): where speech goes on, a segment starts at the time
the last one ended, mostly with the same speaker, so after an end time's ``][`` the tokens up to the text are a guess
worth a step's spare slots (9 of a segment's ~15 tokens of times and tags).
"""

import re
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ..audio import N_FFT, SR, hann, mel_spectrum
from ..drafts import ALL, NMAX, Lookup
from ..sequence import MEDIA, Sequence

HOP, N_MEL = 160, 80
CHUNK = 30 * SR  # (samples of a chunk)
FRAMES = CHUNK // HOP  # (a chunk's mel frames: 3000)
ROWS = FRAMES // 2  # (the encoder's rows of a chunk: its second conv's stride)
PAD = "<|audio_pad|>"  # (the chat template's place of a recording)
# the model card's transcription instruction (OpenMOSS/MOSS-Transcribe-Diarize's ``DEFAULT_PROMPT``): the format the
# model answers in, a segment's start time and speaker tag before its text and its end time after
INSTRUCTION = ("请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
               "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。")


def tokens(n: int, merge: int = 4) -> int:
    """The tokens of a chunk's n samples (1 <= n <= CHUNK): one per ``merge`` encoder rows, the last from the samples
    there are."""
    return (n - 1) // (2 * HOP * merge) + 1


def chunks(samples: np.ndarray) -> list[np.ndarray]:
    """A recording's chunks of ``CHUNK`` samples (float32; the last one zero-padded: its samples are its ``tokens``')."""
    x = np.asarray(samples, np.float32).reshape(-1)
    if not len(x):
        raise ValueError("an audio clip without samples")
    out = [x[i:i + CHUNK] for i in range(0, len(x), CHUNK)]
    if len(out[-1]) < CHUNK:
        out[-1] = np.concatenate([out[-1], np.zeros(CHUNK - len(out[-1]), np.float32)])
    return out


def log_mel(chunk: np.ndarray) -> np.ndarray:
    """A chunk's log-mel spectrogram (N_MEL, FRAMES) float32 (module docstring), computed in float64 as the feature
    extractor does."""
    x = np.asarray(chunk, np.float64)
    assert x.shape == (CHUNK,), x.shape
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, N_FFT // 2, mode="reflect"), N_FFT)[:-1:HOP]
    spec = np.fft.rfft(frames * hann(), axis=1)
    mel = mel_spectrum(spec.real * spec.real + spec.imag * spec.imag, N_MEL)
    log = np.log10(np.maximum(mel, 1e-10))
    return ((np.maximum(log, log.max() - 8) + 4) / 4).astype(np.float32)


@dataclass(frozen=True)
class Span:
    """A recording's audio span in a prompt (module docstring): ``rate`` tokens a second, a time marker every ``every``
    seconds (0: none), the digits' token ids."""

    rate: float
    every: int
    digits: tuple[int, ...]

    @classmethod
    def of(cls, gguf: Any, tokenizer: Any) -> "Span":
        """A model file's (``moss.audio.*``), its tokenizer's digits (a token each)."""
        digits = [tokenizer.encode(d) for d in "0123456789"]
        if any(len(d) != 1 for d in digits):
            raise ValueError(f"{gguf.path}: a digit of several tokens (the time markers are digits)")
        return cls(float(gguf.meta["moss.audio.tokens_per_second"]), int(gguf.meta["moss.audio.time_marker_seconds"]),
                   tuple(d[0] for d in digits))

    def ids(self, count: int) -> np.ndarray:
        """The span's token ids of ``count`` audio tokens: ``MEDIA`` where a row stands, the markers' digits."""
        per = int(self.rate * self.every)  # (pads between two markers)
        if per <= 0:
            return np.full(count, MEDIA, np.int64)
        out: list[int] = []
        for k in range(1, int(count / self.rate) // self.every + 1):
            out += [MEDIA] * per + [self.digits[int(d)] for d in str(k * self.every)]
        return np.array(out + [MEDIA] * (count - sum(t == MEDIA for t in out)), np.int64)

    def sequence(self, rows: np.ndarray) -> Sequence:
        """A recording's tokens: its rows (tokens, hidden) at the span's pads, the markers between them."""
        ids = self.ids(len(rows))
        return Sequence(ids, ids == MEDIA, np.ascontiguousarray(rows, np.float32),
                        np.repeat(np.arange(len(ids), dtype=np.int64)[:, None], 3, 1), len(ids))


class Segment(NamedTuple):
    """A transcript's segment: its seconds from the recording's start, its speaker's tag ("S01"), its text."""

    start: float
    end: float
    speaker: str
    text: str


_TIME = r"\[(\d+(?:\.\d+)?)\]"
_SEGMENT = re.compile(rf"{_TIME}\s*\[(S\d+)\](.*?){_TIME}\s*(?={_TIME}\s*\[S\d+\]|$)", re.DOTALL)


def segments(text: str) -> list[Segment]:
    """A transcript's segments (``[start][Sxx]text[end]`` runs; a segment the answer's budget cut before its end time
    is left out). A bracketed number inside a text is the text's: an end time is followed by the next segment or
    nothing."""
    return [Segment(float(m[1]), float(m[4]), m[2], m[3].strip()) for m in _SEGMENT.finditer(text.strip())
            if float(m[4]) >= float(m[1])]


def plain(text: str) -> str:
    """A transcript's words alone: its segments' texts, a line per run of one speaker's (the text as it is where it
    has no segment)."""
    lines: list[tuple[str, list[str]]] = []
    for s in segments(text):
        if not lines or lines[-1][0] != s.speaker:
            lines.append((s.speaker, []))
        lines[-1][1].append(s.text)
    return "\n".join(" ".join(t for t in ts if t) for _, ts in lines) if lines else text.strip()


@dataclass(frozen=True)
class Marks:
    """The tokens of a transcript's times and tags: ``][`` (an end time's close and the next start's open), ``]``, ``[S``
    and, per token id, whether its text is digits and dots (``number``) or ends with ``[`` (``opens``)."""

    join: int
    close: int
    speaker: int
    number: np.ndarray
    opens: np.ndarray

    @classmethod
    def of(cls, gguf: Any, tokenizer: Any) -> "Marks":
        """A model file's vocabulary's (ValueError: one that writes ``][``, ``]`` or ``[S`` in several tokens)."""
        one = [tokenizer.encode(t) for t in ("][", "]", "[S")]
        if any(len(t) != 1 for t in one):
            raise ValueError(f"{gguf.path}: the transcript's brackets are not single tokens")
        texts = gguf.strings("tokenizer.ggml.tokens")
        return cls(*(t[0] for t in one), np.array([bool(t) and not t.strip("0123456789.") for t in texts]),
                   np.array([t.endswith("[") for t in texts]))


class Drafts(Lookup):
    """A transcript's context drafts (module docstring): after ``][`` the end time's tokens again, ``]`` and the last
    speaker tag, as far as the tokens since agree with them (where the model wrote a different start time, the tag after
    that time's ``]``); else the
    context's (``drafts.Lookup``)."""

    WINDOW = 24  # (tokens back the last ``][`` is looked for: two times and a tag)

    def __init__(self, marks: Marks, tokens: Any):
        super().__init__(tokens)
        self.marks = marks

    def draft(self, k: int) -> tuple[int, int, list[int]]:
        mk, seq = self.marks, self._buf[:self.n]
        tail = seq[-self.WINDOW:]
        at = np.flatnonzero(tail == mk.join)
        if k > 0 and len(at):
            j = self.n - len(tail) + int(at[-1])
            got = seq[j + 1:].tolist()
            i = j
            while i > 0 and mk.number[seq[i - 1]]:  # (the end time before the ``][``, after a token that opens it)
                i -= 1
            tags = np.flatnonzero(seq[:j] == mk.speaker)
            if i < j and i > 0 and mk.opens[seq[i - 1]] and len(tags):
                s = int(tags[-1])
                tag = seq[s:s + 6].tolist()
                tag = tag[:tag.index(mk.close) + 1] if mk.close in tag else []
                want = seq[i:j].tolist() + [mk.close] + tag
                if mk.close in got and all(mk.number[t] for t in got[:got.index(mk.close)]):
                    want = got[:got.index(mk.close) + 1] + tag  # (a different start time, complete)
                if tag and len(got) < len(want) and got == want[:len(got)]:
                    return NMAX, ALL, want[len(got):len(got) + k]
        return super().draft(k)
