"""MOSS-Transcribe-Diarize's host side (models/moss/host.py): a recording's chunks and tokens, the log-mel spectrogram
against Whisper's recipe in torch, the audio span against the checkpoint's processor rule, a transcript's segments and
the drafts its form gives."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.audio import N_FFT, SR, mel_filters
from models.drafts import ALL, NMAX
from models.moss import host as H
from models.sequence import MEDIA


def test_chunks() -> None:
    """Chunks of 30 s, the last one zero-padded; a token per 1280 samples of a chunk's own (the processor's count)."""
    assert [H.tokens(n) for n in (1, 1280, 1281, 16000, H.CHUNK)] == [1, 1, 2, 13, 375]
    x = np.arange(H.CHUNK + 5, dtype=np.float32)
    a, b = H.chunks(x)
    assert a.shape == b.shape == (H.CHUNK,) and np.array_equal(a, x[:H.CHUNK])
    assert np.array_equal(b[:5], x[H.CHUNK:]) and not b[5:].any()
    with pytest.raises(ValueError, match="without samples"):
        H.chunks(np.zeros(0, np.float32))


def test_log_mel() -> None:
    """A chunk's spectrogram is Whisper's: the centred STFT of 400 samples every 160 under a Hann window, 80 mel
    filters, log10 floored 8 below the maximum, (x + 4) / 4."""
    rng = np.random.default_rng(0)
    t = np.arange(H.CHUNK) / SR
    x = (0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.02 * rng.standard_normal(H.CHUNK)).astype(np.float32)
    x[5 * SR:] *= np.linspace(1, 0, H.CHUNK - 5 * SR, dtype=np.float32)  # (fades into silence: the floor's part)
    got = H.log_mel(x)
    spec = torch.stft(torch.from_numpy(x).double(), N_FFT, H.HOP, window=torch.hann_window(N_FFT, dtype=torch.float64),
                      return_complex=True)[:, :-1].abs().square()
    want = (torch.from_numpy(mel_filters(H.N_MEL)) @ spec).clamp(min=1e-10).log10()
    want = (torch.maximum(want, want.max() - 8) + 4) / 4
    assert got.shape == (H.N_MEL, H.FRAMES) and got.dtype == np.float32
    assert np.abs(got - want.numpy()).max() < 1e-5


def processor_span(count: int, digits: tuple[int, ...], rate: float = 12.5, every: int = 5) -> list[int]:
    """The checkpoint's ``MossTranscribeDiarizeProcessor._audio_span_ids`` (pads as ``MEDIA``)."""
    per = int(rate * every)
    out: list[int] = []
    consumed = 0
    for sec in range(every, int(count / rate) + 1, every):
        pos = (sec // every) * per
        out += [MEDIA] * (pos - consumed)
        consumed = pos
        out += [digits[int(d)] for d in str(sec)]
    return out + [MEDIA] * (count - consumed)


SPAN = H.Span(12.5, 5, tuple(range(100, 110)))


def test_span() -> None:
    """The span's ids are the processor's: a marker after every 62 pads (5 s; a full chunk: 375 pads and 11 digits),
    the markers running on through a recording's chunks; no markers where the model has none."""
    for count in (1, 61, 62, 63, 375, 379, 750, 5000, 67500):
        assert SPAN.ids(count).tolist() == processor_span(count, SPAN.digits), count
    ids = SPAN.ids(375)
    assert len(ids) == 386 and int((ids == MEDIA).sum()) == 375
    assert ids[62] == 105 and ids[125:127].tolist() == [101, 100] and ids[-5:].tolist() == [103, 100, MEDIA, MEDIA, MEDIA]
    assert H.Span(12.5, 0, SPAN.digits).ids(400).tolist() == [MEDIA] * 400


def test_span_of_file() -> None:
    """A span's parameters are the model file's, its digits the tokenizer's (each one token)."""
    gguf = SimpleNamespace(path="x", meta={"moss.audio.tokens_per_second": 12.5, "moss.audio.time_marker_seconds": 5})
    span = H.Span.of(gguf, SimpleNamespace(encode=lambda d: [100 + int(d)]))
    assert span == SPAN
    with pytest.raises(ValueError, match="digit"):
        H.Span.of(gguf, SimpleNamespace(encode=lambda d: [1, 2]))


def test_sequence() -> None:
    """A recording's tokens: its rows at the pads, in order, the markers' digits as text, a position each."""
    rows = np.arange(130 * 4, dtype=np.float32).reshape(130, 4)
    seq = SPAN.sequence(rows)
    assert len(seq) == 130 + 3 and seq.next == 133 and np.array_equal(seq.rows, rows)
    assert np.array_equal(seq.given, seq.ids == MEDIA) and int(seq.given.sum()) == 130
    assert seq.ids[~seq.given].tolist() == [105, 101, 100] and np.array_equal(seq.pos[:, 0], np.arange(133))


TEXT = ("[0.19][S01] The English forwarded baskets of flowers.[4.85][4.85][S01] They cost [5] pounds, a plentiful"
        "[8.41][12.26][S02]The French, in return, invited them.[13.61]")


def test_segments() -> None:
    """A transcript's segments: start, end, speaker, text; a bracketed number in a text stays the text's, a segment
    cut before its end time is left out; ``plain``: the words, a line per run of a speaker's segments."""
    segs = H.segments(TEXT)
    assert [(s.start, s.end, s.speaker) for s in segs] == [(0.19, 4.85, "S01"), (4.85, 8.41, "S01"), (12.26, 13.61, "S02")]
    assert segs[1].text == "They cost [5] pounds, a plentiful" and segs[2].text == "The French, in return, invited them."
    assert H.plain(TEXT) == ("The English forwarded baskets of flowers. They cost [5] pounds, a plentiful\n"
                             "The French, in return, invited them.")
    assert len(H.segments(TEXT[:-7])) == 2 and len(H.segments(TEXT + "[13.61][S01] and")) == 3
    assert H.segments("no markers") == [] and H.plain(" no markers ") == "no markers"


# a vocabulary of the transcript's tokens: digits 0 - 9, then ".", "[", "]", "][", "[S", ".[" and words
DOT, OPEN, CLOSE, JOIN, TAG, DOT_OPEN, WORD = 10, 11, 12, 13, 14, 15, 16
MARKS = H.Marks(JOIN, CLOSE, TAG, np.arange(20) <= DOT, np.isin(np.arange(20), (OPEN, DOT_OPEN, JOIN)))


def test_marks() -> None:
    """The marks of a vocabulary: the brackets' single tokens, the tokens of digits and dots, those that open a time."""
    texts = [*"0123456789", ".", "[", "]", "][", "[S", ".[", " the", "", " 12", "x"]
    gguf = SimpleNamespace(path="x", strings=lambda key: texts)
    mk = H.Marks.of(gguf, SimpleNamespace(encode=lambda t: [texts.index(t)]))
    assert (mk.join, mk.close, mk.speaker) == (JOIN, CLOSE, TAG)
    assert np.array_equal(mk.number, MARKS.number) and np.array_equal(mk.opens, MARKS.opens)
    with pytest.raises(ValueError, match="single tokens"):
        H.Marks.of(gguf, SimpleNamespace(encode=lambda t: [0, 1]))


def test_drafts() -> None:
    """After an end time's ``][``: the same time, ``]`` and the last speaker's tag, as far as the tokens since agree; a
    start time of its own: the tag after its ``]``; elsewhere the context's drafts."""
    head = [OPEN, 0, DOT, 1, 9, CLOSE, TAG, 0, 1, CLOSE, WORD, WORD + 1]  # ([0.19][S01] w w)
    end = [DOT_OPEN, 4, DOT, 8, 5, JOIN]  # (.[4.85][)
    want = [4, DOT, 8, 5, CLOSE, TAG, 0, 1, CLOSE]
    d = H.Drafts(MARKS, head + end)
    assert d.draft(3) == (NMAX, ALL, want[:3]) and d.draft(16) == (NMAX, ALL, want)
    d.extend(want[:5])
    assert d.draft(3) == (NMAX, ALL, want[5:8])
    d.extend(want[5:])
    assert d.draft(3)[0] != NMAX  # (the tag is written: the context's drafts, if any)
    own = H.Drafts(MARKS, head + end + [1, 2])  # ([4.85][12: a start time of its own)
    assert own.draft(3)[0] != NMAX
    own.extend([DOT, 2, 6, CLOSE])
    assert own.draft(8) == (NMAX, ALL, [TAG, 0, 1, CLOSE])
    assert H.Drafts(MARKS, head).draft(3)[0] != NMAX and H.Drafts(MARKS, [WORD, JOIN]).draft(3)[0] != NMAX
