"""MOSS-Transcribe's host front end: the samples, the log-mel, the audio span's ids, its time markers and the splice of
the adaptor's rows.

Samples (``models.lfm3a.host.load_audio``: the 16 kHz mono decoder every speech front end shares) are pad-or-
trimmed to one 30 s chunk (``fit_chunk``: 480,000 samples, the processor's ``WhisperFeatureExtractor``), then the
log-mel (``log_mel``: 80 bins of the 400-sample frames every 160 under a periodic Hann window, their power spectrum,
Slaney mel filters, log10 floored at the chunk's global max - 8 dB, then (x + 4) / 4: 3,000 frames). The floor is the
chunk's own (the processor normalises a chunk, never a longer recording).

A clip's audio span is the processor's ``_compute_audio_token_length``: ``span_tokens(n)`` = ``(n - 1) // 1,280 + 1``
tokens (1,280 samples = 80 ms: hop 160 x the encoder's stride 2 x the 4-row merge), at most a 30 s chunk's 375 - 2 s ->
25, 5 s -> 63, 15 s -> 188, a full chunk -> 375. The encoder runs on the whole 30 s chunk (1,500 rows -> 375 merged
rows), and a span reads the first ``span_tokens(n) * 4`` of those rows, the processor's slice of a chunk's features.

The processor's text carries one ``<|audio_pad|>``; the span replaces it with the pads and the time markers' digit
tokens every 5 seconds (at 62-token positions: 5, 10, ...), so a full chunk's span is 386 ids (375 pads + 11 digits) in
7 runs of pads. The VQ adaptor's rows stand at the pads' positions only - a ``Sequence.given`` row a pad, the digits
keeping their text ids (``models.sequence``) - and the decoder's output weight is the tokens' embedding
(``hf_model.HfModel``'s ``output.weight``, tied).

The time markers' positions are the processor's own arithmetic (``int(12.5 * 5) = 62`` tokens a marker, five seconds
being 62.5 tokens): the span drifts by half a token a marker, as the processor's does.

The prompt's length is the span's (the count's pads plus the markers' digit tokens) plus the conversation's text tokens:
414 for this package's test's conversation and a 30 s chunk - the prompt a decoder gate compares must be one text on both
sides, not a fixed count.
"""

from collections.abc import Callable

import numpy as np

from ..sequence import Sequence

SR, N_SAMPLES, N_FFT, HOP, N_MEL = 16000, 480000, 400, 160, 80  # (the processor's 30 s chunk and mel)
CHUNK_FRAMES = 3000  # (the mel's frames of a chunk: one dropped by the extractor's STFT, 3,001 windows)

AUDIO_PAD, AUDIO_START, AUDIO_END = 151671, 151669, 151670

# the official transcription instruction (OpenMOSS/MOSS-Transcribe-Diarize's ``DEFAULT_PROMPT``, verbatim,
# Apache-2.0): the timestamped, speaker-tagged format the model was trained to answer in
INSTRUCTION = ("请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
               "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。")
TOKENS_PER_SECOND, MERGE, MARKER_EVERY = 12.5, 4, 5
CHUNK_ROWS, CHUNK_TOKENS = 1500, 375  # (the encoder's rows and the adaptor's, one 30 s chunk)
SPAN_STRIDE = HOP * 2 * MERGE  # (1,280 samples a token: the encoder's stride 2 and the 4-row merge)


def span_tokens(n_samples: int) -> int:
    """A clip's audio span (the processor's ``_compute_audio_token_length``): ``(n - 1) // 1,280 + 1`` tokens, at least
    one and at most a 30 s chunk's 375 (a longer clip is cut to the chunk)."""
    return min(CHUNK_TOKENS, max(1, (int(n_samples) - 1) // SPAN_STRIDE + 1))


def audio_span(count: int = CHUNK_TOKENS, *, digits: dict[str, int], every: int = MARKER_EVERY,
               rate: float = TOKENS_PER_SECOND, pad: int = AUDIO_PAD) -> np.ndarray:
    """The processor's audio span (``MossTranscribeDiarizeProcessor._audio_span_ids``): ``count`` pad ids with the
    digit token ids of each ``every`` seconds' mark (the seconds as text) at their token position."""
    out: list[int] = []
    per_marker = int(rate * every)
    consumed = 0
    for sec in range(every, int(count / float(rate)) + 1, every):
        pos = (sec // every) * per_marker
        out += [pad] * (pos - consumed)
        consumed = pos
        out += [digits[d] for d in str(sec)]
    out += [pad] * (count - consumed)
    return np.asarray(out, np.int64)


def pad_runs(ids: np.ndarray, *, pad: int = AUDIO_PAD) -> np.ndarray:
    """The first index of each run of pads of a span (the digits split it)."""
    on = ids == pad
    starts = np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))
    return starts


def audio_sequence(rows: np.ndarray, *, digits: dict[str, int], count: int | None = None,
                   every: int = MARKER_EVERY, pad: int = AUDIO_PAD) -> Sequence:
    """The span as a ``Sequence``: one row of ``rows`` (the adaptor's, in order) per pad, the digits their text ids.
    ``count``: the span's tokens, the rows' count by default."""
    count = rows.shape[0] if count is None else count
    if rows.shape[0] != count:
        raise ValueError(f"the adaptor's rows {rows.shape} for {count} audio tokens")
    ids = audio_span(count, digits=digits, every=every, pad=pad)
    given = ids == pad
    return Sequence(np.where(given, -1, ids), given, np.ascontiguousarray(rows, np.float32),
                    np.repeat(np.arange(len(ids), dtype=np.int64)[:, None], 3, 1), len(ids))


def splice_audio(text: str, rows: np.ndarray, encode: Callable[[str], list[int]], *, digits: dict[str, int],
                 marker: str = "<|audio_pad|>", count: int | None = None) -> Sequence:
    """A prompt's text (one ``marker``) with the adaptor's ``rows`` at the pads' positions: the texts around it
    tokenized as they are. ``count``: the span's tokens, the rows' count by default."""
    before, sep, after = text.partition(marker)
    if not sep:
        raise ValueError(f"the text has no {marker}")
    return Sequence.join([encode(before), audio_sequence(rows, digits=digits, count=count), encode(after)])


# -- the samples and the log-mel


def fit_chunk(samples: np.ndarray) -> np.ndarray:
    """One 30 s chunk: the samples zero-padded to 480,000, or cut to them (the processor's pad-or-trim, before its STFT:
    a longer recording is one chunk per call, a shorter one padded)."""
    x = np.asarray(samples, np.float32).reshape(-1)
    if x.size == N_SAMPLES:
        return x
    out = np.zeros(N_SAMPLES, np.float32)
    out[:min(x.size, N_SAMPLES)] = x[:N_SAMPLES]
    return out


def mel_filters(n_mel: int = N_MEL, n_fft: int = N_FFT, sr: int = SR, fmax: float = 8000.0) -> np.ndarray:
    """(n_mel, 1 + n_fft / 2) Slaney-normalised mel bank, the matrix ``transformers.audio_utils.mel_filter_bank`` builds
    for ``WhisperFeatureExtractor`` (norm / mel_scale "slaney", 0 .. 8,000 Hz, f_sp 200/3, logstep 27 / ln 6.4): triangles
    over the FFT bins' Hz frequencies, then the area normalisation 2 / (f[i + 2] - f[i])."""
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, 27.0 / np.log(6.4)

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, np.float64)
        return np.where(f < min_log_hz, 3.0 * f / 200.0, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) * logstep)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, np.float64)
        return np.where(m < min_log_mel, 200.0 * m / 3.0, min_log_hz * np.exp((m - min_log_mel) / logstep))

    freqs = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mel + 2))
    fft_freqs = np.linspace(0.0, sr // 2, 1 + n_fft // 2)
    slopes = freqs[None, :] - fft_freqs[:, None]  # (bins, n_mel + 2)
    diff = np.diff(freqs)
    bank = np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]))
    bank *= (2.0 / (freqs[2:] - freqs[:-2]))[None, :]  # (Slaney's area normalisation)
    return bank.T.astype(np.float32)


def log_mel(samples: np.ndarray, *, n_fft: int = N_FFT, hop: int = HOP, n_mel: int = N_MEL) -> np.ndarray:
    """(80, 3000) log-mel of one 30 s chunk (``fit_chunk``'s samples), ``WhisperFeatureExtractor``'s steps: the periodic
    Hann window of 400, the centred STFT (reflect padding), its last frame dropped, |.|^2, the mel filters, the clamp at
    1e-10 -> log10, the floor at the chunk's global max - 8, then (x + 4) / 4."""
    x = np.asarray(samples, np.float32).reshape(-1)
    if x.size != N_SAMPLES:
        raise ValueError(f"{x.size} samples: a chunk is {N_SAMPLES} (``fit_chunk``)")
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft, dtype=np.float32) / n_fft)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop]  # (3,001, 400)
    spec = np.fft.rfft((frames * win).astype(np.float64), axis=-1)  # (3,001, 201), float64
    mag = (spec.real ** 2 + spec.imag ** 2)[:-1]  # (3,000, 201): the last frame dropped
    mel = mag @ mel_filters(n_mel, n_fft).T.astype(np.float64)  # (3,000, 80)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).T.astype(np.float32)  # (80, 3,000)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """nn.LayerNorm over the last axis, float64 statistics (the encoder's final norm, eps 1e-5, on the host)."""
    xd = np.asarray(x, np.float64)
    mu = xd.mean(-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(-1, keepdims=True)
    return ((xd - mu) / np.sqrt(var + eps) * np.asarray(w, np.float64) + np.asarray(b, np.float64)).astype(np.float32)


def splice_clips(text: str, clips: list[Sequence], encode: Callable[[str], list[int]],
                 *, marker: str = "<|audio_pad|>") -> Sequence:
    """A prompt's text (one ``marker`` per clip) with each clip in the i-th marker's place: the texts around the markers
    tokenized as they are, the clips (``audio_sequence``: a pad per row) between them."""
    parts = text.split(marker)
    if len(parts) != len(clips) + 1:
        raise ValueError(f"{len(clips)} audio clip(s) for {len(parts) - 1} {marker} place(s) in the prompt")
    out: list[Sequence | list[int]] = [encode(parts[0])]
    for clip, t in zip(clips, parts[1:], strict=True):
        out += [clip, encode(t)]
    return Sequence.join(out)
