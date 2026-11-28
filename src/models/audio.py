"""What the speech front ends' host sides share (LFM3-Audio's, MOSS-Transcribe-Diarize's): a clip's samples, a WAV file
of samples, a chat message's audio parts, and the log-mel spectrograms' filter bank and window.

Samples (``load_audio``): 16 kHz mono float32. A WAV file (PCM of 8, 16, 24 or 32 bits, IEEE float of 32 or 64,
WAVE_FORMAT_EXTENSIBLE) at 16 kHz is parsed here, its channels averaged; other containers (MP3, FLAC, Ogg Vorbis) and
rates go through the ``miniaudio`` package, the library llama.cpp's mtmd decodes and resamples with (the same
configuration: float32, one channel, 16 kHz), so they give llama.cpp's samples.

Mel (``mel_filters``, ``mel_spectrum``, ``hann``): transformers' ``mel_filter_bank`` (norm and scale "slaney", 0 .. 8000
Hz) for FFTs of 400 samples under a periodic Hann window: Whisper's front end, which both models' feature extractors
are.
"""

import functools
import struct

import miniaudio
import numpy as np

SR, N_FFT = 16000, 400

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

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, np.float64)
    return np.where(f < 1000, 3 * f / 200, 15 + np.log(np.maximum(f, 1e-9) / 1000) * 27 / np.log(6.4))


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    return np.where(m < 15, 200 * m / 3, 1000 * np.exp((m - 15) * np.log(6.4) / 27))


@functools.cache
def mel_filters(n_mel: int) -> np.ndarray:
    """The Slaney mel filter bank (n_mel, N_FFT / 2 + 1) float64 (transformers' ``mel_filter_bank``, norm and scale
    "slaney", 0 .. SR / 2)."""
    hz = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SR / 2), n_mel + 2))
    fft = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    d = np.diff(hz)
    slopes = hz[None, :] - fft[:, None]
    tri = np.maximum(0, np.minimum(-slopes[:, :-2] / d[:-1], slopes[:, 2:] / d[1:]))
    return (tri * (2 / (hz[2:] - hz[:-2]))[None, :]).T


@functools.cache
def _mel_bands(n_mel: int) -> list[tuple[int, int, np.ndarray]]:
    """Per mel filter its bins [lo, hi) and their weights (a filter is a triangle over a few bins)."""
    out = []
    for row in mel_filters(n_mel):
        nz = np.flatnonzero(row)
        lo, hi = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 1)
        out.append((lo, hi, np.ascontiguousarray(row[lo:hi])))
    return out


def mel_spectrum(power: np.ndarray, n_mel: int) -> np.ndarray:
    """A power spectrogram (frames, N_FFT / 2 + 1) through the mel filters: (n_mel, frames) in power's dtype. Each
    filter sums only the bins it covers, which equals ``mel_filters(n_mel) @ power.T`` without the bank's zeros."""
    out = np.empty((n_mel, len(power)), power.dtype)
    for i, (lo, hi, w) in enumerate(_mel_bands(n_mel)):
        out[i] = power[:, lo:hi] @ w.astype(power.dtype)
    return out


@functools.cache
def hann() -> np.ndarray:
    """The periodic Hann window of N_FFT samples, float64."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


# -- a chat message's audio parts

AUDIO_PARTS = ("audio", "input_audio", "audio_url")  # (a chat message's audio parts)


def part_audio(part: dict) -> str | bytes | np.ndarray:
    """The clip of a chat message's audio part: ``{"type": "audio", "audio": a file, its bytes or 16 kHz samples}``,
    OpenAI's ``{"type": "input_audio", "input_audio": {"data": base64, "format": ...}}`` or ``{"type": "audio_url",
    "audio_url": {"url": a data URL}}``; ValueError: none of them."""
    import base64

    from .sequence import data_bytes
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
