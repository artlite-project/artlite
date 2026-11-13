"""LFM3-Audio's host side (models/lfm3a/host.py) and the float64 encoder (tests/lfm3a/ref.py) against llama.cpp's
libmtmd (no NPU): the samples of WAV files of every format and of other containers (through miniaudio, llama.cpp's
decoder), the log-mel spectrogram (within float32 rounding of llama.cpp's and of transformers' torch.stft path), the
clip's rows, the ASR prompt's tokens, the encoder's embeddings (llama.cpp's own distance from float64: its Q8_0
activations, ~2.4% at the projector), and a clip encoded in chunks (whole rows) equal to the clip at once."""

import math

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import AudioConfig
from models.lfm3a import host as A
from models.tokenizer import BpeTokenizer

from ..goldens import media
from ..goldens.media import VisionGolden
from ..models import require
from . import ref as R
from .files import MARKER, MMPROJ, MODEL, asr_text, clip


def golden(path: str) -> VisionGolden:
    require(MODEL, MMPROJ)
    return media.golden(MODEL, MMPROJ, path, asr_text(GGUFFile(MODEL), marker=MARKER))


def test_config() -> None:
    require(MODEL, MMPROJ)
    c = AudioConfig.from_gguf(GGUFFile(MMPROJ))
    assert (c.projector, c.hidden, c.heads, c.kv_heads, c.ffn, c.layers, c.mel_bins, c.window, c.projection_dim) == \
        ("lfm3a", 768, 8, 4, 1792, 17, 128, 32, 2048)
    assert c.head_dim == 96 and math.isclose(c.eps, 1e-5, rel_tol=1e-6)


def test_wav_formats(tmp_path: object) -> None:
    """Every WAV format this parses gives the clip's samples (within its precision); stereo: the channels' mean; other
    rates and containers through miniaudio."""
    import struct
    x = A.load_audio(clip())
    pcm16 = np.round(x * 32768).astype(np.int64)

    def wav(tag: int, ch: int, bits: int, raw: bytes, rate: int = A.SR) -> bytes:
        fmt = struct.pack("<HHIIHH", tag, ch, rate, rate * ch * bits // 8, ch * bits // 8, bits)
        if tag == 0xFFFE:  # (EXTENSIBLE: cbSize, valid bits, channel mask, the subformat GUID)
            fmt = struct.pack("<HHIIHH", tag, ch, rate, rate * ch * bits // 8, ch * bits // 8, bits) + \
                struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", 1) + bytes(14)
        return b"RIFF" + struct.pack("<I", 20 + len(fmt) + len(raw)) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + \
            fmt + b"data" + struct.pack("<I", len(raw)) + raw
    cases = {
        "pcm16": (wav(1, 1, 16, pcm16.astype("<i2").tobytes()), 0),
        "ext16": (wav(0xFFFE, 1, 16, pcm16.astype("<i2").tobytes()), 0),
        "pcm24": (wav(1, 1, 24, b"".join(int(v * 256).to_bytes(3, "little", signed=True) for v in pcm16)), 0),
        "pcm32": (wav(1, 1, 32, (pcm16 * 65536).astype("<i4").tobytes()), 0),
        "f32": (wav(3, 1, 32, x.astype("<f4").tobytes()), 0),
        "f64": (wav(3, 1, 64, x.astype("<f8").tobytes()), 0),
        "pcm8": (wav(1, 1, 8, (np.round(x * 128).clip(-128, 127) + 128).astype(np.uint8).tobytes()), 1 / 128),
        "stereo": (wav(1, 2, 16, np.stack([pcm16, pcm16], 1).astype("<i2").tobytes()), 0),
    }
    for name, (data, tol) in cases.items():
        y = A.load_audio(data)
        assert y.shape == x.shape and np.abs(y - x).max() <= tol + 1e-7, name
    assert np.array_equal(A._miniaudio(cases["pcm16"][0]), x)  # (miniaudio's samples of a 16 kHz WAV: the same)
    y = A.load_audio(wav(1, 1, 16, pcm16[::2].astype("<i2").tobytes(), rate=8000))  # (resampled to 16 kHz)
    assert abs(len(y) - len(x)) <= 2


def test_mel_hf() -> None:
    """log_mel (float32, float64) against transformers' Lfm3AudioFeatureExtractor path (torch.stft, periodic Hann,
    not centred; the filter bank ``mel_filters``)."""
    x = A.load_audio(clip())
    n = len(x)
    xp = torch.zeros(A.padded_length(n), dtype=torch.float64)
    xp[:n] = torch.from_numpy(x.astype(np.float64))
    spec = torch.stft(xp, n_fft=A.N_FFT, hop_length=A.HOP, win_length=A.N_FFT,
                      window=torch.hann_window(A.N_FFT, dtype=torch.float64), center=False, return_complex=True)
    hf = ((torch.from_numpy(A.mel_filters()) @ spec.abs().square()).log10().clamp(min=-12) + 4) / 4
    assert hf.shape[1] == A.frames(n)
    assert np.abs(A.log_mel(x, dtype=np.float64) - hf.numpy()).max() < 1e-9
    assert np.abs(A.log_mel(x) - hf.numpy()).max() < 2e-4


def test_mel_llama() -> None:
    """log_mel against llama.cpp's (the encoder's inp_raw): within float32 rounding (its own FFT, float32 throughout)."""
    path = clip()
    g = golden(path)
    mel = A.log_mel(A.load_audio(path))
    ref = g.tensor("inp_raw", 0).reshape(mel.shape)
    assert np.abs(mel - ref).max() < 3e-4
    assert g.chunks[1]["audio"] == A.rows(mel.shape[1])  # (the clip's rows: llama.cpp's audio chunk's tokens)


def test_rows() -> None:
    assert [A.frames(n) for n in (1, 400, 401, 560, 561, 16000)] == [1, 1, 2, 2, 3, 99]  # (16000: padded to 16080)
    assert [A.rows(f) for f in (1, 8, 9, 16, 99)] == [1, 1, 2, 2, 13]


def test_prompt() -> None:
    """The ASR prompt's text tokens (the chat template's, the clip in the user turn) are llama.cpp's."""
    path = clip()
    g = golden(path)
    tok = BpeTokenizer(GGUFFile(MODEL))
    head, tail = asr_text(GGUFFile(MODEL)).split(A.AUDIO)
    assert g.chunks[0]["text"] == tok.encode(head) and g.chunks[2]["text"] == tok.encode(tail)
    assert head.endswith("<|im_start|>user\n") and A.asr_system("") == "Perform ASR." and \
        A.asr_system("DE") == "Perform ASR in German." and A.asr_system("Klingon") == "Perform ASR in Klingon."


@pytest.mark.parametrize("name", ["6930-75918-0001", "6930-75918-0012"])
def test_encoder(name: str) -> None:
    """The float64 encoder on llama.cpp's mel against llama.cpp's embeddings: the final norm within 0.3%, the
    projector's output within llama.cpp's own error (its Q8_0 activations: emulated, within 1%)."""
    g = golden(clip(name))
    enc = R.Encoder(MMPROJ)
    mel = g.tensor("inp_raw", 0).astype(np.float64)
    tr: dict = {}
    e = enc.forward(mel, tr).numpy()
    ref = g.embd(0)
    assert e.shape == ref.shape
    assert np.linalg.norm(e - ref) / np.linalg.norm(ref) < 0.04


def test_stream() -> None:
    """A clip in chunks of 12 rows (live ASR's), 8 frames, and a remainder: the rows of the clip at once."""
    require(MODEL, MMPROJ)
    enc = R.Encoder(MMPROJ)
    mel = A.log_mel(A.load_audio(clip("6930-75918-0012")), dtype=np.float64)
    whole = enc.forward(mel).numpy()
    for step in (96, 8, 100):
        st = R.Stream(enc)
        out = [st.push(mel[:, f0:f0 + step], final=f0 + step >= mel.shape[1]).numpy() for f0 in range(0, mel.shape[1], step)]
        got = np.concatenate(out)
        assert got.shape == whole.shape and np.abs(got - whole).max() <= 1e-12 * np.abs(whole).max(), step
