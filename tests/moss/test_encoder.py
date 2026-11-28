"""MOSS-Transcribe-Diarize's audio tower on the NPU (models/moss/encoder.py, npu.py) against the float64 tower
(tests/moss/ref.py) on LibriSpeech speech: per token the relative L2 distance of its row for a full 30 s chunk and for
a recording of two chunks (each with separate tokens); the same recording again gives the same bytes, also after
another one.

The distances: ~1.5% (median; the bfp16 operands of the GEMMs and the residual's bf16 over 24 layers: the model's own
dtype, torch bf16, is 0.7% from float64), and a few tokens much further: the encoder's layer 7 writes a massive
activation (~2500 in channels 584, 884, 985, where a row's other channels are ~2) into a few rows, and a row that
gets a little of it (~100) gets as much as the last bits of its input decide (torch bf16: one token of 375 8% off,
the overlay's bfp16 inputs: 33%)."""

import numpy as np
import pytest
import torch

from models.moss import host as H

from ..model_files import require
from ..numerics import rel_rows
from . import ref as R
from .files import MODEL, chapter


@pytest.fixture(scope="module")
def audio() -> object:
    require(MODEL)
    from models.moss.npu import NpuAudio
    return NpuAudio(MODEL)


@pytest.fixture(scope="module")
def enc() -> R.Encoder:
    return R.Encoder(MODEL)


def reference(enc: R.Encoder, x: np.ndarray) -> np.ndarray:
    """The float64 tower's rows of a recording: its chunks' tokens in order."""
    return torch.cat([enc.forward(torch.from_numpy(H.log_mel(c)), H.tokens(min(H.CHUNK, len(x) - i * H.CHUNK)))
                      for i, c in enumerate(H.chunks(x))]).numpy()


def test_chunk(audio: object, enc: R.Encoder) -> None:
    """A 30 s chunk's 375 rows: within ~1.5% of float64 (median), a few tokens at layer 7's threshold further (module
    docstring)."""
    x = chapter(30.0)
    got, want = audio.embeddings(x), reference(enc, x)
    assert got.shape == want.shape == (375, 1024)
    r = rel_rows(got, want)
    print(f"a chunk: median {np.median(r):.4f}, p90 {np.percentile(r, 90):.4f}, max {r.max():.4f}")
    assert np.median(r) < 0.02 and np.percentile(r, 90) < 0.03, (np.median(r), np.percentile(r, 90))
    assert (r > 0.1).mean() < 0.01 and r.max() < 0.6, np.sort(r)[-4:]  # (the tokens at layer 7's threshold)


def test_recording(audio: object, enc: R.Encoder) -> None:
    """A recording of 41.3 s (a chunk's 375 tokens and the next one's 142, read from its padded chunk's first rows):
    each row as one chunk's. Encoded again (the kept rows bypassed), also after another recording, it gives the same
    bytes."""
    x = chapter(41.3)
    got, want = audio.embeddings(x), reference(enc, x)
    assert got.shape == want.shape == (375 + 142, 1024)
    r = rel_rows(got, want)
    print(f"a recording: median {np.median(r):.4f}, p90 {np.percentile(r, 90):.4f}, max {r.max():.4f}")
    assert np.median(r) < 0.02 and np.percentile(r, 90) < 0.03, (np.median(r), np.percentile(r, 90))
    assert (r > 0.1).mean() < 0.01 and r.max() < 0.6, np.sort(r)[-4:]
    audio.results.clear()
    audio.embeddings(chapter(3.0))
    audio.results.clear()
    assert np.array_equal(audio.embeddings(x), got)
