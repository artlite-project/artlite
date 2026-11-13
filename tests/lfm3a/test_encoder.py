"""LFM3-Audio's speech encoder on the NPU (models/lfm3a/encoder.py, npu.py) against the float64 encoder (tests/lfm3a/
ref.py) on LibriSpeech clips: per row the relative L2 distance (median, p90, max) of one band (1.9 s, 14 s) and of a
clip of 5 bands (the chapter: 179 s, the state carried between bands); the same clip again gives the same bytes, also
after a longer clip (no state left over)."""

import os

import numpy as np
import pytest

from models.lfm3a import host as A

from ..model_files import require
from ..numerics import rel_rows
from . import ref as R
from .files import CLIPS, MMPROJ, clip


@pytest.fixture(scope="module")
def audio() -> object:
    require(MMPROJ)
    from models.lfm3a.npu import NpuAudio
    return NpuAudio(MMPROJ)


@pytest.fixture(scope="module")
def enc() -> R.Encoder:
    return R.Encoder(MMPROJ)


@pytest.mark.parametrize("name", ["6930-75918-0012", "6930-75918-0001"])
def test_band(audio: object, enc: R.Encoder, name: str) -> None:
    """One band: each row within ~3% of float64 (median; the bfp16 operands of the GEMMs, the residual's bf16), every
    row within 5%."""
    x = A.load_audio(clip(name))
    got = audio.embeddings(x)
    want = enc.forward(A.log_mel(x).astype(np.float64)).numpy()
    assert got.shape == want.shape
    r = rel_rows(got, want)
    assert np.median(r) < 0.03 and np.percentile(r, 90) < 0.04 and r.max() < 0.05, (np.median(r), r.max())


def test_bands(audio: object, enc: R.Encoder) -> None:
    """The chapter's 21 utterances as one clip (2236 rows: 5 bands, the state carried): each row as one band's."""
    x = np.concatenate([A.load_audio(clip(f"6930-75918-{i:04d}")) for i in range(21)])
    got = audio.embeddings(x)
    want = enc.forward(A.log_mel(x).astype(np.float64)).numpy()
    assert got.shape == want.shape == (A.rows(A.frames(len(x))), 2048)
    r = rel_rows(got, want)
    assert np.median(r) < 0.03 and np.percentile(r, 90) < 0.04 and r.max() < 0.06, (np.median(r), r.max())


def test_repeat(audio: object) -> None:
    """A clip encoded again (the kept embeddings bypassed), also after a longer one, gives the same bytes."""
    short, long = A.load_audio(clip("6930-75918-0012")), A.load_audio(clip("6930-75918-0001"))
    e = audio.encoder
    a = e(A.log_mel(short))
    e(A.log_mel(long))
    assert np.array_equal(e(A.log_mel(short)), a)
    assert os.path.isdir(CLIPS)
