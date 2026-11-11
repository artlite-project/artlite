"""Live ASR (models/lfm3a/live.py): the encoder's stream of rows as the samples arrive against the clip encoded at once
(the band in progress run again with more frames: its rows to bfp16 rounding the band's; complete bands carried on),
also with a clip encoded between its chunks (the stream's state restored); the live transcript of LibriSpeech clips
against llama.cpp's live mode (llama-server's ``/v1/realtime?intent=transcription``, Vulkan)."""

import os
import re

import numpy as np
import pytest

from artlite.model_config import lfm3a_gguf, lfm3a_mmproj
from models.lfm3a import host as A
from models.lfm3a.live import CHUNK_ROWS, Stream

from . import ref as R
from .test_encoder import rel_rows
from .test_ref import clip

MMPROJ = lfm3a_mmproj()


@pytest.fixture(scope="module")
def audio() -> object:
    if not os.path.exists(MMPROJ):
        pytest.skip(f"{MMPROJ} missing")
    from models.lfm3a.npu import NpuAudio
    return NpuAudio(MMPROJ)


def stream_rows(st: Stream, x: np.ndarray, piece: int, *, between: object = None) -> np.ndarray:
    """x pushed ``piece`` samples at a time, a chunk's rows taken as soon as they are whole, the rest at the end;
    ``between``: called after each chunk (a clip through the encoder)."""
    st.reset()
    out = []
    for i in range(0, len(x), piece):
        st.push(x[i:i + piece])
        while st.available() >= CHUNK_ROWS:
            out.append(st.take(CHUNK_ROWS))
            if between is not None:
                between()
    st.end()
    if st.available():
        out.append(st.take(st.available()))
    return np.concatenate(out)


@pytest.mark.parametrize("name", ["6930-75918-0001"])
def test_stream(audio: object, name: str) -> None:
    """A clip's rows as a stream of 0.1 s pieces: the clip's at once, but for the rows of the 8-key blocks a chunk's
    end cuts (their V^T's bfp16 rounding: ~2% from the clip's), as far from float64 as the clip's."""
    x = A.load_audio(clip(name))
    whole = audio.encoder(A.log_mel(x))
    got = stream_rows(Stream(audio.encoder), x, A.SR // 10)
    assert got.shape == whole.shape
    r = rel_rows(got, whole)
    cut = np.zeros(len(r), bool)  # (the rows of the chunks' ends' blocks: every other chunk of 12 ends inside one)
    for e in range(CHUNK_ROWS, len(r), CHUNK_ROWS):
        cut[e - e % 8:e] = True
    print(f"stream vs whole: {np.count_nonzero(r)} of {len(r)} rows differ, max {r.max():.4f}")
    assert not r[~cut].any() and r.max() < 0.04, r.max()
    want = R.Encoder(MMPROJ).forward(A.log_mel(x).astype(np.float64)).numpy()
    rw, rs = rel_rows(whole, want), rel_rows(got, want)
    print(f"vs float64: the clip's median {np.median(rw):.4f}, max {rw.max():.4f}; the stream's {np.median(rs):.4f}, "
          f"{rs.max():.4f}")
    assert np.median(rs) < np.median(rw) + 1e-3 and rs.max() < 0.05


def test_stream_bands(audio: object) -> None:
    """The chapter (179 s: 5 bands) in pieces of 1 s, with a clip encoded after every 25th chunk (the stream's state
    restored): the same rows as without; the clip's at once (``test_stream``)."""
    x = np.concatenate([A.load_audio(clip(f"6930-75918-{i:04d}")) for i in range(21)])
    short = A.load_audio(clip("6930-75918-0012"))
    whole = audio.encoder(A.log_mel(x))
    st = Stream(audio.encoder)
    got = stream_rows(st, x, A.SR)
    assert got.shape == whole.shape and st.band == 4
    r = rel_rows(got, whole)
    print(f"stream vs whole: {np.count_nonzero(r)} of {len(r)} rows differ, max {r.max():.4f}")
    assert np.median(r) == 0 and r.max() < 0.04, r.max()
    n = [0]

    def between() -> None:
        n[0] += 1
        if n[0] % 25 == 0:
            audio.encoder(A.log_mel(short))

    assert np.array_equal(stream_rows(st, x, A.SR, between=between), got)


def words_wer(want: str, got: str) -> float:
    """Word errors (substitutions, insertions, deletions) over want's words, case and punctuation aside."""
    a, b = (re.sub(r"[^\w' ]", " ", t.lower()).split() for t in (want, got))
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1] / max(len(a), 1)


@pytest.mark.slow  # (the whole model)
def test_live() -> None:
    """``LiveAsr`` on clips pushed 0.1 s at a time: llama.cpp's live transcripts but for a word in 20."""
    from models.lfm.lfm import Lfm
    from models.lfm3a.live import LiveAsr

    for p in (lfm3a_gguf(), MMPROJ):
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")
    app = Lfm(lfm3a_gguf(), mmproj=MMPROJ)
    asr = LiveAsr(app, app.load(capacity=2048), app.audio.encoder)
    for name, want in LIVE.items():
        x = A.load_audio(clip(name))
        asr.start()
        text = "".join(asr.push(x[i:i + A.SR // 10]) for i in range(0, len(x), A.SR // 10)) + asr.end()
        print(f"{name}: {text!r}")
        assert words_wer(want, text) <= 0.05


LIVE = {  # (llama.cpp's live transcripts: llama-server's /v1/realtime?intent=transcription, Vulkan)
    "6930-75918-0001": "The English voted to the French baskets of flowers, of which they had made a plentiful provision to "
                       "greet the arrival of the young princess. The French, in return, invited the English to a supper, "
                       "which was to be given the next day.",
    "6930-75918-0012": "It is annoyance then.",
}
