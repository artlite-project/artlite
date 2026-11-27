"""MOSS-Transcribe-Diarize on the NPU (``models.moss_tr.model.Moss``: the audio tower's four programs on the vision
overlay and the checkpoint's Qwen3-0.6B decoder): a 30 s clip through ``Moss.transcribe`` - the official instruction,
the tower's rows at the prompt's pads and the decoder's greedy answer - comes back as the model's timestamped,
speaker-tagged text.

The criterion is the format (non-empty, the speaker tags), not a word error rate: a CI holds no reference transcripts
to score against (the model page carries the measured rates), and a word the model mishears is not a broken path."""

import re

import numpy as np
import pytest

from artlite.model_config import moss_dir
from models.moss_tr.decoder import Decoder
from models.moss_tr.host import MERGE
from models.moss_tr.model import Moss

from ..model_files import require
from .files import samples

pytestmark = pytest.mark.slow  # (the whole model: the tower's four programs and a chunk's decode)

CAPACITY = 2048  # (the 414-token prompt and the answer over the audio span: 67.7 KB a position)


@pytest.fixture(scope="module")
def app() -> tuple[Moss, Decoder]:
    """The checkpoint as the app opens it, its tower and decoder on the NPU (one load for the module)."""
    require(moss_dir())
    model = Moss(moss_dir())
    return model, model.load(capacity=CAPACITY)


def test_transcribe(app: tuple[Moss, Decoder]) -> None:
    """A 30 s LibriSpeech clip: the transcript is non-empty, and its segments carry the start timestamp and the speaker
    tag the instruction asks for (``[S01]``: the chapter is one speaker) before their text."""
    model, npu = app
    text = model.transcribe(npu, samples())
    assert text.strip(), "an empty transcript"
    assert "[S01]" in text, text  # (the chapter's one speaker: the model's tag)
    assert re.search(r"\[\d+(?:\.\d+)?\]\[S01\]", text), text  # (a segment's start timestamp, then its tag)
    assert len(text.split()) > 10, text  # (words, not just the markers)


def test_tower_rows_follow_the_clip(app: tuple[Moss, Decoder]) -> None:
    """The tower's rows are the clip's own token count (``host.span_tokens``: ``(n - 1) // 1,280 + 1``, at most a 30 s
    chunk's 375), the adaptor's 1,024 channels: 5 s gives 63 rows, 30 s 375, all finite."""
    model, _ = app
    for seconds, count in ((5, 63), (30, 375)):
        rows = model.audio.rows(samples(seconds=seconds))
        assert rows.shape == (count, 1024), (seconds, rows.shape)
        assert np.isfinite(rows).all() and np.abs(rows).max() > 0
    assert 375 * MERGE == 1500  # (a 30 s chunk's encoder rows, four a merged row)
