"""The files of MOSS-Transcribe-Diarize's tests: the checkpoint's directory and its speech clips (LibriSpeech test-clean's
first chapter, the corpus the speech front ends' tests read; ``tests/datasets/fetch_datasets.py`` fetches it)."""

import os

import numpy as np
import pytest

from models.lfm3a.host import SR, load_audio

CLIPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "librispeech")


def chapter() -> list[str]:
    """The chapter's clips in its order (an empty list without the corpus)."""
    if not os.path.isdir(CLIPS):
        return []
    return sorted(os.path.join(CLIPS, n) for n in os.listdir(CLIPS) if n.endswith(".wav"))


def samples(*, seconds: float = 30.0) -> np.ndarray:
    """The chapter's utterances back to back as one ``seconds`` long clip of 16 kHz mono samples (the model's one chunk:
    the tower pads or cuts to 480,000), its speakers' turns the chapter's own."""
    paths = chapter()
    if not paths:
        pytest.skip(f"{CLIPS} missing (tests/datasets/fetch_datasets.py fetches the speech clips)")
    need = int(seconds * SR)
    out: list[np.ndarray] = []
    got = 0
    for path in paths:
        x = np.asarray(load_audio(path), np.float32).reshape(-1)
        out.append(x)
        got += x.size
        if got >= need:
            break
    x = np.concatenate(out)
    if x.size < need:
        x = np.concatenate([x, np.zeros(need - x.size, np.float32)])
    return x[:need]
