"""The files of MOSS-Transcribe-Diarize's tests: the checkpoint directory and the speech clips (LibriSpeech
test-clean's first chapter: ``tests/datasets/fetch_datasets.py``)."""

import os

import numpy as np

from artlite.model_config import checkpoint
from models.audio import SR, load_audio

from ..model_files import require

MODEL = checkpoint("moss-transcribe-diarize")
CLIPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "librispeech")


def clip(name: str = "6930-75918-0001") -> str:
    path = os.path.join(CLIPS, f"{name}.wav")
    require(path)
    return path


def chapter(seconds: float) -> np.ndarray:
    """The chapter's utterances back to back, its first ``seconds`` (16 kHz mono samples)."""
    out: list[np.ndarray] = []
    for i in range(21):
        out.append(load_audio(clip(f"6930-75918-{i:04d}")))
        if sum(map(len, out)) >= seconds * SR:
            break
    return np.concatenate(out)[:int(seconds * SR)]
