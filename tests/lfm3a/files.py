"""The files of LFM3-Audio's tests: the model and its mmproj, the speech clips (the ASR prompt's text, a clip's path), the
TTS text."""

import os

import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import lfm3a_gguf, lfm3a_mmproj
from models.lfm3a import host as A
from models.tokenizer import chat_prompt

CLIPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "librispeech")
MODEL, MMPROJ = lfm3a_gguf(), lfm3a_mmproj()
MARKER = "<__media__>"
TEXT = "Hello there, this is a test of the speech output."


def clip(name: str = "6930-75918-0001") -> str:
    path = os.path.join(CLIPS, f"{name}.wav")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing (tests/datasets/fetch_datasets.py)")
    return path


def asr_text(gguf: GGUFFile, language: str = "en", marker: str = A.AUDIO) -> str:
    """The ASR prompt's text by the file's chat template, the clip at ``marker``."""
    msgs = [{"role": "system", "content": A.asr_system(language)},
            {"role": "user", "content": [{"type": "audio", "audio": "clip"}]}]
    text = chat_prompt(gguf, msgs, enable_thinking=False, bos_token=gguf.strings("tokenizer.ggml.tokens",
                                                                                 gguf.meta["tokenizer.ggml.bos_token_id"])[0])
    return text.replace(A.AUDIO, marker)
