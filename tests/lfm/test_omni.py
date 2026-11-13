"""LFM3-Omni-2B (models.lfm with LFM3-Omni's mmproj: its vision tower, speech encoder and speech generator).

Host (no NPU): the prompts' token ids are llama.cpp's (libmtmd's chunks through ``tests.minicpmv.ref.golden``: text ids,
each media chunk's tokens) for an image, an audio clip, an image with a spoken question, videos with and without a
soundtrack (Omni's layout: per pair its timestamp, its tokens, its piece of the soundtrack), an LFM3-VL-2B video (frames
only); the TTS and interleaved prompts' ids are llama.cpp's tokenization of their text. The encoders here are stand-ins
that give the real token counts (``models.lfm3vl.host``, ``models.lfm3a.host``) and zero rows."""

import os

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig, LfmConfig, lfm3omni_gguf, lfm3omni_mmproj, lfm3vl_gguf, lfm3vl_mmproj
from models.lfm.lfm import Lfm
from models.lfm3a import host as A
from models.lfm3a import speech as S
from models.lfm3vl import host as Q
from models.sequence import MEDIA, Sequence, load_image
from models.tokenizer import BpeTokenizer, chat_prompt

from ..common import ASTRONAUT, SPEECH_CLIP, video_file
from ..llama_cpp import llama_tokenize
from ..minicpmv.ref import golden

MODEL, MMPROJ = lfm3omni_gguf(), lfm3omni_mmproj()
MARKER = "<__media__>"


class _Vision:
    """The vision tower's token counts (``NpuVision``'s bounds and sizes), zero rows."""

    def __init__(self, mmproj: str, hidden: int):
        c = ClipConfig.from_gguf(GGUFFile(mmproj))
        self.min_tokens, self.max_tokens = Q.token_bounds(c.min_pixels, c.max_pixels)
        self.hidden = hidden

    def sequence(self, image: str) -> Sequence:
        px = Q.image_pixels(load_image(image), min_tokens=self.min_tokens, max_tokens=self.max_tokens)
        g = Q.Grid(px.shape[0] // 32, px.shape[1] // 32)
        return g.sequence(np.zeros((g.tokens, self.hidden), np.float32))

    def pairs(self, frames: np.ndarray) -> list[np.ndarray]:
        g = Q.Grid(frames.shape[1] // 32, frames.shape[2] // 32)
        return [np.zeros((g.tokens, self.hidden), np.float32) for _ in range(0, len(frames), 2)]


class _Audio:
    """The speech encoder's rows (``host.rows``), zero rows."""

    def __init__(self, hidden: int):
        self.hidden = hidden

    def embeddings(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((A.rows(A.frames(len(x))), self.hidden), np.float32)

    def sequence(self, clip: str | np.ndarray) -> Sequence:
        x = clip if isinstance(clip, np.ndarray) else A.load_audio(clip)
        return A.clip_sequence(self.embeddings(x))


def _host(model: str, mmproj: str, audio: bool) -> Lfm:
    """The model's host side (template, tokenizer, the prompt's layout) with stand-in encoders: no weights packed."""
    m = object.__new__(Lfm)
    m.gguf = GGUFFile(model)
    m.cfg = LfmConfig.from_gguf(m.gguf)
    m.tokenizer = BpeTokenizer(m.gguf)
    m.bos = int(m.gguf.meta["tokenizer.ggml.bos_token_id"])
    m.vision = _Vision(mmproj, m.cfg.hidden_size)
    m.audio = _Audio(m.cfg.hidden_size) if audio else None
    return m


def _text(m: Lfm, messages: list[dict[str, object]]) -> str:
    """The chat template's text of a conversation (``Lfm.chat``'s, before its media are spliced)."""
    bos = m.gguf.strings("tokenizer.ggml.tokens", m.bos)[0]
    return chat_prompt(m.gguf, messages, enable_thinking=False, bos_token=bos)


def _files(*paths: str) -> None:
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"{p} missing")


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("videos"))


# case -> (the user turn's parts, the files in order: "image", "audio" or a video name)
CASES = {
    "image": [("image", ASTRONAUT), ("text", "Describe the image.")],
    "audio": [("audio", SPEECH_CLIP)],
    "image+audio": [("image", ASTRONAUT), ("audio", SPEECH_CLIP)],
    "video-sound": [("video", "astronaut"), ("text", "Describe what you see and hear.")],
    "video-silent": [("video", "silent"), ("text", "What happens in the video?")],
    "video-odd": [("text", "Two videos:"), ("video", "odd-25fps"), ("video", "ntsc-start")],
    "vl-video": [("video", "big-1080p"), ("text", "What happens in the video?")],  # (LFM3-VL-2B: frames only)
}


@pytest.mark.parametrize("case", list(CASES))
def test_prompt_tokens(case: str, videos: str) -> None:
    """Artlite's prompt (``Lfm.chat``) has llama.cpp's token ids: its text tokens where llama.cpp's are, a media row
    where each of llama.cpp's media chunks' tokens is."""
    vl = case.startswith("vl-")
    model, mmproj = (lfm3vl_gguf(), lfm3vl_mmproj()) if vl else (MODEL, MMPROJ)
    _files(model, mmproj, SPEECH_CLIP)
    m = _host(model, mmproj, audio=not vl)
    content, files, mtmd = [], [], []
    for kind, what in CASES[case]:
        if kind == "text":
            content.append({"type": "text", "text": what})
            mtmd.append({"type": "text", "text": what})
            continue
        path = video_file(videos, what) if kind == "video" else what
        content.append({"type": kind, kind: path})
        files.append(path)
        mtmd.append({"type": "text", "text": MARKER})
    seq = m.chat([{"role": "user", "content": content}])
    # llama-server's prompt: the parts' text with the media marker (no newline around it), the template rendering it
    text = "".join(p["text"] for p in mtmd)
    prompt = _text(m, [{"role": "user", "content": text}])
    gold = golden(model, mmproj, files, prompt)
    want = [t for c in gold.chunks for t in c.get("text", [MEDIA] * (c.get("image", 0) + c.get("audio", 0)))]
    got = [int(t) for t in seq.ids] if isinstance(seq, Sequence) else seq
    assert got == want


@pytest.mark.parametrize("voice, language", [("female", "en"), ("male", "de")])
def test_tts_prompt(voice: str, language: str) -> None:
    """Omni's TTS prompt: the file's ``tts`` template (``<think></think>`` before ``<|audio_start|>``), llama.cpp's ids."""
    _files(MODEL)
    g = GGUFFile(MODEL)
    text = S.tts_prompt(g, "Hello there.", language=language, voice=voice)
    assert text.endswith("<|im_start|>assistant\n<think></think><|audio_start|>")
    assert BpeTokenizer(g).encode(text) == llama_tokenize(MODEL, text)


def test_interleaved_prompt() -> None:
    """A voice reply's prompt: the conversation with ``speech.INTERLEAVED`` as its system turn, the assistant turn opened
    by ``<think></think>`` (the template's default) and ``<|mixed_start|>``: llama.cpp's ids of that text."""
    _files(MODEL, MMPROJ)
    m = _host(MODEL, MMPROJ, audio=True)
    msgs = [{"role": "system", "content": S.INTERLEAVED}, {"role": "user", "content": "What is the capital of France?"}]
    text = _text(m, msgs) + S.MIXED_START
    assert text.endswith("<|im_start|>assistant\n<think></think><|mixed_start|>")
    mixed = m.gguf.strings("tokenizer.ggml.tokens").index(S.MIXED_START)
    assert m.chat(msgs) + [mixed] == llama_tokenize(MODEL, text)
