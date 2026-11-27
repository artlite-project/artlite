"""artlite-cli's MOSS-Transcribe path and artlite-server's transcription prompt, host only: a transcription goes
through ``Moss.transcribe`` (its official instruction as the user turn's text, its silent turn's first token forced)
rather than LFM3-Audio's ASR prompt and decode loop, the report line of a tower without the encoder's timing no longer
raises, and each clip is a transcript.

The NPU is stubbed (``StubNpu``: the decoder's four calls) and the tower a stand-in (``Rows``: one zero row per audio
token): the CLI's path with the whole model on the device is ``test_npu.py``'s (``--run-slow``).
"""

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from artlite.model_config import moss_dir
from models import cli
from models.cli import main
from models.lfm3a.host import load_audio, wav_bytes
from models.moss_tr.host import INSTRUCTION, audio_sequence, span_tokens
from models.moss_tr.model import Moss

from ..model_files import require


@pytest.fixture(scope="module")
def model() -> Moss:
    require(moss_dir())
    return cli.open_model(moss_dir())


class Rows:
    """The audio tower's place: the span's count rows of zeros (the checkpoint's 1,024 channels), the clip's samples
    ``host.span_tokens`` of them. No program, no device - the prompt's splice alone."""

    loaded = True

    def __init__(self) -> None:
        self.rows = np.zeros((375, 1024), np.float32)

    def sequence(self, clip: object, *, digits: dict[str, int]) -> object:
        samples = clip if isinstance(clip, np.ndarray) else load_audio(clip)  # type: ignore[arg-type]
        count = span_tokens(np.asarray(samples).size)
        return audio_sequence(self.rows[:count], digits=digits, count=count)


class StubNpu:
    """The decoder on the NPU's place: ``reset`` / ``prepare`` / ``forward`` / ``generate``, the logits one token's
    peak (never an end-of-sequence, so ``Moss.transcribe``'s guard does not fire), the generation ``token`` repeated."""

    DRAFTS = 0
    lookup = True
    dispatches = 0

    def __init__(self, token: int) -> None:
        self.token, self.capacity, self.position = token, 4096, 0
        self.t_npu = self.t_host = 0.0

    def describe(self) -> str:
        return "a stubbed decoder (no device)"

    def stats(self) -> str:
        return ""

    def prepare(self, prompt_len: int) -> None:
        del prompt_len

    def reset(self) -> None:
        self.position = 0

    def forward(self, tokens: list[int], *, progress: object = None) -> np.ndarray:
        del progress
        self.position += len(tokens)
        logits = np.zeros(151936, np.float32)
        logits[self.token] = 1.0
        return logits

    def generate(self, logits: np.ndarray, max_tokens: int, sampling: object, prompt: object = (), mask: object = None,
                 probs: object = None, *, drafts: int | None = None) -> object:
        del logits, sampling, prompt, mask, probs, drafts
        self.position += max(max_tokens, 0)
        return iter([self.token] * max(max_tokens, 0))


@pytest.fixture()
def clip(tmp_path: Path) -> str:
    """A one-second WAV (16 kHz mono): the span's 13 tokens."""
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes(np.zeros(16000, np.float32)))
    return str(path)


def test_audio_prompt_reports_a_tower_without_its_timing(
        model: Moss, clip: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """``--audio`` in a prompt (``-p``: not a transcription) with a model whose encoder reports no ``t_encode``
    (MOSS-Transcribe's ``Tower``): the report line names the clips, no ``AttributeError``."""
    npu = StubNpu(model.tokenizer.encode("a")[0])
    monkeypatch.setattr(model, "audio", Rows())
    monkeypatch.setattr(cli, "open_model", lambda *a, **kw: model)
    monkeypatch.setattr(Moss, "load", lambda self, **kw: npu)
    main(["-m", moss_dir(), "-p", "Describe the clip.", "--audio", clip, "-n", "1"])
    err = capsys.readouterr().err
    assert "1 audio clip(s) through the encoder\n" in err
    assert npu.position > 0  # (the prompt ran: the model's rows were spliced, the decoder called)


def test_transcription_prompt_is_the_instruction(model: Moss, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Moss.transcribe``'s prompt: the clip, then the official instruction as the user turn's text - no system
    message (LFM3-Audio's "Perform ASR." is not MOSS-Transcribe's prompt)."""
    seen: list[list[dict[str, object]]] = []

    def fake_chat(messages: list[dict[str, object]], **kw: object) -> list[int]:
        del kw
        seen.append(messages)
        return [1, 2, 3]

    monkeypatch.setattr(model, "chat", fake_chat)
    model.transcribe(StubNpu(model.tokenizer.encode("a")[0]), "clip.wav", max_tokens=4)
    assert seen == [[{"role": "user", "content": [{"type": "audio", "audio": "clip.wav"},
                                                  {"type": "text", "text": INSTRUCTION}]}]]


def test_cli_transcribes_each_clip_with_the_model(
        model: Moss, clip: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """``--audio`` alone (no -p / -f): one ``Moss.transcribe(npu, path)`` call a clip, its default instruction (the
    CLI passes no other) and ``-n``; each transcript on its own line."""
    second = tmp_path / "second.wav"
    second.write_bytes(wav_bytes(np.zeros(16000, np.float32)))
    npu = StubNpu(model.tokenizer.encode("a")[0])
    calls: list[tuple[str, dict[str, object]]] = []
    assert inspect.signature(Moss.transcribe).parameters["instruction"].default == INSTRUCTION  # (its prompt)

    def fake_transcribe(self: Moss, npu_: object, audio: object, **kw: object) -> str:
        del npu_
        calls.append((os.path.basename(str(audio)), kw))
        return f"[S01] 00:00.0 - {os.path.basename(str(audio))}"

    monkeypatch.setattr(Moss, "transcribe", fake_transcribe)
    monkeypatch.setattr(Moss, "load", lambda self, **kw: npu)
    monkeypatch.setattr(cli, "open_model", lambda *a, **kw: model)
    main(["-m", moss_dir(), "--audio", clip, "--audio", str(second), "-n", "8"])
    out = capsys.readouterr().out.splitlines()
    assert [name for name, _ in calls] == ["clip.wav", "second.wav"]
    assert [kw for _, kw in calls] == [{"max_tokens": 8}, {"max_tokens": 8}]
    assert out == ["[S01] 00:00.0 - clip.wav", "[S01] 00:00.0 - second.wav"]


def test_server_transcription_prompt() -> None:
    """artlite-server's transcription prompt: a model with its own transcription (MOSS-Transcribe) gets the official
    instruction as the user turn's text and no ASR system message; LFM3-Audio keeps llama.cpp's ("Perform ASR in ...")."""
    from models.server.routes_audio import transcription_messages

    class Owned:
        transcribe = object()  # (the model transcribes itself)

    class Lfm:
        pass  # (an LFM3-Audio model: the ASR system message)

    content = [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]
    m = transcription_messages(Owned(), "de", content)
    assert m == [{"role": "user", "content": content + [{"type": "text", "text": INSTRUCTION}]}]
    m = transcription_messages(Lfm(), "de", content)
    assert m == [{"role": "system", "content": "Perform ASR in German."}, {"role": "user", "content": content}]
