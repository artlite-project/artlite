"""MOSS-Transcribe as an app: ``open_model`` on the checkpoint directory (architecture ``moss``), its capabilities, and
the prompt its audio parts make - the temperature-less (greedy) speech model of ``models.moss_tr.model``.

The tower's rows need the NPU (``models.moss_tr.tower.Tower``: four programs of the vision overlay), so this file
replaces it with a stand-in that returns one row per token: what is tested is the app's splice - the 375 pads of the span
each a ``given`` row (``MEDIA``'s id), the digits their text ids, the rows the tower's in order - and the metadata's
names the tower reads. The tower's own numbers are ``test_npu.py``'s (the whole model on the device), the mel's are
``test_host.py``'s.
"""

import numpy as np
import pytest

from artlite.hf_model import HfModel
from artlite.model_config import moss_dir
from models import architecture, open_model
from models.moss_tr.host import AUDIO_PAD, MERGE, audio_sequence, span_tokens
from models.moss_tr.model import Moss
from models.moss_tr.tower import Tower
from models.sequence import MEDIA
from models.tokenizer import BpeTokenizer

from ..model_files import require

MESSAGES = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "audio", "audio": "chunk.wav"},
                                         {"type": "text", "text": "Please transcribe this audio."}]}]


@pytest.fixture(scope="module")
def model() -> Moss:
    """The checkpoint directory as the app opens it (no NPU: the tower builds at ``load`` or the first clip)."""
    require(moss_dir())
    m = open_model(moss_dir())
    assert isinstance(m, Moss)
    return m


class Rows:
    """A stand-in tower: one row per audio token (the given rows' path, without the NPU's programs)."""

    loaded = True

    def __init__(self, m: Moss) -> None:
        self.tok = m.tokenizer
        self.rows = np.arange(375 * 1024, dtype=np.float32).reshape(375, 1024) * np.float32(1e-6)

    def sequence(self, clip: object, *, digits: dict[str, int], count: int = 375) -> object:
        del clip
        return audio_sequence(self.rows, digits=digits, count=count)


def test_architecture(model: Moss) -> None:
    """``open_model`` opens the checkpoint with ``models.moss_tr.model.Moss`` through the architecture ``moss`` (the
    metadata's keys are ``moss.*``; the language model's tensors keep the Qwen3 map)."""
    assert architecture(moss_dir()) == "moss" and model.gguf.arch == "moss"
    assert model.name == "MOSS-Transcribe" and model.turn == "<|im_start|>"
    assert model.tokenizer.encode("<|im_start|>user") == [151644, 872]  # (the template's turn marks)
    assert not model.thinks and not model.parallel
    assert model.cfg.num_hidden_layers == 28 and model.cfg.hidden_size == 1024 and model.cfg.qk_norm


def test_capabilities(model: Moss) -> None:
    """A speech model: audio parts in prompts, no vision, no speech replies, no live transcription (the checkpoint's
    template has no ASR entry)."""
    caps = model.capabilities
    assert caps.audio and not caps.vision and not caps.video and not caps.speech and not caps.live_asr
    assert isinstance(model.audio, Tower) and not model.audio.loaded  # (built at load, not at open)


def test_prompt_rows(model: Moss, monkeypatch: pytest.MonkeyPatch) -> None:
    """The template's ``<|audio_pad|>`` becomes the audio span: 386 ids of which the 375 pads are ``MEDIA`` tokens with one
    given row each (the tower's, in order) and the 11 time markers their text ids, in 7 runs of pads; the prompt is 414
    tokens for this conversation (the e2e's own text), the span between ``<|audio_start|>`` and ``<|audio_end|>``."""
    rows = Rows(model)
    monkeypatch.setattr(model, "audio", rows)
    seq = model.chat(MESSAGES)
    assert len(seq) == 414 and isinstance(seq.rows, np.ndarray)
    given = np.flatnonzero(seq.given)
    assert len(given) == 375 and np.array_equal(seq.rows, rows.rows)
    assert np.all(seq.ids[seq.given] == MEDIA)
    assert given[0] == 15 and int(seq.ids[14]) == 151669 and int(seq.ids[given[-1] + 1]) == 151670
    on = seq.given
    runs = np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))
    assert np.array_equal(runs, np.array([15, 78, 142, 206, 270, 334, 398]))  # (the 7 runs around the digits)
    span = seq.ids[15:15 + 386]
    digits, times = span[span != MEDIA], seq.rows.shape[0]
    assert times == 375 and len(digits) == 11 and int((seq.ids == AUDIO_PAD).sum()) == 0  # (the pads are MEDIA)


def test_prompt_audio_part(model: Moss, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Model.prompt``'s media path: a clip through ``audio=`` (its part before the text, as the template renders it)
    makes the same span; a prompt without media is the plain text's ids."""
    rows = Rows(model)
    monkeypatch.setattr(model, "audio", rows)
    seq = model.prompt("Please transcribe this audio.", audio=[b"the stand-in's rows: not read"])
    assert len(seq) == 414 and int(seq.given.sum()) == 375
    plain = model.prompt("hello")
    assert isinstance(plain, list) and len(plain) > 1 and plain[:1] == [151644]


def test_tower_clip(model: Moss, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tower's clip path (``Tower.sequence``): a WAV's bytes decoded to 16 kHz mono samples, then the span of the
    clip's own tokens (``span_tokens``: ``(n - 1) // 1,280 + 1``): 1 s gives 13 ids, 5 s 63 pads + the 5 s marker's digit
    = 64 ids, 30 s 386 - one given row a pad (the rows the NPU's, here a stand-in's)."""
    from models.lfm3a.host import wav_bytes
    from models.moss_tr.decoder import digits

    def fake_rows(self: Tower, samples: np.ndarray) -> np.ndarray:
        return np.zeros((span_tokens(np.asarray(samples).size), 1024), np.float32)  # (the adaptor's rows: one a token)

    monkeypatch.setattr(Tower, "rows", fake_rows)
    d = digits(model.tokenizer)
    for sec, want in ((1, 13), (5, 64), (30, 386)):
        seq = model.audio.sequence(wav_bytes(np.zeros(sec * 16000, np.float32)), digits=d)
        count = int(seq.given.sum())
        assert len(seq) == want and count == span_tokens(sec * 16000) and seq.rows.shape == (count, 1024)
    assert span_tokens(5 * 16000) + 1 == 64 and span_tokens(30 * 16000) + 11 == 386  # (the markers' digit ids)


def test_encoder_rows_are_the_spans() -> None:
    """The tower reads ``count * 4`` encoder rows of its 1,500 a chunk (the processor's slice of a chunk's features) and
    the four-row merge makes the adaptor's ``count`` rows: 2 s -> 100 rows -> 25, 5 s -> 252 -> 63, 15 s -> 752 -> 188,
    30 s -> 1,500 -> 375."""
    from models.moss_tr.adaptor import merged

    for sec, count in ((2, 25), (5, 63), (15, 188), (30, 375)):
        assert span_tokens(sec * 16000) == count
        assert merged(np.zeros((count * MERGE, 8), np.float32)).shape == (count, MERGE * 8)


def test_tower_tensors() -> None:
    """The tower's weights are the checkpoint's own names (the GGUF map does not know them): the stem's seven tensors,
    24 layers' fifteen each and the VQ adaptor's six - all in the file, read as float32."""
    require(moss_dir())
    g = HfModel(moss_dir())
    want = ["model.whisper_encoder.conv1.weight", "model.whisper_encoder.conv1.bias",
            "model.whisper_encoder.conv2.weight", "model.whisper_encoder.conv2.bias",
            "model.whisper_encoder.embed_positions.weight", "model.whisper_encoder.layer_norm.weight",
            "model.whisper_encoder.layer_norm.bias"]
    want += [f"model.whisper_encoder.layers.{i}.{n}" for i in range(24)
             for n in ("self_attn_layer_norm.weight", "self_attn_layer_norm.bias", "self_attn.q_proj.weight",
                       "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                       "self_attn.v_proj.bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                       "final_layer_norm.weight", "final_layer_norm.bias", "fc1.weight", "fc1.bias", "fc2.weight",
                       "fc2.bias")]
    want += [f"model.vq_adaptor.layers.{i}.{n}" for i in (0, 2, 3) for n in ("weight", "bias")]
    names = set(g.hf_names())
    assert set(want) <= names and len(want) == 373  # (24 x 15 + the stem 7 + the adaptor 6)
    conv = g.hf_f32("model.whisper_encoder.conv1.weight")
    assert conv.shape == (1024, 80, 3) and g.hf_f32("model.whisper_encoder.embed_positions.weight").shape == (1500, 1024)


def test_digits(model: Moss) -> None:
    """The time markers' digits are one token each (the processor refuses a split one)."""
    tok = BpeTokenizer(model.gguf)
    assert [len(tok.encode(d)) for d in "0123456789"] == [1] * 10


def test_transcribe_forces_a_silent_turns_first_token(model: Moss, monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn whose first token would be an end-of-sequence (the instruction's bf16 near-tie) gets the best other
    token forced, and transcribes on from it in the instruction's format."""
    from models.moss_tr.host import INSTRUCTION

    parts: list[list[str]] = []

    def fake_chat(messages, **kw):
        parts.append([p["type"] for p in messages[0]["content"]])
        return [1, 2, 3]

    monkeypatch.setattr(model, "chat", fake_chat)
    eos = min(model.eos_token_ids)
    other = 5

    class Npu:
        def __init__(self) -> None:
            import numpy as np

            self.calls = 0
            self.forced: list[list[int]] = []
            self._np = np

        def reset(self) -> None:
            pass

        def prepare(self, n: int) -> None:
            pass

        def forward(self, ids):
            self.calls += 1
            logits = self._np.zeros(151936, self._np.float32)  # (the vocabulary's size)
            if self.calls == 1:  # (the instruction's turn: EOS would win, '[' next)
                logits[eos], logits[other] = 10.0, 9.0
            else:  # (the forced token's turn: on from the forced one)
                self.forced.append(list(ids))
                logits[other] = 10.0
            return logits

        def generate(self, logits, n, sampling, ids):
            return [other]

    npu = Npu()
    text = model.transcribe(npu, "clip.wav")
    assert parts == [["audio", "text"]], parts  # (the instruction's turn only: the forced one transcribes)
    assert npu.forced == [[other]], npu.forced  # (the forced first token fed back)
    assert text == model.tokenizer.decode([other, other])
    assert INSTRUCTION.startswith("请将音频转写为文本")
