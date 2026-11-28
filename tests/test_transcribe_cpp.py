"""transcribe.cpp's GGUF files as model files (``artlite.transcribe_cpp``): MOSS-Transcribe-Diarize's Q4_K_M file opens
as the language model and the audio tower its checkpoint directory opens as (``artlite.hf_model``): the same tensor
names and shapes, keys and tokenizer, the values as the file's quantization rounds them (no NPU)."""

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.hf_model import open_mmproj, open_weights
from artlite.model_config import checkpoint, moss_gguf
from artlite.transcribe_cpp import is_stt
from models import architecture
from models.templating import render_chat
from models.tokenizer import BpeTokenizer

from .model_files import require


@pytest.fixture(scope="module")
def files() -> tuple:
    require(moss_gguf(), checkpoint("moss-transcribe-diarize"))
    path, ckpt = moss_gguf(), checkpoint("moss-transcribe-diarize")
    return open_weights(path), open_mmproj(path), open_weights(ckpt), open_mmproj(ckpt)


def test_language_model(files: tuple) -> None:
    """The language model: architecture ``moss``, the checkpoint's keys, tensors (the output weight the embedding's
    rows) and shapes, its chat template and tokenizer."""
    g, _, h, _ = files
    assert architecture(moss_gguf()) == g.arch == "moss" and is_stt(GGUFFile(moss_gguf()))
    assert {k: v for k, v in h.meta.items() if k.startswith("moss.")} == {k: v for k, v in g.meta.items()
                                                                          if k.startswith("moss.")}
    assert set(g.tensor_info) == set(h.tensor_info) and g.tensor_info["output.weight"] == g.tensor_info["token_embd.weight"]
    assert all(g.tensor_info[n][0] == h.tensor_info[n][0] for n in h.tensor_info)
    assert g.tensor_info["blk.0.attn_q.weight"][1] == "Q4_K" and g.tensor_info["token_embd.weight"][1] == "Q6_K"
    a, b = g.rows("token_embd.weight", np.arange(64)), h.rows("token_embd.weight", np.arange(64))
    assert np.abs(a - b).max() < 0.03 * np.abs(b).max()
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": ""}, {"type": "text", "text": "x"}]}]
    text = render_chat(g, msgs) + "[0.19][S01] Hello, 世界.[4.85]"
    assert render_chat(g, msgs) == render_chat(h, msgs) and BpeTokenizer(g).encode(text) == BpeTokenizer(h).encode(text)


def test_tower(files: tuple) -> None:
    """The audio tower: the checkpoint's mmproj names, shapes and keys (eps within float32's rounding)."""
    _, m, _, hm = files
    assert m.arch == "clip" and set(m.tensor_info) == set(hm.tensor_info)
    assert all(m.tensor_info[n][0] == hm.tensor_info[n][0] for n in hm.tensor_info)
    for k, v in hm.meta.items():
        if k.startswith("clip."):
            assert m.meta[k] == pytest.approx(v, rel=1e-6) if isinstance(v, float) else m.meta[k] == v, k
    for name in ("a.conv1d.2.weight", "a.position_embd.weight", "mm.a.mlp.3.bias"):  # (kept as F32 in the file)
        assert np.abs(m.tensor(name).dequantize() - hm.tensor(name).dequantize()).max() < 1e-6, name
