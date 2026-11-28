"""Hugging Face checkpoint directories as model files (``artlite.hf_model``) against llama.cpp's own conversion of each
(``tests/goldens/hf.py``, made once; skipped without it): every metadata key of the converted GGUF but general.* (the
architecture and the sampling defaults kept) and the decode option llama.cpp alone reads (``tokenizer.ggml.clean_spaces``),
every tensor equal as float32 (a Llama's Q / K rows in the checkpoint's rotate-half order: ``HfModel.unpermuted``;
Qwen3.5's A = -exp(A_log) within an ulp: another exp); the virtual GGUF's tokenizer against the checkpoint's
tokenizer.json (gigatoken; Gemma 4: its tokenizer on the converted file) on the essay corpus and the TEXTS of
test_tokenizer.py, its chat prompt against the converted file's (no NPU). On the NPU (slow) a checkpoint gives the same teacher-forced logits as its converted file."""

import gigatoken
import numpy as np
import pytest

from artlite.gguf_model import GGUFFile, bpe_tokenizer_json, unpermute_rows
from artlite.hf_model import HfModel, has_vision_tower, is_checkpoint, open_mmproj, open_weights
from artlite.model_config import AudioConfig, LlamaConfig, checkpoint
from models import architecture, open_model
from models.llama.tokenizer import LLAMA3_PATTERN
from models.templating import render_chat
from models.tokenizer import BpeTokenizer

from .common import essays
from .goldens.hf import converted
from .model_files import require
from .test_tokenizer import MESSAGES, TEXTS

CASES = ["qwen3-0.6B", "qwen3.5-0.8B", "llama-1B", "gemma-4-E2B"]
SKIP_KEYS = ("general.", "tokenizer.ggml.clean_spaces")
KEEP_KEYS = ("general.architecture", "general.sampling.")
_FILES: dict[str, tuple[HfModel, GGUFFile]] = {}


def _files(case: str) -> tuple[HfModel, GGUFFile]:
    if case not in _FILES:
        require(checkpoint(case), converted(case))
        _FILES[case] = HfModel(checkpoint(case)), GGUFFile(converted(case))
    return _FILES[case]


@pytest.mark.parametrize("case", CASES)
def test_metadata(case: str) -> None:
    hf, gg = _files(case)
    assert hf.arch == gg.arch == architecture(checkpoint(case))
    assert is_checkpoint(checkpoint(case)) and isinstance(open_weights(checkpoint(case)), HfModel)
    want = {k for k in [*gg.meta, *gg._string_arrays] if not k.startswith(SKIP_KEYS) or k.startswith(KEEP_KEYS)}
    got = {*hf.meta, *hf._strings}
    assert want == got, (sorted(want - got), sorted(got - want))
    for k in sorted(want):
        if k in gg._string_arrays:
            assert hf.strings(k) == gg.strings(k), k
        elif isinstance(gg.meta[k], np.ndarray):
            assert np.array_equal(hf.meta[k], gg.meta[k]) and hf.meta[k].dtype == gg.meta[k].dtype, k
        elif isinstance(gg.meta[k], float):
            assert np.float32(hf.meta[k]) == np.float32(gg.meta[k]), (k, hf.meta[k], gg.meta[k])
        else:
            assert hf.meta[k] == gg.meta[k], (k, hf.meta[k], gg.meta[k])


@pytest.mark.parametrize("case", CASES)
def test_tensors(case: str) -> None:
    hf, gg = _files(case)
    assert set(hf.tensor_info) == set(gg.tensor_info)
    heads = {"attn_q": hf.hparams["num_attention_heads"], "attn_k": hf.hparams["num_key_value_heads"]}
    for name, (dims, qtype, _, _) in gg.tensor_info.items():
        assert hf.tensor_info[name][:2] == (dims, qtype), (name, hf.tensor_info[name][:2], (dims, qtype))
        want, got = gg.tensor(name).dequantize(), hf.tensor(name).dequantize()
        kind = name.split(".")[-2]
        if hf.arch == "llama" and kind in heads:
            want = want[unpermute_rows(heads[kind], want.shape[0])]
        if name.endswith("ssm_a"):
            assert np.abs(want.view(np.int32) - got.view(np.int32)).max() <= 1, name
        else:
            assert np.array_equal(want, got), name


@pytest.mark.parametrize("case", CASES)
def test_tokenizer(case: str) -> None:
    """The virtual GGUF's vocabulary tokenizes as the checkpoint's tokenizer.json."""
    hf, gg = _files(case)
    with open(f"{checkpoint(case)}/tokenizer.json", "rb") as f:
        ref = gigatoken.Tokenizer.from_json(f.read())
    if hf.arch == "llama":
        mine = gigatoken.Tokenizer.from_json(bpe_tokenizer_json(hf, LLAMA3_PATTERN, ignore_merges=True))
        encode = lambda s: mine.encode(s).tolist()
    elif hf.arch == "gemma4":  # (its own tokenizer on the metadata: against the converted file's)
        from models.gemma4.tokenizer import GemmaTokenizer

        encode, theirs = GemmaTokenizer(hf).encode, GemmaTokenizer(gg).encode
        for text in [*TEXTS, essays()[:400_000]]:
            assert encode(text) == theirs(text), text[:80]
        assert render_chat(hf, MESSAGES) == render_chat(gg, MESSAGES)
        return
    else:
        encode = BpeTokenizer(hf).encode
    for text in [*TEXTS, essays()[:400_000]]:
        assert encode(text) == ref.encode(text).tolist(), text[:80]
    assert render_chat(hf, MESSAGES) == render_chat(gg, MESSAGES)


def test_moss() -> None:
    """MOSS-Transcribe-Diarize's checkpoint (llama.cpp converts none): its language model as architecture ``moss`` with
    Qwen3's keys and tensor names, the output weight the tokens' embedding, the audio span's keys of its processor;
    its audio tower as an mmproj of gguf-py's Whisper encoder names, a conv's filters in the file's order."""
    path = checkpoint("moss-transcribe-diarize")
    require(path)
    hf = HfModel(path)
    assert hf.arch == architecture(path) == "moss" and not has_vision_tower(path)
    cfg = LlamaConfig.from_gguf(hf)
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            cfg.qk_norm) == (28, 1024, 16, 8, 128, True)
    assert (hf.meta["moss.audio.tokens_per_second"], hf.meta["moss.audio.time_marker_seconds"]) == (12.5, 5)
    assert all(n.startswith(("blk.", "token_embd.", "output")) for n in hf.tensor_info) and len(hf.tensor_info) == 311
    assert hf.tensor_info["output.weight"] == hf.tensor_info["token_embd.weight"]
    assert np.shares_memory(hf.tensor("output.weight").raw, hf.tensor("token_embd.weight").raw)
    assert "<|audio_pad|>" in render_chat(hf, [{"role": "user", "content": [{"type": "audio", "audio": ""}]}])
    mm = open_mmproj(path)
    a = AudioConfig.from_gguf(mm)
    assert (a.projector, a.hidden, a.ffn, a.layers, a.heads, a.mel_bins, a.projection_dim) == \
        ("moss", 1024, 4096, 24, 16, 80, 1024)
    assert abs(a.eps - 1e-5) < 1e-12 and mm.meta["clip.audio.projector.stack_factor"] == 4
    assert mm.meta["clip.audio.max_pos_emb"] == 1500 and abs(mm.meta["clip.audio.projector.layer_norm_epsilon"] - 1e-6) < 1e-12
    names = set(mm.tensor_info)
    per_layer = {f"a.blk.5.{t}.{k}" for t in ("attn_q", "attn_v", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2")
                 for k in ("weight", "bias")} | {"a.blk.5.attn_k.weight"}
    assert per_layer <= names and "a.blk.5.attn_k.bias" not in names and len(names) == 24 * 15 + 13
    assert {"a.conv1d.1.weight", "a.conv1d.2.bias", "a.position_embd.weight", "a.post_ln.weight", "mm.a.mlp.0.weight",
            "mm.a.mlp.2.bias", "mm.a.mlp.3.weight"} <= names
    assert mm.tensor_info["a.conv1d.2.weight"][:2] == ((3, 1024, 1024), "BF16")
    w = hf.hf_f32("model.whisper_encoder.conv1.weight")  # ((out, in, taps): a filter's elements in the file's order)
    assert np.array_equal(mm.tensor("a.conv1d.1.weight").dequantize().reshape(w.shape), w)


@pytest.mark.slow
@pytest.mark.parametrize("case", [c for c in CASES if c != "gemma-4-E2B"])  # (E2B's double-wide MLP: not run)
def test_npu_logits(case: str) -> None:
    """A checkpoint and its converted file: the same weight image and logits (Qwen3.5: its A within an ulp, the logits
    within 1e-3 of their range)."""
    require(checkpoint(case), converted(case))
    text = "Photosynthesis is the process by which plants convert light energy into chemical energy."
    out = []
    for path in (checkpoint(case), converted(case)):
        model = open_model(path)
        seq = model.chat([{"role": "user", "content": "Explain photosynthesis."}]) + model.encode(text)
        out.append(model.load(capacity=256).forced_logits(seq, len(seq) - 24, batch=1))
    if case.startswith("qwen3.5"):
        assert np.abs(out[0] - out[1]).max() < 1e-3 * np.ptp(out[1])
        assert (out[0].argmax(-1) == out[1].argmax(-1)).all()
    else:
        assert np.array_equal(out[0], out[1])
