"""transcribe.cpp's GGUF files as model files: a speech model in one file (its ``stt.*`` metadata; handy-computer's
conversions) opens as the files the models read, with the names and keys ``hf_model`` gives the same model's checkpoint
directory.

MOSS-Transcribe-Diarize (``stt.variant`` "moss-transcribe-diarize", architecture ``moss``): ``language_model`` is its
Qwen3 (``dec.*`` under Qwen3's GGUF names, ``output.weight`` the tokens' embedding, the ``moss.*`` keys of
``hf_model._Moss``), ``tower`` its audio tower's mmproj (``enc.*`` and ``adaptor.*`` under gguf-py's Whisper encoder
names, the ``clip.audio`` keys of ``hf_model._MossAudio``). Both are views of the one parsed file: its tensors under
other names, read in place.
"""

import copy
import re

from .gguf_model import GGUFFile

VARIANTS = ("moss-transcribe-diarize",)
# the language model's tensors (``dec.``) and the tower's (``enc.``, ``adaptor.``): the file's name as a pattern -> ours
_DEC = {r"token_embd\.weight": "token_embd.weight", r"output_norm\.weight": "output_norm.weight",
        r"blocks\.(\d+)\.norm_attn\.weight": r"blk.\1.attn_norm.weight",
        r"blocks\.(\d+)\.norm_ffn\.weight": r"blk.\1.ffn_norm.weight",
        r"blocks\.(\d+)\.attn\.([qkv])\.weight": r"blk.\1.attn_\2.weight",
        r"blocks\.(\d+)\.attn\.o\.weight": r"blk.\1.attn_output.weight",
        r"blocks\.(\d+)\.attn\.([qk])_norm\.weight": r"blk.\1.attn_\2_norm.weight",
        r"blocks\.(\d+)\.ffn\.(gate|up|down)\.weight": r"blk.\1.ffn_\2.weight"}
_ENC = {r"enc\.conv\.0\.(\w+)": r"a.conv1d.1.\1", r"enc\.conv\.1\.(\w+)": r"a.conv1d.2.\1",
        r"enc\.pos_emb\.weight": "a.position_embd.weight", r"enc\.final_norm\.(\w+)": r"a.post_ln.\1",
        r"enc\.blocks\.(\d+)\.norm_attn\.(\w+)": r"a.blk.\1.ln1.\2", r"enc\.blocks\.(\d+)\.norm_ffn\.(\w+)": r"a.blk.\1.ln2.\2",
        r"enc\.blocks\.(\d+)\.attn\.([qkv])\.(\w+)": r"a.blk.\1.attn_\2.\3",
        r"enc\.blocks\.(\d+)\.attn\.out\.(\w+)": r"a.blk.\1.attn_out.\2",
        r"enc\.blocks\.(\d+)\.ffn\.fc1\.(\w+)": r"a.blk.\1.ffn_up.\2",
        r"enc\.blocks\.(\d+)\.ffn\.fc2\.(\w+)": r"a.blk.\1.ffn_down.\2",
        r"adaptor\.fc1\.(\w+)": r"mm.a.mlp.0.\1", r"adaptor\.fc2\.(\w+)": r"mm.a.mlp.2.\1",
        r"adaptor\.norm_out\.(\w+)": r"mm.a.mlp.3.\1"}


def is_stt(g: GGUFFile) -> bool:
    """A transcribe.cpp file of a model these views read (NotImplementedError: one of another)."""
    v = g.meta.get("stt.variant")
    if v is not None and v not in VARIANTS:
        raise NotImplementedError(f"{g.path}: transcribe.cpp's {v!r} is not supported (supported: {', '.join(VARIANTS)})")
    return v is not None


def _view(g: GGUFFile, arch: str, meta: dict[str, object], names: dict[str, str], prefix: str = "") -> GGUFFile:
    """The file with architecture ``arch``, ``meta`` over its general and tokenizer keys, and the tensors that match
    ``names`` (after ``prefix``) under their new names."""
    v = copy.copy(g)
    v.arch = arch
    v.meta = {k: x for k, x in g.meta.items() if k.startswith(("general.", "tokenizer."))} | \
        {"general.architecture": arch} | meta
    v.tensor_info = {}
    for name, info in g.tensor_info.items():
        if not name.startswith(prefix):
            continue
        for pat, new in names.items():
            if m := re.fullmatch(pat, name[len(prefix):]):
                v.tensor_info[m.expand(new)] = info
                break
    return v


def language_model(g: GGUFFile) -> GGUFFile:
    """The model's language model as a GGUF file of architecture ``moss`` (module docstring)."""
    m = lambda k: g.meta[f"stt.moss.{k}"]
    d = lambda k: m(f"decoder.{k}")
    if not d("tie_word_embeddings"):
        raise NotImplementedError(f"{g.path}: an output weight of its own")
    v = _view(g, "moss", {
        "moss.block_count": int(d("n_layers")), "moss.context_length": int(d("max_position_embeddings")),
        "moss.embedding_length": int(d("hidden_size")), "moss.feed_forward_length": int(d("intermediate_size")),
        "moss.attention.head_count": int(d("n_heads")), "moss.attention.head_count_kv": int(d("n_kv_heads")),
        "moss.attention.key_length": int(d("head_dim")), "moss.attention.value_length": int(d("head_dim")),
        "moss.attention.layer_norm_rms_epsilon": float(d("rms_norm_eps")), "moss.rope.freq_base": float(d("rope_theta")),
        "moss.audio.tokens_per_second": float(m("audio_tokens_per_second")),
        "moss.audio.time_marker_seconds": int(m("time_marker_every_seconds")) * bool(m("enable_time_marker"))},
        _DEC, "dec.")
    v.tensor_info["output.weight"] = v.tensor_info["token_embd.weight"]
    return v


def tower(g: GGUFFile) -> GGUFFile:
    """The model's audio tower as its mmproj (module docstring)."""
    m = lambda k: g.meta[f"stt.moss.{k}"]
    e = lambda k: m(f"encoder.{k}")
    return _view(g, "clip", {
        "clip.has_audio_encoder": True, "clip.audio.projector_type": "moss",
        "clip.audio.projection_dim": int(m("decoder.hidden_size")), "clip.audio.embedding_length": int(e("d_model")),
        "clip.audio.feed_forward_length": int(e("ffn_dim")), "clip.audio.block_count": int(e("n_layers")),
        "clip.audio.attention.head_count": int(e("n_heads")), "clip.audio.attention.layer_norm_epsilon": 1e-5,
        "clip.audio.num_mel_bins": int(e("num_mel_bins")), "clip.audio.max_pos_emb": int(e("max_source_positions")),
        "clip.audio.projector.stack_factor": int(m("audio_merge_size")),
        "clip.audio.projector.layer_norm_epsilon": float(m("decoder.rms_norm_eps")),
        "clip.audio.activation": str(e("activation"))}, _ENC)
