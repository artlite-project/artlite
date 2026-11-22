"""Hugging Face checkpoints as model files: a directory of ``config.json``, safetensors (one file, or shards listed by
``model.safetensors.index.json``), ``tokenizer.json`` (with ``tokenizer_config.json``, ``generation_config.json``,
``chat_template.jinja``) opens as a *virtual GGUF file* (``HfModel``, a ``GGUFFile``), so the models and their weight
images take a checkpoint as they take a GGUF file (``open_weights``: either, by the path).

The metadata is what llama.cpp's converter writes for the checkpoint (``convert_hf_to_gguf.py``,
``conversion/{llama,qwen}.py``): the architecture's keys from config.json, the vocabulary from tokenizer.json (the
tokens with their types, ``[PAD<i>]`` up to the config's ``vocab_size``) and gguf-py's ``SpecialVocab`` (merges,
special token ids, the chat template), the sampling defaults of generation_config.json. The tensors carry their GGUF
names and are read in place from the memory-mapped safetensors; the converter's transforms (Qwen3.5's norm offsets,
``A_log`` -> A, the conv1d squeeze, its V heads in tiled order) are computed on read as F32 (vectors) or in memory
(row permutations). One difference: a Llama's Q / K rows stay in the checkpoint's rotate-half order (``unpermuted``;
the converter interleaves them and the weight images undo it, ``gguf_model.unpermute_rows``).

Supported: ``Qwen3ForCausalLM`` (``qwen3``), ``LlamaForCausalLM`` (``llama``), ``Qwen3_5ForConditionalGeneration`` /
``Qwen3_5ForCausalLM`` (``qwen35``; its MTP block as the nextn layer), ``Qwen3_5MoeForConditionalGeneration``
(``qwen35moe``), ``Gemma4ForConditionalGeneration`` / ``Gemma4ForCausalLM`` (``gemma4``). Vision and audio towers are not
read here.
"""

import dataclasses
import json
import math
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass

import gguf
import numpy as np

from . import paro
from .gguf_model import GGUFFile, QTensor

# safetensors dtypes -> (GGUF type, numpy dtype of the raw view)
_DTYPES = {"F32": ("F32", np.float32), "F16": ("F16", np.float16), "BF16": ("BF16", np.uint16),
           "I32": ("I32", np.int32), "I16": ("I16", np.int16), "I8": ("I8", np.int8), "U8": ("U8", np.uint8)}
# GGUF token types (llama.cpp's)
T_NORMAL, T_UNKNOWN, T_CONTROL, T_USER, T_UNUSED, T_BYTE = 1, 2, 3, 4, 5, 6


def is_checkpoint(path: str) -> bool:
    """A Hugging Face checkpoint directory (config.json and safetensors)."""
    return os.path.isfile(os.path.join(path, "config.json")) and bool(_shard_names(path))


def open_weights(path: str) -> GGUFFile:
    """A model file: a GGUF file, or a checkpoint directory as its virtual GGUF file (``HfModel``)."""
    return HfModel(path) if os.path.isdir(path) else GGUFFile(path)


def _shard_names(path: str) -> list[str]:
    """The safetensors files of a checkpoint (the index's shards in order, else every ``*.safetensors``)."""
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return list(dict.fromkeys(json.load(f)["weight_map"].values()))
    if not os.path.isdir(path):
        return []
    return sorted(n for n in os.listdir(path) if n.endswith(".safetensors"))


@dataclass(frozen=True)
class _Entry:
    """A tensor of a checkpoint: the GGUF type of its raw view, its shape as the checkpoint has it, and where its bytes
    are (shard index, offset in the shard, bytes); or ``compute``: an array computed on read (``shape`` its shape)."""

    qtype: str
    shape: tuple[int, ...]
    shard: int = -1
    offset: int = 0
    nbytes: int = 0
    compute: Callable[[], np.ndarray] | None = None
    paro: str = ""  # (a PARO linear: its checkpoint name's prefix; ``ParoTensor``)
    rows_of: np.ndarray | None = None  # (PARO) its rows / groups in another order (``ParoTensor``)
    groups_of: np.ndarray | None = None
    experts: int = 0  # (PARO experts stacked: ``paro`` a prefix with "{e}", ``rot`` their shared rotation's; ``ParoExperts``)
    rot: str = ""


class HfModel(GGUFFile):
    """A checkpoint directory as a GGUF file (module docstring): ``meta``, ``tensor_info`` and ``tensor`` in GGUF
    naming; ``hparams``: the config (a multimodal config's ``text_config`` merged in); ``config``: config.json as it
    is; ``files``: the shards and their mappings."""

    unpermuted = True  # (a Llama's Q / K rows in rotate-half order: module docstring)

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "config.json")) as f:
            self.config = json.load(f)
        self.hparams = {**self.config, **self.config.get("text_config", {})}
        qc = self.config.get("quantization_config") or self.hparams.get("quantization_config")
        self.paro_bits = self.paro_group = 0  # (a PARO checkpoint's: ``artlite.paro``)
        if qc:
            if qc.get("quant_method") != "paroquant":
                raise NotImplementedError(f"{path}: quantization method {qc.get('quant_method')!r} (Artlite runs "
                                          f"unquantized and ParoQuant checkpoints)")
            self.paro_bits, self.paro_group = paro.check_config(path, qc)
        hf_arch = (self.config.get("architectures") or [""])[0]
        if hf_arch not in _ADAPTERS:
            raise NotImplementedError(f"{path}: Hugging Face architecture {hf_arch!r} is not supported (supported: "
                                      f"{', '.join(_ADAPTERS)})")
        self._shards: list[tuple[str, np.memmap]] = []
        self._hf: dict[str, _Entry] = {}
        for name in _shard_names(path):
            self._read_header(os.path.join(path, name))
        self.adapter = _ADAPTERS[hf_arch](self)
        self.arch = self.adapter.arch
        self._entries: dict[str, _Entry] = {}
        for name in self._hf:
            for gname, e in self.adapter.tensors(name):
                self._entries[gname] = e
        for gname, e in self.adapter.extra():
            self._entries[gname] = e
        self.meta, self._strings = self.adapter.metadata()
        self._string_arrays = {k: (0, len(v)) for k, v in self._strings.items()}
        self.tensor_info = {n: (tuple(reversed(e.shape)) or (1,), e.qtype, e.offset if e.compute is None else -1,
                                e.nbytes) for n, e in self._entries.items()}

    def _read_header(self, path: str) -> None:
        data = np.memmap(path, dtype=np.uint8, mode="r")
        (n,) = struct.unpack_from("<Q", data, 0)
        header = json.loads(bytes(data[8:8 + n]))
        header.pop("__metadata__", None)
        shard = len(self._shards)
        self._shards.append((path, data))
        for name, e in header.items():
            if e["dtype"] not in _DTYPES:
                raise NotImplementedError(f"{path}: {name}: safetensors dtype {e['dtype']}")
            qtype, dt = _DTYPES[e["dtype"]]
            begin, end = e["data_offsets"]
            shape = tuple(e["shape"])
            if end - begin != math.prod(shape) * np.dtype(dt).itemsize:
                raise ValueError(f"{path}: {name}: a byte range that does not fit the shape")
            self._hf[name] = _Entry(qtype, shape, shard, 8 + n + begin, end - begin)

    # -- the checkpoint's tensors by their Hugging Face names

    def hf_names(self) -> list[str]:
        return list(self._hf)

    def hf_array(self, name: str) -> np.ndarray:
        """A tensor of the checkpoint as it is stored (its shape, BF16 as uint16), read in place."""
        return self._array(self._hf[name])

    def hf_f32(self, name: str) -> np.ndarray:
        """A tensor of the checkpoint as float32 (a copy)."""
        e = self._hf[name]
        a = self._array(e)
        return (a.astype(np.uint32) << 16).view(np.float32) if e.qtype == "BF16" else a.astype(np.float32)

    def _array(self, e: _Entry) -> np.ndarray:
        if e.compute is not None:
            return e.compute()
        data = self._shards[e.shard][1]
        dt = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I32": np.int32, "I16": np.int16,
              "I8": np.int8, "U8": np.uint8}[e.qtype]
        return data[e.offset:e.offset + e.nbytes].view(dt).reshape(e.shape)

    # -- GGUFFile

    @property
    def files(self) -> list[tuple[str, np.ndarray]]:
        return list(self._shards)

    def _mapped(self, name: str) -> tuple[np.ndarray, int, int] | None:
        e = self._entries[name]
        return None if e.compute is not None else (self._shards[e.shard][1], e.offset, e.nbytes)

    def _sources(self, name: str) -> list[tuple[np.ndarray, int, int]]:
        e = self._entries[name]
        if not e.paro:
            return super()._sources(name)
        hf = [f"{p}.{c}" for p in ([e.paro.format(e=i) for i in range(e.experts)] if e.experts else [e.paro])
              for c in _PARO_CODES]
        return [(self._shards[x.shard][1], x.offset, x.nbytes) for x in (self._hf[n] for n in hf if n in self._hf)]

    def strings(self, key: str, first: int = 0) -> list[str]:
        return self._strings[key][first:]

    def strings_json(self, key: str, ids: np.ndarray | None = None) -> bytes:
        s = self._strings[key]
        out = s if ids is None else {s[int(i)]: int(i) for i in ids}
        return json.dumps(out, ensure_ascii=False).encode()

    def tensor(self, name: str) -> QTensor:
        e = self._entries[name]
        if e.paro:
            return self._paro(name, e)
        a = self._array(e)
        if a.ndim < 2:
            a = a.reshape(1, -1)
        a = a.reshape(-1, a.shape[-1])
        return QTensor(name=name, qtype=e.qtype, rows=a.shape[0], cols=a.shape[1], raw=a)

    def _paro(self, name: str, e: _Entry) -> paro.ParoTensor:
        """A PARO linear's ``ParoTensor`` (stacked experts: ``ParoExperts``)."""
        if e.experts:
            r = {p: self.hf_array(f"{e.rot}_weight_{p}") for p in ("theta", "pairs", "channel_scales")}
            rot = self._rotation(r)
            ts = [self._linear(name, e.paro.format(e=i), e.shape[1:], rot, e) for i in range(e.experts)]
            t0 = ts[0]
            return paro.ParoExperts(name=name, qtype=e.qtype, rows=e.experts * t0.rows, cols=t0.cols, raw=t0.raw,
                                    qweight=t0.qweight, qzeros=t0.qzeros, scales=t0.scales, group=self.paro_group,
                                    rotation=rot, qweight_hi=t0.qweight_hi, qzeros_hi=t0.qzeros_hi, experts=ts)
        r = {p: self.hf_array(f"{e.paro}.{p}") for p in ("theta", "pairs", "channel_scales")}
        return self._linear(name, e.paro, e.shape, self._rotation(r), e)

    def _rotation(self, r: dict[str, np.ndarray]) -> paro.Rotation:
        return paro.Rotation(np.asarray(r["theta"], np.float64), np.asarray(r["pairs"], np.int64),
                             np.asarray(r["channel_scales"], np.float64).reshape(-1), self.paro_group)

    def _linear(self, name: str, prefix: str, shape: tuple[int, ...], rot: paro.Rotation, e: _Entry) -> paro.ParoTensor:
        """The ``ParoTensor`` (N, K) = shape of the codes at ``<prefix>.qweight`` .."""
        a = {p: self.hf_array(f"{prefix}.{p}") for p in _PARO_CODES if f"{prefix}.{p}" in self._hf}
        N, K = shape
        return paro.ParoTensor(name=name, qtype=e.qtype, rows=N, cols=K, raw=a["qweight"], qweight=a["qweight"],
                               qzeros=a["qzeros"], scales=a["scales"], group=self.paro_group, rotation=rot,
                               qweight_hi=a.get("qweight_hi"), qzeros_hi=a.get("qzeros_hi"), rows_of=e.rows_of,
                               groups_of=e.groups_of)


# ----------------------------------------------------------------------
# Architectures (llama.cpp's converter: conversion/base.py's TextModel, llama.py, qwen.py)
# ----------------------------------------------------------------------


class _Adapter:
    """An architecture's metadata and tensors (the converter's ``set_gguf_parameters``, ``set_vocab``,
    ``modify_tensors``, ``generate_extra_tensors``)."""

    arch = ""
    model_arch: gguf.MODEL_ARCH
    pre = ""  # tokenizer.ggml.pre (the converter finds it by a hash of a tokenization; one per architecture here)
    split = ""  # the tokenizer.json's split regex that pre stands for
    n_mtp = 0  # (MTP blocks: layers past num_hidden_layers)

    def __init__(self, hf: HfModel):
        self.hf, self.hp = hf, hf.hparams
        self.block_count = int(self.hp["num_hidden_layers"])
        self.names = gguf.get_tensor_name_map(self.model_arch, self.block_count + 1)
        self.rope = self._rope_parameters()

    def _rope_parameters(self) -> dict:
        """TextModel's rope_parameters: the config's (rope_parameters, else rope_scaling) with the global values
        mirrored."""
        hp = self.hp
        rp = dict(hp.get("rope_parameters") or hp.get("rope_scaling") or {})
        if "full_attention" not in rp and "sliding_attention" not in rp:
            if "rope_theta" not in rp and hp.get("rope_theta") is not None:
                rp["rope_theta"] = hp["rope_theta"]
            if "rope_type" not in rp and rp.get("type") is not None:
                rp["rope_type"] = rp["type"]
            if "partial_rotary_factor" not in rp and hp.get("partial_rotary_factor") is not None:
                rp["partial_rotary_factor"] = hp["partial_rotary_factor"]
        return rp

    # metadata

    def write_metadata(self, w: gguf.GGUFWriter) -> None:
        """The metadata into a GGUF writer."""
        self.params(w)
        self.vocab(w)
        self.sampling(w)

    def metadata(self) -> tuple[dict[str, object], dict[str, list[str]]]:
        """(meta, string arrays) as ``GGUFFile`` holds them."""
        w = gguf.GGUFWriter(path=None, arch=self.arch)
        self.write_metadata(w)
        meta: dict[str, object] = {}
        strings: dict[str, list[str]] = {}
        for k, v in w.kv_data[0].items():
            if v.type == gguf.GGUFValueType.ARRAY:
                if v.sub_type == gguf.GGUFValueType.STRING or (v.value and isinstance(v.value[0], str | bytes)):
                    strings[k] = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x for x in v.value]
                else:  # (the element type gguf-py writes: its sub_type, else that of the first element)
                    t = v.sub_type or (gguf.GGUFValueType.get_type(v.value[0]) if len(v.value) else None)
                    meta[k] = np.asarray(v.value, _ARRAY_DTYPES.get(t))
            else:  # (a float32 key as the file holds it)
                meta[k] = float(np.float32(v.value)) if v.type == gguf.GGUFValueType.FLOAT32 else v.value
        return meta, strings

    def key(self, w: gguf.GGUFWriter, name: str, value: object) -> None:
        """``{arch}.<name>`` = value (a key the installed gguf-py may have no ``add_*`` for)."""
        k = f"{self.arch}.{name}"
        if isinstance(value, bool):
            w.add_bool(k, value)
        elif isinstance(value, int):
            w.add_uint32(k, value)
        elif isinstance(value, float):
            w.add_float32(k, value)
        else:
            w.add_array(k, list(value))

    def params(self, w: gguf.GGUFWriter) -> None:
        """TextModel.set_gguf_parameters."""
        hp, rp = self.hp, self.rope
        w.add_block_count(self.block_count + self.n_mtp)
        w.add_context_length(int(hp["max_position_embeddings"]))
        w.add_embedding_length(int(hp["hidden_size"]))
        if (ff := hp.get("intermediate_size")) is not None:
            w.add_feed_forward_length(int(ff))
        w.add_head_count(int(hp["num_attention_heads"]))
        if (kv := hp.get("num_key_value_heads")) is not None:
            w.add_head_count_kv(int(kv))
        if hp.get("is_causal") is False or hp.get("use_bidirectional_attention") is True:
            w.add_causal_attention(False)
        full = rp.get("full_attention", rp)
        if (rt := full.get("rope_type")) is not None:
            if rt == "linear" and full.get("factor") is not None:
                w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                w.add_rope_scaling_factor(full["factor"])
            elif rt == "yarn" and full.get("factor") is not None:
                raise NotImplementedError(f"{self.hf.path}: YaRN RoPE scaling")
            elif rt not in ("default", "llama3", "proportional", "dynamic"):
                raise NotImplementedError(f"{self.hf.path}: RoPE type {rt!r}")
        if "mrope_section" in rp:
            w.add_rope_dimension_sections([*rp["mrope_section"], 0, 0, 0, 0][:4])
        if (theta := full.get("rope_theta")) is not None:
            w.add_rope_freq_base(theta)
        if (theta := rp.get("sliding_attention", {}).get("rope_theta")) is not None:
            w.add_rope_freq_base_swa(theta)
        if (eps := hp.get("rms_norm_eps")) is not None:
            w.add_layer_norm_rms_eps(eps)
        if (n := hp.get("num_experts", hp.get("num_local_experts"))) is not None:
            w.add_expert_count(int(n))
        if (n := hp.get("num_experts_per_tok")) is not None:
            w.add_expert_used_count(int(n))
        if (hd := hp.get("head_dim")) is not None:
            w.add_key_length(int(hd))
            w.add_value_length(int(hd))

    def vocab(self, w: gguf.GGUFWriter) -> None:
        """TextModel._set_vocab_gpt2: GPT-2 byte-level BPE (get_vocab_base: transformers' vocabulary of the
        tokenizer.json, its added tokens CONTROL when special or special-looking, else USER_DEFINED)."""
        path = os.path.join(self.hf.path, "tokenizer.json")
        with open(path, encoding="utf-8") as f:
            tj = json.load(f)
        model = tj["model"]
        if model["type"] != "BPE" or model.get("byte_fallback"):
            raise NotImplementedError(f"{path}: a {model['type']} tokenizer (byte-level BPE expected)")
        split = _split_regex(tj)
        if split != self.split:
            raise NotImplementedError(f"{path}: pre-tokenizer split {split!r}, not {self.arch}'s")
        n = int(self.hp.get("vocab_size", 0))
        added = {t["id"]: t for t in tj.get("added_tokens", [])}
        cfg = os.path.join(self.hf.path, "tokenizer_config.json")
        tc: dict = {}
        if os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                tc = json.load(f)
        for i, t in tc.get("added_tokens_decoder", {}).items():  # (transformers adds these too)
            added.setdefault(int(i), t)
        rev = {i: t for t, i in model["vocab"].items()}
        rev.update({i: t["content"] for i, t in added.items()})
        n = max(n, max(rev) + 1)
        tokens, types = [], []
        for i in range(n):
            if i not in rev:
                tokens.append(f"[PAD{i}]")
                types.append(T_UNUSED)
                continue
            t = rev[i]
            if i in added:
                if added[i].get("special") or _looks_special(t):
                    types.append(T_CONTROL)
                else:
                    t = t.replace("▁", " ")
                    types.append(T_USER)
            else:
                types.append(T_NORMAL)
            tokens.append(t)
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre(self.pre)
        w.add_token_list(tokens)
        w.add_token_types(types)
        if (aps := tc.get("add_prefix_space")) is not None and self.arch == "llama":
            w.add_add_space_prefix(aps)
        _special_vocab(self.hf.path, tj, w)

    def sampling(self, w: gguf.GGUFWriter) -> None:
        """gguf-py's Metadata: the sampling defaults of generation_config.json."""
        path = os.path.join(self.hf.path, "generation_config.json")
        if not os.path.isfile(path):
            return
        with open(path) as f:
            g = json.load(f)
        keys = {"top_k": ("top_k", int), "top_p": ("top_p", float), "min_p": ("min_p", float),
                "temperature": ("temp", float), "repetition_penalty": ("penalty_repeat", float)}
        for src, (dst, kind) in keys.items():
            if g.get(src) is not None:
                (w.add_int32 if kind is int else w.add_float32)(f"general.sampling.{dst}", kind(g[src]))

    # tensors

    def gguf_name(self, name: str) -> str | None:
        """The GGUF name of a (filtered) Hugging Face name (gguf-py's tensor map)."""
        return self.names.get_name(name, try_suffixes=(".weight", ".bias"))

    def tensors(self, name: str) -> list[tuple[str, _Entry]]:
        """The GGUF tensors of a checkpoint tensor (none: not a language-model tensor; a PARO linear's parts: one
        tensor at its ``qweight``)."""
        hf_name = self.filter(name)
        if hf_name is None:
            return []
        if (part := name.rpartition(".")[2]) in _PARO_PARTS or name.endswith(_PARO_SHARED):
            if part != "qweight":
                return []
            prefix = name.removesuffix(".qweight")
            if m := re.fullmatch(r"(.*\.experts)\.(\d+)\.(\w+)", prefix):  # (the stack, at expert 0's qweight)
                if m[2] != "0":
                    return []
                e = self.experts_entry(m[1], m[3])
                hf_name = re.sub(r"\.experts\.0\.(\w+)\.qweight$", r".experts.\1.weight", hf_name)
            else:
                e = self.paro_entry(prefix)
                hf_name = hf_name.removesuffix(".qweight") + ".weight"
        elif ".experts." in name:
            raise NotImplementedError(f"{self.hf.path}: unquantized experts ({name}; PARO experts run)")
        else:
            e = self.hf._hf[name]
        g = self.gguf_name(hf_name)
        if g is None:
            raise NotImplementedError(f"{self.hf.path}: no GGUF name for {name}")
        return [(g, self.transform(name, hf_name, e))]

    def paro_entry(self, prefix: str) -> _Entry:
        """The entry of the PARO linear whose checkpoint tensors are ``<prefix>.qweight`` ..."""
        hf = self.hf
        parts = [f"{prefix}.{p}" for p in _PARO_PARTS if f"{prefix}.{p}" in hf._hf]
        missing = {"qweight", "qzeros", "scales", "theta", "pairs", "channel_scales"} - {p.rpartition(".")[2] for p in parts}
        if missing or (hf.paro_bits == 5) != (f"{prefix}.qweight_hi" in hf._hf):
            raise ValueError(f"{hf.path}: PARO linear {prefix} without {', '.join(sorted(missing)) or 'its fifth bits'}")
        q = hf._hf[f"{prefix}.qweight"]
        K, N = q.shape[0], q.shape[1] * 8
        return _Entry(f"P{hf.paro_bits}_{hf.paro_group}", (N, K), q.shard, q.offset,
                      sum(hf._hf[p].nbytes for p in parts), paro=prefix)

    def experts_entry(self, prefix: str, proj: str) -> _Entry:
        """The entry of an MoE layer's PARO experts ``<prefix>.<e>.<proj>`` stacked (E, N, K), their rotation the shared
        ``<prefix>.gate_up_weight_*`` (gate_proj, up_proj) or ``<prefix>.down_weight_*``."""
        hf = self.hf
        E = int(self.hp["num_experts"])
        rot = f"{prefix}.{'down' if proj == 'down_proj' else 'gate_up'}"
        codes = ("qweight", "qzeros", "scales") + (("qweight_hi", "qzeros_hi") if hf.paro_bits == 5 else ())
        parts = [f"{prefix}.{i}.{proj}.{p}" for i in range(E) for p in codes]
        parts += [f"{rot}_weight_{p}" for p in ("theta", "pairs", "channel_scales")]
        if missing := [p for p in parts if p not in hf._hf]:
            raise ValueError(f"{hf.path}: PARO experts {prefix}.*.{proj} without {missing[0]}")
        q = hf._hf[f"{prefix}.0.{proj}.qweight"]
        K, N = q.shape[0], q.shape[1] * 8
        return _Entry(f"P{hf.paro_bits}_{hf.paro_group}", (E, N, K), q.shard, q.offset,
                      sum(hf._hf[p].nbytes for p in parts), paro=f"{prefix}.{{e}}.{proj}", experts=E, rot=rot)

    def filter(self, name: str) -> str | None:
        """A language-model tensor's name as the tensor map knows it (None: skip the tensor)."""
        if name.startswith(("model.visual.", "visual.", "model.vision_tower.", "mtp.")):
            return None
        return name.replace("model.language_model.", "model.")

    def transform(self, name: str, hf_name: str, e: _Entry) -> _Entry:
        """A tensor's value in the GGUF file: as stored (2 dims and more), or F32 (vectors: the converter writes 1-D
        tensors as F32)."""
        if len(e.shape) < 2:
            return self.computed(name, lambda a: a)
        return e

    def computed(self, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> _Entry:
        """An F32 tensor computed on read from a checkpoint tensor (float32)."""
        hf = self.hf
        shape = np.shape(fn(np.zeros(hf._hf[name].shape, np.float32)))
        return _Entry("F32", tuple(shape), nbytes=4 * math.prod(shape),
                      compute=lambda: np.ascontiguousarray(fn(hf.hf_f32(name)), np.float32))

    def extra(self) -> list[tuple[str, _Entry]]:
        """generate_extra_tensors."""
        return []


def _const(a: np.ndarray) -> _Entry:
    a = np.ascontiguousarray(a, np.float32)
    return _Entry("F32", a.shape, nbytes=a.nbytes, compute=lambda: a)


_ARRAY_DTYPES = {gguf.GGUFValueType.UINT8: np.uint8, gguf.GGUFValueType.INT8: np.int8,
                 gguf.GGUFValueType.UINT16: np.uint16, gguf.GGUFValueType.INT16: np.int16,
                 gguf.GGUFValueType.UINT32: np.uint32, gguf.GGUFValueType.INT32: np.int32,
                 gguf.GGUFValueType.FLOAT32: np.float32, gguf.GGUFValueType.UINT64: np.uint64,
                 gguf.GGUFValueType.INT64: np.int64, gguf.GGUFValueType.FLOAT64: np.float64,
                 gguf.GGUFValueType.BOOL: np.bool_}


def _special_vocab(path: str, tj: dict, w: gguf.GGUFWriter) -> None:
    """llama.cpp's gguf-py ``SpecialVocab(path, load_merges=True).add_to_gguf`` (its current form, which the installed
    gguf-py may predate): the merges, the special token ids (tokenizer_config.json's ``<typ>_token`` found among the
    added tokens, else config.json's ``<typ>_token_id``), ``add_bos`` / ``add_eos`` (the post-processor's: ByteLevel adds
    none, a TemplateProcessing its single template's leading / trailing special tokens; else tokenizer_config.json's),
    the chat template (chat_template.jinja, chat_template.json, tokenizer_config.json's, in that order); an NFC
    normalizer as ``tokenizer.ggml.normalizer.nfc`` (llama.cpp then normalizes a text before it splits it)."""
    merges = tj.get("model", {}).get("merges") or []
    if merges and isinstance(merges[0], list):  # ([a, b] pairs: a space in a part as U+0120)
        merges = [" ".join("".join(chr(ord(c) + 256) if c == " " else c for c in part) for part in pair) for pair in merges]
    if not merges and os.path.isfile(mt := os.path.join(path, "merges.txt")):
        with open(mt, encoding="utf-8") as f:
            merges = [" ".join(p) for line in f if not line.startswith("#") and len(p := line.split()) == 2]
    tc: dict = {}
    if os.path.isfile(cf := os.path.join(path, "tokenizer_config.json")):
        with open(cf, encoding="utf-8") as f:
            tc = json.load(f)
    types = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]
    add: dict[str, bool] = {}
    bos, eos = tc.get("bos_token") or tc.get("cls_token"), tc.get("eos_token") or tc.get("sep_token")
    if not tc.get("bos_token") and bos and tc:
        tc["bos_token"] = bos
    if not tc.get("eos_token") and eos and tc:
        tc["eos_token"] = eos
    pp = tj.get("post_processor") or {}
    for proc in pp.get("processors", [pp]) if pp else []:
        kind = proc.get("type")
        if kind == "ByteLevel":
            add.setdefault("bos", False)
            add.setdefault("eos", False)
        elif kind == "TemplateProcessing":
            single, first, last = proc.get("single", []), None, None
            if len(single) > 1:
                if first := single[0].get("SpecialToken", {}).get("id"):
                    if not tc:
                        bos = first
                    elif first not in (bos, tc.get("cls_token")) and not bos:
                        tc["bos_token"] = bos = first
                    add["bos"] = first in (bos, tc.get("cls_token"))
                if last := single[-1].get("SpecialToken", {}).get("id"):
                    if not tc:
                        eos = last
                    elif last != eos:
                        types.append("eot")
                        tc["eot_token"], tc["eos_token"] = eos, last
                        eos = last
                    add["eos"] = last == eos
            pair = proc.get("pair", [])
            if pair:  # (A [sep] [eos] B: whether a separator goes between a pair's sequences)
                lead = 1 if first and pair[0].get("SpecialToken", {}).get("id") == first else 0
                trail = -1 if last and pair[-1].get("SpecialToken", {}).get("id") == last else None
                pair = pair[lead:trail]
                if (pair and pair[0].get("Sequence", {}).get("id") == "A" and pair[-1].get("Sequence", {}).get("id") == "B"
                        and (mid := pair[1:-1])):
                    seps = (tc.get("sep_token"), eos)
                    sep = bool(mid[0].get("SpecialToken", {}).get("id") in seps and not last)
                    if len(mid) == 2 and mid[1].get("SpecialToken", {}).get("id") in seps:
                        sep = True
                    add["sep"] = sep
        elif kind is not None and kind != "ByteLevel":
            raise NotImplementedError(f"{path}: tokenizer post-processor {kind}")
    template = tc.get("chat_template")
    if os.path.isfile(cj := os.path.join(path, "chat_template.jinja")):
        with open(cj, encoding="utf-8") as f:
            template = f.read()
    elif os.path.isfile(cj := os.path.join(path, "chat_template.json")):
        with open(cj, encoding="utf-8") as f:
            template = json.load(f).get("chat_template")
    ids: dict[str, int] = {}
    added = tj.get("added_tokens", [])
    for typ in types:
        if isinstance(entry := tc.get(f"add_{typ}_token"), bool):
            add.setdefault(typ, entry)
        entry = tc.get(f"{typ}_token")
        content = entry.get("content") if isinstance(entry, dict) else entry
        if isinstance(content, str):
            tid = next((a.get("id") for a in added if a.get("content") == content), None)
            if isinstance(tid, int):
                ids.setdefault(typ, tid)
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    for typ in types:
        tid = config.get(f"{typ}_token_id", config.get("text_config", {}).get(f"{typ}_token_id"))
        if isinstance(tid, int):
            ids.setdefault(typ, tid)
    if _nfc(tj.get("normalizer") or {}):
        w.add_bool("tokenizer.ggml.normalizer.nfc", True)
    if merges:
        w.add_token_merges(merges)
    keys = {"unk": "unknown", "sep": "seperator", "pad": "padding"}  # (gguf-py's key names)
    for typ, tid in ids.items():
        w.add_uint32(f"tokenizer.ggml.{keys.get(typ, typ)}_token_id", tid)
    for typ, value in add.items():
        w.add_bool(f"tokenizer.ggml.add_{typ}_token", value)
    if isinstance(template, str):
        w.add_chat_template(template)


def _nfc(normalizer: dict) -> bool:
    """Whether a tokenizer.json normalizer applies NFC (alone or in a Sequence)."""
    if normalizer.get("type") == "Sequence":
        return any(_nfc(n) for n in normalizer.get("normalizers", []))
    return normalizer.get("type") == "NFC"


def _looks_special(t: str) -> bool:
    """TextModel.does_token_look_special."""
    return (t in ("<pad>", "<mask>", "<2mass>", "[@BOS@]") or (t.startswith("<|") and t.endswith("|>"))
            or (t.startswith("<｜") and t.endswith("｜>")) or (t.startswith("<unused") and t.endswith(">")))


def _split_regex(tj: dict) -> str:
    """The split regex of a tokenizer.json's pre-tokenizer ("" when it has none)."""
    pre = tj.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers", [pre])
    for p in parts:
        if p.get("type") == "Split":
            return p["pattern"].get("Regex", p["pattern"].get("String", ""))
    return ""


# the tokenizer.json split regexes of the vocabularies (Qwen2 / Qwen3, Qwen3.5, Llama 3)
_QWEN2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
                r"\s+(?!\S)|\s+")
_QWEN35_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
                 r"\s*[\r\n]+|\s+(?!\S)|\s+")
_LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
                 r"\s*[\r\n]+|\s+(?!\S)|\s+")


class _Qwen3(_Adapter):
    """Qwen3ForCausalLM (Qwen2Model, Qwen3Model)."""

    arch, model_arch, pre, split = "qwen3", gguf.MODEL_ARCH.QWEN3, "qwen2", _QWEN2_SPLIT


class _Llama(_Adapter):
    """LlamaForCausalLM (LlamaModel): ``vocab_size`` and ``rope.dimension_count`` keys, the llama3 RoPE factors as
    ``rope_freqs``."""

    arch, model_arch, pre, split = "llama", gguf.MODEL_ARCH.LLAMA, "llama-bpe", _LLAMA3_SPLIT

    def params(self, w: gguf.GGUFWriter) -> None:
        super().params(w)
        w.add_vocab_size(int(self.hp["vocab_size"]))
        w.add_rope_dimension_count(int(self.hp.get("head_dim") or self.hp["hidden_size"] // self.hp["num_attention_heads"]))

    def extra(self) -> list[tuple[str, _Entry]]:
        rp = self.rope.get("full_attention", self.rope)
        if rp.get("rope_type", "").lower() != "llama3":
            return []
        base = rp.get("rope_theta", 10000.0)
        dim = self.hp.get("head_dim") or self.hp["hidden_size"] // self.hp["num_attention_heads"]
        freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        factor, lo, hi = rp.get("factor", 8.0), rp.get("low_freq_factor", 1.0), rp.get("high_freq_factor", 4.0)
        old = rp.get("original_max_position_embeddings", 8192)
        lo_wl, hi_wl = old / lo, old / hi
        out = []
        for f in freqs:  # (the converter's loop, its float32 frequencies and Python float arithmetic)
            wl = 2 * math.pi / f
            if wl < hi_wl:
                out.append(1)
            elif wl > lo_wl:
                out.append(factor)
            else:
                s = (old / wl - lo) / (hi - lo)
                out.append(1 / ((1 - s) / factor + s))
        return [("rope_freqs.weight", _const(np.asarray(out, np.float32)))]


class _Qwen35(_Adapter):
    """Qwen3_5ForConditionalGeneration (Qwen3NextModel, _LinearAttentionVReorderBase, Qwen3_5TextModel): the Gated
    DeltaNet keys, norms + 1 (but the GDN's gated norm), A = -exp(A_log), the conv1d squeezed, the V heads of the GDN
    tensors in tiled order (where V heads outnumber K heads), the MTP block as layer ``num_hidden_layers``."""

    arch, model_arch, pre, split = "qwen35", gguf.MODEL_ARCH.QWEN35, "qwen35", _QWEN35_SPLIT

    def __init__(self, hf: HfModel):
        super().__init__(hf)
        self.n_mtp = int(self.hp.get("mtp_num_hidden_layers", 0)) if any(n.startswith("mtp.") for n in hf._hf) else 0

    def params(self, w: gguf.GGUFWriter) -> None:
        hp = self.hp
        super().params(w)
        if hp.get("moe_intermediate_size") is not None:
            w.add_expert_feed_forward_length(int(hp["moe_intermediate_size"]))
        if hp.get("shared_expert_intermediate_size") is not None:
            w.add_expert_shared_feed_forward_length(int(hp["shared_expert_intermediate_size"]))
        w.add_ssm_conv_kernel(int(hp["linear_conv_kernel_dim"]))
        w.add_ssm_state_size(int(hp["linear_key_head_dim"]))
        w.add_ssm_group_count(int(hp["linear_num_key_heads"]))
        w.add_ssm_time_step_rank(int(hp["linear_num_value_heads"]))
        w.add_ssm_inner_size(int(hp["linear_value_head_dim"]) * int(hp["linear_num_value_heads"]))
        if (lt := hp.get("layer_types")) is not None:
            self.key(w, "attention.recurrent_layers", [t == "linear_attention" for t in lt] + [False] * self.n_mtp)
        w.add_full_attention_interval(int(hp.get("full_attention_interval", 4)))
        hd = hp.get("head_dim") or hp["hidden_size"] // hp["num_attention_heads"]
        w.add_rope_dimension_count(int(hd * self.rope.get("partial_rotary_factor", 0.25)))
        if self.n_mtp:
            w.add_nextn_predict_layers(self.n_mtp)
        if "mrope_section" not in self.rope:
            w.add_rope_dimension_sections([11, 11, 10, 0])

    def filter(self, name: str) -> str | None:
        name = name.replace("model.mtp.", "mtp.", 1) if name.startswith("model.mtp.") else name
        if name.startswith("mtp."):
            if not self.n_mtp:
                return None
            remap = {"fc": "eh_proj", "pre_fc_norm_embedding": "enorm", "pre_fc_norm_hidden": "hnorm",
                     "norm": "shared_head.norm"}
            parts = name.split(".", 3)
            if len(parts) == 4 and parts[1] == "layers" and parts[2].isdecimal():
                return f"model.layers.{self.block_count + int(parts[2])}.{parts[3]}"
            if len(parts) == 3 and parts[1] in remap:
                return f"model.layers.{self.block_count}.{remap[parts[1]]}.{parts[2]}"
            raise NotImplementedError(f"{self.hf.path}: MTP tensor {name}")
        return super().filter(name)

    def gguf_name(self, name: str) -> str | None:
        m = re.fullmatch(r"model\.layers\.(\d+)\.(eh_proj|enorm|hnorm|shared_head\.norm)\.weight", name)
        if m:
            return f"blk.{m[1]}.{_NEXTN[m[2]]}.weight"
        return super().gguf_name(name.replace(".dt_bias", ".dt_proj.bias"))

    def transform(self, name: str, hf_name: str, e: _Entry) -> _Entry:
        hp = self.hp
        nk, nv = int(hp["linear_num_key_heads"]), int(hp["linear_num_value_heads"])
        dk, dv = int(hp["linear_key_head_dim"]), int(hp["linear_value_head_dim"])
        r = nv // nk
        tiled = nk != nv and "linear_attn." in name

        def heads(n: int, dim: int) -> np.ndarray:
            """Grouped -> tiled order of n = nk r heads of dim (``_reorder_v_heads``)."""
            return np.arange(n * dim).reshape(nk, r, dim).transpose(1, 0, 2).reshape(-1)

        if name.endswith(".A_log"):  # (exp in float64, rounded once: within an ulp of the converter's torch.exp)
            p = heads(nv, 1) if tiled else slice(None)
            return self.computed(name, lambda a: (-np.exp(a.astype(np.float64))).astype(np.float32)[p])
        if name.endswith(".dt_bias"):
            p = heads(nv, 1) if tiled else slice(None)
            return self.computed(name, lambda a: a[p])
        if "conv1d" in name:
            qk = 2 * nk * dk
            p = np.concatenate([np.arange(qk), qk + heads(nv, dv)]) if tiled else slice(None)
            return self.computed(name, lambda a: a.reshape(a.shape[0], -1)[p])
        if hf_name.endswith("norm.weight") and not hf_name.endswith("linear_attn.norm.weight"):
            return self.computed(name, lambda a: a + 1)
        if tiled and (".in_proj_qkv." in name or ".in_proj_z." in name or ".in_proj_a." in name
                      or ".in_proj_b." in name or ".out_proj." in name):
            if ".in_proj_qkv." in name:
                qk = 2 * nk * dk
                perm, axis = np.concatenate([np.arange(qk), qk + heads(nv, dv)]), 0
            elif ".out_proj." in name:
                perm, axis = heads(nv, dv), 1
            else:
                perm, axis = heads(nv, dv if ".in_proj_z." in name else 1), 0
            return self.permuted(e, perm, axis)
        return super().transform(name, hf_name, e)

    def permuted(self, e: _Entry, perm: np.ndarray, axis: int) -> _Entry:
        """A matrix with its rows (axis 0) or columns (axis 1) in the order ``perm``, in memory; a PARO linear's as its
        index maps (columns: whole groups move, ``ParoTensor.groups_of``)."""
        hf = self.hf
        if e.paro:
            if axis == 0:
                return dataclasses.replace(e, rows_of=perm)
            g = hf.paro_group
            groups = perm.reshape(-1, g)[:, 0] // g
            assert np.array_equal(perm.reshape(-1, g), groups[:, None] * g + np.arange(g)), "columns not in whole groups"
            return dataclasses.replace(e, groups_of=groups)
        return _Entry(e.qtype, e.shape, nbytes=e.nbytes,
                      compute=lambda: np.ascontiguousarray(np.take(hf._array(e), perm, axis=axis)))


class _Qwen35Moe(_Qwen35):
    """Qwen3_5MoeForConditionalGeneration (Qwen3_5MoeTextModel: Qwen2MoeModel's experts stacked per layer, E N rows; a
    PARO checkpoint's at expert 0's ``qweight``, ``ParoExperts``): the shared expert's gate as a vector (F32, as the
    converter writes 1-D tensors)."""

    arch, model_arch = "qwen35moe", gguf.MODEL_ARCH.QWEN35MOE

    def transform(self, name: str, hf_name: str, e: _Entry) -> _Entry:
        if hf_name.endswith("shared_expert_gate.weight"):
            return self.computed(name, lambda a: a.reshape(-1))
        return super().transform(name, hf_name, e)


class _Gemma4(_Adapter):
    """Gemma4ForConditionalGeneration (Gemma3Model, Gemma4Model; its text model: the towers are not read here): the
    per-layer inputs, KV-shared layers (a PARO checkpoint has no K / V projections there), sliding-window layers of
    their own head dim and RoPE base, the double-wide MLP of the KV-shared layers, the full layers' proportional RoPE as
    ``rope_freqs`` (1e30 past the rotated pairs); norms as stored (no offset); the vocabulary of gguf-py's LlamaHfVocab
    (byte-fallback BPE: tokenizer.json's tokens, scored -1000, special ones CONTROL but those the chat parser reads,
    byte tokens BYTE)."""

    arch, model_arch = "gemma4", gguf.MODEL_ARCH.GEMMA4
    VISIBLE = frozenset({"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>",
                         "<tool_response|>", "<|\"|>"})

    def params(self, w: gguf.GGUFWriter) -> None:
        hp = self.hp
        super().params(w)
        for k in ("feed_forward_length", "attention.key_length", "attention.value_length"):  # (written again below)
            w.kv_data[0].pop(f"{self.arch}.{k}", None)
        swa = [t == "sliding_attention" for t in hp["layer_types"]]
        shared = int(hp["num_kv_shared_layers"])
        ff = int(hp["intermediate_size"])
        if hp.get("use_double_wide_mlp"):  # (the KV-shared layers' MLP twice as wide)
            w.add_feed_forward_length([ff if il < self.block_count - shared else 2 * ff for il in range(self.block_count)])
        else:
            w.add_feed_forward_length(ff)
        full_hd = hp.get("global_head_dim")
        full_kv = hp.get("num_global_key_value_heads")
        for il, lc in (hp.get("per_layer_config") or {}).items():  # (the first full layer's own head dim / KV heads)
            if int(il) < len(swa) and not swa[int(il)]:
                full_hd = full_hd if full_hd is not None else lc.get("head_dim")
                full_kv = full_kv if full_kv is not None else lc.get("num_key_value_heads")
        assert full_hd is not None, f"{self.hf.path}: no full-attention head dim"
        if full_kv is not None and hp.get("num_key_value_heads") is not None:
            w.add_head_count_kv([int(hp["num_key_value_heads"]) if s else int(full_kv) for s in swa])
        w.add_key_length(int(full_hd))
        w.add_value_length(int(full_hd))
        w.add_key_length_swa(int(hp["head_dim"]))
        w.add_value_length_swa(int(hp["head_dim"]))
        if (cap := hp.get("final_logit_softcapping")) is not None:
            w.add_final_logit_softcapping(cap)
        if hp.get("sliding_window_pattern") != 1:
            w.add_sliding_window(int(hp["sliding_window"]))
        w.add_shared_kv_layers(shared)
        w.add_embedding_length_per_layer_input(int(hp.get("hidden_size_per_layer_input") or 0))
        w.add_sliding_window_pattern(swa)
        if (eff := hp.get("expert_intermediate_size", hp.get("moe_intermediate_size"))) is not None:
            w.add_expert_feed_forward_length(int(eff))
        w.add_rope_dimension_count(int(full_hd))
        w.add_rope_dimension_count_swa(int(hp["head_dim"] * self.rope.get("sliding_attention", {}).get(
            "partial_rotary_factor", 1.0)))

    def vocab(self, w: gguf.GGUFWriter) -> None:
        path = os.path.join(self.hf.path, "tokenizer.json")
        with open(path, encoding="utf-8") as f:
            tj = json.load(f)
        model = tj["model"]
        if model["type"] != "BPE" or not model.get("byte_fallback") or tj["decoder"]["type"] != "Sequence":
            raise NotImplementedError(f"{path}: not a byte-fallback BPE tokenizer (LlamaHfVocab's)")
        rev = {i: t for t, i in model["vocab"].items()}
        base = len(model["vocab"])  # (transformers' vocab_size: the model's vocabulary)
        added = sorted(tj.get("added_tokens", []), key=lambda a: a["id"])
        special = {a["id"] for a in added if a.get("special")}
        rev.update({a["id"]: a["content"] for a in added if a["id"] < base})
        extra = [a for a in added if a["id"] >= base]  # (added tokens past the vocabulary: appended in id order)
        tokens, types = [], []
        for i in range(base):
            t = rev[i]
            tokens.append(t)
            types.append(T_BYTE if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t) else T_CONTROL if i in special else T_NORMAL)
        for a in extra:
            tokens.append(a["content"])
            types.append(T_CONTROL if a["id"] in special else T_USER)
        types = [T_USER if t in self.VISIBLE else k for t, k in zip(tokens, types, strict=True)]
        w.add_tokenizer_model("gemma4")
        w.add_token_list(tokens)
        w.add_token_scores([-1000.0] * len(tokens))
        w.add_token_types(types)
        _special_vocab(self.hf.path, tj, w)
        w.add_add_space_prefix(False)
        w.add_add_bos_token(True)

    def filter(self, name: str) -> str | None:
        if not name.startswith("model.language_model."):  # (the vision and audio towers, their embedders)
            return None
        name = name.replace("model.language_model.", "model.")
        return f"{name}.weight" if name.endswith(("per_dim_scale", "layer_scalar")) else name

    def extra(self) -> list[tuple[str, _Entry]]:
        full = self.rope["full_attention"]
        assert full["rope_type"] == "proportional", full
        hd = int(self.hp.get("global_head_dim") or 0)
        for il, lc in (self.hp.get("per_layer_config") or {}).items():
            if not hd and self.hp["layer_types"][int(il)] == "full_attention":
                hd = int(lc["head_dim"])
        n_rot = int(hd * full["partial_rotary_factor"] / 2)
        return [("rope_freqs.weight", _const(np.asarray([1.0] * n_rot + [1e30] * (hd // 2 - n_rot), np.float32)))]


# a PARO linear's checkpoint tensors (``artlite.paro``) and the suffixes of an MoE layer's shared rotations
_PARO_PARTS = ("qweight", "qzeros", "scales", "theta", "pairs", "channel_scales", "qweight_hi", "qzeros_hi")
_PARO_CODES = ("qweight", "qzeros", "scales", "qweight_hi", "qzeros_hi")
_PARO_SHARED = ("_weight_theta", "_weight_pairs", "_weight_channel_scales")

# the MTP block's tensors (the converter's names past the tensor map)
_NEXTN = {"eh_proj": "nextn.eh_proj", "enorm": "nextn.enorm", "hnorm": "nextn.hnorm",
          "shared_head.norm": "nextn.shared_head_norm"}

_ADAPTERS: dict[str, type[_Adapter]] = {
    "Qwen3ForCausalLM": _Qwen3,
    "LlamaForCausalLM": _Llama,
    "Qwen3_5ForConditionalGeneration": _Qwen35,
    "Qwen3_5ForCausalLM": _Qwen35,
    "Qwen3_5MoeForConditionalGeneration": _Qwen35Moe,
    "Qwen3_5MoeForCausalLM": _Qwen35Moe,
    "Gemma4ForConditionalGeneration": _Gemma4,
    "Gemma4ForCausalLM": _Gemma4,
}


def architecture(path: str) -> str:
    """The GGUF architecture a checkpoint directory opens as (from its config.json alone)."""
    with open(os.path.join(path, "config.json")) as f:
        hf_arch = (json.load(f).get("architectures") or [""])[0]
    if hf_arch not in _ADAPTERS:
        raise NotImplementedError(f"{path}: Hugging Face architecture {hf_arch!r} is not supported (supported: "
                                  f"{', '.join(_ADAPTERS)})")
    return _ADAPTERS[hf_arch].arch
