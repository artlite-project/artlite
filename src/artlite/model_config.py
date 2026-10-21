"""Model architectures from the metadata of a GGUF file (:class:`LlamaConfig`, :class:`GemmaConfig`,
:class:`QwenConfig`), and where model files live.

Model directories live under :data:`MODELS_DIR` (``ARTLITE_MODELS_DIR``) as ``<org>/<repository>/`` (the Hugging Face
repository a file comes from):

    <org>/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf, ...

:func:`llama_gguf`, :func:`gemma_gguf`, :func:`qwen_gguf` (and the others below) name the files there and
:func:`model_path` resolves a path given relative to it, so nothing else spells the location.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

MODELS_DIR = os.environ.get("ARTLITE_MODELS_DIR", os.path.join(os.path.expanduser("~"), ".artlite", "models"))
# a llama.cpp source tree with a CPU build (build/bin/libllama.so): the golden tensors of the qwen35 models
LLAMA_CPP_DIR = os.environ.get("ARTLITE_LLAMA_CPP_DIR", "/src/llama.cpp")
# uploaders of Llama GGUF files, in order of preference (file names: <repo name>-<quant>.gguf, MaziyarPanahi's
# <repo name>.<quant>.gguf)
LLAMA_ORGS = ("lmstudio-community", "bartowski", "MaziyarPanahi")
LLAMA_ORG = LLAMA_ORGS[0]


def model_path(path: str) -> str:
    """``path`` itself if it exists, else ``path`` under :data:`MODELS_DIR`."""
    return path if os.path.exists(path) else os.path.join(MODELS_DIR, path)


def llama_gguf(quant: str, size: str = "1B") -> str:
    """The Llama Instruct GGUF file of a quantization (``Q4_0``, ``Q4_K_M``, ``BF16``, ...): Llama 3.2
    1B / 3B, Llama 3.1 8B (``size="8B"``); the first uploader in ``LLAMA_ORGS`` that has it (else the
    first's path)."""
    name = "Meta-Llama-3.1-8B-Instruct" if size == "8B" else f"Llama-3.2-{size}-Instruct"
    paths = [os.path.join(MODELS_DIR, org, f"{name}-GGUF", f"{name}{'.' if org == 'MaziyarPanahi' else '-'}{quant}.gguf")
             for org in LLAMA_ORGS]
    return next((p for p in paths if os.path.exists(p)), paths[0])


def llama_safetensors(size: str = "1B") -> str:
    """The bf16 ``model.safetensors`` of the Llama 3.2 Instruct Hugging Face model."""
    return os.path.join(MODELS_DIR, LLAMA_ORG, f"Llama-3.2-{size}-Instruct", "model.safetensors")


def gemma_gguf(quant: str = "Q4_0", size: str = "E4B") -> str:
    """The Gemma 4 instruct GGUF file of a quantization: ggml-org's Q4_0 / Q8_0 / BF16,
    lmstudio-community's Q4_K_M (E4B)."""
    org = "lmstudio-community" if quant == "Q4_K_M" else "ggml-org"
    name = f"gemma-4-{size}-it"
    return os.path.join(MODELS_DIR, org, f"{name}-GGUF", f"{name}-{quant}.gguf")


def qwen_gguf(quant: str = "Q4_K_M", size: str = "27B") -> str:
    """The GGUF file of a qwen35-architecture model: Qwen3.6-27B (``size="27B"``), its MTP variant
    Qwen3.8-27B (``"3.8-27B"``), Qwen3.5-9B (``"9B"``), the language model of MiniCPM-V 4.7 1B (``"1B"``,
    openbmb's: ``minicpmv_gguf``); the MoE Qwen3.6-35B-A3B (``"35B-A3B"``, GGUF architecture ``qwen35moe``)."""
    if size == "1B":
        return minicpmv_gguf(quant)
    name = {"27B": "Qwen3.6-27B", "3.8-27B": "Qwen3.8-27B", "9B": "Qwen3.5-9B", "35B-A3B": "Qwen3.6-35B-A3B"}[size]
    return os.path.join(MODELS_DIR, "lmstudio-community", f"{name}-GGUF", f"{name}-{quant}.gguf")


def minicpmv_gguf(quant: str = "Q4_K_M") -> str:
    """A file of openbmb's MiniCPM-V 4.7 1B GGUF repository: the language model of a quantization (GGUF architecture
    ``qwen35``)."""
    return os.path.join(MODELS_DIR, "openbmb", "MiniCPM-V-4.7-1B-gguf", f"MiniCPM-V-4_7-1B-{quant}.gguf")


def minicpmv_mmproj(compression: int = 16) -> str:
    """MiniCPM-V 4.7 1B's vision encoder and projector (GGUF architecture ``clip``, projector ``minicpmv4_7``): 16x
    token compression (the ViT merger in the encoder: 64 tokens per 448 x 448 slice), or 4x (without it: 256)."""
    assert compression in (4, 16), compression
    return os.path.join(MODELS_DIR, "openbmb", "MiniCPM-V-4.7-1B-gguf",
                        "mmproj-model-f16.gguf" if compression == 16 else "mmproj-model-f16-4x.gguf")


def assistant_of(model: str) -> str:
    """Where the multi-token-prediction drafter (``gemma4-assistant``) of a Gemma 4 model file is
    expected: ``mtp-<model file>`` next to it."""
    return os.path.join(os.path.dirname(os.path.abspath(model)), "mtp-" + os.path.basename(model))


def sd15_dir() -> str:
    """The diffusers directory of Stable Diffusion 1.5 (``model_index.json``, ``tokenizer/``, ``text_encoder/``,
    ``unet/``, ``vae/``, ``scheduler/``)."""
    return os.path.join(MODELS_DIR, "stable-diffusion-v1-5", "stable-diffusion-v1-5")


@dataclass(frozen=True)
class LlamaConfig:
    """Architecture of a Llama 3.x model, from the metadata of its GGUF file (llama.cpp's ``src/models/llama.cpp``).

    ``num_attention_heads`` query heads over ``num_key_value_heads`` KV heads of ``head_dim``; RoPE on every dim, pair
    i turning by p base^(-2 i / head_dim) / ``rope_freq_factors[i]``: llama.cpp's ``rope_freqs`` tensor, the llama3
    frequency scaling of Llama 3.1 / 3.2 (low frequencies stretched for the 128K context) that its converter computes
    from the ``rope_scaling`` of the model's config.json."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_base: float
    rope_freq_factors: tuple[float, ...] | None
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]

    EOG: ClassVar[tuple[str, ...]] = ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>")  # (llama.cpp's end-of-generation)

    @classmethod
    def from_gguf(cls, gguf: Any) -> LlamaConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch == "llama", f"{gguf.path}: architecture {gguf.arch!r}, expected llama"
        m = gguf.arch_meta
        if m("rope.scaling.type", "none") != "none":
            raise NotImplementedError(f"{gguf.path}: RoPE scaling {m('rope.scaling.type')!r}")
        hidden, heads = int(m("embedding_length")), int(m("attention.head_count"))
        hd = int(m("attention.key_length", hidden // heads))
        assert int(m("rope.dimension_count", hd)) == hd and int(m("attention.value_length", hd)) == hd
        vocab = gguf.string_count("tokenizer.ggml.tokens")
        tail = max(0, vocab - 256)  # (the special tokens)
        eos = [int(gguf.meta["tokenizer.ggml.eos_token_id"])]
        eos += [i for i, t in enumerate(gguf.strings("tokenizer.ggml.tokens", tail), tail) if t in cls.EOG]
        freqs = gguf.tensor("rope_freqs.weight").raw.reshape(-1) if "rope_freqs.weight" in gguf else None
        return cls(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=int(m("feed_forward_length")),
            num_hidden_layers=int(m("block_count")),
            num_attention_heads=heads,
            num_key_value_heads=int(m("attention.head_count_kv", heads)),
            head_dim=hd,
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            rope_base=float(m("rope.freq_base", 10000.0)),
            rope_freq_factors=None if freqs is None else tuple(float(x) for x in freqs),
            max_position_embeddings=int(m("context_length")),
            eos_token_ids=tuple(dict.fromkeys(eos)),
        )

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def rope_inv_freq(self) -> np.ndarray:
        """RoPE frequencies (head_dim / 2,) float64: base^(-2 i / head_dim) / rope_freq_factors[i]."""
        d = self.head_dim
        inv = self.rope_base ** (-np.arange(0, d, 2, dtype=np.float64) / d)
        return inv if self.rope_freq_factors is None else inv / np.asarray(self.rope_freq_factors, np.float64)


@dataclass(frozen=True)
class GemmaConfig:
    """Architecture of a Gemma 4 model, from the metadata of its GGUF file (no config.json
    needed; the fields follow llama.cpp's ``src/models/gemma4.cpp``).

    Layers are sliding-window (``swa[l]``: the last ``sliding_window`` positions, head_dim
    ``head_dim_swa``, RoPE base ``rope_base_swa``) or full attention (``head_dim_full``,
    ``rope_base_full`` with the per-pair divisors ``rope_freq_factors``). The last
    ``shared_kv_layers`` layers have no K / V projections: they read the cache of the last
    layer of their kind that has one (``kv_source``). ``per_layer_dim`` > 0: per-layer
    embeddings (E2B / E4B)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    kv_heads: tuple[int, ...]
    swa: tuple[bool, ...]
    head_dim_full: int
    head_dim_swa: int
    sliding_window: int
    shared_kv_layers: int
    per_layer_dim: int
    rms_norm_eps: float
    rope_base_full: float
    rope_base_swa: float
    rope_freq_factors: tuple[float, ...] | None
    final_logit_softcapping: float
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_gguf(cls, gguf: Any) -> GemmaConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch == "gemma4", f"{gguf.path}: architecture {gguf.arch!r}, expected gemma4"
        m = gguf.arch_meta
        n = int(m("block_count"))
        kv = m("attention.head_count_kv")
        kv = (int(kv),) * n if np.isscalar(kv) else tuple(int(x) for x in kv)
        tokens = gguf.strings("tokenizer.ggml.tokens")
        eog = [i for i, t in enumerate(tokens[:1024]) if t in ("<eos>", "<turn|>")]
        freqs = gguf.tensor("rope_freqs.weight").raw.reshape(-1) if "rope_freqs.weight" in gguf else None
        return cls(
            vocab_size=len(tokens),
            hidden_size=int(m("embedding_length")),
            intermediate_size=int(m("feed_forward_length")),
            num_hidden_layers=n,
            num_attention_heads=int(m("attention.head_count")),
            kv_heads=kv,
            swa=tuple(bool(x) for x in m("attention.sliding_window_pattern")),
            head_dim_full=int(m("attention.key_length")),
            head_dim_swa=int(m("attention.key_length_swa")),
            sliding_window=int(m("attention.sliding_window")),
            shared_kv_layers=int(m("attention.shared_kv_layers", 0)),
            per_layer_dim=int(m("embedding_length_per_layer_input", 0)),
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            rope_base_full=float(m("rope.freq_base")),
            rope_base_swa=float(m("rope.freq_base_swa", 10000.0)),
            rope_freq_factors=None if freqs is None else tuple(float(x) for x in freqs),
            final_logit_softcapping=float(m("final_logit_softcapping", 0.0)),
            max_position_embeddings=int(m("context_length")),
            eos_token_ids=tuple(eog),
        )

    @property
    def kv_layers(self) -> int:
        """Layers 0 .. kv_layers - 1 compute K and V (and own a cache)."""
        return self.num_hidden_layers - self.shared_kv_layers

    def head_dim(self, layer: int) -> int:
        return self.head_dim_swa if self.swa[layer] else self.head_dim_full

    def kv_source(self, layer: int) -> int:
        """The layer whose K / V cache ``layer`` attends to (llama.cpp's layer reuse: a layer
        without K / V reads the last SWA layer's cache, kv_layers - 2, or the last full one's,
        kv_layers - 1)."""
        if layer < self.kv_layers:
            return layer
        src = self.kv_layers - (2 if self.swa[layer] else 1)
        assert self.swa[src] == self.swa[layer], "the shared caches are the last SWA and the last full layer"
        return src

    def rope_inv_freq(self, *, swa: bool) -> np.ndarray:
        """RoPE frequencies (head_dim / 2,) float64 of a layer kind: base^(-2 i / head_dim),
        full-attention layers divided by ``rope_freq_factors`` (1e30: the pair is not rotated)."""
        d = self.head_dim_swa if swa else self.head_dim_full
        inv = (self.rope_base_swa if swa else self.rope_base_full) ** (-np.arange(0, d, 2, dtype=np.float64) / d)
        if not swa and self.rope_freq_factors is not None:
            inv = inv / np.asarray(self.rope_freq_factors[:d // 2], np.float64)
        return inv


@dataclass(frozen=True)
class QwenConfig:
    """Architecture of a qwen35 model (Qwen3.5 / 3.6 / 3.8), from the metadata of its GGUF file (llama.cpp's
    ``src/models/qwen35.cpp``).

    Hybrid layers: Gated DeltaNet (GDN, "linear attention") layers and full-attention layers (``attention[l]``:
    every ``full_attention_interval``-th). GDN: ``ssm_k_heads`` q / k heads and ``ssm_v_heads`` v heads of
    ``ssm_state`` dims (v head j reads q / k head j % ssm_k_heads), a causal depthwise conv of ``ssm_conv``
    taps over [q | k | v]. Full attention: ``num_attention_heads`` query heads (each with a sigmoid output
    gate) over ``num_key_value_heads`` KV heads of ``head_dim``; q / k RMS-normed per head, RoPE (NeoX
    pairs) on the first ``rope_dims`` dims only, in ``rope_sections`` of pairs whose positions differ for image
    embeddings only (M-RoPE: ``rope_section_order`` maps an image token's position components t, y, x, z to the
    sections, llama.cpp's "tyxz" or MiniCPM-V's "zyxt"). ``nextn_layers``: multi-token-prediction blocks after the
    trunk's layers (Qwen3.8).

    A ``qwen35moe`` file (Qwen3.6-35B-A3B; ``src/models/qwen35moe.cpp``) has the same layers with an MoE FFN instead of
    the dense one (``intermediate_size`` 0): ``n_experts`` experts of width ``expert_ffn``, the ``n_experts_used``
    largest of a softmax over the router's logits per token (their weights renormalized), plus a shared expert of width
    ``shared_ffn`` times a sigmoid gate (``moe``)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention: tuple[bool, ...]
    ssm_conv: int
    ssm_state: int
    ssm_k_heads: int
    ssm_v_heads: int
    rms_norm_eps: float
    rope_base: float
    rope_dims: int
    rope_sections: tuple[int, ...]
    rope_section_order: str
    nextn_layers: int
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]
    sampling: dict[str, float]
    n_experts: int = 0
    n_experts_used: int = 0
    expert_ffn: int = 0
    shared_ffn: int = 0

    @classmethod
    def from_gguf(cls, gguf: Any) -> QwenConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch in ("qwen35", "qwen35moe"), f"{gguf.path}: architecture {gguf.arch!r}, expected qwen35(moe)"
        m = gguf.arch_meta
        moe = gguf.arch == "qwen35moe"
        nextn = int(m("nextn_predict_layers", 0))
        n = int(m("block_count")) - nextn
        rec = m("attention.recurrent_layers")
        if rec is not None:
            attention = tuple(not bool(x) for x in rec[:n])
        else:
            every = int(m("full_attention_interval", 4))
            attention = tuple((i + 1) % every == 0 for i in range(n))
        ssm_state, inner, v_heads = int(m("ssm.state_size")), int(m("ssm.inner_size")), int(m("ssm.time_step_rank"))
        assert inner == ssm_state * v_heads, (inner, ssm_state, v_heads)
        assert int(m("attention.key_length")) == int(m("attention.value_length"))
        eos = [int(gguf.meta["tokenizer.ggml.eos_token_id"])]
        vocab = gguf.string_count("tokenizer.ggml.tokens")
        last = gguf.strings("tokenizer.ggml.tokens", vocab - 512)
        eos += [i for i, t in enumerate(last, vocab - 512) if t in ("<|im_end|>", "<|endoftext|>")]
        return cls(
            vocab_size=vocab,
            hidden_size=int(m("embedding_length")),
            intermediate_size=0 if moe else int(m("feed_forward_length")),
            num_hidden_layers=n,
            num_attention_heads=int(m("attention.head_count")),
            num_key_value_heads=int(m("attention.head_count_kv")),
            head_dim=int(m("attention.key_length")),
            attention=attention,
            ssm_conv=int(m("ssm.conv_kernel")),
            ssm_state=ssm_state,
            ssm_k_heads=int(m("ssm.group_count")),
            ssm_v_heads=v_heads,
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            rope_base=float(m("rope.freq_base")),
            rope_dims=int(m("rope.dimension_count")),
            rope_sections=tuple(int(x) for x in m("rope.dimension_sections")),
            rope_section_order=str(m("rope.section_order", "tyxz")),
            nextn_layers=nextn,
            max_position_embeddings=int(m("context_length")),
            eos_token_ids=tuple(dict.fromkeys(eos)),
            sampling={k: float(gguf.meta[f"general.sampling.{k}"])
                      for k in ("top_k", "top_p", "temp", "penalty_repeat", "penalty_last_n")
                      if f"general.sampling.{k}" in gguf.meta},
            n_experts=int(m("expert_count")) if moe else 0,
            n_experts_used=int(m("expert_used_count")) if moe else 0,
            expert_ffn=int(m("expert_feed_forward_length")) if moe else 0,
            shared_ffn=int(m("expert_shared_feed_forward_length")) if moe else 0,
        )

    @property
    def moe(self) -> bool:
        return self.n_experts > 0

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i, a in enumerate(self.attention) if not a]

    @property
    def attention_layers(self) -> list[int]:
        return [i for i, a in enumerate(self.attention) if a]

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def rope_theta(self, positions: np.ndarray | range | list) -> np.ndarray:
        """(n, rope_dims / 2) float32 angles of positions: (n,) text positions or (n, 3) M-RoPE ones (T, H, W). Pair i
        turns by its component's p * base^(-2 i / rope_dims), the factor built by repeated float32 multiplication as
        llama.cpp's CPU RoPE (``ggml_mrope_cache_init``); the components interleave by ``rope_sections`` (transformers'
        ``apply_interleaved_mrope``): H for i % 3 == 1 below 3 sections[1], W for i % 3 == 2 below 3 sections[2], T
        otherwise."""
        scale = np.float32(np.power(np.float32(self.rope_base), np.float32(-2.0 / self.rope_dims)))
        n = self.rope_dims // 2
        p = np.asarray(positions, np.float32)
        th = np.repeat(p.reshape(-1, 1), 3, 1) if p.ndim < 2 else p.copy()
        s = self.rope_sections
        comp = [1 if i % 3 == 1 and i < 3 * s[1] else 2 if i % 3 == 2 and i < 3 * s[2] else 0 for i in range(n)]
        out = np.empty((th.shape[0], n), np.float32)
        for i in range(n):
            out[:, i] = th[:, comp[i]]
            th = (th * scale).astype(np.float32)
        return out


@dataclass(frozen=True)
class ClipConfig:
    """A vision encoder and projector, from the metadata of its GGUF file (architecture ``clip``; llama.cpp's
    ``tools/mtmd/clip.cpp``): ``projector`` (``clip.projector_type``: "minicpmv4_7"), the ViT (``layers`` of
    ``hidden`` with ``heads``, ``ffn``, LayerNorm ``eps``), its input (``image_size`` pixels, ``patch_size``, the
    normalization ``mean`` / ``std``), the language model's width ``projection_dim``, ``merge`` (``projector.scale_factor``:
    patches per output token per side, MiniCPM-V 4 or 2), ``max_slices`` and the layer after which the ViT merger runs
    (``merger_layer``: ``wa_layer_indexes``, -1 without one)."""

    projector: str
    image_size: int
    patch_size: int
    hidden: int
    ffn: int
    layers: int
    heads: int
    eps: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    projection_dim: int
    merge: int
    max_slices: int
    merger_layer: int

    @classmethod
    def from_gguf(cls, gguf: Any) -> ClipConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch == "clip", f"{gguf.path}: architecture {gguf.arch!r}, expected clip"
        m = lambda k, d=None: gguf.meta.get(f"clip.vision.{k}", d)
        merge = int(m("projector.scale_factor", 4))
        wa = m("wa_layer_indexes")
        return cls(
            projector=str(gguf.meta["clip.projector_type"]),
            image_size=int(m("image_size")),
            patch_size=int(m("patch_size")),
            hidden=int(m("embedding_length")),
            ffn=int(m("feed_forward_length")),
            layers=int(m("block_count")),
            heads=int(m("attention.head_count")),
            eps=float(m("attention.layer_norm_epsilon")),
            mean=tuple(float(x) for x in m("image_mean")),
            std=tuple(float(x) for x in m("image_std")),
            projection_dim=int(m("projection_dim")),
            merge=merge,
            max_slices=int(m("max_slice_nums", 9)),
            merger_layer=int(wa[0]) if merge == 4 and wa is not None else -1,
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads
