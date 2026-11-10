"""Model architectures from the metadata of a GGUF file (:class:`LlamaConfig`, :class:`GemmaConfig`,
:class:`QwenConfig`, :class:`LfmConfig`), and where model files live.

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


def occamy_gguf() -> str:
    """IsValorum's Occamy 1.0 (Accio-Lab's fine-tune of Qwen3.6-35B-A3B, abliterated) in its APEX-I-MiniPlus V2.1
    quantization (GGUF architecture ``qwen35moe``: IQ3_XXS and Q3_K experts, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32
    tensors)."""
    return os.path.join(MODELS_DIR, "IsValorum", "Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated-GGUF",
                        "Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf")


def swift_gguf() -> str:
    """IsValorum's Swift 1.5 (a fine-tune of Qwen3.8-27B, uncensored) in its APEX-I-MiniPlus V3 quantization (GGUF
    architecture ``qwen35`` with an MTP block: IQ4_XS downs, IQ3_S and IQ3_XXS gate / up, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0,
    Q4_1 and F32 tensors)."""
    return os.path.join(MODELS_DIR, "IsValorum", "Swift-1.5-Qwen3.8-27B-Uncensored-MTP-VAL-APEX-I-MiniPlus-V3-GGUF",
                        "Swift-1.5-Qwen3.8-27B-Uncensored-VAL-APEX-I-MiniPlus-V3.gguf")


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


QWEN3_FILES = {"0.6B": "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_1.gguf",
               "embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
               "embedding-4B": "Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q4_K_M.gguf",
               "reranker-0.6B": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"}


def qwen3_gguf(name: str = "0.6B") -> str:
    """A GGUF file of the ``qwen3`` architecture: Qwen3 0.6B (``"0.6B"``: unsloth's Q4_1), Qwen's Qwen3-Embedding 0.6B
    (Q8_0) and 4B (Q4_K_M), ggml-org's Qwen3-Reranker-0.6B (Q8_0)."""
    return os.path.join(MODELS_DIR, QWEN3_FILES[name])


LFM3_QUANTS = ("QAD-Q4_0", "Q4_K_M", "Q8_0", "BF16")


def lfm3_gguf(quant: str = "QAD-Q4_0") -> str:
    """A file of LiquidAI's LFM3-1.7B GGUF repository: ``QAD-Q4_0`` (Liquid's quantization-aware-distilled Q4_0 layers,
    a Q6_K embedding), ``Q4_K_M``, ``Q8_0``, ``BF16``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-1.7B-GGUF", f"LFM3-1.7B-{quant}.gguf")


def lfm3vl_gguf(quant: str = "QAD-Q4_0") -> str:
    """A language model file of LiquidAI's LFM3-VL-2B GGUF repository (GGUF architecture ``lfm3``: LFM3-1.7B's shapes):
    ``QAD-Q4_0``, ``Q4_K_M``, ``Q8_0``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-VL-2B-GGUF", f"LFM3-VL-2B-{quant}.gguf")


def lfm3vl_mmproj(quant: str = "BF16") -> str:
    """LFM3-VL-2B's vision encoder and projector (GGUF architecture ``clip``, projector ``lfm3vl``): ``BF16``, ``F16``,
    ``Q8_0``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-VL-2B-GGUF", f"mmproj-LFM3-VL-2B-{quant}.gguf")


DECISION_FILES = {"d3-lite":"ggml-org/vllm-d3-lite-GGUF/vllm-d3-lite-Q8_0.gguf",
                  "d3-nano": "ggml-org/vllm-d3-nano-GGUF/vllm-d3-nano-Q8_0.gguf",
                  "kev": "ggml-org/Kev-4B-GGUF/Kev-4B-Q4_K_M.gguf",
                  "kai": "vllm-sr/Decision-2.0-Kai-0.6B/Decision-2.0-Kai-0.6B-Q8_0.gguf",
                  "d1-2b": "LiquidAI/d1-2B-GGUF/d1-2B-Q4_0.gguf"}


def decision_gguf(name: str = "d3-lite") -> str:
    """A decision model's GGUF file (``{arch}.decision.*`` metadata: llama-server's ``/v1/systemone``): vllm-sr's d3-lite
    and d3-nano (``qwen35``, non-causal: ggml-org's Q8_0), Kev-4B (``qwen35``: ggml-org's Q4_K_M), Decision-2.0-Kai-0.6B
    (``vllm-decision2``: Qwen3 0.6B and a head; llama.cpp's converter run on vllm-sr's checkpoint, whose own GGUF
    is a fork's format, Q8_0), LiquidAI's d1-2B (``lfm3``, non-causal full layers: ``d1_gguf``'s Q4_0)."""
    return os.path.join(MODELS_DIR, DECISION_FILES[name])


def d1_gguf(quant: str = "Q4_0") -> str:
    """A file of LiquidAI's d1-2B GGUF repository (an LFM3 decision model): ``Q4_0`` (a Q6_K embedding), ``Q4_K_M``,
    ``Q8_0``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "d1-2B-GGUF", f"d1-2B-{quant}.gguf")


def assistant_of(model: str) -> str:
    """Where the multi-token-prediction drafter (``gemma4-assistant``) of a Gemma 4 model file is
    expected: ``mtp-<model file>`` next to it."""
    return os.path.join(os.path.dirname(os.path.abspath(model)), "mtp-" + os.path.basename(model))


MTP_QUANTS = ("Q4_0", "Q4_K_M", "Q8_0")  # (a qwen35 drafter file by preference: the fewest bytes per draft first)


def qwen_mtp_of(model: str, quant: str = "") -> str | None:
    """The multi-token-prediction drafter (``mtp-*.gguf``, or an uploader's ``<name>-mtp-*.gguf``) of a qwen35 /
    qwen35moe model file without a block of its own: next to it, else in another organization's repository of the same name beside its own (ggml-org publishes
    the drafters apart from other uploaders' models); of several, the first of ``MTP_QUANTS`` (``quant``: that
    quantization's only). None: there is none."""
    d = os.path.dirname(os.path.abspath(model))
    root, repo = os.path.dirname(os.path.dirname(d)), os.path.basename(d)
    others = sorted(os.path.join(root, org, repo) for org in os.listdir(root) if os.path.isdir(os.path.join(root, org, repo)))

    def rank(f: str) -> int:
        return next((i for i, q in enumerate(MTP_QUANTS) if f.endswith(f"-{q}.gguf")), len(MTP_QUANTS))

    for x in dict.fromkeys([d, *others]):
        files = [f for f in os.listdir(x) if (f.startswith("mtp-") or "-mtp-" in f)
                 and f.endswith(f"-{quant}.gguf" if quant else ".gguf")]
        if files:
            return os.path.join(x, min(sorted(files), key=rank))
    return None


def qwen_dflash_of(model: str) -> str | None:
    """The DFlash2 drafter (GGUF architecture ``dflash``) of a qwen35 model file: the Q4_K_M file of a repository
    ``<model's repository>-DFlash2-GGUF`` of any organization (z-lab's for Qwen3.8-27B), or None."""
    d = os.path.dirname(os.path.abspath(model))
    root, repo = os.path.dirname(os.path.dirname(d)), os.path.basename(d).removesuffix("-GGUF")
    for org in sorted(os.listdir(root)):
        x = os.path.join(root, org, f"{repo}-DFlash2-GGUF")
        files = sorted(f for f in os.listdir(x) if f.endswith("-Q4_K_M.gguf")) if os.path.isdir(x) else []
        if files:
            return os.path.join(x, files[0])
    return None


def sd15_dir() -> str:
    """The diffusers directory of Stable Diffusion 1.5 (``model_index.json``, ``tokenizer/``, ``text_encoder/``,
    ``unet/``, ``vae/``, ``scheduler/``)."""
    return os.path.join(MODELS_DIR, "stable-diffusion-v1-5", "stable-diffusion-v1-5")


@dataclass(frozen=True)
class LlamaConfig:
    """Architecture of a Llama 3.x or Qwen3 model, from the metadata of its GGUF file (llama.cpp's
    ``src/models/llama.cpp``, ``qwen3.cpp``; ``vllm-decision2``: Qwen3 with a decision head).

    ``num_attention_heads`` query heads over ``num_key_value_heads`` KV heads of ``head_dim``; RoPE on every dim, pair
    i turning by p base^(-2 i / head_dim) / ``rope_freq_factors[i]``: llama.cpp's ``rope_freqs`` tensor, the llama3
    frequency scaling of Llama 3.1 / 3.2 (low frequencies stretched for the 128K context) that its converter computes
    from the ``rope_scaling`` of the model's config.json. ``neox``: the pairs are dims (i, i + head_dim / 2) of the
    file's Q / K rows (Qwen3); else llama.cpp's converter interleaved them (Llama: ``gguf_model.unpermute_rows``).
    ``qk_norm``: q and k RMS-normed per head with a weight per dim (``attn_q_norm``, ``attn_k_norm``) before RoPE
    (Qwen3)."""

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
    neox: bool = False
    qk_norm: bool = False

    ARCHS: ClassVar[tuple[str, ...]] = ("llama", "qwen3", "vllm-decision2")
    # (llama.cpp's end-of-generation tokens: Llama 3's, Qwen3's)
    EOG: ClassVar[tuple[str, ...]] = ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>")

    @classmethod
    def from_gguf(cls, gguf: Any) -> LlamaConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch in cls.ARCHS, f"{gguf.path}: architecture {gguf.arch!r}, expected one of {cls.ARCHS}"
        m = gguf.arch_meta
        if m("rope.scaling.type", "none") != "none":
            raise NotImplementedError(f"{gguf.path}: RoPE scaling {m('rope.scaling.type')!r}")
        hidden, heads = int(m("embedding_length")), int(m("attention.head_count"))
        hd = int(m("attention.key_length", hidden // heads))
        assert int(m("rope.dimension_count", hd)) == hd and int(m("attention.value_length", hd)) == hd
        vocab = gguf.string_count("tokenizer.ggml.tokens")
        tail = max(0, vocab - 512)  # (the special tokens: Qwen3's before 256 padding tokens)
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
            neox=gguf.arch != "llama",
            qk_norm="blk.0.attn_q_norm.weight" in gguf,
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
        # (llama.cpp's end-of-generation tokens: <|tool_response>, where a tool call's result would follow)
        eog = [i for i, t in enumerate(tokens[:1024]) if t in ("<eos>", "<turn|>", "<|tool_response>")]
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
class LfmConfig:
    """Architecture of a LiquidAI LFM3 model, from the metadata of its GGUF file (llama.cpp's ``src/models/lfm3.cpp``,
    ``lfm-base.cpp``).

    Every layer is [operator | SwiGLU FFN] with sandwich norms (``h = x + post(op(norm(x)))``, the same for the FFN,
    scale 1). The operator by ``kinds[l]``: "conv" (a short convolution: ``in_proj`` -> [B; C; x], y = C *
    conv(B * x) over the last ``conv_width`` positions per channel, ``out_proj``), "swa" (attention over the last
    ``sliding_window`` positions, RoPE base ``rope_base_swa`` on every pair) or "full" (attention over all positions,
    RoPE base ``rope_base`` on the pairs whose ``rope_freq_factors`` is not 1e30: ``rot_full`` pairs). Attention:
    ``num_attention_heads`` query heads over ``num_key_value_heads`` KV heads of ``head_dim``, q / k RMS-normed per
    head with a weight per dim before RoPE (NeoX pairs: dims i, i + head_dim / 2), a sigmoid output gate per head
    (``attn_gate``: hidden -> heads, o_h *= sigmoid(x w_h) of the operator's normed input). The LM head is
    ``token_embd``. ``causal`` False (d1 decision models): the full-attention layers see the whole prompt, the SWA and
    conv layers stay causal."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    kinds: tuple[str, ...]
    sliding_window: int
    conv_width: int
    rms_norm_eps: float
    rope_base: float
    rope_base_swa: float
    rope_freq_factors: tuple[float, ...] | None
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]
    sampling: dict[str, float]
    causal: bool = True

    ARCHS: ClassVar[tuple[str, ...]] = ("lfm3",)

    @classmethod
    def from_gguf(cls, gguf: Any) -> LfmConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch in cls.ARCHS, f"{gguf.path}: architecture {gguf.arch!r}, expected one of {cls.ARCHS}"
        m = gguf.arch_meta
        n = int(m("block_count"))
        kv = [int(x) for x in m("attention.head_count_kv")]
        swa = [bool(x) for x in m("attention.sliding_window_pattern")]
        kinds = tuple("conv" if kv[i] == 0 else "swa" if swa[i] else "full" for i in range(n))
        hd = int(m("attention.key_length"))
        assert int(m("attention.value_length")) == hd and int(m("rope.dimension_count", hd)) == hd
        assert "blk.0.post_attention_norm.weight" in gguf, "LFM3 without sandwich norms"
        heads_kv = {x for x in kv if x}
        assert len(heads_kv) == 1, kv
        vocab = gguf.string_count("tokenizer.ggml.tokens")
        tail = max(0, vocab - 4096)
        eos = [int(gguf.meta["tokenizer.ggml.eos_token_id"])]
        eos += [i for i, t in enumerate(gguf.strings("tokenizer.ggml.tokens", tail), tail)
                if t in ("<|im_end|>", "<|endoftext|>")]
        freqs = gguf.tensor("rope_freqs.weight").raw.reshape(-1) if "rope_freqs.weight" in gguf else None
        return cls(
            vocab_size=vocab,
            hidden_size=int(m("embedding_length")),
            intermediate_size=int(m("feed_forward_length")),
            num_hidden_layers=n,
            num_attention_heads=int(m("attention.head_count")),
            num_key_value_heads=heads_kv.pop(),
            head_dim=hd,
            kinds=kinds,
            sliding_window=int(m("attention.sliding_window")),
            conv_width=int(m("shortconv.l_cache")),
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            rope_base=float(m("rope.freq_base")),
            rope_base_swa=float(m("rope.freq_base_swa", 10000.0)),
            rope_freq_factors=None if freqs is None else tuple(float(x) for x in freqs),
            max_position_embeddings=int(m("context_length")),
            eos_token_ids=tuple(dict.fromkeys(eos)),
            sampling={k: float(gguf.meta[f"general.sampling.{k}"])
                      for k in ("top_k", "top_p", "temp", "penalty_repeat", "penalty_last_n")
                      if f"general.sampling.{k}" in gguf.meta},
            causal=bool(m("attention.causal", True)),
        )

    def layers(self, kind: str) -> list[int]:
        """The layers of a kind ("conv", "swa", "full"; "attn": both attention kinds)."""
        return [i for i, k in enumerate(self.kinds) if k == kind or (kind == "attn" and k != "conv")]

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rot_full(self) -> int:
        """The rotated pairs of the full-attention layers (the first ones: their frequency factor is not 1e30)."""
        f = self.rope_freq_factors
        return self.head_dim // 2 if f is None else next((i for i, x in enumerate(f) if x > 1e20), len(f))

    def rot(self, kind: str) -> int:
        """The rotated pairs of a layer kind."""
        return self.rot_full if kind == "full" else self.head_dim // 2

    def rope_inv_freq(self, kind: str) -> np.ndarray:
        """RoPE frequencies (rot pairs,) float64 of a layer kind: base^(-2 i / head_dim) (the full head dim's
        progression; the unrotated pairs left out)."""
        d = self.head_dim
        base = self.rope_base if kind == "full" else self.rope_base_swa
        return base ** (-np.arange(0, 2 * self.rot(kind), 2, dtype=np.float64) / d)


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
    trunk's layers (Qwen3.8). ``causal`` False (``attention.causal``: vllm-d3's decision models): the full-attention
    layers see the whole prompt (read in one pass), the GDN layers stay recurrent.

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
    causal: bool = True

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
            causal=bool(m("attention.causal", True)),
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
class DFlashConfig:
    """A DFlash2 block drafter (GGUF architecture ``dflash``, llama.cpp's ``src/models/dflash.cpp``), from the metadata of
    its file: ``layers`` of ``hidden`` with ``ffn``, ``heads`` query heads over ``kv_heads`` KV heads of ``head_dim``
    (RMS-normed per dim, NeoX pairs on all of them), a block of ``block`` positions (``mask_token`` past the anchor),
    each layer's attention over the ``window`` positions before a row and the whole block; per group of ``conv_group``
    channels a conv of ``conv_kernel`` taps along the block; a selector of the ``top_k`` best logits per position in
    ``rank`` dims; ``taps``: the model's layers whose inputs feed its cache."""

    layers: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    block: int
    window: int
    conv_kernel: int
    conv_group: int
    rank: int
    top_k: int
    taps: tuple[int, ...]
    mask_token: int
    rms_norm_eps: float
    rope_base: float

    @property
    def hidden_size(self) -> int:
        return self.hidden

    @classmethod
    def from_gguf(cls, gguf: Any) -> DFlashConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch == "dflash", f"{gguf.path}: architecture {gguf.arch!r}, expected dflash"
        m = gguf.arch_meta
        return cls(layers=int(m("block_count")), hidden=int(m("embedding_length")), ffn=int(m("feed_forward_length")),
                   heads=int(m("attention.head_count")), kv_heads=int(m("attention.head_count_kv")),
                   head_dim=int(m("attention.key_length")), block=int(m("block_size")),
                   window=int(m("attention.sliding_window")), conv_kernel=int(m("conv_kernel_size")),
                   conv_group=int(m("conv_group_size")), rank=int(m("selector_rank")), top_k=int(m("selector_top_k")),
                   taps=tuple(int(x) for x in m("target_layers")),
                   mask_token=int(gguf.meta["tokenizer.ggml.mask_token_id"]),
                   rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")), rope_base=float(m("rope.freq_base")))

    def rope_theta(self, positions: np.ndarray | range | list) -> np.ndarray:
        """(n, head_dim / 2) float32 angles of positions (n,): pair i turns by p * base^(-2 i / head_dim), the factor
        built by repeated float32 multiplication as llama.cpp's CPU RoPE."""
        scale = np.float32(np.power(np.float32(self.rope_base), np.float32(-2.0 / self.head_dim)))
        th = np.asarray(positions, np.float32).reshape(-1)
        out = np.empty((th.size, self.head_dim // 2), np.float32)
        for i in range(self.head_dim // 2):
            out[:, i] = th
            th = (th * scale).astype(np.float32)
        return out


@dataclass(frozen=True)
class ClipConfig:
    """A vision encoder and projector, from the metadata of its GGUF file (architecture ``clip``; llama.cpp's
    ``tools/mtmd/clip.cpp``): ``projector`` (``clip.projector_type``: "minicpmv4_7", "qwen3vl_merger"), the ViT
    (``layers`` of ``hidden`` with ``heads``, ``ffn``, LayerNorm ``eps``), its input (``image_size`` pixels,
    ``patch_size``, the normalization ``mean`` / ``std``), the language model's width ``projection_dim``, ``merge``
    (patches per output token per side: MiniCPM-V's ``projector.scale_factor``, 4 or 2; Qwen3-VL's
    ``spatial_merge_size``, 2), ``max_slices`` and the layer after which the ViT merger runs (``merger_layer``:
    ``wa_layer_indexes``, -1 without one) of MiniCPM-V; Qwen3-VL's ``deepstack``: its layers whose outputs feed
    projectors of their own (``is_deepstack_layers``); ``kv_heads`` (grouped-query attention: LFM3-VL's), an image's
    pixels between ``min_pixels`` and ``max_pixels`` and the resampling filter ``resize`` (0, "": the projector's
    defaults). The projector type is ``clip.projector_type`` or (LFM3-VL's files) ``clip.vision.projector_type``."""

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
    deepstack: tuple[int, ...] = ()
    kv_heads: int = 0
    min_pixels: int = 0
    max_pixels: int = 0
    resize: str = ""

    @classmethod
    def from_gguf(cls, gguf: Any) -> ClipConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch == "clip", f"{gguf.path}: architecture {gguf.arch!r}, expected clip"
        m = lambda k, d=None: gguf.meta.get(f"clip.vision.{k}", d)
        merge = int(m("projector.scale_factor", m("spatial_merge_size", 4)))
        wa = m("wa_layer_indexes")
        return cls(
            projector=str(gguf.meta.get("clip.projector_type", m("projector_type"))),
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
            deepstack=tuple(i for i, d in enumerate(np.asarray(m("is_deepstack_layers", []), bool).tolist()) if d),
            kv_heads=int(m("attention.head_count_kv", m("attention.head_count"))),
            min_pixels=int(m("image_min_pixels", 0)),
            max_pixels=int(m("image_max_pixels", 0)),
            resize=str(m("image_resize_algo", "")),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads
