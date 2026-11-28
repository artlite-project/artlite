"""Model architectures from the metadata of a GGUF file (:class:`LlamaConfig`, :class:`GemmaConfig`,
:class:`QwenConfig`, :class:`LfmConfig`, :class:`SparkConfig`, :class:`FlintConfig`, :class:`InklingConfig`), and where
model files live.

Model directories live under :data:`MODELS_DIR` (``ARTLITE_MODELS_DIR``) as ``<org>/<repository>/`` (the Hugging Face
repository a file comes from):

    <org>/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf, ...

:func:`llama_gguf`, :func:`gemma_gguf`, :func:`qwen_gguf` (and the others below) name the files there, :func:`checkpoint`
the Hugging Face checkpoint directories (``artlite.hf_model``), and :func:`model_path` resolves a path given relative to
it, so nothing else spells the location.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from . import settings

MODELS_DIR = settings.MODELS_DIR.get()
# a llama.cpp source tree with a CPU build (build/bin/libllama.so): the tests' golden tensors and token ids
LLAMA_CPP_DIR = settings.LLAMA_CPP_DIR.get()
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
    """The bf16 ``model.safetensors`` of the Llama 3.2 Instruct Hugging Face model (its directory: ``checkpoint``)."""
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


def unsloth_qwen_gguf(quant: str = "UD-IQ4_XS") -> str:
    """unsloth's Qwen3.8-27B in one of its Unsloth Dynamic quantizations (GGUF architecture ``qwen35``, a layer's gate and
    up often in two formats; UD-IQ4_XS: IQ4_XS, Q5_K, Q4_K, IQ3_S, IQ3_XXS, Q6_K, Q3_K, IQ2_S, IQ2_XS, IQ4_NL, Q2_K, Q8_0 and
    F32 tensors, an MTP block; UD-IQ3_XXS: 14 formats, IQ1_S and IQ1_M FFN tensors in a few layers, an MTP block; UD-IQ1_M:
    IQ1_S, IQ2_XXS, IQ1_M and seven other formats, no MTP block)."""
    return os.path.join(MODELS_DIR, "unsloth", "Qwen3.8-27B-GGUF", f"Qwen3.8-27B-{quant}.gguf")


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
               "reranker-0.6B": "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf",
               "reranker-4B": "giladgd/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B.Q4_K_M.gguf"}


def qwen3_gguf(name: str = "0.6B") -> str:
    """A GGUF file of the ``qwen3`` architecture: Qwen3 0.6B (``"0.6B"``: unsloth's Q4_1), Qwen's Qwen3-Embedding 0.6B
    (Q8_0) and 4B (Q4_K_M), ggml-org's Qwen3-Reranker-0.6B (Q8_0), giladgd's Qwen3-Reranker-4B (Q4_K_M: llama.cpp's
    converter, its classifier head; other 4B conversions of before it have none)."""
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


def moss_gguf(quant: str = "Q4_K_M") -> str:
    """A transcribe.cpp GGUF file of MOSS-Transcribe-Diarize (handy-computer's: the whole model in one file,
    ``artlite.transcribe_cpp``): ``Q4_K_M``, ``Q5_K_M``, ``Q8_0``, ``BF16``."""
    return os.path.join(MODELS_DIR, "handy-computer", "moss-transcribe-diarize-gguf", f"MOSS-Transcribe-Diarize-{quant}.gguf")


def lfm3a_gguf(quant: str = "Q4_K_M") -> str:
    """A language model file of LiquidAI's LFM3-Audio-2B GGUF repository (GGUF architecture ``lfm3``: LFM3-1.7B's shapes
    with separate weights): ``Q4_K_M`` (the card's), ``Q4_0``, ``Q8_0``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-Audio-2B-GGUF", f"LFM3-Audio-2B-{quant}.gguf")


def lfm3a_mmproj(quant: str = "Q8_0") -> str:
    """LFM3-Audio-2B's speech encoder and projector (GGUF architecture ``clip``, projector ``lfm3a``; the file also holds
    the speech decoder, ``lfm3a_gen``): ``Q8_0``, ``BF16``, ``F16``, ``Q4_K_M``, ``Q4_0``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-Audio-2B-GGUF", f"mmproj-LFM3-Audio-2B-{quant}.gguf")


def lfm3omni_gguf(quant: str = "Q4_K_M") -> str:
    """A language model file of LiquidAI's LFM3-Omni-2B GGUF repository (GGUF architecture ``lfm3``: LFM3-1.7B's shapes
    with separate weights): ``Q4_K_M`` (the card's), ``Q4_0``, ``Q8_0``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-Omni-2B-GGUF", f"LFM3-Omni-2B-{quant}.gguf")


def lfm3omni_mmproj(quant: str = "Q8_0") -> str:
    """LFM3-Omni-2B's media file (GGUF architecture ``clip``): its vision tower (projector ``lfm3vl``), speech encoder
    (``lfm3a``) and speech generator (``lfm3a_gen``): ``Q8_0``, ``BF16``, ``F16``."""
    return os.path.join(MODELS_DIR, "LiquidAI", "LFM3-Omni-2B-GGUF", f"mmproj-LFM3-Omni-2B-{quant}.gguf")


DECISION_FILES = {"d3-lite":"ggml-org/vllm-d3-lite-GGUF/vllm-d3-lite-Q8_0.gguf",
                  "d3-nano": "ggml-org/vllm-d3-nano-GGUF/vllm-d3-nano-Q8_0.gguf",
                  "kev": "ggml-org/Kev-4B-GGUF/Kev-4B-Q4_K_M.gguf",
                  "kai": "vllm-sr/Decision-2.0-Kai-0.6B/Decision-2.0-Kai-0.6B-Q8_0.gguf",
                  "d1-2b": "LiquidAI/d1-2B-GGUF/d1-2B-Q4_0.gguf",
                  "d1-omni": "LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf"}


def decision_gguf(name: str = "d3-lite") -> str:
    """A decision model's GGUF file (``{arch}.decision.*`` metadata: llama-server's ``/v1/systemone``): vllm-sr's d3-lite
    and d3-nano (``qwen35``, non-causal: ggml-org's Q8_0), Kev-4B (``qwen35``: ggml-org's Q4_K_M), Decision-2.0-Kai-0.6B
    (``vllm-decision2``: Qwen3 0.6B and a head; llama.cpp's converter run on vllm-sr's checkpoint, whose own GGUF
    is a fork's format, Q8_0), LiquidAI's d1-2B (``lfm3``, non-causal full layers: ``d1_gguf``'s Q4_0) and d1-Omni-2B
    (its Q8_0: audio, video frames and spoken questions too)."""
    return os.path.join(MODELS_DIR, DECISION_FILES[name])


DECISION_MMPROJ = {"d3-lite": "ggml-org/vllm-d3-lite-GGUF/mmproj-vllm-d3-lite-Q8_0.gguf",
                   "d3-nano": "ggml-org/vllm-d3-nano-GGUF/mmproj-vllm-d3-nano-Q8_0.gguf",
                   "d1-2b": "LiquidAI/d1-2B-GGUF/mmproj-d1-2B-Q8_0.gguf",
                   "d1-omni": "LiquidAI/d1-Omni-2B-GGUF/mmproj-d1-Omni-2B-Q8_0.gguf"}


def decision_mmproj(name: str = "d3-lite") -> str:
    """A decision model's vision encoder (images in its requests): d3-lite's and d3-nano's Qwen3-VL towers (768 / 1024
    wide: ggml-org's Q8_0), d1-2B's LFM3-VL encoder (LiquidAI's Q8_0), d1-Omni-2B's (its tower and speech encoder)."""
    return os.path.join(MODELS_DIR, DECISION_MMPROJ[name])


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
    qwen35moe model file that has no MTP block itself: next to the file, else in another organization's repository of
    the same name next to the file's repository (ggml-org publishes the drafters separately from other uploaders'
    models; for a PARO checkpoint directory, they are in its base model's ``<base>-GGUF`` repositories); of several,
    the first of ``MTP_QUANTS`` (``quant``: that quantization's only). None: there is none."""
    d = os.path.dirname(os.path.abspath(model))
    if os.path.isdir(model):  # (a checkpoint directory, a PARO one <base>-PARO[-variant]: its base model's GGUF repositories)
        d = os.path.abspath(model)
        root, repo = os.path.dirname(os.path.dirname(d)), re.sub(r"-PARO(-[\w.]+)?$", "", os.path.basename(d)) + "-GGUF"
    else:
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
    ``<model's repository>-DFlash2-GGUF`` of any organization (z-lab's for Qwen3.8-27B), or None; a checkpoint directory's
    (a PARO one, ``<base>-PARO[-<variant>]``): its base model's."""
    d = os.path.abspath(model) if os.path.isdir(model) else os.path.dirname(os.path.abspath(model))
    root, repo = os.path.dirname(os.path.dirname(d)), os.path.basename(d).removesuffix("-GGUF")
    repo = re.sub(r"-PARO(-[\w.]+)?$", "", repo)
    for org in sorted(os.listdir(root)):
        x = os.path.join(root, org, f"{repo}-DFlash2-GGUF")
        files = sorted(f for f in os.listdir(x) if f.endswith("-Q4_K_M.gguf")) if os.path.isdir(x) else []
        if files:
            return os.path.join(x, files[0])
    return None


SPARK_QUANTS = ("Q4_K_M", "Q8_0", "BF16")


def spark_gguf(size: str = "4B", quant: str = "Q4_K_M", org: str = "XHToken") -> str:
    """A file of a Spark-X2.6 GGUF repository: ``size`` "4B" / "1.7B"; ``quant`` one of ``SPARK_QUANTS`` (XHToken's
    official files: Q4_K_M with Q6_K / Q8_0 in the sensitive tensors and a Q8_0 head beside a BF16 ``token_embd``;
    "BF16": the unquantized file) or a community quantization (``org`` "bartowski": "Q4_K_M", "Q4_0")."""
    name = f"Spark-X2.6-{size}"
    return os.path.join(MODELS_DIR, org, f"{name}-GGUF", f"{name}.gguf" if quant == "BF16" else f"{name}-{quant}.gguf")


def spark_dspark_of(model: str) -> str | None:
    """The DSpark drafter of a Spark-X model file (GGUF architecture ``dflash`` with a Markov head, ``markov_w1``):
    the first file of a repository ``<model's repository>-DSpark-GGUF`` of any organization (XHToken's) whose metadata
    fits the model's (``tokenizer.ggml.pre`` "spark2_5", its hidden size and vocabulary, its taps below its layers),
    or None."""
    from .gguf_model import GGUFFile

    d = os.path.dirname(os.path.abspath(model))
    root, repo = os.path.dirname(os.path.dirname(d)), os.path.basename(d).removesuffix("-GGUF")
    m = None
    for org in sorted(os.listdir(root)):
        x = os.path.join(root, org, f"{repo}-DSpark-GGUF")
        for f in sorted(os.listdir(x)) if os.path.isdir(x) else []:
            if not f.endswith(".gguf"):
                continue
            g = GGUFFile(os.path.join(x, f))
            if m is None:
                m = GGUFFile(model)
            ok = (g.arch == "dflash" and "markov_w1.weight" in g and g.meta.get("tokenizer.ggml.pre") == "spark2_5"
                  and g.arch_meta("embedding_length") == m.arch_meta("embedding_length")
                  and g.string_count("tokenizer.ggml.tokens") == m.string_count("tokenizer.ggml.tokens")
                  and max(g.arch_meta("target_layers")) < int(m.arch_meta("block_count")))
            if ok:
                return os.path.join(x, f)
    return None


FLINT_FILES = {"int4": "RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf",
               "UD-Q4_K_XL": "unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf",
               "Q4_K_M": "lmstudio-community/flint-GGUF/flint-Q4_K_M.gguf"}


def flint_dflash_of(model: str) -> str | None:
    """The DFlash2 drafter (GGUF architecture ``dflash``) of a Flint model file: the Q4_K_M file of a repository
    ``Flint-DFlash2-GGUF`` of any organization of the model's models directory (incoai's), or None."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(model))))
    for org in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        x = os.path.join(root, org, "Flint-DFlash2-GGUF")
        files = sorted(f for f in os.listdir(x) if f.endswith("-Q4_K_M.gguf")) if os.path.isdir(x) else []
        if files:
            return os.path.join(x, files[0])
    return None


def flint_gguf(quant: str = "int4") -> str:
    """A Flint file: ``quant`` "int4" (Recursive's official file: the model's trained INT4 codes, MLX's groups of 64
    stored as Q4_1 block pairs), or a community quantization ("UD-Q4_K_XL": unsloth's, "Q4_K_M": lmstudio's)."""
    return os.path.join(MODELS_DIR, FLINT_FILES[quant])


# (unsloth's repository: its root and ``Mainline_Support/`` files the ``inkling2_loop`` layout, ``Old_Support/``'s
# ``inkling2``)
INKLING_REPO = "unsloth/Inkling-2-Mini-GGUF"


def inkling_gguf(quant: str = "UD-Q4_K_XL") -> str:
    """An Inkling 2 Mini file of unsloth's repository (``quant``: "UD-Q4_K_XL", "UD-IQ1_S", ...), in its root: either
    layout (``InklingConfig``; the ``Old_Support/`` files, renamed to .gguf, too)."""
    return os.path.join(MODELS_DIR, INKLING_REPO, f"Inkling-2-Mini-{quant}.gguf")


def _inkling_dirs(model: str) -> list[str]:
    """(where an Inkling file's companions are: its directory, and the repository's root above ``Mainline_Support`` or
    ``Old_Support``)"""
    d = os.path.dirname(os.path.abspath(model))
    return [d] + ([os.path.dirname(d)] if os.path.basename(d) in ("Mainline_Support", "Old_Support") else [])


def inkling_dspark_of(model: str) -> str | None:
    """The DSpark drafter of an Inkling 2 file (GGUF architecture ``dflash`` with a Markov head, ``markov_w1``; its taps
    count the model's visits, ``InklingConfig.visits``): ``dspark-*.gguf`` beside it or in the repository's root, Q8_0
    first, then its ``dspark/`` directory (BF16), whose vocabulary and hidden size fit the model's; or None."""
    from .gguf_model import GGUFFile

    m = GGUFFile(model)
    for d in [x for r in _inkling_dirs(model) for x in (r, os.path.join(r, "dspark"))]:
        files = sorted((f for f in os.listdir(d) if f.startswith("dspark-") and f.endswith(".gguf")),
                       key=lambda f: (not f.endswith("-Q8_0.gguf"), f)) if os.path.isdir(d) else []
        for f in files:
            g = GGUFFile(os.path.join(d, f))
            if (g.arch == "dflash" and "markov_w1.weight" in g and g.meta.get("tokenizer.ggml.pre") == "inkling2"
                    and g.arch_meta("embedding_length") == m.arch_meta("embedding_length")
                    and g.string_count("tokenizer.ggml.tokens") == m.string_count("tokenizer.ggml.tokens")):
                return os.path.join(d, f)
    return None


def inkling_mmproj_of(model: str) -> str | None:
    """The vision tower of an Inkling 2 file (an mmproj of projector ``inkling2v``): ``mmproj-*.gguf`` beside it or in
    the repository's root, F16 first; or None."""
    from .gguf_model import GGUFFile

    for d in _inkling_dirs(model):
        files = sorted((f for f in os.listdir(d) if f.startswith("mmproj") and f.endswith(".gguf")),
                       key=lambda f: (not f.endswith("-F16.gguf"), f)) if os.path.isdir(d) else []
        for f in files:
            meta = GGUFFile(os.path.join(d, f)).meta
            if meta.get("clip.projector_type", meta.get("clip.vision.projector_type")) == "inkling2v":
                return os.path.join(d, f)
    return None


# Hugging Face checkpoint directories (``artlite.hf_model``: a directory opens as a model file): bf16 / fp16 originals,
# and z-lab's ParoQuant (PARO) checkpoints
HF_CHECKPOINTS = {"qwen3-0.6B": "Qwen/Qwen3-0.6B", "qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
                  "llama-1B": "lmstudio-community/Llama-3.2-1B-Instruct",
                  "qwen3-0.6B-paro": "z-lab/Qwen3-0.6B-PARO", "qwen3-8B-paro": "z-lab/Qwen3-8B-PARO",
                  "qwen3.5-0.8B-paro": "z-lab/Qwen3.5-0.8B-PARO", "qwen3.8-27B-paro": "z-lab/Qwen3.8-27B-PARO",
                  "qwen3.8-27B-paro-int5": "Launch80/Qwen3.8-27B-PARO-int5",
                  "qwen3.6-35B-A3B-paro": "z-lab/Qwen3.6-35B-A3B-PARO", "gemma-4-E2B": "google/gemma-4-E2B-it",
                  "gemma-4-E2B-paro": "z-lab/gemma-4-E2B-it-PARO", "gemma-4-E4B-paro": "z-lab/gemma-4-E4B-it-PARO",
                  "llama-8B-paro": "z-lab/Llama-3.1-8B-Instruct-PARO",
                  "moss-transcribe-diarize": "OpenMOSS-Team/MOSS-Transcribe-Diarize"}


def checkpoint(name: str) -> str:
    """A Hugging Face checkpoint directory of ``HF_CHECKPOINTS`` (by its key)."""
    return os.path.join(MODELS_DIR, HF_CHECKPOINTS[name])


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
    file's Q / K rows (Qwen3, a Llama checkpoint directory: ``hf_model.HfModel.unpermuted``); else llama.cpp's converter
    interleaved them (a Llama GGUF: ``gguf_model.unpermute_rows``).
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

    ARCHS: ClassVar[tuple[str, ...]] = ("llama", "qwen3", "vllm-decision2", "moss")
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
            neox=gguf.arch != "llama" or getattr(gguf, "unpermuted", False),
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
class SparkConfig:
    """Architecture of a Spark-X2.5 / X2.6 model (GGUF ``spark2_5``), from the metadata of its GGUF file (llama.cpp's
    ``src/models/spark2-5.cpp``, HF's ``modeling_spark.py``).

    A dense pre-norm transformer: per layer ``h = x + o(gate * attn(norm1(x)))``, ``x' = h + down(gelu(gate(norm2(h)))
    * up(norm2(h)))`` (GELU's erf form, the model's ``gelu``; llama.cpp computes its tanh form), RMSNorm with the weights
    as stored. Attention: ``num_attention_heads`` query heads over ``num_key_value_heads`` KV heads of ``head_dim``
    (256), fused ``attn_qkv`` rows [q | k | v], no q / k norms (scores q k / sqrt(head_dim)), a sigmoid output gate per
    head (``attn_gate``: hidden -> heads, o_h *= sigmoid(x_n g_h) of the attention's normed input). Layer kinds
    (``kinds[l]``): "full" (all positions; NeoX RoPE on the first ``rope_dims`` dims, base ``rope_base``) and "swa"
    (the last ``sliding_window`` positions, keys p - window + 1 .. p; RoPE on ``rope_dims_swa`` dims, base
    ``rope_base_swa``). The LM head: ``output`` (the official files') or the tied ``token_embd`` (``tied``)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    kinds: tuple[str, ...]
    sliding_window: int
    rms_norm_eps: float
    rope_base: float
    rope_base_swa: float
    rope_dims: int
    rope_dims_swa: int
    tied: bool
    max_position_embeddings: int
    bos_token_id: int
    eos_token_ids: tuple[int, ...]
    sampling: dict[str, float]

    ARCHS: ClassVar[tuple[str, ...]] = ("spark2_5",)
    moe: ClassVar[bool] = False  # (``decode.attn256.AttnConfig``)
    causal: ClassVar[bool] = True

    @classmethod
    def from_gguf(cls, gguf: Any) -> SparkConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch in cls.ARCHS, f"{gguf.path}: architecture {gguf.arch!r}, expected one of {cls.ARCHS}"
        m = gguf.arch_meta
        n = int(m("block_count"))
        swa = [bool(x) for x in m("attention.sliding_window_pattern")]
        hd = int(m("attention.key_length"))
        assert int(m("attention.value_length")) == hd and len(swa) == n
        return cls(
            vocab_size=gguf.string_count("tokenizer.ggml.tokens"),
            hidden_size=int(m("embedding_length")),
            intermediate_size=int(m("feed_forward_length")),
            num_hidden_layers=n,
            num_attention_heads=int(m("attention.head_count")),
            num_key_value_heads=int(m("attention.head_count_kv")),
            head_dim=hd,
            kinds=tuple("swa" if swa[i] else "full" for i in range(n)),
            sliding_window=int(m("attention.sliding_window")),
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            rope_base=float(m("rope.freq_base")),
            rope_base_swa=float(m("rope.freq_base_swa", 10000.0)),
            rope_dims=int(m("rope.dimension_count", hd)),
            rope_dims_swa=int(m("rope.dimension_count_swa", hd)),
            tied="output.weight" not in gguf,
            max_position_embeddings=int(m("context_length")),
            bos_token_id=int(gguf.meta["tokenizer.ggml.bos_token_id"]),
            eos_token_ids=(int(gguf.meta["tokenizer.ggml.eos_token_id"]),),
            sampling={k: float(gguf.meta[f"general.sampling.{k}"])
                      for k in ("top_k", "top_p", "temp", "penalty_repeat", "penalty_last_n")
                      if f"general.sampling.{k}" in gguf.meta},
        )

    def layers(self, kind: str) -> list[int]:
        """The layers of a kind ("swa", "full")."""
        return [i for i, k in enumerate(self.kinds) if k == kind]

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def rot(self, kind: str) -> int:
        """The rotated pairs of a layer kind: pairs (i, i + rot) of the head's first 2 rot dims."""
        return (self.rope_dims if kind == "full" else self.rope_dims_swa) // 2

    def rope_theta(self, positions: np.ndarray | range | list, kind: str = "full") -> np.ndarray:
        """(n, rot) float32 angles of positions for a layer kind: pair i turns by p * base^(-2 i / dims), the factor
        built by repeated float32 multiplication as llama.cpp's CPU RoPE."""
        dims, base = (self.rope_dims, self.rope_base) if kind == "full" else (self.rope_dims_swa, self.rope_base_swa)
        scale = np.float32(np.power(np.float32(base), np.float32(-2.0 / dims)))
        th = np.asarray(positions, np.float32).reshape(-1)
        out = np.empty((th.shape[0], dims // 2), np.float32)
        for i in range(dims // 2):
            out[:, i] = th
            th = (th * scale).astype(np.float32)
        return out


@dataclass(frozen=True)
class FlintConfig:
    """Architecture of Recursive's Flint (GGUF ``flint``), from the metadata of its GGUF file (llama.cpp's
    ``src/models/flint.cpp``).

    A pre-norm transformer: per layer ``h = x + o(attn(norm1(x)))``, ``x' = h + ffn(norm2(h))``, RMSNorm with the
    weights as stored. Attention in every layer: ``num_attention_heads`` query heads over ``num_key_value_heads`` KV
    heads of ``head_dim``, q / k RMS-normed per head with a weight per dim, scores q k / sqrt(head_dim), a learned sink
    per query head (``attn_sinks``: an extra logit in the softmax with no value). Layer kinds (``kinds[l]``): "swa"
    (keys p - ``sliding_window`` + 1 .. p; NeoX RoPE on every dim, base ``rope_base``) and "full" (all positions, no
    positional encoding). FFN: the first ``n_dense`` layers a SwiGLU of ``intermediate_size``; the others an MoE of
    ``n_experts`` SwiGLU experts of ``expert_ffn``, the ``n_experts_used`` largest of sigmoid(router logits) per token
    weighted by their sigmoids over the sum of the chosen ones (``weights_norm``), plus an ungated shared expert of
    ``shared_ffn`` (the model's two shared experts merged). The LM head is ``output``."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    kinds: tuple[str, ...]
    sliding_window: int
    rms_norm_eps: float
    rope_base: float
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]
    sampling: dict[str, float]
    n_dense: int
    n_experts: int
    n_experts_used: int
    expert_ffn: int
    shared_ffn: int
    weights_norm: bool = True

    ARCHS: ClassVar[tuple[str, ...]] = ("flint",)
    # (llama.cpp's end-of-generation tokens of Flint: the end of text, the turns that follow an assistant's)
    EOG: ClassVar[tuple[str, ...]] = ("<|end_of_text|>", "<|user|>", "<|observation|>")
    moe: ClassVar[bool] = True
    causal: ClassVar[bool] = True

    @classmethod
    def from_gguf(cls, gguf: Any) -> FlintConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch in cls.ARCHS, f"{gguf.path}: architecture {gguf.arch!r}, expected one of {cls.ARCHS}"
        m = gguf.arch_meta
        n = int(m("block_count"))
        swa = [bool(x) for x in m("attention.sliding_window_pattern")]
        rope = [bool(x) for x in m("attention.rope_pattern", swa)]
        assert swa == rope and len(swa) == n, "Flint's layers: RoPE exactly in the sliding-window ones"
        hd = int(m("attention.key_length"))
        assert int(m("attention.value_length")) == hd and int(m("rope.dimension_count", hd)) == hd
        gating = int(m("expert_gating_func"))
        scale = float(m("expert_weights_scale", 1.0))
        assert gating == 2 and scale == 1.0, f"{gguf.path}: gating {gating}, weights scale {scale}: sigmoid, 1"
        assert "blk.0.attn_sinks.weight" in gguf and "blk.0.attn_q_norm.weight" in gguf
        vocab = gguf.string_count("tokenizer.ggml.tokens")
        tail = max(0, vocab - 512)
        eos = [int(gguf.meta["tokenizer.ggml.eos_token_id"])]
        eos += [i for i, t in enumerate(gguf.strings("tokenizer.ggml.tokens", tail), tail) if t in cls.EOG]
        eos += [int(x) for x in gguf.meta.get("tokenizer.ggml.eog_token_ids", ())]
        return cls(
            vocab_size=vocab,
            hidden_size=int(m("embedding_length")),
            intermediate_size=int(m("feed_forward_length")),
            num_hidden_layers=n,
            num_attention_heads=int(m("attention.head_count")),
            num_key_value_heads=int(m("attention.head_count_kv")),
            head_dim=hd,
            kinds=tuple("swa" if swa[i] else "full" for i in range(n)),
            sliding_window=int(m("attention.sliding_window")),
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            rope_base=float(m("rope.freq_base")),
            max_position_embeddings=int(m("context_length")),
            eos_token_ids=tuple(dict.fromkeys(eos)),
            sampling={k: float(gguf.meta[f"general.sampling.{k}"])
                      for k in ("top_k", "top_p", "temp", "penalty_repeat", "penalty_last_n")
                      if f"general.sampling.{k}" in gguf.meta},
            n_dense=int(m("leading_dense_block_count")),
            n_experts=int(m("expert_count")),
            n_experts_used=int(m("expert_used_count")),
            expert_ffn=int(m("expert_feed_forward_length")),
            shared_ffn=int(m("expert_shared_feed_forward_length")),
            weights_norm=bool(m("expert_weights_norm", True)),
        )

    def layers(self, kind: str) -> list[int]:
        """The layers of a kind ("swa", "full"; "moe", "dense": by FFN)."""
        if kind in ("moe", "dense"):
            return [i for i in range(self.num_hidden_layers) if self.moe_layer(i) == (kind == "moe")]
        return [i for i, k in enumerate(self.kinds) if k == kind]

    def moe_layer(self, layer: int) -> bool:
        return layer >= self.n_dense

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def rot(self, kind: str) -> int:
        """The rotated pairs of a layer kind: all of the head's (NeoX pairs i, i + head_dim / 2) or none."""
        return self.head_dim // 2 if kind == "swa" else 0

    def rope_inv_freq(self, kind: str) -> np.ndarray:
        """RoPE frequencies (rot pairs,) float64 of a layer kind: base^(-2 i / head_dim)."""
        return self.rope_base ** (-np.arange(0, 2 * self.rot(kind), 2, dtype=np.float64) / self.head_dim)

    def rope_theta(self, positions: np.ndarray | range | list, kind: str = "swa") -> np.ndarray:
        """(n, rot) float32 angles of positions for a layer kind: pair i turns by p * base^(-2 i / head_dim), the factor
        built by repeated float32 multiplication as llama.cpp's CPU RoPE."""
        scale = np.float32(np.power(np.float32(self.rope_base), np.float32(-2.0 / self.head_dim)))
        th = np.asarray(positions, np.float32).reshape(-1)
        out = np.empty((th.shape[0], self.rot(kind)), np.float32)
        for i in range(self.rot(kind)):
            out[:, i] = th
            th = (th * scale).astype(np.float32)
        return out


@dataclass(frozen=True)
class Visit:
    """One of a token's visits of a physical layer of Inkling 2 (``InklingConfig.visits``, in the order a token runs
    them): each visit has its own KV cache and convolution histories and uses the layer's weights."""

    index: int  # v, 0 .. visits - 1 (a DSpark drafter's ``target_layers`` count these: the residual entering visit v;
    # ``visits``: the final state, after the output's depth mix and norm)
    layer: int  # the physical layer
    loop: int  # the loop core's pass (0 ..); -1: the prelude or the coda
    slot: int  # llama.cpp's cache slot (its tensors' layer index): the physical layer in the first pass, then
    # layers + (pass - 1) x core + (layer - core_start)
    kind: str  # "swa" or "full"
    block: bool  # the layer begins a block (layer % block_size == 0): its input becomes a block state, the residual
    # restarts from its attention's output
    queries: int  # its attention input's depth-mix queries: 0 (layer 0: the normed embedding itself), 1 (sliding
    # window: Q, K and V from one input), 3 (full: one each)

    @property
    def messages(self) -> bool:
        """The cross-pass messages and their controllers (the loop's passes after the first)."""
        return self.loop > 0


@dataclass(frozen=True)
class InklingConfig:
    """Architecture of Thinking Machines' Inkling 2 (GGUF ``inkling2_loop``: ggml-org's and unsloth's root and
    ``Mainline_Support`` files; unsloth's ``Old_Support`` files, ``inkling2`` with ``num_loops``, map onto it:
    ``layout``), from the metadata of its GGUF file (llama.cpp's ``src/models/inkling2.cpp``, ``inkling2-loop.cpp``).

    ``num_hidden_layers`` physical layers in blocks of ``block_size``; the loop core (layers ``core_start`` ..
    ``core_end`` - 1) runs ``passes`` times: a token visits the prelude, the core ``passes`` times, the coda
    (``visits``). Each visit's inputs are depth mixes ("block attention residuals"): the completed block states and the
    partial residual RMS-normed without weight, softmax over them of q . x / sqrt(hidden) for learned fp32 queries;
    the passes after the first mix in a message per stream (q, k, v, ffn: ``CTRL_STREAMS``) from the previous pass's
    bank (its block states after the core's first, its output) through a controller each (D: 2 hidden -> ``gate_rank``,
    gain and forget: back; out = (1 - tanh(forget g)) x + tanh(gain g) message). Attention: ``num_attention_heads`` over
    ``num_key_value_heads`` of ``head_dim``, no RoPE: K and V a causal conv of ``conv_taps`` taps (full layers: plus the
    taps' per-token deltas), K L2-clipped to ``qk_clip``, V RMS-normed per head, Q scaled per dim and clipped, a
    relative-position bias R . E[:, distance] (R: ``rel_length`` per head, clipped to ``rel_clip``; ``rel_extent``
    distances; ``sliding_window`` on the "swa" layers), Q and R times ``tau`` on full layers, a per-query sink logit,
    the output gated by sigmoid(gate). A conv after the attention's and the FFN's output projections. FFN: ``n_experts``
    experts of ``expert_ffn``, the ``n_experts_used`` largest of sigmoid(router) + bias, weighted by their sigmoids over
    their sum times ``expert_weights_scale``; "situ" activation beta tanh(g / beta) sigmoid(g) lbeta tanh(u / lbeta)
    (``situ_beta``, ``situ_linear_beta``). The LM head's logits softcapped at ``softcap``; the embedding and head have
    ``vocab_padded`` rows, of which the first ``vocab_size`` are tokens."""

    vocab_size: int
    vocab_padded: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    kinds: tuple[str, ...]
    sliding_window: int
    rms_norm_eps: float
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]
    sampling: dict[str, float]
    n_experts: int
    n_experts_used: int
    expert_ffn: int
    expert_weights_scale: float
    situ_beta: float
    situ_linear_beta: float
    block_size: int
    conv_taps: int
    rel_length: int
    rel_extent: int
    qk_clip: float
    rel_clip: float
    log_scale_floor: int
    log_scale_alpha: float
    softcap: float
    core_start: int
    core_end: int
    passes: int
    gate_rank: int
    layout: str  # "loop" (``inkling2_loop``) or "root" (unsloth's ``inkling2`` with ``num_loops``)

    ARCHS: ClassVar[tuple[str, ...]] = ("inkling2_loop", "inkling2")
    CTRL_STREAMS: ClassVar[tuple[str, ...]] = ("q", "k", "v", "ffn")  # (llama.cpp's LOOP_CTRL_* order)
    _ROOT_CTRL: ClassVar[dict[str, str]] = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}
    EOG: ClassVar[tuple[str, ...]] = ("<|endoftext|>",)  # (llama.cpp's end-of-generation tokens beside the EOS)
    MIX_EPS: ClassVar[float] = 1e-6  # (the depth mixes' and controllers' unweighted RMS norms)

    @classmethod
    def from_gguf(cls, gguf: Any) -> InklingConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch in cls.ARCHS, f"{gguf.path}: architecture {gguf.arch!r}, expected one of {cls.ARCHS}"
        m = gguf.arch_meta
        if gguf.arch == "inkling2_loop":
            layout, n = "loop", int(m("block_count"))
            start, end, passes = int(m("loop.start")), int(m("loop.end")), int(m("loop.count"))
            if bool(m("loop.varying", False)):
                raise NotImplementedError(f"{gguf.path}: a varying loop (the varloop research checkpoint)")
        else:
            if m("num_loops") is None:
                raise NotImplementedError(f"{gguf.path}: an inkling2 file without a loop")
            layout = "root"
            start, end, passes = int(m("loop.core_start")), int(m("loop.core_end")), int(m("num_loops"))
            n = int(m("block_count")) - (passes - 1) * (end - start)  # (its blocks count the cache slots)
        block = int(m("attn_res.block_size"))
        assert 0 < start < end <= n and start % block == 0 and end % block == 0 and passes >= 2, \
            f"{gguf.path}: loop core {start} .. {end} of {n} layers, {passes} passes, blocks of {block}"
        swa = [bool(x) for x in m("attention.sliding_window_pattern")][:n]
        heads = {int(x) for x in np.atleast_1d(m("attention.head_count"))}
        kv = {int(x) for x in np.atleast_1d(m("attention.head_count_kv"))}
        hd = int(m("attention.key_length"))
        assert len(heads) == len(kv) == 1 and int(m("attention.value_length")) == hd and \
            int(m("attention.key_length_swa", hd)) == hd and int(m("rope.dimension_count", 0)) == 0, \
            f"{gguf.path}: attention shapes"
        vocab = gguf.string_count("tokenizer.ggml.tokens")
        assert vocab == int(m("vocab_size_unpadded", vocab)), f"{gguf.path}: vocabulary"
        tail = max(0, vocab - 512)
        eos = [int(gguf.meta["tokenizer.ggml.eos_token_id"])]
        eos += [i for i, t in enumerate(gguf.strings("tokenizer.ggml.tokens", tail), tail) if t in cls.EOG]
        return cls(
            vocab_size=vocab,
            vocab_padded=int(m("vocab_size", vocab)),
            hidden_size=int(m("embedding_length")),
            num_hidden_layers=n,
            num_attention_heads=heads.pop(),
            num_key_value_heads=kv.pop(),
            head_dim=hd,
            kinds=tuple("swa" if x else "full" for x in swa),
            sliding_window=int(m("attention.sliding_window")),
            rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
            max_position_embeddings=int(m("context_length")),
            eos_token_ids=tuple(dict.fromkeys(eos)),
            sampling={k: float(gguf.meta[f"general.sampling.{k}"]) for k in ("top_k", "top_p", "min_p", "temp")
                      if f"general.sampling.{k}" in gguf.meta},
            n_experts=int(m("expert_count")),
            n_experts_used=int(m("expert_used_count")),
            expert_ffn=int(m("expert_feed_forward_length")),
            expert_weights_scale=float(m("expert_weights_scale", 1.0)),
            situ_beta=float(m("activation.situ_beta")),
            situ_linear_beta=float(m("activation.situ_linear_beta")),
            block_size=block,
            conv_taps=int(m("shortconv.l_cache")),
            rel_length=int(m("attention.relative_length")),
            rel_extent=int(m("attention.relative_extent")),
            qk_clip=float(m("attention.qk_clip")),
            rel_clip=float(m("attention.relative_clip")),
            log_scale_floor=int(m("attention.log_scale_floor")),
            log_scale_alpha=float(m("attention.log_scale_alpha")),
            softcap=float(m("final_logit_softcapping", 0.0)),
            core_start=start,
            core_end=end,
            passes=passes,
            gate_rank=int(m("loop.gate_rank")),
            layout=layout,
        )

    @property
    def q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def layers(self, kind: str) -> list[int]:
        """The physical layers of a kind ("swa", "full"; "core": the loop's)."""
        if kind == "core":
            return list(range(self.core_start, self.core_end))
        return [i for i, k in enumerate(self.kinds) if k == kind]

    @property
    def visits(self) -> tuple[Visit, ...]:
        """A token's visits in order: the prelude, the core's passes, the coda."""
        n, core = self.num_hidden_layers, self.core_end - self.core_start
        path = [(layer, -1) for layer in range(self.core_start)]
        path += [(layer, p) for p in range(self.passes) for layer in range(self.core_start, self.core_end)]
        path += [(layer, -1) for layer in range(self.core_end, n)]
        return tuple(Visit(index=v, layer=layer, loop=p, kind=self.kinds[layer], block=layer % self.block_size == 0,
                           slot=layer if p <= 0 else n + (p - 1) * core + layer - self.core_start,
                           queries=0 if layer == 0 else 1 if self.kinds[layer] == "swa" else 3)
                     for v, (layer, p) in enumerate(path))

    def tau(self, positions: np.ndarray | range | list) -> np.ndarray:
        """(n,) float32: the full layers' factor of Q and R at positions, 1 + alpha ln(max(1, (p + 1) / floor)) in
        llama.cpp's fp32 operations."""
        p = np.asarray(positions, np.float32).reshape(-1) + np.float32(1)
        x = np.maximum(p * np.float32(1.0 / self.log_scale_floor), np.float32(1))
        return (np.log(x) * np.float32(self.log_scale_alpha) + np.float32(1)).astype(np.float32)

    def ctrl_tensor(self, layer: int, stream: str, part: str) -> str:
        """The name of a loop layer's controller weight for a stream of ``CTRL_STREAMS``: ``part`` "down" (2 hidden ->
        gate_rank), "gain" or "forget" (gate_rank -> hidden)."""
        if self.layout == "loop":
            return f"blk.{layer}.loop_{stream}_ctrl_{part}.weight"
        return f"blk.{layer}.loop_{self._ROOT_CTRL[stream]}_{part}.weight"

    def msg_queries(self, gguf: Any, layer: int) -> np.ndarray:
        """(4, hidden) fp32: a loop layer's message queries, one per stream of ``CTRL_STREAMS`` (the root layout's two
        tables of the core's queries: per layer i of the core rows 2 i, 2 i + 1 of the q and kv tables, as llama.cpp's
        converter stacks them: q[2 i], kv[2 i], kv[2 i + 1], q[2 i + 1])."""
        if self.layout == "loop":
            return gguf.tensor(f"blk.{layer}.loop_msg_query.weight").dequantize()
        i = layer - self.core_start
        q, kv = (gguf.tensor(f"{t}.weight").dequantize() for t in ("loop_res_query", "loop_kv_res_query"))
        return np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])


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
class DSparkConfig:
    """A DSpark block drafter of Spark-X (XHToken's ``<model>-DSpark-GGUF``: GGUF architecture ``dflash`` with a Markov
    head, ``markov_w1``; XHToken's llama.cpp fork's ``src/models/dflash.cpp``), from the metadata of its file:
    ``layers`` sliding-window layers of the model's shape (``hidden``, ``ffn`` with GeGLU, ``heads`` query heads with a
    sigmoid output gate each over ``kv_heads`` KV heads of ``head_dim``, no q / k norms; RoPE on ``rope_dims`` dims,
    base ``rope_base``), a block of ``block`` rows (the anchor, then ``mask_token``) that sees the ``window`` positions
    before its anchor and itself whole; ``taps``: the model's layers whose inputs feed its cache (fc of their
    concatenation, ``fc_norm`` per tap); the Markov head of ``rank`` dims (``markov_w1`` / ``markov_w2``, vocab x rank) and
    the confidence head (``conf_proj``: hidden + rank -> 1)."""

    layers: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    block: int
    window: int
    taps: tuple[int, ...]
    mask_token: int
    vocab_size: int
    rank: int
    rms_norm_eps: float
    rope_base: float
    rope_dims: int

    @property
    def hidden_size(self) -> int:
        return self.hidden

    @property
    def num_hidden_layers(self) -> int:
        return self.layers

    @property
    def intermediate_size(self) -> int:
        return self.ffn

    @classmethod
    def from_gguf(cls, gguf: Any) -> DSparkConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert gguf.arch == "dflash" and "markov_w1.weight" in gguf, f"{gguf.path}: not a DSpark drafter"
        m = gguf.arch_meta
        hd = int(m("attention.key_length"))
        w1 = gguf.tensor("markov_w1.weight")
        return cls(layers=int(m("block_count")), hidden=int(m("embedding_length")), ffn=int(m("feed_forward_length")),
                   heads=int(m("attention.head_count")), kv_heads=int(m("attention.head_count_kv")), head_dim=hd,
                   block=int(m("block_size")), window=int(m("attention.sliding_window")),
                   taps=tuple(int(x) for x in m("target_layers")),
                   mask_token=int(gguf.meta["tokenizer.ggml.mask_token_id"]), vocab_size=w1.rows, rank=w1.cols,
                   rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")),
                   rope_base=float(m("rope.freq_base_swa", 10000.0)), rope_dims=int(m("rope.dimension_count_swa", hd)))

    def rope_theta(self, positions: np.ndarray | range | list) -> np.ndarray:
        """(n, rope_dims / 2) float32 angles of positions (``SparkConfig.rope_theta``'s)."""
        scale = np.float32(np.power(np.float32(self.rope_base), np.float32(-2.0 / self.rope_dims)))
        th = np.asarray(positions, np.float32).reshape(-1)
        out = np.empty((th.shape[0], self.rope_dims // 2), np.float32)
        for i in range(self.rope_dims // 2):
            out[:, i] = th
            th = (th * scale).astype(np.float32)
        return out


@dataclass(frozen=True)
class InklingDSparkConfig:
    """Inkling 2's DSpark block drafter (unsloth's ``dspark-*.gguf``: GGUF architecture ``dflash`` with the recurrent
    Markov head, ``joint_proj``; vLLM's ``models/inkling2/nvidia/dspark.py``), from the metadata of its file: ``layers``
    dense layers of ``hidden`` (SiLU GLU of ``ffn``; ``heads`` query heads over ``kv_heads`` KV heads of ``head_dim``,
    q / k RMS-normed per head, NeoX RoPE of base ``rope_base`` on all dims), each a ``local`` one (its keys the
    ``window`` positions before a block's anchor) or a full one (all of them), a block of ``block`` rows (the anchor's
    normed embedding, then the mask rows) that sees its keys and itself whole; ``taps``: the model's visits whose
    residual feeds its cache (``Visit.index``; ``len(visits)``: the final normed state), mixed per layer and for the
    Markov head; the Markov head's ``rank``; its logits softcapped at ``softcap`` over the ``vocab_size`` tokens.

    The file has one of two forms of the same weights. Unsloth's: the taps' mixes as the softmax of ``dspark_fuse``'s
    rows, one ``mask_embd``, ``dspark_hidden_out`` (the Markov head's latent into the hidden, then the model's head),
    ``markov_w1`` with the head's padding rows. ``mainline`` (llama.cpp's, ``has_markov_context``): the mixes' weights
    themselves (``blk.L.ctx_mix``, ``markov_ctx_mix``), a mask row per block row, ``markov_w2`` (the model's head folded
    with ``dspark_hidden_out``: a row per token, and its bias), ``markov_w1`` of the tokens' rows, and its
    ``sliding_window`` counts the anchor too (llama.cpp's window)."""

    layers: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    block: int
    window: int
    local: tuple[bool, ...]
    taps: tuple[int, ...]
    vocab_size: int
    rank: int
    rms_norm_eps: float
    rope_base: float
    softcap: float
    mainline: bool

    @property
    def hidden_size(self) -> int:
        return self.hidden

    @classmethod
    def from_gguf(cls, gguf: Any) -> InklingDSparkConfig:
        """From an ``artlite.gguf_model.GGUFFile`` (either form)."""
        assert gguf.arch == "dflash" and "joint_proj.weight" in gguf, f"{gguf.path}: not an Inkling DSpark drafter"
        m = gguf.arch_meta
        n = int(m("block_count"))
        mainline = bool(m("has_markov_context", False))
        weights = ("blk.0.ctx_mix.weight", "markov_ctx_mix.weight", "markov_w2.weight") if mainline else \
            ("dspark_fuse.weight", "dspark_hidden_out.weight")
        assert not m("sample_from_anchor", False) and not m("has_confidence_head", False) and \
            not m("attention.causal", False) and all(w in gguf for w in weights), f"{gguf.path}: another DSpark form"
        w1 = gguf.tensor("markov_w1.weight")  # (rows: tokens, padded in unsloth's form; columns: the rank)
        return cls(layers=n, hidden=int(m("embedding_length")), ffn=int(m("feed_forward_length")),
                   heads=int(m("attention.head_count")), kv_heads=int(m("attention.head_count_kv")),
                   head_dim=int(m("attention.key_length")), block=int(m("block_size")),
                   window=int(m("attention.sliding_window")) - mainline,
                   local=tuple(bool(x) for x in m("attention.sliding_window_pattern")),
                   taps=tuple(int(x) for x in m("target_layers")),
                   vocab_size=w1.rows if mainline else int(m("vocab_size_unpadded")), rank=w1.cols,
                   rms_norm_eps=float(m("attention.layer_norm_rms_epsilon")), rope_base=float(m("rope.freq_base")),
                   softcap=float(m("final_logit_softcapping", 0.0)), mainline=mainline)

    def rope_theta(self, positions: np.ndarray | range | list) -> np.ndarray:
        """(n, head_dim / 2) float32 angles of positions (``DFlashConfig.rope_theta``'s)."""
        scale = np.float32(np.power(np.float32(self.rope_base), np.float32(-2.0 / self.head_dim)))
        th = np.asarray(positions, np.float32).reshape(-1)
        out = np.empty((th.shape[0], self.head_dim // 2), np.float32)
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
    separate projectors (``is_deepstack_layers``); ``kv_heads`` (grouped-query attention: LFM3-VL's), an image's
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


@dataclass(frozen=True)
class AudioConfig:
    """A speech encoder and projector, from the ``clip.audio.*`` metadata of a GGUF file (architecture ``clip``; llama.cpp's
    ``tools/mtmd/clip.cpp``): ``projector`` (``clip.audio.projector_type``: "lfm3a"), the encoder (``layers`` of
    ``hidden`` with ``heads`` and ``kv_heads``, ``ffn``, RMSNorm ``eps``, attention over the last ``window`` positions),
    its input (``mel_bins`` log-mel bins) and the language model's width ``projection_dim``."""

    projector: str
    hidden: int
    ffn: int
    layers: int
    heads: int
    kv_heads: int
    eps: float
    mel_bins: int
    window: int
    projection_dim: int

    @classmethod
    def has_audio(cls, gguf: Any) -> bool:
        return gguf.arch == "clip" and bool(gguf.meta.get("clip.has_audio_encoder", False))

    @classmethod
    def from_gguf(cls, gguf: Any) -> AudioConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert cls.has_audio(gguf), f"{gguf.path}: no speech encoder (clip.has_audio_encoder)"
        m = lambda k, d=None: gguf.meta.get(f"clip.audio.{k}", d)
        return cls(
            projector=str(m("projector_type")),
            hidden=int(m("embedding_length")),
            ffn=int(m("feed_forward_length")),
            layers=int(m("block_count")),
            heads=int(m("attention.head_count")),
            kv_heads=int(m("attention.head_count_kv", m("attention.head_count"))),
            eps=float(m("attention.layer_norm_epsilon")),
            mel_bins=int(m("num_mel_bins")),
            window=int(m("window_size", 0)),
            projection_dim=int(m("projection_dim")),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


@dataclass(frozen=True)
class GenConfig:
    """A speech generator, from the ``clip.gen.audio.*`` metadata of a GGUF file (architecture ``clip``; llama.cpp's
    ``tools/mtmd/clip.cpp``): ``projector`` (``clip.gen.audio.projector_type``: "lfm3a_gen", LFM3-Audio's depthformer
    and codec decoder), the depthformer (``layers`` of ``hidden`` with ``heads`` and ``kv_heads``, ``ffn``, RMSNorm
    ``eps``), the language model's width ``projection_dim`` and the speech codes' sampling (``temp``, ``top_k``, the
    repetition penalty ``penalty_repeat`` over the last ``penalty_last_n`` frames: the converter writes the model card's;
    llama.cpp's defaults are LFM3-Omni's card)."""

    projector: str
    projection_dim: int
    hidden: int
    ffn: int
    layers: int
    heads: int
    kv_heads: int
    eps: float
    temp: float
    top_k: int
    penalty_repeat: float
    penalty_last_n: int

    @classmethod
    def has_gen(cls, gguf: Any) -> bool:
        return gguf.arch == "clip" and bool(gguf.meta.get("clip.has_gen_audio_encoder", False))

    @classmethod
    def from_gguf(cls, gguf: Any) -> GenConfig:
        """From an ``artlite.gguf_model.GGUFFile``."""
        assert cls.has_gen(gguf), f"{gguf.path}: no speech generator (clip.has_gen_audio_encoder)"
        m = lambda k, d=None: gguf.meta.get(f"clip.gen.audio.{k}", d)
        return cls(
            projector=str(m("projector_type")),
            projection_dim=int(m("projection_dim")),
            hidden=int(m("embedding_length")),
            ffn=int(m("feed_forward_length")),
            layers=int(m("block_count")),
            heads=int(m("attention.head_count")),
            kv_heads=int(m("attention.head_count_kv", m("attention.head_count"))),
            eps=float(m("attention.layer_norm_epsilon")),
            temp=float(m("sampling.temp", 0.4)),
            top_k=int(m("sampling.top_k", 0)),
            penalty_repeat=float(m("sampling.penalty_repeat", 1.2)),
            penalty_last_n=int(m("sampling.penalty_last_n", 64)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads
