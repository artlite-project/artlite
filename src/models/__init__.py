"""Artlite's models: a package per GGUF architecture (the model on the NPU and its host side: tokenizer, chat
template, the inputs the host computes) or diffusers pipeline, what they share (``model``, ``npu_model``,
``overlay_model``, ``speculative``, ``sampling``, ``templating``, ...) and the command lines (``cli``: ``artlite-cli``;
``bench``: ``artlite-bench``; ``server``: ``artlite-server``, an OpenAI-compatible server with a web chat).

``open_model`` reads a GGUF file's architecture from its metadata (``general.architecture``; a Hugging Face checkpoint
directory: the GGUF architecture it opens as, ``artlite.hf_model``) and opens it with the ``Model`` of its package;
``open_pipeline`` opens a diffusers directory by the pipeline class of its ``model_index.json``:

| architecture / pipeline | package | models |
|---|---|---|
| ``llama`` | ``llama`` | Llama 3.2 1B / 3B, Llama 3.1 8B |
| ``qwen3``, ``vllm-decision2`` | ``llama`` | Qwen3 0.6B, Qwen3-Embedding, Qwen3-Reranker; Decision-2.0-Kai (Qwen3 and a decision head) |
| ``gemma4`` | ``gemma4`` | Gemma 4 E4B |
| ``qwen35`` | ``qwen35`` | Qwen3.5 / Qwen3.6 / Qwen3.8 (Gated DeltaNet and full-attention layers; images: their Qwen3-VL mmproj), MiniCPM-V 4.7 (images: its mmproj) |
| ``lfm3`` | ``lfm`` | LFM3-1.7B (short-convolution, sliding-window and full-attention layers), LFM3-VL-2B (images: its mmproj), d1-2B |
| ``qwen35moe`` | ``qwen35`` | Qwen3.6-35B-A3B (the same layers, a mixture of experts as the FFN; images: its mmproj) |
| ``spark2_5`` | ``spark`` | Spark-X2.5 / X2.6 4B, 1.7B (sliding-window and full attention, head_dim 256) |
| ``flint`` | ``flint`` | Recursive's Flint (40B: 256 experts of which 6 per token; sliding-window and full attention with sinks) |
| ``inkling2_loop``, ``inkling2`` | ``inkling2`` | Thinking Machines' Inkling 2 Mini (26B: 128 experts of which 8 per token, 12 of 24 layers looped 3 times; decode on the NPU, prompts in prefill chunks, its DSpark drafter) |
| ``StableDiffusionPipeline`` | ``sd15`` | Stable Diffusion 1.5, text to image (the UNet and the VAE decoder on the NPU) |
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from typing import TYPE_CHECKING

from artlite.gguf_model import GGUFFile
from artlite.hf_model import architecture as hf_architecture
from artlite.hf_model import is_checkpoint

from .model import Model

if TYPE_CHECKING:
    from .sd15.pipeline import StableDiffusion

# architecture -> (module, Model subclass), imported on use
ARCHITECTURES = {
    "llama": ("models.llama.llama", "Llama"),
    "qwen3": ("models.llama.qwen3", "Qwen3"),
    "vllm-decision2": ("models.llama.qwen3", "Qwen3"),
    "gemma4": ("models.gemma4.gemma", "Gemma"),
    "qwen35": ("models.qwen35.qwen", "Qwen"),
    "qwen35moe": ("models.qwen35.qwen", "Qwen"),
    "lfm3": ("models.lfm.lfm", "Lfm"),
    "spark2_5": ("models.spark.spark", "Spark"),
    "flint": ("models.flint.flint", "Flint"),
    "inkling2_loop": ("models.inkling2.inkling", "Inkling"),
    "inkling2": ("models.inkling2.inkling", "Inkling"),
}
# diffusers pipeline class (``_class_name`` of model_index.json) -> (module, class), imported on use
PIPELINES = {
    "StableDiffusionPipeline": ("models.sd15.pipeline", "StableDiffusion"),
}


def architecture(path: str) -> str:
    """The architecture of a GGUF file (``general.architecture``: ``llama``, ``gemma4``, ...) or of a Hugging Face
    checkpoint directory (the GGUF architecture it opens as: ``artlite.hf_model``)."""
    if is_diffusers(path):
        raise ValueError(f"{path}: a diffusers pipeline, not a GGUF file")
    if is_checkpoint(path):
        return hf_architecture(path)
    if os.path.isdir(path):
        raise ValueError(f"{path}: a directory without config.json and safetensors (not a checkpoint)")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: no such file")
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
    return str(GGUFFile(path).arch)


def model_class(arch: str) -> type[Model]:
    """The ``Model`` subclass of an architecture."""
    if arch not in ARCHITECTURES:
        raise NotImplementedError(f"architecture {arch!r} is not supported (supported: {', '.join(ARCHITECTURES)})")
    module, name = ARCHITECTURES[arch]
    return getattr(importlib.import_module(module), name)


def open_model(path: str, *, chat_template: str | None = None, **options: object) -> Model:
    """Open a GGUF file or a Hugging Face checkpoint directory with the ``Model`` of its architecture; ``chat_template``: a Jinja chat template in place of the
    file's (``Model.chat_template``); ``options``: that class's keyword arguments (one given as False or "none": where
    the class has it, a value that turns a feature off applies to every model, ``spec_type="none"`` to one without a
    drafter)."""
    arch = architecture(path)
    try:
        cls = model_class(arch)
    except NotImplementedError as e:
        raise NotImplementedError(f"{path}: {e}") from None
    params = inspect.signature(cls).parameters
    unknown = {k for k, v in options.items() if k not in params and v is not False and v != "none"}
    if unknown:
        raise ValueError(f"{cls.name} has no option {', '.join(sorted(unknown))}")
    model = cls(path, **{k: v for k, v in options.items() if k in params})
    model.chat_template = chat_template
    return model


def is_diffusers(path: str) -> bool:
    """A diffusers model directory (``model_index.json``)."""
    return os.path.isfile(os.path.join(path, "model_index.json"))


def open_pipeline(path: str, **options: object) -> StableDiffusion:
    """Open a diffusers directory with the class of its pipeline; ``options``: that class's keyword arguments."""
    with open(os.path.join(path, "model_index.json")) as f:
        name = json.load(f).get("_class_name")
    if name not in PIPELINES:
        raise NotImplementedError(f"{path}: pipeline {name!r} is not supported (supported: {', '.join(PIPELINES)})")
    module, cls = PIPELINES[name]
    return getattr(importlib.import_module(module), cls)(path, **options)
