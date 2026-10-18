"""Artlite's models: a package per GGUF architecture (the model on the NPU and its host side: tokenizer, chat
template, the inputs the host computes) or diffusers pipeline, what they share (``common``) and the command lines
(``cli``: ``artlite-cli``; ``bench``: ``artlite-bench``; ``server``: ``artlite-server``, an OpenAI-compatible server
with a web chat).

``open_model`` reads a GGUF file's architecture from its metadata (``general.architecture``) and opens it with the
``Model`` of its package; ``open_pipeline`` opens a diffusers directory by the pipeline class of its
``model_index.json``:

| architecture / pipeline | package | models |
|---|---|---|
| ``llama`` | ``llama3`` | Llama 3.2 1B / 3B, Llama 3.1 8B |
| ``gemma4`` | ``gemma4`` | Gemma 4 E4B |
| ``qwen35`` | ``qwen35`` | Qwen3.5 / Qwen3.6 (Gated DeltaNet and full-attention layers) |
| ``StableDiffusionPipeline`` | ``sd15`` | Stable Diffusion 1.5, text to image (the UNet and the VAE decoder on the NPU) |
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from typing import TYPE_CHECKING

from artlite.gguf_model import GGUFFile

from .common import Model

if TYPE_CHECKING:
    from .sd15.pipeline import StableDiffusion

# architecture -> (module, Model subclass), imported on use
ARCHITECTURES = {
    "llama": ("models.llama3.llama", "Llama"),
    "gemma4": ("models.gemma4.gemma", "Gemma"),
    "qwen35": ("models.qwen35.qwen", "Qwen"),
}
# diffusers pipeline class (``_class_name`` of model_index.json) -> (module, class), imported on use
PIPELINES = {
    "StableDiffusionPipeline": ("models.sd15.pipeline", "StableDiffusion"),
}


def architecture(path: str) -> str:
    """The architecture of a GGUF file (``general.architecture``: ``llama``, ``gemma4``, ...)."""
    if is_diffusers(path):
        raise ValueError(f"{path}: a diffusers pipeline, not a GGUF file")
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


def open_model(path: str, **options: object) -> Model:
    """Open a GGUF file with the ``Model`` of its architecture; ``options``: that class's keyword arguments."""
    arch = architecture(path)
    try:
        cls = model_class(arch)
    except NotImplementedError as e:
        raise NotImplementedError(f"{path}: {e}") from None
    unknown = set(options) - set(inspect.signature(cls).parameters)
    if unknown:
        raise ValueError(f"{cls.name} has no option {', '.join(sorted(unknown))}")
    return cls(path, **options)


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
