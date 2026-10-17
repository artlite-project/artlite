"""Operators of the qwen35 models (Qwen3.5 / 3.6 / 3.8: Gated DeltaNet + full-attention hybrid, GGUF arch
``qwen35``): the weight image (``image``), the decode overlay and the prefill blocks."""

from .image import QwenImage

__all__ = ["QwenImage"]
