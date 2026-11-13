"""The model files the tests run: a size's file, and the tests' skip when it is missing."""

import os

import pytest

from artlite.model_config import decision_gguf, llama_gguf, occamy_gguf, qwen3_gguf, qwen_gguf, swift_gguf


def size_gguf(size: str) -> str:
    """The file a test runs for a size: lmstudio's Q4_K_M of it (``qwen_gguf``), "occamy": Occamy 1.0 APEX
    (``occamy_gguf``: Qwen3.6-35B-A3B's shapes in Q3_K, IQ3_XXS, Q4_K, Q5_K, Q6_K, Q8_0 and F32 tensors), "kev": Kev-4B
    (``decision_gguf``: a hidden size of 2560, part of a K chunk; Q8_0 and Q4_K tensors), or "d3-lite" (``decision_gguf``:
    MiniCPM-V's shapes, every tensor Q8_0: gate / up tiles of 8 rows), "swift": Swift 1.5 (``swift_gguf``: Qwen3.8-27B's
    shapes in IQ4_XS, IQ3_S, IQ3_XXS, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 and F32 tensors)."""
    if size in ("kev", "d3-lite"):
        return decision_gguf(size)
    if size == "swift":
        return swift_gguf()
    return occamy_gguf() if size == "occamy" else qwen_gguf("Q4_K_M", size)


def require(*paths: str) -> None:
    """Skip the test unless every file exists."""
    for path in paths:
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")


def llama_path(quant: str, size: str) -> str:
    """The file of a Llama test case (``size`` "qwen3:<size>": a Qwen3 file, any other the Llama ``quant`` of it), or skip
    the test without it."""
    path = qwen3_gguf(size.removeprefix("qwen3:")) if size.startswith("qwen3:") else llama_gguf(quant, size)
    require(path)
    return path
