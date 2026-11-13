"""The model files the tests run: a size's file, and the tests' skip when it is missing."""

from artlite.model_config import decision_gguf, occamy_gguf, qwen_gguf, swift_gguf


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
