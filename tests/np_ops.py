"""float32 numpy operations of the CPU references (llama.cpp's CPU ops: tokens first, the last axis the feature axis)."""

import numpy as np


def f32(x: object) -> np.ndarray:
    return np.asarray(x, np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = f32(x)
    return x / (np.float32(1) + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = f32(x)
    return np.float32(1) / (np.float32(1) + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    """llama.cpp's: x above 20, else log(1 + e^x)."""
    x = f32(x)
    with np.errstate(over="ignore"):
        return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(np.float32)


def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    """x / sqrt(mean(x^2) + eps) * w over the last axis (every qwen35 norm weight already holds its 1 +)."""
    x = f32(x)
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    y = (x / np.sqrt(ms + eps)).astype(np.float32)
    return y if w is None else y * f32(w)


def softmax(x: np.ndarray) -> np.ndarray:
    """Over the last axis, as ``ggml_soft_max``: the maximum subtracted, the sum in float64."""
    x = f32(x)
    e = np.exp(x - x.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True, dtype=np.float64)).astype(np.float32)
