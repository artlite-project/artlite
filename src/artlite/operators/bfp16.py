"""bfp16ebs8 on the host: conversions and the activation panel layout.

A bfp16ebs8 block is 8 values with one shared exponent byte E and 8 int8
mantissas m: value = m * 2^(E - 133), stored as 9 bytes [E, m0..m7] (the AIE2P
block convention; ``sizeof(bfp16ebs8) == 9`` on the device too).

``blocks`` rounds to nearest even like the kernels' ``to_vector<bfp16ebs8>()``
after ``set_rounding(conv_even)``.

Activation panels (GEMM A / C between prefill ops): ``[rows / 8][cols / 8][8 x 9 B]``,
each panel = 8 rows x 8 columns, one block per row.
"""

import numpy as np
import torch


def blocks(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float (..., 8) -> (exponent bytes (...,), int8 mantissas (..., 8)), round to nearest even.
    As the hardware: a positive maximum that rounds up to 128 raises the exponent, a negative
    one stays at -128; an all-zero block has exponent byte 0."""
    x = np.asarray(w, np.float32)
    amax = np.abs(x).max(-1)
    e = np.floor(np.log2(np.maximum(amax, 2.0 ** -120))).astype(np.int32)
    m = np.rint(x / np.ldexp(np.float32(1.0), e - 6)[..., None])
    e = np.where(m.max(-1) > 127, e + 1, e)  # a positive maximum rounded up to 128
    m = np.rint(x / np.ldexp(np.float32(1.0), e - 6)[..., None])
    e = np.where(amax == 0, -127, e)  # an all-zero block has exponent byte 0
    return np.clip(e + 127, 0, 255).astype(np.uint8), m.astype(np.int8)


def block_bytes(x: np.ndarray) -> np.ndarray:
    """float (..., 8) -> bfp16ebs8 blocks (..., 9) uint8 [E, m0 .. m7] (``blocks``)."""
    e, m = blocks(x)
    return np.concatenate([e[..., None], m.view(np.uint8)], axis=-1)


def block_values(b: np.ndarray) -> np.ndarray:
    """bfp16ebs8 blocks (..., 9) uint8 -> their values (..., 8) float32."""
    return b[..., 1:].view(np.int8).astype(np.float32) * np.ldexp(np.float32(1), b[..., 0].astype(np.int32) - 133)[..., None]


def values(w: np.ndarray) -> np.ndarray:
    """float (..., 8k) rounded to bfp16ebs8 blocks along the last axis (float32 values)."""
    e, m = blocks(np.asarray(w, np.float32).reshape(*w.shape[:-1], -1, 8))
    return (m * np.ldexp(np.float32(1.0), e.astype(np.int32) - 133)[..., None]).reshape(w.shape)


def to_panels(x: torch.Tensor) -> torch.Tensor:
    """(rows, cols) -> activation panels [rows / 8][cols / 8][8 x 9 B] (round to nearest even)
    as a bf16-typed byte buffer."""
    r, c = x.shape
    b = block_bytes(x.float().numpy().reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3))
    return torch.from_numpy(np.ascontiguousarray(b).reshape(-1)).view(torch.bfloat16)


def from_panels(b: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Activation panels -> (rows, cols) float32."""
    val = block_values(b.view(torch.uint8).numpy().reshape(rows // 8, cols // 8, 8, 9))
    return torch.from_numpy(np.ascontiguousarray(val.transpose(0, 2, 1, 3)).reshape(rows, cols))
