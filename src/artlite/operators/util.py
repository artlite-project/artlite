"""Small host-side helpers the operators share: attribute dicts for the kernel templates, 64-byte-aligned regions
(workspaces, buffer layouts), bf16 roundings of constants and arrays, the 8 x 8 bit masks of the MMUL panels, the RoPE
rows, an operator argument's bytes and elements."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
from ml_dtypes import bfloat16

if TYPE_CHECKING:
    from .base import ArtcOperator


class Attr(dict):
    """dict with attribute access (Jinja: ``w_.st``)."""

    __getattr__ = dict.__getitem__


def rup(n: int, a: int = 64) -> int:
    """n rounded up to a multiple of a."""
    return -(-n // a) * a


def regions(sizes: Iterable[tuple[str, int]], align: int = 64) -> dict[str, int]:
    """Byte offsets of consecutive regions ((name, bytes) in order, each from a multiple of ``align`` on) and their end
    ("size"): a core's workspace."""
    out, off = {}, 0
    for name, n in sizes:
        out[name] = off
        off += rup(n, align)
    out["size"] = off
    return out


def bf16_value(x: float) -> float:
    """x rounded to bf16 (round to nearest even)."""
    return float(np.float32(x).astype(bfloat16))


def bf16_bits(x: float) -> int:
    """The bit pattern of x rounded to bf16 (round to nearest even)."""
    return int(np.float32(x).astype(bfloat16).view(np.uint16))


def bf16_hilo(x: float) -> tuple[float, float]:
    """The bf16 hi and lo parts of an fp32 constant (hi + lo: x to ~16 bits)."""
    hi = bf16_value(x)
    return hi, bf16_value(x - hi)


def bf16_bits_array(x: np.ndarray) -> np.ndarray:
    """The bit patterns (uint16) of the values of x (fp32) rounded to bf16 (round to nearest even)."""
    return np.ascontiguousarray(x, np.float32).astype(bfloat16).view(np.uint16)


def bf16_hilo_bits(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The bf16 bit patterns of the hi and lo parts (x - hi) of the values of x (fp32)."""
    x = np.asarray(x, np.float32)
    hi = bf16_bits_array(x)
    return hi, bf16_bits_array(x - (hi.astype(np.uint32) << 16).view(np.float32))


def mask64(keep: Callable[[int, int], bool]) -> int:
    """The 64-bit mask of an 8 x 8 panel's (row, column) pairs that ``keep`` accepts: bit 8 row + column."""
    return sum(1 << (8 * r + c) for r in range(8) for c in range(8) if keep(r, c))


def cos_sin_rows(theta: np.ndarray) -> np.ndarray:
    """(n, 2 k) fp32 [cos | sin] per row of the angles theta (n, k), taken in float64."""
    th = np.asarray(theta, np.float64)
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


def arg_bytes(op: ArtcOperator, i: int) -> int:
    """Bytes of an operator's argument i (``get_arg_spec``)."""
    return op.get_arg_spec()[i].nbytes


def arg_elems(op: ArtcOperator, i: int) -> int:
    """Elements of an operator's argument i (``get_arg_spec``): the sizes its design's types and taps use."""
    spec = op.get_arg_spec()[i]
    return spec.nbytes // spec.dtype.itemsize
