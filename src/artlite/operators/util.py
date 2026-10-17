"""Small host-side helpers the operators share: attribute dicts for the kernel templates, 64-byte-aligned regions
(workspaces, buffer layouts), bf16 roundings of constants, an operator argument's bytes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch

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
    return float(torch.tensor(x).bfloat16())


def bf16_bits(x: float) -> int:
    """The bit pattern of x rounded to bf16 (round to nearest even)."""
    return int(torch.tensor([x]).bfloat16().view(torch.int16)[0]) & 0xFFFF


def bf16_hilo(x: float) -> tuple[float, float]:
    """The bf16 hi and lo parts of an fp32 constant (hi + lo: x to ~16 bits)."""
    hi = bf16_value(x)
    return hi, bf16_value(x - hi)


def arg_bytes(op: ArtcOperator, i: int) -> int:
    """Bytes of an operator's argument i (``get_arg_spec``)."""
    spec = op.get_arg_spec()[i]
    return int(np.prod(spec.shape)) * torch.tensor([], dtype=spec.dtype).element_size()
