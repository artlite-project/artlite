"""What the host computes for a UNet step: the timestep's embedding and, from it, each resnet's time bias.

The UNet conditions its 22 resnets on the timestep through one vector per resnet: temb = the ``time_embedding`` MLP
(linear 320 -> 1280, SiLU, linear 1280 -> 1280) of the sinusoidal embedding of t, then per resnet
``time_emb_proj``(SiLU(temb)), added to every pixel of the resnet's first convolution's output. Both halves of the
CFG batch share t, so a step needs one vector per resnet (out channels,), whatever the batch.
"""

import math
from concurrent.futures import Executor

import numpy as np

TIME_CHANNELS = 320  # the sinusoidal embedding's width (block_out_channels[0])


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def timestep_embedding(t: float, dim: int = TIME_CHANNELS) -> np.ndarray:
    """The sinusoidal embedding (dim,) fp32 of a (fractional) timestep: cos(t f_i), then sin(t f_i), f_i =
    10000^(-i / (dim / 2)) (diffusers' flip_sin_to_cos, freq_shift 0)."""
    half = dim // 2
    e = float(t) * np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
    return np.concatenate([np.cos(e), np.sin(e)])


def time_embedding(weights: dict[str, np.ndarray], t: float) -> np.ndarray:
    """temb (1280,) of a timestep: the UNet's ``time_embedding`` MLP on its sinusoidal embedding."""
    e = weights["time_embedding.linear_1.weight"] @ timestep_embedding(t) + weights["time_embedding.linear_1.bias"]
    return weights["time_embedding.linear_2.weight"] @ silu(e) + weights["time_embedding.linear_2.bias"]


def resnet_time_biases(weights: dict[str, np.ndarray], t: float, *, pool: Executor | None = None) -> dict[str, np.ndarray]:
    """Each UNet resnet's time bias (out channels,) fp32 at a timestep, by the resnet's weight prefix
    (``"down_blocks.0.resnets.0."``, ..., ``"mid_block.resnets.1."``, ..., ``"up_blocks.3.resnets.2."``). pool: the
    projections on its threads (memory-bound GEMVs over 103 MB of weights, the GIL released: ~5 ms on one thread)."""
    s = silu(time_embedding(weights, t))
    suffix = "time_emb_proj.weight"
    names = [name for name in weights if name.endswith(suffix)]

    def proj(name: str) -> np.ndarray:
        return weights[name] @ s + weights[name.removesuffix("weight") + "bias"]
    return dict(zip((n.removesuffix(suffix) for n in names), (pool.map if pool else map)(proj, names), strict=True))
