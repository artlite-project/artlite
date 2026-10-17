"""What the host computes for a UNet step: the timestep's embedding and, from it, each resnet's time bias.

The UNet conditions its 22 resnets on the timestep through one vector per resnet: temb = the ``time_embedding`` MLP
(linear 320 -> 1280, SiLU, linear 1280 -> 1280) of the sinusoidal embedding of t, then per resnet
``time_emb_proj``(SiLU(temb)), added to every pixel of the resnet's first convolution's output. Both halves of the
CFG batch share t, so a step needs one vector per resnet (out channels,), whatever the batch.
"""

import math
from concurrent.futures import Executor

import torch
import torch.nn.functional as F

TIME_CHANNELS = 320  # the sinusoidal embedding's width (block_out_channels[0])


def timestep_embedding(t: float, dim: int = TIME_CHANNELS) -> torch.Tensor:
    """The sinusoidal embedding (dim,) fp32 of a (fractional) timestep: cos(t f_i), then sin(t f_i), f_i =
    10000^(-i / (dim / 2)) (diffusers' flip_sin_to_cos, freq_shift 0)."""
    half = dim // 2
    e = float(t) * torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    return torch.cat([torch.cos(e), torch.sin(e)])


def time_embedding(weights: dict[str, torch.Tensor], t: float) -> torch.Tensor:
    """temb (1280,) of a timestep: the UNet's ``time_embedding`` MLP on its sinusoidal embedding."""
    e = F.linear(timestep_embedding(t)[None], weights["time_embedding.linear_1.weight"],
                 weights["time_embedding.linear_1.bias"])
    return F.linear(F.silu(e), weights["time_embedding.linear_2.weight"], weights["time_embedding.linear_2.bias"])[0]


def resnet_time_biases(weights: dict[str, torch.Tensor], t: float, *, pool: Executor | None = None) -> dict[str, torch.Tensor]:
    """Each UNet resnet's time bias (out channels,) fp32 at a timestep, by the resnet's weight prefix
    (``"down_blocks.0.resnets.0."``, ..., ``"mid_block.resnets.1."``, ..., ``"up_blocks.3.resnets.2."``). pool: the
    projections on its threads (memory-bound GEMVs over 103 MB of weights: ~5 ms on one thread, ~1 ms on 8)."""
    s = F.silu(time_embedding(weights, t))[None]
    suffix = "time_emb_proj.weight"
    names = [name for name in weights if name.endswith(suffix)]

    def proj(name: str) -> torch.Tensor:
        return F.linear(s, weights[name], weights[name.removesuffix("weight") + "bias"])[0]
    return dict(zip((n.removesuffix(suffix) for n in names), (pool.map if pool else map)(proj, names), strict=True))
