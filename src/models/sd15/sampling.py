"""The noise schedule of Stable Diffusion 1.x and the samplers that walk it, in k-diffusion's formulation, as
stable-diffusion.cpp runs them.

The model was trained on the discrete DDPM schedule: betas "scaled_linear" from 0.00085 to 0.012 over 1000
timesteps (``scheduler/scheduler_config.json``), alpha_bar_t = prod(1 - beta). In k-diffusion's form the latents at
timestep t are x = x0 + sigma_t eps, sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t) (0.0292 at t = 0, 14.61 at
t = 999); a fractional t interpolates log sigma between its neighbors. The UNet predicts eps from x / sqrt(sigma^2 + 1)
(the DDPM-scaled latents) and t; the denoised estimate is x - sigma eps, with classifier-free guidance eps =
eps_uncond + scale (eps_cond - eps_uncond) from one UNet batch [uncond, cond] (``cfg_denoiser``).

Sampling starts from Gaussian noise times the first sigma and steps down the sigmas to 0:
- ``euler``: Euler steps of the probability-flow ODE dx/dsigma = (x - denoised) / sigma (deterministic);
- ``euler_a``: Euler ancestral: an Euler step down to sigma_down, then fresh noise of sigma_up (sigma_down^2 +
  sigma_up^2 = the next sigma^2);
- ``ddim``: DDIM with eta 0, which is the Euler step, on its "trailing" timesteps.
``euler`` and ``euler_a`` take sd.cpp's "discrete" sigmas (n timesteps evenly from 999 to 0), ``ddim`` its "simple"
ones (timesteps 999 - floor(i * 1000 / n)); a last step to sigma 0 returns the denoised estimate.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable

import torch

SAMPLERS = ("euler_a", "euler", "ddim")

# denoise(x, sigma): the denoised estimate of latents x at noise level sigma
Denoise = Callable[[torch.Tensor, float], torch.Tensor]


class Schedule:
    """The discrete DDPM schedule (scaled_linear betas) as sigmas: ``sigma(t)``, ``timestep(sigma)`` and the
    sigmas of a sampler's steps."""

    def __init__(self, *, beta_start: float = 0.00085, beta_end: float = 0.012, timesteps: int = 1000):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=torch.float64) ** 2
        alpha_bar = torch.cumprod(1 - betas, 0)
        self.log_sigmas = ((1 - alpha_bar) / alpha_bar).sqrt().log()

    @classmethod
    def from_config(cls, path: str) -> Schedule:
        """From a diffusers ``scheduler_config.json``."""
        with open(path) as f:
            c = json.load(f)
        if c.get("beta_schedule") != "scaled_linear" or c.get("prediction_type", "epsilon") != "epsilon":
            raise NotImplementedError(f"{path}: only scaled_linear betas and epsilon prediction are supported")
        return cls(beta_start=c["beta_start"], beta_end=c["beta_end"], timesteps=c["num_train_timesteps"])

    @property
    def timesteps(self) -> int:
        return len(self.log_sigmas)

    def sigma(self, t: float) -> float:
        """The sigma of a (fractional) timestep."""
        lo = math.floor(t)
        hi = min(lo + 1, self.timesteps - 1)
        w = t - lo
        return float(((1 - w) * self.log_sigmas[lo] + w * self.log_sigmas[hi]).exp())

    def timestep(self, sigma: float) -> float:
        """The fractional timestep of a sigma (log sigma interpolated between the two timesteps around it)."""
        ls = math.log(sigma)
        lo = int(min(max(int((ls - self.log_sigmas >= 0).sum()) - 1, 0), self.timesteps - 2))
        l0, l1 = float(self.log_sigmas[lo]), float(self.log_sigmas[lo + 1])
        w = min(max((l0 - ls) / (l0 - l1), 0.0), 1.0)
        return (1 - w) * lo + w * (lo + 1)

    def sigmas(self, steps: int, sampler: str) -> list[float]:
        """The ``steps`` sigmas a sampler starts its steps at, then 0."""
        last = self.timesteps - 1
        if sampler == "ddim":
            ts = [max(last - int(i * self.timesteps / steps), 0) for i in range(steps)]
        else:
            ts = [last - i * (last / (steps - 1)) for i in range(steps - 1)] + [0] if steps > 1 else [last]
        return [self.sigma(t) for t in ts] + [0.0]


def sample(denoise: Denoise, x: torch.Tensor, sigmas: list[float], *, sampler: str,
           generator: torch.Generator | None = None, callback: Callable[[int], None] | None = None) -> torch.Tensor:
    """Run a sampler from latents ``x`` at ``sigmas[0]`` down the sigmas; ``generator``: euler_a's noise;
    ``callback(i)`` after step i."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r} (samplers: {', '.join(SAMPLERS)})")
    for i, (s, sn) in enumerate(itertools.pairwise(sigmas)):
        d = (x - denoise(x, s)) / s
        if sampler == "euler_a" and sn > 0:
            up = min(sn, math.sqrt(sn ** 2 * (s ** 2 - sn ** 2) / s ** 2))
            down = math.sqrt(sn ** 2 - up ** 2)
            x = x + d * (down - s) + torch.randn(x.shape, generator=generator) * up
        else:
            x = x + d * (sn - s)
        if callback is not None:
            callback(i)
    return x


def cfg_denoiser(unet: Callable[[torch.Tensor, float, torch.Tensor], torch.Tensor], schedule: Schedule,
                 context: torch.Tensor, scale: float) -> Denoise:
    """``denoise`` of an eps-predicting UNet with classifier-free guidance: ``unet(x, t, context)`` runs the batch
    [uncond, cond] (``context`` (2, 77, 768) in that order)."""

    def denoise(x: torch.Tensor, sigma: float) -> torch.Tensor:
        eps = unet(torch.cat([x, x]) * (1.0 / math.sqrt(sigma ** 2 + 1)), schedule.timestep(sigma), context)
        uncond, cond = eps.chunk(2)
        return x - sigma * (uncond + scale * (cond - uncond))

    return denoise
