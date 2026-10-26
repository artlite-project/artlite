"""Stable Diffusion's noise schedule, samplers and classifier-free guidance (models.sd15.sampling) on toy denoisers
(no NPU, no weights)."""

import itertools
import math
import os

import numpy as np
import pytest
import torch

from artlite.model_config import sd15_dir
from models.sd15.sampling import SAMPLERS, Schedule, cfg_denoiser, sample

SIGMA_MIN, SIGMA_MAX = 0.02916716, 14.614641  # SD 1.x's sigmas at timesteps 0 and 999


def test_schedule() -> None:
    s = Schedule()
    assert s.sigma(0) == pytest.approx(SIGMA_MIN, rel=1e-6) and s.sigma(999) == pytest.approx(SIGMA_MAX, rel=1e-6)
    sig = [s.sigma(t) for t in range(1000)]
    assert all(a < b for a, b in itertools.pairwise(sig))
    assert s.sigma(500.25) == pytest.approx(math.exp(0.75 * math.log(sig[500]) + 0.25 * math.log(sig[501])))
    for t in (0.0, 3.2, 517.37, 998.5, 999.0):
        assert s.timestep(s.sigma(t)) == pytest.approx(t, abs=1e-9)
    assert s.timestep(SIGMA_MIN / 2) == 0.0 and s.timestep(SIGMA_MAX * 2) == 999.0  # (clamped)


def test_schedule_config() -> None:
    path = os.path.join(sd15_dir(), "scheduler", "scheduler_config.json")
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    assert torch.equal(Schedule.from_config(path).log_sigmas, Schedule().log_sigmas)


def test_sigmas() -> None:
    s = Schedule()
    for sampler in ("euler", "euler_a"):
        sig = s.sigmas(20, sampler)  # sd.cpp's "discrete": 20 timesteps evenly from 999 to 0
        assert sig == pytest.approx([s.sigma(t) for t in np.linspace(999, 0, 20)] + [0.0], rel=1e-12)
        assert sig[0] == pytest.approx(SIGMA_MAX, rel=1e-6) and sig[-2] == pytest.approx(SIGMA_MIN, rel=1e-6)
    # sd.cpp's "simple" for DDIM: timesteps 999, 949, ..., 49
    assert s.sigmas(20, "ddim") == [s.sigma(999 - 50 * i) for i in range(20)] + [0.0]
    assert s.sigmas(3, "ddim") == [s.sigma(t) for t in (999, 666, 333)] + [0.0]
    for sampler in SAMPLERS:
        assert s.sigmas(1, sampler) == [s.sigma(999), 0.0]


def _recorded(sampler: str, *, seed: int = 0) -> tuple[torch.Tensor, list[float], list[torch.Tensor], torch.Tensor]:
    """A sampler run with a denoiser that always answers x0: (x0, sigmas, the x of each step, the result)."""
    x0 = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(100))
    sigmas = Schedule().sigmas(10, sampler)
    seen: list[torch.Tensor] = []

    def denoise(x: torch.Tensor, sigma: float) -> torch.Tensor:
        seen.append(x)
        return x0

    gen = torch.Generator().manual_seed(seed)
    x = x0 + torch.randn(x0.shape, generator=gen) * sigmas[0]
    return x0, sigmas, seen, sample(denoise, x, sigmas, sampler=sampler, generator=gen)


@pytest.mark.parametrize("sampler", ["euler", "ddim"])
def test_deterministic_samplers(sampler: str) -> None:
    # Euler: x - x0 shrinks with sigma, then x0 at sigma 0
    x0, sigmas, seen, out = _recorded(sampler)
    assert len(seen) == len(sigmas) - 1
    for s, x in zip(sigmas, seen):
        torch.testing.assert_close(x - x0, (seen[0] - x0) * (s / sigmas[0]), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out, x0)


def test_euler_ancestral() -> None:
    # each step's noise tops the shrunk deviation back up to the next sigma
    x0, sigmas, seen, out = _recorded("euler_a")
    for s, x in zip(sigmas, seen):
        assert float((x - x0).std()) == pytest.approx(s, rel=0.03)
    torch.testing.assert_close(out, x0)
    again = _recorded("euler_a")[2]
    assert all(torch.equal(a, b) for a, b in zip(seen, again))
    assert not torch.equal(_recorded("euler_a", seed=1)[2][2], seen[2])


def test_unknown_sampler() -> None:
    with pytest.raises(ValueError, match="sampler"):
        sample(lambda x, s: x, torch.zeros(1), [1.0, 0.0], sampler="heun")


def test_cfg_denoiser() -> None:
    s = Schedule()
    context = torch.randn(2, 77, 768)
    calls = []

    def unet(x: torch.Tensor, t: float, ctx: torch.Tensor) -> torch.Tensor:
        calls.append((x, t, ctx))
        return torch.cat([torch.full_like(x[:1], 1.0), torch.full_like(x[:1], 3.0)])  # [uncond, cond]

    x, sigma, scale = torch.randn(1, 4, 64, 64), 5.0, 7.5
    out = cfg_denoiser(unet, s, context, scale)(x, sigma)
    xin, t, ctx = calls[0]
    torch.testing.assert_close(xin, torch.cat([x, x]) / math.sqrt(sigma ** 2 + 1))
    assert t == s.timestep(sigma) and ctx is context
    torch.testing.assert_close(out, x - sigma * (1.0 + scale * 2.0))
