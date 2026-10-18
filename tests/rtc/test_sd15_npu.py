"""Stable Diffusion 1.5 with the UNet and the VAE decoder on the NPU (models.sd15.npu, artlite.operators.sd15_v2): one
UNet step and one decode against the fp32 reference, and text to image against the CPU backend. Slow: building them
writes their weight images and builds their transactions."""

import numpy as np
import pytest
import torch

from artlite.model_config import sd15_dir
from models import is_diffusers
from models.sd15 import reference
from models.sd15.npu import NpuUNet, NpuVae
from models.sd15.pipeline import StableDiffusion

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model_dir() -> str:
    path = sd15_dir()
    if not is_diffusers(path):
        pytest.skip(f"{path}: not present")
    return path


@pytest.fixture(scope="module")
def npu_unet(model_dir: str) -> NpuUNet:
    return NpuUNet(f"{model_dir}/unet")


@pytest.fixture(scope="module")
def npu_vae(model_dir: str, npu_unet: NpuUNet) -> NpuVae:
    return NpuVae(f"{model_dir}/vae", npu_unet.net.v2)


def test_unet_step(model_dir: str, npu_unet: NpuUNet) -> None:
    """One step on random latents and context against the fp32 UNet: within 2.5e-2 (measured 1.19e-2: bfp16 operands,
    bf16 intermediates, the 64x64 attention's exp2 approximation), and the same again (no state leaks)."""
    torch.manual_seed(0)
    x, ctx = torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768)
    ref = reference.UNet(f"{model_dir}/unet")(x, 500.0, ctx)
    out = npu_unet(x, 500.0, ctx)
    err = float((out - ref).norm() / ref.norm())
    assert err < 2.5e-2, f"rel err {err:.3e}"
    torch.testing.assert_close(npu_unet(x, 500.0, ctx), out, rtol=0, atol=0)


def test_vae(model_dir: str, npu_vae: NpuVae) -> None:
    """One decode of random latents against the fp32 decoder: within 2e-2 (measured 1.33e-2), and the same again."""
    torch.manual_seed(0)
    lat = torch.randn(1, 4, 64, 64)
    ref = reference.VaeDecoder(f"{model_dir}/vae")(lat)
    out = npu_vae(lat)
    err = float((out - ref).norm() / ref.norm())
    assert err < 2e-2, f"rel err {err:.3e}"
    torch.testing.assert_close(npu_vae(lat), out, rtol=0, atol=0)


def test_txt2img(model_dir: str, npu_unet: NpuUNet, npu_vae: NpuVae) -> None:
    """3 euler steps against the CPU backend's image (measured: latents 0.29 relative, pixels 10.1 / 255 mean). The
    guidance is the small difference of two predictions (cond - uncond: ~0.3% of a prediction at t = 500 with the
    real context), which each NPU prediction's ~0.8% error swamps and the scale (7) amplifies; 20-step images show
    the same picture."""
    kw = {"steps": 3, "sampler": "euler", "seed": 7}
    prompt = "a photograph of an astronaut riding a horse"
    cpu = StableDiffusion(model_dir, backend="cpu").txt2img(prompt, **kw)
    sd = StableDiffusion(model_dir, backend="cpu")
    sd.backend.unet, sd.backend.vae = npu_unet, npu_vae
    npu = sd.txt2img(prompt, **kw)
    err = float((npu.latents - cpu.latents).norm() / cpu.latents.norm())
    diff = np.abs(npu.pixels.astype(int) - cpu.pixels.astype(int)).mean()
    print(f"\nlatents rel err {err:.3e}, pixels mean |diff| {diff:.2f}; npu steps {npu.t_steps:.2f} s")
    assert err < 0.8 and diff < 25


def test_load(model_dir: str, npu_unet: NpuUNet, npu_vae: NpuVae) -> None:
    """``npu.load`` (the npu backend's): built and cached, or restored from the cache (the device's images, the
    transactions with their shim addresses relocated), then restored again: a step and a decode bit-identical to the
    models built in this process."""
    from models.sd15 import npu
    torch.manual_seed(0)
    x, ctx, lat = torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768), torch.randn(1, 4, 64, 64)
    y, img = npu_unet(x, 500.0, ctx), npu_vae(lat)
    for _ in range(2):
        unet, vae = npu.load(model_dir)
        torch.testing.assert_close(unet(x, 500.0, ctx), y, rtol=0, atol=0)
        torch.testing.assert_close(vae(lat), img, rtol=0, atol=0)
        del unet, vae
