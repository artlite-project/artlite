"""Stable Diffusion 1.5 with the UNet and the VAE decoder on the NPU (models.sd15.npu on artlite.vision): one
UNet step and one decode against the fp32 reference, and text to image against the reference's. Slow: building them
writes their weight images and builds their transactions."""

import numpy as np
import pytest
import torch

from artlite.model_config import sd15_dir
from artlite.vision.device import VisionDevice
from models import is_diffusers
from models.sd15.npu import NpuUNet, NpuVae
from models.sd15.pipeline import Backend, StableDiffusion

from . import ref as reference

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model_dir() -> str:
    path = sd15_dir()
    if not is_diffusers(path):
        pytest.skip(f"{path}: not present")
    return path


@pytest.fixture(scope="module")
def npu_unet(model_dir: str) -> NpuUNet:
    npu = VisionDevice()
    unet = NpuUNet(f"{model_dir}/unet", npu=npu)
    npu.warm()
    return unet


@pytest.fixture(scope="module")
def npu_vae(model_dir: str, npu_unet: NpuUNet) -> NpuVae:
    return NpuVae(f"{model_dir}/vae", npu_unet.net.npu)


def test_unet_step(model_dir: str, npu_unet: NpuUNet) -> None:
    """One step on random latents and context against the fp32 UNet: within 2.5e-2 (measured 1.19e-2: bfp16 operands,
    bf16 intermediates, the 64x64 attention's exp2 approximation), and the same again (no state leaks)."""
    rng = np.random.default_rng(0)
    x, ctx = rng.standard_normal((2, 4, 64, 64), np.float32), rng.standard_normal((2, 77, 768), np.float32)
    ref = reference.UNet(f"{model_dir}/unet")(torch.from_numpy(x), 500.0, torch.from_numpy(ctx)).numpy()
    out = npu_unet(x, 500.0, ctx)
    err = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    assert err < 2.5e-2, f"rel err {err:.3e}"
    np.testing.assert_array_equal(npu_unet(x, 500.0, ctx), out)


def test_vae(model_dir: str, npu_vae: NpuVae) -> None:
    """One decode of random latents against the fp32 decoder: within 2e-2 (measured 1.33e-2), and the same again."""
    lat = np.random.default_rng(0).standard_normal((1, 4, 64, 64), np.float32)
    ref = reference.VaeDecoder(f"{model_dir}/vae")(torch.from_numpy(lat)).numpy()
    out = npu_vae(lat)
    err = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    assert err < 2e-2, f"rel err {err:.3e}"
    np.testing.assert_array_equal(npu_vae(lat), out)


def test_txt2img(model_dir: str, npu_unet: NpuUNet, npu_vae: NpuVae) -> None:
    """3 euler steps against the fp32 reference's image (measured: latents 0.29 relative, pixels 10.1 / 255 mean). The
    guidance is the small difference of two predictions (cond - uncond: ~0.3% of a prediction at t = 500 with the
    real context), which each NPU prediction's ~0.8% error swamps and the scale (7) amplifies; 20-step images show
    the same picture."""
    kw = {"steps": 3, "sampler": "euler", "seed": 7}
    prompt = "a photograph of an astronaut riding a horse"
    cpu = StableDiffusion(model_dir, backend=reference.backend(model_dir))
    want = cpu.txt2img(prompt, **kw)
    npu = StableDiffusion(model_dir, backend=Backend(cpu.backend.text_encoder, npu_unet, npu_vae)).txt2img(prompt, **kw)
    err = float(np.linalg.norm(npu.latents - want.latents) / np.linalg.norm(want.latents))
    cpu = want
    diff = np.abs(npu.pixels.astype(int) - cpu.pixels.astype(int)).mean()
    print(f"\nlatents rel err {err:.3e}, pixels mean |diff| {diff:.2f}; npu steps {npu.t_steps:.2f} s")
    assert err < 0.8 and diff < 25


def test_load(model_dir: str, npu_unet: NpuUNet, npu_vae: NpuVae) -> None:
    """``npu.load`` (the npu backend's): built and cached, or restored from the cache (the device's images, the
    transactions with their shim addresses relocated), then restored again: a step and a decode bit-identical to the
    models built in this process."""
    from models.sd15 import npu
    rng = np.random.default_rng(0)
    x, ctx = rng.standard_normal((2, 4, 64, 64), np.float32), rng.standard_normal((2, 77, 768), np.float32)
    lat = rng.standard_normal((1, 4, 64, 64), np.float32)
    y, img = npu_unet(x, 500.0, ctx), npu_vae(lat)
    for _ in range(2):
        unet, vae = npu.load(model_dir)
        np.testing.assert_array_equal(unet(x, 500.0, ctx), y)
        np.testing.assert_array_equal(vae(lat), img)
        del unet, vae
