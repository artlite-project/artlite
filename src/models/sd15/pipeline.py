"""Text to image with Stable Diffusion 1.5 (``StableDiffusion.txt2img``): the prompt and the negative prompt through
the text encoder, the sampler's steps on the UNet with classifier-free guidance, the final latents through the VAE
decoder.

A ``Backend`` runs the three networks. Each part is a callable of its own protocol, so that the NPU's parts can
replace the CPU's one at a time:
- ``TextEncoder``: token ids (2, 77) int64 -> the last hidden states (2, 77, 768) fp32 (after the final layer norm);
- ``NoisePredictor``: latents (2, 4, 64, 64) fp32 already scaled by 1 / sqrt(sigma^2 + 1), a fractional timestep
  (``host.resnet_time_biases`` turns it into each resnet's time bias) and the context (2, 77, 768) -> the predicted
  noise (2, 4, 64, 64). The batch is [uncond, cond]. The context is the same tensor for every step of an image:
  what depends only on it (each cross-attention's K and V) can be computed once per image; one with a
  ``prepare(timesteps)`` gets the steps' timesteps before the text encoder runs (per-timestep work started early);
- ``ImageDecoder``: the sampler's final latents (1, 4, 64, 64) -> the image (1, 3, 512, 512) in about [-1, 1].
``load_backend``: the UNet and the VAE decoder on the NPU (``npu``), the text encoder on the host (``text``); tests run
the pipeline on their fp32 references' backend. The noise of a seed: numpy's PCG64 (``np.random.default_rng``).
"""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .sampling import Schedule, cfg_denoiser, sample
from .text import ClipText
from .tokenizer import ClipTokenizer

SIZES = (512,)  # image widths and heights the pipeline runs (the NPU's designs are for 512 x 512)
LATENT_CHANNELS = 4
LATENT_SCALE = 8  # image pixels per latent pixel


class TextEncoder(Protocol):
    def __call__(self, ids: np.ndarray) -> np.ndarray: ...


class NoisePredictor(Protocol):
    def __call__(self, x: np.ndarray, t: float, context: np.ndarray) -> np.ndarray: ...


class ImageDecoder(Protocol):
    def __call__(self, latents: np.ndarray) -> np.ndarray: ...


@dataclass
class Backend:
    """What runs each network."""

    text_encoder: TextEncoder
    unet: NoisePredictor
    vae: ImageDecoder


def load_backend(path: str, *, prompts: list[list[int]] = ()) -> Backend:
    """The networks of a diffusers directory: the UNet and the VAE decoder on the NPU (one design), the text encoder on
    the host (``prompts``: token ids it encodes meanwhile, kept)."""
    from . import npu
    text = ClipText(os.path.join(path, "text_encoder"))
    if prompts:
        text.prepare(np.array(prompts, np.int64))
    unet, vae = npu.load(path)
    return Backend(text_encoder=text, unet=unet, vae=vae)


def check_request(*, width: int, height: int, steps: int) -> None:
    """ValueError for a request the pipeline does not run."""
    if width not in SIZES or height not in SIZES:
        raise ValueError(f"{width}x{height}: only {' and '.join(f'{s}x{s}' for s in SIZES)} images for now")
    if steps < 1:
        raise ValueError(f"{steps} steps: at least one")


def check_pipeline(path: str) -> None:
    """NotImplementedError unless ``path`` is a diffusers directory of a Stable Diffusion 1.x pipeline."""
    with open(os.path.join(path, "model_index.json")) as f:
        cls = json.load(f).get("_class_name")
    with open(os.path.join(path, "unet", "config.json")) as f:
        unet = json.load(f)
    if cls != "StableDiffusionPipeline" or unet.get("cross_attention_dim") != 768 or unet.get("use_linear_projection"):
        raise NotImplementedError(f"{path}: not a Stable Diffusion 1.x pipeline ({cls}, cross_attention_dim "
                                  f"{unet.get('cross_attention_dim')}); only SD 1.x is supported")


def pixels(image: np.ndarray) -> np.ndarray:
    """An image (3, H, W) in [-1, 1] as (H, W, 3) uint8."""
    return np.round((np.clip(image, -1, 1) + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)


@dataclass
class Txt2Img:
    """An image generated from a prompt, and where the time went."""

    pixels: np.ndarray  # (H, W, 3) uint8
    latents: np.ndarray  # (1, 4, H / 8, W / 8): the sampler's result
    t_text: float  # seconds: text encoder
    t_steps: float  # the sampler's steps
    t_vae: float  # VAE decoder


class StableDiffusion:
    """A diffusers directory of Stable Diffusion 1.5 (``backend``: what runs the networks, default ``load_backend``'s)."""

    name = "Stable Diffusion 1.5"

    def __init__(self, path: str, *, backend: Backend | None = None):
        check_pipeline(path)
        self.tokenizer = ClipTokenizer(os.path.join(path, "tokenizer"))
        self.schedule = Schedule.from_config(os.path.join(path, "scheduler", "scheduler_config.json"))
        self.backend = backend or load_backend(path, prompts=[self.tokenizer.encode("")])  # (the usual negative prompt)

    def txt2img(self, prompt: str, *, negative_prompt: str = "", steps: int = 20, cfg_scale: float = 7.0,
                sampler: str = "euler_a", seed: int = 42, width: int = 512, height: int = 512,
                progress: Callable[[int, int], None] | None = None) -> Txt2Img:
        """An image of ``prompt``: ``steps`` steps of ``sampler`` (``SAMPLERS``) with guidance ``cfg_scale`` away from
        ``negative_prompt``, from the noise of ``seed``; ``progress(step, steps)`` after each step."""
        check_request(width=width, height=height, steps=steps)
        b = self.backend
        t0 = time.perf_counter()
        sigmas = self.schedule.sigmas(steps, sampler)
        if hasattr(b.unet, "prepare"):
            b.unet.prepare([self.schedule.timestep(s) for s in sigmas[:-1]])
        ids = np.array([self.tokenizer.encode(negative_prompt), self.tokenizer.encode(prompt)], np.int64)
        context = b.text_encoder(ids)
        t1 = time.perf_counter()
        rng = np.random.default_rng(seed)
        shape = (1, LATENT_CHANNELS, height // LATENT_SCALE, width // LATENT_SCALE)
        x = rng.standard_normal(shape, np.float32) * sigmas[0]
        x = sample(cfg_denoiser(b.unet, self.schedule, context, cfg_scale), x, sigmas, sampler=sampler, rng=rng,
                   callback=None if progress is None else lambda i: progress(i + 1, steps))
        t2 = time.perf_counter()
        image = b.vae(x)[0]
        t3 = time.perf_counter()
        return Txt2Img(pixels=pixels(image), latents=x, t_text=t1 - t0, t_steps=t2 - t1, t_vae=t3 - t2)
