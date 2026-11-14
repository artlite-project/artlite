"""Stable Diffusion 1.5's UNet and VAE decoder on the NPU: ``NpuUNet``, the pipeline's noise predictor
(``model.UNet``: the whole step as one dispatch of the vision overlay, ``artlite.vision``, its weights in DDR images),
and ``NpuVae``, its image decoder (``vae.Decoder`` on the same design: one dispatch).

Per step the host writes the latents and the 22 resnets' time biases (``host``: the steps' computed ahead on a worker
thread, ``NpuUNet.prepare``), runs the dispatch and reads the 4 output channels; per prompt (a new context) it writes
the cross attentions' K / V^T (computed on the host). Every load packs the weight images from the safetensors files
(``artlite.vision.weights``: the native packer, the files read in place); the first one also builds the dispatches'
transactions, which later loads restore from the cache (``load``; the design's compile is cached too).
"""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from artlite.vision.device import VisionDevice
from artlite.vision.shell import attach
from artlite.vision.state import state_dir
from artlite.vision.weights import Source, safetensors, sources

from .host import resnet_time_biases
from .model import UNet
from .vae import EPS, Decoder, latent_conv_in
from .weights import load_safetensors

FILE = "diffusion_pytorch_model.safetensors"
VAE_PREFIXES = ("decoder.", "post_quant_conv.")  # (the VAE decoder's tensors)


class NpuUNet:
    """The UNet of a diffusers directory's ``unet`` on the NPU: (x (2, 4, 64, 64) scaled, t, context (2, 77, 768))
    -> the predicted noise (2, 4, 64, 64)."""

    def __init__(self, directory: str, *, npu: VisionDevice, state: dict | None = None):
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["block_out_channels"] == [320, 640, 1280, 1280] and cfg["attention_head_dim"] == 8, \
            f"{directory}: not SD 1.x's UNet"
        path = os.path.join(directory, FILE)
        self.w = load_safetensors(path, contains=UNet.RUNTIME)  # (the host's)
        if state is None:
            self.net = UNet.build(npu, safetensors(path, tag="unet"), eps=cfg["norm_eps"], runtime=self.w)
        else:  # (a restored device's: ``load``)
            self.net = UNet(npu, self.w, **state)
        self.context: np.ndarray | None = None
        self.pool = ThreadPoolExecutor(8)  # (the time biases' projections)
        self.worker = ThreadPoolExecutor(1)  # (the steps' time biases ahead of them: ``prepare``)
        self.biases: dict[float, Future] = {}

    def prepare(self, timesteps: list[float]) -> None:
        """The steps' time biases computed ahead, on a worker thread (the pipeline: before its text encoder runs)."""
        for t in timesteps:
            if t not in self.biases:
                self.biases[t] = self.worker.submit(resnet_time_biases, self.w, t, pool=self.pool)

    def __call__(self, x: np.ndarray, t: float, context: np.ndarray) -> np.ndarray:
        f = self.biases.pop(t, None)
        self.net.set_time(f.result() if f else resnet_time_biases(self.w, t, pool=self.pool))
        if self.context is None or not np.array_equal(self.context, context):
            self.net.set_context(context, pool=self.pool)
            self.context = np.array(context, np.float32)
        return self.net(x)


class NpuVae:
    """The VAE decoder of a diffusers directory's ``vae`` on the NPU (the device of a ``NpuUNet``'s design, one
    dispatch): the sampler's latents (B, 4, 64, 64) -> the images (B, 3, 512, 512), one at a time."""

    def __init__(self, directory: str, npu: VisionDevice, *, state: dict | None = None):
        if state is None:
            w, scale = vae_weights(directory)
            self.dec = Decoder.build(npu, w, scale=scale, eps=EPS)
        else:  # (a restored device's: ``load``)
            self.dec = Decoder(npu, **state)

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        return np.concatenate([self.dec(latents[i:i + 1]) for i in range(latents.shape[0])])


MODULES = ("models.sd15.npu",)  # (the modules whose code and imports build the cached state: ``state.state_dir``)


def vae_weights(directory: str) -> tuple[dict, float]:
    """A diffusers ``vae`` directory's decoder weights (``weights.safetensors``) and its scaling factor."""
    with open(os.path.join(directory, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["block_out_channels"] == [128, 256, 512, 512] and cfg["layers_per_block"] == 2 and \
        cfg.get("latent_channels", 4) == 4, f"{directory}: not SD 1.x's VAE"
    return safetensors(os.path.join(directory, FILE), VAE_PREFIXES, tag="vae"), cfg.get("scaling_factor", 0.18215)


def weight_sources(path: str) -> dict[str, Source]:
    """The weights' sources of a diffusers directory's UNet and VAE decoder: the files' tensors, the VAE's computed
    conv_in (``vae.latent_conv_in``)."""
    w, scale = vae_weights(os.path.join(path, "vae"))
    return sources(safetensors(os.path.join(path, "unet", FILE), tag="unet")) | sources(w) | \
        sources(latent_conv_in(w, scale))


def load(path: str) -> tuple[NpuUNet, NpuVae]:
    """A diffusers directory's UNet and VAE decoder on one NPU device: restored from the cache
    (``artlite.vision.shell.attach``: the device's layout, the commands and tables the builds wrote, the programs with
    their shim addresses by image; keyed by the model's files and the code; the weights packed from the files), else
    built and cached (one entry per model)."""
    files = [os.path.join(path, sub, f) for sub in ("unet", "vae") for f in ("config.json", FILE)]
    nets: dict[str, NpuUNet | NpuVae] = {}

    def build(npu: VisionDevice, st: dict | None) -> list:
        nets["unet"] = NpuUNet(os.path.join(path, "unet"), npu=npu, state=st and st["unet"])
        nets["vae"] = NpuVae(os.path.join(path, "vae"), npu, state=st and st["vae"])
        return [nets["unet"].net, nets["vae"].dec]

    attach(state_dir("sd15", path, MODULES, files), weight_sources(path), build,
           lambda: {"unet": nets["unet"].net.state(), "vae": nets["vae"].dec.state()})
    return nets["unet"], nets["vae"]
