"""Stable Diffusion 1.5's UNet and VAE decoder on the NPU: ``NpuUNet``, the pipeline's noise predictor
(``artlite.operators.sd15_v2``'s ``model.UNet``: the whole step as one dispatch of its static design, its weights in
DDR images), and ``NpuVae``, its image decoder (``vae.Decoder`` on the same design: two dispatches around the mid
block's attention, which runs on the host in bf16).

Per step the host writes the latents and the 22 resnets' time biases (``host``: the steps' computed ahead on a worker
thread, ``NpuUNet.prepare``), runs the dispatch and reads the 4 output channels; per prompt (a new context) it writes the cross attentions' K / V^T (computed on the host). The first
build writes the weight images and builds the dispatches' transactions (the design's compile is cached).
"""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from artlite.operators.sd15_v2.model import UNet
from artlite.operators.sd15_v2.runtime import V2Device
from artlite.operators.sd15_v2.vae import Decoder

from . import reference
from .host import resnet_time_biases
from .weights import load_safetensors


class NpuUNet:
    """The UNet of a diffusers directory's ``unet`` on the NPU: (x (2, 4, 64, 64) scaled, t, context (2, 77, 768))
    -> the predicted noise (2, 4, 64, 64)."""

    def __init__(self, directory: str):
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["block_out_channels"] == [320, 640, 1280, 1280] and cfg["attention_head_dim"] == 8, \
            f"{directory}: not SD 1.x's UNet"
        self.w = load_safetensors(os.path.join(directory, "diffusion_pytorch_model.safetensors"))
        self.net = UNet(V2Device(), self.w, eps=cfg["norm_eps"])
        self.context: torch.Tensor | None = None
        self.pool = ThreadPoolExecutor(8)  # (the time biases' projections)
        self.worker = ThreadPoolExecutor(1)  # (the steps' time biases ahead of them: ``prepare``)
        self.biases: dict[float, Future] = {}

    def prepare(self, timesteps: list[float]) -> None:
        """The steps' time biases computed ahead, on a worker thread (the pipeline: before its text encoder runs)."""
        for t in timesteps:
            if t not in self.biases:
                self.biases[t] = self.worker.submit(resnet_time_biases, self.w, t, pool=self.pool)

    def __call__(self, x: torch.Tensor, t: float, context: torch.Tensor) -> torch.Tensor:
        f = self.biases.pop(t, None)
        self.net.set_time(f.result() if f else resnet_time_biases(self.w, t, pool=self.pool))
        if self.context is None or not torch.equal(self.context, context):
            self.net.set_context(context)
            self.context = context.clone()
        return self.net(x)


class NpuVae:
    """The VAE decoder of a diffusers directory's ``vae`` on the NPU (the device of a ``NpuUNet``'s design): the
    sampler's latents (B, 4, 64, 64) -> the images (B, 3, 512, 512), one at a time."""

    def __init__(self, directory: str, v2: V2Device):
        self.ref = reference.VaeDecoder(directory)
        cfg = self.ref.cfg
        assert cfg["block_out_channels"] == [128, 256, 512, 512] and cfg["layers_per_block"] == 2 and \
            cfg.get("latent_channels", 4) == 4, f"{directory}: not SD 1.x's VAE"
        self.dec = Decoder(v2, self.ref.w, scale=self.ref.scale, eps=self.ref.EPS)

    def _attend(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return self.ref.attention("decoder.mid_block.attentions.0.", h).float()

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.dec(latents[i:i + 1], self._attend) for i in range(latents.shape[0])])
