"""Stable Diffusion 1.5's UNet on the NPU: ``NpuUNet``, the pipeline's noise predictor (``artlite.operators.sd15_v2``'s
``model.UNet``: the whole step as one dispatch of its static design, its weights in DDR images).

Per step the host writes the latents and the 22 resnets' time biases (``host``), runs the dispatch and reads the 4
output channels; per prompt (a new context) it writes the cross attentions' K / V^T (computed on the host). The first
build writes the weight images and builds the dispatch's transaction (the design's compile is cached).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from artlite.operators.sd15_v2.model import UNet
from artlite.operators.sd15_v2.runtime import V2Device

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

    def __call__(self, x: torch.Tensor, t: float, context: torch.Tensor) -> torch.Tensor:
        self.net.set_time(resnet_time_biases(self.w, t, pool=self.pool))
        if self.context is None or not torch.equal(self.context, context):
            self.net.set_context(context)
            self.context = context.clone()
        return self.net(x)
