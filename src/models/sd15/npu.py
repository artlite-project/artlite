"""Stable Diffusion 1.5's UNet on the NPU: ``NpuUNet``, the pipeline's noise predictor (``artlite.operators.sd15``'s
UNet program: the whole step in one library program; its 16x16 / 8x8 levels by ``artlite.operators.sd15_v2``'s
section, one raw item of the program, whose weights, time biases and context K / V^T are in its own buffers on the
program's device).

Per step the host writes the latents (conv_in's layout) and the 22 resnets' time biases (``host``) into the program's
arena and the section's buffers, runs it and reads the 4 output channels; per prompt (a new context) it writes the
cross attentions' K / V (computed on the host). The first build packs the weight images and compiles the program
(cached).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from artlite.operators.sd15.unet import UNetProgram, latents_layout, output_values, ysz
from artlite.operators.sd15_v2.runtime import V2Device
from artlite.operators.sd15_v2.unet import Section

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
        u = self.unet = UNetProgram(self.w)
        u.build()
        u.prog.compile()
        self.arena = u.arena()
        self.context: torch.Tensor | None = None
        self.pool = ThreadPoolExecutor(8)  # (the time biases' projections)
        dev = V2Device(u.v2_design, device=u.prog.lib.device)
        self.section = Section(dev, self.w, x_in=self.arena.addr(u.v2_in), out=self.arena.addr(u.v2_out),
                               eps=cfg["norm_eps"])
        u.v2_raw.words = self.section.words
        self.bufs = list(dev.dev.buffers.values())

    def __call__(self, x: torch.Tensor, t: float, context: torch.Tensor) -> torch.Tensor:
        a, u = self.arena, self.unet
        a.put("lat", latents_layout(x))
        biases = resnet_time_biases(self.w, t, pool=self.pool)
        for name, img in u.time_bias_images(biases):
            a.put(name, img)
        self.section.set_time(biases)
        if self.context is None or not torch.equal(self.context, context):
            for name, fn in u.kv_packs:
                a.put(name, fn(context))
            self.section.set_context(context)
            self.context = context.clone()
        u.prog.run(a, bufs=self.bufs)
        return output_values(a.get("out", ysz(80, 64)))
