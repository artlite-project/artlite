"""Stable Diffusion 1.5's UNet and VAE decoder on the NPU: ``NpuUNet``, the pipeline's noise predictor
(``artlite.operators.sd15``'s ``model.UNet``: the whole step as one dispatch of its static design, its weights in
DDR images), and ``NpuVae``, its image decoder (``vae.Decoder`` on the same design: one dispatch).

Per step the host writes the latents and the 22 resnets' time biases (``host``: the steps' computed ahead on a worker
thread, ``NpuUNet.prepare``), runs the dispatch and reads the 4 output channels; per prompt (a new context) it writes
the cross attentions' K / V^T (computed on the host). The first build writes the weight images and builds the
dispatches' transactions (the design's compile is cached).
"""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from artlite.operators.sd15.model import UNet
from artlite.operators.sd15.runtime import Sd15Device, load_state, save_state, state_dir
from artlite.operators.sd15.vae import Decoder

from . import reference
from .host import resnet_time_biases
from .weights import load_safetensors


class NpuUNet:
    """The UNet of a diffusers directory's ``unet`` on the NPU: (x (2, 4, 64, 64) scaled, t, context (2, 77, 768))
    -> the predicted noise (2, 4, 64, 64)."""

    def __init__(self, directory: str, *, npu: Sd15Device | None = None, state: dict | None = None):
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["block_out_channels"] == [320, 640, 1280, 1280] and cfg["attention_head_dim"] == 8, \
            f"{directory}: not SD 1.x's UNet"
        path = os.path.join(directory, "diffusion_pytorch_model.safetensors")
        if state is None:
            self.w = load_safetensors(path)
            self.net = UNet(npu or Sd15Device(), self.w, eps=cfg["norm_eps"])
            self.net.npu.warm()
        else:  # (a restored device's: ``load``)
            self.w = load_safetensors(path, contains=UNet.RUNTIME)
            self.net = UNet.restored(npu, self.w, state)
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
            self.net.set_context(context, pool=self.pool)
            self.context = context.clone()
        return self.net(x)


class NpuVae:
    """The VAE decoder of a diffusers directory's ``vae`` on the NPU (the device of a ``NpuUNet``'s design, one
    dispatch): the sampler's latents (B, 4, 64, 64) -> the images (B, 3, 512, 512), one at a time."""

    def __init__(self, directory: str, npu: Sd15Device, *, state: dict | None = None):
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["block_out_channels"] == [128, 256, 512, 512] and cfg["layers_per_block"] == 2 and \
            cfg.get("latent_channels", 4) == 4, f"{directory}: not SD 1.x's VAE"
        if state is None:
            w = load_safetensors(os.path.join(directory, "diffusion_pytorch_model.safetensors"), ("decoder.", "post_quant_conv."))
            self.dec = Decoder(npu, w, scale=cfg.get("scaling_factor", 0.18215), eps=reference.VaeDecoder.EPS)
            npu.warm()
        else:  # (a restored device's: ``load``)
            self.dec = Decoder.restored(npu, state)

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.dec(latents[i:i + 1]) for i in range(latents.shape[0])])


# (what the cache's key covers besides the model's files: the code that builds the images and the transactions)
_SOURCES = ("artlite/operators/sd15", "artlite/txn.py", "artlite/operators/bfp16.py", "artlite/operators/act_lut.py",
            "artlite/operators/fused/ctrlpkt.py", "artlite/operators/paging.py", "artlite/operators/overlay/runtime.py",
            "models/sd15/npu.py",
            "models/sd15/reference.py", "models/sd15/weights.py")


def load(path: str) -> tuple[NpuUNet, NpuVae]:
    """A diffusers directory's UNet and VAE decoder on one NPU device: restored from the cache (``runtime.state_dir``:
    the device's layout, the images the builds wrote (weights, commands, tables), the transactions with their shim
    addresses relocated; keyed by the model's files and the code), else built and cached (one entry per model)."""
    files = [os.path.join(path, sub, f) for sub in ("unet", "vae") for f in ("config.json", "diffusion_pytorch_model.safetensors")]
    cache = state_dir("sd15", path, _SOURCES, files)
    if (got := load_state(cache)) is not None:
        npu, st = got
        unet = NpuUNet(os.path.join(path, "unet"), npu=npu, state=st["unet"])
        vae = NpuVae(os.path.join(path, "vae"), npu, state=st["vae"])
        npu.warm()
        return unet, vae
    unet = NpuUNet(os.path.join(path, "unet"))
    npu = unet.net.npu
    vae = NpuVae(os.path.join(path, "vae"), npu)
    save_state(cache, npu, unet.net.images + vae.dec.images, {"unet": unet.net.state(), "vae": vae.dec.state()})
    return unet, vae
