"""Stable Diffusion 1.5's UNet and VAE decoder on the NPU: ``NpuUNet``, the pipeline's noise predictor
(``artlite.operators.sd15_v2``'s ``model.UNet``: the whole step as one dispatch of its static design, its weights in
DDR images), and ``NpuVae``, its image decoder (``vae.Decoder`` on the same design: two dispatches around the mid
block's attention, which runs on the host in bf16).

Per step the host writes the latents and the 22 resnets' time biases (``host``: the steps' computed ahead on a worker
thread, ``NpuUNet.prepare``), runs the dispatch and reads the 4 output channels; per prompt (a new context) it writes the cross attentions' K / V^T (computed on the host). The first
build writes the weight images and builds the dispatches' transactions (the design's compile is cached).
"""

import glob
import hashlib
import json
import os
import pickle
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import version

import torch

from artlite.operators.base import CACHE_DIR
from artlite.operators.sd15_v2.model import UNet
from artlite.operators.sd15_v2.runtime import V2Device
from artlite.operators.sd15_v2.vae import Decoder

from . import reference
from .host import resnet_time_biases
from .weights import load_safetensors


class NpuUNet:
    """The UNet of a diffusers directory's ``unet`` on the NPU: (x (2, 4, 64, 64) scaled, t, context (2, 77, 768))
    -> the predicted noise (2, 4, 64, 64)."""

    def __init__(self, directory: str, *, v2: V2Device | None = None, state: dict | None = None):
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["block_out_channels"] == [320, 640, 1280, 1280] and cfg["attention_head_dim"] == 8, \
            f"{directory}: not SD 1.x's UNet"
        path = os.path.join(directory, "diffusion_pytorch_model.safetensors")
        if state is None:
            self.w = load_safetensors(path)
            self.net = UNet(v2 or V2Device(), self.w, eps=cfg["norm_eps"])
            self.net.v2.warm()
        else:  # (a restored device's: ``load``)
            self.w = load_safetensors(path, contains=UNet.RUNTIME)
            self.net = UNet.restored(v2, self.w, state)
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
    """The VAE decoder of a diffusers directory's ``vae`` on the NPU (the device of a ``NpuUNet``'s design): the
    sampler's latents (B, 4, 64, 64) -> the images (B, 3, 512, 512), one at a time."""

    def __init__(self, directory: str, v2: V2Device, *, state: dict | None = None):
        self.ref = reference.VaeDecoder(directory)
        cfg = self.ref.cfg
        assert cfg["block_out_channels"] == [128, 256, 512, 512] and cfg["layers_per_block"] == 2 and \
            cfg.get("latent_channels", 4) == 4, f"{directory}: not SD 1.x's VAE"
        if state is None:
            self.dec = Decoder(v2, self.ref.w, scale=self.ref.scale, eps=self.ref.EPS)
            v2.warm()
        else:  # (a restored device's: ``load``)
            self.dec = Decoder.restored(v2, state)

    def _attend(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return self.ref.attention("decoder.mid_block.attentions.0.", h).float()

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.dec(latents[i:i + 1], self._attend) for i in range(latents.shape[0])])


# (what the cache's key covers besides the model's files: the code that builds the images and the transactions)
_SOURCES = ("artlite/operators/sd15_v2", "artlite/txn.py", "artlite/operators/bfp16.py", "artlite/operators/act_lut.py",
            "artlite/operators/fused/ctrlpkt.py", "artlite/operators/overlay/runtime.py", "models/sd15/npu.py",
            "models/sd15/reference.py", "models/sd15/weights.py")


def _key(path: str) -> str:
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # (the src directory)
    h = hashlib.sha256(version("llvm-aie").encode())
    for s in _SOURCES:
        p = os.path.join(src, s)
        for f in sorted(glob.glob(os.path.join(p, "*.py"))) if os.path.isdir(p) else [p]:
            with open(f, "rb") as fh:
                h.update(fh.read())
    for sub in ("unet", "vae"):
        for f in ("config.json", "diffusion_pytorch_model.safetensors"):
            st = os.stat(os.path.join(path, sub, f))
            h.update(f"{sub}/{f} {st.st_size} {st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def load(path: str) -> tuple[NpuUNet, NpuVae]:
    """A diffusers directory's UNet and VAE decoder on one NPU device: restored from the cache (``CACHE_DIR/sd15``:
    the device's layout, the images the builds wrote (weights, commands, tables), the transactions with their shim
    addresses relocated; keyed by the model's files and the code), else built and cached (one entry per model)."""
    root = os.path.join(CACHE_DIR, "sd15", hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:12])
    cache = os.path.join(root, _key(path))
    if os.path.exists(os.path.join(cache, "state.pkl")):
        with open(os.path.join(cache, "state.pkl"), "rb") as f:
            st = pickle.load(f)
        v2 = V2Device()
        v2.restore(st["layout"])
        with open(os.path.join(cache, "images.bin"), "rb") as f:
            for name, at, n in st["images"]:
                f.seek(at)
                f.readinto(v2.view(name)[:n])
                v2.flush(name)
        unet = NpuUNet(os.path.join(path, "unet"), v2=v2, state=st["unet"])
        vae = NpuVae(os.path.join(path, "vae"), v2, state=st["vae"])
        v2.warm()
        return unet, vae
    unet = NpuUNet(os.path.join(path, "unet"))
    v2 = unet.net.v2
    vae = NpuVae(os.path.join(path, "vae"), v2)
    tmp = f"{cache}.{os.getpid()}"
    os.makedirs(tmp)
    index, at = [], 0
    with open(os.path.join(tmp, "images.bin"), "wb") as f:
        for name in dict.fromkeys(unet.net.images + vae.dec.images):
            v = v2.view(name)
            f.write(v.tobytes())
            index.append((name, at, v.size))
            at += v.size
    with open(os.path.join(tmp, "state.pkl"), "wb") as f:
        pickle.dump({"layout": v2.layout(), "images": index, "unet": unet.net.state(), "vae": vae.dec.state()}, f)
    for e in os.listdir(root):  # (the model's older entries)
        if e != os.path.basename(tmp):
            shutil.rmtree(os.path.join(root, e), ignore_errors=True)
    os.replace(tmp, cache)
    return unet, vae
