"""SD 1.x's UNet step on sd15_v2's design (``UNet``): one dispatch from the latents to the predicted noise, every level
on the design (64x64: ``level64``; 32x32 .. 8x8: ``unet.outer``), its weights in DDR images.

Per step the host writes the latents (conv_in's PImage, ``level64.latent_panels``) and the resnets' time biases
(``set_time``), runs the dispatch and reads conv_out's channels (``level64.noise``); per prompt the cross-attentions' K /
V^T slots of the text context (``set_context``)."""

from concurrent.futures import Executor

import numpy as np
import torch
from ml_dtypes import bfloat16

from ..design import COLS
from ..overlay.runtime import Transaction
from . import design as D
from . import layout as L
from .build import TBIAS, Builder, Level
from .level64 import LV64, PImage, conv_in64, conv_out64, down64, latent_panels, noise, resnet64, resnet_cat64, up64
from .ops import Attention
from .runtime import V2Device
from .unet import CTX_KEYS, DTensor, outer, skip_bands, transformer_bands


def unet(bd: Builder, w: dict, lat: PImage, *, biases: dict[str, torch.Tensor], ctx_kv: dict[str, int], kv: dict[int, int],
         eps: float) -> None:
    """The UNet from the latents' PImage to conv_out's image "out": conv_in, down_blocks.0 (64x64), down_blocks.1 ..
    up_blocks.2 (``outer``), up_blocks.2's upsampler, up_blocks.3 (64x64), conv_out. kv: per level size its
    self-attention slots' DDR buffer."""
    d0, u3 = "down_blocks.0.", "up_blocks.3."

    def res(p: str, x: DTensor, st: tuple[int, int] | None) -> DTensor:
        return resnet64(bd, w, p, x, cout=320, time_bias=biases[p], eps=eps, stats=st, name=p[:-1])

    def tr(p: str, x: DTensor, st: tuple[int, int] | None) -> DTensor:
        return transformer_bands(bd, w, p, x, kv=kv[64], ctx_kv=ctx_kv[p], nctx=CTX_KEYS, stats=st, name=p[:-1])

    def cat(p: str, h: DTensor, s: DTensor, st: tuple[int, int] | None) -> DTensor:
        return resnet_cat64(bd, w, p, h, s, cout=320, time_bias=biases[p], eps=eps, stats=st, name=p[:-1])
    x0 = conv_in64(bd, w, lat, stats=(10, 0), name="x0")
    s0 = skip_bands(bd, x0, name="s0", st=(20, 320))
    x = tr(d0 + "attentions.0.", res(d0 + "resnets.0.", x0, (10, 0)), (10, 0))
    s1 = skip_bands(bd, x, name="s1", st=(20, 320))
    x = tr(d0 + "attentions.1.", res(d0 + "resnets.1.", x, (10, 0)), None)
    s2 = skip_bands(bd, x, name="s2", st=(30, 640))
    x5 = down64(bd, w, d0 + "downsamplers.0.", x, stats=(10, 0), name="x5")
    s3 = skip_bands(bd, x5, name="s3", st=(30, 640))
    y = outer(bd, w, x5, s3, biases=biases, ctx_kv=ctx_kv, kv=kv[16], kv32=kv[32], eps=eps)
    x = up64(bd, w, "up_blocks.2.upsamplers.0.", y, stats=(30, 0), name="up2")
    x = tr(u3 + "attentions.0.", cat(u3 + "resnets.0.", x, s2, (10, 0)), (20, 0))
    x = tr(u3 + "attentions.1.", cat(u3 + "resnets.1.", x, s1, (10, 0)), (20, 0))
    x = tr(u3 + "attentions.2.", cat(u3 + "resnets.2.", x, s0, (10, 0)), (10, 0))
    conv_out64(bd, w, x, eps=eps, name="out")


class UNet:
    """The UNet step (``unet``) as one dispatch on a V2Device: ``__call__(x)`` (the scaled latents (2, 4, 64, 64)) ->
    the predicted noise; per step the resnets' time biases (``set_time``), per prompt the context (``set_context``)."""

    PREFIXES = [f"down_blocks.{i}.attentions.{j}." for i in range(3) for j in range(2)] + ["mid_block.attentions.0."] + \
        [f"up_blocks.{i}.attentions.{j}." for i in (1, 2, 3) for j in range(3)]

    def __init__(self, v2: V2Device, w: dict, *, eps: float):
        self.v2, self.w = v2, w
        bd = self.bd = Builder(v2)
        lat = self.lat = PImage(64, "lat")
        bd.ddr(lat.name, COLS * lat.colbytes)  # (zeros: per step only its first kb written)
        lv32 = Level(32, 2)
        kv = {16: bd.ddr("kv_self", 16 * Level(16).np * 2 * D.Q),
              32: bd.ddr("kv_self32", Attention(np=lv32.np, q=0, o=0, kv=0, d=80, nbands=lv32.nb).kv_bytes()),
              64: bd.ddr("kv_self64", Attention(np=LV64.np, q=0, o=0, kv=0, d=40, nbands=LV64.nb).kv_bytes())}
        ctx_kv = {p: bd.image(p + "ctx", np.zeros(len(self._ctx(p, torch.zeros(2, CTX_KEYS, 768))), np.uint8))
                  for p in self.PREFIXES}
        zero = {p[:-len("time_emb_proj.weight")]: torch.zeros(w[p].shape[0]) for p in w if p.endswith("time_emb_proj.weight")}
        unet(bd, w, lat, biases=zero, ctx_kv=ctx_kv, kv=kv, eps=eps)
        self.words = bd.finish()
        self.tr = None

    def _ctx(self, p: str, ctx: torch.Tensor) -> np.ndarray:
        a = p + "transformer_blocks.0.attn2."
        wk, wv = self.w[a + "to_k.weight"], self.w[a + "to_v.weight"]
        return L.kv_slots(ctx @ wk.T, ctx @ wv.T, d=wk.shape[0] // 8)

    def set_time(self, biases: dict[str, torch.Tensor]) -> None:
        """The resnets' time biases (by prefix): their conv1 commands' biases (the conv bias + the time bias, bf16) in
        TBIAS, one write."""
        buf = np.empty(self.bd.tb_used // 2, bfloat16)
        for p, off, rows in self.bd.tb_hooks:
            buf[off // 2:off // 2 + rows.size] = (self.w[p + "conv1.bias"] + biases[p]).numpy()[rows]
        self.v2.write(TBIAS, buf)

    def set_context(self, ctx: torch.Tensor, pool: Executor | None = None) -> None:
        """The cross-attentions' K / V^T slots of the text context (2 images, 77, 768) (pool: computed on its threads)."""
        for p, v in zip(self.PREFIXES, (pool.map if pool else map)(lambda p: self._ctx(p, ctx), self.PREFIXES), strict=True):
            self.v2.write(p + "ctx", v)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for c, p in enumerate(latent_panels(x)):
            self.v2.write(self.lat.name, p, c * self.lat.colbytes)
        if self.tr is None:
            self.tr = Transaction(self.v2.dev, self.words)
        self.v2.run(self.tr)
        return noise(self.v2.read("out"))
