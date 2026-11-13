"""Qwen3-VL's images on the NPU for the qwen35 / qwen35moe language models (``NpuVision``): an image at its encoder
size (``host.image_pixels``: between ``min_tokens`` and ``max_tokens`` merged tokens), its embeddings
(``models.qwen3vl.encoder.Encoder``: on the vision overlay in a hardware context of its own) and its tokens
(``host.Grid.sequence``).

Every load packs the weights from the mmproj file (BF16 / F16 / Q8_0 -> BFP16: ``artlite.vision.weights``, the
native packer). The encoder's built state (the program of the smallest images, the layout and commands) is cached
(``artlite.vision.shell``: keyed by the mmproj file and the code): a load restores it, else builds the programs of the
smallest and the largest images (~1.5 s; the largest's allocates every band's DDR images). Another size's program is
built on its first image (0.3-0.8 s, over the images as they are) and kept in the cache directory too. An image's
embeddings are kept by its pixels' digest, so a conversation's image runs the encoder once.
"""

import numpy as np
from PIL import Image

from artlite.model_config import ClipConfig
from artlite.vision.device import VisionDevice
from artlite.vision.program import ProgramStore
from artlite.vision.shell import (
    MediaEncoder,
    digest,
    require_dims,
    require_half_norm,
    require_projector,
    require_token_bounds,
)
from artlite.vision.weights import gguf_getter
from artlite.vision.windows import TOKEN_PX, WIN

from ..sequence import Sequence, load_image
from . import encoder as V
from .host import MAX_TOKENS, MIN_TOKENS, Grid, image_pixels

PROJECTORS = ("qwen3vl_merger",)  # the mmproj projector types (``clip.projector_type``) of the encoder


class NpuVision(MediaEncoder):
    """The vision encoder and projector of an mmproj file (projector ``qwen3vl_merger``) on the NPU; an image's merged
    tokens between ``min_tokens`` and ``max_tokens`` (llama.cpp's ``--image-min-tokens`` / ``--image-max-tokens``;
    default the mmproj's image_min / max_pixels / 1024 where it has them, else 8 and 4096: llama.cpp's)."""

    KIND = "qwen3vl"
    MODULES = ("models.qwen3vl.npu",)
    markup = ("<|vision_start|>", "<|vision_end|>")  # (around an image's tokens: llama.cpp's mtmd, the chat template's)
    encoder: V.Encoder

    def __init__(self, path: str, *, min_tokens: int | None = None, max_tokens: int | None = None):
        super().__init__(path)
        self.clip = c = ClipConfig.from_gguf(self.gguf)
        require_projector(path, c.projector, PROJECTORS, "Qwen3-VL")
        require_dims(path, "a vision encoder", {"patch": (c.patch_size, V.PATCH), "merge": (c.merge, 2),
                                                "deepstack layers": (len(c.deepstack), 0)})
        self.vit = V.tower(c.hidden, c.heads, c.ffn, c.eps)  # (NotImplementedError: widths the overlay does not run)
        require_half_norm(path, c)
        self.min_tokens = min_tokens or c.min_pixels // 1024 or MIN_TOKENS
        self.max_tokens = max_tokens or c.max_pixels // 1024 or MAX_TOKENS
        require_token_bounds(self.min_tokens, self.max_tokens)
        self.load()

    def weights(self) -> dict:
        return V.weights(gguf_getter(self.gguf), self.clip.layers, self.clip.projection_dim, self.vit)

    def accept(self, state: dict) -> bool:
        return state["max_bands"] >= -(-self.max_tokens // WIN)

    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
        pos = gguf_getter(self.gguf)("v.position_embd.weight").numpy()
        if state is not None:
            state = dict(state, programs=dict(state["programs"]))
            store.restore(state["programs"])  # (the programs built after the state)
            self.encoder = V.Encoder(npu, w, pos, max_bands=state["max_bands"], state=state, store=store)
        else:
            self.encoder = V.Encoder(npu, w, pos, max_bands=max(-(-self.max_tokens // WIN), -(-MAX_TOKENS // WIN)),
                                     store=store)
        return [self.encoder]

    def state(self) -> dict:
        return self.encoder.state() | {"max_bands": self.encoder.max_bands}

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, Grid]:
        """An image's embeddings (its merged tokens in raster order, projection_dim) and its token grid."""
        px = image_pixels(im, min_tokens=self.min_tokens, max_tokens=self.max_tokens)
        grid = Grid(px.shape[0] // TOKEN_PX, px.shape[1] // TOKEN_PX)
        return self.results.get(digest(px), lambda: self.encoder(px)), grid

    def sequence(self, image: str | bytes | Image.Image) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens from position 0 (between the
        chat template's ``<|vision_start|>`` and ``<|vision_end|>``)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, grid = self.embeddings(im)
        return grid.sequence(embd)
