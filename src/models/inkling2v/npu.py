"""Inkling 2's images on the NPU for its language model (``NpuVision``): an image at its encoder size
(``host.image_pixels``: as it is, at most ``max_tokens`` merged tokens once padded), its embeddings
(``models.inkling2v.encoder.Encoder``: on the vision overlay in a hardware context of its own) and its tokens
(``host.grid``'s ``Grid.sequence``).

Every load packs the weights from the mmproj file (F16 / BF16 / F32 -> BFP16: ``artlite.vision.weights``, the native
packer; the layer scales folded into their GEMMs' rows as it packs them). The encoder's built state (the programs of
the smallest and the largest images, the layout and commands) is cached (``artlite.vision.shell``: keyed by the mmproj
file and the code): a load restores it, else builds them. Another size's program is built on its first image and kept
in the cache directory too. An image's embeddings are kept by its pixels' digest, so a conversation's image runs the
encoder once.
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
from artlite.vision.windows import WIN

from ..images import Grid
from ..sequence import Sequence, load_image
from . import encoder as V
from .host import MAX_TOKENS, MERGE, PATCH, grid, image_pixels

PROJECTORS = ("inkling2v",)  # the mmproj projector types (``clip.projector_type``) of the encoder


def bands(tokens: int) -> int:
    """The bands of an image of ``tokens`` merged tokens (its class token one more slot: ``windows.Layout.cls``)."""
    return -(-(tokens + 1) // WIN)


class NpuVision(MediaEncoder):
    """The vision encoder and projector of an mmproj file (projector ``inkling2v``) on the NPU; an image of at most
    ``max_tokens`` merged tokens (llama.cpp's ``--image-max-tokens``; default 4096, llama.cpp's), larger ones resized
    (``host.target_size``)."""

    KIND = "inkling2v"
    MODULES = ("models.inkling2v.npu",)
    markup = ("<|content_image|>", "")  # (before an image's tokens: llama.cpp's mtmd, the chat template's)
    encoder: V.Encoder

    def __init__(self, path: str, *, max_tokens: int | None = None):
        super().__init__(path)
        self.clip = c = ClipConfig.from_gguf(self.gguf)
        require_projector(path, c.projector, PROJECTORS, "Inkling 2")
        require_dims(path, "a vision encoder", {"patch": (c.patch_size, PATCH), "merge": (c.merge, MERGE)})
        self.vit = V.tower(c.hidden, c.heads, c.ffn, c.eps)  # (NotImplementedError: widths the overlay does not run)
        require_half_norm(path, c)
        self.max_tokens = max_tokens or MAX_TOKENS
        require_token_bounds(1, self.max_tokens)
        self.load()

    def weights(self) -> dict:
        return V.weights(gguf_getter(self.gguf), self.clip.layers, self.clip.projection_dim, self.vit)

    def accept(self, state: dict) -> bool:
        return state["max_bands"] >= bands(self.max_tokens)

    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
        if state is not None:
            state = dict(state, programs=dict(state["programs"]))
            store.restore(state["programs"])  # (the programs built after the state)
            self.encoder = V.Encoder(npu, w, max_bands=state["max_bands"], state=state, store=store)
        else:
            self.encoder = V.Encoder(npu, w, max_bands=max(bands(self.max_tokens), bands(MAX_TOKENS)), store=store)
        return [self.encoder]

    def state(self) -> dict:
        return self.encoder.state() | {"max_bands": self.encoder.max_bands}

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, Grid]:
        """An image's embeddings (its merged tokens in raster order, projection_dim) and its token grid."""
        px = image_pixels(im, max_tokens=self.max_tokens)
        return self.results.get(digest(px), lambda: self.encoder(px)), grid(px)

    def sequence(self, image: str | bytes | Image.Image) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens from position 0 (after the chat
        template's ``<|content_image|>``)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, g = self.embeddings(im)
        return g.sequence(embd)
