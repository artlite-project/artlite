"""MiniCPM-V 4.7's images on the NPU for its language model (``NpuVision``): an image's entries (``host.slice_image``:
the overview and the slices), their embeddings (``models.minicpmv.encoder.Encoder``: the whole encoder per
dispatch of a pair of entries or a lone one, on the vision overlay in a hardware context of its own) and its tokens
(``host.Grid.sequence``: the embeddings and the mmproj's separator rows at the official positions).

Every load packs the weights from the mmproj file (F16 / BF16 -> BFP16: ``artlite.vision.weights``, the native
packer). The encoder's built state (its programs, layout and commands) is cached (``artlite.vision.shell``), keyed
by the mmproj file and the code: a load restores it, else builds the programs (~3 s). An image's embeddings are kept
by its pixels' digest, so a conversation's image runs the encoder once.
"""

import numpy as np
from ml_dtypes import bfloat16
from PIL import Image

from artlite.model_config import ClipConfig
from artlite.vision.device import VisionDevice
from artlite.vision.program import ProgramStore
from artlite.vision.shell import MediaEncoder, digest, require_projector
from artlite.vision.weights import gguf_getter

from ..sequence import Sequence
from . import encoder as V
from .host import load_image, slice_image

PROJECTORS = ("minicpmv4_7",)  # the mmproj projector types (``clip.projector_type``) of the encoder


class NpuVision(MediaEncoder):
    """The vision encoder and projector of an mmproj file (projector ``minicpmv4_7``: 16x, 64 tokens per slice, or 4x,
    256) on the NPU; ``max_slices``: an image's slices at most (default: the file's)."""

    KIND = "minicpmv"
    MODULES = ("models.minicpmv.npu",)

    def __init__(self, path: str, *, max_slices: int | None = None):
        super().__init__(path)
        self.clip = ClipConfig.from_gguf(self.gguf)
        require_projector(path, self.clip.projector, PROJECTORS, "MiniCPM-V 4.7")
        self.max_slices = max_slices or self.clip.max_slices
        self.sep = self.gguf.rows("v.tok_embd_sep", [0, 1, 2, 3])
        self.encoder: V.Encoder
        self.load()

    def weights(self) -> dict:
        c = self.clip
        return V.weights(gguf_getter(self.gguf), c.layers, c.merger_layer, c.mean, c.std)

    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
        pos = gguf_getter(self.gguf)("v.position_embd.weight").numpy().astype(bfloat16)
        self.encoder = V.Encoder(npu, w, pos, patch=self.clip.patch_size, state=state)
        return [self.encoder]

    def state(self) -> dict:
        return self.encoder.state()

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, object]:
        """An image's embeddings (its entries' tokens in order: the overview's, then the slices' row by row) and its
        token grid (``host.Grid``)."""
        pixels, grid = slice_image(im, self.clip, self.max_slices)
        return self.results.get(digest(*pixels), lambda: np.concatenate(self.encoder(pixels)), count=len(pixels)), grid

    def sequence(self, image: str | bytes | Image.Image, *, newline_after: bool = True) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens from position 0 (``<image>``
        through its last marker; ``newline_after``: the chat template's "\\n" after it, at its official position)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, grid = self.embeddings(im)
        return grid.sequence(embd, self.sep, newline_after=newline_after)
