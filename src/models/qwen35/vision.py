"""MiniCPM-V 4.7's images on the NPU for its language model (``NpuVision``): an image's entries (``minicpmv.slice_image``:
the overview and the slices), their embeddings (``artlite.operators.vision.minicpmv.Encoder``: the whole encoder per
dispatch of a pair of entries or a lone one, on sd15's design in a hardware context of its own) and its tokens
(``minicpmv.Grid.sequence``: the embeddings and the mmproj's separator rows at the official positions).

The encoder's built state (its weight images and transactions) is cached as sd15's (``runtime.state_dir``), keyed by
the mmproj file and the code: a load restores it, else converts the weights (BF16 -> BFP16) and builds the
transactions (~3 s). An image's embeddings are kept by its pixels' digest, so a conversation's image runs the encoder
once.
"""

import collections
import hashlib
import time

import numpy as np
import torch
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig
from artlite.operators.sd15.runtime import Sd15Device, load_state, save_state, state_dir
from artlite.operators.vision import minicpmv as V

from .minicpmv import load_image, slice_image
from .sequence import Sequence

# (what the cache's key covers besides the mmproj: the code that builds the images and the transactions)
_SOURCES = ("artlite/operators/sd15", "artlite/operators/vision", "artlite/txn.py", "artlite/operators/bfp16.py",
            "artlite/operators/act_lut.py", "artlite/operators/fused/ctrlpkt.py", "artlite/operators/paging.py",
            "artlite/operators/overlay/runtime.py", "models/qwen35/vision.py")
KEEP = 8  # images whose embeddings are kept


class NpuVision:
    """The vision encoder and projector of an mmproj file (projector ``minicpmv4_7``: 16x, 64 tokens per slice, or 4x,
    256) on the NPU; ``max_slices``: an image's slices at most (default: the file's)."""

    def __init__(self, path: str, *, max_slices: int | None = None):
        self.path = path
        self.gguf = GGUFFile(path)
        self.clip = ClipConfig.from_gguf(self.gguf)
        if self.clip.projector != "minicpmv4_7":
            raise NotImplementedError(f"{path}: projector {self.clip.projector!r} (MiniCPM-V 4.7's minicpmv4_7 only)")
        self.max_slices = max_slices or self.clip.max_slices
        self.sep = self.gguf.rows("v.tok_embd_sep", [0, 1, 2, 3]).numpy()
        self._kept: collections.OrderedDict[bytes, torch.Tensor] = collections.OrderedDict()
        self.t_encode, self.entries = 0.0, 0  # (the encoder's time and entries so far)
        self.encoder = self._load()

    def _get(self, name: str) -> torch.Tensor:
        t = self.gguf.tensor(name)
        return t.dequantize().reshape(t.rows, t.cols).float()

    def _load(self) -> V.Encoder:
        pos = self._get("v.position_embd.weight").bfloat16()
        cache = state_dir("minicpmv", self.path, _SOURCES, [self.path])
        if (got := load_state(cache)) is not None:
            npu, st = got
            enc = V.Encoder.restored(npu, st, pos, patch=self.clip.patch_size)
        else:
            npu = Sd15Device()
            enc = V.Encoder(npu, V.weights(self._get, self.clip.layers, self.clip.merger_layer), pos,
                            patch=self.clip.patch_size)
            save_state(cache, npu, enc.images, enc.state())
        npu.warm()
        return enc

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, object]:
        """An image's embeddings (its entries' tokens in order: the overview's, then the slices' row by row) and its
        token grid (``minicpmv.Grid``)."""
        pixels, grid = slice_image(im, self.clip, self.max_slices)
        h = hashlib.sha256()
        for p in pixels:
            h.update(np.asarray(p.shape, np.int64).tobytes() + p.tobytes())
        key = h.digest()
        if key in self._kept:
            self._kept.move_to_end(key)
        else:
            t0 = time.perf_counter()
            self._kept[key] = torch.cat(self.encoder(pixels))
            self.t_encode += time.perf_counter() - t0
            self.entries += len(pixels)
            if len(self._kept) > KEEP:
                self._kept.popitem(last=False)
        return self._kept[key].numpy(), grid

    def sequence(self, image: str | bytes | Image.Image, *, newline_after: bool = True) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens from position 0 (``<image>``
        through its last marker; ``newline_after``: the chat template's "\\n" after it, at its official position)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, grid = self.embeddings(im)
        return grid.sequence(embd, self.sep, newline_after=newline_after)
