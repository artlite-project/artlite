"""Qwen3-VL's images on the NPU for the qwen35 / qwen35moe language models (``NpuVision``): an image at its encoder
size (``host.image_pixels``: between ``min_tokens`` and ``max_tokens`` merged tokens), its embeddings
(``models.qwen3vl.encoder.Encoder``: on the vision overlay in a hardware context of its own) and its tokens
(``host.Grid.sequence``).

Every load packs the weights from the mmproj file (BF16 / F16 / Q8_0 -> BFP16: ``artlite.vision.weights``, the
native packer). The encoder's built state (the program of the smallest images, the layout and commands) is cached as
MiniCPM-V's (``artlite.vision.device.state_dir``), keyed by the mmproj file and the code: a load restores it, else
builds the programs of the smallest and the largest images (~1.5 s; the largest's allocates every band's DDR images).
Another size's program is built on its first image (0.3-0.8 s, over the images as they are) and kept in the cache
directory too. An image's embeddings are kept by its pixels' digest, so a conversation's image runs the encoder once.
"""

import collections
import glob
import hashlib
import os
import pickle
import time

import numpy as np
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig
from artlite.vision.device import VisionDevice, load_state, save_state, state_dir
from artlite.vision.weights import gguf_getter, sources
from artlite.vision.windows import WIN, Piece

from ..sequence import Sequence, load_image
from . import encoder as V
from .host import MAX_TOKENS, MIN_TOKENS, Grid, image_pixels

# (what the cache's key covers besides the mmproj: the code that builds the images and the programs)
_SOURCES = ("artlite/vision", "models/qwen3vl/encoder.py", "artlite/txn.py", "artlite/operators/bfp16.py",
            "artlite/operators/kmath.py", "artlite/operators/fused/ctrlpkt.py", "artlite/operators/paging.py",
            "artlite/overlay.py", "models/qwen3vl/npu.py")
KEEP = 8  # images whose embeddings are kept
PROJECTORS = ("qwen3vl_merger",)  # the mmproj projector types (``clip.projector_type``) of the encoder
TOKENS_LIMIT = 16384  # (an image's merged tokens at most: transformers' default bound)


def _save_program(cache: str, nb: int, pieces: list[Piece], same_layout: bool) -> None:
    """A program that takes only the cached state's images (same_layout: ``WindowEncoder.state``) into its cache
    directory."""
    if not same_layout or not os.path.isdir(cache):
        return
    path = os.path.join(cache, f"program_{nb}.pkl")
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump((nb, pieces), f)
    os.replace(tmp, path)


class NpuVision:
    """The vision encoder and projector of an mmproj file (projector ``qwen3vl_merger``) on the NPU; an image's merged
    tokens between ``min_tokens`` and ``max_tokens`` (llama.cpp's ``--image-min-tokens`` / ``--image-max-tokens``;
    default the mmproj's image_min / max_pixels / 1024 where it has them, else 8 and 4096: llama.cpp's)."""

    markup = ("<|vision_start|>", "<|vision_end|>")  # (around an image's tokens: llama.cpp's mtmd, the chat template's)

    def __init__(self, path: str, *, min_tokens: int | None = None, max_tokens: int | None = None):
        self.path = path
        self.gguf = GGUFFile(path)
        self.clip = c = ClipConfig.from_gguf(self.gguf)
        if c.projector not in PROJECTORS:
            raise NotImplementedError(f"{path}: projector {c.projector!r} (Qwen3-VL's qwen3vl_merger only)")
        if (c.patch_size, c.merge) != (V.PATCH, 2) or c.deepstack:
            raise NotImplementedError(f"{path}: patch {c.patch_size}, merge {c.merge}, deepstack layers "
                                      f"{list(c.deepstack)} (the encoder runs patches of 16, merge 2, no deepstack)")
        self.vit = V.tower(c.hidden, c.heads, c.ffn, c.eps)  # (NotImplementedError: widths the overlay does not run)
        if tuple(c.mean) != (0.5,) * 3 or tuple(c.std) != (0.5,) * 3:
            raise NotImplementedError(f"{path}: pixels normalized by {c.mean} / {c.std} (the encoder folds in 0.5 / 0.5)")
        self.min_tokens = min_tokens or c.min_pixels // 1024 or MIN_TOKENS
        self.max_tokens = max_tokens or c.max_pixels // 1024 or MAX_TOKENS
        if not 1 <= self.min_tokens <= self.max_tokens <= TOKENS_LIMIT:
            raise ValueError(f"image tokens {self.min_tokens} .. {self.max_tokens}: 1 <= min <= max <= {TOKENS_LIMIT}")
        self._kept: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()
        self.t_encode, self.entries = 0.0, 0  # (the encoder's time and images so far)
        self.encoder = self._load()

    def _load(self) -> V.Encoder:
        get = gguf_getter(self.gguf)
        w = V.weights(get, self.clip.layers, self.clip.projection_dim, self.vit)
        pos = get("v.position_embd.weight").numpy()
        bands = -(-self.max_tokens // WIN)
        cache = state_dir("qwen3vl", self.path, _SOURCES, [self.path])
        if (got := load_state(cache, sources(w), accept=lambda st: st["max_bands"] >= bands)) is not None:
            npu, st = got
            for f in glob.glob(os.path.join(cache, "program_*.pkl")):  # (the programs built after the state)
                with open(f, "rb") as fh:
                    nb, pieces = pickle.load(fh)
                st["programs"][nb] = pieces
            enc = V.Encoder(npu, w, pos, max_bands=st["max_bands"], state=st)
        else:
            npu = VisionDevice()
            enc = V.Encoder(npu, w, pos, max_bands=max(bands, -(-MAX_TOKENS // WIN)))
            save_state(cache, npu, [enc.bd.built()], enc.state() | {"max_bands": enc.max_bands})
        enc.built = lambda nb, pieces, same: _save_program(cache, nb, pieces, same)
        npu.warm()
        return enc

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, Grid]:
        """An image's embeddings (its merged tokens in raster order, projection_dim) and its token grid."""
        px = image_pixels(im, min_tokens=self.min_tokens, max_tokens=self.max_tokens)
        grid = Grid(px.shape[0] // 32, px.shape[1] // 32)
        key = hashlib.sha256(np.asarray(px.shape, np.int64).tobytes() + px.tobytes()).digest()
        if key in self._kept:
            self._kept.move_to_end(key)
        else:
            t0 = time.perf_counter()
            self._kept[key] = self.encoder(px)
            self.t_encode += time.perf_counter() - t0
            self.entries += 1
            if len(self._kept) > KEEP:
                self._kept.popitem(last=False)
        return self._kept[key], grid

    def sequence(self, image: str | bytes | Image.Image) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens from position 0 (between the
        chat template's ``<|vision_start|>`` and ``<|vision_end|>``)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, grid = self.embeddings(im)
        return grid.sequence(embd)
