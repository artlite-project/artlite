"""LFM3-VL's images on the NPU for the lfm3 language models (``NpuVision``): an image at its encoder size
(``host.image_pixels``: between ``min_tokens`` and ``max_tokens`` merged tokens), its embeddings
(``models.lfm3vl.encoder.Encoder``: on the vision overlay in a hardware context of its own; with LFM3-Omni's mmproj on
the one its speech networks share, ``artlite.vision.shell.shared_device``) and its tokens
(``host.Grid.sequence``).

A video's frames go through in pairs (``pairs``: a program per size, as an image's).

Every load packs the weights from the mmproj file (BF16 / F16 / Q8_0 -> BFP16: ``artlite.vision.weights``, the native
packer). The encoder's built state (the programs of the smallest and the largest images, the layout and commands) is
cached as Qwen3-VL's (``artlite.vision.shell``), keyed by the mmproj file and the code; another size's program is built
on its first image and kept in the cache directory too. An image's embeddings are kept by its pixels' digest, so a
conversation's image runs the encoder once.
"""

import time
from collections.abc import Iterable, Iterator

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
from artlite.vision.vit import kinds_of
from artlite.vision.weights import gguf_getter
from artlite.vision.windows import GW_MAX, WIN

from ..images import TOKEN_PX
from ..sequence import Sequence, load_image
from . import encoder as V
from .host import MAX_TOKENS, Grid, image_pixels, token_bounds

PROJECTORS = ("lfm3vl",)  # the mmproj projector types (``clip.projector_type``) of the encoder


class NpuVision(MediaEncoder):
    """The vision encoder and projector of an mmproj file (projector ``lfm3vl``) on the NPU; an image's merged tokens
    between ``min_tokens`` and ``max_tokens`` (llama.cpp's ``--image-min-tokens`` / ``--image-max-tokens``; default the
    mmproj's bounds, its ``image_min_pixels`` / ``image_max_pixels`` over 1024, the maximum at most llama.cpp's 4096:
    LFM3-VL-2B 64 .. 4096, LFM3-Omni-2B 64 .. 2048)."""

    KIND = "lfm3vl"
    MODULES = ("models.lfm3vl.npu",)
    markup = ("<|vision_start|>", "<|vision_end|>")  # (around an image's tokens: llama.cpp's mtmd, the chat template's)

    def __init__(self, path: str, *, min_tokens: int | None = None, max_tokens: int | None = None, load: bool = True):
        super().__init__(path)
        self.clip = c = ClipConfig.from_gguf(self.gguf)
        require_projector(path, c.projector, PROJECTORS, "LFM3-VL")
        require_dims(path, "a vision encoder", {"hidden": (c.hidden, V.H), "heads": (c.heads, V.HEADS),
                                                "KV heads": (c.kv_heads, V.KVH), "FFN": (c.ffn, V.FFN),
                                                "patch": (c.patch_size, V.PATCH), "merge": (c.merge, 2)})
        require_half_norm(path, c)
        self.min_tokens, self.max_tokens = token_bounds(c.min_pixels, c.max_pixels, min_tokens, max_tokens)
        require_token_bounds(self.min_tokens, self.max_tokens)
        self.encoder: V.Encoder | None = None
        if load:  # (else ``build`` on a device of the caller's: ``shared_device``)
            self.load()

    def weights(self) -> dict:
        return V.weights(gguf_getter(self.gguf), kinds_of(self.gguf.tensor_info, self.clip.layers, V.CONV_PROBE),
                         self.clip.projection_dim)

    def bands(self) -> int:
        """The encoder's largest level: its largest images' and ``MAX_TOKENS``' (a video's pairs: at most that)."""
        return max(-(-n // WIN) for n in (self.max_tokens, MAX_TOKENS))

    def accept(self, state: dict) -> bool:
        return state["max_bands"] >= self.bands()

    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
        if state is not None:
            state = dict(state, programs=dict(state["programs"]))
            store.restore(state["programs"])  # (the programs built after the state)
            self.encoder = V.Encoder(npu, w, max_bands=state["max_bands"], state=state, store=store)
        else:
            self.encoder = V.Encoder(npu, w, max_bands=self.bands(), store=store)
        return [self.encoder]

    def state(self) -> dict:
        return self.encoder.state() | {"max_bands": self.encoder.max_bands}

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, Grid]:
        """An image's embeddings (its merged tokens in raster order, projection_dim) and its token grid."""
        px = image_pixels(im, min_tokens=self.min_tokens, max_tokens=self.max_tokens)
        grid = Grid(px.shape[0] // TOKEN_PX, px.shape[1] // TOKEN_PX)
        if grid.gw > GW_MAX:
            raise ValueError(f"an image of {grid.gw} tokens per row (at most {GW_MAX})")
        return self.results.get(digest(px), lambda: self.encoder(px)), grid

    def pairs(self, frames: Iterable[np.ndarray]) -> list[np.ndarray]:
        """A video's frames (H, W, 3) uint8 at its size (``host.VideoPlan``; an array of them, or as they are decoded:
        ``models.video.Frames``) -> each pair's embeddings (its merged tokens in raster order): two frames a temporal
        patch, an odd last one alone (both kernels on it, as an image). Pairs encode as soon as they are there: two at a
        time where they run as one (``Encoder.packs``: small pairs, a long video's), a lone last one with itself."""
        keep: list[int] = []  # (per run the pairs whose embeddings it gives: a lone last pair runs with itself)

        def runs() -> Iterator[tuple[np.ndarray, bool]]:
            held, ready, packs = [], [], False
            for i, f in enumerate(frames):
                if f.shape[1] // TOKEN_PX > GW_MAX:
                    raise ValueError(f"a video of {f.shape[1] // TOKEN_PX} tokens per row (at most {GW_MAX})")
                if i == 0:
                    packs = self.encoder.packs(f.shape[0] // TOKEN_PX, f.shape[1] // TOKEN_PX)
                held.append(f)
                if len(held) == 2:
                    ready.append(np.stack(held))
                    held = []
                if len(ready) == 1 + packs:
                    keep.append(len(ready))
                    yield (np.stack(ready), True) if packs else (ready[0], False)
                    ready = []
            if ready:  # (packs: a lone pair)
                keep.append(1)
                yield np.stack(ready * 2), True
            if held:  # (the odd last frame: an image)
                keep.append(1)
                yield held[0], False
        out = []
        t0 = time.perf_counter()
        for emb in self.encoder.stream(runs()):
            n = keep.pop(0)
            out.extend(emb[:n] if emb.ndim == 3 else [emb])
        self.results.add(time.perf_counter() - t0, len(out))
        return out

    def sequence(self, image: str | bytes | Image.Image) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens (between the chat template's
        ``<|vision_start|>`` and ``<|vision_end|>``)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, grid = self.embeddings(im)
        return grid.sequence(embd)
