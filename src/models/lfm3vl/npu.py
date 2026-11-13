"""LFM3-VL's images on the NPU for the lfm3 language models (``NpuVision``): an image at its encoder size
(``host.image_pixels``: between ``min_tokens`` and ``max_tokens`` merged tokens), its embeddings
(``models.lfm3vl.encoder.Encoder``: on the vision overlay in a hardware context of its own; with LFM3-Omni's mmproj on
the one its speech networks share, ``models.lfm.lfm.media_device``) and its tokens
(``host.Grid.sequence``).

A video's frames go through in pairs (``pairs``: a program per size, as an image's).

Every load packs the weights from the mmproj file (BF16 / F16 / Q8_0 -> BFP16: ``artlite.vision.weights``, the native
packer). The encoder's built state (the programs of the smallest and the largest images, the layout and commands) is
cached as Qwen3-VL's (``artlite.vision.device.state_dir``), keyed by the mmproj file and the code; another size's
program is built on its first image and kept in the cache directory too. An image's embeddings are kept by its
pixels' digest, so a conversation's image runs the encoder once.
"""

import collections
import glob
import hashlib
import os
import pickle
import time
from collections.abc import Iterable, Iterator

import numpy as np
from PIL import Image

from artlite.gguf_model import GGUFFile
from artlite.model_config import ClipConfig
from artlite.vision.device import VisionDevice, load_state, save_state, state_dir
from artlite.vision.weights import gguf_getter, sources
from artlite.vision.windows import GW_MAX, WIN, Piece

from ..sequence import Sequence, load_image
from . import encoder as V
from .host import MAX_TOKENS, Grid, image_pixels, token_bounds

# (what the cache's key covers besides the mmproj: the code that builds the images and the programs)
SOURCES = ("artlite/vision", "models/lfm3vl/encoder.py", "artlite/txn.py", "artlite/operators/bfp16.py",
            "artlite/operators/kmath.py", "artlite/operators/fused/ctrlpkt.py", "artlite/operators/paging.py",
            "artlite/overlay.py", "models/lfm3vl/npu.py")
KEEP = 8  # images whose embeddings are kept
PROJECTORS = ("lfm3vl",)  # the mmproj projector types (``clip.projector_type``) of the encoder
TOKENS_LIMIT = 16384  # (an image's merged tokens at most: transformers' default bound)


def _save_program(cache: str, nb: int | tuple[str, int], pieces: list[Piece], same_layout: bool) -> None:
    """A program that takes only the cached state's images (same_layout: ``WindowEncoder.state``) into its cache
    directory (nb: its key, ("pair", nb) a video pair's)."""
    if not same_layout or not os.path.isdir(cache):
        return
    path = os.path.join(cache, f"program_{nb if isinstance(nb, int) else '_'.join(map(str, nb))}.pkl")
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump((nb, pieces), f)
    os.replace(tmp, path)


class NpuVision:
    """The vision encoder and projector of an mmproj file (projector ``lfm3vl``) on the NPU; an image's merged tokens
    between ``min_tokens`` and ``max_tokens`` (llama.cpp's ``--image-min-tokens`` / ``--image-max-tokens``; default the
    mmproj's bounds, its ``image_min_pixels`` / ``image_max_pixels`` over 1024, the maximum at most llama.cpp's 4096:
    LFM3-VL-2B 64 .. 4096, LFM3-Omni-2B 64 .. 2048)."""

    markup = ("<|vision_start|>", "<|vision_end|>")  # (around an image's tokens: llama.cpp's mtmd, the chat template's)

    def __init__(self, path: str, *, min_tokens: int | None = None, max_tokens: int | None = None, load: bool = True):
        self.path = path
        self.gguf = GGUFFile(path)
        self.clip = c = ClipConfig.from_gguf(self.gguf)
        if c.projector not in PROJECTORS:
            raise NotImplementedError(f"{path}: projector {c.projector!r} (LFM3-VL's lfm3vl only)")
        if (c.hidden, c.heads, c.kv_heads, c.ffn, c.patch_size, c.merge) != (V.H, V.HEADS, V.KVH, V.FFN, V.PATCH, 2):
            raise NotImplementedError(f"{path}: a vision encoder of {c.hidden} x {c.heads} heads ({c.kv_heads} KV), FFN "
                                      f"{c.ffn}, patch {c.patch_size}, merge {c.merge} (the encoder runs LFM3-VL's 1024 x "
                                      f"16 (8), 2560, 16, 2)")
        if tuple(c.mean) != (0.5,) * 3 or tuple(c.std) != (0.5,) * 3:
            raise NotImplementedError(f"{path}: pixels normalized by {c.mean} / {c.std} (the encoder folds in 0.5 / 0.5)")
        self.min_tokens, self.max_tokens = token_bounds(c.min_pixels, c.max_pixels, min_tokens, max_tokens)
        if not 1 <= self.min_tokens <= self.max_tokens <= TOKENS_LIMIT:
            raise ValueError(f"image tokens {self.min_tokens} .. {self.max_tokens}: 1 <= min <= max <= {TOKENS_LIMIT}")
        self._kept: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()
        self.t_encode, self.entries = 0.0, 0  # (the encoder's time and images so far)
        self.encoder: V.Encoder | None = self._load() if load else None  # (else ``build`` on a device of the caller's)

    def weights(self) -> dict:
        """The encoder's weights (views of the mmproj's tensors)."""
        return V.weights(gguf_getter(self.gguf), V.kinds_of(self.gguf.tensor_info, self.clip.layers),
                         self.clip.projection_dim)

    def bands(self) -> int:
        """The encoder's largest level: its largest images' and ``MAX_TOKENS``' (a video's pairs: at most that)."""
        return max(-(-n // WIN) for n in (self.max_tokens, MAX_TOKENS))

    def accept(self, state: dict) -> bool:
        """Whether a cached state's encoder takes this one's largest images."""
        return state["max_bands"] >= self.bands()

    def build(self, npu: VisionDevice, w: dict, state: dict | None, cache: str) -> V.Encoder:
        """The encoder on a device (restored from ``state`` and the programs built after it in ``cache``, or built);
        the programs of other sizes kept in ``cache`` as they are built."""
        if state is not None:
            state = dict(state, programs=dict(state["programs"]))
            for f in glob.glob(os.path.join(cache, "program_*.pkl")):  # (the programs built after the state)
                with open(f, "rb") as fh:
                    nb, pieces = pickle.load(fh)
                state["programs"][nb] = pieces
            enc = V.Encoder(npu, w, max_bands=state["max_bands"], state=state)
        else:
            enc = V.Encoder(npu, w, max_bands=self.bands())
        enc.built = lambda nb, pieces, same: _save_program(cache, nb, pieces, same)
        self.encoder = enc
        return enc

    def state(self) -> dict:
        """The built state to cache (``accept``'s and ``build``'s)."""
        return self.encoder.state() | {"max_bands": self.encoder.max_bands}

    def _load(self) -> V.Encoder:
        w = self.weights()
        cache = state_dir("lfm3vl", self.path, SOURCES, [self.path])
        npu, st = load_state(cache, sources(w), accept=self.accept) or (VisionDevice(), None)
        enc = self.build(npu, w, st, cache)
        if st is None:
            save_state(cache, npu, [enc.bd.built()], self.state())
        npu.warm()
        return enc

    def embeddings(self, im: Image.Image) -> tuple[np.ndarray, Grid]:
        """An image's embeddings (its merged tokens in raster order, projection_dim) and its token grid."""
        px = image_pixels(im, min_tokens=self.min_tokens, max_tokens=self.max_tokens)
        grid = Grid(px.shape[0] // 32, px.shape[1] // 32)
        if grid.gw > GW_MAX:
            raise ValueError(f"an image of {grid.gw} tokens per row (at most {GW_MAX})")
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

    def pairs(self, frames: Iterable[np.ndarray]) -> list[np.ndarray]:
        """A video's frames (H, W, 3) uint8 at its size (``host.VideoPlan``; an array of them, or as they are decoded:
        ``models.video.Frames``) -> each pair's embeddings (its merged tokens in raster order): two frames a temporal
        patch, an odd last one alone (both kernels on it, as an image). Pairs encode as soon as they are there: two at a
        time where they run as one (``Encoder.packs``: small pairs, a long video's), a lone last one with itself."""
        keep: list[int] = []  # (per run the pairs whose embeddings it gives: a lone last pair runs with itself)

        def runs() -> Iterator[tuple[np.ndarray, bool]]:
            held, ready, packs = [], [], False
            for i, f in enumerate(frames):
                if f.shape[1] // 32 > GW_MAX:
                    raise ValueError(f"a video of {f.shape[1] // 32} tokens per row (at most {GW_MAX})")
                if i == 0:
                    packs = self.encoder.packs(f.shape[0] // 32, f.shape[1] // 32)
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
        self.t_encode += time.perf_counter() - t0
        self.entries += len(out)
        return out

    def sequence(self, image: str | bytes | Image.Image) -> Sequence:
        """An image (a file, its bytes or a PIL image) as the language model's tokens (between the chat template's
        ``<|vision_start|>`` and ``<|vision_end|>``)."""
        im = image if isinstance(image, Image.Image) else load_image(image)
        embd, grid = self.embeddings(im)
        return grid.sequence(embd)
