"""What the media encoders' loaders share (``MediaEncoder``: Qwen3-VL's, LFM3-VL's, MiniCPM-V's and LFM3-Audio's
``NpuVision`` / ``NpuAudio``): the configuration checks of an mmproj, the load of its networks on a vision device, the
cache of an input's embeddings, and the shared device of several encoders.

An encoder is the mmproj's networks split as ``weights`` (views of the file's tensors), ``accept`` (whether a cached
state has what this load needs), ``build`` (the networks on a device, restored from a cached state or built; their programs'
store: ``program.ProgramStore``) and ``state`` (what a build caches). ``load`` runs them as ``shared_device`` does for
one or several encoders (LFM3-Omni's vision tower and speech networks on one device: one hardware context): the device
restored from the state cache (``device.state_dir``: the modules the encoders name, their imports derived:
``sources``) with its weights packed from the file, else built and cached (``attach``).

Loading is the layers' framework side: ``artlite.vision`` imports no ``models``; a model's ``npu.py`` subclasses
``MediaEncoder`` with its checks, networks and inputs."""

from __future__ import annotations

import collections
import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np

from ..gguf_model import GGUFFile
from ..model_config import ClipConfig
from .device import VisionDevice, load_state, save_state, state_dir
from .program import ProgramStore
from .weights import Source, sources

TOKENS_LIMIT = 16384  # (an image's merged tokens at most: transformers' default bound)


# -- configuration checks of an mmproj (NotImplementedError: what the encoders do not run; ValueError: a bad option)

def _fmt(items: object) -> str:
    return ", ".join(f"{k} {v}" for k, v in items)


def require_dims(path: str, what: str, dims: Mapping[str, tuple[object, object]]) -> None:
    """NotImplementedError unless every dimension of the file's ``what`` is the one the encoder runs: ``dims`` maps a
    dimension's name to (the file's, the encoder's constant); the message names both from the same entries."""
    if any(got != want for got, want in dims.values()):
        raise NotImplementedError(f"{path}: {what} with {_fmt((k, got) for k, (got, _) in dims.items())} (the encoder "
                                  f"runs {_fmt((k, want) for k, (_, want) in dims.items())})")


def require_projector(path: str, projector: str, projectors: tuple[str, ...], name: str) -> None:
    """NotImplementedError unless the mmproj's projector type is one of the encoder's."""
    if projector not in projectors:
        raise NotImplementedError(f"{path}: projector {projector!r} ({name}'s {' / '.join(projectors)} only)")


def require_half_norm(path: str, clip: ClipConfig) -> None:
    """NotImplementedError unless the pixels are normalized by 0.5 / 0.5 (the encoders fold that into their patch
    weights)."""
    if tuple(clip.mean) != (0.5,) * 3 or tuple(clip.std) != (0.5,) * 3:
        raise NotImplementedError(f"{path}: pixels normalized by {clip.mean} / {clip.std} (the encoder folds in 0.5 / "
                                  f"0.5)")


def require_token_bounds(lo: int, hi: int) -> None:
    """ValueError unless 1 <= lo <= hi <= ``TOKENS_LIMIT`` (an image's merged tokens)."""
    if not 1 <= lo <= hi <= TOKENS_LIMIT:
        raise ValueError(f"image tokens {lo} .. {hi}: 1 <= min <= max <= {TOKENS_LIMIT}")


# -- embeddings of the inputs seen

def digest(*arrays: np.ndarray) -> bytes:
    """A key of arrays: their shapes and bytes."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a.shape, np.int64).tobytes() + a.tobytes())
    return h.digest()


class EmbeddingCache:
    """The embeddings of the last ``keep`` inputs by their key (``digest``), so that a conversation's image or clip runs
    the encoder once; the time the encoder ran and the entries (images, slices, clips, pairs) it ran on."""

    def __init__(self, keep: int = 8):
        self.keep = keep
        self._kept: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()
        self.t_encode, self.entries = 0.0, 0

    def get(self, key: bytes, compute: Callable[[], np.ndarray], *, count: int = 1) -> np.ndarray:
        """The embeddings kept under ``key``, else ``compute()`` (an encoder run on ``count`` entries) kept."""
        if key in self._kept:
            self._kept.move_to_end(key)
        else:
            t0 = time.perf_counter()
            self._kept[key] = compute()
            self.add(time.perf_counter() - t0, count)
            if len(self._kept) > self.keep:
                self._kept.popitem(last=False)
        return self._kept[key]

    def add(self, seconds: float, count: int) -> None:
        """An encoder run of ``seconds`` on ``count`` entries not kept here."""
        self.t_encode += seconds
        self.entries += count


# -- loading

def attach(cache: str, srcs: Mapping[str, Source], build: Callable[[VisionDevice, dict | None], list],
           state: Callable[[], dict], *, accept: Callable[[dict], bool] | None = None) -> VisionDevice:
    """A device with the networks ``build(npu, st)`` makes (each with its builder ``bd``): restored from the state
    cache directory ``cache`` (``st``: its state, the weights packed from ``srcs``) if it has one ``accept`` takes, else
    built (``st`` None) and cached (``state()``); warm."""
    npu, st = load_state(cache, srcs, accept=accept) or (VisionDevice(), None)
    nets = build(npu, st)
    if st is None:
        save_state(cache, npu, [n.bd.built() for n in nets], state())
    npu.warm()
    return npu


class MediaEncoder(ABC):
    """The networks of an mmproj file on a vision device (module docstring): ``KIND`` names its cache directory,
    ``MODULES`` the modules whose code and imports build its state."""

    KIND: ClassVar[str]
    MODULES: ClassVar[tuple[str, ...]]

    def __init__(self, path: str):
        self.path = path
        self.gguf = GGUFFile(path)
        self.results = EmbeddingCache()

    @property
    def t_encode(self) -> float:
        """The encoder's time so far."""
        return self.results.t_encode

    @property
    def entries(self) -> int:
        """The entries (images, slices, clips, pairs) run through the encoder so far."""
        return self.results.entries

    @abstractmethod
    def weights(self) -> dict:
        """The networks' weights (views of the mmproj's tensors)."""

    def accept(self, state: dict) -> bool:
        """Whether a cached state (``state()``'s) has what this load needs."""
        return True

    @abstractmethod
    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
        """The networks on a device, in the order their builds allocate: restored from ``state`` (and the programs of
        ``store``) or built; returns them (each with its builder ``bd``)."""

    @abstractmethod
    def state(self) -> dict:
        """The built state to cache (``accept``'s and ``build``'s)."""

    def load(self) -> VisionDevice:
        """The networks on a device of their own (``shared_device``)."""
        return shared_device(self.KIND, self.path, {"net": self})


def shared_device(kind: str, mmproj: str, encoders: Mapping[str, MediaEncoder]) -> VisionDevice:
    """An mmproj's encoders built on one vision overlay (one hardware context: an image and a clip run back to back
    without a switch). Their built state is cached together in ``kind``'s directory, keyed by the mmproj and the
    encoders' modules."""
    w = {n: e.weights() for n, e in encoders.items()}
    cache = state_dir(kind, mmproj, tuple(sorted({m for e in encoders.values() for m in e.MODULES})), [mmproj])
    store = ProgramStore(cache)
    return attach(cache, sources(w),
                  lambda npu, st: [net for n, e in encoders.items() for net in e.build(npu, w[n], st and st[n], store)],
                  lambda: {n: e.state() for n, e in encoders.items()},
                  accept=lambda st: all(e.accept(st[n]) for n, e in encoders.items()))
