"""LFM3-Audio's speech on the NPU for the lfm3 language models (``NpuAudio``): a clip's samples (``host.load_audio``), its
log-mel spectrogram (``host.log_mel``), its rows' embeddings (``models.lfm3a.encoder.Encoder``: on the vision overlay in
a hardware context of its own, band by band) and its tokens (``host.clip_sequence``).

Every load packs the weights from the mmproj file (Q8_0 / BF16 / F16 -> BFP16: ``artlite.vision.weights``, the native
packer). The encoder's built state (its band and carry programs, the layout and commands) is cached as the vision
encoders' (``artlite.vision.device.state_dir``), keyed by the mmproj file and the code. A clip's embeddings are kept by
its samples' digest, so a conversation's clip runs the encoder once.
"""

import collections
import hashlib
import time

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import AudioConfig
from artlite.vision.device import VisionDevice, load_state, save_state, state_dir
from artlite.vision.weights import gguf_getter, sources

from ..sequence import Sequence
from . import encoder as V
from . import host

# (what the cache's key covers besides the mmproj: the code that builds the images and the programs)
_SOURCES = ("artlite/vision", "models/lfm3a/encoder.py", "artlite/txn.py", "artlite/operators/bfp16.py",
            "artlite/operators/kmath.py", "artlite/operators/fused/ctrlpkt.py", "artlite/operators/paging.py",
            "artlite/overlay.py", "models/lfm3a/npu.py")
KEEP = 8  # clips whose embeddings are kept
PROJECTORS = ("lfm3a",)  # the mmproj audio projector types (``clip.audio.projector_type``) of the encoder


def has_audio(path: str) -> bool:
    """Whether an mmproj file holds a speech encoder of a projector this module runs."""
    g = GGUFFile(path)
    return AudioConfig.has_audio(g) and AudioConfig.from_gguf(g).projector in PROJECTORS


class NpuAudio:
    """The speech encoder and projector of an mmproj file (audio projector ``lfm3a``) on the NPU."""

    def __init__(self, path: str):
        self.path = path
        self.gguf = GGUFFile(path)
        self.cfg = c = AudioConfig.from_gguf(self.gguf)
        if c.projector not in PROJECTORS:
            raise NotImplementedError(f"{path}: audio projector {c.projector!r} (LFM3-Audio's lfm3a only)")
        if (c.hidden, c.heads, c.kv_heads, c.ffn, c.layers, c.mel_bins, c.window) != \
                (V.H, V.HEADS, V.KVH, V.FFN_REAL, 17, host.N_MEL, V.WINDOW):
            raise NotImplementedError(f"{path}: a speech encoder of {c.hidden} x {c.heads} heads ({c.kv_heads} KV), FFN "
                                      f"{c.ffn}, {c.layers} layers, {c.mel_bins} mel bins, window {c.window} (the encoder "
                                      f"runs LFM3-Audio's 768 x 8 (4), 1792, 17, 128, 32)")
        self._kept: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()
        self.t_encode, self.clips = 0.0, 0  # (the encoder's time and clips so far)
        self.encoder = self._load()

    def _load(self) -> V.Encoder:
        get = gguf_getter(self.gguf)
        w = V.weights(get, V.kinds_of(self.gguf.tensor_info, self.cfg.layers), self.cfg.projection_dim)
        cache = state_dir("lfm3a", self.path, _SOURCES, [self.path])
        if (got := load_state(cache, sources(w))) is not None:
            npu, st = got
            enc = V.Encoder(npu, w, state=st)
        else:
            npu = VisionDevice()
            enc = V.Encoder(npu, w)
            save_state(cache, npu, [enc.bd.built()], enc.state())
        npu.warm()
        return enc

    def embeddings(self, samples: np.ndarray) -> np.ndarray:
        """A clip's samples (16 kHz mono float32) -> its rows' embeddings (rows, projection_dim)."""
        x = np.ascontiguousarray(samples, np.float32)
        key = hashlib.sha256(x.tobytes()).digest()
        if key in self._kept:
            self._kept.move_to_end(key)
        else:
            t0 = time.perf_counter()
            self._kept[key] = self.encoder(host.log_mel(x))
            self.t_encode += time.perf_counter() - t0
            self.clips += 1
            if len(self._kept) > KEEP:
                self._kept.popitem(last=False)
        return self._kept[key]

    def sequence(self, clip: str | bytes | np.ndarray) -> Sequence:
        """A clip (a file, its bytes or 16 kHz samples) as the language model's tokens."""
        x = clip if isinstance(clip, np.ndarray) else host.load_audio(clip)
        return host.clip_sequence(self.embeddings(x))
