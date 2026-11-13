"""LFM3-Audio's speech on the NPU for the lfm3 language models (``NpuAudio``): a clip's samples (``host.load_audio``), its
log-mel spectrogram (``host.log_mel``), its rows' embeddings (``models.lfm3a.encoder.Encoder``: on the vision overlay in
a hardware context of its own, band by band; with LFM3-Omni's mmproj on the one its vision tower shares,
``models.lfm.lfm.media_device``) and its tokens (``host.clip_sequence``); and, where the mmproj holds the speech
generator (projector ``lfm3a_gen``), its depthformer (``models.lfm3a.depth.Depth``) and codec decoder
(``models.lfm3a.codec.Codec``) on the same device (``models.lfm3a.speech``: the frames' loop).

Every load packs the weights from the mmproj file (Q8_0 / BF16 / F16 -> BFP16: ``artlite.vision.weights``, the native
packer). The networks' built state (their programs, the layout and commands) is cached as the vision encoders'
(``artlite.vision.device.state_dir``), keyed by the mmproj file and the code. A clip's embeddings are kept by its
samples' digest, so a conversation's clip runs the encoder once.
"""

import collections
import hashlib
import time

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import AudioConfig, GenConfig
from artlite.vision.device import VisionDevice, load_state, save_state, state_dir
from artlite.vision.weights import gguf_getter, sources

from ..sequence import Sequence
from . import codec as CD
from . import depth as DP
from . import encoder as V
from . import host

# (what the cache's key covers besides the mmproj: the code that builds the images and the programs)
SOURCES = ("artlite/vision", "models/lfm3a/encoder.py", "models/lfm3a/depth.py", "models/lfm3a/codec.py",
            "artlite/txn.py", "artlite/npu_regs.py", "artlite/operators/bfp16.py", "artlite/operators/kmath.py",
            "artlite/operators/fused/ctrlpkt.py", "artlite/operators/paging.py", "artlite/overlay.py",
            "models/lfm3a/npu.py")
KEEP = 8  # clips whose embeddings are kept
PROJECTORS = ("lfm3a",)  # the mmproj audio projector types (``clip.audio.projector_type``) of the encoder
GEN_PROJECTORS = ("lfm3a_gen",)  # (``clip.gen.audio.projector_type``) of the speech generator


def has_audio(path: str) -> bool:
    """Whether an mmproj file holds a speech encoder of a projector this module runs."""
    g = GGUFFile(path)
    return AudioConfig.has_audio(g) and AudioConfig.from_gguf(g).projector in PROJECTORS


class NpuAudio:
    """The speech encoder and projector of an mmproj file (audio projector ``lfm3a``) on the NPU."""

    def __init__(self, path: str, *, load: bool = True):
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
        self.gen = GenConfig.from_gguf(self.gguf) if GenConfig.has_gen(self.gguf) else None
        if self.gen is not None and (self.gen.projector not in GEN_PROJECTORS or (
                self.gen.hidden, self.gen.heads, self.gen.kv_heads, self.gen.ffn, self.gen.layers) != (DP.H, DP.HEADS, DP.KVH,
                                                                                                     DP.FFN_REAL, 3)):
            self.gen = None  # (a generator this module does not run: speech in only)
        self._kept: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()
        self.t_encode, self.clips = 0.0, 0  # (the encoder's time and clips so far)
        self.depth: DP.Depth | None = None
        self.codec: CD.Codec | None = None
        self.encoder: V.Encoder | None = self._load() if load else None  # (else ``build`` on a device of the caller's)

    def weights(self) -> dict:
        """The networks' weights (views of the mmproj's tensors): "enc", and with the generator "depth", "codec"."""
        get = gguf_getter(self.gguf)
        w: dict = {"enc": V.weights(get, V.kinds_of(self.gguf.tensor_info, self.cfg.layers), self.cfg.projection_dim)}
        if self.gen is not None:
            w["depth"] = DP.weights(get, self.gen.layers)
            w["codec"] = CD.weights(get, CD.kinds_of(self.gguf.tensor_info, 8))
        return w

    def accept(self, state: dict) -> bool:
        """Whether a cached state has this file's networks."""
        return ("depth" in state) == (self.gen is not None)

    def build(self, npu: VisionDevice, w: dict, state: dict | None) -> list:
        """The networks on a device (restored from ``state``, or built): the encoder, the depthformer, the codec."""
        g = self.gguf
        self.encoder = V.Encoder(npu, w["enc"], state=state and state["enc"])
        nets: list = [self.encoder]
        if self.gen is not None:
            mask = g.tensor("a.gen.code.mask_embd").dequantize().reshape(-1)
            emb = lambda rows: g.rows("a.gen.code.embd.weight", rows)
            self.depth = DP.Depth(npu, w["depth"], emb, mask, state=state and state["depth"])
            self.codec = CD.Codec(npu, w["codec"], state=state and state["codec"])
            nets += [self.depth, self.codec]
        self._nets = dict(zip(w, nets, strict=True))
        return nets

    def state(self) -> dict:
        """The built state to cache (``accept``'s and ``build``'s)."""
        return {k: n.state() for k, n in self._nets.items()}

    def _load(self) -> V.Encoder:
        w = self.weights()
        cache = state_dir("lfm3a", self.path, SOURCES, [self.path])
        npu, st = load_state(cache, sources(w), accept=self.accept) or (VisionDevice(), None)
        nets = self.build(npu, w, st)
        if st is None:
            save_state(cache, npu, [n.bd.built() for n in nets], self.state())
        npu.warm()
        return self.encoder

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
