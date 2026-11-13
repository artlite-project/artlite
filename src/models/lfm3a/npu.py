"""LFM3-Audio's speech on the NPU for the lfm3 language models (``NpuAudio``): a clip's samples (``host.load_audio``), its
log-mel spectrogram (``host.log_mel``), its rows' embeddings (``models.lfm3a.encoder.Encoder``: on the vision overlay in
a hardware context of its own, band by band; with LFM3-Omni's mmproj on the one its vision tower shares,
``artlite.vision.shell.shared_device``) and its tokens (``host.clip_sequence``); and, where the mmproj holds the speech
generator (projector ``lfm3a_gen``), its depthformer (``models.lfm3a.depth.Depth``) and codec decoder
(``models.lfm3a.codec.Codec``) on the same device (``models.lfm3a.speech``: the frames' loop).

Every load packs the weights from the mmproj file (Q8_0 / BF16 / F16 -> BFP16: ``artlite.vision.weights``, the native
packer). The networks' built state (their programs, the layout and commands) is cached as the vision encoders'
(``artlite.vision.shell``), keyed by the mmproj file and the code. A clip's embeddings are kept by its
samples' digest, so a conversation's clip runs the encoder once.
"""

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import AudioConfig, GenConfig
from artlite.vision.device import VisionDevice
from artlite.vision.program import ProgramStore
from artlite.vision.shell import MediaEncoder, digest, require_dims, require_projector
from artlite.vision.vit import kinds_of
from artlite.vision.weights import gguf_getter

from ..sequence import Sequence
from . import codec as CD
from . import depth as DP
from . import encoder as V
from . import host

PROJECTORS = ("lfm3a",)  # the mmproj audio projector types (``clip.audio.projector_type``) of the encoder
GEN_PROJECTORS = ("lfm3a_gen",)  # (``clip.gen.audio.projector_type``) of the speech generator


def has_audio(path: str) -> bool:
    """Whether an mmproj file holds a speech encoder of a projector this module runs."""
    g = GGUFFile(path)
    return AudioConfig.has_audio(g) and AudioConfig.from_gguf(g).projector in PROJECTORS


class NpuAudio(MediaEncoder):
    """The speech encoder and projector of an mmproj file (audio projector ``lfm3a``) on the NPU."""

    KIND = "lfm3a"
    MODULES = ("models.lfm3a.npu",)

    def __init__(self, path: str, *, load: bool = True):
        super().__init__(path)
        self.cfg = c = AudioConfig.from_gguf(self.gguf)
        require_projector(path, c.projector, PROJECTORS, "LFM3-Audio")
        require_dims(path, "a speech encoder", {"hidden": (c.hidden, V.H), "heads": (c.heads, V.HEADS),
                                                "KV heads": (c.kv_heads, V.KVH), "FFN": (c.ffn, V.FFN_REAL),
                                                "layers": (c.layers, V.LAYERS), "mel bins": (c.mel_bins, host.N_MEL),
                                                "window": (c.window, V.WINDOW)})
        self.gen = GenConfig.from_gguf(self.gguf) if GenConfig.has_gen(self.gguf) else None
        if self.gen is not None and (self.gen.projector not in GEN_PROJECTORS or (
                self.gen.hidden, self.gen.heads, self.gen.kv_heads, self.gen.ffn, self.gen.layers) != (
                    DP.H, DP.HEADS, DP.KVH, DP.FFN_REAL, DP.LAYERS)):
            self.gen = None  # (a generator this module does not run: speech in only)
        self.depth: DP.Depth | None = None
        self.codec: CD.Codec | None = None
        self.encoder: V.Encoder | None = None
        if load:  # (else ``build`` on a device of the caller's: ``shared_device``)
            self.load()

    def weights(self) -> dict:
        """The networks' weights (views of the mmproj's tensors): "enc", and with the generator "depth", "codec"."""
        get = gguf_getter(self.gguf)
        w: dict = {"enc": V.weights(get, kinds_of(self.gguf.tensor_info, self.cfg.layers, V.CONV_PROBE),
                                    self.cfg.projection_dim)}
        if self.gen is not None:
            w["depth"] = DP.weights(get, self.gen.layers)
            w["codec"] = CD.weights(get, kinds_of(self.gguf.tensor_info, CD.LAYERS, CD.CONV_PROBE))
        return w

    def accept(self, state: dict) -> bool:
        return ("depth" in state) == (self.gen is not None)

    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
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
        return {k: n.state() for k, n in self._nets.items()}

    def embeddings(self, samples: np.ndarray) -> np.ndarray:
        """A clip's samples (16 kHz mono float32) -> its rows' embeddings (rows, projection_dim)."""
        x = np.ascontiguousarray(samples, np.float32)
        return self.results.get(digest(x), lambda: self.encoder(host.log_mel(x)))

    def sequence(self, clip: str | bytes | np.ndarray) -> Sequence:
        """A clip (a file, its bytes or 16 kHz samples) as the language model's tokens."""
        x = clip if isinstance(clip, np.ndarray) else host.load_audio(clip)
        return host.clip_sequence(self.embeddings(x))
