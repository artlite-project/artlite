"""MOSS-Transcribe-Diarize's speech on the NPU for its language model (``NpuAudio``): a recording's samples
(``models.audio.load_audio``), its chunks' log-mel spectrograms (``host.log_mel``) and its tokens' rows
(``models.moss.encoder.Encoder``: on the vision overlay in a hardware context of its own, a dispatch a chunk).

Every load packs the weights from the model's files (BF16 / F16 / F32 / Q8_0 -> BFP16: ``artlite.vision.weights``, the
native packer; a transcribe.cpp file's K-quant matrices dequantized on the host first: 4 to 6 bits a weight in the
file, 9 on the overlay). The tower's built state (its program, the layout and commands) is cached as the vision encoders'
(``artlite.vision.shell``), keyed by the files and the code. A recording's rows are kept by its samples' digest, so a
conversation's clip runs the encoder once.
"""

import numpy as np

from artlite.model_config import AudioConfig
from artlite.vision.device import VisionDevice
from artlite.vision.program import ProgramStore
from artlite.vision.shell import EmbeddingCache, MediaEncoder, digest, require_dims, require_projector
from artlite.vision.weights import gguf_getter

from ..audio import load_audio
from . import encoder as V
from . import host

PROJECTORS = ("moss",)  # the audio projector types (``clip.audio.projector_type``) of the tower


class NpuAudio(MediaEncoder):
    """The audio tower of a model's files (audio projector ``moss``: a checkpoint directory's, a transcribe.cpp file's)
    on the NPU."""

    KIND = "moss"
    MODULES = ("models.moss.npu",)
    encoder: V.Encoder

    def __init__(self, path: str):
        super().__init__(path)
        self.cfg = c = AudioConfig.from_gguf(self.gguf)
        require_projector(path, c.projector, PROJECTORS, "MOSS-Transcribe-Diarize")
        meta = self.gguf.meta
        require_dims(path, "an audio tower", {
            "mel bins": (c.mel_bins, host.N_MEL), "rows": (int(meta["clip.audio.max_pos_emb"]), host.ROWS),
            "rows a token": (int(meta["clip.audio.projector.stack_factor"]), V.MERGE),
            "activation": (meta["clip.audio.activation"], "gelu")})
        self.vit = V.tower(c.hidden, c.heads, c.ffn, c.eps)  # (NotImplementedError: widths the overlay does not run)
        self.results = EmbeddingCache(keep=2)  # (a long recording's rows are large)
        self.load()

    def weights(self) -> dict:
        """(a transcribe.cpp file's K-quant matrices dequantized: the file is the tower's only one)"""
        return V.weights(gguf_getter(self.gguf, dequantize=True), self.cfg.layers, self.cfg.projection_dim, self.vit)

    def build(self, npu: VisionDevice, w: dict, state: dict | None, store: ProgramStore) -> list:
        self.encoder = V.Encoder(npu, w, eps=float(self.gguf.meta["clip.audio.projector.layer_norm_epsilon"]),
                                 state=state)
        return [self.encoder]

    def state(self) -> dict:
        return self.encoder.state()

    def embeddings(self, samples: np.ndarray) -> np.ndarray:
        """A recording's samples (16 kHz mono float32) -> its tokens' rows (tokens, projection_dim): its chunks' in
        order, each its own tokens (``host.tokens``)."""
        x = np.ascontiguousarray(samples, np.float32).reshape(-1)
        counts = [host.tokens(min(host.CHUNK, len(x) - i)) for i in range(0, len(x), host.CHUNK)]
        run = lambda: np.concatenate(list(self.encoder.stream(zip(host.chunks(x), counts, strict=True), host.log_mel)))
        return self.results.get(digest(x), run, count=len(counts))

    def rows(self, clip: str | bytes | np.ndarray) -> np.ndarray:
        """A recording (a file, its bytes or 16 kHz samples) as its tokens' rows (``embeddings``)."""
        return self.embeddings(clip if isinstance(clip, np.ndarray) else load_audio(clip))
