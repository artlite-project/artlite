"""MOSS-Transcribe's audio tower on the vision overlay: a 30 s chunk's samples -> its log-mel (host) -> conv1 (``Stem1``)
-> conv2 (``Stem2``, its A panels assembled from conv1's bytes) -> the 24 encoder layers (``Chain``) -> the final
LayerNorm and the four-row merge (host) -> the VQ adaptor (``Adaptor``) -> the rows a decoder prompt's audio span
reads (``host.span_tokens`` of the clip's samples' count, at most a 30 s chunk's 375).

Four programs of the vision overlay share one ``VisionDevice`` (the array's one hardware context), and the weights are
the checkpoint's own: the audio tower's tensors (``model.whisper_encoder.*`` and ``model.vq_adaptor.*``, fetched into the
same directory as the decoder's shard) are read through ``HfModel``'s checkpoint names (``hf_f32``: the language model's
GGUF map does not know them). ``load`` builds the four programs and dispatches each once (the cold cost: a program's
first dispatch configures its context; the layer chain's 24 transactions for the chain's), so that a chunk's measured
time is the steady one.

The stem's second stage takes conv1's *raw* panels, not its decoded rows: an 8-channel block of conv1's GELU is the
conv2's A block at a shifted token (``stem.assemble_a``), so the 3,136-column A is assembled byte for byte (no float
round trip, no re-rounding) - the host's 24.9 ms of packing is this and the first stage's.
"""

from dataclasses import dataclass, field

import numpy as np

from artlite.hf_model import HfModel
from artlite.operators.design import COLS
from artlite.vision.device import VisionDevice
from models.lfm3a.host import load_audio  # (the 16 kHz mono decoder the speech front ends share)

from . import host
from .adaptor import Adaptor, merged
from .chain import Chain
from .stem import CH, N, Stem1, Stem2

LAYERS = 24  # (the checkpoint's Whisper encoder: the chain's programs are one per layer)
ROWS, CHANNELS = 375, 1024  # (the adaptor's rows and the encoder's channels: a decoder prompt's audio span)
LN_EPS = 1e-5  # (WhisperConfig's final LayerNorm)
PREFIX = "model.whisper_encoder"
ADAPTOR = "model.vq_adaptor"


def weights(g: HfModel, *, layers: int = LAYERS) -> dict[str, object]:
    """The tower's weights (fp32, by the checkpoint's names) in the stages' own dicts: ``conv1`` / ``conv2`` (their
    weights and biases), ``pos`` (the position table), ``ln`` (the final LayerNorm's gamma / beta), ``layers`` (one dict
    a Whisper encoder layer, ``chain``'s keys) and ``adaptor``."""
    f32 = g.hf_f32
    out: dict[str, object] = {
        "conv1": {"c1_w": f32(f"{PREFIX}.conv1.weight"), "c1_b": f32(f"{PREFIX}.conv1.bias")},
        "conv2": {"c2_w": f32(f"{PREFIX}.conv2.weight"), "c2_b": f32(f"{PREFIX}.conv2.bias")},
        "pos": f32(f"{PREFIX}.embed_positions.weight"),
        "ln": (f32(f"{PREFIX}.layer_norm.weight"), f32(f"{PREFIX}.layer_norm.bias")),
        "adaptor": {"a0_w": f32(f"{ADAPTOR}.layers.0.weight"), "a0_b": f32(f"{ADAPTOR}.layers.0.bias"),
                    "a2_w": f32(f"{ADAPTOR}.layers.2.weight"), "a2_b": f32(f"{ADAPTOR}.layers.2.bias"),
                    "a3_w": f32(f"{ADAPTOR}.layers.3.weight"), "a3_b": f32(f"{ADAPTOR}.layers.3.bias")},
        "layers": [],
    }
    for i in range(layers):
        p = f"{PREFIX}.layers.{i}."
        out["layers"].append({  # (``chain``'s keys: the pre-LNs, the four projections, the FFN)
            "ln_w": f32(p + "self_attn_layer_norm.weight"), "ln_b": f32(p + "self_attn_layer_norm.bias"),
            "q_w": f32(p + "self_attn.q_proj.weight"), "q_b": f32(p + "self_attn.q_proj.bias"),
            "k_w": f32(p + "self_attn.k_proj.weight"), "v_w": f32(p + "self_attn.v_proj.weight"),
            "v_b": f32(p + "self_attn.v_proj.bias"), "o_w": f32(p + "self_attn.out_proj.weight"),
            "o_b": f32(p + "self_attn.out_proj.bias"),
            "f_ln_w": f32(p + "final_layer_norm.weight"), "f_ln_b": f32(p + "final_layer_norm.bias"),
            "fc1_w": f32(p + "fc1.weight"), "fc1_b": f32(p + "fc1.bias"),
            "fc2_w": f32(p + "fc2.weight"), "fc2_b": f32(p + "fc2.bias")})
    return out


@dataclass
class Tower:
    """The audio tower on a vision device: the four programs, the weights they were built from, the chunk's path."""

    gguf: HfModel
    npu: VisionDevice | None = None
    stem1: Stem1 | None = None
    stem2: Stem2 | None = None
    chain: Chain | None = None
    adaptor: Adaptor | None = None
    w: dict[str, object] = field(default_factory=dict)
    t_load: float = 0.0  # (the build and the warm, seconds)
    times: dict[str, float] = field(default_factory=dict)  # (the last chunk's stages, seconds: mel, stem1, stem2, the
    # 24 layers, the final norm + merge + adaptor)

    @property
    def loaded(self) -> bool:
        return self.npu is not None

    def load(self) -> None:
        """Build the four programs (the vision overlay's design, an ELF of the cache) and dispatch each once: the weight
        images packed, the contexts configured. Idempotent."""
        import time

        if self.loaded:
            return
        t0 = time.perf_counter()
        self.w = w = weights(self.gguf)
        npu = self.npu = VisionDevice()
        self.stem1 = Stem1(npu, w["conv1"])  # type: ignore[arg-type]
        self.stem2 = Stem2(npu, w["conv2"])  # type: ignore[arg-type]
        self.chain = Chain(npu, w["layers"])  # type: ignore[arg-type]
        self.adaptor = Adaptor(npu, w["adaptor"])  # type: ignore[arg-type]
        for p in (self.stem1, self.stem2, self.chain, self.adaptor):
            p.warm()  # (a program's first dispatch configures its context: the cold cost, paid here)
        self.t_load = time.perf_counter() - t0

    def rows(self, samples: np.ndarray) -> np.ndarray:
        """A clip's samples (16 kHz mono) -> the adaptor's rows (``host.span_tokens(len(samples))``, 1,024) fp32: the
        encoder's first ``count * 4`` rows of its 30 s chunk's 1,500 (the processor's slice of a chunk's features), the
        final LayerNorm and the four-row merge, the adaptor. The stages' seconds of this chunk left in ``times``."""
        import time

        self.load()
        assert self.stem1 and self.stem2 and self.chain and self.adaptor and self.npu
        count = host.span_tokens(np.asarray(samples).size)
        t = self.times
        t0 = time.perf_counter()
        x = host.log_mel(host.fit_chunk(samples))  # (80, 3,000), the host's 7.6 ms
        t["mel"] = time.perf_counter() - t0
        frames = x.T  # (3,000, 80)
        t0 = time.perf_counter()
        self.stem1.write(frames)
        self.stem1.run()
        t["stem1"] = time.perf_counter() - t0
        # (conv1's raw panels: a column's [channel block][token panel][8 tokens][9 B], the 1,280 width's 160 blocks cut
        # to the encoder's 1,024 channels; conv2's A is their blocks at shifted tokens, byte for byte)
        t0 = time.perf_counter()
        src = np.stack([np.frombuffer(self.npu.read(f"{self.stem1.name}.y.{b}"), np.uint8)
                        .reshape(COLS, N // 8, 2, 8, 9)[:, :CH // 8] for b in range(self.stem1.lv.nb)])
        self.stem2.write_bytes(src, self.stem1.lv.nb, real=host.CHUNK_ROWS)
        self.stem2.run()
        pos = self.w["pos"]  # type: ignore[assignment]
        x = self.stem2.read()[:host.CHUNK_ROWS] + np.asarray(pos, np.float32)[:host.CHUNK_ROWS]
        t["stem2"] = time.perf_counter() - t0
        padded = np.zeros((self.chain.lv.nb * COLS * self.chain.lv.m, x.shape[1]), np.float32)  # (the 1,536-row grid)
        padded[:host.CHUNK_ROWS] = x
        t0 = time.perf_counter()
        self.chain.write(padded)
        self.chain.run()
        enc = self.chain.read()[:count * host.MERGE]  # (the clip's span: count merged rows)
        t["layers"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ln_w, ln_b = self.w["ln"]  # type: ignore[misc]
        mg = merged(host.layer_norm(enc, ln_w, ln_b, LN_EPS))  # (375, 4,096)
        self.adaptor.rows_image(mg)
        self.adaptor.run()
        rows = self.adaptor.read()[:mg.shape[0]]
        t["norm+adaptor"] = time.perf_counter() - t0
        return rows

    def sequence(self, clip: object, *, digits: dict[str, int]) -> object:
        """A clip (a file, its bytes or 16 kHz samples) as a decoder prompt's audio span: ``rows`` of it (the span's
        count from its samples, ``host.span_tokens``), then ``host.audio_sequence`` (one ``Sequence.given`` row per
        pad)."""
        samples = clip if isinstance(clip, np.ndarray) else load_audio(clip)  # type: ignore[arg-type]
        return host.audio_sequence(self.rows(samples), digits=digits,
                                   count=host.span_tokens(np.asarray(samples).size))
