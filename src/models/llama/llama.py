"""Llama 3.2 1B / 3B and 3.1 8B on the NPU from a GGUF file (Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q4_0, Q4_1, Q3_K_L,
Q3_K_M, Q2_K; BF16 tensors as BFP16 tiles).

A dispatch runs up to 4 consecutive tokens through every layer and the LM head on one stream of the weights
(``models.llama.overlay.LlamaOverlay``): the projections on the main cores and their helpers (the MMUL
against exact-integer weight panels), the attention and its KV cache on the attention cores. The host reads the tokens'
embedding rows from the mapped file and samples from the logits.

A prompt runs in chunks of up to 4096 tokens (8B: 2048) through the prefill
(``models.llama.prefill.LlamaPrefill``: layer-major, control code assembled per chunk from one library ELF;
the GEMMs read the same weight image, the chunk's K / V land in the decode's caches), each chunk in pieces of a few
layers; its last token's logits come from the LM head alone. Tokens where a chunk cannot start (inside a cache group of
64 positions at head_dim 128, 128 at 64: after a generated reply), and a prompt's last 4 or fewer, run as decode steps
of up to 4 tokens.

The KV caches hold K as one bfp16 block per 8 dims (``kv_k="single"``, the default: 18 / 64.5 / 73.7 KB per position
for 1B / 3B / 8B) or as hi + lo blocks ("hilo": +50% of the caches' bytes, and of their read time at long contexts).

Memory: the weights the NPU streams are re-laid-out into a device buffer at load (packed by the C++ packer in a thread
while the tokenizer and the designs load), at their quantization's size; nothing is written to disk.
"""

import numpy as np

from artlite.decode.kv import kv_precision
from artlite.gguf_model import GGUFFile
from artlite.model_config import LlamaConfig
from artlite.overlay import WeightBuffer

from ..model import Model
from ..overlay_model import OverlayModel
from ..sampling import Sampling
from .image import LlamaImage
from .overlay import LlamaOverlay, check_formats
from .prefill import SIZES, LlamaPrefill
from .tokenizer import BOS, chat_prompt, load_tokenizer


class NpuLlama(OverlayModel):
    """The model on the NPU (Llama 3.x, or Qwen3: ``qwen3.Qwen3`` loads it). ``capacity``: the positions the KV caches hold; ``kv_k``: their K precision ("single",
    "hilo"); ``prefill``: prompts in chunks through the prefill (else as decode steps), of the sizes ``sizes``
    (default ``prefill.SIZES``); ``weights``: the weight image's buffer, allocated ahead (``WeightBuffer``); ``readout``:
    a readout model (``models.readouts.readout_model``): the prefill's precise variants and packed chunks
    (``LlamaPrefill``'s ``precise``, ``pack``)."""

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "single", threads: int = 0,
                 prefill: bool = True, sizes: tuple[int, ...] = SIZES, weights: WeightBuffer | None = None,
                 readout: bool = False):
        self.gguf = gguf
        self.overlay = LlamaOverlay(gguf, max_positions=capacity, kv_k=kv_k, weights=weights)
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.prefill = LlamaPrefill(self.overlay, sizes=sizes, precise=readout, pack=readout) if prefill else None
        self._open(threads=threads)
        self._w_out = gguf.tensor("output_norm.weight").dequantize().reshape(-1)
        gguf.release()
        self.position = 0

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov = self.overlay
        return (f"{ov.image.streamed_bytes() / 1e9:.2f} GB of weights, {ov.sizes['KV'] / 1e6:.0f} MB of KV cache "
                f"({self.capacity} positions, K {'hi + lo' if ov.kp == 2 else 'single'})")

    def reset(self) -> None:
        """Forget the cached positions."""
        self.overlay.reset()
        self.position = 0

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (self.gguf.rows("token_embd.weight", tokens),)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        return (self.gguf.rows("token_embd.weight", tokens),)

    def _chunk_logits(self, token: int, n: int) -> np.ndarray:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))


class Llama(Model):
    """Llama 3.x from a GGUF file: the architecture, tokenizer and chat template from its metadata. Opening the file
    allocates the weight image's buffer and packs it in a thread (``WeightBuffer``) while the tokenizer and the designs
    load. ``kv_k``: the KV caches' K precision ("single": one bfp16 block per 8 dims, "hilo": two)."""

    name = "Llama 3"

    def __init__(self, path: str, *, kv_k: str = "single"):
        kv_precision(kv_k)  # (a K precision the overlay takes, before the file opens)
        self.gguf = GGUFFile(path)
        self.cfg = LlamaConfig.from_gguf(self.gguf)
        image = LlamaImage(self.gguf, self.cfg)
        check_formats(image)
        self.weights = WeightBuffer(image)
        self.tokenizer = load_tokenizer(path)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.kv_k = kv_k

    def encode(self, text: str) -> list[int]:
        return [self.tokenizer.special[BOS]] + self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(chat_prompt(self.gguf, messages, tools=tools, template=self.chat_template,
                                                 **(variables or {})))

    def sampling(self, *, think: bool = False) -> Sampling:
        return Sampling(temp=0.7, top_k=50)

    def load(self, *, capacity: int, prefill: bool = True) -> NpuLlama:
        return NpuLlama(self.gguf, capacity=capacity, kv_k=self.kv_k, prefill=prefill, weights=self.weights)
