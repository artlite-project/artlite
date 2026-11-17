"""Recursive's Flint from a GGUF file (``artlite.model_config.FlintConfig``): 40B parameters in 49 layers, a mixture of
256 experts of which 6 run per token (~2.5B active) and a shared expert, sliding-window layers of 513 keys and full ones
without positional encoding, learned attention sinks. The official file (``model_config.flint_gguf()``: the model's
trained INT4 codes in groups of 64, stored as GGUF Q4_1 pairs) runs as Q4_1_64 tiles, its F32 router as BFP16 planes
(``flint.image``).

A dispatch runs up to 4 consecutive tokens through every layer and the LM head on one stream of the weights
(``flint.overlay.FlintOverlay``): the projections and the experts on the main cores and their helpers, the routing on the
helpers (the experts' tiles pushed by the columns themselves: ``moe.route``), the attention and its KV caches on the
attention cores. The host reads the tokens' embedding rows from the mapped file and samples from the logits.

The chat template (the file's): GLM-style turns ``<|system|>``, ``<|user|>``, ``<|assistant|>``, ``<|observation|>`` (tool
results); the model opens its own ``<think>`` (the prompt never does); tools as JSON in ``<tools>``, calls in GLM-4.5's
syntax (``server.tools``). A generation ends at ``<|end_of_text|>``, ``<|user|>`` or ``<|observation|>``.

The KV caches hold K as hi + lo bfp16 blocks by default (``kv_k``; "single": one block per 8 dims): 14 full layers x 4 KV
heads x 432 B = 24.2 KB per position, and every sliding-window layer's cache (linear in the positions)."""

from artlite.decode.kv import kv_precision
from artlite.gguf_model import GGUFFile
from artlite.model_config import FlintConfig
from artlite.overlay import WeightBuffer

from ..model import Model
from ..overlay_model import OverlayModel
from ..sampling import TOP_K_ALL, Sampling
from ..tokenizer import BpeTokenizer, chat_prompt
from .image import FlintImage
from .overlay import FlintOverlay


class NpuFlint(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the KV caches hold; ``kv_k``: their K precision ("hilo",
    "single"); ``weights``: the weight image's buffer, allocated ahead (``WeightBuffer``). Prompts run as decode steps of
    up to 4 tokens."""

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "hilo", threads: int = 0,
                 weights: WeightBuffer | None = None):
        self.gguf = gguf
        self.overlay = FlintOverlay(gguf, max_positions=capacity, kv_k=kv_k, weights=weights)
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.prefill = None
        self._open(threads=threads)
        self._w_out = gguf.tensor("output_norm.weight").dequantize().reshape(-1)
        gguf.release()
        self.position = 0

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov = self.overlay
        return (f"{ov.image.nbytes / 1e9:.2f} GB of weights ({ov.image.streamed_bytes() / 1e9:.2f} GB dense), "
                f"{ov.sizes['KV'] / 1e6:.0f} MB of KV cache ({self.capacity} positions, K "
                f"{'hi + lo' if ov.kp == 2 else 'single'})")

    def reset(self) -> None:
        """Forget the cached positions."""
        self.overlay.reset()
        self.position = 0

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (self.gguf.rows("token_embd.weight", tokens),)


class Flint(Model):
    """A flint GGUF file: the architecture, tokenizer and chat template from its metadata (module docstring). Opening the
    file allocates the weight image's buffer and packs it in a thread (``WeightBuffer``) while the tokenizer and the
    designs load. ``kv_k``: the KV caches' K precision ("hilo", the default; "single")."""

    name = "Flint"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True

    def __init__(self, path: str, *, kv_k: str = "hilo"):
        kv_precision(kv_k)  # (a K precision the overlay takes, before the file opens)
        self.gguf = GGUFFile(path)
        self.cfg = FlintConfig.from_gguf(self.gguf)
        self.weights = WeightBuffer(FlintImage(self.gguf, self.cfg))
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.kv_k = kv_k

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_text(self, messages: list[dict[str, object]], *, think: bool = False,
                  tools: list[dict[str, object]] | None = None, preserve: bool = False,
                  variables: dict[str, object] | None = None) -> str:
        """The chat template's text of a conversation up to the assistant's turn (``preserve``: the earlier replies keep
        their ``reasoning_content``; the template has no thinking switch: the model thinks when it opens ``<think>``)."""
        if not preserve:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        return chat_prompt(self.gguf, messages, tools=tools, template=self.chat_template, **(variables or {}))

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(self.chat_text(messages, think=think, tools=tools, preserve=preserve,
                                                    variables=variables))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*: temperature 1, top-p 0.95, top-k 20)."""
        s = {"temp": 1.0, "top_k": 20, "top_p": 0.95} | self.cfg.sampling
        k = int(s["top_k"])
        return Sampling(temp=s["temp"], top_k=k if k > 0 else TOP_K_ALL, top_p=s.get("top_p", 1.0),
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))

    def load(self, *, capacity: int, prefill: bool = True) -> NpuFlint:
        return NpuFlint(self.gguf, capacity=capacity, kv_k=self.kv_k, weights=self.weights)


__all__ = ["Flint", "NpuFlint"]
