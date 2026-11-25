"""Thinking Machines' Inkling 2 from a GGUF file (``artlite.model_config.InklingConfig``: 26B weights in 24 layers, a
mixture of 128 experts of which 8 run per token, the middle 12 layers looped 3 times: 48 visits a token).

The chat template (the file's): a conversation is blocks ``<|message_{user,model,system,tool}|>[name]<|content_*|>
.. <|end_message|>``, a model turn ends with ``<|content_model_end_sampling|>``; a system block "Thinking effort level:
x" (``reasoning_effort``: none 0 .. max 0.99, or a number; the template's default 0.9) before the first message that is
not a system one; tools as one JSON line in a ``tool_declare`` block, calls ``<|message_model|>NAME
<|content_invoke_tool_json|>{"name": .., "args": {..}}``; an image part as ``<|content_image|><|unused_200054|>``.
The template opens no turn for the reply: the model writes its own blocks (``<|message_model|><|content_thinking|>``
.. then ``<|message_model|><|content_text|>`` ..), and its turn ends at ``<|content_model_end_sampling|>``.

A decode dispatch runs up to 4 consecutive tokens through the 48 visits and the LM head on one stream of the weights
(``overlay.InklingOverlay``): the projections, the depth mixes, the loop's controllers and the experts on the main cores
and their helpers, the residual stream's state array in their memory, the attention, its KV caches and the K / V convs
on the attention cores; the head screened (``decode.screen``: the host computes its largest logits exactly). Prompts run
in prefill chunks (``prefill.InklingPrefill``; its sliding-window caches a ring that its largest chunk fits in,
``InklingPrefill.ring``), a prompt's last token as a decode step
(the head's); without the prefill as decode steps. The convs' state holds the last positions only: a sequence goes back
to a checkpoint (``recurrent``)."""

from collections.abc import Sequence

import numpy as np

from artlite.decode.kv import kv_precision
from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig

from ..lanes import alone
from ..model import Model
from ..overlay_model import OverlayModel
from ..sampling import TOP_K_ALL, Sampling
from ..tokenizer import BpeTokenizer, chat_prompt
from .overlay import InklingOverlay
from .prefill import InklingPrefill


class NpuInkling(OverlayModel):
    """The model on the NPU (module docstring). ``capacity``: the positions the KV caches hold; ``kv_k``: their K
    precision ("hilo", "single"); ``wrap``: the sliding-window caches' ring (``InklingOverlay``); ``sequences``: the
    sequences the device holds (``artlite-server -np``: lanes whose steps share a dispatch), each with its caches and
    convs' state; ``prefill``: prompts in prefill chunks (``wrap`` by default ``InklingPrefill.ring``)."""

    recurrent = True  # (the K / V convs' rings and the post convs' history windows: the last positions only)
    last_step = True  # (the prefill has no LM head)
    LANE_SHARED = (*OverlayModel.LANE_SHARED, "gguf", "_tn")

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "hilo", threads: int = 0,
                 wrap: int | None = None, sequences: int = 1, prefill: bool = False):
        self.gguf = gguf
        if prefill and wrap is None:  # (a ring a chunk of the largest size fits in, from its multiples)
            wrap = InklingPrefill.ring(capacity)
        self.overlay = InklingOverlay(gguf, max_positions=capacity, kv_k=kv_k, wrap=wrap, sequences=sequences)
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.softcap = self.cfg.softcap
        self.prefill = InklingPrefill(self.overlay) if prefill else None
        self._tn = np.asarray(gguf.tensor("token_embd_norm.weight").dequantize(), np.float32).reshape(-1)
        self._open(threads=threads)
        self._w_out = np.asarray(gguf.tensor("output_norm.weight").dequantize(), np.float32).reshape(-1)
        gguf.release()
        self.position = 0
        self._open_lanes()

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov = self.overlay
        kv = f"{ov.sizes['KV'] / 1e6:.0f} MB of KV cache and conv state"
        return (f"{ov.image.nbytes / 1e9:.2f} GB of weights ({ov.image.streamed_bytes() / 1e9:.2f} GB dense), "
                f"{kv if ov.sequences == 1 else f'{ov.sequences} sequences of {kv} each'} ({self.capacity} positions, K "
                f"{'hi + lo' if ov.kp == 2 else 'single'})")

    @alone
    def reset(self) -> None:
        """Forget the cached positions (and the convs' state)."""
        self._abandon()
        self.overlay.reset()
        self.position = 0

    def _step_inputs(self, tokens: list[int]) -> tuple:
        """(the first visit's input: the embedding rows RMS-normed with ``token_embd_norm``, as one state)"""
        x = self.gguf.rows("token_embd.weight", np.asarray(tokens)).astype(np.float32)
        x = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + self.cfg.rms_norm_eps).astype(np.float32)
        return ((x * self._tn)[None],)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        """(the chunk's normed embeddings (n, hidden): the first visit's input)"""
        return (self._step_inputs(tokens)[0][0],)


class Inkling(Model):
    """An Inkling 2 GGUF file (either layout, ``InklingConfig``): the architecture, tokenizer and chat template from its
    metadata (module docstring)."""

    name = "Inkling 2"
    thinks = True
    reasoning = ("<|content_thinking|>", "<|end_message|>")
    thought_first = True

    def __init__(self, path: str, *, kv_k: str = "hilo"):
        kv_precision(kv_k)  # (a K precision the overlay takes, before the file opens)
        self.gguf = GGUFFile(path)
        self.cfg = InklingConfig.from_gguf(self.gguf)
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.kv_k = kv_k

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_text(self, messages: list[dict[str, object]], *, think: bool = False,
                  tools: list[dict[str, object]] | None = None, preserve: bool = False,
                  variables: dict[str, object] | None = None) -> str:
        """The chat template's text of a conversation (``think``: the template's reasoning effort, the given
        ``reasoning_effort`` or its default; else "none", unless ``variables`` give one; ``preserve``: the earlier
        replies keep their ``reasoning_content``)."""
        if not preserve:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        variables = dict(variables or {})
        if not think:
            variables.setdefault("reasoning_effort", "none")
        return chat_prompt(self.gguf, messages, tools=tools, template=self.chat_template, **variables)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(self.chat_text(messages, think=think, tools=tools, preserve=preserve,
                                                    variables=variables))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*: temperature 1, top-k 0 (all), top-p 1, min-p 0)."""
        s = {"temp": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0} | self.cfg.sampling
        k = int(s["top_k"])
        return Sampling(temp=s["temp"], top_k=k if k > 0 else TOP_K_ALL, top_p=s["top_p"], min_p=s["min_p"])

    def rows(self, tokens: Sequence[int]) -> np.ndarray:
        """The tokens' embedding rows (n, hidden) fp32, from the mapped file (before ``token_embd_norm``, which a
        text token's row takes and an image's does not)."""
        return self.gguf.rows("token_embd.weight", np.asarray(tokens))

    def load(self, *, capacity: int, prefill: bool = True, sequences: int = 1) -> NpuInkling:
        """(``sequences``: ``NpuInkling``'s)"""
        return NpuInkling(self.gguf, capacity=capacity, kv_k=self.kv_k, sequences=sequences, prefill=prefill)


__all__ = ["Inkling", "NpuInkling"]
