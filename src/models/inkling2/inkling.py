"""Thinking Machines' Inkling 2 from a GGUF file (``artlite.model_config.InklingConfig``: 26B weights in 24 layers, a
mixture of 128 experts of which 8 run per token, the middle 12 layers looped 3 times: 48 visits a token).

The chat template (the file's): a conversation is blocks ``<|message_{user,model,system,tool}|>[name]<|content_*|>
.. <|end_message|>``, a model turn ends with ``<|content_model_end_sampling|>``; a system block "Thinking effort level:
x" (``reasoning_effort``: none 0 .. max 0.99, or a number; the template's default 0.9) before the first message that is
not a system one; tools as one JSON line in a ``tool_declare`` block, calls ``<|message_model|>NAME
<|content_invoke_tool_json|>{"name": .., "args": {..}}``; an image part as ``<|content_image|><|unused_200054|>``.
The template opens no turn for the reply: the model writes its own blocks (``<|message_model|><|content_thinking|>``
.. then ``<|message_model|><|content_text|>`` ..), and its turn ends at ``<|content_model_end_sampling|>``.

The NPU path (decode, prefill) is not there yet: ``load`` refuses."""

from collections.abc import Sequence

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import InklingConfig

from ..model import Model
from ..npu_model import NpuModel
from ..sampling import TOP_K_ALL, Sampling
from ..tokenizer import BpeTokenizer, chat_prompt


class Inkling(Model):
    """An Inkling 2 GGUF file (either layout, ``InklingConfig``): the architecture, tokenizer and chat template from its
    metadata (module docstring)."""

    name = "Inkling 2"
    thinks = True
    reasoning = ("<|content_thinking|>", "<|end_message|>")
    thought_first = True

    def __init__(self, path: str):
        self.gguf = GGUFFile(path)
        self.cfg = InklingConfig.from_gguf(self.gguf)
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids

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

    def load(self, *, capacity: int, prefill: bool = True) -> NpuModel:
        raise NotImplementedError(f"{self.name}: the NPU path (decode, prefill) is not implemented yet")


__all__ = ["Inkling"]
