"""XHToken Spark-X2.5 / X2.6 from a GGUF file (``artlite.model_config.SparkConfig``): 4B (36 layers) and 1.7B (28),
three sliding-window layers per full one, head_dim 256, a sigmoid output gate per head, GeGLU; XHToken's Q4_K_M
(Q6_K / Q8_0 in the sensitive tensors, a Q8_0 head) and Q8_0 files, the community's quantizations. A DSpark drafter
(``model_config.spark_dspark_of``: the model's ``<repository>-DSpark-GGUF``) drafts 7 tokens a pass.

The chat template (the file's): every turn ``<｜start▁of▁sentence｜>`` + its role tag (``<|System|>``, ``<|User|>``,
``<|Bot|>``, ``<|Tool|>``) .. ``<｜end▁of▁sentence｜>``, a default system prompt (with its ``thinking_effort`` sentence when
thinking); a reply after ``<|Bot|><think>`` (thinking) or ``<|Bot|></think>``; an earlier reply keeps its thought where
its message has ``reasoning_content`` (``preserve``, else dropped); tools as JSON lines in ``<tools>``, calls in GLM-4.5's
syntax (``server.tools``); ``reasoning_effort`` "high" (the template's default) or "low" (``Spark.efforts``: the levels
clients send mapped to those two)."""

from artlite.gguf_model import GGUFFile
from artlite.model_config import SparkConfig, spark_dspark_of

from ..model import Model
from ..npu_model import NpuModel
from ..sampling import TOP_K_ALL, Sampling
from ..spec_options import drafter_type
from ..tokenizer import BpeTokenizer, chat_prompt


class Spark(Model):
    """A spark2_5 GGUF file: the architecture, tokenizer and chat template from its metadata (module docstring); a raw
    text gets no BOS (the file's ``add_bos_token``, as llama.cpp). ``spec_type`` / ``draft_model``: its DSpark drafter
    (``spec_options``: auto, the one found by name; none)."""

    name = "Spark-X"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<｜start▁of▁sentence｜>", "<｜start▁of▁sentence｜><|User|>"
    efforts = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "high"}  # noqa: RUF012

    def __init__(self, path: str, *, spec_type: str = "auto", draft_model: str | None = None):
        self.gguf = GGUFFile(path)
        self.cfg = SparkConfig.from_gguf(self.gguf)
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        kind = drafter_type(spec_type, draft_model, ("draft-dspark",), self.name)
        self.dspark = (draft_model or spark_dspark_of(path)) if kind in ("auto", "draft-dspark") else None
        if kind == "draft-dspark" and self.dspark is None:
            raise ValueError(f"{path}: no DSpark drafter (a <repository>-DSpark-GGUF) in the models directory")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat_text(self, messages: list[dict[str, object]], *, think: bool = False,
                  tools: list[dict[str, object]] | None = None, preserve: bool = False,
                  variables: dict[str, object] | None = None) -> str:
        """The chat template's text of a conversation up to the assistant's turn (``preserve``: the earlier replies keep
        their ``reasoning_content``)."""
        if not preserve:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        return chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools, template=self.chat_template,
                           **(variables or {}))

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(self.chat_text(messages, think=think, tools=tools, preserve=preserve,
                                                    variables=variables))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*: temperature 1, top-p 0.95, top-k off)."""
        s = {"temp": 1.0, "top_k": -1, "top_p": 0.95} | self.cfg.sampling
        k = int(s["top_k"])
        return Sampling(temp=s["temp"], top_k=k if k > 0 else TOP_K_ALL, top_p=s.get("top_p", 1.0),
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))

    def load(self, *, capacity: int, prefill: bool = True) -> NpuModel:
        raise NotImplementedError(f"{self.gguf.path}: {self.name} does not run on the NPU yet")
