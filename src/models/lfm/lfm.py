"""LiquidAI LFM3-1.7B from a GGUF file (QAD-Q4_0, Q4_K_M, Q8_0; BF16 tensors as BFP16 tiles): 24 layers of three
kinds (short convolutions, sliding-window and full attention), each [operator | SwiGLU FFN] with sandwich norms
(``artlite.model_config.LfmConfig``)."""

from artlite.gguf_model import GGUFFile
from artlite.model_config import LfmConfig

from ..common import Model, Sampling
from ..tokenizer import BpeTokenizer, chat_prompt

# the model card's sampling (the file's general.sampling.* over it)
SAMPLING = {"temp": 0.1, "top_k": 50, "penalty_repeat": 1.1}


class Lfm(Model):
    """An lfm3 GGUF file: the architecture, tokenizer and chat template from its metadata. Thinking: the template opens
    a thought unless ``enable_thinking`` is false (an empty one, ``<think></think>``); a raw text gets no BOS
    (the file's ``add_bos_token``, as llama.cpp)."""

    name = "LFM3"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<|im_start|>", "<|im_start|>user"

    def __init__(self, path: str):
        self.gguf = GGUFFile(path)
        self.cfg = LfmConfig.from_gguf(self.gguf)
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.bos = int(self.gguf.meta["tokenizer.ggml.bos_token_id"])

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        bos = self.gguf.strings("tokenizer.ggml.tokens", self.bos)[0]
        return self.tokenizer.encode(chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools,
                                                 preserve_thinking=preserve, template=self.chat_template,
                                                 **{"bos_token": bos, **(variables or {})}))

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*), else the model card's."""
        s = SAMPLING | self.cfg.sampling
        return Sampling(temp=s["temp"], top_k=int(s["top_k"]), top_p=s.get("top_p", 1.0),
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))
