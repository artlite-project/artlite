"""Qwen3 on the NPU (GGUF architecture ``qwen3``: Qwen3 0.6B, Qwen3-Embedding, Qwen3-Reranker; ``vllm-decision2``:
Decision-2.0-Kai, Qwen3 with a decision head the server runs on the host) on Llama's path: the same layers but q and k
RMS-normed per head with a weight per dim before RoPE, NeoX rotary pairs in the file's rows (``LlamaConfig.neox``,
``qk_norm``), no BOS, Qwen's tokenizer and ChatML template (``models.tokenizer``). Qwen3-Embedding-4B's hidden size of
2560 and ffn of 9728 run in K chunks padded with zero super-blocks (``llama.image``).

The KV caches hold K as hi + lo bfp16 blocks by default: the k_norm weights (up to ~100) leave dims a thousand times
smaller than their block's largest, which a single block per 8 dims loses (KL 0.005-0.008 on a chat and an essay window
of Qwen3 0.6B against ~0 with hi + lo, the CPU model with the cache's roundings).
"""

from artlite.gguf_model import GGUFFile
from artlite.model_config import LlamaConfig
from artlite.overlay import WeightBuffer

from ..common import Model, Sampling
from ..readouts import readout_model
from ..tokenizer import QwenTokenizer, chat_prompt
from .image import LlamaImage
from .llama import NpuLlama
from .overlay import KV_K, check_formats

# the model card's sampling (thinking, not)
SAMPLING = {True: {"temp": 0.6, "top_k": 20, "top_p": 0.95}, False: {"temp": 0.7, "top_k": 20, "top_p": 0.8}}


class Qwen3(Model):
    """A qwen3 (or vllm-decision2) GGUF file: the architecture, tokenizer and chat template from its metadata. Opening
    the file allocates the weight image's buffer and packs it in a thread (``WeightBuffer``) while the tokenizer and the
    designs load. ``kv_k``: the KV caches' K precision ("hilo": two bfp16 blocks per 8 dims, the default; "single")."""

    name = "Qwen3"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<|im_start|>", "<|im_start|>user"

    def __init__(self, path: str, *, kv_k: str = "hilo"):
        if kv_k not in KV_K:
            raise ValueError(f"kv_k: {kv_k!r} (one of {', '.join(KV_K)})")
        self.gguf = GGUFFile(path)
        self.cfg = LlamaConfig.from_gguf(self.gguf)
        image = LlamaImage(self.gguf, self.cfg)
        check_formats(image)
        self.weights = WeightBuffer(image)
        self.tokenizer = QwenTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.kv_k = kv_k

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        return self.tokenizer.encode(chat_prompt(self.gguf, messages, enable_thinking=think, tools=tools,
                                                 preserve_thinking=preserve, template=self.chat_template,
                                                 **(variables or {})))

    def sampling(self, *, think: bool = False) -> Sampling:
        s = SAMPLING[think]
        return Sampling(temp=s["temp"], top_k=s["top_k"], top_p=s["top_p"])

    def load(self, *, capacity: int, prefill: bool = True) -> NpuLlama:
        return NpuLlama(self.gguf, capacity=capacity, kv_k=self.kv_k, prefill=prefill, weights=self.weights,
                        readout=readout_model(self.gguf.meta, self.gguf.arch))
