"""LiquidAI LFM3-1.7B on the NPU from a GGUF file (QAD-Q4_0, Q4_K_M, Q8_0; BF16 tensors as BFP16 tiles): 24 layers
of three kinds (short convolutions, sliding-window and full attention), each [operator | SwiGLU FFN] with sandwich
norms (``artlite.model_config.LfmConfig``).

A dispatch runs up to 4 consecutive tokens through every layer and the LM head on one stream of the weights
(``models.lfm.overlay.LfmOverlay``): the projections on the main cores and their helpers, the attention, its KV caches
and the conv layers' short convolutions on the operator cores. A prompt runs in chunks of up to 4096 tokens through the
prefill (``models.lfm.prefill.LfmPrefill``: layer-major from one library ELF on the same weight image; the chunk's K / V
land in the decode's caches, its conv rows in the conv layers' histories); its last token's logits come from the LM
head alone. Tokens where a chunk cannot start (inside a cache group of 64 positions) and a prompt's last few run as
decode steps.

The conv histories hold the whole sequence (``recurrent``): a position is kept only where a checkpoint was taken (the
histories, ~0.5 MB). Drafts: the context's (``drafts.Lookup``), whose rejected tokens cost nothing (the histories'
lazy shift, the caches' records overwritten).
"""

from concurrent.futures import ThreadPoolExecutor

import torch

from artlite.decode.design import SLOTS
from artlite.gguf_model import GGUFFile
from artlite.model_config import LfmConfig
from artlite.overlay import WeightBuffer

from ..common import Model, OverlayModel, Sampling
from ..tokenizer import BpeTokenizer, chat_prompt
from .image import LfmImage
from .overlay import KV_K, LfmOverlay
from .prefill import SIZES, LfmPrefill

# the model card's sampling (the file's general.sampling.* over it)
SAMPLING = {"temp": 0.1, "top_k": 50, "penalty_repeat": 1.1}


class NpuLfm(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the KV caches hold; ``kv_k``: their K precision ("single",
    "hilo"); ``prefill``: prompts in chunks through the prefill (else as decode steps), of the sizes ``sizes``;
    ``weights``: the weight image's buffer, allocated ahead (``WeightBuffer``)."""

    recurrent = True  # (the conv histories hold the whole sequence)
    STEP_COST = (1.0, 1.02, 1.05, 1.06)  # (decode steps at 1K: QAD-Q4_0 20.2 ms for one slot, Q4_K_M 21.3)

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, kv_k: str = "single", threads: int = 0,
                 prefill: bool = True, sizes: tuple[int, ...] = SIZES, weights: WeightBuffer | None = None):
        self.gguf = gguf
        self.overlay = LfmOverlay(gguf, max_positions=capacity, kv_k=kv_k, weights=weights)
        self.cfg = self.overlay.cfg
        self.causal = self.cfg.causal  # (d1's decision models: readouts of whole prompts, no decode steps)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.prefill = LfmPrefill(self.overlay, sizes=sizes) if prefill else None
        with ThreadPoolExecutor(1) as pool:
            build = pool.submit(self.prefill.compile) if prefill else None  # (while the weights are packed)
            self.overlay.load(threads=threads)
            for S in range(1, SLOTS + 1) if self.causal else ():  # (the steps' transactions: prompt tokens, a prompt's
                self.overlay.prepare(S, logits=False)  # last, generated ones)
                self.overlay.prepare(S, logits=True)
            if build is not None:
                build.result()  # (its error, if any)
                self.prefill.bind()
        self.overlay.ready()
        self._w_out = gguf.tensor("output_norm.weight").dequantize().float()
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
        """Forget the cached positions (the conv histories zero)."""
        self.overlay.reset()
        self.position = 0

    def _step_inputs(self, tokens: list[int]) -> tuple:
        return (self.gguf.rows("token_embd.weight", tokens),)

    def _chunk_inputs(self, tokens: list[int], *, first: bool) -> tuple:
        return (self.gguf.rows("token_embd.weight", tokens),)

    def _chunk_logits(self, token: int, n: int) -> torch.Tensor:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))


class Lfm(Model):
    """An lfm3 GGUF file: the architecture, tokenizer and chat template from its metadata. Thinking: the template opens
    a thought unless ``enable_thinking`` is false (an empty one, ``<think></think>``); a raw text gets no BOS
    (the file's ``add_bos_token``, as llama.cpp)."""

    name = "LFM3"
    thinks = True
    reasoning = ("<think>", "</think>")
    thought_first = True
    turn, user_turn = "<|im_start|>", "<|im_start|>user"

    def __init__(self, path: str, *, kv_k: str = "single"):
        if kv_k not in KV_K:
            raise ValueError(f"kv_k: {kv_k!r} (one of {', '.join(KV_K)})")
        self.gguf = GGUFFile(path)
        self.cfg = LfmConfig.from_gguf(self.gguf)
        self.weights = WeightBuffer(LfmImage(self.gguf, self.cfg))
        self.tokenizer = BpeTokenizer(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.bos = int(self.gguf.meta["tokenizer.ggml.bos_token_id"])
        self.kv_k = kv_k

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

    def load(self, *, capacity: int, prefill: bool = True) -> NpuLfm:
        return NpuLfm(self.gguf, capacity=capacity, kv_k=self.kv_k, prefill=prefill, weights=self.weights)
