"""Qwen3.5 / Qwen3.6 (GGUF architecture ``qwen35``: Gated DeltaNet and full-attention layers) on the NPU.

A dispatch runs up to 4 tokens through every layer and the LM head on one stream of the weights
(``artlite.operators.qwen35.overlay.QwenOverlay``): the projections on the main cores and their helpers (the MMUL
against exact-integer weight panels), the Gated DeltaNet steps (conv states and fp32 states in DDR) on the GDN cores,
the full attention and its KV cache on the attention cores. The host reads the tokens' embedding rows from the mapped
file and samples from the logits.

A prompt is token ids or a ``sequence.Sequence`` (token ids and embedding rows with M-RoPE positions: MiniCPM-V's
images): each cache slot's position (``QwenOverlay.rope``) comes from it, the text after it from its ``next`` on.

A prompt runs in chunks of up to 2048 tokens through the prefill (``artlite.operators.qwen35.prefill.QwenPrefill``:
layer-major, control code assembled per chunk from one library ELF; the GEMMs read the same weight image, the chunk's K /
V, GDN states and conv states land in the decode's buffers), each chunk in pieces of a few layers (dispatches of one HW
context); its last token's logits come from the LM head alone (``QwenOverlay.logits_of``). Tokens where a chunk cannot
start (a position inside a 32-position cache group, after a generated reply; fewer than 3 tokens left) run as decode
steps of up to 4 tokens.

Memory: the weights the NPU streams (the layers and the LM head: 15.95 GB of the 27B's 16.5 GB file) are
re-laid-out into a device buffer at load, at their quantization's size; nothing is written to disk. Load (27B, the
file in the page cache: 1.1 s): the buffer allocated and the image packed (C++) in a thread while the tokenizer,
the designs (found through the render manifest) and the overlay's other buffers load.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig
from artlite.operators.overlay.runtime import WeightBuffer
from artlite.operators.qwen35.image import QwenImage
from artlite.operators.qwen35.overlay import QwenOverlay
from artlite.operators.qwen35.prefill import SIZES, QwenPrefill

from ..common import Model, OverlayModel, Sampling
from .minicpmv import image_prompt
from .sequence import Sequence
from .tokenizer import QwenTokenizer, chat_prompt
from .vision import PROJECTORS, NpuVision

# the model card's sampling (thinking / instruct) when the file names none (general.sampling.*); MiniCPM-V 4.7's (one
# set): the language model of openbmb's files, marked by the M-RoPE order of its vision
# (``QwenConfig.rope_section_order`` "zyxt": llama.cpp's mark of MiniCPM-V 4.7; its general.name is generic)
SAMPLING = {True: {"temp": 1.0, "top_k": 20, "top_p": 0.95}, False: {"temp": 0.7, "top_k": 20, "top_p": 0.8}}
MINICPMV_SAMPLING = {"temp": 0.7, "top_k": 100, "top_p": 0.95, "penalty_repeat": 1.03}


class NpuQwen(OverlayModel):
    """The model on the NPU. ``capacity``: the positions the full-attention layers' caches hold; ``prefill``: prompts
    in chunks through the prefill ELFs (else as decode steps), of the sizes ``sizes`` (default ``prefill.SIZES``)."""

    def __init__(self, gguf: GGUFFile, *, capacity: int = 4096, threads: int = 0, prefill: bool = True,
                 sizes: tuple[int, ...] = SIZES, weights: WeightBuffer | None = None):
        """weights: the weight image's buffer, allocated ahead (``WeightBuffer``)."""
        self.gguf = gguf
        self.overlay = QwenOverlay(gguf, max_positions=capacity, weights=weights)
        self.cfg = self.overlay.cfg
        self.eos_token_ids = self.cfg.eos_token_ids
        self.prefill = QwenPrefill(self.overlay, sizes=sizes) if prefill else None
        with ThreadPoolExecutor(1) as pool:
            build = pool.submit(self.prefill.compile) if prefill else None  # (while the weights are packed)
            self.overlay.load(threads=threads)
            if build is not None:
                build.result()  # (its error, if any)
                self.prefill.bind()
        self.overlay.ready()  # (the weights' thread packs meanwhile)
        gguf.release()
        self.position = 0
        self.rope_next = 0  # (the M-RoPE position of the next text token)
        self._prompting = False

    @property
    def capacity(self) -> int:
        return self.overlay.max_positions

    def describe(self) -> str:
        ov = self.overlay
        return (f"{ov.image.nbytes / 1e9:.2f} GB of weights, {ov.sizes['KV'] / 1e6:.0f} MB of KV cache "
                f"({self.capacity} positions), {ov.sizes['GSTATE'] / 1e6:.0f} MB of GDN states")

    def reset(self) -> None:
        """A new sequence: no cached positions, the GDN states zeroed."""
        self.overlay.reset()
        self.position = self.rope_next = 0

    def rewind(self, n: int) -> int:
        """(all cached positions or none: the GDN layers' states hold the whole sequence and cannot go back)"""
        if n >= self.position:
            return self.position
        self.reset()
        return 0

    def _rows(self, tokens: list[int] | Sequence) -> torch.Tensor:
        """The tokens' embeddings (n, hidden) fp32: their ids' rows, a sequence's given rows."""
        if not isinstance(tokens, Sequence):
            return self.gguf.rows("token_embd.weight", tokens)
        x = torch.empty(len(tokens), self.cfg.hidden_size)
        t = ~tokens.given
        if t.any():
            x[torch.from_numpy(t)] = self.gguf.rows("token_embd.weight", tokens.ids[t])
        if tokens.given.any():
            x[torch.from_numpy(tokens.given)] = torch.from_numpy(tokens.rows)
        return x

    def _step_inputs(self, tokens: list[int] | Sequence) -> tuple:
        return (self._rows(tokens),)

    def _chunk_inputs(self, tokens: list[int] | Sequence, *, first: bool) -> tuple:
        return (self._rows(tokens),)

    def prompt(self, tokens: list[int] | Sequence, *, logits: bool = False,
               progress: Callable[[int], None] | None = None) -> torch.Tensor | None:
        """(a sequence's positions from ``rope_next``)"""
        p, n = self.position, len(tokens)
        if p + n > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        seq = tokens if isinstance(tokens, Sequence) else Sequence.text(tokens)
        self.overlay.rope[p:p + n] = self.rope_next + seq.pos
        self._prompting = True
        try:
            return super().prompt(tokens, logits=logits, progress=progress)
        finally:
            self._prompting = False
            done = self.position - p  # (a prompt that ``progress`` stopped: the text after its last cached token)
            self.rope_next = (self.rope_next + seq.next if done == n else
                              int(self.overlay.rope[self.position - 1].max()) + 1 if done else self.rope_next)

    def steps(self, tokens: list[int], *, logits: bool = True) -> torch.Tensor | None:
        """(tokens after a prompt: text from ``rope_next``)"""
        if self._prompting:
            return super().steps(tokens, logits=logits)
        p, n = self.position, len(tokens)
        if p + n <= self.capacity:
            self.overlay.rope[p:p + n] = self.rope_next + np.arange(n)[:, None]
        out = super().steps(tokens, logits=logits)
        self.rope_next += n
        return out

    def _chunk_logits(self, token: int, n: int) -> torch.Tensor:
        """(the LM head alone on the chunk's last output row)"""
        x = self.prefill.hidden(n - 1)
        return self._dispatch(lambda: self.overlay.logits_of(x))


class Qwen(Model):
    """A qwen35 model from a GGUF file (Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7's language model): the architecture,
    tokenizer and chat template from its metadata. Opening the file allocates the weight image's buffer and packs it in
    a thread (``WeightBuffer``) while the tokenizer and the designs load. ``mmproj``: MiniCPM-V's vision encoder file
    (``vision.NpuVision``: images in prompts; ``max_slices``: an image's slices at most, default the file's)."""

    name = "Qwen3.5"
    thinks = True
    reasoning = ("<think>", "</think>")
    projectors = PROJECTORS

    def __init__(self, path: str, *, mmproj: str | None = None, max_slices: int | None = None):
        self.gguf = GGUFFile(path)
        self.weights = WeightBuffer(QwenImage(self.gguf))
        self.tokenizer = QwenTokenizer(self.gguf)
        self.cfg = QwenConfig.from_gguf(self.gguf)
        self.eos_token_ids = self.cfg.eos_token_ids
        self.minicpmv = self.cfg.rope_section_order == "zyxt"
        if self.minicpmv:
            self.name = "MiniCPM-V 4.7"
        self._npu: NpuQwen | None = None
        if mmproj is not None:
            if not self.minicpmv:
                raise ValueError(f"{path}: an mmproj file goes with MiniCPM-V 4.7's language model")
            self.vision = NpuVision(mmproj, max_slices=max_slices)
            if self.vision.clip.projection_dim != self.cfg.hidden_size:
                raise ValueError(f"{mmproj}: embeddings of {self.vision.clip.projection_dim}, the language model's "
                                 f"{self.cfg.hidden_size}")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False) -> list[int] | Sequence:
        text = chat_prompt(self.gguf, messages, enable_thinking=think)
        images = [p["image"] for m in messages if isinstance(m.get("content"), list) for p in m["content"]
                  if isinstance(p, dict) and p.get("type") == "image"]
        if not images:
            return self.tokenizer.encode(text)
        if self.vision is None:
            raise ValueError(f"{self.name}: images need its vision encoder (an mmproj file)")
        seq = image_prompt(text, images, self.tokenizer.encode, self.vision.sequence)
        if self._npu is not None:  # (the encoder's hardware context ran: ``OverlayDevice.set_prologue``)
            self._npu.overlay.dev.stale = True
        return seq

    def sampling(self, *, think: bool = False) -> Sampling:
        """The file's sampling (general.sampling.*), else the model card's."""
        s = (MINICPMV_SAMPLING if self.minicpmv else SAMPLING[think]) | self.cfg.sampling
        return Sampling(temp=s["temp"], top_k=int(s["top_k"]), top_p=s["top_p"],
                        repeat_penalty=s.get("penalty_repeat", 1.0), repeat_last_n=int(s.get("penalty_last_n", 64)))

    def load(self, *, capacity: int, prefill: bool = True) -> NpuQwen:
        self._npu = NpuQwen(self.gguf, capacity=capacity, prefill=prefill, weights=self.weights)
        return self._npu
