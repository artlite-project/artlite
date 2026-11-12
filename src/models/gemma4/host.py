"""What the host computes for Gemma 4 on the NPU: the scaled token embeddings and the per-layer token rows (E2B / E4B:
the per-layer inputs then come from the NPU's projection, ``gemma4.overlay``, ``gemma4.prefill``), from the GGUF
file's tensors (llama.cpp's ``src/models/gemma4.cpp``).

``GemmaHost`` reads matrices dequantized to fp32 on first use, embedding rows by a lookup in the mapped file
(``GGUFFile.rows``). The fp32 reference of the whole model (tests/gemma4/cpu.py) derives from it.
"""

import math

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import GemmaConfig


class GemmaHost:
    """The tensors of a GGUF file the host reads, and the inputs it computes from them."""

    def __init__(self, gguf: GGUFFile, *, threads: int = 8):
        self.gguf = gguf
        self.cfg = GemmaConfig.from_gguf(gguf)
        self._mats: dict[str, np.ndarray] = {}
        self._threads = threads

    def mat(self, name: str) -> np.ndarray:
        """The dequantized matrix (rows, cols) fp32, or vector (n,)."""
        if name not in self._mats:
            t = self.gguf.tensor(name)
            w = t.dequantize()
            self._mats[name] = w.reshape(-1) if t.rows == 1 else w
        return self._mats[name]

    def embed(self, tokens: list[int]) -> np.ndarray:
        """(T, hidden) fp32: the token embeddings * sqrt(hidden)."""
        return self.gguf.rows("token_embd.weight", tokens) * np.float32(math.sqrt(self.cfg.hidden_size))

    def layer_rows(self, tokens: list[int], layers: int = 0) -> np.ndarray:
        """(T, layers, per_layer_dim) fp32: the per-layer token embedding rows times sqrt(per_layer_dim / 2), the
        per-layer inputs' token part (the decode's F_PLI and the prefill's ``prefill_ple`` add it to the projection's
        normed part); ``layers``: of the first that many layers only (default all)."""
        d = self.cfg.per_layer_dim
        rows = self.gguf.rows("per_layer_token_embd.weight", tokens, layers * d)
        return (rows * np.float32(math.sqrt(d / 2))).reshape(len(tokens), -1, d)
