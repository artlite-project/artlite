"""Stable Diffusion 1.5's CLIP text encoder (transformers' ``CLIPTextModel``) on the host in fp32 numpy: 12 pre-norm
transformer layers (768 wide, 12 heads, causal, quick-GELU MLP of 3072) over token and position embeddings, then the
final layer norm. A prompt's GEMMs on one OpenBLAS thread (77 tokens: ~90 ms), prompts on threads of their own (the
GIL released in them); an image's negative prompt (most often the empty one) repeats, so the last encodings are kept,
and ``prepare`` starts one ahead (the pipeline: the empty prompt's while the NPU loads).
"""

import collections
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from .weights import load_safetensors

KEEP = 8  # prompt encodings kept


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = x.mean(-1, keepdims=True)
    d = x - m
    return d / np.sqrt((d * d).mean(-1, keepdims=True) + eps) * w + b


class ClipText:
    """The CLIP text encoder: ``self(ids)``, ids (B, 77) int64 -> last hidden states (B, 77, 768) fp32."""

    def __init__(self, directory: str):
        with open(os.path.join(directory, "config.json")) as f:
            self.cfg = json.load(f)
        self.w = {k: np.asarray(v, np.float32) for k, v in
                  load_safetensors(os.path.join(directory, "model.safetensors")).items()}
        self._kept: collections.OrderedDict[tuple[int, ...], Future] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(2)

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        return np.stack([f.result() for f in self.prepare(ids)])

    def prepare(self, ids: np.ndarray) -> list[Future]:
        """The encodings of prompts' ids (B, 77): kept ones, others started on the pool."""
        out = []
        with self._lock:
            for row in np.asarray(ids):
                key = tuple(int(t) for t in row)
                f = self._kept.get(key)
                if f is None:
                    f = self._kept[key] = self._pool.submit(self.encode, np.asarray(key, np.int64))
                    if len(self._kept) > KEEP:
                        self._kept.popitem(last=False)
                else:
                    self._kept.move_to_end(key)
                out.append(f)
        return out

    def _linear(self, p: str, x: np.ndarray) -> np.ndarray:
        return x @ self.w[p + "weight"].T + self.w[p + "bias"]

    def _norm(self, p: str, x: np.ndarray) -> np.ndarray:
        return layer_norm(x, self.w[p + "weight"], self.w[p + "bias"])

    def encode(self, ids: np.ndarray) -> np.ndarray:
        """One prompt's token ids (n,) -> its last hidden states (n, 768)."""
        w, heads, pre = self.w, self.cfg["num_attention_heads"], "text_model."
        n = len(ids)
        h = w[pre + "embeddings.token_embedding.weight"][ids] + w[pre + "embeddings.position_embedding.weight"][:n]
        c = h.shape[1]
        d = c // heads
        future = np.triu(np.ones((n, n), bool), 1)
        for i in range(self.cfg["num_hidden_layers"]):
            p = f"{pre}encoder.layers.{i}."
            x = self._norm(p + "layer_norm1.", h)
            a = p + "self_attn."
            q, k, v = (self._linear(a + f"{r}_proj.", x).reshape(n, heads, d).transpose(1, 0, 2) for r in "qkv")
            s = q @ k.transpose(0, 2, 1) * np.float32(d ** -0.5)
            s[:, future] = -np.inf
            s = np.exp(s - s.max(-1, keepdims=True))
            o = (s / s.sum(-1, keepdims=True)) @ v
            h = h + self._linear(a + "out_proj.", o.transpose(1, 0, 2).reshape(n, c))
            x = self._linear(p + "mlp.fc1.", self._norm(p + "layer_norm2.", h))
            h = h + self._linear(p + "mlp.fc2.", x / (1 + np.exp(-1.702 * x)))
        return self._norm(pre + "final_layer_norm.", h)
