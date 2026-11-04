"""A prompt's output rows (``NpuModel.outputs``: the last layer's, output-normed) turned on the host into what
llama.cpp's embedding and rerank APIs return: a pooling of them (llama.cpp's ``--pooling``: none, mean, cls, last,
rank), the embeddings' normalizations (``embd_normalize``), a reranker's score through its classifier head
(``cls.output``) and a classifier's label probabilities (vLLM's ``/classify``: ``classifier.output_labels``), and the
prompts they take (``Readouts``: what a model file's metadata gives them). One request's rows are a few thousand
values: numpy, no NPU dispatch (an LM-head-sized one would cost milliseconds for microseconds of host work)."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from artlite.gguf_model import GGUFFile

POOLING = ("none", "mean", "cls", "last", "rank")  # (a GGUF's ``pooling_type``: llama.cpp's LLAMA_POOLING_TYPE_*)


def pooling_of(meta: Mapping[str, object], arch: str) -> str | None:
    """A model file's pooling (``{arch}.pooling_type``), or None: it has none (a generative model's)."""
    v = meta.get(f"{arch}.pooling_type")
    return POOLING[int(v)] if v is not None and 0 <= int(v) < len(POOLING) else None


def readout_model(meta: Mapping[str, object], arch: str) -> bool:
    """Whether a model file is read out, not generated from: an embedding or rerank model (``pooling_of``) or a decision
    model (``{arch}.decision.type``). Its prompts are short and prefill only, so its prefill takes the precise variants
    of its operators (``NpuModel``'s ``precise``)."""
    return pooling_of(meta, arch) is not None or meta.get(f"{arch}.decision.type") is not None


def rows_of(pooling: str, n: int) -> np.ndarray:
    """The rows of a prompt of n tokens a pooling reads (rank: a causal model's last token, as llama.cpp's)."""
    if pooling in ("none", "mean"):
        return np.arange(n)
    return np.array([0 if pooling == "cls" else n - 1])


def pool(rows: np.ndarray, pooling: str) -> np.ndarray:
    """``rows_of``' rows (n, hidden) pooled: (hidden,), or every row (pooling none)."""
    if pooling == "none":
        return rows
    return rows.mean(0) if pooling == "mean" else rows[0]


def normalize(x: np.ndarray, p: int) -> np.ndarray:
    """llama.cpp's ``common_embd_normalize`` of each row (the last axis): p -1 none, 0 the largest magnitude to 32760
    (int16's range), 2 Euclidean, any other p the p-norm; a zero norm gives zeros."""
    x = np.asarray(x, np.float64)
    if p == -1:
        return x.astype(np.float32)
    if p == 0:
        s = np.abs(x).max(-1, keepdims=True) / 32760.0
    elif p == 2:
        s = np.sqrt((x * x).sum(-1, keepdims=True))
    else:
        s = (np.abs(x) ** p).sum(-1, keepdims=True) ** (1.0 / p)
    return (x * np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)).astype(np.float32)


@dataclass(frozen=True)
class RankHead:
    """A reranker's classifier head on the last row (llama.cpp's rank pooling on a causal model): ``w`` (labels,
    hidden) and ``b`` (labels,) of ``cls.output``, then a softmax over the labels (Qwen3's rerankers: [yes, no]); the
    score is the first output. ``labels``: the outputs' names (``classifier.output_labels``; else their indices)."""

    w: np.ndarray
    b: np.ndarray | None
    softmax: bool
    labels: tuple[str, ...] = ()

    def logits(self, h: np.ndarray) -> np.ndarray:
        return self.w.astype(np.float64) @ np.asarray(h, np.float64) + (0.0 if self.b is None else self.b)

    def score(self, h: np.ndarray) -> float:
        z = self.logits(h)
        if self.softmax:
            z = np.exp(z - z.max())
            z /= z.sum()
        return float(z[0])

    def probs(self, h: np.ndarray) -> np.ndarray:
        """The labels' probabilities (vLLM's classification: a softmax, a sigmoid of a single output)."""
        z = self.logits(h)
        if len(z) == 1:
            return 1 / (1 + np.exp(-z))
        z = np.exp(z - z.max())
        return z / z.sum()

    def label(self, i: int) -> str:
        return self.labels[i] if i < len(self.labels) else str(i)


def _rows(g: GGUFFile, name: str) -> np.ndarray:
    """A tensor's rows as fp32 (rows, cols): small tensors, once."""
    return np.asarray(g.rows(name, np.arange(g.tensor(name).rows)), np.float32)


@dataclass(frozen=True)
class Readouts:
    """What a model's readouts take from its file (``of``): its ``pooling`` (None: none in the file), its classifier
    ``head`` (``cls.output``: rank pooling), the rerank prompt (``tokenizer.chat_template.rerank``, with ``{query}`` and
    ``{document}``; without one llama.cpp's [BOS] query [EOS] [SEP] document [EOS] by the tokenizer's flags) and whether
    an embedding's text ends with EOS (``add_eos``: Qwen3-Embedding's)."""

    pooling: str | None = None
    head: RankHead | None = None
    rerank_template: str | None = None
    add_eos: bool = False
    add_sep: bool = False
    eos: int | None = None  # (SEP where the vocabulary has no EOS)
    sep: int | None = None

    @classmethod
    def of(cls, path: str) -> "Readouts":
        g = GGUFFile(path)
        m, arch = g.meta, g.arch
        head = None
        if "cls.output.weight" in g:
            if "cls.weight" in g:
                raise NotImplementedError(f"{path}: a classifier head with a dense layer before cls.output (BERT's)")
            key = f"{arch}.classifier.output_labels"
            head = RankHead(_rows(g, "cls.output.weight"),
                            _rows(g, "cls.output.bias").reshape(-1) if "cls.output.bias" in g else None,
                            softmax=arch in ("qwen3", "qwen3vl"),
                            labels=tuple(g.strings(key)) if g.has_strings(key) else ())
        sep = m.get("tokenizer.ggml.seperator_token_id", m.get("tokenizer.ggml.separator_token_id"))
        eos = m.get("tokenizer.ggml.eos_token_id", sep)
        return cls(pooling=pooling_of(m, arch), head=head,
                   rerank_template=str(m["tokenizer.chat_template.rerank"]) if "tokenizer.chat_template.rerank" in m
                   else None, add_eos=bool(m.get("tokenizer.ggml.add_eos_token", False)),
                   add_sep=bool(m.get("tokenizer.ggml.add_sep_token", False)),
                   eos=None if eos is None else int(eos), sep=None if sep is None else int(sep))

    def embedding_ids(self, encode: Callable[[str], list[int]], text: str) -> list[int]:
        """An embedding input's tokens: ``encode`` (the model's: after its BOS, if any) and EOS where the file adds it."""
        ids = encode(text)
        return ids + [self.eos] if self.add_eos and self.eos is not None else ids

    def rerank_ids(self, tokenize: Callable[[str], list[int]], bos: list[int], query: str, document: str) -> list[int]:
        """A rerank pair's prompt: ``tokenize`` (no BOS; special tokens parsed), ``bos``: the model's BOS (or none)."""
        if self.rerank_template is not None:  # (one pass over the template: the texts' own braces stay)
            return tokenize(re.sub(r"\{query\}|\{document\}", lambda m: query if m[0] == "{query}" else document,
                                   self.rerank_template))
        eos = [self.eos] if self.add_eos and self.eos is not None else []
        sep = [self.sep] if self.add_sep and self.sep is not None else []
        return [*bos, *tokenize(query), *eos, *sep, *tokenize(document), *eos]


__all__ = ["POOLING", "RankHead", "Readouts", "normalize", "pool", "pooling_of", "rows_of"]
