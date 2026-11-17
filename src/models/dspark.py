"""A DSpark drafter's host side (XHToken's Spark-X2.6 DSpark files: GGUF ``dflash`` with a Markov head; their fork's
``common/speculative.cpp``): the Markov chain and the confidences of a block's drafts from its logits and rows (the NPU's:
``models.spark.dspark``).

Greedy, as the fork's default: prev = the anchor; per row i of the block the drafts' column col_i = logits_i + w2 w1[prev]
over the head's rows (``markov_w1`` / ``markov_w2``: vocab x rank), draft_i = argmax col_i, conf_i = sigmoid(conf .
[h_i; w1[prev]] + b) (``conf_proj``; h_i the row normed by the drafter's output norm), prev = draft_i. The bias is taken
over the ``TOP_K`` best logits of a row only (w2's rows of those: 16 KB a row, read in place from the mapped file; the whole
vocabulary's would be 64 MB a row): within the noise of the whole vocabulary's on forced texts, where the top 16 costs
2-3% of the tokens per round."""

import numpy as np

from artlite.gguf_model import GGUFFile

from .sampling import top_logits

TOP_K = 32


class MarkovChain:
    """The Markov head and confidence of a DSpark file ``gguf`` (module docstring)."""

    TABLES = ("markov_w1.weight", "markov_w2.weight")

    def __init__(self, gguf: GGUFFile):
        self.gguf = gguf
        self.conf = np.asarray(gguf.tensor("conf_proj.weight").dequantize(), np.float32).reshape(-1)
        self.conf_b = float(np.asarray(gguf.tensor("conf_proj.bias").dequantize()).reshape(-1)[0])
        self.rank = gguf.tensor(self.TABLES[0]).cols

    def resident(self) -> None:
        """The tables' pages read ahead (after the drafter's load released the file's): ~230 scattered rows a round."""
        for name in self.TABLES:
            self.gguf.willneed(name)

    def walk(self, logits: np.ndarray, ids: np.ndarray, h: np.ndarray, anchor: int, n: int,
             stop: set[int] | frozenset[int] = frozenset()) -> tuple[list[int], np.ndarray]:
        """Up to n drafts after the anchor token (logits (block, rows) bf16 of the tokens ``ids``, h (block, hidden) the
        rows normed), ending with a token of ``stop``, and their confidences (fp32)."""
        drafts: list[int] = []
        conf = np.zeros(n, np.float32)
        prev = anchor
        hn = h.shape[1]
        for i in range(n):
            v, c = top_logits(logits[i], TOP_K, rows=ids)
            w1p = self.gguf.rows(self.TABLES[0], [prev], prefetch=False)[0]
            w2c = self.gguf.rows(self.TABLES[1], c, prefetch=False)
            col = v + w2c @ w1p
            conf[i] = 1 / (1 + np.exp(-(self.conf[:hn] @ h[i] + self.conf[hn:] @ w1p + self.conf_b)))
            prev = int(c[int(np.argmax(col))])
            drafts.append(prev)
            if prev in stop:
                return drafts, conf[:i + 1]
        return drafts, conf
