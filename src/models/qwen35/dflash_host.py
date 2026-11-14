"""The DFlash2 drafter's host side (``dflash.DFlashOverlay`` runs the dispatches): the selector's greedy walk over a block's
logits and gates, and the rotary angles of a block's or a chunk's positions."""

import numpy as np

from artlite.gguf_model import GGUFFile
from artlite.model_config import DFlashConfig

from ..sampling import top_logits


def angles(cfg: DFlashConfig, positions: np.ndarray) -> np.ndarray:
    """[cos | sin] (n, head_dim) fp32 of the angles of ``positions``."""
    th = cfg.rope_theta(positions).astype(np.float64)
    return np.concatenate([np.cos(th), np.sin(th)], 1).astype(np.float32)


class Selector:
    """DFlash2's selector on the host (llama.cpp's ``build_dflash2_selector`` and its greedy walk): the drafts of a block from its logits and
    gates. Of the candidates' scores only the chosen predecessor's row is needed: per position the rows of its top_k candidates in
    ``selector_successor`` and of the predecessor in ``selector_predecessor``, read for all positions at once (the predecessors': the
    anchor's and every candidate's but the last position's)."""

    TABLES = ("selector_successor.weight", "selector_predecessor.weight")
    SIBLING = 2.0  # a sibling's probability against this times its chain's (``walk``)

    def __init__(self, gguf: GGUFFile, cfg: DFlashConfig):
        self.gguf, self.top_k = gguf, cfg.top_k
        self.sibling: list[int] | None = None

    def resident(self) -> None:
        """The tables' pages read ahead (after the drafter's load released the file's): ~200 scattered rows a round."""
        for name in self.TABLES:
            self.gguf.willneed(name)

    def walk(self, logits: np.ndarray, gates: np.ndarray, anchor: int, ids: np.ndarray, n: int, stop: set[int], *,
             sibling: int = 0) -> list[int]:
        """Up to n drafts after the anchor token (logits (block, vocab): rows of the tokens ``ids``), ending with a token of ``stop``.
        ``sibling`` > 0: also ``self.sibling``, a branch of that many tokens from the first position's second candidate on (its
        walk) when that candidate's probability exceeds ``SIBLING`` times the n drafts' (the probabilities: the lattice's softmax
        over a position's candidates), else None: the drafts that a step gives their slots are the last ones, kept with the
        chain's probability, the branch where the first draft is not (prose: a third of first drafts at 32K, of which the
        second candidate is right in a third to a half)."""
        self.sibling = None
        if n <= 0:
            return []
        k = self.top_k
        top = [top_logits(logits[i], k, rows=ids) for i in range(1, n + 1)]
        c = np.stack([c for _, c in top])
        succ = self.gguf.rows(self.TABLES[0], c.reshape(-1), prefetch=False).reshape(n, k, -1)
        pred = self.gguf.rows(self.TABLES[1], np.concatenate([[anchor], c[:-1].reshape(-1)]), prefetch=False)
        out: list[int] = []
        j = 0  # (the predecessor's row)
        chain, p2, b = 1.0, 0.0, 0
        for i in range(n):
            sc = succ[i] @ (pred[j] * gates[i + 1]) + top[i][0]
            a = int(np.argmax(sc))
            if sibling:
                e = np.exp(sc - sc[a])
                chain /= e.sum()
                if i == 0:
                    e[a] = 0.0
                    b = int(np.argmax(e))
                    p2 = e[b] * chain
            out.append(int(c[i, a]))
            if out[-1] in stop:
                break
            j = 1 + i * k + a
        if sibling and len(out) == n and p2 > self.SIBLING * chain:
            self.sibling, j = [int(c[0, b])], 1 + b
            for i in range(1, sibling):
                a = int(np.argmax(succ[i] @ (pred[j] * gates[i + 1]) + top[i][0]))
                self.sibling.append(int(c[i, a]))
                j = 1 + i * k + a
        return out
