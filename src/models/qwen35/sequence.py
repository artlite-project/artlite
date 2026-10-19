"""A prompt of token ids and embedding rows with M-RoPE positions (images in a qwen35 language model).

``Sequence``: per token its id (``MEDIA`` where only an embedding row stands: an image's patches), whether it takes a
given embedding row instead of its id's (``given``; the rows in ``rows``, in order) and its position (T, H, W) relative
to the sequence's start; ``next``: the position of the text after it. Text from position 0 has the positions of its
indices; an image's take fewer positions than tokens (``models.qwen35.minicpmv``). Sequences concatenate (``+``, a
list of token ids as text) and slice like lists (a slice's positions relative to its first token's T, so that the
slices of a sequence run one after another as the whole), so the model's prompt path runs them in chunks and steps as
token lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEDIA = -1  # the id of a token that only an embedding row stands for


@dataclass(frozen=True)
class Sequence:
    ids: np.ndarray  # (n,) int64
    given: np.ndarray  # (n,) bool
    rows: np.ndarray  # (given.sum(), hidden) float32
    pos: np.ndarray  # (n, 3) int64
    next: int

    @classmethod
    def text(cls, ids: list[int] | np.ndarray, hidden: int = 0) -> Sequence:
        n = len(ids)
        return cls(np.asarray(ids, np.int64).reshape(n), np.zeros(n, bool), np.zeros((0, hidden), np.float32),
                   np.repeat(np.arange(n, dtype=np.int64)[:, None], 3, 1), n)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int | slice) -> int | Sequence:
        if not isinstance(i, slice):
            return int(self.ids[i])
        a, b, step = i.indices(len(self))
        assert step == 1, "contiguous slices"
        b = max(a, b)
        at = lambda k: int(self.pos[k, 0]) if k < len(self) else self.next
        r0, g, base = int(self.given[:a].sum()), self.given[a:b], at(a)
        return Sequence(self.ids[a:b], g, self.rows[r0:r0 + int(g.sum())], self.pos[a:b] - base, at(b) - base)

    def __add__(self, other: Sequence | list[int]) -> Sequence:
        o = other if isinstance(other, Sequence) else Sequence.text(other, self.rows.shape[1])
        hidden = max(self.rows.shape[1], o.rows.shape[1])
        rows = [r.reshape(-1, hidden) if r.size else np.zeros((0, hidden), np.float32) for r in (self.rows, o.rows)]
        return Sequence(np.concatenate([self.ids, o.ids]), np.concatenate([self.given, o.given]),
                        np.concatenate(rows), np.concatenate([self.pos, o.pos + self.next]), self.next + o.next)

    def __radd__(self, other: list[int]) -> Sequence:
        return Sequence.text(other, self.rows.shape[1]) + self
