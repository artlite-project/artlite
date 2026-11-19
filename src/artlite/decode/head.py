"""A drafter's head on its model's LM head (qwen35's MTP and DFlash2 drafters, Spark-X's DSpark): the first ``cols``
rows of each column of the model image's "lm" region (``head_region``), which the model's image lays out so: per column
its share of the head's fixed tokens (``head_tokens``: the special tokens and the first ids of the vocabulary, the
tokenizer's BPE merges, a frequency order), then ``TABLE_ROWS`` / 8 rows of a table of the sequence's own tokens past
them (``HeadTable``: the most recently seen; rows that repeat other rows of the region, the drafter's alone), then the
column's other tokens (``head_rows``). The model's logits come back in id order through the LM rows of the ids
(``lm_order``, ``LogitsOrder``)."""

import itertools
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable

import numpy as np

from ..gguf_model import GGUFFile
from ..operators.design import COLS
from .image import Block, Region
from .protocol import C_LEN

HEAD_ROWS = 65536  # a drafter head's fixed rows: the special tokens and the first ids of the vocabulary
TABLE_ROWS = 512  # its rows of the sequence's own tokens past them (``HeadTable.see``), 64 per column
DFLASH_HEAD_ROWS = 80000  # a DFlash2 drafter's: one pass a round, so 1.3 ms for the rare tokens that end a chain of
# drafts (code, JSON)


def head_tokens(gguf: GGUFFile, rows: int = HEAD_ROWS, table: int = TABLE_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """A drafter head's fixed tokens (rows,) and its table's before any sequence (table,), ascending: the control and
    user-defined tokens (GGUF token types 3, 4), the first other ids to fill the rows, then the next ones (never the
    unused ids, type 5)."""
    types = np.asarray(gguf.meta["tokenizer.ggml.token_type"])
    special = np.flatnonzero((types == 3) | (types == 4))
    other = np.flatnonzero((types != 3) & (types != 4) & (types != 5))
    n = rows - len(special)
    return np.union1d(special, other[:n]), other[n:n + table]


def draft_cols(n_ids: int, R: int, rows: int) -> int:
    """The drafter head's rows per column of an LM region of ``n_ids`` token rows in tiles of R rows: whole tiles and
    whole C objects, at least ``rows`` fixed ones and the table's (``TABLE_ROWS``) or as many as the ids hold (0:
    none)."""
    m = math.lcm(R, C_LEN)
    P = -(-(rows + TABLE_ROWS) // (COLS * m)) * m
    while P and COLS * P - TABLE_ROWS > n_ids:
        P -= m
    assert not P or n_ids % (COLS * C_LEN) == 0, "the LM rows of a model with a drafter head: whole C objects per column"
    return P


def head_rows(n_ids: int, fixed: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The LM rows' tokens (COLS, (n_ids + table) / COLS) of an image with a drafter head: per column its share of the
    fixed tokens, of the table's, then of the other ids."""
    other = np.setdiff1d(np.arange(n_ids), fixed)
    return np.concatenate([fixed.reshape(COLS, -1), table.reshape(COLS, -1), other.reshape(COLS, -1)], 1)


def lm_order(rows: np.ndarray, cols: int) -> np.ndarray:
    """The LM row of each token id (n_ids,) of ``head_rows``' rows: its fixed or other row (the table's repeat them;
    ``cols``: the head's rows per column)."""
    per = rows.shape[1]
    own = np.ones(per, bool)
    own[cols - TABLE_ROWS // COLS:cols] = False
    out = np.empty(per * COLS - TABLE_ROWS, np.int64)
    out[rows[:, own]] = np.arange(COLS * per).reshape(COLS, per)[:, own]
    return out


def head_region(lm: Region, cols: int) -> Region:
    """The drafter head of an LM region: its first ``cols`` rows of each column, in place."""
    b = lm.block("lm")
    assert lm.kc_group == lm.n_kc, "a row tile's K chunks consecutive: the head's rows a prefix of each column"
    assert not b.pt or cols % (2 * b.R * b.G) == 0, "whole pairs of row tiles (their shared tail tiles)"
    return Region(-1, "lm", [Block(b.name, b.tensors, b.rows[:, :cols], b.fmt, b.R, b.G, pt=b.pt)], lm.n_kc, lm.kc_group,
                  lm.tile_bytes, lm.kc, offset=lm.offset, col_stride=lm.col_stride, half=lm.half)


class HeadView:
    """A drafter head (``head_region``) of a model's ``image`` (its "lm" region laid out with ``head_rows``; ``fixed``,
    ``table``, ``repack``: the image's) of ``cols`` rows per column, for ``HeadTable``."""

    def __init__(self, image: object, cols: int):
        self.image, self.fixed, self.table = image, image.fixed, image.table
        self.vocab = COLS * cols
        self._lm = head_region(image.region(-1, "lm"), cols)

    def region(self, layer: int, name: str) -> Region:
        assert (layer, name) == (-1, "lm"), (layer, name)
        return self._lm

    def head_rows(self) -> np.ndarray:
        """The head's tokens (COLS, cols): per column its fixed tokens, then its table's (the model's LM rows)."""
        return self._lm.block("lm").rows

    def repack(self, out: np.ndarray, r: Region, b: Block, col: int, rt: int) -> list[tuple[int, int]]:
        return self.image.repack(out, r, b, col, rt)


class HeadTable:
    """A drafter head's tokens: ``ids`` (the token of each logit) and its table of the sequence's own (module docstring), of a
    ``view`` of the image (a drafter's image, ``HeadView``) for a model of ``vocab_size`` tokens; ``device``: the device that
    holds the view's weights. A model of several lanes has one table, of the tokens its lanes saw last: its rows are
    the drafter's alone (the model's logits come from none of them), so a lane's ``see`` beside another's step costs
    that step's drafter a guess at most."""

    def __init__(self, view: object, vocab_size: int, device: Callable[[], object]):
        self.view, self.device = view, device
        self.ids = view.head_rows().reshape(-1)
        self._per = view.vocab // COLS  # (a column's logits)
        self.rows = np.flatnonzero(np.arange(view.vocab) % self._per >= len(view.fixed) // COLS)  # (the table's logits)
        self._fixed = np.zeros(vocab_size, bool)
        self._fixed[view.fixed] = True
        idle = sorted(zip(self.ids[self.rows].tolist(), self.rows.tolist()), reverse=True)
        self._lru = OrderedDict(idle)  # token -> its logit, least recently seen first (idle: the last ids first)
        self._lock = threading.Lock()  # (``see`` of a model's lanes: each on a thread of its own)

    def see(self, ids: Iterable[int]) -> None:
        """Tokens of the sequence in order (an image's rows, negative: none): the table holds the ``TABLE_ROWS`` most
        recently seen ones outside the fixed rows (the least recently seen leaves), the head's rows of those that
        enter packed into the weight image (the next step's)."""
        a = np.fromiter(ids, np.int64)
        a = a[(a >= 0) & (a < len(self._fixed))]
        a = a[~self._fixed[a]]
        if not a.size:
            return
        u, last = np.unique(a[::-1], return_index=True)
        order = u[np.argsort(-last)][-len(self.rows):].tolist()  # (the last ones, by their last occurrence)
        with self._lock:
            lru = self._lru
            for t in order:
                if t in lru:
                    lru.move_to_end(t)  # (they stay: the others leave first)
            new = [t for t in order if t not in lru]
            for t in new:
                lru[t] = lru.popitem(last=False)[1]
            for t in order:
                lru.move_to_end(t)
            if new:
                self._write([lru[t] for t in new], new)

    def _write(self, rows: list[int], tokens: list[int]) -> None:
        """The head's logits ``rows`` are the ``tokens``' now: their row tiles packed again, flushed."""
        view, dev = self.view, self.device()
        r = view.region(-1, "lm")
        b = r.block("lm")
        tiles = set()
        for i, t in zip(rows, tokens, strict=True):
            c, j = divmod(i, self._per)
            self.ids[i] = b.rows[c, j] = t
            tiles.add((c, j // (b.R * b.G)))
        w = dev.view("W")
        for c, rt in sorted(tiles):
            for off, n in view.repack(w, r, b, c, rt):
                dev.flush("W", off, n)


class LogitsOrder:
    """A model's logits in id order from LM rows in the order ``order`` (the LM row of each token id: ``lm_order``),
    copied by runs of consecutive rows into a buffer of its own (valid until the next copy)."""

    def __init__(self, order: np.ndarray, slots: int):
        cuts = [0, *(np.flatnonzero(np.diff(order) != 1) + 1).tolist(), len(order)]
        self._out = np.empty((slots, len(order)), np.uint16)
        self._runs = [(a, b, int(order[a])) for a, b in itertools.pairwise(cuts)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x (S, rows) uint16 (bf16 bits) -> (S, ids) uint16."""
        S, out = x.shape[0], self._out
        for a, b, r in self._runs:
            out[:S, a:b] = x[:, r:r + b - a]
        return out[:S]
