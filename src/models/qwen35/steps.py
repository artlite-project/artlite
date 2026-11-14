"""A qwen35 step's transaction and the records of the BDs that its position patches.

``Step`` is the built transaction of a step of S slots (its attentions' sub-steps ``subs``: first slot, slots, how the
chains divide their caches' groups) with the words that follow the position:

- the attentions' KV cache and ring reads (A: their address and length): ``Fill`` (the cache; ``FILLS``: the whole cache,
  the sub-step's group, a split column's run) and ``Tail`` (a ring's open record);
- the record drains (C: their address): ``Drain``;
- a chunkwise GDN core's state panels (their GSTATE half follows the block of the step's position):
  ``StatePanel``, as arrays in ``PanelTable``.

Each record is written where the transfer is queued (``attn_steps``, ``gdn_steps``) and read where the position patches it,
by name.
"""

import dataclasses
from typing import NamedTuple

import numpy as np

from artlite.overlay import Transaction

FILLS = ("all", "new", "run")  # a step's cache fills: the whole cache, the sub-step's group, a split column's run


class Fill(NamedTuple):
    """A cache fill: its BD word index, the layer, the column, the sub-step and which groups (``FILLS``)."""

    at: int
    layer: int
    column: int
    sub: int
    kind: str


class Tail(NamedTuple):
    """A tail fill (a ring's open record): its BD word index, the layer, the column, the sub-step, the tail object."""

    at: int
    layer: int
    column: int
    sub: int
    tail: int


class Drain(NamedTuple):
    """A record drain: its BD handle, the layer, the column, the sub-step and which of the sub-step's drains it is (a
    chain's first column's K records 0, 1; its last column's V panels and rows of record 0, then of record 1)."""

    handle: list[int]
    layer: int
    column: int
    sub: int
    index: int


class StatePanel(NamedTuple):
    """A chunkwise core's state transfer: its BD word index (a drain: its handle), its GSTATE offset in a half, its bytes
    and whether it is a drain."""

    at: int | list[int]
    offset: int
    nbytes: int
    drain: bool


@dataclasses.dataclass(eq=False)
class PanelTable:
    """``StatePanel``s as arrays: BD word index, offset, bytes, drain, and the transaction's cache lines they patch."""

    at: np.ndarray
    offset: np.ndarray
    nbytes: np.ndarray
    drain: np.ndarray
    lines: np.ndarray

    @classmethod
    def of(cls, panels: list[StatePanel]) -> "PanelTable":
        at = np.array([p.at[0] if p.drain else p.at for p in panels], np.int64)
        return cls(at, *(np.array([p[k] for p in panels]) for k in (1, 2, 3)),
                   np.unique(np.concatenate([4 * at, 4 * at + 8]) & ~63))


@dataclasses.dataclass(eq=False)
class Step:
    """The transaction of a step of S slots (``b``: which of its two buffers, set by ``ModelOverlay._transaction``)."""

    S: int
    subs: tuple[tuple[int, int, int], ...]
    tr: Transaction | None = None
    b: int = 0
    fills: list[Fill] = dataclasses.field(default_factory=list)
    tails: list[Tail] = dataclasses.field(default_factory=list)
    drains: list[Drain] = dataclasses.field(default_factory=list)
    patch: dict[str, np.ndarray] | None = None  # (``attn_steps.patch_tables``)
    panels: list[StatePanel] = dataclasses.field(default_factory=list)
    table: PanelTable | None = None  # (``panels`` as arrays, at the first patch)
    half: int = -1  # (the GSTATE half the panels are patched for: ``gdn_steps.GdnTransfers.patch_panels``)
