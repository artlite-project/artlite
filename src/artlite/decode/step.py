"""What an attention model's step transaction keeps for its per-step patches (``decode.attn_overlay``): the plan of a
step (``StepPlan``: its slots and sub-steps, its transaction, the BD words that follow the position as records) and
those records as arrays (``PatchTables``), built once per transaction at its first patch.

A step's transaction is built once per pattern of sub-steps and patched per step: each column's range of each layer's
cache (A: address, length), a chain end's tail objects (A: the ring slots of the sub-step's open records, or the V
panels of the first after a prefill) and the record drains (D: address, length).
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..overlay import Transaction


class Fill(NamedTuple):
    """A column's range of a layer's cache in a sub-step (A): the BD's first word in the transaction."""

    at: int
    layer: int
    column: int
    sub: int


class Tail(NamedTuple):
    """A chain end's tail object ``obj`` of a layer in a sub-step (A: a ring slot of an open record, or its V panels)."""

    at: int
    layer: int
    column: int
    sub: int
    obj: int


class Rec(NamedTuple):
    """The ``i``-th record drain of a layer's KV head in a sub-step (D): the BD handle (``Sequence.drain``)."""

    handle: list[int]
    layer: int
    head: int
    sub: int
    i: int


@dataclass
class StepPlan:
    """The transaction of a step of S slots (its attentions' sub-steps: ``subs``) and its words that follow the
    position. ``tok``: its token objects' offset in ACT."""

    S: int
    subs: tuple
    tr: Transaction | None = None
    tok: int = 0
    fills: list[Fill] = field(default_factory=list)
    tails: list[Tail] = field(default_factory=list)
    recs: list[Rec] = field(default_factory=list)
    patch: "PatchTables | None" = None


@dataclass(frozen=True)
class FillTable:
    """The cache fills as arrays: BD word index, cache kind, sub-step, the column's place in its chain, the byte offset in
    KV of group 0 of the KV head's cache."""

    at: np.ndarray
    kind: np.ndarray
    sub: np.ndarray
    place: np.ndarray
    base: np.ndarray


@dataclass(frozen=True)
class TailTable:
    """The tail fills as arrays: BD word index, cache kind, sub-step, tail object, the byte offsets in KV of group 0 of the
    KV head's cache and of its ring, whether the layer owns its cache, and its layer's index in ``layers``."""

    at: np.ndarray
    kind: np.ndarray
    sub: np.ndarray
    obj: np.ndarray
    base: np.ndarray
    ring: np.ndarray
    own: np.ndarray
    layer: np.ndarray
    layers: tuple[int, ...]


@dataclass(frozen=True)
class RecTable:
    """The record drains as arrays: BD word index, cache kind, sub-step, the record's index, the byte offsets in KV of group
    0 of the KV head's cache and of its ring (columns 0, 1)."""

    at: np.ndarray
    kind: np.ndarray
    sub: np.ndarray
    i: np.ndarray
    base: np.ndarray


@dataclass(frozen=True)
class PatchTables:
    fills: FillTable
    tails: TailTable
    recs: RecTable


class PositionWords(NamedTuple):
    """What the token objects of a step at p give its patches, per cache kind and sub-step ``u``: ``pu[kind, u]`` its first
    position and slots; ``fo[kind, u, j]`` the A fill of chain column j (first object, objects); ``ro[kind, u, i]`` its
    i-th record (the base, offset and bytes of its drain)."""

    pu: np.ndarray
    fo: np.ndarray
    ro: np.ndarray
