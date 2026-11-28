"""What an attention model's step transaction keeps for its per-step patches (``decode.attn_overlay``): the plan of a
step (``StepPlan``: its slots and sub-steps, its transaction, the BD words that follow the position as records) and
those records as arrays (``PatchTables``), built once per transaction at its first patch.

A step's transaction is built once per pattern of sub-steps and patched per step: each column's range of each layer's
cache (A: address, length), a chain end's tail objects (A: the ring slots of the sub-step's open records, or the V
panels of the first after a prefill), the record drains (D: address, length) and the transfers of a part's sequence's
own objects (``PartBD``: address).

A step's slots may be several sequences' (``Part``: the device holds each one's caches, ``ModelOverlay.select``): a
sequence's slots form separate sub-steps, so a sub-step's patches take its sequence's buffers and its own position,
and the transaction is the one a single sequence builds for the same pattern of sub-steps.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..overlay import Transaction


class Part(NamedTuple):
    """A sequence's slots of a step: ``n`` consecutive positions from ``position`` on of sequence ``seq`` (a step's
    parts take its slots in order)."""

    seq: int
    position: int
    n: int


def slots_of(parts: tuple[Part, ...]) -> tuple[list[int], list[int], list[bool]]:
    """Per slot of a step of ``parts``: its sequence, its position, whether it is its part's first."""
    return ([q.seq for q in parts for _ in range(q.n)], [q.position + i for q in parts for i in range(q.n)],
            [i == 0 for q in parts for i in range(q.n)])


class Fill(NamedTuple):
    """A column's range of a layer's cache in a sub-step (A): the BD's first word in the transaction; ``piece``: of a ring's
    two fills (``CacheKind.fills``)."""

    at: int
    layer: int
    column: int
    sub: int
    piece: int = 0


class Tail(NamedTuple):
    """A chain end's tail object ``obj`` of a layer in a sub-step (A: a ring slot of an open record, or its V panels)."""

    at: int
    layer: int
    column: int
    sub: int
    obj: int


class PartBD(NamedTuple):
    """A BD in the KV buffer of a step part's sequence at an offset that follows no position (a conv pass's region,
    LFM3's): its first word in the transaction, the part (its index in the step's parts), the offset, its bytes."""

    at: int
    part: int
    offset: int
    nbytes: int


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
    part_bds: list[PartBD] = field(default_factory=list)
    patch: "PatchTables | None" = None


@dataclass(frozen=True)
class FillTable:
    """The cache fills as arrays: BD word index, cache kind, sub-step, the column's place in its chain, the fill's piece,
    the byte offset in KV of group 0 of the KV head's cache."""

    at: np.ndarray
    kind: np.ndarray
    sub: np.ndarray
    place: np.ndarray
    piece: np.ndarray
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
class PartTable:
    """The ``PartBD``s as arrays: BD word index, part, byte offset, bytes."""

    at: np.ndarray
    part: np.ndarray
    offset: np.ndarray
    nbytes: np.ndarray


@dataclass(frozen=True)
class PatchTables:
    fills: FillTable
    tails: TailTable
    recs: RecTable
    parts: PartTable


class PositionWords(NamedTuple):
    """What the token objects of a step give its patches, per cache kind and sub-step ``u``: ``pu[kind, u]`` its first
    position and slots; ``fo[kind, u, j, f]`` the A fill f of chain column j (first object, objects:
    ``attn_overlay.CacheKind.fills``); ``ro[kind, u, i]`` its i-th record (the base, offset and bytes of its drain);
    ``su[kind, u]`` its sequence; ``first[kind, u]`` whether it is its sequence's first sub-step of the step."""

    pu: np.ndarray
    fo: np.ndarray
    ro: np.ndarray
    su: np.ndarray
    first: np.ndarray
