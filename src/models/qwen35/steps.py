"""A qwen35 step's transaction and the records of the BDs that its position patches.

``Step`` is the built transaction of a step of S slots with the words that follow the position: the attentions' cache and
ring reads and record drains (``decode.attn256_steps.AttnStep``'s) and a chunkwise GDN core's state panels (their GSTATE
half follows the block of the step's position): ``StatePanel``, as arrays in ``PanelTable``. Each record is written where
the transfer is queued (``decode.attn256_steps``, ``gdn_steps``) and read where the position patches it, by name.
"""

import dataclasses
from typing import NamedTuple

import numpy as np

from artlite.decode.attn256_steps import AttnStep


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
class Step(AttnStep):
    """The transaction of a step of S slots (``AttnStep``) and its chunkwise GDN cores' state panels."""

    panels: list[StatePanel] = dataclasses.field(default_factory=list)
    table: PanelTable | None = None  # (``panels`` as arrays, at the first patch)
    half: int = -1  # (the GSTATE half the panels are patched for: ``gdn_steps.GdnTransfers.patch_panels``)
