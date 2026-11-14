"""What a drafter adds to the model's chunk prefill (``prefill.QwenPrefill``): its cache's records of the chunk's positions.

``DrafterPrefill`` is the protocol of ``DraftPrefill`` (an MTP drafter's records: ``draft_prefill``) and ``DFlashPrefill`` (the
DFlash2 drafter's: ``dflash_prefill``); the prefill holds its drafters and calls each at the same places of a chunk:

- the buffers it adds (``rows``: bytes per token; ``buffers``, ``external``), its static contents (``static``);
- items of the chunk's runlist: at its start (``start_items``), at the start of a layer (``tap_items``), at its end
  (``end_items``), their scratchpad parameters (``params``);
- host work around the chunk (``before``, ``after``) and its time (``seconds``).

A drafter leaves out what it does not need (the defaults).
"""

import abc
from collections.abc import Callable
from typing import Any

Item = tuple  # a runlist item: (operator, buffer slices ...)
Tok = Callable[..., str]  # ``tok(buf, j, op, i=, row=, extra=)``: the prefill's sub-chunk slice of a token buffer
Slice = Callable[..., str]  # ``sl(buf, off, op, i)``: a slice of a buffer by its byte offset


class DrafterPrefill(abc.ABC):
    """A drafter's part of a chunk (module docstring)."""

    @abc.abstractmethod
    def rows(self) -> dict[str, int]:
        """Bytes per token of the token buffers it adds."""

    @abc.abstractmethod
    def buffers(self) -> dict[str, Any]:
        """The drafter's device buffers by the names its items use."""

    @abc.abstractmethod
    def external(self) -> set[str]:
        """The names of ``buffers`` before the drafter is loaded."""

    @abc.abstractmethod
    def static(self, dev: Any, ov: Any) -> None:
        """Its static contents in the prefill's buffers on ``dev`` (the model's overlay ``ov``)."""

    @abc.abstractmethod
    def end_items(self, T: int, tok: Tok, sl: Slice) -> list[Item]:
        """A chunk's last items (the model's last layer's output is in PX)."""

    @abc.abstractmethod
    def seconds(self, Ss: float, *, weight_s: float, gemm_s: float, norm_s: float, prep_s: float, cfg_s: float) -> float:
        """Estimated time of a chunk's items (``QwenPrefill.layer_seconds``' constants; Ss sub-chunks)."""

    def start_items(self, T: int, tok: Tok) -> list[Item]:
        """A chunk's first items (PX holds the embeddings)."""
        return []

    def tap_items(self, L: int, T: int, tok: Tok) -> list[Item]:
        """A chunk's items at the start of layer L (PX holds its input)."""
        return []

    def params(self, T: int, p0: int) -> dict[str, int]:
        """A chunk's scratchpad parameters."""
        return {}

    def before(self, dev: Any, p0: int, T: int) -> None:
        """Host work before a chunk of size T at position p0 (``dev``: the prefill's device)."""

    def after(self, dev: Any, p0: int, n: int, T: int) -> None:
        """Host work after a chunk of n valid tokens of T at p0."""
