"""Operators compiled together into one full ELF (several designs, one hardware context):

- ``FusedOperator`` (``operator``): a runlist dispatched at once, on named buffers in host BOs;
- ``Library`` (``library``): each operator once, never dispatched whole: programs of its entries, assembled on the
  host (``program``).

Their common part is ``elf.FusedElf``; ``layout`` places the named buffers, ``design`` renders the fused MLIR.
"""

from .layout import SEGMENT_BYTES
from .library import Library
from .operator import FusedOperator

__all__ = ["SEGMENT_BYTES", "FusedOperator", "Library"]
