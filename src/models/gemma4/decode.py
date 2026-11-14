"""The Gemma 4 decode overlay: the shared decode overlay (``decode.design``: per column the main and helper cores and
their GEMV phases on the image's Q4_0 / Q8_0 tiles, ``gemma4.image``) with Gemma's options and its attention cores:

- the GEMV tiles split by rows between the main and helper cores (``decode.gemv.row_split``);
- gate / up: gelu(bf16(gate)) up; the per-layer input gate: gelu(bf16(y)) pli (``F_PLI``: pli from the per-layer
  projection's rows, a phase of its own, and the host's per-layer token rows);
- post-norm residual updates (``F_POST`` headers, ``F_VOUT`` outputs: the residual stays fp32 in the main cores);
- packed inputs (``packed_slots``: the step's slots only);
- the attention cores (c, 5) (``decode.attn``, Gemma's shapes ``gemma4/attn.py``): per KV head a chain of 4 columns
  over the cache, o and the KV records out on D (shim S2MM 1), no relay through the main core.

Phases and buffers: ``gemma4/overlay.py``.
"""

from artlite.decode.attn_operator import AttentionDecodeOperator

from . import attn as AT
from .image import KC, TILE_BYTES


class GemmaDecodeOperator(AttentionDecodeOperator):
    """The Gemma decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} (Q4_0 32, Q8_0
    16); ``attn``: the attention cores' shapes (None: without)."""

    PREFIX = "g4d"
    CHAIN = AT.QH

    def __init__(self, *, hidden: int, rows: dict[str, int], eps: float = 1e-6, attn: AT.GemmaAttn | None = None,
                 win: str = ""):
        assert set(rows) <= {"Q4_0", "Q8_0"}, rows
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=TILE_BYTES, eps=eps, nsb=KC // 256,
                         act="gelu", post=True, pli=True, packed=True, win=win)
