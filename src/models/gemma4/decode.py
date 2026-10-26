"""The Gemma 4 decode overlay: the shared decode overlay (``decode.design``: per column the main and helper cores and
their GEMV phases on the image's Q4_0 / Q8_0 tiles, ``gemma4.image``) with Gemma's options and its attention cores:

- the GEMV tiles split by rows between the main and helper cores (``decode.gemv.row_split``);
- gate / up: gelu(bf16(gate)) up; the per-layer input gate: gelu(bf16(y)) pli (``F_PLI``);
- post-norm residual updates (``F_POST`` headers, ``F_VOUT`` outputs: the residual stays fp32 in the main cores);
- packed inputs (``packed_slots``: the step's slots only);
- the attention cores (c, 5) (``decode.attn``, Gemma's shapes ``gemma4/attn.py``): per KV head a chain of 4 columns
  over the cache, o and the KV records out on D (shim S2MM 1), no relay through the main core.

Phases and buffers: ``gemma4/overlay.py``.
"""

from aie.iron import ObjectFifo, Worker

from artlite.decode.design import C_LEN, DecodeOverlayOperator, _Streams
from artlite.operators.design import COLS

from . import attn as AT
from .image import KC, TILE_BYTES


class GemmaDecodeOperator(DecodeOverlayOperator):
    """The Gemma decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} (Q4_0 32, Q8_0
    16); ``attn``: the attention cores' shapes (None: without)."""

    PREFIX = "g4d"
    ATTN_STACK = 3072
    ATTN_ROW = 5
    A_DEPTH = 12  # (the memtile's BDs: 24 per channel parity)

    def __init__(self, *, hidden: int, rows: dict[str, int], eps: float = 1e-6, attn: AT.GemmaAttn | None = None,
                 win: str = ""):
        assert set(rows) <= {"Q4_0", "Q8_0"}, rows
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == TILE_BYTES)
        self.attn = attn
        super().__init__(hidden=hidden, rows=rows, tile_bytes=TILE_BYTES, eps=eps, nsb=KC // 256, act="gelu",
                         post=True, pli=True, packed=True, win=win)

    def _extra_src(self, p: str, consts: dict[str, int]) -> str:
        if self.attn is None:
            return ""
        return self.attn.render(p, {k: v for k, v in consts.items() if k[:2] in ("M_", "F_")}
                                | {"META": 64, "NX_PER_NORM": 8, "COLS": COLS}, self.win)

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS)) if self.attn is not None else []

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        if self.attn is None:
            return [], []
        return self.attn.workers(s, prefix=self.PREFIX, row=self.ATTN_ROW, stack=self.ATTN_STACK)
