"""The decode design of the attention models (``AttentionDecodeOperator``): the shared decode overlay (``decode.design``)
with the attention cores (``decode.attn``) on row ``ATTN_ROW`` of every column: ``Llama``, ``Gemma 4`` and ``LFM3``'s
operators add their GEMV options, the cores come from here. A model's ``attn`` (None: without the cores) fixes the chain
(``CHAIN`` columns per KV head) and the cache objects (``tile_bytes``)."""

from typing import ClassVar

from aie.iron import ObjectFifo, Worker

from ..operators.design import COLS
from .attn import ATT, TOKEN_ATT, Attn
from .design import DecodeOverlayOperator, _Streams
from .protocol import C_LEN


class AttentionDecodeOperator(DecodeOverlayOperator):
    """The decode overlay with attention cores. ``attn``: their shapes (None: without); the rest: ``DecodeOverlayOperator``'s
    options. A model's operator sets ``PREFIX`` and ``CHAIN`` and passes its design options to ``super().__init__``."""

    META_KINDS = (ATT, TOKEN_ATT)  # (the attention cores')
    CHAIN: ClassVar[int]
    ATTN_STACK = 3072
    ATTN_ROW = 5
    A_DEPTH = 12  # (the memtile's BDs: 24 per channel parity)
    ATTN_A_DEPTH = 1  # (the attention cores' cache objects in L1: 2: one arrives while the core works on the other)

    def __init__(self, *, attn: Attn | None, tile_bytes: int, **design: object):
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == tile_bytes and attn.chain == self.CHAIN)
        self.attn = attn
        super().__init__(tile_bytes=tile_bytes, **design)

    def _extra_src(self, p: str) -> str:
        if self.attn is None:
            return ""
        return self.attn.render(p, self.win)

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS)) if self.attn is not None else []

    def _core_buffers(self) -> dict[int, list] | None:
        """Per column the buffers of the attention core that its kernels reach through their addresses (the pager's state
        words and store: ``Attn.workers(extra=)``)."""
        return None

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        if self.attn is None:
            return [], []
        return self.attn.workers(s, prefix=self.PREFIX, row=self.ATTN_ROW, stack=self.ATTN_STACK,
                                 a_depth=self.ATTN_A_DEPTH, extra=self._core_buffers())
