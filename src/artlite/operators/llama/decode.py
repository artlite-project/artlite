"""The Llama decode overlay: the shared decode overlay (``overlay.decode``: per column the main and helper cores and
their GEMV phases on the image's tiles, ``llama.image``) with Llama's options and the attention cores:

- gate / up: silu(bf16(gate)) up (the gather-LUT SiLU);
- pre-norm residual updates: each column keeps its slice of the residual stream in fp32 (``F_RESID``), the output
  projections' phases send its bf16 copy on (``F_EMITR``);
- packed inputs (``packed_slots``: the step's slots only) and token objects (``tok``: the main cores skip an
  attention's cache objects by the count of its token);
- the attention cores (c, 5) (``overlay.attn``): one column per KV head (a chain of 1) with its 3 or 4 query heads,
  the cache's K as one or two bfp16 blocks per 8 dims (``Attn.kp``: ``LlamaOverlay(kv_k=)``), o and the KV records out
  on D (shim S2MM 1). Row 2 is free.

Phases and buffers: ``llama/overlay.py``.
"""

from aie.iron import ObjectFifo, Worker

from ..design import COLS
from ..overlay import gemv
from ..overlay.attn import Attn
from ..overlay.decode import C_LEN, META, NX_PER_NORM, DecodeOverlayOperator, _Streams


class LlamaDecodeOperator(DecodeOverlayOperator):
    """The Llama decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of 1024
    columns (``tile_bytes``); ``attn``: the attention cores' shapes (None: without); ``gu_win``: the gate / up
    tiles hold fewer than 16 rows (Q5_K, Q6_K: ``DecodeOverlayOperator``); ``paged``: formats whose kernels are code
    pages (``DecodeOverlayOperator``)."""

    PREFIX = "l3d"
    ATTN_STACK = 3072
    ATTN_ROW = 5
    A_DEPTH = 12  # (the memtile's BDs: 24 per channel parity)
    ATTN_A_DEPTH = 2  # (the attention cores' cache objects in L1: one arrives while the core works on the other)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float = 1e-5, attn: Attn | None = None,
                 gu_win: bool = False, paged: tuple[str, ...] = (), win: str = ""):
        assert all(gemv.supported(f, 4) for f in rows), rows
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == tile_bytes and attn.chain == 1)
        self.attn = attn
        super().__init__(hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu", packed=True,
                         tok=True, gu_win=gu_win, paged=paged, win=win)

    def _extra_src(self, p: str, consts: dict[str, int]) -> str:
        if self.attn is None:
            return ""
        return self.attn.render(p, {k: v for k, v in consts.items() if k[:2] in ("M_", "F_")}
                                | {"META": META, "NX_PER_NORM": NX_PER_NORM, "COLS": COLS}, self.win)

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS)) if self.attn is not None else []

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        if self.attn is None:
            return [], []
        return self.attn.workers(s, prefix=self.PREFIX, row=self.ATTN_ROW, stack=self.ATTN_STACK, a_depth=self.ATTN_A_DEPTH)
