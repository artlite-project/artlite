"""The Llama decode overlay: the shared decode overlay (``decode.design``: per column the main and helper cores and
their GEMV phases on the image's tiles, ``llama.image``) with Llama's options and the attention cores:

- gate / up: silu(bf16(gate)) up (the gather-LUT SiLU);
- pre-norm residual updates: each column keeps its slice of the residual stream in fp32 (``F_RESID``), the output
  projections' phases send its bf16 copy on (``F_EMITR``);
- packed inputs (``packed_slots``: the step's slots only) and token objects (``tok``: the main cores skip an
  attention's cache objects by the count of its token);
- the attention cores (c, 5) (``decode.attn``): one column per KV head (a chain of 1) with its 3 or 4 query heads,
  the cache's K as one or two bfp16 blocks per 8 dims (``Attn.kp``: ``LlamaOverlay(kv_k=)``), o and the KV records out
  on D (shim S2MM 1). Row 2 is free.

Phases and buffers: ``llama/overlay.py``.
"""

from artlite.decode import gemv
from artlite.decode.attn import Attn
from artlite.decode.attn_operator import AttentionDecodeOperator


class LlamaDecodeOperator(AttentionDecodeOperator):
    """The Llama decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of 1024
    columns (``tile_bytes``); ``attn``: the attention cores' shapes (None: without); ``gu_win``: the gate / up
    tiles hold fewer than 16 rows (Q5_K, Q6_K: ``DecodeOverlayOperator``); ``paged``: formats whose kernels are code
    pages (``DecodeOverlayOperator``)."""

    PREFIX = "l3d"
    CHAIN = 1
    ATTN_A_DEPTH = 2  # (the attention cores' cache objects in L1: one arrives while the core works on the other)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float = 1e-5, attn: Attn | None = None,
                 gu_win: bool = False, paged: tuple[str, ...] = (), win: str = ""):
        assert all(gemv.supported(f, 4) for f in rows), rows
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu",
                         packed=True, tok=True, gu_win=gu_win, paged=paged, win=win)
