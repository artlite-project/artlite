"""The LFM3 decode overlay: the shared decode overlay (``decode.design``: per column the main and helper cores and their
GEMV phases on the image's tiles of 1024 columns, ``lfm.image``) with LFM3's options and the operator cores:

- gate / up: silu(bf16(gate)) up (the gather-LUT SiLU);
- post-norm residual updates (the sandwich norms: ``F_POST`` headers, ``F_VOUT`` outputs; the residual stays fp32 in
  the main cores);
- packed inputs (``packed_slots``: the step's slots only) and token objects;
- the operator cores (c, 5) (``decode.attn``): per KV head a chain of 2 columns over the cache with all 4 query heads
  (head_dim 128; two kinds of layer: sliding-window, every pair rotated, and full, 32 pairs rotated), q / k RMS-normed
  with weights per dim, a sigmoid output gate per head, the cache's K as one or two bfp16 blocks per 8 dims; and
  between attention layers the conv layers' step (``lfm.conv``); o, the KV records, the conv's y and history out on D
  (shim S2MM 1). Row 2 is free.

The operator cores' program is code pages (``AttentionDecodeOperator.PAGES``): the attention (chain 2, two kinds, the
gate) is ~15.8 KB of kernels and the conv ~2 KB, against 16 KB of program memory with the cores' control code; page 3
runs the conv (~0.8 us a conv layer).

Phases and buffers: ``lfm/overlay.py``.
"""

from artlite.decode import gemv
from artlite.decode.attn import ATT, TOKEN_ATT, Attn
from artlite.decode.attn_operator import AttentionDecodeOperator

from .conv import CONV, CONV_TOKEN


class LfmDecodeOperator(AttentionDecodeOperator):
    """The LFM3 decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of 1024
    columns (``tile_bytes``); ``attn``: the operator cores' attention (and its conv job: ``Attn.conv``; None: without);
    ``gu_win``: the gate / up tiles hold other than 16 rows (``DecodeOverlayOperator``); ``paged``: formats whose kernels
    are code pages; ``split``: gate / up blocks of two formats (``DecodeOverlayOperator``)."""

    META_KINDS = (ATT, TOKEN_ATT, CONV, CONV_TOKEN)  # (the operator cores': attention and conv)
    PREFIX = "lfd"
    CHAIN = 2
    ATTN_A_DEPTH = 2  # (the operator cores' cache objects in L1: one arrives while the core works on the other)
    PAGES = True

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float = 1e-5, attn: Attn | None = None,
                 gu_win: bool = False, paged: tuple[str, ...] = (), win: str = "", split: bool = False):
        assert all(gemv.supported(f, 4) for f in rows), rows
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu",
                         post=True, packed=True, gu_win=gu_win, paged=paged, win=win,
                         split=split)
