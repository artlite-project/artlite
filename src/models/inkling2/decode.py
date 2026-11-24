"""The Inkling 2 decode overlay design: the shared decode overlay (``decode.design``: per column the main and helper
cores and their GEMV phases on the image's tiles, ``inkling2.image``) whose phases read fp32 states (``states``:
``decode.protocol.F_STATES``): the residual stream's vectors (the normed embedding, the depth mixes' states, the
controllers' outputs) as fp32 in DDR, a k range's states mixed by the helper into the main core's res region, then
built into the x panels (``design.cc.jinja``'s ``hsacc``, ``hspan``).

Phases and buffers: ``inkling2/overlay.py``."""

from artlite.decode import gemv
from artlite.decode.attn import Attn
from artlite.decode.attn_operator import AttentionDecodeOperator

MAX_K = 3072  # (a phase's K group: the hidden 2304 padded to whole chunks)


class InklingDecodeOperator(AttentionDecodeOperator):
    """The Inkling decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of
    1024 columns (``tile_bytes``); ``attn``: the attention cores (``Attn(ink=True)``; None: without); ``paged``: formats
    whose kernels are code pages; ``win``: a stats build's event window."""

    PREFIX = "ink"
    CHAIN = 1
    ATTN_A_DEPTH = 1
    PAGES = True  # (the attention cores' ~23 KB of kernels: the conv, clips, the relative bias, sinks, gates)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float, attn: Attn | None = None,
                 paged: tuple[str, ...] = (), win: str = ""):
        assert all(gemv.supported(f, 4) for f in rows), rows
        assert attn is None or attn.ink, attn
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu",
                         packed=True, tok=True, max_k=MAX_K, paged=paged, win=win, pairs=True, states=True)
