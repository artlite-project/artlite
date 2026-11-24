"""The Inkling 2 decode overlay design: the shared decode overlay (``decode.design``: per column the main and helper
cores and their GEMV phases on the image's tiles, ``inkling2.image``) whose phases read fp32 states (``states``:
``decode.protocol.F_STATES``): the residual stream's vectors (the normed embedding, the depth mixes' states, the
controllers' outputs) as fp32 in DDR, a k range's states mixed by the helper into the main core's res region, then
built into the x panels (``design.cc.jinja``'s ``hsacc``, ``hspan``).

The FFN (``moe``) runs on the shared MoE path with an expert's three units (512, 512, 128 channels of h:
``inkling2.image``), routed by sigmoid(l) + bias, weighted by the down's scales, the gate / up rows times the expert's
scale before situ (``moe.route.render(units=)``).

Phases and buffers: ``inkling2/overlay.py``."""

from artlite.decode import gemv
from artlite.decode.attn import Attn
from artlite.decode.attn_operator import AttentionDecodeOperator

MAX_K = 3072  # (a phase's K group: the hidden 2304's chunks, its tail's repeated through the last, ``Region.qt``)


class InklingDecodeOperator(AttentionDecodeOperator):
    """The Inkling decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of
    1024 columns (``tile_bytes``); ``attn``: the attention cores (``Attn(ink=True)``; None: without); ``paged``: formats
    whose kernels are code pages (default: all of them; the cores' memory holds the residual stream's work only beside
    pages); ``win``: a stats build's event window; ``moe``: (experts, used) of the FFN's MoE
    (``decode.design``'s "MoE": the routing's units, sigmoid + bias, the situ activation of ``situ``'s betas). The inputs
    of the hidden are ``Region.qt`` regions (``qtails``): an expert's gate / up rows come in whole [8 gate | 8 up] windows
    of a tail group's row tiles (a file's Q8_0 experts' 8-row tiles in pairs), no gate / up windows."""

    PREFIX = "ink"
    CHAIN = 1
    ATTN_A_DEPTH = 1
    PAGES = True  # (the attention cores' ~23 KB of kernels: the conv, clips, the relative bias, sinks, gates)
    STACK = 2304  # (the main core's: ``collect`` over a tail group's gate / up rows through the windows, situ under it)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float, attn: Attn | None = None,
                 paged: tuple[str, ...] | None = None, win: str = "", moe: tuple[int, int] | None = None,
                 situ: tuple[float, float] = (4.0, 25.0)):
        assert all(gemv.supported(f, 4) for f in rows), rows
        assert attn is None or attn.ink, attn
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4,
                         act="situ" if moe else "silu", packed=True, tok=True, max_k=MAX_K,
                         paged=tuple(sorted(rows)) if paged is None else paged, win=win,
                         pairs=True, states=True, moe=moe, gating="sigmoid", shared_gate=False, units=3 if moe else 0,
                         situ=situ, qtails=True)
