"""The Flint decode overlay: the shared decode overlay (``decode.design``: per column the main and helper cores and their
GEMV phases on the image's tiles, ``flint.image``) with the attention cores and the MoE layers:

- gate / up: silu(bf16(gate)) up (the gather-LUT SiLU); pre-norm residual updates (``F_RESID``, ``F_EMITR``);
- packed inputs (``packed_slots``: the step's slots only) and token objects (``tok``: the main cores skip an attention's
  cache objects by the count of its token);
- the attention cores (c, 5) (``decode.attn``): a chain of 2 columns per KV head with its 8 query heads (32 query rows: a
  step of 4 slots in one sub-step), two kinds by their rotated pairs (sliding-window: all 64 pairs; full: none), q / k
  norms with weights, the heads' sinks; o and the KV records out on D (shim S2MM 1); their program as code pages
  (``AttentionDecodeOperator.PAGES``);
- the MoE layers (``decode.design``'s "MoE", ``moe.steps``): the routing's top 6 of 256 with sigmoid weights
  renormalized over them (``moe.route``), an ungated shared expert (sdown without F_WS), acc in the attention workspace
  (``AttentionDecodeOperator.ACC_AT``), K groups of 2048 (the side chunk next to them).

Phases and buffers: ``flint/overlay.py``.
"""

from artlite.decode import gemv
from artlite.decode.attn import Attn
from artlite.decode.attn_operator import AttentionDecodeOperator

MAX_K = 2048  # (the MoE design's K groups: the hidden size's, ``flint.image``)


class FlintDecodeOperator(AttentionDecodeOperator):
    """The Flint decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of 1024
    columns (``tile_bytes``); ``attn``: the attention cores' shapes; ``experts``, ``used``: the MoE layers' experts and
    experts per token; ``paged``: formats whose kernels are code pages; ``gu_win``: gate / up tiles of other than 16 rows
    (``decode.design``'s windows)."""

    PREFIX = "fld"
    CHAIN = 2
    ATTN_A_DEPTH = 1  # (32 query rows: the attention core's L1 holds one cache object)
    PAGES = True  # (the attention cores' ~18.6 KB of kernels: two kinds, 32 rows, the sinks, the chain's merge)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float, attn: Attn, experts: int,
                 used: int, paged: tuple[str, ...] = (), win: str = "", gu_win: bool = False):
        assert all(gemv.supported(f, 4) for f in rows), rows
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu",
                         packed=True, tok=True, moe=(experts, used), max_k=MAX_K, gating="sigmoid", shared_gate=False,
                         paged=paged, win=win, gu_win=gu_win)
