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

The operator cores' program is code pages (``operators.paging``, the memtile transport; ``decode.attn.page_entries``):
the attention (chain 2, two kinds, the gate) is ~15.8 KB of kernels and the conv ~2 KB, against 16 KB of program
memory with the cores' control code. Page 1 prepares a sub-step's queries (~3 KB), page 2 runs it over the cache (~11
KB), page 3 the conv; each attention sub-step loads pages 1 and 2 (~1.3 + 4.5 us), a conv layer page 3 (~0.8 us).
Their store is in each memtile (``PG_STORE``, channel ``PG_CH``'s BDs from ``PG_BD``), loaded before a dispatch whenever
stale (``page_store``, ``page_prologue``).

Phases and buffers: ``lfm/overlay.py``.
"""

import os

import numpy as np
from aie.iron import Buffer, Lock, ObjectFifo, Worker
from aie.iron.device import Tile

from artlite import txn
from artlite.decode import gemv
from artlite.decode.attn import Attn, AttnPages, page_calls, page_entries
from artlite.decode.design import (
    C_LEN,
    META,
    NX_PER_NORM,
    PG_BD,
    PG_CH,
    PG_STATE,
    PG_STORE,
    PG_STORE_BYTES,
    PG_WAKE,
    DecodeOverlayOperator,
    _Streams,
)
from artlite.operators import paging
from artlite.operators.design import COLS

OP_PAGE_ID, OP_REQ_ID = 5, 6  # the operator cores' pages' and requests' stream ids


class LfmDecodeOperator(DecodeOverlayOperator):
    """The LFM3 decode overlay design (module docstring): ``hidden``, ``rows``: {format: rows per tile} of tiles of 1024
    columns (``tile_bytes``); ``attn``: the operator cores' attention (and its conv job: ``Attn.conv``; None: without);
    ``gu_win``: the gate / up tiles hold other than 16 rows (``DecodeOverlayOperator``); ``paged``: formats whose kernels
    are code pages (``DecodeOverlayOperator``)."""

    PREFIX = "lfd"
    ATTN_STACK = 3072
    ATTN_ROW = 5
    A_DEPTH = 12  # (the memtile's BDs: 24 per channel parity)
    ATTN_A_DEPTH = 2  # (the operator cores' cache objects in L1: one arrives while the core works on the other)

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int, eps: float = 1e-5, attn: Attn | None = None,
                 gu_win: bool = False, paged: tuple[str, ...] = (), win: str = ""):
        assert all(gemv.supported(f, 4) for f in rows), rows
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == tile_bytes and attn.chain == 2)
        assert not paged, "the operator cores page: the formats' kernels stay resident"
        if attn is not None:
            attn.pages = AttnPages(0x70000 + PG_STATE, PG_WAKE, *paging.request(stream_id=OP_REQ_ID, channel=PG_CH), PG_BD)
        self.attn = attn
        super().__init__(hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, act="silu", post=True,
                         packed=True, gu_win=gu_win, paged=paged, win=win)

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
        extra = {}
        P, st_ty = self.PREFIX, np.ndarray[(paging.STATE_WORDS,), np.dtype[np.int32]]
        for c in range(COLS):  # (the pager: per core its state words and wake lock, per column its store and routes)
            mem, core = Tile(c, 1), Tile(c, self.ATTN_ROW)
            store = Buffer(np.ndarray[(PG_STORE_BYTES // 4,), np.dtype[np.int32]], tile=mem, name=f"{P}_pgstore{c}",
                           address=PG_STORE)
            extra[c] = [Buffer(st_ty, tile=core, name=f"{P}_pgst{c}", address=PG_STATE), store]
            self._locks.append(Lock(core, lock_id=PG_WAKE, init=0, name=f"{P}_pgw{c}"))
            self._flows += [paging.page_flow(mem, core, channel=PG_CH, stream_id=OP_PAGE_ID),
                            paging.request_flow(core, mem, stream_id=OP_REQ_ID)]
        return self.attn.workers(s, prefix=self.PREFIX, row=self.ATTN_ROW, stack=self.ATTN_STACK,
                                 a_depth=self.ATTN_A_DEPTH, extra=extra)

    def page_store(self) -> tuple[paging.Store, ...] | None:
        """The compiled design's page stores, per place in a KV head's chain (the columns of one place have one control
        code, whose size moves the resident code: even columns, odd columns), checked against the design's
        configuration."""
        if self.attn is None:
            return None
        wd = os.path.dirname(self._program.insts_path)
        p, conv = self.pfx, self.attn.conv is not None
        stubs = tuple(page_calls(p, conv))
        stores = []
        for j in range(self.attn.chain):
            cols = range(j, COLS, self.attn.chain)
            pg = paging.link(wd, pages=page_entries(p, conv), pager=f"{p}_pg_request", calls=page_calls(p, conv),
                             roots=(f"{p}_abegin", f"{p}_anb", *stubs), stack=self.ATTN_STACK,
                             cores=[(c, self.ATTN_ROW) for c in cols], state=PG_STATE, wake=PG_WAKE,
                             stream_id=OP_PAGE_ID)
            store = paging.Store(base=PG_STORE, size=PG_STORE_BYTES, channel=PG_CH, bd0=PG_BD, pagers=(pg,))
            store.check(wd, cols)
            stores.append(store)
        return tuple(stores)

    def page_prologue(self, stores: tuple[paging.Store, ...]) -> np.ndarray:
        """The transaction that makes the pages loadable: each chain place's store and its BDs in its columns' memtiles,
        the operator cores' state words cleared (a configuration rewrites the slot's start with the startup code)."""
        t = txn.Txn()
        for j, store in enumerate(stores):
            store.load(t, range(j, COLS, len(stores)))
        for c in range(COLS):
            t.write32(txn.tile_address(c, self.ATTN_ROW, PG_STATE), 0)
        return t.words()
