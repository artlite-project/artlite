"""The decode design of the attention models (``AttentionDecodeOperator``): the shared decode overlay (``decode.design``)
with the attention cores (``decode.attn``) on row ``ATTN_ROW`` of every column: ``Llama``, ``Gemma 4``, ``LFM3`` and
``Flint``'s operators add their GEMV options, the cores come from here. A model's ``attn`` (None: without the cores) fixes
the chain (``CHAIN`` columns per KV head) and the cache objects (``tile_bytes``). An MoE design (Flint) holds acc (the
experts' sums, ``decode.design``'s "MoE") in the attention workspace past its persistent words (``ACC_AT``: from ``y`` on,
scratch that the attention core writes before it reads it in each sub-step, idle in the MoE phases).

A model whose attention cores' kernels exceed their program memory (``PAGES``: LFM3's chain of 2 with two kinds, the gate
and the conv; Flint's chain of 2 with two kinds, 32 query rows and the sinks) runs them as code pages (``operators.paging``,
the memtile transport; ``decode.attn.page_entries``): page 1 prepares a sub-step's queries (~3 KB), page 2 runs it over the
cache (~11 KB; LFM3's page 3 the conv); each attention sub-step loads pages 1 and 2 (~1.3 + 4.5 us). Their store is in
each memtile (``PG_STORE``, channel ``PG_CH``'s BDs from ``PG_BD``), loaded before a dispatch whenever stale
(``page_store``, ``page_prologue``); the formats' GEMV kernels then stay resident (their pages would take the channel)."""

import os
from typing import ClassVar

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Lock, ObjectFifo, Worker
from aie.iron.device import Tile

from .. import txn
from ..operators import paging
from ..operators.design import COLS
from .attn import ATT, TOKEN_ATT, Attn, page_calls, page_entries
from .design import DecodeOverlayOperator, _Streams
from .protocol import C_LEN, PG_BD, PG_CH, PG_STATE, PG_STORE, PG_STORE_BYTES, PG_WAKE, SLOTS

OP_PAGE_ID, OP_REQ_ID = 5, 6  # the attention cores' pages' and requests' stream ids (``PAGES``)


class AttentionDecodeOperator(DecodeOverlayOperator):
    """The decode overlay with attention cores. ``attn``: their shapes (None: without); the rest: ``DecodeOverlayOperator``'s
    options. A model's operator sets ``PREFIX`` and ``CHAIN`` and passes its design options to ``super().__init__``."""

    META_KINDS = (ATT, TOKEN_ATT)  # (the attention cores')
    CHAIN: ClassVar[int]
    ATTN_STACK = 3072
    ATTN_ROW = 5
    A_DEPTH = 12  # (the memtile's BDs: 24 per channel parity)
    ATTN_A_DEPTH = 1  # (the attention cores' cache objects in L1: 2: one arrives while the core works on the other)
    PAGES: ClassVar[bool] = False  # (the attention cores' program as code pages: module docstring)

    def __init__(self, *, attn: Attn | None, tile_bytes: int, **design: object):
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == tile_bytes and attn.chain == self.CHAIN)
        self.attn = attn
        if attn is not None and self.PAGES:
            assert not design.get("paged"), "the attention cores page: the formats' kernels stay resident"
            hdr, ctl = paging.request(stream_id=OP_REQ_ID, channel=PG_CH)
            attn.pages = paging.PageSpec(state=0x70000 + PG_STATE, wake=PG_WAKE, hdr=(hdr,), bd0=(PG_BD,), ctl=(ctl,))
        super().__init__(tile_bytes=tile_bytes, **design)
        assert not self.moe or (attn is not None and attn.moe), "an MoE design: its attention cores pass the MoE layers"

    def get_mlir_artifact(self) -> Module:
        self._aws = {}  # (made per program: ``_aws_of``)
        return super().get_mlir_artifact()

    def _aws_of(self, c: int) -> Buffer:
        """Column c's attention workspace (an MoE design's main core holds its acc there: ``_acc``)."""
        if c not in self._aws:
            n = self.attn.workspace()["size"] // 4
            self._aws[c] = Buffer(np.ndarray[(n,), np.dtype[np.int32]], name=f"{self.PREFIX}_aws{c}",
                                  initial_value=np.zeros(n, np.int32), tile=Tile(c, self.ATTN_ROW))
        return self._aws[c]

    @property
    def ACC_AT(self) -> int:
        """(moe) acc in the attention workspace from y on (module docstring)."""
        if not self.moe:
            return 0
        aw = self.attn.workspace()
        assert aw["y"] + (SLOTS * self.hidden + 32) * 4 <= aw["size"], aw
        return aw["y"]

    def _acc_type(self) -> type:
        return np.ndarray[(self.attn.workspace()["size"] // 4,), np.dtype[np.int32]]

    def _acc(self, c: int) -> Buffer:
        return self._aws_of(c)

    def _extra_src(self, p: str) -> str:
        if self.attn is None:
            return ""
        return self.attn.render(p, self.win)

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS)) if self.attn is not None else []

    def _core_buffers(self) -> dict[int, list] | None:
        """Per column the buffers of the attention core that its kernels reach through their addresses (``Attn.workers(
        extra=)``; ``PAGES``: the pager's state words and wake lock per core, its store and routes per column)."""
        if self.attn is None or not self.PAGES:
            return None
        extra = {}
        P, st_ty = self.PREFIX, np.ndarray[(paging.STATE_WORDS,), np.dtype[np.int32]]
        for c in range(COLS):
            mem, core = Tile(c, 1), Tile(c, self.ATTN_ROW)
            store = Buffer(np.ndarray[(PG_STORE_BYTES // 4,), np.dtype[np.int32]], tile=mem, name=f"{P}_pgstore{c}",
                           address=PG_STORE)
            extra[c] = [Buffer(st_ty, tile=core, name=f"{P}_pgst{c}", address=PG_STATE), store]
            self._locks.append(Lock(core, lock_id=PG_WAKE, init=0, name=f"{P}_pgw{c}"))
            self._flows += [paging.page_flow(mem, core, channel=PG_CH, stream_id=OP_PAGE_ID),
                            paging.request_flow(core, mem, stream_id=OP_REQ_ID)]
        return extra

    def page_store(self) -> tuple[paging.Store, ...] | None:
        """(``PAGES``) The compiled design's page stores, per place in a KV head's chain (the columns of one place have one
        control code, whose size moves the resident code: even columns, odd columns), checked against the design's
        configuration."""
        if self.attn is None or not self.PAGES:
            return super().page_store()
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
        """(``PAGES``) The transaction that makes the pages loadable: each chain place's store and its BDs in its
        columns' memtiles, the attention cores' state words cleared (a configuration rewrites the slot's start with the
        startup code)."""
        if self.attn is None or not self.PAGES:
            return super().page_prologue(stores)
        t = txn.Txn()
        for j, store in enumerate(stores):
            store.load(t, range(j, COLS, len(stores)))
        for c in range(COLS):
            t.write32(txn.tile_address(c, self.ATTN_ROW, PG_STATE), 0)
        return t.words()

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        if self.attn is None:
            return [], []
        return self.attn.workers(s, prefix=self.PREFIX, row=self.ATTN_ROW, stack=self.ATTN_STACK,
                                 a_depth=self.ATTN_A_DEPTH, extra=self._core_buffers(),
                                 aws=[self._aws_of(c) for c in range(COLS)] if self.moe else None)
