"""The qwen35 multi-slot decode overlay: the shared decode overlay (``decode.design``: the main and helper
cores of every column, their GEMV phases) with the qwen35 cores:

- GDN (c, 2), with ``gdn``: the Gated DeltaNet step of S slots (``qwen35/gdn.py``) on the gdn_in phase's rows (the main
  core's M stream; 8 slots: through DDR, the core's memory also holding the helper's scratch), outputs on D;
- attention (c, 5), with ``attn``: the full-attention layers' attention of S slots (``decode.attn256``, its design:
  ``decode.attn256_design``).

Metadata words of the model cores (``M_MODEL`` on): ``words.py``.
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, Lock, ObjectFifo, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

from artlite.decode import attn256 as AT
from artlite.decode import gemv
from artlite.decode.attn256_design import Attn256DecodeOperator
from artlite.decode.design import _Streams
from artlite.decode.protocol import C_LEN, MAX_K, META, SLOTS, XPAY
from artlite.operators import kmath
from artlite.operators.design import COLS, range32

from . import gdn as GD
from .words import F_GDN, FLAGS, GDN_IN, M_HPAN


class _ChunkwiseKit(NamedTuple):
    """What a chunkwise GDN core's worker is built from: ``gdn``'s workspace (``gw``), its types, the first state word's
    index in it and the makers of its X kernels and D kernels."""

    gw: dict[str, int]
    gws_ty: Any
    d_ty: Any
    i_g: int
    kx: Callable[[str], Kernel]
    kd: Callable[[str], Kernel]


def _take(x_f: ObjectFifo, ws: Buffer, k: Kernel) -> None:
    """An X object into the workspace."""
    x = x_f.acquire(1)
    k(x, ws)
    x_f.release(1)


def _emit(d_f: ObjectFifo, ws: Buffer, k: Kernel) -> None:
    """A D object out of the workspace."""
    d = d_f.acquire(1)
    k(ws, d)
    d_f.release(1)


class QwenDecodeOperator(Attn256DecodeOperator):
    """The qwen35 decode overlay design (module docstring): ``rows``: {tile format: rows per tile} of tiles of 1024
    columns (``tile_bytes`` 9280; formats the decode GEMV reads, ``gemv.supported``); ``gdn``, ``attn``: the shapes of the GDN and attention cores
    (None: without)."""

    PREFIX = "q35m"
    META_KINDS = (FLAGS, GDN_IN, AT.FLAGS, AT.ATT)  # (the GDN and attention cores')
    GDN_STACK = 4864
    GWS8_AT = GD.GWS_AT  # (chunkwise) the GDN workspace
    GDN_ROW = 2

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int = AT.OBJ_BYTES, eps: float = 1e-6,
                 gdn: GD.GdnShape | None = None, attn: AT.AttnShape | None = None, moe: tuple[int, int] | None = None,
                 max_k: int = MAX_K, paged: tuple[str, ...] = (), win: str = "", slots: int = SLOTS,
                 bcast: bool = False, gu_win: bool = False, pairs: bool = False, split: bool = False,
                 tails: bool = False, tail_pairs: bool = False, rot: bool = False):
        """moe: (experts, experts per token) of an MoE model (``decode.design``'s "MoE"; its router's BFP16 tiles),
        max_k: the K of a phase's group at most; paged: formats whose kernels are code pages (``decode.model.paged_formats``:
        a drafter's Q4_0 beside the model's three); slots: a step's token slots at most (4, 8); bcast: phase inputs on
        one broadcast stream (``decode.design``); gu_win: gate / up tiles of other than 16 rows (``decode.design``);
        pairs: a dense model's F32 alpha planes; split: gate / up blocks of two formats; tails, tail_pairs: half
        regions' inputs and their pairs of row tiles (``decode.design``); rot: rotated inputs (a PARO checkpoint's:
        ``paro_steps``)."""
        assert all(gemv.supported(f, 4, slots) for f in rows) and tile_bytes == AT.OBJ_BYTES, rows
        assert gdn is None or (gdn.c_len == C_LEN and gdn.xpay == XPAY and gdn.meta == META and gdn.slots == slots)
        self.gdn = gdn
        if gdn is not None and slots == 8:  # (the helper computes the v heads' constants: gdn's ``vhead``)
            self.help_flag, self.help_word = F_GDN, M_HPAN
        self._gws: dict[int, Buffer] = {}
        self._tabs: list[Buffer] = []
        self.msg = gdn is not None and slots == 4
        super().__init__(attn=attn, hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, moe=moe, max_k=max_k,
                         paged=paged, win=win, slots=slots, packed=slots == 8, bcast=bcast, gu_win=gu_win, pairs=pairs,
                         split=split, tails=tails, tail_pairs=tail_pairs, **({"rot": True} if rot else {}))

    def _extra_src(self, p: str) -> str:
        return (self.gdn.render(p) if self.gdn is not None else "") + super()._extra_src(p)

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS)) if self.gdn is not None else []

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        workers: list[Worker] = []
        d_fifos: list[tuple[int, ObjectFifo]] = []
        if self.gdn is not None and self.gdn.slots == 8:
            self._gdn8_workers(s, workers, d_fifos)
        elif self.gdn is not None and self.gdn.chunk:
            self._gdn4_workers(s, workers, d_fifos)
        elif self.gdn is not None:
            self._gdn_token_workers(s, workers, d_fifos)
        workers += super()._extra_workers(s)[0]
        return workers, d_fifos

    def _gdn_token_workers(self, s: _Streams, workers: list[Worker], d_fifos: list[tuple[int, ObjectFifo]]) -> None:
        """The token-by-token GDN cores (``gdn``): the step's message on M, per conv pair its objects, its k history, per
        pair of v heads its 32 state panels."""
        p, P, gdn = s.p, self.PREFIX, self.gdn
        bf, i32 = np.dtype[bfloat16], np.dtype[np.int32]
        x_ty, c_ty = s.x_ty, s.c_ty
        gr = self.GDN_ROW
        GW = gdn.workspace()
        gws_ty = np.ndarray[(GW["size"] // 4,), i32]
        d_ty = np.ndarray[(GD.D_BYTES // 2,), bf]
        i_g = GW["st"] // 4
        k_gbegin = Kernel(f"{p}_gbegin", "kernel.o", [x_ty, gws_ty])
        k_gmsg = Kernel(f"{p}_gmsg", "kernel.o", [c_ty, gws_ty])
        k_gconv = Kernel(f"{p}_gconv", "kernel.o", [x_ty, gws_ty, d_ty])
        k_gkold = Kernel(f"{p}_gkold", "kernel.o", [x_ty, gws_ty])
        k_gknew = Kernel(f"{p}_gknew", "kernel.o", [gws_ty, d_ty])
        k_gdl = Kernel(f"{p}_gdl", "kernel.o", [x_ty, gws_ty])
        k_gpanel = Kernel(f"{p}_gpanel", "kernel.o", [x_ty, gws_ty, d_ty])
        k_gdlout = Kernel(f"{p}_gdlout", "kernel.o", [gws_ty, d_ty])
        k_gout = Kernel(f"{p}_gout", "kernel.o", [gws_ty, d_ty])

        def gdn_fn(x_f, m_f, d_f, gbegin, gmsg, gconv, gkold, gknew, gdl, gpanel, gdlout, gout, ws):
            def conv2() -> None:  # two conv objects, their histories in one D object
                d = d_f.acquire(1)
                for _ in range_(2):
                    x = x_f.acquire(1)
                    gconv(x, ws, d)
                    x_f.release(1)
                d_f.release(1)

            for _ in forever_():
                x = x_f.acquire(1)
                gbegin(x, ws)
                x_f.release(1)
                for _ in range_(ws[i_g + GD.G_NSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_g + GD.G_NMSG]):
                    m = m_f.acquire(1)
                    gmsg(m, ws)
                    m_f.release(1)
                for _ in range_(ws[i_g + GD.G_ON]):
                    for _ in range_(ws[i_g + GD.G_NCP]):  # conv pairs: q, k, then each pair of v heads
                        conv2()
                        for _ in range_(ws[i_g + GD.G_KO]):  # (after k)
                            x = x_f.acquire(1)
                            gkold(x, ws)
                            x_f.release(1)
                            d = d_f.acquire(1)
                            gknew(ws, d)
                            d_f.release(1)
                        for _ in range_(ws[i_g + GD.G_PO]):  # (after a pair's v)
                            x = x_f.acquire(1)
                            gdl(x, ws)
                            x_f.release(1)
                            for _ in range32(32, unroll=False):
                                x = x_f.acquire(1)
                                d = d_f.acquire(1)
                                gpanel(x, ws, d)
                                d_f.release(1)
                                x_f.release(1)
                            d = d_f.acquire(1)
                            gdlout(ws, d)
                            d_f.release(1)
                            d = d_f.acquire(1)
                            gout(ws, d)
                            d_f.release(1)

        for c in range(COLS):
            d_fifos.append((c, ObjectFifo(d_ty, name=f"{P}_D{c}", depth=2)))
            gws = Buffer(gws_ty, name=f"{P}_gws{c}", initial_value=np.zeros(GW["size"] // 4, np.int32))
            workers.append(Worker(gdn_fn, [s.x_l1[c].cons(depth=1, **s.x_kw), s.m_fifos[c].cons(), d_fifos[c][1].prod(), k_gbegin,
                                           k_gmsg, k_gconv, k_gkold, k_gknew, k_gdl, k_gpanel, k_gdlout, k_gout,
                                           gws],
                                  tile=Tile(c, gr), stack_size=self.GDN_STACK, dynamic_objfifo_lowering=True,
                                  while_true=False))

    def get_mlir_artifact(self) -> Module:
        self._gws, self._tabs = {}, []  # (made per program: ``_gws_of``, ``_gdn8_workers``)
        return super().get_mlir_artifact()

    def _pins(self) -> dict[str, int]:
        """(and an 8-slot GDN core's helper scratch: a branching step's branch, ``gdn.HS8_AT``)"""
        g8 = self.gdn is not None and self.gdn.slots == 8
        return {f"{self.PREFIX}_gws{c}": self.GWS8_AT for c in self._gws} | ({f"{self.PREFIX}_hs{c}": GD.HS8_AT for c in range(COLS)} if g8 else {})

    def _x_rows(self) -> tuple[int, ...]:
        return (*[self.GDN_ROW] * (self.gdn is not None), *super()._x_rows())

    def _gws_of(self, c: int) -> Buffer:
        """Column c's GDN workspace (chunkwise; the helper's ``_help`` of 8 slots takes it too)."""
        if c not in self._gws:
            GW = self.gdn.workspace()
            self._gws[c] = Buffer(np.ndarray[(GW["size"] // 4,), np.dtype[np.int32]], name=f"{self.PREFIX}_gws{c}",
                                  initial_value=np.zeros(GW["size"] // 4, np.int32), tile=Tile(c, self.GDN_ROW))
        return self._gws[c]

    def _help(self, c: int) -> tuple[Kernel, Buffer]:
        """The helper's GDN work: ``hgdn`` on column c's GDN workspace."""
        GW = self.gdn.workspace()
        k = Kernel(f"{self.pfx}_hgdn", "kernel.o", [np.ndarray[(GW["size"] // 4,), np.dtype[np.int32]],
                                                    np.ndarray[(self.xp_bytes,), np.dtype[np.uint8]]])
        return k, self._gws_of(c)

    def _chunkwise_kit(self, s: _Streams) -> _ChunkwiseKit:
        """The types and kernel makers a chunkwise GDN core shares (``gdn``'s workspace, its D object, the X kernels'
        and D kernels' signatures)."""
        gw = self.gdn.workspace()
        gws_ty = np.ndarray[(gw["size"] // 4,), np.dtype[np.int32]]
        d_ty = np.ndarray[(GD.D_BYTES // 2,), np.dtype[bfloat16]]
        x_ty, p = s.x_ty, s.p
        return _ChunkwiseKit(gw, gws_ty, d_ty, gw["st"] // 4,
                             lambda n: Kernel(f"{p}_{n}", "kernel.o", [x_ty, gws_ty]),
                             lambda n: Kernel(f"{p}_{n}", "kernel.o", [gws_ty, d_ty]))

    def _add_chunkwise_workers(self, s: _Streams, workers: list[Worker], d_fifos: list[tuple[int, ObjectFifo]],
                               fn: Callable, ks: list[Kernel], kit: _ChunkwiseKit, *, msg: bool) -> None:
        """A chunkwise design's GDN core per column: its D fifo (and, of 8 slots, the locks of the helper's work: ``msg``
        False; of 4: the message fifo M), the worker ``fn`` over ``ks`` and the column's workspace and tables."""
        P = self.PREFIX
        for c in range(COLS):
            d_fifos.append((c, ObjectFifo(kit.d_ty, name=f"{P}_D{c}", depth=1)))
            if not msg:
                self._locks += [Lock(Tile(c, self.GDN_ROW), lock_id=GD.L_HRDY, init=0, name=f"{P}_hrdy{c}"),
                                Lock(Tile(c, self.GDN_ROW), lock_id=GD.L_HDONE, init=0, name=f"{P}_hdone{c}")]
            args = [s.x_l1[c].cons(depth=1, **s.x_kw), *([s.m_fifos[c].cons()] if msg else []), d_fifos[c][1].prod(), *ks,
                    self._gws_of(c), *self._gtabs(c)]
            workers.append(Worker(fn, args, tile=Tile(c, self.GDN_ROW), stack_size=self.GDN_STACK,
                                  dynamic_objfifo_lowering=True, while_true=False))

    def _gdn8_workers(self, s: _Streams, workers: list[Worker], d_fifos: list[tuple[int, ObjectFifo]]) -> None:
        """The GDN cores of an 8-slot design (``gdn``'s "8 slots"): per step (after the header) per k head its q block (its
        history out), its k block and K object (its history out), then its v heads (a block and its R object, 16 state
        panels; the history and the gated outputs out); the helper computes the Gram blocks and the chunks' constants."""
        p, gdn = s.p, self.gdn
        kit = self._chunkwise_kit(s)
        x_ty, i_g, kx, kd, gws_ty = s.x_ty, kit.i_g, kit.kx, kit.kd, kit.gws_ty
        assert self.GWS8_AT + kit.gw["size"] <= GD.HS8_AT and GD.HS8_AT + self.hs_bytes <= GD.SILU_TAB[1] - 0x70000, kit.gw
        assert GD.HB_R + 5 * 2 * GD.D <= self.hs_bytes
        ks = [kx("gbegin"), kx("gw"), kx("gconv"), kx("gk"), kx("gvr"), Kernel(f"{p}_gwait", "kernel.o", [gws_ty]),
              Kernel(f"{p}_gpanel", "kernel.o", [x_ty, gws_ty, kit.d_ty]), kx("gpanel0"), kd("ghq"), kd("ghv"), kd("ghv2"),
              kd("gout")]
        nv = gdn.hc // 2

        def gdn_fn(x_f, d_f, gbegin, gw, gconv, gk, gvr, gwait, gpanel, gpanel0, ghq, ghv, ghv2, gout, ws, *_tabs):
            take, emit = functools.partial(_take, x_f, ws), functools.partial(_emit, d_f, ws)  # (a D object from the workspace)

            for _ in forever_():
                take(gbegin)
                for _ in range_(ws[i_g + GD.G_NSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_g + GD.G_ON]):  # (after the header: ``gbegin``)
                    for _ in range32(2, unroll=False):  # the k heads
                        take(gw)  # q
                        take(gconv)
                        emit(ghq)
                        take(gw)  # k
                        take(gconv)
                        take(gk)
                        emit(ghv)
                        emit(ghv2)
                        for _ in range32(nv, unroll=False):  # its v heads
                            take(gw)
                            take(gconv)
                            take(gvr)
                            gwait(ws)
                            for _ in range_(ws[i_g + GD.G_NPD]):  # (a step that ends its block: its update out)
                                x = x_f.acquire(1)
                                d = d_f.acquire(1)
                                gpanel(x, ws, d)
                                d_f.release(1)
                                x_f.release(1)
                            for _ in range_(ws[i_g + GD.G_NPX]):
                                take(gpanel0)
                            emit(ghv)
                            emit(ghv2)
                            emit(gout)

        self._add_chunkwise_workers(s, workers, d_fifos, gdn_fn, ks, kit, msg=False)

    def _gtabs(self, c: int) -> list[Buffer]:
        """Column c's SiLU table copies in the GDN core's memory (used through their addresses only)."""
        P = self.PREFIX
        self._tabs += [Buffer(np.ndarray[(1600 // 4,), np.dtype[np.uint32]], name=f"{P}_gtab{c}_{i}", tile=Tile(c, self.GDN_ROW),
                              address=a - 0x70000, initial_value=kmath.SILU.table_words()) for i, a in enumerate(GD.SILU_TAB)]
        return self._tabs[-2:]

    def _gdn4_workers(self, s: _Streams, workers: list[Worker], d_fifos: list[tuple[int, ObjectFifo]]) -> None:
        """The GDN cores of a chunkwise design of 4 slots (``gdn``'s "Chunkwise", "4 slots"): the step's message on M, then per k
        head its [P, W, K] objects (its q and k blocks, their histories out), per v head its [A, H] objects, 16 state
        panels (a step that ends its block: their update out), its outputs and history out."""
        p, gdn = s.p, self.gdn
        kit = self._chunkwise_kit(s)
        x_ty, i_g, kx, kd, gws_ty = s.x_ty, kit.i_g, kit.kx, kit.kd, kit.gws_ty
        assert self.GWS8_AT + kit.gw["size"] <= GD.SILU_TAB[1] - 0x70000, kit.gw
        ks = [kx("gbegin"), Kernel(f"{p}_gmsg", "kernel.o", [s.c_ty, gws_ty]), kx("gq"), kx("gqw"), kx("gk"), kx("gv"), kx("gvh"),
              kx("gpanel"), Kernel(f"{p}_gupd", "kernel.o", [x_ty, gws_ty, kit.d_ty]), kd("ghq"), kd("ghk"), kd("gvhout"),
              kd("gout")]
        nv = gdn.hc // 2

        def gdn_fn(x_f, m_f, d_f, gbegin, gmsg, gq, gqw, gk, gv, gvh, gpanel, gupd, ghq, ghk, gvhout, gout, ws, *_tabs):
            take, emit = functools.partial(_take, x_f, ws), functools.partial(_emit, d_f, ws)  # (a D object from the workspace)

            for _ in forever_():
                take(gbegin)
                for _ in range_(ws[i_g + GD.G_NSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_g + GD.G_NMSG]):
                    m = m_f.acquire(1)
                    gmsg(m, ws)
                    m_f.release(1)
                for _ in range_(ws[i_g + GD.G_ON]):
                    for _ in range32(2, unroll=False):  # the k heads
                        take(gq)
                        take(gqw)
                        emit(ghq)
                        take(gk)
                        emit(ghk)
                        for _ in range32(nv, unroll=False):  # its v heads
                            take(gv)
                            take(gvh)
                            for _ in range32(16, unroll=False):
                                x = x_f.acquire(1)
                                gpanel(x, ws)
                                for _ in range_(ws[i_g + GD.G_NPD]):  # (1: the step ends its block, the panel's update out)
                                    d = d_f.acquire(1)
                                    gupd(x, ws, d)
                                    d_f.release(1)
                                x_f.release(1)
                            emit(gout)  # (its outputs and its history's first rows, then the rest: one drain for the two)
                            emit(gvhout)

        self._add_chunkwise_workers(s, workers, d_fifos, gdn_fn, ks, kit, msg=True)
