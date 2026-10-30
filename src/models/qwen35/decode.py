"""The qwen35 multi-slot decode overlay: the shared decode overlay (``decode.design``: the main and helper
cores of every column, their GEMV phases) with the qwen35 cores:

- GDN (c, 2), with ``gdn``: the Gated DeltaNet step of S slots (``qwen35/gdn.py``) on the gdn_in phase's rows (the main
  core's M stream; 8 slots: through DDR, the core's memory also holding the helper's scratch), outputs on D;
- attention (c, 5), with ``attn``: the full-attention layers' attention of S slots (``qwen35/attn.py``), the KV cache
  through A (broadcast with the main core's), outputs relayed by the main core (shared-memory FIFO R) to C; a KV head's
  chain of cores passes its partials west to east through shared memory, a token ObjectFifo per neighbour pair.

Metadata words of the model cores (``M_MODEL`` on): the gdn_in phase's for the GDN core (``M_NREP``, ``M_CSHIFT``,
``M_COMMIT``: gdn.py), an F_ATT object's for the attention core (attn.py): the sub-step's first position; split: the
KV head's chain divides the cache's groups (1); the core's first group, the end of its attention groups, the first
group past them (a chain's first core's: the sub-step's group, for its records only); its first slot's row of angles;
record mode (the first tail object is the open record's V panels); the tail objects.
"""

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, Lock, ObjectFifo, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

from artlite.decode import design as OD
from artlite.decode import gemv
from artlite.decode.design import C_LEN, META, XPAY, DecodeOverlayOperator, _Streams
from artlite.operators.design import COLS, range32

from . import attn as AT
from . import gdn as GD

M_NREP, M_CSHIFT, M_COMMIT = OD.M_MODEL, OD.M_MODEL + 1, OD.M_MODEL + 2  # (the gdn_in phase's, for the GDN core)
M_POS, M_SPLIT, M_G0, M_GATT, M_GJ, M_SA, M_ATREC, M_NTAIL = range(OD.M_MODEL + 3, OD.M_MODEL + 11)  # (an F_ATT object's)


class QwenDecodeOperator(DecodeOverlayOperator):
    """The qwen35 decode overlay design (module docstring): ``rows``: {tile format: rows per tile} of tiles of 1024
    columns (``tile_bytes`` 9280; formats the decode GEMV reads, ``gemv.supported``); ``gdn``, ``attn``: the shapes of the GDN and attention cores
    (None: without)."""

    PREFIX = "q35m"
    GDN_STACK, ATTN_STACK = 4864, AT.STACK
    A_FWD = 3  # (the attention core holds its last two A objects: decode.design's A_FWD)
    LENDS_WS = True  # (the attention core's regions in the main core's memory)
    GDN_ROW, ATTN_ROW = 2, 5

    def __init__(self, *, hidden: int, rows: dict[str, int], tile_bytes: int = AT.OBJ_BYTES, eps: float = 1e-6,
                 gdn: GD.GdnShape | None = None, attn: AT.AttnShape | None = None, moe: tuple[int, int] | None = None,
                 max_k: int = OD.MAX_K, paged: tuple[str, ...] = (), win: str = "", slots: int = OD.SLOTS):
        """moe: (experts, experts per token) of an MoE model (``decode.design``'s "MoE"; its router's BFP16 tiles),
        max_k: the K of a phase's group at most; paged: formats whose kernels are code pages (``decode.model.paged_formats``:
        a drafter's Q4_0 beside the model's three); slots: a step's token slots at most (4, 8)."""
        assert all(gemv.supported(f, 4, slots) for f in rows) and tile_bytes == AT.OBJ_BYTES, rows
        assert gdn is None or (gdn.c_len == C_LEN and gdn.xpay == XPAY and gdn.meta == META and gdn.slots == slots)
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == tile_bytes)
        self.gdn, self.attn = gdn, attn
        if gdn is not None and gdn.split:  # (the helper runs half of the GDN core's state panels: gdn's "split")
            self.help_flag, self.help_word = GD.F_GDN, GD.M_HPAN
        self._gws: dict[int, Buffer] = {}
        self.msg, self.relay = gdn is not None and slots == 4, attn is not None
        self.gate = attn is not None and attn.main_gates  # (the main core applies the attention's gates)
        self.R_LEN = attn.r_bytes // 2 if attn is not None else C_LEN
        super().__init__(hidden=hidden, rows=rows, tile_bytes=tile_bytes, eps=eps, nsb=4, moe=moe, max_k=max_k,
                         paged=paged, win=win, slots=slots, packed=slots == 8)

    def _consts(self) -> dict[str, int]:
        return super()._consts() | {k: v for k, v in globals().items() if k[:2] == "M_"}

    def get_mlir_artifact(self) -> Module:
        self._gws = {}  # (made per program: ``_gws_of``)
        return super().get_mlir_artifact()

    def _extra_src(self, p: str, consts: dict[str, int]) -> str:
        src = ""
        if self.gdn is not None:
            src += self.gdn.render(p, {k: v for k, v in consts.items() if k[:2] in ("M_", "F_")})
        if self.attn is not None:
            W = self.workspace()  # (the main core's: its regions past the residual, idle in an attention sub-step)
            assert W["size"] - W["inv"] >= AT.EXT_W and (not self.attn.main_gates or self.xs_bytes >= AT.EXT_X), W
            assert not self.gate or self.xp_bytes >= 2 * self.attn.rows * AT.HD  # (the gate table in the helper's xp)
            src += self.attn.render(p, {k: v for k, v in consts.items() if k[:2] in ("M_", "F_")}
                                    | {"META": META, "NX_PER_NORM": self.nx_per_norm}, tail=W["inv"],
                                    packed8=self.packed and self.slots == 8)
        return src

    @property
    def d_cols(self) -> list[int]:
        return list(range(COLS)) if self.gdn is not None else []

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        p, P, gdn, attn = s.p, self.PREFIX, self.gdn, self.attn
        bf, i32 = np.dtype[bfloat16], np.dtype[np.int32]
        x_ty, c_ty, a_ty = s.x_ty, s.c_ty, s.a_ty
        gr = self.GDN_ROW
        workers: list[Worker] = []
        d_fifos: list[tuple[int, ObjectFifo]] = []
        if gdn is not None and gdn.slots == 8:
            self._gdn8_workers(s, workers, d_fifos)
        elif gdn is not None:
            G = GD
            GW = gdn.workspace()
            gws_ty = np.ndarray[(GW["size"] // 4,), i32]
            d_ty = np.ndarray[(G.D_BYTES // 2,), bf]
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
                    for _ in range_(ws[i_g + G.G_NSKIP]):
                        x_f.acquire(1)
                        x_f.release(1)
                    for _ in range_(ws[i_g + G.G_NMSG]):
                        m = m_f.acquire(1)
                        gmsg(m, ws)
                        m_f.release(1)
                    for _ in range_(ws[i_g + G.G_ON]):
                        for _ in range_(ws[i_g + G.G_NCP]):  # conv pairs: q, k, then each pair of v heads
                            conv2()
                            for _ in range_(ws[i_g + G.G_KO]):  # (after k)
                                x = x_f.acquire(1)
                                gkold(x, ws)
                                x_f.release(1)
                                d = d_f.acquire(1)
                                gknew(ws, d)
                                d_f.release(1)
                            for _ in range_(ws[i_g + G.G_PO]):  # (after a pair's v)
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
                workers.append(Worker(gdn_fn, [s.x_l1[c].cons(depth=1), s.m_fifos[c].cons(), d_fifos[c][1].prod(), k_gbegin,
                                               k_gmsg, k_gconv, k_gkold, k_gknew, k_gdl, k_gpanel, k_gdlout, k_gout,
                                               gws],
                                      tile=Tile(c, gr), stack_size=self.GDN_STACK, dynamic_objfifo_lowering=True,
                                      while_true=False))
        if attn is not None:
            AWs = attn.workspace()
            aws_ty = np.ndarray[(AWs["size"] // 4,), i32]
            i_a = AWs["st"] // 4
            mws_ty = np.ndarray[(self.workspace()["size"] // 4,), i32]
            mxs_ty = np.ndarray[(self.xs_bytes,), np.dtype[np.uint8]]
            k_abegin = Kernel(f"{p}_abegin", "kernel.o", [x_ty, aws_ty, np.int32, mws_ty, mxs_ty])
            k_anb = Kernel(f"{p}_anb", "kernel.o", [aws_ty])
            k_aq = Kernel(f"{p}_aq", "kernel.o", [x_ty, aws_ty])
            k_aqpan = Kernel(f"{p}_aqpan", "kernel.o", [aws_ty])
            k_akv = Kernel(f"{p}_akv", "kernel.o", [x_ty, aws_ty, np.int32])
            k_aobj = Kernel(f"{p}_aobj", "kernel.o", [a_ty, aws_ty])
            k_asoft = Kernel(f"{p}_asoft", "kernel.o", [aws_ty])
            k_aend = Kernel(f"{p}_aend", "kernel.o", [aws_ty, aws_ty])
            tok_ty = np.ndarray[(16,), i32]
            k_aemit = Kernel(f"{p}_aemit", "kernel.o", [aws_ty, s.r_ty])

            def attn_fn(a_f, x_f, r_f, t_in, t_out, abegin, anb, aq, aqpan, akv, aobj, asoft, aend, aemit, ws, pw, mws, mxs,
                        col):
                # t_in, t_out: the tokens from the west and to the east neighbour of the chain (none at its ends), every
                # attention step: a core produces its east token around its scores (a split step's partial stays in its
                # workspace until the east neighbour has merged it), consumes its west token around its end (the merge
                # of the west partial, pw); mws, mxs: the main core's workspace and sum panels (attn.AttnShape.ext)
                for _ in forever_():
                    x = x_f.acquire(1)
                    abegin(x, ws, col, mws, mxs)
                    x_f.release(1)
                    for _ in range_(ws[i_a + AT.A_NNO]):  # (a main core's normed phase: as it consumes them)
                        a_f.acquire(1)
                        anb(ws)
                        for _ in range_(ws[i_a + AT.A_NXN]):
                            x_f.acquire(1)
                            x_f.release(1)
                        a_f.release(1)
                    for _ in range_(ws[i_a + AT.A_NXSKIP]):
                        x_f.acquire(1)
                        x_f.release(1)
                    for _ in range_(ws[i_a + AT.A_NASKIP]):
                        a_f.acquire(1)
                        a_f.release(1)
                    for _ in range_(ws[i_a + AT.A_NQP]):  # (an attention step)
                        for t in t_out:
                            t.acquire(1)
                        for _ in range_(ws[i_a + AT.A_NQ]):
                            x = x_f.acquire(1)
                            aq(x, ws)
                            x_f.release(1)
                        aqpan(ws)
                        for _ in range_(ws[i_a + AT.A_NKV]):
                            x = x_f.acquire(1)
                            akv(x, ws, col)
                            x_f.release(1)
                        for _ in range_(ws[i_a + AT.A_NOBJ]):
                            a = a_f.acquire(1)
                            aobj(a, ws)
                            a_f.release(1)
                            asoft(ws)  # (a group's softmax after its K objects)
                        h = a_f.acquire(AT.HOLD)  # (the last two: held until the emission, which reads records from them)
                        aobj(h[0], ws)
                        asoft(ws)
                        aobj(h[1], ws)
                        asoft(ws)
                        if t_in:
                            t_in[0].acquire(1)
                            aend(ws, pw)
                            t_in[0].release(1)
                            for t in t_out:
                                t.release(1)
                        else:
                            t_out[0].release(1)
                            aend(ws, pw)
                        for _ in range_(ws[i_a + AT.A_NE]):
                            r = r_f.acquire(1)
                            aemit(ws, r)
                            r_f.release(1)
                        a_f.release(AT.HOLD)

            aws = [Buffer(aws_ty, name=f"{P}_aws{c}", initial_value=np.zeros(AWs["size"] // 4, np.int32),
                          tile=Tile(c, self.ATTN_ROW)) for c in range(COLS)]
            n = attn.nph
            toks = {c: ObjectFifo(tok_ty, name=f"{P}_T{c}", depth=1) for c in range(COLS) if c % n < n - 1}  # (c -> c + 1)
            for c in range(COLS):
                t_in = [toks[c - 1].cons()] if c % n else []
                t_out = [toks[c].prod()] if c in toks else []
                workers.append(Worker(attn_fn, [s.a_l1[c].cons(depth=AT.HOLD), s.x_l1[c].cons(depth=1), s.r_fifos[c].prod(),
                                                t_in, t_out, k_abegin, k_anb, k_aq, k_aqpan, k_akv, k_aobj, k_asoft, k_aend,
                                                k_aemit, aws[c], aws[c - 1 if c % n else c], s.ws[c], s.xs[c], c],
                                      tile=Tile(c, self.ATTN_ROW), stack_size=self.ATTN_STACK,
                                      dynamic_objfifo_lowering=True, while_true=False))
        return workers, d_fifos

    def _gws_of(self, c: int) -> Buffer:
        """Column c's GDN workspace (8 slots; the helper's ``_help`` takes it too)."""
        if c not in self._gws:
            GW = self.gdn.workspace()
            self._gws[c] = Buffer(np.ndarray[(GW["size"] // 4,), np.dtype[np.int32]], name=f"{self.PREFIX}_gws{c}",
                                  initial_value=np.zeros(GW["size"] // 4, np.int32), tile=Tile(c, self.GDN_ROW))
        return self._gws[c]

    def _help(self, c: int) -> tuple[Kernel, Buffer]:
        """(split) The helper's GDN panels: ``hgdn`` on column c's GDN workspace."""
        GW = self.gdn.workspace()
        k = Kernel(f"{self.pfx}_hgdn", "kernel.o", [np.ndarray[(GW["size"] // 4,), np.dtype[np.int32]],
                                                    np.ndarray[(self.xp_bytes,), np.dtype[np.uint8]]])
        return k, self._gws_of(c)

    def _gdn8_workers(self, s: _Streams, workers: list[Worker], d_fifos: list[tuple[int, ObjectFifo]]) -> None:
        """The GDN cores of an 8-slot design (``gdn``'s "8 slots"): per step (after the header) per k head its q and k blocks
        (their histories out) and K objects, then its v heads (a block, the deltas and z, 16 state panels; the history,
        the deltas and the gated outputs out)."""
        p, P, gdn = s.p, self.PREFIX, self.gdn
        bf, i32 = np.dtype[bfloat16], np.dtype[np.int32]
        x_ty = s.x_ty
        GW = gdn.workspace()
        gws_ty = np.ndarray[(GW["size"] // 4,), i32]
        d_ty = np.ndarray[(GD.D_BYTES // 2,), bf]
        i_g = GW["st"] // 4
        kx = lambda n: Kernel(f"{p}_{n}", "kernel.o", [x_ty, gws_ty])
        kxd = lambda n: Kernel(f"{p}_{n}", "kernel.o", [x_ty, gws_ty, d_ty])
        kd = lambda n: Kernel(f"{p}_{n}", "kernel.o", [gws_ty, d_ty])
        ks = [kx("gbegin"), kx("gw"), kx("gconv"), kd("ghist"), kx("gkold"), kd("gknew"), kx("gdl"), kx("gz"),
              kxd("gpanel"), kd("gdlout"), kd("gout")]
        hs_ty = np.ndarray[(self.xp_bytes,), np.dtype[np.uint8]]  # (split: the helper's x panels, its slot)
        if gdn.split:
            assert self.xp_bytes >= 4 * (GD.H_OUT + 1024), self.xp_bytes
            ks += [Kernel(f"{p}_ghand", "kernel.o", [x_ty, gws_ty, d_ty, hs_ty]),
                   Kernel(f"{p}_gown", "kernel.o", [x_ty, gws_ty, hs_ty]), Kernel(f"{p}_gwait", "kernel.o", [gws_ty]),
                   Kernel(f"{p}_gput", "kernel.o", [hs_ty, d_ty])]
        nv = gdn.hc // 2

        def gdn_fn(x_f, d_f, gbegin, gw, gconv, ghist, gkold, gknew, gdl, gz, gpanel, gdlout, gout, *rest):
            ghand, gown, gwait, gput, hs, ws = rest if gdn.split else (None,) * 5 + (rest[0],)
            def take(k: Kernel) -> None:  # an X object into the workspace
                x = x_f.acquire(1)
                k(x, ws)
                x_f.release(1)

            def emit(k: Kernel) -> None:  # a D object from the workspace
                d = d_f.acquire(1)
                k(ws, d)
                d_f.release(1)

            def block() -> None:  # a conv block's two objects (its new history in the workspace)
                take(gw)
                take(gconv)

            for _ in forever_():
                take(gbegin)
                for _ in range_(ws[i_g + GD.G_NSKIP]):
                    x_f.acquire(1)
                    x_f.release(1)
                for _ in range_(ws[i_g + GD.G_ON]):  # (after the header: ``gbegin``)
                    for _ in range32(2, unroll=False):  # the k heads
                        block()  # q
                        emit(ghist)
                        take(gkold)
                        block()  # k
                        emit(gknew)
                        emit(ghist)
                        for _ in range32(nv, unroll=False):  # its v heads
                            block()
                            take(gdl)
                            take(gz)
                            if gdn.split:  # (per pair: panel i to the helper, panel 8 + i here; their D objects in order)
                                for _ in range32(8, unroll=False):
                                    x = x_f.acquire(1)
                                    d = d_f.acquire(1)
                                    ghand(x, ws, d, hs)
                                    x_f.release(1)
                                    x = x_f.acquire(1)
                                    gown(x, ws, hs)
                                    x_f.release(1)
                                    gwait(ws)
                                    d_f.release(1)
                                    d = d_f.acquire(1)
                                    gput(hs, d)
                                    d_f.release(1)
                            else:
                                for _ in range32(16, unroll=False):
                                    x = x_f.acquire(1)
                                    d = d_f.acquire(1)
                                    gpanel(x, ws, d)
                                    d_f.release(1)
                                    x_f.release(1)
                            emit(ghist)  # (its history and deltas: one drain)
                            emit(gdlout)
                            emit(gout)

        for c in range(COLS):
            d_fifos.append((c, ObjectFifo(d_ty, name=f"{P}_D{c}", depth=2)))
            gws = self._gws_of(c)
            if gdn.split:  # (the panels' locks: handed over, done)
                self._locks += [Lock(Tile(c, self.GDN_ROW), lock_id=GD.L_HRDY, init=0, name=f"{P}_hrdy{c}"),
                                Lock(Tile(c, self.GDN_ROW), lock_id=GD.L_HDONE, init=0, name=f"{P}_hdone{c}")]
            workers.append(Worker(gdn_fn, [s.x_l1[c].cons(depth=1), d_fifos[c][1].prod(), *ks,
                                           *([s.xp[c], gws] if gdn.split else [gws])],
                                  tile=Tile(c, self.GDN_ROW), stack_size=self.GDN_STACK, dynamic_objfifo_lowering=True,
                                  while_true=False))
