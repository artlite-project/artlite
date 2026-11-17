"""The decode overlay (``decode.design``) with the gated attention of head_dim 256 (``attn256``): the attention core at
(c, 5) of every column, the KV cache through A (broadcast with the main core's), outputs relayed by the main core
(shared-memory FIFO R) to C, the main core applying the gates where the shape says so (``AttnShape.main_gates``); a KV
head's chain of cores passes its partials west to east through shared memory, a token ObjectFifo per neighbour pair (one
pass, a design of 8 slots: each core emits its own rows, no tokens). A model's design subclasses it with its own cores
(qwen35's GDN cores: ``models.qwen35.decode``)."""

from typing import ClassVar

import numpy as np
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.controlflow import forever_, range_
from aie.iron.device import Tile

from ..operators import kmath
from ..operators.design import COLS
from ..operators.words import Kind
from . import attn256 as AT
from .design import DecodeOverlayOperator, _Streams
from .protocol import C_LEN, SLOTS


class Attn256DecodeOperator(DecodeOverlayOperator):
    """The decode overlay design with the attention cores of ``attn`` (an ``attn256.AttnShape``; None: without), the
    other arguments ``DecodeOverlayOperator``'s (its ``a_bcast``: the shape's)."""

    META_KINDS: ClassVar[tuple[Kind, ...]] = (AT.FLAGS, AT.ATT)  # (the attention cores')
    ATTN_STACK = AT.STACK
    GATE_STATE = (AT.GATES,)  # (the main core applies the attention's gates) its state words
    A_FWD = 3  # (the attention core holds its last two A objects: decode.design's A_FWD)
    LENDS_WS = True  # (the attention core's regions in the main core's memory)
    ATTN_ROW = 5

    def __init__(self, *, attn: AT.AttnShape | None, **kw: object):
        assert attn is None or (attn.c_len == C_LEN and attn.a_bytes == kw.get("tile_bytes", AT.OBJ_BYTES))
        self.attn = attn
        self._aws: dict[int, Buffer] = {}
        self.relay = attn is not None
        self.gate = attn is not None and attn.main_gates  # (the main core applies the attention's gates)
        self.R_LEN = attn.r_bytes // 2 if attn is not None else C_LEN
        super().__init__(a_bcast=attn.a_bcast if attn is not None else (), **kw)

    def _extra_src(self, p: str) -> str:
        if self.attn is None:
            return ""
        W = self.workspace()  # (the main core's: its regions past the residual, idle in an attention sub-step)
        assert W["size"] - W["inv"] >= self.attn.ext_w and self.xs_bytes >= self.attn.mx_bytes, W
        assert not self.gate or self.xp_bytes >= 2 * self.attn.rows * AT.HD  # (the gate table in the helper's xp)
        return self.attn.render(p, tail=W["inv"], nx_per_norm=self.nx_per_norm, packed8=self.packed and self.slots == 8)

    def _extra_workers(self, s: _Streams) -> tuple[list[Worker], list[tuple[int, ObjectFifo]]]:
        workers: list[Worker] = []
        if self.attn is not None:
            self._attn_workers(s, workers)
        return workers, []

    def _attn_workers(self, s: _Streams, workers: list[Worker]) -> None:
        """The attention cores (``attn``): per step the q and K / V rows, the cache's objects and their softmax, the end
        (the merge of the west neighbour's partial), the records' emission."""
        p, P, attn = s.p, self.PREFIX, self.attn
        i32 = np.dtype[np.int32]
        x_ty, a_ty = s.x_ty, s.a_ty
        AWs = attn.workspace()
        aws_ty = np.ndarray[(AWs["size"] // 4,), i32]
        i_a = AWs["st"] // 4
        mws_ty = np.ndarray[(self.workspace()["size"] // 4,), i32]
        mxs_ty = np.ndarray[(self.xs_bytes,), np.dtype[np.uint8]]
        k_abegin = Kernel(f"{p}_abegin", "kernel.o", [x_ty, aws_ty, np.int32, mws_ty, mxs_ty])
        k_anb = Kernel(f"{p}_anb", "kernel.o", [aws_ty])
        gt = kmath.SIGMOID.table_words()
        gt_ty = np.ndarray[(gt.size,), np.dtype[np.uint32]]
        k_aq = Kernel(f"{p}_aq", "kernel.o", [x_ty, aws_ty, *[gt_ty] * (2 * attn.gate_tab)])
        k_aqpan = Kernel(f"{p}_aqpan", "kernel.o", [aws_ty])
        k_akv = Kernel(f"{p}_akv", "kernel.o", [x_ty, aws_ty, np.int32])
        k_aobj = Kernel(f"{p}_aobj", "kernel.o", [a_ty, aws_ty])
        k_asoft = Kernel(f"{p}_asoft", "kernel.o", [aws_ty])
        k_aend = Kernel(f"{p}_aend", "kernel.o", [aws_ty, aws_ty])
        tok_ty = np.ndarray[(16,), i32]
        k_aemit = Kernel(f"{p}_aemit", "kernel.o", [aws_ty, s.r_ty])

        def attn_fn(a_f, x_f, r_f, t_in, t_out, abegin, anb, aq, aqpan, akv, aobj, asoft, aend, aemit, ws, pw, mws, mxs,
                    col, tabs):
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
                        aq(x, ws, *tabs)
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
                    if not attn.one:
                        asoft(ws)
                    aobj(h[1], ws)
                    if not attn.one:
                        asoft(ws)
                    if t_in:
                        t_in[0].acquire(1)
                        aend(ws, pw)
                        t_in[0].release(1)
                        for t in t_out:
                            t.release(1)
                    elif t_out:
                        t_out[0].release(1)
                        aend(ws, pw)
                    else:
                        aend(ws, pw)
                    for _ in range_(ws[i_a + AT.A_NE]):
                        r = r_f.acquire(1)
                        aemit(ws, r)
                        r_f.release(1)
                    a_f.release(AT.HOLD)

        aws = [self._aws_of(c) for c in range(COLS)]
        tabs = {c: [Buffer(gt_ty, name=f"{P}_agt{c}", tile=Tile(c, self.ATTN_ROW), initial_value=gt)] * 2 if attn.gate_tab else []
                for c in range(COLS)}
        n = attn.nph
        toks = {c: ObjectFifo(tok_ty, name=f"{P}_T{c}", depth=1) for c in range(COLS)
                if c % n < n - 1 and not attn.one}  # (c -> c + 1)
        for c in range(COLS):
            t_in = [toks[c - 1].cons()] if c % n and not attn.one else []
            t_out = [toks[c].prod()] if c in toks else []
            workers.append(Worker(attn_fn, [s.a_l1[c - c % n if attn.one else c].cons(depth=AT.HOLD), s.x_l1[c].cons(depth=1, **s.x_kw),
                                            s.r_fifos[c].prod(),
                                            t_in, t_out, k_abegin, k_anb, k_aq, k_aqpan, k_akv, k_aobj, k_asoft, k_aend,
                                            k_aemit, aws[c], aws[c - 1 if c % n else c], s.ws[c], s.xs[c], c, tabs[c]],
                                  tile=Tile(c, self.ATTN_ROW), stack_size=self.ATTN_STACK,
                                  dynamic_objfifo_lowering=True, while_true=False))

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
        """(moe) acc in the attention workspace past its state words: its regions are scratch that the attention core
        writes before it reads them in each sub-step, idle in the MoE phases (the attention core only skips their objects,
        on its state words); without an attention core, a buffer of its own on its tile."""
        if not self.moe or self.attn is None:
            return 0
        aw = self.attn.workspace()
        assert aw["y"] == aw["st"] + 192 and aw["y"] + (SLOTS * self.hidden + 32) * 4 <= aw["size"], aw
        return aw["y"]

    def _acc_type(self) -> type:
        n = self.attn.workspace()["size"] // 4 if self.attn is not None else SLOTS * self.hidden + 32
        return np.ndarray[(n,), np.dtype[np.int32]]

    def _acc(self, c: int) -> Buffer:
        if self.attn is not None:
            return self._aws_of(c)
        return Buffer(self._acc_type(), name=f"{self.PREFIX}_acc{c}", tile=Tile(c, self.ATTN_ROW))

    def _x_rows(self) -> tuple[int, ...]:
        return (self.ATTN_ROW,) if self.attn is not None else ()
