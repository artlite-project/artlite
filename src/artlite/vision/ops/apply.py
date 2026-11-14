"""``Apply``: GroupNorm (+ SiLU) / LayerNorm / RMSNorm of a bf16 C-tile tensor into panels, on the cores of the norm roles.

The i-th core of its rows normalizes the 80-channel blocks [blk0 + i nblk, ..) for the column's tokens.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ...operators.design import ROWS
from .. import core as K
from .. import design as D
from .. import layout as L
from ..roles import Role
from ..seq import ColOps, MemTask, RowOps
from .common import Command, check_memtile, f32, gn_window, panel_recv


class Norm(StrEnum):
    """What an ``Apply`` computes."""

    GN = "gn"
    GNSILU = "gnsilu"
    LN = "ln"
    RMS = "rms"


@dataclass
class Apply:
    """mode ``Norm``: GroupNorm / GroupNorm + SiLU / LayerNorm / RMSNorm of the bf16 C-tile tensor x (``ntile`` tiles of m tokens; blocks of 80 channels:
    ``nblk`` per core of ``rows`` from block ``blk0``) into the panel tensor y (``y_np`` token panels per kb; its kb of the
    i-th core's first channel: ``y_kb0`` + 10 i nblk). GroupNorm: groups of ``gsize`` channels, the i-th core's first
    channel ``ch_off`` + 80 i nblk in their space (its groups in one record window: ``gn_window``); the records come on
    the B stream (``row_ops(records=)``). LayerNorm / RMSNorm (no mean, beta zero): the producer's records at ``rec``
    ([tile][core][OCH]); ``C``: the norm's channels (1 / C). rows: its core rows (roles gn / ln); join: the two images are one image's halves (their records'
    sums added); ch_step: a core's channels in the statistics' space (default nblk 80), nreal: its real ones (the rest
    pads, in their last group)."""

    m: int
    nblk: int
    mode: Norm
    x: int
    y: int
    ntile: int = 1
    blk0: int = 0
    y_kb0: int = 0
    y_np: int | None = None
    gsize: int = 0
    ch_off: int = 0
    rec: int = 0
    tokens: int = 0  # (GroupNorm: tokens of an image)
    C: int = 0
    eps: float = 1e-5
    rows: tuple[int, ...] = tuple(range(ROWS))
    join: bool = False
    ch_step: int = 0
    nreal: int = 0

    def __post_init__(self) -> None:
        self.mode = Norm(self.mode)
        assert self.m in (64, 16) and (self.m == 64 or self.ntile == 1)
        self.ln = self.mode in (Norm.LN, Norm.RMS)
        self.np = self.m * self.ntile // 8
        self.y_np = self.y_np or self.np
        self.gn = self.mode in (Norm.GN, Norm.GNSILU)
        ct = L.ct_bytes(self.m)  # (an object's row blocks, then its padding: m 64 a row block, m 16 80 words)
        assert ct - self.m * 160 == (1280 if self.m == 64 else 320)
        # (A halves: m 64 3 slots each (a tile's 8 row blocks and its padding in 3 halves), m 16 a tile per half)
        self.nhalf = self.nblk * self.ntile * (3 if self.m == 64 else 1)
        self.half_step = 960 if self.m == 64 else ct // 4
        self.aobj = -(-self.nhalf // 2) + self.ln
        assert self.nblk * 80 <= 640 and (not self.gn or (self.gsize >= 4 and self.tokens)) and (not self.ln or self.C)
        assert not self.ln or self.ntile <= 4, "LayerNorm records past their A object"
        last = self.x + (self.blk0 + (len(self.rows) - 1) * self.nblk) * self.ntile * ct + 4 * (self.nhalf - 1) * self.half_step
        check_memtile(last, D.A_OBJ // 2, "an apply's walk of x")

    def ch0(self, row: int) -> int:
        """Row ``row``'s first channel in the GroupNorm's channel space."""
        return self.ch_off + self.rows.index(row) * (self.ch_step or self.nblk * 80)

    def command(self, row: int, nrec: int = 0) -> np.ndarray:
        cmd = Command(K.K_APPLY)
        mode = {Norm.GN: K.A_GN, Norm.GNSILU: K.A_GNSILU, Norm.LN: K.A_LN, Norm.RMS: K.A_RMS}[self.mode]
        cmd.set([K.C_M, K.C_NTILE, K.C_NBLK, K.C_MODE, K.C_NREC, K.C_EPS, K.C_JOIN],
                (self.m, self.ntile, self.nblk, mode, nrec, f32(self.eps), self.join))
        if self.gn:
            ch0, n = self.ch0(row), self.nblk * 80
            g0 = gn_window(ch0, self.nreal or n, self.gsize)
            cmd.set([K.C_GSIZE, K.C_CH0, K.C_G0, K.C_INVN, K.C_GINV],
                    (self.gsize, ch0, g0, f32(1.0 / (self.gsize * self.tokens)), K.group_inv(self.gsize)))
            cmd.at(K.C_GTAB, K.group_words(ch0, n, self.gsize, g0, self.nreal))
        elif self.ln:
            cmd.set([K.C_INVN], [f32(1.0 / self.C)])
        return cmd.bytes()

    def gamma_beta(self, gamma: np.ndarray | None, beta: np.ndarray | None, row: int) -> np.ndarray:
        """Row ``row``'s B slot: gamma, beta (fp32) of its channels (gamma, beta: of x's channels from block 0):
        GroupNorm [c16][gamma 16 | beta 16], else [gamma | beta]."""
        out = np.zeros(D.Q // 4, np.float32)
        n = self.nblk * 80
        if gamma is not None:
            c0 = (self.blk0 + self.rows.index(row) * self.nblk) * 80
            g = np.asarray(gamma, np.float32)[c0:c0 + n]
            b = np.zeros_like(g) if beta is None else np.asarray(beta, np.float32)[c0:c0 + n]
            if self.gn:
                out[:2 * n] = np.stack([g.reshape(-1, 16), b.reshape(-1, 16)], 1).reshape(-1)
            else:
                out[:n], out[n:2 * n] = g, b
        return out.view(np.uint8)

    def col_ops(self) -> ColOps:
        ct = L.ct_bytes(self.m)
        co = ColOps()
        for i, r in enumerate(self.rows):
            chain = []
            if self.ln:  # (the tiles' records of the column's 4 cores: [tile][core][S1 m | S2 m], 576 B apart:
                # a tile's 2304 B never straddle the object's halves)
                chain.append({"addr": self.rec, "nwords": D.A_OBJ // 4, "dims": [(144, 1), (4, D.OCH // 4),
                                                                                 (self.ntile, D.OCH)]})
            blk = self.blk0 + i * self.nblk  # (its blocks' objects [block][tile], contiguous: per A half 3 slots (m 64:
            # a tile's 8 row blocks and its padding in 3 halves; m 16: a tile's 2 and its padding), then whatever follows)
            chain.append({"addr": self.x + blk * self.ntile * ct, "nwords": (self.aobj - self.ln) * D.A_OBJ // 4,
                          "dims": [(D.A_OBJ // 16, 1), (2, D.A_OBJ // 16), (self.nhalf, self.half_step)]})
            co.sends[r] = [MemTask(chain)]
            YN = self.y_np
            co.recvs[r] = [MemTask([panel_recv(self.y + (self.y_kb0 + 10 * i * self.nblk) * YN * L.PANEL, YN, self.np, 10,
                                                self.nblk)])]
        return co

    def row_ops(self, img: int, records: list[tuple] = (), nslots: int = 0) -> RowOps:
        """img: the row's command and gamma / beta slots (one run); records (GroupNorm): the runs of its ``nslots``
        record slots after them (``core.REC_SLOT`` records each, the last one's rest anything). (Only the op's ``rows``
        have a B stream.)"""
        runs = [(img, 2 * D.Q)] + list(records)
        return RowOps(runs=runs, b_objects=2 + nslots, parity=False, role=Role.LN if self.ln else Role.GN)

