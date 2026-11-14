"""The depthwise convs of the vision overlay (role d): ``DwConv`` (a 3 x 3 conv over a patch grid of 2 x 2 windows) and
``StripConv`` (convs of strips)."""

from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from ...operators.design import ROWS
from .. import core as K
from .. import design as D
from .. import layout as L
from ..roles import Role
from ..seq import ColOps, MemTask, RowOps
from .common import Command, check_memtile, dims, panel_recv


@dataclass
class DwConv:
    """A depthwise 3 x 3 conv over a patch grid of 2 x 2 windows (LFM3-VL's conv layers; ``core``'s K_DW, role d): per
    core row r one 80-channel block, 4 r + k (step ``k``), of the column's tile, whose token panel 2 q + p holds class q
    = (py, px) of 8 windows (half p: 8 consecutive windows of the image's raster). Its inputs are in the memtile staging
    ``stage``: per (row r, half p, class q') a slot of A_OBJ / 2 bytes at ``slot(r, p, q')``: [strip cy 0 | strip cy != 0
    (the window row above for classes 2, 3, below for 0, 1): cb 10 x window 10 (from the half's first - 1) x 8 bf16][C:
    cb 10 x window 8 x 8 bf16, row block 2 q' + p of the gate][mask words of class q' (as an output): per tap two words,
    bit 8 w + c set where window w's tap is inside the image]: the fills ``strip_fill``, ``c_fill``, ``mask_fill`` put
    them there (``strip_fill``: 4 executions each, 6 a step: an op's fills and sends stay below a lock's 63). The taps
    come on the B stream (``row_ops``: per row the command, then [tap 9][80] bf16, tap 3 (dy + 1) +
    dx + 1). y = C conv(g) goes out as bfp16 panels into the panel tensor ``y`` (``y_np`` token panels a kb: the out
    projection's A), the row's block at kb 10 (4 r + k)."""

    stage: int
    y: int
    k: int
    y_np: int = 8

    SLOT = D.A_OBJ // 2
    KB = 10  # k blocks of an 80-channel block's panels

    def slot(self, r: int, p: int, q: int) -> int:
        return self.stage + ((r * 2 + p) * 4 + q) * self.SLOT

    STAGE_BYTES = ROWS * 2 * 4 * (D.A_OBJ // 2)

    def __post_init__(self) -> None:
        check_memtile(self.stage, self.STAGE_BYTES, "a depthwise conv's staging")
        panels = self.KB * (4 * (ROWS - 1) + self.k) + self.KB
        check_memtile(self.y, panels * self.y_np * L.PANEL, "a depthwise conv's output panels")

    def command(self) -> np.ndarray:
        return Command(K.K_DW).bytes()

    @staticmethod
    def taps(w: np.ndarray, block: int) -> np.ndarray:
        """The B slot of channels [80 block, +80) (zero past w's) of taps w (C, 3, 3): [tap 9][80] bf16 in Q bytes."""
        out = np.zeros(D.Q // 2, bfloat16)
        sel = np.asarray(w, np.float32).reshape(w.shape[0], 9)[80 * block:80 * block + 80]
        t = np.zeros((9, 80), np.float32)
        t[:, :sel.shape[0]] = sel.T
        out[:720] = t.reshape(-1).astype(bfloat16)
        return out.view(np.uint8)

    def col_ops(self) -> ColOps:
        """Per core row its 8 slots as 4 A objects (one task), its block's panels received."""
        co = ColOps()
        for r in range(ROWS):
            co.sends[r] = [MemTask([{"addr": self.slot(r, 0, 0), "nwords": 4 * D.A_OBJ // 4}])]
            co.recvs[r] = [MemTask([panel_recv(self.y + 10 * (4 * r + self.k) * self.y_np * L.PANEL, self.y_np, 8, 10,
                                                1)])]
        return co

    def row_ops(self, img: int) -> dict[int, RowOps]:
        """img: the device address of the rows' [command | taps] pairs (2 Q bytes each)."""
        return {r: RowOps(runs=[(img + 2 * r * D.Q, 2 * D.Q)], b_objects=2, parity=False, role=Role.D) for r in range(ROWS)}

    def strip_fill(self, p: int, part: str, addr: int, tag: object = None) -> MemTask:
        """A strip fill of half p, 4 executions (a core row each): part "mid" (every class, strip cy 0), "up" (classes 2,
        3: the window row above), "down" (classes 0, 1: the row below); from the DDR address ``addr`` of the strip's first
        window's first class's row-0 block in a DDR image [window][class 4][1280] bf16, per execution row r's block (80
        channels, 320 apart) of nq classes of 10 windows ([window][class][cb 10][8] in the stream); tag: the shim
        address's (``seq.MemTask``)."""
        q0, nq, strip = {"mid": (0, 4, 0), "up": (2, 2, 1), "down": (0, 2, 1)}[part]
        S = self.SLOT // 4
        mem = {"addr": self.slot(0, p, q0) + strip * K.DW_STRIP, "nwords": nq * 400,
               "dims": [(4, 1), (10, 40), (nq, S), (0, 4)], "iteration": (8 * S, 4)}
        ddr = (addr, nq * 10 * 160, (40, nq, 640, 2560), (160, 4)) + ((tag,) if tag is not None else ())
        return MemTask([mem], repeat=4, ddr=ddr)

    def c_fill(self) -> dict:
        """The memtile side of the C fill: the rows' blocks' C tiles ([row 4][rb 8][cb 10][64] bf16 in the stream) into
        their slots (row block 2 q + p -> slot (r, p, q))."""
        S = self.SLOT // 4
        return {"addr": self.slot(0, 0, 0) + K.DW_C, "nwords": 4 * 2560, "dims": [(320, 1), (2, 4 * S), (4, S), (0, 8 * S)]}

    @staticmethod
    def c_ddr(addr: int, k: int) -> tuple:
        """The shim side of the C fill from a band's C tiles [block 16][ct_bytes(64)] of the column at addr: blocks 4 r + k."""
        ct = L.ct_bytes(64)
        return (addr + k * ct, 4 * 10240, (640, 4, 640, 4 * ct // 4))

    def mask_fill(self) -> dict:
        """The memtile side of the masks' fill: [row 4][half 2][class 4][18 words] into the slots."""
        S = self.SLOT // 4
        return {"addr": self.slot(0, 0, 0) + K.DW_MASK, "nwords": 4 * 2 * 4 * 18,
                "dims": [(18, 1), (4, S), (2, 4 * S), (0, 8 * S)]}

    MASK_BYTES = 4 * 2 * 4 * 72  # (a column's masks in DDR: the 4 rows' copies)



@dataclass
class StripConv:
    """A depthwise conv of strips (``core``'s K_DW1, role d: LFM3-Audio's causal 1-D conv and its downsampler's stride-2
    3 x 3 convs): per core row r one 80-channel block (``blocks[r]``) of ``nobj`` A objects, the row's tap words
    (B slot: [tap][80] bf16, then the bias [80]), its output panels (8 token panels from ``tp0`` of the panel tensor ``y``,
    ``y_np`` token panels a kb; the block's kb 10 block). mode 1 (a causal 1-D conv of ntap taps): half h of object o =
    row block 2 o + h, its strip [7 + ntap rows][80] bf16 (row j + s for token j and tap s: ``off`` 0, 80, ..) and its C
    row block after it (``core.C_DWC``); y = C conv. mode 2: object o = row block o, half h its channels 40 h .., a
    strip of ``core.DW1_STRIP`` bf16 (the taps' element offsets ``off``, tokens 80 elements apart), SiLU'd first where
    ``silu``; y = conv + b."""

    mode: int
    y: int
    off: tuple[int, ...]
    blocks: tuple[int, ...] = (0, 1, 2, 3)
    nobj: int = 4
    tp0: int = 0
    y_np: int = 8
    silu: bool = False

    def __post_init__(self) -> None:
        assert self.mode in (1, 2) and len(self.off) <= 9 and len(self.blocks) == ROWS
        assert self.nobj in ((1, 2, 4) if self.mode == 1 else (8,))  # (the output tile's row blocks: 2, 4, 8)
        self.nrb = 2 * self.nobj if self.mode == 1 else self.nobj

    def command(self) -> np.ndarray:
        cmd = Command(K.K_DW1)
        cmd.set([K.C_NOBJ, K.C_DWM, K.C_NTAP, K.C_DWS], (self.nobj, self.mode, len(self.off), self.silu))
        cmd.at(K.C_TOFF, self.off)
        if self.mode == 1:
            cmd.set([K.C_DWC], [80 * self.strip_rows(len(self.off))])
        return cmd.bytes()

    def taps(self, w: np.ndarray, bias: np.ndarray | None, block: int) -> np.ndarray:
        """The B slot of channels [80 block, +80) (zero past w's) of taps w (C, ntap) and the bias (C,) (or none)."""
        out = np.zeros(D.Q // 2, bfloat16)
        nt = len(self.off)
        t = np.zeros((nt + 1, 80), np.float32)
        sel = np.asarray(w, np.float32)[80 * block:80 * block + 80]
        t[:nt, :sel.shape[0]] = sel.T
        if bias is not None:
            b = np.asarray(bias, np.float32)[80 * block:80 * block + 80]
            t[nt, :b.shape[0]] = b
        out[:t.size] = t.reshape(-1).astype(bfloat16)
        return out.view(np.uint8)

    def image(self, w: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
        """The rows' [command | taps] pairs (``row_ops``' image)."""
        return np.concatenate([np.concatenate([self.command(), self.taps(w, bias, self.blocks[r])]) for r in range(ROWS)])

    def recvs(self) -> dict[int, list[MemTask]]:
        return {r: [MemTask([panel_recv(self.y + (10 * self.blocks[r] * self.y_np + self.tp0) * L.PANEL, self.y_np,
                                         self.nrb, 10, 1)])] for r in range(ROWS)}

    def row_ops(self, img: int) -> dict[int, RowOps]:
        """img: the device address of the rows' [command | taps] pairs (2 Q bytes each)."""
        return {r: RowOps(runs=[(img + 2 * r * D.Q, 2 * D.Q)], b_objects=2, parity=False, role=Role.D) for r in range(ROWS)}

    @staticmethod
    def strip_rows(ntap: int) -> int:
        """mode 1: rows of a token panel's strip (its 8 tokens and the ntap - 1 before them)."""
        return 7 + ntap

    @staticmethod
    def stage1(ntap: int, panels: int = 8) -> int:
        """mode 1: bytes of the staging of an op's strips, [row][panel][strip [rows][80] bf16]."""
        return ROWS * panels * StripConv.strip_rows(ntap) * 160

    def sends1(self, stage: int, c: int, ct: int = L.ct_bytes(64)) -> dict[int, list[MemTask]]:
        """mode 1: per core row its halves (a token panel each) [strip (the staging at ``stage``) | C row block (the C
        tiles at ``c``: [block][ct], the row's block's row block j) | what follows] (A_OBJ / 2 bytes each)."""
        n1, k = self.strip_rows(len(self.off)) * 40, self.nrb
        return {r: [MemTask([{"addr": stage + r * k * 4 * n1, "nwords": n1, "iteration": (n1, k)},
                             {"addr": c + self.blocks[r] * ct, "nwords": D.A_OBJ // 8 - n1, "iteration": (320, k)}],
                            repeat=k)] for r in range(ROWS)}

    def fill1(self, stage: int, g: int, channels: int, pstride: int) -> MemTask:
        """mode 1: the staging's strips from a DDR image of rows [position][channels] bf16: ``g`` the device address of
        token panel 0's first strip row (ntap - 1 before its first token) at block 0, the panels' first rows ``pstride``
        rows apart (the interleaved layout: 64); per core row (an execution) its block's strip rows of each panel."""
        rw, bs = channels // 2, self.blocks[1] - self.blocks[0]
        assert all(self.blocks[r + 1] - self.blocks[r] == bs for r in range(ROWS - 1))
        rows, k = self.strip_rows(len(self.off)), self.nrb
        n = k * rows * 40
        return MemTask([{"addr": stage, "nwords": n, "iteration": (n, ROWS)}], repeat=ROWS,
                       ddr=(g + 160 * self.blocks[0], k * rows * 160, (40, rows, rw, pstride * rw), (40 * bs, ROWS)))

    STAGE2 = ROWS * 8 * D.A_OBJ  # (mode 2: the staging of an op's objects, [row][object 8][A_OBJ])

    @staticmethod
    def taps2(lines: int) -> tuple[int, ...]:
        """mode 2's tap offsets of a 3 x 3 stride-2 conv (tap 3 a + b: weights [C][3][3] flattened), the strip [3][17][40]
        of 3 lines of the input across the 8 outputs' axis (a line: 17 positions from 2 o - 1 of the outputs o .. o + 7):
        the lines are the taps' first index a (``lines`` 0) or their second b (1), the positions the other."""
        return tuple((a * 680 + b * 40) if lines == 0 else (b * 680 + a * 40) for a in range(3) for b in range(3))

    def fill2(self, stage: int, src: int, r: int, *, pos: int, line: int, step: int) -> MemTask:
        """mode 2: core row r's 8 objects of the staging from a DDR image of positions of 256 (or more) bf16 channels:
        ``src`` the device address of object 0's first strip element (line 0, position 0, channel 0), its positions
        ``pos`` words apart, its 3 lines ``line`` words apart, the objects ``step`` words apart; per position the row's
        block's 80 channels, 40 into each half (block 3 of 256: 64 channels of the next position, garbage the next GEMM
        does not read)."""
        o = stage + r * 8 * D.A_OBJ
        return MemTask([{"addr": o, "nwords": 3 * 17 * 40, "dims": dims((20, 1), (2, D.A_OBJ // 8), (17, 20), (3, 340)),
                         "iteration": (D.A_OBJ // 4, 8)}],
                       repeat=8, ddr=(src + 160 * self.blocks[r], 3 * 17 * 160, (40, 17, pos, line), (step, 8)))

    def sends2(self, stage: int) -> dict[int, list[MemTask]]:
        """mode 2: per core row its 8 objects (one task)."""
        return {r: [MemTask([{"addr": stage + r * 8 * D.A_OBJ, "nwords": 8 * D.A_OBJ // 4}])] for r in range(ROWS)}

