"""The MOSS stem's conv1 on the vision overlay: a 3-tap conv over the mel's 80 bins as one GEMM of the host's im2col
panels, the erf-GELU epilogue of the ``ln`` page, and its rows back for the next stage.

Per band: 128 frames (8 columns of one 16-token tile), K = 3 x 80 channels of the im2col panels (the bias the GEMM's
command words), N = the overlay's 1,280 (the encoder's 1,024 channels and their pads). The output is the GELU's bfp16
panels; the host decodes them to the frames' 1,024 channels.
"""

from dataclasses import dataclass

import numpy as np

from artlite.operators import kmath
from artlite.operators.design import COLS
from artlite.vision import layout as L
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import alloc, free, gelu_table, gemm
from artlite.vision.levels import TokenLevel
from artlite.vision.program import Program
from artlite.vision.vit import k_view, vit_level
from artlite.vision.weights import Wt, computed

MEL, CH, N = 80, 1024, 1280  # (the mel's bins, the encoder's channels, the overlay's width)
K = 3 * MEL  # (the im2col panels' real k; padded to the k-tiles' 256)
KP = -(-K // 64) * 64
TAP = 64  # (the token panels' group: 8 rows a panel)
LEVEL = vit_level(24, 16)  # (24 bands of 16-token tiles a column: 3,072 frames, the mel's 3,000)
NAME = "moss.stem1"


def im2col(frames: np.ndarray, *, stride: int = 1) -> np.ndarray:
    """(T, 80) -> (T_out, 240): a row's three taps flattened tap-major, the taps outside zero."""
    t, c = frames.shape
    t_out = (t + 2 - 3) // stride + 1
    xp = np.zeros((t + 2, c), np.float32)
    xp[1:1 + t] = frames
    idx = np.arange(t_out)[:, None] * stride + np.arange(3)[None, :]
    return np.ascontiguousarray(xp[idx].reshape(t_out, 3 * c))


def folded(w: np.ndarray, b: np.ndarray, k: int = KP) -> tuple[np.ndarray, np.ndarray]:
    """The conv's (C_out, C_in, 3) weights and bias -> the GEMM's (1,280, k) rows (tap-major like ``im2col``), and the
    command's bias at the width."""
    c_out, c_in, taps = w.shape
    m = np.zeros((N, k), np.float32)
    m[:c_out, :taps * c_in] = np.asarray(w, np.float32).transpose(0, 2, 1).reshape(c_out, taps * c_in)
    bi = np.zeros(N, np.float32)
    bi[:c_out] = np.asarray(b, np.float32).reshape(-1)[:c_out]
    return m, bi


def band_image(rows: np.ndarray, lv: TokenLevel) -> np.ndarray:
    """A band's rows (COLS * m, k) -> its DDR image: the columns' panels, one bfp16 pack for the whole band (the pack's
    panels are ordered [channel block][token panel], so a column's 16 rows are its two panels)."""
    m, k = rows.shape
    packed = L.panels(np.ascontiguousarray(rows, np.float32))  # (k/8 * m/8 * 72,)
    return np.ascontiguousarray(packed.reshape(k // 8, -1, m // 8 // COLS, L.PANEL)
                                .transpose(1, 0, 2, 3)).reshape(COLS, -1)


def panels(frames: np.ndarray, *, stride: int = 1, k: int = KP) -> np.ndarray:
    """The frames' A panels (bfp16 bytes, ``layout.panels`` of the im2col rows at k columns)."""
    x = im2col(frames, stride=stride)
    t, real = x.shape
    a = np.zeros((t, k), np.float32)
    a[:, :real] = x
    return L.panels(a)


def front(bd: Builder, w: dict[str, Wt | np.ndarray], *, name: str = NAME, lv: TokenLevel = LEVEL) -> None:
    """The conv1's bands: the A panels filled, the GEMM with the erf-GELU epilogue, the panels drained."""
    for b in range(lv.nb):
        a = alloc(bd, lv, KP, "pn")
        bd.op(bd.load(f"{name}.x.{b}", a.addr, a.nbytes), {})
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        y = alloc(bd, lv, N, "pn")
        gemm(bd, k_view(a, w["c1"]), w["c1"], w["c1_b"], name=f"{name}.g{b}", out=y, kind="gelu", hilo=True)
        free(bd, a)
        bd.op(bd.drain(f"{name}.y.{b}", lv.pn(N), mem=y.addr, nbytes=y.nbytes), {})
        free(bd, y)


@dataclass
class Stem1:
    """conv1 on a vision device: its bands in DDR (the A panels and the GELU's), the weights, the program."""

    npu: VisionDevice
    w: dict[str, np.ndarray]
    lv: TokenLevel = LEVEL
    name: str = NAME

    def __post_init__(self) -> None:
        m, b = folded(self.w["c1_w"], self.w["c1_b"])
        bd = Builder(self.npu)
        front(bd, {"c1": computed(f"{self.name}.c1", m), "c1_b": b}, name=self.name, lv=self.lv)
        self.program = Program.of(self.npu, bd)
        self.transaction = self.program.transaction(self.npu)

    def write(self, frames: np.ndarray, *, stride: int = 1) -> None:
        """The mel's frames (T, 80) into the bands' A panels: the whole im2col on the host (its taps' halo rows are the
        neighbours', so the bands are cut after it), then a band's 128 rows packed in one go."""
        cols = im2col(np.asarray(frames, np.float32), stride=stride)
        for b in range(self.lv.nb):
            at = b * COLS * self.lv.m
            k = min(COLS * self.lv.m, max(0, cols.shape[0] - at))
            rows = np.zeros((COLS * self.lv.m, KP), np.float32)
            if k > 0:
                rows[:k, :cols.shape[1]] = cols[at:at + k]
            self.npu.write(f"{self.name}.x.{b}", band_image(rows, self.lv).reshape(-1))

    def read(self) -> np.ndarray:
        """The GELU's rows (the bands' order), fp32 at the encoder's 1,024 channels."""
        out = np.zeros((self.lv.nb * COLS * self.lv.m, N), np.float32)
        for b in range(self.lv.nb):
            data = self.npu.read(f"{self.name}.y.{b}").reshape(COLS, -1)
            for c in range(COLS):
                at = b * COLS * self.lv.m + c * self.lv.m
                out[at:at + self.lv.m] = L.from_panels(data[c], self.lv.m, N)
        return out[:, :CH]

    def warm(self) -> None:
        """One dispatch before the measured ones (a program's first pays its context's configuration)."""
        self.npu.run(self.transaction)

    def run(self) -> float:
        import time

        t0 = time.perf_counter()
        self.npu.run(self.transaction)
        return time.perf_counter() - t0


def assemble_a(src: np.ndarray, bands: int, *, real: int = 1500, cb: int = 128, m: int = 16,
               co: int = COLS) -> np.ndarray:
    """conv2's A panels from conv1's GELU panels, bytes for bytes: conv1's image per column is [channel block][token
    panel][8 tokens][9 B] and a token's block (one channel block of 8 channels) is the A's own block, so the im2col's
    three taps are the same blocks at shifted token indices - no float round trip and no re-rounding.

    ``src``: conv1's panels (bands, COLS, cb, 2, 8, 9) uint8; returns (bands2, COLS, 3 * cb, 2, 8, 9): the stride-2 conv's
    A, its taps' channel blocks first. Blocks whose token falls outside the mel's 3,000 frames are zero."""
    b2 = bands // 2
    out = np.zeros((b2, co, 3 * cb, 2, 8, 9), np.uint8)
    frames = bands * co * m
    for b in range(b2):
        for c in range(co):
            for pA in range(2):
                for i in range(8):
                    j = b * co * m + c * m + pA * 8 + i  # (the output row)
                    if j >= real:  # (the padded rows past the chunk's outputs stay zero)
                        continue
                    for t in range(3):
                        tau = 2 * j + t - 1  # (the source frame)
                        if tau < 0 or tau >= frames:
                            continue
                        sb, sc = tau // (co * m), (tau % (co * m)) // m
                        sp, si = (tau % m) // 8, tau % 8
                        out[b, c, t * cb:(t + 1) * cb, pA, i] = src[sb, sc, :, sp, si]
    return out


K2 = 3 * CH  # (conv2's im2col channels: 3 x 1,024)
KP2 = -(-K2 // 64) * 64  # (3,136: whole k-tiles)
LEVEL2 = vit_level(12, 16)  # (12 bands of 16-token tiles a column: 1,536 rows, the stride-2 conv's 1,500)
NAME2 = "moss.stem2"


def folded2(w: np.ndarray, b: np.ndarray, k: int = KP2) -> tuple[np.ndarray, np.ndarray]:
    """conv2's (1,024, 1,024, 3) weights and bias -> the (1,280, k) GEMM rows and the width's bias."""
    return folded(w, b, k=k)


def panels2(frames: np.ndarray) -> np.ndarray:
    """conv2's A panels (stride 2 over the 1,024-channel frames)."""
    return panels(frames, stride=2, k=KP2)


def front2(bd: Builder, w: dict[str, Wt | np.ndarray], *, name: str = NAME2, lv: TokenLevel = LEVEL2) -> None:
    """conv2's bands: the A panels (K 3,136), the GEMM with the erf-GELU epilogue, the panels drained."""
    for b in range(lv.nb):
        a = alloc(bd, lv, KP2, "pn")
        bd.op(bd.load(f"{name}.x.{b}", a.addr, a.nbytes), {})
        bd.table("gelu_erf", gelu_table(kmath.GELU_ERF))
        y = alloc(bd, lv, N, "pn")
        gemm(bd, k_view(a, w["c2"]), w["c2"], w["c2_b"], name=f"{name}.g{b}", out=y, kind="gelu", hilo=True)
        free(bd, a)
        bd.op(bd.drain(f"{name}.y.{b}", lv.pn(N), mem=y.addr, nbytes=y.nbytes), {})
        free(bd, y)


@dataclass
class Stem2:
    """conv2 on a vision device (stride 2), and the position table's addition on the host (1,500 rows of 1,024: the
    overlay's residual slots pair with a bf16 C-tile output, not the GELU epilogue's panels)."""

    npu: VisionDevice
    w: dict[str, np.ndarray]
    lv: TokenLevel = LEVEL2
    name: str = NAME2

    def __post_init__(self) -> None:
        m, b = folded2(self.w["c2_w"], self.w["c2_b"])
        bd = Builder(self.npu)
        front2(bd, {"c2": computed(f"{self.name}.c2", m), "c2_b": b}, name=self.name, lv=self.lv)
        self.program = Program.of(self.npu, bd)
        self.transaction = self.program.transaction(self.npu)

    def write_bytes(self, src: np.ndarray, bands: int, *, real: int = 1500) -> None:
        """conv2's A panels assembled from conv1's raw panels (``assemble_a``): the bytes path, no float work."""
        a = assemble_a(src, bands, real=real)
        for b in range(self.lv.nb):
            self.npu.write(f"{self.name}.x.{b}", a[b].reshape(-1))

    def write(self, frames: np.ndarray) -> None:
        """The (3,000, 1,024) frames of conv1's GELU into the bands' A panels: the whole stride-2 im2col on the host
        (its taps' halo rows are the neighbours', so the bands are cut after it), then a band's 128 rows a band."""
        cols = im2col(np.asarray(frames, np.float32), stride=2)  # (1500, 3,072)
        for b in range(self.lv.nb):
            at = b * COLS * self.lv.m
            k = min(COLS * self.lv.m, max(0, cols.shape[0] - at))
            rows = np.zeros((COLS * self.lv.m, KP2), np.float32)
            if k > 0:
                rows[:k, :cols.shape[1]] = cols[at:at + k]
            self.npu.write(f"{self.name}.x.{b}", band_image(rows, self.lv).reshape(-1))

    def read(self) -> np.ndarray:
        """The GELU's rows (the bands' order), fp32 at the encoder's 1,024 channels."""
        out = np.zeros((self.lv.nb * COLS * self.lv.m, N), np.float32)
        for b in range(self.lv.nb):
            data = self.npu.read(f"{self.name}.y.{b}").reshape(COLS, -1)
            for c in range(COLS):
                at = b * COLS * self.lv.m + c * self.lv.m
                out[at:at + self.lv.m] = L.from_panels(data[c], self.lv.m, N)
        return out[:, :CH]

    def rows(self, conv1: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """The stem's rows: conv2's GELU of conv1's rows (stride 2) and the position table added."""
        self.write(conv1)
        self.run()
        return self.read()[:1500] + np.asarray(pos, np.float32)[:1500]

    def warm(self) -> None:
        """One dispatch before the measured ones (a program's first pays its context's configuration)."""
        self.npu.run(self.transaction)

    def run(self) -> float:
        import time

        t0 = time.perf_counter()
        self.npu.run(self.transaction)
        return time.perf_counter() - t0
