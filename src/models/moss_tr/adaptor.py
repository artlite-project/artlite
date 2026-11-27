"""MOSS-Transcribe's VQ adaptor on the vision overlay: the 4-row merge's 4,096 channels -> 1,024 with SiLU -> 1,024, the
LayerNorm and the rows the decoder reads.

The merge takes four consecutive encoder rows per output row (the 4 x 1,024 channels in that order); the adaptor is the
checkpoint's ``model.vq_adaptor``: Linear(4,096 -> 1,024) + b, SiLU, Linear(1,024 -> 1,024) + b, LayerNorm(1,024,
eps 1e-6, gamma and beta). A 30 s chunk's 1,500 rows give 375 merged rows: the overlay's level is 3 bands of one 16-token
tile a column (8 columns = 128 rows a band), 384 rows with the 9 past the chunk zero (their outputs are dropped).

One dispatch: per band the merged rows' panels (K 4,096) -> the first linear with the ``r`` page's SiLU epilogue (the
activation table, ``kmath.SILU``) -> the second linear (K 1,280, the first one's padded width: its pads' SiLU is zero) with
the LayerNorm's records -> the ``ln`` page's apply over the 1,024 real channels -> panels drained to DDR for the host.
The two pages are the layer's (``ln`` 7,056 B, ``r`` 5,184 B): the resident code and either fit 16,384 B.
"""

from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from artlite.operators.design import COLS
from artlite.vision import layout as L
from artlite.vision.builder import Builder
from artlite.vision.device import VisionDevice
from artlite.vision.layers import alloc, free, gemm, silu_words
from artlite.vision.levels import TokenLevel
from artlite.vision.ops import Gemm
from artlite.vision.program import Program
from artlite.vision.vit import k_view, vit_level
from artlite.vision.vit import ln as apply_ln
from artlite.vision.weights import Wt, computed

LEVEL = vit_level(3, 16)  # (3 bands of one 16-token tile a column: 384 rows, the chunk's 375)
MERGE, HID, C = 4, 1024, 1280  # (rows a merged row, the adaptor's hidden, the overlay's width)
EPS = 1e-6  # (the adaptor's LayerNorm)
NAME = "moss.ad"


def merged(rows: np.ndarray) -> np.ndarray:
    """The encoder's rows (1,500, 1,024) -> the merge's rows (375, 4,096): four consecutive rows a row, in order."""
    x = np.asarray(rows, np.float32)
    t = x.shape[0] // MERGE
    return np.ascontiguousarray(x[:t * MERGE].reshape(t, MERGE * x.shape[1]))


def weights(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The adaptor's matrices and vectors in the program's layouts, from the checkpoint's tensors (fp32): ``a0_w`` /
    ``a0_b`` (4,096 -> 1,024, rows padded to the overlay's 1,280), ``a2_w`` / ``a2_b`` (1,024 -> 1,024, the columns of its
    K at the first one's width), the LayerNorm's gamma / beta at the width."""
    out = {}
    fc1 = np.zeros((C, MERGE * HID), np.float32)
    fc1[:HID, :] = np.asarray(w["a0_w"], np.float32)[:HID, :MERGE * HID]
    fc1_b = np.zeros(C, np.float32)
    fc1_b[:HID] = np.asarray(w["a0_b"], np.float32).reshape(-1)[:HID]
    fc2 = np.zeros((C, C), np.float32)
    fc2[:HID, :HID] = np.asarray(w["a2_w"], np.float32)[:HID, :HID]
    fc2_b = np.zeros(C, np.float32)
    fc2_b[:HID] = np.asarray(w["a2_b"], np.float32).reshape(-1)[:HID]
    out.update({"fc1": fc1, "fc1_b": fc1_b, "fc2": fc2, "fc2_b": fc2_b})
    for tag, key in (("ln_g", "a3_w"), ("ln_b", "a3_b")):
        v = np.zeros(C, np.float32)
        v[:HID] = np.asarray(w[key], np.float32).reshape(-1)[:HID]
        out[tag] = v
    return out


def images(w: dict[str, np.ndarray], name: str = NAME) -> dict[str, Wt | np.ndarray]:
    """The adaptor's weight images (``computed``: packed to bfp16 by ``Slots``), the vectors as they are."""
    out = weights(w)
    return {k: (computed(f"{name}.{k}", v) if k in ("fc1", "fc2") else v) for k, v in out.items()}


def as_bf16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def as_bfp16(x: np.ndarray) -> np.ndarray:
    return L.from_panels(L.panels(np.asarray(x, np.float32)), *x.shape)


def silu(x: np.ndarray) -> np.ndarray:
    """exact SiLU (x / (1 + e^-x)), float64 in / float64 out."""
    v = np.asarray(x, np.float64)
    return v / (1.0 + np.exp(-v))


def arithmetic(x: np.ndarray, w: dict[str, np.ndarray], *, real: int = HID) -> np.ndarray:
    """The program's arithmetic on the merged rows x (rows, 4,096) fp32: the bf16 input (rows padded to the panels'
    groups of 8, as the level's 384-row grid), the bfp16 panels and weights, the command's bf16 biases, SiLU, the
    second linear, the LayerNorm (fp32 statistics over ``real`` channels) and the output's bfp16 rounding."""
    n = x.shape[0]
    xp = np.zeros((-(-n // 8) * 8, x.shape[1]), np.float32)
    xp[:n] = as_bf16(x)
    a = as_bfp16(xp)
    h = as_bfp16(silu(as_bfp16(as_bfp16(a) @ as_bfp16(w["fc1"]).T + as_bf16(w["fc1_b"]))))
    y = as_bfp16(h) @ as_bfp16(w["fc2"]).T + as_bf16(w["fc2_b"])
    z = np.zeros((y.shape[0], C), np.float32)
    z[:, :real] = y[:, :real]
    v = np.asarray(z, np.float64)[:, :real]
    mean, var = v.mean(1, keepdims=True), v.var(1, keepdims=True)
    nrm = (np.asarray(z, np.float64) - mean) / np.sqrt(var + EPS) * np.asarray(w["ln_g"], np.float64) + \
        np.asarray(w["ln_b"], np.float64)
    return as_bfp16(nrm)[:n, :real].astype(np.float32)


def reference(x: np.ndarray, w: dict[str, np.ndarray], *, real: int = HID, eps: float = EPS) -> np.ndarray:
    """The checkpoint's adaptor in float64 on the bf16-rounded rows: Linear, SiLU, Linear, the LayerNorm."""
    v = np.asarray(as_bf16(x), np.float64)
    h = silu(v @ np.asarray(w["fc1"], np.float64).T + np.asarray(w["fc1_b"], np.float64))
    y = h @ np.asarray(w["fc2"], np.float64).T + np.asarray(w["fc2_b"], np.float64)
    q = y[:, :real]
    mean, var = q.mean(1, keepdims=True), q.var(1, keepdims=True)
    n = (q - mean) / np.sqrt(var + eps) * np.asarray(w["ln_g"], np.float64)[:real] + \
        np.asarray(w["ln_b"], np.float64)[:real]
    return np.asarray(n, np.float32)


def front(bd: Builder, w: dict[str, Wt | np.ndarray], *, name: str = NAME, lv: TokenLevel = LEVEL) -> None:
    """The adaptor's one dispatch: per band the merged rows' panels, the two linears (SiLU between) and the LayerNorm,
    the rows' panels drained to ``{name}.y.{band}``."""
    xin, xout = f"{name}.x", f"{name}.y"
    for b in range(lv.nb):
        a = alloc(bd, lv, MERGE * HID, "pn")
        bd.op(bd.load(f"{xin}.{b}", a.addr, a.nbytes), {})
        bd.table("silu", silu_words())
        h = alloc(bd, lv, C, "pn")
        bd.gemm(Gemm(m=lv.m, ntile=lv.ntile, K=MERGE * HID, N=C, x=a.addr, y=h.addr, out="gelu", act="silu",
                     x_np=a.np, y_np=h.np), w["fc1"], w["fc1_b"], name=f"{name}.fc1.{b}")
        free(bd, a)
        y = alloc(bd, lv, C)
        rec = gemm(bd, k_view(h, w["fc2"]), w["fc2"], w["fc2_b"], name=f"{name}.fc2.{b}", out=y, ln=True)
        free(bd, h)
        n = alloc(bd, lv, C, "pn")
        apply_ln(bd, y, n, w["ln_g"], w["ln_b"], name=f"{name}.lnf.{b}", rec=rec, C=C, H=HID, eps=EPS)
        free(bd, y)
        bd.op(bd.drain(f"{xout}.{b}", lv.pn(C), mem=n.addr, nbytes=n.nbytes), {})
        free(bd, n)


@dataclass
class Adaptor:
    """The adaptor on a vision device: the merged rows' images (``{name}.x.{band}``: a band's 128 rows a column, the
    host packs them), the output's (``{name}.y.{band}``), the weights, the program and its transaction."""

    npu: VisionDevice
    w: dict[str, np.ndarray]
    lv: TokenLevel = LEVEL
    name: str = NAME

    def __post_init__(self) -> None:
        bd = Builder(self.npu)
        front(bd, images(self.w, self.name), name=self.name, lv=self.lv)
        self.program = Program.of(self.npu, bd)
        self.transaction = self.program.transaction(self.npu)

    def rows_image(self, x: np.ndarray) -> None:
        """The merged rows (n = 128 a band; the 9 past the chunk zero) into the input's band images."""
        n_per_band = COLS * self.lv.m  # (128)
        for b in range(self.lv.nb):
            data = np.zeros((COLS, self.lv.pn(MERGE * HID)), np.uint8)
            for c in range(COLS):
                rows = np.zeros((self.lv.m, MERGE * HID), np.float32)
                at = b * n_per_band + c * self.lv.m
                k = min(self.lv.m, max(0, x.shape[0] - at))
                if k > 0:
                    rows[:k] = x[at:at + k]
                data[c] = L.panels(rows).reshape(-1)
            self.npu.write(f"{self.name}.x.{b}", data.reshape(-1))

    def read(self) -> np.ndarray:
        """The output's rows (the level's order), fp32 (bfp16 panels) at the 1,024 real channels."""
        out = np.zeros((self.lv.nb * COLS * self.lv.m, C), np.float32)
        for b in range(self.lv.nb):
            data = self.npu.read(f"{self.name}.y.{b}").reshape(COLS, -1)
            for c in range(COLS):
                at = b * COLS * self.lv.m + c * self.lv.m
                out[at:at + self.lv.m] = L.from_panels(data[c], self.lv.m, C)
        return out[:, :HID]

    def warm(self) -> None:
        """One dispatch before the measured ones (a program's first pays its context's configuration)."""
        self.npu.run(self.transaction)

    def run(self) -> float:
        """One dispatch (blocking); its seconds."""
        import time

        t0 = time.perf_counter()
        self.npu.run(self.transaction)
        return time.perf_counter() - t0
