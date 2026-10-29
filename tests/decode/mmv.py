"""The decode GEMV (``artlite.decode.gemv``) and its phases on the host, for the tests: the x and sum panels
the cores read (``x_panels``, ``xs_panels``), each format's arithmetic on one tile and K chunk as the kernels compute it
(``ref_tile``: fp32 math with the kernels' rounding points), a phase's rows (``ref_rows``), its normed input
(``norm_input``) and output objects (``out_objects``), and ``MmvOperator``, a harness of the kernels on two cores as the
decode places them (also ``artlite.variant_search``'s target, from the repo root:
``'tests.decode.mmv.MmvOperator(fmt="Q5_K", n_tiles=4)'``)."""

from dataclasses import dataclass

import numpy as np
import torch
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import Module
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile

from artlite.decode.design import C_LEN, F_EMITR, F_GU, F_MSG
from artlite.decode.gemv import (
    KB,
    PANEL,
    SLOTS,
    Chunk,
    helper_y,
    render,
    row_split,
    scr_bytes,
    tile_rows_of,
    xs_slot,
    y_floats,
)
from artlite.decode.image import Region, TileImage
from artlite.decode.phases import valid_rows
from artlite.formats import FORMATS, iq3, slice_fields
from artlite.operators import bfp16
from artlite.operators.base import ArgSpec, ArtcOperator
from artlite.operators.bfp16 import blocks
from artlite.operators.design import COLS

# ----------------------------------------------------------------------
# The x and sum panels
# ----------------------------------------------------------------------


def _bytes(e: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.concatenate([e[..., None], m.view(np.uint8)], axis=-1)


def _vals(e: np.ndarray, m: np.ndarray) -> np.ndarray:
    return m.astype(np.float64) * np.ldexp(1.0, e.astype(np.int32) - 133)[..., None]


def _hilo_blocks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S <= 4, K) float -> the panel rows' blocks (8, K / 8) exponents, (8, K / 8, 8) mantissas: [hi of the slots | lo
    of the slots] (zero rows for unused slots), as the kernels split them (bfp16 blocks of 8 along K)."""
    S, K = x.shape
    assert S <= SLOTS and K % KB == 0
    xs = np.zeros((SLOTS, K), np.float32)
    xs[:S] = x
    e, m = blocks(xs.reshape(SLOTS, K // KB, KB))
    hi = _vals(e, m).reshape(SLOTS, K)
    e2, m2 = blocks((xs - hi).astype(np.float32).reshape(SLOTS, K // KB, KB))
    return np.concatenate([e, e2]), np.concatenate([m, m2])


def hilo_rows(x: np.ndarray) -> np.ndarray:
    """(S <= 4, K) float -> (8, K) the panel rows' values [hi of the slots | lo of the slots] (``_hilo_blocks``)."""
    e, m = _hilo_blocks(x)
    return _vals(e, m).reshape(8, -1).astype(np.float64)


def _panel_bytes(x: np.ndarray) -> np.ndarray:
    """(S <= 4, K) float -> its K / 8 panels [a 8][9 B] (one block per row a: ``_hilo_blocks``; the blocks themselves,
    not their values encoded again: a hi mantissa of -128 would come back at a coarser exponent)."""
    e, m = _hilo_blocks(x)
    return _bytes(e, m).transpose(1, 0, 2).reshape(-1, PANEL)


def _x8_blocks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S <= 8, K) float -> the 8-slot panel rows' blocks (8, K / 8) exponents, (8, K / 8, 8) mantissas: the slots' x as
    single bfp16 (zero rows for unused slots: ``decode.gemv``'s designs of 8 slots)."""
    S, K = x.shape
    assert S <= 8 and K % KB == 0
    xs = np.zeros((8, K), np.float32)
    xs[:S] = x
    return blocks(xs.reshape(8, K // KB, KB))


def x_rows(x: np.ndarray, slots: int = SLOTS) -> np.ndarray:
    """(S, K) float -> (8, K) the x panels' row values: [hi of the slots | their lo] (4 slots), the slots' x (8)."""
    if slots == 4:
        return hilo_rows(x)
    e, m = _x8_blocks(x)
    return _vals(e, m).reshape(8, -1)


def _sums(x: np.ndarray, sub: int, slots: int = SLOTS) -> np.ndarray:
    """The slots' sums over sub-blocks of ``sub`` k of the x panels' values (4 slots: hi + lo, the decode sums the panels
    on the MMUL), fp32 (slots, K / sub)."""
    X = x_rows(x, slots)
    if slots == 4:
        X = X[:SLOTS] + X[SLOTS:]
    return X.reshape(slots, x.shape[1] // sub, sub).sum(-1).astype(np.float32)


def _sum_hilo(t: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """(8, n) fp32 sums -> the blocks (exponents, mantissas) of their hi and lo panels (8 slots: two panels per sum
    panel, rows = slots)."""
    e, m = blocks(t.reshape(8, -1, 8))
    hi = _vals(e, m).reshape(8, -1)
    return (e, m), blocks((t - hi).astype(np.float32).reshape(8, -1, 8))


def xs_rows(x: np.ndarray, sub: int = 32, slots: int = SLOTS) -> np.ndarray:
    """The sum panels' row values (8, K / sub) of x (S, K): ``_sums`` as hi + lo rows (4 slots), or each slot's row the
    sum of its hi and lo panels' values (8)."""
    if slots == 4:
        return hilo_rows(_sums(x, sub))
    (e, m), (e2, m2) = _sum_hilo(_sums(x, sub, 8))
    return (_vals(e, m) + _vals(e2, m2)).reshape(8, -1)


def x_panels(x: np.ndarray, slots: int = SLOTS) -> np.ndarray:
    """The x panels (K / 8, 72) uint8 of S <= ``slots`` slots' x (S, K)."""
    if slots == 4:
        return _panel_bytes(x)
    e, m = _x8_blocks(x)
    return _bytes(e, m).transpose(1, 0, 2).reshape(-1, PANEL)


def _sum_panels(t: np.ndarray, slots: int) -> np.ndarray:
    """(slots, 8 n) fp32 sums -> their n sum panels' bytes (n, 72 [4 slots] or 144 [8: hi panel, lo panel])."""
    if slots == 4:
        return _panel_bytes(t)
    (e, m), (e2, m2) = _sum_hilo(t)
    return np.concatenate([_bytes(e, m).transpose(1, 0, 2).reshape(-1, PANEL),
                           _bytes(e2, m2).transpose(1, 0, 2).reshape(-1, PANEL)], 1)


def xs_panels(x: np.ndarray, nsb: int = 4, slots: int = SLOTS) -> np.ndarray:
    """The sum panels per chunk of x (S, K): (K / (256 nsb), ``Chunk(nsb, slots).xs``) uint8 (the chunk's slots)."""
    K, ch = x.shape[1], Chunk(nsb, slots)
    n = K // ch.k
    pb = PANEL * (1 if slots == 4 else 2)  # (a sum panel's bytes: [hi | lo] rows, or a hi and a lo panel)
    p32 = _sum_panels(_sums(x, 32, slots), slots).reshape(n, nsb // 2, 2 * pb)
    p16 = _sum_panels(_sums(x, 16, slots), slots).reshape(n, nsb, 2 * pb)
    out = np.zeros((n, nsb // 2 + nsb, xs_slot(slots)), np.uint8)
    out[:, :nsb // 2, :2 * pb] = p32
    out[:, nsb // 2:, :2 * pb] = p16
    return out.reshape(n, ch.xs)


# ----------------------------------------------------------------------
# Each format's arithmetic on a tile
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MmvX:
    """x of S <= ``slots`` token slots over one K chunk as the decode GEMV's MMUL sees it: ``X`` (8, K) the x panels'
    rows (``x_rows``) and ``xs32``, ``xs16`` (8, K / 32), (8, K / 16) the sum panels' rows (``xs_rows``), float64."""

    X: np.ndarray
    xs32: np.ndarray
    xs16: np.ndarray
    slots: int = SLOTS

    def csum(self, C: np.ndarray) -> np.ndarray:
        """(8 rows a, ...) MMUL sums -> a sub-block's c (slots, ...): the slots' sums of their hi and lo rows, rounded to
        bf16 (4 slots), or the rows rounded (8)."""
        return _csum(C) if self.slots == 4 else _bf64(C)

    def fold(self, o: np.ndarray) -> np.ndarray:
        """(8 rows a, ...) -> (slots, ...): the hi and lo rows added (4 slots)."""
        return o[:SLOTS] + o[SLOTS:] if self.slots == 4 else o


def _bf64(x: np.ndarray) -> np.ndarray:
    """x rounded to bf16 (as float64)."""
    return torch.from_numpy(np.asarray(x, np.float32)).bfloat16().double().numpy()


def _hilo(x: np.ndarray) -> np.ndarray:
    """fp32 x as the sum of its bf16 hi and lo parts."""
    x = np.asarray(x, np.float32)
    h = _bf64(x)
    return h + _bf64(x - h)


def _csum(C: np.ndarray) -> np.ndarray:
    """(8 rows a, ...) MMUL sums -> the slots' sums of their hi and lo rows, rounded to bf16 (a sub-block's c)."""
    return _bf64(C[:SLOTS] + C[SLOTS:])


def _q4(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    # (centered by the 32-k sum panels' rows: sum q X_a - 8 XS_a)
    q = f["q"].astype(np.float64)
    R, K = q.shape
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 32, 32), q.reshape(R, K // 32, 32)) - 8 * x.xs32[..., None]
    d = _bf64(f["d"]).reshape(R, K // 32)
    y = (_csum(C) * d.T[None]).sum(1)
    if fmt == "Q4_0":
        return y
    # Q4_1's offsets: XS against M = m + 8 d as bfp16 hi + lo blocks of 8 groups, the sums in fp32
    M = (_bf64(f["m"]).reshape(R, K // 32) + 8 * d).astype(np.float32)
    Mh = bfp16.values(M)
    Mhl = Mh.astype(np.float64) + bfp16.values((M - Mh).astype(np.float32))
    o = x.xs32 @ Mhl.T  # (8, R)
    return y + o[:SLOTS] + o[SLOTS:]


def _q8(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    q = f["q"].astype(np.float64)
    R, K = q.shape
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 32, 32), q.reshape(R, K // 32, 32))
    return (_csum(C) * _bf64(f["d"]).reshape(R, K // 32).T[None]).sum(1)


def _scale_min(f: dict[str, np.ndarray], x: MmvX, z: int) -> np.ndarray:
    """The scale / min formats (Q4_K: z 8, Q5_K: z 16): the codes centered by the sum panels' rows (sum q X_a - z XS_a),
    per sub-block c (d sc) with d sc as hi + lo, the offsets hilo(XS sc) z d + hilo(XS mn) (-dmin) (the integer sums
    on the MMUL)."""
    q = f["q"].astype(np.float64)  # (R, K)
    R, K = q.shape
    nsb = K // 256
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 32, 32), q.reshape(R, K // 32, 32)) - z * x.xs32[..., None]
    sc = f["sc"].astype(np.float64).reshape(R, K // 32)
    d = _bf64(f["d"]).reshape(R, nsb)
    dm = _bf64(f["dmin"]).reshape(R, nsb)
    TD = _hilo(np.repeat(d, 8, 1) * sc)  # (R, sub-block)
    y = (x.csum(C) * TD.T[None]).sum(1)
    XS = x.xs32.reshape(8, nsb, 8)
    Ms = np.einsum("asi,rsi->asr", XS, sc.reshape(R, nsb, 8))
    Mm = np.einsum("asi,rsi->asr", XS, f["mn"].astype(np.float64).reshape(R, nsb, 8))
    o = (_hilo(Ms) * (z * d).T[None] - _hilo(Mm) * dm.T[None]).sum(1)
    return y + x.fold(o)


def _q4k(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    return _scale_min(f, x, 8)


def _q5k(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    return _scale_min(f, x, 16)


def _q6k(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    # (centered by the rows of the 16-k sum panels: sum q X_a - 32 XS16_a)
    q = f["q"].astype(np.float64)
    R, K = q.shape
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 16, 16), q.reshape(R, K // 16, 16)) - 32 * x.xs16[..., None]
    TD = _hilo(np.repeat(_bf64(f["d"]).reshape(R, K // 256), 16, 1) * f["sc"].astype(np.float64).reshape(R, K // 16))
    return (x.csum(C) * TD.T[None]).sum(1)


def _q2k(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    # (per super-block: hilo(sum c sc + 2 XS16 sc) d - hilo(XS16 mn) dmin, the codes centered against the panels' rows:
    # c = bf16(sum (q - 2) X_a); sums in fp32)
    q = f["q"].astype(np.float64)
    R, K = q.shape
    nsb = K // 256
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 16, 16), q.reshape(R, K // 16, 16) - 2)
    sc, mn = f["sc"].astype(np.float64).reshape(R, nsb, 16), f["mn"].astype(np.float64).reshape(R, nsb, 16)
    d, dm = _bf64(f["d"]).reshape(R, nsb), _bf64(f["dmin"]).reshape(R, nsb)
    XS = x.xs16.reshape(8, nsb, 16)
    Ms, Mm = np.einsum("abj,rbj->abr", XS, sc), np.einsum("abj,rbj->abr", XS, mn)  # (8, nsb, R)
    Y = np.einsum("sbjr,rbj->sbr", _csum(C).reshape(SLOTS, nsb, 16, R), sc) + 2 * (Ms[:SLOTS] + Ms[SLOTS:])
    o = _hilo(Mm) * dm.T[None]
    return (_hilo(Y) * d.T[None]).sum(1) - (o[:SLOTS] + o[SLOTS:]).sum(1)


def _q3k(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    # (w = d sc (q - 4): per super-block hilo(sum c sc) d, c = bf16(sum (q - 4) X_a): the codes centered against the
    # panels' rows; sums in fp32)
    q = f["q"].astype(np.float64) - 4
    R, K = q.shape
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 16, 16), q.reshape(R, K // 16, 16))
    nsb = K // 256
    Y = np.einsum("sbjr,rbj->sbr", _csum(C).reshape(4, nsb, 16, R), f["sc"].astype(np.float64).reshape(R, nsb, 16))
    return (_hilo(Y) * _bf64(f["d"]).reshape(R, nsb).T[None]).sum(1)


def _iq3(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    # (w = d (2 ls + 1) / 4 q: per super-block hilo(sum c (2 ls + 1) / 4) d, c = bf16(sum q X_a) per 32-k group; sums in
    # fp32)
    q = f["q"].astype(np.float64)
    R, K = q.shape
    C = np.einsum("ask,rsk->asr", x.X.reshape(8, K // 32, 32), q.reshape(R, K // 32, 32))
    nsb = K // 256
    S = (2 * f["ls"].astype(np.float64) + 1) / 4
    Y = np.einsum("sbir,rbi->sbr", _csum(C).reshape(4, nsb, 8, R), S.reshape(R, nsb, 8))
    return (_hilo(Y) * _bf64(f["d"]).reshape(R, nsb).T[None]).sum(1)


def _bfp16(fmt: str, f: dict[str, np.ndarray], x: MmvX) -> np.ndarray:
    w = f["w"]  # (a BF16 tensor's bit patterns or an F16 tensor's values)
    w = w.astype(np.float32) if w.dtype == np.float16 else \
        torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).view(torch.bfloat16).float().numpy()
    C = x.X @ bfp16.values(w).astype(np.float64).T  # (8, R): fp32 sums of exact products
    return C[:SLOTS] + C[SLOTS:]


_REFS = {"Q4_0": _q4, "Q4_1": _q4, "Q8_0": _q8, "Q4_K": _q4k, "Q5_K": _q5k, "Q6_K": _q6k, "Q2_K": _q2k, "Q3_K": _q3k,
         "IQ3_XXS": _iq3, "BFP16": _bfp16}


def ref_tile(fmt: str, f: dict[str, np.ndarray], x: np.ndarray, slots: int = SLOTS) -> np.ndarray:
    """The kernels' arithmetic on one tile and K chunk (the fields ``f`` of its R rows): (slots, R) float64 against
    x (S, 256 nsb), for designs of ``slots`` slots."""
    assert slots == 4 or fmt in ("Q4_K", "Q6_K"), (fmt, slots)
    return _REFS[fmt](fmt, f, MmvX(x_rows(x, slots), xs_rows(x, 32, slots), xs_rows(x, 16, slots), slots))


def tile_rows(y: np.ndarray, R: int, slots: int = SLOTS) -> np.ndarray:
    """A tile's output (..., 8 slots g) [g][slot][r] (g row groups) -> (..., slots, R)."""
    g = y.shape[-1] // (8 * slots)
    y = np.asarray(y).reshape(*y.shape[:-1], g, slots, 8)
    return np.swapaxes(y, -3, -2).reshape(*y.shape[:-3], slots, 8 * g)[..., :R]


# ----------------------------------------------------------------------
# A phase (``decode.phases``)
# ----------------------------------------------------------------------


def bf(x: np.ndarray | torch.Tensor) -> np.ndarray:
    return torch.as_tensor(np.asarray(x, np.float32)).bfloat16().float().numpy()


def norm_input(x: np.ndarray, w: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """F_NORM of S slots' bf16 x (S, H): (x w in fp32 (S, H), 1 / rms(x) (S,))."""
    x = bf(x).astype(np.float64)
    inv = 1 / np.sqrt((x ** 2).mean(-1) + eps)
    return (x * bf(w)).astype(np.float32), inv


def ref_rows(image: TileImage, region: Region, c: int, x: np.ndarray) -> list[np.ndarray]:
    """The rows of column c's blocks against S slots' inputs x (S, K) (fp32 values as the cores take them): per block
    (S, valid rows) float64, as the cores compute them (``ref_tile`` per tile and K chunk, summed)."""
    S, KC = x.shape[0], region.kc
    g = image.gguf
    out = []
    for b in region.blocks:
        tensors = [g.tensor(t) for t in b.tensors]
        fields = [t.fields() for t in tensors]
        rows = b.rows[c]
        rows = rows[rows >= 0]
        y = np.zeros((S, len(rows)))
        for t0 in range(0, len(rows), b.R):
            grp = rows[t0:t0 + b.R]
            acc = np.zeros((SLOTS, len(grp)))
            for k in range(region.n_kc):
                parts = []
                lo = 0
                for t, f in zip(tensors, fields):
                    sel = (grp >= lo) & (grp < lo + t.rows)
                    parts.append((sel, slice_fields(f, grp[sel] - lo, k0=k * KC, cols=t.cols, kc=KC)))
                    lo += t.rows
                fc = {key: np.zeros((len(grp), *v.shape[1:]), v.dtype) for key, v in parts[0][1].items()}
                for sel, f in parts:
                    for key, v in fc.items():
                        v[sel] = f[key]
                acc += ref_tile(b.fmt, fc, x[:, k * KC:(k + 1) * KC])
            y[:, t0:t0 + len(grp)] = acc[:S]
        out.append(y)
    return out


def _out_elements(region: Region, flags: int, *, f32_ab: bool = False) -> int:
    """Output values (bf16 slots) of a phase per column and slot (C objects of C_LEN each; the last zero-padded)."""
    if flags & F_GU:
        return sum(valid_rows(region)) // 2
    f32 = flags & F_MSG or f32_ab
    return sum(n * (2 if f32 and b.name == "ab" else 1) for b, n in zip(region.blocks, valid_rows(region)))


def out_objects(region: Region, flags: int, hidden: int, *, f32_ab: bool = False) -> int:
    """C objects of a phase per column and slot (with F_EMITR: the residual slice)."""
    if flags & F_EMITR:
        return hidden // COLS // C_LEN
    return -(-_out_elements(region, flags, f32_ab=f32_ab) // C_LEN)


# ----------------------------------------------------------------------
# The harness
# ----------------------------------------------------------------------


class MmvOperator(ArtcOperator):
    """n_tiles tiles of one format and K chunk, each run by two cores (0, 2) / (0, 3) as the decode runs them (K-quants:
    super-blocks 0, 1 / 2, 3; row-split formats: their halves of the rows): the second core reads the first one's tile
    objects through shared memory (their address from a command FIFO), the first writes the chunk's x and sum panels
    into the second one's memory.

    Args: A (n_tiles tile_bytes uint8), X (x panels + sum panels of a chunk, uint8), OUT (n_tiles x ``y_floats`` fp32:
    [g][slot][r])."""

    def __init__(self, *, fmt: str = "Q4_K", n_tiles: int = 4, tile_bytes: int = 9280, nsb: int = 4, win: bool = False,
                 slots: int = SLOTS):
        """win: event0 / event1 around each core's half tile (the profiling skill's counters); slots: the design's token
        slots (``decode.gemv``)."""
        self.fmt, self.n_tiles, self.tile_bytes, self.win, self.slots = fmt, n_tiles, tile_bytes, win, slots
        self.chunk = Chunk(nsb, slots)
        self.rows = tile_rows_of(fmt, tile_bytes, nsb)
        self.hy = helper_y(fmt, nsb)  # (where the second core's rows go)
        self.yn = 128 if self.rows > 16 and not row_split(fmt, nsb) else y_floats(slots)  # (a core's output)
        assert slots == 4 or self.rows <= 16, (fmt, slots)
        self.y_floats = self.yn + self.hy
        super().__init__()

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.n_tiles * self.tile_bytes,), torch.uint8),
                ArgSpec("in", (self.chunk.xp + self.chunk.xs,), torch.uint8),
                ArgSpec("out", (self.n_tiles * self.y_floats,), torch.float32)]

    def get_kernel_src(self) -> str:
        fid, ch = FORMATS[self.fmt].fid, self.chunk
        n = ch.xp + ch.xs
        x1 = 0 if row_split(self.fmt, ch.nsb) else ch.xp // 2  # (the second core's panels)
        return render("mmvh", [self.fmt], tile_bytes=self.tile_bytes, nsb=ch.nsb, slots=self.slots) + f"""
extern "C" {{
void mmvh_copy(const uint8_t *__restrict x, uint8_t *__restrict xb)
{{
    al::copy<{n // 4}, 8>((int32_t *)xb, (const int32_t *)x);
}}

// the first core: the command for the second (a as the second core sees it: the first core's memory is its south
// neighbour's), its part of the tile a
void mmvh_tile0(uint8_t *__restrict a, const uint8_t *__restrict x, int32_t *__restrict cmd, uint8_t *__restrict scr,
                float *__restrict y)
{{
    cmd[0] = (int32_t)(uintptr_t)a - 0x30000;
    al::zero<{self.yn}, 32>(y);
    {"event0();" if self.win else ""}
    mmvh_mmv<0>(a, x, x + {ch.xp}, y, scr, {fid});
    {"event1();" if self.win else ""}
}}

void mmvh_tile1(const int32_t *__restrict cmd, const uint8_t *__restrict x, uint8_t *__restrict scr,
                float *__restrict y)
{{
    al::zero<{self.yn}, 32>(y);
    {"event0();" if self.win else ""}
    mmvh_mmv<1>((uint8_t *)(uintptr_t)cmd[0], x + {x1}, x + {ch.xp}, y, scr, {fid});
    {"event1();" if self.win else ""}
}}

// the tile's output: the second core's rows added (K-quants) or after the first's (row-split formats)
void mmvh_add(const float *__restrict r, const float *__restrict y, float *__restrict out)
{{
    for (int i = 0; i < {self.yn}; i += 32) {{
        const aie::vector<float, 32> a = aie::load_v<32>(y + i), b = aie::load_v<32>(r + i);
        if ({self.hy}) {{
            aie::store_v(out + i, a);
            aie::store_v(out + {self.hy} + i, b);
        }} else {{
            aie::store_v(out + i, aie::add(a, b));
        }}
    }}
}}
}}
"""

    def get_mlir_artifact(self) -> Module:
        u8, f32, i32 = np.dtype[np.uint8], np.dtype[np.float32], np.dtype[np.int32]
        a_ty = np.ndarray[(self.tile_bytes,), u8]
        x_ty = np.ndarray[(self.chunk.xp + self.chunk.xs,), u8]
        y_ty = np.ndarray[(self.yn,), f32]
        o_ty = np.ndarray[(self.y_floats,), f32]
        s_ty = np.ndarray[(scr_bytes(self.slots),), u8]
        cmd_ty = np.ndarray[(16,), i32]
        k_copy = Kernel("mmvh_copy", "kernel.o", [x_ty, x_ty])
        k0 = Kernel("mmvh_tile0", "kernel.o", [a_ty, x_ty, cmd_ty, s_ty, y_ty])
        k1 = Kernel("mmvh_tile1", "kernel.o", [cmd_ty, x_ty, s_ty, y_ty])
        k_add = Kernel("mmvh_add", "kernel.o", [y_ty, y_ty, o_ty])
        a_l3 = ObjectFifo(a_ty, name="mA", depth=4)
        a_fifo = a_l3.cons(depth=4).forward(tile=Tile(0, 1), depth=2, name="mA_l1")
        x_fifo = ObjectFifo(x_ty, name="mX", depth=1)
        cmd = ObjectFifo(cmd_ty, name="mcmd", depth=2)
        res = ObjectFifo(y_ty, name="mres", depth=2)
        y_fifo = ObjectFifo(o_ty, name="mY", depth=2)
        x_buf = Buffer(x_ty, name="m_x", tile=Tile(0, 3))  # (the panels in the second core's memory, as the decode)
        scr0, scr1, yb = Buffer(s_ty, name="m_scr0"), Buffer(s_ty, name="m_scr1"), Buffer(y_ty, name="m_y0")
        n = self.n_tiles

        def body0(af, xf, cf, rf, yf, xb, s, y0, kc, kt, ka):
            x = xf.acquire(1)
            kc(x, xb)
            xf.release(1)
            for _ in range_(n):
                a = af.acquire(1)
                c = cf.acquire(1)
                kt(a, xb, c, s, y0)
                cf.release(1)
                r = rf.acquire(1)
                y = yf.acquire(1)
                ka(r, y0, y)
                yf.release(1)
                rf.release(1)
                af.release(1)

        def body1(cf, rf, xb, s, kt):
            for _ in range_(n):
                c = cf.acquire(1)
                r = rf.acquire(1)
                kt(c, xb, s, r)
                rf.release(1)
                cf.release(1)

        grid = []  # (IQ3_XXS: the cores' grid tables, placed, used through their addresses only)
        if self.fmt == "IQ3_XXS":
            t_ty = np.ndarray[(iq3.TAB_BYTES // 4,), np.dtype[np.uint32]]
            grid = [[Buffer(t_ty, tile=Tile(0, 2 + h), name=f"m_grid{h}{i}", address=iq3.TAB[h][i] - 0x70000,
                            initial_value=iq3.grid_words()) for i in range(2)] for h in range(2)]

        def core0(af, xf, cf, rf, yf, xb, s, y0, kc, kt, ka, *tabs):
            body0(af, xf, cf, rf, yf, xb, s, y0, kc, kt, ka)

        def core1(cf, rf, xb, s, kt, *tabs):
            body1(cf, rf, xb, s, kt)

        w0 = Worker(core0, [a_fifo.cons(), x_fifo.cons(), cmd.prod(), res.cons(), y_fifo.prod(), x_buf, scr0, yb,
                            k_copy, k0, k_add, *(grid[0] if grid else [])], tile=Tile(0, 2), stack_size=2048)
        w1 = Worker(core1, [cmd.cons(), res.prod(), x_buf, scr1, k1, *(grid[1] if grid else [])], tile=Tile(0, 3),
                    stack_size=2048)
        n_a = n * self.tile_bytes

        def seq(A, X, OUT, a_h, x_h, y_h):
            a_h.fill(A, TensorAccessPattern((n_a,), 0, [1, 1, 1, n_a], [0, 0, 0, 1]))
            x_h.fill(X)
            y_h.drain(OUT, wait=True)

        rt = Runtime(seq, [np.ndarray[(n_a,), u8], x_ty, np.ndarray[(n * self.y_floats,), f32],
                           a_l3.prod(tile=Tile(0, 0)), x_fifo.prod(tile=Tile(0, 0)), y_fifo.cons(tile=Tile(0, 0))])
        return Program(NPU2(), rt, workers=[w0, w1]).resolve_program()
