"""The decode overlay's multi-slot GEMV (operators/overlay/gemv.py and the slots' ``mmv/<fmt>`` kernels) on every tile
format it reads: the C++ tile packer against the slots' ``pack_tile``, and the two cores' rows of real tiles
(``mmv.MmvOperator``) against the kernels' arithmetic (``mmv.ref_tile``) and the exact product (the tiles' bf16
scales)."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import gemma_gguf, llama_gguf
from artlite.operators import bfp16
from artlite.operators.formats import FORMATS, pack_tile, slice_fields, tile_format
from artlite.operators.overlay import gemv
from artlite.operators.overlay.image import pack_rows

from . import mmv

# per GGUF block: (columns, bytes, the offsets of its fp16 fields)
BLOCK = {"Q4_0": (32, 18, (0,)), "Q4_1": (32, 20, (0, 2)), "Q8_0": (32, 34, (0,)), "Q4_K": (256, 144, (0, 2)),
         "Q5_K": (256, 176, (0, 2)), "Q6_K": (256, 210, (208,)), "Q2_K": (256, 84, (80, 82)), "Q3_K": (256, 110, (108,))}
# (GGUF type, tile bytes, super-blocks per tile) the packer is checked at: Gemma's 512-column tiles, Llama's and Qwen's
# 1024 (the tile sizes of the images that hold them)
PACK = [("Q4_0", 9216, 2), ("Q8_0", 9216, 2), ("Q4_K", 9280, 4), ("Q5_K", 9280, 4), ("Q6_K", 9280, 4), ("Q4_0", 9280, 4),
        ("Q4_1", 9280, 4), ("Q4_1", 10240, 4), ("Q2_K", 9408, 4), ("Q3_K", 9408, 4), ("Q3_K", 9856, 4), ("BF16", 9216, 4),
        ("F16", 9216, 4)]


def _random_tensor(qtype: str, rows: int, cols: int, seed: int) -> QTensor:
    """Random GGUF blocks (fp16 scales of a plausible range), or random BF16 / F16 values."""
    rng = np.random.default_rng(seed)
    if qtype in ("BF16", "F16"):
        w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
        raw = w.astype(np.float16) if qtype == "F16" else torch.from_numpy(w).bfloat16().view(torch.int16).numpy()
        return QTensor(name="t", qtype=qtype, rows=rows, cols=cols, raw=raw.view(np.uint16 if qtype == "BF16" else np.float16))
    per, nb, f16 = BLOCK[qtype]
    n = cols // per
    raw = rng.integers(0, 256, (rows, n, nb), dtype=np.uint8)
    for o in f16:
        d = (rng.standard_normal((rows, n)) * 0.01).astype(np.float16)
        raw[..., o:o + 2] = d.view(np.uint8).reshape(rows, n, 2)
    return QTensor(name="t", qtype=qtype, rows=rows, cols=cols, raw=raw.reshape(rows, -1))


@pytest.mark.parametrize(("qtype", "S_BYTES", "NSB"), PACK, ids=[f"{f}-{s}-nsb{n}" for f, s, n in PACK])
def test_pack_native(qtype: str, S_BYTES: int, NSB: int) -> None:
    """The C++ packer (pack_rows) is bit-identical to the slot's pack_tile: two stacked tensors, rows in a random order
    with padding rows, 3 K chunks."""
    fmt = tile_format(qtype)
    KC = 256 * NSB
    R = FORMATS[fmt].rows(S_BYTES, NSB, gemv.row_split(fmt, NSB) if gemv.supported(fmt, NSB) else 0)
    a, b = _random_tensor(qtype, 40, 3 * KC, 1), _random_tensor(qtype, 9, 3 * KC, 2)
    rows = np.concatenate([np.random.default_rng(3).permutation(49), [-1] * 5])
    rows = np.concatenate([rows, np.full((-rows.size) % R, -1)])
    tiles = pack_rows([a, b], rows, fmt, R, S_BYTES, 3, kc=KC)
    # (a padding row: a zero GGUF row, as the packer reads it; Q3_K's scales then decode to -32)
    zero = QTensor(name="z", qtype=qtype, rows=1, cols=a.cols, raw=np.zeros_like(a.raw[:1]))
    fa, fb, fz = a.fields(), b.fields(), zero.fields()
    for g in range(rows.size // R):
        for k in range(3):
            per = []
            for r in rows[g * R:(g + 1) * R]:
                src, idx = (fz, 0) if r < 0 else ((fa, r) if r < 40 else (fb, r - 40))
                per.append(slice_fields(src, np.array([idx]), k0=k * KC, cols=a.cols, kc=KC))
            f = {key: np.concatenate([p[key] for p in per]) for key in per[0]}
            assert np.array_equal(tiles[g, k], pack_tile(fmt, f, S_BYTES)), (g, k)


def _bf(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(x, np.float32)).bfloat16().double().numpy()


def _weights(fmt: str, f: dict[str, np.ndarray]) -> np.ndarray:
    """(R, K) the tile's weights as the kernels take them: GGUF's formulas with the scales in bf16 (the tiles')."""
    q = f["q"].astype(np.float64) if "q" in f else None

    def per(v: np.ndarray, n: int) -> np.ndarray:  # a value per n columns
        return np.repeat(_bf(v).reshape(v.shape[0], -1), n, 1)

    if fmt == "BFP16":
        w = f["w"]
        w = w.astype(np.float32) if w.dtype == np.float16 else torch.from_numpy(w.view(np.int16)).view(torch.bfloat16).float().numpy()
        return bfp16.values(w).astype(np.float64)
    if fmt == "Q4_0":
        return per(f["d"], 32) * (q - 8)
    if fmt == "Q4_1":
        return per(f["d"], 32) * q + per(f["m"], 32)
    if fmt == "Q8_0":
        return per(f["d"], 32) * q
    sub = 32 if fmt in ("Q4_K", "Q5_K") else 16
    sc = np.repeat(f["sc"].astype(np.float64).reshape(q.shape[0], -1), sub, 1)
    if fmt in ("Q4_K", "Q5_K", "Q2_K"):
        mn = np.repeat(f["mn"].astype(np.float64).reshape(q.shape[0], -1), sub, 1)
        return per(f["d"], 256) * sc * q - per(f["dmin"], 256) * mn
    return per(f["d"], 256) * sc * (q - (32 if fmt == "Q6_K" else 4))


# (format, the tile size of the image that holds it, super-blocks per tile, the file, a tensor of that format)
GEMV = [
    ("Q4_0", 9216, 2, ("gemma", "Q4_0"), "blk.0.ffn_gate.weight"),
    ("Q8_0", 9216, 2, ("gemma", "Q4_0"), "token_embd.weight"),
    ("Q4_K", 9280, 4, ("llama", "Q4_K_M", "1B"), "blk.0.ffn_gate.weight"),
    ("Q5_K", 9280, 4, ("llama", "Q5_K_M", "1B"), "blk.0.ffn_gate.weight"),
    ("Q5_K", 9856, 4, ("llama", "Q3_K_M", "3B"), "blk.0.attn_v.weight"),
    ("Q6_K", 9280, 4, ("llama", "Q4_K_M", "1B"), "blk.0.attn_v.weight"),
    ("BFP16", 9216, 4, ("llama", "BF16", "1B"), "blk.0.ffn_gate.weight"),
    ("Q4_0", 9280, 4, ("llama", "Q4_0", "1B"), "blk.0.ffn_gate.weight"),
    ("Q4_1", 9280, 4, ("llama", "Q4_0", "1B"), "blk.0.ffn_down.weight"),  # (14 rows)
    ("Q4_1", 10240, 4, ("llama", "Q4_1", "1B"), "blk.0.ffn_gate.weight"),
    ("Q2_K", 9408, 4, ("llama", "Q2_K", "3B"), "blk.0.ffn_gate.weight"),  # (26 rows)
    ("Q3_K", 9408, 4, ("llama", "Q2_K", "3B"), "blk.0.attn_v.weight"),  # (21 rows)
    ("Q3_K", 9856, 4, ("llama", "Q3_K_M", "3B"), "blk.0.ffn_gate.weight"),  # (22 rows)
]


def _path(key: tuple[str, ...]) -> str:
    return gemma_gguf(key[1]) if key[0] == "gemma" else llama_gguf(*key[1:])


@pytest.mark.parametrize("S", [1, 4])
@pytest.mark.parametrize(("fmt", "S_BYTES", "NSB", "key", "name"), GEMV, ids=[f"{c[0]}-{c[1]}" for c in GEMV])
def test_gemv(fmt: str, S_BYTES: int, NSB: int, key: tuple[str, ...], name: str, S: int) -> None:
    _check_gemv(fmt, S_BYTES, NSB, key, name, S)


def _check_gemv(fmt: str, S_BYTES: int, NSB: int, key: tuple[str, ...], name: str, S: int) -> None:
    path = _path(key)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    t = GGUFFile(path).tensor(name)
    assert tile_format(t.qtype) == fmt
    KC = 256 * NSB
    n_tiles = 3
    op = mmv.MmvOperator(fmt=fmt, n_tiles=n_tiles, tile_bytes=S_BYTES, nsb=NSB)
    R = op.rows
    t.rows, t.raw = n_tiles * R, t.raw[:n_tiles * R]
    f = t.fields()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((S, KC)) * np.exp(rng.standard_normal((S, KC)))).astype(np.float32)
    x = torch.from_numpy(x).bfloat16().float().numpy()
    tiles, ref, exact = [], [], []
    for i in range(n_tiles):
        fc = slice_fields(f, np.arange(R * i, R * (i + 1)), k0=KC, cols=t.cols, kc=KC)  # (the second K chunk)
        tiles.append(pack_tile(fmt, fc, S_BYTES))
        ref.append(mmv.ref_tile(fmt, fc, x)[:S])
        exact.append(x.astype(np.float64) @ _weights(fmt, fc).T)
    X = np.concatenate([mmv.x_panels(x).reshape(-1), mmv.xs_panels(x, NSB).reshape(-1)])
    got = op.compile().run(torch.from_numpy(np.concatenate(tiles)), torch.from_numpy(X)).numpy()
    gs = mmv.tile_rows(got.reshape(n_tiles, op.y_floats), R)[:, :S]
    rs, ex = np.stack(ref), np.stack(exact)
    assert np.abs(gs - rs).max() <= 1e-5 * np.abs(rs).max(), np.argwhere(np.abs(gs - rs) > 1e-5 * np.abs(rs).max())[:8]
    # (vs the exact product: the sub-block sums rounded to bf16 once each, 1.5-2.0e-3 rel-rms for every integer format)
    assert np.sqrt(((gs - ex) ** 2).mean() / (ex ** 2).mean()) < 2.5e-3
