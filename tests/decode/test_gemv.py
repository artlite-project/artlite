"""The decode overlay's multi-slot GEMV (decode/gemv.py and the slots' ``mmv/<fmt>`` kernels) on every tile
format it reads: the C++ tile packer against the slots' ``pack_tile``, and the two cores' rows of real tiles
(``mmv.MmvOperator``) against the kernels' arithmetic (``mmv.ref_tile``) and the exact product (the tiles' bf16
scales)."""


import dataclasses

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite import _native
from artlite.decode import gemv
from artlite.decode.image import f32_planes, pack_rows
from artlite.formats import DERIVED, FORMATS, pack_tile, q4, q6h, q8h, slice_fields, tile_format
from artlite.gguf_model import IQ4_VALUES, MXFP4_VALUES, QTensor
from artlite.hf_model import open_weights
from artlite.model_config import checkpoint, flint_gguf, gemma_gguf, inkling_gguf, llama_gguf, occamy_gguf, swift_gguf
from artlite.operators import bfp16
from artlite.paro import ParoTensor

from ..model_files import require
from ..numerics import bf16_round
from ..tensors import random_tensor
from . import mmv

# (GGUF type, tile bytes, super-blocks per tile) the packer is checked at: Gemma's 512-column tiles, Llama's and Qwen's
# 1024 (the tile sizes of the images that hold them)
PACK = [("Q4_0", 9216, 2), ("Q8_0", 9216, 2), ("Q4_K", 9280, 4), ("Q5_K", 9280, 4), ("Q6_K", 9280, 4), ("Q4_0", 9280, 4),
        ("Q4_1", 9280, 4), ("Q4_1", 10240, 4), ("Q2_K", 9408, 4), ("Q3_K", 9408, 4), ("Q3_K", 9856, 4), ("BF16", 9216, 4),
        ("F16", 9216, 4), ("Q3_K", 9280, 4), ("IQ3_XXS", 9280, 4), ("Q8_0", 9280, 4), ("IQ3_S", 9280, 4),
        ("IQ4_NL", 9280, 4), ("IQ4_XS", 9280, 4), ("IQ4_XS", 4352, 4), ("IQ2_XXS", 9280, 4),
        ("IQ2_XS", 9280, 4), ("IQ2_S", 9280, 4), ("IQ2_S", 9216, 4), ("IQ1_S", 9280, 4), ("IQ1_M", 9280, 4),
        ("IQ1_S", 9216, 4), ("Q6_KH", 9280, 4), ("Q4_1_64", 9216, 4), ("NVFP4", 9216, 4), ("NVFP4", 9280, 4),
        ("Q8_0H", 9216, 4)]


@pytest.mark.parametrize(("qtype", "S_BYTES", "NSB"), PACK, ids=[f"{f}-{s}-nsb{n}" for f, s, n in PACK])
def test_pack_native(qtype: str, S_BYTES: int, NSB: int) -> None:
    """The C++ packer (pack_rows) is bit-identical to the slot's pack_tile: two stacked tensors, rows in a random order
    with padding rows, 3 K chunks (Q6_KH: tiles of Q6_K tensors, ``formats.q6h``)."""
    fmt, qtype = (qtype, DERIVED[qtype]) if qtype in DERIVED else (tile_format(qtype), qtype)
    KC = 256 * NSB
    R = FORMATS[fmt].rows(S_BYTES, NSB, gemv.row_split(fmt, NSB) if gemv.supported(fmt, NSB) else 0)
    gen = fmt if fmt == "Q4_1_64" else qtype  # (Q4_1_64: Q4_1 blocks in pairs)
    a, b = random_tensor(gen, 40, 3 * KC, 1), random_tensor(gen, 9, 3 * KC, 2)
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


# (GGUF type, tile bytes, rows per tile row G): an MoE's downs (K 256 and 512 in tiles of 1024 columns)
GROUPS = [("Q4_K", 9280, 4), ("Q6_K", 9280, 4), ("Q4_K", 9280, 2), ("Q6_K", 9280, 2), ("Q3_K", 9280, 4),
          ("IQ3_XXS", 9280, 4), ("Q5_K", 9280, 2), ("Q4_1_64", 9216, 4)]


@pytest.mark.parametrize(("qtype", "S_BYTES", "G"), GROUPS, ids=[f"{f}-G{g}" for f, _, g in GROUPS])
def test_pack_native_groups(qtype: str, S_BYTES: int, G: int) -> None:
    """Row-group tiles (pack_rows G): a tile row of G rows of 1024 / G columns, one after the other along K, as the
    slot's pack_tile packs those rows' fields side by side; two stacked tensors, padding rows, a partial last tile."""
    fmt, K = (qtype if qtype in DERIVED else tile_format(qtype)), 1024 // G
    R = FORMATS[fmt].rows(S_BYTES, 4)
    a, b = random_tensor(qtype, 30, K, 4), random_tensor(qtype, 11, K, 5)
    qtype = DERIVED.get(qtype, qtype)
    rows = np.concatenate([np.random.default_rng(6).permutation(41), [-1] * 3])
    rows = np.concatenate([rows, np.full((-rows.size) % (R * G), -1)])
    tiles = pack_rows([a, b], rows, fmt, R, S_BYTES, 1, G=G)
    zero = QTensor(name="z", qtype=qtype, rows=1, cols=K, raw=np.zeros_like(a.raw[:1]))
    fa, fb, fz = a.fields(), b.fields(), zero.fields()
    for t in range(tiles.shape[0]):
        per = []
        for v in range(t * R, (t + 1) * R):  # (a tile row: its G rows' fields along K)
            parts = []
            for r in rows[v * G:(v + 1) * G]:
                src, idx = (fz, 0) if r < 0 else ((fa, r) if r < 30 else (fb, r - 30))
                parts.append(slice_fields(src, np.array([idx]), k0=0, cols=K, kc=K))
            per.append({key: np.concatenate([p[key] for p in parts], 1) for key in parts[0]})
        f = {key: np.concatenate([p[key] for p in per]) for key in per[0]}
        assert np.array_equal(tiles[t, 0], pack_tile(fmt, f, S_BYTES)), t


def test_pack_q4_1_64_checks() -> None:
    """Q4_1_64 tiles of Q4_1 blocks not in pairs: the packer refuses them (an unequal m in one pair, a d bf16 does not
    hold), and ``q4.pairs64`` tells them from the paired ones."""
    t = random_tensor("Q4_1_64", 16, 1024, 8)
    rows = np.arange(16)
    assert q4.pairs64(t.raw)
    pack_rows([t], rows, "Q4_1_64", 16, 9216, 1)
    for off, val in ((20 * 7 + 2, 1), (20 * 4, 1)):  # (block 7's m: its pair's differs; block 4's d: 11 bits)
        raw = t.raw.copy()
        raw[3, off] ^= val
        bad = QTensor(name="t", qtype="Q4_1", rows=16, cols=1024, raw=raw)
        assert not q4.pairs64(bad.raw)
        with pytest.raises(_native.Error, match="Q4_1_64"):
            pack_rows([bad], rows, "Q4_1_64", 16, 9216, 1)


def test_pack_native_planes() -> None:
    """BFP16 tiles of F32 rows (an MoE router's): index 2 i + p is plane p of row i, as the slot's pack_tile packs w
    (hi = bfp16(w): its blocks, not hi's own, which a mantissa of -128 coarsens) and w - hi (lo); hi + lo within 3e-5
    of w (relative RMS per row); values over many octaves, an all-zero block, padding rows."""
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((9, 2048)) * np.exp2(rng.integers(-20, 4, (9, 1)))).astype(np.float32)
    w[3, 8:16] = 0
    t = QTensor(name="r", qtype="F32", rows=9, cols=2048, raw=w)
    rows = np.concatenate([rng.permutation(18), [-1] * 6])
    tiles = pack_rows([t], rows, "BFP16", 8, 9280, 2)
    hi = bfp16.values(w)
    src = np.zeros((19, 2048), np.float32)  # (what pack_tile rounds for index 2 i + p; 18: a zero row)
    src[0:18:2], src[1:18:2] = w, w - hi
    for g in range(3):
        for k in range(2):
            f = {"w": src[np.where(rows[8 * g:8 * g + 8] < 0, 18, rows[8 * g:8 * g + 8]), 1024 * k:1024 * (k + 1)]}
            assert np.array_equal(tiles[g, k], pack_tile("BFP16", f, 9280)), (g, k)
    err = np.sqrt(((hi + bfp16.values(w - hi) - w) ** 2).mean(1) / (w ** 2).mean(1))
    assert err.max() < 3e-5, err


def test_pack_native_one_plane() -> None:
    """An F32 tensor of bf16 values as one plane (``f32_planes(rows, 1)``: plane 0 of each row, padding rows): the tiles
    of the same values as a BF16 tensor."""
    rng = np.random.default_rng(8)
    w = (rng.standard_normal((9, 2048)) * np.exp2(rng.integers(-12, 4, (9, 1)))).astype(bfloat16)
    t = QTensor(name="r", qtype="F32", rows=9, cols=2048, raw=w.astype(np.float32))
    tb = QTensor(name="r", qtype="BF16", rows=9, cols=2048, raw=w.view(np.uint16))
    rows = np.concatenate([np.random.default_rng(9).permutation(9), [-1] * 7])
    assert np.array_equal(pack_rows([t], f32_planes(rows[None], 1)[0], "BFP16", 8, 9216, 2),
                          pack_rows([tb], rows, "BFP16", 8, 9216, 2))


def _weights(fmt: str, f: dict[str, np.ndarray]) -> np.ndarray:
    """(R, K) the tile's weights as the kernels take them: GGUF's formulas with the scales in bf16 (the tiles')."""
    q = f["q"].astype(np.float64) if "q" in f else None

    def per(v: np.ndarray, n: int) -> np.ndarray:  # a value per n columns
        return np.repeat(bf16_round(v).astype(np.float64).reshape(v.shape[0], -1), n, 1)

    if fmt == "BFP16":
        w = f["w"]
        w = w.astype(np.float32) if w.dtype == np.float16 else torch.from_numpy(w.view(np.int16)).view(torch.bfloat16).float().numpy()
        return bfp16.values(w).astype(np.float64)
    if fmt == "Q4_0":
        return per(f["d"], 32) * (q - 8)
    if fmt in ("Q4_1", "Q4_1_64"):
        return per(f["d"], 32) * q + per(f["m"], 32)
    if fmt == "Q8_0":
        return per(f["d"], 32) * q
    if fmt == "IQ2_XXS":
        return per(f["d"], 256) * np.repeat((2 * f["ls"].astype(np.float64) + 1) / 8, 32, 1) * q
    if fmt in ("IQ2_XS", "IQ2_S"):
        return per(f["d"], 256) * np.repeat((2 * f["sc"].astype(np.float64) + 1) / 8, 16, 1) * q
    if fmt in ("IQ1_S", "IQ1_M"):
        return per(f["d"], 256) * np.repeat((2 * f["sc"].astype(np.float64) + 1) / 8, 32 if fmt == "IQ1_S" else 16, 1) * q
    if fmt == "IQ3_XXS":
        return per(f["d"], 256) * np.repeat((2 * f["ls"].astype(np.float64) + 1) / 4, 32, 1) * q
    if fmt == "IQ3_S":
        return per(f["d"], 256) * np.repeat(1 + 2 * f["sc"].astype(np.float64), 32, 1) * q
    if fmt in ("IQ4_NL", "IQ4_XS"):
        v = IQ4_VALUES[f["q"]].astype(np.float64)
        return per(f["d"], 32) * v if fmt == "IQ4_NL" else \
            per(f["d"], 256) * np.repeat(f["ls"].astype(np.float64) - 32, 32, 1) * v
    if fmt == "NVFP4":  # (d: exact in bf16)
        return np.repeat(f["d"].astype(np.float64), 16, 1) * MXFP4_VALUES[f["q"]]
    if fmt in ("P4_128", "P5_128"):
        return per(f["s"], 128) * (q - np.repeat(f["z"].astype(np.float64), 128, 1))
    if fmt == "Q8_0H":  # (Q8_0's fields: its high plane's weights, ``formats.q8h``)
        h = q8h.fields(f)
        return per(h["d"], 32) * (h["q"].astype(np.float64) - 7.5)
    if fmt == "Q6_KH":  # (Q6_K's fields: its high plane's weights, ``formats.q6h``)
        h = q6h.fields(f)
        return per(h["d"], 256) * np.repeat(h["sc"].astype(np.float64).reshape(q.shape[0], -1), 16, 1) * ((f["q"] >> 4) - 1.5)
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
    ("Q6_KH", 9280, 4, ("llama", "Q4_K_M", "1B"), "blk.0.attn_v.weight"),  # (27 rows: the high plane of a Q6_K tensor)
    ("BFP16", 9216, 4, ("llama", "BF16", "1B"), "blk.0.ffn_gate.weight"),
    ("Q4_0", 9280, 4, ("llama", "Q4_0", "1B"), "blk.0.ffn_gate.weight"),
    ("Q4_1", 9280, 4, ("llama", "Q4_0", "1B"), "blk.0.ffn_down.weight"),  # (14 rows)
    ("Q4_1", 10240, 4, ("llama", "Q4_1", "1B"), "blk.0.ffn_gate.weight"),
    ("Q2_K", 9408, 4, ("llama", "Q2_K", "3B"), "blk.0.ffn_gate.weight"),  # (26 rows)
    ("Q3_K", 9408, 4, ("llama", "Q2_K", "3B"), "blk.0.attn_v.weight"),  # (21 rows)
    ("Q3_K", 9856, 4, ("llama", "Q3_K_M", "3B"), "blk.0.ffn_gate.weight"),  # (22 rows)
    ("Q8_0", 9280, 4, ("occamy",), "blk.0.attn_gate.weight"),  # (8 rows: the super-block split)
    ("IQ3_XXS", 9280, 4, ("occamy",), "blk.10.ffn_gate_exps.weight"),  # (23 rows)
    ("Q3_K", 9280, 4, ("occamy",), "blk.0.attn_qkv.weight"),  # (21 rows)
    ("IQ4_XS", 9280, 4, ("swift",), "blk.0.ffn_down.weight"),  # (16 rows)
    ("IQ4_NL", 9280, 4, ("random",), "IQ4_NL"),  # (16 rows; no file of ours has IQ4_NL tensors)
    ("IQ3_S", 9280, 4, ("swift",), "blk.0.ffn_gate.weight"),  # (20 rows)
    ("IQ2_XXS", 9280, 4, ("random",), "IQ2_XXS"),  # (24 rows)
    ("IQ2_XS", 9280, 4, ("random",), "IQ2_XS"),
    ("IQ2_S", 9280, 4, ("random",), "IQ2_S"),
    ("IQ1_S", 9280, 4, ("random",), "IQ1_S"),  # (32 rows)
    ("IQ1_M", 9280, 4, ("random",), "IQ1_M"),
    ("Q4_1_64", 9216, 4, ("flint",), "blk.0.attn_q.weight"),
    ("P4_128", 9216, 4, ("paro",), "blk.0.ffn_down.weight"),  # (Qwen3-0.6B-PARO: AWQ codes, integer zeros)
    ("P5_128", 9280, 4, ("paro5",), "blk.0.ffn_down.weight"),  # (Qwen3.8-27B-PARO-int5: 13 rows)
    ("NVFP4", 9216, 4, ("inkling",), "blk.0.ffn_gate_up_exps.weight"),  # (16 rows: expert 0's gate rows)
    ("NVFP4", 9280, 4, ("random",), "NVFP4"),  # (every scale byte: subnormals, 0x7F, the sign bit)
    ("Q8_0H", 9216, 4, ("inkling",), "output.weight"),  # (16 rows: the high plane of a Q8_0 head)
    ("Q8_0H", 9280, 4, ("random",), "Q8_0"),
]


def _path(key: tuple[str, ...]) -> str:
    if key[0] in ("paro", "paro5"):
        return checkpoint("qwen3-0.6B-paro" if key[0] == "paro" else "qwen3.8-27B-paro-int5")
    if key[0] == "flint":
        return flint_gguf()
    if key[0] == "inkling":
        return inkling_gguf(*key[1:])
    if key[0] in ("occamy", "swift"):
        return occamy_gguf() if key[0] == "occamy" else swift_gguf()
    return gemma_gguf(key[1]) if key[0] == "gemma" else llama_gguf(*key[1:])


@pytest.mark.parametrize(("fmt", "S_BYTES", "NSB", "key", "name"), GEMV, ids=[f"{c[0]}-{c[1]}" for c in GEMV])
def test_gemv(fmt: str, S_BYTES: int, NSB: int, key: tuple[str, ...], name: str) -> None:
    """One design, run with 1 and 4 slots' activations."""
    _check_gemv(fmt, S_BYTES, NSB, key, name, (1, 4))


# (the 8-slot images (qwen35, spark, dflash) hold 9280-byte tiles only)
GEMV8_SKIP = {("Q5_K", 9856), ("Q3_K", 9856), ("Q4_1", 10240), ("Q3_K", 9408)}
GEMV8 = [c for c in GEMV if c[0] in mmv.GEMV8_FORMATS and not gemv.row_split(c[0], c[2]) and c[:2] not in GEMV8_SKIP]


@pytest.mark.parametrize(("fmt", "S_BYTES", "NSB", "key", "name"), GEMV8, ids=[f"{c[0]}-{c[1]}" for c in GEMV8])
def test_gemv8(fmt: str, S_BYTES: int, NSB: int, key: tuple[str, ...], name: str) -> None:
    """Designs of 8 slots (x as single bfp16 rows, the sums as hi and lo panels): against their arithmetic and the
    exact product (with the activations' bfp16 rounding: 0.9-1.2e-2 rel-rms on these heavy-tailed x, 2e-3 as hi + lo)."""
    _check_gemv(fmt, S_BYTES, NSB, key, name, (1, 8), slots=8)


def _check_gemv(fmt: str, S_BYTES: int, NSB: int, key: tuple[str, ...], name: str, slot_counts: tuple[int, ...],
                slots: int = 4) -> None:
    if key[0] == "random":
        t = random_tensor(name, 256, 2048, 4)
    else:
        path = _path(key)
        require(path)
        t = open_weights(path).tensor(name)
    assert t.qtype == DERIVED[fmt] if fmt in DERIVED else tile_format(t.qtype) == fmt
    KC = 256 * NSB
    n_tiles = 3
    op = mmv.MmvOperator(fmt=fmt, n_tiles=n_tiles, tile_bytes=S_BYTES, nsb=NSB, slots=slots)
    R = op.rows
    if isinstance(t, ParoTensor):
        t = dataclasses.replace(t, rows=n_tiles * R, rows_of=np.arange(n_tiles * R))
    else:
        t.rows, t.raw = n_tiles * R, t.raw[:n_tiles * R]
    f = t.fields()
    rng = np.random.default_rng(0)
    # (the second K chunk)
    fcs = [slice_fields(f, np.arange(R * i, R * (i + 1)), k0=KC, cols=t.cols, kc=KC) for i in range(n_tiles)]
    tiles = [pack_tile(fmt, fc, S_BYTES) for fc in fcs]
    compiled = op.compile()
    for S in slot_counts:
        x = (rng.standard_normal((S, KC)) * np.exp(rng.standard_normal((S, KC)))).astype(np.float32)
        x = torch.from_numpy(x).bfloat16().float().numpy()
        ref = [mmv.ref_tile(fmt, fc, x, slots)[:S] for fc in fcs]
        exact = [x.astype(np.float64) @ _weights(fmt, fc).T for fc in fcs]
        X = np.concatenate([mmv.x_panels(x, slots).reshape(-1), mmv.xs_panels(x, NSB, slots).reshape(-1)])
        got = compiled.run(np.concatenate(tiles), X)
        gs = mmv.tile_rows(got.reshape(n_tiles, op.y_floats), R, slots)[:, :S]
        rs, ex = np.stack(ref), np.stack(exact)
        bad = np.abs(gs - rs) > 1e-5 * np.abs(rs).max()
        assert not bad.any(), (S, np.argwhere(bad)[:8])
        # (vs the exact product: the sub-block sums rounded to bf16 once each, 1.5-2.0e-3 rel-rms for every integer
        # format; x as single bfp16 (8 slots): its rounding too)
        err = np.sqrt(((gs - ex) ** 2).mean() / (ex ** 2).mean())
        assert err < (2.5e-3 if slots == 4 else 1.5e-2), (S, err)


# (format, tile bytes, row groups G, the file, a tensor of that format): the row-group tiles of a qtails design
# (``decode.image.Region.qt``: a group's tail tile, B_G4; an expert unit's down of K 128, B_G8)
GROUPS = [
    ("BFP16", 9216, 4, ("random",), "BF16"),
    ("Q8_0", 9216, 4, ("inkling",), "blk.0.attn_q.weight"),
    ("NVFP4", 9216, 4, ("inkling",), "blk.0.ffn_gate_up_exps.weight"),
    ("Q5_K", 9216, 4, ("inkling", "UD-Q4_K_M"), "blk.20.ffn_gate_up_exps.weight"),  # (13 rows)
    ("Q8_0H", 9216, 4, ("inkling",), "output.weight"),
    ("NVFP4", 9216, 8, ("inkling",), "blk.0.ffn_down_exps.weight"),
    ("Q8_0", 9216, 8, ("inkling",), "blk.20.ffn_down_exps.weight"),
]


@pytest.mark.parametrize(("fmt", "S_BYTES", "G", "key", "name"), GROUPS, ids=[f"{c[0]}-G{c[2]}" for c in GROUPS])
def test_gemv_groups(fmt: str, S_BYTES: int, G: int, key: tuple[str, ...], name: str) -> None:
    """Tiles of G row groups (a tile row: G rows of K 1024 / G, ``decode.image.Block.G``) on a qtails design's kernels:
    each group's rows bit-exact against the plain kernel on a tile of that group's rows alone (zeros outside its K
    range: exact zero terms), and within 1e-5 of the host model, with 1 and 4 slots (Q5_K: its compact kernels, as
    Inkling's design takes them)."""
    if key[0] == "random":
        t = random_tensor(name, 256, 1024, 5)
    else:
        path = _path(key)
        require(path)
        t = open_weights(path).tensor(name)
    qtype = "Q8_0" if fmt == "Q8_0H" else ("BF16" if fmt == "BFP16" else fmt)
    assert t.qtype == qtype, t.qtype
    n_tiles, K = 2, 1024 // G
    op, op1 = mmv.MmvOperator(fmt=fmt, n_tiles=n_tiles, tile_bytes=S_BYTES, G=G, compact=fmt == "Q5_K"), \
        mmv.MmvOperator(fmt=fmt, n_tiles=n_tiles * G, tile_bytes=S_BYTES)
    R = op.rows
    raw = np.ascontiguousarray(t.raw[:n_tiles * R * G])
    kb = raw.shape[1] * K // t.cols  # (a row's bytes of K columns: whole GGUF blocks)
    tg = QTensor(name="g", qtype=t.qtype, rows=len(raw), cols=K, raw=np.ascontiguousarray(raw[:, :kb]))
    tiles = pack_rows([tg], np.arange(len(raw)), fmt, R, S_BYTES, 1, G=G).reshape(-1)
    alone, fields = [], []  # (per tile and group: its rows alone in a tile of 1024 columns)
    for i in range(n_tiles):
        for j in range(G):
            r1 = np.zeros((R, raw.shape[1] * 1024 // t.cols), raw.dtype)
            r1[:, j * kb:(j + 1) * kb] = raw[i * R * G + j:(i + 1) * R * G:G, :kb]
            t1 = QTensor(name="a", qtype=t.qtype, rows=R, cols=1024, raw=r1)
            alone.append(pack_rows([t1], np.arange(R), fmt, R, S_BYTES, 1).reshape(-1))
            fields.append(t1.fields())
    a, a1 = op.compile(), op1.compile()
    rng = np.random.default_rng(1)
    for S in (1, 4):
        x = (rng.standard_normal((S, 1024)) * np.exp(rng.standard_normal((S, 1024)))).astype(np.float32)
        x = torch.from_numpy(x).bfloat16().float().numpy()
        X = np.concatenate([mmv.x_panels(x).reshape(-1), mmv.xs_panels(x, 4).reshape(-1)])
        got = a.run(tiles, X).reshape(n_tiles, G, -1)  # ([core 0's groups | core 1's]: G blocks of a 16-row tile)
        ref = a1.run(np.concatenate(alone), X).reshape(n_tiles, G, -1)
        gs, rs = mmv.tile_rows(got, R)[..., :S, :], mmv.tile_rows(ref, R)[..., :S, :]
        assert np.array_equal(gs, rs), (S, np.argwhere(gs != rs)[:8])
        host = np.stack([mmv.ref_tile(fmt, f, x)[:S] for f in fields]).reshape(gs.shape)
        assert not (np.abs(gs - host) > 1e-5 * np.abs(host).max()).any(), S
