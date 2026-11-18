"""The Spark-X weight image (models/spark/image.py) on the host (no NPU): the layout of each file (every row of every
matrix in exactly one column's block, the KV heads' parts and the gate rows per column, the inputs of part of a K chunk
padded, the regions in step order, the tile formats, column- or region-major) and the packed image's tiles and
norm-weight objects at their places (an F32 gate through its BF16 copy)."""

import time

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode.image import pack_rows
from artlite.gguf_model import GGUFFile
from artlite.operators.design import COLS
from models.spark.image import KC, NORMS, SparkImage

from ..decode.image_checks import check_tails
from . import ref

FILES = {"4B": ("4B", "Q4_K_M", "XHToken"), "1.7B": ("1.7B", "Q4_K_M", "XHToken"), "4B-Q8_0": ("4B", "Q8_0", "XHToken"),
         "1.7B-Q8_0": ("1.7B", "Q8_0", "XHToken"), "bartowski-4B": ("4B", "Q4_K_M", "bartowski"),
         "bartowski-4B-Q4_0": ("4B", "Q4_0", "bartowski")}
FORMATS = {"4B": {"BFP16", "Q4_K", "Q6_K", "Q8_0"}, "1.7B": {"BFP16", "Q4_K", "Q6_K", "Q8_0"}, "4B-Q8_0": {"Q8_0"},
           "1.7B-Q8_0": {"Q8_0"}, "bartowski-4B": {"BFP16", "Q4_K", "Q5_K", "Q6_K"},
           "bartowski-4B-Q4_0": {"Q4_0", "Q4_1", "Q6_K"}}


@pytest.mark.parametrize("name", FILES)
def test_layout(name: str) -> None:
    g = ref.gguf(*FILES[name])
    img = SparkImage(g, tails=False)
    cfg = img.cfg
    four = cfg.num_hidden_layers == 36
    assert img.kc == KC and img.tile_bytes == 9280 and img.formats() == FORMATS[name] and img.column_major == four
    want = []
    for _ in range(cfg.num_hidden_layers):
        want += ["attn_in_norm", "attn_in", "attn_out", "gate_up_norm", "gate_up", "down"] if four else \
            ["attn_in", "attn_out", "gate_up", "down"]
    assert [r.name for r in img.regions] == want + (["lm_norm", "lm"] if four else ["lm"])
    for r in img.regions:
        if r.data:
            continue
        assert {r.tile(rt, kc) for rt in range(r.n_rt) for kc in range(r.n_kc)} == set(range(r.n_rt * r.n_kc))
        for b in r.blocks:
            rows = b.rows[b.rows >= 0]
            total = sum(g.tensor_info[t][0][1] for t in b.tensors)
            if r.name == "attn_in" and b.name != "g":  # (q, k, v: their thirds of the fused attn_qkv)
                hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
                lo = {"q": 0, "k": nh * hd, "v": (nh + nkv) * hd}[b.name]
                n = (nh if b.name == "q" else nkv) * hd
                assert sorted(rows.tolist()) == list(range(lo, lo + n)), (r.layer, b.name)
            else:
                assert sorted(rows.tolist()) == list(range(total)), (r.layer, r.name, b.name)
    a = img.region(3, "attn_in")
    if four:  # (column c: heads 2 c, 2 c + 1; half c % 2 of KV head c // 2)
        assert a.block("q").rows[5, :512].tolist() == list(range(10 * 256, 12 * 256))
        assert a.block("k").rows[5, :128].tolist() == list(range(4096 + 2 * 256 + 128, 4096 + 3 * 256))
        assert a.block("g").rows[7, :2].tolist() == [14, 15] and (a.block("g").rows[7, 2:] < 0).all()
        assert (a.n_kc, img.region(3, "gate_up").n_kc, img.region(-1, "lm").n_kc) == (3, 3, 3)  # (2560 padded)
        assert (img.region(3, "down").n_kc, img.region(3, "down").kc_group) == (10, 5)
    else:  # (column c: head c; quarter c % 4 of KV head c // 4)
        assert a.block("q").rows[5, :256].tolist() == list(range(5 * 256, 6 * 256))
        assert a.block("v").rows[6, :64].tolist() == list(range(2048 + 512 + 256 + 128, 2048 + 512 + 256 + 192))
        assert a.block("g").rows[6, :1].tolist() == [6]
        assert (img.region(3, "down").n_kc, img.region(3, "down").kc_group) == (7, 1)  # (6656 padded: region-major)
    lm = img.region(-1, "lm").block("lm")
    assert lm.tensors == ("token_embd.weight" if cfg.tied else "output.weight",) and lm.rows[7][lm.rows[7] >= 0][-1] == 131071
    assert img.region(0, "attn_in").block("g").fmt == ("Q4_0" if "Q4_0" in name else "Q8_0" if "Q8_0" in name
                                                       else "BFP16")


@pytest.mark.parametrize("name", ["4B", "bartowski-4B"])
def test_pack_image(name: str) -> None:
    """SparkImage.pack of a sliding-window and a full layer and a slice of the LM head: sampled tiles of every block at
    their places as pack_rows packs their rows (the padded K: zero past 2560 columns; bartowski's F32 gate: as its BF16
    copy; its gate / up of two formats: a split block's parts), the norm-weight objects in every column; the whole
    image's pack time."""
    g = ref.gguf(*FILES[name])
    img = SparkImage(g, layers=[2, 3], vocab=COLS * 64, tails=False)
    T = img.tile_bytes
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out)
    rng = np.random.default_rng(0)
    for r in img.regions:
        cols = [out[r.offset + c * r.col_stride:][:r.col_bytes] for c in range(COLS)]
        if r.data:
            phase = r.name.removesuffix("_norm")
            w = g.tensor("output_norm.weight" if r.layer < 0 else f"blk.{r.layer}.{NORMS[phase]}.weight").dequantize()
            for col in cols:
                v = col.view(np.uint16)
                assert np.array_equal(v[:img.cfg.hidden_size], w.reshape(-1).astype(bfloat16).view(np.uint16))
            continue
        for b in r.blocks:  # (a split gate / up block: a tile of each of its parts, ``Block.parts``)
            for fmt, R, rt0, names, rows in b.parts(img.tensor(b.tensors[0]).rows)[:2]:
                c, t = int(rng.integers(COLS)), int(rng.integers(rows.shape[1] // R))
                ref_tiles = pack_rows([img.tensor(n) for n in names], rows[c, t * R:(t + 1) * R], fmt, R, T, r.n_kc)
                col = cols[c].reshape(-1, T)
                for kc in range(r.n_kc):
                    assert np.array_equal(col[r.tile(r.rt0[b.name] + rt0 + t, kc)], ref_tiles[0, kc]), \
                        (r.layer, r.name, b.name, fmt, kc)
    if name == "bartowski-4B":  # (the F32 gate: its BF16 copy's values)
        gate = img.tensor("blk.3.attn_gate.weight")
        assert gate.qtype == "BF16" and np.array_equal(gate.dequantize(), g.tensor("blk.3.attn_gate.weight").dequantize()
                                                       .astype(bfloat16).astype(np.float32))
    full = SparkImage(g, tails=False)
    buf = np.zeros(full.nbytes, np.uint8)
    t0 = time.perf_counter()
    full.pack(buf)
    print(f"{name}: {full.nbytes / 1e6:.0f} MB packed in {time.perf_counter() - t0:.2f} s")


@pytest.mark.parametrize("name", ["4B", "bartowski-4B", "bartowski-4B-Q4_0", "4B-Q8_0"])
def test_pack_tails(name: str) -> None:
    """SparkImage(tails=True): the hidden's regions (attn_in, gate_up, lm) are half regions, the LM head's block and
    gate / up's of the formats the prefill reads in pairs (``prefill.gemm.pair_tails``) in pairs of row tiles (5 tiles
    each: 2 p's chunks 0, 1, the tail tile, 2 p + 1's chunks 0, 1), the other blocks (attn_in's; bartowski's gate / up
    of two formats: a split block's parts) as in the padded image (``image_checks.check_tails``)."""
    g = ref.gguf(*FILES[name])
    img = SparkImage(g, layers=[1, 3], vocab=COLS * 64 * 3, tails=True)
    pad = SparkImage(g, layers=[1, 3], vocab=COLS * 64 * 3, tails=False)
    check_tails(img, pad, {"attn_in", "gate_up", "lm"})


def test_padded_k_is_zero() -> None:
    """The 4B's hidden 2560 in 3 chunks: a tile's last 512 columns of a 2560-column matrix are zero super-blocks (the
    row tile's values match the file's dequantized rows over the first 2560 columns)."""
    g: GGUFFile = ref.gguf("4B")
    img = SparkImage(g, layers=[0], vocab=COLS * 64, tails=False)
    r = img.region(0, "gate_up")
    b = r.blocks[0]
    tiles = pack_rows([g.tensor(n) for n in b.tensors], b.rows[0, :b.R], b.fmt, b.R, img.tile_bytes, r.n_kc)
    empty = pack_rows([g.tensor(n) for n in b.tensors], np.full(b.R, -1), b.fmt, b.R, img.tile_bytes, r.n_kc)
    assert r.n_kc == 3 and not np.array_equal(tiles[0, 2], empty[0, 2])  # (chunk 2: columns 2048 .. 2559 and zeros)
