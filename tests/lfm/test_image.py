"""The LFM3 config and weight image (``model_config.LfmConfig``, models/lfm/image.py), on the host (no NPU): the config
from the GGUF metadata, the layout of each file (every row of every matrix in exactly one column's block, the conv
channels and the KV heads' halves per column, down's K padded to whole chunks, the regions in step order, the tile
size per file) and the packed image's tiles and norm-weight objects at their places."""

import os
import time

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode.design import POST_K
from artlite.decode.image import COL_SLOT, pack_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import LFM3_QUANTS, LfmConfig, lfm3_gguf
from artlite.operators.design import COLS
from models.lfm.image import CH, KC, LfmImage

TILES = {"QAD-Q4_0": 9280, "Q4_K_M": 9280, "Q8_0": 9216, "BF16": 9216}


def _gguf(quant: str) -> GGUFFile:
    path = lfm3_gguf(quant)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    return GGUFFile(path)


def test_config() -> None:
    cfg = LfmConfig.from_gguf(_gguf("QAD-Q4_0"))
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.head_dim) == (24, 2048, 7680, 16, 4, 128)
    assert cfg.layers("full") == [2, 6, 10, 14, 18, 22] and cfg.layers("swa") == [4, 8, 12, 16, 20]
    assert cfg.layers("conv") == [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    assert (cfg.sliding_window, cfg.conv_width, cfg.rope_base, cfg.rope_base_swa) == (256, 7, 1e6, 1e4)
    assert cfg.rot("full") == 32 and cfg.rot("swa") == 64 and cfg.vocab_size == 128000
    assert cfg.eos_token_ids[0] == 124900 and cfg.sampling == pytest.approx({"top_k": 50, "temp": 0.1, "penalty_repeat": 1.1})
    inv = cfg.rope_inv_freq("full")
    assert inv.shape == (32,) and inv[31] == 1e6 ** (-62 / 128) and cfg.rope_inv_freq("swa")[63] == 1e4 ** (-126 / 128)


@pytest.mark.parametrize("quant", LFM3_QUANTS)
def test_layout(quant: str) -> None:
    g = _gguf(quant)
    img = LfmImage(g)
    cfg = img.cfg
    assert img.tile_bytes == TILES[quant] and img.kc == KC and img.vocab == 128000
    names = [r.name for r in img.regions]
    want = []
    for kind in cfg.kinds:
        want += ["in_norm", "in", "out"] if kind == "conv" else ["qkv_norm", "qkv", "o"]
        want += ["gate_up_norm", "gate_up", "down"]
    assert names == want + ["lm_norm", "lm"]
    blocks = img._blocks()
    assert len(blocks) == 1
    end = img.regions[0].offset
    for r in blocks[0]:
        assert r.offset == end
        end = r.offset + r.col_bytes
        assert {r.tile(rt, kc) for rt in range(r.n_rt) for kc in range(r.n_kc)} == set(range(r.n_rt * r.n_kc))
        for b in r.blocks:
            rows = b.rows[b.rows >= 0]
            total = sum(g.tensor_info[t][0][1] for t in b.tensors)
            if r.name == "in":  # (each of B, C, x: a third of in_proj's rows)
                i = "bcx".index(b.name)
                assert sorted(rows.tolist()) == list(range(i * cfg.hidden_size, (i + 1) * cfg.hidden_size))
            else:
                assert sorted(rows.tolist()) == list(range(total)), (r.layer, r.name, b.name)
    stride = blocks[0][0].col_stride
    assert stride % (2 * COL_SLOT) == COL_SLOT
    # conv channels: column c's 256 of B, C, x; down's K padded to 8 chunks, 2 groups of 4
    inp = img.region(1, "in")
    assert inp.block("c").rows[3, :CH].tolist() == list(range(2048 + 3 * CH, 2048 + 4 * CH))
    assert img.region(0, "down").n_kc == 8 and img.region(0, "down").kc_group == 4
    # qkv: column c = 2 h + j: query heads 4 h + 2 j, + 1, half j of K_h and V_h, the two heads' gate rows
    qkv = img.region(2, "qkv")
    assert qkv.block("q").rows[5, :256].tolist() == list(range(10 * 128, 12 * 128))
    assert qkv.block("k").rows[5, :64].tolist() == list(range(2 * 128 + 64, 3 * 128))
    assert qkv.block("v").rows[2, :64].tolist() == list(range(128, 192))
    assert qkv.block("g").rows[7, :2].tolist() == [14, 15] and (qkv.block("g").rows[7, 2:] < 0).all()
    lm = img.region(-1, "lm").block("lm")
    assert lm.tensors == ("token_embd.weight",) and lm.rows[7, 15999] == 127999


@pytest.mark.parametrize("quant", ["QAD-Q4_0", "Q4_K_M"])
def test_pack_image(quant: str) -> None:
    """LfmImage.pack of a conv, a full-attention and a sliding-window layer and a slice of the LM head: sampled tiles of
    every block at their places as pack_rows packs their rows (down's padded K: zero past its 7680 columns), the
    norm-weight objects (a header's [w1 | w_post], the first layer's plain w1) in every column; the whole image's pack
    time."""
    g = _gguf(quant)
    img = LfmImage(g, layers=[3, 4, 6], vocab=COLS * 64)
    T = img.tile_bytes
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out, release=False)
    assert not out[:img.regions[0].offset].any()
    rng = np.random.default_rng(0)

    def bf(name: str) -> np.ndarray:
        return g.tensor(name).dequantize().astype(bfloat16).view(np.uint16).reshape(-1)

    for r in img.regions:
        cols = [out[r.offset + c * r.col_stride:][:r.col_bytes] for c in range(COLS)]
        if r.data:
            w1, wp = img.norms(r.layer, r.name.removesuffix("_norm"))
            for col in cols:
                v = col.view(np.uint16)
                if wp is None:
                    assert np.array_equal(v[:2048], bf(w1)) and not v[2048:].any()
                else:
                    o = v[:8 * 2 * POST_K]
                    assert np.array_equal(o[:2048], bf(w1)) and np.array_equal(o[2048:4096], bf(wp))
            continue
        for b in r.blocks:
            c, t = int(rng.integers(COLS)), int(rng.integers(b.tiles))
            ref = pack_rows([g.tensor(n) for n in b.tensors], b.rows[c, t * b.R:(t + 1) * b.R], b.fmt, b.R, T, r.n_kc)
            col = cols[c].reshape(-1, T)
            for kc in range(r.n_kc):
                assert np.array_equal(col[r.tile(r.rt0[b.name] + t, kc)], ref[0, kc]), (r.layer, r.name, b.name, kc)
    full = LfmImage(g)
    buf = np.zeros(full.nbytes, np.uint8)
    t0 = time.perf_counter()
    full.pack(buf, release=False)
    print(f"{quant}: {full.nbytes / 1e6:.0f} MB packed in {time.perf_counter() - t0:.2f} s")
