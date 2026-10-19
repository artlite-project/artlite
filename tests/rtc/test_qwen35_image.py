"""The qwen35 weight image (operators/qwen35/image.py), on the host (no NPU): the layout of a GGUF file (every row of
every matrix in exactly one column's block, the GDN heads' column partition, MiniCPM-V's padded vocabulary and down K)
and the packed image's tiles at their places (the C++ packer against the slots' ``pack_tile``: test_overlay_gemv.py)."""

import os

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.operators.design import COLS
from artlite.operators.formats import pack_tile, slice_fields
from artlite.operators.overlay.image import pack_rows
from artlite.operators.qwen35.image import KC, QwenImage, gdn_v_heads

from .qwen35_ref import QwenConfig, qwen_gguf

S = 9280


def _padded(f: dict[str, np.ndarray], kc: int) -> dict[str, np.ndarray]:
    """Fields of a chunk narrower than kc columns (a padded K's last) with zero super-blocks to kc."""
    sb = f["q" if "q" in f else "w"].shape[1] // 256
    out = {}
    for k, v in f.items():
        n = kc if k in ("q", "w") else v.shape[1] * (kc // 256) // sb
        out[k] = np.concatenate([v, np.zeros((v.shape[0], n - v.shape[1], *v.shape[2:]), v.dtype)], axis=1)
    return out


@pytest.mark.parametrize("size", ["9B", "27B", "1B"])
def test_layout(size: str) -> None:
    path = qwen_gguf("Q4_K_M", size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    img = QwenImage(g, cfg)
    assert img.tile_bytes == S
    end = 0
    for r in img.regions:
        assert r.offset >= end and r.offset % 64 == 0
        end = r.offset + r.nbytes
        assert {r.tile(rt, kc) for rt in range(r.n_rt) for kc in range(r.n_kc)} == set(range(r.n_rt * r.n_kc))
        for b in r.blocks:
            rows = b.rows[b.rows >= 0]
            total = sum(g.tensor_info[t][0][1] for t in b.tensors)
            want = total if b.name != "lm" else min(img.vocab, cfg.vocab_size)
            assert sorted(rows.tolist()) == list(range(want)), (r.layer, r.name, b.name)
    assert end <= img.nbytes
    assert img.vocab % (COLS * 64) == 0 and img.vocab - cfg.vocab_size < COLS * 64
    # the GDN columns: q / k heads 2c, 2c + 1 and their v heads, in increasing order
    for c in range(COLS):
        vh = gdn_v_heads(cfg, c)
        assert len(vh) == cfg.ssm_v_heads // COLS and all(j % cfg.ssm_k_heads // 2 == c for j in vh)
    L = cfg.gdn_layers[0]
    qkv = img.region(L, "gdn_in").block("qkv").rows
    d = cfg.ssm_state
    assert qkv[1, 0] == 2 * d and qkv[1, 2 * d] == cfg.ssm_k_heads * d + 2 * d
    assert qkv[1, 4 * d] == 2 * cfg.ssm_k_heads * d + gdn_v_heads(cfg, 1)[0] * d
    # a packed tile of the file against pack_tile of its rows
    blk = img.region(L, "gdn_in").block("z")
    t = g.tensor(blk.tensors[0])
    tiles = pack_rows([t], blk.rows[3, :blk.R], blk.fmt, blk.R, S, cfg.hidden_size // KC)
    f = slice_fields(t.fields(), blk.rows[3, :blk.R], k0=cfg.hidden_size - KC, cols=t.cols, kc=KC)
    assert np.array_equal(tiles[0, -1], pack_tile(blk.fmt, f, S))
    # down's K: the ffn padded to whole chunks (MiniCPM-V: 3584 -> 4096), its last chunk's columns past it zero
    dn = img.region(L, "down")
    assert dn.K == -(-cfg.intermediate_size // KC) * KC
    blk = dn.block("down")
    t = g.tensor(blk.tensors[0])
    tiles = pack_rows([t], blk.rows[5, :blk.R], blk.fmt, blk.R, S, dn.n_kc)
    k0 = (dn.n_kc - 1) * KC
    f = slice_fields(t.fields(), blk.rows[5, :blk.R], k0=k0, cols=t.cols, kc=KC)
    assert np.array_equal(tiles[0, -1], pack_tile(blk.fmt, _padded(f, KC) if t.cols - k0 < KC else f, S))


@pytest.mark.parametrize("size", ["9B", "1B"])
def test_pack_image(size: str) -> None:
    """QwenImage.pack of a GDN and an attention layer and a slice of the LM head (9B: down's K chunks in groups of 4;
    1B: down's padded K in one group): sampled tiles of every block at their places (``Region.tile``) as pack_rows
    packs their rows."""
    path = qwen_gguf("Q4_K_M", size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    img = QwenImage(g, cfg, layers=[cfg.gdn_layers[0], cfg.attention.index(True)], vocab=COLS * 64)
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out)
    assert not out[:img.regions[0].offset].any()
    rng = np.random.default_rng(0)
    for r in img.regions:
        assert r.name != "down" or (r.kc_group < r.n_kc) == (size == "9B")
        for b in r.blocks:
            c, t = int(rng.integers(COLS)), int(rng.integers(b.tiles))
            ref = pack_rows([g.tensor(n) for n in b.tensors], b.rows[c, t * b.R:(t + 1) * b.R], b.fmt, b.R, S, r.n_kc)
            col = out[r.offset + c * r.col_bytes:r.offset + (c + 1) * r.col_bytes].reshape(-1, S)
            for kc in range(r.n_kc):
                assert np.array_equal(col[r.tile(r.rt0[b.name] + t, kc)], ref[0, kc]), (r.layer, r.name, b.name, kc)
