"""The Llama config and weight image (``model_config.LlamaConfig``, models/llama/image.py; Qwen3's too), on the host
(no NPU): the config from the GGUF metadata, the layout of a file (every row of every matrix in exactly one column's
block, Q / K heads in rotate-half order, the padded vocabulary, the regions in step order) and the packed image's tiles
and norm-weight objects at their places."""

import os

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode.image import BLOCK_BYTES, COL_SLOT, pack_rows
from artlite.gguf_model import GGUFFile
from artlite.model_config import LlamaConfig, llama_gguf, qwen3_gguf
from artlite.operators.design import COLS
from models.llama.image import KC, LlamaImage

SIZES = {"1B": (16, 2048, 8192, 32, 64, 32.0), "3B": (28, 3072, 8192, 24, 128, 32.0),
         "8B": (32, 4096, 14336, 32, 128, 8.0)}  # layers, hidden, ffn, heads, head_dim, RoPE factor


def _gguf(size: str, quant: str = "Q4_K_M") -> GGUFFile:
    path = llama_gguf(quant, size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    return GGUFFile(path)


@pytest.mark.parametrize("size", sorted(SIZES))
def test_config(size: str) -> None:
    cfg = LlamaConfig.from_gguf(_gguf(size))
    n, H, I, nh, hd, factor = SIZES[size]
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.head_dim) == \
        (n, H, I, nh, hd)
    assert cfg.num_key_value_heads == 8 and cfg.vocab_size == 128256 and cfg.max_position_embeddings == 131072
    assert set(cfg.eos_token_ids) == {128001, 128008, 128009} and cfg.rope_base == 500000.0
    plain = 500000.0 ** (-np.arange(0, hd, 2) / hd)
    inv = cfg.rope_inv_freq()
    assert inv[0] == plain[0] and inv[-1] == plain[-1] / factor  # (llama3 scaling: high frequencies kept, low ones / factor)


@pytest.mark.parametrize(("name", "tile_bytes", "vocab", "eos"), [("0.6B", 10240, 151936, {151645, 151643}),
                                                                   ("embedding-0.6B", 9216, 151669, {151643, 151645})])
def test_qwen3(name: str, tile_bytes: int, vocab: int, eos: set[int]) -> None:
    """Qwen3 on the Llama path: q / k norms, NeoX rows (the image's Q / K blocks take the file's rows as they are),
    Qwen's end-of-generation tokens, the tile size of Q4_1 (16 rows) and of a file of Q8_0 alone (8 rows: the floor's
    9216 B, not 9 rows' 9792 of which 8 hold rows)."""
    path = qwen3_gguf(name)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    cfg = LlamaConfig.from_gguf(g)
    assert cfg.neox and cfg.qk_norm and (cfg.num_hidden_layers, cfg.hidden_size, cfg.head_dim, cfg.q_per_kv) == \
        (28, 1024, 128, 2)
    assert cfg.vocab_size == vocab and set(cfg.eos_token_ids) == eos and cfg.rope_freq_factors is None
    img = LlamaImage(g, layers=[0, 1])
    assert img.tile_bytes == tile_bytes and img.vocab == -(-vocab // 512) * 512
    qkv = img.region(1, "qkv")
    assert qkv.block("q").rows[3].tolist() == list(range(3 * 256, 4 * 256)) and \
        qkv.block("k").rows[3].tolist() == list(range(3 * 128, 4 * 128))


@pytest.mark.parametrize("size", sorted(SIZES))
def test_layout(size: str) -> None:
    g = _gguf(size)
    img = LlamaImage(g)
    cfg = img.cfg
    hd, qh = cfg.head_dim, cfg.q_per_kv
    assert img.tile_bytes == 9280 and img.kc == KC and img.vocab == 128512
    names = [r.name for r in img.regions]
    assert names == ["qkv_norm", "qkv", "o", "gate_up_norm", "gate_up", "down"] * cfg.num_hidden_layers + ["lm_norm", "lm"]
    # column-major in layout blocks of whole layers (8B: two, each under BLOCK_BYTES): a column's regions in a row, the
    # column stride the next odd multiple of COL_SLOT
    blocks = img._blocks()
    assert len(blocks) == (2 if size == "8B" else 1) and blocks[-1][-1].name == "lm"
    end = img.regions[0].offset
    for blk in blocks:
        start = end
        for r in blk:
            assert r.offset == end
            end = r.offset + r.col_bytes
            assert {r.tile(rt, kc) for rt in range(r.n_rt) for kc in range(r.n_kc)} == set(range(r.n_rt * r.n_kc))
            for b in r.blocks:
                rows = b.rows[b.rows >= 0]
                total = sum(g.tensor_info[t][0][1] for t in b.tensors)
                assert sorted(rows.tolist()) == list(range(total)), (r.layer, r.name, b.name)
        stream, stride = end - start, blk[0].col_stride
        assert all(r.col_stride == stride for r in blk) and COLS * stream <= BLOCK_BYTES
        assert stride % (2 * COL_SLOT) == COL_SLOT and stream <= stride < stream + 2 * COL_SLOT
        assert img.runs(blk[0], blk[-1]) == [(blk[0], stream)]
        end = start + COLS * stride
    assert len(img.runs(img.regions[0], img.regions[-1])) == len(blocks)
    assert img.region(0, "down").kc_group == (2 if size == "8B" else 4)
    # Q / K: column c holds query heads qh c .., KV head c, each in rotate-half order (llama.cpp interleaves the pairs)
    qkv = img.region(1, "qkv")
    q, k = qkv.block("q").rows, qkv.block("k").rows
    h = qh * 2 + 1  # (column 2's second head)
    assert q[2, hd:hd + hd // 2].tolist() == list(range(h * hd, (h + 1) * hd, 2))
    assert q[2, hd + hd // 2:2 * hd].tolist() == list(range(h * hd + 1, (h + 1) * hd, 2))
    assert k[5, :hd // 2].tolist() == list(range(5 * hd, 6 * hd, 2))
    assert qkv.block("v").rows[5, :hd].tolist() == list(range(5 * hd, 6 * hd))
    # the LM head: vocab / 8 rows per column (then the row tiles' padding), the last column's last 256 rows padding
    lm = img.region(-1, "lm").block("lm")
    assert lm.tensors == ("output.weight" if size == "8B" else "token_embd.weight",)
    assert (lm.rows[:7, :16064] >= 0).all() and lm.rows[7, 16064 - 257] == 128255 and (lm.rows[7, 16064 - 256:] < 0).all()


def test_pack_image() -> None:
    """LlamaImage.pack of two layers and a slice of the LM head (1B Q5_K_M: Q5_K and Q6_K tiles): sampled tiles of every
    block at their places as pack_rows packs their rows, the norm-weight objects in every column."""
    g = _gguf("1B", "Q5_K_M")
    img = LlamaImage(g, layers=[0, 15], vocab=COLS * 64)
    T = img.tile_bytes
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out, release=False)
    assert not out[:img.regions[0].offset].any()
    rng = np.random.default_rng(0)
    for r in img.regions:
        cols = [out[r.offset + c * r.col_stride:][:r.col_bytes] for c in range(COLS)]
        if r.data:
            name = "output_norm" if r.layer < 0 else f"blk.{r.layer}.{'attn' if r.name == 'qkv_norm' else 'ffn'}_norm"
            w = g.tensor(name + ".weight").dequantize().astype(bfloat16).view(np.uint16).reshape(-1)
            for col in cols:
                assert np.array_equal(col[:w.nbytes].view(np.uint16), w) and not col[w.nbytes:].any()
            continue
        for b in r.blocks:
            c, t = int(rng.integers(COLS)), int(rng.integers(b.tiles))
            ref = pack_rows([g.tensor(n) for n in b.tensors], b.rows[c, t * b.R:(t + 1) * b.R], b.fmt, b.R, T, r.n_kc)
            col = cols[c].reshape(-1, T)
            for kc in range(r.n_kc):
                assert np.array_equal(col[r.tile(r.rt0[b.name] + t, kc)], ref[0, kc]), (r.layer, r.name, b.name, kc)
