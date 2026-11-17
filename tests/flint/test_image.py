"""The Flint weight image (models/flint/image.py), on the host (no NPU): the config, the layout of the official int4
file (every row of every matrix in exactly one column's block, a KV head's q heads and K / V halves on its two columns,
the dense layers' FFN and the MoE layers' router planes, ungated shared expert and expert slabs, the padded vocabulary,
column-major with the slabs after), and the packed image: Q4_1_64 tiles decoded back to the file's weights exactly
(the tile's inverse), sampled tiles of every block at their places, the router's planes."""

import numpy as np
import pytest
import torch

from artlite.decode.image import pack_rows
from artlite.formats.common import _word_order
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import FlintConfig, flint_gguf
from artlite.operators import bfp16
from artlite.operators.design import COLS
from models.flint.image import KC, FlintImage

from ..model_files import require

S = 9216


@pytest.fixture(scope="module")
def flint() -> tuple[GGUFFile, FlintConfig]:
    path = flint_gguf()
    require(path)
    g = GGUFFile(path)
    return g, FlintConfig.from_gguf(g)


def test_config(flint: tuple[GGUFFile, FlintConfig]) -> None:
    _, cfg = flint
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size) == (49, 2048, 6144, 150144)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.q_per_kv) == (32, 4, 128, 8)
    assert (cfg.n_dense, cfg.n_experts, cfg.n_experts_used, cfg.expert_ffn, cfg.shared_ffn) == (2, 256, 6, 512, 1024)
    assert cfg.layers("full") == [0, 1, *range(5, 46, 4), 48] and len(cfg.layers("swa")) == 35
    assert (cfg.sliding_window, cfg.rope_base, cfg.rot("swa"), cfg.rot("full")) == (513, 1e4, 64, 0)
    assert cfg.layers("dense") == [0, 1] and cfg.layers("moe") == list(range(2, 49))
    assert set(cfg.eos_token_ids) >= {150001, 150003, 150005}
    assert cfg.sampling == pytest.approx({"temp": 1.0, "top_p": 0.95, "top_k": 20})


def test_layout(flint: tuple[GGUFFile, FlintConfig]) -> None:
    g, cfg = flint
    img = FlintImage(g, cfg)
    E, F, Fs, hd = cfg.n_experts, cfg.expert_ffn, cfg.shared_ffn, cfg.head_dim
    assert img.tile_bytes == S and img.formats() == {"Q4_1_64", "BFP16"} and img.R["Q4_1_64"] == 16
    assert img.column_major and len(img._blocks()) == 1 and img.vocab == 150528
    names = [r.name for r in img.regions]
    assert all(names[i - 1] == n + "_norm" for i, n in enumerate(names) if n in ("qkv", "gate_up", "moe_in", "lm"))
    for r in img.regions:
        if r.data:
            continue
        for b in r.blocks:
            rows = b.rows[b.rows >= 0]
            total = sum(g.tensor_info[t][0][1] for t in b.tensors)
            if b.name == "router":
                total *= 2  # (hi and lo planes)
            want = cfg.vocab_size if b.name == "lm" else total
            assert sorted(rows.tolist()) == list(range(want)), (r.layer, r.name, b.name)
    qkv = img.region(5, "qkv")
    for c in range(COLS):  # (KV head c / 2, half c % 2: q heads 8 (c / 2) + 4 (c % 2) ..)
        h, j = c // 2, c % 2
        q = qkv.block("q").rows[c]
        assert np.array_equal(q[:4 * hd], hd * (8 * h + 4 * j) + np.arange(4 * hd))
        assert np.array_equal(qkv.block("k").rows[c][:hd // 2], hd * h + hd // 2 * j + np.arange(hd // 2))
    assert (img.region(5, "o").n_kc, img.region(5, "o").kc_group, img.region(0, "down").n_kc) == (4, 2, 6)
    assert {r.name for r in img.regions if r.layer == 0} == {"qkv_norm", "qkv", "o", "gate_up_norm", "gate_up", "down"}
    mi = img.region(2, "moe_in")
    assert [b.name for b in mi.blocks] == ["sgu", "router"] and mi.block("router").pairs
    sd = img.region(2, "sdown").block("sdown")
    assert (img.region(2, "sdown").n_kc, sd.G) == (1, 1)
    egu, edn = img.region(2, "exp_gu"), img.region(2, "exp_down")
    assert egu.cols == edn.cols == 2 * E and (edn.block("down").G, edn.n_kc) == (2 * KC // F, 1)
    assert img.slabs[0].offset >= img.regions[-1].offset + COLS * img.regions[-1].col_stride - img.regions[-1].col_stride
    per_token = img.streamed_bytes() + sum(r.nbytes for r in img.slabs) * cfg.n_experts_used / E
    assert 1.45e9 < per_token < 1.48e9, per_token  # (the plan's 1457 MB, with the norm objects)
    assert Fs == 1024


def _q4_1_64_weights(tile: np.ndarray, R: int, nsb: int) -> np.ndarray:
    """A Q4_1_64 tile's weights (R, 256 nsb) float32: the inverse of its packing (``formats.q4``)."""
    hdr = tile[:16 * nsb * R].view(np.uint16).reshape(R, nsb, 2, 4)
    dm = torch.from_numpy(hdr.astype(np.int32).astype(np.int16).copy()).view(torch.bfloat16).float().numpy()
    d, m = np.repeat(dm[:, :, 0], 64, 2).reshape(R, -1), np.repeat(dm[:, :, 1], 64, 2).reshape(R, -1)
    codes = tile[16 * nsb * R:16 * nsb * R + 128 * nsb * R].reshape(nsb, R, 128).transpose(1, 0, 2).reshape(R, -1)
    q = np.stack([codes & 15, codes >> 4], -1).reshape(R, -1)  # (the word order's nibbles)
    w = np.empty((R, 256 * nsb), np.float32)
    w[:, _word_order(nsb)] = q
    return d * w + m


def test_pack(flint: tuple[GGUFFile, FlintConfig]) -> None:
    g, cfg = flint
    img = FlintImage(g, cfg, layers=[0, 5], vocab=COLS * 64)
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out)
    rng = np.random.default_rng(0)

    def ratio(L: int) -> np.ndarray:  # (the router's rows times w / bf16(w))
        w = np.asarray(g.tensor(f"blk.{L}.ffn_norm.weight").raw, np.float32).reshape(-1)
        return (w / torch.from_numpy(w.copy()).bfloat16().float().numpy()).astype(np.float32)

    def tensor(name: str) -> QTensor:
        t = img.tensor(name)
        if t.qtype != "F32":
            return t
        raw = (np.asarray(t.raw, np.float32).reshape(t.rows, t.cols) * ratio(int(name.split(".")[1]))).astype(np.float32)
        return QTensor(name=name, qtype="F32", rows=t.rows, cols=t.cols, raw=raw)

    for r in img.regions + img.slabs:
        for b in r.blocks:
            for _ in range(2):
                c, t = int(rng.integers(r.cols)), int(rng.integers(b.tiles))
                n = b.R * b.G
                rows = b.rows[c, t * n:(t + 1) * n]
                ref = pack_rows([tensor(x) for x in b.tensors], rows, b.fmt, b.R, S, r.n_kc, G=b.G)
                col = out[r.offset + c * r.col_stride:r.offset + c * r.col_stride + r.col_bytes].reshape(-1, S)
                for kc in range(r.n_kc):
                    tile = col[r.tile(r.rt0[b.name] + t, kc)]
                    assert np.array_equal(tile, ref[0, kc]), (r.layer, r.name, b.name, kc)
                    if b.fmt == "Q4_1_64" and b.G == 1 and len(b.tensors) == 1:  # (the tile's inverse: the weights)
                        ok = rows >= 0
                        k0, k1 = kc * KC, min((kc + 1) * KC, img.tensor(b.tensors[0]).cols)
                        want = g.rows(b.tensors[0], rows[ok])[:, k0:k1]
                        assert np.array_equal(_q4_1_64_weights(tile, b.R, 4)[ok, :k1 - k0], want), (r.name, kc)
    # the router's planes: hi exactly bfp16 of the F32 rows times w / bf16(w), hi + lo within 4e-5 of them
    mi = img.region(5, "moe_in")
    w = g.tensor("blk.5.ffn_gate_inp.weight").dequantize().reshape(cfg.n_experts, -1) * ratio(5)
    rt0 = mi.rt0["router"]
    for c in (0, 7):
        col = out[mi.offset + c * mi.col_stride:].reshape(-1)[:mi.col_bytes].reshape(-1, S)
        vals = np.stack([np.concatenate([_bfp16_rows(col[mi.tile(rt0 + rt, kc)]) for kc in range(2)], 1)
                         for rt in range(8)])  # (tile, 8 rows, H)
        hi, lo = vals.reshape(32, 2, -1).transpose(1, 0, 2)
        rows = w[32 * c:32 * c + 32]
        assert np.array_equal(hi, bfp16.values(rows)) and np.array_equal(lo, bfp16.values(rows - hi)), c
        err = np.sqrt(((hi + lo - rows) ** 2).mean(1) / (rows ** 2).mean(1))
        assert err.max() < 4e-5, err.max()


def _bfp16_rows(tile: np.ndarray) -> np.ndarray:
    """A BFP16 tile of 1024 columns -> its 8 rows' values (8, 1024) ([k8][row][exp, 8 int8])."""
    return bfp16.block_values(tile[:128 * 8 * 9].reshape(128, 8, 9)).transpose(1, 0, 2).reshape(8, 1024)
