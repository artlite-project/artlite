"""The qwen35 weight image (operators/qwen35/image.py), on the host (no NPU): the layout of a GGUF file (every row of
every matrix in exactly one column's block, the GDN heads' column partition, MiniCPM-V's padded vocabulary and down K,
column-major where the decode's K groups allow it; an MoE file's router planes, shared expert and expert slabs) and the
packed image's tiles and norm-weight objects at their places (the C++ packer against the slots' ``pack_tile``:
test_overlay_gemv.py), an MoE's experts against the oracle's matrices (``qwen35_ref.experts``)."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile, QTensor
from artlite.operators import bfp16
from artlite.operators.design import COLS
from artlite.operators.formats import pack_tile, slice_fields
from artlite.operators.overlay.decode import norm_objects
from artlite.operators.overlay.image import BLOCK_BYTES, pack_rows
from artlite.operators.qwen35.image import KC, NORMS, QwenImage, gdn_v_heads

from .qwen35_ref import QwenConfig, experts, qwen_gguf

S = 9280
MOE = qwen_gguf("Q4_K_M", "35B-A3B")


@pytest.mark.parametrize("size", ["9B", "27B", "1B"])
def test_layout(size: str) -> None:
    path = qwen_gguf("Q4_K_M", size)
    if not os.path.exists(path):
        pytest.skip(f"{path}: not present")
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    img = QwenImage(g, cfg)
    assert img.tile_bytes == S
    assert img.column_major == (size != "27B")  # (the 27B: its output projections' and down's groups are strided)
    if img.column_major:  # a column's regions in a row (each normed phase's norm objects first), one run per block
        names = [r.name for r in img.regions]
        assert all(names[i - 1] == n + "_norm" for i, n in enumerate(names) if n in NORMS)
        blocks = img._blocks()
        assert len(blocks) == (2 if size == "9B" else 1)
        for blk in blocks:
            stream = sum(r.col_bytes for r in blk)
            assert img.runs(blk[0], blk[-1]) == [(blk[0], stream)] and COLS * stream <= BLOCK_BYTES
    end = 0
    for r in img.regions:
        assert r.offset >= end and r.offset % 64 == 0
        end = r.offset + (r.col_bytes if img.column_major else r.nbytes)
        if r.data:
            assert r.data == img.n_norm * S
            continue
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
    assert np.array_equal(tiles[0, -1], pack_tile(blk.fmt, f, S))


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
        if r.data:  # (every column's copy of the phase's norm-weight objects)
            L, name = r.layer, r.name.removesuffix("_norm")
            w = np.asarray(g.tensor("output_norm.weight" if L < 0 else f"blk.{L}.{NORMS[name]}.weight").raw, np.float32)
            want = norm_objects(w.reshape(-1), S).view(np.uint8).reshape(-1)
            for c in range(COLS):
                assert np.array_equal(out[r.offset + c * r.col_stride:][:want.size], want), (L, r.name, c)
        for b in r.blocks:
            c, t = int(rng.integers(COLS)), int(rng.integers(b.tiles))
            ref = pack_rows([g.tensor(n) for n in b.tensors], b.rows[c, t * b.R:(t + 1) * b.R], b.fmt, b.R, S, r.n_kc)
            col = out[r.offset + c * r.col_stride:r.offset + c * r.col_stride + r.col_bytes].reshape(-1, S)
            for kc in range(r.n_kc):
                assert np.array_equal(col[r.tile(r.rt0[b.name] + t, kc)], ref[0, kc]), (r.layer, r.name, b.name, kc)


def _moe_image(layers: list[int] | None = None) -> tuple[GGUFFile, QwenConfig, QwenImage]:
    if not os.path.exists(MOE):
        pytest.skip(f"{MOE}: not present")
    g = GGUFFile(MOE)
    cfg = QwenConfig.from_gguf(g)
    return g, cfg, QwenImage(g, cfg, layers=layers, vocab=None if layers is None else COLS * 64)


def _group_out(n: int, R: int, G: int) -> np.ndarray:
    """The output of each of a row-group block's n entries per column (tile i, row r, slot j: G R i + R j + r)."""
    v, j = np.divmod(np.arange(n), G)
    return G * R * (v // R) + R * j + v % R


def test_layout_moe() -> None:
    """Qwen3.6-35B-A3B: the config's MoE fields; per layer the shared expert's gate / up (every row once), the
    router's planes (each of the column's 32 experts' hi, lo), the shared gate's planes in every column, its down (row groups of
    2: tile i's row r, slot j = row 2 R i + R j + r of the column's 256); the expert slabs: half 2 e + h's gate / up
    rows (outputs h F / 2 .. of expert e), its down as row groups of 4 (slot j of tile i's row r: output 4 R i + R j
    + r over the half's K), every row of the 3-D tensors (the down's view: halves) once; an expert's two halves one
    run, the layers' slab areas after the column-major regions."""
    _, cfg, img = _moe_image()
    E, F, H = cfg.n_experts, cfg.expert_ffn, cfg.hidden_size
    assert (cfg.moe, E, cfg.n_experts_used, F, cfg.shared_ffn, cfg.intermediate_size) == (True, 256, 8, 512, 512, 0)
    assert (cfg.num_hidden_layers, H, cfg.num_key_value_heads, cfg.attention_layers[:2]) == (40, 2048, 2, [3, 7])
    assert img.tile_bytes == S and img.column_major and len(img._blocks()) == 1
    assert {r.name for r in img.regions} == {"gdn_in_norm", "gdn_in", "gdn_out", "attn_in_norm", "attn_in", "attn_out",
                                             "moe_in_norm", "moe_in", "sdown", "lm_norm", "lm"}
    end = 0
    for r in img.regions:
        assert r.offset >= end and r.offset % 64 == 0 and r.cols == COLS
        end = r.offset + r.col_bytes
    end = img.regions[0].offset + COLS * img.regions[0].col_stride  # (the column-major block)
    for L in range(cfg.num_hidden_layers):
        b = f"blk.{L}."
        mi = img.region(L, "moe_in")
        assert [x.name for x in mi.blocks] == ["sgu", "router", "sgate"] and mi.n_kc == 2
        router, sgate = mi.block("router").rows, mi.block("sgate").rows
        mine = 32 * np.arange(COLS)[:, None] + np.arange(32)
        assert np.array_equal(router, np.stack([2 * mine, 2 * mine + 1], -1).reshape(COLS, -1))  # (8 tiles per column)
        assert (sgate == [2 * E, 2 * E + 1, -1, -1, -1, -1, -1, -1]).all()
        gu = mi.block("sgu").rows
        assert sorted(gu[gu >= 0].tolist()) == list(range(2 * cfg.shared_ffn))
        sd = img.region(L, "sdown")
        blk = sd.block("sdown")
        assert (sd.n_kc, blk.G, blk.fmt) == (1, 2, img.fmt(b + "ffn_down_shexp.weight"))
        want = _group_out(blk.rows.shape[1], blk.R, 2)
        for c in range(COLS):
            assert np.array_equal(blk.rows[c], np.where(want < 256, 256 * c + want, -1))
        # the experts
        egu, edn = img.region(L, "exp_gu"), img.region(L, "exp_down")
        assert egu.cols == edn.cols == 2 * E and img.slabs.index(egu) == 2 * L and img.slabs.index(edn) == 2 * L + 1
        assert egu.offset == end and edn.offset == egu.offset + egu.col_bytes
        assert egu.col_stride == edn.col_stride == egu.col_bytes + edn.col_bytes
        end += 2 * E * egu.col_stride
        gu = egu.block("gu").rows.reshape(2 * E, -1, 2, 8)  # (8 x [8 gate | 8 up] per group of 64 outputs)
        k = np.arange(2 * E)[:, None]
        assert np.array_equal(np.sort(gu[:, :, 0].reshape(2 * E, -1), 1), F // 2 * k + np.arange(F // 2))
        assert np.array_equal(gu[:, :, 1], gu[:, :, 0] + E * F)  # (half k = 2 e + h: expert e's outputs h F / 2 ..)
        dn = edn.block("down")
        assert (edn.n_kc, dn.G, dn.fmt) == (1, 4, img.fmt(b + "ffn_down_exps.weight"))
        o = _group_out(dn.rows.shape[1], dn.R, 4)
        assert np.array_equal(dn.rows, np.where(o < H, 2 * (H * (k // 2) + o) + k % 2, -1))
        assert sorted(dn.rows[dn.rows >= 0].tolist()) == list(range(2 * E * H))
        assert img.tensor(dn.tensors[0]).rows == 2 * E * H and img.tensor(dn.tensors[0]).cols == F // 2
    assert end + 65536 <= img.nbytes < os.path.getsize(MOE)


def test_pack_moe() -> None:
    """QwenImage.pack of an attention layer (Q6_K expert downs) and a GDN layer (Q4_K): sampled tiles of every block,
    slab blocks too, at their places as pack_rows packs their rows; the router's planes decoded from the image (hi
    exactly bfp16.values of the F32 rows times w / bf16(w) of the layer's post_attention_norm, hi + lo within 4e-5 of
    them, relative RMS per row; the shared gate's in every column); an expert's slab rows as the oracle's matrices
    (``qwen35_ref.experts``: gate / up rows, the down's columns of the half); the MoE model's K groups of 2048 (gdn_out,
    attn_out: 2 groups of 2 chunks)."""
    g, cfg, img = _moe_image([3, 5])
    E, F, H = cfg.n_experts, cfg.expert_ffn, cfg.hidden_size
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out)
    rng = np.random.default_rng(0)

    def ratio(L: int) -> np.ndarray:  # (the router's rows times w / bf16(w))
        w = np.asarray(g.tensor(f"blk.{L}.post_attention_norm.weight").raw, np.float32).reshape(-1)
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
                ref = pack_rows([tensor(x) for x in b.tensors], b.rows[c, t * n:(t + 1) * n], b.fmt, b.R, S, r.n_kc,
                                G=b.G)
                col = out[r.offset + c * r.col_stride:r.offset + c * r.col_stride + r.col_bytes].reshape(-1, S)
                for kc in range(r.n_kc):
                    assert np.array_equal(col[r.tile(r.rt0[b.name] + t, kc)], ref[0, kc]), (r.layer, r.name, b.name, kc)
    for L in (3, 5):
        b = f"blk.{L}."
        w = np.concatenate([g.tensor(b + "ffn_gate_inp.weight").raw, g.tensor(b + "ffn_gate_inp_shexp.weight").raw]) * ratio(L)
        mi = img.region(L, "moe_in")
        assert img.region(L, "attn_out" if cfg.attention[L] else "gdn_out").kc_group == 2
        rt0 = mi.rt0["router"]
        for c in range(COLS):
            col = out[mi.offset + c * mi.col_stride:].reshape(-1)[:mi.col_bytes].reshape(-1, S)
            vals = np.stack([np.concatenate([_bfp16_rows(col[mi.tile(rt0 + rt, kc)]) for kc in range(2)], 1)
                             for rt in range(9)])  # (tile, 8 rows, H)
            hi, lo = vals[:8].reshape(32, 2, H).transpose(1, 0, 2)
            rows = w[32 * c:32 * c + 32]
            assert np.array_equal(hi, bfp16.values(rows)), (L, c)
            assert np.array_equal(lo, bfp16.values(rows - hi)), (L, c)
            err = np.sqrt(((hi + lo - rows) ** 2).mean(1) / (rows ** 2).mean(1))
            assert err.max() < 4e-5, (L, c, err.max())
            assert np.array_equal(vals[8, 0], bfp16.values(w[E])) and np.array_equal(vals[8, 1], bfp16.values(w[E] - vals[8, 0]))
            assert not vals[8, 2:].any()
        # an expert's slab rows: the oracle's matrices
        e = int(rng.integers(E))
        gate, up, down = experts(g, L)(e)
        gu, dn = img.region(L, "exp_gu").block("gu"), img.region(L, "exp_down").block("down")
        dview = img.tensor(dn.tensors[0])
        for h in range(2):
            rows = gu.rows[2 * e + h]
            for name, ref, base in ((b + "ffn_gate_exps.weight", gate, 0), (b + "ffn_up_exps.weight", up, E * F)):
                idx = rows[(rows >= base) & (rows < base + E * F)] - base
                assert np.array_equal(np.sort(idx), e * F + h * F // 2 + np.arange(F // 2))
                assert np.array_equal(g.rows(name, idx).numpy(), ref[idx - e * F])
            v = dn.rows[2 * e + h]
            o = np.flatnonzero(v >= 0)
            got = QTensor(name="d", qtype=dview.qtype, rows=o.size, cols=F // 2, raw=dview.raw[v[o]]).dequantize().numpy()
            want = down[_group_out(v.size, dn.R, 4)[o], h * F // 2:(h + 1) * F // 2]
            assert np.array_equal(got, want), (L, e, h)


def _bfp16_rows(tile: np.ndarray) -> np.ndarray:
    """A BFP16 tile of 1024 columns -> its 8 rows' values (8, 1024) ([k8][row][exp, 8 int8])."""
    return bfp16.block_values(tile[:128 * 8 * 9].reshape(128, 8, 9)).transpose(1, 0, 2).reshape(8, 1024)
