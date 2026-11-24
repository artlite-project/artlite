"""The Inkling 2 weight image (models/inkling2/image.py), on the host (no NPU): the layout of UD-Q4_K_XL (every row of
every matrix in exactly one column's block: a KV head's attention on its column, the residual's channels 288 c ..; the
loop layers' controllers; the expert units' slabs back to back per expert), the ink core's objects (the column slices
of the queries and taps, the controllers' rows), and the packed image: sampled tiles of every block at their places
(a unit's down from its K slice of the file's rows, the planes of F32 and BF16 tensors)."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from artlite.decode.image import Block, Region, pack_rows
from artlite.gguf_model import GGUFFile, QTensor
from artlite.model_config import InklingConfig, inkling_gguf
from artlite.operators.design import COLS
from models.inkling2.image import CTL_RANK, EXPERT_UNITS, KC, InklingImage, unit_k

from ..model_files import require

S = 9216


@pytest.fixture(scope="module")
def ink() -> tuple[GGUFFile, InklingConfig]:
    path = inkling_gguf()
    require(path)
    g = GGUFFile(path)
    return g, InklingConfig.from_gguf(g)


def test_layout(ink: tuple[GGUFFile, InklingConfig]) -> None:
    g, cfg = ink
    img = InklingImage(g, cfg)
    H, E, F, hd = cfg.hidden_size, cfg.n_experts, cfg.expert_ffn, cfg.head_dim
    assert img.tile_bytes == S and img.formats() == {"Q8_0", "NVFP4", "BFP16", "Q8_0H"} and img.screen
    assert img.vocab == 201216 and img.column_major and img.h8 == 288
    names = [r.name for r in img.regions]
    assert all(names[i - 1] == n + "_norm" for i, n in enumerate(names) if n in ("q", "k", "v", "router", "lm"))
    for r in img.regions:
        if r.data:
            continue
        for b in r.blocks:
            rows = b.rows[b.rows >= 0]
            total = sum(img.tensor(t).rows for t in b.tensors) * (2 if b.pairs else 1)
            want = cfg.vocab_size if b.name == "lm" else total
            assert len(rows) == want and len(np.unique(rows)) == want, (r.layer, r.name, b.name)
    for L in (0, 5, 6, 11, 20):  # (KV head c's query heads on column c, its K / V rows; the residual's channels 288 c ..)
        q = img.region(L, "q")
        for c in (0, 7):
            assert np.array_equal(q.block("q").rows[c][:3 * hd], 3 * hd * c + np.arange(3 * hd))
            assert np.array_equal(q.block("r").rows[c][:48], 48 * c + np.arange(48))
            assert np.array_equal(q.block("sink").rows[c][:6], 6 * c + np.arange(6))  # (heads 3 c .., two planes)
            assert np.array_equal(img.region(L, "k").blocks[0].rows[c][:hd], hd * c + np.arange(hd))
            assert np.array_equal(img.region(L, "o").blocks[0].rows[c][:288], 288 * c + np.arange(288))
        assert ("kd" in [b.name for b in img.region(L, "k").blocks]) == (cfg.kinds[L] == "full"), L
        loop = cfg.core_start <= L < cfg.core_end
        assert all(((L, f"d{s}a") in img._index) == loop and ((L, f"d{s}b") in img._index) == loop
                   and ((L, f"ctl_{s}") in img._index) == loop for s in ("q", "k", "v", "ffn")), L
    # the experts: per expert its units' gate / up and down back to back, a unit's gate / up its h's rows
    for L in (0, 23):
        offs = img.unit_offsets(L)
        gu0 = img.region(L, "exp_gu0")
        for u, (h0, n) in enumerate(EXPERT_UNITS):
            gu, dn = img.region(L, f"exp_gu{u}"), img.region(L, f"exp_dn{u}")
            assert gu.offset == gu0.offset + offs[u] and dn.offset == gu.offset + gu.col_bytes
            assert gu.col_stride == dn.col_stride == offs[-1] and gu.cols == dn.cols == E
            rows = gu.blocks[0].rows[5]
            rows = rows[rows >= 0] - 2 * F * 5
            assert sorted(rows) == [*range(h0, h0 + n), *range(F + h0, F + h0 + n)], u
            b = dn.blocks[0]
            assert b.kslice == (h0, n) and b.G == KC // unit_k(n) and sorted(b.rows[5][b.rows[5] >= 0]) == \
                list(range(5 * H, 6 * H))


def _tensor(img: InklingImage, name: str, kslice: tuple[int, int]) -> QTensor:
    """A block's tensor as its tiles hold it (a K slice: its columns, zero-padded to whole super-blocks)."""
    t = img.tensor(name)
    k0, k = kslice
    if not k:
        return t
    raw = t.raw if t.raw.dtype == np.uint8 else t.raw.view(np.uint8)
    rb = raw.shape[1]
    out = np.zeros((t.rows, unit_k(k) * rb // t.cols), np.uint8)
    out[:, :k * rb // t.cols] = raw[:, k0 * rb // t.cols:(k0 + k) * rb // t.cols]
    return QTensor(name=name, qtype=t.qtype, rows=t.rows, cols=unit_k(k), raw=out)


def _tail(img: InklingImage, r: Region, b: Block, c: int, q: int) -> np.ndarray:
    """The tail tile of group q of a ``Region.qt`` block in column c, packed from a tensor whose tile row v is the group's
    row tiles' rows v over the tail's columns, one after the other (zero past the group's row tiles; F32 planes: each
    row's own)."""
    k0, kt = r.K_in - 256 * r.qt, 256 * r.qt
    ts = [img.tensor(x) for x in b.tensors]
    raws = [t.raw if t.raw.dtype == np.uint8 else t.raw.view(np.uint8) for t in ts]
    cb = raws[0].shape[1] * kt // ts[0].cols  # (a row's bytes of the tail)
    off = raws[0].shape[1] * k0 // ts[0].cols
    stack = np.concatenate(raws)
    pairs = ts[0].qtype == "F32"
    m = min(r.qg, b.tiles - q * r.qg)
    rows = np.zeros((b.R, r.qg * cb), np.uint8)
    lo = np.zeros(b.R, np.int64)
    for j in range(m):
        for v in range(b.R):
            i = int(b.rows[c, (q * r.qg + j) * b.R + v])
            if i >= 0:
                src = i // 2 if pairs else i
                rows[v, j * cb:(j + 1) * cb] = stack[src, off:off + cb]
                lo[v] = i % 2 if pairs else 0
    t = QTensor(name="tail", qtype=ts[0].qtype, rows=b.R, cols=r.qg * kt,
                raw=rows.view(np.float32) if pairs else rows)
    idx = 2 * np.arange(b.R) + lo if pairs else np.arange(b.R)
    return pack_rows([t], idx, b.fmt, b.R, S, 1)[0, 0]


def test_pack(ink: tuple[GGUFFile, InklingConfig]) -> None:
    """Sampled tiles of every block against ``pack_rows``: a row tile's full chunks, a ``Region.qt`` block's group's tail
    tile (its row tiles' tails side by side, ``_tail``), every tile of a block's last group."""
    g, cfg = ink
    img = InklingImage(g, cfg, layers=[0, 5, 11, 20], vocab=COLS * 64)  # (SWA, full, a full loop layer, Q8_0 experts)
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out)
    rng = np.random.default_rng(0)
    for r in img.regions + img.slabs:
        if r.data:
            continue
        for b in r.blocks:
            for t in {int(rng.integers(b.tiles)), int(rng.integers(b.tiles)), b.tiles - 1}:
                c = int(rng.integers(r.cols))
                n = b.R * b.G
                rows = b.rows[c, t * n:(t + 1) * n]
                ref = pack_rows([_tensor(img, x, b.kslice) for x in b.tensors], rows, b.fmt, b.R, S, r.n_kc, G=b.G)
                col = out[r.offset + c * r.col_stride:r.offset + c * r.col_stride + r.col_bytes].reshape(-1, S)
                for kc in range(r.n_kc - (1 if r.qt else 0)):
                    assert np.array_equal(col[r.tile(r.rt0[b.name] + t, kc)], ref[0, kc]), (r.layer, r.name, b.name, kc)
                if r.qt:
                    tail = col[r.tile(r.rt0[b.name] + t, r.n_kc - 1)]
                    assert np.array_equal(tail, _tail(img, r, b, c, t // r.qg)), (r.layer, r.name, b.name, t)


def test_tails(ink: tuple[GGUFFile, InklingConfig]) -> None:
    """No K padding: every input of the hidden is a ``Region.qt`` region of 2 chunks and a quarter, an expert unit's down
    its K slice; zero super-blocks only in a block's last group of fewer than 4 row tiles (counted over the whole image,
    against the tiles' bytes)."""
    g, cfg = ink
    img = InklingImage(g, cfg)
    H = cfg.hidden_size
    zero = total = 0
    for r in img.regions + img.slabs:
        if r.data:
            continue
        total += r.nbytes
        if r.name == "o":
            assert r.K_in == r.K == cfg.num_attention_heads * cfg.head_dim and not r.qt
            continue
        if r.name.startswith("exp_dn"):
            assert not r.qt and r.blocks[0].kslice[1] * r.blocks[0].G <= KC
            continue
        assert r.qt == 1 and r.K_in == H and r.n_kc == 3, (r.layer, r.name)
        for b in r.blocks:
            assert r.block_tiles(b) == 2 * b.tiles + r.groups(b)
            zero += (4 * r.groups(b) - b.tiles) * r.cols * S // 4
    print(f"zero super-blocks: {zero / 1e6:.2f} MB of {total / 1e9:.3f} GB of tiles ({100 * zero / total:.3f}%)")
    assert zero < 2e-3 * total


def test_ink_objects(ink: tuple[GGUFFile, InklingConfig]) -> None:
    g, cfg = ink
    img = InklingImage(g, cfg, layers=[6, 11], vocab=COLS * 64)
    H = cfg.hidden_size
    for L in (6, 11):
        q = img.ink_objects(L, "inkq").reshape(COLS, -1)
        J = 1 if cfg.kinds[L] == "swa" else 3
        want = np.concatenate([g.tensor(f"blk.{L}.attn_res_query.weight").dequantize().reshape(J, H),
                               cfg.msg_queries(g, L)])
        for c in (0, 3):
            got = q[c][:(J + 4) * 288 * 4].view(np.float32).reshape(J + 4, 288)
            assert np.array_equal(got, want[:, 288 * c:288 * c + 288]), (L, c)
        f = img.ink_objects(L, "inkf").reshape(COLS, -1)
        taps = g.tensor(f"blk.{L}.attn_post_conv.weight").dequantize().reshape(H, 4).T  # (a channel's taps per row)
        got = f[2][:2304].view(bfloat16).astype(np.float32).reshape(4, 288)
        assert np.array_equal(got, taps[:, 576:864])  # (bf16-exact)
        qf = g.tensor(f"blk.{L}.ffn_res_query.weight").dequantize().reshape(H)
        assert np.array_equal(img.ink_objects(L, "inkfq").reshape(COLS, -1)[2][:1152].view(np.float32), qf[576:864])
        ctl = img.ink_objects(L, "ctl_k").reshape(COLS, 2, -1)
        for i, part in enumerate(("forget", "gain")):
            w = g.tensor(cfg.ctrl_tensor(L, "k", part)).dequantize().reshape(H, CTL_RANK)
            got = ctl[7, i, :288 * CTL_RANK * 2].view(bfloat16).astype(np.float32).reshape(288, CTL_RANK)
            assert np.array_equal(got, w[7 * 288:]), part
