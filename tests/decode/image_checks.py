"""Checks of a weight image's half regions (``decode.image.Region.half``: an input half a chunk short of whole chunks),
shared by the models' image tests: the layout of their pairs of row tiles and the packed tiles against ``pack_rows`` of
the file's rows."""

import numpy as np

from artlite.decode.image import TileImage, group_rows, pack_rows
from artlite.formats import slot
from artlite.gguf_model import QTensor
from artlite.operators.design import COLS


def tail_tensor(t: QTensor) -> QTensor:
    """A tensor's last 512 columns as a tensor of their own (contiguous rows)."""
    raw = t.raw if t.raw.dtype == np.uint8 else t.raw.view(np.uint8)
    per = raw.shape[1] // (t.cols // 256)  # (bytes per super-block of a row)
    return QTensor(name=t.name, qtype=t.qtype, rows=t.rows, cols=512,
                   raw=np.ascontiguousarray(raw[:, per * (t.cols // 256 - 2):]))


def check_tails(img: TileImage, pad: TileImage, names: set[str], *, seed: int = 0) -> None:
    """``img`` (half regions ``names``, laid out as ``pad``, the same file's image without them): every tile index of a
    column used once; a ``pt`` block's pairs of row tiles in 2 n_kc - 1 tiles each (an odd block's last row tile in n_kc,
    as an unpaired block's), exactly the blocks the image pairs (``TileImage.pairs``); packed, a row tile's whole chunks
    as ``pack_rows`` packs them, a pair's tail tile its last 512 columns as row groups of 2 (``group_rows``: a single's
    second half zero rows), an unpaired block's tiles as in the padded image."""
    T = img.tile_bytes
    out = np.zeros(img.nbytes, np.uint8)
    img.pack(out)
    rng = np.random.default_rng(seed)
    for r, rp in zip(img.regions, pad.regions, strict=True):
        if r.data:
            continue
        assert r.half == (r.name in names) and r.K_in == (r.K - r.kc // 2 if r.half else r.K), (r.layer, r.name)
        assert sorted(r.tile(rt, kc) for rt in range(r.n_rt) for kc in range(r.n_kc) if r.tail_half(rt) == 0
                      or kc < r.n_kc - 1) == list(range(r.n_t)), (r.layer, r.name)
        singles = sum(b.tiles % 2 if b.pt else b.tiles for b in r.blocks)
        assert r.n_t == (r.n_rt - singles) // 2 * r.pair + r.n_kc * singles if r.half else r.n_t == rp.n_t
        for b in r.blocks:
            assert b.pt == (r.half and not b.fmt2 and not b.pairs and not slot(b.fmt).mmv_split(b.fmt, img.nsb)
                            and img.pairs(r.name, b.name, b.fmt, b.R)), (r.layer, r.name, b.name)
            cols = rng.choice(COLS, 2, replace=False)
            for fmt, R, prt, names_, rows in b.parts(img.tensor(b.tensors[0]).rows)[:3]:
                ts = [img.tensor(n) for n in names_]
                for c in cols:
                    col = out[r.offset + c * r.col_stride:][:r.col_bytes].reshape(-1, T)
                    n = rows.shape[1] // R
                    for j in sorted({0, min(1, n - 1), n - 1, int(rng.integers(n))}):
                        rt = r.rt0[b.name] + prt + j
                        want = pack_rows(ts, rows[c, j * R:(j + 1) * R], fmt, R, T, r.n_kc)
                        for kc in range(r.n_kc - b.pt):
                            assert np.array_equal(col[r.tile(rt, kc)], want[0, kc]), (r.layer, r.name, fmt, j, kc)
                        if b.pt:
                            j0 = j - j % 2
                            pr = np.full(2 * R, -1, np.int64)
                            for e in range(min(2, n - j0)):
                                pr[e * R:(e + 1) * R] = rows[c, (j0 + e) * R:(j0 + e + 1) * R]
                            tail = pack_rows([tail_tensor(t) for t in ts], group_rows(pr[None], R, 2)[0], fmt, R, T, 1,
                                             G=2)
                            assert r.tail_half(rt) == j % 2
                            assert np.array_equal(col[r.tile(rt, r.n_kc - 1)], tail[0, 0]), (r.layer, r.name, fmt, j)


def check_unpaired(img: TileImage, pad: TileImage) -> None:
    """An image of half regions without pairs packs to the bytes of the padded image (the regions' places and tiles)."""
    assert not any(b.pt for r in img.regions for b in r.blocks) and img.nbytes == pad.nbytes
    assert [(r.offset, r.col_stride, r.n_t) for r in img.regions] == [(r.offset, r.col_stride, r.n_t) for r in pad.regions]
    a, b = np.zeros(img.nbytes, np.uint8), np.zeros(pad.nbytes, np.uint8)
    img.pack(a)
    pad.pack(b)
    assert np.array_equal(a, b)
