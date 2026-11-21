"""GGUF files on the host (``artlite.gguf_model``, ``artlite.model_config``): the IQ formats' fields and weights against
gguf-py, the native row dequantization against ``QTensor.dequantize``, the drafter files' names."""

from pathlib import Path

import gguf
import numpy as np
import pytest
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS

from artlite import _native
from artlite.formats import FORMATS
from artlite.gguf_model import IQ4_VALUES, GGUFFile
from artlite.model_config import qwen_mtp_of

from .tensors import random_tensor


def test_iq3xxs_dequantize() -> None:
    """IQ3_XXS weights bit-identical to gguf-py's on random blocks (every index, sign field and scale), its fields
    consistent: the codes on the grid with the signs' even parity per 8 values."""
    t = random_tensor("IQ3_XXS", 64, 2048, 0)
    w = t.dequantize()
    ref = IQ3_XXS.dequantize_rows(np.ascontiguousarray(t.raw)).reshape(t.rows, t.cols).astype(np.float32)
    assert np.array_equal(w.view(np.uint32), ref.view(np.uint32))
    f = t.fields()
    q = f["q"].astype(np.int32)
    assert set(np.unique(np.abs(q))) <= {4, 12, 20, 28, 36, 44, 52, 62}
    assert ((q < 0).reshape(t.rows, -1, 8).sum(-1) % 2 == 0).all()


@pytest.mark.parametrize("qtype", ["IQ2_XXS", "IQ2_XS", "IQ2_S"])
def test_iq2_dequantize(qtype: str) -> None:
    """IQ2_XXS, IQ2_XS and IQ2_S weights bit-identical to gguf-py's on random blocks (every index, sign field and scale),
    their fields consistent: the codes on the grid ({8, 25, 43}) times the scales, XXS and XS signs of even parity per 8
    values."""
    t = random_tensor(qtype, 64, 2048, 0)
    w = t.dequantize()
    q = {"IQ2_XXS": IQ2_XXS, "IQ2_XS": IQ2_XS, "IQ2_S": IQ2_S}[qtype]
    ref = q.dequantize_rows(np.ascontiguousarray(t.raw)).reshape(t.rows, t.cols).astype(np.float32)
    assert np.array_equal(w.view(np.uint32), ref.view(np.uint32))
    f = t.fields()
    c = f["q"].astype(np.int32)
    assert set(np.unique(np.abs(c))) == {8, 25, 43}
    assert f["idx"].max() == {"IQ2_XXS": 255, "IQ2_XS": 511, "IQ2_S": 1023}[qtype]
    if qtype != "IQ2_S":
        assert ((c < 0).reshape(t.rows, -1, 8).sum(-1) % 2 == 0).all()
    s = f["ls"] if qtype == "IQ2_XXS" else f["sc"]
    n = 32 if qtype == "IQ2_XXS" else 16
    d = np.repeat(f["d"], 256 // n, -1) * (2 * s.astype(np.float32) + 1) / 8
    assert np.array_equal(w, (d[..., None] * c.reshape(t.rows, -1, n)).reshape(w.shape))


def _iq1m_ggml(raw: np.ndarray) -> np.ndarray:
    """ggml's dequantize_row_iq1_m (ggml-quants.c) on (rows, 56 n) blocks, line by line (gguf-py's IQ1_M reads every
    delta as negative: ``qh & 0x08 == 0`` is ``qh & False``)."""
    IQ1_S.init_grid()
    grid = IQ1_S.grid.reshape(2048, 8).astype(np.float32)
    out = []
    for blk in raw.reshape(-1, 56):
        qs, qh, sc = blk[0:32].astype(np.int64), blk[32:48], blk[48:56].view("<u2").astype(np.int64)
        d = np.uint16((sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000)).view(np.float16)
        for ib in range(8):
            dl1 = np.float32(d) * np.float32(2 * ((sc[ib // 2] >> (6 * (ib % 2))) & 7) + 1)
            dl2 = np.float32(d) * np.float32(2 * ((sc[ib // 2] >> (6 * (ib % 2) + 3)) & 7) + 1)
            h0, h1 = int(qh[2 * ib]), int(qh[2 * ib + 1])
            idx = [qs[4 * ib] | ((h0 << 8) & 0x700), qs[4 * ib + 1] | ((h0 << 4) & 0x700),
                   qs[4 * ib + 2] | ((h1 << 8) & 0x700), qs[4 * ib + 3] | ((h1 << 4) & 0x700)]
            delta = [np.float32(-0.125 if h & m else 0.125) for h, m in ((h0, 8), (h0, 128), (h1, 8), (h1, 128))]
            for la in range(4):
                out.append((dl1 if la < 2 else dl2) * (grid[idx[la]] + delta[la]))
    return np.concatenate(out).reshape(raw.shape[0], -1)


@pytest.mark.parametrize("qtype", ["IQ1_S", "IQ1_M"])
def test_iq1_dequantize(qtype: str) -> None:
    """IQ1_S and IQ1_M weights bit-identical to gguf-py's (IQ1_M: ggml's C, ``_iq1m_ggml``) on random blocks (every
    index, scale and delta sign), their fields consistent: the codes 8 g + delta on the grid ({-1, 0, 1}) and the deltas'
    signs, times the scales."""
    t = random_tensor(qtype, 64 if qtype == "IQ1_S" else 8, 2048, 0)
    w = t.dequantize()
    raw = np.ascontiguousarray(t.raw)
    ref = (IQ1_S.dequantize_rows(raw) if qtype == "IQ1_S" else _iq1m_ggml(raw)).reshape(t.rows, t.cols).astype(np.float32)
    assert np.array_equal(w.view(np.uint32), ref.view(np.uint32))
    f = t.fields()
    c = f["q"].astype(np.int32)
    assert set(np.unique(c)) == {-9, -7, -1, 1, 7, 9} and f["idx"].max() == 2047 and f["sc"].max() == 7
    n = 32 if qtype == "IQ1_S" else 8  # (k per delta)
    assert np.array_equal(np.repeat(f["dneg"], n, -1) != 0, (c % 8) == 7)  # (delta -1: q = 8 g - 1)
    n = 32 if qtype == "IQ1_S" else 16  # (k per scale)
    d = np.repeat(f["d"], 256 // n, -1) * (2 * f["sc"].astype(np.float32) + 1) / 8
    assert np.array_equal(w, (d[..., None] * c.reshape(t.rows, -1, n)).reshape(w.shape))


@pytest.mark.parametrize("qtype", ["IQ3_S", "IQ4_NL", "IQ4_XS"])
def test_iq_dequantize(qtype: str) -> None:
    """IQ3_S, IQ4_NL and IQ4_XS weights bit-identical to gguf-py's on random blocks (every index, sign, scale and
    code), their fields consistent: the codes on the grid or the 16 values times the scales."""
    t = random_tensor(qtype, 64, 2048, 0)
    w = t.dequantize()
    q = {"IQ3_S": IQ3_S, "IQ4_NL": IQ4_NL, "IQ4_XS": IQ4_XS}[qtype]
    ref = q.dequantize_rows(np.ascontiguousarray(t.raw)).reshape(t.rows, t.cols).astype(np.float32)
    assert np.array_equal(w.view(np.uint32), ref.view(np.uint32))
    f = t.fields()
    if qtype == "IQ3_S":
        assert set(np.unique(np.abs(f["q"].astype(np.int32)))) <= set(range(1, 16, 2))
        assert f["idx"].max() > 255 and f["sc"].max() <= 15
        d = np.repeat(f["d"], 8, -1) * (1 + 2 * f["sc"].astype(np.float32))
        assert np.array_equal(w, (d[..., None] * f["q"].reshape(t.rows, -1, 32)).reshape(w.shape))
    else:
        assert f["q"].max() <= 15 and (qtype == "IQ4_NL" or f["ls"].max() <= 63)
        s = f["d"] if qtype == "IQ4_NL" else np.repeat(f["d"], 8, -1) * (f["ls"].astype(np.float32) - 32)
        assert np.array_equal(w, (s[..., None] * IQ4_VALUES[f["q"]].reshape(t.rows, -1, 32)).reshape(w.shape))


@pytest.mark.parametrize("qtype", ["BF16", "F16", "Q4_0", "Q4_1", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL",
                                   "IQ4_XS"])
def test_dequant_rows(qtype: str) -> None:
    """The native rows (``_native.dequant_rows``: an embedding lookup) bit-identical to ``QTensor.dequantize``."""
    assert _native.dequant_supported(qtype)
    t = random_tensor(qtype, 33, 1024, 1)
    out = np.empty((t.rows, t.cols), np.float32)
    _native.dequant_rows(qtype, np.ascontiguousarray(t.raw).view(np.uint8), out)
    assert np.array_equal(out.view(np.uint32), t.dequantize().view(np.uint32))


def test_qwen_mtp_of(tmp_path: object) -> None:
    """A drafter next to the model (``mtp-*`` or an uploader's ``<name>-mtp-*``), else in another organization's
    repository of the same name, by ``MTP_QUANTS``."""
    repo = tmp_path / "org" / "Model-GGUF"
    repo.mkdir(parents=True)
    model = repo / "Model-Q4_K_M.gguf"
    model.touch()
    assert qwen_mtp_of(str(model)) is None
    other = tmp_path / "ggml-org" / "Model-GGUF"
    other.mkdir(parents=True)
    (other / "mtp-Model-Q8_0.gguf").touch()
    (other / "mtp-Model-Q4_0.gguf").touch()
    assert qwen_mtp_of(str(model)) == str(other / "mtp-Model-Q4_0.gguf")
    (repo / "model-1.0-mtp-Q8_0.gguf").touch()
    assert qwen_mtp_of(str(model)) == str(repo / "model-1.0-mtp-Q8_0.gguf")
    (repo / "model-1.0-mtp-Q4_K_M.gguf").touch()
    assert qwen_mtp_of(str(model)) == str(repo / "model-1.0-mtp-Q4_K_M.gguf")


def _write_gguf(path: Path, rows: int, cols: int) -> np.ndarray:
    """A GGUF file of two metadata arrays and the Q4_0 tensor ``w.weight`` (rows, cols); its raw rows."""
    w = gguf.GGUFWriter(str(path), "test")
    w.add_array("test.scores", [1, 2, 3, 4])
    w.add_array("tokenizer.ggml.tokens", ["<a>", "<b>"])
    raw = random_tensor("Q4_0", rows, cols, 7).raw
    w.add_tensor("w.weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return raw


def _file_pages() -> int | None:
    """Bytes of file mappings resident in this process (Linux: RssFile), else None."""
    try:
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) << 10 for line in f if line.startswith("RssFile:"))
    except (OSError, StopIteration):
        return None


def test_pack_reads_own_views(tmp_path: Path) -> None:
    """``_native.pack_tiles`` reads a file's tensors through views of its own: the tiles are those of the tensor's
    copy in memory, the process holds no more of the file after it than before (the caller's mapping untouched, the
    views unmapped), and a tensor outside the files' mappings is read as the memory it is."""
    rows, cols, R, nsb = 4096, 4096, 16, 4  # (9.2 MB of Q4_0 rows)
    path = tmp_path / "w.gguf"
    raw = _write_gguf(path, rows, cols)
    g = GGUFFile(str(path))
    S = -(-FORMATS["Q4_0"].used_bytes(R, nsb) // 64) * 64
    n_kc, n_rt = cols // (256 * nsb), rows // R
    index = np.arange(rows, dtype=np.int64).reshape(1, rows)

    def block(tensor: np.ndarray) -> tuple:
        return ("Q4_0", R, S, nsb, n_kc, n_kc, n_rt, 0, 0, n_rt * n_kc * S, [tensor], index, cols // 256, 1)

    want, got = np.zeros(n_rt * n_kc * S, np.uint8), np.zeros(n_rt * n_kc * S, np.uint8)
    _native.pack_tiles(want, [block(np.array(raw))], threads=2)
    before = _file_pages()
    _native.pack_tiles(got, [block(g.tensor("w.weight").raw)], threads=2, files=g.files)
    after = _file_pages()
    assert want.any() and np.array_equal(got, want)
    if before is not None:
        assert after - before < 1 << 20, (before, after)
    got[:] = 0
    _native.pack_tiles(got, [block(np.array(raw))], threads=1, files=g.files)
    assert np.array_equal(got, want)


def test_gguf_page_hints(tmp_path: Path) -> None:
    """``GGUFFile``'s page hints leave the file readable: the metadata and a tensor's rows after ``release``, after
    ``willneed`` and ``populate`` of the tensor, and after a row prefetch."""
    path = tmp_path / "mini.gguf"
    raw = _write_gguf(path, 64, 256)
    g = GGUFFile(str(path))
    scores = np.array(g.meta["test.scores"])
    for hint in (g.release, lambda: g.willneed("w.weight"), lambda: g.populate("w.weight"), g.release):
        hint()
        assert np.array_equal(g.meta["test.scores"], scores)
        assert g.strings("tokenizer.ggml.tokens") == ["<a>", "<b>"]
        assert np.array_equal(g.tensor("w.weight").raw, raw)
    want = random_tensor("Q4_0", 64, 256, 7).dequantize()
    assert np.array_equal(g.rows("w.weight", [5, 3, 5, 60]), want[[5, 3, 5, 60]])
    with pytest.raises(_native.Error, match="outside the mapping"):
        _native.page_hint(g.data, "willneed", np.array([g.data.size], np.int64), 4096)
