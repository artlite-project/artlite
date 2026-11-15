"""GGUF files on the host (``artlite.gguf_model``, ``artlite.model_config``): the IQ formats' fields and weights against
gguf-py, the native row dequantization against ``QTensor.dequantize``, the drafter files' names."""

import numpy as np
import pytest
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS

from artlite import _native
from artlite.gguf_model import IQ4_VALUES
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
