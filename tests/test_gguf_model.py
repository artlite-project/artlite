"""GGUF files on the host (``artlite.gguf_model``, ``artlite.model_config``): the IQ formats' fields and weights against
gguf-py, the native row dequantization against ``QTensor.dequantize``, the drafter files' names."""

import numpy as np
import pytest
from gguf.quants import IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS

from artlite import _native
from artlite.gguf_model import IQ4_VALUES
from artlite.model_config import qwen_mtp_of

from .decode.test_gemv import _random_tensor


def test_iq3xxs_dequantize() -> None:
    """IQ3_XXS weights bit-identical to gguf-py's on random blocks (every index, sign field and scale), its fields
    consistent: the codes on the grid with the signs' even parity per 8 values."""
    t = _random_tensor("IQ3_XXS", 64, 2048, 0)
    w = t.dequantize()
    ref = IQ3_XXS.dequantize_rows(np.ascontiguousarray(t.raw)).reshape(t.rows, t.cols).astype(np.float32)
    assert np.array_equal(w.view(np.uint32), ref.view(np.uint32))
    f = t.fields()
    q = f["q"].astype(np.int32)
    assert set(np.unique(np.abs(q))) <= {4, 12, 20, 28, 36, 44, 52, 62}
    assert ((q < 0).reshape(t.rows, -1, 8).sum(-1) % 2 == 0).all()


@pytest.mark.parametrize("qtype", ["IQ3_S", "IQ4_NL", "IQ4_XS"])
def test_iq_dequantize(qtype: str) -> None:
    """IQ3_S, IQ4_NL and IQ4_XS weights bit-identical to gguf-py's on random blocks (every index, sign, scale and
    code), their fields consistent: the codes on the grid or the 16 values times the scales."""
    t = _random_tensor(qtype, 64, 2048, 0)
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
    t = _random_tensor(qtype, 33, 1024, 1)
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
