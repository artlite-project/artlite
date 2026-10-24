"""GGUF files on the host (``artlite.gguf_model``, ``artlite.model_config``): IQ3_XXS's fields and weights against
gguf-py, the native row dequantization against ``QTensor.dequantize``, the drafter files' names."""

import numpy as np
import pytest
from gguf.quants import IQ3_XXS

from artlite import _native
from artlite.model_config import qwen_mtp_of

from .test_overlay_gemv import _random_tensor


def test_iq3xxs_dequantize() -> None:
    """IQ3_XXS weights bit-identical to gguf-py's on random blocks (every index, sign field and scale), its fields
    consistent: the codes on the grid with the signs' even parity per 8 values."""
    t = _random_tensor("IQ3_XXS", 64, 2048, 0)
    w = t.dequantize().numpy()
    ref = IQ3_XXS.dequantize_rows(np.ascontiguousarray(t.raw)).reshape(t.rows, t.cols).astype(np.float32)
    assert np.array_equal(w.view(np.uint32), ref.view(np.uint32))
    f = t.fields()
    q = f["q"].astype(np.int32)
    assert set(np.unique(np.abs(q))) <= {4, 12, 20, 28, 36, 44, 52, 62}
    assert ((q < 0).reshape(t.rows, -1, 8).sum(-1) % 2 == 0).all()


@pytest.mark.parametrize("qtype", ["Q4_0", "Q4_K", "Q6_K", "Q3_K"])
def test_dequant_rows(qtype: str) -> None:
    """The native rows (``_native.dequant_rows``: an embedding lookup) bit-identical to ``QTensor.dequantize``."""
    t = _random_tensor(qtype, 33, 1024, 1)
    out = np.empty((t.rows, t.cols), np.float32)
    _native.dequant_rows(qtype, np.ascontiguousarray(t.raw), out)
    assert np.array_equal(out.view(np.uint32), t.dequantize().numpy().view(np.uint32))


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
