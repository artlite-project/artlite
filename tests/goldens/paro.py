"""A PARO checkpoint's reference (``artlite.paro``): its pseudo-quantized model, an F32 GGUF whose every PARO linear is
the plain linear it computes, W_eff = W M^T (W = s (q - z), M the exact fp64 matrix of its input transform), every other
tensor as the virtual GGUF has it (``artlite.hf_model``, whose metadata and names llama.cpp's converter agrees with:
tests/test_hf_model.py), so that llama.cpp runs the PARO model exactly (``goldens.text.golden`` on the file). Written
tensor by tensor (an 8B's F32 file is 32 GB: large models get F16 tensors, or a slice of their first layers as a model
of that many: a 27B's F16 file would be 54 GB) into ``tests/refs/<repo>/`` (gitignored)."""

import dataclasses
import os

import gguf
import numpy as np

from artlite.hf_model import HfModel
from artlite.paro import ParoTensor

from .. import common


def pseudo_gguf(path: str, *, f16: bool = False, layers: int = 0) -> str:
    """The pseudo-quantized GGUF of a PARO checkpoint directory (``f16``: F16 tensors; ``layers``: its first layers
    only, a model of that many layers without an MTP block), made on first use."""
    hf = HfModel(path)
    ident = [(n, os.path.getsize(f), os.path.getmtime(f)) for n, (f, _) in enumerate(hf.files)]
    key = (os.path.basename(os.path.normpath(path)), ident, f16) + ((layers,) if layers else ())
    out = common.ref_entry(path, "pseudo", key, common.source_version([__file__]), ".gguf")
    if common.hit(out):
        return out
    common.computing()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp"
    w = gguf.GGUFWriter(tmp, hf.arch)
    hf.adapter.write_metadata(w)
    names = sorted(hf.tensor_info)
    if layers:  # (the slice: its block count, no MTP block, the other layers' tensors left out)
        kv = w.kv_data[0]
        for k, v in ((f"{hf.arch}.block_count", layers), (f"{hf.arch}.nextn_predict_layers", 0)):
            if k in kv:
                kv[k] = dataclasses.replace(kv[k], value=v)
        names = [n for n in names if not n.startswith("blk.") or int(n.split(".")[1]) < layers]
    dt = np.float16 if f16 else np.float32
    shapes = {n: tuple(reversed(hf.tensor_info[n][0])) for n in names}
    for n in names:
        w.add_tensor_info(n, shapes[n], np.dtype(dt if len(shapes[n]) > 1 else np.float32),
                          np.dtype(dt if len(shapes[n]) > 1 else np.float32).itemsize * int(np.prod(shapes[n])))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for n in names:
        t = hf.tensor(n)
        a = t.effective() if isinstance(t, ParoTensor) else t.dequantize()
        w.write_tensor_data(np.ascontiguousarray(a, dt if len(shapes[n]) > 1 else np.float32).reshape(shapes[n]))
    w.close()
    os.replace(tmp, out)
    common.stored(out)
    return out

