"""A PARO checkpoint's reference (``artlite.paro``): its pseudo-quantized model, an F32 GGUF whose every PARO linear is
the plain linear it computes, W_eff = W M^T (W = s (q - z), M the exact fp64 matrix of its input transform), every other
tensor as the virtual GGUF has it (``artlite.hf_model``, whose metadata and names llama.cpp's converter agrees with:
tests/test_hf_model.py), so that llama.cpp runs the PARO model exactly (``goldens.text.golden`` on the file; Gemma 4:
``tests/gemma4/cpu.py``). Written tensor by tensor (an 8B's F32 file is 32 GB: large models get F16 tensors, or a slice
of their first layers as a model of that many, Gemma's per-layer tables cut to them and its layers' KV sharing to the
slice's: a 27B's F16 file would be 54 GB) into ``tests/refs/<repo>/`` (gitignored)."""

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
    if layers:  # (the slice: its block count, its per-layer arrays cut, no MTP block, the other layers' tensors left out)
        kv = w.kv_data[0]
        n = kv[f"{hf.arch}.block_count"].value
        for k, v in list(kv.items()):
            if v.type == gguf.GGUFValueType.ARRAY and len(v.value) == n:
                kv[k] = dataclasses.replace(v, value=v.value[:layers])
        shared = f"{hf.arch}.attention.shared_kv_layers"  # (Gemma 4: the layers that share a cache, at the end)
        own = n - kv[shared].value if shared in kv else n
        for k, v in ((f"{hf.arch}.block_count", layers), (f"{hf.arch}.nextn_predict_layers", 0),
                     (shared, max(0, layers - own))):
            if k in kv:
                kv[k] = dataclasses.replace(kv[k], value=v)
        names = [n for n in names if not n.startswith("blk.") or int(n.split(".")[1]) < layers]
    shapes = {n: tuple(reversed(hf.tensor_info[n][0])) for n in names}
    # (Gemma 4's per-layer tables: the slice's layers' rows of 256 each)
    npl = int(hf.meta.get(f"{hf.arch}.embedding_length_per_layer_input", 0))
    cut = {"per_layer_token_embd.weight": 1, "per_layer_model_proj.weight": 0} if layers and npl else {}
    for n, axis in cut.items():
        shapes[n] = tuple(layers * npl if i == axis else d for i, d in enumerate(shapes[n]))
    # (f16: the matrices llama.cpp multiplies, the PARO linears' and the embeddings' and head's; the rest F32, e.g. a GDN
    # layer's conv weights, which its ssm_conv takes in F32 only)
    half = ("token_embd.weight", "output.weight", "per_layer_token_embd.weight", "per_layer_model_proj.weight")
    dts = {n: np.dtype(np.float16 if f16 and (hf.tensor_info[n][1].startswith("P") or n in half) else np.float32)
           for n in names}
    for n in names:
        w.add_tensor_info(n, shapes[n], dts[n], dts[n].itemsize * int(np.prod(shapes[n])))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for n in names:
        t = hf.tensor(n)
        if n in cut:  # (the slice's rows of a per-layer table, read in place: a whole one is 11 GB of fp32)
            a = np.asarray(t.raw)[:, :shapes[n][1]] if cut[n] else np.asarray(t.raw)[:shapes[n][0]]
        else:
            a = t.effective() if isinstance(t, ParoTensor) else t.dequantize()
        w.write_tensor_data(np.ascontiguousarray(a, dts[n]).reshape(shapes[n]))
    w.close()
    os.replace(tmp, out)
    common.stored(out)
    return out

