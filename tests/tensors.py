"""Random quantized tensors for the tests of the tile formats and the packers."""

import numpy as np
import torch

from artlite.gguf_model import QTensor

# per GGUF block: (columns, bytes, the offsets of its fp16 fields)
BLOCK = {"Q4_0": (32, 18, (0,)), "Q4_1": (32, 20, (0, 2)), "Q8_0": (32, 34, (0,)), "Q4_K": (256, 144, (0, 2)),
         "Q5_K": (256, 176, (0, 2)), "Q6_K": (256, 210, (208,)), "Q2_K": (256, 84, (80, 82)), "Q3_K": (256, 110, (108,)),
         "IQ3_XXS": (256, 98, (0,)), "IQ3_S": (256, 110, (0,)), "IQ4_NL": (32, 18, (0,)), "IQ4_XS": (256, 136, (0,))}


def random_tensor(qtype: str, rows: int, cols: int, seed: int) -> QTensor:
    """Random GGUF blocks (fp16 scales of a plausible range), or random BF16 / F16 values."""
    rng = np.random.default_rng(seed)
    if qtype in ("BF16", "F16"):
        w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
        raw = w.astype(np.float16) if qtype == "F16" else torch.from_numpy(w).bfloat16().view(torch.int16).numpy()
        return QTensor(name="t", qtype=qtype, rows=rows, cols=cols, raw=raw.view(np.uint16 if qtype == "BF16" else np.float16))
    per, nb, f16 = BLOCK[qtype]
    n = cols // per
    raw = rng.integers(0, 256, (rows, n, nb), dtype=np.uint8)
    for o in f16:
        d = (rng.standard_normal((rows, n)) * 0.01).astype(np.float16)
        raw[..., o:o + 2] = d.view(np.uint8).reshape(rows, n, 2)
    return QTensor(name="t", qtype=qtype, rows=rows, cols=cols, raw=raw.reshape(rows, -1))
