"""Random quantized tensors for the tests of the tile formats and the packers."""

import numpy as np
import torch

from artlite.gguf_model import QTensor

# per GGUF block: (columns, bytes, the offsets of its fp16 fields)
BLOCK = {"Q4_0": (32, 18, (0,)), "Q4_1": (32, 20, (0, 2)), "Q8_0": (32, 34, (0,)), "Q4_K": (256, 144, (0, 2)),
         "Q5_K": (256, 176, (0, 2)), "Q6_K": (256, 210, (208,)), "Q2_K": (256, 84, (80, 82)), "Q3_K": (256, 110, (108,)),
         "IQ1_S": (256, 50, (0,)), "IQ1_M": (256, 56, ()), "IQ2_XXS": (256, 66, (0,)), "IQ2_XS": (256, 74, (0,)),
         "IQ2_S": (256, 82, (0,)), "IQ3_XXS": (256, 98, (0,)), "IQ3_S": (256, 110, (0,)), "IQ4_NL": (32, 18, (0,)), "IQ4_XS": (256, 136, (0,))}


def random_tensor(qtype: str, rows: int, cols: int, seed: int) -> QTensor:
    """Random GGUF blocks (fp16 scales of a plausible range), or random BF16 / F16 values; "Q4_1_64": a Q4_1 tensor
    whose blocks come in pairs of equal (d, m) exact in bf16 (``formats.q4``'s group-64 Q4_1)."""
    rng = np.random.default_rng(seed)
    if qtype == "Q4_1_64":
        t = random_tensor("Q4_1", rows, cols, seed)
        b = t.raw.reshape(rows, cols // 64, 2, 20).copy()
        dm = b[:, :, 0, :4].copy().view(np.float16)  # (bf16-exact: the fp16 values rounded to 8 significant bits)
        dm = torch.from_numpy(dm.astype(np.float32)).bfloat16().float().numpy().astype(np.float16)
        b[:, :, :, :4] = dm.view(np.uint8)[:, :, None, :]
        return QTensor(name="t", qtype="Q4_1", rows=rows, cols=cols, raw=b.reshape(rows, -1))
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
    if qtype == "IQ1_M":  # (d's fp16 in the top nibbles of the 4 scale words, word m's at bits 4 m)
        d = (rng.standard_normal((rows, n)) * 0.01).astype(np.float16).view(np.uint16)
        sw = raw[..., 48:56].copy().view("<u2") & 0x0FFF
        sw |= ((d[..., None] >> (4 * np.arange(4, dtype=np.uint16))) & 0xF) << 12
        raw[..., 48:56] = sw.view(np.uint8)
    return QTensor(name="t", qtype=qtype, rows=rows, cols=cols, raw=raw.reshape(rows, -1))
