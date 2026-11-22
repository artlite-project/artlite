"""A Gemma 4 PARO checkpoint on the NPU (models/gemma4/paro_steps.py): z-lab's E4B PARO, a slice of its first layers
(sliding-window and full-attention layers) through ``GemmaOverlay.step`` in steps of one, three and four tokens, against
the fp32 model (gemma_cpu.GemmaCpu) of the slice's pseudo-quantized GGUF (``goldens.paro.pseudo_gguf``: the effective
weights deq(Q) M^T as F16): the logits of the first ``VOCAB`` rows (the screened head's exact ones) and the residual
after the slice, each token's the same bit for bit at every step size."""

import numpy as np
import torch

from artlite.gguf_model import GGUFFile
from artlite.hf_model import HfModel
from artlite.model_config import checkpoint
from models.gemma4.host import GemmaHost
from models.gemma4.overlay import GemmaOverlay
from models.gemma4.tokenizer import GemmaTokenizer

from ..common import cached_ref, essay
from ..goldens.paro import pseudo_gguf
from ..model_files import require
from ..numerics import rel_finite
from . import cpu as gemma_cpu

E4B = "gemma-4-E4B-paro"
LAYERS = 6  # (layers 0 .. 5: five sliding-window layers, one full)
VOCAB = 512
T = 12


def _ref(path: str, ids: list[int]) -> dict[str, torch.Tensor]:
    """The fp32 slice's raw logits of the first VOCAB rows and its residual per token (cached)."""
    ref = GGUFFile(pseudo_gguf(path, f16=True, layers=LAYERS))

    def run() -> dict[str, torch.Tensor]:
        cpu = gemma_cpu.GemmaCpu(ref)
        trace: dict[str, torch.Tensor] = {}
        h = cpu.hidden(ids, trace=trace)
        return {"logits": (h @ cpu.mat("token_embd.weight")[:VOCAB].T).float(), "r": trace[f"l_out.{LAYERS - 1}"]}

    return cached_ref(ref.path, "paro_slice", (ids, VOCAB), run, sources=gemma_cpu.SOURCES)


def test_paro_slice() -> None:
    """Steps of 1, 3 and 4 tokens (``reset`` between the runs): the logits and residual within 2% (bf16 rotated copies,
    the PARO tiles' bf16 scales against the fp32 model; measured 0.4-0.9%), and the same bits at every step size."""
    path = checkpoint(E4B)
    require(path)
    g = HfModel(path)
    ids = GemmaTokenizer(g).encode(essay("apple.txt"))[:T]
    want = _ref(path, ids)
    host = GemmaHost(g)
    x, e = host.embed(ids), host.layer_rows(ids)
    ov = GemmaOverlay(g, layers=list(range(LAYERS)), vocab=VOCAB, max_positions=256)
    ov.load()
    runs = []
    for batch in (1, 3, 4):
        ov.reset()
        logits, res = [], []
        for t0 in range(0, T, batch):
            n = min(batch, T - t0)
            logits.append(np.asarray(ov.step(x[t0:t0 + n], e[t0:t0 + n], t0), np.float32))
            res.append(ov.residuals(n))
        lg, r = np.concatenate(logits), np.concatenate(res)
        err = {"logits": max(rel_finite(torch.from_numpy(lg[t]), want["logits"][t]) for t in range(T)),
               "r": max(rel_finite(torch.from_numpy(r[t]), want["r"][t]) for t in range(T))}
        print(f"{batch} per dispatch:", {k: round(v, 4) for k, v in err.items()})
        assert all(v < 0.02 for v in err.values()), (batch, err)
        runs.append((lg, r))
    for lg, r in runs[1:]:
        assert np.array_equal(lg, runs[0][0]) and np.array_equal(r, runs[0][1])
