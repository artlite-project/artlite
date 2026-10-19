"""The qwen35 prefill ELFs (operators/qwen35/prefill.py) on a slice of four layers in the middle of a model (three GDN
layers and a full-attention layer), fed llama.cpp's residual stream before the slice (prompt mode): a chunk of two
sub-chunks (the second mostly padding), then a chunk of one with padding tokens, against llama.cpp's residual after
the slice; then decode steps (the overlay) continuing from the state the prefill left (KV cache, GDN states, conv
states), against llama.cpp and against the same overlay running every position as a decode step."""

import os

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile

from . import qwen35_ref as ref
from .common import essay


@pytest.mark.parametrize("size", ["9B", "27B", "1B"])
def test_prefill_slice(size: str) -> None:
    from artlite.operators.qwen35.overlay import QwenOverlay
    from artlite.operators.qwen35.prefill import QwenPrefill

    path = ref.qwen_gguf("Q4_K_M", size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    g = GGUFFile(path)
    cfg = ref.QwenConfig.from_gguf(g)
    La = cfg.attention_layers[len(cfg.attention_layers) // 2]
    layers = list(range(La - 3, La + 1))
    n1, n2, K = 288, 45, 6  # (the chunks' valid tokens: n1 of 512, n2 of 256; then K decode steps)
    P = n1 + n2
    tokens = ref.llama_tokenize(path, essay("apple.txt"))[:P + K]
    G = ref.golden(path, tokens, layers=(layers[0] - 1, La), names="l_out", logits="none")
    x_in = np.asarray(G["l_out", layers[0] - 1], np.float32)
    want = np.asarray(G["l_out", La], np.float32)
    ov = QwenOverlay(g, layers=layers, vocab=512, max_positions=1024)
    pf = QwenPrefill(ov, sizes=(256, 512)).compile()
    ov.load(release=False)
    pf.bind()

    def x(t0: int, n: int) -> torch.Tensor:
        return torch.from_numpy(x_in[t0:t0 + n].copy())

    err = []
    for t0, n, T in [(0, n1, 512), (n1, n2, 256)]:
        pf.run(x(t0, n), t0, T)
        got = pf.hidden(0, n).numpy()
        err += [ref.rel_err(got[i], want[t0 + i]) for i in range(n)]
    cont = []
    for t in range(P, P + K):
        ov.step(x(t, 1)[0], t, logits=False)
        cont.append(ref.rel_err(ov.hidden(-1).numpy(), want[t]))
    ov.reset()
    dec = []
    for t in range(P + K):
        ov.step(x(t, 1)[0], t, logits=False)
        if t >= P:
            dec.append(ref.rel_err(ov.hidden(-1).numpy(), want[t]))
    print(f"{size} layers {layers}: prefill error mean {np.mean(err):.4f} max {np.max(err):.4f} (sub-chunk 2 mean "
          f"{np.mean(err[256:n1]):.4f}, chunk 2 mean {np.mean(err[n1:]):.4f}); decode after the prefill "
          f"{np.mean(cont):.4f}, decode only {np.mean(dec):.4f}")
    # (the prefill GEMMs multiply bfp16 activations by the weights dequantized to bfp16: ~1.4% rms per projection vs
    # the decode GEMV's bf16 x; 9B: mean 1.0%, max 6.3% at token 0, an outlier dimension cancelling in layer 19; 1B:
    # llama.cpp's own noise is ~3x the 9B's, the decode path's error 1.5%, the prefill's 1.7%)
    assert np.max(err) < 8e-2 and np.mean(err) < max(1.5e-2, 1.3 * np.mean(dec))
    assert np.mean(cont) < 1.5 * np.mean(dec) + 2e-3
