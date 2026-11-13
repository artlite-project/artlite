"""The qwen35 prefill ELFs (models/qwen35/prefill.py) on a slice of four layers in the middle of a model (three GDN
layers and a full-attention layer), fed llama.cpp's residual stream before the slice (prompt mode): a chunk of two
sub-chunks (the second mostly padding), then chunks of the two ``SMALL`` sizes (the last with padding tokens), against
llama.cpp's residual after the slice; then decode steps (the overlay) continuing from the state the prefill left (KV cache, GDN states, conv
states), against llama.cpp and against the same overlay running every position as a decode step. Occamy 1.0 APEX: its
IQ3_XXS experts (layers 20 .. 23) and its Q3_K ones (layers 4 .. 7)."""


import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig

from ..common import essay
from ..goldens.text import golden
from ..llama_cpp import llama_tokenize
from ..model_files import require, size_gguf
from ..numerics import rel_max


@pytest.mark.parametrize(("size", "La"), [("9B", 0), ("27B", 0), ("1B", 0), ("35B-A3B", 0), ("occamy", 0), ("occamy", 7),
                                         ("kev", 0)],
                         ids=["9B", "27B", "1B", "35B-A3B", "occamy", "occamy-7", "kev"])
def test_prefill_slice(size: str, La: int) -> None:
    """(La: the slice's attention layer, its last; 0: the one in the middle)"""
    from models.qwen35.overlay import QwenOverlay
    from models.qwen35.prefill import QwenPrefill

    path = size_gguf(size)
    require(path)
    g = GGUFFile(path)
    cfg = QwenConfig.from_gguf(g)
    La = La or cfg.attention_layers[len(cfg.attention_layers) // 2]
    layers = list(range(La - 3, La + 1))
    n1, n2, n3, K = 288, 32, 45, 6  # (the chunks' valid tokens: n1 of 512, n2 of 64, n3 of 128; then K decode steps)
    P = n1 + n2 + n3
    tokens = llama_tokenize(path, essay("apple.txt"))[:P + K]
    moe = bool(g.arch_meta("expert_count"))
    G = golden(path, tokens, layers=(layers[0] - 1, *layers) if moe else (layers[0] - 1, La),
                   names="l_out|ffn_moe_logits" if moe else "l_out", logits="none")
    x_in = np.asarray(G["l_out", layers[0] - 1], np.float32)
    want = np.asarray(G["l_out", La], np.float32)
    ov = QwenOverlay(g, layers=layers, vocab=512, max_positions=1024)
    pf = QwenPrefill(ov, sizes=(64, 128, 512)).compile()
    ov.load(release=False)
    pf.bind()

    def x(t0: int, n: int) -> np.ndarray:
        return x_in[t0:t0 + n]

    err = []
    for t0, n, T in [(0, n1, 512), (n1, n2, 64), (n1 + n2, n3, 128)]:
        pf.run(x(t0, n), t0, T)
        got = pf.hidden(0, n)
        err += [rel_max(got[i], want[t0 + i]) for i in range(n)]
    cont = []
    for t in range(P, P + K):
        ov.step(x(t, 1)[0], t, logits=False)
        cont.append(rel_max(ov.hidden(-1), want[t]))
    ov.reset()
    dec = []
    for t in range(P + K):
        ov.step(x(t, 1)[0], t, logits=False)
        if t >= P:
            dec.append(rel_max(ov.hidden(-1), want[t]))
    err = np.asarray(err)
    # (the prefill GEMMs multiply bfp16 activations by the weights dequantized to bfp16: ~1.4% rms per projection vs
    # the decode GEMV's bf16 x, which adds 0.2-0.6% to the decode path's mean error (9B 0.34 -> 0.79%, 27B 0.45 -> 0.62%,
    # 1B 1.26 -> 1.69%: llama.cpp's own noise is ~3x the 9B's, Kev 0.62 -> 1.25%); 9B: max 6.4%, an outlier dimension
    # cancelling in layer 19; 35B-A3B: mean 1.8%, the experts' two
    # GEMMs and routing flips: x ~1% off changes a few % of the top-8 sets per layer, and a token whose 8th and 9th
    # logits nearly tie in a layer of the slice may take another expert in either run (up to 14% at one such token;
    # the NPU's own prefill and decode paths up to 10% apart at one), so the largest error is over the tokens llama.cpp
    # routes clearly: the 8th / 9th gap above 0.01 in every layer (2 / 3 of them; max 7%))
    clear = np.ones(P, bool)
    if moe:
        clear = np.min([np.diff(np.sort(np.asarray(G["ffn_moe_logits", L], np.float32), 1)[:, -9:-7], axis=1)[:, 0]
                        for L in layers], 0)[:P] > 0.01
    print(f"{size} layers {layers}: prefill error mean {np.mean(err):.4f} max {np.max(err):.4f} (sub-chunk 2 mean "
          f"{np.mean(err[256:n1]):.4f}, chunks 2, 3 mean {np.mean(err[n1:n1 + n2]):.4f}, {np.mean(err[n1 + n2:]):.4f}); "
          f"decode after the prefill "
          f"{np.mean(cont):.4f}, decode only {np.mean(dec):.4f}; median {np.median(err):.4f}, p95 "
          f"{np.percentile(err, 95):.4f}, above 5%: {int((err > 5e-2).sum())} of {len(err)}"
          + (f"; clearly routed {int(clear.sum())}: max {np.max(err[clear]):.4f}" if moe else ""))
    if moe:
        assert np.max(err[clear]) < 0.12 and np.mean(err) < 2.4 * np.mean(dec)
    else:
        assert np.max(err) < 8e-2 and np.mean(err) < np.mean(dec) + 8e-3
    assert np.mean(cont) < 1.5 * np.mean(dec) + 2e-3
