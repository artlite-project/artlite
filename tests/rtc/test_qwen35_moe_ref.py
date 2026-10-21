"""The MoE oracles of tests/rtc/qwen35_ref.py against llama.cpp's own tensors of Qwen3.6-35B-A3B (GGUF ``qwen35moe``;
no NPU): per layer (a GDN and an attention layer) the router's logits from the F32 rows, its top 8 and weights, the
routed experts' sum and the gated shared expert from llama.cpp's FFN input, and the layer's residual. llama.cpp
rounds the experts' activations to Q8_K, so the experts' oracles (fp32) are off by that (~1%); exact to ~2e-7 once
fed llama.cpp's rounded activations."""

import os

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile

from . import qwen35_ref as ref
from .common import essay

MODEL = ref.qwen_gguf("Q4_K_M", "35B-A3B")
NAMES = ("attn_residual|attn_post_norm|ffn_moe_logits|ffn_moe_topk|ffn_moe_weights_norm|ffn_moe_out|"
         "shared_expert_gate|ffn_shexp_gated|l_out")
LAYERS = (0, 3)


@pytest.fixture(scope="module")
def gold() -> tuple[GGUFFile, ref.Golden]:
    if not os.path.exists(MODEL):
        pytest.skip(f"{MODEL} missing")
    tokens = ref.llama_tokenize(MODEL, essay("apple.txt"))[:32]
    return GGUFFile(MODEL), ref.golden(MODEL, tokens, layers=LAYERS, names=NAMES, logits="none")


@pytest.mark.parametrize("L", LAYERS)
def test_moe_block(gold: tuple[GGUFFile, ref.Golden], L: int) -> None:
    g, G = gold
    b = f"blk.{L}."
    x = np.asarray(G["attn_post_norm", L], np.float32)
    wr = g.tensor(b + "ffn_gate_inp.weight").dequantize().numpy()
    assert ref.rel_err(x @ wr.T, G["ffn_moe_logits", L]) < 2e-5
    ids, w = ref.moe_route(G["ffn_moe_logits", L], int(g.arch_meta("expert_used_count")))
    assert np.array_equal(ids, G["ffn_moe_topk", L])
    assert ref.rel_err(w, G["ffn_moe_weights_norm", L]) < 2e-5
    moe = ref.moe_ffn(x, ids, w, ref.experts(g, L))
    shared = ref.shared_expert(x, *(g.tensor(b + f"ffn_{n}_shexp.weight").dequantize().numpy() for n in ("gate", "up", "down")),
                               g.tensor(b + "ffn_gate_inp_shexp.weight").dequantize().numpy()[0])
    assert ref.rel_err(moe, G["ffn_moe_out", L]) < 3e-2
    assert ref.rel_err(shared, G["ffn_shexp_gated", L]) < 3e-2
    out = np.asarray(G["attn_residual", L]) + G["ffn_moe_out", L] + G["ffn_shexp_gated", L]
    assert ref.rel_err(out, G["l_out", L]) < 1e-6
