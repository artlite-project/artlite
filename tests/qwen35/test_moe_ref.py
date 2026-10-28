"""The MoE oracles of tests/qwen35/ref.py against llama.cpp's own tensors of Qwen3.6-35B-A3B (GGUF ``qwen35moe``;
no NPU): per layer (a GDN and an attention layer) the router's logits from the F32 rows, its top 8 and weights, the
routed experts' sum and the gated shared expert from llama.cpp's FFN input, and the layer's residual. llama.cpp
rounds the experts' activations to Q8_K, so the experts' oracles (fp32) are off by that (~1%); exact to ~2e-7 once
fed llama.cpp's rounded activations. The MTP block (``ref.mtp_block``) against llama.cpp's run of the drafter
file over the same rows."""

import os

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import QwenConfig, qwen_mtp_of

from ..common import essay
from . import ref

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


def test_mtp_block() -> None:
    """The MTP block (``mtp_block``: fp32) against llama.cpp's on 64 rows of an essay (the model's h_nextn of each
    previous row, the drafter file's token embeddings): every tensor within ~2% rms of llama.cpp's (its matmuls round
    their activations to Q8_0: 0.7% at eh_proj, ~1.9% at the block's end), the argmax of the head's first 65536 rows
    llama.cpp's but where its logits nearly tie."""
    mtp = qwen_mtp_of(MODEL) if os.path.exists(MODEL) else None
    if mtp is None:
        pytest.skip("no Qwen3.6-35B-A3B model file or MTP drafter")
    n, V = 64, 65536
    tokens = ref.llama_tokenize(MODEL, essay("apple.txt"))[:n]
    G = ref.golden(MODEL, tokens, layers=(40, "g"), names="mtp_.*|h_nextn|result_norm", mtp=mtp)
    g = GGUFFile(mtp)
    emb = np.asarray(G.mtp("mtp_tok_embd", 40), np.float32).reshape(n, -1)
    h = np.asarray(G["result_norm"], np.float32).reshape(n, -1)
    r = ref.mtp_block(g, QwenConfig.from_gguf(g), emb, np.concatenate([np.zeros_like(h[:1]), h[:-1]]), np.arange(n),
                      head=np.arange(V))

    def rms(a: np.ndarray, b: np.ndarray) -> float:
        a, b = np.asarray(a, np.float64).reshape(n, -1), np.asarray(b, np.float64).reshape(n, -1)
        return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))

    for name, tol in (("mtp_eh_proj", 1e-2), ("mtp_attn_residual", 1.5e-2), ("mtp_ffn_out", 3e-2),
                      ("mtp_post_ffn", 2.5e-2)):
        assert rms(r[name], G.mtp(name, 40)) < tol, name
    assert rms(r["h_nextn"], G.mtp("h_nextn")) < 2.5e-2
    ml = np.asarray(G.mtp_logits, np.float32)[:, :V]
    top2 = np.sort(ml, -1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > 0.1
    assert (r["logits"].argmax(-1) == ml.argmax(-1))[clear].all()
