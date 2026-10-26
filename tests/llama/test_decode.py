"""The Llama decode overlay (models/llama: ``LlamaOverlay``) on 2-layer slices of GGUF files with a truncated
vocabulary, against the overlay's arithmetic on the host (ref.LlamaNpuRef), in both K formats of the caches:

- a text from position 0 in steps of 1 .. 4 tokens (record pairs, records and cache groups crossed at every alignment;
  a step of S slots gives each the logits of a step of one): logits, attention outputs, the LM head's input, then the
  caches and the open record's ring;
- steps at positions 960 .. 1039 after a cache filled with random K, V (record, group and object boundaries deep in
  the cache), against the reference attending to the cache as stored.

Files: 1B Q4_K_M (head_dim 64, 4 query heads per KV head; layer 0's V / down Q6_K), 1B Q5_K_M (13-row Q5_K tiles), 1B
Q6_K (12-row tiles of 10112 B), 3B Q4_K_M (head_dim 128, 3 query heads per KV head), 8B Q4_K_S (hidden 4096, Q4_K,
Q5_K and Q6_K tiles; down in 7 K groups), 1B BF16 (8-row BFP16 tiles of 9216 B), 1B Q4_0 (Q4_0 of 16 rows, layer 0's
down Q4_1 of 14), 1B Q4_1 (16 rows, 10240 B).
"""

import os

import numpy as np
import pytest
import torch

from artlite.decode import kv
from artlite.decode.vring import ring_rows
from artlite.gguf_model import GGUFFile
from artlite.operators.design import COLS
from models.llama.overlay import KV_K, LlamaOverlay

from ..common import essay, llama_gguf
from . import ref

VOCAB = 2048
# (GGUF quantization, model size, layers)
CASES = {"1B_Q4_K_M": ("Q4_K_M", "1B", [0, 2]), "1B_Q5_K_M": ("Q5_K_M", "1B", [0, 3]),
         "1B_Q6_K": ("Q6_K", "1B", [0, 1]), "3B_Q4_K_M": ("Q4_K_M", "3B", [0, 3]),
         "8B_Q4_K_S": ("Q4_K_S", "8B", [0, 4]), "1B_BF16": ("BF16", "1B", [0, 1]), "1B_Q4_0": ("Q4_0", "1B", [0, 2]),
         "1B_Q4_1": ("Q4_1", "1B", [0, 2]), "3B_Q2_K": ("Q2_K", "3B", [0, 3]), "3B_Q3_K_M": ("Q3_K_M", "3B", [0, 3]),
         "1B_Q3_K_L": ("Q3_K_L", "1B", [0, 2])}
STEPS = (1, 4, 3, 2, 4, 4, 1, 3, 3, 2, 1, 4)  # (step sizes, repeated: every alignment to the records and their pairs)
# relative rms error: the reference leaves out the GEMV's panels (14 bits) and its 32-k sums rounded to bf16, which move
# a projection's rows by ~1% (the NPU's o projection is bit-exact to ``mmv.ref_rows``); up to 4.2% here
TOL = 0.05


def _gguf(quant: str, size: str) -> GGUFFile:
    path = llama_gguf(quant, size)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    return GGUFFile(path)


def _tokens(n: int) -> list[int]:
    from models.llama.tokenizer import load_tokenizer

    return load_tokenizer(llama_gguf("Q4_K_M", "1B")).encode(essay("addiction.txt"))[:n]


@pytest.mark.parametrize("kv_k", list(KV_K))
@pytest.mark.parametrize("case", list(CASES))
def test_llama_decode_steps(case: str, kv_k: str) -> None:
    quant, size, layers = CASES[case]
    g = _gguf(quant, size)
    ids = _tokens(136)  # (crosses a cache group: 64 positions at head_dim 128, 128 at 64)
    model = ref.LlamaNpuRef(g, layers, VOCAB, kp=KV_K[kv_k])
    trace: dict[str, torch.Tensor] = {}
    want = model.run(ids, 0, trace)
    x = model.embed(ids)
    ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=256, kv_k=kv_k)
    ov.load()
    worst: dict[str, float] = {}

    def keep(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
        worst[name] = max(worst.get(name, 0.0), ref.relative(got, want))

    p, i = 0, 0
    while p < len(ids):
        n = min(STEPS[i % len(STEPS)], len(ids) - p)
        i += 1
        logits = ov.step(x[p:p + n], p).float()
        for s in range(n):
            keep("logits", logits[s], want[p + s])
            keep("lm_in", ov.hidden(-1, s), trace["r"][p + s])
            for L in layers:
                keep(f"attn.{L}", ov.attn_out(L, s), trace[f"attn.{L}"][p + s])
        p += n
    r0 = (len(ids) - 1) // 8 * 8  # (the last record: its values in the ring)
    ov.dev.fetch("KV")
    for L, cache in model.caches.items():  # the caches as the last token left them
        k, v = ov.read_kv(L, range(len(ids)))
        keep(f"K.{L}", k, cache.K)
        keep(f"V.{L}", v, cache.V)
        rows = [ring_rows(ov.dev.view("KV")[ov.ring(L, c) + (r0 // 8 & 1) * ov.T:][:ov.T], ov.hd) for c in range(COLS)]
        keep(f"ring.{L}", torch.from_numpy(np.stack(rows)[:, :len(ids) - r0]), cache.rows[:, r0:])
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < TOL for v in worst.values()), worst


@pytest.mark.parametrize("kv_k", list(KV_K))
@pytest.mark.parametrize("case", ["1B_Q4_K_M", "3B_Q4_K_M", "8B_Q4_K_S"])
def test_llama_decode_deep(case: str, kv_k: str) -> None:
    """Steps at 960 .. 1039 (one token each, and steps of 4 from 1008) after a cache of 960 random positions."""
    quant, size, layers = CASES[case]
    g = _gguf(quant, size)
    P0, N = 960, 80
    model = ref.LlamaNpuRef(g, layers, VOCAB, kp=KV_K[kv_k])
    ov = LlamaOverlay(g, layers=layers, vocab=VOCAB, max_positions=1152, kv_k=kv_k)
    ov.load()
    gen = torch.Generator().manual_seed(7)
    nkv, hd = ov.cfg.num_key_value_heads, ov.hd
    kvb = ov.dev.view("KV")
    for L in layers:  # (K, V as a model's: K of rotated rows, larger than V)
        K = (torch.randn(nkv, P0, hd, generator=gen) * 2).bfloat16().float()
        V = torch.randn(nkv, P0, hd, generator=gen).bfloat16().float()
        pad = (0, -P0 % ov.pg)  # (whole groups; the steps write the positions from P0 on)
        for c in range(nkv):
            grp = kv.kv_groups(np.pad(K[c].numpy(), (pad, (0, 0))), np.pad(V[c].numpy(), (pad, (0, 0))), ov.T, ov.kp)
            kvb[ov.kv_base(L, c):][:grp.size] = grp.reshape(-1)
    ov.dev.flush("KV")
    for L in layers:  # (the reference attends to the cache as stored)
        k, v = ov.read_kv(L, range(P0))
        cache = model.caches[L]
        cache.K, cache.V, cache.rows = k, v, v.clone()
    ids = _tokens(N)
    want = model.run(ids, P0)
    x = model.embed(ids)
    worst = 0.0
    p = P0
    while p < P0 + N:
        n = 4 if p >= 1008 else 1
        logits = ov.step(x[p - P0:p - P0 + n], p).float()
        for s in range(n):
            worst = max(worst, ref.relative(logits[s], want[p - P0 + s]))
        p += n
    for L in layers:
        k, v = ov.read_kv(L, range(P0, P0 + N))
        cache = model.caches[L]
        worst = max(worst, ref.relative(k, cache.K[:, P0:]), ref.relative(v, cache.V[:, P0:]))
    print(f"worst relative error {worst:.4f}")
    assert worst < TOL, worst
