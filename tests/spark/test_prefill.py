"""The Spark-X prefill (models/spark/prefill.py: ``SparkPrefill``; its GEMMs: prefill/test_gemm.py, its attention:
prefill/test_attn256.py, its norms: prefill/test_norm.py) on the NPU:

- a slice of four layers (three sliding-window ones and a full one) against the decode path on the same tokens: a
  prompt from position 0, one after a first prompt's chunks and decode steps, one past the rings' wrap:
  the full layer's cache, the sliding-window layers' rings (their window's records) and the last token's logits (the
  LM head alone on the chunk's last output row: ``SparkOverlay.logits_of``, or a decode step after the chunks);
- the whole model (slow): prompts through the prefill, then the text after them as decode steps, against the fp32 CPU
  model: KL of the next-token distributions and argmax agreement (the prompt's last token's included).
"""

import numpy as np
import pytest
import torch

from artlite.decode import attn256 as AT
from artlite.decode import kv as KV
from artlite.gguf_model import GGUFFile
from models.spark.overlay import SparkOverlay
from models.spark.prefill import SparkPrefill

from ..common import to_torch
from ..numerics import kl_rows, rel_finite
from . import cpu, ref

VOCAB = 512
SLICE = [0, 1, 2, 3]
SIZES = (64, 128, 256, 512)  # (the tests' library: fewer designs to compile)
# (a first prompt's tokens through the prefill, then decode steps (their sizes), then the prompt's tokens, sizes)
CASES = [(0, (), 300, SIZES), (100, (8,) * 3 + (4,), 500, SIZES), (0, (), 3000, (64, 256, 2048))]


def _prompt(ov: SparkOverlay, pf: SparkPrefill, x: torch.Tensor, p: int) -> torch.Tensor:
    """x[p:] through the prefill's chunks, the tokens they leave as decode steps: the last token's logits."""
    n = x.shape[0]
    for T in pf.plan(p, n - p):
        m = min(T, n - p)
        pf.run(x[p:p + m].numpy(), p, T)
        p += m
    if p == n:
        return to_torch(ov.logits_of(pf.hidden(m - 1)[0])).float()
    out = None
    while p < n:
        S = min(8, n - p)
        out = to_torch(ov.step(x[p:p + S].numpy(), p)).float()
        p += S
    return out[-1]


def _steps(ov: SparkOverlay, x: torch.Tensor, p: int, sizes: tuple[int, ...]) -> int:
    for S in sizes:
        ov.step(x[p:p + S].numpy(), p, logits=False)
        p += S
    return p


def _records(ov: SparkOverlay, L: int, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Layer L's K, V per KV head of the positions a step at n reads (a ring: its window's last 512 + 32; a full layer:
    all n), as positions (lo, n): (K, V) each (n - lo, 256)."""
    nr = ov.ring_groups(L)
    lo = max(0, (n - 544) // 32 * 32) if nr else 0
    g0, g1 = lo // 32, -(-n // 32)
    hb = nr * AT.OBJS * ov.T if nr else ov.ash.head_bytes
    ov.dev.fetch("KV")
    kv = ov.dev.view("KV")
    out = []
    for h in range(ov.ash.kvh):
        base = ov.kv_off[L] + h * hb
        grp = np.stack([kv[base + (g % nr if nr else g) * AT.OBJS * ov.T:][:AT.OBJS * ov.T].reshape(AT.OBJS, ov.T)
                        for g in range(g0, g1)])
        K, V = KV.kv_from_groups(grp, AT.HD, dmajor=True)
        out.append((np.asarray(K, np.float32)[lo - 32 * g0:n - 32 * g0], np.asarray(V, np.float32)[lo - 32 * g0:n - 32 * g0]))
    return out


@pytest.mark.parametrize(("first", "steps", "n", "sizes"), CASES,
                         ids=[f"{f}-after{f + sum(s)}-n{n}-T{max(z)}" for f, s, n, z in CASES])
def test_spark_prefill_layers(first: int, steps: tuple[int, ...], n: int, sizes: tuple[int, ...]) -> None:
    """The caches and rings the prefill leaves against the decode path's (steps of 8) for the same essay tokens, up to
    the two paths' roundings (the GEMMs' inputs as bfp16 panels in both; the attention's P and records); the last
    token's logits. A prompt from position 0, after decode steps, after a first prompt's chunks (chunk, decode steps,
    chunk), past the rings' wrap (3000 tokens: 2048 + 952 positions, the rings 82 groups)."""
    g = ref.gguf()
    p0 = first + sum(steps)
    ids = ref.corpus_ids(p0 + n)
    x = cpu.SparkCpu(g, layers=[]).emb[torch.tensor(ids)]
    ov = SparkOverlay(g, layers=SLICE, vocab=VOCAB, max_positions=4096)
    ov.load()
    p = 0
    while p < len(ids):
        S = min(8, len(ids) - p)
        want_l = to_torch(ov.step(x[p:p + S].numpy(), p)).float()[-1]
        p += S
    want = {L: _records(ov, L, len(ids)) for L in SLICE}
    ov.reset()
    pf = SparkPrefill(ov, sizes=sizes).compile().bind()
    if first:
        _prompt(ov, pf, x[:first], 0)
    got_l = _prompt(ov, pf, x, _steps(ov, x, first, steps))
    worst = {"logits": rel_finite(got_l, want_l)}
    for L in SLICE:
        for h, ((k, v), (wk, wv)) in enumerate(zip(_records(ov, L, len(ids)), want[L], strict=True)):
            worst[f"K.{L}.{h}"], worst[f"V.{L}.{h}"] = rel_finite(k, wk), rel_finite(v, wv)
    print({k: round(v, 4) for k, v in worst.items()})
    # (the caches within 5%; the last token's logits, one row of 512, within 10%: the decode path's alone against the
    # fp32 model up to 7.8% on a row, ``test_overlay.py``)
    assert all(v < (0.1 if k == "logits" else 0.05) for k, v in worst.items()), worst


@pytest.mark.slow
def test_spark_prefill_teacher_forced() -> None:
    """The whole 4B model: a prompt through the prefill (a chat's first tokens, an essay's first 768), the text after it
    as decode steps (one token each on the chat, 8 on the essay), against the fp32 model: KL of the next-token
    distributions of the prompt's last token and the text after it, argmax agreement."""
    size = "4B"
    spec = (size, "Q4_K_M", "XHToken")
    g = GGUFFile(ref.path(*spec))
    ov = SparkOverlay(g, max_positions=1100)
    ov.load()
    pf = SparkPrefill(ov, sizes=SIZES).compile().bind()
    emb = cpu.SparkCpu(g, layers=[]).emb
    chat = ref.chat_ids(size)
    for ids, P, S in ((chat, len(chat) - 16, 1), (ref.essay_ids(1024, size), 768, 8)):
        want = ref.cpu_logits(spec, ids, from_row=P - 1)
        x = emb[torch.tensor(ids)]
        ov.reset()
        rows = [_prompt(ov, pf, x[:P], 0)]
        p = P
        while p < len(ids):
            n = min(S, len(ids) - p)
            rows += list(to_torch(ov.step(x[p:p + n].numpy(), p)).float())
            p += n
        got = torch.stack(rows)
        k = kl_rows(want, got)
        agree = float((want.argmax(-1) == got.argmax(-1)).double().mean())
        lk = float(kl_rows(want, ref.llama_logits(spec, ids, from_row=P - 1)).mean())
        print(f"{size} prompt {P}, {len(ids) - P} after: KL {float(k.mean()):.5f} (llama.cpp {lk:.5f}) "
              f"max {float(k.max()):.4f}, first {float(k[0]):.5f}, argmax {agree:.4f}")
        assert float(k.mean()) < max(4e-3, 1.6 * lk) and agree > 0.95
