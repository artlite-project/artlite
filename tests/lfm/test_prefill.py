"""The LFM3 prefill (models/lfm/prefill.py: ``LfmPrefill``; its GEMMs: prefill/test_gemm.py, its conv:
test_conv_prefill.py, its attention: test_attn_prefill.py) on the NPU:

- a slice of six layers (conv, conv, full, conv, sliding-window, conv) against the decode path on the same tokens: a
  prompt from position 0 and one after decode steps whose last one leaves the conv histories shifted (``conv_base``),
  K single and hi + lo: the caches, the conv histories and the last token's logits (the LM head alone on the chunk's last
  output row: ``LfmOverlay.logits_of``, or a decode step after the chunks);
- the whole model (slow): prompts through the prefill, then the text after them as decode steps, against the fp32 CPU
  model: KL of the next-token distributions and argmax agreement (the prompt's last token's included);
- a non-causal file's (d1-2B's) slice: a prompt from position 0, its full layer's attention both ways as a state and
  questions after it (``sets``), against the fp32 CPU model's output rows of the same slice.
"""

import itertools

import numpy as np
import pytest
import torch

from artlite.gguf_model import GGUFFile
from models.lfm.overlay import LfmOverlay
from models.lfm.prefill import LfmPrefill

from ..common import to_torch
from ..numerics import kl_rows, rel_finite
from . import ref

VOCAB = 2048
SIZES = (64, 128, 256, 512)  # (the tests' library: fewer designs to compile)
BIG = (64, 1024, 2048)  # (chunks of 1024-token sub-chunks: ``prefill.BIG``, norm and prep blocks of 1024)
# (a first prompt's tokens through the prefill, then steps (their sizes; the last one's slots shift the conv histories),
# then the prompt's tokens, K, sizes)
CASES = [(0, (), 300, "single", SIZES), (0, (4,) * 31 + (1, 3), 400, "single", SIZES),
         (0, (4,) * 15 + (2, 2), 333, "hilo", SIZES), (150, (2,) + (4,) * 10, 250, "single", SIZES),
         (0, (), 2100, "single", BIG)]


def _steps(ov: LfmOverlay, x: torch.Tensor, p: int, sizes: tuple[int, ...]) -> int:
    for S in sizes:
        ov.step(x[p:p + S], p, logits=False)
        p += S
    return p


def _prompt(ov: LfmOverlay, pf: LfmPrefill, x: torch.Tensor, p: int) -> torch.Tensor:
    """x[p:] through the prefill's chunks, the tokens they leave as decode steps: the last token's logits."""
    n = x.shape[0]
    for T in pf.plan(p, n - p):
        m = min(T, n - p)
        pf.run(x[p:p + m], p, T)
        p += m
    if p == n:
        return to_torch(ov.logits_of(pf.hidden(m - 1))).float()
    while p < n:
        S = min(4, n - p)
        out = to_torch(ov.step(x[p:p + S], p)).float()
        p += S
    return out[-1]


@pytest.mark.parametrize(("first", "steps", "n", "kv_k", "sizes"), CASES,
                         ids=[f"{f}-after{f + sum(s)}-n{n}-{k}-T{max(z)}" for f, s, n, k, z in CASES])
def test_lfm_prefill_layers(first: int, steps: tuple[int, ...], n: int, kv_k: str, sizes: tuple[int, ...]) -> None:
    """The caches and conv histories the prefill leaves against the decode path's (steps of 4) for the same essay
    tokens, up to the two paths' roundings (GEMM inputs as bfp16 panels here, bf16 there; the attention's P blocks);
    the last token's logits. A prompt from position 0 or after decode steps (the conv histories shifted: ``conv_base``),
    also after a first prompt's chunks (chunk, decode steps to the next cache group, chunk)."""
    g = ref.gguf()
    layers = [0, 1, 2, 3, 4, 5]
    p0 = first + sum(steps)
    ids = ref.essay_ids(p0 + n)
    x = ref.cpu.LfmCpu(g, layers=[], vocab=0).emb[torch.tensor(ids)]
    ov = LfmOverlay(g, layers=layers, vocab=VOCAB, max_positions=1024 if n < 1024 else 4096, kv_k=kv_k)
    ov.load()
    p = 0
    while p < len(ids):
        S = min(4, len(ids) - p)
        want_l = to_torch(ov.step(x[p:p + S], p)).float()[-1]
        p += S
    att = [L for L in layers if ov.kind(L) != "conv"]
    want = {L: ov.read_kv(L, range(len(ids))) for L in att}
    want_c = {L: ov.read_conv(L)[ov.conv_s:ov.conv_s + 6] for L in layers if ov.kind(L) == "conv"}
    ov.reset()
    pf = LfmPrefill(ov, sizes=sizes).compile().bind()
    if first:
        _prompt(ov, pf, x[:first], 0)
    got_l = _prompt(ov, pf, x, _steps(ov, x, first, steps))
    worst = {"logits": rel_finite(got_l, want_l)}
    for L in att:
        k, v = ov.read_kv(L, range(len(ids)))
        worst[f"K.{L}"], worst[f"V.{L}"] = rel_finite(k, want[L][0]), rel_finite(v, want[L][1])
    for L, rows in want_c.items():
        worst[f"conv.{L}"] = rel_finite(ov.read_conv(L)[ov.conv_s:ov.conv_s + 6], rows)
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < 0.05 for v in worst.values()), worst


@pytest.mark.slow
@pytest.mark.parametrize("quant", ["QAD-Q4_0", "Q4_K_M", "Q8_0"])
def test_lfm_prefill_teacher_forced(quant: str) -> None:
    """The whole model: a prompt through the prefill (a chat's first tokens, an essay's first 768), the text after it
    as decode steps (one token each on the chat, 4 on the essay), against the fp32 model: KL of the next-token
    distributions of the prompt's last token and the text after it within the prefill's floor (its GEMMs' 8-bit
    operands: QAD-Q4_0's chat 1.15e-3 where decode steps alone give 6.5e-4 on the same rows) or llama.cpp's own (its Q8
    activations), the argmax mostly the same."""
    g = ref.gguf(quant)
    cpu = ref.cpu.LfmCpu(g)
    ov = LfmOverlay(g, max_positions=1024)
    ov.load()
    pf = LfmPrefill(ov, sizes=SIZES).compile().bind()
    chat = ref.chat_ids(quant)
    for ids, P, S in ((chat, len(chat) - 32, 1), (ref.essay_ids(1024, quant), 768, 4)):
        want = ref.cpu_logits(quant, ids, model=cpu)[P - 1:]
        x = cpu.emb[torch.tensor(ids)]
        ov.reset()
        rows = [_prompt(ov, pf, x[:P], 0)]
        p = P
        while p < len(ids):
            n = min(S, len(ids) - p)
            rows += list(to_torch(ov.step(x[p:p + n], p)).float())
            p += n
        got = torch.stack(rows)
        k = kl_rows(want, got)
        agree = float((want.argmax(-1) == got.argmax(-1)).double().mean())
        lk = float(kl_rows(want, ref.llama_logits(quant, ids)[P - 1:]).mean())
        print(f"{quant} prompt {P}, {len(ids) - P} after: KL {float(k.mean()):.5f} (llama.cpp {lk:.5f}) "
              f"max {float(k.max()):.4f}, first {float(k[0]):.5f}, argmax {agree:.4f}")
        assert float(k.mean()) < max(1.5e-3, 1.5 * lk) and agree > 0.95


# (a non-causal prompt's tokens, its bounds: the state's end, each question's)
SET_CASES = [(50, (50,)), (300, (230, 252, 270, 300)), (480, (100, 140, 141, 200, 260, 330, 400, 480)),
             (1500, (1300, 1400, 1500))]


@pytest.mark.parametrize(("n", "sets"), SET_CASES, ids=[f"n{n}-q{len(s) - 1}" for n, s in SET_CASES])
def test_lfm_prefill_set(n: int, sets: tuple[int, ...]) -> None:
    """d1-2B's first six layers (conv, conv, full, conv, sliding-window, conv) on a prompt read whole from position 0:
    the full layer non-causal as a state and questions after it, the others causal from zero histories; every output
    row against the fp32 CPU model's (output-normed) within 5% (the prefill's bfp16 GEMM operands)."""
    g = GGUFFile(ref.d1_path())
    layers = [0, 1, 2, 3, 4, 5]
    ids = ref.essay_ids(n, path=g.path)
    cpu_m = ref.cpu.LfmCpu(g, layers=layers, vocab=0)
    want = cpu_m.outputs(ids, slice(None), sets=sets)
    ov = LfmOverlay(g, layers=layers, vocab=VOCAB, max_positions=2048)
    ov.load()
    sizes = BIG if n > 512 else SIZES
    pf = LfmPrefill(ov, sizes=sizes).compile().bind()
    assert pf.nc and not pf.attn[("swa", sizes[0])].nc
    ov.reset()
    pf.run(cpu_m.emb[torch.tensor(ids)], 0, sets=sets)
    got = cpu_m.rms(torch.from_numpy(pf.hidden(0, n)), cpu_m.norm)
    worst = {"all": rel_finite(got, want)}
    for k, (a, b) in enumerate(itertools.pairwise((0, *sets))):  # (the state, each question)
        worst[f"part{k}"] = rel_finite(got[a:b], want[a:b])
    print({k: round(v, 4) for k, v in worst.items()})
    assert all(v < 0.05 for v in worst.values()), worst
    assert np.isfinite(got.numpy()).all()
