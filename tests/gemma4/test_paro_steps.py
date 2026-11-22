"""A Gemma 4 PARO checkpoint on the NPU (models/gemma4/paro_steps.py): z-lab's E4B PARO, a slice of its first layers
(sliding-window and full-attention layers) through ``GemmaOverlay.step`` in steps of one, three and four tokens, against
the fp32 model (gemma_cpu.GemmaCpu) of the slice's pseudo-quantized GGUF (``goldens.paro.pseudo_gguf``: the effective
weights deq(Q) M^T as F16): the logits of the first ``VOCAB`` rows (the screened head's exact ones) and the residual
after the slice, each token's the same bit for bit at every step size; the prefill (models/gemma4/prefill.py's PARO
path) of the same slice: its caches against the decode path's, the residual it leaves and the last token's logits after
a step from the slice's last layer against the fp32 model; the whole model (slow), teacher-forced after a prompt through
the prefill, against the fp32 model of the checkpoint's linears computed exactly (``ParoTensor.effective``)."""

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

from artlite.gguf_model import GGUFFile
from artlite.hf_model import HfModel
from artlite.model_config import checkpoint
from artlite.paro import ParoTensor
from models.gemma4.host import GemmaHost
from models.gemma4.overlay import GemmaOverlay
from models.gemma4.tokenizer import GemmaTokenizer

from ..common import cached_ref, essay
from ..goldens.paro import pseudo_gguf
from ..model_files import require
from ..numerics import rel, rel_finite
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


def _ref_prefill(path: str, ids: list[int]) -> dict[str, torch.Tensor]:
    """The fp32 slice's residual into its last layer per token and the last token's raw logits of the first VOCAB rows
    (cached)."""
    ref = GGUFFile(pseudo_gguf(path, f16=True, layers=LAYERS))

    def run() -> dict[str, torch.Tensor]:
        cpu = gemma_cpu.GemmaCpu(ref)
        trace: dict[str, torch.Tensor] = {}
        h = cpu.hidden(ids, trace=trace)
        return {"logits": (h[-1:] @ cpu.mat("token_embd.weight")[:VOCAB].T).float(), "r": trace[f"l_out.{LAYERS - 2}"]}

    return cached_ref(ref.path, "paro_prefill", (ids, VOCAB), run, sources=gemma_cpu.SOURCES)


@pytest.mark.parametrize(("n", "p0"), [(200, 0), (300, 64), (1000, 0)], ids=["256", "512-after-decode", "1024"])
def test_paro_prefill(n: int, p0: int) -> None:
    """The prefill (``GemmaPrefill``) of the slice, p0 tokens as decode steps first, then n in a chunk (of 256, 512 and
    1024 tokens: its operators' blocks of each size, ``_paro_init``): the caches within
    5% of the decode path's (rel; its GEMMs' bfp16 inputs against the decode's bf16), the residual into the last layer
    within 3% of the fp32 model's (measured ~1.5%), the last token's logits after a step from the last layer within 2%."""
    from models.gemma4.prefill import GemmaPrefill

    path = checkpoint(E4B)
    require(path)
    g = HfModel(path)
    ids = GemmaTokenizer(g).encode(essay("apple.txt"))[:p0 + n]
    want = _ref_prefill(path, ids)
    host = GemmaHost(g)
    x, e = host.embed(ids).astype(bfloat16).astype(np.float32), host.layer_rows(ids).astype(bfloat16)
    layers = list(range(LAYERS))
    ov = GemmaOverlay(g, layers=layers, vocab=VOCAB, max_positions=1024)
    ov.load()
    for t in range(0, p0 + n, 4):  # (the decode path's caches)
        ov.step(x[t:t + 4], e[t:t + 4], t, logits=False)
    kv = {L: ov.read_kv(L, range(p0 + n)) for L in layers}
    ov.reset()
    for t in range(0, p0, 4):
        ov.step(x[t:t + 4], e[t:t + 4], t, logits=False)
    pf = GemmaPrefill(ov).compile().bind()
    t = t_last = p0
    for T in pf.plan(p0, n):
        m = min(T, p0 + n - t)
        pf.run(x[t:t + m], e[t:t + m], t, T)
        t_last, t = t, t + m
    err = {}
    for L in layers:
        k, v = ov.read_kv(L, range(p0 + n))
        err[f"K.{L}"], err[f"V.{L}"] = rel(k, kv[L][0]), rel(v, kv[L][1])
    r = np.stack([pf.residual(i) for i in range(t - t_last)])  # (the last chunk's rows, into the last layer)
    err["r"] = max(rel_finite(torch.from_numpy(r[i]), want["r"][t_last + i]) for i in range(t - t_last))
    lg = ov.step(r[-1:], e[-1:], p0 + n - 1, start=layers[-1], emb=x[-1:])
    err["logits"] = rel_finite(torch.from_numpy(np.asarray(lg, np.float32)[0]), want["logits"][0])
    print({k: round(v, 4) for k, v in err.items()})
    assert all(v < 0.05 for k, v in err.items() if k[0] in "KV"), err
    assert err["r"] < 0.03 and err["logits"] < 0.02, err


class _ParoCpu(gemma_cpu.GemmaCpu):
    """The fp32 model of a PARO checkpoint: each PARO linear as the plain linear it computes (``ParoTensor.effective``:
    deq(Q) M^T exactly), every other tensor as the virtual GGUF has it."""

    def mat(self, name: str) -> torch.Tensor:
        if name not in self._mats and isinstance(t := self.gguf.tensor(name), ParoTensor):
            self._mats[name] = t.effective().astype(np.float32)
        return super().mat(name)


@pytest.mark.slow
def test_paro_model() -> None:
    """NpuGemma on the checkpoint, a chat prompt through the prefill then its continuation teacher-forced at 1 and 4
    tokens per dispatch: KL of the next-token distributions against the fp32 model (``_ParoCpu``) and argmax
    agreement (measured: 0.00029 both, 100% of 34 tokens)."""
    from models.gemma4.gemma import NpuGemma
    from models.gemma4.tokenizer import chat_prompt

    from ..numerics import kl_rows
    from .test_npu import LONG, TEXT

    path = checkpoint(E4B)
    require(path)
    g = HfModel(path)
    tok = GemmaTokenizer(g)
    prompt = tok.encode(chat_prompt(g, [{"role": "user", "content": LONG}]))
    seq = prompt + tok.encode(TEXT)
    first = len(prompt) - 1

    def run() -> torch.Tensor:
        cpu = _ParoCpu(g)
        cpu.preload()
        return cpu.logits(cpu.hidden(seq))[first:]

    want = cached_ref(path, "paro_model_logits", (seq, first), run, sources=gemma_cpu.SOURCES)
    npu = NpuGemma(g, capacity=1024, drafter=None)
    assert npu.prefill is not None and sum(npu.prefill.plan(0, first)) >= first, "the prompt through the prefill"
    for batch in (1, 4):
        got = torch.from_numpy(npu.forced_logits(seq, first, batch=batch))
        kl = float(kl_rows(want, got).mean())
        agree = float((want.argmax(-1) == got.argmax(-1)).float().mean())
        print(f"{batch} per dispatch: KL {kl:.5f}, argmax agreement {agree:.3f} over {len(seq) - first} tokens")
        assert kl < 0.003 and agree > 0.9, (batch, kl, agree)


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
