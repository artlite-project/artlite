"""Decisions on the NPU (``models.server.decisions``; vllm-d3's d3-lite: qwen35, non-causal, every tensor Q8_0): each
question's prompt through ``NpuModel.outputs`` (one chunk from position 0), its label logits on the host, against the same
label rows times llama.cpp's last row of the same prompt (``qwen35/dump.cpp``'s ``result_norm``: the whole prompt in
one graph, the file's non-causal attention), the answers' probabilities within 0.03 and the same choices. llama.cpp's
CPU and Vulkan builds are 0.019 apart at most on 320 such questions, the NPU 0.087 (median 0.003): the prefill's bfp16
GEMMs, ~3% of each layer's own output. vllm-decision2's Decision-2.0-Kai-0.6B (qwen3 on the Llama path, its joint head on
the host) against the fp32 CPU model's rows (``llama/cpu.py``) through the same head: within 0.05 (against llama-server's
CPU answers on 320 questions: 0.040 at most, 313 of 320 choices the same). LiquidAI's d1-2B (lfm3, its full layers
non-causal): each question alone against llama.cpp's last row, a state's questions in one sequence (its ``set``
attention) against the fp32 CPU model's rows (``lfm/cpu.py``) of the same sequence, within 0.03 and the same choices."""

import os

import numpy as np
import pytest

from artlite.model_config import decision_gguf
from models import open_model
from models.server.decisions import Decisions

from .common import essay
from .qwen35 import ref

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is the passage mostly about?",
              "criteria": {"programming": None, "cooking": None, "sports": None, "music": None}},
    "tone": {"type": "score", "instructions": "How optimistic is the tone of the passage?",
             "criteria": ["very pessimistic", "pessimistic", "neutral", "optimistic", "very optimistic"]},
    "advice": {"type": "noul", "instructions": "Does the passage give advice to the reader?"},
}


def test_d3_lite() -> None:
    path = decision_gguf("d3-lite")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    model = open_model(path)
    npu = model.load(capacity=4096)
    assert not npu.causal
    d = Decisions.of(path, model.tokenizer.encode)
    text = essay("apple.txt")
    worst = 0.0
    for state in (text[:400], text[2000:5000]):  # (~170 and ~800 prompt tokens: chunks of 256 and 1024)
        body = {"state": state, "questions": QUESTIONS}
        qs = d.questions(body)
        for p in d.prompts(state, qs, model.tokenizer.encode):
            q = qs[p.question]
            npu.reset()
            got = d.answer(q, [d.scores(q, npu.outputs(p.ids, p.rows))])
            g = ref.golden(path, p.ids, layers=("g",), names="result_norm", logits="none")
            want = d.answer(q, [d.scores(q, np.asarray(g["result_norm"], np.float64).reshape(-1, model.cfg.hidden_size))])
            assert got.get("choice") == want.get("choice"), (q.id, got, want)
            pw, pg = (a.get("probabilities", {"true": a.get("noul")}) for a in (want, got))
            worst = max(worst, *(abs(pw[k] - pg[k]) for k in pw))
    print(f"max |dp| {worst:.4f}")
    assert worst < 0.03


def test_kai() -> None:
    from artlite.gguf_model import GGUFFile

    from .llama.cpu import LlamaCpu

    path = decision_gguf("kai")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    model = open_model(path)
    npu = model.load(capacity=4096)
    d = Decisions.of(path, model.tokenizer.encode)
    cpu = LlamaCpu(GGUFFile(path))
    worst = 0.0
    state = essay("apple.txt")[:1200]
    qs = d.questions({"state": state, "questions": QUESTIONS})
    for p in d.prompts(state, qs, model.tokenizer.encode):
        q = qs[p.question]
        npu.reset()
        got = d.answer(q, [d.scores(q, npu.outputs(p.ids, p.rows))])
        want = d.answer(q, [d.scores(q, cpu.outputs(p.ids, list(p.rows)).numpy())])  # (rows ascending: options, question)
        pw, pg = (a.get("probabilities", {"true": a.get("noul")}) for a in (want, got))
        worst = max(worst, *(abs(pw[k] - pg[k]) for k in pw))
    print(f"max |dp| {worst:.4f}")
    assert worst < 0.05


def test_d1() -> None:
    from artlite.gguf_model import GGUFFile

    from .lfm.cpu import LfmCpu

    path = decision_gguf("d1-2b")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    model = open_model(path)
    npu = model.load(capacity=4096)
    assert not npu.causal
    d = Decisions.of(path, model.tokenizer.encode)
    cpu = LfmCpu(GGUFFile(path), vocab=0)
    text = essay("apple.txt")
    worst = {1: 0.0, 16: 0.0}
    for state in (text[:400], text[2000:5000]):  # (~130 and ~750 tokens a question alone)
        qs = d.questions({"state": state, "questions": QUESTIONS})
        for size in (1, 16):
            for p in d.prompts(state, qs, model.tokenizer.encode, set_size=size):
                npu.reset()
                rows = npu.outputs(p.ids, p.rows, sets=p.sets)
                if size == 1:  # (llama.cpp reads each question alone)
                    g = ref.golden(path, p.ids, layers=("g",), names="result_norm", logits="none")
                    want = np.asarray(g["result_norm"], np.float64).reshape(-1, model.cfg.hidden_size)[-1:]
                else:
                    want = cpu.outputs(p.ids, list(p.rows), sets=p.sets).double().numpy()
                assert len(p.questions) == (1 if size == 1 else len(qs))
                for j, i in enumerate(p.questions):
                    q = qs[i]
                    a, b = (d.answer(q, [d.scores(q, x[j:j + 1])]) for x in (rows, want))
                    assert a.get("choice") == b.get("choice"), (q.id, a, b)
                    pa, pb = (x.get("probabilities", {"true": x.get("noul")}) for x in (a, b))
                    worst[size] = max(worst[size], *(abs(pa[k] - pb[k]) for k in pa))
    print(f"max |dp|: alone vs llama.cpp {worst[1]:.4f}, a sequence vs the CPU model {worst[16]:.4f}")
    assert max(worst.values()) < 0.03
