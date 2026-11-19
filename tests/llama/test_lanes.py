"""Llama's and Qwen3's lanes on the NPU (``NpuLlama(sequences=)``; models/lanes.py): three conversations on one model
object, each on a thread of its own, against each one alone on the same object: greedy replies token for token, with
the context's drafts and without, the lanes' decode rounds in joint steps; one conversation's prompt arrives while the
others decode (its chunks and steps with the device alone, then its rounds with theirs)."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from models import open_model
from models.npu_model import NpuModel
from models.sampling import Sampling

from ..common import essay, essays
from ..model_files import llama_path

pytestmark = pytest.mark.slow  # whole models on the NPU: run with --run-slow

GREEDY = Sampling(greedy=True)
N = 96  # a reply's tokens


@pytest.mark.parametrize(("quant", "size"), [("Q4_K_M", "1B"), ("Q4_1", "qwen3:0.6B")])
def test_lanes_greedy(quant: str, size: str) -> None:
    model = open_model(llama_path(quant, size))
    npu = model.load(capacity=4096, sequences=3)
    assert len(npu.lanes) == 3 and "3 sequences" in npu.describe()
    texts = [essay("addiction.txt")[:1200], essay("avg.txt")[2000:3000], essay("addiction.txt")[3000:3600]]
    asks = ["Repeat this text exactly, word for word:\n\n", "Repeat this text exactly, word for word:\n\n",
            "Summarize this text in three sentences:\n\n"]
    prompts = [model.chat([{"role": "user", "content": a + t}]) for a, t in zip(asks, texts, strict=True)]

    def reply(lane: NpuModel, ids: list[int], drafts: int, wait: float = 0.0) -> list[int]:
        time.sleep(wait)
        lane.reset()
        return list(lane.generate(lane.forward(ids), N, GREEDY, ids, drafts=drafts))

    rounds = npu._rounds
    for drafts in (0, 3):
        want = [reply(npu, ids, drafts) for ids in prompts]  # each alone, on lane 0
        assert len({tuple(w) for w in want}) == 3, "the replies differ"
        k, d0 = len(rounds.history), [lane.dispatches for lane in npu.lanes]
        with ThreadPoolExecutor(3) as pool:  # (the third's prompt while the others decode)
            jobs = [pool.submit(reply, lane, ids, drafts, wait)
                    for lane, ids, wait in zip(npu.lanes, prompts, (0.0, 0.0, 0.3), strict=True)]
            got = [j.result(timeout=120) for j in jobs]
        for s in range(3):
            diff = next((i for i, (a, b) in enumerate(zip(got[s], want[s], strict=False)) if a != b), None)
            assert got[s] == want[s], f"{drafts} drafts, lane {s}: token {diff} of {len(want[s])} differs from the reply alone"
        hist = list(rounds.history)[k:]
        by = [sum(len(r.parts) == n for r in hist) for n in (1, 2, 3)]
        slots = sum(n for r in hist for _, n, _ in r.parts)
        print(f"{quant} {size}, {drafts} drafts: {len(hist)} rounds of 1 / 2 / 3 lanes: {by}, {slots} slots, "
              f"{sum(bool(r.late) for r in hist)} waited in vain; {npu.stats()}")
        assert by[2] >= 30 and by[1] + by[2] > 0.8 * len(hist), by
        assert all(sum(n for _, n, _ in r.parts) <= npu.overlay.SLOTS for r in hist)
        if drafts:  # (two lanes: a draft each; alone: up to three)
            assert max(n for r in hist if len(r.parts) == 2 for _, n, _ in r.parts) == 2
        # (a lane counts the steps it had a part in)
        assert sum(lane.dispatches - d for lane, d in zip(npu.lanes, d0, strict=True)) > len(hist)


def test_lanes_prompt_beside_decode() -> None:
    """A long prompt on one lane while another generates: the generating lane keeps getting steps, between the
    prompt's chunk's pieces (the chunk lends the NPU: ``Rounds.lend``), and both lanes' tokens are those of each
    alone."""
    model = open_model(llama_path("Q4_K_M", "1B"))
    npu = model.load(capacity=4096, sequences=2)
    a, b = npu.lanes
    text = essays()
    short = model.chat([{"role": "user", "content": "Summarize this text in a few sentences:\n\n" + text[:1200]}])
    long = model.chat([{"role": "user", "content": "List the main claims of this text:\n\n" + text[1200:14000]}])
    assert 2500 < len(long) < 3800

    def reply(lane: NpuModel, ids: list[int], n: int, wait: float = 0.0, during: list[int] | None = None) -> list[int]:
        time.sleep(wait)
        lane.reset()
        k = len(npu._rounds.history)
        logits = lane.forward(ids)
        if during is not None:
            during.append(len(npu._rounds.history) - k)
        return list(lane.generate(logits, n, GREEDY, ids, drafts=0))

    want = [reply(a, short, 160), reply(a, long, 32)]
    during: list[int] = []
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(reply, a, short, 160), pool.submit(reply, b, long, 32, 0.4, during)]
        got = [j.result(timeout=120) for j in jobs]
    print(f"{len(long)} prompt tokens beside a generation: {during[0]} of its rounds ran during the prompt")
    assert got == want
    assert during[0] >= 3, during
