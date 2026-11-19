"""The dense Qwen3.5 / 3.6 models' lanes on the NPU (``NpuQwen(sequences=)``; models/lanes.py; the 9B: on its design
of 8 slots, every format's kernel a code page): a conversation alone on lane 1 against lane 0 (its prompt's chunks
into its own caches, GDN states and conv states), then three conversations on one model object, each on a thread of
its own, against each one alone: greedy replies token for token, with the context's drafts and without, the lanes'
decode rounds in joint steps of up to 8 slots (an attention pass and a GDN chain per sequence). No drafter (files
without an MTP block), the LM head's rows whole."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from models import open_model
from models.npu_model import NpuModel
from models.sampling import Sampling

from ..common import essay
from ..model_files import require, size_gguf

pytestmark = pytest.mark.slow  # the whole model on the NPU: run with --run-slow

GREEDY = Sampling(greedy=True)
N = 40  # a reply's tokens


@pytest.mark.parametrize("size", ["9B", "27B"])
def test_lanes_greedy(size: str) -> None:
    path = size_gguf(size)
    require(path)
    model = open_model(path)
    assert model.parallel
    npu = model.load(capacity=4096, sequences=3)
    assert len(npu.lanes) == 3 and "3 sequences" in npu.describe() and npu.overlay.SLOTS == 8
    texts = [essay("addiction.txt")[:1200], essay("avg.txt")[2000:3000], essay("addiction.txt")[3000:3600]]
    asks = ["Repeat this text exactly, word for word:\n\n", "Repeat this text exactly, word for word:\n\n",
            "Summarize this text in three sentences:\n\n"]
    prompts = [model.chat([{"role": "user", "content": a + t}]) for a, t in zip(asks, texts, strict=True)]

    def reply(lane: NpuModel, ids: list[int], drafts: int, wait: float = 0.0) -> list[int]:
        time.sleep(wait)
        lane.reset()
        return list(lane.generate(lane.forward(ids), N, GREEDY, ids, drafts=drafts))

    first = reply(npu, prompts[2], 0)
    assert reply(npu.lanes[1], prompts[2], 0) == first, "a reply alone on lane 1 differs from lane 0's"
    rounds = npu._rounds
    for drafts in (0, 7):
        want = [reply(npu, ids, drafts) for ids in prompts]  # each alone, on lane 0
        assert len({tuple(w) for w in want}) == 3, "the replies differ"
        k = len(rounds.history)
        with ThreadPoolExecutor(3) as pool:  # (the third's prompt while the others decode)
            jobs = [pool.submit(reply, lane, ids, drafts, wait)
                    for lane, ids, wait in zip(npu.lanes, prompts, (0.0, 0.0, 0.3), strict=True)]
            got = [j.result(timeout=300) for j in jobs]
        for s in range(3):
            diff = next((i for i, (a, b) in enumerate(zip(got[s], want[s], strict=False)) if a != b), None)
            assert got[s] == want[s], f"{drafts} drafts, lane {s}: token {diff} of {len(want[s])} differs from the reply alone"
        hist = list(rounds.history)[k:]
        by = [sum(len(r.parts) == n for r in hist) for n in (1, 2, 3)]
        slots = sum(n for r in hist for _, n, _ in r.parts)
        print(f"{drafts} drafts: {len(hist)} rounds of 1 / 2 / 3 lanes: {by}, {slots} slots, "
              f"{sum(bool(r.late) for r in hist)} waited in vain; {npu.stats()}")
        assert all(sum(n for _, n, _ in r.parts) <= npu.overlay.SLOTS for r in hist)
        # (with 7 drafts the lanes that copy take a few steps of 8 slots: fewer rounds meet)
        assert by[1] + by[2] >= (4 if drafts else 20), by
