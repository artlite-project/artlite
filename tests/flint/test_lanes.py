"""Flint's lanes on the NPU (``NpuFlint(sequences=)``; models/lanes.py): three conversations on one model object, each
on a thread of its own, against each one alone on the same object: greedy replies token for token, with the context's
drafts and without, the lanes' decode rounds in joint steps (an MoE layer runs the union of their slots' experts)."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from artlite.model_config import flint_gguf
from models import open_model
from models.npu_model import NpuModel
from models.sampling import Sampling

from ..common import essay
from ..model_files import require

pytestmark = pytest.mark.slow  # the whole model on the NPU: run with --run-slow

GREEDY = Sampling(greedy=True)
N = 64  # a reply's tokens


def test_lanes_greedy() -> None:
    path = flint_gguf()
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096, sequences=3)  # (with a drafter's file there: without it)
    assert len(npu.lanes) == 3 and "3 sequences" in npu.describe() and npu.dflash is None
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
        k = len(rounds.history)
        with ThreadPoolExecutor(3) as pool:  # (the third's prompt while the others decode)
            jobs = [pool.submit(reply, lane, ids, drafts, wait)
                    for lane, ids, wait in zip(npu.lanes, prompts, (0.0, 0.0, 0.5), strict=True)]
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
        # (a lane's prompt takes as long as some twenty of the others' rounds, which run between its chunk's pieces;
        # with drafts the lanes that copy are through before the third's prompt)
        assert by[1] + by[2] >= 30 and (drafts or by[2] >= 15), by
