"""Qwen3.8-27B's lanes with its MTP drafter (``NpuQwen(sequences=)``, ``drafting.MtpDrafting`` per lane; the DFlash2
drafter's place too): three conversations on one model object, each on a thread of its own, against each one alone:
greedy replies token for token, the drafter's guesses kept; and after the rounds the lanes shared, each lane's drafter
gives the logits (bf16 bits) it gives after the conversation alone: its cache of the sequence, the hidden state and
the pending rows are those."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from models import open_model
from models.npu_model import NpuModel
from models.sampling import Sampling

from ..common import essay
from ..model_files import require, size_gguf

pytestmark = pytest.mark.slow  # the whole model on the NPU: run with --run-slow

GREEDY = Sampling(greedy=True)
N = 40  # a reply's tokens


def test_lanes_drafter() -> None:
    path = size_gguf("3.8-27B")
    require(path)
    model = open_model(path)
    assert model.parallel
    npu = model.load(capacity=4096, sequences=3)
    assert len(npu.lanes) == 3 and npu.drafter is not None and npu.dflash is None and "MTP" in npu.describe()
    assert all(lane.mtp is not None and lane.mtp.model is lane for lane in npu.lanes)
    for lane in npu.lanes:
        lane.lookup = False  # (the drafter's guesses alone)
    texts = [essay("addiction.txt")[:1200], essay("avg.txt")[2000:3000], essay("addiction.txt")[3000:3600]]
    asks = ["Repeat this text exactly, word for word:\n\n", "Summarize this text in three sentences:\n\n",
            "List the main claims of this text:\n\n"]
    prompts = [model.chat([{"role": "user", "content": a + t}]) for a, t in zip(asks, texts, strict=True)]

    def reply(lane: NpuModel, ids: list[int], wait: float = 0.0) -> list[int]:
        time.sleep(wait)
        lane.reset()
        return list(lane.generate(lane.forward(ids), N, GREEDY, ids))

    def probe(lane: NpuModel, token: int) -> np.ndarray:
        """The drafter's logits of the token's row (the next position's, with the rows that wait), bf16 bits."""
        with lane.device():
            lane.mtp._flush()
            rows = [*lane.mtp.pend, (lane._rows([token])[0], lane.mtp.h)]
            return np.asarray(lane.mtp.step(rows, logits=True)[-1]).view(np.uint16).copy()

    def counters() -> tuple[int, int]:
        return npu._total("drafted"), npu._total("accepted")

    want = [reply(npu, ids) for ids in prompts]  # (each alone, on lane 0; the head's table has seen all of them then)
    assert len({tuple(w) for w in want}) == 3, "the replies differ"
    d0, a0 = counters()
    alone = []
    for ids, w in zip(prompts, want, strict=True):
        assert reply(npu, ids) == w
        alone.append(probe(npu, w[-1]))
    d1, a1 = counters()
    assert d1 - d0 > 30 and a1 - a0 > 0.4 * (d1 - d0), (d1 - d0, a1 - a0)
    assert reply(npu.lanes[2], prompts[1]) == want[1], "a reply alone on lane 2 differs from lane 0's"
    assert np.array_equal(probe(npu.lanes[2], want[1][-1]), alone[1]), "the drafter alone on lane 2: other logits"
    rounds = npu._rounds
    k = len(rounds.history)
    d2, a2 = counters()
    with ThreadPoolExecutor(3) as pool:  # (the third's prompt while the others decode)
        jobs = [pool.submit(reply, lane, ids, wait)
                for lane, ids, wait in zip(npu.lanes, prompts, (0.0, 0.0, 0.3), strict=True)]
        got = [j.result(timeout=600) for j in jobs]
    d3, a3 = counters()
    for s in range(3):
        diff = next((i for i, (a, b) in enumerate(zip(got[s], want[s], strict=False)) if a != b), None)
        assert got[s] == want[s], f"lane {s}: token {diff} of {len(want[s])} differs from the reply alone"
    hist = list(rounds.history)[k:]
    by = [sum(len(r.parts) == n for r in hist) for n in (1, 2, 3)]
    print(f"alone: {d1 - d0} drafts, {a1 - a0} kept; together: {d3 - d2} drafts, {a3 - a2} kept in {len(hist)} rounds "
          f"of 1 / 2 / 3 lanes: {by}; {npu.stats()}")
    assert all(sum(n for _, n, _ in r.parts) <= npu.overlay.SLOTS for r in hist)
    assert by[1] + by[2] >= 6 and d3 - d2 > 15 and a3 - a2 > 0.4 * (d3 - d2), (by, d3 - d2, a3 - a2)
    for s, lane in enumerate(npu.lanes):
        assert np.array_equal(probe(lane, want[s][-1]), alone[s]), f"lane {s}'s drafter after the shared rounds"
