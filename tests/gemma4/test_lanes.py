"""Gemma 4's lanes on the NPU (``NpuGemma(sequences=)``; models/lanes.py): three conversations on one model object,
each on a separate thread, against each one alone on the same object: greedy replies token for token, without
drafts and with them (the drafter's guesses for a lane before the lanes' joint step, the context's), the lanes'
decode rounds in joint steps; the drafter's steps on a lane's caches are those on the same text alone, bit for bit."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from artlite.model_config import gemma_gguf
from models import open_model
from models.npu_model import NpuModel
from models.sampling import Sampling

from ..common import essay
from ..model_files import require

pytestmark = pytest.mark.slow  # the whole model on the NPU: run with --run-slow

GREEDY = Sampling(greedy=True)
N = 80  # a reply's tokens


@pytest.fixture(scope="module")
def loaded() -> tuple[object, NpuModel]:
    path = gemma_gguf("Q4_0")
    require(path)
    model = open_model(path)
    return model, model.load(capacity=2048, sequences=3)


def _prompts(model: object) -> list[list[int]]:
    texts = [essay("addiction.txt")[:1200], essay("avg.txt")[2000:3000], essay("addiction.txt")[3000:3600]]
    asks = ["Repeat this text exactly, word for word:\n\n", "Repeat this text exactly, word for word:\n\n",
            "Summarize this text in three sentences:\n\n"]
    return [model.chat([{"role": "user", "content": a + t}]) for a, t in zip(asks, texts, strict=True)]


def test_lanes_greedy(loaded: tuple[object, NpuModel]) -> None:
    model, npu = loaded
    assert len(npu.lanes) == 3 and "3 sequences" in npu.describe()
    prompts = _prompts(model)

    def reply(lane: NpuModel, ids: list[int], drafts: int, wait: float = 0.0) -> list[int]:
        time.sleep(wait)
        lane.reset()
        return list(lane.generate(lane.forward(ids), N, GREEDY, ids, drafts=drafts))

    rounds = npu._rounds
    for drafts in (0, 3):
        want = [reply(npu, ids, drafts) for ids in prompts]  # each alone, on lane 0
        assert len({tuple(w) for w in want}) == 3, "the replies differ"
        k, d0 = len(rounds.history), [lane.dispatches for lane in npu.lanes]
        mtp0 = npu._total("drafted") - npu._total("drafted_ctx")
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
        if not drafts:
            assert by[2] >= 30 and by[1] + by[2] > 0.8 * len(hist), by
        else:  # (three lanes: the fourth slot is a draft of one of them)
            assert by[2] >= 15 and sum(sum(n for _, n, _ in r.parts) == 4 for r in hist if len(r.parts) == 3) >= 10, by
            if npu.drafter is not None:  # (its guesses ran beside the other lanes' rounds)
                assert npu._total("drafted") - npu._total("drafted_ctx") > mtp0
        assert sum(lane.dispatches - d for lane, d in zip(npu.lanes, d0, strict=True)) > len(hist)


def test_lanes_drafter(loaded: tuple[object, NpuModel]) -> None:
    """A draft step on a lane's caches (after a prompt through the prefill and decode steps, the other lanes holding
    other texts at other positions) gives the logits of the same step on that text alone, bit for bit."""
    model, npu = loaded
    if npu.drafter is None:
        pytest.skip("no drafter next to the model")
    prompts = _prompts(model)

    def drafts(lane: NpuModel, ids: list[int]) -> list[np.ndarray]:
        """Two draft steps after the prompt but its last 5 tokens, and two after those (decode steps)."""
        out = []
        for part in (ids[:-5], ids[-5:]):
            token = int(np.argmax(lane.forward(part)))
            with lane.device():
                h = lane.hidden_state()
                for _ in range(2):
                    raw = lane.drafter.step(lane.host.embed([token])[0], h, lane.position)
                    out.append(np.array(raw).view(np.uint16))
                    token, h = lane.drafter.token(raw), None
        return out

    want = []
    for ids in prompts:  # each alone, as lane 0
        npu.reset()
        want.append(drafts(npu, ids))
    for lane in npu.lanes:
        lane.reset()
    order = (2, 0, 1)
    got = {s: drafts(npu.lanes[s], prompts[s]) for s in order}  # (the lanes one after another: the others' caches stay)
    for s in order:
        assert all(np.array_equal(a, b) for a, b in zip(got[s], want[s], strict=True)), f"lane {s}: its draft steps"
    assert not np.array_equal(want[0][0], want[1][0])
