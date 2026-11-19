"""Flint's lanes on the NPU (``NpuFlint(sequences=)``; models/lanes.py): three conversations on one model object, each
on a thread of its own, against each one alone: greedy replies token for token, without drafts and with the DFlash2
drafter's (one drafter, its records one lane's sequence's at a time: ``models.dflash``' ``own``; a lane's rows of
the shared steps injected from the staging region), the lanes' decode rounds in joint steps; after the rounds the
lanes shared, each lane's drafter gives the block (bf16 bits of its logits) it gives after the conversation alone."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from artlite.model_config import flint_gguf
from models import open_model
from models.flint.flint import NpuFlint
from models.npu_model import NpuModel
from models.sampling import Sampling

from ..common import essay
from ..model_files import require

pytestmark = pytest.mark.slow  # the whole model on the NPU: run with --run-slow

GREEDY = Sampling(greedy=True)
N = 64  # a reply's tokens


def test_lanes_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    # (a block in every round that has a slot for a draft: the drafter's records change lanes between them)
    monkeypatch.setattr(NpuFlint, "_block_pays", lambda self, k: True)
    path = flint_gguf()
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096, sequences=3)
    assert len(npu.lanes) == 3 and "3 sequences" in npu.describe() and npu.dflash is not None
    assert all(lane.flash is not None and lane.flash.model is lane for lane in npu.lanes)
    texts = [essay("addiction.txt")[:1200], essay("avg.txt")[2000:3000], essay("addiction.txt")[3000:3600]]
    asks = ["Repeat this text exactly, word for word:\n\n", "Repeat this text exactly, word for word:\n\n",
            "Summarize this text in three sentences:\n\n"]
    prompts = [model.chat([{"role": "user", "content": a + t}]) for a, t in zip(asks, texts, strict=True)]

    def reply(lane: NpuModel, ids: list[int], drafts: int, wait: float = 0.0) -> list[int]:
        time.sleep(wait)
        lane.reset()
        return list(lane.generate(lane.forward(ids), N, GREEDY, ids, drafts=drafts))

    def probe(lane: NpuModel, token: int) -> np.ndarray:
        """The drafter's block after the token (the next position's), the lane's rows that wait injected: its logits'
        bits."""
        with lane.device():
            fd, d = lane.flash, npu.dflash
            assert not fd.lost
            fd.own()
            fd._run(np.stack([lane._rows([token])[0], *[lane._mask] * (d.cfg.block - 1)]), lane.position)
            return d.logits().view(np.uint16).copy()

    def counters() -> tuple[int, int, int]:
        return npu._total("blocks"), npu._total("drafted") - npu._total("drafted_ctx"), \
            npu._total("accepted") - npu._total("accepted_ctx")

    first = reply(npu, prompts[2], 0)
    assert reply(npu.lanes[1], prompts[2], 0) == first, "a reply alone on lane 1 differs from lane 0's"
    rounds = npu._rounds
    for drafts in (0, 3):
        want = [reply(npu, ids, drafts) for ids in prompts]  # each alone, on lane 0
        assert len({tuple(w) for w in want}) == 3, "the replies differ"
        c0 = counters()
        alone = []  # (again: the head's table has seen every reply by now)
        for ids, w in zip(prompts, want, strict=True):
            assert reply(npu, ids, drafts) == w
            alone.append(probe(npu, w[-1]))
        c1 = counters()
        k = len(rounds.history)
        with ThreadPoolExecutor(3) as pool:  # (the third's prompt while the others decode)
            jobs = [pool.submit(reply, lane, ids, drafts, wait)
                    for lane, ids, wait in zip(npu.lanes, prompts, (0.0, 0.0, 0.5), strict=True)]
            got = [j.result(timeout=300) for j in jobs]
        c2 = counters()
        for s in range(3):
            diff = next((i for i, (a, b) in enumerate(zip(got[s], want[s], strict=False)) if a != b), None)
            assert got[s] == want[s], f"{drafts} drafts, lane {s}: token {diff} of {len(want[s])} differs from the reply alone"
        hist = list(rounds.history)[k:]
        by = [sum(len(r.parts) == n for r in hist) for n in (1, 2, 3)]
        slots = sum(n for r in hist for _, n, _ in r.parts)
        print(f"{drafts} drafts: {len(hist)} rounds of 1 / 2 / 3 lanes: {by}, {slots} slots, "
              f"{sum(bool(r.late) for r in hist)} waited in vain; blocks, their drafts, kept: alone "
              f"{tuple(b - a for a, b in zip(c0, c1, strict=True))}, together "
              f"{tuple(b - a for a, b in zip(c1, c2, strict=True))}; {npu.stats()}")
        assert all(sum(n for _, n, _ in r.parts) <= npu.overlay.SLOTS for r in hist)
        # (a lane's prompt takes as long as some twenty of the others' rounds, which run between its chunk's pieces;
        # with drafts the lanes that copy are through before the third's prompt)
        assert by[1] + by[2] >= (6 if drafts else 30) and (drafts or by[2] >= 15), by
        if drafts:  # (the drafter drafted in both, and its guesses were kept)
            assert c1[0] > c0[0] and c2[0] > c1[0] and c2[2] - c1[2] > 0.25 * (c2[1] - c1[1]), (c0, c1, c2)
        for s, lane in enumerate(npu.lanes):  # (each lane's records and the rows that wait: the conversation's alone)
            assert np.array_equal(probe(lane, want[s][-1]), alone[s]), f"{drafts} drafts: lane {s}'s drafter after the shared rounds"
