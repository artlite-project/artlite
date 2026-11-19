"""artlite-server with several sequences on the NPU (``-np``: Llama 3.2 1B, Gemma 4 E4B with its drafter, Flint, LFM3
and Spark-X 1.7B, on three lanes): requests that meet give each one's answer alone - greedy ones, and a sampled one with a seed, whose
draws are its own -, their decode steps shared; a conversation's next turn continues on its lane."""

import threading
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from artlite.model_config import flint_gguf, gemma_gguf, lfm3_gguf, llama_gguf, spark_gguf
from models import open_model
from models.server.api import create_app
from models.server.engine import Engine
from models.server.settings import ModelInfo, Options

from ..common import essay
from ..model_files import require

pytestmark = pytest.mark.slow  # a GGUF model on the NPU: run with --run-slow

PATHS = {"llama-1B": llama_gguf("Q4_K_M", "1B"), "gemma-E4B": gemma_gguf("Q4_0"), "flint": flint_gguf(),
         "lfm3": lfm3_gguf("QAD-Q4_0"), "spark-1.7B": spark_gguf("1.7B")}


@pytest.fixture(scope="module", params=list(PATHS))
def served(request: pytest.FixtureRequest) -> Iterator[tuple[TestClient, Engine]]:
    path = PATHS[request.param]
    require(path)
    model = open_model(path)
    npu = model.load(capacity=4096, sequences=3)
    engine = Engine(lambda: (model, npu))
    engine.start()
    with TestClient(create_app(engine, Options(alias="test", n_ctx=4096), ModelInfo.from_gguf(path))) as c:
        assert engine.wait_ready(300)
        yield c, engine
    engine.stop()
    engine.join(30)


def test_requests_meet(served: tuple[TestClient, Engine]) -> None:
    client, engine = served
    assert client.get("/props").json()["total_slots"] == 3 and len(client.get("/slots").json()) == 3
    text = essay("addiction.txt")
    asks = [("Summarize this text in a few sentences:\n\n" + text[:1500], {"temperature": 0}),
            ("List the main claims of this text:\n\n" + text[1500:2600], {"temperature": 0}),
            ("Write a short poem about the sea and its weather.", {"temperature": 0.8, "seed": 7})]

    def ask(i: int, out: dict[int, dict], **body: object) -> None:
        content, sampling = asks[i]
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}],
                                                      "max_tokens": 96, "cache_prompt": False, **sampling, **body})
        assert r.status_code == 200, r.text
        out[i] = r.json()

    alone: dict[int, dict] = {}
    for i in range(3):
        ask(i, alone)
    rounds = engine.npu._rounds
    k = len(rounds.history)
    together: dict[int, dict] = {}
    threads = [threading.Thread(target=ask, args=(i, together)) for i in range(3)]
    for t in threads:
        t.start()
    busy, t0 = 0, time.perf_counter()
    while any(t.is_alive() for t in threads) and time.perf_counter() - t0 < 60:
        busy = max(busy, sum(s["is_processing"] for s in client.get("/slots").json()))
        time.sleep(0.005)
    for t in threads:
        t.join(60)
    for i in range(3):  # (an answer: its thought, where the model thinks first, and its text)
        a, b = ((m.get("reasoning_content") or "") + m["content"]
                for m in (x[i]["choices"][0]["message"] for x in (alone, together)))
        assert a == b and len(a) > 40, f"request {i}: beside the others {b!r}, alone {a!r}"
    hist = list(rounds.history)[k:]
    shared = sum(len(r.parts) > 1 for r in hist)
    print(f"{len(hist)} rounds, {shared} of several lanes, {sum(len(r.parts) == 3 for r in hist)} of all three; "
          f"{busy} lanes processing at once")
    assert busy == 3 and shared > 0.6 * len(hist)
    # (a conversation's next turn: on the lane that keeps it)
    first = [{"role": "user", "content": asks[0][0]}, {"role": "assistant", "content": together[0]["choices"][0][
        "message"]["content"]}, {"role": "user", "content": "And in one sentence?"}]
    kept = [list(lane.cache.tokens) for lane in engine.lanes]
    r = client.post("/v1/chat/completions", json={"messages": first, "max_tokens": 24, "temperature": 0})
    assert r.status_code == 200 and r.json()["timings"]["cache_n"] > 300, r.text
    # (one lane's cache went on: longer, or where the answer was a thought the prompt leaves out, other tokens)
    moved = [list(lane.cache.tokens) != old for lane, old in zip(engine.lanes, kept, strict=True)]
    assert sum(moved) == 1, moved
