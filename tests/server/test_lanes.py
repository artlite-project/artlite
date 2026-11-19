"""artlite-server with several sequences (``-np``; ``Engine``'s lanes) on the host: the fake model with a fake per
lane. Requests that meet run at once, each on a lane; the lane a job goes to (the one it names, the one that keeps its
prompt, the least recently used); ``/slots`` and ``/props``; a call beside a generation; a failure and a cancel of one
lane's job; the shutdown."""

import threading
import time

import httpx
from fastapi.testclient import TestClient

from .fakes import FakeNpu, app_of, chat, deltas, events, make


def _ran(npu: FakeNpu) -> int:
    """The prompts a lane's fake ran."""
    return sum(c[0] == "forward" for c in npu.calls)


def test_requests_meet() -> None:
    """Two requests at a time on two lanes: both generate at once (``/slots``: both processing), each its lane's
    reply; a third waits for a lane."""
    _model, npu, engine = make(lanes=2, delay=0.01)
    a, b = npu.lanes
    a.replies, b.replies = [[f" a{i}" for i in range(40)], [" third"]], [[f" b{i}" for i in range(40)], [" third"]]
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5) and len(engine.lanes) == 2
        assert client.get("/props").json()["total_slots"] == 2
        out: dict[str, httpx.Response] = {}

        def post(key: str, text: str, **body: object) -> None:
            out[key] = chat(client, text, stream=True, **body)

        threads = [threading.Thread(target=post, args=("a", "one"), kwargs={"id_slot": 0}),
                   threading.Thread(target=post, args=("b", "two"), kwargs={"id_slot": 1})]
        for t in threads:
            t.start()
        both, t0 = False, time.perf_counter()
        while not both and time.perf_counter() - t0 < 5:
            slots = client.get("/slots").json()
            both = len(slots) == 2 and all(s["is_processing"] for s in slots)
            time.sleep(0.002)
        assert both, "both lanes generate at once"
        threads.append(threading.Thread(target=post, args=("c", "three")))
        threads[-1].start()
        for t in threads:
            t.join(10)
        assert deltas(events(out["a"])).split() == [f"a{i}" for i in range(40)]
        assert deltas(events(out["b"])).split() == [f"b{i}" for i in range(40)]
        assert deltas(events(out["c"])) == " third" and ":" in events(out["c"])  # (it waited: pings)
        assert [s["is_processing"] for s in client.get("/slots").json()] == [False, False]
        assert [s["id"] for s in client.get("/slots").json()] == [0, 1]
        r = chat(client, "x", id_slot=2)
        assert r.status_code == 400 and "id_slot" in r.text
        assert chat(client, "x", id_slot=-1).status_code == 200


def test_lane_of_a_prompt() -> None:
    """One request after another: a new conversation takes the least recently used lane, a conversation's next turn
    the lane whose cache keeps its prompt; a request without the prompt cache the least recently used."""
    _model, npu, engine = make(lanes=3)
    a, b, c = npu.lanes
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        first = [{"role": "user", "content": "tell me about the first of the many things"}]
        other = [{"role": "user", "content": "and now something else entirely, at length please"}]
        assert chat(client, first).status_code == 200 and [_ran(x) for x in npu.lanes] == [1, 0, 0]
        assert chat(client, other).status_code == 200 and [_ran(x) for x in npu.lanes] == [1, 1, 0]
        turn = [*first, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        assert chat(client, turn).status_code == 200 and [_ran(x) for x in npu.lanes] == [2, 1, 0]
        assert a.position > b.position > 0 == c.position
        turn2 = [*other, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        assert chat(client, turn2).status_code == 200 and [_ran(x) for x in npu.lanes] == [2, 2, 0]
        assert chat(client, turn, cache_prompt=False).status_code == 200 and [_ran(x) for x in npu.lanes] == [2, 2, 1]
        assert len(engine.lanes[0].cache.tokens) == a.position and engine.cached == a.seq[:a.position]


def test_call_beside_a_job() -> None:
    """Work that is no job (a tokenizer route's call on the worker: lane 0's) runs while another lane generates; a job
    on lane 0 waits for it."""
    _model, npu, engine = make(lanes=2, delay=0.01)
    npu.lanes[1].replies = [[f" b{i}" for i in range(60)]]
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        out: list[httpx.Response] = []
        t = threading.Thread(target=lambda: out.append(chat(client, "go", id_slot=1)))
        t.start()
        while engine.lanes[1].current is None:
            time.sleep(0.002)
        order: list[str] = []
        done = engine.run(lambda: order.append("call") or threading.current_thread().name)
        assert done.result(5) == "artlite-npu" and engine.lanes[1].current is not None  # (it did not wait for lane 1)
        t.join(10)
        assert out[0].status_code == 200 and len(out[0].json()["choices"][0]["message"]["content"].split()) == 60


def test_one_lane_fails() -> None:
    """A lane whose job fails starts over (its cache cleared, its sequence reset before its next job); the other
    lane's job ends as it would."""
    _model, npu, engine = make(lanes=2, delay=0.005)
    a, b = npu.lanes
    a.replies, b.replies = [[f" a{i}" for i in range(30)]], [[" never"], [" again"]]
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        assert chat(client, "warm", id_slot=1).status_code == 200  # (lane 1 holds a sequence)
        b.fail = RuntimeError("boom")
        out: dict[str, httpx.Response] = {}
        threads = [threading.Thread(target=lambda: out.setdefault("a", chat(client, "one", id_slot=0))),
                   threading.Thread(target=lambda: out.setdefault("b", chat(client, "two", id_slot=1)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert out["a"].status_code == 200 and len(out["a"].json()["choices"][0]["message"]["content"].split()) == 30
        assert out["b"].status_code == 500 and "boom" in out["b"].text
        assert engine.lanes[1].dirty and not engine.lanes[0].dirty and engine.lanes[1].cache.tokens == []
        b.fail, n = None, b.calls.count(("reset",))
        r = chat(client, "three", id_slot=1)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == " again"
        assert b.calls.count(("reset",)) == n + 1 and not engine.lanes[1].dirty


def test_cancel_and_stop() -> None:
    """The UI's Stop reaches a job on any lane; the shutdown ends every lane's worker, the running and the waiting
    jobs cancelled."""
    _model, npu, engine = make(lanes=2, delay=0.01)
    for lane in npu.lanes:
        lane.replies = [[f" w{i}" for i in range(500)]] * 2
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        out: dict[str, httpx.Response] = {}

        def post(key: str, slot: int) -> None:
            out[key] = client.post("/v1/chat/completions", headers={"X-Conversation-Id": key}, json={
                "messages": [{"role": "user", "content": key}], "stream": True, "id_slot": slot})

        threads = [threading.Thread(target=post, args=(k, s)) for k, s in (("c0", 0), ("c1", 1), ("c2", 1))]
        for t in threads[:2]:
            t.start()
        while any(lane.current is None or lane.current.n_gen < 3 for lane in engine.lanes):
            time.sleep(0.005)
        threads[2].start()  # (it waits for its lane)
        while engine.waiting() != 1:
            time.sleep(0.005)
        assert client.delete("/v1/stream", params={"conv_id": "c1"}).json()["cancelled"] == 1
        threads[1].join(10)
        assert events(out["c1"])[-1]["error"]["message"] == "the request was cancelled"
        while engine.lanes[1].current is None or engine.lanes[1].current.n_gen < 3:  # (the waiting one took its lane)
            time.sleep(0.005)
        assert engine.lanes[0].current is not None
        engine.stop()
        engine.join(10)
        for t in threads:
            t.join(10)
        assert all(lane.thread is None or not lane.thread.is_alive() for lane in engine.lanes)
        assert all(events(out[k])[-1]["error"]["message"] == "the request was cancelled" for k in ("c0", "c2"))
