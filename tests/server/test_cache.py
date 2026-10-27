"""The prompt cache (``models.server.cache``) through the server's routes on ``test_server``'s fake model: a recurrent
model's checkpoints (where the last message and the first user message begin; the fake's turns: its role tags),
conversations saved to RAM and restored, the RAM limit and the checkpoints per sequence, a failure, image prompts. The
fake's restores check that the device holds the tokens of the checkpoint they restore; saving copies a sequence past 8
positions here (``MIN_SAVE``)."""

import base64
import contextlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from models.server import cache
from models.server.engine import Engine

from .test_server import (  # noqa: F401 (_stop_engines: the engines' cleanup, autouse)
    FakeModel,
    FakeNpu,
    _stop_engines,
    app_of,
    chat,
    make,
)

TURNS = ("<|assistant|>", "<|user|>")  # (the fake template's: the answer's tag, the user's)
GROUP = 4  # (the fake's cache group: a checkpoint's mark rounded down to it)


@pytest.fixture(autouse=True)
def _min_save(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache, "MIN_SAVE", 8)


@contextlib.contextmanager
def served(**kw: object) -> Iterator[tuple[FakeModel, FakeNpu, Engine, TestClient]]:
    model, npu, engine = make(capacity=4096, **kw)
    with TestClient(app_of(engine, n_ctx=4096)) as client:
        assert engine.wait_ready(5)
        yield model, npu, engine, client


def complete(client: TestClient, ids: list[int]) -> dict:
    r = client.post("/completion", json={"prompt": ids})
    assert r.status_code == 200, r.text
    return r.json()


def words(n: int, w: str = "w") -> str:
    return " ".join(f"{w}{i}" for i in range(n))


def test_last_message() -> None:
    """A recurrent model continues from the checkpoint where the last prompt's answer began when the next prompt
    re-renders the answer otherwise (a client's own echo of it), and when it asks again (another answer)."""
    with served(exact=True, turns=TURNS) as (model, npu, engine, client):
        turn = [{"role": "user", "content": words(40)}]
        ids = model.chat(turn)
        npu.replies = [["x", " y"], ["z"], ["w"]]
        chat(client, turn)
        at = (len(ids) - 1) // GROUP * GROUP
        assert ("checkpoint", at) in npu.calls
        r = chat(client, turn + [{"role": "assistant", "content": "x z"}, {"role": "user", "content": "more"}]).json()
        assert r["timings"]["cache_n"] == at and ("restore", at, True) in npu.calls
        r = chat(client, turn).json()
        assert r["timings"]["cache_n"] == at and r["timings"]["prompt_n"] == len(ids) - at
        assert engine.cached == npu.seq[:npu.position]


@pytest.mark.parametrize("exact", [True, False], ids=["recurrent", "rewinds"])
def test_conversations(exact: bool) -> None:
    """Two conversations in turns: each one's next turn continues it from RAM (its records from where the device holds
    the same tokens, its end checkpoint)."""
    with served(exact=exact) as (model, npu, engine, client):
        tok = model.tokenizer
        a, b = model.encode(words(40, "a")), model.encode(words(30, "b"))
        npu.replies = [["r1"], ["r2"], ["r3"], ["r4"]]
        complete(client, a)
        complete(client, b)  # (a saved: its records and a checkpoint at its end)
        assert ("read", 0, len(a) + 1) in npu.calls and len(engine.cache.entries) == 1
        a2 = a + tok.encode("r1 more")
        r = complete(client, a2)
        assert r["timings"]["cache_n"] == len(a) + 1  # (its end: inside a cache group of the model that rewinds too)
        assert ("write", 1, len(a) + 1) in npu.calls and ("restore", len(a) + 1, True) in npu.calls
        r = complete(client, b + tok.encode("r2 more"))
        assert r["timings"]["cache_n"] == len(b) + 1
        assert engine.cached == npu.seq[:npu.position]


def test_first_user_message() -> None:
    """A recurrent model's checkpoint where the first user message begins: another conversation with the same system
    prompt continues from it."""
    with served(exact=True, turns=TURNS) as (model, npu, _engine, client):
        system = {"role": "system", "content": words(40, "s")}
        npu.replies = [["ok"], ["fine"]]
        chat(client, [system, {"role": "user", "content": "q1"}])
        at = model.chat([system, {"role": "user", "content": "q1"}]).index(model.tokenizer.id("<|user|>"))
        at -= at % GROUP
        assert ("checkpoint", at) in npu.calls
        r = chat(client, [system, {"role": "user", "content": "q2 and more"}]).json()
        assert r["timings"]["cache_n"] == at and ("restore", at, True) in npu.calls


def test_ram_limit() -> None:
    """Within the RAM limit: the least recently used conversations cut back to their first checkpoint (held once:
    by the device's conversation when it shares it), then dropped."""
    ram = 1600  # (bytes: the fake's 8 per position of records and of each checkpoint)
    with served(exact=True, turns=TURNS, cache={"cache_ram": ram}) as (model, npu, engine, client):
        system = {"role": "system", "content": words(20, "s")}
        at = model.chat([system, {"role": "user", "content": "q"}]).index(model.tokenizer.id("<|user|>"))
        at -= at % GROUP
        for i in range(4):
            npu.replies = [[f"r{i}"]]
            chat(client, [system, {"role": "user", "content": words(30, f"q{i}_")}])
            assert engine.cache.nbytes() <= ram
        npu.replies = [["x"]]
        chat(client, [{"role": "system", "content": words(30, "t")}, {"role": "user", "content": "other"}])
        assert engine.cache.nbytes() <= ram and [len(e.tokens) for e in engine.cache.entries] == [at]
        r = chat(client, [system, {"role": "user", "content": "new"}]).json()
        assert r["timings"]["cache_n"] == at and ("write", 2, at) in npu.calls


def test_checkpoints_per_sequence() -> None:
    """At most ``checkpoints`` per conversation: the oldest go, the lowest (its first user message) stays."""
    with served(exact=True, turns=TURNS, cache={"checkpoints": 3}) as (model, npu, engine, client):
        messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q0"}]
        at = model.chat(messages).index(model.tokenizer.id("<|user|>")) // GROUP * GROUP
        for i in range(6):
            npu.replies = [[f"r{i}"]]
            r = chat(client, messages).json()
            messages += [{"role": "assistant", "content": f"r{i}"}, {"role": "user", "content": f"q{i + 1}"}]
        assert len(engine.cache.active.checkpoints) == 3 and min(engine.cache.active.checkpoints) == at
        assert r["timings"]["cache_n"] > at


def test_no_ram() -> None:
    """``cache_ram`` 0: nothing saved; another conversation in between costs the whole prompt."""
    with served(exact=True, cache={"cache_ram": 0}) as (model, npu, _engine, client):
        a, b = model.encode(words(40, "a")), model.encode(words(30, "b"))
        npu.replies = [["r1"], ["r2"], ["r3"]]
        complete(client, a)
        complete(client, b)
        r = complete(client, a + model.tokenizer.encode("r1 more"))
        assert r["timings"]["cache_n"] == 0 and not [c for c in npu.calls if c[0] in ("read", "write")]


def test_failure_keeps_saved() -> None:
    """A failure drops the device's state; the conversations in RAM stay."""
    with served(exact=True) as (model, npu, _engine, client):
        a, b = model.encode(words(40, "a")), model.encode(words(30, "b"))
        npu.replies = [["r1"], ["r2"]]
        complete(client, a)
        complete(client, b)
        npu.fail = RuntimeError("DispatchError: TIMEOUT")
        assert client.post("/completion", json={"prompt": b + model.tokenizer.encode("r2 x")}).status_code == 500
        npu.fail = None
        r = complete(client, a + model.tokenizer.encode("r1 more"))
        assert r["timings"]["cache_n"] == len(a) + 1 and ("write", 0, len(a) + 1) in npu.calls


def test_images() -> None:
    """An image prompt's checkpoints (a sequence of ids and embedding rows): the next turn after the same image."""
    with served(exact=True, vision=True, turns=TURNS) as (_model, npu, _engine, client):
        url = "data:image/png;base64," + base64.b64encode(b"c" * 24).decode()
        image = {"type": "image_url", "image_url": {"url": url}}
        turn = [{"role": "system", "content": words(10, "s")},
                {"role": "user", "content": [{"type": "text", "text": "look"}, image]}]
        npu.replies = [["a", " cat"], ["x"]]
        n = chat(client, turn).json()["timings"]["prompt_n"]
        more = [{"role": "assistant", "content": "a dog"}, {"role": "user", "content": "sure?"}]
        r = chat(client, turn + more).json()
        at = (n - 1) // GROUP * GROUP
        assert r["timings"]["cache_n"] == at and ("restore", at, True) in npu.calls
