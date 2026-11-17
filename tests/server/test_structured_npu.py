"""artlite-server's structured output on the NPU, one small model per family: a JSON schema extraction that parses and
validates, an enum classification, a thought then JSON; greedy, through the server's routes (the engine over the
model, a test client)."""

import json
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from artlite.model_config import minicpmv_gguf
from models import open_model
from models.server.api import create_app
from models.server.engine import Engine
from models.server.settings import ModelInfo, Options

from ..model_files import require

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

CASES = [minicpmv_gguf()]  # (qwen35, a 248K vocabulary, a screened LM head)
PERSON = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "city": {"type": "string"}}, "required": ["name", "age", "city"]}


@pytest.fixture(scope="module", params=CASES, ids=[os.path.basename(p) for p in CASES])
def client(request: pytest.FixtureRequest) -> Iterator[TestClient]:
    require(request.param)
    model = open_model(request.param)
    npu = model.load(capacity=4096)
    engine = Engine(lambda: (model, npu))
    engine.start()
    with TestClient(create_app(engine, Options(alias="test", n_ctx=4096), ModelInfo.from_gguf(request.param))) as c:
        assert engine.wait_ready(300)
        yield c
    engine.stop()
    engine.join(30)


def ask(client: TestClient, text: str, **body: object) -> dict:
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}], "temperature": 0,
                                                  "max_tokens": 400, **body})
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]


def test_extraction(client: TestClient) -> None:
    rf = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON}}
    choice = ask(client, "Extract the person: John Smith is 42 years old and lives in Paris.", response_format=rf)
    out = json.loads(choice["message"]["content"])
    assert list(out) == ["name", "age", "city"] and out["age"] == 42 and "Paris" in out["city"], out
    assert choice["finish_reason"] == "stop"


def test_classification(client: TestClient) -> None:
    rf = {"type": "json_schema", "json_schema": {"name": "sentiment",
                                                 "schema": {"enum": ["positive", "negative", "neutral"]}}}
    for text, want in (("I love this product, it works great!", "positive"),
                       ("This is the worst purchase I ever made.", "negative")):
        choice = ask(client, f"Classify the sentiment: {text}", response_format=rf)
        assert json.loads(choice["message"]["content"]) == want


def test_thought_then_json(client: TestClient) -> None:
    choice = ask(client, "Think about it, then extract the person: Ada Lovelace, 36, London.",
                 json_schema=PERSON, chat_template_kwargs={"enable_thinking": True}, max_tokens=1500)
    msg = choice["message"]
    assert msg.get("reasoning_content") and choice["finish_reason"] == "stop", msg
    out = json.loads(msg["content"])
    assert out["age"] == 36 and "London" in out["city"]
