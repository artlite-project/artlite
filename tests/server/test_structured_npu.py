"""artlite-server's structured output on the NPU, one small model per family and call syntax: a JSON schema extraction
that parses and validates, an enum classification, a thought then JSON; tool calls with typed and nested arguments
(required, a named function, auto with parallel_tool_calls false, a response format beside the tools); custom tools'
inputs (a Lark grammar, a regex); a decision's options with their probabilities. Greedy, through the server's routes (the engine over the model, a test client)."""

import json
import math
import os
import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from artlite.model_config import gemma_gguf, lfm3_gguf, llama_gguf, minicpmv_gguf, spark_gguf
from models import open_model
from models.server.api import create_app
from models.server.engine import Engine
from models.server.settings import ModelInfo, Options
from models.server.tools import LLAMA3, syntax_of

from ..model_files import require

pytestmark = pytest.mark.slow  # GGUF models on the NPU: run with --run-slow

# (qwen35 with a screened head: Qwen's XML calls; LFM3: the same syntax, its tags user-defined tokens; Gemma 4; Llama
# 3's JSON calls; Spark-X: GLM-4.5's calls)
CASES = [minicpmv_gguf(), lfm3_gguf("QAD-Q4_0"), gemma_gguf("Q4_0"), llama_gguf("Q4_K_M", "1B"),
         spark_gguf("1.7B")]
PERSON = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "city": {"type": "string"}}, "required": ["name", "age", "city"]}
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The weather forecast for a city",
                                            "parameters": {"type": "object", "properties": {
                                                "city": {"type": "string", "description": "The city's name"},
                                                "days": {"type": "integer", "description": "Days ahead, 1 to 7"},
                                                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                                "options": {"type": "object", "properties": {
                                                    "wind": {"type": "boolean"}, "hourly": {"type": "boolean"}}}},
                                                "required": ["city", "days"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "The current time in a time zone",
                                         "parameters": {"type": "object", "properties": {
                                             "zone": {"type": "string", "description": "An IANA time zone"}},
                                             "required": ["zone"]}}}


@pytest.fixture(scope="module", params=CASES, ids=[os.path.basename(p) for p in CASES])
def served(request: pytest.FixtureRequest) -> Iterator[tuple[TestClient, object]]:
    require(request.param)
    model = open_model(request.param)
    npu = model.load(capacity=4096)
    engine = Engine(lambda: (model, npu))
    engine.start()
    with TestClient(create_app(engine, Options(alias="test", n_ctx=4096), ModelInfo.from_gguf(request.param))) as c:
        assert engine.wait_ready(300)
        yield c, model
    engine.stop()
    engine.join(30)


def ask(client: TestClient, text: str, **body: object) -> dict:
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}], "temperature": 0,
                                                  "max_tokens": 400, **body})
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]


def weather_args(call: dict) -> dict:
    assert call["function"]["name"] == "get_weather"
    args = json.loads(call["function"]["arguments"])
    assert set(args) <= {"city", "days", "units", "options"} and isinstance(args["city"], str), args
    assert isinstance(args["days"], int) and not isinstance(args["days"], bool), args
    assert args.get("units", "celsius") in ("celsius", "fahrenheit"), args
    assert set(args.get("options", {})) <= {"wind", "hourly"}, args
    assert all(isinstance(v, bool) for v in args.get("options", {}).values()), args
    return args


def test_extraction(served: tuple) -> None:
    client, _ = served
    rf = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON}}
    choice = ask(client, "Extract the person: John Smith is 42 years old and lives in Paris.", response_format=rf)
    out = json.loads(choice["message"]["content"])
    assert list(out) == ["name", "age", "city"] and out["age"] == 42 and "Paris" in out["city"], out
    assert choice["finish_reason"] == "stop"


def test_classification(served: tuple) -> None:
    client, _ = served
    rf = {"type": "json_schema", "json_schema": {"name": "sentiment",
                                                 "schema": {"enum": ["positive", "negative", "neutral"]}}}
    for text, want in (("I love this product, it works great!", "positive"),
                       ("This is the worst purchase I ever made.", "negative")):
        choice = ask(client, f"Classify the sentiment: {text}", response_format=rf)
        assert json.loads(choice["message"]["content"]) == want


def test_thought_then_json(served: tuple) -> None:
    client, model = served
    if not model.thinks:
        pytest.skip("the model has no thinking mode")
    choice = ask(client, "Think about it, then extract the person: Ada Lovelace, 36, London.",
                 json_schema=PERSON, chat_template_kwargs={"enable_thinking": True}, max_tokens=1500)
    msg = choice["message"]
    assert msg.get("reasoning_content") and choice["finish_reason"] == "stop", msg
    out = json.loads(msg["content"])
    assert out["age"] == 36 and "London" in out["city"]


def test_tool_required(served: tuple) -> None:
    """A call whatever the model would rather write: typed, nested arguments its schema takes."""
    client, _ = served
    choice = ask(client, "What will the weather be like in Paris over the next 3 days? Include the wind.",
                 tools=[WEATHER, TIME], tool_choice="required")
    (call,) = choice["message"]["tool_calls"]
    args = weather_args(call)
    assert "Paris" in args["city"] and choice["finish_reason"] == "tool_calls"


def test_tool_named(served: tuple) -> None:
    client, _ = served
    choice = ask(client, "I am flying to Tokyo tomorrow.", tools=[WEATHER, TIME],
                 tool_choice={"type": "function", "function": {"name": "get_time"}})
    (call,) = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "get_time" and isinstance(json.loads(call["function"]["arguments"])["zone"],
                                                                 str)
    assert not choice["message"]["content"]


def test_tool_auto(served: tuple) -> None:
    """auto, one call at most: a call is valid, the answer ends after it."""
    client, _ = served
    choice = ask(client, "Use the tools: what is the weather in Berlin tomorrow and what time is it there?",
                 tools=[WEATHER, TIME], parallel_tool_calls=False)
    calls = choice["message"].get("tool_calls") or []
    assert len(calls) <= 1
    for c in calls:
        if c["function"]["name"] == "get_weather":
            weather_args(c)
        else:
            assert isinstance(json.loads(c["function"]["arguments"])["zone"], str)


def test_tool_or_json(served: tuple) -> None:
    """A response format beside the tools: a call, or the format's JSON (no prose)."""
    client, _ = served
    rf = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {
        "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}
    for text in ("What is the weather in Rome today?", "What is the capital of France?"):
        choice = ask(client, text, tools=[WEATHER, TIME], response_format=rf)
        msg = choice["message"]
        if msg.get("tool_calls"):
            assert msg["tool_calls"][0]["function"]["name"] in ("get_weather", "get_time")
        else:
            assert isinstance(json.loads(msg["content"])["answer"], str), msg


def test_custom_tool(served: tuple) -> None:
    """OpenAI's custom tools: a Lark grammar's input held in the template's call (Llama 3's JSON calls: free text), a
    regex's everywhere."""
    client, _ = served
    calc = {"type": "custom", "custom": {"name": "calc", "description": "Evaluates an arithmetic expression", "format": {
        "type": "grammar", "grammar": {"syntax": "lark", "definition": 'start: NUM (OP NUM)*\nNUM: /[0-9]+/\nOP: "+" | "-" | "*"\n'}}}}
    date = {"type": "custom", "custom": {"name": "remind", "description": "Sets a reminder on a date", "format": {
        "type": "grammar", "grammar": {"syntax": "regex", "definition": r"\d{4}-\d{2}-\d{2}"}}}}
    choice = ask(client, "What is 12 plus 30, times 2? Use the calculator.", tools=[calc, date],
                 tool_choice={"type": "custom", "custom": {"name": "calc"}})
    (call,) = choice["message"]["tool_calls"]
    assert call["type"] == "custom" and call["custom"]["name"] == "calc", call
    if syntax_of(client.get("/props").json()["chat_template"]) is not LLAMA3:
        assert re.fullmatch(r"[0-9]+([-+*][0-9]+)*", call["custom"]["input"]), call
    choice = ask(client, "Remind me about the dentist on the 3rd of March 2027.", tools=[calc, date],
                 tool_choice={"type": "custom", "custom": {"name": "remind"}})
    (call,) = choice["message"]["tool_calls"]
    assert call["custom"]["name"] == "remind" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", call["custom"]["input"]), call


def test_decision(served: tuple) -> None:
    """A classification with logprobs (Halogen's decision shape): the options' probabilities, summing to 1, the most
    likely one the content and the one a constrained generation writes; options that part later (decode steps from a
    checkpoint) the same probabilities in a second request, the most likely one the generation's."""
    client, _ = served
    rf = {"type": "json_schema", "json_schema": {"name": "sentiment",
                                                 "schema": {"enum": ["positive", "negative", "neutral"]}}}
    text = "Classify the sentiment: This is the worst purchase I ever made."
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}],
                                                  "response_format": rf, "logprobs": True, "top_logprobs": 3})
    assert r.status_code == 200, r.text
    c = r.json()["choices"][0]
    top = c["logprobs"]["content"][0]["top_logprobs"]
    assert {t["token"] for t in top} == {"positive", "negative", "neutral"}
    assert abs(sum(math.exp(t["logprob"]) for t in top) - 1) < 1e-4
    assert json.loads(c["message"]["content"]) == top[0]["token"] == "negative"
    gen = ask(client, text, response_format=rf)
    assert json.loads(gen["message"]["content"]) == "negative"
    deep = {"type": "json_schema", "json_schema": {"name": "d", "schema": {"enum": ["very good", "very bad", "fine"]}}}
    body = {"messages": [{"role": "user", "content": "How was the movie? It was a total waste of time."}],
            "response_format": deep, "logprobs": True}
    first = client.post("/v1/chat/completions", json=body).json()["choices"][0]["logprobs"]["content"][0]
    again = client.post("/v1/chat/completions", json=body).json()["choices"][0]["logprobs"]["content"][0]
    lp = {t["token"]: t["logprob"] for t in first["top_logprobs"]}
    assert abs(sum(math.exp(v) for v in lp.values()) - 1) < 1e-4
    assert all(abs(lp[t["token"]] - t["logprob"]) < 1e-3 for t in again["top_logprobs"])
    gen = ask(client, body["messages"][0]["content"], response_format=deep)  # (the model's own pick, generated)
    assert json.loads(gen["message"]["content"]) == first["token"]
