"""artlite-server on the host (no NPU): its routes through FastAPI's TestClient and a live uvicorn (the ``openai``
client), with the scripted fake model of ``fakes``. Serving the fake model with the web UI, for a manual check in a
browser:

    python -m tests.server.test_server [--port 8080] [--thinking]
"""

import argparse
import base64
import json
import threading
import time
import types
from collections.abc import Iterator

import httpx
import numpy as np
import openai
import pytest
import uvicorn
from fastapi.testclient import TestClient

from models.common import MEDIA_MARKER, Sampling
from models.sequence import MEDIA
from models.server import ui
from models.server.engine import Engine
from models.server.jobs import Job

from .fakes import (
    BOS,
    ENGINES,
    GEMMA,
    QWEN,
    TEMPLATE,
    TIME,
    TOOLS_TEMPLATE,
    VOCAB,
    WEATHER,
    FakeEncoder,
    FakeLive,
    FakeModel,
    FakeNpu,
    FakeSpeaker,
    FakeVideo,
    FakeVision,
    app_of,
    call_pieces,
    chat,
    deltas,
    events,
    live,
    make,
)


@pytest.fixture
def fake() -> Iterator[tuple[FakeModel, FakeNpu, Engine, TestClient]]:
    model, npu, engine = make()
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        yield model, npu, engine, client


# ----------------------------------------------------------------------
# Streams and responses
# ----------------------------------------------------------------------


def test_stream(fake: tuple) -> None:
    _model, npu, engine, client = fake
    npu.replies = [["Hello", " world", "!"]]
    r = chat(client, "Hi", stream=True, return_progress=True, timings_per_token=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = events(r)
    assert evs[-1] == "[DONE]"
    first = evs[0]
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"] == {"role": "assistant",
                                                                                           "content": None}
    assert deltas(evs) == "Hello world!"
    progress = [e["prompt_progress"] for e in evs[:-1] if "prompt_progress" in e]
    assert progress[0]["processed"] == 0 and progress[-1]["processed"] == progress[-1]["total"]
    end = evs[-2]
    assert end["choices"][0]["finish_reason"] == "stop" and end["choices"][0]["delta"] == {}
    t = end["timings"]
    assert t["predicted_n"] == 4 and t["prompt_n"] == progress[-1]["total"] and t["cache_n"] == 0
    assert all("timings" in e for e in evs[1:-1] if isinstance(e, dict) and e["choices"][0]["delta"].get("content"))
    assert len({e["id"] for e in evs[:-1]}) == 1 and all(e["model"] == "fake" for e in evs[:-1])
    assert engine.cached == npu.seq[:npu.position]


def test_response_and_usage(fake: tuple) -> None:
    model, npu, _engine, client = fake
    npu.replies = [["One", " two", " three", " four"]]
    r = chat(client, "Count", max_tokens=3)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "One two three"}
    assert body["choices"][0]["finish_reason"] == "length"
    u = body["usage"]
    assert u["completion_tokens"] == 3 and u["total_tokens"] == u["prompt_tokens"] + 3
    assert u["prompt_tokens"] == len(model.chat([{"role": "user", "content": "Count"}]))
    evs = events(chat(client, "Count", stream=True, stream_options={"include_usage": True}))
    assert evs[-2]["choices"] == [] and evs[-2]["usage"]["prompt_tokens"] == u["prompt_tokens"]
    assert evs[-3]["choices"][0]["finish_reason"] == "stop"


def test_completions(fake: tuple) -> None:
    _model, npu, _engine, client = fake
    npu.replies = [["A", "B"], [" C"], [" D"]]
    r = client.post("/v1/completions", json={"prompt": "Once upon"})
    body = r.json()
    assert body["object"] == "text_completion" and body["choices"][0]["text"] == "AB"
    assert body["usage"]["prompt_tokens"] == 3  # (BOS and two words)
    r = client.post("/completion", json={"prompt": [BOS, 5], "n_predict": 4})
    body = r.json()
    assert body["content"] == " C" and body["stop"] is True and body["stop_type"] == "eos"
    assert body["tokens_evaluated"] == 2 and body["tokens_cached"] == npu.position
    lines = [json.loads(b[6:]) for b in client.post("/completion", json={"prompt": "x", "stream": True}).text.split(
        "\n\n") if b.startswith("data: ")]
    assert lines[0]["content"] == " D" and lines[-1]["stop"] is True  # (no [DONE] in the native stream)


def test_penalties() -> None:
    """The penalties of the command line and of a request, over the prompt's tokens too: a reply of a prompt's word
    whose logit the frequency penalty pushes below the others' gives another token."""
    _model, npu, engine = make()
    with TestClient(app_of(engine, repeat_penalty=1.2)) as client:
        assert engine.wait_ready(5)
        assert client.get("/props").json()["default_generation_settings"]["params"]["repeat_penalty"] == 1.2
        body = {"prompt": "hello there", "temperature": 0}
        npu.replies = [["hello"]]
        r = client.post("/completion", json=body | {"repeat_last_n": 16, "presence_penalty": 0.5}).json()
        g = r["generation_settings"]
        assert r["content"] == "hello"  # (a penalized logit of 0 stays 0)
        assert (g["repeat_penalty"], g["repeat_last_n"], g["presence_penalty"], g["frequency_penalty"]) == \
            (1.2, 16, 0.5, 0.0)
        npu.replies = [["hello"]]
        assert client.post("/completion", json=body | {"frequency_penalty": 150}).json()["content"] != "hello"
        npu.replies = [["hello"]]
        assert client.post("/completion", json=body | {"frequency_penalty": 150, "repeat_last_n": 0}
                           ).json()["content"] == "hello"


# ----------------------------------------------------------------------
# Reasoning, stop strings
# ----------------------------------------------------------------------


def test_reasoning_qwen() -> None:
    """The prompt opens the thought; the model closes it."""
    _model, npu, engine = make(reasoning=QWEN)
    with TestClient(app_of(engine)) as client:
        npu.replies = [["Let", " me", " think", "\n", "</think>", "\n\n", "The", " answer"]] * 3
        on = {"chat_template_kwargs": {"enable_thinking": True}}
        evs = events(chat(client, "Q", stream=True, **on))
        assert deltas(evs, "reasoning_content") == "Let me think" and deltas(evs) == "The answer"
        body = chat(client, "Q", **on).json()["choices"][0]["message"]
        assert body == {"role": "assistant", "content": "The answer", "reasoning_content": "Let me think"}
        raw = chat(client, "Q", reasoning_format="none", **on).json()["choices"][0]["message"]
        assert raw["content"] == "Let me think\n</think>\n\nThe answer" and "reasoning_content" not in raw
        assert "<think>\n\n</think>" in client.post("/apply-template", json={
            "messages": [{"role": "user", "content": "Q"}]}).json()["prompt"]  # (-rea off: an empty thought)


def test_reasoning_gemma() -> None:
    """The model opens and closes the thought, and may end with a stray close marker."""
    _model, npu, engine = make(reasoning=GEMMA)
    with TestClient(app_of(engine, reasoning="on")) as client:
        npu.replies = [["<|channel>thought", "\n", "Hmm", ",", " yes", "\n", "<channel|>", "Yes", ".", "<channel|>"]]
        evs = events(chat(client, "Q", stream=True))
        assert deltas(evs, "reasoning_content") == "Hmm, yes" and deltas(evs) == "Yes."


def test_preserve_reasoning() -> None:
    """The earlier answers' thoughts stay in the prompt (the templates' preserve_thinking) unless the server's option
    or a request's chat_template_kwargs (preserve_reasoning, llama-server's, or preserve_thinking) says otherwise."""
    model, _npu, engine = make(reasoning=QWEN)
    for options, kwargs, want in [({}, None, True), ({"preserve_reasoning": False}, None, False),
                                  ({}, {"preserve_reasoning": False}, False), ({}, {"preserve_thinking": False}, False),
                                  ({"preserve_reasoning": False}, {"preserve_thinking": True}, True),
                                  ({"chat_template_kwargs": {"preserve_reasoning": False}}, None, False),
                                  ({"chat_template_kwargs": {"preserve_reasoning": False}},
                                   {"preserve_thinking": True}, True)]:
        with TestClient(app_of(engine, **options)) as client:
            assert engine.wait_ready(5)
            body = {"chat_template_kwargs": kwargs} if kwargs is not None else {}
            assert chat(client, "Hi", **body).status_code == 200
            assert model.preserved is want, (options, kwargs)
            client.post("/apply-template", json={"messages": [{"role": "user", "content": "Hi"}], **body})
            assert model.preserved is want, (options, kwargs)


def test_template_variables() -> None:
    """A request's chat_template_kwargs reach the template as its variables, over the server's (but enable_thinking
    and the preserve switches, which are the model's, and the conversation's own names). reasoning_effort, the
    server's --reasoning-effort or a request's field (as llama-server's; the Responses API's reasoning.effort):
    none turns the thought off, another level on and is the template's variable; a level from a request's field that
    the template refuses gives way to the server's, else the template's default; one in chat_template_kwargs fails."""
    model, _npu, engine = make(reasoning=QWEN)
    model.efforts = ("xhigh", "medium", "low")
    hi = [{"role": "user", "content": "Hi"}]
    with TestClient(app_of(engine, chat_template_kwargs={"add_vision_id": True, "x": 1})) as client:
        assert engine.wait_ready(5)
        kw = {"x": 2, "tools_in_user_message": False, "messages": [], "add_generation_prompt": False,
              "enable_thinking": True, "preserve_thinking": False}
        assert chat(client, "Hi", chat_template_kwargs=kw).status_code == 200
        assert model.variables == {"add_vision_id": True, "x": 2, "tools_in_user_message": False}
        assert model.thought is True and model.preserved is False and model.rendered[0] == hi
        assert chat(client, "Hi", reasoning_effort="low").status_code == 200
        assert model.variables == {"add_vision_id": True, "x": 1, "reasoning_effort": "low"} and model.thought is True
        assert chat(client, "Hi", reasoning_effort="none").status_code == 200
        assert "reasoning_effort" not in model.variables and model.thought is False
        assert chat(client, "Hi", reasoning_effort="low", chat_template_kwargs={"enable_thinking": False}).status_code \
            == 200 and model.thought is False  # (chat_template_kwargs over the field)
        assert chat(client, "Hi", reasoning_effort="high").status_code == 200
        assert "reasoning_effort" not in model.variables and model.thought is True
        r = chat(client, "Hi", chat_template_kwargs={"reasoning_effort": "high"})
        assert r.status_code == 400 and "Unexpected reasoning effort high" in r.json()["error"]["message"]
        client.post("/apply-template", json={"messages": hi, "reasoning_effort": "medium"})
        assert model.variables["reasoning_effort"] == "medium"
        assert client.post("/v1/responses", json={"input": "Hi", "reasoning": {"effort": "low"}}).status_code == 200
        assert model.variables["reasoning_effort"] == "low" and model.thought is True
    with TestClient(app_of(engine, reasoning_effort="medium")) as client:
        assert chat(client, "Hi").status_code == 200
        assert model.variables == {"reasoning_effort": "medium"} and model.thought is False  # (-rea off)
        assert chat(client, "Hi", reasoning_effort="high").status_code == 200
        assert model.variables == {"reasoning_effort": "medium"} and model.thought is True
        assert chat(client, "Hi", reasoning_effort="none").status_code == 200 and model.variables == {}


def test_chat_template_option() -> None:
    """The server's chat template in place of the file's (--chat-template, ModelInfo's): /props reports it and the
    template's tool-call syntax, content parts and system messages are read from it."""
    model, _npu, engine = make()
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        assert client.get("/props").json()["chat_template_caps"]["supports_tools"] is False
    template = "{%- for m in messages %}{{ m.content if m.content is string else m.content|map(attribute='text')|" \
               "join }}{% endfor %}{# <tool_call> <function= <parameter= #}"
    with TestClient(app_of(engine, template=template)) as client:
        p = client.get("/props").json()
        assert p["chat_template"] == template and p["chat_template_caps"]["supports_tools"] is True
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert chat(client, [{"role": "user", "content": parts}]).status_code == 200
        assert model.rendered[0][0]["content"] == parts  # (the template joins the parts itself)


def test_stop_strings(fake: tuple) -> None:
    _model, npu, engine, client = fake
    npu.replies = [["Hello", " wor", "ld", " STO", "P", " more"], ["ST", "ART", " ok"]]
    body = chat(client, "Hi", stop="STOP").json()
    assert body["choices"][0]["message"]["content"] == "Hello world " and body["choices"][0]["finish_reason"] == "stop"
    assert npu.position == len(engine.cached) and engine.cached == npu.seq[:npu.position]
    evs = events(chat(client, "Hi", stream=True, stop=["STOP"]))
    assert deltas(evs) == "START ok"


def test_stop_strings_drafts() -> None:
    """A stop string inside a step of several tokens: the cache holds the tokens the step ran past it too."""
    _model, npu, engine = make(drafts=3)
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        npu.replies = [["Hello", " STO", "P", " more", " text", " end"]]
        body = chat(client, "Hi", stop="STOP").json()
        assert body["choices"][0]["message"]["content"] == "Hello "
        assert npu.position == len(engine.cached) and engine.cached == npu.seq[:npu.position]


# ----------------------------------------------------------------------
# Tool calls
# ----------------------------------------------------------------------

@pytest.fixture
def tooled() -> Iterator[tuple[FakeModel, FakeNpu, Engine, TestClient]]:
    model, npu, engine = make(reasoning=QWEN)
    with TestClient(app_of(engine, template=TOOLS_TEMPLATE)) as client:
        assert engine.wait_ready(5)
        yield model, npu, engine, client


def streamed_calls(evs: list[object]) -> list[dict[str, object]]:
    """The tool calls of a stream's deltas, joined as clients join them."""
    calls: dict[int, dict[str, object]] = {}
    for e in evs:
        if not isinstance(e, dict) or not e.get("choices"):
            continue
        for d in e["choices"][0]["delta"].get("tool_calls") or []:
            if "id" in d:
                assert d["index"] not in calls and d["type"] == "function" and d["function"]["name"]
                calls[d["index"]] = {"id": d["id"], "name": d["function"]["name"], "arguments": ""}
            else:
                assert set(d) == {"index", "function"} and set(d["function"]) == {"arguments"}
            calls[d["index"]]["arguments"] += d["function"]["arguments"]
    return [calls[i] for i in sorted(calls)]


def test_tool_calls(tooled: tuple) -> None:
    """A call after content and thoughts, streamed and whole; the tools reach the template."""
    model, npu, _engine, client = tooled
    reply = ["Checking", " the", " sky", "\n", "</think>", "\n\n", "Let", " me", " look.", "\n\n"] + \
        call_pieces("get_weather", city="Paris", days="3")
    npu.replies = [reply, reply]
    body = {"tools": [WEATHER, TIME], "chat_template_kwargs": {"enable_thinking": True}}
    r = chat(client, "Weather?", **body).json()
    msg = r["choices"][0]["message"]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["content"] == "Let me look." and msg["reasoning_content"] == "Checking the sky"
    (call,) = msg["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'}
    assert [t["function"]["name"] for t in model.rendered[1]] == ["get_weather", "get_time"]
    assert model.rendered[1][1]["function"] == {"name": "get_time", "description": "",
                                                 "parameters": {"type": "object", "properties": {}}}
    evs = events(chat(client, "Weather?", stream=True, **body))
    assert deltas(evs) == "Let me look." and deltas(evs, "reasoning_content") == "Checking the sky"
    (c,) = streamed_calls(evs)
    assert (c["name"], json.loads(c["arguments"])) == ("get_weather", {"city": "Paris", "days": 3})
    assert evs[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_calls_parallel(tooled: tuple) -> None:
    """Two calls in one answer, or the first alone (parallel_tool_calls false: the generation ends after it)."""
    _model, npu, _engine, client = tooled
    reply = call_pieces("get_weather", city="Oslo") + ["\n"] + call_pieces("get_time") + ["\n", "Extra"]
    npu.replies = [reply, reply]
    msg = chat(client, "Q", tools=[WEATHER, TIME]).json()["choices"][0]["message"]
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in msg["tool_calls"]] == \
        [("get_weather", '{"city": "Oslo"}'), ("get_time", "{}")]
    assert msg["content"] == "Extra"
    r = chat(client, "Q", tools=[WEATHER, TIME], parallel_tool_calls=False).json()
    assert [c["function"]["name"] for c in r["choices"][0]["message"]["tool_calls"]] == ["get_weather"]
    assert r["choices"][0]["message"]["content"] is None and r["usage"]["completion_tokens"] < len(reply)


def test_tool_choice(tooled: tuple) -> None:
    """none: the call stays text; required and a named function: the call's opener is forced, after the thought when
    the model thinks (the generation runs it and continues)."""
    model, npu, engine, client = tooled
    raw = call_pieces("get_time")
    npu.replies = [raw]
    msg = chat(client, "Q", tools=[TIME], tool_choice="none").json()["choices"][0]["message"]
    assert "tool_calls" not in msg and msg["content"].startswith("<tool_call>")
    assert model.rendered[1] is not None  # (the tools stay in the prompt)
    npu.replies = [call_pieces("get_time")[1:]]  # (after the forced "<tool_call>\n<function=")
    npu.replies[0][0] = "get_time>"
    r = chat(client, "Q", tools=[WEATHER, TIME], tool_choice="required").json()
    assert [c["function"]["name"] for c in r["choices"][0]["message"]["tool_calls"]] == ["get_time"]
    assert ("<|assistant|><think>\n\n</think>\n\n<tool_call>\n<function=get_time>\n</function>"
            in model.tokenizer.decode_bytes(npu.seq).decode())
    npu.replies = [["Need", " Oslo", "</think>", "No", " call"], ["\n<parameter=city>", "\nOslo", "\n</parameter>",
                                                                 "\n</function>", "\n</tool_call>"]]
    r = chat(client, "Q", tools=[WEATHER, TIME], tool_choice={"type": "function", "function": {"name": "get_weather"}},
             chat_template_kwargs={"enable_thinking": True}).json()
    msg = r["choices"][0]["message"]
    assert msg["reasoning_content"] == "Need Oslo" and msg["content"] is None
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in msg["tool_calls"]] == \
        [("get_weather", '{"city": "Oslo"}')]
    text = model.tokenizer.decode_bytes(npu.seq).decode()
    assert "Need Oslo</think>\n\n<tool_call>\n<function=get_weather>\n" in text and "No call" not in text
    assert engine.cached == npu.seq[:npu.position]
    npu.replies = [call_pieces("get_time")[2:] + ["\n"] + call_pieces("get_weather", city="X")]  # (a named call alone)
    r = chat(client, "Q", tools=[WEATHER, TIME], tool_choice={"type": "function", "function": {"name": "get_time"}})
    assert [c["function"]["name"] for c in r.json()["choices"][0]["message"]["tool_calls"]] == ["get_time"]


def test_tool_choice_own_thought() -> None:
    """A model that opens its thought itself (Gemma): the forced opener follows the thought, or the answer's first
    text when it does not think."""
    model, npu, engine = make(reasoning=GEMMA)
    with TestClient(app_of(engine, template=TOOLS_TEMPLATE)) as client:
        assert engine.wait_ready(5)
        body = {"tools": [WEATHER], "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": True}}
        rest = call_pieces("get_weather", city="Rome")[1:]
        rest[0] = "get_weather>"
        npu.replies = [["<|channel>thought", "\n", "Hmm", "<channel|>", "Text"], rest]
        msg = chat(client, "Q", **body).json()["choices"][0]["message"]
        assert msg["reasoning_content"] == "Hmm" and msg["content"] is None
        assert [c["function"]["arguments"] for c in msg["tool_calls"]] == ['{"city": "Rome"}']
        assert "Hmm<channel|><tool_call>\n<function=get_weather>" in model.tokenizer.decode_bytes(npu.seq).decode()
        npu.replies = [["Sure", ",", " later"], rest]
        msg = chat(client, "Q", **body).json()["choices"][0]["message"]
        assert msg["content"] == "Sure" and [c["function"]["name"] for c in msg["tool_calls"]] == ["get_weather"]
        assert engine.cached == npu.seq[:npu.position]


def test_tool_call_names(tooled: tuple) -> None:
    """A call's name is one of the functions': where the model writes another (none fits), the tokens that write a
    function's name win; the arguments stay the model's."""
    model, npu, _engine, client = tooled
    for piece in ("get_time>", "\n<function=get_weather>"):  # (the vocabulary's: before the first request with tools)
        model.tokenizer.id(piece)
    npu.replies = [["none>", "\n</function>", "\n</tool_call>"]]  # (after the forced "<tool_call>\n<function=")
    r = chat(client, "Hi!", tools=[WEATHER, TIME], tool_choice="required").json()
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in r["choices"][0]["message"]["tool_calls"]] == \
        [("get_time", "{}")]
    npu.replies = [["Sure.", "\n\n", "<tool_call>", "\n<function=lookup>", "\n<parameter=city>", "\nOslo",
                    "\n</parameter>", "\n</function>", "\n</tool_call>"]]
    msg = chat(client, "Hi!", tools=[WEATHER, TIME]).json()["choices"][0]["message"]
    assert msg["content"] == "Sure." and [(c["function"]["name"], c["function"]["arguments"])
                                          for c in msg["tool_calls"]] == [("get_weather", '{"city": "Oslo"}')]
    assert "<tool_call>\n<function=get_weather>\n<parameter=city>" in model.tokenizer.decode_bytes(npu.seq).decode()


def test_tool_calls_drafts() -> None:
    """Steps of several tokens (a drafter): the forced opener and the end after a call wait for a step's end, so the
    cache holds exactly the tokens of the answer."""
    _model, npu, engine = make(reasoning=QWEN, drafts=3)
    with TestClient(app_of(engine, template=TOOLS_TEMPLATE)) as client:
        assert engine.wait_ready(5)
        npu.replies = [["Need", " Oslo", "</think>", "\n\n", "Fine."], call_pieces("get_weather", city="Oslo")[2:]]
        r = chat(client, "Q", tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "get_weather"}},
                 chat_template_kwargs={"enable_thinking": True}).json()
        msg = r["choices"][0]["message"]
        assert msg["reasoning_content"] == "Need Oslo" and msg["content"] == "Fine."
        assert [c["function"]["arguments"] for c in msg["tool_calls"]] == ['{"city": "Oslo"}']
        assert engine.cached == npu.seq[:npu.position] and npu.position == len(engine.cached)
        reply = call_pieces("get_time") + ["\n"] + call_pieces("get_weather", city="X")
        npu.replies = [reply]
        r = chat(client, "Q", tools=[WEATHER, TIME], parallel_tool_calls=False).json()
        assert [c["function"]["name"] for c in r["choices"][0]["message"]["tool_calls"]] == ["get_time"]
        assert engine.cached == npu.seq[:npu.position] and npu.position == len(engine.cached)


def test_tool_messages(tooled: tuple) -> None:
    """A conversation with calls and their results: arguments as objects for the template."""
    model, npu, _engine, client = tooled
    messages = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather",
                                                                      "arguments": '{"city": "Paris"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}]
    npu.replies = [["Sunny", " in", " Paris."]]
    msg = chat(client, messages, tools=[WEATHER]).json()["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "Sunny in Paris."}
    rendered = model.rendered[0]
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert rendered[2] == {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}
    prompt = client.post("/apply-template", json={"messages": messages, "tools": [WEATHER]}).json()["prompt"]
    assert 'call get_weather {"city": "Paris"}' in prompt and "tools: get_weather" in prompt


def test_content_parts() -> None:
    """A message's text parts go to a template that renders lists as a list (Qwen's, Gemma's), else joined by
    newlines."""
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    listing = "{% for m in messages %}{% for p in m.content %}{{ p.text }}{% endfor %}{% endfor %}"
    for template, want in [(listing, parts), (TEMPLATE, "a\nb")]:
        model, _npu, engine = make()
        with TestClient(app_of(engine, template=template)) as client:
            assert chat(client, [{"role": "user", "content": parts}]).status_code == 200
            assert model.rendered[0][0]["content"] == want


def test_tool_errors(tooled: tuple, fake: tuple) -> None:
    client = tooled[3]
    bad = [({"tools": [{"type": "function", "function": {"description": "no name"}}]}, "a tool must be"),
           ({"tools": [WEATHER], "tool_choice": "any"}, "tool_choice"),
           ({"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "nope"}}}, "not one of"),
           ({"tool_choice": "required"}, "without tools")]
    for body, says in bad:
        r = chat(client, "Q", **body)
        assert r.status_code == 400 and says in r.json()["error"]["message"], body
    r = chat(client, [{"role": "user", "content": "Q"}, {"role": "assistant", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{oops"}}]}], tools=[WEATHER])
    assert r.status_code == 400 and "not JSON" in r.json()["error"]["message"]
    fake[1].replies = [call_pieces("get_weather", city="X")]  # (a template without tools: they are ignored)
    msg = chat(fake[3], "Q", tools=[WEATHER]).json()["choices"][0]["message"]
    assert "tool_calls" not in msg and msg["content"].startswith("<tool_call>") and fake[0].rendered[1] is None
    r = chat(fake[3], "Q", tools=[WEATHER], tool_choice="required")
    assert r.status_code == 400 and "takes no tools" in r.json()["error"]["message"]
    assert fake[3].get("/props").json()["chat_template_caps"]["supports_tools"] is False
    assert client.get("/props").json()["chat_template_caps"] == {
        "supports_tools": True, "supports_tool_calls": True, "supports_parallel_tool_calls": True,
        "supports_preserve_reasoning": False}


def test_tool_calls_openai_client() -> None:
    """The openai package's stream of a call."""
    _model, npu, engine = make(reasoning=QWEN)
    with live(engine, template=TOOLS_TEMPLATE) as url:
        client = openai.OpenAI(base_url=f"{url}/v1", api_key="none")
        npu.replies = [call_pieces("get_weather", city="Rome", days="2")] * 2
        args = {"model": "fake", "messages": [{"role": "user", "content": "Rome?"}], "tools": [WEATHER]}
        calls: dict[int, dict[str, str]] = {}
        finish = None
        for c in client.chat.completions.create(**args, stream=True):
            for d in (c.choices[0].delta.tool_calls or []) if c.choices else []:
                e = calls.setdefault(d.index, {"name": "", "arguments": ""})
                e["name"] += d.function.name or ""
                e["arguments"] += d.function.arguments or ""
            finish = c.choices[0].finish_reason if c.choices else finish
        assert finish == "tool_calls" and calls == {0: {"name": "get_weather", "arguments": '{"city": "Rome", "days": 2}'}}
        r = client.chat.completions.create(**args)
        assert r.choices[0].message.tool_calls[0].function.name == "get_weather"


# ----------------------------------------------------------------------
# The prompt cache
# ----------------------------------------------------------------------


def test_prompt_cache(fake: tuple) -> None:
    model, npu, _engine, client = fake
    turn1 = [{"role": "user", "content": "one two three four five six seven"}]
    npu.replies = [["Reply", " one"], ["Reply", " two"], ["x"], ["y"]]
    b1 = chat(client, turn1).json()
    assert b1["timings"]["cache_n"] == 0
    turn2 = turn1 + [{"role": "assistant", "content": "Reply one"}, {"role": "user", "content": "more"}]
    b2 = chat(client, turn2).json()
    ids2 = model.chat(turn2)
    kept = len(model.chat(turn1)) + 2  # (the reply's tokens ran; its end-of-sequence token did not)
    assert b2["timings"]["cache_n"] == kept and not [c for c in npu.calls if c[0] == "rewind"]
    assert b2["timings"]["prompt_n"] == len(ids2) - kept and b2["usage"]["prompt_tokens_details"]["cached_tokens"] == kept
    assert npu.seq[:len(ids2)] == ids2
    # another answer to it: the multiple of the group below its last token (which runs again)
    b3 = chat(client, turn2).json()
    n = len(ids2) - 1 - (len(ids2) - 1) % npu.group
    assert npu.calls[-1 - [c[0] for c in reversed(npu.calls)].index("rewind")] == ("rewind", n, n)
    assert b3["timings"]["cache_n"] == n and b3["timings"]["prompt_n"] == len(ids2) - n
    # without the cache
    b4 = chat(client, turn2, cache_prompt=False).json()
    assert b4["timings"]["cache_n"] == 0 and ("restore", 0, False) in npu.calls[-3:]


def _image(data: bytes) -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def test_images() -> None:
    """image_url parts (data URIs) reach the model's chat as image parts, on the NPU worker (the encoder's dispatches
    between jobs); the prompt cache tells images apart by their rows; vision in /props and /v1/models."""
    model, _npu, engine = make(vision=True)
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        turn1 = [{"role": "user", "content": [{"type": "text", "text": "look"}, _image(b"cat" * 8),
                                              {"type": "text", "text": "what is it"}]}]
        b1 = chat(client, turn1).json()
        assert b1["timings"]["cache_n"] == 0 and model.threads == ["artlite-npu"]
        turn2 = turn1 + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        b2 = chat(client, turn2).json()
        assert b2["timings"]["cache_n"] >= 8  # (past the image)
        other = [{"role": "user", "content": [{"type": "text", "text": "look"}, _image(b"dog" * 8),
                                              {"type": "text", "text": "what is it"}]}]
        b3 = chat(client, other).json()
        assert b3["timings"]["cache_n"] <= 3 and len(model.threads) == 3  # (the image's rows differ: cut before it)
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]}).status_code == 400
        p = client.get("/props").json()
        assert p["modalities"]["vision"] is True
        m = client.get("/v1/models").json()
        assert "multimodal" in m["models"][0]["capabilities"] and "image" in m["data"][0]["architecture"]["input_modalities"]
        prompt = client.post("/apply-template", json={"messages": turn1}).json()["prompt"]
        assert prompt == f"<s><|user|>look\n{MEDIA_MARKER}\nwhat is it<|assistant|>"


def test_videos() -> None:
    """input_video parts (llama-server's: data, base64 or a data URL) and video_url ones reach the model's chat as video
    parts on the NPU worker; video in /props and /v1/models; a model without video input refuses them."""
    clip = b"mp4" * 8
    parts = [{"type": "input_video", "input_video": {"data": base64.b64encode(clip).decode()}},
             {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(clip).decode()}}]
    model, _npu, engine = make(vision=True)
    model.__class__ = FakeVideo
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        for part in parts:
            b = chat(client, [{"role": "user", "content": [part, {"type": "text", "text": "what happens"}]}])
            assert b.status_code == 200
            video = next(p for p in model.rendered[0][0]["content"] if p["type"] == "video")
            assert video["video"] == clip
        assert model.threads == ["artlite-npu"] * 2
        bad = {"type": "input_video", "input_video": {"data": "not base64!"}}
        assert chat(client, [{"role": "user", "content": [bad]}]).status_code == 400
        assert client.get("/props").json()["modalities"]["video"] is True
        m = client.get("/v1/models").json()
        assert m["data"][0]["architecture"]["input_modalities"] == ["text", "image", "video"]
        prompt = client.post("/apply-template", json={"messages": [{"role": "user", "content": [parts[0]]}]}).json()
        assert MEDIA_MARKER in prompt["prompt"]
    model, _npu, engine = make(vision=True)
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        r = chat(client, [{"role": "user", "content": [parts[0]]}])
        assert r.status_code == 400 and "images (image_url)" in r.json()["error"]["message"]
        assert client.get("/props").json()["modalities"]["video"] is False


def test_images_refused(fake: tuple) -> None:
    """A model without a vision encoder refuses images."""
    client = fake[3]
    r = chat(client, [{"role": "user", "content": [_image(b"x"), {"type": "text", "text": "hi"}]}])
    assert r.status_code == 400 and "text only" in r.json()["error"]["message"]


def test_audio() -> None:
    """A model with a speech encoder: input_audio parts and audio_url data URIs in chat (the encoder on the NPU
    worker), /v1/audio/transcriptions (json, text, a stream: llama-server's shapes; the ASR system message, no
    thought), the openai client's transcriptions; audio in /props and /v1/models; audio output refused."""
    from models.lfm3a.host import wav_bytes

    model, npu, engine = make(audio=True)
    clip = wav_bytes(np.linspace(-0.5, 0.5, 1600))
    data = base64.b64encode(clip).decode()
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        b = chat(client, [{"role": "user", "content": [
            {"type": "text", "text": "hear"}, {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}])
        assert b.status_code == 200 and model.threads == ["artlite-npu"]
        b = chat(client, [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + data}}]}])
        assert b.status_code == 200 and len(model.threads) == 2
        r = chat(client, [{"role": "user", "content": [_image(b"x")]}])
        assert r.status_code == 400 and "audio (input_audio)" in r.json()["error"]["message"]
        assert chat(client, "hi", modalities=["text", "audio"]).status_code == 501
        npu.replies = [["hello", " world"]]
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", clip, "audio/wav")},
                        data={"language": "de", "model": "fake"})
        j = r.json()
        assert r.status_code == 200 and j["type"] == "transcript.text.done" and j["text"] == "hello world"
        assert j["usage"]["output_tokens"] == 3 and j["usage"]["type"] == "tokens" and model.thought is False
        system = model.rendered[0][0]
        assert system["role"] == "system" and "German" in str(system["content"])
        npu.replies = [["hi"]]
        r = client.post("/audio/transcriptions", files={"file": ("a.wav", clip)}, data={"response_format": "text"})
        assert r.status_code == 200 and r.text == "hi"
        npu.replies = [["a", " b"]]
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", clip)}, data={"stream": "true"})
        evs = [e for e in events(r) if isinstance(e, dict)]
        assert [e["delta"] for e in evs if e["type"] == "transcript.text.delta"] == ["a", " b"]
        assert evs[-1]["type"] == "transcript.text.done" and evs[-1]["text"] == "a b"
        assert client.post("/v1/audio/transcriptions", data={"language": "en"}).status_code == 400
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"not audio")})
        assert r.status_code == 400 and "decode" in r.json()["error"]["message"]
        p = client.get("/props").json()
        assert p["modalities"]["audio"] is True and p["modalities"]["vision"] is False
        m = client.get("/v1/models").json()
        assert m["data"][0]["architecture"]["input_modalities"] == ["text", "audio"]
    with live(engine) as url:
        npu.replies = [["via", " openai"]]
        c = openai.OpenAI(base_url=f"{url}/v1", api_key="none")
        t = c.audio.transcriptions.create(model="fake", file=("a.wav", clip), language="en")
        assert t.text == "via openai"


def test_realtime(monkeypatch: pytest.MonkeyPatch, fake: tuple) -> None:
    """/v1/realtime?intent=transcription (llama-server's live transcription events) over ``LiveAsr``'s steps on the
    worker; a model without live ASR, a session without the intent: an error event, closed."""
    monkeypatch.setattr("models.lfm3a.live.LiveAsr", FakeLive)
    _model, _npu, engine = make(audio=True)
    pcm = base64.b64encode((np.full(1600, 0.25) * 32768).astype("<i2").tobytes()).decode()
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
            ev = ws.receive_json()
            assert ev["type"] == "session.created" and ev["session"]["audio"]["input"]["format"]["rate"] == 16000
            ws.send_json({"type": "session.update", "session": {"type": "transcription"}})
            assert ws.receive_json()["type"] == "session.updated"
            ws.send_json({"type": "input_audio_buffer.append", "audio": pcm})
            ev = ws.receive_json()
            assert ev["type"] == "conversation.item.input_audio_transcription.delta" and ev["delta"] == "[1600 0.25]"
            item = ev["item_id"]
            ws.send_json({"type": "input_audio_buffer.commit"})
            evs = [ws.receive_json() for _ in range(3)]
            assert [e["type"] for e in evs] == ["input_audio_buffer.committed",
                                                "conversation.item.input_audio_transcription.delta",
                                                "conversation.item.input_audio_transcription.completed"]
            assert evs[0]["item_id"] == item and evs[2]["transcript"] == "[1600 0.25]|"
            ws.send_json({"type": "nonsense"})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "input_audio_buffer.commit", "final": True})
            evs = [ws.receive_json() for _ in range(2)]
            assert evs[1]["type"] == "conversation.item.input_audio_transcription.completed"
            assert evs[1]["transcript"] == "." and evs[1]["item_id"] != item
        assert FakeLive.threads and set(FakeLive.threads) == {"artlite-npu"}
    client = fake[3]  # (a model without live ASR)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error" and "does not support" in ev["error"]["message"]


def test_live_hold() -> None:
    """``Engine.hold`` / ``ran`` around a live stream's chunks: the device's own sequence runs nothing again; after a
    request displaced it, the prompt cache's copy (a sequence of more than MIN_SAVE positions) comes back."""
    model, npu, engine = make(audio=True, capacity=4096, exact=True)
    assert engine.wait_ready(5)
    words = model.tokenizer.encode(" ".join(f"w{i % 50}" for i in range(700)))

    def chunk(prev: list[int], seq: list[int]) -> None:  # (seq: prev and the chunk's tokens)
        npu.forward(seq[engine.hold(seq):])
        engine.ran(seq)

    run = lambda f: engine.live(f).result(10)
    run(lambda: chunk([], words[:600]))
    n = len(npu.calls)
    run(lambda: chunk(words[:600], words[:650]))
    assert npu.calls[n:] == [("forward", 50)] and npu.seq == words[:650]
    with TestClient(app_of(engine)) as client:
        assert chat(client, "something else").status_code == 200
    n = len(npu.calls)
    run(lambda: chunk(words[:650], words[:700]))
    assert npu.calls[n:][-1] == ("forward", 50) and not any(c[0] == "prompt" for c in npu.calls[n:])
    assert npu.seq == words[:700]


def test_audio_refused(fake: tuple) -> None:
    """A model without a speech encoder refuses audio parts and transcriptions."""
    client = fake[3]
    r = chat(client, [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA=="}}]}])
    assert r.status_code == 400 and "text only" in r.json()["error"]["message"]
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF")})
    assert r.status_code == 501


def test_prompt_images() -> None:
    """Model.prompt places images at the text's media markers, else before it."""
    model = FakeVision()
    got = model.prompt(f"a {MEDIA_MARKER} b {MEDIA_MARKER}", images=[b"1", b"2"])
    want = model.chat([{"role": "user", "content": [{"type": "text", "text": "a "}, {"type": "image", "image": b"1"},
                                                    {"type": "text", "text": " b "}, {"type": "image", "image": b"2"}]}])
    assert got.cache_keys() == want.cache_keys()
    first = model.prompt("describe", images=[b"1"])
    assert first.cache_keys() == model.chat([{"role": "user", "content": [
        {"type": "image", "image": b"1"}, {"type": "text", "text": "describe"}]}]).cache_keys()
    with pytest.raises(ValueError, match="marker"):
        model.prompt(f"a {MEDIA_MARKER}", images=[b"1", b"2"])
    with pytest.raises(ValueError, match="vision"):
        FakeModel().prompt("hi", images=[b"1"])


def test_prompt_cache_exact() -> None:
    """Qwen keeps an exact continuation of the cached tokens, else what a checkpoint holds."""
    model, npu, engine = make(exact=True)
    with TestClient(app_of(engine)) as client:
        npu.replies = [["a"], ["b"], ["c"]]
        ids = model.encode("one two three")
        first = client.post("/completion", json={"prompt": ids}).json()
        n = first["tokens_cached"]
        assert n == len(ids) + 1  # (the reply's token ran; the end-of-sequence token did not)
        cont = client.post("/completion", json={"prompt": ids + model.tokenizer.encode("a more")}).json()
        assert cont["timings"]["cache_n"] == n
        other = client.post("/completion", json={"prompt": model.encode("one two four")}).json()
        assert other["timings"]["cache_n"] == 0 and ("restore", 0, False) in npu.calls[-4:]  # (none below "four")


# ----------------------------------------------------------------------
# Cancelling, the queue
# ----------------------------------------------------------------------


def run_job(engine: Engine, ids: list[int], *, until: object = None, **kw: object) -> tuple[Job, list]:
    """A job submitted directly, its events collected; ``until(events)``: set its cancel flag once true."""
    got: list[tuple[str, dict]] = []
    done = threading.Event()

    def emit(kind: str, data: dict) -> None:
        got.append((kind, data))
        if until is not None and until(got):
            job.cancel.set()
        if kind in ("done", "error", "cancelled"):
            done.set()

    job = Job(ids=ids, emit=emit, sampling=Sampling(greedy=True), **kw)
    engine.submit(job)
    assert done.wait(10)
    return job, got


def test_cancel_between_tokens() -> None:
    model, npu, engine = make(delay=0.002)
    assert engine.wait_ready(5)
    npu.replies = [[f" w{i}" for i in range(100)]]
    ids = model.encode("tell me a long story")
    _job, got = run_job(engine, ids, until=lambda g: sum(k == "delta" for k, _ in g) == 5)
    assert got[-1][0] == "cancelled" and sum(k == "delta" for k, _ in got) == 5
    assert npu.position < len(ids) + 10 and engine.cached == npu.seq[:npu.position]
    npu.replies = [["fine"]]
    _, got = run_job(engine, ids + [5])
    assert got[-1][0] == "done" and got[-1][1]["timings"]["cache_n"] > 0  # (the next request continues)


def test_cancel_between_chunks() -> None:
    model, npu, engine = make(chunk=4, delay=0.002)
    assert engine.wait_ready(5)
    ids = model.encode(" ".join(f"w{i}" for i in range(40)))
    _job, got = run_job(engine, ids, until=lambda g: g[-1][0] == "progress" and g[-1][1]["processed"] >= 8,
                       progress=True)
    assert got[-1][0] == "cancelled" and npu.position == 8 and engine.cached == ids[:8]


def test_cancel_by_conversation(fake: tuple) -> None:
    """The UI's Stop: DELETE /v1/stream?conv_id= for the request with its X-Conversation-Id."""
    _model, npu, engine, client = fake
    npu.delay = 0.01
    npu.replies = [[f" w{i}" for i in range(500)]]
    out: dict[str, httpx.Response] = {}
    t = threading.Thread(target=lambda: out.setdefault("r", client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "go"}], "stream": True},
        headers={"X-Conversation-Id": "conv-1"})))
    t.start()
    while engine.current is None or engine.current.n_gen < 3:
        time.sleep(0.005)
    assert client.delete("/v1/stream", params={"conv_id": "conv-1"}).json()["cancelled"] == 1
    t.join(10)
    evs = events(out["r"])
    assert evs[-1] != "[DONE]" and evs[-1]["error"]["message"] == "the request was cancelled"
    assert 3 <= len(deltas(evs).split()) < 500


def test_fifo(fake: tuple) -> None:
    """Two requests at a time: the second waits (with pings), then runs."""
    _model, npu, engine, client = fake
    npu.delay = 0.01
    npu.replies = [[f" a{i}" for i in range(30)], [" b"]]
    out: dict[str, httpx.Response] = {}

    def post(key: str, text: str) -> None:
        out[key] = chat(client, text, stream=True)

    first = threading.Thread(target=post, args=("first", "one"))
    first.start()
    while engine.current is None:
        time.sleep(0.002)
    second = threading.Thread(target=post, args=("second", "two"))
    second.start()
    first.join(10)
    second.join(10)
    assert deltas(events(out["first"])).startswith(" a0 a1") and deltas(events(out["second"])) == " b"
    assert ":" in events(out["second"])  # (pings while it waited)
    forwards = [c for c in npu.calls if c[0] == "forward"]
    assert len(forwards) == 2


# ----------------------------------------------------------------------
# Errors, loading, keys
# ----------------------------------------------------------------------


def test_errors(fake: tuple) -> None:
    _model, npu, _engine, client = fake
    r = chat(client, " ".join(["word"] * 600))
    e = r.json()["error"]
    assert r.status_code == 400 and e["type"] == "exceed_context_size_error" and e["n_ctx"] == 512
    assert e["n_prompt_tokens"] > 512
    r = chat(client, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}])
    assert r.status_code == 400 and "text only" in r.json()["error"]["message"]
    assert chat(client, "x", n=2).status_code == 400
    assert chat(client, "x", model="gpt-4o").status_code == 200  # (as llama-server: any name)
    assert chat(client, "x", stream=None, n=None, return_progress=None).status_code == 200
    # values that sample nothing or overflow keep the most likely token, and the worker survives them
    for extreme in ({"top_p": 0.0}, {"min_p": 1.5}, {"top_k": 50000}, {"seed": 2**64 + 5}, {"temperature": -1}):
        npu.replies = [["fine"]]
        r = chat(client, "x", **extreme)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "fine", extreme
    assert client.post("/completion", json={"prompt": [BOS, -5, 3]}).status_code == 400
    assert client.post("/detokenize", json={"tokens": [VOCAB]}).json()["error"]["type"] == "invalid_request_error"
    assert client.post("/v1/chat/completions", json={"messages": "x"}).json()["error"]["type"] == \
        "invalid_request_error"
    assert client.post("/v1/embeddings", json={}).status_code == 501
    assert client.get("/tools").status_code == 403
    for bad in ({"repeat_penalty": 0}, {"repeat_last_n": -1}):
        r = chat(client, "x", **bad)
        assert r.status_code == 400 and next(iter(bad)) in r.json()["error"]["message"], bad
    # the text parts of a content list, and the fields the UI sends (accepted)
    npu.replies = [["fine"]]
    parts = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    r = chat(client, parts, stream=True, reasoning_format="auto", repeat_penalty=1.1, dry_multiplier=0.0,
             samplers=["top_k"], tools=[{"type": "function", "function": {"name": "get_datetime"}}],
             reasoning_control=True, sse_ping_interval=1)
    assert r.status_code == 200 and deltas(events(r)) == "fine"


def test_model_failure(fake: tuple) -> None:
    _model, npu, engine, client = fake
    npu.fail = RuntimeError("DispatchError: TIMEOUT")
    r = chat(client, "a b c d e f")
    assert r.status_code == 500 and r.json()["error"] == {"code": 500, "type": "server_error",
                                                          "message": "RuntimeError: DispatchError: TIMEOUT"}
    r = chat(client, "a b c d e f", stream=True, sse_ping_interval=5)  # (before the stream's first event: the same)
    assert r.status_code == 500 and r.json()["error"]["message"] == "RuntimeError: DispatchError: TIMEOUT"
    evs = events(chat(client, "a b c d e f", stream=True, return_progress=True))  # (after it: the stream's error)
    assert evs[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    assert evs[-1]["error"]["type"] == "server_error"
    npu.fail = None
    n = len(npu.calls)
    assert chat(client, "a").status_code == 200 and engine.cached == npu.seq[:npu.position]
    assert ("reset",) in npu.calls[n:]  # (after a failure the caches start over)

    def broken(position: int, checkpoint: object = None) -> None:
        raise RuntimeError("restore failed")

    npu.restore = broken  # (a failure before the prompt runs: the worker survives it too)
    assert chat(client, "a").json()["error"]["message"] == "RuntimeError: restore failed"
    del npu.restore
    npu.replies = [["back"]]
    assert chat(client, "a").json()["choices"][0]["message"]["content"] == "back"


def test_loading() -> None:
    gate = threading.Event()
    model = FakeModel()
    npu = FakeNpu(model)

    def load() -> tuple[FakeModel, FakeNpu]:
        gate.wait(10)
        return model, npu

    _, _, engine = make(load=load)
    with TestClient(app_of(engine)) as client:
        assert client.get("/health").status_code == 503
        r = client.get("/props")
        assert r.status_code == 503 and r.json()["error"]["message"] == "Loading model"
        assert chat(client, "x").status_code == 503
        assert client.get("/v1/models").status_code == 200
        gate.set()
        assert engine.wait_ready(5)
        assert client.get("/health").json() == {"status": "ok"}
    failed = []
    engine = Engine(lambda: 1 / 0, on_exit=lambda: failed.append(1))
    ENGINES.append(engine)
    engine.start()
    engine.join(5)
    assert failed and engine.error.startswith("ZeroDivisionError") and not engine.ready


def test_api_keys(tmp_path: object) -> None:
    _model, _npu, engine = make()
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<!doctype html><title>ui</title>")
    (ui_dir / "app.js").write_text("console.log(1)")
    app = app_of(engine, api_keys=frozenset({"k1", "k2"}), ui=lambda a: ui.mount(a, str(ui_dir)))
    with TestClient(app) as client:
        assert engine.wait_ready(5)
        r = client.get("/props")
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        assert client.get("/props", headers={"Authorization": "Bearer k2"}).status_code == 200
        assert client.get("/props", headers={"X-Api-Key": "k1"}).status_code == 200
        assert client.get("/props", headers={"Authorization": "Bearer k3"}).status_code == 401
        assert client.get("/health").status_code == 200
        r = client.get("/")
        assert r.status_code == 200 and "<title>ui</title>" in r.text and r.headers["cache-control"] == "no-cache"
        assert client.get("/app.js").text == "console.log(1)"
        assert client.get("/v1/nothing", headers={"X-Api-Key": "k1"}).status_code == 404
        r = client.options("/v1/chat/completions", headers={"Origin": "http://x", "Access-Control-Request-Method": "POST"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"


# ----------------------------------------------------------------------
# What the web UI reads, the utilities
# ----------------------------------------------------------------------


def test_props_and_models(fake: tuple) -> None:
    _model, npu, _engine, client = fake
    p = client.get("/props", params={"autoload": "false"}).json()
    g = p["default_generation_settings"]
    assert g["n_ctx"] == npu.capacity and g["params"]["temperature"] == 0.7 and g["params"]["top_k"] == 8
    assert g["params"]["top_p"] == 0.9 and g["params"]["min_p"] == 0.0 and g["params"]["max_tokens"] == -1
    assert g["params"]["repeat_penalty"] == 1.0 and g["params"]["repeat_last_n"] == 64
    assert g["params"]["samplers"][0] == "penalties"
    assert p["modalities"] == {"vision": False, "audio": False, "video": False} and "role" not in p
    assert p["chat_template"] == TEMPLATE and p["model_path"].endswith("fake.gguf") and p["total_slots"] == 1
    m = client.get("/v1/models").json()
    assert m["data"][0]["id"] == "fake" and m["data"][0]["meta"]["n_ctx_train"] == 4096 and m["models"]
    s = client.get("/slots").json()
    assert s[0]["n_ctx"] == npu.capacity and s[0]["is_processing"] is False
    assert client.post("/v1/streams/lookup", json={"conversation_ids": ["a"]}).json() == []
    assert client.get("/v1/stream", params={"conv_id": "a", "from": 0}).status_code == 404


def test_speculative_n_max() -> None:
    """A request's ``speculative.n_max`` (llama-server's field) caps its generation's drafts per step (``generate``'s
    ``drafts``); /props and a running job's /slots report the cap, at most the model's (``NpuModel.DRAFTS``); llama-server's
    other speculative fields are ignored; below 0 is a bad request."""
    _model, npu, engine = make(drafts=3)
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        assert client.get("/props").json()["default_generation_settings"]["params"]["speculative.n_max"] == 3
        assert chat(client, "hi").status_code == 200
        assert chat(client, "hi", **{"speculative.n_max": 1, "speculative.n_min": 0}).status_code == 200
        npu.delay, npu.replies = 0.1, [[" a"] * 8]
        out: list[httpx.Response] = []
        t = threading.Thread(target=lambda: out.append(chat(client, "hi", **{"speculative.n_max": 9})))
        t.start()
        while engine.current is None:
            time.sleep(0.002)
        assert client.get("/slots").json()[0]["params"]["speculative.n_max"] == 3
        t.join(10)
        assert out[0].status_code == 200 and npu.generated == [None, 1, 9]
        r = chat(client, "hi", **{"speculative.n_max": -1})
        assert r.status_code == 400 and "speculative.n_max" in r.text


def test_utilities(fake: tuple) -> None:
    model, _npu, _engine, client = fake
    t = client.post("/tokenize", json={"content": "hello world"}).json()["tokens"]
    assert t == model.tokenizer.encode("hello world")
    t2 = client.post("/tokenize", json={"content": "hello world", "add_special": True, "with_pieces": True}).json()
    assert [p["piece"] for p in t2["tokens"]] == ["<s>", "hello", " world"]
    assert client.post("/detokenize", json={"tokens": t}).json() == {"content": "hello world"}
    prompt = client.post("/apply-template", json={"messages": [{"role": "user", "content": "hi"}]}).json()["prompt"]
    assert prompt == "<s><|user|>hi<|assistant|>"


def test_openai_client() -> None:
    """The openai package against a live server of the fake model."""
    _model, npu, engine = make()
    with live(engine) as url:
        client = openai.OpenAI(base_url=f"{url}/v1", api_key="none")
        assert [m.id for m in client.models.list()] == ["fake"]
        npu.replies = [["Hello", " there"], ["Bye"]]
        stream = client.chat.completions.create(model="fake", messages=[{"role": "user", "content": "Hi"}],
                                                stream=True)
        assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "Hello there"
        r = client.chat.completions.create(model="fake", messages=[{"role": "user", "content": "Hi"}], max_tokens=8)
        assert r.choices[0].message.content == "Bye" and r.usage.completion_tokens == 2


@pytest.mark.parametrize("stream", [True, False])
def test_disconnect(stream: bool) -> None:
    """A client that goes away (the UI's Stop aborts its request) cancels its generation."""
    _model, npu, engine = make(delay=0.01)
    npu.replies = [[f" w{i}" for i in range(1000)], ["next"]]
    with live(engine) as url:
        body = {"messages": [{"role": "user", "content": "go"}], "stream": stream}
        with httpx.Client(timeout=10) as client:
            if stream:
                with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
                    next(line for line in r.iter_lines() if '" w' in line)  # (a token, then away)
            else:
                with pytest.raises(httpx.ReadTimeout):
                    client.post(f"{url}/v1/chat/completions", json=body, timeout=0.3)
        t0 = time.perf_counter()
        while engine.current is not None and time.perf_counter() - t0 < 5:
            time.sleep(0.01)
        assert engine.current is None and npu.position < 500  # (cancelled long before the reply's end)
        assert httpx.post(f"{url}/v1/chat/completions", json={**body, "stream": False}).json()["choices"][0][
            "message"]["content"] == "next"


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Serve the fake model with the web UI (a manual check)")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--thinking", action="store_true", help="a thinking model (Qwen's markers)")
    a.add_argument("--path", help="the UI's files (default: the pinned build)")
    args = a.parse_args()
    fm, fnpu, eng = make(reasoning=QWEN if args.thinking else None, capacity=8192, delay=0.03)
    reply = ["Here", "is", "**markdown**:\n\n-", "one\n-", "two\n\n```python\nprint('hi')\n```\n\nDone."]
    pieces = [w if i == 0 else " " + w for i, w in enumerate(reply)]
    fnpu.replies = [(["Thinking", " about", " it", "\n", "</think>", "\n\n"] if args.thinking else []) + pieces] * 1000
    d = args.path or ui.fetch()
    uvicorn.run(app_of(eng, ui=lambda app: ui.mount(app, d), reasoning="on", sse_ping_interval=1.0), port=args.port)
    eng.stop()


def test_speech() -> None:
    """llama-server's /tts and OpenAI's /v1/audio/speech on a model that speaks: a WAV of the chunks (16-bit, 24 kHz),
    float32 PCM, a streamed WAV (its sizes 0xFFFFFFFF), SSE deltas of base64 16-bit chunks; the fields passed on (seed,
    voice, language, frames); the refusals (speaker_ref, unknown voices and formats, speed); /props' voices and
    /v1/models' output modalities; a model that does not speak: 501."""
    model = FakeSpeaker()
    npu = FakeNpu(model)
    engine = Engine(lambda: (model, npu), warmup=False)
    ENGINES.append(engine)
    engine.start()
    want = (np.concatenate([np.full(480, i / 10, np.float32) for i in range(3)]) * 32767).round().astype("<i2")
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        r = client.post("/tts", json={"input": "hello", "seed": 7, "voice": "male", "lang": "de", "n_predict": 20})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
        assert int.from_bytes(r.content[40:44], "little") == 2 * 1440 and r.content[24:28] == (24000).to_bytes(4, "little")
        assert np.array_equal(np.frombuffer(r.content[44:], "<i2"), want)
        assert model.spoken[-1] | {"stats": None} == {"text": "hello", "voice": "male", "language": "de", "seed": 7,
                                                      "max_frames": 20, "stats": None}
        r = client.post("/tts", json={"prompt": "hi", "response_format": "pcm"})
        assert r.headers["content-type"].startswith("audio/pcm;rate=24000") and len(r.content) == 4 * 1440
        assert model.spoken[-1]["seed"] is None and model.spoken[-1]["voice"] == "female"
        r = client.post("/tts", json={"input": "hi", "stream": True})
        assert r.content[4:8] == b"\xff\xff\xff\xff" and np.array_equal(np.frombuffer(r.content[44:], "<i2"), want)
        r = client.post("/tts", data={"input": "a form"})
        assert r.status_code == 200 and model.spoken[-1]["text"] == "a form"
        for body, word in (({"input": "x", "speaker_ref_b64": "AA=="}, "voice encoder"), ({"input": ""}, "input"),
                           ({"input": "x", "voice": "alloy"}, "female, male"), ({"input": "x", "response_format": "mp3"}, "wav")):
            r = client.post("/tts", json=body)
            assert r.status_code == 400 and word in r.json()["error"]["message"], (body, r.text)
        r = client.post("/v1/audio/speech", json={"model": "fake", "input": "hello", "voice": "female"})
        assert r.headers["content-type"] == "audio/wav" and np.array_equal(np.frombuffer(r.content[44:], "<i2"), want)
        r = client.post("/v1/audio/speech", json={"input": "hello", "response_format": "pcm"})
        assert np.array_equal(np.frombuffer(r.content, "<i2"), want)
        ev = events(client.post("/v1/audio/speech", json={"input": "hello", "stream_format": "sse"}))
        pcm = np.concatenate([np.frombuffer(base64.b64decode(e["audio"]), "<i2") for e in ev
                              if isinstance(e, dict) and e["type"] == "speech.audio.delta"])
        assert np.array_equal(pcm, want) and ev[-1] == {"type": "speech.audio.done", "usage": {
            "input_tokens": 5, "output_tokens": 3, "total_tokens": 8}}
        for body, word in (({"input": "x", "response_format": "mp3"}, "no encoder"), ({"input": "x", "voice": "alloy"},
                           "female, male"), ({"input": "x", "speed": 1.5}, "speed")):
            r = client.post("/v1/audio/speech", json=body)
            assert r.status_code == 400 and word in r.json()["error"]["message"]
        assert client.get("/props").json()["voices"] == ["female", "male"]
        assert client.get("/v1/models").json()["data"][0]["architecture"]["output_modalities"] == ["text", "audio"]


def test_speech_refused(fake: tuple) -> None:
    """A model that does not speak: /tts and /v1/audio/speech 501, no voices in /props."""
    client = fake[3]
    assert client.post("/tts", json={"input": "x"}).status_code == 501
    assert client.post("/v1/audio/speech", json={"input": "x"}).status_code == 501
    assert "voices" not in client.get("/props").json()


def test_chat_audio() -> None:
    """Chat completions with audio output (modalities text and audio: interleaved; audio alone: speech first): the
    spoken reply's transcript and its WAV (or
    pcm16) in message.audio, streamed as delta.audio pieces; a later turn's assistant message with that audio id goes to
    the model as it was said (its text and its frames' feedback rows); an unknown voice or format: 400."""
    model = FakeSpeaker()
    npu = FakeNpu(model)
    engine = Engine(lambda: (model, npu), warmup=False)
    ENGINES.append(engine)
    engine.start()
    want = (np.concatenate([np.full(480, i / 10, np.float32) for i in range(2)]) * 32767).round().astype("<i2")
    msgs = [{"role": "user", "content": "hi"}]
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        r = client.post("/v1/chat/completions", json={"messages": msgs, "modalities": ["text", "audio"],
                                                      "audio": {"voice": "female", "format": "wav"}})
        m = r.json()["choices"][0]["message"]
        assert m["content"] is None and m["audio"]["transcript"] == " hello there"
        data = base64.b64decode(m["audio"]["data"])
        assert data[:4] == b"RIFF" and np.array_equal(np.frombuffer(data[44:], "<i2"), want)
        assert r.json()["usage"]["completion_tokens_details"] == {"text_tokens": 2, "audio_tokens": 2}
        assert r.json()["usage"]["prompt_tokens"] == 7
        ev = events(client.post("/v1/chat/completions", json={"messages": msgs, "modalities": ["text", "audio"],
                                                               "audio": {"voice": "female", "format": "pcm16"},
                                                               "stream": True}))
        deltas = [e["choices"][0]["delta"].get("audio", {}) for e in ev if isinstance(e, dict)]
        assert "".join(d.get("transcript", "") for d in deltas) == " hello there"
        pcm = np.concatenate([np.frombuffer(base64.b64decode(d["data"]), "<i2") for d in deltas if "data" in d])
        assert np.array_equal(pcm, want) and ev[-1] == "[DONE]"
        turn = msgs + [{"role": "assistant", "audio": {"id": m["audio"]["id"]}}, {"role": "user", "content": "more"}]
        client.post("/v1/chat/completions", json={"messages": turn, "modalities": ["text", "audio"]})
        seen = model.spoken[-1]["messages"]
        assert [x["content"] for x in (seen[0], seen[2])] == ["hi", "more"] and model.spoken[-1]["mode"] == "mixed"
        said = seen[1]["content"]  # (the earlier reply as said: its text and its frame's feedback row)
        assert [p["type"] for p in said] == ["text", "audio_out", "text"] and said[0]["text"] == " hello"
        assert np.array_equal(said[1]["row"], np.full(8, 0.5, np.float32))
        client.post("/v1/chat/completions", json={"messages": msgs, "modalities": ["audio"]})
        assert model.spoken[-1]["mode"] == "audio"  # (speech, then text: <|audio_start|>)
        for audio, word in (({"voice": "alloy"}, "female, male"), ({"format": "mp3"}, "wav or pcm16")):
            r = client.post("/v1/chat/completions", json={"messages": msgs, "modalities": ["text", "audio"], "audio": audio})
            assert r.status_code == 400 and word in r.json()["error"]["message"]


def _speaker() -> tuple[FakeSpeaker, FakeNpu, Engine]:
    model = FakeSpeaker()
    npu = model.npu = FakeNpu(model, capacity=1024)
    engine = Engine(lambda: (model, npu), warmup=False)
    ENGINES.append(engine)
    engine.start()
    return model, npu, engine


def _response(ws: object) -> list[dict]:
    evs = []
    while not evs or evs[-1]["type"] != "response.done":
        evs.append(ws.receive_json())
    return evs


def test_realtime_conversation() -> None:
    """/v1/realtime without the transcription intent on a model that speaks: a conversation (llama-server's events) on
    the prompt cache - a text turn and an audio turn, each response's text and audio deltas between its start and done
    events, the output_modalities as the reply's mode, the instructions as the system prompt; a response runs only its
    turn's new tokens (its audio's rows ran as they came; the earlier reply stays as it was said: its tokens and its
    frames' feedback rows), and a request between turns does not lose the conversation."""
    model, npu, engine = _speaker()
    want = (np.concatenate([np.full(480, i / 10, np.float32) for i in range(2)]) * 32767).round().astype("<i2")
    pcm = lambda n: base64.b64encode((np.full(n, 0.25) * 32768).astype("<i2").tobytes()).decode()
    enc = model.tokenizer.encode
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        with client.websocket_connect("/v1/realtime") as ws:
            ev = ws.receive_json()
            assert ev["type"] == "session.created" and ev["session"]["audio"]["output"]["format"]["rate"] == 24000
            ws.send_json({"type": "session.update", "session": {"instructions": "Be brief.",
                                                                 "output_modalities": ["text", "audio"]}})
            assert ws.receive_json()["type"] == "session.updated"
            ws.send_json({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                                                                        "content": [{"type": "input_text", "text": "hi"}]}})
            assert [ws.receive_json()["type"] for _ in range(2)] == ["conversation.item.added", "conversation.item.done"]
            ws.send_json({"type": "response.create"})
            evs = _response(ws)
            assert evs[0]["type"] == "response.created" and evs[-1]["response"]["status"] == "completed"
            assert "".join(e["delta"] for e in evs if e["type"] == "response.output_text.delta") == " hello there"
            got = np.concatenate([np.frombuffer(base64.b64decode(e["delta"]), "<i2") for e in evs
                                  if e["type"] == "response.output_audio.delta"])
            assert np.array_equal(got, want) and model.spoken[-1]["mode"] == "mixed"
            first = enc("<s><|system|>Be brief.<|user|>") + enc("hi") + enc("<|assistant|>")
            assert list(model.spoken[-1]["tokens"].ids) == first
            ws.send_json({"type": "input_audio_buffer.append", "audio": pcm(3200)})  # (8 rows: they run at once)
            ws.send_json({"type": "input_audio_buffer.append", "audio": pcm(1000)})
            ws.send_json({"type": "response.create", "response": {"output_modalities": ["audio"]}})
            evs = _response(ws)
            assert evs[0]["type"] == "input_audio_buffer.committed" and model.spoken[-1]["mode"] == "audio"
            assert list(model.spoken[-1]["tokens"].ids) == enc("<|assistant|>")  # (the commit ran the clip's last rows)
            seq = engine.cached
            assert len(seq) == npu.position and list(seq.ids[len(first):len(first) + 4]) == \
                [enc(" hello")[0], enc("<|audio_out|>")[0], enc("<|audio_out|>")[0], enc(" there")[0]]
            assert np.array_equal(seq.rows[:2, 0], [len(first) + 1, len(first) + 2])  # (the frames' feedback rows)
            clip = len(first) + 4 + len(enc("<|user|>"))
            assert list(seq.ids[clip - 1:clip + 12]) == enc("<|user|>") + [MEDIA] * 11 + enc("<|assistant|>")
            assert np.array_equal(seq.rows[2:13, 0], np.arange(11) + 0.5)  # (the clip's 11 rows: 8 ran as they came)
            n = len(seq)
            assert chat(client, "something else").status_code == 200  # (another request displaces the conversation)
            ws.send_json({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                                                                        "content": [{"type": "input_text", "text": "and?"}]}})
            [ws.receive_json() for _ in range(2)]
            ws.send_json({"type": "response.create", "response": {"output_modalities": ["text"]}})
            _response(ws)
            assert model.spoken[-1]["mode"] == "text" and engine.cached.cache_keys()[:n] == seq.cache_keys()


def test_realtime_tools_images() -> None:
    """A realtime conversation's tools (the Responses API's functions in the session): the model's call as a
    function_call item with its arguments' deltas, its output back as a function_call_output item and the next
    response answering it; an input_image part (a camera frame) in a user turn: the image's tokens."""
    model, _, engine = _speaker()
    model.vision = FakeEncoder()
    model.chat_template = "<tool_call> <function= <parameter="  # (Qwen's XML calls: tools.syntax_of)
    model.say = ["<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"]
    tools = [{"type": "function", "name": "weather", "description": "the weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}]
    image = "data:image/png;base64," + base64.b64encode(b"cam" * 8).decode()
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            ws.send_json({"type": "session.update", "session": {"tools": tools, "output_modalities": ["text"]}})
            assert ws.receive_json()["session"]["tools"] == tools
            ws.send_json({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": image}, {"type": "input_text", "text": "weather here?"}]}})
            assert ws.receive_json()["item"]["content"][0] == {"type": "input_image"}
            ws.receive_json()
            ws.send_json({"type": "response.create"})
            evs = _response(ws)
            tokens = model.spoken[-1]["tokens"]
            assert list(tokens.ids[:5]) == model.tokenizer.encode("<s><|system|>tools: weather<|user|>") and \
                int(tokens.given.sum()) == 2  # (the camera frame's rows)
            call = next(e["item"] for e in evs if e["type"] == "response.output_item.done" and
                        e["item"]["type"] == "function_call")
            assert call["name"] == "weather" and json.loads(call["arguments"]) == {"city": "Paris"}
            assert any(e["type"] == "response.function_call_arguments.done" for e in evs)
            ws.send_json({"type": "session.update", "session": {"instructions": "x"}})
            assert ws.receive_json()["type"] == "error"  # (fixed once the conversation began)
            ws.send_json({"type": "conversation.item.create", "item": {"type": "function_call_output",
                                                                        "call_id": call["call_id"], "output": "sunny"}})
            assert ws.receive_json()["item"]["type"] == "function_call_output"
            ws.receive_json()
            model.say = [" It is sunny."]
            ws.send_json({"type": "response.create"})
            evs = _response(ws)
            assert list(model.spoken[-1]["tokens"].ids) == model.tokenizer.encode("<|tool|>sunny") + \
                model.tokenizer.encode("<|assistant|>")
            assert "".join(e["delta"] for e in evs if e["type"] == "response.output_text.delta") == " It is sunny."


def test_realtime_transcription_offline() -> None:
    """A transcription session of a model with a speech encoder and no live ASR (LFM3-Omni): each committed buffer
    transcribed alone with the offline ASR prompt (a chat job), its text as deltas, then the item's transcript."""
    model, npu, engine = make(audio=True)
    model.gguf = types.SimpleNamespace(meta={})  # (no asr template)
    pcm = base64.b64encode((np.full(1600, 0.25) * 32768).astype("<i2").tobytes()).decode()
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
            assert ws.receive_json()["session"]["type"] == "transcription"
            npu.replies = [["hello", " world"]]
            ws.send_json({"type": "input_audio_buffer.append", "audio": pcm})
            ws.send_json({"type": "input_audio_buffer.commit"})
            evs = [ws.receive_json() for _ in range(4)]
            assert [e["type"] for e in evs] == ["input_audio_buffer.committed"] + \
                ["conversation.item.input_audio_transcription.delta"] * 2 + \
                ["conversation.item.input_audio_transcription.completed"]
            assert evs[-1]["transcript"] == "hello world"
            assert "ASR" in str(model.rendered[0][0]["content"])
