"""artlite-server's Anthropic Messages and OpenAI Responses routes on the host (no NPU), with test_server's scripted
fake model: the responses and their streams, the conversions on requests shaped as Claude Code's and Codex's, the
stored responses, the errors, and the SDKs' own stream parsers (anthropic, openai)."""

import json
import re
from collections.abc import Iterator

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from models.server.agents import Blocks, Store, billing
from models.server.tools import ToolCall

from .test_server import (  # noqa: F401 (_stop_engines: the engines' cleanup, autouse)
    QWEN,
    TOOLS_TEMPLATE,
    WEATHER,
    FakeModel,
    FakeNpu,
    _stop_engines,
    app_of,
    call_pieces,
    chat,
    live,
    make,
)

# a template that takes a system message only first, as Qwen's (the fake model renders the prompt its own way)
FIRST = "{% for m in messages %}{% if m.role == 'system' and not loop.first %}" \
    "{{ raise_exception('System message must be at the beginning.') }}{% endif %}{% endfor %}" + TOOLS_TEMPLATE
TOOL = {"name": "get_weather", "description": "The weather in a city", "input_schema": WEATHER["function"]["parameters"]}
FUNCTION = {"type": "function", "name": "get_weather", "parameters": WEATHER["function"]["parameters"], "strict": False}
THOUGHT = ["Checking", " the", " sky", "\n", "</think>", "\n\n"]
REPLY = THOUGHT + ["Let", " me", " look.", "\n\n"] + call_pieces("get_weather", city="Paris", days="3")


@pytest.fixture
def agent() -> Iterator[tuple[FakeModel, FakeNpu, TestClient]]:
    """A thinking Qwen-like fake with tool calls, whose template takes a system message only first."""
    model, npu, engine = make(reasoning=QWEN)
    with TestClient(app_of(engine, template=FIRST)) as client:
        assert engine.wait_ready(5)
        yield model, npu, client


def named(r: httpx.Response) -> list[tuple[str, dict]]:
    """A stream's named events: (event, data)."""
    out = []
    for block in r.text.split("\n\n"):
        if block in ("", ":"):
            continue
        name, data = block.split("\n")
        assert name.startswith("event: ") and data.startswith("data: "), block
        out.append((name[7:], json.loads(data[6:])))
        assert out[-1][1]["type"] == out[-1][0]
    return out


def messages(client: TestClient, content: str | list[dict], **body: object) -> httpx.Response:
    msgs = [{"role": "user", "content": content}] if isinstance(content, str) else content
    return client.post("/v1/messages", json={"model": "m", "max_tokens": 200, "messages": msgs, **body})


def responses(client: TestClient, inputs: str | list[dict], **body: object) -> httpx.Response:
    return client.post("/v1/responses", json={"model": "m", "input": inputs, **body})


def test_blocks() -> None:
    """Parts as the ordered blocks of a message: another kind, or another call, opens the next block."""
    b = Blocks()
    changes = b.feed([("reasoning_content", "a"), ("reasoning_content", "b"), ("content", "c"),
                      ("tool_call", ToolCall(0, "call_1", "f", "{")), ("tool_call", ToolCall(0, arguments="}")),
                      ("tool_call", ToolCall(1, "call_2", "g", "{}")), ("content", "d")]) + b.end()
    assert [(w, x.kind, x.index, t) for w, x, t in changes] == [
        ("start", "thinking", 0, ""), ("delta", "thinking", 0, "a"), ("delta", "thinking", 0, "b"),
        ("stop", "thinking", 0, ""), ("start", "text", 1, ""), ("delta", "text", 1, "c"), ("stop", "text", 1, ""),
        ("start", "call", 2, ""), ("delta", "call", 2, "{"), ("delta", "call", 2, "}"), ("stop", "call", 2, ""),
        ("start", "call", 3, ""), ("delta", "call", 3, "{}"), ("stop", "call", 3, ""),
        ("start", "text", 4, ""), ("delta", "text", 4, "d"), ("stop", "text", 4, "")]
    assert [x.kind for x in b.complete(1)] == ["thinking", "text", "call", "text"]
    assert b.text("text") == "cd"


# ----------------------------------------------------------------------
# Anthropic's Messages
# ----------------------------------------------------------------------


def test_messages(agent: tuple) -> None:
    """A thought, text and a call as content blocks; stop_reason, usage with the cached tokens."""
    _model, npu, client = agent
    npu.replies = [REPLY, ["Sunny", "."]]
    body = {"tools": [TOOL], "thinking": {"type": "enabled", "budget_tokens": 2048}}
    r = messages(client, "Weather?", **body).json()
    assert r["type"] == "message" and r["role"] == "assistant" and r["id"].startswith("msg_") and r["model"] == "fake"
    thought, text, call = r["content"]
    assert thought == {"type": "thinking", "thinking": "Checking the sky", "signature": ""}
    assert text == {"type": "text", "text": "Let me look."}
    assert call["type"] == "tool_use" and call["name"] == "get_weather" and call["input"] == {"city": "Paris", "days": 3}
    assert r["stop_reason"] == "tool_use" and r["stop_sequence"] is None
    u = r["usage"]
    assert u["output_tokens"] == len(REPLY) + 1 and u["cache_read_input_tokens"] == 0 and u["input_tokens"] > 0
    turn = [{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": r["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "Sunny"}]}]
    r2 = messages(client, turn, tools=[TOOL], thinking={"type": "disabled"}).json()
    assert r2["content"] == [{"type": "text", "text": "Sunny."}] and r2["stop_reason"] == "end_turn"
    assert r2["usage"]["cache_read_input_tokens"] > 0  # (the second turn continued the cache)


def test_messages_stream(agent: tuple) -> None:
    """Anthropic's events: message_start with the usage, each block closed before the next (a thought's signature
    first), the call's arguments as partial JSON, message_delta's stop_reason."""
    _model, npu, client = agent
    npu.replies = [REPLY]
    evs = named(messages(client, "Weather?", stream=True, tools=[TOOL], thinking={"type": "adaptive"}))
    names = [n for n, _ in evs]
    assert names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
    start = evs[0][1]["message"]
    assert start["content"] == [] and start["stop_reason"] is None and start["usage"]["output_tokens"] == 0
    blocks: dict[int, dict] = {}
    open_index = None
    for name, d in evs[1:-2]:
        if name == "ping":
            continue
        if name == "content_block_start":
            assert open_index is None and d["index"] == len(blocks)
            open_index, blocks[d["index"]] = d["index"], {**d["content_block"], "deltas": []}
        elif name == "content_block_delta":
            assert d["index"] == open_index
            blocks[open_index]["deltas"].append(d["delta"])
        else:
            assert name == "content_block_stop" and d["index"] == open_index
            open_index = None
    assert [b["type"] for b in blocks.values()] == ["thinking", "text", "tool_use"]
    thought, text, call = blocks.values()
    assert "".join(x.get("thinking", "") for x in thought["deltas"]) == "Checking the sky"
    assert thought["deltas"][-1] == {"type": "signature_delta", "signature": ""}
    assert "".join(x["text"] for x in text["deltas"]) == "Let me look."
    assert call["name"] == "get_weather" and call["input"] == {}
    assert json.loads("".join(x["partial_json"] for x in call["deltas"])) == {"city": "Paris", "days": 3}
    end = evs[-2][1]
    assert end["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert end["usage"]["output_tokens"] == len(REPLY) + 1


def test_messages_conversion(agent: tuple) -> None:
    """A Claude Code conversation: the system blocks joined, the billing header's stamps fixed; an assistant's thought,
    text and calls one message; tool results where they stand, before the text after them; a system message between
    the turns as a user message (the template takes one only first)."""
    model, npu, client = agent
    system = [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.286.9d6; cc_entrypoint=sdk-cli;"},
              {"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}}]
    turns = [
        {"role": "user", "content": [{"type": "text", "text": "Fix it."}]},
        {"role": "system", "content": "# Environment"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Look first.", "signature": ""},
            {"type": "text", "text": "Reading."},
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.py"}},
            {"type": "tool_use", "id": "t2", "name": "Read", "input": {"path": "b.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "A"},
            {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
             "content": [{"type": "text", "text": "B"}, {"type": "text", "text": "2"}]},
            {"type": "text", "text": "<system-reminder>x</system-reminder>"}]},
        {"role": "system", "content": [{"type": "text", "text": "<total_tokens>9</total_tokens>"}]}]
    npu.replies = [["Done", "."]]
    r = messages(client, turns, system=system, tools=[TOOL, {"type": "web_search_20250305", "name": "web_search"}],
                 thinking={"type": "disabled"})
    assert r.status_code == 200, r.text
    rendered, tools = model.rendered
    assert [t["function"]["name"] for t in tools] == ["get_weather"]
    assert [(m["role"], m["content"]) for m in rendered] == [
        ("system", "x-anthropic-billing-header: cc_version=2.1.286; cc_entrypoint=sdk-cli;You are an agent."),
        ("user", "Fix it."), ("user", "# Environment"), ("assistant", "Reading."), ("tool", "A"), ("tool", "B2"),
        ("user", "<system-reminder>x</system-reminder>"), ("user", "<total_tokens>9</total_tokens>")]
    assistant = rendered[3]
    assert assistant["reasoning_content"] == "Look first."
    assert [(c["id"], c["function"]["arguments"]) for c in assistant["tool_calls"]] == \
        [("t1", {"path": "a.py"}), ("t2", {"path": "b.py"})]
    assert [m.get("tool_call_id") for m in rendered[4:6]] == ["t1", "t2"]
    assert "<think>\n\n</think>" in model.tokenizer.decode_bytes(npu.seq).decode()  # (thinking disabled)
    assert billing("x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You") == \
        "x-anthropic-billing-header: cc_version=2.1.101; cc_entrypoint=cli; cch=fffff;You"


def test_messages_tool_choice(agent: tuple) -> None:
    """any: a call is forced; tool: that function's call alone; none: the call stays text; stop_sequences."""
    _model, npu, client = agent
    rest = call_pieces("get_weather", city="Oslo")[1:]
    rest[0] = "get_weather>"
    npu.replies = [rest]
    r = messages(client, "Q", tools=[TOOL], tool_choice={"type": "any"}).json()
    assert [c["type"] for c in r["content"]] == ["tool_use"] and r["stop_reason"] == "tool_use"
    npu.replies = [call_pieces("get_weather", city="X")[2:] + ["\n"] + call_pieces("get_weather", city="Y")]
    r = messages(client, "Q", tools=[TOOL], tool_choice={"type": "tool", "name": "get_weather"}).json()
    assert [c["input"] for c in r["content"]] == [{"city": "X"}]
    npu.replies = [call_pieces("get_weather", city="X")]
    r = messages(client, "Q", tools=[TOOL], tool_choice={"type": "none"}).json()
    assert r["content"][0]["type"] == "text" and r["content"][0]["text"].startswith("<tool_call>")
    npu.replies = [["One", " STOP", " two"]]
    r = messages(client, "Q", stop_sequences=["STOP"]).json()
    assert r["content"] == [{"type": "text", "text": "One"}]
    assert (r["stop_reason"], r["stop_sequence"]) == ("stop_sequence", "STOP")
    npu.replies = [["a", " b", " c"]]
    r = messages(client, "Q", max_tokens=2).json()
    assert r["stop_reason"] == "max_tokens"


def test_count_tokens(agent: tuple) -> None:
    model, _npu, client = agent
    r = client.post("/v1/messages/count_tokens", json={"model": "m", "messages": [{"role": "user", "content": "a b"}],
                                                       "tools": [TOOL]})
    assert r.json() == {"input_tokens": len(model.chat([{"role": "user", "content": "a b"}],
                                                       tools=[{"type": "function", "function": {
                                                           "name": "get_weather"}}]))}


def test_messages_errors(agent: tuple) -> None:
    """Errors in Anthropic's shape; an overflow in the words Claude Code reads."""
    _model, _npu, client = agent
    r = messages(client, [{"role": "user", "content": [{"type": "document", "source": {}}]}])
    assert r.status_code == 400 and r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "invalid_request_error" and "document" in r.json()["error"]["message"]
    r = client.post("/v1/messages", json={"model": "m", "messages": "nope"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = messages(client, "word " * 600)
    assert r.status_code == 400
    m = re.search(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", r.json()["error"]["message"],
                  re.IGNORECASE)  # (Claude Code's pattern)
    assert m and int(m[1]) > int(m[2]) == 512
    assert client.post("/v1/chat/completions", json={"messages": "nope"}).json()["error"]["code"] == 400


def test_anthropic_sdk() -> None:
    """The anthropic package's stream parser on a live server: the final message it assembles."""
    anthropic = pytest.importorskip("anthropic")
    _model, npu, engine = make(reasoning=QWEN)
    with live(engine, template=FIRST) as url:
        client = anthropic.Anthropic(base_url=url, api_key="none")
        npu.replies = [REPLY, REPLY]
        args = {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [TOOL], "thinking": {"type": "enabled", "budget_tokens": 1024}}
        with client.messages.stream(**args) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        assert text == "Let me look." and final.stop_reason == "tool_use"
        assert [b.type for b in final.content] == ["thinking", "text", "tool_use"]
        assert final.content[0].thinking == "Checking the sky" and final.content[2].input == {"city": "Paris",
                                                                                               "days": 3}
        whole = client.messages.create(**args)
        assert [b.type for b in whole.content] == ["thinking", "text", "tool_use"]


# ----------------------------------------------------------------------
# OpenAI's Responses
# ----------------------------------------------------------------------


def test_responses(agent: tuple) -> None:
    """Output items: reasoning, message, function_call; usage; incomplete at max_output_tokens."""
    _model, npu, client = agent
    npu.replies = [REPLY, ["a", " b", " c"]]
    r = responses(client, "Weather?", tools=[FUNCTION], reasoning={"effort": "low"},
                  include=["reasoning.encrypted_content"]).json()
    assert r["object"] == "response" and r["status"] == "completed" and r["id"].startswith("resp_")
    thought, message, call = r["output"]
    assert thought["type"] == "reasoning" and thought["content"] == [{"type": "reasoning_text",
                                                                     "text": "Checking the sky"}]
    assert thought["encrypted_content"] == "" and thought["summary"] == []
    assert message["type"] == "message" and message["role"] == "assistant"
    assert message["content"][0]["type"] == "output_text" and message["content"][0]["text"] == "Let me look."
    assert call["type"] == "function_call" and call["name"] == "get_weather" and call["id"].startswith("fc_")
    assert call["call_id"].startswith("call_") and json.loads(call["arguments"]) == {"city": "Paris", "days": 3}
    u = r["usage"]
    assert u["output_tokens"] == len(REPLY) + 1 and u["total_tokens"] == u["input_tokens"] + u["output_tokens"]
    assert r["tools"] == [FUNCTION] and r["reasoning"] == {"effort": "low"}
    r = responses(client, "Count", max_output_tokens=2).json()
    assert r["status"] == "incomplete" and r["incomplete_details"] == {"reason": "max_output_tokens"}


def test_responses_stream(agent: tuple) -> None:
    """The Responses events: numbered from 0, each item added, its deltas, its .done events when it ends, then
    response.completed with the whole response."""
    _model, npu, client = agent
    npu.replies = [REPLY]
    evs = named(responses(client, "Weather?", tools=[FUNCTION], stream=True, reasoning={"effort": "high"}))
    assert [d["sequence_number"] for _, d in evs] == list(range(len(evs)))
    names = [n for n, _ in evs]
    assert names[:2] == ["response.created", "response.in_progress"] and names[-1] == "response.completed"
    assert evs[0][1]["response"]["status"] == "in_progress" and evs[0][1]["response"]["output"] == []
    added = [d for n, d in evs if n == "response.output_item.added"]
    done = [d for n, d in evs if n == "response.output_item.done"]
    assert [d["output_index"] for d in added] == [0, 1, 2] == [d["output_index"] for d in done]
    assert [d["item"]["type"] for d in done] == ["reasoning", "message", "function_call"]
    assert names.index("response.output_item.done") < names.index("response.output_text.delta")  # (in order)
    assert "".join(d["delta"] for n, d in evs if n == "response.reasoning_text.delta") == "Checking the sky"
    assert "".join(d["delta"] for n, d in evs if n == "response.output_text.delta") == "Let me look."
    args = "".join(d["delta"] for n, d in evs if n == "response.function_call_arguments.delta")
    (fdone,) = [d for n, d in evs if n == "response.function_call_arguments.done"]
    assert fdone["arguments"] == args == done[2]["item"]["arguments"] and fdone["item_id"] == done[2]["item"]["id"]
    for n, d in evs:
        if "item_id" in d:
            assert d["item_id"] == added[d["output_index"]]["item"]["id"], n
    final = evs[-1][1]["response"]
    assert final["output"] == [d["item"] for d in done] and final["usage"]["output_tokens"] == len(REPLY) + 1


def test_responses_conversion(agent: tuple) -> None:
    """A Codex request: instructions and a developer message one system message; reasoning, message and calls of a
    turn one assistant message; the outputs as tool messages; tools other than functions left out; a named
    tool_choice."""
    model, npu, client = agent
    items = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions>"},
                                                             {"type": "input_text", "text": "<skills>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix it."}]},
        {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "Run tests."}],
         "encrypted_content": ""},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Running."}]},
        {"type": "function_call", "name": "exec_command", "arguments": '{"cmd": "pytest"}', "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "1 failed"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Read it."}]},
        {"type": "function_call", "name": "exec_command", "arguments": '{"cmd": "cat a"}', "call_id": "c2"},
        {"type": "function_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "x = 1"}]},
        {"role": "user", "content": "Go on."}]
    rest = call_pieces("get_weather", city="Oslo")[2:]
    npu.replies = [rest]
    tools = [FUNCTION, {"type": "namespace", "name": "multi_agent_v1", "tools": []}, {"type": "web_search"}]
    r = responses(client, items, instructions="You are Codex.\n", tools=tools, store=False,
                  tool_choice={"type": "function", "name": "get_weather"}, parallel_tool_calls=False)
    assert r.status_code == 200, r.text
    assert [o["type"] for o in r.json()["output"]] == ["function_call"]
    rendered, fns = model.rendered
    assert [f["function"]["name"] for f in fns] == ["get_weather"]
    assert [(m["role"], m["content"]) for m in rendered] == [
        ("system", "You are Codex.\n\n\n<permissions>\n<skills>"), ("user", "Fix it."), ("assistant", "Running."),
        ("tool", "1 failed"), ("assistant", ""), ("tool", "x = 1"), ("user", "Go on.")]
    assert rendered[2]["reasoning_content"] == "Run tests." and rendered[4]["reasoning_content"] == "Read it."
    assert [c["function"]["arguments"] for m in rendered[2:5:2] for c in m["tool_calls"]] == \
        [{"cmd": "pytest"}, {"cmd": "cat a"}]


def test_previous_response(agent: tuple) -> None:
    """previous_response_id continues a stored response's items (instructions only the request's own); store false
    keeps none; GET and DELETE of a stored response."""
    model, npu, client = agent
    npu.replies = [["Hi", "!"], ["Fine", "."], ["x"]]
    r1 = responses(client, "Hello", instructions="Old.").json()
    r2 = responses(client, [{"role": "user", "content": "How are you?"}], previous_response_id=r1["id"],
                   instructions="New.").json()
    assert [(m["role"], m["content"]) for m in model.rendered[0]] == [
        ("system", "New."), ("user", "Hello"), ("assistant", "Hi!"), ("user", "How are you?")]
    assert r2["previous_response_id"] == r1["id"]
    assert client.get(f"/v1/responses/{r2['id']}").json()["output"] == r2["output"]
    r3 = responses(client, "Bye", store=False).json()
    r = responses(client, "Again", previous_response_id=r3["id"])
    assert r.status_code == 400 and r.json()["error"]["code"] == "previous_response_not_found"
    assert client.delete(f"/v1/responses/{r1['id']}").json() == {"id": r1["id"], "object": "response.deleted",
                                                                 "deleted": True}
    assert client.get(f"/v1/responses/{r1['id']}").status_code == 404
    assert responses(client, "x", previous_response_id=r2["id"]).status_code == 400  # (the chain lost its start)
    store = Store(size=2)
    for i in range(3):
        store.put({"id": f"r{i}", "previous_response_id": f"r{i - 1}" if i else None, "output": [{"o": i}]},
                  [{"i": i}])
    with pytest.raises(KeyError):
        store.items("r2")  # (r0 dropped)


def test_responses_errors(agent: tuple) -> None:
    """An overflow: context_length_exceeded (streamed: response.failed); input tokens; items not taken."""
    model, _npu, client = agent
    r = responses(client, "word " * 600)
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    evs = named(responses(client, "word " * 600, stream=True))
    assert [n for n, _ in evs] == ["response.created", "response.in_progress", "response.failed"]
    assert evs[-1][1]["response"]["error"]["code"] == "context_length_exceeded"
    r = responses(client, [{"type": "item_reference", "id": "x"}])
    assert r.status_code == 400 and "item_reference" in r.json()["error"]["message"]
    r = client.post("/v1/responses/input_tokens", json={"model": "m", "input": "a b c"}).json()
    assert r == {"object": "response.input_tokens",
                 "input_tokens": len(model.chat([{"role": "user", "content": "a b c"}]))}


def test_responses_openai_client() -> None:
    """The openai package's Responses stream helper (it checks the events' order and items) on a live server."""
    _model, npu, engine = make(reasoning=QWEN)
    with live(engine, template=FIRST) as url:
        client = openai.OpenAI(base_url=f"{url}/v1", api_key="none")
        npu.replies = [REPLY, REPLY]
        args = {"model": "m", "input": "Weather?", "tools": [FUNCTION], "reasoning": {"effort": "medium"}}
        with client.responses.stream(**args) as stream:
            kinds = [e.type for e in stream]
            final = stream.get_final_response()
        assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
        assert [o.type for o in final.output] == ["reasoning", "message", "function_call"]
        assert final.output_text == "Let me look." and final.output[2].name == "get_weather"
        whole = client.responses.create(**args)
        assert json.loads(whole.output[2].arguments) == {"city": "Paris", "days": 3}
        assert whole.usage.output_tokens == len(REPLY) + 1


# ----------------------------------------------------------------------
# System and developer messages on every route
# ----------------------------------------------------------------------


def test_system_messages(agent: tuple) -> None:
    """developer is system; a template that takes a system message only first gets the leading ones joined and the
    later ones as user messages; another template gets them as they are."""
    model, _npu, client = agent
    msgs = [{"role": "developer", "content": "A"}, {"role": "system", "content": [{"type": "text", "text": "B"}]},
            {"role": "user", "content": "Q"}, {"role": "system", "content": "C"}]
    assert chat(client, msgs).status_code == 200
    assert [(m["role"], m["content"]) for m in model.rendered[0]] == [("system", "A\n\nB"), ("user", "Q"),
                                                                       ("user", "C")]
    model, _npu, engine = make()
    with TestClient(app_of(engine)) as client:
        assert chat(client, msgs).status_code == 200
        assert [(m["role"], m["content"]) for m in model.rendered[0]] == [("system", "A"), ("system", "B"),
                                                                           ("user", "Q"), ("system", "C")]
