"""artlite-server's structured output on the fake model (no NPU): every spelling of a request parsed into one
constraint (``constraints.constraint_of``) and its errors, then generations held to it - a model that writes something
else is kept to the grammar, a thought stays free, logprobs before and after the mask, completions' grammar, the
server's default (``Options.constraint``); the agents' APIs' spellings, ``allowed_tools``, the tools' options and the
SDKs' helpers."""

import json

import llguidance
import pydantic
import pytest

from models.server import agents
from models.server.constraints import constraint_of
from models.server.errors import ApiError
from models.structured import Constraint, ConstraintError, add_arguments, compiled, from_args

from .fakes import GEMMA, QWEN, TIME, TOOLS_TEMPLATE, WEATHER, call_pieces, chat, client_of, deltas, events, live, make

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "required": ["a"]}
# (a schema the fake completes from any reply: greedy where none of its tokens is allowed takes the lowest allowed one,
# which in a string is a blank, forever)
FLAGS = {"type": "object", "properties": {"a": {"type": "boolean"}, "n": {"type": "integer", "minimum": 0, "maximum": 9}},
         "required": ["a", "n"]}


def fits(engine: object, c: Constraint, text: str) -> bool:
    """Whether a text is a prefix of what the constraint takes (its characters through a fresh matcher)."""
    v = engine.vocab()
    m = llguidance.LLMatcher(v.tokenizer, compiled(c, v).grammar, log_level=0)
    return all(m.consume_token(t) for t in v.tokenizer.tokenize_str(text))


def test_spellings() -> None:
    """llama-server's, OpenAI's and vLLM's fields: the constraint and the field it came from; null is unset."""
    c, f = constraint_of({"grammar": 'root ::= "x"'})
    assert (c.kind, f) == ("gbnf", "grammar")
    assert constraint_of({"grammar": '%llguidance {}\nstart: "x"'})[0].kind == "lark"
    c, f = constraint_of({"json_schema": SCHEMA})
    assert (c.kind, json.loads(c.source), c.strict, f) == ("json_schema", SCHEMA, False, "json_schema")
    assert constraint_of({"json_schema": {}})[0].kind == "json_object"
    assert constraint_of({"response_format": {"type": "json_object"}})[0] == Constraint("json_object", "",
                                                                                        name="response_format")
    c = constraint_of({"response_format": {"type": "json_object", "schema": SCHEMA}})[0]
    assert c.kind == "json_schema" and not c.strict
    c = constraint_of({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}}})[0]
    assert c.kind == "json_schema" and c.strict and c.name == "x"
    c = constraint_of({"response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA, "strict": False}}})[0]
    assert not c.strict  # (strict false: lenient)
    assert constraint_of({"response_format": {"type": "json_schema", "schema": SCHEMA}})[0].strict  # (the flat form)
    for empty in ({}, {"response_format": {"type": "text"}}, {"grammar": None, "json_schema": None,
                                                                "response_format": None, "structured_outputs": None},
                  {"grammar": ""}, {"response_format": {"type": ""}}):
        assert constraint_of(empty) is None
    # (llama-server's: json_object without a schema takes json_schema's; on /completion grammar wins over json_schema)
    c, f = constraint_of({"response_format": {"type": "json_object"}, "json_schema": SCHEMA})
    assert (c.kind, f) == ("json_schema", "json_schema")
    assert constraint_of({"grammar": 'root ::= "x"', "json_schema": SCHEMA}, route="completion")[1] == "grammar"
    c, f = constraint_of({"structured_outputs": {"choice": ["yes", "no"]}})
    assert (c.kind, json.loads(c.source), f) == ("choice", ["yes", "no"], "structured_outputs")
    c = constraint_of({"structured_outputs": {"json": json.dumps(SCHEMA), "disable_any_whitespace": True,
                                              "disable_additional_properties": True, "whitespace_pattern": None}})[0]
    assert c.kind == "json_schema" and c.compact and c.closed
    assert constraint_of({"structured_outputs": {"regex": "[a-z]+"}})[0].kind == "regex"
    assert constraint_of({"structured_outputs": {"grammar": 'root ::= "a"'}})[0].kind == "gbnf"
    assert constraint_of({"structured_outputs": {"grammar": 'start: "a"'}})[0].kind == "lark"
    assert constraint_of({"structured_outputs": {"json_object": True}})[0].kind == "json_object"
    c, f = constraint_of({"guided_json": SCHEMA, "guided_whitespace_pattern": " ?"})
    assert (c.kind, c.whitespace, f) == ("json_schema", " ?", "guided_json")
    assert constraint_of({"guided_choice": ["a"]})[0].kind == "choice"
    tag = {"type": "structural_tag", "structures": [{"begin": "<f=a>", "schema": SCHEMA, "end": "</f>"}],
           "triggers": ["<f="]}
    c = constraint_of({"response_format": tag})[0]
    assert c.kind == "structural_tag" and json.loads(c.source)[0]["trigger"] == "<f="
    assert constraint_of({"structured_outputs": {"structural_tag": json.dumps(tag)}})[0].kind == "structural_tag"


@pytest.mark.parametrize(("body", "message"), [
    ({"grammar": 'root ::= "x"', "json_schema": SCHEMA}, "Cannot use both grammar and json_schema"),
    ({"json_schema": SCHEMA, "response_format": {"type": "json_schema", "schema": SCHEMA}},
     "Cannot use both json_schema and response_format"),
    ({"guided_json": SCHEMA, "guided_regex": "a"}, "Cannot use both"),
    ({"response_format": "json"}, "response_format must be an object"),
    ({"response_format": {"type": "sound"}}, "response_format type must be one of text, json_object, json_schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}, "json_schema needs its schema"),
    ({"json_schema": 123}, "a JSON schema is an object"),
    ({"structured_outputs": {"regex": "a", "choice": ["a"]}}, "2 given: regex, choice"),
    ({"structured_outputs": {}}, "none given"),
    ({"structured_outputs": {"choice": []}}, "choice is a non-empty list"),
    ({"response_format": {"type": "structural_tag", "structures": [{"begin": "<g>", "end": ""}],
                          "triggers": ["<f"]}}, "begins with none of the triggers"),
    ({"response_format": {"type": "structural_tag", "format": {}}}, "'format' (xgrammar's) is not supported"),
])
def test_parse_errors(body: dict, message: str) -> None:
    with pytest.raises(ApiError, match=message.replace("(", r"\(").replace(")", r"\)")) as e:
        constraint_of(body)
    assert e.value.status == 400


@pytest.mark.parametrize(("body", "message"), [
    ({"grammar": 'root ::= "x"', "json_schema": SCHEMA}, "Cannot use both grammar and json_schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "u", "schema": {
        "type": "array", "uniqueItems": True}}}}, 'response_format: Unimplemented keys: ["uniqueItems"]'),
    ({"grammar": "root ::= this is (not valid GBNF"}, "grammar: the GBNF grammar"),
    ({"json_schema": {"type": "hiccup"}}, "json_schema:"),
    ({"response_format": {"type": "sound"}}, "response_format type must be one of"),
])
def test_errors(body: dict, message: str) -> None:
    """A field that does not parse or compile: a 400 in OpenAI's shape that names it, before any job runs; Anthropic's
    shape on /v1/messages."""
    with client_of(chars=True) as (_model, npu, _engine, client):
        r = chat(client, "Q", **body)
        assert r.status_code == 400 and message in r.json()["error"]["message"], r.json()
        assert r.json()["error"]["type"] == "invalid_request_error" and not npu.generated
        if "grammar" not in body or "json_schema" not in body:  # (/completion: grammar wins, as llama-server)
            r = client.post("/completion", json={"prompt": "x", **body})
            assert r.status_code == 400 and message in r.json()["error"]["message"]


def test_json_kept() -> None:
    """A reply that already fits passes as it is; a model that writes something else is held to the grammar (greedy:
    the lowest allowed token where none it wants is allowed), whole and streamed: the content loads as JSON and
    validates, or is cut where max_tokens ends it, a prefix of what the grammar takes; the end of sequence once
    complete."""
    with client_of(chars=True) as (_model, npu, engine, client):
        rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}}
        good = list('{"a": 42, "b": "hi"}')
        npu.replies = [good]
        r = chat(client, "Q", response_format=rf, temperature=0).json()
        assert r["choices"][0]["message"]["content"] == "".join(good)
        assert r["choices"][0]["finish_reason"] == "stop"
        bad = ["Sure", "!", " Here", " is", " JSON:", "```"]
        npu.replies = [bad]
        r = chat(client, "Q", json_schema=FLAGS, temperature=0, max_tokens=200).json()
        out = json.loads(r["choices"][0]["message"]["content"])
        assert isinstance(out["a"], bool) and 0 <= out["n"] <= 9 and r["choices"][0]["finish_reason"] == "stop"
        npu.replies = [bad]
        r = chat(client, "Q", response_format=rf, temperature=0, max_tokens=60).json()
        text = r["choices"][0]["message"]["content"]
        assert fits(engine, Constraint.json_schema(SCHEMA, strict=True, name="x"), text) and len(text) > 20
        npu.replies = [good]
        evs = events(chat(client, "Q", response_format=rf, temperature=0, stream=True))
        assert deltas(evs) == "".join(good) and evs[-2]["choices"][0]["finish_reason"] == "stop"
        npu.replies = [list('{"a": 1')]  # (max_tokens: cut, incomplete)
        r = chat(client, "Q", response_format=rf, temperature=0, max_tokens=4).json()
        assert r["choices"][0]["finish_reason"] == "length"


def test_choice_regex_grammar() -> None:
    """vLLM's choice and regex, llama-server's GBNF on /completion and /v1/completions (a raw prompt: the constraint
    starts at once)."""
    with client_of(chars=True) as (_model, npu, _engine, client):
        npu.replies = [["n", "o"]]
        r = chat(client, "Q", structured_outputs={"choice": ["yes", "no"]}, temperature=0).json()
        assert r["choices"][0]["message"]["content"] == "no"
        npu.replies = [["maybe"]]
        r = chat(client, "Q", structured_outputs={"choice": ["yes", "no"]}, temperature=0).json()
        assert r["choices"][0]["message"]["content"] in ("yes", "no")
        npu.replies = [["Z"]]
        r = chat(client, "Q", structured_outputs={"regex": "[0-9]{3}"}, temperature=0).json()
        assert len(r["choices"][0]["message"]["content"]) == 3
        npu.replies = [["ok"]]
        r = client.post("/completion", json={"prompt": "x", "grammar": 'root ::= "a"{5,5}', "temperature": 0}).json()
        assert r["content"] == "aaaaa" and r["stop_type"] == "eos"
        npu.replies = [["ok"]]
        r = client.post("/v1/completions", json={"prompt": "x", "json_schema": {"type": "boolean"},
                                                 "temperature": 0}).json()
        assert json.loads(r["choices"][0]["text"]) in (True, False)  # (after up to 8 blanks: the lowest tokens)


def test_thought_free() -> None:
    """The thought is never constrained: a Qwen thought (the prompt opens it) and a Gemma one (the model opens it, its
    opener allowed beside the grammar's tokens), then the answer held to the schema; thinking off: at once."""
    for reasoning, reply in ((QWEN, ["Let", " me", " think", "```", "</think>", "\n\n", "{", '"', "a", '"', ":",
                                     " ", "7", "}"]),
                             (GEMMA, ["<|channel>thought", "Hmm", " ```", "<channel|>", "{", '"', "a", '"', ":",
                                      "7", "}"])):
        with client_of(reasoning=reasoning, chars=True) as (_model, npu, _engine, client):
            npu.replies = [reply]
            r = chat(client, "Q", json_schema=SCHEMA, temperature=0,
                     chat_template_kwargs={"enable_thinking": True}).json()
            msg = r["choices"][0]["message"]
            assert json.loads(msg["content"]) == {"a": 7} and "```" in msg["reasoning_content"], msg
            npu.replies = [["Thinking", " out", " loud"]]
            r = chat(client, "Q", json_schema=FLAGS, temperature=0, max_tokens=300,
                     chat_template_kwargs={"enable_thinking": False}).json()
            assert isinstance(json.loads(r["choices"][0]["message"]["content"])["a"], bool)


def test_logprobs() -> None:
    """logprobs: the model's distribution before the mask (a token the grammar forbids among the top ones); with
    post_sampling_probs the sampler's candidates after it (only allowed tokens)."""
    with client_of(chars=True) as (_model, npu, _engine, client):
        npu.replies = [["Nope"]]
        r = chat(client, "Q", json_schema={"type": "boolean"}, temperature=0, logprobs=True, top_logprobs=3).json()
        first = r["choices"][0]["logprobs"]["content"][0]
        assert first["top_logprobs"][0]["token"] == "Nope" and first["token"] != "Nope"
        npu.replies = [["Nope"]]
        r = client.post("/completion", json={"prompt": "x", "json_schema": {"type": "boolean"}, "n_probs": 3,
                                             "post_sampling_probs": True, "temperature": 0}).json()
        first = r["completion_probabilities"][0]  # (a boolean, or blanks before it)
        assert {t["token"] for t in first["top_probs"]} <= set("tf \t\n\r") and first["token"] in "tf \t\n\r"


def test_server_default() -> None:
    """The server's constraint (--grammar, --json-schema) holds a request that asks for none; one that asks for its own
    replaces it."""
    default = {"constraint": Constraint("choice", json.dumps(["A", "B"]))}
    with client_of(default, chars=True) as (_model, npu, _engine, client):
        npu.replies = [["Z"]]
        assert chat(client, "Q", temperature=0).json()["choices"][0]["message"]["content"] in ("A", "B")
        npu.replies = [["7"]]
        r = chat(client, "Q", structured_outputs={"regex": "[0-9]"}, temperature=0).json()
        assert r["choices"][0]["message"]["content"] == "7"


# ----------------------------------------------------------------------
# The other APIs' spellings, the tools' options, the SDKs
# ----------------------------------------------------------------------


def test_agents_spellings() -> None:
    """Responses' text.format and Anthropic's output_config.format (output_format: its deprecated spelling) as the
    chat body's response_format; function tools keep strict; Responses' allowed_tools as the chat API's."""
    note = lambda name: None
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    out = agents.responses_chat({"input": "Q", "text": {"format": {"type": "json_schema", "name": "n",
                                                                   "schema": schema, "strict": True}}}, [], note)
    assert out["response_format"] == {"type": "json_schema", "json_schema": {"name": "n", "schema": schema,
                                                                             "strict": True}}
    assert constraint_of(out)[0].strict
    out = agents.responses_chat({"input": "Q", "text": {"format": {"type": "json_object"}}}, [], note)
    assert out["response_format"] == {"type": "json_object"}
    assert "response_format" not in agents.responses_chat({"input": "Q", "text": {"format": {"type": "text"}}}, [], note)
    with pytest.raises(ValueError, match="text.format"):
        agents.responses_chat({"input": "Q", "text": {"format": {"type": "grammar"}}}, [], note)
    tool = {"type": "function", "name": "f", "parameters": schema, "strict": True}
    out = agents.responses_chat({"input": "Q", "tools": [tool], "tool_choice": {
        "type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "f"}]}}, [], note)
    assert out["tools"][0]["function"]["strict"] is True
    assert out["tool_choice"] == {"type": "allowed_tools", "allowed_tools": {
        "mode": "required", "tools": [{"type": "function", "function": {"name": "f"}}]}}
    body = {"messages": [{"role": "user", "content": "Q"}], "max_tokens": 10}
    for key in ({"output_config": {"format": {"type": "json_schema", "schema": schema}}},
                {"output_format": {"type": "json_schema", "schema": schema}}):
        out = agents.anthropic_chat({**body, **key}, note)
        c = constraint_of(out)[0]
        assert c.kind == "json_schema" and c.strict and json.loads(c.source) == schema
    with pytest.raises(ValueError, match="deprecated"):
        agents.anthropic_chat({**body, "output_config": {"format": {"type": "json_schema", "schema": schema}},
                               "output_format": {"type": "json_schema", "schema": schema}}, note)
    with pytest.raises(ValueError, match="output_config.format"):
        agents.anthropic_chat({**body, "output_config": {"format": {"type": "json_object"}}}, note)
    out = agents.anthropic_chat({**body, "tools": [{"name": "f", "input_schema": schema, "strict": True}]}, note)
    assert out["tools"][0]["function"]["strict"] is True


def test_allowed_tools() -> None:
    """allowed_tools: the prompt keeps every tool, the calls are the allowed ones' (required: one is forced); a name
    that is not a tool: a 400."""
    with client_of({"template": TOOLS_TEMPLATE}) as (model, npu, _engine, client):
        reply = call_pieces("get_time")[1:]  # (after the forced "<tool_call>\n<function=")
        reply[0] = "get_time>"
        npu.replies = [reply]
        choice = {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
            {"type": "function", "function": {"name": "get_time"}}]}}
        r = chat(client, "Q", tools=[WEATHER, TIME], tool_choice=choice).json()
        assert [c["function"]["name"] for c in r["choices"][0]["message"]["tool_calls"]] == ["get_time"]
        assert [t["function"]["name"] for t in model.rendered[1]] == ["get_weather", "get_time"]
        choice["allowed_tools"]["tools"] = [{"type": "function", "function": {"name": "nope"}}]
        r = chat(client, "Q", tools=[WEATHER, TIME], tool_choice=choice)
        assert r.status_code == 400 and "nope" in r.json()["error"]["message"]


def test_tool_options(tmp_path: object) -> None:
    """--grammar, --grammar-file, -j, --json-schema-file: one of them, a file read, a schema that is not JSON."""
    import argparse
    p = argparse.ArgumentParser()
    add_arguments(p)
    assert from_args(p.parse_args([])) is None
    assert from_args(p.parse_args(["--grammar", 'root ::= "a"'])).kind == "gbnf"
    assert from_args(p.parse_args(["-j", "{}"])).kind == "json_object"
    f = tmp_path / "s.json"
    f.write_text(json.dumps(SCHEMA))
    c = from_args(p.parse_args(["--json-schema-file", str(f)]))
    assert c.kind == "json_schema" and json.loads(c.source) == SCHEMA
    for argv, message in ((["--grammar", "x", "-j", "{}"], "one at most"), (["-j", "{nope"], "not JSON"),
                          (["--grammar-file", str(tmp_path / "missing")], "grammar-file")):
        with pytest.raises(ConstraintError, match=message):
            from_args(p.parse_args(argv))


class Person(pydantic.BaseModel):
    name: str
    age: int
    tags: list[str]


def test_sdks() -> None:
    """The SDKs' structured-output helpers round-trip: openai's chat.completions.parse and responses.parse with a
    pydantic model, anthropic's messages.parse with one; a strict schema llguidance cannot take: Anthropic's error
    shape on /v1/messages."""
    import anthropic
    import openai
    _model, npu, engine = make(chars=True)
    reply = list('{"name": "Ann", "age": 7, "tags": ["x"]}')
    with live(engine) as url:
        oc = openai.OpenAI(base_url=f"{url}/v1", api_key="none")
        npu.replies = [reply]
        r = oc.chat.completions.parse(model="m", messages=[{"role": "user", "content": "Q"}], response_format=Person)
        assert r.choices[0].message.parsed == Person(name="Ann", age=7, tags=["x"])
        npu.replies = [reply]
        r = oc.responses.parse(model="m", input="Q", text_format=Person)
        assert r.output_parsed == Person(name="Ann", age=7, tags=["x"])
        ac = anthropic.Anthropic(base_url=url, api_key="none")
        npu.replies = [reply]
        m = ac.messages.parse(model="m", max_tokens=200, messages=[{"role": "user", "content": "Q"}],
                              output_format=Person)
        assert m.parsed_output == Person(name="Ann", age=7, tags=["x"])
        with pytest.raises(anthropic.BadRequestError) as e:
            ac.messages.create(model="m", max_tokens=20, messages=[{"role": "user", "content": "Q"}],
                               output_config={"format": {"type": "json_schema", "schema": {
                                   "type": "array", "uniqueItems": True}}})
        assert e.value.body["type"] == "error" and "uniqueItems" in e.value.body["error"]["message"]
