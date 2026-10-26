"""The tool-call syntaxes of ``models.server.tools`` on the host: llama.cpp's test vectors (``tests/test-chat.cpp``)
for each syntax, each output fed whole, in pieces of 3 characters and one character at a time (the content and the
arguments' fragments join to the same result); detection from the templates of the model files; requests' functions
and call arguments."""

import json
import os

import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import gemma_gguf, llama_gguf, minicpmv_gguf, qwen_gguf
from models.server.tools import GEMMA4, LLAMA3, QWEN_XML, Syntax, ToolCall, call_arguments, functions, syntax_of, typed


def tool(name: str, props: dict[str, object] | None = None) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "description": f"{name}'s description",
                                             "parameters": {"type": "object", "properties": props or {}}}}


TOOLS = functions([
    tool("special_function", {"arg1": {"type": "integer"}}),
    tool("special_function_with_opt", {"arg1": {"type": "integer"}, "arg2": {"type": "integer"}}),
    tool("python", {"code": {"type": "string"}}),
    tool("edit", {"filename": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}}),
    tool("empty_args"),
    tool("get_time", {"city": {"type": "string"}}),
    tool("get_weather", {"city": {"type": "string"}}),
    tool("mixed", {"n": {"type": ["integer", "string"]}, "flag": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                   "any": {}, "items": {"type": "array"}}),
])


def parse(syntax: Syntax, text: str, step: int, *, parallel: bool = True) -> tuple[str, list[tuple[str, object]]]:
    """The content and the calls (name, arguments) of a generated text fed ``step`` characters at a time."""
    p = syntax.parser(TOOLS, parallel=parallel)
    parts = []
    for i in range(0, len(text), step):
        parts += p.parts([("content", text[i:i + step])])
    parts += p.flush()
    content = "".join(v for k, v in parts if k == "content")
    calls: dict[int, dict[str, str]] = {}
    for kind, c in parts:
        if kind == "tool_call":
            assert isinstance(c, ToolCall)
            if c.index not in calls:
                assert c.id and c.id.startswith("call_") and c.name, "a call's first part names it"
                calls[c.index] = {"name": c.name, "arguments": ""}
            else:
                assert c.id is None and c.name is None
            calls[c.index]["arguments"] += c.arguments
    assert sorted(calls) == list(range(len(calls)))
    assert p.calls == len(calls)
    return content, [(c["name"], json.loads(c["arguments"])) for c in calls.values()]


def check(syntax: Syntax, text: str, content: str, calls: list[tuple[str, object]], **kw: object) -> None:
    for step in (len(text) or 1, 3, 1):
        assert parse(syntax, text, step, **kw) == (content, calls), f"fed {step} characters at a time"


def qwen_call(name: str, **args: str) -> str:
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


QWEN = [
    (qwen_call("special_function", arg1="1"), "", [("special_function", {"arg1": 1})]),
    (qwen_call("special_function", arg1="1") + "\n" + qwen_call("special_function_with_opt", arg1="1", arg2="2"), "",
     [("special_function", {"arg1": 1}), ("special_function_with_opt", {"arg1": 1, "arg2": 2})]),
    (qwen_call("python", code='def hello():\n    print("Hello, world!")\n\nhello()'), "",
     [("python", {"code": 'def hello():\n    print("Hello, world!")\n\nhello()'})]),
    (qwen_call("edit", filename="foo.c", oldString="#iclunde", newString="#include"), "",
     [("edit", {"filename": "foo.c", "oldString": "#iclunde", "newString": "#include"})]),
    # (a value that ends in a newline: the syntax's newline before </parameter> follows it)
    (qwen_call("edit", filename="foo.c", oldString="#iclunde", newString="#include\n"), "",
     [("edit", {"filename": "foo.c", "oldString": "#iclunde", "newString": "#include\n"})]),
    (qwen_call("python", code='    print("Hello, world!")'), "", [("python", {"code": '    print("Hello, world!")'})]),
    (qwen_call("python", code=""), "", [("python", {"code": ""})]),
    (qwen_call("python", code='x = "\\u00e9" < 1 and "</parameter"'), "",
     [("python", {"code": 'x = "\\u00e9" < 1 and "</parameter"'})]),
    (qwen_call("empty_args"), "", [("empty_args", {})]),
    ("Let me check.\n\n" + qwen_call("get_time", city="Paris"), "Let me check.", [("get_time", {"city": "Paris"})]),
    (qwen_call("mixed", n="7", flag="true", any='{"a": [1]}', items="[1, 2]"), "",
     [("mixed", {"n": 7, "flag": True, "any": {"a": [1]}, "items": [1, 2]})]),
    (qwen_call("mixed", n="seven", flag="maybe", any="plain text", items="not json"), "",
     [("mixed", {"n": "seven", "flag": "maybe", "any": "plain text", "items": "not json"})]),
    (qwen_call("mixed", n='"7"'), "", [("mixed", {"n": '"7"'})]),  # (a string's text as written)
    (qwen_call("unknown", x="1"), "", [("unknown", {"x": 1})]),
    # lenient: no newlines around a value, no </tool_call>
    ("<tool_call><function=get_time><parameter=city>Rome</parameter></function>", "", [("get_time", {"city": "Rome"})]),
    ("Hello, world!\nWhat's up?", "Hello, world!\nWhat's up?", []),
    ("a <tool_call> b", "a <tool_call> b", []),
    ("text <tool_call>\n<function", "text <tool_call>\n<function", []),
    ("x\n\n", "x", []),
]


@pytest.mark.parametrize("text,content,calls", QWEN)
def test_qwen(text: str, content: str, calls: list) -> None:
    check(QWEN_XML, text, content, calls)


def test_qwen_streams_strings() -> None:
    """A string parameter's fragments go out as its text comes, holding back only what may begin its end."""
    p = QWEN_XML.parser(TOOLS)
    out = p.parts([("content", "<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\nprint(2)")])
    assert out[0] == ("tool_call", out[0][1]) and out[0][1].name == "python"
    assert "".join(c.arguments for _, c in out) == '{"code": "print(1)\\nprint(2)'
    assert p.calling and not p.done
    out = p.parts([("content", "\n</param")])
    assert out == []
    out = p.parts([("content", "eter>\n</function>\n</tool_call>")]) + p.flush()
    assert "".join(c.arguments for _, c in out) == '"}'


def test_qwen_not_parallel() -> None:
    text = qwen_call("get_time", city="A") + "\n" + qwen_call("get_time", city="B")
    check(QWEN_XML, text, "", [("get_time", {"city": "A"})], parallel=False)
    p = QWEN_XML.parser(TOOLS, parallel=False)
    p.parts([("content", qwen_call("get_time", city="A"))])
    assert p.done


def test_qwen_unfinished() -> None:
    """A call cut off (max_tokens): no call is complete; reasoning parts pass through."""
    p = QWEN_XML.parser(TOOLS)
    out = p.parts([("reasoning_content", "hmm"), ("content", "<tool_call>\n<function=python>\n<parameter=code>\nx")])
    out += p.flush()
    assert out[0] == ("reasoning_content", "hmm") and p.calls == 0 and p.started == 1


GEMMA = [
    ('<|tool_call>call:get_time{city:<|"|>London<|"|>}<tool_call|>', "", [("get_time", {"city": "London"})]),
    ('<|tool_call>call:get_time{city:<|"|>San Francisco<|"|>}<tool_call|>', "",
     [("get_time", {"city": "San Francisco"})]),
    ("<|tool_call>call:empty_args{}<tool_call|>", "", [("empty_args", {})]),
    ('Hello, world!\nWhat\'s up?<|tool_call>call:get_time{city:<|"|>Paris<|"|>}<tool_call|>',
     "Hello, world!\nWhat's up?", [("get_time", {"city": "Paris"})]),
    (('<|tool_call>call:get_time{city:<|"|>London<|"|>}<tool_call|><|tool_call>call:get_weather{city:<|"|>Paris<|"|>}'
      "<tool_call|>"), "", [("get_time", {"city": "London"}), ("get_weather", {"city": "Paris"})]),
    ("<|tool_call>call:special_function{arg1:42}<tool_call|>", "", [("special_function", {"arg1": 42})]),
    ("<|tool_call>call:special_function{arg1:-7}<tool_call|>", "", [("special_function", {"arg1": -7})]),
    ("<|tool_call>call:amount{orig:3.14}<tool_call|>", "", [("amount", {"orig": 3.14})]),
    ("<|tool_call>call:amount{orig:1.5e10}<tool_call|>", "", [("amount", {"orig": 1.5e10})]),
    ("<|tool_call>call:toggle{enabled:true}<tool_call|>", "", [("toggle", {"enabled": True})]),
    ("<|tool_call>call:toggle{enabled:false}<tool_call|>", "", [("toggle", {"enabled": False})]),
    ("<|tool_call>call:set_nullable{value:null}<tool_call|>", "", [("set_nullable", {"value": None})]),
    ('<|tool_call>call:todo_list{todos:[<|"|>buy milk<|"|>,<|"|>walk dog<|"|>]}<tool_call|>', "",
     [("todo_list", {"todos": ["buy milk", "walk dog"]})]),
    ('<|tool_call>call:set_config{config:{theme:<|"|>dark<|"|>,count:3}}<tool_call|>', "",
     [("set_config", {"config": {"theme": "dark", "count": 3}})]),
    ("<|tool_call>call:todo_list{todos:[]}<tool_call|>", "", [("todo_list", {"todos": []})]),
    ("<|tool_call>call:set_config{config:{}}<tool_call|>", "", [("set_config", {"config": {}})]),
    ('<|tool_call>call:f{<|"|>quoted key<|"|>:1,text:<|"|>a, b: {c}<|"|>}<tool_call|>', "",
     [("f", {"quoted key": 1, "text": "a, b: {c}"})]),
    ("Hello, world!\nWhat's up?", "Hello, world!\nWhat's up?", []),
    ("x <|tool_call>call:broken{a:<tool_call|> y", "x <|tool_call>call:broken{a: y", []),
]


@pytest.mark.parametrize("text,content,calls", GEMMA)
def test_gemma(text: str, content: str, calls: list) -> None:
    check(GEMMA4, text, content, calls)


LLAMA = [
    ('{"name": "special_function", "parameters": {"arg1": 1}}', "", [("special_function", {"arg1": 1})]),
    ('{"type": "function", "name": "get_time", "parameters": {"city": "Paris"}}', "",
     [("get_time", {"city": "Paris"})]),
    (' {"name": "a", "parameters": {"x": 1}}; {"name": "b", "arguments": {"y": "z"}}', "",
     [("a", {"x": 1}), ("b", {"y": "z"})]),
    ('{"name": "a", "parameters": "{\\"x\\": 1}"}', "", [("a", {"x": 1})]),
    # (Llama 3.1 quotes values: typed by the schema)
    ('{"name": "special_function", "parameters": {"arg1": "1"}}', "", [("special_function", {"arg1": 1})]),
    ('{"name": "mixed", "parameters": {"n": "7", "flag": "False", "items": "[1, \'a\']"}}', "",
     [("mixed", {"n": "7", "flag": False, "items": [1, "a"]})]),
    ("Hello, world!\nWhat's up?", "Hello, world!\nWhat's up?", []),
    ('{"answer": 42}', '{"answer": 42}', []),
    ('{"name": "a", "parameters": {"x": 1}} and more', '{"name": "a", "parameters": {"x": 1}} and more', []),
]


@pytest.mark.parametrize("text,content,calls", LLAMA)
def test_llama(text: str, content: str, calls: list) -> None:
    check(LLAMA3, text, content, calls)


def test_typed() -> None:
    """Strings where a schema allows none: the value they encode, if of an allowed type."""
    i, b, arr = {"type": "integer"}, {"type": "boolean"}, {"type": "array"}
    assert typed("1", i) == 1 and typed("1.5", {"type": "number"}) == 1.5 and typed("false", b) is False
    assert typed("False", b) is False and typed("[0.8, 0.7]", arr) == [0.8, 0.7] and typed("('a', 'b')", arr) == ["a", "b"]
    assert typed("x", i) == "x" and typed("1.5", i) == "1.5" and typed("{1, 2}", arr) == "{1, 2}"
    assert typed("1", {"type": ["integer", "string"]}) == "1" and typed("1", None) == "1" and typed(3, i) == 3


def test_openers() -> None:
    """A forced opener, then the model's text: the parser reads them as one call."""
    for syntax, rest, name in [(QWEN_XML, "get_time>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
                                None),
                               (QWEN_XML, "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
                                "get_time"),
                               (GEMMA4, 'get_time{city:<|"|>Oslo<|"|>}<tool_call|>', None),
                               (GEMMA4, 'city:<|"|>Oslo<|"|>}<tool_call|>', "get_time"),
                               (LLAMA3, 'get_time", "parameters": {"city": "Oslo"}}', None),
                               (LLAMA3, '{"city": "Oslo"}}', "get_time")]:
        check(syntax, syntax.opener(name) + rest, "", [("get_time", {"city": "Oslo"})])


def test_detection() -> None:
    """The three syntaxes from templates like the model files' (their marker lines), none from a template without
    tools."""
    qwen = "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{{- '<parameter=' + args_name + '>\\n' }}"
    gemma = "{{- '<|tool_call>call:' + function['name'] + '{' -}}"
    llama = ("<|start_header_id|>{%- if tools is not none %}{{- '{\"name\": \"' + tool_call.name + '\", ' }}"
             "{{- '\"parameters\": ' }}{%- endif %}")
    assert syntax_of(qwen) is QWEN_XML and syntax_of(gemma) is GEMMA4 and syntax_of(llama) is LLAMA3
    assert syntax_of("{% for message in messages %}{{ message['content'] }}{% endfor %}") is None


def test_functions_and_arguments() -> None:
    assert functions(None) == []
    assert functions([{"type": "function", "function": {"name": "f", "strict": True}}]) == \
        [{"type": "function", "function": {"name": "f", "description": "", "parameters": {}}}]
    for bad in ({"name": "f"}, [{"type": "code_interpreter"}], [{"type": "function", "function": {}}]):
        with pytest.raises(ValueError):
            functions(bad)
    m = {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"a": [1]}'}},
                        {"id": "2", "type": "function", "function": {"name": "g", "arguments": ""}},
                        {"id": "3", "type": "function", "function": {"name": "h", "arguments": {"b": 2}}}]}
    out = call_arguments(m)
    assert [c["function"]["arguments"] for c in out["tool_calls"]] == [{"a": [1]}, {}, {"b": 2}]
    assert m["tool_calls"][0]["function"]["arguments"] == '{"a": [1]}'  # (not changed in place)
    with pytest.raises(ValueError):
        call_arguments({"tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]})


FILES = {"qwen": (lambda: qwen_gguf("Q4_K_M", "35B-A3B"), QWEN_XML), "minicpmv": (minicpmv_gguf, QWEN_XML),
         "gemma": (gemma_gguf, GEMMA4), "llama": (lambda: llama_gguf("Q4_K_M", "8B"), LLAMA3)}


@pytest.mark.parametrize("family", FILES)
def test_model_templates(family: str) -> None:
    """The model files' templates: their syntax, and a call with its result rendered as the model writes it."""
    path, syntax = FILES[family][0](), FILES[family][1]
    if not os.path.exists(path):
        pytest.skip(f"{path}: not here")
    g = GGUFFile(path)
    assert syntax_of(str(g.meta["tokenizer.chat_template"])) is syntax
    fns = functions([tool("get_time", {"city": {"type": "string"}})])
    msgs = [{"role": "user", "content": "Time in Oslo?"},
            call_arguments({"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "get_time", "arguments": '{"city": "Oslo"}'}}]}),
            {"role": "tool", "tool_call_id": "c0", "content": "09:30"}]
    if family == "llama":
        from models.llama.tokenizer import chat_prompt
        text = chat_prompt(g, msgs, tools=fns, date="26 Jul 2024")
        assert '{"name": "get_time", "parameters": {"city": "Oslo"}}<|eot_id|>' in text
        # (several calls in a message: a message each, as the template takes one)
        two = call_arguments({**msgs[1], "tool_calls": msgs[1]["tool_calls"] * 2})
        assert chat_prompt(g, [msgs[0], two], tools=fns).count('<|end_header_id|>\n\n{"name": "get_time"') == 2
    elif family == "gemma":
        from models.gemma4.tokenizer import chat_prompt
        text = chat_prompt(g, msgs, tools=fns)
        assert "<|tool_call>call:get_time{city:<|\"|>Oslo<|\"|>}<tool_call|>" in text
        # (the answer's content closed the model's turn: a new one opens)
        assert text.endswith("Checking.<turn|>\n<|turn>model\n")
        assert chat_prompt(g, msgs[:1], tools=fns).endswith("<|turn>user\nTime in Oslo?<turn|>\n<|turn>model\n")
    else:
        from models.qwen35.tokenizer import chat_prompt
        text = chat_prompt(g, msgs, tools=fns)
        assert "<tool_call>\n<function=get_time>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>" in text
        assert '"name": "get_time"' in text
    assert "09:30" in text
