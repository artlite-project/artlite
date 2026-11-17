"""Tool calls (OpenAI's function calling): a request's functions as the chat templates take them (``functions``), the
arguments of the calls in its messages as objects (``call_arguments``: the templates iterate them), and the tool-call
syntax of a model's chat template (``syntax_of``, detected from the template's source as llama.cpp detects it): how the
model writes a call, parsed as the text streams (``Parser``: text in, content and calls out) and opened by force
(``Syntax.opener``: ``tool_choice`` required or a named function).

| syntax | templates | a call in the generated text |
|---|---|---|
| ``QwenXml`` | Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7, LFM3 | ``<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>`` |
| ``Gemma4`` | Gemma 4 | ``<|tool_call>call:NAME{k:V,...}<tool_call|>``, strings between ``<|"|>`` |
| ``Llama3`` | Llama 3.1 / 3.2 | the whole message ``{"name": NAME, "parameters": {...}}`` |
| ``Glm45`` | Spark-X2.5 / X2.6 (GLM-4.5's syntax) | ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`` |

A Qwen parameter's value is the text up to ``</parameter>`` (one newline after the tag and one before the closing tag
are the syntax's): a string where the function's schema allows only strings, streamed as it comes; otherwise JSON when
it parses to a type the schema allows, else the text (llama.cpp's rule), a Python literal where the schema allows no
string (Qwen writes True and False). A GLM-4.5 value (``<arg_value>``) the same, white space between the tags
(GLM-4.5's newlines) dropped. A Gemma or Llama call goes out whole at its end, its arguments typed the same way
(``typed``). A call that does not parse is content.

A call is held to the request's functions by a grammar built from these parsers' texts (``call_grammar``): its name
is one of theirs, its arguments follow the function's schema in this syntax. Without it a model invents names
(``tool_choice`` required where no function fits: ``tool_call``, ``none``) and arguments.

OpenAI's custom tools (text in a grammar's form: ``Custom``) are functions of one string parameter ``input``
(``functions``, ``customs``); a call's input is that argument (``custom_input``; as it streams: ``InputStream``)."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .ids import alnum_id
from .textflow import held


def call_id() -> str:
    """A new tool call's id (OpenAI's form)."""
    return alnum_id(24, "call_")


INPUT = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}  # (a custom tool's)


@dataclass(frozen=True)
class Custom:
    """A custom tool's input format (OpenAI's ``custom`` tools): ``syntax`` text (any text), lark or regex, its
    ``definition``."""

    syntax: str = "text"
    definition: str = ""


def _custom(t: dict[str, Any]) -> tuple[str, str, Custom]:
    """A custom tool's name, description and format (``{"type": "custom", "custom": {"name", "description",
    "format"}}``; a format ``{"type": "text"}`` or ``{"type": "grammar", "grammar": {"syntax": "lark" or "regex",
    "definition"}}``). ValueError: not one."""
    c = t.get("custom")
    if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"]:
        raise ValueError(f"a custom tool must be {{\"type\": \"custom\", \"custom\": {{\"name\": ...}}}}: {t!r}")
    fmt = c.get("format") or {"type": "text"}
    g = fmt.get("grammar") if isinstance(fmt, dict) else None
    if isinstance(fmt, dict) and fmt.get("type") == "text":
        return c["name"], c.get("description") or "", Custom()
    if not isinstance(fmt, dict) or fmt.get("type") != "grammar" or not isinstance(g, dict) or \
            g.get("syntax") not in ("lark", "regex") or not isinstance(g.get("definition"), str):
        raise ValueError(f"custom tool {c['name']!r}: a format {{\"type\": \"text\"}} or {{\"type\": \"grammar\", "
                         f"\"grammar\": {{\"syntax\": \"lark\" or \"regex\", \"definition\": ...}}}}: {fmt!r}")
    return c["name"], c.get("description") or "", Custom(g["syntax"], g["definition"])


def functions(tools: object) -> list[dict[str, Any]]:
    """A request's ``tools`` as the templates take them (llama.cpp's form): ``{"type": "function", "function": {"name",
    "description", "parameters"}}`` each; a custom tool (OpenAI's ``{"type": "custom", ...}``) a function of one
    string parameter ``input`` (its format held by the call grammar: ``customs``). ValueError: not a list of
    functions and custom tools."""
    if tools is None:
        return []
    out = []
    for t in tools if isinstance(tools, list) else [tools]:
        if isinstance(t, dict) and t.get("type") == "custom":
            name, description, _ = _custom(t)
            out.append({"type": "function", "function": {"name": name, "description": description,
                                                         "parameters": INPUT}})
            continue
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(t, dict) or t.get("type") != "function" or not isinstance(f, dict) or \
                not isinstance(f.get("name"), str) or not f["name"]:
            raise ValueError(f"a tool must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}: {t!r}")
        out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description") or "",
                                                     "parameters": f.get("parameters") or {}}})
    return out


def customs(tools: object) -> dict[str, Custom]:
    """A request's custom tools' formats by name (its tools checked: ``functions``)."""
    return {n: c for n, _, c in (_custom(t) for t in tools or [] if isinstance(t, dict) and t.get("type") == "custom")}


def tool_name(t: dict[str, Any]) -> object:
    """The name of a tool (or a ``tool_choice``'s): ``{"type": "function", "function": {"name"}}`` or
    ``{"type": "custom", "custom": {"name"}}``."""
    inner = t.get("custom" if t.get("type") == "custom" else "function")
    return inner.get("name") if isinstance(inner, dict) else None


def custom_input(arguments: str) -> str:
    """A custom tool's input from its call's arguments (``{"input": ...}``; others: their text)."""
    try:
        args = json.loads(arguments)
    except ValueError:
        return arguments
    return args["input"] if isinstance(args, dict) and isinstance(args.get("input"), str) else arguments


class InputStream:
    """A custom tool call's input as its arguments stream (``{"input": "...``, the parsers' JSON text): ``feed`` takes
    the next fragment and returns the input's new text (its string decoded as far as its escapes are complete)."""

    HEAD = re.compile(r'\s*\{\s*"input"\s*:\s*"')

    def __init__(self) -> None:
        self.args, self.sent = "", ""

    def feed(self, fragment: str) -> str:
        self.args += fragment
        m = self.HEAD.match(self.args)
        if m is None:
            return ""
        body, i = self.args[m.end():], 0
        while i < len(body) and body[i] != '"':
            if body[i] == "\\":
                n = 6 if body[i + 1:i + 2] == "u" else 2
                if i + n > len(body):
                    break
                i += n
            else:
                i += 1
        text = json.loads(f'"{body[:i]}"')
        if text and "\ud800" <= text[-1] <= "\udbff":  # (half a surrogate pair: the rest comes next)
            text = text[:-1]
        delta, self.sent = text[len(self.sent):], text
        return delta


def call_arguments(message: dict[str, Any]) -> dict[str, Any]:
    """A message whose tool calls' ``arguments`` (JSON text in OpenAI's API) are objects, as the templates iterate
    them; a custom tool's call (``{"type": "custom", "custom": {"name", "input"}}``) a function's, its input the
    argument ``input``. ValueError: arguments that are not a JSON object."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return message
    out = []
    for c in calls:
        if isinstance(c, dict) and c.get("type") == "custom" and isinstance(c.get("custom"), dict):
            out.append({"id": c.get("id"), "type": "function", "function": {
                "name": c["custom"].get("name"), "arguments": {"input": str(c["custom"].get("input") or "")}}})
            continue
        f = c.get("function") if isinstance(c, dict) else None
        if isinstance(f, dict) and isinstance(f.get("arguments"), str):
            try:
                args = json.loads(f["arguments"]) if f["arguments"].strip() else {}
            except ValueError:
                raise ValueError(f"tool call arguments are not JSON: {f['arguments']!r}") from None
            c = {**c, "function": {**f, "arguments": args}}
        out.append(c)
    return {**message, "tool_calls": out}


@dataclass(frozen=True)
class ToolCall:
    """A part of a tool call in the generated text: its first part has ``id`` and ``name``, the others (and the
    first) carry fragments of the arguments' JSON text."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


Part = tuple[str, Any]  # ("content", text), ("reasoning_content", text) or ("tool_call", ToolCall)


def resolve(root: dict[str, Any], schema: object) -> object:
    """A function's parameter schema with its local ``$ref`` followed (``#/$defs/...``, cycles stopped): what the
    parser types a value by and the call grammar holds it to."""
    seen: set[str] = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str) and schema["$ref"].startswith("#/") and \
            schema["$ref"] not in seen:
        seen.add(schema["$ref"])
        node: Any = root
        for part in schema["$ref"][2:].split("/"):
            node = node.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            break
        schema = node
    return schema


def _types(schema: object) -> set[str]:
    """The JSON types a schema allows (empty: any)."""
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    out: set[str] = set()
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            s = _types(sub)
            if not s:
                return set()
            out |= s
    values = schema.get("enum", [schema["const"]] if "const" in schema else [])
    for v in values if isinstance(values, list) else []:
        out.add({str: "string", bool: "boolean", int: "integer", float: "number", type(None): "null"}.get(type(v),
                                                                                                          "object"))
    return out


def _allows(value: object, types: set[str]) -> bool:
    kind = {bool: "boolean", int: "integer", float: "number", type(None): "null", list: "array", dict: "object",
            str: "string"}.get(type(value), "")
    return kind in types or (kind == "integer" and "number" in types)


def typed(value: object, schema: object) -> object:
    """A value the model wrote as a string where its schema allows none (Llama 3 quotes numbers, booleans and lists):
    the value the string encodes when it is of an allowed type (JSON, else a Python literal), else the string."""
    return _typed(value, _types(schema))


def _typed(value: object, types: set[str]) -> object:
    if not isinstance(value, str) or not types or "string" in types:
        return value
    for parse in (json.loads, ast.literal_eval):
        try:
            v = json.loads(json.dumps(parse(value)))  # (a literal's tuples as lists)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        if _allows(v, types):
            return v
    return value


class Parser:
    """A generation's text split into content and tool calls as it streams: ``parts`` passes the reasoning parts and
    parses the content parts; ``flush`` at the end. ``calls``: the calls complete so far; ``calling``: a call has
    begun; ``done``: the generation ends (a call is complete and ``parallel`` is off: what follows is dropped).
    Subclasses parse ``_feed`` / ``_flush``; a call's name comes after ``NAME_HEAD`` and before ``NAME_END`` (the call
    grammars write them: ``call_grammar``)."""

    NAME_HEAD, NAME_END = "", ""

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        self.functions = {f["function"]["name"]: f["function"] for f in functions}
        self.parallel = parallel
        self.calls = 0
        self.started = 0  # calls begun
        self.buf = ""
        self.gap = ""  # (the white space before a call's opener: content again if it is not a call)
        self._after = False  # (a call just ended: the white space after it is dropped)

    @property
    def calling(self) -> bool:
        return self.started > self.calls

    @property
    def done(self) -> bool:
        return self.calls > 0 and not self.parallel

    def parts(self, parts: list[Part]) -> list[Part]:
        out: list[Part] = []
        for kind, value in parts:
            if kind == "content":
                out += self._feed(value)
            else:
                out.append((kind, value))
        return out

    def flush(self) -> list[Part]:
        return self._flush()

    def _feed(self, text: str) -> list[Part]:
        raise NotImplementedError

    def _flush(self) -> list[Part]:
        raise NotImplementedError

    def _text(self, out: list[Part], opener: str, *, final: bool = False) -> int:
        """Content up to ``opener`` (where a call begins) or up to what may begin it; white space before a call or
        between calls is dropped. Returns where the opener begins in the buffer, -1 if not there (the rest held)."""
        i = self.buf.find(opener)
        if i >= 0:
            text = self.buf[:i].rstrip()
            self.gap, self.buf = self.buf[len(text):i], self.buf[i + len(opener):]
            self._content(out, text)
            return 0
        k = 0 if final else held(self.buf, (opener,))
        body = self.buf[:len(self.buf) - k]
        text = body.rstrip()
        self.buf = body[len(text):] + self.buf[len(self.buf) - k:]
        if final:
            self.buf = ""
        self._content(out, text)
        return -1

    def _content(self, out: list[Part], text: str) -> None:
        if self._after:
            text = text.lstrip()
        if text:
            self._after = False
            out.append(("content", text))

    def _ended(self) -> None:
        """A call is complete."""
        self.calls += 1
        self._after = True

    def _begin(self, name: str, arguments: str = "") -> ToolCall:
        self.started += 1
        return ToolCall(self.started - 1, call_id(), name, arguments)

    def _arguments(self, name: str, args: dict[str, Any]) -> str:
        """A whole call's arguments as JSON text, typed by the function's schema (``typed``)."""
        root = (self.functions.get(name) or {}).get("parameters") or {}
        props = root.get("properties") or {}
        return json.dumps({k: typed(v, resolve(root, props.get(k))) for k, v in args.items()}, ensure_ascii=False)


class QwenParser(Parser):
    OPEN, CLOSE = "<tool_call>", "</tool_call>"
    FUNCTION = re.compile(r"\s*<function=([^>\n]+)>")
    PARAMETER = re.compile(r"\s*<parameter=([^>\n]+)>")
    PARAM_HEAD, END, FUNCTION_END = "<parameter=", "</parameter>", "</function>"
    NAME_HEAD, NAME_END = "\n<function=", ">"  # (the template's: one newline after <tool_call>)

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.state = "text"  # text, function (after <tool_call>), args, value, end (after </function>)
        self.name = ""
        self.n_args = 0
        self.key = ""
        self.value = ""  # a parameter's text so far (a non-string's, or a string's not sent)
        self.lead = True  # (a value's first newline not yet seen)

    def _param_types(self, key: str) -> set[str]:
        root = (self.functions.get(self.name) or {}).get("parameters") or {}
        return _types(resolve(root, (root.get("properties") or {}).get(key)))

    def _arg(self, out: list[Part], text: str) -> None:
        out.append(("tool_call", ToolCall(self.started - 1, arguments=text)))

    def _feed(self, text: str) -> list[Part]:
        self.buf += text
        out: list[Part] = []
        while self._step(out):
            pass
        return out

    def _step(self, out: list[Part], *, final: bool = False) -> bool:
        """One step of the parse; False: it waits for more text."""
        b = self.buf
        if self.state == "text":
            if self.done:  # (the generation ends here)
                self.buf = ""
                return False
            if self._text(out, self.OPEN, final=final) < 0:
                return False
            self.state = "function"
            return True
        if self.state == "function":
            m = self.FUNCTION.match(b)
            if m:
                self.name, self.n_args, self.buf = m.group(1).strip(), 0, b[m.end():]
                out.append(("tool_call", self._begin(self.name)))
                self.state = "args"
                return True
            s = b.lstrip()
            if not final and len(s) < 256 and ("<function=".startswith(s) or (s.startswith("<function=") and
                                                                              ">" not in s and "\n" not in s)):
                return False
            self._content(out, self.gap + self.OPEN + b)  # (not a call)
            self.state, self.buf = "text", ""
            return True
        if self.state == "args":
            m = self.PARAMETER.match(b)
            s = b.lstrip()
            if m:
                self.key, self.buf = m.group(1).strip(), b[m.end():]
                self._arg(out, ("{" if self.n_args == 0 else ", ") + json.dumps(self.key, ensure_ascii=False) + ": ")
                self.n_args += 1
                self.value, self.lead = "", True
                if self._param_types(self.key) == {"string"}:
                    self._arg(out, '"')
                self.state = "value"
                return True
            if s.startswith(self.FUNCTION_END):
                self._arg(out, "}" if self.n_args else "{}")
                self.buf, self.state = s[len(self.FUNCTION_END):], "end"
                return True
            if not final and (any(w.startswith(s) for w in (self.PARAM_HEAD, self.FUNCTION_END)) or
                              (s.startswith(self.PARAM_HEAD) and ">" not in s and "\n" not in s)):
                return False
            self._arg(out, "}" if self.n_args else "{}")  # (something else: the call ends here)
            self._ended()
            self.state = "text"
            return True
        if self.state == "value":
            stream = self._param_types(self.key) == {"string"}
            if self.lead and b:
                b = b[1:] if b[0] == "\n" else b
                self.lead = False
            elif self.lead:
                return False
            i = b.find(self.END)
            if i >= 0:
                text, self.buf = b[:i], b[i + len(self.END):]
                text = self.value + text.removesuffix("\n")
                if stream:
                    self._arg(out, json.dumps(text, ensure_ascii=False)[1:] if text else '"')
                else:
                    self._arg(out, json.dumps(self._value(text), ensure_ascii=False))
                self.state = "args"
                return True
            if final:
                self.buf = ""
                return False
            k = held(b, ("\n" + self.END, self.END))
            text, self.buf = b[:len(b) - k], b[len(b) - k:]
            if stream and text:
                self._arg(out, json.dumps(self.value + text, ensure_ascii=False)[1:-1])
                self.value = ""
            else:
                self.value += text
            return False
        # end: </tool_call> after </function> (a call that lacks it ends anyway)
        s = b.lstrip()
        if s.startswith(self.CLOSE):
            self.buf = s[len(self.CLOSE):]
        elif not final and self.CLOSE.startswith(s):
            return False
        self._ended()
        self.state = "text"
        return True

    def _value(self, text: str) -> object:
        types = self._param_types(self.key)
        if types == {"string"}:
            return text
        try:
            v = json.loads(text)
        except ValueError:
            return _typed(text, types)  # (Qwen writes True and False as Python does)
        if isinstance(v, str) or (types and "string" in types and not _allows(v, types)):
            return text
        return v

    def _flush(self) -> list[Part]:
        out: list[Part] = []
        while self._step(out, final=True):
            pass
        return out


class Glm45Parser(Parser):
    """GLM-4.5's calls (``Glm45``): ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>..</tool_call>``, white
    space between the elements allowed; a value typed as a Qwen parameter's (``QwenParser._value``), a string-only
    one streamed."""

    OPEN, CLOSE = "<tool_call>", "</tool_call>"
    KEY, KEY_END, VALUE, VALUE_END = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
    NAME_HEAD, NAME_END = "", "<"  # (the name, then a tag)

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.state = "text"  # text, name (after <tool_call>), args, key, open (before <arg_value>), value
        self.name = ""
        self.n_args = 0
        self.key = ""
        self.value = ""

    _param_types = QwenParser._param_types
    _arg = QwenParser._arg
    _value = QwenParser._value
    _feed = QwenParser._feed
    _flush = QwenParser._flush

    def _step(self, out: list[Part], *, final: bool = False) -> bool:
        """One step of the parse; False: it waits for more text."""
        b = self.buf
        if self.state == "text":
            if self.done:
                self.buf = ""
                return False
            if self._text(out, self.OPEN, final=final) < 0:
                return False
            self.state = "name"
            return True
        if self.state == "name":
            i = b.find("<")
            name = b[:i].strip() if i >= 0 else ""
            if i >= 0 and name and "\n" not in name and (b[i:].startswith((self.KEY, self.CLOSE)) or
                                                         final or not any(t.startswith(b[i:]) for t in
                                                                          (self.KEY, self.CLOSE))):
                self.name, self.n_args, self.buf = name, 0, b[i:]
                out.append(("tool_call", self._begin(self.name)))
                self.state = "args"
                return True
            if not final and i < 0 and len(b) < 256 and "\n" not in b.strip():
                return False
            if not final and i >= 0 and name and any(t.startswith(b[i:]) for t in (self.KEY, self.CLOSE)):
                return False
            self._content(out, self.gap + self.OPEN + b)  # (not a call)
            self.state, self.buf = "text", ""
            return True
        if self.state == "args":
            s = b.lstrip()
            if s.startswith(self.KEY):
                self.buf, self.state = s[len(self.KEY):], "key"
                return True
            if s.startswith(self.CLOSE):
                self._arg(out, "}" if self.n_args else "{}")
                self.buf = s[len(self.CLOSE):]
                self._ended()
                self.state = "text"
                return True
            if not final and any(t.startswith(s) for t in (self.KEY, self.CLOSE)):
                return False
            self._arg(out, "}" if self.n_args else "{}")  # (something else: the call ends here)
            self._ended()
            self.state = "text"
            return True
        if self.state == "key":
            i = b.find(self.KEY_END)
            if i < 0:
                if final:
                    self.buf = ""
                    self._arg(out, "}" if self.n_args else "{}")
                    self._ended()
                    self.state = "text"
                return False
            self.key, self.buf = b[:i].strip(), b[i + len(self.KEY_END):]
            self._arg(out, ("{" if self.n_args == 0 else ", ") + json.dumps(self.key, ensure_ascii=False) + ": ")
            self.n_args += 1
            self.state = "open"
            return True
        if self.state == "open":
            s = b.lstrip()
            if s.startswith(self.VALUE):
                self.buf, self.value, self.state = s[len(self.VALUE):], "", "value"
                if self._param_types(self.key) == {"string"}:
                    self._arg(out, '"')
                return True
            if not final and self.VALUE.startswith(s):
                return False
            self._arg(out, "null")  # (a key without its value)
            self.state = "args"
            return True
        # value: the text up to </arg_value>
        stream = self._param_types(self.key) == {"string"}
        i = b.find(self.VALUE_END)
        if i >= 0:
            text, self.buf = self.value + b[:i], b[i + len(self.VALUE_END):]
            if stream:
                self._arg(out, json.dumps(text, ensure_ascii=False)[1:] if text else '"')
            else:
                self._arg(out, json.dumps(self._value(text), ensure_ascii=False))
            self.state = "args"
            return True
        if final:
            self.buf = ""
            return False
        k = held(b, (self.VALUE_END,))
        text, self.buf = b[:len(b) - k], b[len(b) - k:]
        if stream and text:
            self._arg(out, json.dumps(self.value + text, ensure_ascii=False)[1:-1])
            self.value = ""
        else:
            self.value += text
        return False


class _GemmaValue:
    """Gemma 4's value syntax (its template's ``format_argument``): strings between ``<|"|>``, numbers, true / false /
    null, ``{key:value,...}`` with bare keys, ``[value,...]``."""

    Q = '<|"|>'
    NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def value(self) -> object:
        s = self.s
        self.ws()
        if s.startswith(self.Q, self.i):
            j = s.find(self.Q, self.i + len(self.Q))
            if j < 0:
                raise ValueError("an unterminated string")
            out, self.i = s[self.i + len(self.Q):j], j + len(self.Q)
            return out
        if s.startswith("{", self.i):
            return self.dict()
        if s.startswith("[", self.i):
            self.i += 1
            items: list[object] = []
            self.ws()
            if s.startswith("]", self.i):
                self.i += 1
                return items
            while True:
                items.append(self.value())
                self.ws()
                if s.startswith(",", self.i):
                    self.i += 1
                elif s.startswith("]", self.i):
                    self.i += 1
                    return items
                else:
                    raise ValueError("expected , or ]")
        for word, v in (("true", True), ("false", False), ("null", None)):
            if s.startswith(word, self.i):
                self.i += len(word)
                return v
        m = self.NUMBER.match(s, self.i)
        if not m:
            raise ValueError(f"unexpected text at {self.i}")
        self.i = m.end()
        return json.loads(m.group())

    def dict(self) -> dict[str, object]:
        s = self.s
        assert s[self.i] == "{"
        self.i += 1
        out: dict[str, object] = {}
        self.ws()
        if s.startswith("}", self.i):
            self.i += 1
            return out
        while True:
            self.ws()
            if s.startswith(self.Q, self.i):
                key = self.value()
            else:
                j = s.find(":", self.i)
                if j < 0:
                    raise ValueError("a key without a value")
                key = s[self.i:j].strip()
                self.i = j
            self.ws()
            if not s.startswith(":", self.i):
                raise ValueError("expected :")
            self.i += 1
            out[str(key)] = self.value()
            self.ws()
            if s.startswith(",", self.i):
                self.i += 1
            elif s.startswith("}", self.i):
                self.i += 1
                return out
            else:
                raise ValueError("expected , or }")


class GemmaParser(Parser):
    OPEN, CLOSE = "<|tool_call>", "<tool_call|>"
    CALL = re.compile(r"\s*call:([^{\s]+)\s*(?={)")
    NAME_HEAD, NAME_END = "call:", "{"

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.in_call = False

    def _close(self) -> int:
        """Where the call's end marker is in the buffer, outside a string (``<|"|>`` pairs), or -1."""
        q, at, inside = _GemmaValue.Q, 0, False
        while True:
            i, j = self.buf.find(self.CLOSE, at), self.buf.find(q, at)
            if j >= 0 and (i < 0 or j < i):
                inside, at = not inside, j + len(q)
            elif i >= 0 and not inside:
                return i
            elif i >= 0:
                at = i + len(self.CLOSE)
            else:
                return -1

    def _call(self, out: list[Part], body: str) -> None:
        """A call's text between its markers as a call, or as content when it does not parse."""
        m = self.CALL.match(body)
        try:
            if not m:
                raise ValueError("no call:NAME{")
            v = _GemmaValue(body)
            v.i = m.end()
            args = v.dict()
            if body[v.i:].strip():
                raise ValueError("text after the arguments")
        except ValueError:
            self._content(out, self.gap + self.OPEN + body)
            return
        out.append(("tool_call", self._begin(m.group(1), self._arguments(m.group(1), args))))
        self._ended()

    def _feed(self, text: str, *, final: bool = False) -> list[Part]:
        self.buf += text
        out: list[Part] = []
        while True:
            if not self.in_call:
                if self.done:  # (the generation ends here)
                    self.buf = ""
                    break
                if self._text(out, self.OPEN, final=final) < 0:
                    break
                self.in_call = True
            i = self._close()
            if i < 0 and not final:
                break
            body, self.buf = (self.buf[:i], self.buf[i + len(self.CLOSE):]) if i >= 0 else (self.buf, "")
            self.in_call = False
            self._call(out, body)
            if final and not self.buf:
                break
        return out

    def _flush(self) -> list[Part]:
        return self._feed("", final=True)


class LlamaParser(Parser):
    """Llama 3's calls: a message that is JSON objects ``{"name": ..., "parameters": {...}}`` (``arguments`` taken
    too), one or more (``;`` or white space between them); held until the end, content when it is anything else.
    The arguments typed by the schema (``typed``: Llama 3.1 writes most values as strings). A name is the string
    value of an object's ``"name"`` key after its other plain keys (``"type": "function"``)."""


    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.state = "start"  # start, json (held), text

    def _feed(self, text: str) -> list[Part]:
        out: list[Part] = []
        if self.state == "start":
            self.buf += text
            s = self.buf.lstrip()
            if not s:
                return out
            self.state = "json" if s.startswith("{") else "text"
            text, self.buf = (s, "") if self.state == "text" else ("", s)
            if self.state == "json":
                self.started = 1  # (a call may be under way: the content waits)
        if self.state == "json":
            self.buf += text
        elif text:
            out.append(("content", text))
        return out

    @staticmethod
    def _calls(s: str) -> list[tuple[str, dict[str, Any]]] | None:
        """The calls of a message's text, or None: it is not calls."""
        calls = []
        dec, i = json.JSONDecoder(), 0
        while i < len(s):
            try:
                obj, i = dec.raw_decode(s, i)
                args = obj.get("parameters", obj.get("arguments")) if isinstance(obj, dict) else None
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                return None
            if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
                return None
            calls.append((obj["name"], args))
            while i < len(s) and (s[i].isspace() or s[i] == ";"):
                i += 1
        return calls

    def _flush(self) -> list[Part]:
        if self.state != "json":
            return []
        self.started = 0
        calls = self._calls(self.buf)
        if calls is None:
            return [("content", self.buf.rstrip())]
        out: list[Part] = []
        for name, args in calls[:None if self.parallel else 1]:
            out.append(("tool_call", self._begin(name, self._arguments(name, args))))
            self._ended()
        return out


@dataclass(frozen=True)
class Syntax:
    """A chat template's tool-call syntax: ``parser`` reads it in generated text; ``opener(name)`` is the text that
    begins a call (of ``name``: the arguments come next), ``gap`` the text between a thought's end and a forced call;
    ``ends``: text that ends an open thought and begins a call (Qwen: a call without ``</think>``)."""

    name: str
    parser_class: type[Parser]
    open: str  # a call's first text
    named: str  # the text that begins a call of {name}, up to its arguments
    gap: str = ""
    ends: tuple[str, ...] = ()

    def parser(self, functions: list[dict[str, Any]], *, parallel: bool = True) -> Parser:
        return self.parser_class(functions, parallel=parallel)

    def opener(self, name: str | None = None) -> str:
        return self.open if name is None else self.named.format(name=name)


@dataclass(frozen=True)
class Calls:
    """The tool calls of a request's answer: its ``functions`` (the ones it may call), the template's ``syntax``,
    ``parallel_tool_calls``, ``force``: the text the answer's call begins with (``tool_choice`` required or a named
    function), or None; ``choice``: auto, required (a call, ``name``: of that function); ``strict``: the functions whose
    schemas compile exactly (the call grammar: ``call_grammar.answer``); ``custom``: the custom tools' formats."""

    syntax: Syntax
    functions: list[dict[str, Any]]
    parallel: bool = True
    force: str | None = None
    choice: str = "auto"
    name: str | None = None
    strict: frozenset[str] = frozenset()
    custom: dict[str, Custom] = field(default_factory=dict)

    def parser(self) -> Parser:
        return self.syntax.parser(self.functions, parallel=self.parallel)


QWEN_XML = Syntax("qwen-xml", QwenParser, "<tool_call>\n<function=", "<tool_call>\n<function={name}>\n", gap="\n\n",
                  ends=("<tool_call>",))
GEMMA4 = Syntax("gemma4", GemmaParser, "<|tool_call>call:", "<|tool_call>call:{name}{{")
LLAMA3 = Syntax("llama3-json", LlamaParser, '{"name": "', '{{"name": "{name}", "parameters": ')
GLM45 = Syntax("glm-4.5", Glm45Parser, "<tool_call>", "<tool_call>{name}")


def syntax_of(template: str) -> Syntax | None:
    """The tool-call syntax of a chat template's source (as llama.cpp's ``common_chat_template_detect_params_init``
    and its autoparser find these four), or None: the template takes no tools."""
    if "'<|tool_call>call:'" in template:
        return GEMMA4
    if "<tool_call>" in template and "<arg_key>" in template and "<arg_value>" in template:
        return GLM45
    if "<tool_call>" in template and "<function=" in template and "<parameter=" in template:
        return QWEN_XML
    if '"parameters": ' in template and "tools" in template and "<|start_header_id|>" in template:
        return LLAMA3
    return None
