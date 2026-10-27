"""Tool calls (OpenAI's function calling): a request's functions as the chat templates take them (``functions``), the
arguments of the calls in its messages as objects (``call_arguments``: the templates iterate them), and the tool-call
syntax of a model's chat template (``syntax_of``, detected from the template's source as llama.cpp detects it): how the
model writes a call, parsed as the text streams (``Parser``: text in, content and calls out) and opened by force
(``Syntax.opener``: ``tool_choice`` required or a named function).

| syntax | templates | a call in the generated text |
|---|---|---|
| ``QwenXml`` | Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7 | ``<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>`` |
| ``Gemma4`` | Gemma 4 | ``<|tool_call>call:NAME{k:V,...}<tool_call|>``, strings between ``<|"|>`` |
| ``Llama3`` | Llama 3.1 / 3.2 | the whole message ``{"name": NAME, "parameters": {...}}`` |

A Qwen parameter's value is the text up to ``</parameter>`` (one newline after the tag and one before the closing tag
are the syntax's): a string where the function's schema allows only strings, streamed as it comes; otherwise JSON when
it parses to a type the schema allows, else the text (llama.cpp's rule), a Python literal where the schema allows no
string (Qwen writes True and False). A Gemma or Llama call goes out whole at its end, its arguments typed the same way
(``typed``). A call that does not parse is content.

A call's name is one of the request's functions: while the model writes it, the sampling's candidates are the tokens
that continue the text towards one of them (``NameMask``, over the tokens by their text: ``Vocabulary``), as
llama.cpp's grammars constrain it. Without it a model invents names (``tool_choice`` required where no function fits:
``tool_call``, ``none``). The arguments are not constrained."""

from __future__ import annotations

import ast
import bisect
import json
import re
import secrets
import string
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..common import Detokenizer, StreamDecoder

ALNUM = string.ascii_letters + string.digits


def call_id() -> str:
    """A new tool call's id (OpenAI's form)."""
    return "call_" + "".join(secrets.choice(ALNUM) for _ in range(24))


def functions(tools: object) -> list[dict[str, Any]]:
    """A request's ``tools`` as the templates take them (llama.cpp's form): ``{"type": "function", "function": {"name",
    "description", "parameters"}}`` each. ValueError: not a list of functions."""
    if tools is None:
        return []
    out = []
    for t in tools if isinstance(tools, list) else [tools]:
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(t, dict) or t.get("type") != "function" or not isinstance(f, dict) or \
                not isinstance(f.get("name"), str) or not f["name"]:
            raise ValueError(f"a tool must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}: {t!r}")
        out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description") or "",
                                                     "parameters": f.get("parameters") or {}}})
    return out


def call_arguments(message: dict[str, Any]) -> dict[str, Any]:
    """A message whose tool calls' ``arguments`` (JSON text in OpenAI's API) are objects, as the templates iterate
    them. ValueError: arguments that are not a JSON object."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return message
    out = []
    for c in calls:
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


def held(text: str, words: tuple[str, ...] | list[str]) -> int:
    """The length of the longest end of ``text`` that begins one of ``words`` (without being all of it): text held
    back until what follows decides."""
    best = 0
    for w in words:
        for k in range(min(len(w) - 1, len(text)), best, -1):
            if text.endswith(w[:k]):
                best = k
                break
    return best


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
    Subclasses parse ``_feed`` / ``_flush`` and say where a call's name is written (``naming``): ``NAME_HEAD``, the
    name, then ``NAME_END``, a name having none of ``NAME_BAD``."""

    NAME_HEAD, NAME_END, NAME_BAD = "", "", ""

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

    def naming(self) -> str | None:
        """The text of the call being written from where ``NAME_HEAD`` begins, while it may still be before the
        name's end; None elsewhere."""
        raise NotImplementedError

    def heads(self) -> list[str]:
        """The text from ``NAME_HEAD`` to the name's end of a call of each function that can be written."""
        return [self.NAME_HEAD + name + self.NAME_END for name in self.functions
                if not any(c in self.NAME_BAD for c in name)]

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
        props = ((self.functions.get(name) or {}).get("parameters") or {}).get("properties") or {}
        return json.dumps({k: typed(v, props.get(k)) for k, v in args.items()}, ensure_ascii=False)


class QwenParser(Parser):
    OPEN, CLOSE = "<tool_call>", "</tool_call>"
    FUNCTION = re.compile(r"\s*<function=([^>\n]+)>")
    PARAMETER = re.compile(r"\s*<parameter=([^>\n]+)>")
    END = "</parameter>"
    NAME_HEAD, NAME_END, NAME_BAD = "\n<function=", ">", ">\n"  # (the template's: one newline after <tool_call>)

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.state = "text"  # text, function (after <tool_call>), args, value, end (after </function>)
        self.name = ""
        self.n_args = 0
        self.key = ""
        self.value = ""  # a parameter's text so far (a non-string's, or a string's not sent)
        self.lead = True  # (a value's first newline not yet seen)

    def _param_types(self, key: str) -> set[str]:
        f = self.functions.get(self.name) or {}
        props = (f.get("parameters") or {}).get("properties") or {}
        return _types(props.get(key))

    def _arg(self, out: list[Part], text: str) -> None:
        out.append(("tool_call", ToolCall(self.started - 1, arguments=text)))

    def naming(self) -> str | None:
        return self.buf if self.state == "function" else None  # (the text after <tool_call>)

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
            if s.startswith("</function>"):
                self._arg(out, "}" if self.n_args else "{}")
                self.buf, self.state = s[len("</function>"):], "end"
                return True
            if not final and (any(w.startswith(s) for w in ("<parameter=", "</function>")) or
                              (s.startswith("<parameter=") and ">" not in s and "\n" not in s)):
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
    NAME_HEAD, NAME_END, NAME_BAD = "call:", "{", "{" + string.whitespace

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.in_call = False

    def naming(self) -> str | None:
        return self.buf if self.in_call and "{" not in self.buf else None  # (the text after <|tool_call>)

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
            i = self.buf.find(self.CLOSE)
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

    NAME_END, NAME_BAD = '"', '"\\'
    NAMING = re.compile(r'\{\s*(?:"[^"\\]*"\s*:\s*(?:"[^"\\]*"|[-+.\w]+)\s*,\s*)*"name"\s*:\s*"([^"\\]*)')

    def __init__(self, functions: list[dict[str, Any]], *, parallel: bool = True):
        super().__init__(functions, parallel=parallel)
        self.state = "start"  # start, json (held), text
        self._at = 0  # (where the call after the complete ones begins)
        self._tried = 0  # (the text up to here could end no more calls)

    def naming(self) -> str | None:
        if self.state != "json":
            return None
        s = self.buf
        while True:
            i = self._at
            while i < len(s) and (s[i].isspace() or s[i] == ";"):
                i += 1
            m = self.NAMING.match(s, i)
            if m and m.end() == len(s):
                return m.group(1)
            if "}" not in s[self._tried:]:  # (a call ends at a "}" only)
                return None
            self._tried = len(s)
            try:
                self._at = self._tried = json.JSONDecoder().raw_decode(s, i)[1]
            except ValueError:
                return None

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
    """The tool calls of a request's answer: its ``functions``, the template's ``syntax``, ``parallel_tool_calls``,
    ``force``: the text the answer's call begins with (``tool_choice`` required or a named function), or None, and
    ``vocab``: the model's tokens for the names' mask (None: names not constrained)."""

    syntax: Syntax
    functions: list[dict[str, Any]]
    parallel: bool = True
    force: str | None = None
    vocab: Vocabulary | None = None

    def parser(self) -> Parser:
        return self.syntax.parser(self.functions, parallel=self.parallel)

    def mask(self, tokenizer: Detokenizer) -> NameMask | None:
        """A mask over a generation's tokens that keeps its calls' names the functions' (``tokenizer``: the
        model's)."""
        return NameMask(self.parser(), tokenizer, self.vocab) if self.vocab is not None else None


class Vocabulary:
    """A model's tokens by their text as the server decodes them (``decode_bytes`` of each of the ``n`` ids; control
    tokens, which decode to nothing, left out), sorted: ``fitting`` finds the tokens that may come where a text must
    follow."""

    def __init__(self, tokenizer: Detokenizer, n: int):
        pairs = sorted((b, i) for i in range(n) if (b := tokenizer.decode_bytes([i])))
        self.texts = [b for b, _ in pairs]
        self.ids = np.fromiter((i for _, i in pairs), np.int32, len(pairs))

    def fitting(self, rest: bytes) -> list[np.ndarray]:
        """The tokens whose text is a part of ``rest`` from its start, or ``rest`` and more (ids, in groups)."""
        texts, out = self.texts, []
        for k in range(1, len(rest)):
            p = rest[:k]
            lo = bisect.bisect_left(texts, p)
            if lo == len(texts) or not texts[lo].startswith(p):
                return out  # (no token begins with p: none is longer)
            if texts[lo] == p:
                out.append(self.ids[lo:bisect.bisect_right(texts, p, lo)])
        lo = bisect.bisect_left(texts, rest)
        end = rest.rstrip(b"\xff")  # (the texts that begin with rest end before the next prefix of its length)
        hi = bisect.bisect_left(texts, end[:-1] + bytes([end[-1] + 1]), lo) if end else len(texts)
        out.append(self.ids[lo:hi])
        return out


class NameMask:
    """The tokens that keep a call's name one of the functions' (``common.Mask``): ``parser``, a parser of the answer's
    syntax, follows the generated text (``accept``, each token as it is sampled: the thoughts' text too, where a call
    does not begin); while it is in a call's name (``Parser.naming``), the tokens ``allowed`` are those that continue
    the text towards a function's head (a part of the rest, or the rest and more), else any."""

    def __init__(self, parser: Parser, tokenizer: Detokenizer, vocab: Vocabulary):
        self.parser, self.vocab = parser, vocab
        self.text = StreamDecoder(tokenizer)
        self.heads = [h.encode() for h in parser.heads()]
        self._memo: dict[bytes, np.ndarray | None] = {}

    def accept(self, token: int) -> None:
        self.parser.parts([("content", self.text.feed(token))])

    def allowed(self) -> np.ndarray | None:
        w = self.parser.naming()
        if w is None:
            return None
        w = w.encode() + self.text.pending
        if w not in self._memo:
            groups = [g for h in self.heads if len(h) > len(w) and h.startswith(w)
                      for g in self.vocab.fitting(h[len(w):])]
            ids = np.unique(np.concatenate(groups)) if groups else np.empty(0, np.int32)
            self._memo[w] = ids if len(ids) else None  # (no head fits: the text left them, nothing to mask)
        return self._memo[w]


QWEN_XML = Syntax("qwen-xml", QwenParser, "<tool_call>\n<function=", "<tool_call>\n<function={name}>\n", gap="\n\n",
                  ends=("<tool_call>",))
GEMMA4 = Syntax("gemma4", GemmaParser, "<|tool_call>call:", "<|tool_call>call:{name}{{")
LLAMA3 = Syntax("llama3-json", LlamaParser, '{"name": "', '{{"name": "{name}", "parameters": ')


def syntax_of(template: str) -> Syntax | None:
    """The tool-call syntax of a chat template's source (as llama.cpp's ``common_chat_template_detect_params_init``
    and its autoparser find these three), or None: the template takes no tools."""
    if "'<|tool_call>call:'" in template:
        return GEMMA4
    if "<tool_call>" in template and "<function=" in template and "<parameter=" in template:
        return QWEN_XML
    if '"parameters": ' in template and "tools" in template and "<|start_header_id|>" in template:
        return LLAMA3
    return None
