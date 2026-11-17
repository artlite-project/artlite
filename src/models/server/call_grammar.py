"""The grammar of an answer with tool calls (``answer``): the calls in the template's own syntax (``tools.Syntax``),
built from the texts its parser reads them by (``tools.QwenParser``, ``Glm45Parser``, ``GemmaParser``,
``LlamaParser``), so that grammar and parser agree; a Lark grammar for llguidance (``structured.Constraint`` lark).

| ``tool_choice`` | the answer |
|---|---|
| auto | free text and calls; ``parallel_tool_calls`` false: one call at most, then the end of the turn |
| required | calls (one with ``parallel_tool_calls`` false) |
| a named function | that function's call alone |
| auto with a ``response_format`` | calls, or the response format's JSON value |

Free text is any token but a call's opener (a special token: Qwen's ``<tool_call>``, Gemma's ``<|tool_call>``) and
the end of the turn (which ends the grammar there): no lexer runs over it (masks of a few microseconds); an opener that
is no special token is found in the text (llguidance's lazy lexemes); Llama 3's calls are the whole message, which a
call begins with ``{``. A call's name is one of the functions', its arguments the function's ``parameters``:

| syntax | arguments |
|---|---|
| ``QwenXml`` | ``<parameter=K>\\nV\\n</parameter>\\n`` per parameter in the schema's order, the required ones present |
| ``Glm45`` | ``<arg_key>K</arg_key><arg_value>V</arg_value>`` the same (a newline between them allowed) |
| ``Gemma4`` | ``{k:V,...}``, keys sorted (the template's ``dictsort``), strings between ``<|"|>``, nested objects too |
| ``Llama3`` | the JSON ``{"name": NAME, "parameters": {...}}`` |

A Qwen or GLM-4.5 value is the raw text of a parameter whose schema allows only strings (its ``enum``, ``const``,
``pattern``, ``minLength``, ``maxLength`` held; up to the closing tag, which it cannot contain), else the schema's JSON
(Qwen's booleans also ``True`` / ``False``: the parser maps them); a Gemma value is its own syntax, numbers, booleans
and null its JSON. A function's schema compiles exactly when the tool is ``strict`` (a keyword not held: an error that
names it), else leniently; an object that declares properties takes no others unless ``additionalProperties`` says so
(extra Qwen / GLM parameters, extra Gemma keys at the end).

A custom tool (``tools.Custom``) is a function of one string parameter ``input``, written as any string parameter of the
syntax: its text any text, a regex's match (the parameter's ``pattern``), or a Lark grammar's text (the grammar a
sub-grammar, ``@name``: OpenAI's grammars are llguidance's Lark; Qwen's closing newline optional where its parser's
taking it off changes nothing). A grammar's lexemes are greedy and a sub-grammar has no stop text, so one with a lexeme
that goes on with the input's closing text (``lexemes.ending``: a catch-all such as Codex's ``/[\\s\\S]+/``) would keep
the call from ever ending: its input is free text there, as in Llama 3's JSON calls (a JSON string holds a regex, not a
Lark grammar), logged once. OpenAI's calls end with a special token, which no lexeme takes; these syntaxes' closing
texts are text tokens."""

import json
import logging
from typing import Any

import llguidance

from ..structured import Constraint, ConstraintError, Vocab, closed_objects
from .lexemes import ending
from .tools import Calls, Custom, GemmaParser, Glm45Parser, QwenParser, _GemmaValue, _types, resolve

log = logging.getLogger("artlite.server")

ANY = r"(.|\n)*"  # (a regex of any text: a raw string's)
LEAD = r"/[ \t\r\n]{1,8}/"  # (blanks before an answer's first call or value: a forced opener's gap)
SEP = r"/[ \t\n]{1,4}/"  # (between required calls)
LLAMA_SEP = r"/[ \t\n;]{1,6}/"  # (Llama 3's: white space or ";", ``LlamaParser``)
STRING_KEYS = {"enum", "const", "pattern", "minLength", "maxLength"}  # (a raw string's keywords held)
NOTES = {"description", "title", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly", "$schema",
         "$id", "type", "nullable", "contentMediaType", "contentEncoding"}  # (keywords that constrain nothing here)


def lit(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


class Grammar:
    """A Lark grammar under construction: ``rule`` names an expansion, ``token`` a text that may be a special token
    (``<[id]>``) or plain (a literal)."""

    def __init__(self, v: Vocab):
        self.v = v
        self.rules: dict[str, str] = {}
        self.subs: dict[str, str] = {}  # (sub-grammars by name: custom tools' Lark grammars)
        self._n = 0

    def rule(self, body: str, hint: str = "r", *, attrs: str = "") -> str:
        self._n += 1
        name = f"{hint}_{self._n}"
        self.rules[f"{name}{attrs}"] = body
        return name

    def token(self, text: str) -> str:
        i = self.v.special_id(text)
        return f"<[{i}]>" if i is not None else lit(text)

    def special(self, text: str) -> int | None:
        return self.v.special_id(text)

    def text(self, start: str) -> str:
        return "%llguidance {}\n" + f"start: {start}\n" + "".join(f"{k}: {b}\n" for k, b in self.rules.items())

    def sub(self, lark: str) -> str:
        """A Lark grammar as a sub-grammar: its reference."""
        name = f"input_{len(self.subs)}"
        self.subs[name] = lark
        return f"@{name}"

    def constraint(self, start: str) -> Constraint:
        return Constraint("lark", self.text(start), name="tools", subs=tuple(self.subs.items()))


def _ref(root: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """A local ``$ref``'s target in the function's parameters (as the parser types it: ``tools.resolve``)."""
    out = resolve(root, schema)
    return out if isinstance(out, dict) else schema


def _with_defs(root: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """A subschema that compiles on its own: the root's definitions with it (its local ``$ref``)."""
    defs = {k: root[k] for k in ("$defs", "definitions") if isinstance(root.get(k), dict) and k not in schema}
    return {**defs, **schema}


class Tool:
    """A function's call grammar under construction: ``g``, its name, its ``parameters`` (``root``), ``strict``; a
    custom tool's (``custom``): its input's grammar ``sub`` (a Lark grammar's reference), or its pattern in ``root``."""

    def __init__(self, g: Grammar, fn: dict[str, Any], strict: bool, custom: Custom | None = None, *,
                 end: str | None = None):
        self.g, self.name, self.strict, self.sub = g, fn["name"], strict or custom is not None, None
        self.newline = False  # (a sub-grammar's input: Qwen's closing newline optional, ``lexemes.Ending``)
        params = fn.get("parameters")
        self.root = params if isinstance(params, dict) else {}
        if custom is not None:
            value: dict[str, Any] = {"type": "string"}
            rx = _unanchored(custom.definition) if custom.syntax == "regex" else ""
            if custom.syntax != "text":  # (a regex as the call holds it: text before and after it)
                msg = _check_grammar(custom.definition if custom.syntax == "lark" else
                                     llguidance.LLMatcher.grammar_from_regex(f"a(?:{rx})a"))
                if msg is not None:
                    raise ConstraintError(f"custom tool {self.name!r}: its {custom.syntax} grammar: {msg}")
            if custom.syntax == "regex":
                value["pattern"] = f"^(?:{rx})$"
            elif custom.syntax == "lark":
                e = ending(custom.definition, end) if end is not None else None
                if e is not None and e.over is None:
                    self.sub, self.newline = g.sub(custom.definition), e.newline
                elif (self.name, custom.definition, end) not in _warned:  # (free text: the call can end)
                    _warned.add((self.name, custom.definition, end))
                    why = "a JSON string cannot hold a Lark grammar" if e is None else \
                        f"its grammar's {e.over} can go on with the call's closing text {end!r}"
                    log.warning("custom tool %r: %s: its input is free text in this template's calls", self.name, why)
            self.root = {"type": "object", "properties": {"input": value}, "required": ["input"],
                         "additionalProperties": False}
        self.json = Constraint("json_schema", "{}", strict=self.strict, name=self.name)

    def unheld(self, schema: dict[str, Any], held: set[str], where: str) -> None:
        """A strict tool's schema keyword this syntax does not hold: an error that names it."""
        if self.strict:
            extra = sorted(set(schema) - held - NOTES)
            if extra:
                raise ConstraintError(f"tool {self.name!r}: {where}: {', '.join(extra)} cannot be held in this "
                                      f"template's call syntax (strict)")

    def json_value(self, schema: dict[str, Any]) -> str:
        """A JSON value of a subschema (llguidance's JSON, the response format's rules: closed objects, bounded white
        space), its errors named by the tool."""
        s = self.json.schema(closed_objects(_with_defs(self.root, schema)))
        msg = _check(s)
        if msg is not None:
            raise ConstraintError(f"tool {self.name!r}: {msg}")
        return "%json " + json.dumps(s, ensure_ascii=False)

    def properties(self) -> tuple[list[tuple[str, dict[str, Any], bool]], bool]:
        """The parameters in the schema's order (name, schema, required) and whether others are allowed."""
        root = _ref(self.root, self.root)
        props = root.get("properties") if isinstance(root.get("properties"), dict) else {}
        required = set(root.get("required") or []) if isinstance(root.get("required"), list) else set()
        extra = root.get("additionalProperties", "properties" not in root) not in (False, None)
        out = [(k, s if isinstance(s, dict) else {}, k in required) for k, s in props.items()]
        return out, extra

    def raw_string(self, schema: dict[str, Any]) -> str | None:
        """The regex of a raw string value (the text between its tags), or None: the schema allows more
        than strings (its JSON)."""
        s = _ref(self.root, schema)
        types = _types(s)
        if not types or ("string" in types and types != {"string"}):  # (any text: the parser types it, a string as is)
            self.unheld(s, set(), "a parameter of strings and other types")
            return ANY
        if types != {"string"}:
            return None
        self.unheld(s, STRING_KEYS, "a string parameter")
        if "const" in s or "enum" in s:
            values = [s["const"]] if "const" in s else [x for x in s["enum"] if isinstance(x, str)]
            return "(" + "|".join(_escape(x) for x in values) + ")"
        if isinstance(s.get("pattern"), str):  # (JSON Schema's: unanchored, the text contains a match)
            p = s["pattern"]
            body = p[1:] if p.startswith("^") else f"{ANY}({p})"
            body = body[:-1] if body.endswith("$") and not body.endswith("\\$") else body + ANY
            return llguidance.regex_to_lark(body, "dw")
        lo, hi = s.get("minLength"), s.get("maxLength")
        if isinstance(lo, int) or isinstance(hi, int):
            return rf"(.|\n){{{lo if isinstance(lo, int) else 0},{hi if isinstance(hi, int) else ''}}}"
        return ANY


_warned: set[tuple[str, str, str]] = set()


def _escape(text: str) -> str:
    """A text as a regex of itself (Lark's ``/.../``)."""
    out = []
    for c in text:
        if c in "\\.^$|?*+()[]{}/":
            out.append("\\" + c)
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        else:
            out.append(c)
    return "".join(out)


def _check(schema: dict[str, Any]) -> str | None:
    """llguidance's error of a JSON schema alone (its own message), or None."""
    try:
        return _check_grammar(llguidance.LLMatcher.grammar_from_json_schema(schema))
    except ValueError as e:
        return str(e)


def _unanchored(rx: str) -> str:
    """A custom tool's regex without the ``^`` and ``$`` around it (the whole input matches it anyway)."""
    rx = rx.removeprefix("^")
    if rx.endswith("$") and (len(rx) - 1 - len(rx[:-1].rstrip("\\"))) % 2 == 0:  # (an even run of \\: not \\$)
        rx = rx[:-1]
    return rx


def _check_grammar(grammar: str) -> str | None:
    """llguidance's error of a grammar alone (its own message), or None."""
    error, messages = llguidance.LLMatcher.validate_grammar_with_warnings(grammar)
    return messages[0].removesuffix("<non-verbose/>").strip() if error else None


# ----------------------------------------------------------------------
# The syntaxes
# ----------------------------------------------------------------------


def _qwen_value(t: Tool, k: str, schema: dict[str, Any]) -> str:
    """A Qwen parameter after ``<parameter=K>\\n``: its value, ``\\n</parameter>\\n``."""
    g, end = t.g, "\n" + QwenParser.END
    if t.sub is not None:  # (the newline optional where a lexeme of the input may take it: ``lexemes``)
        return f'{t.sub} "\\n"{"?" if t.newline else ""} {lit(QwenParser.END + chr(10))}'
    raw = t.raw_string(schema)
    if raw is not None:
        v = g.rule(f"/{raw}{_escape(end)}/", "v", attrs="[lazy]")
        return f'{v} "\\n"'
    alt = t.json_value(schema)
    types = _types(_ref(t.root, schema))
    python = [w for k, ws in (("boolean", ('"True"', '"False"')), ("null", ('"None"',))) if k in types for w in ws]
    if python:  # (Qwen's template writes Python's literals; the parser maps them)
        alt = f"({alt} | {' | '.join(python)})"
    return f"{alt} {lit(end + chr(10))}"


def _qwen_call(t: Tool) -> str:
    g = t.g
    props, extra = t.properties()
    items = [(g.rule(f"{lit(QwenParser.PARAM_HEAD + k + QwenParser.NAME_END + chr(10))} {_qwen_value(t, k, s)}", "p"),
              req) for k, s, req in props]
    if extra:
        any_value = g.rule(f"/{ANY}{_escape(chr(10) + QwenParser.END)}/", "v", attrs="[lazy]")
        items.append((g.rule(f'({lit(QwenParser.PARAM_HEAD)} /[^>\\n]+/ ">\\n" {any_value} "\\n")*', "x"), False))
    return f'{lit(t.name + QwenParser.NAME_END + chr(10))} {_seq(g, items, "")} ' \
           f'{lit(QwenParser.FUNCTION_END + chr(10))} {g.token(QwenParser.CLOSE)}'


def _glm_call(t: Tool) -> str:
    g = t.g
    P = Glm45Parser
    nl = '"\\n"?'
    end_special = g.special(P.VALUE_END) is not None
    props, extra = t.properties()

    def value(schema: dict[str, Any]) -> str:
        if t.sub is not None:
            return f"{t.sub} {g.token(P.VALUE_END)}"
        raw = t.raw_string(schema)
        if raw is None:
            return f"{t.json_value(schema)} {g.token(P.VALUE_END)}"
        if end_special:  # (the value cannot contain the special token)
            body = g.rule(r"/(.|\n)+/" if raw == ANY else f"/{raw}/", "v")
            return f"{body}? {g.token(P.VALUE_END)}"
        return g.rule(f"/{raw}{_escape(P.VALUE_END)}/", "v", attrs="[lazy]")

    items = [(g.rule(f"{nl} {g.token(P.KEY)} {lit(k)} {g.token(P.KEY_END)} {nl} {g.token(P.VALUE)} {value(s)}", "p"),
              req) for k, s, req in props]
    if extra:
        items.append((g.rule(f"({nl} {g.token(P.KEY)} /[^<\\n]+/ {g.token(P.KEY_END)} {nl} {g.token(P.VALUE)} "
                             f"{value({})})*", "x"), False))
    return f"{lit(t.name)} {_seq(g, items, '')} {nl} {g.token(P.CLOSE)}"


def _llama_call(t: Tool) -> str:
    schema = {"type": "object", "properties": {"name": {"const": t.name}, "parameters": t.root or {"type": "object"}},
              "required": ["name", "parameters"], "additionalProperties": False}
    return t.json_value(schema)


class _Gemma:
    """Gemma 4's value syntax of a schema (``tools._GemmaValue``: the template's ``format_argument``): rules per
    subschema, local references as rules of their own (recursion)."""

    def __init__(self, t: Tool):
        self.t, self.g = t, t.g
        self.q = self.g.token(_GemmaValue.Q)
        self.refs: dict[str, str] = {}
        self._any: str | None = None

    def any(self) -> str:
        """Any value (a schema that says nothing)."""
        if self._any is None:
            g = self.g
            name = self._any = g.rule("", "any")
            key = "/[^:,{}\\[\\]<\\s]+/"
            obj = f'"{{" ({key} ":" {name} ("," {key} ":" {name})*)? "}}"'
            g.rules[name] = (f"{self.string()} | {_scalar('number')} | \"true\" | \"false\" | \"null\" | "
                             f'{obj} | "[" ({name} ("," {name})*)? "]"')
        return self._any

    def string(self, raw: str | None = None) -> str:
        """A string between ``<|"|>`` (``raw``: its pattern; the text up to the closing one, which it cannot
        contain)."""
        q = _GemmaValue.Q
        if self.g.special(q) is not None:
            body = r"/(.|\n)+/" if raw in (None, ANY) else self.g.rule(f"/{raw}/", "s")
            return f"{self.q} {body}? {self.q}"
        return f"{lit(q)} {self.g.rule(f'/{raw or ANY}{_escape(q)}/', 's', attrs='[lazy]')}"

    def value(self, schema: object) -> str:
        t, g = self.t, self.g
        if not isinstance(schema, dict) or not schema:
            return self.any()
        if isinstance(schema.get("$ref"), str):
            ref = schema["$ref"]
            if ref not in self.refs:
                self.refs[ref] = g.rule("", "ref")
                g.rules[self.refs[ref]] = self.value(_ref(t.root, {"$ref": ref}) if _ref(t.root, schema) is not schema
                                                     else {})
            return self.refs[ref]
        for key in ("anyOf", "oneOf"):
            if isinstance(schema.get(key), list):
                t.unheld(schema, {key}, "a union")
                return "(" + " | ".join(self.value(s) for s in schema[key]) + ")"
        if "const" in schema or "enum" in schema:
            values = [schema["const"]] if "const" in schema else list(schema["enum"])
            return "(" + " | ".join(self.literal(v) for v in values) + ")"
        kind = schema.get("type")
        if isinstance(kind, list):
            return "(" + " | ".join(self.value({**schema, "type": k}) for k in kind) + ")"
        if kind == "string":
            return f"{self.q} {t.sub} {self.q}" if t.sub is not None else self.string(t.raw_string(schema))
        if kind in ("integer", "number", "boolean", "null"):
            return t.json_value(schema)
        if kind == "array" or "items" in schema:
            t.unheld(schema, {"items", "minItems", "maxItems"}, "an array")
            item = self.value(schema.get("items"))
            lo, hi = schema.get("minItems", 0), schema.get("maxItems")
            if not lo and hi is None:
                return f'"[" ({item} ("," {item})*)? "]"'
            rest = f'("," {item}){{{max(lo - 1, 0)},{"" if hi is None else max(hi - 1, 0)}}}'
            return f'"[" {item} {rest} "]"' if lo else f'"[" ({item} {rest})? "]"'
        if kind == "object" or "properties" in schema:
            t.unheld(schema, {"properties", "required", "additionalProperties"}, "an object")
            return self.obj(schema)
        return self.any()

    def obj(self, schema: dict[str, Any]) -> str:
        g = self.g
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = set(schema.get("required") or [])
        items = [(g.rule(f"{lit(k + ':')} {self.value(s)}", "k"), k in required) for k, s in sorted(props.items())]
        if schema.get("additionalProperties", "properties" not in schema) not in (False, None):
            extra = self.value(schema["additionalProperties"]) if isinstance(schema.get("additionalProperties"),
                                                                             dict) else self.any()
            key = "/[^:,{}\\[\\]<\\s]+/"
            items.append((g.rule(f'{key} ":" {extra} ("," {key} ":" {extra})*', "x"), False))  # (extra keys, last)
        return f'"{{" {_seq(g, items, chr(34) + "," + chr(34))} "}}"'

    def literal(self, value: object) -> str:
        """A value as the template writes it (``format_argument``)."""
        if isinstance(value, str):
            return f"{self.q} {lit(value)} {self.q}" if value else f"{self.q} {self.q}"  # (q: a token or its text)
        if isinstance(value, bool):
            return '"true"' if value else '"false"'
        if value is None:
            return '"null"'
        if isinstance(value, dict):
            inner = ' "," '.join(f"{lit(str(k) + ':')} {self.literal(v)}" for k, v in sorted(value.items()))
            return f'"{{" {inner} "}}"'
        if isinstance(value, list):
            return '"[" ' + ' "," '.join(self.literal(v) for v in value) + ' "]"'
        return lit(str(value))


def _scalar(kind: str) -> str:
    return "%json " + json.dumps({"type": kind})


def _gemma_call(t: Tool) -> str:
    gv = _Gemma(t)
    root = _ref(t.root, t.root)
    obj = gv.obj(root if root else {"type": "object", "properties": {}})
    return f"{lit(t.name)} {obj} {t.g.token(GemmaParser.CLOSE)}"


def _seq(g: Grammar, items: list[tuple[str, bool]], sep: str) -> str:
    """Items in their order, each optional unless required, ``sep`` between those present (a rule; empty: "")."""
    if not items:
        return ""
    after: list[str] = [""] * (len(items) + 1)  # (the rest from item i, one item already present)
    first: list[str] = [""] * (len(items) + 1)  # (the rest from item i, none present yet)
    for i in range(len(items) - 1, -1, -1):
        item, req = items[i]
        s = f"{sep} " if sep else ""
        a = f"{s}{item} {after[i + 1]}".strip()
        after[i] = g.rule(a, "a") if req else g.rule(f"({a})?", "a") if not after[i + 1] else \
            g.rule(f"{a} | {after[i + 1]}", "a")
        if req:
            first[i] = g.rule(f"{item} {after[i + 1]}".strip(), "f")
        elif first[i + 1]:
            first[i] = g.rule(f"{item} {after[i + 1]} | {first[i + 1]}".strip(), "f")
        else:
            first[i] = g.rule(f"({item} {after[i + 1]})?".replace(" )", ")"), "f")
    return first[0]


# ----------------------------------------------------------------------
# The answer
# ----------------------------------------------------------------------


def answer(calls: Calls, v: Vocab, response: Constraint | None = None) -> Constraint:
    """The grammar of an answer with ``calls`` (``Calls.choice``: auto, required or a named function;
    ``Calls.parallel``) over ``v``'s tokens, and with ``response`` (a JSON response format, under auto: a call or that
    value). ConstraintError: a strict tool's schema that does not compile exactly, a response format that is not JSON."""
    if response is not None and not response.json:
        raise ConstraintError("a grammar, a regex or choices cannot be combined with tools (a JSON schema can)")
    g = Grammar(v)
    syntax = calls.syntax
    fns = [f["function"] for f in calls.functions if calls.name is None or f["function"]["name"] == calls.name]
    if not fns:
        raise ConstraintError("no function to call")
    llama = syntax.parser_class.__name__ == "LlamaParser"
    end = {"QwenParser": "\n" + QwenParser.END, "Glm45Parser": Glm45Parser.VALUE_END,
           "GemmaParser": _GemmaValue.Q}.get(syntax.parser_class.__name__)  # (a string value's closing text)
    bodies = []
    for fn in fns:
        t = Tool(g, fn, fn["name"] in calls.strict, calls.custom.get(fn["name"]), end=end)
        body = {"QwenParser": _qwen_call, "Glm45Parser": _glm_call, "GemmaParser": _gemma_call,
                "LlamaParser": _llama_call}[syntax.parser_class.__name__](t)
        bodies.append(g.rule(body, "fn"))
    names = " | ".join(bodies)
    P = syntax.parser_class
    if llama:
        call = g.rule(names, "call")
        opener = None
    else:
        head = {"QwenParser": QwenParser.NAME_HEAD, "Glm45Parser": "", "GemmaParser": GemmaParser.NAME_HEAD}[P.__name__]
        opener = g.special(P.OPEN)
        after = g.rule(f"{lit(head) if head else ''} ({names})".strip(), "after")  # (the call after its opener)
        call = g.rule(f"{g.token(P.OPEN)} {after}", "call")
    sep = LLAMA_SEP if llama else SEP
    calls_rule = g.rule(f"{call} ({sep}? {call})*" if calls.parallel else call, "calls")
    if response is not None and calls.choice == "auto":
        return g.constraint(f"{LEAD}? ({calls_rule} | {response_value(response)})")
    if calls.choice != "auto":
        return g.constraint(f"{LEAD}? {calls_rule}")
    if llama:
        text = g.rule(r"/[^ \t\r\n{](.|\n)*/", "text")
        return g.constraint(f"{LEAD}? ({calls_rule} | {text})?")
    if opener is not None:  # (free text: any token but the opener and the end of the turn)
        free = g.rule("<[^" + ",".join(str(i) for i in sorted({opener, *v.eos})) + "]>*", "free")
        rest = f"({call} {free})*" if calls.parallel else f"{call}?"
        return g.constraint(f"{free} {rest}")
    # (an opener that is text: free text up to it, lazily; the text after the last call cannot hold one)
    trig = g.rule(f"/{ANY}{_escape(P.OPEN)}/", "trig", attrs="[lazy]")
    seg = g.rule(f"{trig} {after}", "seg")
    g.rules["TAIL"] = f"/(.|\\n)+/ & ~/{ANY}{_escape(P.OPEN)}{ANY}/"
    rest = f"{seg}* TAIL?" if calls.parallel else f"({seg} | TAIL)?"
    return g.constraint(rest)


def response_value(response: Constraint) -> str:
    """A JSON response format's value, as a grammar's alternative."""
    schema = json.loads(response.source) if response.kind == "json_schema" or response.source else {"type": "object"}
    return "%json " + json.dumps(response.schema(schema), ensure_ascii=False)
