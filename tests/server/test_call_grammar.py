"""The call grammars (``models.server.call_grammar``) against the parsers that read the calls (``tools``), no NPU:
llama.cpp's test vectors (``test_tools``) as a toy tokenizer gives them, every valid call in the template's form taken
and every other refused; each template's form of a call through the model files' tokenizers (where they are here),
token by token under the mask; random walks under the grammar parsed back into calls their schemas take; the
``tool_choice`` modes, a response format beside the tools, ``tool_choice`` none's ban; strict tools' errors; custom
tools' inputs (a Lark grammar, a regex, text)."""

import functools
import json
import os
import random
from collections.abc import Callable

import numpy as np
import pytest

from artlite.gguf_model import GGUFFile
from artlite.model_config import gemma_gguf, lfm3_gguf, llama_gguf, minicpmv_gguf, spark_gguf
from models import structured as S
from models.server import call_grammar as CG
from models.server.tools import GEMMA4, GLM45, LLAMA3, QWEN_XML, Calls, Syntax, customs, functions

from .test_tools import GEMMA, GLM, LLAMA, QWEN, TOOLS, glm_call, parse, qwen_call

TAGS = ["<tool_call>", "</tool_call>", "<|tool_call>", "<tool_call|>", '<|"|>']  # (user-defined: text, one token)
SPECIALS = ["<|end|>"]  # (control)


class Toy:
    """Byte tokens, a few words, the syntaxes' tags (user-defined tokens, as the models' vocabularies have them: text,
    each one token; GLM-4.5's other tags: bytes), a control token; text encodes greedily, the longest piece first, the
    control token parsed (``encode``) or not (``encode_plain``)."""

    def __init__(self) -> None:
        words = [b"<function=", b"</parameter>", b"call:", b"get_time", b"city", b"Paris", b"\n<", b"name"]
        self.pieces = [bytes([i]) for i in range(256)] + words + [t.encode() for t in TAGS] + \
            [s.encode() for s in SPECIALS]
        self.special = list(range(len(self.pieces) - len(SPECIALS), len(self.pieces)))
        self.index = {p: i for i, p in enumerate(self.pieces) if i not in self.special}

    def vocabulary(self) -> tuple[list[bytes], list[int]]:
        return list(self.pieces), list(self.special)

    def encode_plain(self, text: str) -> list[int]:
        b, out, i = text.encode(), [], 0
        while i < len(b):
            n = next(n for n in range(min(16, len(b) - i), 0, -1) if b[i:i + n] in self.index)  # (tags: whole)
            out.append(self.index[b[i:i + n]])
            i += n
        return out

    def encode(self, text: str) -> list[int]:
        out, rest = [], text
        while rest:
            hits = [(rest.find(s), s) for s in SPECIALS if s in rest]
            if not hits:
                return out + self.encode_plain(rest)
            i, s = min(hits)
            out += self.encode_plain(rest[:i]) + [self.pieces.index(s.encode())]
            rest = rest[i + len(s):]
        return out


TOY = Toy()
V = S.Vocab(TOY, 0, [TOY.pieces.index(b"<|end|>")])


def grammar(syntax: Syntax, fns: list[dict], v: S.Vocab = V, **kw: object) -> str:
    return S.compiled(CG.answer(Calls(syntax, fns, **kw), v), v).grammar


def takes(g: str, ids: list[int], v: S.Vocab = V) -> bool:
    """Whether a grammar takes the tokens whole and may end there (the matcher: any split of the bytes)."""
    m = S.GrammarMask(v, g)
    for t in ids:
        if m.phase == "off" or not m.matcher.consume_token(t):
            return False
    return m.matcher.is_accepting()


def valid(name: str, args: object) -> bool:
    """Whether a call is of one of ``TOOLS`` and its arguments are its schema's (the types these use)."""
    fn = next((f["function"] for f in TOOLS if f["function"]["name"] == name), None)
    if fn is None or not isinstance(args, dict):
        return False
    props = fn["parameters"].get("properties") or {}
    return set(args) <= set(props) and all(_is(v, props[k]) for k, v in args.items())


def _is(v: object, s: dict) -> bool:
    if "anyOf" in s:
        return any(_is(v, x) for x in s["anyOf"])
    kinds = s.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    checks = {"integer": lambda x: isinstance(x, int) and not isinstance(x, bool), "string": lambda x: isinstance(x, str),
              "boolean": lambda x: isinstance(x, bool), "null": lambda x: x is None, "array": lambda x: isinstance(x, list),
              "number": lambda x: isinstance(x, int | float) and not isinstance(x, bool),
              "object": lambda x: isinstance(x, dict)}
    return not kinds or any(checks[k](v) for k in kinds)


# the parsers' leniency the grammars do not have: the template's form only (Qwen's newlines, Llama's name first), JSON
# values (but Qwen's True / False), not Python's tuples
LENIENT = {"<tool_call><function=get_time><parameter=city>Rome</parameter></function>",
           '{"type": "function", "name": "get_time", "parameters": {"city": "Paris"}}', qwen_call("mixed", flag="True",
                                                                                                items="(1, 2)")}


@pytest.mark.parametrize("syntax,vectors", [(QWEN_XML, QWEN), (GEMMA4, GEMMA), (LLAMA3, LLAMA), (GLM45, GLM)],
                         ids=["qwen", "gemma", "llama", "glm"])
def test_vectors(syntax: Syntax, vectors: list) -> None:
    """llama.cpp's vectors: the grammar (tool_choice auto) takes a text exactly when its calls are complete, of the
    functions, their arguments the schemas' (the parser reads more: invented names, mistyped values, cut calls)."""
    g = grammar(syntax, TOOLS)
    for text, _content, calls in vectors:
        good = bool(calls) and all(valid(n, a) for n, a in calls) and text not in LENIENT and \
            parse(syntax, text, len(text))[1] == calls and (syntax is not QWEN_XML or "</tool_call>" in text) and \
            (syntax is not GLM45 or text.endswith("</tool_call>"))
        if syntax is LLAMA3 and calls:  # (Llama's arguments as written: the parser types quoted values)
            try:
                written = json.loads(text)
                good = good and valid(written.get("name"), written.get("parameters"))
            except ValueError:
                good = False
        if not calls:  # (free text: any but a Llama call's, or an opener (a special token) that begins no call)
            good = (syntax is not LLAMA3 or not text.lstrip().startswith("{")) and \
                not any(o in text for o in ("<tool_call>", "<|tool_call>"))
        assert takes(g, TOY.encode(text)) == good or (not calls and syntax is GLM45), (text, calls)


TEMPLATES = {  # (each syntax's call as its template writes it; a function of every type)
    "qwen": [QWEN_XML, lambda: minicpmv_gguf(), qwen_call("special_function_with_opt", arg1="1", arg2="2"),
             qwen_call("edit", filename="a.c", oldString="x\n</para", newString="y")],
    "lfm3": [QWEN_XML, lambda: lfm3_gguf("QAD-Q4_0"), qwen_call("get_time", city="Paris")],
    "gemma": [GEMMA4, gemma_gguf, '<|tool_call>call:get_time{city:<|"|>Paris<|"|>}<tool_call|>',
              "<|tool_call>call:special_function_with_opt{arg1:1,arg2:2}<tool_call|>"],
    "spark": [GLM45, lambda: spark_gguf("1.7B"), glm_call("get_time", city="Paris"),
              glm_call("special_function_with_opt", "\n", arg1="1", arg2="2")],
    "llama": [LLAMA3, lambda: llama_gguf("Q4_K_M", "8B"), '{"name": "get_time", "parameters": {"city": "Paris"}}'],
}


def tokenizer_of(syntax: Syntax, path: str) -> object:
    if syntax is GEMMA4:
        from models.gemma4.tokenizer import GemmaTokenizer
        return GemmaTokenizer(GGUFFile(path))
    if syntax is LLAMA3:
        from models.llama.tokenizer import load_tokenizer
        return load_tokenizer(path)
    from models.tokenizer import BpeTokenizer
    return BpeTokenizer(GGUFFile(path))


@functools.cache  # (a model file's tokenizer and vocabulary: seconds to build, read-only)
def model_vocab(family: str) -> tuple[Syntax, object, S.Vocab]:
    syntax, path = TEMPLATES[family][0], TEMPLATES[family][1]()
    if not os.path.exists(path):
        pytest.skip(f"{path}: not here")
    tok = tokenizer_of(syntax, path)
    g = GGUFFile(path)
    eos = [int(g.meta["tokenizer.ggml.eos_token_id"])] + \
        [int(i) for i in np.flatnonzero(np.asarray(g.meta["tokenizer.ggml.token_type"]) == 3)
         if g.strings("tokenizer.ggml.tokens")[int(i)] in ("<|eot_id|>", "<|eom_id|>", "<turn|>", "<|im_end|>")]
    return syntax, tok, S.vocab(tok, 0, sorted(set(eos)))


@pytest.mark.parametrize("family", TEMPLATES)
def test_templates(family: str) -> None:
    """Each template's form of a call through the model file's tokenizer, token by token under the mask (its canonical
    tokens: the model's own), after free text: taken whole, the end of the turn allowed after it; the parser reads it."""
    syntax, tok, v = model_vocab(family)
    g = grammar(syntax, TOOLS, v)
    for text in TEMPLATES[family][2:]:
        lead = "" if syntax is LLAMA3 else "Let me look. "
        ids = tok.encode(lead + text)
        m = S.GrammarMask(v, g)
        for i, t in enumerate(ids):
            b = m.bits()
            assert b is None or (b[t >> 5] >> (t & 31)) & 1, (family, i, tok.decode_bytes(ids[:i + 1], special=True))
            m.accept(t)
        b = m.bits()
        assert m.active and any((b[e >> 5] >> (e & 31)) & 1 for e in v.eos), family
        _, calls = parse(syntax, lead + text, len(text))
        assert calls and all(valid(n, a) for n, a in calls)


TYPED = functions([{"type": "function", "function": {"name": "set", "parameters": {
    "type": "object", "properties": {
        "mode": {"type": "string", "enum": ["fast", "slow"]}, "n": {"type": "integer", "minimum": 0, "maximum": 99},
        "on": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]},
                                            "maxItems": 3},
        "box": {"type": "object", "properties": {"w": {"type": "integer", "maximum": 9}, "deep": {"type": "boolean"}},
                "required": ["w"]},
        "code": {"type": "string", "maxLength": 4}},
    "required": ["mode", "box"]}}},
    {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}])


def _valid_typed(name: str, args: dict) -> bool:
    if name == "ping":
        return args == {}
    box = args.get("box")
    return (name == "set" and args.get("mode") in ("fast", "slow") and isinstance(box, dict) and
            set(args) <= {"mode", "n", "on", "tags", "box", "code"} and 0 <= args.get("n", 0) <= 99 and
            isinstance(args.get("on", False), bool) and set(args.get("tags", [])) <= {"a", "b"} and
            len(args.get("tags", [])) <= 3 and isinstance(box.get("w"), int) and box["w"] <= 9 and
            set(box) <= {"w", "deep"} and len(str(args.get("code", ""))) <= 4)


@pytest.mark.parametrize("family", TEMPLATES)
def test_random_walks(family: str) -> None:
    """Random walks under a call grammar (required: from the forced opener; tokens drawn among the allowed ones,
    short ones first) until only the end of the turn is allowed: the parser reads a call that its schema takes,
    whatever path the grammar allowed."""
    syntax, tok, v = model_vocab(family)
    g = grammar(syntax, TYPED, v, choice="required", parallel=False)
    rng = random.Random(0)
    lens = np.array([len(p) for p in v.pieces])
    for walk in range(24):
        m = S.GrammarMask(v, g)
        ids: list[int] = []
        for _ in range(400):
            b = m.bits()
            a = np.flatnonzero(np.unpackbits(b.view(np.uint8), bitorder="little")[:v.n])
            a = a[~np.isin(a, v.eos)]
            if not len(a):
                break
            short = a[lens[a] <= rng.choice([1, 2, 4, 64])]
            t = int(rng.choice(short if len(short) else a))
            ids.append(t)
            m.accept(t)
            assert m.active, (family, walk)
        text = tok.decode_bytes(ids, special=True).decode("utf-8", errors="replace")
        p = syntax.parser(TYPED, parallel=False)
        parts = p.parts([("content", text)]) + p.flush()
        got = [c for k, c in parts if k == "tool_call"]
        name = got[0].name if got else None
        args = json.loads("".join(c.arguments for c in got)) if got else None
        assert name is not None and _valid_typed(name, args), (family, walk, text)


@pytest.mark.parametrize("syntax", [QWEN_XML, GEMMA4, GLM45, LLAMA3], ids=["qwen", "gemma", "glm", "llama"])
def test_modes(syntax: Syntax) -> None:
    """auto: free text, calls; parallel false: one call, then the end; required: a call (leading blanks: a forced
    opener's gap), no text; a named function: its call alone; a JSON response format beside the tools: a call or the
    value, no text."""
    call = {QWEN_XML: qwen_call("get_time", city="Paris"),
            GEMMA4: '<|tool_call>call:get_time{city:<|"|>Paris<|"|>}<tool_call|>',
            GLM45: glm_call("get_time", city="Paris"),
            LLAMA3: '{"name": "get_time", "parameters": {"city": "Paris"}}'}[syntax]
    other = call.replace("get_time", "get_weather")
    sep = "; " if syntax is LLAMA3 else "\n"
    text = "" if syntax is LLAMA3 else "Sure. "
    auto = grammar(syntax, TOOLS)
    assert takes(auto, TOY.encode(text + call + sep + other))
    one = grammar(syntax, TOOLS, parallel=False)
    assert takes(one, TOY.encode(text + call)) and not takes(one, TOY.encode(text + call + sep + other))
    required = grammar(syntax, TOOLS, choice="required")
    assert takes(required, TOY.encode("\n\n" + call)) and not takes(required, TOY.encode("Sure. " + call))
    named = grammar(syntax, TOOLS, choice="required", name="get_weather", parallel=False)
    assert takes(named, TOY.encode(other)) and not takes(named, TOY.encode(call))
    rf = S.Constraint.json_schema({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    both = S.compiled(CG.answer(Calls(syntax, TOOLS), V, rf), V).grammar
    assert takes(both, TOY.encode(call)) and takes(both, TOY.encode('{"ok": true}'))
    assert not takes(both, TOY.encode("Sure. " + call)) and not takes(both, TOY.encode("Plain text."))
    with pytest.raises(S.ConstraintError, match="combined with tools"):
        CG.answer(Calls(syntax, TOOLS), V, S.Constraint("regex", "a+"))


def test_ban() -> None:
    """tool_choice none: the call's opener (one token) is never sampled; everything else is."""
    opener = TOY.encode("<tool_call>")[0]
    m = S.Ban(V, [opener])
    a = np.unpackbits(m.bits().view(np.uint8), bitorder="little")[:V.n]
    assert not a[opener] and a.sum() == V.n - 1


def test_strict() -> None:
    """A strict tool's schema compiles exactly: a keyword its call syntax cannot hold is an error that names the tool;
    a JSON part's unimplemented keyword too; the same tool not strict compiles (the keyword not held)."""
    fns = functions([{"type": "function", "function": {"name": "mail", "parameters": {
        "type": "object", "properties": {"to": {"type": "string", "format": "email"},
                                         "ids": {"type": "array", "uniqueItems": True}}}}}])
    for syntax in (QWEN_XML, GEMMA4, GLM45):
        with pytest.raises(S.ConstraintError, match="'mail'"):
            CG.answer(Calls(syntax, fns, strict=frozenset({"mail"})), V)
        assert grammar(syntax, fns)
    with pytest.raises(S.ConstraintError, match="uniqueItems"):
        CG.answer(Calls(LLAMA3, fns, strict=frozenset({"mail"})), V)


AGENT_TOOLS: list[Callable[[], list[dict]]] = []  # (the agents' tool sets: the compile cost, research scripts)


def test_agreement() -> None:
    """Where grammar and parser must agree: a parameter of strings and other types (or of any type) takes raw text, a
    string stays a string; a local reference types its value as the grammar holds it; Qwen's None for a null; a Gemma
    string that holds the call's end marker."""
    fns = functions([{"type": "function", "function": {"name": "m", "parameters": {
        "type": "object", "properties": {"u": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "a": {},
                                         "n": {"type": ["integer", "null"]}, "id": {"$ref": "#/$defs/S"}},
        "$defs": {"S": {"type": "string"}}}}}])
    g = grammar(QWEN_XML, fns)
    for args, want in (({"u": "x"}, {"u": "x"}), ({"u": "null"}, {"u": None}), ({"a": "hi"}, {"a": "hi"}),
                       ({"n": "None"}, {"n": None}), ({"id": "123"}, {"id": "123"})):
        text = qwen_call("m", **args)
        assert takes(g, TOY.encode(text)), text
        p = QWEN_XML.parser(fns)
        parts = p.parts([("content", text)]) + p.flush()
        assert json.loads("".join(c.arguments for k, c in parts if k == "tool_call")) == want, text
    # a Qwen value ends at its first closing tag, as the parser reads it (the newline before it optional)
    tight = "<tool_call>\n<function=m>\n<parameter=u>\nx</parameter>\n</function>\n</tool_call>"
    assert takes(g, TOY.encode(tight)) and not takes(g, TOY.encode(tight.replace("x<", "x</parameter>\n<")))
    p = QWEN_XML.parser(fns)
    parts = p.parts([("content", tight)]) + p.flush()
    assert json.loads("".join(c.arguments for k, c in parts if k == "tool_call")) == {"u": "x"}
    gemma = '<|tool_call>call:m{id:<|"|>a<tool_call|>b<|"|>}<tool_call|>'
    assert takes(grammar(GEMMA4, fns), TOY.encode(gemma))
    p = GEMMA4.parser(fns)
    parts = p.parts([("content", gemma)]) + p.flush()
    assert [(c.name, json.loads(c.arguments)) for k, c in parts if k == "tool_call"] == [("m", {"id": "a<tool_call|>b"})]


CALC = 'start: expr\nexpr: NUM (OP NUM)*\nNUM: /[0-9]+/\nOP: "+" | "-"\n'
CUSTOM = [{"type": "custom", "custom": {"name": "calc", "description": "Arithmetic", "format": {
              "type": "grammar", "grammar": {"syntax": "lark", "definition": CALC}}}},
          {"type": "custom", "custom": {"name": "code", "format": {
              "type": "grammar", "grammar": {"syntax": "regex", "definition": "[a-z]{2,4}-[0-9]+"}}}},
          {"type": "custom", "custom": {"name": "note"}},
          {"type": "custom", "custom": {"name": "day", "format": {
              "type": "grammar", "grammar": {"syntax": "regex", "definition": "^[0-9]{4}-[0-9]{2}$"}}}},
          {"type": "custom", "custom": {"name": "run", "format": {"type": "grammar", "grammar": {
              "syntax": "lark", "definition": "start: SOURCE\nSOURCE: /[\\s\\S]+/\n"}}}}]


def custom_call(syntax: Syntax, name: str, text: str) -> str:
    """A custom tool's call (a function of one string parameter ``input``) as the template writes it."""
    if syntax is QWEN_XML:
        return qwen_call(name, input=text)
    if syntax is GLM45:
        return glm_call(name, input=text)
    if syntax is GEMMA4:
        return f'<|tool_call>call:{name}{{input:<|"|>{text}<|"|>}}<tool_call|>'
    return json.dumps({"name": name, "parameters": {"input": text}})


@pytest.mark.parametrize("syntax", [QWEN_XML, GEMMA4, GLM45, LLAMA3], ids=["qwen", "gemma", "glm", "llama"])
def test_custom(syntax: Syntax) -> None:
    """Custom tools: a Lark grammar's input, a regex's (anchored or not), any text, each taken exactly and read back by
    the parser as written; a grammar whose regex runs over the call's closing text (a catch-all), or any Lark grammar in Llama 3's
    JSON calls, leaves the input free; a grammar that does not compile: an error that names the tool."""
    fns = functions(CUSTOM)
    g = grammar(syntax, fns, custom=customs(CUSTOM))
    cases = [("calc", "12+3-4", True), ("calc", "12+", syntax is LLAMA3), ("calc", "1 + 2", syntax is LLAMA3),
             ("code", "ab-12", True), ("code", "a-1", False), ("note", "any\ntext, <b>", True),
             ("day", "2027-03", True), ("day", "2027-3", False),
             ("run", "print(1)\nprint(2)", True)]
    for name, text, good in cases:
        call = custom_call(syntax, name, text)
        assert takes(g, TOY.encode(call)) == good, (name, text)
        if good:
            _, calls = parse(syntax, call, len(call))
            assert calls == [(name, {"input": text})], (name, text, calls)
    bad = [{"type": "custom", "custom": {"name": "calc", "format": {"type": "grammar", "grammar": {
        "syntax": "lark", "definition": "start: nope\n"}}}}]
    with pytest.raises(S.ConstraintError, match="custom tool 'calc'"):
        CG.answer(Calls(syntax, functions(bad), custom=customs(bad)), V)
