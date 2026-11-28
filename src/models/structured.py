"""Structured output: a generation's answer held to a JSON schema, a grammar or a regex (``Constraint``), compiled by
llguidance against the model's tokens (``Vocab``) and followed token by token (``GrammarMask``: the sampler's
``sampling.Mask``, a vocabulary bitmask per token).

llguidance (MIT; llama.cpp's, vLLM's and SGLang's option) splits a grammar into a lexer and an Earley parser over
lexemes: a mask over a 248K vocabulary takes 5 - 50 us, a schema compiles in about a millisecond, and what it does not
implement is an error, not a keyword skipped. Lark grammars, GBNF (converted: ``llguidance.gbnf_to_lark``), regexes and
JSON schemas all compile to it; xgrammar needs torch, llama.cpp's GBNF engine has no lexer (slow masks on these
vocabularies) and skips schema keywords silently.

The rules (docs/server.md, "Structured output"):

- a JSON answer's keys come in the schema's order; an object that declares ``properties`` admits no other keys unless
  ``additionalProperties`` allows them, one without admits any unless it is false (llama.cpp's and Gufo's rule over
  the JSON Schema default, open: ``closed_objects``; ``Constraint.closed``: every object without
  ``additionalProperties``, vLLM's ``disable_additional_properties``);
- white space: up to ``LEAD`` blanks before the value, none after it, between its tokens llama.cpp's ``space`` (a
  blank, or one or two newlines and up to 20 blanks or tabs: ``WHITESPACE``), so a model held from what it wants to
  write (a Markdown fence) cannot pad forever; no fences (the content loads with ``json.loads``); a run of digits outside
  strings has ``DIGITS`` at most (llama.cpp's bound on an integral part, a fraction and an exponent: ``GrammarMask``);
- a strict schema compiles exactly: a keyword llguidance does not implement is an error that names it; a lenient one
  (tool schemas, llama-server's ``json_schema``) keeps it as guidance, logged once per schema;
- ``MAX_SCHEMA`` bytes of schema, ``MAX_PROPERTIES`` properties, ``MAX_ENUM`` enum values at most (Gufo's), and
  llguidance's parser limits;
- the answer is constrained, never the thought: the mask starts where the thought ends (``Thought``);
- when the grammar is complete only the end-of-sequence tokens remain.

A model's ``Vocab`` (``vocab``: built once per tokenizer, ~0.7 s and ~90 MB on a 248K vocabulary) gives llguidance every
token's bytes, the control tokens marked special (llama.cpp's llguidance adapter's rule: a grammar names them
``<[id]>``, text never matches them; the user-defined ones, the tags a model writes, are text), and an encoder that
never turns text into special tokens. Compiled grammars are kept by their constraint (``compiled``: an agent sends the same tools every
turn)."""

import argparse
import collections
import concurrent.futures
import copy
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import llguidance
import numpy as np
from llguidance.gbnf_to_lark import GbnfToLarkError, any_to_lark

from .sampling import pack_mask

if TYPE_CHECKING:
    from .model import Blocks

log = logging.getLogger("artlite")

KINDS = ("json_schema", "json_object", "lark", "gbnf", "regex", "choice", "structural_tag")
LEAD = 8  # blanks before a JSON value at most
WHITESPACE = r"[ ]|\n{1,2}[ \t]{0,20}"  # between a JSON value's tokens (llama.cpp's ``space``, its empty case aside)
DIGITS = 16  # a run of digits outside JSON strings at most
MAX_SCHEMA = 2 << 20  # bytes of a schema's JSON text
MAX_PROPERTIES = 5000
MAX_ENUM = 1000  # enum and const values
MAX_DEPTH = 256  # a schema's nesting (its JSON's objects and arrays)
COMPILED = 128  # compiled grammars kept
LIMITS = llguidance.LLParserLimits(verbose_errors=False)


class ConstraintError(ValueError):
    """A constraint that does not compile: llguidance's message, or the limit it exceeds."""


@dataclasses.dataclass(frozen=True)
class Constraint:
    """What a generation's answer must be: ``kind`` ``json_schema`` (``source``: the schema's JSON text, its keys in
    their order), ``json_object`` (any JSON object, or the object schema in ``source``: llama.cpp's
    ``{"type": "json_object", "schema": ...}``), ``lark`` / ``gbnf`` (a grammar), ``regex``, ``choice`` (one of the
    texts of a JSON list, exactly) or ``structural_tag`` (free text, and after a trigger a structure: a JSON list of
    {trigger, begin, schema, end}, llguidance's ``StructTag``). A schema compiles exactly when ``strict``, else
    ``lenient``. ``whitespace``: the
    pattern between a JSON value's tokens (vLLM's ``whitespace_pattern``; None: ``WHITESPACE``), ``compact``: none
    (``disable_any_whitespace``), ``closed``: every object without ``additionalProperties`` is closed
    (``disable_additional_properties``); ``name``: for the log (a schema's); ``subs``: a Lark grammar's sub-grammars
    (name, Lark), which it references as ``@name`` (custom tools' inputs: ``call_grammar``); ``turn``: the grammar holds
    a block model's whole turn, its thoughts too (``call_grammar``'s: ``Compiled.turn``)."""

    kind: str
    source: str
    strict: bool = False
    whitespace: str | None = None
    compact: bool = False
    closed: bool = False
    name: str = ""
    subs: tuple[tuple[str, str], ...] = ()
    turn: bool = False

    def __post_init__(self) -> None:
        assert self.kind in KINDS, self.kind

    @staticmethod
    def json_schema(schema: object, *, strict: bool = False, name: str = "", **options: Any) -> "Constraint":
        """A JSON schema's constraint (``options``: ``whitespace``, ``compact``, ``closed``). ConstraintError: not a
        schema (an object or a boolean) or over the limits."""
        if not isinstance(schema, dict | bool):
            raise ConstraintError(f"a JSON schema is an object or a boolean, not {type(schema).__name__}")
        text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) > MAX_SCHEMA:
            raise ConstraintError(f"the JSON schema is larger than {MAX_SCHEMA >> 20} MiB")
        properties, values, depth = _counts(schema)
        if depth > MAX_DEPTH:
            raise ConstraintError(f"the JSON schema is nested deeper than {MAX_DEPTH} levels")
        if properties > MAX_PROPERTIES:
            raise ConstraintError(f"the JSON schema has {properties} properties, more than {MAX_PROPERTIES}")
        if values > MAX_ENUM:
            raise ConstraintError(f"the JSON schema has {values} enum and const values, more than {MAX_ENUM}")
        return Constraint("json_schema", text, strict=strict, name=name, **options)

    @staticmethod
    def of_grammar(text: object) -> "Constraint":
        """llama-server's ``grammar``: GBNF, or Lark after ``%llguidance`` (llama.cpp built with LLAMA_LLGUIDANCE)."""
        if not isinstance(text, str):
            raise ConstraintError("a grammar is a text (GBNF)")
        return Constraint("lark" if text.lstrip().startswith("%llguidance") else "gbnf", text)

    @staticmethod
    def of_schema(schema: object, *, strict: bool = False, name: str = "", **options: Any) -> "Constraint":
        """A JSON schema's constraint; an empty one: any JSON object (llama.cpp's)."""
        if schema == {}:
            return Constraint("json_object", "", name=name, **options)
        return Constraint.json_schema(schema, strict=strict, name=name, **options)

    @property
    def json(self) -> bool:
        """Whether the answer is JSON (its digit runs bounded)."""
        return self.kind in ("json_schema", "json_object")

    def grammar(self, v: "Vocab | None" = None) -> str:
        """The llguidance grammar (not validated: ``compiled``; ``v``: the tokens, whose special ones a structural tag's
        trigger may be). ConstraintError: a GBNF grammar that does not convert, choices that are not a list of texts."""
        if self.kind == "structural_tag":
            tags = json.loads(self.source)
            special = v is not None and all(v.special_name(t["trigger"]) for t in tags
                                            if t["trigger"].startswith("<") and t["trigger"].endswith(">"))
            return llguidance.StructTag.to_grammar(
                [llguidance.StructTag(trigger=t["trigger"], begin=t["begin"], end=t["end"],
                                      grammar=self.schema(t["schema"] if t["schema"] != {} else {"type": "object"}))
                 for t in tags], assume_special=special)
        if self.json:
            schema = json.loads(self.source) if self.kind == "json_schema" or self.source else {"type": "object"}
            body = json.dumps(self.schema(schema), ensure_ascii=False)
            return f"%llguidance {{}}\nstart: LEAD? value\nLEAD: /[ \\t\\r\\n]{{1,{LEAD}}}/\nvalue: %json {body}\n"
        if self.kind == "lark" and self.subs:
            return json.dumps({"grammars": [{"lark_grammar": self.source},
                                            *({"name": n, "lark_grammar": g} for n, g in self.subs)]})
        if self.kind == "lark":
            return llguidance.LLMatcher.grammar_from_lark(self.source)
        if self.kind == "gbnf":
            try:
                return llguidance.LLMatcher.grammar_from_lark(any_to_lark(self.source))
            except GbnfToLarkError as e:
                raise ConstraintError(f"the GBNF grammar does not parse: {e}") from None
            except Exception as e:  # noqa: BLE001 (the converter raises Exception for a missing rule)
                raise ConstraintError(f"the GBNF grammar does not convert: {e}") from None
        if self.kind == "regex":
            return llguidance.LLMatcher.grammar_from_regex(self.source)
        choices = json.loads(self.source)
        if not isinstance(choices, list) or not choices or not all(isinstance(c, str) for c in choices):
            raise ConstraintError("choice: a non-empty list of texts")
        return "start: " + " | ".join(json.dumps(c, ensure_ascii=False) for c in choices) + "\n"

    def schema(self, schema: dict[str, Any] | bool) -> dict[str, Any] | bool:
        """A JSON schema as llguidance compiles it: objects closed (``closed_objects``) and the compile options in its
        ``x-guidance`` (the schema's own over these)."""
        if isinstance(schema, bool):
            return schema
        options: dict[str, Any] = {"whitespace_flexible": False} if self.compact else \
            {"whitespace_pattern": self.whitespace or WHITESPACE}
        if not self.strict:
            options["lenient"] = True
        out = closed_objects(schema, every=self.closed)
        return {**out, "x-guidance": {**options, **(schema.get("x-guidance") or {})}}


def _counts(schema: object) -> tuple[int, int, int]:
    """A schema's properties and enum / const values, in all its subschemas, and its nesting depth."""
    properties = values = depth = 0
    stack = [(schema, 1)]
    while stack:
        s, d = stack.pop()
        depth = max(depth, d)
        if isinstance(s, dict):
            if isinstance(s.get("properties"), dict):
                properties += len(s["properties"])
            if isinstance(s.get("enum"), list):
                values += len(s["enum"])
            values += "const" in s
            stack.extend((v, d + 1) for v in s.values())
        elif isinstance(s, list):
            stack.extend((v, d + 1) for v in s)
    return properties, values, depth


_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_ONE = ("additionalProperties", "items", "not", "contains", "propertyNames", "unevaluatedProperties",
               "unevaluatedItems", "additionalItems")
_EXTENDERS = ("allOf", "anyOf", "oneOf", "if", "then", "else", "dependentSchemas", "$ref")  # (more keys, same object)


def closed_objects(schema: dict[str, Any], *, every: bool = False) -> dict[str, Any]:
    """A copy of a schema whose objects that declare ``properties`` without ``additionalProperties`` admit no other keys
    (``every``: also those without ``properties`` that are objects). An object whose keys another subschema of the
    same instance adds to stays open: one next to ``allOf`` / ``anyOf`` / ``oneOf`` / ``if`` / ``$ref``, an ``allOf``'s
    members, the members of an ``anyOf`` / ``oneOf`` of an object with properties, and the definitions an ``allOf``'s
    member refers to (pydantic's inheritance)."""
    extended = {r for m in _all_of_members(schema) for r in _refs(m)}
    return _close(schema, every, False, extended, "#")


def _all_of_members(schema: object) -> list[object]:
    out, stack = [], [schema]
    while stack:
        s = stack.pop()
        if isinstance(s, dict):
            if isinstance(s.get("allOf"), list):
                out += s["allOf"]
            stack.extend(s.values())
        elif isinstance(s, list):
            stack.extend(s)
    return out


def _refs(member: object) -> list[str]:
    """The local references an ``allOf`` member makes for the instance it applies to (its ``$ref``, its combinators')."""
    if not isinstance(member, dict):
        return []
    out = [member["$ref"]] if isinstance(member.get("$ref"), str) else []
    for key in ("allOf", "anyOf", "oneOf"):
        for m in member.get(key) or []:
            out += _refs(m)
    return out


def _close(schema: dict[str, Any], every: bool, member: bool, extended: set[str], path: str) -> dict[str, Any]:
    out = dict(schema)
    is_object = "properties" in out or (every and (out.get("type") == "object" or (isinstance(out.get("type"), list)
                                                                                  and "object" in out["type"])))
    if is_object and "additionalProperties" not in out and not member and path not in extended and \
            not any(k in out for k in _EXTENDERS):
        out["additionalProperties"] = False

    def sub(v: object, key: str, *, of: bool = False) -> object:
        return _close(v, every, of, extended, f"{path}/{key}") if isinstance(v, dict) else v

    for key in _SCHEMA_MAPS:
        if isinstance(out.get(key), dict):
            out[key] = {k: sub(v, f"{key}/{k}") for k, v in out[key].items()}
    shared = "properties" in out  # (an anyOf / oneOf member adds keys to this object's)
    for key in ("anyOf", "oneOf"):
        if isinstance(out.get(key), list):
            out[key] = [sub(v, f"{key}/{i}", of=shared) for i, v in enumerate(out[key])]
    if isinstance(out.get("allOf"), list):
        out["allOf"] = [sub(v, f"allOf/{i}", of=True) for i, v in enumerate(out["allOf"])]
    for key in ("if", "then", "else"):
        if isinstance(out.get(key), dict):
            out[key] = sub(out[key], key, of=True)
    if isinstance(out.get("dependentSchemas"), dict):
        out["dependentSchemas"] = {k: sub(v, f"dependentSchemas/{k}", of=True) for k, v in out["dependentSchemas"].items()}
    if isinstance(out.get("prefixItems"), list):
        out["prefixItems"] = [sub(v, f"prefixItems/{i}") for i, v in enumerate(out["prefixItems"])]
    for key in _SCHEMA_ONE:
        v = out.get(key)
        if isinstance(v, dict):
            out[key] = sub(v, key)
        elif key == "items" and isinstance(v, list):
            out[key] = [sub(x, f"items/{i}") for i, x in enumerate(v)]
    return out


# ----------------------------------------------------------------------
# The model's tokens
# ----------------------------------------------------------------------


class GrammarTokenizer(Protocol):
    """What a model's tokenizer gives llguidance (the GGUF tokenizers: ``models.tokenizer.BpeTokenizer``, Llama's,
    Gemma's)."""

    def vocabulary(self) -> tuple[list[bytes], list[int]]:
        """Every token's bytes and the special tokens' ids."""
        ...

    def encode_plain(self, text: str) -> list[int]:
        """Token ids of a text that has no special tokens (their text encodes as text)."""
        ...


class _Wrapper:
    """llguidance's tokenizer protocol (``llguidance.TokenizerWrapper``'s attributes): the tokens' bytes, the special
    ones, an encoder of text without special tokens (llguidance calls it with text whose bytes a grammar forces; it must
    not raise)."""

    is_tokenizer_wrapper = True

    def __init__(self, pieces: list[bytes], special: list[int], eos: int, encode: Any):
        self.tokens, self.special_token_ids = pieces, special
        self.eos_token_id, self.bos_token_id = eos, None
        self._encode = encode

    def __call__(self, text: str | bytes) -> list[int]:
        s = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
        try:
            return self._encode(s)
        except Exception:  # (a panic in llguidance otherwise)
            log.exception("structured output: the tokenizer failed on %r", s[:80])
            return []


class Vocab:
    """A model's tokens for llguidance: ``tokenizer`` (``llguidance.LLTokenizer``) over every token's bytes
    (``pieces``, padded with empty ones to the logits' width ``n``), the special ones marked, the end-of-sequence tokens
    ``eos`` (allowed once a grammar is complete); per token its leading ASCII digits (``digit_bits(j)``: the tokens with
    at least j, for ``DIGITS``)."""

    def __init__(self, tokenizer: GrammarTokenizer, n: int, eos: Sequence[int]):
        t0 = time.perf_counter()
        pieces, special = tokenizer.vocabulary()
        n = max(n, len(pieces))
        self.n, self.words = n, (n + 31) // 32
        self.pieces = pieces + [b""] * (n - len(pieces))
        self.special = frozenset(special)
        self.eos = tuple(int(t) for t in eos)
        self.tokenizer = llguidance.LLTokenizer(_Wrapper(self.pieces, sorted(special), self.eos[0],
                                                         tokenizer.encode_plain), eos_token=list(self.eos))
        lead = np.fromiter((len(p) - len(p.lstrip(b"0123456789")) for p in self.pieces), np.int32, n)
        lead[sorted(special)] = 0
        self.max_digits = int(lead.max(initial=0))
        self._digits = {j: pack_mask(lead >= j, self.words) for j in range(1, self.max_digits + 1)}
        self.seconds = time.perf_counter() - t0
        self._names = {self.pieces[i]: i for i in special}

    def special_name(self, text: str) -> bool:
        """Whether a text is a special token's (a grammar names it ``<...>``)."""
        return text.encode() in self._names

    def special_id(self, text: str) -> int | None:
        """The special token whose text is ``text`` (a grammar names it ``<[id]>``), or None."""
        return self._names.get(text.encode())

    def digit_bits(self, j: int) -> np.ndarray:
        """The tokens that begin with at least j digits (a bitmask)."""
        return self._digits[j]


_aside = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="artlite-mask")  # (``GrammarMask.prefetch``)
_vocabs: dict[int, tuple[object, Vocab]] = {}
_vocab_lock = threading.Lock()


def vocab(tokenizer: GrammarTokenizer, n: int, eos: Sequence[int]) -> Vocab:
    """The ``Vocab`` of a model's tokenizer (``n``: its logits' width; ``eos``: its end-of-sequence tokens), built on
    the first call (others wait for it), logged."""
    with _vocab_lock:
        hit = _vocabs.get(id(tokenizer))
        if hit is not None and hit[0] is tokenizer:
            return hit[1]
        v = Vocab(tokenizer, n, eos)
        _vocabs[id(tokenizer)] = (tokenizer, v)
    log.info("structured output: a vocabulary of %d tokens in %.0f ms", v.n, v.seconds * 1e3)
    return v


@dataclasses.dataclass(frozen=True)
class Compiled:
    """A constraint's validated grammar (``compiled``): what a ``GrammarMask`` follows; ``turn``: it holds the whole
    turn, the thoughts too (a block model's: no ``Thought`` before it); ``opens``: the tokens that open its answer
    (a framed one's text block: the digit runs counted after them)."""

    constraint: Constraint
    grammar: str
    turn: bool = False
    opens: tuple[int, ...] = ()

    @property
    def json(self) -> bool:
        return self.constraint.json


_compiled: collections.OrderedDict[tuple[int, Constraint, object], Compiled] = collections.OrderedDict()
_compiled_lock = threading.Lock()
_warned: set[tuple[str, str]] = set()


def turn_tokens(blocks: "Blocks", v: Vocab) -> dict[str, int]:
    """A block model's markers as special tokens of ``v`` (header, thought, text, call, end). ConstraintError: one is
    not a special token."""
    out = {}
    for k in ("header", "thought", "text", "call", "end"):
        t = getattr(blocks, k)
        if t and (out.setdefault(k, v.special_id(t))) is None:
            raise ConstraintError(f"{t!r} is no special token")
    return out


def thought_blocks(blocks: "Blocks", v: Vocab) -> str:
    """A Lark expansion of a block model's thought blocks before the answer (any number; their text free: any token
    but the blocks' markers that cannot be in it, the end, a header, a call's, and the end of the turn)."""
    t = turn_tokens(blocks, v)
    stops = {t[k] for k in ("end", "header", "call") if k in t} | set(v.eos)
    free = "<[^" + ",".join(str(i) for i in sorted(stops)) + "]>*"
    return f"(<[{t['header']}]> <[{t['thought']}]> {free} <[{t['end']}]>)*"


def framed(grammar: str, blocks: "Blocks", v: Vocab) -> str:
    """A grammar (``Constraint.grammar``'s) as a block model's turn: its thought blocks, then the answer's text block
    holding the grammar (llama.cpp's response format for Inkling)."""
    if grammar.lstrip().startswith("{"):  # (llguidance's grammars: the first is the start)
        top = json.loads(grammar)
        first, *rest = top["grammars"]
    else:
        top, first, rest = {}, {"lark_grammar": grammar}, []
    t = turn_tokens(blocks, v)
    frame = (f"%llguidance {{}}\nstart: {thought_blocks(blocks, v)} <[{t['header']}]> <[{t['text']}]> @answer "
             f"<[{t['end']}]>\n")
    return json.dumps({**top, "grammars": [{"lark_grammar": frame}, {**first, "name": "answer"}, *rest]},
                      ensure_ascii=False)


def compiled(constraint: Constraint, v: Vocab, *, blocks: "Blocks | None" = None) -> Compiled:
    """A constraint's grammar for ``v``'s tokens, validated (kept: the last ``COMPILED``); a lenient schema's ignored
    keywords logged once; ``blocks``: a block model's (the answer its text block, after its thoughts: ``framed``; a
    constraint that holds the turn already, ``Constraint.turn``, as it is). ConstraintError: it does not compile
    (llguidance's message)."""
    frame = blocks is not None and not constraint.turn
    key = (id(v), constraint, blocks if frame else None)
    with _compiled_lock:
        if key in _compiled:
            _compiled.move_to_end(key)
            return _compiled[key]
    grammar = constraint.grammar(v)
    if frame:
        grammar = framed(grammar, blocks, v)
    messages: list[str] = []
    if constraint.kind == "json_schema":  # (the schema alone first: its messages without the grammar around it)
        schema = constraint.schema(json.loads(constraint.source))
        messages = _validate(llguidance.LLMatcher.grammar_from_json_schema(schema), v)
    _validate(grammar, v)
    for m in messages:
        if (constraint.name, m) not in _warned:
            _warned.add((constraint.name, m))
            log.info("structured output%s: %s (a lenient schema: the keyword is not enforced)",
                     f" {constraint.name!r}" if constraint.name else "", _message(m))
    out = Compiled(constraint, grammar, turn=constraint.turn or frame,
                   opens=tuple(turn_tokens(blocks, v)[k] for k in ("header", "text")) if frame else ())
    with _compiled_lock:
        _compiled[key] = out
        while len(_compiled) > COMPILED:
            _compiled.popitem(last=False)
    return out


def _validate(grammar: str, v: Vocab) -> list[str]:
    """A grammar's warnings. ConstraintError: it does not compile."""
    try:
        error, messages = llguidance.LLMatcher.validate_grammar_with_warnings(grammar, v.tokenizer, limits=LIMITS)
    except ValueError as e:  # (options on a schema llguidance cannot take them on)
        raise ConstraintError(str(e)) from None
    if error:
        raise ConstraintError(_message(messages[0] if messages else "the grammar does not compile"))
    return messages


def _message(m: str) -> str:
    return m.removesuffix("<non-verbose/>").strip()


def add_arguments(p: argparse._ActionsContainer) -> None:
    """The tools' options of a structured output for every answer (llama-server's and llama-cli's)."""
    p.add_argument("--grammar", help="a GBNF grammar the answers follow (Lark after %%llguidance; llama.cpp's "
                                     "grammars/README.md)")
    p.add_argument("--grammar-file", metavar="FILE", help="the same from a file")
    p.add_argument("-j", "--json-schema", metavar="SCHEMA",
                   help="a JSON schema the answers follow ({}: any JSON object)")
    p.add_argument("-jf", "--json-schema-file", metavar="FILE", help="the same from a file")


def from_args(args: argparse.Namespace) -> Constraint | None:
    """The constraint of the options (``add_arguments``), or None. ConstraintError: two of them, a file that does not
    open, a schema that is not JSON."""
    given = [k for k in ("grammar", "grammar_file", "json_schema", "json_schema_file") if getattr(args, k, None)]
    if len(given) > 1:
        raise ConstraintError(f"--{given[0].replace('_', '-')} and --{given[1].replace('_', '-')}: one at most")
    if not given:
        return None
    text = getattr(args, given[0])
    if given[0].endswith("_file"):
        try:
            with open(text, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise ConstraintError(f"--{given[0].replace('_', '-')}: {e}") from None
    if given[0].startswith("grammar"):
        return Constraint.of_grammar(text)
    try:
        schema = json.loads(text)
    except ValueError as e:
        raise ConstraintError(f"the JSON schema is not JSON: {e}") from None
    return Constraint.of_schema(schema, name="--json-schema")


# ----------------------------------------------------------------------
# The mask
# ----------------------------------------------------------------------


def _tail(tokenizer: Any, ids: Sequence[int]) -> str:
    """A prompt's last text (token ids; an image's rows, negative, are none), without trailing white space."""
    last = [int(t) for t in list(ids)[-8:] if t >= 0]
    return tokenizer.decode_bytes(last, special=True).decode("utf-8", errors="replace").rstrip()


def ends_inside(tokenizer: Any, ids: Sequence[int], opener: str) -> bool:
    """Whether a prompt (token ids; an image's rows, negative, are none) ends inside a thought: its last text is the
    model's open marker (the template opened it)."""
    return _tail(tokenizer, ids).endswith(opener)


def prompt_thinks(tokenizer: Any, ids: Sequence[int], reasoning: tuple[str, str] | None, think: bool, *,
                  blocks: "Blocks | None" = None) -> bool:
    """Whether the model thinks first after a prompt, as its template rendered it: the prompt ends inside a thought
    (the template opened one) or after an empty one (it closed it), else ``think`` (what the template was asked; a
    model that opens its thought itself). A template may decide against ``enable_thinking`` (inline tags such as
    ``<|think_off|>``, its own spellings of ``reasoning_effort``); every model's own template follows it. A model that
    writes its turn as blocks (``Model.blocks``: every block ends with the thought's close marker) thinks first unless
    the prompt opened its answer's text: it writes a thought at any effort."""
    if reasoning is None:
        return think
    tail = _tail(tokenizer, ids)
    if blocks is not None:
        return not tail.endswith(blocks.answer)
    return True if tail.endswith(reasoning[0]) else False if tail.endswith(reasoning[1]) else think


def thought_of(reasoning: tuple[str, str] | None, tokenizer: Any, *, opened: bool, think: bool,
               ends: tuple[str, ...] = ()) -> "Thought | None":
    """Where a model's thought is (``Model.reasoning``: its markers; ``opened``: the prompt opened one, ``ends_inside``;
    ``think``: the model thinks first; ``ends``: texts that end it and begin the answer), or None: the answer starts
    at once (a grammar that holds the whole turn, ``Compiled.turn``, has none before it)."""
    if reasoning is None or not (opened or think):
        return None
    return Thought(reasoning[1], inside=opened, opener=None if opened else tokenizer.encode(reasoning[0])[0], ends=ends)


class Ban:
    """Tokens never sampled (``sampling.Mask``: ``tool_choice`` none, a call's opener)."""

    def __init__(self, v: Vocab, ids: Sequence[int]):
        allowed = np.ones(v.n, bool)
        allowed[list(ids)] = False
        self._bits = pack_mask(allowed, v.words)

    def bits(self) -> np.ndarray:
        return self._bits

    def accept(self, token: int) -> None:
        pass

    def first_allowed(self, candidates: Sequence[int]) -> int:
        return next((i for i, t in enumerate(candidates) if self._bits[t >> 5] >> (t & 31) & 1), -1)


@dataclasses.dataclass(frozen=True)
class Thought:
    """Where a generation's thought is (the answer's constraint starts after it): ``close``, the text that ends it
    (``Model.reasoning``); ``inside``: the prompt opened it; ``opener``: the token that opens one, which the model may
    write first (it thinks and the prompt did not open it), or None; ``ends``: texts that end it and begin the answer (a
    Qwen tool call: ``tools.Syntax.ends``)."""

    close: str
    inside: bool = False
    opener: int | None = None
    ends: tuple[str, ...] = ()


class GrammarMask:
    """A grammar's tokens (``sampling.Mask``): an llguidance matcher of ``grammar`` over ``v``'s tokens, told each token
    of the answer (``accept``), and the next one's ``bits``. With ``thought`` the mask starts where the thought ends (a
    thought the model may open first: its opener is allowed beside the grammar's tokens). ``digits``: a JSON answer's
    digit runs outside strings stop at ``DIGITS`` (``opens``: counted after these tokens, which open the answer in a
    grammar of the whole turn: ``Compiled.opens``). A token the matcher rejects (the generation already holds it: a
    forced opener out of step) ends the constraint with a log line (``name``: the request's): the answer goes on
    unconstrained rather than in garbage."""

    def __init__(self, v: Vocab, grammar: str, *, thought: Thought | None = None, digits: bool = False,
                 name: str = "", opens: Sequence[int] = ()):
        self.v, self.name, self.digits = v, name, digits
        self._opens, self._last = tuple(opens), ()  # (the answer's opening tokens; the last tokens until they came)
        self.matcher = llguidance.LLMatcher(v.tokenizer, grammar, log_level=0, limits=LIMITS)
        if self.matcher.is_error():
            raise ConstraintError(_message(self.matcher.get_error()))
        self._row = np.empty((1, v.words), np.int32)
        self._bits: np.ndarray | None = None
        self.phase = "answer"  # answer, maybe (a thought may open first), thought, off
        self.thought = thought
        self._tail = b""  # (a thought's last bytes: its end marker)
        if thought is not None:
            self.phase = "thought" if thought.inside else "maybe" if thought.opener is not None else "answer"
        self._in_string = self._escape = False
        self._run = 0  # (the digits ending the answer outside a string)
        # (``prefetch``: the masks after the tokens accepted since, its drafts, computed aside; those tokens)
        self._pre: tuple[concurrent.futures.Future, np.ndarray, list[int]] | None = None
        self._since: list[int] = []
        self._shared = False  # (a fork's matcher is still its origin's: a fork made inside a thought)

    def fork(self) -> "GrammarMask":
        """An independent copy (a drafter's: its guesses follow the grammar, this one does not move). Inside a thought
        it shares the matcher, which neither moves there, and takes a separate copy where its thought ends (a
        drafter's guesses go on past the end marker into the answer: Spark-X's DSpark chains, Gemma's MTP)."""
        out = copy.copy(self)
        out._shared = self.phase not in ("answer", "maybe")
        out.matcher = self.matcher if out._shared else self.matcher.deep_copy()
        out._row, out._bits, out._pre, out._since = np.empty_like(self._row), None, None, []
        return out

    def _answer(self) -> None:
        """The answer begins: the matcher moves from here (a fork that shared its origin's takes a copy)."""
        self.phase = "answer"
        if self._shared:
            self.matcher, self._shared = self.matcher.deep_copy(), False

    def valid_prefix(self, tokens: Sequence[int]) -> int:
        """How many of ``tokens`` the grammar takes from here (drafts: a round's are cut there)."""
        if self.phase in ("off", "thought") or not tokens:
            return len(tokens)
        if self.phase == "maybe" and tokens[0] == self.thought.opener:
            return len(tokens)  # (a thought opens: free)
        self._join()
        return self.matcher.validate_tokens(list(tokens))

    def first_allowed(self, candidates: Sequence[int]) -> int:
        """The index of the first of ``candidates`` the grammar takes next, or -1 (a drafter's guesses, best first: a
        check of each, not the whole vocabulary's mask)."""
        if self.phase in ("off", "thought"):
            return 0 if len(candidates) else -1
        for i, t in enumerate(candidates):
            if self.valid_prefix([int(t)]) == 1:
                return i
        return -1

    def forced(self) -> list[int]:
        """The tokens the grammar forces next (its canonical ones: a round's drafts that it keeps by construction)."""
        if self.phase != "answer":
            return []
        self._join()
        return self.matcher.compute_ff_tokens()

    def prefetch(self, drafts: Sequence[int] = ()) -> None:
        """Compute the next choices' masks aside while the NPU runs a step (one worker thread; llguidance drops the
        GIL): the one after the tokens accepted so far, and after each of ``drafts`` in turn as far as the grammar takes
        them, on a copy of the matcher; ``bits`` takes them as the step's tokens are accepted."""
        if self.phase != "answer":
            return
        self._join()
        fork = self.matcher.deep_copy()
        rows = np.empty((len(drafts) + 1, self.v.words), np.int32)
        d = list(drafts)

        def run() -> int:
            for i in range(len(d) + 1):
                fork.unsafe_compute_mask_ptr(rows[i].ctypes.data, rows[i].nbytes)
                if i == len(d) or not (rows[i, d[i] >> 5] >> (d[i] & 31)) & 1 or not fork.consume_token(d[i]):
                    return i + 1
            return len(d) + 1

        self._pre, self._since = (_aside.submit(run), rows, d), []

    def _join(self) -> int:
        """Wait for the masks computed aside (the matcher's copy is done with); how many there are."""
        return self._pre[0].result() if self._pre is not None else 0

    @property
    def active(self) -> bool:
        """The grammar holds the next token (the answer has begun and the constraint was not given up)."""
        return self.phase in ("answer", "maybe")

    def bits(self) -> np.ndarray | None:
        if not self.active:
            return None
        if self._bits is None:
            k, n = len(self._since), self._join()
            if self._pre is not None and k < n and self._since == self._pre[2][:k]:  # (computed aside: ``prefetch``)
                bits = self._pre[1][k].view(np.uint32)
            else:
                self._pre = None
                self.matcher.unsafe_compute_mask_ptr(self._row.ctypes.data, self._row.nbytes)
                bits = self._row[0].view(np.uint32)
            if self.phase == "maybe":
                bits = bits.copy()
                o = self.thought.opener
                bits[o // 32] |= np.uint32(1 << (o % 32))
            elif self.digits and not self._in_string and self._run + self.v.max_digits > DIGITS:
                cut = bits & ~self.v.digit_bits(max(DIGITS - self._run + 1, 1))
                if cut.any():
                    bits = cut
            self._bits = bits
        return self._bits

    def accept(self, token: int) -> None:
        self._bits = None
        if self.phase == "off":
            return
        if self.phase == "thought":
            self._in_thought(token)
            return
        if self.phase == "maybe":
            if token == self.thought.opener:
                self.phase = "thought"
                return
            self.phase = "answer"
        self._consume(token)

    def _consume(self, token: int) -> None:
        self._join()
        if self._pre is not None:
            self._since.append(token)
        if not self.matcher.consume_token(token):
            piece = self.v.pieces[token][:40] if token < len(self.v.pieces) else b"?"
            self.give_up(f"the grammar rejects token {token} ({piece!r}): "
                         f"{_message(self.matcher.get_error()).splitlines()[0]}")
            return
        if self.digits and self._opens:
            self._last = (*self._last, token)[-len(self._opens):]
            if self._last == self._opens:
                self._opens, self._in_string, self._run = (), False, 0
        elif self.digits:
            self._digits(self.v.pieces[token])

    def _in_thought(self, token: int) -> None:
        """A token of the thought: the answer begins after its end marker (or with a text that ends it)."""
        th = self.thought
        tail = self._tail + self.v.pieces[token]
        close = th.close.encode()
        for end in th.ends:
            if tail.endswith(end.encode()) and close not in tail:
                self._answer()
                self._consume(token)  # (the text that ended the thought begins the answer)
                return
        i = tail.find(close)
        if i >= 0:
            self._answer()
            if tail[i + len(close):].strip():
                self.give_up("the answer's first text came with the thought's end marker")
            return
        keep = max(len(close), *(len(e.encode()) for e in th.ends)) if th.ends else len(close)
        self._tail = tail[-keep:]

    def _digits(self, piece: bytes) -> None:
        """The JSON answer's state after a token's bytes: inside a string, after a backslash, the digits ending it."""
        for b in piece:
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif b == 0x5C:
                    self._escape = True
                elif b == 0x22:
                    self._in_string = False
            elif b == 0x22:
                self._in_string, self._run = True, 0
            elif 0x30 <= b <= 0x39:
                self._run += 1
            else:
                self._run = 0

    def give_up(self, why: str) -> None:
        """End the constraint (the rest of the answer unconstrained), with a log line."""
        if self.phase != "off":
            self.phase = "off"
            log.warning("%sstructured output given up: %s", f"{self.name}: " if self.name else "", why)

    @property
    def complete(self) -> bool:
        """The grammar is satisfied (only the end of the answer may follow)."""
        return self.phase == "answer" and self.matcher.is_stopped() and not self.matcher.is_error()
