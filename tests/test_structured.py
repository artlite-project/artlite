"""Structured output's core on a toy tokenizer (no model files, no NPU): constraints compiled by llguidance
(``models.structured``) and followed token by token by ``GrammarMask`` - the schema rules (key order, closed objects,
bounded white space, no fences, digit runs), strict and lenient schemas, the limits, GBNF / Lark / regex / choice /
structural tags, the thought gate, the end of the grammar, a desync - and the sampler's choices under the mask."""

import json

import numpy as np
import pytest

from models import structured as S
from models.sampling import Penalties, Sampling, seed

SPECIAL = ["<|end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
WORDS = [b'{"', b'":', b'",', b'"}', b" {", b"true", b"false", b"null", b"name", b"city", b"123", b"45", b"```",
         b"\n\n", b"   ", b"0.", b"thought"]


class ToyTokenizer:
    """256 byte tokens, some words, then the special tokens (``END``: the end of sequence); text encodes greedily,
    longest piece first (no special tokens: ``encode_plain``; with them: ``encode``)."""

    def __init__(self) -> None:
        self.pieces = [bytes([i]) for i in range(256)] + WORDS + [s.encode() for s in SPECIAL]
        self.special = list(range(256 + len(WORDS), len(self.pieces)))
        self.index = {p: i for i, p in enumerate(self.pieces) if i not in self.special}
        self.longest = max(map(len, self.pieces))

    def id(self, s: str) -> int:
        return self.pieces.index(s.encode())

    def vocabulary(self) -> tuple[list[bytes], list[int]]:
        return list(self.pieces), list(self.special)

    def encode_plain(self, text: str) -> list[int]:
        b, out, i = text.encode(), [], 0
        while i < len(b):
            n = next(n for n in range(min(self.longest, len(b) - i), 0, -1) if b[i:i + n] in self.index)
            out.append(self.index[b[i:i + n]])
            i += n
        return out

    def encode(self, text: str) -> list[int]:
        out, rest = [], text
        while rest:
            hits = [(rest.find(s), s) for s in SPECIAL if s in rest]
            if not hits:
                return out + self.encode_plain(rest)
            i, s = min(hits)
            out += self.encode_plain(rest[:i]) + [self.id(s)]
            rest = rest[i + len(s):]
        return out


TOK = ToyTokenizer()
V = S.Vocab(TOK, 0, [TOK.id("<|end|>")])
END = TOK.id("<|end|>")


def allowed(bits: np.ndarray | None) -> set[int] | None:
    return None if bits is None else set(np.flatnonzero(np.unpackbits(bits.view(np.uint8), bitorder="little")).tolist())


def mask_of(c: S.Constraint, **kw: object) -> S.GrammarMask:
    return S.GrammarMask(V, S.compiled(c, V).grammar, digits=c.json, **kw)


def walk(m: S.GrammarMask, text: str | list[int]) -> int:
    """Feed a text's tokens to a mask, each checked against the mask before it; returns how many were allowed (all:
    len)."""
    ids = TOK.encode(text) if isinstance(text, str) else text
    for n, t in enumerate(ids):
        a = allowed(m.bits())
        if a is not None and t not in a:
            return n
        m.accept(t)
    return len(ids)


def accepts(c: S.Constraint, text: str) -> bool:
    """Whether the constraint takes the whole text and then ends: its canonical tokens each allowed by the mask
    (llguidance's mask offers a forced text's canonical tokens only), its bytes as tokens through the matcher (any
    split of them fits)."""
    m = mask_of(c)
    ids = TOK.encode_plain(text)
    if walk(m, ids) != len(ids) or END not in allowed(m.bits()):
        return False
    b = mask_of(c).matcher
    return all(b.consume_token(x) for x in text.encode()) and b.is_accepting()


OBJ = S.Constraint.json_schema({"type": "object", "properties": {"name": {"type": "string"}, "n": {"type": "integer"},
                                                                 "x": {"type": "number"}}, "required": ["name"]})


def test_json_rules() -> None:
    """Keys in the schema's order, declared objects closed and undeclared ones open, white space bounded (a few blanks
    before the value, llama.cpp's space between tokens, none after), no fences, the end of sequence once complete."""
    assert accepts(OBJ, '{"name": "a", "n": 1}')
    assert accepts(OBJ, ' {"name":"a"}') and accepts(OBJ, '\n\n{\n  "name": "a",\n  "x": 1.5\n}')
    assert not accepts(OBJ, '{"n": 1, "name": "a"}')  # (the schema's order)
    assert not accepts(OBJ, '{"name": "a", "other": 1}')  # (properties declared: closed)
    assert not accepts(OBJ, '```json\n{"name": "a"}\n```') and not accepts(OBJ, '{"name": "a"} ')
    assert not accepts(OBJ, " " * (S.LEAD + 1) + '{"name": "a"}')
    assert not accepts(OBJ, '{"name":    "a"}')  # (llama.cpp's space: one blank)
    assert accepts(OBJ, '{"name":\n\n\t"a"}') and not accepts(OBJ, '{"name":\n\n\n"a"}')
    free = S.Constraint.json_schema({"type": "object"})
    assert accepts(free, '{"anything": [1, {"b": null}]}')
    opened = S.Constraint.json_schema({"type": "object", "properties": {"a": {"type": "integer"}},
                                       "additionalProperties": True})
    assert accepts(opened, '{"a": 1, "b": 2}')
    closed = S.Constraint.json_schema({"type": "object", "properties": {"o": {"type": "object"}}}, closed=True)
    assert not accepts(closed, '{"o": {"k": 1}}') and accepts(closed, '{"o": {}}')
    compact = S.Constraint.json_schema({"type": "object", "properties": {"a": {"type": "integer"}}}, compact=True)
    assert accepts(compact, '{"a":1}') and not accepts(compact, '{"a": 1}')
    m = mask_of(OBJ)
    assert walk(m, '{"name": "a"}') == len(TOK.encode('{"name": "a"}')) and allowed(m.bits()) == {END} and m.complete


def test_closed_objects() -> None:
    """``closed_objects``: every object that declares properties, in every subschema, but those whose keys another
    subschema of the same instance adds to (an allOf's members, the anyOf / oneOf members of an object with properties,
    an object next to a combinator, the definitions an allOf member refers to)."""
    s = {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {}}}},
         "$defs": {"d": {"properties": {"c": {}}}, "base": {"properties": {"i": {}}}},
         "anyOf": [{"properties": {"e": {}}}], "allOf": [{"properties": {"f": {}}}, {"$ref": "#/$defs/base"}],
         "items": {"properties": {"h": {}}}, "additionalProperties": True}
    out = S.closed_objects(s)
    assert out["additionalProperties"] is True
    assert out["properties"]["a"]["additionalProperties"] is False and out["$defs"]["d"]["additionalProperties"] is False
    assert out["items"]["additionalProperties"] is False
    assert "additionalProperties" not in out["anyOf"][0] and "additionalProperties" not in out["$defs"]["base"]
    assert all("additionalProperties" not in m for m in out["allOf"])
    assert "additionalProperties" not in s["properties"]["a"]  # (a copy)
    union = S.closed_objects({"anyOf": [{"properties": {"x": {}}}, {"properties": {"y": {}}}]})  # (alternatives)
    assert all(m["additionalProperties"] is False for m in union["anyOf"])
    both = S.Constraint.json_schema({"allOf": [{"properties": {"f": {"type": "integer"}}},
                                               {"properties": {"g": {"type": "integer"}}}]})
    assert accepts(both, '{"f": 1, "g": 2}')
    extends = S.Constraint.json_schema({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"],
                                        "anyOf": [{"properties": {"b": {"type": "integer"}}, "required": ["b"]},
                                                  {"properties": {"c": {"type": "integer"}}, "required": ["c"]}]})
    assert accepts(extends, '{"a": 1, "b": 2}')
    inherits = S.Constraint.json_schema({"$defs": {"Base": {"type": "object", "properties": {"id": {"type": "integer"}},
                                                            "required": ["id"]}},
                                         "allOf": [{"$ref": "#/$defs/Base"}, {"properties": {"extra": {"type": "integer"}},
                                                                              "required": ["extra"]}]})
    assert accepts(inherits, '{"id": 1, "extra": 2}')


def test_digits() -> None:
    """A run of digits outside strings stops at ``DIGITS`` (an integral part, a fraction, an exponent); inside a string
    it does not."""
    c = S.Constraint.json_schema({"type": "object", "properties": {"n": {"type": "number"}, "s": {"type": "string"}}})
    d = "1" * S.DIGITS
    assert accepts(c, f'{{"n": {d}.{d}e{d}}}') and not accepts(c, f'{{"n": {d}1}}')
    m = mask_of(c)
    assert walk(m, f'{{"n": 1.{d}') == len(TOK.encode(f'{{"n": 1.{d}'))
    a = allowed(m.bits())
    assert not any(TOK.pieces[t][:1].isdigit() for t in a) and TOK.encode_plain(",")[0] in a
    assert accepts(c, f'{{"s": "{d * 3}"}}')  # (in a string: any)
    m = mask_of(c)
    walk(m, '{"s": "\\"' + d)  # (an escaped quote: still in the string)
    assert any(TOK.pieces[t][:1].isdigit() for t in allowed(m.bits()))
    exact = S.Constraint.json_schema({"type": "integer", "minimum": 10 ** 17})  # (more digits needed: not cut)
    assert accepts(exact, str(10 ** 17 + 7))


def test_strict_lenient() -> None:
    """A strict schema with a keyword llguidance does not implement: an error that names it; lenient: compiled, the
    keyword not enforced; the limits; a schema that is not one."""
    s = {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
    with pytest.raises(S.ConstraintError, match="uniqueItems"):
        S.compiled(S.Constraint.json_schema(s, strict=True), V)
    assert accepts(S.Constraint.json_schema(s), "[1, 1]")
    with pytest.raises(S.ConstraintError, match="properties"):
        S.Constraint.json_schema({"type": "object", "properties": {f"p{i}": {} for i in range(S.MAX_PROPERTIES + 1)}})
    with pytest.raises(S.ConstraintError, match="enum"):
        S.Constraint.json_schema({"enum": list(range(S.MAX_ENUM + 1))})
    with pytest.raises(S.ConstraintError, match="MiB"):
        S.Constraint.json_schema({"description": "x" * S.MAX_SCHEMA})
    with pytest.raises(S.ConstraintError):
        S.Constraint.json_schema(42)
    deep: dict = {"type": "integer"}
    for _ in range(S.MAX_DEPTH):
        deep = {"type": "array", "items": deep}
    with pytest.raises(S.ConstraintError, match="nested"):
        S.Constraint.json_schema(deep)
    with pytest.raises(S.ConstraintError):
        S.compiled(S.Constraint.json_schema({"type": "kaboom"}, strict=True), V)
    with pytest.raises(S.ConstraintError):
        S.compiled(S.Constraint.json_schema({"$ref": "#/$defs/missing"}), V)


def test_grammars() -> None:
    """GBNF (llama.cpp's syntax: repetitions, classes, comments) converted to Lark, Lark, a regex, choices, a structural
    tag (free text, then a structure after its trigger: a text one and a special token's)."""
    gbnf = 'root ::= "a"{2,3} digit+ # a comment\ndigit ::= [0-9]\n'
    assert accepts(S.Constraint("gbnf", gbnf), "aa12") and not accepts(S.Constraint("gbnf", gbnf), "a1")
    with pytest.raises(S.ConstraintError, match="GBNF"):
        S.compiled(S.Constraint("gbnf", "root ::= [a-z"), V)
    lark = '%llguidance {}\nstart: "x" /[0-9]{2}/\n'
    assert accepts(S.Constraint("lark", lark), "x42") and not accepts(S.Constraint("lark", lark), "x4")
    assert accepts(S.Constraint("regex", r"[a-c]+\d"), "abc7")
    choice = S.Constraint("choice", json.dumps(["yes", "no way"]))
    assert accepts(choice, "no way") and not accepts(choice, "nope")
    tag = [{"trigger": "<f=", "begin": "<f=w>", "schema": {"type": "object", "properties": {"c": {"type": "string"}},
                                                           "required": ["c"]}, "end": "</f>"}]
    c = S.Constraint("structural_tag", json.dumps(tag))
    assert accepts(c, 'some text <f=w>{"c": "x"}</f> more')
    m = mask_of(c)
    assert walk(m, "hello <f=w>{\"q") < len(TOK.encode("hello <f=w>{\"q"))
    tag = [{**tag[0], "trigger": "<tool_call>", "begin": "<tool_call>"}]
    m = mask_of(S.Constraint("structural_tag", json.dumps(tag)))
    ids = TOK.encode('ok <tool_call>{"c": "x"}</f>')
    assert walk(m, ids) == len(ids)


def test_thought() -> None:
    """The mask starts where the thought ends: inside one (the prompt opened it), after its end marker; one the model
    may open first (its opener allowed beside the grammar's tokens); a text that ends the thought begins the answer."""
    close = "</think>"
    m = mask_of(OBJ, thought=S.Thought(close, inside=True))
    assert m.bits() is None and walk(m, "anything at all ```") == len(TOK.encode("anything at all ```"))
    walk(m, close)
    assert m.active and TOK.encode('{"')[0] in allowed(m.bits())
    assert walk(m, '\n\n{"name": "a"}') == len(TOK.encode('\n\n{"name": "a"}'))
    opener = TOK.id("<think>")
    m = mask_of(OBJ, thought=S.Thought(close, opener=opener))
    a = allowed(m.bits())
    assert opener in a and TOK.encode('{"')[0] in a and TOK.encode("x")[0] not in a
    m.accept(opener)
    assert m.bits() is None
    m = mask_of(OBJ, thought=S.Thought(close, opener=opener))
    assert walk(m, '{"name": "b"}') == len(TOK.encode('{"name": "b"}'))  # (no thought: the answer at once)
    call = S.Constraint("lark", '%llguidance {}\nstart: <tool_call> "x" </tool_call>\n')
    m = mask_of(call, thought=S.Thought(close, inside=True, ends=("<tool_call>",)))
    walk(m, "hmm ")
    m.accept(TOK.id("<tool_call>"))
    assert m.active and allowed(m.bits()) == {TOK.encode_plain("x")[0]}
    # a drafter's fork inside the thought, its guesses past the end marker into the answer: the mask stays put
    m = mask_of(call, thought=S.Thought(close, inside=True))
    walk(m, "hmm ")
    f = m.fork()
    walk(f, close)
    f.accept(TOK.id("<tool_call>"))
    assert allowed(f.bits()) == {TOK.encode_plain("x")[0]}
    walk(m, close)
    assert allowed(m.bits()) == {TOK.id("<tool_call>")}


def test_give_up() -> None:
    """A token the grammar rejects (the generation holds it already) ends the constraint, logged, instead of masking
    garbage."""
    m = mask_of(OBJ)
    m.accept(TOK.encode_plain("x")[0])
    assert m.phase == "off" and m.bits() is None
    m.accept(TOK.encode_plain("y")[0])  # (unconstrained from there)


def test_sampling() -> None:
    """The sampler under a grammar's mask: greedy over random logits is the argmax of the allowed ones, a sampled
    choice is always allowed, and the answer parses."""
    rng = np.random.default_rng(0)
    seed(0)
    for sampling in (Sampling(greedy=True), Sampling(temp=1.0, top_k=40)):
        m = mask_of(OBJ)
        p = Penalties(sampling, (), m)
        out = []
        for _ in range(2000):
            logits = rng.standard_normal(V.n).astype(np.float32) * 3
            a = allowed(m.bits())
            v, i = p.top(logits, sampling.k)
            assert set(i.tolist()) <= a
            if sampling.greedy:
                assert int(i[0]) == max(a, key=lambda t: (logits[t], -t))
            t = sampling.pick(v, i)
            if t == END:
                break
            out.append(t)
            p.accept(t)
        text = b"".join(TOK.pieces[t] for t in out)
        assert END in allowed(m.bits()) and set(json.loads(text)) <= {"name", "n", "x"}


def test_cache() -> None:
    """A constraint compiles once (the same object back); another one is another grammar."""
    a = S.compiled(S.Constraint.json_schema({"type": "string"}), V)
    assert S.compiled(S.Constraint.json_schema({"type": "string"}), V) is a
    assert S.compiled(S.Constraint.json_schema({"type": "integer"}), V) is not a


# ----------------------------------------------------------------------
# The masks and the multi-token steps (``speculative.SpeculativeMixin``)
# ----------------------------------------------------------------------


def test_prefetch() -> None:
    """Masks computed aside (``prefetch``: after the accepted tokens, after each draft in turn) are the masks computed
    when they are taken, bit for bit, whatever part of the drafts the step keeps; ``fork`` moves alone; ``valid_prefix``
    cuts drafts where the grammar refuses them; ``forced`` is the grammar's canonical next tokens."""
    rng = np.random.default_rng(3)
    text = TOK.encode_plain('{"name": "Bob 123", "n": 45, "x": 0.5}')
    for keep in range(5):
        a, b = mask_of(OBJ), mask_of(OBJ)
        i = 0
        while i < len(text):
            drafts = text[i + 1:i + 4] if rng.random() < 0.7 else rng.integers(0, V.n, 3).tolist()
            a.prefetch(drafts)
            for j in range(min(keep, len(drafts)) + 1):
                if i >= len(text):
                    break
                assert np.array_equal(a.bits(), b.bits()), (keep, i)
                a.accept(text[i])
                b.accept(text[i])
                i += 1
                if j < len(drafts) and drafts[j] != text[i - 1]:
                    break
    m = mask_of(OBJ)
    walk(m, '{"name": "a"')
    f = m.fork()
    f.accept(TOK.encode_plain("}")[0])
    assert f.complete and not m.complete  # (the fork moved alone)
    assert m.valid_prefix(TOK.encode_plain(', "n": 1')) == len(TOK.encode_plain(', "n": 1'))
    assert m.valid_prefix(TOK.encode_plain(', "q": 1')) < len(TOK.encode_plain(', "q": 1'))
    fixed = S.Constraint.json_schema({"type": "object", "properties": {"alpha": {"type": "integer"}},
                                      "required": ["alpha"]}, compact=True)
    m = mask_of(fixed)
    assert m.forced() == []  # (blanks may come first)
    walk(m, "{")
    assert b"".join(TOK.pieces[t] for t in m.forced()) == b'"alpha":'[:len(b"".join(TOK.pieces[t] for t in m.forced()))]
    assert len(m.forced()) >= 2


class TextModel:
    """A model whose next token after position q is ``text[q + 1]`` (``test_drafts.Text``) over the toy vocabulary."""

    @staticmethod
    def make(text: list[int]) -> object:
        from models.overlay_model import OverlayModel

        class M(OverlayModel):
            eos_token_ids = (END,)
            capacity = 512
            overlay = type("O", (), {"once_s": 0.0})()

            def __init__(self) -> None:
                self.text, self.position, self.calls = text, 0, []

            def steps(self, tokens: list[int], *, logits: bool = True) -> np.ndarray:
                from ml_dtypes import bfloat16
                p = self.position
                self.calls.append(list(tokens))
                self.position += len(tokens)
                out = np.full((len(tokens), V.n), -8.0, np.float32)
                for i in range(len(tokens)):
                    out[i, self.text[min(p + i + 1, len(self.text) - 1)]] = 8.0
                return out.astype(bfloat16)

            def _crossing(self) -> tuple[bool, ...]:
                return (False,) * (self.DRAFTS + 1)

            def first(self, n: int) -> np.ndarray:
                self.position = n
                out = np.full(V.n, -8.0, np.float32)
                out[self.text[n]] = 8.0
                return out

        return M()


def test_forced_drafts() -> None:
    """A model that writes a JSON answer, greedy: the same tokens with and without drafts, with and without the
    grammar; with it, the tokens the grammar forces (a fixed schema's keys) become separate drafts, kept by
    construction, and the answer takes fewer steps than tokens; the context's drafts the grammar refuses are cut."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"type": "string",
                                                                                    "enum": ["alpha", "beta"]},
                                               "count": {"type": "integer"}}, "required": ["name", "kind", "count"]}
    c = S.Constraint.json_schema(schema, compact=True)
    answer = TOK.encode_plain('{"name":"Bob","kind":"alpha","count":12}')
    prompt = TOK.encode_plain("Q: ")
    text = prompt + answer + [END]
    P = len(prompt)
    greedy = Sampling(greedy=True)
    plain = TextModel.make(text)
    plain.lookup = False
    want = list(plain.generate(plain.first(P), -1, greedy, text[:P]))
    assert want == answer + [END]
    for lookup in (False, True):
        m = TextModel.make(text)
        m.lookup = lookup
        out = list(m.generate(m.first(P), -1, greedy, text[:P], mask_of(c)))
        assert out == want and m.forced > 0 and len(m.calls) < len(out) - 1, (lookup, m.calls)
        for call in m.calls:  # (every step's drafts the grammar's or kept: none refused)
            assert len(call) <= m.DRAFTS + 1
