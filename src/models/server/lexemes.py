"""Whether a custom tool's Lark grammar can end inside a call (``ending``). llguidance lexes greedily: inside a lexeme
it takes every character a lexeme under way can go on with, and only where none can does it end one and try the
lexemes that may start there, the call's closing text among them. So a grammar ends inside the call unless a lexeme
allowed where the input's last one starts goes on, after a complete match, with the closing text's first character (a
catch-all such as Codex's ``/[\\s\\S]+/``, ``/@[\\s\\S]*/``, a line's ``/(.+)/`` before GLM-4.5's ``</arg_value>``, the
literal ``"a<b"`` beside ``"a"``), or the lexemes that may come after the input spell two characters of the closing text
or more, one after another (``"</"`` before ``</arg_value>``, ``"\\n</parameter>x"``: where they may end inside it,
llguidance takes them for the input; one character, ``<`` or a line's end, does no harm), or go on past one of the
closing text's own lexemes (Qwen's newline, then its tag: a ``"\\n<b>"`` or ``"\\n>"`` after the input; llguidance keeps to
the longer lexeme); there the input is free text, which ends (``call_grammar``). A lexeme that matches the empty text
too and may follow itself, or in a grammar that ignores lexemes, counts as going on (llguidance's lexing of one is its
own).

Qwen's closing text is ``"\\n</parameter>"``, whose newline its parser takes off the parameter's text. The newline is
optional (``Ending.newline``) where taking it off changes nothing: no lexeme that may end the input can end with a
newline, and an ignored one (``%ignore WS``) is still one, or none, without its last newline; a lexeme may then take
that newline if it can end the input right after it and not go on with ``<``. Elsewhere it is the call's own (``(LINE LF)+``: the parameter's text ends with the input's newline and
the call's), and a lexeme must not go on with it.

The rules give the lexemes' order: the ones that may start and end the input and the ones that may follow each one
(FIRST, LAST, FOLLOW over the rules' expressions; an ignored lexeme anywhere). The lexemes - terminals (an imported
``common`` one too), the regular expressions and literals of the rules and ``%ignore`` lines - become Python regular
expressions, an NFA each (Thompson's construction on ``re._parser``'s parse; a bounded repeat beyond ``MAX_REPEAT``, an
anchor or a lookaround: a larger language), and the ones allowed where a last one starts a DFA (subset construction
over characters that tell the transitions apart). What does not translate (llguidance's ``&`` and ``~``, its ``%regex``
and inline grammars, templates, ``%llguidance`` options, an unknown import, a back reference, a class Rust reads
otherwise than Python: ``[[:punct:]]``, ``[a&&b]``), or exceeds the limits, counts as going on; a repetition's blanks
are Rust's (``{ 1,}``: one or more)."""

import ast
import collections
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from re import _constants as K
from re import _parser as P

MAX_GRAMMAR = 1 << 16  # a grammar's text analysed at most
MAX_PATTERN = 1 << 15  # a lexeme's regular expression at most (its terminals inlined)
MAX_NFA = 1 << 15  # NFA states at most
MAX_STATES = 4096  # DFA states at most
MAX_WORK = 1 << 21  # transitions tested at most (DFA states x characters x their NFA states' edges)
MAX_REPEAT = 8  # a bounded repeat's copies at most (more: unbounded)
CACHED = 64  # grammars' endings kept

# lark's common terminals (llguidance's ``%import common``), as Python regular expressions
COMMON = {
    "DIGIT": r"[0-9]", "HEXDIGIT": r"[a-fA-F0-9]", "INT": r"[0-9]+", "SIGNED_INT": r"[+-]?[0-9]+",
    "DECIMAL": r"[0-9]+\.[0-9]*|\.[0-9]+", "_EXP": r"[eE][+-]?[0-9]+",
    "FLOAT": r"[0-9]+[eE][+-]?[0-9]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
    "SIGNED_FLOAT": r"[+-]?(?:[0-9]+[eE][+-]?[0-9]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)",
    "NUMBER": r"[0-9]+[eE][+-]?[0-9]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+",
    "SIGNED_NUMBER": r"[+-]?(?:[0-9]+[eE][+-]?[0-9]+|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+)",
    "ESCAPED_STRING": r'"(?:[^"\\]|\\.)*"', "LCASE_LETTER": r"[a-z]", "UCASE_LETTER": r"[A-Z]",
    "LETTER": r"[a-zA-Z]", "WORD": r"[a-zA-Z]+", "CNAME": r"[_a-zA-Z][_a-zA-Z0-9]*", "WS_INLINE": r"[ \t]+",
    "WS": r"[ \t\f\r\n]+", "CR": r"\r", "LF": r"\n", "NEWLINE": r"(?:\r?\n)+", "SH_COMMENT": r"#[^\n]*",
    "CPP_COMMENT": r"//[^\n]*", "C_COMMENT": r"/\*(?:.|\n)*\*/", "SQL_COMMENT": r"--[^\n]*",
}
# characters that stand for the classes regular expressions tell apart (with each expression's own: ``_Nfa.chars``)
SAMPLES = "\x00\t\n\x0b\x0c\r !\"#$%&'()*+,-./09:;<=>?@AZ[\\]^_`az{|}~\x7f\x85\xa0é\xdfΩ٣中 　\U0001f600"
_TOKEN = re.compile(r'(?P<s>"(?:\\.|[^"\\\n])*"i?)|(?P<r>/(?:\\.|[^/\\\n])+/[imslux]*)|(?P<n>[A-Za-z_][A-Za-z0-9_]*)'
                    r"|(?P<num>\d+)|(?P<op>\.\.|[()\[\]|*+?~])|(?P<ws>\s+)|(?P<c>//[^\n]*)|(?P<x>.)", re.DOTALL)
_HEAD = re.compile(r"^[?!]?([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?(?:\.-?\d+)?\s*:")


class Unsupported(ValueError):
    """A lexeme this module does not translate, or one over the limits."""


@dataclass(frozen=True)
class Ending:
    """How a custom tool's input ends inside a call: ``over``, the lexeme that would go on with the closing text (or
    why the grammar is not analysed), None where none does; ``newline``: Qwen's closing newline optional."""

    over: str | None
    newline: bool = False


MAX_LEXEMES = 512  # places the rules name a lexeme at most
MAX_PAIRS = 1 << 15  # pairs of lexemes one after the other at most
MAX_ROUNDS = 64  # rounds of the rules' fixpoint at most

Node = tuple  # a rule's expression: ("sym", lexeme), ("ref", rule), ("seq", [...]), ("alt", [...]), ("opt", x),
# ("rep", x, lo, hi: None unbounded), ("empty",)


@dataclass(frozen=True)
class _Order:
    """A grammar's lexemes in their rules' order, each place where a rule names one a separate position (``label``: the
    lexeme as the grammar writes it): their patterns, the ones that may start and end the input, the ones that may follow
    each, the ignored ones, whether the input may be empty."""

    patterns: dict[str, str]
    first: frozenset[str]
    last: frozenset[str]
    follow: dict[str, frozenset[str]]
    ignored: frozenset[str]
    empty: bool


def label(pos: str) -> str:
    """The lexeme of a position, as the grammar writes it."""
    return pos.split("\x00")[0]


_cache: collections.OrderedDict[tuple[str, str], Ending] = collections.OrderedDict()
_lock = threading.Lock()


def ending(lark: str, end: str) -> Ending:
    """How a Lark grammar's input ends before ``end`` (the call's closing text; one with a newline first: Qwen's)."""
    if not end:
        raise ValueError("an empty closing text")
    key = (lark, end)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    if len(lark) > MAX_GRAMMAR:
        return Ending(f"a grammar of more than {MAX_GRAMMAR} characters")
    try:
        out = _ending(lark, end)
    except Unsupported as e:
        out = Ending(str(e)[:200])
    except Exception as e:  # noqa: BLE001 (the analysis failing: the input free text, which ends)
        out = Ending(f"the grammar, not analysed ({type(e).__name__})")
    with _lock:
        _cache[key] = out
        while len(_cache) > CACHED:
            _cache.popitem(last=False)
    return out


def _ending(lark: str, end: str) -> Ending:
    o = _order(lark)
    if not o.patterns:
        return Ending(None)
    newline = len(end) > 1 and end[0] == "\n" and _newline_free(o)
    walk = _Walk(o)
    co, after = _frontier(o, walk.nfa.chars | set(end))
    pats = sorted({o.patterns[x] for x in co})
    d = _Dfa(pats, end)
    ending_ok = {i for i, p in enumerate(pats) if any(o.patterns[x] == p for x in o.last & co)}
    if _goes_on(d, end, newline, ending_ok):
        for x in sorted(co):  # (the lexeme that does it alone, else the union's)
            if _goes_on(_Dfa([o.patterns[x]], end), end, newline):
                return Ending(label(x), newline)
        return Ending(" | ".join(sorted({label(x) for x in co}))[:200], newline)
    if newline:  # (a lexeme that takes the closing newline ends the input there: one that may end it)
        for x in sorted(co - o.last):
            if _takes_newline(_Dfa([o.patterns[x]], end)):
                return Ending(label(x), newline)
    # (the closing text's lexemes: Qwen's newline and its tag, which the model may write without the newline where
    # it is optional; the others' one literal)
    parts = [(end, {0}), (end[1:], set())] if end[0] == "\n" and len(end) > 1 else [(end, set())]
    for text, ends in parts[:1 + newline]:
        if walk.spells(text, after, ends):
            return Ending("the lexemes after the input: " + " | ".join(sorted({label(x) for x in after}))[:160],
                          newline)
    return Ending(None, newline)


def _frontier(o: _Order, chars: set[str]) -> tuple[set[str], set[str]]:
    """The lexemes allowed where the input's last one starts, and the ones that may come after the input, over the
    parser's configurations (llguidance's parser keeps every reading of the input): the sets of positions that may
    have taken the last lexeme, the positions allowed there whose lexemes can match one same text taken together
    (``chars``: the lexemes' own characters, for their DFAs); an ignored lexeme keeps the configuration."""
    real_last = o.last - o.ignored
    dfas: dict[str, _Dfa] = {}
    met: dict[tuple[str, str], bool] = {}

    def meets(x: str, y: str) -> bool:
        a, b = sorted((o.patterns[x], o.patterns[y]))
        if a == b:
            return True
        if (a, b) not in met:
            da = dfas.setdefault(a, _Dfa([a], "".join(sorted(chars))))
            db = dfas.setdefault(b, _Dfa([b], "".join(sorted(chars))))
            met[a, b] = da.meets(db)
        return met[a, b]

    start: frozenset[str] = frozenset({""})
    seen, todo = {start}, [start]
    co: set[str] = set()
    after: set[str] = set()
    while todo:
        c = todo.pop()
        allowed = o.first if c == start else frozenset().union(*(o.follow[q] for q in c)) | o.ignored
        ends = (o.empty if c == start else bool(c & real_last))
        if ends:
            after |= allowed
        for x in allowed:
            nxt = c if x in o.ignored else frozenset(y for y in allowed if y not in o.ignored and meets(x, y))
            if (o.empty if nxt == start else bool(nxt & real_last)):
                co |= allowed
            if nxt not in seen:
                if len(seen) >= MAX_STATES:
                    raise Unsupported(f"more than {MAX_STATES} parser configurations")
                seen.add(nxt)
                todo.append(nxt)
    return co, after


def _takes_newline(d: "_Dfa") -> bool:
    """Whether a lexeme of ``d`` takes a newline after a complete match."""
    return any(d.accepting(s) and d.alive(d.step(s, "\n")) for s in d.after)


def _newline_free(o: _Order) -> bool:
    """Whether a trailing newline taken off the input leaves it the grammar's: no lexeme that may end the input (but
    an ignored one) can end with a newline, and an ignored one without its last newline is still one, or none."""
    used = sorted({o.patterns[x] for x in o.last - o.ignored})
    if used:
        d = _Dfa(used, "\n")
        if any(d.accepting(d.step(s, "\n")) for s in d.graph):
            return False
    if o.ignored:
        d = _Dfa(sorted({o.patterns[x] for x in o.ignored}), "\n")
        if any(d.accepting(d.step(s, "\n")) and not d.accepting(s) and s != d.first for s in d.graph):
            return False
    return True


def _goes_on(d: "_Dfa", end: str, newline: bool, ending_ok: set[int] | None = None) -> bool:
    """Whether a lexeme of ``d`` goes on with ``end`` after a complete match (``newline``: its optional first newline
    taken where the lexeme cannot end right after it, ends where it cannot end the input - ``ending_ok``: the
    patterns that may -, or goes on with the next character)."""
    for s in d.after:
        if not d.accepting(s):
            continue
        if newline:
            t = d.step(s, "\n")
            if d.alive(t) and (not d.accepting(t) or d.alive(d.step(t, end[1])) or
                               (ending_ok is not None and not d.which(t) & ending_ok)):
                return True
        elif d.alive(d.step(s, end[0])):
            return True
    return False


class _Walk:
    """The lexemes of a grammar as one NFA, each with its own start and end, walked over a text in the rules' order."""

    def __init__(self, o: _Order):
        self.o, self.nfa = o, _Nfa()
        self.starts: dict[str, int] = {}
        self.ends: dict[int, str] = {}
        self.owner: list[str] = []  # (each NFA state's position)
        for x, pattern in o.patterns.items():
            try:
                parsed = P.parse(_rust_braces(pattern), 0)
            except (re.error, OverflowError, RecursionError):
                raise Unsupported(pattern[:80]) from None
            s, e = self.nfa.seq(parsed, parsed.state.flags)
            self.starts[x], self.ends[e] = s, x
            self.owner += [x] * (len(self.nfa.eps) - len(self.owner))

    def closure(self, states: set[int]) -> set[int]:
        out, stack = set(states), list(states)
        while stack:
            for t in self.nfa.eps[stack.pop()]:
                if t not in out:
                    out.add(t)
                    stack.append(t)
        return out

    def done(self, states: set[int]) -> set[str]:
        return {self.ends[q] for q in states if q in self.ends}

    def step(self, states: set[int], ch: str) -> set[int]:
        return self.closure({b for q in states for pred, b in self.nfa.edges[q] if pred(ch)})

    def spells(self, text: str, allowed: set[str], ends: set[int]) -> bool:
        """Whether the lexemes, from ``allowed`` and one after another in the rules' order, spell two characters of
        the closing ``text`` or more (one ending inside it and the next going on with it, or one ending after two
        characters), are under way where one of its lexemes ends (``ends``: the indices of their last characters;
        llguidance keeps to the longer lexeme), or go on past all of it."""
        states = self.closure({self.starts[x] for x in allowed})
        for k, ch in enumerate(text):
            done = self.done(states) if k else set()
            if done:  # (a lexeme ends here, a next one starts: where it goes on with the closing text, llguidance
                # follows the closing text's longer lexeme, or the input's, and drops the other)
                nxt = self.closure({self.starts[y] for x in done for y in self.o.follow[x]})
                if self.step(nxt, ch):
                    return True
                states |= nxt
            states = self.step(states, ch)
            if not states:
                return False
            if k >= 1 and self.done(states):
                return True
            if k in ends and {self.owner[q] for q in states} - self.done(states):
                return True  # (a lexeme of the closing text ends while a longer one of the input's is under way)
        return True


# ----------------------------------------------------------------------
# Lark: the rules' order and the lexemes as Python regular expressions
# ----------------------------------------------------------------------


def _tokens(text: str) -> Iterator[tuple[str, str]]:
    for m in _TOKEN.finditer(text):
        if m.lastgroup not in ("ws", "c"):
            yield m.lastgroup, m.group()


def _order(lark: str) -> _Order:
    """A grammar's lexemes in its rules' order (``_Order``)."""
    terminals: dict[str, tuple[str, str]] = {}  # (name -> ("lark", its expression) or ("re", a Python regex))
    bodies: dict[str, str] = {}  # (a rule's name -> its expressions' text)
    ignored: list[tuple[str, str]] = []
    name, kind = None, ""
    for line in lark.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("%import"):
            m = re.fullmatch(r"%import\s+common\s*(?:\.\s*(\w+)(?:\s*->\s*(\w+))?|\(([^)]*)\))\s*(?://.*)?", s)
            if m is None:
                raise Unsupported(s)
            pairs = [(m.group(1), m.group(2) or m.group(1))] if m.group(1) else \
                [(n.strip(), n.strip()) for n in m.group(3).split(",") if n.strip()]
            for src, alias in pairs:
                if src not in COMMON:
                    raise Unsupported(f"common.{src}")
                terminals[alias] = ("re", COMMON[src])
            name = None
            continue
        if s.startswith("%"):
            if s.startswith("%ignore"):
                ignored += [t for t in _tokens(s[len("%ignore"):]) if t[0] in ("s", "r", "n")]
            elif not re.fullmatch(r"%llguidance\s*\{\s*\}\s*(?://.*)?", s):  # (options may change the lexing)
                raise Unsupported(s[:80])
            name = None
            continue
        head = _HEAD.match(s)
        if head is not None:
            name, s = head.group(1), s[head.end():]
            kind = "terminal" if name.lstrip("_")[:1].isupper() else "rule"
            if kind == "terminal":
                terminals[name] = ("lark", "")
            elif name in bodies:
                raise Unsupported(f"{name} defined twice")
            else:
                bodies[name] = ""
        if name is None:
            raise Unsupported(s[:80])
        if kind == "terminal":
            terminals[name] = ("lark", (terminals[name][1] + "\n" + s).strip())  # (a line's comment ends there)
        else:
            bodies[name] += "\n" + s
    if "start" not in bodies:
        raise Unsupported("no start rule")
    patterns: dict[str, str] = {}  # (a position -> its lexeme's pattern)
    regexes: dict[str, str] = {}  # (a lexeme -> its pattern)
    done: dict[str, str] = {}

    def lexeme(k: str, text: str) -> str:
        """A new position of a lexeme (each place where a rule names it, with a separate FOLLOW: Glushkov's construction)."""
        if text not in regexes:
            regexes[text] = _regex(text, terminals, done) if k == "n" else _literal(text) if k == "r" else \
                _string(text)
        if len(patterns) >= MAX_LEXEMES:
            raise Unsupported(f"more than {MAX_LEXEMES} lexemes")
        pos = f"{text}\x00{len(patterns)}"
        patterns[pos] = regexes[text]
        return pos

    rules = {r: _expression(list(_tokens(body)), r, bodies, terminals, lexeme) for r, body in bodies.items()}
    empty_ok = {x: _nullable(p) for x, p in patterns.items()}  # (a lexeme that matches nothing too)
    props: dict[str, tuple[bool, frozenset[str], frozenset[str], frozenset[tuple[str, str]]]] = {
        r: (False, frozenset(), frozenset(), frozenset()) for r in rules}
    changed, rounds = True, 0
    while changed:
        changed, rounds = False, rounds + 1
        if rounds > MAX_ROUNDS:
            raise Unsupported(f"rules that take more than {MAX_ROUNDS} rounds")
        for r, node in rules.items():
            new = _props(node, props, empty_ok)
            if new != props[r]:
                props[r], changed = new, True
    empty, first, last, pairs = props["start"]
    for x, e in empty_ok.items():  # (one that may follow itself, or ignored lexemes beside it: llguidance's own)
        if e and ((x, x) in pairs or ignored):
            raise Unsupported(f"{label(x)}: a lexeme that matches the empty text, repeated or beside %ignore")
    ign = frozenset(lexeme(k, t) for k, t in ignored if k != "n" or t in terminals)
    used = {x for x in first | last} | {x for p in pairs for x in p}
    patterns = {x: patterns[x] for x in used | ign}
    follow = {x: frozenset({b for a, b in pairs if a == x}) | ign for x in used}
    follow.update({x: frozenset(used) | ign for x in ign})
    return _Order(patterns, first | ign, last | ign, follow, ign, empty)


def _props(node: Node, props: dict, empty_ok: dict[str, bool]
           ) -> tuple[bool, frozenset[str], frozenset[str], frozenset[tuple[str, str]]]:
    """An expression's (may be empty, its first lexemes, its last ones, the pairs of lexemes one after the other)."""
    kind = node[0]
    if kind == "sym":
        return empty_ok[node[1]], frozenset({node[1]}), frozenset({node[1]}), frozenset()
    if kind == "ref":
        return props[node[1]]
    if kind == "empty":
        return True, frozenset(), frozenset(), frozenset()
    if kind == "alt":
        parts = [_props(x, props, empty_ok) for x in node[1]]
        return (any(p[0] for p in parts), frozenset().union(*(p[1] for p in parts)),
                frozenset().union(*(p[2] for p in parts)), _pairs(*(p[3] for p in parts)))
    if kind == "seq":
        empty, first, last, pairs = True, set(), set(), frozenset()
        for x in node[1]:
            e, f, la, p = _props(x, props, empty_ok)
            pairs = _pairs(pairs, p, _cross(last, f))
            if empty:
                first |= f
            last = set(la) | (last if e else set())
            empty = empty and e
        return empty, frozenset(first), frozenset(last), pairs
    e, f, la, p = _props(node[1], props, empty_ok)
    if kind == "opt":
        return True, f, la, p
    lo, hi = node[2], node[3]  # ("rep")
    return lo == 0 or e, f, la, _pairs(p, _cross(la, f) if hi is None or hi > 1 else frozenset())


def _cross(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> frozenset[tuple[str, str]]:
    if len(a) * len(b) > MAX_PAIRS:
        raise Unsupported(f"more than {MAX_PAIRS} pairs of lexemes")
    return frozenset((x, y) for x in a for y in b)


def _pairs(*sets: frozenset[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    out = frozenset().union(*sets)
    if len(out) > MAX_PAIRS:
        raise Unsupported(f"more than {MAX_PAIRS} pairs of lexemes")
    return out


def _nullable(pattern: str) -> bool:
    """Whether a lexeme's regular expression matches the empty text (its NFA's start reaches its end)."""
    nfa = _Nfa()
    try:
        parsed = P.parse(_rust_braces(pattern), 0)
    except (re.error, OverflowError, RecursionError):
        raise Unsupported(pattern[:80]) from None
    s, e = nfa.seq(parsed, parsed.state.flags)
    seen, stack = {s}, [s]
    while stack:
        for t in nfa.eps[stack.pop()]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return e in seen


def _expression(toks: list[tuple[str, str]], rule: str, bodies: dict[str, str],
                terminals: dict[str, tuple[str, str]], lexeme: Callable[[str, str], str]) -> Node:
    """A rule's expressions (its alternatives) as a ``Node``."""
    pos = 0

    def peek() -> tuple[str, str]:
        return toks[pos] if pos < len(toks) else ("", "")

    def take() -> tuple[str, str]:
        nonlocal pos
        if pos >= len(toks):
            raise Unsupported(f"{rule}: an expression cut short")
        pos += 1
        return toks[pos - 1]

    def alternatives() -> Node:
        out = [sequence()]
        while peek() == ("op", "|"):
            take()
            out.append(sequence())
        return out[0] if len(out) == 1 else ("alt", out)

    def sequence() -> Node:
        items = []
        while peek()[0] and peek() not in (("op", "|"), ("op", ")"), ("op", "]")):
            if peek() == ("x", "-"):  # (an alias: -> name)
                take()
                if take() != ("x", ">") or take()[0] != "n":
                    raise Unsupported(f"{rule}: an alias")
                continue
            items.append(item())
        return ("seq", items) if items else ("empty",)

    def item() -> Node:
        a = atom()
        while peek() in (("op", "*"), ("op", "+"), ("op", "?"), ("op", "~")):
            op = take()[1]
            if op == "?":
                a = ("opt", a)
            elif op in "*+":
                a = ("rep", a, 0 if op == "*" else 1, None)
            else:
                lo = take()
                hi = lo
                if peek() == ("op", ".."):
                    take()
                    hi = take()
                if lo[0] != "num" or hi[0] != "num":
                    raise Unsupported(f"{rule}: a repetition")
                a = ("rep", a, int(lo[1]), int(hi[1]))
        return a

    def atom() -> Node:
        kind, text = take()
        if kind == "op" and text in "([":
            inner = alternatives()
            if take() != ("op", ")" if text == "(" else "]"):
                raise Unsupported(f"{rule}: an unbalanced group")
            return inner if text == "(" else ("opt", inner)
        if kind in ("s", "r"):
            return ("sym", lexeme(kind, text))
        if kind == "n" and text in terminals:
            return ("sym", lexeme(kind, text))
        if kind == "n" and text in bodies:
            return ("ref", text)
        if (kind, text) == ("x", "<"):  # (a special token: no lexeme)
            while take() != ("x", ">"):
                pass
            return ("empty",)
        if (kind, text) == ("x", "%") and peek() == ("n", "json"):  # (a JSON value: its lexemes cannot go on)
            take()
            depth = 0
            while True:
                k, t = take()
                depth += (k, t) == ("x", "{")
                depth -= (k, t) == ("x", "}")
                if depth == 0:
                    return ("empty",)
        raise Unsupported(f"{rule}: {text}")

    out = alternatives()
    if pos != len(toks):
        raise Unsupported(f"{rule}: {toks[pos][1]}")
    return out


def _literal(token: str) -> str:
    """A Lark ``/.../flags`` as a Python regular expression."""
    flags = "".join(f for f in token[token.rindex("/") + 1:] if f in "imsx")
    core = token[1:token.rindex("/")].replace("\\/", "/")
    return f"(?{flags}:{core})" if flags else f"(?:{core})"


def _string(token: str) -> str:
    """A Lark ``"..."`` (``i``: either case) as a Python regular expression."""
    value = _value(token)
    return f"(?i:{re.escape(value)})" if token.endswith("i") else re.escape(value)


def _value(token: str) -> str:
    try:
        value = ast.literal_eval(token.removesuffix("i"))
    except (SyntaxError, ValueError):
        raise Unsupported(token) from None
    if not isinstance(value, str):
        raise Unsupported(token)
    return value


def _regex(name: str, terminals: dict[str, tuple[str, str]], done: dict[str, str],
           seen: frozenset[str] = frozenset()) -> str:
    """A terminal's expression as a Python regular expression (its references to other terminals inlined)."""
    if name in done:
        return done[name]
    if name in seen:
        raise Unsupported(f"{name} (recursive)")
    kind, body = terminals[name]
    if kind == "re":
        done[name] = body
        return body
    toks = list(_tokens(body))
    pos = 0

    def peek() -> tuple[str, str]:
        return toks[pos] if pos < len(toks) else ("", "")

    def take() -> tuple[str, str]:
        nonlocal pos
        if pos >= len(toks):
            raise Unsupported(f"{name}: {body[:80]}")
        pos += 1
        return toks[pos - 1]

    def alternatives() -> str:
        out = [sequence()]
        while peek() == ("op", "|"):
            take()
            out.append(sequence())
        return out[0] if len(out) == 1 else "(?:" + "|".join(out) + ")"

    def sequence() -> str:
        out = ""
        while peek()[0] and peek() not in (("op", "|"), ("op", ")"), ("op", "]")):
            out += item()
            if len(out) > MAX_PATTERN:
                raise Unsupported(f"{name}: longer than {MAX_PATTERN} characters")
        return out

    def item() -> str:
        a = atom()
        while peek() in (("op", "*"), ("op", "+"), ("op", "?"), ("op", "~")):
            op = take()[1]
            if op == "~":
                lo = take()
                hi = lo
                if peek() == ("op", ".."):
                    take()
                    hi = take()
                if lo[0] != "num" or hi[0] != "num":
                    raise Unsupported(f"{name}: {body[:80]}")
                a = f"(?:{a}){{{lo[1]},{hi[1]}}}"
            else:
                a = f"(?:{a}){op}"
        return a

    def atom() -> str:
        kind, text = take()
        if kind == "op" and text in "([":
            inner = alternatives()
            if take() != ("op", ")" if text == "(" else "]"):
                raise Unsupported(f"{name}: {body[:80]}")
            return f"(?:{inner})" + ("?" if text == "[" else "")
        if kind == "s":
            if peek() == ("op", ".."):
                take()
                k2, t2 = take()
                lo, hi = _value(text), _value(t2) if k2 == "s" else ""
                if len(lo) != 1 or len(hi) != 1:
                    raise Unsupported(f"{name}: {body[:80]}")
                return f"[{re.escape(lo)}-{re.escape(hi)}]"
            return _string(text)
        if kind == "r":
            return _literal(text)
        if kind == "n" and text in terminals:
            return f"(?:{_regex(text, terminals, done, seen | {name})})"
        raise Unsupported(f"{name}: {text}")

    out = alternatives()
    if pos != len(toks) or len(out) > MAX_PATTERN:
        raise Unsupported(f"{name}: {body[:80]}")
    done[name] = out
    return out


def _rust_braces(pattern: str) -> str:
    """A regular expression as Python's ``re`` reads llguidance's (Rust's) syntax: a repetition's blanks dropped
    (``{ 1,}``: Rust's repetition, Python's literal text). Unsupported: a class that Python reads otherwise, a class in
    a class (``[[:punct:]]``, ``[a[b]]``) or one with ``&&``, ``--`` or ``~~`` (Python: a class, then literals)."""
    out, i, inside = [], 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if not inside and c == "[":
            j = i + 1
            j += pattern.startswith("^", j)
            j += pattern.startswith("]", j)  # (a first "]" is the class's)
            out.append(pattern[i:j])
            inside, i = True, j
            continue
        if inside and (c == "[" or pattern[i:i + 2] in ("&&", "--", "~~")):
            raise Unsupported(f"a class of Rust's regex syntax: {pattern[:80]}")
        if inside and c == "]":
            inside = False
        if not inside and c == "{" and (m := re.match(r"\{([\s\d,]*\d[\s\d,]*)\}", pattern[i:])):
            out.append("{" + re.sub(r"\s+", "", m.group(1)) + "}")
            i += m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ----------------------------------------------------------------------
# The automaton
# ----------------------------------------------------------------------


Pred = Callable[[str], bool]


class _Nfa:
    def __init__(self) -> None:
        self.eps: list[list[int]] = []
        self.edges: list[list[tuple[Pred, int]]] = []
        self.chars: set[str] = set()  # (the expressions' own characters: their classes' bounds)

    def state(self) -> int:
        if len(self.eps) >= MAX_NFA:
            raise Unsupported(f"more than {MAX_NFA} NFA states")
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def seq(self, items: object, flags: int) -> tuple[int, int]:
        s = e = self.state()
        for op, av in items:
            a, b = self.node(op, av, flags)
            self.eps[e].append(a)
            e = b
        return s, e

    def node(self, op: object, av: object, flags: int) -> tuple[int, int]:
        if op in (K.LITERAL, K.NOT_LITERAL, K.ANY, K.IN):
            a, b = self.state(), self.state()
            self.edges[a].append((self.pred(op, av, flags), b))
            return a, b
        if op is K.BRANCH:
            a, b = self.state(), self.state()
            for alt in av[1]:
                s, e = self.seq(alt, flags)
                self.eps[a].append(s)
                self.eps[e].append(b)
            return a, b
        if op is K.SUBPATTERN:
            _group, add, remove, p = av
            return self.seq(p, (flags | add) & ~remove)
        if op is K.ATOMIC_GROUP:
            return self.seq(av, flags)
        if op in (K.MAX_REPEAT, K.MIN_REPEAT, K.POSSESSIVE_REPEAT):
            lo, hi, p = av
            s = e = self.state()
            for _ in range(min(lo, MAX_REPEAT)):
                a, b = self.seq(p, flags)
                self.eps[e].append(a)
                e = b
            if hi is K.MAXREPEAT or hi > MAX_REPEAT or lo > MAX_REPEAT:
                a, b = self.seq(p, flags)
                self.eps[e].append(a)
                self.eps[b].append(e)
                return s, e
            end = self.state()
            self.eps[e].append(end)
            for _ in range(hi - lo):
                a, b = self.seq(p, flags)
                self.eps[e].append(a)
                self.eps[b].append(end)
                e = b
            return s, end
        if op in (K.AT, K.ASSERT, K.ASSERT_NOT):  # (anchors and lookarounds: dropped, a larger language)
            a = self.state()
            return a, a
        raise Unsupported(str(op))

    def pred(self, op: object, av: object, flags: int) -> Pred:
        fold = bool(flags & re.IGNORECASE)
        if op is K.LITERAL or op is K.NOT_LITERAL:
            c = chr(av)
            self.chars.update((c, c.lower(), c.upper()))
            same = (lambda x: x.lower() == c.lower()) if fold else (lambda x: x == c)
            return same if op is K.LITERAL else (lambda x: not same(x))
        if op is K.ANY:
            return (lambda x: True) if flags & re.DOTALL else (lambda x: x != "\n")
        negate, tests = False, []
        for kind, value in av:
            if kind is K.NEGATE:
                negate = True
            elif kind is K.LITERAL:
                tests.append(self.pred(K.LITERAL, value, flags))
            elif kind is K.RANGE:
                lo, hi = chr(value[0]), chr(value[1])
                self.chars.update((chr(max(value[0] - 1, 0)), lo, hi, chr(min(value[1] + 1, 0x10FFFF))))
                tests.append((lambda x, lo=lo, hi=hi: lo <= x <= hi or (fold and (lo <= x.lower() <= hi or
                                                                                  lo <= x.upper() <= hi))))
            elif kind is K.CATEGORY:
                rx = re.compile({K.CATEGORY_DIGIT: r"\d", K.CATEGORY_NOT_DIGIT: r"\D", K.CATEGORY_SPACE: r"\s",
                                 K.CATEGORY_NOT_SPACE: r"\S", K.CATEGORY_WORD: r"\w",
                                 K.CATEGORY_NOT_WORD: r"\W"}[value])
                tests.append(lambda x, rx=rx: rx.fullmatch(x) is not None)
            else:
                raise Unsupported(str(kind))
        return lambda x: any(t(x) for t in tests) != negate


class _Dfa:
    """The union of regular expressions as a DFA over sample characters (``SAMPLES``, the expressions' own and
    ``chars``): its states reached by a character (``after``), the accepting ones (``which`` patterns), the ones an
    accepting one is reached from (``alive``)."""

    def __init__(self, patterns: list[str], chars: str = ""):
        nfa = _Nfa()
        start, self.accept = nfa.state(), nfa.state()
        self.ends: dict[int, int] = {}  # (each pattern's end state -> its index)
        for i, p in enumerate(patterns):
            try:
                parsed = P.parse(_rust_braces(p), 0)
            except (re.error, OverflowError, RecursionError):
                raise Unsupported(p[:80]) from None
            s, e = nfa.seq(parsed, parsed.state.flags)
            nfa.eps[start].append(s)
            nfa.eps[e].append(self.accept)
            self.ends[e] = i
        alphabet = sorted(set(SAMPLES) | nfa.chars | set(chars))

        def closure(states: set[int]) -> frozenset[int]:
            out, stack = set(states), list(states)
            while stack:
                for t in nfa.eps[stack.pop()]:
                    if t not in out:
                        out.add(t)
                        stack.append(t)
            return frozenset(out)

        self.first = closure({start})
        self.graph: dict[frozenset[int], dict[str, frozenset[int]]] = {}
        self.after: set[frozenset[int]] = set()
        todo, work = [self.first], 0
        while todo:
            d = todo.pop()
            if d in self.graph:
                continue
            work += len(alphabet) * sum(len(nfa.edges[q]) for q in d)
            if len(self.graph) >= MAX_STATES or work > MAX_WORK:
                raise Unsupported(f"an automaton of more than {MAX_STATES} states or {MAX_WORK} transitions")
            self.graph[d] = {}
            for ch in alphabet:
                t = closure({b for q in d for pred, b in nfa.edges[q] if pred(ch)})
                if t:
                    self.graph[d][ch] = t
                    self.after.add(t)
                    todo.append(t)
        self.live = {d for d in self.graph if self.accept in d}
        grew = True
        while grew:
            grew = False
            for d, nxt in self.graph.items():
                if d not in self.live and any(t in self.live for t in nxt.values()):
                    self.live.add(d)
                    grew = True

    def step(self, s: frozenset[int] | None, ch: str) -> frozenset[int] | None:
        return None if s is None else self.graph[s].get(ch)

    def meets(self, other: "_Dfa") -> bool:
        """Whether some text is in both languages (the DFAs over the same characters)."""
        seen, stack = {(self.first, other.first)}, [(self.first, other.first)]
        while stack:
            a, b = stack.pop()
            if self.accepting(a) and other.accepting(b):
                return True
            for ch, t in self.graph[a].items():
                u = other.graph[b].get(ch)
                if u is not None and (t, u) not in seen:
                    if len(seen) >= MAX_STATES:
                        return True  # (too many to tell: they may)
                    seen.add((t, u))
                    stack.append((t, u))
        return False

    def accepting(self, s: frozenset[int] | None) -> bool:
        return s is not None and self.accept in s

    def which(self, s: frozenset[int] | None) -> set[int]:
        """The patterns a state is a complete match of (their indices)."""
        return set() if s is None else {self.ends[q] for q in s if q in self.ends}

    def alive(self, s: frozenset[int] | None) -> bool:
        return s is not None and s in self.live
