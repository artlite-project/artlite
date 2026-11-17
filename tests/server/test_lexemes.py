"""Whether a custom tool's Lark grammar can end inside a call (``models.server.lexemes``), against llguidance's own
lexing, for each closing text of the syntaxes: a grammar the analysis clears ends in the call (the closing text, with
Qwen's newline optional or not as the analysis says, after each valid input is taken); one it flags, after an input
that ends where its lexemes go on, does not as a sub-grammar - or the flag is conservative (llguidance's lazy ``*?``,
lexemes that spell the closing text but llguidance tells apart). Grammars it does not translate, or over its limits,
count as going on, fast. Random grammars, their inputs sampled under llguidance's own masks, check the analysis
the same way."""

import json
import random
import time

import llguidance
import numpy as np
import pytest

from models.server.lexemes import ending

from .test_call_grammar import TOY, V, takes

APPLY_PATCH = """start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: add_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
update_hunk: "*** Update File: " filename LF change?
filename: /(.+)/
add_line: "+" /(.*)/ LF -> line
change: (change_context | change_line)+
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
%import common.LF
"""  # (a patch grammar in the shape of Codex's apply_patch)

QWEN, GLM, GEMMA = "\n</parameter>", "</arg_value>", '<|"|>'
C, S, F = "clear", "stuck", "flag"  # (cleared; flagged, the witness cannot close; flagged, conservatively)
# name: (grammar, inputs it takes, {closing text: C, S or F}, Qwen's newline optional, the witness)
CASES = {
    "patch": (APPLY_PATCH, ["*** Begin Patch\n*** Add File: a\n+x\n*** End Patch\n",
                            "*** Begin Patch\n*** Update File: a\n@@ f\n-1\n+2\n*** End Patch"],
              {QWEN: C, GLM: C, GEMMA: C}, False, None),
    "catch-all": ("start: SOURCE\nSOURCE: /[\\s\\S]+/\n", ["print(1)\nprint(2)"], {QWEN: S, GLM: S, GEMMA: S}, None,
                  "print(1)"),
    "prefixed": ("start: A\nA: /@[\\s\\S]*/\n", ["@x"], {QWEN: S, GLM: S, GEMMA: S}, None, "@x"),
    "composed": ("start: SOURCE\nSOURCE: CHAR+\nCHAR: /[\\s\\S]/\n", ["ab"], {QWEN: S, GLM: S, GEMMA: S}, None, "ab"),
    "lines": ("start: L\nL: /[a-z]+(\\n[a-z]+)*/\n", ["abc\ndef"], {QWEN: S, GLM: C, GEMMA: C}, None, "abc"),
    "ignored blanks": ('start: "SELECT" NAME ("," NAME)*\nNAME: /[a-z]+/\n%import common.WS\n%ignore WS\n',
                       ["SELECT a, b", "SELECT a,\n b\n", "SELECT a \n"], {QWEN: C, GLM: C, GEMMA: C}, True, None),
    "trailing newline": ("start: (L LF)+\nL: /[a-z]+/\n%import common.LF\n", ["ab\n", "ab\ncd\n"],
                         {QWEN: C, GLM: C, GEMMA: C}, False, None),
    "tag inside": ("start: A\nA: /[a-z]+(<[a-z]+>)?/\n", ["abc", "abc<b>"], {QWEN: C, GLM: S, GEMMA: S}, True, "abc"),
    "numbers": ('start: NUM (OP NUM)*\nNUM: /[0-9]+/\nOP: "+" | "-"\n', ["12+3"], {QWEN: C, GLM: C, GEMMA: C}, True,
                None),
    "literals": ('start: "a" | "a<b"\n', ["a", "a<b"], {QWEN: C, GLM: S, GEMMA: S}, True, "a"),
    "terminals": ('start: X | Y\nX: /[a-z]/\nY: /[a-z]/ "<b"\n', ["a"], {QWEN: C, GLM: S, GEMMA: S}, True, "a"),
    "newline literal": ('start: "a" | "a\\n<b"\n', ["a"], {QWEN: S, GLM: C, GEMMA: C}, None, "a"),
    "closing inside": ('start: "a"? | "\\n</parameter>x"\n', ["a"], {QWEN: S, GLM: C, GEMMA: C}, None, ""),
    "spaced repeat": ("start: A\nA: /[\\s\\S]{ 1,}/\n", ["ab"], {QWEN: S, GLM: S, GEMMA: S}, None, "ab"),
    "comment in a terminal": ('start: A\nA: "a" // c\n | "a<b"\n', ["a"], {QWEN: C, GLM: S, GEMMA: S}, True, "a"),
    "closing prefix": ('start: "</"? "a"*\n', ["</a", "a"], {QWEN: F, GLM: S, GEMMA: C}, None, "</a"),
    "newline prefix": ('start: "b"* "\\n<"?\n', ["b"], {QWEN: S, GLM: C, GEMMA: C}, None, "b"),
    "spelled prefix": ('start: ("<" | "|" | "a")*\n', ["<|a", "a"], {QWEN: C, GLM: C, GEMMA: S}, True, "<|a"),
    "lone newline": ('start: ("\\n" | "<" | "a")*\n', ["a\n<a", "a"], {QWEN: F, GLM: C, GEMMA: C}, False, None),
    "first character": ('start: "<"? /[a-z]+/\n', ["<ab", "ab"], {QWEN: C, GLM: C, GEMMA: C}, True, None),
    "tags": ('start: TAG*\nTAG: "<" /[a-z]+/ ">" | "</" /[a-z]+/ ">"\n', ["<a></a>"], {QWEN: F, GLM: C, GEMMA: C},
             None, None),
    "comment": ("start: C_COMMENT\n%import common.C_COMMENT\n", ["/* a */"], {QWEN: F, GLM: F, GEMMA: F}, None, None),
    "newline then tag": ('start: "a" ("\\n<b>")?\n', ["a", "a\n<b>"], {QWEN: S, GLM: C, GEMMA: C}, None, "a"),
    "newline then two": ('start: "a" ("\\n<<")?\n', ["a"], {QWEN: S, GLM: C, GEMMA: C}, None, "a"),
    "nullable tail": ("start: BODY TAIL\nBODY: /[^>]+/\nTAIL: /[ \\t]*/\n", ["abc"], {QWEN: S, GLM: S, GEMMA: S},
                      None, "abc"),
    "newline, then more": ('start: /[ab\\n]*/ "p" | /a+/\n', ["a", "ab\np"], {QWEN: S, GLM: C, GEMMA: C}, None, "a"),
}


def closing(end: str, newline: bool) -> str:
    """The call's text after a sub-grammar's input, as the call grammar writes it (Qwen's newline optional or not)."""
    if end == QWEN:
        return '"\\n"? "</parameter>"' if newline else '"\\n" "</parameter>"'
    return json.dumps(end)


def grammar_of(sub: str, end: str, newline: bool) -> str:
    main = f"%llguidance {{}}\nstart: \"<p>\" @s {closing(end, newline)}\n"
    return json.dumps({"grammars": [{"lark_grammar": main}, {"name": "s", "lark_grammar": sub}]})


@pytest.mark.parametrize("name", CASES)
def test_ending(name: str) -> None:
    """The analysis, and llguidance agrees: cleared, every valid input then the closing text is taken; flagged, the
    closing text after an input that ends where the lexemes go on is not."""
    grammar, inputs, want, newline, witness = CASES[name]
    for end, w in want.items():
        e = ending(grammar, end)
        assert (e.over is not None) == (w != C), (name, end, e)
        if end == QWEN and newline is not None:
            assert e.newline == newline, (name, e)
        g = grammar_of(grammar, end, e.newline)
        if w == C:
            assert all(takes(g, TOY.encode("<p>" + text + end)) for text in inputs), (name, end)
        elif w == S:
            assert not takes(g, TOY.encode("<p>" + witness + end)), (name, end)


@pytest.mark.parametrize("grammar", [
    "start: (" + " | ".join(f'"w{i}"' for i in range(500)) + ")*\n", 'start: A\nA: ("a"\n', 'start: A\nA: "a" ~\n', 'start: A\nA: ' + "(" * 3000 + '"a"' + ")" * 3000 + "\n",
    "start: T0\n" + "".join(f"T{i}: T{i + 1} T{i + 1}\n" for i in range(40)) + 'T40: "a"\n',
    "start: A\nA: /" + "a?" * 5000 + "/\n", "start: A\nA: /[a-z]+/ & /[a-c]+/\n",
    "start: X\n%import common.NOPE\n", "start: A\nA: /(a)\\1/\n", 'start: "x"\n' + "y" * 70000,
    "start: A\nA: /[[:punct:]]+/\n", "start: A\nA: /[\\x20-\\x7f&&[^a]]+/\n", "start: A\nA: /\\p{L}+/\n",
    'start: %regex {"substring_words": "a <b c"}\n', '%llguidance {"ignore_once": true}\nstart: "a"\n',
], ids=["many-alternatives", "unbalanced", "repeat-of-nothing", "deep", "doubling", "optionals", "intersection", "unknown-import",
        "backreference", "long", "posix-class", "class-intersection", "unicode-class", "substring", "options"])
def test_not_analysed(grammar: str) -> None:
    """What does not translate, or is over the limits, counts as going on (the input free text), in well under a
    second."""
    t0 = time.perf_counter()
    assert ending(grammar, QWEN).over is not None
    assert time.perf_counter() - t0 < 1.0


def test_large_repeats() -> None:
    """Bounded repeats beyond ``MAX_REPEAT`` analysed as unbounded: nested ones in no time, and cleared where the
    unbounded ones are."""
    t0 = time.perf_counter()
    assert ending("start: A\nA: /(((a{0,32}){0,32}){0,32}){0,32}/\n", QWEN).over is None
    assert ending("start: A\nA: /(?:a|b){0,32}/ /(?:a|b){0,32}/\n", GLM).over is None
    assert time.perf_counter() - t0 < 1.0


def test_harmless() -> None:
    """Comments, a regex with an escaped slash, a range, a class with "]" first or an escaped "[", a rule's alias and
    attributes, a special token and %json are read as such."""
    assert ending('start: "a" /x\\/y/ // a /[\\s\\S]+/ comment\nB: "a".."z"\nr: B -> s\n', GEMMA).over is None
    assert ending("start: A\nA: /[]a\\[-]+/\n", GEMMA).over is None
    assert ending('%llguidance {}\nstart: a %json {"type": "integer"}\na[lazy]: /[a-z]+/\n', GEMMA).over is None
    assert ending('start: "x" <[7]>\n', GLM).over is None


CHARS = ["a", "b", "<", "/", "\n", "p", ">", "|", '"', " ", "r", "g", "_", "v", "l", "u", "e"]
REGEXES = ["/[a-z]+/", "/[ab\\n]*/", "/[^<]+/", "/a?/", "/[\\s\\S]/", "/\\n?<[a-z]/", "/[a<]*/", "/[ \\t]*/",
           "/[^>\\n]+/", "/<[a-z]+>/", "/\\n/", "/[a-z\\n]+/", "/<\\/?[a-z_]+>?/", "/[a-z_]+/", "/<\\|[^|]*\\|>/"]


def random_grammar(rng: random.Random) -> str:
    """A small Lark grammar over the closing texts' characters: literals, regexes, terminals, groups, repeats."""

    def lit() -> str:
        return json.dumps("".join(rng.choice(CHARS) for _ in range(rng.randint(1, 3))))

    def atom(depth: int, names: list[str]) -> str:
        r = rng.random()
        if r < 0.4:
            return lit()
        if r < 0.6:
            return rng.choice(REGEXES)
        if r < 0.75 and names:
            return rng.choice(names)
        return f"({alts(depth + 1, names)})" if depth < 2 else lit()

    def alts(depth: int, names: list[str]) -> str:
        return " | ".join(" ".join(atom(depth, names) + rng.choice(["", "", "", "?", "*", "+"])
                                   for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 2)))

    terms = [f"T{i}" for i in range(rng.randint(0, 2))]
    lines = [f"start: {alts(0, terms)}"] + [f"{t}: {rng.choice(REGEXES) if rng.random() < 0.5 else lit()}"
                                             for t in terms]
    return "\n".join(lines + (["%import common.WS", "%ignore WS"] if rng.random() < 0.2 else [])) + "\n"


def valid_inputs(g: str, rng: random.Random, n: int = 4) -> list[str]:
    """Inputs a grammar takes: random walks over the characters under its own mask, ended where it accepts."""
    pieces = [(c, TOY.encode_plain(c)[0]) for c in CHARS]
    out: set[str] = set()
    for _ in range(3 * n):
        m = llguidance.LLMatcher(V.tokenizer, g, log_level=0)
        if m.is_error():
            return []
        text = ""
        for _ in range(10):
            row = np.zeros((1, V.words), np.int32)
            m.unsafe_compute_mask_ptr(row.ctypes.data, row.nbytes)
            bits = row[0].view(np.uint32)
            ok = [(c, t) for c, t in pieces if bits[t >> 5] >> (t & 31) & 1]
            if not ok or (m.is_accepting() and rng.random() < 0.3):
                break
            c, t = rng.choice(ok)
            m.consume_token(t)
            text += c
        if m.is_accepting():
            out.add(text)
    return sorted(out)[:n]


def test_fuzz() -> None:
    """Random grammars against llguidance: every one the analysis clears ends in the call after each of its sampled
    inputs, for each closing text (many more seeds: the analysis's research runs)."""
    wrong = []
    for seed in range(300):
        rng = random.Random(seed)
        g = random_grammar(rng)
        inputs = valid_inputs(g, rng)
        for end in (QWEN, GLM, GEMMA) if inputs else ():
            e = ending(g, end)
            if e.over is None and not all(takes(grammar_of(g, end, e.newline), TOY.encode("<p>" + x + end))
                                          for x in inputs):
                wrong.append((seed, end, g))
    assert not wrong, wrong[:3]
