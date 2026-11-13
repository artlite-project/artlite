"""The C++ scanner of the developer tools (``measure_ii``, ``variant_search``): tokens without comments and
preprocessor directives, matching brackets, the extents of statements, function definitions and loop statements."""

import bisect
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    column: int  # 1-based, as in DebugLocs


_TOKEN = re.compile(
    r"(?P<skip>\s+|//[^\n]*|/\*.*?\*/)"
    r'|(?P<literal>[uUL8]*R"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)"'
    r'|[uUL8]*"(?:\\.|[^"\\\n])*"|[uUL8]*\'(?:\\.|[^\'\\\n])+\'|\.?\d(?:\'?[\w.]|[eEpP][-+])*)'
    r"|(?P<word>[A-Za-z_]\w*)"
    r"|(?P<punct>.)",
    re.DOTALL,
)
_OPENING = {"(", "[", "{"}
_CLOSING = {")", "]", "}"}
_STATEMENT_KEYWORDS = {"for", "while", "do", "if", "switch", "{"}


def tokenize(text: str) -> list[Token]:
    """C++ tokens without comments and preprocessor directives."""
    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    tokens = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        assert match is not None  # (punct matches any character)
        position = match.end()
        if match.lastgroup == "skip":
            continue
        line = bisect.bisect_right(line_starts, match.start())
        line_start = line_starts[line - 1]
        if match.group() == "#" and not text[line_start : match.start()].strip():
            while True:  # a directive: up to the end of its (continued) line
                newline = text.find("\n", position)
                if newline < 0:
                    position = len(text)
                    break
                position = newline + 1
                end = newline - 1 if newline and text[newline - 1] == "\r" else newline
                if not (end and text[end - 1] == "\\"):
                    break
            continue
        tokens.append(Token(match.group(), line, match.start() - line_start + 1))
    return tokens


def matching_close(tokens: list[Token], index: int) -> int:
    """Index of the bracket closing tokens[index]."""
    depth = 0
    for position in range(index, len(tokens)):
        text = tokens[position].text
        if text in _OPENING:
            depth += 1
        elif text in _CLOSING:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("unbalanced brackets")


def matching_open(tokens: list[Token], index: int) -> int:
    """Index of the bracket opening the one at tokens[index]."""
    depth = 0
    for position in range(index, -1, -1):
        text = tokens[position].text
        if text in _CLOSING:
            depth += 1
        elif text in _OPENING:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("unbalanced brackets")


def _pragma_like(text: str) -> bool:
    """AIE_LOOP_* and _Pragma(...) in front of a statement."""
    return text == "_Pragma" or re.fullmatch(r"[A-Z][A-Z0-9_]*", text) is not None


def statement_end(tokens: list[Token], index: int) -> int:
    """Index of the last token of the statement starting at tokens[index]."""
    text = tokens[index].text
    if text == "{":
        return matching_close(tokens, index)
    if text in {"for", "while", "switch"}:
        return statement_end(tokens, matching_close(tokens, index + 1) + 1)
    if text == "if":
        condition = index + 2 if tokens[index + 1].text == "constexpr" else index + 1
        end = statement_end(tokens, matching_close(tokens, condition) + 1)
        if end + 1 < len(tokens) and tokens[end + 1].text == "else":
            return statement_end(tokens, end + 2)
        return end
    if text == "do":
        body_end = statement_end(tokens, index + 1)  # then: while (...) ;
        return matching_close(tokens, body_end + 2) + 1
    if _pragma_like(text):
        following = index + 1
        if tokens[following].text == "(":
            following = matching_close(tokens, following) + 1
        if tokens[following].text in _STATEMENT_KEYWORDS or _pragma_like(tokens[following].text):
            return statement_end(tokens, following)
    if text == "[" and tokens[index + 1].text == "[":  # [[attributes]]
        return statement_end(tokens, matching_close(tokens, index) + 1)
    depth = 0
    for position in range(index, len(tokens)):
        text = tokens[position].text
        if text in _OPENING:
            depth += 1
        elif text in _CLOSING:
            depth -= 1
            if depth < 0:
                break
        elif text == ";" and depth == 0:
            return position
    raise ValueError("statement without an end")


@dataclass(frozen=True)
class Statement:
    """A loop statement of a source file."""

    line: int  # of its keyword
    column: int
    end_line: int
    exits: bool = False  # a break, return or goto leaves it before its condition does
    scope: tuple[int, int] | None = None  # lines of the function body around it


_CONTROL = {"if", "for", "while", "switch", "catch", "constexpr"}  # (a brace after their condition is no function)
_KEYWORDS = _CONTROL | {"return", "sizeof", "decltype", "alignof", "static_assert"}  # (followed by a parenthesis)
_FUNCTION_QUALIFIERS = {"const", "noexcept", "override", "final", "volatile", "&", "&&"}


def function_bodies(tokens: list[Token]) -> list[tuple[int, int]]:
    """Token ranges of the function (and lambda) bodies: a brace after a parameter list and its qualifiers, not
    after a control statement's condition."""
    bodies = []
    for index, token in enumerate(tokens):
        if token.text != "{" or index == 0:
            continue
        before = index - 1
        while before > 0 and tokens[before].text in _FUNCTION_QUALIFIERS:
            before -= 1
        if tokens[before].text != ")":
            continue
        depth, opening = 0, before
        while opening >= 0:  # the parenthesis it closes
            text = tokens[opening].text
            depth += text == ")"
            depth -= text == "("
            if depth == 0:
                break
            opening -= 1
        if opening <= 0 or tokens[opening - 1].text in _CONTROL:
            continue
        try:
            bodies.append((index, matching_close(tokens, index)))
        except ValueError:
            continue
    return bodies


@dataclass(frozen=True)
class Definition:
    name: str
    start: int  # token index of its first token (template header, attributes, return type)
    body: int  # token index of the body's opening brace
    end: int  # token index of the body's closing brace


def function_definitions(tokens: list[Token], name: str) -> list[Definition]:
    """The function definitions whose name contains name ("": all)."""
    found = []
    for index, token in enumerate(tokens[:-1]):
        if (
            name not in token.text
            or token.text in _KEYWORDS
            or not re.fullmatch(r"[A-Za-z_]\w*", token.text)
            or tokens[index + 1].text != "("
        ):
            continue
        after = matching_close(tokens, index + 1) + 1
        while after < len(tokens) and tokens[after].text in _FUNCTION_QUALIFIERS:
            after += 1
        if after >= len(tokens) or tokens[after].text != "{":
            continue  # a call or a declaration
        start = index
        while start > 0 and tokens[start - 1].text not in (";", "}", "{"):
            start -= 1
        found.append(Definition(token.text, start, after, matching_close(tokens, after)))
    return found


def loop_statements(tokens: list[Token]) -> list[Statement]:
    """The loop statements of a file."""
    spans = []  # token ranges of the loops and switches: a break leaves the innermost one
    closing: set[int] = set()  # the `while` of a do statement
    for index, token in enumerate(tokens):
        if token.text not in {"for", "while", "do", "switch"} or index in closing:
            continue
        try:
            end = statement_end(tokens, index)
            if token.text == "do":
                closing.add(statement_end(tokens, index + 1) + 1)
        except (ValueError, IndexError):
            continue
        spans.append((index, end))
    exits: set[int] = set()
    for index, token in enumerate(tokens):
        if token.text == "break":
            around = [span for span in spans if span[0] < index <= span[1]]
            if around:
                exits.add(max(around)[0])
        elif token.text in {"return", "goto"}:
            exits.update(start for start, end in spans if start < index <= end)
    bodies = function_bodies(tokens)

    def scope(start: int) -> tuple[int, int] | None:
        around = [body for body in bodies if body[0] < start <= body[1]]
        if not around:
            return None
        first, last = max(around)  # (the innermost: a lambda in a function)
        return tokens[first].line, tokens[last].line

    return [
        Statement(tokens[start].line, tokens[start].column, tokens[end].line, start in exits, scope(start))
        for start, end in spans
        if tokens[start].text != "switch"
    ]
