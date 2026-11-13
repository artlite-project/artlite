"""Source patches on the kernel's C++: loop pragmas on the target loops, substitutions, replaced definitions."""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..cxx_scan import Definition, Token, function_definitions, matching_open, statement_end, tokenize

POST = "AIE_PREPARE_FOR_POSTPIPELINING"
_PIPELINING = {"AIE_PREPARE_FOR_POSTPIPELINING", "AIE_TRY_INITIATION_INTERVAL"}


def ii(n: int) -> str:
    return f"AIE_TRY_INITIATION_INTERVAL({n})"


def pragma_text(spec: str) -> str | None:
    """A pragma shorthand: base / keep = as rendered (None), none = no pipelining pragma, post = POST, iiN =
    AIE_TRY_INITIATION_INTERVAL(N), joined by + for several (post+ii44); anything else is the pragma text."""
    if "+" in spec and "(" not in spec:
        return " ".join(filter(None, (pragma_text(part) for part in spec.split("+"))))
    if spec in ("base", "keep"):
        return None
    if spec == "none":
        return ""
    if spec == "post":
        return POST
    match = re.fullmatch(r"ii(\d+)", spec)
    return ii(int(match.group(1))) if match else spec


def _is_loop_macro(text: str) -> bool:
    return text == "_Pragma" or re.fullmatch(r"AIE_[A-Z0-9_]+", text) is not None


@dataclass(frozen=True)
class _Loop:
    keyword: int  # token index of for / while / do
    end: int  # token index of its last token
    prefix: tuple[tuple[int, int], ...]  # token spans of the AIE_* macros in front of it

    def macros(self, tokens: list[Token]) -> list[str]:
        return [tokens[first].text for first, _ in self.prefix]

    def rolled(self, tokens: list[Token]) -> bool:
        return "AIE_LOOP_UNROLL_FULL" not in self.macros(tokens)


def _loops(tokens: list[Token], definition: Definition) -> list[_Loop]:
    """The loop statements of a function body, in source order."""
    loops = []
    do_whiles: set[int] = set()
    for index in range(definition.body + 1, definition.end):
        if tokens[index].text not in ("for", "while", "do") or index in do_whiles:
            continue
        end = statement_end(tokens, index)
        if tokens[index].text == "do":
            do_whiles.add(statement_end(tokens, index + 1) + 1)
        prefix: list[tuple[int, int]] = []
        position = index
        while position > definition.body + 1:
            last = position - 1
            if tokens[last].text == ")":
                first = matching_open(tokens, last)
                if not _is_loop_macro(tokens[first - 1].text):
                    break
                prefix.insert(0, (first - 1, last))
                position = first - 1
            elif _is_loop_macro(tokens[last].text):
                prefix.insert(0, (last, last))
                position = last
            else:
                break
        loops.append(_Loop(index, end, tuple(prefix)))
    return loops


def _offsets(text: str) -> Callable[[Token], int]:
    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    return lambda token: line_starts[token.line - 1] + token.column - 1


def _calls(tokens: list[Token], start: int, end: int, names: Iterable[str]) -> set[str]:
    """The names called between token indices start and end (name( or name<...>()."""
    names = set(names)
    return {
        tokens[index].text
        for index in range(start, end)
        if tokens[index].text in names and tokens[index + 1].text in ("(", "<") and tokens[index - 1].text != "."
    }


def _helpers(tokens: list[Token], definitions: Sequence[Definition]) -> list[Definition]:
    """The always_inline functions the definitions call, also through each other, in source order: their loops are
    compiled into the targets."""
    inline: dict[str, list[Definition]] = {}
    for definition in function_definitions(tokens, ""):
        if any(token.text == "always_inline" for token in tokens[definition.start : definition.body]):
            inline.setdefault(definition.name, []).append(definition)
    found: dict[Definition, None] = {}
    pending = list(definitions)
    while pending:
        definition = pending.pop()
        for name in _calls(tokens, definition.body + 1, definition.end, inline):
            for helper in inline[name]:
                if helper not in found and helper not in definitions:
                    found[helper] = None
                    pending.append(helper)
    return sorted(found, key=lambda definition: definition.start)


def _target_loops(
    source: str, functions: Sequence[str], loops: Sequence[int] | None
) -> tuple[list[Token], list[_Loop]]:
    """The loops that take a pragma: the rolled loops with the given indices (in source order over the target
    functions, then over the always_inline helpers they call), by default the innermost rolled loops (a loop that
    calls a helper with rolled loops is not). A helper's pragma applies to every caller of the helper."""
    tokens = tokenize(source)
    definitions = [definition for name in functions for definition in function_definitions(tokens, name)]
    if not definitions:
        raise ValueError(f"no definition of {', '.join(functions)} in the kernel source")
    definitions = sorted(set(definitions), key=lambda definition: definition.start)
    helpers = _helpers(tokens, definitions)
    rolled = [
        loop for definition in [*definitions, *helpers] for loop in _loops(tokens, definition) if loop.rolled(tokens)
    ]
    if loops is not None:
        if any(not 0 <= k < len(rolled) for k in loops):
            raise ValueError(f"loop index out of range: {len(rolled)} rolled loop(s) in {', '.join(functions)} and "
                             "the always_inline helpers they call")
        return tokens, [rolled[k] for k in loops]
    looped = {  # the helpers with rolled loops, also in the helpers they call
        helper.name
        for helper in helpers
        if any(loop.rolled(tokens) for other in [helper, *_helpers(tokens, [helper])] for loop in _loops(tokens, other))
    }
    innermost = [
        loop
        for loop in rolled
        if not any(loop.keyword < other.keyword <= loop.end for other in rolled)
        and not _calls(tokens, loop.keyword, loop.end, looped)
    ]
    if not innermost:
        raise ValueError(f"no rolled loop in {', '.join(functions)}")
    return tokens, innermost


def apply_pragma(source: str, functions: Sequence[str], pragma: str, *, loops: Sequence[int] | None = None) -> str:
    """The source with the pipelining pragmas (AIE_PREPARE_FOR_POSTPIPELINING, AIE_TRY_INITIATION_INTERVAL) of
    the target loops (see _target_loops) replaced by pragma ("" removes them); other pragmas stay."""
    tokens, targets = _target_loops(source, functions, loops)
    offset = _offsets(source)
    edits: list[tuple[int, int, str]] = []
    for loop in targets:
        for first, last in loop.prefix:
            if tokens[first].text not in _PIPELINING:
                continue
            start, end = offset(tokens[first]), offset(tokens[last]) + len(tokens[last].text)
            line_start = source.rfind("\n", 0, start) + 1
            line_end = source.find("\n", end)
            line_end = len(source) if line_end < 0 else line_end
            if not source[line_start:start].strip() and not source[end:line_end].strip():
                start, end = line_start, min(line_end + 1, len(source))  # the macro's own line
            else:
                while end < len(source) and source[end] in " \t":
                    end += 1
            edits.append((start, end, ""))
        if pragma:
            at = offset(tokens[loop.keyword])
            indent = source[source.rfind("\n", 0, at) + 1 : at]
            edits.append((at, at, f"{pragma}\n{indent}" if not indent.strip() else f"{pragma} "))
    for start, end, text in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        source = source[:start] + text + source[end:]
    return source


def pragma_lines(source: str, functions: Sequence[str], *, loops: Sequence[int] | None = None) -> list[int]:
    """The lines of the loop statements a pragma goes on (see apply_pragma)."""
    tokens, targets = _target_loops(source, functions, loops)
    return [tokens[loop.keyword].line for loop in targets]


def substitute(source: str, old: str, new: str) -> str:
    """The source with old replaced by new; old must occur exactly once."""
    count = source.count(old)
    if count != 1:
        raise ValueError(f"substitution text matches {count} times (must match once): {old[:80]!r}")
    return source.replace(old, new)


def replace_function(source: str, name: str, definition: str) -> str:
    """The source with the definition of the function whose name contains name (exactly one) replaced."""
    tokens = tokenize(source)
    found = function_definitions(tokens, name)
    if len(found) != 1:
        raise ValueError(f"{len(found)} definitions of a function named *{name}* (must be one)")
    offset = _offsets(source)
    start, end = offset(tokens[found[0].start]), offset(tokens[found[0].end]) + 1
    return source[:start] + definition.strip() + source[end:]
