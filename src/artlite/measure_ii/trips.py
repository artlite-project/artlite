"""Constant trip counts parsed from a for-header and the template arguments of its function."""

import functools
import re
from collections.abc import Mapping
from pathlib import Path

from ..cxx_scan import Token, function_bodies, function_definitions, tokenize
from .demangle import demangle, function_name, unqualified

_FOR_HEADER = re.compile(
    r"\bfor\s*\(\s*(?:[\w:]+\s+)*(\w+)\s*=\s*([^;]+?)\s*;"  # [type] i = A;
    r"\s*(\w+)\s*(<=|<|!=)\s*([^;]+?)\s*;"  # i < B;
    r"\s*(?:\+\+\s*(\w+)|(\w+)\s*\+\+|(\w+)\s*\+=\s*([^)]+?))\s*\)"  # ++i | i++ | i += S
)


@functools.lru_cache(maxsize=8)
def _tokens(text: str) -> list[Token]:
    """A source's tokens (by its text: a kernel's loops are looked up in the same file)."""
    return tokenize(text)


@functools.lru_cache(maxsize=8)
def _lines(text: str) -> list[str]:
    return text.splitlines()


def _constant(expr: str) -> int | None:
    """Value of a constant integer expression (C semantics)."""
    expr = re.sub(r"(?<=\d)[uUlL]+\b", "", expr)
    if not re.fullmatch(r"[\d\s+\-*/()]+", expr):
        return None
    try:
        value = eval(expr.replace("/", "//"), {"__builtins__": {}}, {})
    except (SyntaxError, ZeroDivisionError, NameError, TypeError):
        return None
    return value if isinstance(value, int) else None


def template_arguments(source: Path, line_no: int, function: str) -> dict[str, str]:
    """Values of the template parameters of the definition that contains line line_no, in the code of function (an
    instance, `name<256, 4>` demangled): {"N": "256", ...}; empty when unknown. A line in another definition (a
    template helper inlined into function) takes the arguments of the helper's calls in function's body, when they
    are constants (function's own parameters substituted) and all the same."""
    name = function_name(demangle(function)) if function else ""
    try:
        tokens = _tokens(source.read_text(errors="replace"))
    except OSError:
        return {}
    bodies = [body for body in function_bodies(tokens) if tokens[body[0]].line <= line_no <= tokens[body[1]].line]
    if not bodies:
        return {}
    start = max(bodies)[0]
    parameters = _template_parameters(tokens, start)
    caller = unqualified(name)
    definitions = {d.body: d for d in function_definitions(tokens, "")}
    if start in definitions and definitions[start].name != caller:
        own: dict[str, str] = {}
        for d in definitions.values():
            if d.name == caller:
                own = _bind(_template_parameters(tokens, d.body), _instance_arguments(name))
                break
        arguments = _call_arguments(tokens, caller, definitions[start].name, own)
    else:
        arguments = _instance_arguments(name)
    return _bind(parameters, arguments)


def _instance_arguments(name: str) -> list[str]:
    """The template arguments of a demangled instance name (`f<8, 4u>`: ["8", "4u"])."""
    if "<" not in name:
        return []
    return [text.strip() for _, text in _split_top_level(name[name.index("<") + 1 : name.rindex(">")])]


def _bind(parameters: list[str], arguments: list[str]) -> dict[str, str]:
    return {key: value for key, value in zip(parameters, arguments) if key and re.fullmatch(r"-?\d+[uUlL]*", value)}


def _call_arguments(tokens: list[Token], caller: str, callee: str, own: Mapping[str, str]) -> list[str]:
    """The constant template arguments of the calls `callee<...>(` in caller's definitions (own: caller's parameters'
    values), when every call has the same ones; else []."""
    found: set[tuple[str, ...]] = set()
    for d in function_definitions(tokens, caller):
        if d.name != caller:
            continue
        for index in range(d.body, d.end - 1):
            if tokens[index].text != callee or tokens[index + 1].text != "<":
                continue
            depth, position, parts, current = 0, index + 1, [], []
            while position < d.end:
                text = tokens[position].text
                position += 1
                depth += text in ("<", "(")
                depth -= text in (">", ")")
                if depth == 0:
                    parts.append(" ".join(current))
                    break
                if text == "," and depth == 1:
                    parts.append(" ".join(current))
                    current = []
                elif depth > 1 or text != "<":
                    current.append(text)
            if tokens[position].text != "(":
                continue  # (not a call)
            values = []
            for part in parts:
                for key, value in own.items():
                    part = re.sub(rf"\b{key}\b", f"({value})", part)
                values.append(_constant(part))
            if None in values:
                return []
            found.add(tuple(str(value) for value in values))
    return list(found.pop()) if len(found) == 1 else []


def _template_parameters(tokens: list[Token], start: int) -> list[str]:
    """The template parameters' names of the definition whose body opens at tokens[start] ("" for an unnamed one);
    [] when it is no template."""
    index = start  # back to the `template <...>` in front of the declaration
    while index > 0 and tokens[index].text not in {";", "}"} and tokens[index].text != "template":
        index -= 1
    if tokens[index].text != "template" or tokens[index + 1].text != "<":
        return []
    depth, position, parameters, current = 0, index + 1, [], []
    while position < start:
        text = tokens[position].text
        position += 1
        if text == "<":
            depth += 1
            if depth == 1:
                continue
        elif text == ">":
            depth -= 1
            if depth == 0:
                parameters.append(current)
                break
        elif text == "," and depth == 1:
            parameters.append(current)
            current = []
            continue
        current.append(text)
    names = []
    for parameter in parameters:
        if "=" in parameter:
            parameter = parameter[: parameter.index("=")]
        names.append(parameter[-1] if parameter and re.fullmatch(r"[A-Za-z_]\w*", parameter[-1]) else "")
    return names


def _split_top_level(text: str) -> list[tuple[int, str]]:
    """Comma-separated parts of a template argument list, outside nested brackets."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append((start, text[start:index]))
            start = index + 1
    parts.append((start, text[start:]))
    return parts


def parse_source_trip(source: Path, line_no: int, constants: Mapping[str, str] | None = None) -> int | None:
    """Parse a constant trip count from a `for (i = A; i < B; i += S)` header (also `<=`, `!=`, `i++`);
    constants: names to substitute (template parameters of the function instance)."""
    try:
        lines = _lines(source.read_text(errors="replace"))
    except OSError:
        return None
    if not 1 <= line_no <= len(lines):
        return None
    # The for-header may spill across lines; join three and try.
    text = " ".join(lines[line_no - 1 : line_no + 2])
    match = _FOR_HEADER.search(text)
    if not match or match.start() >= len(lines[line_no - 1]):
        return None
    var, init, cond_var, op, bound, pre_inc, post_inc, step_var, step = match.groups()
    if not var == cond_var == (pre_inc or post_inc or step_var):
        return None
    for name, value in (constants or {}).items():
        init, bound = (re.sub(rf"\b{name}\b", f"({value})", text) for text in (init, bound))
        step = None if step is None else re.sub(rf"\b{name}\b", f"({value})", step)
    start, end = _constant(init), _constant(bound)
    stride = 1 if step is None else _constant(step)
    if start is None or end is None or stride is None or stride <= 0:
        return None
    if op == "<=":
        end += 1
    elif op == "!=" and (end - start) % stride:
        return None
    return max(0, -(-(end - start) // stride))
