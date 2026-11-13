"""Constant trip counts parsed from a for-header and the template arguments of its function."""

import re
from collections.abc import Mapping
from pathlib import Path

from ..cxx_scan import function_bodies, tokenize
from .demangle import demangle, function_name

_FOR_HEADER = re.compile(
    r"\bfor\s*\(\s*(?:[\w:]+\s+)*(\w+)\s*=\s*([^;]+?)\s*;"  # [type] i = A;
    r"\s*(\w+)\s*(<=|<|!=)\s*([^;]+?)\s*;"  # i < B;
    r"\s*(?:\+\+\s*(\w+)|(\w+)\s*\+\+|(\w+)\s*\+=\s*([^)]+?))\s*\)"  # ++i | i++ | i += S
)


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
    """Values of the template parameters of the function (an instance: `name<256, 4>`, demangled) whose
    definition contains line line_no: {"N": "256", ...}; empty when unknown."""
    name = function_name(demangle(function)) if function else ""
    if "<" not in name:
        return {}
    arguments = [text.strip() for _, text in _split_top_level(name[name.index("<") + 1 : name.rindex(">")])]
    try:
        tokens = tokenize(source.read_text(errors="replace"))
    except OSError:
        return {}
    bodies = [body for body in function_bodies(tokens) if tokens[body[0]].line <= line_no <= tokens[body[1]].line]
    if not bodies:
        return {}
    start = max(bodies)[0]
    index = start  # back to the `template <...>` in front of the declaration
    while index > 0 and tokens[index].text not in {";", "}"} and tokens[index].text != "template":
        index -= 1
    if tokens[index].text != "template" or tokens[index + 1].text != "<":
        return {}
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
    return {key: value for key, value in zip(names, arguments) if key and re.fullmatch(r"-?\d+[uUlL]*", value)}


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
        lines = source.read_text(errors="replace").splitlines()
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
