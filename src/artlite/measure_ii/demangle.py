"""Itanium demangling of the assembly's symbols (llvm-cxxfilt) and the short names the report prints."""

import re
import shutil
import subprocess
from pathlib import Path

_demangle_cache: dict[str, str] = {}


def _demangle_tool() -> str | None:
    try:
        from artlite.operators.base import CLANGPP

        llvm_cxxfilt = Path(CLANGPP).with_name("llvm-cxxfilt")
        if llvm_cxxfilt.exists():
            return str(llvm_cxxfilt)
    except (ImportError, AttributeError, SystemExit):
        pass
    return shutil.which("llvm-cxxfilt") or shutil.which("c++filt")


def demangle(mangled: str) -> str:
    if mangled in _demangle_cache:
        return _demangle_cache[mangled]
    tool = _demangle_tool()
    demangled = mangled
    if tool:
        try:
            result = subprocess.run(
                [tool, mangled],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                demangled = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    _demangle_cache[mangled] = demangled
    return demangled


# `(.exit186.7)`: the suffix of a clone or of an inlined function's exit block
_CLONE_SUFFIX = re.compile(r"\s+\((\.[^()]+)\)$")
# operator names hold brackets and spaces the signature scan must not see
_OPERATOR_NAME = re.compile(
    r"\boperator(?:\(\)|\[\]|<=>|->\*|<<=|>>=|->|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|[-+*/%^&|~!=<>,]=?"
    r'|""\s*\w+|\s+[^()]+?(?=\())'
)
_QUALIFIERS = re.compile(r"(?:\s*(?:const|volatile|&&|&|noexcept))*\s*")


def _top_level(text: str) -> list[tuple[int, str]]:
    """(index, char) of the characters outside parentheses and template arguments."""
    out = []
    parens = angles = 0
    for index, char in enumerate(text):
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif parens == 0 and char == "<":
            angles += 1
        elif parens == 0 and char == ">":
            angles -= 1
        elif parens == 0 and angles == 0:
            out.append((index, char))
    return out


def function_name(demangled: str) -> str:
    """Qualified name of a demangled function, with its template arguments.

    Drops the return type, the parameters, cv/ref qualifiers and the anonymous
    namespace: ``void (anonymous namespace)::srec<32, 4>(unsigned char const*)``
    is ``srec<32, 4>``. A clone suffix stays (``f.cold``)."""
    suffix = ""
    clone = _CLONE_SUFFIX.search(demangled)
    if clone:
        demangled, suffix = demangled[: clone.start()], clone.group(1)
    operators: list[str] = []

    def protect(match: re.Match[str]) -> str:
        operators.append(match.group(0))
        return f"\0{len(operators) - 1}\0"

    text = _OPERATOR_NAME.sub(protect, demangled)
    # the parameter list: the last top-level (...) that only qualifiers follow
    parens = angles = 0
    name_end = len(text)
    group_start = 0
    for index, char in enumerate(text):
        if char == "(":
            if parens == 0 and angles == 0:
                group_start = index
            parens += 1
        elif char == ")":
            parens -= 1
            if parens == 0 and angles == 0 and _QUALIFIERS.fullmatch(text, index + 1):
                name_end = group_start
                break
        elif parens == 0 and char == "<":
            angles += 1
        elif parens == 0 and char == ">":
            angles -= 1
    name = text[:name_end]
    # a return type (template functions) ends at the last top-level space
    spaces = [index for index, char in _top_level(name) if char == " "]
    if spaces:
        name = name[spaces[-1] + 1 :]
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub("\0(\\d+)\0", lambda match: operators[int(match.group(1))], name)
    return name + suffix


def _unqualified(name: str) -> str:
    """Last component of a qualified name, without its template arguments."""
    separators = [index for index, char in _top_level(name) if char == ":"]
    if separators:
        name = name[separators[-1] + 1 :]
    if name.endswith(">") and "<" in name and not name.startswith("operator"):
        name = name[: name.index("<")]
    return name


def demangle_short(mangled: str) -> str:
    if not mangled.startswith("_Z"):
        return mangled
    full = demangle(mangled)
    if full == mangled:
        return mangled
    return function_name(full) or mangled


def block_label(block: str) -> str:
    """A block name for printing: an inlined function's exit block ``_ZN3aie...3popEv.exit186.7`` is
    ``pop.exit186.7``."""
    if not block.startswith("_Z"):
        return block
    full = demangle(block)
    if full == block:
        return block
    clone = _CLONE_SUFFIX.search(full)
    suffix = clone.group(1) if clone else ""
    name = function_name(full[: clone.start()] if clone else full)
    return (_unqualified(name) or name) + suffix
