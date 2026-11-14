"""The code a cached state depends on, derived from the import graph (``import_closure``): the modules a network's
loader imports within ``artlite`` and ``models`` (statically: every ``import`` of a module's source outside
``if TYPE_CHECKING:``, function-local ones included), their packages' ``__init__`` modules (not what those import), and
each module's kernel templates (``x.py``'s ``x.*.jinja``). A state's key (``state.state_dir``) hashes these files, so
that a change to anything that builds the state's programs and images makes its cache miss; nothing lists them by
hand."""

from __future__ import annotations

import ast
import functools
import glob
import os
from collections.abc import Iterator

from ..operators.toolchain import PACKAGE_DIR

ROOTS = ("artlite", "models")  # (the packages the closure follows)
SRC = os.path.dirname(PACKAGE_DIR)  # (their directory)


def module_path(name: str) -> str | None:
    """The source file of module ``name`` of ``ROOTS`` (a package: its ``__init__.py``), None where there is none."""
    base = os.path.join(SRC, *name.split("."))
    for p in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(p):
            return p
    return None


def _walk(tree: ast.AST) -> Iterator[ast.AST]:
    """The nodes of ``tree`` outside ``if TYPE_CHECKING:`` blocks (annotations only: no code of theirs runs)."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for n in node.orelse:
                yield n
                yield from _walk(n)
            continue
        yield node
        yield from _walk(node)


@functools.cache
def imports(name: str) -> frozenset[str]:
    """The modules of ``ROOTS`` that module ``name``'s source imports (``from p import m``: ``p.m`` where that is a
    module, and ``p`` itself where ``m`` is a name of its own)."""
    path = module_path(name)
    assert path is not None, name
    with open(path, "rb") as f:
        tree = ast.parse(f.read(), path)
    package = name if os.path.basename(path) == "__init__.py" else name.rpartition(".")[0]
    out: set[str] = set()
    for node in _walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = package.split(".")[:len(package.split(".")) - node.level + 1]
                base = ".".join([*up, *([base] if base else [])])
            subs = [f"{base}.{a.name}" for a in node.names]
            out.update(subs)
            if any(module_path(m) is None for m in subs):  # (a name of the package's own: its code is used)
                out.add(base)
    return frozenset(m for m in out if m.split(".")[0] in ROOTS and module_path(m) is not None)


def import_closure(*modules: str) -> list[str]:
    """The modules (names, sorted) that ``modules`` import, themselves included; the ``__init__`` of each one's packages
    too, but not what those import (a package's ``__init__`` that others import explicitly is followed like a
    module)."""
    seen: set[str] = set()
    todo = list(modules)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo += imports(m)
    for m in tuple(seen):
        parts = m.split(".")
        seen.update(".".join(parts[:i]) for i in range(1, len(parts)))
    return sorted(seen)


def source_files(*modules: str) -> list[str]:
    """The files ``import_closure(*modules)`` stands for: each module's source and its kernel templates."""
    out = []
    for m in import_closure(*modules):
        p = module_path(m)
        out += [p, *sorted(glob.glob(os.path.splitext(glob.escape(p))[0] + ".*.jinja"))]
    return out
