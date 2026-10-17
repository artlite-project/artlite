"""Kernel sources named by their content: every definition of a translation unit appears once.

A kernel template writes each namespace-scope name (function, table, type, macro) as ``{{ p }}_<local>`` with ``p =
kernel_prefix(stem)``, a placeholder (``__artlite_<stem>``), and its design declares its kernels with the same names
(``Kernel(f"{p}_<local>", ...)``). ``KernelSource`` splits the rendered source into its top-level items (definitions
and declarations with their leading comments, inside ``namespace {`` / ``extern "C" {`` or not) and names every
placeholder name ``<stem>_<local>_<hash>``: the hash of the text of the items that define it, with the names they
use resolved the same way (so a name changes with anything its code reaches; items that reach each other, e.g. a
declaration and its definition, share one hash). ``rename`` resolves the design's MLIR.

Two operators (or two configurations of one) that render the same helper therefore render the same name, and
``translation_unit`` emits it once: a fused ELF's ``kernel.cc`` holds one section per operator with only its new
items, the shared ones listed in the section's header. Equal code is never renamed apart, different code never
shares a name (a hash collision is an error). Identical items under different local names stay distinct (a table
and its copy in another bank).

The stem is the readable part of the names; a helper renderer shared by several operators renders under its own stem
(``act_lut``: ``lut``, ``gemm.mac``: ``mac``), so that its items are shared across operator classes too.

Macros without the placeholder (``#define AW(T, off) ...``) are context: an item's hash includes the definitions of
the ones it uses, and ``translation_unit`` emits a definition before the first item that needs it in force. Items
without placeholder names (operators with fixed names) are emitted once per text.
"""

import hashlib
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

MARK = "__artlite_"
_NAME = re.compile(r"\b__artlite_\w+")
_TOKEN = re.compile(r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|\w+|\s+|.""", re.DOTALL)
_REST = re.compile(r"[ \t]*(//[^\n]*)?\n?")  # the rest of a line: blanks, a comment
_HASH_SIGN = re.compile(r"[ \t]*#")  # (a directive, at the start of a line)
_DIRECTIVE = re.compile(r"^[ \t]*#.*", re.DOTALL | re.MULTILINE)
_HASH = 8  # hex digits of a name's hash
_WRAPPERS = {"namespace": "namespace {", "C": 'extern "C" {'}
_CLOSERS = {"namespace": "}  // namespace", "C": '}  // extern "C"'}
_TAGS = ("struct", "class", "union", "enum")


def kernel_prefix(stem: str) -> str:
    """The placeholder prefix of a kernel source's names under ``stem`` (the module docstring)."""
    assert re.fullmatch(r"[A-Za-z]\w*", stem), stem
    return MARK + stem


class KernelSourceError(ValueError):
    pass


@dataclass
class Item:
    """A top-level item of a kernel source: its text (leading comments included), the block it stands in
    ("namespace", "C" or "" at file scope), the names it defines and the placeholder names it uses."""

    text: str
    wrapper: str
    defines: tuple[str, ...]
    refs: frozenset[str]
    macro: str | None = None  # an unprefixed macro it defines (context: ``translation_unit``)
    macro_refs: tuple[str, ...] = ()  # the unprefixed macros (of its source) it uses


@dataclass
class _Tok:
    text: str
    start: int
    end: int

    @property
    def significant(self) -> bool:
        return not (self.text[0].isspace() or self.text.startswith(("//", "/*")))


def _tokens(text: str) -> Iterator[_Tok]:
    """Tokens (comments, strings, words, whitespace, single characters); a preprocessor directive (with its
    continuation lines) is one token."""
    pos, line_start = 0, True
    while pos < len(text):
        if line_start:
            m = _HASH_SIGN.match(text, pos)
            if m:
                end = pos
                while True:
                    nl = text.find("\n", end)
                    if nl < 0:
                        end = len(text)
                        break
                    end = nl + 1
                    if not text[:nl].rstrip("\r").endswith("\\"):
                        break
                yield _Tok(text[pos:end], pos, end)
                pos = end
                continue
        m = _TOKEN.match(text, pos)
        tok = m.group()
        line_start = (line_start or "\n" in tok) if tok.isspace() else False
        yield _Tok(tok, pos, m.end())
        pos = m.end()


def _strip_groups(head: list[str]) -> list[str]:
    """Declaration head tokens without attributes (``__attribute__((..))``, ``[[..]]``, ``alignas(..)``) and template
    parameter lists."""
    out, i = [], 0
    while i < len(head):
        t = head[i]
        if t in ("__attribute__", "alignas", "__declspec") and i + 1 < len(head) and head[i + 1] == "(":
            i = _skip(head, i + 1, "(", ")")
            continue
        if t == "[" and i + 1 < len(head) and head[i + 1] == "[":
            i = _skip(head, i, "[", "]")
            continue
        if t == "template" and i + 1 < len(head) and head[i + 1] == "<":
            i = _skip(head, i + 1, "<", ">")
            continue
        out.append(t)
        i += 1
    return out


def _skip(toks: list[str], i: int, op: str, cl: str) -> int:
    """Index after the bracket closing toks[i] (an ``op``)."""
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == op:
            depth += 1
        elif toks[j] == cl:
            depth -= 1
            if depth == 0:
                return j + 1
    raise KernelSourceError(f"unbalanced {op}{cl} in a declaration: {' '.join(toks)[:200]}")


def _defined(head: list[str]) -> tuple[str, ...]:
    """The names a declaration defines, from its head tokens (up to its body or end)."""
    if head and head[0].startswith("#"):
        m = re.match(r"\s*#\s*define\s+(\w+)", head[0])
        return (m.group(1),) if m else ()
    h = _strip_groups(head)
    for i, t in enumerate(h[:-1]):
        if t in _TAGS and re.fullmatch(r"[A-Za-z_]\w*", h[i + 1]) and h[i + 1] != "class":
            return (h[i + 1],)
        if t == "enum" and h[i + 1] == "class":
            return (h[i + 2],)
    if h[:1] == ["using"] and len(h) > 2 and h[2] == "=":
        return (h[1],)
    names, depth, i = [], 0, 0
    while i < len(h):
        t = h[i]
        if t == "(" and depth == 0:
            nxt = h[i + 1] if i + 1 < len(h) else ""
            if nxt == "*":  # (a declarator in parentheses: a function pointer)
                j = i + 2
                while j < len(h) and h[j] in ("const", "volatile", "__restrict"):
                    j += 1
                return (h[j],) if j < len(h) else tuple(names)
            prev = h[i - 1] if i else ""
            if re.fullmatch(r"[A-Za-z_]\w*", prev) and prev not in ("operator",):
                return (prev,)
            depth += 1
        else:
            if depth == 0 and t in ("[", "=", ",", ";") and i and re.fullmatch(r"[A-Za-z_]\w*", h[i - 1]):
                names.append(h[i - 1])
            if t in ("(", "[", "{"):
                depth += 1
            elif t in (")", "]", "}"):
                depth -= 1
        i += 1
    if depth == 0 and h and re.fullmatch(r"[A-Za-z_]\w*", h[-1]) and len(h) > 1:
        names.append(h[-1])
    return tuple(dict.fromkeys(names))


def split(text: str) -> list[Item]:
    """The top-level items of a kernel source (module docstring)."""
    items: list[Item] = []
    wrappers: list[str] = []
    start = 0  # where the current item's text (its leading comments) starts
    head: list[_Tok] = []  # its significant tokens before its body
    depth = 0
    toks = list(_tokens(text))

    def finish(end: int) -> None:
        nonlocal start, head
        if text[end - 1] != "\n":  # (a comment on the rest of the line belongs to the item)
            end = _REST.match(text, end).end()
        words = [t.text for t in head]
        items.append(Item(text=text[start:end], wrapper=wrappers[-1] if wrappers else "", defines=_defined(words),
                          refs=frozenset()))
        start, head = end, []

    for k, tok in enumerate(toks):
        if tok.end <= start or not tok.significant:
            continue
        t = tok.text
        if depth == 0:
            if t.startswith("#"):
                if head:
                    raise KernelSourceError(f"a preprocessor directive inside a declaration: {t.strip()}")
                head = [tok]
                finish(tok.end)
                continue
            words = [h.text for h in head]
            if t == "{" and (words in (["namespace"], ['extern', '"C"']) or
                             (len(words) == 2 and words[0] == "namespace")):
                if len(words) == 2 and words[0] == "namespace":
                    raise KernelSourceError(f"named namespace {words[1]}: kernel sources use prefixed names")
                if text[start:head[0].start].strip():  # (comments before the block: an item of their own)
                    items.append(Item(text=text[start:head[0].start], wrapper=wrappers[-1] if wrappers else "",
                                      defines=(), refs=frozenset()))
                wrappers.append("namespace" if words[0] == "namespace" else "C")
                start, head = _REST.match(text, tok.end).end(), []
                continue
            if t == "}":
                if head or not wrappers:
                    raise KernelSourceError(f"unbalanced '}}' at {text[max(0, tok.start - 80):tok.end]!r}")
                if text[start:tok.start].strip():  # (trailing comments of the block: with its last item)
                    if items and items[-1].wrapper == wrappers[-1]:
                        items[-1].text += text[start:tok.start]
                    else:
                        items.append(Item(text=text[start:tok.start], wrapper=wrappers[-1], defines=(),
                                          refs=frozenset()))
                wrappers.pop()
                start = _REST.match(text, tok.end).end()
                continue
            if t == ";":
                head.append(tok)
                finish(tok.end)
                continue
            if t == "{":
                depth = 1
            head.append(tok)
            continue
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1
            if depth == 0:
                nxt = next((toks[j] for j in range(k + 1, len(toks)) if toks[j].significant), None)
                if nxt is None or nxt.text not in (";", ","):
                    finish(tok.end)
    if depth or head or wrappers:
        raise KernelSourceError("the source ends inside a declaration or block")
    if text[start:].strip():
        if items:
            items[-1].text += text[start:]
        else:
            items.append(Item(text=text[start:], wrapper="", defines=(), refs=frozenset()))
    for item in items:
        own = set(item.defines)
        item.refs = frozenset(_NAME.findall(item.text)) - own
        m = re.match(r"(?:\s|//[^\n]*|/\*.*?\*/)*#\s*define\s+(\w+)", item.text, re.DOTALL)
        if m and not m.group(1).startswith(MARK):
            item.macro = m.group(1)
    return items


def _normal(text: str) -> str:
    """Text for hashing and comparing: lines without trailing blanks, no blank lines."""
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def _words(text: str) -> set[str]:
    """The identifiers of C++ text outside comments, strings and preprocessor directives."""
    return {t.text for t in _tokens(text) if t.significant and (t.text[0].isalpha() or t.text[0] == "_")}


class KernelSource:
    """A rendered kernel source with its placeholder names resolved (module docstring). ``label``: its section
    header in a translation unit, ``detail``: what tells it apart from others of the label (in a translation unit of
    several sources)."""

    def __init__(self, raw: str, label: str = "", detail: str = ""):
        self.raw, self.label, self.detail = raw, label, detail
        self.items = split(raw)
        macros: dict[str, str] = {}  # unprefixed macro -> its definition in force
        self.macro_text: dict[int, dict[str, str]] = {}  # item -> {macro it uses: definition}
        for i, item in enumerate(self.items):
            if macros:
                item.macro_refs = tuple(sorted(_words(item.text) & set(macros) - {item.macro}))
                self.macro_text[i] = {m: macros[m] for m in item.macro_refs}
            if item.macro is not None:
                macros[item.macro] = item.text.strip("\n")
        self.names = self._resolve()
        self.texts = [self._sub(item.text) for item in self.items]

    # -- naming -------------------------------------------------------------------------------------------------

    def _resolve(self) -> dict[str, str]:
        """{placeholder name: resolved name}: Tarjan's SCCs of the items (an item -> the items defining the names it
        uses; items defining one name: mutually), hashed dependencies first."""
        by_name: dict[str, list[int]] = {}
        for i, item in enumerate(self.items):
            for n in item.defines:
                if n.startswith(MARK):
                    by_name.setdefault(n, []).append(i)
        unknown = sorted({n for item in self.items for n in item.refs} - set(by_name))
        if unknown:
            raise KernelSourceError(f"{self.label}: placeholder names used but not defined: {', '.join(unknown[:8])}")
        edges = [sorted({j for n in item.refs | {d for d in item.defines if d.startswith(MARK)}
                         for j in by_name.get(n, ()) if j != i}) for i, item in enumerate(self.items)]
        names: dict[str, str] = {}
        for scc in _sccs(edges):
            members = sorted(scc)
            h = hashlib.sha256()
            for i in members:
                item = self.items[i]
                text = _NAME.sub(lambda m: names.get(m.group(), m.group()), _normal(item.text))
                h.update(f"{item.wrapper}\0{text}\0".encode())
                for m, t in sorted(self.macro_text.get(i, {}).items()):
                    h.update(f"{m}\0{_normal(_DIRECTIVE.search(t).group())}\0".encode())
            digest = h.hexdigest()[:_HASH]
            for i in members:
                for n in self.items[i].defines:
                    if n.startswith(MARK):
                        names[n] = f"{n[len(MARK):]}_{digest}"
        return names

    def _sub(self, text: str) -> str:
        return _NAME.sub(lambda m: self.names[m.group()], text)

    def rename(self, text: str) -> str:
        """``text`` (e.g. the operator's MLIR) with this source's placeholder names resolved; any other placeholder
        name is an error."""
        def sub(m: re.Match) -> str:
            name = self.names.get(m.group())
            if name is None:
                raise KernelSourceError(f"{self.label}: {m.group()} is not defined by the kernel source")
            return name
        return _NAME.sub(sub, text)

    def name(self, raw: str) -> str:
        """The resolved name of placeholder name ``raw``."""
        return self.names[raw]

    @property
    def text(self) -> str:
        """The source alone as a translation unit (``translation_unit``)."""
        return translation_unit([self])


def _sccs(edges: list[list[int]]) -> list[list[int]]:
    """Strongly connected components of a graph (node -> successors), each after those it reaches."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    on: set[int] = set()
    out: list[list[int]] = []
    counter = 0
    for root in range(len(edges)):
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work.pop()
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on.add(v)
            recurse = False
            for k in range(pi, len(edges[v])):
                w = edges[v][k]
                if w not in index:
                    work.append((v, k + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                out.append(scc)
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return out


INCLUDE = '#include "operators.h"\n'


def translation_unit(sources: Iterable[KernelSource]) -> str:
    """One ``kernel.cc`` of several kernel sources: a section per source in order, each holding the source's items
    that no earlier section emitted (their names listed in its header); unprefixed macro definitions emitted where an
    item needs them in force. One source's header is its label alone: equal code and designs, equal files."""
    sources = list(sources)
    out = [INCLUDE]
    emitted: set[tuple[str, str]] = set()  # (block, normalized text) of the items emitted
    owner: dict[str, int] = {}  # resolved name -> the source whose items define it (a hash collision check)
    macros: dict[str, str] = {}  # unprefixed macro -> the definition in force
    for k, src in enumerate(sources):
        body: list[tuple[str, str]] = []  # (block, text)
        shared: list[str] = []
        for i, item in enumerate(src.items):
            if item.macro is not None:
                continue
            text = src.texts[i]
            key = (item.wrapper, _normal(text))
            if key in emitted:
                shared += [src.names.get(n, n) for n in item.defines]
                continue
            for n in item.defines:
                if n in src.names and owner.setdefault(src.names[n], k) != k:
                    raise KernelSourceError(f"{src.names[n]}: other definitions in an earlier source than in "
                                            f"{src.label} (a hash collision)")
            for m in item.macro_refs:
                definition = src.macro_text[i][m]
                if macros.get(m) != definition:
                    body.append((item.wrapper, definition))
                    macros[m] = definition
            emitted.add(key)
            body.append((item.wrapper, text))
        out.append(_section(src.label, src.detail if len(sources) > 1 else "", body, shared))
    return "\n".join(out)


def _section(label: str, detail: str, body: Sequence[tuple[str, str]], shared: Sequence[str]) -> str:
    lines = [f"// ==== {label} ====" if label else "// ===="]
    if detail:
        lines += _wrap("// ", detail.split(" "), sep="")
    if shared:
        lines += _wrap("// shared with an earlier section: ", list(dict.fromkeys(shared)))
    out = "\n".join(lines) + "\n"
    block = ""
    for wrapper, text in body:
        if wrapper != block:
            if block:
                out += "\n" + _CLOSERS[block] + "\n"
            if wrapper:
                out += "\n" + _WRAPPERS[wrapper] + "\n"
            block = wrapper
        out += "\n" + text.strip("\n") + "\n"
    if block:
        out += "\n" + _CLOSERS[block] + "\n"
    return out


def _wrap(first: str, words: Sequence[str], width: int = 120, sep: str = ",") -> list[str]:
    lines, cur = [], first
    for i, w in enumerate(words):
        piece = w + (sep if i + 1 < len(words) else "")
        if len(cur) + len(piece) + 1 > width and cur.strip() != first.strip():
            lines.append(cur.rstrip())
            cur = "//     "
        cur += piece + " "
    lines.append(cur.rstrip())
    return lines
