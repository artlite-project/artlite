"""Declared layouts of the int32 word arrays the host and the cores share: an X object's metadata, a core's state
words, a command. Each word array is a ``WordMap`` of ``size`` words whose words are declared by name and offset
(``Kind.word``, ``Kind.words``, ``Kind.block``, ``Kind.alias``), the flag bits of a flags word (``Kind.flag``) and the values a word
takes (``Kind.values``), under the kind of object or feature that owns them. The declarations check themselves where
they are made:

- a word lies inside the array, outside the ``common`` words (the words every kind keeps), and overlaps no word of a
  kind it can meet in one object;
- ``at=`` names a word, no ``at`` the first free one after the kind's last word (the first kind's: from word 0, past what it meets);
- a flag bit or value is declared once per field (``of``: "flags" by default) within a kind, and ``consts`` refuses a
  composition in which two kinds spell one name differently or one bit twice.

Which kinds meet is the map's mode. ``objects=True``: each kind is an object of its own (the metadata of a phase, of a
token object, of an attention object: their words may reuse one another's offsets) and meets only the kinds it
``extends`` (a model's words in a phase object: the model's kind extends the phase's, whose words it may not take).
``objects=False``: the kinds are features of one array (the state words of the main core: those of the post-norm
designs, of the MoE layers) and all meet, but the ones declared ``exclusive`` (they may share words; ``together``,
which a design calls with the kinds it enables, refuses them in one design).

``consts`` is what a kernel template receives: the names and values of the common words, the given kinds and what they
extend, and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable


class LayoutError(ValueError):
    """A word layout that overlaps, leaves its array or spells a name twice."""


class Kind:
    """The words, flag bits and values of one kind of object (or one feature) of a ``WordMap``."""

    def __init__(self, owner: WordMap, name: str, extends: tuple[Kind, ...]):
        self.owner, self.name, self.extends = owner, name, extends
        self._words: dict[str, tuple[int, int]] = {}  # name -> (first word, words)
        self._bits: dict[str, dict[str, int]] = {}  # field -> name -> bit
        self._values: dict[str, dict[str, int]] = {}  # field -> name -> value
        self._aliases: dict[str, int] = {}  # name -> word

    def __repr__(self) -> str:
        return f"Kind({self.owner.name}.{self.name})"

    # -- declarations --------------------------------------------------------------------------------------------

    def word(self, name: str, *, at: int | None = None) -> int:
        """Declare word ``name`` (at word ``at``, else the next free one): its index."""
        return self.block(name, 1, at=at)

    def words(self, names: str, *, at: int | None = None) -> tuple[int, ...]:
        """Declare the words of the blank-separated ``names`` one after the other (from word ``at``, else from the next
        free one): their indices."""
        first = self.block(names.split()[0], 1, at=at)
        return (first, *(self.word(n) for n in names.split()[1:]))

    def block(self, name: str, n: int, *, at: int | None = None) -> int:
        """Declare ``n`` consecutive words under ``name`` (exported: the first one): its index."""
        start = self.next_free(n) if at is None else at
        self._claim(name, start, n)
        self._words[name] = (start, n)
        return start

    def alias(self, name: str, word: int) -> int:
        """Another name for a word that is declared (the array's modes read one word as different things): ``word``."""
        if not any(s <= word < s + n for k in (self, *self.owner._meets(self)) for s, n in k._words.values()):
            raise LayoutError(f"{self}: {name} names word {word}, which no word of the kinds it meets covers")
        if name in self._aliases or name in self._words:
            raise LayoutError(f"{self}: {name} is declared twice")
        self._aliases[name] = word
        return word

    def flag(self, name: str, bit: int, *, of: str = "flags") -> int:
        """Declare bit ``bit`` of the flag field ``of`` (a word's bits: M_FLAGS by default): its value ``1 << bit``."""
        taken = self._bits.setdefault(of, {})
        if any(b == bit for b in taken.values()):
            raise LayoutError(f"{self}: bit {bit} of {of} is {next(k for k, b in taken.items() if b == bit)} and {name}")
        taken[name] = bit
        return 1 << bit

    def values(self, names: str, *, of: str, start: int = 0) -> tuple[int, ...]:
        """Declare the values ``start``, ``start + 1``, .. of the field ``of`` (what a word holds: a command's kind) under
        the blank-separated ``names``."""
        taken = self._values.setdefault(of, {})
        out = []
        for i, n in enumerate(names.split()):
            if start + i in taken.values():
                raise LayoutError(f"{self}: value {start + i} of {of} is declared twice ({n})")
            taken[n] = start + i
            out.append(start + i)
        return tuple(out)

    # -- the layout ------------------------------------------------------------------------------------------------

    def spans(self) -> dict[str, tuple[int, int]]:
        """The kind's own words: name -> (first word, words)."""
        return dict(self._words)

    def next_free(self, n: int = 1) -> int:
        """The first word where ``n`` words are free of the kinds it meets, from the end of this kind's last word on (from
        word 0 for the first)."""
        start = max((s + m for s, m in self._words.values()), default=0)
        taken = sorted(span for k in self.owner._meets(self) for span in k._words.values())
        for s, m in taken:
            if start < s + m and s < start + n:
                start = s + m
        return start

    def _claim(self, name: str, start: int, n: int) -> None:
        if n < 1 or start < 0 or start + n > self.owner.size:
            raise LayoutError(f"{self}: {name} words {start}..{start + n - 1} lie outside {self.owner.name}'s "
                              f"{self.owner.size} words")
        if name in self._words:
            raise LayoutError(f"{self}: {name} is declared twice")
        for k in (self, *self.owner._meets(self)):
            for other, (s, m) in k._words.items():
                if start < s + m and s < start + n:
                    raise LayoutError(f"{self}: {name} (words {start}..{start + n - 1}) overlaps {k.name}'s {other} "
                                      f"(words {s}..{s + m - 1})")

    def consts(self) -> dict[str, int]:
        """The kind's own names for a template: words (a block: its first), flags (``1 << bit``), values."""
        out = {n: s for n, (s, _) in self._words.items()} | self._aliases
        for field in self._bits.values():
            out |= {n: 1 << b for n, b in field.items()}
        for field in self._values.values():
            out |= field
        return out

    def _fields(self) -> Iterable[tuple[str, str, int]]:
        """(field, name, bit) of the flags, (field, name, value) of the values: what ``WordMap.consts`` keeps apart."""
        for field, names in self._bits.items():
            yield from ((f"bit:{field}", n, 1 << b) for n, b in names.items())
        for field, names in self._values.items():
            yield from ((f"value:{field}", n, v) for n, v in names.items())


class WordMap:
    """One int32 word array of ``size`` words (module docstring): the kinds that own its words."""

    def __init__(self, name: str, size: int, *, objects: bool = False):
        self.name, self.size, self.objects = name, size, objects
        self.common = Kind(self, "common", ())
        self._kinds: dict[str, Kind] = {}
        self._apart: set[frozenset[str]] = set()

    def kind(self, name: str, *, extends: tuple[Kind, ...] = (), exclusive_with: tuple[Kind, ...] = ()) -> Kind:
        """The kind ``name`` (declared once; ``extends``: kinds whose words it keeps, ``exclusive_with``: kinds that never
        meet it, whose words and flag bits it may reuse)."""
        if name in self._kinds:
            raise LayoutError(f"{self.name}: kind {name} is declared twice")
        if any(k.owner is not self for k in extends + exclusive_with):
            raise LayoutError(f"{self.name}: kind {name} extends or excludes a kind of another map")
        k = self._kinds[name] = Kind(self, name, extends)
        for other in exclusive_with:
            self.exclusive(k, other)
        return k

    def exclusive(self, *kinds: Kind) -> None:
        """Kinds that never meet in one design (of a map of features): pairwise, they may share words and flag bits."""
        for i, a in enumerate(kinds):
            for b in kinds[i + 1:]:
                if self._meets_by_extends(a, b):
                    raise LayoutError(f"{self.name}: {a.name} and {b.name} extend one another, they meet")
                self._apart.add(frozenset((a.name, b.name)))

    def together(self, *kinds: Kind) -> None:
        """A design that enables ``kinds`` in one array: refuses two kinds declared ``exclusive``."""
        for i, a in enumerate(kinds):
            for b in kinds[i + 1:]:
                if frozenset((a.name, b.name)) in self._apart:
                    raise LayoutError(f"{self.name}: {a.name} and {b.name} never meet in one design")

    def consts(self, *kinds: Kind) -> dict[str, int]:
        """What a kernel template receives (module docstring): the common words, ``kinds`` and what they extend. A name
        declared by two of them keeps one value; two names do not share a flag bit or a value of one field."""
        ks: list[Kind] = [self.common]
        for k in kinds:
            ks += [e for e in (*self._closure(k), k) if e not in ks]
        out: dict[str, int] = {}
        owner: dict[tuple[str, int], str] = {}
        for k in ks:
            for name, v in k.consts().items():
                if out.setdefault(name, v) != v:
                    raise LayoutError(f"{self.name}: {name} is {out[name]} and {v} (kind {k.name})")
            for field, name, v in k._fields():
                if owner.setdefault((field, v), name) != name:
                    raise LayoutError(f"{self.name}: {owner[field, v]} and {name} take one {field.split(':')[0]} of "
                                      f"{field.split(':')[1]} ({v})")
        return out

    def layout(self) -> list[tuple[str, str, int, int]]:
        """(kind, name, first word, words) of every declared word, by first word."""
        return sorted(((k.name, n, s, m) for k in (self.common, *self._kinds.values()) for n, (s, m) in k._words.items()),
                      key=lambda w: (w[2], w[0]))

    # -- which kinds meet ------------------------------------------------------------------------------------------

    def _closure(self, k: Kind) -> list[Kind]:
        out: list[Kind] = []
        for e in k.extends:
            out += [x for x in (*self._closure(e), e) if x not in out]
        return out

    def _meets_by_extends(self, a: Kind, b: Kind) -> bool:
        return b in self._closure(a) or a in self._closure(b)

    def _meets(self, k: Kind) -> list[Kind]:
        """The kinds whose words ``k``'s must not overlap: the common words and, of the others, in a map of objects those
        ``k`` extends or that extend it, in a map of features all but the exclusive ones."""
        if k is self.common:
            return [x for x in self._kinds.values()]
        out = [self.common]
        for o in self._kinds.values():
            if o is k:
                continue
            if self._meets_by_extends(k, o) or (not self.objects and frozenset((k.name, o.name)) not in self._apart):
                out.append(o)
        return out
