"""The generated text's filters, pure host state machines over a stream of text: stop strings (``StopMatcher``), the
split of a thinking model's text into thoughts and answer (``ReasoningSplitter``; a turn of blocks: ``BlockSplitter``)
and a thought's token budget (``ThoughtBudget``). Text that may begin a word they look for is held back until what
follows decides (``held``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import Blocks


def held(text: str, words: tuple[str, ...] | list[str]) -> int:
    """The length of the longest end of ``text`` that begins one of ``words`` (without being all of it): text held
    back until what follows decides."""
    best = 0
    for w in words:
        for k in range(min(len(w) - 1, len(text)), best, -1):
            if text.endswith(w[:k]):
                best = k
                break
    return best


class StopMatcher:
    """Stop strings in a stream of text: ``feed`` returns the text that may be sent and whether a stop string ended it
    (the text is cut before the stop string, ``word``). Text that may begin a stop string is held back until what
    follows decides."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.word: str | None = None
        self._held = ""

    def feed(self, text: str) -> tuple[str, bool]:
        if not self.stops:
            return text, False
        s = self._held + text
        hits = [(i, w) for w in self.stops if (i := s.find(w)) >= 0]
        if hits:
            i, self.word = min(hits)
            self._held = ""
            return s[:i], True
        k = held(s, self.stops)
        self._held = s[len(s) - k:]
        return s[:len(s) - k], False

    def flush(self) -> str:
        """The text held back, at the end of the generation."""
        out, self._held = self._held, ""
        return out


class ReasoningSplitter:
    """A thinking model's generated text split into its thoughts (``reasoning_content``) and its answer (``content``) by
    its thought markers (``Model.reasoning``): ``feed`` returns the (kind, text) parts a text completes. The markers
    are dropped, and so is the white space around them; text that may begin a marker and trailing white space are held
    back until what follows decides. A close marker outside a thought is dropped (Gemma may end an answer with one).
    ``inside``: the text starts inside a thought (the prompt opened it); ``ends``: text that ends a thought and begins
    the answer (kept: a Qwen tool call without the close marker); ``once``: a thought only begins the text (Qwen's),
    markers after it are answer; ``lead``: the text's leading white space is dropped (not where it continues a message's
    text). ``answering``: the answer has begun (a thought ended, or answer text came)."""

    def __init__(self, markers: tuple[str, str], *, inside: bool = False, ends: tuple[str, ...] = (),
                 once: bool = False, lead: bool = True):
        self.open, self.close = markers
        self.inside = inside
        self.ends, self.once = ends, once
        self._held = ""
        self._lead = lead  # (a part begins: its leading white space is dropped)
        self._over = False  # (once: the thought is over, or the text began outside one)
        self.answering = False

    @property
    def kind(self) -> str:
        return "reasoning_content" if self.inside else "content"

    def _part(self, out: list[tuple[str, str]], text: str) -> None:
        if self._lead:
            text = text.lstrip()
            self._lead = not text
        if text:
            out.append((self.kind, text))
            if not self.inside:
                self._over = self.once
                self.answering = True

    def _marks(self) -> list[str]:
        if self.inside:
            return [self.close, *self.ends]
        return [] if self._over else [self.open, self.close]

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        s = self._held + text
        while True:
            marks = self._marks()
            hits = [(i, m) for m in marks if (i := s.find(m)) >= 0]
            if not hits:
                break
            i, m = min(hits)
            self._part(out, s[:i].rstrip())
            s = s[i:] if m in self.ends and self.inside else s[i + len(m):]
            self.answering = self.answering or self.inside
            self.inside = m == self.open
            self._over = self.once and not self.inside
            self._lead = True
        k = held(s, marks)
        body = s[:len(s) - k]
        text = body.rstrip()
        self._held = body[len(text):] + s[len(s) - k:]
        self._part(out, text)
        return out

    def flush(self) -> list[tuple[str, str]]:
        """The text held back, at the end of the generation."""
        out: list[tuple[str, str]] = []
        self._part(out, self._held.rstrip())
        self._held = ""
        return out


class BlockSplitter:
    """A turn the model writes as blocks (``Model.blocks``: Inkling's ``header`` KIND .. ``end``) split as a
    ``ReasoningSplitter`` splits a thinking model's text: a thought block's text is a thought (``reasoning`` off: the
    content, its kind and end markers kept, as llama-server leaves it), a text block's the answer's content, any other
    block (a tool call's: ``header`` NAME ``call`` ..) content as it is, its markers too, for the call parser; text
    outside the blocks (a continued message's) is content, an end marker there dropped. The thought and text blocks'
    markers and the white space around their text are dropped; text that may begin a marker, and a block's trailing
    white space, are held back until what follows decides. ``inside``: the text starts inside a thought (the prompt
    opened it); ``lead``: the text's leading white space is dropped. ``answering``: the answer has begun (a thought
    ended, or answer text came)."""

    def __init__(self, blocks: Blocks, *, inside: bool = False, reasoning: bool = True, lead: bool = True):
        self.b = blocks
        self.reasoning = reasoning
        self.state = "thought" if inside else "out"  # out (between blocks), head (after a header), thought, text, other
        self._held = ""
        self._lead = lead
        self.answering = False

    @property
    def inside(self) -> bool:
        return self.state == "thought"

    def _part(self, out: list[tuple[str, str]], text: str, *, thought: bool = False) -> None:
        if self._lead:
            text = text.lstrip()
            self._lead = not text
        if text:
            kind = "reasoning_content" if thought and self.reasoning else "content"
            out.append((kind, text))
            self.answering = self.answering or kind == "content"

    def _raw(self, out: list[tuple[str, str]], text: str) -> None:
        if text:
            out.append(("content", text))
            self.answering = True

    def _step(self, out: list[tuple[str, str]], s: str, final: bool) -> tuple[str, bool]:
        """One step of the split over ``s``: what is left and whether it goes on (False: what is left waits)."""
        b = self.b
        if self.state == "out":
            hits = [(i, m) for m in (b.header, b.end) if (i := s.find(m)) >= 0]
            if not hits:
                k = 0 if final else held(s, (b.header, b.end))
                self._part(out, s[:len(s) - k])
                return s[len(s) - k:], False
            i, m = min(hits)
            self._part(out, s[:i])
            self.state = "head" if m == b.header else "out"
            return s[i + len(m):], True
        if self.state == "head":
            for kind, state in ((b.thought, "thought"), (b.text, "text")):
                if s.startswith(kind):
                    self.state, self._lead = state, True
                    if state == "thought" and not self.reasoning:
                        self._raw(out, kind)
                    return s[len(kind):], True
            if not final and any(k.startswith(s) for k in (b.thought, b.text)):
                return s, False
            self.state = "other"
            self._raw(out, b.header)
            return s, True
        if self.state == "other":  # (as it is, to its end marker)
            i = s.find(b.end)
            if i >= 0:
                self._raw(out, s[:i + len(b.end)])
                self.state = "out"
                return s[i + len(b.end):], True
            k = 0 if final else held(s, (b.end,))
            self._raw(out, s[:len(s) - k])
            return s[len(s) - k:], False
        thought = self.state == "thought"
        i = s.find(b.end)
        if i >= 0:
            self._part(out, s[:i].rstrip(), thought=thought)
            if thought and not self.reasoning:
                self._raw(out, b.end)
            self.state, self._lead = "out", True
            self.answering = self.answering or thought
            return s[i + len(b.end):], True
        k = 0 if final else held(s, (b.end,))
        body = s[:len(s) - k]
        text = body.rstrip()
        self._part(out, text, thought=thought)
        return "" if final else body[len(text):] + s[len(s) - k:], False

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        s, more = self._held + text, True
        while more:
            s, more = self._step(out, s, False)
        self._held = s
        return out

    def flush(self) -> list[tuple[str, str]]:
        """The text held back, at the end of the generation."""
        out: list[tuple[str, str]] = []
        s, more = self._held, True
        while more and s:
            s, more = self._step(out, s, True)
        self._held = ""
        return out


def thought_split(markers: tuple[str, str], blocks: Blocks | None, *, inside: bool = False,
                  ends: tuple[str, ...] = (), once: bool = False, lead: bool = True,
                  reasoning: bool = True) -> ReasoningSplitter | BlockSplitter:
    """The split of a model's text into its thoughts and answer: by its thought ``markers`` (``ReasoningSplitter``),
    or by its ``blocks`` (``BlockSplitter``; ``reasoning`` off: the thoughts in the content)."""
    if blocks is not None:
        return BlockSplitter(blocks, inside=inside, reasoning=reasoning, lead=lead)
    return ReasoningSplitter(markers, inside=inside, ends=ends, once=once, lead=lead)


class ThoughtBudget:
    """llama-server's reasoning budget over a generation's text, a token's at a time (``feed``): the tokens a thought
    may take (``budget``; < 0: any) and its end on demand (``asked``: the control route's ``reasoning_end``). ``due``:
    the thought is open and its end goes next (where the thought is: ``split``'s, a ``ReasoningSplitter`` or a
    dedicated ``BlockSplitter``); a thought that opens later gets a separate budget, and an end asked outside a
    thought lapses."""

    def __init__(self, split: ReasoningSplitter | BlockSplitter, *, budget: int):
        self.split = split
        self.budget = budget
        self.spent = 0  # (the open thought's tokens)
        self.asked = False

    def feed(self, text: str) -> None:
        """A token's text (forced text too: the thought's end)."""
        was = self.split.inside
        self.split.feed(text)
        self.spent = self.spent + 1 if was and self.split.inside else 0

    def due(self) -> bool:
        if not self.split.inside:
            self.asked = False
            return False
        return self.asked or 0 <= self.budget <= self.spent
