"""The generated text's filters, pure host state machines over a stream of text: stop strings (``StopMatcher``) and the
split of a thinking model's text into thoughts and answer (``ReasoningSplitter``). Text that may begin a word they look
for is held back until what follows decides (``held``)."""


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
    markers after it are answer. ``answering``: the answer has begun (a thought ended, or answer text came)."""

    def __init__(self, markers: tuple[str, str], *, inside: bool = False, ends: tuple[str, ...] = (),
                 once: bool = False):
        self.open, self.close = markers
        self.inside = inside
        self.ends, self.once = ends, once
        self._held = ""
        self._lead = True  # (a part begins: its leading white space is dropped)
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
