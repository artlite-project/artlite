"""A realtime conversation's sequence on the model (llama-server's realtime slot; ``realtime``'s websocket sessions
drive it): the text around the turns from the model's chat template (``Turns``) and the sequence of every token and
embedding row the model ran (``Conversation``), on the engine's worker."""

from typing import Any

import numpy as np

from ..sequence import Sequence
from .cache import common_prefix
from .engine import Engine

ROWS_STEP = 8  # (a user turn's audio rows run on the language model as they come, 8 at a time: 0.64 s)


class Turns:
    """The text around a realtime conversation's turns, from the model's chat template rendered with placeholder
    messages (llama-server's ``derive_turn_text``): ``first`` (the system turn - instructions, tools - and the first user
    turn's header), ``user`` / ``user_end`` (a later user turn's header, a user turn's end), ``reply`` (the generation
    prompt: the assistant's header and its empty thought), ``reply_end`` (after an answer), ``tool`` / ``tool_end``
    (around a function's output), ``audio`` / ``image`` (around a clip's rows, an image's tokens in a user turn:
    LFM3-Omni's vision markers; LFM3-Audio's clip bare)."""

    def __init__(self, model: Any, instructions: str, tools: list[dict[str, Any]] | None):
        u, r, x = "@@USER@@", "@@REPLY@@", "@@RESULT@@"
        system = [{"role": "system", "content": instructions}] if instructions else []

        def render(msgs: list[dict[str, Any]], gen: bool = False) -> str:  # (``Lfm.chat_text``)
            return model.chat_text(system + msgs, tools=tools or None, generation=gen)
        user = {"role": "user", "content": u}
        a, g = render([user]), render([user], True)
        c = render([user, {"role": "assistant", "content": r}])
        d = render([user, {"role": "assistant", "content": r}, user])
        pa, pc, pd = a.find(u), c.find(r), d.find(u, len(c))
        if min(pa, pc, pd) < 0 or not (c.startswith(a) and d.startswith(c) and g.startswith(a)):
            raise ValueError("the chat template's turns do not compose: no realtime conversation")
        self.first, self.user_end, self.reply = a[:pa], a[pa + len(u):], g[len(a):]
        self.reply_end, self.user = c[pc + len(r):], d[len(c):pd]
        self.tool = self.tool_end = ""
        if tools:
            fn = {"name": tools[0]["function"]["name"], "arguments": {}}
            call = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": fn}]}
            e = render([user, call])
            f = render([user, call, {"role": "tool", "tool_call_id": "call_0", "content": x}])
            pf = f.find(x, len(e))
            if pf < 0 or not f.startswith(e):
                raise ValueError("the chat template's tool results do not compose: no realtime tools")
            self.tool, self.tool_end = f[len(e):pf], f[pf + len(x):]

        def around(kind: str, marker: str) -> tuple[str, str]:
            t = render([{"role": "user", "content": [{"type": "text", "text": "@@A@@"}, {"type": kind},
                                                     {"type": "text", "text": "@@B@@"}]}])
            mid = t[t.find("@@A@@") + 5:t.find("@@B@@")]
            i = mid.find(marker)
            return (mid[:i], mid[i + len(marker):]) if i >= 0 else ("", "")
        self.audio, self.image = around("audio", "<|audio|>"), around("image", "<|image|>")


class Conversation:
    """A realtime conversation's sequence on the model (llama-server's realtime slot): every token and embedding row
    it ran, in order (``seq``): the turns' text (``Turns``), a user turn's audio rows as the speech encoder's stream
    gives them (``models.lfm3a.live.Stream``), an image's tokens, each reply's tokens and its frames' feedback rows
    (``models.lfm3a.speech.Reply``: the next turns see a spoken reply as the model said it). Its methods run on the
    engine's worker; the device holds the sequence through the prompt cache (``Engine.hold``: requests that ran
    between its turns displace it, the cache brings it back; ``Engine.ran``), and a turn's audio runs on the language
    model while it arrives (``ROWS_STEP`` rows at a time), so that a response starts with its last rows and the
    assistant's header."""

    def __init__(self, engine: Engine, model: Any):
        self.engine, self.model = engine, model
        self.seq = Sequence.text([], model.cfg.hidden_size)
        self.turns: Turns | None = None
        self.open = False  # (a user turn is being filled)
        self.after = ""  # (the text before the next turn: the last answer's end)
        self.n = 0  # (responses so far)
        self.results = False  # (a function's output since the last response: the next answers it)
        self.stream: Any = None  # (the open turn's audio: a speech encoder stream)

    def start(self, instructions: str, tools: list[dict[str, Any]] | None) -> None:
        """The session's system turn (fixed at its first audio, message or response)."""
        if self.turns is None:
            self.turns = Turns(self.model, instructions, tools)

    def _text(self, text: str) -> None:
        if text:
            self.seq = self.seq + self.model.tokenizer.encode(text)

    def _turn(self) -> None:
        if not self.open:
            self._text(self.turns.first if self.n == 0 else self.after + self.turns.user)
            self.after, self.open = "", True

    def sync(self) -> None:
        """The device holds the sequence: what the prompt cache keeps of it, the rest run (nothing when the device's
        sequence is it already: a prompt cache keeps a whole prompt only with its last token's logits)."""
        cached = self.engine.cached
        if len(cached) == len(self.seq) and common_prefix(cached, self.seq) == len(self.seq):
            return
        kept = self.engine.hold(self.seq)
        if kept < len(self.seq):
            self.engine.npu.prompt(self.seq[kept:])
        self.engine.ran(self.seq)

    def audio(self, samples: np.ndarray) -> None:
        """More of the user turn's audio (16 kHz): its rows run as they come."""
        self._turn()
        if self.stream is None:
            self._text(self.turns.audio[0])
            self.stream = self.model.audio_stream()
        self.stream.push(samples)
        if self.stream.available() >= ROWS_STEP:
            self._rows(self.stream.available())
            self.sync()

    def _rows(self, n: int) -> None:
        from ..lfm3a.host import clip_sequence
        self.seq = self.seq + clip_sequence(self.stream.take(n))

    def commit(self) -> bool:
        """The turn's audio ends (its last rows, the text after a clip); False: no audio."""
        if self.stream is None:
            return False
        self.stream.end()
        if self.stream.available():
            self._rows(self.stream.available())
        self.stream = None
        self._text(self.turns.audio[1])
        self.sync()
        return True

    def text(self, text: str) -> None:
        self._turn()
        self._text(text)

    def image(self, data: bytes) -> None:
        """An image in the user turn (a camera frame): its tokens between the template's text around one."""
        self._turn()
        self._text(self.turns.image[0])
        try:
            self.seq = self.seq + self.model.vision.sequence(data)
        finally:
            self.model._vision_ran()
        self._text(self.turns.image[1])

    def output(self, text: str) -> None:
        """A function's output (the last response called it)."""
        self._text(self.after + self.turns.tool + text + self.turns.tool_end)
        self.after, self.results = "", True

    def respond(self, mode: str, max_tokens: int, stats: Any, seed: int | None = None) -> Any:
        """The reply's ("text", token) and ("audio", samples) as they come (``Lfm.reply``; ``seed``: its speech codes'
        sampling); its tokens and frames stay in the sequence."""
        from ..lfm3a.speech import Reply
        if self.open:
            self._text(self.turns.user_end)
        self._text(self.turns.reply)
        self.seq = self.seq + self.model.opener(mode)
        stats.prompt = len(self.seq)
        kept = self.engine.hold(self.seq)
        reply = Reply()
        try:
            yield from self.model.reply(self.seq[kept:], mode=mode, seed=seed, max_tokens=max_tokens, reply=reply,
                                        stats=stats)
        finally:
            self.seq = self.seq + reply.sequence(self.model.cfg.hidden_size)
            self.engine.ran(self.seq)
            self.after, self.open, self.results = self.turns.reply_end, False, False
            self.n += 1
