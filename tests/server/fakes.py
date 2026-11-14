"""The scripted fake model of artlite-server's host tests (no NPU): a tokenizer of words and special tokens, a chat
template of role tags, and an ``NpuModel`` that replies with given pieces and keeps rewind / position semantics (a cache
group of 4 positions, or Qwen's exact continuation), its checkpoints and records the tokens they stand for; fake
speech encoders and speakers; and the harness the test modules share (``make``: a started engine, ``app_of``: the app
over it, ``live``: it under uvicorn, ``client_of``: it under a test client, the events of a stream). The engines
``make`` starts are stopped after each test (``conftest``)."""

import contextlib
import dataclasses
import json
import re
import socket
import threading
import time
import types
from collections.abc import Iterator
from typing import ClassVar

import httpx
import jinja2
import numpy as np
import uvicorn
from fastapi.testclient import TestClient

from artlite.decode.model import HostRecords
from models.model import Capabilities, Model
from models.npu_model import Checkpoint, NpuModel
from models.readouts import Readouts
from models.sampling import Mask, Penalties, Probs, Sampling, top_logits
from models.sequence import MEDIA, Sequence
from models.server.api import create_app
from models.server.decisions import Decisions
from models.server.engine import Engine
from models.server.settings import ModelInfo, Options

VOCAB = 1024
SPECIAL = ["<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<|channel>thought",
           "<channel|>"]
BOS, EOS = 0, 1
QWEN = ("<think>", "</think>")
GEMMA = ("<|channel>thought", "<channel|>")
TEMPLATE = "{% if enable_thinking %}<think>{% endif %}"  # (/props: the UI shows its thinking toggle)
TOOLS_TEMPLATE = TEMPLATE + "{# <tool_call> <function= <parameter= #}"  # (Qwen's tool-call syntax)


class FakeTokenizer:
    """Special tokens, then words with their leading white space; a new piece gets the next id."""

    def __init__(self) -> None:
        self.pieces = list(SPECIAL)
        self.ids = {p: i for i, p in enumerate(self.pieces)}
        self._special = re.compile("(" + "|".join(re.escape(s) for s in sorted(SPECIAL, key=len, reverse=True)) + ")")

    def id(self, piece: str) -> int:
        if piece not in self.ids:
            self.ids[piece] = len(self.pieces)
            self.pieces.append(piece)
            assert len(self.pieces) < VOCAB
        return self.ids[piece]

    def encode(self, text: str) -> list[int]:
        out = []
        for i, part in enumerate(self._special.split(text)):
            out += [self.id(part)] if i % 2 else [self.id(w) for w in re.findall(r"\s*\S+|\s+", part)]
        return out

    def decode_bytes(self, ids: list[int], *, special: bool = False) -> bytes:
        return "".join(self.pieces[i] if i < len(self.pieces) else "" for i in ids
                       if special or i not in (BOS, EOS)).encode()


class FakeModel(Model):
    """``reasoning``: the thought markers (QWEN: the prompt opens the thought when the model thinks); ``efforts``: the
    ``reasoning_effort`` levels its template takes (Qwen3.8's raises for another; None: any)."""

    name = "Fake"

    def __init__(self, *, reasoning: tuple[str, str] | None = None):
        self.tokenizer = FakeTokenizer()
        self.eos_token_ids = (EOS,)
        self.reasoning = reasoning
        self.thinks = reasoning is not None
        self.thought_first = reasoning == QWEN
        self.rendered: tuple[list[dict[str, object]], list[dict[str, object]] | None] = ([], None)
        self.preserved: bool | None = None  # (the last chat's preserve)
        self.thought: bool | None = None  # (the last chat's think)
        self.variables: dict[str, object] = {}  # (the last chat's template variables)
        self.efforts: tuple[str, ...] | None = None

    def encode(self, text: str) -> list[int]:
        return [BOS] + self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int]:
        """(tools: their names in a system turn; an assistant's tool calls after its content)"""
        effort = (variables or {}).get("reasoning_effort")
        if self.efforts is not None and effort is not None and effort not in self.efforts:
            raise jinja2.TemplateError(f"Unexpected reasoning effort {effort}")
        self.rendered, self.preserved, self.thought, self.variables = (messages, tools), preserve, think, variables or {}
        text = "<s>" + (f"<|system|>tools: {' '.join(t['function']['name'] for t in tools)}" if tools else "")
        for m in messages:
            calls = "".join(f" call {c['function']['name']} {json.dumps(c['function']['arguments'])}"
                            for c in m.get("tool_calls") or [])
            text += f"<|{m['role']}|>{m['content']}{calls}"
        text += "<|assistant|>"
        if self.reasoning == QWEN:
            text += "<think>\n" if think else "<think>\n\n</think>\n\n"
        return self.tokenizer.encode(text)

    def sampling(self, *, think: bool = False) -> Sampling:
        return Sampling(temp=0.6 if think else 0.7, top_k=8, top_p=0.9)


class FakeEncoder:
    """A vision encoder: an image's bytes as two embedding rows (their first 16 bytes), its threads recorded."""

    markup = ("<|system|>", "<|assistant|>")  # (two special tokens of the fake tokenizer)
    encoder = None  # (a speech encoder's: ``FakeLive``'s)

    def __init__(self) -> None:
        self.threads: list[str] = []

    def sequence(self, image: bytes) -> Sequence:
        self.threads.append(threading.current_thread().name)
        rows = np.frombuffer(bytes(image)[:16].ljust(16, b"\0"), np.uint8).astype(np.float32)
        return Sequence(np.full(2, MEDIA), np.ones(2, bool), rows.reshape(2, 8), np.zeros((2, 3), np.int64), 1)


class FakeVision(FakeModel):
    """A model with a vision encoder (``FakeEncoder``), its threads recorded."""

    def __init__(self) -> None:
        super().__init__()
        self.vision = FakeEncoder()
        self.threads = self.vision.threads

    def chat(self, messages: list[dict[str, object]], *, think: bool = False,
             tools: list[dict[str, object]] | None = None, preserve: bool = False,
             variables: dict[str, object] | None = None) -> list[int] | Sequence:
        self.rendered, self.thought = (messages, tools), think
        seq, text = Sequence.text([], 8), "<s>"
        for m in messages:
            text += f"<|{m['role']}|>"
            for part in m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]:
                if part["type"] in ("image", "audio", "video"):
                    enc = self.audio if part["type"] == "audio" else self.vision
                    seq, text = seq + self.tokenizer.encode(text) + enc.sequence(part[part["type"]]), ""
                else:
                    text += str(part["text"])
        return seq + self.tokenizer.encode(text + "<|assistant|>")


class FakeVideo(FakeVision):
    """A model that takes videos too (``models.lfm.lfm.Lfm.video_sequence``'s presence; a video's bytes as two rows)."""

    @property
    def capabilities(self) -> Capabilities:
        return dataclasses.replace(super().capabilities, video=True)

    def video_sequence(self, video: bytes) -> Sequence:
        return self.vision.sequence(video)


class FakeAudio(FakeVision):
    """A model with a speech encoder (``FakeEncoder``: a clip's bytes as two rows) and no vision encoder, and live ASR's
    template (``FakeLive``)."""

    def __init__(self) -> None:
        super().__init__()
        self.audio, self.vision = self.vision, None
        self.gguf = types.SimpleNamespace(meta={"tokenizer.chat_template.asr": ""})

    @property
    def capabilities(self) -> Capabilities:
        return dataclasses.replace(super().capabilities, live_asr="tokenizer.chat_template.asr" in self.gguf.meta)


class FakeLive:
    """``LiveAsr``'s interface: a push's text its samples' count and first value; commit "|", end "."; on the worker
    (its thread recorded), its sync (the engine's hold / ran) called on each step."""

    threads: ClassVar[list[str]] = []

    def __init__(self, model: object, npu: object, encoder: object, *, sync: tuple) -> None:
        self.sync, self.chunks, self.t_text = sync, 0, 0.0
        self.stream = types.SimpleNamespace(t_encode=0.0)

    def _step(self, out: str) -> str:
        FakeLive.threads.append(threading.current_thread().name)
        self.chunks += 1
        return out

    def push(self, x: np.ndarray) -> str:
        return self._step(f"[{len(x)} {x[0]:.2f}]")

    def commit(self) -> str:
        return self._step("|")

    def end(self) -> str:
        return self._step(".")


def fake_row(seq: list[int]) -> np.ndarray:
    """A fake model's output row (8 values) of a sequence's last token: from all of its tokens, as a causal model's."""
    s = sum((i + 1) * t for i, t in enumerate(seq))
    return np.cos(np.arange(8) * (s % 101 + 1) / 7).astype(np.float32)


class FakeNpu(NpuModel):
    """Replies with the pieces of ``replies`` (one list per generation, then an end-of-sequence token; default
    "ok"); ``seq``: the tokens at the cached positions; ``fail``: an exception for forward to raise. ``group``: rewinds
    keep multiples of it; ``exact``: all positions or none (Qwen: recurrent); ``chunk``: prompt tokens per dispatch;
    ``delay``: seconds per dispatch. A checkpoint is the tokens before its position, the records the tokens of their
    positions: a restore checks that the device holds the checkpoint's tokens."""

    def __init__(self, model: FakeModel, *, capacity: int = 512, group: int = 4, exact: bool = False, chunk: int = 8,
                 delay: float = 0.0, drafts: int = 0):
        self.tok = model.tokenizer
        self.eos_token_ids = model.eos_token_ids
        self.capacity, self.group, self.exact, self.chunk, self.delay = capacity, group, exact, chunk, delay
        self.recurrent = exact
        self.drafts = drafts  # (> 0: steps of drafts, every draft accepted: tokens run before they are yielded)
        self.DRAFTS = drafts
        self.generated: list[int | None] = []
        self.position = 0
        self.seq: list[int] = []
        self.replies: list[list[str]] = []
        self.calls: list[tuple] = []
        self.fail: Exception | None = None  # what forward raises
        self._next: Iterator[int] = iter(())

    def describe(self) -> str:
        return "fake"

    def reset(self) -> None:
        self.calls.append(("reset",))
        self.position, self.seq = 0, []

    def rewind(self, n: int) -> int:
        if self.exact:
            if n < self.position:
                self.reset()
            keep = self.position
        else:
            keep = min(n, self.position)
            keep -= keep % self.group
        self.calls.append(("rewind", n, keep))
        self.position, self.seq = keep, self.seq[:keep]
        return keep

    def checkpoint(self) -> Checkpoint:
        self.calls.append(("checkpoint", self.position))
        return Checkpoint(self.position, list(self.seq[:self.position]), 8 * self.position)

    def restore(self, position: int, checkpoint: Checkpoint | None = None) -> None:
        self.calls.append(("restore", position, checkpoint is not None))
        assert position <= len(self.seq)
        if checkpoint is None:
            assert position % self.group == 0 and not (self.exact and position), position
        else:
            assert checkpoint.position == position and checkpoint.state == self.seq[:position], \
                "the caches hold another sequence than the checkpoint's"
        self.position, self.seq = position, self.seq[:position]

    def read_records(self, start: int, end: int) -> list[HostRecords]:
        self.calls.append(("read", start, end))
        return [HostRecords(start, end, [(1, np.array(self.seq[start:end], np.int64).view(np.uint8).reshape(1, -1, 8))])]

    def write_records(self, records: list[HostRecords], start: int, end: int) -> None:
        self.calls.append(("write", start, end))
        r = records[0]
        assert r.start <= start <= len(self.seq) and end <= r.end
        tokens = r.parts[0][1].reshape(-1).view(np.int64)[start - r.start:end - r.start]
        self.seq[start:end] = [int(t) for t in tokens]
        self.position = min(self.position, start)  # (positions past the records written: another sequence's)

    def prompt(self, tokens: list[int], *, logits: bool = False, progress: object = None) -> np.ndarray | None:
        assert not logits
        self.calls.append(("prompt", len(tokens)))
        for i in range(0, len(tokens), self.chunk):
            self._run(tokens[i:i + self.chunk])
            if progress is not None:
                progress(min(i + self.chunk, len(tokens)))
        return None

    def outputs(self, tokens: list[int], at: object = None, *, norm: bool = True, progress: object = None,
                sets: object = None) -> np.ndarray:
        """(a causal model's rows: each a function of the tokens up to it, ``fake_row``; ``sets`` recorded)"""
        self.calls.append(("outputs", len(tokens)) if sets is None else ("outputs", len(tokens), tuple(sets)))
        if self.fail is not None:
            raise self.fail
        p0 = self.position
        for i in range(0, len(tokens), self.chunk):
            self._run(tokens[i:i + self.chunk])
        idx = np.arange(len(tokens)) if at is None else np.asarray(at) % len(tokens)
        return np.stack([fake_row(self.seq[:p0 + i + 1]) for i in idx])

    def _logits(self, token: int) -> np.ndarray:
        out = np.full(VOCAB, -100.0, np.float32)
        out[token] = 0.0
        return out

    def _run(self, tokens: list[int]) -> None:
        if self.position + len(tokens) > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        time.sleep(self.delay)
        self.seq[self.position:] = tokens
        self.position += len(tokens)
        self.dispatches += 1

    def forward(self, tokens: list[int], *, progress: object = None) -> np.ndarray:
        self.calls.append(("forward", len(tokens)))
        if self.fail is not None:
            raise self.fail
        for i in range(0, len(tokens), self.chunk):
            self._run(tokens[i:i + self.chunk])
            if progress is not None:
                progress(min(i + self.chunk, len(tokens)))
        pieces = self.replies.pop(0) if self.replies else ["ok"]
        self._next = iter([self.tok.id(p) for p in pieces] + [EOS])
        return self._logits(next(self._next))

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._run([token])
        logits = self._logits(next(self._next, EOS))
        return top_logits(logits, k) if penalties is None else penalties.top(logits, k)

    def step(self, token: int) -> np.ndarray:
        self._run([token])
        return self._logits(next(self._next, EOS))

    def generate(self, logits: np.ndarray, max_tokens: int, sampling: Sampling, prompt: object = (),
                 mask: Mask | None = None, probs: Probs | None = None, *, drafts: int | None = None) -> Iterator[int]:
        """(with ``drafts``: as a model's multi-token steps, the token and its drafts run, then they are yielded;
        ``generated``: each generation's ``drafts`` argument)"""
        self.generated.append(drafts)
        if not self.drafts:
            yield from super().generate(logits, max_tokens, sampling, prompt, mask, probs, drafts=drafts)
            return
        token, n = int(logits.argmax()), 1
        yield token
        while n != max_tokens and token not in self.eos_token_ids:
            new = []
            while len(new) <= self.drafts and (not new or new[-1] != EOS):
                new.append(next(self._next, EOS))
            self._run([token, *new[:-1]])
            for token in new:
                if probs is not None:
                    row = self._logits(token)
                    probs(row, token, 0.0, top_logits(row, 1))
                n += 1
                yield token

ENGINES: list[Engine] = []  # (stopped after each test: a worker left running keeps pytest from exiting)

def make(*, reasoning: tuple[str, str] | None = None, load: object = None, vision: bool = False, audio: bool = False,
         turns: tuple[str, str] | None = None, cache: dict[str, int] | None = None, **kw: object
         ) -> tuple[FakeModel, FakeNpu, Engine]:
    """A fake model's engine, started: ``turns``: the model's ``turn`` and ``user_turn`` (its prompt cache's marks),
    ``cache``: the prompt cache's options; ``kw``: the ``FakeNpu``'s."""
    model = FakeAudio() if audio else FakeVision() if vision else FakeModel(reasoning=reasoning)
    model.turn, model.user_turn = turns or (None, None)
    npu = FakeNpu(model, **kw)
    engine = Engine(load or (lambda: (model, npu)), warmup=False, **(cache or {}))
    ENGINES.append(engine)
    engine.start()
    return model, npu, engine


def app_of(engine: Engine, *, template: str = TEMPLATE, readouts: Readouts | None = None,
           decisions: Decisions | None = None, **options: object) -> object:
    """(``decisions``: the model is a decision model of that context)"""
    opts = {"alias": "fake", "n_ctx": 512, "sse_ping_interval": 0.05} | options
    return create_app(engine, Options(**opts), ModelInfo(path="/models/org/repo/fake.gguf", chat_template=template,
                                                         n_ctx_train=4096, n_vocab=VOCAB,
                                                         readouts=readouts or Readouts(),
                                                         decision=decisions.type if decisions else None,
                                                         decisions=decisions))

@contextlib.contextmanager
def client_of(app: dict[str, object] | None = None, /, **make_options: object
              ) -> Iterator[tuple[FakeModel, FakeNpu, Engine, TestClient]]:
    """A fake model's engine (``make``'s keywords) served under a test client, once its model is loaded (``app``:
    ``app_of``'s keywords)."""
    model, npu, engine = make(**make_options)
    with TestClient(app_of(engine, **(app or {}))) as client:
        assert engine.wait_ready(5)
        yield model, npu, engine, client

def events(r: httpx.Response) -> list[object]:
    """A stream's events: its data lines' JSON, "[DONE]", and ":" for a ping."""
    out: list[object] = []
    for block in r.text.split("\n\n"):
        if block == ":":
            out.append(":")
        elif block.startswith("data: "):
            data = block[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
        else:
            assert block == "", block
    return out


def deltas(evs: list[object], key: str = "content") -> str:
    return "".join(e["choices"][0]["delta"].get(key) or "" for e in evs if isinstance(e, dict) and e.get("choices"))


def chat(client: TestClient, content: str | list[dict[str, str]], **body: object) -> httpx.Response:
    messages = [{"role": "user", "content": content}] if isinstance(content, str) else content
    return client.post("/v1/chat/completions", json={"messages": messages, **body})

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                                                            "days": {"type": "integer"}},
                                                           "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}


def call_pieces(name: str, **args: str) -> list[str]:
    """A Qwen tool call as the fake model's pieces."""
    out = ["<tool_call>", f"\n<function={name}>"]
    for k, v in args.items():
        out += [f"\n<parameter={k}>", f"\n{v}", "\n</parameter>"]
    return out + ["\n</function>", "\n</tool_call>"]

@contextlib.contextmanager
def live(engine: Engine, **app: object) -> Iterator[str]:
    """The app of ``engine`` (``app``: ``app_of``'s keywords) under uvicorn on a free port: its URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app_of(engine, **app), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(10)

class FakeStream:
    """A speech encoder's stream: a row per 400 samples (the last partial one at the end), row k's values k + 0.5."""

    def __init__(self) -> None:
        self.n, self.taken, self.ended = 0, 0, False

    def push(self, samples: np.ndarray) -> None:
        self.n += len(samples)

    def end(self) -> None:
        self.ended = True

    def available(self) -> int:
        return (-(-self.n // 400) if self.ended else self.n // 400) - self.taken

    def take(self, k: int) -> np.ndarray:
        rows = np.arange(self.taken, self.taken + k, dtype=np.float32)[:, None] + np.full((k, 8), 0.5, np.float32)
        self.taken += k
        return rows


class FakeSpeaker(FakeAudio):
    """A model that speaks (``Lfm``'s interface): per request 3 chunks of 480 samples (their values the chunk's index
    / 10), its arguments recorded; its generator's mmproj config a namespace. A reply (``reply``, ``voice_reply``): the
    pieces of ``say`` (default " hello" " there") with 2 frames of 480 samples after the first, run on the fake NPU
    (``npu``) as an ``Lfm`` reply runs: each frame an ``<|audio_out|>`` with its feedback row."""

    def __init__(self) -> None:
        super().__init__()
        self.audio.gen = types.SimpleNamespace()
        self.spoken: list[dict[str, object]] = []
        self.cfg = types.SimpleNamespace(hidden_size=8)
        self.say = [" hello", " there"]
        self.npu: FakeNpu | None = None
        self.streams: list[FakeStream] = []

    @property
    def capabilities(self) -> Capabilities:
        return dataclasses.replace(super().capabilities, speech=True)

    def voice_reply(self, messages: list[dict[str, object]], **kw: object) -> Iterator[tuple[str, object]]:
        """Interleaved: the words " hello" " there", 2 chunks of 480 samples between them; the messages recorded."""
        self.spoken.append({"messages": messages, **kw})
        st, rp = kw.get("stats"), kw.get("reply")
        if st is not None:
            st.prompt = 7
        yield "text", self.tokenizer.id(" hello")
        for i in range(2):
            if st is not None:
                st.frames += 1
            yield "audio", np.full(480, i / 10, np.float32)
        yield "text", self.tokenizer.id(" there")
        if rp is not None:  # (the reply as said: a frame between the words)
            rp.ids += [self.tokenizer.id(" hello"), self.tokenizer.id("<|audio_out|>"), self.tokenizer.id(" there")]
            rp.rows[1] = np.full(8, 0.5, np.float32)

    def chat_text(self, messages: list[dict[str, object]], *, tools: list[dict[str, object]] | None = None,
                  generation: bool = True, **kw: object) -> str:
        """(FakeModel's format: media parts as <|kind|>)"""
        text = "<s>" + (f"<|system|>tools: {' '.join(t['function']['name'] for t in tools)}" if tools else "")
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                c = "".join(str(p.get("text", "")) if p.get("type") == "text" else f"<|{p['type']}|>" for p in c)
            calls = "".join(f" call {c_['function']['name']}" for c_ in m.get("tool_calls") or [])
            text += f"<|{m['role']}|>{c}{calls}"
        return text + ("<|assistant|>" if generation else "")

    def opener(self, mode: str) -> list[int]:
        return []

    def audio_stream(self) -> FakeStream:
        self.streams.append(FakeStream())
        return self.streams[-1]

    def reply(self, tokens: list[int] | Sequence, *, mode: str = "mixed", reply: object = None, stats: object = None,
              **kw: object) -> Iterator[tuple[str, object]]:
        self.spoken.append({"tokens": tokens, "mode": mode, **kw})
        self.npu.prompt(tokens)
        out = self.npu.position
        for i, piece in enumerate(self.say):
            t = self.tokenizer.id(piece)
            yield "text", t
            reply.ids.append(t)
            self.npu._run([t])
            for f in range(2 if i == 0 else 0):
                if stats is not None:
                    stats.frames += 1
                yield "audio", np.full(480, f / 10, np.float32)
                reply.rows[len(reply.ids)] = np.full(8, out + len(reply.ids), np.float32)
                reply.ids.append(self.tokenizer.id("<|audio_out|>"))
                self.npu._run([MEDIA])

    def tts(self, text: str, **kw: object) -> Iterator[np.ndarray]:
        self.spoken.append({"text": text, **kw})
        st = kw.get("stats")
        if st is not None:
            st.prompt = 5
        for i in range(3):
            if st is not None:
                st.frames += 1
            yield np.full(480, i / 10, np.float32)
