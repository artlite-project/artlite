"""artlite-server on the host (no NPU): its routes through FastAPI's TestClient and a live uvicorn (the ``openai``
client), with a scripted fake model - a tokenizer of words and special tokens, a chat template of role tags, and an
``NpuModel`` that replies with given pieces and keeps rewind / position semantics (a cache group of 4 positions, or
Qwen's exact continuation). Serving the fake model with the web UI, for a manual check in a browser:

    python tests/rtc/test_server.py [--port 8080] [--thinking]
"""

import argparse
import base64
import contextlib
import json
import re
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import numpy as np
import openai
import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient

from models.common import MEDIA_MARKER, Model, NpuModel, Penalties, Sampling
from models.qwen35.sequence import MEDIA, Sequence
from models.server import ui
from models.server.api import ModelInfo, Options, create_app
from models.server.engine import Engine, Job, ReasoningSplitter, StopMatcher

VOCAB = 1024
SPECIAL = ["<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<|channel>thought",
           "<channel|>"]
BOS, EOS = 0, 1
QWEN = ("<think>", "</think>")
GEMMA = ("<|channel>thought", "<channel|>")
TEMPLATE = "{% if enable_thinking %}<think>{% endif %}"  # (/props: the UI shows its thinking toggle)


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
        return "".join(self.pieces[i] for i in ids if special or i not in (BOS, EOS)).encode()


class FakeModel(Model):
    """``reasoning``: the thought markers (QWEN: the prompt opens the thought when the model thinks)."""

    name = "Fake"

    def __init__(self, *, reasoning: tuple[str, str] | None = None):
        self.tokenizer = FakeTokenizer()
        self.eos_token_ids = (EOS,)
        self.reasoning = reasoning
        self.thinks = reasoning is not None

    def encode(self, text: str) -> list[int]:
        return [BOS] + self.tokenizer.encode(text)

    def chat(self, messages: list[dict[str, object]], *, think: bool = False) -> list[int]:
        text = "<s>" + "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"
        if self.reasoning == QWEN:
            text += "<think>\n" if think else "<think>\n\n</think>\n\n"
        return self.tokenizer.encode(text)

    def sampling(self, *, think: bool = False) -> Sampling:
        return Sampling(temp=0.6 if think else 0.7, top_k=8, top_p=0.9)


class FakeVision(FakeModel):
    """A model with a vision encoder: an image part's bytes as two embedding rows (their first 16 bytes), the encoder's
    threads recorded."""

    vision = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[str] = []

    def chat(self, messages: list[dict[str, object]], *, think: bool = False) -> list[int] | Sequence:
        seq, text = Sequence.text([], 8), "<s>"
        for m in messages:
            text += f"<|{m['role']}|>"
            for part in m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]:
                if part["type"] == "image":
                    self.threads.append(threading.current_thread().name)
                    rows = np.frombuffer(bytes(part["image"])[:16].ljust(16, b"\0"), np.uint8).astype(np.float32)
                    img = Sequence(np.full(2, MEDIA), np.ones(2, bool), rows.reshape(2, 8), np.zeros((2, 3), np.int64), 1)
                    seq, text = seq + self.tokenizer.encode(text) + img, ""
                else:
                    text += str(part["text"])
        return seq + self.tokenizer.encode(text + "<|assistant|>")


class FakeNpu(NpuModel):
    """Replies with the pieces of ``replies`` (one list per generation, then an end-of-sequence token; default
    "ok"); ``seq``: the tokens at the cached positions; ``fail``: an exception for forward to raise. ``group``: rewinds keep multiples of it; ``exact``: all
    positions or none (Qwen); ``chunk``: prompt tokens per dispatch; ``delay``: seconds per dispatch."""

    def __init__(self, model: FakeModel, *, capacity: int = 512, group: int = 4, exact: bool = False, chunk: int = 8,
                 delay: float = 0.0):
        self.tok = model.tokenizer
        self.eos_token_ids = model.eos_token_ids
        self.capacity, self.group, self.exact, self.chunk, self.delay = capacity, group, exact, chunk, delay
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

    def _logits(self, token: int) -> torch.Tensor:
        out = torch.full((VOCAB,), -100.0)
        out[token] = 0.0
        return out

    def _run(self, tokens: list[int]) -> None:
        if self.position + len(tokens) > self.capacity:
            raise ValueError(f"the KV cache holds {self.capacity} positions")
        time.sleep(self.delay)
        self.seq[self.position:] = tokens
        self.position += len(tokens)
        self.dispatches += 1

    def forward(self, tokens: list[int], *, progress: object = None) -> torch.Tensor:
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

    def top_k(self, token: int, k: int, penalties: Penalties | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self._run([token])
        logits = self._logits(next(self._next, EOS))
        return torch.topk(logits, k) if penalties is None else penalties.top(logits, k)


ENGINES: list[Engine] = []  # (stopped after each test: a worker left running keeps pytest from exiting)


@pytest.fixture(autouse=True)
def _stop_engines() -> Iterator[None]:
    yield
    while ENGINES:
        e = ENGINES.pop()
        e.stop()
        e.join(5)


def make(*, reasoning: tuple[str, str] | None = None, load: object = None, vision: bool = False, **kw: object
         ) -> tuple[FakeModel, FakeNpu, Engine]:
    model = FakeVision() if vision else FakeModel(reasoning=reasoning)
    npu = FakeNpu(model, **kw)
    engine = Engine(load or (lambda: (model, npu)), warmup=False)
    ENGINES.append(engine)
    engine.start()
    return model, npu, engine


def app_of(engine: Engine, **options: object) -> object:
    opts = {"alias": "fake", "n_ctx": 512, "sse_ping_interval": 0.05} | options
    return create_app(engine, Options(**opts), ModelInfo(path="/models/org/repo/fake.gguf", chat_template=TEMPLATE,
                                                         n_ctx_train=4096, n_vocab=VOCAB))


@pytest.fixture
def fake() -> Iterator[tuple[FakeModel, FakeNpu, Engine, TestClient]]:
    model, npu, engine = make()
    with TestClient(app_of(engine)) as client:
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


# ----------------------------------------------------------------------
# Streams and responses
# ----------------------------------------------------------------------


def test_stream(fake: tuple) -> None:
    _model, npu, engine, client = fake
    npu.replies = [["Hello", " world", "!"]]
    r = chat(client, "Hi", stream=True, return_progress=True, timings_per_token=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = events(r)
    assert evs[-1] == "[DONE]"
    first = evs[0]
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"] == {"role": "assistant",
                                                                                           "content": None}
    assert deltas(evs) == "Hello world!"
    progress = [e["prompt_progress"] for e in evs[:-1] if "prompt_progress" in e]
    assert progress[0]["processed"] == 0 and progress[-1]["processed"] == progress[-1]["total"]
    end = evs[-2]
    assert end["choices"][0]["finish_reason"] == "stop" and end["choices"][0]["delta"] == {}
    t = end["timings"]
    assert t["predicted_n"] == 4 and t["prompt_n"] == progress[-1]["total"] and t["cache_n"] == 0
    assert all("timings" in e for e in evs[1:-1] if isinstance(e, dict) and e["choices"][0]["delta"].get("content"))
    assert len({e["id"] for e in evs[:-1]}) == 1 and all(e["model"] == "fake" for e in evs[:-1])
    assert engine.cached == npu.seq[:npu.position]


def test_response_and_usage(fake: tuple) -> None:
    model, npu, _engine, client = fake
    npu.replies = [["One", " two", " three", " four"]]
    r = chat(client, "Count", max_tokens=3)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "One two three"}
    assert body["choices"][0]["finish_reason"] == "length"
    u = body["usage"]
    assert u["completion_tokens"] == 3 and u["total_tokens"] == u["prompt_tokens"] + 3
    assert u["prompt_tokens"] == len(model.chat([{"role": "user", "content": "Count"}]))
    evs = events(chat(client, "Count", stream=True, stream_options={"include_usage": True}))
    assert evs[-2]["choices"] == [] and evs[-2]["usage"]["prompt_tokens"] == u["prompt_tokens"]
    assert evs[-3]["choices"][0]["finish_reason"] == "stop"


def test_completions(fake: tuple) -> None:
    _model, npu, _engine, client = fake
    npu.replies = [["A", "B"], [" C"], [" D"]]
    r = client.post("/v1/completions", json={"prompt": "Once upon"})
    body = r.json()
    assert body["object"] == "text_completion" and body["choices"][0]["text"] == "AB"
    assert body["usage"]["prompt_tokens"] == 3  # (BOS and two words)
    r = client.post("/completion", json={"prompt": [BOS, 5], "n_predict": 4})
    body = r.json()
    assert body["content"] == " C" and body["stop"] is True and body["stop_type"] == "eos"
    assert body["tokens_evaluated"] == 2 and body["tokens_cached"] == npu.position
    lines = [json.loads(b[6:]) for b in client.post("/completion", json={"prompt": "x", "stream": True}).text.split(
        "\n\n") if b.startswith("data: ")]
    assert lines[0]["content"] == " D" and lines[-1]["stop"] is True  # (no [DONE] in the native stream)


def test_penalties() -> None:
    """The penalties of the command line and of a request, over the prompt's tokens too: a reply of a prompt's word
    whose logit the frequency penalty pushes below the others' gives another token."""
    _model, npu, engine = make()
    with TestClient(app_of(engine, repeat_penalty=1.2)) as client:
        assert engine.wait_ready(5)
        assert client.get("/props").json()["default_generation_settings"]["params"]["repeat_penalty"] == 1.2
        body = {"prompt": "hello there", "temperature": 0}
        npu.replies = [["hello"]]
        r = client.post("/completion", json=body | {"repeat_last_n": 16, "presence_penalty": 0.5}).json()
        g = r["generation_settings"]
        assert r["content"] == "hello"  # (a penalized logit of 0 stays 0)
        assert (g["repeat_penalty"], g["repeat_last_n"], g["presence_penalty"], g["frequency_penalty"]) == \
            (1.2, 16, 0.5, 0.0)
        npu.replies = [["hello"]]
        assert client.post("/completion", json=body | {"frequency_penalty": 150}).json()["content"] != "hello"
        npu.replies = [["hello"]]
        assert client.post("/completion", json=body | {"frequency_penalty": 150, "repeat_last_n": 0}
                           ).json()["content"] == "hello"


# ----------------------------------------------------------------------
# Reasoning, stop strings
# ----------------------------------------------------------------------


def test_reasoning_qwen() -> None:
    """The prompt opens the thought; the model closes it."""
    _model, npu, engine = make(reasoning=QWEN)
    with TestClient(app_of(engine)) as client:
        npu.replies = [["Let", " me", " think", "\n", "</think>", "\n\n", "The", " answer"]] * 3
        on = {"chat_template_kwargs": {"enable_thinking": True}}
        evs = events(chat(client, "Q", stream=True, **on))
        assert deltas(evs, "reasoning_content") == "Let me think" and deltas(evs) == "The answer"
        body = chat(client, "Q", **on).json()["choices"][0]["message"]
        assert body == {"role": "assistant", "content": "The answer", "reasoning_content": "Let me think"}
        raw = chat(client, "Q", reasoning_format="none", **on).json()["choices"][0]["message"]
        assert raw["content"] == "Let me think\n</think>\n\nThe answer" and "reasoning_content" not in raw
        assert "<think>\n\n</think>" in client.post("/apply-template", json={
            "messages": [{"role": "user", "content": "Q"}]}).json()["prompt"]  # (-rea off: an empty thought)


def test_reasoning_gemma() -> None:
    """The model opens and closes the thought, and may end with a stray close marker."""
    _model, npu, engine = make(reasoning=GEMMA)
    with TestClient(app_of(engine, reasoning="on")) as client:
        npu.replies = [["<|channel>thought", "\n", "Hmm", ",", " yes", "\n", "<channel|>", "Yes", ".", "<channel|>"]]
        evs = events(chat(client, "Q", stream=True))
        assert deltas(evs, "reasoning_content") == "Hmm, yes" and deltas(evs) == "Yes."


def test_splitter_and_stops() -> None:
    s = ReasoningSplitter(QWEN, inside=True)
    parts = [p for t in ["\nA", " b\n", "\n</th", "ink>", "\n", "\nC", "\n\nD", "\n"] for p in s.feed(t)] + s.flush()
    assert parts == [("reasoning_content", "A"), ("reasoning_content", " b"), ("content", "C"), ("content", "\n\nD")]
    s = ReasoningSplitter(GEMMA)
    parts = [p for t in ["Plain <|chan", "nel", " not a marker"] for p in s.feed(t)] + s.flush()
    assert "".join(t for _, t in parts) == "Plain <|channel not a marker" and {k for k, _ in parts} == {"content"}
    m = StopMatcher(["STOP", "\n\n"])
    assert [m.feed(t) for t in ["Hello", " wor", "ld ST", "O"]] == [("Hello", False), (" wor", False), ("ld ", False),
                                                                     ("", False)]
    assert m.feed("P more") == ("", True) and m.word == "STOP"
    m = StopMatcher(["STOP"])
    assert m.feed("ST") == ("", False) and m.feed("ART") == ("START", False)


def test_stop_strings(fake: tuple) -> None:
    _model, npu, engine, client = fake
    npu.replies = [["Hello", " wor", "ld", " STO", "P", " more"], ["ST", "ART", " ok"]]
    body = chat(client, "Hi", stop="STOP").json()
    assert body["choices"][0]["message"]["content"] == "Hello world " and body["choices"][0]["finish_reason"] == "stop"
    assert npu.position == len(engine.cached) and engine.cached == npu.seq[:npu.position]
    evs = events(chat(client, "Hi", stream=True, stop=["STOP"]))
    assert deltas(evs) == "START ok"


# ----------------------------------------------------------------------
# The prompt cache
# ----------------------------------------------------------------------


def test_prompt_cache(fake: tuple) -> None:
    model, npu, _engine, client = fake
    turn1 = [{"role": "user", "content": "one two three four five six seven"}]
    npu.replies = [["Reply", " one"], ["Reply", " two"], ["x"], ["y"]]
    b1 = chat(client, turn1).json()
    assert b1["timings"]["cache_n"] == 0
    turn2 = turn1 + [{"role": "assistant", "content": "Reply one"}, {"role": "user", "content": "more"}]
    b2 = chat(client, turn2).json()
    ids2 = model.chat(turn2)
    rewind = [c for c in npu.calls if c[0] == "rewind"][-1]
    kept = rewind[2]
    assert b2["timings"]["cache_n"] == kept > 0 and kept % npu.group == 0 and rewind[1] - kept < npu.group
    assert b2["timings"]["prompt_n"] == len(ids2) - kept and b2["usage"]["prompt_tokens_details"]["cached_tokens"] == kept
    assert npu.seq[:len(ids2)] == ids2
    # the same prompt again: all cached but its last token, which runs again
    b3 = chat(client, turn2).json()
    assert [c for c in npu.calls if c[0] == "rewind"][-1][1] == len(ids2) - 1 and b3["timings"]["prompt_n"] >= 1
    # without the cache
    b4 = chat(client, turn2, cache_prompt=False).json()
    assert b4["timings"]["cache_n"] == 0 and npu.calls[-2] == ("reset",)


def _image(data: bytes) -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def test_images() -> None:
    """image_url parts (data URIs) reach the model's chat as image parts, on the NPU worker (the encoder's dispatches
    between jobs); the prompt cache tells images apart by their rows; vision in /props and /v1/models."""
    model, _npu, engine = make(vision=True)
    with TestClient(app_of(engine)) as client:
        assert engine.wait_ready(5)
        turn1 = [{"role": "user", "content": [{"type": "text", "text": "look"}, _image(b"cat" * 8),
                                              {"type": "text", "text": "what is it"}]}]
        b1 = chat(client, turn1).json()
        assert b1["timings"]["cache_n"] == 0 and model.threads == ["artlite-npu"]
        turn2 = turn1 + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
        b2 = chat(client, turn2).json()
        assert b2["timings"]["cache_n"] >= 8  # (past the image)
        other = [{"role": "user", "content": [{"type": "text", "text": "look"}, _image(b"dog" * 8),
                                              {"type": "text", "text": "what is it"}]}]
        b3 = chat(client, other).json()
        assert b3["timings"]["cache_n"] <= 3 and len(model.threads) == 3  # (the image's rows differ: cut before it)
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]}).status_code == 400
        p = client.get("/props").json()
        assert p["modalities"]["vision"] is True
        m = client.get("/v1/models").json()
        assert "multimodal" in m["models"][0]["capabilities"] and "image" in m["data"][0]["architecture"]["input_modalities"]
        prompt = client.post("/apply-template", json={"messages": turn1}).json()["prompt"]
        assert prompt == f"<s><|user|>look\n{MEDIA_MARKER}\nwhat is it<|assistant|>"


def test_images_refused(fake: tuple) -> None:
    """A model without a vision encoder refuses images."""
    client = fake[3]
    r = chat(client, [{"role": "user", "content": [_image(b"x"), {"type": "text", "text": "hi"}]}])
    assert r.status_code == 400 and "text only" in r.json()["error"]["message"]


def test_prompt_images() -> None:
    """Model.prompt places images at the text's media markers, else before it."""
    model = FakeVision()
    got = model.prompt(f"a {MEDIA_MARKER} b {MEDIA_MARKER}", images=[b"1", b"2"])
    want = model.chat([{"role": "user", "content": [{"type": "text", "text": "a "}, {"type": "image", "image": b"1"},
                                                    {"type": "text", "text": " b "}, {"type": "image", "image": b"2"}]}])
    assert got.cache_keys() == want.cache_keys()
    first = model.prompt("describe", images=[b"1"])
    assert first.cache_keys() == model.chat([{"role": "user", "content": [
        {"type": "image", "image": b"1"}, {"type": "text", "text": "describe"}]}]).cache_keys()
    with pytest.raises(ValueError, match="marker"):
        model.prompt(f"a {MEDIA_MARKER}", images=[b"1", b"2"])
    with pytest.raises(ValueError, match="vision"):
        FakeModel().prompt("hi", images=[b"1"])


def test_prompt_cache_exact() -> None:
    """Qwen keeps an exact continuation of the cached tokens, else nothing."""
    model, npu, engine = make(exact=True)
    with TestClient(app_of(engine)) as client:
        npu.replies = [["a"], ["b"], ["c"]]
        ids = model.encode("one two three")
        first = client.post("/completion", json={"prompt": ids}).json()
        n = first["tokens_cached"]
        assert n == len(ids) + 1  # (the reply's token ran; the end-of-sequence token did not)
        cont = client.post("/completion", json={"prompt": ids + model.tokenizer.encode("a more")}).json()
        assert cont["timings"]["cache_n"] == n
        other = client.post("/completion", json={"prompt": model.encode("one two four")}).json()
        assert other["timings"]["cache_n"] == 0 and ("reset",) in npu.calls


# ----------------------------------------------------------------------
# Cancelling, the queue
# ----------------------------------------------------------------------


def run_job(engine: Engine, ids: list[int], *, until: object = None, **kw: object) -> tuple[Job, list]:
    """A job submitted directly, its events collected; ``until(events)``: set its cancel flag once true."""
    got: list[tuple[str, dict]] = []
    done = threading.Event()

    def emit(kind: str, data: dict) -> None:
        got.append((kind, data))
        if until is not None and until(got):
            job.cancel.set()
        if kind in ("done", "error", "cancelled"):
            done.set()

    job = Job(ids=ids, emit=emit, sampling=Sampling(greedy=True), **kw)
    engine.submit(job)
    assert done.wait(10)
    return job, got


def test_cancel_between_tokens() -> None:
    model, npu, engine = make(delay=0.002)
    assert engine.wait_ready(5)
    npu.replies = [[f" w{i}" for i in range(100)]]
    ids = model.encode("tell me a long story")
    _job, got = run_job(engine, ids, until=lambda g: sum(k == "delta" for k, _ in g) == 5)
    assert got[-1][0] == "cancelled" and sum(k == "delta" for k, _ in got) == 5
    assert npu.position < len(ids) + 10 and engine.cached == npu.seq[:npu.position]
    npu.replies = [["fine"]]
    _, got = run_job(engine, ids + [5])
    assert got[-1][0] == "done" and got[-1][1]["timings"]["cache_n"] > 0  # (the next request continues)


def test_cancel_between_chunks() -> None:
    model, npu, engine = make(chunk=4, delay=0.002)
    assert engine.wait_ready(5)
    ids = model.encode(" ".join(f"w{i}" for i in range(40)))
    _job, got = run_job(engine, ids, until=lambda g: g[-1][0] == "progress" and g[-1][1]["processed"] >= 8,
                       progress=True)
    assert got[-1][0] == "cancelled" and npu.position == 8 and engine.cached == ids[:8]


def test_cancel_by_conversation(fake: tuple) -> None:
    """The UI's Stop: DELETE /v1/stream?conv_id= for the request with its X-Conversation-Id."""
    _model, npu, engine, client = fake
    npu.delay = 0.01
    npu.replies = [[f" w{i}" for i in range(500)]]
    out: dict[str, httpx.Response] = {}
    t = threading.Thread(target=lambda: out.setdefault("r", client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "go"}], "stream": True},
        headers={"X-Conversation-Id": "conv-1"})))
    t.start()
    while engine.current is None or engine.current.n_gen < 3:
        time.sleep(0.005)
    assert client.delete("/v1/stream", params={"conv_id": "conv-1"}).json()["cancelled"] == 1
    t.join(10)
    evs = events(out["r"])
    assert evs[-1] != "[DONE]" and evs[-1]["error"]["message"] == "the request was cancelled"
    assert 3 <= len(deltas(evs).split()) < 500


def test_fifo(fake: tuple) -> None:
    """Two requests at a time: the second waits (with pings), then runs."""
    _model, npu, engine, client = fake
    npu.delay = 0.01
    npu.replies = [[f" a{i}" for i in range(30)], [" b"]]
    out: dict[str, httpx.Response] = {}

    def post(key: str, text: str) -> None:
        out[key] = chat(client, text, stream=True)

    first = threading.Thread(target=post, args=("first", "one"))
    first.start()
    while engine.current is None:
        time.sleep(0.002)
    second = threading.Thread(target=post, args=("second", "two"))
    second.start()
    first.join(10)
    second.join(10)
    assert deltas(events(out["first"])).startswith(" a0 a1") and deltas(events(out["second"])) == " b"
    assert ":" in events(out["second"])  # (pings while it waited)
    forwards = [c for c in npu.calls if c[0] == "forward"]
    assert len(forwards) == 2


# ----------------------------------------------------------------------
# Errors, loading, keys
# ----------------------------------------------------------------------


def test_errors(fake: tuple) -> None:
    _model, npu, _engine, client = fake
    r = chat(client, " ".join(["word"] * 600))
    e = r.json()["error"]
    assert r.status_code == 400 and e["type"] == "exceed_context_size_error" and e["n_ctx"] == 512
    assert e["n_prompt_tokens"] > 512
    r = chat(client, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}])
    assert r.status_code == 400 and "text only" in r.json()["error"]["message"]
    assert chat(client, "x", n=2).status_code == 400
    assert chat(client, "x", model="gpt-4o").status_code == 200  # (as llama-server: any name)
    assert chat(client, "x", stream=None, n=None, return_progress=None).status_code == 200
    # values that sample nothing or overflow keep the most likely token, and the worker survives them
    for extreme in ({"top_p": 0.0}, {"min_p": 1.5}, {"top_k": 50000}, {"seed": 2**64 + 5}, {"temperature": -1}):
        npu.replies = [["fine"]]
        r = chat(client, "x", **extreme)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "fine", extreme
    assert client.post("/completion", json={"prompt": [BOS, -5, 3]}).status_code == 400
    assert client.post("/detokenize", json={"tokens": [VOCAB]}).json()["error"]["type"] == "invalid_request_error"
    assert client.post("/v1/chat/completions", json={"messages": "x"}).json()["error"]["type"] == \
        "invalid_request_error"
    assert client.post("/v1/embeddings", json={}).status_code == 501
    assert client.get("/tools").status_code == 403
    for bad in ({"repeat_penalty": 0}, {"repeat_last_n": -1}):
        r = chat(client, "x", **bad)
        assert r.status_code == 400 and next(iter(bad)) in r.json()["error"]["message"], bad
    # the text parts of a content list, and the fields the UI sends (accepted)
    npu.replies = [["fine"]]
    parts = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    r = chat(client, parts, stream=True, reasoning_format="auto", repeat_penalty=1.1, dry_multiplier=0.0,
             samplers=["top_k"], tools=[{"type": "function", "function": {"name": "get_datetime"}}],
             reasoning_control=True, sse_ping_interval=1)
    assert r.status_code == 200 and deltas(events(r)) == "fine"


def test_model_failure(fake: tuple) -> None:
    _model, npu, engine, client = fake
    npu.fail = RuntimeError("DispatchError: TIMEOUT")
    r = chat(client, "a b c d e f")
    assert r.status_code == 500 and r.json()["error"] == {"code": 500, "type": "server_error",
                                                          "message": "RuntimeError: DispatchError: TIMEOUT"}
    evs = events(chat(client, "a b c d e f", stream=True))
    assert evs[-1]["error"]["type"] == "server_error"
    npu.fail = None
    n = len(npu.calls)
    assert chat(client, "a").status_code == 200 and engine.cached == npu.seq[:npu.position]
    assert ("reset",) in npu.calls[n:]  # (after a failure the caches start over)

    def broken(n: int) -> int:
        raise RuntimeError("rewind failed")

    npu.rewind = broken  # (a failure before the prompt runs: the worker survives it too)
    assert chat(client, "a").json()["error"]["message"] == "RuntimeError: rewind failed"
    del npu.rewind
    npu.replies = [["back"]]
    assert chat(client, "a").json()["choices"][0]["message"]["content"] == "back"


def test_loading() -> None:
    gate = threading.Event()
    model = FakeModel()
    npu = FakeNpu(model)

    def load() -> tuple[FakeModel, FakeNpu]:
        gate.wait(10)
        return model, npu

    _, _, engine = make(load=load)
    with TestClient(app_of(engine)) as client:
        assert client.get("/health").status_code == 503
        r = client.get("/props")
        assert r.status_code == 503 and r.json()["error"]["message"] == "Loading model"
        assert chat(client, "x").status_code == 503
        assert client.get("/v1/models").status_code == 200
        gate.set()
        assert engine.wait_ready(5)
        assert client.get("/health").json() == {"status": "ok"}
    failed = []
    engine = Engine(lambda: 1 / 0, on_exit=lambda: failed.append(1))
    ENGINES.append(engine)
    engine.start()
    engine.join(5)
    assert failed and engine.error.startswith("ZeroDivisionError") and not engine.ready


def test_api_keys(tmp_path: object) -> None:
    _model, _npu, engine = make()
    ui_dir = tmp_path / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<!doctype html><title>ui</title>")
    (ui_dir / "app.js").write_text("console.log(1)")
    app = app_of(engine, api_keys=frozenset({"k1", "k2"}), ui=lambda a: ui.mount(a, str(ui_dir)))
    with TestClient(app) as client:
        assert engine.wait_ready(5)
        r = client.get("/props")
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        assert client.get("/props", headers={"Authorization": "Bearer k2"}).status_code == 200
        assert client.get("/props", headers={"X-Api-Key": "k1"}).status_code == 200
        assert client.get("/props", headers={"Authorization": "Bearer k3"}).status_code == 401
        assert client.get("/health").status_code == 200
        r = client.get("/")
        assert r.status_code == 200 and "<title>ui</title>" in r.text and r.headers["cache-control"] == "no-cache"
        assert client.get("/app.js").text == "console.log(1)"
        assert client.get("/v1/nothing", headers={"X-Api-Key": "k1"}).status_code == 404
        r = client.options("/v1/chat/completions", headers={"Origin": "http://x", "Access-Control-Request-Method": "POST"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"


# ----------------------------------------------------------------------
# What the web UI reads, the utilities
# ----------------------------------------------------------------------


def test_props_and_models(fake: tuple) -> None:
    _model, npu, _engine, client = fake
    p = client.get("/props", params={"autoload": "false"}).json()
    g = p["default_generation_settings"]
    assert g["n_ctx"] == npu.capacity and g["params"]["temperature"] == 0.7 and g["params"]["top_k"] == 8
    assert g["params"]["top_p"] == 0.9 and g["params"]["min_p"] == 0.0 and g["params"]["max_tokens"] == -1
    assert g["params"]["repeat_penalty"] == 1.0 and g["params"]["repeat_last_n"] == 64
    assert g["params"]["samplers"][0] == "penalties"
    assert p["modalities"] == {"vision": False, "audio": False, "video": False} and "role" not in p
    assert p["chat_template"] == TEMPLATE and p["model_path"].endswith("fake.gguf") and p["total_slots"] == 1
    m = client.get("/v1/models").json()
    assert m["data"][0]["id"] == "fake" and m["data"][0]["meta"]["n_ctx_train"] == 4096 and m["models"]
    s = client.get("/slots").json()
    assert s[0]["n_ctx"] == npu.capacity and s[0]["is_processing"] is False
    assert client.post("/v1/streams/lookup", json={"conversation_ids": ["a"]}).json() == []
    assert client.get("/v1/stream", params={"conv_id": "a", "from": 0}).status_code == 404


def test_utilities(fake: tuple) -> None:
    model, _npu, _engine, client = fake
    t = client.post("/tokenize", json={"content": "hello world"}).json()["tokens"]
    assert t == model.tokenizer.encode("hello world")
    t2 = client.post("/tokenize", json={"content": "hello world", "add_special": True, "with_pieces": True}).json()
    assert [p["piece"] for p in t2["tokens"]] == ["<s>", "hello", " world"]
    assert client.post("/detokenize", json={"tokens": t}).json() == {"content": "hello world"}
    prompt = client.post("/apply-template", json={"messages": [{"role": "user", "content": "hi"}]}).json()["prompt"]
    assert prompt == "<s><|user|>hi<|assistant|>"


@contextlib.contextmanager
def live(engine: Engine) -> Iterator[str]:
    """The app of ``engine`` under uvicorn on a free port: its URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app_of(engine), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(10)


def test_openai_client() -> None:
    """The openai package against a live server of the fake model."""
    _model, npu, engine = make()
    with live(engine) as url:
        client = openai.OpenAI(base_url=f"{url}/v1", api_key="none")
        assert [m.id for m in client.models.list()] == ["fake"]
        npu.replies = [["Hello", " there"], ["Bye"]]
        stream = client.chat.completions.create(model="fake", messages=[{"role": "user", "content": "Hi"}],
                                                stream=True)
        assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "Hello there"
        r = client.chat.completions.create(model="fake", messages=[{"role": "user", "content": "Hi"}], max_tokens=8)
        assert r.choices[0].message.content == "Bye" and r.usage.completion_tokens == 2


@pytest.mark.parametrize("stream", [True, False])
def test_disconnect(stream: bool) -> None:
    """A client that goes away (the UI's Stop aborts its request) cancels its generation."""
    _model, npu, engine = make(delay=0.01)
    npu.replies = [[f" w{i}" for i in range(1000)], ["next"]]
    with live(engine) as url:
        body = {"messages": [{"role": "user", "content": "go"}], "stream": stream}
        with httpx.Client(timeout=10) as client:
            if stream:
                with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
                    next(line for line in r.iter_lines() if '" w' in line)  # (a token, then away)
            else:
                with pytest.raises(httpx.ReadTimeout):
                    client.post(f"{url}/v1/chat/completions", json=body, timeout=0.3)
        t0 = time.perf_counter()
        while engine.current is not None and time.perf_counter() - t0 < 5:
            time.sleep(0.01)
        assert engine.current is None and npu.position < 500  # (cancelled long before the reply's end)
        assert httpx.post(f"{url}/v1/chat/completions", json={**body, "stream": False}).json()["choices"][0][
            "message"]["content"] == "next"


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Serve the fake model with the web UI (a manual check)")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--thinking", action="store_true", help="a thinking model (Qwen's markers)")
    a.add_argument("--path", help="the UI's files (default: the pinned build)")
    args = a.parse_args()
    fm, fnpu, eng = make(reasoning=QWEN if args.thinking else None, capacity=8192, delay=0.03)
    reply = ["Here", "is", "**markdown**:\n\n-", "one\n-", "two\n\n```python\nprint('hi')\n```\n\nDone."]
    pieces = [w if i == 0 else " " + w for i, w in enumerate(reply)]
    fnpu.replies = [(["Thinking", " about", " it", "\n", "</think>", "\n\n"] if args.thinking else []) + pieces] * 1000
    d = args.path or ui.fetch()
    uvicorn.run(app_of(eng, ui=lambda app: ui.mount(app, d), reasoning="on", sse_ping_interval=1.0), port=args.port)
    eng.stop()
