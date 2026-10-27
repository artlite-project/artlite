"""artlite-server's router mode (no ``-m``), as llama-server's: the models of a directory (``catalog``), each served,
once a request or the web UI asks for it, by a server process of its own (``main``'s, through ``child``) that the
router starts, proxies to and stops. The web UI shows them in its model picker (``/props`` says ``role: router``).

Why a process per model: the router never opens the NPU, and a model's process leaving gives its memory, buffers and
hardware contexts back whatever its state; a model that crashes ends its own process only. A model's server is never
killed (AGENTS.md: a process holding an NPU context is not): its stdin's end asks it to stop, which it does at its
running request's next token or prompt chunk, as on SIGTERM; when the router ends, its models' servers see their
stdin end too. One server process is kept started ahead of the next load, its interpreter and imports done (the
spare: ~1.5 s of a load); it starts once no model is loading (its imports would slow the load).

Loading (``Router.ensure``): a request for a model that is not loaded loads it (``--models-autoload``; a request's
``?autoload=false`` does not) and waits; at most ``--models-max`` models run (loading, loaded or stopping), so a load
first stops the least recently used loaded model that has no request running and none waiting for it, or waits until
there is one: a model is not stopped under a request. Loads wait their turn in the order requests asked for them.
``POST /models/load`` loads in the background, ``POST /models/unload`` stops at once (its running request is
cancelled; during a load, the requests waiting for it fail: the UI's cancel). A model's status (``GET /v1/models``,
the feed ``GET /models/sse``): unloaded (``failed`` and its ``exit_code`` when its last load failed or its server
crashed), loading - with the share of its last load's time passed (``progress``) - and loaded.

Requests: the routes of a model (``/v1/chat/completions``, ``/completion``, ``/tokenize``, ..., ``/props?model=``,
``/slots``) go to the server of the model their ``model`` field or parameter names (an id, an alias or the file's path;
none: the one model running), their responses stream back as they come (``Proxied``; ``/props?model=`` with the
router's ``ui``: the web UI leaves out the models whose props say they have none); a client that goes away closes the
model server's response, which cancels its generation. The UI's Stop (``DELETE /v1/stream?conv_id=``) goes to the
server of the conversation's last request (``X-Conversation-Id``), or cancels the request still waiting for its model.
Each model's server listens on 127.0.0.1 with an API key of its own (``ARTLITE_ARG_API_KEY``) that the router sends."""

import argparse
import asyncio
import collections
import contextlib
import json
import logging
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from artlite import portable

from . import main as server
from .api import ApiError, add_ui, add_unsupported, bad_request, key_check, new_app, sse
from .catalog import Catalog, Entry, read_preset

log = logging.getLogger("artlite.server")

# the routes of a model: POST by the body's "model", GET by the "model" parameter
POSTED = ["/v1/chat/completions", "/chat/completions", "/v1/completions", "/completion", "/completions", "/tokenize",
          "/detokenize", "/apply-template", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses",
          "/responses", "/v1/responses/input_tokens", "/responses/input_tokens"]
CONVERSATIONS = 1024  # conversations whose model the router remembers (the UI's Stop)
PROGRESS_S = 0.5  # seconds between a load's progress events
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te", "trailer",
       "transfer-encoding", "upgrade"}
REQUEST_DROP = HOP | {"host", "content-length", "authorization", "x-api-key", "accept-encoding"}
RESPONSE_DROP = HOP | {"content-length", "server", "date"}
LOG_LINE = re.compile(r"([DIWEC]) ([\w.\-]+): (.*)")
LEVELS = {"D": logging.DEBUG, "I": logging.INFO, "W": logging.WARNING, "E": logging.ERROR, "C": logging.CRITICAL}


class Cancelled(Exception):
    """A request stopped waiting for its model: its client went away, the UI's Stop, the router's shutdown."""


class LoadError(Exception):
    """A model's load failed."""


class Child:
    """A model server's process (``python -m models.server.child``, in a process group of its own: a terminal's
    signals go to the router alone): ``start`` gives it its command line, ``stop`` closes its stdin. Threads read its
    messages (``on_message``, on the event loop), its log (into the router's, under ``name``) and its exit
    (``on_exit``, on the event loop); ``key``: its API key."""

    def __init__(self, command: list[str], env: dict[str, str], loop: asyncio.AbstractEventLoop):
        self.key = secrets.token_urlsafe(24)
        self.name = "spare"
        self.loop = loop
        self.on_message: Callable[[dict[str, Any]], None] | None = None
        self.on_exit: Callable[[int], None] | None = None
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env={**env, server.ENV + "API_KEY": self.key}, encoding="utf-8",
                                     errors="replace", bufsize=1, **portable.group_options())
        threading.Thread(target=self._messages, name="child-messages", daemon=True).start()
        threading.Thread(target=self._log, name="child-log", daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def _call(self, fn: Callable[..., None] | None, *args: object) -> None:
        if fn is not None:
            with contextlib.suppress(RuntimeError):  # (the event loop is gone: the router ends)
                self.loop.call_soon_threadsafe(fn, *args)

    def _messages(self) -> None:
        for line in self.proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                log.info("[%s] %s", self.name, line.rstrip())
                continue
            self._call(self.on_message, message)
        self._call(lambda code: self.on_exit and self.on_exit(code), self.proc.wait())

    def _log(self) -> None:
        level = logging.INFO
        for line in self.proc.stderr:
            line = line.rstrip("\n")
            if m := LOG_LINE.fullmatch(line):
                level = LEVELS[m[1]]
                logging.getLogger(m[2]).log(level, "[%s] %s", self.name, m[3])
            elif line:  # (a traceback's lines, a library's output)
                log.log(level, "[%s] %s", self.name, line)

    def start(self, argv: list[str]) -> None:
        try:
            self.proc.stdin.write(json.dumps(argv) + "\n")
            self.proc.stdin.flush()
        except OSError:  # (it ended: its exit says so)
            pass

    def stop(self) -> None:
        with contextlib.suppress(OSError):
            self.proc.stdin.close()


@dataclass(eq=False)
class Instance:
    """A model's server: ``state`` loading (until it reports ready), loaded, stopping (asked to stop), exited;
    ``requests``: the proxied requests running."""

    entry: Entry
    child: Child
    args: list[str]
    signature: tuple
    state: str = "loading"
    port: int = 0
    requests: int = 0
    loaded: bool = False  # it was loaded (a stopping server is still reported loaded)
    started: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    error: str | None = None
    settled: asyncio.Event = field(default_factory=asyncio.Event)  # loaded or exited
    exited: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.child.key}"}


class Router:
    """The models of ``catalog`` and their servers (module docstring). ``models_max``: models running at a time (0:
    any number); ``autoload``: a request loads its model; ``defaults``: the command line's values (a model's context:
    ``-c``); ``command``: a model server's process (default: ``python -m models.server.child``); ``spare``: keep one
    started ahead."""

    def __init__(self, catalog: Catalog, *, models_max: int = 1, autoload: bool = True,
                 defaults: argparse.Namespace | None = None, command: list[str] | None = None, spare: bool = True):
        self.catalog = catalog
        self.models_max, self.autoload, self.spare_on = models_max, autoload, spare
        self.defaults = defaults or server.arg_parser(env=False).parse_args([])
        self.command = command or [sys.executable, "-m", "models.server.child"]
        hidden = server.ROUTER | server.MODEL  # (the children's env: none of the router's own, nor a model's)
        self.env = {k: v for k, v in os.environ.items()
                    if not (k.startswith(server.ENV) and k[len(server.ENV):].lower() in hidden)}
        self.instances: dict[str, Instance] = {}
        self.failures: dict[str, dict[str, Any]] = {}  # id -> exit_code, error of its last failed load
        self.queue: list[str] = []  # the models requests wait to load, in the order they asked
        self.convs: collections.OrderedDict[str, str] = collections.OrderedDict()  # conversation -> model
        self.parked: dict[str, set[asyncio.Event]] = {}  # conversation -> requests waiting for their model
        self.feeds: set[asyncio.Queue] = set()
        self.children: list[Child] = []
        self.spare: Child | None = None
        self.closing = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.client: httpx.AsyncClient | None = None
        self.scanned = asyncio.Event()
        self._changed = asyncio.Event()
        self._scan_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Start, models, shutdown
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """(the app's startup) Scan the models in a thread, start the spare, then load the preset's startup models."""
        self.loop = asyncio.get_running_loop()
        transport = httpx.AsyncHTTPTransport(  # (a request's headers and body are two writes: no Nagle delay)
            socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=32))
        self.client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(None, connect=10.0), trust_env=False)
        self._refill()
        self._spawn_task(self._startup())

    async def _startup(self) -> None:
        try:
            await self.reload()
        except ValueError as e:
            log.error("%s", e)
            return
        for entry in self.catalog.entries.values():
            if entry.load_on_startup:
                self.load_later(entry)

    async def reload(self) -> None:
        """Scan the models again: a running model whose file or server's options changed, or that is gone, stops."""
        async with self._scan_lock:
            try:
                await asyncio.to_thread(self.catalog.scan)
            finally:
                first = not self.scanned.is_set()
                self.scanned.set()
            log.info("%d models in %s%s", len(self.catalog.entries), self.catalog.models_dir,
                     f" and {self.catalog.preset}" if self.catalog.preset else "")
        for inst in list(self.instances.values()):
            entry = self.catalog.entries.get(inst.entry.id)
            if entry is None or self.signature(entry) != inst.signature:
                self._stop(inst, "its file or options changed" if entry else "it is gone")
        if not first:
            self._publish({"model": "*", "event": "models_reload"})

    def signature(self, entry: Entry) -> tuple:
        """What a model's server is started from (a changed one restarts it)."""
        try:
            st = os.stat(entry.path)
            stamp = (st.st_size, st.st_mtime_ns)
        except OSError:
            stamp = ()
        return (entry.path, stamp, tuple(self.catalog.argv(entry)))

    def close(self) -> None:
        """(any thread) The shutdown: stop every model's server and the spare, end the feeds, cancel the requests
        waiting for their models."""
        if self.loop is not None and self.loop.is_running():
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self._close)
                return
        self.closing = True
        for child in self.children:
            child.stop()

    def _close(self) -> None:
        self.closing = True
        for inst in list(self.instances.values()):
            self._stop(inst, "shutdown")
        for child in self.children:
            child.stop()
        for q in self.feeds:
            q.put_nowait(None)
        self._change()

    def wait(self) -> None:
        """(after the event loop) Until every model's server has ended: they are never killed."""
        for child in self.children:
            child.stop()
            while True:
                try:
                    child.proc.wait(10)
                    break
                except subprocess.TimeoutExpired:
                    log.warning("waiting for %s's server to end (pid %d)", child.name, child.proc.pid)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, entry: Entry) -> dict[str, Any]:
        """A model's status, as llama-server's router reports it."""
        inst = self.instances.get(entry.id)
        if inst is not None:
            return {"value": "loaded" if inst.loaded else "loading", "args": inst.args}
        st: dict[str, Any] = {"value": "unloaded", "args": ["artlite-server", *self.catalog.argv(entry)]}
        if (f := self.failures.get(entry.id)) is not None:
            st |= {"failed": True, **f}
        return st

    def n_ctx(self, entry: Entry) -> int:
        """The KV cache positions its server will have."""
        return server.capacity(entry.options.get("ctx_size", self.defaults.ctx_size), entry.header["n_ctx_train"])

    def listing(self) -> dict[str, Any]:
        """GET /v1/models: every model with its status, and (aligned) details as llama-server's ``models``."""
        data, details = [], []
        for e in self.catalog.entries.values():
            h, vision = e.header, e.mmproj is not None
            data.append({"id": e.id, "aliases": e.aliases, "tags": [], "object": "model", "owned_by": "artlite",
                         "created": int(time.time()), "path": e.path, "status": self.status(e),
                         "architecture": {"input_modalities": ["text"] + ["image"] * vision,
                                          "output_modalities": ["text"]},
                         "source": e.source, "can_remove": False, "context_length": h["n_ctx_train"],
                         "meta": {"architecture": h["arch"], "n_vocab": h["n_vocab"], "n_ctx": self.n_ctx(e),
                                  "n_ctx_train": h["n_ctx_train"], "n_embd": h["n_embd"], "n_params": h["n_params"],
                                  "size": h["size"], "ftype": h["ftype"]}})
            details.append({"name": e.id, "model": e.id, "modified_at": "", "size": "", "digest": "", "type": "model",
                            "description": "", "tags": [""], "capabilities": ["completion"] + ["multimodal"] * vision,
                            "parameters": "",
                            "details": {"parent_model": "", "format": "gguf", "family": h["arch"],
                                        "families": [h["arch"]], "parameter_size": h["size_label"],
                                        "quantization_level": e.tag}})
        return {"object": "list", "data": data, "models": details}

    def _publish(self, event: dict[str, Any]) -> None:
        for q in self.feeds:
            q.put_nowait(event)

    def _notify(self, entry: Entry, status: str, **data: object) -> None:
        self._publish({"model": entry.id, "event": "model_status", "data": {"status": status, **data}})

    def _change(self) -> None:
        """Wake whatever waits for a change (a load's turn, a free slot, an idle model)."""
        self._changed.set()
        self._changed = asyncio.Event()

    def _spawn_task(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------
    # Loading and stopping
    # ------------------------------------------------------------------

    def find(self, name: str | None) -> Entry:
        """The model a request names (none: the one model running, or the only one there is)."""
        if not name:
            if len(self.instances) == 1:
                return next(iter(self.instances.values())).entry
            if len(self.catalog.entries) == 1:
                return next(iter(self.catalog.entries.values()))
            raise bad_request("model name is missing from the request")
        entry = self.catalog.find(name)
        if entry is None:
            raise bad_request(f"model '{name}' not found")
        return entry

    async def ensure(self, entry: Entry, *, load: bool = True,
                     cancelled: Callable[[], Awaitable[bool]] | None = None) -> Instance:
        """The model's server once it is loaded: loaded (``load``), or one loading waited for (module docstring).
        ApiError 400: not loaded (``load`` false); LoadError: its load failed; Cancelled: ``cancelled`` became true
        or the router shuts down."""
        self.queue.append(entry.id)
        try:
            while True:
                if self.closing:
                    raise Cancelled("the server is shutting down")
                inst = self.instances.get(entry.id)
                if inst is not None and inst.state == "loaded":
                    return inst
                if inst is not None and inst.state == "loading":
                    await self._wait(inst.settled, cancelled)
                    if inst.state == "exited":
                        raise LoadError(inst.error or "its server ended")
                    if inst.state == "stopping":  # (an unload during the load: the UI's cancel)
                        raise LoadError(inst.error)
                    continue
                if inst is not None:  # (stopping: load it again once it ended)
                    await self._wait(inst.exited, cancelled)
                    continue
                if not load:
                    raise bad_request("model is not loaded")
                if self._turn(entry.id):
                    if self.models_max <= 0 or len(self.instances) < self.models_max:
                        self._spawn(entry)
                        continue
                    if (victim := self._victim()) is not None:
                        self._stop(victim, f"for {entry.id}")
                await self._wait(self._changed, cancelled)
        finally:
            self.queue.remove(entry.id)
            self._change()

    def load_later(self, entry: Entry) -> None:
        """Load a model in the background (POST /models/load, a preset's load-on-startup)."""

        async def load() -> None:
            with contextlib.suppress(LoadError, Cancelled):
                await self.ensure(entry)

        self._spawn_task(load())

    async def _wait(self, event: asyncio.Event, cancelled: Callable[[], Awaitable[bool]] | None) -> None:
        while True:
            try:
                await asyncio.wait_for(event.wait(), 0.5)
                return
            except TimeoutError:
                if self.closing:
                    raise Cancelled("the server is shutting down") from None
                if cancelled is not None and await cancelled():
                    raise Cancelled("the request was cancelled while its model was loading") from None

    def _turn(self, model_id: str) -> bool:
        """The model is the first one requests wait for that is not running."""
        return next((m for m in self.queue if m not in self.instances), None) == model_id

    def _victim(self) -> Instance | None:
        """The least recently used loaded model without requests, running or waiting."""
        idle = [i for i in self.instances.values() if i.state == "loaded" and i.requests == 0
                and i.entry.id not in self.queue]
        return min(idle, key=lambda i: i.last_used, default=None)

    def _child(self) -> Child:
        self.children = [c for c in self.children if c.alive]
        child = Child(self.command, self.env, self.loop)
        self.children.append(child)
        return child

    def _refill(self) -> None:
        """Start the spare when there is none and no model loads."""
        if not self.spare_on or self.closing or (self.spare is not None and self.spare.alive):
            return
        if any(i.state == "loading" for i in self.instances.values()):
            return
        child = self.spare = self._child()

        def ended(code: int) -> None:
            if self.spare is child:
                self.spare = None
                if not self.closing:
                    log.warning("the spare server process ended (exit code %d)", code)

        child.on_exit = ended

    def _spawn(self, entry: Entry) -> Instance:
        child, self.spare = self.spare, None
        if child is None or not child.alive:
            child = self._child()
        argv = [*self.catalog.argv(entry), "--host", "127.0.0.1", "--port", "0", "--no-ui"]
        inst = Instance(entry=entry, child=child, args=["artlite-server", *argv], signature=self.signature(entry))
        child.name = entry.name
        child.on_message = lambda m: self._message(inst, m)
        child.on_exit = lambda code: self._exited(inst, code)
        self.instances[entry.id] = inst
        self.failures.pop(entry.id, None)
        log.info("loading %s: artlite-server %s", entry.id, " ".join(argv))
        child.start(argv)
        self._notify(entry, "loading")
        if entry.load_s > 0:
            self._spawn_task(self._progress(inst))
        self._change()
        return inst

    async def _progress(self, inst: Instance) -> None:
        """The loading bar: the share of the model's last load time passed."""
        while inst.state == "loading":
            await asyncio.sleep(PROGRESS_S)
            if inst.state == "loading":
                value = min((time.monotonic() - inst.started) / inst.entry.load_s, 0.99)
                self._notify(inst.entry, "loading",
                             progress={"stages": ["text_model"], "current": "text_model", "value": round(value, 3)})

    def _message(self, inst: Instance, message: dict[str, Any]) -> None:
        if "port" in message:
            inst.port = int(message["port"])
        if "error" in message:
            inst.error = str(message["error"])
        if "ready" in message and inst.state == "loading":
            seconds = time.monotonic() - inst.started
            inst.state, inst.loaded = "loaded", True
            log.info("%s: loaded in %.1f s (%s)", inst.entry.id, seconds, message["ready"].get("describe", ""))
            self.catalog.record_load(inst.entry, seconds)
            inst.settled.set()
            self._notify(inst.entry, "loaded")
            self._change()
            self._refill()

    def _exited(self, inst: Instance, code: int) -> None:
        stopped = inst.state == "stopping"
        inst.state = "exited"
        if self.instances.get(inst.entry.id) is inst:
            del self.instances[inst.entry.id]
        failed = code != 0 or not (inst.loaded or stopped)
        extra: dict[str, Any] = {}
        if failed:
            error = inst.error or f"its server ended with exit code {code}"
            extra = {"exit_code": code or 1}
            self.failures[inst.entry.id] = {**extra, "error": error}
            inst.error = error
            log.warning("%s: %s", inst.entry.id, error)
        else:
            log.info("%s: unloaded", inst.entry.id)
        inst.settled.set()
        inst.exited.set()
        self._notify(inst.entry, "unloaded", **extra)
        self._change()
        self._refill()

    def _stop(self, inst: Instance, why: str = "") -> None:
        if inst.state in ("stopping", "exited"):
            return
        log.info("unloading %s%s", inst.entry.id, f" ({why})" if why else "")
        inst.state = "stopping"
        inst.child.stop()
        if not inst.loaded:  # (the requests waiting for its load fail now)
            inst.error = f"its load was cancelled ({why})" if why else "its load was cancelled"
            inst.settled.set()
        self._change()

    def unload(self, inst: Instance) -> None:
        self._stop(inst, "unload")

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def remember(self, conv_id: str | None, entry: Entry) -> None:
        if conv_id:
            self.convs[conv_id] = entry.id
            self.convs.move_to_end(conv_id)
            while len(self.convs) > CONVERSATIONS:
                self.convs.popitem(last=False)

    @contextlib.contextmanager
    def parking(self, conv_id: str | None) -> Any:
        """A request waiting for its model, cancelled by its conversation's Stop: the event its Stop sets."""
        stop = asyncio.Event()
        if conv_id:
            self.parked.setdefault(conv_id, set()).add(stop)
        try:
            yield stop
        finally:
            if conv_id:
                self.parked[conv_id].discard(stop)
                if not self.parked[conv_id]:
                    del self.parked[conv_id]

    def stop_parked(self, conv_id: str) -> int:
        """The UI's Stop of requests still waiting for their models; how many."""
        events = self.parked.get(conv_id, set())
        for e in events:
            e.set()
        return len(events)

    async def forward(self, request: Request, inst: Instance, body: bytes | None) -> Response:
        """The request to the model's server; its response, streamed through (``Proxied``)."""
        headers = {k: v for k, v in request.headers.items() if k.lower() not in REQUEST_DROP} | inst.headers
        url = inst.url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        inst.requests += 1
        inst.last_used = time.monotonic()

        def done() -> None:
            inst.requests -= 1
            inst.last_used = time.monotonic()
            self._change()

        try:
            r = await self.client.send(self.client.build_request(request.method, url, headers=headers, content=body),
                                       stream=True)
        except httpx.HTTPError as e:
            done()
            raise ApiError(502, "server_error", f"{inst.entry.id}: its server did not answer ({e})") from None
        return Proxied(r, done)


class Proxied(StreamingResponse):
    """A model server's response, streamed through as it comes; its end, or the client's going away, closes it (the
    model's server then cancels its generation)."""

    def __init__(self, r: httpx.Response, done: Callable[[], None]):
        super().__init__(r.aiter_raw(), status_code=r.status_code,
                         headers={k: v for k, v in r.headers.items() if k.lower() not in RESPONSE_DROP})
        self.r, self.done = r, done

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.r.aclose()
            self.done()


@dataclass
class RouterOptions:
    """The router's settings (the command line, ``serve``)."""

    api_keys: frozenset[str] = frozenset()
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    ui: Callable[[FastAPI], None] | None = None
    no_ui: str = "The web UI is off: the API is at /v1."
    build_info: str = "Artlite"


def create_router_app(router: Router, options: RouterOptions) -> FastAPI:
    """The router's FastAPI app: its own routes, the models' routes proxied (module docstring), the web UI."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await router.start()
        yield
        await router.client.aclose()

    app = new_app(build_info=options.build_info, cors_origins=options.cors_origins, lifespan=lifespan)
    public = APIRouter()
    api = APIRouter(dependencies=[Depends(key_check(options.api_keys))])

    def autoload(request: Request) -> bool:
        value = request.query_params.get("autoload")
        return router.autoload if value is None else value.lower() in ("1", "true")

    async def instance(request: Request, name: str | None) -> Instance:
        """The server of a request's model once it is loaded (module docstring)."""
        await router.scanned.wait()
        entry = router.find(name)
        conv_id = request.headers.get("x-conversation-id")
        router.remember(conv_id, entry)
        with router.parking(conv_id) as stop:

            async def cancelled() -> bool:
                return stop.is_set() or await request.is_disconnected()

            try:
                inst = await router.ensure(entry, load=autoload(request), cancelled=cancelled)
            except LoadError as e:
                raise ApiError(500, "server_error", f"{entry.id} failed to load: {e}") from None
            except Cancelled as e:
                raise ApiError(503 if router.closing else 400,
                               "unavailable_error" if router.closing else "invalid_request_error", str(e)) from None
        return inst

    async def route(request: Request, name: str | None, body: bytes | None = None) -> Response:
        """A model's request, to its server once loaded."""
        return await router.forward(request, await instance(request, name), body)

    @public.get("/health")
    @public.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/props")
    async def props(request: Request, model: str | None = None) -> Any:
        if model:  # (the model's, with the router's web UI: the UI leaves out the models whose props say ui false)
            inst = await instance(request, model)
            try:
                r = await router.client.get(f"{inst.url}/props", headers=inst.headers)
            except httpx.HTTPError as e:
                raise ApiError(502, "server_error", f"{inst.entry.id}: its server did not answer ({e})") from None
            body = r.json()
            if r.status_code == 200:
                body["ui"] = options.ui is not None
            return JSONResponse(body, status_code=r.status_code)
        return {"role": "router", "max_instances": router.models_max, "models_autoload": router.autoload,
                "model_alias": "artlite-server", "model_path": "none",
                "default_generation_settings": {"params": {}, "n_ctx": 0}, "ui": options.ui is not None,
                "ui_settings": {}, "build_info": options.build_info, "cors_proxy_enabled": False}

    @api.get("/v1/models")
    @api.get("/models")
    async def models(reload: str = "") -> dict[str, Any]:
        if reload:
            try:
                await router.reload()
            except ValueError as e:
                raise bad_request(str(e)) from None
        else:
            await router.scanned.wait()
        return router.listing()

    async def named(request: Request) -> Entry:
        try:
            body = await request.json()
        except ValueError:
            raise bad_request("the request body is not JSON") from None
        name = body.get("model") if isinstance(body, dict) else None
        await router.scanned.wait()
        if not isinstance(name, str) or (entry := router.catalog.find(name)) is None:
            raise ApiError(404, "not_found_error", "model is not found")
        return entry

    @api.post("/models/load")
    async def load(request: Request) -> dict[str, bool]:
        entry = await named(request)
        if entry.id in router.instances:
            raise bad_request("model is already running")
        router.load_later(entry)
        return {"success": True}

    @api.post("/models/unload")
    async def unload(request: Request) -> dict[str, bool]:
        entry = await named(request)
        inst = router.instances.get(entry.id)
        if inst is None:
            raise bad_request("model is not running")
        router.unload(inst)
        return {"success": True}

    @api.get("/models/sse")
    async def feed() -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue()

        async def events() -> AsyncIterator[str]:
            router.feeds.add(q)
            try:
                while not router.closing:
                    try:
                        event = await asyncio.wait_for(q.get(), 15)
                    except TimeoutError:
                        yield ":\n\n"
                        continue
                    if event is None:
                        return
                    yield sse(event)
            finally:
                router.feeds.discard(q)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @api.post("/models")
    @api.delete("/models")
    async def downloads() -> None:
        raise ApiError(501, "not_supported_error", "artlite-server does not download models: put GGUF files in "
                       f"{router.catalog.models_dir}/<account>/<repository>/ (hf download ... --local-dir), then "
                       "GET /v1/models?reload=1")

    async def posted(request: Request) -> Response:
        body = await request.body()
        try:
            data = json.loads(body) if body else {}
        except ValueError:
            raise bad_request("the request body is not JSON") from None
        name = data.get("model") if isinstance(data, dict) else None
        return await route(request, name if isinstance(name, str) else None, body)

    for path in POSTED:
        api.add_api_route(path, posted, methods=["POST"])

    @api.get("/slots")
    async def slots(request: Request, model: str | None = None) -> Response:
        return await route(request, model)

    @api.get("/v1/responses/{rid}")
    @api.get("/responses/{rid}")
    @api.delete("/v1/responses/{rid}")
    @api.delete("/responses/{rid}")
    async def stored_response(request: Request, rid: str, model: str | None = None) -> Response:
        """A stored response (the Responses API's): at the model's server (``?model=``, else the one loaded)."""
        return await route(request, model)

    @api.delete("/v1/stream")
    async def stop_stream(request: Request, conv_id: str = "") -> Any:
        """The UI's Stop: the conversation's request, at its model's server or waiting for its model."""
        if not conv_id:
            return {"success": True, "cancelled": 0}
        if n := router.stop_parked(conv_id):
            return {"success": True, "cancelled": n}
        inst = router.instances.get(router.convs.get(conv_id, ""))
        if inst is None or inst.state != "loaded":
            return {"success": True, "cancelled": 0}
        return await router.forward(request, inst, None)

    @api.get("/v1/stream")
    async def resume_stream() -> None:
        raise ApiError(404, "not_found_error", "no resumable streams")

    @api.post("/v1/streams/lookup")
    async def lookup_streams() -> list[Any]:
        return []

    add_unsupported(api)
    app.include_router(public)
    app.include_router(api)
    add_ui(app, options.ui, options.no_ui)
    return app


def serve(p: argparse.ArgumentParser, args: argparse.Namespace, given: set[str]) -> None:
    """Router mode (``main``): ``given``, the options of the command line (to the models' servers: the ones that are
    not the router's)."""
    if given & server.MODEL:
        p.error(f"{', '.join(sorted('--' + d for d in given & server.MODEL))}: one model's option (router mode: a "
                f"preset's, --models-preset)")
    if args.models_max < 0:
        p.error("--models-max: 0 (any number) or more")
    try:
        keys = server.api_keys(args)
        if args.models_preset:
            read_preset(args.models_preset, p)
    except (OSError, ValueError) as e:
        sys.exit(f"artlite-server: {e}")
    if not os.path.isdir(args.models_dir) and not args.models_preset:
        sys.exit(f"artlite-server: {args.models_dir}: no such directory (--models-dir, ARTLITE_MODELS_DIR, or a "
                 f"model: -m)")
    forwarded = {d: getattr(args, d) for d in given - server.ROUTER - server.MODEL}
    catalog = Catalog(args.models_dir, parser=p, preset=args.models_preset, options=forwarded)
    router = Router(catalog, models_max=args.models_max, autoload=args.models_autoload, defaults=args)
    ui, no_ui = server.web_ui(args)
    options = RouterOptions(api_keys=keys, cors_origins=[o.strip() for o in args.cors_origins.split(",") if o.strip()],
                            ui=ui, no_ui=no_ui or RouterOptions.no_ui, build_info=server.build_info())
    config = uvicorn.Config(create_router_app(router, options), host=args.host, port=args.port, log_config=None,
                            access_log=args.verbose, timeout_keep_alive=int(args.timeout), lifespan="on")
    srv = server.Server(config, router.close)
    sock = server.listen(config)
    log.info("router mode: the models of %s (at most %s at a time); %s, the API at /v1", catalog.models_dir,
             args.models_max or "any number", "the web UI at /" if ui else "no web UI")
    try:
        srv.run(sockets=[sock])
    finally:
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, lambda *_: log.warning("stopping: waiting for the models' servers to end"))
        router.close()
        router.wait()
