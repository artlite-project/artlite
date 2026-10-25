"""artlite-server: the command line, the engine and uvicorn, the shutdown. The options follow llama.cpp's
``llama-server`` where it has an equivalent; each also reads ``ARTLITE_ARG_<NAME>`` (the long option's name in upper
case, dashes as underscores: ``ARTLITE_ARG_CTX_SIZE``), which the command line overrides. With ``-m`` the server
serves that model; without, the models of a directory, each in a server of its own (``router``):

    artlite-server -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf
    artlite-server --models-dir ~/models
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from types import FrameType
from typing import Any

import uvicorn
from fastapi import FastAPI

from artlite.model_config import MODELS_DIR, model_path

from .. import open_model
from ..common import Model, NpuModel, Sampling, performance_mode, stale_toolchain
from . import ui
from .api import PENALTIES, ModelInfo, Options, create_app
from .engine import Engine

log = logging.getLogger("artlite.server")

ENV = "ARTLITE_ARG_"
DEFAULT_CTX = 32768  # -c 0 (capped at the model's context)
# the router's own options (in router mode each model's server gets its own: ``router``)
ROUTER = frozenset({"host", "port", "api_key", "api_key_file", "cors_origins", "path", "no_ui", "timeout", "log_file",
                    "models_dir", "models_preset", "models_max", "models_autoload"})
MODELS = frozenset({"models_dir", "models_preset", "models_max", "models_autoload"})  # (router mode's)
MODEL = frozenset({"model", "alias", "mmproj"})  # one model's (in router mode: the router's, a preset's)
Report = Callable[[dict[str, Any]], None]  # a router's child: a message to the router


def arg_parser(*, env: bool = True) -> argparse.ArgumentParser:
    """The options; ``env``: their defaults from ``ARTLITE_ARG_<NAME>`` (``with_env``)."""
    p = argparse.ArgumentParser(prog="artlite-server", description="An OpenAI-compatible server and web chat for GGUF "
                                "models on the NPU (llama-server's API and options where they apply): of one model "
                                "(-m), or of the models of a directory (router mode, without -m)")
    p.add_argument("-m", "--model", type=model_path,
                   help=f"GGUF file; a relative path is also looked up in the models directory ({MODELS_DIR}, "
                        f"ARTLITE_MODELS_DIR); without: router mode")
    p.add_argument("-a", "--alias", help="the model's id in the API (default: the file name)")
    net = p.add_argument_group("server")
    net.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    net.add_argument("--port", type=int, default=8080, help="port to listen on (default: 8080)")
    net.add_argument("--api-key", help="API keys clients must send, comma-separated (default: none)")
    net.add_argument("--api-key-file", help="a file of API keys, one per line ('#' starts a comment)")
    net.add_argument("--cors-origins", default="*", help="origins allowed by CORS, comma-separated (default: *)")
    net.add_argument("--path", help="serve the web UI's files from this directory (default: llama.cpp's prebuilt "
                                    f"UI, build {ui.TAG}, fetched on the first start)")
    net.add_argument("--no-ui", action="store_true", help="no web UI (the API only)")
    net.add_argument("-to", "--timeout", type=float, default=3600,
                     help="seconds an idle connection is kept open (default: 3600)")
    net.add_argument("--sse-ping-interval", type=float, default=30,
                     help="seconds between pings while a stream waits for the NPU (default: 30)")
    net.add_argument("--slots", action=argparse.BooleanOptionalAction, default=True,
                     help="the /slots endpoint (default: on)")
    models = p.add_argument_group("router mode (without -m)")
    models.add_argument("--models-dir", default=MODELS_DIR,
                        help=f"serve the GGUF models of this directory (default: {MODELS_DIR}, ARTLITE_MODELS_DIR)")
    models.add_argument("--models-preset", help="an INI file of models' options and more models (llama-server's "
                                                "format: [*] for all, [<model id>] for one)")
    models.add_argument("--models-max", type=int, default=1,
                        help="models loaded at a time, 0: any number (default: 1; their dispatches take turns)")
    models.add_argument("--models-autoload", action=argparse.BooleanOptionalAction, default=True,
                        help="load the model a request names (default: on; a request's ?autoload=false: not)")
    gen = p.add_argument_group("generation")
    gen.add_argument("-c", "--ctx-size", type=int, default=0,
                     help=f"KV cache positions, rounded up to a power of two >= 4096 (default: 0, {DEFAULT_CTX} capped "
                          f"at the model's context)")
    gen.add_argument("-n", "--n-predict", type=int, default=-1,
                     help="tokens to generate when a request does not say (default: -1, until an end-of-sequence "
                          "token or a full KV cache)")
    gen.add_argument("-np", "--parallel", type=int, default=1, help="sequences at a time (1 only: the NPU runs one)")
    gen.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=True,
                     help="continue from the prefix a prompt shares with the cached tokens (default: on)")
    gen.add_argument("--temp", type=float, help="sampling temperature (default: the model's; 0: greedy)")
    gen.add_argument("--top-k", type=int, help="sample among the k most likely tokens (default: the model's)")
    gen.add_argument("--top-p", type=float, help="nucleus sampling threshold (default: the model's)")
    gen.add_argument("--min-p", type=float, help="drop tokens below this fraction of the most likely one's "
                                                 "probability (default: the model's, 0)")
    gen.add_argument("--repeat-penalty", type=float, help="divide the logits of the last --repeat-last-n tokens by "
                                                          "this (default: the model's; 1: none)")
    gen.add_argument("--repeat-last-n", type=int,
                     help="the tokens the penalties look back at (default: the model's, 64; 0: none)")
    gen.add_argument("--presence-penalty", type=float, help="subtract this from the logits of the last "
                                                            "--repeat-last-n tokens (default: the model's, 0)")
    gen.add_argument("--frequency-penalty", type=float, help="subtract this times their count from the logits of the "
                     "last --repeat-last-n tokens (default: the model's, 0)")
    gen.add_argument("-s", "--seed", type=int, default=-1, help="sampling seed (default: -1, random per request)")
    gen.add_argument("-rea", "--reasoning", choices=("on", "off", "auto"), default="off",
                     help="let a model with a thinking mode think first (auto: when it has one; default: off); a "
                          "request's chat_template_kwargs.enable_thinking overrides it")
    gen.add_argument("--reasoning-format", choices=("auto", "deepseek", "none"), default="auto",
                     help="thoughts in reasoning_content (auto, deepseek) or in content (none) (default: auto)")
    gen.add_argument("--chat-template-kwargs", type=json.loads, default={},
                     help="chat template variables as a JSON object, under each request's (enable_thinking)")
    vis = p.add_argument_group("images (MiniCPM-V 4.7)")
    vis.add_argument("--mmproj", type=model_path, help="the vision encoder's GGUF file (mmproj-model-f16.gguf): images "
                     "in chat messages (image_url parts of data URIs)")
    vis.add_argument("--max-slices", type=int,
                     help="an image's slices at most (default: the mmproj's, 9; the model card uses 36)")
    npu = p.add_argument_group("NPU")
    npu.add_argument("--no-prefill", action="store_true", help="run prompts as decode steps")
    npu.add_argument("--kv-k", choices=("single", "hilo"),
                     help="K precision of the KV cache (Llama, Gemma), as artlite-cli's --kv-k")
    npu.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                     help="run one token after loading (default: on)")
    out = p.add_argument_group("logging")
    out.add_argument("-v", "--verbose", action="store_true", help="debug messages and every HTTP request")
    out.add_argument("--log-file", help="also write the log to this file")
    if env:
        with_env(p)
    return p


def env_name(a: argparse.Action) -> str | None:
    """An option's ``ARTLITE_ARG_<NAME>`` (None: help)."""
    longs = sorted((s for s in a.option_strings if s.startswith("--")), key=lambda s: s.startswith("--no-"))
    return None if not longs or a.dest == "help" else ENV + longs[0][2:].upper().replace("-", "_")


def with_env(p: argparse.ArgumentParser) -> None:
    """Each option's default from ``ARTLITE_ARG_<NAME>`` when it is set."""
    for a in p._actions:
        name = env_name(a)
        if name is None:
            continue
        a.help = f"{a.help} (env: {name})" if a.help else f"(env: {name})"
        if name not in os.environ:
            continue
        value = os.environ[name]
        if a.nargs == 0:  # (a switch)
            a.default = value.lower() in ("1", "true", "on", "yes", "enabled")
        else:
            if a.choices is not None and value not in a.choices:
                p.error(f"{name}: {value!r} (one of {', '.join(a.choices)})")
            a.default = value  # (argparse converts a string default with the option's type)
        a.required = False


def given(argv: list[str]) -> set[str]:
    """The options the command line gives (not defaults, not environment variables)."""
    p = arg_parser(env=False)
    for a in p._actions:
        a.default = argparse.SUPPRESS
    return set(vars(p.parse_args(argv)))


def capacity(ctx_size: int, n_ctx_train: int) -> int:
    """KV cache positions of ``-c``: a power of two from 4096 (a decode design per size), at most the model's
    context (when it is a power of two from 4096)."""
    want = ctx_size or DEFAULT_CTX
    if n_ctx_train:
        want = min(want, n_ctx_train)
    cap = 4096
    while cap < want:
        cap *= 2
    while n_ctx_train and cap > max(n_ctx_train, 4096):
        cap //= 2
    return cap


def api_keys(args: argparse.Namespace) -> frozenset[str]:
    keys = [k.strip() for k in (args.api_key or "").split(",")]
    if args.api_key_file:
        with open(args.api_key_file) as f:
            keys += [line.split("#", 1)[0].strip() for line in f]
    return frozenset(k for k in keys if k)


def build_info() -> str:
    from ..bench import environment

    env = environment()
    return f"Artlite b{env['build_number']}-{env['build_commit']}"


def web_ui(args: argparse.Namespace) -> tuple[Callable[[FastAPI], None] | None, str]:
    """The web UI of the options (``ui.mount`` of its files), or None and what ``/`` says instead."""
    if args.no_ui:
        return None, "The web UI is off (--no-ui): the API is at /v1."
    try:
        directory = args.path or ui.fetch()
    except (OSError, ValueError) as e:
        log.warning("no web UI: %s", e)
        return None, (f"The web UI could not be fetched ({e}). It is llama.cpp's prebuilt UI, build {ui.TAG}, "
                      f"downloaded from {ui.URL.format(tag=ui.TAG)} on the first start; serve a local build with "
                      f"--path DIR (llama.cpp's build/tools/ui/dist), or run with --no-ui. The API is at /v1.")
    return (lambda app: ui.mount(app, directory)), ""


def listen(config: uvicorn.Config) -> socket.socket:
    """uvicorn's listening socket (``bind_socket``: a busy port fails before the model loads) as a socket of protocol
    TCP: asyncio sets TCP_NODELAY only on the connections of such a socket, and without it a keep-alive connection's
    response, written in parts, waits for the client's delayed ACK (~40 ms each)."""
    sock = config.bind_socket()
    return socket.socket(sock.family, sock.type, socket.IPPROTO_TCP, fileno=sock.detach())


class Server(uvicorn.Server):
    """uvicorn, whose shutdown (SIGINT, SIGTERM) first calls ``stop`` - the engine's: the running job stops at its next
    token or prompt chunk; the router's: its models' servers stop - and never ends the process during a dispatch
    (AGENTS.md: a process holding an NPU context is never killed): a second signal only stops waiting for the
    connections."""

    def __init__(self, config: uvicorn.Config, stop: Callable[[], None]):
        super().__init__(config)
        self.stop = stop

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if self.should_exit:
            log.warning("stopping: waiting for the NPU's dispatch to finish")
            self.force_exit = True
        self.stop()
        super().handle_exit(sig, frame)

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        """(as uvicorn's, but the signals are not raised again after the shutdown: SIGTERM's default action would end
        the process before the worker's dispatch)"""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        old = {s: signal.signal(s, self.handle_exit) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)


def setup_logging(args: argparse.Namespace, *, child: bool = False) -> None:
    """The log on stderr (and ``--log-file``); a router's child logs without times (the router adds them). Without
    ``-v``: no line per proxied request (httpx's), nor a child's uvicorn banner."""
    fmt = "%(levelname).1s %(name)s: %(message)s" if child else "%(asctime)s %(levelname).1s %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=fmt, datefmt="%H:%M:%S")
    if not args.verbose:
        for name in ("httpx", *(("uvicorn.error",) if child else ())):
            logging.getLogger(name).setLevel(logging.WARNING)
    if args.log_file:
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)
        logging.getLogger().addHandler(handler)


def main(argv: list[str] | None = None, *, report: Report | None = None) -> None:
    """The server of ``argv``'s options. ``report``: the server is a router's child (``child``): it reports its port,
    its readiness and a failed load through it, and the end of its stdin shuts it down."""
    argv = sys.argv[1:] if argv is None else argv
    p = arg_parser()
    args = p.parse_args(argv)
    explicit = given(argv)
    if args.parallel != 1:
        p.error("-np/--parallel: only 1 (the NPU runs one sequence at a time)")
    setup_logging(args, child=report is not None)
    if report is None:  # (a router's children: its checks, under its mode)
        if stale := stale_toolchain():
            log.warning("%s", stale)
        performance_mode(log.info)
    if args.model is None:
        from .router import serve

        serve(p, args, explicit)
        return
    if explicit & MODELS:
        p.error(f"{', '.join(sorted('--' + d.replace('_', '-') for d in explicit & MODELS))}: router mode's options "
                f"(without -m)")
    serve_model(args, report)


def serve_model(args: argparse.Namespace, report: Report | None) -> None:
    """One model's server (``-m``)."""
    name = os.path.basename(args.model)
    try:
        info = ModelInfo.from_gguf(args.model)
        keys = api_keys(args)
        penalties = {k: getattr(args, k) for k in PENALTIES if getattr(args, k) is not None}
        Sampling(**penalties)  # (ValueError: out of range)
    except (OSError, ValueError) as e:
        if report is not None:
            report({"error": f"{type(e).__name__}: {e}"})
        sys.exit(f"artlite-server: {e}")
    n_ctx = capacity(args.ctx_size, info.n_ctx_train)
    options = Options(alias=args.alias or name, n_ctx=n_ctx, api_keys=keys, n_predict=args.n_predict, temp=args.temp,
                      top_k=args.top_k, top_p=args.top_p, min_p=args.min_p, **penalties, seed=args.seed,
                      reasoning=args.reasoning, reasoning_format="none" if args.reasoning_format == "none" else "auto",
                      chat_template_kwargs=args.chat_template_kwargs, cache_prompt=args.cache_prompt,
                      sse_ping_interval=args.sse_ping_interval, slots=args.slots,
                      cors_origins=[o.strip() for o in args.cors_origins.split(",") if o.strip()],
                      build_info=build_info())
    options.ui, no_ui = web_ui(args)
    options.no_ui = no_ui or options.no_ui

    def load() -> tuple[Model, NpuModel]:
        t0 = time.perf_counter()
        opts = {"kv_k": args.kv_k, "mmproj": args.mmproj, "max_slices": args.max_slices}
        model = open_model(args.model, **{k: v for k, v in opts.items() if v})
        if args.reasoning == "on" and not model.thinks:
            log.warning("%s has no thinking mode: --reasoning on has no effect", model.name)
        npu = model.load(capacity=n_ctx, prefill=not args.no_prefill)
        log.info("%s (%s): loaded in %.1f s: %s", name, model.name, time.perf_counter() - t0, npu.describe())
        return model, npu

    server: Server | None = None

    def failed() -> None:
        if server is not None:
            server.should_exit = True

    def ready() -> None:
        if report is None:
            return
        while server is not None and not server.started and not server.should_exit:  # (and uvicorn listens)
            time.sleep(0.01)
        report({"ready": {"name": engine.model.name, "n_ctx": engine.npu.capacity, "describe": engine.npu.describe()}})

    engine = Engine(load, warmup=args.warmup, on_ready=ready, on_exit=failed)
    config = uvicorn.Config(create_app(engine, options, info), host=args.host, port=args.port, log_config=None,
                            access_log=args.verbose, timeout_keep_alive=int(args.timeout), lifespan="off")
    server = Server(config, engine.stop)
    sock = listen(config)
    if report is not None:
        report({"port": sock.getsockname()[1]})

        def watch() -> None:  # (the router closes this process's stdin to stop it; so does its end)
            sys.stdin.read()
            server.handle_exit(signal.SIGTERM, None)

        threading.Thread(target=watch, name="router", daemon=True).start()
    log.info("loading %s with %d KV cache positions (%s; the API at /v1)", name, n_ctx,
             "the web UI at /" if options.ui else "no web UI")
    engine.start()
    try:
        server.run(sockets=[sock])
    finally:
        for s in (signal.SIGINT, signal.SIGTERM):  # (while the worker ends)
            signal.signal(s, lambda *_: log.warning("stopping: waiting for the NPU's dispatch to finish"))
        engine.stop()
        engine.join()
    if engine.error:
        if report is not None:
            report({"error": engine.error})
        sys.exit(f"artlite-server: {name}: {engine.error}")


if __name__ == "__main__":
    main()
