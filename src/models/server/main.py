"""artlite-server: the command line, the engine and uvicorn, the shutdown. The options follow llama.cpp's
``llama-server`` where it has an equivalent; each also reads ``ARTLITE_ARG_<NAME>`` (the long option's name in upper
case, dashes as underscores: ``ARTLITE_ARG_CTX_SIZE``), which the command line overrides.

    artlite-server -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from types import FrameType

import uvicorn

from artlite.model_config import MODELS_DIR, model_path

from .. import open_model
from ..common import Model, NpuModel
from . import ui
from .api import ModelInfo, Options, create_app
from .engine import Engine

log = logging.getLogger("artlite.server")

ENV = "ARTLITE_ARG_"
DEFAULT_CTX = 32768  # -c 0 (capped at the model's context)


def arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="artlite-server", description="An OpenAI-compatible server and web chat for a "
                                "GGUF model on the NPU (llama-server's API and options where they apply)")
    p.add_argument("-m", "--model", required=True, type=model_path,
                   help=f"GGUF file; a relative path is also looked up in the models directory ({MODELS_DIR}, "
                        f"ARTLITE_MODELS_DIR)")
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
    gen.add_argument("-s", "--seed", type=int, default=-1, help="sampling seed (default: -1, random per request)")
    gen.add_argument("-rea", "--reasoning", choices=("on", "off", "auto"), default="off",
                     help="let a model with a thinking mode think first (auto: when it has one; default: off); a "
                          "request's chat_template_kwargs.enable_thinking overrides it")
    gen.add_argument("--reasoning-format", choices=("auto", "deepseek", "none"), default="auto",
                     help="thoughts in reasoning_content (auto, deepseek) or in content (none) (default: auto)")
    gen.add_argument("--chat-template-kwargs", type=json.loads, default={},
                     help="chat template variables as a JSON object, under each request's (enable_thinking)")
    npu = p.add_argument_group("NPU")
    npu.add_argument("--no-prefill", action="store_true", help="run prompts as decode steps")
    npu.add_argument("--kv-k", choices=("single", "hilo"),
                     help="K precision of the KV cache (Llama, Gemma), as artlite-cli's --kv-k")
    npu.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                     help="run one token after loading (default: on)")
    out = p.add_argument_group("logging")
    out.add_argument("-v", "--verbose", action="store_true", help="debug messages and every HTTP request")
    out.add_argument("--log-file", help="also write the log to this file")
    with_env(p)
    return p


def with_env(p: argparse.ArgumentParser) -> None:
    """Each option's default from ``ARTLITE_ARG_<NAME>`` when it is set."""
    for a in p._actions:
        longs = sorted((s for s in a.option_strings if s.startswith("--")), key=lambda s: s.startswith("--no-"))
        if not longs or a.dest == "help":
            continue
        name = ENV + longs[0][2:].upper().replace("-", "_")
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


class Server(uvicorn.Server):
    """uvicorn, whose shutdown (SIGINT, SIGTERM) first cancels the engine's jobs - the running one stops at its next
    token or prompt chunk - and never ends the process during a dispatch (AGENTS.md: a process holding an NPU context
    is never killed): a second signal only stops waiting for the connections."""

    def __init__(self, config: uvicorn.Config, engine: Engine):
        super().__init__(config)
        self.engine = engine

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if self.should_exit:
            log.warning("stopping: waiting for the NPU's dispatch to finish")
            self.force_exit = True
        self.engine.stop()
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


def main(argv: list[str] | None = None) -> None:
    p = arg_parser()
    args = p.parse_args(argv)
    if args.parallel != 1:
        p.error("-np/--parallel: only 1 (the NPU runs one sequence at a time)")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.log_file:
        handler = logging.FileHandler(args.log_file)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)
        logging.getLogger().addHandler(handler)
    name = os.path.basename(args.model)
    try:
        info = ModelInfo.from_gguf(args.model)
        keys = api_keys(args)
    except (OSError, ValueError) as e:
        sys.exit(f"artlite-server: {e}")
    n_ctx = capacity(args.ctx_size, info.n_ctx_train)
    options = Options(alias=args.alias or name, n_ctx=n_ctx, api_keys=keys, n_predict=args.n_predict, temp=args.temp,
                      top_k=args.top_k, top_p=args.top_p, min_p=args.min_p, seed=args.seed, reasoning=args.reasoning,
                      reasoning_format="none" if args.reasoning_format == "none" else "auto",
                      chat_template_kwargs=args.chat_template_kwargs, cache_prompt=args.cache_prompt,
                      sse_ping_interval=args.sse_ping_interval, slots=args.slots,
                      cors_origins=[o.strip() for o in args.cors_origins.split(",") if o.strip()],
                      build_info=build_info())
    if args.no_ui:
        options.no_ui = "The web UI is off (--no-ui): the API is at /v1."
    else:
        try:
            directory = args.path or ui.fetch()
            options.ui = lambda app: ui.mount(app, directory)
        except (OSError, ValueError) as e:
            log.warning("no web UI: %s", e)
            options.no_ui = (f"The web UI could not be fetched ({e}). It is llama.cpp's prebuilt UI, build {ui.TAG}, "
                             f"downloaded from {ui.URL.format(tag=ui.TAG)} on the first start; serve a local build with "
                             f"--path DIR (llama.cpp's build/tools/ui/dist), or run with --no-ui. The API is at /v1.")

    def load() -> tuple[Model, NpuModel]:
        t0 = time.perf_counter()
        model = open_model(args.model, **({"kv_k": args.kv_k} if args.kv_k else {}))
        if args.reasoning == "on" and not model.thinks:
            log.warning("%s has no thinking mode: --reasoning on has no effect", model.name)
        npu = model.load(capacity=n_ctx, prefill=not args.no_prefill)
        log.info("%s (%s): loaded in %.1f s: %s", name, model.name, time.perf_counter() - t0, npu.describe())
        return model, npu

    server: Server | None = None

    def failed() -> None:
        if server is not None:
            server.should_exit = True

    engine = Engine(load, warmup=args.warmup, on_exit=failed)
    config = uvicorn.Config(create_app(engine, options, info), host=args.host, port=args.port, log_config=None,
                            access_log=args.verbose, timeout_keep_alive=int(args.timeout), lifespan="off")
    server = Server(config, engine)
    sock = config.bind_socket()  # (a busy port ends the process before the model loads)
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
        sys.exit(f"artlite-server: {name}: {engine.error}")


if __name__ == "__main__":
    main()
