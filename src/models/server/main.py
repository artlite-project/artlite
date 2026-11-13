"""artlite-server: the command line, the engine and uvicorn, the shutdown. The options follow llama.cpp's
``llama-server`` where it has an equivalent; each also reads ``ARTLITE_ARG_<NAME>`` (the long option's name in upper
case, dashes as underscores: ``ARTLITE_ARG_CTX_SIZE``), which the command line overrides. With ``-m`` the server
serves that model; without, the models of a directory, each in a server of its own (``router``):

    artlite-server -m lmstudio-community/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf
    artlite-server --models-dir ~/models
"""

import argparse
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import uvicorn

from .. import open_model
from ..hostenv import performance_mode, stale_install
from ..model import Model
from ..npu_model import NpuModel
from ..sampling import Sampling
from ..spec_options import spec_options
from ..templating import TEMPLATE_RESERVED, check_chat_template, template_option
from .api import create_app
from .cli_options import MODELS, SHARE, api_keys, arg_parser, capacity, given, web_ui
from .engine import Engine
from .process import Server, listen, setup_logging
from .router import serve
from .settings import PENALTIES, ModelInfo, Options

log = logging.getLogger("artlite.server")

Report = Callable[[dict[str, Any]], None]  # a router's child: a message to the router


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
        if stale := stale_install():
            log.warning("%s", stale)
        performance_mode(log.info)
    if args.model is None:
        serve(p, args, explicit)
        return
    if explicit & MODELS:
        p.error(f"{', '.join(sorted('--' + d.replace('_', '-') for d in explicit & MODELS))}: router mode's options "
                f"(without -m)")
    serve_model(args, report)


def serve_model(args: argparse.Namespace, report: Report | None) -> None:
    """One model's server (``-m``)."""
    name = os.path.basename(args.model)
    effort = None if args.reasoning_effort == "default" else args.reasoning_effort
    try:
        template = template_option(args.chat_template, args.chat_template_file)
        info = ModelInfo.from_gguf(args.model)
        if template is not None:
            info = dataclasses.replace(info, chat_template=template)
        variables = {k: v for k, v in args.chat_template_kwargs.items() if k not in TEMPLATE_RESERVED} | \
            ({"reasoning_effort": effort} if effort else {})
        if info.chat_template and (template is not None or variables):  # (the file's own, alone: as it is)
            check_chat_template(info.chat_template, **variables)
        keys = api_keys(args)
        penalties = {k: getattr(args, k) for k in PENALTIES if getattr(args, k) is not None}
        Sampling(**penalties)  # (ValueError: out of range)
        pooling = "rank" if args.rerank else args.pooling or info.readouts.pooling
        if pooling == "rank" and info.readouts.head is None:
            raise ValueError(f"{name}: rank pooling (--rerank) needs a reranker's file: this one has no classifier head "
                             f"(cls.output)")
    except (OSError, ValueError, NotImplementedError) as e:
        if report is not None:
            report({"error": f"{type(e).__name__}: {e}"})
        sys.exit(f"artlite-server: {e}")
    n_ctx = capacity(args.ctx_size, info.n_ctx_train)
    options = Options.from_args(args, alias=args.alias or name, n_ctx=n_ctx, api_keys=keys, reasoning_effort=effort,
                                penalties=penalties, pooling=pooling)
    options.ui, no_ui = web_ui(args)
    options.no_ui = no_ui or options.no_ui

    def load() -> tuple[Model, NpuModel]:
        t0 = time.perf_counter()
        opts = {"kv_k": args.kv_k, "mmproj": args.mmproj, "max_slices": args.max_slices,
                "image_min_tokens": args.image_min_tokens, "image_max_tokens": args.image_max_tokens, **spec_options(args)}
        model = open_model(args.model, chat_template=template, **{k: v for k, v in opts.items() if v is not None})
        if args.reasoning == "on" and not model.thinks:
            log.warning("%s has no thinking mode: --reasoning on has no effect", model.name)
        npu = model.load(capacity=n_ctx, prefill=not args.no_prefill)
        npu.lookup = args.lookup
        if args.spec_draft_n_max is not None:
            npu.DRAFTS = min(npu.DRAFTS, args.spec_draft_n_max)
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

    engine = Engine(load, warmup=args.warmup, on_ready=ready, on_exit=failed,
                    cache_ram=None if args.cache_ram is None else args.cache_ram << 20 if args.cache_ram >= 0 else -1,
                    share=int(os.environ.get(SHARE) or 1), checkpoints=args.ctx_checkpoints)
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
