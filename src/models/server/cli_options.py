"""artlite-server's command line: the option table (``arg_parser``: llama-server's options where it has an
equivalent; each also reads ``ARTLITE_ARG_<NAME>``, the long option's name in upper case, dashes as underscores:
``ARTLITE_ARG_CTX_SIZE``, which the command line overrides), the sets of options that belong to the router or to one
model's server, and what the options turn into (the KV cache's size, the API keys, the web UI). Shared by ``main`` (one
model's server), ``router`` (router mode) and ``catalog`` (a model's options as presets and arguments)."""

import argparse
import json
import logging
from collections.abc import Callable

from fastapi import FastAPI

from artlite import settings
from artlite.model_config import MODELS_DIR, model_path

from ..readouts import POOLING
from ..spec_options import add_spec_options
from . import ui

log = logging.getLogger("artlite.server")

DEFAULT_CTX = 32768  # -c 0 (capped at the model's context)
# the router's own options (in router mode each model's server gets its own: ``router``)
ROUTER = frozenset({"host", "port", "api_key", "api_key_file", "cors_origins", "path", "no_ui", "timeout", "log_file",
                    "models_dir", "models_preset", "models_max", "models_autoload"})
MODELS = frozenset({"models_dir", "models_preset", "models_max", "models_autoload"})  # (router mode's)
# one model's (in router mode: the router's, a preset's), of them its files (``FILES``: a preset's found as ``model``)
MODEL = frozenset({"model", "alias", "mmproj", "spec_draft_model"})
FILES = frozenset({"mmproj", "spec_draft_model"})


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
    gen.add_argument("-cram", "--cache-ram", type=int,
                     help="MiB of host RAM for the prompt cache: the conversations it saves and the checkpoints "
                          "(default: 8192, at most half of the RAM still available once the model is loaded, and in "
                          "router mode that over --models-max; -1: no limit, 0: the device's conversation alone)")
    gen.add_argument("-ctxcp", "--ctx-checkpoints", type=int, default=32,
                     help="checkpoints of a conversation at most (the states of Qwen3.5 / 3.6 / 3.8, MiniCPM-V and "
                          "LFM3 that a prompt continues from; default: 32)")
    gen.add_argument("--lookup", action=argparse.BooleanOptionalAction, default=True,
                     help="draft tokens from the context (prompt lookup) for the multi-token steps (default: on)")
    add_spec_options(gen)
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
    gen.add_argument("--reasoning-preserve", action=argparse.BooleanOptionalAction, default=True,
                     help="earlier answers keep their thoughts in the prompt where the chat template can (Qwen3.6, "
                          "Gemma 4): every turn extends the cached tokens (default: on); a request's "
                          "chat_template_kwargs.preserve_reasoning overrides it")
    gen.add_argument("--reasoning-effort", metavar="LEVEL", default="default",
                     help="the chat template's reasoning_effort (Qwen3.8's: xhigh, medium, low; default: default, the "
                          "template's own); a request's reasoning_effort overrides it")
    gen.add_argument("--chat-template-kwargs", type=json.loads, default={},
                     help="chat template variables as a JSON object, under each request's (enable_thinking, "
                          "reasoning_effort, ...)")
    gen.add_argument("--chat-template", metavar="JINJA_TEMPLATE",
                     help="a Jinja chat template in place of the model file's (tokenizer.chat_template)")
    gen.add_argument("--chat-template-file", metavar="FILE",
                     help="the same from a file (one template for several models: a preset's sections)")
    vis = p.add_argument_group("images and audio (Qwen3.5 / 3.6 / 3.8, MiniCPM-V 4.7, LFM3-VL, LFM3-Audio)")
    vis.add_argument("--mmproj", type=model_path, help="the vision or speech encoder's GGUF file (mmproj-*.gguf): images "
                     "in chat messages (image_url parts of data URIs), audio (input_audio parts, "
                     "/v1/audio/transcriptions)")
    vis.add_argument("--max-slices", type=int,
                     help="an image's slices at most (MiniCPM-V; default: the mmproj's, 9; the model card uses 36)")
    vis.add_argument("--image-min-tokens", type=int,
                     help="an image's tokens at least (Qwen3-VL's, LFM3-VL's mmproj; default 8 and 64, llama.cpp's)")
    vis.add_argument("--image-max-tokens", type=int,
                     help="an image's tokens at most (Qwen3-VL's, LFM3-VL's mmproj; default 4096, llama.cpp's)")
    emb = p.add_argument_group("embeddings and rerank")
    emb.add_argument("--embedding", "--embeddings", action="store_true",
                     help="the embedding routes (/v1/embeddings, /embeddings), as llama-server's: --pooling's, else the "
                          "model file's (a generative model has none: one row per token, /embeddings only)")
    emb.add_argument("--pooling", choices=POOLING,
                     help="the embeddings' pooling (default: the model file's)")
    emb.add_argument("--rerank", "--reranking", action="store_true",
                     help="the rerank routes (/v1/rerank, ...): --embedding with pooling rank (a reranker's file: its "
                          "classifier head)")
    emb.add_argument("--embd-normalize", type=int, default=2,
                     help="the embeddings' normalization (-1: none, 0: max absolute as int16, 2: Euclidean, any other "
                          "p: the p-norm; default: 2; a request's embd_normalize overrides it)")
    npu = p.add_argument_group("NPU")
    npu.add_argument("--no-prefill", action="store_true", help="run prompts as decode steps")
    npu.add_argument("--kv-k", choices=("single", "hilo"),
                     help="K precision of the KV cache (Llama, Gemma, LFM3), as artlite-cli's --kv-k")
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
    return None if not longs or a.dest == "help" else settings.arg_name(longs[0][2:].upper().replace("-", "_"))


def with_env(p: argparse.ArgumentParser) -> None:
    """Each option's default from ``ARTLITE_ARG_<NAME>`` when it is set."""
    for a in p._actions:
        name = env_name(a)
        if name is None:
            continue
        a.help = f"{a.help} (env: {name})" if a.help else f"(env: {name})"
        if (value := settings.arg_read(name)) is None:
            continue
        if a.nargs == 0:  # (a switch)
            a.default = settings.arg_flag(value)
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


def parse_cors(origins: str) -> list[str]:
    """``--cors-origins``: the comma-separated origins."""
    return [o.strip() for o in origins.split(",") if o.strip()]


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
