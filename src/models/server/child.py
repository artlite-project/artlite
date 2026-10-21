"""A model's server under artlite-server's router (``router``): ``python -m models.server.child`` imports the server
and the models first, then reads its command line as one JSON line from stdin and serves its model (``main.main``):
the router keeps one such process ready, so that a model's load does not wait for the interpreter and its imports
(~1.5 s).

The protocol: the router writes the command line (``--port 0``: the server reports the port it bound) and keeps stdin
open; its end - the router closes it, or the router is gone - shuts the server down as SIGTERM does. The server writes
its messages to the router as JSON lines on stdout (``{"port": N}``, ``{"ready": {...}}``, ``{"error": "..."}``)
and its log to stderr; whatever else writes to stdout (a library, a compiler's output) goes to stderr too."""

import contextlib
import importlib
import json
import os
import sys
import threading
from typing import Any


def main() -> None:
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)  # (the messages; fd 1 becomes the log's)
    os.dup2(2, 1)
    lock = threading.Lock()

    def report(message: dict[str, Any]) -> None:
        with lock, contextlib.suppress(OSError):  # (the router is gone: stdin's end stops this server)
            out.write(json.dumps(message) + "\n")

    from .. import ARCHITECTURES
    from . import main as server

    for module, _ in ARCHITECTURES.values():
        importlib.import_module(module)
    line = sys.stdin.readline()
    if not line.strip():  # (the router no longer needs this process)
        return
    server.main(json.loads(line), report=report)


if __name__ == "__main__":
    main()
