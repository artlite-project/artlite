"""The server process: uvicorn's listening socket and server (the shutdown that never ends the process during a NPU
dispatch) and the log."""

import argparse
import contextlib
import logging
import signal
import socket
import threading
from collections.abc import Callable, Iterator
from types import FrameType

import uvicorn

log = logging.getLogger("artlite.server")


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
