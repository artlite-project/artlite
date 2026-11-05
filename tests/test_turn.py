"""The NPU's turn between processes (``csrc/artlite/turn.h``) on the host (a lock file: no NPU): another process that
used the turn is a peer (``turn_peers``) until it ends; a wait for another's turn is counted (``turn_waited``)."""

import subprocess
import sys
import threading
import time

from artlite import _native

PEER = """
import sys
from artlite import _native
_native.turn_enter()
_native.turn_leave()
print("ready", flush=True)
sys.stdin.read()
"""


def test_peers() -> None:
    before = _native.turn_peers()  # (another NPU process of the machine is a peer too)
    p = subprocess.Popen([sys.executable, "-c", PEER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "ready"
        assert _native.turn_peers()
        assert not _native.turn_contended()  # (it holds no turn, nor waits for one)
    finally:
        p.stdin.close()
        p.wait(30)
    t0 = time.monotonic()
    while not before and _native.turn_peers():  # (its locks go with its descriptor; others' processes may end later)
        assert time.monotonic() - t0 < 5
        time.sleep(0.01)

HOLDER = """
import sys
from artlite import _native
_native.turn_enter()
print("holding", flush=True)
sys.stdin.readline()
print("contended" if _native.turn_contended() else "alone", flush=True)
_native.turn_leave()
sys.stdin.read()
"""


def test_waited() -> None:
    """A process waiting for another's turn: a waiter (``turn_contended`` in the holder), the wait counted
    (``turn_waited``: what timings of its own work subtract), the turn then a foreign one."""
    p = subprocess.Popen([sys.executable, "-c", HOLDER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "holding"
        w0, f0 = _native.turn_waited(), _native.turn_foreign()

        def release() -> None:
            time.sleep(0.3)
            p.stdin.write("\n")
            p.stdin.flush()

        threading.Thread(target=release).start()
        t0 = time.perf_counter()
        _native.turn_enter()
        t = time.perf_counter() - t0
        _native.turn_leave()
        assert p.stdout.readline().strip() == "contended"
        waited = _native.turn_waited() - w0
        assert 0.25 < waited <= t + 1e-3 and _native.turn_foreign() == f0 + 1, (waited, t)
        w1 = _native.turn_waited()
        _native.turn_enter()  # (no other holder: no wait counted)
        _native.turn_leave()
        assert _native.turn_waited() == w1
    finally:
        p.stdin.close()
        p.wait(30)
