"""The operating system's part of the Python side (Linux, Windows): executable names, file locks, process groups.
(``gguf_model``'s page hints test for ``mmap.madvise`` themselves.)"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

WINDOWS = sys.platform == "win32"
EXE = ".exe" if WINDOWS else ""

if WINDOWS:
    import msvcrt

    def _lock(fd: int) -> None:
        while True:  # (LK_LOCK would retry once a second and give up after 10)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.02)

    def _unlock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(path: str) -> Iterator[None]:
    """An exclusive lock between processes and threads (each holder opens the file itself): flock, on Windows a lock of
    the file's first byte (its position, which stays 0)."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _lock(fd)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def group_options() -> dict[str, Any]:
    """``subprocess.Popen`` options of a child that leads a process group of its own (``kill_group``)."""
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}


def kill_group(proc: subprocess.Popen) -> None:
    """Kill a child started with ``group_options`` while it runs, and the processes it started (Windows: its process
    tree); then reap it."""
    if proc.poll() is None:
        if WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
            if proc.poll() is None:
                proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
