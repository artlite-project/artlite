"""The operating system's part of the Python side (Linux, Windows): executable names, file locks, process groups, a
thread's CPU, the memory still available.
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


@contextlib.contextmanager
def on_cpu(cpu: int) -> Iterator[None]:
    """The calling thread on CPU ``cpu`` meanwhile (Linux; if its affinity allows that CPU), its affinity restored
    after; else (cpu < 0, Windows) nothing changes."""
    get, put = getattr(os, "sched_getaffinity", None), getattr(os, "sched_setaffinity", None)
    old = get(0) if get is not None and cpu >= 0 else set()
    if cpu not in old:
        yield
        return
    put(0, {cpu})
    try:
        yield
    finally:
        put(0, old)


def available_memory() -> int | None:
    """Bytes of RAM the system can still give processes without swapping (Linux: MemAvailable; Windows: the available
    physical memory), or None when unknown."""
    if WINDOWS:
        import ctypes

        class Status(ctypes.Structure):  # (MEMORYSTATUSEX)
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                [(name, ctypes.c_ulonglong) for name in ("total", "avail", "total_page", "avail_page", "total_virtual",
                                                          "avail_virtual", "avail_extended")]

        st = Status()
        st.length = ctypes.sizeof(st)
        return int(st.avail) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else None
    try:
        with open("/proc/meminfo") as f:
            return next((int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:")), None)
    except OSError:
        return None
