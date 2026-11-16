"""The operating system's part of the Python side (Linux, Windows): executable names, file locks, process groups, a
thread's CPU, the memory the process holds and the system has left, a pipe's buffer, the time since the system
started, how the NPU wakes.
(A file mapping's page hints are the native module's: ``_native.page_hint``.)"""

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
# Dispatches of an overlay's prologue after the NPU may have powered down (``overlay.OverlayDevice._fresh``). Linux's
# driver resumes the device and re-creates the context before the dispatch runs: one. On Windows the dispatch that
# wakes the array does not reliably leave the stores it writes there, the next one does (Krackan Point, Ryzen AI 1.9).
WAKE_DISPATCHES = 2 if WINDOWS else 1

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


def uptime() -> float:
    """Seconds since the system started, its sleeps included (Linux: CLOCK_BOOTTIME, as ``time.monotonic`` stands
    still there while the system sleeps)."""
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    return time.monotonic() if clock is None else time.clock_gettime(clock)


def process_memory() -> tuple[int, int] | None:
    """(bytes of RAM this process holds, the most it held so far): Linux's VmRSS and VmHWM, Windows' working set and
    its peak; None when unknown. The NPU's buffers are among them (pinned memory of the process), a mapped model
    file's pages only while the process has them mapped."""
    if WINDOWS:
        import ctypes

        class Counters(ctypes.Structure):  # (PROCESS_MEMORY_COUNTERS)
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + \
                [(name, ctypes.c_size_t) for name in ("peak", "now", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
                                                      "pagefile", "peak_pagefile")]

        pm = Counters()
        pm.cb = ctypes.sizeof(pm)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        return (int(pm.now), int(pm.peak)) if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pm), pm.cb) \
            else None
    try:
        with open("/proc/self/status") as f:
            kb = {k: int(v.split()[0]) for k, _, v in (line.partition(":") for line in f) if k in ("VmRSS", "VmHWM")}
        return kb["VmRSS"] << 10, kb["VmHWM"] << 10
    except (OSError, KeyError, ValueError):
        return None


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


def widen_pipe(fd: int, nbytes: int) -> None:
    """A pipe's buffer raised to hold ``nbytes`` (Linux: up to the system's ``pipe-max-size``, 1 MB by default), so that
    a child writes that much without waiting for the reader; elsewhere (or refused) it stays as it is."""
    setsize = getattr(fcntl, "F_SETPIPE_SZ", None) if not WINDOWS else None
    if setsize is None:
        return
    try:
        with open("/proc/sys/fs/pipe-max-size") as f:
            nbytes = min(nbytes, int(f.read()))
        fcntl.fcntl(fd, setsize, nbytes)
    except (OSError, ValueError):
        pass
