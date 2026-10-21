"""Parallel compilation: the single place where Artlite compiles several designs at once.

Compiling an operator has two parts:

- **rendering** (``build_spec()``): Python builds the kernel source and the MLIR design. aie.iron is
  not thread-safe, so this runs in the calling thread. The result is a ``BuildSpec`` of plain
  strings (picklable, hashable).
- **building** (``operators.base.build``): clang++ and aiecc subprocesses write
  ``CACHE_DIR/<kind>_<key>/``. Concurrent builds of one key wait for each other on a lock file and
  then find the cache, so a build may run in any thread or process.

This module runs the builds on one process-wide pool of ``jobs()`` threads (``ARTLITE_COMPILE_JOBS``,
default: the CPU count; 1 builds serially):

- ``compile_all(operators)``: compile ArtcOperators and fused ELFs (``FusedOperator``, ``Library``), the
  builds in parallel (their ``compile()`` runs it for one). Use it wherever several designs are compiled
  before any of them runs (apps, scripts, experiments). A design whose build the render manifest knows is
  not rendered (``operators.render_cache``).
- ``parallel_map(fn, items)``: other compile-like work (kernel-only compiles, variant builds).
- ``BuildQueue``: background builds that wait behind the foreground ones (the tests' precompilation).
- ``Precompiler(cmd)``: run a command (tests, an app, a script) in planning mode in a subprocess
  (``ARTLITE_PLANNING=1``: every ``compile()`` hands its spec to this process instead of building, the
  first access to a device raises ``PlanStop``; its subprocesses inherit it and plan too) and build the designs
  it renders in the background, so that the real run finds them in the cache.
  The test session does this for the selected tests (``tests/rtc/conftest.py``). From a shell:
  ``python -m artlite.parallel_compile CMD ...`` builds everything CMD would compile, then exits.

A task already running on the pool runs a nested ``parallel_map`` / ``compile_all`` inline.
"""

from __future__ import annotations

import contextlib
import gc
import heapq
import itertools
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection, Listener
from typing import TYPE_CHECKING, Any

from . import portable
from .operators.base import BuildSpec, build, buildable

if TYPE_CHECKING:
    from . import Device
    from .operators.base import ArtcOperator
    from .operators.fused.elf import FusedElf

JOBS_ENV = "ARTLITE_COMPILE_JOBS"
PLANNING_ENV = "ARTLITE_PLANNING"
PLAN_SINK_ENV = "ARTLITE_PLAN_SINK"  # "<key>:<address>": the Precompiler's listener that its planning processes send to
FOREGROUND, BACKGROUND = 0, 1  # pool priorities


def jobs() -> int:
    """Concurrent builds: ``ARTLITE_COMPILE_JOBS``, default the CPU count."""
    value = os.environ.get(JOBS_ENV)
    if value is None:
        return os.cpu_count() or 1
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{JOBS_ENV}={value!r}: expected a positive integer")
    return int(value)


def aiecc_jobs(concurrent: int) -> int:
    """aiecc compile threads for one of ``concurrent`` simultaneous builds (they share the CPUs)."""
    return max(1, (os.cpu_count() or 1) // max(1, concurrent))


class _Pool:
    """Worker threads taking tasks by priority, then in submission order (a build thread only waits
    for its subprocesses, so threads suffice)."""

    def __init__(self, workers: int):
        self.workers = workers
        self._heap: list[tuple[int, int, Future, Callable[[], Any]]] = []
        self._cv = threading.Condition()
        self._seq = itertools.count()
        for i in range(workers):
            threading.Thread(target=self._work, name=f"artlite-compile-{i}", daemon=True).start()

    def submit(self, fn: Callable[[], Any], *, priority: int = FOREGROUND) -> Future:
        fut: Future = Future()
        with self._cv:
            heapq.heappush(self._heap, (priority, next(self._seq), fut, fn))
            self._cv.notify()
        return fut

    def _work(self) -> None:
        _local.in_pool = True
        while True:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                _, _, fut, fn = heapq.heappop(self._heap)
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 (handed to the waiter)
                fut.set_exception(exc)


_local = threading.local()
_pool: _Pool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> _Pool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _Pool(jobs())
        return _pool


def parallel_map[T, R](fn: Callable[[T], R], items: Iterable[T], *, limit: int | None = None) -> list[R]:
    """``[fn(item) for item in items]`` on the pool: results in order. Every call finishes before an
    exception (the first one) is raised. limit: at most this many of these items at a time (default:
    the pool's ``jobs()``)."""
    items = list(items)
    width = min(limit or jobs(), jobs(), len(items))
    if width <= 1 or getattr(_local, "in_pool", False):
        return [fn(item) for item in items]
    pool = _get_pool()
    slots = threading.BoundedSemaphore(width)
    futs = []
    for item in items:
        slots.acquire()
        fut = pool.submit(lambda item=item: fn(item))
        fut.add_done_callback(lambda _: slots.release())
        futs.append(fut)
    errors = [f.exception() for f in futs]
    first = next((e for e in errors if e is not None), None)
    if first is not None:
        raise first
    return [f.result() for f in futs]


def compile_all(operators: Sequence[ArtcOperator | FusedElf], *, device: Device | None = None,
                limit: int | None = None) -> None:
    """Compile ArtcOperators and fused ELFs: render each (in this thread) unless the render manifest knows its
    build (``operators.render_cache``), build them in parallel, then attach the artifacts. ``device`` goes to the
    ArtcOperators."""
    from .operators import render_cache

    ops = list(operators)
    dirs = [render_cache.lookup(op) for op in ops]  # (a hit is built: nothing to render, plan or build)
    todo = [i for i, d in enumerate(dirs) if d is None]
    specs = [ops[i].build_spec() for i in todo]
    if planning_active():
        for spec in specs:
            planned(spec)
        for op in ops:
            op._mark_planned()
        return
    if specs:
        # a rendered design's aie.iron objects are reference cycles that hold the operator (its core and sequence
        # functions are closures over it) until the cycle collector runs; meanwhile a dropped operator keeps its
        # hardware context, and the device has 16 for all processes (CREATE_HWCTX fails with EINVAL)
        gc.collect()
    width = min(limit or jobs(), jobs(), len(specs))
    built = parallel_map(lambda spec: build(spec, aiecc_jobs=aiecc_jobs(width) if width > 1 else None), specs,
                         limit=limit)
    for i, work_dir in zip(todo, built):
        render_cache.store(ops[i], work_dir)
        dirs[i] = work_dir
    for op, work_dir in zip(ops, dirs):
        op._attach(work_dir, device)


class BuildQueue:
    """Background builds of specs, in submission order, after any foreground work on the pool (each
    spec once). A failed build is only counted: the operator's own ``compile()`` reports it.
    ``close()`` before the process exits."""

    def __init__(self) -> None:
        self._seen: set[BuildSpec] = set()
        self._futs: list[Future] = []
        self._lock = threading.Lock()
        self.failed = 0

    def submit(self, spec: BuildSpec) -> None:
        """(a spec this process does not build as rendered is left to its own: ``buildable``)"""
        if not buildable(spec):
            return
        with self._lock:
            if spec in self._seen:
                return
            self._seen.add(spec)
        pool = _get_pool()

        def run() -> None:
            try:  # (no ARTLITE_MEASURE_II report: the operator's own compile prints it)
                build(spec, aiecc_jobs=aiecc_jobs(pool.workers), measure=False)
            except Exception:  # noqa: BLE001 (counted: the operator's own compile reports it)
                with self._lock:
                    self.failed += 1

        fut = pool.submit(run, priority=BACKGROUND)
        with self._lock:
            self._futs.append(fut)

    @property
    def submitted(self) -> int:
        return len(self._seen)

    def wait(self) -> None:
        """Wait until every submitted build has finished."""
        with self._lock:
            futs = list(self._futs)
        for fut in futs:
            fut.exception()

    def close(self) -> None:
        """Cancel the builds that have not started and wait for the running ones (a killed build
        could leave a partial cache entry)."""
        with self._lock:
            futs = list(self._futs)
        for fut in futs:
            fut.cancel()
        for fut in futs:
            if not fut.cancelled():
                fut.exception()


# -- Planning mode ------------------------------------------------------------------------------------------------

class PlanStop(BaseException):
    """Raised in planning mode where the code would need a built design or the NPU (a BaseException:
    ``except Exception`` blocks let it through)."""


class _Planned:
    """The program of an operator compiled in planning mode: any use raises PlanStop."""

    def __getattr__(self, name: str) -> Any:
        raise PlanStop(f"planning: {name} needs a built design")


PLANNED: Any = _Planned()


def _sink_from_env() -> Callable[[BuildSpec], None] | None:
    """A process started in planning mode (or a subprocess of one): the specs go to its Precompiler's listener (an
    authenticated connection, opened at the first), else nowhere."""
    if os.environ.get(PLANNING_ENV) != "1":
        return None
    key, _, address = os.environ.get(PLAN_SINK_ENV, "").partition(":")
    if not address:
        return lambda spec: None
    lock = threading.Lock()
    conn: Connection | None = None

    def sink(spec: BuildSpec) -> None:
        nonlocal conn
        with lock, contextlib.suppress(OSError, EOFError, AuthenticationError):  # (the Precompiler stopped listening)
            if conn is None:
                conn = Client(address, authkey=bytes.fromhex(key))
            conn.send(spec)

    return sink


_plan_sink: Callable[[BuildSpec], None] | None = _sink_from_env()


def planning_active() -> bool:
    return _plan_sink is not None


def planned(spec: BuildSpec) -> bool:
    """In planning mode: hand ``spec`` to the sink and return True (``compile()`` then builds nothing)."""
    sink = _plan_sink
    if sink is None:
        return False
    sink(spec)
    return True


def no_device() -> None:
    """Called before opening the NPU: raises PlanStop in planning mode."""
    if _plan_sink is not None:
        raise PlanStop("planning: the NPU is not used")


class Precompiler:
    """Run ``cmd`` in planning mode in a subprocess (it builds nothing and never opens the NPU) and
    build every design it renders in the background (``BuildQueue``), while this process goes on. The
    specs come over a listener (a Unix socket, on Windows a named pipe) that the planning processes connect
    to with the key in their environment. ``close()`` stops the planning process (safe to kill: it holds no
    NPU context) and the builds that have not started; ``wait()`` lets both finish."""

    def __init__(self, cmd: Sequence[str], *, cwd: str | None = None):
        key = self._key = os.urandom(32)
        self._listener = Listener(authkey=key)
        self._stopping = False
        self._readers: list[threading.Thread] = []
        env = {**os.environ, PLANNING_ENV: "1", PLAN_SINK_ENV: f"{key.hex()}:{self._listener.address}"}
        self._proc = subprocess.Popen(list(cmd), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      **portable.group_options())
        self.queue = BuildQueue()
        self._accepter = threading.Thread(target=self._accept, name="artlite-plan", daemon=True)
        self._accepter.start()

    def _accept(self) -> None:
        """A reader per planning process that connects, until ``_stop`` (its own connection ends the wait)."""
        while True:
            try:
                conn = self._listener.accept()
            except (OSError, EOFError, AuthenticationError):
                if self._stopping:
                    return
                continue
            if self._stopping:
                conn.close()
                return
            reader = threading.Thread(target=self._read, args=(conn,), name="artlite-plan-read", daemon=True)
            reader.start()
            self._readers.append(reader)

    def _read(self, conn: Connection) -> None:
        with conn:
            while True:
                try:
                    spec = conn.recv()
                except (OSError, EOFError):
                    return
                self.queue.submit(spec)

    def _stop(self) -> None:
        """No more connections (one of its own ends the listener's wait); wait for the readers (their processes
        ended)."""
        if self._stopping:
            return
        self._stopping = True
        with contextlib.suppress(OSError, EOFError, AuthenticationError):
            Client(self._listener.address, authkey=self._key).close()
        self._accepter.join()
        self._listener.close()
        for reader in self._readers:
            reader.join()

    def wait(self) -> None:
        """Wait for the planning process to end and every design it rendered to be built."""
        self._proc.wait()
        self._stop()
        self.queue.wait()

    def close(self) -> None:
        """Stop planning (the process group: its subprocesses plan too, none holds an NPU context),
        cancel the builds not started, wait for the running ones."""
        portable.kill_group(self._proc)
        self._stop()
        self.queue.close()


def main(argv: Sequence[str] | None = None) -> None:
    """python -m artlite.parallel_compile CMD [ARGS...]: build every design CMD would compile (CMD runs
    in planning mode: no NPU, nothing else built), ARTLITE_COMPILE_JOBS at a time; then run CMD
    normally, it finds them in the cache. E.g. ``python -m artlite.parallel_compile python app.py``."""
    cmd = list(sys.argv[1:] if argv is None else argv)
    if not cmd or cmd[0] in ("-h", "--help"):
        print(main.__doc__)
        sys.exit(0 if cmd else 2)
    pre = Precompiler(cmd)
    try:
        pre.wait()
    finally:
        pre.close()
    print(f"artlite.parallel_compile: {pre.queue.submitted} design(s), {pre.queue.failed} failed "
          f"(their compile() reports why), {jobs()} at a time", file=sys.stderr)
    sys.exit(1 if pre.queue.failed else 0)


if __name__ == "__main__":
    main()
