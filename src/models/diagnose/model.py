"""The diagnosis' model part: Llama 3.2 1B's prompt of 1024 tokens in variants - host buffers of 2 MB pages (user BOs
from 2 MB on) or of 4 KB pages (the driver's BOs below 64 MB), packed C drains in whole 64-byte lines or not
(``prefill.gemm.LINES``), a busy CPU - in two rounds, with each run's state: the chunk's time per operator kind (each
kind alone, in a telemetry step of its own: which kinds the platform throttles), the largest kind at lower duties, a
design switch's cost inside its program, the host's row lookups, the huge pages the process got, how much of the file
the page cache holds."""

from __future__ import annotations

import contextlib
import ctypes
import gc
import mmap
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator

import numpy as np
import torch

from artlite import _native as native
from artlite.gguf_model import GGUFFile
from artlite.prefill import gemm as prefill_gemm

from .. import open_model
from ..bench import Test, file_info, measure, token_source

MB = 1 << 20
FILE = "unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
# (name, host pages, whole-line drains, a busy CPU): V0 is Artlite's own, V4 V0 with one CPU spinning elsewhere (CPU
# load raises the fabric clock on Phoenix; it keeps cores out of deep C-states)
VARIANTS = [("V0: 2 MB pages, whole lines", "2 MB", True, False),
            ("V1: 4 KB pages, whole lines", "4 KB", True, False),
            ("V2: 2 MB pages, partial lines", "2 MB", False, False),
            ("V3: 4 KB pages, partial lines", "4 KB", False, False),
            ("V4: V0 with a busy CPU", "2 MB", True, True)]


def default_file() -> str:
    """The issue's 1B file in the models directory (unsloth's), else lmstudio-community's."""
    from artlite.model_config import model_path

    paths = [model_path(FILE), model_path(FILE.replace("unsloth/", "lmstudio-community/"))]
    return next((p for p in paths if os.path.exists(p)), paths[0])


@contextlib.contextmanager
def variant(pages: str, lines: bool) -> Iterator[None]:
    """Models loaded meanwhile get host buffers of ``pages`` pages and drains of whole lines or not."""
    native.set_user_bo_bytes(2 * MB if pages == "2 MB" else 64 * MB)
    prefill_gemm.LINES = lines
    try:
        yield
    finally:
        native.set_user_bo_bytes(2 * MB)
        prefill_gemm.LINES = True


def resident(path: str) -> float | None:
    """The share of the file's pages in the page cache (mincore; None off Linux)."""
    if sys.platform != "linux":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            m = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
    except (OSError, ValueError):
        return None
    try:
        n = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vec = (ctypes.c_ubyte * n)()
        addr = np.frombuffer(m, np.uint8).ctypes.data  # (a read-only mapping's address)
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec):
            return None
        return sum(v & 1 for v in vec) / n
    finally:
        m.close()


def anon_huge_kb() -> int:
    """AnonHugePages of this process (kB): its user BOs' 2 MB pages and any other THP (0 off Linux)."""
    if sys.platform != "linux":
        return 0
    with open("/proc/self/smaps_rollup") as f:
        return next((int(line.split()[1]) for line in f if line.startswith("AnonHugePages:")), 0)


@contextlib.contextmanager
def busy_cpu(on: bool, avoid: int) -> Iterator[None]:
    """A process spinning on one CPU other than ``avoid`` (the NPU's interrupts') meanwhile, if ``on``."""
    if not on:
        yield
        return
    proc = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        if hasattr(os, "sched_setaffinity"):  # (elsewhere the scheduler places it)
            cpus = sorted(os.sched_getaffinity(0) - {avoid}) or [0]
            os.sched_setaffinity(proc.pid, {cpus[-1]})
        yield
    finally:
        proc.kill()
        proc.wait()


@contextlib.contextmanager
def timed_lookups(into: list[float]) -> Iterator[None]:
    """Each GGUFFile.rows call's seconds into ``into`` meanwhile (the host's embedding lookups of a prompt)."""
    rows = GGUFFile.rows

    def wrapped(self: GGUFFile, *args: object, **kw: object) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            return rows(self, *args, **kw)
        finally:
            into.append(time.perf_counter() - t0)

    GGUFFile.rows = wrapped
    try:
        yield
    finally:
        GGUFFile.rows = rows


def kind_of(pf: object, op: object) -> str:
    """An operator's kind: a GEMM by its phase (where its plan names it), format and shape, else its class."""
    for (_, name), plan in pf.plans.items():
        for g in plan.gemms:
            if g.op is op:
                return f"gemm {name} {g.op.fmt} {g.op.K}x{g.op.N}"
    if hasattr(op, "fmt") and hasattr(op, "K"):
        return f"{type(op).__name__} {op.fmt} {op.K}x{op.N}"
    return type(op).__name__ + (f":{op.mode}" if hasattr(op, "mode") else "")


KIND_SECONDS, IDLE_SECONDS = 1.5, 1.5  # (a kind looped alone; idle before it, so that a throttle it set off lets go)
DUTIES = (1.0, 0.75, 0.5, 0.25)  # (the duty sweep's shares of time the NPU runs)


def _run_ms(pf: object, prog: list, reps: int) -> float:
    """The median ms of a library program run ``reps`` times (after one), in pieces of at most 400 items."""
    bufs = list(pf.ov.dev.buffers.values())
    splits = list(range(400, len(prog), 400))
    t = []
    for _ in range(reps + 1):
        t0 = time.perf_counter()
        pf.lib.run(prog, splits=splits, bufs=bufs)
        t.append(time.perf_counter() - t0)
    return statistics.median(t[1:]) * 1e3


def _loop_ms(pf: object, prog: list, seconds: float) -> float:
    """The median ms of a library program run for ``seconds`` (after one run), in pieces of at most 400 items."""
    bufs = list(pf.ov.dev.buffers.values())
    splits = list(range(400, len(prog), 400))
    pf.lib.run(prog, splits=splits, bufs=bufs)
    t = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or len(t) < 3:
        t0 = time.perf_counter()
        pf.lib.run(prog, splits=splits, bufs=bufs)
        t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e3


def kinds(npu: object, *, step: Callable[[str], None], end: Callable[[str], None], tag: str,
          T: int = 1024) -> dict[str, object]:
    """The chunk program's ms as a whole and per operator kind (each kind's items run as a program of their own:
    wrong results, a configuration per design change), with the items and configurations of each. Each runs for
    KIND_SECONDS after IDLE_SECONDS idle, in a telemetry step of its own (``{tag}: kinds: {kind}``): which kinds the
    platform throttles when they run alone."""
    pf = npu.prefill
    T = min(s for s in pf.sizes if s >= T)
    items = pf._program(T)
    names = [kind_of(pf, op) for op, *_ in pf.runlists[T]]
    asm = pf.lib.load()._asm
    blocks = asm.blocks
    parity = asm.parity
    code = asm.assemble(items)  # (the whole chunk's control code, as one piece would hold it)
    asm.parity = parity

    def configurations(prog: list) -> int:
        return sum(1 for i, (e, _) in enumerate(prog) if i == 0 or blocks[e].design != blocks[prog[i - 1][0]].design)

    out: dict[str, object] = {"T": T, "whole_ms": 0.0, "items": len(items), "configurations": configurations(items),
                              "txn_ops": int(code[2]), "control_bytes": int(code[3]), "kinds": {}}
    progs = {"whole program": items} | {k: [it for it, kk in zip(items, names) if kk == k] for k in sorted(set(names))}
    for k, prog in progs.items():
        time.sleep(IDLE_SECONDS)
        step(f"{tag}: kinds: {k}")
        ms = _loop_ms(pf, prog, KIND_SECONDS)
        end(f"{tag}: kinds: {k}")
        if k == "whole program":
            out["whole_ms"] = ms
        else:
            out["kinds"][k] = {"items": len(prog), "configurations": configurations(prog), "ms": ms}
    return out


def duty_sweep(npu: object, kind: str, *, step: Callable[[str], None], end: Callable[[str], None], tag: str,
               T: int = 1024) -> dict[str, object]:
    """The items of one operator kind of the chunk run one at a time, each followed by sleep for duty d ((1 - d) / d
    of its time) for KIND_SECONDS per duty in DUTIES (telemetry steps ``{tag}: duty {d}``): a throttle that follows
    the NPU's average power over milliseconds lets go at a lower duty (the items then run at full speed), one that
    follows each item's load holds at every duty."""
    pf = npu.prefill
    T = min(s for s in pf.sizes if s >= T)
    items = pf._program(T)
    prog = [it for it, (op, *_) in zip(items, pf.runlists[T]) if kind_of(pf, op) == kind]
    bufs = list(pf.ov.dev.buffers.values())
    out: dict[str, object] = {"kind": kind, "items": len(prog), "duties": {}}
    for d in DUTIES:
        time.sleep(IDLE_SECONDS)
        step(f"{tag}: duty {d:.0%}")
        t = []
        stop = time.perf_counter() + KIND_SECONDS
        while time.perf_counter() < stop or len(t) < 3:
            t0 = time.perf_counter()
            pf.lib.run([prog[len(t) % len(prog)]], splits=[], bufs=bufs)
            t.append(time.perf_counter() - t0)
            if d < 1:
                time.sleep(t[-1] * (1 - d) / d)
        end(f"{tag}: duty {d:.0%}")
        out["duties"][f"{d:.0%}"] = {"item_ms": statistics.median(t) * 1e3, "items_run": len(t)}
    return out


def design_switch_us(npu: object, T: int = 1024, reps: int = 5) -> float | None:
    """A design switch's cost inside a program (us): two GEMM items of different designs alternated 32 times against
    each repeated 64 times."""
    pf = npu.prefill
    T = min(s for s in pf.sizes if s >= T)
    items = pf._program(T)
    blocks = pf.lib.load()._asm.blocks
    gemms = [it for it, (op, *_) in zip(items, pf.runlists[T]) if hasattr(op, "fmt") and hasattr(op, "K")]
    a = gemms[0] if gemms else None
    b = next((it for it in gemms if a is not None and blocks[it[0]].design != blocks[a[0]].design), None)
    if a is None or b is None:
        return None
    ta, tb, tab = (_run_ms(pf, prog, reps) for prog in ([a] * 64, [b] * 64, [a, b] * 32))
    return (tab - (ta + tb) / 2) / 63 * 1e3


def run(path: str, *, step: Callable[[str], None], end: Callable[[str], None], irq_cpu: int, rounds: int = 2,
        reps: int = 5) -> dict[str, object]:
    """Each variant ``rounds`` times in turn (V0 V1 V2 V3 V0 ...): its pp1024 samples (ms; a warmup first), its
    lookups, huge pages; per-kind times and the design switch for V0 and V1 (their first round), V0's largest kind at
    lower duties (``duty_sweep``). ``step`` / ``end``:
    the telemetry sampler's (``Sampler.step``, ``Sampler.end``)."""
    out: dict[str, object] = {"file": path, "resident_before": resident(path), "runs": []}
    rng = torch.Generator().manual_seed(0)
    step("compile")  # (each design set once before anything is timed: whole-line drains and not)
    for lines in (True, False):
        with variant("2 MB", lines):
            npu = open_model(path).load(capacity=4096)
            npu.prepare(1024)
        del npu
        gc.collect()
    end("compile")
    for r in range(rounds):
        for name, pages, lines, busy in VARIANTS:
            tag = f"{name}, round {r + 1}"
            step(f"{tag}: load")
            with variant(pages, lines):
                t0 = time.perf_counter()
                model = open_model(path)
                npu = model.load(capacity=4096)
                load_s = time.perf_counter() - t0
            tokens = token_source(file_info(path)[1], model.encode("")[:1], rng)
            end(f"{tag}: load")
            step(tag)
            lookups: list[float] = []
            with timed_lookups(lookups), busy_cpu(busy, irq_cpu):
                samples = measure(npu, Test(path, 1024, 0, 0), tokens, reps=reps, warmup=True, progress=None)
            entry: dict[str, object] = {"variant": name, "round": r + 1, "load_s": load_s,
                                        "pp1024_ms": [s / 1e6 for s in samples],
                                        "lookups_ms_per_prompt": sum(lookups) * 1e3 / (reps + 1),
                                        "anon_huge_mb": anon_huge_kb() / 1024, "holds": npu.describe()}
            end(tag)
            if r == 0 and lines and not busy:  # (V0 and V1: what the page size moves, per kind)
                entry["kinds"] = kinds(npu, step=step, end=end, tag=tag)
                step(f"{tag}: design switch")
                entry["design_switch_us"] = design_switch_us(npu)
                end(f"{tag}: design switch")
                if pages == "2 MB":  # (V0: the largest kind's items at lower duties)
                    gemm = max(entry["kinds"]["kinds"].items(), key=lambda kv: kv[1]["ms"])[0]
                    entry["duty"] = duty_sweep(npu, gemm, step=step, end=end, tag=tag)
            out["runs"].append(entry)
            del model, npu, tokens
            gc.collect()
    out["resident_after"] = resident(path)
    return out
