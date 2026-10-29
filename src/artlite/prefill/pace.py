"""The pace levels a machine needs for the prefill's dense GEMMs (``gemm.gemm_q``'s "Pacing"): the platform of some
laptops asserts PROCHOT while those GEMMs run at full rate, and the whole SoC then runs at its lowest clocks (a Krackan
Point laptop's Q4_K GEMMs 2.3x slower, on any power source), while at a lower duty they run unthrottled.

``resolve`` finds a prefill's ``PACE`` word by timing, once per machine, power source and library (``CACHE``): the
heaviest pace class (by its items' MACs) at levels 0 and ``QUICK``; if none of those runs clearly faster than
level 0 (``GAIN``), nothing throttles and every level stays 0 (two or three runs of the class's items: ~0.1-0.3 s).
Else each class from level 0 up keeps the fastest level (``MARGIN``: a higher level only when clearly faster; the scan
stops two levels past a level faster than 0), and the GEMMs of all classes together confirm them: those levels or every
class one level deeper (when clearly faster: classes unthrottled alone can still trip it together), against level 0.
The items are a chunk's GEMMs (``T``) on random activations (``activations``), with their outputs in a scratch buffer:
calibrating never writes the prefill's or the overlay's buffers. The activations matter: the array's current follows
its operands' bits (Strix Halo, the 1B's gate/up at the same speed: 26-29 W socket power on zeros, 35-36 on random or a
prompt's activations), and the prefill's buffers hold zeros until a first chunk, on which a throttling laptop's GEMMs
run unthrottled. GEMMs that read a parameter of the chunk (``mc_param``: an MoE layer's token chunks) run paced at
their class's level, not in the calibration.

``ARTLITE_PACE``: unset or ``auto`` (the cached or calibrated levels), ``off`` (level 0), ``recalibrate`` (calibrate
again), or levels: ``<class>=<level>`` by ``PACE_CLASSES`` name, comma-separated, or one level for every class (e.g.
``gu=3,qkv=2``, ``4``).
"""

from __future__ import annotations

import glob
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..gemm.gemm_q import PACE, PACE_CLASSES, PACE_IDLE, pace_levels, pace_word
from ..operators.base import CACHE_DIR
from ..operators.util import arg_bytes

CACHE = os.path.join(CACHE_DIR, "pace.json")
VERSION = 2  # (of the calibration: a cached entry of another version calibrates again)
T_MAX = 1024  # tokens of the chunk whose GEMMs calibrate (the largest size up to it)
ITEMS = 16  # items of a class at most
QUICK = (3, 5)  # levels against level 0 of the heaviest class: a throttle shows as a faster paced run
GAIN = 0.9  # a paced run this much of level 0's time or less: the platform throttles
MARGIN = 0.03  # a higher level is kept only when this much faster
IDLE_S = 0.02  # idle before each measurement (a throttle's tail lets go: PROCHOT ~3 ms after a GEMM item)
RUN_S = 0.03  # seconds of timed runs per measurement at least (3 runs at least)
PIECE = 4  # items per dispatch (well within the driver's 2 s at the slowest level)


def parse(spec: str) -> dict[int, int] | None:
    """The levels of an ``ARTLITE_PACE`` value (None: ``auto`` / ``recalibrate``; ``off``: none)."""
    spec = spec.strip().lower()
    if spec in ("", "auto", "recalibrate"):
        return None
    if spec == "off":
        return {}
    top = len(PACE_IDLE) - 1
    if spec.isdigit():
        return {c: min(int(spec), top) for c in range(len(PACE_CLASSES))}
    out = {}
    for part in spec.split(","):
        name, _, level = part.partition("=")
        if name.strip() not in PACE_CLASSES or not level.strip().isdigit():
            raise ValueError(f"ARTLITE_PACE: {part!r} is not <class>=<level> ({', '.join(PACE_CLASSES)})")
        out[PACE_CLASSES.index(name.strip())] = min(int(level), top)
    return out


def describe(word: int) -> str:
    """The nonzero levels of a PACE word by class name ("" for none)."""
    return ", ".join(f"{PACE_CLASSES[c]} {v}" for c, v in sorted(pace_levels(word).items()))


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def machine_key() -> str:
    """The machine and its power source (the throttle can follow either)."""
    parts = [platform.node(), platform.processor() or platform.machine()]
    if platform.system() == "Linux":
        parts += [_read(f"/sys/class/dmi/id/{k}") for k in ("sys_vendor", "product_name", "bios_version")]
        mains = [_read(os.path.join(p, "online")) for p in glob.glob("/sys/class/power_supply/*")
                 if _read(os.path.join(p, "type")) in ("Mains", "USB")]
        parts.append("ac" if "1" in mains else "battery" if mains else "")
    return "|".join(parts)


def _cache() -> dict[str, Any]:
    try:
        with open(CACHE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _store(key: str, entry: dict[str, Any]) -> None:
    data = _cache()
    data[key] = entry
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = f"{CACHE}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, CACHE)


def search(classes: list[int], measure: Callable[[list[int] | None, int], float],
           note: Callable[[str], None] = lambda text: None) -> tuple[dict[int, int], dict]:
    """The levels per class (``classes``: the heaviest first) and the times behind them; ``measure(classes, word)``:
    seconds per run of those classes' items at that PACE word (None: every class's items together); ``note``: told
    when the scan starts (a throttle found) and its levels."""
    table: dict[str, Any] = {}
    heavy = classes[0]
    t0 = measure([heavy], 0)
    quick = {L: measure([heavy], pace_word({heavy: L})) for L in QUICK}
    table["quick"] = {"class": PACE_CLASSES[heavy], "s": {0: t0, **quick}}
    if min(quick.values()) > GAIN * t0:
        return {}, table
    note(f"the platform throttles the prefill's GEMMs at full rate ({PACE_CLASSES[heavy]}: "
         f"{min(quick.values()) / t0:.2f} of level 0's time paced): calibrating their pace (once, a few seconds)")
    levels: dict[int, int] = {}
    table["classes"] = {}
    for c in classes:
        times = {0: t0 if c == heavy else measure([c], 0)}
        best = 0
        for L in range(1, len(PACE_IDLE)):
            times[L] = quick[L] if c == heavy and L in quick else measure([c], pace_word({c: L}))
            if times[L] < times[best] * (1 - MARGIN):
                best = L
            elif best and L - best >= 2 and times[L] > times[best]:  # (past the fastest: the time rises from there,
                break  # while above a trip point it rises too, then drops below it)
        table["classes"][PACE_CLASSES[c]] = {"s": times, "level": best}
        if best:
            levels[c] = best
    if levels:
        deeper = {c: min(levels.get(c, 0) + 1, len(PACE_IDLE) - 1) for c in classes}
        s = {"level 0": measure(None, 0), "levels": measure(None, pace_word(levels)),
             "deeper": measure(None, pace_word(deeper))}
        table["all"] = {"s": s}
        t = s["levels"]
        if s["deeper"] < t * (1 - MARGIN):
            levels, t = deeper, s["deeper"]
        if t > s["level 0"] * (1 - MARGIN):
            note("paced, the GEMMs together are not faster: unpaced")
            return {}, table
    note(f"paced: {describe(pace_word(levels)) or 'none'}")
    return levels, table


def activations(n: int, seed: int = 0) -> np.ndarray:
    """n bytes of activation panels (bfp16ebs8 blocks [E, m0..m7], ``operators.bfp16``) of N(0, 1) values: a pattern
    of 64 K blocks, repeated."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((min(1 << 16, -(-n // 9)), 8), np.float32)
    e = np.floor(np.log2(np.abs(x).max(1))).astype(np.int32)
    blocks = np.empty((len(x), 9), np.uint8)
    blocks[:, 0] = e + 127
    m = np.clip(np.rint(x * np.ldexp(np.float32(1), 6 - e)[:, None]), -128, 127)
    blocks[:, 1:] = m.astype(np.int8).view(np.uint8)
    return np.resize(blocks.reshape(-1), n)


def _items(pf: Any) -> tuple[dict[int, list], dict[int, float], dict[str, Any], int]:
    """Per pace class the calibration items of the prefill ``pf`` (its GEMMs' of chunk size ``T``, A and C in the
    scratch buffers, A random ``activations``), their MACs, the buffers they address, T."""
    T = max([t for t in pf.sizes if t <= T_MAX] or [min(pf.sizes)])
    runlist = [(op, f"PACE_A[0:{arg_bytes(op, 0)}]", w, f"PACE_C[0:{arg_bytes(op, 2)}]")
               for op, _, w, *_ in pf.runlists[T]
               if getattr(op, "pace", None) is not None and not op.mc_param]  # (mc_param: a chunk's own parameter)
    by: dict[int, list] = {}
    for item in runlist:
        if len(by.setdefault(item[0].pace, [])) < ITEMS:
            by[item[0].pace].append(item)
    macs = {c: float(sum(op.M * op.K * op.N for op, *_ in rl)) for c, rl in by.items()}
    dev = pf.ov.dev.device
    pa = dev.alloc_host_buffer(max(arg_bytes(op, 0) for op, *_ in runlist))
    pa.write(activations(pa.size))
    pa.sync_to_device()
    pc = dev.alloc_host_buffer(max(arg_bytes(op, 2) for op, *_ in runlist))
    buffers = {**pf._buffers(), "PACE_A": pa, "PACE_C": pc}
    return {c: pf.lib.items(rl, buffers) for c, rl in by.items()}, macs, buffers, T


def calibrate(pf: Any) -> tuple[int, dict]:
    """The PACE word of the prefill ``pf`` (bound, its library loaded) by timing, with its table."""
    progs, macs, buffers, T = _items(pf)
    bufs = list(buffers.values())

    def run(items: list) -> None:
        pf.lib.run(items, splits=list(range(PIECE, len(items), PIECE)), bufs=bufs)

    def measure(classes: list[int] | None, word: int) -> float:
        items = [it for c in (classes or sorted(progs)) for it in progs[c]]
        pf.lib.set_params({PACE: word})
        time.sleep(IDLE_S)
        run(items)  # (the cores read the word: a run's first item of each design)
        t = []
        end = time.perf_counter() + RUN_S
        while len(t) < 3 or time.perf_counter() < end:
            t0 = time.perf_counter()
            run(items)
            t.append(time.perf_counter() - t0)
        return statistics.median(t)

    t_start = time.perf_counter()
    levels, table = search(sorted(progs, key=lambda c: -macs[c]), measure,
                           note=lambda text: sys.stderr.write(f"artlite: {text}\n"))
    pf.lib.set_params({PACE: 0})
    pf.ov.dev.stale = True  # (the library ran: a paged overlay's state comes back before its next step)
    table.update(T=T, seconds=time.perf_counter() - t_start)
    return pace_word(levels), table


def resolve(pf: Any) -> int:
    """The PACE word of the prefill ``pf`` (``ARTLITE_PACE``; auto: cached, else calibrated and cached)."""
    spec = os.environ.get("ARTLITE_PACE", "auto")
    fixed = parse(spec)
    if fixed is not None:
        return pace_word(fixed)
    key = f"{machine_key()}|{os.path.basename(os.path.dirname(pf.lib.elf_path))}"
    hit = _cache().get(key)
    if spec.strip().lower() != "recalibrate" and hit is not None and hit.get("version") == VERSION:
        return int(hit["word"])
    word, table = calibrate(pf)
    _store(key, {"word": word, "levels": describe(word), "version": VERSION,
                 "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(table["seconds"], 3)})
    return word
