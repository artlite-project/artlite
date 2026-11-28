"""The pace levels a machine needs for the prefill's dense GEMMs (``gemm.pacing``): the platform of some
laptops asserts PROCHOT while those GEMMs run at full rate, and the whole SoC then runs at its lowest clocks (a Krackan
Point laptop's Q4_K GEMMs 2.3x slower, on any power source), while at a lower duty they run unthrottled.

``resolve`` finds a prefill's ``PACE`` word by timing, once per machine, power source, platform profile and library
(``_path``). Every pace class runs alone at level 0 and at the ``PROBE`` levels: a throttle is a class's own (that
laptop's Q4_K GEMMs trip it, the Q6_K ones beside them do not, and pacing hardly changes their time), so no class stands
for the others, not even one with the most work (Llama 3.1 8B Q4_K_M's Q6_K down and Qwen3.8 27B Q4_K_M's Q6_K qkv take
as long as their Q4_K gate / up). A class with a paced level clearly faster than level 0 (``GAIN``) is throttled: it
keeps its fastest level from level 1 up (``MARGIN``: a higher level only when clearly faster; the scan stops two levels
past the fastest), the others stay at 0. The GEMMs of all classes together then confirm them: those levels or every
class one level deeper (when clearly faster: classes unthrottled alone can still trip it together), against level 0.
Where nothing throttles, the probes are all (three measurements per class: about a second). Timing, not the SMU's
PROCHOT residency: the pace is for speed, timing works where ``gpu_metrics`` is missing (Windows), and that counter
advances every ~10 ms, coarse for runs of a few tens of ms.

The items are a chunk's GEMMs (``T``), a few per class (``ITEMS``: each distinct GEMM first; one trips it within about
a millisecond, and a measurement's runs follow each other), on random activations (``activations``), with their
outputs in a scratch buffer: calibrating never writes the prefill's or the overlay's buffers. The activations matter:
the array's current follows its operands' bits (Strix Halo, the 1B's gate/up at the same speed: 26-29 W socket power on
zeros, 35-36 on random or a prompt's activations), and the prefill's buffers hold zeros until a first chunk, on which a
throttling laptop's GEMMs run unthrottled. GEMMs that read a parameter of the chunk (``mc_param``: an MoE layer's
router and shared expert over the whole chunk) run with their item's (``item_params``).

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

from .. import _native, cache, settings
from ..gemm.pacing import PACE, PACE_CLASSES, PACE_IDLE, pace_levels, pace_word
from ..operators.fused.layout import parse_name
from ..operators.util import arg_bytes

VERSION = 3  # (of the calibration: a cached entry of another version calibrates again)
T_MAX = 1024  # tokens of the chunk whose GEMMs calibrate (the largest size up to it)
ITEMS = 4  # items of a class (more where it has more distinct GEMMs: one of each)
PROBE = (1, 5)  # levels every class runs at against level 0: a throttle shows as a faster paced run (1: the least
# idle; 5, duty 0.55: about the deepest that can win, a paced GEMM there takes 1.7-1.8x its unthrottled time, a
# throttled one 2.2-2.5x)
GAIN = 0.9  # a paced run this much of level 0's time or less: the platform throttles the class
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
            raise ValueError(f"{settings.PACE.name}: {part!r} is not <class>=<level> ({', '.join(PACE_CLASSES)})")
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
    """The machine, its power source and platform profile (the throttle can follow each: a laptop's limits follow its
    profile)."""
    parts = [platform.node(), platform.processor() or platform.machine()]
    if platform.system() == "Linux":
        parts += [_read(f"/sys/class/dmi/id/{k}") for k in ("sys_vendor", "product_name", "bios_version")]
        mains = [_read(os.path.join(p, "online")) for p in glob.glob("/sys/class/power_supply/*")
                 if _read(os.path.join(p, "type")) in ("Mains", "USB")]
        parts.append("ac" if "1" in mains else "battery" if mains else "")
        parts.append(_read("/sys/firmware/acpi/platform_profile"))
    return "|".join(parts)


def _path() -> str:
    """The calibrations' file (``common/pace.json`` of the cache)."""
    return cache.adopt(cache.common("pace.json"), "pace.json")


def _cache() -> dict[str, Any]:
    try:
        with open(_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _store(key: str, entry: dict[str, Any]) -> None:
    data = _cache()
    data[key] = entry
    path = _path()
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def search(classes: list[int], measure: Callable[[list[int] | None, int], float],
           note: Callable[[str], None] = lambda text: None) -> tuple[dict[int, int], dict]:
    """The levels per class and the times behind them; ``measure(classes, word)``: seconds per run of those classes'
    items at that PACE word (None: every class's items together); ``note``: told what the probes found (a throttle or
    none) and the levels."""
    top = len(PACE_IDLE) - 1
    times = {c: {L: measure([c], pace_word({c: L})) for L in (0, *PROBE)} for c in classes}
    gain = {c: min(times[c][L] for L in PROBE) / times[c][0] for c in classes}  # (the best paced run / level 0's)
    table: dict[str, Any] = {"classes": {PACE_CLASSES[c]: {"s": times[c], "level": 0} for c in classes}}
    throttled = [c for c in classes if gain[c] <= GAIN]
    if not throttled:
        c = min(classes, key=gain.__getitem__)
        note(f"the prefill's GEMMs run unthrottled on this machine: unpaced (no class faster paced; the closest, "
             f"{PACE_CLASSES[c]}: {gain[c]:.2f} of level 0's time)")
        return {}, table
    note("the platform throttles the prefill's GEMMs at full rate ("
         + ", ".join(f"{PACE_CLASSES[c]} {gain[c]:.2f}" for c in throttled)
         + " of level 0's time paced): calibrating their pace (once, a few seconds)")
    levels: dict[int, int] = {}
    for c in throttled:
        t, best = times[c], 0
        for L in range(1, top + 1):
            if L not in t:
                t[L] = measure([c], pace_word({c: L}))
            if t[L] < t[best] * (1 - MARGIN):
                best = L
            elif best and L - best >= 2 and t[L] > t[best]:  # (past the fastest: the time rises from there,
                break  # while above a trip point it rises too, then drops below it)
        table["classes"][PACE_CLASSES[c]] = {"s": dict(sorted(t.items())), "level": best}
        levels[c] = best  # (>= 1: a probe ran in GAIN of level 0's time)
    deeper = {c: min(levels.get(c, 0) + 1, top) for c in classes}
    s = {"level 0": measure(None, 0), "levels": measure(None, pace_word(levels)),
         "deeper": measure(None, pace_word(deeper))}
    table["all"] = {"s": s}
    t = s["levels"]
    if s["deeper"] < t * (1 - MARGIN):
        levels, t = deeper, s["deeper"]
    if t > s["level 0"] * (1 - MARGIN):
        note("paced, the GEMMs together are not faster: unpaced")
        return {}, table
    note(f"paced: {describe(pace_word(levels))}")
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


def pick(groups: list[list], n: int) -> list:
    """Up to n entries of the groups (at least one of each), round robin from each group's first, kept in group order:
    each group's picked entries one after the other."""
    take = [0] * len(groups)
    left = max(n, len(groups))
    for k in range(max(map(len, groups), default=0)):
        for g, entries in enumerate(groups):
            if k < len(entries) and left:
                take[g] += 1
                left -= 1
    return [e for g, entries in enumerate(groups) for e in entries[:take[g]]]


def _items(pf: Any) -> tuple[dict[int, list], dict[str, int], dict[str, Any], int]:
    """Per pace class the calibration items of the prefill ``pf`` (its GEMMs' of chunk size ``T``: ``ITEMS``, each
    distinct GEMM first, with A in a scratch buffer of random ``activations`` and C in another), the scratchpad
    parameters they read besides PACE (``item_params``), the buffers they address, T."""
    T = max([t for t in pf.sizes if t <= T_MAX] or [min(pf.sizes)])
    by: dict[int, dict[int, list]] = {}  # (class -> the GEMM's id -> its items: runlist order)
    for op, _, w, *_ in pf.runlists[T]:
        if getattr(op, "pace", None) is not None:
            item = (op, f"PACE_A[0:{arg_bytes(op, 0)}]", w, f"PACE_C[0:{arg_bytes(op, 2)}]")
            by.setdefault(op.pace, {}).setdefault(id(op), []).append(item)
    runlists = {c: pick(list(gemms.values()), ITEMS) for c, gemms in by.items()}
    ops = [op for rl in runlists.values() for op, *_ in rl]
    params: dict[str, int] = {}
    for op in ops:
        for name, value in op.item_params().items():
            assert params.setdefault(name, value) == value, (name, value, params[name])
    pa, pc = pf.ov.dev.device.alloc_host_buffers([max(arg_bytes(op, i) for op in ops) for i in (0, 2)])
    pa.write(activations(pa.size))
    pa.sync_to_device()
    # (the buffers the items address, not every buffer of the prefill: a command lists what its dispatch is given, and
    # on the XRT backend at most 5 XRT buffers, ``backend``)
    used = {parse_name(ref)[0] for rl in runlists.values() for _, *refs in rl for ref in refs}
    buffers = {name: b for name, b in {**pf._buffers(), "PACE_A": pa, "PACE_C": pc}.items() if name in used}
    return {c: pf.lib.items(rl, buffers) for c, rl in runlists.items()}, params, buffers, T


def calibrate(pf: Any) -> tuple[int, dict]:
    """The PACE word of the prefill ``pf`` (bound, its library loaded) by timing, with its table."""
    progs, params, buffers, T = _items(pf)
    bufs = list(buffers.values())

    def run(items: list) -> None:
        pf.lib.run(items, splits=list(range(PIECE, len(items), PIECE)), bufs=bufs)

    def measure(classes: list[int] | None, word: int) -> float:
        items = [it for c in (classes or sorted(progs)) for it in progs[c]]
        pf.lib.set_params({**params, PACE: word})
        time.sleep(IDLE_S)
        run(items)  # (the cores read the word: a run's first item of each design)
        t = []
        end = time.perf_counter() + RUN_S
        while len(t) < 3 or time.perf_counter() < end:
            t0, w0 = time.perf_counter(), _native.turn_waited()
            run(items)
            t.append(time.perf_counter() - t0 - (_native.turn_waited() - w0))  # (another process's turns: not its)
        return statistics.median(t)

    t_start = time.perf_counter()
    levels, table = search(sorted(progs), measure, note=lambda text: sys.stderr.write(f"artlite: {text}\n"))
    pf.lib.set_params({PACE: 0})
    pf.ov.dev.stale = True  # (the library ran: a paged overlay's state comes back before its next step)
    table.update(T=T, seconds=time.perf_counter() - t_start)
    return pace_word(levels), table


def resolve(pf: Any) -> int:
    """The PACE word of the prefill ``pf`` (``ARTLITE_PACE``; auto: cached, else calibrated and cached)."""
    spec = settings.PACE.get()
    fixed = parse(spec)
    if fixed is not None:
        return pace_word(fixed)
    key = f"{machine_key()}|{os.path.basename(os.path.dirname(pf.lib.elf_path))}"
    hit = _cache().get(key)
    if spec.strip().lower() != "recalibrate" and hit is not None and hit.get("version") == VERSION:
        return int(hit["word"])
    word, table = calibrate(pf)
    _store(key, {"word": word, "levels": describe(word), "version": VERSION,
                 "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(table["seconds"], 3),
                 "ms": _ms(table)})
    return word


def _ms(table: dict[str, Any]) -> dict[str, Any]:
    """The calibration's times in ms (a cache entry's record of what it measured): per class by level, then all
    classes together."""
    out: dict[str, Any] = {name: {str(L): round(s * 1e3, 3) for L, s in e["s"].items()}
                           for name, e in table["classes"].items()}
    if "all" in table:
        out["all"] = {k: round(s * 1e3, 3) for k, s in table["all"]["s"].items()}
    return out
