"""NPU-side profiling for Artlite operators.

Two complementary mechanisms, selected by the ``ARTLITE_PROFILE`` env var:

``counters`` (default when enabled, ``ARTLITE_PROFILE=1|counters``)
    Hardware performance counters in each AIE2P core tile (AM029 CORE_MODULE).
    The operator's MLIR is rewritten at compile time (post-placement
    ``aiex.npu.write32`` ops configure counters at the start of the runtime
    sequence), and after every dispatch the counter values are read back over
    the context-scoped ``AMDXDNA_AIE_TILE_READ`` ioctl (no root required) and
    appended as Chromium JSON counter events. Typical use: count cycles
    between ``event0()`` / ``event1()`` markers in the kernel. A "host"
    process adds per dispatch its wall time, the driver's device busy time
    (fdinfo) and the H clock.

``trace``  (``ARTLITE_PROFILE=trace``)
    Hardware event streaming over the AIE stream switches to DDR
    (mlir-aie trace flow, ``aie.iron.Program.enable_trace``). Produces
    per-event waveform data, converted at process exit to Chromium JSON via
    ``aie.utils.trace.parse`` (mlir-aie). Meant for detailed pipeline analysis
    on designs whose stream channels leave room for the trace overlay
    (see the profiling skill for limitations). The shim writes the 32-byte
    trace packets to DDR in 64-byte bursts: when the stream ends on an odd
    packet, the last one stays in the DMA, and the final events of the last
    dispatch are missing from the dump.

Trace-only env vars:
    ARTLITE_PROFILE_TRACE_SIZE   trace buffer size in bytes (default 4 MiB)
    ARTLITE_PROFILE_EVENTS       comma-separated core tile event names or hex
                                 (default: mlir-aie defaults: INSTR_EVENT_0/1,
                                 INSTR_VECTOR, stalls, PORT_RUNNING_0/1)
    ARTLITE_PROFILE_TILES        "all" (default), "first:N", or a
                                 "col,row;col,row" pinned-tile list
    ARTLITE_PROFILE_LATERAL=1    pre-lower with lateral-routing (spare shim)
    ARTLITE_PROFILE_DISTRIBUTE=1 pre-lower with distribute-channels (2 S2MM)
    ARTLITE_PROFILE_CONVERT=1    convert .ttbin dumps to .json at exit (default)

Counters-mode env vars:
    ARTLITE_PROFILE_COUNTERS     comma-separated "start:stop" event-id pairs,
                                 max 4 (default "33:34,34:33,33:33,34:34"
                                 = event0->event1 busy cycles, event1->event0
                                 idle cycles, event0 count, event1 count).
                                 Each configured pair costs one firmware
                                 round trip (~20 us) per tile per sample;
                                 fewer pairs = lower overhead.
    ARTLITE_PROFILE_SAMPLE_EVERY sample counters every Nth dispatch
                                 (default 1). With N>1, deltas cover the
                                 previous N dispatches (per-dispatch cost of
                                 the readback is divided by N).

Common:
    ARTLITE_PROFILE_DIR          output directory (default ./artlite_profile)
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import _native

_CONFIG: ProfileConfig | None = None
_CONFIG_PARSED = False
_HOOK_INSTALLED = False
_PROFILERS: list[_native.Profiler] = []
_TRACE_DUMPS: list[tuple[str, str]] = []  # (ttbin path, physical mlir path)

# AIE2P performance counter registers (AM029).
# Core module (per core tile): 4 counters. Memory module (per core tile's 64KB
# DM + locks/DMAs): 2 counters. Memory-tile module (row 1): 6 counters.
# PL module (shim tile, row 0): 6 counters; start/stop ctrl regs are at
# 0x31000/0x3100C/0x31014 (odd ctrl indices are reset-event regs).
_CORE_CTRL0 = 0x31500  # core counters 0/1 start/stop events
_CORE_CTRL1 = 0x31504  # core counters 2/3
_CORE_CNT_BASE = 0x31520  # 4 x u32, +4 each
_MEM_CTRL0 = 0x11000  # memory-module counters 0/1 (2 max)
_MEM_CNT_BASE = 0x11020
# ctrl layout: counters 1-0 / 3-2 start-stop (AM029). Counters 4/5 are not
# used: the 3rd start/stop register (0x9100C / 0x31014) silently ignores
# writes32 from the runtime sequence on this KMD, and reading counter
# registers >= 4 wedges the context.
_MT_CTRL_BASES = (0x91000, 0x91004)
_MT_CNT_BASE = 0x91020
_SH_CTRL_BASES = (0x31000, 0x3100C)
_SH_CNT_BASE = 0x31020

_MAX_COUNTERS = {"core": 4, "mem": 2, "mt": 4, "sh": 4}
# module name -> (trace-events enum attr, counter value base)
_MOD_ENUM = {
    "core": "CoreEvent",
    "mem": "MemEvent",
    "mt": "MemTileEvent",
    "sh": "ShimTileEvent",
}
_MOD_CNT_BASE = {
    "core": _CORE_CNT_BASE,
    "mem": _MEM_CNT_BASE,
    "mt": _MT_CNT_BASE,
    "sh": _SH_CNT_BASE,
}

# AIE2P core event IDs (aie-rt xaie_events_aie2p.h); used for labels only —
# full name resolution goes through aie.utils.trace.events.
_EVENT_NAMES = {33: "INSTR_EVENT_0", 34: "INSTR_EVENT_1", 37: "INSTR_VECTOR"}

_DEFAULT_COUNTER_PAIRS = [(33, 34), (34, 33), (33, 33), (34, 34)]


def _event_name(module: str, code: int) -> str:
    try:
        from aie.utils.trace.events import get_events_for_device

        enums = get_events_for_device("npu2p")
        return getattr(enums, _MOD_ENUM[module])(code).name
    except Exception:  # noqa: BLE001 - best-effort pretty name, fall back to the table
        return _EVENT_NAMES.get(code, f"event{code}")


@dataclass
class ProfileConfig:
    mode: str  # "counters" | "trace"
    out_dir: str
    # (module, start_event, stop_event); module "core" (max 4) or "mem" (max 2)
    counter_specs: list[tuple[str, int, int]] = field(
        default_factory=lambda: [("core", s, e) for s, e in _DEFAULT_COUNTER_PAIRS]
    )
    trace_size: int = 4 * 1024 * 1024
    trace_events: list | None = None
    trace_tiles: str = "all"
    trace_lateral: bool = False
    trace_distribute: bool = False
    convert: bool = True
    sample_every: int = 1

    def counter_readout(
        self, modules: tuple[str, ...] = ("core", "mem")
    ) -> list[tuple[int, str]]:
        """(register address, label) for each configured counter in `modules`."""
        out: list[tuple[int, str]] = []
        for module in modules:
            specs = [p for p in self.counter_specs if p[0] == module]
            for i, (_, s, e) in enumerate(specs):
                lbl = (
                    f"{_event_name(module, s)}->{_event_name(module, e)}"
                    + (" cycles" if s != e else " count")
                )
                out.append((_MOD_CNT_BASE[module] + 4 * i, lbl))
        return out


def _resolve_event(module: str, tok: str) -> int:
    """Event id from a number or an enum name (CoreEvent/MemEvent, npu2p)."""
    tok = tok.strip()
    try:
        return int(tok, 0)
    except ValueError:
        pass
    from aie.utils.trace.events import get_events_for_device

    enums = get_events_for_device("npu2p")
    cls = getattr(enums, _MOD_ENUM[module])
    return int(cls[tok]._value_ if hasattr(cls[tok], "_value_") else cls[tok])


_MOD_PREFIXES = {
    "m": "mem",
    "mem": "mem",
    "memmod": "mem",
    "mt": "mt",
    "memtile": "mt",
    "sh": "sh",
    "shim": "sh",
    "pl": "sh",
}


def _parse_counter_specs(text: str) -> list[tuple[str, int, int]]:
    """Parse "33:34,m:77:77,mt:21:29,sh:14:22" into (module, start, stop)
    tuples; module prefixes select the tile module: none = core module,
    m: = core-tile memory module, mt: = memory tile (row 1), sh: = shim/PL
    tile (row 0). Event names are accepted."""
    specs: list[tuple[str, int, int]] = []
    for item in text.split(","):
        parts = [p.strip() for p in item.strip().split(":")]
        module = "core"
        if parts[0] in _MOD_PREFIXES:
            module = _MOD_PREFIXES[parts[0]]
            parts = parts[1:]
        if len(parts) != 2:
            raise ValueError(f"bad counter pair '{item}': expected start:stop")
        specs.append((module, _resolve_event(module, parts[0]),
                      _resolve_event(module, parts[1])))
    counts: dict[str, int] = {}
    for p in specs:
        counts[p[0]] = counts.get(p[0], 0) + 1
    for mod, cnt in counts.items():
        if not cnt <= _MAX_COUNTERS[mod]:
            raise ValueError(
                f"ARTLITE_PROFILE_COUNTERS: at most {_MAX_COUNTERS[mod]} "
                f"{mod}-module pairs (got {cnt})"
            )
    if not specs:
        raise ValueError("ARTLITE_PROFILE_COUNTERS: empty spec")
    return specs


def _parse_events(text: str) -> list[object]:
    from aie.utils.trace.events import CoreEvent

    events = []
    for tok in text.split(","):
        tok = tok.strip()
        events.append(CoreEvent[tok] if not tok.startswith("0x") else int(tok, 16))
    return events


def get_config() -> ProfileConfig | None:
    """Return the profiling config, or None when profiling is disabled."""
    global _CONFIG, _CONFIG_PARSED
    if _CONFIG_PARSED:
        return _CONFIG
    _CONFIG_PARSED = True

    raw = os.environ.get("ARTLITE_PROFILE", "0").lower()
    if raw in ("0", "off", "", "false", "no"):
        return None
    mode = raw if raw in ("counters", "trace") else "counters"

    counter_specs = [("core", s, e) for s, e in _DEFAULT_COUNTER_PAIRS]
    if spec := os.environ.get("ARTLITE_PROFILE_COUNTERS"):
        counter_specs = _parse_counter_specs(spec)

    trace_events = None
    if spec := os.environ.get("ARTLITE_PROFILE_EVENTS"):
        trace_events = _parse_events(spec)

    _CONFIG = ProfileConfig(
        mode=mode,
        out_dir=os.environ.get("ARTLITE_PROFILE_DIR", "artlite_profile"),
        counter_specs=counter_specs,
        trace_size=int(os.environ.get("ARTLITE_PROFILE_TRACE_SIZE", str(4 * 1024 * 1024))),
        trace_events=trace_events,
        trace_tiles=os.environ.get("ARTLITE_PROFILE_TILES", "all"),
        trace_lateral=os.environ.get("ARTLITE_PROFILE_LATERAL", "0") == "1",
        trace_distribute=os.environ.get("ARTLITE_PROFILE_DISTRIBUTE", "0") == "1",
        convert=os.environ.get("ARTLITE_PROFILE_CONVERT", "1") == "1",
        sample_every=max(1, int(os.environ.get("ARTLITE_PROFILE_SAMPLE_EVERY", "1"))),
    )
    os.makedirs(_CONFIG.out_dir, exist_ok=True)
    atexit.register(_finalize)
    return _CONFIG


def _select_workers(cfg: ProfileConfig, workers: list) -> list | None:
    """Pick the workers to trace from a Program's worker list."""
    sel = cfg.trace_tiles
    if sel == "all":
        return workers
    if sel.startswith("first:"):
        return workers[: int(sel.split(":", 1)[1])]
    tiles: set[tuple[int, int]] = set()
    for tok in sel.split(";"):
        c, r = tok.split(",")
        tiles.add((int(c), int(r)))
    picked = [w for w in workers if (w.tile.col, w.tile.row) in tiles]
    if not picked:
        raise ValueError(
            f"ARTLITE_PROFILE_TILES={sel}: no pinned worker tiles match; "
            "use 'all', 'first:N', or ';'-separated 'col,row' of pinned tiles"
        )
    return picked


def install_iron_hook() -> None:
    """Patch aie.iron.Program.resolve_program to enable stream tracing.

    Idempotent. Only effective in `trace` mode; in `counters` mode the hook
    resolves programs unchanged.
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    _HOOK_INSTALLED = True

    import aie.iron

    orig_resolve = aie.iron.Program.resolve_program

    def resolving(self, *args, **kwargs):
        cfg = get_config()
        if cfg is not None and cfg.mode == "trace":
            workers = _select_workers(cfg, [w for c in self._configurations for w in c.workers])
            self.enable_trace(
                cfg.trace_size,
                workers=workers,
                coretile_events=cfg.trace_events,
            )
        return orig_resolve(self, *args, **kwargs)

    aie.iron.Program.resolve_program = resolving


def _find_aie_opt() -> str:
    from importlib.util import find_spec

    pkg = find_spec("mlir_aie").submodule_search_locations[0]
    return os.path.join(pkg, "bin", "aie-opt")


def _ctrl_writes(pairs: list[tuple[int, int]], bases: tuple[int, ...],
                 cap: int) -> list[tuple[int, int]]:
    """Pack (start, stop) event pairs into per-two-counters control regs."""
    out: list[tuple[int, int]] = []
    p = pairs + [(0, 0)] * (cap - len(pairs))
    for grp in range((len(pairs) + 1) // 2):
        a, b = p[2 * grp], p[2 * grp + 1]
        out.append((bases[grp], a[0] | (a[1] << 8) | (b[0] << 16) | (b[1] << 24)))
    return out


def instrumented_tiles_of(
    mlir_text: str, counter_specs: list[tuple[str, int, int]]
) -> dict[str, list[tuple[int, int]]]:
    """tiles_by_class: "core" (core tiles; core+mem module counters), "mt"
    (row-1 mem tiles), "sh" (row-0 shim tiles). Only classes with configured
    counter specs are populated."""
    tiles: dict[str, tuple[int, int]] = {}
    for m in re.finditer(r"(%\S+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)", mlir_text):
        tiles[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    core_tiles = sorted(
        {
            tiles[m.group(1)]
            for m in re.finditer(r"aie\.core\((%\S+?)(?:\s,|\))", mlir_text)
            if m.group(1) in tiles
        }
    )
    all_xy = set(tiles.values())
    mods = {p[0] for p in counter_specs}
    out: dict[str, list[tuple[int, int]]] = {}
    if mods & {"core", "mem"}:
        out["core"] = core_tiles
    if "mt" in mods:
        out["mt"] = sorted((c, r) for c, r in all_xy if r == 1)
    if "sh" in mods:
        out["sh"] = sorted((c, r) for c, r in all_xy if r == 0)
    return out


_CTRL_PLANS: dict[str, tuple[tuple[int, ...], int]] = {
    "core": ((_CORE_CTRL0, _CORE_CTRL1), 4),
    "mem": ((_MEM_CTRL0,), 2),
    "mt": (_MT_CTRL_BASES, 6),
    "sh": (_SH_CTRL_BASES, 6),
}
# module -> tile classes that carry it
_MOD_TILE_CLASSES = {"core": ("core",), "mem": ("core",), "mt": ("mt",), "sh": ("sh",)}


def inject_perf_counters(
    mlir_text: str, counter_specs: list[tuple[str, int, int]], work_dir: str
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    """Place tiles and inject perf-counter configuration into the MLIR.

    Returns (new mlir text, tiles_by_class) where tiles_by_class maps
    "core"/"mt"/"sh" to the instrumented (col, row) lists.
    The re-placed module goes through the normal aiecc pipeline afterwards;
    re-running aie-place-tiles on already-placed tiles is a no-op (this is
    also how mlir-aie pre-lowers trace options that aiecc cannot pass).
    """
    src_path = os.path.join(work_dir, "design_preprofile.mlir")
    placed_path = os.path.join(work_dir, "design_placed.mlir")
    with open(src_path, "w") as f:
        f.write(mlir_text)
    aie_opt = _find_aie_opt()
    r = subprocess.run(
        [
            aie_opt,
            "--pass-pipeline=builtin.module(aie.device(aie-place-tiles))",
            src_path,
            "-o",
            placed_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"aie-place-tiles failed:\n{r.stderr}")
    with open(placed_path) as f:
        placed = f.read()

    tiles_by_class = instrumented_tiles_of(placed, counter_specs)
    if not tiles_by_class:
        raise RuntimeError("profiling: design has no tiles to instrument")

    # module -> packed control register writes
    writes_per_module = {
        mod: _ctrl_writes(
            [(s, e) for m2, s, e in counter_specs if m2 == mod],
            _CTRL_PLANS[mod][0],
            _CTRL_PLANS[mod][1],
        )
        for mod in _CTRL_PLANS
        if any(p[0] == mod for p in counter_specs)
    }

    lines: list[str] = []
    for cls, tile_list in tiles_by_class.items():
        writes: list[tuple[int, int]] = []
        for mod, w in writes_per_module.items():
            if cls in _MOD_TILE_CLASSES[mod]:
                writes.extend(w)
        for col, row in tile_list:
            for addr, val in writes:
                lines.append(
                    f"    %pc{addr}_{col}_{row} = arith.constant {addr} : i32\n"
                    f"    %pv{addr}_{col}_{row} = arith.constant {val} : i32\n"
                    f"    aiex.npu.write32(%pc{addr}_{col}_{row}, %pv{addr}_{col}_{row}) "
                    f"{{column = {col} : i32, row = {row} : i32}} : i32, i32"
                )
    m = re.search(
        r"aie\.runtime_sequence\([^)]*\)\s*(?:attributes\s+[^\{]*)?\{\n", placed
    )
    if not m:
        raise RuntimeError("profiling: runtime sequence not found in placed MLIR")
    injected = placed[: m.end()] + "\n".join(lines) + "\n" + placed[m.end() :]
    return injected, tiles_by_class


def pre_lower_trace(mlir_path: str, cfg: ProfileConfig, work_dir: str) -> str:
    """Pre-lower trace ops with aie-insert-trace-flows options aiecc cannot
    pass (lateral-routing, distribute-channels). Returns path to the pre-lowered
    MLIR; if no options are requested the original path is returned unchanged.
    """
    if not (cfg.trace_lateral or cfg.trace_distribute):
        return mlir_path
    out_path = os.path.join(work_dir, "design_prelowered.mlir")
    opts = ""
    if cfg.trace_distribute:
        opts += "distribute-channels=true "
    if cfg.trace_lateral:
        opts += "lateral-routing=true"
    r = subprocess.run(
        [
            _find_aie_opt(),
            (
                "--pass-pipeline=builtin.module(aie.device("
                "aie-place-tiles,"
                f"aie-insert-trace-flows{{{opts}}},"
                "aie-trace-to-config,aie-trace-pack-reg-writes,"
                "aie-inline-trace-config),aie-fuse-trace-buffers)"
            ),
            mlir_path,
            "-o",
            out_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"trace pre-lowering failed:\n{r.stderr[-2000:]}")
    return out_path


def register_profiler(profiler: _native.Profiler) -> None:
    _PROFILERS.append(profiler)


def register_trace_dump(ttbin_path: str, work_dir: str) -> None:
    phys = os.path.join(work_dir, "design.mlir.prj", "input_with_addresses.mlir")
    _TRACE_DUMPS.append((ttbin_path, phys))


def convert_trace_dump(ttbin_path: str, phys_mlir_path: str) -> str | None:
    """Convert a raw trace dump to Chromium JSON via mlir-aie's parse_trace.

    Additionally appends synthesized "X" (complete) kernel-window slices that
    span INSTR_EVENT_0 -> INSTR_EVENT_1 per tile, named with the kernel
    functions of that tile (see artlite.profile_analysis).
    """
    import numpy as np
    from aie.utils.trace.parse import parse_trace

    from .profile_analysis import kernel_names_by_tile, synthesize_kernel_slices

    buf = np.fromfile(ttbin_path, dtype=np.uint32)
    with open(phys_mlir_path) as f:
        mlir_str = f.read()
    try:
        events = parse_trace(buf, mlir_str)
    except ValueError as e:
        print(f"profiling: skipping conversion of {ttbin_path}: {e}", file=sys.stderr)
        return None
    events = synthesize_kernel_slices(events, kernel_names_by_tile(mlir_str))
    out = os.path.splitext(ttbin_path)[0] + ".json"
    with open(out, "w") as f:
        json.dump(events, f)
    return out


def _cli() -> None:
    """python -m artlite.profiling <dump.ttbin> <physical_mlir> [dump2 ttbin...]

    Convert raw trace dumps to Chromium JSON (same conversion as the exit-time
    hook, for use with ARTLITE_PROFILE_CONVERT=0).
    """
    if len(sys.argv) < 3:
        print(_cli.__doc__)
        sys.exit(1)
    mlir = sys.argv[2]
    for dump in sys.argv[1:2] + sys.argv[3:]:
        if dump == mlir:
            continue
        out = convert_trace_dump(dump, mlir)
        if out:
            print(f"{dump} -> {out}")


def _finalize() -> None:
    for p in _PROFILERS:
        p.close()
    cfg = _CONFIG
    if cfg is None or not cfg.convert:
        return
    for ttbin, phys in _TRACE_DUMPS:
        if os.path.exists(phys):
            convert_trace_dump(ttbin, phys)
        else:
            print(
                f"profiling: physical MLIR not found for {ttbin} (compile with "
                "--dump-intermediates is required); raw dump kept",
                file=sys.stderr,
            )


if __name__ == "__main__":
    _cli()
