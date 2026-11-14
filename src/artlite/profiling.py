"""NPU-side profiling for Artlite operators.

Two complementary mechanisms, selected by the ``ARTLITE_PROFILE`` env var:

``counters`` (default when enabled, ``ARTLITE_PROFILE=1|counters``)
    Hardware performance counters in each AIE2P core tile (AM029 CORE_MODULE).
    The operator's MLIR is rewritten at compile time (post-placement
    ``aiex.npu.write32`` ops configure counters at the start of the runtime
    sequence), and after every dispatch the counter values are read back over
    the context-scoped ``AMDXDNA_AIE_TILE_READ`` ioctl (no root required;
    xdna-driver's module only) and
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from . import settings
from .npu_regs import MEMTILE_ROW, MODULES, SHIM_ROW

if TYPE_CHECKING:
    from . import _native

# AIE2P core event IDs (aie-rt xaie_events_aie2p.h); used for labels only —
# full name resolution goes through aie.utils.trace.events.
_EVENT_NAMES = {33: "INSTR_EVENT_0", 34: "INSTR_EVENT_1", 37: "INSTR_VECTOR"}

_DEFAULT_COUNTER_PAIRS = [(33, 34), (34, 33), (33, 33), (34, 34)]


def _event_name(module: str, code: int) -> str:
    try:
        from aie.utils.trace.events import get_events_for_device

        enums = get_events_for_device("npu2p")
        return getattr(enums, MODULES[module].events)(code).name
    except Exception:  # noqa: BLE001 - best-effort pretty name, fall back to the table
        return _EVENT_NAMES.get(code, f"event{code}")


class CounterSpec(NamedTuple):
    """One performance counter: the tile module it is in and the events that start and stop it."""

    module: str  # "core" | "mem" | "mt" | "sh" (``npu_regs.MODULES``)
    start: int
    stop: int


def validate_counter_specs(specs: Sequence[CounterSpec]) -> None:
    """At most each module's counters (a read past them wedges the hardware context), at least one."""
    if not specs:
        raise ValueError(f"{settings.PROFILE_COUNTERS.name}: empty spec")
    counts: dict[str, int] = {}
    for spec in specs:
        if spec.module not in MODULES:
            raise ValueError(f"{settings.PROFILE_COUNTERS.name}: unknown counter module {spec.module!r}")
        counts[spec.module] = counts.get(spec.module, 0) + 1
    for name, count in counts.items():
        if count > MODULES[name].max_counters:
            raise ValueError(
                f"{settings.PROFILE_COUNTERS.name}: at most {MODULES[name].max_counters} {name}-module pairs (got {count})"
            )


@dataclass(frozen=True)
class TileSelection:
    """The workers a trace covers: all of a program's, the first N, or those pinned to the listed tiles."""

    text: str = "all"  # as given: "all", "first:N" or "col,row;col,row"
    first: int | None = None
    tiles: frozenset[tuple[int, int]] | None = None

    @classmethod
    def parse(cls, text: str) -> TileSelection:
        name = settings.PROFILE_TILES.name
        if text == "all":
            return cls(text)
        try:
            if text.startswith("first:"):
                return cls(text, first=int(text.split(":", 1)[1]))
            tiles = set()
            for token in text.split(";"):
                col, row = token.split(",")
                tiles.add((int(col), int(row)))
            return cls(text, tiles=frozenset(tiles))
        except ValueError:
            raise ValueError(
                f"{name}={text!r}: use 'all', 'first:N', or ';'-separated 'col,row' of pinned tiles"
            ) from None

    def pick(self, workers: list[Any]) -> list[Any]:
        """The selected workers of a Program's worker list."""
        if self.first is not None:
            return workers[: self.first]
        if self.tiles is None:
            return workers
        picked = [w for w in workers if (w.tile.col, w.tile.row) in self.tiles]
        if not picked:
            raise ValueError(
                f"{settings.PROFILE_TILES.name}={self.text}: no pinned worker tiles match; "
                "use 'all', 'first:N', or ';'-separated 'col,row' of pinned tiles"
            )
        return picked


@dataclass(frozen=True)
class ProfileConfig:
    """The profiling configuration of a process: built once from the environment (``from_env``)."""

    mode: str  # "counters" | "trace"
    out_dir: str
    counter_specs: list[CounterSpec] = field(
        default_factory=lambda: [CounterSpec("core", s, e) for s, e in _DEFAULT_COUNTER_PAIRS]
    )
    trace_size: int = 4 * 1024 * 1024
    trace_events: list | None = None
    trace_tiles: TileSelection = TileSelection()
    trace_lateral: bool = False
    trace_distribute: bool = False
    convert: bool = True
    sample_every: int = 1

    # The fields that never change a design or its build (where the output goes, what is done with it): every other
    # field is in ``cache_key``.
    RUNTIME_FIELDS = frozenset({"out_dir", "convert", "sample_every"})

    def __post_init__(self) -> None:
        validate_counter_specs(self.counter_specs)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> ProfileConfig | None:
        """The configuration the environment selects (module docstring), None when profiling is off. A malformed
        value raises a ValueError naming its variable."""
        raw = settings.PROFILE.get(environ)
        if raw in ("0", "off", "", "false", "no"):
            return None
        counter_specs = [CounterSpec("core", s, e) for s, e in _DEFAULT_COUNTER_PAIRS]
        if spec := settings.PROFILE_COUNTERS.get(environ):
            counter_specs = _parse_counter_specs(spec)
        trace_events = _parse_events(spec) if (spec := settings.PROFILE_EVENTS.get(environ)) else None
        return cls(
            mode=raw if raw in ("counters", "trace") else "counters",
            out_dir=settings.PROFILE_DIR.get(environ),
            counter_specs=counter_specs,
            trace_size=settings.PROFILE_TRACE_SIZE.get(environ),
            trace_events=trace_events,
            trace_tiles=TileSelection.parse(settings.PROFILE_TILES.get(environ)),
            trace_lateral=settings.PROFILE_LATERAL.get(environ),
            trace_distribute=settings.PROFILE_DISTRIBUTE.get(environ),
            convert=settings.PROFILE_CONVERT.get(environ),
            sample_every=max(1, settings.PROFILE_SAMPLE_EVERY.get(environ)),
        )

    def cache_key(self) -> tuple:
        """What the configuration changes in a design (part of the build's cache key): trace parameters shape the
        MLIR, counters are injected by the build."""
        if self.mode == "trace":
            return (
                "trace", self.trace_size, str(self.trace_events), self.trace_tiles.text, self.trace_lateral,
                self.trace_distribute,
            )
        return "counters", [tuple(spec) for spec in self.counter_specs]

    def counter_readout(self, modules: tuple[str, ...] = ("core", "mem")) -> list[tuple[int, str]]:
        """(register address, label) for each configured counter in `modules`."""
        out: list[tuple[int, str]] = []
        for module in modules:
            specs = [p for p in self.counter_specs if p.module == module]
            for i, (_, s, e) in enumerate(specs):
                lbl = (
                    f"{_event_name(module, s)}->{_event_name(module, e)}"
                    + (" cycles" if s != e else " count")
                )
                out.append((MODULES[module].address(i), lbl))
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
    cls = getattr(enums, MODULES[module].events)
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


def _parse_counter_specs(text: str) -> list[CounterSpec]:
    """Parse "33:34,m:77:77,mt:21:29,sh:14:22" into counter specs; module prefixes select the tile module: none =
    core module, m: = core-tile memory module, mt: = memory tile (row 1), sh: = shim/PL tile (row 0). Event names
    are accepted."""
    specs: list[CounterSpec] = []
    for item in text.split(","):
        parts = [p.strip() for p in item.strip().split(":")]
        module = "core"
        if parts[0] in _MOD_PREFIXES:
            module = _MOD_PREFIXES[parts[0]]
            parts = parts[1:]
        if len(parts) != 2:
            raise ValueError(f"bad counter pair '{item}': expected start:stop")
        specs.append(CounterSpec(module, _resolve_event(module, parts[0]), _resolve_event(module, parts[1])))
    validate_counter_specs(specs)
    return specs


def _parse_events(text: str) -> list[object]:
    from aie.utils.trace.events import CoreEvent

    events = []
    for tok in text.split(","):
        tok = tok.strip()
        events.append(CoreEvent[tok] if not tok.startswith("0x") else int(tok, 16))
    return events


class _Session:
    """The process's profiling state: its configuration (from the environment at the first ``get_config``), the
    profilers and trace dumps to finish at exit, the IRON hook."""

    def __init__(self) -> None:
        self.config: ProfileConfig | None = None
        self.parsed = False
        self.hook_installed = False
        self.profilers: list[_native.Profiler] = []
        self.trace_dumps: list[tuple[str, str]] = []  # (ttbin path, physical mlir path)


_SESSION = _Session()


def get_config() -> ProfileConfig | None:
    """Return the profiling config, or None when profiling is disabled."""
    if _SESSION.parsed:
        return _SESSION.config
    _SESSION.parsed = True
    config = ProfileConfig.from_env(os.environ)
    if config is not None:
        os.makedirs(config.out_dir, exist_ok=True)
        atexit.register(_finalize)
    _SESSION.config = config
    return config


def install_iron_hook() -> None:
    """Patch aie.iron.Program.resolve_program to enable stream tracing.

    Idempotent. Only effective in `trace` mode; in `counters` mode the hook
    resolves programs unchanged.
    """
    if _SESSION.hook_installed:
        return
    _SESSION.hook_installed = True

    import aie.iron

    orig_resolve = aie.iron.Program.resolve_program

    def resolving(self: Any, *args: Any, **kwargs: Any) -> Any:
        cfg = get_config()
        if cfg is not None and cfg.mode == "trace":
            workers = cfg.trace_tiles.pick([w for c in self._configurations for w in c.workers])
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


def instrumented_tiles_of(
    mlir_text: str, counter_specs: list[CounterSpec]
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
    classes = {MODULES[spec.module].tiles for spec in counter_specs}
    out: dict[str, list[tuple[int, int]]] = {}
    if "core" in classes:
        out["core"] = core_tiles
    if "mt" in classes:
        out["mt"] = sorted((c, r) for c, r in all_xy if r == MEMTILE_ROW)
    if "sh" in classes:
        out["sh"] = sorted((c, r) for c, r in all_xy if r == SHIM_ROW)
    return out


def _run_aie_opt(pipeline: str, src_path: str, out_path: str, *, what: str, tail: int | None = None) -> None:
    """aie-opt with a pass pipeline: src_path -> out_path; a failure raises with its stderr (its last ``tail``
    characters)."""
    r = subprocess.run(
        [_find_aie_opt(), f"--pass-pipeline={pipeline}", src_path, "-o", out_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{r.stderr[-tail:] if tail else r.stderr}")


def inject_perf_counters(
    mlir_text: str, counter_specs: list[CounterSpec], work_dir: str
) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    """Place tiles and inject perf-counter configuration into the MLIR.

    Returns (new mlir text, tiles_by_class) where tiles_by_class maps
    "core"/"mt"/"sh" to the instrumented (col, row) lists.
    The re-placed module goes through the normal aiecc pipeline afterwards;
    re-running aie-place-tiles on already-placed tiles is a no-op (this is
    also how mlir-aie pre-lowers trace options that aiecc cannot pass).
    """
    validate_counter_specs(counter_specs)
    src_path = os.path.join(work_dir, "design_preprofile.mlir")
    placed_path = os.path.join(work_dir, "design_placed.mlir")
    with open(src_path, "w") as f:
        f.write(mlir_text)
    _run_aie_opt("builtin.module(aie.device(aie-place-tiles))", src_path, placed_path, what="aie-place-tiles")
    with open(placed_path) as f:
        placed = f.read()

    tiles_by_class = instrumented_tiles_of(placed, counter_specs)
    if not tiles_by_class:
        raise RuntimeError("profiling: design has no tiles to instrument")
    cores = len(re.findall(r"\baie\.core\(", placed))
    if "core" in tiles_by_class and len(tiles_by_class["core"]) != cores:
        # (the printer's format changed: the counters would miss cores)
        raise RuntimeError(
            f"profiling: {cores} aie.core ops in the placed MLIR, {len(tiles_by_class['core'])} core tiles recognized"
        )

    # module -> packed control register writes
    writes_per_module = {
        name: module.ctrl_writes([(spec.start, spec.stop) for spec in counter_specs if spec.module == name])
        for name, module in MODULES.items()
        if any(spec.module == name for spec in counter_specs)
    }

    lines: list[str] = []
    for cls, tile_list in tiles_by_class.items():
        writes: list[tuple[int, int]] = []
        for mod, w in writes_per_module.items():
            if cls == MODULES[mod].tiles:
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
    _run_aie_opt(
        "builtin.module(aie.device("
        "aie-place-tiles,"
        f"aie-insert-trace-flows{{{opts}}},"
        "aie-trace-to-config,aie-trace-pack-reg-writes,"
        "aie-inline-trace-config),aie-fuse-trace-buffers)",
        mlir_path,
        out_path,
        what="trace pre-lowering",
        tail=2000,
    )
    return out_path


def register_profiler(profiler: _native.Profiler) -> None:
    _SESSION.profilers.append(profiler)


def register_trace_dump(ttbin_path: str, work_dir: str) -> None:
    phys = os.path.join(work_dir, "design.mlir.prj", "input_with_addresses.mlir")
    _SESSION.trace_dumps.append((ttbin_path, phys))


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
    for p in _SESSION.profilers:
        p.close()
    cfg = _SESSION.config
    if cfg is None or not cfg.convert:
        return
    for ttbin, phys in _SESSION.trace_dumps:
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
