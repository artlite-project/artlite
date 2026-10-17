"""Analysis helpers for NPU profiling output (artlite.profiling).

Two consumers:

- ``synthesize_kernel_slices(events, phys_mlir_text)`` turns the raw
  marker/signal events produced by mlir-aie's ``parse_trace`` (event0/event1
  are 1-cycle-remote B/E pulse pairs, which look like meaningless 1-us slivers
  in perfetto) into "X" (complete) slices spanning event0 -> event1, named
  with the kernel function(s) of the tile they were measured on (recovered
  from the placed MLIR). Timestamps stay in NPU cycles (see note in args).

- ``summarize_counters(events)`` aggregates a counters-mode JSON into
  per-tile/per-counter totals and cross-tile imbalance statistics.

Both are pure Python on parsed JSON-style event lists, so they also work as a
library for downstream analysis scripts.
"""

import re
import statistics
from collections import defaultdict


def kernel_names_by_tile(mlir_text: str) -> dict[tuple[int, int], list[str]]:
    """Map (col, row) of placed tiles to the kernel functions they call.

    Works on any MLIR containing ``aie.tile(c, r)``, ``aie.core(%t)`` blocks
    and ``func.call @name`` inside them (e.g. design.mlir.prj/
    input_with_addresses.mlir).
    """
    tiles: dict[str, tuple[int, int]] = {}
    for m in re.finditer(r"(%\S+)\s*=\s*aie\.tile\((\d+),\s*(\d+)\)", mlir_text):
        tiles[m.group(1)] = (int(m.group(2)), int(m.group(3)))

    result: dict[tuple[int, int], list[str]] = {}
    for m in re.finditer(
        r"aie\.core\((%\S+?)(?:\s*,|\))[^\{]*\{(.*?)\n\s*aie\.end[^\n]*\n\s*\}",
        mlir_text,
        re.DOTALL,
    ):
        tile = tiles.get(m.group(1))
        if tile is None:
            continue
        body = m.group(2)
        names = []
        for cm in re.finditer(r"func\.call\s+@([\w.$:]+)", body):
            fn = cm.group(1)
            if fn not in names:
                names.append(fn)
        result[tile] = names
    return result


def _pid_tile(events: list[dict]) -> dict[int, tuple[int, int] | None]:
    """pid -> (col, row) from the process_name metadata events.

    mlir-aie's process metadata names look like "core_trace for tile(row,col)"
    (note: row first, column second).
    """
    out: dict[int, tuple[int, int] | None] = {}
    for e in events:
        if e.get("ph") != "M" or e.get("name") != "process_name":
            continue
        name = e.get("args", {}).get("name", "")
        # mlir-aie names processes like "core_trace for tile2,0" where the
        # numbers are "row,col".
        m = re.search(r"tile\(?(\d+),(\d+)\)?", name)
        out[int(e["pid"])] = (
            (int(m.group(2)), int(m.group(1))) if m else None
        )  # -> (col, row)
    return out


def synthesize_kernel_slices(
    events: list[dict], kernels_by_tile: dict[tuple[int, int], list[str]]
) -> list[dict]:
    """Append "X" complete slices for every event0 -> event1 window.

    Existing B/E pulse events are kept; the synthesized slices are what you
    should look at in perfetto. Slice ``name`` lists the tile's kernel
    functions; ``args`` carry the tile, duration in cycles, and a
    ``window_class`` ("shortest"/"longest") separating the typical
    init/zeroing kernel window from the compute kernel window when a tile has
    two kernels with clearly separated durations.
    """
    tiles = _pid_tile(events)

    # collect pulse timestamps per pid
    pulses: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "B" and e.get("name") in ("INSTR_EVENT_0", "INSTR_EVENT_1"):
            pulses[int(e["pid"])].append((float(e["ts"]), e["name"]))

    new_events: list[dict] = []
    for pid, seq in pulses.items():
        seq.sort(key=lambda x: x[0])
        loc = tiles.get(pid)
        names: list[str] = (
            kernels_by_tile.get(loc, []) if loc is not None else []
        )
        label = "/".join(names) if names else "kernel"
        new_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 9,
                "args": {"name": f"kernel windows ({label})"},
            }
        )

        windows: list[tuple[float, float]] = []  # (ts, dur)
        i = 0
        while i < len(seq) - 1:
            if seq[i][1] == "INSTR_EVENT_0":
                j = i + 1
                while j < len(seq) and seq[j][1] != "INSTR_EVENT_1":
                    j += 1
                if j >= len(seq):
                    break
                windows.append((seq[i][0], seq[j][0] - seq[i][0]))
                i = j + 1
            else:
                i += 1
        if not windows:
            continue

        durs = [d for _, d in windows]
        lo, hi = min(durs), max(durs)
        split = (lo * hi) ** 0.5  # geometric midpoint
        bimodal = hi > 4 * lo and len(names) > 1

        for ts, dur in windows:
            cls = "n/a"
            if bimodal:
                cls = "zero/init (short)" if dur <= split else "compute (long)"
            new_events.append(
                {
                    "name": label,
                    "cat": "npu_kernel_window",
                    "ph": "X",
                    "ts": ts,
                    "dur": dur,
                    "pid": pid,
                    "tid": 9,
                    "args": {
                        "cycles": int(dur),
                        "tile": f"({loc[0]},{loc[1]})" if loc else "?",
                        "window_class": cls,
                    },
                }
            )
    return events + new_events


def summarize_counters(events: list[dict]) -> dict:
    """Summarize counters-mode Chromium JSON.

    Returns ``{labels, tiles, by_label_tile, dispatches}``.
    by_label_tile[label][tile] = {"total", "mean_per_dispatch", "min", "max",
    "samples"}  (values are per-dispatch deltas; "total" is post-first-sample
    accumulation, reflecting steady-state measurements).
    """
    labels_by_pid_tid: dict[tuple[int, int], str] = {}
    tiles_by_pid: dict[int, str] = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e.get("name") == "process_name":
            tiles_by_pid[int(e["pid"])] = e["args"]["name"]
        elif e.get("name") == "thread_name":
            labels_by_pid_tid[(int(e["pid"]), int(e["tid"]))] = e["args"]["name"]

    acc: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    dispatch_ts: set[float] = set()
    for e in events:
        if e.get("ph") != "C":
            continue
        key = (int(e["pid"]), int(e["tid"]))
        label = labels_by_pid_tid.get(key, f"c{key[1]}")
        tname = tiles_by_pid.get(key[0]) or str(
            (e.get("args", {}).get("col"), e.get("args", {}).get("row"))
        )
        v = int(e["args"]["value"])
        acc[label][tname].append(v)
        dispatch_ts.add(float(e["ts"]))

    def _stat(xs: list[int]) -> dict:
        return {
            "total": sum(xs),
            "mean_per_dispatch": (sum(xs) / len(xs)) if xs else 0.0,
            "min": min(xs) if xs else 0,
            "max": max(xs) if xs else 0,
            "samples": len(xs),
        }

    summary = {
        "labels": sorted({n for n in labels_by_pid_tid.values()}),
        "tiles": sorted(tiles_by_pid.values()),
        "dispatches": len(dispatch_ts),
        "by_label_tile": {
            lbl: {t: _stat(xs) for t, xs in tiles.items()}
            for lbl, tiles in acc.items()
        },
    }
    return summary


def print_counter_summary(events: list[dict]) -> None:
    """Print a human-readable table of summarize_counters() results."""
    s = summarize_counters(events)
    print(f"dispatches sampled: {s['dispatches']}, tiles: {len(s['tiles'])}")
    for lbl, tiles in s["by_label_tile"].items():
        vals = [v["mean_per_dispatch"] for v in tiles.values()]
        if not vals:
            continue
        imax = max(tiles.items(), key=lambda kv: kv[1]["mean_per_dispatch"])
        imin = min(tiles.items(), key=lambda kv: kv[1]["mean_per_dispatch"])
        print(f"\n{lbl}:")
        print(
            f"  mean over tiles {statistics.mean(vals):.1f}, "
            f"min {imin[1]['mean_per_dispatch']:.1f} {imin[0]}, "
            f"max {imax[1]['mean_per_dispatch']:.1f} {imax[0]}, "
            f"imbalance {(max(vals) - min(vals)) / (statistics.mean(vals) or 1):.2%}"
        )
