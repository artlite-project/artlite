"""Tests for NPU-side profiling (artlite.profiling).

Each test runs in a subprocess because the profiling configuration is parsed
once per process (from ARTLITE_PROFILE_* env vars) and the trace-mode iron
hook is process-global.

Counters mode: compile-time perf-counter config injection + ioctl readback.
Trace mode: hardware event streaming to DDR + Chromium JSON conversion.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ONE_CORE_SCRIPT = """
import torch
from tests.operators.elementwise import GeluOperator

# 64 kernel calls (512-element tiles) per dispatch on one core
op = GeluOperator(size=64 * 512, num_columns=1, tile_size=512).compile()
x = torch.randn(64 * 512).bfloat16()
op.run(x)
op.run(x)
"""


def _run_profiled(tmp_path: Path, profile_args: dict[str, str], script: str) -> str:
    env = dict(os.environ)
    env["ARTLITE_PROFILE_DIR"] = str(tmp_path / "artlite_profile")
    env.update(profile_args)
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode and "NotImplementedError" in r.stderr:  # (counters read AIE tiles: not on Linux's own driver)
        pytest.skip(r.stderr.strip().splitlines()[-1])
    assert r.returncode == 0, f"profiled run failed:\n{r.stdout}\n{r.stderr}"
    return env["ARTLITE_PROFILE_DIR"]


def test_profiling_counters(tmp_path):
    """Counters mode reads back event0/event1 windows per dispatch."""
    out_dir = _run_profiled(tmp_path, {"ARTLITE_PROFILE": "1"}, ONE_CORE_SCRIPT)

    files = [f for f in os.listdir(out_dir) if f.endswith("_counters.json")]
    assert len(files) == 1, f"expected 1 counters json, got {files}"
    with open(os.path.join(out_dir, files[0])) as f:
        events = json.load(f)

    counters = [e for e in events if e.get("ph") == "C" and e["cat"] == "npu_counter"]
    assert counters, "no counter events produced"

    # two dispatches x 4 counters x 1 tile
    assert len(counters) == 8

    # host tracks per dispatch: wall time, the driver's device busy time, H clock
    host = {}
    for e in events:
        if e.get("ph") == "C" and e["cat"] == "host":
            host.setdefault(e["name"], []).append(e["args"]["value"])
    assert sorted(host) == ["H clock MHz", "device busy us", "dispatch wall us"]
    assert all(0 < b <= w for b, w in zip(host["device busy us"], host["dispatch wall us"]))
    assert all(0 < mhz <= 2000 for mhz in host["H clock MHz"])

    busy = [e for e in counters if e["name"] == "INSTR_EVENT_0->INSTR_EVENT_1 cycles"]
    counts = [e for e in counters if e["name"] == "INSTR_EVENT_0->INSTR_EVENT_0 count"]

    # 64 kernel calls (tiles) per dispatch; each window is tens of cycles
    assert all(e["args"]["value"] == 64 for e in counts)
    assert all(0 < e["args"]["value"] for e in busy)
    # windows sum: 64 calls x ~0.5K cycles, well below 64 x 10k
    assert all(e["args"]["value"] < 64 * 10_000 for e in busy)

    meta = [e for e in events if e.get("ph") == "M"]
    assert any(
        e.get("args", {}).get("name") == "tile(0,2)" for e in meta
    ), "tile metadata missing"


def test_profiling_trace(tmp_path):
    """Trace mode streams events to DDR and converts to Chromium JSON."""
    out_dir = _run_profiled(
        tmp_path,
        {"ARTLITE_PROFILE": "trace", "ARTLITE_PROFILE_TRACE_SIZE": str(256 * 1024)},
        ONE_CORE_SCRIPT,
    )

    # Dumps are cumulative across dispatches; the converted JSON covers all.
    files = sorted(os.listdir(out_dir))
    bins = [f for f in files if f.endswith(".ttbin")]
    jsons = [f for f in files if f.endswith(".json")]
    assert len(bins) == 1, f"expected 1 trace dump, got {files}"
    assert len(jsons) == 1, f"expected 1 converted json, got {files}"

    with open(os.path.join(out_dir, jsons[0])) as f:

        events = json.load(f)
    for name in ("INSTR_EVENT_0", "INSTR_EVENT_1"):
        markers = [e for e in events if e.get("ph") == "B" and e.get("name") == name]
        # 2 dispatches x 64 kernel calls; an odd last trace packet stays in the shim DMA (64-byte
        # bursts of 32-byte packets), losing the last dispatch's final events
        assert len(markers) in (127, 128), f"{name}: got {len(markers)} markers"


ARRAY_SCRIPT = """
import torch
from tests.operators.elementwise import GeluOperator

op = GeluOperator(size=16 * 1024, num_columns=8, num_channels=2, tile_size=512).compile()
op.run(torch.randn(16 * 1024).bfloat16())
"""


def test_profiling_counters_mem_module(tmp_path):
    """Memory-module counters (m: prefix, DM bank conflicts) are supported."""
    out_dir = _run_profiled(
        tmp_path,
        {
            "ARTLITE_PROFILE": "1",
            "ARTLITE_PROFILE_COUNTERS": (
                "INSTR_EVENT_0:INSTR_EVENT_1," "m:CONFLICT_DM_BANK_0:CONFLICT_DM_BANK_0"
            ),
        },
        ONE_CORE_SCRIPT,
    )
    files = [f for f in os.listdir(out_dir) if f.endswith("_counters.json")]
    assert len(files) == 1
    with open(os.path.join(out_dir, files[0])) as f:
        events = json.load(f)
    names = {e.get("name") for e in events if e.get("ph") == "C"}
    assert "INSTR_EVENT_0->INSTR_EVENT_1 cycles" in names
    assert "CONFLICT_DM_BANK_0->CONFLICT_DM_BANK_0 count" in names

    busy = [
        e
        for e in events
        if e.get("ph") == "C" and e["name"] == "INSTR_EVENT_0->INSTR_EVENT_1 cycles"
    ]
    assert busy and all(e["args"]["value"] > 0 for e in busy)


def test_counter_limits():
    """Each tile module takes at most its AIE2P counters: core 4, memory module 2, memtile 4, shim 2."""
    from artlite.profiling import _parse_counter_specs

    for prefix, n in (("", 4), ("m:", 2), ("mt:", 4), ("sh:", 2)):
        _parse_counter_specs(",".join([f"{prefix}1:1"] * n))
        with pytest.raises(ValueError, match="at most"):
            _parse_counter_specs(",".join([f"{prefix}1:1"] * (n + 1)))


def test_profiling_counters_shim(tmp_path):
    """Shim counters (sh: prefix) count the shim DMA's tasks."""
    out_dir = _run_profiled(
        tmp_path,
        {
            "ARTLITE_PROFILE": "1",
            "ARTLITE_PROFILE_COUNTERS": (
                "INSTR_EVENT_0:INSTR_EVENT_1,"
                "sh:DMA_MM2S_0_START_TASK:DMA_MM2S_0_START_TASK,"
                "sh:DMA_S2MM_0_FINISHED_TASK:DMA_S2MM_0_FINISHED_TASK"
            ),
        },
        ONE_CORE_SCRIPT,
    )
    files = [f for f in os.listdir(out_dir) if f.endswith("_sh_counters.json")]
    assert len(files) == 1, f"expected 1 shim counters json, got {os.listdir(out_dir)}"
    with open(os.path.join(out_dir, files[0])) as f:
        events = json.load(f)
    counts = {}
    for e in events:
        if e.get("ph") == "C":
            counts.setdefault(e["name"], []).append(e["args"]["value"])
    assert sorted(counts) == [
        "DMA_MM2S_0_START_TASK->DMA_MM2S_0_START_TASK count",
        "DMA_S2MM_0_FINISHED_TASK->DMA_S2MM_0_FINISHED_TASK count",
    ]
    assert all(v > 0 for vs in counts.values() for v in vs), counts


def test_profiling_counters_multi_column(tmp_path):
    """Counters mode instruments every core of a design on the whole array."""
    out_dir = _run_profiled(tmp_path, {"ARTLITE_PROFILE": "counters"}, ARRAY_SCRIPT)

    files = [f for f in os.listdir(out_dir) if f.endswith("_counters.json")]
    assert len(files) == 1, f"expected 1 counters json, got {files}"
    with open(os.path.join(out_dir, files[0])) as f:
        events = json.load(f)

    counters = [e for e in events if e.get("ph") == "C" and e["cat"] == "npu_counter"]
    tiles = {e["pid"] for e in counters}
    # 8 columns x 2 rows: 16 pids
    assert len(tiles) == 16, f"expected 16 instrumented tiles, got {len(tiles)}"

    busy = [e for e in counters if e["name"] == "INSTR_EVENT_0->INSTR_EVENT_1 cycles"]
    assert busy and all(e["args"]["value"] > 0 for e in busy)
