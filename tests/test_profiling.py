"""Tests for NPU-side profiling (artlite.profiling).

Each test runs in a subprocess because the profiling configuration is parsed
once per process (from ARTLITE_PROFILE_* env vars) and the trace-mode iron
hook is process-global.

Counters mode: compile-time perf-counter config injection + ioctl readback.
Trace mode: hardware event streaming to DDR + Chromium JSON conversion.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ONE_CORE_SCRIPT = """
import numpy as np
from ml_dtypes import bfloat16
from tests.operators.elementwise import GeluOperator

# 64 kernel calls (512-element tiles) per dispatch on one core
op = GeluOperator(size=64 * 512, num_columns=1, tile_size=512).compile()
x = np.random.default_rng(0).standard_normal(64 * 512, np.float32).astype(bfloat16)
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
import numpy as np
from ml_dtypes import bfloat16
from tests.operators.elementwise import GeluOperator

op = GeluOperator(size=16 * 1024, num_columns=8, num_channels=2, tile_size=512).compile()
op.run(np.random.default_rng(0).standard_normal(16 * 1024, np.float32).astype(bfloat16))
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


def test_counter_registers():
    """The register map's counters: addresses and control words, and no address past a module's counters."""
    from artlite import npu_regs

    assert [npu_regs.CORE.address(i) for i in range(4)] == [0x31520, 0x31524, 0x31528, 0x3152C]
    assert [npu_regs.MEM.address(i) for i in range(2)] == [0x11020, 0x11024]
    assert [npu_regs.SHIM.address(i) for i in range(2)] == [0x31020, 0x31024]
    assert [npu_regs.MEMTILE.address(i) for i in range(4)] == [0x91020, 0x91024, 0x91028, 0x9102C]
    for module in npu_regs.MODULES.values():
        for index in (-1, module.max_counters):
            with pytest.raises(ValueError, match="counters"):
                module.address(index)
    assert npu_regs.CORE.ctrl_writes([(33, 34), (26, 26), (1, 1)]) == [
        (0x31500, 33 | 34 << 8 | 26 << 16 | 26 << 24),
        (0x31504, 1 | 1 << 8),
    ]
    assert npu_regs.window_counter_writes() == [
        (0x31500, 33 | 34 << 8 | 26 << 16 | 26 << 24),
        (0x31504, 1 | 1 << 8 | 33 << 16 | 33 << 24),
    ]
    assert npu_regs.window_counter_writes(stall=23, memory_events=[7, 9])[2:] == [(0x11000, 7 | 7 << 8 | 9 << 16 | 9 << 24)]
    with pytest.raises(ValueError, match="counters"):
        npu_regs.MEM.ctrl_writes([(1, 1)] * 3)


AIE_RT_PARAMS = Path(__file__).resolve().parents[2] / "mlir-aie/third_party/aie-rt/driver/src/global/xaie2pgbl_params.h"


@pytest.mark.skipif(not AIE_RT_PARAMS.exists(), reason="no mlir-aie checkout next to this one")
def test_counter_registers_match_aie_rt():
    """The register map is the NPU's (aie-rt's xaie2pgbl_params.h), not AM029's."""
    from artlite import npu_regs

    text = AIE_RT_PARAMS.read_text()

    def define(name: str) -> int:
        return int(re.search(rf"#define XAIE2PGBL_{name}\s+(0x[0-9A-Fa-f]+)", text).group(1), 16)

    for module, prefix, ctrl in (
        (npu_regs.CORE, "CORE_MODULE_PERFORMANCE_", ("CONTROL0", "CONTROL1")),
        (npu_regs.MEM, "MEMORY_MODULE_PERFORMANCE_", ("CONTROL0",)),
        (npu_regs.MEMTILE, "MEM_TILE_MODULE_PERFORMANCE_", ("CONTROL0", "CONTROL1")),
        (npu_regs.SHIM, "PL_MODULE_PERFORMANCE_", ("CTRL0",)),
    ):
        assert list(module.ctrl) == [define(prefix + name) for name in ctrl]
        assert [module.address(i) for i in range(module.max_counters)] == [
            define(f"{prefix}COUNTER{i}") for i in range(module.max_counters)
        ]


def test_profile_config_from_env():
    """The configuration is built from the environment once; a malformed value names its variable."""
    from artlite.profiling import CounterSpec, ProfileConfig, TileSelection

    assert ProfileConfig.from_env({}) is None
    assert ProfileConfig.from_env({"ARTLITE_PROFILE": "off"}) is None
    config = ProfileConfig.from_env({"ARTLITE_PROFILE": "1"})
    assert config.mode == "counters" and config.sample_every == 1 and config.convert
    assert [tuple(spec) for spec in config.counter_specs] == [("core", 33, 34), ("core", 34, 33), ("core", 33, 33),
                                                              ("core", 34, 34)]
    config = ProfileConfig.from_env({
        "ARTLITE_PROFILE": "trace", "ARTLITE_PROFILE_TRACE_SIZE": "4096", "ARTLITE_PROFILE_TILES": "first:3",
        "ARTLITE_PROFILE_SAMPLE_EVERY": "0", "ARTLITE_PROFILE_LATERAL": "1", "ARTLITE_PROFILE_CONVERT": "0",
    })
    assert (config.mode, config.trace_size, config.trace_tiles.first, config.sample_every) == ("trace", 4096, 3, 1)
    assert config.trace_lateral and not config.trace_distribute and not config.convert
    assert TileSelection.parse("0,2;1,3").tiles == frozenset({(0, 2), (1, 3)})
    for name, value in (("TRACE_SIZE", "4k"), ("SAMPLE_EVERY", "x"), ("TILES", "first:x"), ("TILES", "0;2"),
                        ("COUNTERS", "1:2:3")):
        with pytest.raises(ValueError, match=f"ARTLITE_PROFILE_{name}|bad counter pair"):
            ProfileConfig.from_env({"ARTLITE_PROFILE": "1", f"ARTLITE_PROFILE_{name}": value})
    with pytest.raises(ValueError, match="at most"):
        ProfileConfig(mode="counters", out_dir="x", counter_specs=[CounterSpec("mem", 1, 1)] * 3)


def test_profile_cache_key():
    """Every field that is not runtime-only is in the cache key: a new field cannot reuse an unprofiled build."""
    import dataclasses

    from artlite.profiling import CounterSpec, ProfileConfig, TileSelection

    base = ProfileConfig(mode="trace", out_dir="x")
    changes = {
        "mode": "counters", "counter_specs": [CounterSpec("core", 1, 2)], "trace_size": 1, "trace_events": [1],
        "trace_tiles": TileSelection.parse("first:1"), "trace_lateral": True, "trace_distribute": True,
    }
    for field in dataclasses.fields(ProfileConfig):
        if field.name in ProfileConfig.RUNTIME_FIELDS:
            continue
        assert field.name in changes, f"{field.name}: add it to this test"
        other = dataclasses.replace(base, **{field.name: changes[field.name]})
        if field.name == "counter_specs":  # (counters mode only)
            base_c = dataclasses.replace(base, mode="counters")
            other = dataclasses.replace(base_c, counter_specs=changes[field.name])
            assert other.cache_key() != base_c.cache_key(), field.name
        else:
            assert other.cache_key() != base.cache_key(), field.name


def test_instrumented_tiles():
    """The tiles of a placed design come from its tile and aie.core ops."""
    from artlite.profiling import CounterSpec, instrumented_tiles_of

    mlir = """%t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t01 = aie.tile(0, 1)
    %t00 = aie.tile(0, 0)
    %c = aie.core(%t02) { aie.end }
    %d = aie.core(%t12) { aie.end }"""
    specs = [CounterSpec("core", 33, 34), CounterSpec("mt", 1, 1), CounterSpec("sh", 1, 1)]
    assert instrumented_tiles_of(mlir, specs) == {"core": [(0, 2), (1, 2)], "mt": [(0, 1)], "sh": [(0, 0)]}
    assert instrumented_tiles_of(mlir, specs[1:]) == {"mt": [(0, 1)], "sh": [(0, 0)]}


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
