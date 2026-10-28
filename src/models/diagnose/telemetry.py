"""What the SoC reports while the diagnosis runs: amdgpu's ``gpu_metrics`` (version 3.0 on Strix Point, Krackan Point
and Strix Halo: SMU 14.0.x), the clock of the CPU that takes the NPU's interrupts, CPU load, memory events.

``gpu_metrics`` values are the SMU's time-filtered averages (a ~1 s constant: ``time_filter_alphavalue``): they tell
the clocks and limits a measurement ran under, not the rate of a single dispatch. Its bandwidth fields are uint16 MB/s
(at most 65.5 GB/s); on Strix Halo the NPU's own fields (column activity, reads, writes, power) stay 0, and FCLK runs at
the top of its DPM table whenever the NPU's DMA does, also with ``pp_dpm_fclk`` pinned low."""

from __future__ import annotations

import ctypes as C
import glob
import os
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Callable

_u16, _u32, _u64 = C.c_uint16, C.c_uint32, C.c_uint64


class GpuMetricsV3(C.Structure):
    """``struct gpu_metrics_v3_0`` (drivers/gpu/drm/amd/include/kgd_pp_interface.h; natural alignment, 264 bytes)."""

    _fields_ = [("structure_size", _u16), ("format_revision", C.c_uint8), ("content_revision", C.c_uint8),
                ("temperature_gfx", _u16), ("temperature_soc", _u16), ("temperature_core", _u16 * 16),
                ("temperature_skin", _u16), ("average_gfx_activity", _u16), ("average_vcn_activity", _u16),
                ("average_ipu_activity", _u16 * 8), ("average_core_c0_activity", _u16 * 16),
                ("average_dram_reads", _u16), ("average_dram_writes", _u16), ("average_ipu_reads", _u16),
                ("average_ipu_writes", _u16), ("system_clock_counter", _u64), ("average_socket_power", _u32),
                ("average_ipu_power", _u16), ("average_apu_power", _u32), ("average_gfx_power", _u32),
                ("average_dgpu_power", _u32), ("average_all_core_power", _u32), ("average_core_power", _u16 * 16),
                ("average_sys_power", _u16), ("stapm_power_limit", _u16), ("current_stapm_power_limit", _u16),
                ("average_gfxclk_frequency", _u16), ("average_socclk_frequency", _u16),
                ("average_vpeclk_frequency", _u16), ("average_ipuclk_frequency", _u16),
                ("average_fclk_frequency", _u16), ("average_vclk_frequency", _u16),
                ("average_uclk_frequency", _u16), ("average_mpipu_frequency", _u16), ("current_coreclk", _u16 * 16),
                ("current_core_maxfreq", _u16), ("current_gfx_maxfreq", _u16), ("throttle_residency_prochot", _u32),
                ("throttle_residency_spl", _u32), ("throttle_residency_fppt", _u32), ("throttle_residency_sppt", _u32),
                ("throttle_residency_thm_core", _u32), ("throttle_residency_thm_gfx", _u32),
                ("throttle_residency_thm_soc", _u32), ("time_filter_alphavalue", _u32)]


# (the fields a sample keeps: name -> (field, scale); temperatures in centi-degrees)
FIELDS = {
    "fclk_mhz": ("average_fclk_frequency", 1), "uclk_mhz": ("average_uclk_frequency", 1),
    "socclk_mhz": ("average_socclk_frequency", 1), "npu_clk_mhz": ("average_ipuclk_frequency", 1),
    "mpnpu_clk_mhz": ("average_mpipu_frequency", 1), "dram_read_mbs": ("average_dram_reads", 1),
    "dram_write_mbs": ("average_dram_writes", 1), "npu_read_mbs": ("average_ipu_reads", 1),
    "npu_write_mbs": ("average_ipu_writes", 1), "npu_mw": ("average_ipu_power", 1),
    "socket_mw": ("average_socket_power", 1), "soc_c": ("temperature_soc", 0.01),
    "skin_c": ("temperature_skin", 0.01),
}
THROTTLE = ("prochot", "spl", "fppt", "sppt", "thm_core", "thm_gfx", "thm_soc")


def gpu_metrics_path() -> str | None:
    """The amdgpu device's ``gpu_metrics`` (world-readable), if the APU's iGPU driver is loaded."""
    return next(iter(sorted(glob.glob("/sys/class/drm/card*/device/gpu_metrics"))), None)


def read_gpu_metrics(path: str) -> GpuMetricsV3 | None:
    """One reading (an SMU table transfer each: sample at a few Hz), or None when it is not version 3."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    size = C.sizeof(GpuMetricsV3)
    if len(raw) < 4 or raw[2] != 3:
        return None
    return GpuMetricsV3.from_buffer_copy(raw[:size].ljust(size, b"\0"))


PROCHOT_WARN = 0.05  # (share of a measurement; Strix Halo never asserts it)


def prochot_ms() -> int | None:
    """The SMU's PROCHOT residency counter (ms; compare two readings): how long the platform has held the SoC's
    PROCHOT_L input, its external throttle (the laptop's embedded controller, charger or a voltage regulator), which
    puts CPUs, fabric, memory and NPU at their lowest clocks. The chip's own temperature limits count in the thm_*
    counters instead. None without gpu_metrics v3."""
    path = gpu_metrics_path()
    m = read_gpu_metrics(path) if path else None
    return None if m is None else m.throttle_residency_prochot


def prochot_share(start: int | None, seconds: float) -> float | None:
    """The share of the last ``seconds`` the platform held PROCHOT, from a ``prochot_ms`` reading at their start."""
    now = prochot_ms()
    if start is None or now is None or seconds <= 0:
        return None
    return min(1.0, (now - start) / 1e3 / seconds)


def prochot_warning(share: float | None) -> str | None:
    """What to tell the user when the platform held PROCHOT for ``share`` of a measurement (None below PROCHOT_WARN)."""
    if share is None or share < PROCHOT_WARN:
        return None
    return (f"the platform held PROCHOT for {share:.0%} of the measurement: its external throttle input (the laptop's "
            "embedded controller, charger or a voltage regulator; not the chip's temperature) kept the whole SoC at its "
            "lowest clocks, so the numbers are the platform's, not the NPU's. On the laptop measured it follows the NPU's "
            "load (the dense prefill GEMMs), on any power source; docs/probe.md, PROCHOT")


def dpm_levels(name: str) -> list[int]:
    """The MHz levels of amdgpu's ``pp_dpm_<name>`` (fclk, mclk, socclk): the SMU's DPM table."""
    out = []
    for path in sorted(glob.glob(f"/sys/class/drm/card*/device/pp_dpm_{name}"))[:1]:
        try:
            with open(path) as f:
                lines = f.read().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 2 and parts[1].lower().endswith("mhz"):
                    out.append(int(parts[1][:-3]))
        except (OSError, ValueError):
            pass
    return out


def _cpu_times() -> tuple[float, float] | None:
    """(busy, total) clock ticks of all CPUs (None off Linux)."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return float(sum(v) - idle), float(sum(v))


def _own_cpu() -> float:
    t = os.times()
    return t.user + t.system


VMSTAT = ("thp_fault_alloc", "thp_fault_fallback", "compact_stall", "pswpin", "pswpout", "pgmajfault",
          "numa_hint_faults")


def vmstat() -> dict[str, int]:
    out = {}
    try:
        with open("/proc/vmstat") as f:
            lines = f.read().splitlines()
        for line in lines:
            k, _, v = line.partition(" ")
            if k in VMSTAT:
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


class Sampler:
    """Samples in a thread every ``period`` s, each tagged with the step that ran (``step``); ``summary`` gives each
    step's median / min / max per field, the CPU load of other processes, the throttle residencies' and /proc/vmstat's
    increments."""

    def __init__(self, irq_cpu: int, period: float = 0.2, extra: Callable[[], dict[str, float]] | None = None):
        self.path = gpu_metrics_path()
        self.extra = extra  # (more fields per sample: the driver's NPU sensors)
        self.irq_cpu, self.period = irq_cpu, period
        self.tag = "setup"
        self.samples: dict[str, list[dict[str, float]]] = defaultdict(list)
        self._marks: dict[str, dict[str, object]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="diagnose-telemetry", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def step(self, tag: str) -> None:
        """The samples from now on belong to ``tag``; marks its start for the increments."""
        self.tag = tag
        self._marks.setdefault(tag, self._mark())

    def end(self, tag: str) -> None:
        """Ends ``tag``: its increments from its start."""
        start = self._marks.get(tag)
        if start is None or "end" in start:
            return
        start["end"] = self._mark()
        self.tag = "between"

    def _mark(self) -> dict[str, object]:
        m = read_gpu_metrics(self.path) if self.path else None
        return {"t": time.perf_counter(), "cpu": _cpu_times(), "own": _own_cpu(), "vmstat": vmstat(),
                "throttle": {k: getattr(m, f"throttle_residency_{k}") for k in THROTTLE} if m else None}

    def _irq_cpu_khz(self) -> float | None:
        try:
            with open(f"/sys/devices/system/cpu/cpu{self.irq_cpu}/cpufreq/scaling_cur_freq") as f:
                return float(f.read())
        except (OSError, ValueError):
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            s: dict[str, float] = {}
            m = read_gpu_metrics(self.path) if self.path else None
            if m is not None:
                for name, (field, scale) in FIELDS.items():
                    s[name] = getattr(m, field) * scale
                s["npu_busy_pct"] = max(m.average_ipu_activity)
            if (khz := self._irq_cpu_khz()) is not None:
                s["irq_cpu_mhz"] = khz / 1000
            if self.extra is not None:
                s.update(self.extra())
            self.samples[self.tag].append(s)

    def summary(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for tag, mark in self._marks.items():
            entry: dict[str, object] = {}
            rows = self.samples.get(tag, [])
            for name in sorted({k for r in rows for k in r}):
                v = [r[name] for r in rows if name in r]
                entry[name] = {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
            end = mark.get("end")
            if isinstance(end, dict):
                dt = end["t"] - mark["t"]
                entry["seconds"] = dt
                if mark["cpu"] and end["cpu"]:
                    busy = end["cpu"][0] - mark["cpu"][0]
                    total = end["cpu"][1] - mark["cpu"][1]
                    ncpu = os.cpu_count() or 1
                    # (clock ticks: SC_CLK_TCK a second per CPU; the process's own CPU time taken out)
                    other = busy / os.sysconf("SC_CLK_TCK") - (end["own"] - mark["own"])
                    entry["other_cpu_load"] = max(0.0, other / dt / ncpu) if total else None
                entry["vmstat"] = {k: end["vmstat"].get(k, 0) - mark["vmstat"].get(k, 0) for k in end["vmstat"]}
                if mark["throttle"] and end["throttle"]:
                    entry["throttle"] = {k: end["throttle"][k] - mark["throttle"][k] for k in THROTTLE}
            out[tag] = entry
        return out
