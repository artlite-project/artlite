"""``artlite-probe``: the machine and its NPU's hardware numbers in one table, next to those of the Strix Halo that
Artlite is measured on and of a Krackan Point and a Strix Point laptop (issue #5): the array clock as the cores count
it, the dispatch round trip, the DDR read and write rates of the shim DMA channels (one, one shim, all 16), and
small-row gathers (``artlite.operators.probe``). About a minute, no model file. A report on another machine pastes its
output: model numbers alone cannot tell a slower clock from a narrower memory path."""

import argparse
import glob
import os
import platform
import statistics
import time
from collections.abc import Callable

import torch

import artlite
from artlite import backend
from artlite.operators.overlay.runtime import OverlayDevice, Transaction, irq_cpu
from artlite.operators.probe import ClockProbeOperator, DmaProbeOperator, Rows, dma_transaction
from artlite.parallel_compile import compile_all

from .bench import _git, _read
from .common import stale_toolchain

MB = 1 << 20
CHANNEL_BYTES = 32 * MB  # (per channel and task execution: a region of its own)
POWER_MODES = ("default", "low", "medium", "high", "turbo")


def all_channels(cols: range, chans: tuple[int, ...] = (0, 1)) -> list[tuple[int, int]]:
    return [(c, k) for c in cols for k in chans]


# (name, reads, writes, rows): the DDR cases; the reference machines' rates next to them in REFERENCES
CASES: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]], Rows | None]] = [
    ("DDR read, 1 channel (GB/s)", [(0, 0)], [], None),
    ("DDR read, 1 shim: 2 channels", all_channels(range(1)), [], None),
    ("DDR read, 8 shims: 16 channels", all_channels(range(8)), [], None),
    ("DDR read, 8 shims: 1 channel each", all_channels(range(8), (0,)), [], None),
    ("DDR write, 16 channels", [], all_channels(range(8)), None),
    ("DDR read 8 + write 8 channels", all_channels(range(8), (0,)), all_channels(range(8), (0,)), None),
    ("gather 256 B rows 64 KB apart, 16 channels", all_channels(range(8)), [], Rows(256, 256, 64 << 10)),
    ("gather 512 B rows 64 KB apart", all_channels(range(8)), [], Rows(512, 256, 64 << 10)),
    ("gather 1 KB rows 64 KB apart", all_channels(range(8)), [], Rows(1024, 256, 64 << 10)),
    ("gather 256 B rows 64 KB + 256 B apart", all_channels(range(8)), [], Rows(256, 256, (64 << 10) + 256)),
    # (a row next to the previous group's: 128 B rows cover whole 64-byte lines, 132 B rows share their end lines)
    ("write 128 B rows 64 KB + 128 B apart, 16 channels", [], all_channels(range(8)), Rows(128, 256, (64 << 10) + 128)),
    ("write 132 B rows 64 KB + 132 B apart", [], all_channels(range(8)), Rows(132, 256, (64 << 10) + 132)),
]
CLOCK = "array clock, measured (MHz)"
CLOCK_LOAD = "array clock, 32 cores of MMULs (MHz)"
PER_CYCLE = "16-channel read per array cycle (B)"
ROUND_TRIP = "dispatch round trip (us)"
# (column, its values in the table's order): Strix Halo (Ryzen AI Max+ 395, LPDDR5X-8000 256-bit; xdna-driver), Krackan
# Point (Ryzen AI 7 350, LPDDR5X-7500 128-bit; Linux 7.0's amdxdna), Strix Point (Ryzen AI 9 HX PRO 370, two DDR5-5600
# SO-DIMMs; Windows 11, XRT); all NPU power mode default, plugged in
ROWS = [CLOCK, CLOCK_LOAD, ROUND_TRIP, *(name for name, *_ in CASES), PER_CYCLE]
REFERENCES: list[tuple[str, dict[str, str]]] = [
    ("Strix Halo", dict(zip(ROWS, ("1810", "1809", "49", "13.9", "26.9", "56.8", "56.4", "57.6", "76.1", "15.6",
                                   "30.8", "56.6", "56.9", "57.2", "41.4", "31.4"), strict=True))),
    ("Krackan Point", dict(zip(ROWS, ("1808", "1810", "56", "14.2", "28.1", "97.8", "80.1", "87.2", "88.2", "28.6",
                                      "55.9", "95.6", "89.6", "?", "?", "54.1"), strict=True))),
    ("Strix Point", dict(zip(ROWS, ("1809", "1808", "94", "11.5", "21.4", "53.2", "47.3", "39.9", "53.4", "12.5",
                                    "25.1", "41.9", "14.0", "?", "?", "29.4"), strict=True))),
]


def timed(fn: Callable[[], None], *, seconds: float, warmup: int = 2) -> float:
    """The median duration (s) of ``fn`` called for ``seconds``, after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end or len(times) < 3:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def array_clock(op: ClockProbeOperator, *, seconds: float) -> tuple[float, float]:
    """(MHz, dispatch round trip in us): one core spinning for n cycles of its counter (``op.load``: all cores, with
    MMULs), timed at two n."""
    out = artlite.empty((ClockProbeOperator.WORDS,), dtype=torch.int32, device=op.device)
    durations = {}
    for n in (1 << 10, 1 << 24, 1 << 27):
        n_in = artlite.from_torch(torch.tensor([n] + [0] * (ClockProbeOperator.WORDS - 1), dtype=torch.int32),
                                  op.device)
        durations[n] = timed(lambda n_in=n_in: op.dispatch(n_in, out), seconds=seconds / 3)
        counted = int(out.to_torch()[0])
        assert counted >= n, f"the core counted {counted} of {n} cycles"
    mhz = ((1 << 27) - (1 << 24)) / (durations[1 << 27] - durations[1 << 24]) / 1e6
    return mhz, durations[1 << 10] * 1e6


def ddr_rates(op: DmaProbeOperator, *, seconds: float) -> dict[str, float]:
    """GB/s of each of CASES (bytes over the median dispatch)."""
    dev = OverlayDevice(op)
    n = 16 * CHANNEL_BYTES
    dev.alloc("src", n)
    dev.alloc("dst", n)
    src, dst = dev.address("src"), dev.address("dst")
    out = {}
    for i, (name, reads, writes, rows) in enumerate([CASES[2], *CASES]):  # (first: the DPM ramp's warm-up)
        group = 4 if rows is None else rows.n * rows.nbytes  # (a task moves whole groups of rows)
        words, total = dma_transaction(reads=reads, writes=writes, src=src, dst=dst,
                                       nbytes=CHANNEL_BYTES // group * group, repeat=2, rows=rows)
        tr = Transaction(dev, words)
        t = timed(lambda tr=tr: dev.dispatch(tr), seconds=seconds)
        if i:
            out[name] = total / t / 1e9
    return out


def _driver(dev: artlite.Device) -> str:
    """The amdxdna driver: xdna-driver's module (it has a version file) or Linux's own, and the DRM version."""
    if not os.path.isdir("/sys/module/amdxdna"):
        return "amdxdna not loaded"
    kind = "xdna-driver" if _read("/sys/module/amdxdna/version") else "Linux's own"
    return f"{dev.driver_version() or 'amdxdna'} ({kind})"


def _power() -> str:
    profiles = sorted({_read(p) for p in glob.glob("/sys/class/platform-profile/*/profile")} - {""})
    legacy = _read("/sys/firmware/acpi/platform_profile")
    ac = [_read(p) for p in glob.glob("/sys/class/power_supply/*/online")]
    adapter = "plugged in" if "1" in ac else "on battery" if ac else "adapter unknown"
    return f"platform profile {legacy or '-'} ({', '.join(profiles) or '-'}), {adapter}"


def system() -> list[str]:
    """Lines describing the machine, OS, driver, firmware and power settings (Linux; elsewhere what the OS tells)."""
    dmi = " ".join(x for x in (_read("/sys/class/dmi/id/sys_vendor"), _read("/sys/class/dmi/id/product_name"),
                               _read("/sys/class/dmi/id/product_version")) if x)
    cpu_name = _read("/proc/cpuinfo", "model name") or platform.processor()
    mem = _read("/proc/meminfo", "MemTotal").removesuffix(" kB")
    machine = f"machine: {dmi or platform.node()}; {cpu_name}, {os.cpu_count()} CPUs" + (
        f"; {int(mem) / (1 << 20):.1f} GB of memory" if mem.isdigit() else "")
    if platform.system() != "Linux":
        return [machine, f"OS: {platform.system()} {platform.version()} (driver, firmware, power mode: xrt-smi examine)"]
    dev = artlite.Device()
    # (a heap BO more: refused by a driver that gives a process one, silently on Linux 7.0 / 7.1)
    heap = "grows" if dev.grow_heap() else "one 64 MB heap BO per process" if dev.heap_fixed else "at its maximum"
    fw = _read("/sys/class/accel/accel0/device/fw_version") or "?"
    try:
        code = dev.power_mode()
    except (NotImplementedError, artlite.Error):
        code = -1
    mode = POWER_MODES[code] if 0 <= code < len(POWER_MODES) else "?"
    cpu = irq_cpu()
    cpu_mhz = _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq") if cpu >= 0 else ""
    irq = f"CPU {cpu}" + (f" (max {int(cpu_mhz) // 1000} MHz)" if cpu_mhz else "") if cpu >= 0 else "a CPU not known yet"
    return [machine, f"OS: Linux {platform.release()}; {_driver(dev)}, firmware {fw}", f"power: {_power()}",
            f"NPU{' (through XRT)' if backend.XRT else ''}: power mode {mode}; interrupts on {irq}; device heap: {heap}"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="artlite-probe", description=__doc__.split("\n\n")[0])
    p.add_argument("--seconds", type=float, default=1.0, help="seconds of dispatches per measurement (default: 1)")
    args = p.parse_args(argv)
    lines = system() + ([f"warning: {stale}"] if (stale := stale_toolchain()) else [])
    clock, loaded, dma = ClockProbeOperator(), ClockProbeOperator(load=True), DmaProbeOperator()
    compile_all([clock, loaded, dma])
    mhz, round_trip = array_clock(clock, seconds=args.seconds)
    mhz_load, _ = array_clock(loaded, seconds=2 * args.seconds)
    rates = ddr_rates(dma, seconds=args.seconds)
    rows = [(CLOCK, f"{mhz:.0f}"), (CLOCK_LOAD, f"{mhz_load:.0f}"), (ROUND_TRIP, f"{round_trip:.0f}"),
            *((name, f"{gbps:.1f}") for name, gbps in rates.items()),
            (PER_CYCLE, f"{rates[CASES[2][0]] * 1e3 / mhz:.1f}")]
    print("artlite-probe")
    for line in lines:
        print(line)
    print()
    print("| | this machine | " + " | ".join(m for m, _ in REFERENCES) + " |")
    print("|---|---:|" + "---:|" * len(REFERENCES))
    for name, value in rows:
        print(f"| {name} | {value} | " + " | ".join(ref[name] for _, ref in REFERENCES) + " |")
    commit, number = _git("rev-parse", "--short=9", "HEAD") or "unknown", _git("rev-list", "--count", "HEAD") or "0"
    print(f"\nbuild: {commit} ({number})")


if __name__ == "__main__":
    main()
