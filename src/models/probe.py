"""``artlite-probe``: the machine and its NPU's hardware numbers in one table, next to the Strix Halo's that Artlite is
measured on: the array clock as the cores count it, the dispatch round trip, the DDR read and write rates of the shim
DMA channels (one, one shim, all 16), and small-row gathers (``artlite.operators.probe``). About a minute, no model
file. A report on another machine pastes its output: model numbers alone cannot tell a slower clock from a narrower
memory path."""

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

MB = 1 << 20
CHANNEL_BYTES = 32 * MB  # (per channel and task execution: a region of its own)
POWER_MODES = ("default", "low", "medium", "high", "turbo")


def all_channels(cols: range, chans: tuple[int, ...] = (0, 1)) -> list[tuple[int, int]]:
    return [(c, k) for c in cols for k in chans]


# (name, reads, writes, rows): the DDR cases; Strix Halo's rates next to them in REFERENCE
CASES: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]], Rows | None]] = [
    ("DDR read, 1 channel (GB/s)", [(0, 0)], [], None),
    ("DDR read, 1 shim: 2 channels", all_channels(range(1)), [], None),
    ("DDR read, 8 shims: 16 channels", all_channels(range(8)), [], None),
    ("DDR write, 16 channels", [], all_channels(range(8)), None),
    ("DDR read 8 + write 8 channels", all_channels(range(8), (0,)), all_channels(range(8), (0,)), None),
    ("gather 256 B rows 64 KB apart, 16 channels", all_channels(range(8)), [], Rows(256, 256, 64 << 10)),
    ("gather 512 B rows 64 KB apart", all_channels(range(8)), [], Rows(512, 256, 64 << 10)),
    ("gather 1 KB rows 64 KB apart", all_channels(range(8)), [], Rows(1024, 256, 64 << 10)),
    ("gather 256 B rows 64 KB + 256 B apart", all_channels(range(8)), [], Rows(256, 256, (64 << 10) + 256)),
]
CLOCK = "array clock, measured (MHz)"
PER_CYCLE = "16-channel read per array cycle (B)"
ROUND_TRIP = "dispatch round trip (us)"
# Strix Halo (Ryzen AI Max+ 395, LPDDR5X-8000 256-bit; xdna-driver, NPU power mode default)
REFERENCE = {CLOCK: "1810", ROUND_TRIP: "49", PER_CYCLE: "31.4",
             CASES[0][0]: "13.9", CASES[1][0]: "26.9", CASES[2][0]: "56.8", CASES[3][0]: "57.6", CASES[4][0]: "76.1",
             CASES[5][0]: "15.6", CASES[6][0]: "30.8", CASES[7][0]: "56.6", CASES[8][0]: "56.9"}


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
    """(MHz, dispatch round trip in us): one core spinning for n cycles of its counter, timed at two n."""
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
        words, total = dma_transaction(reads=reads, writes=writes, src=src, dst=dst, nbytes=CHANNEL_BYTES,
                                       repeat=2, rows=rows)
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
    """Lines describing the machine, OS, driver, firmware and power settings."""
    dmi = " ".join(x for x in (_read("/sys/class/dmi/id/sys_vendor"), _read("/sys/class/dmi/id/product_name"),
                               _read("/sys/class/dmi/id/product_version")) if x)
    cpu_name = _read("/proc/cpuinfo", "model name") or platform.processor()
    mem = _read("/proc/meminfo", "MemTotal").removesuffix(" kB")
    machine = f"machine: {dmi or platform.node()}; {cpu_name}, {os.cpu_count()} CPUs; " + (
        f"{int(mem) / (1 << 20):.1f} GB of memory" if mem.isdigit() else "memory ?")
    dev = artlite.Device()
    driver = _driver(dev) if platform.system() == "Linux" else ""
    fw = _read("/sys/class/accel/accel0/device/fw_version") or "?"
    lines = [machine, f"OS: {platform.system()} {platform.release()}; {driver}, firmware {fw}", f"power: {_power()}"]
    try:
        code = dev.power_mode()
    except (NotImplementedError, artlite.Error):
        code = -1
    mode = POWER_MODES[code] if 0 <= code < len(POWER_MODES) else "?"
    cpu = irq_cpu()
    cpu_mhz = _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq") if cpu >= 0 else ""
    # (a heap BO more: refused by a driver that gives a process one, silently on Linux 7.0 / 7.1)
    heap = "grows" if dev.grow_heap() else "one 64 MB heap BO per process" if dev.heap_fixed else "at its maximum"
    lines.append(f"NPU{' (through XRT)' if backend.XRT else ''}: power mode {mode}; interrupts on CPU {cpu if cpu >= 0 else '?'}"
                 f"{f' (max {int(cpu_mhz) // 1000} MHz)' if cpu_mhz else ''}; device heap: {heap}")
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="artlite-probe", description=__doc__.split("\n\n")[0])
    p.add_argument("--seconds", type=float, default=1.0, help="seconds of dispatches per measurement (default: 1)")
    args = p.parse_args(argv)
    lines = system()
    clock, dma = ClockProbeOperator(), DmaProbeOperator()
    compile_all([clock, dma])
    mhz, round_trip = array_clock(clock, seconds=args.seconds)
    rates = ddr_rates(dma, seconds=args.seconds)
    rows = [(CLOCK, f"{mhz:.0f}"), (ROUND_TRIP, f"{round_trip:.0f}"),
            *((name, f"{gbps:.1f}") for name, gbps in rates.items()),
            (PER_CYCLE, f"{rates[CASES[2][0]] * 1e3 / mhz:.1f}")]
    print("artlite-probe")
    for line in lines:
        print(line)
    print()
    print("| | this machine | Strix Halo |")
    print("|---|---:|---:|")
    for name, value in rows:
        print(f"| {name} | {value} | {REFERENCE[name]} |")
    commit, number = _git("rev-parse", "--short=9", "HEAD") or "unknown", _git("rev-list", "--count", "HEAD") or "0"
    print(f"\nbuild: {commit} ({number})")


if __name__ == "__main__":
    main()
