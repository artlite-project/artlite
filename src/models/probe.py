"""``artlite-probe``: the machine and its NPU's hardware numbers in one table, next to those of the Strix Halo that
Artlite is measured on and of a Krackan Point and a Strix Point laptop (issue #5): the array clock as the cores count
it, the dispatch round trip, the DDR read and write rates of the shim DMA channels (one, one shim, all 16), and
small-row gathers (``artlite.probe``). About a minute, no model file. A report on another machine pastes its
output: model numbers alone cannot tell a slower clock from a narrower memory path.

``--pages``: a second table instead, of what the NPU's address translation costs rows on pages of their own
(``PAGE_CASES``): how many pages its translation cache holds, for reads and writes, per channel or for all; the
rate of translations with one to 16 channels; page-table locality; rows that share a page; long rows."""

import argparse
import glob
import os
import platform
import re
import statistics
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch

import artlite
from artlite import backend, portable
from artlite.overlay import OverlayDevice, Transaction, irq_cpu
from artlite.parallel_compile import compile_all
from artlite.probe import ClockProbeOperator, DmaProbeOperator, Rows, dma_transaction

from .bench import _git, _read
from .common import stale_install, windows_power_mode, xrt_smi

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
                                      "55.9", "95.6", "89.6", "73.2", "15.6", "54.1"), strict=True))),
    ("Strix Point", dict(zip(ROWS, ("1809", "1808", "94", "11.5", "21.4", "53.2", "47.3", "39.9", "53.4", "12.5",
                                    "25.1", "41.9", "14.0", "?", "?", "29.4"), strict=True))),
]

PAGE = 4096
PAGE_TARGET = 32 * MB  # (bytes a channel moves per dispatch, where its task repeats allow)


class PageCase(NamedTuple):
    """A row of ``--pages``: ``chans`` read, then write (``mixed``: read and write at once, its rows counted together),
    ``nbytes`` per task as ``rows`` (None: a stream), the channels' regions ``region`` apart, ``skew`` within their
    pages and the task's ``iteration`` (``dma_transaction``)."""

    name: str
    chans: list[tuple[int, int]]
    rows: Rows | None
    nbytes: int
    region: int = CHANNEL_BYTES
    skew: int = 256
    mixed: bool = False
    iteration: tuple[int, int] | None = None

    def tasks(self) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
        """The (reads, writes) of its measurements."""
        return [(self.chans, self.chans)] if self.mixed else [(self.chans, []), ([], self.chans)]

    @property
    def repeat(self) -> int:
        return max(1, min(1024, PAGE_TARGET // self.nbytes))


def first(n: int) -> list[tuple[int, int]]:
    """n channels, one per shim up to 8."""
    return [(i % 8, i // 8) for i in range(n)]


def cycle(n: int, chans: list[tuple[int, int]], *, stride: int = PAGE, region: int = CHANNEL_BYTES,
          name: str = "", mixed: bool = False) -> PageCase:
    """256 B rows, each on a page of its own: a channel's n pages ``stride`` apart in turn (rows ``stride`` + 256 B
    apart: a channel's rows on all DRAM channels), 16 times, each time 256 B further into them (channel i from
    (i * 256) mod 4 KB on). Up to 1023 pages the 16 turns are a task (a BD dimension's 1023 rows), from 1024 on each a
    repeat of the task: groups of 512 pages, the iteration 256 B on."""
    name = name or f"{n} pages per channel, {len(chans)} channel{'s' * (len(chans) > 1)}"
    if n < 1024:
        return PageCase(name, chans, Rows(256, n, stride + 256), 16 * n * 256, region, mixed=mixed)
    assert n % 512 == 0
    return PageCase(name, chans, Rows(256, 512, stride + 256, 512 * (stride + 256)), n * 256, region, mixed=mixed,
                    iteration=(256, 16))


def in_pages(name: str, nbytes: int, k: int, *, pitch: int = 256, pages: int = 512, stride: int = PAGE,
             skew: int = 0) -> PageCase:
    """k rows of ``nbytes`` ``pitch`` apart (rows that touch: one) in each of a channel's ``pages`` pages, ``stride``
    apart (channel i (i * ``skew``) mod 4 KB into them): no page visited twice in a task, which the next repeat runs
    again."""
    return PageCase(name, first(16), Rows(nbytes, k, pitch, stride), pages * k * nbytes, skew=skew)


# (the translation cache: rows cycling through n pages of each channel; per channel or shared: 4 channels at 4x the
# pages; the cold rate of 1-16 channels; page-table entries 16 KB, 64 KB and 1 MB apart (2 a 64-byte line, a line
# each, a page-table page each; the 1 MB channels interleaved); 1-16 rows of 128 B sharing a page never seen before
# (pages next to each other, 64 KB apart) or one seen a few rows before; rows of whole pages; the prefill's reads of
# the decode image's tiles; reads and writes at once)
PAGE_CASES: list[PageCase] = [
    PageCase("16-channel stream (GB/s)", first(16), None, CHANNEL_BYTES, skew=0),
    *(cycle(n, first(16)) for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)),
    *(cycle(n, first(4)) for n in (16, 64, 256, 1024, 4096)),
    *(cycle(4096, first(c)) for c in (1, 2, 8)),
    *(cycle(256, first(16), stride=d << 10, name=f"256 pages per channel {d} KB apart, 16 channels")
      for d in (16, 64)),
    cycle(256, first(16), stride=1 << 20, region=64 << 10, name="256 pages per channel 1 MB apart, 16 channels"),
    *(in_pages(f"128 B rows, {k} a page: 512 pages per channel, 4 KB apart", 128, k, skew=256 * k)
      for k in (1, 2, 4, 8, 16)),
    *(in_pages(f"128 B rows, {k} a page: 512 pages per channel, 64 KB apart", 128, k, stride=64 << 10, skew=256 * k)
      for k in (1, 8)),
    in_pages("128 B rows, 8 a page: 8 pages per channel, 4 KB apart", 128, 8, pages=8, skew=2048),
    in_pages("4 KB rows (a page): 512 per channel, 64 KB apart", 2048, 2, pitch=2048, stride=64 << 10),
    in_pages("16 KB rows (4 pages): 512 per channel, 64 KB apart", 2048, 8, pitch=2048, stride=64 << 10),
    PageCase("2 KB rows 9280 B apart (Q4_K weight slices): 512 per channel", first(16), Rows(2048, 512, 9280),
             512 * 2048, skew=0),
    cycle(4096, first(8), name="8 channels read + 8 write at once, 4096 pages per channel", mixed=True),
]
PAGE_ROWS = [CLOCK, *(case.name for case in PAGE_CASES)]
# (Strix Halo as in REFERENCES, transparent huge pages madvise and never)
PAGE_REFERENCES: list[tuple[str, dict[str, str]]] = [
    ("Strix Halo, 2 MB pages", dict(zip(PAGE_ROWS, (
        "1808", "56.5 / 57.3", "221 / 222", "222 / 224", "221 / 224", "221 / 224", "221 / 224", "221 / 224",
        "221 / 224", "221 / 224", "221 / 224", "221 / 224", "199 / 218", "198 / 219", "199 / 218", "199 / 219",
        "200 / 218", "53.1 / 54.5", "104 / 109", "219 / 222", "221 / 224", "222 / 224", "221 / 224", "448 / 447",
        "447 / 446", "446 / 445", "446 / 445", "445 / 445", "444 / 445", "446 / 445", "435 / 435", "13.8 / 14.0",
        "3.5 / 3.5", "27.7 / 28.0", "270"), strict=True))),
    ("Strix Halo, 4 KB pages", dict(zip(PAGE_ROWS, (
        "1808", "56.5 / 56.9", "221 / 222", "222 / 224", "221 / 224", "221 / 224", "157 / 129", "158 / 133",
        "160 / 133", "160 / 132", "159 / 132", "160 / 132", "200 / 219", "202 / 219", "171 / 136", "122 / 121",
        "122 / 122", "47.4 / 54.7", "90.2 / 108", "143 / 132", "119 / 117", "113 / 100", "114 / 87.3", "163 / 143",
        "370 / 162", "396 / 188", "406 / 251", "421 / 309", "64.8 / 64.6", "385 / 146", "433 / 434", "13.8 / 13.4",
        "3.5 / 3.4", "27.7 / 26.2", "162"), strict=True))),
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


def page_transaction(case: PageCase, reads: list[tuple[int, int]], writes: list[tuple[int, int]], src: int,
                     dst: int) -> tuple[np.ndarray, int]:
    """One measurement of a PAGE_CASES row (``dma_transaction``), its channels' rows inside the probe's buffers."""
    if case.rows is not None:
        shift = 0 if case.iteration is None else (case.iteration[1] - 1) * case.iteration[0]
        ends = (i * case.region + i * case.skew % PAGE + case.rows.span(case.nbytes) + shift
                for i in range(len(reads) + len(writes)))
        assert max(ends) <= 16 * CHANNEL_BYTES, case.name
    return dma_transaction(reads=reads, writes=writes, src=src, dst=dst, nbytes=case.nbytes, repeat=case.repeat,
                           rows=case.rows, region=case.region, skew=case.skew, iteration=case.iteration)


def page_rates(op: DmaProbeOperator, *, seconds: float) -> dict[str, list[float]]:
    """Each of PAGE_CASES' rates, read and write (or both at once): millions of rows a second (a stream: GB/s) over
    the median dispatch."""
    dev = OverlayDevice(op)
    n = 16 * CHANNEL_BYTES
    dev.alloc("src", n)
    dev.alloc("dst", n)
    src, dst = dev.address("src"), dev.address("dst")
    out = {}
    for i, case in enumerate([PAGE_CASES[0], *PAGE_CASES]):  # (first: the DPM ramp's warm-up)
        rates = []
        for reads, writes in case.tasks():
            words, total = page_transaction(case, reads, writes, src, dst)
            tr = Transaction(dev, words)
            t = timed(lambda tr=tr: dev.dispatch(tr), seconds=seconds)
            rates.append(total / t / 1e9 if case.rows is None else total / case.rows.run / t / 1e6)
        if i:
            out[case.name] = rates
    return out


def _rate(x: float) -> str:
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}"


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


def host_pages() -> str:
    """The page size of the host buffers the NPU reads and writes (a DMA row on a page of its own costs a translation:
    with 4 KB pages 128-byte rows write at a fifth of the rate, ``docs/probe.md``)."""
    if platform.system() == "Windows":
        return "4 KB pages (Windows maps system memory for the NPU in 4 KB pages)"
    thp = _read("/sys/kernel/mm/transparent_hugepage/enabled")
    if "[never]" in thp:
        return "4 KB pages (transparent huge pages are off: /sys/kernel/mm/transparent_hugepage/enabled)"
    return "2 MB pages" if thp else "? pages"


def _cpu_name() -> str:
    if portable.WINDOWS:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return _read("/proc/cpuinfo", "model name") or platform.processor()


def system() -> list[str]:
    """Lines describing the machine, OS, driver, firmware and power settings (Linux; on Windows what the OS and
    ``xrt-smi`` tell)."""
    dmi = " ".join(x for x in (_read("/sys/class/dmi/id/sys_vendor"), _read("/sys/class/dmi/id/product_name"),
                               _read("/sys/class/dmi/id/product_version")) if x)
    mem = _read("/proc/meminfo", "MemTotal").removesuffix(" kB")
    machine = f"machine: {dmi or platform.node()}; {_cpu_name()}, {os.cpu_count()} CPUs" + (
        f"; {int(mem) / (1 << 20):.1f} GB of memory" if mem.isdigit() else "")
    if platform.system() != "Linux":
        smi = xrt_smi("examine") or ""
        drv, fw = (m[1] if (m := re.search(rf"NPU {name} Version\s*:\s*(\S+)", smi)) else "?"
                   for name in ("Driver", "Firmware"))
        return [machine, f"OS: {platform.system()} {platform.version()}; NPU driver {drv}, firmware {fw}",
                f"NPU: power mode {windows_power_mode() or '?'}; host buffers: {host_pages()}"]
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
            (f"NPU{' (through XRT)' if backend.XRT else ''}: power mode {mode}; interrupts on {irq}; device heap: {heap}; "
             f"host buffers: {host_pages()}")]


def pages(seconds: float) -> tuple[list[str], list[tuple[str, str]]]:
    """``--pages``: (lines before the table, its rows). On Windows the NPU runs them in its ``performance`` power mode
    (from ``default``, which is lower on some machines; put back after), as ``artlite-cli`` does."""
    clock, dma = ClockProbeOperator(), DmaProbeOperator()
    compile_all([clock, dma])
    raised = (portable.WINDOWS and windows_power_mode() == "default"
              and xrt_smi("configure", "--pmode", "performance") is not None)
    try:
        mhz, _ = array_clock(clock, seconds=seconds)
        rates = page_rates(dma, seconds=seconds)
    finally:
        if raised:
            xrt_smi("configure", "--pmode", "default")
    lines = ["NPU power mode: default -> performance for these rows (default again after)"] if raised else []
    lines.append("M rows a second, read / write; rows of 256 B, each on a page of its own, a channel's pages 4 KB apart "
                 "(the rows 4 KB + 256 B) and visited in turn, unless the row says otherwise; rows sharing a page 256 B "
                 "apart (docs/probe.md)")
    return lines, [(CLOCK, f"{mhz:.0f}"), *((name, " / ".join(_rate(r) for r in v)) for name, v in rates.items())]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="artlite-probe", description=__doc__.split("\n\n")[0])
    p.add_argument("--seconds", type=float, default=1.0, help="seconds of dispatches per measurement (default: 1)")
    p.add_argument("--pages", action="store_true",
                   help="instead: what the NPU's address translation costs rows on pages of their own (~2 minutes)")
    args = p.parse_args(argv)
    lines = system() + ([f"warning: {stale}"] if (stale := stale_install()) else [])
    if args.pages:
        more, rows = pages(args.seconds)
        lines += more
        references = PAGE_REFERENCES
    else:
        clock, loaded, dma = ClockProbeOperator(), ClockProbeOperator(load=True), DmaProbeOperator()
        compile_all([clock, loaded, dma])
        mhz, round_trip = array_clock(clock, seconds=args.seconds)
        mhz_load, _ = array_clock(loaded, seconds=2 * args.seconds)
        rates = ddr_rates(dma, seconds=args.seconds)
        rows = [(CLOCK, f"{mhz:.0f}"), (CLOCK_LOAD, f"{mhz_load:.0f}"), (ROUND_TRIP, f"{round_trip:.0f}"),
                *((name, f"{gbps:.1f}") for name, gbps in rates.items()),
                (PER_CYCLE, f"{rates[CASES[2][0]] * 1e3 / mhz:.1f}")]
        references = REFERENCES
    print("artlite-probe" + " --pages" * args.pages)
    for line in lines:
        print(line)
    print()
    print("| | this machine | " + " | ".join(m for m, _ in references) + " |")
    print("|---|---:|" + "---:|" * len(references))
    for name, value in rows:
        print(f"| {name} | {value} | " + " | ".join(ref[name] for _, ref in references) + " |")
    commit, number = _git("rev-parse", "--short=9", "HEAD") or "unknown", _git("rev-list", "--count", "HEAD") or "0"
    print(f"\nbuild: {commit} ({number})")


if __name__ == "__main__":
    main()
