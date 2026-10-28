"""The diagnosis' NPU rows: ``artlite-probe``'s DMA table on host buffers of 2 MB pages and again of 4 KB pages, rows
of channel placement and of the prefill's own row shapes, and the control path (what the NPU's command processor,
not its array, spends on a transaction's operations, on small DMA tasks and on dispatches)."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator

import numpy as np

from artlite import _native as native
from artlite import txn
from artlite.overlay import OverlayDevice, Transaction
from artlite.probe import ClockProbeOperator, DmaProbeOperator, Rows, dma_transaction

from ..probe import CASES, CHANNEL_BYTES, all_channels, array_clock, timed

MB = 1 << 20
DEFAULT_USER_BO = 2 * MB  # (Device.alloc_host_buffer: user BOs from this size on)


@contextlib.contextmanager
def host_pages(kind: str) -> Iterator[None]:
    """Host buffers allocated meanwhile have 2 MB pages ("2 MB": user BOs) or 4 KB pages ("4 KB": the driver's BOs)."""
    native.set_user_bo_bytes(DEFAULT_USER_BO if kind == "2 MB" else 1 << 62)
    try:
        yield
    finally:
        native.set_user_bo_bytes(DEFAULT_USER_BO)


# (name, reads, writes, rows, region): rows of channel placement and of the prefill's shapes, on top of CASES; region
# 0: CHANNEL_BYTES apart (each channel's region starts at the same offset in its 2 MB pages)
STAGGER = CHANNEL_BYTES + (68 << 10)  # (regions at other offsets in their 2 MB pages: other DRAM bank bits)
EXTRA: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]], Rows | None, int]] = [
    ("DDR read, 16 channels, regions at other 2 MB-page offsets", all_channels(range(8)), [], None, STAGGER),
    ("DDR write, 16 channels, regions at other 2 MB-page offsets", [], all_channels(range(8)), None, STAGGER),
    ("DDR read 8 + write 8, regions at other 2 MB-page offsets", all_channels(range(8), (0,)),
     all_channels(range(8), (0,)), None, STAGGER),
    ("write 576 B rows 64 KB + 576 B apart (whole lines)", [], all_channels(range(8)), Rows(576, 64, (64 << 10) + 576),
     0),
    ("write 132 B rows, regions at other 2 MB-page offsets", [], all_channels(range(8)),
     Rows(132, 256, (64 << 10) + 132), STAGGER),
    ("read 2 KB rows 9280 B apart (Q4_K weight slices)", all_channels(range(8)), [], Rows(2048, 256, 9280), 0),
]


ROWS = [(name, r, w, rows, 0) for name, r, w, rows in CASES] + EXTRA
BUFFER = 16 * CHANNEL_BYTES + 16 * (68 << 10)  # (each of the rows' src and dst: 16 regions at the largest spacing)


def row_transaction(row: tuple, src: int, dst: int) -> tuple[np.ndarray, int]:
    """A row of ROWS as a transaction on buffers of BUFFER bytes at device addresses src / dst (``dma_transaction``),
    its channels' rows inside them; the words and the bytes moved."""
    name, reads, writes, rows, region = row
    group = 4 if rows is None else rows.n * rows.nbytes
    nbytes = CHANNEL_BYTES // group * group
    span = nbytes if rows is None else rows.span(nbytes)
    assert (len(reads) + len(writes) - 1) * (region or nbytes) + span <= BUFFER, name
    return dma_transaction(reads=reads, writes=writes, src=src, dst=dst, nbytes=nbytes, repeat=2, rows=rows,
                           region=region)


def dma_rates(op: DmaProbeOperator, *, seconds: float, pages: str) -> dict[str, float]:
    """GB/s of ROWS on buffers of ``pages`` pages (bytes over the median dispatch)."""
    with host_pages(pages):
        dev = OverlayDevice(op)
        dev.alloc("src", BUFFER)
        dev.alloc("dst", BUFFER)
    src, dst = dev.address("src"), dev.address("dst")
    out = {}
    for i, row in enumerate([ROWS[2], *ROWS]):  # (first: the DPM ramp's warm-up)
        words, total = row_transaction(row, src, dst)
        tr = Transaction(dev, words)
        t = timed(lambda tr=tr: dev.dispatch(tr), seconds=seconds)
        if i:
            out[row[0]] = total / t / 1e9
    return out


SCRATCH = 0x7E000  # (memory tile 7's last 8 KB: no buffer of DmaProbeOperator's design there)


def _control_words(kind: str, n: int, src: int) -> np.ndarray:
    t = txn.Txn()
    if kind == "noop":
        for _ in range(n):
            t._add([txn.OP_NOOP])
    elif kind == "write32":
        for i in range(n):
            t.write32(txn.tile_address(7, 1, SCRATCH + 4 * (i % 1024)), i)
    elif kind == "blockwrite 1 KB":
        for i in range(n):
            t.blockwrite(txn.tile_address(7, 1, SCRATCH + 1024 * (i % 4)), list(range(256)))
    elif kind == "64 B DMA task + wait":
        t.token_controller(0, txn.MM2S, 0)
        for i in range(n):
            t.shim_task(0, txn.MM2S, 0, 0, address=src + 64 * (i % 1024), nbytes=64, token=True)
            t.sync(col=0, direction=txn.MM2S, channel=0)
    elif kind == "64 B DMA tasks of 8 columns + one wait":
        for c in range(8):
            t.token_controller(c, txn.MM2S, 0)
        for i in range(n // 8):
            for c in range(8):
                t.shim_task(c, txn.MM2S, 0, 0, address=src + 64 * c + 4096 * (i % 256), nbytes=64, token=True)
            t.sync(col=0, direction=txn.MM2S, channel=0, ncol=8)
    return t.words()


CONTROL = ("noop", "write32", "blockwrite 1 KB", "64 B DMA task + wait", "64 B DMA tasks of 8 columns + one wait")


def control_rates(op: DmaProbeOperator, clock: ClockProbeOperator, *, seconds: float) -> dict[str, float]:
    """The command processor's cost per operation (ns: the difference of 4096 and 64 operations, the dispatch taken
    out) and per byte of control code; the dispatch round trip, NOP dispatches back to back blocking and with 1 or 3
    queued (the rate of ``xrt-smi validate``'s throughput test)."""
    dev = OverlayDevice(op)
    dev.alloc("src", 4 * MB)
    dev.alloc("dst", 4 * MB)
    src = dev.address("src")
    out: dict[str, float] = {}
    for kind in CONTROL:
        res = {}
        for n in (64, 4096):
            words = _control_words(kind, n, src)
            tr = Transaction(dev, words)
            res[n] = (timed(lambda tr=tr: dev.dispatch(tr), seconds=seconds / 2), words.nbytes)
        dt = res[4096][0] - res[64][0]
        out[f"{kind} (ns per op)"] = dt / (4096 - 64) * 1e9
        if kind in ("noop", "write32", "blockwrite 1 KB"):
            out[f"{kind} (ns per byte)"] = dt / (res[4096][1] - res[64][1]) * 1e9
    tr = Transaction(dev, _control_words("noop", 1, src))
    out["NOP dispatch, blocking (us)"] = timed(lambda: dev.dispatch(tr), seconds=seconds) * 1e6
    kern, bufs = dev._prog._kernel(), list(dev.buffers.values())
    for depth in (1, 3):
        n = max(200, int(seconds * 40000))
        t0 = time.perf_counter()
        for _ in range(n):
            kern.dispatch_async(tr.buffer, bufs)
            kern.dispatch_wait(keep=depth)
        kern.dispatch_wait(keep=0)
        out[f"NOP dispatches, {depth} queued (per s)"] = n / (time.perf_counter() - t0)
    mhz, round_trip = array_clock(clock, seconds=seconds)
    out["array clock (MHz)"], out["dispatch round trip (us)"] = mhz, round_trip
    return out
