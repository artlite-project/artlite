"""The NPU's hardware numbers (``artlite-probe``): the array clock and the shim DMA's DDR rates, on two small designs.

``ClockProbeOperator``: one core spins until its timer has advanced by the count in its input, and writes the cycles it
counted; the host times two counts, whose difference cancels the dispatch: the array (H-) clock as the cores
see it, not the driver's table. ``load=True``: all 32 cores run MMULs meanwhile (the clock under load: a power or
thermal limit lowers it there and not for one idle-looking core).

``DmaProbeOperator``: a design with no cores. Every column's two shim MM2S channels stream DDR into its memtile's
S2MM channels 0 / 1, whose BDs loop on a 64 KB buffer with no locks (a sink that never back-pressures); the memtile's
MM2S channels loop the same way on two source buffers into the shim's S2MM channels (a source that is always ready:
writes). The host builds each case's transaction (``dma_transaction``): which shim channels read or write how many
bytes, as rows of a size at a stride. All channels share one port to the SoC's fabric, 32 B per array clock cycle
(the shim channels: 8 B per cycle each, a shim's two 16 B); rows of 256 B or less at a stride of a multiple of 4 KB
all fall on one DRAM channel (the interleave below 4 KB), larger rows on several.
"""

from typing import NamedTuple

import numpy as np
import torch
from aie.dialects.aiex import npu_maskwrite32
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import NPU2, Tile

from . import txn
from .operators.base import JINJA_ENV, ArgSpec, ArtcOperator
from .operators.design import COLS
from .operators.kernel_source import kernel_prefix

P = kernel_prefix("probe")

_SPIN = JINJA_ENV.from_string(r"""
#include <aie2pintrin.h>

// the core's timer (Timer_Low / Timer_High, AM029) through the processor bus
static inline unsigned long long {{ p }}_cycles()
{
    unsigned hi = read_tm(0x340FC);
    __builtin_aie2p_sched_barrier();
    unsigned lo = read_tm(0x340F8);
    __builtin_aie2p_sched_barrier();
    unsigned hi2 = read_tm(0x340FC);
    __builtin_aie2p_sched_barrier();
    if (hi2 != hi)
        lo = read_tm(0x340F8);
    return ((unsigned long long)hi2 << 32) | lo;
}

extern "C" void {{ p }}_spin(const int32_t *__restrict n, int32_t *__restrict out)
{
    const unsigned long long t0 = {{ p }}_cycles();
    const unsigned long long want = (unsigned)n[0];
    unsigned long long t = t0;
    while (t - t0 < want)
        t = {{ p }}_cycles();
    out[0] = (int32_t)(t - t0);
}

// 64 MMULs: 4 accumulators over 2 x 2 panels loaded from data ({{ LOAD_BYTES }} B of random mantissas)
__attribute__((noinline)) void {{ p }}_mmuls(const int8_t *__restrict data, v64accfloat *__restrict acc)
{
    v2int32 ev;
    ev[0] = ev[1] = 0x70707070;
    const v8int8 e = (v8int8)ev;
    v64accfloat a0 = acc[0], a1 = acc[1], a2 = acc[2], a3 = acc[3];
    const int8_t *p = data;
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int i = 0; i < 16; i++) {
        v64bfp16ebs8 x0, x1, w0, w1;
        x0.mantissa = *(const v64int8 __aie_dm_resource_a *)p;
        x1.mantissa = *(const v64int8 __aie_dm_resource_b *)(p + 64);
        w0.mantissa = *(const v64int8 __aie_dm_resource_a *)(p + 128);
        w1.mantissa = *(const v64int8 __aie_dm_resource_b *)(p + 192);
        x0.exponent = x1.exponent = w0.exponent = w1.exponent = e;
        a0 = mac_8x8_8x8T(x0, 1, w0, 1, a0);
        a1 = mac_8x8_8x8T(x0, 1, w1, 1, a1);
        a2 = mac_8x8_8x8T(x1, 1, w0, 1, a2);
        a3 = mac_8x8_8x8T(x1, 1, w1, 1, a3);
        p += 256;
    }
    acc[0] = a0;
    acc[1] = a1;
    acc[2] = a2;
    acc[3] = a3;
}

// the spin with MMULs: blocks of 64 until the core's counter has advanced n[0] cycles (the sums after data's panels)
extern "C" void {{ p }}_load(const int32_t *__restrict n, int8_t *__restrict data, int32_t *__restrict out)
{
    const unsigned long long t0 = {{ p }}_cycles();
    const unsigned long long want = (unsigned)n[0];
    v64accfloat *acc = (v64accfloat *)(data + {{ LOAD_BYTES }});
    unsigned long long t = t0;
    while (t - t0 < want) {
        {{ p }}_mmuls(data, acc);
        t = {{ p }}_cycles();
    }
    out[0] = (int32_t)(t - t0);
}
""")
LOAD_BYTES = 4096  # (16 x 256 B of panels; the 4 accumulators' 1 KB after them)

CORE_PROCESSOR_BUS = 0x32038  # (bit 0: the core reaches its tile's registers with read_tm / write_tm)


class ClockProbeOperator(ArtcOperator):
    """One core (column 0, row 2) spinning for ``n[0]`` cycles of its cycle counter; ``out[0]``: the cycles it
    counted. ``load``: every core of the array (4 rows x 8 columns, ``n`` broadcast in each column) runs MMULs on random
    data meanwhile, until its own counter has advanced as far: the clock under the power a prefill's GEMMs draw."""

    WORDS = 16

    def __init__(self, *, load: bool = False):
        self.load = load
        super().__init__()

    def get_kernel_src(self) -> str:
        return _SPIN.render(p=P, LOAD_BYTES=LOAD_BYTES)

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (self.WORDS,), torch.int32), ArgSpec("out", (self.WORDS,), torch.int32)]

    def get_mlir_artifact(self) -> object:
        t = np.ndarray[(self.WORDS,), np.dtype[np.int32]]
        cols, rows = (range(COLS), range(2, 6)) if self.load else (range(1), range(2, 3))
        n_in = [ObjectFifo(t, name=f"n{c}", depth=1) for c in cols]
        n_out = ObjectFifo(t, name="c", depth=1)
        d_ty = np.ndarray[(LOAD_BYTES + 1024,), np.dtype[np.int8]]
        spin = Kernel(f"{P}_spin", "kernel.o", [t, t])
        load = Kernel(f"{P}_load", "kernel.o", [t, d_ty, t])

        def core(i: object, o: object, k: object) -> None:
            k(i.acquire(1), o.acquire(1))
            i.release(1)
            o.release(1)

        def busy(i: object, o: object, k: object, d: object) -> None:  # (o: the counted cycles' object, or scratch)
            k(i.acquire(1), d, o.acquire(1) if not isinstance(o, Buffer) else o)
            i.release(1)
            if not isinstance(o, Buffer):
                o.release(1)

        rng = np.random.default_rng(0)
        workers = []
        for c in cols:
            for r in rows:
                if not self.load:
                    workers.append(Worker(core, [n_in[c].cons(), n_out.prod(), spin], tile=Tile(c, r)))
                    continue
                o = n_out.prod() if (c, r) == (0, 2) else Buffer(t, name=f"o{c}_{r}")
                data = Buffer(d_ty, name=f"d{c}_{r}", initial_value=np.concatenate(
                    [rng.integers(-128, 128, LOAD_BYTES, np.int8), np.zeros(1024, np.int8)]))
                workers.append(Worker(busy, [n_in[c].cons(), o, load, data], tile=Tile(c, r)))
        fills, drain = [f.prod() for f in n_in], n_out.cons()

        def seq(a: object, b: object, handles: object) -> None:
            for c in cols:
                for r in rows:
                    npu_maskwrite32(CORE_PROCESSOR_BUS, 1, 1, column=c, row=r)
            tg = TaskGroup()
            for f in fills:
                f.fill(a, group=tg)
            drain.drain(b, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [t, t, [*fills, drain]])
        return Program(NPU2(), rt, workers=workers).resolve_program()


SINK_WORDS = 16384  # (64 KB per memtile buffer)


class DmaProbeOperator(ArtcOperator):
    """The core-free DMA design (the module docstring); its runtime sequence is never dispatched: the cases run as
    host-built transactions (``OverlayDevice``)."""

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec("in", (1024,), torch.int32), ArgSpec("out", (1024,), torch.int32)]

    def get_mlir_artifact(self) -> str:
        mt = f"memref<{SINK_WORDS}xi32>"
        lines = ["module {", "  aie.device(npu2) {"]
        for c in range(COLS):
            lines += [f"    %s{c} = aie.tile({c}, 0)", f"    %m{c} = aie.tile({c}, 1)"]
        for c in range(COLS):
            for k in range(2):
                lines.append(f"    aie.flow(%s{c}, DMA : {k}, %m{c}, DMA : {k})")
                lines.append(f"    aie.flow(%m{c}, DMA : {k}, %s{c}, DMA : {k})")
        for c in range(COLS):
            for nm in ("snk0", "snk1", "src0", "src1"):
                lines.append(f"    %{nm}_{c} = aie.buffer(%m{c}) {{sym_name = \"{nm}_{c}\"}} : {mt}")
            lines.append(f"    %md{c} = aie.memtile_dma(%m{c}) {{")
            chans = [("S2MM", 0, "snk0"), ("S2MM", 1, "snk1"), ("MM2S", 0, "src0"), ("MM2S", 1, "src1")]
            for j, (d, k, nm) in enumerate(chans):  # (each BD its own successor, no locks: free-running)
                nxt = f"^s{j + 1}" if j + 1 < len(chans) else "^end"
                lines += [f"      %d{j} = aie.dma_start({d}, {k}, ^b{j}, {nxt})", f"    ^b{j}:",
                          f"      aie.dma_bd(%{nm}_{c} : {mt})", f"      aie.next_bd ^b{j}", f"    {nxt}:"]
            lines += ["      aie.end", "    }"]
        for c in range(COLS):
            lines += [f"    aie.shim_dma_allocation @r0_{c} (%s{c}, MM2S, 0)",
                      f"    aie.shim_dma_allocation @r1_{c} (%s{c}, MM2S, 1)",
                      f"    aie.shim_dma_allocation @w0_{c} (%s{c}, S2MM, 0)",
                      f"    aie.shim_dma_allocation @w1_{c} (%s{c}, S2MM, 1)"]
        lines.append("    aie.runtime_sequence(%in: memref<1024xi32>, %out: memref<1024xi32>) {")
        for c in range(COLS):  # (one small task per channel: the shim allocations' users)
            for k, (nm, arg) in enumerate([("r0", "%in"), ("r1", "%in"), ("w0", "%out"), ("w1", "%out")]):
                tok = ", issue_token = true" if nm[0] == "w" else ""
                lines.append(f"      aiex.npu.dma_memcpy_nd ({arg}[0, 0, 0, {128 * k}][1, 1, 1, 64][0, 0, 0, 1]) "
                             f"{{id = {k} : i64, metadata = @{nm}_{c}{tok}}} : memref<1024xi32>")
        for c in range(COLS):
            for nm in ("w0", "w1"):
                lines.append(f"      aiex.npu.dma_wait {{symbol = @{nm}_{c}}}")
        lines += ["    }", "  }", "}"]
        return "\n".join(lines) + "\n"

    def check_endpoints(self, insts: np.ndarray) -> None:
        """The compiled sequence uses every column's four shim channels (``OverlayDevice``'s check)."""
        queue = {reg: (d, ch) for d in (txn.S2MM, txn.MM2S) for ch, reg in enumerate(txn.QUEUE[d])}
        got = {(f["addr"] >> 25, *queue[f["addr"] & 0xFFFFF]) for name, f in txn.decode(insts)
               if name == "write32" and (f["addr"] & 0xFFFFF) in queue}
        want = {(c, d, k) for c in range(COLS) for d in (txn.S2MM, txn.MM2S) for k in range(2)}
        assert got == want, f"shim channels differ: {sorted(got ^ want)}"


class Rows(NamedTuple):
    """A channel's transfer as rows of ``nbytes`` at a stride of ``stride`` bytes, in groups of ``n`` rows (each group
    one row past the previous one's start)."""

    nbytes: int
    n: int
    stride: int


def dma_transaction(*, reads: list[tuple[int, int]], writes: list[tuple[int, int]], src: int, dst: int,
                    nbytes: int, repeat: int, rows: Rows | None = None) -> tuple[np.ndarray, int]:
    """A transaction in which each (column, channel) of ``reads`` (MM2S) and ``writes`` (S2MM) moves ``nbytes`` from
    / to its own ``nbytes`` region of the buffers at device addresses ``src`` / ``dst``, ``repeat`` times (``rows``:
    as rows at a stride; the region then spans n x stride). Returns the words and the bytes moved."""
    t = txn.Txn()
    total = 0
    tasks = [(c, k, txn.MM2S, src) for c, k in reads] + [(c, k, txn.S2MM, dst) for c, k in writes]
    for i, (c, k, d, base) in enumerate(tasks):
        dims = None
        if rows is not None:  # (dim 0: a row's words, dim 1: n rows at the stride, dim 2: the next group a row on)
            w = rows.nbytes // 4
            assert rows.nbytes % 4 == 0 and rows.stride % 4 == 0 and nbytes // (rows.n * rows.nbytes) * w <= rows.stride
            dims = (w, rows.n, rows.stride // 4, w)
        t.token_controller(c, d, k)
        bd = 2 * k + (0 if d == txn.MM2S else 8)
        t.shim_bd(c, bd, address=base + i * nbytes, nbytes=nbytes, dims=dims)
        t.shim_push(c, d, k, bd, token=True, repeat=repeat)
        total += nbytes * repeat
    for c, k, d, _ in tasks:
        t.sync(col=c, direction=d, channel=k)
    return t.words(), total
