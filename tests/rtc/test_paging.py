"""Code pages (artlite.operators.paging): the linker's checks, and self-DMA paging on every core of the array.

The probe design: each core runs a resident main over a ring of ``NS`` L1 slots that its row's broadcast stream fills
(shim MM2S 1 of columns 0, 2, 4, 6 -> the row's 8 cores); a slot is a command ([kind, x]: kind 1 calls the loaded
page's entry with x) or, at ring position 0, the first of a page's ``NS`` slots, which the core's MM2S 1 streams into
its own TileControl (``paging.loop_flow``) while it sleeps in ``pg_self``. Results in a pinned buffer (``RES``): y, the
command count, the page count, ``get_coreid()``, the state's page id."""

import os
from typing import ClassVar

import numpy as np
import pytest
import torch
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern as TAP
from aie.iron import Acquire, Bd, Buffer, DmaChannel, Flow, Kernel, Lock, Program, Release, Runtime, TileDma, Worker
from aie.iron.device import NPU2, Tile

from artlite import txn
from artlite.operators import paging
from artlite.operators.base import JINJA_ENV, ArgDirection, ArgSpec, ArtcOperator
from artlite.operators.kernel_source import kernel_prefix
from artlite.operators.overlay.runtime import OverlayDevice, Transaction

ROWS = 4
B_SHIM = (0, 2, 4, 6)
NS, SW = 5, 1440  # (slots in a core's ring, words per slot: sd15's B slot)
SLOTS, ST, RES = 0x8000, 0x7F00, 0x7E00  # (L1 byte offsets: the ring, the pager's state words, the results)
LE, LF, LS, LD, LW = range(5)  # (core locks: ring slots empty / full, MM2S send / done, the pager's wake)
STACK = 1024
CHAIN = 700  # (statements of the big page's entry)
STEM = "pgt"
P = kernel_prefix(STEM)

_SRC = r"""
{{ pager }}

// the pages' entries (only the pager's state words reach them)
int32_t {{ p }}_pg_a(int32_t x) { return x * 3 + 1; }
int32_t {{ p }}_pg_b(int32_t x) {
{% if variant == "reach" %}
  {{ p }}_pg_self();
{% endif %}
{% if variant == "data" %}
  switch (x & 7) {
  case 0: return x * 5 + 7;
  case 1: return x - 3;
  case 2: return x * 9;
  case 3: return x ^ 0x55;
  case 4: return x + 11;
  case 5: return x * x;
  case 6: return x >> 2;
  default: return 1 - x;
  }
{% endif %}
  return x * 5 + 7;
}
int32_t {{ p }}_pg_big(int32_t x) {
{% for k in range(chain) %}
  x = x * {{ 3 + k % 5 }} + {{ k }};
  x ^= x >> {{ 1 + k % 7 }};
{% endfor %}
  return x;
}

// (resident) the loaded page's entry 0
__attribute__((noinline)) int32_t {{ p }}_call(const volatile int32_t *st, int32_t x) {
  return ((int32_t (*)(int32_t))(intptr_t)st[1])(x);
}

extern "C" {

void {{ p }}_main(int32_t *slots, int32_t *res, int32_t *st) {
  res[3] = get_coreid();
  int pos = 0, seq = 0, pages = 0;
  for (;;) {
    acquire_greater_equal({{ L + LF }}, 1);
    volatile int32_t *b = slots + pos * {{ SW }};
    if (b[0] == (int32_t){{ header }}u) {  // (a page: the ring's NS slots from position 0)
{% if variant == "acq" %}
      acquire_greater_equal({{ L + LW }}, 0);
{% endif %}
      acquire_greater_equal({{ L + LF }}, {{ NS - 1 }});
      event0();
      {{ p }}_pg_self();
      event1();
      acquire_greater_equal({{ L + LD }}, {{ NS }});
      release({{ L + LE }}, {{ NS }});
      res[2] = ++pages;
      continue;
    }
    const int k = b[0], x = b[1];
    release({{ L + LE }}, 1);
    pos = pos + 1 == {{ NS }} ? 0 : pos + 1;
    if (k == 0)
      continue;
    res[0] = {{ p }}_call((const volatile int32_t *)st, x);
    res[1] = ++seq;
    res[4] = st[0];
  }
}

}  // extern "C"
"""


def big(x: int, chain: int = CHAIN) -> int:
    """The big page's function on the host (int32 wrap-around, arithmetic shifts)."""
    x = np.int32(x)
    with np.errstate(over="ignore"):
        for k in range(chain):
            x = np.int32(x * np.int32(3 + k % 5) + np.int32(k))
            x = np.int32(x ^ (x >> np.int32(1 + k % 7)))
    return int(x)


class PagingProbe(ArtcOperator):
    """The probe design on ``cols`` columns (the module docstring); ``variant``: a kernel the linker must refuse
    ("reach": a page calls the pager, "data": a page with a jump table, "acq": the wake lock acquired outside the
    pager)."""

    ENDPOINTS: ClassVar[dict[str, tuple[int, int]]] = {}
    BD_BASE: ClassVar[dict[str, int]] = {}

    def __init__(self, *, cols: int = 8, variant: str = "", chain: int = CHAIN):
        self.cols, self.variant, self.chain = cols, variant, chain
        super().__init__()

    def get_kernel_src(self) -> str:
        pager = paging.source(P, wake=LW, send=LS, nslots=NS)
        return JINJA_ENV.from_string(_SRC).render(p=P, pager=pager, variant=self.variant, chain=self.chain, SW=SW, NS=NS,
                                                  L=paging.LOCK_BASE, LE=LE, LF=LF, LD=LD, LW=LW,
                                                  header=paging.header())

    def get_arg_spec(self) -> list[ArgSpec]:
        return [ArgSpec(ArgDirection.IN, (1024,), torch.int32), ArgSpec(ArgDirection.OUT, (1024,), torch.int32)]

    def check_endpoints(self, insts: np.ndarray) -> None:
        pass

    def get_mlir_artifact(self):
        i32 = np.dtype[np.int32]

        def ty(n: int) -> type[np.ndarray]:
            return np.ndarray[(n,), i32]
        k = Kernel(f"{P}_main", "kernel.o", [ty(NS * SW), ty(16), ty(16)])
        shims = [Tile(c, 0) for c in range(self.cols)]
        cores = [[Tile(c, r + 2) for r in range(ROWS)] for c in range(self.cols)]
        b = [Flow(shims[B_SHIM[r] % self.cols], [cores[c][r] for c in range(self.cols)], src_channel=1, name=f"pgt_b{r}")
             for r in range(ROWS)] if self.cols > 1 else \
            [Flow(shims[0], cores[0][r], src_channel=1 if r % 2 else 0, name=f"pgt_b{r}") for r in range(2)]
        flows = list(b)
        dmas, workers = [], []
        for c in range(self.cols):
            for r in range(ROWS if self.cols > 1 else 2):
                t = cores[c][r]
                ring = Buffer(type=ty(NS * SW), tile=t, name=f"pgt_ring{c}_{r}", address=SLOTS)
                res = Buffer(type=ty(16), tile=t, name=f"pgt_res{c}_{r}", address=RES)
                st = Buffer(type=ty(16), tile=t, name=f"pgt_st{c}_{r}", address=ST)
                le = Lock(t, lock_id=LE, init=NS, name=f"pgt_le{c}_{r}")
                lf = Lock(t, lock_id=LF, init=0, name=f"pgt_lf{c}_{r}")
                ls = Lock(t, lock_id=LS, init=0, name=f"pgt_ls{c}_{r}")
                ld = Lock(t, lock_id=LD, init=0, name=f"pgt_ld{c}_{r}")
                Lock(t, lock_id=LW, init=0, name=f"pgt_lw{c}_{r}")

                def part(j: int) -> TAP:
                    return TAP((NS * SW,), j * SW, [SW], [1])
                flows.append(paging.loop_flow(t))
                dmas.append(TileDma(t, [
                    DmaChannel(DMAChannelDir.S2MM, b[r].endpoint(t),
                               [Bd(ring, tap=part(j), acquires=[Acquire(le)], releases=[Release(lf)]) for j in range(NS)]),
                    DmaChannel(DMAChannelDir.MM2S, 1,
                               [Bd(ring, tap=part(j), acquires=[Acquire(ls)], releases=[Release(ld)]) for j in range(NS)]),
                ]))

                def fn(ri, rs, s, kk):
                    kk(ri, rs, s)
                workers.append(Worker(fn, [ring, res, st, k], tile=t, stack_size=STACK, while_true=False))
        ty_arg = np.ndarray[(1024,), i32]

        def seq(X, Y):
            for f in b:
                f.fill(X, tap=TAP((1024,), 0, [1, 1, 1, 64], [0, 0, 0, 1]))
        rt = Runtime(seq, [ty_arg, ty_arg])
        for f in flows:
            rt.add_flow(f)
        for d in dmas:
            rt.add_tile_dma(d)
        return Program(NPU2(), rt, workers=workers).resolve_program()


def _work_dir(op: ArtcOperator) -> str:
    return os.path.dirname(op._program.insts_path)


def _link(op: PagingProbe) -> paging.Pager:
    cores = [(c, r + 2) for c in range(op.cols) for r in range(ROWS if op.cols > 1 else 2)]
    return paging.link(_work_dir(op), stem=STEM, pages={"a": ("pg_a",), "b": ("pg_b",), "big": ("pg_big",)},
                       pager="pg_self", caller="call", root="main", stack=STACK, cores=cores, state=ST, wake=LW)


def _compiled(op: PagingProbe) -> PagingProbe:
    op.compile()
    return op


@pytest.fixture(scope="module")
def probe() -> PagingProbe:
    return _compiled(PagingProbe())


def test_link(probe: PagingProbe) -> None:
    """Every page at the slot (where the crt's startup code was), inside program memory, its entry first."""
    pg = _link(probe)
    assert set(pg.pages) == {"a", "b", "big"}
    for name, page in pg.pages.items():
        assert page.entries[0] >= pg.slot and pg.slot + page.nbytes <= paging.PM_BYTES, name
    assert pg.pages["big"].nbytes > 10 * 1024, pg.pages["big"].nbytes  # (the test reaches far into program memory)
    s = pg.slots("big", words=SW, n=NS)
    assert s.size == NS * SW and s[0] == paging.header()
    with pytest.raises(paging.PagingError, match="does not fit"):
        pg.slots("big", words=SW, n=2)


@pytest.mark.parametrize("variant, message", [("reach", "reaches the pager"), ("data", "needs data"),
                                              ("acq", "wake lock 4 acquired in")])
def test_link_refuses(variant: str, message: str) -> None:
    """The linker refuses a page that calls the pager, a page with data of its own, a wake lock acquired outside the
    pager (compile only: a one-column design)."""
    op = _compiled(PagingProbe(cols=1, variant=variant, chain=4))
    with pytest.raises(paging.PagingError, match=message):
        _link(op)


class _Bench:
    """The probe on the NPU: row streams of slots from one host buffer, the cores' results."""

    def __init__(self, op: PagingProbe):
        self.op, self.dev = op, OverlayDevice(op)
        self.dev.alloc("src", 8 << 20)
        self.tiles = [(c, r + 2) for c in range(op.cols) for r in range(ROWS)]

    def run(self, slots: list[np.ndarray], *, counters: bool = False) -> None:
        img = np.concatenate(slots).astype(np.uint32)
        assert img.size % (NS * SW) == 0, "a stream of whole rings (pages at ring position 0)"
        self.dev.view("src")[:img.nbytes] = img.view(np.uint8)
        self.dev.flush("src", 0, img.nbytes)
        t = txn.Txn()
        if counters:  # (counter 0: cycles from event0 to event1)
            for c, r in self.tiles:
                t.write32(txn.tile_address(c, r, 0x31500), 33 | 34 << 8)
        for col in B_SHIM:
            t.token_controller(col, txn.MM2S, 1)
            t.shim_task(col, txn.MM2S, 1, 1, address=self.dev.address("src"), nbytes=img.nbytes, token=True)
        for col in B_SHIM:
            t.sync(col=col, direction=txn.MM2S, channel=1)
        self.dev.dispatch(Transaction(self.dev, t.words()))

    def reg(self, c: int, r: int, addr: int) -> int:
        return self.op._program._ctx.read_aie_reg(c, r, addr) & 0xFFFFFFFF

    def res(self, k: int) -> list[int]:
        return [self.reg(c, r, RES + 4 * k) for c, r in self.tiles]

    def cycles(self) -> np.ndarray:
        return np.array([self.reg(c, r, 0x31520) for c, r in self.tiles], np.int64)


def _cmd(x: int, kind: int = 1) -> np.ndarray:
    s = np.zeros(SW, np.uint32)
    s[0], s[1] = kind, np.uint32(np.int32(x).view(np.uint32))
    return s


def _cmds(*xs: int) -> list[np.ndarray]:
    """Commands padded with NOPs to a whole ring."""
    out = [_cmd(x) for x in xs]
    return out + [_cmd(0, 0)] * (-len(out) % NS)


@pytest.fixture(scope="module")
def bench(probe: PagingProbe) -> _Bench:
    return _Bench(probe)


def test_self_dma(probe: PagingProbe, bench: _Bench) -> None:
    """Every core pages its own code: two pages in turn (64 times), the big page, a page written up to the last word
    of program memory; the cycles of a page (pg_self's window)."""
    pg = _link(probe)
    ring = {n: pg.slots(n, words=SW, n=NS) for n in pg.pages}
    bench.run([ring["a"], *_cmds(10)])
    assert bench.res(0) == [31] * 32 and bench.res(4) == [pg.page_id("a")] * 32
    assert bench.res(3) == [c << 16 | r for c, r in bench.tiles]  # (get_coreid)
    stream = []
    for i in range(64):
        stream += [ring["a" if i % 2 else "b"], *_cmds(100 + i)]
    pages0 = bench.res(2)
    c0 = bench.cycles()
    bench.run(stream, counters=True)
    per = float(np.median((bench.cycles() - c0) % (1 << 32))) / 64
    assert bench.res(0) == [3 * 163 + 1] * 32 and [p - q for p, q in zip(bench.res(2), pages0)] == [64] * 32
    bench.run([ring["big"], *_cmds(5, -77)])
    assert bench.res(0) == [big(-77) & 0xFFFFFFFF] * 32 and bench.res(4) == [pg.page_id("big")] * 32
    # page "a" padded to the end of program memory: its last packets write the last program word
    a = pg.pages["a"]
    full = paging.Pager(pg.slot, pg.state, pg.wake, pg.stream_id,
                        {"a": paging.Page(np.concatenate([a.code, np.full((paging.PM_BYTES - pg.slot - a.nbytes) // 4,
                                                                          0xFFFFFFFF, np.uint32)]), a.entries)})
    bench.run([full.slots("a", words=SW, n=NS), *_cmds(-4)])
    assert bench.res(0) == [-11 & 0xFFFFFFFF] * 32
    print(f"\nslot {pg.slot:#x}; pages {', '.join(f'{n} {p.nbytes} B' for n, p in pg.pages.items())}; "
          f"a page of {NS} slots ({4 * NS * SW} B): {per:.0f} cycles")
    assert per < 3 * NS * SW * 4 / 2.4, per  # (~2.4 B / cycle: 12000 cycles for 5 slots of 5760 B)


def test_entries_resident_refused(probe: PagingProbe) -> None:
    """A resident function cannot be a page entry."""
    with pytest.raises(paging.PagingError, match="is resident"):
        paging.link(_work_dir(probe), stem=STEM, pages={"x": ("call",)}, pager="pg_self", caller="call", root="main",
                    stack=STACK, cores=[(0, 2)], state=ST, wake=LW)


