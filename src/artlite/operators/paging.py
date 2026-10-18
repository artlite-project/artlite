"""Code pages: core programs larger than the 16 KB of program memory, loaded by the cores themselves.

A core's program is its resident code - aiecc's link of the core: the startup, its control code and every kernel
function that reaches - and pages: code that only a page's entries reach, which the core loads into program memory at
run time. ``link`` links each page at the slot, the end of the resident code, from a copy of the kernel object whose
page sections are renamed (``.text.<f>`` -> ``.ptxt.<f>``) and placed there by the core's own linker script, and checks
that the resident code and data come out as aiecc's, byte for byte, on every core that shares the pages. ``_main_init``
(the crt's startup function, run once before the core's main) is linked last: the slot starts at it.

A page is control packets to the core's own TileControl (``Pager.stream``): its code at the slot, the pager's state
words in data memory (``STATE_WORDS``: the page's id, its entries), then the core's wake lock. The core triggers the
delivery and sleeps on the wake lock in one resident noinline function (``source``: ``pg_self``, ``pg_request``), and
calls page code only through the entry words, from resident code. Hazards and their guards:

- program memory written while the core fetches from its bank fails (AM027: the fetch has priority, the other access
  gets SLVERR): the trigger is followed at once by the acquire of the wake lock, so the core is stalled when the
  page's packets come;
- a page fetched before it is written: an acquire scheduled into the delay slots of a jump runs after the jump's
  target is fetched (the core then runs stale code, and its fetch blocks the writes). The wake lock is acquired only in
  the pager, which has no jumps (``check_pager``, on the resident disassembly), and page code is entered after the pager
  returned;
- code the core still runs overwritten: pages lie in [slot, 16 KB) above the resident code, and no page reaches the
  pager (``link``: the call graph), so no page is on the stack while another one is written. Program memory ends at
  16 KB: a page past it writes the tile's other registers and breaks the core;
- page data: the core's data memory is the resident ELF's: ``link`` refuses a page whose functions need data the
  resident code does not have (constant or jump tables);
- the stack: the deepest call of the core, page entries called from their resident callers, fits its stack_size;
- a stale page: a configuration (a PDI load) rewrites the slot's start with the startup code and leaves the state
  words: after one, the state is cleared (the memtile transport's reload) or a dispatch's first page is always sent
  (self-DMA: the host's).

Transports, how the packets reach the TileControl:

- self-DMA: the page arrives in the core's input stream as data, in whole slots (``Pager.slots``: a slot's BD ends with
  TLAST, which cuts a packet that crosses it; the wake lock is the last slot's last packet), and the core's own MM2S
  channel streams the slots into its own TileControl (``loop_flow``: a packet flow inside the tile's switchbox, one id
  on every core) while it sleeps in ``pg_self``. The sender of the stream (the host) knows which page each core needs.
  The TileControl takes ~1.9-2.4 B / cycle: two slots of 5760 B in ~4830 cycles, whatever the code in them;
- memtile: the core's request (``pg_request`` with ``request``'s words: a control packet from its stream port to its
  memtile's TileControl, ``request_flow``) pushes a BD of the page's stream, stored in the memtile (``Store``), on a
  memtile MM2S channel of its own (``page_flow``): 86 cycles + ~0.39 us per KB of code. A queue drops a push when full:
  a channel serves at most 4 cores, one request in flight each; a queue shared with tasks that wait for the cores
  deadlocks. The store and its BDs are host-written (``Store.load``) after the design's configuration: again after
  another hardware context ran (its configuration resets the BDs and may use the memtile).
"""

import hashlib
import os
import re
import struct
import subprocess
from dataclasses import dataclass, field

import numpy as np
from aie.dialects._aie_enum_gen import WireBundle
from aie.iron import PacketFlow
from aie.iron.device import Tile

from .. import txn
from .base import JINJA_ENV, PEANO_DIR
from .checks import CORE_FRAME, MAIN_INIT_FRAME, core_frames, stack_usage
from .fused.ctrlpkt import _parity, packets
from .kernel_source import MARK

PM = 0x20000  # (a core tile's program memory, tile-local: the control packets' addresses)
PM_BYTES = 0x4000
CORE_LOCK0 = 0x1F000  # (a core tile's lock values, 16 B apart)
LOCK_BASE = 48  # (core lock 0 in the core's lock numbering: acquire / release ids)
STATE_WORDS = 4  # the pager's state: [page id, entry 0, entry 1, entry 2] (0: no page)
STARTUP = "_main_init"
LOOP_ID = 7  # (the self-DMA flows' stream id: a packet flow inside each tile's switchbox)
OBJDUMP = os.path.join(PEANO_DIR, "bin", "llvm-objdump")
_CLANG = os.path.join(PEANO_DIR, "bin", "clang")


class PagingError(RuntimeError):
    pass


@dataclass
class _Elf:
    """An ELF32 file: its bytes, sections (name, type, flags, address, data, link, info) and their names' offsets in
    the file, symbols (name, value, size, info, section index), relocations by the section they apply to (symbol
    indices)."""

    raw: bytes
    sections: list[tuple[str, int, int, int, bytes, int, int]] = field(default_factory=list)
    name_at: list[int] = field(default_factory=list)
    symbols: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    relocs: dict[int, list[int]] = field(default_factory=dict)

    def section(self, name: str) -> tuple[str, int, int, int, bytes, int, int] | None:
        return next((s for s in self.sections if s[0] == name), None)

    def address(self, name: str) -> int:
        out = [v for n, v, _, info, shn in self.symbols if n == name and shn]
        assert len(out) == 1, (name, out)
        return out[0]


def _read(path: str) -> _Elf:
    with open(path, "rb") as f:
        b = f.read()
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    hdr = [struct.unpack_from("<IIIIIIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    strtab = hdr[shstrndx][4]

    def cstr(o: int) -> str:
        return b[o:b.index(b"\0", o)].decode()
    e = _Elf(b)
    for h in hdr:
        e.sections.append((cstr(strtab + h[0]), h[1], h[2], h[3], b[h[4]:h[4] + h[5]] if h[1] != 8 else b"", h[6], h[7]))
        e.name_at.append(strtab + h[0])
    for i, h in enumerate(hdr):
        if h[1] == 2:  # (SHT_SYMTAB: 16-byte entries, names in the section h[6] links)
            names = hdr[h[6]][4]
            for o in range(h[4], h[4] + h[5], 16):
                st_name, value, size, info, _, shndx = struct.unpack_from("<IIIBBH", b, o)
                e.symbols.append((cstr(names + st_name) if st_name else "", value, size, info, shndx))
        elif h[1] == 4:  # (SHT_RELA: offset, info (symbol << 8 | type), addend; it applies to section h[7])
            e.relocs[h[7]] = [struct.unpack_from("<II", b, o)[1] >> 8 for o in range(h[4], h[4] + h[5], 12)]
    return e


def symbols(names: list[str], name: str) -> list[str]:
    """The linked names of kernel name ``name`` (a placeholder name, ``{p}_<local>``: ``kernel_source`` links it as
    ``<p without the mark>_<local>_<hash>``, in a mangled name or alone; a template's instances: several)."""
    assert name.startswith(MARK), name
    pat = re.compile(rf"(?<![A-Za-z_]){re.escape(name[len(MARK):])}_[0-9a-f]{{8}}(?![0-9a-f])")
    out = sorted({n for n in names if pat.search(n)})
    if not out:
        raise PagingError(f"{name}: no definition")
    return out


def symbol(names: list[str], name: str) -> str:
    """The linked name of kernel name ``name`` (``symbols``: exactly one)."""
    out = symbols(names, name)
    if len(out) != 1:
        raise PagingError(f"{name}: {len(out)} definitions ({', '.join(out)})")
    return out[0]


@dataclass(frozen=True)
class Page:
    """A page's code (program-memory words from ``Pager.slot``) and its entries' program addresses."""

    code: np.ndarray
    entries: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return 4 * self.code.size


@dataclass(frozen=True)
class Pager:
    """The pages of the cores that share them: where they go (``slot``), the pager's state words (data-memory byte
    ``state``), the wake lock, the packets' stream id (their route to the core's TileControl). Page ids count from 1
    in ``pages``' order (0 in the state: no page)."""

    slot: int
    state: int
    wake: int
    stream_id: int
    pages: dict[str, Page]

    def page_id(self, name: str) -> int:
        return list(self.pages).index(name) + 1

    def _state(self, name: str) -> np.ndarray:
        e = self.pages[name].entries
        return np.array([self.page_id(name), *e, *[0] * (STATE_WORDS - 1 - len(e))], np.uint32)

    def stream(self, name: str) -> np.ndarray:
        """Page ``name`` as control packets: its code at the slot, the state words, the wake lock (last)."""
        return packets(self.stream_id, [(PM + self.slot, self.pages[name].code), (self.state, self._state(name)),
                                        (CORE_LOCK0 + 16 * self.wake, np.array([1], np.uint32))])

    def slots(self, name: str, *, words: int, n: int) -> np.ndarray:
        """Page ``name`` in ``n`` slots of ``words`` words of whole packets (self-DMA), the wake lock the last slot's
        last packet; the room left in a slot filled with writes of the state words' own values."""
        pk = _split(self.stream(name))
        lock, body = pk[-1], pk[:-1]
        st = self._state(name)
        out = []
        for k in range(n):
            cap = words - (lock.size if k == n - 1 else 0)
            cur, used = [], 0
            while body and used + body[0].size <= cap:
                cur.append(body.pop(0))
                used += cur[-1].size
            while cap - used in (1, 2):  # (a packet has 3-6 words: the last one waits for the next slot)
                body.insert(0, cur.pop())
                used -= body[0].size
            cur += _filler(self.stream_id, self.state, st, cap - used)
            if k == n - 1:
                cur.append(lock)
            out.append(np.concatenate(cur))
        if body:
            raise PagingError(f"page {name} ({self.pages[name].nbytes} B) does not fit {n} slots of {4 * words} B")
        return np.concatenate(out)


def _split(words: np.ndarray) -> list[np.ndarray]:
    """A stream of control packets cut into its packets (each: stream header, control header, 1-4 words)."""
    out, i = [], 0
    while i < words.size:
        n = 3 + ((int(words[i + 1]) >> 20) & 3)
        out.append(words[i:i + n])
        i += n
    return out


def _filler(stream_id: int, addr: int, values: np.ndarray, n: int) -> list[np.ndarray]:
    """Packets of n words in all (0 or >= 3) rewriting the first words at ``addr`` with ``values``."""
    sizes = [6] * (n // 6)
    m = n % 6
    if m in (1, 2):
        sizes = sizes[:-1] + ([4, 3] if m == 1 else [4, 4])
    elif m:
        sizes.append(m)
    return [packets(stream_id, [(addr, values[:k - 2])]) for k in sizes]


def header(stream_id: int = LOOP_ID) -> int:
    """A page's first word (its first packet's stream header): how a core tells a page from its other inputs."""
    return _parity(stream_id)


def loop_flow(tile: Tile, *, channel: int = 1, stream_id: int = LOOP_ID) -> PacketFlow:
    """The self-DMA route of a core: its MM2S ``channel`` into its own TileControl (the design gives that channel a
    static chain over the slots the pages arrive in)."""
    return PacketFlow(stream_id, tile, tile, src_port=WireBundle.DMA, src_channel=channel,
                      dst_port=WireBundle.TileControl, keep_pkt_header=True,
                      name=f"pg_loop{tile.col}_{tile.row}")


def page_flow(mem: Tile, core: Tile, *, channel: int, stream_id: int) -> PacketFlow:
    """The memtile transport's route of a core's pages: its memtile's MM2S ``channel`` into its TileControl (the stored
    streams carry ``stream_id``: one per core of the column)."""
    return PacketFlow(stream_id, mem, core, src_channel=channel, dst_port=WireBundle.TileControl, keep_pkt_header=True,
                      name=f"pg_page{core.col}_{core.row}")


def request_flow(core: Tile, mem: Tile, *, stream_id: int) -> PacketFlow:
    """The route of a core's page requests: its stream port into its memtile's TileControl."""
    return PacketFlow(stream_id, core, mem, src_port=WireBundle.Core, src_channel=0, dst_port=WireBundle.TileControl,
                      keep_pkt_header=True, name=f"pg_req{core.col}_{core.row}")


def request(*, stream_id: int, channel: int) -> tuple[int, int]:
    """``pg_request``'s first two words: the stream header of a core's request route (``request_flow``) and a control
    header writing one word at its memtile MM2S ``channel``'s start queue (the third: the page BD's id)."""
    return _parity(stream_id), _parity((txn.MEM_MM2S_CTRL + 8 * channel + 4) & 0xFFFFF)


@dataclass(frozen=True)
class Store:
    """A column's page store (memtile transport): the page streams of ``pagers`` (one per core of the column that
    pages, its ``stream_id`` its ``page_flow``'s) in the ``size`` memtile bytes from ``base`` (the design's reserved
    buffer), a BD per (pager, page) from ``bd0`` in that order (BDs of ``channel``'s parity: even channels 0-23, odd
    24-47)."""

    base: int
    size: int
    channel: int
    bd0: int
    pagers: tuple[Pager, ...]

    def __post_init__(self) -> None:
        n = sum(len(p.pages) for p in self.pagers)
        lo = 24 * (self.channel % 2)
        if not (lo <= self.bd0 and self.bd0 + n <= lo + 24):
            raise PagingError(f"BDs {self.bd0}..{self.bd0 + n - 1} are not of memtile channel {self.channel}'s parity")
        if self.nbytes > self.size:
            raise PagingError(f"the pages' streams ({self.nbytes} B) do not fit the store ({self.size} B)")

    def bd(self, k: int, name: str) -> int:
        """The BD of pager k's page ``name`` (``pg_request``'s third word)."""
        return self.bd0 + sum(len(p.pages) for p in self.pagers[:k]) + list(self.pagers[k].pages).index(name)

    def streams(self) -> list[np.ndarray]:
        return [p.stream(n) for p in self.pagers for n in p.pages]

    @property
    def nbytes(self) -> int:
        return sum(4 * s.size for s in self.streams())

    def load(self, t: txn.Txn, cols: range) -> None:
        """The store's words and BDs written in columns ``cols`` (blockwrites)."""
        words = np.concatenate(self.streams())
        off = 0
        bds = []
        for s in self.streams():
            bds.append(txn.mem_bd_words(addr=self.base + 4 * off, nwords=s.size))
            off += s.size
        for c in cols:
            for i in range(0, words.size, 1024):
                t.blockwrite(txn.tile_address(c, 1, self.base + 4 * i), [int(w) for w in words[i:i + 1024]])
            for k, w in enumerate(bds):
                t.mem_bd(c, self.bd0 + k, w)

    def check(self, work_dir: str, cols: range) -> None:
        """None of the store's BDs, its channel's registers or its bytes is in the design's configuration (aiecc's
        CDO: ``cdo_writes``; a check passed once is remembered in the design's directory)."""
        done = os.path.join(work_dir, "design.mlir.prj",
                            f"store_{_key(self.base, self.size, self.channel, self.bd0, self.nbytes, tuple(cols))}.ok")
        if os.path.exists(done):
            return
        regs = cdo_writes(work_dir)
        for c in cols:
            mine = {txn.tile_address(c, 1, txn.MEM_BD0 + 32 * (self.bd0 + k) + 4 * i)
                    for k in range(sum(len(p.pages) for p in self.pagers)) for i in range(8)}
            mine |= {txn.tile_address(c, 1, txn.MEM_MM2S_CTRL + 8 * self.channel + i) for i in (0, 4)}
            lo, hi = txn.tile_address(c, 1, self.base), txn.tile_address(c, 1, self.base + self.nbytes)
            hit = sorted(mine & regs) + [a for a in regs if lo <= a < hi][:1]
            if hit:
                raise PagingError(f"the design configures {hex(hit[0])} (column {c}): the page store's")
        with open(done, "w"):
            pass


def cdo_writes(work_dir: str) -> set[int]:
    """The register and memory addresses a compiled design's configuration writes (aiecc's CDO files: writes,
    masked writes and block writes, 64-bit addresses)."""
    out: set[int] = set()
    prj = os.path.join(work_dir, "design.mlir.prj")
    for name in ("main_aie_cdo_init.bin", "main_aie_cdo_enable.bin", "main_aie_cdo_elfs.bin"):
        path = next((p for p in (os.path.join(prj, "cdo_main", name), os.path.join(prj, name)) if os.path.exists(p)), None)
        if path is None:
            raise PagingError(f"{work_dir}: no CDO {name}")
        with open(path, "rb") as f:
            b = f.read()
        w = struct.unpack(f"<{len(b) // 4}I", b[:len(b) // 4 * 4])
        i = 5  # (the header: its length, "CDO", the version, the length, the checksum)
        while i < len(w):
            api, n = w[i] & 0xFF, (w[i] >> 16) & 0xFF
            i += 1
            if n == 0xFF:
                n, i = w[i], i + 1
            p = w[i:i + n]
            if api in (5, 7, 8):  # (block write, masked write, write: the address in two words)
                a = p[0] << 32 | p[1]
                out.update(a + 4 * k for k in range(n - 2 if api == 5 else 1))
            i += n
    return out


_SRC = r"""
{% if send is not none %}
// (paging: ``artlite.operators.paging``) a page's slots, held by the core, out through its own MM2S into its
// TileControl; asleep until the page's last packet sets the wake lock: nothing runs in between (a write to program
// memory loses to the core's fetch), and page code is entered only after this returns (an acquire in a jump's delay
// slots completes after the jump's target is fetched)
__attribute__((noinline)) void {{ p }}_pg_self() {
  release({{ base + send }}, {{ nslots }});
  acquire_greater_equal({{ base + wake }}, 1);
  __builtin_aie2p_sched_barrier();
}
{% endif %}
{% if request %}
// (paging: ``artlite.operators.paging``) a page's request (hdr, ctl: a write of the page BD's id to its memtile
// channel's start queue) out through the core's stream port; asleep until the page's last packet sets the wake lock
__attribute__((noinline)) void {{ p }}_pg_request(int32_t hdr, int32_t ctl, int32_t bd) {
  put_ms(hdr, 0);
  put_ms(ctl, 0);
  put_ms(bd, 1);
  acquire_greater_equal({{ base + wake }}, 1);
  __builtin_aie2p_sched_barrier();
}
{% endif %}
"""


def source(p: str, *, wake: int, send: int | None = None, nslots: int = 2, request: bool = False) -> str:
    """The pager's kernel source under prefix ``p``: ``{p}_pg_self()`` (self-DMA: releases core lock ``send`` by
    ``nslots``, the slots' MM2S BDs' acquire) and / or ``{p}_pg_request(hdr, ctl, bd)`` (memtile); both sleep on core
    lock ``wake``."""
    return JINJA_ENV.from_string(_SRC).render(p=p, wake=wake, send=send, nslots=nslots, request=request, base=LOCK_BASE)


def check_pager(elf_path: str, *, pager: str, wake: int) -> None:
    """The wake lock is acquired only in the pager function, which has no jumps (a resident ELF's disassembly)."""
    out = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn", elf_path], capture_output=True, text=True,
                         check=True).stdout
    cur, acq, jumps = None, set(), []
    want = LOCK_BASE + wake
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if m:
            if not m.group(1).startswith("."):
                cur = m.group(1)
            continue
        for a in re.finditer(r"\bacq(?:\.\w+)?\s+#(0x[0-9a-f]+|\d+)", line):
            if int(a.group(1), 0) == want:
                acq.add(cur)
        if cur == pager and re.search(r"\b(?:j|jl|jz|jnz)\s", line):
            jumps.append(line.strip())
    if acq != {pager}:
        raise PagingError(f"{elf_path}: the wake lock {wake} acquired in {sorted(acq)}, not only in {pager}")
    if jumps:
        raise PagingError(f"{elf_path}: the pager {pager} jumps ({jumps[0]})")


def _reach(obj: _Elf, start: set[int]) -> set[int]:
    """The sections of a relocatable object reachable from ``start`` through their relocations."""
    seen, todo = set(start), list(start)
    while todo:
        s = todo.pop()
        for k in obj.relocs.get(s, ()):
            t = obj.symbols[k][4]
            if 0 < t < len(obj.sections) and t not in seen:
                seen.add(t)
                todo.append(t)
    return seen


def _core_elf(prj: str, c: int, r: int) -> str:
    return os.path.join(prj, f"elfs_main_core_{c}_{r}", f"elfs_main_core_{c}_{r}.elf")


def _same_kernels(ref: _Elf, other: _Elf, end: int, where: str) -> None:
    """Two cores' resident code agrees below ``end`` but for their own control code (``core_<c>_<r>``), and their data."""
    def funcs(e: _Elf) -> dict[str, tuple[int, int]]:
        return {n: (v, s) for n, v, s, info, shn in e.symbols if info & 15 == 2 and shn and n}
    fa, fb = funcs(ref), funcs(other)
    own = {n for n in list(fa) + list(fb) if re.fullmatch(r"core_\d+_\d+", n)}
    if {n: v for n, v in fa.items() if n not in own} != {n: v for n, v in fb.items() if n not in own}:
        raise PagingError(f"{where}: its kernel functions are not where the pages' core has them")
    ta, tb = bytearray(ref.section(".text")[4][:end]), bytearray(other.section(".text")[4][:end])
    for e, t in ((fa, ta), (fb, tb)):
        for n in own & set(e):
            v, s = e[n]
            t[v:v + s] = bytes(s)
    if ta != tb:
        raise PagingError(f"{where}: its resident code differs from the pages' core's")
    for name in [s[0] for s in ref.sections if s[2] & 2 and not s[2] & 4]:  # (SHF_ALLOC, not SHF_EXECINSTR)
        if (other.section(name) or ("", 0, 0, 0, b""))[4] != ref.section(name)[4]:
            raise PagingError(f"{where}: its data section {name} differs from the pages' core's")


def link(work_dir: str, *, pages: dict[str, tuple[str, ...]], pager: str, calls: dict[str, tuple[str, ...]],
         roots: tuple[str, ...], stack: int, cores: list[tuple[int, int]], state: int, wake: int,
         stream_id: int = LOOP_ID) -> Pager:
    """The pages of the cores ``cores`` (their resident code alike: checked) of a compiled design in ``work_dir``:
    ``pages`` maps a page's name to its entries (the state's entry words in order); ``pager``: the pager function
    (``pg_self`` / ``pg_request``), ``calls``: per resident function the entries it calls (through the state words),
    ``roots``: the core's kernels (their stack, ``stack`` bytes). Names are kernel names (``{p}_<local>``: ``symbols``; a
    caller or root may be a template). Each page's ELF is kept in the project directory (``page_<name>.elf``), the
    result too (``pages_<key>.npz``: a design's directory is named by its content, the key is the arguments and this
    module's source)."""
    prj = os.path.join(work_dir, "design.mlir.prj")
    key = _key(pages, pager, calls, roots, stack, cores, state, wake, stream_id)
    cached = os.path.join(prj, f"pages_{key}.npz")
    if os.path.exists(cached):
        with np.load(cached) as z:
            return Pager(int(z["slot"]), state, wake, stream_id,
                         {n: Page(z[f"code_{i}"], tuple(int(e) for e in z[f"entries_{i}"])) for i, n in enumerate(pages)})
    c0, r0 = cores[0]
    ref_path = _core_elf(prj, c0, r0)
    ref = _read(ref_path)
    obj = _read(os.path.join(work_dir, "kernel.o"))
    names = [s[0] for s in obj.symbols if s[0]]
    entries = {e for es in pages.values() for e in es}
    if {e for es in calls.values() for e in es} - entries:
        raise PagingError(f"calls of functions no page enters: {sorted({e for es in calls.values() for e in es} - entries)}")
    lname = {n: symbol(names, n) for n in {pager, *entries}}
    resident = {n for n, v, s, info, shn in ref.symbols if n and shn}
    many = {n: [x for x in symbols(names, n) if x in resident] for n in {*roots, *calls}}  # (their instances here)
    if lname[pager] not in resident or not all(many.values()):
        raise PagingError(f"the pager {pager}, a caller of entries ({', '.join(calls)}) or a root ({', '.join(roots)}) "
                          "is not resident")
    entry_of = {n: shn for n, v, s, info, shn in obj.symbols if n and shn}
    sec_resident = {shn for n, v, s, info, shn in obj.symbols if n in resident and 0 < shn < len(obj.sections)}
    text = ref.section(".text")[4]
    slot = ref.address(STARTUP)
    if slot + _size(ref, STARTUP) < len(text):
        raise PagingError(f"{STARTUP} is not the last resident function")
    for c, r in cores[1:]:
        _same_kernels(ref, _read(_core_elf(prj, c, r)), slot, f"core ({c}, {r})")
    check_pager(ref_path, pager=lname[pager], wake=wake)
    usage = _stack(work_dir, {c: {lname[e] for e in es} for n, es in calls.items() for c in many[n]})
    need = max(usage.get(r, 0) for n in roots for r in many[n]) + \
        core_frames(work_dir).get(f"{c0},{r0}", CORE_FRAME) + MAIN_INIT_FRAME
    if need > stack:
        raise PagingError(f"the core's deepest call with page entries needs {need} B of stack, the core has {stack}")
    with open(os.path.join(prj, f"ldScripts_main_core_{c0}_{r0}.ld.script")) as f:
        script = f.read()
    pager_sec = entry_of[lname[pager]]
    out: dict[str, Page] = {}
    for name, entries in pages.items():
        esec = {entry_of[lname[e]] for e in entries}
        if esec & sec_resident:
            raise PagingError(f"page {name}: an entry is resident (the resident code reaches it)")
        own = _reach(obj, esec) - sec_resident
        if pager_sec in _reach(obj, own):
            raise PagingError(f"page {name} reaches the pager")
        bad = [obj.sections[s][0] for s in own if not obj.sections[s][0].startswith(".text.")]
        if bad:
            raise PagingError(f"page {name} needs data the resident code does not have: {', '.join(sorted(bad))}")
        elf = _link_page(prj, obj, own, script, slot, name, (c0, r0))
        e = _read(elf)
        if _without_start(e, slot) != _without_start(ref, slot):
            raise PagingError(f"page {name}: its link moved the resident code")
        for s in ref.sections:
            if s[2] & 2 and not s[2] & 4 and (e.section(s[0]) or ("", 0, 0, 0, b""))[4] != s[4]:
                raise PagingError(f"page {name}: its link changed the data section {s[0]}")
        pt = e.section(".ptxt")
        if pt is None or pt[3] != slot or slot + len(pt[4]) > PM_BYTES:
            raise PagingError(f"page {name}: its code at {pt and hex(pt[3])} ({pt and len(pt[4])} B) is not within "
                              f"[{slot:#x}, {PM_BYTES:#x})")
        code = pt[4] + bytes(-len(pt[4]) % 4)
        out[name] = Page(np.frombuffer(code, np.uint32).copy(), tuple(e.address(lname[x]) for x in entries))
    tmp = f"{cached}.{os.getpid()}.npz"
    np.savez(tmp, slot=slot, **{f"code_{i}": p.code for i, p in enumerate(out.values())},
             **{f"entries_{i}": np.array(p.entries, np.int64) for i, p in enumerate(out.values())})
    os.replace(tmp, cached)
    return Pager(slot, state, wake, stream_id, out)


def _key(*args: object) -> str:
    """A cache key of ``args`` and this module's source."""
    with open(__file__, "rb") as f:
        h = hashlib.sha1(f.read())
    h.update(repr(args).encode())
    return h.hexdigest()[:16]


def _without_start(e: _Elf, end: int) -> bytes:
    """The resident code below ``end`` but the crt's ``__start`` (its call of the startup code, moved in a page's link:
    the core runs aiecc's)."""
    t = bytearray(e.section(".text")[4][:end])
    v = e.address("__start")
    t[v:v + _size(e, "__start")] = bytes(_size(e, "__start"))
    return bytes(t)


def _size(e: _Elf, name: str) -> int:
    return next(s for n, v, s, info, shn in e.symbols if n == name and shn)


def _stack(work_dir: str, calls: dict[str, set[str]]) -> dict[str, int]:
    """Each function's deepest stack (``checks.stack_usage``) with the resident functions calling page entries."""
    su = os.path.join(work_dir, "kernel.su")
    if not os.path.exists(su):
        return {}
    return stack_usage(os.path.join(work_dir, "kernel.o"), su, calls=calls)


def _link_page(prj: str, obj: _Elf, own: set[int], script: str, slot: int, name: str, core: tuple[int, int]) -> str:
    """Link page ``name``: kernel.o with its sections ``own`` renamed .text. -> .ptxt., placed at ``slot`` (the crt's
    startup code after it, outside program memory: it ran before any page)."""
    path = os.path.join(prj, f"page_{name}.elf")
    raw = bytearray(obj.raw)
    for s in own:
        at = obj.name_at[s]
        assert raw[at:at + 6] == b".text.", obj.sections[s][0]
        raw[at:at + 6] = b".ptxt."
    tmp = f"{path}.{os.getpid()}"
    with open(tmp + ".o", "wb") as f:
        f.write(raw)
    m = re.search(r"INPUT\((\S*kernel\.o)\)", script)
    old = "     *(.text*)\n  } > program\n"
    if m is None or old not in script:
        raise PagingError("aiecc's linker script is not the form paging edits")
    s = script.replace(m.group(0), f"INPUT({tmp}.o)")
    s = s.replace("MEMORY\n{\n", "MEMORY\n{\n   startup (RX) : ORIGIN = 0x8000, LENGTH = 0x1000\n")
    s = s.replace(old, "     *(EXCLUDE_FILE(*crt1.o) .text*)\n  } > program\n"
                  f"  .ptxt {slot:#x} : {{ KEEP(*(.ptxt.*)) }} > program\n"
                  "  .startup : { *crt1.o(.text*) } > startup\n")
    if "startup (RX)" not in s:
        raise PagingError("aiecc's linker script is not the form paging edits")
    with open(tmp + ".ld", "w") as f:
        f.write(s)
    c, r = core
    try:
        subprocess.run([_CLANG, "-O2", "--target=aie2p-none-unknown-elf", "-fuse-ld=lld",
                        os.path.join(prj, f"objects_main_core_{c}_{r}", f"objects_main_core_{c}_{r}.o"),
                        "-Wl,--gc-sections", "-Wl,--orphan-handling=error", f"-Wl,-T,{tmp}.ld", "-o", tmp],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise PagingError(f"page {name}: link failed: {e.stderr.strip()}") from None
    finally:
        for x in (tmp + ".o", tmp + ".ld"):
            if os.path.exists(x):
                os.remove(x)
    os.replace(tmp, path)
    return path

