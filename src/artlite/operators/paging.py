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
  resident code does not have (constant or jump tables); a page may write words of its own into data memory (``Pager.data``:
  a table in a buffer the design placed for it, which the page's code reads);
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
import subprocess
from dataclasses import dataclass, field

import numpy as np
from aie.dialects._aie_enum_gen import WireBundle
from aie.iron import PacketFlow
from aie.iron.device import Tile

from .. import txn
from ..elf_parser import SHF_ALLOC, SHF_EXECINSTR, ObjectFile, read_object
from ..npu_regs import CORE_TILE
from . import cdo, objdump, templating, toolchain
from .checks import CORE_FRAME, MAIN_INIT_FRAME, core_frames, stack_usage
from .fused.ctrlpkt import packets, parity
from .kernel_source import MARK

PM = CORE_TILE.program[0]  # (a core tile's program memory, tile-local: the control packets' addresses)
PM_BYTES = CORE_TILE.program[1] - CORE_TILE.program[0]
CORE_LOCK0 = CORE_TILE.locks[0]  # (a core tile's lock values, 16 B apart)
LOCK_BASE = 48  # (core lock 0 in the core's lock numbering: acquire / release ids)
STATE_WORDS = 4  # the pager's state: [page id, entry 0, entry 1, entry 2] (0: no page)
STARTUP = "_main_init"
LOOP_ID = 7  # (the self-DMA flows' stream id: a packet flow inside each tile's switchbox)


class PagingError(RuntimeError):
    pass


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
    in ``pages``' order (0 in the state: no page). ``data``: words a page writes into data memory besides (per page
    (byte address, words) runs: a table its code reads), before its state words."""

    slot: int
    state: int
    wake: int
    stream_id: int
    pages: dict[str, Page]
    data: dict[str, tuple[tuple[int, np.ndarray], ...]] = field(default_factory=dict)

    def page_id(self, name: str) -> int:
        return list(self.pages).index(name) + 1

    def _state(self, name: str) -> np.ndarray:
        e = self.pages[name].entries
        return np.array([self.page_id(name), *e, *[0] * (STATE_WORDS - 1 - len(e))], np.uint32)

    def stream(self, name: str) -> np.ndarray:
        """Page ``name`` as control packets: its code at the slot, its data, the state words, the wake lock (last)."""
        return packets(self.stream_id, [(PM + self.slot, self.pages[name].code), *self.data.get(name, ()),
                                        (self.state, self._state(name)),
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
    return parity(stream_id)


class _PageFlow(PacketFlow):
    """A route of pages into a core's TileControl: they write its program memory, the pager's state words and its wake
    lock, never its DMA, so IRON's refusal of a TileControl route into a tile with a TileDma program (a dispatch's
    reconfiguration would rewrite that DMA) does not hold for it."""

    def control_tiles(self) -> list[Tile]:
        return []


def loop_flow(tile: Tile, *, channel: int = 1, stream_id: int = LOOP_ID) -> PacketFlow:
    """The self-DMA route of a core: its MM2S ``channel`` into its own TileControl (the design gives that channel a
    static chain over the slots the pages arrive in)."""
    return _PageFlow(stream_id, tile, tile, src_port=WireBundle.DMA, src_channel=channel,
                     dst_port=WireBundle.TileControl, keep_pkt_header=True, name=f"pg_loop{tile.col}_{tile.row}")


def page_flow(mem: Tile, core: Tile, *, channel: int, stream_id: int) -> PacketFlow:
    """The memtile transport's route of a core's pages: its memtile's MM2S ``channel`` into its TileControl (the stored
    streams carry ``stream_id``: one per core of the column)."""
    return _PageFlow(stream_id, mem, core, src_channel=channel, dst_port=WireBundle.TileControl, keep_pkt_header=True,
                     name=f"pg_page{core.col}_{core.row}")


def request_flow(core: Tile, mem: Tile, *, stream_id: int) -> PacketFlow:
    """The route of a core's page requests: its stream port into its memtile's TileControl."""
    return PacketFlow(stream_id, core, mem, src_port=WireBundle.Core, src_channel=0, dst_port=WireBundle.TileControl,
                      keep_pkt_header=True, name=f"pg_req{core.col}_{core.row}")


def request(*, stream_id: int, channel: int) -> tuple[int, int]:
    """``pg_request``'s first two words: the stream header of a core's request route (``request_flow``) and a control
    header writing one word at its memtile MM2S ``channel``'s start queue (the third: the page BD's id)."""
    return parity(stream_id), parity((txn.MEM_MM2S_CTRL + 8 * channel + 4) & 0xFFFFF)


@dataclass(frozen=True)
class PageSpec:
    """Where a core's pages come from, as its kernel source needs it (the memtile transport; ``source``'s ``pg_request``
    takes ``hdr`` and ``ctl``): the pager's state words (a core address), its wake lock (a core lock id) and, per core
    half that pages (a core of the column, or two sharing the kernels: a decode design's main and helper cores), the
    stream header of the half's request route (``request``), the BD of its first page (``Store.bd``: page i at
    bd0 + i - 1) and the control header writing its page channel's start queue (``request``). The cores' resident stubs (the code
    that asks for a page: the decode GEMV's, the attention's) stay in their own sources: they differ in the scratch
    pointer and in what they keep across the call."""

    state: int
    wake: int
    hdr: tuple[int, ...]
    bd0: tuple[int, ...]
    ctl: tuple[int, ...]


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
        CDO: ``cdo.writes``; a check passed once is remembered in the design's directory)."""
        done = os.path.join(work_dir, "design.mlir.prj",
                            f"store_{_key(self.base, self.size, self.channel, self.bd0, self.nbytes, tuple(cols))}.ok")
        if os.path.exists(done):
            return
        regs = cdo.writes(work_dir)
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


_SRC = templating.load("artlite/operators/paging.cc.jinja")


def source(p: str, *, wake: int, send: int | None = None, nslots: int = 2, request: bool = False) -> str:
    """The pager's kernel source under prefix ``p``: ``{p}_pg_self()`` (self-DMA: releases core lock ``send`` by
    ``nslots``, the slots' MM2S BDs' acquire) and / or ``{p}_pg_request(hdr, ctl, bd)`` (memtile); both sleep on core
    lock ``wake``."""
    return _SRC.render(p=p, wake=wake, send=send, nslots=nslots, request=request, base=LOCK_BASE)


def check_pager(elf_path: str, *, pager: str, wake: int) -> None:
    """The wake lock is acquired only in the pager function, which has no jumps (a resident ELF's disassembly)."""
    funcs = objdump.functions(elf_path)
    acq = {fn for fn, f in funcs.items() if LOCK_BASE + wake in f.acquires}
    jumps = funcs[pager].jumps if pager in funcs else ()
    if acq != {pager}:
        raise PagingError(f"{elf_path}: the wake lock {wake} acquired in {sorted(acq)}, not only in {pager}")
    if jumps:
        raise PagingError(f"{elf_path}: the pager {pager} jumps ({jumps[0]})")


def _reach(obj: ObjectFile, start: set[int]) -> set[int]:
    """The sections of a relocatable object reachable from ``start`` through their relocations."""
    seen, todo = set(start), list(start)
    while todo:
        s = todo.pop()
        for k in obj.relocs.get(s, ()):
            t = obj.symbols[k].section_index
            if 0 < t < len(obj.sections) and t not in seen:
                seen.add(t)
                todo.append(t)
    return seen


def _core_elf(prj: str, c: int, r: int) -> str:
    return os.path.join(prj, f"elfs_main_core_{c}_{r}", f"elfs_main_core_{c}_{r}.elf")


def _data(obj: ObjectFile, name: str) -> bytes:
    """The contents of section ``name`` (empty where the object has none)."""
    sec = obj.section(name)
    return sec.data if sec is not None else b""


def _same_kernels(ref: ObjectFile, other: ObjectFile, end: int, where: str) -> None:
    """Two cores' resident code agrees below ``end`` but for their own control code (``core_<c>_<r>``), and their data."""
    def funcs(e: ObjectFile) -> dict[str, tuple[int, int]]:
        return {s.name: (s.value, s.size) for s in e.symbols if s.is_function and s.section_index and s.name}
    fa, fb = funcs(ref), funcs(other)
    own = {n for n in list(fa) + list(fb) if re.fullmatch(r"core_\d+_\d+", n)}
    if {n: v for n, v in fa.items() if n not in own} != {n: v for n, v in fb.items() if n not in own}:
        raise PagingError(f"{where}: its kernel functions are not where the pages' core has them")
    ta, tb = bytearray(_data(ref, ".text")[:end]), bytearray(_data(other, ".text")[:end])
    for e, t in ((fa, ta), (fb, tb)):
        for n in own & set(e):
            v, s = e[n]
            t[v:v + s] = bytes(s)
    if ta != tb:
        raise PagingError(f"{where}: its resident code differs from the pages' core's")
    for name in [s.name for s in ref.sections if s.flags & SHF_ALLOC and not s.flags & SHF_EXECINSTR]:
        if _data(other, name) != _data(ref, name):
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
    ref = read_object(ref_path)
    obj = read_object(os.path.join(work_dir, "kernel.o"))
    names = [s.name for s in obj.symbols if s.name]
    entries = {e for es in pages.values() for e in es}
    if {e for es in calls.values() for e in es} - entries:
        raise PagingError(f"calls of functions no page enters: {sorted({e for es in calls.values() for e in es} - entries)}")
    lname = {n: symbol(names, n) for n in {pager, *entries}}
    resident = {s.name for s in ref.symbols if s.name and s.section_index}
    many = {n: [x for x in symbols(names, n) if x in resident] for n in {*roots, *calls}}  # (their instances here)
    if lname[pager] not in resident or not all(many.values()):
        raise PagingError(f"the pager {pager}, a caller of entries ({', '.join(calls)}) or a root ({', '.join(roots)}) "
                          "is not resident")
    entry_of = {s.name: s.section_index for s in obj.symbols if s.name and s.section_index}
    sec_resident = {s.section_index for s in obj.symbols
                    if s.name in resident and 0 < s.section_index < len(obj.sections)}
    text = _data(ref, ".text")
    slot = ref.address(STARTUP)
    if slot + _size(ref, STARTUP) < len(text):
        raise PagingError(f"{STARTUP} is not the last resident function")
    for c, r in cores[1:]:
        _same_kernels(ref, read_object(_core_elf(prj, c, r)), slot, f"core ({c}, {r})")
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
        bad = [obj.sections[s].name for s in own if not obj.sections[s].name.startswith(".text.")]
        if bad:
            raise PagingError(f"page {name} needs data the resident code does not have: {', '.join(sorted(bad))}")
        elf = _link_page(prj, obj, own, script, slot, name, (c0, r0))
        e = read_object(elf)
        if _without_start(e, slot) != _without_start(ref, slot):
            raise PagingError(f"page {name}: its link moved the resident code")
        for s in ref.sections:
            if s.flags & SHF_ALLOC and not s.flags & SHF_EXECINSTR and _data(e, s.name) != s.data:
                raise PagingError(f"page {name}: its link changed the data section {s.name}")
        pt = e.section(".ptxt")
        if pt is None or pt.addr != slot or slot + len(pt.data) > PM_BYTES:
            raise PagingError(f"page {name}: its code at {pt and hex(pt.addr)} ({pt and len(pt.data)} B) is not within "
                              f"[{slot:#x}, {PM_BYTES:#x})")
        code = pt.data + bytes(-len(pt.data) % 4)
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


def _without_start(e: ObjectFile, end: int) -> bytes:
    """The resident code below ``end`` but the crt's ``__start`` (its call of the startup code, moved in a page's link:
    the core runs aiecc's)."""
    t = bytearray(_data(e, ".text")[:end])
    v = e.address("__start")
    t[v:v + _size(e, "__start")] = bytes(_size(e, "__start"))
    return bytes(t)


def _size(e: ObjectFile, name: str) -> int:
    return next(s.size for s in e.symbols if s.name == name and s.section_index)


def _stack(work_dir: str, calls: dict[str, set[str]]) -> dict[str, int]:
    """Each function's deepest stack (``checks.stack_usage``) with the resident functions calling page entries."""
    su = os.path.join(work_dir, "kernel.su")
    if not os.path.exists(su):
        return {}
    return stack_usage(os.path.join(work_dir, "kernel.o"), su, calls=calls)


def _link_page(prj: str, obj: ObjectFile, own: set[int], script: str, slot: int, name: str,
               core: tuple[int, int]) -> str:
    """Link page ``name``: kernel.o with its sections ``own`` renamed .text. -> .ptxt., placed at ``slot`` (the crt's
    startup code after it, outside program memory: it ran before any page)."""
    path = os.path.join(prj, f"page_{name}.elf")
    raw = bytearray(obj.raw)
    for s in own:
        at = obj.sections[s].name_offset
        assert raw[at:at + 6] == b".text.", obj.sections[s].name
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
        subprocess.run([toolchain.peano_tool("clang"), "-O2", "--target=aie2p-none-unknown-elf", "-fuse-ld=lld",
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

