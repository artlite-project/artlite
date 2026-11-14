"""Programs assembled on the host from a library ELF's control program (``Library``; no device here).

aiecc lowers a library's runlist to one block per entry: a ConfigureOp (an expanded PDI load: the array reset and its
design's configuration, ending with the cores' enables) and a RunOp. A program lists items, each an entry with the
device addresses of its arguments; ``Assembler.assemble`` writes the items' RunOps with those addresses, an entry's
configuration only where the previous item is of another design:

- the configurations' PDI loads alternate between the library's two (the firmware skips the load, and with it the
  array reset, of the PDI it loaded last); a piece's first loads the other one first (skipped after the previous
  piece: 0.25 us), so that its own load resets whatever ran before (a piece assembled and never dispatched, one a
  TDR stopped, another context's, a fresh context: neither PDI loaded yet);
- a RunOp again right after its own design: the form with a poll for a free task-queue slot before each shim task
  push (``Run.guarded``);
- memory-tile BDs the registers already hold since the last configuration left out (``Run.emit``);
- the configurations without the writes their PDI load's array reset makes redundant, a tile's stream switch ports
  as blockwrites (``prune``: 57% fewer configuration operations in a Llama prefill chunk, -6% time at 1K tokens);
- the cores' program and data memory by control packets where they cost less than blockwrites (``ctrlpkt``: a Llama
  1B chunk of 128 tokens 43 -> 35 ms), from a packet buffer the library allocates (``Assembler.place``);
- a configuration after another of the same piece without the array reset (``Quiet``: the PDI loaded last again, which
  the firmware skips, 0.25 against 37 us; a 64-token chunk of Qwen3 0.6B or qwen35 0.8B -7 to -8 ms): the previous
  design's cores outside the next one held in reset, its cores that wait on a neighbour's locks too and those locks
  flushed (``locks``), each tile of the design quiesced by its packets (``ctrlpkt.Loader(reused=True)``). The reset
  stays where a design could still disturb the next one (``Assembler._quiet``: a stale stream switch master on a slave
  the next design uses, a memory tile DMA of the previous one on a neighbour's locks, a channel of the next one that
  sends before any lock release), at a piece's first configuration (another context may have run before it), and
  everywhere with ``ARTLITE_PDI_RESET=1``.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ... import settings, txn
from ...npu_regs import CORE_CONTROL_RESET, CORE_TILE, MEM_TILE, tile_map
from . import ctrlpkt, quiet
from .bdelide import Run
from .elf import ShimBd
from .prune import prune

if TYPE_CHECKING:
    from .locks import Census

# Bytes of a program's piece (a dispatch: ``Library.run``) at most: the process's two piece BOs share its device heap
# with the PDIs (smaller on a heap that cannot grow: ``library.Pieces``)
PIECE_BYTES = 24 << 20


def quiet_enabled() -> bool:
    """Whether a configuration may go without the array reset (``Quiet``): not with ``ARTLITE_PDI_RESET=1``; read at
    each use. It also needs the control packets (``ctrlpkt.enabled``): the quiet configuration is built on them."""
    return not settings.PDI_RESET.get() and ctrlpkt.enabled()


Item = tuple[int, Sequence[int]]  # (entry, the device addresses of its arguments)
Slot = tuple[int, int, int, int]  # (scratch start, end, entry, argument): where the ELF placed an entry's argument


@dataclass(eq=False)
class Block:
    """An entry of the library: its configuration (words 0 .. 3: its PDI load), operation count and RunOp, the
    argument of each data BD and its byte offset in it. ``design``: the first entry with the same configuration and
    packets (operators that share one array design and differ in their runtime sequences; designs that differ only in
    program memory differ only in their packets). ``packets``: the control packets
    of its program memory (``ctrlpkt``) and the configuration word of each one's shim BD address (``Assembler.place``)."""

    config: np.ndarray
    n_config: int
    run: Run
    bd_arg: np.ndarray
    bd_rel: np.ndarray
    design: int
    packets: list[tuple[int, np.ndarray]] = field(default_factory=list)
    ports: dict[int, dict[int, int]] = field(default_factory=dict)  # (``quiet.ports`` of its configuration)
    quiet: quiet.Quiet | None = None


def _cut(code: np.ndarray, bds: Sequence[ShimBd], slots: Sequence[Slot]
         ) -> tuple[np.ndarray, np.ndarray, list[Block], list[np.ndarray]]:
    """A library ELF's control program (patched, 32-bit words) -> its header, its preamble (the operations before the
    first PDI load: the scratchpad's creation), one ``Block`` per entry and the two PDI loads its configurations
    alternate between. ``bds``: its data BDs, ``slots``: where the ELF placed each entry's arguments."""
    ops = [(p + 4, op) for p, op in txn.op_offsets(code[4:])]
    starts = [p for p, _ in ops] + [len(code)]
    loads = [i for i, (_, op) in enumerate(ops) if op == txn.OP_LOAD_PDI] + [len(ops)]
    if loads[-2] == len(ops) - 1:  # (a PDI load closing the control code, after the last entry: not an entry)
        ops, starts = ops[:-1], starts[:-1]
        loads[-2:] = [len(ops)]
    assert all(op == txn.OP_CREATE_SCRATCHPAD for _, op in ops[:loads[0]]), "operations before the first PDI load"
    slot_starts = np.array([s[0] for s in slots])
    order = np.argsort(slot_starts)
    bd_at = np.array([bd.offset // 4 + 1 for bd in bds], np.int64)  # (the address word)
    bd_start = np.array([bd.start for bd in bds], np.int64)
    designs: dict[bytes, int] = {}  # configuration (after its PDI load) and its packets -> the first entry with them
    blocks: list[Block] = []
    for e, (a, b) in enumerate(itertools.pairwise(loads)):
        enables = [i for i in range(a, b) if ops[i][1] == txn.OP_MASKWRITE
                   and code[starts[i] + 2] & 0xFFFFF == CORE_TILE.control[0] and code[starts[i] + 4] & 1]
        assert enables, f"block {e}: no core enables"
        r = enables[-1] + 1  # (the RunOp: after the configuration's core enables)
        lo, mid, hi = starts[a], starts[r], starts[b]
        assert not ((bd_at >= lo) & (bd_at < mid)).any(), f"block {e}: a data BD in the configuration"
        sel = (bd_at >= mid) & (bd_at < hi)
        k = order[np.searchsorted(slot_starts[order], bd_start[sel], side="right") - 1]
        assert all(slots[i][2] == e for i in k) and (bd_start[sel] < [slots[i][1] for i in k]).all(), \
            f"block {e}: BD arguments"
        cfg, loader, pm = code[lo + 4:mid], None, None
        if ctrlpkt.enabled():
            without, pm = ctrlpkt.split(cfg)
            if pm and ctrlpkt.worth(candidate := ctrlpkt.Loader(pm)):
                cfg, loader = without, candidate
        rest, left = prune(cfg, loader.ports() if loader is not None else ())
        packets = []
        if loader is not None:
            pre, _, at = loader.prelude(left + loader.slots(rest))
            rest = np.concatenate([pre, rest])
            packets = [(4 + w, p) for w, p in at]
        config = np.concatenate([code[lo:lo + 4], rest])
        run = Run(code[mid:hi].copy(), b - r, bd_at[sel] - mid)
        blocks.append(Block(config, 1 + len(txn.op_offsets(rest)), run,
                            np.array([slots[i][3] for i in k], np.int64), bd_start[sel] - slot_starts[k],
                            designs.setdefault(b"".join([config[4:].tobytes(), *(p.tobytes() for _, p in packets)]), e),
                            packets, quiet.ports(rest), quiet.quiet(code[lo:lo + 4], without, pm, run) if pm else None))
    loadpdi = {tuple(int(v) for v in blk.config[:4]) for blk in blocks}
    assert len(loadpdi) == 2, f"configurations alternate between two PDI loads: {len(loadpdi)}"
    return code[:4].copy(), code[4:starts[loads[0]]].copy(), blocks, [np.array(v, np.uint32) for v in sorted(loadpdi)]


class Assembler:
    """Programs of a library ELF's blocks (``_cut``)."""

    def __init__(self, code: np.ndarray, bds: Sequence[ShimBd], slots: Sequence[Slot], census: Census | None = None):
        """``census``: the library's cores' acquires of a neighbour's lock (``locks.census``; None: every
        configuration with the array reset)."""
        self.header, self.preamble, self.blocks, self.loadpdi = _cut(code, bds, slots)
        # (the PDI load of the next configuration with the reset, the other most likely the firmware's last; the first:
        # the second, as aiecc starts)
        self.parity = 1
        self.item_ops: list[int] = []  # (the last ``assemble``'s operation count through the end of each of its items)
        self.reuse = quiet_enabled() and census is not None  # (configurations without the array reset: ``Quiet``)
        # (per entry: its cores that acquire a neighbour's lock, and those (tile, lock))
        self._neighbours = {e: (frozenset(t for t, _, _ in v), frozenset((lt, k) for _, lt, k in v))
                            for e, v in (census or {}).items()}
        self._pre: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        self._offsets: dict[bytes, int] = {}  # (each distinct packet stream's byte offset in the packet buffer)
        self.packet_bytes = 0
        for blk in self.blocks:
            for _, p in [*blk.packets, *(blk.quiet.packets if blk.quiet is not None else ())]:
                if (key := p.tobytes()) not in self._offsets:
                    self._offsets[key] = self.packet_bytes
                    self.packet_bytes += len(key)

    def place(self, view: np.ndarray, address: int) -> None:
        """Write the configurations' control packets into the packet buffer (``view``: its uint32 words, at the device
        address ``address``) and their shim BDs' addresses into the configurations."""
        for data, off in self._offsets.items():
            view[off // 4:off // 4 + len(data) // 4] = np.frombuffer(data, np.uint32)
        for cfg, packets in [(b.config, b.packets) for b in self.blocks] + \
                            [(b.quiet.config, b.quiet.packets) for b in self.blocks if b.quiet is not None]:
            for w, p in packets:
                a = address + self._offsets[p.tobytes()]
                cfg[w], cfg[w + 1] = a & 0xFFFFFFFF, a >> 32

    def _configure(self, e: int, prev: int | None) -> bool:
        """Whether item ``e`` after item ``prev`` takes its design's configuration."""
        return prev is None or self.blocks[prev].design != self.blocks[e].design

    def _before(self, p: int, e: int) -> tuple[np.ndarray, int]:
        """The transaction before entry ``e``'s ``Quiet`` after entry ``p``'s (words, operation count): ``p``'s cores
        outside ``e``'s tiles held in reset, and its cores that acquire a neighbour's lock (and those locks' tiles'),
        then each of those locks written ``ctrlpkt.LOCK_FLUSH`` (granting the acquire such a core left pending: neither
        its reset cancels it nor, when the lock's tile comes first, that tile's own quiescing) and 0."""
        if (p, e) not in self._pre:
            qp, qe = self.blocks[p].quiet, self.blocks[e].quiet
            crossers, locks = self._neighbours.get(p, ((), ()))
            stop = (qp.cores - qe.tiles) | set(crossers) | {t for t, _ in locks if t in qp.cores}
            t = txn.Txn()
            for c, r in sorted(stop):
                t.write32(txn.tile_address(c, r, CORE_TILE.control[0]), CORE_CONTROL_RESET)
            for (c, r), k in sorted(locks):
                for v in (ctrlpkt.LOCK_FLUSH, 0):
                    t.write32(txn.tile_address(c, r, ctrlpkt.LOCKS[2][0] + 16 * k), v)
            self._pre[(p, e)] = (t.words()[4:], t.n_ops)
        return self._pre[(p, e)]

    def _quiet(self, p: int | None, e: int, ports: dict[int, dict[int, int]]) -> bool:
        """Whether entry ``e``'s configuration after entry ``p``'s goes without the array reset (``ports``: the stream
        switch port configurations since the last reset): not where ``p`` leaves an acquire pending that no flush
        reaches (a memory tile DMA's on a neighbour's lock, a core's on a neighbour's lock its tile's DMA also takes),
        where a channel of ``e`` sends before any lock release (the previous routes would take it), or where a master
        port enabled since the last reset that ``e`` does not configure would take a stream of ``e`` (its slave enabled)
        or packets (in packet mode)."""
        if not self.reuse or p is None or self.blocks[p].quiet is None or self.blocks[e].quiet is None:
            return False
        qp, qe = self.blocks[p].quiet, self.blocks[e].quiet
        _, locks = self._neighbours.get(p, ((), ()))
        if qp.far or qe.early or any(k in qp.dma_locks.get(t, ()) for t, k in locks):
            return False
        for tile, regs in qe.ports.items():
            old = ports.get(tile)
            if not old:
                continue
            (m0, m1), (s0, _) = tile_map(txn.address_row(tile)).ports
            for reg, v in old.items():
                if m0 <= reg < m1 and v >> 31 and reg not in regs and \
                        (v >> 30 & 1 or regs.get(s0 + 4 * (v & 0x7F), old.get(s0 + 4 * (v & 0x7F), 0)) >> 31):
                    return False
        return True

    def assemble(self, items: Sequence[Item]) -> np.ndarray:
        """A control program (a TXN) of the items; advances ``parity``; ``item_ops`` holds where each item ends."""
        parts, n_ops, prev = [self.header, self.preamble], len(self.preamble) // 4, None  # (CREATE_SCRATCHPAD: 4 words)
        self.item_ops = ends = []
        state = np.full((64, (MEM_TILE.bds[1] - MEM_TILE.bds[0]) // 4), -1, np.int64)  # (per column: the BDs held)
        ports: dict[int, dict[int, int]] = {}  # (the port configurations since the last array reset, per tile)
        for e, addrs in items:
            blk = self.blocks[e]
            configure = self._configure(e, prev)
            if configure and self._quiet(prev, e, ports):
                pre, n_pre = self._before(prev, e)
                parts += [self.loadpdi[self.parity ^ 1], pre, blk.quiet.config[4:]]  # (the PDI loaded last: skipped)
                n_ops += blk.quiet.n_config + n_pre
                for tile, regs in blk.quiet.ports.items():
                    ports.setdefault(tile, {}).update(regs)
                state[:] = -1
            elif configure:
                cfg = blk.config.copy()
                cfg[:4] = self.loadpdi[self.parity]
                if prev is None:  # (a piece's first: the other PDI loaded first, so that this load is never skipped)
                    parts.append(self.loadpdi[self.parity ^ 1])
                    n_ops += 1
                self.parity ^= 1
                parts.append(cfg)
                n_ops += blk.n_config
                ports = {tile: dict(regs) for tile, regs in blk.ports.items()}
                state[:] = -1
            words, bd_words, n = (blk.run if configure else blk.run.guarded).emit(state)
            words = words.copy()
            addr = np.asarray(addrs, np.int64)[blk.bd_arg] + blk.bd_rel
            if addr.size and (addr.min() < 0 or addr.max() >= 1 << 48):
                raise txn.HardwareLimitError(f"item {len(ends)} (entry {e}): BD addresses {int(addr.min()):#x} .. "
                                             f"{int(addr.max()):#x} do not fit a BD's 48 bits")
            words[bd_words] = (addr & 0xFFFFFFFC).astype(np.uint32)
            words[bd_words + 1] = (words[bd_words + 1] & 0xFFFF0000) | (addr >> 32).astype(np.uint32)
            parts.append(words)
            n_ops += n
            ends.append(n_ops)
            prev = e
        out = np.concatenate(parts)
        out[2], out[3] = n_ops, 4 * out.size
        return out

    def bounds(self, items: Sequence[Item], splits: Collection[int], limit: int = PIECE_BYTES) -> list[int]:
        """The item bounds of a program's pieces: ``splits``, and a cut at the last change of design before a piece
        would exceed ``limit`` bytes."""
        def nbytes(i: int, prev: int | None) -> int:  # (as ``assemble`` emits it, all memory-tile BDs written)
            e = items[i][0]
            blk = self.blocks[e]
            if prev is None:  # (two PDI loads, then the configuration)
                return 4 * (4 + blk.config.size + blk.run.words.size)
            if self._configure(e, prev):
                quiet = blk.quiet.config.size + self._before(prev, e)[0].size \
                    if self.reuse and blk.quiet is not None and self.blocks[prev].quiet else 0
                return 4 * (max(blk.config.size, quiet) + blk.run.words.size)
            return 4 * blk.run.guarded.words.size

        empty = 4 * (4 + self.preamble.size)
        forced = set(splits) | {len(items)}
        bounds, size, prev, last_change = [0], empty, None, 0
        for i, (e, _) in enumerate(items):
            if i in forced and i:
                bounds.append(i)
                size, prev = empty, None
            n = nbytes(i, prev)
            if prev is None or self.blocks[e].design != self.blocks[prev].design:
                last_change = i
            if size + n > limit and i > bounds[-1]:
                cut = last_change if last_change > bounds[-1] else i
                bounds.append(cut)
                size, prev = empty, None
                for j in range(cut, i + 1):  # (the items moved into the new piece)
                    size += nbytes(j, prev)
                    prev = items[j][0]
                continue
            size += n
            prev = e
        bounds.append(len(items))
        return sorted(set(bounds))
