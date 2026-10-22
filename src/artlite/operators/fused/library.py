"""``Library``: operators compiled into one ELF, each once, as a library of blocks: never dispatched whole, it runs
programs assembled on the host (``program.Assembler``), each item an operator with its arguments at any device
addresses (``items``), in pieces of one hardware context (``run``).

Every argument of an operator is its own slot in the ELF's scratch (a BD's address tells its argument); no data BOs
are allocated: the items address the caller's buffers.
"""

import itertools
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ... import portable, txn
from ...elf_parser import ParsedElf
from ..base import ArtcOperator
from ..overlay.runtime import SPIN_MARGIN_US, irq_cpu
from .elf import FusedElf
from .layout import Runlist, distinct, parse_name
from .program import PIECE_BYTES, Assembler, Item


class Library(FusedElf):
    LIBRARY = True
    # bytes of assembled pieces kept for programs run again: every piece of a prefill's chunk sizes at both parities
    # (Qwen3.6-35B-A3B's 2048-token chunk: 83 pieces, 44 MB; a large piece assembled again ~1.3 ms of host time, on the
    # critical path where the previous piece is short)
    PIECES_KEPT = 512 << 20
    SPIN_US = SPIN_MARGIN_US  # (a piece's wait polls from this long before its shortest seen end: ``wait``)

    def __init__(self, name: str, operators: Iterable[ArtcOperator]):
        """``operators``: the library's entries (each operator object once: repeats are dropped), in order."""
        ops = distinct(operators)
        super().__init__(name=name, runlist=[(op, *[f"e{e}_{k}" for k in range(len(op.get_arg_spec()))])
                                             for e, op in enumerate(ops)])
        self._entry = {id(op): e for e, op in enumerate(ops)}
        self._asm: Assembler | None = None
        self._piece_bos: list = []  # (two instruction BOs: piece k + 1 assembled while k runs)
        self._held: list[tuple | None] = [None, None]  # (per piece BO: the program, bytes and parity after it holds)
        # (assembled pieces: (id(items), a, b, parity) -> (items, piece, parity after); a program's items are a list the
        # caller keeps, e.g. ChunkPrefill's per chunk size)
        self._pieces: dict[tuple[int, int, int, int], tuple[Sequence[Item], np.ndarray, int]] = {}
        self._pieces_bytes = 0
        self._ends: dict[tuple[int, int, int], float] = {}  # ((id(items), a, b) -> a piece's shortest seen duration)

    def _keep(self, key: tuple[int, int, int, int], entry: tuple[Sequence[Item], np.ndarray, int]) -> None:
        """Keep an assembled piece (all dropped when they would exceed PIECES_KEPT bytes)."""
        if self._pieces_bytes + entry[1].nbytes > self.PIECES_KEPT:
            self._pieces.clear()
            self._pieces_bytes = 0
        self._ends: dict[tuple[int, int, int], float] = {}  # ((id(items), a, b) -> a piece's shortest seen duration)
        self._pieces[key] = entry
        self._pieces_bytes += entry[1].nbytes

    def __contains__(self, op: ArtcOperator) -> bool:
        return id(op) in self._entry

    def items(self, runlist: Runlist, buffers: Mapping[str, Any]) -> list[Item]:
        """A program's items: per run of ``runlist`` ((operator, *buffer names)) its entry and the device addresses of
        its arguments in ``buffers`` (host buffers by name; ``"NAME[a:b]"`` from byte a)."""
        ctx = self.load()._ctx  # (on XRT: open, its addresses)

        def address(name: str) -> int:
            base, start, _ = parse_name(name)
            return txn.device_address(buffers[base], start, ctx)

        return [(self._entry[id(op)], [address(n) for n in names]) for op, *names in runlist]

    def _load(self, elf: ParsedElf) -> None:
        super()._load(elf)
        code, bds = self._control_program(elf, lambda bd: 0)  # (the items' addresses: written per item)
        slots = [(s.offset, s.offset + s.nbytes, e, k)
                 for e, (_, *names) in enumerate(self.runlist) for k, s in enumerate(map(self.layout.span, names))]
        self._asm = Assembler(code, bds, slots)
        assert len(self._asm.blocks) == len(self.runlist), (len(self._asm.blocks), len(self.runlist))
        self._dispatch_bufs = list(self._pdis.values())
        if self._asm.packet_bytes:  # (the configurations' control packets: ``ctrlpkt``)
            bo = self._device.alloc_host_buffer(-(-self._asm.packet_bytes // 4096) * 4096)
            self._asm.place(bo.map().view(np.uint32), txn.device_address(bo, 0, self._ctx))
            bo.sync_to_device()
            self._dispatch_bufs.append(bo)

    def run(self, items: Sequence[Item], *, splits: Iterable[int] = (), bufs: Sequence[Any],
            hooks: Mapping[int, Callable[[], None]] | None = None) -> int:
        """Dispatch the program of ``items`` (``items()``; consecutive items of one design share its configuration): a
        new piece of control code (a dispatch) at each item index of ``splits``, and where a piece would exceed
        PIECE_BYTES. ``bufs``: the data BOs the items address. ``hooks``: per item index of ``splits``, a host function
        called before the piece it starts is dispatched (while the previous one runs: e.g. inputs that only later
        items read). Returns the dispatches.

        The calling thread runs on the CPU that takes the NPU's interrupts meanwhile (``irq_cpu``): it waits for each
        piece there (napping, then polling from shortly before its shortest seen end), and the host work between
        pieces runs on that warm core (a thread let go after each wait ran it ~2.5x slower, on a CPU the scheduler
        woke for it)."""
        cpu = irq_cpu()
        with portable.on_cpu(cpu):
            return self._run(items, splits=splits, bufs=bufs, hooks=hooks, cpu=cpu)

    def _run(self, items: Sequence[Item], *, splits: Iterable[int], bufs: Sequence[Any],
             hooks: Mapping[int, Callable[[], None]] | None, cpu: int) -> int:
        asm = self.load()._asm
        kern = self._kernel()
        bounds = asm.bounds(items, set(splits))
        refs = self._dispatch_bufs + [b for b in bufs if all(b is not d for d in self._dispatch_bufs)]
        if not self._piece_bos:
            self._piece_bos = [self._device.alloc_insts_buffer(PIECE_BYTES) for _ in range(2)]
        key = None
        if len(bounds) == 2 and not hooks:  # (a program of one piece run again: still in the BO of its parity)
            key = (hash(tuple((e, tuple(addrs)) for e, addrs in items)), asm.parity)
            held = self._held[asm.parity]
            if held is not None and held[0] == key:
                kern.dispatch_elf_async(self._piece_bos[asm.parity], refs, held[1])
                asm.parity = held[2]
                kern.dispatch_wait()
                return 1
        k, starts = 0, []
        submit, pkey = 0.0, None

        def wait() -> None:  # (the dispatched piece: napping on the interrupts' CPU, then polling near its end)
            est = self._ends.get(pkey)
            after = int(est * 1e6) - self.SPIN_US if est is not None else 1 << 30
            kern.dispatch_wait(0, max(0, after) if cpu >= 0 else -1, max(cpu, 0))
            d = time.perf_counter() - submit
            self._ends[pkey] = min(d, est) if est is not None else d

        for k, (a, b) in enumerate(itertools.pairwise(bounds)):
            starts.append(asm.parity)
            slot = asm.parity if key is not None else k % 2
            ck = (id(items), a, b, asm.parity)
            hit = self._pieces.get(ck)
            if hit is not None and hit[0] is items:
                piece, asm.parity = hit[1], hit[2]
            else:
                piece = asm.assemble(items[a:b])
                self._keep(ck, (items, piece, asm.parity))
            assert piece.nbytes <= PIECE_BYTES, piece.nbytes
            bo = self._piece_bos[slot]
            np.copyto(bo.map()[:piece.nbytes], piece.view(np.uint8))
            bo.sync_to_device(piece.nbytes, 0)
            self._held[slot] = (key, piece.nbytes, asm.parity) if key is not None else None
            if hooks and a in hooks:
                hooks[a]()
            if k:
                wait()  # (piece k - 1; meanwhile piece k was assembled)
            submit, pkey = time.perf_counter(), (id(items), a, b)
            kern.dispatch_elf_async(bo, refs, piece.nbytes)
        if key is not None and asm.parity != slot:
            self._ahead(items, key, asm)
        elif asm.parity != starts[0]:  # (the next run's pieces start at the other parities: assembled while this one ends)
            end = asm.parity
            for (a, b), p in zip(itertools.pairwise(bounds), starts):
                ck = (id(items), a, b, p ^ 1)
                hit = self._pieces.get(ck)
                if hit is None or hit[0] is not items:
                    asm.parity = p ^ 1
                    piece = asm.assemble(items[a:b])
                    self._keep(ck, (items, piece, asm.parity))
            asm.parity = end
        wait()
        return k + 1

    def _ahead(self, items: Sequence[Item], key: tuple, asm: Assembler) -> None:
        """While a program of one piece runs from the BO of one parity: the program again from the other (its
        configurations' PDI slots swapped), assembled into that BO, so that a run again needs neither (a prompt's
        second chunk of a size: 6 ms)."""
        p = asm.parity
        key = (key[0], p)  # (the key of a run that starts at this parity)
        held = self._held[p]
        if held is not None and held[0] == key:
            return
        ck = (id(items), 0, len(items), p)
        hit = self._pieces.get(ck)
        if hit is not None and hit[0] is items:
            piece, end = hit[1], hit[2]
        else:
            piece = asm.assemble(items)
            end, asm.parity = asm.parity, p
            self._keep(ck, (items, piece, end))
        bo = self._piece_bos[p]
        np.copyto(bo.map()[:piece.nbytes], piece.view(np.uint8))
        bo.sync_to_device(piece.nbytes, 0)
        self._held[p] = (key, piece.nbytes, end)
