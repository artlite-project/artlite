"""sd15's design on the NPU (``Sd15Device``): its hardware context, DDR buffers, the roles' code (``role_code``), and
dispatches of ``seq.Seq`` programs."""

import os
import struct
import subprocess
from dataclasses import dataclass

import numpy as np

from ..base import PEANO_DIR
from ..fused import ctrlpkt
from ..kernel_source import MARK
from ..overlay.runtime import OverlayDevice, Transaction
from . import core as K
from . import design as D
from .seq import Seq


@dataclass(frozen=True)
class Role:
    """A role's code (``core.ROLES``): its program-memory words from byte ``start`` (the end of the shared code) and its
    entry (the ext word's value)."""

    start: int
    words: np.ndarray
    entry: int


def _elf(path: str) -> tuple[dict[str, bytes], dict[str, int], dict[str, set[str]]]:
    """An ELF32 file's sections, symbol values by name and, from .rela.text (a link with --emit-relocs), per function
    the symbols its code refers to."""
    with open(path, "rb") as f:
        b = f.read()
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    hdr = [struct.unpack_from("<IIIIIIIIII", b, shoff + i * shentsize) for i in range(shnum)]

    def name(table: int, off: int) -> str:
        o = hdr[table][4] + off
        return b[o:b.index(b"\0", o)].decode()
    secs = {name(shstrndx, h[0]): b[h[4]:h[4] + h[5]] for h in hdr}
    syms, table, funcs = {}, [], []
    for h in hdr:
        if h[1] == 2:  # (SHT_SYMTAB: entries of 16 bytes, names in the section h[6] links)
            for o in range(h[4], h[4] + h[5], 16):
                st_name, st_value, st_size, st_info = struct.unpack_from("<IIIB", b, o)
                nm = name(h[6], st_name) if st_name else ""
                table.append(nm)
                if nm:
                    syms[nm] = st_value
                    if st_info & 15 == 2:  # (STT_FUNC)
                        funcs.append((st_value, st_value + st_size, nm))
    refs: dict[str, set[str]] = {}
    for h in hdr:
        if h[1] == 4 and name(shstrndx, h[0]) == ".rela.text":  # (SHT_RELA: offset, info (symbol << 8 | type), addend)
            for o in range(h[4], h[4] + h[5], 12):
                off, info = struct.unpack_from("<II", b, o)
                caller = next((f for lo, hi, f in funcs if lo <= off < hi), None)
                if caller:
                    refs.setdefault(caller, set()).add(table[info >> 8])
    return secs, syms, refs


def _stack_sizes(work_dir: str) -> dict[str, int]:
    """The kernel's functions' stack bytes (clang's -fstack-usage: ``kernel.su``) by symbol."""
    out = {}
    with open(os.path.join(work_dir, "kernel.su")) as f:
        for line in f:
            loc, n, _ = line.rstrip("\n").split("\t")
            out[loc.rsplit(":", 1)[-1]] = int(n)
    return out


def _depth(f: str, calls: dict[str, set[str]], su: dict[str, int], seen: tuple = ()) -> int:
    """The deepest stack from function f down its calls."""
    assert f not in seen, f"recursion through {f}"
    return su.get(f, 0) + max((_depth(g, calls, su, seen + (f,)) for g in calls.get(f, ()) if g in su), default=0)


def role_code(work_dir: str) -> dict[str, Role]:
    """Each role's code, from its ELF: core (0, 2)'s link with that role's main in place of its own (an alias in the
    core's linker script; ``role_<x>.r.elf`` in the design's project directory). The shared code and the data are
    identical in every role's ELF (asserted: the switch rewrites neither); a role's stack (the main loop's, every
    role function's and the largest shared function's: a bound, aiecc checks only the design's own ELFs) fits."""
    prj = os.path.join(work_dir, "design.mlir.prj")
    stem = K.P.removeprefix(MARK)  # (the linked names: <stem>_<name>_<hash>)
    base = D.ROLE[0]
    own = os.path.join(prj, "elfs_main_core_0_2", "elfs_main_core_0_2.elf")
    mains = {r: next(n for n in _elf(own if r == base else os.path.join(work_dir, "kernel.o"))[1]
                     if n.startswith(f"{stem}_main_{r}_")) for r in K.ROLES}
    own = os.path.join(prj, "role_base.r.elf")  # (the base role's ELF again, with the relocations: its call graph)
    elfs = {}
    for r in K.ROLES:
        path = own if r == base else os.path.join(prj, f"role_{r}.r.elf")
        if not os.path.exists(path):
            with open(os.path.join(prj, "ldScripts_main_core_0_2.ld.script")) as f:
                script = f.read() + (f"\n{mains[base]} = {mains[r]};\n" if r != base else "")
            tmp = f"{path}.{os.getpid()}"
            with open(tmp + ".ld", "w") as f:
                f.write(script)
            subprocess.run([os.path.join(PEANO_DIR, "bin", "clang"), "-O2", "--target=aie2p-none-unknown-elf",
                            "-fuse-ld=lld", os.path.join(prj, "objects_main_core_0_2", "objects_main_core_0_2.o"),
                            "-Wl,--gc-sections", "-Wl,--emit-relocs", "-Wl,--orphan-handling=error", f"-Wl,-T,{tmp}.ld",
                            "-o", tmp],
                           check=True)
            os.replace(tmp, path)
            os.remove(tmp + ".ld")
        elfs[r] = _elf(path)
    start = {elfs[r][1][mains[r]] for r in K.ROLES}
    assert len(start) == 1, f"the roles' shared code ends at {start}"
    end = start.pop()
    first = min(v for r in K.ROLES for n, v in elfs[r][1].items() if f"{stem}_" in n and v < end)  # (kernel.o's code)
    ref = elfs[base][0]
    su = _stack_sizes(work_dir)
    run = next(n for n in su if f"{stem}_run_" in n)
    out = {}
    for r, (secs, syms, refs) in elfs.items():
        assert secs[".text"][first:end] == ref[".text"][first:end] and secs[".data"] == ref[".data"], r
        entry = next(n for n in syms if f"{stem}_ext_{r}_" in n)
        calls = dict(refs)
        calls[run] = calls.get(run, set()) | {entry}  # (the main loop calls the role's entry through ext)

        stack = _depth(mains[r], calls, su)
        assert stack <= D.STACK, f"role {r}: its code's stack reaches {stack} B (the cores': {D.STACK})"
        code = secs[".text"][end:]
        out[r] = Role(end, np.frombuffer(code + bytes(-len(code) % 4), np.uint32), syms[entry])
    return out


def role_packets(role: Role, row: int) -> np.ndarray:
    """A switch of core row ``row`` to ``role``: control packets (packet id ``design.CTRL_ID[row]``, the row's cores)
    of its code, then of the ext words (its entry, the marker 1: ``seq.Seq.role``)."""
    return ctrlpkt.packets(D.CTRL_ID[row], [(D.PM + role.start, role.words),
                                            (D.L1["ext"], np.array([role.entry, 1], np.uint32))])


class Sd15Device:
    """The compiled design and its DDR images by name (``alloc`` / ``write`` / ``read`` / ``address``), sub-allocated
    from a few large host buffers (``CHUNK`` bytes each: a dispatch lists every buffer, and the driver's command holds
    some 500)."""

    CHUNK = 64 << 20

    def __init__(self, design: D.Sd15Design | None = None):
        self.design = design or D.Sd15Design()
        self.dev = OverlayDevice(self.design)
        self._images: dict[str, tuple[str, int, int]] = {}  # (name -> chunk, offset, bytes)
        self._top = self.CHUNK  # (the bytes used of the newest chunk)
        self.roles = role_code(os.path.dirname(self.dev._prog.insts_path))
        nop = np.zeros(D.Q // 4, np.int32)
        nop[K.C_KIND] = K.K_NOP
        self.alloc("_nop", D.Q)
        self.write("_nop", nop.view(np.uint8))
        self.ctrl = {}
        for role, code in self.roles.items():
            for r in range(len(D.ROLE)):
                pk = role_packets(code, r)
                self.alloc(f"_role_{role}_{r}", pk.nbytes)
                self.write(f"_role_{role}_{r}", pk)
                self.ctrl[(role, r)] = (self.address(f"_role_{role}_{r}"), pk.nbytes)

    def has(self, name: str) -> bool:
        return name in self._images

    def alloc(self, name: str, nbytes: int) -> np.ndarray:
        """A DDR image of at least nbytes (an existing one of that name if it is large enough; else a new one), zeroed
        when new; returns its host view."""
        nbytes = -(-nbytes // 4096) * 4096
        if name in self._images and self._images[name][2] >= nbytes:
            return self._view(name)
        if nbytes > self.CHUNK:  # (a buffer of its own)
            chunk = f"_big{len(self.dev.buffers)}"
            self.dev.alloc(chunk, nbytes)
            self._images[name] = (chunk, 0, nbytes)
            return self._view(name)
        if self._top + nbytes > self.CHUNK:
            self.dev.alloc(f"_chunk{len(self.dev.buffers)}", self.CHUNK)
            self._chunk, self._top = f"_chunk{len(self.dev.buffers) - 1}", 0
        self._images[name] = (self._chunk, self._top, nbytes)
        self._top += nbytes
        return self._view(name)

    def _view(self, name: str) -> np.ndarray:
        chunk, off, n = self._images[name]
        return self.dev.view(chunk)[off:off + n]

    def view(self, name: str) -> np.ndarray:
        """An image's host bytes (written there: then ``flush``)."""
        return self._view(name)

    def flush(self, name: str) -> None:
        chunk, off, n = self._images[name]
        self.dev.flush(chunk, off, n)

    def write(self, name: str, data: np.ndarray, offset: int = 0) -> None:
        """Host bytes -> an image (flushed)."""
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        chunk, off, n = self._images[name]
        assert offset + data.size <= n, (name, offset, data.size, n)
        self.dev.view(chunk)[off + offset:off + offset + data.size] = data
        self.dev.flush(chunk, off + offset, data.size)

    def read(self, name: str, nbytes: int | None = None, offset: int = 0) -> np.ndarray:
        chunk, off, n = self._images[name]
        nbytes = n - offset if nbytes is None else nbytes
        self.dev.fetch(chunk, off + offset, nbytes)
        return self.dev.view(chunk)[off + offset:off + offset + nbytes].copy()

    def address(self, name: str, offset: int = 0) -> int:
        chunk, off, _ = self._images[name]
        return self.dev.address(chunk, off + offset)

    def seq(self) -> Seq:
        return Seq(nop=self.address("_nop"), ctrl=self.ctrl)

    def transaction(self, seq: Seq) -> Transaction:
        return Transaction(self.dev, seq.finish())

    def run(self, tr: Transaction) -> None:
        self.dev.dispatch(tr)

    def layout(self) -> dict:
        """The DDR layout: the host buffers in creation order (name, bytes), per image its buffer, offset and bytes
        (``restore`` rebuilds it on another device)."""
        return {"buffers": [(n, self.dev.view(n).nbytes) for n in self.dev.buffers], "images": dict(self._images),
                "chunk": getattr(self, "_chunk", None), "top": self._top}

    def restore(self, layout: dict) -> None:
        """``layout()`` of a device built from the same design: the buffers this one lacks allocated in that order (the
        same names), the images where they were (their bytes zero: the owner writes them)."""
        for n, nbytes in layout["buffers"]:
            if n not in self.dev.buffers:
                self.dev.alloc(n, nbytes)
        self._images = dict(layout["images"])
        self._chunk, self._top = layout["chunk"], layout["top"]

    def relocations(self, addrs: list[tuple[int, int]]) -> np.ndarray:
        """Shim BD addresses of a transaction (``txn.Txn.addrs``) as [word index, buffer number, offset] rows: where
        ``relocate`` writes them for another device's buffers."""
        names = list(self.dev.buffers)
        base = np.array([self.dev.address(n) for n in names], np.int64)
        order = np.argsort(base)
        a = np.array([x for _, x in addrs], np.int64)
        k = order[np.searchsorted(base[order], a, side="right") - 1]
        size = np.array([self.dev.view(n).nbytes for n in names], np.int64)
        assert ((a >= base[k]) & (a < base[k] + size[k])).all(), "a shim address outside the buffers"
        return np.stack([np.array([i for i, _ in addrs], np.int64), k, a - base[k]], 1)

    def relocate(self, words: np.ndarray, rel: np.ndarray) -> np.ndarray:
        """words with the addresses of ``relocations`` rows rewritten for this device's buffers (a copy)."""
        base = np.array([self.dev.address(n) for n in self.dev.buffers], np.int64)
        a = base[rel[:, 1]] + rel[:, 2]
        out = words.copy()
        out[rel[:, 0]] = (a & 0xFFFFFFFF).astype(np.uint32)
        out[rel[:, 0] + 1] = (a >> 32).astype(np.uint32)
        return out

    def warm(self) -> None:
        """An empty dispatch: the first one after new host buffers takes ~0.15 s more than later ones (a model's load
        ends with it, so that its first step does not)."""
        self.run(self.transaction(self.seq()))
