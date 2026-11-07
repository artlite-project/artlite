"""The locks a library's cores acquire in a neighbour's memory module (``neighbour_locks``): a configuration without the
array reset (``program``) holds those cores in reset and flushes those locks first.

A core waiting on a lock when its design ends leaves the acquire pending in the lock's module; holding the core in
reset does not cancel it, and the first release of that lock by a later design is taken by it (a hang: the swallowed
release's object never reaches its consumer). A tile's own locks are flushed by its control packets
(``ctrlpkt.Loader(reused=True)``) after its core's reset, but a neighbour's packets may arrive before the waiting core's
(a core north of the lock, or in another column), so those cores are held in reset and their locks flushed before any
packet.

The lock ids come from the cores' code: aiecc leaves each core's object in the build directory
(``design.mlir.prj/objects_op<entry>_<class>_core_<col>_<row>/``), and its acquires name their lock by an immediate (a
core tile's 0-15 the south module's, 16-31 the west's, 32-47 the north's, 48-63 its own: mlir-aie's
``getLockLocalBaseIndex``).
"""

import json
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_OBJECTS = re.compile(r"objects_op(\d+)_\w+?_core_(\d+)_(\d+)$")
_ACQUIRE = re.compile(r"\bacq\s+#0x([0-9a-f]+)")
_FILE = re.compile(r"^(\S+\.o):\s+file format", re.MULTILINE)
_DIRECTION = {0: (0, -1), 1: (-1, 0), 2: (0, 1)}  # (lock id // 16: south, west, north: (col, row) offsets)
_ARGS = 16384  # characters of object paths per llvm-objdump (Windows' command line: 32767 at most)
FILE = "neighbour_locks.json"

Census = dict[int, list[tuple[tuple[int, int], tuple[int, int], int]]]  # (entry -> [(core tile, lock's tile, lock)])


def _scan(prj: str, paths: list[str]) -> dict[str, set[int]]:
    """Each object's acquired lock ids (immediates): one llvm-objdump for all of ``paths`` (relative to ``prj``: no
    spaces in the file heads ``_FILE`` reads)."""
    from ..base import peano_tool

    text = subprocess.run([peano_tool("llvm-objdump"), "-d", "--triple=aie2p", *paths], cwd=prj, capture_output=True,
                          text=True, check=True).stdout
    heads = list(_FILE.finditer(text))
    return {m[1]: {int(x, 16) for x in _ACQUIRE.findall(text, m.end(), n.start() if n else len(text))}
            for m, n in zip(heads, heads[1:] + [None])}


def census(work_dir: str) -> Census | None:
    """Per library entry, its cores' acquires of a neighbour's lock, from the build directory's per-core objects
    (cached in ``FILE``); None when the objects are not there."""
    path = os.path.join(work_dir, FILE)
    try:
        with open(path) as f:
            return {int(e): [((c, r), (lc, lr), k) for c, r, lc, lr, k in v] for e, v in json.load(f).items()}
    except (OSError, ValueError):
        pass
    prj = os.path.join(work_dir, "design.mlir.prj")
    cores = {}
    for name in os.listdir(prj) if os.path.isdir(prj) else ():
        if (m := _OBJECTS.match(name)) is not None:
            for f in os.listdir(os.path.join(prj, name)):
                if f.endswith(".o"):
                    cores[os.path.join(name, f)] = (int(m[1]), int(m[2]), int(m[3]))
    if not cores:
        return None
    parts, size = [[]], 0
    for p in sorted(cores):
        if parts[-1] and size + len(p) >= _ARGS:
            parts.append([])
            size = 0
        parts[-1].append(p)
        size += len(p) + 1
    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        found = {}
        for part in ex.map(lambda part: _scan(prj, part), parts):
            found.update(part)
    out: Census = defaultdict(list)
    for p, (e, c, r) in cores.items():
        for k in sorted(found.get(p, ())):
            if k < 48:
                dc, dr = _DIRECTION[k // 16]
                out[e].append(((c, r), (c + dc, r + dr), k % 16))
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({e: [[*t, *lt, k] for t, lt, k in v] for e, v in out.items()}, f)
    os.replace(tmp, path)
    return dict(out)
