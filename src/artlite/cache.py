"""The cache directory (``ARTLITE_CACHE_DIR``, default ``~/.cache/artlite``): its layout, the last use of its entries,
their cleaning and packing.

The root may be shared with other files, because a user may point the variable at any directory. So Artlite writes only
two kinds of directories there, and marks each with a ``CACHEDIR.TAG`` that carries Artlite's line (``owned``):

- ``tc-<id>/`` holds what one toolchain built: builds (``rtc_<key>/``, ``fused_<key>/``), precompiled headers
  (``pch/``), the render manifest (``render/``), the vision networks' states (``vision/``), the build locks (``locks/``)
  and the tools' caches (``ledger/``, ``variant_search/``). ``<id>`` is a hash of the llvm-aie and mlir-aie wheels'
  contents (their ``RECORD`` files, ``toolchain_id``): a reinstall of the same wheels, or another venv with them, finds
  the same directory, and a locally built wheel with the same version string gets another one. The keys inside are
  content-only (no tool modification times, no absolute paths: ``operators.build``), so a moved cache keeps them.
- ``common/`` holds what no toolchain builds: math fits (``kmath/``), the goldens' tools, the server's model index and
  web UI, the pace calibration.

The tag never goes at the root: tar ``--exclude-caches``, borg and restic skip a directory that has one, which must
not happen to a user's directory that holds the cache too.

The last use of an entry is its modification time (``touch``), because the filesystem may keep no access times
(``noatime``). A directory entry (a build, a vision state) is stamped through its ``entry.json`` (``write_entry``):
the directory's own time changes whenever files are created or deleted in it.

Cleaning (``clean``, ``artlite-cache clean``) removes entries by the rules of a ``Policy`` (``select``): superseded
(an older version of a design whose identity was built again since, ``_superseded``) or unused for a while, of the
layout before the owned directories (the ``legacy`` kinds, whose state ``adopt_legacy`` moves into ``common/`` first),
of other toolchains, the least recently used above a size; always the trash, and leftovers of killed writers and failed
builds an hour old. It also packs the intermediates of builds unused for a while (``pack``: the files ``cold_files``
names, which the runtime never reads, into the build's ``temps.tar.xz``; ``unpack`` restores them). A process that
built something starts the automatic cleaning at its exit, once a day per cache (``note_build``: a detached child at the
lowest priority, ``ARTLITE_CACHE_GC`` its policy, ``common/gc.json`` its last run). What a cleaning may touch is the
table ``KINDS``, and nothing else:

- at the root, only ``tc-<id>`` and ``common`` directories with Artlite's tag, and the legacy kinds' exact names;
- below an owned directory, only names that match a kind's components (whole-name regular expressions);
- never a link: a link named like an entry is skipped, a link inside a removed entry is unlinked, not followed; nor an
  entry on another filesystem than its directory;
- an entry with a writer's lock (a build) only when the lock is free, taken without waiting; the entry is renamed into
  its owned directory's ``trash/`` under the lock, then deleted (an interrupted cleaning leaves only trash).
"""

import atexit
import contextlib
import functools
import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import metadata

from . import portable, settings

ROOT = settings.CACHE_DIR.get()

MARKER = "CACHEDIR.TAG"
_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"  # (https://bford.info/cachedir/)
_LINE = "# Artlite cache: artlite-cache manages this directory."
TAG = f"{_SIGNATURE}\n{_LINE}\n# For information about cache directory tags, see https://bford.info/cachedir/\n"
TOOLCHAIN_JSON = "toolchain.json"
ENTRY_JSON = "entry.json"
COMMON = "common"
TC_PREFIX = "tc-"
WHEELS = ("llvm_aie", "mlir_aie")  # (the toolchain: Peano and aiecc)
TOUCH_S = 600  # a last use is recorded at most this often per entry
IN_PROGRESS = ".building"  # (in a build directory until its build succeeded: ``operators.build``)


def record_files(record: str) -> list[str]:
    """The lines of a wheel's ``RECORD`` that name its package's own files with their hashes: not the entry scripts
    in ``../bin`` (their shebang names the venv), nor the dist-info (its INSTALLER, REQUESTED and direct_url.json
    describe the install), nor files an installer added without a hash (bytecode)."""
    return sorted(line for line in record.splitlines()
                  if ",sha256=" in line and not line.startswith("..") and ".dist-info/" not in line.split(",", 1)[0])


@functools.cache
def _toolchain() -> tuple[str, dict[str, str | None]]:
    """The toolchain's id and the wheels' versions."""
    h = hashlib.sha256()
    versions: dict[str, str | None] = {}
    for name in WHEELS:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            h.update(f"{name} -\n".encode())
            continue
        versions[name] = dist.version
        h.update(f"{name} {dist.version}\n".encode())
        lines = record_files(dist.read_text("RECORD") or "")
        if lines:
            h.update("\n".join(lines).encode())
        else:  # (no wheel metadata: the tools' modification times, as a reinstall changes them)
            from .operators import toolchain
            h.update(repr((os.path.getmtime(toolchain.clangpp()), os.path.getmtime(toolchain.aiecc()))).encode())
    return h.hexdigest()[:12], versions


def toolchain_id() -> str:
    """The id of this process's toolchain: a hash of the llvm-aie and mlir-aie wheels' versions and files."""
    return _toolchain()[0]


def is_tag(path: str) -> bool:
    """Whether ``path`` is a regular file (not a link) with Artlite's cache directory tag."""
    try:
        if not os.path.isfile(path) or os.path.islink(path):
            return False
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(1024)
    except OSError:
        return False
    return head.startswith(_SIGNATURE) and _LINE in head


def _own(path: str, info: dict | None = None) -> None:
    """Create the owned directory ``path`` with its tag (and ``info`` as its ``toolchain.json``) unless it has one."""
    tag = os.path.join(path, MARKER)
    if os.path.isfile(tag):
        return
    os.makedirs(path, exist_ok=True)
    if info is not None and not os.path.exists(os.path.join(path, TOOLCHAIN_JSON)):
        _write_json(os.path.join(path, TOOLCHAIN_JSON), info)
    _write_text(tag, TAG)


def toolchain_dir(*parts: str) -> str:
    """The path ``parts`` in this toolchain's directory (created, with its tag, on first use)."""
    tc_id, versions = _toolchain()
    path = os.path.join(ROOT, TC_PREFIX + tc_id)
    _own(path, {**versions, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    touch(os.path.join(path, TOOLCHAIN_JSON))
    return os.path.join(path, *parts)


def common(*parts: str) -> str:
    """The path ``parts`` in the directory of what no toolchain builds (created, with its tag, on first use)."""
    path = os.path.join(ROOT, COMMON)
    _own(path)
    return os.path.join(path, *parts)


def touch(path: str) -> None:
    """Record a use of the entry ``path`` (a file: an entry, or a directory entry's ``entry.json``): its modification
    time, at most every ``TOUCH_S`` seconds. A missing or read-only file is left as it is."""
    now = time.time()
    with contextlib.suppress(OSError):
        if os.stat(path).st_mtime < now - TOUCH_S:
            os.utime(path, (now, now))


def write_entry(directory: str, kind: str, label: str) -> None:
    """The ``entry.json`` of a directory entry: what it is (kind, label: an ELF's or operator's name) and, by its
    modification time, its last use."""
    _write_json(os.path.join(directory, ENTRY_JSON),
                {"kind": kind, "label": label, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})


def touch_entry(directory: str) -> None:
    """Record a use of a directory entry (``write_entry``)."""
    touch(os.path.join(directory, ENTRY_JSON))


@contextlib.contextmanager
def lock(name: str) -> Iterator[None]:
    """The build lock ``name`` of this toolchain's directory (``locks/<name>.lock``), held meanwhile."""
    path = toolchain_dir("locks", f"{name}.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with portable.file_lock(path):
        yield


def adopt(path: str, *legacy: str) -> str:
    """``path`` (in an owned directory), first moved there from the root's ``legacy`` path (where the cache kept it
    before its owned directories) when only that exists, reached without links."""
    old = os.path.join(ROOT, *legacy)
    real = all(not os.path.islink(os.path.join(ROOT, *legacy[:i + 1])) for i in range(len(legacy)))
    if real and not os.path.lexists(path) and os.path.lexists(old):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with contextlib.suppress(OSError):
            os.replace(old, path)
    return path


def _write_text(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"  # (the build pool's threads may write one tag at once)
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_json(path: str, data: dict) -> None:
    _write_text(path, json.dumps(data))


# ---------------------------------------------------------------------------
# The entries, as a cleaning reads them
# ---------------------------------------------------------------------------

_H = "[0-9a-f]"
DAY = 86400.0
TEMP_S = 3600.0  # a leftover of a killed writer (or a failed build, a free lock) older than this is removed
RECENT_S = 3600.0  # the size rule never removes an entry used this recently
_VISION = "(minicpmv|qwen3vl|lfm3vl|lfm3a|moss|inkling2v|sd15)"  # (the legacy layout's vision state directories)
_GOLDENS = "(qwen35_dump|minicpmv_dump|lfm3a_gen_dump)"  # (the legacy layout's goldens tools)


@dataclass(frozen=True)
class Kind:
    """A kind of entry. ``where``: ``tc`` (a toolchain's directory), ``common``, or ``legacy`` (the root, in the layout
    before the owned directories); ``parts``: its path's components below that directory (regular expressions that
    match whole names); ``directory``: whether the entry is a directory (None: either); ``stamp``: the file in a
    directory entry whose modification time is its last use (None: the entry's own); ``lock``: the lock file its writer
    holds, a path below the same directory (``{name}``: the entry's name, ``{stem}``: its name up to the first dot);
    ``role``: ``entry`` (the rules decide), ``state`` (only ``all`` removes it), ``temp`` (a killed writer's leftover or
    a lock file: removed ``TEMP_S`` after its last change) or ``trash`` (removed by every cleaning)."""

    name: str
    where: str
    parts: tuple[str, ...]
    directory: bool | None = False
    stamp: str | None = None
    lock: str | None = None
    role: str = "entry"


KINDS = (
    # a toolchain's directory
    Kind("rtc", "tc", (rf"rtc_{_H}{{16}}",), True, ENTRY_JSON, "locks/{name}.lock"),
    Kind("fused", "tc", (rf"fused_{_H}{{16}}",), True, ENTRY_JSON, "locks/{name}.lock"),
    Kind("pch", "tc", ("pch", rf"{_H}{{16}}\.pch"), lock="locks/pch_{stem}.lock"),
    Kind("render", "tc", ("render", rf"{_H}{{32}}\.json")),
    Kind("vision", "tc", ("vision", r"[a-z0-9_]+", rf"{_H}{{12}}", rf"{_H}{{16}}"), True, ENTRY_JSON),
    Kind("ledger", "tc", ("ledger", rf"(tu|cores_rtc|cores_fused)_{_H}{{16}}\.json")),
    Kind("variant_search", "tc", ("variant_search", rf"{_H}{{12}}"), True),
    Kind("temp", "tc", ("pch", rf"{_H}{{16}}\.tmp\.pch"), role="temp"),
    Kind("temp", "tc", ("render", rf"{_H}{{32}}\.json\.\d+\.\d+\.tmp"), role="temp"),
    Kind("temp", "tc", ("vision", r"[a-z0-9_]+", rf"{_H}{{12}}", rf"{_H}{{16}}\.\d+"), True, role="temp"),
    Kind("temp", "tc", ("ledger", r"\w+\.json\.\d+\.tmp"), role="temp"),
    Kind("temp", "tc", ("ledger", "work", rf"{_H}{{16}}"), True, role="temp"),
    Kind("temp", "tc", (r"(CACHEDIR\.TAG|toolchain\.json)\.\d+\.\d+\.tmp",), role="temp"),
    Kind("lock", "tc", ("locks", r"[\w.-]+\.lock"), lock="locks/{name}", role="temp"),
    Kind("trash", "tc", ("trash", r".+"), None, role="trash"),
    # common
    Kind("kmath", "common", ("kmath", rf"\w+_{_H}{{16}}\.npy")),
    Kind("goldens", "common", ("goldens", rf"\w+_{_H}{{16}}")),
    Kind("ui", "common", ("ui", r"(?!tmp)[\w.-]+"), True, "index.html"),
    Kind("pace", "common", ("pace.json",), role="state"),
    Kind("server", "common", ("server", "models.json"), role="state"),
    Kind("temp", "common", ("kmath", rf"\w+_{_H}{{16}}\.npy\.\d+\.npy"), role="temp"),
    Kind("temp", "common", ("goldens", rf"\w+_{_H}{{16}}\.\d+\.tmp"), role="temp"),
    Kind("temp", "common", ("ui", r"tmp\w{8}"), True, role="temp"),
    Kind("temp", "common", ("server", r"\.models\.\w+"), role="temp"),
    Kind("temp", "common", (r"(pace\.json\.\d+|(CACHEDIR\.TAG|gc\.json)\.\d+\.\d+\.tmp)",), role="temp"),
    Kind("trash", "common", ("trash", r".+"), None, role="trash"),
    # the legacy layout, at the root
    Kind("rtc", "legacy", (rf"rtc_{_H}{{16}}",), True, lock="{name}.lock"),
    Kind("fused", "legacy", (rf"fused_{_H}{{16}}",), True, lock="{name}.lock"),
    Kind("lock", "legacy", (rf"(rtc|fused)_{_H}{{16}}\.lock",), lock="{name}"),
    Kind("pch", "legacy", ("operators", rf"{_H}{{16}}(\.tmp)?\.pch"), lock="operators/{stem}.pch.lock"),
    Kind("lock", "legacy", ("operators", rf"{_H}{{16}}\.pch\.lock"), lock="operators/{name}"),
    Kind("render", "legacy", ("render", rf"{_H}{{32}}\.json(\.\d+\.\d+\.tmp)?")),
    Kind("kmath", "legacy", ("kmath", rf"\w+_{_H}{{16}}\.npy(\.\d+\.npy)?")),
    Kind("ledger", "legacy", ("ledger", rf"(tu|cores_rtc|cores_fused)_{_H}{{16}}\.json(\.\d+\.tmp)?")),
    Kind("ledger", "legacy", ("ledger", "work", rf"{_H}{{16}}"), True),
    Kind("vision", "legacy", (_VISION, rf"{_H}{{12}}", rf"{_H}{{16}}(\.\d+)?"), True),
    Kind("variant_search", "legacy", ("variant_search", rf"{_H}{{12}}"), True),
    Kind("goldens", "legacy", (rf"{_GOLDENS}_{_H}{{16}}(\.\d+\.tmp)?",)),
    Kind("ui", "legacy", ("ui", r"[\w.-]+"), True),
    Kind("pace", "legacy", (r"pace\.json(\.\d+)?",)),
    Kind("server", "legacy", ("server", r"models\.json|\.models\.\w+")),
)
_STRUCTURE = {"tc": (MARKER, TOOLCHAIN_JSON), "common": (MARKER, "gc.json", "gc.lock")}
# (directories a cleaning removes once empty, deepest first: below an owned directory, and the legacy layout's)
_DIRS = {"tc": ("pch", "render", "vision/*/*", "vision/*", "vision", "ledger/work", "ledger", "variant_search",
                "locks", "trash"),
         "common": ("kmath", "goldens", "ui", "server", "trash"),
         "legacy": ("operators", "render", "kmath", "ledger/work", "ledger", "variant_search", "ui", "server",
                    f"{_VISION}/*", _VISION)}


def _usage(st: os.stat_result) -> int:
    return st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size


def disk_usage(path: str) -> int:
    """The bytes ``path`` takes on disk, links not followed."""
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    total = _usage(st)
    if stat.S_ISDIR(st.st_mode):
        for top, dirs, files in os.walk(path):
            for name in dirs + files:
                with contextlib.suppress(OSError):
                    total += _usage(os.lstat(os.path.join(top, name)))
    return total


@dataclass
class Owned:
    """An owned directory: ``where`` (``tc``, ``common``), its toolchain's versions and last use (``tc``)."""

    path: str
    where: str
    versions: dict = field(default_factory=dict)
    used: float = 0.0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Entry:
    """An entry of the cache: its kind, path, the directory its kind's parts start at (an owned directory, or the root
    for the legacy layout) and its last use."""

    kind: Kind
    path: str
    base: str
    used: float
    _size: int | None = None

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = disk_usage(self.path)
        return self._size

    @property
    def lock_path(self) -> str | None:
        if self.kind.lock is None:
            return None
        name = os.path.basename(self.path)
        return os.path.join(self.base, self.kind.lock.format(name=name, stem=name.split(".", 1)[0]))

    @property
    def failed(self) -> bool:
        """A build directory whose build did not finish."""
        return self.kind.name in ("rtc", "fused") and os.path.exists(os.path.join(self.path, IN_PROGRESS))

    def info(self) -> dict:
        """What a directory entry's ``entry.json`` or a manifest says about it ({}: nothing)."""
        path = os.path.join(self.path, ENTRY_JSON) if self.kind.stamp == ENTRY_JSON else \
            self.path if self.kind.name == "render" else None
        if path is None:
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


@dataclass
class Scan:
    """What a cache holds: its owned directories and entries, what was skipped (links, other filesystems, directories
    without Artlite's tag: path and why) and the names in owned directories no kind matches."""

    root: str
    owned: list[Owned] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def _mtime(path: str) -> float | None:
    try:
        return os.lstat(path).st_mtime
    except OSError:
        return None


def _is_link(e: os.DirEntry) -> bool:
    return e.is_symlink() or e.is_junction()


def _walk(base: str, kind: Kind, scan: Scan, *, dev: int, parts: tuple[str, ...] | None = None,
          top: str | None = None) -> None:
    """The entries of ``kind`` below ``base`` (``parts``: the components left, ``top``: the directory scanned)."""
    parts = kind.parts if parts is None else parts
    top = base if top is None else top
    pattern = re.compile(parts[0])
    try:
        it = os.scandir(top)
    except OSError:
        return
    with it:
        for e in it:
            if not pattern.fullmatch(e.name):
                continue
            if _is_link(e):
                scan.skipped.append((e.path, "a link"))
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            if st.st_dev != dev:
                scan.skipped.append((e.path, "another filesystem"))
                continue
            is_dir = stat.S_ISDIR(st.st_mode)
            if len(parts) > 1:
                if is_dir:
                    _walk(base, kind, scan, dev=dev, parts=parts[1:], top=e.path)
                continue
            if kind.directory is not None and is_dir != kind.directory:
                continue
            used = _mtime(os.path.join(e.path, kind.stamp)) if kind.stamp else None
            scan.entries.append(Entry(kind, e.path, base, used if used is not None else st.st_mtime))


def scan(root: str | None = None) -> Scan:
    """The owned directories and entries of the cache at ``root`` (default: ``ROOT``), links never followed."""
    root = ROOT if root is None else root
    out = Scan(root)
    try:
        dev = os.lstat(root).st_dev
        names = list(os.scandir(root))
    except OSError:
        return out
    for e in names:
        where = "tc" if re.fullmatch(rf"{TC_PREFIX}{_H}{{12}}", e.name) else "common" if e.name == COMMON else None
        if where is None:
            continue
        if _is_link(e) or not e.is_dir(follow_symlinks=False):
            out.skipped.append((e.path, "a link" if _is_link(e) else "not a directory"))
        elif e.stat(follow_symlinks=False).st_dev != dev:
            out.skipped.append((e.path, "another filesystem"))
        elif not is_tag(os.path.join(e.path, MARKER)):
            out.skipped.append((e.path, f"no {MARKER} of Artlite's"))
        else:
            owned = Owned(e.path, where)
            if where == "tc":
                with contextlib.suppress(OSError, ValueError), open(os.path.join(e.path, TOOLCHAIN_JSON)) as f:
                    owned.versions = json.load(f)
                owned.used = _mtime(os.path.join(e.path, TOOLCHAIN_JSON)) or 0.0
            out.owned.append(owned)
    for owned in out.owned:
        kinds = [k for k in KINDS if k.where == owned.where]
        for kind in kinds:
            _walk(owned.path, kind, out, dev=dev)
        firsts = [re.compile(k.parts[0]) for k in kinds]
        with contextlib.suppress(OSError):
            out.unknown += sorted(os.path.join(owned.path, n) for n in os.listdir(owned.path)
                                  if n not in _STRUCTURE[owned.where] and not any(p.fullmatch(n) for p in firsts))
    for kind in KINDS:
        if kind.where == "legacy":
            _walk(root, kind, out, dev=dev)
    return out


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": DAY, "w": 7 * DAY}
_SIZES = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def parse_duration(text: str) -> float | None:
    """Seconds of ``30m``, ``12h``, ``1d``, ``2w`` (a bare number: seconds), None for ``off``."""
    text = text.strip().lower()
    if text in ("off", "never", "none"):
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhdw]?)", text)
    if m is None:
        raise ValueError(f"{text!r}: expected a duration such as 30m, 12h, 1d, 2w, or off")
    return float(m[1]) * _UNITS[m[2] or "s"]


def parse_size(text: str) -> int | None:
    """Bytes of ``500M``, ``30G`` (binary units; a bare number: bytes), None for ``off``."""
    text = text.strip().lower()
    if text in ("off", "none"):
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?)(i?b)?", text)
    if m is None:
        raise ValueError(f"{text!r}: expected a size such as 500M, 30G, or off")
    return int(float(m[1]) * _SIZES[m[2]])


@dataclass(frozen=True)
class Policy:
    """What a cleaning removes besides trash and leftovers: superseded entries not used for ``superseded`` seconds
    (``superseded``), entries unused for ``unused`` seconds, the legacy layout's entries unchanged for ``legacy``
    seconds, the other toolchains' directories (``other_toolchains``), the least recently used entries while the cache
    is above ``size`` bytes, or everything Artlite owns (``all``). Builds not used for ``pack`` seconds get their
    intermediate files packed (``pack``). A None rule is off. ``interval``: how often a process that built something
    starts the automatic cleaning (None: never)."""

    superseded: float | None = DAY
    unused: float | None = 30 * DAY
    legacy: float | None = DAY
    pack: float | None = DAY
    size: int | None = None
    interval: float | None = DAY
    other_toolchains: bool = False
    all: bool = False

    def describe(self) -> str:
        def d(x: float | None) -> str:
            if x is None:
                return "off"
            for unit, n in (("w", 7 * DAY), ("d", DAY), ("h", 3600), ("m", 60)):
                if x >= n and x % n == 0:
                    return f"{int(x // n)}{unit}"
            return f"{x:g}s"
        parts = [f"superseded={d(self.superseded)}", f"unused={d(self.unused)}"]
        if self.legacy != self.superseded:
            parts.append(f"legacy={d(self.legacy)}")
        parts.append(f"pack={d(self.pack)}")
        if self.size is not None:
            parts.append(f"size={self.size >> 20}M")
        return ",".join(parts)


_RULES = ("superseded", "unused", "legacy", "pack", "interval")


def parse_policy(text: str) -> Policy | None:
    """The automatic cleaning's policy of an ``ARTLITE_CACHE_GC`` value: ``auto`` (``Policy``'s defaults), ``off``
    (None), or rules that replace some defaults (``superseded=4h,size=30G``; ``legacy`` follows ``superseded``)."""
    text = text.strip().lower()
    if text == "off":
        return None
    fields: dict = {}
    for part in [] if text in ("", "auto") else text.split(","):
        name, _, value = (x.strip() for x in part.partition("="))
        if name in _RULES:
            fields[name] = parse_duration(value)
        elif name == "size":
            fields[name] = parse_size(value)
        else:
            raise ValueError(f"{settings.CACHE_GC.name}: {part!r} is not <rule>=<value> ({', '.join(_RULES)}, size)")
    if "superseded" in fields and "legacy" not in fields:
        fields["legacy"] = fields["superseded"]
    return Policy(**fields)


def policy() -> Policy:
    """The policy ``ARTLITE_CACHE_GC`` sets (``off``: the defaults, with no automatic cleaning)."""
    return parse_policy(settings.CACHE_GC.get()) or Policy(interval=None)


def _superseded(found: Scan, after: float, now: float) -> dict[str, str]:
    """The superseded entries (path -> reason) not used for ``after`` seconds. Per identity of a render (the manifest's
    ``identity``: its key without the sources' hash, the toolchain left out), the most recently used manifest wins; a
    build that manifests name, none of them a winner, is superseded, as are the losing manifests themselves, and
    manifests whose build is gone. A precompiled header not used for that long is too: one rebuilds in 2 s."""
    out: dict[str, str] = {}
    manifests: list[tuple[Entry, str, str]] = []
    for m in found.entries:
        if m.kind.name == "render" and m.kind.where == "tc":
            info = m.info()
            identity, work = info.get("identity"), info.get("work_dir")
            if isinstance(identity, str) and isinstance(work, str):
                manifests.append((m, identity, os.path.join(m.base, work)))
    winners: dict[str, Entry] = {}
    for m, identity, _ in manifests:
        if identity not in winners or m.used > winners[identity].used:
            winners[identity] = m
    named, won = set(), set()
    for m, identity, target in manifests:
        named.add(target)
        if winners[identity] is m:
            won.add(target)
            if not os.path.isdir(target) and m.used < now - TEMP_S:
                out[m.path] = "dangling"
        elif m.used < now - after:
            out[m.path] = "superseded"
    for e in found.entries:
        if e.kind.where != "tc" or e.used >= now - after:
            continue
        if (e.kind.name in ("rtc", "fused") and e.path in named and e.path not in won) or e.kind.name == "pch":
            out[e.path] = "superseded"
    return out


def select(found: Scan, policy: Policy, *, now: float | None = None) -> list[tuple[Entry, str]]:
    """The entries a cleaning with ``policy`` removes, each with its reason, in removal order."""
    now = time.time() if now is None else now
    current = TC_PREFIX + toolchain_id()
    superseded = _superseded(found, policy.superseded, now) if policy.superseded is not None else {}
    out: dict[str, tuple[Entry, str]] = {}
    keep: list[Entry] = []
    for e in found.entries:
        role, legacy = e.kind.role, e.kind.where == "legacy"
        tc = os.path.basename(e.base) if e.kind.where == "tc" else None
        if policy.all:
            out[e.path] = (e, "all")
        elif role == "trash":
            out[e.path] = (e, "trash")
        elif role == "temp" and e.used < now - TEMP_S:
            out[e.path] = (e, "leftover")
        elif e.failed and e.used < now - TEMP_S:
            out[e.path] = (e, "failed build")
        elif legacy and policy.legacy is not None and e.used < now - policy.legacy:
            out[e.path] = (e, "legacy layout")
        elif role in ("state", "temp") or legacy:
            continue
        elif policy.other_toolchains and tc is not None and tc != current:
            out[e.path] = (e, "other toolchain")
        elif e.path in superseded:
            out[e.path] = (e, superseded[e.path])
        elif policy.unused is not None and e.used < now - policy.unused:
            out[e.path] = (e, "unused")
        else:
            keep.append(e)
    if policy.size is not None:
        total = sum(e.size for e in found.entries if e.path not in out)
        for e in sorted(keep, key=lambda e: e.used):
            if total <= policy.size:
                break
            if e.used < now - RECENT_S:
                out[e.path] = (e, "size")
                total -= e.size
    return sorted(out.values(), key=lambda x: (x[0].kind.role != "trash", x[0].used))


@dataclass
class Report:
    """What a cleaning did: entries and bytes removed per reason, the entries left (path, why), errors."""

    removed: dict[str, list[int]] = field(default_factory=dict)  # reason -> [entries, bytes]
    packed: list[int] = field(default_factory=lambda: [0, 0])  # [entries, bytes freed]
    left: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    when: float = 0.0

    @property
    def entries(self) -> int:
        return sum(n for n, _ in self.removed.values())

    @property
    def bytes(self) -> int:
        return sum(b for _, b in self.removed.values())

    def to_json(self) -> dict:
        return {"when": self.when, "removed": self.removed, "packed": self.packed, "left": len(self.left),
                "errors": self.errors[:20]}


_counter = itertools.count()


def _delete(path: str, errors: list[str]) -> None:
    """Delete a renamed-away entry (links inside unlinked, never followed)."""
    def failed(fn: object, p: str, exc: BaseException) -> None:
        errors.append(f"{p}: {exc}")
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, onexc=failed)
        else:
            os.remove(path)
    except OSError as exc:
        errors.append(f"{path}: {exc}")


def _trash(e: Entry, found: Scan) -> str:
    base = e.base if e.kind.where != "legacy" else os.path.join(found.root, COMMON)
    if e.kind.where == "legacy":
        _own(base)
    return os.path.join(base, "trash")


def _remove(e: Entry, found: Scan, report: Report) -> bool:
    """Remove an entry: under its writer's lock (held by a build: left), renamed into its owned directory's trash, then
    deleted."""
    lock = e.lock_path
    if not os.path.lexists(e.path):
        return False
    if lock is not None:
        os.makedirs(os.path.dirname(lock), exist_ok=True)
    with portable.file_lock(lock, wait=False) if lock else contextlib.nullcontext(True) as held:
        if not held:
            report.left.append((e.path, "in use"))
            return False
        if os.path.islink(e.path) or not os.path.lexists(e.path):
            return False
        if e.kind.role == "trash":
            dest = e.path
        else:
            trash = _trash(e, found)
            os.makedirs(trash, exist_ok=True)
            dest = os.path.join(trash, f"{os.path.basename(e.path)}.{os.getpid()}.{next(_counter)}")
            try:
                os.rename(e.path, dest)
            except OSError as exc:  # (Windows: a file in it is open)
                report.left.append((e.path, str(exc)))
                return False
        if lock is not None and lock != e.path:
            with contextlib.suppress(OSError):
                os.remove(lock)
    _delete(dest, report.errors)
    return True


def _tidy(found: Scan, policy: Policy) -> None:
    """Remove the directories a cleaning emptied (never the root, never a link), and owned directories with nothing
    left: another toolchain's once unused for ``TEMP_S``, any with ``all``."""
    current = TC_PREFIX + toolchain_id()
    bases = [(o.path, o.where) for o in found.owned] + [(found.root, "legacy")]
    for base, where in bases:
        for rel in _DIRS[where]:
            for path in _matching(base, rel):
                with contextlib.suppress(OSError):
                    os.rmdir(path)
    now = time.time()
    for o in found.owned:
        if not (policy.all or (o.where == "tc" and o.name != current and o.used < now - TEMP_S)):
            continue
        try:
            names = set(os.listdir(o.path))
        except OSError:
            continue
        if names <= set(_STRUCTURE[o.where]):
            for n in sorted(names, key=lambda n: n == MARKER):  # (the tag last)
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(o.path, n))
            with contextlib.suppress(OSError):
                os.rmdir(o.path)


def _matching(base: str, rel: str) -> list[str]:
    """The real directories below ``base`` whose path matches ``rel`` (components: regular expressions, ``*``: any)."""
    out = [base]
    for part in rel.split("/"):
        pattern = re.compile(".+" if part == "*" else part)
        nxt = []
        for top in out:
            with contextlib.suppress(OSError), os.scandir(top) as it:
                nxt += [e.path for e in it if pattern.fullmatch(e.name) and not _is_link(e)
                        and e.is_dir(follow_symlinks=False)]
        out = nxt
    return out


def adopt_legacy() -> None:
    """Move the legacy layout's state (the pace calibrations, the server's model index, the web UI builds) into
    ``common/``, where nothing newer is there yet."""
    common()
    adopt(common("pace.json"), "pace.json")
    adopt(common("server", "models.json"), "server", "models.json")
    legacy_ui = os.path.join(ROOT, "ui")
    if os.path.isdir(legacy_ui) and not os.path.islink(legacy_ui):
        with os.scandir(legacy_ui) as it:
            names = [e.name for e in it if re.fullmatch(r"(?!tmp)[\w.-]+", e.name) and not _is_link(e)
                     and e.is_dir(follow_symlinks=False)]
        for name in names:
            adopt(common("ui", name), "ui", name)


@contextlib.contextmanager
def _clean_lock(wait: bool) -> Iterator[bool]:
    with portable.file_lock(common("gc.lock"), wait=wait) as held:
        yield held


def _clean(policy: Policy, *, dry_run: bool) -> Report:
    report = Report(when=time.time())
    if not dry_run:
        adopt_legacy()
    found = scan()
    report.skipped = found.skipped
    gone = set()
    for e, reason in select(found, policy, now=report.when):
        size = e.size
        if dry_run or _remove(e, found, report):
            gone.add(e.path)
            n = report.removed.setdefault(reason, [0, 0])
            n[0] += 1
            n[1] += size
    if policy.pack is not None:
        _pack_cold(found, policy.pack, report, dry_run=dry_run, skip=gone)
    if not dry_run:
        _tidy(found, policy)
    return report


# ---------------------------------------------------------------------------
# Packing a build's intermediate files
# ---------------------------------------------------------------------------

TEMPS = "temps.tar.xz"
# The files of a build directory the runtime never reads: the design's text and aiecc's intermediates. Paging reads the
# main cores' ELFs (elfs_main_*) and link scripts (ldScripts_main_*), the configuration checks cdo_main, the census
# and the frames objects_*, profiling input_with_addresses.mlir: those, and any name not listed here, stay plain.
_COLD = re.compile(r"design\.mlir")
_COLD_PRJ = re.compile(r"(peano-compat|peano-opted)_.+\.ll|canonical_[0-9A-F]+(\.ll)?|probe(Elfs|Scripts)_.+"
                       r"|ldScripts_op\d+_.+|cdo_(op\d+_.+|empty_\d+|closing)|(op\d+_\w+|empty_\d+|closing)\.pdi"
                       r"|bif_.+\.bif")


def cold_files(directory: str) -> list[str]:
    """The intermediate files present in a build directory (paths relative to it), which packing archives."""
    def names(top: str, pattern: re.Pattern) -> list[str]:
        if not os.path.isdir(top) or os.path.islink(top):
            return []
        with os.scandir(top) as it:
            return sorted(e.name for e in it if pattern.fullmatch(e.name) and not _is_link(e))
    prj = "design.mlir.prj"
    return names(directory, _COLD) + [f"{prj}/{n}" for n in names(os.path.join(directory, prj), _COLD_PRJ)]


def _cold_size(directory: str) -> int:
    return sum(disk_usage(os.path.join(directory, n)) for n in cold_files(directory))


def is_build_dir(path: str) -> bool:
    """Whether ``path`` is a build directory (``rtc_<key>``, ``fused_<key>``) of a toolchain's directory."""
    return bool(re.fullmatch(rf"(rtc|fused)_{_H}{{16}}", os.path.basename(path))) and \
        bool(re.fullmatch(rf"{TC_PREFIX}{_H}{{12}}", os.path.basename(os.path.dirname(path)))) and \
        os.path.isdir(path) and not os.path.islink(path)


def _entry_lock(directory: str) -> str:
    """The build lock of a build directory of a toolchain's directory (``build``'s)."""
    return os.path.join(os.path.dirname(directory), "locks", f"{os.path.basename(directory)}.lock")


def pack(directory: str, errors: list[str]) -> int:
    """Pack a build directory's intermediate files into ``TEMPS`` (lzma preset 0: the never-read files of a prefill
    ELF's build 65x at ~290 MB/s) and delete them; returns the bytes freed. The archive is written under a temporary
    name and renamed before anything is deleted, so a killed packing leaves the plain files, or both: where the archive
    exists its files are deleted (``build`` removes the archive of a directory it builds again, so the plain files are
    then the same)."""
    archive = os.path.join(directory, TEMPS)
    for name in os.listdir(directory):
        if re.fullmatch(re.escape(TEMPS) + r"\.\d+\.tmp", name):
            os.remove(os.path.join(directory, name))
    names = cold_files(directory)
    if not names:
        return 0
    before = sum(disk_usage(os.path.join(directory, n)) for n in names)
    if not os.path.exists(archive):
        tmp = f"{archive}.{os.getpid()}.tmp"
        with tarfile.open(tmp, "w:xz", preset=0) as t:
            for n in names:
                t.add(os.path.join(directory, n), arcname=n)
        os.replace(tmp, archive)
        before -= disk_usage(archive)
    for n in names:
        _delete(os.path.join(directory, n), errors)
    return max(before, 0)


def _pack_entry(e: Entry, report: Report) -> int:
    """``pack`` a scanned build entry under its build lock (taken without waiting: a build runs, nothing packed)."""
    lock = e.lock_path
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with portable.file_lock(lock, wait=False) as held:
        if not held or os.path.islink(e.path) or not os.path.isdir(e.path):
            return 0
        try:
            return pack(e.path, report.errors)
        except (OSError, tarfile.TarError) as exc:
            report.errors.append(f"{e.path}: {exc}")
            return 0


def _pack_cold(found: Scan, after: float, report: Report, *, dry_run: bool, skip: set[str] | None = None) -> None:
    """Pack the builds not used for ``after`` seconds (not those in ``skip``, nor failed ones)."""
    for e in found.entries:
        if (e.kind.where == "tc" and e.kind.name in ("rtc", "fused") and e.used <= report.when - after
                and not e.failed and e.path not in (skip or ())):
            freed = _cold_size(e.path) if dry_run else _pack_entry(e, report)
            if freed:
                report.packed[0] += 1
                report.packed[1] += freed


def pack_all(after: float, *, dry_run: bool = False) -> Report:
    """Pack the intermediate files of the builds not used for ``after`` seconds (``dry_run``: only count them)."""
    report = Report(when=time.time())
    _pack_cold(scan(), after, report, dry_run=dry_run)
    return report


def unpack(directory: str) -> bool:
    """Restore a build directory's packed intermediate files in place (under its build lock) and remove the archive;
    False when nothing is packed."""
    archive = os.path.join(directory, TEMPS)
    if not os.path.exists(archive):
        return False
    lock = _entry_lock(directory)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with portable.file_lock(lock):
        if not os.path.exists(archive):
            return False
        with tarfile.open(archive, "r:xz") as t:
            t.extractall(directory, filter="data")
        os.remove(archive)
    return True


def clean(policy: Policy, *, dry_run: bool = False, wait: bool = True) -> Report | None:
    """Clean the cache with ``policy`` (``dry_run``: only report what would go, changing nothing). One cleaning at a
    time: None when another one runs and ``wait`` is false."""
    if dry_run:
        return _clean(policy, dry_run=True)
    with _clean_lock(wait) as held:
        return _clean(policy, dry_run=False) if held else None


# ---------------------------------------------------------------------------
# The automatic cleaning
# ---------------------------------------------------------------------------

_grew = False


def last_auto() -> dict:
    """The automatic cleaning's last run (``common/gc.json``: when, what it removed, errors; {}: none)."""
    try:
        with open(os.path.join(ROOT, COMMON, "gc.json")) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _due(p: Policy, now: float) -> bool:
    when = last_auto().get("when", 0.0)
    return p.interval is not None and (not isinstance(when, (int, float)) or now - when >= p.interval)


def note_build() -> None:
    """A build added an entry: at its exit this process starts the automatic cleaning when it is due (only builds grow
    the cache, so a process that only loads never needs one)."""
    global _grew
    if not _grew:
        _grew = True
        atexit.register(_start_auto)


def _start_auto() -> None:
    """Start the automatic cleaning in a detached child at the lowest priority, when it is due: the exiting process
    (which may hold an NPU context) does not wait, and no cleaning runs inside an NPU job's process."""
    with contextlib.suppress(OSError, ValueError):  # (a bad ARTLITE_CACHE_GC, a failed start: no cleaning, no failure)
        if not _due(policy(), time.time()):
            return
        options: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if portable.WINDOWS:
            options["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                        | subprocess.IDLE_PRIORITY_CLASS)
        else:
            options["start_new_session"] = True
        subprocess.Popen([sys.executable, "-c", "from artlite import cache; cache.run_auto()"],
                         env={**os.environ, settings.CACHE_DIR.name: ROOT}, **options)


def auto() -> Report | None:
    """The automatic cleaning: once per ``interval`` and cache, with the policy of ``ARTLITE_CACHE_GC``; its result
    in ``common/gc.json``. None: not due, or another cleaning runs."""
    p = policy()
    with _clean_lock(wait=False) as held:
        if not held or not _due(p, time.time()):
            return None
        report = _clean(p, dry_run=False)
        _write_json(common("gc.json"), report.to_json())
    return report


def run_auto() -> None:
    """The automatic cleaning's child process (``_start_auto``), at the lowest priority."""
    if hasattr(os, "nice"):
        os.nice(19)
    auto()


def discard(path: str) -> None:
    """Remove a cache entry its owner replaces (a vision state's older entries, a failed UI download): never a link,
    nothing outside the cache's root."""
    real_root = os.path.join(os.path.realpath(ROOT), "")
    if os.path.islink(path) or not os.path.realpath(path).startswith(real_root):
        return
    errors: list[str] = []
    if os.path.lexists(path):
        _delete(path, errors)
