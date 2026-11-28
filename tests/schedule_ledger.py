"""The schedule ledger: the loop schedules, call cycles, code bytes and stack of every kernel the apps ship, and the
program memory of their cores, as rows of a committed baseline (``schedule_ledger.jsonl``) that
``test_app_designs.py`` checks app by app. The apps are ``apps.py``'s (what the models run); a toolchain bump's survey
is ``record`` against the baseline; ``rank`` weights an app's loops by their executions in a prefill chunk.

    python -m tests.schedule_ledger record [APP ...]            # the apps present (or those named) against the baseline
    python -m tests.schedule_ledger record [APP ...] --update   # their rows into the baseline (other apps' rows kept)
    python -m tests.schedule_ledger record [APP ...] --out FILE
    python -m tests.schedule_ledger diff BASE NEW               # two ledgers
    python -m tests.schedule_ledger rank APP [-p TOKENS]        # a prefill chunk's kinds of items and loops by weight

Rows, one JSON object per line, sorted; a function or core row lists the apps it holds for:

- a section, an operator's part of a translation unit: ``{"section": id, "label", "detail"}``, its class and module and
  its scalar constructor arguments (``kernel_source``'s section header; a single operator's translation unit: the
  operator's ``kernel_detail``), the id a digest of both;
- a function, ``{"fn", "in", "apps", "cycles", "upper", "bytes", "stack", "calls", "loops"}``: its name without the
  content hash (demangled), the section that defines it (a helper several sections use: the first one), measure_ii's
  cycles of a call (callees included; ``upper``: branches counted on every path or an early exit), code bytes, stack,
  calls per call of the functions it calls (by key; null: a trip count unknown) and its loops in tree order (``at``: the
  item its line is in and the line within it, "<name>+N", or a header's file:line (a helper added to ``kernel_lib.h``
  goes at its end: lines inserted above others move every app's rows); ``parent``: the enclosing loop's
  index, -1 at the top; sched, II, NS; ``bound``: a one-block loop's resource bound, its busiest issue slot or port;
  ``iters``: iterations per entry; ``spills``: spill and reload instructions; ``bytes``: its body's);
- a core row, ``{"core", "design", "entry", "apps", "bytes"}``: the most program memory a core of that row takes in a
  design entry (a fused ELF's operator; ``design`` "": an operator's own design), a paged core's slot and largest page
  included.

Keys: a function by (section, name), a core row by (design, entry, row): the same across a toolchain change (the sources
do not change) and across a change of shared kernel code (names lose their hashes, sections keep their arguments). A
name with other code under one key in one app (arguments the detail does not show) keeps its hash.

Measurements are cached (``ledger/`` in the toolchain's cache directory, ``artlite.cache``) by translation unit, compile
command without paths and headers (``build.compile_key``), measure_ii's sources and this module's extraction; core rows
by build directory (named by content). A warm ledger costs the apps' construction (their builds cached).

Weights (``rank``, ``chunk_weights``): a prefill chunk's items are static (``runlists``), and each item's design runs
one pass of its cores' main loops, whose bounds and branches are constants or arithmetic on the chunk's scratchpad
parameters: ``core_calls`` evaluates them (the MLIR's scf / arith ops) to the kernel calls of each core, times
measure_ii's call cycles. Against the NPU's busy cycles per core (ACTIVE - LOCK_STALL - STREAM_STALL, every item alone)
on Qwen3.5 9B's 1024-token chunk the sum of the cells' errors was 11% of the busy cycles, mostly the memory stalls the
static cycles leave out; a kernel whose branches measure_ii counts on both paths (the prefill attentions) is an upper
bound (1.9x there), shown as each kind's ``upper`` share. A loop's slack, (II - resource bound) x its weight, is what a
better schedule could save; a loop at its bound only gets faster with fewer operations. The decode overlays have no
static weights (their cores run what the host's commands say: loop bounds from data); counted on the NPU (PC events at
the functions' entries), their calls times the static cycles missed a Qwen3.5 9B step's busy cycles by 36% (their
kernels' trip counts come from arguments, their dispatchers' branches are counted on every path): ``rank`` covers the
prefill.
"""

import argparse
import functools
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from artlite import cache, parallel_compile
from artlite.elf_parser import SHF_EXECINSTR, read_object
from artlite.measure_ii import call_cycles, count_slots, measure
from artlite.measure_ii.demangle import demangle_short
from artlite.measure_ii.report import KernelReport, LoopReport, tree_order
from artlite.operators import build, kernel_source, toolchain
from artlite.operators.fused.elf import FusedElf
from artlite.operators.paging import PM_BYTES

from .apps import APPS, AppBuild

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "schedule_ledger.jsonl")
LOW = 256  # bytes of free program memory below which a core row is listed first (a loop newly pipelined: +80..288 B)
_HASHED = re.compile(r"\b[A-Za-z_]\w*_[0-9a-f]{8}\b")  # (a kernel name: ``kernel_source``)
_HASH = re.compile(r"(?<=\w)_[0-9a-f]{8}(?![0-9a-f])")
_ELFS = re.compile(r"elfs_(?:op(\d+)_\w+?|main)_core_(\d+)_(\d+)$")
_SHARED = "// shared with an earlier section:"

Key = tuple  # ("fn", section, name) | ("core", design, entry, row)


def _cache(*parts: str) -> str:
    """A path in the ledger's cache (``ledger/`` of the toolchain's cache directory), its directory made."""
    path = cache.toolchain_dir("ledger", *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def section_id(label: str, detail: str) -> str:
    return hashlib.sha256(f"{label}\0{detail}".encode()).hexdigest()[:12]


@functools.cache
def _tool_hash() -> str:
    """measure_ii's sources and this module's extraction (``_EXTRACTION``): a change of either measures again."""
    import artlite.measure_ii as mi

    h = hashlib.sha256()
    pkg = os.path.dirname(mi.__file__)
    for path in sorted(os.path.join(pkg, f) for f in os.listdir(pkg) if f.endswith(".py")):
        with open(path, "rb") as f:
            h.update(f.read())
    for obj in _EXTRACTION:
        h.update(inspect.getsource(obj).encode())
    return h.hexdigest()[:16]


# One translation unit -----------------------------------------------------------------------------------------------

@dataclass
class Section:
    label: str
    detail: str
    names: set[str]  # the kernel names its items define
    items: list[tuple[int, int, str]]  # its items that define names: (first line, last line, the first name; 1-based)


def sections(source: str) -> list[Section]:
    """A translation unit's sections (``kernel_source.translation_unit``): header, detail lines, the names defined and
    where their items are."""
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("// ====")]
    out = []
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        head = lines[i].rstrip("\n")
        label = head.removeprefix("// ==== ").removesuffix(" ====") if head != "// ====" else ""
        detail, j = [], i + 1
        while j < end and lines[j].startswith("//") and not lines[j].startswith(_SHARED):
            detail.append(lines[j][2:].strip())
            j += 1
        text, pos, names, items = "".join(lines[i:end]), 0, set(), []
        for item in kernel_source.split(text):
            names.update(item.defines)
            at = text.find(item.text, pos)
            if at < 0 or not item.defines:
                continue
            pos = at + len(item.text)
            first = i + 1 + text.count("\n", 0, at)
            items.append((first, first + item.text.count("\n"), item.defines[0]))
        out.append(Section(label, " ".join(detail), names, items))
    return out


def _place(secs: list[Section], line: int) -> str:
    """A line of the translation unit as the item it is in and the line within it ("<name>+N", the name without its
    hash): it moves only with that item's text."""
    for s in secs:
        for first, last, name in s.items:
            if first <= line <= last:
                return f"{_HASH.sub('', name)}+{line - first}"
    return f"kernel.cc:{line}"


def _calls(report: KernelReport, fn: str, loops: list[LoopReport]) -> dict[str, int | None]:
    """Calls per call of ``fn`` of each function it calls: outside its loops, and in each loop times the iterations of
    the loop and the loops around it (None: one of them unknown)."""
    out: dict[str, int | None] = defaultdict(int)
    for callee in report.outside_loops.get(fn, (0, [], False))[1]:
        out[callee] += 1
    by_header = {lp.header: lp for lp in loops if lp.header}
    for lp in loops:
        times: int | None = 1
        cur: LoopReport | None = lp
        while cur is not None:
            iters = cur.effective_iters()
            times = None if times is None or iters is None else times * iters
            cur = by_header.get(cur.parent) if cur.parent else None
        for callee in lp.callees:
            out[callee] = None if times is None or out[callee] is None else out[callee] + times
    return dict(out)


def extract(report: KernelReport, source: str) -> dict:
    """A measured translation unit: its sections and per function its measurements (JSON-ready)."""
    if report.returncode or report.parse_error:
        return {"error": (report.issues or ["measure_ii failed"])[0][:500]}
    secs = sections(source)
    owner = {n: k for k, s in enumerate(secs) for n in s.names}
    by_fn: dict[str, list[LoopReport]] = defaultdict(list)
    for lp in tree_order(report.loops):
        by_fn[lp.function].append(lp)
    fns = []
    for f in report.functions:
        short = demangle_short(f)
        m = _HASHED.search(short)
        sec = owner.get(m.group(0)) if m else None
        loops = by_fn.get(f, [])
        index = {lp.header: i for i, lp in enumerate(loops) if lp.header}
        rows = []
        for lp in loops:
            at = _place(secs, lp.kernel_line) if lp.kernel_line is not None else lp.source_ref or "?"
            rows.append({"at": at, "parent": index.get(lp.parent, -1) if lp.parent else -1, "sched": lp.sched,
                         "ii": lp.ii, "ns": lp.ns, "bound": count_slots(lp.asm_body).bound if lp.num_blocks == 1 else None,
                         "iters": lp.effective_iters(), "spills": lp.loop_spills + lp.loop_reloads,
                         "bytes": lp.body_bytes})
        cycles = call_cycles(report, f)
        fns.append({"sym": f, "short": short, "section": sec, "cycles": cycles[0] if cycles else None,
                    "upper": cycles[1] if cycles else None, "bytes": sum(b.bytes for b in report.layout.get(f, [])),
                    "stack": report.stack_bytes.get(f), "calls": _calls(report, f, loops), "loops": rows})
    return {"sections": [[s.label, s.detail] for s in secs], "functions": fns}


_EXTRACTION = (Section, sections, _place, _calls, extract)


def measured(source: str, clang_cmd: list[str]) -> dict:
    """``extract`` of a translation unit compiled with ``clang_cmd`` (cached)."""
    key = toolchain.cache_key(source, build.compile_key(clang_cmd), _tool_hash())
    path = _cache(f"tu_{key}.json")
    if os.path.exists(path):
        cache.touch(path)
        with open(path) as f:
            return json.load(f)
    build.compile_pch(clang_cmd=clang_cmd)
    work = _cache("work", key)
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "kernel.cc")
    with open(src, "w") as f:
        f.write(source)
    out = extract(measure(src, clang_cmd=clang_cmd, verbose=False, cache=False), source)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    shutil.rmtree(work, ignore_errors=True)
    return out


def core_bytes(work_dir: str) -> dict[tuple[int, int, int], int]:
    """Per core ELF of a build ((entry, column, row); entry -1: an operator's own design) the end of its code in program
    memory (cached by the build directory, named by its content)."""
    path = _cache(f"cores_{os.path.basename(work_dir.rstrip(os.sep))}.json")
    if os.path.exists(path):
        cache.touch(path)
        with open(path) as f:
            return {tuple(k): v for k, v in json.load(f)}
    prj = os.path.join(work_dir, "design.mlir.prj")
    out = {}
    for d in sorted(os.listdir(prj)):
        m = _ELFS.match(d)
        if m is None:
            continue
        obj = read_object(os.path.join(prj, d, f"{d}.elf"))
        end = max((s.addr + len(s.data) for s in obj.sections if s.flags & SHF_EXECINSTR and s.addr < PM_BYTES),
                  default=0)
        out[(int(m.group(1)) if m.group(1) else -1, int(m.group(2)), int(m.group(3)))] = end
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump([[list(k), v] for k, v in out.items()], f)
    os.replace(tmp, path)
    return out


# The ledger ---------------------------------------------------------------------------------------------------------

@dataclass
class Ledger:
    sections: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (label, detail)
    apps: dict[str, dict[Key, dict]] = field(default_factory=dict)  # app -> key -> value

    def __repr__(self) -> str:
        return f"Ledger({len(self.apps)} apps, {sum(map(len, self.apps.values()))} rows)"

    def merge(self, other: "Ledger") -> None:
        """``other``'s apps replace this ledger's (its sections added)."""
        self.sections |= other.sections
        self.apps |= other.apps

    def rows(self) -> list[dict]:
        """The ledger as sorted rows (the module docstring), a row per key and value with the apps that have it."""
        groups: dict[tuple[Key, str], list[str]] = defaultdict(list)
        values: dict[tuple[Key, str], dict] = {}
        for app, entries in self.apps.items():
            for key, value in entries.items():
                k = (key, json.dumps(value, sort_keys=True))
                groups[k].append(app)
                values[k] = value
        used = {k[0][1] for k in groups if k[0][0] == "fn"} | {k[0][2] for k in groups if k[0][0] == "core"}

        def label(sid: str) -> tuple[str, str]:
            return self.sections.get(sid, ("", ""))

        out = [{"section": s, "label": label(s)[0], "detail": label(s)[1]}
               for s in sorted(used, key=lambda s: (*label(s), s))]
        fns, cores = [], []
        for (key, text), apps in groups.items():
            if key[0] == "fn":
                fns.append(((*label(key[1]), key[2], text), {"fn": key[2], "in": key[1], "apps": sorted(apps)}
                            | values[(key, text)]))
            else:
                cores.append(((key[1], *label(key[2]), key[3], text), {"core": key[3], "design": key[1],
                                                                       "entry": key[2], "apps": sorted(apps)}
                              | values[(key, text)]))
        return out + [r for _, r in sorted(fns, key=lambda x: x[0])] + [r for _, r in sorted(cores, key=lambda x: x[0])]

    def save(self, path: str) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in self.rows())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Ledger":
        led = cls()
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "section" in row:
                    led.sections[row["section"]] = (row["label"], row["detail"])
                    continue
                apps = row.pop("apps")
                if "fn" in row:
                    key: Key = ("fn", row.pop("in"), row.pop("fn"))
                else:
                    key = ("core", row.pop("design"), row.pop("entry"), row.pop("core"))
                for app in apps:
                    led.apps.setdefault(app, {})[key] = row
        return led


@dataclass
class _Unit:
    """A design's translation unit to measure and how its rows are keyed."""

    source: str
    clang_cmd: list[str]
    detail: str  # a single operator's (its header has none)


def _units(b: AppBuild) -> tuple[list[_Unit], list[tuple[str, str, str]], dict[Key, dict]]:
    """An app's translation units, its core rows' sections ((id, label, detail) besides the key's) and its core rows."""
    units, secs, cores = [], [], {}
    pagers = {core: pg for pg in b.pagers for core in pg.cores}  # (the first design's: the overlay, the vision overlay)
    for k, design in enumerate(b.designs):
        if isinstance(design, FusedElf):
            if design.elf_path is None:  # (planning mode: nothing built)
                raise parallel_compile.PlanStop(f"{design.name} is not built")
            work_dir = os.path.dirname(design.elf_path)
            flags = {tuple(op.clang_flags) for op in design.operators}
            ops = [op for op, *_ in design.runlist]
            name = design.name
        else:
            work_dir = os.path.dirname(design._program.insts_path)
            flags = {tuple(design.clang_flags)}
            ops = [design]
            name = ""
        with open(os.path.join(work_dir, "kernel.cc")) as f:
            source = f.read()
        units.append(_Unit(source, [*toolchain.clang_cmd_common(), f"--target={toolchain.AIE_TARGET}", *flags.pop()],
                           "" if isinstance(design, FusedElf) else design.kernel_detail()))
        rows: dict[tuple[str, int], int] = {}
        for (entry, c, r), end in core_bytes(work_dir).items():
            op = ops[entry] if entry >= 0 else ops[0]
            sid = section_id(op.kernel_label(), op.kernel_detail())
            secs.append((sid, op.kernel_label(), op.kernel_detail()))
            pg = pagers.get((c, r)) if k == 0 else None
            if pg is not None:
                end = max(end, pg.slot + max((p.nbytes for p in pg.pages.values()), default=0))
            rows[(sid, r)] = max(rows.get((sid, r), 0), end)
        for (sid, r), end in rows.items():
            cores[("core", name, sid, r)] = {"bytes": end}
    return units, secs, cores


def _app_rows(units: list[_Unit], results: list[dict]) -> tuple[dict[str, tuple[str, str]], dict[Key, dict]]:
    """The function rows of an app's measured translation units, and their sections."""
    secs: dict[str, tuple[str, str]] = {}
    found: dict[Key, dict[str, dict]] = defaultdict(dict)  # key -> hashed name -> value
    for unit, res in zip(units, results):
        if "error" in res:
            raise RuntimeError(f"measure_ii failed on a translation unit ({len(unit.source)} B): {res['error']}")
        ids = []
        for label, detail in res["sections"]:
            detail = detail or (unit.detail if len(res["sections"]) == 1 else "")
            sid = section_id(label, detail)
            secs[sid] = (label, detail)
            ids.append(sid)
        none_id = section_id("", "")
        sym = {}  # asm symbol -> (section id, name, hashed name)
        for fn in res["functions"]:
            sid = ids[fn["section"]] if fn["section"] is not None else none_id
            sym[fn["sym"]] = (sid, _HASH.sub("", fn["short"]), fn["short"])
        for fn in res["functions"]:
            sid, name, hashed = sym[fn["sym"]]
            calls = sorted([[*sym[c][:2], n] if c in sym else ["", c, n] for c, n in fn["calls"].items()],
                           key=lambda x: (x[0], x[1]))
            found[("fn", sid, name)][hashed] = {"cycles": fn["cycles"], "upper": fn["upper"], "bytes": fn["bytes"],
                                                "stack": fn["stack"], "calls": calls, "loops": fn["loops"]}
        if none_id in {s[0] for s in sym.values()}:
            secs[none_id] = ("", "")
    rows: dict[Key, dict] = {}
    for key, by in found.items():
        distinct = {json.dumps(v, sort_keys=True) for v in by.values()}
        if len(by) == 1 or len(distinct) == 1:
            rows[key] = next(iter(by.values()))
        else:  # (other code under one key: each keeps its hash)
            for hashed, value in by.items():
                rows[(key[0], key[1], hashed)] = value
    return secs, rows


def available() -> list[str]:
    """The apps whose model files are present."""
    return [app for app, (path, _) in APPS.items() if os.path.exists(path)]


_Parts = tuple[str, list[_Unit], list[tuple[str, str, str]], dict[Key, dict]]


def _parts(app: str, b: AppBuild) -> _Parts:
    """What the ledger needs of a built app (the build's objects can go)."""
    return (app, *_units(b))


def _ledger(parts: list[_Parts], *, log: bool = False) -> Ledger:
    """The ledger of built apps: their translation units measured (in parallel, cached), their core rows."""
    led = Ledger()
    for app, units, secs, cores in parts:
        for sid, label, detail in secs:
            led.sections[sid] = (label, detail)
        if log:
            print(f"{app}: {len(units)} translation unit(s), {len(cores)} core rows", file=sys.stderr)
    distinct = list(dict.fromkeys((u.source, tuple(u.clang_cmd)) for _, units, _, _ in parts for u in units))
    distinct.sort(key=lambda k: -len(k[0]))  # (the largest first: they set the wall time)
    if log:
        print(f"measuring {len(distinct)} distinct translation unit(s)", file=sys.stderr)
    results = dict(zip(distinct, parallel_compile.parallel_map(lambda k: measured(k[0], list(k[1])), distinct)))
    for app, units, _, cores in parts:
        secs, rows = _app_rows(units, [results[(u.source, tuple(u.clang_cmd))] for u in units])
        led.sections |= secs
        led.apps[app] = rows | cores
    return led


def ledger_of(app: str, b: AppBuild) -> Ledger:
    """The ledger of one built app (``apps.APPS``' builder's result)."""
    return _ledger([_parts(app, b)])


def record(apps: Iterable[str], *, log: bool = True) -> Ledger:
    """The ledger of ``apps``, built by their builders (``apps.py``: cached builds are not rebuilt); an app whose
    builder skips (a drafter file missing) is left out."""
    parts = []
    for app in apps:
        path, builder = APPS[app]
        try:
            parts.append(_parts(app, builder(path)))
        except pytest.skip.Exception as e:
            if log:
                print(f"{app}: skipped ({e})", file=sys.stderr)
    return _ledger(parts, log=log)


# Weights: a prefill chunk's calls -----------------------------------------------------------------------------------

_BIN = {
    "arith.addi": lambda a, b: a + b, "arith.subi": lambda a, b: a - b, "arith.muli": lambda a, b: a * b,
    "arith.divsi": lambda a, b: int(a / b) if b else None, "arith.divui": lambda a, b: a // b if b else None,
    "arith.floordivsi": lambda a, b: a // b if b else None, "arith.ceildivsi": lambda a, b: -(-a // b) if b else None,
    "arith.ceildivui": lambda a, b: -(-a // b) if b else None,
    "arith.remsi": lambda a, b: a - b * int(a / b) if b else None, "arith.remui": lambda a, b: a % b if b else None,
    "arith.maxsi": max, "arith.minsi": min, "arith.maxui": max, "arith.minui": min,
    "arith.andi": lambda a, b: a & b, "arith.ori": lambda a, b: a | b, "arith.xori": lambda a, b: a ^ b,
    "arith.shli": lambda a, b: a << b, "arith.shrsi": lambda a, b: a >> b, "arith.shrui": lambda a, b: a >> b,
}
_CMP = (lambda a, b: a == b, lambda a, b: a != b, lambda a, b: a < b, lambda a, b: a <= b, lambda a, b: a > b,
        lambda a, b: a >= b, lambda a, b: a < b, lambda a, b: a <= b, lambda a, b: a > b, lambda a, b: a >= b)
_CAST = {"arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci"}
_ITERATED = 1 << 16  # iterations walked one by one at most (a loop whose body's control depends on its index)


def _int(attr: object) -> int | None:
    s = str(attr)
    if s in ("true", "false"):
        return int(s == "true")
    m = re.match(r"-?\d+", s)
    return int(m.group(0)) if m else None


class _Calls:
    """The kernel calls of one run of a core body (``core_calls``): ``calls`` per function, ``unknown`` the walks that
    met control on a value unknown here (data, a call's result)."""

    def __init__(self, params: dict[str, int]):
        self.params = params
        self.calls: Counter[str] = Counter()
        self.unknown = 0

    def block(self, block: Any, env: dict, mult: int) -> None:
        for op in block.operations:
            self.op(op, env, mult)

    def op(self, op: Any, env: dict, mult: int) -> None:
        name = op.operation.name
        val = env.get
        if name == "arith.constant":
            env[op.results[0]] = _int(op.attributes["value"])
        elif name == "aiex.read_scratchpad_parameter":
            env[op.results[0]] = self.params.get(str(op.attributes["parameter"]).lstrip("@"))
        elif name in _BIN:
            a, b = (val(x) for x in op.operands)
            env[op.results[0]] = None if a is None or b is None else _BIN[name](a, b)
        elif name in _CAST:
            env[op.results[0]] = val(op.operands[0])
        elif name == "arith.cmpi":
            a, b = (val(x) for x in op.operands)
            env[op.results[0]] = None if a is None or b is None else int(_CMP[_int(op.attributes["predicate"])](a, b))
        elif name == "arith.select":
            c, a, b = (val(x) for x in op.operands)
            env[op.results[0]] = None if c is None else (a if c else b)
        elif name == "func.call":
            self.calls[str(op.attributes["callee"]).lstrip("@")] += mult
        elif name == "scf.if":
            c = val(op.operands[0])
            if c is None:
                self.unknown += 1
            elif (region := op.regions[0 if c else 1]).blocks:
                self.block(region.blocks[0], env, mult)
        elif name == "scf.for":
            self.loop(op, env, mult)
        elif name == "scf.while":  # (a core's main loop, while (true): one run is one pass of its body)
            before = op.regions[0].blocks[0]
            ops = list(before.operations)
            self.block(before, env, mult)
            if len(ops) == 2 and val(ops[-1].operands[0]) == 1:
                self.block(op.regions[1].blocks[0], env, mult)
            else:
                self.unknown += 1
        for r in op.results:
            env.setdefault(r, None)

    def loop(self, op: Any, env: dict, mult: int) -> None:
        lb, ub, step = (env.get(x) for x in op.operands[:3])
        body = op.regions[0].blocks[0]
        if lb is None or ub is None or not step:
            self.unknown += 1
            return
        trips = max(0, -(-(ub - lb) // step))
        if not trips:
            return
        probe = _Calls(self.params)  # (the body once with an unknown index: its calls if its control does not need it)
        probe.block(body, dict(env) | dict.fromkeys(body.arguments), 1)
        if not probe.unknown:
            for f, n in probe.calls.items():
                self.calls[f] += n * trips * mult
        elif trips > _ITERATED:
            self.unknown += 1
        else:
            for i in range(lb, ub, step):
                self.block(body, dict(env) | dict.fromkeys(body.arguments) | {body.arguments[0]: i}, mult)


def core_calls(mlir: str, params: dict[str, int]) -> dict[str, dict[tuple[int, int], _Calls]]:
    """Per device of a design (a library's: one per distinct entry design) and core (column, row): the kernel calls of
    one run (``_Calls``), its loops' bounds and branches evaluated with the scratchpad ``params``."""
    import aie.dialects.aie  # noqa: F401  (the dialects)
    from aie.ir import Context, Location, Module

    out: dict[str, dict[tuple[int, int], _Calls]] = {}
    with Context(), Location.unknown():
        for dev in Module.parse(mlir).body.operations:
            if dev.operation.name != "aie.device":
                continue
            cores = out[str(dev.attributes["sym_name"]).strip('"')] = {}
            for op in dev.regions[0].blocks[0].operations:
                if op.operation.name == "aie.core":
                    tile = op.operands[0].owner
                    calls = cores[(_int(tile.attributes["col"]), _int(tile.attributes["row"]))] = _Calls(params)
                    calls.block(op.regions[0].blocks[0], {}, 1)
    return out


def _kind(op: Any) -> str:
    """An operator's kind in a ranking: its class, a GEMM's format and shape."""
    fmt, k, n = getattr(op, "fmt", None), getattr(op, "K", None), getattr(op, "N", None)
    return type(op).__name__ + (f" {fmt}" if fmt else "") + (f" {k}x{n}" if k and n else "")


@dataclass
class ChunkWeights:
    """A prefill chunk's static work (``chunk_weights``): per kind of item (items, cycles of the busiest core of each item
    summed, all cores' cycles, the part of those from functions whose cycles are an upper bound: measure_ii counts both
    paths of a branch, e.g. an attention's masked and unmasked blocks) and per loop of the library's kernels its weighted cycles (calls x executions x iterations
    x II) and its slack (the same with II - its resource bound)."""

    T: int
    items: int
    kinds: dict[str, list[int]]  # kind -> [items, busiest core's cycles, all cores' cycles, their upper bounds' part]
    loops: dict[tuple[str, int], tuple[int, int]]  # (function, loop index) -> (weighted cycles, slack cycles)
    functions: dict[str, dict]  # the library's measured functions by symbol
    unknown: int  # calls of unknown cost and walks with control on unknown values


def chunk_weights(b: AppBuild, tokens: int) -> ChunkWeights:
    """The static work of an app's prefill chunk of ``tokens`` (its largest chunk size up to it), unpaced: per item the
    calls per core from its design's core bodies with the chunk's scratchpad parameters (position 0, all tokens valid)
    times measure_ii's call cycles of the library's kernels (``measured``)."""
    from artlite.gemm.pacing import PACE

    pf = b.prefill
    if pf is None:
        raise ValueError("the app has no chunk prefill")
    T = max([t for t in pf.sizes if t <= tokens] or [min(pf.sizes)])
    params = pf.params(T, 0, T) if "n" in inspect.signature(pf.params).parameters else pf.params(T, 0)
    params = {**params, PACE: 0}
    lib = pf.lib
    work_dir = os.path.dirname(lib.elf_path)
    cache.unpack(work_dir)
    with open(os.path.join(work_dir, "design.mlir")) as f:
        mlir = f.read()
    with open(os.path.join(work_dir, "kernel.cc")) as f:
        source = f.read()
    flags = {tuple(op.clang_flags) for op in lib.operators}.pop()
    res = measured(source, [*toolchain.clang_cmd_common(), f"--target={toolchain.AIE_TARGET}", *flags])
    if "error" in res:
        raise RuntimeError(f"measure_ii failed on {work_dir}/kernel.cc: {res['error']}")
    fns = {f["sym"]: f for f in res["functions"]}
    devices = core_calls(mlir, params)
    # (entries of identical designs share the first one's device: main configures each entry's, in entry order)
    device_of = re.findall(r"aiex\.configure @(\w+)", mlir)
    entry = {id(op): k for k, (op, *_) in enumerate(lib.runlist)}

    def closure(calls: Counter) -> Counter:
        out: Counter = Counter()
        todo = list(calls.items())
        while todo:
            f, n = todo.pop()
            out[f] += n
            todo += [(g, n * k) for g, k in (fns.get(f, {}).get("calls") or {}).items() if k is not None and g in fns]
        return out

    kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    loops: Counter = Counter()
    slack: Counter = Counter()
    unknown = 0
    per_device: dict[str, tuple[dict[tuple[int, int], int], int, Counter, Counter, int]] = {}
    for op, *_ in pf.runlists[T]:
        name = device_of[entry[id(op)]]
        if name not in per_device:
            cycles, upper, lw, ls, unk = {}, 0, Counter(), Counter(), 0
            for core, c in devices[name].items():
                unk += c.unknown
                cyc = 0
                for f, n in c.calls.items():
                    if (v := fns.get(f, {}).get("cycles")) is None:
                        unk += n
                    else:
                        cyc += n * v
                        upper += n * v if fns[f]["upper"] else 0
                cycles[core] = cyc
                for f, n in closure(c.calls).items():
                    lps = fns[f]["loops"] if f in fns else []
                    for i, lp in enumerate(lps):
                        times, p = n * (lp["iters"] or 0), lp["parent"]
                        while p >= 0 and times:
                            times *= lps[p]["iters"] or 0
                            p = lps[p]["parent"]
                        if times and lp["ii"] is not None:
                            lw[(f, i)] += times * lp["ii"]
                            if lp["bound"] is not None:
                                ls[(f, i)] += times * max(0, lp["ii"] - lp["bound"])
            per_device[name] = (cycles, upper, lw, ls, unk)
        cycles, upper, lw, ls, unk = per_device[name]
        k = kinds[_kind(op)]
        k[0] += 1
        k[1] += max(cycles.values(), default=0)
        k[2] += sum(cycles.values())
        k[3] += upper
        loops.update(lw)
        slack.update(ls)
        unknown += unk
    return ChunkWeights(T, len(pf.runlists[T]), dict(kinds), {key: (v, slack[key]) for key, v in loops.items()},
                        fns, unknown)


def rank_text(app: str, w: ChunkWeights, *, top: int = 15) -> str:
    """A chunk's ranking: its kinds of items by their busiest cores' cycles, its loops by weighted cycles and by slack."""
    mhz = 1810.0
    total = sum(k[1] for k in w.kinds.values())
    out = [(f"{app}: a prefill chunk of {w.T} tokens, {w.items} items, unpaced; the busiest core of each item: "
            f"{total / mhz / 1e3:.1f} ms of compute ({w.unknown} calls or branches unknown)"),
           f"{'kind':44s} {'items':>5s} {'busiest ms':>10s} {'share':>6s} {'all cores Mcy':>13s} {'upper':>6s}"]
    for name, (n, busiest, cores, upper) in sorted(w.kinds.items(), key=lambda x: -x[1][1]):
        out.append(f"{name:44s} {n:5d} {busiest / mhz / 1e3:10.2f} {busiest / max(total, 1):6.1%} {cores / 1e6:13.1f} "
                   f"{upper / max(cores, 1):6.0%}")
    all_cycles = sum(k[2] for k in w.kinds.values())
    for title, idx in (("loops by weighted cycles (calls x executions x iterations x II)", 0),
                       ("loops by slack ((II - resource bound) x the same)", 1)):
        out.append(f"\n{title}, of all cores' {all_cycles / 1e6:.0f} Mcy:")
        for (f, i), v in sorted(w.loops.items(), key=lambda x: -x[1][idx])[:top]:
            if not v[idx]:
                break
            fn, lp = w.functions[f], w.functions[f]["loops"][i]
            out.append(f"  {v[idx] / 1e6:9.1f} Mcy {v[idx] / max(all_cycles, 1):6.1%}  {_HASH.sub('', fn['short'])} "
                       f"loop {i} ({lp['at']}): {lp['sched']} II {lp['ii']} bound {lp['bound']} x {lp['iters']}")
    return "\n".join(out)


# The diff -----------------------------------------------------------------------------------------------------------

# change classes, worst first (the improvements after the regressions)
CLASSES = ("failed", "pipelining lost", "spills up", "II up", "NS up", "loops changed", "cycles up", "stack up",
           "bytes up", f"cores below {LOW} B free", "core bytes up", "removed", "added", "pipelining gained",
           "spills down", "II down", "NS down", "cycles down", "stack down", "bytes down", "core bytes down")


def _name(led: Ledger, key: Key) -> str:
    if key[0] == "fn":
        label, detail = led.sections.get(key[1], ("?", ""))
        return f"{label.split(' (')[0] or '-'}[{detail[:60]}{'...' if len(detail) > 60 else ''}] {key[2]}"
    label, detail = led.sections.get(key[2], ("?", ""))
    return f"{key[1] or '-'} / {label.split(' (')[0]}[{detail[:60]}{'...' if len(detail) > 60 else ''}] row {key[3]}"


def _changes(key: Key, a: dict | None, b: dict | None) -> list[tuple[str, str]]:
    """(class, text) of one key's change."""
    if a is None:
        return [("added", "")]
    if b is None:
        return [("removed", "")]
    if key[0] == "core":
        out = [("core bytes up" if b["bytes"] > a["bytes"] else "core bytes down",
                f"{a['bytes']} -> {b['bytes']} B ({PM_BYTES - b['bytes']} free)")]
        if PM_BYTES - b["bytes"] < LOW:
            out.insert(0, (f"cores below {LOW} B free", f"{PM_BYTES - b['bytes']} B free (was {PM_BYTES - a['bytes']})"))
        return out
    out = []
    la, lb = a["loops"], b["loops"]
    if len(la) != len(lb):
        out.append(("loops changed", f"{len(la)} -> {len(lb)} loops"))
    else:
        for i, (x, y) in enumerate(zip(la, lb)):
            where = f"loop {i} ({y['at']}, {y['iters']} iters)"
            if (x["sched"] == "none") != (y["sched"] == "none"):
                cls = "pipelining lost" if y["sched"] == "none" else "pipelining gained"
                out.append((cls, f"{where}: {x['sched']} II {x['ii']} -> {y['sched']} II {y['ii']}"))
            elif x["ii"] != y["ii"] and x["ii"] is not None and y["ii"] is not None:
                out.append(("II up" if y["ii"] > x["ii"] else "II down",
                            f"{where}: II {x['ii']} -> {y['ii']} (bound {y['bound']})"))
            if x["ns"] != y["ns"] and (x["sched"] == "none") == (y["sched"] == "none"):
                out.append(("NS up" if y["ns"] > x["ns"] else "NS down", f"{where}: NS {x['ns']} -> {y['ns']}"))
            if x["spills"] != y["spills"]:
                out.append(("spills up" if y["spills"] > x["spills"] else "spills down",
                            f"{where}: {x['spills']} -> {y['spills']} spill / reload instructions"))
    for fld, word in (("cycles", "cycles"), ("stack", "stack"), ("bytes", "bytes")):
        if a[fld] != b[fld]:
            if a[fld] is None or b[fld] is None:
                out.append(("cycles up" if fld == "cycles" else f"{word} up", f"{word} {a[fld]} -> {b[fld]}"))
            else:
                pct = f" ({100 * (b[fld] - a[fld]) / a[fld]:+.1f}%)" if a[fld] else ""
                out.append((f"{word} up" if b[fld] > a[fld] else f"{word} down", f"{word} {a[fld]} -> {b[fld]}{pct}"))
    if not out:  # (calls, loop positions, bounds or iterations)
        out.append(("loops changed", "calls, loop places, bounds or iterations"))
    return out


def diff(base: Ledger, new: Ledger, apps: Iterable[str] | None = None) -> list[tuple[str, str, str, list[str]]]:
    """The changes between two ledgers over ``apps`` (default: the apps of both): (class, key's name, text, apps),
    in ``CLASSES`` order."""
    apps = sorted(set(base.apps) & set(new.apps)) if apps is None else list(apps)
    found: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for app in apps:
        a, b = base.apps.get(app, {}), new.apps.get(app, {})
        if app not in base.apps or app not in new.apps:
            found[("failed", app, "not in " + ("the base" if app not in base.apps else "the new ledger"))].append(app)
            continue
        for key in sorted(set(a) | set(b)):
            if a.get(key) == b.get(key):
                continue
            led = new if key in b else base
            for cls, text in _changes(key, a.get(key), b.get(key)):
                found[(cls, _name(led, key), text)].append(app)
    return sorted(((c, n, t, apps) for (c, n, t), apps in found.items()), key=lambda x: (CLASSES.index(x[0]), x[1]))


def report(changes: list[tuple[str, str, str, list[str]]], *, limit: int = 40) -> str:
    """The changes as text: a block per class, ``limit`` lines each."""
    if not changes:
        return "no changes"
    by: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
    for cls, name, text, apps in changes:
        by[cls].append((name, text, apps))
    out = []
    for cls in CLASSES:
        if cls not in by:
            continue
        out.append(f"{cls} ({len(by[cls])}):")
        for name, text, apps in by[cls][:limit]:
            where = apps[0] if len(apps) == 1 else f"{len(apps)} apps"
            out.append(f"  {name}: {text}" + (f" [{where}]" if text else f"  [{where}]"))
        if len(by[cls]) > limit:
            out.append(f"  ... {len(by[cls]) - limit} more")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tests.schedule_ledger", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record apps' ledger; compare with the baseline, or write it")
    r.add_argument("apps", nargs="*", help="app names (apps.APPS; default: those whose files exist)")
    r.add_argument("--update", action="store_true", help="write their rows into the baseline")
    r.add_argument("--out", help="write their ledger to this file")
    r.add_argument("--limit", type=int, default=40, help="lines per change class")
    k = sub.add_parser("rank", help="an app's prefill chunk: its kinds of items and loops by static weight")
    k.add_argument("app")
    k.add_argument("-p", "--tokens", type=int, default=1024, help="the chunk's tokens (its largest size up to it)")
    k.add_argument("--top", type=int, default=15, help="loops listed")
    d = sub.add_parser("diff", help="compare two ledger files")
    d.add_argument("base")
    d.add_argument("new")
    d.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)
    if args.cmd == "rank":
        if args.app not in APPS:
            ap.error(f"unknown app {args.app}; known: {', '.join(APPS)}")
        path, builder = APPS[args.app]
        print(rank_text(args.app, chunk_weights(builder(path), args.tokens), top=args.top))
        return
    if args.cmd == "diff":
        print(report(diff(Ledger.load(args.base), Ledger.load(args.new)), limit=args.limit))
        return
    unknown = [a for a in args.apps if a not in APPS]
    if unknown:
        ap.error(f"unknown apps {unknown}; known: {', '.join(APPS)}")
    new = record(args.apps or available())
    if args.out:
        new.save(args.out)
    if args.update:
        base = Ledger.load(BASELINE) if os.path.exists(BASELINE) else Ledger()
        base.merge(new)
        base.save(BASELINE)
        print(f"{BASELINE}: {len(new.apps)} app(s) written")
    elif not args.out:
        base = Ledger.load(BASELINE) if os.path.exists(BASELINE) else Ledger()
        print(report(diff(base, new, new.apps), limit=args.limit))


if __name__ == "__main__":
    main()
