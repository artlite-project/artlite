"""``artlite-cache``: what the build cache holds, and its cleaning (``artlite.cache``).

It opens no NPU context, so it is safe to run while models run. Only ``clean --all``, ``--unused-for 0`` and
``--other-toolchains`` (another venv may be running) can remove an entry a running process uses.
"""

import argparse
import dataclasses
import os
import sys
import time

from artlite import cache


def human_size(n: int) -> str:
    for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def ago(t: float, now: float) -> str:
    s = max(0.0, now - t)
    for unit, n in (("d", cache.DAY), ("h", 3600), ("min", 60)):
        if s >= n:
            return f"{s / n:.0f} {unit} ago"
    return "just now"


def _rel(path: str) -> str:
    return os.path.relpath(path, cache.ROOT)


def status(args: argparse.Namespace) -> int:
    now = time.time()
    found = cache.scan()
    if not os.path.isdir(cache.ROOT):
        print(f"cache: {cache.ROOT} (empty)")
        return 0
    total = sum(e.size for e in found.entries)
    print(f"cache: {cache.ROOT} ({human_size(total)} in {len(found.entries)} entries)")
    current = cache.TC_PREFIX + cache.toolchain_id()
    for o in sorted(found.owned, key=lambda o: (o.where != "tc", -o.used)):
        mine = [e for e in found.entries if e.base == o.path]
        size = human_size(sum(e.size for e in mine))
        if o.where == "tc":
            versions = ", ".join(f"{k.replace('_', '-')} {v}" for k, v in o.versions.items() if k in cache.WHEELS)
            print(f"  {o.name}  {versions}{'  (this toolchain)' if o.name == current else ''}")
            print(f"  {'':{len(o.name)}}  {size} in {len(mine)} entries, last used {ago(o.used, now)}")
        else:
            print(f"  {o.name + '/':{len(current)}}  {size} in {len(mine)} entries")
    legacy = [e for e in found.entries if e.kind.where == "legacy"]
    if legacy:
        print(f"  {'legacy layout':{len(current)}}  {human_size(sum(e.size for e in legacy))} in {len(legacy)} entries "
              f"(`artlite-cache clean --legacy` removes them now)")
    for path, why in found.skipped:
        print(f"skipped: {_rel(path)} ({why})")
    if found.unknown:
        print(f"not Artlite's, left alone: {', '.join(_rel(p) for p in found.unknown)}")
    policy = cache.policy()
    victims = cache.select(found, policy, now=now)
    print(f"a cleaning now ({policy.describe()}) would remove {len(victims)} entries "
          f"({human_size(sum(e.size for e, _ in victims))})")
    last = cache.last_auto()
    if policy.interval is None:
        print("automatic cleaning: off (ARTLITE_CACHE_GC)")
    elif last:
        removed = last.get("removed", {})
        n, size = sum(v[0] for v in removed.values()), sum(v[1] for v in removed.values())
        errors = f", {len(last['errors'])} errors" if last.get("errors") else ""
        print(f"last automatic cleaning: {ago(last.get('when', 0.0), now)}, removed {n} entries ({human_size(size)})"
              f"{errors}")
    else:
        print("automatic cleaning: not run yet (it runs after a process that built something, once a day)")
    return 0


def list_entries(args: argparse.Namespace) -> int:
    now = time.time()
    entries = [e for e in cache.scan().entries if args.all or e.kind.role in ("entry", "state")]
    if args.kind:
        entries = [e for e in entries if e.kind.name in args.kind]
    entries.sort(key=(lambda e: -e.size) if args.sort == "size" else (lambda e: -e.used))
    print(f"{'LAST USE':12} {'SIZE':>10}  {'KIND':14} {'LABEL':24} PATH")
    for e in entries:
        label = e.kind.name if e.kind.where != "legacy" else f"{e.kind.name} (legacy)"
        if os.path.exists(os.path.join(e.path, cache.TEMPS)):
            label += " (packed)"
        print(f"{ago(e.used, now):12} {human_size(e.size):>10}  {label:14} {e.info().get('label', '')!s:24} "
              f"{_rel(e.path)}")
    return 0


def clean(args: argparse.Namespace) -> int:
    try:
        policy = cache.policy()
        if args.superseded_for is not None:
            policy = dataclasses.replace(policy, superseded=cache.parse_duration(args.superseded_for))
        if args.unused_for is not None:
            policy = dataclasses.replace(policy, unused=cache.parse_duration(args.unused_for))
        if args.max_size is not None:
            policy = dataclasses.replace(policy, size=cache.parse_size(args.max_size))
    except ValueError as e:
        print(f"artlite-cache: {e}", file=sys.stderr)
        return 2
    policy = dataclasses.replace(policy, other_toolchains=args.other_toolchains, all=args.all,
                                       legacy=0.0 if args.legacy or args.all else policy.legacy)
    report = cache.clean(policy, dry_run=args.dry_run, wait=False)
    if report is None:
        print("waiting for another cleaning to finish ...", file=sys.stderr)
        report = cache.clean(policy, dry_run=args.dry_run, wait=True)
    verb = "would remove" if args.dry_run else "removed"
    reasons = ", ".join(f"{reason} {n}" for reason, (n, _) in sorted(report.removed.items()))
    print(f"{verb} {report.entries} entries ({human_size(report.bytes)}){': ' + reasons if reasons else ''}")
    if report.packed[0]:
        print(f"{'would pack' if args.dry_run else 'packed'} the intermediate files of {report.packed[0]} builds "
              f"({human_size(report.packed[1])} {'to free' if args.dry_run else 'freed'})")
    for path, why in report.left:
        print(f"left: {_rel(path)} ({why})")
    for path, why in report.skipped:
        print(f"skipped: {_rel(path)} ({why})")
    for err in report.errors:
        print(f"error: {err}", file=sys.stderr)
    return 1 if report.errors else 0


def pack(args: argparse.Namespace) -> int:
    try:
        after = cache.parse_duration(args.unused_for) if args.unused_for is not None else cache.policy().pack
    except ValueError as e:
        print(f"artlite-cache: {e}", file=sys.stderr)
        return 2
    if after is None:
        print("packing is off (ARTLITE_CACHE_GC pack=off); --unused-for D packs anyway", file=sys.stderr)
        return 0
    report = cache.pack_all(after, dry_run=args.dry_run)
    print(f"{'would pack' if args.dry_run else 'packed'} the intermediate files of {report.packed[0]} builds "
          f"({human_size(report.packed[1])} {'to free' if args.dry_run else 'freed'})")
    for err in report.errors:
        print(f"error: {err}", file=sys.stderr)
    return 1 if report.errors else 0


def unpack(args: argparse.Namespace) -> int:
    status = 0
    for p in args.paths:
        directory = os.path.abspath(p)
        while directory != os.path.dirname(directory) and not cache.is_build_dir(directory):
            directory = os.path.dirname(directory)
        if not cache.is_build_dir(directory):
            print(f"artlite-cache: {p}: not in a build directory (rtc_<key>/, fused_<key>/)", file=sys.stderr)
            status = 2
        elif cache.unpack(directory):
            print(f"unpacked {_rel(directory)}")
        else:
            print(f"{_rel(directory)}: nothing packed")
    return status


def path(args: argparse.Namespace) -> int:
    print(cache.common() if args.common else cache.toolchain_dir())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="artlite-cache", description=(
        "Show and clean Artlite's build cache (ARTLITE_CACHE_DIR). Only Artlite's own entries are ever removed; the "
        "command opens no NPU context."))
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="the toolchain directories, their sizes, what a cleaning would remove (default)")
    p = sub.add_parser("list", help="one line per entry: last use, size, kind, label, path")
    p.add_argument("--kind", action="append", help="only entries of this kind (rtc, fused, pch, render, vision, ...)")
    p.add_argument("--sort", choices=("used", "size"), default="used", help="most recently used first, or largest")
    p.add_argument("--all", action="store_true", help="also leftovers, locks and trash")
    p = sub.add_parser("clean", help="remove old entries: the automatic cleaning's rules now (ARTLITE_CACHE_GC; "
                                     "default: superseded for a day, unused for 30 days)")
    p.add_argument("-n", "--dry-run", action="store_true", help="only print what would be removed")
    p.add_argument("--unused-for", metavar="D", help="entries not used during the last D (30m, 12h, 1d, 2w)")
    p.add_argument("--superseded-for", metavar="D",
                   help="superseded entries (an older version of a design built again since) not used during the "
                        "last D")
    p.add_argument("--other-toolchains", action="store_true", help="every toolchain directory but this venv's")
    p.add_argument("--max-size", metavar="S", help="then the least recently used entries down to S (500M, 30G)")
    p.add_argument("--legacy", action="store_true", help="every entry of the layout before toolchain directories")
    p.add_argument("--all", action="store_true",
                   help="every entry Artlite owns, also state (pace calibrations, the server's model index) and "
                        "designs a running process uses")
    p = sub.add_parser("pack", help="pack the intermediate files of builds not used for a day (ARTLITE_CACHE_GC "
                                    "pack=); the automatic cleaning does it too")
    p.add_argument("-n", "--dry-run", action="store_true", help="only print what would be packed")
    p.add_argument("--unused-for", metavar="D", help="builds not used during the last D (0: every build)")
    p = sub.add_parser("unpack", help="restore a build's packed intermediate files (kernel sources stay plain)")
    p.add_argument("paths", nargs="+", metavar="PATH", help="a build directory, or a path inside one")
    p = sub.add_parser("path", help="print this toolchain's cache directory")
    p.add_argument("--common", action="store_true", help="print the directory of what no toolchain builds")
    args = parser.parse_args(argv)
    return {"list": list_entries, "clean": clean, "pack": pack, "unpack": unpack, "path": path}.get(
        args.command, status)(args)


if __name__ == "__main__":
    sys.exit(main())
