#!/usr/bin/env python3
"""Bump the mlir-aie / llvm-aie pins in pyproject.toml to the newest published wheels.

Steps: fetch the find-links release pages, pick the highest version of each wheel,
rewrite the pins, `uv lock`, `uv sync --extra examples`, then verify the installed
versions and that `artlite._native` imports.

Usage (from anywhere inside the repo):
    python .claude/skills/update-deps/update_deps.py            # update + install + verify
    python .claude/skills/update-deps/update_deps.py --check    # only report available updates
    python .claude/skills/update-deps/update_deps.py --no-sync  # update pins + uv.lock only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "pyproject.toml"

# package name as written in pyproject -> (release page, wheel-filename version regex)
SOURCES: dict[str, tuple[str, str]] = {
    "mlir_aie": (
        "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4",
        r"mlir_aie-(\d+\.\d+\.\d+\.dev\d+\+g[0-9a-f]+)-",
    ),
    "llvm_aie": (
        "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly",
        r"llvm_aie-(\d+\.\d+\.\d+\.\d+\+[0-9a-f]+)-",
    ),
}


def pin_pattern(name: str) -> re.Pattern[str]:
    # Accept either mlir_aie / mlir-aie spelling in pyproject.
    alt = name.replace("_", "[-_]")
    return re.compile(rf'"({alt})==([^"]+)"')


def current_pin(text: str, name: str) -> str:
    m = pin_pattern(name).search(text)
    if not m:
        sys.exit(f"error: no '{name}==...' pin found in {PYPROJECT}")
    return m.group(2)


def latest_version(url: str, regex: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode()
    found = set(re.findall(regex, html))
    if not found:
        sys.exit(f"error: no wheels matched on {url} (page layout changed?)")
    # Local segment (+hash) doesn't order meaningfully; compare the public part.
    # For llvm-aie the 4th component is YYYYMMDDNN, so the newest nightly wins.
    return max(found, key=lambda v: Version(v.split("+")[0]))


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def verify(expected: dict[str, str]) -> None:
    py = REPO / ".venv" / "bin" / "python"
    env = {**os.environ, "PATH": f"{REPO / '.venv' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    code = (
        "import importlib.metadata as m, json, artlite, artlite._native;"
        f"print(json.dumps({{n: m.version(n) for n in {list(expected)!r}}}))"
    )
    out = subprocess.run([str(py), "-c", code], cwd=REPO, env=env, check=True,
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    installed: dict[str, str] = json.loads(out)
    bad = {n: (installed[n], v) for n, v in expected.items() if installed[n] != v}
    for n, v in installed.items():
        print(f"installed {n} {v}")
    if bad:
        sys.exit(f"error: installed versions differ from pins: {bad}")
    print("artlite._native imports OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report available updates, change nothing")
    ap.add_argument("--no-sync", action="store_true", help="update pins and uv.lock, skip uv sync/verify")
    args = ap.parse_args()

    text = PYPROJECT.read_text()
    pins = {n: current_pin(text, n) for n in SOURCES}
    latest = {n: latest_version(*SOURCES[n]) for n in SOURCES}

    changed = False
    for n in SOURCES:
        status = "up to date" if pins[n] == latest[n] else "UPDATE"
        print(f"{n}: pinned {pins[n]}  latest {latest[n]}  [{status}]")
        if pins[n] != latest[n]:
            text = pin_pattern(n).sub(lambda m, v=latest[n]: f'"{m.group(1)}=={v}"', text)
            changed = True

    if args.check:
        sys.exit(0)
    if not changed:
        print("nothing to update")
        return

    PYPROJECT.write_text(text)
    print(f"updated {PYPROJECT.relative_to(REPO)}")
    run(["uv", "lock"])
    if args.no_sync:
        return
    run(["uv", "sync", "--extra", "examples"])
    verify(latest)


if __name__ == "__main__":
    main()
