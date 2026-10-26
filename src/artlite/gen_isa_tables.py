"""Regenerate ``aie2p_isa.py``: the issue slot of each AIE2P assembly mnemonic and the register / memory ports that
ops of several slots share, from llvm-aie's TableGen sources at the commit of the installed llvm-aie wheel.

    python -m artlite.gen_isa_tables [--llvm-aie DIR] [--tblgen PATH] [--check]

Run it after a toolchain upgrade (skill ``update-deps``). It needs an llvm-aie checkout that has the wheel's commit
(default: ``llvm-aie`` next to Artlite; ``git fetch`` there when the commit is missing) and its ``llvm-tblgen``
(default: the checkout's ``build/bin``). The .td files are read from git at the wheel's commit, so the checkout's
HEAD and working tree do not matter. ``--check`` changes nothing and exits 1 when ``aie2p_isa.py`` is stale
(``tests/test_variant_search.py`` runs it when a checkout is there).

Per mnemonic (the first word of an AsmString, non-pseudo instructions with a slot): its slots, and per port the
cycles it occupies the port's unit (InstrItinData stages of its Itinerary, Reserved stages skipped), common to every
instruction of that mnemonic.
"""

import argparse
import io
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "aie2p_isa.py"
LLVM_AIE = ROOT.parents[2] / "llvm-aie"  # (the checkout next to Artlite's)

# The units that ops of several issue slots share, which the per-slot counts do not show (short name: unit). Units
# of control registers, address generators and the like also span slots but do not bound kernel loops.
PORTS = {
    "rm": "L_RM_PORT",
    "wm": "L_WM_PORT",
    "qwa": "QEY_WA_L0_PORT",
    "qwb": "QEY_WB_L0_PORT",
    "qrs": "QEY_RS_L0_PORT",
    "dwm": "DM_WM_L0_PORT",
    "ada": "DM_ADA",
}
SLOTS = ("lda", "ldb", "st", "vec", "mv", "alu", "lng")


def clang_commit(clang: str | Path) -> str | None:
    """The llvm-aie commit a clang++ was built from (its --version), None when unknown."""
    try:
        result = subprocess.run([str(clang), "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"llvm-aie(?:\.git)?\s+([0-9a-f]{7,40})\)", result.stdout)
    return match.group(1) if match else None


def _sources(llvm_aie: Path, commit: str, directory: Path) -> Path:
    """Extract the .td files of commit into directory; the path of AIE2P.td."""
    process = subprocess.run(
        ["git", "-C", str(llvm_aie), "archive", "--format=tar", commit, "--",
         ":(glob)llvm/include/llvm/**/*.td", ":(glob)llvm/lib/Target/AIE/**/*.td"],
        capture_output=True, check=False,
    )
    if process.returncode:
        raise RuntimeError(f"git archive {commit} in {llvm_aie} failed (fetch the commit there?): "
                           f"{process.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(process.stdout)) as archive:
        archive.extractall(directory, filter="data")
    return directory / "llvm/lib/Target/AIE/aie2p/AIE2P.td"


def _records(text: str) -> dict[str, dict[str, str]]:
    """The fields of each def of llvm-tblgen -print-records output (one-line values only)."""
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("def "):
            current = records.setdefault(line[4:].split(None, 1)[0], {})
        elif line == "}":
            current = None
        elif current is not None and line.startswith("  ") and line.endswith(";") and " = " in line:
            left, value = line[2:-1].split(" = ", 1)
            current[left.rsplit(None, 1)[-1]] = value
    return records


def _list(value: str) -> list[str]:
    return re.findall(r"\w+", value)


def tables(records: dict[str, dict[str, str]]) -> tuple[dict[str, list[str]], dict[str, Counter[str]]]:
    """Per mnemonic its slots (sorted) and the cycles it occupies each port (short names of PORTS)."""
    stages = {}
    for name, fields in records.items():
        if "Units" in fields and "Cycles" in fields and "Kind" in fields:
            stages[name] = (_list(fields["Units"]), int(fields["Cycles"]), "Reserved" in fields["Kind"])
    itineraries: dict[str, list[str]] = {}
    for fields in records.values():
        if "TheClass" in fields and "Stages" in fields:
            itineraries.setdefault(fields["TheClass"], _list(fields["Stages"]))
    units_of = {unit: port for port, unit in PORTS.items()}
    slots: dict[str, set[str]] = {}
    ports: dict[str, Counter[str]] = {}
    for fields in records.values():
        asm = fields.get("AsmString", '""').strip('"').split()
        slot = fields.get("Slot", "").removesuffix("_slot")
        if not asm or fields.get("isPseudo") == "1" or slot in ("", "?", "nop") or "Itinerary" not in fields:
            continue
        mnemonic = asm[0]
        if mnemonic.startswith("nop"):
            continue  # (not counted)
        busy: Counter[str] = Counter()
        for stage in itineraries.get(fields["Itinerary"], []):
            units, cycles, reserved = stages.get(stage, ([], 0, True))
            if not reserved:
                for unit in units:
                    if unit in units_of:
                        busy[units_of[unit]] += max(cycles, 1)
        slots.setdefault(mnemonic, set()).add(slot)
        ports[mnemonic] = busy if mnemonic not in ports else ports[mnemonic] & busy
    unknown = {slot for found in slots.values() for slot in found} - set(SLOTS)
    if unknown:
        raise RuntimeError(f"unknown issue slots {sorted(unknown)}: update SLOTS")
    return {mnemonic: sorted(found, key=SLOTS.index) for mnemonic, found in slots.items()}, ports


def _wrapped(words: list[str], indent: str) -> list[str]:
    """Python string literals of the words, space-separated, at most 120 columns per line."""
    lines, line = [], ""
    for word in words:
        if line and len(indent) + len(line) + len(word) + 4 > 120:
            lines.append(line)
            line = ""
        line += word + " "
    lines.append(line.rstrip())
    return [f'{indent}"{text}"' for text in lines]


def render(commit: str, slots: dict[str, list[str]], ports: dict[str, Counter[str]]) -> str:
    out = [
        f'"""AIE2P issue slots and shared ports per assembly mnemonic, from llvm-aie {commit[:12]}.',
        "",
        "Generated by ``python -m artlite.gen_isa_tables`` (run it after a toolchain upgrade); do not edit.",
        '"""',
        "",
        f'LLVM_AIE_COMMIT = "{commit}"',
        "",
        "# The issue slot of each mnemonic of one slot (a lng op takes the alu and the mv slot of its bundle).",
        "SLOT_MNEMONICS = {",
    ]
    for slot in SLOTS:
        words = sorted(mnemonic for mnemonic, found in slots.items() if found == [slot])
        out += [f'    "{slot}": (', *_wrapped(words, "        "), "    ),"]
    out += [
        "}",
        "",
        "# Mnemonics of several slots: the operands tell them apart (variant_search.slot_of).",
        "SHARED_MNEMONICS = {",
        *(f'    "{mnemonic}": "{" ".join(found)}",' for mnemonic, found in sorted(slots.items()) if len(found) > 1),
        "}",
        "",
        "# Register-file and memory ports that ops of several slots share (short name: llvm-aie FuncUnit): a loop's",
        "# II is at least the cycles its ops occupy each. A mnemonic*N occupies the port N cycles.",
        "PORT_UNITS = {",
        *(f'    "{port}": "{unit}",' for port, unit in PORTS.items()),
        "}",
        "PORT_MNEMONICS = {",
    ]
    for port in PORTS:
        words = sorted(
            mnemonic + (f"*{busy[port]}" if busy[port] > 1 else "")
            for mnemonic, busy in ports.items()
            if busy[port]
        )
        out += [f'    "{port}": (', *_wrapped(words, "        "), "    ),"]
    out += ["}", ""]
    return "\n".join(out)


def generate(*, llvm_aie: Path = LLVM_AIE, tblgen: Path | None = None, commit: str | None = None) -> str:
    """The text of aie2p_isa.py for the wheel's commit (or commit)."""
    if commit is None:
        from .operators.base import CLANGPP

        commit = clang_commit(CLANGPP)
        if commit is None:
            raise RuntimeError(f"no llvm-aie commit in `{CLANGPP} --version`")
    tblgen = tblgen or llvm_aie / "build" / "bin" / "llvm-tblgen"
    if not tblgen.exists():
        raise RuntimeError(f"no llvm-tblgen at {tblgen} (build llvm-aie, or pass --tblgen)")
    with tempfile.TemporaryDirectory() as directory:
        top = _sources(llvm_aie, commit, Path(directory))
        include = Path(directory) / "llvm"
        process = subprocess.run(
            [str(tblgen), "-print-records", "-I", str(include / "include"), "-I", str(top.parents[1]),
             "-I", str(top.parent), str(top)],
            capture_output=True, text=True, check=False,
        )
    if process.returncode:
        raise RuntimeError(f"llvm-tblgen failed: {process.stderr.strip()[:2000]}")
    return render(commit, *tables(_records(process.stdout)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m artlite.gen_isa_tables", description=__doc__.split("\n\n")[0])
    parser.add_argument("--llvm-aie", type=Path, default=LLVM_AIE, help="llvm-aie checkout (default: %(default)s)")
    parser.add_argument("--tblgen", type=Path, help="llvm-tblgen (default: the checkout's build/bin)")
    parser.add_argument("--commit", help="llvm-aie commit (default: the installed wheel's)")
    parser.add_argument("--check", action="store_true", help="exit 1 when aie2p_isa.py is stale; write nothing")
    args = parser.parse_args(argv)
    text = generate(llvm_aie=args.llvm_aie, tblgen=args.tblgen, commit=args.commit)
    current = TABLES.read_text() if TABLES.exists() else ""
    if args.check:
        print(f"{TABLES.name}: {'up to date' if text == current else 'stale: run python -m artlite.gen_isa_tables'}")
        return int(text != current)
    if text != current:
        TABLES.write_text(text)
    print(f"{TABLES.name}: {'unchanged' if text == current else 'written'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
