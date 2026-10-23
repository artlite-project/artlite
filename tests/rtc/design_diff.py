"""Compare two ``pytest tests/rtc --dump-designs DIR`` dumps (a refactoring must not change designs).

    python tests/rtc/design_diff.py BEFORE AFTER [--alpha] [--asm] [--partial] [--rename REGEX=REPL ...] [--show N]
    python tests/rtc/design_diff.py BEFORE --affected REGEX [--compile]

Per test, the designs that changed are paired (same MLIR and kernel entry points first). Reported:
tests only in one dump, pairs whose MLIR differs (with the first differing lines) and pairs whose
kernel source alone differs. Kernel names are compared without their hashes (``kernel_source``: a change of a
kernel's text, comments included, renames it and what reaches it). ``--alpha`` compares MLIR modulo symbol names
(renamed FIFOs); ``--rename`` rewrites AFTER's sources and MLIR (e.g. a renamed kernel prefix); ``--asm`` compiles
both sources of every source-only pair and compares the sets of distinct function bodies (labels and symbol operands
normalized, names ignored: a function moved to another name, or two copies merged, compares equal).

A change of shared kernel code needs no full dump after it. Before the change, ``--affected REGEX --compile`` prints
the tests whose designs' kernel sources in BEFORE match (the code the change will touch) and caches those designs'
compiled functions next to their sources (``<design>.asm.json``, compiled with the headers of the tree that runs it:
still the old ``kernel_lib.h``); after it, ``pytest $(...) --dump-designs AFTER`` renders just those tests and
``--partial --asm`` compares them. (A BEFORE made in another checkout needs its gitignored ``tests/rtc/refs`` and
``tests/datasets``: tests that read them skip before rendering.)
"""

import argparse
import collections
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile

_HASHED = re.compile(r"\b(\w+)_[0-9a-f]{8}\b")  # (a kernel name: ``kernel_source``)
_MANGLED = re.compile(r"(?<=[A-Za-z0-9])_[0-9a-f]{8}(?![0-9a-f])")  # (its hash in a mangled C++ name too)


def unhashed(name: str) -> str:
    return _HASHED.sub(r"\1_H", name)


class Dump:
    def __init__(self, path: str, renames: list[tuple[str, str]], alpha: bool):
        self.path, self.renames, self.alpha = path, renames, alpha
        self.tests: dict[str, list[str]] = collections.defaultdict(list)
        with open(os.path.join(path, "index.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                self.tests[r["test"]].append(r["design"])

    def text(self, design: str, ext: str, *, hashes: bool = False) -> str:
        """A design's file with the renames applied (and its kernel names' hashes dropped unless ``hashes``)."""
        with open(os.path.join(self.path, design + ext)) as f:
            text = f.read()
        for pat, repl in self.renames:
            text = re.sub(pat, repl, text)
        return text if hashes else unhashed(text)

    def mlir(self, design: str) -> str:
        text = self.text(design, ".mlir")
        if not self.alpha:
            return text
        names: dict[str, int] = {}
        return re.sub(r"@([A-Za-z_][\w$.]*)", lambda m: f"@S{names.setdefault(m.group(1), len(names))}", text)

    def key(self, design: str) -> tuple[str, list[str]]:
        """Pairing key: the MLIR and the kernel entry points it declares."""
        return self.mlir(design), sorted(re.findall(r"func\.func private @(\w+)", self.text(design, ".mlir")))


def pairs(a: Dump, b: Dump) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for test in set(a.tests) & set(b.tests):
        ca, cb = collections.Counter(a.tests[test]), collections.Counter(b.tests[test])
        left = sorted((cb - ca).elements())
        rest = []
        for x in sorted((ca - cb).elements()):
            match = next((y for y in left if a.key(x) == b.key(y)), None)
            if match is None:
                rest.append(x)
            else:
                left.remove(match)
                out.add((x, match))
        out.update(zip(rest, left))  # (unpaired by key: MLIR changed)
    return out


def functions(asm: str) -> dict[str, list[str]]:
    """{function: instructions}, local labels (blocks, jump and switch tables) and symbol operands normalized; symbol
    assignments (aliases) dropped."""
    out: dict[str, list[str]] = {}
    cur = None
    for line in asm.splitlines():
        m = re.match(r"^([A-Za-z_$][\w$.]*):", line)
        if m and not m.group(1).startswith(".L"):
            cur = m.group(1)
            out[cur] = []
            continue
        t = line.split("//")[0].strip()
        if cur is None or not t or t.startswith(".") or re.fullmatch(r"[\w$.]+ = [\w$.]+", t):
            continue
        t = re.sub(r"\.LJTI\d+_\d+|\.Lswitch\.table\.[\w$.]+", "L", t)
        t = re.sub(r"#[A-Za-z_]\w*", "#SYM", t)
        out[cur].append(re.sub(r"\s+", " ", re.sub(r"\.LBB\d+_\d+|\.L_LEnd\d+|\.Ltmp\d+", "L", t)))
    return out


class Asm:
    """Designs' compiled functions ({name without hash: instructions}, None without kernels, a string: the compile
    error), cached as ``<design>.asm.json`` in the dump; compiled with this tree's headers."""

    def __init__(self) -> None:
        from artlite.operators import base

        self.cmd = [*base.CLANG_CMD_COMMON, f"--target={base.AIE_TARGET}"]
        self.pch = base._compile_pch(clang_cmd=self.cmd)
        self.tmp = tempfile.mkdtemp(prefix="design_diff_")

    def __call__(self, dump: Dump, design: str) -> dict[str, list[str]] | str | None:
        cache = os.path.join(dump.path, design + ".asm.json")
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)
        text = dump.text(design, ".cc", hashes=True)  # (compiled with their names: one hash dropped, two copies)
        if not text.strip():
            out = None
        else:
            src = os.path.join(self.tmp, f"{abs(hash(dump.path))}_{design}.cc")
            with open(src, "w") as f:
                f.write(text)
            r = subprocess.run([*self.cmd, "-include-pch", self.pch, "-S", src, "-o", src + ".s"], capture_output=True,
                               text=True, check=False)
            if r.returncode:
                return "compile failed: " + r.stderr[-300:]
            with open(src + ".s") as f:
                out = {unhashed(k): v for k, v in functions(f.read()).items()}
        with open(cache, "w") as f:
            json.dump(out, f)
        return out


def compare_asm(a: Dump, b: Dump, todo: list[tuple[str, str]]) -> None:
    from artlite import parallel_compile

    asm, dumps = Asm(), (a, b)
    distinct = sorted({(0, x) for x, _ in todo} | {(1, y) for _, y in todo})  # (each compiled once)
    got = dict(zip(distinct, parallel_compile.parallel_map(lambda d: asm(dumps[d[0]], d[1]), distinct)))

    def one(pair: tuple[str, str]) -> tuple[str, str]:
        fa, fb = got[0, pair[0]], got[1, pair[1]]
        if isinstance(fa, str) or isinstance(fb, str):
            return pair[0], fa if isinstance(fa, str) else fb
        if fa is None or fb is None:
            return pair[0], "identical" if fa == fb else "kernel added or removed"
        ba, bb = ({tuple(v) for v in f.values()} for f in (fa, fb))
        if ba == bb:
            return pair[0], "identical"
        gone = sorted({_MANGLED.sub("_H", k) for k, v in fa.items() if tuple(v) not in bb})
        new = sorted({_MANGLED.sub("_H", k) for k, v in fb.items() if tuple(v) not in ba})
        return pair[0], f"bodies before only: {', '.join(gone)}; after only: {', '.join(new)}"

    results = [one(pair) for pair in todo]
    print("asm:", dict(collections.Counter("identical" if r == "identical" else "different" for _, r in results)))
    by = collections.defaultdict(list)  # (a result: the designs that have it)
    for design, r in results:
        if r != "identical":
            by[r].append(design)
    for r, designs in sorted(by.items(), key=lambda x: -len(x[1])):
        print(f"  {len(designs)} design(s), e.g. {designs[0]}: {r}")


def affected(dump: Dump, pattern: str) -> tuple[list[str], list[str]]:
    """The designs whose kernel source matches ``pattern`` and the tests that render them."""
    rx = re.compile(pattern)
    designs = sorted({d for ds in dump.tests.values() for d in ds if rx.search(dump.text(d, ".cc", hashes=True))})
    hit = set(designs)
    return designs, sorted(t for t, ds in dump.tests.items() if hit.intersection(ds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after", nargs="?")
    ap.add_argument("--alpha", action="store_true", help="compare MLIR modulo symbol names")
    ap.add_argument("--asm", action="store_true", help="compile source-only changes and compare the asm")
    ap.add_argument("--partial", action="store_true", help="compare only the tests in AFTER")
    ap.add_argument("--affected", metavar="REGEX", help="print the tests of BEFORE's designs whose source matches")
    ap.add_argument("--compile", action="store_true", help="with --affected: compile those designs (the asm cache)")
    ap.add_argument("--rename", action="append", default=[], metavar="REGEX=REPL", help="rewrite AFTER's texts")
    ap.add_argument("--show", type=int, default=5, help="MLIR diffs to print")
    args = ap.parse_args()
    a = Dump(args.before, [], args.alpha)
    if args.affected:
        designs, tests = affected(a, args.affected)
        if args.compile:
            from artlite import parallel_compile

            asm = Asm()
            failed = [d for d, r in zip(designs, parallel_compile.parallel_map(lambda d: asm(a, d), designs))
                      if isinstance(r, str)]
            print(f"{len(designs)} designs compiled ({len(failed)} failed: {failed[:4]})", file=sys.stderr)
        print("\n".join(tests))
        return
    b = Dump(args.after, [tuple(r.split("=", 1)) for r in args.rename], args.alpha)
    if args.partial:
        a.tests = collections.defaultdict(list, {t: a.tests[t] for t in b.tests if t in a.tests})
    missing, extra = sorted(set(a.tests) - set(b.tests)), sorted(set(b.tests) - set(a.tests))
    changed = sorted(pairs(a, b))
    mlir = [(x, y) for x, y in changed if a.mlir(x) != b.mlir(y)]
    src = [(x, y) for x, y in changed if a.mlir(x) == b.mlir(y)]
    print(f"tests {len(a.tests)} vs {len(b.tests)} (only before: {len(missing)}, only after: {len(extra)}); "
          f"changed designs: {len(changed)}, MLIR differs: {len(mlir)}, kernel source only: {len(src)}")
    for t in missing[:args.show]:
        print("  only before:", t)
    for t in extra[:args.show]:
        print("  only after:", t)
    print("  MLIR diffs by operator:", dict(collections.Counter(x.rsplit("_", 1)[0] for x, _ in mlir)))
    print("  source diffs by operator:", dict(collections.Counter(x.rsplit("_", 1)[0] for x, _ in src)))
    for x, y in mlir[:args.show]:
        d = [line for line in difflib.unified_diff(a.mlir(x).splitlines(), b.mlir(y).splitlines(), lineterm="", n=0)
             if not line.startswith(("---", "+++", "@@"))]
        print(f"--- {x} vs {y}: {len(d)} lines differ")
        for line in d[:6]:
            print("     ", line[:160])
    if args.asm and src:
        compare_asm(a, b, src)


if __name__ == "__main__":
    main()
