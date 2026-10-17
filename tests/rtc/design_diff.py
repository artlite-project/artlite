"""Compare two ``pytest tests/rtc --dump-designs DIR`` dumps (a refactoring must not change designs).

    python tests/rtc/design_diff.py BEFORE AFTER [--alpha] [--asm] [--rename REGEX=REPL ...] [--show N]

Per test, the designs that changed are paired (same MLIR and kernel entry points first). Reported:
tests only in one dump, pairs whose MLIR differs (with the first differing lines) and pairs whose
kernel source alone differs. Kernel names are compared without their hashes (``kernel_source``: a change of a
kernel's text, comments included, renames it and what reaches it). ``--alpha`` compares MLIR modulo symbol names
(renamed FIFOs); ``--rename`` rewrites AFTER's sources and MLIR (e.g. a renamed kernel prefix); ``--asm`` compiles
both sources of every source-only pair and compares the functions' instructions (labels and
symbol operands normalized).
"""

import argparse
import collections
import difflib
import json
import os
import re
import subprocess
import tempfile

_HASHED = re.compile(r"\b(\w+)_[0-9a-f]{8}\b")  # (a kernel name: ``kernel_source``)


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


def _key(item: tuple[str, list[str]]) -> tuple[str, tuple[str, ...]]:
    return item[0], tuple(item[1])


def compare_asm(a: Dump, b: Dump, todo: list[tuple[str, str]]) -> None:
    from artlite import parallel_compile
    from artlite.operators import base

    cmd = [*base.CLANG_CMD_COMMON, f"--target={base.AIE_TARGET}"]
    pch = base._compile_pch(clang_cmd=cmd)
    tmp = tempfile.mkdtemp(prefix="design_diff_")

    def asm(dump: Dump, design: str, tag: str) -> list[tuple[str, list[str]]] | str | None:
        text = dump.text(design, ".cc", hashes=True)  # (compiled with their names: one hash dropped, two copies)
        if not text.strip():
            return None
        src = os.path.join(tmp, f"{tag}_{design}.cc")
        with open(src, "w") as f:
            f.write(text)
        r = subprocess.run([*cmd, "-include-pch", pch, "-S", src, "-o", src + ".s"], capture_output=True, text=True,
                           check=False)
        if r.returncode:
            return "compile failed: " + r.stderr[-300:]
        with open(src + ".s") as f:
            return sorted((unhashed(k), v) for k, v in functions(f.read()).items())

    def one(pair: tuple[str, str]) -> tuple[str, str]:
        fa, fb = asm(a, pair[0], "a"), asm(b, pair[1], "b")
        if isinstance(fa, str) or isinstance(fb, str):
            return pair[0], fa if isinstance(fa, str) else fb
        if fa is None or fb is None:
            return pair[0], "identical" if fa == fb else "kernel added or removed"
        diff = sorted({k for k, _ in set(map(_key, fa)) ^ set(map(_key, fb))})
        return pair[0], "identical" if fa == fb else f"DIFF in {diff[:6]}"

    results = parallel_compile.parallel_map(one, todo)
    print("asm:", dict(collections.Counter("identical" if r == "identical" else "different" for _, r in results)))
    for design, r in results:
        if r != "identical":
            print(f"  {design}: {r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--alpha", action="store_true", help="compare MLIR modulo symbol names")
    ap.add_argument("--asm", action="store_true", help="compile source-only changes and compare the asm")
    ap.add_argument("--rename", action="append", default=[], metavar="REGEX=REPL", help="rewrite AFTER's texts")
    ap.add_argument("--show", type=int, default=5, help="MLIR diffs to print")
    args = ap.parse_args()
    a = Dump(args.before, [], args.alpha)
    b = Dump(args.after, [tuple(r.split("=", 1)) for r in args.rename], args.alpha)
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
