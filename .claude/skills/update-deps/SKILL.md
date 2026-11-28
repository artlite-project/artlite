---
name: update-deps
description: Upgrade Artlite's pinned mlir-aie and llvm-aie (Peano) toolchain wheels to the latest nightly builds, relock with uv, reinstall, and verify. Use when asked to update/bump/upgrade the toolchain, mlir-aie, llvm-aie, Peano, aiecc, or nightly wheels, or when a fix is known to exist only in a newer toolchain build.
---

# Update toolchain dependencies

One command performs the whole upgrade (fetch the latest wheels → rewrite the pins in `pyproject.toml` → `uv lock` →
`uv sync` → verify the installed versions and `import artlite._native`):

```bash
python .claude/skills/update-deps/update_deps.py
```

Run it from the repo root. Useful flags:
- `--check` — only print pinned vs. latest versions; changes nothing. Use this first if the user only asked whether
  updates exist.
- `--no-sync` — update the pins and `uv.lock` but do not install.

The script exits non-zero on any failure (no wheels matched, `uv lock` / `uv sync` failed, installed version
mismatch, import failure). Report its output to the user; do not retry blindly.

## After a successful update

- No cache clearing is needed: RTC cache keys include the `mlir_aie` / `llvm_aie` versions, so kernels recompile
  automatically. Old `rtc_*` directories in `~/.cache/artlite/` simply become orphaned; delete them to reclaim disk
  space if desired.
- `requires-python` in `pyproject.toml` ends below the first Python version without an `mlir_aie` wheel (uv picks the
  newest allowed Python). Raise it when the new wheels add a Python version, then run `uv lock`.
- Run the survey: `python -m tests.schedule_ledger record` builds every app's designs with the new wheels (~50 minutes
  cold, spent on the builds) and prints what changed against the committed baseline, worst first: loops that lost
  pipelining, higher II / NS / spills, function cycles and bytes, and cores with less than 256 B free. Act on each
  entry: fix a kernel (run `variant_search` on the loop to find a hint the new compiler needs) or a design that the
  new mlir-aie refuses; `rank APP` tells whether a slower loop carries weight in a prefill chunk. Then run
  `record --update` and commit the refreshed baseline together with the pins; its diff shows the bump's effect on every
  shipped kernel. Run benchmarks only for the models whose ranked loops changed.
- Run `pytest tests -m smoke` (one NPU case per component, plus the apps' full designs and ledgers), not the full
  suite unless asked.
- Regenerate the issue-slot and shared-port tables of `variant_search` and the `ARTLITE_MEASURE_II` report from the
  new llvm-aie commit: `python -m artlite.gen_isa_tables`. It reads the `.td` files of the wheel's commit from the
  `../llvm-aie` checkout (run `git -C ../llvm-aie fetch` when the checkout lacks that commit) and uses its
  `build/bin/llvm-tblgen`.
- `variant_search --why` traces come from the debug build in `../llvm-aie/build`; its report warns when that build's
  commit differs from the wheel's. Tell the user if the build should be rebuilt.
- Commit. The diff should usually contain `pyproject.toml`, `uv.lock` and `tests/schedule_ledger.jsonl` (plus
  `src/artlite/aie2p_isa.py`).

## Gotchas

- Wheels come from the GitHub release pages listed in `[tool.uv] find-links` (`mlir-aie` `latest-wheels-5`,
  `llvm-aie` `nightly`). mlir-aie occasionally moves its rolling wheels to the next numbered tag, and the old tag stops
  getting builds. The script stops when that next tag exists; point `SOURCES` and `find-links` at it. If the script
  finds no matches, the page layout has changed.
- llvm-aie's 4th version component is `YYYYMMDDNN`. It advances nightly even when the commit hash does not, so a "new"
  version may contain no compiler changes.
- A pin held below the newest wheel states the reason on its `pyproject.toml` line. Before the script moves it, check
  that the reason no longer applies (the upstream fix is in the newer wheel).
- Only wheels that this machine can install count: a nightly can publish its Windows wheel without the Linux one (when
  its Linux build job failed), and the script then keeps the newest Linux build. A failed `uv sync` leaves no `.venv`.
- A toolchain bump can change kernel schedules (II) or break compilation of RTC templates. If a test regresses,
  compare against the previous pin (`git diff pyproject.toml`) before debugging the operator.
- Check that every flag Artlite passes still exists: `toolchain.CORE_LLC_FLAGS`, the designs' `AIECC_FLAGS`, and the
  loop hints in `include/operators.h`. These are llc options, except `aie-loop-versioning`, which belongs to an IR pass;
  they are documented in llvm-aie's PERF_OPT_GUIDE.md, whose "Changes" section lists what a newer Peano reads
  differently.
- mlir-aie moves passes between anchors (for example, `aie-place-tiles` runs per module, not per device). Artlite's
  own pass pipelines are in `operators/l1_layout.py` and `profiling.py`.
