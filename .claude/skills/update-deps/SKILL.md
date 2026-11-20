---
name: update-deps
description: Upgrade Artlite's pinned mlir-aie and llvm-aie (Peano) toolchain wheels to the latest nightly builds, relock with uv, reinstall, and verify. Use when asked to update/bump/upgrade the toolchain, mlir-aie, llvm-aie, Peano, aiecc, or nightly wheels, or when a fix is known to exist only in a newer toolchain build.
---

# Update toolchain dependencies

One command does the whole upgrade (fetch latest wheels → rewrite pins in `pyproject.toml` → `uv lock` → `uv sync` → verify installed versions and `import artlite._native`):

```bash
python .claude/skills/update-deps/update_deps.py
```

Run from the repo root. Useful flags:
- `--check` — only print pinned vs. latest versions; changes nothing. Use this first if the user just asked whether updates exist.
- `--no-sync` — update pins and `uv.lock` but don't install.

The script exits non-zero on any failure (no wheels matched, `uv lock`/`uv sync` failed, installed version mismatch, import failure). Report its output to the user; don't retry blindly.

## After a successful update

- No cache clearing needed: RTC cache keys include the `mlir_aie`/`llvm_aie` versions, so kernels recompile automatically. Old `rtc_*` dirs in `~/.cache/artlite/` just become orphaned; delete them to reclaim disk if desired.
- `requires-python` in `pyproject.toml` ends below the first Python without an `mlir_aie` wheel (uv picks the newest
  allowed Python): raise it when the new wheels add a Python version, then `uv lock`.
- The survey: `python -m tests.schedule_ledger record` builds every app's designs with the new wheels (cold: ~50
  minutes, the builds) and prints what changed against the committed baseline, worst first: loops that lost
  pipelining, II / NS / spills up, function cycles and bytes, cores below 256 B free. Act on each: fix a kernel
  (`variant_search` on the loop: a hint the new compiler needs) or a design the new mlir-aie refuses; `rank APP` tells
  whether a slower loop weighs in a prefill chunk. Then `record --update` and commit the refreshed baseline with the
  pins: its diff is the bump's effect on every shipped kernel. Benches only for the models whose ranked loops changed.
- Run `pytest tests -m smoke` (one NPU case per component and the apps' full designs and ledgers), not the full suite
  unless asked.
- Regenerate the issue slot / shared port tables of `variant_search` and the `ARTLITE_MEASURE_II` report from the new
  llvm-aie commit: `python -m artlite.gen_isa_tables` (reads the .td files of the wheel's commit from the `../llvm-aie`
  checkout: `git -C ../llvm-aie fetch` when it lacks the commit; uses its `build/bin/llvm-tblgen`).
- `variant_search --why` traces come from the debug build in `../llvm-aie/build`; its report warns while that build's
  commit is not the wheel's. Tell the user if it should be rebuilt.
- Commit; the diff usually should be `pyproject.toml` + `uv.lock` + `tests/schedule_ledger.jsonl` (+
  `src/artlite/aie2p_isa.py`).

## Gotchas

- Wheels come from GitHub release pages listed in `[tool.uv] find-links` (`mlir-aie` `latest-wheels-5`, `llvm-aie` `nightly`). mlir-aie moves its rolling wheels to the next numbered tag now and then (the old one stops getting builds): the script stops when that tag exists; point `SOURCES` and `find-links` at it. If the script finds no matches, the page layout changed.
- llvm-aie's 4th version component is `YYYYMMDDNN`; it advances nightly even when the commit hash doesn't, so a "new" version may contain no compiler changes.
- A pin held below the newest wheel says why on its `pyproject.toml` line: check that the reason is gone (the upstream
  fix is in the newer wheel) before the script moves it.
- Only wheels this machine can install count: a nightly can publish its Windows wheel without its Linux one (its
  Linux build job failed), and the script then keeps the newest Linux build. A failed `uv sync` leaves no `.venv`.
- A toolchain bump can change kernel schedules (II) or break compilation of RTC templates. If a test regresses, compare against the previous pin (`git diff pyproject.toml`) before debugging the operator.
- Check that every flag Artlite passes still exists: `toolchain.CORE_LLC_FLAGS`, the designs' `AIECC_FLAGS`, the loop
  hints of `include/operators.h` (llc options; `aie-loop-versioning` is an IR pass's, documented in llvm-aie's
  PERF_OPT_GUIDE.md, whose "Changes" section lists what a newer Peano reads differently).
- mlir-aie moves passes between anchors (`aie-place-tiles`: per module, not per device): Artlite's own pass pipelines
  are `operators/l1_layout.py`'s and `profiling.py`'s.
