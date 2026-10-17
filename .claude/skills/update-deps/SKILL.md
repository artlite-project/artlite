---
name: update-deps
description: Upgrade Artlite's pinned mlir-aie and llvm-aie (Peano) toolchain wheels to the latest nightly builds, relock with uv, reinstall, and verify. Use when asked to update/bump/upgrade the toolchain, mlir-aie, llvm-aie, Peano, aiecc, or nightly wheels, or when a fix is known to exist only in a newer toolchain build.
---

# Update toolchain dependencies

One command does the whole upgrade (fetch latest wheels → rewrite pins in `pyproject.toml` → `uv lock` → `uv sync --extra examples` → verify installed versions and `import artlite._native`):

```bash
python .claude/skills/update-deps/update_deps.py
```

Run from the repo root. Useful flags:
- `--check` — only print pinned vs. latest versions; changes nothing. Use this first if the user just asked whether updates exist.
- `--no-sync` — update pins and `uv.lock` but don't install.

The script exits non-zero on any failure (no wheels matched, `uv lock`/`uv sync` failed, installed version mismatch, import failure). Report its output to the user; don't retry blindly.

## After a successful update

- No cache clearing needed: RTC cache keys include the `mlir_aie`/`llvm_aie` versions, so kernels recompile automatically. Old `rtc_*` dirs in `~/.cache/artlite/` just become orphaned; delete them to reclaim disk if desired.
- If `uv sync` dropped the editable C++ build, rebuild with `uv pip install --no-build-isolation -ve .`.
- Run `pytest tests/rtc -m smoke` (one NPU case per component and the apps' full designs: a bump can push a core past
  its program memory), not the full suite unless asked.
- Regenerate the issue slot / shared port tables of `variant_search` and the `ARTLITE_MEASURE_II` report from the new
  llvm-aie commit: `python -m artlite.gen_isa_tables` (reads the .td files of the wheel's commit from the `../llvm-aie`
  checkout: `git -C ../llvm-aie fetch` when it lacks the commit; uses its `build/bin/llvm-tblgen`).
- `variant_search --why` traces come from the debug build in `../llvm-aie/build`; its report warns while that build's
  commit is not the wheel's. Tell the user if it should be rebuilt.
- Commit; the diff usually should be `pyproject.toml` + `uv.lock` (+ `src/artlite/aie2p_isa.py`).

## Gotchas

- Wheels come from GitHub release pages listed in `[tool.uv] find-links` (`mlir-aie` `latest-wheels-4`, `llvm-aie` `nightly`). If the script finds no matches, the release tag or page layout changed — check the URLs in `SOURCES` in the script against `find-links`.
- llvm-aie's 4th version component is `YYYYMMDDNN`; it advances nightly even when the commit hash doesn't, so a "new" version may contain no compiler changes.
- A toolchain bump can change kernel schedules (II) or break compilation of RTC templates. If a test regresses, compare against the previous pin (`git diff pyproject.toml`) before debugging the operator.
