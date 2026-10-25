import hashlib
import json
import os
import sys
import threading

import pytest

from artlite import parallel_compile


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true",
                     help="also run the tests marked slow (whole-model llama runs against CPU references)")
    parser.addoption("--refresh-refs", action="store_true",
                     help="recompute the cached CPU references of the slow tests (tests/rtc/refs/)")
    parser.addoption("--dump-designs", metavar="DIR", default=None,
                     help="instead of compiling and running, render every operator's kernel source and MLIR design "
                          "into DIR (index.jsonl: test -> designs; the tests are skipped there). Diff two dumps to "
                          "check that a refactoring leaves the designs unchanged.")


# The smoke set (``pytest tests/rtc -m smoke``, about a minute with the designs cached): the host-side tests, the
# apps' full designs build, and one or two NPU cases per component (each model's decode slice, attention, prefill
# GEMM / attention / norm, the shared GEMV, fused ELFs, the activation table, kmath's variants). Entries: test files or
# test ids.
SMOKE = (
    "test_app_designs.py", "test_overlay_kv.py", "test_overlay_runtime.py", "test_gemma_host.py", "test_llama_image.py", "test_qwen35_image.py",
    "test_gemm_q_dma.py", "test_render_cache.py", "test_server.py", "test_sampling.py", "test_minicpmv_host.py",
    "test_minicpmv_npu.py", "test_probe.py", "test_kmath.py",
    "test_router.py::test_catalog", "test_router.py::test_preset", "test_router.py::test_router",
    "test_fused.py::test_library_program", "test_fused.py::test_library_program_memory_variants",
    "test_fused.py::TestFusedOperator::test_silu_add",
    "test_overlay_gemv.py::test_gemv[Q4_0-9216-4]",
    "test_prefill_gemm.py::test_prefill_gemm[llama-1B-Q4_K_M-L3-gate_up-T256]",
    "test_prefill_gemm.py::test_prefill_gemm[llama-3B-Q3_K_M-L0-qkv-T64]",
    "test_prefill_norm.py::test_prefill_norm[T64-add-E3072-dp396-c2]",
    "test_llama_decode.py::test_llama_decode_steps[1B_Q4_K_M-single]",
    "test_llama_decode.py::test_llama_decode_steps[1B_Q5_K_M-single]",
    "test_llama_decode.py::test_llama_decode_steps[3B_Q4_K_M-hilo]",
    "test_llama_decode.py::test_llama_decode_steps[8B_Q4_K_S-single]",
    "test_llama_decode.py::test_llama_decode_deep[1B_Q4_K_M-single]",
    "test_llama_prefill.py::test_attn[hd128-qh3-kp1-T256-p0]", "test_llama_prefill.py::test_attn[hd64-qh4-kp2-T256-p128]",
    "test_gemma_npu.py::test_gemma_overlay_layers[swa+full-batch3-hilo]",
    "test_gemma_npu.py::test_gemma_overlay_layers[shared-kv-one-token-single]",
    "test_prefill_gemm.py::test_prefill_gemm[gemma4-E4B-Q4_0-L0-gate_up-T256]",
    "test_prefill_norm.py::test_residual_norm[post-E2560-d1-p320]",
    "test_prefill_norm.py::test_residual_norm[plain-E2560-d2-p384]",
    "test_gemma_prefill.py::test_attn_prefill[hd512-p96-T64]", "test_gemma_prefill.py::test_ple",
    "test_qwen35_decode.py::test_overlay_slice[s3-9B]", "test_qwen35_decode.py::test_overlay_slice[split-1B]",
    "test_qwen35_decode.py::test_attention[27B-150-2-True-True]", "test_qwen35_decode.py::test_attention[1B-150-2-True-True]",
    "test_qwen35_decode.py::test_phase[9B-gdn_in-4]", "test_qwen35_decode.py::test_moe_layer[3-35B-A3B-0]", "test_moe_route.py",
    "test_moe_plan.py", "test_moe_prefill.py::test_moe_block[35B-A3B-5]",
    "test_qwen35_decode.py::test_overlay_slice[lazy-35B-A3B]",
    "test_qwen35_prefill_elf.py::test_prefill_slice[9B]", "test_qwen35_prefill_elf.py::test_prefill_slice[27B]",
    "test_qwen35_prefill_elf.py::test_prefill_slice[35B-A3B]", "test_qwen35_prefill_elf.py::test_prefill_slice[occamy]",
    "test_qwen35_attn_prefill.py::test_attn_prefill[9B-64]", "test_qwen35_attn_prefill.py::test_attn_prefill[1B-192]",
    "test_qwen35_gdn_prefill.py::test_gdn_scan[64-9B]", "test_prefill_gemm.py::test_prefill_gemm[qwen35-9B-Q4_K_M-L3-attn_in-T256]",
    "test_sd15_unet.py::test_gemm[16-2-640-640-True-bf16]", "test_sd15_unet.py::test_conv[16-1-640-640-True]",
    "test_sd15_unet.py::test_norm[gnsilu-64-640-2-4]", "test_sd15_unet.py::test_attention[16-80-2-2-False-8-False]",
    "test_sd15_vae.py::test_conv[512-128-128]",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked slow run on demand: with --run-slow, or when -m selects them (or the smoke set). A smoke entry
    that names no test of a collected file (a renamed parameter) is reported (``pytest_terminal_summary``)."""
    hit: set[str] = set()
    for item in items:
        name = item.nodeid.removeprefix("tests/rtc/")
        for s in SMOKE:
            if name == s or name.startswith(s + "::"):
                item.add_marker(pytest.mark.smoke)
                hit.add(s)
    files = {item.nodeid.removeprefix("tests/rtc/").split("::")[0] for item in items}
    _stale_smoke[:] = [s for s in SMOKE if s not in hit and s.split("::")[0] in files]
    selected = config.getoption("-m") or ""
    if config.getoption("--run-slow") or "slow" in selected or "smoke" in selected:
        return
    skip = pytest.mark.skip(reason="slow: run with --run-slow (or -m slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


_stale_smoke: list[str] = []


# Parallel compilation ---------------------------------------------------------------------------
# A planning subprocess runs the selected tests (nothing built, no NPU) and streams their designs to
# a background build queue (artlite.parallel_compile, ARTLITE_COMPILE_JOBS at a time) while the tests
# run here: their compile() finds the cache, or waits for the build of the same design in progress.

_precompiler: list[parallel_compile.Precompiler] = []


def pytest_collection_finish(session: pytest.Session) -> None:
    config = session.config
    if (parallel_compile.planning_active() or config.getoption("--dump-designs") or config.option.collectonly
            or parallel_compile.jobs() == 1 or os.environ.get("ARTLITE_IGNORE_CACHE", "0") == "1"):
        return
    from _pytest.skipping import evaluate_skip_marks

    ids = [item.nodeid for item in session.items if not evaluate_skip_marks(item)]
    if ids:
        _precompiler.append(parallel_compile.Precompiler(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--run-slow", *ids],
            cwd=str(config.rootpath)))


def pytest_sessionfinish(session: pytest.Session) -> None:
    for pre in _precompiler:
        pre.close()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _stale_smoke and "smoke" in (terminalreporter.config.getoption("-m") or ""):
        terminalreporter.write_line(f"SMOKE entries naming no collected test: {', '.join(_stale_smoke)}", red=True)
    for pre in _precompiler:
        terminalreporter.write_line(
            f"artlite.parallel_compile: {pre.queue.submitted} design(s) of the selected tests compiled in the "
            f"background ({parallel_compile.JOBS_ENV}={parallel_compile.jobs()}; {pre.queue.failed} failed)")


# --dump-designs ---------------------------------------------------------------------------------

_dump_state: dict[str, object] = {"test": "", "lock": threading.Lock()}


def _dump(out: str, op: object) -> None:
    """An operator's kernel source and MLIR, or a fused ELF's (its translation unit and fused design)."""
    from artlite.operators.fused.elf import FusedElf

    if isinstance(op, FusedElf):
        spec = op.build_spec()
        src, mlir = spec.source or "", spec.mlir
    else:
        ks = op.kernel_source()
        src, mlir = ks.text if ks is not None else "", op.mlir_text()
    name = f"{type(op).__name__}_{hashlib.sha256((src + mlir).encode()).hexdigest()[:16]}"
    with _dump_state["lock"]:
        for ext, text in ((".cc", src), (".mlir", mlir)):
            path = os.path.join(out, name + ext)
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(text)
        with open(os.path.join(out, "index.jsonl"), "a") as f:
            f.write(json.dumps({"test": _dump_state["test"], "op": type(op).__name__, "design": name}) + "\n")


def pytest_report_header(config: pytest.Config) -> str | None:
    from models.common import stale_toolchain

    return f"WARNING: {stale}" if (stale := stale_toolchain()) else None


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--refresh-refs"):
        from . import common

        common.REFRESH_REFS = True
    out = config.getoption("--dump-designs")
    if not out:
        return
    from artlite.operators.base import ArtcOperator
    from artlite.operators.fused.elf import FusedElf

    os.makedirs(out, exist_ok=True)

    def compile_all(operators: list[ArtcOperator | FusedElf], **kwargs: object) -> None:
        for op in operators:
            for one in op.operators if isinstance(op, FusedElf) else [op]:
                _dump(out, one)
            if isinstance(op, FusedElf):
                _dump(out, op)
        pytest.skip("--dump-designs")

    def compile_op(self: ArtcOperator, *args: object, **kwargs: object) -> None:
        compile_all([self])

    def compile_fused(self: FusedElf) -> None:
        compile_all([self])

    ArtcOperator.compile = compile_op
    FusedElf.compile = compile_fused
    parallel_compile.compile_all = compile_all


def pytest_runtest_setup(item: pytest.Item) -> None:
    _dump_state["test"] = item.nodeid
