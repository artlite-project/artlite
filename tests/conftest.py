import hashlib
import json
import os
import sys
import threading
from typing import TYPE_CHECKING

import pytest

# Idle OpenMP threads sleep instead of spinning (libgomp reads this as torch loads it: the tests' references; a value
# already set stays): spinning, torch's pool keeps ~1.2 CPUs busy while the NPU works (Llama 1B prefill: 1.16 vs 0.11
# CPUs)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
from artlite import parallel_compile, settings

# every host-built transaction is verified where it is built or loaded (``txn.verify``)
os.environ.setdefault(settings.VERIFY_TXN.name, "1")

# A module or package of tests/ named like a package of src/ would shadow it for a script run from tests/ (``python
# tests/txn_dump.py``: the script's directory comes first on sys.path) and wherever the editable install's finder is not
# first (worktrees, wheel installs).
_HERE = os.path.dirname(os.path.abspath(__file__))
if shadows := [p for p in ("artlite", "models") if os.path.exists(os.path.join(_HERE, p + ".py"))
               or os.path.isdir(os.path.join(_HERE, p))]:
    raise RuntimeError(f"tests/ has a module named like a src package: {shadows}")

if TYPE_CHECKING:
    from artlite.vision.device import VisionDevice


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true",
                     help="also run the tests marked slow (whole-model llama runs against CPU references)")
    parser.addoption("--refresh-refs", action="store_true",
                     help="recompute the cached CPU references of the slow tests (tests/refs/)")
    parser.addoption("--dump-designs", metavar="DIR", default=None,
                     help="instead of compiling and running, render every operator's kernel source and MLIR design "
                          "into DIR (index.jsonl: test -> designs; the tests are skipped there). Diff two dumps to "
                          "check that a refactoring leaves the designs unchanged.")


@pytest.fixture(scope="module")
def npu() -> "VisionDevice":
    """The vision overlay's device, once per module (the image networks' tests; a module with its own ``npu`` fixture
    overrides it)."""
    from artlite.vision.device import VisionDevice

    return VisionDevice()


# The smoke set (``pytest tests -m smoke``, about a minute with the designs cached): host-side tests, one full app
# design per architecture (the largest member: it meets the program-memory, stack and BD limits first), and one NPU
# case per component: each model's decode slice, the attention cores, the prefill GEMMs (GemmQ, GemmQd), norms,
# attention and MoE block, a chunk program with its decode continuation, the shared GEMV, fused ELFs and libraries,
# kmath's variants the kernels instantiate, kernel_lib's gathers, the vision overlay's ops and weight packer, an image
# prompt end to end. Entries: test files or test ids.
SMOKE = (
    # host
    "decode/test_kv.py", "test_overlay.py", "test_gguf_model.py", "operators/test_render_cache.py", "test_sampling.py",
    "prefill/test_pace.py", "minicpmv/test_host.py", "moe/test_route.py", "moe/test_plan.py", "spark/test_host.py",
    "moss_tr/test_host.py",
    "vision/test_weights.py", "test_probe.py::test_page_cases", "qwen3vl/test_ref.py::test_sizes",
    "qwen3vl/test_encoder.py::test_layout[64-64]",
    "gemm/test_dma.py::test_gemm_q_dataflow", "gemm/test_dma.py::test_gemm_qd_planar_dataflow",
    "gemm/test_dma.py::test_gemm_half_dataflow", "gemm/test_dma.py::test_check_shim_tap",
    "gemm/test_dma.py::test_walk_offsets_outer_loop", "gemm/test_dma.py::test_gemm_half_plans[minicpmv-1B]",
    "llama/test_image.py::test_config[1B]", "llama/test_image.py::test_layout[1B]", "llama/test_image.py::test_pack_image",
    "llama/test_image.py::test_qwen3[0.6B-10240-151936-eos0]", "llama/test_image.py::test_pack_tails[embedding-4B-True]",
    "qwen35/test_image.py::test_layout[9B]", "qwen35/test_image.py::test_pack_image[9B]",
    "qwen35/test_image.py::test_pack_moe", "qwen35/test_image.py::test_gate_up_windows",
    "qwen35/test_image.py::test_pack_tails[kev-pairs]", "qwen35/test_image.py::test_pack_tails[d3-lite-pairs]",
    "server/test_server.py::test_stream", "server/test_server.py::test_tool_calls",
    "server/test_server.py::test_prompt_cache", "server/test_server.py::test_errors",
    "server/test_server.py::test_stop_strings", "server/test_router.py::test_catalog", "server/test_router.py::test_preset",
    # the apps' full designs (no NPU)
    "test_app_designs.py::test_app_designs[llama-8B-Q4_K_S]", "test_app_designs.py::test_app_designs[llama-1B-Q4_K_M-hilo]",
    "test_app_designs.py::test_app_designs[qwen3-0.6B-Q4_1]", "test_app_designs.py::test_app_designs[gemma-E4B-Q4_0]",
    "test_app_designs.py::test_app_designs[qwen-27B-Q4_K_M]", "test_app_designs.py::test_app_designs[qwen-1B-Q4_K_M]",
    "test_app_designs.py::test_app_designs[qwen-35B-A3B-Q4_K_M-mtp]", "test_app_designs.py::test_app_designs[lfm3-QAD-Q4_0]",
    "test_app_designs.py::test_app_designs[flint-int4]", "test_app_designs.py::test_app_designs[sd15]",
    # NPU
    "test_probe.py::test_probe",
    "operators/test_kmath.py::test_kmath_sources", "operators/test_kmath.py::test_kmath_shared_dependency",
    "operators/test_kmath.py::test_kmath[exp2]", "operators/test_kmath.py::test_kmath[exp2-zero]",
    "operators/test_kmath.py::test_kmath[exp2-fp32]", "operators/test_kmath.py::test_kmath[exp2-b192-bounded]",
    "operators/test_kmath.py::test_kmath[exp2-chord2]", "operators/test_kmath.py::test_kmath[recip-normal]",
    "operators/test_kmath.py::test_kmath[sigmoid-bf16]", "operators/test_kmath.py::test_kmath[exp-bf16-neg]",
    "operators/test_kmath.py::test_kmath[act-silu]", "operators/test_kmath.py::test_kmath[silu_tanh]",
    "operators/test_kmath.py::test_kmath_rows[rms-256-w]", "operators/test_kmath.py::test_kmath_rows[rope-64]",
    "operators/test_kmath.py::test_kmath_softmax[inplace-decode]",
    "operators/test_kernel_lib.py",
    "operators/test_fused.py::test_library_program", "operators/test_fused.py::test_library_without_reset",
    "operators/test_fused.py::TestFusedOperator::test_silu_add",
    "operators/test_fused.py::TestFusedLifecycle::test_sync_lines",
    "decode/test_gemv.py::test_gemv[Q4_0-9216]",
    "prefill/test_gemm.py::test_prefill_gemm[llama-1B-Q4_K_M-L3-gate_up-T256]",
    "prefill/test_gemm.py::test_prefill_gemm[gemma4-E4B-Q4_0-L0-gate_up-T256]",
    "prefill/test_norm.py::test_prefill_norm[T64-add-E3072-dp396-c2]",
    "prefill/test_norm.py::test_residual_norm[post-E2560-d1-p320]",
    "prefill/test_attn256.py::test_attn_prefill[1B-192-2]",
    "llama/test_decode.py::test_llama_decode_steps[1B_Q4_K_M-single]",
    "llama/test_decode.py::test_llama_decode_steps[3B_Q4_K_M-hilo]",
    "llama/test_prefill.py::test_attn[hd128-qh3-kp1-T256-p0]",
    "llama/test_prefill.py::test_attn[hd64-qh4-kp2-T256-p128]",
    "gemma4/test_attn.py", "gemma4/test_npu.py::test_gemma_overlay_layers[swa+full-hilo]",
    "gemma4/test_prefill.py::test_attn_prefill[hd512-p96-T64]", "gemma4/test_prefill.py::test_ple",
    "qwen35/test_decode.py::test_overlay_slice[s3-9B]", "qwen35/test_decode.py::test_overlay_slice[split-1B]",
    "qwen35/test_decode.py::test_overlay_slice[lazy-35B-A3B]", "qwen35/test_decode.py::test_moe_layer[35B-A3B-0]",
    "qwen35/test_prefill.py::test_prefill_slice[9B]", "qwen35/test_gdn_prefill.py::test_gdn_scan[64-9B]",
    "moe/test_prefill.py::test_moe_block[35B-A3B-5]",
    "spark/test_overlay.py::test_spark_overlay_steps", "flint/test_decode.py::test_flint_overlay_steps",
    "vision/test_ops.py::test_gemm[16-2-640-640-True-bf16]", "vision/test_ops.py::test_conv[16-1-640-640-True]",
    "vision/test_ops.py::test_norm[gnsilu-64-640-2-4]", "vision/test_ops.py::test_attention[16-80-2-2-False-8-False]",
    "vision/test_ops.py::test_gemm_rope", "vision/test_ops.py::test_attention_dense[5-0.83-16-0]",
    "qwen3vl/test_encoder.py::test_layer[nb1-l10]", "minicpmv/test_npu.py::test_app_image_prompt",
    "sd15/test_unet.py::test_resnet[8-1280-1280]",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked slow run on demand: with --run-slow, or when the -m expression selects them by their markers
    (``-m slow``, ``-m smoke``; not ``-m "not smoke"``, which selects unmarked tests too). A smoke entry that names no
    test of a collected file (a renamed parameter) is reported (``pytest_terminal_summary``)."""
    hit: set[str] = set()
    for item in items:
        name = item.nodeid.removeprefix("tests/")
        for s in SMOKE:
            if name == s or name.startswith((s + "::", s + "[")):
                item.add_marker(pytest.mark.smoke)
                hit.add(s)
    files = {item.nodeid.removeprefix("tests/").split("::")[0] for item in items}
    _stale_smoke[:] = [s for s in SMOKE if s not in hit and s.split("::")[0] in files]
    if config.getoption("--run-slow"):
        return
    chosen = None
    if expr := config.getoption("-m"):
        from _pytest.mark.expression import Expression

        compiled = Expression.compile(expr)
        if not compiled.evaluate(lambda name, /, **kwargs: False):  # (an unmarked test is not selected)
            chosen = compiled
    skip = pytest.mark.skip(reason="slow: run with --run-slow (or -m slow)")
    for item in items:
        if "slow" in item.keywords:
            names = {m.name for m in item.iter_markers()}
            if chosen is None or not chosen.evaluate(lambda name, /, **kwargs: name in names):  # noqa: B023
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
            or parallel_compile.jobs() == 1 or settings.IGNORE_CACHE.get()):
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
            f"background ({settings.COMPILE_JOBS.name}={parallel_compile.jobs()}; {pre.queue.failed} failed)")


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
    from models.hostenv import stale_toolchain

    return f"WARNING: {stale}" if (stale := stale_toolchain()) else None


def pytest_configure(config: pytest.Config) -> None:
    from artlite.backend import stale_native

    if stale := stale_native():  # (the tests would test the previous native code: packers, dequantization, the shim)
        raise pytest.UsageError(stale)
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
