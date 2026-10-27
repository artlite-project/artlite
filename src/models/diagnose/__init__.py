"""``artlite-probe --diagnose``: everything that can make Artlite slower on a machine, measured and recorded in one
run, so that a slower number comes with what explains it (issue #5). About 5 minutes; Linux: as root
(``sudo $(which artlite-probe) --diagnose``).

1. As root: what only root reads (``system.root_readings``: the memory devices, the NPU's PCI capabilities, the data
   fabric's DRAM map, debugfs, the kernel log, other processes on the NPU); then the process becomes the user who ran
   sudo and starts again as them (``ARTLITE_DIAGNOSE_ROOT``: the readings' file), so that caches and model files are
   theirs.
2. The machine and the checkout (``system.identity``, ``system.code_state``).
3. Telemetry through every step (``telemetry.Sampler``).
4. The NPU's DMA rows on host buffers of 2 MB pages and of 4 KB pages, its control path (``rows``).
5. Llama 3.2 1B's 1024-token prompt in four variants, two rounds (``model``).
6. ``xrt-smi validate`` where XRT is installed: the latency / throughput numbers other runtimes' users post.
7. Verdicts (``verdict``), the report (Markdown) and every number (a JSON file to attach)."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from . import model

ROOT_ENV = "ARTLITE_DIAGNOSE_ROOT"


def become_user(argv: list[str]) -> None:
    """As root: the root readings into a file of the sudo user's, then this command again as that user (exec; does
    not return). Without SUDO_UID (a root login) it returns and the run stays root."""
    from . import system

    if not os.environ.get("SUDO_UID"):
        return
    invoking = system.invoking_environment()  # (sudo reset the environment: the user's own, before root drops)
    readings = system.root_readings()
    fd, path = tempfile.mkstemp(prefix="artlite-diagnose-root-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(readings, f, default=str)
    os.chown(path, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    user = system.drop_privileges()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUDO_")}
    # (the user's PATH (their aiecc first, as without sudo) and settings: ARTLITE_MODELS_DIR, ARTLITE_CACHE_DIR, ...)
    env.update({k: v for k, v in invoking.items() if k == "PATH" or k.startswith("ARTLITE_")})
    env[ROOT_ENV] = path
    sys.stderr.write(f"artlite-probe --diagnose: root readings taken; continuing as {user}\n")
    os.execve(sys.executable, [sys.executable, "-m", "models.probe", *argv], env)


def _xrt_validate() -> dict[str, object] | None:
    """``xrt-smi validate``'s latency, throughput and gemm lines (when installed; Windows: the driver's copy)."""
    from ..common import xrt_smi

    if platform.system() == "Windows":
        text = xrt_smi("validate")
        if text is None:
            return None
        out: dict[str, object] = {}
    else:
        smi = shutil.which("xrt-smi")
        if smi is None:
            return None
        try:
            r = subprocess.run([smi, "validate"], capture_output=True, text=True, timeout=300, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"error": str(e)}
        text = r.stdout + r.stderr
        out = {"exit": r.returncode}
    for key, pat in (("gemm_tops", r"TOPS:\s*([\d.]+)"), ("latency_us", r"Average latency:\s*([\d.]+)"),
                     ("throughput_ops", r"Average throughput:\s*([\d.]+)"), ("power_mode", r"Power Mode\s*:\s*(\S+)")):
        if (m := re.search(pat, text)):
            out[key] = m[1]
    return out


def run(*, model_file: str, seconds: float, out_dir: str) -> str:
    """The whole diagnosis; prints the report and returns the JSON file's path."""
    import artlite
    from artlite.overlay import irq_cpu
    from artlite.parallel_compile import compile_all
    from artlite.probe import ClockProbeOperator, DmaProbeOperator

    from . import report, rows, system, telemetry, verdict

    data: dict[str, object] = {"version": 1, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "platform": platform.system()}
    root_file = os.environ.pop(ROOT_ENV, "")
    if root_file:
        with open(root_file) as f:
            data["root"] = json.load(f)
        os.unlink(root_file)
    elif platform.system() == "Linux":
        data["root"] = system.root_readings()  # (a root login: still root)
    data["identity"] = system.identity()
    data["code"] = system.code_state()
    sensors = artlite.Device()  # (the sampler's: a DRM client of its own)
    data["npu_info"] = system.npu_info(sensors)
    cpu = irq_cpu()
    data["irq_cpu"] = cpu
    data["dpm_mhz"] = {k: telemetry.dpm_levels(k) for k in ("fclk", "mclk", "socclk")}
    sampler = telemetry.Sampler(cpu, extra=lambda: system.npu_sample(sensors))
    sampler.start()
    try:
        say = lambda what: sys.stderr.write(f"artlite-probe --diagnose: {what}\n")
        clock, loaded, dma = ClockProbeOperator(), ClockProbeOperator(load=True), DmaProbeOperator()
        compile_all([clock, loaded, dma])
        say("warm-up: 4 s of all 32 cores (the NPU's clock ramps after a DPM change)")
        sampler.step("warm-up")
        rows.array_clock(loaded, seconds=4.0)
        sampler.end("warm-up")
        for pages in ("2 MB", "4 KB"):
            say(f"DMA rows on {pages} pages")
            sampler.step(f"DMA rows, {pages} pages")
            data.setdefault("dma", {})[pages] = rows.dma_rates(dma, seconds=seconds, pages=pages)
            sampler.end(f"DMA rows, {pages} pages")
        say("control path")
        sampler.step("control path")
        data["control"] = rows.control_rates(dma, clock, seconds=seconds)
        mhz, _ = rows.array_clock(loaded, seconds=2 * seconds)
        data["control"]["array clock, 32 cores of MMULs (MHz)"] = mhz
        sampler.end("control path")
        sampler.step("array clock with a busy CPU")
        with model.busy_cpu(True, cpu):
            mhz, _ = rows.array_clock(loaded, seconds=2 * seconds)
        data["control"]["array clock, 32 cores of MMULs, a busy CPU (MHz)"] = mhz
        sampler.end("array clock with a busy CPU")
        say(f"Llama 3.2 1B: {model.VARIANTS[0][0]} .. {model.VARIANTS[-1][0]}, two rounds (the first run of a "
            "variant compiles its designs)")
        data["model"] = model.run(model_file, step=sampler.step, end=sampler.end, irq_cpu=cpu)
        say("xrt-smi validate")
        sampler.step("xrt-smi validate")
        data["xrt_validate"] = _xrt_validate()
        sampler.end("xrt-smi validate")
    finally:
        sampler.stop()
    data["telemetry"] = sampler.summary()
    data["verdicts"] = verdict.evaluate(data)
    data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    name = f"artlite-diagnose-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)
    print(report.markdown(data, path))
    return path
