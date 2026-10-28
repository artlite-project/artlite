"""artlite-probe --diagnose (models.diagnose): its DMA rows and control transactions stay inside their buffers and the
probe design's free memory (the decoded transactions, no NPU); gpu_metrics v3's layout; on the NPU, the control path
in its ranges."""

import ctypes

import pytest

from artlite import txn
from artlite.parallel_compile import compile_all
from artlite.probe import ClockProbeOperator, DmaProbeOperator
from models.diagnose import rows, telemetry
from tests.test_probe import _channel_rows


def _extents(words: object) -> list[tuple[int, int]]:
    """(first byte, end) of each shim BD's transfer in a transaction: its strided rows (``_channel_rows``) or, without
    dimensions, its contiguous bytes."""
    out, strided = [], {}
    for name, f in txn.decode(words):
        reg = f.get("addr", 0) & 0xFFFFF
        if name == "blockwrite" and txn.SHIM_BD0 <= reg < txn.SHIM_BD0 + txn.SHIM_BDS * txn.BD_BYTES:
            d = f["data"]
            if (d[3] >> 20) & 0x3FF:
                strided[f["addr"] >> 25, (reg - txn.SHIM_BD0) // txn.BD_BYTES] = True
            else:
                addr = d[1] + (d[2] << 32)
                out.append((addr, addr + 4 * d[0]))
    if strided:
        out += [(int(starts.min()), int(starts.max()) + run) for starts, run, _ in _channel_rows(words).values()]
    return out


def test_rows_inside_buffers() -> None:
    src, dst = 1 << 32, 2 << 32
    names = [row[0] for row in rows.ROWS]
    assert len(set(names)) == len(names)
    for row in rows.ROWS:
        words, total = rows.row_transaction(row, src, dst)
        extents = _extents(words)
        assert len(extents) == len(row[1]) + len(row[2]), row[0]
        for lo, hi in extents:
            base = src if lo < dst else dst
            assert base <= lo and hi <= base + rows.BUFFER, row[0]
        assert total > 0


def test_control_inside_scratch() -> None:
    """The control path's transactions write only memory tile 7's free end and shim registers; their DMA reads stay
    in the first 4 MB of the source buffer."""
    src = 1 << 32
    for kind in rows.CONTROL:
        if kind == "noop":  # (no operation touches anything)
            continue
        words = rows._control_words(kind, 64, src)
        for name, f in txn.decode(words):
            if name in ("write32", "blockwrite", "maskwrite32"):
                col, row_, reg = f["addr"] >> 25, (f["addr"] >> 20) & 0x1F, f["addr"] & 0xFFFFF
                if row_ == 1:
                    n = 4 * len(f["data"]) if name == "blockwrite" else 4
                    assert col == 7 and rows.SCRATCH <= reg and reg + n <= 0x80000, (kind, hex(reg))
                else:
                    assert row_ == 0, (kind, name, row_)
        for lo, hi in _extents(words):
            assert src <= lo and hi <= src + 4 * rows.MB, kind


def test_gpu_metrics_layout() -> None:
    assert ctypes.sizeof(telemetry.GpuMetricsV3) == 264  # (the kernel's structure_size on Strix Halo, MP1 14.0.1)
    # (kgd_pp_interface.h's order and C alignment: the u64 counter after 102 bytes of u16 fields, at 104)
    assert telemetry.GpuMetricsV3.system_clock_counter.offset == 104
    assert telemetry.GpuMetricsV3.average_fclk_frequency.offset == 182


@pytest.mark.smoke
def test_control_rates() -> None:
    op, clock = DmaProbeOperator(), ClockProbeOperator()
    compile_all([op, clock])
    r = rows.control_rates(op, clock, seconds=0.2)
    assert 1600 < r["array clock (MHz)"] < 2000
    assert 10 < r["write32 (ns per op)"] < 1000 and 0.2 < r["blockwrite 1 KB (ns per byte)"] < 20
    assert 10 < r["NOP dispatch, blocking (us)"] < 1000 and 1000 < r["NOP dispatches, 3 queued (per s)"] < 500000


def _data(control_scale: float = 1.0, v1_ms: float | None = None, prochot: float = 0.0) -> dict:
    """A diagnosis' data from Strix Halo's reference numbers (the control path scaled; V1's prompt time set; the
    platform holding PROCHOT for ``prochot`` of the prompts, of the Q4_K kinds run alone and of the full-duty sweep)."""
    from models.diagnose import reference

    ref = reference.HALO
    ms = {k: 1024 / v * 1e3 for k, v in ref["pp1024"].items()}
    if v1_ms is not None:
        ms[next(k for k in ms if k.startswith("V1"))] = v1_ms
    runs = []
    for rnd in (1, 2):
        for name, m in ms.items():
            r = {"variant": name, "round": rnd, "pp1024_ms": [m] * 5, "lookups_ms_per_prompt": 2.0,
                 "anon_huge_mb": 1100.0, "holds": "0.81 GB of weights, 79 MB of KV cache"}
            if rnd == 1 and name.startswith(("V0", "V1")):
                r["kinds"] = {"T": 1024, "whole_ms": ref["kinds"]["whole program"], "items": 241,
                              "configurations": 137, "txn_ops": 150000, "control_bytes": 5_650_000,
                              "kinds": {k: {"items": 16, "configurations": 1, "ms": v}
                                        for k, v in ref["kinds"].items() if k != "whole program"}}
                r["design_switch_us"] = ref["design_switch_us"] * control_scale
            runs.append(r)
    control = {k: v * control_scale if "ns per" in k else v for k, v in ref["control"].items()}
    def step(share: float) -> dict:
        return {"fclk_mhz": {"median": 2000, "max": 2000}, "npu_clk_mhz": {"median": 1810, "max": 1810},
                "seconds": 1.5, "throttle": {"prochot": int(1500 * share), "thm_core": 0}}

    tel = {f"{r['variant']}, round {r['round']}": step(prochot) for r in runs}
    v0 = next(r for r in runs if r["variant"].startswith("V0"))
    for k in v0["kinds"]["kinds"]:
        tel[f"{v0['variant']}, round 1: kinds: {k}"] = step(prochot if "Q4_K" in k else 0.0)
    v0["duty"] = {"kind": "GemmQOperator Q4_K 2048x8192", "items": 16, "duties": {}}
    for d, share in (("100%", prochot), ("50%", 0.0)):
        v0["duty"]["duties"][d] = {"item_ms": 2.2, "items_run": 600}
        tel[f"{v0['variant']}, round 1: duty {d}"] = step(share)
    return {"code": {"local_commits": [], "changed_files": [], "stale_install": ""},
            "dma": ref["dma"], "control": control, "model": {"runs": runs, "resident_before": 1.0},
            "telemetry": tel, "dpm_mhz": {"fclk": [400, 2000]}, "identity": {"cpus": {}, "npu": {}},
            "root": {"kernel_log": [], "npu_users": []}, "npu_info": {"power_mode": 0}, "irq_cpu": 0}


def test_verdicts() -> None:
    """Strix Halo's own numbers point to no cause; a control path 1.6x slower to H11, its cost on the 1B chunk the
    control code's bytes times the extra ns per byte and the configurations times the extra switch; 4 KB pages faster
    than 2 MB to H2's possibility; PROCHOT for most of the prompts to H12, with the kinds that held it alone."""
    from models.diagnose import report, verdict

    v = {x["cause"][:3]: x for x in verdict.evaluate(_data())}
    assert all(not x["verdict"].startswith("yes") for x in v.values()), v
    slow = {x["cause"][:3]: x for x in verdict.evaluate(_data(control_scale=1.6))}
    assert slow["H11"]["verdict"] == "yes" and "1.60x" in slow["H11"]["evidence"], slow["H11"]
    pages = {x["cause"][:3]: x for x in verdict.evaluate(_data(v1_ms=100.0))}
    assert pages["H1 "]["verdict"].startswith("no: 2 MB pages are slower"), pages["H1 "]
    assert pages["H2 "]["verdict"] == "possible", pages["H2 "]
    hot = {x["cause"][:3]: x for x in verdict.evaluate(_data(prochot=0.84))}
    assert hot["H12"]["verdict"] == "yes" and "84% of the prompts" in hot["H12"]["evidence"], hot["H12"]
    assert "held during ['GemmQOperator Q4_K" in hot["H12"]["evidence"] and "50% 0%" in hot["H12"]["evidence"]
    assert v["H12"]["verdict"] == "no", v["H12"]
    d = _data(prochot=0.84)
    d.update(verdicts=verdict.evaluate(d), identity={**d["identity"], "dmi": dict.fromkeys(
        ("sys_vendor", "product_name", "product_version", "board_name", "bios_vendor", "bios_version", "bios_date"), ""),
        "cpus": {"model": "?"}, "ip_versions": {}, "kernel": "?", "thp": {"enabled": "?"}, "platform_profile": {},
        "power_supplies": {}}, code={**d["code"], "commit": "x", "describe": "x"})
    md = report.markdown(d, "x.json")
    assert "### Verdicts" in md and "| 100% | 2.2 | 600 | 2.9 | 84% |" in md, md
