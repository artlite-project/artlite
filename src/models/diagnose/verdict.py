"""Which candidate cause the measurements point to (H0-H11 of the diagnosis), each from several measurements and with
the numbers that decide it, and how many ms of the 1B chunk's difference to Strix Halo each measured cause explains.

A variant counts as faster or slower only beyond its noise: the largest of the samples' spread, the two rounds'
difference of the same variant and 3% (between loads on Strix Halo). Every threshold is stated in the evidence it
produces."""

from __future__ import annotations

import re
import statistics

from . import reference

V0, V1, V2, V3, V4 = "V0", "V1", "V2", "V3", "V4"
PHASES = (": load", ": kinds")  # (telemetry tags of a run's other phases than its prompts)


def _pp_ms(data: dict[str, object]) -> dict[str, list[float]]:
    """Each variant's prompt times (ms) over both rounds."""
    out: dict[str, list[float]] = {}
    for r in data["model"]["runs"]:
        out.setdefault(r["variant"][:2], []).extend(r["pp1024_ms"])
    return out


NOISE_FLOOR = 0.03  # (Strix Halo: one variant's medians 1.7-3.6% apart between its loads)


def _noise(data: dict[str, object]) -> float:
    """The relative noise of a variant's median: the larger of the samples' spread (stdev / median) and the rounds'
    difference of one variant, at least NOISE_FLOOR."""
    worst = NOISE_FLOOR
    by: dict[str, list[float]] = {}
    for r in data["model"]["runs"]:
        s = r["pp1024_ms"]
        if len(s) > 1:
            worst = max(worst, statistics.stdev(s) / statistics.median(s))
        by.setdefault(r["variant"][:2], []).append(statistics.median(s))
    for meds in by.values():
        if len(meds) > 1:
            worst = max(worst, (max(meds) - min(meds)) / min(meds))
    return worst


def _cmp(a: float, b: float, noise: float) -> str:
    """'faster' / 'slower' / 'equal' of time a against time b, beyond the noise."""
    if a < b * (1 - noise):
        return "faster"
    if a > b * (1 + noise):
        return "slower"
    return "equal"


def _v(cause: str, verdict: str, evidence: str) -> dict[str, str]:
    return {"cause": cause, "verdict": verdict, "evidence": evidence}


def evaluate(data: dict[str, object]) -> list[dict[str, str]]:
    ref = reference.HALO
    out = []
    code = data["code"]
    local, changed = code.get("local_commits") or [], code.get("changed_files") or []
    out.append(_v("H0 the code is not upstream's",
                  "yes" if local or changed or code.get("stale_install") else "no",
                  f"{len(local)} local commits {[c[:72] for c in local[:3]]}, {len(changed)} changed files, "
                  f"{code.get('stale_install') or 'native build current'}"))

    pp = {k: statistics.median(v) for k, v in _pp_ms(data).items()}
    noise = _noise(data)
    dma2, dma4 = data["dma"]["2 MB"], data["dma"]["4 KB"]
    skew, w128, w132 = ("gather 256 B rows 64 KB + 256 B apart", "write 128 B rows 64 KB + 128 B apart, 16 channels",
                        "write 132 B rows 64 KB + 132 B apart")
    if V0 in pp and V1 in pp:
        c = _cmp(pp[V0], pp[V1], noise)
        out.append(_v("H1 4 KB pages cost the NPU translations (2 MB pages help)",
                      {"faster": "yes", "slower": "no: 2 MB pages are slower here (see H2)", "equal": "no effect"}[c],
                      f"pp1024 V0 (2 MB) {pp[V0]:.1f} ms vs V1 (4 KB) {pp[V1]:.1f} ms, noise {noise:.1%}; rows on "
                      f"4 KB / 2 MB pages: skewed 256 B gathers {dma4[skew]:.1f} / {dma2[skew]:.1f} GB/s, whole-line "
                      f"128 B writes {dma4[w128]:.1f} / {dma2[w128]:.1f}"))
    stag = "DDR read, 16 channels, regions at other 2 MB-page offsets"
    aligned = "DDR read, 8 shims: 16 channels"
    stag_w, aligned_w = "DDR write, 16 channels, regions at other 2 MB-page offsets", "DDR write, 16 channels"
    gain = max(dma2[stag] / dma2[aligned], dma2[stag_w] / dma2[aligned_w]) - 1
    slower_2mb = V0 in pp and V1 in pp and _cmp(pp[V0], pp[V1], noise) == "slower"
    out.append(_v("H2 2 MB-aligned buffers collide in DRAM channels / banks",
                  "yes" if gain > 0.05 and slower_2mb else "possible" if gain > 0.05 or slower_2mb else "no",
                  f"16-channel reads / writes at other 2 MB-page offsets {gain:+.1%} vs aligned (threshold +5%); "
                  f"pp1024 2 MB pages {'slower' if slower_2mb else 'not slower'} than 4 KB"))
    if V0 in pp and V2 in pp:
        c = _cmp(pp[V0], pp[V2], noise)
        out.append(_v("H3 / H4 whole-line windows (partial 64-byte line writes cost, the windows' own cost)",
                      {"faster": "whole lines help", "slower": "whole lines cost more than they save",
                       "equal": "no effect"}[c],
                      f"pp1024 V0 (whole lines) {pp[V0]:.1f} ms vs V2 (partial lines) {pp[V2]:.1f} ms; rows 132 B / "
                      f"128 B writes {dma2[w132]:.1f} / {dma2[w128]:.1f} GB/s ({dma2[w128] / dma2[w132]:.1f}x; Strix "
                      f"Halo 1.4x)"))

    tel = data["telemetry"]
    steps = [t for t in tel if t.startswith("V0") and not t.endswith(PHASES)]  # (V0's prompts, not its loads)
    top_f = max(data["dpm_mhz"].get("fclk") or [0]) or None
    fclk = [tel[t]["fclk_mhz"]["median"] for t in steps if isinstance(tel[t].get("fclk_mhz"), dict)]
    busy = _cmp(pp[V4], pp[V0], noise) if V4 in pp and V0 in pp else "?"
    if fclk and top_f:
        low = statistics.median(fclk) < 0.9 * top_f
        out.append(_v("H5 the fabric clock stays below its top while the NPU works",
                      "yes" if low else "no",
                      f"FCLK during V0's prompts median {statistics.median(fclk):.0f} MHz of {top_f} (threshold 90%); "
                      f"with a busy CPU (V4) the prompt is {busy}"))
    thr: dict[str, int] = {}
    for t in [t for t in tel if t.startswith(("V0", "V4")) and not t.endswith(PHASES)]:
        for k, v in (tel[t].get("throttle") or {}).items():
            thr[k] = thr.get(k, 0) + v
    ctl = data["control"]
    clk, clk_busy = ctl.get("array clock, 32 cores of MMULs (MHz)"), ctl.get("array clock, 32 cores of MMULs, a busy "
                                                                             "CPU (MHz)")
    drift = []
    for r in data["model"]["runs"]:
        drift.append((r["variant"][:2], r["round"], statistics.median(r["pp1024_ms"])))
    r1 = {v: m for v, rnd, m in drift if rnd == 1}
    r2 = {v: m for v, rnd, m in drift if rnd == 2}
    later = statistics.median([r2[v] / r1[v] for v in r1 if v in r2]) if r1 and r2 else 1.0
    npu_clk = [tel[t]["npu_clk_mhz"]["median"] for t in steps if isinstance(tel[t].get("npu_clk_mhz"), dict)]
    clk_prompts = statistics.median(npu_clk) if npu_clk else None
    effects = [what for what, hit in (
        ("array clock under load below 1780 MHz", clk and clk < 1780),
        ("a busy CPU lowers the array clock by 20 MHz or more", clk and clk_busy and clk_busy < clk - 20),
        ("the prompt slower with a busy CPU", busy == "slower"),
        ("the SMU's NPU clock during the prompts below 1700 MHz", clk_prompts is not None and clk_prompts < 1700),
        ("the second round slower", later > 1 + noise)) if hit]
    out.append(_v("H6 power / thermal limits lower the NPU or the fabric under load",
                  "yes: " + "; ".join(effects) if effects else "no",
                  f"array clock under load {clk:.0f} MHz, with a busy CPU {clk_busy:.0f}; the SMU's NPU clock during "
                  f"the prompts {clk_prompts}; with a busy CPU (V4) the prompt is {busy}; round 2 / round 1 "
                  f"{later:.3f}; throttle residency increments during the prompts {thr or 'none'} (context only: on "
                  f"Strix Halo they tick with no effect)"))
    runs = data["model"]["runs"]
    look = statistics.median(r["lookups_ms_per_prompt"] for r in runs)
    other = max((tel[t].get("other_cpu_load") or 0) for t in tel)
    irq = data["identity"]["cpus"]
    out.append(_v("H7 / H8 host work or the page cache on the prompt's path",
                  "yes" if look > 0.05 * pp.get(V0, 1e9) or other > 0.25 else "no",
                  f"embedding lookups {look:.1f} ms per prompt of {pp.get(V0, 0):.0f} ms; file in the page cache "
                  f"{data['model'].get('resident_before')} before; other processes' CPU load up to {other:.0%}; "
                  f"CPUs {irq.get('groups (max kHz, governor, EPP: CPUs)')}, the NPU's interrupts on CPU "
                  f"{data.get('irq_cpu')}"))
    huge = {r["variant"][:2]: r["anon_huge_mb"] for r in runs if r["round"] == 1}
    vm = {k: sum((tel[t].get("vmstat") or {}).get(k, 0) for t in tel) for k in ("thp_fault_fallback", "pswpin",
                                                                                   "pswpout", "compact_stall")}
    held = next((r.get("holds", "") for r in runs if r["variant"].startswith(V0)), "")
    m = re.search(r"([\d.]+) GB of weights", held)
    weights_mb = float(m[1]) * 1000 if m else 0.0
    no_thp = huge.get(V0, 0) < 0.8 * weights_mb  # (the weight image alone is a user BO of 2 MB pages)
    out.append(_v("H9 memory pressure: no 2 MB pages, swapping, compaction",
                  "yes" if no_thp or vm["pswpin"] or vm["pswpout"] else "no",
                  f"huge pages after loading V0 {huge.get(V0, 0):.0f} MB (its weights {weights_mb:.0f} MB, threshold "
                  f"80%), V1 {huge.get(V1, 0):.0f} MB; {vm}"))
    root = data.get("root", {})
    errors = [line for line in root.get("kernel_log", []) if any(w in line for w in ("IO_PAGE_FAULT", "TDR", "rror"))]
    users = root.get("npu_users") or []
    out.append(_v("H10 the NPU's state (other users, errors, power mode)",
                  "yes" if users or errors else "no",
                  f"other processes on the NPU {users}; kernel errors {errors[-3:]}; power mode "
                  f"{data.get('npu_info', {}).get('power_mode')}; runtime PM "
                  f"{data['identity']['npu'].get('runtime_pm')}"))
    rc = ref.get("control", {})
    slow = {k: ctl[k] / rc[k] for k in ("write32 (ns per op)", "blockwrite 1 KB (ns per byte)", "noop (ns per op)",
                                        "64 B DMA task + wait (ns per op)") if rc.get(k) and ctl.get(k)}
    ratio = statistics.median(slow.values()) if slow else None
    k0 = next((r.get("kinds") for r in runs if r.get("kinds") and r["variant"].startswith(V0)), None)
    sw, sw_ref = (next((r.get("design_switch_us") for r in runs if r.get("kinds") and r["variant"].startswith(V0)),
                       None), ref.get("design_switch_us"))
    if ratio is not None and k0:
        extra = (ctl["blockwrite 1 KB (ns per byte)"] - rc["blockwrite 1 KB (ns per byte)"]) * k0["control_bytes"] / 1e6
        extra_sw = (sw - sw_ref) * k0["configurations"] / 1e3 if sw is not None and sw_ref else 0.0
        gap = k0["whole_ms"] - ref["kinds"].get("whole program", k0["whole_ms"])
        out.append(_v("H11 the NPU's command / control path is slower here",
                      "yes" if ratio > 1.2 else "no",
                      f"control ops {ratio:.2f}x Strix Halo's ({ {k: round(v, 2) for k, v in slow.items()} }); a "
                      f"design switch {sw or 0:.0f} us vs {sw_ref or 0:.0f}; for the 1B chunk ({k0['control_bytes'] / 1e6:.1f} "
                      f"MB of control code, {k0['configurations']} configurations): ~{extra:.1f} + {extra_sw:.1f} ms "
                      f"of its {gap:+.1f} ms difference to Strix Halo"))
    if k0 and ref["kinds"]:
        diffs = sorted(((v["ms"] - ref["kinds"].get(k, v["ms"]), k) for k, v in k0["kinds"].items()), reverse=True)
        out.append(_v("where the 1B chunk's time goes (vs Strix Halo, its kinds alone)", "per kind",
                      "; ".join(f"{k} {d:+.1f} ms" for d, k in diffs[:5])))
    return out
