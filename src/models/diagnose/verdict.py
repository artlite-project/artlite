"""Which candidate cause the measurements point to (H0-H13 of the diagnosis), each from several measurements and with
the numbers that decide it, and how many ms of the 1B chunk's difference to Strix Halo each measured cause explains.

A variant counts as faster or slower only beyond its noise: the largest of the samples' spread, the two rounds'
difference of the same variant and 3% (between loads on Strix Halo). Every threshold is stated in the evidence it
produces."""

from __future__ import annotations

import re
import statistics

from . import reference
from .telemetry import PROCHOT_WARN

V0, V1, V2, V3, V4, V5 = "V0", "V1", "V2", "V3", "V4", "V5"
PROMPTS = re.compile(r"V[0-4]: .*, round \d+$")  # (the telemetry tags of the unpaced runs' prompts, not their load, kinds)


def _pp_ms(data: dict[str, object]) -> dict[str, list[float]]:
    """Each variant's prompt times (ms) over both rounds."""
    out: dict[str, list[float]] = {}
    for r in data["model"]["runs"]:
        out.setdefault(r["variant"][:2], []).extend(r["pp1024_ms"])
    return out


NOISE_FLOOR = 0.03  # (Strix Halo: one variant's medians 1.7-3.6% apart between its loads)


def _noise(data: dict[str, object]) -> float:
    """The relative noise of a variant's median: the larger of the samples' spread (1.4826 median absolute deviations
    over the median: a stdev that one stray sample does not set) and the rounds' difference of one variant, at least
    NOISE_FLOOR."""
    worst = NOISE_FLOOR
    by: dict[str, list[float]] = {}
    for r in data["model"]["runs"]:
        s = r["pp1024_ms"]
        if len(s) > 1:
            med = statistics.median(s)
            worst = max(worst, 1.4826 * statistics.median(abs(x - med) for x in s) / med)
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


def prochot_shares(tel: dict[str, dict[str, object]]) -> dict[str, float]:
    """Each telemetry step's share of time under PROCHOT (its residency increment, ms, over the step's length)."""
    return {t: e["throttle"]["prochot"] / (e["seconds"] * 1e3) for t, e in tel.items()
            if (e.get("throttle") or {}).get("prochot") is not None and e.get("seconds")}


def _power_state(ident: dict[str, object]) -> str:
    """Plugged in or on battery and through what (a USB-C source's PD contract), the system battery's state (not a
    mouse's), Lenovo's conservation mode."""
    sup = ident.get("power_supplies", {})
    online = []
    for name, v in sup.items():
        if v.get("type") in ("Mains", "USB") and v.get("online") == "1":
            vmax, imax = v.get("voltage_max", "0") or "0", v.get("current_max", "0") or "0"
            online.append(name + (f" {int(vmax) / 1e6:.0f} V x {int(imax) / 1e6:.2f} A" if int(vmax) and int(imax)
                                  else ""))
    mains = [v.get("online") for v in sup.values() if v.get("type") == "Mains"]
    out = (f"plugged in ({', '.join(online)})" if online else "on battery" if mains else "adapter unknown")
    if (bat := next((v for k, v in sup.items() if v.get("type") == "Battery" and k.startswith("BAT")), None)):
        out += f", battery {bat.get('status')} {bat.get('capacity')}%"
    if (cm := ident.get("power_path", {}).get("conservation_mode")):
        out += f", conservation mode {cm}"
    return out


def prochot(data: dict[str, object]) -> dict[str, str]:
    """H12: the platform holds the SoC's external PROCHOT input during the (unpaced) prompts (its EC, charger or a
    voltage regulator), which kinds set it off alone."""
    tel = data["telemetry"]
    share = prochot_shares(tel)
    prompts = [share[t] for t in tel if PROMPTS.match(t) and t in share]
    if not prompts:
        return _v("H12 the platform holds PROCHOT (its external throttle) while the NPU works", "?",
                  "no PROCHOT residency counter (gpu_metrics v3)")
    held = statistics.median(prompts)
    thermal = sum(sum((tel[t].get("throttle") or {}).get(k, 0) for k in ("thm_core", "thm_gfx", "thm_soc"))
                  for t in tel if PROMPTS.match(t))
    soc = max((tel[t]["soc_c"]["max"] for t in tel if PROMPTS.match(t) and isinstance(tel[t].get("soc_c"), dict)),
              default=None)
    alone = {t.split(": kinds: ", 1)[1]: v for t, v in share.items() if t.startswith(V0) and ": kinds: " in t}
    trips = sorted(k for k, v in alone.items() if v >= 0.2 and k != "whole program")
    calm = sorted(k for k, v in alone.items() if v < 0.05 and k != "whole program")
    return _v("H12 the platform holds PROCHOT (its external throttle) while the NPU works",
              "yes" if held >= PROCHOT_WARN else "no",
              f"PROCHOT held for {held:.0%} of the prompts (thermal limits {thermal} ms meanwhile, SoC up to "
              f"{soc if soc is None else round(soc)} °C: the platform's input, not the chip's temperature); kinds run "
              f"alone: held during {trips or 'none'}, not during {calm or 'none'}; power: "
              f"{_power_state(data['identity'])}")


def pacing(data: dict[str, object]) -> dict[str, str]:
    """H13: pacing the dense GEMMs (``prefill.pace``) lets them run unthrottled: the largest paced kind at each level
    (its time, PROCHOT's share), the calibration's levels, V5 (V0 at them) against V0."""
    cause = "H13 the dense GEMMs run faster paced (prefill.pace: the platform throttles them at full rate)"
    runs = data["model"]["runs"]
    sweep = next((r["pace_sweep"] for r in runs if r.get("pace_sweep")), None)
    cal = next((r["calibration"] for r in runs if r.get("calibration")), None)
    if sweep is None or cal is None:
        return _v(cause, "?", "no pace sweep")
    share = prochot_shares(data["telemetry"])
    lv = sweep["levels"]
    cells = []
    for level, v in lv.items():
        p = next((share[t] for t in share if t.startswith(V0) and t.endswith(f": pace {level}")), None)
        cells.append(f"{level} {v['ms']:.1f} ms" + (f" (PROCHOT {p:.0%})" if p is not None else ""))
    best = min(lv, key=lambda level: lv[level]["ms"])
    gain = lv["0"]["ms"] / lv[best]["ms"]
    pp = {k: statistics.median(v) for k, v in _pp_ms(data).items()}
    v5 = f"; pp1024 V5 {pp[V5]:.1f} ms vs V0 {pp[V0]:.1f}" if V5 in pp and V0 in pp else ""
    return _v(cause, "yes" if best != "0" and gain >= 1.1 else "no",
              f"{sweep['kind']} at pace levels {', '.join(cells)}: fastest at {best} ({gain:.2f}x level 0's speed); "
              f"calibrated: {cal['levels'] or 'every level 0'} ({cal['table']['seconds']:.1f} s){v5}")


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
    steps = [t for t in tel if t.startswith(V0) and PROMPTS.match(t)]
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
    for t in [t for t in tel if t.startswith((V0, V4)) and PROMPTS.match(t)]:
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
                  f"{later:.3f}; throttle residency increments during the prompts (ms) {thr or 'none'} (fast PPT and "
                  f"thm_core tick on Strix Halo with no effect; PROCHOT: H12)"))
    out.append(prochot(data))
    out.append(pacing(data))
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
        share = prochot_shares(tel)
        held = {k: share.get(t) for k in k0["kinds"] for t in tel if t.startswith(V0) and t.endswith(f": kinds: {k}")}
        out.append(_v("where the 1B chunk's time goes (vs Strix Halo, its kinds alone)", "per kind",
                      "; ".join(f"{k} {d:+.1f} ms" + (f" (PROCHOT {held[k]:.0%})" if held.get(k) else "")
                                for d, k in diffs[:5])))
    return out
