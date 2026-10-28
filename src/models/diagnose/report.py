"""The diagnosis as Markdown for an issue comment: the verdicts first, then each measurement next to Strix Halo's
(``reference.HALO``), the state it ran in, and the JSON file that holds every number."""

from __future__ import annotations

import statistics

from . import reference
from .verdict import _power_state, prochot_shares


def _f(x: object, digits: int = 1) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.0f}" if abs(x) >= 100 else f"{x:.{digits}f}"
    return "?" if x is None else str(x)


def _table(head: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    out += ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return out


def _pp(run: dict[str, object]) -> float:
    return 1024 / statistics.median(run["pp1024_ms"]) * 1e3


def markdown(data: dict[str, object], path: str) -> str:
    ref = reference.HALO
    ident, code, npu = data["identity"], data["code"], data.get("npu_info", {})
    lines = ["## artlite-probe --diagnose", ""]
    lines += ["### Verdicts", ""] + [f"- **{v['cause']}**: {v['verdict']} - {v['evidence']}" for v in data["verdicts"]]
    dmi = ident["dmi"]
    mem = data.get("root", {}).get("dmi_memory") or []
    memory = (f"{len(mem)} x {mem[0]['size_mb'] // 1024} GB {mem[0]['type']} {mem[0]['configured_mts']} MT/s "
              f"({mem[0]['data_width']}-bit each)" if mem else "? (root: DMI)")
    lines += ["", "### Machine", "",
              (f"- {dmi['sys_vendor']} {dmi['product_name']} {dmi['product_version']}; board {dmi['board_name']}; "
               f"BIOS {dmi['bios_vendor']} {dmi['bios_version']} ({dmi['bios_date']})"),
              f"- {ident['cpus']['model']}; memory {memory}; IP {ident['ip_versions']}",
              (f"- Linux {ident['kernel']}; THP {ident['thp']['enabled']}; NPU {npu.get('driver')} firmware "
               f"{npu.get('firmware')}, PCI rev {ident['npu'].get('revision')}, power mode {npu.get('power_mode')}, "
               f"interrupts on CPU {data.get('irq_cpu')}"),
              (f"- power: platform profile {sorted(set(ident['platform_profile'].values()) - {''})}; "
               f"{_power_state(ident)}; firmware attributes "
               f"{ident.get('power_path', {}).get('firmware_attributes') or 'none'}"),
              (f"- Artlite {code['commit']} ({code['describe']}); not upstream: {len(code['local_commits'])} "
               f"commits; changed files: {len(code['changed_files'])}; "
               f"{code['stale_install'] or 'native build current'}")]
    if (df := data.get("root", {}).get("df_dram_map", {})).get("maps"):
        m = df["maps"][0]
        lines.append(f"- data fabric {df['df_revision']}: interleave mode {m['mode']} from bit {m['first_bit']}, "
                     f"hash {m['hash'] or 'none'}")
    lines += ["", "### DMA (GB/s)", ""]
    dma = data["dma"]
    lines += _table(["", "2 MB pages", "4 KB pages", "Strix Halo 2 MB", "Strix Halo 4 KB"],
                    [[k, dma["2 MB"][k], dma["4 KB"][k], ref["dma"]["2 MB"].get(k), ref["dma"]["4 KB"].get(k)]
                     for k in dma["2 MB"]])
    lines += ["", "### Control path", ""]
    lines += _table(["", "this machine", "Strix Halo"],
                    [[k, v, ref["control"].get(k)] for k, v in data["control"].items()])
    lines += ["", "### Llama 3.2 1B, 1024-token prompt (t/s; each run's samples in the JSON)", ""]
    runs = data["model"]["runs"]
    lines += _table(["variant", "round", "pp1024", "lookups ms/prompt", "huge pages MB", "Strix Halo"],
                    [[r["variant"], r["round"], _pp(r), r["lookups_ms_per_prompt"], r["anon_huge_mb"],
                      ref["pp1024"].get(r["variant"])] for r in runs])
    k0 = next((r for r in runs if "kinds" in r and r["variant"].startswith("V0")), None)
    k1 = next((r for r in runs if "kinds" in r and r["variant"].startswith("V1")), None)
    if k0:
        share = prochot_shares(data["telemetry"])

        def held(k: str) -> str:
            v = next((share[t] for t in share if t.startswith("V0") and t.endswith(f": kinds: {k}")), None)
            return "?" if v is None else f"{v:.0%}"

        lines += ["", (f"Chunk of {k0['kinds']['T']} tokens, ms per operator kind (its items run alone), and the share "
                       "of each kind's run the platform held PROCHOT (2 MB pages):"), ""]
        rows = [["whole program", "", "", k0["kinds"]["whole_ms"], k1["kinds"]["whole_ms"] if k1 else None,
                 ref["kinds"].get("whole program"), held("whole program")]]
        for k, v in k0["kinds"]["kinds"].items():
            rows.append([k, str(v["items"]), str(v["configurations"]), v["ms"],
                         k1["kinds"]["kinds"].get(k, {}).get("ms") if k1 else None, ref["kinds"].get(k), held(k)])
        lines += _table(["kind", "items", "configurations", "2 MB pages", "4 KB pages", "Strix Halo 2 MB", "PROCHOT"],
                        rows)
        lines.append(f"\nA design switch inside a program: {_f(k0.get('design_switch_us'))} us "
                     f"(Strix Halo {_f(ref.get('design_switch_us'))})")
        if (sweep := k0.get("duty")):
            lines += ["", (f"{sweep['kind']}, one item at a time, each followed by sleep for a duty (ms an item; "
                           "PROCHOT's share of the run):"), ""]
            lines += _table(["duty", "ms an item", "items run", "Strix Halo ms", "PROCHOT"],
                            [[d, v["item_ms"], str(v["items_run"]), ref.get("duty_item_ms", {}).get(d),
                              next((f"{share[t]:.0%}" for t in share if t.startswith("V0") and t.endswith(f": duty {d}")),
                                   "?")] for d, v in sweep["duties"].items()])
    lines += ["", "### Telemetry per step (median; max)", ""]
    tel = data["telemetry"]
    fields = ("fclk_mhz", "uclk_mhz", "npu_clk_mhz", "npu_h_clk_mhz", "npu_column_util_max", "dram_read_mbs",
              "dram_write_mbs", "socket_mw", "npu_sensor_mw", "soc_c", "irq_cpu_mhz")
    rows = []
    for tag, e in tel.items():
        if tag in ("setup", "between"):
            continue
        cells = [tag] + [f"{_f(e[f]['median'])}; {_f(e[f]['max'])}" if isinstance(e.get(f), dict) else "-"
                         for f in fields]
        thr = e.get("throttle") or {}
        cells.append(", ".join(f"{k} +{v}" for k, v in thr.items() if v) or "none")
        cells.append(_f(e.get("other_cpu_load"), 2) if e.get("other_cpu_load") is not None else "?")
        rows.append(cells)
    lines += _table(["step", *fields, "throttling", "other CPU load"], rows)
    if (xv := data.get("xrt_validate")):
        lines += ["", f"xrt-smi validate: {xv}"]
    users = data.get("root", {}).get("npu_users")
    if users:
        lines += ["", f"Other processes on the NPU: {users}"]
    lines += ["", f"Every number: `{path}` (please attach it to the comment)."]
    return "\n".join(lines)
